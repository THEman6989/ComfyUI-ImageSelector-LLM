"""
BeatDrop Selector Embedding Node — embedding-based outfit selection.

Replaces Vision-LLM judging with a 3-stage architecture:
  1. Qwen3-VL-Embedding-8B: image+text embeddings → cosine-similarity scoring
  2. Re-Ranker (Cross-Encoder): Cross-Attention scoring (same as existing pipeline)
  3. VLM Fallback (optional): Vision-LLM judge only when stages 1+2 are uncertain

Built on the BeatDropSelectorNode structure — reuses history, reranker,
folder-loading, downsampling, and contact-sheet helpers from the parent module.

Place in: ComfyUI-ImageSelector-LLM/beatdrop_selector_embedding.py
"""

import json
import re
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image

# Reuse helpers from the parent module
try:
    from .openai_llm_node import (
        DEFAULT_SELECTOR_SYSTEM_PROMPT,
        _image_batch_to_pil_images,
        _build_contact_sheet,
        _encode_pil_to_data_url,
        _image_url_part,
        _extract_choice_content,
    )
except ImportError:
    from openai_llm_node import (
        DEFAULT_SELECTOR_SYSTEM_PROMPT,
        _image_batch_to_pil_images,
        _build_contact_sheet,
        _encode_pil_to_data_url,
        _image_url_part,
        _extract_choice_content,
    )

try:
    from .embedding_cache import EmbeddingCache
except ImportError:
    from embedding_cache import EmbeddingCache

# Import shared embedding cache from the loader module
try:
    from .qwen_vl_embedding_loader import _QWEN_EMBEDDING_CACHE, _get_qwen_embedding_model as _loader_get_model
except ImportError:
    from qwen_vl_embedding_loader import _QWEN_EMBEDDING_CACHE, _get_qwen_embedding_model as _loader_get_model

try:
    from .qwen_vl_reranker_loader import _RERANKER_CACHE
except ImportError:
    try:
        from qwen_vl_reranker_loader import _RERANKER_CACHE
    except ImportError:
        _RERANKER_CACHE = {}


def _make_blank_image(h=64, w=64):
    return torch.zeros(1, h, w, 3)


# ── Model cache (shared with qwen_vl_embedding_loader) ───
# _QWEN_EMBEDDING_CACHE is imported above from qwen_vl_embedding_loader
# _get_qwen_embedding_model is imported as _loader_get_model


def _get_qwen_embedding_model(model_path, device, dtype_str):
    """Redirect to shared loader function."""
    return _loader_get_model(model_path, device, dtype_str)


def _compute_embeddings_batch(model, processor, images_tensor, device, batch_size=8):
    """Compute Qwen3-VL-Embedding vectors for a batch of images.

    images_tensor: (B, H, W, 3) float tensor [0,1] in ComfyUI format.
    Returns: (B, dim) normalized tensor.
    """
    B = images_tensor.shape[0]
    arr = (images_tensor.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    pil_images = [Image.fromarray(arr[i]) for i in range(B)]

    # ── Qwen3VLEmbedder path ──
    if processor == "qwen3vl_embedder":
        inputs = [{"image": img} for img in pil_images]
        embs = model.process(inputs)
        if isinstance(embs, np.ndarray):
            embs = torch.from_numpy(embs)
        elif hasattr(embs, 'cpu'):
            embs = embs.cpu()
        embs = embs.float()
        embs = F.normalize(embs, p=2, dim=1)
        return embs

    # ── sentence-transformers path ──
    if processor == "sentence_transformers":
        model.to(device)
        embs = model.encode(pil_images, batch_size=batch_size, show_progress_bar=False, convert_to_tensor=True)
        if isinstance(embs, np.ndarray):
            embs = torch.from_numpy(embs)
        embs = embs.float()
        embs = F.normalize(embs, p=2, dim=1)
        return embs

    # ── transformers path ──
    all_embs = []
    for i in range(0, B, batch_size):
        batch_imgs = pil_images[i : i + batch_size]
        dummy_texts = ["Represent this image for visual search."] * len(batch_imgs)
        inputs = processor(text=dummy_texts, images=batch_imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
            emb = hidden.mean(dim=1)
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output
        elif isinstance(outputs, torch.Tensor):
            emb = outputs
            if emb.dim() == 3:
                emb = emb.mean(dim=1)
        else:
            emb = torch.zeros(len(batch_imgs), 2048)
        all_embs.append(emb.cpu())
    embs = torch.cat(all_embs, dim=0)
    embs = F.normalize(embs, p=2, dim=1)
    return embs


def _compute_text_embeddings(model, processor, texts, device):
    """Compute Qwen3-VL-Embedding vectors for text queries.

    texts: list of strings.
    Returns: (len(texts), dim) normalized tensor.
    """
    # ── Qwen3VLEmbedder path ──
    if processor == "qwen3vl_embedder":
        inputs = [{"text": t} for t in texts]
        embs = model.process(inputs)
        if isinstance(embs, np.ndarray):
            embs = torch.from_numpy(embs)
        elif hasattr(embs, 'cpu'):
            embs = embs.cpu()
        embs = embs.float()
        embs = F.normalize(embs, p=2, dim=1)
        return embs

    # ── sentence-transformers path ──
    if processor == "sentence_transformers":
        model.to(device)
        embs = model.encode(texts, batch_size=len(texts), show_progress_bar=False, convert_to_tensor=True)
        if isinstance(embs, np.ndarray):
            embs = torch.from_numpy(embs)
        embs = embs.float()
        embs = F.normalize(embs, p=2, dim=1)
        return embs

    # ── transformers path ──
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    if hasattr(outputs, "last_hidden_state"):
        hidden = outputs.last_hidden_state
        # Mean-pool over sequence, excluding padding
        if "attention_mask" in inputs:
            mask = inputs["attention_mask"].unsqueeze(-1).float().to(hidden.device)
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            emb = hidden.mean(dim=1)
    elif isinstance(outputs, torch.Tensor):
        emb = outputs
        if emb.dim() == 3:
            emb = emb.mean(dim=1)
    else:
        emb = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state.mean(dim=1)

    emb = emb.cpu()
    emb = F.normalize(emb, p=2, dim=1)
    return emb


def _as_text_list(value):
    """Normalize strings/lists into a clean list of embedding queries."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_as_text_list(item))
        return out
    text = str(value).strip()
    return [text] if text else []


def _color_signature(frame_tensor):
    """Small RGB signature for pair constraints like same/similar color.

    Uses the central body-ish crop to reduce background influence. This is not a
    detector; it is a cheap same-color cue layered on top of semantic scoring.
    """
    try:
        t = frame_tensor.detach().float().cpu()
        if t.dim() == 4:
            t = t[0]
        h, w = int(t.shape[0]), int(t.shape[1])
        y0, y1 = max(0, int(h * 0.12)), min(h, int(h * 0.90))
        x0, x1 = max(0, int(w * 0.18)), min(w, int(w * 0.82))
        crop = t[y0:y1, x0:x1, :3].reshape(-1, 3).clamp(0, 1)
        if crop.numel() == 0:
            crop = t.reshape(-1, 3).clamp(0, 1)
        mean = crop.mean(dim=0)
        std = crop.std(dim=0)
        return torch.cat([mean, std], dim=0)
    except Exception:
        return None


def _color_similarity(sig_a, sig_b):
    """0..1 similarity from compact RGB signatures."""
    if sig_a is None or sig_b is None:
        return 0.0
    try:
        diff = torch.abs(sig_a - sig_b).mean().item()
        return max(0.0, min(1.0, 1.0 - diff * 2.0))
    except Exception:
        return 0.0


def _make_pair_pil(left_pil, right_pil, label_left="Bild 1 / Phase 0", label_right="Bild 2 / Phase 1"):
    """Build one side-by-side image so the reranker can judge a PAIR."""
    from PIL import ImageDraw

    max_h = max(left_pil.height, right_pil.height)
    def _resize_keep(img):
        if img.height == max_h:
            return img.convert("RGB")
        scale = max_h / max(1, img.height)
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", 1)
        return img.convert("RGB").resize((max(1, int(img.width * scale)), max_h), resample)

    left = _resize_keep(left_pil)
    right = _resize_keep(right_pil)
    pad = 10
    label_h = 28
    out = Image.new("RGB", (left.width + right.width + pad, max_h + label_h), (20, 20, 20))
    out.paste(left, (0, label_h))
    out.paste(right, (left.width + pad, label_h))
    draw = ImageDraw.Draw(out)
    draw.text((6, 6), str(label_left), fill=(255, 255, 255))
    draw.text((left.width + pad + 6, 6), str(label_right), fill=(255, 255, 255))
    return out


def _phase_semantic_spec(value):
    """Parse a per-phase semantic query spec.

    Supported AI-Stack shapes:
      "0": "full body, covered outfit"                    # legacy/simple
      "0": {"query": "...", "must": [...], "avoid": [...]}  # preferred
      "0": {"include": [...], "exclude": [...], "filter": true}

    The node does NOT hardcode domain labels. The LLM/AI Stack must translate
    intent into positive and negative visual phrases. This makes the same path
    work for clothing coverage, full-body crops, colors, logos, backgrounds, etc.
    """
    if isinstance(value, dict):
        must = []
        avoid = []
        for key in ("must", "include", "positive", "required", "requirements"):
            must.extend(_as_text_list(value.get(key)))
        for key in ("avoid", "exclude", "negative", "forbidden", "reject"):
            avoid.extend(_as_text_list(value.get(key)))

        query_parts = _as_text_list(
            value.get("query") or value.get("description") or value.get("text")
        )
        query = "; ".join(query_parts or must)

        filter_cfg = value.get("semantic_filter", value.get("filter", value.get("gate", False)))
        filter_enabled = False
        filter_threshold = 0.0
        if isinstance(filter_cfg, dict):
            filter_enabled = bool(filter_cfg.get("enabled", True))
            try:
                filter_threshold = float(filter_cfg.get("threshold", 0.0))
            except (TypeError, ValueError):
                filter_threshold = 0.0
        else:
            filter_enabled = bool(filter_cfg)
            try:
                filter_threshold = float(value.get("threshold", 0.0))
            except (TypeError, ValueError):
                filter_threshold = 0.0

        return {
            "text": query,
            "must": must or query_parts,
            "avoid": avoid,
            "filter_enabled": filter_enabled,
            "filter_threshold": filter_threshold,
        }

    text = str(value or "").strip()
    return {
        "text": text,
        "must": [text] if text else [],
        "avoid": [],
        "filter_enabled": False,
        "filter_threshold": 0.0,
    }


def _safe_axis_name(text):
    """Stable key for a semantic relation axis."""
    text = str(text or "").strip().lower()
    text = re.sub(r"[^a-z0-9_äöüß-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:64] or "axis"


def _safe_path_component(value, fallback="default"):
    """Return one safe filename/path component; never preserves separators."""
    text = str(value or "").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    while ".." in text:
        text = text.replace("..", "_")
    text = text.strip(".-/\\")
    if not text or text in {".", ".."}:
        text = fallback
    return text[:96]


def _safe_child_dir(base, *parts):
    base = Path(base).expanduser().resolve()
    out = base
    for part in parts:
        out = out / _safe_path_component(part)
    out = out.resolve()
    if base != out and base not in out.parents:
        raise ValueError(f"Unsafe output path escaped base: {out}")
    return out


def _pair_relation_specs(pair_constraints):
    """Extract generic cross-phase relation axes from pair constraints.

    Domain-agnostic relation language. A preset can define an arbitrary visual
    axis such as coverage, scariness, luxury, brightness, formalness, etc.

    Supported shapes inside one pair constraint:
      "relations": [{
        "name": "coverage",
        "positive": ["covered outfit, lots of fabric"],
        "negative": ["revealing outfit, exposed skin"],
        "direction": "decrease",
        "min_score": 0.55,
        "gate": true
      }]

    Backwards compatible single-query relation:
      {"axis": "clothing coverage amount", "direction": "decrease"}
    """
    specs = []
    if isinstance(pair_constraints, str):
        try:
            pair_constraints = json.loads(pair_constraints)
        except (json.JSONDecodeError, TypeError):
            pair_constraints = []
    if isinstance(pair_constraints, dict):
        pair_constraints = [pair_constraints]
    if not isinstance(pair_constraints, list):
        return specs

    for pc in pair_constraints:
        if not isinstance(pc, dict):
            continue
        raw_relations = pc.get("relations", pc.get("relation_axes", pc.get("axes")))
        if raw_relations is None and any(k in pc for k in ("axis", "axis_query", "direction", "mode", "positive", "negative")):
            raw_relations = [pc]
        if isinstance(raw_relations, dict):
            raw_relations = [raw_relations]
        if not isinstance(raw_relations, list):
            continue
        for rel in raw_relations:
            if not isinstance(rel, dict):
                continue
            positives = []
            negatives = []
            for key in ("positive", "positives", "must", "include", "high", "more_like"):
                positives.extend(_as_text_list(rel.get(key)))
            for key in ("negative", "negatives", "avoid", "exclude", "low", "less_like"):
                negatives.extend(_as_text_list(rel.get(key)))
            query = str(rel.get("query") or rel.get("text") or rel.get("axis_query") or rel.get("axis") or "").strip()
            if not query and positives:
                query = "; ".join(positives)
            if not query and negatives:
                query = "opposite of: " + "; ".join(negatives)
            if not query and not positives and not negatives:
                continue

            name = _safe_axis_name(rel.get("name") or rel.get("key") or query or (positives[0] if positives else "axis"))
            mode = str(rel.get("direction") or rel.get("mode") or rel.get("relation") or "maximize_difference").strip().lower().replace("-", "_")
            try:
                weight = float(rel.get("weight", 4.0))
            except (TypeError, ValueError):
                weight = 4.0
            try:
                scale = float(rel.get("scale", rel.get("pair_scale", rel.get("amplify", 4.0))))
            except (TypeError, ValueError):
                scale = 4.0
            try:
                axis_scale = float(rel.get("axis_scale", rel.get("value_scale", 4.0)))
            except (TypeError, ValueError):
                axis_scale = 4.0
            try:
                min_score = float(rel.get("min_score", rel.get("threshold", 0.0)))
            except (TypeError, ValueError):
                min_score = 0.0
            gate = bool(rel.get("gate", rel.get("required", False)))
            try:
                gate_penalty = float(rel.get("gate_penalty", 999.0))
            except (TypeError, ValueError):
                gate_penalty = 999.0

            specs.append({
                "name": name,
                "query": query,
                "positive": positives,
                "negative": negatives,
                "bipolar": bool(positives and negatives),
                "mode": mode,
                "weight": weight,
                "scale": scale,
                "axis_scale": axis_scale,
                "min_score": min_score,
                "gate": gate,
                "gate_penalty": gate_penalty,
            })

    # de-dupe by name+query+positive+negative but keep first scoring config.
    seen = set()
    unique = []
    for spec in specs:
        key = (spec["name"], spec["query"], tuple(spec.get("positive", [])), tuple(spec.get("negative", [])))
        if key in seen:
            continue
        seen.add(key)
        unique.append(spec)
    return unique


def _axis_value_from_embedding(frame_emb, axis_entry, axis_scale=4.0):
    """Return (0..1 axis_value, detail_dict) for one frame and semantic axis."""
    detail = {"mode": "single_query"}
    if not axis_entry:
        return None, detail
    pos_fit = None
    neg_fit = None
    raw_fit = None
    if "positive_embeddings" in axis_entry:
        pe = axis_entry["positive_embeddings"]
        ps = F.cosine_similarity(frame_emb.expand(pe.shape[0], -1), pe)
        pos_fit = float(((ps + 1.0) / 2.0).mean().item())
    if "negative_embeddings" in axis_entry:
        ne = axis_entry["negative_embeddings"]
        ns = F.cosine_similarity(frame_emb.expand(ne.shape[0], -1), ne)
        neg_fit = float(((ns + 1.0) / 2.0).mean().item())
    if "query_embedding" in axis_entry:
        qs = float(F.cosine_similarity(frame_emb, axis_entry["query_embedding"]).item())
        raw_fit = (qs + 1.0) / 2.0

    if pos_fit is not None and neg_fit is not None:
        margin = pos_fit - neg_fit
        value = max(0.0, min(1.0, 0.5 + float(axis_scale) * margin))
        detail.update({
            "mode": "positive_minus_negative",
            "positive_fit": round(pos_fit, 4),
            "negative_fit": round(neg_fit, 4),
            "margin": round(margin, 4),
        })
        if raw_fit is not None:
            value = 0.85 * value + 0.15 * raw_fit
            detail["raw_query_fit"] = round(raw_fit, 4)
        return value, detail
    if pos_fit is not None:
        detail.update({"mode": "positive_only", "positive_fit": round(pos_fit, 4)})
        return pos_fit, detail
    if neg_fit is not None:
        value = 1.0 - neg_fit
        detail.update({"mode": "negative_only", "negative_fit": round(neg_fit, 4)})
        return value, detail
    if raw_fit is not None:
        detail.update({"mode": "single_query", "raw_query_fit": round(raw_fit, 4)})
        return raw_fit, detail
    return None, detail


def _relation_pair_score(source_value, target_value, mode, scale=4.0):
    """0..1 reward for a source→target relation on a semantic axis."""
    try:
        sv = float(source_value)
        tv = float(target_value)
        scale = float(scale)
    except (TypeError, ValueError):
        return None
    delta = tv - sv
    mode = str(mode or "maximize_difference").lower().replace("-", "_")
    if mode in ("increase", "more", "higher", "target_more", "target_gt_source"):
        return max(0.0, min(1.0, 0.5 + scale * delta))
    if mode in ("decrease", "less", "lower", "target_less", "target_lt_source"):
        return max(0.0, min(1.0, 0.5 - scale * delta))
    if mode in ("same", "similar", "minimize_difference", "min_diff", "keep"):
        return max(0.0, min(1.0, 1.0 - scale * abs(delta)))
    # default: prefer a large visible semantic change on this axis
    return max(0.0, min(1.0, scale * abs(delta)))


# ── BeatDropSelectorEmbeddingNode ──────────────────────────────────────

class BeatDropSelectorEmbeddingNode:
    """3-stage embedding-based outfit selector for beatdrop pipelines.

    Stage 1: Qwen3-VL-Embedding-8B → cosine similarity scoring
    Stage 2: Re-Ranker Cross-Encoder → Cross-Attention scoring
    Stage 3: VLM Fallback (optional) → Vision-LLM judge

    Much faster than pure LLM judging — typically <200ms for stages 1+2
    vs 2-15 seconds for a Vision-LLM contact-sheet call.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_frames_per_window": ("INT", {"default": 4, "min": 2, "max": 20,
                    "tooltip": "Max frames per drop window. For 2+ outfits."}),
                "num_outfits_mode": (["auto_from_beats", "manual"], {"default": "auto_from_beats",
                    "tooltip": "auto_from_beats: derive from beats_used windows."}),
                "num_outfits": ("INT", {"default": 2, "min": 2, "max": 10, "step": 1,
                    "tooltip": "Manual: minimum number of VISIBLY DIFFERENT outfits."}),
            },
            "optional": {
                # ── Pre-loaded embedding model (from QwenVLEmbeddingLoader) ──
                "embedding_model": ("QWEN_VL_EMBEDDING", {"tooltip": "Connect QwenVLEmbeddingLoader here. Overrides embedding_model_path if connected."}),
                "reference_frames": ("IMAGE", {"tooltip": "Video frames from FrameSequenceGenerator"}),
                "context_frames": ("IMAGE", {"tooltip": "Video frames OUTSIDE drop windows (low fps context)"}),
                "beats_used": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "beats_used JSON from FrameSequenceGenerator"}),

                # ── AI Stack config (forceInput = wire-only, not a text field) ──
                "ai_stack_config_json": ("STRING", {"default": "{}", "multiline": True, "forceInput": True,
                    "tooltip": "🔧 Connect BeatDropConfigPipe here. Overrides ALL static params."}),

                # ── Stage 1: Embedding Scoring (model comes from QwenVLEmbeddingLoader) ──
                "embedding_confidence_threshold": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Below this → fall through to Stage 2 (Re-Ranker).",
                }),
                "embedding_scene_fit_weight": ("FLOAT", {
                    "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Weight for scene-fit in composite embedding score.",
                }),
                "embedding_change_strength_weight": ("FLOAT", {
                    "default": 0.50, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Weight for change-strength in composite score. Higher = prioritize difference.",
                }),
                "embedding_diversity_weight": ("FLOAT", {
                    "default": 0.20, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Weight for diversity (spread across embedding space).",
                }),
                # Text queries for semantic matching (DAS KANN NUR QWEN!)
                "text_query_scene_fit": ("STRING", {
                    "default": "", "multiline": True,
                    "placeholder": "'beach party vibe, summer, casual, bright' — scene-fit query for embedding space",
                    "tooltip": "Text description of the desired scene aesthetic. Embedded and compared with frames.",
                }),
                "text_query_change_target": ("STRING", {
                    "default": "", "multiline": True,
                    "placeholder": "'dramatic silhouette change, contrasting style, different era' — what makes a good swap",
                    "tooltip": "Text description of what a good outfit change looks like. Compared with outfit pairs.",
                }),

                # ── Stage 2: Re-Ranker (Cross-Encoder) ──
                "reranker_model": ("QWEN_VL_RERANKER", {"tooltip": "Connect QwenVLRerankerLoader here. Overrides API endpoint + model_path."}),
                "reranker_mode": (["api", "transformers"], {
                    "default": "api",
                    "tooltip": "api: SGLang/vLLM endpoint. transformers: direct Qwen3VLReranker (load/unload on GPU).",
                }),
                "reranker_endpoint": ("STRING", {
                    "default": "", "multiline": False,
                    "placeholder": "http://127.0.0.1:8012 — vLLM/SGLang reranker (api mode only)",
                    "tooltip": "Re-Ranker endpoint. Only used when reranker_mode=api.",
                }),
                "reranker_model_path": ("STRING", {
                    "default": "Qwen/Qwen3-VL-Reranker-8B",
                    "multiline": False,
                    "placeholder": "HF model ID or local path (transformers mode)",
                    "tooltip": "Reranker model for transformers mode.",
                }),
                "reranker_device": (["auto", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "cpu"], {
                    "default": "auto",
                    "tooltip": "CUDA device for reranker (transformers mode). Can differ from embedding device.",
                }),
                "reranker_dtype": (["fp16", "bf16", "int8", "fp32"], {
                    "default": "fp16",
                    "tooltip": "Dtype for direct reranker loading.",
                }),
                "reranker_top_k": ("INT", {
                    "default": 12, "min": 0, "max": 200, "step": 1,
                    "tooltip": "Pre-filter: only top-K reranker candidates enter scoring. 0 = all.",
                }),
                "reranker_query": ("STRING", {
                    "default": "", "multiline": True,
                    "placeholder": "Custom reranker query. Empty = auto.",
                    "tooltip": "What the reranker should look for in candidates.",
                }),
                "reranker_blend_weight": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much to blend reranker scores (0=ignore, 1=full reranker).",
                }),
                "reranker_confidence_threshold": ("FLOAT", {
                    "default": 0.70, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Below this → fall through to Stage 3 (VLM Fallback).",
                }),

                # ── Stage 3: VLM Fallback (optional) ──
                "use_vlm_fallback": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable VLM judge as Stage 3 when embedding + reranker are both uncertain.",
                }),
                "vlm_endpoint": ("STRING", {
                    "default": "http://127.0.0.1:8080/v1/chat/completions",
                    "multiline": False,
                }),
                "vlm_api_token": ("STRING", {"default": "", "multiline": False}),
                "vlm_model": ("STRING", {"default": "local-model", "multiline": False}),
                "vlm_system_prompt": ("STRING", {
                    "default": DEFAULT_SELECTOR_SYSTEM_PROMPT,
                    "multiline": True,
                }),
                "vlm_max_tokens": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "vlm_temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "vlm_timeout": ("INT", {"default": 120, "min": 5, "max": 600, "step": 1}),

                # ── Shared: Frame Management ──
                "max_total_frames": ("INT", {"default": 100, "min": 5, "max": 500, "step": 5,
                    "tooltip": "Max total frames. Exceeding triggers downsampling."}),
                "job_fps": ("FLOAT", {"default": 5.0, "min": 0.5, "max": 60.0, "step": 0.5}),
                "context_fps": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 30.0, "step": 0.1}),
                "downsample_mode": (["global", "per_job"], {"default": "global"}),
                "image_resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64,
                    "tooltip": "Max pixel dimension for embedding model input."}),

                # ── Shared: History ──
                "history_file": ("STRING", {"default": "", "multiline": False,
                    "placeholder": "/path/to/selection_history.json"}),
                "history_penalty": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 50.0, "step": 0.5}),
                "history_decay_rate": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 2.0, "step": 0.05}),
                "history_max_entries": ("INT", {"default": 200, "min": 10, "max": 10000, "step": 10}),

                # ── Shared: Extra ──
                "extra_instructions": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "Zusaetzliche Anweisungen (wird als Text-Query embedded!)"}),
                "extra_penalty_json": ("STRING", {"default": "{}", "multiline": True,
                    "placeholder": "JSON dict {frame_id: penalty_value} from Judge"}),
                "penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "conversation_id": ("STRING", {"default": "", "multiline": False}),
                "grid_columns": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
                "add_id_labels": ("BOOLEAN", {"default": True}),

                # ── Folder-based candidate loading ──
                "candidate_folders": ("STRING", {"default": "", "multiline": False,
                    "placeholder": "/path/to/outfits/ — root folder with subdirectories",
                    "tooltip": "Root folder with outfit subdirectories."}),
                "max_candidate_images": ("INT", {"default": 30, "min": 5, "max": 500, "step": 5}),
                "use_random_sample": ("BOOLEAN", {"default": True}),
                # Per-phase folder assignment (Approach B: embedding-based auto)
                "text_query_per_phase_json": ("STRING", {"default": "{}", "multiline": True,
                    "placeholder": '{"0": "wearing jacket, formal, covered", "1": "no jacket, casual, bare shoulders"}',
                    "tooltip": "Per-phase text queries for embedding-based folder matching. Empty = auto from folder names + scene."}),
                "folder_assignments_json": ("STRING", {"default": "", "multiline": True,
                    "placeholder": '{"0": "jackets", "1": "no_jacket"}',
                    "tooltip": "Manual folder-to-phase mapping. Overrides auto-assignment. Empty = use embedding auto-assign."}),
                # Embedding cache (SQLite-backed, persists across runs)
                "use_embedding_cache": ("BOOLEAN", {"default": True,
                    "tooltip": "Cache embeddings in SQLite DB. First run: embed all. Subsequent runs: instant load."}),
                "cache_db_path": ("STRING", {"default": "", "multiline": False,
                    "placeholder": "Auto: {candidate_folders}/.embedding_cache.db",
                    "tooltip": "SQLite DB path for embedding cache. Empty = auto-detect from candidate_folders."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("selected_indices", "count", "metadata", "contact_sheet", "raw_response", "ai_stack_context")
    FUNCTION = "select"
    CATEGORY = "Amin/Beatdrop"

    # ═══════════════════════════════════════════════════════════════════
    # History helpers (same pattern as BeatDropSelectorNode)
    # ═══════════════════════════════════════════════════════════════════

    def _load_history(self, history_file):
        path = str(history_file or "").strip()
        if not path:
            return {"selections": [], "total_selections": 0}
        hp = Path(path).expanduser()
        if not hp.exists():
            return {"selections": [], "total_selections": 0}
        try:
            with open(hp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return {"selections": [], "total_selections": 0}
            return data
        except (json.JSONDecodeError, OSError):
            return {"selections": [], "total_selections": 0}

    def _save_history(self, history_file, history, max_entries):
        path = str(history_file or "").strip()
        if not path:
            return
        hp = Path(path).expanduser()
        hp.parent.mkdir(parents=True, exist_ok=True)
        selections = history.get("selections", [])
        max_entries = max(10, int(max_entries))
        if len(selections) > max_entries:
            selections = selections[-max_entries:]
        try:
            with open(hp, "w", encoding="utf-8") as f:
                json.dump({"selections": selections, "total_selections": len(selections)},
                          f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def _history_key(self, frame_idx):
        return f"beatdrop_emb_frame_{int(frame_idx)}"

    def _history_penalty_for(self, frame_idx, history, base_penalty, decay_rate=0.3):
        if base_penalty <= 0:
            return 0.0
        selections = history.get("selections", [])
        if not selections:
            return 0.0
        key = self._history_key(frame_idx)
        positions = []
        for pos, entry in enumerate(reversed(selections)):
            if entry.get("key") == key:
                positions.append(pos)
        if not positions:
            return 0.0
        most_recent = min(positions)
        count = min(len(positions), 5)
        decay = 1.0 / (1.0 + most_recent * max(0.05, float(decay_rate)))
        freq = 1.0 + min(count - 1, 4) * 0.15
        return min(base_penalty * decay * freq, base_penalty * 1.5)

    # ═══════════════════════════════════════════════════════════════════
    # Re-Ranker (same pattern as BeatDropSelectorNode)
    # ═══════════════════════════════════════════════════════════════════

    def _reranker_urls(self, endpoint):
        if not endpoint:
            return []
        if "://" not in endpoint:
            endpoint = "http://" + endpoint
        endpoint = endpoint.rstrip("/")
        last = endpoint.rsplit("/", 1)[-1]
        if last in {"rerank", "reranking"}:
            return [endpoint]
        return [endpoint + p for p in ("/v1/rerank", "/v2/rerank", "/rerank", "/reranking")]

    def _reranker_payload(self, query, documents, model="", top_n=0):
        p = {"query": query, "documents": documents, "return_documents": False}
        if model:
            p["model"] = model
        if int(top_n) > 0:
            p["top_n"] = int(top_n)
        return p

    def _parse_reranker_scores(self, result, doc_count):
        if isinstance(result, list):
            raw = result
        elif isinstance(result, dict):
            raw = result.get("results") or result.get("data") or result.get("scores")
        else:
            raw = None
        scores = []
        if isinstance(raw, list):
            for i, item in enumerate(raw):
                if isinstance(item, dict):
                    idx = int(item.get("index", i))
                    sc = float(item.get("relevance_score", item.get("score", 0)))
                else:
                    idx, sc = i, float(item)
                if 0 <= idx < doc_count:
                    scores.append((idx, sc))
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def _run_reranker(self, endpoint, headers, model, query, documents, top_n, timeout):
        """Call reranker API with auto-detection. Returns sorted (index, score) list or None."""
        import requests

        resolved_model = str(model or "").strip()
        urls = self._reranker_urls(endpoint)
        if not urls:
            return None

        # Try model auto-detection if not set
        if not resolved_model:
            for url in urls:
                try:
                    base = url.rsplit("/", 1)[0] if "/rerank" in url else url
                    r = requests.get(base + "/v1/models", headers=headers,
                                     timeout=min(max(int(timeout), 1), 10))
                    if r.ok:
                        models = r.json().get("data", [])
                        if models:
                            resolved_model = str(models[0].get("id", "")).strip()
                            break
                except Exception:
                    continue

        for url in urls:
            try:
                payload = self._reranker_payload(query, documents, resolved_model, top_n)
                r = requests.post(url, headers=headers, json=payload,
                                  timeout=min(max(int(timeout), 1), 30))
                r.raise_for_status()
                scores = self._parse_reranker_scores(r.json(), len(documents))
                if scores:
                    return scores
            except Exception:
                continue
        return None

    # ═══════════════════════════════════════════════════════════════════
    # Folder-based loading (same as BeatDropSelectorNode)
    # ═══════════════════════════════════════════════════════════════════

    def _scan_folders(self, root_path):
        root = Path(root_path).expanduser()
        if not root.is_dir():
            return []
        folders = []
        for entry in sorted(root.iterdir()):
            if entry.is_dir():
                imgs = sorted([
                    str(p) for p in entry.iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
                ])
                if imgs:
                    folders.append((entry.name, imgs))
        if not folders:
            # Outside the Configurator Amin often tests with one flat image
            # directory. Treat loose root images as one reusable pool instead
            # of failing with "no subdirectories with images found".
            loose_imgs = sorted([
                str(p) for p in root.iterdir()
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            ])
            if loose_imgs:
                folders.append(("_root", loose_imgs))
        return folders

    def _load_filtered_candidates(self, folders, selected_names, max_images,
                                   history, history_penalty, decay_rate, use_random,
                                   min_image_height=0, min_aspect_ratio=0.0):
        import random

        all_paths = []
        for name, paths in folders:
            if name in selected_names:
                all_paths.extend([(name, p) for p in paths])

        if not all_paths:
            return None, [], [], []

        # ── Pre-filter: reject non-full-body images ──
        if min_image_height > 0 or min_aspect_ratio > 0:
            filtered = []
            for folder_name, path in all_paths:
                try:
                    with Image.open(path) as img:
                        w, h = img.size
                    if min_image_height > 0 and h < min_image_height:
                        continue  # too short — skip
                    if min_aspect_ratio > 0 and (h / max(w, 1)) < min_aspect_ratio:
                        continue  # wrong aspect — skip
                    filtered.append((folder_name, path))
                except Exception:
                    continue
            if filtered:
                all_paths = filtered
                print(f"[EmbeddingMatcher] Pre-filter: {len(filtered)}/{len(all_paths) if filtered else 0} images pass "
                      f"(min_h={min_image_height}, min_ar={min_aspect_ratio})")

        scored_paths = []
        for folder_name, path in all_paths:
            hist_key = f"folder_{Path(path).stem}"
            hist_pen = self._history_penalty_for(
                hash(hist_key) % 100000, history, float(history_penalty), float(decay_rate),
            )
            scored_paths.append((folder_name, path, hist_pen))

        scored_paths.sort(key=lambda x: x[2])
        max_img = min(int(max_images), len(scored_paths))

        if use_random:
            pool_size = min(len(scored_paths), max(int(max_img * 2), max_img))
            pool = scored_paths[:pool_size]
            selected = random.sample(pool, min(max_img, len(pool)))
        else:
            selected = scored_paths[:max_img]

        loaded = []
        loaded_meta = []
        folder_map = {}
        for idx, (fname, path, penalty) in enumerate(selected):
            try:
                pil_img = Image.open(path).convert("RGB")
                arr = np.array(pil_img, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(arr)
                folder_map[len(loaded)] = fname
                loaded.append(tensor)
                loaded_meta.append((fname, path))
            except Exception:
                continue

        if not loaded:
            return None, [], [], []

        # Resize all to match
        h, w = loaded[0].shape[:2]
        uniform = []
        for t in loaded:
            if t.shape[0] != h or t.shape[1] != w:
                t = F.interpolate(
                    t.permute(2, 0, 1).unsqueeze(0),
                    size=(h, w), mode="bilinear", align_corners=False,
                ).squeeze(0).permute(1, 2, 0)
            uniform.append(t)

        images_tensor = torch.stack(uniform, dim=0)
        folder_list = [folder_map.get(i, "unknown") for i in range(len(loaded_meta))]
        image_stems = [Path(path).stem for _, path in loaded_meta]
        image_paths = [str(Path(path)) for _, path in loaded_meta]
        return images_tensor, folder_list, image_stems, image_paths

    def _assign_folders_via_embedding(self, folders, reference_frames, windows,
                                        text_query_per_phase, model_path, device, dtype_str):
        """Embedding-based folder-to-phase assignment.

        1. Embed sample images from each folder → folder_embedding
        2. Embed scene frames from each window → window_embedding
        3. If text queries per phase: embed those too, blend with visual
        4. Cosine similarity matrix → best folder per phase (no duplicates)
        """
        if not folders or not windows or not model_path:
            return None

        try:
            model, processor, dim = _get_qwen_embedding_model(model_path, device, dtype_str)
        except Exception as e:
            print(f"[EmbeddingMatcher] Folder assignment: model load failed ({e}) — falling back to order-based")
            return None

        # ── 1. Embed folder samples (3 images per folder) ──
        folder_names = []
        folder_embs = []
        for folder_name, img_paths in folders:
            # Load up to 3 sample images per folder
            import random as _random
            samples = _random.sample(img_paths, min(3, len(img_paths)))
            pil_images = []
            for sp in samples:
                try:
                    pil_img = Image.open(sp).convert("RGB")
                    pil_images.append(pil_img)
                except Exception:
                    continue
            if not pil_images:
                continue

            # Stack manually into ComfyUI format: (B, H, W, 3)
            tensors = []
            for pimg in pil_images:
                arr = np.array(pimg, dtype=np.float32) / 255.0
                tensors.append(torch.from_numpy(arr))
            # Pad to same size
            max_h = max(t.shape[0] for t in tensors)
            max_w = max(t.shape[1] for t in tensors)
            uniform = []
            for t in tensors:
                if t.shape[0] != max_h or t.shape[1] != max_w:
                    t = F.interpolate(
                        t.permute(2, 0, 1).unsqueeze(0),
                        size=(max_h, max_w), mode="bilinear", align_corners=False,
                    ).squeeze(0).permute(1, 2, 0)
                uniform.append(t)
            batch = torch.stack(uniform, dim=0)  # (N, H, W, 3)

            emb = _compute_embeddings_batch(model, processor, batch, device, batch_size=4)
            folder_emb = emb.mean(dim=0, keepdim=True)  # (1, dim)
            folder_emb = F.normalize(folder_emb, p=2, dim=1)

            folder_names.append(folder_name)
            folder_embs.append(folder_emb)

        if not folder_embs:
            return None

        folder_embs = torch.cat(folder_embs, dim=0)  # (F, dim)

        # ── 2. Embed window scene frames ──
        window_embs = []
        for win in windows:
            ws = win.get("batch_start", 0)
            we = min(win.get("batch_end", reference_frames.shape[0]), reference_frames.shape[0])
            if we <= ws:
                window_embs.append(None)
                continue
            win_frames = reference_frames[ws:we]
            # Downsample to max 5 frames per window
            if win_frames.shape[0] > 5:
                keep = torch.linspace(0, win_frames.shape[0] - 1, 5).long()
                win_frames = win_frames[keep]

            emb = _compute_embeddings_batch(model, processor, win_frames, device, batch_size=4)
            win_emb = emb.mean(dim=0, keepdim=True)
            win_emb = F.normalize(win_emb, p=2, dim=1)
            window_embs.append(win_emb)

        # ── 3. Text query embeddings per phase ──
        text_per_phase = {}
        if text_query_per_phase:
            try:
                tqp = json.loads(text_query_per_phase) if isinstance(text_query_per_phase, str) else text_query_per_phase
                for phase_str, query_text in tqp.items():
                    if query_text and str(query_text).strip():
                        text_emb = _compute_text_embeddings(
                            model, processor, [str(query_text).strip()], device,
                        )
                        text_per_phase[int(phase_str)] = text_emb  # (1, dim)
            except (json.JSONDecodeError, ValueError):
                pass

        # ── 4. Cosine similarity matrix + assignment ──
        assignments = []
        used_folders = set()
        num_windows = len(windows)

        for phase in range(num_windows):
            win_emb = window_embs[phase] if phase < len(window_embs) else None

            best_folder = None
            best_score = -1.0
            best_reason = ""

            for fi, fname in enumerate(folder_names):
                if fname in used_folders:
                    continue  # no duplicates — each folder assigned to at most one phase

                folder_emb = folder_embs[fi:fi+1]  # (1, dim)

                # Visual similarity: folder ↔ window scene
                visual_score = 0.5  # neutral default
                if win_emb is not None:
                    vs = float(F.cosine_similarity(folder_emb, win_emb).item())
                    visual_score = (vs + 1.0) / 2.0  # [-1,1] → [0,1]

                # Text-query alignment
                text_score = 0.5
                if phase in text_per_phase:
                    ts = float(F.cosine_similarity(folder_emb, text_per_phase[phase]).item())
                    text_score = (ts + 1.0) / 2.0

                # Composite: blend visual scene-match + semantic text-match
                score = 0.4 * visual_score + 0.6 * text_score

                if score > best_score:
                    best_score = score
                    best_folder = fname
                    best_reason = (
                        f"visual={visual_score:.3f}, text={text_score:.3f} "
                        f"(query: {text_per_phase.get(phase, 'none')})"
                    )

            if best_folder:
                used_folders.add(best_folder)
                assignments.append({
                    "phase": phase,
                    "folder": best_folder,
                    "reason": best_reason,
                    "score": round(best_score, 4),
                })

        # Any remaining unassigned folders → append to extra phases
        for fname in folder_names:
            if fname not in used_folders:
                assignments.append({
                    "phase": len(assignments),
                    "folder": fname,
                    "reason": "extra (no window match)",
                })
                used_folders.add(fname)

        if not assignments:
            return None

        print(f"[EmbeddingMatcher] Folder assignment: {[(a['folder'], a.get('score', '?')) for a in assignments]}")
        return assignments

    def _load_from_folders(self, root_path, history, history_penalty, decay_rate,
                            max_images, use_random, num_windows=2,
                            folder_assignments_json="",
                            text_query_per_phase_json="{}",
                            embedding_model_path=None, embedding_device="auto",
                            embedding_dtype="fp16",
                            reference_frames=None, windows=None,
                            use_embedding_cache=False,
                            cache_db_path="",
                            min_image_height=0, min_aspect_ratio=0.0):
        """Per-phase loading with embedding-based folder assignment + persistent cache.

        Priority:
        1. folder_assignments_json (manual) → direct mapping
        2. embedding-based auto-assign (Approach B) → cosine similarity
        3. Fallback: folders assigned in alphabetical order

        When use_embedding_cache=True: pre-computes embeddings for ALL outfit images,
        caches them in SQLite. Subsequent runs load cached embeddings instantly.
        """
        folders = self._scan_folders(root_path)
        if not folders:
            return None, {"error": "no subdirectories with images found"}

        # ── Determine assignments ──
        assignments = None

        # Priority 1: Manual JSON mapping
        if folder_assignments_json and folder_assignments_json.strip():
            try:
                manual = json.loads(folder_assignments_json)
                assignments = []
                for phase_str, folder_name in manual.items():
                    phase = int(phase_str)
                    if any(fname == folder_name for fname, _ in folders):
                        assignments.append({
                            "phase": phase, "folder": folder_name,
                            "reason": "manual",
                        })
                print(f"[EmbeddingMatcher] Manual folder assignment: {[(a['folder'], a['phase']) for a in assignments]}")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[EmbeddingMatcher] Invalid folder_assignments_json: {e}")

        # Priority 2: Embedding-based auto-assign
        if not assignments and embedding_model_path and isinstance(reference_frames, torch.Tensor):
            try:
                assignments = self._assign_folders_via_embedding(
                    folders, reference_frames, windows or [],
                    text_query_per_phase_json,
                    embedding_model_path, embedding_device, embedding_dtype,
                )
            except Exception as e:
                print(f"[EmbeddingMatcher] Embedding folder assignment failed: {e}")

        # Priority 3: Fallback — alphabetical order
        if not assignments:
            if len(folders) == 1 and int(num_windows) > 1:
                # One folder/pool must be usable for all phases; otherwise a
                # flat folder can only populate phase 0 and the later drop has
                # no candidates. Duplicate the same pool per phase; per-phase
                # text scoring then chooses different images from the same pool.
                assignments = [
                    {"phase": i, "folder": folders[0][0], "reason": "fallback (single folder reused for all phases)"}
                    for i in range(int(num_windows))
                ]
            else:
                assignments = [{"phase": i, "folder": name, "reason": "fallback (alpha order)"}
                              for i, (name, _) in enumerate(folders)]

        # ── Pre-embed ALL images from assigned folders via cache ──
        cached_embeddings = {}  # {path: tensor}
        cache_stats = None
        cache_hits = 0
        cache_misses = 0
        if use_embedding_cache and embedding_model_path:
            try:
                # Determine cache DB path
                if cache_db_path and cache_db_path.strip():
                    db_path = cache_db_path.strip()
                else:
                    db_path = str(Path(root_path).expanduser() / ".embedding_cache.db")

                model_hash = EmbeddingCache._model_hash(
                    embedding_model_path, embedding_dtype, embedding_device,
                )
                cache = EmbeddingCache(db_path)

                # Collect ALL image paths from assigned folders
                all_assigned_paths = []
                path_to_folder = {}
                assigned_folder_names = {a["folder"] for a in assignments}
                for fname, paths in folders:
                    if fname in assigned_folder_names:
                        all_assigned_paths.extend(paths)
                        for p in paths:
                            path_to_folder[p] = fname

                if all_assigned_paths:
                    # Check cache
                    cached, needs_embed = cache.load_or_prepare(
                        all_assigned_paths, model_hash, path_to_folder,
                    )
                    cache_hits = len(cached)
                    cache_misses = len(needs_embed)
                    cached_embeddings = cached  # {path: tensor}

                    if needs_embed:
                        print(f"[EmbeddingMatcher] Cache: {len(cached)} cached, "
                              f"{len(needs_embed)} need embedding")
                        # Batch-embed new images
                        model, processor, dim = _get_qwen_embedding_model(
                            embedding_model_path, embedding_device, embedding_dtype,
                        )
                        pil_batch = []
                        path_batch = []
                        new_embs = {}
                        for path in needs_embed:
                            try:
                                pil_img = Image.open(path).convert("RGB")
                                # Resize to max 512 for embedding speed
                                w, h = pil_img.size
                                if max(w, h) > 512:
                                    scale = 512 / max(w, h)
                                    pil_img = pil_img.resize(
                                        (int(w * scale), int(h * scale)), Image.LANCZOS,
                                    )
                                pil_batch.append(pil_img)
                                path_batch.append(path)
                            except Exception:
                                continue

                        if pil_batch:
                            # Convert to ComfyUI tensor format
                            tensors = []
                            for pimg in pil_batch:
                                arr = np.array(pimg, dtype=np.float32) / 255.0
                                tensors.append(torch.from_numpy(arr))
                            # Pad to uniform size
                            max_h = max(t.shape[0] for t in tensors)
                            max_w = max(t.shape[1] for t in tensors)
                            uniform = []
                            for t in tensors:
                                if t.shape[0] != max_h or t.shape[1] != max_w:
                                    t = F.interpolate(
                                        t.permute(2, 0, 1).unsqueeze(0),
                                        size=(max_h, max_w), mode="bilinear", align_corners=False,
                                    ).squeeze(0).permute(1, 2, 0)
                                uniform.append(t)
                            batch_tensor = torch.stack(uniform, dim=0)

                            embs = _compute_embeddings_batch(
                                model, processor, batch_tensor, embedding_device, batch_size=4,
                            )
                            for path, emb in zip(path_batch, embs):
                                new_embs[path] = emb
                                cached_embeddings[path] = emb

                            # Store to cache
                            cache.store(new_embs, model_hash, path_to_folder)
                            print(f"[EmbeddingMatcher] Cached {len(new_embs)} new embeddings")

                    cache_stats = cache.stats()
                    print(f"[EmbeddingMatcher] Cache DB: {cache_stats['db_size_mb']} MB, "
                          f"{cache_stats['total_embeddings']} total embeddings")
            except Exception as e:
                print(f"[EmbeddingMatcher] Cache preload failed ({e}) — falling through to live embedding")
                import traceback
                traceback.print_exc()
                cached_embeddings = {}
                cache_stats = {"error": str(e)}

        max_per_phase = max(2, int(max_images) // max(1, len(assignments)))
        all_images = []
        folder_map_global = {}
        image_stems_global = {}
        image_paths_global = {}
        phase_info = []

        for assignment in assignments:
            phase = assignment.get("phase", 0)
            folder_name = assignment.get("folder", "")
            phase_paths = []
            for fname, paths in folders:
                if fname == folder_name:
                    phase_paths = [(fname, p) for p in paths]
                    break
            if not phase_paths:
                continue

            phase_images, phase_list, phase_stems, phase_image_paths = self._load_filtered_candidates(
                [(folder_name, [p for _, p in phase_paths])], [folder_name],
                max_per_phase, history, history_penalty, decay_rate, use_random,
                min_image_height=min_image_height, min_aspect_ratio=min_aspect_ratio,
            )
            if phase_images is not None:
                # Offset must be frame count, not number of phase tensors.
                # len(all_images) broke global metadata for phase 1+ (frame 10
                # was recorded as 1), which also prevented duplicate-source
                # penalties across reused flat folders.
                offset = sum(t.shape[0] for t in all_images)
                all_images.append(phase_images)
                for i, stem in enumerate(phase_stems or []):
                    folder_map_global[offset + i] = folder_name
                    image_stems_global[offset + i] = stem
                    if i < len(phase_image_paths or []):
                        image_paths_global[offset + i] = phase_image_paths[i]
                phase_info.append({
                    "phase": phase,
                    "folder": folder_name,
                    "reason": assignment.get("reason", ""),
                    "images_loaded": phase_images.shape[0],
                })

        if not all_images:
            return None, {"error": "no images loaded from any folder"}

        # Resize all phase images to uniform size before concatenating
        max_h = max(t.shape[1] for t in all_images)
        max_w = max(t.shape[2] for t in all_images)
        uniform_phases = []
        for phase_tensor in all_images:
            if phase_tensor.shape[1] != max_h or phase_tensor.shape[2] != max_w:
                phase_tensor = F.interpolate(
                    phase_tensor.permute(0, 3, 1, 2),
                    size=(max_h, max_w), mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1)
            uniform_phases.append(phase_tensor)
        images_tensor = torch.cat(uniform_phases, dim=0)

        info = {
            "source": "folders",
            "root": root_path,
            "folders_found": [name for name, _ in folders],
            "folder_assignments": assignments,
            "assignment_method": assignments[0].get("reason", "unknown") if assignments else "none",
            "phase_info": phase_info,
            "images_loaded": images_tensor.shape[0],
            "max_candidate_images": int(max_images),
            "max_per_phase": max_per_phase,
            "use_random_sample": bool(use_random),
            "_folder_list": [folder_map_global.get(i, "unknown")
                            for i in range(images_tensor.shape[0])],
            "_image_stems": image_stems_global,
            "_image_paths": image_paths_global,
            # Cache metadata
            "embedding_cache": {
                "enabled": use_embedding_cache,
                "available_for_loaded_images": sum(
                    1 for i in range(images_tensor.shape[0])
                    if image_paths_global.get(i) in cached_embeddings
                ),
                "preload_hits": cache_hits,
                "preload_misses": cache_misses,
                "cached_count": len(cached_embeddings),
                "stats": cache_stats,
            } if use_embedding_cache else {"enabled": False},
            "_cached_embeddings": cached_embeddings if cached_embeddings else {},
        }
        return images_tensor, info

    # ═══════════════════════════════════════════════════════════════════
    # Frame management (copied from BeatDropSelectorNode)
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _resize_frame(tensor, max_dim):
        h, w = tensor.shape[:2]
        if h <= max_dim and w <= max_dim:
            return tensor
        scale = max_dim / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        return F.interpolate(
            tensor.permute(2, 0, 1).unsqueeze(0),
            size=(new_h, new_w), mode="bilinear", align_corners=False,
        ).squeeze(0).permute(1, 2, 0)

    @staticmethod
    def _downsample_frames(frames, max_total, job_mask, job_fps, context_fps, mode):
        B = frames.shape[0]
        if B <= max_total:
            return frames, job_mask

        to_remove = B - max_total
        job_count = int(job_mask.sum().item())
        ctx_count = B - job_count

        if job_count == 0 or ctx_count == 0:
            keep = torch.linspace(0, B - 1, max_total).long()
            return frames[keep], job_mask[keep]

        job_weight = job_fps / max(job_fps, context_fps)
        ctx_weight = context_fps / max(job_fps, context_fps)
        total_weight = job_weight * job_count + ctx_weight * ctx_count

        job_keep_target = int(job_count * (1.0 - to_remove * job_weight / total_weight))
        ctx_keep_target = int(ctx_count * (1.0 - to_remove * ctx_weight / total_weight))
        job_keep_target = max(1, min(job_keep_target, job_count))
        ctx_keep_target = max(1, min(ctx_keep_target, ctx_count))

        job_indices = torch.where(job_mask)[0]
        ctx_indices = torch.where(~job_mask)[0]

        if len(job_indices) > 0:
            job_keep = torch.linspace(0, len(job_indices) - 1, job_keep_target).long()
            job_keep = job_indices[job_keep]
        else:
            job_keep = torch.tensor([], dtype=torch.long)

        if len(ctx_indices) > 0:
            ctx_keep = torch.linspace(0, len(ctx_indices) - 1, ctx_keep_target).long()
            ctx_keep = ctx_indices[ctx_keep]
        else:
            ctx_keep = torch.tensor([], dtype=torch.long)

        keep = torch.cat([job_keep, ctx_keep]).sort().values
        return frames[keep], job_mask[keep]

    # ═══════════════════════════════════════════════════════════════════
    # STAGE 1: Embedding-based scoring
    # ═══════════════════════════════════════════════════════════════════

    def _stage1_embedding_score(self, reference_frames, windows, model_path,
                                 device, dtype_str, scene_fit_w, change_w, diversity_w,
                                 text_query_scene, text_query_change,
                                 text_query_per_phase_json="{}",
                                 preloaded_model=None,
                                 phase_text_blend_weight=0.80,
                                 pair_constraints_json="[]",
                                 cached_frame_embs=None):
        """Compute embedding-based scores for all frames.

        If preloaded_model is provided (from QwenVLEmbeddingLoader), uses that
        instead of loading from model_path. The preloaded_model is a cache_key
        tuple (model_path, device, dtype_str) that resolves to the cached model.
        """
        B = reference_frames.shape[0]

        # ── Get model (preloaded or load fresh) ──
        if preloaded_model is not None:
            # preloaded_model is the cache_key from QwenVLEmbeddingLoader
            cache_key = preloaded_model
            if cache_key not in _QWEN_EMBEDDING_CACHE:
                raise RuntimeError(f"Preloaded model not found in cache. Key: {cache_key}")
            model, processor, dim = _QWEN_EMBEDDING_CACHE[cache_key]
            # Override model_path/device/dtype from cache
            actual_model_path, actual_device, actual_dtype = cache_key
            print(f"[EmbeddingMatcher] Using preloaded model: {actual_model_path} ({actual_device}/{actual_dtype})")
        else:
            if not model_path:
                raise RuntimeError("No model_path and no preloaded_model")
            model, processor, dim = _get_qwen_embedding_model(model_path, device, dtype_str)

        # Compute frame embeddings. Folder candidates can provide SQLite-cached
        # image embeddings in exact frame order; text/query embeddings still use
        # the loaded model below, but the expensive image pass is skipped.
        cache_used_for_frames = False
        if cached_frame_embs is not None:
            try:
                if isinstance(cached_frame_embs, torch.Tensor) and cached_frame_embs.shape[0] == B:
                    frame_embs = F.normalize(cached_frame_embs.float().cpu(), p=2, dim=1)
                    cache_used_for_frames = True
                else:
                    raise ValueError("cached_frame_embs has wrong shape")
            except Exception as e:
                print(f"[EmbeddingMatcher] Cached frame embeddings ignored: {e}")
                frame_embs = _compute_embeddings_batch(
                    model, processor, reference_frames, device, batch_size=4,
                )
        else:
            frame_embs = _compute_embeddings_batch(
                model, processor, reference_frames, device, batch_size=4,
            )  # (B, dim)

        # ── Scene embedding: mean-pool frames OUTSIDE windows as "scene reference" ──
        in_window = torch.zeros(B, dtype=torch.bool)
        for win in windows:
            in_window[win["batch_start"]:win["batch_end"]] = True

        if in_window.sum() < B:
            scene_emb = frame_embs[~in_window].mean(dim=0, keepdim=True)
            scene_emb = F.normalize(scene_emb, p=2, dim=1)
        else:
            # All frames are in windows — use mean of all frames
            scene_emb = frame_embs.mean(dim=0, keepdim=True)
            scene_emb = F.normalize(scene_emb, p=2, dim=1)

        # ── Text query embeddings (semantic matching) ──
        text_queries = {}
        if text_query_scene and text_query_scene.strip():
            text_queries["scene_fit"] = _compute_text_embeddings(
                model, processor, [text_query_scene.strip()], device,
            )
        if text_query_change and text_query_change.strip():
            text_queries["change_target"] = _compute_text_embeddings(
                model, processor, [text_query_change.strip()], device,
            )

        # ── Per-phase semantic requirements ──
        # IMPORTANT: text_query_per_phase_json is the user-facing place where
        # phrases like "full body + fully clothed" / "full body + minimal
        # clothing" are configured. Historically it was only used for folder
        # assignment, so Stage 1 never actually scored individual images against
        # those requirements. Embed the per-phase texts here and compare each
        # candidate image against the query for its own beat/drop window.
        phase_text_queries = {}
        if text_query_per_phase_json and str(text_query_per_phase_json).strip() not in ("", "{}"):
            try:
                raw_phase_queries = json.loads(text_query_per_phase_json)
                if isinstance(raw_phase_queries, dict):
                    for phase_key, phase_spec_value in raw_phase_queries.items():
                        spec = _phase_semantic_spec(phase_spec_value)
                        if not spec.get("text") and not spec.get("must") and not spec.get("avoid"):
                            continue
                        phase = int(phase_key)
                        entry = {
                            "text": spec.get("text", ""),
                            "must": spec.get("must", []),
                            "avoid": spec.get("avoid", []),
                            "filter_enabled": bool(spec.get("filter_enabled", False)),
                            "filter_threshold": float(spec.get("filter_threshold", 0.0)),
                            "contrast_mode": "must_minus_avoid" if spec.get("avoid") else "must_only",
                        }
                        if entry["text"]:
                            entry["embedding"] = _compute_text_embeddings(
                                model, processor, [entry["text"]], device,
                            )
                        if entry["must"]:
                            entry["must_embeddings"] = _compute_text_embeddings(
                                model, processor, entry["must"], device,
                            )
                        if entry["avoid"]:
                            entry["avoid_embeddings"] = _compute_text_embeddings(
                                model, processor, entry["avoid"], device,
                            )
                        phase_text_queries[phase] = entry
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                print(f"[EmbeddingMatcher] Invalid text_query_per_phase_json for scoring: {e}")

        # ── Generic relation-axis embeddings ──
        # These power cross-phase logic such as:
        #   coverage decrease, scariness increase, color brightness change,
        #   luxury similarity, etc. No domain labels are hardcoded here — presets
        #   or the AI Stack provide the axis query text.
        relation_axis_specs = _pair_relation_specs(pair_constraints_json)
        relation_axis_embs = {}
        for rel in relation_axis_specs:
            key = rel["name"]
            if key in relation_axis_embs:
                continue
            entry = {"spec": rel}
            if rel.get("query"):
                entry["query_embedding"] = _compute_text_embeddings(
                    model, processor, [rel["query"]], device,
                )
            if rel.get("positive"):
                entry["positive_embeddings"] = _compute_text_embeddings(
                    model, processor, rel["positive"], device,
                )
            if rel.get("negative"):
                entry["negative_embeddings"] = _compute_text_embeddings(
                    model, processor, rel["negative"], device,
                )
            relation_axis_embs[key] = entry

        frame_to_phase = {}
        for phase_idx, win in enumerate(windows):
            for fidx in range(win.get("batch_start", 0), win.get("batch_end", 0)):
                frame_to_phase[fidx] = phase_idx

        phase_text_blend_w = max(0.0, min(1.0, float(phase_text_blend_weight)))

        # ── Per-frame scoring ──
        scores = {}
        for i in range(B):
            frame_emb = frame_embs[i:i+1]  # (1, dim)

            # Scene fit: cosine similarity to scene embedding
            scene_fit = float(F.cosine_similarity(frame_emb, scene_emb).item())
            scene_fit = (scene_fit + 1.0) / 2.0  # [-1, 1] → [0, 1]

            # Text-query scene fit (if provided)
            text_scene_fit = 0.5  # neutral default
            if "scene_fit" in text_queries:
                ts = float(F.cosine_similarity(frame_emb, text_queries["scene_fit"]).item())
                text_scene_fit = (ts + 1.0) / 2.0
                # Blend: 50% visual scene, 50% text-query scene
                scene_fit = 0.5 * scene_fit + 0.5 * text_scene_fit

            # Change strength: average distance to frames in OTHER windows
            change_scores = []
            for other_win in windows:
                if other_win["batch_start"] <= i < other_win["batch_end"]:
                    continue  # skip own window
                ow_start = other_win["batch_start"]
                ow_end = min(other_win["batch_end"], B)
                if ow_end > ow_start:
                    other_embs = frame_embs[ow_start:ow_end]
                    dists = 1.0 - F.cosine_similarity(
                        frame_emb.expand(other_embs.shape[0], -1), other_embs,
                    )
                    change_scores.append(float(dists.mean().item()))

            change_strength = float(np.mean(change_scores)) if change_scores else 0.5

            # Text-query change target (if provided)
            if "change_target" in text_queries:
                tc = float(F.cosine_similarity(frame_emb, text_queries["change_target"]).item())
                text_change = (tc + 1.0) / 2.0
                # Blend: visual change × text-alignment ("is this the kind of change we want?")
                change_strength = change_strength * (0.5 + 0.5 * text_change)

            # Per-phase semantic fit. Preferred AI-Stack format is structured
            # must/avoid constraints, e.g. must=["full body", "covered outfit"],
            # avoid=["cropped close-up", "red clothing"]. This stays generic:
            # the node does not know clothing classes; it only computes
            # must-similarity minus avoid-similarity.
            phase_idx = frame_to_phase.get(i)
            phase_text_fit = None
            raw_phase_fit = None
            must_fit = None
            avoid_fit = None
            contrast_margin = None
            semantic_filter_failed = False
            phase_query = None
            if phase_idx in phase_text_queries:
                phase_entry = phase_text_queries[phase_idx]
                phase_query = phase_entry.get("text", "")

                if "embedding" in phase_entry:
                    pq = float(F.cosine_similarity(
                        frame_emb, phase_entry["embedding"],
                    ).item())
                    raw_phase_fit = (pq + 1.0) / 2.0

                if "must_embeddings" in phase_entry:
                    must_embs = phase_entry["must_embeddings"]
                    must_sims = F.cosine_similarity(
                        frame_emb.expand(must_embs.shape[0], -1), must_embs,
                    )
                    must_fit = float(((must_sims + 1.0) / 2.0).mean().item())

                if "avoid_embeddings" in phase_entry:
                    avoid_embs = phase_entry["avoid_embeddings"]
                    avoid_sims = F.cosine_similarity(
                        frame_emb.expand(avoid_embs.shape[0], -1), avoid_embs,
                    )
                    avoid_fit = float(((avoid_sims + 1.0) / 2.0).mean().item())

                if must_fit is not None and avoid_fit is not None:
                    contrast_margin = must_fit - avoid_fit
                    # Qwen embedding margins are small on real image sets.
                    # Amplify generically: positive margin = closer to must
                    # than avoid. This works for coverage, crop, color, style.
                    contrast_fit = max(0.0, min(1.0, 0.5 + 4.0 * contrast_margin))
                    phase_text_fit = contrast_fit
                    if raw_phase_fit is not None:
                        phase_text_fit = 0.25 * raw_phase_fit + 0.75 * contrast_fit
                elif must_fit is not None:
                    phase_text_fit = must_fit
                    if raw_phase_fit is not None:
                        phase_text_fit = 0.35 * raw_phase_fit + 0.65 * must_fit
                elif avoid_fit is not None:
                    phase_text_fit = 1.0 - avoid_fit
                    if raw_phase_fit is not None:
                        phase_text_fit = 0.35 * raw_phase_fit + 0.65 * phase_text_fit
                elif raw_phase_fit is not None:
                    phase_text_fit = raw_phase_fit

                if phase_text_fit is not None:
                    if phase_entry.get("filter_enabled", False):
                        threshold = float(phase_entry.get("filter_threshold", 0.0))
                        filter_score = contrast_margin if contrast_margin is not None else phase_text_fit
                        semantic_filter_failed = filter_score < threshold
                    scene_fit = (1.0 - phase_text_blend_w) * scene_fit + phase_text_blend_w * phase_text_fit

            # Generic relation-axis values for pair reasoning. Example axes:
            # coverage, scariness, brightness, luxury, streetwear, formal, etc.
            relation_axis_values = {}
            relation_axis_details = {}
            for axis_name, axis_entry in relation_axis_embs.items():
                spec = axis_entry.get("spec", {})
                axis_value, axis_detail = _axis_value_from_embedding(
                    frame_emb, axis_entry, spec.get("axis_scale", 4.0),
                )
                if axis_value is None:
                    continue
                relation_axis_values[axis_name] = axis_value
                relation_axis_details[axis_name] = axis_detail

            # Composite score
            composite = (
                scene_fit_w * scene_fit +
                change_w * change_strength
            )
            # diversity_w is applied later (across selected set)

            scores[i] = {
                "scene_fit": round(scene_fit, 4),
                "change_strength": round(change_strength, 4),
                "composite": round(composite, 4),
            }
            if phase_text_fit is not None:
                scores[i]["phase_text_fit"] = round(phase_text_fit, 4)
                if raw_phase_fit is not None:
                    scores[i]["raw_phase_text_fit"] = round(raw_phase_fit, 4)
                if must_fit is not None:
                    scores[i]["phase_must_fit"] = round(must_fit, 4)
                if avoid_fit is not None:
                    scores[i]["phase_avoid_fit"] = round(avoid_fit, 4)
                if contrast_margin is not None:
                    scores[i]["phase_contrast_margin"] = round(contrast_margin, 4)
                    scores[i]["phase_contrast_mode"] = phase_text_queries[phase_idx].get("contrast_mode", "none")
                if semantic_filter_failed:
                    scores[i]["semantic_filter_failed"] = True
                scores[i]["phase"] = phase_idx
                scores[i]["phase_query"] = phase_query
            if relation_axis_values:
                scores[i]["relation_axes"] = {
                    k: round(float(v), 4) for k, v in relation_axis_values.items()
                }
                scores[i]["relation_axis_details"] = relation_axis_details
                for k, v in relation_axis_values.items():
                    scores[i][f"relation_axis_{k}"] = round(float(v), 4)

        # ── Confidence: how spread out are the top scores? ──
        composites = sorted([s["composite"] for s in scores.values()], reverse=True)
        if len(composites) >= 2:
            # High spread = high confidence (clear winners)
            spread = composites[0] - composites[min(len(composites) - 1, len(composites) // 2)]
            confidence = min(1.0, max(0.0, spread * 2.0))
        else:
            confidence = 0.5

        stage_info = {
            "stage": "embedding",
            "model": model_path,
            "device": device,
            "dtype": dtype_str,
            "dim": dim,
            "frames_embedded": B,
            "frame_embedding_source": "sqlite_cache" if cache_used_for_frames else "model",
            "scene_fit_weight": scene_fit_w,
            "change_strength_weight": change_w,
            "diversity_weight": diversity_w,
            "phase_text_blend_weight": phase_text_blend_w,
            "text_queries_used": list(text_queries.keys()),
            "per_phase_text_queries_used": {
                str(k): v["text"] for k, v in phase_text_queries.items()
            },
            "per_phase_contrast_modes": {
                str(k): v.get("contrast_mode", "none") for k, v in phase_text_queries.items()
            },
            "relation_axes_used": [
                {
                    "name": rel["name"],
                    "query": rel["query"],
                    "positive": rel.get("positive", []),
                    "negative": rel.get("negative", []),
                    "bipolar": rel.get("bipolar", False),
                    "mode": rel["mode"],
                    "weight": rel["weight"],
                    "scale": rel["scale"],
                    "axis_scale": rel.get("axis_scale", 4.0),
                    "min_score": rel.get("min_score", 0.0),
                    "gate": rel.get("gate", False),
                }
                for rel in relation_axis_specs
            ],
            "top_composite_range": [composites[0], composites[-1]] if composites else [0, 0],
            "confidence": round(confidence, 4),
            "_frame_embs_internal": frame_embs,
        }

        return scores, stage_info, confidence

    # ═══════════════════════════════════════════════════════════════════
    # VLM Fallback (Stage 3)
    # ═══════════════════════════════════════════════════════════════════

    def _vlm_judge_window(self, endpoint, headers, model, system_prompt,
                           contact_sheet_pil, window_info, max_tokens,
                           temperature, timeout):
        import requests
        data_url = _encode_pil_to_data_url(contact_sheet_pil)
        frame_count = window_info.get("window_frames", 0)

        prompt_text = (
            f"Beatdrop window. This window contains {frame_count} frames. "
            f"Each frame is labeled with its 1-based index in the contact sheet. "
            f"Select the {window_info.get('max_frames', 4)} best frames. "
            f"You MUST select at least {window_info.get('num_outfits', 2)} VISIBLY DIFFERENT outfits.\n"
        )
        extra = str(window_info.get('extra_instructions', '')).strip()
        if extra:
            prompt_text += f"\nADDITIONAL INSTRUCTIONS:\n{extra}\n"

        prompt_text += "Return ONLY valid JSON:\n"
        prompt_text += '{"selected_ids": [1, 3, 5], "confidence": 0.85, "reason": "..."}'

        content = [
            {"type": "text", "text": prompt_text},
            _image_url_part(data_url, detail="auto"),
        ]
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        return _extract_choice_content(resp.json())

    def _vlm_judge_pairs(self, endpoint, headers, model, system_prompt,
                         pair_images, pair_summaries, query, max_tokens,
                         temperature, timeout, grid_columns=2):
        """Optional final visual judge for BEFORE/AFTER pair candidates.

        The selector/reranker first narrows the search. This VLM sees only a
        small labeled contact sheet of side-by-side pairs and returns the best
        1-based pair id. It is opt-in via pair_vlm.enabled and never runs unless
        an endpoint is configured.
        """
        import requests
        if not pair_images:
            return {"run": False, "reason": "no pair images"}

        labels = [str(i + 1) for i in range(len(pair_images))]
        sheet = _build_contact_sheet(pair_images, max(1, int(grid_columns)), labels)
        data_url = _encode_pil_to_data_url(sheet)
        summary_text = "\n".join(pair_summaries[:len(pair_images)])
        prompt_text = (
            "You are judging BEFORE/AFTER beatdrop image pairs. Each labeled tile is one pair: "
            "left image = BEFORE/source phase, right image = AFTER/target phase.\n\n"
            f"TASK:\n{query}\n\n"
            "Candidate pair score summaries:\n"
            f"{summary_text}\n\n"
            "Choose exactly ONE best pair. Prefer pairs that satisfy the semantic relation axes, "
            "match the reference pose/framing, and avoid cropped/lying/sitting/low-quality images.\n"
            "Return ONLY valid JSON:\n"
            '{"selected_pair_id": 1, "confidence": 0.85, "reason": "..."}'
        )
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                _image_url_part(data_url, detail="auto"),
            ],
        })
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        raw = _extract_choice_content(resp.json())
        parsed = None
        try:
            cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
            parsed = json.loads(cleaned)
        except Exception:
            parsed = None
        selected = None
        if isinstance(parsed, dict):
            for key in ("selected_pair_id", "selected_id", "pair_id", "id"):
                if key in parsed:
                    try:
                        selected = int(parsed[key])
                        break
                    except (TypeError, ValueError):
                        pass
            if selected is None and isinstance(parsed.get("selected_ids"), list) and parsed["selected_ids"]:
                try:
                    selected = int(parsed["selected_ids"][0])
                except (TypeError, ValueError):
                    selected = None
        return {
            "run": True,
            "raw": raw,
            "parsed": parsed,
            "selected_pair_id": selected,
            "selected_doc_index": selected - 1 if selected is not None else None,
        }

    # ═══════════════════════════════════════════════════════════════════
    # MAIN SELECT
    # ═══════════════════════════════════════════════════════════════════

    def select(self, max_frames_per_window, num_outfits_mode, num_outfits,
               embedding_model=None,  # ← from QwenVLEmbeddingLoader
               reference_frames=None, context_frames=None, beats_used="",
               # Stage 1 (scoring params only — model comes from embedding_model)
               embedding_confidence_threshold=0.75,
               embedding_scene_fit_weight=0.30,
               embedding_change_strength_weight=0.50,
               embedding_diversity_weight=0.20,
               text_query_scene_fit="",
               text_query_change_target="",
               # Stage 2
               #[Stage 2 Reranker params]
               reranker_model=None,  # ← from QwenVLRerankerLoader
               reranker_mode="api",
                reranker_endpoint="", reranker_api_model="",
               reranker_model_path="Qwen/Qwen3-VL-Reranker-8B",
               reranker_device="auto", reranker_dtype="fp16",
               reranker_top_k=12, reranker_query="",
               reranker_blend_weight=0.5,
               reranker_confidence_threshold=0.70,
               # Stage 3
               use_vlm_fallback=False,
               vlm_endpoint="", vlm_api_token="", vlm_model="",
               vlm_system_prompt="", vlm_max_tokens=512,
               vlm_temperature=0.0, vlm_timeout=120,
               # Shared
               max_total_frames=100, job_fps=5.0, context_fps=1.0,
               downsample_mode="global", image_resolution=512,
               history_file="", history_penalty=10.0,
               history_max_entries=200, history_decay_rate=0.3,
               extra_instructions="", extra_penalty_json="{}", penalty=0.0,
               conversation_id="",
               grid_columns=4, add_id_labels=True,
               candidate_folders="", max_candidate_images=30,
               min_image_height=0, min_aspect_ratio=0.0,
               use_random_sample=True,
               text_query_per_phase_json="{}",
               folder_assignments_json="",
               use_embedding_cache=False,
               cache_db_path="",
               ai_stack_config_json="{}"):

        pair_constraints = []
        pair_reranker_config = {}
        phase_text_blend_weight = 0.80

        # ── Apply AI Stack config overrides (Python merge) ──
        if ai_stack_config_json and ai_stack_config_json.strip() not in ("", "{}"):
            try:
                cfg = json.loads(ai_stack_config_json)
                if isinstance(cfg, dict):
                    # Text overrides
                    if "extra_instructions" in cfg: extra_instructions = str(cfg["extra_instructions"])
                    if "text_query_scene_fit" in cfg: text_query_scene_fit = str(cfg["text_query_scene_fit"])
                    if "text_query_change_target" in cfg: text_query_change_target = str(cfg["text_query_change_target"])
                    if "reranker_query" in cfg: reranker_query = str(cfg["reranker_query"])
                    if "conversation_id" in cfg: conversation_id = str(cfg["conversation_id"])
                    # Weights
                    if "weights" in cfg and isinstance(cfg["weights"], dict):
                        w = cfg["weights"]
                        if "scene_fit" in w: embedding_scene_fit_weight = float(w["scene_fit"])
                        if "change_strength" in w: embedding_change_strength_weight = float(w["change_strength"])
                        if "diversity" in w: embedding_diversity_weight = float(w["diversity"])
                        if "reranker_blend" in w: reranker_blend_weight = float(w["reranker_blend"])
                        if "phase_text_blend" in w: phase_text_blend_weight = float(w["phase_text_blend"])
                        if "reference_match" in w: phase_text_blend_weight = 1.0 - float(w["reference_match"])
                    # Thresholds
                    if "thresholds" in cfg and isinstance(cfg["thresholds"], dict):
                        t = cfg["thresholds"]
                        if "embedding_confidence" in t: embedding_confidence_threshold = float(t["embedding_confidence"])
                        if "reranker_confidence" in t: reranker_confidence_threshold = float(t["reranker_confidence"])
                    # Johnson history
                    if "history" in cfg and isinstance(cfg["history"], dict):
                        h = cfg["history"]
                        if "penalty" in h: history_penalty = float(h["penalty"])
                        if "decay_rate" in h: history_decay_rate = float(h["decay_rate"])
                        if "max_entries" in h: history_max_entries = int(h["max_entries"])
                    # Limits
                    if "max_frames_per_window" in cfg: max_frames_per_window = int(cfg["max_frames_per_window"])
                    if "max_candidate_images" in cfg: max_candidate_images = int(cfg["max_candidate_images"])
                    # Stages
                    if "use_vlm_fallback" in cfg: use_vlm_fallback = bool(cfg["use_vlm_fallback"])
                    # Per-phase text queries
                    if "text_query_per_phase" in cfg and isinstance(cfg["text_query_per_phase"], dict):
                        text_query_per_phase_json = json.dumps(cfg["text_query_per_phase"])
                    if "pair_constraints" in cfg:
                        pc = cfg["pair_constraints"]
                        if isinstance(pc, dict):
                            pair_constraints = [pc]
                        elif isinstance(pc, list):
                            pair_constraints = [p for p in pc if isinstance(p, dict)]
                    if "pair_reranker" in cfg and isinstance(cfg["pair_reranker"], dict):
                        pair_reranker_config = cfg["pair_reranker"]
                    if "folder_assignments" in cfg and isinstance(cfg["folder_assignments"], dict):
                        folder_assignments_json = json.dumps(cfg["folder_assignments"])
                    print(f"[EmbeddingMatcher] AI Stack config applied: {list(cfg.keys())}")
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                print(f"[EmbeddingMatcher] AI Stack config parse error: {e}")

        # ── Load images ──
        folder_info = None  # populated when loading from folders
        if reference_frames is None or not isinstance(reference_frames, torch.Tensor):
            folder_path = str(candidate_folders or "").strip()
            if folder_path:
                history = self._load_history(history_file)

                # Parse windows early for embedding-based folder assignment
                try:
                    bu_early = json.loads(beats_used or "[]")
                    early_windows = []
                    if isinstance(bu_early, list):
                        for entry in bu_early:
                            offset = int(entry.get("batch_offset", -1))
                            count = int(entry.get("batch_frame_count", 0))
                            if offset >= 0 and count > 0:
                                early_windows.append({
                                    **entry,
                                    "batch_start": offset,
                                    "batch_end": offset + count,
                                })
                    nw = max(2, len(early_windows) if early_windows else 2)
                except Exception:
                    early_windows = []
                    nw = 2

                folder_embedding_model_path = ""
                folder_embedding_device = "auto"
                folder_embedding_dtype = "fp16"
                if not folder_embedding_model_path and isinstance(embedding_model, (tuple, list)) and len(embedding_model) >= 3:
                    folder_embedding_model_path = str(embedding_model[0])
                    folder_embedding_device = str(embedding_model[1])
                    folder_embedding_dtype = str(embedding_model[2])

                images, folder_info = self._load_from_folders(
                    folder_path, history, history_penalty, history_decay_rate,
                    max_candidate_images, use_random_sample, num_windows=nw,
                    folder_assignments_json=folder_assignments_json,
                    text_query_per_phase_json=text_query_per_phase_json,
                    embedding_model_path=folder_embedding_model_path,
                    embedding_device=folder_embedding_device,
                    embedding_dtype=folder_embedding_dtype,
                    reference_frames=None,  # video frames not yet loaded
                    windows=early_windows if early_windows else None,
                    use_embedding_cache=use_embedding_cache,
                    cache_db_path=cache_db_path,
                    min_image_height=min_image_height,
                    min_aspect_ratio=min_aspect_ratio,
                )
                if images is None:
                    return ("", 0, json.dumps({"error": "no images in folders"}),
                            _make_blank_image(), "", "")
                reference_frames = images
            else:
                return ("", 0, json.dumps({"error": "no images provided and no candidate_folders set"}),
                        _make_blank_image(), "", "")

        B = reference_frames.shape[0]
        max_total = max(5, int(max_total_frames))

        # ── Parse windows from beats_used ──
        windows = []
        try:
            bu = json.loads(beats_used or "[]")
            if isinstance(bu, list):
                for entry in bu:
                    offset = int(entry.get("batch_offset", -1))
                    count = int(entry.get("batch_frame_count", 0))
                    if offset >= 0 and count > 0 and offset < B:
                        windows.append({
                            **entry,
                            "batch_start": offset,
                            "batch_end": min(B, offset + count),
                            "frame_indices": list(range(offset, min(B, offset + count))),
                        })
        except Exception:
            pass

        if not windows:
            windows = [{"batch_start": 0, "batch_end": B,
                        "frame_indices": list(range(B)), "_flat": True}]

        # ── Build job mask + merge context frames ──
        job_mask = torch.zeros(B, dtype=torch.bool)
        for w in windows:
            job_mask[w["batch_start"]:w["batch_end"]] = True

        if context_frames is not None and isinstance(context_frames, torch.Tensor):
            ctx_B = context_frames.shape[0]
            if ctx_B > 0:
                # Folder candidates and driving-video frames often have different
                # H/W. Resize context frames to candidate tensor size before cat;
                # otherwise reference matching crashes before embedding.
                if context_frames.shape[1] != reference_frames.shape[1] or context_frames.shape[2] != reference_frames.shape[2]:
                    context_frames = F.interpolate(
                        context_frames.permute(0, 3, 1, 2),
                        size=(reference_frames.shape[1], reference_frames.shape[2]),
                        mode="bilinear",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)
                ctx_mask = torch.zeros(ctx_B, dtype=torch.bool)
                reference_frames = torch.cat([reference_frames, context_frames], dim=0)
                job_mask = torch.cat([job_mask, ctx_mask], dim=0)
                B = reference_frames.shape[0]

        # ── Downsample ──
        if B > max_total:
            reference_frames, job_mask = self._downsample_frames(
                reference_frames, max_total, job_mask,
                float(job_fps), float(context_fps), str(downsample_mode),
            )
            B = reference_frames.shape[0]
            windows = [{"batch_start": 0, "batch_end": B,
                        "frame_indices": list(range(B)),
                        "_flat": True, "_downsampled": True}]

        # ── Resize for embedding model ──
        max_res = max(64, int(image_resolution))
        resized = []
        for i in range(B):
            resized.append(self._resize_frame(reference_frames[i], max_res))
        reference_frames = torch.stack(resized, dim=0)

        n_per_window = max(1, min(int(max_frames_per_window), B))

        # ── extra_penalties ──
        extra_penalties = {}
        try:
            extra_penalties = json.loads(extra_penalty_json or "{}")
        except Exception:
            pass
        if not isinstance(extra_penalties, dict):
            extra_penalties = {}

        # ── num_outfits ──
        if str(num_outfits_mode).strip() == "auto_from_beats":
            num_outfits = max(2, len(windows))
        else:
            num_outfits = max(2, int(num_outfits))
        num_windows = len(windows)
        min_per_window = max(1, (num_outfits + num_windows - 1) // num_windows)
        n_per_window = max(n_per_window, min_per_window)
        n_per_window = min(n_per_window, B)

        # Load history
        history = self._load_history(history_file)
        image_stems_by_frame = {}
        cached_frame_embs = None
        if isinstance(folder_info, dict):
            image_stems_by_frame = folder_info.get("_image_stems", {}) or {}
            image_paths_by_frame = folder_info.get("_image_paths", {}) or {}
            cached_by_path = folder_info.get("_cached_embeddings", {}) or {}
            if image_paths_by_frame and cached_by_path:
                ordered = []
                for i in range(B):
                    path = image_paths_by_frame.get(i)
                    emb = cached_by_path.get(path)
                    if emb is None:
                        ordered = []
                        break
                    ordered.append(emb if emb.dim() == 1 else emb.squeeze(0))
                if len(ordered) == B:
                    cached_frame_embs = torch.stack(ordered, dim=0)

        # ═══════════════════════════════════════════════════════════════
        # STAGE 1: Embedding-based scoring
        # ═══════════════════════════════════════════════════════════════
        stage_used = "none"
        embedding_scores = {}
        embedding_conf = 0.0
        pair_frame_embs = None

        if embedding_model is not None:
            try:
                embedding_scores, stage1_info, embedding_conf = self._stage1_embedding_score(
                    reference_frames, windows,
                    None, "auto", "fp16",  # not used when preloaded_model is set
                    embedding_scene_fit_weight, embedding_change_strength_weight,
                    embedding_diversity_weight,
                    text_query_scene_fit, text_query_change_target,
                    text_query_per_phase_json,
                    preloaded_model=embedding_model,
                    phase_text_blend_weight=phase_text_blend_weight,
                    pair_constraints_json=json.dumps(pair_constraints),
                    cached_frame_embs=cached_frame_embs,
                )
                pair_frame_embs = stage1_info.pop("_frame_embs_internal", None)
                stage_used = "embedding"
                print(f"[EmbeddingMatcher] Stage 1 confidence: {embedding_conf:.4f} "
                      f"(threshold: {embedding_confidence_threshold})")
            except Exception as e:
                print(f"[EmbeddingMatcher] Stage 1 FAILED: {e}")
                import traceback
                traceback.print_exc()
                stage1_info = {"stage": "embedding", "error": str(e)}
                embedding_conf = 0.0
        else:
            stage1_info = {"stage": "embedding", "skipped": "no model path provided"}
            embedding_conf = 0.0

        # ── Embedding model auto-unload after Stage 1 ──
        stage2_transformers_will_run = (
            str(reranker_mode).strip() == "transformers"
            and (stage_used == "none" or embedding_conf < embedding_confidence_threshold)
        )
        pair_transformers_will_run = bool(pair_reranker_config.get("enabled", False)) and (
            str(pair_reranker_config.get("mode", reranker_mode)).strip() == "transformers"
        )
        needs_vram_for_reranker = stage2_transformers_will_run or pair_transformers_will_run
        if stage_used == "embedding" and embedding_model is not None and needs_vram_for_reranker:
            global _QWEN_EMBEDDING_CACHE
            if embedding_model in _QWEN_EMBEDDING_CACHE:
                m, p, d = _QWEN_EMBEDDING_CACHE.pop(embedding_model)
                del m, p
                import gc; gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[EmbeddingMatcher] Auto-unloaded embedding model (freed VRAM for reranker)")

        # ═══════════════════════════════════════════════════════════════
        # STAGE 2: Re-Ranker (api or transformers)
        # ═══════════════════════════════════════════════════════════════
        reranker_scores = {}
        reranker_conf = 0.0
        top_k_set = set()
        stage2_info = {"stage": "reranker", "run": False}

        if stage_used == "embedding" and embedding_conf < embedding_confidence_threshold:
            print(f"[EmbeddingMatcher] Stage 1 uncertain ({embedding_conf:.4f} < {embedding_confidence_threshold})"
                  f" → Stage 2: Re-Ranker")
        elif stage_used != "embedding":
            print("[EmbeddingMatcher] Stage 1 skipped → Stage 2: Re-Ranker")
        reranker_will_run = (stage_used == "none" or embedding_conf < embedding_confidence_threshold)

        if reranker_will_run:
            # ── Build query + documents (shared between api and transformers mode) ──
            if reranker_query and reranker_query.strip():
                query = reranker_query
            else:
                query = (
                    "Find outfits that fit TWO criteria simultaneously:\n"
                    "1) SCENE FIT: Does this outfit match the scene lighting, pose, camera angle, vibe?\n"
                    "2) CHANGE STRENGTH: Is this outfit VISIBLY DIFFERENT from the old outfit — "
                    "different silhouette, cut, shape, style — so the change is immediately noticeable?\n"
                    "Outfits that fail EITHER criterion should score low."
                )
            if extra_instructions and extra_instructions.strip():
                query += f"\n\nADDITIONAL INSTRUCTIONS: {extra_instructions.strip()}"

            documents = []
            for i in range(B):
                if embedding_scores and i in embedding_scores:
                    es = embedding_scores[i]
                    doc = (f"Frame {i}: scene_fit={es['scene_fit']:.3f}, "
                           f"change={es['change_strength']:.3f}, composite={es['composite']:.3f}")
                else:
                    doc = f"Frame {i}: beatdrop candidate image"
                documents.append(doc)

            top_k = max(0, int(reranker_top_k))

            if reranker_mode == "transformers" and reranker_model_path:
                # ── Direct transformers reranker ──
                rr = self._run_reranker_transformers(
                    query=query,
                    documents=documents,
                    document_images=reference_frames,
                    top_n=max(top_k, B),
                    model_path=reranker_model_path,
                    device=reranker_device,
                    dtype_str=reranker_dtype,
                    preloaded_model=reranker_model,
                )
            elif reranker_endpoint and reranker_endpoint.strip():
                # ── API reranker (existing) ──
                reranker_headers = {"Content-Type": "application/json"}
                if vlm_api_token:
                    reranker_headers["Authorization"] = f"Bearer {vlm_api_token}"
                rr = self._run_reranker(
                    reranker_endpoint, reranker_headers, reranker_api_model,
                    query, documents, top_n=max(top_k, B), timeout=vlm_timeout,
                )
            else:
                rr = None

            if rr:
                # Normalize scores
                rr_scores = [s for _, s in rr]
                r_min, r_max = min(rr_scores), max(rr_scores)
                r_range = r_max - r_min if r_max > r_min else 1.0
                for idx, score in rr:
                    normalized = ((score - r_min) / r_range) * 100.0
                    reranker_scores[idx] = normalized
                if top_k > 0 and top_k < len(rr):
                    for idx, _ in rr[:top_k]:
                        top_k_set.add(idx)
                stage_used = "reranker"

                # Confidence: spread of top scores
                top_scores = [s for _, s in rr[:min(5, len(rr))]]
                if len(top_scores) >= 2:
                    reranker_conf = min(1.0, max(0.0, (top_scores[0] - top_scores[-1]) / max(r_range, 0.01) * 2))
                else:
                    reranker_conf = 0.5

                stage2_info = {
                    "stage": "reranker",
                    "run": True,
                    "frames_scored": len(reranker_scores),
                    "visual_documents": reranker_mode == "transformers",
                    "top_scores_range": [round(top_scores[0], 4), round(top_scores[-1], 4)] if top_scores else [0, 0],
                    "confidence": round(reranker_conf, 4),
                }
                print(f"[EmbeddingMatcher] Stage 2 (Re-Ranker) confidence: {reranker_conf:.4f}")
            else:
                stage2_info = {"stage": "reranker", "run": True, "error": "reranker returned no scores"}
                print("[EmbeddingMatcher] Stage 2 (Re-Ranker) returned no scores — using raw scores")
        elif not reranker_will_run:
            stage2_info = {"stage": "reranker", "run": False, "reason": "embedding confidence sufficient"}

        blend_w = max(0.0, min(1.0, float(reranker_blend_weight)))

        # ═══════════════════════════════════════════════════════════════
        # STAGE 3: VLM Fallback (if both embedding + reranker uncertain)
        # ═══════════════════════════════════════════════════════════════
        vlm_fallback_used = False
        vlm_raw_responses = []

        current_conf = reranker_conf if stage_used == "reranker" else embedding_conf
        needs_vlm = (
            use_vlm_fallback
            and vlm_endpoint and vlm_endpoint.strip()
            and vlm_model and vlm_model.strip()
            and current_conf < reranker_confidence_threshold
        )

        # ── Build scoring function ──
        def _score_frame(fidx, penalties_to_apply, all_selected_so_far, n_per_win, hist_pen):
            s = 0.0

            # Judge extra penalties
            for key, val in penalties_to_apply.items():
                if str(fidx) in str(key):
                    s += float(val)

            # History penalty
            hist_p = self._history_penalty_for(fidx, history, float(hist_pen), float(history_decay_rate))
            s += hist_p

            # Embedding score (invert: high composite = low penalty)
            if embedding_scores and fidx in embedding_scores:
                if embedding_scores[fidx].get("semantic_filter_failed"):
                    s += float(hist_pen) * 10.0
                emb_score = embedding_scores[fidx]["composite"]
                s -= emb_score * 5.0  # bonus for high embedding score

            # Re-Ranker blend
            if fidx in reranker_scores and blend_w > 0:
                rr_score = reranker_scores[fidx]
                s = (1.0 - blend_w) * s + blend_w * (100.0 - rr_score)

            # Diversity penalty
            for prev_idx in all_selected_so_far:
                # If a flat/root pool is reused across phases, the same source
                # image can appear at two different frame indices. Penalize by
                # source stem, not just nearby frame index, so pre/post drop do
                # not pick the exact same outfit image.
                if image_stems_by_frame.get(fidx) and image_stems_by_frame.get(fidx) == image_stems_by_frame.get(prev_idx):
                    s += float(hist_pen) * 10.0
                dist = abs(fidx - prev_idx)
                if dist < n_per_win:
                    s += float(hist_pen) * (1.0 - dist / max(n_per_win, 1))

            return s

        def _score_frame_vlm(fidx, penalties_to_apply, all_selected_so_far, n_per_win, hist_pen):
            """Score without embedding/reranker (pure VLM path)."""
            s = 0.0
            for key, val in penalties_to_apply.items():
                if str(fidx) in str(key):
                    s += float(val)
            hist_p = self._history_penalty_for(fidx, history, float(hist_pen), float(history_decay_rate))
            s += hist_p
            for prev_idx in all_selected_so_far:
                dist = abs(fidx - prev_idx)
                if dist < n_per_win:
                    s += float(hist_pen) * (1.0 - dist / max(n_per_win, 1))
            return s

        forced_selected_by_window = {}
        pair_constraint_debug = []
        hist_pen = float(history_penalty)

        # ── Optional joint pair scoring ──
        # Per-phase must/avoid scores judge each candidate alone. Pair constraints
        # judge combinations, e.g. phase 0 covered + phase 1 more revealing with
        # similar color. This can choose a slightly lower individual phase-0
        # candidate if it forms a better before/after pair.
        if pair_constraints and len(windows) >= 2 and n_per_window == 1:
            color_sigs = [_color_signature(reference_frames[i]) for i in range(B)]
            frame_pils_for_pairs = None
            try:
                frame_pils_for_pairs = _image_batch_to_pil_images(reference_frames)
            except Exception:
                frame_pils_for_pairs = None

            for pc in pair_constraints:
                try:
                    source_phase = int(pc.get("source_phase", pc.get("from_phase", pc.get("source", 0))))
                    target_phase = int(pc.get("target_phase", pc.get("to_phase", pc.get("target", 1))))
                except (TypeError, ValueError):
                    continue
                if source_phase >= len(windows) or target_phase >= len(windows):
                    continue
                match = pc.get("match", pc.get("same", pc.get("constraints", [])))
                if isinstance(match, str):
                    match = [match]
                match = [str(m).lower().replace("-", "_") for m in (match or [])]
                relation_specs = _pair_relation_specs([pc])

                same_flags = {"color", "same_color", "similar_color", "colour", "same_style", "same_style_family"}
                diff_color_flags = {"different_color", "color_change", "colour_change", "contrast_color", "opposite_color"}
                diff_visual_flags = {"visual_change", "strong_visual_change", "different_style", "style_change", "different", "maximize_difference"}
                has_generic_pair_logic = bool(
                    any(m in same_flags or m in diff_color_flags or m in diff_visual_flags for m in match)
                    or relation_specs
                )
                if not has_generic_pair_logic:
                    continue
                try:
                    weight = float(pc.get("weight", pc.get("color_weight", 8.0)))
                except (TypeError, ValueError):
                    weight = 8.0

                source_indices = windows[source_phase].get("frame_indices", [])
                target_indices = windows[target_phase].get("frame_indices", [])

                # Base-score candidates first. PairRanker only needs Top-K×Top-K,
                # not all images; this keeps latency bounded.
                try:
                    top_k_per_phase = int(pair_reranker_config.get("top_k_per_phase", pc.get("top_k_per_phase", 10)))
                except (TypeError, ValueError):
                    top_k_per_phase = 10
                top_k_per_phase = max(1, min(30, top_k_per_phase))

                def _ranked_candidates_for_phase(indices, phase_role):
                    """Union multiple buckets: base score + relation-axis extremes + change strength."""
                    base = [
                        (idx, _score_frame(idx, extra_penalties, [], n_per_window, hist_pen))
                        for idx in indices
                    ]
                    base_sorted = sorted(base, key=lambda x: x[1])
                    selected = {}

                    def _add_many(items):
                        for idx, score in items:
                            if idx not in selected:
                                selected[idx] = score

                    _add_many(base_sorted[:top_k_per_phase])

                    # Relation buckets: for increase, source should be low and target high;
                    # for decrease, source should be high and target low. This lets a
                    # pair candidate survive even when its individual phase score is not top-1.
                    for rel in relation_specs:
                        axis = rel.get("name")
                        mode_rel = str(rel.get("mode", "maximize_difference")).lower().replace("-", "_")
                        vals = []
                        for idx, score in base:
                            axes = (embedding_scores.get(idx, {}) or {}).get("relation_axes", {})
                            if axis in axes:
                                vals.append((idx, score, float(axes[axis])))
                        if not vals:
                            continue
                        if mode_rel in ("increase", "more", "higher", "target_more", "target_gt_source"):
                            reverse = phase_role == "target"
                            _add_many([(i, s) for i, s, _ in sorted(vals, key=lambda x: x[2], reverse=reverse)[:top_k_per_phase]])
                        elif mode_rel in ("decrease", "less", "lower", "target_less", "target_lt_source"):
                            reverse = phase_role == "source"
                            _add_many([(i, s) for i, s, _ in sorted(vals, key=lambda x: x[2], reverse=reverse)[:top_k_per_phase]])
                        elif mode_rel in ("same", "similar", "minimize_difference", "min_diff", "keep"):
                            # Keep strong exemplars of the axis on both sides; actual closeness is pair-scored later.
                            _add_many([(i, s) for i, s, _ in sorted(vals, key=lambda x: x[2], reverse=True)[:top_k_per_phase]])
                        else:
                            # For maximize-difference, include both extremes.
                            _add_many([(i, s) for i, s, _ in sorted(vals, key=lambda x: x[2])[:max(1, top_k_per_phase // 2)]])
                            _add_many([(i, s) for i, s, _ in sorted(vals, key=lambda x: x[2], reverse=True)[:max(1, top_k_per_phase // 2)]])

                    if any(m in diff_visual_flags for m in match):
                        change_vals = []
                        for idx, score in base:
                            es = embedding_scores.get(idx, {}) or {}
                            change_vals.append((idx, score, float(es.get("change_strength", 0.0))))
                        _add_many([(i, s) for i, s, _ in sorted(change_vals, key=lambda x: x[2], reverse=True)[:top_k_per_phase]])

                    try:
                        max_candidates = int(pair_reranker_config.get(
                            "max_candidates_per_phase", pc.get("max_candidates_per_phase", top_k_per_phase * 3)
                        ))
                    except (TypeError, ValueError):
                        max_candidates = top_k_per_phase * 3
                    max_candidates = max(top_k_per_phase, min(80, max_candidates))
                    ranked = sorted(selected.items(), key=lambda x: x[1])[:max_candidates]
                    return ranked

                source_ranked = _ranked_candidates_for_phase(source_indices, "source")
                target_ranked = _ranked_candidates_for_phase(target_indices, "target")

                pair_items = []
                for sidx, s0 in source_ranked:
                    for tidx, _ in target_ranked:
                        s1 = _score_frame(tidx, extra_penalties, [sidx], n_per_window, hist_pen)
                        if image_stems_by_frame.get(sidx) and image_stems_by_frame.get(sidx) == image_stems_by_frame.get(tidx):
                            s1 += hist_pen * 10.0
                        color_sim = _color_similarity(color_sigs[sidx], color_sigs[tidx])
                        image_sim = 0.0
                        if pair_frame_embs is not None:
                            try:
                                image_sim = (float(F.cosine_similarity(
                                    pair_frame_embs[sidx:sidx+1], pair_frame_embs[tidx:tidx+1],
                                ).item()) + 1.0) / 2.0
                            except Exception:
                                image_sim = 0.0
                        pair_same_score = 0.0
                        pair_diff_score = 0.0
                        pair_match_parts = []
                        if any(m in same_flags for m in match):
                            # Back-compat path: old same_color+same_style_family behavior.
                            pair_same_score = 0.35 * color_sim + 0.65 * image_sim
                            pair_match_parts.append({"mode": "same", "score": pair_same_score})
                        if any(m in diff_color_flags for m in match):
                            color_diff_score = 1.0 - color_sim
                            pair_diff_score += color_diff_score
                            pair_match_parts.append({"mode": "different_color", "score": color_diff_score})
                        if any(m in diff_visual_flags for m in match):
                            visual_diff_score = 1.0 - image_sim
                            pair_diff_score += visual_diff_score
                            pair_match_parts.append({"mode": "visual_change", "score": visual_diff_score})

                        # If only generic relations exist, pair_match_score can stay 0;
                        # relation rewards below carry the pair decision.
                        pair_sim = pair_same_score + pair_diff_score

                        relation_total = 0.0
                        relation_details = []
                        relation_gate_failed = False
                        relation_gate_penalty = 0.0
                        src_axes = (embedding_scores.get(sidx, {}) or {}).get("relation_axes", {})
                        tgt_axes = (embedding_scores.get(tidx, {}) or {}).get("relation_axes", {})
                        src_axis_details = (embedding_scores.get(sidx, {}) or {}).get("relation_axis_details", {})
                        tgt_axis_details = (embedding_scores.get(tidx, {}) or {}).get("relation_axis_details", {})
                        for rel in relation_specs:
                            axis_name = rel["name"]
                            rel_score = _relation_pair_score(
                                src_axes.get(axis_name), tgt_axes.get(axis_name),
                                rel.get("mode"), rel.get("scale", 4.0),
                            )
                            if rel_score is None:
                                continue
                            rel_weight = float(rel.get("weight", 4.0))
                            min_score = float(rel.get("min_score", 0.0))
                            gate = bool(rel.get("gate", False))
                            gate_ok = rel_score >= min_score if min_score > 0 else True
                            if gate and not gate_ok:
                                relation_gate_failed = True
                                relation_gate_penalty += float(rel.get("gate_penalty", 999.0))
                            relation_total += rel_weight * rel_score
                            relation_details.append({
                                "name": axis_name,
                                "mode": rel.get("mode"),
                                "source": round(float(src_axes.get(axis_name, 0.0)), 4),
                                "target": round(float(tgt_axes.get(axis_name, 0.0)), 4),
                                "source_detail": src_axis_details.get(axis_name, {}),
                                "target_detail": tgt_axis_details.get(axis_name, {}),
                                "score": round(float(rel_score), 4),
                                "min_score": round(float(min_score), 4),
                                "gate": gate,
                                "gate_ok": gate_ok,
                                "weight": round(float(rel_weight), 4),
                            })

                        heuristic_total = s0 + s1 - weight * pair_sim - relation_total + relation_gate_penalty
                        pair_items.append({
                            "source": sidx,
                            "target": tidx,
                            "pair_similarity": pair_sim,
                            "color_similarity": color_sim,
                            "image_similarity": image_sim,
                            "pair_match_parts": pair_match_parts,
                            "relation_total": relation_total,
                            "relation_details": relation_details,
                            "relation_gate_failed": relation_gate_failed,
                            "relation_gate_penalty": relation_gate_penalty,
                            "source_score": s0,
                            "target_score": s1,
                            "heuristic_total": heuristic_total,
                        })

                if not pair_items:
                    continue

                pair_items.sort(key=lambda x: x["heuristic_total"])
                best = pair_items[0]
                pair_ranker_info = {"run": False, "reason": "disabled"}
                pair_vlm_info = {"run": False, "reason": "disabled"}

                try:
                    max_pairs = int(pair_reranker_config.get("max_pairs", pc.get("max_pairs", 80)))
                except (TypeError, ValueError):
                    max_pairs = 80
                max_pairs = max(1, min(200, max_pairs))
                pair_subset = pair_items[:max_pairs]
                pair_docs = []
                pair_images = []
                if frame_pils_for_pairs is not None:
                    for rank, item in enumerate(pair_subset):
                        sidx, tidx = item["source"], item["target"]
                        relation_summary = ""
                        if item.get("relation_details"):
                            relation_summary = " Relations=" + json.dumps(item["relation_details"], ensure_ascii=False)
                        pair_docs.append(
                            f"Pair {rank}: IMAGE 1 frame {sidx} (phase {source_phase}) -> IMAGE 2 frame {tidx} (phase {target_phase}). "
                            f"Base source_score={item['source_score']:.3f}, target_score={item['target_score']:.3f}, "
                            f"color_similarity={item['color_similarity']:.3f}, image_similarity={item['image_similarity']:.3f}, "
                            f"pair_match_score={item['pair_similarity']:.3f}, relation_total={item.get('relation_total', 0.0):.3f}, "
                            f"heuristic_total={item.get('heuristic_total', 0.0):.3f}."
                            f"{relation_summary}"
                        )
                        pair_images.append(_make_pair_pil(
                            frame_pils_for_pairs[sidx], frame_pils_for_pairs[tidx],
                            f"Bild 1 / Phase {source_phase}", f"Bild 2 / Phase {target_phase}",
                        ))

                pair_query = str(pair_reranker_config.get("query") or pc.get("query") or "").strip()
                if not pair_query:
                    pair_query = (
                        "Choose the best BEFORE/AFTER pair. IMAGE 1 is before the drop and must satisfy source-phase requirements. "
                        "IMAGE 2 is after the drop and must satisfy target-phase requirements. Prefer the requested cross-phase relation: "
                        "same attributes when the config asks for similarity, different attributes when it asks for visual change, "
                        "and semantic increases/decreases when relation axes are provided. Reject cropped, mismatched, or low-quality pairs."
                    )

                pair_reranker_enabled = bool(pair_reranker_config.get("enabled", False))
                if pair_reranker_enabled and pair_images:
                    mode = str(pair_reranker_config.get("mode", reranker_mode)).strip() or reranker_mode
                    pair_rr = None
                    if mode == "transformers" and reranker_model_path:
                        pair_rr = self._run_reranker_transformers(
                            query=pair_query,
                            documents=pair_docs,
                            document_images=pair_images,
                            top_n=len(pair_docs),
                            model_path=str(pair_reranker_config.get("model_path") or reranker_model_path),
                            device=str(pair_reranker_config.get("device") or reranker_device),
                            dtype_str=str(pair_reranker_config.get("dtype") or reranker_dtype),
                            preloaded_model=reranker_model,
                        )
                    if pair_rr:
                        best_doc_idx, best_score = pair_rr[0]
                        rr_scores = {int(i): float(s) for i, s in pair_rr if 0 <= int(i) < len(pair_subset)}
                        vals = list(rr_scores.values())
                        rr_min = min(vals) if vals else 0.0
                        rr_max = max(vals) if vals else 1.0
                        rr_span = max(1e-6, rr_max - rr_min)
                        try:
                            rr_weight = float(pair_reranker_config.get("blend_weight", pair_reranker_config.get("weight", pc.get("reranker_weight", 2.0))))
                        except (TypeError, ValueError):
                            rr_weight = 2.0
                        for doc_idx, item in enumerate(pair_subset):
                            raw_rr = rr_scores.get(doc_idx)
                            rr_norm = ((raw_rr - rr_min) / rr_span) if raw_rr is not None else 0.0
                            item["pair_reranker_score"] = raw_rr
                            item["pair_reranker_norm"] = rr_norm
                            item["final_pair_score"] = float(item["heuristic_total"]) - rr_weight * rr_norm
                        best_visual = pair_subset[int(best_doc_idx)] if 0 <= int(best_doc_idx) < len(pair_subset) else best
                        best = min(pair_subset, key=lambda x: x.get("final_pair_score", x["heuristic_total"]))
                        pair_ranker_info = {
                            "run": True,
                            "mode": mode,
                            "selection_mode": "blend_with_heuristic",
                            "blend_weight": round(float(rr_weight), 4),
                            "pairs_scored": len(pair_docs),
                            "top_score": round(float(best_score), 4),
                            "top_doc_index": int(best_doc_idx),
                            "top_visual_pair": [int(best_visual["source"]), int(best_visual["target"])],
                            "selected_doc_index": int(pair_subset.index(best)) if best in pair_subset else None,
                            "selected_reranker_score": round(float(best.get("pair_reranker_score", 0.0)), 4),
                            "selected_reranker_norm": round(float(best.get("pair_reranker_norm", 0.0)), 4),
                            "selected_final_pair_score": round(float(best.get("final_pair_score", best["heuristic_total"])), 4),
                        }
                    else:
                        pair_ranker_info = {"run": True, "mode": mode, "error": "pair reranker returned no scores", "pairs_scored": len(pair_docs)}

                pair_vlm_config = pc.get("pair_vlm") or pair_reranker_config.get("pair_vlm") or pair_reranker_config.get("vlm_judge") or {}
                if isinstance(pair_vlm_config, dict) and bool(pair_vlm_config.get("enabled", False)):
                    endpoint = str(pair_vlm_config.get("endpoint") or vlm_endpoint or "").strip()
                    model_name = str(pair_vlm_config.get("model") or vlm_model or "").strip()
                    if not endpoint or not model_name:
                        pair_vlm_info = {
                            "run": False,
                            "enabled": True,
                            "reason": "missing endpoint or model",
                        }
                    elif not pair_images:
                        pair_vlm_info = {"run": False, "enabled": True, "reason": "no pair images"}
                    else:
                        try:
                            try:
                                pair_vlm_max_pairs = int(pair_vlm_config.get("max_pairs", 12))
                            except (TypeError, ValueError):
                                pair_vlm_max_pairs = 12
                            pair_vlm_max_pairs = max(1, min(len(pair_images), min(24, pair_vlm_max_pairs)))
                            ordered_for_vlm = sorted(
                                pair_subset,
                                key=lambda x: x.get("final_pair_score", x["heuristic_total"]),
                            )[:pair_vlm_max_pairs]
                            vlm_images = []
                            vlm_summaries = []
                            for rank, item in enumerate(ordered_for_vlm):
                                sidx_v, tidx_v = item["source"], item["target"]
                                vlm_images.append(_make_pair_pil(
                                    frame_pils_for_pairs[sidx_v], frame_pils_for_pairs[tidx_v],
                                    f"Bild 1 / Phase {source_phase}", f"Bild 2 / Phase {target_phase}",
                                ))
                                final_for_summary = item.get("final_pair_score")
                                if final_for_summary is None:
                                    final_for_summary = item.get("heuristic_total", 0.0)
                                vlm_summaries.append(
                                    f"Pair {rank + 1}: frames [{sidx_v}, {tidx_v}], "
                                    f"final_score={float(final_for_summary):.3f}, "
                                    f"heuristic={float(item.get('heuristic_total', 0.0)):.3f}, relation_total={item.get('relation_total', 0.0):.3f}, "
                                    f"relation_details={json.dumps(item.get('relation_details', []), ensure_ascii=False)}"
                                )
                            pair_headers = {"Content-Type": "application/json"}
                            token = str(pair_vlm_config.get("api_token") or vlm_api_token or "").strip()
                            if token:
                                pair_headers["Authorization"] = f"Bearer {token}"
                            pair_vlm_info = self._vlm_judge_pairs(
                                endpoint=endpoint,
                                headers=pair_headers,
                                model=model_name,
                                system_prompt=str(pair_vlm_config.get("system_prompt") or vlm_system_prompt or ""),
                                pair_images=vlm_images,
                                pair_summaries=vlm_summaries,
                                query=str(pair_vlm_config.get("query") or pair_query),
                                max_tokens=int(pair_vlm_config.get("max_tokens", vlm_max_tokens)),
                                temperature=float(pair_vlm_config.get("temperature", vlm_temperature)),
                                timeout=int(pair_vlm_config.get("timeout", vlm_timeout)),
                                grid_columns=int(pair_vlm_config.get("grid_columns", 2)),
                            )
                            doc_idx = pair_vlm_info.get("selected_doc_index")
                            if isinstance(doc_idx, int) and 0 <= doc_idx < len(ordered_for_vlm):
                                best = ordered_for_vlm[doc_idx]
                                best["pair_vlm_selected"] = True
                                pair_vlm_info["selection_mode"] = "final_visual_judge"
                                pair_vlm_info["selected_pair"] = [int(best["source"]), int(best["target"])]
                            else:
                                pair_vlm_info.setdefault("selection_mode", "no_valid_selection")
                        except Exception as e:
                            pair_vlm_info = {"run": True, "error": str(e)}

                sidx, tidx = best["source"], best["target"]
                forced_selected_by_window[source_phase] = [sidx]
                forced_selected_by_window[target_phase] = [tidx]
                pair_constraint_debug.append({
                    "source_phase": source_phase,
                    "target_phase": target_phase,
                    "match": match,
                    "weight": weight,
                    "selected_pair": [sidx, tidx],
                    "pair_similarity": round(float(best["pair_similarity"]), 4),
                    "color_similarity": round(float(best["color_similarity"]), 4),
                    "image_embedding_similarity": round(float(best["image_similarity"]), 4),
                    "pair_match_parts": best.get("pair_match_parts", []),
                    "relation_total": round(float(best.get("relation_total", 0.0)), 4),
                    "relation_details": best.get("relation_details", []),
                    "relation_gate_failed": bool(best.get("relation_gate_failed", False)),
                    "relation_gate_penalty": round(float(best.get("relation_gate_penalty", 0.0)), 4),
                    "pair_reranker_score": None if best.get("pair_reranker_score") is None else round(float(best.get("pair_reranker_score", 0.0)), 4),
                    "pair_reranker_norm": round(float(best.get("pair_reranker_norm", 0.0)), 4),
                    "final_pair_score": round(float(best.get("final_pair_score", best["heuristic_total"])), 4),
                    "base_scores": [round(float(best["source_score"]), 4), round(float(best["target_score"]), 4)],
                    "heuristic_total": round(float(best["heuristic_total"]), 4),
                    "pair_reranker": pair_ranker_info,
                    "pair_vlm": pair_vlm_info,
                })

        # ── Run selection per window ──
        all_selected = []
        window_results = []

        for wi, win in enumerate(windows):
            indices = win["frame_indices"]
            if not indices:
                continue

            # Filter by reranker top-k if available
            valid_indices = indices
            if top_k_set:
                valid_indices = [i for i in indices if i in top_k_set]
                if not valid_indices:
                    valid_indices = indices  # fallback: all

            # Score all valid frames
            score_fn = _score_frame
            hist_pen = float(history_penalty)
            scored = []
            for fidx in valid_indices:
                s = score_fn(fidx, extra_penalties, all_selected, n_per_window, hist_pen)
                scored.append((fidx, s))

            scored.sort(key=lambda x: x[1])
            w_n = min(n_per_window, len(scored))
            if wi in forced_selected_by_window:
                w_sel = [f for f in forced_selected_by_window[wi] if f in indices]
                if not w_sel:
                    w_sel = [fidx for fidx, _ in scored[:w_n]]
            else:
                w_sel = [fidx for fidx, _ in scored[:w_n]]

            # ── VLM Fallback for this window (if needed) ──
            if needs_vlm and wi not in forced_selected_by_window:
                try:
                    # Build Top-K candidate list from scored frames (NOT all window frames!)
                    # The LLM should only see the best candidates, with scores as context
                    top_k_for_vlm = min(len(scored), max(w_n * 3, 12))  # show top 12-20
                    vlm_candidates = scored[:top_k_for_vlm]

                    # Collect frames + scores for the LLM
                    vlm_frame_indices = []  # absolute frame indices
                    score_lines = []
                    for rank, (fidx, score) in enumerate(vlm_candidates):
                        vlm_frame_indices.append(fidx)
                        parts = [f"  Frame {rank + 1} (global idx {fidx}): penalty={score:.1f}"]
                        if fidx in reranker_scores:
                            parts.append(f"reranker={reranker_scores[fidx]:.1f}")
                        if embedding_scores and fidx in embedding_scores:
                            es = embedding_scores[fidx]
                            parts.append(f"scene_fit={es['scene_fit']:.3f}")
                            parts.append(f"change={es['change_strength']:.3f}")
                        score_lines.append(", ".join(parts))

                    # Build contact sheet from ONLY the top-K frames
                    vlm_frames = reference_frames[torch.tensor(vlm_frame_indices, dtype=torch.long)]
                    vlm_pils = _image_batch_to_pil_images(vlm_frames)
                    labels = [str(i + 1) for i in range(len(vlm_pils))]
                    cs_sheet = _build_contact_sheet(
                        vlm_pils, grid_columns, labels if add_id_labels else None,
                    )

                    # Enrich the prompt with scores as guidance
                    score_context = (
                        f"Top candidates (Re-Ranker + History scored):\n" +
                        "\n".join(score_lines) +
                        "\n\nUse these scores as guidance but make your own visual judgment."
                    )

                    vlm_headers = {"Content-Type": "application/json"}
                    if vlm_api_token:
                        vlm_headers["Authorization"] = f"Bearer {vlm_api_token}"

                    raw = self._vlm_judge_window(
                        vlm_endpoint, vlm_headers, vlm_model, vlm_system_prompt,
                        cs_sheet,
                        {
                            "window_frames": len(vlm_pils),
                            "max_frames": w_n,
                            "num_outfits": num_outfits,
                            "extra_instructions": extra_instructions,
                            "_score_context": score_context,  # injected into prompt
                        },
                        vlm_max_tokens, vlm_temperature, vlm_timeout,
                    )
                    vlm_raw_responses.append(raw)
                    vlm_fallback_used = True
                    stage_used = "vlm_fallback"

                    try:
                        parsed = json.loads(re.sub(r'```.*', '', raw).strip())
                        llm_ids = parsed.get("selected_ids", [])
                        if llm_ids:
                            # Map 1-based contact-sheet indices → absolute frame indices
                            llm_sel = [vlm_frame_indices[i - 1] for i in llm_ids
                                       if 1 <= i <= len(vlm_frame_indices)]
                            if llm_sel:
                                w_sel = llm_sel
                    except Exception:
                        pass
                except Exception as e:
                    print(f"[EmbeddingMatcher] Stage 3 (VLM) FAILED for window {wi}: {e}")

            all_selected.extend(w_sel)

            def _score_debug_item(fidx, s):
                item = {"frame": fidx, "penalty": round(s, 2)}
                es = embedding_scores.get(fidx, {}) if embedding_scores else {}
                if es:
                    for key in (
                        "scene_fit", "change_strength", "composite", "phase_text_fit",
                        "raw_phase_text_fit", "phase_must_fit", "phase_avoid_fit",
                        "phase_contrast_margin", "phase_contrast_mode", "phase",
                        "phase_query", "semantic_filter_failed",
                    ):
                        if key in es:
                            item[key] = es[key]
                    if "relation_axes" in es:
                        item["relation_axes"] = es["relation_axes"]
                    if "relation_axis_details" in es:
                        item["relation_axis_details"] = es["relation_axis_details"]
                if fidx in reranker_scores:
                    item["reranker_score"] = round(float(reranker_scores[fidx]), 4)
                if fidx in w_sel:
                    item["selected"] = True
                return item

            debug_scored = list(scored[:max(w_n, min(10, len(scored)))])
            debug_seen = {f for f, _ in debug_scored}
            score_lookup = {f: s for f, s in scored}
            for f in w_sel:
                if f not in debug_seen and f in score_lookup:
                    debug_scored.append((f, score_lookup[f]))
                    debug_seen.add(f)

            wr = {
                "drop_index": wi,
                "beat_time": win.get("time_seconds"),
                "frame_index": win.get("frame_index"),
                "is_drop": win.get("is_drop", False),
                "range_start": win.get("range_start"),
                "range_end": win.get("range_end"),
                "batch_start": win["batch_start"],
                "batch_end": win["batch_end"],
                "window_frames": len(indices),
                "num_outfits": num_outfits,
                "selected_count": len(w_sel),
                "selected": w_sel,
                "scores": [_score_debug_item(fidx, s) for fidx, s in debug_scored],
            }
            window_results.append(wr)

        # ── Record history ──
        for fidx in all_selected:
            history.setdefault("selections", []).append({
                "key": self._history_key(fidx),
                "frame": int(fidx),
            })
        history["total_selections"] = len(history.get("selections", []))
        self._save_history(history_file, history, history_max_entries)

        # ── Contact sheet ──
        contact_sheet = _make_blank_image()
        if all_selected:
            try:
                sel_tensor = reference_frames[torch.tensor(all_selected, dtype=torch.long)]
                sel_pils = _image_batch_to_pil_images(sel_tensor)
                labels = [f"F{fidx}" for fidx in all_selected]
                cs_pil = _build_contact_sheet(sel_pils, grid_columns, labels if add_id_labels else None)
                arr = np.array(cs_pil.convert("RGB"), dtype=np.float32) / 255.0
                contact_sheet = torch.from_numpy(arr).unsqueeze(0)
            except Exception:
                pass

        # ── Metadata ──
        meta = json.dumps({
            "mode": "embedding_3stage",
            "total_frames": B,
            "windows_count": len(windows),
            "selected_total": len(all_selected),
            "num_outfits": num_outfits,
            "num_outfits_mode": str(num_outfits_mode),
            "stage_used": stage_used,
            "stages": {
                "stage1_embedding": stage1_info,
                "stage2_reranker": stage2_info,
                "stage3_vlm_fallback": {
                    "used": vlm_fallback_used,
                    "responses": len(vlm_raw_responses),
                } if use_vlm_fallback else {"enabled": False},
            },
            "confidence": {
                "embedding": round(embedding_conf, 4),
                "reranker": round(reranker_conf, 4),
                "thresholds": {
                    "embedding_to_reranker": embedding_confidence_threshold,
                    "reranker_to_vlm": reranker_confidence_threshold,
                },
            },
            "weights": {
                "scene_fit": embedding_scene_fit_weight,
                "change_strength": embedding_change_strength_weight,
                "diversity": embedding_diversity_weight,
                "reranker_blend": round(blend_w, 2),
            },
            "history_penalty": float(history_penalty),
            "history_decay_rate": float(history_decay_rate),
            "reranker_used": bool(reranker_scores),
            "reranker_frames_scored": len(reranker_scores),
            "folder_info": {
                "source": folder_info.get("source"),
                "images_loaded": folder_info.get("images_loaded"),
                "max_candidate_images": folder_info.get("max_candidate_images"),
                "max_per_phase": folder_info.get("max_per_phase"),
                "embedding_cache": folder_info.get("embedding_cache"),
            } if isinstance(folder_info, dict) else None,
            "pair_constraints": pair_constraint_debug,
            "window_results": window_results,
        }, indent=2)

        return (
            "\n".join(str(i) for i in all_selected),
            len(all_selected),
            meta,
            contact_sheet,
            "\n---\n".join(vlm_raw_responses) if vlm_raw_responses else "",
            self._build_ai_stack_context(
                conversation_id=conversation_id,
                stage_used=stage_used,
                all_selected=all_selected,
                window_results=window_results,
                extra_instructions=extra_instructions,
                reference_frames=reference_frames,
                contact_sheet=contact_sheet,
                candidate_folders=candidate_folders,
                embedding_scores=embedding_scores,
                reranker_scores=reranker_scores,
                vlm_raw_responses=vlm_raw_responses,
                folder_info=folder_info,
            ),
        )

    def _build_ai_stack_context(self, conversation_id, stage_used, all_selected,
                                  window_results, extra_instructions, reference_frames,
                                  contact_sheet, candidate_folders, embedding_scores,
                                  reranker_scores, vlm_raw_responses, folder_info):
        """Build structured context for AlphaRavis AI Stack integration.

        Includes per-phase decisions, rejected candidates, scores, and HTTP-accessible
        file paths for selected images. The AI Stack can POST this to LangGraph
        for memory persistence and next-iteration context.
        """
        import os as _os
        import base64 as _b64
        from io import BytesIO as _BytesIO

        context = {
            "schema_version": "1.0",
            "node": "BeatDropSelectorEmbeddingNode",
            "thread_id": str(conversation_id or "").strip(),
            "stage_used": stage_used,
            "timestamp": __import__("time").time(),
        }

        # ── Per-phase decisions ──
        phase_decisions = []
        for wr in window_results:
            phase = wr.get("drop_index", 0)
            sel_frames = wr.get("selected", [])
            scores = wr.get("scores", [])

            # Rejected = scored but NOT selected
            scored_frames = {s["frame"] for s in scores}
            rejected = [f for f in scored_frames if f not in sel_frames]

            decision = {
                "phase": phase,
                "beat_time": wr.get("beat_time"),
                "is_drop": wr.get("is_drop", False),
                "num_outfits": wr.get("num_outfits", 2),
                "selected_frames": sel_frames,
                "selected_count": len(sel_frames),
                "rejected_frames": rejected,
                "top_scores": [
                    {"frame": s["frame"], "penalty": s.get("penalty", 0),
                     "selected": s.get("selected", False),
                     "vlm_promoted": s.get("vlm_promoted", False),
                     "vlm_demoted": s.get("vlm_demoted", False)}
                    for s in scores[:10]
                ],
                "vlm_overrides": wr.get("vlm_overrides"),
            }
            phase_decisions.append(decision)

        context["phase_decisions"] = phase_decisions
        context["total_selected"] = len(all_selected)

        # ── Embedding/Reranker top scores ──
        if embedding_scores:
            top_emb = sorted(
                [{"frame": k, **v} for k, v in embedding_scores.items()],
                key=lambda x: x["composite"], reverse=True,
            )[:10]
            context["embedding_top10"] = top_emb

        context["reranker_frames_scored"] = len(reranker_scores)

        # ── Save selected images for HTTP download ──
        saved_files = []
        # Use ComfyUI output directory → accessible via /view API
        try:
            import folder_paths
            output_base = Path(folder_paths.get_output_directory())
        except Exception:
            output_base = Path(candidate_folders).expanduser() if candidate_folders else Path("/tmp")

        safe_thread_component = _safe_path_component(conversation_id, "default") if conversation_id and str(conversation_id).strip() else None
        if safe_thread_component:
            sel_dir = _safe_child_dir(output_base, "_beatdrop_selections", safe_thread_component)
            sel_subfolder = f"_beatdrop_selections/{safe_thread_component}"
        else:
            sel_dir = _safe_child_dir(output_base, "_beatdrop_selections")
            sel_subfolder = "_beatdrop_selections"
        sel_dir.mkdir(parents=True, exist_ok=True)

        # Save contact sheet
        if contact_sheet is not None and contact_sheet.shape[0] > 0:
            cs_path = sel_dir / "contact_sheet.png"
            try:
                arr = (contact_sheet[0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
                Image.fromarray(arr).save(str(cs_path))
                saved_files.append({
                    "type": "contact_sheet",
                    "path": str(cs_path),
                    "filename": cs_path.name,
                })
            except Exception:
                pass

        # Save individual selected frames
        if reference_frames is not None and all_selected:
            for fidx in all_selected:
                if 0 <= fidx < reference_frames.shape[0]:
                    frame_path = sel_dir / f"frame_{fidx:04d}.png"
                    try:
                        arr = (reference_frames[fidx].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
                        Image.fromarray(arr).save(str(frame_path))
                        saved_files.append({
                            "type": "selected_frame",
                            "frame_index": fidx,
                            "path": str(frame_path),
                            "filename": frame_path.name,
                        })
                    except Exception:
                        pass

        context["saved_files"] = saved_files
        context["selections_dir"] = str(sel_dir)

        # ── Folder assignment info ──
        if folder_info:
            context["folder_source"] = {
                "root": folder_info.get("root", ""),
                "assignment_method": folder_info.get("assignment_method", "unknown"),
                "assignments": folder_info.get("folder_assignments", []),
                "images_loaded": folder_info.get("images_loaded", 0),
                "cache_stats": folder_info.get("embedding_cache", {}).get("stats"),
            }

        # ── Instructions echo (for AI Stack to know what was asked) ──
        if extra_instructions and extra_instructions.strip():
            context["instructions"] = extra_instructions.strip()

        # ── VLM responses (if any) ──
        if vlm_raw_responses:
            context["vlm_responses"] = vlm_raw_responses

        result_json = json.dumps(context, indent=2, ensure_ascii=False)

        # ── Save plan JSON alongside images (PlanWriter pattern) ──
        plan_path = sel_dir / "drop_plan.json"
        try:
            plan_path.write_text(result_json, encoding="utf-8")
            saved_files.append({
                "type": "drop_plan",
                "path": str(plan_path),
                "filename": plan_path.name,
            })
        except Exception:
            pass

        return result_json

    # ═══════════════════════════════════════════════════════════════════
    # Model unloading + direct reranker
    # ═══════════════════════════════════════════════════════════════════

    def _unload_embedding_model(self, model_path, device, dtype_str):
        """Free VRAM by removing embedding model from cache."""
        global _QWEN_EMBEDDING_CACHE
        cache_key = (model_path, device, dtype_str)
        if cache_key in _QWEN_EMBEDDING_CACHE:
            model, processor, dim = _QWEN_EMBEDDING_CACHE.pop(cache_key)
            del model, processor
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[EmbeddingMatcher] Unloaded embedding model (freed VRAM)")

    def _run_reranker_transformers(self, query, documents, document_images, top_n, model_path,
                                     device, dtype_str, preloaded_model=None):
        """Run Qwen3-VL-Reranker directly via transformers (no API server).

        Loads the model, scores all documents against the query,
        then unloads to free VRAM. Returns sorted (index, score) list.
        """
        if not documents:
            return None

        print(f"[EmbeddingMatcher] Loading reranker: {model_path}")
        preloaded_used = False
        allowed_remote = {"Qwen/Qwen3-VL-Reranker-8B"}
        if str(model_path) not in allowed_remote and not Path(str(model_path)).expanduser().exists():
            raise ValueError(
                f"Refusing to load untrusted remote reranker model_path={model_path!r}. "
                "Use the built-in Qwen/Qwen3-VL-Reranker-8B or a local path."
            )
        try:
            # Prefer QwenVLRerankerLoader cache when connected.
            if preloaded_model is not None and preloaded_model in _RERANKER_CACHE:
                cached_model, cached_type = _RERANKER_CACHE[preloaded_model]
                if cached_type == "qwen3vl":
                    reranker = cached_model
                    preloaded_used = True
                else:
                    reranker = None
            else:
                reranker = None

            # Try official Qwen3VLReranker wrapper first
            processor = None
            if reranker is None:
                try:
                    from src.models.qwen3_vl_reranker import Qwen3VLReranker
                    reranker = Qwen3VLReranker(
                        model_name_or_path=model_path,
                        torch_dtype={"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(dtype_str, torch.float16),
                    )
                except ImportError:
                    # Fallback: use transformers directly
                    from transformers import AutoModel, AutoProcessor
                    print("[EmbeddingMatcher] Qwen3VLReranker not found — using AutoModel fallback")
                    reranker = AutoModel.from_pretrained(
                        model_path, trust_remote_code=True,
                        torch_dtype=torch.float16 if dtype_str == "fp16" else torch.float32,
                    )
                    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

            # Resolve device
            if device == "auto":
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if "cuda" in device and not torch.cuda.is_available():
                device = "cpu"

            # Build reranker input. The official Qwen3VLReranker supports
            # multimodal documents ({"text": ..., "image": PIL.Image}). The
            # old implementation passed only numeric text summaries, so the
            # reranker could not see whether a candidate was full-body, close-up,
            # fully clothed, etc. Keep the text metadata, but attach the actual
            # candidate image whenever available.
            doc_payloads = []
            doc_pils = []
            if document_images is not None:
                try:
                    if isinstance(document_images, (list, tuple)):
                        doc_pils = [img.convert("RGB") for img in document_images if hasattr(img, "convert")]
                    else:
                        doc_pils = _image_batch_to_pil_images(document_images)
                except Exception as e:
                    print(f"[EmbeddingMatcher] Reranker image conversion failed: {e}")
                    doc_pils = []

            for i, doc in enumerate(documents):
                item = {"text": doc}
                if i < len(doc_pils):
                    item["image"] = doc_pils[i]
                doc_payloads.append(item)

            inputs = {
                "instruction": (
                    "Score whether the candidate IMAGE satisfies the query for outfit selection. "
                    "Use the visual content first; use document text only as metadata."
                ),
                "query": {"text": query},
                "documents": doc_payloads,
                "fps": 1.0,
                "max_frames": 64,
            }

            if hasattr(reranker, 'process'):
                # Official Qwen3VLReranker API
                scores = reranker.process(inputs)
                if hasattr(scores, 'tolist'):
                    scores = scores.tolist()
                if isinstance(scores, (int, float)):
                    scores = [float(scores)]
            else:
                # Transformers fallback — manual scoring
                # Build text pairs: [query + doc_0, query + doc_1, ...]
                reranker.to(device)
                reranker.eval()
                all_scores = []
                batch_size = 8
                for i in range(0, len(documents), batch_size):
                    batch_docs = documents[i:i + batch_size]
                    # Simple relevance: encode query+doc, use hidden state
                    texts = [f"Query: {query}\nDocument: {doc}" for doc in batch_docs]
                    tok = processor(text=texts, return_tensors="pt", padding=True, truncation=True,
                                    max_length=2048).to(device)
                    with torch.no_grad():
                        out = reranker(**tok)
                    if hasattr(out, "last_hidden_state"):
                        # Score = mean of last hidden state
                        batch_scores = out.last_hidden_state.mean(dim=(1, 2)).cpu().tolist()
                    else:
                        batch_scores = [0.5] * len(batch_docs)
                    all_scores.extend(batch_scores)
                scores = all_scores

            # Build sorted (index, score) list
            indexed = [(i, float(s)) for i, s in enumerate(scores)]
            indexed.sort(key=lambda x: x[1], reverse=True)

            # Unload reranker only if this call loaded it; connected loader cache owns preloaded models.
            if not preloaded_used:
                del reranker
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[EmbeddingMatcher] Reranker done: {len(indexed)} documents scored, unloaded")
            else:
                print(f"[EmbeddingMatcher] Reranker done: {len(indexed)} documents scored, kept preloaded")

            return indexed[:top_n] if top_n > 0 else indexed

        except Exception as e:
            print(f"[EmbeddingMatcher] Direct reranker failed: {e}")
            import traceback
            traceback.print_exc()
            return None


# ── Node registration ──────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "BeatDropSelectorEmbeddingNode": BeatDropSelectorEmbeddingNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BeatDropSelectorEmbeddingNode": "🎵 BeatDrop Selector (Embedding)",
}
