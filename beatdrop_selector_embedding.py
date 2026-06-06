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


def _make_blank_image(h=64, w=64):
    return torch.zeros(1, h, w, 3)


# ── Model cache ────────────────────────────────────────────────────────

_QWEN_EMBEDDING_CACHE = {}  # (model_path, device, dtype) → (model, processor, dim)


def _get_qwen_embedding_model(model_path, device, dtype_str):
    """Lazy-load and cache Qwen3-VL-Embedding model.

    Supports: fp16, bf16, int8 (bitsandbytes), fp8 (via torchao if available).
    Returns (model, processor, embedding_dim).
    """
    cache_key = (model_path, device, dtype_str)
    if cache_key in _QWEN_EMBEDDING_CACHE:
        return _QWEN_EMBEDDING_CACHE[cache_key]

    print(f"[EmbeddingMatcher] Loading model: {model_path}")
    print(f"[EmbeddingMatcher] Device: {device} | Dtype: {dtype_str}")

    torch_dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }

    load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": False,
    }

    # Handle quantization
    if dtype_str == "int8":
        load_kwargs["load_in_8bit"] = True
        load_kwargs["device_map"] = "auto" if device == "auto" or device == "cuda" else device
    elif dtype_str == "fp8":
        # FP8 via torchao — experimental
        load_kwargs["torch_dtype"] = torch.bfloat16  # load in bf16 first
        # Will quantize to fp8 after loading
        fp8_quantize = True
    elif dtype_str in torch_dtype_map:
        load_kwargs["torch_dtype"] = torch_dtype_map[dtype_str]
        fp8_quantize = False
    else:
        load_kwargs["torch_dtype"] = torch.float16
        fp8_quantize = False

    if "device_map" not in load_kwargs:
        load_kwargs["device_map"] = device if device != "auto" else "auto"

    try:
        from transformers import AutoModel, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        model = AutoModel.from_pretrained(model_path, **load_kwargs)

        # FP8 post-load quantization (if requested)
        if dtype_str == "fp8":
            try:
                from torchao.quantization import quantize_, float8_weight_only
                quantize_(model, float8_weight_only())
                print("[EmbeddingMatcher] FP8 quantization applied via torchao")
            except ImportError:
                print("[EmbeddingMatcher] torchao not available — keeping bf16")

        model.eval()

        # Determine embedding dimension
        # Qwen3-VL-Embedding-8B uses the hidden_size of the base model
        if hasattr(model.config, "hidden_size"):
            dim = model.config.hidden_size
        elif hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
            dim = model.config.text_config.hidden_size
        else:
            dim = 2048  # fallback for 8B models

        _QWEN_EMBEDDING_CACHE[cache_key] = (model, processor, dim)
        print(f"[EmbeddingMatcher] Model loaded (dim={dim})")
        return _QWEN_EMBEDDING_CACHE[cache_key]

    except Exception as e:
        raise RuntimeError(
            f"Failed to load Qwen3-VL-Embedding model from '{model_path}'.\n"
            f"Error: {e}\n\n"
            f"Troubleshooting:\n"
            f"  1. pip install transformers accelerate\n"
            f"  2. For int8: pip install bitsandbytes\n"
            f"  3. Verify model exists: ls ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-Embedding-8B/\n"
            f"  4. Pre-download: huggingface-cli download Qwen/Qwen3-VL-Embedding-8B\n"
        )


def _compute_embeddings_batch(model, processor, images_tensor, device, batch_size=8):
    """Compute Qwen3-VL-Embedding vectors for a batch of images.

    images_tensor: (B, H, W, 3) float tensor [0,1] in ComfyUI format.
    Returns: (B, dim) normalized tensor.
    """
    B = images_tensor.shape[0]
    all_embs = []

    for i in range(0, B, batch_size):
        batch = images_tensor[i : i + batch_size]

        # Convert ComfyUI tensor → PIL images
        arr = (batch.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        pil_images = [Image.fromarray(arr[j]) for j in range(arr.shape[0])]

        # Process via Qwen3-VL processor
        inputs = processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Extract embeddings: mean-pool the last_hidden_state
        if hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state  # (B, seq_len, dim)
            emb = hidden.mean(dim=1)  # mean pooling over sequence
        elif isinstance(outputs, torch.Tensor):
            emb = outputs
            if emb.dim() == 3:
                emb = emb.mean(dim=1)
        else:
            # Try pooler_output or first element
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output
            else:
                raise TypeError(
                    f"Unexpected model output type: {type(outputs)}. "
                    "Expected last_hidden_state, pooler_output, or tensor."
                )

        all_embs.append(emb.cpu())

    embeddings = torch.cat(all_embs, dim=0)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


def _compute_text_embeddings(model, processor, texts, device):
    """Compute Qwen3-VL-Embedding vectors for text queries.

    texts: list of strings.
    Returns: (len(texts), dim) normalized tensor.
    """
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
                "reference_frames": ("IMAGE", {"tooltip": "Video frames from FrameSequenceGenerator"}),
                "context_frames": ("IMAGE", {"tooltip": "Video frames OUTSIDE drop windows (low fps context)"}),
                "beats_used": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "beats_used JSON from FrameSequenceGenerator"}),

                # ── Stage 1: Embedding Model ──
                "embedding_model_path": ("STRING", {
                    "default": "Qwen/Qwen3-VL-Embedding-8B",
                    "multiline": False,
                    "placeholder": "HF model ID or local path",
                    "tooltip": "Qwen3-VL-Embedding model. Leave at default unless using DINOv2 fallback.",
                }),
                "embedding_device": (["auto", "cuda", "cpu"], {
                    "default": "auto",
                    "tooltip": "auto = CUDA if available. Use 'cpu' if VRAM is tight.",
                }),
                "embedding_dtype": (["fp16", "bf16", "int8", "fp8", "fp32"], {
                    "default": "fp16",
                    "tooltip": "fp16 for 3090 (~16GB). int8/fp8 for less VRAM. bf16 for A100/H100.",
                }),
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
                "reranker_endpoint": ("STRING", {
                    "default": "", "multiline": False,
                    "placeholder": "http://127.0.0.1:8012 — vLLM/SGLang/llama.cpp reranker",
                    "tooltip": "Re-Ranker Cross-Encoder endpoint. Stage 2: used when embedding confidence < threshold.",
                }),
                "reranker_model": ("STRING", {
                    "default": "", "multiline": False,
                    "placeholder": "Auto-detected if empty",
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
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("selected_indices", "count", "metadata", "contact_sheet", "raw_response")
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
        return folders

    def _load_filtered_candidates(self, folders, selected_names, max_images,
                                   history, history_penalty, decay_rate, use_random):
        import random

        all_paths = []
        for name, paths in folders:
            if name in selected_names:
                all_paths.extend([(name, p) for p in paths])

        if not all_paths:
            return None, [], []

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
        folder_map = {}
        for idx, (fname, path, penalty) in enumerate(selected):
            try:
                pil_img = Image.open(path).convert("RGB")
                arr = np.array(pil_img, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(arr)
                loaded.append(tensor)
                folder_map[idx] = fname
            except Exception:
                continue

        if not loaded:
            return None, [], []

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
        folder_list = [folder_map.get(i, "unknown") for i in range(len(selected))]
        image_stems = [Path(selected[i][1]).stem for i in range(len(selected))]
        return images_tensor, folder_list, image_stems

    def _load_from_folders(self, root_path, history, history_penalty, decay_rate,
                            max_images, use_random, num_windows=2):
        """Folder loading WITHOUT LLM — all folders included equally."""
        folders = self._scan_folders(root_path)
        if not folders:
            return None, {"error": "no subdirectories with images found"}

        max_per_phase = max(2, int(max_images) // max(1, len(folders)))
        all_images = []
        folder_map_global = {}
        phase_info = []

        for phase, (folder_name, _) in enumerate(folders):
            phase_images, phase_list, phase_stems = self._load_filtered_candidates(
                [(folder_name, [p for _, p in folders if _ == folder_name])],
                [folder_name],
                max_per_phase, history, history_penalty, decay_rate, use_random,
            )
            if phase_images is not None:
                offset = len(all_images)
                all_images.append(phase_images)
                for i, stem in enumerate(phase_stems or []):
                    folder_map_global[offset + i] = folder_name
                phase_info.append({
                    "phase": phase,
                    "folder": folder_name,
                    "images_loaded": phase_images.shape[0],
                })

        if not all_images:
            return None, {"error": "no images loaded from any folder"}

        images_tensor = torch.cat(all_images, dim=0)

        info = {
            "source": "folders",
            "root": root_path,
            "folders_found": [name for name, _ in folders],
            "phase_info": phase_info,
            "images_loaded": images_tensor.shape[0],
            "max_candidate_images": int(max_images),
            "max_per_phase": max_per_phase,
            "use_random_sample": bool(use_random),
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
                                 text_query_scene, text_query_change):
        """Compute embedding-based scores for all frames.

        Returns:
            embedding_scores: dict {frame_idx: composite_score}
            stage_info: dict with details
            confidence: float (how decisive the scores are)
        """
        B = reference_frames.shape[0]

        # Load model
        model, processor, dim = _get_qwen_embedding_model(model_path, device, dtype_str)

        # Compute frame embeddings
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
            "scene_fit_weight": scene_fit_w,
            "change_strength_weight": change_w,
            "diversity_weight": diversity_w,
            "text_queries_used": list(text_queries.keys()),
            "top_composite_range": [composites[0], composites[-1]] if composites else [0, 0],
            "confidence": round(confidence, 4),
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

    # ═══════════════════════════════════════════════════════════════════
    # MAIN SELECT
    # ═══════════════════════════════════════════════════════════════════

    def select(self, max_frames_per_window, num_outfits_mode, num_outfits,
               reference_frames=None, context_frames=None, beats_used="",
               # Stage 1
               embedding_model_path="Qwen/Qwen3-VL-Embedding-8B",
               embedding_device="auto", embedding_dtype="fp16",
               embedding_confidence_threshold=0.75,
               embedding_scene_fit_weight=0.30,
               embedding_change_strength_weight=0.50,
               embedding_diversity_weight=0.20,
               text_query_scene_fit="",
               text_query_change_target="",
               # Stage 2
               reranker_endpoint="", reranker_model="",
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
               use_random_sample=True):

        # ── Load images ──
        if reference_frames is None or not isinstance(reference_frames, torch.Tensor):
            folder_path = str(candidate_folders or "").strip()
            if folder_path:
                history = self._load_history(history_file)
                try:
                    bu = json.loads(beats_used or "[]")
                    nw = max(2, len(bu) if isinstance(bu, list) else 2)
                except Exception:
                    nw = 2

                images, folder_info = self._load_from_folders(
                    folder_path, history, history_penalty, history_decay_rate,
                    max_candidate_images, use_random_sample, num_windows=nw,
                )
                if images is None:
                    return ("", 0, json.dumps({"error": "no images in folders"}),
                            _make_blank_image(), "")
                reference_frames = images
            else:
                return ("", 0, json.dumps({"error": "no images provided and no candidate_folders set"}),
                        _make_blank_image(), "")

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

        # ═══════════════════════════════════════════════════════════════
        # STAGE 1: Embedding-based scoring
        # ═══════════════════════════════════════════════════════════════
        stage_used = "none"
        embedding_scores = {}
        embedding_conf = 0.0

        if embedding_model_path and embedding_model_path.strip():
            try:
                embedding_scores, stage1_info, embedding_conf = self._stage1_embedding_score(
                    reference_frames, windows,
                    embedding_model_path, embedding_device, embedding_dtype,
                    embedding_scene_fit_weight, embedding_change_strength_weight,
                    embedding_diversity_weight,
                    text_query_scene_fit, text_query_change_target,
                )
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

        # ═══════════════════════════════════════════════════════════════
        # STAGE 2: Re-Ranker Cross-Encoder (if embedding uncertain)
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

        if reranker_will_run and reranker_endpoint and reranker_endpoint.strip():
            reranker_headers = {"Content-Type": "application/json"}
            if vlm_api_token:
                reranker_headers["Authorization"] = f"Bearer {vlm_api_token}"

            # Build query for reranker
            if reranker_query and reranker_query.strip():
                query = reranker_query
            else:
                # Auto-query: blend scene-fit + change-strength
                query = (
                    "Find outfits that fit TWO criteria simultaneously:\n"
                    "1) SCENE FIT: Does this outfit match the scene lighting, pose, camera angle, vibe?\n"
                    "2) CHANGE STRENGTH: Is this outfit VISIBLY DIFFERENT from the old outfit — "
                    "different silhouette, cut, shape, style — so the change is immediately noticeable?\n"
                    "Outfits that fail EITHER criterion should score low."
                )
            if extra_instructions and extra_instructions.strip():
                query += f"\n\nADDITIONAL INSTRUCTIONS: {extra_instructions.strip()}"

            # Documents: one per frame, optionally enriched with embedding info
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
            rr = self._run_reranker(
                reranker_endpoint, reranker_headers, reranker_model,
                query, documents, top_n=max(top_k, B), timeout=vlm_timeout,
            )

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
                emb_score = embedding_scores[fidx]["composite"]
                s -= emb_score * 5.0  # bonus for high embedding score

            # Re-Ranker blend
            if fidx in reranker_scores and blend_w > 0:
                rr_score = reranker_scores[fidx]
                s = (1.0 - blend_w) * s + blend_w * (100.0 - rr_score)

            # Diversity penalty
            for prev_idx in all_selected_so_far:
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
            score_fn = _score_frame if not needs_vlm else _score_frame_vlm
            hist_pen = float(history_penalty)
            scored = []
            for fidx in valid_indices:
                s = score_fn(fidx, extra_penalties, all_selected, n_per_window, hist_pen)
                scored.append((fidx, s))

            scored.sort(key=lambda x: x[1])
            w_n = min(n_per_window, len(scored))
            w_sel = [fidx for fidx, _ in scored[:w_n]]

            # ── VLM Fallback for this window (if needed) ──
            if needs_vlm:
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
                "scores": [{"frame": fidx, "penalty": round(s, 2)}
                           for fidx, s in scored[:max(w_n, min(10, len(scored)))]],
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
            "window_results": window_results,
        }, indent=2)

        return (
            "\n".join(str(i) for i in all_selected),
            len(all_selected),
            meta,
            contact_sheet,
            "\n---\n".join(vlm_raw_responses) if vlm_raw_responses else "",
        )


# ── Node registration ──────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "BeatDropSelectorEmbeddingNode": BeatDropSelectorEmbeddingNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BeatDropSelectorEmbeddingNode": "🎵 BeatDrop Selector (Embedding)",
}
