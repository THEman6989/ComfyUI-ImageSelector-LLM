"""
BeatDrop Selector Node — window-aware frame selection on top of LLM Image Selector.

Builds on the LLMImageSelectorNode infrastructure (contact sheets, Johnson history,
re-ranker, OpenAI API) and adds beatdrop-specific window grouping.

Place this in ComfyUI-ImageSelector-LLM alongside openai_llm_node.py.
"""
import json
import re
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image

# Reuse helpers from the parent module (relative import) or standalone
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
    # Standalone fallback
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


class BeatDropSelectorNode:
    """Window-aware frame selector for beatdrop/outfit-change pipelines.

    Takes an IMAGE batch from FrameSequenceGenerator and beats_used metadata,
    groups frames per drop window, and selects the best frames within each
    window using Johnson history penalty + optional LLM judging.

    Two modes:
      LOCAL mode:  penalty-based scoring within windows (fast, no API)
      LLM mode:    sends contact sheets to an LLM for visual judging
                   (uses the same API infrastructure as LLMImageSelectorNode)

    Inputs:
      - images: IMAGE batch from FrameSequenceGenerator
      - beats_used: JSON from FrameSequenceGenerator (window boundaries)
      - max_frames_per_window: int — max frames to select from each window
      - penalty: float 0-1 — Johnson history penalty shift
      - extra_penalty_json: JSON dict {frame_id: penalty} from Judge feedback loop

    LLM mode (optional):
      - endpoint: OpenAI-compatible chat/completions URL
      - api_token: Bearer token
      - model: model name
      - system_prompt: LLM system prompt for judging
      - use_llm: BOOLEAN — enable LLM judging

    History (Johnson penalty persistence):
      - history_file: path to JSON file for selection history

    Re-ranker (optional):
      - reranker_endpoint: URL of a re-ranker API (vLLM, SGLang, llama.cpp)
      - reranker_model: model name (auto-detected if empty)
      - reranker_blend_weight: 0.0-1.0 — how much to blend reranker scores

    Outputs:
      - selected_indices: newline-separated frame indices (e.g. "0\\n3\\n21\\n25")
      - count: total selected frames
      - metadata: JSON with window_results + scores
      - contact_sheet: IMAGE grid of all selected frames
      - raw_response: LLM raw response (empty if LOCAL mode)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_frames_per_window": ("INT", {"default": 4, "min": 2, "max": 20,
                    "tooltip": "Max frames per drop window. BeatDropSelector is for 2+ outfits (use LLMImageSelectorNode for single-outfit selection)"}),
                "num_outfits_mode": (["auto_from_beats", "manual"], {"default": "auto_from_beats",
                    "tooltip": "auto_from_beats: derive from beats_used windows (one outfit per window). manual: use num_outfits value"}),
                "num_outfits": ("INT", {"default": 2, "min": 2, "max": 10, "step": 1,
                    "tooltip": "Manual: minimum number of VISIBLY DIFFERENT outfits. Ignored when mode=auto_from_beats"}),
            },
            "optional": {
                "reference_frames": ("IMAGE", {"tooltip": "Video frames from FrameSequenceGenerator — context/reference, NOT outfit candidates"}),
                "context_frames": ("IMAGE", {"tooltip": "Video frames OUTSIDE drop windows (low fps context)"}),
                "beats_used": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "beats_used JSON from FrameSequenceGenerator"}),
                "penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Johnson history penalty shift from Judge feedback"}),
                "extra_penalty_json": ("STRING", {"default": "{}", "multiline": True,
                    "placeholder": "JSON dict {frame_id: penalty_value} from Judge"}),
                # Frame management
                "max_total_frames": ("INT", {"default": 100, "min": 5, "max": 500, "step": 5,
                    "tooltip": "Max total frames before LLM. Exceeding triggers downsampling."}),
                "job_fps": ("FLOAT", {"default": 5.0, "min": 0.5, "max": 60.0, "step": 0.5,
                    "tooltip": "Frames per second INSIDE drop windows"}),
                "context_fps": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 30.0, "step": 0.1,
                    "tooltip": "Frames per second OUTSIDE drop windows (lower = sparser context)"}),
                "downsample_mode": (["global", "per_job"], {"default": "global",
                    "tooltip": "global: downsample across all frames. per_job: downsample each window separately."}),
                "image_resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64,
                    "tooltip": "Max pixel dimension before sending to LLM/Re-Ranker. Images resized to fit."}),
                # LLM mode
                "use_llm": ("BOOLEAN", {"default": False,
                    "tooltip": "Enable LLM-based visual judging per window"}),
                "endpoint": ("STRING", {"default": "http://127.0.0.1:8080/v1/chat/completions",
                    "multiline": False}),
                "api_token": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"default": "local-model", "multiline": False}),
                "system_prompt": ("STRING", {"default": DEFAULT_SELECTOR_SYSTEM_PROMPT,
                    "multiline": True}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "timeout": ("INT", {"default": 120, "min": 5, "max": 600, "step": 1}),
                "grid_columns": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
                "add_id_labels": ("BOOLEAN", {"default": True}),
                # History
                "history_file": ("STRING", {"default": "", "multiline": False,
                    "placeholder": "/path/to/selection_history.json"}),
                "history_penalty": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 50.0, "step": 0.5}),
                "history_decay_rate": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 2.0, "step": 0.05,
                    "tooltip": "How fast history penalty decays. 0.1=slow (frames stay penalized longer). 1.0=fast (frames cycle quickly)."}),
                "history_max_entries": ("INT", {"default": 200, "min": 10, "max": 10000, "step": 10}),
                # Re-ranker
                "reranker_endpoint": ("STRING", {"default": "", "multiline": False,
                    "placeholder": "http://127.0.0.1:8012 — vLLM/SGLang/llama.cpp reranker"}),
                "reranker_model": ("STRING", {"default": "", "multiline": False,
                    "placeholder": "Auto-detected if empty"}),
                "reranker_top_k": ("INT", {"default": 12, "min": 0, "max": 200, "step": 1,
                    "tooltip": "Pre-filter: only top-K reranker candidates enter scoring loop. 0 = all pass through"}),
                "reranker_query": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "Custom reranker query. Empty = auto: 'Outfits with strong visual difference, clear silhouette, beatdrop-suitable'",
                    "tooltip": "What the reranker should look for in candidates"}),
                "reranker_blend_weight": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much to blend reranker scores (0=ignore, 1=full reranker)"}),
                "extra_instructions": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "Zusaetzliche Anweisungen, z.B. 'Outfits muessen verschiedene Farben haben, mind. 2 komplett verschiedene Styles'",
                    "tooltip": "Supplementary instructions injected into selection prompt"}),
                "conversation_id": ("STRING", {"default": "", "multiline": False,
                    "tooltip": "AlphaRavis thread ID — shared with Judge so folder-LLM and Judge are same context"}),
                # Folder-based candidate loading (alternative to direct IMAGE input)
                "candidate_folders": ("STRING", {"default": "", "multiline": False,
                    "placeholder": "/path/to/outfits/ — root folder with subdirectories (jackets/, no-jacket/, exotic/, chill/)",
                    "tooltip": "Root folder with outfit subdirectories. Vision LLM selects which folders to use."}),
                "max_candidate_images": ("INT", {"default": 30, "min": 5, "max": 500, "step": 5,
                    "tooltip": "Max images to load from selected folders. Pre-filtered by history + random sample."}),
                "use_random_sample": ("BOOLEAN", {"default": True,
                    "tooltip": "Randomly sample from history-filtered pool. OFF = take best by history score only."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("selected_indices", "count", "metadata", "contact_sheet", "raw_response")
    FUNCTION = "select"
    CATEGORY = "Amin/Beatdrop"

    # ── History helpers (same pattern as LLMImageSelectorNode) ──────────

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
        return f"beatdrop_frame_{int(frame_idx)}"

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
        # decay_rate controls speed: 0.1=slow, 1.0=fast
        # decay = 1/(1 + most_recent * decay_rate)
        # After 10 other selections at rate 0.3: decay=1/(1+3)=0.25 (75% penalty gone)
        # After 10 other selections at rate 0.1: decay=1/(1+1)=0.50 (50% penalty gone)
        decay = 1.0 / (1.0 + most_recent * max(0.05, float(decay_rate)))
        freq = 1.0 + min(count - 1, 4) * 0.15  # capped at 1.6x for repeated selection
        return min(base_penalty * decay * freq, base_penalty * 1.5)

    # ── Re-ranker (compact, same logic as LLMImageSelectorNode) ──────

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

        # Auto-detect: probe the endpoint
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

        # Try each URL pattern (vLLM, SGLang, llama.cpp all use /rerank or /v1/rerank)
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

    # ── LLM call ───────────────────────────────────────────────────────

    def _llm_judge_window(self, endpoint, headers, model, system_prompt,
                          contact_sheet_pil, window_info, max_tokens,
                          temperature, timeout):
        """Send a single window's contact sheet to the LLM for judging."""
        import requests
        data_url = _encode_pil_to_data_url(contact_sheet_pil)
        beat_time = window_info.get("beat_time", "?")
        is_drop = window_info.get("is_drop", False)
        frame_count = window_info.get("window_frames", 0)

        prompt_text = (
            f"Beatdrop window at t={beat_time}s (drop={is_drop}). "
            f"This window contains {frame_count} frames. "
            f"Each frame is labeled with its 1-based index in the contact sheet. "
            f"Select the {window_info.get('max_frames', 4)} best frames for outfit analysis. "
            f"You MUST select at least {window_info.get('num_outfits', 2)} VISIBLY DIFFERENT outfits.\n"
        )
        # Inject extra_instructions if provided
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

    # ── Folder-based candidate loading ────────────────────────────────

    def _scan_folders(self, root_path):
        """Scan subdirectories, return [(folder_name, [image_paths]), ...]."""
        import os as _os
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

    def _select_folders_via_llm(self, folders, endpoint, headers, model,
                                 extra_instructions, timeout,
                                 scene_frames=None):
        """Send sample images from each folder + scene context to Vision LLM.
        Returns list of folder names to use, in order.
        scene_frames: video frames as context (what the scene looks like)"""
        import requests, random

        if not folders or not endpoint or not model:
            return [name for name, _ in folders]  # use all

        # Build: show scene context first
        content = [{
            "type": "text",
            "text": (
                "You are selecting outfit categories for a beatdrop video effect.\n\n"
            ),
        }]

        # ── Scene context (video frames) ──
        if scene_frames is not None and isinstance(scene_frames, torch.Tensor) and scene_frames.shape[0] > 0:
            content.append({
                "type": "text",
                "text": "SCENE CONTEXT: This is what the video scene looks like. Choose outfits that fit this scene.",
            })
            # Downsample scene to max 4 frames for the LLM
            sf = scene_frames
            if sf.shape[0] > 4:
                keep = torch.linspace(0, sf.shape[0] - 1, 4).long()
                sf = sf[keep]
            for i in range(sf.shape[0]):
                try:
                    arr = (sf[i].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
                    pil_img = Image.fromarray(arr)
                    data_url = _encode_pil_to_data_url(pil_img)
                    content.append(_image_url_part(data_url))
                except Exception:
                    pass

        # ── Folder samples ──
        content.append({
            "type": "text",
            "text": (
                "Below are sample images from different outfit folders.\n"
                "Decide which folder(s) to use and in what ORDER.\n\n"
                "Consider: the outfits should create a VISIBLE CHANGE at the beatdrop AND fit the scene above.\n"
                "For example: first a jacket outfit, then without jacket = strong change.\n\n"
            ),
        })
        if extra_instructions:
            content.append({
                "type": "text",
                "text": f"SPECIAL INSTRUCTIONS:\n{extra_instructions}\n",
            })

        folder_labels = []
        for folder_name, img_paths in folders:
            samples = random.sample(img_paths, min(3, len(img_paths)))
            for sp in samples:
                try:
                    pil_img = Image.open(sp).convert("RGB")
                    data_url = _encode_pil_to_data_url(pil_img)
                    content.append({
                        "type": "text",
                        "text": f"From folder '{folder_name}':",
                    })
                    content.append(_image_url_part(data_url))
                except Exception:
                    pass
            folder_labels.append(folder_name)

        content.append({
            "type": "text",
            "text": (
                f"Available folders: {json.dumps(folder_labels)}\n\n"
                "Return ONLY valid JSON:\n"
                '{"selected_folders": ["folder_name"], "reason": "..."}\n'
                "List folders in the order they should be used (first at beatdrop, second after, etc.)"
            ),
        })

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a fashion curator for video beatdrop effects. Select outfit categories that create strong visual contrast."},
                {"role": "user", "content": content},
            ],
            "max_tokens": 256,
            "temperature": 0.0,
            "stream": False,
        }
        try:
            resp = requests.post(endpoint, headers=headers, json=payload,
                                timeout=min(max(int(timeout), 1), 60))
            resp.raise_for_status()
            body = resp.json()
            choices = body.get("choices", [])
            text = choices[0].get("message", {}).get("content", "") if choices else ""
            parsed = json.loads(re.sub(r"```.*", "", text).strip())
            selected = parsed.get("selected_folders", [])
            if isinstance(selected, list) and selected:
                # Validate against actual folder names
                valid = [s for s in selected if s in folder_labels]
                return valid if valid else [name for name, _ in folders]
        except Exception:
            pass
        return [name for name, _ in folders]

    def _load_filtered_candidates(self, folders, selected_names, max_images,
                                    history, history_penalty, decay_rate,
                                    use_random):
        """Load ALL images from selected folders, score by history, optionally random-sample.

        All images are loaded and passed to the Re-Ranker. History filtering just
        down-ranks recently-used images; they still participate but with penalty.
        max_images caps the total. use_random=True randomly picks from the top pool.
        """
        import random, numpy as np

        # Collect all image paths from selected folders
        all_paths = []
        for name, paths in folders:
            if name in selected_names:
                all_paths.extend([(name, p) for p in paths])

        if not all_paths:
            return None, [], []

        # Score by history: use the SAME history system as frame indices
        # History key = file stem, tracked in the shared history file
        scored_paths = []
        for folder_name, path in all_paths:
            hist_key = f"folder_{Path(path).stem}"
            hist_pen = self._history_penalty_for(
                hash(hist_key) % 100000, history, float(history_penalty), float(decay_rate),
            )
            scored_paths.append((folder_name, path, hist_pen))

        # Sort by penalty (low = less penalized = preferred, fresh images first)
        scored_paths.sort(key=lambda x: x[2])

        # Cap at max_images
        max_img = min(int(max_images), len(scored_paths))

        if use_random:
            # Random sample from top 2x pool (keeps variety)
            pool_size = min(len(scored_paths), max(int(max_img * 2), max_img))
            pool = scored_paths[:pool_size]
            selected = random.sample(pool, min(max_img, len(pool)))
        else:
            # Take best by score (no randomness — deterministic, freshest images win)
            selected = scored_paths[:max_img]

        # Load images into uniform tensor
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

        # Resize all to match first image
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
        # Also track image stems for history recording
        image_stems = [Path(selected[i][1]).stem for i in range(len(selected))]
        return images_tensor, folder_list, image_stems

    def _load_from_folders(self, root_path, endpoint, api_token, model,
                           history_file, history_penalty, decay_rate,
                           extra_instructions, timeout, max_images,
                           use_random, scene_frames=None,
                           conversation_id=""):
        """Full folder-based loading pipeline: scan → LLM-select → pre-filter → load."""
        import requests
        folders = self._scan_folders(root_path)
        if not folders:
            return None, {"error": "no subdirectories with images found"}

        headers = {"Content-Type": "application/json"}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        if conversation_id:
            headers["x-conversation-id"] = str(conversation_id)
            headers["x-thread-id"] = str(conversation_id)

        history = self._load_history(history_file)
        selected_folders = []

        if endpoint and model:
            selected_folders = self._select_folders_via_llm(
                folders, endpoint, headers, model, extra_instructions, timeout,
                scene_frames=scene_frames,
            )
        if not selected_folders:
            selected_folders = [name for name, _ in folders]

        images, folder_list, image_stems = self._load_filtered_candidates(
            folders, selected_folders, max_images,
            history, history_penalty, decay_rate,
            use_random,
        )

        info = {
            "source": "folders",
            "root": root_path,
            "folders_found": [name for name, _ in folders],
            "folders_selected": selected_folders,
            "images_loaded": images.shape[0] if images is not None else 0,
            "max_candidate_images": int(max_images),
            "use_random_sample": bool(use_random),
            "_folder_list": folder_list if images is not None else [],
            "_image_stems": image_stems if images is not None else [],
        }
        return images, info

    @staticmethod
    def _resize_frame(tensor, max_dim):
        """Resize a single frame tensor (H,W,3) so longest side ≤ max_dim."""
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
        """Downsample frames proportionally by fps ratio when exceeding max_total.

        job_mask: boolean tensor (B,) — True if frame is in a job window
        job_fps / context_fps: ratio determines how many frames to cut from each group

        mode='global': cut from entire batch proportionally
        mode='per_job': cut from each window independently
        """
        import torch as _torch
        B = frames.shape[0]
        if B <= max_total:
            return frames, job_mask

        to_remove = B - max_total
        job_count = int(job_mask.sum().item())
        ctx_count = B - job_count

        if job_count == 0 or ctx_count == 0:
            # Only one type — uniform downsampling
            keep = _torch.linspace(0, B - 1, max_total).long()
            return frames[keep], job_mask[keep]

        # Weighted removal: keep proportionally more from higher-fps group
        job_weight = job_fps / max(job_fps, context_fps)
        ctx_weight = context_fps / max(job_fps, context_fps)
        total_weight = job_weight * job_count + ctx_weight * ctx_count

        job_keep_target = int(job_count * (1.0 - to_remove * job_weight / total_weight))
        ctx_keep_target = int(ctx_count * (1.0 - to_remove * ctx_weight / total_weight))
        job_keep_target = max(1, min(job_keep_target, job_count))
        ctx_keep_target = max(1, min(ctx_keep_target, ctx_count))

        # Build keep indices
        job_indices = _torch.where(job_mask)[0]
        ctx_indices = _torch.where(~job_mask)[0]

        if len(job_indices) > 0:
            job_keep = _torch.linspace(0, len(job_indices) - 1, job_keep_target).long()
            job_keep = job_indices[job_keep]
        else:
            job_keep = _torch.tensor([], dtype=_torch.long)

        if len(ctx_indices) > 0:
            ctx_keep = _torch.linspace(0, len(ctx_indices) - 1, ctx_keep_target).long()
            ctx_keep = ctx_indices[ctx_keep]
        else:
            ctx_keep = _torch.tensor([], dtype=_torch.long)

        keep = _torch.cat([job_keep, ctx_keep]).sort().values
        return frames[keep], job_mask[keep]

    # ── Main selection logic ───────────────────────────────────────────

    def select(self, max_frames_per_window, num_outfits_mode, num_outfits,
               reference_frames=None, context_frames=None, beats_used="",
               penalty=0.0, extra_penalty_json="{}",
               max_total_frames=100, job_fps=5.0, context_fps=1.0,
               downsample_mode="global", image_resolution=512,
               use_llm=False, endpoint="", api_token="", model="",
               system_prompt="", max_tokens=512, temperature=0.0,
               timeout=120, grid_columns=4, add_id_labels=True,
               history_file="", history_penalty=10.0, history_max_entries=200,
               history_decay_rate=0.3,
               reranker_endpoint="", reranker_model="", reranker_blend_weight=0.3,
               reranker_top_k=12, reranker_query="",
               extra_instructions="",
               conversation_id="",
               candidate_folders="", max_candidate_images=30,
               use_random_sample=True):

        folder_info = None  # populated when loading from folders

        if reference_frames is None or not isinstance(reference_frames, torch.Tensor):
            # Try folder-based loading as fallback
            folder_path = str(candidate_folders or "").strip()
            if folder_path:
                images, folder_info = self._load_from_folders(
                    folder_path, endpoint, api_token, model,
                    history_file, history_penalty, history_decay_rate,
                    extra_instructions, timeout, max_candidate_images,
                    use_random_sample,
                    scene_frames=reference_frames,
                    conversation_id=conversation_id,
                )
                if images is None:
                    return ("", 0, json.dumps({"error": "no images in folders", "folders_scanned": folder_path}),
                            _make_blank_image(), "")
            else:
                return ("", 0, json.dumps({"error": "no images provided and no candidate_folders set"}),
                        _make_blank_image(), "")

        B = reference_frames.shape[0]
        max_total = max(5, int(max_total_frames))

        # ── Build job mask for downsampling ──
        # Parse windows first to know which frames are in job windows
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
                        "frame_indices": list(range(B)),
                        "_flat": True}]

        # Build job mask: True for frames inside drop windows
        job_mask = torch.zeros(B, dtype=torch.bool)
        for w in windows:
            job_mask[w["batch_start"]:w["batch_end"]] = True

        # ── Merge context frames (outside windows, low fps) ──
        if context_frames is not None and isinstance(context_frames, torch.Tensor):
            ctx_B = context_frames.shape[0]
            if ctx_B > 0:
                # Context frames are OUTSIDE job windows → job_mask = False
                ctx_mask = torch.zeros(ctx_B, dtype=torch.bool)
                reference_frames = torch.cat([reference_frames, context_frames], dim=0)
                job_mask = torch.cat([job_mask, ctx_mask], dim=0)
                B = reference_frames.shape[0]

        # ── Downsample if exceeding max_total_frames ──
        if B > max_total:
            reference_frames, job_mask = self._downsample_frames(
                reference_frames, max_total, job_mask,
                float(job_fps), float(context_fps), str(downsample_mode),
            )
            B = reference_frames.shape[0]
            # Rebuild windows from downsampled frames
            # (windows need updating since frame indices changed)
            windows = [{"batch_start": 0, "batch_end": B,
                        "frame_indices": list(range(B)),
                        "_flat": True,
                        "_downsampled": True}]

        # ── Resize frames for LLM ──
        max_res = max(64, int(image_resolution))
        resized = []
        for i in range(B):
            resized.append(self._resize_frame(reference_frames[i], max_res))
        reference_frames = torch.stack(resized, dim=0)

        n_per_window = max(1, min(int(max_frames_per_window), B))

        # Parse extra penalties
        extra_penalties = {}
        try:
            extra_penalties = json.loads(extra_penalty_json or "{}")
        except Exception:
            pass
        if not isinstance(extra_penalties, dict):
            extra_penalties = {}

        # ── Judge feedback: internal retry when corrections exist ──
        # If the Judge sent penalties, run selection once without them (discover
        # what would be selected), then again with corrections applied. The second
        # pass returns the corrected result.
        has_judge_corrections = bool(extra_penalties)
        correction_pass = False  # True during the second (corrected) run

        # ── Determine num_outfits: auto_from_beats or manual ──
        if str(num_outfits_mode).strip() == "auto_from_beats":
            # One outfit per drop window — derived from beats_used
            num_outfits = max(2, len(windows))  # min 2 even in auto mode
        else:
            num_outfits = max(2, int(num_outfits))
        num_windows = len(windows)
        # Distribute: at least ceil(num_outfits / num_windows) per window
        min_per_window = max(1, (num_outfits + num_windows - 1) // num_windows)
        n_per_window = max(n_per_window, min_per_window)
        n_per_window = min(n_per_window, B)

        # Load history
        history = self._load_history(history_file)

        # Prepare LLM headers if needed
        llm_headers = {}
        if use_llm:
            llm_headers = {"Content-Type": "application/json"}
            if api_token:
                llm_headers["Authorization"] = f"Bearer {api_token}"

        # ── Re-ranker: score all frames, pre-filter to top-K ──
        reranker_scores = {}
        top_k_set = set()
        top_k = max(0, int(reranker_top_k))
        if reranker_endpoint and reranker_endpoint.strip():
            reranker_headers = {"Content-Type": "application/json"}
            if api_token:
                reranker_headers["Authorization"] = f"Bearer {api_token}"
            query = str(reranker_query or "").strip() or (
                "Find outfits that fit TWO criteria simultaneously:\n"
                "1) SCENE FIT: Does this outfit match the scene lighting, pose, camera angle, vibe?\n"
                "2) CHANGE STRENGTH: Is this outfit VISIBLY DIFFERENT from the old outfit — "
                "different silhouette, cut, shape, style — so the change is immediately noticeable?\n"
                "Outfits that fail EITHER criterion should score low."
            )
            documents = [f"Frame {i}: beatdrop candidate image" for i in range(B)]
            rr = self._run_reranker(
                reranker_endpoint, reranker_headers, reranker_model,
                query, documents, top_n=max(top_k, B), timeout=timeout,
            )
            if rr:
                # Normalize reranker scores to 0-100 range
                rr_scores = [s for _, s in rr]
                r_min, r_max = min(rr_scores), max(rr_scores)
                r_range = r_max - r_min if r_max > r_min else 1.0
                for idx, score in rr:
                    normalized = ((score - r_min) / r_range) * 100.0
                    reranker_scores[idx] = normalized
                # Pre-filter: only top-K candidates proceed to scoring loop
                if top_k > 0 and top_k < len(rr):
                    for idx, _ in rr[:top_k]:
                        top_k_set.add(idx)

        blend_w = max(0.0, min(1.0, float(reranker_blend_weight)))

        # ── Internal retry wrapper: if Judge sent corrections, run twice ──
        # Pass 1: without corrections (discover baseline)
        # Pass 2: with corrections (return corrected result)
        def _run_selection(penalties_to_apply):
            all_sel = []
            win_results = []
            raw_resps = []
            for wi, win in enumerate(windows):
                indices = win["frame_indices"]
                if not indices:
                    continue
                scored = []
                for fidx in indices:
                    if top_k_set and fidx not in top_k_set:
                        continue
                    s = 0.0
                    for key, val in penalties_to_apply.items():
                        if str(fidx) in str(key):
                            s += float(val)
                    hist_pen = self._history_penalty_for(fidx, history, float(history_penalty), float(history_decay_rate))
                    s += hist_pen
                    if fidx in reranker_scores and blend_w > 0:
                        rr_score = reranker_scores[fidx]
                        s = (1.0 - blend_w) * s + blend_w * (100.0 - rr_score)
                    for prev_idx in all_sel:
                        dist = abs(fidx - prev_idx)
                        if dist < n_per_window:
                            s += float(history_penalty) * (1.0 - dist / max(n_per_window, 1))
                    scored.append((fidx, s))
                scored.sort(key=lambda x: x[1])
                w_n = min(n_per_window, len(scored))
                w_sel = [fidx for fidx, _ in scored[:w_n]]
                all_sel.extend(w_sel)
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
                    "extra_instructions": str(extra_instructions or "").strip(),
                    "selected_count": len(w_sel),
                    "selected": w_sel,
                    "scores": [{"frame": fidx, "penalty": round(s, 2)}
                               for fidx, s in scored[:max(w_n, min(10, len(scored)))]],
                }
                # LLM judging (optional)
                if use_llm and endpoint and model:
                    try:
                        win_frames = reference_frames[win["batch_start"]:win["batch_end"]]
                        win_pils = _image_batch_to_pil_images(win_frames)
                        labels = [str(i + 1) for i in range(len(win_pils))]
                        cs_sheet = _build_contact_sheet(win_pils, grid_columns, labels if add_id_labels else None)
                        raw = self._llm_judge_window(
                            endpoint, llm_headers, model, system_prompt,
                            cs_sheet, {**wr, "max_frames": w_n},
                            max_tokens, temperature, timeout,
                        )
                        raw_resps.append(raw)
                        try:
                            parsed = json.loads(re.sub(r'```.*', '', raw).strip())
                            llm_ids = parsed.get("selected_ids", [])
                            if llm_ids:
                                llm_sel = [indices[i - 1] for i in llm_ids if 1 <= i <= len(indices)]
                                if llm_sel:
                                    wr["selected"] = llm_sel
                                    wr["selected_count"] = len(llm_sel)
                                    wr["llm_confidence"] = parsed.get("confidence")
                                    wr["llm_reason"] = parsed.get("reason", "")
                        except Exception:
                            wr["llm_raw"] = raw[:200]
                    except Exception as e:
                        wr["llm_error"] = str(e)[:200]
                win_results.append(wr)
            return all_sel, win_results, raw_resps

        # ── Run selection ──
        if has_judge_corrections:
            # Pass 1: without corrections (just for comparison/delta)
            baseline_sel, baseline_results, _ = _run_selection({})
            # Pass 2: with judge corrections — this is the result we return
            all_selected, window_results, raw_responses = _run_selection(extra_penalties)
            correction_info = {
                "judge_corrections_applied": True,
                "correction_count": len(extra_penalties),
                "baseline_selection": baseline_sel,
                "corrected_selection": all_selected,
                "changed_frames": [f for f in all_selected if f not in baseline_sel],
            }
        else:
            all_selected, window_results, raw_responses = _run_selection(extra_penalties)
            correction_info = {"judge_corrections_applied": False}

        # Record history for selected frames (frame indices + folder images)
        for fidx in all_selected:
            history.setdefault("selections", []).append({
                "key": self._history_key(fidx),
                "frame": int(fidx),
            })
        # Also record folder-based images in shared history
        if folder_info and folder_info.get("source") == "folders":
            stems = folder_info.get("_image_stems", [])
            for fidx in all_selected:
                if 0 <= fidx < len(stems):
                    stem = stems[fidx]
                    # Use SAME key format as _history_key so lookup matches
                    hist_key = self._history_key(hash(f"folder_{stem}") % 100000)
                    history.setdefault("selections", []).append({
                        "key": hist_key,
                        "frame": -1,
                        "folder_stem": stem,
                    })
        history["total_selections"] = len(history.get("selections", []))
        self._save_history(history_file, history, history_max_entries)

        # ── Build global contact sheet ──
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
            "mode": "window_aware" if any(not w.get("_flat") for w in windows) else "flat",
            "total_frames": B,
            "windows_count": len(windows),
            "selected_total": len(all_selected),
            "num_outfits": num_outfits,
            "num_outfits_mode": str(num_outfits_mode),
            "diversity_penalty_enabled": True,
            "use_llm": use_llm,
            "penalty_shift": round(max(0.0, min(1.0, float(penalty))), 2),
            "history_penalty": float(history_penalty),
            "history_decay_rate": float(history_decay_rate),
            "reranker_used": bool(reranker_scores),
            "reranker_blend_weight": round(blend_w, 2),
            "reranker_frames_scored": len(reranker_scores),
            "correction": correction_info,
            "folder_loading": folder_info,
            "window_results": window_results,
        }, indent=2)

        return (
            "\n".join(str(i) for i in all_selected),
            len(all_selected),
            meta,
            contact_sheet,
            "\n---\n".join(raw_responses) if raw_responses else "",
        )
