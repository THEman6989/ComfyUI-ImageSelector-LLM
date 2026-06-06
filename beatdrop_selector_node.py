"""
BeatDrop Selector Node — window-aware frame selection on top of LLM Image Selector.

Builds on the LLMImageSelectorNode infrastructure (contact sheets, Johnson history,
re-ranker, OpenAI API) and adds beatdrop-specific window grouping.

Place this in ComfyUI-ImageSelector-LLM alongside openai_llm_node.py.
"""
import json
import re
import torch
import numpy as np
from pathlib import Path

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
                "images": ("IMAGE",),
                "beats_used": ("STRING", {"default": "", "multiline": True,
                    "placeholder": "beats_used JSON from FrameSequenceGenerator"}),
                "penalty": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Johnson history penalty shift from Judge feedback"}),
                "extra_penalty_json": ("STRING", {"default": "{}", "multiline": True,
                    "placeholder": "JSON dict {frame_id: penalty_value} from Judge"}),
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

    def _history_penalty_for(self, frame_idx, history, base_penalty):
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
        decay = 1.0 / (1.0 + most_recent * 0.5)
        freq = 1.0 + min(count - 1, 4) * 0.2
        return min(base_penalty * decay * freq, base_penalty * 2.0)

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
        """Call reranker API, return sorted (index, score) list or None."""
        import requests
        urls = self._reranker_urls(endpoint)
        if not urls:
            return None
        payload = self._reranker_payload(query, documents, model, top_n)
        for url in urls:
            try:
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

    # ── Main selection logic ───────────────────────────────────────────

    def select(self, max_frames_per_window, num_outfits_mode, num_outfits,
               images=None, beats_used="",
               penalty=0.0, extra_penalty_json="{}",
               use_llm=False, endpoint="", api_token="", model="",
               system_prompt="", max_tokens=512, temperature=0.0,
               timeout=120, grid_columns=4, add_id_labels=True,
               history_file="", history_penalty=10.0, history_max_entries=200,
               reranker_endpoint="", reranker_model="", reranker_blend_weight=0.3,
               reranker_top_k=12, reranker_query="",
               extra_instructions=""):

        if images is None or not isinstance(images, torch.Tensor):
            return ("", 0, '{"error":"no images provided"}', _make_blank_image(), "")

        B = images.shape[0]
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

        # Parse beats_used for window boundaries
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
            # Flat mode: one window = all frames
            windows = [{"batch_start": 0, "batch_end": B,
                        "frame_indices": list(range(B)),
                        "_flat": True}]

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
                "Outfits with strong visual difference from the old outfit, "
                "clear silhouette, distinct color, suitable for a visible beatdrop change."
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
                    hist_pen = self._history_penalty_for(fidx, history, float(history_penalty))
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
                        win_frames = images[win["batch_start"]:win["batch_end"]]
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

        # Record history for selected frames
        for fidx in all_selected:
            history.setdefault("selections", []).append({
                "key": self._history_key(fidx),
                "frame": int(fidx),
            })
        history["total_selections"] = len(history.get("selections", []))
        self._save_history(history_file, history, history_max_entries)

        # ── Build global contact sheet ──
        contact_sheet = _make_blank_image()
        if all_selected:
            try:
                sel_tensor = images[torch.tensor(all_selected, dtype=torch.long)]
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
            "reranker_used": bool(reranker_scores),
            "reranker_blend_weight": round(blend_w, 2),
            "reranker_frames_scored": len(reranker_scores),
            "correction": correction_info,
            "window_results": window_results,
        }, indent=2)

        return (
            "\n".join(str(i) for i in all_selected),
            len(all_selected),
            meta,
            contact_sheet,
            "\n---\n".join(raw_responses) if raw_responses else "",
        )
