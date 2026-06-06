"""
AlphaRavisOutfitReferenceJudgeNode — two-stage outfit selection for beatdrop changes.

Pipeline:
  1. Re-Ranker (BeatDropSelectorNode) prefilters Top-K candidates
  2. THIS node takes Top-K, builds contact sheets, sends to Vision LLM
  3. Vision LLM judges scene_fit, change_strength, beatdrop_impact, render_safety
  4. Final decision: best outfit for beatdrop change, not just prettiest

Contact-sheet chunking: max_images_per_call candidates per sheet.
"""

import json
import re
import requests
import torch
from pathlib import Path
from PIL import Image
import numpy as np

# Reuse helpers from openai_llm_node
try:
    from .openai_llm_node import (
        _build_contact_sheet,
        _encode_pil_to_data_url,
        _image_url_part,
    )
except ImportError:
    from openai_llm_node import (
        _build_contact_sheet,
        _encode_pil_to_data_url,
        _image_url_part,
    )

# Local helper (same as in beatdrop_selector_node)
def _make_blank_image(h=64, w=64):
    return torch.zeros(1, h, w, 3, dtype=torch.float32)

# ── Tensor ↔ PIL helpers ──────────────────────────────────────────────

def _tensor_to_pil(tensor):
    """BCHW float tensor [0,1] → list of PIL Images."""
    tensor = tensor.detach().cpu()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    batch = tensor.shape[0]
    arr = (tensor.clamp(0, 1) * 255).to(torch.uint8).numpy()
    return [Image.fromarray(arr[i]) for i in range(batch)]


def _pil_to_tensor(image):
    """PIL Image → CHW float tensor [0,1]."""
    arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


# ── Judge Prompt ───────────────────────────────────────────────────────

BUILTIN_JUDGE_PROMPT = """You are an outfit selection judge for a beatdrop video effect system.

CRITICAL: The OLD OUTFIT reference image shows a DIFFERENT PERSON than the one in the video. You are comparing CLOTHING/OUTFITS, not people. Ignore the person's face, body type, or identity — only compare the clothing items: color, silhouette, style, material, and overall impression.

For EACH candidate outfit, evaluate these dimensions (0.0-1.0):

1. scene_fit_score: How well does this CLOTHING fit the scene lighting, pose, camera angle, and video vibe? The person wearing it in the reference is NOT the video person — judge only the clothes.
2. change_strength_score: How DIFFERENT is this outfit from the old outfit? Consider the overall VIBE — silhouette, cut, shape, style, material. A good beatdrop needs an immediately noticeable change. The eye should see a completely different look. IGNORE the models — compare only what they're wearing.
3. beatdrop_impact_score: Would seeing this outfit change at a beatdrop moment create a strong, noticeable impact?
4. render_safety_score: Is the outfit visually clear, not too chaotic in detail, and likely to render stably with image-to-video models?
5. too_similar_to_old_outfit: true/false — is this CANDIDATE OUTFIT's clothing too similar to the old outfit's clothing to create a meaningful beatdrop?

IMPORTANT: Don't just pick the prettiest outfit or outfit image. Pick the OUTFIT (clothing) that creates the STRONGEST, most VISIBLE beatdrop change while still fitting the scene.

If the user requests TWO different outfits (e.g., one before and one after a beatdrop), select two outfits that are VISIBLY DIFFERENT FROM EACH OTHER in addition to being different from the old outfit.

Return ONLY valid JSON with this exact structure:
{
  "candidates": [
    {
      "index": 0,
      "scene_fit_score": 0.85,
      "change_strength_score": 0.92,
      "beatdrop_impact_score": 0.88,
      "render_safety_score": 0.80,
      "too_similar_to_old_outfit": false,
      "reason": "Strong color contrast, clear silhouette change."
    }
  ],
  "best_index": 0,
  "best_reason": "Candidate 3 has the best combination of change strength and scene fit.",
  "needs_user_review": false,
  "needs_more_candidates": false
}"""


# ── AlphaRavisOutfitReferenceJudgeNode ─────────────────────────────────

class AlphaRavisOutfitReferenceJudgeNode:
    """Two-stage outfit selection: Re-Ranker prefilter → Vision Judge final decision.

    Takes Top-K candidates from a re-ranker, builds contact sheets, sends them
    to a Vision-LLM (AlphaRavis / Qwen3.6 Vision) for semantic judging.

    Scores NOT just "best looking" but best for beatdrop impact: change strength,
    silhouette contrast, render safety, scene fit.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "endpoint": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "http://192.168.x.x:8123/v1/chat/completions — AlphaRavis Vision endpoint",
                }),
                "model": ("STRING", {
                    "default": "my-agent",
                    "multiline": False,
                }),
                "max_images_per_call": ("INT", {
                    "default": 8, "min": 2, "max": 30, "step": 1,
                    "tooltip": "Candidates per contact-sheet chunk",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1,
                }),
                "timeout": ("INT", {
                    "default": 120, "min": 10, "max": 600, "step": 10,
                }),
            },
            "optional": {
                # Images
                "candidate_images": ("IMAGE", {"tooltip": "Top-K candidates from Re-Ranker"}),
                "old_outfit_crop": ("IMAGE", {"tooltip": "Current/old outfit for comparison"}),
                "scene_reference_image": ("IMAGE", {"tooltip": "Scene context frame"}),
                "reference_video_frames": ("IMAGE", {"tooltip": "Additional video frames for context"}),
                # Context JSON
                "reranker_scores_json": ("STRING", {
                    "default": "{}", "multiline": True,
                    "tooltip": "Pre-filter scores from Re-Ranker",
                }),
                "drop_context_json": ("STRING", {
                    "default": "{}", "multiline": True,
                    "tooltip": "Beatdrop context (time, energy, is_drop)",
                }),
                "embedding_change_json": ("STRING", {
                    "default": "{}", "multiline": True,
                    "tooltip": "DINOv2/SigLIP embedding change scores",
                }),
                "mask_quality_json": ("STRING", {
                    "default": "{}", "multiline": True,
                    "tooltip": "Mask quality report",
                }),
                # AlphaRavis bridge
                "conversation_id": ("STRING", {
                    "default": "", "multiline": False,
                }),
                "run_id": ("STRING", {
                    "default": "", "multiline": False,
                }),
                "drop_id": ("STRING", {
                    "default": "", "multiline": False,
                }),
                "api_token": ("STRING", {
                    "default": "", "multiline": False,
                }),
                # Prompt override
                "judge_prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "placeholder": "Custom judge prompt (empty = use builtin)",
                }),
                "extra_instructions": ("STRING", {
                    "default": "", "multiline": True,
                    "placeholder": "Zusätzliche Anweisungen, z.B. 'mindestens 2 verschiedene Outfits, eins vor und eins nach dem Beatdrop'",
                    "tooltip": "Supplementary instructions appended to the judge prompt",
                }),
                "grid_columns": ("INT", {
                    "default": 4, "min": 2, "max": 10, "step": 1,
                }),
                "add_id_labels": ("BOOLEAN", {
                    "default": True,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING", "STRING", "FLOAT", "FLOAT", "FLOAT",
                    "BOOLEAN", "STRING")
    RETURN_NAMES = ("selected_image", "selected_index", "selected_outfit_id",
                    "judge_json", "confidence", "change_strength_score",
                    "beatdrop_impact_score", "too_similar_to_old_outfit",
                    "raw_response")
    FUNCTION = "judge"
    CATEGORY = "Amin/Beatdrop"

    # ── API call ──────────────────────────────────────────────────────

    def _parse_json(self, text):
        """Extract first JSON object from text (handles markdown fences)."""
        text = str(text or "").strip()
        # Try markdown code fences
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
        # Find first { ... }
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def _call_vision_llm(self, endpoint, headers, model, system_prompt,
                         user_content, max_tokens, temperature, timeout):
        """Send vision LLM request, return parsed JSON response."""
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        # AlphaRavis bridge metadata
        conv_id = headers.get("x-conversation-id", "")
        if conv_id:
            payload["conversation_id"] = conv_id
            payload.setdefault("metadata", {})["conversation_id"] = conv_id

        resp = requests.post(
            endpoint, headers=headers, json=payload,
            timeout=min(max(int(timeout), 1), 600),
        )
        resp.raise_for_status()
        body = resp.json()

        # Extract assistant message content
        choices = body.get("choices", [])
        if choices:
            text = choices[0].get("message", {}).get("content", "")
        else:
            text = body.get("content", "") or json.dumps(body)

        return text, body

    # ── Scoring ───────────────────────────────────────────────────────

    def _merge_candidate_scores(self, all_candidates):
        """Merge candidate scores across chunks. If same index appears in
        multiple chunks (shouldn't happen with proper chunking), take max."""
        merged = {}
        for cand in all_candidates:
            idx = cand.get("index", -1)
            if idx < 0:
                continue
            # Keep best scoring entry for duplicate indices
            if idx not in merged or cand.get("beatdrop_impact_score", 0) > merged[idx].get("beatdrop_impact_score", 0):
                merged[idx] = cand
        return merged

    def _compute_weighted_score(self, candidate):
        """Compute composite score: change_strength (40%) + beatdrop_impact (30%)
        + render_safety (20%) + scene_fit (10%). Heavily penalize too_similar."""
        weights = {
            "change_strength_score": 0.40,
            "beatdrop_impact_score": 0.30,
            "render_safety_score": 0.20,
            "scene_fit_score": 0.10,
        }
        score = 0.0
        for key, w in weights.items():
            val = float(candidate.get(key, 0))
            score += val * w
        # Heavy penalty for too_similar
        if candidate.get("too_similar_to_old_outfit", False):
            score *= 0.1
        return score

    def _select_best(self, merged_candidates, all_responses):
        """Select best candidate based on weighted composite score."""
        if not merged_candidates:
            return None, all_responses

        scored = []
        for idx, cand in merged_candidates.items():
            s = self._compute_weighted_score(cand)
            scored.append((idx, cand, s))

        scored.sort(key=lambda x: x[2], reverse=True)

        # Check fallback: best score too low
        best = scored[0]
        if best[2] < 0.3:
            # All candidates weak
            return {
                "selected_outfit_id": None,
                "selected_index": -1,
                "confidence": round(best[2], 2),
                "change_strength_score": 0.0,
                "beatdrop_impact_score": 0.0,
                "too_similar_to_old_outfit": True,
                "needs_user_review": True,
                "reason": "No candidate has sufficient change strength or beatdrop impact.",
            }, all_responses

        cand = best[1]
        return {
            "selected_outfit_id": f"outfit_{best[0]:03d}",
            "selected_index": best[0],
            "confidence": round(best[2], 2),
            "change_strength_score": round(cand.get("change_strength_score", 0), 2),
            "beatdrop_impact_score": round(cand.get("beatdrop_impact_score", 0), 2),
            "too_similar_to_old_outfit": bool(cand.get("too_similar_to_old_outfit", False)),
            "needs_user_review": bool(cand.get("needs_user_review", False)),
            "reason": cand.get("reason", ""),
            "rejected_candidates": [
                {
                    "outfit_id": f"outfit_{idx:03d}",
                    "reason": c.get("reason", "Lower composite score"),
                }
                for idx, c, _ in scored[1:6]
            ],
        }, all_responses

    # ── Main judge ────────────────────────────────────────────────────

    def judge(self, endpoint, model, max_images_per_call, temperature, timeout,
              candidate_images=None, old_outfit_crop=None,
              scene_reference_image=None, reference_video_frames=None,
              reranker_scores_json="{}", drop_context_json="{}",
              embedding_change_json="{}", mask_quality_json="{}",
              conversation_id="", run_id="", drop_id="",
              api_token="", judge_prompt="", extra_instructions="",
              grid_columns=4, add_id_labels=True):

        # ── Validate inputs ──
        if candidate_images is None or not isinstance(candidate_images, torch.Tensor):
            blank = _make_blank_image()
            return (blank, -1, "", '{"error":"no candidate images"}',
                    0.0, 0.0, 0.0, True, "")

        N = candidate_images.shape[0]
        if N == 0:
            blank = _make_blank_image()
            return (blank, -1, "", '{"error":"empty candidate images"}',
                    0.0, 0.0, 0.0, True, "")

        endpoint = str(endpoint or "").strip()
        if not endpoint:
            blank = _make_blank_image()
            return (blank, -1, "", '{"error":"no endpoint configured"}',
                    0.0, 0.0, 0.0, True, "")

        # ── Build headers with AlphaRavis metadata ──
        headers = {"Content-Type": "application/json"}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        if conversation_id:
            headers["x-conversation-id"] = str(conversation_id)
            headers["x-thread-id"] = str(conversation_id)
        if run_id:
            headers["x-run-id"] = str(run_id)

        # ── Build reference images ──
        reference_parts = []

        if old_outfit_crop is not None and isinstance(old_outfit_crop, torch.Tensor):
            old_pil = _tensor_to_pil(old_outfit_crop)
            if old_pil:
                reference_parts.append({"type": "text", "text": "REFERENCE: Old outfit (compare against this):"})
                reference_parts.append(_image_url_part(_encode_pil_to_data_url(old_pil[0])))

        if scene_reference_image is not None and isinstance(scene_reference_image, torch.Tensor):
            scene_pil = _tensor_to_pil(scene_reference_image)
            if scene_pil:
                reference_parts.append({"type": "text", "text": "SCENE CONTEXT: Reference frame from the video:"})
                reference_parts.append(_image_url_part(_encode_pil_to_data_url(scene_pil[0])))

        if reference_video_frames is not None and isinstance(reference_video_frames, torch.Tensor):
            ref_frames_pil = _tensor_to_pil(reference_video_frames)
            if ref_frames_pil:
                ref_sheet = _build_contact_sheet(
                    ref_frames_pil[:8], columns=min(grid_columns, len(ref_frames_pil)),
                    labels=[str(i) for i in range(min(8, len(ref_frames_pil)))],
                )
                reference_parts.append({"type": "text", "text": "ADDITIONAL CONTEXT: Nearby video frames:"})
                reference_parts.append(_image_url_part(_encode_pil_to_data_url(ref_sheet)))

        # ── Build context text ──
        context_parts = []
        try:
            dc = json.loads(drop_context_json or "{}")
            if dc:
                context_parts.append(f"Drop context: time={dc.get('time_seconds','?')}s, "
                                     f"is_drop={dc.get('is_drop',False)}, "
                                     f"energy={dc.get('energy_jump',0):.2f}")
        except json.JSONDecodeError:
            pass

        try:
            ec = json.loads(embedding_change_json or "{}")
            if ec:
                context_parts.append(f"Embedding info: {json.dumps(ec)[:200]}")
        except json.JSONDecodeError:
            pass

        try:
            mq = json.loads(mask_quality_json or "{}")
            if mq:
                context_parts.append(f"Mask quality: {json.dumps(mq)[:200]}")
        except json.JSONDecodeError:
            pass

        context_text = "\n".join(context_parts) if context_parts else ""

        # ── Judge prompt ──
        system_prompt = str(judge_prompt or "").strip() or BUILTIN_JUDGE_PROMPT

        # ── Chunk candidates into contact sheets ──
        chunk_size = max(2, int(max_images_per_call))
        all_candidates_pil = _tensor_to_pil(candidate_images)

        all_responses = []
        all_candidate_scores = []

        for chunk_start in range(0, N, chunk_size):
            chunk_end = min(chunk_start + chunk_size, N)
            chunk_pil = all_candidates_pil[chunk_start:chunk_end]
            chunk_indices = list(range(chunk_start, chunk_end))

            labels = [str(i) for i in chunk_indices] if add_id_labels else None
            sheet = _build_contact_sheet(chunk_pil, columns=min(grid_columns, len(chunk_pil)),
                                         labels=labels)

            # Build user content
            user_text = (
                f"Select the BEST outfit for a beatdrop moment.\n\n"
                f"Candidates {chunk_start}–{chunk_end - 1}.\n"
                f"Each candidate is labeled with its index.\n\n"
                f"IMPORTANT: Prioritize VISIBLE CHANGE from the old outfit, "
                f"not just aesthetic quality. Focus on the CLOTHING, not the person "
                f"wearing it — the person in reference images is different from the video person.\n"
            )
            # Inject extra instructions
            extra = str(extra_instructions or "").strip()
            if extra:
                user_text += f"\nADDITIONAL INSTRUCTIONS:\n{extra}\n"

            user_content = []
            user_content.append({"type": "text", "text": user_text})
            if context_text:
                user_content.append({"type": "text", "text": context_text})
            user_content.extend(reference_parts)
            user_content.append(_image_url_part(_encode_pil_to_data_url(sheet)))

            try:
                raw_text, raw_body = self._call_vision_llm(
                    endpoint=endpoint,
                    headers=headers,
                    model=model,
                    system_prompt=system_prompt,
                    user_content=user_content,
                    max_tokens=2048,
                    temperature=temperature,
                    timeout=timeout,
                )
                all_responses.append({"chunk": f"{chunk_start}-{chunk_end - 1}", "raw": raw_text})

                parsed = self._parse_json(raw_text)
                candidates = parsed.get("candidates", [])
                if isinstance(candidates, list):
                    all_candidate_scores.extend(candidates)

            except Exception as e:
                all_responses.append({"chunk": f"{chunk_start}-{chunk_end - 1}", "error": str(e)})
                continue

        # ── Merge and select best ──
        merged = self._merge_candidate_scores(all_candidate_scores)
        best_result, _ = self._select_best(merged, all_responses)

        # ── Build outputs ──
        selected_idx = best_result.get("selected_index", -1)
        selected_image = _make_blank_image()

        if selected_idx >= 0 and selected_idx < N:
            selected_image = candidate_images[selected_idx].unsqueeze(0)

        # Build metadata for AlphaRavis
        ravis_metadata = {
            "schema_version": 1,
            "node": "AlphaRavisOutfitReferenceJudgeNode",
            "source": "comfyui_researcher",
        }
        if conversation_id:
            ravis_metadata["conversation_id"] = conversation_id
        if run_id:
            ravis_metadata["run_id"] = run_id
        if drop_id:
            ravis_metadata["drop_id"] = drop_id

        judge_json = json.dumps({**best_result, "metadata": ravis_metadata,
                                 "chunks_processed": len(all_responses),
                                 "total_candidates": N}, indent=2)

        return (
            selected_image,
            selected_idx,
            best_result.get("selected_outfit_id", ""),
            judge_json,
            best_result.get("confidence", 0.0),
            best_result.get("change_strength_score", 0.0),
            best_result.get("beatdrop_impact_score", 0.0),
            best_result.get("too_similar_to_old_outfit", True),
            json.dumps({"responses": all_responses, "candidates_scored": len(all_candidate_scores)}, indent=2),
        )
