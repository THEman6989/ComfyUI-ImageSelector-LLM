"""
BeatDrop PlanWriter Pipe — consolidates ALL pipeline outputs into one artifact.

Pure consolidation node. Takes every output from the Beatdrop pipeline and
writes a single drop_plan.json to the ComfyUI output directory.

Wiring (all upstream nodes → this node):
  FrameSequenceGenerator.beats_used       → beats_json
  DINOv2FrameChangeDetector.change_json  → change_detection_json
  MaskQualityFilter.report               → mask_quality_json
  Selector.ai_stack_context              → ai_stack_context  (has thread_id, instructions, selection, saved_files)
  BeatChangeSynchronizer / render planner → render_instructions_json

Place in: ComfyUI-ImageSelector-LLM/beatdrop_plan_writer.py
"""

import json
import re
import uuid
from pathlib import Path


def _safe_path_component(value, fallback="default"):
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


def _first_audio_drop(beats):
    if not isinstance(beats, list) or not beats:
        return None
    return next((beat for beat in beats if beat.get("is_drop")), beats[0])


def _frame_mapping(beat, local_batch_index):
    for frame in beat.get("frames", []) if isinstance(beat, dict) else []:
        if int(frame.get("batch_index", -1)) == int(local_batch_index):
            return frame
    return None


def _build_drop_decision(beats, change_detection, force_audio=False):
    """Choose DINO only for a detected visual outfit change; otherwise use audio."""
    beat = _first_audio_drop(beats)
    if beat is None:
        return None

    change = change_detection if isinstance(change_detection, dict) else {}
    has_visual_change = bool(change.get("has_existing_visual_change")) and not bool(force_audio)
    raw_best_change = change.get("best_change")
    best_change = raw_best_change if isinstance(raw_best_change, dict) else {}
    if has_visual_change and best_change.get("to_frame") is not None:
        local_index = int(best_change["to_frame"])
        mapped = _frame_mapping(beat, local_index)
        return {
            "source": "dinov2_visual_change",
            "dino_used": True,
            "source_frame_index": int(
                mapped.get("source_frame_index", local_index) if mapped else local_index
            ),
            "time_seconds": float(
                mapped.get("time_seconds", beat.get("time_seconds", 0.0))
                if mapped else beat.get("time_seconds", 0.0)
            ),
            "local_batch_index": local_index,
            "needs_generated_outfit_drop": bool(change.get("needs_generated_outfit_drop", False)),
        }

    local_index = int(change.get("beat_frame", beat.get("batch_offset", 0)))
    return {
        "source": "audio_beat",
        "dino_used": False,
        "source_frame_index": int(beat.get("frame_index", 0)),
        "time_seconds": float(beat.get("time_seconds", 0.0)),
        "local_batch_index": local_index,
        "needs_generated_outfit_drop": bool(change.get("needs_generated_outfit_drop", True)),
    }


def _build_beat_decisions(beats):
    """Create one deterministic audio transition for every selected beat window."""
    if not isinstance(beats, list):
        return []
    decisions = []
    for transition_index, beat in enumerate(beat for beat in beats if isinstance(beat, dict)):
        source_frame = int(beat.get("frame_index", 0))
        frames = [frame for frame in beat.get("frames", []) if isinstance(frame, dict)]
        mapped = min(
            frames,
            key=lambda frame: abs(int(frame.get("source_frame_index", source_frame)) - source_frame),
        ) if frames else None
        decisions.append({
            "transition_index": transition_index,
            "outfit_state_before": transition_index,
            "outfit_state_after": transition_index + 1,
            "source": "audio_beat",
            "dino_used": False,
            "source_frame_index": source_frame,
            "time_seconds": float(beat.get("time_seconds", 0.0)),
            "local_batch_index": int(
                mapped.get("batch_index", beat.get("batch_offset", 0))
                if mapped else beat.get("batch_offset", 0)
            ),
            "selection_mode": str(beat.get("selection_mode", "legacy")),
            "relative_to_anchor": beat.get("relative_to_anchor"),
            "anchor_drop_time_seconds": beat.get("anchor_drop_time_seconds"),
            "needs_generated_outfit_drop": True,
        })
    return decisions


def _build_outfit_state_plan(beat_decisions, selection):
    """Map N transitions to N+1 outfit states, cycling a smaller library safely."""
    if not beat_decisions or not isinstance(selection, dict):
        return []
    saved_by_frame = {}
    for item in selection.get("saved_files", []):
        if isinstance(item, dict) and item.get("frame_index") is not None:
            try:
                saved_by_frame[int(item["frame_index"])] = item
            except (TypeError, ValueError):
                pass

    candidates = []
    seen_identities = set()
    for phase in selection.get("phase_decisions", []):
        if not isinstance(phase, dict):
            continue
        for frame in phase.get("selected_frames", []):
            try:
                frame = int(frame)
            except (TypeError, ValueError):
                continue
            saved = saved_by_frame.get(frame, {})
            identity = str(
                saved.get("source_identity")
                or saved.get("source_path")
                or f"frame:{frame}"
            )
            if identity not in seen_identities:
                candidates.append({"frame": frame, "identity": identity, "saved": saved})
                seen_identities.add(identity)
    if not candidates:
        return []
    if len(candidates) < 2:
        raise ValueError(
            "BeatDrop transitions require at least two unique outfit candidates; "
            "duplicate batch copies do not count as distinct outfits."
        )

    state_plan = []
    for state_index in range(len(beat_decisions) + 1):
        candidate = candidates[state_index % len(candidates)]
        candidate_frame = candidate["frame"]
        state = {
            "outfit_state": state_index,
            "candidate_frame": candidate_frame,
            "source_identity": candidate["identity"],
            "reused_candidate": state_index >= len(candidates),
        }
        saved = candidate["saved"]
        if saved:
            state["candidate_path"] = saved.get("path", "")
            state["source_path"] = saved.get("source_path", "")
        state_plan.append(state)
    return state_plan


class BeatDropPlanWriterPipe:
    """Consolidates pipeline outputs → drop_plan.json.

    thread_id, instructions, and selection data are extracted from
    ai_stack_context (the BeatDropSelectorEmbeddingNode output).
    No manual fields needed — everything flows from upstream nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "job_policy": (["main_job_only", "drops_only", "all_frames", "beats_before_drop", "beats_after_drop"], {
                    "default": "drops_only",
                    "tooltip": "Which frames to render.",
                }),
            },
            "optional": {
                # ── Pipeline outputs (all forceInput = must be wired) ──
                "beats_json": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "beats_used from FrameSequenceGenerator.",
                }),
                "change_detection_json": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "change_json from DINOv2FrameChangeDetector.",
                }),
                "mask_quality_json": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "report from MaskQualityFilter.",
                }),
                "ai_stack_context": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "ai_stack_context from BeatDropSelectorEmbeddingNode. Contains thread_id, instructions, selection, saved_files.",
                }),
                "render_instructions_json": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "Render boundaries: first_new_outfit_frame, last_old_outfit_frame, black_frames.",
                }),
                "plan_id": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Plan identifier. Auto-generated if blank.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("plan_path", "plan_json")
    FUNCTION = "write_plan"
    CATEGORY = "Amin/Beatdrop"

    def write_plan(
        self,
        job_policy="drops_only",
        beats_json="",
        change_detection_json="",
        mask_quality_json="",
        ai_stack_context="",
        render_instructions_json="",
        plan_id="",
    ):
        # ── Auto-generate plan_id ──
        if not plan_id or not str(plan_id).strip():
            plan_id = f"drop_{uuid.uuid4().hex[:8]}"

        # ── Extract thread_id + instructions from ai_stack_context ──
        thread_id = ""
        instructions = ""
        selection = {}
        if ai_stack_context and str(ai_stack_context).strip():
            try:
                sel_ctx = json.loads(ai_stack_context)
                thread_id = str(sel_ctx.get("thread_id", "")).strip()
                instructions = str(sel_ctx.get("instructions", "")).strip()
                selection = {
                    "stage_used": sel_ctx.get("stage_used", ""),
                    "phase_decisions": sel_ctx.get("phase_decisions", []),
                    "total_selected": sel_ctx.get("total_selected", 0),
                    "saved_files": sel_ctx.get("saved_files", []),
                    "selections_dir": sel_ctx.get("selections_dir", ""),
                    "folder_source": sel_ctx.get("folder_source", {}),
                    "embedding_top10": sel_ctx.get("embedding_top10", []),
                }
            except (json.JSONDecodeError, TypeError):
                pass

        # ── Build consolidated plan ──
        plan = {
            "schema_version": "2.0",
            "plan_id": plan_id,
            "thread_id": thread_id,
            "timestamp": __import__("time").time(),
            "job_policy": str(job_policy),
        }

        if instructions:
            plan["instructions"] = instructions

        if selection:
            plan["selection"] = selection

        # ── Merge pipeline outputs ──
        parsed_outputs = {}
        for key, source in [
            ("beats", beats_json),
            ("change_detection", change_detection_json),
            ("mask_quality", mask_quality_json),
            ("render_instructions", render_instructions_json),
        ]:
            if source and str(source).strip() not in ("", "{}"):
                try:
                    parsed_outputs[key] = json.loads(source)
                except (json.JSONDecodeError, TypeError):
                    plan[f"{key}_raw"] = str(source)[:10000]

        beats = parsed_outputs.get("beats", [])
        change_detection = parsed_outputs.get("change_detection", {})
        multi_beat_mode = str(job_policy) in {"beats_before_drop", "beats_after_drop"}
        if isinstance(change_detection, dict):
            change_detection["ignored_for_drop_decision"] = multi_beat_mode or not bool(
                change_detection.get("has_existing_visual_change")
            )
        plan.update(parsed_outputs)
        drop_decision = _build_drop_decision(
            beats, change_detection, force_audio=multi_beat_mode,
        )
        if drop_decision is not None:
            plan["drop_decision"] = drop_decision
        beat_decisions = _build_beat_decisions(beats)
        if beat_decisions:
            plan["beat_decisions"] = beat_decisions
            plan["required_outfit_states"] = len(beat_decisions) + 1
            outfit_state_plan = _build_outfit_state_plan(beat_decisions, selection)
            if outfit_state_plan:
                plan["outfit_state_plan"] = outfit_state_plan
                for decision in beat_decisions:
                    before = outfit_state_plan[decision["outfit_state_before"]]
                    after = outfit_state_plan[decision["outfit_state_after"]]
                    decision["outfit_candidate_before"] = before["candidate_frame"]
                    decision["outfit_candidate_after"] = after["candidate_frame"]

        # ── Serialize ──
        plan_json = json.dumps(plan, indent=2, ensure_ascii=False)

        # ── Save to ComfyUI output directory ──
        try:
            import folder_paths
            output_dir = Path(folder_paths.get_output_directory())
        except Exception:
            output_dir = Path("/tmp/comfyui_output")

        plan_id = _safe_path_component(plan_id, "drop_plan")
        safe_thread_id = _safe_path_component(thread_id, "default") if thread_id else ""
        if safe_thread_id:
            plan_dir = _safe_child_dir(output_dir, "_beatdrop_plans", safe_thread_id)
            plan_subfolder = f"_beatdrop_plans/{safe_thread_id}"
        else:
            plan_dir = _safe_child_dir(output_dir, "_beatdrop_plans")
            plan_subfolder = "_beatdrop_plans"
        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_path = plan_dir / f"{plan_id}.json"
        plan_path.write_text(plan_json, encoding="utf-8")

        print(f"[PlanWriterPipe] Plan saved: {plan_path}")
        if safe_thread_id:
            print(f"[PlanWriterPipe] HTTP: /view?filename={plan_id}.json&type=output&subfolder={plan_subfolder}")

        return (str(plan_path), plan_json)


# ── Node registration ──────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "BeatDropPlanWriterPipe": BeatDropPlanWriterPipe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BeatDropPlanWriterPipe": "🔗 BeatDrop Plan Pipe (PlanWriter)",
}
