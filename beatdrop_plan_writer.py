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
                "job_policy": (["main_job_only", "drops_only", "all_frames"], {
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
        for key, source in [
            ("beats", beats_json),
            ("change_detection", change_detection_json),
            ("mask_quality", mask_quality_json),
            ("render_instructions", render_instructions_json),
        ]:
            if source and str(source).strip() not in ("", "{}"):
                try:
                    plan[key] = json.loads(source)
                except (json.JSONDecodeError, TypeError):
                    plan[f"{key}_raw"] = str(source)[:10000]

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
