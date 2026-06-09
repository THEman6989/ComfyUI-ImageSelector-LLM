"""
BeatDrop Config Pipe — AI Stack → Selector bridge.

Bundles all AI-Stack-controlled parameters into a single ai_stack_config_json
output. The AI Stack (AlphaRavis) only needs to wire ONE cable to the
BeatDropSelectorEmbeddingNode instead of filling 10+ fields individually.

Preset system:
  Named JSON presets in presets/ provide pre-configured parameter sets.
  Select a preset via the dropdown — it loads as the BASE layer.
  Individual field values OVERRIDE the preset, so you can pick
  "outfit_transformation" and still tweak scene_fit_weight manually.

Wiring:
  AlphaRavis (workflow injection) → 🔧 Config Pipe → ai_stack_config_json → 🎵 Selector

Place in: ComfyUI-ImageSelector-LLM/beatdrop_config_pipe.py
"""

import json
import os
from pathlib import Path


class BeatDropConfigPipe:
    """AI Stack → Selector bridge: bundles config into one JSON output.

    Presets are loaded from presets/ (relative to this file) and merged as
    the base layer. Individual input fields override preset values, so a
    preset gives you sensible defaults that you can still customize per-run.

    Preset JSON format — same structure as ai_stack_config_json output:
      {
        "name": "Human-readable name",
        "description": "What this preset does",
        "category": "grouping tag",
        "weights": {...},
        "thresholds": {...},
        "text_query_per_phase": {...},
        "pair_constraints": [...],
        "pair_reranker": {...},
        "reranker_query": "...",
        "max_candidate_images": 82,
        "use_vlm_fallback": false
      }
    """

    # ── Preset discovery ──────────────────────────────────────────────
    @staticmethod
    def _presets_dir():
        return Path(__file__).resolve().parent / "presets"

    @classmethod
    def _list_presets(cls):
        """Return preset IDs for the ComfyUI dropdown."""
        presets = ["none"]
        pd = cls._presets_dir()
        if pd.is_dir():
            for f in sorted(pd.glob("*.json")):
                presets.append(f.stem)
        return presets

    @classmethod
    def INPUT_TYPES(cls):
        preset_list = cls._list_presets()
        return {
            "required": {},
            "optional": {
                # ── Preset (base layer) ──
                "preset": (preset_list, {
                    "default": "none",
                    "tooltip": "Load a preset as base config. Individual fields override preset values.",
                }),
                # ── Core AI Stack fields ──
                "thread_id": ("STRING", {
                    "default": "", "multiline": False, "forceInput": True,
                    "tooltip": "AlphaRavis conversation ID. Passed through to selector + plan.",
                }),
                "extra_instructions": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "User instructions: 'Streetwear, Phase 0 Jacke, Phase 1 ohne'.",
                }),
                # ── Text queries (semantic embedding matching) ──
                "text_query_scene_fit": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "Scene aesthetic: 'urban nightclub, dark lighting, edgy'.",
                }),
                "text_query_change_target": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "What makes a good swap: 'dramatic silhouette change'.",
                }),
                "text_query_per_phase_json": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "Per-phase text: {\"0\":\"jacket streetwear\",\"1\":\"casual no jacket\"}.",
                }),
                # ── Re-Ranker control ──
                "reranker_query": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "Custom reranker query. Empty = auto.",
                }),
                # ── Weights ──
                "scene_fit_weight": ("FLOAT", {
                    "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Weight for scene-fit in composite score.",
                }),
                "change_strength_weight": ("FLOAT", {
                    "default": 0.50, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Weight for change-strength.",
                }),
                "diversity_weight": ("FLOAT", {
                    "default": 0.20, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Weight for diversity across embedding space.",
                }),
                "reranker_blend_weight": ("FLOAT", {
                    "default": 0.50, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How much to blend reranker scores.",
                }),
                # ── Thresholds ──
                "embedding_confidence_threshold": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Stage 1→2 threshold.",
                }),
                "reranker_confidence_threshold": ("FLOAT", {
                    "default": 0.70, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Stage 2→3 threshold.",
                }),
                # ── History / Johnson penalty ──
                "history_penalty": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 50.0, "step": 0.5,
                    "tooltip": "Johnson history base penalty.",
                }),
                "history_decay_rate": ("FLOAT", {
                    "default": 0.3, "min": 0.05, "max": 2.0, "step": 0.05,
                    "tooltip": "How fast history penalty decays.",
                }),
                # ── Limits ──
                "max_frames_per_window": ("INT", {
                    "default": 4, "min": 2, "max": 20,
                    "tooltip": "Max frames per drop window.",
                }),
                "max_candidate_images": ("INT", {
                    "default": 30, "min": 5, "max": 500, "step": 5,
                    "tooltip": "Max outfit images to load.",
                }),
                # ── Stages ──
                "use_vlm_fallback": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable Stage 3 VLM fallback.",
                }),
                # ── Folder assignment ──
                "folder_assignments_json": ("STRING", {
                    "default": "", "multiline": True, "forceInput": True,
                    "tooltip": "Manual: {\"0\":\"jackets\",\"1\":\"no_jacket\"}. Empty = auto.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ai_stack_config_json",)
    FUNCTION = "bundle"
    CATEGORY = "Amin/Beatdrop"

    def _deep_merge(self, base, override):
        """Recursive dict merge for preset inheritance. Lists/scalars override."""
        if not isinstance(base, dict):
            base = {}
        out = json.loads(json.dumps(base))
        if not isinstance(override, dict):
            return out
        for key, val in override.items():
            if key in ("name", "description", "category"):
                out[key] = val
            elif isinstance(val, dict) and isinstance(out.get(key), dict):
                out[key] = self._deep_merge(out[key], val)
            else:
                out[key] = val
        return out

    def _load_preset(self, preset_name, _seen=None):
        """Load a preset JSON file with optional `extends` inheritance."""
        if not preset_name or preset_name == "none":
            return None
        _seen = set(_seen or [])
        if preset_name in _seen:
            print(f"[ConfigPipe] Preset inheritance cycle ignored: {preset_name}")
            return None
        _seen.add(preset_name)

        preset_path = self._presets_dir() / f"{preset_name}.json"
        if not preset_path.is_file():
            print(f"[ConfigPipe] Preset not found: {preset_path}")
            return None
        try:
            data = json.loads(preset_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"[ConfigPipe] Failed to load preset {preset_name}: {e}")
            return None

        parents = data.get("extends") or data.get("inherits") or []
        if isinstance(parents, str):
            parents = [parents]
        merged = {}
        for parent in parents:
            parent_data = self._load_preset(str(parent), _seen=_seen)
            if parent_data:
                merged = self._deep_merge(merged, parent_data)
        child = {k: v for k, v in data.items() if k not in ("extends", "inherits")}
        merged = self._deep_merge(merged, child)
        print(f"[ConfigPipe] Loaded preset: {merged.get('name', preset_name)}")
        return merged

    def bundle(
        self,
        preset="none",
        thread_id="",
        extra_instructions="",
        text_query_scene_fit="",
        text_query_change_target="",
        text_query_per_phase_json="",
        reranker_query="",
        scene_fit_weight=0.30,
        change_strength_weight=0.50,
        diversity_weight=0.20,
        reranker_blend_weight=0.50,
        embedding_confidence_threshold=0.75,
        reranker_confidence_threshold=0.70,
        history_penalty=10.0,
        history_decay_rate=0.3,
        max_frames_per_window=4,
        max_candidate_images=30,
        use_vlm_fallback=False,
        folder_assignments_json="",
    ):
        # ── Layer 1: Preset (base) ──
        cfg = {}
        preset_data = self._load_preset(preset)
        if preset_data:
            # Copy all preset keys EXCEPT metadata fields
            for key, val in preset_data.items():
                if key in ("name", "description", "category"):
                    continue
                cfg[key] = val

        # ── Layer 2: Individual field overrides (win over preset) ──

        # Text
        if thread_id and str(thread_id).strip():
            cfg["conversation_id"] = str(thread_id).strip()
        if extra_instructions and str(extra_instructions).strip():
            cfg["extra_instructions"] = str(extra_instructions).strip()
        if text_query_scene_fit and str(text_query_scene_fit).strip():
            cfg["text_query_scene_fit"] = str(text_query_scene_fit).strip()
        if text_query_change_target and str(text_query_change_target).strip():
            cfg["text_query_change_target"] = str(text_query_change_target).strip()
        if reranker_query and str(reranker_query).strip():
            cfg["reranker_query"] = str(reranker_query).strip()

        # Per-phase text queries
        if text_query_per_phase_json and str(text_query_per_phase_json).strip() not in ("", "{}"):
            try:
                cfg["text_query_per_phase"] = json.loads(text_query_per_phase_json)
            except json.JSONDecodeError:
                pass

        # Folder assignments
        if folder_assignments_json and str(folder_assignments_json).strip() not in ("", "{}"):
            try:
                cfg["folder_assignments"] = json.loads(folder_assignments_json)
            except json.JSONDecodeError:
                pass

        # Weights — only set if user explicitly changed from defaults, OR if
        # preset didn't already set a weights dict (preset wins over defaults).
        if "weights" not in cfg:
            cfg["weights"] = {}
        w = cfg["weights"]

        # Track which weights were explicitly provided (non-default from pipe params).
        # The pipe defaults are: scene_fit=0.30, change=0.50, diversity=0.20,
        # reranker_blend=0.50. If the preset already set these and the user left
        # them at pipe defaults, keep preset values. Otherwise override.
        def _was_changed(param_value, pipe_default):
            return abs(float(param_value) - pipe_default) > 0.001

        if "scene_fit" not in w or _was_changed(scene_fit_weight, 0.30):
            w["scene_fit"] = scene_fit_weight
        if "change_strength" not in w or _was_changed(change_strength_weight, 0.50):
            w["change_strength"] = change_strength_weight
        if "diversity" not in w or _was_changed(diversity_weight, 0.20):
            w["diversity"] = diversity_weight
        if "reranker_blend" not in w or _was_changed(reranker_blend_weight, 0.50):
            w["reranker_blend"] = reranker_blend_weight

        # Thresholds
        if "thresholds" not in cfg:
            cfg["thresholds"] = {}
        t = cfg["thresholds"]
        if "embedding_confidence" not in t or _was_changed(embedding_confidence_threshold, 0.75):
            t["embedding_confidence"] = embedding_confidence_threshold
        if "reranker_confidence" not in t or _was_changed(reranker_confidence_threshold, 0.70):
            t["reranker_confidence"] = reranker_confidence_threshold

        # History
        if "history" not in cfg:
            cfg["history"] = {}
        h = cfg["history"]
        if "penalty" not in h or _was_changed(history_penalty, 10.0):
            h["penalty"] = history_penalty
        if "decay_rate" not in h or _was_changed(history_decay_rate, 0.3):
            h["decay_rate"] = history_decay_rate

        # Limits
        # Important: when a preset is selected, do NOT inject UI defaults unless
        # the user changed them. Otherwise the ConfigPipe silently overrides the
        # Selector node's own max_frames_per_window/max_candidate_images values.
        if preset_data:
            if _was_changed(max_frames_per_window, 4.0):
                cfg["max_frames_per_window"] = max_frames_per_window
            if _was_changed(max_candidate_images, 30.0):
                cfg["max_candidate_images"] = max_candidate_images
        else:
            cfg["max_frames_per_window"] = max_frames_per_window
            cfg["max_candidate_images"] = max_candidate_images

        # Stages
        if preset_data:
            if _was_changed(float(use_vlm_fallback), 0.0):
                cfg["use_vlm_fallback"] = use_vlm_fallback
        elif "use_vlm_fallback" not in cfg:
            cfg["use_vlm_fallback"] = use_vlm_fallback

        if preset_data:
            print(f"[ConfigPipe] Preset '{preset_data.get('name', preset)}' applied + field overrides")

        return (json.dumps(cfg),)


# ── Node registration ──────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "BeatDropConfigPipe": BeatDropConfigPipe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BeatDropConfigPipe": "🔧 BeatDrop Config Pipe (AI Stack)",
}
