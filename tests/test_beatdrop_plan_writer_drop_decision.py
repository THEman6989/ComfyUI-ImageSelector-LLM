import importlib
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO.parents[1]
for path in (str(COMFY_ROOT), str(REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

module = importlib.import_module("beatdrop_plan_writer")
folder_paths = importlib.import_module("folder_paths")
BeatDropPlanWriterPipe = module.BeatDropPlanWriterPipe


def test_state_plan_rejects_single_unique_outfit_for_transitions():
    decisions = [{"transition_index": 0}]
    selection = {
        "phase_decisions": [{"selected_frames": [4, 12]}],
        "saved_files": [
            {"frame_index": 4, "source_identity": "outfit-a"},
            {"frame_index": 12, "source_identity": "outfit-a"},
        ],
    }

    try:
        module._build_outfit_state_plan(decisions, selection)
    except ValueError as exc:
        assert "at least two unique outfit candidates" in str(exc)
    else:
        raise AssertionError("single unique outfit must not create adjacent duplicate states")


def _beats():
    return [{
        "beat_index": 0,
        "time_seconds": 7.32,
        "frame_index": 220,
        "confidence": 1.0,
        "is_drop": True,
        "method": "beat_this",
        "batch_offset": 0,
        "batch_frame_count": 3,
        "frames": [
            {"batch_index": 0, "source_frame_index": 219, "time_seconds": 7.30},
            {"batch_index": 1, "source_frame_index": 220, "time_seconds": 7.32},
            {"batch_index": 2, "source_frame_index": 221, "time_seconds": 7.36},
        ],
    }]


def _write(tmp_path, monkeypatch, change):
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    _, plan_json = BeatDropPlanWriterPipe().write_plan(
        beats_json=json.dumps(_beats()),
        change_detection_json=json.dumps(change),
        mask_quality_json="{}",
        ai_stack_context="{}",
        plan_id="gate-test",
    )
    return json.loads(plan_json)


def test_plan_uses_audio_drop_when_dino_found_no_outfit_change(tmp_path, monkeypatch):
    plan = _write(tmp_path, monkeypatch, {
        "has_existing_visual_change": False,
        "needs_generated_outfit_drop": True,
        "beat_frame": 1,
        "best_change": None,
    })

    assert plan["drop_decision"] == {
        "source": "audio_beat",
        "dino_used": False,
        "source_frame_index": 220,
        "time_seconds": 7.32,
        "local_batch_index": 1,
        "needs_generated_outfit_drop": True,
    }
    assert plan["change_detection"]["ignored_for_drop_decision"] is True


def test_plan_uses_dino_only_for_real_outfit_change(tmp_path, monkeypatch):
    plan = _write(tmp_path, monkeypatch, {
        "has_existing_visual_change": True,
        "needs_generated_outfit_drop": False,
        "beat_frame": 1,
        "best_change": {"from_frame": 1, "to_frame": 2, "score": 0.8},
    })

    assert plan["drop_decision"] == {
        "source": "dinov2_visual_change",
        "dino_used": True,
        "source_frame_index": 221,
        "time_seconds": 7.36,
        "local_batch_index": 2,
        "needs_generated_outfit_drop": False,
    }
    assert plan["change_detection"]["ignored_for_drop_decision"] is False


def test_plan_emits_one_audio_transition_per_selected_beat(tmp_path, monkeypatch):
    beats = [
        {
            "beat_index": index,
            "time_seconds": time_seconds,
            "frame_index": frame_index,
            "is_drop": False,
            "selection_mode": "beats_after_drop",
            "relative_to_anchor": "after",
            "anchor_drop_time_seconds": 4.12,
            "batch_offset": index * 3,
            "batch_frame_count": 3,
            "frames": [
                {
                    "batch_index": index * 3 + 1,
                    "source_frame_index": frame_index,
                    "time_seconds": time_seconds,
                }
            ],
        }
        for index, (time_seconds, frame_index) in enumerate([
            (4.62, 139),
            (5.12, 154),
            (5.62, 169),
        ])
    ]
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))

    _, plan_json = BeatDropPlanWriterPipe().write_plan(
        job_policy="beats_after_drop",
        beats_json=json.dumps(beats),
        change_detection_json=json.dumps({
            "has_existing_visual_change": True,
            "best_change": {"from_frame": 0, "to_frame": 2, "score": 0.9},
        }),
        mask_quality_json="{}",
        ai_stack_context=json.dumps({
            "phase_decisions": [{"selected_frames": [4, 12, 5]}],
            "saved_files": [
                {"frame_index": 4, "source_identity": "outfit-a", "source_path": "/library/a.png"},
                {"frame_index": 12, "source_identity": "outfit-a", "source_path": "/library/a.png"},
                {"frame_index": 5, "source_identity": "outfit-b", "source_path": "/library/b.png"},
            ],
        }),
        plan_id="multi-beat-test",
    )
    plan = json.loads(plan_json)

    assert [decision["time_seconds"] for decision in plan["beat_decisions"]] == [4.62, 5.12, 5.62]
    assert [decision["transition_index"] for decision in plan["beat_decisions"]] == [0, 1, 2]
    assert [decision["outfit_state_after"] for decision in plan["beat_decisions"]] == [1, 2, 3]
    assert plan["required_outfit_states"] == 4
    assert all(decision["dino_used"] is False for decision in plan["beat_decisions"])
    assert plan["drop_decision"]["source"] == "audio_beat"
    assert plan["drop_decision"]["dino_used"] is False
    assert plan["change_detection"]["ignored_for_drop_decision"] is True
    assert [state["candidate_frame"] for state in plan["outfit_state_plan"]] == [4, 5, 4, 5]
    assert [decision["outfit_candidate_after"] for decision in plan["beat_decisions"]] == [5, 4, 5]
