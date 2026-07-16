import importlib
import json
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO.parents[1]
for path in (str(COMFY_ROOT), str(REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)

selector_module = importlib.import_module("beatdrop_selector_embedding")
config_module = importlib.import_module("beatdrop_config_pipe")
BeatDropSelectorEmbeddingNode = selector_module.BeatDropSelectorEmbeddingNode
BeatDropConfigPipe = config_module.BeatDropConfigPipe


def test_identical_candidate_tensors_share_stable_source_identity():
    node = BeatDropSelectorEmbeddingNode()
    frames = torch.stack([
        torch.zeros((2, 2, 3), dtype=torch.float32),
        torch.zeros((2, 2, 3), dtype=torch.float32),
    ])

    first = node._candidate_source_identity(0, {}, frames)
    repeated = node._candidate_source_identity(1, {}, frames)

    assert first == repeated
    assert first.startswith("sha256:")


def test_selector_schema_separates_scene_references_from_outfit_candidates():
    inputs = BeatDropSelectorEmbeddingNode.INPUT_TYPES()

    assert "reference_frames" in inputs["optional"]
    assert "outfit_candidates" in inputs["optional"]
    assert inputs["required"]["max_frames_per_window"][1]["min"] == 1
    assert inputs["required"]["max_frames_per_window"][1]["default"] == 1


def test_config_pipe_allows_one_outfit_per_phase():
    inputs = BeatDropConfigPipe.INPUT_TYPES()
    assert inputs["optional"]["max_frames_per_window"][1]["min"] == 1


def test_candidate_windows_preserve_folder_phase_boundaries():
    node = BeatDropSelectorEmbeddingNode()
    info = {
        "phase_info": [
            {"phase": 0, "folder": "before", "images_loaded": 2},
            {"phase": 1, "folder": "after", "images_loaded": 3},
        ]
    }

    windows = node._candidate_windows_from_folder_info(info, candidate_count=5)

    assert windows == [
        {
            "phase": 0,
            "folder": "before",
            "batch_start": 0,
            "batch_end": 2,
            "frame_indices": [0, 1],
        },
        {
            "phase": 1,
            "folder": "after",
            "batch_start": 2,
            "batch_end": 5,
            "frame_indices": [2, 3, 4],
        },
    ]


def test_scene_reference_downsampling_preserves_phase_windows():
    node = BeatDropSelectorEmbeddingNode()
    frames = torch.arange(10, dtype=torch.float32).reshape(10, 1, 1, 1)
    windows = [
        {"phase": 0, "batch_start": 0, "batch_end": 5, "frame_indices": list(range(5))},
        {"phase": 1, "batch_start": 5, "batch_end": 10, "frame_indices": list(range(5, 10))},
    ]

    sampled, sampled_windows = node._downsample_scene_references(frames, windows, max_frames=4)

    assert sampled[:, 0, 0, 0].tolist() == [0.0, 4.0, 5.0, 9.0]
    assert sampled_windows == [
        {"phase": 0, "batch_start": 0, "batch_end": 2, "frame_indices": [0, 1], "_downsampled": True},
        {"phase": 1, "batch_start": 2, "batch_end": 4, "frame_indices": [2, 3], "_downsampled": True},
    ]


def test_reference_frames_do_not_replace_folder_candidates():
    scene_frames = torch.ones((4, 8, 8, 3), dtype=torch.float32)
    candidates = torch.stack(
        [torch.full((8, 8, 3), value, dtype=torch.float32) for value in (0.1, 0.2, 0.3, 0.4)]
    )
    beats_used = json.dumps(
        [
            {
                "beat_index": 0,
                "time_seconds": 1.0,
                "batch_offset": 0,
                "batch_frame_count": 4,
                "range_start": 0.5,
                "range_end": 1.5,
            }
        ]
    )

    class RecordingSelector(BeatDropSelectorEmbeddingNode):
        def __init__(self):
            self.load_call = None

        def _load_from_folders(self, *args, **kwargs):
            self.load_call = {"args": args, "kwargs": kwargs}
            return candidates, {
                "source": "folders",
                "root": "/fake/outfits",
                "phase_info": [
                    {"phase": 0, "folder": "pool", "images_loaded": 2},
                    {"phase": 1, "folder": "pool", "images_loaded": 2},
                ],
                "folder_assignments": [
                    {"phase": 0, "folder": "pool", "reason": "test"},
                    {"phase": 1, "folder": "pool", "reason": "test"},
                ],
                "images_loaded": 4,
                "max_candidate_images": 4,
                "max_per_phase": 2,
                "_image_stems": {0: "a", 1: "b", 2: "c", 3: "d"},
                "_image_paths": {},
                "_cached_embeddings": {},
                "embedding_cache": {"enabled": False},
            }

        def _build_ai_stack_context(self, **kwargs):
            return json.dumps(
                {
                    "candidate_pixel_mean": round(float(kwargs["candidate_frames"].mean()), 3),
                    "scene_reference_count": int(kwargs["scene_reference_frames"].shape[0]),
                }
            )

    node = RecordingSelector()
    result = node.select(
        max_frames_per_window=1,
        num_outfits_mode="auto_from_beats",
        num_outfits=2,
        reference_frames=scene_frames,
        beats_used=beats_used,
        candidate_folders="/fake/outfits",
        max_candidate_images=4,
        use_random_sample=False,
        use_embedding_cache=False,
    )

    assert node.load_call is not None
    assert node.load_call["kwargs"]["reference_frames"] is scene_frames
    metadata = json.loads(result[2])
    assert metadata["candidate_source"] == "folders"
    assert metadata["total_candidates"] == 4
    assert metadata["scene_reference_frames"] == 4
    assert metadata["windows_count"] == 2
    context = json.loads(result[5])
    assert context == {"candidate_pixel_mean": 0.25, "scene_reference_count": 4}


def test_auto_from_beats_requires_one_more_outfit_state_than_transitions():
    node = BeatDropSelectorEmbeddingNode()

    assert node._required_outfit_states(beat_count=1, mode="auto_from_beats", manual_count=2) == 2
    assert node._required_outfit_states(beat_count=3, mode="auto_from_beats", manual_count=2) == 4
    assert node._required_outfit_states(beat_count=8, mode="manual", manual_count=5) == 5
