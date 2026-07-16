# BeatDrop outfit-change workflow modes

The workflow supports four distinct timing modes:

1. `drops_only` — change outfits only on detected drop events.
2. `all_beats` — change outfits on every eligible detected beat.
3. `beats_before_drop` — change on every eligible beat before the selected main-drop anchor; the anchor itself is excluded.
4. `beats_after_drop` — change on every eligible beat after the selected main-drop anchor; the anchor itself is excluded.

Included portable API workflows:

- `beatdrop-outfit-change-mode-3-before-drop.json`
- `beatdrop-outfit-change-mode-4-after-drop.json`

Before running either workflow, replace these placeholders in nodes 1–3 and 12:

- `SET_VIDEO_PATH.mp4`
- `SET_OUTFIT_LIBRARY_FOLDER`

The example cache files are relative to the ComfyUI working directory under `beatdrop_cache/`.
Mode 3 requires one more outfit state than selected pre-drop beats. Mode 4 does the same for post-drop beats. If fewer unique outfit candidates are available than required states, the plan writer cycles the selected candidates while avoiding immediate duplicates.
