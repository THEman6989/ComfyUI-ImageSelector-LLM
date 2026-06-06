# ARCHITECTURE — ComfyUI-ImageSelector-LLM

Komplexe Nodes und wie sie intern funktionieren.

---

## BeatDropSelectorNode

**Datei:** `beatdrop_selector_node.py`

### Zweck

Wählt pro Drop-Fenster die besten Frames für Outfit-Wechsel aus.  
Nicht für Single-Outfit — dafür gibt's `LLMImageSelectorNode`.  
BeatDropSelector ist für **2+ Outfits**, mit Diversity-Erzwingung.

### Architektur — Flow

```
beats_used JSON (von FrameSequenceGenerator)
        │
        ▼
┌───────────────────┐
│ Fenster bauen      │  pro Eintrag: batch_offset + batch_frame_count
│                    │  → window 0: frames 0..20, window 1: frames 21..41
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│ Re-Ranker Pre-Filter│ (optional, reranker_endpoint gesetzt)
│                    │  Query: Scene-Fit + Change-Strength
│                    │  → Top-K (reranker_top_k, default 12)
│                    │  → Nur diese Kandidaten kommen weiter
└───────┬───────────┘
        │
        ▼
┌───────────────────────────────────────────────┐
│ Scoring-Loop (pro Fenster, pro Frame)          │
│                                                 │
│  penalty = 0                                    │
│  + Judge-Extra-Penalty (extra_penalty_json)     │
│  + Johnson-History-Penalty (history_decay_rate) │
│  + Re-Ranker-Blend (blend_w × (100-score))      │
│  + Diversity-Penalty (Abstand zu vorherigen)    │
│                                                 │
│  → Sortiert nach Penalty (niedrigster = bester) │
│  → Top-N selektiert                             │
└───────────────────────┬───────────────────────┘
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
    ┌──────────────┐        ┌──────────────────┐
    │ LOCAL Mode   │        │ LLM Mode         │
    │ Penalty-bas. │        │ Kontaktbogen →   │
    │ direkt Top-N │        │ Vision-LLM →     │
    │              │        │ selected_ids     │
    └──────────────┘        └──────────────────┘
            │                       │
            └───────────┬───────────┘
                        ▼
              selected_indices
              + contact_sheet
              + metadata JSON
```

### Judge-Retry (interner Loop)

Wenn `extra_penalty_json` nicht leer ist (Judge hat Fehler gemeldet):

```
Pass 1: Selektion OHNE Penalties → Baseline
Pass 2: Selektion MIT Penalties  → Korrigiert (wird returned)

Metadata enthält:
  correction: {
    judge_corrections_applied: true,
    baseline_selection: [...],
    corrected_selection: [...],
    changed_frames: [...]       // Frames die durch Korrektur getauscht wurden
  }
```

### Diversity-Penalty

Erzwingt, dass selektierte Frames aus verschiedenen Videobereichen kommen:

```python
for prev_idx in all_selected:
    dist = abs(current_idx - prev_idx)
    if dist < n_per_window:
        penalty += history_penalty * (1.0 - dist / n_per_window)
```

Je näher ein Kandidat an einem bereits selektierten Frame liegt, desto höher der Penalty.

### Johnson-History-Penalty

Verhindert, dass immer dieselben Frames genommen werden:

```python
decay = 1 / (1 + most_recent_selections_ago × decay_rate)
freq  = 1 + (times_selected - 1) × 0.15
penalty = base × decay × freq   (capped at base × 1.5)
```

`history_decay_rate` steuert, wie schnell der Penalty abnimmt:
- `0.1` = langsam (Frame bleibt länger bestraft)
- `0.3` = default (moderat)
- `1.0` = schnell (Frame cycled schnell zurück)

### num_outfits_mode

- `auto_from_beats`: `num_outfits = anzahl windows in beats_used` (min 2)
- `manual`: Slider-Wert (min 2)

### Folder-Based Candidate Loading (NEU)

Statt IMAGE-Tensor direkt — aus Ordnerstruktur laden:

```
candidate_folders = "/path/to/outfits/"
    │
    ├── jackets/      (5 Bilder)
    ├── no_jacket/    (5 Bilder)
    ├── exotic/       (3 Bilder)
    └── chill/        (4 Bilder)

        │
        ▼
┌───────────────────────────┐
│ 1. _scan_folders()        │
│    → Liste aller Ordner   │
│    + Bildpfade            │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│ 2. _select_folders_via_llm│  (wenn endpoint+model)
│    3 Sample-Bilder/Ordner │
│    → Vision LLM           │
│    → selected_folders     │  z.B. ["jackets", "no_jacket"]
│    extra_instructions mit │
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐
│ 3. _load_filtered_candidates│
│    History-Pre-Filter:    │  Bereits benutzte Bilder raus
│    Random-Sample:         │  Top-Pool → zufällig max 30
│    → images_tensor        │
└───────────┬───────────────┘
            │
            ▼
     Re-Ranker → Scoring-Loop (wie vorher)
```

Wenn kein `endpoint`/`model` gesetzt: alle Ordner werden benutzt (Fallback).

### Wichtige Inputs

| Input | Default | Bedeutung |
|-------|---------|-----------|
| `beats_used` | `""` | JSON von FrameSequenceGenerator (batch_offset + batch_frame_count) |
| `reranker_top_k` | 12 | Re-Ranker filtert auf Top-K vor. 0 = alle durch |
| `reranker_query` | `""` | Custom Query. Empty = Scene-Fit + Change-Strength |
| `reranker_blend_weight` | 0.3 | Wie stark Re-Ranker-Score gewichtet wird |
| `history_decay_rate` | 0.3 | Wie schnell History-Penalty abklingt |
| `extra_penalty_json` | `{}` | Judge-Feedback: `{"frame_3": 5.0}` bestraft Frame 3 |
| `extra_instructions` | `""` | Zusatz-Anweisungen für LLM-Mode |
| `use_llm` | false | LLM-Mode: Kontaktbogen → Vision-LLM |

---

## AlphaRavisOutfitReferenceJudgeNode

**Datei:** `outfit_reference_judge.py`

### Zweck

Vision-LLM-Judge: Bewertet Kandidaten-Outfits semantisch.  
NICHT nur "schönstes Outfit" — sondern **bestes für sichtbaren Beatdrop-Wechsel**.

### Architektur

```
Top-K Kandidaten (vom Re-Ranker)
+ old_outfit_crop (altes Outfit zum Vergleich)
+ scene_reference_image (Szenen-Kontext)
        │
        ▼
┌───────────────────────────────┐
│ Chunking                      │
│ max_images_per_call (8)       │
│ → Kontaktbögen pro Chunk     │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Vision-LLM Call               │
│ Prompt: 5 Scoring-Dimensionen │
│ + old_outfit + scene + crops  │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Scores pro Kandidat:          │
│ 1. scene_fit_score       10%  │
│ 2. change_strength_score 40%  │← WICHTIGSTE
│ 3. beatdrop_impact_score 30%  │
│ 4. render_safety_score   20%  │
│ 5. too_similar_to_old    ×0.1 │← Kill-Switch
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Merge über Chunks             │
│ → Bestes Outfit selected      │
│ → judge_json mit allen Scores │
│ → AlphaRavis Metadata         │
└───────────────────────────────┘
```

### Scoring-Gewichte (WARUM)

| Kriterium | Gewicht | Begründung |
|-----------|---------|-----------|
| change_strength | **40%** | Der Wechsel MUSS sichtbar anders sein |
| beatdrop_impact | **30%** | Der Wechsel muss zum Beat passen |
| render_safety | 20% | Outfit muss für WAN/CG2 renderbar sein |
| scene_fit | 10% | Outfit muss zur Szene passen |
| too_similar | **×0.1** | Wenn zu ähnlich → Score gekillt |

### Person ≠ Outfit

Der Judge-Prompt betont: Die Person im Referenzbild ist NICHT die Video-Person.  
Verglichen wird nur die KLEIDUNG: Silhouette, Schnitt, Stil, Vibe.  
Nicht: Gesicht, Körperbau, Identität.

### AlphaRavis Bridge

Payload enthält `conversation_id` + `metadata`:
```json
{
  "conversation_id": "thread_abc",
  "metadata": {
    "conversation_id": "thread_abc",
    "run_id": "run_001",
    "drop_id": "drop_003",
    "source": "comfyui_researcher"
  }
}
```
Gesendet via HTTP-Body UND Header (`x-conversation-id`, `x-thread-id`).

### Fallbacks

| Situation | Output |
|-----------|--------|
| Kein Kandidat über Schwelle | `needs_user_review: true`, conf < 0.3 |
| Alle zu ähnlich | `too_similar: true`, conf niedrig |
| Keine Kandidaten | `error` in judge_json |

### Wichtige Inputs

| Input | Default | Bedeutung |
|-------|---------|-----------|
| `max_images_per_call` | 8 | Kandidaten pro Kontaktbogen-Chunk |
| `extra_instructions` | `""` | Zusatz-Anweisungen in den Judge-Prompt |
| `conversation_id` | `""` | AlphaRavis Thread-ID |
| `old_outfit_crop` | — | Altes Outfit zum Vergleich |
| `candidate_images` | — | Top-K vom Re-Ranker |
