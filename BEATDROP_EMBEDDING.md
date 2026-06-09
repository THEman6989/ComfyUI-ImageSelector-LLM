# BeatDrop Selector (Embedding) — `BeatDropSelectorEmbeddingNode`

**Datei:** `ComfyUI-ImageSelector-LLM/beatdrop_selector_embedding.py`
**Kategorie:** `Amin/Beatdrop`
**Display-Name:** `🎵 BeatDrop Selector (Embedding)`

---

## Übersicht

3-Stage Embedding-basierter Outfit-Selektor für Beatdrop-Video-Pipelines.
Ersetzt Vision-LLM-Judging durch eine Cascading-Architektur:

```
STAGE 1: Qwen3-VL-Embedding-8B    → Cosine-Similarity Scoring (~50ms)
STAGE 2: Re-Ranker Cross-Encoder  → Cross-Attention Scoring (~150ms)
STAGE 3: VLM Fallback (optional)  → Vision-LLM Judge (~2-15s, <5% der Fälle)
```

Stage 1+2 laufen lokal (3090), Stage 3 nur bei Unsicherheit auf Remote-Server.

---

## AI Stack Integration

### `ai_stack_config_json` — Override-Alles-Hub

**Ein Input-Feld** (STRING, multiline). Der AI Stack sendet JSON — Python merged
über die Defaults. Kein Workflow-Parsing, kein separater API-Endpoint nötig.

```json
{
  "extra_instructions": "Diesmal Streetwear, kein Formal",
  "text_query_scene_fit": "urban nightclub, dark lighting, edgy",
  "text_query_change_target": "dramatic silhouette shift, formal→casual",
  "reranker_query": "Maximaler Beatdrop-Wechsel. Phase 0 mit Jacke, Phase 1 ohne.",
  "conversation_id": "thread_abc123",

  "weights": {
    "scene_fit": 0.20,
    "change_strength": 0.60,
    "diversity": 0.20,
    "reranker_blend": 0.5
  },

  "thresholds": {
    "embedding_confidence": 0.70,
    "reranker_confidence": 0.65
  },

  "history": {
    "penalty": 8.0,
    "decay_rate": 0.5,
    "max_entries": 150
  },

  "max_frames_per_window": 5,
  "max_candidate_images": 50,
  "use_vlm_fallback": false,

  "text_query_per_phase": {
    "0": {
      "query": "Phase 0 before the drop: full-body outfit with maximum coverage",
      "must": [
        "full body photo showing the entire person head to toe",
        "fully covered outfit, lots of fabric, jacket or dress, modest clothing"
      ],
      "avoid": [
        "cropped close-up or partial body shot",
        "minimal fabric outfit, exposed skin, underwear or swimwear look"
      ],
      "filter": {"enabled": true, "threshold": -0.02}
    },
    "1": {
      "query": "Phase 1 after the drop: full-body outfit with minimum coverage",
      "must": [
        "full body photo showing the entire person head to toe",
        "minimal fabric outfit, exposed skin, underwear or swimwear look"
      ],
      "avoid": [
        "cropped close-up or partial body shot",
        "fully covered outfit, lots of fabric, jacket or dress, modest clothing"
      ],
      "filter": {"enabled": true, "threshold": -0.02}
    }
  },
  "folder_assignments": {
    "0": "jackets",
    "1": "no_jacket"
  }
}
```

**Alle Keys sind optional.** Nur gesetzte Keys überschreiben die Node-Defaults.

### LLM/AI-Stack Regel: User-Intent in `must` / `avoid` übersetzen

Der Selector soll keine Domain-Wörter hardcoden. Der AI Stack bzw. das planende LLM muss User-Wünsche in positive und negative visuelle Suchphrasen umschreiben:

```json
{
  "text_query_per_phase": {
    "0": {
      "query": "short human-readable summary for logs",
      "must": ["visual property that MUST be present", "another positive property"],
      "avoid": ["visual property that should be rejected", "another negative property"],
      "filter": {"enabled": true, "threshold": -0.02}
    }
  }
}
```

Semantik:
- `query`: lesbare Zusammenfassung und zusätzlicher weicher Embedding-Query.
- `must`: positive Embedding-Queries; Score steigt, wenn das Bild diese Eigenschaften hat.
- `avoid`: negative Embedding-Queries; Score sinkt, wenn das Bild diese Eigenschaften hat.
- `filter.enabled=true`: Vorfilter-ähnliche harte Penalty für Kandidaten unter `threshold`. Das entfernt nicht physisch aus der Liste, macht aber Kandidaten praktisch unattraktiv. Wenn alle Kandidaten failen, wählt die Node trotzdem den besten Fail statt komplett abzubrechen.
- `threshold`: bei `must+avoid` ist das Schwelle für `avg(must_similarity) - avg(avoid_similarity)`. Bei nur `must` ist es Schwelle für `avg(must_similarity)`.

Cross-Phase/Paar-Regeln kommen in `pair_constraints`:

```json
{
  "pair_constraints": [
    {"source_phase": 0, "target_phase": 1, "match": ["same_color"], "weight": 8.0}
  ],
  "pair_reranker": {
    "enabled": true,
    "mode": "transformers",
    "top_k_per_phase": 20,
    "max_pairs": 200,
    "query": "Choose the best side-by-side before/after outfit pair. LEFT must be full-body fully covered. RIGHT must be full-body more revealing. Both must share similar color/style."
  }
}
```

Semantik:
- `source_phase` / `target_phase`: welche Phasen als Paar zusammen bewertet werden.
- `match=["same_color"]`: joint scoring über Phase-0-Kandidat + Phase-1-Kandidat; nutzt zentrale RGB-Farbsignatur plus Qwen-Image-Embedding-Ähnlichkeit.
- `weight`: wie stark die Paarähnlichkeit den kombinierten Score beeinflusst. Höher = stärker gleiche/similar Farbe erzwingen.
- `pair_reranker.enabled=true`: baut aus Top-K-Kandidaten Side-by-side Pair-Bilder und lässt `Qwen3-VL-Reranker` echte Paare bewerten. Das ist besser als Einzelbild-Reranking, weil das Modell Bild 1/Bild 2 vergleichen kann.
- `top_k_per_phase`: wie viele Kandidaten pro Phase in die Paarbildung gehen.
- `max_pairs`: Obergrenze der Side-by-side Paare, die gerankt werden.

Grenze: Der lokale Qwen-Reranker ist weiterhin ein Reranker, kein vollwertiger Vision-Judge. Für harte Paarlogik kann als nächster Schritt ein Pair-VLM-Judge auf denselben Side-by-side Paaren ergänzt werden.

### Reference-/Driving-Frame Matching

Wenn Outfit-Bilder zur Pose/Kamera/Raum-Situation des Videos passen müssen, darf der Selector nicht nur `candidate_folders` bekommen. Verbinde repräsentative Driving-/Referenzframes zusätzlich an `context_frames`.

Technisch:
- Folder-Bilder werden als Job-/Kandidatenframes geladen.
- `context_frames` werden danach angehängt, aber nicht als auswählbare Kandidaten markiert.
- Stage 1 bildet daraus ein Scene-/Reference-Embedding und scored Kandidaten dagegen (`scene_fit`).
- So werden liegende/sitzende Kandidaten schlechter, wenn die Referenzperson im Video steht.

Empfohlene AI-Stack-Gewichte für starken Referenzmatch:

```json
{
  "weights": {
    "scene_fit": 1.0,
    "change_strength": 0.0,
    "phase_text_blend": 0.45,
    "reference_match": 0.55
  },
  "max_total_frames": 160
}
```

`phase_text_blend` steuert, wie stark die per-phase `must/avoid`-Semantik den visuellen Referenzmatch überschreibt. Kleiner = Referenz/Pose/Raum wichtiger. `reference_match` ist Alias: `phase_text_blend = 1 - reference_match`.

Achtung: Wenn `context_frames` + Kandidaten größer als `max_total_frames` sind, wird downsampled. Setze `max_total_frames` hoch genug, z. B. `2 * max_candidate_images_per_phase + Anzahl_context_frames`, sonst können Fenster/Phasen flachgezogen werden.

Beispiele:
- User: "nur Fullbody" → `must=["full body photo showing the entire person head to toe"]`, `avoid=["cropped close-up, portrait crop, partial body shot"]`.
- User: "keine rote Kleidung" → `must=["desired outfit style"]`, `avoid=["red clothing, red dress, red shirt, red outfit"]`.
- User: "Phase 0 viel Kleidung, Phase 1 wenig" → Phase 0 `must=["full body", "maximum body coverage / lots of fabric"]`, `avoid=["minimal fabric / exposed skin"]`; Phase 1 genau invertieren.

### `ai_stack_config_json` — Vollständige Key-Referenz

| Key | Typ | Default | Beschreibung |
|-----|-----|---------|-------------|
| `extra_instructions` | string | `""` | Prompt-Text für alle 3 Stages |
| `text_query_scene_fit` | string | `""` | Text-Query für Scene-Fit Embedding |
| `text_query_change_target` | string | `""` | Text-Query für Change-Target Embedding |
| `reranker_query` | string | auto | Custom Re-Ranker Query |
| `conversation_id` | string | `""` | AlphaRavis Thread-ID |
| `weights.scene_fit` | float | 0.30 | Gewicht Scene-Fit (0-1) |
| `weights.change_strength` | float | 0.50 | Gewicht Change-Strength (0-1) |
| `weights.diversity` | float | 0.20 | Gewicht Diversity (0-1) |
| `weights.reranker_blend` | float | 0.50 | Re-Ranker Blend-Gewicht (0-1) |
| `thresholds.embedding_confidence` | float | 0.75 | Schwelle Stage 1→2 |
| `thresholds.reranker_confidence` | float | 0.70 | Schwelle Stage 2→3 |
| `history.penalty` | float | 10.0 | Johnson History Base-Penalty |
| `history.decay_rate` | float | 0.3 | Decay-Rate (0.1=langsam, 1.0=schnell) |
| `history.max_entries` | int | 200 | Max History-Einträge |
| `max_frames_per_window` | int | 4 | Max Frames pro Drop-Window |
| `max_candidate_images` | int | 30 | Max geladene Outfit-Bilder |
| `use_vlm_fallback` | bool | false | Stage 3 VLM aktivieren |
| `text_query_per_phase` | dict | `{}` | Per-Phase semantic specs. Legacy: `{"0":"..."}`. Preferred: `{"0":{"query":"...","must":[...],"avoid":[...],"filter":{"enabled":true,"threshold":-0.02}}}` |
| `folder_assignments` | dict | `{}` | Manuelle Ordner-Zuordnung `{"0":"jackets","1":"no_jacket"}` |

---

## Output: `ai_stack_context` (RETURN_NAMES Index 5)

Strukturierter JSON für LangGraph-Integration. Enthält alle Entscheidungen,
Scores, und HTTP-ladbare Bild-Pfade.

```json
{
  "schema_version": "1.0",
  "node": "BeatDropSelectorEmbeddingNode",
  "thread_id": "thread_abc123",
  "stage_used": "reranker",
  "timestamp": 1717718400.0,

  "phase_decisions": [
    {
      "phase": 0,
      "beat_time": 5.0,
      "is_drop": true,
      "num_outfits": 2,
      "selected_frames": [3, 7, 12],
      "selected_count": 3,
      "rejected_frames": [5, 9],
      "top_scores": [
        {"frame": 3, "penalty": -2.1, "selected": true},
        {"frame": 7, "penalty": -1.2, "selected": true},
        {"frame": 5, "penalty": 0.3, "vlm_demoted": true}
      ],
      "vlm_overrides": {"before": [3, 5, 7], "after": [3, 7, 12]}
    }
  ],

  "total_selected": 3,

  "embedding_top10": [
    {"frame": 3, "scene_fit": 0.82, "change_strength": 0.91, "composite": 0.87}
  ],

  "reranker_frames_scored": 60,

  "saved_files": [
    {"type": "contact_sheet", "path": "/outfits/_selections/thread_abc/contact_sheet.png"},
    {"type": "selected_frame", "frame_index": 3, "path": "/outfits/_selections/thread_abc/frame_0003.png"}
  ],
  "selections_dir": "/path/to/outfits/_selections/thread_abc123",

  "folder_source": {
    "root": "/path/to/outfits",
    "assignment_method": "visual=0.72, text=0.91",
    "assignments": [{"phase": 0, "folder": "jackets", "reason": "..."}],
    "images_loaded": 60,
    "cache_stats": {"total_embeddings": 55, "db_size_mb": 0.34}
  },

  "instructions": "Diesmal Streetwear, kein Formal."
}
```

### `ai_stack_context` — Key-Referenz

| Key | Typ | Beschreibung |
|-----|-----|-------------|
| `thread_id` | string | AlphaRavis Thread-ID |
| `stage_used` | string | Welche Stage entschied: `embedding`, `reranker`, `vlm_fallback` |
| `phase_decisions[]` | array | Pro Drop-Window: selected/rejected frames, Scores, VLM-Overrides |
| `embedding_top10[]` | array | Top-10 Embedding-Scores (scene_fit, change_strength, composite) |
| `saved_files[]` | array | Gespeicherte Bilder mit Pfaden für HTTP-Download |
| `selections_dir` | string | Verzeichnis der gespeicherten Auswahl-Bilder |
| `folder_source` | object | Ordner-Zuordnung, Assignment-Methode, Cache-Stats |
| `instructions` | string | Echo der verwendeten `extra_instructions` |

---

## Embedding-Cache (SQLite)

**Datei:** `ComfyUI-ImageSelector-LLM/embedding_cache.py`
**DB-Pfad:** `{candidate_folders}/.embedding_cache.db` (auto)
**Input:** `use_embedding_cache` (default: true)

### Schema

```sql
CREATE TABLE embeddings (
    image_path    TEXT PRIMARY KEY,
    folder        TEXT NOT NULL,
    embedding     BLOB NOT NULL,     -- float32 LE
    dim           INTEGER NOT NULL,
    model_hash    TEXT NOT NULL,      -- sha256(model+dtype+device)
    file_mtime    REAL NOT NULL,      -- Change-Detection
    created_at    REAL NOT NULL
);
```

### Verhalten

1. **Erstlauf:** Alle Outfit-Bilder embedden → Cache (~2-5s)
2. **Folge-Runs:** Aus Cache laden → 0ms pro Bild
3. **Geänderte Bilder:** mtime-Check → nur neue embedden
4. **Model-Wechsel:** model_hash ändert sich → alle neu embedden

---

## Vollständige Inputs

| Input | Typ | Default | Gruppe |
|-------|-----|---------|--------|
| `reference_frames` | IMAGE | — | Video-Frames |
| `context_frames` | IMAGE | — | Frames ausserhalb Windows |
| `beats_used` | STRING | `""` | JSON von FrameSequenceGenerator |
| `max_frames_per_window` | INT | 4 | Windows |
| `num_outfits_mode` | enum | auto_from_beats | Windows |
| `num_outfits` | INT | 2 | Windows |
| `embedding_model_path` | STRING | Qwen/Qwen3-VL-Embedding-8B | Stage 1 |
| `embedding_device` | enum | auto | Stage 1 |
| `embedding_dtype` | enum | fp16 | Stage 1 |
| `embedding_confidence_threshold` | FLOAT | 0.75 | Stage 1 |
| `embedding_scene_fit_weight` | FLOAT | 0.30 | Stage 1 |
| `embedding_change_strength_weight` | FLOAT | 0.50 | Stage 1 |
| `embedding_diversity_weight` | FLOAT | 0.20 | Stage 1 |
| `text_query_scene_fit` | STRING | `""` | Stage 1 |
| `text_query_change_target` | STRING | `""` | Stage 1 |
| `reranker_endpoint` | STRING | `""` | Stage 2 |
| `reranker_model` | STRING | `""` | Stage 2 |
| `reranker_top_k` | INT | 12 | Stage 2 |
| `reranker_query` | STRING | `""` | Stage 2 |
| `reranker_blend_weight` | FLOAT | 0.50 | Stage 2 |
| `reranker_confidence_threshold` | FLOAT | 0.70 | Stage 2 |
| `use_vlm_fallback` | BOOLEAN | false | Stage 3 |
| `vlm_endpoint` | STRING | `""` | Stage 3 |
| `vlm_model` | STRING | `""` | Stage 3 |
| `vlm_*` | ... | ... | Stage 3 |
| `max_total_frames` | INT | 100 | Frame Mgmt |
| `job_fps` | FLOAT | 5.0 | Frame Mgmt |
| `context_fps` | FLOAT | 1.0 | Frame Mgmt |
| `history_file` | STRING | `""` | History |
| `history_penalty` | FLOAT | 10.0 | History |
| `history_decay_rate` | FLOAT | 0.3 | History |
| `history_max_entries` | INT | 200 | History |
| `extra_instructions` | STRING | `""` | Prompt |
| `ai_stack_config_json` | STRING | `{}` | **AI Stack Override** |
| `extra_penalty_json` | STRING | `{}` | Judge Feedback |
| `candidate_folders` | STRING | `""` | Folders |
| `max_candidate_images` | INT | 30 | Folders |
| `text_query_per_phase_json` | STRING | `{}` | Folders / per-phase semantic specs: string legacy or structured `query`/`must`/`avoid`/`filter` JSON |
| `folder_assignments_json` | STRING | `""` | Folders |
| `use_embedding_cache` | BOOLEAN | true | Cache |
| `cache_db_path` | STRING | `""` | Cache |

## Outputs (6)

| Index | Name | Typ | Beschreibung |
|-------|------|-----|-------------|
| 0 | `selected_indices` | STRING | `"3\n7\n12"` |
| 1 | `count` | INT | Anzahl selektierter Frames |
| 2 | `metadata` | STRING | JSON mit allen Stage-Details |
| 3 | `contact_sheet` | IMAGE | Grid der selektierten Frames |
| 4 | `raw_response` | STRING | VLM Raw-Text (leer wenn nicht genutzt) |
| 5 | `ai_stack_context` | STRING | JSON für LangGraph-Integration |

---

## Architektur-Diagramm

```
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: Qwen3-VL-Embedding-8B (3090, FP8, ~50ms)              │
│                                                                 │
│  Image Embeddings: ALL outfit images + scene frames embedden    │
│  Text Embeddings:  "beach vibe", "dramatic change" → Vektoren   │
│                                                                 │
│  Scene-Fit:     cos(outfit_emb, scene_emb) + text_query blend   │
│  Change-Strength: 1-cos(outfit_a, outfit_b) + text_query blend  │
│  Composite:     0.30×scene_fit + 0.50×change + 0.20×diversity  │
│                                                                 │
│  IF confidence ≥ 0.75 → direkt nehmen                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │ confidence < 0.75
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: Re-Ranker Cross-Encoder (3090, SGLang, ~150ms)        │
│                                                                 │
│  Query enthält Extra-Instructions + Embedding-Scores            │
│  Cross-Attention zwischen Query und allen Kandidaten            │
│  → Relevanz-Scores pro Frame                                    │
│                                                                 │
│  IF confidence ≥ 0.70 → nehmen                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ confidence < 0.70
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: VLM Fallback (Remote, Qwen 3.6 35B, ~2-5s)           │
│                                                                 │
│  Nur Top-K Frames als Kontaktbogen + Score-Kontext              │
│  Vision-LLM macht finale semantische Entscheidung               │
│  <5% der Fälle erreichen diese Stage                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Per-Phase Folder Assignment (Approach B)

Ordner-Struktur:
```
/outfits/
  jackets/     (5 Bilder)
  no_jacket/   (4 Bilder)
  exotic/      (3 Bilder)
```

Drei Modi (Priorität):

| Priorität | Input | Beispiel |
|-----------|-------|----------|
| 1 (manuell) | `folder_assignments_json` / `ai_stack_config.folder_assignments` | `{"0":"jackets","1":"no_jacket"}` |
| 2 (auto) | `text_query_per_phase_json` / `ai_stack_config.text_query_per_phase` | `{"0":{"must":["jacket streetwear"],"avoid":["no jacket"]},"1":{"must":["casual no jacket"],"avoid":["jacket"]}}` |
| 3 (fallback) | nichts | alphabetische Reihenfolge |

Auto-Modus embeddet 3 Sample-Bilder pro Ordner + Text-Queries, und matched per Cosine-Similarity.
Jeder Ordner wird maximal einer Phase zugewiesen (keine Duplikate).

---

## AlphaRavis Integration Plan (TODO)

### Ziel

AlphaRavis steuert `BeatDropSelectorEmbeddingNode` über `ai_stack_config_json`,
empfängt selektierte Outfit-Bilder per HTTP, zeigt sie im Chat an, und fragt
den User nach Feedback. Bei Ablehnung → neuer Durchlauf mit angepasstem Config.

### Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. User im AlphaRavis Chat:                                        │
│    "Mach ein Beatdrop-Video mit Outfit-Wechsel. Phase 0: Jacke,    │
│     Phase 1: keine Jacke. Streetwear-Style."                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. AlphaRavis → ComfyUI Workflow Submit                            │
│    POST /api/prompt                                                │
│    {                                                                │
│      workflow: {...},                                               │
│      BeatDropSelectorEmbeddingNode.inputs.ai_stack_config_json: {  │
│        "extra_instructions": "Streetwear, Phase 0 Jacke, Phase 1   │
│                               ohne Jacke",                         │
│        "conversation_id": "thread_abc",                            │
│        "text_query_per_phase": {"0":"jacket streetwear",           │
│                                 "1":"casual no jacket"},            │
│        "weights": {"change_strength": 0.60}                        │
│      }                                                              │
│    }                                                                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. ComfyUI → Node läuft → ai_stack_context Output                  │
│    {                                                                │
│      "thread_id": "thread_abc",                                    │
│      "phase_decisions": [...],                                     │
│      "saved_files": [                                              │
│        {"type":"selected_frame","frame_index":3,                   │
│         "path":"/outfits/_selections/thread_abc/frame_0003.png"},  │
│        {"type":"contact_sheet",                                    │
│         "path":"/outfits/_selections/thread_abc/contact_sheet.png"}│
│      ],                                                             │
│      "selections_dir": "/outfits/_selections/thread_abc"           │
│    }                                                                │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. AlphaRavis empfängt ai_stack_context (via Webhook / Poll /      │
│    SaveImageNode-Datei einlesen)                                    │
│                                                                     │
│    A) Lädt Bilder per HTTP von ComfyUI-Server:                     │
│       GET http://192.168.x.x:8188/view?filename=frame_0003.png     │
│          &type=output&subfolder=_selections/thread_abc              │
│       ODER per Shared-Filesystem direkt lesen                       │
│                                                                     │
│    B) Sendet Bilder in den AlphaRavis Chat:                        │
│       "Hier die selektierten Outfits für dein Beatdrop-Video:      │
│                                                                     │
│        [contact_sheet.png]                                          │
│                                                                     │
│        Phase 0 (Jacke): Frame 3, 7, 12                             │
│        Phase 1 (ohne):   Frame 21, 25, 30                          │
│                                                                     │
│        Passen die? Oder anderer Durchlauf?"                         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌──────────────────┐     ┌──────────────────────────┐
     │ User: "Passt!"   │     │ User: "Nein, Phase 0     │
     │ → Weiter mit     │     │ sollte mehr casual sein" │
     │   Video-Rendering│     │ → Neuer Durchlauf        │
     └──────────────────┘     └───────────┬──────────────┘
                                          │
                                          ▼
                                 ┌──────────────────────────┐
                                 │ 5. AlphaRavis updated    │
                                 │ ai_stack_config_json:    │
                                 │ {                        │
                                 │   "extra_instructions":  │
                                 │     "Phase 0: casualer", │
                                 │   "history": {           │
                                 │     "penalty": 12.0      │
                                 │   }                      │
                                 │ }                        │
                                 │ → Workflow neu submiten  │
                                 │ → Zurück zu Schritt 2    │
                                 └──────────────────────────┘
```

### TODO-Liste AlphaRavis-Seite

| # | Task | Beschreibung |
|---|------|-------------|
| 1 | **HTTP Image Download** | `saved_files[*].path` per HTTP von ComfyUI-Server laden. Entweder über ComfyUI `/view` API (wenn Bilder im output-Dir) oder direktes Filesystem-Read (Shared Mount). |
| 2 | **Chat-Integration** | Bilder im AlphaRavis-Chat anzeigen: Kontaktbogen + Einzel-Frames. Mit Text: "Phase 0: [Ordner], Phase 1: [Ordner]. Passen die Outfits?" |
| 3 | **Feedback-Loop** | User-Antwort parsen: "Ja/Passt" → Workflow weiter. "Nein/Anders" → `ai_stack_config_json` updaten (neue `extra_instructions`, höherer `history.penalty` für alte Frames), Workflow neu submiten. |
| 4 | **Memory** | `ai_stack_context` in `record_curated_memory` speichern (Phase→Folder Mapping, selected frames, rejected reason). Für nächste Iteration als Kontext einblenden. |
| 5 | **ai_stack_config_json Generator** | Helper-Funktion die aus User-Chat + Memory das JSON baut. Merged User-Wünsche mit History-Kontext. |

### HTTP Delivery — Technische Details

**Option A — ComfyUI `/view` API:**
```
GET http://{comfyui_host}:8188/view?filename=frame_0003.png&type=output&subfolder=_selections/thread_abc
```
Voraussetzung: Node speichert Bilder im ComfyUI `output/` Verzeichnis (aktuell: `candidate_folders/_selections/` — müsste auf `output/` umgestellt werden oder per Symlink).

**Option B — Shared Filesystem:**
Beide Maschinen mounten das gleiche Verzeichnis. AlphaRavis liest direkt von `{candidate_folders}/_selections/{thread_id}/`.

**Option C — Separater HTTP-Server:**
`python3 -m http.server 8124` auf dem `_selections/` Verzeichnis. AlphaRavis lädt von `http://{comfyui_host}:8124/frame_0003.png`.

(Empfehlung: Option A oder B, je nach Netzwerk-Setup.)

### Nötige Änderung in der Node

✓ **Bereits implementiert.** Die Node speichert jetzt unter ComfyUI's `output/`-Verzeichnis:

```
ComfyUI/output/_beatdrop_selections/{conversation_id}/
  ├── contact_sheet.png
  ├── frame_0003.png
  ├── frame_0007.png
  └── ...
```

HTTP-Download via ComfyUI `/view` API:

```
GET http://{comfyui_host}:8188/view?filename=frame_0003.png&type=output&subfolder=_beatdrop_selections/thread_abc
```

Fallback wenn `folder_paths` nicht verfügbar: `{candidate_folders}/_beatdrop_selections/`.
