# Miyuki Writing Corpus 0.2

**Version:** `miyuki-writing-0.2` — expanded from 0.1 (38) + 82 new targeted exemplars = 120 total.  
**Source:** human-authored, human-edited, reviewed, `CC-BY-4.0`.  
**Validation:** `120/120 valid, 0 warnings, 0 duplicates, 0 tiny`, tokenization OK (16088 toks, 134.1 avg, max 233, 57558 chars, 6318 words), SHA `8cda31a91...` (original preserved SHA `9e1b97...`).  
**Purpose:** Fix measured failures from Phase 1–3C (anti-meta, dialogue, character voice, context retention, rewriting, quiet, emotional, action, description, transitions).

**Structure:**
```
datasets/miyuki-writing-0.2/
  dataset.jsonl (120 conversational: user → assistant)
  manifest.json (120, categories, tokens, splits train108/eval12)
  schema.json (same as 0.1)
  report.json (token/category counts)
  review.yaml (120 keep)
  DATASET_SPEC.md
  README.md (this)
```

**Categories (120):** `scene 98, quiet 26, anti_meta 32, description 40, pacing 36, character_consistency 32, dialogue 24, subtext 25, character_voice 19, context_retention 20, rewriting 7, transitional 13, emotion 12, action 9, atmospheric 4, etc.` (see `manifest.json`/`report.json`).

**Training:** Use with `assistant_only_loss True` (Phase 4) — only assistant tokens trainable (system/user masked). Same as 0.1 but larger, deliberately targeted.

**Usage:**
```powershell
python -m miyuki.cli dataset validate --path datasets/miyuki-writing-0.2/dataset.jsonl --format conversational
python tools/manifest_0_2.py
```

**Preservation:** `miyuki-writing-0.1` remains byte-for-byte (SHA `9e1b9...`), 0.2 is derived, not in-place modification.
