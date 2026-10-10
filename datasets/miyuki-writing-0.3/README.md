# Miyuki Writing Corpus 0.3

**Version:** `miyuki-writing-0.3` — Phase 4B targeted expansion: 120 from 0.2 (SHA `8cda31...`) + 34 new = **154 total**.
**Source:** human-authored, human-edited, reviewed, `CC-BY-4.0`. No synthetic filler, no training run in this milestone.
**Validation:** `154/154 valid, 0 warnings, 0 duplicates, 0 tiny`, tokenization OK (21027 toks full-conversation, avg 136.5, max 233/1024; new 34 avg 145.3 max 184), SHA `f3f5f636ea6d412d...`. Note: tok/word ~2.56 counts full-conversation tokens per assistant-word (prompts included), consistent with 0.2 methodology.

**New targets (34):**
- **A character voice (8, 121–128):** distinct vocabulary/rhythm/reactions/humor/habits, same situation different voices (ferry, medicine, storm, hallway, hungry, apology, knock, spill). Tags: `distinct_voice`.
- **B exact context (8, 129–136):** binary facts notebook/phone, upstairs/downstairs, morning/evening, grandmother/mother, lavender/chamomile, SW/river, station/market, red/green, ribbons, heavy/light, warm/cold. Tags: `binary_context`.
- **C Jinx/witty (6, 137–142):** quick dry one-joke-then-competent, no catchphrase repeat (map, fog, tide, scared, bag, wait). Tags: `witty_voice`.
- **D action+object (6, 143–148):** notebook/keys/chair/cup/drawer/phone established → action → tracked location/state. Tags: `object_tracking`.
- **E viewpoint (6, 149–154):** child height-literal, first-person limited, third-limited, restrained task-only, smell-first, vendor-limited. Tags: `viewpoint_constraint`.

**Balance:** Supplements preserved coverage (scene/dialogue/description/quiet/emotion/action/pacing/subtext/voice/consistency/transitions/rewriting/retention/anti-meta). New cats: dialogue 9, voice 15, consistency 7, scene 25, subtext 8, description 10, quiet 6, humor 5, emotion 3, retention 8, continuity 2, pacing 4, action 6.

**Structure:** `dataset.jsonl (154)`, `manifest.json`, `report.json` (138 train/16 eval split), `review.yaml` (154 keep), `schema.json` (pattern `^miyuki-0\.[123]-\d{3}$`), `DATASET_SPEC.md`, `README.md` (this).

**Preservation:** `miyuki-writing-0.2` byte-for-byte (SHA `8cda31...` verified). 0.3 is derived, not in-place.

**Usage (next training, NOT run here):** Same Phase 4 config shape (r16 alpha32, batch1 accum8, max1024, lr2e-4, seed42, CPU, `assistant_only_loss: true`), 138/16 split → ~17 steps/epoch. Recommended: 3 epochs (~51 steps) to match Phase4 signal depth on larger data.
