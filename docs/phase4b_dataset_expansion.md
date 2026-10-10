# Phase 4B — Targeted Dataset Expansion (`miyuki-writing-0.3`)

**Status: PASS (dataset only, no training).** 120 → **154 examples** (+34 targeted). `miyuki-writing-0.2` preserved byte-for-byte.

## Why the Expansion Was Needed
Phase 4 (0.5B + LoRA-120, assistant-only, 39 steps, loss 2.62) moved original benchmark from 0/3/9 (base) to **5 keep / 4 edit / 3 discard** and eliminated meta behavior (0 vs 4). Remaining failures are specific, not global: flat dialogue with interchangeable voices, binary context facts missed (mug/lavender, compass, scarf, flag, ribbons, bag, scryglass), Jinx earnest instead of witty (109 tokens but mission-speak), action object/state untracked (notebook never retrieved), viewpoint constraints violated (adult summary instead of child/limited perception). 1.5B base (2/5/5) is more fluent but worse on tone/context/description + refusal risk; its micro-LoRA was correctly stopped at memory preflight (2.82 < 3.0 GB). Evidence supports more targeted 0.5B data before any scaling or training.

## Phase 4 Strengths (Preserved)
Quiet/subtext (bench-whispers), scene grounding (damp teacup), description POV (lantern/stream/backpack-27), rewriting (bed/hands/click), restrained tone (kettle/paper), no hedging essays, no `The answer is`, no Chinese in original 12. All 120 preserved examples remain byte-for-byte (first 120 lines identical, SHA `8cda31...` verified).

## Remaining Weaknesses (Targeted)
Dialogue distinct voices, binary context retention, Jinx witty voice, action object tracking, viewpoint/age/knowledge constraints — mapped from `human_eval.yaml` (dialogue/Jinx/context discard) and expanded eval (1/11 base, flat voices).

## Target Categories (34 New, IDs 121–154)
- **A character voice (8, 121–128):** same situation, noticeably different voices (spare vs run-on, bursts vs explanatory, proverb vs breathless, internal vs blunt, dry vs earnest, over-talk vs silent task, counting vs stillness, precise vs joke). Tags: `distinct_voice`.
- **B exact context (8, 129–136):** binary pairs notebook/phone, upstairs/downstairs, morning/evening, grandmother/mother, chamomile/lavender-accept vs lavender-hate, SW/river, station/market, red-high/green-low, ribbons-only-where-tied, heavy/light, warm/cold. Natural scenes where fact matters, including continuations. Tags: `binary_context`.
- **C Jinx/witty (6, 137–142):** quick dry one-joke-then-competent (map, fog, tide, scared, bag, wait), short punchy, loyal underneath, no catchphrase repeat. Tags: `witty_voice`.
- **D action+object (6, 143–148):** notebook/keys/chair/cup/drawer/phone — establish → action → tracked location/state (under-table dusty, wet-stones leaf, middle-room shin, knee-height hold, open-drawer stripe, one-hand pour). Scene-state tracking, harmless everyday actions. Tags: `object_tracking`.
- **E viewpoint (6, 149–154):** child height-literal (hands, kill-time), first-person limited (doesn't know call), third-limited (no motive), restrained task-only (no feeling words), smell-first ordering, vendor-limited (prices/weather only). Demonstrated, not instructed. Tags: `viewpoint_constraint`.

## Review Results
`review.yaml`: **154 keep (120 preserved + 34 new, 0 edit, 0 discard)**. Preserved 120 reference 0.2 review; new 34 each rated 5/clarity/quality/behavior/category/adherence/character/context. Rejected: none (all 34 survived; no padding — count is exact 34, within 30–40 target). Anti-pattern labels (`distinct_voice`, `binary_context`, `witty_voice`, `object_tracking`, `viewpoint_constraint`) in `notes` metadata only, never in assistant targets (verified by test).

## Tokenizer Statistics (Production Qwen2-0.5B, 151646, chat True — NOT replaced)
- Total 21027 full-conversation tokens, avg 136.5/ex, max 233/1024 (22.8% window, safe); new 34 total 4939, avg 145.3, max 184.
- 75731 chars, 8217 assistant-words; tok/word ~2.56 counts full-conversation tokens per assistant-word (prompts included), consistent with 0.2 methodology.
- Split (deterministic seed42 90/10): train 138 / eval 16. Expected 1-epoch optimizer steps at batch1/accum8: ~17 (138/8=17.25).

## Validation Results
`154/154 valid, 0 invalid, 0 warnings, 0 duplicates, 0 empty, 0 tiny` (`miyuki dataset validate --format conversational`). IDs unique (154). Near-duplicate scan (normalized assistant ratio >0.75): **0 pairs**. First 120 lines identical to 0.2. Schema `^miyuki-0\.[123]-\d{3}$` (0.3 IDs `miyuki-0.3-121..154`).

## Dataset SHA
- Old (0.2): `8cda31a91a76b1d6e8dc67ea2b779de06334836dad089b6bd59210dbdc6985d6` (120 ex) — unchanged, verified.
- New (0.3): `f3f5f636ea6d412d0d06d99a0a19df743105334ea9bdc7fccad19a418759f635` (full in `manifest.json` + `data/manifests/miyuki-writing-0.3.json`).

## Recommended Next Training Configuration (NOT Run Here)
Same proven Phase 4 shape to isolate data effect: base `Qwen/Qwen2-0.5B-Instruct`, dataset `miyuki-writing-0.2` → `miyuki-writing-0.3`, `assistant_only_loss: true`, LoRA r16/alpha32/dropout0.05/bias-none/targets qwen2-7, batch1/accum8/max1024/lr2e-4/seed42/CPU float32, **3 epochs** (~51 optimizer steps at 138 train: 17/epoch ×3, vs Phase4 39). Example config name: `configs/training/phase4c_lora_03.yaml` with `output_dir checkpoints/phase4c-miyuki-0.3` (next experiment 0007 when hardware allows ~50 min). Evaluate on same original 12 + expanded 12 before any backend switch.

## Files
`datasets/miyuki-writing-0.3/{dataset.jsonl (154), manifest.json, report.json, review.yaml, schema.json, DATASET_SPEC.md, README.md}` + `data/manifests/miyuki-writing-0.3.json`; tools `create_miyuki_0_3.py`, `manifest_0_3.py`, `gen_review_0_2.py` (reused), `check_near_duplicates_03.py`; tests `tests/test_phase4b.py`.
