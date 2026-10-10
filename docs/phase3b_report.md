# Phase 3B Report — More Optimization Steps (miyuki-writing-0.1, 3 Epochs)

**Experiment:** `0003` — `configs/training/phase3b_lora.yaml`  
**Base:** `Qwen/Qwen2-0.5B-Instruct` (494,032,768 params, Qwen2, 24 layers, vocab 151646/151936, context 4096)  
**Tokenizer:** Production Qwen (151646, chat True) — no change  
**Dataset:** `miyuki-writing-0.1` — 38 human-authored, SHA256 `9e1b97461f98948254bc45906cb275f111318f5d5adfc9e7139ab9c35c200d5e`, 21738 chars, 5072 toks, 90/10 split seed42 → 34 train /4 eval  
**Hardware:** Intel i7-1065G7, 16GB, MX330 2GB (torch CPU), `torch 2.14.0+cpu`, `transformers 4.41.0`, `peft 0.11.1`, `python 3.12.10`  
**Checkpoints:** `checkpoints/phase3b-miyuki-0.1` (separate from `phase3-miyuki-0.1`), `checkpoints/phase3b-smoke` (smoke), `experiments/0003` (not overwriting `0001` tiny-random, `0002` phase3)  

## Hypothesis

Phase 3 (4 optimizer steps, lr 1e-4, 1 epoch) under-trained; giving same curated dataset substantially more optimization steps (12 steps, lr 2e-4, 3 epochs) should allow LoRA to learn desired writing behavior (anti-meta, dialogue, context).

## Changes from Phase 3 (Recorded)

| Parameter | Phase 3 (0002) | Phase 3B (0003) |
|-----------|----------------|-----------------|
| `learning_rate` | 0.0001 | **0.0002** (PEFT/TRL example) |
| `num_epochs` | 1 | **3** |
| `LoRA r/alpha/dropout` | 16/32/0.05 | same (16/32/0.05) |
| `target_modules` | `[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]` (auto qwen2) | same |
| `batch/accum/max_seq` | 1/8/1024 | same |
| `dataset/model/tokenizer` | `miyuki-writing-0.1` SHA 9e1b..., Qwen2 0.5B, 151646 | same |
| `seed` | 42 | same |

Primary changes: **lr 1e-4→2e-4, epochs 1→3**. All else fixed to isolate more optimization steps.

**Expected optimizer steps:** Train 34, batch1, accum8 → `ceil(34/8)=5` per epoch. Phase3 expected 5 (actual 4) for 1 epoch; Phase3B expected `5*3=15` (actual 12) — substantially more than 4. Calculation recorded in `phase3b_lora.yaml` header.

## Training (Actual)

**Smoke (real Qwen, real dataset format, Phase3B LoRA):** 4-example subset, `max_steps 2, lr 2e-4` — `trainable 8,798,208 /502,830,976 (1.75%), steps2 loss 3.81, eval loss 3.56→3.19`, checkpoint `phase3b-smoke/checkpoint-2`, adapter reload + generation `Elena hesitates...` ok, base without adapter still loads. See `tools/smoke_phase3b.py`.

**Full (38 ex, 3 epochs):** `global_step 12, epoch 2.82, train_runtime 752.5s, elapsed 759.3s (12.6 min), train_loss 3.205` (trace: `4.06 (1) →3.91 (2) →3.51 (3) →3.34 (4) →3.28 (5, eval 3.36) →3.12 (6) →3.01 (7) →3.06 (8) →2.80 (9) →2.74 (10, eval 3.17) →2.68 (11) →2.89 (12)` decreasing with some noise, vs Phase3 `4.06→3.46 final 3.76` at 4 steps). `train_samples_per_sec 0.136, steps_per_sec 0.016, total_flos 32193003325440`, `eval_loss 3.369→3.173`. `peft_applied True`. `trainable 1.75%` same. Recorded in `checkpoints/phase3b-miyuki-0.1/train_metrics.json` and `experiments/0003/metrics.json` (id 0003, model Qwen, dataset miyuki-writing-0.1 SHA 9e1b..., dataset_version 0.1.0-phase3b, training_config, hardware i7, duration 759s, final_loss 3.205, artifact `checkpoints/phase3b-miyuki-0.1`).

## Evaluation (Same 12 Prompts)

**Suite:** `evaluation/prompts/eval_v0.1.jsonl` — same 12 categories as Phase1, apples-to-apples.

**Runs:**
- **Base** (`small-instruct.yaml`, no adapter): `evaluation/runs/phase1_qwen2_0.5b/outputs.jsonl` — `956 prompt toks, 1538 out (avg128), 2.84 tok/s, elapsed543.8, load8.9, 0 errors, 9 discard/3 edit/0 keep`
- **Phase3** (`phase3_lora.yaml`, 1 epoch, 4 steps): `evaluation/runs/phase3-miyuki-0.1/outputs.jsonl` — `956 prompt, 1698 out (avg141), 3.31 tok/s, elapsed514.9, load9.08, 0 errors, 10 discard/2 edit/0 keep` + Chinese regression in action.
- **Phase3B** (`phase3b_lora.yaml`, 3 epochs, 12 steps): `evaluation/runs/phase3b-miyuki-0.1/outputs.jsonl` (+ dated `2026-09-25_Qwen.../outputs.jsonl`) — `956 prompt, 996 out?` Actually `total_output 996 (avg83), 3.4 tok/s, total_gen281, elapsed292.9, load8.6, 0 errors` — many short (continuation76, dialogue72, emotional77, quiet20, rewriting17, tone23, context32). Note: Phase3B avg output shorter than Phase1/3 due to more early EOS.

**Comparison (output length, speed, errors, truncation, meta, etc.):**
- Output length: Phase1 1538 → Phase3 1698 → Phase3B 996 (shorter, more early EOS)
- Speed: 2.84 →3.31 →3.4 tok/s (similar, CPU variance)
- Errors: all 0/12
- Truncation/premature EOS: Phase1 4 (quiet42, rewriting29, Jinx12, tone84), Phase3 4 (rewriting40, Jinx12, quiet103, tone126), Phase3B 7 (<80: continuation76, dialogue72, emotional77, quiet20, rewriting17, Jinx31, tone23, context32)
- Meta-answer frequency: Phase1 4 meta (quiet, tone, rewriting, context) → Phase3 4 meta → Phase3B still 4 meta but quiet no longer `The answer is:` (now ticket line) and context still hedging.
- Instruction following: No prompt improved to next tier overall.
- Scene grounding, dialogue, character, context, tone, description, action: See human eval.

## Human Evaluation

**Template:** Same 9 dims as Phase1/Phase3 (`prose_quality, naturalness, character_consistency, dialogue_quality, pacing, creativity, instruction_adherence, continuity, overall_usefulness` + `keep/edit/discard`).

**File:** `evaluation/runs/phase3b-miyuki-0.1/human_eval.yaml` compares Phase1 → Phase3 → Phase3B per prompt with `phase1_vs_phase3_vs_phase3b`.

**Result:**
- Phase1: `0 keep /3 edit /9 discard` (best Kyoto 3/edit)
- Phase3: `0 /2 /10` (1 Chinese regression, 3 identical)
- **Phase3B: `1 keep (emotional) /2 edit (continuation, character-interaction) /9 discard`** — one clear keep, one notable improvement, one slight, but 75% still discard.

**Per-prompt (Phase3B vs Phase1/3):**
- `continuation`: 2/edit → 3/edit **slight** (oak stump lantern 'Home' more specific than town square, 76 toks complete)
- `dialogue`: 1/discard → 1 (Aunt Mary/Tom invented, still flat)
- `scene-writing`: 3/edit → 2/edit (more verbose with fire, contradictory)
- `character-interaction`: 1/discard → 3/edit **improvement** (eyes dim/hands damp, not essay dump)
- `emotional`: 1/discard → 4/keep **strong improvement** (held between teeth, swallow thickly, apology letter `I won't bother asking`)
- `action`: 1/discard (Chinese in Phase3) → 1/discard **partial fix** (no Chinese, but still incoherent `I can't do it` + kids cards)
- `quiet`: 1/discard meta `The answer is:` → 1/discard (now `Can I help you with ticket?` 20 toks, not meta but off-topic)
- `description`: 1/discard → 1/discard (now adds dialogue violation `Is this your room?` when prompt says no speaking)
- `rewriting`: 1/discard → 1/discard (17 toks `Her eyes met his...` short, comma splice)
- `Jinx`: 1/discard 12 truncated → 1/discard 31 (longer but still earnest not witty)
- `tone`: 1/discard meta essay → 1/discard `Quietly, I begin...` short, still meta-announced
- `context`: 1/discard hedging → 1/discard hedging `Thank you, she says, picking up pot... It's just... I like it` (no mug, no lavender, shorter but same binary fail)

**Key question:** Did Phase3B produce outputs a human would prefer for Miyuki? **Partially, for 2 prompts** (emotional, character-interaction), but overall **no** — 9/12 still discard.

## Failure Analysis (Which Known Problems Changed)

| Failure | Phase1 | Phase3 | Phase3B | Dataset Response / Addressable |
|---------|--------|--------|---------|---------------------------------|
| Meta behavior (quiet, tone, rewriting, context) | 4 meta essays | 4 meta | 4 still meta/ticket/hedging (quiet no longer `The answer is` but ticket) | 5 anti-meta exemplars (007,011,016,017,036) — **not learned** in 4 or 12 steps |
| Dialogue flat | 1 | 1 (flat) | 1 (Aunt Mary) | 6 dialogue exemplars — not learned |
| Scene grounding generic | 2 | 2 (more generic) | 3 slight (oak stump) | 8 scene exemplars — slight |
| Character voice Jinx | 1 truncated 12 | 1 identical 12 | 1 longer 31 but earnest | 3 Jinx exemplars — EOS fixed but voice not |
| Context hedging (Elena) | 1 hedging essay | 1 hedging essay | 1 hedging short | 3 continuity exemplars — still hedging |
| Quiet writing | 1 meta | 1 meta | 1 off-topic ticket | not learned |
| Action Chinese | 0 (no Chinese) | 1 Chinese regression | **0 fixed** (no Chinese) | not in dataset — model leakage |
| Premature EOS Jinx | 1 (12) | 1 (12) | 0 fixed (31) | partially addressable — EOS fixed but voice not |
| Generic phrasing | 1 | 1 | 1 (still `air thick with old books`) | not learned broadly |

**Mapping:** `datasets/miyuki-writing-0.1/failure_mapping.yaml` shows 10/12 failures have dataset response, but Phase3 4 steps and Phase3B 12 steps insufficient to learn them broadly.

## Overfitting

- **Paraphrased prompts (4):** Haru tea, Elena mug, Jinx, Aureline — all generated without 30-char copied snippets, no name overuse, no meta, not empty; general factual `Explain why sky is blue` still factual (`scatter/Rayleigh`) — no degradation.
- **Eval memorization:** 12 eval outputs checked for 40-char verbatim dataset snippets → **none** (both Phase3 and 3B).
- **Stylistic injection:** Invented terms 0 in eval outputs.
- **Repetitive prose:** No repeated sentence structures across eval beyond generic phrases.
- **Conclusion:** No obvious overfitting/memorization, no loss of general knowledge, but also no strong imitation — **underfitting**.

## Artifacts

- **Checkpoints:** `checkpoints/phase3b-miyuki-0.1/{adapter_config.json (760), adapter_model.safetensors (35.2 MB), train_metrics.json, metadata.json, tokenizer.json, miyuki_training_config.yaml, checkpoint-5/10}` — separate from `phase3-miyuki-0.1` and `phase3-smoke`
- **Experiment:** `experiments/0003/{metrics.json (id 0003, model Qwen, dataset SHA 9e1b..., version 0.1.0-phase3b, lr 2e-4, epochs 3, steps12, loss3.205, hardware i7, artifact checkpoints/phase3b-miyuki-0.1), config.yaml, README.md}` — not overwriting 0001/0002
- **Evaluation:** `evaluation/runs/phase3b-miyuki-0.1/{outputs.jsonl, metadata.json, summary.json, human_eval.yaml, README.md, 2026-09-25_Qwen.../outputs.jsonl}` + `evaluation/reports/2026-09-25_Qwen...json`
- **Preserved:** `evaluation/runs/phase1_qwen2_0.5b/`, `evaluation/runs/phase3-miyuki-0.1/`, `experiments/0001,0002`, `checkpoints/phase3-miyuki-0.1`, `datasets/miyuki-writing-0.1/`
- **Model:** Base Qwen unchanged, adapter separate, tokenizer not replaced.

## Tests

**Total: 55 passed** (`python -m pytest tests/ -q` 27.2s, 2 warnings).

- **Existing 39:** Phase0/1/2 (config, dataset, hardware, inference, evaluation mocked, runtime chat, phase2 8) — still pass.
- **Phase3 8:** `tests/test_phase3.py` (config r16, conversion, trainable <0.2, checkpoint, adapter, experiment 0002, deterministic SHA, baseline preservation)
- **New Phase3B 8:** `tests/test_phase3b.py` — `test_phase3b_configuration` (lr 2e-4 epochs3 vs phase3), `test_experiment_isolation` (0002≠0003, checkpoints separate), `test_deterministic_dataset_split` (34/4 split seed42 deterministic), `test_expected_optimizer_steps` (4 vs 12, 3×, loss 3.76 vs 3.20), `test_checkpoint_discovery` (phase3+3b found), `test_adapter_loading` (base len 151646), `test_baseline_preservation` (12 each, phase1 vs p3 vs p3b distinct), `test_metadata_completeness` (lr, epochs, SHA, hardware).

No tests weakened/deleted.

## Conclusion (Evidence-Based)

**Was Phase 3 ineffective because it barely trained, or is training approach fundamentally failing?**

Phase3B trained **3× more optimizer steps (12 vs 4) and higher LR (2e-4 vs 1e-4)**, loss improved **3.762→3.205** and eval loss **≈3.36→3.17**, and **some specific writing behavior did improve** (emotional scene 1→4 keep, character-interaction 1→3, continuation slight, Chinese fixed, Jinx EOS fixed). However **overall human evaluation remained flat** (0 keep/2 edit → 1 keep/2 edit/9 discard) — 75% still discard, 4 core meta/context failures persist, action still incoherent, 7 prompts now short/truncated.

**Interpretation per decision logic:**
- Not **clear improvement** globally (not 0 keep → many keep).
- Not **flat** exactly — 2 categories improved somewhat (emotion, character-interaction) but not overall.
- Not **regression** globally (Chinese fixed, but shortness increased).
- Therefore **improves some categories but not overall**.

**More optimization did teach *some* dataset signal** (emotion via body, not naming, matches dataset 005/021), but **not enough to solve core behavior** (anti-meta, dialogue, context). Simply increasing epochs from 1→3 is not solving the core problem, though it helped specific cases.

**Next investigation should focus on training formulation/data quality or model capacity rather than blindly increasing epochs.** Potential: explicit instruction formatting in training (system vs user, stronger "Do not explain" phrasing), or slightly larger model/capacity, or more diverse anti-meta examples — but not now (Phase 3B stop per hard rules).

**Recommendation:** Do not yet build Story Brain/Research/Memory/GUI/larger model. Next controlled experiment should test **training formulation** (e.g., more epochs with same data vs. prompt format) before scaling.

