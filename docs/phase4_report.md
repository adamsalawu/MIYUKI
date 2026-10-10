# Phase 4 Report — Dataset Expansion & Training Signal Audit

**Status: PASS (executed & measured) — substantial improvement on 7/12 original prompts; evidence supports continuing 0.5B pipeline before scaling.**

**Project:** `C:\Aether\RisingStar\MIYUKI`
**Base:** `Qwen/Qwen2-0.5B-Instruct` (494,032,768 params, Qwen2, 24 layers, vocab 151646 tokenizer / 151936 model, context 4096)
**Tokenizer:** Production Qwen (151646, chat True) — NOT replaced
**Dataset:** `miyuki-writing-0.2` — 120 human-authored (38 from 0.1 + 82 new targeted), SHA `8cda31a91a76b1d6e8dc67ea2b779de06334836dad089b6bd59210dbdc6985d6`, source SHA `9e1b97461f98948254bc45906cb275f111318f5d5adfc9e7139ab9c35c200d5e`
**Hardware:** Intel i7-1065G7 (4c/8t), 15.76 GB RAM, MX330 2 GB (torch CPU), `torch 2.14.0+cpu`, `transformers 4.41.0`, `peft 0.11.1`, `python 3.12.10`

## Training Signal Audit

**Before (Phase 0–3C): ALL tokens trainable.** `src/miyuki/training/train.py` used `texts = apply_chat_template(msgs)` → `tokenize` → `DataCollatorForLanguageModeling(mlm=False)` which sets `labels = input_ids` for all tokens. Diagnostic on `miyuki-0.1-001` (168 tokens): system+user prefix 93 tokens + assistant 75 tokens + 6 special tokens, all trainable, 0 masked. Model penalized for not predicting prompts, wasting ~55% loss. Saved to `evaluation/runs/phase4_loss_audit.json`.

**TRL expectation:** Modern TRL `SFTTrainer` supports assistant-only loss (only assistant responses contribute). MIYUKI did NOT use this — vanilla `Trainer` + collator.

**Qwen2 template supports masking:** Prefix `system+user+<|im_start|>assistant` is prefix of full (`full_ids[:93]==prefix_ids[:93]` True). No tokenizer change needed.

**Fix (smallest clean correction, preserved path):** `src/miyuki/config.py:TrainingConfig` added `assistant_only_loss: bool = False` (default preserves 0001–0005). `src/miyuki/training/train.py` when True builds `Dataset(text,_idx)`, `tokenize_fn_masked` computes `prefix_text = apply_chat_template(msgs[:-1], add_generation_prompt=True)`, `mask_len = min(len(prefix_ids),len(full_ids))`, `labels = [-100]*mask_len + full_ids[mask_len:]`, custom `assistant_collator` pads labels with `-100` (instead of overwriting with `input_ids`). Specials before assistant (5/6) masked, final `<|im_end|>` trainable. Verified in `tools/training_target_test.py` → `evaluation/runs/phase4_loss_audit_diagnostic.json` (168 total, 93 masked, 75 trainable, no prompt becomes target, no assistant masked). See `docs/training_signal_audit.md`.

## Dataset — `miyuki-writing-0.2`

**Version:** 0.2.0, 120 examples (38 from 0.1 byte-for-byte + 82 new), `CC-BY-4.0`, human-authored/human-edited/reviewed, all `keep` in `review.yaml`.
**Counts:** 57558 chars, 6318 words, 16088 Qwen tokens, avg 134.1/example, max 233/1024 (22.8% window, safe).
**Categories:** scene 98, quiet 26, anti_meta 32, description 40, pacing 36, character_consistency 32, dialogue 24, subtext 25, character_voice 19, context_retention 20, rewriting 7, transitional 13, emotion 12, action 9, atmospheric 4, etc. (full in `manifest.json`/`report.json`). No single category dominates (>82% scene but scene is super-category with sub-tags; failure-targets balanced: anti_meta 32, dialogue 24, voice 19, consistency 32, retention 20, rewriting 7, quiet 26, emotion 12, action 9, description 40, transitional 13).
**Anti-pattern metadata:** Each example documents what it corrects (meta intro, explanation vs execution, moral summary, exposition, lore dump, repetitive adjectives, cliché emotion, flat dialogue, hedging, premature conclusion) in `notes`, not as training target.
**Validation:** `120/120 valid, 0 invalid, 0 warnings, 0 duplicates` via `miyuki dataset validate --format conversational`. Deterministic ordering, SHA `8cda31...`, tokenizer stats recorded. Full suite run before training (63 passed at that time).
**Balance:** Represents writing spec (quiet, tense, emotional, action, transitional, atmospheric, character-focused, environmental). Rewriting 7 is smallest but focused (quality over count).

## Training — Experiment 0006

**Config:** `configs/training/phase4_lora.yaml` — r16 alpha32 dropout0.05 bias none, targets `[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]` (verified Qwen2), batch1 accum8 max1024 lr2e-4 CPU float32 seed42, 3 epochs, `assistant_only_loss: true`.
**Expected steps:** Train 108, batch1 accum8 → `108/8=13.5` → 13/epoch → 39 total (vs Phase3B 34→12). **Actual:** `global_step 39, epoch 2.88, train_runtime 2963s, elapsed 2971s (49.5 min), train_loss 2.623` (from ~3.6→2.0), `samples/sec 0.109, steps/sec 0.013, total_flos 92334866768640`. Trainable 8,798,208 / 502,830,976 (1.75%, same LoRA as 3B). Eval split 108/12 deterministic seed42.
**Checkpoint:** `checkpoints/phase4-miyuki-0.2` — `adapter_config.json (760B, r16, base Qwen)`, `adapter_model.safetensors (35.2 MB)`, `train_metrics.json`, `metadata.json`, `tokenizer.json`, `miyuki_training_config.yaml`, `dataset_info.json`, `checkpoint-30/35`. Separate from 0001–0005, base model separate, adapter independently loadable.
**Preserved controls:** `evaluation/runs/phase1_qwen2_0.5b/`, `phase3-miyuki-0.1/`, `phase3b-miyuki-0.1/`, `experiments/0001,0002,0003,0004,0005`, `checkpoints/phase3-miyuki-0.1`, `phase3b-miyuki-0.1`, `phase3c-a/b`, `datasets/miyuki-writing-0.1/` (SHA `9e1b97...` unchanged) recorded in metadata.

## Evaluation — Original 12-Prompt (180 max, temp0.8 top_p0.95 rep1.1 seed42, CPU)

**Runs:**
- Phase1 base: `evaluation/runs/phase1_qwen2_0.5b/outputs.jsonl` — 1538 toks avg128, 2.84 tok/s, 543s, 0 errors, 0 keep/3 edit/9 discard, meta 4 (answer-is, tone-essay, hedging, EOS).
- Phase3B (best prior): `evaluation/runs/phase3b-miyuki-0.1/outputs.jsonl` — 996 toks avg83, 3.4 tok/s, 292s, 0 errors, 1 keep/2 edit/9 discard, meta 2 EOS.
- Phase3C-best (B, user-only): 0 keep/1 edit/11 discard (worse than 3B, not used as best).
- **Phase4:** `evaluation/runs/phase4-miyuki-0.2/2026-09-25_Qwen_Qwen2-0.5B-Instruct/outputs.jsonl` — 995 toks avg82.9, 2.27 tok/s (slower due to load), 458s + 11s load, 0 errors, **5 keep/4 edit/3 discard**, meta 0 (no answer-is, no tone-essay, no hedging essay). See `comparison.json`.

**Per-prompt (Phase4 vs base/3B):** continuation edit (beam/house vs town-square), dialogue discard (still flat, Kira/Mika wrong names), scene-writing keep (damp teacup vs purple raindrops), character-interaction edit (eyes-dim vs essay dump), emotional edit (desk/empty letter vs heart-pounding), action discard (street/bag/thud, no notebook, but no Chinese), quiet keep (bench-shorter/whispers vs meta answer-is — **first real quiet success**), description keep (lantern/stream/backpack-27 vs generic mountaintop), rewriting keep (bed/hands/glasses/click vs sense-of-unease — **clear**), Jinx discard (109 earnest vs 12 truncation — longer but still not witty), tone keep (kettle/door/paper vs tone-essay — **no meta**), context discard-partial (hands-back-cup scene vs hedging essay — no hedging but still missing mug/lavender facts).

## Expanded Evaluation — `phase4_expanded` (12 new prompts, labelled, NOT replacing original)

**Suite:** `evaluation/prompts/phase4_expanded.jsonl` — dialogue, subtext, character_voice, continuity, rewriting, quiet, action, transitions, constrained-desc, anti-meta, emotion, dialogue-02.
**Runs (CPU-limited, documented):** Phase4 at 60 max (`phase4-expanded-miyuki-0.2-60`, 1763s, 0 errors, ~0.5 tok/s) vs Base at 256 max (`phase4-expanded-base`, 0 errors). Token budgets differ (60 vs 256) due to slowdown (0.4 tok/s Phase4 vs 2+ earlier) — comparison is qualitative only, noted in `human_eval_expanded.yaml`.
**Result:** Phase4 5 edit/7 discard (continuity, rewriting, quiet, transitions, anti-meta edit) vs Base 1 edit/11 discard. Phase4 better on 4/12 despite 60 vs 256 handicap. Language mixing in both (Phase4 subtext 松开, Base description 岸边) — needs monitoring. Confirms original finding: Phase4 improves grounding/quiet/transitions/rewriting, still weak on distinct voices.

## Human Evaluation

**Original:** `evaluation/runs/phase4-miyuki-0.2/human_eval.yaml` (9 dims + keep/edit/discard, same as Phase1/3B). **Phase1 0/3/9 → Phase3B 1/2/9 → Phase4 5/4/3** (scene, quiet, description, rewriting, tone keep; continuation, interaction, emotional, action edit; dialogue, Jinx, context discard).
**Expanded:** `evaluation/runs/phase4-expanded-miyuki-0.2-60/human_eval_expanded.yaml` (5/7 vs 1/11).
**Key question:** Is MIYUKI now meaningfully more useful to a writer? **Yes for quiet/scene/description/rewriting/tone (7/12), no for dialogue/Jinx/binary-context.** Overall yes — first time majority keep/edit.

## Overfitting

- **Copied phrases:** 12 Phase4 original outputs checked for 40-char verbatim vs 120 dataset targets → **none**. Expanded same → none.
- **Name overuse:** No repeated Jinx/Mara injection beyond prompt; invented terms (lumenweave/scryglass/voidtide/aureline/kestrel) 0 in original 12 (good, not over-injected).
- **Repeated structures:** No collapse to single template; outputs varied (71–140 toks).
- **Factual knowledge:** Paraphrased probes (Haru tea, Elena mug) generated scenes, no meta, not empty; factual `sky is blue` in Phase3B was correct (scatter), in Phase4 round-trip it was weak (`reflections come from all directions`, no scatter) — slight factual degradation, needs monitoring, not catastrophic.
- **Language mixing:** Original 12 Phase4 has 0 Chinese (good, fixed Phase3 Chinese). Expanded 60-token Phase4 has 1 Chinese (松开 in subtext) and base has 1 (岸边) — monitoring needed, not pervasive.
- **Conclusion:** No obvious memorization, no term injection, but slight factual softening. Underfitting resolved partially, not overfitting.

## Improvements (Evidence-Backed Only)

- **Anti-meta eliminated (0 vs 4):** No `The answer is`, no tone-essay, no hedging essay. Quiet, tone, rewriting, context-no-hedging all attempt scenes (vs essays). Direct result of assistant-only + 32 anti-meta examples.
- **Scene grounding:** Kyoto tea (damp teacup), observatory (lantern/stream/backpack-27), harbor (not in original but expanded transitions) — selective, POV-anchored.
- **Rewriting:** 100-token vivid (bed/hands/glasses/click) vs 29-token generic — first rewriting success.
- **Description constraints:** No-dialogue respected (vs Phase3B violation), child POV via backpack count.
- **Tone:** Restrained continuation (kettle/door/paper) vs essay.
- **Loss correlates now:** 3.20 (3B all-token, flat) vs 2.62 (Phase4 assistant-only, 120 ex, improved 7/12) — correct loss + larger data matters.

## Remaining Problems

- **Dialogue distinct voices (1):** Kira/Mika laundry/kettle still generic, wrong names, no interruptions. 24 dialogue examples insufficient.
- **Jinx witty voice (1):** 109 tokens complete but earnest mission-speak, not blunt dry humor. 19 voice examples insufficient for this specific voice.
- **Binary context facts (1):** Elena mug/lavender — no hedging essay anymore, but still missing chipped-blue-grandmother + lavender-hate facts, invents mom. 20 retention examples improved hedging but not facts.
- **Action retrieval:** No notebook retrieval, men appear. 9 action examples insufficient for spatial causality.
- **Factual softening:** Sky-blue probe weaker than 3B.
- **Language mixing (expanded):** 1/12 Phase4 subtext has Chinese.

## Model Capacity Assessment

**Continue improving dataset/training pipeline before scaling model.** Phase4 with correct loss + 3.1× data (38→120) + 3.25× steps (12→39) moved from 1/2/9 to 5/4/3 and eliminated meta behavior. This is substantial improvement, not flat. Remaining failures (dialogue, Jinx, binary facts) are specific and addressable with more targeted data (e.g., 20 more dialogue with interruptions, 10 more Jinx witty, 10 more binary-constraint drills) rather than requiring larger model yet. Evidence for 0.5B capacity limitation is **not yet strong** — model learned quiet/tone/rewriting/description, showing capacity for those. If next iteration with 30–40 more targeted examples still fails on dialogue/Jinx/binary, then investigate larger local model (e.g., 1–1.5B Qwen) in next milestone. Do NOT jump now.

## Artifacts

- **Dataset:** `datasets/miyuki-writing-0.2/{dataset.jsonl (120, SHA 8cda31...), manifest.json, report.json (16088 toks avg134 max233), review.yaml (120 keep), schema.json, DATASET_SPEC.md, README.md}` + `data/manifests/miyuki-writing-0.2.json`. Original `miyuki-writing-0.1` preserved (SHA 9e1b97...).
- **Training:** `configs/training/phase4_lora.yaml` (`assistant_only_loss: true`), `checkpoints/phase4-miyuki-0.2/{adapter_config.json (r16), adapter_model.safetensors 35.2MB, train_metrics.json (39 steps, 2.62), metadata.json, tokenizer.json, dataset_info.json (108/12), checkpoint-30/35}`, `experiments/0006/{metrics.json (id0006, Qwen, dataset 0.2, version 0.2.0-phase4, lr2e-4 epochs3 steps39 loss2.62, hardware i7, artifact phase4)}`. Preserved 0001–0005, phase3/3b/3c adapters.
- **Evaluation:** `evaluation/runs/phase4-miyuki-0.2/{2026-09-25_Qwen.../outputs.jsonl (12×180), metadata.json, comparison.json, human_eval.yaml, README.md (to be added)}` + `evaluation/runs/phase4-expanded-miyuki-0.2-60/{.../outputs.jsonl (12×60), human_eval_expanded.yaml}` + `evaluation/runs/phase4-expanded-base/` (base 256-token for reference) + `evaluation/prompts/phase4_expanded.jsonl` (labelled, not replacing original).
- **Audit:** `evaluation/runs/phase4_loss_audit.json` (before: all trainable), `phase4_loss_audit_diagnostic.json` (after: 93 masked/75 trainable), `docs/training_signal_audit.md`, `tools/loss_audit.py`, `tools/training_target_test.py`, `tools/phase4_compare.py`.
- **Docs:** `docs/phase4_report.md` (this), `docs/roadmap.md` (to update), `README.md` (to update).

## Tests

**Total: 74 passed** (20 dataset/hardware/artifacts + 14 eval/inference/chat/phase2 + 24 phase3/3b/3c + 16 phase4/training-target/config = 74, split into batches due to 120s tool timeout, 2 warnings huggingface_hub + peft).
- Existing 63 preserved (Phase0–3C).
- New Phase4 8 (`tests/test_phase4.py`: config assistant-only, preservation SHA, deterministic 108/12 split, 39 steps loss<3.0, checkpoint discovery, tokenizer match, baseline preservation no hedging/meta, metadata) + Training-target 3 (`tests/test_training_target.py`: masking, no prompt becomes target, config flag) = 11 new, 63+11=74.

