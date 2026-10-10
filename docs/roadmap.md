# Miyuki Roadmap

### Phase 0 — Infrastructure (this milestone)

- [x] Project structure, configs, .gitignore
- [x] Model abstraction (YAML, replaceable)
- [x] Dataset schemas + validation + versioning
- [x] Writing spec (`docs/miyuki_writing_spec.md`)
- [x] Hardware detection
- [x] Baseline inference pipeline + CLI (`miyuki infer`)
- [x] Evaluation prompts (12) + runner + human template
- [x] Training pipeline (SFT/LoRA, checkpointing, resume)
- [x] Smoke training dataset + smoke run
- [x] Checkpoint round-trip + reload generation
- [x] Experiment tracking + model artifacts
- [x] Tests + docs

Exit: smoke run works, checkpoint loads, evaluation runs, docs explain workflow.

### Phase 1 — Baseline model ✅ COMPLETE (2026-09-24)

- [x] Activated `Qwen/Qwen2-0.5B-Instruct` (494M, Qwen2) via `configs/model/small-instruct.yaml` (baseline `tiny-random` retained for regression)
- [x] Verified tokenizer/model load, chat template (`<|im_start|>...`), family detection, real generation (not fallback) on CPU float32 (i7-1065G7, 16 GB, MX330 2 GB, torch CPU)
- [x] Fixed runtime generically: chat template auto-use when `tokenizer.chat_template` exists + evaluation runner reuses single `TransformersRuntime` (regression tests added)
- [x] Ran 12-prompt suite (`eval_v0.1`) — `evaluation/runs/phase1_qwen2_0.5b/` (956 prompt toks → 1538 out, 2.84 tok/s, 543.8 s elapsed, 8.9 s load, 0 errors, 494M params)
- [x] Small sweep: A 0.7/0.9/1.05, B 0.8/0.95/1.1, C 1.0/0.95/1.0 on 4-prompt subset — no temp/top_p fixes systematic meta-answer failures (evidence in `generation_configs.json` + `sweep_*`)
- [x] Human eval vs `docs/miyuki_writing_spec.md` — `evaluation/runs/phase1_qwen2_0.5b/human_eval.yaml` (9 discard, 3 edit, 0 keep; strong gaps in instruction adherence, dialogue, continuity, context retention)
- [x] 31 tests passing (added `test_runtime_chat_template.py`)

Result: Qwen2-0.5B is a **real but generic baseline**, far from MIYUKI's target. Next is Phase 2 writing dataset, not training or Story Brain yet.

### Phase 2 — Miyuki writing dataset ✅ COMPLETE (2026-09-24)

- [x] Curated `miyuki-writing-0.1` — 38 human-authored exemplars (human-authored, human-edited) aligned to `docs/miyuki_writing_spec.md` and Phase 1 failures (anti-meta, dialogue, context retention, etc.)
- [x] Before/after pairs for explain-vs-write, hedging vs commit, lore dump vs natural reveal; each exemplar has `category/skills/source/notes` and is validated
- [x] Validated: `38/38 valid, 0 warnings, 0 duplicates`, tokenization OK (5072 toks, 133.5 tok/ex, 21738 chars), `manifest.json` with SHA256 `9e1b97461f...`, `review.yaml` all 38 `keep`
- [x] Vocabulary study on Qwen2-0.5B-Instruct (151646 vocab, 151936 model vocab, 4.29 chars/tok, 1.36 tok/word) — report in `evaluation/runs/phase2_vocabulary/tokenizer_report.{json,md}`
- [x] Identified Miyuki-specific vocab: recurring invented terms (`lumenweave 4 toks`, `Aureline 3`, `voidtide 3`, names 2) — fragmented but not catastrophic (max 4)
- [x] Tokenizer experiment: `miyuki-experimental` 8192/16000/32000 requested → actual 2572 each (corpus-limited), 5071→4776 toks (−5.8% compression), invented terms sometimes 4→1 (`lumenweave`) but `scryglass` 3→4 — modest gain
- [x] Special tokens investigated: model 151936 vs tokenizer 151646 (+290 reserved), 3 specials within vocab, warning expected not harmful, no `resize_token_embeddings` on production model (per spec)
- [x] Mapping Phase1 failures → dataset examples (see `datasets/miyuki-writing-0.1/failure_mapping.yaml`), distinct keep/edit/discard review
- [x] 39 tests passing (added 8 Phase 2 tests), reproducible (python 3.12.10, transformers 4.41.0, deterministic corpus hash)

Result: Corpus foundation ready; vocab mismatch is small (5.8% efficiency), prioritize dataset quality over tokenizer adaptation. Next is Phase 3 LoRA with Qwen tokenizer (no vocab change), not Story Brain yet.

### Phase 3 — First SFT ✅ COMPLETE (2026-09-24)

- [x] First LoRA SFT on 0.5B against `miyuki-writing-0.1` (38 ex, human-authored) — `configs/training/phase3_lora.yaml` (r16 alpha32, 1 epoch, batch1 accum8, max_seq1024, lr 1e-4, CPU float32, seed 42)
- [x] Reproducible: smoke verified LoRA attach (trainable 8,798,208 / 1.75%), 2 steps loss 3.81→3.31, checkpoint `checkpoints/phase3-smoke`; full train 4 steps, `train_loss 3.762` (from 4.06), `elapsed 382s`, `experiments/0002`
- [x] Round-trip: base vs adapter both load (`PeftModel` true via auto `checkpoints/phase3-miyuki-0.1` + explicit `load_adapter_from`), generations differ but not random, no tokenizer mismatch
- [x] Same 12-prompt eval `evaluation/runs/phase3-miyuki-0.1/` (956 prompt toks, 1698 out vs 1538 baseline, avg 3.31 tok/s, elapsed 514.9, 0 errors) — apples-to-apples with Phase 1
- [x] Human eval `human_eval.yaml` (10 discard, 2 edit, 0 keep — no prompt improved to next tier; 1 regression with Chinese mixing, 3 identical, 5 same failure class). Loss ↓ did NOT → writing improvement.
- [x] Failure/overfitting analysis: no verbatim 40-char memorization, no name over-injection, general factual answer still ok, invented term injection 0, but anti-meta/context/dialogue failures persist (5 anti-meta dataset exemplars not learned in 4 steps). No obvious overfitting, clear underfitting.
- [x] 47 tests passing (added 8 Phase 3 tests: config, conversion, LoRA, checkpoint, adapter, experiment, deterministic, baseline preservation)

Result: Hypothesis **not supported** with this conservative config (4 optimizer steps). Dataset correct but training signal too weak. Next should try more epochs/steps, not new dataset or vocab.

### Phase 3B — More Optimization Steps ✅ COMPLETE (2026-09-25)

- [x] More steps: same dataset/model/tokenizer, `configs/training/phase3b_lora.yaml` (r16 alpha32, **3 epochs, lr 2e-4**, batch1 accum8, max_seq1024, seed42) — expected 12 optimizer steps vs Phase3's 4 (3×)
- [x] Smoke 4-ex `max_steps 2, lr 2e-4`: trainable 1.75% same, steps2 loss3.81, checkpoint `phase3b-smoke`, adapter reload ok, base still loads
- [x] Full 3 epochs: `global_step 12, epoch 2.82, train_runtime 752.5s, elapsed 759.3s (12.6 min), train_loss 3.205` (from 4.06→3.91→3.51→3.34→3.28→3.12→3.01→3.06→2.80→2.74→2.68→2.89), `eval_loss 3.369→3.173`, `experiments/0003`, `checkpoints/phase3b-miyuki-0.1` (35.2 MB, separate, not overwriting 0002)
- [x] Round-trip Base/Phase3/Phase3B all load (is Peft False/True/True), same tokenizer 151646, generations distinct and not random (lighthouse: base `her ship...` vs Phase3 `shore... awe...` vs Phase3B `call it a night... silence at base...`)
- [x] Same 12-prompt eval `evaluation/runs/phase3b-miyuki-0.1/` (956 prompt, 996 total vs 1538/1698, many short: 76/72/180/180/77/108/20/180/17/31/23/32, avg 83, 3.4 tok/s, elapsed292.9, load8.6, 0 errors) — apples-to-apples
- [x] Human eval `human_eval.yaml` (1 keep emotional, 2 edit continuation/character-interaction, 9 discard) — **partial improvement** (emotional 1→4 keep, character-interaction 1→3, Chinese fixed) but 75% still discard, quiet/tone/context still meta/hedging, description adds dialogue violation, Jinx longer but still earnest
- [x] Overfitting: 4 paraphrased prompts → no 30-char copied, no name overuse, no meta, not empty; eval 0 verbatim 40-char memorization, 0 invented injection, factual `sky is blue` still correct (scatter) — no overfitting, underfitting
- [x] 55 tests passing (added 8 Phase3B tests: config isolation, expected steps 4 vs 12, deterministic split 34/4 seed42, checkpoint discovery phase3+3b, adapter, baseline preservation, metadata)

Result: More optimization helped **some categories** (emotional, character-interaction, Chinese fix) but overall still flat (1 keep vs 0). Loss 3.76→3.20 shows training, but writing improvement limited. Suggests dataset has signal but 12 steps still small; next should test formulation (prompt format) or slightly larger model, not just more epochs blindly.

### Phase 3C — Training Formulation Experiment ✅ COMPLETE (2026-09-25)

- [x] Preserved `miyuki-writing-0.1` byte-for-byte (SHA 9e1b9...), derived `datasets/miyuki-writing-0.1-phase3c/{dataset-a.jsonl (b518f30b, 38, system contract 64 toks), dataset-b.jsonl (e408e23e, 38, user suffix 42 toks), manifest.json}` — targets identical, only instruction formulation differs
- [x] Formulations: **A** system-level `You are Miyuki, a direct story-writing assistant...` (64 toks system) vs **B** user-only `Requirements: Perform...` suffix (42 toks). Training config same as Phase3B (r16 alpha32, 3 epochs, lr2e-4, batch1 accum8, max_seq1024, seed42, CPU) — only formulation variable.
- [x] Tokenization verified: orig 5869 total (154 avg, max233) → A 8073 (212 avg, max291, +37.6%) → B 7428 (195 avg, max274, +26.6%); all <30% of 1024 window, not excessive. See `tools/check_tokenization_phase3c.py`.
- [x] Smoke both A/B (4-ex, max_steps2, lr2e-4): trainable 1.75% same, steps2 loss 4.17 (A) /3.99 (B), checkpoint `phase3c-*-smoke`, adapter reload + generation ok (though generation still poor `Elena takes a sip of lavender tea...`).
- [x] Full training both 3 epochs, 12 steps each: **0004 A** `global_step12, epoch2.82, train_loss3.239, elapsed1220.9s, eval 3.31→2.78`, `checkpoints/phase3c-a-miyuki-0.1` (35.2 MB); **0005 B** `global_step12, epoch2.82, train_loss3.143, elapsed1211.9s, eval 3.29→2.82`, `checkpoints/phase3c-b-miyuki-0.1` — separate, not overwriting 0002/0003, not modifying original dataset.
- [x] Round-trip Base/Phase3/Phase3B/3C-A/3C-B all load (is Peft False/True...), same tokenizer 151646, generations distinct and not random (lighthouse: base `her ship...` vs Phase3 `shore...` vs Phase3B `call it a night...` vs A `return. The sound of distant sea, whirring generator...` vs B `leave. The sound of engine died...`).
- [x] Same 12-prompt eval for both: **A** `evaluation/runs/phase3c-a-miyuki-0.1/2026-.../outputs.jsonl` (898 total avg75, 369s, 0 errors) and **B** `phase3c-b...` (754 total avg63, 314s, 0 errors) — apples-to-apples with Phase1/3/3B.
- [x] Human eval `human_eval.yaml` for each: **A** `0 keep /1 edit (scene-writing) /11 discard` (emotional 28 discard vs Phase3B keep, character-interaction 113 discard vs 3/edit), **B** `0/1/11` (emotional 8 `The door slammed` discard, context 5 `Yay!` worst). Both **worse than Phase3B's 1/2/9**; system vs user not the lever.
- [x] Failure checks: anti-meta still 4 (quiet ticket, tone meta), dialogue flat, scene not grounded, Jinx longer but still earnest (87/109 vs Phase3B 31), context hedging still (11 `Thank you I won't drink it` /5 `Yay!` vs Phase3B 32 pot), premature EOS many short (<30: A2, B6), generic phrasing same, action no Chinese (stays fixed) but still incoherent, factual now shows **Chinese `空气(air)` leakage** in both A/B vs Phase3B correct.
- [x] Overfitting: 0 verbatim 40-char memorization in eval for both, 0 invented injection, factual Chinese leakage indicates language degradation (worse than Phase3B), no name overuse.
- [x] 63 tests passing (added 8 Phase3C tests: original preservation SHA 9e1b..., derived determinism b518/e408, formulation A/B structure (system vs user, assistant identical), token stats 5869/8073/7428, experiment isolation 0004≠0005, adapter len 151646, metadata).

Result: Formulation alone (system vs user contract, 64 vs 42 toks overhead) **did NOT materially improve** writing behavior vs Phase3B; both 0 keep vs Phase3B 1 keep. System vs user similar (both 0/1/11). More optimization helped somewhat (Phase3B), but formulation is not primary lever — evidence points to **model capacity (0.5B) or dataset size/quality limits** rather than instruction placement. Next should investigate larger model or expanded dataset, not formulation.

### Phase 4 — Dataset Expansion & Training Signal Audit ✅ COMPLETE (2026-09-26)

- [x] Audited loss: Phase0–3C trained ALL tokens (system+user+assistant, 168 total, 0 masked); fixed with `assistant_only_loss: true` (93 masked, 75 trainable, specials handled). See `docs/training_signal_audit.md`, `evaluation/runs/phase4_loss_audit*.json`.
- [x] Created `miyuki-writing-0.2` — 120 ex (38 from 0.1 SHA 9e1b97... + 82 new targeted), 57558 chars, 6318 words, 16088 toks avg134 max233, categories scene98/quiet26/anti_meta32/dialogue24/voice19/retention20/etc., `120/120 valid`, review 120 keep, manifest SHA 8cda31...
- [x] Trained 0006: Qwen2-0.5B, LoRA r16 alpha32, batch1 accum8 max1024 lr2e-4 3 epochs seed42 CPU, 108/12 split, 39 steps (vs 12), loss 2.62 (vs 3.20), 2971s, `checkpoints/phase4-miyuki-0.2`, preserved 0001–0005.
- [x] Original 12-prompt eval (180 max): Phase1 0/3/9 (1538 toks) vs 3B 1/2/9 (996) vs **Phase4 5/4/3 (995 toks, meta 0 vs 4)** — quiet/tone/rewriting/description keep, no hedging essay. See `evaluation/runs/phase4-miyuki-0.2/human_eval.yaml`, `comparison.json`.
- [x] Expanded suite `phase4_expanded` (12 new, labelled, NOT replacing original): Phase4 60-token 5/7 vs Base 256-token 1/11 (token handicap noted) — Phase4 better on continuity/rewriting/quiet/transitions/anti-meta. See `human_eval_expanded.yaml`.
- [x] Overfitting: 0 verbatim 40-char, 0 term injection in original; expanded has 1 Chinese (松开) vs base 1 (岸边) — monitoring. Factual slight softening.
- [x] 74 tests passing (11 new: 8 phase4 + 3 training-target), split batches due to tool timeout.

Result: Correct loss + larger targeted data **substantially improves** (5 keep vs 0/1). Remaining: dialogue distinct voices, Jinx witty voice, binary context facts. Evidence supports continuing 0.5B pipeline before scaling. Next: 30–40 more targeted dialogue/voice/binary examples, not larger model yet.

### Phase 5 — Evaluation and iteration (next)

- Compare BASE vs MIYUKI-0.2 on identical prompts (done for Phase4: 5/4/3 vs 0/3/9).
- Update writing spec from failure modes (dialogue/Jinx/binary still fail).
- Iterate dataset and hyperparameters.

### Model Scale Test — Qwen2.5-1.5B Base ✅ COMPLETE (2026-09-26)

- [x] Added `configs/model/qwen2.5-1.5b-instruct.yaml` (selectable, 0.5B preserved, no hard-coding)
- [x] Loaded standard `Qwen/Qwen2.5-1.5B-Instruct` (1,543,714,304 params, vocab 151665, 32K ctx, chat True) via existing TransformersRuntime, CPU float32, no quantization — 451s load, 3.10GB download, RAM 5.74→1.58GB (90.7%), 0.7–1.78 tok/s sanity (5 prompts, no crashes, 1 false refusal noted)
- [x] Ran EXACT original 12 prompts (180 max, temp0.8 top_p0.95 rep1.1 seed42) — `evaluation/runs/model-scale-qwen2.5-1.5b/outputs.jsonl` (1971 toks avg164, 1.53 tok/s, 1346s, 0 errors)
- [x] Compared Phase1 0/3/9 vs 3B 1/2/9 vs Phase4 5/4/3 vs 1.5B 2/5/5 — 1.5B ties scene/rewriting/quiet/emotional, wins Jinx/action, loses tone/context/description + refusal risk. See `human_eval.yaml`, `comparison.json`.
- [x] Targeted 5 prompts (dialogue, Jinx, binary Elena, child observatory, action retrieval) — `targeted_eval.jsonl` (507s): Jinx 41-token witty best yet, but context accepts lavender + silver-mug hallucination (worse than Phase4).
- [x] Resource report (`resource_report.json`, `sanity.json`): fits RAM but no training headroom, MX330 unused, disk 455GB free.
- [x] 81 tests passing (7 new scale tests, split batches).
- [x] No training, no checkpoint, Phase4 adapter remains backend.

Result: CASE B — only slightly better (specific categories). Do NOT move backend yet; small LoRA test on 1.5B with `miyuki-writing-0.2` justified but NOT run.

### Qwen2.5-1.5B Micro-LoRA ✅ COMPLETE — RESULT D: MEMORY NO-GO (2026-09-26)

- [x] Created `configs/training/qwen2.5-1.5b_micro_lora.yaml` (1 epoch, r16, lr2e-4, assistant-only true, output `checkpoints/qwen2.5-1.5b-micro-lora/`), Phase4 config/dataset/checkpoint unchanged
- [x] Preflight: loaded 1.5B as benchmark + attached LoRA r16 (trainable 18,464,768 / 1.18%), RAM 9.47GB → 2.82GB (83.4%) < 3.0GB threshold → **NO-GO**. No training attempted, no checkpoint, no 0007. See `evaluation/runs/qwen2.5-1.5b-micro-lora/preflight.json`, `tools/micro_preflight.py`.
- [x] Estimated cost: 108 train/12 eval → ~13 optimizer steps for 1 epoch, ~45–65 min + load (not exact, printed pre-training)
- [x] No micro 12-prompt/targeted/human eval possible (no adapter); existing 1.5B base vs Phase4 stands (2/5/5 vs 5/4/3)
- [x] 92 tests passing (11 new micro tests, split batches)
- [x] No quantization hacks, Phase4 backend remains.

Result: RESULT D — keep 1.5B inference-only, Phase4 0.5B adapter remains backend. Next needs ≥32GB RAM machine or lighter approach, not blind retries.

### Phase 4C — Dataset-Effect Experiment (0.2→0.3) ✅ COMPLETE (2026-09-29, resumed to completion)

- [x] Compat fix: narrow numpy allowlist (`_reconstruct`, `ndarray`, `dtype`, `UInt32DType`) in resume path only (torch≥2.6 weights_only rejection of trusted local `rng_state.pth`); 6 regression tests incl. security (malicious pickle rejected). See `tools/diagnose_rng_state.py`, `tests/test_checkpoint_resume_compat.py`.
- [x] Resumed from checkpoint-30 → completed 51 steps/epoch 2.97, 1659s, segment loss 0.938 → **exp 0007** (mechanism-created). A second resume (ckpt-45) auto-recorded **exp 0008** (duplicate; 465.6s, segment loss 0.252) — final adapter in same dir; evaluations used it. Not faked, kept per no-delete rule.
- [x] Loss trace: 3.35→2.69→~2.0→0.25; **eval_loss plateau 2.86** while train collapsed → overfitting on 154 ex in epochs 2–3.
- [x] Original 12 (180/0.8/0.95/1.1/42): **P4 5/4/3 vs P4C 1/8/3** — meta 0 both, 0 Chinese/EOS. Improved: dialogue (discard→edit), Jinx (discard→edit, in-voice dry). Regressed: scene (keep→edit, surreal), rewriting (keep→edit, 29 toks), emotional (edit→discard), continuation. Same: context binary fail, action no retrieval.
- [x] Expanded 12 (`phase4_expanded`, same suite): Chinese leakage fixed (松开 gone), voidtide-speaker fixed, mug-ring learned (transitions); action retrieval/Jinx-in-scenes/constrained-desc/rewriting still fail.
- [x] Targeted 5: Jinx voice improved (in-voice dry) but content drifts surreal; context binary still fails; dialogue teen-vague partial; child-POV fail; action notebook tracked but ends with vendor.
- [x] 107 tests passing (6 new compat). No commit. Phase 4 checkpoint/datasets preserved.

Result: Dataset 0.3 did NOT improve the original benchmark over Phase 4 (net keep→edit shift; overfit in epochs 2–3). Some targeted behaviors learned (dialogue roles, Jinx voice, mug-ring, Chinese fix). Next: controlled 1–2 epoch run on 0.3 to isolate overfitting from data effect, or revert to 0.2 recipe.

### Phase 4B — Targeted Dataset Expansion ✅ COMPLETE (2026-09-26, no training)

- [x] Created `miyuki-writing-0.3` — 120 from 0.2 (SHA 8cda31..., byte-for-byte) + 34 new (A voice 8, B binary 8, C Jinx 6, D object 6, E viewpoint 6) = 154 total, SHA f3f5f636..., all human-authored/reviewed keep
- [x] Validated 154/154, 0 duplicates/near-duplicates, tokenizer 21027 toks avg136.5 max233, split 138/16, manifest + review + report + README
- [x] 101 tests passing (9 new phase4b, split batches). No 0007, no checkpoint, Phase4 backend unchanged.

### Phase 5 — Story Brain

- Future app subsystem that assembles structured context:
  `CURRENT_SCENE / CHARACTERS / STATES / RELATIONSHIPS / WORLD FACTS / TIMELINE / RELEVANT MEMORIES / RESEARCH / AUTHOR INTENT / RECENT PROSE`
- Model interface accepts that context as prompt; model does NOT own persistent memory.

### Phase 6 — Character intelligence

- Pipeline: `raw → extraction → normalization → conflict detection → missing-info → research → proposed improvements → user approval → structured record`
- User remains in control; Miyuki proposes, not silently rewrites.

### Phase 7 — Research system

- `WEB → retrieve → extract → identify relevant → preserve sources → store structured → provide context`
- Model treats web text as untrusted.

### Phase 8 — Local runtime optimization

- GGUF / llama.cpp backend behind the existing `RuntimeBase` abstraction.
- Quantized CPU-efficient local inference; multi-hardware backends.

### Phase 9 — Miyuki application integration

- Wire model runtime into the desktop application.
- Retrieval, timeline, memory, lorebook replacement — full story intelligence loop.

---

*Do not prematurely implement later phases. Each phase is gated on measured results from the prior one.*
