# Phase 4C — Dataset-Effect Experiment (miyuki-writing-0.3, resumed to completion)

**Result: trained to completion (51 steps, 3 epochs) via resume; dataset 0.3 did NOT improve the original 12-prompt benchmark over Phase 4 (0.2). Overfitting in epochs 2–3 is the likely cause.**

## Training

- **Experiment 0007** (intended, created by the mechanism): resume from `checkpoints/phase4c-miyuki-0.3/checkpoint-30`, completed 51 steps / epoch 2.97, 1659s, segment train_loss 0.938 (steps 31–51 average).
- **Experiment 0008** (unintended auto-recorded duplicate): a second resume from checkpoint-45 completed after 0007 existed, so the mechanism created 0008 — 465.6s, segment train_loss 0.252 (steps 46–51). Same final checkpoint directory; **the final adapter (saved 22:24) is the 0008 segment's weights**, and all evaluations used it. Not manually faked; kept per no-delete rule.
- Config: `configs/training/phase4c_lora_03.yaml` — identical recipe to Phase 4 (r16 alpha32 dropout0.05 bias none, qwen2-7 targets, batch1 accum8 max1024 lr2e-4 seed42 CPU float32, assistant_only_loss true); **only variable: dataset 0.2→0.3** (SHA `8cda31...`→`f3f5f6...`).
- Loss trace (campaign): 3.35 → 2.69 → ~2.0 → 0.25; **eval_loss 3.17 → 2.86 → 2.86 (plateau)** while train loss collapsed — classic overfit signal on 154 examples in epochs 2–3.
- Memory guard never tripped (avail 4.8–5.7 GB after runs).

## Compatibility Fix

`src/miyuki/training/train.py` — `_enable_local_checkpoint_rng_compat()`: narrow allowlist of exactly 4 numpy globals (`numpy._core.multiarray._reconstruct`, `numpy.ndarray`, `numpy.dtype`, `numpy.dtypes.UInt32DType` — empirically determined via `tools/diagnose_rng_state.py`), invoked **only** on the trusted local resume path. Additive (`add_safe_globals`), never disables weights_only globally; malicious pickles still rejected (verified in test). Fixed the torch≥2.6 `weights_only=True` rejection of the trusted local `rng_state.pth`.

## Original 12 Evaluation (180/0.8/0.95/1.1/42)

`evaluation/runs/phase4c-miyuki-0.3/{outputs.jsonl, metadata.json, human_eval.yaml, comparison.json}` (run 2026-09-29, 596s, 0 errors).

- **Phase 4:** 5 keep / 4 edit / 3 discard, 995 toks avg82.9, 2.28 tok/s, meta 0
- **Phase 4C:** **1 keep (tone) / 8 edit / 3 discard**, 876 toks avg73.0, 1.35 tok/s, meta 0, 0 Chinese, 0 premature-EOS

Per-prompt: **improved** dialogue (discard→edit: "I'm not thirsty, just tired" — roles closer, dry reply) and **Jinx** (discard→edit: "Great... An even better plan... The geniuses' plans were always the same" — in-voice dry/blunt vs earnest). **Regressed** scene-writing (keep→edit: "window glass hard and then soft... test to see what tea would do" — surreal), rewriting (keep→edit: 29 toks, much less vivid), emotional (edit→discard: "thumb on the doorknob — no finger"), continuation (weaker imagery). **Same:** quiet (terse but quiet), description (structured, less POV), tone (restrained, no essay), context (still fails binary — no mug/lavender), action (still no retrieval).

## Expanded 12 Evaluation (`phase4_expanded.jsonl`, same suite as Phase 4's expanded)

`evaluation/runs/phase4c-expanded12-miyuki-0.3/{outputs.jsonl, comparison.json}` (12/12, 0 errors). Note: P4C had 180-token budget vs P4's 60-token budget (handicap reversed from previous comparison).

- **Improved:** Chinese leakage fixed (subtext 松开 gone), voidtide-as-speaker fixed (emotional), mug-ring behavior learned (transitions: "changed position by the mug ring" — from dataset 055/115), continuity reasoning ("didn't put it on because he knew already that it would be cold").
- **Still failing:** action retrieval (scale creaks but notebook never retrieved), Jinx witty in multi-character scenes ("Jinx's voice is soft now"), constrained description (adjectives used, confused negations), rewriting (short).
- **Meta behavior:** 0 across both suites (no answer-is, no tone-essay, no announcements).

## Targeted 5 (scale suite, `targeted_eval.jsonl`)

- Jinx: in-voice dry/blunt but drifts to invented surreal content — voice improved, content incoherent
- Context (Elena): FAIL binary — surreal mug ("cold, hard black from the fridge"), no lavender/grandmother
- Dialogue (teen/officer): teen vagueness present ("blue — blue, blue"), officer becomes confused — partial
- Description (child): not child-anchored, contradictory (dark but sun high) — fail
- Action: notebook tracked but ends with vendor; violence present — fail

## Phase 4 vs Phase 4C Comparison

| | Phase 4 (0.2) | Phase 4C (0.3) |
|---|---|---|
| Original keep/edit/discard | **5/4/3** | 1/8/3 |
| Meta / Chinese / EOS | 0 / 0 / 0 | 0 / 0 / 0 |
| Train loss (final) | 2.62 (39 steps) | 0.25 segment / eval plateau 2.86 |
| Jinx voice | earnest | **in-voice dry** |
| Dialogue | wrong names | **closer roles, dry reply** |
| Scene/rewriting | keep | surreal / short (regressed) |
| Context binary | fail | fail (same) |
| Expanded: Chinese/voidtide-speaker/mug-ring | 松开 / speaker / objects | **fixed / fixed / learned** |

**Measured:** dataset 0.3 did not improve the original benchmark (net keep→edit shift). **Interpretation:** the 34 targeted examples taught some specific behaviors (dialogue roles, Jinx voice, mug-ring, Chinese fix) but epochs 2–3 overfit (train 0.25 vs eval plateau 2.86), degrading scene/rewriting with surreal imagery.

## Tests

**107 passed** (25 phase4/training/config/4b + 24 dataset/hardware/eval + 52 phase2/3/3b/3c/micro/scale/chat + **6 new** `tests/test_checkpoint_resume_compat.py`: existing format loads with narrow allowlist, security preserved (malicious pickle rejected), single guarded call site, safetensors unaffected, checkpoint-30 trusted-local verified, no-allowlist failure documented). No tests weakened.

## Files Changed

- `src/miyuki/training/train.py` (compat fix + helper), `tests/test_checkpoint_resume_compat.py` (new), `tools/diagnose_rng_state.py` (new), `tools/run_phase4c_resume.py` (updated ckpt-45), `tools/run_phase4c_eval.py` (new), `tools/phase4c_compare.py` (new), `tools/phase4c_expanded_compare.py` (new), `tools/phase4c_targeted_dump.py` (new)
- Artifacts: `checkpoints/phase4c-miyuki-0.3/` (final adapter, checkpoints 25/30/40/45/50), `experiments/0007/` + `0008/` (auto), `evaluation/runs/phase4c-miyuki-0.3/`, `phase4c-expanded12-miyuki-0.3/`, `logs/phase4c_*.log`
- Preserved: datasets 0.1/0.2/0.3 SHAs, Phase 4 checkpoint, experiments 0001–0006, configs

## Unresolved Issues

- **Experiment 0008 auto-recorded** (duplicate of 0007 from the double-resume) — created by the normal mechanism, kept per no-delete rule; final adapter weights are the 0008 segment's.
- **Silent process deaths** during long detached runs (3×, no traceback) — mitigated by checkpoint/resume; root cause unclear (likely OOM under 83–90% RAM or session cleanup).
- **Overfitting signal:** train 0.25 vs eval plateau 2.86 — 154 examples × 3 epochs overfits.
- Surreal-imagery failure mode appeared in 4C (new vs Phase 4).

## Recommendation (not run)

Next experimental step: a controlled **1–2 epoch run on 0.3** (same recipe, stop before the overfit plateau) vs Phase 4 — isolating whether 0.3's targeted examples help before overfitting sets in, or whether 0.2 remains the better training set. Do not scale to 1.5B (memory-blocked). Do not add more data until the epoch-count question is measured.
