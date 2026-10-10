# Phase 5 Readiness Audit

**Status: READINESS ONLY. No training was run. No dataset, checkpoint, or experiment was changed. No experiment 0009 was created.**

Audit date: 2026-09-30. Repository evidence reviewed includes `AGENTS.md`, migration/current-state/Phase 4/Phase 4C/roadmap and strategy docs, training configs and implementation, dataset 0.3 metadata/examples, evaluation suites/results, experiment records, and hardware reports.

## Decision summary

- **Validated baseline remains experiment 0006 / Phase 4:** `Qwen/Qwen2-0.5B-Instruct` + LoRA, dataset `miyuki-writing-0.2`, original 12 prompts: **Keep/Edit/Discard = 5/4/3**, meta flags 0.
- Phase 4C is informative evidence about 0.3, not a production promotion: it produced **1/8/3**, did not improve the original benchmark, and showed training/evaluation loss divergence.
- **Do not train yet.** 0.3 contains relevant targeted examples, but the current evidence cannot separate their benefit from the long, resumed three-epoch run’s overfit and regressions. The training code also does not automatically load the best validation checkpoint.
- Once explicitly authorized, the first experiment should be a **single controlled epoch on frozen 0.3** with the Phase 4 recipe, a new isolated checkpoint path and mechanism-created experiment record. Evaluate the epoch-end and available within-epoch checkpoints against Phase 4 before considering any further run.
- Keep Story Brain as the source of story facts and state; train the writing model to use supplied context, not memorize the story universe.

## Current baseline and known failures

Experiment 0006 trained Qwen2-0.5B-Instruct with LoRA on the 120-example 0.2 dataset using assistant-only loss, 39 steps, and the Phase 4 recipe (r16/alpha32, dropout .05, batch 1, accumulation 8, max sequence 1024, learning rate 2e-4, seed 42, CPU float32). Its train loss was 2.62. The original 12-prompt result was **5 Keep / 4 Edit / 3 Discard, meta 0**. This is the best validated candidate and remains the production direction.

Persistent or insufficiently resolved writing failures recorded in the failure taxonomy and Phase 4/4C results:

1. **Exact/binary context:** fails to preserve positive and negative facts (for example, mug ownership, lavender dislike, and other paired alternatives).
2. **Viewpoint constraints:** drifts outside child, first-person, or limited third-person knowledge.
3. **Object/detail retrieval:** does not reliably retrieve or maintain a specified object through action (notebook retrieval remains a failure).
4. **Multi-character voice:** voices collapse or roles become vague, especially in group scenes.
5. **Action continuity:** object movement and cause/effect are not consistently coherent.
6. **Dialogue/subtext:** distinct voices, interruptions, and implied meaning remain weak.
7. **Coherent imagery:** surreal image drift appeared in Phase 4C; vividness and scene quality regressed there.

Meta commentary, hedging essays, explaining instead of writing, moralizing, premature EOS, and original-suite Chinese leakage were substantially controlled by Phase 4 and remain regression checks, not assumed permanent fixes.

## Why Phase 4C is not the production baseline

Phase 4C (the 0.3 data effect run; experiments 0007 and 0008) changed dataset from 0.2 to 0.3 while keeping the intended Phase 4 recipe. Its final evaluated adapter is from the duplicate resumed segment recorded as 0008. That provenance is preserved as recorded; the result must not be simplified into a clean independent 0007 endpoint.

On the same original 12 suite and generation settings, Phase 4C scored **1/8/3**, compared with Phase 4’s **5/4/3**. It did fix or improve some targeted behavior: dry Jinx voice, closer dialogue roles, dialogue behavior, selected mug-ring/continuity details, and Chinese leakage in expanded results. It did not resolve binary context or action retrieval. It regressed several previously kept scene/rewriting outputs, including a very short rewrite and surreal imagery, and weakened an emotional result. Hence it is a useful experiment, but not a better validated model.

### Overfitting evidence

The Phase 4C loss trace fell from about 3.35 toward 0.25, while evaluation loss moved to about 2.86 and plateaued. The run reached 51 steps / nearly 3 epochs on 154 examples; report evidence attributes the behavioral regressions and collapse to overfitting in epochs 2–3. The final loss is not evidence of writing quality. There was also a second resume and auto-recorded duplicate experiment, complicating endpoint provenance. The defensible interpretation is that some 0.3 signals were learned, but the long run did not preserve general writing quality.

## Dataset and experiment lineage

| Dataset | Lineage and status |
|---|---|
| 0.1 | 38 human-authored examples; frozen. |
| 0.2 | 120 examples: 0.1 plus 82 targeted additions; frozen, SHA recorded in manifests, training set of record for the current baseline. |
| 0.3 | 154 examples: preserved 0.2 plus 34 targeted additions; frozen, separate SHA/manifest/review. Experimental, not current production training set. |

0.3’s new examples are groups 121–154: **121–128 voice/dialogue** (different rhythms/reactions and habits); **129–136 binary context** (including grandmother vs mother, chamomile vs lavender, compass/river, signal/route, and heat); **137–142 Jinx dry wit**; **143–148 action/object tracking** (notebook, keys, chair, cup, drawer, phone); and **149–154 viewpoint constraints** (child-limited, first person, third-limited, restrained grief, smell-first, vendor-limited). These map directly to persistent failures. Some exact target examples are useful but highly schematic/paired; they should be evaluated for naturalness and generalization, not presumed effective merely because their review is “keep.”

No individual 0.3 example is demonstrated by repository evidence to have caused Phase 4C’s scene/rewriting regression. It is not supportable to label particular IDs as causes. The regression is associated with the run’s optimization behavior and evaluated output, while 0.3’s new examples are associated with the specific behaviors they were designed to target. Freeze 0.3; do not automatically make 0.4.

Experiment lineage: 0001 pipeline proof; 0002–0003 early 0.1 LoRA; 0004–0005 formulation A/B on derived 0.1; 0006 Phase 4 / 0.2 validated baseline; 0007 intended Phase 4C resume on 0.3; 0008 auto-recorded duplicate resume whose final adapter was evaluated. Preserve all IDs, records, and artifacts. The next available ID being 0009 is not authorization to use it.

## Training implementation and hardware constraints

Phase 4 introduced opt-in `training.assistant_only_loss: true`. For conversational examples, `train.py` creates full chat-template tokenization, computes the prefix ending at the assistant generation marker, masks system/user/prefix labels to `-100`, and leaves assistant output tokens (including assistant end token) trainable. Its custom collator preserves masked labels and pads labels with `-100`. The config default remains false to retain the historical 0001–0005 behavior; Phase 4 and 4C explicitly enable it. The audit diagnostic confirmed 93 masked prompt tokens and 75 assistant tokens trainable in the example. Non-conversational fallback behavior remains all-token loss.

The machine is an i7-1065G7 (4 cores / 8 threads), 16 GB RAM, CPU-only PyTorch; the MX330 is idle for this stack. 0.5B CPU training is slow (Phase 4 took about 49.5 minutes; Phase 4C needed long resumable runs). Inference can drop to roughly 0.4–0.5 tokens/s under memory pressure, and detached training processes have died silently before. The 1.5B model is inference-only on this host: its LoRA preflight left 2.82 GB available, below the 3.0 GB safety threshold. Do not retry 1.5B training here. Measure available RAM before any authorized run, use one process, durable logs/checkpoints, and do not assume the GPU helps.

## A. Recommended Phase 5 experiment sequence

Smallest useful sequence; only proceed through a gate if the previous result warrants it:

1. **Before training:** review 0.3’s 34 examples for behavior clarity and overly templatic targets; validate the frozen dataset and manifests, verify no evaluation contamination, inspect free RAM and training process state. This audit did not modify or revalidate the dataset. Confirm a new output path cannot overwrite prior artifacts. The Phase 4C config is 3 epochs and points at its existing output; it cannot be reused unchanged.
2. **One controlled 0.3 epoch:** hold base model, LoRA, assistant-only loss, optimizer, seed, split, sequence length, and generation recipe constant with the Phase 4 recipe; change only dataset and epoch budget relative to the Phase 4 baseline. Use a fresh Phase 5 output path and the experiment tracker. Save checkpoints at the established five-step interval and evaluate candidate checkpoints, including epoch end. At 138 train examples, the seed-42 90/10 split yields about 17 optimizer steps for one epoch. Expected wall time is roughly 17–20 minutes by Phase 4 throughput, plus load/evaluation; it is an estimate, not a guarantee.
3. **Compare all candidates to Phase 4, then stop.** Select only by validation loss as a diagnostic plus full behavioral evidence; do not select the final checkpoint by default. Do not continue directly into epoch 2. If no candidate improves without regressions, retain Phase 4 and stop training.
4. **Only after reviewing the one-epoch evidence:** if targeted behaviors improve but broader writing is stable and validation has not turned upward, consider a separately authorized continuation/configuration using the best checkpoint. If validation worsens or full-suite behavior regresses, stop and investigate data quality/format before any new dataset or training run. Never silently change multiple variables.

Why not choose two epochs now: Phase 4C already shows divergence during later epochs, while 0.3 provides only 16 held-out examples and the code sets `load_best_model_at_end=False`; its eval loss is a limited signal and its final saved model is not guaranteed best. One epoch with checkpoint-level behavioral selection minimizes exposure and measures whether the targeted examples help before the known late-epoch risk. A 1-epoch result is an experiment, not a presumed improvement.

## B. Evaluation strategy and gates

Compare each candidate with **the exact Phase 4 adapter**, using identical prompt files, input composition, and generation settings: max 180 tokens, temperature .8, top-p .95, repetition penalty 1.1, seed 42. Keep Phase 4 outputs immutable. Report suites independently:

- **Original 12:** preserve the existing human Keep/Edit/Discard rubric and nine dimensions; baseline is 5/4/3, meta 0. Review prompt-level changes, not only aggregate counts.
- **Expanded 12:** use the frozen `phase4_expanded.jsonl`. Prior Phase 4 vs Phase 4C expanded runs used different output budgets (60 vs 180), so they are not fully apples-to-apples; rerun both baseline and candidate with matched settings if making a new comparative claim.
- **Targeted 5:** retain `targeted_scale.jsonl` for voice, binary context, viewpoint, and action/object probes. These are diagnostic probes, reported separately, not substituted for the original benchmark.
- **Regression checks:** meta/hedging, Chinese leakage, premature EOS, output length, errors, and repeated/collapsed text; inspect factual consistency, scene coherence, vividness, and quality of rewriting. Compare runtime/tokens per second and failures under the same machine conditions.

Human evaluation should be blind to candidate identity where practical and reuse the existing template. Record Keep/Edit/Discard and prose quality, naturalness, character consistency, dialogue, pacing, creativity, instruction adherence, continuity, usefulness, plus evidence-based taxonomy labels (fixed/improved/partial/persistent/regressed/unknown). Automated flags are triage signals, not substitutes for human ratings.

**Promotion gate:** candidate must at least match baseline Keep count and not increase Discard count on original 12; targeted gains must occur on more than one relevant probe and align with the intended categories; expanded suite must show no material broad regression; no serious new instruction, continuity, meta, language, or coherence regression. Any aggregate trade-off requires prompt-level review and retains Phase 4 as baseline until explicitly resolved. Small suite counts are noisy; no single sample proves improvement.

**Validation-loss gate:** capture per-checkpoint train and eval loss. A falling train loss with stalled/rising eval loss is a stop signal, not a reason for more epochs. The 16-example held-out split is noisy and only a training diagnostic; human behavioral evaluation is the selection gate.

Only add automated tests that guard a concrete risk: preserve 0.3/Phase 4 hashes and experiment isolation; verify assistant-only label masking and deterministic split/step expectations; verify candidate evaluation uses frozen prompts and same generation config; verify best-checkpoint selection/path provenance. Existing tests cover several of these already, so inspect coverage before adding any. Do not add tests just to increase count.

## C. Dataset decision

**Use 0.3 only for the first controlled, explicitly authorized Phase 5 candidate; do not promote it as the new default and do not create 0.4.** It contains direct targeted coverage absent or sparse in 0.2:

- Voice/dialogue: 121–128.
- Binary facts/negative alternatives: 129–136.
- Jinx’s short dry response followed by competence: 137–142.
- Object location/state through action: 143–148.
- Child, first-person, third-person, restrained, sensory-order, and bounded-knowledge viewpoints: 149–154.

The possible weakness is not proven causation: many binary drills are tightly specified and examples are compact; some Jinx examples risk making a joke formula; object examples encode one canonical tracking pattern. Inspect these for target quality and variety before training. The review marks all as keep, but the Phase 4C outcome shows human review and training fit do not guarantee generalization. Phase 4C’s surreal imagery and short rewriting are observed model regressions, not attributable to named data rows from current evidence.

## D. Training decision

**Decision: no training yet.** This readiness request does not authorize training, and `AGENTS.md` explicitly requires milestone authorization. Repository docs propose a 1–2 epoch 0.3 run but also explicitly state it is not started/not authorized; that proposal is not permission. Do not create experiment 0009, checkpoint files, or modified data as part of this audit.

After explicit authorization, the exact first experiment is: **one epoch, Qwen2-0.5B-Instruct + same Phase 4 LoRA and assistant-only-loss configuration, frozen `miyuki-writing-0.3` SHA, seed 42 and same deterministic 90/10 split; otherwise Phase 4 recipe unchanged; fresh `checkpoints/phase5-0.3-1ep/` output path; new mechanism-created experiment record (next ID only if tracker confirms it); five-step checkpoints and per-checkpoint eval-loss capture; evaluate candidate checkpoints against Phase 4 on matched original 12, expanded 12, and targeted 5 suites before deciding whether to keep or stop.** Do not use the existing Phase 4C output path. This run must be preceded by RAM/process preflight and an operational method to load/evaluate each checkpoint without accidental resume or overwrite. No second epoch without a separate decision after reviewing results.

## E. Production architecture

The separation of **Writing Model + Story Brain** remains appropriate and is a prerequisite to dependable story continuity. The writing model should convert context into prose and follow constraints; it should not be expected to store canon across sessions or memorize a universe in LoRA weights. Story Brain should eventually retrieve and supply relevant characters, relationships, chronology, world facts, research with provenance, scene state, persistent canon, author intent, and recent prose. It should track uncertainty/conflicts and provide only relevant context. This audit recommends the architecture; it does not implement Story Brain.

## Production-readiness checklist

- [ ] Phase 4 / exp 0006 remains recoverable and the named baseline artifacts are immutable.
- [ ] Dataset and experiment hashes/lineage are recorded; evaluation prompts are excluded from training.
- [ ] Assistant-only loss and its mask/assistant-token counts are verified for the actual run.
- [ ] CPU/RAM preflight passes; only one process runs; logs and checkpoints are durable and resumable.
- [ ] Best candidate checkpoint can be identified and loaded; it is not assumed to be the final adapter.
- [ ] Original 12, expanded 12, and targeted 5 are run with matched settings against Phase 4.
- [ ] Human Keep/Edit/Discard and nine dimensions are complete; taxonomy and automated regression flags are recorded.
- [ ] No broad regression in scenes, rewriting, instruction following, continuity, language, or output completion.
- [ ] Candidate passes the promotion gate; otherwise Phase 4 remains production baseline.
- [ ] Story Brain supplies relevant story facts/state at runtime; writing model is not treated as persistent canon storage.

## Files and tests

This audit changed only this readiness document. No tests were run: the request was a repository readiness audit, not an implementation or training task, and the relevant claims were checked against existing recorded test/results and artifacts. No training, evaluation generation, dataset mutation, checkpoint mutation, experiment creation, commit, or push was performed.
