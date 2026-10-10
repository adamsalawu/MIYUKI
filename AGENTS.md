# MIYUKI — Codex Operating Contract

> This file is the permanent operating contract for Codex sessions in this repository.
> The repository itself is the source of truth. If this file conflicts with actual
> files or measured results, trust the files. See `docs/opencode_to_codex_migration.md`
> for the full history and `docs/current_state.md` for the live operational snapshot.

## 1. Project Identity

MIYUKI is a **local-first AI-assisted story-writing system** — the evolution of the
OffScript writing app. It is NOT merely a text generator.

The eventual system has two halves, designed together:

- **Writing model** (this repo, the Model Lab): trained and evaluated specifically
  for story writing — narration, dialogue, character voice, continuity, pacing.
- **Story Brain** (future application): owns persistent structured story state —
  characters, relationships, timeline, lore, scene state, author notes, research —
  and assembles context for the model. The model must NOT own story memory.

The model is trained to **USE provided context correctly** (characters, constraints,
canon). It does not need every story fact memorized in weights.

Target behavior is defined in `docs/miyuki_writing_spec.md`. The model should WRITE
when asked (continue, dialogue, rewrite, scene) — never explain how to write, never
announce, never moralize, unless the task explicitly asks for explanation.

## 2. Current Model Strategy (verified 2026-09-29)

- **Production direction: `Qwen/Qwen2-0.5B-Instruct` + LoRA.** All training on this
  machine is CPU-only (`torch 2.14.0+cpu`, no CUDA, 16 GB RAM, MX330 idle).
- **Current best validated candidate: Phase 4 adapter**
  (`checkpoints/phase4-miyuki-0.2/`, dataset `miyuki-writing-0.2`, 120 ex,
  assistant-only loss, r16/alpha32, 39 steps, train loss 2.62).
  Original 12-prompt benchmark: **5 Keep / 4 Edit / 3 Discard, meta 0**.
- **`Qwen/Qwen2.5-1.5B-Instruct` is inference-only.** LoRA training was correctly
  stopped at memory preflight (2.82 GB < 3.0 GB threshold). Do NOT casually retry
  1.5B training. Do NOT change model size without an explicit controlled experiment.
- **Do not train anything unless the milestone explicitly authorizes it.**
  Dataset expansion, audits, and evaluation are valid non-training milestones.

## 3. Training Principles

- **Preserve assistant-only loss.** `training.assistant_only_loss: true` masks
  system/user tokens (`-100`); only assistant writing tokens are trainable.
  See `docs/training_signal_audit.md`. Default in `src/miyuki/config.py` is `false`
  to preserve historical experiments 0001–0005 — never change that default.
- **Immutable datasets.** Never silently modify an existing dataset version.
  New data = new version (`miyuki-writing-0.4`, …) with new manifest + SHA.
- **Never overwrite historical experiments** (`experiments/0001`–`0008`),
  checkpoints, or evaluation results. New work gets new IDs/paths.
- **One major variable at a time** (dataset OR epochs OR formulation — never all).
- **Evaluate before declaring improvement.** Same prompts, same generation settings
  (`180 / 0.8 / 0.95 / 1.1 / seed 42`), Keep/Edit/Discard + existing 9 dimensions.
- **Never select a model on training loss.** Phase 4C proved loss collapse
  (0.25) can accompany behavioral regression (eval plateau 2.86).
- **Guard against overfitting.** Known threshold: 154 ex × 3 epochs overfits.
  Prefer the best validated checkpoint, not the final one. If validation quality
  stalls while train loss falls: STOP.
- **Memory/hardware guard is mandatory.** Measure RAM before any training run;
  refuse unsafe runs (see `tools/micro_preflight.py` pattern). Never retry a known
  memory failure. Never launch duplicate training processes.

## 4. Dataset Principles

- Human-authored / human-edited examples preferred. Every example needs a
  deliberate training purpose (documented in `notes` metadata, never in the target).
- Four schemas in `src/miyuki/datasets/schemas.py`: `raw`, `prompt_completion`,
  `conversational` (TRL SFT), `story` (converted at train time).
- Validate before training (`miyuki dataset validate`): missing fields, empty
  content, malformed conversations, duplicates (SHA256), tiny examples,
  train/eval contamination. Check near-duplicates (>0.75 normalized ratio).
- Every version has a manifest with SHA, token stats, splits, provenance.
  Current SHAs: 0.1 `9e1b9746…`, 0.2 `8cda31a9…`, 0.3 `f3f5f636…`.
- **Never train on evaluation prompts** (`evaluation/prompts/*.jsonl`).

## 5. Evaluation Principles

- Keep/Edit/Discard is the primary human metric — never replace it.
- Two suites, always: **original 12** (`eval_v0.1.jsonl`) + **expanded 12**
  (`phase4_expanded.jsonl`). Report separately. Never modify old suites.
- Targeted probes (`targeted_scale.jsonl`, 5 prompts) for voice/binary/POV/retrieval.
- Failure taxonomy: `evaluation/failure_taxonomy.json` + `docs/failure_taxonomy.md`.
  Classify every failure as fixed / improved / partially improved / persistent /
  regressed / unknown — with evaluation evidence, never assumptions.
- One good sample proves nothing. Compare full-suite human evals.
- Track automated flags too: meta phrases, hedging, Chinese leakage, premature EOS,
  output length, tok/s, errors. See `tools/phase4c_compare.py` pattern.

## 6. Agent Workflow

`inspect → understand → plan → implement → build/train → test → evaluate → verify → report → stop`

Before modifying important systems:
- Inspect the existing implementation and its dependents.
- Identify historical constraints (read the relevant phase report first).
- Make the smallest justified change.
- Run relevant tests; verify the actual result (open the output files).

Rules:
- Never restart an experiment that has a valid checkpoint unless instructed
  (prefer resume; see `tools/run_phase4c_resume.py` pattern).
- Never silently change hyperparameters to make a run easier.
- Never delete old experiments/checkpoints to "clean up".
- Never fake experiment records — the recording mechanism must create them.
- Never weaken, skip, or rewrite tests to make them pass.
- Never train on evaluation data. Never report training loss as quality.
- **Never commit or push.** Leave changes in the working tree.

## 7. Key Locations

| What | Where |
|---|---|
| Model/training/inference/eval configs | `configs/{model,training,inference,evaluation}/` |
| Source | `src/miyuki/{config,cli,hardware,datasets,training,runtime,inference,evaluation,experiments,artifacts}/` |
| Training entry | `src/miyuki/training/train.py` |
| Eval runner | `src/miyuki/evaluation/runner.py` |
| Human-eval template (9 dims) | `evaluation/human_template.yaml` |
| Eval suites | `evaluation/prompts/{eval_v0.1,phase4_expanded,targeted_scale}.jsonl` |
| Eval runs | `evaluation/runs/<phase>-*/` (outputs.jsonl, metadata.json, human_eval.yaml) |
| Datasets | `datasets/miyuki-writing-{0.1,0.2,0.3}/` + `data/manifests/` |
| Experiments | `experiments/0001`–`0008/` (metrics.json, config.yaml) |
| Checkpoints/adapters | `checkpoints/<phase>-*/` (adapter_model.safetensors) |
| Phase reports | `docs/phase{2,3,3b,3c,4,4b_dataset_expansion,4c}_report.md` |
| Strategy docs | `docs/{architecture,model_strategy,dataset_strategy,evaluation_strategy,roadmap}.md` |
| Failure taxonomy | `evaluation/failure_taxonomy.json`, `docs/failure_taxonomy.md` |

## 8. Current State (snapshot; details in `docs/current_state.md`)

- Baseline: **Phase 4 adapter** (`checkpoints/phase4-miyuki-0.2/`, exp 0006).
- Phase 4C (0.3 dataset, exp 0007+0008): dataset effect measured — did NOT improve
  original benchmark (1/8/3, overfit epochs 2–3). Some targeted behaviors learned.
- Next authorized question (not started): controlled **1–2 epoch run on 0.3** to
  isolate overfitting from data effect, OR revert to 0.2 recipe. Do not start
  training until a milestone explicitly authorizes it.
- **DO NOT DO YET:** Story Brain, Research, Memory, GUI, 1.5B training, new
  datasets, Phase 5 training, commits.
