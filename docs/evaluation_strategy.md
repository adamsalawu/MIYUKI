# Evaluation Strategy

## Philosophy

Automated metrics cannot judge creative writing. Use **small, hand-crafted prompts** + **human ratings** + **comparative runs** on identical prompts.

## Baseline Evaluation Set

Located at `evaluation/prompts/eval_v0.1.jsonl` — 12 prompts covering:

- continuation
- dialogue
- scene writing
- character interaction
- emotional scene
- action scene
- quiet scene
- description
- rewriting
- character consistency
- tone consistency
- context retention (binary pass/fail)

Quality > quantity. Each prompt has `id`, `category`, `prompt`, `instruction`, `context`.

**Never train on evaluation prompts.**

## Automated Runner

`src/miyuki/evaluation/runner.py` — `miyuki evaluate --config <yaml>`

1. Loads `MiyukiConfig`
2. Loads prompts
3. Composes `Context + Instruction + Prompt` → `full_prompt`
4. Calls inference for each
5. Saves `evaluation/runs/<YYYY-MM-DD>_<model>/outputs.jsonl` + `metadata.json`
6. Writes `evaluation/reports/<YYYY-MM-DD>_<model>.json`

Run identical prompts for `BASE vs MIYUKI-0.1 vs MIYUKI-0.2` and compare.

## Human Evaluation

Template: `evaluation/human_template.yaml`.

Per output, rate 1–5 + notes:

- prose quality
- naturalness
- character consistency
- dialogue quality
- pacing
- creativity
- instruction adherence
- continuity
- overall usefulness

Plus `keep | edit | discard` and raw notes. Keep raw notes; do not collapse to a single meaningless average unless justified.

## Future Automated Signals (not in Phase 0)

- Embedding similarity for diversity / repetition detection
- Lightweight rule checks: repeated n-grams, exposition density, continuity keyword check
- These supplement, never replace, human judgment.

## Linking to Experiments

Training records `experiment_id` + `final_loss` + `artifact_location`. Evaluation reports reference `model` + `run_dir`. Together they let you answer "was 0.2 actually better than 0.1?"
