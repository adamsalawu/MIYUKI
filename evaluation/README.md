# evaluation

Prompts: `evaluation/prompts/eval_v0.1.jsonl` (12, versioned, never train on these)
Runner: `src/miyuki/evaluation/runner.py` — `miyuki evaluate --config <yaml>`
Human template: `evaluation/human_template.yaml`
Outputs: `evaluation/runs/<date>_<model>/` + `evaluation/reports/`

Compare: BASE vs MIYUKI-0.1 vs MIYUKI-0.2 on identical prompts.
