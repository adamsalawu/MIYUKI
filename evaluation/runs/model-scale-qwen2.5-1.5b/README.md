# Model Scale Test — Qwen2.5-1.5B-Instruct Base (No Training)

**Hypothesis:** A larger, newer 1.5B instruction model is already substantially better for MIYUKI writing tasks before fine-tuning.

**Model:** `Qwen/Qwen2.5-1.5B-Instruct` (1,543,714,304 params, Qwen2, vocab 151665, 32K context, chat True) via `configs/model/qwen2.5-1.5b-instruct.yaml` (selectable, 0.5B configs preserved).

**Runtime:** Same `TransformersRuntime` (AutoTokenizer/AutoModel + Qwen chat template), CPU float32, no CUDA, no quantization (architecturally clean). Load 451.4s, 3.10 GB download, RAM 5.74→1.58 GB avail (90.7% used), tok/s 0.7–1.78 sanity / 1.53 benchmark avg (vs Phase4 2.27, base 2.84).

**Benchmark:** EXACT same 12 prompts `evaluation/prompts/eval_v0.1.jsonl`, same generation (180 max, temp0.8 top_p0.95 rep1.1 seed42). Outputs `outputs.jsonl` (1971 toks avg164, 1346s, 0 errors) + `metadata.json` + `comparison.json` + `human_eval.yaml`. Targeted 5 prompts `evaluation/prompts/targeted_scale.jsonl` → `targeted_eval.jsonl` (507s).

**Human:** Same 9 dims + keep/edit/discard. **1.5B base: 2 keep /5 edit /5 discard** (scene, rewriting keep; continuation, emotional, action, quiet, Jinx edit; dialogue, interaction, description, tone, context discard) vs **Phase1 0/3/9, 3B 1/2/9, Phase4 5/4/3**.

**Key findings:**
- More fluent/verbose (1971 vs 995 toks), best action pacing and Jinx voice attempt, ties Phase4 on scene/rewriting/quiet/emotional.
- Worse than Phase4 adapter on anti-meta: tone `Certainly! Here's your sentence...` intro, context numbered-list essay (hedging like Phase1), description second-person not child POV.
- Sanity false refusal on lighthouse (`not able to provide answers on political matters`) — misalignment risk.
- Targeted: Jinx 41-token witty (`Those idiots at headquarters`) is best Jinx yet, but context accepts lavender + silver-mug hallucination (binary violation, worse than Phase4).

**Resources:** See `resource_report.json`, `sanity.json`. Fits RAM but no headroom for LoRA training (would need ~8GB+ for optimizer).

**Recommendation:** See `docs/model_scale_test.md` — CASE B (only slightly better, specific categories). Do NOT move backend yet; small LoRA test on 1.5B with `miyuki-writing-0.2` is justified but NOT run in this milestone.

**Not done:** No LoRA, no dataset conversion, no training, no checkpoint, no Story Brain. Phase4 adapter remains current backend.
