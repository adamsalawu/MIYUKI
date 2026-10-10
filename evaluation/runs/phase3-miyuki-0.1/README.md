# Phase 3 — First Writing SFT / LoRA (miyuki-writing-0.1 on Qwen2-0.5B)

**Status:** Trained and evaluated; hypothesis not supported with this conservative config — no measurable writing improvement vs Phase 1 baseline.

## Training

- **Base:** `Qwen/Qwen2-0.5B-Instruct` (494M, vocab 151646, model vocab 151936, context 4096)
- **Tokenizer:** Production Qwen tokenizer (no vocab change, chat template True)
- **Dataset:** `miyuki-writing-0.1` — 38 human-authored, human-edited, 21738 chars, 5072 toks, SHA256 `9e1b97461f989482...`, validated `38/38`
- **LoRA:** `r 16, alpha 32, dropout 0.05, bias none, target_modules [q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]` (Qwen2 auto, verified via `model.layers.0.*` inspection), trainable 8,798,208 / 502,830,976 (1.75%), `TaskType CAUSAL_LM`
- **Training:** `configs/training/phase3_lora.yaml` — `epochs 1, batch 1, accum 8, max_seq 1024, lr 1e-4, weight_decay 0.01, warmup 0.03, cosine, seed 42, CPU float32, i7-1065G7 16GB (MX330 unused), torch 2.14.0+cpu`
- **Output:** `checkpoints/phase3-miyuki-0.1` — `adapter_config.json (760 B)`, `adapter_model.safetensors (35.2 MB)`, `train_metrics.json`, `tokenizer.json`, `miyuki_training_config.yaml`, `metadata.json`, `dataset_info.json`
- **Metrics:** `global_step 4` (34 train examples / (batch1 * accum8) ≈4 optimizer steps), `epoch 0.94`, `train_runtime 375.4 s`, `elapsed 382.4 s`, `train_loss 3.762` (from 4.06→3.46, decreasing), `train_samples_per_sec 0.091`
- **Experiment:** `experiments/0002/metrics.json` — `model Qwen/Qwen2-0.5B-Instruct, dataset miyuki-writing-0.1, dataset_version 0.1.0-phase3, hardware i7-1065G7, duration 382s, final_loss 3.762, adapter_location checkpoints/phase3-miyuki-0.1`

## Smoke Test

Before full training, smoke with real Qwen+LoRA on 4 examples, `max_steps 2`, `save_steps 1`: `trainable 8.7M (1.75%)`, `steps 2, loss 3.81, eval loss 3.56→3.31`, checkpoint `checkpoints/phase3-smoke/checkpoint-2/adapter_config.json`, adapter load ok, generation ok (`Elena hesitates before accepting lavender tea...`), base without adapter still loads. See `tools/smoke_phase3.py`.

## Round-Trip Verification

- Base without adapter: `MiyukiConfig small-instruct.yaml` → `TransformersRuntime.load()` → `prompt 44 toks, 60 max, generated 'it was time to get back to her ship...'` (coherent, not random)
- With adapter via config (`phase3_lora.yaml` auto-loads `checkpoints/phase3-miyuki-0.1`): `is Peft True`, generates `'it was time to get back to shore. She stood in awe...'` (different, adapter effect), `chat_template_used True`, `not random 'atedton'`
- Explicit `load_adapter_from('checkpoints/phase3-miyuki-0.1')` also works, same output as auto-load. Tokenizer mismatch none (151646 both). No corrupted checkpoint.

## Evaluation (12 prompts, apples-to-apples with Phase 1)

- **Prompts:** `evaluation/prompts/eval_v0.1.jsonl` (12 categories: continuation, dialogue, scene_writing, character_interaction, emotional_scene, action_scene, quiet_scene, description, rewriting, character_consistency, tone_consistency, context_retention) — same wording/order/structure.
- **Run:** `python -m miyuki.training.train` with Phase 3 config then `python -c "run_evaluation(model_config=cfg, prompts_file='eval_v0.1.jsonl', output_dir='evaluation/runs/phase3-miyuki-0.1')"` — `evaluation/runs/phase3-miyuki-0.1/2026-09-24_Qwen_Qwen2-0.5B-Instruct/outputs.jsonl` (12 entries, `num_errors 0`, `load_s 9.08`, `elapsed 514.9`, `total_generation 503.4 s`)
- **Also copied:** `evaluation/runs/phase3-miyuki-0.1/outputs.jsonl`, `metadata.json`, `summary.json` (objective metrics) for versioned access.
- **Phase 1 vs Phase 3 objective:**
  - Prompt tokens: both 956 (same prompts + chat template)
  - Output tokens: Phase1 1538 (avg 128) vs Phase3 1698 (avg 141.5) — Phase3 slightly longer
  - Avg tps: 2.84 vs 3.31 (adapter slightly faster, but CPU variance)
  - Total gen: 541.3 vs 503.4, elapsed 543.8 vs 514.9 — similar
  - Truncation: Phase1 had 4 truncated early (quiet 42, rewriting 29, Jinx 12, tone 84); Phase3 similar (rewriting 40, Jinx 12, quiet 103, tone 126) — no fix for early EOS.

## Human Evaluation

- **Template:** Same 9 dims as Phase 1 (`prose_quality, naturalness, character_consistency, dialogue_quality, pacing, creativity, instruction_adherence, continuity, overall_usefulness` + `keep/edit/discard`)
- **File:** `evaluation/runs/phase3-miyuki-0.1/human_eval.yaml` — rated per prompt, with `phase1_vs_phase3` notes.
- **Result:** Phase1 `9 discard, 3 edit, 0 keep` → Phase3 `10 discard, 2 edit, 0 keep` (no prompt moved to better tier; 1 regression with Chinese, 3 identical, 5 same failure class). Best were continuation/scene-writing edits (2), but overall **no measurable improvement**.

## Failure Analysis (Phase 1 failures → Phase 3)

- **Meta-analysis (quiet, tone, rewriting, context):** Still fails — quiet `The answer is: As the train begins to move...`, tone `The narrator's voice is soft...`, context `Elena's reaction might be one of disgust... depending...` (hedging). Dataset had 5 anti-meta exemplars (007,011,016,017,036) but 4 steps insufficient to learn.
- **Generic AI phrasing / weak grounding:** Still generic (`air heavy with anticipation`, `familiar warmth reflecting off face` vs Phase1 `wind of history`). No distinct improvement.
- **Flat dialogue:** Still flat (Phase1 `Sister: Mom?` vs Phase3 `Hi, Mom... Sister... Sorry sis...`), no interruptions/distinct voices despite 6 dialogue exemplars.
- **Action incoherence:** **Regression** — Phase3 action contains Chinese `狭窄的街道，快速的脚步声...` mixed English/Chinese mid-sentence, not in Phase1, not in dataset (English only). Indicates multilingual leakage.
- **Character consistency (Jinx):** Identical 12-token truncation `Jinx, I know it sounds ridiculous but...` — no learning.
- **Lore dump / exposition:** Still essay dump (`The mentor has been teaching...` both).
- **Repetition/cliché:** Still cliché coil.
- **Conclusion:** Loss 4.06→3.46→3.76 decreasing did **not** translate to writing improvement (as warned in Phase 2). Hypothesis not supported with this config.

## Overfitting Check

- **Paraphrased prompts (4):** Haru tea, Elena mug, Jinx compass, Aureline ribbon — all generated without copied 30-char snippets from dataset, no name overuse (>2), no meta-answer, not empty. General factual prompt `Explain why sky is blue` correctly answered factually, not as scene — general instruction following not degraded.
- **Memorization in eval:** 12 eval outputs checked for 40-char verbatim dataset snippets → **none found**.
- **Stylistic injection:** Invented terms `lumenweave/scryglass/voidtide/aureline` appear 0 times in eval outputs (unless prompt contains them) — no over-injection.
- **Outside categories:** Factual answer still correct → no degradation outside trained categories.
- **Conclusion:** No obvious memorization/overfitting, but also no strong imitation of dataset's good prose — underfitting.

## Baseline Preservation

- Phase 1 Qwen baseline remains reproducible: `evaluation/runs/phase1_qwen2_0.5b/outputs.jsonl` + `metadata.json` + `human_eval.yaml` untouched; `configs/model/small-instruct.yaml` unchanged (adapter disabled). Phase 3 stored separately (`checkpoints/phase3-miyuki-0.1`, `experiments/0002`).

## Where Outputs Live

- **Checkpoint:** `checkpoints/phase3-miyuki-0.1/{adapter_config.json, adapter_model.safetensors, train_metrics.json, metadata.json, tokenizer.json, miyuki_training_config.yaml}`
- **Experiment:** `experiments/0002/{metrics.json, config.yaml, README.md}`
- **Evaluation:** `evaluation/runs/phase3-miyuki-0.1/{outputs.jsonl, metadata.json, summary.json, human_eval.yaml, 2026-09-24_Qwen.../outputs.jsonl}` + `evaluation/reports/2026-09-24_Qwen...json` (overwrote previous report with Phase3 version but dated run preserved; Phase1 report at `evaluation/runs/phase1_qwen2_0.5b/report.json`)
- **Smoke:** `checkpoints/phase3-smoke/checkpoint-2/`

## Reproduction

```powershell
python -m miyuki.cli dataset validate --path datasets/miyuki-writing-0.1/dataset.jsonl --format conversational
python tools/smoke_phase3.py
python -c "from miyuki.training.train import train; train(config_path='configs/training/phase3_lora.yaml', dataset_path='datasets/miyuki-writing-0.1/dataset.jsonl')"
python -m miyuki.cli evaluate --config configs/training/phase3_lora.yaml  # or via run_evaluation with phase3 config
Get-Content evaluation/runs/phase3-miyuki-0.1/summary.json
```

## Known Limitations

- Only 4 optimizer steps (1 epoch, 38 ex, accum 8) — conservative but too weak for measurable behavior change.
- CPU float32 slow (0.02 steps/sec) but reliable.
- Chat template quoting edge (embedded `"` in Jinx prompt) still causes early EOS 12 toks for both baseline and LoRA.
- Phase 3 still produces meta-answers where Phase 1 did, indicating dataset not yet learned.

## Not Changed

- Qwen tokenizer unchanged (151646), no vocab addition, no `resize_token_embeddings` on production model (per spec).
- No Story Brain/Research/Character/Memory/GUI/more models.
- No architecture redesign, no baseline overwrite, no auto-commit.
