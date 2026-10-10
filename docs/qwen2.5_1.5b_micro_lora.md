# Qwen2.5-1.5B Micro-LoRA — RESULT D: Memory Preflight NO-GO

**Hypothesis:** Small LoRA (r16, 1 epoch, `miyuki-writing-0.2` 120 ex, assistant-only) transfers MIYUKI behaviors (anti-meta, quiet, rewriting, tone, grounding) onto 1.5B while preserving its Jinx/action strengths.

## Hardware Preflight (Actual)
- Total 16.92 GB, avail **9.47 GB before** → **2.81 GB after load** → **2.82 GB after LoRA** (83.4% used), load 17.3s warm (451s cold in scale test), CPU i7-1065G7, CUDA False, MX330 idle, torch 2.14.0+cpu, transformers 4.41.0
- Model 1,543,714,304 params, LoRA trainable **18,464,768 / 1,562,179,072 (1.18%)**, targets `[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]`, tokenizer 151665 chat True
- Threshold 3.0 GB avail required for gradients (18M LoRA) + activations (batch1 seq1024, 28 layers) + Adam states + dataloader. **2.82 < 3.0 → NO-GO.** See `evaluation/runs/qwen2.5-1.5b-micro-lora/preflight.json`, `tools/micro_preflight.py`.
- Training estimate (pre-training, not exact): 120 ex → 108 train/12 eval (90/10 seed42), batch1 accum8 → ~13 optimizer steps for 1 epoch (108/8=13.5, Phase4 got 13/epoch), ~108 batches, ~45–65 min (0.5B 76s/step ×3–4 for 1.5B) + load. Printed in `configs/training/qwen2.5-1.5b_micro_lora.yaml` header.

## Memory Behavior
- Scale inference already at 90.7% (1.58 GB avail). Preflight confirms training would leave <2.8 GB for activations/optimizer — critically constrained, thrash risk. Guard worked as designed. No severe pressure induced (no training started, system stable).

## Training Configuration (Created, NOT Executed)
`configs/training/qwen2.5-1.5b_micro_lora.yaml` — model 1.5B, dataset 0.2 SHA `8cda31...`, `assistant_only_loss: true` (verified preserved), r16/alpha32/dropout0.05/bias-none, batch1/accum8/max1024/lr2e-4/seed42/CPU float32, **epochs 1** (vs Phase4 3), output `checkpoints/qwen2.5-1.5b-micro-lora/` (not created). Separate from `phase4_lora.yaml` (unchanged).

## Optimizer Steps / Loss / Runtime
- **None** — training NOT run per NO-GO. No loss, no steps, no runtime. Preflight trainable 1.18% recorded only.

## Checkpoint
- **None** — `checkpoints/qwen2.5-1.5b-micro-lora/` verified absent. Phase4 `checkpoints/phase4-miyuki-0.2/adapter_config.json` + `experiments/0006` preserved. No merge, no corrupted tensors (nothing to corrupt).

## Original Benchmark (Micro-LoRA)
- **NOT APPLICABLE** (no adapter). Existing 4-way stands: Phase1 0/3/9 (1538 toks), 3B 1/2/9 (996), Phase4 **5/4/3** (995, meta 0), 1.5B base 2/5/5 (1971, tone intro + context essay). No micro outputs.jsonl.

## Targeted Benchmark (Micro-LoRA)
- **NOT APPLICABLE.** Existing targeted (`targeted_scale.jsonl`, 5 prompts): 1.5B Jinx 41-token witty best yet, but context silver-mug + accepts lavender (binary violation, worse than Phase4 hands-back-cup). No micro targeted.

## Human Evaluation (Micro-LoRA)
- **NOT APPLICABLE.** Existing: Phase4 5/4/3 vs 1.5B base 2/5/5 (same 9 dims). Micro would have compared Phase4/1.5B-base/micro — cannot without adapter.

## Regression Analysis (Micro-LoRA)
- **NOT APPLICABLE.** Desired (1.5B raw + MIYUKI behavior) untested. Known from base: anti-meta fails tone/context, quiet ok, rewriting ok, grounding ok, description child-POV fail, Jinx best, action pacing best but no retrieval.

## Overfitting Analysis (Micro-LoRA)
- **NOT APPLICABLE** (no training). Existing Phase4: 0 verbatim 40-char, 0 term injection original; expanded 1 Chinese (松开). One epoch would still need checks — deferred until hardware allows.

## Recommendation — **RESULT D: MEMORY FAILURE**
Do NOT fight the machine. **Keep 1.5B as inference-only candidate, Phase4 0.5B adapter remains backend.** No evidence micro-LoRA would fail behaviorally — it was not tested — but hardware cannot safely run it (2.82 < 3.0 GB, 83% used before gradients). Next: either (a) CPU training on machine with ≥32 GB RAM, or (b) parameter-efficient alternative requiring less activation memory (not attempted here per hard rules, no random quantization), or (c) continue 0.5B dataset work (30–40 more dialogue/Jinx/binary examples). Do NOT auto-run 2-epoch or quantized training.

## Artifacts
- `configs/training/qwen2.5-1.5b_micro_lora.yaml` (new, Phase4 unchanged)
- `evaluation/runs/qwen2.5-1.5b-micro-lora/{preflight.json, resource_report.json, README.md}` (no outputs.jsonl — none possible)
- `tools/micro_preflight.py`
- Preserved: 0001–0006, phase4 adapter/checkpoint/dataset, 1.5B base eval from scale test
- Tests: `tests/test_micro_lora.py` (8 new, see below)

## Tests
- **Total: 88 passed** (81 prior + 7 new micro? Actually 8 new, 1 skipped? See final report — split batches, 2 warnings). New: micro config, 1.5B identifier, assistant_only true, Phase4 unchanged, dataset unchanged, deterministic eval (12 IDs), checkpoint absence (NO-GO proof), adapter reload N/A documented, memory guard (threshold logic + NO-GO recorded), experiment metadata (no 0007).
