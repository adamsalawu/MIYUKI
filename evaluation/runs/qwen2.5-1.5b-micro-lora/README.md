# Qwen2.5-1.5B Micro-LoRA — RESULT D: Memory Preflight NO-GO (No Training)

**Hypothesis:** Small LoRA (r16, 1 epoch, 120 ex, assistant-only) transfers MIYUKI behaviors onto 1.5B while preserving Jinx/action strengths.

**Preflight (actual, not predicted):** Loaded 1.5B exactly as scale benchmark + attached LoRA r16 (trainable 18,464,768 / 1,562,179,072 = 1.18%). RAM 9.47 GB avail before → 2.81 GB after load → 2.82 GB after LoRA (83.4% used). Threshold 3.0 GB required for gradients/activations/optimizer/dataloader. **Decision: NO-GO (2.82 < 3.0).** See `preflight.json` (17.3s warm load; cold 451s in scale test).

**Training:** NOT attempted per guard. No checkpoint `checkpoints/qwen2.5-1.5b-micro-lora/` created (verified absent). No experiment 0007 created (0006 remains latest). No quantization hacks attempted.

**Evaluation for micro-LoRA:** NOT APPLICABLE (no adapter). Existing comparison stands:
- Phase1 0.5B base 0/3/9, Phase3B 1/2/9, Phase4 0.5B+LoRA-120 **5/4/3** (meta 0), 1.5B base 2/5/5 (tone intro, context essay, child-POV fail, false refusal risk)
- Targeted: 1.5B Jinx 41-token witty best yet, but context accepts lavender + silver-mug hallucination (binary violation)

**Human/regression/overfitting for micro-LoRA:** NOT APPLICABLE (no adapter). Phase4 remains backend.

**Preserved:** experiments 0001–0006, `checkpoints/phase4-miyuki-0.2/`, `datasets/miyuki-writing-0.2/` SHA `8cda31...`, `configs/training/phase4_lora.yaml` (assistant_only true).

**Recommendation:** See `docs/qwen2.5_1.5b_micro_lora.md` — RESULT D. Keep 1.5B as inference-only candidate, Phase4 0.5B adapter remains backend. Do NOT fight the machine.

**Files:** `preflight.json`, `resource_report.json`, `README.md` (this). No `outputs.jsonl`/`metadata.json` for micro-LoRA (no training).
