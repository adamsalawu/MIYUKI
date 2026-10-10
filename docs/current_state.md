# MIYUKI Current State (operational snapshot — 2026-09-30, verified)

> Reflects actual repository state. Update after each milestone.

- **Current phase:** Phase 4C complete. Next proposed (NOT started): controlled 1–2 epoch run on 0.3.
- **Current validated baseline:** **Phase 4 adapter** — `checkpoints/phase4-miyuki-0.2/` (exp 0006, Qwen2-0.5B + LoRA-120, assistant-only, 39 steps, loss 2.62). Original 12: **5 Keep / 4 Edit / 3 Discard, meta 0**.
- **Current best adapter/checkpoint:** `checkpoints/phase4-miyuki-0.2/adapter_model.safetensors` (35.2 MB). Phase 4C adapter (`checkpoints/phase4c-miyuki-0.3/`) exists but did NOT beat the benchmark (1/8/3, overfit).
- **Current dataset:** `miyuki-writing-0.2` (120 ex, SHA `8cda31a9…200d5e`). 0.3 exists (154 ex, SHA `f3f5f636…418759f635`) but is experimental.
- **Current tokenizer:** `Qwen/Qwen2-0.5B-Instruct` (151646, chat True). Unchanged since Phase 1.
- **Current training recipe:** LoRA r16/alpha32/dropout0.05/bias-none, qwen2-7 targets, batch1/accum8/max1024/lr2e-4/seed42/CPU float32, `assistant_only_loss: true`. See `configs/training/phase4_lora.yaml`.
- **Current evaluation status:** Original 12 + expanded 12 + targeted 5 suites frozen. Latest runs: Phase 4C both suites (2026-09-29). Taxonomy: `evaluation/failure_taxonomy.json`.
- **Current known regressions:** Surreal imagery (P4C-only, overfit); scene/rewriting vividness loss vs Phase 4; emotional edit→discard. Persistent: binary context, viewpoint, object retrieval, multi-char voice collapse, action continuity.
- **Current hardware constraints:** CPU-only, 16 GB RAM, 1.5B LoRA blocked (NO-GO), eval 0.4–2.8 tok/s depending on memory pressure, silent process deaths on long runs (use background + logs + resume).
- **Current next recommended experiment:** Controlled 1–2 epoch run on 0.3 (`checkpoints/phase4d-miyuki-0.3`, exp 0009). **Not authorized — do not start.**
- **DO NOT DO YET:** train anything · Story Brain · Research · Memory · GUI · 1.5B training · new datasets · Phase 5 training · commits/pushes.
