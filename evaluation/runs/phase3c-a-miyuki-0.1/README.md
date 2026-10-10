# Phase 3C-A — System Contract (miyuki-writing-0.1-phase3c A)

**Formulation:** `system` role `Miyuki direct story-writing assistant... (64 toks)` + user as original + assistant same. Derived SHA `b518f30b...`, original preserved.

**Training:** Same as Phase3B (r16 alpha32, 3 epochs, lr2e-4, batch1 accum8, max_seq1024, seed42, CPU) — only formulation differs. `global_step 12, epoch2.82, train_loss 3.239, elapsed1220.9s, eval 3.31→2.78`, `checkpoints/phase3c-a-miyuki-0.1`, `experiments/0004`.

**Evaluation:** Same 12 prompts `eval_v0.1.jsonl`, `180 max, temp0.8 top_p0.95 rep1.1 seed42`. Outputs `outputs.jsonl` (898 total avg75, max 291, 0 errors, 369s) + `human_eval.yaml` (0 keep/1 edit/11 discard — worse than Phase3B 1/2/9, emotional keep lost). See `docs/phase3c_report.md`.

**Comparison:** System vs user (B) similar, both not improving over Phase3B; no material gain.
