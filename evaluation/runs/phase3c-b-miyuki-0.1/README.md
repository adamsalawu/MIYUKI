# Phase 3C-B — User-Only Contract (miyuki-writing-0.1-phase3c B)

**Formulation:** `user` suffix `Requirements: Perform... (42 toks)` appended to original user, no system, assistant same. Derived SHA `e408e23e...`.

**Training:** Same as Phase3B (r16 alpha32, 3 epochs, lr2e-4, batch1 accum8, max_seq1024, seed42). `global_step12, epoch2.82, train_loss3.143, elapsed1211.9s, eval 3.29→2.82`, `checkpoints/phase3c-b-miyuki-0.1`, `experiments/0005`.

**Evaluation:** Same 12 prompts, `180 max`. Outputs `outputs.jsonl` (754 total avg63, max274, 0 errors, 314s) + `human_eval.yaml` (0 keep/1 edit/11 discard — also worse than Phase3B, 0 keep, many short <30, context Yay!).

**Comparison:** User-only similar to system (both 0/1/11), neither improves over Phase3B; formulation not lever.
