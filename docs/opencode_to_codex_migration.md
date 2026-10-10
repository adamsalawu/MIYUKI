# OpenCode → Codex Migration

> For future Codex sessions. The repository is the source of truth; this document
> summarizes what the previous OpenCode/Cline workflow built and how to continue
> in Codex. Verified against actual files on 2026-09-30.

## 1. Previous Agent Workflow

- Agent: OpenCode (Muse Spark model) via tool-call interface (`read`/`edit`/`write`/
  `shell`/`subagent`, one call per message, parallel calls as separate messages).
- Style: terse milestone prompts with HARD RULES / HARD STOP sections; each phase
  ended with a fixed-format final report (PASS/FAIL, metrics, artifacts).
- Tests run constantly (`python -m pytest tests/ -q`, split into batches due to a
  120s tool timeout). 107 tests at last count (phase4c_report.md).
- Long training/eval jobs ran detached in background with log redirection
  (`logs/phase4c_resume*.log`), monitored by waiter loops; sessions were
  interrupted by server restarts, so checkpoint/resume was load-bearing.
- Git: initialized, **zero commits** — everything untracked in working tree.
  Never commit or push (standing rule, preserved here).

## 2. Current Codex Workflow

- Use standard Codex file/shell tools; `AGENTS.md` (repo root) is the operating
  contract. Prefer dedicated file tools over shell for reads/edits.
- Same milestone discipline: inspect → plan → implement → test → evaluate →
  verify → report → stop. Same prohibitions: no auto-commit, no history rewrites,
  no weakening tests, no fake records.
- Checkpoints/resume remain the survival mechanism for long CPU training runs.

## 3. Project Architecture

See `docs/architecture.md`. Model Lab (`configs/`, `src/miyuki/`, `datasets/`,
`training/`, `inference/`, `evaluation/`, `runtime/`, `tools/`) is separate from
the future desktop app. Model identity lives only in YAML
(`configs/model/*.yaml`); Python imports `MiyukiConfig`, never hard-codes models.
`src/miyuki/`: `config.py`, `cli.py`, `hardware.py`, `datasets/` (schemas,
validate, versioning), `training/train.py` (TRL SFT + PEFT LoRA), `runtime/`
(`RuntimeBase` ABC; `TransformersRuntime` implemented; GGUF future),
`inference/pipeline.py`, `evaluation/runner.py`, `experiments/tracking.py`,
`artifacts/manager.py`. CLI: `python -m miyuki.cli {hardware,models,infer,evaluate,dataset,train,experiment}`.

## 4. Model Architecture

- Base: `Qwen/Qwen2-0.5B-Instruct` (494,032,768 params, Qwen2, 24 layers, vocab
  151646 tokenizer / 151936 model, 4096 ctx). Tokenizer: production Qwen, chat
  template True. **Never replaced, never resized.**
- Adapter: LoRA r16 / alpha32 / dropout 0.05 / bias none, targets
  `[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]`, trainable
  8,798,208 / 502,830,976 (1.75%). Same recipe since Phase 3.
- `Qwen/Qwen2.5-1.5B-Instruct` (1.54B) is **inference-only**; LoRA training
  correctly refused at preflight (2.82 < 3.0 GB). See `docs/model_scale_test.md`,
  `docs/qwen2.5_1.5b_micro_lora.md`.

## 5. Training Architecture

- `src/miyuki/training/train.py`: chat-template tokenization, 90/10 split
  (seed 42), `transformers.Trainer` + PEFT, reproducible seeds.
- **Assistant-only loss** (`training.assistant_only_loss: true`, Phase 4+):
  system/user tokens masked (`-100`), only assistant trainable — e.g. 93 masked /
  75 trainable on the reference example. Config default stays `false` to preserve
  experiments 0001–0005. See `docs/training_signal_audit.md`.
- Standard recipe: batch1 / accum8 / max_seq1024 / lr2e-4 / cosine / seed42 /
  CPU float32. Deterministic 34/4 (0.1), 108/12 (0.2), 138/16 (0.3) splits.
- Resume: narrow numpy allowlist (`_reconstruct`, `ndarray`, `dtype`,
  `UInt32DType`) in resume path only — fixes torch≥2.6 `weights_only` rejection
  of trusted local `rng_state.pth`. See `tests/test_checkpoint_resume_compat.py`.
- Memory/hardware guard mandatory before training (see `tools/micro_preflight.py`).

## 6. Evaluation Architecture

- Suites (never modify): original 12 (`eval_v0.1.jsonl`), expanded 12
  (`phase4_expanded.jsonl`), targeted 5 (`targeted_scale.jsonl`).
- Generation fixed: 180 / 0.8 / 0.95 / 1.1 / seed 42. Runner reuses one runtime.
- Human eval: 9 dims (prose, naturalness, character, dialogue, pacing, creativity,
  instruction, continuity, usefulness) + Keep/Edit/Discard
  (`evaluation/human_template.yaml`). Automated flags: meta phrases, hedging,
  Chinese leakage, premature EOS, length, tok/s.
- Comparison pattern: `tools/phase4c_compare.py`. Taxonomy:
  `evaluation/failure_taxonomy.json` + `docs/failure_taxonomy.md`.

## 7. Dataset Lineage

| Version | Examples | SHA (full) | Status |
|---|---|---|---|
| 0.1 | 38 human-authored | `9e1b97461f989482…200d5e` | frozen |
| 0.2 | 120 (38 + 82 targeted) | `8cda31a91a76b1d6…200d5e` | frozen, **training set of record** |
| 0.3 | 154 (120 + 34: A voice 8, B binary 8, C Jinx 6, D object 6, E viewpoint 6) | `f3f5f636ea6d412d…418759f635` | frozen |
| phase3c A/B | 38 each (derived, system/user contract) | b518… / e408… | frozen |

Each: `dataset.jsonl` + `manifest.json` + `report.json` + `review.yaml` +
`schema.json` + mirrored in `data/manifests/`. SHAs verified 2026-09-30.

## 8. Experiment Lineage

| ID | Base | Dataset | Key change | Result |
|---|---|---|---|---|
| 0001 | tiny-random | smoke 20 | pipeline proof | infra OK |
| 0002 | 0.5B | 0.1 (38) | first LoRA, 1ep, 4 steps, lr1e-4 | 0/2/10, loss 3.76 |
| 0003 | 0.5B | 0.1 | 3ep, 12 steps, lr2e-4 | 1/2/9, loss 3.20 |
| 0004/0005 | 0.5B | 0.1-A/B contracts | formulation A vs B | 0/1/11 both — formulation not the lever |
| 0006 | 0.5B | **0.2** (120) | assistant-only, 39 steps | **5/4/3, loss 2.62 — current baseline** |
| 0007 | 0.5B | 0.3 (154) | resume ckpt-30 → 51 steps | segment loss 0.938 |
| 0008 | 0.5B | 0.3 (154) | auto-recorded 2nd resume (ckpt-45) | segment loss 0.252; **final adapter weights** |

0007 = intended experiment; 0008 = unintended duplicate from a second resume
(same checkpoint dir, kept per no-delete rule). Final adapter
(`checkpoints/phase4c-miyuki-0.3/`, ckpts 25/30/40/45/50) is the 0008 segment's
weights; all Phase 4C evaluations used it (see `docs/phase4c_report.md`).

## 9. Current Best Validated Candidate

**Phase 4 adapter** — `checkpoints/phase4-miyuki-0.2/` (exp 0006): original 12 →
**5 Keep / 4 Edit / 3 Discard, meta 0, Chinese 0, EOS 0**. Phase 4C (0.3) scored
1/8/3 with overfitting (train 0.25 vs eval plateau 2.86) — did NOT improve the
benchmark, though it learned Jinx voice, dialogue roles, mug-ring, Chinese fix.
Phase 4 remains the backend until an experiment beats it.

## 10. Current Known Failures

Surviving (all failed P1–4C unless noted): exact/binary context (mug/lavender,
negative facts) — rank 1; viewpoint constraints (child/limited) — 2; object
retrieval in action (notebook never retrieved) — 3; multi-character voice
collapse — 4; action continuity — 5; dialogue subtext/interruptions — 6;
surreal imagery (P4C-only, overfit) — 7; rewriting vividness loss (P4C
regression) — 8. Eliminated: meta commentary, hedging essays, explaining-instead-
of-writing, summarization, moralizing, premature EOS, Chinese leakage (original).
Full taxonomy: `docs/failure_taxonomy.md`.

## 11. Current Hardware Constraints

i7-1065G7 (4c/8t), 16 GB RAM, MX330 2 GB idle, **CPU-only** (`torch 2.14.0+cpu`,
no CUDA), Python 3.12.10, transformers 4.41.0 / peft 0.11.1 / datasets 5.0.1 /
accelerate 1.15.0. 1.5B LoRA blocked (NO-GO). Eval speed degrades under memory
pressure (2.84 → 0.4–0.5 tok/s). Overfit threshold observed: 154 ex × 3 epochs.
Silent process deaths hit long detached runs 3× — checkpoint/resume is the
mitigation. Long jobs need background execution + logs + waiters.

## 12. Important Commands

```powershell
python -m miyuki.cli hardware
python -m miyuki.cli models --list
python -m miyuki.cli infer --config configs/model/small-instruct.yaml --prompt "..." --max-new-tokens 180
python -m miyuki.cli evaluate --config configs/training/phase4_lora.yaml
python -m miyuki.cli dataset validate --path datasets/miyuki-writing-0.3/dataset.jsonl --format conversational
python -m pytest tests/ -q   # split into batches; 120s tool timeout forces this
```

## 13. Important File Locations

See AGENTS.md §7 table. Phase reports: `docs/phase{2,3,3b,3c,4,4b_dataset_expansion,4c}_report.md`.
Strategy: `docs/{architecture,model_strategy,dataset_strategy,evaluation_strategy,roadmap}.md`.
Future systems (documented, not implemented): `docs/{story_brain,research_architecture,character_pipeline,memory_architecture}.md`.

## 14. Rules for Future Experiments

One variable at a time; new IDs/paths (next: 0009); same prompts + generation
settings; Keep/Edit/Discard + existing dims; never train on eval prompts; never
pick on train loss; stop at overfit plateau; prefer best validated checkpoint;
mechanism-created records only; evaluate original + expanded suites separately.

## 15. Rules for Preserving Historical Artifacts

Never modify/delete: `miyuki-writing-0.1/0.2/0.3`, experiments 0001–0008,
checkpoints, eval runs, manifests/SHAs, tokenizer config, Phase 4 config.
Never rewrite git history. Never commit/push without explicit instruction.

## 16. Exact Current Recommended Next Milestone

Controlled **1–2 epoch run on 0.3** (same Phase 4C recipe, stop before the
overfit plateau) vs Phase 4 — isolates whether 0.3's targeted examples help
before overfitting, or whether 0.2 remains the better set. Example:
`configs/training/phase4c_lora_03.yaml` → new output dir
`checkpoints/phase4d-miyuki-0.3` (next exp 0009, ~50–60 min). Evaluate same
original 12 + expanded 12. **Not authorized yet — do not start training.**
