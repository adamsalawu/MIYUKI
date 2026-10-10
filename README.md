# Miyuki Model Lab

Foundation infrastructure for the **Miyuki** creative-writing model — the evolution of OffScript into a serious local-first story-authoring system.

> **Status: Phase 4C complete — 107 tests.** The validated baseline is the **Phase 4 adapter** (`checkpoints/phase4-miyuki-0.2/`, experiment `0006`): original-12 benchmark **5 Keep / 4 Edit / 3 Discard**. The newer Phase 4C adapter (`0007`+`0008`, 154-example dataset, 51 steps) scored **1/8/3** — it did **not** beat Phase 4 (overfit in epochs 2–3: train loss 0.25 vs eval plateau 2.86), though it learned some targeted behaviors. Newest ≠ best here; see [Experiment history](#experiment-history) and `docs/current_state.md`.

Miyuki's long-term goal is not "put an LLM inside a writing app." It is:

> *Build a writing model and story-intelligence system designed together.*

## Contents

- [What Miyuki is (and what this repo is)](#what-miyuki-is-and-what-this-repo-is)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Datasets](#datasets)
- [Experiment history](#experiment-history)
- [Evaluation](#evaluation)
- [Writing spec](#writing-spec)
- [Documentation index](#documentation-index)
- [Roadmap](#roadmap)
- [Known limitations](#known-limitations)
- [No Fake AI](#no-fake-ai)
- [License](#license)

## What Miyuki is (and what this repo is)

Three things, kept deliberately separate:

1. **Long-term vision — a creative-writing and story-intelligence system.** Miyuki keeps OffScript's core (projects, stories, characters, lore, AI-assisted writing) and aims to add chapters/scenes, characters and relationships, timeline/state tracking with persistent story memory, a structured-retrieval lorebook replacement, and optional internet research and character enrichment.
2. **This repository — the Python model-development lab.** Dataset tooling, LoRA training, evaluation, and local inference for the writing model. It owns training, evaluation, artifacts, and configs. It does *not* own persistent story memory — by design, the model is trained to *use* provided context (characters, constraints, canon), not to memorize every story fact in its weights.
3. **The future desktop writing application (not implemented here).** Projects, chapters, characters, lore, timeline, memory, retrieval UI, and the "Story Brain" that assembles context for the model live in the application design docs (`docs/architecture.md`, `docs/story_brain.md`, `docs/roadmap.md` phases 5–9) — documented, not built.

See `docs/architecture.md` and `docs/roadmap.md` for the full plan.

## Project structure

```text
MIYUKI/
├── configs/{model,training,inference,evaluation}/  # 15 YAMLs: base models, LoRA recipes, runtime/eval defaults
├── src/miyuki/              # Python package: cli, config, hardware, datasets, training,
│                            #   runtime, inference, evaluation, experiments, artifacts
├── datasets/miyuki-writing-{0.1,0.1-phase3c,0.2,0.3}/  # versioned, immutable datasets + manifests
├── data/manifests/          # dataset manifests (0.1–0.3, smoke-0.0.1); data/processed/smoke_demo.jsonl
├── training/ runtime/ inference/ model/  # notes (the pipeline itself lives in src/miyuki/)
├── evaluation/prompts/      # frozen suites: original 12, expanded 12, targeted 5
├── evaluation/runs/ evaluation/reports/  # measured outputs + human evals per phase
├── experiments/0001-0008/   # immutable experiment records (metrics.json + config.yaml)
├── tools/                   # phase runners, dataset builders, comparison/analysis scripts
├── tests/                   # 18 files, 107 tests (pytest)
├── docs/                    # architecture, roadmap, phase reports, strategy docs
├── checkpoints/ logs/ artifacts/  # local-only, gitignored (adapters, run logs, tokenizer experiments)
├── pyproject.toml requirements.txt
└── AGENTS.md                # operating contract for coding agents working in this repo
```

## Installation

**Requirements:** Python 3.10+, PyTorch + Transformers ecosystem. Tested on Python 3.12.

```powershell
# 1. Clone / enter
git clone https://github.com/adamsalawu/MIYUKI.git
cd MIYUKI

# 2. Create venv (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install (editable + dev tooling)
pip install -e ".[dev]"
# or minimal:
pip install -r requirements.txt
```

Hardware is auto-detected; the project works CPU-only as well as on iGPU / low-VRAM / stronger GPUs. Start with `miyuki hardware` (below) before any training or evaluation run.

## Quick start

### Hardware check

```powershell
python -m miyuki.cli hardware
# or, once installed:
miyuki hardware
```

Prints platform, CPU/RAM, torch build, CUDA/MPS availability, and the recommended device/dtype. Verified on this machine class: CPU-only (`torch 2.14.0+cpu`), recommended device `cpu`, dtype `float32` — training is slow here, so small models + LoRA + smoke configs are the norm.

### Baseline inference

```powershell
# List available model configs (tiny-random baseline + real Qwen configs)
python -m miyuki.cli models --list

# Tiny-random baseline (fastest, for regression)
python -m miyuki.cli infer --config configs/model/baseline.yaml --prompt "The lighthouse had been dark..." --max-new-tokens 64

# Real baseline: Qwen2-0.5B-Instruct (requires ~1 GB download; ~2.8 tok/s on CPU)
python -m miyuki.cli infer --config configs/model/small-instruct.yaml --prompt "The lighthouse had been dark for twenty years, until tonight. Continue." --max-new-tokens 180

# With saving
python -m miyuki.cli infer --config configs/model/small-instruct.yaml --prompt "Continue the scene:" --save --output outputs/sample.json
```

> CPU note: model downloads and generation are slow without a GPU. Treat tiny `--max-new-tokens` smoke checks as quick; full 12-prompt evaluations take minutes (see [Evaluation](#evaluation)).

### Evaluation

```powershell
# Phase 1 base (Qwen 0.5B, no adapter) — 12 prompts, ~9 min, 0 errors
python -m miyuki.cli evaluate --config configs/model/small-instruct.yaml
# Phase 3 LoRA (1 epoch, 4 steps) — 12 prompts, ~8.5 min, 0 errors
python -m miyuki.cli evaluate --config configs/training/phase3_lora.yaml  # loads adapter checkpoints/phase3-miyuki-0.1
# Phase 3B LoRA (3 epochs, 12 steps, lr 2e-4) — 12 prompts, ~5 min, 0 errors
python -m miyuki.cli evaluate --config configs/training/phase3b_lora.yaml  # loads adapter checkpoints/phase3b-miyuki-0.1
# Phase 3C-A/B (system vs user contract, 3 epochs, 12 steps) — 12 prompts each, ~6 min, 0 errors
python -m miyuki.cli evaluate --config configs/training/phase3c_a_lora.yaml  # system contract
python -m miyuki.cli evaluate --config configs/training/phase3c_b_lora.yaml  # user suffix
```

(Timings measured on the reference CPU-only machine.) Model-to-model comparisons use the same prompts and generation settings (`180 tokens / 0.8 / 0.95 / 1.1 / seed 42`) — see `tools/phase4_compare.py` and `tools/phase4c_compare.py` for the comparison pattern.

### Dataset validation

```powershell
miyuki dataset validate --path data/processed/smoke_demo.jsonl --format conversational
miyuki dataset stats --path data/processed/smoke_demo.jsonl
```

Also available: `miyuki dataset manifest` (SHA + token stats + splits + provenance for a new version). Schemas (`raw`, `prompt_completion`, `conversational`, `story`) are defined in `src/miyuki/datasets/schemas.py`.

### Smoke training (proves the full loop)

```powershell
# Tiny dataset → tokenizer → model → training → checkpoint → reload → generate
miyuki train --config configs/training/smoke.yaml

# The checkpoint is written to checkpoints/smoke-*/
# Verify it loads and generates:
miyuki infer --config checkpoints/smoke-*/config.yaml --prompt "The knight entered the tavern..."
```

Do not start real (non-smoke) training unless a milestone explicitly authorizes it — see `AGENTS.md` and [Roadmap](#roadmap).

### Tests

```powershell
pytest -v
# or
python -m pytest tests/ -v
```

107 tests collected across 18 files (verified). They cover configs, hardware detection, dataset validation/versioning, the training target mask, the eval runner, experiment records, and per-phase regression gates.

### Experiment tracking

```powershell
miyuki experiment list
miyuki experiment show --id 0006
```

## Configuration

All model identity lives in YAML, not scattered through code. Changing the base model means changing a file:

```yaml
# configs/model/baseline.yaml
model:
  name: "hf-internal-testing/tiny-random-gpt2"  # replaceable
  family: "gpt2"
  tokenizer: "hf-internal-testing/tiny-random-gpt2"
  context_length: 1024
runtime:
  device: "auto"
  dtype: "auto"
generation:
  temperature: 0.8
  top_p: 0.95
  max_new_tokens: 256
```

- `configs/model/` — base model + runtime + generation defaults (`baseline`, `small-instruct` Qwen2-0.5B, `qwen2.5-1.5b-instruct` for inference-only scale probes).
- `configs/training/` — SFT/LoRA/optimizer/checkpointing recipes per phase (plus `smoke.yaml` and the proposed-but-unauthorized `phase5_exp0009_lora_03_1ep.yaml`).
- `configs/inference/`, `configs/evaluation/` — runtime overrides and eval settings.

See `src/miyuki/config.py` and `docs/architecture.md`.

## Datasets

Versioned, immutable, manifest-pinned (SHA + token stats + splits + provenance in each `manifest.json` and `data/manifests/`):

| Dataset | Size | Purpose |
|---|---|---|
| `miyuki-writing-0.1` (+ `0.1-phase3c` A/B formulation variants) | 38 examples | First SFT; system-vs-user contract experiment |
| `miyuki-writing-0.2` | 120 examples | Assistant-only-loss training — produced the validated baseline (exp 0006) |
| `miyuki-writing-0.3` | 154 examples (0.2 + 34: voice, binary-context, viewpoint, object, dialogue-role probes) | Experimental; Phase 4C showed it overfits at 3 epochs |

Rules that earned their place the hard way: never silently modify a released version (new data = new version), never train on evaluation prompts, validate (`miyuki dataset validate`) before training. See `docs/dataset_strategy.md` and each dataset's `DATASET_SPEC.md` / `report.json` / `review.yaml`.

## Experiment history

Human grading is Keep / Edit / Discard on the frozen original-12 suite (plus expanded-12 and targeted-5 suites in later phases), always with identical generation settings. Full detail lives in `experiments/0001`–`0008/` and the `docs/phase*_report.md` files — this table is the short version:

| Exp | Phase | Setup | Train loss | Human (orig 12) | Verdict |
|---|---|---|---|---|---|
| 0001 | Smoke | pipeline proof | — | — | infra works |
| 0002 | 3 | 0.1, 38 ex, 1 ep, 4 steps | 3.76 | 0 / 2 / 10 | no improvement over base |
| 0003 | 3B | 0.1, 3 ep, 12 steps, lr 2e-4 | 3.20 | 1 / 2 / 9 | partial (kept emotional beat, fixed a Chinese leak) |
| 0004 / 0005 | 3C-A / 3C-B | system (64t) vs user (42t) contract | 3.23 / 3.14 | 0 / 1 / 11 both | worse than 3B; keep the 3B formulation |
| **0006** | **4** | **0.2, 120 ex, assistant-only loss, r16, 39 steps** | **2.62** | **5 / 4 / 3, meta 0** | **best validated — current baseline/production adapter** |
| — | scale probe | Qwen2.5-1.5B base, inference-only | — | 2 / 5 / 5 | slightly better, *not* promoted; 1.5B LoRA stopped at memory preflight (2.82 GB < 3.0 GB) — NO-GO, do not retry casually |
| 0007 + 0008 | 4C (latest) | 0.3, 154 ex, 51 steps via resume | 0.25 seg (eval plateau 2.86) | 1 / 8 / 3 | **did NOT beat Phase 4** — overfit in epochs 2–3; targeted gains only (voice, dialogue roles, object ring, Chinese fix), regressed scene/rewriting vividness and emotional beats |

The standing lesson, now enforced by tests and contract: **never select on training loss** — Phase 4C's collapsed loss accompanied behavioral regression. Prefer the best validated checkpoint, not the final one. See `docs/phase4c_report.md`, `docs/training_signal_audit.md`, and `docs/failure_taxonomy.md`.

## Evaluation

- **Frozen suites** (never modify old ones): `evaluation/prompts/eval_v0.1.jsonl` (original 12), `phase4_expanded.jsonl` (expanded 12), `targeted_scale.jsonl` (5 probes), plus the 9-dimension human template (`evaluation/human_template.yaml`) with a Keep/Edit/Discard verdict.
- **Runs are recorded**, not just asserted: `evaluation/runs/<phase>-*/` holds `outputs.jsonl`, `metadata.json`, and `human_eval.yaml` per run; the failure taxonomy (`evaluation/failure_taxonomy.json`) classifies every failure as fixed / improved / partial / persistent / regressed with evidence.
- **One good sample proves nothing** — compare full-suite human evals (see `docs/evaluation_strategy.md`).

## Writing spec

What "Miyuki writes like Miyuki" means is defined in `docs/miyuki_writing_spec.md` — the living spec guiding dataset creation and evaluation: narration, dialogue, character voice, pacing, sensory detail, and what Miyuki must *not* do (repetitive phrasing, generic AI filler, robotic dialogue, explaining instead of writing, etc.).

## Documentation index

Start with `docs/current_state.md` (the verified operational snapshot), then:

| Doc | What it is |
|---|---|
| `docs/roadmap.md` | Phase plan 0–9; phases 5+ are future/gated |
| `docs/architecture.md` | Lab-vs-application separation; runtime and context flow |
| `docs/miyuki_writing_spec.md` | Target behavior spec |
| `docs/model_strategy.md`, `docs/dataset_strategy.md`, `docs/evaluation_strategy.md` | Standing strategies |
| `docs/phase2_report.md`, `phase3_report.md`, `phase3b_report.md`, `phase3c_report.md`, `phase4_report.md`, `phase4b_dataset_expansion.md`, `phase4c_report.md` | Per-phase evidence |
| `docs/phase5_readiness.md` | Readiness audit for the proposed 1-epoch run (audit only — run not started) |
| `docs/training_signal_audit.md`, `docs/failure_taxonomy.md` | Why assistant-only loss; how failures are classified |
| `docs/model_scale_test.md`, `docs/qwen2.5_1.5b_micro_lora.md` | 1.5B inference results and the LoRA NO-GO |
| `docs/story_brain.md`, `docs/memory_architecture.md`, `docs/research_architecture.md`, `docs/character_pipeline.md` | Future-application designs (not implemented) |
| `docs/opencode_to_codex_migration.md` | Repo history and contract migration |
| `AGENTS.md` | Operating contract for coding agents (source of truth = the files) |

## Roadmap

| Phase | Name | Status |
|---|---|---|
| 0 | Infrastructure | complete (smoke run, 29 tests) |
| 1 | Baseline model — Qwen2-0.5B-Instruct measured | complete (12 prompts + generation sweep, 31 tests, `evaluation/runs/phase1_qwen2_0.5b/`) |
| 2 | Miyuki writing dataset — 38 human-authored, tokenizer study | complete (39 tests, `docs/phase2_report.md`) |
| 3 | First SFT — LoRA r16, 1 epoch, 4 steps, loss 3.76, no improvement | complete (47 tests, `evaluation/runs/phase3-miyuki-0.1/`) |
| 3B | More steps — 3 epochs, 12 steps, lr 2e-4, loss 3.20, partial (1 keep) | complete (55 tests, `docs/phase3b_report.md`) |
| 3C | Formulation — system vs user contract, both 0/1/11, no gain over 3B | complete (63 tests, `docs/phase3c_report.md`) |
| 4 | 0.2 dataset (120 ex) + assistant-only loss — exp 0006, 5/4/3, **validated baseline** | complete (74 tests, `docs/phase4_report.md`) |
| 4B | Dataset expansion to 0.3 (154 ex) | complete (101 tests, `docs/phase4b_dataset_expansion.md`) |
| 4C | Train on 0.3 (51 steps via resume) — 1/8/3, overfit, did not beat Phase 4 | complete (107 tests, `docs/phase4c_report.md`) |
| Scale | Qwen2.5-1.5B inference probe + micro-LoRA preflight NO-GO | complete, inference-only (`docs/model_scale_test.md`) |
| 5 readiness | Audit for a controlled 1-epoch run on 0.3 (exp 0009) | audit only — **run not started, not authorized** |
| 5–9 | Story Brain, character intelligence, research, local-runtime optimization, app integration | documented, not implemented |

Later phases are *documented, not implemented* in this repo. `docs/roadmap.md` has the detail.

## Known limitations

- **Overfitting threshold is real:** 154 examples × 3 epochs overfits (Phase 4C proof). Any future 0.3 run must be short and validated, not just trained.
- **Persistent weaknesses** (see `docs/current_state.md` + failure taxonomy): binary-context tracking, viewpoint drift, object retrieval, multi-character voice collapse, action continuity, dialogue subtext.
- **Phase 4C-only regressions:** surreal imagery drift, scene/rewriting vividness loss vs Phase 4, emotional beats slipping from edit to discard.
- **Hardware reality:** CPU-only on the reference machine (16 GB RAM), generation at a few tokens/sec, and long runs can die silently — use background processes with logs plus checkpoint resume. 1.5B LoRA training is blocked by memory; do not retry without a new authorized experiment.
- **Small suites are noisy:** 12-prompt human evals plus a 16-example held-out set — compare full suites, and never report training loss as quality.

## No Fake AI

This repo never presents canned responses or fake metrics as model output. Test fixtures are explicitly labelled (`is_fixture: true`). If something is not yet implemented, it raises `NotImplementedError` with an explanation.

## License

MIT — see `LICENSE`.

---

## Feedback

Issues: https://github.com/adamsalawu/MIYUKI/issues
