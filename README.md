# Miyuki Model Lab

Foundation infrastructure for the **Miyuki** creative-writing model — the evolution of OffScript into a serious local-first story authoring system.

> **Status: Phase 4C complete — 0.3 dataset (154 ex) trained to 51 steps via resume (compat fix for torch≥2.6 RNG state); original 12: P4 5/4/3 vs P4C 1/8/3 — dataset 0.3 did NOT improve over 0.2 (overfit in epochs 2–3: train 0.25 vs eval plateau 2.86). Some targeted behaviors learned (Jinx voice, dialogue roles, mug-ring, Chinese fix). See `docs/phase4c_report.md`, 107 tests.**

Miyuki's long-term goal is not "put an LLM inside a writing app." It is:

> *Build a writing model and story-intelligence system designed together.*

---

## What Miyuki Is

Miyuki keeps OffScript's core (projects, stories, characters, lore, AI-assisted writing) and adds:

- story authoring with chapters / scenes
- characters, relationships, worldbuilding
- timeline / state tracking and persistent story memory
- a lorebook replacement with structured retrieval
- optional internet research & character enrichment
- contextual retrieval and AI-assisted writing
- eventually a custom Miyuki writing model trained for this use case

The **Miyuki Model Lab** (this repo) is the *model-development* side — dataset tooling, training, evaluation, and local inference. It is kept separate from the future desktop application.

See `docs/architecture.md` and `docs/roadmap.md` for the full plan.

---

## Project Structure

```
MIYUKI/
├── configs/{model,training,inference,evaluation}/
├── data/{raw,processed,splits,manifests}/
├── datasets/{schemas,tools}/
├── src/miyuki/              # Python package (miyuki.*)
├── model/  training/  inference/  evaluation/  runtime/  tools/
├── docs/
├── tests/
├── models/                  # local model artifacts (gitignored)
├── checkpoints/             # training checkpoints (gitignored)
├── outputs/  evaluation/runs/  experiments/
└── pyproject.toml
```

---

## Installation

**Requirements:** Python 3.10+, PyTorch, Transformers ecosystem. Tested on Python 3.12.

```powershell
# 1. Clone / enter
cd C:\Aether\RisingStar\MIYUKI

# 2. Create venv (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install (editable + dev tooling)
pip install -e ".[dev]"
# or minimal:
pip install -r requirements.txt
```

Hardware is auto-detected; the project works on CPU-only, iGPU, low-VRAM (2 GB) and stronger GPUs. See `miyuki hardware` below.

---

## Quick Start

### Hardware check

```powershell
python -m miyuki.hardware info
# or
miyuki hardware
```

### Baseline inference

```powershell
# List available model configs (baseline tiny-random + real Qwen 0.5B)
python -m miyuki.cli models --list

# Tiny-random baseline (fastest, for regression)
python -m miyuki.cli infer --config configs/model/baseline.yaml --prompt "The lighthouse had been dark..." --max-new-tokens 64

# Real baseline: Qwen2-0.5B-Instruct (requires ~1 GB download, CPU ~2.8 tok/s)
python -m miyuki.cli infer --config configs/model/small-instruct.yaml --prompt "The lighthouse had been dark for twenty years, until tonight. Continue." --max-new-tokens 180

# With saving
python -m miyuki.cli infer --config configs/model/small-instruct.yaml --prompt "Continue the scene:" --save --output outputs/sample.json
```

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
# Outputs: phase1 1538 avg128 vs phase3 1698 avg141 vs phase3b 996 avg83 vs phase3c-a 898 avg75 vs phase3c-b 754 avg63
# Human: phase1 0/3/9 vs phase3 0/2/10 vs phase3b 1/2/9 vs phase3c-a 0/1/11 vs phase3c-b 0/1/11 — neither formulation improves over Phase3B
```

### Dataset validation

```powershell
miyuki dataset validate --path data/processed/example.jsonl --format conversational
miyuki dataset stats --path data/processed/example.jsonl
```

### Smoke training (proves the full loop)

```powershell
# Tiny dataset → tokenizer → model → training → checkpoint → reload → generate
miyuki train --config configs/training/smoke.yaml

# The checkpoint is written to checkpoints/smoke-*/
# Verify it loads and generates:
miyuki infer --config checkpoints/smoke-*/config.yaml --prompt "The knight entered the tavern..."
```

### Tests

```powershell
pytest -v
# or
python -m pytest tests/ -v
```

---

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

- `configs/model/` — model + runtime + generation defaults
- `configs/training/` — SFT / LoRA / optimizer / checkpointing
- `configs/inference/` — inference overrides
- `configs/evaluation/` — evaluation settings

See `src/miyuki/config.py` and `docs/architecture.md`.

---

## Writing Spec

What "Miyuki writes like Miyuki" means is defined in `docs/miyuki_writing_spec.md` — the living spec that will guide dataset creation and evaluation. It covers narration, dialogue, character voice, pacing, sensory detail, and what Miyuki must *not* do (repetitive phrasing, generic AI filler, robotic dialogue, etc.).

---

## Roadmap

| Phase | Name | Status |
|-------|------|--------|
| 0 | Infrastructure | ✅ complete (smoke run, 29 tests) |
| 1 | Baseline model — Qwen2-0.5B-Instruct measured | ✅ complete (12 prompts, sweep, 31 tests, see evaluation/runs/phase1_qwen2_0.5b/README.md) |
| 2 | Miyuki writing dataset — 38 human-authored, vocab study 151646 | ✅ complete (39 tests, see docs/phase2_report.md) |
| 3 | First SFT — LoRA r16 1 epoch 4 steps loss3.76 no improvement | ✅ complete (47 tests, see evaluation/runs/phase3-miyuki-0.1/README.md) |
| 3B | More steps — LoRA r16 3 epochs 12 steps lr2e-4 loss3.20 partial (1 keep) | ✅ complete (55 tests, see evaluation/runs/phase3b-miyuki-0.1/README.md + docs/phase3b_report.md) |
| 3C | Formulation — system vs user contract, 38 ex, 12 steps, A 3.23 vs B 3.14, both 0/1/11 no improvement over Phase3B | ✅ complete (63 tests, see evaluation/runs/phase3c-a/b-miyuki-0.1/README.md + docs/phase3c_report.md) |
| 4 | Evaluation & iteration | planned |
| 5 | Story Brain | documented |
| 6 | Character intelligence | documented |
| 7 | Research system | documented |
| 8 | Local runtime optimization (GGUF/llama.cpp) | documented |
| 9 | Miyuki application integration | planned |

See `docs/roadmap.md` for detail. Later phases are *documented, not implemented* in this milestone.

---

## No Fake AI

This repo never presents canned responses or fake metrics as model output. Test fixtures are explicitly labelled (`is_fixture: true`). If something is not yet implemented, it raises `NotImplementedError` with an explanation.

---

## License

MIT — see `LICENSE`.

---

## Feedback

Issues: https://github.com/anomalyco/opencode
