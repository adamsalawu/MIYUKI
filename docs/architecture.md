# Miyuki Architecture

## Vision

> Build a writing model and story-intelligence system designed together.

The Miyuki Model Lab is the **model-development side** of Miyuki. It is deliberately separate from the future desktop application (OffScript successor). The app will own story state, memory, and retrieval; the model owns writing.

```
┌─────────────────────────────────────────┐
│         Miyuki Application (future)      │
│  projects / chapters / characters / lore │
│  timeline / memory / retrieval / UI      │
│              │                           │
│              ▼                           │
│     Story Brain (context assembly) ──────┼──► Research system
│              │                           │    Character pipeline
│              ▼                           │
│     Runtime abstraction                  │
│   ┌──────────────┬──────────────┐        │
│   │ Transformers │ GGUF/llama.cpp│ ...   │
│   └──────────────┴──────────────┘        │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│        Miyuki Model Lab (this repo)      │
│  configs / datasets / training / eval    │
│  inference pipeline / experiments        │
└─────────────────────────────────────────┘
```

## Separation of Concerns

| Layer | Owns | Does NOT own |
|-------|------|--------------|
| Model Lab | training, evaluation, artifacts, configs | persistent story memory |
| Story Brain (future app) | context assembly, retrieval, state tracking | model weights |
| Runtime | device/dtype/quantization, generation | story logic |
| Research / Character pipelines | extraction, enrichment, approval flow | silent canon rewrites |

## Model Abstraction

All model identity is in YAML (`configs/model/*.yaml`). Python imports `MiyukiConfig` — never hard-codes a model string.

```yaml
model: { name, family, tokenizer, context_length }
runtime: { device, dtype, quantization }
generation: { temperature, top_p, max_new_tokens, ... }
training: { optimizer, batch, LoRA, checkpointing, ... }
adapter: { type, r, alpha, dropout, target_modules }
```

Switching base models (e.g., `tiny-random-gpt2` → `Qwen2-0.5B-Instruct` → future) is a config change.

## Dataset System

Four first-class schemas (`src/miyuki/datasets/schemas.py`):

- `raw` — `{ text }` for continued pretraining
- `prompt_completion` — `{ prompt, completion }`
- `conversational` — `{ messages: [{role, content}] }` (TRL SFT-compatible)
- `story` — `{ instruction, story_context, characters, scene, world_information, recent_prose, response }` + `messages` alias

Validation (`validate.py`) checks: missing fields, empty/ tiny examples, malformed conversations (consecutive roles, empty content), duplicate hashes, impossible metadata, train/eval contamination via hash intersection.

Versioning (`versioning.py`): `DatasetManifest` records name/version/source/license/splits/token stats/file hashes/preprocessing — never silently overwrite.

Provenance: `source` field tracks `user-created | synthetic | public-domain | licensed | permitted` — never scrape copyrighted works without rights.

## Runtime Abstraction

```
RuntimeBase (ABC)
  ├── TransformersRuntime  (implemented, CPU/CUDA/MPS, HF transformers)
  └── GGURuntime           (future, llama.cpp - documented not implemented)
      └── (future backends)
```

`TransformersRuntime` handles device/dtype auto-detection, tokenizer loading, quantization stubs (4bit/8bit via bitsandbytes when available), generation with seed reproducibility, and adapter (PEFT) loading. `runtime/transformers_runtime.py:1`.

The model is *stateless* — the future app assembles structured context and passes it as a prompt; the model does not persist story memory.

## Evaluation

- **Prompts** live in `evaluation/prompts/*.jsonl` — small, hand-crafted, versioned, never in training.
- **Runner** (`evaluation/runner.py`) loads a `MiyukiConfig`, iterates prompts, saves `evaluation/runs/<date>_<model>/outputs.jsonl` + `metadata.json`, plus `evaluation/reports/<date>_<model>.json`.
- **Human eval** uses `evaluation/human_template.yaml` — per-output 1–5 ratings for 9 dimensions (prose, naturalness, character consistency, dialogue, pacing, creativity, instruction adherence, continuity, overall usefulness) plus raw notes and keep/edit/discard.

Comparison is `BASE vs MIYUKI-0.1 vs MIYUKI-0.2` on identical prompts.

## Training

- `src/miyuki/training/train.py` — SFT via `transformers.Trainer` + `peft.LoraConfig`, tokenizes via chat template or fallback concatenation, splits 90/10 train/eval, supports resume, gradient checkpointing, mixed precision flags, reproducible seeds.
- `configs/training/smoke.yaml` — deliberately tiny (20 synthetic examples, 1 epoch, LoRA r=4) proving `DATASET → TOKENIZER → MODEL → TRAINING → CHECKPOINT → LOAD → INFERENCE`.
- Larger configs (`example.yaml`) build on the same code path.

## Experiment Tracking

`src/miyuki/experiments/tracking.py` — every run records `id / model / dataset / version / training config / hardware / duration / loss / eval results / artifact location` under `experiments/<id>/`. No "I think that version was better".

## Model Artifacts

`src/miyuki/artifacts/manager.py`:

```
models/miyuki-0.1/
  config/  tokenizer/  weights/  adapters/
  metadata.json  README.md

checkpoints/smoke/
  config.json  adapter_config.json  pytorch_model.bin / adapter_model.safetensors
  metadata.json  train_metrics.json  dataset_info.json  miyuki_training_config.yaml
```

`metadata.json` records base model, dataset version, training config, tokenizer, context length, quantization, creation date, evaluation results.

## Future Systems (documented, not implemented in Phase 0)

- **Story Brain** — assembles `CURRENT_SCENE / CHARACTERS / STATES / RELATIONSHIPS / WORLD FACTS / TIMELINE / RELEVANT MEMORIES / RESEARCH / AUTHOR INTENT / RECENT PROSE` into a prompt. Model does not store canon.
- **Research** — `WEB → retrieve → extract → identify relevant → preserve sources → store structured → provide context`. Model never treats web text as truth.
- **Character intelligence** — `raw → extraction → normalization → conflict detection → missing-info detection → research → proposed improvements → user approval → structured record`. User remains in control.
- **Memory categories** — permanent canon, character/world/scene memory, temporary context, author notes, research, timeline events → future retrieval engine.
- **GGUF/llama.cpp** — future runtime backend for efficient local CPU/quantized inference.

## Hardware Strategy

`src/miyuki/hardware.py` detects CPU/RAM/GPU/VRAM/CUDA/torch version and recommends `device` (`cuda`/`mps`/`cpu`) and `dtype`. Miyuki must run on CPU-only, iGPU, low-VRAM (2 GB MX330), and stronger machines. Training defaults to `per_device_train_batch_size=1` + gradient accumulation.

## Reproducibility

- Configs versioned in git
- Manifests version datasets
- Seeds set (`random`, `numpy`, `torch`)
- `miyuki_training_config.yaml` snapshot saved per checkpoint
- Experiment metadata links model/dataset/hardware/metrics
