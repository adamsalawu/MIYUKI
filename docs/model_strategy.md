# Model Strategy

## Baseline

- **First baseline:** `hf-internal-testing/tiny-random-gpt2` (~ 100K params).
  - Reason: offline-capable, CPU-friendly, proves full pipeline on 2 GB VRAM / CPU-only hardware. No download barrier for first test.
  - Not a writing model — its outputs will be incoherent. That's expected and honest.

- **First real instruct baseline:** `Qwen/Qwen2-0.5B-Instruct` (~0.5B, ~1 GB).
  - Reason: smallest capable instruct model that can actually write, fits 2 GB with quantization, strong multilingual/chat template.
  - Alternatives kept replaceable: `TinyLlama/TinyLlama-1.1B-Chat`, `HuggingFaceTB/SmolLM2-360M-Instruct`, etc.
  - All swaps are config-only (`configs/model/small-instruct.yaml`).

## Growth Path

1. **Phase 0–1:** Baseline infra + 0.5B instruct as reference for eval.
2. **Phase 2:** Curate `miyuki-writing-0.1` dataset; don't scale model yet.
3. **Phase 3:** First SFT via LoRA on 0.5B — measure on eval set before scaling.
4. **Scale only when measured:** Graduate to 1–3B if 0.5B + data shows signal; keep LoRA to stay VRAM-friendly.

## PEFT / Quantization

- **LoRA** is default (r=8–16, alpha 16–32) — full finetuning is not viable on 2 GB and not needed early.
- **Quantization:** 4bit/8bit via bitsandbytes when available; otherwise float16/float32. Config `runtime.quantization` controls it.
- **Gradient checkpointing** enabled for >0.5B runs.

## Runtime Backends

- **Transformers** — implemented, CPU/CUDA/MPS.
- **GGUF / llama.cpp** — future, for CPU quantized local deployment. Abstraction exists (`runtime/base.py`) but backend not yet implemented.

## Reproducibility

- Model identity is config, not code.
- Seeds set (random, numpy, torch, CUDA).
- Training saves `miyuki_training_config.yaml` + `metadata.json` + experiment record.
- Never train on evaluation prompts.

## Anti-goals

- No huge-model training before infra is proven.
- No hard-coded vendor lock-in.
- No fake metrics or canned "AI" responses.
