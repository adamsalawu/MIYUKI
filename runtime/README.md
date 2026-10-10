# runtime

Abstraction for local inference backends.

- `src/miyuki/runtime/base.py` — `RuntimeBase` ABC + `GenerationResult`
- `src/miyuki/runtime/transformers_runtime.py` — Transformers (implemented)
- Future: GGUF / llama.cpp behind the same interface (documented, not yet implemented)

The Miyuki app and model lab both depend on `RuntimeBase`, not on a specific vendor.
