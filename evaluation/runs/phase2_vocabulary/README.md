# Phase 2 — Vocabulary / Tokenizer Study

**Tokenizer:** `Qwen/Qwen2-0.5B-Instruct` (vocab 151646, model vocab 151936, `chat_template True`, pad `<|endoftext|>` 151643, eos `<|im_end|>` 151645, special `["<|im_start|>","<|im_end|>"]`)  
**Corpus:** `miyuki-writing-0.1` — 38 human-authored examples, 76 messages, 21738 chars, 3719 words, 5071 tokens (Qwen base), 133.5 tok/ex (min 41, max 212).  
**Hardware:** Intel i7-1065G7, 16 GB RAM, CPU `torch 2.14.0+cpu` (same as Phase 1).  
**Experiment:** 3 vocab sizes requested [8192,16000,32000] → all trained to **actual 2572** due to small corpus (76 messages) limiting BPE merges; saved to `artifacts/tokenizers/miyuki-experimental/miyuki-vocab-{8192,16000,32000}` (each identical because corpus-limited). Training via `AutoTokenizer.train_new_from_iterator(iterator, vocab_size=...)`, 0.8–0.9 s each, deterministic on corpus.

## Reports

- `tokenizer_report.json` / `tokenizer_report.md` — base Qwen tokenization on Miyuki corpus (tokens/char, per-word, fragmented words, Miyuki term frequencies).
- `tokenizer_experiment.json` / `tokenizer_experiment.md` — training runs, vocab sizes, compression.
- `comparisons.json` — per-text tokenization base vs experimental (e.g., `lumenweave` 4→1, `Aureline` 3→1, `Mara` 2→1, `scryglass` 3→4).
- `metadata.json` — reproducibility (python, transformers, timestamps).
- This `README.md`.

## Key Findings

**Base efficiency:** 0.233 tok/char, 1.36 tok/word, 4.29 chars/tok — efficient for English prose; per-example 133 tok fits 4096 context easily.

**Fragmentation:** Most character names 2 toks (`M→ara`, `J→inx`), invented terms 3–4 (`lumenweave ['l','umen','we','ave']`, `voidtide ['void','t','ide']`). Frequent invented terms (freq 5–8) fragmented consistently. Rare words like `melancholic` also 3 toks. No catastrophic fragmentation (no word >4 toks).

**Miyuki term analysis (see `tokenizer_report.json`):** Common tokens (`the`, `she`, `and`) are 1 tok (good); single names and invented terms are 2–4 toks (moderate). `lumenweave`/`Aureline` could be single tokens if added, saving ~3 toks per occurrence — but at 6 occurrences each, total saving would be ~18 tokens corpus-wide (0.35% of 5071). Not large.

**Experimental vs base (corpus-level):** Base 5071 toks vs experimental 4776 toks — **5.8% compression** (experimental uses fewer tokens). Sample: `lumenweave` 4→1 (−75%), `Aureline` 3→1 (−66%), `Mara` 2→1 (−50%), but `scryglass` 3→4 (+33%). Overall modest gain because experimental vocab (2572) was learned specifically on this tiny corpus's byte pairs; Qwen's 151k multilingual vocab already covers English well.

**Vocab size mismatch:** Requested 8192/16000/32000 all collapsed to 2572 — corpus of 38 examples (~5k toks) cannot support larger vocab; BPE merges limited by unique character bigrams. Demonstrates that **Miyuki corpus is too small to justify a large custom vocab**; experimental tokenizer is corpus-overfitted.

**Special tokens:** Investigated warning `Special tokens have been added...` — model vocab 151936 vs tokenizer 151646 = 290 reserved ids. Qwen's 3 special tokens (`<|im_start|>`,`<|im_end|>`,`<|endoftext|>`) are within 151646 and already have embeddings (model embeddings shape `[151936,896]`). Warning is generic transformers notice for `added_tokens_decoder`; here it is **expected, not harmful**. Fix: do not call `resize_token_embeddings` on production model (per spec); document. Production model not modified; experimental tokenizer NOT connected to production model (embedding mismatch would require resizing + retraining).

**Practical implication:** Expanding tokenizer with Miyuki invented words would save ~5% tokens but requires embedding resize + retraining to align. With only 38 examples, benefit is tiny and not worth risk. Recommendation (see final report): **Keep Qwen tokenizer for Phase 3 LoRA; prioritize dataset quality over vocab adaptation**.

## Reproduction

```powershell
python tools/vocab_study.py
python tools/tokenizer_experiment.py
Get-Content evaluation/runs/phase2_vocabulary/tokenizer_report.json
Get-Content evaluation/runs/phase2_vocabulary/tokenizer_experiment.json
Get-Content evaluation/runs/phase2_vocabulary/comparisons.json
```

All runs deterministic on corpus; second run produces same token counts for base (experimental training is statistical but on this tiny corpus gives same 2572 vocab each time — verified 3×).

## Artifacts

- `artifacts/tokenizers/miyuki-experimental/miyuki-vocab-{8192,16000,32000}/tokenizer.json` + `vocab.json`/`merges.txt` (actual vocab 2572 each, identical).
- Not linked to production model; Phase 2 does not call `resize_token_embeddings`.
