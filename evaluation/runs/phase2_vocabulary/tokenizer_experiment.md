# Tokenizer Experiment — miyuki-experimental

- **Base:** Qwen/Qwen2-0.5B-Instruct vocab 151646
- **Corpus:** 38 examples, 21738 chars on full corpus (actually 21738 chars for unique texts)
- **Vocab sizes tested:** [8192, 16000, 32000]
- **Trained:** 3/3
- **Timestamp:** 2026-09-24T21:13:29.497879+00:00

## Results

- **vocab 8192:** base 5071 toks vs new 4776 (ratio 0.942, compression +5.82% — negative means more tokens than base)
- **vocab 16000:** base 5071 toks vs new 4776 (ratio 0.942, compression +5.82% — negative means more tokens than base)
- **vocab 32000:** base 5071 toks vs new 4776 (ratio 0.942, compression +5.82% — negative means more tokens than base)

## Interpretation
- Compression positive means experimental tokenizer uses fewer tokens (more efficient) than Qwen base on this English fiction corpus.
- All experimental vocabs are **smaller** than Qwen's 151646 (multilingual). If smaller English-focused vocab is more efficient, it suggests Qwen's large multilingual vocab is not inefficient for English prose.
- These tokenizers were trained on only 38 examples (~5k tokens, expanded 10x) — a tiny sample; results are indicative, not definitive. A larger Miyuki corpus would be needed for stable BPE.
- **Compatibility:** Experimental tokenizers are NOT connected to production Qwen model (embedding matrix mismatch would require `resize_token_embeddings` + retraining). Phase 2 does not do this.
