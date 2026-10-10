# Special Tokens Investigation — Phase 1 Warning

**Warning observed (Phase 1 and Phase 2):**
```
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
```

**Where:** Appears on `AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")` and on `AutoModelForCausalLM.from_pretrained` (transformers 4.41.0, huggingface_hub). Not caused by MIYUKI code adding tokens.

**Investigation (2026-09-24):**
```python
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
len(tok)  # 151646
tok.pad_token, tok.pad_token_id  # <|endoftext|> 151643
tok.eos_token, tok.eos_token_id  # <|im_end|> 151645
tok.additional_special_tokens  # ['<|im_start|>', '<|im_end|>']
tok.all_special_tokens  # ['<|im_end|>', '<|endoftext|>', '<|im_start|>']
tok.all_special_ids  # [151645, 151643, 151644]

cfg = AutoConfig.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
cfg.vocab_size  # 151936
model.get_input_embeddings().weight.shape  # torch.Size([151936, 896])
```

- Tokenizer vocab (151646) includes the 3 special tokens (`<|im_start|>` 151644, `<|im_end|>` 151645, `<|endoftext|>` 151643) within range.
- Model embeddings are **151936**, i.e., 290 ids larger than tokenizer (151646..151935 reserved for future tokens). This is intentional by Qwen2 design.
- The warning is emitted by `transformers.tokenization_utils_base` when `added_tokens_decoder` contains special tokens, regardless of whether they are already trained. Here they are **already trained** (part of the released 0.5B instruct weights), not newly added by MIYUKI.
- Our runtime does `if tok.pad_token is None: tok.pad_token = tok.eos_token` — for Qwen `pad_token` is not None, so no new token is added. We never call `add_tokens` or `resize_token_embeddings` on the production model.

**Is it necessary?** Yes, in this context it is expected and harmless. The 3 special tokens are already in both tokenizer and model embeddings (within 151646), and the 290 extra embedding rows are unused reserved capacity. No embedding rows are untrained due to MIYUKI actions.

**Fix:** No code change to suppress globally (would hide real future warnings). We document it here. Generic handling: production tokenizer/model remain unmodified; experimental tokenizer is **not** connected to production model (per spec, would require `model.resize_token_embeddings(len(new_tok))` + training). Phase 2 does not do that, does not save altered production model, does not modify `Qwen/Qwen2-0.5B-Instruct` weights.

**If we were to add Miyuki invented tokens (e.g., make `lumenweave` a single token):**
1. Call `tok.add_tokens(["lumenweave"], special_tokens=False)` → `len(tok)` 151647
2. Call `model.resize_token_embeddings(len(tok))` → embedding rows 151937 (new row randomly initialized, must be trained)
3. That would modify the production model — **disallowed in Phase 2** (no Trainer.train). Therefore experimental tokenizers stay as artifacts, not live.

**Reproduction:**
```powershell
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B-Instruct')"  # shows warning
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; tok=AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B-Instruct'); m=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-0.5B-Instruct'); print(m.get_input_embeddings().weight.shape[0], len(tok))"
```
