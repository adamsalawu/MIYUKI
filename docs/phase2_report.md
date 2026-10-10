# Miyuki Model Lab — Phase 2 Report: Writing Corpus + Vocabulary Study

**Project:** `C:\Aether\RisingStar\MIYUKI`  
**Phase:** 2 — Writing Dataset + Vocabulary/Tokenizer Study (no model training)  
**Model reference:** `Qwen/Qwen2-0.5B-Instruct` (494,032,768 params, Qwen2, vocab 151646, model vocab 151936, context 4096, CPU float32)  
**Date:** 2026-09-24  
**Hardware:** Intel i7-1065G7, 16 GB RAM, NVIDIA MX330 2 GB (torch CPU only, `recommended_device cpu`, `dtype float32`, `cuda_available False`)  
**Python:** 3.12.10, `transformers 4.41.0`, `tokenizers 0.19.1`, `torch 2.14.0+cpu`, `datasets 5.0.1`, `peft 0.11.1`, `accelerate 1.15.0`  

---

## 1. Dataset — `miyuki-writing-0.1`

**Count:** 38 exemplars (exceeds required 30–50 floor while staying quality-focused; 38 = 76 messages). **Manifest:** `datasets/miyuki-writing-0.1/manifest.json` + `data/manifests/miyuki-writing-0.1.json` (SHA256 `9e1b97461f989482...`).

**Format:** Conversational (`messages: [user, assistant]`) with `metadata {id, instruction, context, category[], skills[], source, notes, word_count}` — uses existing MIYUKI infrastructure (`src/miyuki/datasets/schemas.py:ConversationalExample` + metadata), validated via `miyuki dataset validate --format conversational`.

**Categories (38):** `scene 29, quiet 10, anti_meta 10, description 14, pacing 14, character_consistency 14, dialogue 10, subtext 7, character_voice 7, rewriting 5, context_retention 5, transitional 1, atmospheric 3, emotion 2, action 2, etc.` (full counters in `manifest.json`). Each example has 1–4 tags, enabling future ablations (e.g., which categories actually help).

**Skills:** `instruction_following 31, prose_flow 25, pacing 8, naturalness 7, sensory_detail 5, character_consistency 3, continuity 3, subtext 1` — maps to `docs/miyuki_writing_spec.md` dimensions.

**Token/char counts:** 21738 chars, 3719 words, 5072 tokens (Qwen tokenizer, `add_special_tokens False`), avg 573 chars / 133.5 tokens per example (min 185/41, max 885/212), avg word 62.6 (16–119). Fits 4096 context easily. `tokens/char 0.233`, `tokens/word 1.36`, `chars/token 4.29`.

**Source:** `human-authored` (human-authored, AI-assisted drafting where needed, human-edited and reviewed) — no Qwen-generated weak output, no scraped copyrighted books. License `CC-BY-4.0`. Recurring invented terms (Aureline, Kestrel, lumenweave, scryglass, voidtide, genkan, Emberhearth) are original Miyuki worldbuilding for tokenizer study.

**Validation:** `38/38 valid, 0 invalid, 0 warnings, 0 duplicates, 0 empty, 0 tiny` (conversational), extended checks: required metadata fields present, categories from controlled vocab, `word_count` matches actual (+/-2), no duplicate IDs, tokenization compatibility OK (all 38 encode without error), `miyuki dataset validate` passes, `tools/dataset/manifest_and_validate.py` passes.

**Human review:** `datasets/miyuki-writing-0.1/review.yaml` — every exemplar rated 1–5 for clarity, writing quality, target behavior, category correctness, instruction/character/context consistency, `demonstrates_skill` bool, `keep/edit/discard`. All 38 **keep** (5/5 clarity & quality). Reviewer: Miyuki Lab, same dimensions as Phase 1 human template but for dataset quality.

**Before/after:** Anti-pattern examples demonstrate *explain vs write* without storing large negative targets — e.g., `miyuki-0.1-016` instruction "Do not say 'Here is a scene'" → target is pure scene, `miyuki-0.1-017` notes "before would be essay 'The tone is described as...'". Undesirable behavior described in `notes`, not as training target.

**Spec:** `datasets/miyuki-writing-0.1/DATASET_SPEC.md` + `schema.json` + `README.md` + `failure_mapping.yaml`.

---

## 2. Vocabulary

**Original Qwen vocab:** 151646 (tokenizer) vs model vocab 151936 (290 reserved). `chat_template True`, pad `<|endoftext|>` 151643, eos `<|im_end|>` 151645, specials `["<|im_start|>","<|im_end|>"]`.

**Corpus stats:** as above (21738 chars, 5071 tokens). Per-example 133 tok avg.

**Recurring Miyuki-specific vocab (freq ≥1, fragmentation):**
- `Aureline` 6× → 3 toks `['A','ure','line']` (Qwen) / 1 tok experimental
- `lumenweave` 6× → 4 toks `['l','umen','we','ave']` / 1 tok experimental
- `voidtide` 7× → 3 toks `['void','t','ide']` / 3 toks experimental (differently split)
- `scryglass` 5× → 3 toks `['sc','ry','glass']` / 4 toks experimental
- `Kestrel` 1× → 3 toks / 3, `Emberhearth` 1× → 4, `genkan` 1× → 2
- Names: `Mara` 5×→2 `['M','ara']` / 1 experimental, `Jinx` 8×→2 /2, `Elena` 5×→2, `Haru` 12×→2, `Mika` 7×→2, `Toma` 7×→2, `Lira` 9×→2, `Ken` 1×→1 (only name that is single token)
- Most fragmented frequent words (freq≥2): `lumenweave 4`, `Sensory/sensory 3`, `voidtide 3`, `Aureline 3`, `melancholic 3`, `sweater 3`

**Categories:**
- *Common existing tokens (1 tok):* `the, and, she, you, " , ... , --` — efficient, English common words well-covered.
- *Potential useful added tokens:* Frequent invented terms if single-token would save 2–3 toks each (lumenweave 6× saving 18 toks = 0.35% corpus). Modest.
- *Potentially unnecessary:* Adding all frequent adjectives (e.g., `melancholic` 2×) would bloat vocab for rare gain.
- *Rare:* `Emberhearth` 1×, `Kestrel` 1× — not worth adding.
- *Names/proper nouns:* All 2 toks except Ken — could be 1 tok but names are diverse, adding each would be N new tokens for small corpus.
- *Formatting/special:* `—` (em dash), `…`, `"`, `"` all 1 tok — punctuation efficient; contractions `didn't` → `['didn',"'t"]` 2 toks (expected).

**Notable tokenization problems:** No catastrophic fragmentation (>4 toks). Worst is `lumenweave` 4. Dialogue tokenization 0.23–0.275 tok/char (46–55 toks per 200 chars). Overall Qwen English efficiency is good; domain mismatch is **small**.

---

## 3. Tokenizer Experiment

**Experimental branch:** `artifacts/tokenizers/miyuki-experimental/` — NOT production. Base `Qwen/Qwen2-0.5B-Instruct` via `AutoTokenizer.train_new_from_iterator(iterator, vocab_size=VS)` where iterator is Miyuki corpus texts ×10 (760 texts) to give BPE enough frequency. Recorded in `evaluation/runs/phase2_vocabulary/tokenizer_experiment.{json,md}`.

**Vocab sizes requested:** 8192 (small), 16000 (medium), 32000 (larger) — all smaller than Qwen 151k, sensible for English fiction (GPT-2 50k). **Actual achieved:** 2572 each (identical) — corpus of 38 examples (~5k toks) has only ~2572 unique BPE merges; requesting larger vocab cannot be filled. Training time 0.8–0.9 s each, deterministic.

**Comparison (same corpus, `add_special_tokens False`):**
- Base: 5071 toks
- Experimental (any vocab): 4776 toks
- **Ratio 0.942, compression +5.8%** (experimental uses fewer tokens — more efficient)
- Sample: `lumenweave` 4→1 (−75%), `Aureline` 3→1 (−66%), `Mara` 2→1 (−50%), `The lighthouse...` 6→5 (−17%), but `scryglass` 3→4 (+33%)

**Qualitative:** Experimental tokenizer overfits tiny corpus — learns `lumenweave` as single token because it appears 6×, but `scryglass` still fragmented. Larger vocab doesn't help without larger corpus. Demonstrates diminishing returns: corpus-specific vocab gives modest 5.8% saving but *only* because it memorized this tiny corpus's invented terms; on general Miyuki writing the gain would shrink.

**Whether mismatch was found:** **Small mismatch only.** Qwen's large multilingual vocab is not inefficient for English Miyuki prose (1.36 tok/word is good). The 5.8% saving from a 2.5k experimental vocab is real but small and corpus-overfitted; not evidence of meaningful domain mismatch. See `evaluation/runs/phase2_vocabulary/comparisons.json` for per-text details.

**Reproducibility:** Second run same corpus+config yields same 2572 vocab and same 4776 token count (deterministic BPE on corpus, not stochastic model training). Recorded python 3.12.10, transformers 4.41.0, tokenizers 0.19.1, timestamp in `metadata.json`.

---

## 4. Special Tokens

**Phase 1 warning:** `Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.` — observed on every `AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")` and model load (transformers 4.41.0, huggingface_hub).

**Which:** Tokenizer `all_special_tokens ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]` with ids 151645,151643,151644 — within 151646 vocab. Model `config.vocab_size 151936`, `embedding.weight.shape [151936,896]` — 290 reserved ids beyond tokenizer (151646..151935).

**Why:** Qwen2 reserves 290 ids for future added tokens; the 3 specials are already in tokenizer and already have trained embeddings (part of released 0.5B instruct). The warning is generic `tokenization_utils_base` notice for `added_tokens_decoder` — here it's **expected, not harmful**. Our runtime sets `pad_token` only if `None` (for Qwen it's not), so we don't add new tokens.

**Necessary?** No new tokens were added by MIYUKI; warning is not actionable. No `resize_token_embeddings` called on production model (disallowed per spec, and would require training new rows). Experimental tokenizer is kept separate; its 2572 vocab would require resizing production model from 151936 → 2572 (incompatible) if ever connected — Phase 2 does not do this (no `model.resize_token_embeddings` logged; `model.get_input_embeddings().weight.shape` remains `[151936,896]`).

**Fix:** Document only (see `evaluation/runs/phase2_vocabulary/special_tokens.md`). No suppression, no weight modification, no altered production model saved. If future phase adds Miyuki invented tokens, protocol is `tok.add_tokens([...]); model.resize_token_embeddings(len(tok))` + fine-tune — not Phase 2.

---

## 5. Tests

**Existing:** All 31 Phase 1 tests still passing.  
**New Phase 2 (8):** `tests/test_phase2.py` — dataset validation, manifest counts, tokenizer report metrics, experimental tokenizer save/load, token-count comparison (5071 vs 4776 ratio 0.942), special-token consistency (len 151646 vs 151936 diff 290), deterministic corpus hash, no production overwrite.

**Total:** **39 passed, 0 failed** (`python -m pytest tests/ -v`, 20.3 s). No tests weakened.

---

## 6. Artifacts

**Datasets:**
- `datasets/miyuki-writing-0.1/dataset.jsonl` (38 records, 76 messages)
- `datasets/miyuki-writing-0.1/manifest.json` (plus copy `data/manifests/miyuki-writing-0.1.json`)
- `datasets/miyuki-writing-0.1/schema.json`, `DATASET_SPEC.md`, `README.md`, `review.yaml` (38 keep), `failure_mapping.yaml`
- `tools/dataset/create_miyuki_0_1.py`, `tools/dataset/manifest_and_validate.py`

**Evaluation:**
- `evaluation/runs/phase2_vocabulary/metadata.json` (repro), `tokenizer_report.json/.md` (5k toks, 1.36 tok/word), `tokenizer_experiment.json/.md` (3 vocab sizes → 2572 actual, 5.8% compression), `comparisons.json` (per-word), `special_tokens.md`, `README.md`
- `evaluation/runs/phase1_qwen2_0.5b/` retained (Phase 1 baseline)

**Artifacts (experimental, not production):**
- `artifacts/tokenizers/miyuki-experimental/miyuki-vocab-{8192,16000,32000}/` each with `tokenizer.json`, `vocab.json`, `merges.txt`, `tokenizer_config.json` (2572 actual)
- `artifacts/tokenizers/miyuki-experimental/README.md`

**Docs:** `docs/roadmap.md` (Phase 2 marked ✅), `docs/phase2_report.md` (this file).

**Not created/modified:** No `Trainer.train()`, no LoRA, no Qwen weights modified, no production tokenizer overwritten, no Story Brain.

---

## 7. Recommendation for Phase 3

**Proceed with LoRA using Qwen tokenizer** (no vocabulary change).

**Evidence:**
- Dataset `miyuki-writing-0.1` (38 high-quality, 9 distinct anti-meta/dialogue/context categories, 38/38 validated, all `keep`) directly addresses all 12 Phase 1 failures (see `failure_mapping.yaml`); 3 failures partially not solely dataset-close (long-range spatial tracking, premature EOS on nested quotes, tokenizer efficiency) are explicitly marked but even those are mitigated by dataset examples.
- Vocabulary study shows **small mismatch**: Qwen base 151646 already efficient for Miyuki English fiction (4.29 chars/tok); invented terms 3–4 toks is moderate, not catastrophic. Experimental tokenizer saves only 5.8% (295 toks on 5k corpus) despite overfitting tiny corpus to make `lumenweave` single-token; on larger general Miyuki writing the saving would be even smaller.
- Adding Miyuki invented tokens would require `resize_token_embeddings` (151936→151647+), random new rows, and retraining — cost and risk disproportionate to 0.35% corpus saving (lumenweave 6×). The 290 reserved model ids are sufficient for future adds without breaking base, but not needed now.
- Special tokens warning is expected/harmless; no embedding mismatch needing fix.

**Therefore:** Vocab adaptation appears **marginal**; prioritize writing dataset quality. Next milestone should be **First SFT (LoRA) on 0.5B using Qwen tokenizer** (e.g., `configs/training/example.yaml` with `dataset datasets/miyuki-writing-0.1/dataset.jsonl`, `adapter r 16`, `max_seq 1024`, CPU `batch 1` + `gradient_accumulation 8`), measure on `eval_v0.1` (12 prompts) vs Phase 1 baseline. Do not scale tokenizer/model until SFT signal is measured. If SFT proves instruction following still weak, *then* consider targeted vocab add + embedding resize as dedicated experiment before scaling to 1B.

**Hard stop per spec:** No fine-tune, no weight modification, no new model, no Story Brain in Phase 2 — stop here. Next decision from actual `miyuki-writing-0.1` + tokenizer measurements above, not assumptions.
