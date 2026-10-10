# Miyuki Writing Corpus 0.1

**Version:** `miyuki-writing-0.1` (first human-authored writing dataset)  
**Size:** 38 exemplars, 21776 chars, 5072 tokens (Qwen2-0.5B tokenizer, 133.5 tok/ex), avg 62.6 words target, range 16–119.  
**Source:** human-authored (AI-assisted drafting, human-edited), `CC-BY-4.0`.  
**Validation:** `38/38 valid, 0 warnings`, tokenization compatibility OK, SHA256 `9e1b97461f989482...` (see `manifest.json`).  

## Purpose

Teach `PROMPT → UNDERSTAND → WRITE (not explain)` — actual story behavior. Each `assistant` target **performs** the request; never explains it. Directly addresses Phase 1 failures (meta-analysis, flat dialogue, lore dumping, quiet-scene failure, context hedging, etc.).

## Structure

```
datasets/miyuki-writing-0.1/
  dataset.jsonl   — 38 conversational records (user → assistant)
  manifest.json   — version, counts, token stats, provenance
  schema.json     — JSON Schema for validation
  review.yaml     — human quality review per exemplar (keep/edit/discard)
  DATASET_SPEC.md — specification
  README.md       — this file
```

Each record: `messages` (TRL SFT shape, chat template applied at train) + `metadata {id, instruction, context, category[], skills[], source, notes, word_count}`. Categories use controlled vocabulary (allows multiple tags) for later ablations.

## Categories (38 examples)

- `scene` 29, `quiet` 10, `anti_meta` 10, `description` 14, `pacing` 14, `character_consistency` 14, `dialogue` 10, `subtext` 7, `character_voice` 7, `rewriting` 5, `context_retention` 5, etc. (full counts in `manifest.json`).

## Provenance & Licensing

- No scraped copyrighted text. Recurring invented terms (Aureline, Kestrel, lumenweave, scryglass, voidtide, Emberhearth) are original for Miyuki.
- Every example traceable via `id` and `notes` (which Phase 1 failure it addresses).

## Usage

```powershell
python -m miyuki.cli dataset validate --path datasets/miyuki-writing-0.1/dataset.jsonl --format conversational
python tools/dataset/manifest_and_validate.py
```

For training (Phase 3, not Phase 2): `python -m miyuki.training.train --config configs/training/example.yaml --dataset datasets/miyuki-writing-0.1/dataset.jsonl` (uses conversational format, chat template). Do NOT train on `evaluation/prompts/eval_v0.1.jsonl`.

## Human Review

See `review.yaml` — each exemplar rated for clarity, writing quality, target behavior, category correctness, keep/edit/discard. All 38 marked `keep` after edit.

## Links

- Spec: `DATASET_SPEC.md`
- Phase 1 failures mapped in `../evaluation/runs/phase2_vocabulary/` and `review.yaml`
- Tokenizer study: `evaluation/runs/phase2_vocabulary/`
