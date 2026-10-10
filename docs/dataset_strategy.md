# Dataset Strategy

## Formats

Miyuki supports four first-class formats (see `src/miyuki/datasets/schemas.py`):

- **raw** `{ text }` — continued pretraining
- **prompt_completion** `{ prompt, completion }`
- **conversational** `{ messages: [{role, content}] }` — TRL SFT-compatible
- **story** `{ instruction, story_context, characters, scene, world_information, recent_prose, response }`

`story` is converted to `conversational` at training time via structured user-message assembly.

## Provenance & Licensing

Every dataset has a `source` and `license` field tracked in `DatasetManifest`:

- `user-created` — author's own writing (preferred)
- `synthetic` — explicitly generated for Miyuki (labelled)
- `public-domain` — e.g., Project Gutenberg where rights are clear
- `licensed` — only with explicit permission
- `permitted` — other explicitly permitted training material

**Do NOT** build a corpus by scraping copyrighted books or fanfiction without checking rights. `source` and `license` are mandatory in manifests for real datasets.

## Versioning

- Identity: `miyuki-writing-0.1`, `0.2`, …
- `DatasetManifest` records: name, version, source, license, creation date, num_examples, split sizes, token stats, file hashes, preprocessing config, notes.
- Never silently overwrite — new version = new files + new manifest in `data/manifests/`.

## Validation

Before training, run `miyuki dataset validate --path <file> --format <auto|...>`.

Detects:

- missing fields, empty examples
- malformed conversations (empty content, consecutive same-role without alternation)
- duplicate examples (SHA256 hash)
- suspiciously tiny examples (<10 chars)
- impossible metadata (e.g., `word_count` mismatched)
- train/eval contamination (hash intersection via `check_contamination()`)

See `src/miyuki/datasets/validate.py`.

## Splits

- `data/raw/` — raw sources (gitignored if large)
- `data/processed/` — cleaned, validated examples
- `data/splits/` — train/eval/test splits with manifests
- `data/manifests/` — versioned manifests (committed)

Split sizes and hashes are in the manifest — reproducible.

## Training Integration

- TRL's `SFTTrainer` supports language-modeling, prompt-completion, and conversational — we emit the format TRL expects.
- Training pipeline (`src/miyuki/training/train.py`) uses `format_for_training()` to normalize all inputs.

## Roadmap

- **0.1 (Phase 0):** `smoke_demo.jsonl` — 20 synthetic examples proving the loop.
- **0.2 (Phase 2):** Hand-authored Miyuki writing examples aligned to `docs/miyuki_writing_spec.md`.
- **Later:** Public-domain extracts, user writing, larger synthetic sets — each with manifest.

## Hygiene

- Deduplicate before training.
- Validate before training — fail fast on errors.
- Check train/eval hash contamination.
- Track every dataset version used in `metadata.json` + experiment record.
