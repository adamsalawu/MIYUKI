# Miyuki Writing Corpus 0.1 — Dataset Specification

**Version:** `miyuki-writing-0.1`  
**Goal:** Teach `PROMPT → UNDERSTAND → WRITE (not explain)` — actual story behavior, not chatbot commentary.

## Format

Each record is a **conversational** example (`src/miyuki/datasets/schemas.py:ConversationalExample`) with:

```json
{
  "messages": [
    { "role": "user", "content": "Instruction + Context + Request" },
    { "role": "assistant", "content": "Target prose (desired MIYUKI behavior)" }
  ],
  "metadata": {
    "id": "miyuki-0.1-001",
    "instruction": "Continue the story naturally...",
    "context": "Mara is a former lighthouse keeper's daughter...",
    "category": ["scene","continuation","pacing"],
    "skills": ["instruction_following","continuity","prose_flow"],
    "source": "human-authored",
    "notes": "Purpose / before-after / anti-pattern targeted",
    "word_count": 142
  }
}
```

- `messages` is the TRL SFT training shape (chat template will be applied at train time).
- `metadata` preserves traceability (instruction, context, category, skills, source, notes). Required.
- `category` values from controlled vocabulary (allows multiple): `scene, dialogue, character_voice, character_consistency, context_retention, emotion, action, description, pacing, subtext, rewriting, instruction_following, anti_meta, transitional, atmospheric, quiet`.
- `source` is `human-authored` or `human-edited` (never raw Qwen output; no synthetic weak data).

## Coverage (35 examples, 38 total with variants)

**Scene writing (8):** quiet morning, tense harbor storm, transitional doorway, atmospheric night market tail, environmental observatory, lighthouse continuation (fixes Phase 1 generic), Kyoto tea refined, storm harbor.

**Dialogue (8):** siblings kitchen with subtext (fixes flat dialogue), mentor/student dramatized (fixes lore dump), interruptions/overlap, awkward train platform (fixes meta-answer), emotional letter (fixes melodrama), distinct character voices (Jinx), disagreement with power dynamic, casual reveal.

**Character consistency (6):** Jinx witty blunt, Mara Lighthouse, Elena mug (2 variants for continuity), Haru writer Kyoto, Kestrel scout.

**Context retention (6):** Elena chipped blue mug (grandmother's, only mug, hates lavender) → reaction correctly uses mug/declines tea; blue scarf lost yesterday → doesn't wear; Kestrel compass broken → Jinx sarcasm; lumenweave invented term; Aureline city lore.

**Anti-meta / Instruction following (7):** quiet scene must be scene not "The answer is...", tone must be continuation not essay, rewriting must be vivid edit not generic, context must commit not hedge ("It could be..."), no "Here is a scene...", no moral summary, no repetitive adjectives.

**Style skills:** varied sentence rhythm, controlled description (selective), natural prose, sensory detail (non-visual), emotional subtext (show via action), pacing (fast action short sentences vs slow lingering), restrained exposition (show via objects/behavior).

## Quality Rules

- Every `assistant` target **performs** the request; never explains the task.
- No `As an AI`, `Here is a scene`, `This scene demonstrates`, `In conclusion`.
- Dialogue: contractions, interruptions, distinct diction per character; tags beat adverbs.
- Emotion via action (`tea slopped`) not naming (`she felt anxious`).
- Transitions are images/actions, not labels (`She followed him into the hall` vs `Meanwhile`).
- Continuity: respect all context facts; no retcon.
- Truncation-safe: complete thought even if short; no mid-sentence cut.

## Provenance

- Authored by human for MIYUKI, checked against `docs/miyuki_writing_spec.md`.
- No copyrighted text, no scraped books/fanfiction. Recurring fictional terms (Aureline, Kestrel, lumenweave, scryglass, voidtide) are original.
- License: `CC-BY-4.0` for this corpus (permissive for training, attribution required).
- Each example has `source` and `notes` explaining Phase 1 failure addressed.

## Versioning

- Checksum via `DatasetManifest.file_hashes` (SHA256), token stats, category counts.
- Never silently overwrite `miyuki-writing-0.1`; next version is `0.2`.
