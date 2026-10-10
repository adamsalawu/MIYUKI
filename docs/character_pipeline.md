# Character Intelligence — Future Pipeline

> Documented, not implemented in Phase 0.

## Pipeline

```
Raw character (freeform author notes)
        │
        ▼
  Extraction  (structured fields: name, age, traits, goals, voice, history)
        │
        ▼
  Normalization (consistent schema, types)
        │
        ▼
  Conflict detection (contradictions in traits / timeline / relationships)
        │
        ▼
  Missing-information detection (gaps: voice sample? motivation? relationships?)
        │
        ▼
  Optional research (real-world analogues, terminology)
        │
        ▼
  Proposed improvements (diffs, not overwrites)
        │
        ▼
  User approval (approve / edit / reject each proposal)
        │
        ▼
  Structured character record (canonical)
```

## Principles

- **User in control** — Miyuki proposes changes rather than silently rewriting canon.
- **Diffs, not overwrites** — every suggestion is an inspectable diff with rationale.
- **Voice samples matter** — character dossiers include exemplar dialogue for consistency checks.

## Interaction with Story Brain

- Structured records feed the Story Brain's `CHARACTERS / CHARACTER STATES / RELATIONSHIPS` context.
- Timeline events linked to characters feed `TIMELINE / RELEVANT MEMORIES`.

## Evaluation

- Character consistency prompts (see `evaluation/prompts`) measure voice adherence.
- Human ratings dimension: `character_consistency`.

## When to Build

Phase 6 — after first SFT and evaluation iteration prove the base writing loop.
