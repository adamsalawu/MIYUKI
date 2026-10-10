# Memory Architecture — Future

> Documented, not implemented in Phase 0.

## Categories

| Category | Example | Mutability |
|---|---|---|
| Permanent canon | World facts, character dossiers, established events | Append + approved edits only |
| Character memory | What a character knows / has experienced | Grows with story |
| World memory | Locations, factions, lore, rules | Curated |
| Scene memory | Recent scenes / chapters (full prose) | Rolling window |
| Temporary context | Active scene context, author intent | Ephemeral |
| Author notes | Outlines, to-dos, directions to Miyuki | User-owned |
| Research | Web research with sources | Cached, versioned |
| Timeline events | Dated events with participants / consequences | Canonical |

## Future Retrieval / Context Engine

- Not implemented in this milestone.
- Will surface **relevant** memories for a given scene/prompt (embedding + keyword + recency).
- Feeds the Story Brain's context assembly:

```
CURRENT SCENE
CHARACTERS
CHARACTER STATES
RELATIONSHIPS
WORLD FACTS
TIMELINE
RELEVANT MEMORIES
RESEARCH NOTES
AUTHOR INTENT
RECENT PROSE
       │
       ▼
   Model prompt
```

## Model Separation

The model itself does **not** store memory permanently. The Miyuki application owns memory and retrieval and provides relevant context per generation. This keeps the model stateless and prompt-driven.

## When to Build

Phase 5 — Story Brain and memory jointly, after writing quality has been baselined.
