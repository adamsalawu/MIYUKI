# Research System — Future Architecture

> Documented, not implemented in Phase 0.

## Purpose

When the author references real locations, historical people, terminology, organizations, or worldbuilding, Miyuki should be able to research without hallucinating.

## Pipeline

```
User context / story need
        │
        ▼
  Query planning (what to research? why?)
        │
        ▼
  WEB retrieval (search + fetch, with rate limits)
        │
        ▼
  Extraction (facts, quotes, dates, definitions)
        │
        ▼
  Relevance filter (is this useful for *this* story?)
        │
        ▼
  Structured storage (fact + source + confidence + timestamp)
        │
        ▼
  Context provision to model (relevant facts with citations)
```

## Key Principles

- **Sources preserved** — every fact retains URL/title/retrieved date.
- **Model treats web text as untrusted** — never blindly insert internet text as truth; surface source notes.
- **No uncontrolled scraping** — allowlists / query-scoped retrieval, not blind crawl.
- **User approval** for canon-relevant research that would change world facts.

## Storage

- Research notes are a memory category (see `memory_architecture.md`).
- Retrieval engine surfaces top-k relevant facts by story context.

## Questions to Answer in Phase 7

- Search provider abstraction?
- Caching / offline reuse?
- Fact extraction quality metrics?
