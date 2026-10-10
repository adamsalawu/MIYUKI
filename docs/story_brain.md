# Story Brain — Future Interface

> Documented, not implemented in Phase 0.

## Role

The Story Brain is the context-assembly layer that sits between the Miyuki application (which owns projects, characters, lore, timeline) and the Miyuki model (which writes). The model does NOT own persistent story state.

## Intended Model Interface

Eventually the model should be able to receive structured context like:

```yaml
current_scene:
  chapter: 3
  scene: 2
  location: "Abandoned observatory, mountain"
  time: "Late afternoon"
characters:
  - name: "Mara"
    state: "Tired, guarded, hopeful"
    voice_sample: "..."
  - name: "Jinx"
    state: "Restless, quipping"
relationships:
  - pair: ["Mara", "Jinx"]
    dynamic: "Mentor/student, tense after last chapter"
world_facts:
  - "The lighthouse has been dark for 20 years."
  - "Storm season blocks the mountain pass."
timeline:
  - date: "2026-09-20"
    event: "Mara returned to town"
relevant_memories:
  - scene: "Ch.2, Scene 4 - The letter"
    excerpt: "..."
research_notes:
  - fact: "Observatories of this era used refracting telescopes"
    source: "https://example.com"
author_intent: "Continue the scene, keep it quiet, no reveal yet."
recent_prose: "The wind moved through the pines..."
```

## How It Works

1. Application gathers current project state.
2. Retrieval engine surfaces relevant memories/research/world facts.
3. Story Brain formats the context (template, token budget, ordering).
4. Prompt is sent to `RuntimeBase.generate(prompt)`.
5. Output is shown to author; author edits/approves; story state updates.

## Separation

- Model: stateless, prompt → continuation
- Application: stateful, owns canon, retrieval, approval flow

This separation is load-bearing — do not blur it.

## Status

- Runtime abstraction (`src/miyuki/runtime/base.py`) is ready.
- Actual Story Brain assembly is Phase 5, after baseline writing is measured.
