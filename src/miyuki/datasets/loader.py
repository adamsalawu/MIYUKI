"""Dataset loading helpers for training - formats for TRL."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if isinstance(data, list):
            return data
    result = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        result.append(json.loads(line))
    return result


def format_for_training(
    example: dict[str, Any],
    tokenizer: Any = None,
    fmt: str = "auto",
) -> dict[str, Any]:
    """Convert an example to a training-ready dict.

    Supports TRL's prompt-completion and conversational formats.
    Returns dict with 'messages' or 'prompt'/'completion' or 'text'.
    """
    from .schemas import detect_format

    eff = fmt if fmt != "auto" else detect_format(example)

    if eff == "conversational":
        return {"messages": example["messages"]}
    if eff == "prompt_completion":
        return {"prompt": example["prompt"], "completion": example["completion"]}
    if eff == "story":
        # Convert story example to conversational with injected context
        instruction = example.get("instruction", "")
        story_context = example.get("story_context", "")
        response = example.get("response") or example.get("completion") or ""
        # Build a rich user message from story fields
        parts = [f"Instruction: {instruction}"]
        if story_context:
            parts.append(f"Story context: {story_context}")
        if example.get("scene"):
            parts.append(f"Scene: {example['scene']}")
        if example.get("world_information"):
            parts.append(f"World: {example['world_information']}")
        if example.get("characters"):
            chars = example["characters"]
            if isinstance(chars, list):
                parts.append(f"Characters: {json.dumps(chars, ensure_ascii=False)}")
        if example.get("recent_prose"):
            parts.append(f"Recent prose: {example['recent_prose']}")
        user_content = "\n\n".join(parts)
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response},
            ]
        }
    if eff == "raw":
        return {"text": example["text"]}
    # fallback: try to pass through
    return example
