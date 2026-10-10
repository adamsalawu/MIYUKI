"""Dataset schemas for Miyuki - supports raw, prompt/completion, conversational, story-specific."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Raw text - for continued pretraining experiments
# ---------------------------------------------------------------------------

class RawTextExample(BaseModel):
    text: str = Field(min_length=1, description="Raw text content")
    source: Optional[str] = Field(default=None)
    metadata: Optional[dict[str, Any]] = Field(default=None)


# ---------------------------------------------------------------------------
# Prompt / Completion
# ---------------------------------------------------------------------------

class PromptCompletionExample(BaseModel):
    prompt: str = Field(min_length=1)
    completion: str = Field(min_length=1)
    metadata: Optional[dict[str, Any]] = Field(default=None)


# ---------------------------------------------------------------------------
# Conversational (TRL-compatible)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(description="Message role")
    content: str = Field(min_length=1)

    @field_validator("role", mode="before")
    @classmethod
    def normalize_role(cls, v: str) -> str:
        if isinstance(v, str):
            v = v.strip().lower()
        return v


class ConversationalExample(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)

    @field_validator("messages")
    @classmethod
    def check_conversation(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        if len(v) == 0:
            raise ValueError("messages must not be empty")
        # Must contain at least one user and one assistant
        roles = [m.role for m in v]
        if "user" not in roles:
            raise ValueError("conversation must contain at least one user message")
        if "assistant" not in roles:
            raise ValueError("conversation must contain at least one assistant message")
        # Alternation is soft-checked (warn elsewhere), but basic sanity:
        if v[0].role == "assistant" and len(v) == 1:
            raise ValueError("single-message conversation cannot be only assistant")
        return v

    metadata: Optional[dict[str, Any]] = Field(default=None)


# ---------------------------------------------------------------------------
# Story-specific - Miyuki's core creative writing format
# Extends prompt/completion with structured story context.
# ---------------------------------------------------------------------------

class StoryContext(BaseModel):
    """Structured story context provided to the model."""
    instruction: str = Field(min_length=1, description="Author intent / writing objective")
    story_context: Optional[str] = Field(default=None, description="Summary or prior prose")
    characters: Optional[list[dict[str, Any]]] = Field(default=None)
    scene: Optional[str] = Field(default=None)
    world_info: Optional[str] = Field(default=None, alias="world_information")
    recent_prose: Optional[str] = Field(default=None)

    model_config = {"populate_by_name": True}


class StoryExample(BaseModel):
    """Full story training example with structured context + desired response."""
    instruction: str = Field(min_length=1, description="Writing objective")
    story_context: Optional[str] = Field(default=None)
    characters: Optional[list[dict[str, Any]]] = Field(default=None)
    scene: Optional[str] = Field(default=None)
    world_information: Optional[str] = Field(default=None)
    recent_prose: Optional[str] = Field(default=None)
    # The desired assistant response (what Miyuki should write)
    response: str = Field(min_length=1, description="Desired model output")
    # Alias: some pipelines use 'completion' instead of 'response'
    completion: Optional[str] = Field(default=None)

    metadata: Optional[dict[str, Any]] = Field(default=None)

    @field_validator("response", mode="before")
    @classmethod
    def coerce_response(cls, v: Any, info) -> Any:
        # allow completion alias
        return v

    def effective_response(self) -> str:
        return self.response or self.completion or ""

    # Also allow conversational shape for story data: messages with injected context
    messages: Optional[list[ChatMessage]] = Field(default=None, description="Alternative conversational form")


# ---------------------------------------------------------------------------
# Helpers to detect format
# ---------------------------------------------------------------------------

def detect_format(obj: dict[str, Any]) -> str:
    """Infer which schema an object belongs to."""
    if "messages" in obj:
        return "conversational"
    if "response" in obj or "world_information" in obj or "story_context" in obj:
        # story uses response + structured fields
        if "instruction" in obj and "response" in obj:
            return "story"
        if "response" in obj:
            return "story"
    if "prompt" in obj and "completion" in obj:
        return "prompt_completion"
    if "text" in obj and len(obj) <= 3:
        return "raw"
    # fallback
    if "instruction" in obj:
        return "story"
    return "unknown"
