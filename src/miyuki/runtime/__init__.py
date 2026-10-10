"""Runtime abstraction - supports multiple backends behind a common interface."""
from .base import GenerationResult, RuntimeBase
from .transformers_runtime import TransformersRuntime

__all__ = ["RuntimeBase", "GenerationResult", "TransformersRuntime"]
