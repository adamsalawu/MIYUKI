"""Abstract runtime interface - implemented by each backend."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from ..config import MiyukiConfig


@dataclass
class GenerationResult:
    prompt: str
    generated: str
    full_text: str
    model_name: str
    config_snapshot: dict[str, Any]
    runtime_info: dict[str, Any]
    generation_params: dict[str, Any]
    timings: dict[str, float] = field(default_factory=dict)


class RuntimeBase(ABC):
    """Common interface for all Miyuki runtimes.

    Conceptual hierarchy:
        Miyuki Model
        ├── TransformersRuntime (implemented, CPU/CUDA)
        ├── GGURuntime (future, llama.cpp)
        └── FutureRuntimes
    """

    def __init__(self, config: MiyukiConfig):
        self.config = config

    @abstractmethod
    def load(self) -> None:
        """Load model + tokenizer onto target device."""

    @abstractmethod
    def generate(self, prompt: str, **overrides) -> GenerationResult:
        """Generate text from prompt."""

    @abstractmethod
    def is_loaded(self) -> bool:
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str:
        ...
