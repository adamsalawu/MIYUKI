"""Model configuration system - single source of truth for model identity.

All model names/identifiers live in YAML configs, not scattered through code.
Changing base model requires editing a config file, not rewriting Python.

Supports:
  - model identifier, family, tokenizer, context length
  - precision / quantization / device
  - generation defaults
  - training + adapter configuration
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class ModelIdentity(BaseModel):
    name: str = Field(description="HF model identifier, e.g. 'gpt2' or 'Qwen/Qwen2-0.5B-Instruct'")
    family: str = Field(description="Model family: gpt2, qwen2, llama, mistral, etc.")
    tokenizer: Optional[str] = Field(default=None, description="Tokenizer identifier (defaults to model name)")
    context_length: int = Field(default=2048, ge=1, description="Max context length in tokens")
    trust_remote_code: bool = Field(default=False)

    def resolved_tokenizer(self) -> str:
        return self.tokenizer or self.name


class RuntimeConfig(BaseModel):
    device: str = Field(default="auto", description="auto | cpu | cuda | cuda:0 | mps")
    dtype: str = Field(default="auto", description="auto | float32 | float16 | bfloat16")
    quantization: Optional[str] = Field(default=None, description="None | 4bit | 8bit | gguf")
    attn_implementation: Optional[str] = Field(default=None)

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        allowed = {"auto", "cpu", "mps"}
        if v in allowed or v.startswith("cuda"):
            return v
        raise ValueError(f"device must be one of auto/cpu/mps/cuda* got {v}")


class GenerationConfig(BaseModel):
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    max_new_tokens: int = Field(default=256, ge=1, le=8192)
    repetition_penalty: float = Field(default=1.0, ge=0.5, le=2.0)
    do_sample: bool = Field(default=True)
    seed: Optional[int] = Field(default=None)


class AdapterConfig(BaseModel):
    type: Optional[str] = Field(default=None, description="None | lora | qlora")
    r: int = Field(default=8, ge=1)
    alpha: int = Field(default=16, ge=1)
    dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    target_modules: Optional[list[str]] = Field(default=None)
    enabled: bool = Field(default=False)

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if v not in {"lora", "qlora"}:
            raise ValueError(f"adapter type must be lora|qlora got {v}")
        return v


class TrainingConfig(BaseModel):
    output_dir: str = Field(default="checkpoints/miyuki-0.1")
    num_epochs: int = Field(default=1, ge=1)
    per_device_train_batch_size: int = Field(default=1, ge=1)
    per_device_eval_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_ratio: float = Field(default=0.03, ge=0, le=1)
    lr_scheduler_type: str = Field(default="cosine")
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=100, ge=1)
    eval_steps: Optional[int] = Field(default=None)
    save_total_limit: int = Field(default=3, ge=1)
    max_seq_length: int = Field(default=1024, ge=1)
    bf16: bool = Field(default=False)
    fp16: bool = Field(default=False)
    gradient_checkpointing: bool = Field(default=False)
    seed: int = Field(default=42)
    resume_from_checkpoint: Optional[str] = Field(default=None)
    push_to_hub: bool = Field(default=False)
    report_to: str = Field(default="none")
    assistant_only_loss: bool = Field(default=False, description="If true, mask system/user tokens with -100, only assistant tokens contribute to loss (TRL-style). Preserved as False for previous experiments (Phase0-3C). Phase4 enables it.")

    # mixed precision sanity
    @field_validator("bf16")
    @classmethod
    def check_precision(cls, v: bool, info) -> bool:  # type: ignore
        return v


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class MiyukiConfig(BaseModel):
    """Root configuration - loaded from YAML."""

    model: ModelIdentity
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    adapter: AdapterConfig = Field(default_factory=AdapterConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    # optional metadata
    version: str = Field(default="0.1.0")
    description: Optional[str] = Field(default=None)

    # helpers
    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> "MiyukiConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Config {p} must be a YAML mapping")
        return cls.model_validate(data)

    def to_yaml(self, path: str | os.PathLike) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(self.model_dump(mode="python"), sort_keys=False), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MiyukiConfig":
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python")


def list_configs(search_dir: str | os.PathLike = "configs/model") -> list[Path]:
    base = Path(search_dir)
    if not base.exists():
        return []
    return sorted(base.glob("*.yaml")) + sorted(base.glob("*.yml"))


def load_config_or_default(path: Optional[str | os.PathLike] = None) -> MiyukiConfig:
    """Load given config or fall back to baseline."""
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    # search default locations
    candidates.extend([
        Path("configs/model/baseline.yaml"),
        Path("C:/Aether/RisingStar/MIYUKI/configs/model/baseline.yaml"),
        Path(__file__).parent.parent.parent / "configs" / "model" / "baseline.yaml",
    ])
    for c in candidates:
        if c.exists():
            return MiyukiConfig.from_yaml(c)
    # ultimate fallback - tiny in-memory default (works offline)
    return MiyukiConfig(
        model=ModelIdentity(
            name="hf-internal-testing/tiny-random-gpt2",
            family="gpt2",
            tokenizer="hf-internal-testing/tiny-random-gpt2",
            context_length=1024,
        ),
        runtime=RuntimeConfig(device="auto", dtype="auto"),
        generation=GenerationConfig(temperature=0.8, max_new_tokens=64),
    )
