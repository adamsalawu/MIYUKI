"""Tests: configuration loading."""
import tempfile
from pathlib import Path

import pytest

from miyuki.config import AdapterConfig, MiyukiConfig


def test_load_baseline_config():
    cfg = MiyukiConfig.from_yaml("C:/Aether/RisingStar/MIYUKI/configs/model/baseline.yaml")
    assert cfg.model.name == "hf-internal-testing/tiny-random-gpt2"
    assert cfg.model.family == "gpt2"
    assert cfg.runtime.device in ("auto", "cpu", "cuda", "mps", "cuda:0")


def test_load_smoke_config_has_lora():
    cfg = MiyukiConfig.from_yaml("C:/Aether/RisingStar/MIYUKI/configs/training/smoke.yaml")
    assert cfg.adapter.enabled is True
    assert cfg.adapter.type == "lora"


def test_config_roundtrip(tmp_path: Path):
    cfg = MiyukiConfig.from_yaml("C:/Aether/RisingStar/MIYUKI/configs/model/baseline.yaml")
    out = tmp_path / "out.yaml"
    cfg.to_yaml(out)
    cfg2 = MiyukiConfig.from_yaml(out)
    assert cfg2.model.name == cfg.model.name
    assert cfg2.generation.temperature == cfg.generation.temperature


def test_adapter_validation():
    with pytest.raises(Exception):
        AdapterConfig(type="invalid_type")


def test_model_tokenizer_fallback():
    cfg = MiyukiConfig.from_yaml("C:/Aether/RisingStar/MIYUKI/configs/model/baseline.yaml")
    assert cfg.model.resolved_tokenizer() == cfg.model.tokenizer
    # if tokenizer is None, falls back to name
    cfg.model.tokenizer = None
    assert cfg.model.resolved_tokenizer() == cfg.model.name
