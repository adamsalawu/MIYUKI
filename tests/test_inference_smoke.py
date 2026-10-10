"""Tests: inference pipeline shape (mocked model)."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from miyuki.config import MiyukiConfig
from miyuki.inference.pipeline import run_inference, save_result


def test_inference_result_shape():
    cfg = MiyukiConfig.from_yaml("C:/Aether/RisingStar/MIYUKI/configs/model/baseline.yaml")
    fake_result = MagicMock()
    fake_result.prompt = "hello"
    fake_result.generated = "world"
    fake_result.full_text = "hello world"
    fake_result.model_name = cfg.model.name
    fake_result.config_snapshot = cfg.to_dict()
    fake_result.runtime_info = {"backend": "mock", "device": "cpu"}
    fake_result.generation_params = {"temperature": 0.8}
    fake_result.timings = {"total_s": 0.1}

    with patch("miyuki.inference.pipeline.TransformersRuntime") as MockRT:
        MockRT.return_value.generate.return_value = fake_result
        record = run_inference(cfg, "hello")
    assert record["prompt"] == "hello"
    assert record["generated"] == "world"
    assert "timestamp" in record
    assert "model" in record


def test_save_result(tmp_path: Path):
    cfg = MiyukiConfig.from_yaml("C:/Aether/RisingStar/MIYUKI/configs/model/baseline.yaml")
    fake_result = MagicMock()
    fake_result.prompt = "hi"
    fake_result.generated = "there"
    fake_result.full_text = "hi there"
    fake_result.model_name = cfg.model.name
    fake_result.config_snapshot = cfg.to_dict()
    fake_result.runtime_info = {"backend": "mock"}
    fake_result.generation_params = {}
    fake_result.timings = {}
    with patch("miyuki.inference.pipeline.TransformersRuntime") as MockRT:
        MockRT.return_value.generate.return_value = fake_result
        rec = run_inference(cfg, "hi")
    out = tmp_path / "out.json"
    save_result(rec, out)
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["generated"] == "there"
