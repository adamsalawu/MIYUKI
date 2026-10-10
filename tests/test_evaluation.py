"""Tests: evaluation runner (mocked inference for determinism)."""
import json
from pathlib import Path
from unittest.mock import patch

from miyuki.config import MiyukiConfig
from miyuki.evaluation.runner import load_prompts, run_evaluation


def test_load_prompts():
    prompts = load_prompts("C:/Aether/RisingStar/MIYUKI/evaluation/prompts/eval_v0.1.jsonl")
    assert len(prompts) >= 12
    ids = {p["id"] for p in prompts}
    assert "continuation-01" in ids
    assert "context-retention-01" in ids


def test_evaluation_runner_mocked(tmp_path: Path):
    cfg = MiyukiConfig.from_yaml("C:/Aether/RisingStar/MIYUKI/configs/model/baseline.yaml")
    # Mock the TransformersRuntime to avoid loading real model
    # Runner now reuses a single runtime instance — patch that class
    from miyuki.runtime.base import GenerationResult

    fake_result = GenerationResult(
        prompt="hello",
        generated="mocked output",
        full_text="hello mocked output",
        model_name=cfg.model.name,
        config_snapshot=cfg.to_dict(),
        runtime_info={"backend": "mock", "device": "cpu", "prompt_tokens": 5, "output_tokens": 3, "tokens_per_sec": 10.0},
        generation_params={},
        timings={"generate_s": 0.1, "total_s": 0.12},
    )

    with patch("miyuki.runtime.transformers_runtime.TransformersRuntime") as MockRT:
        mock_inst = MockRT.return_value
        mock_inst.load.return_value = None
        mock_inst.generate.return_value = fake_result
        # Ensure ._model exists for param counting (not needed but set)
        mock_inst._model = None
        result = run_evaluation(
            model_config=cfg,
            prompts_file="C:/Aether/RisingStar/MIYUKI/evaluation/prompts/eval_v0.1.jsonl",
            output_dir=str(tmp_path / "runs"),
            report_dir=str(tmp_path / "reports"),
        )
    assert result["num_prompts"] >= 12
    assert Path(result["outputs_path"]).exists()
    assert Path(result["report_path"]).exists()
    # Verify outputs.jsonl has same number of entries
    lines = Path(result["outputs_path"]).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == result["num_prompts"]
    for line in lines:
        obj = json.loads(line)
        assert obj["generated"] == "mocked output"
    # Verify runtime reuse: load called once, generate called per prompt
    assert mock_inst.load.call_count == 1
    assert mock_inst.generate.call_count >= 12
