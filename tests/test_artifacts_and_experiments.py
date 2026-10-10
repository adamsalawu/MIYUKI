"""Tests: model artifacts, experiment tracking, checkpoint discovery."""
import json
from pathlib import Path

from miyuki.artifacts.manager import save_model_metadata, load_model_metadata, discover_checkpoints
from miyuki.experiments.tracking import record_experiment, list_experiments


def test_model_metadata_roundtrip(tmp_path: Path):
    out = save_model_metadata(
        model_dir=tmp_path / "miyuki-test",
        base_model="hf-internal-testing/tiny-random-gpt2",
        version="0.1.0-test",
        dataset="data/processed/smoke_demo.jsonl",
        dataset_version="0.1.0",
        training_config={"seed": 42},
        tokenizer_id="hf-internal-testing/tiny-random-gpt2",
        context_length=512,
    )
    assert out.exists()
    loaded = load_model_metadata(tmp_path / "miyuki-test")
    assert loaded["base_model"] == "hf-internal-testing/tiny-random-gpt2"
    assert loaded["version"] == "0.1.0-test"


def test_experiment_record_and_list(tmp_path: Path, monkeypatch):
    # Patch experiments dir to tmp_path
    import miyuki.experiments.tracking as trk
    monkeypatch.setattr(trk, "_experiments_dir", lambda: tmp_path / "experiments")
    exp_id = record_experiment(
        model_name="test-model",
        dataset="test-dataset.jsonl",
        dataset_version="0.1",
        training_config={"seed": 42},
        hardware={"cpu": "test"},
        duration_s=1.0,
        artifact_location=str(tmp_path / "ckpt"),
    )
    assert exp_id == "0001"
    exps = list_experiments()
    assert len(exps) == 1
    assert exps[0]["model"] == "test-model"


def test_discover_checkpoints_empty(tmp_path: Path):
    found = discover_checkpoints(tmp_path / "nope")
    assert found == []
