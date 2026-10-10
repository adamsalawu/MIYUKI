"""Tests: dataset versioning & manifest, serialization."""
import json
from pathlib import Path

from miyuki.datasets.versioning import compute_manifest, DatasetManifest
from miyuki.datasets.loader import load_jsonl, format_for_training


def test_manifest_compute(tmp_path: Path):
    p = tmp_path / "train.jsonl"
    p.write_text('{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}\n', encoding="utf-8")
    manifest = compute_manifest(
        name="miyuki-writing-test",
        version="0.0.1",
        files={"train": p},
        source="synthetic",
        license="MIT",
    )
    assert manifest.name == "miyuki-writing-test"
    assert manifest.version == "0.0.1"
    assert manifest.num_examples == 1
    assert manifest.splits["train"] == 1
    assert manifest.source == "synthetic"


def test_manifest_roundtrip(tmp_path: Path):
    p = tmp_path / "train.jsonl"
    p.write_text('{"text": "hello world"}\n', encoding="utf-8")
    manifest = compute_manifest("test", "0.1", {"train": p})
    out = tmp_path / "manifest.json"
    manifest.to_json(out)
    loaded = DatasetManifest.from_json(out)
    assert loaded.name == "test"
    assert loaded.num_examples == 1


def test_load_jsonl(tmp_path: Path):
    p = tmp_path / "data.jsonl"
    p.write_text('{"prompt": "a", "completion": "b"}\n{"prompt": "c", "completion": "d"}\n', encoding="utf-8")
    data = load_jsonl(p)
    assert len(data) == 2


def test_format_for_training_conversational():
    ex = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
    out = format_for_training(ex)
    assert "messages" in out


def test_format_for_training_story():
    ex = {"instruction": "Write a scene", "story_context": "A village", "response": "The village slept."}
    out = format_for_training(ex)
    assert "messages" in out
    assert out["messages"][1]["content"] == "The village slept."


def test_format_for_training_prompt_completion():
    ex = {"prompt": "Continue:", "completion": " the end."}
    out = format_for_training(ex)
    assert out["prompt"] == "Continue:"
