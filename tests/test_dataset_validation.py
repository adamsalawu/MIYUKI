"""Tests: dataset validation."""
import json
import tempfile
from pathlib import Path

from miyuki.datasets.validate import validate_file
from miyuki.datasets.schemas import detect_format


def _write_jsonl(tmp_path: Path, name: str, objs: list[dict]) -> Path:
    p = tmp_path / name
    with open(p, "w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    return p


def test_detect_format():
    assert detect_format({"messages": [{"role": "user", "content": "hi"}]}) == "conversational"
    assert detect_format({"prompt": "hi", "completion": "there"}) == "prompt_completion"
    assert detect_format({"text": "hello world"}) == "raw"
    assert detect_format({"instruction": "do x", "response": "y"}) == "story"


def test_validate_conversational_valid(tmp_path: Path):
    p = _write_jsonl(tmp_path, "valid.jsonl", [
        {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]},
        {"messages": [{"role": "user", "content": "Write a scene"}, {"role": "assistant", "content": "The rain fell."}]},
    ])
    report = validate_file(p, format="conversational")
    assert report.passed
    assert report.valid == 2
    assert report.invalid == 0


def test_validate_missing_fields(tmp_path: Path):
    p = _write_jsonl(tmp_path, "bad.jsonl", [
        {"messages": [{"role": "user", "content": "Hello"}]},  # missing assistant
        {"not_a_field": "oops"},
    ])
    report = validate_file(p, format="conversational")
    assert not report.passed
    assert report.invalid >= 1


def test_validate_empty_content(tmp_path: Path):
    p = _write_jsonl(tmp_path, "empty.jsonl", [
        {"messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": "hi"}]},
    ])
    report = validate_file(p, format="conversational")
    assert not report.passed


def test_validate_duplicates(tmp_path: Path):
    obj = {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]}
    p = _write_jsonl(tmp_path, "dup.jsonl", [obj, obj, obj])
    report = validate_file(p, format="conversational")
    assert report.duplicate_count == 1
    assert any(w.code == "duplicate" for w in report.warnings)


def test_validate_tiny(tmp_path: Path):
    p = _write_jsonl(tmp_path, "tiny.jsonl", [
        {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "K"}]},
    ])
    report = validate_file(p, format="conversational", tiny_threshold=10)
    assert report.tiny_count >= 1


def test_validate_prompt_completion(tmp_path: Path):
    p = _write_jsonl(tmp_path, "pc.jsonl", [
        {"prompt": "Continue:", "completion": " The story continued."},
    ])
    report = validate_file(p, format="prompt_completion")
    assert report.passed


def test_validate_story(tmp_path: Path):
    p = _write_jsonl(tmp_path, "story.jsonl", [
        {"instruction": "Write a scene", "response": "The wind moved through the pines."},
    ])
    report = validate_file(p, format="story")
    assert report.passed


def test_validate_malformed_conversation_consecutive_roles(tmp_path: Path):
    p = _write_jsonl(tmp_path, "consec.jsonl", [
        {"messages": [{"role": "user", "content": "Hi"}, {"role": "user", "content": "Again"}, {"role": "assistant", "content": "Hello"}]},
    ])
    report = validate_file(p, format="conversational")
    # Should warn but still valid (consecutive same role is warning, not error)
    assert report.valid == 1
    assert any(w.code == "consecutive_same_role" for w in report.warnings)
