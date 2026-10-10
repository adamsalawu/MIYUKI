"""Dataset validation - detects issues before training."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from .schemas import (
    ConversationalExample,
    PromptCompletionExample,
    RawTextExample,
    StoryExample,
    detect_format,
)


class ValidationIssue(BaseModelLike := object):
    pass  # placeholder for type, actual defined below

from pydantic import BaseModel, Field


class ValidationIssueModel(BaseModel):
    level: str = Field(description="error | warning")
    code: str
    message: str
    line: Optional[int] = None
    field: Optional[str] = None


class ValidationReport(BaseModel):
    path: str
    format: str
    total: int
    valid: int
    invalid: int
    errors: list[ValidationIssueModel] = Field(default_factory=list)
    warnings: list[ValidationIssueModel] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    duplicate_count: int = 0
    empty_count: int = 0
    tiny_count: int = 0  # suspiciously short

    @property
    def passed(self) -> bool:
        return self.invalid == 0

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python")

    def summary(self) -> str:
        return (
            f"Validation {self.path}: {self.valid}/{self.total} valid, "
            f"{self.invalid} invalid, {len(self.warnings)} warnings, "
            f"{self.duplicate_count} duplicates, {self.empty_count} empty, {self.tiny_count} tiny"
        )


TINY_THRESHOLD = 10  # characters


def _hash_example(obj: dict[str, Any]) -> str:
    canonical = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def validate_file(
    path: str | Path,
    format: Optional[str] = None,
    tiny_threshold: int = TINY_THRESHOLD,
    check_duplicates: bool = True,
) -> ValidationReport:
    """Validate a JSONL dataset file.

    Args:
        path: path to .jsonl file (one JSON object per line)
        format: force format (raw|prompt_completion|conversational|story|auto)
        tiny_threshold: characters below which example is 'tiny' warning
        check_duplicates: detect duplicate examples via hash
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    lines = p.read_text(encoding="utf-8").splitlines()
    # filter empty lines but count them
    raw_lines = lines
    # allow JSON array file as well?
    is_json_array = False
    text = p.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                is_json_array = True
                objects = arr
                total = len(arr)
            else:
                objects = []
                total = 0
        except json.JSONDecodeError:
            objects = []
            total = 0
    else:
        objects = []
        for idx, line in enumerate(raw_lines, 1):
            stripped = line.strip()
            if stripped == "":
                continue
            try:
                obj = json.loads(stripped)
                objects.append((idx, obj))
            except json.JSONDecodeError as e:
                # will be counted as invalid
                objects.append((idx, {"__json_error__": str(e), "__raw__": stripped}))

        if objects and isinstance(objects[0], tuple):
            total = len(objects)
        else:
            total = 0

    # Normalize to list of (lineno, obj)
    if is_json_array:
        enumerated = list(enumerate(objects, 1))
        # objects is list[dict]
        items: list[tuple[int, Any]] = [(i, o) for i, o in enumerated]
        total = len(items)
    else:
        items = objects  # type: ignore

    if total == 0 and not is_json_array:
        # maybe file was empty or all blank
        return ValidationReport(
            path=str(p),
            format=format or "unknown",
            total=0,
            valid=0,
            invalid=0,
            stats={"note": "empty file or no parseable lines"},
        )

    fmt = format or "auto"
    errors: list[ValidationIssueModel] = []
    warnings: list[ValidationIssueModel] = []
    valid = 0
    invalid = 0
    empty_count = 0
    tiny_count = 0
    seen_hashes: Counter[str] = Counter()
    hash_to_lines: dict[str, list[int]] = {}

    # For contamination check: collect hashes to compare train/eval later (caller responsibility)
    token_lengths: list[int] = []
    char_lengths: list[int] = []

    for lineno, obj in items:
        if isinstance(obj, dict) and "__json_error__" in obj:
            errors.append(ValidationIssueModel(level="error", code="json_parse_error", message=obj["__json_error__"], line=lineno))
            invalid += 1
            continue

        if not isinstance(obj, dict):
            errors.append(ValidationIssueModel(level="error", code="not_an_object", message=f"Line is not a JSON object: {type(obj)}", line=lineno))
            invalid += 1
            continue

        # Empty check
        if len(obj) == 0:
            empty_count += 1
            warnings.append(ValidationIssueModel(level="warning", code="empty_object", message="Empty JSON object", line=lineno))
            invalid += 1
            continue

        # Detect format if auto
        eff_format = fmt if fmt != "auto" else detect_format(obj)
        if eff_format == "unknown":
            warnings.append(ValidationIssueModel(level="warning", code="unknown_format", message=f"Could not detect format for keys {list(obj.keys())}", line=lineno))

        # Validate against schema
        try:
            if eff_format == "raw":
                RawTextExample.model_validate(obj)
                text_val = obj.get("text", "")
            elif eff_format == "prompt_completion":
                PromptCompletionExample.model_validate(obj)
                text_val = (obj.get("prompt", "") + obj.get("completion", ""))
            elif eff_format == "conversational":
                ConversationalExample.model_validate(obj)
                # check malformed conversations: consecutive same role, empty content already validated
                msgs = obj.get("messages", [])
                for i in range(1, len(msgs)):
                    if msgs[i].get("role") == msgs[i-1].get("role") and msgs[i].get("role") != "system":
                        warnings.append(ValidationIssueModel(level="warning", code="consecutive_same_role", message=f"Consecutive {msgs[i].get('role')} messages at index {i}", line=lineno))
                # malformed: empty content strings
                for mi, m in enumerate(msgs):
                    if isinstance(m.get("content"), str) and m["content"].strip() == "":
                        errors.append(ValidationIssueModel(level="error", code="empty_message_content", message=f"Message {mi} has empty content", line=lineno))
                        raise ValueError("empty content")
                text_val = " ".join(m.get("content", "") for m in msgs)
            elif eff_format == "story":
                StoryExample.model_validate(obj)
                text_val = obj.get("response", "") or obj.get("completion", "") or ""
                # impossible metadata check
                meta = obj.get("metadata")
                if isinstance(meta, dict) and "word_count" in meta:
                    try:
                        wc = int(meta["word_count"])
                        actual = len(text_val.split())
                        if abs(wc - actual) > max(10, actual * 0.5):
                            warnings.append(ValidationIssueModel(level="warning", code="metadata_word_count_mismatch", message=f"metadata word_count {wc} vs actual {actual}", line=lineno))
                    except Exception:
                        pass
            else:
                text_val = json.dumps(obj, ensure_ascii=False)
                # try generic validation - at least check not empty
                if not text_val.strip():
                    raise ValueError("empty")

            # tiny check
            clen = len(text_val.strip()) if isinstance(text_val, str) else 0
            char_lengths.append(clen)
            # approximate tokens as ~4 chars per token
            token_lengths.append(max(1, clen // 4))
            if clen < tiny_threshold and clen > 0:
                tiny_count += 1
                warnings.append(ValidationIssueModel(level="warning", code="tiny_example", message=f"Example only {clen} chars (< {tiny_threshold})", line=lineno))
            if clen == 0:
                empty_count += 1
                warnings.append(ValidationIssueModel(level="warning", code="empty_text", message="Example has empty text/content", line=lineno))

            # duplicate detection
            if check_duplicates:
                h = _hash_example(obj)
                seen_hashes[h] += 1
                hash_to_lines.setdefault(h, []).append(lineno)

            valid += 1

        except (ValidationError, ValueError) as e:
            # pydantic error
            if isinstance(e, ValidationError):
                for err in e.errors():
                    loc = ".".join(str(x) for x in err["loc"])
                    errors.append(ValidationIssueModel(level="error", code="schema_error", message=f"{loc}: {err['msg']}", line=lineno, field=loc))
            else:
                errors.append(ValidationIssueModel(level="error", code="validation_error", message=str(e), line=lineno))
            invalid += 1
        except Exception as e:
            errors.append(ValidationIssueModel(level="error", code="unexpected_error", message=str(e), line=lineno))
            invalid += 1

    duplicate_count = sum(1 for h, c in seen_hashes.items() if c > 1)
    if duplicate_count > 0:
        for h, c in seen_hashes.items():
            if c > 1:
                warnings.append(ValidationIssueModel(level="warning", code="duplicate", message=f"Duplicate example appears {c} times at lines {hash_to_lines[h]}"))

    # stats
    stats: dict[str, Any] = {}
    if char_lengths:
        stats["char_length"] = {
            "mean": sum(char_lengths) / len(char_lengths),
            "min": min(char_lengths),
            "max": max(char_lengths),
        }
    if token_lengths:
        stats["approx_tokens"] = {
            "mean": sum(token_lengths) / len(token_lengths),
            "min": min(token_lengths),
            "max": max(token_lengths),
            "total_approx": sum(token_lengths),
        }
    stats["format_detected"] = fmt if fmt != "auto" else "auto"
    stats["duplicate_groups"] = duplicate_count

    report = ValidationReport(
        path=str(p),
        format=fmt,
        total=total,
        valid=valid,
        invalid=invalid,
        errors=errors,
        warnings=warnings,
        stats=stats,
        duplicate_count=duplicate_count,
        empty_count=empty_count,
        tiny_count=tiny_count,
    )
    return report


def check_contamination(train_hashes: set[str], eval_hashes: set[str]) -> list[str]:
    """Return list of hashes present in both train and eval (data leakage)."""
    return sorted(train_hashes & eval_hashes)


def file_hashes(path: str | Path) -> set[str]:
    p = Path(path)
    hashes: set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            hashes.add(_hash_example(obj))
        except Exception:
            continue
    return hashes
