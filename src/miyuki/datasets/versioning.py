"""Dataset versioning - identity, manifests, token stats."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class DatasetManifest(BaseModel):
    name: str = Field(description="Dataset name, e.g. miyuki-writing-0.1")
    version: str
    source: Optional[str] = Field(default=None, description="Provenance: user-created, synthetic, public-domain, licensed, etc.")
    license: Optional[str] = Field(default=None)
    creation_date: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    num_examples: int = Field(ge=0)
    splits: dict[str, int] = Field(default_factory=dict, description="split name -> count")
    preprocessing: Optional[dict[str, Any]] = Field(default=None)
    token_stats: Optional[dict[str, Any]] = Field(default=None)
    file_hashes: dict[str, str] = Field(default_factory=dict, description="file path -> sha256")
    notes: Optional[str] = Field(default=None)

    def to_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.model_dump(mode="python"), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "DatasetManifest":
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.model_validate(data)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_manifest(
    name: str,
    version: str,
    files: dict[str, Path | str],
    source: Optional[str] = None,
    license: Optional[str] = None,
    preprocessing: Optional[dict[str, Any]] = None,
    notes: Optional[str] = None,
) -> DatasetManifest:
    """Compute manifest for a set of dataset files.

    Args:
        name: dataset name
        version: version string
        files: mapping split_name -> file path (or single file)
        source: provenance string
        license: license identifier
        preprocessing: preprocessing config dict
        notes: freeform notes
    """
    splits: dict[str, int] = {}
    file_hashes: dict[str, str] = {}
    total = 0
    token_stats: dict[str, Any] = {"approx_tokens_total": 0}

    for split_name, fpath in files.items():
        p = Path(fpath)
        if not p.exists():
            continue
        # count lines / json array
        text = p.read_text(encoding="utf-8").strip()
        count = 0
        approx_tokens = 0
        if text.startswith("["):
            try:
                arr = json.loads(text)
                count = len(arr) if isinstance(arr, list) else 0
                for obj in arr:
                    s = json.dumps(obj, ensure_ascii=False)
                    approx_tokens += max(1, len(s) // 4)
            except Exception:
                count = 0
        else:
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                count += 1
                approx_tokens += max(1, len(line) // 4)
        splits[split_name] = count
        total += count
        token_stats["approx_tokens_total"] += approx_tokens
        token_stats[f"approx_tokens_{split_name}"] = approx_tokens
        file_hashes[str(p)] = sha256_file(p)

    return DatasetManifest(
        name=name,
        version=version,
        source=source,
        license=license,
        num_examples=total,
        splits=splits,
        preprocessing=preprocessing,
        token_stats=token_stats,
        file_hashes=file_hashes,
        notes=notes,
    )
