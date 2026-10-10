"""Model artifact management - standard directory format."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def save_model_metadata(
    model_dir: str | Path,
    base_model: str,
    version: str,
    dataset: str,
    dataset_version: str,
    training_config: dict[str, Any],
    tokenizer_id: str,
    context_length: int,
    quantization: Optional[str] = None,
    evaluation_results: Optional[dict[str, Any]] = None,
) -> Path:
    p = Path(model_dir)
    p.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_name": p.name,
        "version": version,
        "base_model": base_model,
        "training_dataset": dataset,
        "dataset_version": dataset_version,
        "training_config": training_config,
        "tokenizer": tokenizer_id,
        "context_length": context_length,
        "quantization": quantization,
        "creation_date": datetime.now(timezone.utc).isoformat(),
        "evaluation_results": evaluation_results,
    }
    out = p / "metadata.json"
    out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    # README
    readme = p / "README.md"
    if not readme.exists():
        readme.write_text(
            f"# {p.name}\n\n- Base: {base_model}\n- Version: {version}\n- Dataset: {dataset} ({dataset_version})\n- Tokenizer: {tokenizer_id}\n- Context: {context_length}\n- Quantization: {quantization}\n- Created: {metadata['creation_date']}\n\nSee metadata.json for full details.\n",
            encoding="utf-8",
        )
    return out


def load_model_metadata(model_dir: str | Path) -> dict[str, Any]:
    p = Path(model_dir)
    mf = p / "metadata.json"
    if not mf.exists():
        raise FileNotFoundError(f"metadata.json not found in {p}")
    return json.loads(mf.read_text(encoding="utf-8"))


def discover_checkpoints(checkpoints_dir: str | Path = "checkpoints") -> list[Path]:
    base = Path(checkpoints_dir)
    if not base.is_absolute():
        base = Path("C:/Aether/RisingStar/MIYUKI") / base
    if not base.exists():
        return []
    # Each subdir with config or pytorch_model.bin / adapter
    candidates = []
    for sub in base.iterdir():
        if sub.is_dir() and any((sub / f).exists() for f in ["config.json", "pytorch_model.bin", "model.safetensors", "adapter_config.json", "metadata.json"]):
            candidates.append(sub)
        # also nested checkpoint-*
        for nested in sub.glob("checkpoint-*"):
            if nested.is_dir():
                candidates.append(nested)
    return sorted(candidates)
