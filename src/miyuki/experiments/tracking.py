"""Lightweight experiment tracking - prevents 'I think that version was better'."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _experiments_dir() -> Path:
    # Try project-root relative
    candidates = [Path("experiments"), Path("C:/Aether/RisingStar/MIYUKI/experiments"), Path(__file__).parents[3] / "experiments"]
    for c in candidates:
        if c.exists() or c.parent.exists():
            return c if c.is_absolute() else Path("C:/Aether/RisingStar/MIYUKI") / c
    p = Path("experiments")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _next_id() -> str:
    exp_dir = _experiments_dir()
    exp_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted([d.name for d in exp_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    if not existing:
        return "0001"
    return f"{int(existing[-1]) + 1:04d}"


def record_experiment(
    model_name: str,
    dataset: str,
    dataset_version: str,
    training_config: dict[str, Any],
    hardware: dict[str, Any],
    duration_s: float,
    final_loss: Optional[float] = None,
    metrics: Optional[dict[str, Any]] = None,
    artifact_location: Optional[str] = None,
    notes: Optional[str] = None,
    evaluation_results: Optional[dict[str, Any]] = None,
) -> str:
    exp_id = _next_id()
    exp_dir = _experiments_dir() / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "id": exp_id,
        "created": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "dataset": dataset,
        "dataset_version": dataset_version,
        "training_config": training_config,
        "hardware": hardware,
        "duration_s": duration_s,
        "final_loss": final_loss,
        "metrics": metrics or {},
        "evaluation_results": evaluation_results,
        "artifact_location": artifact_location,
        "notes": notes,
        "status": "completed",
    }
    (exp_dir / "metrics.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    (exp_dir / "config.yaml").write_text(
        __import__("yaml").safe_dump(training_config, sort_keys=False), encoding="utf-8"
    )
    if evaluation_results:
        (exp_dir / "evaluation.json").write_text(json.dumps(evaluation_results, indent=2, ensure_ascii=False), encoding="utf-8")
    (exp_dir / "README.md").write_text(
        f"# Experiment {exp_id}\n\n- Model: {model_name}\n- Dataset: {dataset} ({dataset_version})\n- Duration: {duration_s:.1f}s\n- Final loss: {final_loss}\n- Artifact: {artifact_location}\n- Notes: {notes or ''}\n",
        encoding="utf-8",
    )
    return exp_id


def list_experiments() -> list[dict[str, Any]]:
    exp_dir = _experiments_dir()
    if not exp_dir.exists():
        return []
    results = []
    for d in sorted(exp_dir.iterdir()):
        if not d.is_dir():
            continue
        mf = d / "metrics.json"
        if mf.exists():
            try:
                data = json.loads(mf.read_text(encoding="utf-8"))
                results.append(data)
            except Exception:
                results.append({"id": d.name, "status": "corrupt"})
        else:
            results.append({"id": d.name, "status": "no metrics"})
    return results


def show_experiment(exp_id: str) -> dict[str, Any]:
    exp_dir = _experiments_dir() / exp_id
    mf = exp_dir / "metrics.json"
    if not mf.exists():
        raise FileNotFoundError(f"Experiment {exp_id} not found at {exp_dir}")
    return json.loads(mf.read_text(encoding="utf-8"))
