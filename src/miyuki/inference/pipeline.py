"""Baseline inference pipeline: load configurable base model, generate, save."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..config import MiyukiConfig
from ..runtime.transformers_runtime import TransformersRuntime


def run_inference(
    config: MiyukiConfig,
    prompt: str,
    **overrides,
) -> dict[str, Any]:
    """Run inference through the configured runtime and return a serializable record."""
    runtime = TransformersRuntime(config)
    result = runtime.generate(prompt, **overrides)
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt": result.prompt,
        "generated": result.generated,
        "full_text": result.full_text,
        "model": result.model_name,
        "config": result.config_snapshot,
        "runtime": result.runtime_info,
        "generation_params": result.generation_params,
        "timings": result.timings,
    }
    return record


def save_result(record: dict[str, Any], output_path: str | Path) -> Path:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def run_inference_and_maybe_save(
    config: MiyukiConfig,
    prompt: str,
    save: bool = False,
    output: Optional[str | Path] = None,
    **overrides,
) -> dict[str, Any]:
    record = run_inference(config, prompt, **overrides)
    if save:
        if output is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output = Path("outputs") / f"inference_{ts}.json"
        save_result(record, output)
        print(f"[info] Saved inference to {output}")
    return record
