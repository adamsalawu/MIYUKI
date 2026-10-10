"""Automated evaluation runner - runs every prompt through a model and saves outputs."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..config import MiyukiConfig
from ..inference.pipeline import run_inference


def load_prompts(prompts_file: str | Path) -> list[dict[str, Any]]:
    p = Path(prompts_file)
    if not p.exists():
        raise FileNotFoundError(f"Prompts file not found: {p}")
    prompts: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        prompts.append(json.loads(line))
    return prompts


def run_evaluation(
    model_config: MiyukiConfig,
    prompts_file: str | Path | None = None,
    output_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
    generation_overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run all evaluation prompts through the model.

    Saves:
      <output_dir>/<date>_<model_sanitized>/outputs.jsonl
      <output_dir>/<date>_<model_sanitized>/metadata.json
      <report_dir>/<date>_<model_sanitized>.json  (summary report)

    Optimized: reuses a single TransformersRuntime across all prompts
    (avoids reloading the model 12 times — critical for 0.5B on CPU).
    """
    prompts_file = Path(prompts_file) if prompts_file else Path("evaluation/prompts/eval_v0.1.jsonl")
    # Resolve relative to project root if needed
    candidates = [prompts_file, Path("C:/Aether/RisingStar/MIYUKI") / prompts_file, Path(__file__).parents[3] / prompts_file]
    for c in candidates:
        if c.exists():
            prompts_file = c
            break

    prompts = load_prompts(prompts_file)

    # Prepare run directory
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    model_slug = model_config.model.name.replace("/", "_").replace(":", "_")[:40]
    run_name = f"{date_str}_{model_slug}"

    base_out = Path(output_dir) if output_dir else Path("evaluation/runs")
    # Try to resolve relative to project root
    if not base_out.is_absolute() and not base_out.exists():
        base_out = Path("C:/Aether/RisingStar/MIYUKI") / base_out
    run_dir = base_out / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    base_report = Path(report_dir) if report_dir else Path("evaluation/reports")
    if not base_report.is_absolute() and not base_report.exists():
        base_report = Path("C:/Aether/RisingStar/MIYUKI") / base_report
    base_report.mkdir(parents=True, exist_ok=True)

    outputs_path = run_dir / "outputs.jsonl"
    metadata_path = run_dir / "metadata.json"
    report_path = base_report / f"{run_name}.json"

    generation_overrides = generation_overrides or {}

    # Single runtime instance — load once
    from ..runtime.transformers_runtime import TransformersRuntime

    t_load0 = time.time()
    runtime = TransformersRuntime(model_config)
    try:
        runtime.load()
        load_s = time.time() - t_load0
        load_error = None
        # Capture model metadata after load
        try:
            model_params = sum(p.numel() for p in runtime._model.parameters()) if runtime._model is not None else None  # type: ignore
        except Exception:
            model_params = None
    except Exception as e:
        load_s = time.time() - t_load0
        load_error = str(e)
        model_params = None

    results: list[dict[str, Any]] = []
    t_start = time.time()

    for item in prompts:
        prompt = item.get("prompt", "")
        # Compose full prompt: optionally prepend instruction
        instruction = item.get("instruction", "")
        full_prompt = prompt
        if instruction:
            # Keep it simple: instruction as prefix
            full_prompt = f"{instruction}\n\n{prompt}"
        # Use context if present (prepend)
        context = item.get("context", "")
        if context:
            full_prompt = f"Context: {context}\n\n{full_prompt}"

        if load_error is not None:
            entry = {
                "id": item.get("id"),
                "category": item.get("category"),
                "prompt": prompt,
                "instruction": instruction,
                "context": context,
                "full_prompt": full_prompt,
                "generated": None,
                "full_text": None,
                "model": model_config.model.name,
                "generation_params": generation_overrides,
                "timings": {},
                "runtime": None,
                "error": f"model load failed: {load_error}",
            }
            results.append(entry)
            continue

        try:
            result = runtime.generate(full_prompt, **generation_overrides)
            entry = {
                "id": item.get("id"),
                "category": item.get("category"),
                "prompt": prompt,
                "instruction": instruction,
                "context": context,
                "full_prompt": full_prompt,
                "generated": result.generated,
                "full_text": result.full_text,
                "model": result.model_name,
                "generation_params": result.generation_params,
                "timings": result.timings,
                "runtime": result.runtime_info,
                "error": None,
            }
        except Exception as e:
            entry = {
                "id": item.get("id"),
                "category": item.get("category"),
                "prompt": prompt,
                "instruction": instruction,
                "context": context,
                "full_prompt": full_prompt,
                "generated": None,
                "full_text": None,
                "model": model_config.model.name,
                "generation_params": generation_overrides,
                "timings": {},
                "runtime": None,
                "error": str(e),
            }
        results.append(entry)

    # Save outputs.jsonl
    with open(outputs_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    elapsed = time.time() - t_start

    metadata = {
        "run_name": run_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_config.to_dict(),
        "prompts_file": str(prompts_file),
        "num_prompts": len(prompts),
        "num_outputs": len(results),
        "num_errors": sum(1 for r in results if r["error"] is not None),
        "generation_overrides": generation_overrides,
        "elapsed_s": elapsed,
        "load_s": load_s,
        "load_error": load_error,
        "model_params": model_params,
        "run_dir": str(run_dir),
        "outputs_path": str(outputs_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary report (lightweight, human-reviewable)
    report = {
        "run_name": run_name,
        "timestamp": metadata["timestamp"],
        "model": model_config.model.name,
        "model_version": model_config.version,
        "num_prompts": len(prompts),
        "num_errors": metadata["num_errors"],
        "elapsed_s": elapsed,
        "load_s": load_s,
        "categories": sorted({r["category"] for r in results if r.get("category")}),
        "prompt_ids": [r["id"] for r in results],
        "outputs_path": str(outputs_path),
        "metadata_path": str(metadata_path),
        "notes": "Automated generation complete. Fill human_template.yaml for qualitative ratings.",
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "outputs_path": str(outputs_path),
        "num_prompts": len(prompts),
        "num_outputs": len(results),
        "report": report,
    }
