"""Hardware detection - reports CPU/RAM/GPU/VRAM/CUDA and recommends device."""
from __future__ import annotations

import platform
import sys
from typing import Any, Optional

import psutil  # type: ignore - optional, fallback if missing
from pathlib import Path


def _safe_import_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def detect_hardware() -> dict[str, Any]:
    info: dict[str, Any] = {}
    info["platform"] = platform.platform()
    info["python_version"] = sys.version
    info["python_executable"] = sys.executable

    # CPU
    info["cpu"] = platform.processor() or platform.machine()
    try:
        import cpuinfo  # type: ignore
        ci = cpuinfo.get_cpu_info()
        info["cpu_brand"] = ci.get("brand_raw", info["cpu"])
    except Exception:
        info["cpu_brand"] = info["cpu"]
    try:
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
    except Exception:
        import os
        info["cpu_count_logical"] = os.cpu_count()

    # RAM
    try:
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / (1024**3), 2)
        info["ram_available_gb"] = round(vm.available / (1024**3), 2)
    except Exception:
        info["ram_total_gb"] = None
        info["ram_available_gb"] = None

    # Torch / CUDA
    torch = _safe_import_torch()
    if torch is None:
        info["torch_version"] = None
        info["torch_available"] = False
        info["cuda_available"] = False
        info["cuda_version"] = None
        info["gpu_name"] = None
        info["gpu_vram_gb"] = None
        info["gpu_count"] = 0
    else:
        info["torch_version"] = torch.__version__
        info["torch_available"] = True
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = getattr(torch.version, "cuda", None) if info["cuda_available"] else None
        info["gpu_count"] = torch.cuda.device_count() if info["cuda_available"] else 0
        if info["cuda_available"] and info["gpu_count"] > 0:
            try:
                props = torch.cuda.get_device_properties(0)
                info["gpu_name"] = props.name
                info["gpu_vram_gb"] = round(props.total_memory / (1024**3), 2)
            except Exception as e:
                info["gpu_name"] = str(e)
                info["gpu_vram_gb"] = None
        else:
            # Try to detect GPU via nvidia-smi fallback even if torch says no cuda?
            info["gpu_name"] = _detect_gpu_nvidia_smi()
            info["gpu_vram_gb"] = None

        # MPS (Apple)
        try:
            info["mps_available"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        except Exception:
            info["mps_available"] = False

    # Recommendation
    info["recommended_device"] = recommend_device(info)
    info["recommended_dtype"] = recommend_dtype(info)
    info["notes"] = _notes(info)
    return info


def _detect_gpu_nvidia_smi() -> Optional[str]:
    import subprocess, shutil
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            timeout=5,
            text=True,
        )
        return out.strip().splitlines()[0] if out.strip() else None
    except Exception:
        return None


def recommend_device(info: dict[str, Any]) -> str:
    if info.get("cuda_available") and info.get("gpu_count", 0) > 0:
        # Check VRAM - if < 4GB, still recommend cuda but warn
        return "cuda"
    if info.get("mps_available"):
        return "mps"
    return "cpu"


def recommend_dtype(info: dict[str, Any]) -> str:
    dev = info.get("recommended_device")
    if dev == "cuda":
        # Prefer bf16 if GPU supports it, else fp16, else float32
        vram = info.get("gpu_vram_gb")
        if vram is not None and vram < 6:
            return "float16"  # conserve VRAM
        return "bfloat16" if vram and vram >= 8 else "float16"
    if dev == "mps":
        return "float16"
    return "float32"


def _notes(info: dict[str, Any]) -> list[str]:
    notes = []
    if not info.get("cuda_available"):
        notes.append("CUDA not available - using CPU. Training will be slow; use small models and LoRA for smoke tests.")
    vram = info.get("gpu_vram_gb")
    if vram is not None and vram < 4:
        notes.append(f"Low VRAM ({vram} GB) detected - use quantization (4bit/8bit) or small models (0.5B) with LoRA.")
    elif vram is not None and vram < 8:
        notes.append(f"Moderate VRAM ({vram} GB) - suitable for 0.5B-1B models with LoRA; consider gradient checkpointing.")
    ram = info.get("ram_total_gb")
    if ram is not None and ram < 8:
        notes.append(f"Low RAM ({ram} GB) - keep batch size 1 and use gradient accumulation.")
    return notes


def print_report(info: Optional[dict[str, Any]] = None) -> None:
    if info is None:
        info = detect_hardware()
    print("=" * 60)
    print("MIYUKI HARDWARE REPORT")
    print("=" * 60)
    for k, v in info.items():
        if k == "notes":
            continue
        print(f"{k:24s}: {v}")
    if info.get("notes"):
        print("\nNotes:")
        for n in info["notes"]:
            print(f"  - {n}")
    print("=" * 60)


if __name__ == "__main__":
    print_report()
