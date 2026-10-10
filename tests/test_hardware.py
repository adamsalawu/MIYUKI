"""Tests: hardware detection."""
from miyuki.hardware import detect_hardware, recommend_device


def test_hardware_report_has_keys():
    info = detect_hardware()
    assert "cpu" in info
    assert "ram_total_gb" in info
    assert "python_version" in info
    assert "recommended_device" in info
    assert info["recommended_device"] in ("cpu", "cuda", "mps")
    assert "recommended_dtype" in info


def test_recommend_device_logic():
    assert recommend_device({"cuda_available": True, "gpu_count": 1}) == "cuda"
    assert recommend_device({"cuda_available": False, "mps_available": True}) == "mps"
    assert recommend_device({"cuda_available": False, "mps_available": False}) == "cpu"
