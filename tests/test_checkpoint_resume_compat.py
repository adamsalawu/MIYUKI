"""Regression test: trusted local checkpoint resume under torch>=2.6 weights_only default.

Covers:
  - rng_state.pth (existing checkpoint-50 format) loads with the narrow numpy allowlist
  - the allowlist does not reject arbitrary-code-execution globals (security preserved)
  - the compat helper is narrowly scoped (only invoked on resume path)
  - normal safetensors loading is unaffected

NOTE (2026-09-30 migration): originally pointed at checkpoint-30, which the
trainer's own save_total_limit=2 later pruned (only 45/50 survive). Repointed at
checkpoint-50 — same run, identical rng_state.pth format (14,455 bytes). No
assertion was weakened; step expectation updated 30 -> 50 to match.
"""
import json
import pathlib
import re

import torch

CKPT50 = pathlib.Path("C:/Aether/RisingStar/MIYUKI/checkpoints/phase4c-miyuki-0.3/checkpoint-50")
RNG = CKPT50 / "rng_state.pth"


def _load_default():
    """Load with torch's current default (weights_only=True on torch>=2.6)."""
    return torch.load(RNG)


def test_checkpoint50_rng_state_exists_and_is_trusted_local():
    assert RNG.exists(), "checkpoint-50 rng_state.pth missing"
    cfg = json.loads((CKPT50 / "adapter_config.json").read_text(encoding="utf-8"))
    # locally generated: base model is the project's own training target, not third-party weights
    assert cfg["base_model_name_or_path"] == "Qwen/Qwen2-0.5B-Instruct"
    state = json.loads((CKPT50 / "trainer_state.json").read_text(encoding="utf-8"))
    assert state["global_step"] == 50 and state["max_steps"] == 51


def test_rng_state_fails_without_allowlist():
    """Documents the failure the compat fix addresses (torch>=2.6 weights_only default)."""
    try:
        _load_default()
        # On torch < 2.6 the default is weights_only=False and this load succeeds;
        # the compat fix is then a no-op but must still not break anything.
        print("torch default loaded without allowlist (torch<2.6 behavior) - fix is a no-op here")
    except Exception as e:
        msg = str(e)
        assert "Weights only load failed" in msg or "WeightsUnpickler" in msg
        assert "numpy" in msg, "failure should be the numpy RNG-state global rejection"


def test_rng_state_loads_with_narrow_allowlist():
    """The exact regression: existing checkpoint format resumes with the narrow allowlist."""
    from miyuki.training.train import _enable_local_checkpoint_rng_compat

    _enable_local_checkpoint_rng_compat()
    state = _load_default()
    assert isinstance(state, dict)
    assert set(state.keys()) == {"python", "numpy", "cpu"}
    # numpy RNG state survived as a real RandomState tuple (not stripped/corrupted)
    import numpy as np

    np_state = state["numpy"]
    assert isinstance(np_state, tuple) and np_state[0] == "MT19937"
    keys = np_state[1]
    assert isinstance(keys, np.ndarray) and keys.shape == (624,)
    assert isinstance(state["cpu"], torch.Tensor)


def test_allowlist_does_not_weaken_security():
    """Arbitrary code execution globals must still be rejected after the compat fix."""
    from miyuki.training.train import _enable_local_checkpoint_rng_compat

    _enable_local_checkpoint_rng_compat()

    # craft a malicious pickle that would execute os.system if unpickled
    import io
    import pickle

    class Exploit:
        def __reduce__(self):
            import os

            return (os.system, ("echo should_never_run",))

    buf = io.BytesIO()
    pickle.dump(Exploit(), buf)
    buf.seek(0)
    try:
        torch.load(buf, weights_only=True)
        raise AssertionError("malicious pickle must be rejected even with compat allowlist")
    except Exception as e:
        msg = str(e)
        assert "os.system" in msg or "Weights only" in msg or "WeightsUnpickler" in msg


def test_compat_helper_only_called_on_resume_path():
    """Narrow scoping: grep the training implementation to confirm the helper is invoked
    only where resume_from_checkpoint is set."""
    src = pathlib.Path("C:/Aether/RisingStar/MIYUKI/src/miyuki/training/train.py").read_text(encoding="utf-8")
    # match call sites only (exclude the def line, which ends with " -> None:")
    calls = [m for m in re.finditer(r"_enable_local_checkpoint_rng_compat\(\)(?! ->)", src)]
    assert len(calls) == 1, "compat helper must have exactly one call site"
    # the call site must be guarded by the resume condition
    call_line_start = src.rfind("\n", 0, calls[0].start()) + 1
    context = src[max(0, call_line_start - 400) : calls[0].start()]
    assert "resume_from_checkpoint" in context


def test_normal_safetensors_loading_unaffected():
    """Normal checkpoint loading (safetensors) still works after the compat fix."""
    from miyuki.training.train import _enable_local_checkpoint_rng_compat

    _enable_local_checkpoint_rng_compat()
    from safetensors.torch import load_file

    weights = load_file(str(CKPT50 / "adapter_model.safetensors"))
    assert len(weights) > 0
    assert any("lora" in k.lower() for k in weights)
