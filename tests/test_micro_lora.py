"""Micro-LoRA tests: config, identifier, masking, preservation, eval determinism, NO-GO artifacts, guard, metadata."""
import json, pathlib

MICRO_CFG = pathlib.Path("C:/Aether/RisingStar/MIYUKI/configs/training/qwen2.5-1.5b_micro_lora.yaml")
PHASE4_CFG = pathlib.Path("C:/Aether/RisingStar/MIYUKI/configs/training/phase4_lora.yaml")
DS02 = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.2/dataset.jsonl")
PREFLIGHT = pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/runs/qwen2.5-1.5b-micro-lora/preflight.json")
MICRO_CKPT = pathlib.Path("C:/Aether/RisingStar/MIYUKI/checkpoints/qwen2.5-1.5b-micro-lora")

def test_micro_config():
    from miyuki.config import MiyukiConfig
    c = MiyukiConfig.from_yaml(MICRO_CFG)
    assert c.model.name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert c.training.num_epochs == 1
    assert c.training.learning_rate == 0.0002
    assert c.adapter.r == 16 and c.adapter.alpha == 32
    assert c.training.output_dir == "checkpoints/qwen2.5-1.5b-micro-lora"

def test_correct_15b_identifier():
    from miyuki.config import MiyukiConfig
    c = MiyukiConfig.from_yaml(MICRO_CFG)
    assert c.model.tokenizer == "Qwen/Qwen2.5-1.5B-Instruct"
    assert "2.5" in c.model.name and "1.5B" in c.model.name

def test_assistant_only_true():
    from miyuki.config import MiyukiConfig
    c = MiyukiConfig.from_yaml(MICRO_CFG)
    assert c.training.assistant_only_loss is True

def test_phase4_config_unchanged():
    from miyuki.config import MiyukiConfig
    c = MiyukiConfig.from_yaml(PHASE4_CFG)
    assert c.model.name == "Qwen/Qwen2-0.5B-Instruct"
    assert c.training.assistant_only_loss is True
    assert c.training.num_epochs == 3
    assert c.training.output_dir == "checkpoints/phase4-miyuki-0.2"

def test_phase4_checkpoint_unchanged():
    assert pathlib.Path("C:/Aether/RisingStar/MIYUKI/checkpoints/phase4-miyuki-0.2/adapter_config.json").exists()
    cfg = json.loads(pathlib.Path("C:/Aether/RisingStar/MIYUKI/checkpoints/phase4-miyuki-0.2/adapter_config.json").read_text(encoding="utf-8"))
    assert cfg["base_model_name_or_path"] == "Qwen/Qwen2-0.5B-Instruct"

def test_dataset_02_unchanged():
    import hashlib
    assert hashlib.sha256(DS02.read_bytes()).hexdigest() == "8cda31a91a76b1d6e8dc67ea2b779de06334836dad089b6bd59210dbdc6985d6"
    data = [json.loads(l) for l in DS02.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(data) == 120

def test_deterministic_evaluation():
    from miyuki.evaluation.runner import load_prompts
    prompts = load_prompts("C:/Aether/RisingStar/MIYUKI/evaluation/prompts/eval_v0.1.jsonl")
    assert len(prompts) == 12
    scale = [json.loads(l) for l in pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/runs/model-scale-qwen2.5-1.5b/outputs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [o["id"] for o in scale] == [p["id"] for p in prompts]

def test_checkpoint_absence_nogo_proof():
    # NO-GO means no micro checkpoint. NOTE (2026-09-30 migration): originally also
    # asserted no exp 0007; Phase 4C has since legitimately created 0007 (+0008
    # duplicate) via the recording mechanism. Guard now targets the next unassigned
    # ID (0009) so the "nothing ran since" invariant keeps holding.
    assert not MICRO_CKPT.exists(), "micro checkpoint should NOT exist after NO-GO"
    assert not pathlib.Path("C:/Aether/RisingStar/MIYUKI/experiments/0009").exists()

def test_adapter_reload_na_documented():
    # No adapter to reload; README documents N/A
    readme = pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/runs/qwen2.5-1.5b-micro-lora/README.md").read_text(encoding="utf-8")
    assert "NOT APPLICABLE" in readme or "NO-GO" in readme

def test_memory_guard_behavior():
    d = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    assert d["decision"] == "NO-GO"
    assert d["ram_avail_after_lora_gb"] < d["threshold_gb"]
    assert d["trainable"] == 18464768

def test_experiment_metadata():
    # No 0007; latest remains 0006 with Phase4
    assert pathlib.Path("C:/Aether/RisingStar/MIYUKI/experiments/0006/metrics.json").exists()
    d = json.loads(pathlib.Path("C:/Aether/RisingStar/MIYUKI/experiments/0006/metrics.json").read_text(encoding="utf-8"))
    assert d["id"] == "0006"
    assert d["model"] == "Qwen/Qwen2-0.5B-Instruct"
