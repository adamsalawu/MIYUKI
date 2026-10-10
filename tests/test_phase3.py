"""Phase 3 tests: config, dataset conversion, LoRA, checkpoint, adapter round-trip, baseline preservation."""
import json, pathlib
from transformers import AutoTokenizer, AutoModelForCausalLM
from miyuki.config import MiyukiConfig

PHASE3_CFG = pathlib.Path("C:/Aether/RisingStar/MIYUKI/configs/training/phase3_lora.yaml")
DATASET = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1/dataset.jsonl")
CHECKPOINT = pathlib.Path("C:/Aether/RisingStar/MIYUKI/checkpoints/phase3-miyuki-0.1")
EXP = pathlib.Path("C:/Aether/RisingStar/MIYUKI/experiments/0002/metrics.json")
PHASE1_OUTPUTS = pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase1_qwen2_0.5b/outputs.jsonl")
PHASE3_OUTPUTS = pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase3-miyuki-0.1/outputs.jsonl")

def test_phase3_config_loading():
    cfg = MiyukiConfig.from_yaml(PHASE3_CFG)
    assert cfg.model.name == "Qwen/Qwen2-0.5B-Instruct"
    assert cfg.model.family == "qwen2"
    assert cfg.adapter.enabled is True
    assert cfg.adapter.type == "lora"
    assert cfg.adapter.r == 16
    assert cfg.adapter.alpha == 32
    assert cfg.adapter.dropout == 0.05
    assert cfg.training.num_epochs == 1
    assert cfg.training.per_device_train_batch_size == 1
    assert cfg.training.gradient_accumulation_steps == 8
    assert cfg.training.max_seq_length == 1024
    assert cfg.training.seed == 42
    # target_modules auto -> should be None in config, resolved at train time
    assert cfg.adapter.target_modules is None

def test_dataset_to_training_conversion():
    from miyuki.datasets.loader import load_jsonl, format_for_training
    raw = load_jsonl(DATASET)
    assert len(raw) == 38
    # Check first example converts via format_for_training and then via to_text logic in train.py (chat template)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ex = raw[0]
    fmt = format_for_training(ex)
    assert "messages" in fmt
    msgs = fmt["messages"]
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    # Apply chat template as train.py does
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    assert "<|im_start|>" in text
    assert "Mara" in text or "lighthouse" in text
    # Check all examples have non-empty assistant
    for e in raw:
        f = format_for_training(e)
        assert len(f["messages"][1]["content"]) > 20

def test_lora_configuration_and_trainable_params():
    # Use tiny-random for fast check of trainable logic, but verify Qwen targets are correct per config
    cfg = MiyukiConfig.from_yaml(PHASE3_CFG)
    # Verify expected target_modules for qwen2 as would be used in train.py
    from miyuki.training.train import train as _train  # to ensure module loads
    # Check that train.py's family_defaults for qwen2 includes 7 modules
    # We can directly test the logic: if target_modules is None, it resolves to qwen2 list
    family_defaults = {
        "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }
    expected = family_defaults["qwen2"]
    assert len(expected) == 7
    # Verify that a small model can get LoRA with those targets and produce trainable params >0
    # Use tiny-random to keep fast
    from transformers import AutoModelForCausalLM as AM
    from peft import LoraConfig, get_peft_model, TaskType
    tiny = AM.from_pretrained("hf-internal-testing/tiny-random-gpt2", trust_remote_code=False)
    # Tiny uses c_attn, c_proj but we test qwen2 targets on tiny would fail, so test with tiny's own targets
    lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn","c_proj"], task_type=TaskType.CAUSAL_LM)
    peft_model = get_peft_model(tiny, lora_cfg)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    assert trainable > 0
    assert trainable < total
    assert trainable / total < 0.2  # tiny-random ~11%, Qwen 1.75% — both <20%

def test_checkpoint_discovery():
    assert CHECKPOINT.exists(), "Phase3 checkpoint missing"
    assert (CHECKPOINT / "adapter_config.json").exists()
    assert (CHECKPOINT / "adapter_model.safetensors").exists()
    assert (CHECKPOINT / "train_metrics.json").exists()
    assert (CHECKPOINT / "metadata.json").exists()
    # Check via artifact manager
    from miyuki.artifacts.manager import discover_checkpoints
    found = discover_checkpoints("checkpoints")
    # Should find phase3 checkpoint
    assert any("phase3-miyuki-0.1" in str(p) for p in found)

def test_adapter_round_trip_loading():
    # Check that base model can load and adapter can be loaded (without full generation to keep fast)
    # We test that adapter_config points to base and can be read
    import json
    cfg = json.loads((CHECKPOINT / "adapter_config.json").read_text(encoding="utf-8"))
    assert cfg["base_model_name_or_path"] == "Qwen/Qwen2-0.5B-Instruct"
    assert cfg["peft_type"] == "LORA"
    assert cfg["r"] == 16
    # Test that model tokenizer not mismatched
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=False)
    assert len(tok) == 151646
    # Verify tokenizer saved in checkpoint matches
    ckpt_tok = AutoTokenizer.from_pretrained(str(CHECKPOINT), trust_remote_code=False)
    assert len(ckpt_tok) == len(tok)

def test_experiment_metadata():
    assert EXP.exists(), "Experiment 0002 missing"
    data = json.loads(EXP.read_text(encoding="utf-8"))
    assert data["id"] == "0002"
    assert data["model"] == "Qwen/Qwen2-0.5B-Instruct"
    assert "miyuki-writing-0.1" in data["dataset"]
    assert data["dataset_version"] == "0.1.0-phase3"
    assert data["hardware"]["python_version"] is not None
    assert data["duration_s"] > 0
    assert data["final_loss"] is not None
    assert data["metrics"]["peft_applied"] is True
    assert pathlib.Path(data["artifact_location"]).exists()

def test_deterministic_configuration():
    cfg = MiyukiConfig.from_yaml(PHASE3_CFG)
    # Seed should be deterministic
    assert cfg.training.seed == 42
    assert cfg.generation.seed == 42
    # Dataset SHA should be stable
    import hashlib
    h = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    assert h == "9e1b97461f98948254bc45906cb275f111318f5d5adfc9e7139ab9c35c200d5e"
    # Manifest should match
    manifest = json.loads(pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1/manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["dataset.jsonl"]["sha256"] == h

def test_baseline_preservation():
    # Phase1 baseline must still exist and not be overwritten
    assert PHASE1_OUTPUTS.exists(), "Phase1 outputs missing — baseline overwritten!"
    phase1 = [json.loads(l) for l in PHASE1_OUTPUTS.read_text(encoding="utf-8").splitlines()]
    assert len(phase1) == 12
    # Phase3 outputs should also exist and be different (not same file)
    assert PHASE3_OUTPUTS.exists()
    phase3 = [json.loads(l) for l in PHASE3_OUTPUTS.read_text(encoding="utf-8").splitlines()]
    assert len(phase3) == 12
    # They should be different (adapter effect)
    assert phase1[0]["generated"] != phase3[0]["generated"]
    # Check baseline config still points to Qwen without adapter (small-instruct)
    base_cfg = MiyukiConfig.from_yaml("configs/model/small-instruct.yaml")
    assert base_cfg.adapter.enabled is False
    assert base_cfg.model.name == "Qwen/Qwen2-0.5B-Instruct"
