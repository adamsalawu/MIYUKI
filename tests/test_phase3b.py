"""Phase 3B tests: config isolation, expected steps, checkpoint, adapter, baseline preservation, metadata."""
import json, pathlib, hashlib

PHASE3B_CFG = pathlib.Path("C:/Aether/RisingStar/MIYUKI/configs/training/phase3b_lora.yaml")
PHASE3_CFG = pathlib.Path("C:/Aether/RisingStar/MIYUKI/configs/training/phase3_lora.yaml")
DATASET = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1/dataset.jsonl")
CKPT3 = pathlib.Path("C:/Aether/RisingStar/MIYUKI/checkpoints/phase3-miyuki-0.1")
CKPT3B = pathlib.Path("C:/Aether/RisingStar/MIYUKI/checkpoints/phase3b-miyuki-0.1")
EXP2 = pathlib.Path("C:/Aether/RisingStar/MIYUKI/experiments/0002/metrics.json")
EXP3 = pathlib.Path("C:/Aether/RisingStar/MIYUKI/experiments/0003/metrics.json")
P1_OUTPUTS = pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase1_qwen2_0.5b/outputs.jsonl")
P3_OUTPUTS = pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase3-miyuki-0.1/outputs.jsonl")
P3B_OUTPUTS = pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase3b-miyuki-0.1/outputs.jsonl")

def test_phase3b_configuration():
    from miyuki.config import MiyukiConfig
    cfg = MiyukiConfig.from_yaml(PHASE3B_CFG)
    assert cfg.model.name == "Qwen/Qwen2-0.5B-Instruct"
    assert cfg.adapter.r == 16 and cfg.adapter.alpha == 32
    assert cfg.training.learning_rate == 0.0002
    assert cfg.training.num_epochs == 3
    assert cfg.training.per_device_train_batch_size == 1
    assert cfg.training.gradient_accumulation_steps == 8
    assert cfg.training.max_seq_length == 1024
    assert cfg.adapter.enabled is True
    # Changes from Phase3: lr 1e-4->2e-4, epochs 1->3
    cfg3 = MiyukiConfig.from_yaml(PHASE3_CFG)
    assert cfg3.training.learning_rate == 0.0001
    assert cfg3.training.num_epochs == 1
    assert cfg.training.learning_rate != cfg3.training.learning_rate
    assert cfg.training.num_epochs != cfg3.training.num_epochs
    # Same dataset/model/tokenizer not changed
    assert cfg.model.tokenizer == cfg3.model.tokenizer
    assert cfg.model.name == cfg3.model.name

def test_experiment_isolation():
    assert EXP2.exists() and EXP3.exists()
    d2 = json.loads(EXP2.read_text(encoding="utf-8"))
    d3 = json.loads(EXP3.read_text(encoding="utf-8"))
    assert d2["id"] == "0002"
    assert d3["id"] == "0003"
    assert d2["id"] != d3["id"]
    # Checkpoints separate
    assert CKPT3.exists() and CKPT3B.exists()
    assert CKPT3 != CKPT3B
    assert (CKPT3 / "adapter_config.json").exists()
    assert (CKPT3B / "adapter_config.json").exists()
    # Adapter configs should have same r but different training
    c2 = json.loads((CKPT3 / "adapter_config.json").read_text(encoding="utf-8"))
    c3 = json.loads((CKPT3B / "adapter_config.json").read_text(encoding="utf-8"))
    assert c2["r"] == 16 and c3["r"] == 16
    # But training epochs differ via experiment metadata
    assert d2["training_config"]["training"]["num_epochs"] == 1
    assert d3["training_config"]["training"]["num_epochs"] == 3

def test_deterministic_dataset_split():
    # Both Phase3 and 3B should use same 90/10 split with seed 42 -> 34 train, 4 eval
    from miyuki.config import MiyukiConfig
    cfg = MiyukiConfig.from_yaml(PHASE3B_CFG)
    assert cfg.training.seed == 42
    # Load dataset and split as train.py does
    from datasets import Dataset
    from transformers import AutoTokenizer
    import json
    raw = [json.loads(l) for l in DATASET.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(raw) == 38
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    def to_text(ex):
        msgs = ex["messages"]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) if tok.chat_template else str(ex)
    texts = [to_text(ex) for ex in raw]
    ds = Dataset.from_dict({"text": texts})
    def tok_fn(b):
        return tok(b["text"], truncation=True, max_length=1024, padding=False)
    tok_ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
    split = tok_ds.train_test_split(test_size=0.1, seed=42)
    assert len(split["train"]) == 34
    assert len(split["test"]) == 4
    # Check deterministic: second split same
    split2 = tok_ds.train_test_split(test_size=0.1, seed=42)
    assert list(split["train"]["input_ids"][0]) == list(split2["train"]["input_ids"][0])

def test_expected_optimizer_steps():
    # Phase3: 4 steps, Phase3B: 12 steps (3x)
    d2 = json.loads(EXP2.read_text(encoding="utf-8"))
    d3 = json.loads(EXP3.read_text(encoding="utf-8"))
    steps2 = d2["metrics"]["global_step"]
    steps3 = d3["metrics"]["global_step"]
    assert steps2 == 4, f"Phase3 expected 4 got {steps2}"
    assert steps3 == 12, f"Phase3B expected 12 got {steps3}"
    # Phase3B should be substantially more than 4
    assert steps3 > 10
    assert steps3 == steps2 * 3
    # Also check train_loss decreased more in Phase3B (more training)
    assert d3["final_loss"] < d2["final_loss"]  # 3.20 < 3.76

def test_checkpoint_discovery():
    from miyuki.artifacts.manager import discover_checkpoints
    found = discover_checkpoints("checkpoints")
    strs = [str(p) for p in found]
    assert any("phase3-miyuki-0.1" in s for s in strs)
    assert any("phase3b-miyuki-0.1" in s for s in strs)
    assert any("phase3b" in s for s in strs)

def test_adapter_loading():
    import json
    c3 = json.loads((CKPT3B / "adapter_config.json").read_text(encoding="utf-8"))
    assert c3["base_model_name_or_path"] == "Qwen/Qwen2-0.5B-Instruct"
    # Tokenizer not mismatched
    from transformers import AutoTokenizer
    tok_base = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=False)
    tok_ckpt = AutoTokenizer.from_pretrained(str(CKPT3B), trust_remote_code=False)
    assert len(tok_base) == len(tok_ckpt) == 151646

def test_baseline_preservation():
    assert P1_OUTPUTS.exists()
    assert P3_OUTPUTS.exists()
    assert P3B_OUTPUTS.exists()
    p1 = [json.loads(l) for l in P1_OUTPUTS.read_text(encoding="utf-8").splitlines()]
    p3 = [json.loads(l) for l in P3_OUTPUTS.read_text(encoding="utf-8").splitlines()]
    p3b = [json.loads(l) for l in P3B_OUTPUTS.read_text(encoding="utf-8").splitlines()]
    assert len(p1)==12 and len(p3)==12 and len(p3b)==12
    # Baselines not overwritten: Phase1 still has its original first output start
    assert "it was time to return home" in p1[0]["generated"]
    # Phase3 and 3B are different from base
    assert p1[0]["generated"][:50] != p3[0]["generated"][:50] or p1[0]["generated"] != p3[0]["generated"]
    # Phase3 vs 3B also different (proves separate adapters)
    assert p3[0]["generated"][:100] != p3b[0]["generated"][:100]

def test_metadata_completeness():
    d3 = json.loads(EXP3.read_text(encoding="utf-8"))
    assert "model" in d3
    assert "dataset" in d3
    assert "dataset_version" in d3
    assert "training_config" in d3
    assert "hardware" in d3
    assert "duration_s" in d3
    assert "final_loss" in d3
    assert "artifact_location" in d3
    # Check that Phase3B metadata records changed hyperparams
    assert d3["training_config"]["training"]["learning_rate"] == 0.0002
    assert d3["training_config"]["training"]["num_epochs"] == 3
    # Record dataset SHA
    assert d3["dataset_version"] == "0.1.0-phase3b"
    # Hardware present
    assert d3["hardware"]["cpu"] is not None
