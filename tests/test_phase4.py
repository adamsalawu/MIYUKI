"""Phase 4 tests: assistant-only config, dataset 0.2, split, checkpoint, metadata, baseline preservation."""
import json, pathlib, hashlib

ORIG = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1/dataset.jsonl")
DS02 = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.2/dataset.jsonl")
CKPT4 = pathlib.Path("C:/Aether/RisingStar/MIYUKI/checkpoints/phase4-miyuki-0.2")
EXP6 = pathlib.Path("C:/Aether/RisingStar/MIYUKI/experiments/0006/metrics.json")
P1 = pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase1_qwen2_0.5b/outputs.jsonl")
P4 = pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase4-miyuki-0.2/2026-09-25_Qwen_Qwen2-0.5B-Instruct/outputs.jsonl")

def test_phase4_config_assistant_only():
    from miyuki.config import MiyukiConfig
    cfg = MiyukiConfig.from_yaml("C:/Aether/RisingStar/MIYUKI/configs/training/phase4_lora.yaml")
    assert cfg.training.assistant_only_loss is True
    assert cfg.model.name == "Qwen/Qwen2-0.5B-Instruct"
    assert cfg.adapter.r == 16 and cfg.adapter.alpha == 32
    assert cfg.training.num_epochs == 3 and cfg.training.learning_rate == 0.0002
    # previous preserved as False
    cfg3b = MiyukiConfig.from_yaml("C:/Aether/RisingStar/MIYUKI/configs/training/phase3b_lora.yaml")
    assert cfg3b.training.assistant_only_loss is False

def test_dataset_02_preservation_and_counts():
    assert ORIG.exists() and DS02.exists()
    assert hashlib.sha256(ORIG.read_bytes()).hexdigest() == "9e1b97461f98948254bc45906cb275f111318f5d5adfc9e7139ab9c35c200d5e"
    assert hashlib.sha256(DS02.read_bytes()).hexdigest() == "8cda31a91a76b1d6e8dc67ea2b779de06334836dad089b6bd59210dbdc6985d6"
    data = [json.loads(l) for l in DS02.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(data) == 120
    # first 38 should match original IDs
    orig_ids = [json.loads(l)["metadata"]["id"] for l in ORIG.read_text(encoding="utf-8").splitlines() if l.strip()]
    new_ids = [d["metadata"]["id"] for d in data[:38]]
    assert orig_ids == new_ids

def test_deterministic_split_108_12():
    from datasets import Dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    raw = [json.loads(l) for l in DS02.read_text(encoding="utf-8").splitlines() if l.strip()]
    def to_text(ex):
        msgs = ex["messages"]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    texts = [to_text(ex) for ex in raw]
    ds = Dataset.from_dict({"text": texts})
    def tok_fn(b):
        return tok(b["text"], truncation=True, max_length=1024, padding=False)
    tok_ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
    split = tok_ds.train_test_split(test_size=0.1, seed=42)
    assert len(split["train"]) == 108 and len(split["test"]) == 12
    split2 = tok_ds.train_test_split(test_size=0.1, seed=42)
    assert list(split["train"]["input_ids"][0]) == list(split2["train"]["input_ids"][0])

def test_expected_optimizer_steps_39():
    d = json.loads(EXP6.read_text(encoding="utf-8"))
    assert d["id"] == "0006"
    assert d["metrics"]["global_step"] == 39
    assert d["final_loss"] < 3.0  # 2.62
    assert d["training_config"]["training"]["num_epochs"] == 3

def test_checkpoint_discovery():
    from miyuki.artifacts.manager import discover_checkpoints
    found = [str(p) for p in discover_checkpoints("checkpoints")]
    assert any("phase4-miyuki-0.2" in s for s in found)
    assert (CKPT4 / "adapter_config.json").exists()
    assert (CKPT4 / "adapter_model.safetensors").exists()
    cfg = json.loads((CKPT4 / "adapter_config.json").read_text(encoding="utf-8"))
    assert cfg["r"] == 16 and cfg["base_model_name_or_path"] == "Qwen/Qwen2-0.5B-Instruct"

def test_adapter_loading_tokenizer_match():
    from transformers import AutoTokenizer
    tok_base = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=False)
    tok_ckpt = AutoTokenizer.from_pretrained(str(CKPT4), trust_remote_code=False)
    assert len(tok_base) == len(tok_ckpt) == 151646

def test_baseline_preservation_and_comparison():
    assert P1.exists() and P4.exists()
    p1 = [json.loads(l) for l in P1.read_text(encoding="utf-8").splitlines()]
    p4 = [json.loads(l) for l in P4.read_text(encoding="utf-8").splitlines()]
    assert len(p1) == 12 and len(p4) == 12
    # Phase4 should have no hedging/meta essay flags that Phase1 had
    assert "could be one of surprise" not in p4[11]["generated"]
    assert "The answer is:" not in p4[6]["generated"]

def test_metadata_completeness():
    d = json.loads(EXP6.read_text(encoding="utf-8"))
    assert d["model"] == "Qwen/Qwen2-0.5B-Instruct"
    assert "miyuki-writing-0.2" in d["dataset"]
    assert d["training_config"]["training"]["assistant_only_loss"] is True
    assert d["final_loss"] is not None and d["duration_s"] > 0
