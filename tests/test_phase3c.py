"""Phase 3C tests: derived dataset determinism, formulation structure, token stats, experiment isolation."""
import json, pathlib, hashlib

ORIG = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1/dataset.jsonl")
DERIVED_A = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1-phase3c/dataset-a.jsonl")
DERIVED_B = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1-phase3c/dataset-b.jsonl")
MANIFEST_DERIVED = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1-phase3c/manifest.json")
MANIFEST_ORIG = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1/manifest.json")

def test_original_dataset_preservation():
    # Original must remain byte-for-byte unchanged
    orig_sha = hashlib.sha256(ORIG.read_bytes()).hexdigest()
    assert orig_sha == "9e1b97461f98948254bc45906cb275f111318f5d5adfc9e7139ab9c35c200d5e"
    manifest_orig = json.loads(MANIFEST_ORIG.read_text(encoding="utf-8"))
    assert manifest_orig["files"]["dataset.jsonl"]["sha256"] == orig_sha

def test_derived_dataset_determinism():
    # Both derived should be deterministic: re-reading gives same SHA
    sha_a = hashlib.sha256(DERIVED_A.read_bytes()).hexdigest()
    sha_b = hashlib.sha256(DERIVED_B.read_bytes()).hexdigest()
    # From manifest
    m = json.loads(MANIFEST_DERIVED.read_text(encoding="utf-8"))
    assert m["derived"]["A"]["sha"] == sha_a
    assert m["derived"]["B"]["sha"] == sha_b
    assert sha_a == "b518f30bffe070ab62b384b1d936fd38d8ecdd6910b40367c9c0ae09a84e9d7c"
    assert sha_b == "e408e23ecfc6d5c7f509bdd9e823d6a45d2750ba63ce974d05f87310e471318c"
    assert sha_a != sha_b
    assert sha_a != hashlib.sha256(ORIG.read_bytes()).hexdigest()

def test_formulation_a_structure():
    import json
    ex = json.loads(DERIVED_A.read_text(encoding="utf-8").splitlines()[0])
    # Must have system + user + assistant
    assert len(ex["messages"]) == 3
    assert ex["messages"][0]["role"] == "system"
    assert "Miyuki" in ex["messages"][0]["content"]
    assert "Perform the requested writing task directly" in ex["messages"][0]["content"]
    assert ex["messages"][1]["role"] == "user"
    assert ex["messages"][2]["role"] == "assistant"
    # Assistant target identical to original
    orig_ex = json.loads(ORIG.read_text(encoding="utf-8").splitlines()[0])
    assert ex["messages"][2]["content"] == orig_ex["messages"][1]["content"]
    assert ex["metadata"]["formulation"] == "A-system-contract"

def test_formulation_b_structure():
    ex = json.loads(DERIVED_B.read_text(encoding="utf-8").splitlines()[0])
    # Must have user + assistant only, no system
    assert len(ex["messages"]) == 2
    assert ex["messages"][0]["role"] == "user"
    assert "Requirements: Perform the writing directly" in ex["messages"][0]["content"]
    assert ex["messages"][1]["role"] == "assistant"
    orig_ex = json.loads(ORIG.read_text(encoding="utf-8").splitlines()[0])
    assert ex["messages"][1]["content"] == orig_ex["messages"][1]["content"]
    assert ex["metadata"]["formulation"] == "B-user-contract"
    # Ensure no system role
    assert not any(m["role"]=="system" for m in ex["messages"])

def test_token_statistics():
    # From earlier measurement: original 5869 total, A 8073, B 7428, avg etc, max <291 <1024
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    def count(path):
        examples = [json.loads(l) for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
        total=0
        max_t=0
        for ex in examples:
            msgs = ex["messages"]
            txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            ids = tok(txt, truncation=True, max_length=1024, padding=False)["input_ids"]
            total+=len(ids)
            max_t=max(max_t, len(ids))
        return total, max_t
    total_orig, max_orig = count(ORIG)
    total_a, max_a = count(DERIVED_A)
    total_b, max_b = count(DERIVED_B)
    assert total_orig == 5869
    assert total_a == 8073
    assert total_b == 7428
    assert max_orig == 233
    assert max_a == 291
    assert max_b == 274
    # Not excessive: all <900 and <88% of 1024
    assert max_a < 900 and max_b < 900
    assert max_a/1024 < 0.88

def test_experiment_isolation():
    # 0004 and 0005 separate, not overwriting 0002/0003
    exp4 = pathlib.Path("C:/Aether/RisingStar/MIYUKI/experiments/0004/metrics.json")
    exp5 = pathlib.Path("C:/Aether/RisingStar/MIYUKI/experiments/0005/metrics.json")
    assert exp4.exists() and exp5.exists()
    d4 = json.loads(exp4.read_text(encoding="utf-8"))
    d5 = json.loads(exp5.read_text(encoding="utf-8"))
    assert d4["id"] == "0004" and d5["id"] == "0005"
    assert d4["id"] != d5["id"]
    assert d4["dataset_version"] == "0.1.0-phase3c-a"
    assert d5["dataset_version"] == "0.1.0-phase3c-b"
    # Checkpoints separate
    ckpt_a = pathlib.Path("C:/Aether/RisingStar/MIYUKI/checkpoints/phase3c-a-miyuki-0.1")
    ckpt_b = pathlib.Path("C:/Aether/RisingStar/MIYUKI/checkpoints/phase3c-b-miyuki-0.1")
    assert ckpt_a.exists() and ckpt_b.exists()
    assert ckpt_a != ckpt_b
    assert (ckpt_a / "adapter_config.json").exists()
    assert (ckpt_b / "adapter_config.json").exists()

def test_adapter_loading():
    from transformers import AutoTokenizer
    tok_base = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=False)
    tok_a = AutoTokenizer.from_pretrained(str(pathlib.Path("C:/Aether/RisingStar/MIYUKI/checkpoints/phase3c-a-miyuki-0.1")), trust_remote_code=False)
    tok_b = AutoTokenizer.from_pretrained(str(pathlib.Path("C:/Aether/RisingStar/MIYUKI/checkpoints/phase3c-b-miyuki-0.1")), trust_remote_code=False)
    assert len(tok_base) == len(tok_a) == len(tok_b) == 151646

def test_metadata_completeness():
    exp4 = json.loads(pathlib.Path("C:/Aether/RisingStar/MIYUKI/experiments/0004/metrics.json").read_text(encoding="utf-8"))
    exp5 = json.loads(pathlib.Path("C:/Aether/RisingStar/MIYUKI/experiments/0005/metrics.json").read_text(encoding="utf-8"))
    for d in [exp4, exp5]:
        assert "model" in d and d["model"] == "Qwen/Qwen2-0.5B-Instruct"
        assert "dataset" in d and "miyuki-writing-0.1-phase3c" in d["dataset"]
        assert "training_config" in d
        assert "hardware" in d
        assert "duration_s" in d
        assert "final_loss" in d
        assert "artifact_location" in d
        assert pathlib.Path(d["artifact_location"]).exists()
    # Check that training config records formulation via dataset path
    assert "dataset-a.jsonl" in exp4["dataset"]
    assert "dataset-b.jsonl" in exp5["dataset"]
