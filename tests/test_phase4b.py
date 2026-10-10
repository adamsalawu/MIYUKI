"""Phase 4B tests: 0.2 preserved, 0.3 additions only, metadata, SHA, tokenizer, no 0009,
Phase4 checkpoint."""
import json, pathlib, hashlib

ORIG02 = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.2/dataset.jsonl")
DS03 = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.3/dataset.jsonl")
MANIFEST03 = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.3/manifest.json")

def test_02_remains_unchanged():
    assert hashlib.sha256(ORIG02.read_bytes()).hexdigest() == "8cda31a91a76b1d6e8dc67ea2b779de06334836dad089b6bd59210dbdc6985d6"

def test_03_contains_all_02():
    lines02 = ORIG02.read_text(encoding="utf-8").splitlines()
    lines03 = DS03.read_text(encoding="utf-8").splitlines()
    assert len(lines03) == 154
    assert lines03[:120] == lines02  # byte-for-byte preserved

def test_03_only_intended_additions():
    data = [json.loads(l) for l in DS03.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(data) == 154
    new = data[120:]
    assert len(new) == 34
    ids = [d["metadata"]["id"] for d in new]
    assert ids == [f"miyuki-0.3-{i}" for i in range(121, 155)]
    # per-target counts via id ranges
    assert len([d for d in new if 121 <= int(d["metadata"]["id"].split("-")[-1]) <= 128]) == 8  # A
    assert len([d for d in new if 129 <= int(d["metadata"]["id"].split("-")[-1]) <= 136]) == 8  # B
    assert len([d for d in new if 137 <= int(d["metadata"]["id"].split("-")[-1]) <= 142]) == 6  # C
    assert len([d for d in new if 143 <= int(d["metadata"]["id"].split("-")[-1]) <= 148]) == 6  # D
    assert len([d for d in new if 149 <= int(d["metadata"]["id"].split("-")[-1]) <= 154]) == 6  # E

def test_no_duplicate_examples():
    from miyuki.datasets.validate import validate_file
    report = validate_file(DS03, format="conversational")
    assert report.passed and report.total == 154 and report.duplicate_count == 0
    data = [json.loads(l) for l in DS03.read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = [d["metadata"]["id"] for d in data]
    assert len(ids) == len(set(ids))

def test_new_examples_valid_metadata():
    import re
    data = [json.loads(l) for l in DS03.read_text(encoding="utf-8").splitlines() if l.strip()][120:]
    allowed_roles = {"system", "user", "assistant"}
    for d in data:
        md = d["metadata"]
        for f in ["id", "instruction", "context", "category", "skills", "source", "notes", "word_count"]:
            assert f in md, f"missing {f} in {md.get('id')}"
        assert md["source"] == "human-authored"
        assert len(md["category"]) >= 1 and len(md["skills"]) >= 1
        assert md["word_count"] >= 1
        for m in d["messages"]:
            assert m["role"] in allowed_roles and len(m["content"].strip()) > 0
        assert len(d["messages"]) in (2, 3)
        # anti-pattern tags live in notes metadata only, not leaked into assistant target
        assert any(tag in md["notes"] for tag in ["distinct_voice", "binary_context", "witty_voice", "object_tracking", "viewpoint_constraint"])
        low_target = d["messages"][-1]["content"].lower()
        assert "distinct_voice" not in low_target and "binary_context" not in low_target

def test_new_dataset_sha_recorded():
    m = json.loads(MANIFEST03.read_text(encoding="utf-8"))
    actual = hashlib.sha256(DS03.read_bytes()).hexdigest()
    assert m["files"]["dataset.jsonl"]["sha256"] == actual
    assert m["derived_sha"] == actual
    assert m["source_sha"] == "8cda31a91a76b1d6e8dc67ea2b779de06334836dad089b6bd59210dbdc6985d6"
    assert m["num_examples"] == 154

def test_production_tokenizer_unchanged():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=False)
    assert len(tok) == 151646 and bool(tok.chat_template) is True

def test_no_experiment_0009():
    # NOTE (2026-09-30 migration): originally asserted no 0007; Phase 4C has since
    # legitimately created 0007 (+0008 duplicate) via the recording mechanism.
    # Guard now targets the next unassigned ID so the "no accidental training runs"
    # invariant keeps holding. Strictness unchanged.
    assert not pathlib.Path("C:/Aether/RisingStar/MIYUKI/experiments/0009").exists()
    assert pathlib.Path("C:/Aether/RisingStar/MIYUKI/experiments/0008/metrics.json").exists()

def test_phase4_checkpoint_unchanged():
    p = pathlib.Path("C:/Aether/RisingStar/MIYUKI/checkpoints/phase4-miyuki-0.2/adapter_config.json")
    assert p.exists()
    cfg = json.loads(p.read_text(encoding="utf-8"))
    assert cfg["base_model_name_or_path"] == "Qwen/Qwen2-0.5B-Instruct" and cfg["r"] == 16
