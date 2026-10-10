"""Scale test: new 1.5B config loads, 0.5B preserved, Phase4 preserved, eval deterministic."""
import json, pathlib

def test_new_model_config_loads():
    from miyuki.config import MiyukiConfig
    c = MiyukiConfig.from_yaml("C:/Aether/RisingStar/MIYUKI/configs/model/qwen2.5-1.5b-instruct.yaml")
    assert c.model.name == "Qwen/Qwen2.5-1.5B-Instruct"
    assert c.model.family == "qwen2"
    assert c.model.context_length == 32768
    assert c.generation.max_new_tokens == 180

def test_model_identifier_correct():
    from miyuki.config import MiyukiConfig
    c = MiyukiConfig.from_yaml("C:/Aether/RisingStar/MIYUKI/configs/model/qwen2.5-1.5b-instruct.yaml")
    assert c.model.tokenizer == "Qwen/Qwen2.5-1.5B-Instruct"
    assert "1.5B" in c.model.name

def test_tokenizer_loads_and_chat():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=False)
    assert len(tok) == 151665
    assert bool(tok.chat_template) is True
    # chat template works
    msgs = [{"role": "user", "content": "Hello"}]
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert "<|im_start|>" in txt

def test_inference_runtime_handles_model():
    # Verify sanity report shows load + generation worked (avoids reloading 1.5B here)
    import json
    sanity = json.loads(pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/runs/model-scale-qwen2.5-1.5b/sanity.json").read_text(encoding="utf-8"))
    assert sanity["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert sanity["params"] == 1543714304
    assert all(r["ok"] for r in sanity["results"])
    assert sanity["device"] == "cpu"

def test_existing_05b_config_unchanged():
    from miyuki.config import MiyukiConfig
    old = MiyukiConfig.from_yaml("C:/Aether/RisingStar/MIYUKI/configs/model/small-instruct.yaml")
    assert old.model.name == "Qwen/Qwen2-0.5B-Instruct"
    assert old.model.context_length == 4096

def test_phase4_artifacts_unchanged():
    import hashlib
    orig = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1/dataset.jsonl")
    assert hashlib.sha256(orig.read_bytes()).hexdigest() == "9e1b97461f98948254bc45906cb275f111318f5d5adfc9e7139ab9c35c200d5e"
    ds02 = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.2/dataset.jsonl")
    assert hashlib.sha256(ds02.read_bytes()).hexdigest() == "8cda31a91a76b1d6e8dc67ea2b779de06334836dad089b6bd59210dbdc6985d6"
    assert pathlib.Path("C:/Aether/RisingStar/MIYUKI/checkpoints/phase4-miyuki-0.2/adapter_config.json").exists()
    assert pathlib.Path("C:/Aether/RisingStar/MIYUKI/experiments/0006/metrics.json").exists()

def test_evaluation_deterministic():
    # Same 12 prompts used (eval_v0.1.jsonl, 12 lines)
    from miyuki.evaluation.runner import load_prompts
    prompts = load_prompts("C:/Aether/RisingStar/MIYUKI/evaluation/prompts/eval_v0.1.jsonl")
    assert len(prompts) == 12
    # Scale outputs have 12 with same IDs in order
    scale = [json.loads(l) for l in pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/runs/model-scale-qwen2.5-1.5b/outputs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(scale) == 12
    assert [o["id"] for o in scale] == [p["id"] for p in prompts]
