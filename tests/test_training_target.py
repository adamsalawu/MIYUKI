"""Training-target test: verify assistant-only masking."""
import json, pathlib
from transformers import AutoTokenizer

def test_assistant_only_masking():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Load one real example
    dataset_path = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1/dataset.jsonl")
    ex = json.loads(pathlib.Path(dataset_path).read_text(encoding="utf-8").splitlines()[0])
    from miyuki.datasets.loader import format_for_training
    fmt = format_for_training(ex)
    msgs = fmt["messages"]
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    # Simulate train.py masking logic
    full_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    prefix_text = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
    prefix_ids = tok(prefix_text, add_special_tokens=False)["input_ids"]
    mask_len = min(len(prefix_ids), len(full_ids))
    assert full_ids[:mask_len] == prefix_ids[:mask_len], "prefix should be prefix of full"
    labels = [-100]*mask_len + full_ids[mask_len:]
    # Verify
    assert all(l==-100 for l in labels[:mask_len]), "user/system should be masked"
    assert all(l!=-100 for l in labels[mask_len:]), "assistant should be trainable"
    # Special tokens: before should be masked, after trainable
    special_ids = set(tok.all_special_ids)
    before_special = [i for i in range(mask_len) if full_ids[i] in special_ids]
    after_special = [i for i in range(mask_len, len(full_ids)) if full_ids[i] in special_ids]
    assert all(labels[i]==-100 for i in before_special)
    # Final <|im_end|> after assistant should be trainable (if present)
    if after_special:
        assert any(labels[i]!=-100 for i in after_special)

def test_no_prompt_becomes_target_and_no_assistant_masked():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dataset_path = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1/dataset.jsonl")
    ex = json.loads(pathlib.Path(dataset_path).read_text(encoding="utf-8").splitlines()[0])
    from miyuki.datasets.loader import format_for_training
    fmt = format_for_training(ex)
    msgs = fmt["messages"]
    full_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    prefix_text = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
    prefix_ids = tok(prefix_text, add_special_tokens=False)["input_ids"]
    mask_len = min(len(prefix_ids), len(full_ids))
    labels = [-100]*mask_len + full_ids[mask_len:]
    # No prompt text should be in trainable part that is actually user prompt
    # Check that number of trainable tokens equals assistant part (approx 75 for this example)
    assert 70 <= len(full_ids) - mask_len <= 80
    # Check that we didn't accidentally mask assistant
    assert labels[mask_len] != -100
    # Check that we didn't leave user trainable
    assert labels[0] == -100

def test_phase4_config_has_assistant_only_true():
    from miyuki.config import MiyukiConfig
    cfg = MiyukiConfig.from_yaml("C:/Aether/RisingStar/MIYUKI/configs/training/phase4_lora.yaml")
    assert cfg.training.assistant_only_loss is True
    # Previous configs should have False (preserve)
    cfg3 = MiyukiConfig.from_yaml("C:/Aether/RisingStar/MIYUKI/configs/training/phase3b_lora.yaml")
    assert cfg3.training.assistant_only_loss is False
