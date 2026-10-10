"""Regression test: chat template generic handling + runtime reuse."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from miyuki.config import MiyukiConfig
from miyuki.runtime.transformers_runtime import TransformersRuntime
from miyuki.runtime.base import GenerationResult


def _mock_tokenizer_with_template(has_template: bool):
    tok = MagicMock()
    tok.chat_template = "dummy template" if has_template else None
    if has_template:
        tok.apply_chat_template.return_value = "<|im_start|>user prompt<|im_end|> <|im_start|>assistant "
        tok.return_value = {"input_ids": MagicMock(shape=[1, 10])}
        # Mock tokenization for chat path
        tok.__call__ = MagicMock(return_value={"input_ids": MagicMock(shape=[1, 10], __len__=lambda s: 10)})
        # For templated case, tokenizer returns 10 tokens
        from unittest.mock import PropertyMock
        # Simplify: make tokenizer call return dict with input_ids tensor-like
        import torch
        tok.return_value = {"input_ids": torch.tensor([[1]*10])}
        tok.apply_chat_template = MagicMock(return_value="<|im_start|>user hello<|im_end|> <|im_start|>assistant ")
    else:
        import torch
        tok.return_value = {"input_ids": torch.tensor([[1]*5])}
        tok.__call__ = MagicMock(return_value={"input_ids": torch.tensor([[1]*5])})
    tok.pad_token = "<pad>"
    tok.eos_token = "<eos>"
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    tok.decode = MagicMock(return_value="generated text")
    # Tokenizer length
    return tok


def test_chat_template_used_when_available(monkeypatch):
    cfg = MiyukiConfig.from_yaml("C:/Aether/RisingStar/MIYUKI/configs/model/baseline.yaml")
    # Use a mock that has chat_template
    cfg.model.name = "dummy-model"
    cfg.model.family = "qwen2"

    # Patch tokenizer and model to avoid download
    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_cls, \
         patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
         patch("miyuki.hardware.detect_hardware") as mock_hw:

        mock_hw.return_value = {"recommended_device": "cpu", "recommended_dtype": "float32", "cuda_available": False, "mps_available": False, "gpu_count": 0}

        # Mock tokenizer with template
        import torch
        tok = MagicMock()
        tok.chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
        tok.apply_chat_template = MagicMock(return_value="formatted prompt")
        tok.return_value = {"input_ids": torch.tensor([[1,2,3,4,5]])}
        tok.decode = MagicMock(return_value="gen")
        tok.pad_token = "<pad>"
        tok.eos_token = "<eos>"
        tok.eos_token_id = 1
        tok.pad_token_id = 0
        # Make tokenizer callable
        tok.__call__ = MagicMock(return_value={"input_ids": torch.tensor([[1,2,3,4,5]])})
        tok.apply_chat_template.return_value = "formatted prompt"
        mock_tok_cls.return_value = tok

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_model.generate.return_value = torch.tensor([[1,2,3,4,5,6,7]])
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None
        mock_model_cls.return_value = mock_model

        rt = TransformersRuntime(cfg)
        rt.load()
        # Generate should use chat template path
        res = rt.generate("hello", max_new_tokens=10)
        # apply_chat_template should have been called
        assert tok.apply_chat_template.called, "chat template should be used when present"
        assert res.runtime_info["chat_template_used"] is True


def test_chat_template_not_used_for_plain_models(monkeypatch):
    cfg = MiyukiConfig.from_yaml("C:/Aether/RisingStar/MIYUKI/configs/model/baseline.yaml")
    cfg.model.name = "dummy-model"

    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_cls, \
         patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model_cls, \
         patch("miyuki.hardware.detect_hardware") as mock_hw:

        mock_hw.return_value = {"recommended_device": "cpu", "recommended_dtype": "float32", "cuda_available": False, "mps_available": False, "gpu_count": 0}

        import torch
        tok = MagicMock()
        tok.chat_template = None
        tok.__call__ = MagicMock(return_value={"input_ids": torch.tensor([[1,2,3]])})
        tok.return_value = {"input_ids": torch.tensor([[1,2,3]])}
        tok.decode = MagicMock(return_value="gen")
        tok.pad_token = "<pad>"
        tok.eos_token = "<eos>"
        tok.eos_token_id = 1
        tok.pad_token_id = 0
        mock_tok_cls.return_value = tok

        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_model.generate.return_value = torch.tensor([[1,2,3,4]])
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = None
        mock_model_cls.return_value = mock_model

        rt = TransformersRuntime(cfg)
        rt.load()
        res = rt.generate("hello", max_new_tokens=10)
        assert res.runtime_info["chat_template_used"] is False
