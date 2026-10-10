"""Transformers backend - the first working runtime."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import torch

from ..config import MiyukiConfig
from ..hardware import detect_hardware
from .base import GenerationResult, RuntimeBase


def _resolve_device_dtype(config: MiyukiConfig) -> tuple[str, torch.dtype]:
    hw = detect_hardware()
    # Device
    dev = config.runtime.device
    if dev == "auto":
        dev = hw["recommended_device"]
    # dtype
    dtype_str = config.runtime.dtype
    if dtype_str == "auto":
        dtype_str = hw["recommended_dtype"]
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)
    # CPU cannot use bf16 reliably on older hardware? torch handles it but we fallback
    if dev == "cpu" and dtype == torch.bfloat16:
        # keep bf16 if supported, else float32
        try:
            # quick check: bf16 supported on this CPU build?
            _ = torch.tensor([1.0], dtype=torch.bfloat16)
        except Exception:
            dtype = torch.float32
    return dev, dtype


class TransformersRuntime(RuntimeBase):
    backend_name = "transformers"

    def __init__(self, config: MiyukiConfig):
        super().__init__(config)
        self._model = None
        self._tokenizer = None
        self._device: str = "cpu"
        self._dtype: torch.dtype = torch.float32

    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device, self._dtype = _resolve_device_dtype(self.config)
        model_id = self.config.model.name
        tok_id = self.config.model.resolved_tokenizer()
        trust = self.config.model.trust_remote_code

        # Tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=trust)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Model - handle quantization stubs (actual 4bit requires bitsandbytes)
        quant = self.config.runtime.quantization
        model_kwargs: dict[str, Any] = {"trust_remote_code": trust}
        if quant in ("4bit", "8bit"):
            # bitsandbytes integration if available; else warn and fallback
            try:
                import bitsandbytes  # noqa: F401
                if quant == "4bit":
                    model_kwargs["load_in_4bit"] = True
                else:
                    model_kwargs["load_in_8bit"] = True
            except ImportError:
                print(f"[warn] quantization={quant} requested but bitsandbytes not installed; loading in {self._dtype}")
                model_kwargs["torch_dtype"] = self._dtype
        else:
            model_kwargs["torch_dtype"] = self._dtype
            # attn
            if self.config.runtime.attn_implementation:
                model_kwargs["attn_implementation"] = self.config.runtime.attn_implementation

        self._model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        # Move to device if not quantized (quantized models are already placed)
        if quant not in ("4bit", "8bit"):
            if self._device.startswith("cuda"):
                self._model = self._model.to(self._device)
            elif self._device == "mps":
                try:
                    self._model = self._model.to("mps")
                except Exception:
                    self._model = self._model.to("cpu")
                    self._device = "cpu"
            else:
                self._model = self._model.to("cpu")

        # Try to load adapter if config says so and checkpoint has adapter
        # Adapter loading is handled by PEFT's PeftModel if adapter path exists
        # We check for adapter_config.json next to model or in output_dir
        self._maybe_load_adapter()

        self._model.eval()

    def _maybe_load_adapter(self) -> None:
        if not self.config.adapter.enabled or self.config.adapter.type is None:
            return
        # Look for adapter in model dir or training output dir
        candidates = [
            Path(self.config.model.name) / "adapter_config.json",
            Path(self.config.training.output_dir) / "adapter_config.json",
        ]
        for cand in candidates:
            if cand.exists():
                try:
                    from peft import PeftModel
                    self._model = PeftModel.from_pretrained(self._model, str(cand.parent))
                    print(f"[info] Loaded adapter from {cand.parent}")
                    break
                except Exception as e:
                    print(f"[warn] Failed to load adapter from {cand.parent}: {e}")
                break

    def load_adapter_from(self, adapter_path: str | Path) -> None:
        """Explicitly load a LoRA adapter from a checkpoint directory."""
        from peft import PeftModel
        p = Path(adapter_path)
        if not p.exists():
            raise FileNotFoundError(f"Adapter path not found: {p}")
        # Need base model already loaded
        if not self.is_loaded():
            self.load()
        self._model = PeftModel.from_pretrained(self._model, str(p))
        # Merge or keep as peft model - keep as peft for now

    def generate(self, prompt: str, **overrides) -> GenerationResult:
        if not self.is_loaded():
            self.load()
        assert self._model is not None and self._tokenizer is not None

        gen_cfg = self.config.generation
        params: dict[str, Any] = {
            "temperature": overrides.get("temperature", gen_cfg.temperature),
            "top_p": overrides.get("top_p", gen_cfg.top_p),
            "top_k": overrides.get("top_k", gen_cfg.top_k),
            "max_new_tokens": overrides.get("max_new_tokens", gen_cfg.max_new_tokens),
            "repetition_penalty": overrides.get("repetition_penalty", gen_cfg.repetition_penalty),
            "do_sample": overrides.get("do_sample", gen_cfg.do_sample),
            "seed": overrides.get("seed", gen_cfg.seed),
        }

        seed = params.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        t0 = time.time()
        # Use chat template for instruct models when available (generic, not Qwen-specific).
        # If tokenizer has a chat template, treat the string prompt as a user message.
        prompt_for_tokenizer = prompt
        use_chat = False
        try:
            tmpl = getattr(self._tokenizer, "chat_template", None)
            if tmpl is not None:
                # Generic heuristic: all HF instruct models expose chat_template; plain causal models do not.
                # Using apply_chat_template gives correct special tokens (<|im_start|> etc.).
                messages = [{"role": "user", "content": prompt}]
                # add_generation_prompt ensures the assistant start token is appended
                prompt_for_tokenizer = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                use_chat = True
        except Exception:
            prompt_for_tokenizer = prompt
            use_chat = False

        inputs = self._tokenizer(prompt_for_tokenizer, return_tensors="pt")
        # move inputs to model's device
        try:
            device = next(self._model.parameters()).device  # type: ignore
        except StopIteration:
            device = torch.device("cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        tok_time = time.time() - t0
        # Record whether chat template was used and token counts for metrics
        prompt_tokens = int(inputs["input_ids"].shape[1])

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(params["max_new_tokens"]),
            "do_sample": bool(params["do_sample"]),
            "pad_token_id": self._tokenizer.eos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if params["do_sample"]:
            gen_kwargs["temperature"] = float(params["temperature"])
            gen_kwargs["top_p"] = float(params["top_p"])
            if params["top_k"] is not None:
                gen_kwargs["top_k"] = int(params["top_k"])
        if params["repetition_penalty"] != 1.0:
            gen_kwargs["repetition_penalty"] = float(params["repetition_penalty"])

        t1 = time.time()
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)
        gen_time = time.time() - t1

        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        new_ids = output_ids[0][input_len:]
        generated = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        full_text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Output token count measured from decoded new_ids
        output_tokens = int(new_ids.shape[0]) if hasattr(new_ids, "shape") else len(new_ids)
        # tokens/sec (generation only)
        tps = (output_tokens / gen_time) if gen_time > 0 else 0.0

        hw = detect_hardware()
        result = GenerationResult(
            prompt=prompt,
            generated=generated,
            full_text=full_text,
            model_name=self.config.model.name,
            config_snapshot=self.config.to_dict(),
            runtime_info={
                "backend": self.backend_name,
                "device": str(device),
                "dtype": str(self._dtype),
                "hardware": hw,
                "chat_template_used": use_chat,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "tokens_per_sec": round(tps, 2),
            },
            generation_params=params,
            timings={
                "tokenize_s": tok_time,
                "generate_s": gen_time,
                "total_s": time.time() - t0,
            },
        )
        return result
