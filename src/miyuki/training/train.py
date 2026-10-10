"""Training pipeline - SFT + LoRA/PEFT, checkpointing, reproducible.

Uses TRL/Transformers/PEFT where appropriate.
Deterministic/reproducible as far as practical (seeds set).

Example:
  python -m miyuki.training.train --config configs/training/smoke.yaml
"""
from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch

from ..config import MiyukiConfig
from ..datasets.loader import load_jsonl, format_for_training


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _enable_local_checkpoint_rng_compat() -> None:
    """Allowlist the exact NumPy globals needed to torch.load() a trusted, locally-generated
    MIYUKI checkpoint's rng_state.pth under torch >= 2.6 (weights_only=True default).

    Narrowly scoped compatibility: called ONLY from the trusted local training-resume path.
    - Additive (add_safe_globals), never replaces the allowlist; arbitrary code execution
      globals remain rejected.
    - Does not touch checkpoint files; does not affect any other load path.
    Globals verified empirically against checkpoint-30/rng_state.pth
    (tools/diagnose_rng_state.py): _reconstruct, ndarray, dtype, UInt32DType.
    """
    try:
        import numpy as np
        from torch.serialization import add_safe_globals
        add_safe_globals([
            np._core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
            np.dtypes.UInt32DType,
        ])
    except Exception as e:
        # Never block training on the compat shim itself; the resume will fail with a
        # clear torch error if the allowlist is insufficient on this numpy build.
        print(f"[warn] checkpoint RNG compat allowlist unavailable on this numpy/torch build: {e}")


def _resolve_device_dtype(config: MiyukiConfig) -> tuple[str, torch.dtype]:
    from ..hardware import detect_hardware
    hw = detect_hardware()
    dev = config.runtime.device
    if dev == "auto":
        dev = hw["recommended_device"]
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
    return dev, dtype


def _get_smoke_dataset(path: str | Path, max_seq_length: int = 128) -> Path:
    """Ensure a tiny synthetic smoke dataset exists and return its path."""
    p = Path(path)
    if p.exists():
        return p
    # Create 20 synthetic story-like examples (explicitly synthetic, not pretending to be real data)
    p.parent.mkdir(parents=True, exist_ok=True)
    examples = [
        {
            "messages": [
                {"role": "user", "content": f"Instruction: Continue the scene.\n\nStory context: Scene {i} in a quiet village.\n\nRecent prose: The wind moved through the pines."},
                {"role": "assistant", "content": f"The village slept under pale moonlight. Footsteps echoed on the stone path as the traveler returned home after a long journey, number {i}."},
            ]
        }
        for i in range(20)
    ]
    # Mix in some prompt/completion style too for coverage
    extra = [
        {"prompt": f"Continue: The lighthouse beam {i}", "completion": f" swept across the dark water, revealing a small boat adrift {i}."}
        for i in range(5)
    ]
    # Write conversational examples (primary for smoke)
    with open(p, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return p


def train(
    config: Optional[MiyukiConfig] = None,
    config_path: Optional[str | Path] = None,
    dataset_path: Optional[str | Path] = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Launch training.

    Args:
        config: already-loaded config (takes precedence)
        config_path: path to YAML config
        dataset_path: override dataset file (if None, uses synthetic smoke dataset)
        smoke: force smoke mode - ensures tiny dataset
    """
    # Load config
    if config is None:
        if config_path is None:
            config_path = "configs/training/smoke.yaml"
        # Try multiple locations
        candidates = [Path(config_path), Path("C:/Aether/RisingStar/MIYUKI") / config_path, Path(__file__).parents[2] / config_path]
        found = None
        for c in candidates:
            if c.exists():
                found = c
                break
        if found is None:
            # fallback in-memory tiny config
            from ..config import ModelIdentity, RuntimeConfig, GenerationConfig, AdapterConfig, TrainingConfig, MiyukiConfig as MC
            config = MC(
                model=ModelIdentity(name="hf-internal-testing/tiny-random-gpt2", family="gpt2", context_length=128),
                runtime=RuntimeConfig(device="auto", dtype="auto"),
                generation=GenerationConfig(max_new_tokens=32),
                adapter=AdapterConfig(type="lora", r=4, alpha=8, enabled=True, target_modules=["c_attn", "c_proj"]),
                training=TrainingConfig(output_dir="checkpoints/smoke", num_epochs=1, per_device_train_batch_size=1, max_seq_length=128, logging_steps=1, save_steps=5),
            )
        else:
            config = MiyukiConfig.from_yaml(found)

    smoke = smoke or "smoke" in str(config_path or "") or config.version.endswith("smoke")
    training_cfg = config.training
    set_seed(training_cfg.seed)

    # Resolve dataset
    if dataset_path is None:
        if smoke:
            dataset_path = Path(config.training.output_dir).parent / "smoke_dataset.jsonl"
            # Ensure smoke dataset exists
            _get_smoke_dataset(dataset_path)
        else:
            # Check for a real dataset; if missing, create smoke as fallback with warning
            dataset_path = Path("data/processed/train.jsonl")
            if not Path(dataset_path).exists():
                alt = Path("C:/Aether/RisingStar/MIYUKI") / dataset_path
                if alt.exists():
                    dataset_path = alt
                else:
                    print(f"[warn] dataset {dataset_path} not found, creating synthetic smoke dataset for demo")
                    dataset_path = _get_smoke_dataset("data/processed/smoke_demo.jsonl")
            else:
                dataset_path = Path(dataset_path)

    dataset_path = Path(dataset_path)
    if not dataset_path.exists() and smoke:
        dataset_path = _get_smoke_dataset(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}. Provide --dataset or use --smoke.")

    print(f"[info] Training config: model={config.model.name} family={config.model.family}")
    print(f"[info] Dataset: {dataset_path}")
    print(f"[info] Output dir: {training_cfg.output_dir}")
    print(f"[info] Seed: {training_cfg.seed}, epochs={training_cfg.num_epochs}, batch={training_cfg.per_device_train_batch_size}")

    # Lazy imports - heavy
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from datasets import Dataset  # type: ignore

    # Tokenizer
    tok_id = config.model.resolved_tokenizer()
    tokenizer = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=config.model.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load raw examples
    raw = load_jsonl(dataset_path)
    print(f"[info] Loaded {len(raw)} examples from {dataset_path}")
    if len(raw) == 0:
        raise ValueError(f"Dataset {dataset_path} is empty")

    # Convert to conversational/text format and then tokenize for LM
    # Strategy: for SFT, we use chat template if available, else prompt+completion concatenation
    def to_text(ex: dict[str, Any]) -> str:
        fmt_ex = format_for_training(ex)
        if "messages" in fmt_ex:
            msgs = fmt_ex["messages"]
            # Use tokenizer chat template if available
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
                try:
                    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)  # type: ignore
                except Exception:
                    pass
            # Fallback: simple concatenation
            parts = []
            for m in msgs:
                role = m["role"]
                content = m["content"]
                if role == "system":
                    parts.append(f"System: {content}")
                elif role == "user":
                    parts.append(f"User: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
            return "\n\n".join(parts) + tokenizer.eos_token
        elif "prompt" in fmt_ex and "completion" in fmt_ex:
            return fmt_ex["prompt"] + " " + fmt_ex["completion"] + tokenizer.eos_token
        elif "text" in fmt_ex:
            return fmt_ex["text"] + tokenizer.eos_token
        else:
            return json.dumps(ex, ensure_ascii=False) + tokenizer.eos_token

    texts = [to_text(ex) for ex in raw]

    # Tokenize
    max_len = training_cfg.max_seq_length

    # Assistant-only loss handling (Phase 4): mask system/user tokens with -100
    # When assistant_only_loss is False (Phase0-3C), labels = input_ids for all tokens (old behavior)
    # When True, only assistant tokens are trainable
    if training_cfg.assistant_only_loss:
        print(f"[info] Using assistant-only loss (masking system/user, only assistant trainable)")
        # Create a Dataset with text and also raw index to map back to original example for masking
        dataset_dict = Dataset.from_dict({"text": texts, "_idx": list(range(len(texts)))})

        def tokenize_fn_masked(batch: dict[str, Any]) -> dict[str, Any]:
            input_ids_batch = []
            labels_batch = []
            attention_mask_batch = []
            for text, idx in zip(batch["text"], batch["_idx"]):
                raw_ex = raw[idx]
                # Get full tokenization
                full = tokenizer(text, truncation=True, max_length=max_len, padding=False, add_special_tokens=False)
                # For assistant-only, we need to find where assistant starts
                # Use chat template to get prefix (system+user + assistant start) length
                fmt_ex = format_for_training(raw_ex)
                if "messages" in fmt_ex and len(fmt_ex["messages"]) >= 2:
                    msgs = fmt_ex["messages"]
                    # Build prefix: all messages except last assistant, plus assistant start marker
                    # Use tokenizer's chat_template to get prefix text
                    try:
                        # Prefix is system+user (+ assistant start)
                        prefix_msgs = msgs[:-1]  # all except last assistant
                        prefix_text = tokenizer.apply_chat_template(prefix_msgs, tokenize=False, add_generation_prompt=True)
                        # Tokenize prefix to get its length (without special tokens)
                        prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
                        # But if truncation happened, we need to ensure prefix length fits within max_len
                        # Full ids may be truncated, so we mask min(prefix len, len(full))
                        full_ids = full["input_ids"]
                        # Find prefix length that matches start of full
                        # Since full = prefix + assistant_content + end, prefix_ids should be prefix of full_ids
                        # Handle truncation: if full was truncated, prefix may be longer than full
                        mask_len = min(len(prefix_ids), len(full_ids))
                        # Verify prefix matches start of full (check first mask_len tokens)
                        # If not, fallback to string search method
                        if full_ids[:mask_len] != prefix_ids[:mask_len]:
                            # Fallback: try to find assistant content tokenization inside full
                            # Use assistant content alone
                            assistant_content = msgs[-1]["content"]
                            # Find its tokenization inside full by searching for its first tokens
                            # Simplistic: mask everything before assistant content's first occurrence
                            # For now, use mask_len as before
                            pass
                        labels = [-100] * mask_len + full_ids[mask_len:]
                        # Special handling: if full was truncated and assistant is partially truncated, keep as is
                        # Ensure labels length matches input_ids
                        # Handle EOS: keep as trainable if in assistant part
                    except Exception as e:
                        print(f"[warn] assistant-only masking failed for idx {idx}: {e}, falling back to all trainable")
                        labels = full["input_ids"].copy()
                else:
                    # Fallback: not conversational, all trainable
                    labels = full["input_ids"].copy()
                input_ids_batch.append(full["input_ids"])
                labels_batch.append(labels)
                attention_mask_batch.append(full["attention_mask"])
            return {"input_ids": input_ids_batch, "labels": labels_batch, "attention_mask": attention_mask_batch}

        tokenized = dataset_dict.map(tokenize_fn_masked, batched=True, remove_columns=["text", "_idx"])
    else:
        dataset_dict = Dataset.from_dict({"text": texts})

        def tokenize_fn(batch: dict[str, Any]) -> dict[str, Any]:
            return tokenizer(batch["text"], truncation=True, max_length=max_len, padding=False)

        tokenized = dataset_dict.map(tokenize_fn, batched=True, remove_columns=["text"])
    # For causal LM, labels = input_ids
    # Trainer with DataCollatorForLanguageModeling will handle this, but we set labels here for simplicity
    def add_labels(batch: dict[str, Any]) -> dict[str, Any]:
        batch["labels"] = batch["input_ids"].copy()  # type: ignore
        return batch

    # Actually let DataCollator handle labels - simpler: just ensure input_ids present
    # We'll use DataCollatorForLanguageModeling with mlm=False

    # Split train/eval (90/10) if no eval dataset provided
    split = tokenized.train_test_split(test_size=0.1, seed=training_cfg.seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    # Model
    device, dtype = _resolve_device_dtype(config)
    model_kwargs: dict[str, Any] = {"trust_remote_code": config.model.trust_remote_code}
    # Only set torch_dtype for non-quantized
    if config.runtime.quantization not in ("4bit", "8bit"):
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(config.model.name, **model_kwargs)

    # PEFT
    peft_applied = False
    if config.adapter.enabled and config.adapter.type in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model, TaskType

        # Auto target modules if not specified
        target_modules = config.adapter.target_modules
        if target_modules is None:
            # sensible defaults per family
            family_defaults = {
                "gpt2": ["c_attn", "c_proj"],
                "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            }
            target_modules = family_defaults.get(config.model.family, ["c_attn", "c_proj"])

        lora_cfg = LoraConfig(
            r=config.adapter.r,
            lora_alpha=config.adapter.alpha,
            lora_dropout=config.adapter.dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        peft_applied = True
        print(f"[info] Applied LoRA r={config.adapter.r} alpha={config.adapter.alpha} targets={target_modules}")

    # Prepare output dir
    output_dir = Path(training_cfg.output_dir)
    # For smoke, ensure unique subdir if output_dir already exists with checkpoints?
    # We respect exactly what config says
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save training config snapshot for reproducibility
    (output_dir / "miyuki_training_config.yaml").write_text(
        __import__("yaml").safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    # Also save dataset info
    manifest_info = {
        "dataset_path": str(dataset_path),
        "num_examples": len(raw),
        "train_size": len(train_ds),
        "eval_size": len(eval_ds),
        "max_seq_length": max_len,
    }
    (output_dir / "dataset_info.json").write_text(json.dumps(manifest_info, indent=2), encoding="utf-8")

    # TrainingArguments
    # Handle bf16/fp16 logic
    use_bf16 = training_cfg.bf16 and device != "cpu"
    use_fp16 = training_cfg.fp16 and device != "cpu"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_cfg.num_epochs,
        per_device_train_batch_size=training_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=training_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=training_cfg.gradient_accumulation_steps,
        learning_rate=training_cfg.learning_rate,
        weight_decay=training_cfg.weight_decay,
        warmup_ratio=training_cfg.warmup_ratio,
        lr_scheduler_type=training_cfg.lr_scheduler_type,
        logging_steps=training_cfg.logging_steps,
        save_steps=training_cfg.save_steps,
        eval_steps=training_cfg.eval_steps,
        save_total_limit=training_cfg.save_total_limit,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=training_cfg.gradient_checkpointing,
        seed=training_cfg.seed,
        report_to=training_cfg.report_to,  # type: ignore
        # evaluation_strategy deprecated but needed for eval_steps to work; use eval_strategy
        eval_strategy="steps" if training_cfg.eval_steps is not None else "no",  # type: ignore
        save_strategy="steps",  # type: ignore
        load_best_model_at_end=False,
        push_to_hub=False,
    )

    # Choose collator based on assistant_only_loss: if True, we already have labels with -100 and must not overwrite
    if training_cfg.assistant_only_loss:
        # Custom collator that preserves existing labels (assistant-only) and pads with -100
        def assistant_collator(features):
            # features: list of dicts {input_ids, labels, attention_mask}
            max_len = max(len(f["input_ids"]) for f in features)
            batch_input_ids = []
            batch_attention = []
            batch_labels = []
            for f in features:
                pad_len = max_len - len(f["input_ids"])
                batch_input_ids.append(f["input_ids"] + [tokenizer.pad_token_id] * pad_len)
                batch_attention.append(f["attention_mask"] + [0] * pad_len)
                # labels already contains -100 for masked, pad with -100
                labs = f.get("labels", f["input_ids"])
                # Ensure labels length matches input_ids before padding
                if len(labs) != len(f["input_ids"]):
                    # fallback: if labels not provided, use input_ids
                    labs = f["input_ids"]
                batch_labels.append(labs + [-100] * pad_len)
            return {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
                "labels": torch.tensor(batch_labels, dtype=torch.long),
            }
        data_collator = assistant_collator
        print(f"[info] Using assistant-only collator (preserves -100 labels)")
    else:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,  # type: ignore
        eval_dataset=eval_ds,  # type: ignore
        data_collator=data_collator,  # type: ignore
    )

    # Train
    t0 = time.time()
    print("[info] Starting training...")
    # Compatibility fix (narrowly scoped): resuming a locally-generated MIYUKI checkpoint
    # on torch >= 2.6 (default weights_only=True) fails because Trainer's _load_rng_state
    # calls torch.load() without weights_only=False, and rng_state.pth contains NumPy RNG
    # state (arrays/dtypes). We allowlist ONLY the exact numpy globals required by that
    # trusted local file — determined empirically (see tools/diagnose_rng_state.py):
    #   numpy._core.multiarray._reconstruct, numpy.ndarray, numpy.dtype,
    #   numpy.dtypes.UInt32DType
    # This does NOT globally weaken PyTorch serialization security: arbitrary code
    # execution globals remain rejected; the allowlist is additive and invoked only on
    # the trusted local resume path (resume_from_checkpoint set). Normal checkpoint
    # loading elsewhere is unchanged. Checkpoint files themselves are never modified.
    if training_cfg.resume_from_checkpoint:
        _enable_local_checkpoint_rng_compat()
        print(f"[info] Resume compatibility: numpy RNG-state globals allowlisted for trusted local checkpoint {training_cfg.resume_from_checkpoint}")
    result = trainer.train(resume_from_checkpoint=training_cfg.resume_from_checkpoint)
    elapsed = time.time() - t0

    # Save final checkpoint (Trainer already saves)
    trainer.save_model(str(output_dir))
    if tokenizer:
        tokenizer.save_pretrained(str(output_dir))

    # Also save adapter explicitly if PEFT
    final_metrics = {
        "train_runtime_s": elapsed,
        "global_step": getattr(result, "global_step", None),
        "training_loss": getattr(result, "metrics", {}).get("train_runtime", None) or getattr(result, "metrics", {}).get("train_loss", None) if hasattr(result, "metrics") else None,
        "metrics": getattr(result, "metrics", {}),
        "elapsed_s": elapsed,
        "peft_applied": peft_applied,
    }
    # Try to get actual loss from trainer state
    try:
        hist = trainer.state.log_history
        # find last loss
        for entry in reversed(hist):
            if "loss" in entry:
                final_metrics["last_loss"] = entry["loss"]
                break
            if "train_loss" in entry:
                final_metrics["last_loss"] = entry["train_loss"]
                break
    except Exception:
        pass

    (output_dir / "train_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")

    # Record experiment
    try:
        from ..experiments.tracking import record_experiment
        from ..hardware import detect_hardware

        exp_id = record_experiment(
            model_name=config.model.name,
            dataset=str(dataset_path),
            dataset_version=config.version,
            training_config=config.to_dict(),
            hardware=detect_hardware(),
            duration_s=elapsed,
            final_loss=final_metrics.get("last_loss"),
            metrics=final_metrics,
            artifact_location=str(output_dir),
            notes=f"smoke={smoke} peft={peft_applied}",
        )
        print(f"[info] Experiment recorded: {exp_id}")
        final_metrics["experiment_id"] = exp_id
    except Exception as e:
        print(f"[warn] Failed to record experiment: {e}")

    # Save metadata for model artifact
    try:
        from ..artifacts.manager import save_model_metadata

        save_model_metadata(
            model_dir=output_dir,
            base_model=config.model.name,
            version=config.version,
            dataset=str(dataset_path),
            dataset_version=config.version,
            training_config=config.to_dict(),
            tokenizer_id=tok_id,
            context_length=config.model.context_length,
            quantization=config.runtime.quantization,
        )
    except Exception as e:
        print(f"[warn] Failed to save model metadata: {e}")

    print(f"[info] Training complete in {elapsed:.1f}s. Checkpoint at {output_dir}")
    if final_metrics.get("last_loss") is not None:
        print(f"[info] Last loss: {final_metrics['last_loss']}")
    return final_metrics
