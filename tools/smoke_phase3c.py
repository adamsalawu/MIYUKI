"""Smoke test both Phase3C formulations A and B."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except: pass
import json, pathlib, time, hashlib
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

def smoke_one(dataset_path, out_dir_name, label):
    print(f"\n=== Smoke {label}: {dataset_path} ===")
    tok_id = "Qwen/Qwen2-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"1. tok vocab {len(tok)} chat {bool(tok.chat_template)}")
    model = AutoModelForCausalLM.from_pretrained(tok_id, trust_remote_code=False, torch_dtype="auto")
    print(f"2. model {sum(p.numel() for p in model.parameters())}")
    raw = [json.loads(l) for l in pathlib.Path(dataset_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"3. dataset {len(raw)} SHA {hashlib.sha256(pathlib.Path(dataset_path).read_bytes()).hexdigest()[:8]}")
    smoke_raw = raw[:4]
    def to_text(ex):
        msgs = ex["messages"]
        if hasattr(tok, "apply_chat_template") and tok.chat_template is not None:
            try:
                return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            except:
                pass
        return "\n\n".join(f"{m['role']}: {m['content']}" for m in msgs) + tok.eos_token
    sample = to_text(smoke_raw[0])
    print(f"4. chat format len {len(sample)} has_system? {'system' in sample.lower() or '<|im_start|>system' in sample}")
    lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], task_type=TaskType.CAUSAL_LM, bias="none")
    model_peft = get_peft_model(model, lora_cfg)
    print(f"5. LoRA r16")
    trainable = sum(p.numel() for p in model_peft.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model_peft.parameters())
    print(f"6. trainable {trainable} pct {trainable/total*100:.2f}%")
    assert trainable==8798208
    texts = [to_text(ex) for ex in smoke_raw]
    ds = Dataset.from_dict({"text": texts})
    def tok_fn(b):
        return tok(b["text"], truncation=True, max_length=1024, padding=False)
    tok_ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
    train_ds = tok_ds
    eval_ds = tok_ds.select([0])
    out_dir = pathlib.Path(f"C:/Aether/RisingStar/MIYUKI/checkpoints/{out_dir_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    if (out_dir / "adapter_config.json").exists():
        shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=0.0002,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_steps=1,
        eval_steps=1,
        save_total_limit=1,
        seed=42,
        report_to="none",
        eval_strategy="steps",
        save_strategy="steps",
        bf16=False, fp16=False,
        gradient_checkpointing=False,
        max_steps=2,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
    trainer = Trainer(model=model_peft, args=training_args, train_dataset=train_ds, eval_dataset=eval_ds, data_collator=collator)
    print("7. training 2 steps...")
    t0=time.time()
    res = trainer.train()
    print(f"   done {time.time()-t0:.1f}s step {res.global_step} loss {res.metrics.get('train_loss'):.4f}")
    assert res.global_step==2
    print("8. checkpoint...")
    ckpts = list(out_dir.glob("checkpoint-*"))
    print(f"   ckpts {ckpts}")
    trainer.save_model(str(out_dir))
    assert (out_dir / "adapter_config.json").exists()
    print(f"   saved to {out_dir}")
    print("9. reload...")
    from peft import PeftModel
    base2 = AutoModelForCausalLM.from_pretrained(tok_id, trust_remote_code=False, torch_dtype="auto")
    peft2 = PeftModel.from_pretrained(base2, str(out_dir))
    print(f"   ok {type(peft2)}")
    print("10. generate...")
    tok2 = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=False)
    if tok2.pad_token is None:
        tok2.pad_token = tok2.eos_token
    # Use a Phase3C-style prompt with contract to verify adapter active
    prompt = "Context: Elena carries a chipped blue mug.\n\nRequirements: Perform the writing directly. Do not explain.\n\nWrite her reaction when offered lavender tea."
    msgs = [{"role":"user","content": prompt}]
    # For A, system would be separate, but for smoke we just test user prompt
    formatted = tok2.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok2(formatted, return_tensors="pt")
    peft2.eval()
    import torch
    with torch.no_grad():
        out_ids = peft2.generate(**inputs, max_new_tokens=32, do_sample=True, temperature=0.8, top_p=0.95, pad_token_id=tok2.eos_token_id)
    gen = tok2.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"   gen: {gen[:150].replace(chr(10),' ')}")
    assert len(gen.strip())>0
    print(f"=== Smoke {label} PASSED ===")
    return res

# Run both
smoke_one("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1-phase3c/dataset-a.jsonl", "phase3c-a-smoke", "A-system")
smoke_one("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1-phase3c/dataset-b.jsonl", "phase3c-b-smoke", "B-user")

print("\n=== BOTH SMOKES PASSED ===")
