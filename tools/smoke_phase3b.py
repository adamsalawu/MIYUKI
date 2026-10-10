"""Smoke test for Phase 3B: Qwen + LoRA r16 lr 2e-4 epochs 3 config — verifies 10 checks quickly."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import json, pathlib, time, hashlib
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

print("=== Phase 3B Smoke Test (lr 2e-4, r16) ===")
tok_id = "Qwen/Qwen2-0.5B-Instruct"
print("1. tokenizer...")
tok = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print(f"   vocab {len(tok)}, chat {bool(tok.chat_template)}")
print("2. model...")
model = AutoModelForCausalLM.from_pretrained(tok_id, trust_remote_code=False, torch_dtype="auto")
print(f"   params {sum(p.numel() for p in model.parameters())}")
print("3. dataset...")
ds_path = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1/dataset.jsonl")
raw = [json.loads(l) for l in ds_path.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"   {len(raw)} ex SHA {hashlib.sha256(ds_path.read_bytes()).hexdigest()[:16]}")
smoke_raw = raw[:4]
print("4. chat formatting...")
def to_text(ex):
    msgs = ex["messages"]
    if hasattr(tok, "apply_chat_template") and tok.chat_template is not None:
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in msgs) + tok.eos_token
sample = to_text(smoke_raw[0])
assert "<|im_start|>" in sample
print(f"   ok len {len(sample)}")
print("5. LoRA r16 alpha32 lr2e-4...")
lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], task_type=TaskType.CAUSAL_LM, bias="none")
model_peft = get_peft_model(model, lora_cfg)
print("   ok")
print("6. trainable...")
trainable = sum(p.numel() for p in model_peft.parameters() if p.requires_grad)
total = sum(p.numel() for p in model_peft.parameters())
pct = trainable/total*100
print(f"   trainable {trainable} total {total} pct {pct:.2f}%")
assert trainable==8798208, f"expected 8798208 got {trainable}"
assert 1 < pct < 10
print("   ok 1.75% as expected")
print("7. training steps (max_steps 2, lr 2e-4)...")
texts = [to_text(ex) for ex in smoke_raw]
ds = Dataset.from_dict({"text": texts})
def tok_fn(batch):
    return tok(batch["text"], truncation=True, max_length=1024, padding=False)
tok_ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
train_ds = tok_ds
eval_ds = tok_ds.select([0])
out_dir = pathlib.Path("C:/Aether/RisingStar/MIYUKI/checkpoints/phase3b-smoke")
out_dir.mkdir(parents=True, exist_ok=True)
import shutil
if (out_dir / "adapter_config.json").exists():
    shutil.rmtree(out_dir, ignore_errors=True)
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
t0=time.time()
res = trainer.train()
print(f"   done {time.time()-t0:.1f}s global_step {res.global_step} loss {res.metrics.get('train_loss'):.4f}")
assert res.global_step==2
print("8. checkpoint...")
import pathlib as pl
ckpts = list(out_dir.glob("checkpoint-*"))
print(f"   ckpts {ckpts}")
assert len(ckpts)>=1 or (out_dir / "adapter_config.json").exists()
trainer.save_model(str(out_dir))
print(f"   saved to {out_dir}")
print("9. adapter reload...")
from peft import PeftModel
base2 = AutoModelForCausalLM.from_pretrained(tok_id, trust_remote_code=False, torch_dtype="auto")
peft2 = PeftModel.from_pretrained(base2, str(out_dir))
print(f"   ok {type(peft2)}")
print("10. generation...")
tok2 = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=False)
if tok2.pad_token is None:
    tok2.pad_token = tok2.eos_token
prompt = "Context: Elena carries a chipped blue mug.\n\nWrite her reaction when offered lavender tea."
msgs = [{"role":"user","content": prompt}]
formatted = tok2.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok2(formatted, return_tensors="pt")
peft2.eval()
import torch
with torch.no_grad():
    out_ids = peft2.generate(**inputs, max_new_tokens=32, do_sample=True, temperature=0.8, top_p=0.95, pad_token_id=tok2.eos_token_id)
gen = tok2.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"   gen: {gen[:200].replace(chr(10),' ')}")
assert len(gen.strip())>0
print("   ok")
print("\n=== SMOKE 3B PASSED ===")
print(f"trainable {trainable} pct {pct:.2f}%, steps 2, lr 2e-4")
print("Base without adapter still loads...")
base_only = AutoModelForCausalLM.from_pretrained(tok_id, trust_remote_code=False, torch_dtype="auto")
print(f"   base params {sum(p.numel() for p in base_only.parameters())} ok")
