"""Smoke test for Phase 3: Qwen + LoRA on Miyuki corpus (real path) — verifies 10 checks."""
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

print("=== Phase 3 Smoke Test ===")
# 1 tokenizer loads
tok_id = "Qwen/Qwen2-0.5B-Instruct"
print("1. tokenizer loads...")
tok = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print(f"   ok: vocab {len(tok)}, pad {tok.pad_token}, chat {bool(tok.chat_template)}")

# 2 model loads
print("2. model loads...")
model = AutoModelForCausalLM.from_pretrained(tok_id, trust_remote_code=False, torch_dtype="auto")
print(f"   ok: {model.config.model_type}, params {sum(p.numel() for p in model.parameters())}")

# 3 dataset loads
print("3. dataset loads...")
ds_path = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1/dataset.jsonl")
raw = [json.loads(l) for l in ds_path.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"   ok: {len(raw)} examples, SHA {hashlib.sha256(ds_path.read_bytes()).hexdigest()[:16]}")
# Use 4 examples for smoke
smoke_raw = raw[:4]
print(f"   smoke subset {len(smoke_raw)}")

# 4 chat formatting works
print("4. chat formatting...")
def to_text(ex):
    msgs = ex["messages"]
    if hasattr(tok, "apply_chat_template") and tok.chat_template is not None:
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    parts=[]
    for m in msgs:
        parts.append(f"{m['role']}: {m['content']}")
    return "\n\n".join(parts) + tok.eos_token

sample_text = to_text(smoke_raw[0])
print(f"   sample text len {len(sample_text)} chars, preview: {sample_text[:120].replace(chr(10),' | ')}")
assert "<|im_start|>" in sample_text or "user" in sample_text.lower(), "chat formatting failed"
print("   ok")

# 5 LoRA attaches
print("5. LoRA attach...")
target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
lora_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=target_modules, task_type=TaskType.CAUSAL_LM, bias="none")
model_peft = get_peft_model(model, lora_cfg)
print(f"   ok: LoRA r16 alpha32 on {target_modules}")

# 6 trainable params non-zero
print("6. trainable params...")
trainable = sum(p.numel() for p in model_peft.parameters() if p.requires_grad)
total = sum(p.numel() for p in model_peft.parameters())
pct = trainable/total*100 if total else 0
print(f"   trainable {trainable}, total {total}, pct {pct:.2f}%")
assert trainable > 0, "trainable params zero"
assert 0 < pct < 10, "unexpected pct"
print("   ok")

# 7 one/few training steps execute
print("7. training steps...")
texts = [to_text(ex) for ex in smoke_raw]
# tokenize
max_len=1024
ds = Dataset.from_dict({"text": texts})
def tok_fn(batch):
    return tok(batch["text"], truncation=True, max_length=max_len, padding=False)
tok_ds = ds.map(tok_fn, batched=True, remove_columns=["text"])
# No eval split for smoke: train on 4, eval on same 1
train_ds = tok_ds
eval_ds = tok_ds.select([0])

# Prepare output dir
import tempfile, pathlib as pl
out_dir = pl.Path("C:/Aether/RisingStar/MIYUKI/checkpoints/phase3-smoke")
out_dir.mkdir(parents=True, exist_ok=True)
import shutil
# clean previous
if (out_dir / "adapter_config.json").exists():
    shutil.rmtree(out_dir / "checkpoint-1", ignore_errors=True)

training_args = TrainingArguments(
    output_dir=str(out_dir),
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=0.0001,
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
    max_steps=2,  # just 2 optimizer steps
)
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)
trainer = Trainer(model=model_peft, args=training_args, train_dataset=train_ds, eval_dataset=eval_ds, data_collator=collator)
t0=time.time()
res = trainer.train()
elapsed=time.time()-t0
print(f"   train done in {elapsed:.1f}s, global_step {res.global_step}, loss {res.metrics.get('train_loss')}")
# check loss recorded
assert res.global_step > 0
# check log history has loss
hist = trainer.state.log_history
has_loss = any("loss" in h or "train_loss" in h for h in hist)
print(f"   has_loss {has_loss}")
assert has_loss

# 8 checkpoint saves
print("8. checkpoint saves...")
# Trainer already saves to out_dir and checkpoint-1 etc
import pathlib
ckpts = list(out_dir.glob("checkpoint-*"))
print(f"   checkpoints: {ckpts}")
assert len(ckpts) >=1 or (out_dir / "adapter_config.json").exists()
# Also check adapter files
assert (out_dir / "adapter_config.json").exists() or any((c/"adapter_config.json").exists() for c in ckpts), "no adapter"
print(f"   adapter_config exists: {(out_dir / 'adapter_config.json').exists()}")
trainer.save_model(str(out_dir))
print(f"   saved to {out_dir}")

# 9 adapter can be loaded again
print("9. adapter load...")
from peft import PeftModel
# Reload base model
base2 = AutoModelForCausalLM.from_pretrained(tok_id, trust_remote_code=False, torch_dtype="auto")
peft_model2 = PeftModel.from_pretrained(base2, str(out_dir))
print(f"   ok: loaded adapter from {out_dir}, is peft {type(peft_model2)}")

# 10 generation works with adapter
print("10. generation with adapter...")
# Use runtime abstraction to test generation via adapter path
# For quick test, use the peft model directly
from transformers import AutoTokenizer as AT2
tok2 = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=False)
if tok2.pad_token is None:
    tok2.pad_token = tok2.eos_token
prompt = "Context: Elena carries a chipped blue mug.\n\nWrite her reaction when offered lavender tea."
# Format via chat template
msgs = [{"role":"user","content": prompt}]
formatted = tok2.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tok2(formatted, return_tensors="pt")
# need to put model on cpu
peft_model2.eval()
inputs = {k: v for k,v in inputs.items()}
import torch
with torch.no_grad():
    out_ids = peft_model2.generate(**inputs, max_new_tokens=32, do_sample=True, temperature=0.8, top_p=0.95, pad_token_id=tok2.eos_token_id)
input_len = inputs["input_ids"].shape[1]
gen = tok2.decode(out_ids[0][input_len:], skip_special_tokens=True)
print(f"   generated: {gen[:200].replace(chr(10),' ')}")
assert len(gen.strip()) > 0, "empty generation"
print("   ok")

print("\n=== SMOKE TEST PASSED ===")
print(f"trainable {trainable} ({pct:.2f}%), steps {res.global_step}, loss {res.metrics.get('train_loss'):.4f} if any")
print(f"checkpoint {out_dir}, adapter load ok, generation ok")
# Also test base without adapter still loads
print("Base without adapter still loads...")
base_only = AutoModelForCausalLM.from_pretrained(tok_id, trust_remote_code=False, torch_dtype="auto")
print(f"   base only params {sum(p.numel() for p in base_only.parameters())} ok")
print("=== All 10 checks passed ===")
