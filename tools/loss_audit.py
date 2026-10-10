"""Loss audit for Phase 4: which tokens currently receive labels and contribute to loss."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except: pass
import json, pathlib
from transformers import AutoTokenizer

tok_id = "Qwen/Qwen2-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# Load one real example from miyuki-writing-0.1
dataset_path = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1/dataset.jsonl")
ex = json.loads(pathlib.Path(dataset_path).read_text(encoding="utf-8").splitlines()[0])
print("Example ID:", ex["metadata"]["id"])
print("Messages:", json.dumps(ex["messages"], indent=2, ensure_ascii=False)[:800])

# Current training pipeline's to_text (as in train.py)
def to_text_current(ex):
    from miyuki.datasets.loader import format_for_training
    fmt = format_for_training(ex)
    msgs = fmt["messages"]
    if hasattr(tok, "apply_chat_template") and tok.chat_template is not None:
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception as e:
            print("chat template error", e)
    # fallback
    parts=[]
    for m in msgs:
        parts.append(f"{m['role']}: {m['content']}")
    return "\n\n".join(parts) + tok.eos_token

text_current = to_text_current(ex)
print("\n=== Current to_text (full conversation) ===")
print(text_current[:1000].replace("\n","\\n\n"))

# Tokenize as train.py does
tokens = tok(text_current, truncation=True, max_length=1024, padding=False)
input_ids = tokens["input_ids"]
print(f"\nTotal tokens: {len(input_ids)}")
print(f"First 20 tokens: {tok.convert_ids_to_tokens(input_ids[:20])}")
print(f"Last 20 tokens: {tok.convert_ids_to_tokens(input_ids[-20:])}")

# Simulate DataCollatorForLanguageModeling behavior (labels = input_ids)
# This is what train.py currently does: collator sets labels = input_ids for all tokens
labels_current = input_ids.copy()
print(f"\nCurrent labels (DataCollatorForLanguageModeling): labels = input_ids for ALL {len(labels_current)} tokens")
print(f"Number of -100 in current labels: {sum(1 for l in labels_current if l==-100)} (should be 0)")
print(f"SYSTEM/USER/ASSISTANT all contribute to loss? YES (all tokens have labels)")

# Now check what TRL assistant-only would do vs our current
# For Qwen chat template, we need to find where assistant starts
# The template is: <|im_start|>system ...<|im_end|> <|im_start|>user ...<|im_end|> <|im_start|>assistant ...<|im_end|>
# We can try to find assistant token sequence
assistant_text = ex["messages"][1]["content"]
# Find assistant start in text_current
# The chat template adds <|im_start|>assistant {content}<|im_end|>
# So we can find the substring
assistant_start_marker = "<|im_start|>assistant"
idx = text_current.find(assistant_start_marker)
if idx != -1:
    before = text_current[:idx]
    after = text_current[idx:]
    before_tokens = tok(before, add_special_tokens=False)["input_ids"]
    after_tokens = tok(after, add_special_tokens=False)["input_ids"]
    # But after includes the assistant start marker + content + end
    # The whole text is before + after, but tokenization may differ due to BPE boundaries
    # Simpler: we can show that current labels include before (system/user) tokens
    print(f"\n=== Intended assistant-only ===")
    print(f"Before assistant marker chars: {len(before)}, after chars: {len(after)}")
    print(f"Before token count (approx, add_special_tokens False): {len(before_tokens)}")
    print(f"After token count: {len(after_tokens)}")
    print(f"Total current tokens: {len(input_ids)}, before+after approx: {len(before_tokens)+len(after_tokens)} (close but not exact due to tokenization boundaries)")
    print(f"With assistant-only masking, {len(before_tokens)} tokens would be masked (-100), {len(after_tokens)} would be trainable")
    print(f"Current behavior: 0 masked, {len(input_ids)} trainable")
    print(f"Correct behavior (TRL assistant-only): {len(before_tokens)} masked, {len(after_tokens)} trainable")
else:
    print("Could not find assistant marker in text_current")

# Check special tokens
print(f"\n=== Special tokens ===")
print(f"pad_token {tok.pad_token} ({tok.pad_token_id}), eos {tok.eos_token} ({tok.eos_token_id}), all_special {tok.all_special_tokens}")
# In current labels, special tokens are also trainable (since labels = input_ids)
# Check if any special token ids are in labels and would contribute to loss
special_ids = set(tok.all_special_ids)
special_in_labels = [i for i in input_ids if i in special_ids]
print(f"Special token ids in current input_ids: {special_in_labels[:10]} count {len(special_in_labels)}")
print(f"With assistant-only, special tokens before assistant would be masked, those in/after assistant would be trainable? Depends")
print(f"Current: special tokens ARE trainable (since all labels)")

# Summary report for one example
print("\n" + "="*60)
print("SUMMARY FOR ONE REAL TRAINING EXAMPLE (miyuki-0.1-001)")
print("="*60)
print(f"SYSTEM TOKENS       → {'included (trainable)' if len(input_ids)>0 else 'unknown'} — CURRENT: ALL tokens trainable, so system included")
print(f"USER TOKENS         → included — CURRENT: ALL")
print(f"ASSISTANT TOKENS    → included — CURRENT: ALL")
print(f"SPECIAL TOKENS      → included — CURRENT: ALL (e.g., <|im_start|>, <|im_end|> are trainable)")
print(f"TRL assistant-only would mask system+user, only assistant trainable — MIYUKI DOES NOT CURRENTLY DO THIS")
print(f"Implication: Model is penalized for not predicting user/system prompts, not just assistant writing.")

# Save diagnostic artifact
out = pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase4_loss_audit.json")
out.parent.mkdir(parents=True, exist_ok=True)
import datetime, json as js
report = {
    "example_id": ex["metadata"]["id"],
    "total_tokens": len(input_ids),
    "before_tokens_approx": len(before_tokens) if idx!=-1 else None,
    "after_tokens_approx": len(after_tokens) if idx!=-1 else None,
    "current_labels_all_trainable": True,
    "current_masked_minus100": 0,
    "system_included": True,
    "user_included": True,
    "assistant_included": True,
    "special_included": True,
    "trl_assistant_only_would_mask": len(before_tokens) if idx!=-1 else None,
    "trl_assistant_only_trainable": len(after_tokens) if idx!=-1 else None,
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "tokenizer": tok_id,
    "conclusion": "MIYUKI currently trains on ALL tokens (system+user+assistant+special), not assistant-only. Needs fix."
}
out.write_text(js.dumps(report, indent=2), encoding="utf-8")
print(f"\nSaved diagnostic to {out}")
