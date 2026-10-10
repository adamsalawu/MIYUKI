"""Verify tokenization overhead for Phase 3C formulations."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except: pass
import json, pathlib
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def count_tokens_for_dataset(path):
    examples = [json.loads(l) for l in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    total=0
    per=[]
    max_t=0
    for ex in examples:
        msgs = ex["messages"]
        # Use same to_text logic as train.py: apply_chat_template with add_generation_prompt=False
        if hasattr(tok, "apply_chat_template") and tok.chat_template is not None:
            try:
                txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            except:
                txt = "\n\n".join(f"{m['role']}: {m['content']}" for m in msgs) + tok.eos_token
        else:
            txt = "\n\n".join(f"{m['role']}: {m['content']}" for m in msgs) + tok.eos_token
        ids = tok(txt, truncation=True, max_length=1024, padding=False)["input_ids"]
        total+=len(ids)
        per.append(len(ids))
        max_t = max(max_t, len(ids))
    return total, sum(per)/len(per) if per else 0, max_t, per

orig = "C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1/dataset.jsonl"
a = "C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1-phase3c/dataset-a.jsonl"
b = "C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1-phase3c/dataset-b.jsonl"

for name, path in [("original", orig), ("A-system", a), ("B-user", b)]:
    total, avg, max_t, per = count_tokens_for_dataset(path)
    print(f"{name:12s}: total {total}, avg {avg:.1f}, max {max_t}, examples {len(per)}")

# Compute increase
total_orig, avg_orig, max_orig, _ = count_tokens_for_dataset(orig)
total_a, avg_a, max_a, _ = count_tokens_for_dataset(a)
total_b, avg_b, max_b, _ = count_tokens_for_dataset(b)
print(f"\nIncrease A vs orig: total {(total_a-total_orig)/total_orig*100:.1f}%, avg +{avg_a-avg_orig:.1f}, max {max_orig}->{max_a} (+{max_a-max_orig})")
print(f"Increase B vs orig: total {(total_b-total_orig)/total_orig*100:.1f}%, avg +{avg_b-avg_orig:.1f}, max {max_orig}->{max_b} (+{max_b-max_orig})")
# Check not consuming most of 1024 window
print(f"\nMax tokens vs max_seq 1024: orig {max_orig}/1024 {(max_orig/1024*100):.1f}%, A {max_a}/1024 {(max_a/1024*100):.1f}%, B {max_b}/1024 {(max_b/1024*100):.1f}%")
if max_a > 900 or max_b > 900:
    print("WARNING: formulation consumes most of context window!")
else:
    print("OK: formulations do not consume most of window (all <900, <88% of 1024)")

# Also count contract tokens
from transformers import AutoTokenizer as AT
tok2 = AT.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=False)
sys_contract = "You are Miyuki, a direct story-writing assistant. Perform the requested writing task directly. Do not explain, announce, summarize, or discuss the task. Follow constraints, preserve character and context facts, do not invent dialogue when forbidden, continue the story rather than discussing it, and when rewriting output the rewritten text directly."
user_suffix = "\n\nRequirements: Perform the writing directly. Do not explain, announce, summarize, or discuss. Follow constraints, preserve context, do not invent dialogue when forbidden, continue story not discuss, when rewriting output directly."
print(f"\nSystem contract tokens: {len(tok2(sys_contract, add_special_tokens=False)['input_ids'])}")
print(f"User suffix tokens: {len(tok2(user_suffix, add_special_tokens=False)['input_ids'])}")
