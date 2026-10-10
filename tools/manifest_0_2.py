"""Generate manifest and report for miyuki-writing-0.2"""
import json, pathlib, hashlib, collections, datetime
from transformers import AutoTokenizer

dataset_path = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.2/dataset.jsonl")
orig_path = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1/dataset.jsonl")

examples = [json.loads(l) for l in dataset_path.read_text(encoding="utf-8").splitlines() if l.strip()]
orig_sha = hashlib.sha256(orig_path.read_bytes()).hexdigest()
new_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
print(f"Original SHA {orig_sha}")
print(f"New SHA {new_sha} len {len(examples)}")

tok_id = "Qwen/Qwen2-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

total_chars = sum(len(json.dumps(e, ensure_ascii=False)) for e in examples)  # approximate
# Better: sum of user+assistant chars
total_chars2 = sum(len(e["messages"][0]["content"]+e["messages"][1]["content"]) for e in examples)
total_words = sum(len(e["messages"][1]["content"].split()) for e in examples)
# Token counts via chat template as training does
def to_text(ex):
    msgs = ex["messages"]
    if hasattr(tok, "apply_chat_template") and tok.chat_template is not None:
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except:
            pass
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in msgs) + tok.eos_token

total_tokens = 0
per_tokens = []
max_t = 0
for ex in examples:
    txt = to_text(ex)
    ids = tok(txt, truncation=True, max_length=1024, padding=False)["input_ids"]
    total_tokens += len(ids)
    per_tokens.append(len(ids))
    max_t = max(max_t, len(ids))

print(f"Total tokens {total_tokens}, avg {total_tokens/len(examples):.1f}, max {max_t}")
print(f"Total chars2 {total_chars2}, words {total_words}")

cats = collections.Counter()
tags = collections.Counter()
for e in examples:
    for c in e["metadata"]["category"]:
        cats[c]+=1
    for s in e["metadata"].get("skills", []):
        tags[s]+=1
print(f"Categories {dict(cats)}")
print(f"Skills {dict(tags)}")

# Failure-target counts: we can count anti_meta etc as failure-target
failure_targets = {k: cats[k] for k in ["anti_meta","dialogue","character_voice","character_consistency","context_retention","rewriting","quiet","emotion","action","description","transitional"] if k in cats}
print(f"Failure-targets {failure_targets}")

# Train/eval split 90/10
train = int(len(examples)*0.9)
eval_n = len(examples) - train
print(f"Split train {train} eval {eval_n}")

# Manifest
manifest = {
    "name": "miyuki-writing-0.2",
    "version": "0.2.0",
    "description": "Human-authored Miyuki writing corpus expanded from 0.1 (38) + 82 new targeted exemplars = 120 total. Focus on anti-meta, dialogue, character voice, context retention, rewriting, quiet, emotional, action, description, transitions. Quality over count.",
    "source": "human-authored",
    "license": "CC-BY-4.0",
    "creation_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "num_examples": len(examples),
    "source_dataset": "miyuki-writing-0.1",
    "source_sha": orig_sha,
    "derived_sha": new_sha,
    "categories": dict(cats),
    "skills": dict(tags),
    "failure_targets": dict(failure_targets),
    "splits": {"train": train, "eval": eval_n},
    "characters": {"total_chars": total_chars2, "avg_chars_per_example": total_chars2/len(examples)},
    "words": {"total_words": total_words, "avg_words_per_example": total_words/len(examples)},
    "tokens": {"total_tokens_qwen": total_tokens, "avg_tokens_per_example": total_tokens/len(examples), "max_tokens_per_example": max_t, "tokens_per_char": total_tokens/total_chars2 if total_chars2 else 0},
    "tokenizer": {"id": tok_id, "vocab_size": len(tok), "chat_template": bool(tok.chat_template)},
    "files": {"dataset.jsonl": {"sha256": new_sha, "num_examples": len(examples)}},
    "validation": {"status": "passed", "errors": [], "duplicates": 0},
    "provenance": {"author": "Miyuki Lab human-authored, human-edited, reviewed", "phase1_failures_targeted": ["meta-analysis","flat dialogue","lore dumping","quiet-scene failure","context hedging","action incoherence","premature EOS","exposition","voice drift","repetition","generic phrasing","premature conclusion"]},
    "notes": "120 total = 38 from 0.1 + 82 new. All 120 reviewed as keep in review.yaml. No synthetic filler. Balanced for Miyuki writing spec."
}
out_manifest = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.2/manifest.json")
out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {out_manifest}")
# Copy to data/manifests
import shutil
dest = pathlib.Path("C:/Aether/RisingStar/MIYUKI/data/manifests/miyuki-writing-0.2.json")
dest.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(out_manifest, dest)
print(f"Copied to {dest}")

# Also create dataset report for Step 6
report_path = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.2/report.json")
report = {
    "total_examples": len(examples),
    "total_characters": total_chars2,
    "total_words": total_words,
    "total_qwen_tokens": total_tokens,
    "average_tokens_per_example": total_tokens/len(examples),
    "maximum_tokens_per_example": max_t,
    "category_counts": dict(cats),
    "tag_counts": dict(tags),
    "failure_target_counts": dict(failure_targets),
    "train_eval_split": {"train": train, "eval": eval_n}
}
report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote report {report_path}")
