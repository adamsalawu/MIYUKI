"""Manifest + extended validation for miyuki-writing-0.1"""
import json, pathlib, hashlib, collections, datetime
from transformers import AutoTokenizer

p = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1/dataset.jsonl")
out_manifest = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1/manifest.json")
tok_id = "Qwen/Qwen2-0.5B-Instruct"

examples = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"Loaded {len(examples)} examples")

# Extended validation: check required metadata fields, categories, duplicate IDs
allowed_cats = {"scene","dialogue","character_voice","character_consistency","context_retention","emotion","action","description","pacing","subtext","rewriting","instruction_following","anti_meta","transitional","atmospheric","quiet","character_focus","environmental","tension","humor","sensory_detail","continuity","continuation"}
errors=[]
for idx, e in enumerate(examples,1):
    md = e.get("metadata",{})
    for field in ["id","instruction","context","category","skills","source","notes","word_count"]:
        if field not in md:
            errors.append(f"{md.get('id','?')} missing {field}")
    # category validity
    for c in md.get("category",[]):
        if c not in allowed_cats:
            errors.append(f"{md['id']} invalid category {c}")
    # duplicate check done later
    # word_count check
    actual = len(e["messages"][1]["content"].split())
    if abs(actual - md.get("word_count",0)) > 2:
        errors.append(f"{md['id']} word_count mismatch {md.get('word_count')} vs {actual}")

if errors:
    print("EXTENDED VALIDATION ERRORS:")
    for er in errors: print(" ", er)
else:
    print("Extended validation: no errors")

# Tokenization compatibility
tok = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
char_counts=[]
tok_counts=[]
issues=[]
for e in examples:
    user = e["messages"][0]["content"]
    assistant = e["messages"][1]["content"]
    full = user + "\n" + assistant
    try:
        toks = tok(full, return_tensors=None)
        ids = toks["input_ids"]
        tok_counts.append(len(ids))
        char_counts.append(len(full))
    except Exception as ex:
        issues.append((e["metadata"]["id"], str(ex)))
print(f"Tokenization: total chars {sum(char_counts)}, total tokens {sum(tok_counts)}")
print(f"Avg tokens per example {sum(tok_counts)/len(tok_counts):.1f}, avg chars {sum(char_counts)/len(char_counts):.1f}")
if issues:
    print("tokenization issues", issues)
else:
    print("tokenization compatibility: ok")

# Checksums
def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

file_hash = sha256_file(p)
print(f"SHA256 dataset.jsonl {file_hash[:16]}...")

# Category counts for manifest
cats = collections.Counter()
for e in examples:
    for c in e["metadata"]["category"]:
        cats[c]+=1

manifest = {
    "name": "miyuki-writing-0.1",
    "version": "0.1.0",
    "description": "Human-authored Miyuki writing exemplars teaching PROMPT->WRITE not explain. 38 examples covering Phase1 failures.",
    "source": "human-authored",
    "license": "CC-BY-4.0",
    "creation_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "num_examples": len(examples),
    "categories": dict(cats),
    "skills": dict(collections.Counter(s for e in examples for s in e["metadata"]["skills"])),
    "splits": {"train": len(examples)},
    "characters": {"total_chars": sum(char_counts), "avg_chars_per_example": sum(char_counts)/len(char_counts)},
    "tokens": {"total_tokens_qwen": sum(tok_counts), "avg_tokens_per_example": sum(tok_counts)/len(tok_counts), "tokens_per_char": sum(tok_counts)/sum(char_counts) if sum(char_counts) else 0},
    "tokenizer": {"id": tok_id, "vocab_size": len(tok), "chat_template": bool(tok.chat_template)},
    "files": {"dataset.jsonl": {"sha256": file_hash, "num_examples": len(examples)}},
    "validation": {"status": "passed" if not errors else "failed", "errors": errors, "duplicates": 0 if len(set(e["metadata"]["id"] for e in examples))==len(examples) else 1},
    "provenance": {"author": "Miyuki Lab human-authored (AI-assisted drafting, human-edited)", "phase1_failures_addressed": ["meta-analysis","flat dialogue","lore dumping","quiet-scene failure","context hedging","action incoherence","premature EOS","repetition","exposition","voice drift"]},
    "notes": "First writing corpus. Quality over volume. Each example has keep/edit/discard reviewed in review.yaml. Before/after anti-meta pairs included."
}
out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote manifest to {out_manifest}")

# Also copy to data/manifests for versioning convention
import shutil
dest = pathlib.Path("C:/Aether/RisingStar/MIYUKI/data/manifests/miyuki-writing-0.1.json")
dest.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(out_manifest, dest)
print(f"Copied to {dest}")
