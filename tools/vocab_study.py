"""Vocabulary study for miyuki-writing-0.1 using Qwen/Qwen2-0.5B-Instruct tokenizer."""
import sys, io
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import json, pathlib, collections, re
from transformers import AutoTokenizer

tok_id = "Qwen/Qwen2-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=False)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

dataset_path = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1/dataset.jsonl")
examples = [json.loads(l) for l in dataset_path.read_text(encoding="utf-8").splitlines() if l.strip()]

# Gather all text (user + assistant)
all_texts = []
for e in examples:
    all_texts.append(e["messages"][0]["content"])
    all_texts.append(e["messages"][1]["content"])
corpus = "\n".join(all_texts)

total_chars = sum(len(t) for t in all_texts)
total_tokens = sum(len(tok(t)["input_ids"]) for t in all_texts)
total_words = sum(len(t.split()) for t in all_texts)
print(f"Total texts: {len(all_texts)} (38 pairs -> 76 msgs)")
print(f"Total chars: {total_chars}, words: {total_words}, tokens: {total_tokens}")
print(f"Tokens/char: {total_tokens/total_chars:.4f}, tokens/word: {total_tokens/total_words:.3f}, chars/token: {total_chars/total_tokens:.2f}")
print(f"Vocab size: {len(tok)}, chat_template: {bool(tok.chat_template)}")
print(f"Pad: {tok.pad_token} ({tok.pad_token_id}), EOS: {tok.eos_token} ({tok.eos_token_id})")
print(f"Additional special tokens: {getattr(tok, 'additional_special_tokens', [])}")

# Per example stats
per_example_tokens=[]
per_example_chars=[]
for e in examples:
    txt = e["messages"][0]["content"] + "\n" + e["messages"][1]["content"]
    c=len(txt)
    t=len(tok(txt)["input_ids"])
    per_example_tokens.append(t)
    per_example_chars.append(c)
print(f"Per example tokens: min {min(per_example_tokens)}, max {max(per_example_tokens)}, avg {sum(per_example_tokens)/len(per_example_tokens):.1f}")
print(f"Per example chars: min {min(per_example_chars)}, max {max(per_example_chars)}, avg {sum(per_example_chars)/len(per_example_chars):.1f}")

# Tokenization of specific categories: dialogue, punctuation, names, invented terms
def tokenize_word(word):
    # tokenization without added special tokens
    ids = tok(word, add_special_tokens=False)["input_ids"]
    toks = tok.convert_ids_to_tokens(ids)
    return ids, toks

# Names and invented terms
test_words = [
    "Mara","Jinx","Elena","Haru","Mika","Bren","Toma","Lira","Kira","Aoi","Ken",
    "Aureline","Kestrel","lumenweave","scryglass","voidtide","Emberhearth","genkan",
    "\"","—","…","\"", "'","...","--"
]
# Also recurring phrases
phrases = [
    "The lighthouse had been dark for twenty years",
    "chipped blue mug",
    "broken compass",
    "She said, glancing away",
    "“Oh, great,” she said",
    "lumenweave cloak",
    "voidtide flats"
]

print("\n--- Word tokenization ---")
for w in test_words:
    ids, toks = tokenize_word(w)
    print(f"{w:20s} -> {len(ids):2d} toks {toks}")

print("\n--- Phrase tokenization ---")
for p in phrases:
    ids = tok(p, add_special_tokens=False)["input_ids"]
    print(f"{p:45s} -> {len(ids):2d} toks")

# Dialogue punctuation fragmentation
dialogue_samples = [e["messages"][1]["content"] for e in examples if "dialogue" in e["metadata"]["category"]]
print(f"\n--- Dialogue tokenization efficiency (sample) ---")
for s in dialogue_samples[:3]:
    # take first 200 chars
    snippet = s[:200]
    ids = tok(snippet)["input_ids"]
    print(f"Snippet: {snippet[:80].replace(chr(10),' ')} -> {len(ids)} toks, chars {len(snippet)}, t/c {len(ids)/len(snippet):.3f}")

# Punctuation-heavy vs prose
punct_tests = {
    "em dash": "She didn't run—she stood on the shingle.",
    "ellipsis": "He watched... then turned.",
    "quotes": "\"Oh, great,\" she said, eyeing the broken compass.",
    "contraction": "She didn't know what to say. It's cold.",
}
print("\n--- Punctuation ---")
for name, txt in punct_tests.items():
    ids = tok(txt, add_special_tokens=False)["input_ids"]
    toks = tok.convert_ids_to_tokens(ids)
    print(f"{name:15s} {txt[:40]:45s} -> {len(ids):2d} toks {toks[:8]}")

# Frequency of invented terms
counter=collections.Counter()
for txt in all_texts:
    words=re.findall(r"\b\w+\b", txt)
    for w in words:
        counter[w]+=1
# Show recurring Miyuki terms
miyuki_terms=["Aureline","Kestrel","lumenweave","scryglass","voidtide","Mara","Jinx","Elena","Haru","Mika","Toma","Lira","genkan"]
print("\n--- Miyuki term frequencies ---")
for term in miyuki_terms:
    freq = counter[term]
    ids, toks = tokenize_word(term)
    print(f"{term:15s} freq {freq:2d} -> {len(ids):2d} toks {toks}")

# Unusual fragmentation check: show most fragmented words (tokens >1)
fragmented=[]
for word, freq in counter.items():
    if freq>=2 and len(word)>=4:
        ids, toks = tokenize_word(word)
        if len(ids)>2:
            fragmented.append((word, freq, len(ids), toks))
fragmented=sorted(fragmented, key=lambda x: -x[2])
print("\n--- Most fragmented frequent words (>=2 chars, freq>=2) ---")
for w,f,n,toks in fragmented[:20]:
    print(f"{w:20s} freq {f:2d} -> {n} toks {toks}")

# Save report
report = {
    "tokenizer_id": tok_id,
    "vocab_size": len(tok),
    "chat_template": bool(tok.chat_template),
    "pad_token": tok.pad_token,
    "eos_token": tok.eos_token,
    "total_examples": len(examples),
    "total_messages": len(all_texts),
    "total_chars": total_chars,
    "total_words": total_words,
    "total_tokens": total_tokens,
    "tokens_per_char": total_tokens/total_chars,
    "tokens_per_word": total_tokens/total_words,
    "avg_tokens_per_example": sum(per_example_tokens)/len(per_example_tokens),
    "min_tokens_per_example": min(per_example_tokens),
    "max_tokens_per_example": max(per_example_tokens),
    "test_words": {w: {"tokens": tokenize_word(w)[1], "num": len(tokenize_word(w)[0])} for w in test_words},
    "phrases": {p: len(tok(p, add_special_tokens=False)["input_ids"]) for p in phrases},
    "miyuki_terms": {t: {"freq": counter[t], "tokens": tokenize_word(t)[1], "num": len(tokenize_word(t)[0])} for t in miyuki_terms},
    "fragmented_frequent": [{"word":w,"freq":f,"num":n,"tokens":toks} for w,f,n,toks in fragmented[:20]],
}
out = pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase2_vocabulary/tokenizer_report.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved report to {out}")

# Also save readable md
md = f"""# Phase 2 Vocabulary Study — Qwen2-0.5B-Instruct on miyuki-writing-0.1

- **Tokenizer:** {tok_id}, vocab {len(tok)}, chat_template {bool(tok.chat_template)}
- **Corpus:** {len(examples)} examples, {len(all_texts)} messages, {total_chars} chars, {total_words} words, {total_tokens} tokens
- **Efficiency:** {total_tokens/total_chars:.4f} tok/char, {total_tokens/total_words:.3f} tok/word, {total_chars/total_tokens:.2f} chars/token
- **Per example:** {sum(per_example_tokens)/len(per_example_tokens):.1f} avg toks (min {min(per_example_tokens)}, max {max(per_example_tokens)})

## Test words

""" 
for w in test_words:
    ids, toks = tokenize_word(w)
    md+=f"- `{w}` → {len(ids)} toks `{toks}`\n"
md+="\n## Miyuki term frequencies\n"
for term in miyuki_terms:
    ids,toks=tokenize_word(term)
    md+=f"- `{term}` freq {counter[term]} → {len(ids)} toks `{toks}`\n"
md+="\n## Most fragmented frequent words\n"
for w,f,n,toks in fragmented[:15]:
    md+=f"- `{w}` freq {f} → {n} `{toks}`\n"
md_out = pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase2_vocabulary/tokenizer_report.md")
md_out.write_text(md, encoding="utf-8")
print(f"Saved md to {md_out}")
