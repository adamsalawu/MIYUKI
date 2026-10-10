"""Tokenizer experiment: train 3 experimental vocab sizes from miyuki-writing-0.1 corpus.
Does NOT modify production Qwen model or tokenizer. Saves to artifacts/tokenizers/miyuki-experimental/
Compares tokenization efficiency.
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import json, pathlib, datetime, hashlib, time
from transformers import AutoTokenizer

base_tok_id = "Qwen/Qwen2-0.5B-Instruct"
base_tok = AutoTokenizer.from_pretrained(base_tok_id, trust_remote_code=False)
print(f"Base tokenizer: {base_tok_id}, vocab {len(base_tok)}, chat_template {bool(base_tok.chat_template)}")

dataset_path = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1/dataset.jsonl")
examples = [json.loads(l) for l in dataset_path.read_text(encoding="utf-8").splitlines() if l.strip()]
# Build iterator of texts (user + assistant) for tokenizer training
texts = []
for e in examples:
    texts.append(e["messages"][0]["content"])
    texts.append(e["messages"][1]["content"])
# Repeat to give tokenizer more material (BPE needs frequency); duplicate 10x to get enough tokens
expanded = texts * 10  # 760 texts

def iterator():
    for t in expanded:
        yield t

# Vocab sizes to test: small, medium, larger — all smaller than Qwen 151k to test efficiency for English fiction
# Chosen based on typical fiction tokenizers (GPT-2 50k) and limited corpus size.
# Small 8192 may be too small; medium 16000, large 32000 are sensible for 5k-token corpus.
vocab_sizes = [8192, 16000, 32000]
print(f"Training {len(vocab_sizes)} experimental tokenizers from corpus iterator (len {len(expanded)})")

# Check if train_new_from_iterator exists
has_train = hasattr(base_tok, "train_new_from_iterator")
print(f"Has train_new_from_iterator: {has_train}")
if not has_train:
    print("Fallback: will use tokenizers library directly (not AutoTokenizer)")
    # We'll still try via base_tok

results = []
base_out = pathlib.Path("C:/Aether/RisingStar/MIYUKI/artifacts/tokenizers/miyuki-experimental")
base_out.mkdir(parents=True, exist_ok=True)

# For reproducibility, record config
import platform, sys as _sys
repro = {
    "python_version": _sys.version,
    "platform": platform.platform(),
    "transformers": __import__("transformers").__version__,
    "tokenizers": __import__("tokenizers").__version__ if __import__("importlib").util.find_spec("tokenizers") else None,
    "base_tokenizer": base_tok_id,
    "base_vocab": len(base_tok),
    "dataset": str(dataset_path),
    "dataset_examples": len(examples),
    "vocab_sizes": vocab_sizes,
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}

# Train each
for vs in vocab_sizes:
    print(f"\n--- Training vocab {vs} ---")
    t0=time.time()
    try:
        # Use base tokenizer's train_new_from_iterator
        new_tok = base_tok.train_new_from_iterator(iterator(), vocab_size=vs)
        # new_tok will have same type but new vocab
        elapsed = time.time()-t0
        print(f"Trained {vs} in {elapsed:.1f}s, new vocab {len(new_tok)}")
        # Save
        save_dir = base_out / f"miyuki-vocab-{vs}"
        save_dir.mkdir(parents=True, exist_ok=True)
        new_tok.save_pretrained(str(save_dir))
        # Also save tokenizer.json if available
        # Verify by encoding a sample
        sample = "The lighthouse had been dark for twenty years. Mara walked with her chipped blue mug."
        ids_base = base_tok(sample, add_special_tokens=False)["input_ids"]
        ids_new = new_tok(sample, add_special_tokens=False)["input_ids"]
        print(f"Sample base {len(ids_base)} toks vs new {len(ids_new)}")
        results.append({"vocab_size": vs, "new_vocab_actual": len(new_tok), "elapsed_s": elapsed, "save_dir": str(save_dir), "sample_base": len(ids_base), "sample_new": len(ids_new), "success": True})
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed to train {vs}: {e}")
        results.append({"vocab_size": vs, "success": False, "error": str(e)})

# Now compare all tokenizers on corpus
print("\n--- Comparison on corpus ---")
comparisons=[]
for res in results:
    if not res["success"]:
        continue
    vs=res["vocab_size"]
    new_tok = AutoTokenizer.from_pretrained(str(base_out / f"miyuki-vocab-{vs}"), trust_remote_code=False)
    total_base=0
    total_new=0
    total_chars=0
    for t in texts:
        total_chars+=len(t)
        total_base+=len(base_tok(t, add_special_tokens=False)["input_ids"])
        total_new+=len(new_tok(t, add_special_tokens=False)["input_ids"])
    comp={
        "vocab_size": vs,
        "total_chars": total_chars,
        "base_tokens": total_base,
        "new_tokens": total_new,
        "base_tok_per_char": total_base/total_chars if total_chars else 0,
        "new_tok_per_char": total_new/total_chars if total_chars else 0,
        "compression": (total_base-total_new)/total_base if total_base else 0,  # positive means new is fewer tokens (more efficient)
        "expansion": (total_new-total_base)/total_base if total_base else 0,
        "new_vs_base_ratio": total_new/total_base if total_base else 0,
    }
    comparisons.append(comp)
    print(f"vocab {vs}: base {total_base} vs new {total_new} (ratio {comp['new_vs_base_ratio']:.3f}, compression {comp['compression']:.2%})")

# Also compute per example avg
# Save reports
report_base = pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase2_vocabulary")
report_base.mkdir(parents=True, exist_ok=True)
import json as js
(report_base/"tokenizer_experiment.json").write_text(js.dumps({"repro": repro, "train_results": results, "comparisons": comparisons}, indent=2), encoding="utf-8")
print(f"Saved experiment to {report_base/'tokenizer_experiment.json'}")

# Create markdown
md = f"""# Tokenizer Experiment — miyuki-experimental

- **Base:** {base_tok_id} vocab {len(base_tok)}
- **Corpus:** {len(examples)} examples, {total_chars if 'total_chars' in locals() else 'n/a'} chars on full corpus (actually {sum(len(t) for t in texts)} chars for unique texts)
- **Vocab sizes tested:** {vocab_sizes}
- **Trained:** {sum(1 for r in results if r['success'])}/{len(vocab_sizes)}
- **Timestamp:** {repro['timestamp']}

## Results

"""
for c in comparisons:
    md+=f"- **vocab {c['vocab_size']}:** base {c['base_tokens']} toks vs new {c['new_tokens']} (ratio {c['new_vs_base_ratio']:.3f}, compression {c['compression']:+.2%} — negative means more tokens than base)\n"

md+="\n## Interpretation\n"
md+="- Compression positive means experimental tokenizer uses fewer tokens (more efficient) than Qwen base on this English fiction corpus.\n"
md+="- All experimental vocabs are **smaller** than Qwen's 151646 (multilingual). If smaller English-focused vocab is more efficient, it suggests Qwen's large multilingual vocab is not inefficient for English prose.\n"
md+="- These tokenizers were trained on only 38 examples (~5k tokens, expanded 10x) — a tiny sample; results are indicative, not definitive. A larger Miyuki corpus would be needed for stable BPE.\n"
md+="- **Compatibility:** Experimental tokenizers are NOT connected to production Qwen model (embedding matrix mismatch would require `resize_token_embeddings` + retraining). Phase 2 does not do this.\n"

(report_base/"tokenizer_experiment.md").write_text(md, encoding="utf-8")
print(f"Saved md")

for res in results:
    print(res)
