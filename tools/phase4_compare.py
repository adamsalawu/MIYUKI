"""Phase 4 comparison: Base vs 3B vs 4 on original 12 prompts (read-only, fast)."""
import json, pathlib, re, collections

def load(p):
    return [json.loads(l) for l in pathlib.Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]

base = load("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase1_qwen2_0.5b/outputs.jsonl")
p3b = load("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase3b-miyuki-0.1/outputs.jsonl")
p4 = load("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase4-miyuki-0.2/2026-09-25_Qwen_Qwen2-0.5B-Instruct/outputs.jsonl")

def stats(outputs):
    total_out = sum(o["generated"].__len__() for o in outputs)
    # token counts from runtime if present
    toks = []
    tps = []
    for o in outputs:
        rt = o.get("runtime", {})
        toks.append(rt.get("output_tokens", len(o["generated"].split())))
        tps.append(rt.get("tokens_per_sec", 0))
    return {
        "num": len(outputs),
        "total_chars": total_out,
        "avg_chars": total_out/len(outputs),
        "total_tokens": sum(toks),
        "avg_tokens": sum(toks)/len(toks),
        "avg_tps": sum(tps)/len(tps) if tps else 0,
    }

def meta_flags(text):
    t = text.lower()
    return {
        "the_answer_is": "the answer is" in t,
        "here_is_scene": "here is a scene" in t or "here is your scene" in t,
        "as_ai": "as an ai" in t,
        "tone_described": "the literary fiction tone is described" in t or "the narrator's voice is soft and calm" in t,
        "hedging": ("could be one of" in t or "it depends" in t or "might be one of" in t),
        "chinese": bool(re.search(r'[\u4e00-\u9fff]', text)),
        "premature_eos": len(text.split()) < 15,
    }

for name, data in [("base", base), ("phase3b", p3b), ("phase4", p4)]:
    print(f"=== {name} ===")
    print(stats(data))
    # meta counts
    c = collections.Counter()
    for o in data:
        f = meta_flags(o["generated"])
        for k,v in f.items():
            if v: c[k]+=1
    print("meta counts", dict(c))
    print()

# Detailed per-prompt comparison
for i in range(len(base)):
    pid = base[i]["id"]
    print("="*100)
    print(pid, base[i]["category"])
    for name, data in [("base", base), ("3B", p3b), ("P4", p4)]:
        g = data[i]["generated"]
        preview = g[:220].replace("\n"," | ")
        f = meta_flags(g)
        print(f"{name:6s} len={len(g):4d} toks={data[i].get('runtime',{}).get('output_tokens','?')} flags={[k for k,v in f.items() if v]} -> {preview}")

# Overfitting: check phase4 outputs vs dataset 0.2 for 40-char verbatim
ds_path = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.2/dataset.jsonl")
targets = [json.loads(l)["messages"][1]["content"] for l in ds_path.read_text(encoding="utf-8").splitlines() if l.strip()]
print("\n=== Overfitting check Phase4 ===")
for o in p4:
    gen = o["generated"]
    found=False
    for tgt in targets:
        for j in range(0, len(tgt)-40, 20):
            snip = tgt[j:j+40].strip()
            if len(snip)<30: continue
            if snip in gen:
                print(f"{o['id']} MEMORIZED: {snip[:60]}")
                found=True
                break
        if found: break
    if not found:
        print(f"{o['id']}: no 40-char verbatim")
# invented terms
for o in p4:
    txt=o["generated"].lower()
    terms=[t for t in ["lumenweave","scryglass","voidtide","aureline","kestrel"] if t in txt]
    if terms:
        print(f"{o['id']} invented terms: {terms}")
print("done invented check")

# Save comparison
out = pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase4-miyuki-0.2/comparison.json")
out.parent.mkdir(parents=True, exist_ok=True)
comp = {
    "base_stats": stats(base),
    "phase3b_stats": stats(p3b),
    "phase4_stats": stats(p4),
}
out.write_text(json.dumps(comp, indent=2), encoding="utf-8")
print(f"saved {out}")
