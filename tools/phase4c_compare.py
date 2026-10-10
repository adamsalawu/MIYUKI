"""Phase 4C vs Phase 4 comparison: original 12 (read-only, fast)."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import json, pathlib, re, collections

def load(p):
    return [json.loads(l) for l in pathlib.Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]

p4 = load("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase4-miyuki-0.2/2026-09-25_Qwen_Qwen2-0.5B-Instruct/outputs.jsonl")
p4c = load("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase4c-miyuki-0.3/2026-09-29_Qwen_Qwen2-0.5B-Instruct/outputs.jsonl")
p1 = load("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase1_qwen2_0.5b/outputs.jsonl")

def flags(t):
    tl = t.lower()
    return {k for k, v in {
        "answer_is": "the answer is" in tl,
        "tone_essay": "the literary fiction tone is described" in tl or "the narrator's voice is soft and calm" in tl,
        "hedge": ("could be one of" in tl or "it depends" in tl or "might be one of" in tl),
        "chinese": bool(re.search(r'[\u4e00-\u9fff]', t)),
        "short_eos": len(t.split()) < 15,
    }.items() if v}

def stats(d):
    toks = [o.get("runtime", {}).get("output_tokens", 0) for o in d]
    tps = [o.get("runtime", {}).get("tokens_per_sec", 0) for o in d]
    return {"n": len(d), "toks": sum(toks), "avg": sum(toks)/len(toks), "tps": round(sum(tps)/len(tps), 2)}

for n, d in [("P1", p1), ("P4", p4), ("P4C", p4c)]:
    s = stats(d)
    c = collections.Counter()
    for o in d:
        for f in flags(o["generated"]):
            c[f] += 1
    print(f"{n}: {s} flags={dict(c)}")
print()
for i in range(len(p4)):
    print("=" * 100)
    print(p4[i]["id"], p4[i]["category"])
    for n, d in [("P4 ", p4), ("P4C", p4c)]:
        g = d[i]["generated"]
        print(f"{n} tok{d[i].get('runtime',{}).get('output_tokens','?'):>4} flags={sorted(flags(g))} -> {g[:230].replace(chr(10), ' | ')}")

out = pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase4c-miyuki-0.3/comparison.json")
out.write_text(json.dumps({"phase4": stats(p4), "phase4c": stats(p4c), "phase1": stats(p1)}, indent=2), encoding="utf-8")
print(f"\nsaved {out}")
