"""Compare 1.5B vs Phase1/3B/4 on original 12 (read-only)."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except: pass
import json, pathlib, re, collections
def load(p):
    return [json.loads(l) for l in pathlib.Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
base = load("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase1_qwen2_0.5b/outputs.jsonl")
p3b = load("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase3b-miyuki-0.1/outputs.jsonl")
p4 = load("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase4-miyuki-0.2/2026-09-25_Qwen_Qwen2-0.5B-Instruct/outputs.jsonl")
new = load("C:/Aether/RisingStar/MIYUKI/evaluation/runs/model-scale-qwen2.5-1.5b/outputs.jsonl")
def stats(d):
    toks=[o.get("runtime",{}).get("output_tokens",0) for o in d]
    tps=[o.get("runtime",{}).get("tokens_per_sec",0) for o in d]
    return {"n":len(d),"toks":sum(toks),"avg":sum(toks)/len(toks),"tps":sum(tps)/len(tps)}
def flags(t):
    tl=t.lower()
    return {k for k,v in {
        "answer_is":"the answer is" in tl,
        "tone_essay":"the literary fiction tone is described" in tl or "the narrator's voice is soft and calm" in tl,
        "hedge":("could be one of" in tl or "it depends" in tl or "might be one of" in tl),
        "refusal":("i'm sorry, but i'm not able" in tl or "as an ai" in tl),
        "chinese":bool(re.search(r'[\u4e00-\u9fff]',t)),
        "short":len(t.split())<15,
    }.items() if v}
for n,d in [("base",base),("3B",p3b),("P4",p4),("1.5B",new)]:
    s=stats(d)
    c=collections.Counter()
    for o in d:
        for f in flags(o["generated"]): c[f]+=1
    print(f"{n}: toks={s['toks']} avg={s['avg']:.1f} tps={s['tps']:.2f} flags={dict(c)}")
    print()
for i in range(len(base)):
    print("="*100)
    print(base[i]["id"], base[i]["category"])
    for n,d in [("base",base),("3B",p3b),("P4",p4),("1.5B",new)]:
        g=d[i]["generated"]
        print(f"{n:6s} tok{d[i].get('runtime',{}).get('output_tokens','?'):>4} flags={sorted(flags(g))} -> {g[:180].replace(chr(10),' ')}")
# save comparison
out=pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/runs/model-scale-qwen2.5-1.5b/comparison.json")
out.write_text(json.dumps({k:stats(v) for k,v in [("base",base),("phase3b",p3b),("phase4",p4),("qwen15b",new)]},indent=2),encoding="utf-8")
print(f"saved {out}")
