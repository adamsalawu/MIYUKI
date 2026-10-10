"""Phase 4C expanded-12 vs Phase 4 expanded-12 (apples-to-apples, read-only)."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import json, pathlib, re, collections

def load(p):
    return [json.loads(l) for l in pathlib.Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]

p4e = load("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase4-expanded-miyuki-0.2-60/2026-09-25_Qwen_Qwen2-0.5B-Instruct/outputs.jsonl")
p4ce = load("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase4c-expanded12-miyuki-0.3/2026-09-29_Qwen_Qwen2-0.5B-Instruct/outputs.jsonl")

def flags(t):
    tl = t.lower()
    return {k for k, v in {
        "answer_is": "the answer is" in tl,
        "here_is_scene": "here is a scene" in tl or "here is your scene" in tl,
        "meta_announce": ("quietly, i begin" in tl or "the narrator begins" in tl),
        "chinese": bool(re.search(r'[\u4e00-\u9fff]', t)),
        "short_eos": len(t.split()) < 15,
    }.items() if v}

def stats(d):
    toks = [o.get("runtime", {}).get("output_tokens", 0) for o in d]
    return {"n": len(d), "toks": sum(toks), "avg": round(sum(toks)/len(toks), 1)}

print("P4 expanded (60-token budget):", stats(p4e))
print("P4C expanded (180-token budget):", stats(p4ce))
print()
# map by id — both suites use same ids (phase4_expanded.jsonl)
ids4 = {o["id"]: o for o in p4e}
for o in p4ce:
    other = ids4.get(o["id"])
    print("=" * 100)
    print(o["id"], o["category"])
    if other:
        f4 = sorted(flags(other["generated"]))
        f4c = sorted(flags(o["generated"]))
        print(f"P4  tok{other['runtime']['output_tokens']:>4} flags={f4} -> {other['generated'][:200].replace(chr(10), ' | ')}")
    print(f"P4C tok{o['runtime']['output_tokens']:>4} flags={f4c} -> {o['generated'][:200].replace(chr(10), ' | ')}")

out = pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase4c-expanded12-miyuki-0.3/comparison.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"phase4_expanded_60tok": stats(p4e), "phase4c_expanded_180tok": stats(p4ce)}, indent=2), encoding="utf-8")
print(f"\nsaved {out}")
