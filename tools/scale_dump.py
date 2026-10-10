"""Dump full 1.5B outputs for human review (no truncation)."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except: pass
import json, pathlib
p = pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/runs/model-scale-qwen2.5-1.5b/outputs.jsonl")
for line in p.read_text(encoding="utf-8").splitlines():
    o = json.loads(line)
    print("="*100)
    print(o["id"], o["category"], f"toks={o['runtime']['output_tokens']} tps={o['runtime']['tokens_per_sec']}")
    print("GENERATED:")
    print(o["generated"])
    print()
