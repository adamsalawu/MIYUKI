"""Phase 4C 5-prompt targeted results (voice/binary/Jinx/object/viewpoint)."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import json, pathlib
p = pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase4c-miyuki-0.3/targeted_eval.jsonl")
for line in p.read_text(encoding="utf-8").splitlines():
    o = json.loads(line)
    print("=" * 100)
    print(o["id"], o["category"], f"toks={o['runtime']['output_tokens']}")
    print(o["generated"])
    print()
