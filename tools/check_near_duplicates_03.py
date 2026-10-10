"""Near-duplicate check for 0.3: normalized assistant text overlap."""
import json, pathlib, re
from difflib import SequenceMatcher
path = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.3/dataset.jsonl")
examples = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
def norm(s):
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()
texts = [(e["metadata"]["id"], norm(e["messages"][1]["content"])) for e in examples]
worst = []
for i in range(len(texts)):
    for j in range(i+1, len(texts)):
        r = SequenceMatcher(None, texts[i][1], texts[j][1]).ratio()
        if r > 0.75:
            worst.append((texts[i][0], texts[j][0], round(r, 3)))
print(f"Pairs with ratio > 0.75: {len(worst)}")
for w in worst[:20]:
    print(w)
# Also check IDs unique
ids = [e["metadata"]["id"] for e in examples]
print(f"IDs unique: {len(ids)==len(set(ids))} ({len(ids)})")
# Check first 120 match 0.2 byte-for-byte (messages + metadata)
p02 = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.2/dataset.jsonl")
lines02 = p02.read_text(encoding="utf-8").splitlines()
lines03 = path.read_text(encoding="utf-8").splitlines()
print(f"First 120 identical to 0.2: {lines03[:120]==lines02}")
