"""Extract exact unsupported-global names from weights_only load errors."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import re
import torch

rng = r"C:\Aether\RisingStar\MIYUKI\checkpoints\phase4c-miyuki-0.3\checkpoint-30\rng_state.pth"

def unsupported_global():
    try:
        torch.load(rng, weights_only=True)
        return None
    except Exception as e:
        m = re.search(r"Unsupported global: GLOBAL ([\w.]+)", str(e))
        if m:
            return m.group(1)
        m = re.search(r"but got <class '([\w.]+)'>", str(e))
        if m:
            return m.group(1)
        return f"NO-MATCH: {str(e)[-400:]}"

import numpy as np
needed = []
for i in range(12):
    g = unsupported_global()
    if g is None:
        print(f"\nLOAD SUCCEEDED after {len(needed)} allowlisted globals")
        break
    if g.startswith("NO-MATCH"):
        print("Unexpected error:", g)
        break
    print(f"[{i}] Unsupported global: {g}")
    # resolve the dotted name to an object
    parts = g.split(".")
    obj = np
    try:
        for p in parts[1:]:
            obj = getattr(obj, p)
        needed.append(obj)
        torch.serialization.add_safe_globals([obj])
        print(f"    -> allowlisted {g}")
    except AttributeError as e:
        print(f"    -> cannot resolve: {e}")
        break
print("\nNeeded globals:", [type(x).__module__ + "." + type(x).__name__ if not isinstance(x, type) else x.__module__ + "." + x.__name__ for x in needed])
