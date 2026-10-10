"""Overfitting check for Phase 3C A and B."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except: pass
import json, pathlib, collections

dataset_targets = [json.loads(l)["messages"][1]["content"] for l in pathlib.Path("datasets/miyuki-writing-0.1/dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

for label, eval_path in [
    ("Phase3C-A", "C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase3c-a-miyuki-0.1/2026-09-25_Qwen_Qwen2-0.5B-Instruct/outputs.jsonl"),
    ("Phase3C-B", "C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase3c-b-miyuki-0.1/2026-09-25_Qwen_Qwen2-0.5B-Instruct/outputs.jsonl"),
]:
    print(f"\n=== {label} ===")
    evals = [json.loads(l) for l in pathlib.Path(eval_path).read_text(encoding="utf-8").splitlines()]
    # Memorization: 40-char verbatim
    found=False
    for e in evals:
        gen = e["generated"]
        for tgt in dataset_targets:
            for i in range(0, len(tgt)-40, 20):
                snippet = tgt[i:i+40].strip()
                if len(snippet)<30: continue
                if snippet in gen:
                    print(f"{e['id']} memorized {snippet[:50]}")
                    found=True
                    break
            if found: break
    if not found:
        print("No verbatim 40-char memorization")
    # Invented term injection
    term_counts = collections.Counter()
    for e in evals:
        txt = e["generated"].lower()
        for term in ["lumenweave","scryglass","voidtide","aureline"]:
            if term in txt:
                term_counts[term]+=1
    print(f"Invented term injection: {dict(term_counts) if term_counts else 'none'}")
    # Check generation outside categories: factual
    from miyuki.config import MiyukiConfig
    from miyuki.runtime.transformers_runtime import TransformersRuntime
    # Load the appropriate adapter
    cfg_path = "configs/training/phase3c_a_lora.yaml" if "A" in label else "configs/training/phase3c_b_lora.yaml"
    cfg = MiyukiConfig.from_yaml(cfg_path)
    cfg.generation.max_new_tokens=60
    rt = TransformersRuntime(cfg)
    rt.load()
    res = rt.generate("Explain why the sky is blue in one short paragraph.")
    print(f"Factual test: {res.generated[:200].replace(chr(10),' ')}")
    print(f"Is factual? {('scatter' in res.generated.lower() or 'Rayleigh' in res.generated)}")
    # Check for meta behavior in eval
    meta_count = sum(1 for e in evals if "the answer is" in e["generated"].lower() or "the literary" in e["generated"].lower() or "the narrator's voice" in e["generated"].lower())
    print(f"Meta-answer count in eval: {meta_count}/12")

# Also check for repeated structures: check if many eval outputs are very short (<30 toks) indicating premature EOS
for label, eval_path in [
    ("Phase3C-A", "C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase3c-a-miyuki-0.1/2026-09-25_Qwen_Qwen2-0.5B-Instruct/outputs.jsonl"),
    ("Phase3C-B", "C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase3c-b-miyuki-0.1/2026-09-25_Qwen_Qwen2-0.5B-Instruct/outputs.jsonl"),
]:
    evals = [json.loads(l) for l in pathlib.Path(eval_path).read_text(encoding="utf-8").splitlines()]
    short = sum(1 for e in evals if e["runtime"]["output_tokens"] < 30)
    print(f"{label} short (<30 toks): {short}/12")
