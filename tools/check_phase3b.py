"""Check Phase3B: round-trip Base, Phase3, Phase3B and overfitting."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except: pass
import json, pathlib
from miyuki.config import MiyukiConfig
from miyuki.runtime.transformers_runtime import TransformersRuntime

def generate_with(cfg_path, prompt, max_new_tokens=60, note=""):
    cfg = MiyukiConfig.from_yaml(cfg_path)
    cfg.generation.max_new_tokens = max_new_tokens
    rt = TransformersRuntime(cfg)
    rt.load()
    is_peft = "Peft" in str(type(rt._model))
    res = rt.generate(prompt)
    print(f"{note} is_peft={is_peft} tok={res.runtime_info['prompt_tokens']} out={res.runtime_info['output_tokens']} tps={res.runtime_info['tokens_per_sec']}")
    print(f"  gen: {res.generated[:200].replace(chr(10),' ')}")
    print(f"  chat {res.runtime_info['chat_template_used']} not_random {'atedton' not in res.generated}")
    return res

print("=== Round-trip Base ===")
r_base = generate_with("configs/model/small-instruct.yaml", "The lighthouse had been dark for twenty years, until tonight. When Mara saw the beam sweep across the water, she knew", note="BASE")
print("\n=== Phase3 (0002) ===")
r_p3 = generate_with("configs/training/phase3_lora.yaml", "The lighthouse had been dark for twenty years, until tonight. When Mara saw the beam sweep across the water, she knew", note="PHASE3")
print("\n=== Phase3B (0003) ===")
r_p3b = generate_with("configs/training/phase3b_lora.yaml", "The lighthouse had been dark for twenty years, until tonight. When Mara saw the beam sweep across the water, she knew", note="PHASE3B")

print("\n=== Compare outputs distinct? ===")
print("base vs p3 same100?", r_base.generated[:100]==r_p3.generated[:100])
print("p3 vs p3b same100?", r_p3.generated[:100]==r_p3b.generated[:100])
print("base vs p3b same100?", r_base.generated[:100]==r_p3b.generated[:100])

# Overfitting checks for Phase3B
print("\n=== Overfitting checks Phase3B ===")
dataset_path = pathlib.Path("datasets/miyuki-writing-0.1/dataset.jsonl")
targets = [json.loads(l)["messages"][1]["content"] for l in dataset_path.read_text(encoding="utf-8").splitlines() if l.strip()]
# Test paraphrased prompts with Phase3B
cfg = MiyukiConfig.from_yaml("configs/training/phase3b_lora.yaml")
cfg.generation.max_new_tokens = 80
rt = TransformersRuntime(cfg)
rt.load()
paraphrases = [
    ("Haru tea", "Context: Haru, 34, writer blocked for months in Kyoto. Instruction: Write a quiet scene of him making tea as rain falls. Prompt: The kettle clicked."),
    ("Elena mug", "Context: Elena always carries her chipped blue mug from grandmother, hates lavender. Instruction: Write her ordering coffee at a new cafe where barista offers lavender latte. Prompt: The barista smiled and held up a small purple sachet."),
]
for name, prompt in paraphrases:
    res = rt.generate(prompt)
    gen = res.generated
    copied = []
    for tgt in targets:
        for i in range(0, len(tgt)-30, 20):
            snippet = tgt[i:i+30].strip()
            if len(snippet) < 20: continue
            if snippet in gen:
                copied.append(snippet[:40])
                break
    print(f"{name}: copied {copied[:1] if copied else 'none'}, len {len(gen)}, meta? {gen.strip().lower().startswith('the answer is')}")
# Check eval memorization for Phase3B (will be done after eval, but quick check for previous Phase3 eval)
import pathlib as pl
eval_p3 = pl.Path("evaluation/runs/phase3-miyuki-0.1/2026-09-24_Qwen_Qwen2-0.5B-Instruct/outputs.jsonl")
if eval_p3.exists():
    evals = [json.loads(l) for l in eval_p3.read_text(encoding="utf-8").splitlines()]
    from collections import Counter
    term_counts = Counter()
    for e in evals:
        txt = e["generated"].lower()
        for term in ["lumenweave","scryglass","voidtide","aureline"]:
            if term in txt:
                term_counts[term]+=1
    print(f"Phase3 invented term injection: {dict(term_counts)}")

# Test general instruction outside categories
print("\n=== General instruction test Phase3B ===")
res = rt.generate("Explain why the sky is blue in one short paragraph.")
print(f"Gen: {res.generated[:300].replace(chr(10),' ')}")
print(f"Is factual (contains 'Rayleigh' or 'scatter')? {('scatter' in res.generated.lower() or 'Rayleigh' in res.generated)}")

print("\n=== All checks done ===")
