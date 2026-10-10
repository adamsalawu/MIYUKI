"""Overfitting check for Phase 3: test prompts semantically related but not exact copies, look for memorization."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import json, pathlib, re
from transformers import AutoTokenizer
from miyuki.config import MiyukiConfig
from miyuki.runtime.transformers_runtime import TransformersRuntime

# Load dataset assistant targets for copy check
dataset_path = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1/dataset.jsonl")
dataset_targets = [json.loads(l)["messages"][1]["content"] for l in dataset_path.read_text(encoding="utf-8").splitlines() if l.strip()]
dataset_text = "\n".join(dataset_targets)

# Overfitting test prompts: paraphrases of training examples, not in eval set
paraphrases = [
    ("Haru tea paraphrase", "Context: Haru, 34, writer blocked for months in Kyoto. Instruction: Write a quiet scene of him making tea as rain falls, sensory detail. Prompt: The kettle clicked."),
    ("Elena mug paraphrase", "Context: Elena always carries her chipped blue mug from grandmother, hates lavender. Instruction: Write her ordering coffee at a new cafe where barista offers lavender latte. Prompt: The barista smiled and held up a small purple sachet."),
    ("Jinx compass paraphrase", "Context: Jinx, 24, witty scout, broken compass. Instruction: Continue her voice. Prompt: She tossed the broken compass in the air and caught it."),
    ("Aureline lumenweave paraphrase", "Context: Aureline workshop with lumenweave ribbons marking safe voidtide paths. Instruction: Describe a character tying a ribbon. Prompt: He crouched to tie the lumenweave."),
]

print("=== Overfitting Check: Paraphrased prompts ===")
cfg = MiyukiConfig.from_yaml("configs/training/phase3_lora.yaml")
cfg.generation.max_new_tokens = 80
cfg.generation.temperature = 0.8
cfg.generation.top_p = 0.95

rt = TransformersRuntime(cfg)
rt.load()
print(f"Loaded adapter? {'Peft' in str(type(rt._model))}")

for name, prompt in paraphrases:
    res = rt.generate(prompt)
    gen = res.generated
    # Check for copied phrases: look for 20-char exact substrings from dataset in generation
    copied = []
    for target in dataset_targets:
        # Use 30-char windows
        for i in range(0, len(target)-30, 20):
            snippet = target[i:i+30]
            if len(snippet.strip()) < 20:
                continue
            if snippet.strip() in gen:
                copied.append(snippet.strip()[:40])
                break
    # Check for repeated names overfitting
    names = ["Mara","Jinx","Elena","Haru","Mika","Bren","Toma","Lira","Aureline","lumenweave"]
    name_counts = {n: gen.count(n) for n in names}
    overfit_names = {k:v for k,v in name_counts.items() if v>2}
    # Check for excessive imitation of individual example structure (e.g., "She didn't run. She stood" from dataset 001)
    print(f"\n--- {name} ---")
    print(f"Prompt: {prompt[:100]}")
    print(f"Generated ({len(gen)} chars, {res.runtime_info['output_tokens']} toks): {gen[:400].replace(chr(10),' ')}")
    print(f"Copied 30-char snippets from dataset: {copied[:2] if copied else 'none'}")
    print(f"Name counts over 2: {overfit_names if overfit_names else 'none'}")
    # Check instruction following: should be scene, not meta
    is_meta = gen.strip().lower().startswith("the answer is") or "this scene demonstrates" in gen.lower() or "here is a scene" in gen.lower()
    print(f"Meta-answer? {is_meta}")
    # Check general instruction following vs degradation
    is_empty = len(gen.strip()) < 20
    print(f"Empty/degraded? {is_empty}")

# Also check for memorization by searching Phase3 eval outputs for exact dataset phrases
print("\n=== Memorization in eval outputs ===")
eval_path = pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase3-miyuki-0.1/2026-09-24_Qwen_Qwen2-0.5B-Instruct/outputs.jsonl")
evals = [json.loads(l) for l in eval_path.read_text(encoding="utf-8").splitlines()]
for e in evals:
    gen = e["generated"]
    # Check if any 40-char snippet from dataset appears verbatim
    found = False
    for target in dataset_targets:
        for i in range(0, len(target)-40, 20):
            snippet = target[i:i+40].strip()
            if len(snippet) < 30:
                continue
            if snippet in gen:
                print(f"{e['id']} contains memorized snippet: {snippet[:60]}")
                found=True
                break
        if found:
            break
    if not found:
        print(f"{e['id']}: no verbatim 40-char memorization")

print("\n=== Check for repeated phrases across eval outputs (stylistic overfitting) ===")
# Count if same invented term appears in many eval outputs (would be overfitting if model injects lumenweave where not requested)
from collections import Counter
term_counts = Counter()
for e in evals:
    txt = e["generated"].lower()
    for term in ["lumenweave","scryglass","voidtide","aureline"]:
        if term in txt:
            term_counts[term]+=1
print(f"Invented term injection in eval (should be low unless prompt contains them): {dict(term_counts)}")

print("\n=== General instruction following outside trained categories ===")
# Test a prompt outside trained categories: simple factual
prompt = "Explain why the sky is blue in one short paragraph."
res = rt.generate(prompt)
print(f"Prompt: {prompt}")
print(f"Generated: {res.generated[:400].replace(chr(10),' ')}")
print(f"Is factual answer (good) vs scene (bad)? Should be factual — check if model still does scene vs answer")
