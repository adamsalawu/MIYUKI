import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except: pass
import json, pathlib, time
from miyuki.config import MiyukiConfig
from miyuki.runtime.transformers_runtime import TransformersRuntime

cfg = MiyukiConfig.from_yaml("configs/training/phase4_lora.yaml")
cfg.generation.max_new_tokens=60
rt = TransformersRuntime(cfg)
print("loading...")
t0=time.time()
rt.load()
print(f"loaded in {time.time()-t0:.1f}s")

prompts_path = pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/prompts/phase4_expanded.jsonl")
prompts = [json.loads(l) for l in prompts_path.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"loaded {len(prompts)} prompts")

for i, item in enumerate(prompts):
    prompt = item.get("prompt","")
    instruction = item.get("instruction","")
    context = item.get("context","")
    full_prompt = prompt
    if instruction:
        full_prompt = f"{instruction}\n\n{prompt}"
    if context:
        full_prompt = f"Context: {context}\n\n{full_prompt}"
    print(f"\n--- {i+1}/{len(prompts)} {item['id']} {item['category']} ---")
    print(f"full_prompt len {len(full_prompt)}")
    t1=time.time()
    res = rt.generate(full_prompt)
    print(f"generated in {time.time()-t1:.1f}s tok {res.runtime_info['output_tokens']} tps {res.runtime_info['tokens_per_sec']}")
    print(f"preview: {res.generated[:150].replace(chr(10),' ')}")
    if i>=2:
        print("Stopping after 3 for test")
        break
print("done")
