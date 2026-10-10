import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except: pass
import time
from miyuki.config import MiyukiConfig
from miyuki.runtime.transformers_runtime import TransformersRuntime
cfg = MiyukiConfig.from_yaml("configs/training/phase4_lora.yaml")
cfg.generation.max_new_tokens=60
rt = TransformersRuntime(cfg)
print("loading")
rt.load()
print("loaded")
orig = "Context: Mara is a former lighthouse keeper's daughter returning to her coastal hometown. The lighthouse has been dark for 20 years.\n\nInstruction: Continue the story naturally. Keep tense and POV consistent (close third, Mara).\n\nPrompt: The lighthouse had been dark for twenty years, until tonight. When Mara saw the beam sweep across the water, she knew"
exp = "Context: Inn is in Aureline, courier is Mina, late due to washed road.\n\nInstruction: Distinct voices, natural interruptions, action beats, no exposition dump.\n\nPrompt: Write a short dialogue where a tired innkeeper (50s, curt, formal) and a young courier (17, breathless, clipped) argue about a late delivery. Use interruptions and distinct voices."
for name, prompt in [("orig", orig), ("exp", exp)]:
    print(f"--- {name} ---")
    t0=time.time()
    res=rt.generate(prompt)
    elapsed=time.time()-t0
    print(f"done {elapsed:.1f}s tok {res.runtime_info['output_tokens']} tps {res.runtime_info['tokens_per_sec']}")
    print(res.generated[:100].replace(chr(10), " "))
