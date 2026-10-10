"""Sanity test for Qwen2.5-1.5B-Instruct via existing TransformersRuntime (5 prompts)."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except: pass
import json, time, pathlib, psutil
from miyuki.config import MiyukiConfig
from miyuki.runtime.transformers_runtime import TransformersRuntime
from miyuki.hardware import detect_hardware

cfg = MiyukiConfig.from_yaml("configs/model/qwen2.5-1.5b-instruct.yaml")
print(f"Model: {cfg.model.name} family={cfg.model.family} ctx={cfg.model.context_length}")
print(f"Hardware before: {detect_hardware()['ram_available_gb']} GB avail")

rt = TransformersRuntime(cfg)
t0 = time.time()
try:
    rt.load()
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"LOAD FAILED: {e}")
    sys.exit(2)
load_s = time.time() - t0
print(f"Loaded in {load_s:.1f}s device={rt._device} dtype={rt._dtype} loaded={rt.is_loaded()}")

# Model metadata
try:
    import torch
    n_params = sum(p.numel() for p in rt._model.parameters())
    print(f"Params: {n_params}")
    # model file size on disk (HF cache)
    from huggingface_hub import scan_cache_dir
    try:
        cache = scan_cache_dir()
        total = 0
        for repo in cache.repos:
            if "Qwen2.5-1.5B" in repo.repo_id:
                total = repo.size_on_disk
                print(f"HF cache size for 1.5B: {total/1e9:.2f} GB ({repo.repo_id})")
    except Exception as e:
        print(f"cache scan failed: {e}")
except Exception as e:
    print(f"param count failed: {e}")
    n_params = None

print(f"Tokenizer vocab: {len(rt._tokenizer)}, chat_template={bool(rt._tokenizer.chat_template)}")
print(f"RAM after load: {psutil.virtual_memory().available/1e9:.2f} GB avail / {psutil.virtual_memory().total/1e9:.2f} total, used {psutil.virtual_memory().percent}%")

prompts = [
    ("prose-continuation", "The lighthouse had been dark for twenty years, until tonight. When Mara saw the beam sweep across the water, she knew"),
    ("dialogue", "Write a short dialogue: a tired innkeeper and a young courier argue about a late delivery. Use interruptions."),
    ("rewriting", "Rewrite to be more vivid without adding plot: 'She walked into the room and saw him sitting there. He looked sad.'"),
    ("character-interaction", "A cautious mentor and an idealistic student disagree about revealing a dangerous truth. Write their confrontation in dialogue."),
    ("constrained-desc", "Describe a harbor at night with verbs and nouns, no adjectives, no dialogue. Sound focus."),
]

results = []
for name, prompt in prompts:
    t1 = time.time()
    try:
        res = rt.generate(prompt, max_new_tokens=60)
        dt = time.time() - t1
        ok = len(res.generated.strip()) > 0
        print(f"\n--- {name} ---")
        print(f"gen {dt:.1f}s tok={res.runtime_info['output_tokens']} tps={res.runtime_info['tokens_per_sec']} chat={res.runtime_info['chat_template_used']} empty={not ok}")
        print(f"preview: {res.generated[:200].replace(chr(10),' ')}")
        results.append({"name": name, "ok": ok, "seconds": dt, "tokens": res.runtime_info["output_tokens"], "tps": res.runtime_info["tokens_per_sec"], "preview": res.generated[:300]})
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"{name} FAILED: {e}")
        results.append({"name": name, "ok": False, "error": str(e)})

# Save sanity report
out = pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/runs/model-scale-qwen2.5-1.5b/sanity.json")
out.parent.mkdir(parents=True, exist_ok=True)
import datetime
report = {
    "model": cfg.model.name,
    "family": cfg.model.family,
    "params": n_params,
    "tokenizer_vocab": len(rt._tokenizer),
    "chat_template": bool(rt._tokenizer.chat_template),
    "device": rt._device,
    "dtype": str(rt._dtype),
    "load_s": load_s,
    "ram_after_gb_avail": psutil.virtual_memory().available/1e9,
    "results": results,
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\nSaved sanity to {out}")
print("SANITY DONE")
