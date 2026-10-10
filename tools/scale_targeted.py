"""Run targeted 5-prompt benchmark for 1.5B (180 tokens)."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except: pass
import time
from miyuki.config import MiyukiConfig
from miyuki.evaluation.runner import run_evaluation
cfg = MiyukiConfig.from_yaml("configs/model/qwen2.5-1.5b-instruct.yaml")
print(f"Targeted {cfg.model.name}")
t0=time.time()
res=run_evaluation(model_config=cfg, prompts_file="evaluation/prompts/targeted_scale.jsonl", output_dir="evaluation/runs/model-scale-qwen2.5-1.5b-targeted", report_dir="evaluation/reports")
print(f"Done {res['outputs_path']} elapsed={time.time()-t0:.1f}s")
