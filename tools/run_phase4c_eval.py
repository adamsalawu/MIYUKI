"""Phase 4C evaluation (exp 0007): original 12 + expanded 12 against trained adapter.
Run AFTER training completes. Uses same generation settings as all prior comparisons:
180 max, temp 0.8, top_p 0.95, rep 1.1, seed 42.
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import time
from miyuki.config import MiyukiConfig
from miyuki.evaluation.runner import run_evaluation

def run_suite(prompts_file, out_dir, label):
    cfg = MiyukiConfig.from_yaml("configs/training/phase4c_lora_03.yaml")
    cfg.generation.max_new_tokens = 180
    cfg.generation.temperature = 0.8
    cfg.generation.top_p = 0.95
    cfg.generation.repetition_penalty = 1.1
    cfg.generation.seed = 42
    print(f"[phase4c-eval] {label}: {prompts_file} -> {out_dir}")
    t0 = time.time()
    res = run_evaluation(model_config=cfg, prompts_file=prompts_file, output_dir=out_dir, report_dir="evaluation/reports")
    print(f"[phase4c-eval] {label} done: {res['outputs_path']} elapsed={time.time()-t0:.1f}s")
    return res

# 1) Original 12
run_suite("evaluation/prompts/eval_v0.1.jsonl", "evaluation/runs/phase4c-miyuki-0.3", "original-12")

# 2) Expanded 12 (voice/binary/Jinx/object/viewpoint targets)
run_suite("evaluation/prompts/targeted_scale.jsonl", "evaluation/runs/phase4c-expanded-miyuki-0.3", "expanded-12")

print("[phase4c-eval] ALL DONE")
