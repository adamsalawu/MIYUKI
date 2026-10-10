"""Phase 4C resume entrypoint (exp 0007): resume from checkpoint-30 with compat fix.
Detached execution via run_phase4c_resume_detached.ps1; logs to logs/phase4c_resume.log
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import time, hashlib, pathlib, psutil
from miyuki.config import MiyukiConfig
from miyuki.training.train import train

cfg_path = "configs/training/phase4c_lora_03.yaml"
dataset_path = "datasets/miyuki-writing-0.3/dataset.jsonl"
resume = "checkpoints/phase4c-miyuki-0.3/checkpoint-45"
print(f"[phase4c-0007] resume from {resume}")
cfg = MiyukiConfig.from_yaml(cfg_path)
print(f"[phase4c-0007] model={cfg.model.name} assistant_only={cfg.training.assistant_only_loss} r={cfg.adapter.r} lr={cfg.training.learning_rate} epochs={cfg.training.num_epochs} batch={cfg.training.per_device_train_batch_size} accum={cfg.training.gradient_accumulation_steps} max_seq={cfg.training.max_seq_length} seed={cfg.training.seed}")
print(f"[phase4c-0007] dataset SHA {hashlib.sha256(pathlib.Path(dataset_path).read_bytes()).hexdigest()}")
assert pathlib.Path(resume).exists(), "resume checkpoint missing"
cfg.training.resume_from_checkpoint = resume
vm = psutil.virtual_memory()
print(f"[phase4c-0007] memory before resume: avail={vm.available/1e9:.2f}GB used={vm.percent}% (threshold 3.0GB)")
assert vm.available / 1e9 >= 3.0, "memory guard: below threshold, STOP"
t0 = time.time()
res = train(config=cfg, dataset_path=dataset_path, smoke=False)
print(f"[phase4c-0007] RESULT {res}")
print(f"[phase4c-0007] ELAPSED {time.time()-t0:.1f}s")
vm2 = psutil.virtual_memory()
print(f"[phase4c-0007] memory after: avail={vm2.available/1e9:.2f}GB used={vm2.percent}%")
print("[phase4c-0007] DONE")
