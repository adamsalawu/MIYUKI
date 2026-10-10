"""Phase 4C training entrypoint with explicit logging (exp 0007 expected)."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except: pass
import time, hashlib, pathlib, psutil
from miyuki.config import MiyukiConfig
from miyuki.training.train import train

cfg_path = "configs/training/phase4c_lora_03.yaml"
dataset_path = "datasets/miyuki-writing-0.3/dataset.jsonl"
print(f"[phase4c] Loading {cfg_path}")
cfg = MiyukiConfig.from_yaml(cfg_path)
print(f"[phase4c] model={cfg.model.name} dataset={dataset_path}")
print(f"[phase4c] assistant_only={cfg.training.assistant_only_loss} r={cfg.adapter.r} alpha={cfg.adapter.alpha} lr={cfg.training.learning_rate} epochs={cfg.training.num_epochs} batch={cfg.training.per_device_train_batch_size} accum={cfg.training.gradient_accumulation_steps} max_seq={cfg.training.max_seq_length} seed={cfg.training.seed}")
ds_sha = hashlib.sha256(pathlib.Path(dataset_path).read_bytes()).hexdigest()
print(f"[phase4c] dataset SHA {ds_sha}")
vm = psutil.virtual_memory()
print(f"[phase4c] memory before: avail={vm.available/1e9:.2f}GB used={vm.percent}% (threshold 3.0GB)")
assert vm.available/1e9 >= 3.0, "memory guard: below threshold, STOP"
# Expected steps: 154 -> 138 train / 16 eval, batch1 accum8 -> ~17/epoch -> ~51 total
print(f"[phase4c] expected: 138 train / 16 eval, ~17 opt steps/epoch, ~51 total for 3 epochs")
t0 = time.time()
res = train(config_path=cfg_path, dataset_path=dataset_path, smoke=False)
print(f"[phase4c] RESULT {res}")
print(f"[phase4c] ELAPSED {time.time()-t0:.1f}s")
vm2 = psutil.virtual_memory()
print(f"[phase4c] memory after: avail={vm2.available/1e9:.2f}GB used={vm2.percent}%")
