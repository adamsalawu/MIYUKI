"""Preflight guard for 1.5B micro-LoRA: load model as benchmark, measure RAM, attach LoRA, decide."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except: pass
import json, time, pathlib, psutil
from miyuki.config import MiyukiConfig
from miyuki.runtime.transformers_runtime import TransformersRuntime

def gb(x): return x/1e9

vm0 = psutil.virtual_memory()
print(f"BEFORE: total={gb(vm0.total):.2f}GB avail={gb(vm0.available):.2f}GB used={vm0.percent}%")
print(f"CPU: Intel i7-1065G7, GPU: MX330 (no CUDA expected)")

cfg = MiyukiConfig.from_yaml("configs/training/qwen2.5-1.5b_micro_lora.yaml")
print(f"Model: {cfg.model.name}, assistant_only={cfg.training.assistant_only_loss}, r={cfg.adapter.r}")

rt = TransformersRuntime(cfg)
t0=time.time()
try:
    rt.load()
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"LOAD FAILED: {e}")
    sys.exit(3)
load_s=time.time()-t0
vm1 = psutil.virtual_memory()
print(f"AFTER LOAD: avail={gb(vm1.available):.2f}GB used={vm1.percent}% load_s={load_s:.1f}s device={rt._device} dtype={rt._dtype}")
try:
    n_params=sum(p.numel() for p in rt._model.parameters())
    print(f"Base params: {n_params}")
except Exception as e:
    print(f"param count failed: {e}"); n_params=None

# Attach LoRA (no training) to measure additional memory
try:
    from peft import LoraConfig, get_peft_model, TaskType
    lora_cfg=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], task_type=TaskType.CAUSAL_LM, bias="none")
    rt._model=get_peft_model(rt._model, lora_cfg)
    trainable=sum(p.numel() for p in rt._model.parameters() if p.requires_grad)
    total=sum(p.numel() for p in rt._model.parameters())
    print(f"LoRA attached: trainable={trainable} total={total} pct={trainable/total*100:.2f}%")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"LORA ATTACH FAILED: {e}")
    sys.exit(4)

vm2 = psutil.virtual_memory()
print(f"AFTER LORA: avail={gb(vm2.available):.2f}GB used={vm2.percent}%")
# Estimate training overhead: activations for batch1 seq1024 on 1.5B (28 layers, hidden 1536) ~2-4GB + optimizer states for LoRA (~0.2GB) + dataloader
# Conservative threshold: require at least 3.0GB avail after load+LoRA to attempt training
threshold_gb = 3.0
avail = gb(vm2.available)
print(f"Threshold for safe training: {threshold_gb:.1f}GB avail required after load+LoRA")
decision = "GO" if avail >= threshold_gb else "NO-GO"
print(f"DECISION: {decision} (avail {avail:.2f}GB {'>=' if decision=='GO' else '<'} {threshold_gb}GB)")
# Save preflight report
out=pathlib.Path("C:/Aether/RisingStar/MIYUKI/evaluation/runs/qwen2.5-1.5b-micro-lora/preflight.json")
out.parent.mkdir(parents=True, exist_ok=True)
import datetime
report={
    "model":cfg.model.name,"params":n_params,"device":rt._device,"dtype":str(rt._dtype),
    "ram_total_gb":gb(vm0.total),"ram_avail_before_gb":gb(vm0.available),
    "ram_avail_after_load_gb":gb(vm1.available),"ram_avail_after_lora_gb":gb(vm2.available),
    "ram_used_percent_after":vm2.percent,"load_s":load_s,
    "trainable":trainable if 'trainable' in dir() else None,
    "threshold_gb":threshold_gb,"decision":decision,
    "timestamp":datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "note":"Preflight loads model exactly as benchmark + LoRA attach, no training step. Training needs gradients+activations+optimizer beyond this.",
}
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(f"Saved {out}")
# Exit code 0 for GO, 10 for NO-GO (so caller can branch without treating as error)
sys.exit(0 if decision=="GO" else 10)
