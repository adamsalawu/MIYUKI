import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except: pass
import json, pathlib
paths = {
    'base': 'C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase1_qwen2_0.5b/outputs.jsonl',
    'phase3': 'C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase3-miyuki-0.1/2026-09-24_Qwen_Qwen2-0.5B-Instruct/outputs.jsonl',
    'phase3b': 'C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase3b-miyuki-0.1/2026-09-25_Qwen_Qwen2-0.5B-Instruct/outputs.jsonl',
    'phase3c-a': 'C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase3c-a-miyuki-0.1/2026-09-25_Qwen_Qwen2-0.5B-Instruct/outputs.jsonl',
    'phase3c-b': 'C:/Aether/RisingStar/MIYUKI/evaluation/runs/phase3c-b-miyuki-0.1/2026-09-25_Qwen_Qwen2-0.5B-Instruct/outputs.jsonl',
}
datas = {k: [json.loads(l) for l in pathlib.Path(p).read_text(encoding="utf-8").splitlines()] for k,p in paths.items()}
for i, pid in enumerate([d['id'] for d in datas['base']]):
    print("="*80)
    print(pid, datas['base'][i]['category'])
    for k in ['base','phase3','phase3b','phase3c-a','phase3c-b']:
        gen = datas[k][i]['generated']
        lower = gen.lower()
        is_meta = 'the answer is' in lower or 'the literary' in lower or 'this scene demonstrates' in lower or 'here is a scene' in lower or "the narrator's voice" in lower
        is_hedge = 'could be one of' in lower or 'it depends' in lower or 'might be one of' in lower
        tok = datas[k][i]['runtime']['output_tokens']
        preview = gen[:120].replace(chr(10),' ')
        print(f"{k:12s} tok{tok:3d} meta:{is_meta} hedge:{is_hedge} -> {preview}")
