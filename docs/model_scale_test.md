# Model Scale Test — Qwen2.5-1.5B-Instruct Base vs Phase 4 Qwen2-0.5B Adapter

**Hypothesis:** A larger, newer 1.5B instruction model is already substantially better for MIYUKI writing tasks before fine-tuning, justifying moving the backend.

## Model
- **Candidate:** `Qwen/Qwen2.5-1.5B-Instruct` — 1,543,714,304 params (3.1× Phase4 base 494M), Qwen2, vocab 151665 (+19 vs 0.5B 151646), 32K context (config), 8K generation support (not tested, eval uses 180), chat True
- **Config:** `configs/model/qwen2.5-1.5b-instruct.yaml` (selectable, 0.5B `small-instruct.yaml` preserved, no hard-coding)
- **Official variants noted:** GPTQ-Int4 and GGUF exist per model card, NOT downloaded (standard Transformers used for clean comparison; only fall back if RAM failed — it fits, so no quantization tested)

## Runtime
- Same `TransformersRuntime` (AutoTokenizer/AutoModelForCausalLM + `apply_chat_template`, same as 0.5B), `src/miyuki/runtime/transformers_runtime.py`, `src/miyuki/inference/pipeline.py`, `src/miyuki/evaluation/runner.py` (single-runtime reuse)
- Transformers 4.41.0, torch 2.14.0+cpu, python 3.12.10, CPU i7-1065G7 (4/8), RAM 15.76 total, 6.4 avail idle / 5.74 before load / 1.58 after (90.7% used), CUDA False, MX330 2GB not used, disk 455GB free
- No environment reinstall (used existing).

## Benchmark Configuration
- **Original 12:** `evaluation/prompts/eval_v0.1.jsonl` EXACT (no rewrite/add/remove), same generation (180 max, temp0.8 top_p0.95 rep1.1 seed42)
- **Targeted 5:** `evaluation/prompts/targeted_scale.jsonl` (dialogue distinct ages, Jinx voice, binary Elena mug/lavender, child observatory, tight action retrieval) — same prompts saved for future candidates, run for 1.5B now (507s), compared to Phase4 original/expanded results

## Original 12 Results
- **Phase1 base (0.5B):** 1538 toks avg128, 2.84 tok/s, 0/3/9, meta 4 (answer-is, tone-essay, hedging, EOS)
- **Phase3B (0.5B+LoRA 38ex):** 996 avg83, 3.4 tok/s, 1/2/9, meta 2 EOS
- **Phase4 (0.5B+LoRA 120ex assistant-only):** 995 avg83, 2.27 tok/s, **5/4/3**, meta 0 — scene/quiet/description/rewriting/tone keep
- **1.5B base (no LoRA):** 1971 avg164, 1.53 tok/s, 1346s (22.4 min), 0 errors, **2/5/5** — scene/rewriting keep; continuation/emotional/action/quiet/Jinx edit; dialogue/interaction/description/tone/context discard. No meta answer-is/hedging/Chinese, but tone `Certainly! Here's...` intro + context numbered-list essay are meta. See `evaluation/runs/model-scale-qwen2.5-1.5b/human_eval.yaml`, `comparison.json`.

## Targeted Results (Phase4 Weaknesses)
1. **Distinct dialogue voices (tgt-dialogue-01):** 1.5B invents Teen/Officer POV labels, mom-birthday-scarf, dry officer — still generic, roles confused. Phase4 expanded similar flat. **Neither passes.**
2. **Character personality Jinx (tgt-character-01):** 1.5B 41 tokens `Great, I muttered... Those idiots at headquarters...` — **best Jinx yet** (witty/blunt, short), better than Phase4 109 earnest and original 180 sneer-but-narrated. **1.5B wins here.**
3. **Binary context Elena (tgt-context-01):** 1.5B `usual silver mug, broken/stained, black hue... Despite dislike... accepting might be right` — **hallucinates silver/black mug, accepts lavender** (direct violation, worse than Phase4 hands-back-cup scene which at least refused essay but missed facts). **Phase4 better (less wrong).**
4. **Exact object/detail retrieval (tgt-retrieval-01 child observatory):** 1.5B `looms over us... my heart races... rows of telescopes` — first-person plural, adult, not 10yo. Phase4 `lantern/stream/backpack-27` child-anchored keep. **Phase4 wins.**
5. **Action continuity (tgt-action-01 notebook retrieval):** 1.5B vendor grabs shoulder `What is this you've got there?` — violence, no notebook retrieval. Phase4 street/bag/thud also no retrieval. **Both fail, 1.5B slightly better pacing.**

## Human Evaluation
Same 9 dims + keep/edit/discard (no new score). **1.5B 2/5/5 vs Phase4 5/4/3 vs Phase1 0/3/9.** 1.5B ties Phase4 on scene/rewriting/quiet/emotional, wins Jinx voice + action pacing, loses tone/context/description. Key question (which more reliably performs writing task?): **Phase4 adapter more reliable** (no meta intro, no context essay, child POV, no refusal).

## Resource Usage (Measured, Not Invented)
- Files/download: 3.10 GB HF cache (`models--Qwen--Qwen2.5-1.5B-Instruct`), vs 0.5B ~1GB
- RAM: 5.74 GB avail before → 1.58 GB after (90.7% used, 16.92 total with cache counting) — fits but no headroom for training (LoRA optimizer would need ~2× model RAM)
- GPU: 0% (CPU only, cuda False, MX330 2GB idle)
- CPU: i7-1065G7, generation 0.7–1.78 tok/s sanity / 1.53 benchmark avg (vs 0.5B 2.27–3.4)
- Load: 451.4s (7.5 min, includes 3.1GB download first time; warm would be ~40s like 0.5B? Actually 0.5B warm 8–11s, 1.5B warm unknown but likely 30s+)
- Disk free: 455GB, no issue

## Comparison with Phase 4
|  | Phase1 0.5B | Phase3B 0.5B+LoRA | Phase4 0.5B+LoRA-120 | 1.5B base |
|---|---|---|---|---|
| Keep/Edit/Discard | 0/3/9 | 1/2/9 | **5/4/3** | 2/5/5 |
| Meta (answer/tone/hedge) | 4 | 2 EOS | 0 | 0 answer + 2 meta-intro/essay (tone, context) |
| Premature EOS (<15w) | 1 | 2 | 0 | 0 |
| Dialogue | discard | discard | discard | discard (slightly more voice) |
| Jinx voice | truncation | earnest 31 | earnest 109 | **witty 41/180 (best)** |
| Context binary | hedging essay | hedging short | scene no-hedge but missing facts | essay numbered-list (worse) |
| Scene/description/rewriting/tone | 1 edit | 1 keep | 4 keep | 2 keep (scene/rewriting), tone fail |
| Action | discard | discard | discard | edit (best pacing, still no retrieval) |
| Refusal risk | none | none | none | **1 false refusal in sanity** (political) |
| Speed | 2.84 | 3.4 | 2.27 | 1.53 |

## Recommendation — **CASE B: Only Slightly Better (Specific Categories)**

1.5B base is more fluent/verbose (1971 vs 995 toks) and wins **Jinx voice + action pacing**, ties scene/rewriting/quiet/emotional, but **loses tone/context/description** where Phase4 adapter succeeds (no meta intro, no context essay, child POV) and carries **refusal risk + 90% RAM + 2× slower**. It is NOT clearly better overall (2 keep vs 5 keep).

Do NOT move backend yet. **Next milestone should be a SMALL LoRA test on 1.5B with `miyuki-writing-0.2` (same 120 ex, r16, assistant-only, 1–2 epochs, CPU with monitoring for OOM)** — justified because 1.5B shows Jinx/action promise and verbosity, and LoRA fixed anti-meta for 0.5B (Phase4). Do NOT run it yet (per hard rules). If 1.5B+LoRA still fails tone/context/binary, then decide between targeted dataset work vs alternative 1.5B (e.g., different family) based on measured failures. Keep Phase4 0.5B adapter as current backend.

**STOP** — No training, no checkpoint, no Story Brain. Phase4 adapter remains current.
