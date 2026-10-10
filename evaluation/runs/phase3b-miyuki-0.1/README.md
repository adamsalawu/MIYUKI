# Phase 3B — More Optimization Steps (miyuki-writing-0.1, 3 epochs, lr 2e-4)

**Hypothesis:** Phase 3 (4 optimizer steps, lr 1e-4) under-trained; 3× more steps with slightly higher LR may teach anti-meta/dialogue/context.

**Status:** Trained and evaluated — **partial, specific improvement** (1 keep, 2 edit vs 0 keep/2 edit/10 discard in Phase3; 0 keep/3 edit/9 discard in Phase1). Loss ↓, but 75% still discard; hypothesis partially supported (some categories learn, core meta/context still fails).

## Training

- **Base:** `Qwen/Qwen2-0.5B-Instruct` (494M, Qwen2, 24 layers, vocab 151646/151936, context 4096)
- **Tokenizer:** Production Qwen (no change, chat True)
- **Dataset:** `miyuki-writing-0.1` — 38 human-authored, SHA `9e1b97461f989482...`, 21738 chars, 5072 toks, 90/10 split seed42 → 34 train / 4 eval
- **LoRA:** `r16 alpha32 dropout0.05 bias none, targets [q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]` (Qwen2 auto), trainable 8,798,208 / 502,830,976 (1.75%)
- **Config:** `configs/training/phase3b_lora.yaml` — `epochs 3 (was 1), lr 2e-4 (was 1e-4, per PEFT/TRL example), batch1 accum8 max_seq1024 weight_decay0.01 warmup0.03 cosine seed42 CPU float32`
- **Expected steps:** Train 34 *3 /8 = 12.75 → 12 optimizer steps (vs Phase3 4) — 3×. Total fwd passes 102.
- **Actual:** `global_step 12, epoch 2.82, train_runtime 752.5s, elapsed 759.3s, train_loss 3.205` (from 4.06→3.91→3.51→3.34→3.28→3.12→3.01→3.06→2.80→2.74→2.68→2.89, decreasing). `eval_loss 3.369→3.173`. `train_samples_per_sec 0.136, steps_per_sec 0.016, total_flos 32193003325440.0`.
- **Checkpoint:** `checkpoints/phase3b-miyuki-0.1/{adapter_config.json (760), adapter_model.safetensors (35.2 MB), train_metrics.json, metadata.json, checkpoint-5/10}` — separate from `checkpoints/phase3-miyuki-0.1` and `phase3-smoke`.
- **Experiment:** `experiments/0003` — `id 0003, model Qwen, dataset miyuki-writing-0.1, dataset_version 0.1.0-phase3b, training_config (lr 2e-4, epochs 3), hardware i7-1065G7, duration 759s, final_loss 3.205, artifact checkpoints/phase3b-miyuki-0.1`

## Smoke Test

Before full training, 4-example `max_steps 2, lr 2e-4` smoke: `trainable 1.75%, steps2 loss3.81, eval loss 3.56→3.19`, checkpoint `phase3b-smoke/checkpoint-2`, adapter reload + generation `Elena hesitates...` ok, base without adapter still loads. See `tools/smoke_phase3b.py`.

## Round-Trip Verification (Base, Phase3, Phase3B)

All three load with same tokenizer (151646) and base (494M):

- **Base** (`small-instruct.yaml`, no adapter): `44 prompt toks` → `it was time to get back to her ship...` (coherent)
- **Phase3** (`phase3_lora.yaml` auto-loads `phase3-miyuki-0.1`, is Peft True): `it was time to get back to shore. She stood in awe...` (different)
- **Phase3B** (`phase3b_lora.yaml` auto-loads `phase3b-miyuki-0.1`, is Peft True): `it was time to call it a night. She stood in silence at the base of the tower...` (different again, not random)

All use `chat_template True`, not random `atedton`, independently loadable.

## Evaluation (12 prompts, apples-to-apples)

**Prompts:** `evaluation/prompts/eval_v0.1.jsonl` — same 12 categories as Phase1.

**Runs:**
- Phase1: `evaluation/runs/phase1_qwen2_0.5b/outputs.jsonl` — `956 prompt, 1538 out (avg128), 2.84 tok/s, elapsed543.8, load8.9, 0 errors, 9 discard/3 edit/0 keep`
- Phase3: `evaluation/runs/phase3-miyuki-0.1/outputs.jsonl` — `956 prompt, 1698 out (avg141.5), 3.31 tok/s, elapsed514.9, load9.08, 0 errors, 10 discard/2 edit/0 keep` (Chinese regression)
- Phase3B: `evaluation/runs/phase3b-miyuki-0.1/outputs.jsonl` (+ dated `2026-09-25_Qwen...`) — `956 prompt, 1105 out?` Actually 32+72+180+180+77+108+20+180+17+31+23+32 = 952? Wait sum: 76+72+180+180+77+108+20+180+17+31+23+32 = 976, with avg ~81, `2.9-3.7 tok/s`, `elapsed292.9, load8.6, 0 errors`. Detailed: continuation76, dialogue72, scene180, char180, emotional77, action108, quiet20, description180, rewriting17, Jinx31, tone23, context32 — many short (premature EOS).
- Collected: `outputs.jsonl, metadata.json, summary comparison, human_eval.yaml` per run.

## Human Evaluation

Same 9 dims (`prose_quality, naturalness, character_consistency, dialogue_quality, pacing, creativity, instruction_adherence, continuity, overall_usefulness` + `keep/edit/discard`).

File: `evaluation/runs/phase3b-miyuki-0.1/human_eval.yaml` compares Phase1 → Phase3 → Phase3B per prompt.

**Result:** Phase1 `0 keep /3 edit /9 discard` → Phase3 `0/2/10` (regression) → **Phase3B `1 keep (emotional) /2 edit (continuation, character-interaction) /9 discard`** — one clear keep, one notable improvement, one slight, but 75% still discard. Chinese fixed (no Chinese in Phase3B action vs Phase3), but new shortness (quiet20, rewriting17, tone23) indicates more early EOS.

**Per-prompt vs Phase3:**
- `continuation`: 2/edit → 3/edit slight improvement (oak stump lantern 'Home' more specific)
- `dialogue`: 1/discard → 1 (still flat, now Aunt Mary/Tom invented)
- `scene-writing`: 2/edit → 2/edit similar
- `character-interaction`: 1/discard → 3/edit **improvement** (eyes dim/hands damp, not essay dump)
- `emotional`: 1/discard → 4/keep **strong improvement** (held between teeth, swallow thickly, not 'relief wash')
- `action`: 1/discard Chinese → 1/discard (Chinese fixed but still incoherent)
- `quiet`: 1/discard meta → 1/discard (still not quiet scene, now ticket line not answer is)
- `description`: 1/discard → 1/discard (now dialogue violation with `Is this your room?`)
- `rewriting`: 1/discard → 1/discard (shorter)
- `Jinx`: 1/discard (12 truncated) → 1/discard (31 not truncated but still earnest not witty)
- `tone`: 1/discard → 1/discard (meta essay 126→23)
- `context`: 1/discard hedging → 1/discard (still hedging but shorter)

## Failure Checks

- **Meta:** Still fails quiet/tone/context/rewriting (all 4 still meta or hedging, though quiet no longer `The answer is:` but ticket line).
- **Dialogue:** Still flat, roles confused.
- **Scene grounding:** Slightly better in continuation/character-interaction, but action still incoherent.
- **Character voice (Jinx):** Longer (31 vs 12) but still earnest, not witty.
- **Context (Elena):** Phase3B 32 toks `Thank you, she says, picking up pot... I like it` — still hedging via ellipsis, no mug, no lavender refusal — binary fail persists despite 3 dataset exemplars.
- **Quiet:** Still not performing scene.
- **Action Chinese:** **Fixed** (no Chinese in Phase3B vs Phase3).
- **Premature EOS:** More pronounced in Phase3B (7 prompts <80 toks vs Phase3 2), but Jinx improved (12→31).
- **Generic phrasing:** Still generic (`air thick with smell of old books`).

## Overfitting

- **Paraphrased prompts (4):** Haru tea, Elena mug, Jinx, Aureline — all generated without 30-char copied snippets, no name overuse, no meta, not empty; general factual `Explain why sky is blue` still factual (Rayleigh/scatter) — no degradation.
- **Eval memorization:** 12 eval outputs checked for 40-char verbatim dataset snippets → **none**.
- **Stylistic injection:** Invented terms 0 in eval outputs (not over-injected).
- **Conclusion:** No obvious overfitting/memorization, but also no strong imitation — underfitting.

## Baseline Preservation

Phase1 outputs (`phase1_qwen2_0.5b/outputs.jsonl` 12, 44522 bytes) still exist, Phase3 outputs (`phase3-miyuki-0.1` 46410 bytes) and Phase3B (38419 bytes) are separate; `experiments/0001,0002,0003` all present; `checkpoints/phase3-miyuki-0.1` and `phase3b-miyuki-0.1` separate.

## Reproduction

```powershell
python -m miyuki.cli dataset validate --path datasets/miyuki-writing-0.1/dataset.jsonl --format conversational
python tools/smoke_phase3b.py
python -c "from miyuki.training.train import train; train(config_path='configs/training/phase3b_lora.yaml', dataset_path='datasets/miyuki-writing-0.1/dataset.jsonl')"
python -c "from miyuki.evaluation.runner import run_evaluation; from miyuki.config import MiyukiConfig; cfg=MiyukiConfig.from_yaml('configs/training/phase3b_lora.yaml'); run_evaluation(cfg, prompts_file='evaluation/prompts/eval_v0.1.jsonl', output_dir='evaluation/runs/phase3b-miyuki-0.1')"
Get-Content evaluation/runs/phase3b-miyuki-0.1/summary.json
```

## Known Limitations

- 12 optimizer steps still small for 38 examples; 3 epochs may still under-train for anti-meta/context which needs stronger signal.
- Early EOS/shortness in Phase3B suggests model learning to stop early (maybe over-regularized by short targets avg 62 words?).
- Chinese fixed but action still not grounded.

## Not Changed

- Model/dataset/tokenizer unchanged (Qwen 0.5B, 151646, miyuki-writing-0.1 SHA 9e1b9...), no vocab add, no resize, no Story Brain/Research/Memory/GUI/larger model.
