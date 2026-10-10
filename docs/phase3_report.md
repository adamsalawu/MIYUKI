# Phase 3 Report — First Real Writing SFT / LoRA

**Status: PASS (training and evaluation executed and measured) — hypothesis NOT supported with this conservative config (no measurable writing improvement).**

**Project:** `C:\Aether\RisingStar\MIYUKI`  
**Base:** `Qwen/Qwen2-0.5B-Instruct` (494,032,768 params, Qwen2, 24 layers, hidden 896, vocab 151646 tokenizer / 151936 model vocab)  
**Tokenizer:** Production `Qwen/Qwen2-0.5B-Instruct` (151646, chat_template True, pad `<|endoftext|>` 151643, eos `<|im_end|>` 151645) — NOT replaced, NOT added tokens.  
**Dataset:** `miyuki-writing-0.1` — 38 human-authored, human-edited, 21738 chars, 5072 tokens, SHA256 `9e1b97461f98948254bc45906cb275f111318f5d5adfc9e7139ab9c35c200d5e`, `38/38 valid`, `CC-BY-4.0`, categories `scene 29, dialogue 10, anti_meta 10, etc.`  
**Hardware:** Intel i7-1065G7 (4c/8t), 16 GB RAM, NVIDIA MX330 2 GB (unused, torch CPU), `torch 2.14.0+cpu`, `transformers 4.41.0`, `peft 0.11.1`, `accelerate 1.15.0`, `datasets 5.0.1` — detected `recommended_device cpu`, `recommended_dtype float32`, `cuda_available False`.  
**Training:** `configs/training/phase3_lora.yaml` — `LoRA r 16, alpha 32, dropout 0.05, bias none, target_modules [q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]` (auto for qwen2, verified via `model.layers.0.*`), `TaskType CAUSAL_LM`, `batch 1, accum 8, max_seq 1024, epochs 1, lr 1e-4, weight_decay 0.01, warmup 0.03, cosine, seed 42, bf16/fp16 false, gradient_checkpointing false, CPU`  
**Output:** `checkpoints/phase3-miyuki-0.1` — `adapter_config.json (760 B)`, `adapter_model.safetensors (35,237,104 B)`, `train_metrics.json`, `metadata.json`, `tokenizer.json`, `miyuki_training_config.yaml`  
**Experiment:** `experiments/0002` — `metrics.json` (`id 0002, model Qwen, dataset miyuki-writing-0.1, dataset_version 0.1.0-phase3, hardware i7-1065G7, duration 382s, final_loss 3.762, peft True, artifact checkpoints/phase3-miyuki-0.1`)  
**Smoke:** `tools/smoke_phase3.py` on 4-example subset, `max_steps 2` — `trainable 8,798,208 / 502,830,976 (1.75%), steps 2, loss 3.81, eval loss 3.56→3.31`, checkpoint `checkpoints/phase3-smoke/checkpoint-2`, adapter load + generation ok, base without adapter still loads.  
**Full train:** `global_step 4` (34 train examples / (1*8)), `epoch 0.94`, `train_runtime 375.4 s`, `elapsed 382.4 s`, `train_loss 3.762` (loss trace 4.06→3.91→3.60→3.46 decreasing). `trainable 8,798,208 (1.75%)` confirmed.  
**Adapter:** Loadable via `TransformersRuntime` auto (`[info] Loaded adapter from checkpoints\phase3-miyuki-0.1`, `PeftModel true`) and explicit `load_adapter_from('checkpoints/phase3-miyuki-0.1')`; both generate, not random (`atedton` not in output), tokenizer match (151646).

---

## Training Results (Actual Metrics)

- Trainable params: **8,798,208** / total 502,830,976 = **1.75%** (Qwen2-0.5B with r16 on 7 modules) — recorded via smoke and `peft` print.
- Total params: 494,032,768 base + 8,798,208 LoRA = 502,830,976.
- Steps: **4 optimizer steps** (1 epoch, 38 examples → 34 train / 4 eval after 90/10 split, batch1 accum8 → 4 steps).
- Loss: `loss 4.0687 (step1, lr 1e-4) → 3.9142 (step2, 7.5e-05) → 3.6011 (step3, 2.5e-05) → 3.4652 (step4, 0)`; final `train_loss 3.76229327917099`, `eval_loss` not separately logged but training decreasing. No NaN, no OOM.
- Runtime: 382.4 s elapsed, 375.4 s train_runtime, `train_samples_per_sec 0.091`, `train_steps_per_sec 0.011`, `total_flos 10560854016000.0`.
- Tokens processed: ~5072 tokens corpus × ~1 epoch ≈ 5k tokens (plus chat template overhead).
- Checkpoint: `checkpoints/phase3-miyuki-0.1` with adapter + tokenizer; separate from `checkpoints/phase3-smoke` and `checkpoints/smoke` and `checkpoints/baseline`.

---

## Evaluation — Same 12-Prompt Suite (Apples-to-Apples)

**Suite:** `evaluation/prompts/eval_v0.1.jsonl` — 12 prompts same wording/order/categories (continuation, dialogue, scene_writing, character_interaction, emotional_scene, action_scene, quiet_scene, description, rewriting, character_consistency, tone_consistency, context_retention).

**Runs:**
- Phase 1 baseline: `evaluation/runs/phase1_qwen2_0.5b/outputs.jsonl` — `Qwen/Qwen2-0.5B-Instruct` without adapter, `Qwen small-instruct.yaml`, `180 max_new_tokens, temp 0.8, top_p 0.95, rep 1.1, seed 42`, `956 prompt tokens, 1538 output, avg 128, 2.84 tok/s, elapsed 543.8, load 8.9, 0 errors`, `human_eval.yaml: 9 discard, 3 edit, 0 keep`.
- Phase 3 LoRA: `evaluation/runs/phase3-miyuki-0.1/2026-09-24_Qwen_Qwen2-0.5B-Instruct/outputs.jsonl` (also `evaluation/runs/phase3-miyuki-0.1/outputs.jsonl`) — `Qwen + LoRA miyuki-writing-0.1`, `phase3_lora.yaml`, same generation config (180, 0.8, 0.95, 1.1, 42), `956 prompt tokens, 1698 output, avg 141.5, 3.31 tok/s, total_gen 503.4, elapsed 514.9, load 9.08, 0 errors`.

**Comparison per prompt (Phase1 → Phase3, same 180 limit):**
- `continuation-01`: Phase1 `it was time to return home... town square where ... wind of history` (180) → Phase3 `it was time to return home... town square where her mother had once stood...` (180) — very similar generic, no beam payoff. No improvement.
- `dialogue-01`: `Sister: Mom? It's been five years... Bret...` → `"Hi, Mom," eldest brother... Sister... father's passing...` — different but still flat, no tags, roles confused. No improvement.
- `scene-writing-01`: Phase1 `raindrops danced...dripping tea spoons...` (edit 3) → Phase3 `raindrops danced...dripping water... old wooden doorframes... piles of books and scrolls... fresh brewed coffee... bread... smoke` — more verbose, contradictory noise. **Regression** (more generic).
- `character-interaction-01`: both essay dumps (`The mentor has been teaching for years...`). No improvement.
- `emotional-scene-01`: both summary `heart pounding/sank... sense of relief...` — still tells not shows. No improvement.
- `action-scene-01`: Phase1 `finds herself running... trying to get away...` → Phase3 `eyes were locked... unstoppable force...` + **mixed Chinese `狭窄的街道，快速的脚步声...`** — **regression with language mixing** (not in dataset, multilingual leakage).
- `quiet-scene-01`: both `The answer is:` meta-answer (Phase1 42 toks, Phase3 103 toks longer but same class) — **persistent anti-meta failure** despite 5 dataset exemplars (007).
- `description-01`: Phase1 and Phase3 **identical first sentence** (`The abandoned observatory stood alone... deep purple...`) then same generic list — no improvement, no child POV.
- `rewriting-01`: Phase1 `"I found myself walking... sense of unease. Air thick..."` (29) → Phase3 `"I found myself walking... sense of unease. Air was cool and damp in the air..."` (40, redundant `in the air... in the air`) — same generic, not vivid.
- `character-consistency-01`: both **identical 12-token truncation** `Jinx, I know it sounds ridiculous but...` — early EOS persists (chat template quoting edge).
- `tone-consistency-01`: both meta essays (`The literary fiction tone is described as...` → `The narrator's voice is soft and calm...`) — same class, longer (84→126).
- `context-retention-01`: both hedging essays (`Elena's reaction could be/might be one of... depending... It depends...`) — no mug, no lavender refusal, **critical binary fail persists** despite 3 dataset exemplars (012,015,030).

**Objective:** Phase3 generated 160 more tokens (1698 vs 1538) and 0.47 tok/s faster, but no category moved to better tier. Success/error: both 12/12 success, 0 errors. Truncation: both have 4 early truncations (rewriting 29→40, Jinx 12 both, quiet 42→103, tone 84→126) — no fix.

---

## Human Evaluation

**Template:** Same 9 dims as Phase 1 (`prose_quality, naturalness, character_consistency, dialogue_quality, pacing, creativity, instruction_adherence, continuity, overall_usefulness` + `keep/edit/discard`) plus `phase1_vs_phase3` notes.

**File:** `evaluation/runs/phase3-miyuki-0.1/human_eval.yaml` (12 entries, `Miyuki Lab`, `2026-09-24`, `Qwen + LoRA r16`).

**Result:** Phase1 `9 discard, 3 edit (continuation, scene-writing, description edit), 0 keep` → Phase3 `10 discard, 2 edit (continuation, scene-writing), 0 keep` — **no prompt improved to next tier; 1 regression (action Chinese), 3 identical (description, Jinx, tone class), 5 same failure class**.

**Per-prompt ratings (Phase3, 1–5):** continuation 2/edit (generic), dialogue 1/discard (flat), scene-writing 2/edit (verbose regression), character-interaction 1/discard (dump), emotional 1/discard (summary), action 1/discard (Chinese), quiet 1/discard (meta), description 1/discard (identical), rewriting 1/discard (redundant), Jinx 1/discard (truncated), tone 1/discard (meta), context 1/discard (hedging). Details in `human_eval.yaml`.

**Vs spec:** No measurable progress toward `miyuki_writing_spec.md` (natural prose, varied rhythm, distinct voices, controlled exposition, emotional subtext, continuity).

---

## What Improved (Evidence-Backed Only)

- **Training plumbing:** Loss decreased 4.06→3.46 (final 3.762) over 4 steps — shows LoRA gradients flow on CPU with Qwen2, not just tiny-random.
- **Slight verbosity:** Phase3 outputs average 141.5 toks vs 128 (+13.5) — model more willing to generate to limit, not early EOS except Jinx.
- **No catastrophic forgetting:** General factual prompt `Explain why sky is blue` still answered factually, not as scene; no verbatim 40-char memorization in eval (checked 12 outputs vs 38 dataset targets), no invented term over-injection (`lumenweave` 0 in eval).
- **Otherwise:** No evidence-backed writing improvement. Loss ↓ ≠ writing quality.

---

## What Did Not Improve

- **Meta-analysis:** 4 prompts (quiet, tone, rewriting, context) still meta-answers / hedging — 5 anti-meta dataset examples (007,011,016,017,036) not learned in 4 steps.
- **Flat dialogue / indistinguishable voices:** dialogue-01 still flat, no interruptions/tags.
- **Weak scene grounding / generic AI phrasing:** continuation still `wind of history`, scene-writing more generic.
- **Weak spatial coherence:** action still incoherent, now with Chinese.
- **Character inconsistency:** Jinx still truncated 12 toks, no witty voice.
- **Context retention:** Elena mug/lavender hedging persists, no binary commit.
- **Premature EOS:** Jinx prompt still 12 toks for both.
- **Excessive exposition / lore dumping:** character-interaction still essay dump.
- **Repetitive/clichéd:** `sense of unease` repeated identical.

---

## Regression Risks

- **Language mixing:** Action scene `狭窄的街道` Chinese mid-English — new regression not in Phase1, likely Qwen multilingual + LoRA underfitting causing language switch under uncertainty.
- **Verbosity without quality:** Phase3 scene-writing added contradictory `fresh brewed coffee... bread... smoke` not in prompt, making it more generic.
- **Longer meta-answers:** quiet 42→103, tone 84→126 — model generates longer meta essays, not scenes.
- **No degradation in general ability** (sky blue still factual), but story writing no improvement.

---

## Artifacts

- **Checkpoint:** `checkpoints/phase3-miyuki-0.1/{adapter_config.json, adapter_model.safetensors (35.2 MB), train_metrics.json, metadata.json, tokenizer.json, tokenizer_config.json, vocab.json, merges.txt, special_tokens_map.json, miyuki_training_config.yaml, dataset_info.json}` — separate from `checkpoints/phase3-smoke/` and Phase1 baselines.
- **Experiment:** `experiments/0002/{metrics.json (id 0002, model Qwen, dataset miyuki-writing-0.1 SHA 9e1b9..., dataset_version 0.1.0-phase3, training_config, hardware i7-1065G7, duration 382s, final_loss 3.762, peft True, artifact checkpoints/phase3-miyuki-0.1), config.yaml, README.md}` — next available, not overwriting `experiments/0001` (smoke tiny-random).
- **Evaluation:** `evaluation/runs/phase3-miyuki-0.1/{outputs.jsonl, metadata.json, summary.json, human_eval.yaml, README.md, 2026-09-24_Qwen_Qwen2-0.5B-Instruct/outputs.jsonl}` + `evaluation/reports/2026-09-24_Qwen_Qwen2-0.5B-Instruct.json` (overwrites previous report but dated subfolder preserved; Phase1 outputs at `evaluation/runs/phase1_qwen2_0.5b/` untouched).
- **Overfitting check:** `tools/overfitting_check.py` — 4 paraphrased prompts (Haru tea, Elena mug, Jinx compass, Aureline) + eval memorization scan (40-char) + invented term counts + general instruction test.
- **Dataset:** `datasets/miyuki-writing-0.1/` unchanged (38 ex, manifest, review, failure_mapping).

---

## Tests

**Total: 47 passed** (`python -m pytest tests/ -q` 25.7 s, 2 warnings huggingface_hub + peft).

- **Existing 31:** Phase0/1 (config, dataset validation, hardware, inference, evaluation mocked, runtime chat template) — still pass.
- **Phase2 8:** `test_phase2.py` (dataset 38 valid, manifest 5072 toks, tokenizer report 1.36 tok/word, experimental 4776 ratio 0.942, special tokens 151646 vs 151936 diff 290, deterministic hash).
- **New Phase3 8:** `tests/test_phase3.py` — `test_phase3_config_loading` (r16 alpha32 etc), `test_dataset_to_training_conversion` (38 ex, chat template contains `<|im_start|>`), `test_lora_configuration_and_trainable_params` (trainable 14080 tiny, <0.2), `test_checkpoint_discovery` (adapter exists, `discover_checkpoints`), `test_adapter_round_trip_loading` (adapter r16, tokenizer 151646), `test_experiment_metadata` (0002, dataset SHA, loss 3.762, peft True), `test_deterministic_configuration` (seed 42, SHA 9e1b9...), `test_baseline_preservation` (Phase1 outputs 12 exist, Phase3 different, baseline config still Qwen no adapter).

No tests weakened/deleted.

---

## Recommendation (Evidence-Based, Next Milestone Only)

**Do not yet add new dataset, change tokenizer, scale model, or build Phase 4+ systems.**

The curated dataset is correct but **training signal was too weak**: 4 optimizer steps (1 epoch, 38 ex, accum 8, 382 s) with `lr 1e-4` on `r16` produced loss drop but no behavioral change. Hypothesis is **not disproven** — underfitting, not dataset failure.

**Next milestone should be Phase 3 continued — LoRA hyperparameter tuning, not new data:**

1. **Increase optimizer steps:** 3–5 epochs (not 1) with same 38 ex → ~12–20 steps (still CPU-reasonable, ~15–30 min). Keep batch1 accum8, `eval_steps 5`.
2. **Slightly higher LR:** Try `2e-4` (was 1e-4) — Phase 1 sweep showed decoding params don't fix failures; training LR may.
3. **Keep dataset, tokenizer, model identical** to isolate training effect.
4. **Re-evaluate on same 12 prompts** and compare `experiments/0003` vs `0002` vs Phase1 baseline.

If after 3–5 epochs still no improvement on anti-meta/context/dialogue, *then* consider targeted prompt engineering (system vs user, explicit "Do not explain" phrasing in training user messages) or slightly larger model (1B) — not now.

**Hard stop per spec:** No Story Brain, Research, Memory, Character Intelligence, larger model, GUI, Phase 4 — wait for instruction.

