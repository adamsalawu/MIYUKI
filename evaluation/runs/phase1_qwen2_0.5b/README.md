# Phase 1 — Qwen2-0.5B-Instruct Baseline

**Status:** Phase 1 trustworthy writing baseline (not final MIYUKI model)

## Purpose

Replace the intentionally useless `hf-internal-testing/tiny-random-gpt2` baseline with a real small instruct model (`Qwen/Qwen2-0.5B-Instruct`, 494M params) and measure MIYUKI's 12-prompt evaluation suite on actual hardware. The model is a baseline, not a MIYUKI writer.

## Model

- **ID:** `Qwen/Qwen2-0.5B-Instruct` (`qwen2`, 494032768 params)
- **Tokenizer:** `Qwen/Qwen2-0.5B-Instruct` (vocab 151646, `<|im_end|>` eos, `<|endoftext|>` pad)
- **Revision:** Hugging Face cache `models--Qwen--Qwen2-0.5B-Instruct` (symlink warning due to Windows; cache at `~/.cache/huggingface/hub`)
- **Context length:** 4096 (config), 180 `max_new_tokens` used for eval
- **Family detection:** `qwen2` correct
- **Chat template:** `True` — generic runtime now uses `tokenizer.apply_chat_template([{role:user, content:prompt}], add_generation_prompt=True)` when `chat_template is not None`. Prompt tokens include system + user framing (prompt_tokens 68–104 vs raw ~30).

## Hardware / Runtime

- **Machine:** Intel i7-1065G7, 16 GB RAM, NVIDIA MX330 2 GB, CPU-only `torch 2.14.0+cpu` (no CUDA stack installed per Phase 1 requirement)
- **Detected:** `recommended_device=cpu`, `recommended_dtype=float32`, `gpu_name=NVIDIA GeForce MX330`, `cuda_available=False`
- **Runtime:** `TransformersRuntime` (no Qwen-only pipeline, generic fix). Device `cpu`, dtype `torch.float32`, no quantization, `AutoModelForCausalLM` + `AutoTokenizer`
- **Load:** 8.9 s (warm cache) / 95.8 s cold (first download ~1 GB), includes safetensors
- **Fix:** evaluation runner now reuses a single runtime instance (`load()` once, `generate()` 12×) — avoids 12 reloads, verified regression test. Generation records `prompt_tokens/output_tokens/tokens_per_sec` and `chat_template_used`.

## Configuration

- **Active config:** `configs/model/small-instruct.yaml` (Hugging Face ID configurable; `baseline.yaml` retained for regression with tiny-random)
- **Generation defaults (Phase 1 eval):** `temperature 0.8, top_p 0.95, top_k null, max_new_tokens 180, repetition_penalty 1.1, do_sample true, seed 42`
- **Sweep configs** (subset 4 prompts to keep CPU time reasonable): `A_conservative (0.7/0.9/1.05)`, `B_default (0.8/0.95/1.1)`, `C_creative (1.0/0.95/1.0)` — see `generation_configs.json`. Sweep reused same chat-template path.

## Evaluation

- **Suite:** `evaluation/prompts/eval_v0.1.jsonl` — 12 prompts, 12 categories (continuation, dialogue, scene_writing, character_interaction, emotional_scene, action_scene, quiet_scene, description, rewriting, character_consistency, tone_consistency, context_retention). Never trained on.
- **Run:** `python -m miyuki.cli evaluate --config configs/model/small-instruct.yaml` → `evaluation/runs/phase1_qwen2_0.5b/` + dated `evaluation/runs/2026-09-24_Qwen_Qwen2-0.5B-Instruct/`
- **Outputs:** `outputs.jsonl` (12 entries, `num_errors 0`, each with `prompt/full_prompt/generated/full_text/model/generation_params/timings/runtime/error/timestamp`), `metadata.json` (model snapshot, `model_params 494032768`, `load_s`), `report.json`, `summary.json` (aggregated objective metrics)
- **Traceability:** each sample has `model/prompt/generation_config/run/timestamp`. Run ID `phase1_qwen2_0.5b` / dated `2026-09-24_Qwen_Qwen2-0.5B-Instruct`.

## Objective Metrics

- **Prompt tokens:** 956 total, 68–104 per prompt (inflated by chat template system prefix)
- **Output tokens:** 1538 total, avg 128, range 12–180 (truncated at 180 limit; many hit limit except failures like quiet_scene 42, rewriting 29, character_consistency 12, tone 84)
- **Generation time:** 541.3 s total, 9.4–78.8 s per prompt, elapsed 543.8 s (including tokenization)
- **Throughput:** 2.84 tokens/sec avg (2.07–3.37), CPU float32
- **Load:** 8.9 s warm, 95.8 s cold
- **Errors:** 0/12 failed loads; 4/12 produced weak/truncated outputs (see human eval)

## Human Evaluation

See `human_eval.yaml` (fills `evaluation/human_template.yaml` dimensions per output, 1–5 + notes, keep/edit/discard). Summary below is **observed, not scored numerically as quality**:

- **Continuations & quiet Kyoto scene & description:** Grammatical, some sensory detail, but clichéd filler ("wind of history", "mix of excitement and trepidation"), purple hedging, abrupt truncation mid-sentence at token limit. Edit-worthy, not keep-worthy.
- **Dialogue:** Unnatural, repetitive ("Sister/Bret"), expository, confused roles. Fails distinct voices.
- **Character interaction & emotional & action:** Generic exposition, tells rather than shows, misses instruction specifics (action scene repeats protagonist chased, not retrieving notebook).
- **Quiet-scene / rewriting / tone / context-retention / character-consistency:** Systematic failures — model answers *about* the prompt (analysis/meta) rather than *performing* it (e.g., "The answer is: ..." for quiet_scene; "The literary fiction tone is described as ..." for tone). Context-retention (Elena's mug/lavender) produces generic hedging ("depends on...") and contradicts/misses facts. Indicates 0.5B + current prompt framing insufficient for strict instruction adherence.

Overall: **coherent but generic AI phrasing, limited scene grounding, weak instruction following on 4/12 categories, no severe repetitive loops due to rep_penalty 1.1 but still flat dialogue and moralizing summaries.** Distance to `docs/miyuki_writing_spec.md` is large: misses natural prose rhythm, distinct voices, controlled exposition, emotional subtext, continuity.

## Generation Sweep

- **Subset:** 4 prompts (continuation, dialogue, quiet, context) × 3 configs, each ~140–182 s, total ~488 s.
- **Observations:** Varying temperature (0.7→1.0) and top_p (0.9→0.95) and rep (1.05→1.0) did **not** fix systematic meta-answer failures (quiet/context still analysis). Conservative (0.7) slightly more deterministic but still generic; creative (1.0) produced shorter continuation (77 tokens, early stop) and no improvement in instruction adherence. No config declared "best" — sweep provides comparative evidence: prompt framing / model capacity matters more than decoding params for MIYUKI's tasks.

## Known Limitations

- 0.5B is minimal; larger context-aware writing will need bigger model or fine-tuning.
- Prompt composition (`Context + Instruction + Prompt` → single user message) is simplistic for instruct chat models; Story Brain (Phase 5) will provide structured context instead. For now, instruction adherence is weak.
- CPU float32 is reliable but slow (1.7–3.4 tok/s); VRAM not used (MX330 present but torch CPU build).
- Token limit 180 truncates scenes mid-sentence; longer outputs needed for full scenes but increase latency.

## Reproduction

```powershell
python -m miyuki.cli hardware
python -m miyuki.cli models --list          # shows baseline + small-instruct
python -c "from miyuki.config import MiyukiConfig; cfg=MiyukiConfig.from_yaml('configs/model/small-instruct.yaml'); print(cfg.model.name)"
# First run downloads ~1 GB
python -m miyuki.cli evaluate --config configs/model/small-instruct.yaml
# Inspect outputs
Get-Content evaluation/runs/phase1_qwen2_0.5b/outputs.jsonl
Get-Content evaluation/runs/phase1_qwen2_0.5b/summary.json
```

## Where Outputs Live

- Main baseline: `evaluation/runs/phase1_qwen2_0.5b/outputs.jsonl` (+ dated copy `evaluation/runs/2026-09-24_Qwen_Qwen2-0.5B-Instruct/`)
- Sweep: `evaluation/runs/phase1_qwen2_0.5b/sweep_{A_conservative,B_default,C_creative}/2026-09-24_Qwen.../outputs.jsonl`
- Reports: `evaluation/reports/2026-09-24_Qwen_Qwen2-0.5B-Instruct.json`
- Human ratings: `evaluation/runs/phase1_qwen2_0.5b/human_eval.yaml`
- No checkpoints/models committed (gitignored)

## Not Changed

- `configs/model/baseline.yaml` still tiny-random for regression
- No Story Brain / dataset curation / LoRA / training / GUI / extra model added
- No hard-coded Qwen branches; runtime remains generic; `small-instruct.yaml` is config-only activation
