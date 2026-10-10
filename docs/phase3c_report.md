# Phase 3C Report — Training Formulation Experiment (System vs User Contract)

**Status: PASS (executed & measured) — Neither formulation materially improved writing behavior vs Phase3B; system vs user alone is not the lever.**

**Project:** `C:\Aether\RisingStar\MIYUKI`  
**Base:** `Qwen/Qwen2-0.5B-Instruct` (494,032,768 params, Qwen2, vocab 151646/151936, context 4096)  
**Tokenizer:** Production `Qwen/Qwen2-0.5B-Instruct` (151646, chat True, not replaced)  
**Dataset:** `miyuki-writing-0.1` — 38 human-authored, SHA `9e1b97461f98948254bc45906cb275f111318f5d5adfc9e7139ab9c35c200d5e` (preserved byte-for-byte, `manifest.json` 38 ex)  
**Derived:** `datasets/miyuki-writing-0.1-phase3c/{dataset-a.jsonl SHA b518f30bffe070ab62b..., dataset-b.jsonl SHA e408e23ecfc6d5c7f509..., manifest.json}` — transformation: added concise behavioral contract **without changing 38 writing targets** (assistant identical).  
**Hardware:** Intel i7-1065G7, 16GB, MX330 2GB (torch CPU), `torch 2.14.0+cpu`, `transformers 4.41.0`, `peft 0.11.1`, `python 3.12.10`  

## Experiments

- **0004 = Phase 3C-A (system contract):** `configs/training/phase3c_a_lora.yaml` — `You are Miyuki, a direct story-writing assistant... (64 toks system)` as `system` role + user as original + assistant same. `checkpoints/phase3c-a-miyuki-0.1`, `experiments/0004`.
- **0005 = Phase 3C-B (user-only):** `configs/training/phase3c_b_lora.yaml` — same behavioral requirements as `Requirements: Perform...` suffix (42 toks) appended to user, no system, assistant same. `checkpoints/phase3c-b-miyuki-0.1`, `experiments/0005`.

Both reuse Phase3B training config: `LoRA r16 alpha32 dropout0.05 bias none targets [q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj] (Qwen2 auto), batch1 accum8 max_seq1024 lr2e-4 3 epochs seed42 CPU float32` — **only instruction formulation changes**.

## Training (Actual)

**Expected:** 38 ex → 34 train /4 eval (90/10 seed42), batch1 accum8 → `ceil(34/8)=5` opt steps per epoch → 15 total for 3 epochs (Phase3 had 4 for 1 epoch, Phase3B had 12 for 3 epochs). Actual both gave **12 steps** (epoch 2.82, short of 15 due to Trainer dropping last partial? Same as Phase3B).

- **0004 A:** `global_step 12, epoch 2.82, train_runtime 1213.2s elapsed 1220.9s (20.3 min), train_loss 3.239` (trace 4.48→4.32→3.85→3.59→3.39→3.16→2.97→2.89→2.63→2.53→2.42→2.60, eval 3.31→2.78), `train_samples_per_sec 0.084, steps_per_sec 0.01, total_flos 44443593984000`, `trainable 8,798,208 (1.75%)`.
- **0005 B:** `global_step 12, epoch 2.82, train_runtime 1202.7s elapsed 1211.9s (20.1 min), train_loss 3.143` (4.26→4.08→3.69→3.46→3.33→3.10→2.91→2.80→2.56→2.50→2.39→2.57, eval 3.29→2.82), `train_samples_per_sec 0.085, steps_per_sec 0.01, total_flos 40859504152320`, same trainable.

Both: 12 steps substantially more than Phase3's 4, loss lower than Phase3's 3.76, similar to Phase3B's 3.20 (B slightly lower at 3.14). Smoke for both (4-ex, max_steps2, lr2e-4): trainable 1.75% same, steps2 loss 4.17→3.53 (A) / 3.99 (B), checkpoint `phase3c-*-smoke/checkpoint-2`, adapter reload + generation ok (though generation still poor `Elena takes a sip of lavender tea...`).

## Dataset

- **Original SHA:** `9e1b97461f98948254bc45906cb275f111318f5d5adfc9e7139ab9c35c200d5e` (38 ex, 21738 chars, 5072 toks orig)
- **Derived A SHA:** `b518f30bffe070ab62b384b1d936fd38d8ecdd6910b40367c9c0ae09a84e9d7c` (38 ex, 8073 toks total, avg 212.4, max 291, +37.6% total, +58 tok avg vs orig)
- **Derived B SHA:** `e408e23ecfc6d5c7f509bdd9e823d6a45d2750ba63ce974d05f87310e471318c` (7428 toks, avg 195.5, max 274, +26.6%)
- **System contract tokens:** 64, **user suffix:** 42. Both <30% of 1024 window (max 291/274 <900, not excessive per spec). Verified in `tools/check_tokenization_phase3c.py` and `tests/test_phase3c.py::test_token_statistics` (5869 vs 8073 vs 7428, max 233→291/274).

## Formulation (Exactly What Differed)

**A (system):** `messages = [system: "You are Miyuki, a direct story-writing assistant. Perform the requested writing task directly. Do not explain, announce, summarize, or discuss the task. Follow constraints, preserve character and context facts, do not invent dialogue when forbidden, continue the story rather than discussing it, and when rewriting output the rewritten text directly." (64 toks), user: original context+instruction+prompt, assistant: original target]` — behavioral contract in system role.

**B (user-only):** `messages = [user: original user + "\n\nRequirements: Perform the writing directly. Do not explain, announce, summarize, or discuss. Follow constraints, preserve context, do not invent dialogue when forbidden, continue story not discuss, when rewriting output directly." (42 toks suffix), assistant: original target]` — same requirements but in user, no system.

Targets identical (38 assistant writings byte-for-byte same as original). Only system vs user placement differs.

## Evaluation (Same 12 Prompts, `eval_v0.1.jsonl`, 180 max, temp0.8 top_p0.95 rep1.1 seed42)

**Runs:**
- Phase1 base: `evaluation/runs/phase1_qwen2_0.5b/outputs.jsonl` — `956 prompt, 1538 out avg128, 2.84 tok/s, elapsed543.8 load8.9 0 errors, 0 keep/3 edit/9 discard`
- Phase3: `evaluation/runs/phase3-miyuki-0.1/2026-.../outputs.jsonl` — `956 prompt, 1698 out avg141, 3.31 tok/s, elapsed514.9 0 errors, 0/2/10, Chinese regression in action`
- Phase3B: `evaluation/runs/phase3b-miyuki-0.1/2026-.../outputs.jsonl` — `956 prompt, 996 out? Actually summary 996 total avg83, 3.4 tok/s, elapsed292.9, 1/2/9 (emotional keep, character-interaction 1→3, Chinese fixed)`
- **Phase3C-A:** `evaluation/runs/phase3c-a-miyuki-0.1/2026-09-25_Qwen.../outputs.jsonl` — `956 prompt, ~? (66+108+100+113+28+58+36+171+70+87+49+11 = 898) avg75, 3.5 tok/s, elapsed369, 0 errors`
- **Phase3C-B:** `evaluation/runs/phase3c-b-miyuki-0.1/2026-09-25_Qwen.../outputs.jsonl` — `956 prompt, ~? (26+180+80+108+8+87+17+102+11+109+20+5 = 754) avg63, 3.5 tok/s, elapsed314, 0 errors`

**Comparison highlights (per 12, meta/hedge flags):**
- `continuation`: base 180 generic town square, Phase3 180 same, Phase3B 76 oak stump lantern 'Home' (slight improvement), **Phase3C-A 66 generator, Phase3C-B 26 engine died** — both shorter, more generic than Phase3B.
- `dialogue`: all flat; Phase3B 72 Aunt Mary/Tom, Phase3C-A 108 Hi brother, Phase3C-B 180 Aunt Mary who died — still flat.
- `scene-writing`: Phase1 180 best, Phase3B 180 old books/smoke, **Phase3C-A 100 puddles, Phase3C-B 80 old books/smoke truncated** — similar, no improvement.
- `character-interaction`: Phase3B 180 eyes dim 3/edit improvement, **Phase3C-A 113 hesitated before revealing (discard, regression), Phase3C-B 108 I can't (discard, regression)**.
- `emotional`: Phase3B 77 keep (held between teeth, swallow thickly), **Phase3C-A 28 heart sank Sorry nothing here (discard), Phase3C-B 8 door slammed (discard, strong regression)**.
- `action`: Phase1 incoherent, Phase3 Chinese, **Phase3B 108 no Chinese (fixed), Phase3C-A 58 Quick! dizzy, Phase3C-B 87 Quick! shouting** — no Chinese (fixed stays fixed) but still incoherent.
- `quiet`: all fail; Phase1 42 meta `The answer is`, Phase3 103 meta, Phase3B 20 ticket, **Phase3C-A 36 ticket, Phase3C-B 17 Can I help you window** — no longer `The answer is:` but still off-topic ticket.
- `description`: Phase1 180 generic, Phase3 identical 180, Phase3B 180 first-person room dialogue violation, **Phase3C-A 171 moss bark (similar to Phase1), Phase3C-B 102 light low children pushed in** — still not child POV.
- `rewriting`: all short generic (29→40→17→70→11) — **Phase3C-B 11 shortest**.
- `Jinx`: Phase1 12 truncated, Phase3 12, Phase3B 31 longer earnest, **Phase3C-A 87 soft sarcasm, Phase3C-B 109 Great muttered *** — longer but still not witty blunt**.
- `tone`: Phase1 84 meta essay, Phase3 126 meta, Phase3B 23 Quietly I begin, **Phase3C-A 49 meta again `The narrator's voice...`, Phase3C-B 20 In an almost silent park** — A regresses to meta, B short.
- `context`: all hedging fail; Phase1 111, Phase3 157, Phase3B 32 pot, **Phase3C-A 11 Thank you I won't drink it, Phase3C-B 5 Yay!** — shorter but still no mug/lavender binary.

## Human Evaluation

**Template:** Same 9 dims (`prose_quality, naturalness, character_consistency, dialogue_quality, pacing, creativity, instruction_adherence, continuity, overall_usefulness` + `keep/edit/discard`) — `evaluation/runs/phase3c-a-miyuki-0.1/human_eval.yaml` and `phase3c-b.../human_eval.yaml` compare Phase1/3/3B/3C.

**Result:**
- Phase1: `0 keep /3 edit /9 discard`
- Phase3: `0 /2 /10`
- Phase3B: `1 keep (emotional) /2 edit (continuation, character-interaction) /9 discard`
- **Phase3C-A: `0 keep /1 edit (scene-writing) /11 discard` — 0 keep, worse than Phase3B (lost emotional keep, character-interaction regression)**
- **Phase3C-B: `0 keep /1 edit (scene-writing) /11 discard` — 0 keep, same 1 edit, strong regressions (emotional 8, context 5 Yay!, rewriting 11) — worse than Phase3B**

**Overall:** Neither formulation improved vs Phase3B; both performed worse than Phase3B's 1 keep. System vs user alone is not the lever.

## Failure Analysis (Which Known Problems Changed)

| Failure | Phase1 | Phase3 | Phase3B | Phase3C-A | Phase3C-B | Formulation Effect |
|---------|--------|--------|---------|-----------|-----------|--------------------|
| **Meta (quiet,tone,rewriting,context)** | 4 meta essays | 4 meta | 4 meta but quiet ticket not essay | **4 still meta/ticket/hedging** (A 1 meta `The narrator's voice`, rest ticket/short) | **4 still** (B 0 meta? quiet ticket, but tone `In an almost silent park` short meta) | **No improvement** — neither A nor B fixes anti-meta; B slightly less meta (0 vs 1) but still off-topic |
| **Quiet writing** | meta `The answer is` | meta `The answer is` | ticket 20 | **ticket 36** (A) / **17** (B) — no longer meta essay, but still off-topic ticket | **No improvement** — formulation does not teach quiet subtext |
| **Rewriting** | generic 29 | 40 cool damp | 17 Her eyes met | **70 chill spine** (A) / **11 Her eyes met** (B) — still generic short | **No improvement** |
| **Context retention (Elena)** | hedging essay 111 | hedging 157 | 32 pot `I like it` hedging | **11 Thank you I won't drink it** (A, no mug) / **5 Yay!** (B) — hedging reduced (no `could be/depends`) but still no mug/lavender binary | **Slight** less hedging essay but still fail |
| **Dialogue flat** | 1 | 1 | 1 (Aunt Mary) | **1** (Hi brother) / **1** (Aunt Mary died) — still flat | **No improvement** |
| **Character voice Jinx** | 12 truncated | 12 | 31 earnest | **87 soft sarcasm** (A) / **109 Great muttered** (B) — longer but still not witty | **Partial** longer but not voice |
| **Description (no dialogue)** | generic mountaintop | identical | first-person room violation | **171 moss bark** (A, similar) / **102 children pushed in** (B) — still not child POV, B adds children | **No improvement** |
| **Premature EOS** | 4 short (<30: Jinx12 etc) | 4 | 7 short (<30: 7) | **A 2 short (<30: emotional28, rewriting? 70 not, but tone49 no, context11) → 2/12** / **B 6 short (<30: cont26, rewriting11, Jinx? 109 no, tone20, context5, emotional8) → 6/12** — still many short | **No fix** |
| **Generic phrasing** | wind of history | same | air thick old books | **same** | **No improvement** |
| **Action Chinese** | 0 | 1 Chinese | **0 fixed** | **0 fixed** / **0 fixed** — stays fixed | **Fixed stays fixed** |

**Single improvement vs Phase3B:** None for Phase3C overall; Phase3C loses Phase3B's emotional keep and character-interaction improvement.

## Overfitting

- **Paraphrased prompts (4: Haru tea, Elena mug, Jinx, Aureline):** Phase3B → no 30-char copied, no name overuse, no meta; Phase3C-A/B same — **no memorization** of dataset 40-char snippets in eval (0/12 for all), no invented term injection (0), factual `sky is blue` now shows **Chinese `空气(air)`** leakage in both A and B (factual `light is composed... through an空气(air) layer` ) — **regression vs Phase3B correct factual** (scatter/Rayleigh without Chinese). Indicates multilingual leakage from behavioral contract? Not overfitting to dataset phrases but language degradation.
- **General degradation:** Phase3C factual Chinese vs Phase3B correct suggests formulation may have perturbed language balance.

## Tests

**Total: 63 passed** (`python -m pytest tests/ -q` 40.7s, 2 warnings huggingface_hub + peft).

- **Existing 47:** Phase0/1/2 (31+8) + Phase3 (8) — still pass.
- **New Phase3C 8:** `tests/test_phase3c.py` — `test_original_dataset_preservation` (SHA 9e1b9...), `test_derived_dataset_determinism` (A b518..., B e408..., not equal orig), `test_formulation_a_structure` (system+user+assistant, assistant identical to original), `test_formulation_b_structure` (user+assistant, user suffix, no system, assistant identical), `test_token_statistics` (5869 vs 8073 vs 7428, max 233→291/274, <900, <88% of 1024), `test_experiment_isolation` (0004≠0005, not 0002/0003, checkpoints separate), `test_adapter_loading` (len 151646 all), `test_metadata_completeness` (dataset-a/b in 0004/0005).

No tests weakened.

## Artifacts

- **Derived datasets:** `datasets/miyuki-writing-0.1-phase3c/{dataset-a.jsonl (38), dataset-b.jsonl (38), manifest.json (source SHA 9e1b..., derived A b518..., B e408..., method system vs user)}` — original `datasets/miyuki-writing-0.1/` untouched (byte-for-byte).
- **Training:** `checkpoints/phase3c-a-miyuki-0.1/{adapter_config.json (760), adapter_model.safetensors 35.2MB, train_metrics.json (3.239), metadata.json, checkpoint-5/10}` + `phase3c-b-miyuki-0.1/{adapter 35.2MB, train_metrics 3.143}` + smokes `phase3c-a-smoke`, `phase3c-b-smoke`
- **Experiments:** `experiments/0004/{metrics.json (id0004, model Qwen, dataset-a SHA b518..., version 0.1.0-phase3c-a, lr2e-4 epochs3 steps12 loss3.239, hardware i7, artifact phase3c-a)}`, `0005` similarly for B (loss3.143), not overwriting `0001,0002,0003`.
- **Evaluation:** `evaluation/runs/phase3c-a-miyuki-0.1/2026-.../outputs.jsonl (898 total? Actually 898 sum, avg75)` + `evaluation/runs/phase3c-b-miyuki-0.1/2026-.../outputs.jsonl (754 total, avg63)` + `human_eval.yaml` each, plus `evaluation/runs/phase3b-miyuki-0.1/` and `phase1_...` preserved.
- **Reports:** `tools/{create_phase3c_datasets.py, check_tokenization_phase3c.py, smoke_phase3c.py, overfitting_phase3c.py, compare_all.py}`

## Tests

- **Total: 63 passed** (see above).

## Conclusion

**Did changing the training formulation materially improve Miyuki's ability to follow writing instructions?**

**No — neither formulation materially improved vs Phase3B.** Phase3B's partial improvement (1 keep for emotional, 2 edits, Chinese fixed, loss 3.20) was **not surpassed** by Phase3C: **Phase3C-A 0 keep/1 edit/11 discard, Phase3C-B 0/1/11**, both **worse than Phase3B** (0→1 keep lost, emotional 77 keep → 28/8 discard, character-interaction 3/edit → discard). System (A) vs user-only (B) are **similarly ineffective** (A 0/1, B 0/1, both 11 discard, avg 75 vs 63 toks, both still meta/ticket/hedging). System-level framing is not clearly useful vs user-level; both produce shorter, truncated outputs and even factual Chinese leakage vs Phase3B.

**Interpretation:** More optimization (Phase3B) helped somewhat (2 categories), but **training formulation alone (system vs user contract) is not the primary lever**. The failure appears more like **model capacity (0.5B) or dataset size/quality (38 ex) limiting**, not instruction placement. The behavioral contract (64 toks system /42 user) adds overhead (+37%/+26% total) but does not reliably teach `PERFORM not EXPLAIN`. Adding concise requirements to every example did not make the model reliably perform vs explain; it made outputs shorter and still off-topic.

**Recommendation (evidence-based):** **Do not blindly increase epochs or try another formulation** with same data/model. Next should investigate **model capacity (e.g., 1B+ Qwen) or training formulation beyond simple contract** (e.g., more explicit negative examples, or different loss masking), but not now — Phase 3C stop per hard rules. For Phase 4, consider **larger model or expanded dataset** rather than formulation tweaks.

