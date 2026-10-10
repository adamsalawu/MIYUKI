# Training Signal Audit — Phase 4 (Qwen2-0.5B)

**Date:** 2026-09-25  
**Model:** `Qwen/Qwen2-0.5B-Instruct` (vocab 151646, chat template True)  
**Dataset example:** `miyuki-0.1-001` (Mara lighthouse, 168 tokens total)  
**File:** `src/miyuki/training/train.py` (before Phase 4) + `tools/loss_audit.py` + `tools/training_target_test.py`

## Finding

**Current MIYUKI implementation (Phase 0–3C) trains on ALL tokens.**

- **Method:** `Dataset.from_dict({"text": texts})` where `texts = tokenizer.apply_chat_template(msgs, add_generation_prompt=False)` (full conversation: system + user + assistant + `<|im_end|>`), then `tokenize` → `input_ids`, `DataCollatorForLanguageModeling(tokenizer, mlm=False)` sets `labels = input_ids.clone()` for **all** tokens.
- **Diagnostic (one real example, `miyuki-0.1-001`):**
  - Full conversation: 168 tokens total
  - System+User prefix: 93 tokens ( `<|im_start|>system You are a helpful assistant.<|im_end|> <|im_start|>user Context: Mara...<|im_end|> <|im_start|>assistant` )
  - Assistant writing: 75 tokens (`someone had finally kept the promise...<|im_end|>`)
  - Special tokens: 6 (`<|im_start|>`×3, `<|im_end|>`×3) — all counted
  - **Current labels:** `labels = input_ids` for all 168, `sum(labels==-100) = 0` → `SYSTEM 93/168 trainable, USER 93 trainable, ASSISTANT 75 trainable, SPECIAL 6 trainable`
  - **Expected TRL assistant-only:** `93 masked (-100), 75 trainable` → `45% trainable, 55% masked`

**Evidence:**
- `evaluation/runs/phase4_loss_audit.json` (from `tools/loss_audit.py`): `total_tokens 168, before 93, after 75, current_masked 0, system_included true, user_included true, assistant_included true, special_included true, conclusion: MIYUKI currently trains on ALL tokens`
- `evaluation/runs/phase4_loss_audit_diagnostic.json` (from `tools/training_target_test.py` after fix): `full_len 168, mask_len 93, assistant_len 75, system_user masked, assistant trainable, special before masked (-100), special after trainable, no_prompt_becomes_target true, no_assistant_masked true`

**Implication:** Model is penalized for not predicting **user/system prompts** (e.g., "Context: Mara is... Instruction: Continue...") as well as assistant writing. This wastes ~55% of loss on copying prompts, not learning to write. It also trains the model to predict `<|im_start|>system` etc., which is not the desired writing behavior. Phase 1–3B evaluations showed no improvement despite loss ↓ (3.76→3.20), partially because loss was diluted.

## TRL Support

Modern TRL `SFTTrainer` supports `dataset_kwargs={"assistant_only_loss": True}` or `DataCollatorForCompletionOnlyLM` where only assistant responses contribute. MIYUKI did **NOT** use this — it used vanilla `Trainer` + `DataCollatorForLanguageModeling` (all tokens). Not a TRL bug, just MIYUKI's `train.py` not implementing it.

Qwen2 chat template **can** support assistant-only masking safely: the prefix `system+user+<|im_start|>assistant` is a prefix of the full conversation (verified `full_ids[:93] == prefix_ids[:93]` True for both original and derived A). No special handling needed beyond prefix length. No tokenizer replacement required.

## Fix (Smallest Clean Correction)

**Preserve existing path** (default `assistant_only_loss False` for 0001–0005) and **add opt-in** `assistant_only_loss True` for Phase 4.

- **Config:** `src/miyuki/config.py:TrainingConfig` added `assistant_only_loss: bool = False` (default preserves previous experiments). Phase 4 config sets it `True`.
- **Training:** `src/miyuki/training/train.py` — when `assistant_only_loss True`, build `Dataset.from_dict({"text": texts, "_idx": ...})` and `tokenize_fn_masked` that:
  1. Takes `raw` example and `text` (full conversation)
  2. Computes `prefix_text = tokenizer.apply_chat_template(msgs[:-1], add_generation_prompt=True)` and `full_ids` / `prefix_ids`
  3. `mask_len = min(len(prefix_ids), len(full_ids))`, `labels = [-100]*mask_len + full_ids[mask_len:]`
  4. Handles special tokens: those before assistant (5 of 6) become -100, final `<|im_end|>` after assistant stays trainable.
  5. Saves `input_ids, labels, attention_mask` per example.
  6. Uses custom `assistant_collator` that pads `input_ids` with `pad_token_id` and `labels` with `-100` (instead of `DataCollatorForLanguageModeling` which would overwrite labels with `input_ids`).
- **When False:** Keeps old `Dataset.from_dict({"text": texts})` + `DataCollatorForLanguageModeling(mlm=False)` (all trainable) — previous experiments unchanged.

**Special-token handling:** Verified: 5 specials before assistant (`<|im_start|>system`, `<|im_end|>`×2, `<|im_start|>user`, `<|im_start|>assistant`) → masked, 1 final `<|im_end|>` after assistant → trainable. Correct.

**No prompt becomes target:** User/system tokens are -100, so model not penalized for predicting "Context: Mara...".

**No assistant masked:** All 75 assistant tokens trainable.

**Tests:** `tests/test_training_target.py` (to be added) covers masked vs trainable counts, special handling, deterministic prefix match.

**Diagnostic artifacts:** `evaluation/runs/phase4_loss_audit.json` (before) and `evaluation/runs/phase4_loss_audit_diagnostic.json` (after fix) saved.

**Limitation:** None for Qwen2 — chat template is prefix-stable. No tokenizer change needed.

---

*Next: Use this fix in Phase 4 training config `configs/training/phase4_lora.yaml` with `assistant_only_loss: true`.*
