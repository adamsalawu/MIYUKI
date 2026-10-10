# MIYUKI Failure Taxonomy (Phase 5.0)

*Machine-readable version: `evaluation/failure_taxonomy.json`. Sources: all human evals P1-4C + 1.5B + expanded.*

## Baseline (quality reference)
- Phase 4: Qwen2-0.5B + LoRA-120 (assistant-only) - **5/4/3, meta 0, Chinese 0, EOS 0**

## Eliminated (3)
explaining_instead_of_writing, meta_commentary, summarization_instead_of_continuation

## Improved (13)
atmosphere, avoiding_narrator_like_dialogue, continuation, description, formatting, premature_endings, previous_actions, quiet_scenes, sensory_grounding, silence, tone, transitions, unnecessary_moralizing

## Partially Improved (17)
cause_effect, character_reactions, character_voice, color, do_not_change_x_constraints, emotional_behavior, emotional_writing, forbidden_details, natural_responses, pacing, paragraph_rhythm, reactions, relationships, relationships, rewriting, sentence_variation, style_constraints

## Reduced (1)
generic_ai_phrases

## Failing (26)
action_scenes, action_sequencing, believable_movement, binary_facts, character_knowledge_boundaries, character_positioning, character_specific_vocabulary, child_viewpoint, conversational_rhythm, distinct_voices, exact_facts, hallucinated_context, internal_knowledge_boundaries, interruptions, multi_character_scenes, negative_facts, object_identity, object_retrieval, ownership, personality_consistency, physical_continuity, requested_details, restricted_knowledge, spatial_relationships, subtext, third_limited

## Regressed In 4C (2)
imagery, surreal_incoherent_imagery

## Failing 15B Only (2)
list_answers, unnecessary_disclaimers

## Untested (3)
goals, implication, length_constraints

## Untested Isolated (1)
first_person

## Untested Dedicated (1)
chronology

## Surviving Failure Priority (Phase 5 targets)
1. **exact/binary context (mug/lavender, negative facts)** (failed P1-4C all) - 0.3 B-target 8 examples not learned at 3 epochs
2. **viewpoint constraints (child, limited knowledge)** (failed P1-4C all) - 0.3 E-target 6 examples; description regressed
3. **object/detail retrieval in action** (failed P1-4C all) - 0.3 D-target 6 examples; notebook still never retrieved
4. **multi-character voice collapse** (failed P1-4C all) - 0.3 A-target 8 examples; single-scene Jinx improved
5. **action continuity/spatial** (failed P1-4C all) - 0.3 D-target partial
6. **dialogue subtext/interruptions** (failed P1-4C mostly) - 0.3 A/C partial; quiet-scene PASS
7. **surreal imagery (NEW, overfitting)** (failed P4C only) - not data - training duration issue
8. **rewriting vividness loss** (failed P4C regression) - P4 (0.2) keep was best

## Eliminated by Phase 4 recipe
- meta commentary (answer-is, tone-essay)
- hedging essays (could-be/depends)
- explaining instead of writing
- summarization instead of continuation
- moralizing
- premature EOS in original suite
- Chinese leakage in original suite

## Hardware Constraints
- **training_backend:** Qwen2-0.5B-Instruct + PEFT (only trainable)
- **blocked:** Qwen2.5-1.5B LoRA (NO-GO: 2.82GB < 3.0GB after load+LoRA)
- **eval_speed_variance:** 2.84 tok/s (Sep 24) -> 0.4-0.5 tok/s (Sep 25-26) under memory pressure
- **overfit_threshold_observed:** 154 examples x 3 epochs -> train 0.25 / eval plateau 2.86