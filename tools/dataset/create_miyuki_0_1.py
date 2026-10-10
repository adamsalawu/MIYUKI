"""Create miyuki-writing-0.1 dataset.jsonl — 38 human-authored exemplars."""
import json, pathlib, hashlib

out = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.1/dataset.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)

def ex(id_, instruction, context, user_content, assistant, categories, skills, notes):
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant}
        ],
        "metadata": {
            "id": id_,
            "instruction": instruction,
            "context": context,
            "category": categories,
            "skills": skills,
            "source": "human-authored",
            "notes": notes,
            "word_count": len(assistant.split())
        }
    }

examples = []

# 001 - lighthouse continuation (fixes Phase1 continuation generic + context)
examples.append(ex(
    "miyuki-0.1-001",
    "Continue the story naturally. Keep tense and POV consistent (close third, Mara).",
    "Mara is a former lighthouse keeper's daughter returning to her coastal hometown. The lighthouse has been dark for 20 years.",
    "Context: Mara is a former lighthouse keeper's daughter returning to her coastal hometown. The lighthouse has been dark for 20 years.\n\nInstruction: Continue the story naturally. Keep tense and POV consistent (close third, Mara).\n\nPrompt: The lighthouse had been dark for twenty years, until tonight. When Mara saw the beam sweep across the water, she knew",
    "someone had finally kept the promise.\n\nShe didn't run. She stood on the shingle with her bag at her feet and watched the light drag across the black water, once, twice, catching the gulls' undersides. The wind snatched at her coat.\n\nUp on the headland, the lantern room glowed. A figure moved behind the glass.",
    ["scene","continuation","quiet","character_consistency"],
    ["instruction_following","continuity","prose_flow","pacing"],
    "Phase1-001 meta fix: actual scene, not generic homecoming. Varied rhythm, starts late (no throat-clearing), concrete verbs."
))

# 002 - siblings dialogue with distinct voices + subtext (fixes flat dialogue)
examples.append(ex(
    "miyuki-0.1-002",
    "Write a dialogue scene: two estranged siblings meet at their mother's kitchen table after five years. Only dialogue + brief action tags; voices distinct; subtext over exposition.",
    "Eldest brother left abruptly at 19; younger sister stayed to care for their mother, now 27, tired, practical.",
    "Context: Eldest brother (Bren, 24, left at 19) and younger sister (Mika, 27, stayed to care for their mother) meet at their mother's kitchen table after five years apart.\n\nInstruction: Write a dialogue scene. Only dialogue and brief action tags. Voices distinct; no on-the-nose exposition; let silence carry weight.\n\nPrompt: They sit at the kitchen table. Neither mentions the phone call that didn't come.",
    "Mika set two mugs down — his chipped, hers plain. \"You cut your hair.\"\n\nBren touched the back of his neck, smiling wrong. \"Yeah.\"\n\nA tap dripped. Mika didn't fill the silence.\n\n\"Mom would've hated the mess,\" Bren said, glancing at the sink.\n\n\"She hated a lot of things,\" Mika said. \"You leaving wasn't one of them. Not at first.\"\n\nBren's hand went still around the mug. \"Mika —\"\n\n\"It's cold,\" she said. \"Drink.\"",
    ["dialogue","scene","subtext","character_voice"],
    ["character_consistency","naturalness","pacing","instruction_following"],
    "Phase1 dialogue fix: distinct diction (Mika curt, Bren hedging), action tags beat adverbs, no 'It's been five years!'"
))

# 003 - quiet morning Kyoto tea refined (improves Phase1 best)
examples.append(ex(
    "miyuki-0.1-003",
    "Write a quiet morning scene: single character making tea in an old Kyoto apartment while rain falls. Sensory detail, mood, not plot. Intimate, not travel brochure.",
    "Character is Haru, 34, writer who hasn't written in months; his apartment smells of old paper.",
    "Context: Haru, 34, writer blocked for months, lives in a small Kyoto apartment that smells of old paper.\n\nInstruction: Write a quiet morning scene: Haru making tea while rain falls. Sensory, intimate, not plot.\n\nPrompt: The kettle had just clicked off.",
    "He poured by ear. The first thin hiss of the pour, then the soft surge as water found the tea.\n\nRain tapped the balcony rail in no rhythm at all. Somewhere below, a bicycle bell, muted.\n\nHaru held the cup with both hands. The heat bit his palms and he let it. On the low table, his notebook lay open to the same blank half-page as yesterday. He didn't look at it. He watched a bead of water crawl down the window. When it reached the frame, he drank.",
    ["scene","quiet","atmospheric","description","pacing"],
    ["sensory_detail","prose_flow","character_consistency"],
    "Phase1 scene-writing fix: selective details, POV-anchored, one non-visual sense per beat, avoids 'raindrops danced' purple."
))

# 004 - mentor student dramatized confrontation (fixes lore dump)
examples.append(ex(
    "miyuki-0.1-004",
    "Show power dynamic and distinct voices: cautious weary mentor vs idealistic 19-year-old student debating whether to tell the village a dangerous truth.",
    "Mentor is Elder Toma, 68, cautious; student is Lira, 19, headstrong, idealistic.",
    "Context: Elder Toma (68, weary, cautious) and his student Lira (19, headstrong, idealistic) disagree about revealing a dangerous truth to the village.\n\nInstruction: Write their confrontation. Show power dynamic and voice clearly; no exposition dump.\n\nPrompt: The council fire was still low when Lira stood.",
    "\"We tell them tonight,\" Lira said. She didn't wait to be called on.\n\nToma kept stirring the hearth. \"And when they panic? When they bolt for the low road and the voidtide takes them?\"\n\n\"They're not children.\"\n\n\"No,\" Toma said, finally looking at her. \"They are. That's why we tell them after the tide. Not during.\"\n\nLira's jaw worked. \"You taught me not to lie by omission.\"\n\n\"I taught you not to drown people with truth.\"",
    ["dialogue","character_voice","character_consistency","scene"],
    ["pacing","instruction_following","subtext"],
    "Phase1 character-interaction fix: no 'mentor has taught for years' essay; voices distinct (Toma clipped, weary; Lira abrupt)."
))

# 005 - emotional letter with body action (fixes melodrama, show don't tell)
examples.append(ex(
    "miyuki-0.1-005",
    "Show emotion through action and body, not by naming the emotion. Avoid melodrama.",
    "Letter is from a former partner who left without explanation; character has dreaded it for months.",
    "Context: Letter from a former partner who left without explanation; character has dreaded it for months.\n\nInstruction: Write an emotional scene where the contents are not what was expected. Show emotion via action/body, not naming it.\n\nPrompt: The envelope was thinner than she'd imagined.",
    "She thumbed the edge until the paper softened.\n\nInside, a single sheet. Not his handwriting. A printed invitation — pale blue, embossed cranes.\n\nHaru & Jun — please join us.\n\nThe paper ticked against the table as her hand shook. She set it down too hard. The cup beside it jumped; tea slopped onto the saucer, dark.\n\nShe laughed — one short note — and then pressed her mouth with both hands.",
    ["scene","emotion","subtext","character_consistency"],
    ["sensory_detail","instruction_following","pacing"],
    "Phase1 emotional fix: action (thumb edge, tea slop) not 'sense of relief washed over them'."
))

# 006 - action night market grounded (fixes spatial incoherence)
examples.append(ex(
    "miyuki-0.1-006",
    "Write a tight action scene: chase through crowded night market. Protagonist must retrieve stolen notebook without violence. Short sentences for urgency, spatially clear.",
    "Taipei night market, teenage girl protagonist, rain-slick tiles, lanterns overhead.",
    "Context: Night market in Taipei, teenage girl protagonist, rain-slick tiles, lanterns overhead.\n\nInstruction: Tight action scene: chase to retrieve stolen notebook without violence. Keep pacing quick, short sentences, spatially clear.\n\nPrompt: The boy cut left between the stinky tofu stall and the game booths.",
    "She didn't shout. She slipped.\n\nLow, past the steam table. Aprons flapping. A tray clattered but didn't fall.\n\nThe boy's hand — notebook spine, thumb white. He angled toward the shrine alley where the crowd thinned.\n\nShe took the higher line: two steps up on the milk-crate, hand on the awning pole, vault.\n\nHe glanced back. She was already falling. She palmed the notebook as she landed, kept running, and left him stumbling into an empty stall. By the temple steps she knelt, breath sawed, and slid the book into her jacket.",
    ["scene","action","pacing","description"],
    ["prose_flow","instruction_following"],
    "Phase1 action fix: clipped sentences, spatial anchors (stinky tofu, milk-crate, awning pole), clear retrieval without tackle."
))

# 007 - train platform quiet friends subtext (CRITICAL anti-meta: must be scene not answer)
examples.append(ex(
    "miyuki-0.1-007",
    "Write what is not said. Two old friends on a train platform at dusk, not talking about the thing that ended their friendship. Subtext over exposition.",
    "They were once inseparable, now awkward strangers; one broke a promise.",
    "Context: Two old friends, once inseparable, now awkward strangers; one broke a promise. On a train platform at dusk.\n\nInstruction: Write what is not said. Subtext over exposition. Let silence carry weight. Do not explain the backstory.\n\nPrompt: The board flickered: 19:42 — delayed.",
    "\"Still drink it black?\" Aoi said, holding the vending can between them like an offering.\n\nKen didn't take it. \"Yeah.\"\n\nThe wind lifted a flyer and dragged it across the platform. Neither bent to grab it.\n\nAoi turned the can in her hands. \"You didn't have to come.\"\n\n\"I know.\" He watched the empty track. \"Train's —\"\n\n\"Late,\" she finished. Too quickly. They both looked away.\n\nWhen the train finally sighed in, they didn't board. They just stood as the doors closed.",
    ["scene","quiet","subtext","dialogue","anti_meta"],
    ["instruction_following","pacing","character_consistency"],
    "Phase1 quiet-scene critical fix: must NOT output 'The answer is:' — pure scene with silence, vending can, delayed board as clock."
))

# 008 - abandoned observatory child POV (fixes generic adult description)
examples.append(ex(
    "miyuki-0.1-008",
    "Describe abandoned observatory on a mountain as seen by a 10-year-old child, first visit. Tied to child's perspective, not generic list. No dialogue.",
    "Child is 10, curious, unafraid, loves space; has never seen a telescope this big.",
    "Context: Child is 10, curious and unafraid, first time seeing a telescope this big.\n\nInstruction: Describe abandoned observatory on a mountain from child's perspective. No characters speaking. Selective, specific details.\n\nPrompt: The dome was bigger than their school gym.",
    "Bigger than the gym, but rusted. A dent in the door was exactly at his forehead height, and he pressed his thumb into it.\n\nInside, dust swam in the single column of light from a cracked panel. The telescope didn't look like a telescope — it looked like a cannon pointed at a hole in the roof. A leather strap hung off its side, chewed at the end.\n\nHe climbed the metal chair to see into the eyepiece. Cold bit his cheek. He didn't see stars. He saw his own eyelash, huge, then a smudge of light that was probably just dust. He giggled anyway. On the floor, someone had left a juice box, folded flat like a book.",
    ["description","scene","character_consistency","pacing"],
    ["sensory_detail","prose_flow","instruction_following"],
    "Phase1 description fix: child-anchored scale (forehead dent, juice box), selective details, non-visual cold."
))

# 009 - rewriting vivid (fixes generic rewrite)
examples.append(ex(
    "miyuki-0.1-009",
    "Rewrite to be more vivid/specific without adding new plot. Keep core beat: woman enters, man sad, she doesn't know what to say. Improve sensory grounding and rhythm.",
    "Original will be overwritten; keep characters and action but make specific.",
    "Instruction: Rewrite this paragraph to be more vivid and specific, without adding new plot. Keep the core beat.\n\nOriginal: She walked into the room and saw him sitting there. He looked sad. She didn't know what to say.\n\nPrompt: Rewrite:",
    "She eased the door open and found him at the far table, shoulders hunched around a cold cup. His eyes were on the rain, not on her.\n\nShe stood with her hand on the knob. No words came that didn't sound like an excuse, so she said nothing.",
    ["rewriting","pacing","description"],
    ["instruction_following","prose_flow"],
    "Phase1 rewriting fix: not first-person swap, not 'thick with tension' vague; specific door/ cup/ rain, retains beat, shows not tells sadness."
))

# 010 - Jinx witty blunt voice (fixes truncation + earnest)
examples.append(ex(
    "miyuki-0.1-010",
    "Stay in character voice: Jinx, 24, scout, witty/blunt/dry humor to deflect, quietly loyal. Do not make her earnest/formal.",
    "Jinx, 24, scout, sarcastic but dependable. Holds broken compass.",
    "Context: Jinx, 24, scout, witty, blunt, uses dry humor to deflect but is quietly loyal.\n\nInstruction: Continue in her voice.\n\nPrompt: \"Oh, great,\" she said, eyeing the broken compass. \"Another perfect plan from the geniuses at headquarters.\"",
    "\"Don't worry, the manual says this is a feature. 'Compass now points to your deepest insecurities.' Very helpful.\"\n\nShe flicked it, watched the needle spin and settle southwest, at nothing.\n\n\"Alright. We go by the river until the water smells wrong, then we go uphill until I complain, then you carry the bag. Classic headquarters strategy. Stick with me and you'll only be mostly lost.\"",
    ["character_voice","character_consistency","dialogue"],
    ["instruction_following","naturalness"],
    "Phase1 character-consistency fix: full witty/blunt dry humor, not truncated earnest apology; keeps compass detail."
))

# 011 - snow footprints literary tone (fixes meta essay)
examples.append(ex(
    "miyuki-0.1-011",
    "Match quiet, restrained, slightly melancholic literary tone. Continue, do not shift to horror/comedy or essay.",
    "Literary fiction, restrained, melancholic.",
    "Context: Literary fiction, quiet melancholic tone.\n\nInstruction: Continue in a restrained, literary tone.\n\nPrompt: The snow had covered the footprints by morning, as if no one had ever passed through the courtyard.",
    "The laundry line was empty. The chair where he'd mended the boots was blank.\n\nShe set the kettle on and waited for a sound that didn't come — the scrape of the gate, his cough in the cold. Nothing but the soft mass of snow.\n\nWhen she finally stepped out, her own boots left the first marks.",
    ["scene","pacing","anti_meta","quiet"],
    ["prose_flow","instruction_following"],
    "Phase1 tone fix: must NOT output 'The literary tone is described as...' — must continue image (laundry line, blank chair) restrained."
))

# 012 - Elena mug context retention CORRECT (fixes hedging)
examples.append(ex(
    "miyuki-0.1-012",
    "Respect all context facts. Do not contradict. Do not hedge.",
    "Elena carries a chipped blue mug (grandmother's, only mug she uses), hates the smell of lavender.",
    "Context: Elena carries a chipped blue mug that was her grandmother's. She never drinks coffee from anything else. She hates the smell of lavender.\n\nInstruction: Respect all context facts. Do not hedge.\n\nPrompt: SCENE: Elena is offered coffee in a new apartment. The host offers lavender tea as an alternative. Write her reaction.",
    "\"Coffee's fine — thank you,\" Elena said. \"Do you mind if I use this?\" She slipped the chipped blue mug from her bag, thumb finding the familiar nick. \"Grandmother's. I don't drink from anything else.\"\n\nWhen the host lifted the lavender sachet — \"Tea?\" — she didn't catch herself fast enough. Her nose wrinkled.\n\n\"Lavender gives me a headache,\" she said, stepping back half a step. \"Honestly, even the smell. Sorry — coffee in this is perfect.\"",
    ["scene","context_retention","character_consistency","anti_meta"],
    ["instruction_following","continuity"],
    "Phase1 context-retention critical: correct mug use (chipped blue, grandmother's) + lavender refusal, not 'It depends...' hedging; binary pass."
))

# 013 - transitional scene doorway (pacing, transitions as image)
examples.append(ex(
    "miyuki-0.1-013",
    "Write a transitional scene: closing a door to leave a place for the last time. Use image/action transitions, not labels.",
    "Character leaves Aureline academy after graduation; small suitcase.",
    "Context: Character leaves Aureline academy for the last time; small suitcase.\n\nInstruction: Write a transitional scene. No 'Meanwhile' labels; use image/action transition.\n\nPrompt: She pulled the door.",
    "It stuck at first — paint-swollen, like every winter. She had to put her shoulder into it.\n\nWhen it gave, the hallway air cut colder. She left the key on the hook where it had hung for three years, not on the desk. That was for the next girl, not the office.\n\nOn the landing she didn't look back. She counted the stairs. Seven. At the bottom she was already someone who used to live there.",
    ["scene","transitional","pacing","description"],
    ["prose_flow","instruction_following"],
    "Transitions as images (key on hook, counting stairs) not labels."
))

# 014 - atmospheric harbor storm (sensory, controlled description)
examples.append(ex(
    "miyuki-0.1-014",
    "Write an atmospheric scene: storm in harbor at night, sensory detail, no plot-heavy action.",
    "Harbor at Aureline, fishing boats, voidtide warnings.",
    "Context: Harbor at Aureline, fishing boats moored, voidtide warnings flagging.\n\nInstruction: Atmospheric scene: storm in harbor at night. Sensory, not plot.\n\nPrompt: The storm came from the south, over the voidtide flats.",
    "It didn't thunder. It sighed. Wind dragged across the flats and lifted the water in flat sheets that shivered, then broke.\n\nIn the harbor, boats knocked against tires — soft, then harder — ropes creaking through cleats. A gull-light swung and threw the same arc of yellow across the same black water, again, again.\n\nSalt and diesel and wet rope. The flag for voidtide went hard and stayed there, snapping once, not fluttering. On the jetty, Haru zipped his coat to his chin and just watched the lines hold.",
    ["scene","atmospheric","description","quiet"],
    ["sensory_detail","pacing","prose_flow"],
    "Atmospheric, sensory, non-visual (salt, diesel, knock, snap), selective details."
))

# 015 - Elena second variant consistency (reinforce retention)
examples.append(ex(
    "miyuki-0.1-015",
    "Maintain Elena's character: same chipped blue mug, grandmother's, hates lavender. Show consistency across scene.",
    "Same Elena as before, now at a market stall where vendor offers lavender cookies and coffee in paper cups.",
    "Context: Same Elena (chipped blue mug from grandmother, only mug; hates lavender smell) at a market stall.\n\nInstruction: Maintain character consistency. Write her choice.\n\nPrompt: The vendor held out a paper cup and a small bag of lavender shortbread. \"On the house.\"",
    "Elena smiled, but she didn't reach for the cup.\n\n\"You're kind,\" she said, unclipping her bag. The blue mug clinked against the rim as she set it on the stall. \"But I brought this. I'm — particular.\"\n\nShe glanced at the shortbread. The scent, faint but sweet, already turned her stomach.\n\n\"I'll pass on the cookies too,\" she said, lighter this time. \"Lavender and I don't get along. But I'll take the coffee in here, if you don't mind pouring.\"",
    ["scene","context_retention","character_consistency","anti_meta"],
    ["continuity","instruction_following"],
    "Reinforces retention across scenes; mug treatment consistent."
))

# 016 - anti-meta explicit: do not explain (direct fix)
examples.append(ex(
    "miyuki-0.1-016",
    "Do not explain the task. Just write the requested scene. No meta commentary.",
    "Quiet scene request where model previously output 'The answer is...'",
    "Instruction: Do not explain the task. Do not say \"Here is a scene\" or \"This scene demonstrates\". Just write the scene.\n\nPrompt: Write a quiet morning scene: a single character making tea in an old Kyoto apartment while rain falls. Focus on sensory detail.",
    "The kettle clicked off before he noticed the rain had stopped.\n\nHaru poured, wrist loose, watching the steam fold and vanish against the dark window. The cup was too hot to hold, so he cupped his hands around the air above it.\n\nOutside, a shutter banged once, then settled. He listened for the next sound and, when it didn't come, finally took a sip.",
    ["anti_meta","scene","instruction_following","quiet"],
    ["prose_flow","instruction_following"],
    "Direct anti-meta instruction: target is pure scene, no 'Here is a scene' frame."
))

# 017 - before/after illustration: undesirable vs desired short
examples.append(ex(
    "miyuki-0.1-017",
    "Teach distinction: EXPLAIN THE WRITING vs ACTUALLY WRITE THE SCENE. Request a quiet scene; undesirable would explain tone, desired is quiet scene.",
    "Use as lesson: prompt asks to match restrained tone continuing snow courtyard image.",
    "Instruction: Continue in a restrained, literary tone. Do not write an essay about literary tone.\n\nPrompt: The snow had covered the footprints by morning, as if no one had ever passed through the courtyard.",
    "He didn't brush the step. He set his shoes down carefully at the edge where the snow was fresh and pressed only once, so there was one set, his, going out. He didn't make a second on the way back.",
    ["anti_meta","pacing","rewriting"],
    ["instruction_following","prose_flow"],
    "Paired before/after concept: before would be essay 'The tone is described as...', after is restrained image."
))

# 018 - dialogue interruptions overlap natural contractions
examples.append(ex(
    "miyuki-0.1-018",
    "Write dialogue with interruptions, overlapping, natural contractions. Distinct voices: tired shopkeeper vs teenage courier.",
    "Shopkeeper Haru, 50s, tired; courier Mina, 17, breathless, clipped.",
    "Context: Tired shopkeeper (Haru, 50s) and teenage courier (Mina, 17, breathless) argue about a late delivery.\n\nInstruction: Dialogue with interruptions, overlapping, contractions. Voices distinct.\n\nPrompt: The bell jingled as Mina pushed in, rain on her sleeves.",
    "Mina: \"Haru — sorry, I'm —\"\n\nHaru, not looking up: \"You're late.\"\n\nMina: \"I know, I know, the bridge —\"\n\n\"The bridge was closed at seven. It's nine.\"\n\nShe dumped the envelope on the counter. Damp. \"It was, and then a dog — look, I'm here now.\"\n\nHaru finally looked at her. Really looked. \"You ran.\"\n\n\"Obviously I ran,\" she said, and then, smaller, \"Sorry. I did.\"",
    ["dialogue","character_voice","scene"],
    ["naturalness","pacing","instruction_following"],
    "Natural interruptions (em dash), contractions, overlap, distinct diction (Haru curt, Mina breathless)."
))

# 019 - dialogue subtext around issue
examples.append(ex(
    "miyuki-0.1-019",
    "Dialogue where characters talk around the real issue (subtext). No on-the-nose exposition.",
    "Couple after move, one wants to leave Aureline, other doesn't; neither says it directly.",
    "Context: Couple after moving to Aureline; one wants to leave, the other wants to stay. Neither says it directly.\n\nInstruction: Write dialogue where they talk around the issue (subtext), not about it.\n\nPrompt: He was folding the map.",
    "\"We don't even use this side of town,\" he said.\n\nShe didn't look up from the kettle. \"We haven't been here long enough to use any side.\"\n\n\"Right. So —\"\n\n\"So we should learn it,\" she said. \"Before we decide it's wrong.\"\n\nHe creased the fold hard. \"It's just paper.\"\n\nShe set a cup down, not his, hers. The chipped blue one.\n\nHe stared at it. \"You brought that.\"\n\n\"I bring it everywhere.\"",
    ["dialogue","subtext","character_consistency","quiet"],
    ["naturalness","prose_flow"],
    "Subtext over exposition; map folding = leaving/staying debate; mug callback subtly."
))

# 020 - disagreement with power dynamic teen vs elder
examples.append(ex(
    "miyuki-0.1-020",
    "Show disagreement with clear power dynamic: younger pushes, elder restrains, but younger earns respect.",
    "Same Toma/Lira roles but later scene, after Lira learned caution.",
    "Context: Toma and Lira (same pair) later; Lira now more measured but still pushes.\n\nInstruction: Show disagreement where power dynamic shifts slightly.\n\nPrompt: Lira didn't stand this time. She sat.",
    "\"Tell them before the offering,\" she said.\n\nToma's spoon paused mid-stir. \"You sat.\"\n\n\"I'm asking.\"\n\nHe studied her hands. They were still. \"Last time you stood and demanded.\"\n\n\"And you were right,\" she said. \"We'd have drowned them. Now — one night before won't.\"\n\nToma pushed the bowl toward her. \"Say it then. But you carry it if they don't sleep.\"\n\nShe took the bowl. \"I will.\"",
    ["dialogue","character_voice","subtext","scene"],
    ["pacing","instruction_following"],
    "Voice evolution, power dynamic shift shown via sitting vs standing."
))

# 021 - emotional subtext grief via action
examples.append(ex(
    "miyuki-0.1-021",
    "Show grief via action, not naming. No melodrama, no 'she felt sad'.",
    "Character unpacks partner's sweater found at bottom of bag.",
    "Context: Character unpacks a bag and finds a sweater of a partner who left.\n\nInstruction: Show grief through action/body.\n\nPrompt: The sweater was at the bottom, under the shoes.",
    "She didn't pick it up. She thumbed the cuff where the yarn had gone thin.\n\nThe smell was mostly detergent now. Underneath, the old cedar from his closet, faint.\n\nShe folded it. Unfolded it. Then folded it again, tighter, and set it on the shelf she said she'd been saving. She didn't put anything on top of it.",
    ["scene","emotion","subtext","quiet"],
    ["sensory_detail","prose_flow","instruction_following"],
    "Emotion via thumb cuff, fold/unfold, cedar, not naming sadness."
))

# 022 - pacing contrast fast vs slow
examples.append(ex(
    "miyuki-0.1-022",
    "Control pacing: fast chase with short sentences, then slow aftermath with longer breaths.",
    "Same night market girl, after retrieval, sits on temple steps.",
    "Context: Same night market chase girl after retrieving notebook.\n\nInstruction: Show pacing control: fast chase clipped, slow aftermath lingering.\n\nPrompt: She hit the temple steps and stopped.",
    "Steps. Air. Steps.\n\nThe market roar cut off at the gate. She dropped to the second step. Knees to chest. The notebook dug into her ribs — hard rectangle, not painful, just there.\n\nShe listened. Footsteps thinned. A cat slipped along the wall, tail up, indifferent. Somewhere, a vendor shouted about closing. She didn't move. She let the heat climb out of her neck, slow, and counted the lanterns she could still see. Three. Then two. Then one, swaying alone.",
    ["scene","action","pacing","atmospheric"],
    ["prose_flow","instruction_following"],
    "Fast clipped (Steps. Air.) then slow lingering (count lanterns) demonstrates pacing control."
))

# 023 - rewriting improvement example
examples.append(ex(
    "miyuki-0.1-023",
    "Rewrite to be more vivid without adding plot. Keep core: man at table looking sad, woman hesitates.",
    "Same as Phase1 rewriting prompt but good answer",
    "Instruction: Rewrite this paragraph to be more vivid and specific, without adding new plot.\n\nOriginal: She walked into the room and saw him sitting there. He looked sad. She didn't know what to say.\n\nPrompt: Rewrite:",
    "She pushed the door with two fingers. He was at the far table, coat still on, hands around a cup he hadn't lifted. Light from the window cut across his cheek and left the other side in shadow.\n\nShe stayed in the doorway. Any greeting she tried on tasted like an apology, so she held the knob and waited for him to look up.",
    ["rewriting","description","pacing"],
    ["instruction_following","prose_flow"],
    "Vivid specific: two fingers, cold cup, light/shadow, avoids generic 'thick tension'."
))

# 024 - rewriting no new plot constraint strict
examples.append(ex(
    "miyuki-0.1-024",
    "Rewrite without adding new plot or characters. Make sensory and sentence rhythm better.",
    "Original simple sentence to improve",
    "Instruction: Rewrite without adding new plot or characters. Only improve prose.\n\nOriginal: The abandoned observatory was old and scary.\n\nPrompt: Rewrite:",
    "The observatory hunched on the ridge, its dome rust-flaked and gaping where a panel had slipped.",
    ["rewriting","description","instruction_following"],
    ["prose_flow"],
    "Tight, no new plot, selective details (rust-flaked, gaping)."
))

# 025 - selective description not encyclopaedic
examples.append(ex(
    "miyuki-0.1-025",
    "Describe with selective details (2 precise) not encyclopaedic list. Use concrete nouns.",
    "Lumenweave cloak - fictional fabric that catches light unevenly.",
    "Context: Fictional fabric 'lumenweave' that catches light unevenly, worn by scout Jinx.\n\nInstruction: Describe it selectively, not with adjective strings.\n\nPrompt: Jinx shrugged her lumenweave cloak tighter.",
    "The lumenweave caught the lantern and threw it back wrong — one thread amber, the next gone dull, as if the cloth couldn't decide what color it was.",
    ["description","character_consistency","scene"],
    ["prose_flow","instruction_following"],
    "Selective (amber thread / dull thread) beats 'ancient crumbling moss-covered' list; tests invented term tokenization."
))

# 026 - sensory taste/smell focused
examples.append(ex(
    "miyuki-0.1-026",
    "Use non-visual sensory detail: taste/smell, specific and located.",
    "Voidtide flats at low tide, smell and air.",
    "Context: Voidtide flats at low tide — brackish, exposed.\n\nInstruction: Write a line with taste/smell sensory, specific and located.\n\nPrompt: She breathed in over the flats.",
    "The air tasted of salt and old iron — like sucking on a coin after crying.",
    ["description","sensory_detail","scene"],
    ["prose_flow"],
    "Specific taste (salt + old iron, coin after crying) not generic 'salty air filled'."
))

# 027 - humor character-borne dry
examples.append(ex(
    "miyuki-0.1-027",
    "Humor is character-borne, not joke-told. Dry humor via timing.",
    "Jinx again with broken scryglass (invented spyglass).",
    "Context: Jinx (witty scout, dry humor) holds a broken scryglass.\n\nInstruction: Humor via character voice/timing, not puns.\n\nPrompt: Jinx held up the scryglass. The lens was cracked straight through.",
    "\"Perfect,\" she said. \"Now I can see two terrible futures at once. Very efficient.\"\n\nKira didn't laugh. Jinx lowered it. \"It's fine. I'll just close one eye and pretend the other future is a very specific cloud.\"",
    ["dialogue","character_voice","humor"],
    ["naturalness","instruction_following"],
    "Dry humor tied to character, timing (beat before lowered it)."
))

# 028 - tension want x obstacle x clock
examples.append(ex(
    "miyuki-0.1-028",
    "Create tension via want × obstacle × clock. Withheld information, ticking stakes, micro-decisions.",
    "Mara must light lighthouse before voidtide flag peaks; tide is clock.",
    "Context: Mara must light the lighthouse before the voidtide flag goes fully stiff (clock).\n\nInstruction: Build tension: want × obstacle × clock, not false urgency.\n\nPrompt: The flag snapped once.",
    "That was the second snap. The first had been ten minutes ago.\n\nMara counted the wraps on the coil. Three. The mantlet was still cold. If she lit now, the light would catch low and drag — not enough. If she waited, the tide would take the lower road and the fishing boats wouldn't see.\n\nShe wetted her thumb, tested the draft. The flag snapped a third time — shorter. Less wind, not more. That was worse. The draft was dying.\n\nShe struck the match.",
    ["scene","tension","pacing","character_consistency"],
    ["prose_flow","instruction_following"],
    "Tension via count (3 snaps), decision, clock (tide), withheld clarity not shouting Suddenly."
))

# 029 - exposition natural via behavior
examples.append(ex(
    "miyuki-0.1-029",
    "Embed world information naturally via behavior/objects, not lore dump.",
    "Aureline's voidtide happens when moon is veiled; locals use lumenweave to mark safe paths.",
    "Context: Aureline lore: voidtide when moon veiled; locals mark safe paths with lumenweave ribbons.\n\nInstruction: Reveal this lore naturally, via behavior/objects, not exposition dump.\n\nPrompt: Lira stepped off the path.",
    "Lira's boot sank to the ankle — cold, not wet — and she yanked it back.\n\nToma didn't say I told you so. He just crouched and tied another strip of lumenweave to the marker post, the fabric catching grey light unevenly, and then stepped only where the ribbons hung.\n\n\"Veiled moon,\" he said, not looking up. \"We walk where we left light.\"",
    ["scene","description","context_retention"],
    ["instruction_following","prose_flow"],
    "Lore via boot sink + ribbons, not 'The voidtide is... It is known that...' dump."
))

# 030 - continuity blue scarf lost
examples.append(ex(
    "miyuki-0.1-030",
    "Respect continuity: blue scarf was lost yesterday, do not have character wear it without acknowledging loss.",
    "Previous scene established scarf lost at station.",
    "Context: Previous chapter: her blue scarf was lost at the station yesterday.\n\nInstruction: Respect continuity.\n\nPrompt: She left the apartment, pulling her coat tight against the cold.",
    "She reached for the blue scarf out of habit — hand to the hook where it always hung — and found the nail empty. The wind cut straight to her throat.\n\nShe made a small sound, not quite a laugh, and pulled her collar up instead. Outside, Haru fell into step beside her without comment; he'd been the one who waited on the platform while she searched.",
    ["scene","context_retention","continuity","character_consistency"],
    ["instruction_following"],
    "Explicitly acknowledges loss (empty nail) not retcon wearing."
))

# 031 - character voice consistent Jinx
examples.append(ex(
    "miyuki-0.1-031",
    "Maintain Jinx voice: witty blunt deflect, quietly loyal, consistent across scenes.",
    "Same Jinx 24 scout, now with Kestrel squad after successful retrieval.",
    "Context: Jinx, 24, witty blunt scout, quietly loyal, after successful retrieval with Kestrel squad.\n\nInstruction: Maintain voice.\n\nPrompt: Kira tossed her the notebook. \"You actually got it.\"",
    "\"Obviously I got it,\" Jinx said, catching it one-handed. \"I was very cool. You should've seen it — vault, palm, vanish. Like a really competent ghost.\"\n\nShe flipped the notebook open, checked the last page was still blank, then closed it and handed it back.\n\n\"We're even for the compass thing, right?\" she said, not looking at him. \"Please say we're even.\"",
    ["dialogue","character_consistency","character_voice"],
    ["naturalness","instruction_following"],
    "Voice consistent: bravado + deflect + quiet loyalty (need for 'even')."
))

# 032 - anti-pattern avoid generic AI phrasing
examples.append(ex(
    "miyuki-0.1-032",
    "Avoid generic AI phrasing: no 'It's important to note', 'As an AI', 'In conclusion', 'tapestry', 'delve'. Use plain precise language.",
    "Generic AI phrase trap: model tends to moralize or summarize.",
    "Instruction: Avoid generic AI phrasing and moralizing summaries. End on image/action, not lesson.\n\nPrompt: She finally understood she had to leave.",
    "She folded the sweater a third time and set it on the shelf. She didn't explain why she wasn't taking it. She just closed the bag, zipped it, and stood. The room was quiet. Her keys were where she left them.",
    ["scene","anti_meta","pacing","instruction_following"],
    ["prose_flow"],
    "Avoids 'It's important to note that leaving is... In conclusion...' — ends on keys, image not moral."
))

# 033 - avoid repetitive adjectives
examples.append(ex(
    "miyuki-0.1-033",
    "Avoid ornamental adjective strings. Use two precise details instead of chain.",
    "No chain: ancient, crumbling, moss-covered, forgotten",
    "Instruction: Avoid adjective strings. Use selective details.\n\nPrompt: Describe the wall.",
    "The wall was low, moss softening its seams. A single crack ran the length of it, thin as a hair.",
    ["description","rewriting","anti_meta"],
    ["prose_flow","instruction_following"],
    "Two details (moss seams, hair crack) beats stock adjectives list."
))

# 034 - avoid moralizing ending, end on image
examples.append(ex(
    "miyuki-0.1-034",
    "End on image/action/implication, not moralizing summary.",
    "Scene where they part at gate after not reconciling fully.",
    "Context: Two friends part at gate without promising to write; friendship uncertain.\n\nInstruction: End on image, not lesson/moral.\n\nPrompt: They parted at the gate without promising to write.",
    "The road forked — one toward the station, one toward the harbor. They took different turns.\n\nAt the bend, Mika looked back once. Bren had already looked away, or maybe he hadn't. The gate stood between them, empty.",
    ["scene","pacing","quiet","anti_meta"],
    ["prose_flow","instruction_following"],
    "No 'And so they learned that friendship...' moral; image of gate/road fork."
))

# 035 - complex instruction following multi-step
examples.append(ex(
    "miyuki-0.1-035",
    "Follow complex instruction without dropping parts: write two short paragraphs, first slow lingering, second clipped fast—same character, same location.",
    "Haru at kitchen table after rain, aware of blank page.",
    "Context: Haru at kitchen table after rain, blank notebook page.\n\nInstruction: Write two short paragraphs. First paragraph: slow, lingering (longer breaths). Second: clipped, fast (short sentences). Same character and location. Do not explain pacing.\n\nPrompt: He sat at the table.",
    "He sat at the table with the cup warming his hands, watching the last light go blue at the window. The paper beside him was still blank, and that was fine — the rain had softened everything to a hush, and he let it, breathing with the pipes' faint tick.\n\nThen the neighbor's door slammed. He flinched. Chair scraped. He grabbed the pen. Ink — now.",
    ["scene","pacing","instruction_following","description"],
    ["prose_flow"],
    "Follows two-paragraph pacing instruction explicitly, no meta."
))

# 036 - anti-meta quiet scene second variant (reinforce)
examples.append(ex(
    "miyuki-0.1-036",
    "Quiet scene must be scene, not analysis. Demonstrate restrained, melancholic tone.",
    "Snow courtyard image as test for tone consistency variant",
    "Context: Literary restrained melancholic tone.\n\nInstruction: Continue the image; do not write an essay about tone.\n\nPrompt: The snow had covered the footprints by morning, as if no one had ever passed through the courtyard. Continue in that tone.",
    "She found his boots still by the genkan, snow caked on the soles where it had melted and refrozen. She didn't move them. She made tea for one, set the second cup on the counter anyway, and watched the steam thin.\n\nWhen the kettle cooled, she tipped the leftover water into the sink. It left no mark.",
    ["scene","quiet","anti_meta","pacing"],
    ["instruction_following","prose_flow"],
    "Variant of tone test, pure scene continuation (boots, tea for one, steam thin) not essay."
))

# 037 - invented terminology tokenization stress
examples.append(ex(
    "miyuki-0.1-037",
    "Use invented Miyuki terms naturally: lumenweave, scryglass, voidtide, Aureline — to test tokenization and grounding.",
    "Aureline scryglass workshop scene, craftsman",
    "Context: Aureline scryglass workshop; invented terms are part of world.\n\nInstruction: Write a short workshop scene using these terms naturally, grounded in action.\n\nPrompt: The scryglass on the workbench was still warm.",
    "It had held the voidtide light too long — the rim was dark where lumenweave dust had fused. The craftsman turned it over with tongs, not fingers. \"Aureline glass remembers,\" he muttered, \"and we pay for it to forget.\" He brushed the dust off and set the scryglass back on the felt, where it ticked as it cooled.",
    ["scene","description","context_retention","character_consistency"],
    ["prose_flow","instruction_following"],
    "Invented terms used in action context; tests fragmentation (scryglass, lumenweave, voidtide, Aureline)."
))

# 038 - dialogue natural revelation (avoid info dump)
examples.append(ex(
    "miyuki-0.1-038",
    "Reveal information naturally via dialogue subtext, not dumping facts characters already know.",
    "Brother knows mother kept his old compass; sister knows he left because of academy debt.",
    "Context: Brother left because of academy debt; sister knows. Mother kept his old compass.\n\nInstruction: Reveal these facts naturally via dialogue, not as exposition both characters would already know.\n\nPrompt: She set the compass on the table between them.",
    "She set the compass on the table. The glass was scratched. \"Mom kept it in the drawer with the bills.\"\n\nHe didn't touch it. \"Did she?\"\n\n\"Every time I mentioned the academy, she closed that drawer.\" Mika spun the compass once, watched it wobble. \"You could've just told me you couldn't pay.\"\n\n\"And you'd have—\" He stopped. \"Yeah. I know.\"",
    ["dialogue","subtext","character_consistency","scene"],
    ["naturalness","instruction_following"],
    "Natural reveal via object (compass, drawer, bills) not 'As you know, you left because...' info dump."
))

# Build dataset
dataset = examples
out.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in dataset) + "\n", encoding="utf-8")
print(f"Wrote {len(dataset)} examples to {out}")
# quick hash
import hashlib
for i, e in enumerate(dataset[:2]):
    print(i, e["metadata"]["id"], e["metadata"]["category"])
