"""Create miyuki-writing-0.3: 120 from 0.2 + 34 new targeted = 154 total. All human-authored."""
import json, pathlib, hashlib

orig_path = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.2/dataset.jsonl")
orig_examples = [json.loads(l) for l in orig_path.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"0.2: {len(orig_examples)} ex")

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

N = []
# ============ TARGET A: CHARACTER VOICE (8) ============
N.append(ex(
    "miyuki-0.3-121",
    "Show Mara (practical, spare, short sentences) and Lira (19, idealistic, eager run-on sentences) hearing the ferry is cancelled. Distinct rhythm, no exposition.",
    "Ferry to Kestrel cancelled due to washed road; Mara (practical) and Lira (19, idealistic) at the dock.",
    "Context: Ferry to Kestrel cancelled due to washed road; Mara (practical, spare) and Lira (19, idealistic, eager) at the dock.\n\nInstruction: Show their distinct reactions to the same cancellation. Mara uses short spare sentences; Lira uses eager run-on sentences. No exposition.\n\nPrompt: The ferryman shook his head. No crossing today.",
    "Mara checked the rope, checked the sky, and sat down. Done. She would mend the net instead.\n\nLira was already talking before he finished shaking his head, about how they could walk around by the shrine path because it was only a little flooded and last year someone did it and it was fine, and maybe if they hurried they could still make the noon tide, couldn't they?",
    ["dialogue","character_voice","character_consistency","scene"],
    ["naturalness","instruction_following"],
    "TARGET A distinct_voice: Mara spare short vs Lira eager run-on, same news, behavior not explanation."
))
N.append(ex(
    "miyuki-0.3-122",
    "Bren (abrupt, 3-6 word sentences) and Mika (tired, practical, longer explanatory) discuss mother's missed medicine dose.",
    "Bren (24, abrupt) and Mika (27, tired practical) at kitchen table, mother's medicine missed.",
    "Context: Bren (24, abrupt, short) and Mika (27, tired, practical) at kitchen table; mother's evening medicine was missed.\n\nInstruction: Distinct voices: Bren in 3-6 word bursts, Mika in longer tired practical sentences. No exposition dump.\n\nPrompt: The pillbox showed Tuesday still full on Thursday.",
    "Bren stared at it. \u201cMissed two.\u201d He pushed it away. \u201cMy fault.\u201d\n\nMika picked it up before it slid off the edge, because that was what she always did, caught things before they fell. \u201cYou were on the road Tuesday, and I was covering the market stall, so neither of us was here at six, which is when she takes it, so we need a six o'clock plan, not blame.\u201d",
    ["dialogue","character_voice","subtext"],
    ["naturalness","instruction_following"],
    "TARGET A distinct_voice: Bren bursts vs Mika long practical, same guilt differently."
))
N.append(ex(
    "miyuki-0.3-123",
    "Toma (68, weary, speaks in short proverb-like observations) and Kira (young courier, breathless, present-tense) describe the same storm.",
    "Toma (68, weary) and Kira (courier, breathless) describe storm over harbor.",
    "Context: Toma (68, weary, proverb-like) and Kira (young courier, breathless, present-tense) describe the same storm over the harbor.\n\nInstruction: Same storm, two voices: Toma in short weather-worn observations, Kira in breathless present-tense. No exposition.\n\nPrompt: The storm hit the harbor just after noon.",
    "Toma watched the flats go white. \u201cWind before rain. Always.\u201d He tied the shutter and came inside. \u201cIt will take what it wants.\u201d\n\nKira burst in still running even though she had stopped, rain flying off her sleeves, talking fast: I am coming down the hill and the wind is shoving me sideways and the awning is flapping like it wants to leave and I am thinking the noodles will get wet!",
    ["character_voice","description","scene"],
    ["naturalness","instruction_following","sensory_detail"],
    "TARGET A distinct_voice: Toma clipped weather proverbs vs Kira breathless present-tense."
))
N.append(ex(
    "miyuki-0.3-124",
    "Haru (quiet, internal, notices small sounds) and Jinx (blunt, external, names what she sees) wait in the same dark hallway.",
    "Haru (34, quiet internal) and Jinx (24, blunt external) wait in dark hallway during power cut.",
    "Context: Haru (34, quiet, internal) and Jinx (24, blunt, external) wait in a dark hallway during a power cut.\n\nInstruction: Same hallway, distinct perception: Haru notices small sounds internally, Jinx names what she sees bluntly. No exposition.\n\nPrompt: The hallway went dark all at once.",
    "Haru listened. The fridge ticked as it died. Somewhere a tap kept dripping, which seemed louder now, and his own breathing, which he tried to make quieter and couldn't.\n\nJinx said, \u201cGreat. Dark hallway. Smells like old socks and someone's soup. Anyone got a light, or are we doing the spooky standing thing all night?\u201d",
    ["character_voice","quiet","subtext"],
    ["instruction_following","naturalness"],
    "TARGET A distinct_voice: Haru internal sound-noticing vs Jinx blunt external naming."
))
N.append(ex(
    "miyuki-0.3-125",
    "Mika (dry, tired humor, one-line deflections) and Bren (earnest, over-explains when nervous) are asked the same question: are you hungry?",
    "Mika (dry tired humor) and Bren (earnest over-explainer) asked if hungry after long day.",
    "Context: Mika (dry, tired humor) and Bren (earnest, over-explains when nervous) are asked if hungry after a long day.\n\nInstruction: Same question, distinct answers showing humor vs earnestness. No exposition.\n\nPrompt: Haru held up the pot. \u201cHungry?\u201d",
    "Mika didn't look up from the sink. \u201cOnly for sleep. Sleep with a side of sitting down.\u201d\n\nBren sat up straight, as if the question were a test. \u201cYes, actually, I am, because I skipped lunch to finish the railing, and then the bus was late, so I have had nothing since the rice ball at noon, which was small, so yes, please, thank you.\u201d",
    ["dialogue","character_voice","humor"],
    ["naturalness","instruction_following"],
    "TARGET A distinct_voice: Mika one-line dry deflection vs Bren earnest over-explanation."
))
N.append(ex(
    "miyuki-0.3-126",
    "Lira (apologizes by over-talking) and Toma (apologizes by doing a small task) after the same mistake: breaking a marker post.",
    "Lira and Toma both broke a lumenweave marker post; show different apologies.",
    "Context: Lira and Toma each broke a lumenweave marker post; both must apologize.\n\nInstruction: Same mistake, distinct apologies: Lira over-talks, Toma does a small task silently. No exposition.\n\nPrompt: The marker post lay in two pieces.",
    "Lira was already talking: I am so sorry, I didn't see it, I was looking at the tide not the post, I will tie a new one right now, I know where the spare ribbons are, I will do two to make up for it, really \u2014\n\nToma said nothing. He picked up both pieces, carried them to the shed, and came back with a new post, a mallet, and a strip of lumenweave. He set them down next to her and waited.",
    ["character_voice","scene","subtext"],
    ["instruction_following","naturalness"],
    "TARGET A distinct_voice: apology via over-talking vs via silent task."
))
N.append(ex(
    "miyuki-0.3-127",
    "Kira (counts things when nervous) and Haru (goes still when nervous) hear the same unexpected knock.",
    "Kira counts when nervous, Haru goes still; same knock at night.",
    "Context: Kira (counts when nervous) and Haru (goes still when nervous) hear an unexpected knock at night.\n\nInstruction: Same knock, distinct nervous habits shown through action, not named. No exposition.\n\nPrompt: Three knocks, after midnight.",
    "Kira counted without meaning to: three knocks, four seconds apart, two breaths between each. She was already counting the floorboards to the door \u2014 six \u2014 when Haru put a hand on her sleeve.\n\nHaru didn't move at all. He stood with his hand on her sleeve and his eyes on the door, and breathed so shallowly the curtain beside him didn't stir.",
    ["character_voice","quiet","emotion","subtext"],
    ["instruction_following","prose_flow"],
    "TARGET A distinct_voice: nervous counting vs nervous stillness, shown not named."
))
N.append(ex(
    "miyuki-0.3-128",
    "Elena (polite, precise, says sorry with specific reason) and Jinx (blunt, deflects with joke) both spill tea. Distinct cleanup.",
    "Elena (polite precise) and Jinx (blunt joke) both spill tea.",
    "Context: Elena (polite, precise) and Jinx (blunt, deflects with joke) both spill tea on a table.\n\nInstruction: Same spill, distinct cleanup voices. No exposition.\n\nPrompt: Tea went across the table in one quick sheet.",
    "Elena was already moving: \u201cSorry \u2014 my elbow, entirely my fault \u2014\u201d She folded the cloth in half to make a thick edge and drew it toward herself in one steady pull, then wrung it over the bowl.\n\nJinx looked at the spreading tea and said, \u201cExcellent. I have baptized the table. It is now holy and sticky.\u201d She mopped at it with her sleeve, which made it worse, and grinned without looking up.",
    ["character_voice","humor","scene"],
    ["naturalness","instruction_following"],
    "TARGET A distinct_voice: precise apology-cleanup vs joke-sleeve-mop."
))
# ============ TARGET B: EXACT CONTEXT (8) ============
N.append(ex(
    "miyuki-0.3-129",
    "Binary fact: the notebook (not phone) was left upstairs (not downstairs) in the morning (not evening). Continue correctly.",
    "Notebook left upstairs in the morning; phone was downstairs in the evening (distractor). Must use notebook/upstairs/morning.",
    "Context: The notebook was left upstairs in the morning. The phone was downstairs in the evening.\n\nInstruction: Continue the scene using the exact established facts: notebook, upstairs, morning. Do not swap.\n\nPrompt: She went back for it, taking the stairs two at a time.",
    "She took the stairs two at a time, breath loud on the landing. The notebook was where she had left it that morning, spine up against the pillow, pages still bent from where she had held it open with her thumb. She grabbed it and didn't touch the phone at all.",
    ["scene","context_retention","continuity"],
    ["instruction_following","continuity"],
    "TARGET B binary_context: notebook/upstairs/morning vs phone/downstairs/evening distractors, natural continuation."
))
N.append(ex(
    "miyuki-0.3-130",
    "Elena hates lavender but likes chamomile; she uses grandmother's (not mother's) chipped blue mug. New host offers chamomile correctly.",
    "Elena: grandmother's chipped blue mug, hates lavender, likes chamomile. Host offers chamomile tea.",
    "Context: Elena uses her grandmother's (not mother's) chipped blue mug. She hates lavender but likes chamomile.\n\nInstruction: Write Elena accepting chamomile in her grandmother's mug, declining nothing. Preserve exact facts.\n\nPrompt: The new host held up a tin. \u201cChamomile? No lavender, I promise.\u201d",
    "\u201cChamomile I can do,\u201d Elena said, and meant it. She slipped the chipped blue mug from her bag \u2014 thumb to the nick the way her grandmother used to hold it \u2014 and held it out. \u201cGrandmother drank this every night. Pour it in here and I am home.\u201d",
    ["scene","context_retention","character_consistency"],
    ["instruction_following","continuity"],
    "TARGET B binary_context: grandmother not mother, chamomile accepted vs lavender hated, mug used."
))
N.append(ex(
    "miyuki-0.3-131",
    "Kestrel compass always points southwest; river is true direction. Show character ignoring compass, following river.",
    "Compass broken SW, river true; new character tempted to trust compass.",
    "Context: Jinx's compass always points southwest regardless of direction; the river runs north to the village and is the true guide.\n\nInstruction: Show the character correctly ignoring the compass and following the river.\n\nPrompt: The needle settled southwest, toward the cliffs.",
    "Southwest was the cliffs, which everyone knew was wrong \u2014 the village was north, up the river. Jinx didn't even frown at the needle anymore. She checked the water instead: quick and loud over the stones, running north. She turned north with it and left the compass swinging in her pocket.",
    ["scene","context_retention","continuity"],
    ["instruction_following","continuity"],
    "TARGET B binary_context: SW wrong/cliffs vs river north true, natural navigation."
))
N.append(ex(
    "miyuki-0.3-132",
    "Blue scarf lost at the station (not the market), yesterday (not last week). Haru's sister asks about it; he must answer correctly.",
    "Scarf lost at station yesterday; sister misremembers market last week.",
    "Context: Haru's blue scarf was lost at the station yesterday (not the market, not last week).\n\nInstruction: Haru corrects his sister gently with exact facts, in dialogue.\n\nPrompt: \u201cDid you lose it at the market last week?\u201d his sister asked.",
    "\u201cStation,\u201d Haru said. \u201cYesterday. I had it on the platform and then I didn't.\u201d He touched his bare neck out of habit. \u201cNot the market. I haven't been to the market in weeks.\u201d",
    ["dialogue","context_retention","character_consistency"],
    ["instruction_following","continuity","naturalness"],
    "TARGET B binary_context: station/yesterday vs market/last-week correction, natural dialogue."
))
N.append(ex(
    "miyuki-0.3-133",
    "Voidtide flag red means stay high; green means go low. Flag is red; character must stay high despite shortcut temptation.",
    "Flag red = stay high; low road is shortcut temptation.",
    "Context: Voidtide flag is red, meaning stay high (not low). The low road is a tempting shortcut.\n\nInstruction: Show the character correctly choosing high road per red flag.\n\nPrompt: The low road was empty and fast-looking. The flag above it was red and hard.",
    "The low road was empty and fast-looking, which made it worse. The flag above it was red and hard, snapping once, not fluttering. Haru looked at it, then at his shoes, then took the high steps. They were longer and steeper and his bag banged his hip the whole way, but the flag stayed red, so he stayed high.",
    ["scene","context_retention","description"],
    ["instruction_following","continuity"],
    "TARGET B binary_context: red=high vs low-shortcut temptation, correct choice."
))
N.append(ex(
    "miyuki-0.3-134",
    "Lumenweave ribbons mark safe path only where tied; beyond last ribbon is unsafe. Character must stop at last ribbon.",
    "Ribbons only where tied; beyond last is unsafe.",
    "Context: Lumenweave ribbons mark the safe path only where tied; beyond the last ribbon is unsafe voidtide flats.\n\nInstruction: Show the character correctly stopping at the last ribbon and waiting.\n\nPrompt: She counted three ribbons, then nothing.",
    "Three ribbons: one on the fence post, one on the bent reed, one on a stone that barely stuck up. After the stone, nothing \u2014 grey flats all the way to the mist. She stopped with her toe just short of where the third ribbon's light ended and called back, \u201cLast one here. I need the next.\u201d",
    ["scene","context_retention","description"],
    ["instruction_following","continuity"],
    "TARGET B binary_context: three ribbons then nothing = stop, not proceed."
))
N.append(ex(
    "miyuki-0.3-135",
    "Jinx's bag is heavy with scryglass tools (not light with lunch). New porter offers to carry it lightly; she refuses correctly.",
    "Bag heavy with tools, not light lunch.",
    "Context: Jinx's bag is heavy with scryglass tools (not a light lunch).\n\nInstruction: Show the bag's correct heaviness when a porter underestimates it.\n\nPrompt: The porter reached for her bag with one hand. \u201cI will take that, miss, it looks light.\u201d",
    "\u201cCareful,\u201d Jinx said, but too late \u2014 the porter took it one-handed and staggered sideways, eyes wide. The tools inside shifted all at once with a low clank. \u201cIt is not lunch,\u201d she said, taking it back with both hands. \u201cIt is my entire workshop trying to escape.\u201d",
    ["scene","context_retention","character_voice"],
    ["instruction_following","continuity","naturalness"],
    "TARGET B binary_context: heavy tools vs light lunch, one-hand stagger."
))
N.append(ex(
    "miyuki-0.3-136",
    "Scryglass is warm after voidtide use (not cold/room-temperature). Character must test heat correctly, not grab it.",
    "Scryglass warm after voidtide; must hover hand, not grab.",
    "Context: Kira's scryglass is always warm after voidtide use (not cold).\n\nInstruction: Show correct heat-testing (hover, not grab).\n\nPrompt: The voidtide light faded. Kira reached for the scryglass.",
    "She stopped her hand an inch above it. Heat came off it in a faint shimmer, the way it always did after voidtide \u2014 warm, not hot enough to burn, but wrong to grab. She fanned her fingers over it once, nodded to herself, and picked up the tongs instead.",
    ["scene","context_retention","description"],
    ["instruction_following","continuity","sensory_detail"],
    "TARGET B binary_context: warm shimmer hover vs cold grab, tongs."
))
# ============ TARGET C: JINX / WITTY (6) ============
N.append(ex(
    "miyuki-0.3-137",
    "Jinx witty deflection when map is upside down; short punchy, not every line a joke, believable.",
    "Jinx with upside-down map, Kira watching.",
    "Context: Jinx holds the map upside down and Kira notices.\n\nInstruction: Jinx deflects with quick dry wit, then fixes it. Short punchy responses, one joke, then competent.\n\nPrompt: Kira pointed. \u201cYour map is upside down.\u201d",
    "\u201cIt is a test,\u201d Jinx said without missing a beat. \u201cTo see if you are paying attention. You passed.\u201d She turned it right-side up in one smooth motion and kept walking, which rather ruined the effect of having been caught.",
    ["dialogue","character_voice","humor"],
    ["naturalness","instruction_following"],
    "TARGET C witty_voice: one-beat test joke then competent fix, not joke chain."
))
N.append(ex(
    "miyuki-0.3-138",
    "Jinx dry reaction to scryglass fogging; playful but does the work.",
    "Scryglass fogs; Jinx wipes and reports.",
    "Context: Scryglass fogs during watch; Jinx on duty.\n\nInstruction: Dry reaction, playful deflection, then competent report. No catchphrase repeat.\n\nPrompt: The scryglass fogged from edge to center.",
    "\u201cOh, good, it is dreaming,\u201d Jinx said. She breathed on it once, which was the wrong thing, and wiped it properly with the corner of her sleeve. \u201cNorth road clear to the bend. South road \u2014 also clear, also boring. Report ends.\u201d",
    ["dialogue","character_voice","scene"],
    ["naturalness","instruction_following"],
    "TARGET C witty_voice: dreaming joke once, then proper wipe + report."
))
N.append(ex(
    "miyuki-0.3-139",
    "Jinx confident dialogue when tide plan fails; short punchy, loyal underneath.",
    "Tide plan fails; squad looks at Jinx.",
    "Context: Tide plan fails; squad looks at Jinx for next move.\n\nInstruction: Confident, short, playful deflection covering quick thinking. Loyal, not flippant.\n\nPrompt: The tide took the lower road an hour early. Everyone looked at Jinx.",
    "\u201cRight. New plan, same as the old plan but wetter.\u201d Jinx was already retying her boots. \u201cWe go high, we go now, and nobody dies beautifully on my watch. Move.\u201d",
    ["dialogue","character_voice","pacing"],
    ["naturalness","instruction_following"],
    "TARGET C witty_voice: wetter-plan joke once, then clear orders, loyal."
))
N.append(ex(
    "miyuki-0.3-140",
    "Jinx playful deflection when asked if scared; honest underneath, not joke chain.",
    "Asked if scared before night watch.",
    "Context: Kira asks Jinx if scared before night watch.\n\nInstruction: Playful deflection first, then brief honesty. Personality via dialogue, not exposition.\n\nPrompt: \u201cAre you scared?\u201d Kira asked, very quietly.",
    "\u201cTerrified,\u201d Jinx said cheerfully. \u201cI am shaking so hard the compass thinks we are walking.\u201d Then, quieter, because Kira was actually asking: \u201cA little. The normal amount. Stay where I can see your lantern?\u201d",
    ["dialogue","character_voice","emotion","subtext"],
    ["naturalness","instruction_following"],
    "TARGET C witty_voice: cheerful terrified joke, then honest quiet request."
))
N.append(ex(
    "miyuki-0.3-141",
    "Jinx short punchy responses while carrying heavy bag uphill; humor from situation, not puns.",
    "Carrying heavy bag uphill, Haru asks if need help.",
    "Context: Jinx carrying heavy bag uphill; Haru offers help.\n\nInstruction: Short punchy responses, humor from situation (heavy bag), not forced puns. Believable.\n\nPrompt: \u201cWant help with that?\u201d Haru asked, falling into step.",
    "\u201cNope.\u201d Step. Breathe. \u201cGot it.\u201d Step. The bag banged her knee. \u201cOkay. Maybe a little help. Just the top handle. Do not tell anyone I said that.\u201d",
    ["dialogue","character_voice","humor"],
    ["naturalness","instruction_following","pacing"],
    "TARGET C witty_voice: Nope/Got it staccato, then honest ask, situation humor."
))
N.append(ex(
    "miyuki-0.3-142",
    "Jinx dry reaction to waiting; confident, not complaining at length.",
    "Waiting for tide to pass, long wait.",
    "Context: Waiting for voidtide to pass, long boring wait.\n\nInstruction: Dry confident reaction to waiting, one joke, then settles. Not complaining chain.\n\nPrompt: The tide would not pass for another hour. Jinx sat down.",
    "Jinx sat down, stretched her legs, and announced to no one, \u201cIntermission.\u201d She pulled out a spare ribbon and started braiding it into her sleeve for something to do. After a minute she was humming, off-key, on purpose.",
    ["scene","character_voice","quiet","humor"],
    ["naturalness","instruction_following"],
    "TARGET C witty_voice: Intermission once, then braiding/humming, settles."
))
# ============ TARGET D: ACTION + OBJECT (6) ============
N.append(ex(
    "miyuki-0.3-143",
    "Object tracking: notebook established on shelf, knocked during chase, must land and stay tracked.",
    "Notebook on shelf, chase knocks it, track where it lands.",
    "Context: Notebook on the shelf by the door; a chase through the room knocks it.\n\nInstruction: Track the notebook: establish, knock, land, keep track. Spatial clarity.\n\nPrompt: The notebook was on the shelf by the door, spine out.",
    "The notebook was on the shelf by the door, spine out, easy to grab. When Bren shouldered past chasing the dog, his elbow caught the shelf. The notebook tipped, slid, and landed flat under the table \u2014 still in the room, now dusty. Mika saw where it went and didn't pick it up yet; she stepped over it carefully on her way to the dog.",
    ["scene","action","pacing","description"],
    ["instruction_following","prose_flow"],
    "TARGET D object_tracking: shelf spine-out, elbow catch, under-table dusty, stepped over."
))
N.append(ex(
    "miyuki-0.3-144",
    "Object tracking: keys on hook, grabbed in hurry, dropped in courtyard, must stay dropped until picked up.",
    "Keys on hook, grabbed, dropped in courtyard.",
    "Context: Keys on the hook by the door; grabbed in a hurry.\n\nInstruction: Track keys: hook, grab, drop in courtyard, stay dropped.\n\nPrompt: She grabbed the keys off the hook without looking.",
    "She grabbed the keys off the hook without looking and ran. In the courtyard the strap snagged her sleeve \u2014 a hard tug \u2014 and the keys flew. They hit the wet stones with a flat clink and lay there, half under a leaf. She kept running toward the gate; the keys stayed where they fell, glinting each time the lantern swung.",
    ["scene","action","pacing"],
    ["instruction_following","prose_flow"],
    "TARGET D object_tracking: hook grab, sleeve snag, wet stones leaf, stay, lantern glint."
))
N.append(ex(
    "miyuki-0.3-145",
    "Object tracking: chair moved to reach shelf, must stay moved and affect next movement.",
    "Chair moved to reach shelf, stays moved, blocks path.",
    "Context: Chair against wall; moved to reach high shelf.\n\nInstruction: Track chair: against wall, dragged, stays and blocks path.\n\nPrompt: Haru dragged the chair to the shelf to reach the top box.",
    "Haru dragged the chair to the shelf \u2014 legs screeching once on the tile \u2014 and stood on it to lift the top box down. He set the box on the table and turned with the empty space in his hands, forgetting the chair entirely, and walked straight into it. His shin found the edge he had left in the middle of the room.",
    ["scene","action","description"],
    ["instruction_following","prose_flow"],
    "TARGET D object_tracking: wall drag screech, middle-of-room stays, shin finds it."
))
N.append(ex(
    "miyuki-0.3-146",
    "Object tracking: cup caught before falling, must show hand trajectory and final hold.",
    "Cup tipping off table, caught mid-air.",
    "Context: Cup tipping off the table edge.\n\nInstruction: Track cup: tip, trajectory, catch, final hold. Cause-effect readable.\n\nPrompt: The cup went over the edge.",
    "The cup went over the edge \u2014 not pushed, just nudged by her sleeve as she turned. It dropped straight, turning once. Mika's hand shot out without her deciding to move it, palm up, and the cup landed in it with a soft slap, tea sloshing but staying in. She stood there holding it at knee height, surprised at her own hand.",
    ["scene","action","pacing"],
    ["instruction_following","prose_flow","sensory_detail"],
    "TARGET D object_tracking: sleeve nudge, drop turn once, palm-up slap, knee-height hold."
))
N.append(ex(
    "miyuki-0.3-147",
    "Object tracking: drawer opened for compass, must stay open and affect light/sound.",
    "Drawer opened for compass, stays open, drawer sound/light change.",
    "Context: Compass in drawer with bills; drawer opened.\n\nInstruction: Track drawer: closed, opened, stays open affecting scene.\n\nPrompt: She pulled the drawer open for the compass.",
    "She pulled the drawer \u2014 wood sticking a little at the back, the way it always did \u2014 and the bills inside shifted with a paper whisper. The compass was on top, needle stuck southwest. She took it and didn't close the drawer; it stayed open behind her, a dark gap in the dresser, letting the lamplight fall in a thin stripe across the floor while they talked.",
    ["scene","action","description"],
    ["instruction_following","prose_flow"],
    "TARGET D object_tracking: stick at back, paper whisper, stays open dark gap, lamplight stripe."
))
N.append(ex(
    "miyuki-0.3-148",
    "Object tracking: phone picked up, screen shows time, must stay in hand and constrain next actions (one hand free).",
    "Phone picked up showing 19:42, stays in hand constraining actions.",
    "Context: Phone on table showing 19:42; picked up.\n\nInstruction: Track phone: table, pick up, stays in hand constraining next actions to one hand.\n\nPrompt: The phone showed 19:42. She picked it up.",
    "The phone showed 19:42, bright in the dim kitchen. She picked it up with her right hand and kept it there, screen still lit, while she tried to pour tea with her left \u2014 badly, splashing the counter. She didn't set the phone down to do it properly; she needed to see if the time changed. It didn't. 19:42 stayed while the tea went cold.",
    ["scene","action","quiet"],
    ["instruction_following","prose_flow"],
    "TARGET D object_tracking: 19:42 bright, right-hand hold, left-hand pour splash, stays."
))
# ============ TARGET E: VIEWPOINT (6) ============
N.append(ex(
    "miyuki-0.3-149",
    "Child viewpoint: 10-year-old at night market, notices height-level things (hands, smells at waist), misunderstands adult talk literally.",
    "10-year-old at night market, height-level perception, literal misunderstanding.",
    "Context: 10-year-old at night market with older sister, first evening out.\n\nInstruction: Child viewpoint: what a 10-year-old at waist height perceives; literal misunderstanding of adult phrase. No adult summary.\n\nPrompt: The market was too crowded to see over.",
    "It was too crowded to see over, so he watched hands instead. A hand holding noodles high. A hand counting coins. A hand that almost hit his face and pulled back just in time, and the man said sorry to the air above his head. His sister said they were killing time before the bus, and he wondered, for a second, how you kill time and whether it would bleed.",
    ["description","scene","character_consistency"],
    ["instruction_following","sensory_detail","naturalness"],
    "TARGET E viewpoint_constraint: waist-height hands, sorry to air above head, kill-time literal."
))
N.append(ex(
    "miyuki-0.3-150",
    "First-person Haru limited knowledge: he does not know Mika already called the doctor; his narration must not reveal it.",
    "Haru first-person, does not know Mika called doctor.",
    "Context: Haru narrates in first person; he does NOT know Mika already called the doctor that morning.\n\nInstruction: First-person limited: Haru must wonder/assume, must not state the call as fact.\n\nPrompt: I found Mika by the phone, looking relieved.",
    "I found Mika by the phone, looking relieved, which made no sense because the doctor hadn't called back as far as I knew. Maybe she had finally slept. Maybe the kettle working again was enough. I wanted to ask but the relief on her face made me not want to break it, so I just said the water was hot.",
    ["scene","quiet","subtext","character_consistency"],
    ["instruction_following","prose_flow"],
    "TARGET E viewpoint_constraint: first-person limited, as-far-as-I-knew, maybe-slept, doesn't state call."
))
N.append(ex(
    "miyuki-0.3-151",
    "Third-person limited Mara: she does not know why lighthouse keeper left; narration must not explain motive.",
    "Mara third-person limited, keeper motive unknown.",
    "Context: Mara watches the lighthouse keeper leave; his motive is unknown to her.\n\nInstruction: Third-person limited to Mara: show his leaving, do not explain why. No omniscient motive.\n\nPrompt: The keeper locked the lantern room and walked down without a word.",
    "The keeper locked the lantern room \u2014 three turns of the key, loud in the stairwell \u2014 and walked down without a word. Mara watched his hands, which were steady, and his mouth, which wasn't. She thought of three reasons and believed none of them. He didn't look back at the light, which told her nothing except that he wasn't looking.",
    ["scene","subtext","character_consistency"],
    ["instruction_following","prose_flow"],
    "TARGET E viewpoint_constraint: limited to Mara hands/mouth, three reasons none believed, no motive given."
))
N.append(ex(
    "miyuki-0.3-152",
    "Emotionally restrained Toma narration: grief shown only via task precision, no feeling words.",
    "Toma grieving, narration restrained, only task precision.",
    "Context: Toma grieving a loss; narration must be emotionally restrained, no feeling words (sad, grief, lonely).\n\nInstruction: Restrained narration: grief only via task precision. No feeling words.\n\nPrompt: Toma mended the marker post the morning after.",
    "Toma mended the marker post the morning after. He measured the depth twice with a stick before setting it. He drove it straight, checked with his eye from two sides, tied the ribbon with two knots and trimmed the ends even. He tested it with one firm pull. It held.",
    ["scene","quiet","emotion","subtext"],
    ["instruction_following","prose_flow"],
    "TARGET E viewpoint_constraint: restrained, measured twice, two knots, trimmed even, pull test, no feeling words."
))
N.append(ex(
    "miyuki-0.3-153",
    "Kira smell-focused perception: she navigates market by smell, not sight; prose reflects smell-first ordering.",
    "Kira navigates by smell first, sight second.",
    "Context: Kira navigates the night market by smell first, sight second.\n\nInstruction: Smell-first perception ordering: smell sentence before sight sentence. No visual list.\n\nPrompt: Kira entered the market with her eyes watering from the rain.",
    "First the smoke, then the fat underneath it \u2014 pork, then something sweet burning at the edges. Only then the stalls themselves, low shapes in the steam. She turned toward the sweet-burning one before she could see what it sold, because her nose got there first and her feet followed.",
    ["description","scene","sensory_detail"],
    ["instruction_following","sensory_detail","naturalness"],
    "TARGET E viewpoint_constraint: smell-first ordering (smoke/fat/sweet before stalls), nose before feet."
))
N.append(ex(
    "miyuki-0.3-154",
    "Elderly vendor limited knowledge: he knows prices and weather, not the children's backstory; must not narrate what he can't know.",
    "Elderly vendor limited to prices/weather, cannot know children's history.",
    "Context: Elderly vendor at market stall; he knows prices and weather, not the children's backstory.\n\nInstruction: Limited vendor viewpoint: only prices/weather/observed actions, no backstory narration.\n\nPrompt: Two children stopped at his stall, whispering.",
    "Two children stopped at his stall, whispering, the taller one holding the smaller one's sleeve. Buns were four coppers, day-old three \u2014 he said so without being asked, because that was usually the question. The rain had made the paper wrappers soft. Whether they were siblings or runaways or just lost, he couldn't say; they bought one bun and split it walking away.",
    ["scene","description","character_consistency"],
    ["instruction_following","prose_flow"],
    "TARGET E viewpoint_constraint: prices/weather/observed only (four coppers, soft wrappers), couldn't say backstory."
))

out_path = pathlib.Path("C:/Aether/RisingStar/MIYUKI/datasets/miyuki-writing-0.3/dataset.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)
all_examples = orig_examples + N
out_path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in all_examples) + "\n", encoding="utf-8")
print(f"Wrote {len(all_examples)} (120 + {len(N)}) to {out_path}")
import collections
cats = collections.Counter()
for e in all_examples:
    for c in e["metadata"]["category"]:
        cats[c] += 1
print(dict(cats))
