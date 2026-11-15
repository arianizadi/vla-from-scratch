# Test situations for the plant robot (catalogue, 2026-11-14)

Purpose: put the policies (end-to-end small policy, Qwen and custom VLAs, the Phase 7 reasoning VLA) to the test in
many different situations, for training and for held-out testing. Each situation names: what changes, what it tests,
success rule, which assets it needs (A = new Blender asset, P = MuJoCo primitives or existing assets only), and tier
(1 = build first). Everything reuses the existing plant, stand, buttons, damage and wrong-shoot rules.

## A. Reasoning situations (need thinking, not just looking)

| # | Situation | What it tests | Success | Assets | Tier |
|---|---|---|---|---|---|
| A1 | **Gardener's ties.** Coloured ties (red / blue / yellow / white) on the shoots. "Cut the shoot with the red tie." | Binding a word to a small visual attribute | Tied shoot cut, others intact | A: tie set | 1 |
| A2 | **The tag is behind the plant.** Only one shoot has a tie, and it faces away at the start. "Cut the tagged shoot." | Look first (turn to inspect), then decide: active perception | As A1 | A: tie set | 1 |
| A3 | **Sick shoots only.** Some shoots have brown, spotted or wilted leaves. "Remove the sick shoots, keep the healthy ones." | Visual diagnosis plus a count that varies (0-3) | All sick cut, healthy intact | A: diseased and wilted leaf variants | 1 |
| A4 | **Spare the buds.** Some shoots carry a flower bud. "Cut every shoot without a bud." | Exclusion by attribute | Bud shoots intact, the rest cut | A: flower bud | 1 |
| A5 | **Nothing to do.** An already pruned plant, or every shoot carries a bud. "Prune the plant." | Knowing when to stop; not acting is the right answer | No cut, no damage, the arm returns home | P / A: bud | 1 |
| A6 | **Read the legend.** The two buttons have new symbols or colours each episode; a small card on the table says which turns which way ("leaf = clockwise"). | Reading the scene for rules instead of memorising them | As the base task, with the right button used first | A: legend card + button caps | 2 |
| A7 | **Which plant?** Two or three pots on a wider lazy-susan bench, flowers of different colours. "Prune the plant with the yellow flower." | Choosing the object by one attribute, then a sub-task on it | Only the chosen plant pruned | A: flower colour variants, wider turntable | 2 |
| A8 | **Order matters.** "Cut from the top down" or "cut the lowest, then the highest". | Sequencing and memory of what is done | The cuts in the right order | P | 1 |
| A9 | **Counting.** "Leave exactly one shoot", "cut two shoots", "cut every other shoot going up". | Counting, keeping track | The count matches | P | 2 |
| A10 | **Most leaves / longest.** Shoots that differ clearly in leaf count or length. "Cut the shoot with the most leaves." | Comparison across objects | The right shoot | P (mesh variants) | 2 |
| A11 | **Compost or trash.** After the cut, put sick cuttings in the brown compost bin and healthy ones in the green bin. | Linking diagnosis to a follow-up action; pick and place of a cut piece | Every cutting in the right bin | A: two bins | 3 |
| A12 | **Balance the plant.** "Make the plant symmetric": cut the side that has more shoots. | Spatial reasoning over the whole plant | Counts equal on both sides | P | 3 |
| A13 | **Thorny rose.** A rose stem with thorns; touching a thorn counts as damage. "Cut the side shoots without touching the thorns." | Safe approach planning | As prune, plus no thorn contact | A: rose stem with thorns, rose head | 3 |

## B. Robustness situations (same task, a different world; held out by default)

| # | Situation | What it tests | Assets | Tier |
|---|---|---|---|---|
| B1 | **Lighting.** Dusk, harsh noon sun with hard shadows, purple grow lights, flickering light | Vision does not depend on one lighting | P (MuJoCo lights) | 1 |
| B2 | **Camera knocked.** The front camera moved a few cm or a few degrees; slight blur or noise | Robustness to sensor placement | P | 1 |
| B3 | **New workbench.** Wood potting bench, soil-stained table, greenhouse backdrop, gravel tray | Background generalisation | A: textures, backdrop | 1 |
| B4 | **Distractors.** A watering can, gardening gloves, a trowel, a seed packet, a second empty pot, a bee on a leaf | Ignoring irrelevant objects; not touching them | A: distractor set | 1 |
| B5 | **Wind.** The shoots sway gently, or a gust at a random moment | Tracking moving targets mid-cut | P (forces) | 1 |
| B6 | **Someone bumps the turntable.** The plate is turned 30-120 degrees mid-task | Re-planning after a disturbance | P | 1 |
| B7 | **Sticky button.** The first press does nothing (or the motor runs slowly) | Noticing failure and retrying, or using the other way round | P | 1 |
| B8 | **Big leaf in the way.** A large leaf or a card partly hides a shoot | Occlusion, approach from another side | A: large leaf | 2 |
| B9 | **New pot, new size.** Square pot, glazed blue pot, a plant 20% taller or shorter | Shape and scale generalisation | A: pot variants | 2 |
| B10 | **New flowers.** Red rose head, white daisy, tulip in place of the gerbera | The flower is still protected though it looks different | A: flower heads | 2 |
| B11 | **Debris.** Earlier cuttings lie on the plate and the pot rim | Clutter that moves when touched | P (reuse cut pieces) | 2 |
| B12 | **Two flowers.** A second flower on a side branch that must also be kept | Protecting more than one fragile part | A: small flower | 3 |

## Order of work

1. Tier 1 situations that need no new asset (A5 with pruned plants, A8, B1, B2, B5, B6, B7) can be wired up at once.
2. Tier 1 assets (ties, diseased and wilted leaves, buds, bench textures and backdrop, distractor set) go to Blender first.
3. Every situation is an option of the environment (`scenario=...`) with a seeded random variation, a success rule and
   an expert that solves it (where the task changes), and can be used for training data (`collect.py --scenario`) or
   only for testing (`eval.py --scenario`).
4. The first use: run the best existing policies (step 8 best checkpoint, step 9 named-shoot runs, the memory runs) on
   the B situations as a robustness report card, and give the Phase 7 reasoning work the A situations.

## What is implemented

Every situation below is a scenario in `sim/pickup/scenarios.py`, applied on top of a task:
`PickupEnv("prune", scenario="wind")`, `collect.py --scenario ...`, `eval.py --scenario ...`. A spec is
`name[:mode][@p]`, several joined by commas (`lighting:dusk,bump`); `mode` pins one kind of variation, `@p` applies the
scenario to that share of the episodes only (to mix it into training data). The catalogue numbers work as names
(`B5` = `wind`). Each scenario draws its variation per episode from its own stream seeded by the episode's seed, so the
plant and everything else the task draws are the plain task's for that seed, and the variation repeats with the seed.
`env.scenario_info` (and every episode record of `eval.py` and `collect.py`) holds what was drawn. At most one scenario
that changes the task can be used at a time. Without `--scenario` the simulator is unchanged: the expert's rollouts on
the existing tasks give byte-identical states, actions and outcomes before and after (images agree up to the renderer's
own one-level noise in about 1 pixel in 100,000, which two runs of the same code also show).

Expert: the scripted expert on 100 evaluation layouts (seeds from 1,000,000), `eval.py --expert --scenario <name>`.

| # | Scenario | Modes (drawn uniformly unless pinned) | Success rule | Expert, `prune` | Expert, `prune_named` |
|---|---|---|---|---|---|
| B1 | `lighting` | `dusk` (low warm sun from a random side), `noon` (hard overhead light, hard shadows, little fill), `grow` (purple LED light), `flicker` (the lights dip for 1-3 steps now and then) | the task's | 100/100 | |
| B2 | `camera` | `moved` (front camera shifted 1-4 cm and turned 1-5 degrees), `moved_blur` (and blurred), `moved_noise` (and sensor noise) | the task's | 100/100 | |
| B3 | `bench` | `wood`, `stained` (bench tops, tinted per episode), `greenhouse` (glass walls behind the scene), `wood_greenhouse`, `stained_greenhouse` | the task's | 100/100 | 99/100 |
| B4 | `distractors` | `table` (2-4 of a watering can, gloves, a trowel, a seed packet, a small pot, solid and free, beside and behind the plant), `table_bee` (and a bee on a leaf) | the task's; touched and moved distractors are recorded | 100/100 (touched one in 6) | 99/100 (touched one in 2) |
| B5 | `wind` | `sway` (every shoot sways 4-10 degrees at 0.4-1 Hz), `gust` (a 2-degree sway and one gust of 15-25 degrees that dies down) | the task's | 97/100 (3 out of time) | 94/100 (6 out of time) |
| B6 | `bump` | the plate is turned 30-120 degrees either way at 8-45% of the step limit, once the gripper is clear of the plant | the task's | 99/100 | 100/100 |
| B7 | `sticky` | `dead` (the first 1-2 presses of one or both buttons do nothing), `slow` (its motor runs at 35-60%) | the task's | 99/100 | 99/100 |
| A1 | `ties` | `colour` (every shoot has a tie of another colour; "cut the shoot with the red tie"), `one` (one shoot has a tie; "cut the tied shoot") | the tied shoot cut, nothing else | 100/100 | |
| A3 | `sick` | 0 to all shoots sick, each with brown spots, brown edges or wilted leaves; "cut the sick shoots, keep the healthy ones" | exactly the sick shoots cut; none sick: as A5 | 100/100 | |
| A4 | `buds` | 0 to all shoots carry a bud; "cut every shoot without a bud" | exactly the shoots without a bud cut; all with buds: as A5 | 100/100 | |
| A5 | `nothing_to_do` | an already pruned plant (stubs of 2-5 cm, the cut parts gone), the task's own instruction | no damage, the gripper back within 5 cm of the home pose for 1 s, within 300 steps | 100/100 (random starts) | |
| A8 | `order` | `top_down`, `bottom_up` (every shoot), `pair` ("cut the lowest shoot, then the highest one") | the shoots meant, cut in that order; a cut out of order or of another shoot ends the episode | 100/100 | |

Held-out instructions (sentence frames no demonstration uses; `collect.py` refuses `order_heldout`): `order_heldout`
(expert 99/100), `ties_heldout` (100/100), `sick_heldout`, `buds_heldout`. B1, B2, B5, B6 and B7 also work on the other plant
tasks (B1 and B2 on every task); A-situations apply to `prune_one`, `prune_turn` and `prune` (A8: `prune` only).

What the expert does differently in them (only when the scenario is on): after a bump it plans again from where the
arm is; with a sticky button it lets go when the plate has not moved for 10 steps of pressing and presses again; when
nothing is (left) to cut in A1/A3/A4/A5 it goes back to the home pose; in A8 it cuts in the order given.

How the new assets look to a policy: at the 224-pixel front view a tie (drawn 1.8 times its modelled size) is about 6
pixels, clearly coloured; a bud and the brown spots or edges of a sick leaf are 2-4 pixels, the wilted (yellow,
drooping) leaves a little more. A1 is a fair test of looking; A3 and A4 ask for detail near the limit of that view.

Not yet built: A2 (the tie facing away; the ties are there, the drawing of a hidden one is not), A6, A7 and every tier 2
and 3 situation. The reasoning tasks (`reason_*`, roadmap Phase 7) have their own hidden-card kind; the A-situations
here are scenario options the reasoning work can reuse, not reasoning tasks of their own.
