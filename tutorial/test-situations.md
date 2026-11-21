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

Held-out parts of the world (`name/split`, e.g. `lighting/train@0.3`), for training on some variations and testing on
the rest:

| Scenario | `/train` | `/test` |
|---|---|---|
| `lighting` | dusk, noon, flicker | grow lights (a colour of light never seen) |
| `camera` | shift 1-3 cm, turn 1-3.5 degrees, blur up to 1 px, noise up to 7 | shift 3-4 cm, turn 3.5-5 degrees, blur 1-1.5 px, noise 7-10 |
| `bench` | wood, stained | the greenhouse, with the old table or either bench |
| `distractors` | watering can, gloves, seed packet; no bee | trowel, small pot, and a bee on a leaf |

Held-out instructions (sentence frames no demonstration uses; `collect.py` refuses `order_heldout`): `order_heldout`
(expert 99/100), `ties_heldout` (100/100), `sick_heldout`, `buds_heldout`. B1, B2, B5, B6 and B7 also work on the other plant
tasks (B1 and B2 on every task); A-situations apply to `prune_one`, `prune_turn` and `prune` (A8: `prune` only).

What the expert does differently in them (only when the scenario is on): after a bump it plans again from where the
arm is; with a sticky button it lets go when the plate has not moved for 10 steps of pressing and presses again; when
nothing is (left) to cut in A1/A3/A4/A5 it goes back to the home pose; in A8 it cuts in the order given.

How the new assets look to a policy: at the 224-pixel front view a tie (drawn 1.8 times its modelled size) is about 6
pixels, clearly coloured; a bud and the brown spots or edges of a sick leaf are 2-4 pixels, the wilted (yellow,
drooping) leaves a little more. A1 is a fair test of looking; A3 and A4 ask for detail near the limit of that view.

The leaves: `PickupEnv(leaf_fix=True)` (`collect.py --leaf_fix`, `eval.py --leaf_fix`) draws each leaf blade where its
collision box is. Without it a leaf is drawn as before: centred on the shoot's tip and turned on its side, up to 6 cm
from its box (the pose was written without the offset MuJoCo's compiler gives a mesh). Only what is seen changes: the
collision boxes, and so the expert's actions and every outcome, are the same with and without it. Off by default, so
that the existing policies are evaluated as they were trained.

Paired re-rendering (`python -m sim.pickup.rerender`): the episodes of a dataset made by `collect.py` (without
scenarios) played again and drawn in other looks, one look per copy for the whole episode, with the recorded states and
actions unchanged (checked bit for bit). Each trajectory then appears under several looks with exactly the same
actions. A dataset collected live with `--scenario` gives the expert's trajectories in each look instead, which drift
apart from the plain ones: a policy trained on such data (2000 demonstrations, 73% of them varied) came out insensitive
to the looks but less precise everywhere, and pruned 5/100 against 76/100 without the variation.

Reproducible episodes: `PickupEnv(park_reset=True)` (`collect.py --park_reset`, `eval.py --park_reset`) puts a side shoot
that an episode's plant does not use (and an unused leaf of a shoot in use) back as built at every reset. Without it the
parked shoot keeps the last episode's shape and leaves on the floor; that changes how contacts are found a little, and
an episode's physics then depends on the episodes the environment ran before it (in a dataset, on which seeds a
collection worker ran earlier), drifting apart from a fresh environment's within a few hundred steps. Off by default so
that existing datasets can be reproduced (`rerender.py` re-runs a collection in its own worker chunks for that); new
datasets should use it.

## Full 3D reach: no turntable, heights, pitched shoots, a blocked side

In the base task the turntable brings each shoot round to the front, so the arm repeats one level reach from the front
at a few heights. These situations (`sim/pickup/scenarios_reach.py`, on `prune` only; each changes the task, so one at
a time, and combines with the B situations) take the turntable away and make the arm use every axis. In all of them the
two buttons are gone (not drawn, not solid) and the plate is held on its hinge, and every layout is drawn so that each
shoot can be cut: the arm's planner (below) must find a reachable, collision-free way in to every shoot from its
canonical start, with 3 mm more clearance than the expert later asks of itself; a layout where it cannot is drawn again
from the scenario's own stream (`draws` in the variation says how many it took; after 40 draws heights and elevations
are drawn afresh too, and after 400 the shoots still without a way in are taken off the plant, `dropped`, which one
`pitched` layout in 100 needs). The plant's other draws (flower, leaves, shoot lengths) are prune's for the seed.
`env.flags` holds `reach3d` and `locked`.

| # | Scenario | What changes | Success rule |
|---|---|---|---|
| R1 | `locked` | No spinning: each shoot points to a world direction drawn all round (at least 100 degrees apart, as in prune), heights and elevations as prune draws them. The arm reaches round the main stem: from the sides, diagonally behind, from below or above, the wrist turned for each | prune's |
| R2 | `heights` | As R1, the plant a fifth shorter to a fifth taller (the stem and the flower on it: 19 to 29 cm), the shoots from 3 cm above the stem's base (just above the pot) to 6.5 cm under its top (just under the flower), at least 3.5 cm apart | prune's, and every cut within 10 mm of the shoot's axis in height (`HEIGHT_BAND`); a cut outside ends the episode (`cut_off_band`) |
| R3 | `pitched` | As R1, every shoot pitched 40 to 60 degrees up or down (prune: 15 to 35 up), at heights where its leaves stay clear of the pot and the flower. Jaws that close straight up and down would close along such a shoot, so the wrist pitches with it | prune's |
| R4 | `blocked` | As R1, and a large leaf (a flat green blade, 8 x 4.4 cm, on a short stalk from the main stem) lies along a shoot 24 to 27 mm above or below its axis over the whole cut zone, where the jaws' upper or lower finger would be: the arm comes in from the clear side, the jaws closing level. Above a shoot if the arm still has a way in with it there (from below), else below it (from above), else that shoot has none; at least one leaf a layout | prune's; touching a blocking leaf damages the plant (`leaf touched`) |
| R5 | `height_named` | The plant of R2, and the instruction names one shoot by its height: "cut the shoot just below the flower" (the highest) or "the one nearest the pot" (the lowest), three frames each | that shoot cut, within the band, nothing else cut (as A1) |

What is arm-specific and what is not. The scenarios say only what the plant looks like, what is in the way and what
counts as success. Everything about the robot is in `sim/pickup/arm.py` (`PandaArm`, chosen by `arm_for(env)`): the
gripper's frame and which way its jaws close, how wide they open, the inverse kinematics and its joint limits, which
collision shapes are the arm's, the candidate approaches (the wrist's orientation for a cut of a segment with a given
direction: in from the side, from above or below, leaning along the shoot) and the check of a way in (followed as the
expert's waypoint follower moves its set-point, every pose solved with the expert's own inverse kinematics, the arm's
collision shapes kept a margin from every obstacle, the arm left in the posture it starts from), and how the hand
settles before the jaws close. The scenarios call only `plan_cut` (through `scenarios_reach.plan_cut`, which adds the
plant's obstacles and one task rule: the main stem must not lie between the closing jaws). Another arm is another class
with the same methods.

The expert on these situations (`ScriptedExpert`, `_plan_reach`): of the shoots left, the first, from the one pointing
nearest the robot round to the one pointing farthest away, that the arm has a way in to from where the hand is; back to
the lane in front of the plant without turning, along it turning to the cut's orientation, to a point 13 to 15 cm back
along the approach, in at 10 cm/s, the jaws at the plan's opening (48, 36 or 28 mm), close until the shoot is cut, out
the same way. The existing expert's orientation (level, the jaws upright) is tried first, so where it is clear the cut
looks as it does in prune. With nowhere to go from where the hand is it goes to the canonical start and plans from
there.

Settling before the cut (`ScriptedExpert(settle=True)`, `collect.py --expert_settle`, `eval.py --expert_settle`; off by
default): at the cut, before the jaws close, the set-point is corrected for the arm's sag every control step until the
hand is within 1 mm of the shoot's axis along the closing jaws and slower than 5 mm/s, or 2 s have passed
(`arm.SETTLE_*`).

Expert, 100 evaluation layouts (seeds from 1,000,000), random starts (`python -m sim.pickup.reach_report expert`):

| Scenario | Solved | Damaged | Median steps (solved) | Approaches: level as prune / other level / from below / from above | Cut: across the shoot, median (90th pct) | in height (z): median, mean, sd | laterally: median, sd |
|---|---|---|---|---|---|---|---|
| `prune` (for comparison) | 99 | 0 | 584 | all as prune | 5.8 (9.7) mm | 5.6, -4.8, 1.4 | 0.5, 4.4 |
| `locked` | 100 | 0 | 311 | 29 / 51 / 20 / 0% | 5.2 (6.3) | 5.0, -5.0, 0.8 | 1.4, 1.5 |
| `heights` | 99 | 0 | 311 | 22 / 50 / 29 / 0% | 5.3 (6.4) | 5.1, -5.1, 0.8 | 1.3, 1.4 |
| `pitched` | 95 | 1 | 298 | 0 / 0 / 44 / 56% | 4.2 (6.9) | 3.8, -4.0, 1.8 | 1.7, 1.9 |
| `blocked` | 100 | 0 | 325 | 18 / 39 / 43 / 0% | 5.4 (6.4) | 5.1, -5.1, 0.7 | 1.5, 1.6 |
| `height_named` | 100 | 0 | 94 | 20 / 49 / 30 / 0% | 5.3 (6.3) | 5.1, -5.2, 0.7 | 1.1, 1.4 |
| **with settling** (`--expert_settle`): `prune` | 99 | 0 | 641 | | 0.6 (7.1) | 0.6, -1.2, 2.9 | 0.3, 1.9 |
| `locked` | 100 | 0 | 339 | | 0.6 (0.7) | 0.6, +0.6, 0.1 | 0.2, 0.2 |
| `heights` | 99 | 0 | 339 | | 0.6 (0.7) | 0.6, +0.6, 0.1 | 0.2, 0.2 |
| `pitched` | 97 | 0 | 339 | | 0.5 (0.6) | 0.5, +0.4, 0.2 | 0.2, 0.2 |
| `blocked` | 99 | 0 | 356 | | 0.6 (0.7) | 0.6, +0.6, 0.1 | 0.2, 0.2 |
| `height_named` | 100 | 0 | 104 | | 0.6 (0.7) | 0.6, +0.6, 0.1 | 0.2, 0.2 |

Clear starts (`expert.clear_start`, as architecture v3's evaluations start), seeds from 1,000,000, 100 episodes a row;
height (z) is the offset across the shoot in its vertical plane, + up; laterally, level across it. Every failure is
out of time (a cut that does not close, re-planned until the time runs out) but three: a flower touched in `pitched`
and none elsewhere. From random starts (`expert.random_start`) the existing expert solves 79 of 100 `prune` layouts,
20 of them lost touching the flower on its way from a start among or behind the plant, and the new situations lose
about as many for the same reason (89 to 91 of 100 on `locked`, `heights`, `height_named`); demonstrations from random
starts keep only the solved episodes. The expert's own height error is a bias, not a spread: it cuts 5 mm below the
axis in every situation (the arm sags under its position servos; the spread is 0.7 to 1.8 mm), and settling removes
it.

The reach map (`python -m sim.pickup.reach_report map`): one shoot on the plant of `prune` (3 seeds), turned every 15
degrees round the stem, at 5 heights from 3 to 17.5 cm above the stem's base and 7 elevations (15 to 35 degrees up as in
prune, 45 and 60 up and down), each checked from the arm's canonical start as a draw checks it. At prune's heights and
elevations the arm has a way in to 70% of the directions: 96% of shoots pointing at the robot, 95% pointing to its front
left, 86% to its left, 100% back left, 75% straight away from it, 58% to its front right, 44% to its right and 7% back
right (sectors of 45 degrees, seen from the robot; why the right side is so much harder is not analysed here). Low
shoots are the hardest (14% at 3 cm above the stem's base, 60 to 83% from 7 cm
up), and steep ones (43% of the cells at 45 to 60 degrees up or down). Over all 2016 cells: 53%. The draws keep only
layouts where every shoot has a way in, so these shares are also how much of the plain task's variety each situation can
show.

Not yet built: A2 (the tie facing away; the ties are there, the drawing of a hidden one is not), A6, A7 and every tier 2
and 3 situation. The reasoning tasks (`reason_*`, roadmap Phase 7) have their own hidden-card kind; the A-situations
here are scenario options the reasoning work can reuse, not reasoning tasks of their own.
