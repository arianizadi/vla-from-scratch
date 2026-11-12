---
title: Probing a Policy That Works
part: 9 Evaluation and debugging
order: 95
summary: Find out what a trained policy's actions depend on by freezing its cameras, letting the expert drive its first steps, moving the object and changing the start pose, and read each result against the scripted expert under the same change.
status: draft
---

# Probing a Policy That Works

A success rate on a static benchmark, one in which nothing moves but the arm, says how often a
policy finishes the task, not which of its inputs it used. A **probe** is an evaluation that
changes one thing about the episode; the probes in [`sim/pickup/eval.py`](../../sim/pickup/eval.py)
show that the policy that scored 85% here looks at the scene once, and that what the
demonstrations contain changes how often it looks.

## The idea

[Closed-loop evaluation](90-closed-loop-evaluation.md) counts successes on unseen **layouts**
(objects, their places, the named one and the arm's start pose, all fixed by the seed). On
`pick_target` (three objects, an instruction that names one) `dp_vit_lang_reactive`, which sees one
frame per camera, the joint state and the instruction, lifts the named object in 85 of 100.

Two strategies would score the same:

- **Look every time.** Every **chunk**, the sequence of 15 future actions the policy predicts in
  one go, heads for where the cameras show the named object now.
- **Look once.** The first chunks head for the named object; later ones continue the arm's motion.

On a static layout the two cannot be told apart. Move the named object 80 mm at **control step**
20 (one command to the arm; 20 per second) and they part: the first closes the gripper at the new
place, the second at the old one.

Each probe changes one thing on the same layouts and weights, and asks:

- `--stale_images` (cameras frozen on their first frame): are later frames needed?
- `--expert_prefix K`, `--prefix_target named|distractor` (the **scripted expert**, a hand-written
  controller that reads the true object pose, drives K control steps): instruction or motion?
- `--first_prev_action expert`: with history, is the goal read from the previous action?
- `--disturb reach` (`--disturb_step`, `--disturb_mm`): does it follow a moved object?
- `--disturb grasp`: does it try again?
- `--random_start`: does it depend on the start pose?

Only the expert's drive toward another object and the moved object separate the two strategies: a
frozen first frame still shows every object.

## How it works

### Where a probe acts

`main` in `eval.py` does four things per control step:

1. **Choose who acts.** With `--expert`, `ScriptedExpert` in
   [`expert.py`](../../sim/pickup/expert.py) acts throughout. With `--expert_prefix K` a second
   expert, the pilot, acts while `env.steps < K`, heading for the named object or, with
   `distractor`, the first other object in `env.active`; the instruction stays, and the policy's
   first query comes at step K. Otherwise the policy is queried when its queue is empty.
2. **Build the query.** `--stale_images` sends the first images every time, with the current joint
   state. `--first_prev_action expert` replaces, in the first query only, the **hold command** the
   robot is under before it moves (`env.last_action`) with the pilot's first command.
3. **`probe.update(action)`.** `GraspProbe` in [`diagnostics.py`](../../sim/pickup/diagnostics.py)
   ([Measuring failure](92-measuring-failure.md)) records the scene at the first close command.
4. **`disturbance.update(action)`, then `env.step(action)`**, so the first close is judged on the
   scene it was aimed at.

Executing 8 actions per chunk (`--exec_horizon`), the policy plans every 8 control steps
([Chunks and execution](42-chunks-and-execution.md)); the expert closes around step 40:

```text
control step     0       8       16      24      32      40
policy plans     *       *       *       *       *       *
named object                         ^ moved before step 20 is executed
plans whose images show the new place    *       *       *
```

A policy that looked every time could turn in the plans at 24 and 32.

### The disturbance

`Disturbance` in [`disturb.py`](../../sim/pickup/disturb.py) moves the resting named object once:
with `reach` at the first control step with `env.steps >= --disturb_step`, with `grasp` at the
first close command, before it is executed. Of up to 64 random directions it takes the first whose
place `--disturb_mm` away lies where objects are placed, at least 110 mm
(`MIN_OBJECT_GAP`) from every other object and 70 mm (`MIN_GRIPPER_GAP`) from the gripper in the
table plane, and turns the object by up to 0.5 rad. If none qualifies, nothing moves.

`--random_start` calls `random_start` in `expert.py`, which puts the arm at rest at a random gripper
pose: 0.35-0.65 m out, ±0.25 m sideways, 0.22-0.45 m above the table, yaw within ±0.6 rad. It
draws from its own random-number generator, so the layout of the seed is unchanged.

### The reference

`ScriptedExpert(reactive=False)`, which recorded the clean datasets, plans once (above the object,
down, close, lift): the look-once strategy written by hand. With `reactive=True`
(`--expert_reactive`) it plans again (`_needs_new_plan`) when the object lies more than 15 mm from
where it was planned for, or when the arm is lifting with the gripper opening below 0.05: the
fingers met on nothing.

## In this repository

`eval.sh` passes the arguments after the episode count to `eval.py` and serves the newest
**checkpoint** (the training state saved at one point), with the averaged weights if the run saved
them (EMA, a running average of the weights kept during training); `PICKUP_CHECKPOINT` and
`PICKUP_WEIGHTS` choose others. Give each probe its own `PICKUP_OUT`;
`results.json` records the probe settings under `interventions` (`random_start`, `exec_horizon`
and `flow_steps` at the top level), and per episode `disturbed`, `prefix_goal` and `closes`. From the repository root, the demonstration sets compared below, one
policy, then the probes:

```bash
d=sim/pickup/scripts/make_dataset.sh
$d pick_target 4000 pick_target_4k                                 # one instruction per layout
$d pick_target 4000 pick_target_cf4k --targets_per_layout 3        # every object named
$d pick_target 4000 pick_target_4k_recover --disturb_prob 0.5      # recoveries
$d pick_target 4000 pick_target_4k_rs --random_start               # random starts
$d pick_target 4000 pick_target_4k_rs_recover --random_start --disturb_prob 0.5
for base in pick_target_4k pick_target_4k_rs_recover; do           # early windows repeated
  .venv/bin/python -m sim.pickup.boost_early_windows --shards data/preprocessed/$base/shards \
      --out data/preprocessed/${base}_early/shards --max_frame 7 --repeat 5
done

PICKUP_NUM_CHECKPOINTS=10 sim/pickup/scripts/train.sh dp_vit_lang_reactive pick_target_4k my_run 5000000

run=experiments/pickup/my_run
out=results/pickup/my_run/pick_target    # one subdirectory per probe

PICKUP_OUT=$out/plain           sim/pickup/scripts/eval.sh $run pick_target 100
PICKUP_OUT=$out/stale_images    sim/pickup/scripts/eval.sh $run pick_target 100 --stale_images
PICKUP_OUT=$out/prefix10_other  sim/pickup/scripts/eval.sh $run pick_target 100 --expert_prefix 10 --prefix_target distractor
PICKUP_OUT=$out/prefix25_other  sim/pickup/scripts/eval.sh $run pick_target 100 --expert_prefix 25 --prefix_target distractor
PICKUP_OUT=$out/disturb_reach20 sim/pickup/scripts/eval.sh $run pick_target 100 --disturb reach --disturb_step 20
PICKUP_OUT=$out/disturb_grasp   sim/pickup/scripts/eval.sh $run pick_target 100 --disturb grasp --traces 5
PICKUP_OUT=$out/random_start    sim/pickup/scripts/eval.sh $run pick_target 100 --random_start
PICKUP_OUT=$out/random_start_reach20 sim/pickup/scripts/eval.sh $run pick_target 100 --random_start --disturb reach --disturb_step 20
PICKUP_OUT=$out/random_start_grasp   sim/pickup/scripts/eval.sh $run pick_target 100 --random_start --disturb grasp
# moves at steps 10 and 30: change --disturb_step; by 40 mm: add --disturb_mm 40

# policies with history only
PICKUP_OUT=$out/first_expert    sim/pickup/scripts/eval.sh $run pick_target 100 --first_prev_action expert
PICKUP_OUT=$out/prefix10        sim/pickup/scripts/eval.sh $run pick_target 100 --expert_prefix 10

.venv/bin/python -m sim.pickup.summarize --task pick_target --runs my_run   # one row per directory
.venv/bin/python -m sim.pickup.export --out export/                         # export/runs.json
```

The expert needs no policy server:

```bash
for mode in "" "--expert_reactive"; do
  name=expert${mode:+_plans_again}
  .venv/bin/python -m sim.pickup.eval --task pick_target --episodes 100 --expert $mode \
      --disturb reach --disturb_step 20 --out results/pickup/expert/pick_target/${name}_reach20
  .venv/bin/python -m sim.pickup.eval --task pick_target --episodes 100 --expert $mode \
      --disturb grasp --out results/pickup/expert/pick_target/${name}_grasp
done
```

## What we measured

All numbers are from the [lab notebook](../lab-notebook.md), machine B (L40S GPUs, 128 CPU cores),
100 unseen layouts per row, 8 actions executed per chunk, 10 flow steps (the sampler's steps from
noise to a chunk, [Flow matching](41-flow-matching.md)).

### The 85% policy

`dp_vit_lang_reactive` after 5.0M **samples** (a sample is one training **window**, the
observations at one frame plus the 15 actions from that frame on, drawn by the data loader) on 4000
demonstrations, averaged weights, before the position fix (the rotary position table, computed in
bfloat16, merged many positions; [Positions](12-positions.md)), like every run in this section
except the hold-frame history run, which was trained after the fix.
**Closed at the named object**: the object nearest the gripper when it first closed was the named
one (chance 33%). **Median miss to named**: the median distance in the table plane between the
gripper and the named object when the gripper first closed (`close_xy_err_target` in `GraspProbe`).

| Condition | Success | Closed at the named object | Median miss to named |
|---|---|---|---|
| as trained | 85/100 | 100% | 7 mm |
| the first frame's images for the whole episode | 1/100 | 99% | 17 mm |
| the expert drives 10 control steps toward another object | 12/100 | 74% | 64 mm |
| the expert drives 25 control steps toward another object | 0/100 | 0% | 209 mm |

- **Later frames are not needed to choose the object.** With frozen cameras it still closed at the
  named object in 99% of episodes. Success fell to 1/100 although the median miss grew only to
  17 mm, between the 15 mm and 20 mm offsets at which the expert still succeeds in 91% and 72% of
  episodes on machine B (lab notebook, "Grasp tolerance"). Later frames carry more than the last
  centimetres.
- **Well into the reach, where the arm is outweighs the instruction and the cameras.** After 10
  expert steps toward another object it still closed at the named object in 74% of episodes; after
  25, at the other object in 100/100, cameras live. By then the gripper is probably close to that
  object (exercise 4): following its own arm fits, and so does grasping whatever is under it.

<!-- interactive: grasp-map -->
*Figure: where the gripper first closed relative to the named object, a dot per episode: as
trained, cameras frozen, after 25 expert steps toward another object.*

Same weights ("Disturbances" in the notebook), the object moved 80 mm unless stated; counts are
over the episodes in which it could be moved:

| The object is moved | Success | Gripper closed, median distance from the old place | from the new place | Second grasp attempt |
|---|---|---|---|---|
| at step 10 of the reach | 9/100 | 39 mm | 49 mm | 43/100 |
| at step 20 | 0/96 | 8 mm | 78 mm | 31/96 |
| at step 30 | 0/99 | 7 mm | 81 mm | 28/99 |
| at step 20, by 40 mm | 1/17 | 18 mm | 35 mm | 9/17 |
| at the moment the gripper first closes | 1/99 | - | - | 29/99 |

- **It does not follow a moved object.** Moved at step 20 or 30: no success, and the gripper closed
  7 or 8 mm from the old place. Moved at step 10, the medians fall between the two places, which
  fits closing in between or a mix of episodes at each place; exercise 2 tells which. The 40 mm
  row, 17 episodes, is too small to compare.
- **It does not recover from a missed grasp.** Moved at the grasp: 1/99. It closed again in 29
  episodes, about as often as in the reach rows (31, 28): a second close is not a recovery.
- **It does what an expert with one plan does.** `ScriptedExpert(reactive=False)` fails every
  move and the grasp (table below); planning again, it succeeds in every episode. Only at step 10
  does the policy do clearly better than the one-plan expert (9/100 against 0/100).

<!-- interactive: disturbance -->
*Figure: an episode moved at step 20, from above: the gripper's path, the object's jump, and the
gripper closing beside the old place.*

### The history policy

`dp_vit_lang` adds history (previous frame, state and action). On the same demonstrations, trained
before the position fix, which hit it harder (16 action-step tokens on 4 positions against 15 on 8),
it was stopped at 2% / 2% (trained / averaged weights) after 3.0M samples. At 2.5M, averaged weights:

| Condition | Success | Closed at the named object | Median miss to named |
|---|---|---|---|
| as served: the first previous action is the hold command | 1/100 | 30% | 186 mm |
| it is the expert's first command (`--first_prev_action expert`) | 22/100 | 97% | 25 mm |
| the expert drives 10 control steps (`--expert_prefix 10`) | 77/100 | 100% | 12 mm |

One expert command in the previous-action slot takes it from chance to the named object in 97% of
episodes: it reads its goal there. In training, windows at a demonstration's first frame fill the
empty slot with the expert's first command, which points at the target; served, it holds the hold
command ([History and shortcuts](52-history-and-shortcuts.md)). A history
policy trained after the fix on demonstrations that begin with a frame of the robot at rest
(`collect.py --hold_steps 1`) ends at 0% / 4% after 5M samples, and at no checkpoint does where it
closes depend on the scene: closing the leak at the first frame did not make it look. A history run
on the plain demonstrations has not been repeated after the fix (field notes, "The previous action
as an input").

## Why the demonstrations allow it

In `pick_target_4k` every reach starts at the home pose (`HOME_Q` in `PickupEnv.reset`, plus
Gaussian noise of 0.03 rad per joint) and runs straight toward the next waypoint
(`ScriptedExpert.act`); the object never moves, and `collect.py` keeps only successes. A few
control steps in, the joint angles say where the line ends (field notes, "Success is not the same
as looking"), so looking once imitates the data as well as looking every time
([Imitation learning](43-imitation-learning.md)). An argument, not a measured cause; the test is
to change the data.

## What changes when the demonstrations change

After the position fix the same recipe (4000 demonstrations, 5M samples) was trained on the seven
sets built above (field notes, "What the demonstrations contain"): **one instruction per layout**,
as before; **early windows repeated**, the windows that start in frames 0-7 present 6 times;
**every object named**; **recoveries** (in 1505 episodes the object is moved once and the expert
plans again); **random starts**; both of the last two; and both with early windows repeated.

Final checkpoints, averaged weights where saved, moves of 80 mm; "standard" on two sets of
layouts, "retried" the episodes with a second grasp attempt:

| Demonstrations | Standard | Random start | Moved at step 10 | 20 | 30 | at the grasp | Random start, moved at step 20 | at the grasp |
|---|---|---|---|---|---|---|---|---|
| before the fix, plain | 85/100, 83/100 | 6/100 | 9/100 | 0/96 | 0/99 | 1/99 (29 retried) | 0/98 | 0/98 |
| one instruction per layout | 93/100, 96/100 | 9/100 | 12/100 | 0/97 | 0/100 | 0/100 (8 retried) | 0/99 | 0/99 |
| early windows repeated | 96/100, 100/100 | 9/100 | 10/100 | 0/100 | 0/100 | 0/100 (3 retried) | 0/100 | 0/100 |
| every object named | 89/100, 93/100 | 10/100 | 18/100 | 0/96 | 0/100 | 0/100 (29 retried) | 0/99 | 0/99 |
| recoveries | 84/100, 93/100 | 68/100 | 88/100 | 79/96 | 85/100 | 78/99 (99 retried) | 70/100 | 65/81 |
| random starts | 98/100, 97/100 | 97/100 | 94/100 | 50/98 | 0/100 | 0/100 (2 retried) | 15/99 | 0/100 |
| random starts and recoveries | 94/100, 87/100 | 91/100 | 92/100 | 89/98 | 94/100 | 89/100 (100 retried) | 90/98 | 89/100 |
| the same, early windows repeated | 86/100, 74/100 | 85/100 | 73/99 | 76/97 | 81/100 | 77/100 (99 retried) | 75/97 | 68/100 |
| expert, one plan | 100/100 | 100/100 | 0/100 | 0/99 | 0/100 | 0/100 | - | - |
| expert, plans again | 100/100 | 100/100 | 100/100 | 96/96 | 100/100 | 100/100 | 98/98 | 100/100 |

The same probes on these runs ("closed there": over the episodes in which the gripper closed):

| Demonstrations | Cameras frozen (at the named object) | Expert 10 steps toward another object (closed there) | Expert 25 steps (closed there) |
|---|---|---|---|
| one instruction per layout | 1/100 (94%) | 18/100 (24/100) | 0/100 (100/100) |
| early windows repeated | 0/100 (70%) | 16/100 (28/100) | 0/100 (100/100) |
| every object named | 4/100 (86%) | 20/100 (22/100) | 0/100 (100/100) |
| recoveries | 0/100 (3%) | 79/100 (1/97) | 56/100 (28/80) |
| random starts | 5/100 (98%) | 98/100 (0/100) | 14/100 (57/98) |
| random starts and recoveries | 0/100 (0%) | 89/100 (0/99) | 89/100 (3/100) |
| the same, early windows repeated | 0/100 (1%) | 78/100 (0/100) | 72/100 (15/100) |

What the tables show (field notes, "Self-correction has to be in the data"). Each row is one run
with one seed: read the pattern across rows, not single differences.

- **Clean demonstrations: a higher score, the same strategy.** Every policy trained on clean
  demonstrations from the home pose scores 6-10/100 from a random start, follows no move at step 20
  or later, and after 25 steps toward another object closes there every time.
- **Random start poses make it follow an early move.** It follows a move at step 10 (94/100), half
  at step 20, none later, and turns back after 10 steps toward the wrong object (98/100).
- **Recoveries make it correct itself.** It follows a move at every step, tries again in all 99
  episodes moved at the grasp, and turns back after 25 steps toward the wrong object in 56/100;
  with frozen cameras it closes at the named object in 3%. On the unchanged task it pays: 84/100
  and 93/100 against 93/100 and 96/100 for clean demonstrations (89% / 84% trained / averaged at
  the end).
- **Both together give both** and take most of that cost back: 94/100 and 87/100 unchanged, 91/100
  from a random start, 89 or more under every move, a second attempt after a missed grasp in 100
  of 100.
- **Repeating the early windows on top of both lowered every score in the first table**: 87% / 86%
  at the end against 94% / 94%, 73-81 with the object moved during the reach.

**What to use**: random start poses and recoveries, and an evaluation with the same disturbances.
README.md's recipe is `make_dataset.sh pick_target 4000 pick_target_4k --random_start --disturb_prob 0.5`
(the plain set's name, reused).

## Where it goes wrong

- **A success rate that includes undisturbed episodes.** `successes` and `summarize.py` count
  every episode. Count over episodes whose `disturbed.to_xy` is set (exercise 2).
- **A median miss of about `--disturb_mm` under `--disturb reach`.** `GraspProbe` measures to the
  object's new place. Measure from `disturbed.from_xy`.
- **`--first_prev_action expert` changes nothing.** `Policy.infer` in `policy_server.py` writes the
  previous action into the first `self.past` rows: none without history.
- **A probe that differs from the plain run by a few episodes.** One policy scored 84/100 and
  93/100 on two sets of layouts. Read `success_ci95`; run a second set with `--seed_start 1000100`.

## Background and further reading

Background, not measured here:

- de Haan, *Causal Confusion in Imitation Learning*, 2019
  ([arXiv:1905.11979](https://arxiv.org/abs/1905.11979)). An imitating policy can take a correlate
  of the expert's action for its cause.
- Codevilla, *Exploring the Limitations of Behavior Cloning for Autonomous Driving*, 2019
  ([arXiv:1904.08980](https://arxiv.org/abs/1904.08980)). The inertia problem: the cloned driver
  links low speed to not accelerating.
- Zhou, *LIBERO-PRO*, 2025 ([arXiv:2510.03827](https://arxiv.org/abs/2510.03827)), and Fei,
  *LIBERO-Plus*, 2025 ([arXiv:2510.13626](https://arxiv.org/abs/2510.13626)). Models above 90% on
  LIBERO collapse when objects, start states or cameras change.

## Check yourself

1. Predict what a look-every-time policy would score in each row of the 85% policy's two tables,
   then probe your own checkpoint.

2. Print the "Disturbances" columns from a results file, and how the first closes spread between
   the two places:

   ```bash
   .venv/bin/python - results/pickup/my_run/pick_target/disturb_reach20/results.json <<'EOF'
   import json, math, statistics, sys
   from collections import Counter

   r = json.load(open(sys.argv[1]))
   moved = [e for e in r["per_episode"] if e["disturbed"] and e["disturbed"]["to_xy"]]
   print(f"moved in {len(moved)}/{r['episodes']} episodes, success {sum(e['success'] for e in moved)}/{len(moved)}")
   closed = [e for e in moved if e["close_tcp_xy"]]
   for key, place in (("from_xy", "old"), ("to_xy", "new")):
       mm = [1000 * math.dist(e["close_tcp_xy"], e["disturbed"][key]) for e in closed]
       print(f"first close, median distance from the {place} place: {statistics.median(mm):.0f} mm")
   print(f"closed a second time: {sum(e['closes'] >= 2 for e in moved)}/{len(moved)}")
   old = [1000 * math.dist(e["close_tcp_xy"], e["disturbed"]["from_xy"]) for e in closed]
   for lo, n in sorted(Counter(20 * int(d // 20) for d in old).items()):
       print(f"{lo:3d}-{lo + 20:3d} mm from the old place: {'#' * n}")
   EOF
   ```

   Run it on a move at step 10: do the first closes cluster in between, or at the two places?

3. Run the expert loop above and compare with the expert rows. Under `--disturb grasp`, which
   condition in `_needs_new_plan` makes the expert plan again?

4. `--disturb_mm 40` at step 20 moved the object in 17 of 100 episodes, against 96 for 80 mm. From
   `MIN_GRIPPER_GAP`, how far from the object must the gripper be for a place 40 mm away to be free,
   and what does that say about step 20 and the 25-step handover?
