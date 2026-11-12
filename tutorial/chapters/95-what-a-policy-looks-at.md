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
show that the policy that scored 85% here looks at the scene once and then follows its own arm.

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
2. **Build the query.** `--stale_images` sends the first frame's images in both image slots
   (`{cam}_t-1` and `{cam}_t0`) every time, while the joint state in `state` stays live: the probe
   removes what the cameras show, not where the arm is. `--first_prev_action expert` replaces, in
   the first query only, the **hold command** the robot is under before it moves
   (`env.last_action`) with the pilot's first command.
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

### What each probe predicts

Before reading a result, write down what each strategy would do. This is reasoning about the two
strategies, not a measurement:

| Probe | Look once | Look every time |
|---|---|---|
| cameras frozen | closes at the named object, steered by the joint state; may miss the last centimetres | images that no longer match the arm may throw it off before it closes anywhere |
| expert drives toward another object, then hands over | continues toward the other object | turns back to the named one |
| object moved during the reach | closes at the old place | closes at the new place, if a plan is left after the move |
| object moved at the grasp | closes once on nothing, or closes again where the object was | opens, goes back, closes at the new place |
| random start pose | depends on how much of its motion it learned from one start | unaffected, if it learned to read the images |

Cameras frozen does not separate the two strategies on its own: it shows whether later frames
carry anything, and whether the first look was enough to choose the object. The handover and the
move are the probes that separate them.

### The disturbance

`Disturbance` in [`disturb.py`](../../sim/pickup/disturb.py) moves the resting named object once:
with `reach` at the first control step with `env.steps >= --disturb_step`, with `grasp` at the
first close command, before it is executed. Of up to 64 random directions it takes the first whose
place `--disturb_mm` away lies where objects are placed, at least 110 mm
(`MIN_OBJECT_GAP`) from every other object and 70 mm (`MIN_GRIPPER_GAP`) from the gripper in the
table plane, and turns the object by up to 0.5 rad. If none qualifies, nothing moves. It draws from
its own random-number generator (`default_rng([seed, 11])`), so the layout of the seed is
unchanged and every policy is offered the same directions in the same order; which one qualifies
depends on where its gripper is at that moment.

A third kind, `--disturb drift`, slides the named object along a smooth random path at
`--drift_cm_s` (default 2.0) from the first step until it is held, turning away from the edge of
the area and from the other objects. The lab notebook has no measurement with it.

`--random_start` calls `random_start` in `expert.py`, which puts the arm at rest at a random gripper
pose: 0.35-0.65 m out, ±0.25 m sideways, 0.22-0.45 m above the table, yaw within ±0.6 rad. It
draws from its own random-number generator, so the layout of the seed is unchanged.

### The reference

`ScriptedExpert(reactive=False)`, which recorded the clean datasets, plans once (above the object,
down, close, lift): the look-once strategy written by hand. With `reactive=True`
(`--expert_reactive`) it checks `_needs_new_plan` every control step: true when the object lies
more than 15 mm from where it was planned for, or when the arm is lifting with the gripper opening
below 0.05, the fingers having met on nothing. It then first tries `_track`, which shifts the
remaining waypoints with the object if it moved at most 40 mm (`TRACK_LIMIT`) and the gripper is
not closing; otherwise it plans again from where the arm is. An 80 mm move is always a new plan.

## In this repository

`eval.sh` passes the arguments after the episode count to `eval.py` and serves the newest
**checkpoint** (the training state saved at one point), with the averaged weights if the run saved
them (EMA, a running average of the weights kept during training); `PICKUP_CHECKPOINT` and
`PICKUP_WEIGHTS` choose others. It also passes `--expect_run` (and `--expect_number`,
`--expect_weights` when you chose them), and `check_server` stops the client if the server on the
port is serving another run's weights file. Give each probe its own `PICKUP_OUT`;
`results.json` records the probe settings under `interventions` (`random_start`, `exec_horizon`
and `flow_steps` at the top level), and per episode `disturbed`, `prefix_goal` and `closes`.
`interventions.disturbed_rate` is the share of episodes in which the object really moved.

From the repository root, one dataset, one policy, then the probes (the other demonstration sets
are built in [Which demonstrations teach](72-which-demonstrations-teach.md)):

```bash
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k
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
# moves at steps 10 and 30: change --disturb_step; by 40 mm: add --disturb_mm 40

# policies with history only
PICKUP_OUT=$out/first_expert    sim/pickup/scripts/eval.sh $run pick_target 100 --first_prev_action expert
PICKUP_OUT=$out/prefix10        sim/pickup/scripts/eval.sh $run pick_target 100 --expert_prefix 10

.venv/bin/python -m sim.pickup.summarize --task pick_target --runs my_run   # one row per directory
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
demonstrations, averaged weights, trained before the position fix ([Positions](12-positions.md)).
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
  expert steps toward another object it still closed at the named object in 74% of episodes and at
  the other object in 16/100; after 25, at the other object in 100/100, cameras live. By then the
  gripper is probably close to that object (exercise 4): following its own arm fits, and so does
  grasping whatever is under it.

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
| at step 20, by 120 mm | 0/99 | 9 mm | 116 mm | 35/99 |
| at the moment the gripper first closes | 1/99 | - | - | 29/99 |

- **It does not follow a moved object.** Moved at step 20 or 30, by 80 or 120 mm: no success, and
  the gripper closed 7-9 mm from the old place. Moved at step 10, the medians fall between the two
  places, which fits closing in between or a mix of episodes at each place; exercise 2 tells which.
  The 40 mm row, 17 episodes, is too small to compare.
- **It does not recover from a missed grasp.** Moved at the grasp: 1/99. It closed again in 29
  episodes, about as often as in the reach rows (31, 28): a second close is not a recovery.
- **It does what an expert with one plan does.** The one-plan expert fails every move and the grasp
  (0/100 at step 10, 0/99 at step 20, 0/100 at step 30 and at the grasp); planning again, it
  succeeds in every episode. Only at step 10 does the policy do clearly better than the one-plan
  expert (9/100 against 0/100).

<!-- interactive: disturbance -->
*Figure: an episode moved at step 20, from above: the gripper's path, the object's jump, and the
gripper closing beside the old place.*

### The same probes on other policies

The probes give other patterns on other policies, and that is what makes them tools rather than one
result. On the three-object policy with history, `--first_prev_action expert` and
`--expert_prefix 10` show that it reads its goal from the previous-action slot, not from the
cameras ([History and shortcuts](52-history-and-shortcuts.md)). On a policy trained on
demonstrations with random start poses and recoveries (episodes in which the object is moved and
the expert that plans again goes after it: `--random_start --disturb_prob 0.5`), the probes give
the look-every-time pattern: with frozen cameras it closes at the named object in 0% of episodes
(counted over all episodes, closed or not), and handed over after 25 expert steps toward another
object it turns back and succeeds in 89/100 ([A policy that corrects
itself](103-case-self-correction.md), where every demonstration set is probed).

## Why the demonstrations allow it

In `pick_target_4k` every reach starts at the home pose (`HOME_Q` in `PickupEnv.reset`, plus
Gaussian noise of 0.03 rad per joint) and runs straight toward the next waypoint
(`ScriptedExpert.act`); the object never moves, and `collect.py` keeps only successes. A few
control steps in, the joint angles say where the line ends (field notes, "Success is not the same
as looking"), so looking once imitates the data as well as looking every time
([Imitation learning](43-imitation-learning.md)). An argument, not a measured cause.

The test is to change the data, and it was run: the same recipe trained on demonstrations with
random start poses, with recoveries recorded by the expert that plans again, and with both.
[Which demonstrations teach](72-which-demonstrations-teach.md) has what each change does to the
success rate and when the policy starts to look; [A policy that corrects
itself](103-case-self-correction.md) has every policy under every probe, against both experts, and
the recipe README.md now recommends. The probes and the commands above are the same in both.

## Where it goes wrong

- **A success rate that includes undisturbed episodes.** `successes` and `summarize.py` count
  every episode. Count over episodes whose `disturbed.to_xy` is set (exercise 2), and read
  `interventions.disturbed_rate` before comparing two runs.
- **A median miss of about `--disturb_mm` under `--disturb reach`.** `GraspProbe` measures to the
  object's new place. Measure from `disturbed.from_xy`.
- **"Closed at the named object" near 0% under frozen cameras.** `summarize` in `diagnostics.py`
  divides `closed_nearest_is_target_rate` by every episode, including those in which the gripper
  never closed. Read `closed_gripper_rate` next to it before saying where the policy went.
- **`--first_prev_action expert` changes nothing.** `Policy.infer` in `policy_server.py` writes the
  previous action into the first `self.past` rows: none without history.
- **`--prefix_target distractor` on `pick_cube` stops with `StopIteration`.** The pilot's goal is
  `next(n for n in env.active if n != named)`, and with one object there is none. The handover probe
  needs a task with at least two objects.
- **`--stale_images` read as "blind".** The joint state is still current. A policy that scores well
  with frozen cameras may be steering by its own arm; one that scores badly may still have chosen
  the right object (the 99% above).
- **A probe that differs from the plain run by a few episodes.** One policy scored 84/100 and
  93/100 on two sets of layouts ([A policy that corrects itself](103-case-self-correction.md)).
  Read `success_ci95`; run a second set with `--seed_start 1000100`.

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

1. Fill in the table under "What each probe predicts" for your own checkpoint before you run the
   probes, then run them. Which of your predictions failed, and which input does that point to?

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

3. Run the expert loop above and compare with the expert's numbers. Under `--disturb grasp`, which
   condition in `_needs_new_plan` makes the expert plan again, and why does `_track` not handle it?

4. `--disturb_mm 40` at step 20 moved the object in 17 of 100 episodes, against 96 for 80 mm. From
   `MIN_GRIPPER_GAP`, how far from the object must the gripper be for a place 40 mm away to be free,
   and what does that say about step 20 and the 25-step handover?

5. Predict, then measure, what a look-once policy does when the target never stands still:

   ```bash
   .venv/bin/python -m sim.pickup.eval --task pick_target --episodes 100 --expert --expert_reactive \
       --disturb drift --out results/pickup/expert/pick_target/expert_plans_again_drift
   PICKUP_OUT=results/pickup/my_run/pick_target/drift \
       sim/pickup/scripts/eval.sh experiments/pickup/my_run pick_target 100 --disturb drift
   ```

   The reactive expert is written to follow a slow drift with `_track`, not new plans; print
   `expert.tracks` and `expert.replans` at the end of each episode in `eval.py` to check. Compare
   `disturbed.path_m` in successful and failed episodes of your policy.
