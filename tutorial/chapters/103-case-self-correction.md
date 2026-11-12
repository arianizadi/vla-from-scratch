---
title: "Case Study: A Policy That Corrects Itself"
part: 10 Case studies
order: 103
summary: Disturb your own policy the way this repository does, read the result against an expert that plans again, and record demonstrations that teach a policy to follow a moved object and to try a missed grasp again.
status: draft
---

# Case Study: A Policy That Corrects Itself

A policy that lifts the named object in 93 of 100 unseen layouts succeeds in none of 97 episodes
when the object is moved 80 mm a second into the reach (control step 20), and in none of 100 when
it is moved as the gripper closes. This chapter follows how it was taught to follow a moved object
and to retry a missed grasp: the reference, the disturbance, the data, the results. The model did
not change.

## Where it started: a policy that looks once

The task, `pick_target`, puts three of four objects on the table and names one in the instruction;
a **layout** (which objects, where, which one is named, the arm's start pose) is fixed by the seed.
The policy is `dp_vit_lang_reactive` (patch tokens, 32 instruction tokens, no history; 48.9M
parameters), trained on 4000 demonstrations for 5M **samples** (training windows drawn by the data
loader). After the position fix (the rotary position table had been computed in bfloat16;
[Positions](12-positions.md)) it ends at 93% with its trained weights (lab notebook, "The same run
before and after the fix"). It chooses its object from the first look and then follows its own arm
([What a policy looks at](95-what-a-policy-looks-at.md)).

## The reference: one plan against planning again

First, whether a disturbance can be recovered from at all. `ScriptedExpert` in
[`sim/pickup/expert.py`](../../sim/pickup/expert.py) reads the true object pose from the simulator
and, in `_plan`, lays out four waypoints for the named object where it is now:

1. above it, 10-14 cm over the grasp point, gripper open;
2. down to the grasp point, open, then a dwell of 2 **control steps** (a control step is one
   command to the arm, 20 per second);
3. close, and wait 8 steps;
4. lift 18-24 cm.

It stores the object's position at planning time as `planned_xy`. With `reactive=False`, the
default, that plan is made once per episode in `reset` and followed to the end: the look-once
strategy, written by hand.

With `reactive=True`, `act` first calls `_needs_new_plan`, which returns true when the named object
is still on the table and either

- it lies more than 15 mm (in the table plane) from `planned_xy`, or
- the expert is on its last waypoint (lifting) and the gripper opening in the state is below 0.05:
  the fingers met on nothing.

It then calls `_track`. A move of up to 40 mm (`TRACK_LIMIT`), while the current waypoint does not
command the gripper closed, only shifts the remaining waypoints with the object (counted in
`self.tracks`). A larger move, a move during the close or the lift, or a grasp that met nothing
calls `_plan(again=True)` (counted in `self.replans`). If the gripper's set-point is more than 2 cm
below the new approach height, `_plan` inserts a **retreat waypoint** first: straight up to the
approach height, gripper open, so the fingers clear the objects before the arm moves sideways.

**A worked example.** The object is moved 80 mm at the moment the gripper first closes. On the next
control step the object is 80 mm from `planned_xy`, beyond the 40 mm `_track` would follow, and the
current waypoint is the close, so the expert plans again. The set-point is 10-14 cm below the new
approach point, so the plan begins with the retreat: the gripper opens, the arm rises at 0.32-0.48
m/s, moves above the new place, descends at 0.14-0.22 m/s, closes and lifts: about 4-9 control
steps up and 9-19 down, from those speeds (the arm lags its set-point). The episode now holds two
close commands: a **second grasp attempt**.

<!-- interactive: arm-replay -->
*Figure: a 3D replay of the expert that plans again, with the object moved at the grasp
(`eval.py --expert --expert_reactive --disturb grasp --traces 5`): the reach, the object jumping as
the fingers close, the retreat straight up, the second approach, the lift.*

The expert through the evaluation loop (lab notebook, "How each policy copes outside the
conditions it was trained in", 100 layouts each; counts are over the episodes in which the object
could be moved):

| Expert | Standard | Random start | Moved at step 10 | Moved at step 20 | Moved at step 30 | Moved at the grasp | Random start, moved at step 20 | Random start, moved at the grasp |
|---|---|---|---|---|---|---|---|---|
| one plan per episode | 100/100 | 100/100 | 0/100 | 0/99 | 0/100 | 0/100 | | |
| plans again | 100/100 | 100/100 | 100/100 | 96/96 | 100/100 | 100/100 | 98/98 | 100/100 |

Every disturbance is recoverable, by planning again. A random start costs a one-plan expert
nothing, so a policy that fails from one fails for another reason than lacking a second plan.

## Moving the object

`Disturbance` in [`sim/pickup/disturb.py`](../../sim/pickup/disturb.py) moves the named object
once per episode (kinds `reach` and `grasp`; a `drift` kind that keeps it sliding until it is held
also exists, unused here); `update(action)` sees every action before it is executed:

- **When.** With `kind="reach"`, at the first control step with `env.steps >= step`. With
  `kind="grasp"`, at the first close command (gripper below 0.5), before it is executed, so the
  fingers close on empty table. Never once the object has left the table.
- **Where.** Of up to 64 random directions, the first whose point `distance` away lies where
  objects are placed (`SPAWN_X`, `SPAWN_Y`), at least 110 mm from every other object
  (`MIN_OBJECT_GAP`) and at least 70 mm from the gripper (`MIN_GRIPPER_GAP`). The object is set
  down there at rest, turned by up to 0.5 rad.
- **What is recorded.** `dict(step, kind, from_xy, to_xy)`; `to_xy` stays `None` when no
  direction qualified, and the episode runs undisturbed. Its random stream (`[seed, 11]`) leaves
  the layout unchanged.

[`sim/pickup/eval.py`](../../sim/pickup/eval.py) exposes it as `--disturb none|reach|grasp|drift`,
`--disturb_step` (default 15) and `--disturb_mm` (default 80). In its loop `probe.update(action)`
runs before `disturbance.update(action)`, so `GraspProbe` (in
[`sim/pickup/diagnostics.py`](../../sim/pickup/diagnostics.py); it records where and when the
gripper first closed) judges that close against the scene it was aimed at. Each episode in
`results.json` carries `disturbed` (the record) and `closes`.

The first time this was run, on the policy before the position fix (lab notebook, "Disturbances"),
the gripper closed a median 8 mm from the object's old place and 78 mm from its new one when it was
moved at step 20; a second close came in 31 of 96 episodes, and none succeeded.

`--random_start` calls `random_start` in `expert.py`, which puts the arm at rest with the gripper
0.35-0.65 m out, ±0.25 m sideways, 0.22-0.45 m up, yaw within ±0.6 rad.

**Counting retries.** `GraspProbe` also counts `closes`, the times the gripper command went from
open to closed; `closes >= 2` is a retry. `summarize` divides retries by all episodes
(`retried_rate`); `export.py` counts them over the moved episodes (`moved.retried`), as the lab
notebook does.

<!-- interactive: disturbance -->
*Figure: an episode from above in which the named object is moved during the reach: the gripper's
path, the object's old and new place, and where the gripper closed, for a policy trained on clean
demonstrations and one trained with recoveries.*

## Demonstrations of recovery

[`sim/pickup/collect.py`](../../sim/pickup/collect.py) has two options for this:

- `--disturb_prob P`: for each episode a generator seeded with `[seed, 7]` decides, with
  probability P, to disturb it; the kind is `reach` or `grasp` with equal chance, the step an integer
  from 5 to 29, the distance uniform between 50 and 100 mm. With P above 0 the expert is built with
  `reactive=True` for every episode, so it reacts whenever `_needs_new_plan` says so.
- `--random_start`: every episode starts from `random_start`, as in the evaluation.

Only successful episodes are kept. `meta/episodes.jsonl` records `disturbed` (the kind, or `none`),
`replans` and `tracks` per episode, `meta/info.json` the total `disturbed_episodes`. With
`--disturb_prob 0.5` about half the episodes draw a move, but a move happens only where a free place
exists (otherwise `disturbed` is `none`): it happened in 1505 of 4000 (`pick_target_4k_recover`)
and in 1456 with random starts as well (`pick_target_4k_rs_recover`).

The four datasets share their seeds, so their objects, places and instructions are the same (lab notebook, "Datasets"):

| Name | Options | Windows | Frames per episode (windows / 4000) |
|---|---|---|---|
| `pick_target_4k` | none | 263,289 | 65.8 |
| `pick_target_4k_rs` | `--random_start` | 241,318 | 60.3 |
| `pick_target_4k_recover` | `--disturb_prob 0.5` | 293,541 | 73.4 |
| `pick_target_4k_rs_recover` | `--random_start --disturb_prob 0.5` | 275,550 | 68.9 |

The last column is arithmetic: a recovery adds frames, a random start removes some. The last
dataset is the one README.md recommends (as `pick_target_4k`); `pick_target_4k_rs_recover_early`
repeats its windows that start in frames 0-7.

## In this repository

From the repository root. The config `sim/pickup/configs/dp_vit_lang_reactive.yaml` switches
averaged weights on; `eval.sh` serves the newest checkpoint with them unless `PICKUP_WEIGHTS=raw`.

```bash
# 1. the datasets (make_dataset.sh replaces a dataset of the same name)
d=sim/pickup/scripts/make_dataset.sh
$d pick_target 4000 pick_target_4k
$d pick_target 4000 pick_target_4k_rs --random_start
$d pick_target 4000 pick_target_4k_recover --disturb_prob 0.5
$d pick_target 4000 pick_target_4k_rs_recover --random_start --disturb_prob 0.5
.venv/bin/python -m sim.pickup.boost_early_windows --shards data/preprocessed/pick_target_4k_rs_recover/shards \
    --out data/preprocessed/pick_target_4k_rs_recover_early/shards --max_frame 7 --repeat 5
.venv/bin/python -c "import json; i = json.load(open('data/lerobot/pick_target_4k_recover/meta/info.json')); print(i['disturbed_episodes'], 'of', i['total_episodes'])"

# 2. one run per dataset, 10 checkpoints each, every checkpoint evaluated
for ds in pick_target_4k pick_target_4k_rs pick_target_4k_recover pick_target_4k_rs_recover \
          pick_target_4k_rs_recover_early; do
  PICKUP_NUM_CHECKPOINTS=10 sim/pickup/scripts/train_resilient.sh dp_vit_lang_reactive $ds $ds 5000000
  sim/pickup/scripts/eval_checkpoints.sh experiments/pickup/$ds pick_target
done

# 3. the battery, on the final checkpoint of one run (repeat for each)
name=pick_target_4k_rs_recover
run=experiments/pickup/$name; out=results/pickup/$name/pick_target; e=sim/pickup/scripts/eval.sh
PICKUP_OUT=$out/standard     $e $run pick_target 100
PICKUP_OUT=$out/second_set   $e $run pick_target 100 --seed_start 1000100
PICKUP_OUT=$out/random_start $e $run pick_target 100 --random_start
for s in 10 20 30; do
  PICKUP_OUT=$out/reach$s    $e $run pick_target 100 --disturb reach --disturb_step $s --traces 5
done
PICKUP_OUT=$out/grasp        $e $run pick_target 100 --disturb grasp --traces 5
PICKUP_OUT=$out/rs_reach20   $e $run pick_target 100 --random_start --disturb reach --disturb_step 20
PICKUP_OUT=$out/rs_grasp     $e $run pick_target 100 --random_start --disturb grasp
PICKUP_OUT=$out/stale        $e $run pick_target 100 --stale_images
PICKUP_OUT=$out/handover25   $e $run pick_target 100 --expert_prefix 25 --prefix_target distractor

# 4. the expert under the same conditions, one plan and planning again (no policy server)
for mode in "" "--expert_reactive"; do
  x=results/pickup/expert/pick_target/expert${mode:+_plans_again}
  ev=(.venv/bin/python -m sim.pickup.eval --task pick_target --episodes 100 --expert $mode)
  "${ev[@]}" --out ${x}_standard
  "${ev[@]}" --random_start --out ${x}_random_start
  for s in 10 20 30; do
    "${ev[@]}" --disturb reach --disturb_step $s --out ${x}_reach$s
  done
  "${ev[@]}" --disturb grasp --traces 5 --out ${x}_grasp
  "${ev[@]}" --random_start --disturb reach --disturb_step 20 --out ${x}_rs_reach20
  "${ev[@]}" --random_start --disturb grasp --out ${x}_rs_grasp
done

# 5. counts over the moved episodes, and the maps of where the gripper closed
.venv/bin/python -m sim.pickup.export --out export/ --runs $name   # probes[].moved in export/runs.json
.venv/bin/python -m sim.pickup.grasp_map $out/standard/results.json $out/stale/results.json --extent_cm 30
```

## What we measured

All on machine B (L40S GPUs, 128 CPU cores), after the position fix, one training seed per run,
100 unseen layouts per cell, 10 flow steps, and 8 actions executed of each **chunk** (the 15 future
actions the policy predicts in one go) before it plans again. Objects are moved 80 mm. "Standard"
is evaluation seeds 1,000,000-1,000,099; "second set of layouts" the next 100
(`--seed_start 1000100`).

### How each policy copes outside its conditions

Final checkpoint of each run, averaged weights where the run saved them; the run on clean
demonstrations did not, so its row is trained weights (lab notebook, "How each policy copes outside
the conditions it was trained in"). Counts are over the episodes in which the object could be moved:

| Demonstrations | Standard | Second set of layouts | Random start | Moved at step 10 | Moved at step 20 | Moved at step 30 | Moved at the grasp | Random start, moved at step 20 | Random start, moved at the grasp |
|---|---|---|---|---|---|---|---|---|---|
| one instruction per layout | 93/100 | 96/100 | 9/100 | 12/100 | 0/97 | 0/100 | 0/100 (8 retried) | 0/99 | 0/99 (2 retried) |
| random starts | 98/100 | 97/100 | 97/100 | 94/100 | 50/98 | 0/100 | 0/100 (2 retried) | 15/99 | 0/100 (1 retried) |
| recoveries | 84/100 | 93/100 | 68/100 | 88/100 | 79/96 | 85/100 | 78/99 (99 retried) | 70/100 | 65/81 (78 retried) |
| random starts and recoveries | 94/100 | 87/100 | 91/100 | 92/100 | 89/98 | 94/100 | 89/100 (100 retried) | 90/98 | 89/100 (99 retried) |
| random starts, recoveries, early windows repeated | 86/100 | 74/100 | 85/100 | 73/99 | 76/97 | 81/100 | 77/100 (99 retried) | 75/97 | 68/100 (99 retried) |

The notebook's three other clean-data rows behave like the first
([Which demonstrations teach](72-which-demonstrations-teach.md)). Row by row:

- **Clean demonstrations.** 9/100 from a random start (where the one-plan expert scores 100/100),
  and no better than the one-plan expert once the object is moved at step 20 or later: 0 at steps
  20 and 30 and at the grasp (12/100 at step 10, against the expert's 0/100). It retried in 8 of
  100 grasp-moved episodes and succeeded in none.
- **Random starts alone.** It works from any start (97/100) and follows a move early in the reach
  (94/100 at step 10), half of them at step 20 (50/98), none at step 30 or at the grasp. A reading,
  not a measured cause: from a varied start the arm's position does not give the goal away early,
  so the policy has to keep looking; by step 30 the expert, closing around step 40 (lab notebook),
  is already over the object, and a policy's arm plausibly says by itself where it is going. Its
  data never went wrong late.
- **Recoveries alone.** It follows a move at every step (88/100, 79/96, 85/100) and tries again
  after a missed grasp in all 99 episodes, 78 of them successfully; 68/100 from a random start.
- **Both.** 89-94% under every disturbance, also from a random start (90/98 and 89/100), a retry
  in 100 of 100 grasp-moved episodes, and 94/100 on the unchanged task.
- **Both, with early windows repeated.** Lower in every column.

### What each ingredient costs

Recoveries alone cost on the unchanged task (89% / 84%, trained / averaged weights, final
checkpoint, against 93% without them and 98% / 98% with random starts alone); random starts on top
take most of it back (94% / 94%). Pooled over each run's last three checkpoints, trained and
averaged weights together, recoveries alone lose on the can (77%), the combined policy on the T
block (77%, against 92% with random starts alone). On the second set the combined policy
scored 87/100. Differences of a few percent between single runs are noise (lab notebook, opening
paragraph); the pattern across the disturbance columns is not.

The combined run started to look earliest: 12% / 54% with a slope of 1.00 after 0.5M samples
([Which demonstrations teach](72-which-demonstrations-teach.md) has the full tables).

### It uses its cameras throughout

The same final checkpoints, probed (lab notebook, "What the same policies depend on"). "At the
named object" is the share of episodes in which the object nearest the gripper when it first closed
was the one the instruction named (chance 33%):

| Demonstrations | Cameras frozen on the first frame | Expert drives 25 steps toward another object |
|---|---|---|
| one instruction per layout | 1/100 (at the named object 94%) | 0/100 (closed at the other object 100/100) |
| random starts | 5/100 (at the named object 98%) | 14/100 (closed at the other object 57/98) |
| recoveries | 0/100 (at the named object 3%) | 56/100 (closed at the other object 28/80) |
| random starts and recoveries | 0/100 (at the named object 0%) | 89/100 (closed at the other object 3/100) |

Shown only the first frame, the clean policy still closes at the named object in 94% of episodes:
the first look was all it used to choose. The combined policy closes there in none, as a policy
that steers by its cameras at every chunk would (0% includes episodes in which the gripper never
closed; the notebook does not split the two). The recovery policy alone behaves the same way (3%);
random starts alone do not (98%). Keeping the cameras in use comes with the recoveries. Handed over
after the expert has driven 25 control steps toward another object, the combined policy turns back
and succeeds in 89 of 100; the clean one closes on the other object every time.

<!-- interactive: grasp-map -->
*Figure: where the gripper first closed relative to the named object, with live cameras and with
the cameras frozen on the first frame (command 5), for the clean and the combined policy: frozen,
the clean policy's closes stay by the named object, the combined policy's do not.*

### After the lift

An open question (field notes, "Open questions"): what a policy does once the task is done.
`PickupEnv.step` ends an episode once the object has been held up for 10 control steps
(`done = success or ...`), and `collect.py` records until `done`: every demonstration ends at the
lift, and teaches nothing about what comes after.

## Where it goes wrong

- **A disturbed success rate diluted by undisturbed episodes.** `successes` in `results.json`
  counts every episode, including those in which no free place was found. Count over episodes whose
  `disturbed["to_xy"]` is set; under a random start with a move at the grasp, only 81 of 100 were
  moved for the recovery policy.
- **A retry is not a recovery.** A second close says nothing about success; read `closes` and
  `retried_rate` next to it.
- **A miss of about `--disturb_mm` under `--disturb reach`.** `GraspProbe` measures to where the
  object is when the gripper closes, its new place. A policy closing at the old place shows a miss
  near 80 mm; measure from `disturbed["from_xy"]`.
- **Replans in undisturbed demonstrations.** With `--disturb_prob` above 0 the expert is reactive
  in every episode, so a grasp that closed on nothing, or an object the fingers push more than
  15 mm, also makes it react: a push of up to 40 mm outside the close shifts the plan (`tracks`),
  a larger one or one during the close starts a new plan (`replans`). Check `replans` and `tracks`
  in `meta/episodes.jsonl` against `disturbed`.
- **Testing only what was trained.** The data moves the object once, 50-100 mm, at control steps
  5-29 or at the grasp. An object that keeps moving, or a look-alike beside the named one, was not
  tested (field notes, "Open questions"). The code can move it continuously
  (`eval.py --disturb drift`, `collect.py --drift_prob`); the lab notebook has no result with it.

## Background and further reading

Background, not measured here:

- Ross, Gordon and Bagnell, *A Reduction of Imitation Learning and Structured Prediction to
  No-Regret Online Learning* (DAgger), 2011
  ([arXiv:1011.0686](https://arxiv.org/abs/1011.0686)). Trains on the states the learner's own
  policy reaches, labelled by the expert, so the data covers the policy's own mistakes.
- Laskey et al., *DART: Noise Injection for Robust Imitation Learning*, 2017
  ([arXiv:1703.09327](https://arxiv.org/abs/1703.09327)). Adds noise to the supervisor's actions
  while demonstrating, so the data shows recoveries, without querying the expert on the learner's
  rollouts. Here the object is moved instead of the arm.

The route to the 93% policy: [Case study: three objects](101-case-pick-target.md); beyond the
simulator: [Toward real tasks](110-toward-real-tasks.md).

## Check yourself

1. Before running command 4, predict the one-plan expert's score from a random start and with the
   object moved at step 20. Which column of the policy table does the first number explain?

2. After command 1, count how often the expert planned again or shifted its plan in the recovery
   dataset, split by the kind of disturbance:

   ```bash
   .venv/bin/python - data/lerobot/pick_target_4k_recover/meta/episodes.jsonl <<'EOF'
   import json, sys
   from collections import Counter
   eps = [json.loads(line) for line in open(sys.argv[1])]
   print(Counter(e["disturbed"] for e in eps))
   for kind in ("none", "reach", "grasp"):
       r = [e for e in eps if e["disturbed"] == kind]
       print(kind, "with a replan:", sum(e["replans"] > 0 for e in r),
             "with a shift:", sum(e.get("tracks", 0) > 0 for e in r), "of", len(r))
   EOF
   ```

3. Under `--disturb grasp`, which of the two conditions in `_needs_new_plan` makes the expert plan
   again, why does `_track` not shift the plan instead, and when would the other condition fire?

4. Run command 3 on the random-starts run with `--disturb_step 15` and `25`, and find where between
   step 10 (94/100) and step 30 (0/100) it stops following the object.
