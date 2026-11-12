---
title: Toward Real Tasks
part: 11 Toward real tasks
order: 110
summary: Split a real manipulation task into properties, say for each one what this repository has measured and what it has not, and run and read the first experiments that add those properties to the simulator.
status: draft
---

# Toward Real Tasks

Every robot result in this repository is one simulated arm picking up one of four rigid objects
from a bare table. The task it is heading for is a robot that trims agricultural flowers. Taken
apart property by property, that task can be set against what the pick-up task measured and what it
did not.

## The idea

The worked example is **flower trimming**. A robot has to find one particular flower among others
that look much like it, among leaves that move and hide it, and cut its stem in the right place,
correcting itself as things move. Written as properties:

1. **A target that keeps moving**: the stem sways while the cutter approaches.
2. **Look-alikes**: flowers of one plant, one shape, a shade apart.
3. **Clutter and occlusion**: leaves hide the flower from one camera or the other.
4. **A cut that succeeds only in the right place** along the right stem.
5. **A task that goes on after contact**: let go, back off, next flower.
6. **Real cameras**: light and colour no renderer reproduces exactly.

`pick_target` (in [`sim/pickup/env.py`](../../sim/pickup/env.py)) has almost none of these. Its
**layout**, fixed by a seed, puts 3 of 4 objects (red cube, green can, blue hex nut, yellow T block)
on the table at least 110 mm apart, names one in the instruction ("pick up the green can"), and
starts the arm at a pose. Success is the named object held 8 cm above its rest height for 10
**control steps** (commands to the arm, 20 per second).

| Property | In `pick_target` | Nearest measurement here | Status |
|---|---|---|---|
| Moving target | the object rests; in evaluation it can jump once, 80 mm | the jump, at four moments ([Self-correction](103-case-self-correction.md)) | one jump measured; continuous motion not |
| Look-alikes | four objects that differ in colour and in shape | closed at the named object 98-100%, slope near 1 | not measured |
| Clutter, occlusion | three objects, 110 mm apart, nothing in front of them | cameras frozen on the first frame | not measured |
| Where the cut closes | a top-down grasp, any height that holds | grasp tolerance, sideways only ([Measuring failure](92-measuring-failure.md)) | sideways measured; height band not |
| After contact | the episode ends at the lift | none | not measured |
| Real cameras | rendered 224x224 images, fixed lights | none | not measured |

**One worked example.** Take the first property. The only motion measured here is a jump: once per
episode the named object is moved to a place 80 mm away (40 and 120 mm were tried once each, on the
plain policy before the position fix described below). The best policy so far was trained on
demonstrations that start from random gripper poses and in which, in 1456 of 4000 episodes, the
object is moved once and the scripted expert plans again or retries the grasp (**recoveries**;
`collect.py --random_start --disturb_prob 0.5`). It lifts the object in 89 of 98 episodes when the
jump comes at control step 20. A swaying stem is different. At 2 cm/s, the drift's default speed,
it has travelled up to 4 cm on a curving path by the time the expert closes the gripper (around
control step 40, after 2 s). The policy predicts a **chunk** of 15 actions and executes 8 before it
plans again, every 0.4 s; the stem moves 8 mm in between. That is arithmetic, not a measurement:
whether a policy that follows one 80 mm jump also tracks steady motion is what the first experiment
asks.

## What was measured that bears on each property

Unless stated otherwise, numbers are from the [lab notebook](../lab-notebook.md), machine B (L40S
GPUs, 128 CPU cores), 100 unseen layouts per evaluation, one training seed per run. The policy
throughout is `dp_vit_lang_reactive`: patch tokens of a small vision transformer trained here, the
instruction as tokens, no history, 48.9M parameters, trained on 4000 demonstrations for 5M
**samples** (a sample is one training **window** drawn by the data loader: the observations at one
frame plus the actions that follow), after the position fix (the rotary position table had been
computed in bfloat16 and held only 641 distinct positions of 2048; [Positions](12-positions.md)).

### A target that keeps moving

The disturbance (`Disturbance` in [`sim/pickup/disturb.py`](../../sim/pickup/disturb.py)) moves the
resting named object once, 80 mm in a random direction, and turns it by up to 0.5 rad. In the
notebook table "How each policy copes outside the conditions it was trained in", the scripted expert
with one plan per episode fails every jump and the one that plans again succeeds in all: following
a moved object takes planning again, in the expert and in the data. Policies trained on clean
demonstrations succeed in 0-18% of jumps: they decide where to go from their first look and follow
their own arm ([Probing a policy](95-what-a-policy-looks-at.md)). Recoveries alone give 79-88%, with
a second attempt in 99 of 99 after a move at the grasp; random start poses and recoveries together,
89-94% at every moment ([Self-correction](103-case-self-correction.md) has the table).

<!-- interactive: disturbance -->
*Figure: a top-down replay of an episode in which the named object is moved during the reach: its
old place, its new place, the gripper's path and where it closed. A drift, not yet run, would
replace the one jump with a path the gripper has to follow.*

What this does not show: a target that never stands still. After the jump the object rests; a
drifting stem is somewhere new in every frame.

### Look-alike objects

After 3.0M samples, each of the seven no-history policies in the notebook table "What the
demonstrations contain" closes at the named object (the object nearest the gripper when it first
closed was the one the instruction named) in 98-100% of episodes, against a chance of 33%. Their
**slope**, the least-squares slope of where the gripper closed against where the named object was,
is 0.95-1.01: near 1 the gripper goes where the object
is, near 0 it closes in the same place whatever the scene ([Measuring failure](92-measuring-failure.md)).
A policy could still tell the green can from the red cube by shape; nothing here separates two
objects of one shape.

### Clutter and occlusion

Not measured. Nothing stands between the two cameras (`front`, fixed; `wrist`, on the hand) and
the objects but the arm itself. The nearest probe freezes the cameras on their first frame
(`--stale_images`): the best policy then closes at the named object in 0% of episodes (notebook
table "What the same policies depend on"). It uses the later frames throughout, so a leaf hiding the
flower in them would reach its decisions; whether that hurts is for the clutter experiment to
measure.

### Where the cut closes

Cutting succeeds only inside a band of height along one stem. The nearest measurement is the grasp
tolerance: the scripted expert with a fixed sideways offset added to its grasp point
([`sim/pickup/grasp_tolerance.py`](../../sim/pickup/grasp_tolerance.py)), 320 episodes per offset,
machine B:

| Offset | Cube | T block | Can | Hex nut | All |
|---|---|---|---|---|---|
| 0 mm | 100% | 100% | 100% | 100% | 100% |
| 10 mm | 99% | 100% | 94% | 97% | 98% |
| 15 mm | 95% | 100% | 76% | 93% | 91% |
| 20 mm | 93% | 98% | 29% | 70% | 72% |
| 30 mm | 32% | 76% | 3% | 15% | 32% |

<!-- interactive: tolerance -->
*Figure: grasp success of the offset expert against the sideways offset, one curve per object.
Cutting would need the same curve along the stem's height.*

The offset is sideways only. Height is recorded (`close_z_above_grasp` in `GraspProbe`,
[`sim/pickup/diagnostics.py`](../../sim/pickup/diagnostics.py)), but a band of height has never been
the success test.

The tolerance also differs between the two machines: at 10 mm the can was grasped in 60% of episodes
on machine A and 94% on machine B ("Grasp tolerance"). The MuJoCo build is the one known difference;
it has not been isolated as the cause.

### A task that continues after contact

Not measured. `PickupEnv.step` ends the episode at success, so a demonstration ends at the lift
(about 66 control steps) and no training window shows what follows.

### Real cameras and the gap from simulation

Not measured. Everything here is rendered by MuJoCo at 224x224, from a fixed front camera and a
camera on the hand, under fixed lights (`build_model` in `env.py`). The one gap measured between
two setups is the machine difference above.

## In this repository: the experiments now starting

Phase 4 of the [roadmap](../roadmap.md), "toward flower trimming", adds the properties one at a time,
asking of each: can the scripted expert do it, does the best policy so far cope without new data, and
what must the demonstrations contain for a policy to learn it. Each step reports the probes of
[Probing a policy](95-what-a-policy-looks-at.md), not only a success rate.

| Step | Code today | Results |
|---|---|---|
| A target that keeps moving | `eval.py --disturb drift --drift_cm_s`; `collect.py --drift_prob --drift_cm_s`; the expert follows it | none yet |
| Look-alikes | task `pick_lookalike` in `env.py` | none yet |
| Clutter and occlusion (`pick_clutter`) | not written | none |
| Cutting: a stem, success inside a height band | not written | none |
| A pretrained vision encoder, a higher wrist resolution | not started | none |

**The drift.** With `--disturb drift`, `Disturbance._drift` moves the named object from the first
control step on at `--drift_cm_s` (default 2.0), along a heading that turns at random and away from
the edge of the placement area and the other objects, and stops while the gripper is commanded
closed. In `collect.py`, `--drift_prob` is the share of the episodes *not
drawn for a jump* that drift, each at a speed between half and one and a half times `--drift_cm_s`;
`--disturb_prob` or `--drift_prob` above zero makes the expert reactive.
`ScriptedExpert(reactive=True)` ([`expert.py`](../../sim/pickup/expert.py)) acts once the object
lies more than 15 mm from where it planned for: up to `TRACK_LIMIT` (40 mm) it shifts its remaining waypoints (`_track`, not while
closing), beyond that it plans again.

**The look-alikes.** `pick_lookalike` draws 3 of the red cube, an orange cube, a pink cube and a
dark red cube: one shape, colours a step apart. Only the colour word of the instruction tells them
apart.

From the repository root, the expert first, one plan and planning again, against speed:

```bash
for v in 1 2 4 8; do
  for mode in "" "--expert_reactive"; do
    .venv/bin/python -m sim.pickup.eval --task pick_target --episodes 100 --expert $mode \
        --disturb drift --drift_cm_s $v --out results/pickup/expert/pick_target/drift${v}${mode:+_plans_again}
  done
done
.venv/bin/python -m sim.pickup.eval --task pick_lookalike --episodes 100 --expert \
    --out results/pickup/expert/pick_lookalike/plain
```

Then the best policy so far, unchanged (`my_run` trained with README.md's recipe), and the probes:

```bash
run=experiments/pickup/my_run
out=results/pickup/my_run
for v in 1 2 4 8; do
  PICKUP_OUT=$out/pick_target/drift$v sim/pickup/scripts/eval.sh $run pick_target 100 --disturb drift --drift_cm_s $v
done
PICKUP_OUT=$out/pick_target/drift2_stale sim/pickup/scripts/eval.sh $run pick_target 100 \
    --disturb drift --drift_cm_s 2 --stale_images
PICKUP_OUT=$out/pick_target/drift2_random_start sim/pickup/scripts/eval.sh $run pick_target 100 \
    --disturb drift --drift_cm_s 2 --random_start
PICKUP_OUT=$out/pick_lookalike/plain sim/pickup/scripts/eval.sh $run pick_lookalike 100
PICKUP_OUT=$out/pick_lookalike/prefix10_other sim/pickup/scripts/eval.sh $run pick_lookalike 100 \
    --expert_prefix 10 --prefix_target distractor
```

Then demonstrations that contain the new property, on top of the ingredients that already work:

```bash
d=sim/pickup/scripts/make_dataset.sh
$d pick_target 4000 pick_target_4k_rs_recover_drift --random_start --disturb_prob 0.5 --drift_prob 0.5
$d pick_lookalike 4000 pick_lookalike_4k_rs_recover --random_start --disturb_prob 0.5
PICKUP_NUM_CHECKPOINTS=10 sim/pickup/scripts/train_resilient.sh dp_vit_lang_reactive \
    pick_target_4k_rs_recover_drift drift_run 5000000
sim/pickup/scripts/eval_checkpoints.sh experiments/pickup/drift_run pick_target
```

## How to read the results when they come

**Against the expert, at each speed.** A policy's drift score means something only next to the
expert's at the same speed. If the expert that plans again fails at a speed, this expert and its
tracking rule (`_track`, `TRACK_LIMIT`) cannot do the task there, and there are no demonstrations to
learn from: change the expert before you train a policy at that speed.

**With the probes of chapter 95.**

- With the drift alone: compare where the gripper first closed (`close_tcp_xy`) with
  `disturbed.from_xy` and with the object's place at the close (`close_object_xy`). A policy that
  closes near the first place looked once and followed its arm.
- `--stale_images` with the drift: does the policy depend on frames after the first? One that looks
  all the way loses the object (the best policy: 0% at the named object, above); one that still
  succeeds never needed the later frames.
- `--random_start`: whether the arm's start pose still gives the goal away.
- `--expert_prefix 10 --prefix_target distractor` on `pick_lookalike`: the expert heads for a
  look-alike, then hands over. A policy that turns back reads the colour word; one that closes on the
  look-alike follows its arm.

**For look-alikes, the confusion, not only the rate.** Success counts the named object; the
interesting failure is closing precisely on its neighbour in colour. `close_nearest` in each episode
says which object was nearest at the first close:

```bash
.venv/bin/python - results/pickup/my_run/pick_lookalike/plain/results.json <<'EOF'
import json, sys
from collections import Counter

r = json.load(open(sys.argv[1]))
pairs = Counter((e["target"], e["close_nearest"]) for e in r["per_episode"] if e["close_nearest"])
for (named, nearest), n in sorted(pairs.items()):
    print(f"named {named:12s} closed nearest {nearest:12s} {n}")
EOF
```

Read it with the slope: a small miss to the nearest
object with a large miss to the named one is an accurate grasp of the wrong object, a different
failure from not looking.

**Which data to start from.** Random start poses and recoveries: 94% / 94% on the unchanged task
and 89-94% under every jump tried ([Which demonstrations teach](72-which-demonstrations-teach.md)).
Adding repeated early windows ended lower in the one run made (87% / 86%; one seed each), probably
because every window of that data already needs the scene and the copies take training from the
later windows, where the corrections are (field notes, "What the demonstrations contain"). Leave
them out unless a second seed says otherwise.

When the policy starts to look depends on the data: after 1.0M samples the plain recipe was at 8%,
random start poses and recoveries at 68% / 86% (closed at the named object 100%, slope 1.02;
[How much data and training](71-how-much-data-and-training.md)). A new property may move that point,
so evaluate every checkpoint (`eval_checkpoints.sh`) before you call it unlearnable.

## Reasoning in words (planned, not started)

Nothing in the roadmap's Phase 7 has been run. Here the instruction has one
job: to say which object. In trimming, which flower to cut is a judgement (open enough, long enough
stem, not already cut) that no instruction can name in advance.

Two planned forms. In the first, the scripted expert, which knows its phase and where every object
is, writes a line of text per frame, and a policy whose backbone is a language model learns to write
such a line before each chunk: the from-scratch or the Qwen VLA of
[Pretrained backbones](102-case-pretrained-backbones.md) (86% / 87% and 70% / 72% without text), not
`dp_vit_lang_reactive`, which cannot generate text. The action head reads the line with the rest of
the sequence, which `conditioning: all_tokens` already hands it
([What the action head sees](50-what-the-action-head-sees.md)). In the second, a vision-language
model names the next sub-task at a low rate and the reactive policy carries it out: choosing the
flower is the reasoning, following the stem is the control.

The test that matters: swap in the text of another object or phase and measure
whether the action follows it, as `--prefix_target distractor` does for motion. A text the action
ignores is a narration.

## Where it goes wrong

- **A drift score with the stem boxed in.** `_drift` stands the object still in a control step where
  every turn would leave the area or come within 110 mm of another object. Check `disturbed.path_m`
  against `speed` times the time to the first close before reading an episode as a tracked target.
- **A dataset read as all drift.** With `--drift_prob 0.5` next to `--disturb_prob 0.5`, half the
  episodes are drawn for a jump and half of the rest drift: about a quarter. One drawn for a jump
  that finds no free place is neither moved nor drifted (`pick_target_4k_rs_recover`: 1456 of 4000
  moved). `info.json` has `disturbed_episodes` (jumps and drifts together) and `drifting_episodes`; the
  jumps are the difference. In evaluation, count
  episodes whose `disturbed.to_xy` is set.
- **"red cube" inside "dark red cube".** In `pick_lookalike` the instruction "pick up the red cube"
  is contained in "pick up the dark red cube". Red, orange and pink differ by one swapped word; "dark
  red" differs from orange and pink by two words in place of one. A policy that confuses exactly
  red and dark red may be reading the end of the phrase; look at that pair in the confusion table
  first.
- **A pick-up policy scored on look-alikes it never saw.** Orange, pink and dark red are not in
  `pick_target`'s instructions or images. A low score there does not say look-alikes cannot be
  learned.
- **A tolerance carried from one setup to another.** Measure tolerance, and any cutting band, on
  the machine you evaluate on.

The limits of everything above: one simulated arm (a Franka Panda in MuJoCo), seven rigid objects in
all (`pick_target`'s four; in `pick_lookalike` four cubes of one shape, the red cube shared), three
per layout, top-down grasps, rendered images, one training seed per run, 100 layouts per evaluation.

## Background and further reading

Background, not measured here:

- Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the
  Real World*, 2017 ([arXiv:1703.06907](https://arxiv.org/abs/1703.06907)). Random textures in
  simulation; a real-world detector robust to distractors and partial occlusion.
- Peng et al., *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization*, 2018
  ([arXiv:1710.06537](https://arxiv.org/abs/1710.06537)). Randomised dynamics in training; a pushing
  policy that works on a real arm without training on it.
- Laskey et al., *DART: Noise Injection for Robust Imitation Learning*, 2017
  ([arXiv:1703.09327](https://arxiv.org/abs/1703.09327)). Noise injected into the demonstrator's
  actions so that the data shows recoveries.
- Zawalski et al., *Robotic Control via Embodied Chain-of-Thought Reasoning*, 2024
  ([arXiv:2407.08693](https://arxiv.org/abs/2407.08693)). A VLA trained to reason about plans,
  sub-tasks and object positions before acting.
- Shi et al., *Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action
  Models*, 2025 ([arXiv:2502.19417](https://arxiv.org/abs/2502.19417)). A VLM chooses the next step
  and a low-level policy carries it out: the two-level form of Phase 7.

## Check yourself

1. From `MIN_OBJECT_GAP` in [`sim/pickup/disturb.py`](../../sim/pickup/disturb.py) (0.11 m, the
   gap `PickupEnv.reset` uses) and the area in `SPAWN_X`, `SPAWN_Y` (`env.py`), how much room does a
   drifting object have between two others? At `--drift_cm_s 8`, how far does it go between two plans of a policy
   that executes 8 actions per chunk?
2. Run the expert loop above at 1 and 8 cm/s. At which speed does the one-plan expert stop
   succeeding, and does it fit the tolerance table?
3. Read `ScriptedExpert._track` and `_needs_new_plan`. A drift at 2 cm/s: does the expert track or
   plan again, and how often (`tracks` and `replans` in the dataset's `episodes.jsonl`)?
4. Collect 10 `pick_lookalike` demonstrations (`.venv/bin/python -m sim.pickup.collect --task
   pick_lookalike --episodes 10 --out data/lerobot/lookalike_check`) and print
   `meta/tasks.jsonl`. Which instruction is contained in another?
