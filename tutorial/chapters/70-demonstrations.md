---
title: Demonstrations
part: 7 Data
order: 70
summary: Generate a demonstration dataset with the scripted expert, choose what its episodes contain, say what its files and arrays hold, and open a training shard to check it.
status: draft
---

# Demonstrations

A policy trained by imitation learns only from recorded examples of the task being done. This
chapter covers where they come from, what one contains, how episodes become the windows the trainer
reads, and what they leave out unless the collection script puts it in.

## The idea

A **demonstration** is one recorded episode of the task done correctly: at every control step, what
the robot observed and the command that was sent next. An episode of `pick_target` is about 66
control steps at 20 Hz (lab notebook), and each one stores:

| Field | Shape | Content |
|---|---|---|
| `front`, `wrist` | 224x224x3 each, JPEG | the two camera frames |
| `state` | `[8]` | 7 measured joint angles and the gripper opening in [0, 1] |
| `action` | `[8]` | 7 joint position targets and the gripper command (1 = open, 0 = closed) |

plus one instruction for the whole episode, for example "pick up the green can". `action[t]` is the
command applied after observing frame `t`.

Background: demonstrations for real robots usually come from teleoperation, a person steering the
arm while the robot records. In a simulator a **scripted expert** can do it: a
hand-written controller that reads the true object pose from the simulator, which the policy never
sees, and follows a plan made from it. Its success rate can be measured before anything is trained.

<!-- interactive: pipeline -->
*Figure: the whole loop from scripted expert to success rate. This chapter is the first three
boxes.*

## How it works

### The task and its randomisation

`PickupEnv.reset(seed)` in [`env.py`](../../sim/pickup/env.py) builds one **layout** from
`np.random.default_rng(seed)`: the arm's start pose, which objects are on the table and where
(the scene), and which one the instruction names. It draws:

1. the arm's start pose: `HOME_Q` plus Gaussian noise of 0.03 rad per joint;
2. the objects: for `pick_target`, 3 of the 4 in `OBJECTS`;
3. the target among them, which fixes the instruction `"pick up the <label>"`;
4. for each object a position, uniform in `SPAWN_X = (0.38, 0.62)` and `SPAWN_Y = (-0.22, 0.22)`
   metres and redrawn until it is more than 0.11 m from the objects already placed, and a yaw
   uniform in $[-\pi, \pi]$.

Demonstrations use seeds counting up from 0 (`collect.py --seed`). Evaluation uses seeds from
`EVAL_SEED_START = 1_000_000` in [`eval.py`](../../sim/pickup/eval.py), so no evaluated layout was
trained on ([Closed-loop evaluation](90-closed-loop-evaluation.md)).

An episode succeeds when the named object has been more than `LIFT_HEIGHT = 0.08` m above its rest
height for `HOLD_STEPS = 10` consecutive control steps, and ends then or after `max_steps = 200`.

### The scripted expert

`ScriptedExpert._plan` in [`expert.py`](../../sim/pickup/expert.py), called from `reset`, reads the
target's position and yaw and writes four **waypoints** for the tool centre point (TCP, the point
between the fingertips), each with a gripper command, a speed limit and a wait after arrival:

1. 0.10 to 0.14 m above the grasp point, gripper open, at most 0.40 m/s x `speed`;
2. the grasp point, gripper open, at most 0.18 m/s x `speed`, then wait 2 control steps;
3. the same point, gripper closed, wait 8 control steps;
4. 0.18 to 0.24 m above it, gripper closed, at most 0.30 m/s x `speed`, until the episode ends.

The grasp point is the object's position plus Gaussian noise (3 mm standard deviation on x and y);
`speed` is drawn once per plan from 0.8 to 1.2. The fingers are turned to the yaw of the cube and
the T block; for the can and the hex nut, which can be grasped at any angle, the yaw offset is drawn
from -0.3 to 0.3 rad. The noise keeps demonstrations from being copies of one trajectory.

Once per control step, `ScriptedExpert.act` moves a set-point toward the current waypoint by at most
`speed limit / 20` metres, and advances when the set-point has arrived, the real TCP is within
12 mm and the wait has passed. The recorded action is in joint space, so each set-point goes
through **inverse kinematics** (IK: joint angles that put the TCP at a wanted pose). `solve_ik`
runs at most 30 iterations of damped least squares (damping 0.05) from the previous step's solution and clips to the actuator
limits. Its result plus the gripper command is the 8-D action.

That is one plan per episode. `ScriptedExpert(env, reactive=True)` also checks before every action
whether the named object has moved more than 15 mm from where the plan found it, or whether the
fingers have met on nothing during the lift; if so it plans again from where the gripper is, rising
first when the gripper is low.

<!-- interactive: arm-replay -->
*Figure: a 3D replay of one episode: the gripper moves above the named object, descends, closes
and rises.*

### Recording

Each control step, `_worker` in [`collect.py`](../../sim/pickup/collect.py) asks the expert for an
action, appends the two current frames (JPEG, quality 95), the state and the action, then steps the
simulator. An episode that does not end in success is dropped. `main` repeats rounds of attempts
until enough succeeded, sorts by seed and keeps the first `--episodes`.

The output uses the file layout of dataset format v2.0 of LeRobot, Hugging Face's robot-learning
library, with one Parquet file (a column-oriented table format) per episode:

```text
data/lerobot/<name>/
  meta/info.json        fps, totals, feature shapes, expert_success_rate, expert_attempted, the options used
  meta/episodes.jsonl   per episode: episode_index, tasks (the instruction), length, seed, disturbed, replans
  meta/tasks.jsonl      the distinct instructions
  data/chunk-000/episode_000000.parquet   one row per frame: front, wrist, state, action, indices
```

It holds only what the framework's `--type lerobot` converter reads; the LeRobot library itself
was not tried on it.

### From episodes to windows

A **window** is one training example: the observations at one frame, its **anchor**, plus the
actions that follow. [`preprocess.yaml`](../../sim/pickup/configs/preprocess.yaml) sets
`past_lowdim_steps: 1` ("lowdim": the low-dimensional arrays, `state` and `action`),
`future_lowdim_steps: 14`, `image_indices: [-1, 0]` and `stride: 1`: a window anchored at frame `t`
holds `state` and `action` for frames `t-1` to `t+14`, shape `[16, 8]` each, and both cameras at
`t-1` and `t`. The 15 actions from the anchor on are the **chunk** a policy learns to predict;
slot 0 is the past, given only to a policy with history
([Chunks and execution](42-chunks-and-execution.md)).

`BaseRoboticsConverter.process_episode`
([`converters/base.py`](../../vla_foundry/data/preprocessing/robotics/converters/base.py)) cuts one
window for every anchor in `range(0, episode_length, stride)`: one per frame. For an episode of 66
frames (0 to 65):

| Anchor | Frames in the 16 slots | Copied slots | `future_mask` true |
|---|---|---|---|
| 0 | 0, 0, 1, ..., 14 | slot 0 | slots 1-15 |
| 65 | 64, 65, 65, ..., 65 | slots 2-15 | slot 1 |

Where a window reaches past the episode, `padding_strategy: copy` repeats the first or last frame.
Here 15 of the 66 windows (anchors 0 and 52 to 65) contain copied slots.

`create_past_and_future_masks` marks the real slots (`past_mask`, `future_mask`).
`DiffusionPolicy.forward` adds noise only where `future_mask` is true and
[`vla_foundry/train.py`](../../vla_foundry/train.py) passes it to the masked MSE loss, so the
copies at the end of an episode never enter the loss or the statistics.

The copy at the start is different: slot 0 is what a policy with history is given as its previous
state and action. In the window anchored at frame 0 that action is `action[0]` again, the expert's
first command, which already heads for the target. Served, the policy finds the **hold command**
there: the start joint angles with the gripper open
([History and shortcuts](52-history-and-shortcuts.md)).

[`preprocess_robotics_to_tar.py`](../../vla_foundry/data/preprocessing/preprocess_robotics_to_tar.py)
shuffles all windows and packs them 256 at a time ([`make_dataset.sh`](../../sim/pickup/scripts/make_dataset.sh))
into **shards**, tar files read sequentially by the `webdataset` loader; `shards/manifest.jsonl`
lists them, and `shards/stats.json` holds the per-channel statistics of `state` and `action` that
`RoboticsNormalizer` normalises with ([What an action is](40-action-spaces.md)).

### What the options add

By default every demonstration is one clean pass that starts near `HOME_Q`, with the one
instruction its seed drew. Four options of `collect.py` change that:

| Option | What it records | Why it exists |
|---|---|---|
| `--targets_per_layout N` | every seed's scene `N` times, each with another object named | by default no scene appears with two instructions |
| `--disturb_prob p` | in a share `p` of the episodes the named object is moved once, 5 to 10 cm, during the reach or as the gripper closes ([`disturb.py`](../../sim/pickup/disturb.py)); the expert runs with `reactive=True` | by default nothing moves and no grasp is repeated |
| `--random_start` | every episode from a random gripper pose above the table (`random_start` in `expert.py`) | from one start pose the arm's position gives the goal away a few steps into a reach |
| `--hold_steps K` | `K` frames of the robot at rest, under its hold command, first | with `K = 1`, slot 0 of the window anchored at frame 0 holds the hold command, as when served; it did not make a history policy look ([Where it goes wrong](#where-it-goes-wrong)) |

With one instruction per scene, a function of the image alone that has stored which object was
picked in each training scene fits the data as well as one that reads the instruction: the loss
cannot tell them apart, whatever training then finds ([What we measured](#what-we-measured)).
`PickupEnv.reset(seed, target_index=i)` names the seed's `i`-th object in an identical scene, so a
model that ignores the instruction can at best draw one of the `N` trajectories at random.

[`boost_early_windows.py`](../../sim/pickup/boost_early_windows.py) reweights a preprocessed
dataset instead: in its copy the windows anchored at frames 0 to `--max_frame` (default 7) appear
`--repeat` (default 5) extra times: only the first frames need the scene to choose the target. On
a 66-frame episode 8 of 66 windows (12%) become 48 of 106 (45%).

## In this repository

```bash
# writes data/lerobot/<name> and data/preprocessed/<name>, replacing existing ones;
# arguments after the name go to collect.py
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k_rs_recover --random_start --disturb_prob 0.5

# the other datasets of the lab notebook
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_cf4k --targets_per_layout 3
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k_recover --disturb_prob 0.5
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k_rs --random_start
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k_hold --hold_steps 1

# copies with the windows anchored at frames 0-7 repeated
.venv/bin/python -m sim.pickup.boost_early_windows \
    --shards data/preprocessed/pick_target_4k/shards \
    --out data/preprocessed/pick_target_4k_early/shards --max_frame 7 --repeat 5
.venv/bin/python -m sim.pickup.boost_early_windows \
    --shards data/preprocessed/pick_target_4k_rs_recover/shards \
    --out data/preprocessed/pick_target_4k_rs_recover_early/shards --max_frame 7 --repeat 5
```

The first command makes the data of the run [README](../../README.md) recommends, which names the
folder `pick_target_4k`: `dp_vit_lang_reactive`, 5M samples (windows drawn by the data loader),
94% / 94%. Its config saves averaged weights itself (`ema: enabled: true`). The names and options
are the lab notebook's.

To inspect the first window of a shard:

```bash
.venv/bin/python - <<'EOF'
import io, json, tarfile
import numpy as np

shard = "data/preprocessed/pick_target_4k/shards/shard_000000.tar"
with tarfile.open(shard) as tar:
    members = tar.getmembers()
    key = members[0].name.split(".", 1)[0]  # shared by the files of one window
    files = {m.name.split(".", 1)[1]: tar.extractfile(m).read() for m in members if m.name.startswith(key + ".")}

print(sorted(files))
lowdim = np.load(io.BytesIO(files["lowdim.npz"]))
print({name: lowdim[name].shape for name in lowdim.files})
print(json.loads(files["language_instructions.json"]))
meta = json.loads(files["metadata.json"])
print({k: meta[k] for k in ("episode_index", "frame_index", "anchor_relative_idx")})
EOF
```

It lists seven files: four JPEGs (`front_t-1.jpg` to `wrist_t0.jpg`), `lowdim.npz` (`state` and
`action` of shape `(16, 8)`, two masks of shape `(16,)`), `metadata.json` and
`language_instructions.json`. `anchor_relative_idx` is 1: the anchor is slot 1 of the 16.
`episode_index` and `frame_index` are stored as strings; convert with `int()` before comparing.

A training config may use less than what is stored
([`dp_vit_lang_reactive.yaml`](../../sim/pickup/configs/dp_vit_lang_reactive.yaml) sets
`lowdim_past_timesteps: 0` and `image_indices: [0]`), not more: `crop_sequence` in
[`vla_foundry/data/robotics/utils.py`](../../vla_foundry/data/robotics/utils.py) asserts it.

## What we measured

All numbers are from the [lab notebook](../lab-notebook.md).

**The datasets.** Every frame anchors a window, so windows = frames; 5M samples on
`pick_target_4k` are 19.0 passes over its windows (epochs).

| Dataset | Episodes | Windows | What is in it |
|---|---|---|---|
| `pick_cube_1k` | 1000 | 65,391 | one red cube |
| `pick_target_1500` | 1500 | 98,841 | 1500 layouts, one instruction each |
| `pick_target_4k` | 4000 | 263,289 | 4000 layouts, one instruction each |
| `pick_target_cf` | 1500 | 98,616 | 500 scenes, each once per object |
| `pick_target_cf4k` | 4000 | 263,552 | about 1333 scenes, each once per object |
| `pick_target_cf_early` | 1500 | 158,616 | `pick_target_cf`, windows of frames 0-7 present 6 times |
| `pick_target_4k_early` | 4000 | 423,289 | `pick_target_4k`, the same |
| `pick_target_4k_recover` | 4000 | 293,541 | object moved once in 1505 episodes |
| `pick_target_4k_rs` | 4000 | 241,318 | random start poses |
| `pick_target_4k_rs_recover` | 4000 | 275,550 | random start poses; object moved once in 1456 episodes |
| `pick_target_4k_rs_recover_early` | 4000 | 435,550 | `pick_target_4k_rs_recover`, the same |
| `pick_target_4k_hold` | 4000 | 267,075 | one frame of the robot at rest first |

**The expert.** Through the evaluation loop on 100 unseen `pick_target` layouts it scored 100/100
on machine B (L40S GPUs, 128 CPU cores), closing 5 mm (median) from the object and within 10 mm in
98% of episodes, and 99/100 on machine A (one RTX 4090, 24 GB). With the named object moved 80 mm
once per episode (`eval.py --expert --disturb reach --disturb_step 10`, `20` or `30`, or
`--disturb grasp`; 100 layouts each), the one-plan expert scored 0/100, 0/99, 0/100 and 0/100, the
expert that plans again (`--expert_reactive`) 100/100, 96/96, 100/100 and 100/100, and from a random
start pose 98/98 (moved at control step 20) and 100/100 (at the grasp).

**Policies trained on these data.** `dp_vit_lang_reactive` (instruction tokens next to the patch
tokens, no history; [Language conditioning](51-language-conditioning.md)) on machine B, 100 unseen
layouts, raw / EMA weights (the trained weights / their running average). **Closed at the named
object**: the object nearest the gripper when it first closed was the named one (chance 33%). The
**slope** of where the gripper first closed against where the named object was is 1 if it closes
at the object and 0 if the place does not depend on the scene (`summarize.py` prints both).

Before the position fix ([Positions](12-positions.md#the-position-table-under-bfloat16)), the
policy scored 3% / 5% on `pick_target_1500` after 1M samples and 64% / 62% after 3M; after 1M,
0% / 1% on `pick_target_cf` and 8% / 15% on `pick_target_cf_early`; on `pick_target_4k`, 83% / 85%
after 5M. The 1M numbers are not verdicts: this policy starts to use the scene late and all at
once, and 1M samples on 1500 demonstrations end before or inside that transition
([How much data and training](71-how-much-data-and-training.md)).

After the fix, with 4000 demonstrations and 5M samples (the first two runs saved no EMA weights;
the last three columns use EMA weights where saved):

| Dataset | Success | Closed at the named object, 1.0M | Random start | Moved at step 10 / 20 / 30 | Moved at the grasp |
|---|---|---|---|---|---|
| `pick_target_4k` | 93% | 65% | 9/100 | 12/100, 0/97, 0/100 | 0/100 |
| `pick_target_4k_early` | 96% | 99% | 9/100 | 10/100, 0/100, 0/100 | 0/100 |
| `pick_target_cf4k` | 91% / 89% | 98% | 10/100 | 18/100, 0/96, 0/100 | 0/100 |
| `pick_target_4k_recover` | 89% / 84% | 81% | 68/100 | 88/100, 79/96, 85/100 | 78/99 |
| `pick_target_4k_rs` | 98% / 98% | 91% | 97/100 | 94/100, 50/98, 0/100 | 0/100 |
| `pick_target_4k_rs_recover` | 94% / 94% | 100% | 91/100 | 92/100, 89/98, 94/100 | 89/100 |
| `pick_target_4k_rs_recover_early` | 87% / 86% | 93% | 85/100 | 73/99, 76/97, 81/100 | 77/100 |

The policy starts to look sooner whenever more of its windows cannot be answered without looking.
On the unchanged task the differences are small. One instruction per layout did not keep the policy
from following the instruction (on `pick_target_4k` it closed at the named object in 100% of
episodes before the fix and 99% after), and every object named ends no higher. Under disturbances
they are large: random starts follow an early move, recoveries add late moves and a second attempt
(99 of 99 at the grasp), and both together give 89-94% everywhere, also from a random start pose (90/98 moved at control step 20, 89/100 at the grasp). One run each (field notes,
"What the demonstrations contain", "Self-correction has to be in the data").

## Where it goes wrong

- **The expert fails and the data hides it.** Failed episodes are dropped silently. Check
  `expert_success_rate` and `expert_attempted` in `meta/info.json`. A low rate means the kept
  episodes over-represent easy layouts, and with `--targets_per_layout 3` some scenes keep fewer
  than three targets.
- **Same seed, different trajectory.** The layout depends only on the seed; the expert's speed,
  heights and noise come from one generator per worker process, so they also depend on
  `PICKUP_WORKERS` and on the collection rounds.
- **A policy that succeeds and cannot correct.** The policy trained on `pick_target_4k` (93/100)
  scored 0/97 with the named object moved 80 mm at control step 20 of the reach
  (`eval.py --disturb reach --disturb_step 20`) and 0/100 with it moved as the gripper first closed
  (`--disturb grasp`): nothing in its demonstrations moves and no grasp in them fails
  ([Imitation learning](43-imitation-learning.md),
  [What a policy looks at](95-what-a-policy-looks-at.md)). Random start poses and recoveries in the
  data give 89-94% under every disturbance. Evaluate with the disturbances the data was meant to
  teach.
- **A policy with history stays near 0%.** Before the position fix, `dp_vit_lang`, the same policy
  with history, was at 2% / 2% after 3.0M samples on `pick_target_4k`; one checkpoint (2.5M
  samples, EMA) scored 1/100 as served, 22/100 with the expert's first command as its first
  previous action (`eval.py --first_prev_action expert`) and 77/100 with the expert driving the
  first 10 control steps (`--expert_prefix 10`): it took its target from the previous action. After
  the fix, trained on `pick_target_4k_hold`, it ends at 0% / 4% after 5M samples, and where it
  closes never depends on the scene (slope -0.11 to 0.12): the hold frame closes the leak at the
  first frame, but from the second step on the previous action still gives the next one away. Start
  without history ([History and shortcuts](52-history-and-shortcuts.md)).

## Background and further reading

- Todorov, Erez and Tassa, *MuJoCo: A physics engine for model-based control*, IROS 2012
  ([doi:10.1109/IROS.2012.6386109](https://doi.org/10.1109/IROS.2012.6386109)). The simulator
  used here.
- Buss, *Introduction to Inverse Kinematics with Jacobian Transpose, Pseudoinverse and Damped Least
  Squares methods*, unpublished survey, 2009
  ([PDF](https://mathweb.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf)). Explains the update
  used in `solve_ik` next to the alternatives.
- Cadene et al., *LeRobot: An Open-Source Library for End-to-End Robot Learning*, 2026
  ([arXiv:2602.22818](https://arxiv.org/abs/2602.22818)). The library whose dataset format
  `collect.py` writes.
- Ross, Gordon and Bagnell, *A Reduction of Imitation Learning and Structured Prediction to
  No-Regret Online Learning*, AISTATS 2011 ([arXiv:1011.0686](https://arxiv.org/abs/1011.0686)).
  DAgger: the expert labels the states the learner reaches, and they are added to the data. Not
  implemented here.

## Check yourself

1. Make a small dataset. How many attempts did the expert need for 50 kept episodes, and how many
   shards do you expect at 256 windows per shard?

   ```bash
   sim/pickup/scripts/make_dataset.sh pick_target 50 pick_target_50
   cat data/lerobot/pick_target_50/meta/info.json
   grep -c shard data/preprocessed/pick_target_50/shards/manifest.jsonl
   ```

2. An episode has 70 frames. Which anchors give windows with at least one copied slot, and how many
   are there? In which of them does a policy with history read the copy as its previous action, and
   what does that slot hold after `--hold_steps 1`?

3. Collect 50 episodes with recoveries and count the disturbed ones. Why are they usually fewer
   than `--disturb_prob` asks for (`Disturbance.update`)? Why are the episodes marked
   `"disturbed": "grasp"` longer than the others?

   ```bash
   sim/pickup/scripts/make_dataset.sh pick_target 50 pick_target_50_recover --disturb_prob 0.5
   grep -vc '"disturbed": "none"' data/lerobot/pick_target_50_recover/meta/episodes.jsonl
   ```
