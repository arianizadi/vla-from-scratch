---
title: History and the Shortcut It Opens
part: 5 Conditioning the action head
order: 52
summary: Tell which history inputs a config gives a policy and how they are filled in training and when served, explain how the previous action lets a policy fit demonstrations without looking at the scene, and test a trained policy for that shortcut with two probes and the as-served baseline.
status: draft
---

# History and the Shortcut It Opens

**History** is what a policy is told about the moment before the current one: here each camera's
previous frame, the previous joint state and the last action. It shows motion and lets a policy
continue what it started, but in demonstrations (the expert recordings a policy is trained on) the
previous action also predicts the next one without the scene. On `pick_target` (three objects,
an instruction that names one) a policy given history, trained before a position bug was fixed,
stayed at 2%; probes show it read its target from its previous action. Trained again after the fix,
on demonstrations that begin with one frame of the robot at rest, a history policy still ends at
0% / 4% after 5M samples, against 93% without history.

## The idea

One observation of the pickup simulator is two camera frames and an 8-number state (seven joint
angles, the gripper opening). Nothing in it says whether the arm is moving or what the policy
decided last. History supplies both:

- the observation one **control step** (one command to the arm, 50 ms) before: one frame cannot
  tell an arm at rest from one moving at 0.4 m/s, two can;
- the **previous action**, the command sent last, so that a new **chunk** (the 15 future actions a
  policy predicts in one go) can carry on from the one before.

The demonstrations are recordings of the **scripted expert** (`ScriptedExpert` in
[`expert.py`](../../sim/pickup/expert.py)), a hand-written controller that reads the object's true
position. In `_plan` and `act`, its set-point (`des_pos`, the gripper position it steers toward)
moves in straight lines at no more than 0.40 x 1.2 m/s during the reach, 24 mm per control step;
each action is the seven joint angles `solve_ik` finds for it, plus the gripper command, 1 (open)
until the grasp, then 0: one change in about 66 control steps (lab notebook, "Tasks and data").

So "the previous action, a little further the same way" is close to the expert's next action at
almost every control step, without a camera. Only the direction of the reach depends on the
scene: the objects lie anywhere in a 24 cm by 44 cm area (`SPAWN_X`, `SPAWN_Y` in
[`env.py`](../../sim/pickup/env.py)).

A **shortcut** is a rule that fits the training data through an input that does not carry the same
information when the policy runs; the literature calls this causal confusion, and the case of the
expert's previous action the copycat problem ([Imitation learning](43-imitation-learning.md)). The
expert's previous action was caused by what the expert saw. In a **rollout** (the policy driving
the simulated arm) it is the policy's own, and before the first control step it is the **hold
command**: the start joint angles, gripper open.

<!-- interactive: copycat -->
*Figure: a joint target and the gripper command of one demonstration beside the previous action
repeated: a small, steady gap, and one jump where the gripper closes.*

## How it works

### What history adds to the sequence

The **action head**, the transformer that turns noise into a chunk, reads one sequence: a time
token for the noise level, the **conditioning** (the tensor the backbone hands it,
`[batch, tokens, width]`: here 32 instruction tokens and 196 patch tokens per image), state tokens, and **action-step tokens**, one per row of the action tensor
([What the action head sees](50-what-the-action-head-sees.md)):

```text
dp_vit_lang_reactive.yaml, no history                          1 + 424 + 1 + 15 = 441 tokens
  image_indices: [0]        lowdim_past_timesteps: 0
[ time ][ 32 text + 2 x 196 patch ][ state t ][ 15 action-step tokens ]

dp_vit_lang.yaml, history                                      1 + 816 + 2 + 16 = 835 tokens
  image_indices: [-1, 0]    lowdim_past_timesteps: 1
[ time ][ 32 text + 4 x 196 patch ][ state t-1 ][ state t ][ previous action ][ 15 action-step tokens ]
```

The previous action is row 0 of the action tensor, `[batch, 16, 8]`: the **previous-action slot**.
Training adds noise to, and scores, only the rows the policy has to predict
([Flow matching](41-flow-matching.md)). Row 0 gets no noise and no loss: it enters as recorded,
directly in front of the first action to predict.

<!-- interactive: sequence-layout -->
*Figure: the policy sequence for both configs. History doubles the patch cells and adds a state
cell and a noise-free sixteenth action-step cell.*

### What the slot holds

A **window** is one training example: the observations at one frame of a demonstration, its
**anchor**, and the actions that follow. The history inputs in training and when the policy is
**served** (answers the simulator's queries for a chunk):

| | Previous frame | Previous state | Previous-action slot |
|---|---|---|---|
| Window anchored at frame 1 or later | frame $t-1$ | state $t-1$ | the expert's command at $t-1$ |
| Window anchored at frame 0 | frame 0 again | state 0 again | a copy of the expert's command at frame 0 |
| Served, first query | the first frame again | the first state again | the hold command |
| Served, later queries | one control step back | one control step back | the command executed last |

Rows two and three are the same moment: frames and states agree, the slot does not.

The joint angles of the expert's first command put the gripper 16 to 24 mm from the start,
straight toward a point above the named object (`0.40 * speed / 20` m). Every reach starts near
the same pose (`HOME_Q` plus noise of standard deviation 0.03 rad per joint, in
`PickupEnv.reset`), so its direction tells where the reach ends: an argument from geometry, which a
probe below tests.

A worked example: of the 66 windows of a 66-frame demonstration, 65 have a real previous action
and one, at frame 0, a copy of the first row to be predicted. A rollout of the same length makes 9
queries (8 actions executed per chunk); the first, which chooses the object, carries the hold
command: a **train/inference skew** ([Train/inference skew](93-train-inference-skew.md)).

### Why the policy takes it

An argument, not a measurement. The loss asks only for the expert's chunk. One way to produce it
finds the named object among 784 patch tokens and 32 instruction tokens; the other continues the
command in the slot and gets close in every training window, above all the one anchored at frame 0,
where the slot holds a copy of the very command to output first.

Without history, only the windows of the first few frames cannot be fitted without looking:
later, the joint state shows which way the arm is going (field notes, "Looking at the scene arrives
late, and all at once"). With history no window requires looking. Without it, looking is learned
late: between 1.5M and 2.5M **samples** (training windows drawn by the data loader) on the same
data with the old position table (see below); after the fix, by about 1.0M (closed at the named
object 65%, slope 0.58 at 1.0M; [How much data and training](71-how-much-data-and-training.md)).

## In this repository

**The keys.** [`preprocess.yaml`](../../sim/pickup/configs/preprocess.yaml) stores every window
with history (`past_lowdim_steps: 1`, `image_indices: [-1, 0]`, `padding_strategy: copy`);
`extract_robotics_fields` in [`pipelines/robotics.py`](../../vla_foundry/data/pipelines/robotics.py)
crops action, state and masks to the config's `lowdim_past_timesteps`. One key covers state and
action, so the previous action goes only with the previous state. Six of the ten policy configs in
`sim/pickup/configs/` have history; the four `*_reactive` configs set `image_indices: [0]` and
`lowdim_past_timesteps: 0`.

**The copy.** For the window anchored at frame 0, `LeRobotConverter.extract_sample_data` in
[`converters/lerobot.py`](../../vla_foundry/data/preprocessing/robotics/converters/lerobot.py):

1. sets `lowdim_start = anchor_timestep - past_lowdim_steps`, here -1, so `past_padding` is 1;
2. slices the 15 rows that exist and passes each low-dimensional array through `self.pad_fn`, set
   from `padding_strategy` in
   [`converters/base.py`](../../vla_foundry/data/preprocessing/robotics/converters/base.py);
3. with `copy`, that is `PaddingStrategy.copy_edge` in
   [`preprocess_masks.py`](../../vla_foundry/data/preprocessing/robotics/preprocess_masks.py),
   `np.pad(..., mode="edge")`, which repeats the first row.

Image indices are clipped to frame 0. `create_past_and_future_masks` marks the copy
(`past_mask[0]` is false), but `DiffusionPolicy.forward`
([`diffusion_policy.py`](../../vla_foundry/models/diffusion_policy/diffusion_policy.py)) does not
read `past_mask`: `torch.where(future_mask.unsqueeze(-1), noisy_action, actions)` passes row 0
clean, real previous action or copy.

**Serving.** `main` in [`eval.py`](../../sim/pickup/eval.py) sends with every query the frames
and states of `prev_obs` and `obs`, and `env.last_action`; before the first control step
`prev_obs` is `obs` and `env.last_action` the hold command from `PickupEnv.reset`. `Policy.infer`
in [`policy_server.py`](../../sim/pickup/policy_server.py) then:

1. decodes `self.image_names` (four images with history, two without) and keeps
   `state[-(self.past + 1):]`;
2. writes the previous action into `action_seq[: self.past]` (row 0, or none) and marks it in a
   `past_mask`;
3. calls `generate_actions`, which holds the marked row fixed through every **flow step** (one
   step of the sampler from noise toward actions);
4. returns the 15 rows behind it (`out[0, self.past:]`).

**The probes.** A **probe** is an evaluation of a trained policy with one condition changed, read
against the **as-served baseline**, the policy evaluated as it is served. Two options of `eval.py`
test the slot:

- `--first_prev_action expert` (default `hold`): the first query carries the scripted expert's
  first command, not executed, as previous action.
- `--expert_prefix K` (default 0): the scripted expert drives the first `K` control steps toward
  the named object (`--prefix_target named`, the default), then the policy takes over.

```bash
run=experiments/pickup/my_history_run
out=results/pickup/my_history_run/pick_target
PICKUP_OUT=$out/as_served    sim/pickup/scripts/eval.sh $run pick_target 100
PICKUP_OUT=$out/first_expert sim/pickup/scripts/eval.sh $run pick_target 100 --first_prev_action expert
PICKUP_OUT=$out/prefix_10    sim/pickup/scripts/eval.sh $run pick_target 100 --expert_prefix 10
.venv/bin/python -m sim.pickup.summarize --task pick_target --runs my_history_run
```

A policy that does far better in the second evaluation than in the first reads its target from
the slot; a gain in the third alone cannot tell the slot from the arm's own motion ("Where it goes
wrong"). `PICKUP_CHECKPOINT=<path>` picks a **checkpoint**, the training state saved at one point
(default: the latest), and `PICKUP_WEIGHTS=raw|ema` its trained or **averaged weights** (EMA, an
exponential moving average of the trained weights; the default).

**A hold frame at the start.** `collect.py --hold_steps K` (default 0) records `K` frames of the
robot at rest, with `env.last_action.copy()` as their action (`_worker`). With `K = 1` the slot
holds the hold command in the windows anchored at frames 0 and 1, and the second has the expert's
first chunk as target: what the first served query must produce. That removes the copied answer
from the first window; from the second control step on, the previous action still predicts the
next one.

```bash
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k_hold --hold_steps 1
PICKUP_NUM_CHECKPOINTS=10 sim/pickup/scripts/train_resilient.sh dp_vit_lang pick_target_4k_hold \
    my_hold_run 5000000
sim/pickup/scripts/eval_checkpoints.sh experiments/pickup/my_hold_run pick_target
```

## What we measured

The three-object runs are on machine B (L40S GPUs, 128 CPU cores): the patch-token policy with
instruction tokens, one seed per run, 100 unseen **layouts** per evaluation (fixed by a seed: the
objects, their places, the named one, the start pose). The runs with history in the first two
tables, and the no-history run they are compared with, were trained with the old position table.
The hold-frame run, and the no-history run at 93% (it saved no averaged weights), were trained
after the fix. Pairs are trained / averaged weights (EMA). **Closed at the named object**: the object nearest the gripper when it first closed was the named one (chance 33%).
**Slope**: of where the gripper first closed against where the named object was; 1 if it closes
at the object, 0 if the place ignores the scene.

**With and without history, three objects.** `dp_vit_lang_reactive` and `dp_vit_lang`, same 4000
demonstrations, every checkpoint (the notebook's "4000 demonstrations" tables):

| Samples | Success, no history | Slope, no history | Success, history | Slope, history |
|---|---|---|---|---|
| 0.5M | 0% / 0% | 0.02 / -0.06 | 0% / 0% | -0.06 / -0.05 |
| 1.0M | 0% / 1% | 0.11 / 0.14 | 1% / 1% | 0.03 / 0.04 |
| 1.5M | 4% / 2% | 0.47 / 0.37 | 1% / 1% | 0.10 / 0.07 |
| 2.0M | 4% / 36% | 0.81 / 0.96 | 3% / 1% | 0.11 / 0.06 |
| 2.5M | 56% / 70% | 0.94 / 0.98 | 1% / 1% | 0.06 / 0.03 |
| 3.0M | 81% / 75% | 0.98 / 0.97 | 2% / 2% | 0.11 / 0.08 |

Without history the slope reaches 0.81 / 0.96 at 2.0M samples. With history it
never passes 0.11; that run, stopped at 3.0M samples, closed at the named object in 26% to 35% of
episodes at every checkpoint. Without history the run went on to
83% / 85% at 5.0M samples.

**What the history policy depends on.** The history run's EMA weights at 2.5M samples, 100
layouts per row:

| Condition | Success | Closed at the named object | Median miss |
|---|---|---|---|
| As served: hold command as first previous action | 1/100 | 30% | 186 mm |
| Expert's first command as first previous action | 22/100 | 97% | 25 mm |
| Expert drives the first 10 control steps toward the named object | 77/100 | 100% | 12 mm |

- **It reads the target from the slot.** Rows one and two differ in one input of one query, a
  command never executed, and the gripper closes at the named object in 97% of episodes instead
  of 30%. Finding is not grasping: 25 mm off its grasp point, the scripted expert itself succeeds
  in 52% (grasp tolerance, machine B).
- **It learned the grasp, not where to start**: handed over ten control steps in, it succeeds in
  77/100.

**A hold frame, after the fix.** `dp_vit_lang_pt4k_hold_posfix`: `dp_vit_lang` on
`pick_target_4k_hold` (4000 demonstrations, each begun with a hold frame), 5M samples: 0% / 4% at
the final checkpoint. Its slope stays between -0.11 and 0.12 at every checkpoint, and it closes at
the named object in 28% to 43% of episodes. Without history, on the plain demonstrations, the same
budget ends at 93% (trained weights).

**The single cube.** `pick_cube` has one object, so there is nothing to choose (1000
demonstrations, 1.5M samples, final checkpoints). `dp_vit` (history, no language input) ended at
62% / 70% with the old position table and at 96% / 98%, slope 0.99 / 0.99, with the corrected one; `dp_vit_lang_reactive` (no history)
at 95% / 91% before the fix and 91% (trained weights only) after. Both pairs differ in more than history (language
input, pixel range), one seed each; in the first, the old table also hit the history policy harder
(16 action-step tokens on 4 positions against 15 on 8), and that, not history, held it at 62% / 70%.

### What is and is not established

The **rotary position table** gives each token its place in the sequence
([Positions](12-positions.md)). Up to and including the second round it was computed in bfloat16
under autocast: 641 distinct positions out of 2048 (fixed in `vla_foundry/positional_embedding.py`).
Tokens that share a position get the same rotation, so attention cannot tell from place which step
of the chunk each one stands for.

Established, one run each: the three-object history policy, as trained, reads its target from the
slot; and a hold frame at the start of every demonstration does not make a history policy look.
The first rests on the second probe, which
changes only the previous action sent with the first query (2.5M samples, averaged weights), not on
the comparison of success rates.

Not established:

- **How much of the three-object gap was history.** The 2% against 81% / 75% at 3.0M samples
  compares runs trained with the old table; no history run on the plain demonstrations has been
  repeated after the fix. The one made after it (hold frame) ended at 0% / 4%, so the old table
  does not explain the gap alone.
- **Which history input keeps the policy from looking.** The slot carries the target at the first
  query, but the previous frame and state also show motion; no run removes the previous action
  alone or hides the slot during training.

The whole route on three objects is in [the case study](101-case-pick-target.md).

## Where it goes wrong

| What you see | Why | How to check | What to do |
|---|---|---|---|
| A policy with history near 0% at every checkpoint, slope near 0 | It takes its target from the slot, which holds the hold command at the first query | The two probes above | Train without history (`dp_vit_lang_reactive`) |
| History looks costly, or free | The runs differ in more than history, or neither has started to look ([How much data and training](71-how-much-data-and-training.md)) | Diff the configs; evaluate every checkpoint, read the slope | Change one key at a time, on a budget at which one run works |
| The prefix probe is read as proof of the slot | The prefix also moves the arm, and a policy without history trained on clean demonstrations from one start pose follows its arm (after 25 expert steps toward another object it closed there in 100/100 episodes; policies trained with recoveries turn back) | `--first_prev_action expert` changes the slot alone | [Probing a policy that works](95-what-a-policy-looks-at.md) |
| A no-history policy with a suspiciously low open-loop error | `lowdim_past_timesteps: 0` read as unset hands it the state one step ahead | A batch's `proprioception` tensor has `lowdim_past_timesteps + 1` rows | The `is None` check in `RoboticsProcessor.add_action_and_proprioception_fields`; fixed here, check forks |

## Background and further reading

Background, not measured here:

- de Haan et al., *Causal Confusion in Imitation Learning*, 2019
  ([arXiv:1905.11979](https://arxiv.org/abs/1905.11979)). Policies given the previous action fit held-out demonstrations as well or better and earn
  less reward.
- Wen et al., *Fighting Copycat Agents in Behavioral Cloning from Observation Histories*, 2020
  ([arXiv:2010.14876](https://arxiv.org/abs/2010.14876)). Names the copycat problem (the imitator
  repeats the expert's previous action, recovered from its history) and trains features
  adversarially to shed it.
- Chi et al., *Diffusion Policy*, 2023 ([arXiv:2303.04137](https://arxiv.org/abs/2303.04137)).
  Conditions on the latest two observations for most tasks, without the previous action.
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, 2023
  ([arXiv:2304.13705](https://arxiv.org/abs/2304.13705)). ACT predicts a chunk from the current
  observation alone and argues that chunking handles pauses without the causal confusion of
  history-conditioned policies. The `*_reactive` configs make the same choice.

## Check yourself

1. Predict what this prints for the window anchored at frame 0, and say which row is the copy:

   ```bash
   .venv/bin/python - <<'EOF'
   import numpy as np
   from vla_foundry.data.preprocessing.robotics.preprocess_masks import PaddingStrategy

   actions = np.arange(66, dtype=np.float32)[:, None]                # row t: the command at frame t
   window = PaddingStrategy.get_pad_fn("copy")(actions[0:15], 1, 0)  # one row missing on the left
   print(window.shape, window[:3, 0])
   EOF
   ```

2. Build plain demonstrations (the README's `pick_target_4k` adds random starts and recoveries).
   Compare the error of repeating the previous action with that of replaying the mean of all
   demonstrations, whatever the layout (a policy that ignores scene and slot). What does the ratio
   say?

   ```bash
   sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k_plain
   .venv/bin/python - <<'EOF'
   import glob
   import numpy as np
   import pyarrow.parquet as pq

   def actions(path):                                           # [control steps, 8]
       return np.array(pq.read_table(path, columns=["action"])["action"].to_pylist(), dtype=np.float32)

   paths = sorted(glob.glob("data/lerobot/pick_target_4k_plain/data/chunk-000/episode_*.parquet"))[:200]
   episodes = [actions(p) for p in paths]
   length = min(len(e) for e in episodes)
   a = np.stack([e[:length] for e in episodes])                 # [episodes, control steps, 8]
   repeat = np.abs(a[:, 1:] - a[:, :-1]).mean(axis=(0, 1))
   average = np.abs(a - a.mean(axis=0)).mean(axis=(0, 1))
   print("repeat the previous action :", np.round(repeat, 4))
   print("the average demonstration  :", np.round(average, 4))
   print("ratio:", np.round(repeat / average, 3))
   EOF
   ```

3. Train a policy with history on those demonstrations and run the baseline and the two probes on
   it. Predict the three rows for a `dp_vit_lang_reactive` run, and name the step of
   `Policy.infer` that keeps `--first_prev_action expert` from reaching it.

   ```bash
   PICKUP_NUM_CHECKPOINTS=10 sim/pickup/scripts/train_resilient.sh dp_vit_lang pick_target_4k_plain \
       my_history_run 5000000
   ```

4. Collect 20 demonstrations with a hold frame. Which action row matches the resting pose, and
   which two windows have it in the slot?

   ```bash
   sim/pickup/scripts/make_dataset.sh pick_target 20 pick_target_20_hold --hold_steps 1
   .venv/bin/python - <<'EOF'
   import pyarrow.parquet as pq

   path = "data/lerobot/pick_target_20_hold/data/chunk-000/episode_000000.parquet"
   rows = pq.read_table(path, columns=["state", "action"]).to_pylist()
   for name, row in (("state 0 ", rows[0]["state"]), ("action 0", rows[0]["action"]),
                     ("action 1", rows[1]["action"])):
       print(name, [round(v, 3) for v in row])
   EOF
   ```
