---
title: "Case Study: One Cube"
part: 10 Case studies
order: 100
summary: Repeat, run by run, the route that took the single-cube task from a scripted expert to policies above 90%, and say what each run tested, what it measured and what it changed.
status: draft
---

# Case Study: One Cube

`pick_cube` is the smallest task in this repository: one red cube on a table and one instruction.
With nothing to choose, a failure here is a failure to see or to grasp, so the cube is the
baseline for `pick_target`, where the instruction names one of three objects on the table
([Case Study: Three Objects, From 0% to 98%](101-case-pick-target.md)).

## The idea

A **case study** takes one task through every run made on it, in order, with the commands to
repeat them. Each step says what was believed, run, measured and changed.

**The task.** `PickupEnv.reset(seed)` in [`sim/pickup/env.py`](../../sim/pickup/env.py) builds
the **layout** from the seed: where the 45 mm cube lies in a 24 cm by 44 cm area (`SPAWN_X`,
`SPAWN_Y`), its yaw (rotation about the vertical axis, any value) and the arm's start pose near
home. The instruction is always "pick up the red cube". An episode succeeds when the cube has been
more than 8 cm above its rest height for 10 consecutive **control steps** (one command to the arm;
20 per second) and ends after 200.

**The policies.** Each maps the front and wrist camera images (224x224) and the 8-number joint
state to a **chunk**, the next 15 actions (7 joint targets and a gripper command each). A backbone
encodes the images; the **action head**, a transformer, turns noise into the chunk in a few **flow
steps** ([Flow matching](41-flow-matching.md)). Policies are judged **closed-loop**: the policy
drives the simulated arm on layouts it never saw (evaluation seeds start at 1,000,000).

All numbers are from the [lab notebook](../lab-notebook.md), one training seed per run, on machine A
(one RTX 4090, 24 GB) or machine B (L40S GPUs, 128 CPU cores).

## The route, step by step

### 1. The scripted expert

**Believed.** That a top-down grasp from the home pose solves the task, and that the success test
fires when it does.

A **scripted expert** is a hand-written controller that reads the cube's true pose from the
simulator. `ScriptedExpert._plan` in [`expert.py`](../../sim/pickup/expert.py) writes four
waypoints (above the cube, down to it, close, lift), turns the fingers to the nearest of the
cube's four grasp angles and adds 3 mm of Gaussian noise to the grasp point.

**Ran.**

```bash
uv sync --group preprocessing --group sim && sim/pickup/scripts/fetch_assets.sh   # once
.venv/bin/python -m sim.pickup.smoke_test pick_cube 10
# the expert through the evaluation loop: the seeds and the success test a policy gets
.venv/bin/python -m sim.pickup.eval --task pick_cube --episodes 50 --expert \
    --out results/pickup/expert/pick_cube
```

**Measured.** The notebook records 49/50 in a spot check ("The scripted expert") without its
command; `eval.py --expert` repeats it on the evaluation seeds.

**Changed.** The expert became the data source. A **demonstration** is one recorded episode of it;
[`collect.py`](../../sim/pickup/collect.py) keeps only the successful ones.

### 2. 200 demonstrations: one vector against patch tokens

**Believed.** That VLA Foundry's diffusion-policy preset, which `dp_clip.yaml` uses unchanged,
would do, and that the lower training loss would mark the better policy. Its backbone, CLIP
ViT-B/32 (a pretrained image-text encoder), gives one **pooled** vector per image and one for the
instruction: the **conditioning**, the tensor the backbone hands to the action head, is
`[batch, 5, 512]` (two cameras at two frames, plus the text). In `dp_vit.yaml` a small vision
transformer trained from scratch makes every 16-pixel patch of every image a token:
`[batch, 784, 256]` ([One vector or many tokens](21-pooled-vs-patch-tokens.md)).

**Ran.**

```bash
sim/pickup/scripts/make_dataset.sh pick_cube 200 pick_cube_200   # replaces a dataset of that name
# dp_clip downloads CLIP's weights from the Hugging Face Hub on its first start
for cfg in dp_clip dp_vit; do
  sim/pickup/scripts/train.sh $cfg pick_cube_200 ${cfg}_pick_cube_200 300000
  sim/pickup/scripts/eval.sh experiments/pickup/${cfg}_pick_cube_200 pick_cube 50
done
```

A **window** is one training example: the observations at one frame plus the actions that follow.
A **sample** is one window drawn by the data loader; 300k samples on 200 demonstrations are 23
passes over the data (**epochs**). `eval.sh` serves the averaged weights (step 5) unless
`PICKUP_WEIGHTS=raw`.

**Measured.** Machine A, 50 unseen layouts (notebook, "`pick_cube` (machine A)"). The pooled
policy ended at the lower loss, 0.040 against 0.091, and at the lower success, 2/50 = 4% against
15/50 = 30%; [One vector or many tokens](21-pooled-vs-patch-tokens.md) has the intervals and what
the pooled policy did instead of grasping.

**Changed.** Every later policy conditions on patch tokens, and no later decision was taken from a
loss ([Loss is not success](91-loss-is-not-success.md)). It is not a controlled test of pooling:
the policies also differ in pretraining, patch size, model size and text input.

### 3. 1000 demonstrations, 500k and 1.5M samples

**Believed.** That the patch-token policy, at 30% on 200 demonstrations, lacked data and training.

**Ran.** A **checkpoint** is the training state saved at one point (weights, optimiser state, data
position); machine A's runs died or hung several times and resumed from one.

```bash
sim/pickup/scripts/make_dataset.sh pick_cube 1000 pick_cube_1k   # 65,391 windows
# two runs: the notebook gives each budget its own row and its own final loss
sim/pickup/scripts/train.sh dp_vit pick_cube_1k dp_vit_pick_cube_1k_500k 500000
# restarts from the newest checkpoint if the run dies or its log stops moving;
# 1.5M samples at about 620 samples/s (notebook, Throughput): about 40 minutes
sim/pickup/scripts/train_resilient.sh dp_vit pick_cube_1k dp_vit_pick_cube_1k 1500000
sim/pickup/scripts/eval.sh experiments/pickup/dp_vit_pick_cube_1k_500k pick_cube 50
sim/pickup/scripts/eval.sh experiments/pickup/dp_vit_pick_cube_1k pick_cube 50
```

**Measured.** Machine A, 50 unseen layouts, same notebook table, `dp_vit` on 1000 demonstrations,
with Wilson 95% intervals ([Closed-loop evaluation](90-closed-loop-evaluation.md)).

| Samples (epochs) | Final loss | Success | 95% CI |
|---|---|---|---|
| 500k (8) | 0.084 | 20/50 = 40% | 28-54% |
| 1.5M (23) | 0.058 | 37/50 = 74% | 60-84% |

The interval of 40% overlaps that of step 2's 30% (19-44%); 74% is separated from both.

<!-- interactive: loss-vs-success -->
*Figure: training loss next to closed-loop success for the four machine A runs.*

**Changed.** 1.5M samples on `pick_cube_1k`, 22.9 epochs, became the single-cube budget (steps 5
and 6). The first `pick_target` runs got 10.1 (1M samples on 1500 demonstrations) and
ended before where the gripper closed began to depend on where the named object was (the slope of
step 5); the runs that work use 19.0 (5M samples on `pick_target_4k`;
[How much data and training](71-how-much-data-and-training.md)).

### 4. The inference settings

**Believed.** That part of the missing 26 points lay in how a chunk is sampled and executed:
[`eval.py`](../../sim/pickup/eval.py)'s `--flow_steps` (default 10) and `--exec_horizon`, the
actions executed per chunk before the policy is asked again (default 8).

**Ran.**

```bash
run=experiments/pickup/dp_vit_pick_cube_1k; out=results/pickup/dp_vit_pick_cube_1k/pick_cube
PICKUP_OUT=$out/exec_4  sim/pickup/scripts/eval.sh $run pick_cube 50 --exec_horizon 4
PICKUP_OUT=$out/exec_15 sim/pickup/scripts/eval.sh $run pick_cube 50 --exec_horizon 15
PICKUP_OUT=$out/flow_4  sim/pickup/scripts/eval.sh $run pick_cube 50 --flow_steps 4
PICKUP_OUT=$out/flow_25 sim/pickup/scripts/eval.sh $run pick_cube 50 --flow_steps 25
```

**Measured.** The 74% checkpoint, 50 episodes per setting (notebook, below the same table).

| Setting | Success |
|---|---|
| default: 8 actions executed per chunk, 10 flow steps | 74% |
| 4 actions executed per chunk | 84% |
| 15 actions executed per chunk | 78% |
| 4 flow steps | 86% |
| 25 flow steps | 84% |

All intervals overlap.

**Changed.** No setting. Five evaluations of one weights file spread over 12 points, as
50-episode evaluations can by chance. Later evaluations use 100 layouts
([Chunks and execution](42-chunks-and-execution.md) has the 100-layout sweep).

### 5. Without history against with history, on machine B

**Believed.** That **history** (the previous camera frames, joint state and action as extra
inputs) lets a policy stop looking at the scene: on the three-object task, before the position
fix (step 6), the patch-token policy with instruction tokens had failed with history and worked without it
([History and shortcuts](52-history-and-shortcuts.md)). The cube would show what history costs
with nothing to choose.

**Ran.** `dp_vit_lang_reactive.yaml` has no history (`image_indices: [0]`,
`lowdim_past_timesteps: 0`). `dp_vit.yaml`, the config of step 3, has it (`image_indices: [-1, 0]`
and the preset's one past step).

```bash
# dp_vit_lang_reactive loads a tokenizer from the Hugging Face Hub on its first start.
for cfg in dp_vit_lang_reactive dp_vit; do
  PICKUP_WATCH=1 sim/pickup/scripts/eval_checkpoints.sh experiments/pickup/${cfg}_cube1k pick_cube 100 &
  PICKUP_NUM_CHECKPOINTS=10 sim/pickup/scripts/train_resilient.sh $cfg pick_cube_1k ${cfg}_cube1k 1500000
done
wait
.venv/bin/python -m sim.pickup.summarize --task pick_cube --runs dp_vit_lang_reactive_cube1k dp_vit_cube1k
```

These commands now train with step 6's corrected position table. To train with the
old table, as the runs below were, append `--model.transformer.positional_embedding_type
rotary_legacy_autocast` to each `train_resilient.sh` line.

Both configs save a running average of the weights (**EMA**, `ema: enabled: true`) with every
checkpoint; "raw / EMA" below are the trained weights and the average. `GraspProbe` in
[`diagnostics.py`](../../sim/pickup/diagnostics.py) records the **miss**, the horizontal distance
from gripper to cube when the gripper first closed;
`summarize.py` adds the **slope** of where it closed against where the cube was: 1 if it closes
where the cube is, 0 if the scene makes no difference.

**Measured.** Machine B, 100 unseen layouts per cell, six of each run's eleven checkpoints
(notebook, "Single cube"). `PICKUP_NUM_CHECKPOINTS=10` saves one every tenth of the budget and one more at the end ("final").

| Samples | No history: success | Median miss | History: success | Median miss |
|---|---|---|---|---|
| 0.3M | 13% / 54% | 40 / 16 mm | 11% / 38% | 40 / 24 mm |
| 0.6M | 89% / 84% | 11 / 9 mm | 57% / 45% | 16 / 16 mm |
| 0.9M | 77% / 94% | 10 / 7 mm | 73% / 56% | 13 / 15 mm |
| 1.2M | 96% / 94% | 6 / 7 mm | 65% / 68% | 12 / 14 mm |
| 1.5M | 91% / 91% | 7 / 6 mm | 66% / 69% | 12 / 13 mm |
| 1.5M (final) | 95% / 91% | 7 / 6 mm | 62% / 70% | 14 / 14 mm |

From 0.3M samples on, the slope of both runs is between 0.93 and 1.07: both go to the cube. From
0.6M on the history policy closes further from it, 14 / 14 mm against 7 / 6 mm at the end.

**Changed.** The gap was read as the cost of history, skipping two other differences
(notebook, above the history table): `dp_vit` has no instruction tokens (`processor: "none"` drops
the text) and is fed pixels in 0-255; `dp_vit_lang_reactive` adds the instruction as 32 token
slots and its `simple_vlm` processor scales pixels to [-1, 1].

### 6. The same two configurations after the position fix

**Believed.** That step 5 had measured the cost of history, or of one of the other two
differences.

**Rotary position encoding** tells the action head where in its sequence a token sits by rotating
its queries and keys by an angle that grows with the position. The table of those angles had been
computed in bfloat16, so neighbouring high positions got the same angle;
[Positions](12-positions.md) has the mechanism and the count (notebook, "Bugs found and fixed").
Tokens that share an angle look to attention as if they stood at the same place, and only their
content tells them apart.

How hard each config was hit depends on where its actions sit. The action head's sequence is one
token for the noise level (how far the current flow step is between noise and action), the
conditioning, the joint state, then the **action-step tokens**, one per action:

| Config | Tokens before the actions | Action-step tokens | Positions | Distinct angles |
|---|---|---|---|---|
| `dp_vit_lang_reactive` | 426: noise level, 32 instruction, 392 patches, state | 15 | 426-440 | 8 |
| `dp_vit` | 787: noise level, 784 patches, two states | 16 (with the previous action) | 787-802 | 4 |

Past position 512 four neighbours share one angle, so the longer conditioning of the history config
cost it more: step 5 compared two amounts of damage as well as two sets of inputs.

**Ran.** `RotaryEmbedding._update_cos_sin_tables` in
[`vla_foundry/positional_embedding.py`](../../vla_foundry/positional_embedding.py) now computes
the angles in float32, so step 5's commands without the extra argument are this step. The
notebook's runs are `dp_vit_lang_reactive_cube1k_posfix` and `dp_vit_cube1k_posfix`.

**Measured.** Machine B, 100 layouts per cell, six of eleven checkpoints (notebook, the two
"Single cube ... after the fix" sections); the no-history run saved trained weights only.

| Samples | No history: success (raw) | Median miss | History: success | Median miss |
|---|---|---|---|---|
| 0.30M | 28% | 29 mm | 30% / 37% | 27 / 23 mm |
| 0.60M | 70% | 13 mm | 52% / 86% | 20 / 12 mm |
| 0.89M | 81% | 11 mm | 92% / 91% | 9 / 8 mm |
| 1.2M | 96% | 6 mm | 96% / 97% | 7 / 7 mm |
| 1.5M | 97% | 6 mm | 99% / 98% | 7 / 6 mm |
| 1.5M (final) | 91% | 6 mm | 96% / 98% | 8 / 7 mm |

At 0.15M samples the slope was 0.22 / 0.08 with history and 0.68 without: the history policy did
not yet go to the cube, and the other only partly did. From 0.30M on it is at least 0.93 in both
runs.

**Changed.** With only the table corrected, the history configuration went from 62% / 70% to
96% / 98%; the less damaged no-history one ended where it had (91% trained weights, against
95% / 91%). Still without instruction tokens and on pixels in 0-255, the history configuration was
held back on one cube (one run per cell) by the position table, not by history.

### 7. The tolerance of the cube

**Believed.** That a miss of a few millimetres is close enough, without a number for it.

**Ran.** [`grasp_tolerance.py`](../../sim/pickup/grasp_tolerance.py) runs the expert with a fixed
sideways offset, in a random direction, added to its four waypoints.

```bash
mkdir -p results/pickup
.venv/bin/python -m sim.pickup.grasp_tolerance --out results/pickup/grasp_tolerance.json
```

**Measured.** 320 episodes per offset on `pick_target` layouts (the default `--task`), about a
quarter naming the cube; the cube's column is the same on both machines (notebook, "Grasp
tolerance").

| Offset | 0 mm | 5 mm | 10 mm | 15 mm | 20 mm | 25 mm | 30 mm |
|---|---|---|---|---|---|---|---|
| Cube | 100% | 97% | 99% | 95% | 93% | 73% | 32% |

<!-- interactive: tolerance -->
*Figure: grasp success of the expert against the sideways offset added to its grasp point, one
curve per object.*

**Changed.** The miss got a scale. At 0.3M samples in step 5 the raw weights of both runs closed a
median 40 mm away, past the largest offset measured, and scored 13% and 11%. The final
checkpoints of step 6 close 6 to 8 mm away, where the offset expert still succeeds in 97% to 99%
of episodes. That reference is optimistic: it adds one error and keeps angle and timing.

The working policies still fail in 2 to 9 of 100 layouts, not broken down in the notebook. On the
three-object task, before the fix, failed cube grasps were a median 16 mm off and 32 degrees from a valid grasp
angle, against 8 mm and 20 degrees for successes (notebook, "The failures that remain");
exercise 2 asks whether the single cube is alike.

## In this repository

Paths are under `sim/pickup/` unless given in full.

| What | Where | Keys and options |
|---|---|---|
| Task and success test | [`env.py`](../../sim/pickup/env.py): `TASKS`, `PickupEnv.reset`, `PickupEnv.step` | `SPAWN_X`, `SPAWN_Y`, `LIFT_HEIGHT`, `HOLD_STEPS` |
| The three policies | `configs/dp_clip.yaml`, `dp_vit.yaml`, `dp_vit_lang_reactive.yaml` | `vision_language_backbone.type`, `text_tokens`, `processor`, `image_indices`, `lowdim_past_timesteps`, `ema.enabled` |
| Training and evaluation | `scripts/train_resilient.sh`, `eval.sh`, `eval_checkpoints.sh` | `PICKUP_NUM_CHECKPOINTS`, `PICKUP_WATCH`, `PICKUP_OUT`, `PICKUP_WEIGHTS` |
| Miss and slope | `diagnostics.py`: `GraspProbe`; `summarize.py`: `close_slope` | `--task`, `--runs` |
| Position table | `vla_foundry/positional_embedding.py`: `get_pos_embed` | `model.transformer.positional_embedding_type`: `rotary` (default), `rotary_legacy_autocast` (the old table) |

## What one cube can and cannot tell you

What the route shows:

- **Lower loss, worse policy.** The pooled CLIP policy had the lowest loss of step 2 and the lowest
  success.
- **Going to the cube comes early; precision takes the budget.** In all four machine B runs the
  slope is at least 0.93 from 0.3M samples on, with median misses of 16 to 40 mm there; the three
  that end above 90% end at 6 to 8 mm.
- **Averaged weights often lead early, not always.** 54% against 13% (step 5, no history, 0.3M
  samples), 86% against 52% (step 6, history, 0.60M), but 45% against 57% (step 5, history, 0.6M).
  Evaluate both.

What it cannot show:

- **Whether the policy uses the instruction.** There is one instruction, and `dp_vit` never reads
  it; with one object, closing at the named object only measures whether the gripper closed at
  all (100% at nearly every checkpoint; 91% / 89% at 0.15M samples with history after the fix).
- **Whether the policy keeps looking.** A policy that looks once and then follows its own arm
  succeeds on a cube that never moves, as the `pick_target` policy trained on demonstrations from
  one start pose, with nothing moved, did ([Probing a Policy That Works](95-what-a-policy-looks-at.md);
  [Case Study: A Policy That Corrects Itself](103-case-self-correction.md) has the policies that
  keep looking). The cube was not probed (exercise 3).
- **That history is safe.** On `pick_target` the history policy read its target from the previous
  action ([History and shortcuts](52-history-and-shortcuts.md)). With nothing to choose, one cube
  cannot show that shortcut.

## Where it goes wrong

| What you see | Why | What to do |
|---|---|---|
| Your `dp_vit` runs do not match the machine A results | Steps 2 to 4 used the old position table and were not repeated after the fix; `eval.sh` now serves averaged weights, which the machine A table does not record | Evaluate both (`PICKUP_WEIGHTS=raw` for the trained weights); compare with step 6, which shares your code |
| Two settings or two checkpoints differ by 10 points | Evaluation noise: 37/50 has an interval of 60-84%; two checkpoints at about 1.5M samples differed by 6 points on 100 layouts | Use 100 layouts and a second set (`--seed_start 1000100`); never report the best of several |
| A comparison flips after a fix to shared code | The bug hit the two sides differently, here through sequence length | Rerun both sides; list every difference between two configs before naming a cause |

## Background and further reading

- Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", 2023.
  [arXiv:2303.04137](https://arxiv.org/abs/2303.04137). The policy family of the `dp_*` configs,
  which use flow matching instead of the paper's denoising diffusion.
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", 2021.
  [arXiv:2103.00020](https://arxiv.org/abs/2103.00020). CLIP, the encoder of `dp_clip`, trained on
  400 million image-text pairs.
- de Haan et al., "Causal Confusion in Imitation Learning", 2019.
  [arXiv:1905.11979](https://arxiv.org/abs/1905.11979). More information can make an imitation
  policy worse: why step 5's reading was plausible.

## Check yourself

1. Measure the tolerance on cube-only layouts and compare it with the cube column of step 7.

   ```bash
   mkdir -p results/pickup
   .venv/bin/python -m sim.pickup.grasp_tolerance --task pick_cube --out results/pickup/grasp_tolerance_cube.json
   ```

2. After step 5's commands, split the episodes of the last no-history checkpoint by outcome. Are
   your failures further off, more turned, or neither?

   ```bash
   .venv/bin/python - <<'EOF'
   import glob, json, re
   import numpy as np

   paths = glob.glob("results/pickup/dp_vit_lang_reactive_cube1k/pick_cube/ckpt_*_raw/results.json")
   last = max(paths, key=lambda p: int(re.search(r"ckpt_(\d+)_", p).group(1)))
   episodes = [e for e in json.load(open(last))["per_episode"] if e["close_step"] is not None]
   for outcome in (True, False):
       rows = [e for e in episodes if e["success"] == outcome]
       if rows:
           miss = 1000 * np.median([e["close_xy_err_target"] for e in rows])
           yaw = np.degrees(np.median([e["close_yaw_err_target"] for e in rows]))
           print("successes" if outcome else "failures", len(rows), f"miss {miss:.0f} mm, yaw error {yaw:.0f} deg")
   EOF
   ```

3. Test whether your no-history cube policy keeps looking: freeze its cameras on the first frame,
   then move the cube 80 mm at control step 20. Compare both with its standard score.

   ```bash
   run=experiments/pickup/dp_vit_lang_reactive_cube1k; out=results/pickup/dp_vit_lang_reactive_cube1k/pick_cube
   PICKUP_OUT=$out/stale  sim/pickup/scripts/eval.sh $run pick_cube 100 --stale_images
   PICKUP_OUT=$out/moved  sim/pickup/scripts/eval.sh $run pick_cube 100 --disturb reach --disturb_step 20
   ```

4. Which config in `sim/pickup/configs/` keeps history but removes both other differences from
   the comparison of step 5?

5. Reproduce the last column of step 6's position table without a GPU. The first angle column of
   the rotary table has frequency 1, so under the old table its angle is the position rounded to
   bfloat16.

   ```bash
   .venv/bin/python -c "
   import torch
   t = torch.arange(2048, dtype=torch.float32).to(torch.bfloat16)
   for name, first, last in [('dp_vit_lang_reactive', 426, 440), ('dp_vit', 787, 802)]:
       print(name, t[first:last + 1].unique().numel(), 'distinct of', last - first + 1)
   "
   ```
