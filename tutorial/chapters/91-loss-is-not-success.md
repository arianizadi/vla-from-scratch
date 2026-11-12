---
title: Loss Is Not Success
part: 9 Evaluation and debugging
order: 91
summary: Explain why a low training loss can come with a 0% success rate and why the open-loop action error ranks a 4% policy above a 74% one, use the open-loop check for the one thing it can show, and evaluate closed-loop from the first checkpoint.
status: draft
---

# Loss Is Not Success

Training a policy produces a loss curve, and one more script gives an open-loop error: the
distance between predicted and recorded actions. Neither says whether the robot picks up the
object. In this repository a policy with a final training loss of about 0.03, as low as any
recorded on `pick_target`, succeeded in 0 of 100 episodes, and a policy that ended at 83% / 85%
(trained / averaged weights) was at 0% / 1% after its first million training samples. A robot
policy needs a third number, measured by running it at every checkpoint.

## The idea

A **window** is one training example: the observations at one frame of a demonstration plus the
15 actions the expert took from there. A **chunk** is the sequence of 15 future actions a policy
predicts in one go. A **sample** is one window drawn by the data loader. Three numbers can be
computed for a policy.

| Number | Input the policy gets | Compared with |
|---|---|---|
| **Training loss** | a recorded window, the expert's actions blended with random noise ([Flow matching](41-flow-matching.md)) | the direction from the expert's actions to the noise |
| **Open-loop error** | a recorded window | the expert's recorded actions |
| **Closed-loop success** | what the simulator shows after the policy's own earlier actions | the task's success test |

**Open-loop** means the predictions are compared with recorded data and never executed: every
input comes from a recording in which the expert was in control. **Closed-loop** means the
policy's actions move the arm and the next observation is the result
([Closed-loop evaluation](90-closed-loop-evaluation.md)).

Three mechanisms keep the first two numbers from predicting the third. The first two follow from
the expert's code and are tested under "What we measured".

1. **Most windows are easy.** Once the arm is moving, the previous action and the arm's own
   motion say what comes next: the object need not be found.
2. **The windows that decide the episode are few.** Where to go is decided at the start.
3. **Errors compound when actions are executed** (background: Ross, 2010). After a first chunk
   aimed at the wrong place, the next query comes from where no demonstration has been
   ([Imitation learning](43-imitation-learning.md)).

1500 `pick_target` demonstrations give 98,841 windows, 65.9 per demonstration, one per frame; the
expert closes the gripper around control step 40. In a 66-frame demonstration:

| Frames | The arm | What predicts the next actions |
|---|---|---|
| 0 | rests in the start pose | the cameras and the instruction (with history: see "Why most windows are easy") |
| 1 to about 40 | reaches and descends | the previous action and the joint state: by default every reach is a straight line from a nearly fixed start pose, so the arm's position soon says where it ends |
| the rest | closes the gripper once, then lifts | the previous action |

One window in 66 is 1.5% of the data. The first eight frames, 12%, are covered by the first
chunk, of whose 15 actions 8 are executed ([Chunks and execution](42-chunks-and-execution.md)).
How much of the loss each row carries was not measured.

## How it works

### What the loss averages

For `dp_vit` a batch is 64 windows (`per_gpu_batch_size: 64` in
[`sim/pickup/configs/dp_vit.yaml`](../../sim/pickup/configs/dp_vit.yaml)) and the action tensor is
`[64, 16, 8]`: one past control step, the current one and 14 future ones, 8 numbers each.
`DiffusionPolicyBatchHandler.prepare_inputs_and_targets` (`vla_foundry/models/batch_handlers.py`)
sets the target to `noise - actions`. `DiffusionPolicy.forward`
(`vla_foundry/models/diffusion_policy/diffusion_policy.py`) draws one noise level per window and
blends every slot where `future_mask` is true toward the noise by that amount. `masked_mse_loss`
(`vla_foundry/losses.py`) averages the squared error over the same slots, at most
64 x 15 x 8 = 7,680 numbers per batch.

- **Every unmasked value has the same weight.** `F.mse_loss(..., weight=mask)` divides the summed
  squared error by the number of unmasked values in the batch: the values of a frame-0 window
  count like any others, and a batch of 64 holds 64 / 65.9 = 0.97 such windows on average.
- **It has no physical unit and no common zero.** The target is in normalised action units when a
  normaliser is configured, so a loss cannot be converted to millimetres at the gripper.
  Background: squared error is minimised by the mean, so what the inputs do not determine stays
  in the loss as a floor that differs between tasks: 0.058 on `pick_cube` and 0.069 on
  `pick_target` are not on one scale.
- **It is the loss of the raw weights.** `train_one_checkpoint` (`vla_foundry/train.py`) logs the
  loss of the weights it trains, the raw weights. The **EMA** weights (an exponential moving
  average of them over optimiser steps, saved next to them by the policy configs in
  `sim/pickup/configs/`, the Qwen ones excepted) are never run in training: a checkpoint has two
  sets of weights and one logged loss.

### Why most windows are easy

`ScriptedExpert.act` ([`sim/pickup/expert.py`](../../sim/pickup/expert.py)) moves its set-point
(where it wants the gripper) by at most `speed * dt` per control step,
0.40 x 1.2 m/s x 0.05 s = 2.4 cm for the fastest waypoint, and solves for joint angles starting
from its previous joint command. Consecutive actions therefore differ by one small move, and the
gripper command changes once per episode. Background: predicting the next action from the
previous one instead of from the scene is the copycat problem (Wen, 2020).

What a policy is given decides which windows need the scene.

- **With history** (frames t-1 and t, states t-1 and t, the previous action; `dp_vit`,
  `dp_vit_lang`), a window from frame 1 on contains the expert's previous command. The frame-0
  window has none: [`preprocess.yaml`](../../sim/pickup/configs/preprocess.yaml) pads its past
  slot by copying (`padding_strategy: copy`, `PaddingStrategy.copy_edge`), and
  `DiffusionPolicy.forward` does not read `past_mask`, so its previous-action input is the
  expert's first command, which already points at the target. No window needs the cameras to
  choose which way to go.
- **Without history** (one frame per camera, the current state; the `_reactive` configs), the
  frame-0 window needs the cameras and the instruction. Later windows contain the joint state of
  an arm already moving toward its target.

A served policy never receives the expert's first command: `PickupEnv.reset` sets `last_action`
to the start joint targets with the gripper open, the **hold command**, and
[`eval.py`](../../sim/pickup/eval.py) sends that with the first query.

<!-- interactive: copycat -->
*Figure: one joint target and the gripper command of a demonstration against the control step,
each with a line repeating the previous control step's action. The joint lines stay one small
move apart; the gripper lines differ once, at the close.*

## In this repository

**The open-loop check.** [`sim/pickup/open_loop_check.py`](../../sim/pickup/open_loop_check.py),
for policies with and without history:

1. Builds `Policy(args.checkpoint_dir, None, "cuda")`, the class the policy server uses
   (`sim/pickup/policy_server.py`), with the newest saved weights, EMA if the run has them.
2. Reads `--samples` (default 64) stored windows and skips one that is padded where the policy
   reads (`window_rows`) or has no action before the current step: a 66-frame demonstration
   contributes frames 1 to 51, never frame 0. `samples:` prints how many remained.
3. Calls `policy.infer` with the recorded images, states, instruction and previous action (which a
   policy without history does not read), and
   prints three lines of mean absolute error (`q1` to `q7` in radians, `grip` in command units):
   the predicted chunk against the expert's, the first predicted action against the expert's,
   and a baseline that repeats the previous action for the whole chunk.

```bash
.venv/bin/python -m sim.pickup.open_loop_check \
    --checkpoint_dir experiments/pickup/<run> \
    --shards data/preprocessed/<dataset>/shards --samples 64
```

**What it is good for.** `Policy.infer` is what the server calls for every query, so the check
runs the serving path (JPEG decoding, `process_inputs`, `generate_actions`, denormalisation) on
inputs whose correct answer is known. A chunk error not well under the repeat baseline means a
barely trained policy or a serving path that differs from training.

The test is one-sided. A low error does not mean the policy works, or even that the images arrive
correctly: every window it uses carries the expert's previous action, which a policy with history
can score well from alone, and none is the first query of an episode
([Train/inference skew](93-train-inference-skew.md) compares input tensors directly).

**Closed-loop evaluation from the first checkpoint.**
[`sim/pickup/scripts/eval_checkpoints.sh`](../../sim/pickup/scripts/eval_checkpoints.sh) evaluates
every checkpoint of a run, newest first, with its raw weights and, if saved, its EMA weights,
skipping those that have a `results.json`. With `PICKUP_WATCH=1` it follows a run that is still
training. The plain recipe of the transition table below (the README's recommended run adds
`--random_start --disturb_prob 0.5` to the first line), with a checkpoint every 0.5M samples as on
machine B (`PICKUP_NUM_CHECKPOINTS`, default 6); the config switches averaged weights on itself:

```bash
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k
PICKUP_NUM_CHECKPOINTS=10 sim/pickup/scripts/train_resilient.sh dp_vit_lang_reactive pick_target_4k my_run 5000000 &
PICKUP_WATCH=1 PICKUP_EVAL_JOBS=2 sim/pickup/scripts/eval_checkpoints.sh experiments/pickup/my_run pick_target 100 &
# at any time after the first checkpoint has been evaluated:
.venv/bin/python -m sim.pickup.summarize --task pick_target --runs my_run
```

[`summarize.py`](../../sim/pickup/summarize.py) prints one row per evaluation: the samples seen,
success with its 95% interval, and where the gripper first closed
([Measuring failure](92-measuring-failure.md)), including:

- **closed at the named object**: how often the object nearest the gripper at that moment was the
  one the instruction named (chance: 33%);
- **slope**: the least-squares slope of where it closed against where the named object was, 1 if
  it closes where the object is, 0 if where it closes does not depend on the scene.

## What we measured

All numbers are from the [lab notebook](../lab-notebook.md), one training seed per run,
closed-loop on **layouts** the policy never saw unless stated (a layout: the objects on the
table, their poses, the named object and the arm's start pose, fixed by the seed).

**Across policies**, first round, on machine A (one RTX 4090, 24 GB).

| Policy, training samples | Final loss | Success |
|---|---|---|
| `pick_cube`: pooled CLIP, 200 demos, 300k | 0.040 | 2/50 = 4% |
| `pick_cube`: `dp_vit`, 200 demos, 300k | 0.091 | 15/50 = 30% |
| `pick_cube`: `dp_vit`, 1000 demos, 1.5M | 0.058 | 37/50 = 74% |
| `pick_target`: `dp_vit`, 1500 demos, 1.0M | 0.069 | 1/100 |
| `pick_target`: `vla_qwen`, 1500 demos, 400k | 0.03-0.04 | 6/100 |
| `pick_target`: `vla_custom`, 1500 demos, 1.0M | ~0.03 | 0/100 |
| same, VLM randomly initialised | ~0.03 | 1/100 |

The lowest loss on `pick_cube` (0.040) belongs to the 4% policy, which scored 0/20 even on its
own training layouts. On `pick_target`, losses of about 0.03 belong to policies at 0/100 and
1/100. A loss cannot be read as a level of success or used to rank models. The `pick_target` rows do not show what those designs can reach: all had history,
the VLAs passed one conditioning vector to the action head, none was trained past 1.0M samples on
1500 demonstrations, and all predate the position fix below. With every token handed to the
action head, no history and 4000 demonstrations, the from-scratch VLA ends at 86% / 87% after 3M
samples (75% / 85% with a randomly initialised VLM) and the Qwen VLA at 70% / 72% after 0.5M
samples ([The from-scratch path](62-the-from-scratch-path.md)).

**The late transition.** The strongest case is inside one run: `dp_vit_lang_reactive` (the
instruction as tokens, no history) on 4000 demonstrations, machine B (L40S GPUs, 128 CPU cores),
100 layouts per checkpoint, raw / EMA weights:

| Samples | Success | Slope |
|---|---|---|
| 0.5M | 0% / 0% | 0.02 / -0.06 |
| 1.0M | 0% / 1% | 0.11 / 0.14 |
| 1.5M | 4% / 2% | 0.47 / 0.37 |
| 2.0M | 4% / 36% | 0.81 / 0.96 |
| 2.5M | 56% / 70% | 0.94 / 0.98 |
| 3.0M | 81% / 75% | 0.98 / 0.97 |
| 5.0M (final) | 83% / 85% | 0.98 / 0.98 |

<!-- interactive: loss-vs-success -->
*Figure: the success (raw and EMA weights) and slope of each checkpoint of this run against
training samples, next to the training loss from the run's log (not a notebook number).*

For a million samples nothing shows; between 1.5M and 2.5M the policy goes
from not using the scene to using it, the slope moving first ([field notes](../notes.md),
"Looking at the scene arrives late, and all at once"). On 1500 demonstrations the same policy was
at 0-3% up to 1.2M samples and at 64% / 62% after 3.0M; given a budget of 1.0M samples it ended
at 3% / 5% and read as a design that does not work.

The notebook records no loss for these checkpoints; the field notes ("What did not work") put it
as "the loss is flat while success goes from 0% to 80%". At 2.0M samples one checkpoint holds
trained weights that succeed in 4% of the layouts and averaged weights that succeed in 36%, with one
logged loss between them (the average leads in this transition, not in every run). The loss also
levels off at the end of any budget, where the learning rate reaches almost zero
([The training loop](80-the-training-loop.md)); the 1.0M-sample run ended that way, inside the
transition. What announced the transition was a closed-loop number: a slope of 0.47 / 0.37 at
1.5M samples, with success still at 4% / 2%.

The table predates a fix: the rotary position table had been computed in bfloat16
([Positions](12-positions.md)). The same recipe after the fix (trained weights, one seed) was at
8% after 1.0M samples, 58% after 1.5M and 93% at the end. Windows that need the scene bring the
transition forward, as mechanisms 1 and 2 predict: with the windows that start in frames 0-7 present 6
times instead of once (`sim/pickup/boost_early_windows.py`) it closed at the named object in 99%
of the layouts after 1.0M samples against 65%, and with every episode starting from a random pose
in 91% (field notes, "What the demonstrations contain"; [Demonstrations](70-demonstrations.md)).
Do not judge a design on a budget that ends before the transition of a comparable run
([How much data and training](71-how-much-data-and-training.md),
[the `pick_target` case study](101-case-pick-target.md)).

**Open-loop error.** On recorded windows the four `pick_cube` policies reproduce the expert's
chunks to within 0.01-0.04 rad per joint (copying the previous action: 0.05-0.2 rad); the 4%
policy has the lowest error of the four, below the 74% one.

**The first previous action.** `dp_vit_lang`, the same policy with history, was at 2% / 2% after
3.0M samples on the 4000 demonstrations, where the run was stopped. Its 2.5M-sample checkpoint
(EMA), 100 layouts each: as served, the first query carrying the hold command as previous action,
1/100, closing at the named object in 30% of the layouts. With the expert's first command in that
place (`--first_prev_action expert`): 22/100 and 97%. With the expert driving the first 10
control steps (`--expert_prefix 10`): 77/100 and 100%. The policy learned the grasp and reads
where to go from its previous action, which every window of the open-loop check supplies and the
first query of an episode does not ([History and shortcuts](52-history-and-shortcuts.md)). These
runs predate the position fix.

**The first chunk.** An open-loop test aimed at the deciding window does separate a policy from
the expert. Given the first frame of an episode (60 layouts), the first chunk of
`dp_vit_lang_reactive` trained on 1500 demonstrations (the 1M-sample run) at 0.9M samples, trained
weights, at the start of its transition, headed for the named object in 35% of training layouts
and 48% of unseen ones (the expert's: 100% of both). Its end point was 12.1 cm and 11.2 cm
(median) from the expert's; with another object of the layout named instead, its end point moved
10.9 cm and 10.8 cm (median, one draw per instruction), about as far as it was from the expert's,
and the chunk then headed for the newly named object in 42% and 43% of the layouts. No script for
this test is in the repository.

## Where it goes wrong

- **Loss falling or flat, success at 0-1%.** The policy may not have started to look. Read the
  slope per checkpoint in `summarize` (near 0: not looking; near 1 with misses: imprecise) and
  train past the point where a comparable run started to look.
- **A policy with history at 0-3%, open-loop error low.** It reads the target from its previous
  action, which the first query lacks: `--first_prev_action expert` and `--expert_prefix 10` score
  far better than as served. Start without history. `collect.py --hold_steps 1`, which starts
  every demonstration with a frame at rest, did not help: a history policy trained on such
  demonstrations, with the position table fixed, ends at 0% / 4% after 5M samples, and where it
  closes never depends on the scene.
- **High success read as "it watches the scene".** Sent the first frame's images for the whole
  episode (`--stale_images`), the 85% policy still closed at the named object in 99% of the
  layouts, and succeeded in 1/100: where nothing moves, one look is enough
  ([What a policy looks at](95-what-a-policy-looks-at.md)).
- **`open_loop_check` stops with "no window without padding among the samples read".** Every
  window it read was padded where the policy reads: raise `--samples`.

## Background and further reading

- Ross, *Efficient Reductions for Imitation Learning*, 2010
  ([PMLR](https://proceedings.mlr.press/v9/ross10a.html)). The learner's actions change the
  states it later meets, so errors compound.
- Codevilla, *On Offline Evaluation of Vision-based Driving Models*, 2018
  ([arXiv:1809.04843](https://arxiv.org/abs/1809.04843)). For driving policies, offline prediction
  error alone does not indicate driving quality.
- Wen, *Fighting Copycat Agents in Behavioral Cloning from Observation Histories*, 2020
  ([arXiv:2010.14876](https://arxiv.org/abs/2010.14876)). The shortcut of copying the expert's
  previous action.

## Check yourself

1. `vla_custom` reached a loss of about 0.03 and 0/100. Explain with the frame table how both can
   hold, and name the input that held the answer in its frame-0 windows.
2. Run the open-loop check on a run with history. Say what a chunk error (first line) close to
   the repeat baseline (third line) would mean, and what one far below it does not mean. Then
   evaluate the run as served and with the expert's first command: which does the check resemble?

   ```bash
   PICKUP_OUT=results/pickup/<run>/pick_target/served \
     sim/pickup/scripts/eval.sh experiments/pickup/<run> pick_target 100
   PICKUP_OUT=results/pickup/<run>/pick_target/first_expert \
     sim/pickup/scripts/eval.sh experiments/pickup/<run> pick_target 100 --first_prev_action expert
   ```
3. Train and evaluate the recipe (commands above). Find the first checkpoint with a slope above
   0.5 in the `summarize` rows, then open the run's page in the monitor, which draws loss and
   success against training samples, and say what the loss alone told you one checkpoint earlier:

   ```bash
   .venv/bin/python -m sim.pickup.monitor_server   # then open the URL it prints
   ```
