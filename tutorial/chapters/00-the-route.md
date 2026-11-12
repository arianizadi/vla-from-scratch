---
title: The Route on One Page
part: 0 Orientation
order: 00
summary: Follow the ten-step order of work that takes a robot policy to a measured closed-loop success rate, and say where this repository stands on it.
status: draft
---

# The Route on One Page

A robot policy is a function from camera images, joint state and an instruction to motor commands.
Its training loss can fall whether or not the robot can do the task, so the work needs an order in
which each step makes the next result interpretable. This page is that order, and where this
repository stands on it.

## The checklist

*Closed-loop* means the policy's own actions produce the next observation. *Serving* a policy
means running it behind a policy server that answers the simulator. A *layout* is which objects
are on the table, where they are, which one the instruction names and the arm's start pose; a
seed fixes it. A *scripted expert* is a hand-written controller that reads object positions from
the simulator; a *demonstration* is one recorded episode of it. A *window* is one training
example: the observations at one frame plus the actions that follow. A *sample* is one window
drawn by the data loader. A *token* is one vector in the sequence a transformer processes; an
image becomes tokens by being cut into square *patches*. The *action head* is the part of the
policy that outputs actions; its *conditioning* is the tensor `[batch, tokens, width]` handed to
it by the backbone, the network that encodes images and text. A *checkpoint* is the training
state saved at one point: weights, optimiser state and position in the data.

| # | Step | Done when | Chapter |
|---|---|---|---|
| 1 | Verify the task with a scripted expert | the expert succeeds on nearly every layout | [Demonstrations](70-demonstrations.md) |
| 2 | Judge by closed-loop success | every claim is a success count on unseen layouts, with an interval | [Closed-loop evaluation](90-closed-loop-evaluation.md), [Loss is not success](91-loss-is-not-success.md) |
| 3 | Measure how a policy fails | you know where the gripper closed and whether that depends on the scene | [Measuring failure](92-measuring-failure.md) |
| 4 | Know the task tolerance | you know how many millimetres of error each object forgives | [Measuring failure](92-measuring-failure.md) |
| 5 | Give the action head spatial tokens | its conditioning has one token per image patch | [Pooled vs patch tokens](21-pooled-vs-patch-tokens.md) |
| 6 | Make sure the instruction reaches it | the trained policy closes at the named object (step 3) far above chance | [Language conditioning](51-language-conditioning.md) |
| 7 | Start without history | the previous action is not an input, or the probes show the policy does not lean on it | [History and shortcuts](52-history-and-shortcuts.md) |
| 8 | Keep training and serving identical | one sample through both paths reaches the model as the same tensors | [Train/inference skew](93-train-inference-skew.md) |
| 9 | Checkpoint, auto-resume, evaluate every checkpoint | a dead run restarts by itself; every checkpoint has a success count and a slope (step 3) | [The training loop](80-the-training-loop.md), [Surviving long runs](81-surviving-long-runs.md), [How much data and training](71-how-much-data-and-training.md) |
| 10 | Probe the policy that works | you know what it does when the cameras freeze and when the object moves | [What a policy looks at](95-what-a-policy-looks-at.md) |

Steps 1 and 4 need no trained policy; 5 to 8 are settled before a long run, 9 during it, 10 after
it. The [debugging flowchart](94-debugging-flowchart.md) walks the list from symptom to check.

Numbers are from the [lab notebook](../lab-notebook.md): one training run each and 100 unseen
layouts unless stated, so a few percent is noise. The two computers are machine A (one RTX 4090,
24 GB) and machine B (L40S GPUs, 128 CPU cores). A pair such as `3% / 5%` is for the raw weights
(the trained ones) and the EMA weights (their exponential moving average over optimiser steps).

## What "works" means

A policy **works** on a task when its closed-loop success rate, on layouts it never saw during
training, clears a stated threshold, and the rate is reported with a confidence interval.

- **Success** is a test on the simulator's state: the named object is more than 8 cm above its
  rest height for 10 consecutive control steps, within 200 (`LIFT_HEIGHT` and `HOLD_STEPS` in
  `sim/pickup/env.py`).
- **Unseen layouts** come from seeds the demonstrations never used: evaluation starts at seed
  1,000,000.
- **An interval** is the range of true success rates consistent with the count at 95%
  confidence. Measured here: the patch-token policy `dp_vit` on `pick_cube` (1000 demonstrations,
  1.5M samples, machine A) succeeded in 37 of 50 episodes, 74%, Wilson interval 60-84%.

The threshold is a decision, not a measurement. The [roadmap](../roadmap.md) counts `pick_target`
as working at 93%: `dp_vit_lang_reactive`, 4000 demonstrations, 5.0M samples, trained weights,
machine B, after the position fix (token positions had been computed in half precision, merging
neighbouring ones; [Positions](12-positions.md)).

## The steps in order

**1. Verify the task with a scripted expert.** If the expert fails through the evaluation loop,
the simulator, the success test or the task is broken. Measured here: the expert scored 100/100 on
`pick_target` on machine B, closing the gripper 5 mm (median) from the object, and 99/100 on
machine A. The first round of policies
trained on that task (machine A) ended between 0/100 and 6/100, so the fault was in the policies.

**2. Judge by closed-loop success.** Training loss measures how well the model reproduces expert
actions on expert observations. Measured here on machine A, final checkpoints: `dp_vit` ended at
loss 0.058 with its 37/50 on `pick_cube`. On `pick_target` (1500 demonstrations, 1.0M samples) the
same policy ended at 0.069 and 1/100, and `vla_custom`, a vision-language-action policy (VLA)
trained from scratch, at ~0.03 and 0/100. The lowest loss belongs to the policy that never
succeeded.

**3. Measure how a policy fails.** A success rate of 1% does not say where the gripper went; the
record of where it first closed does, in two numbers. *Closed at the named object*: the object
nearest the gripper was the one the instruction named; with three objects, chance is 33%. The
*slope* of where the gripper closed against where the named object was: 1 if it closes where the
object is, 0 if the place does not depend on the scene. Together they pick the next step. A slope
near 0 says the policy is not using the scene: look at its inputs (steps 5 to 8) or its budget
(step 9), not at grasp precision. A slope near 1 with low success says it looks but misses: compare
the miss with the tolerance of step 4. Measured here: the failing first-round `pick_target`
policies closed at the named object at chance, and where they closed did not depend on the scene;
they did not reach between the objects or average them ([Measuring failure](92-measuring-failure.md)).

**4. Know the task tolerance.** The tolerance is the placement error at which a grasp still
succeeds; it turns a success target into a precision target. Measured here with a fixed sideways
offset added to the expert's grasp point, 320 episodes per offset: at 10 mm the cube stays at 99%
on both machines and the can drops to 60% on machine A and 94% on machine B, whose MuJoCo builds
differ (not isolated as the cause). Measure the tolerance where you evaluate.

**5. Give the action head spatial tokens.** A pooled vector is one embedding for a whole image;
patch tokens keep where things are. Worked example from
`sim/pickup/configs/dp_vit_lang_reactive.yaml`: two cameras give 2 images; each 224-pixel image
cut into 16-pixel patches gives 14 x 14 = 196 tokens; with 32 slots for the instruction's tokens
the conditioning is `[batch, 424, 256]`. The two VLA recipes (`vla_custom.yaml`, `vla_qwen.yaml`)
add one *action token* to the prompt and hand over only its hidden states from the last 4 layers:
`[batch, 1, 4 x width]`; `conditioning: all_tokens` hands over every token instead. Measured
here on `pick_cube` (200 demonstrations, 300k samples, machine A): pooled vectors from CLIP (a
pretrained image-text encoder) 2/50 = 4% against patch tokens 15/50 = 30%, with the encoder
differing too ([Pooled vs patch tokens](21-pooled-vs-patch-tokens.md)).

**6. Make sure the instruction reaches it.** With three objects on the table, only the instruction
says which one: `dp_vit`, which has no language input, scored 74% on `pick_cube` and 1/100 on
`pick_target` (machine A). `dp_vit_lang_reactive` receives the instruction in those 32 slots, in
front of the patch tokens; its final checkpoint (EMA weights) closed at the named object in 100% of
episodes (machine B, 4000 demonstrations, 5.0M samples). The input alone was not enough: with
history (`dp_vit_lang`, step 7) or too small a budget (step 9) the policy stayed near chance.

**7. Start without history.** *History* is the previous frame, state and action as extra inputs.
The previous action is the risky one: it is close to the next action, so a policy can lean on it
instead of on the cameras, and the loss will not show it. Start without it (two config keys,
`image_indices: [0]` and `lowdim_past_timesteps: 0`) and add history only together with the probes
that show what the policy does with it (a *probe* changes one input during an episode:
`eval.py --first_prev_action expert`, `--expert_prefix 10`). Measured here: on three objects the
policy with history stayed near chance, read the target from its previous action, and was not cured by a frame of the
robot at rest at the start of every demonstration; on `pick_cube`, with nothing to choose, history
costs nothing ([History and shortcuts](52-history-and-shortcuts.md)).

**8. Keep training and serving identical.** *Train/inference skew* is any difference between the
tensors the model receives in training and when served; it raises no error. Measured here: a
vision-language model (VLM) trained on pixel values of 0-255 and served values of 0-1 listed the
objects of a held-out scene in the right order in 1 of 300 cases; after the fix, 256. The
previous-action slot of step 7 is another case, and an action token placed after the batch
padding a third ([Train/inference skew](93-train-inference-skew.md)).

**9. Checkpoint, auto-resume, evaluate every checkpoint.** Long runs die: machine A's died or
hung several times. `train_resilient.sh` restarts a run from its newest checkpoint when the trainer
exits or its log stalls; `eval_checkpoints.sh` evaluates every checkpoint, raw and EMA, skips those
already evaluated, and can run next to training. That turns a run into a curve, and here
the curve was flat, then steep: before the position fix,
`dp_vit_lang_reactive` on 4000 demonstrations was at 0% after 1.0M samples and at 83% / 85% after
5.0M ([How much data and training](71-how-much-data-and-training.md)). The slope moves first, then
success. A flat 0% says nothing about what comes next; a slope that has left 0 does.

**10. Probe the policy that works.** A success rate on static scenes does not show what a policy's
choices depend on. The probes here: the cameras freeze on the first
frame, the expert drives the first steps toward another object, or the named object is moved.
Measured here: trained on clean demonstrations from one start pose, the policy looks once and then
follows its own arm ([What a policy looks at](95-what-a-policy-looks-at.md)); random start poses and
recoveries (demonstrations where the object is moved once and the expert replans) make it
follow a moved object and try again
([Self-correction](103-case-self-correction.md)).

## In this repository

<!-- interactive: pipeline -->
*Figure: the loop the checklist runs on. The scripted expert records demonstrations; they become
shards (tar files of 256 training windows); `vla_foundry/main.py` trains on them and writes
checkpoints; a policy server loads one checkpoint's weights and answers the simulator, which
counts successes.*

| Step | Code | How to run it, or which key controls it |
|---|---|---|
| 1, 2 | [`expert.py`](../../sim/pickup/expert.py) `ScriptedExpert`; [`eval.py`](../../sim/pickup/eval.py) `main`, with `--expert` for the expert; [`env.py`](../../sim/pickup/env.py) `PickupEnv.step` | "Check yourself", item 1; `sim/pickup/scripts/eval.sh experiments/pickup/<run> pick_target 100` |
| 3 | [`diagnostics.py`](../../sim/pickup/diagnostics.py) `GraspProbe`: the gripper and object positions at the first action with `action[7]` below 0.5; [`summarize.py`](../../sim/pickup/summarize.py) `close_slope` | `.venv/bin/python -m sim.pickup.summarize --task pick_target`; `.venv/bin/python -m sim.pickup.grasp_map results/pickup/<run>/<task>/results.json` |
| 4 | [`grasp_tolerance.py`](../../sim/pickup/grasp_tolerance.py) | `mkdir -p results/pickup && .venv/bin/python -m sim.pickup.grasp_tolerance --out results/pickup/grasp_tolerance.json` |
| 5, 6 | [`vit_backbone.py`](../../vla_foundry/models/vision_language_backbones/vit_backbone.py) `ViTBackboneWrapper` and its `_embed_text` | `vision_language_backbone.type: vit_backbone`, `text_tokens: 32`, `text_vocab_size: 49280`, `processor: simple_vlm`; for a VLM backbone, `conditioning: all_tokens` (`vla_custom_tokens_reactive.yaml`) |
| 7 | `dp_vit_lang_reactive.yaml` | `image_indices: [0]`, `lowdim_past_timesteps: 0`; the probes: `eval.py --first_prev_action expert`, `--expert_prefix 10` |
| 8 | [`policy_server.py`](../../sim/pickup/policy_server.py) `Policy.infer` calls `process_inputs` and `add_action_and_proprioception_fields` of `RoboticsProcessor`, as the training pipeline does | the regression tests below |
| 9 | `sim/pickup/scripts/train_resilient.sh` (restarts a dead or stalled run from its newest checkpoint), `eval_checkpoints.sh` | `PICKUP_STALL_SECONDS`, `PICKUP_NUM_CHECKPOINTS`, `PICKUP_WATCH=1`, `ema.enabled` (on in the pickup configs) |
| 10 | `eval.py`; [`disturb.py`](../../sim/pickup/disturb.py) `Disturbance` | `--stale_images`; `--expert_prefix 25 --prefix_target distractor`; `--disturb reach --disturb_step 20`; `--disturb grasp` |

The recommended first real run, as in the [README](../../README.md), adds random start poses and
recoveries to the demonstrations; it reached 94% / 94%, and 89-94% with the object moved
([Self-correction](103-case-self-correction.md)):

```bash
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k --random_start --disturb_prob 0.5
# 6 checkpoints by default; PICKUP_NUM_CHECKPOINTS=10 in front saves one about every 0.5M samples
sim/pickup/scripts/train_resilient.sh dp_vit_lang_reactive pick_target_4k my_run 5000000
sim/pickup/scripts/eval_checkpoints.sh experiments/pickup/my_run pick_target
.venv/bin/python -m sim.pickup.summarize --task pick_target --runs my_run
```

Four of the bugs in the lab notebook's table have a regression test:

```bash
.venv/bin/python -m pytest \
    tests/essential/models/test_simple_vlm_processor_scaling.py \
    tests/essential/models/test_vlm_foundry_backbone_conditioning.py \
    tests/essential/data/test_proprioception_past_steps.py \
    tests/essential/models/test_rotary_precision.py
```

None yet feeds one sample through both the training pipeline and `Policy.infer`, as step 8 asks.

## What we measured: where `pick_target` stands

`pick_target` works, as defined above (93% after the position fix), with `dp_vit_lang_reactive`
(patch tokens, instruction tokens, no history; 48.9M parameters) on 4000 demonstrations and 5.0M
samples. Each problem on the way was caught by a step of the checklist, mostly in runs whose loss
looked fine:

| What held the policies back | What catches it | Step | Told in full |
|---|---|---|---|
| the policies did not look at the scene | where the gripper closed did not depend on where the objects were | 3 | [Measuring failure](92-measuring-failure.md) |
| one vector for a whole image, or for a whole VLM prompt | comparing the conditioning on the same data: patch tokens, or every token of the VLM, did far better | 5 | [Pooled vs patch tokens](21-pooled-vs-patch-tokens.md), [What the action head sees](50-what-the-action-head-sees.md) |
| the previous action as an input | the probes of step 7 | 7 | [History and shortcuts](52-history-and-shortcuts.md) |
| a run stopped inside the flat stretch | evaluating every checkpoint: the slope had started to move | 9 | [How much data and training](71-how-much-data-and-training.md) |
| pixel range, action-token position, a zero read as "unset" | comparing what training and serving feed the model | 8 | [Train/inference skew](93-train-inference-skew.md) |
| positions merged by a half-precision table | only a test of the table itself; training and serving agreed | 8 | [Positions](12-positions.md) |
| demonstrations in which nothing moves and the arm always starts at home | the probes of step 10 | 10 | [Which demonstrations teach](72-which-demonstrations-teach.md), [Self-correction](103-case-self-correction.md) |

The single conditioning vector of the VLA recipes was tested only together with history and a short
budget, so that row is not isolated. Before the position fix, the remaining failures were
grasp-angle errors on the T block (74/138 = 54%) and the cube; most of that was the position table:
after it, the T block reached 60/69 = 87% (last three checkpoints pooled). The case studies tell
both tasks in order:
[`pick_cube`](100-case-pick-cube.md) and [`pick_target`](101-case-pick-target.md); pretraining the
backbone has its own ([Pretrained backbones](102-case-pretrained-backbones.md)).

## Where it goes wrong

| What you see | Why | Check | Do |
|---|---|---|---|
| A comparison comes out equal | neither side can do the task: pretrained VLM 0/100, random-init VLM 1/100, both closing at the named object at chance | that share and the slope for both | fix what both lack, then compare again checkpoint by checkpoint; here the pretrained backbone mainly started to look earlier ([Pretrained backbones](102-case-pretrained-backbones.md)) |
| A change seems to do nothing | the run ended in the flat stretch before the policy starts to look | the slope of the last checkpoints | train longer before judging the change (step 9) |
| Two evaluations of one run disagree | `eval.sh` without `PICKUP_WEIGHTS` serves the EMA weights of a run that saved them (the pickup configs switch them on); one checkpoint here scored 4% raw, 36% EMA | `server.checkpoint` in `results.json` | set `PICKUP_WEIGHTS=raw` or `ema` |
| A result belongs to another run | two evaluations on one port both reached the policy server that bound it first; about 40 results here were filed under the wrong run and redone | `server.checkpoint` against the result's directory | `eval.sh` takes a free port and `eval.py` checks which checkpoint is served |
| Nothing: the loss falls, and training agrees with serving | the rotary position table was computed in bfloat16 under autocast: 641 distinct positions out of 2048 | `tests/essential/models/test_rotary_precision.py` | fixed in `vla_foundry/positional_embedding.py`; `rotary_legacy_autocast` reproduces the old table; with the fix the policy starts to look earlier and ends higher ([Positions](12-positions.md)) |

## Background and further reading

Background, not measured here:

- Ross, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*,
  2010. [arXiv:1011.0686](https://arxiv.org/abs/1011.0686). A policy's actions change what it
  sees next, so test inputs drift from training inputs: why step 2 distrusts training loss.
- Kress-Gazit, *Robot Learning as an Empirical Science: Best Practices for Policy Evaluation*, 2024.
  [arXiv:2409.09491](https://arxiv.org/abs/2409.09491). Recommends reporting evaluation
  conditions, statistics and failure modes with the success rate.
- Wilson, *Probable Inference, the Law of Succession, and Statistical Inference*, 1927.
  [doi:10.1080/01621459.1927.10502953](https://doi.org/10.1080/01621459.1927.10502953). The
  interval for a proportion that `eval.py` reports.

## Check yourself

1. Run the expert through the evaluation loop and open `results.json`. Which fields tell you the
   task is sound, and which tell you how precisely the expert grasps?

   ```bash
   .venv/bin/python -m sim.pickup.eval --task pick_target --episodes 100 --expert \
       --out results/pickup/expert/pick_target
   ```

2. Reproduce the interval for 37 successes in 50 episodes with the formula in `eval.py`. The result
   should round to 60-84%. Then set `k, n = 85, 100`, `dp_vit_lang_reactive`'s EMA weights after
   5.0M samples, before the position fix (step 9): is 83/100, the same checkpoint on a second set
   of 100 layouts, inside the interval?

   ```bash
   .venv/bin/python -c "
   k, n, z = 37, 50, 1.96
   p, d = k / n, 1 + z*z/n
   c = (p + z*z/(2*n)) / d
   h = z * (p*(1-p)/n + z*z/(4*n*n)) ** 0.5 / d
   print(round(c - h, 3), round(c + h, 3))"
   ```

3. Step 10's probes were first run on a policy trained on clean demonstrations. Predict what the
   recommended run, trained with random start poses and recoveries, does with frozen cameras and
   with the object moved at control step 20, then run both and compare with the table in
   [Self-correction](103-case-self-correction.md).

   ```bash
   PICKUP_OUT=results/pickup/my_run/pick_target/stale_images \
       sim/pickup/scripts/eval.sh experiments/pickup/my_run pick_target 100 --stale_images
   PICKUP_OUT=results/pickup/my_run/pick_target/disturb_reach20 \
       sim/pickup/scripts/eval.sh experiments/pickup/my_run pick_target 100 --disturb reach --disturb_step 20
   ```
