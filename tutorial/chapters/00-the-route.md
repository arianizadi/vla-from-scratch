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
| 6 | Make sure the instruction reaches it | the trained policy closes at the named object far above chance | [Language conditioning](51-language-conditioning.md) |
| 7 | Start without history | the previous action is not an input, or the probes show the policy does not lean on it | [History and shortcuts](52-history-and-shortcuts.md) |
| 8 | Keep training and serving identical | one sample through both paths reaches the model as the same tensors | [Train/inference skew](93-train-inference-skew.md) |
| 9 | Checkpoint, auto-resume, evaluate every checkpoint | a dead run restarts by itself; every checkpoint has a success count and a slope | [The training loop](80-the-training-loop.md), [Surviving long runs](81-surviving-long-runs.md), [How much data and training](71-how-much-data-and-training.md) |
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
as working at 93%, the plain recipe trained after the position fix (83% / 85% before it).

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
object is, 0 if the place does not depend on the scene. Measured here: three first-round
`pick_target` policies closed at the named object in 29-33% of episodes, a median of 89-110 mm
from the nearest object; second-round policies that stayed at chance ended with slopes of 0.01 to
0.17 (machine B). Where they closed did not depend on the scene: they did not reach between the
objects or average them.

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
`[batch, 1, 4 x width]`. Measured here on `pick_cube` (machine A, 200 demonstrations, 300k
samples each): pooled vectors from CLIP, a pretrained image-text encoder, 2/50 (1-14%); patch
tokens from a small vision transformer trained from scratch, 15/50 (19-44%). The encoder differs
too, so one variable is not isolated.

**6. Make sure the instruction reaches it.** With three objects on the table, only the instruction
says which one: `dp_vit`, which has no language input, scored 74% on `pick_cube` and 1/100 on
`pick_target` (machine A). `dp_vit_lang_reactive` receives the instruction in those 32 slots, in
front of the patch tokens; its final checkpoint (EMA weights) closed at the named object in 100% of
episodes (machine B, 4000 demonstrations, 5.0M samples). The input alone was not enough: with
history (step 7) or too small a budget (step 9) the same policy stayed near chance.

**7. Start without history.** *History* is the previous frame, state and action as extra inputs.
Measured here (machine B, 4000 demonstrations, 3.0M samples, both before the position fix): 2% / 2%
with history, 81% / 75% without. The history policy is not blind: at 2.5M samples (EMA weights) it
scored 1/100 as served, 22/100 when its first query carried the expert's first command as the
previous action, and 77/100 when the expert drove the first 10 control steps. It reads the target
from its previous action. In training, a window that starts at a demonstration's first frame fills
that slot with the expert's first command, which points at the target; served, it holds the
command that keeps the robot at rest. With `collect.py --hold_steps 1` every demonstration starts
with a frame of the robot at rest; trained on those with the fixed position table, the history
policy still ends at 0% / 4% after 5M samples: from the second step on the previous action still
predicts the next one. On `pick_cube`, where there is nothing to choose, history costs nothing:
trained with the fixed table the history policy ends at 96% / 98% (its earlier 62% / 70% was the
position bug).

**8. Keep training and serving identical.** *Train/inference skew* is any difference between the
tensors the model receives in training and when served; it raises no error. Measured here: a
vision-language model (VLM) trained on pixel values of 0-255 and served values of 0-1 listed the
objects of a held-out scene in the right order in 1 of 300 cases; after the fix, 256. The
previous-action slot of step 7 is another case, and an action token placed after the batch
padding a third ([Train/inference skew](93-train-inference-skew.md)).

**9. Checkpoint, auto-resume, evaluate every checkpoint.** Long runs die: machine A's runs died or
hung several times. Evaluating every checkpoint, raw and EMA, turns a run into a curve: here flat,
then steep. Measured here (machine B), `dp_vit_lang_reactive` on 4000 demonstrations, before the
position fix: 0% / 1% after 1.0M samples (slope 0.11 / 0.14), 4% / 36% after 2.0M (0.81 / 0.96),
56% / 70% after 2.5M, 83% / 85% after 5.0M; trained again with the fixed table it is at 58% after
1.5M samples and ends at 93%. The slope moves first, then success; in this transition the EMA
weights lead, though not in every run. A flat 0% says nothing about what comes next; a slope that
has left 0 does.

**10. Probe the policy that works.** A success rate on static scenes does not show what a policy's
choices depend on. Probes of the 85% policy (EMA weights): with the first frame's images sent for
the whole episode, it closed at the named object in 99% of episodes and succeeded in 1/100; with the
named object moved 80 mm at control step 20, it closed 8 mm (median) from where the object had been,
0/96. It decides where to go from its first look and then follows its own arm; nothing in its
demonstrations moves or fails. Trained with random start poses (`--random_start`), the policy
follows an object moved early in the reach (94/100); with recoveries (`--disturb_prob 0.5`) it
follows a moved object and retries (79-88%); with both, 89-94% under every disturbance
([field notes](../notes.md), "Self-correction has to be in the data").

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

The recommended first real run, as in the [README](../../README.md), reached 94% / 94% and 89-94%
with the object moved:

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

`pick_target` is solved: `dp_vit_lang_reactive` (48.9M parameters) on 4000 demonstrations and
5.0M samples, trained with the fixed position table, ends at 93% (trained weights), 96% / 96% with
other seeds; with random start poses and recoveries in the demonstrations it ends at 94% / 94%. The
table shows the route there, final checkpoints on machine B, all trained before the fix ("Where it
goes wrong"); the last row's checkpoint scored 83/100 on a second set of 100 layouts.

| Policy | Demos, samples | Success | Closed at the named object | Slope |
|---|---|---|---|---|
| `dp_vit_lang`: patch and instruction tokens, history | 4000, 3.0M (stopped) | 2% / 2% | 30% / 35% | 0.11 / 0.08 |
| `dp_vit_lang_reactive`: the same without history | 1500, 1.0M | 3% / 5% | 49% / 41% | 0.29 / 0.18 |
| same | 1500, 3.0M | 64% / 62% | 99% / 100% | 0.96 / 0.97 |
| same | 4000, 5.0M | 83% / 85% | 100% / 100% | 0.98 / 0.98 |

- History and budget were isolated on the patch-token policy. The single conditioning vector was
  not: `vla_custom` also had history and 1.0M samples. The from-scratch VLA that hands every token
  to the action head (no history, 4000 demonstrations, 3M samples, after the fix) ends at 86% / 87%,
  though history, the number of demonstrations and the position table changed with it.
- Before the fix, what failed looked like the grasp angle (T block 55%, 64 degrees off on failed
  grasps); most of it was the position bug: after the fix the T block is at 87% and the cube at 94%
  (pooled over the last three checkpoints).

Which demonstrations teach what is chapter 72, still to come. The case studies tell both tasks in
order: [`pick_cube`](100-case-pick-cube.md) and [`pick_target`](101-case-pick-target.md).

## Where it goes wrong

| What you see | Why | Check | Do |
|---|---|---|---|
| A comparison comes out equal | neither side can do the task: pretrained VLM 0/100, random-init VLM 1/100, both closing at the named object at chance (33%, 29%) | that share for both | fix what both lack, then compare again by checkpoint: with every token and no history, pretrained 86% / 87% against random-init 75% / 85%, the pretrained one looking after 0.25M samples instead of about 1.0M |
| Two evaluations of one run disagree | `eval.sh` without `PICKUP_WEIGHTS` serves the EMA weights of a run that saved them (the pickup configs switch them on); one checkpoint here scored 4% raw, 36% EMA | `server.checkpoint` in `results.json` | set `PICKUP_WEIGHTS=raw` or `ema` |
| A result belongs to another run | two evaluations on one port both reached the policy server that bound it first; about 40 results here were filed under the wrong run and redone | `server.checkpoint` against the result's directory | `eval.sh` takes a free port and `eval.py` checks which checkpoint is served |
| Nothing: the loss falls, and training agrees with serving | the rotary position table was computed in bfloat16 under autocast: 641 distinct positions out of 2048 | `tests/essential/models/test_rotary_precision.py` | fixed in `vla_foundry/positional_embedding.py`; `rotary_legacy_autocast` reproduces the old table; with the fix the policy starts to look earlier and the same recipe ends at 93% instead of 83%, a second pair of seeds at 96% / 96% instead of 90% / 89% ([Positions](12-positions.md)) |

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
   should round to 60-84%. Then set `k, n = 85, 100`: is 83/100, the same checkpoint on a second
   set of layouts, inside the interval?

   ```bash
   .venv/bin/python -c "
   k, n, z = 37, 50, 1.96
   p, d = k / n, 1 + z*z/n
   c = (p + z*z/(2*n)) / d
   h = z * (p*(1-p)/n + z*z/(4*n*n)) ** 0.5 / d
   print(round(c - h, 3), round(c + h, 3))"
   ```

3. Step 10's probes were of a policy trained on clean demonstrations. Predict what the
   recommended run, trained with recoveries, does in each, then run them (notebook: frozen cameras,
   at the named object 0%; moved at control step 20, 89/98).

   ```bash
   PICKUP_OUT=results/pickup/my_run/pick_target/stale_images \
       sim/pickup/scripts/eval.sh experiments/pickup/my_run pick_target 100 --stale_images
   PICKUP_OUT=results/pickup/my_run/pick_target/disturb_reach20 \
       sim/pickup/scripts/eval.sh experiments/pickup/my_run pick_target 100 --disturb reach --disturb_step 20
   ```
