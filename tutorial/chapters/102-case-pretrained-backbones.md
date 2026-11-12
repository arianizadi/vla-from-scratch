---
title: "Case Study: A Backbone From Scratch Against a Pretrained One"
part: 10 Case studies
order: 102
summary: Run a pretrained-against-random backbone comparison that can show a difference, read it checkpoint by checkpoint at equal training samples, and say what one seed per side supports.
status: draft
---

# Case Study: A Backbone From Scratch Against a Pretrained One

The **backbone** of a robot policy turns camera images and the instruction into tokens; the
**action head** turns those tokens into motion. A backbone can start from random weights or from a model
trained earlier on other data. This chapter asks what a pretrained backbone buys a policy that picks
up the object an instruction names. A first attempt said nothing; a second said: mainly an earlier
start.

## The idea

**Pretraining** is training a model on one task so that its weights become the starting point for
another. A **control** is a second run identical to the first except in the one thing under test,
here the backbone's initialisation: the same policy with a randomly initialised backbone.

The task is `pick_target`: three of four objects on a table, an instruction such as "pick up the
green can", success when the named object is held 8 cm up for 10 control steps. The seed fixes the
**layout** (which objects are where, which one is named, the arm's start pose). In the
demonstrations, only where the expert moves ties the words "green can" to the can ([The instruction
has to reach the action head](51-language-conditioning.md)). A backbone that has already learned to
describe these scenes might bring that tie with it.

Three measurements are read at every **checkpoint** (the training state saved at one point of a
run), on 100 layouts the policy never saw:

- **Success**, as trained / averaged weights: the optimiser's weights and a running average of them
  saved by the same run (raw / EMA in the [lab notebook](../lab-notebook.md)).
- **Closed at the named object**: the object nearest the gripper when it first closed was the named
  one. Chance is 33%.
- **Slope** of where the gripper first closed against where the named object was: 1 when it closes
  at the object, 0 when the place does not depend on the scene ([Measuring how a policy
  fails](92-measuring-failure.md)).

A worked example of why all three are needed. After 0.25M training **samples** (one sample is one
training window drawn by the data loader), the averaged weights of the pretrained policy below
succeed in 43 of 100 episodes and those of the random one in 0. Read alone, a verdict. But the
random one closes at the named object 28% of the time with a slope of -0.04: it has not started to
look yet. At the end the two are 2 points apart with averaged weights.

## The first comparison, and why it said nothing

The **from-scratch path** ([The from-scratch path](62-the-from-scratch-path.md)) trains every stage
here from random weights. Lab notebook, "The from-scratch stages", machine A (one RTX 4090, 24 GB):
a 91.4M-parameter language model on 61,000 web documents (loss 6.3 -> 4.1); a 96.5M **VLM**
(vision-language model: that language model plus a ViT, [Images become
tokens](20-images-to-tokens.md), and a projector) on 40,000 captioned sim renders, which then lists
the objects left to right in 256/300 held-out scenes; and a 130.0M **VLA** (the VLM plus a
flow-matching action head, which turns noise into a chunk of 15 actions in a few steps; [Flow
matching](41-flow-matching.md)) on 1500 demonstrations for 1.0M samples. The captions say where
objects are; the hope was that stage 3 would inherit that link.

Notebook, "`pick_target`, first round":

| | Pretrained VLM | Random-init VLM |
|---|---|---|
| Success | 0/100 | 1/100 |
| Final training loss | ~0.03 | ~0.03 |
| Closed at the named object | 33% | 29% |
| Median miss to the named object | 175 mm | 174 mm |

<!-- interactive: loss-vs-success -->
*Figure: a run's training loss next to the closed-loop success of the same checkpoints. In the
first comparison both VLAs ended at a loss of about 0.03 with 0/100 and 1/100; the loss does not say
when a policy starts to look ([Loss is not success](91-loss-is-not-success.md)).*

Both closed at chance, at places unrelated to the scene ([Measuring how a policy
fails](92-measuring-failure.md)). When both sides sit at the floor, no difference between them can
show. Three properties of the recipe, `vla_custom.yaml`, each held another policy near zero:

1. **One conditioning vector.** The **conditioning** is the tensor the backbone hands to the action
   head, `[batch, tokens, width]`. This recipe hands over one row, the hidden states of **the action
   token** appended to the prompt ([What the action head sees](50-what-the-action-head-sees.md)).
   On `pick_cube` one vector per image gave 4%, patch tokens 30%
   ([One vector or many tokens](21-pooled-vs-patch-tokens.md)).
2. **History.** With its previous action as an input, a policy can read its target there instead
   of in the image: from the second step of a demonstration on, the previous action already points
   at it ([History and the shortcut it opens](52-history-and-shortcuts.md)).
3. **The budget.** On 1500 demonstrations the patch-token policy was at 3% / 5% after 1M samples
   and 64% / 62% after 3M ([How much data, how much training](71-how-much-data-and-training.md)).

Two defects found later also applied to it: the position table computed in bfloat16, which merged
positions above 256 (the prompt held four images of 256 tokens), and the action token placed after
the batch padding, so it was read at another position in training than when served (effect not
measured). Which of these held the pair down was not isolated. The field notes ("Pretraining a
backbone: mainly an earlier start") conclude that neither policy could have found the object.

## How the second comparison was built

The rerun removes all three obstacles; its two sides differ only in initialisation.

**Every token.** `sim/pickup/configs/vla_custom_tokens_reactive.yaml` sets
`conditioning: all_tokens`. In this mode `VLMFoundryBackboneWrapper`
(`vla_foundry/models/vision_language_backbones/vlm_foundry_backbone.py`) adds no action token, and
`conditioning_from_hidden_states` (`base.py`) concatenates the last `num_vlm_layers_to_use: 4`
layers along the width (4 x 512 = 2048), moves the real tokens to the right end, zeroes the padding
and pads in front to `conditioning_len: 528`. At batch 32 the conditioning is `[32, 528, 2048]` (3
prompt tokens, 2 images x 256, an instruction of at most 8 tokens). With the noise level, the joint
state and 15 **action-step tokens** the action head's sequence is 545 tokens
(`vla_foundry/models/diffusion_policy/diffusion_policy.py`, `_build_transformer_input`).

**No history**: `image_indices: [0]`, `lowdim_past_timesteps: 0`. **Data and budget**: 4000
demonstrations (`pick_target_4k`, 263,289 windows), 3M samples, 11.4 passes over the data
(arithmetic), `lr: 3e-4` on a cosine down to 1e-5.

**The position table.** Both runs were trained after the rotary position table was moved from
bfloat16 to float32 ([Telling tokens where they are](12-positions.md)); the old table merged
positions above 256, and the prompt and the action head's sequence are both longer. Stage 1 was not
retrained: its sequences are 256 tokens, none merged. Stage 2 was (320 tokens, merged in pairs above
256): before the fix it ordered the objects right in 249/300 = 83% of held-out scenes and named the
leftmost in 279/300 = 93%; retrained, loss 0.059, 267/300 = 89% and 286/300 = 95% (notebook,
"From-scratch stages on machine B").

**The pair.** `custom_path.sh vla ... pretrained` passes
`--model.vision_language_backbone.resume_from_checkpoint` with the newest stage-2 checkpoint;
`... scratch` passes only `--model.vision_language_backbone.vlm_experiment_dir`, so
`create_vlm_foundry_backbone` builds the VLM from its `config_model.yaml` and loads no weights.

## In this repository

[`sim/pickup/scripts/custom_path.sh`](../../sim/pickup/scripts/custom_path.sh) runs one stage per
call: `data`, `llm`, `vlm`, `check`, and `vla <dataset> <run_name> <samples> [pretrained|scratch]
[config]`, which ends in `train_resilient.sh` (it restarts a run that dies or stalls). Without the
fifth argument the config is `vla_custom_tokens`, which has history.
[`vla_qwen_tokens_reactive.yaml`](../../sim/pickup/configs/vla_qwen_tokens_reactive.yaml) applies
the same conditioning to Qwen3-VL-2B (`conditioning_len: 128`, `num_trainable_vlm_layers: 4`). Like
`vla_qwen.yaml`, and unlike the other pickup policy configs, it leaves averaged weights off (their
float32 copy of the backbone takes about 9 GB, per its comment).

From the repository root:

```bash
# stages 1 and 2, once (about a quarter of an hour of training each on machine A)
sim/pickup/scripts/custom_path.sh data
sim/pickup/scripts/custom_path.sh llm
sim/pickup/scripts/custom_path.sh vlm
sim/pickup/scripts/custom_path.sh check        # held-out captioning check: run it before stage 3

# plain demonstrations, one instruction per layout (what the pair was trained on)
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k

# the pair: 12 checkpoints, one every 0.25M samples, each evaluated as it appears
for init in pretrained scratch; do
  PICKUP_NUM_CHECKPOINTS=12 sim/pickup/scripts/custom_path.sh vla pick_target_4k vla_$init 3000000 \
    $init vla_custom_tokens_reactive &
  PICKUP_WATCH=1 sim/pickup/scripts/eval_checkpoints.sh experiments/pickup/vla_$init pick_target 100 &
  wait
done
.venv/bin/python -m sim.pickup.summarize --task pick_target --runs vla_pretrained vla_scratch

# the Qwen backbone with every token: 10 checkpoints over 0.5M samples, averaged weights on
PICKUP_NUM_CHECKPOINTS=10 sim/pickup/scripts/train_resilient.sh vla_qwen_tokens_reactive \
  pick_target_4k qwen_tokens 500000 --ema.enabled True
sim/pickup/scripts/eval_checkpoints.sh experiments/pickup/qwen_tokens pick_target 100

# the first comparison's recipe (one vector, history, 1500 demonstrations, 1M samples). The code
# has changed since (position table, where the action token goes, averaged weights) and stage 2 is
# the newest one, so this will not reproduce the measured 0/100 and 1/100 exactly.
sim/pickup/scripts/make_dataset.sh pick_target 1500 pick_target_1500
sim/pickup/scripts/custom_path.sh vla pick_target_1500 vla1_pre 1000000 pretrained vla_custom
sim/pickup/scripts/custom_path.sh vla pick_target_1500 vla1_rand 1000000 scratch vla_custom
```

`summarize.py` prints success, closed at the named object and slope per checkpoint and set of
weights.

## What we measured

Machine B (L40S GPUs, 128 CPU cores), one training seed per run, 100 unseen layouts per evaluation.

### The pair, checkpoint by checkpoint

Lab notebook, "From-scratch VLA, every token, no history": its two tables side by side, without the
median misses and the 1.7M, 2.2M, 2.7M and non-final 3.0M rows. Trained / averaged weights:

| Samples | Pretrained: success | closed at named | slope | Random: success | closed at named | slope |
|---|---|---|---|---|---|---|
| 0.25M | 5% / 43% | 81% / 97% | 0.87 / 0.97 | 0% / 0% | 33% / 28% | -0.09 / -0.04 |
| 0.5M | 31% / 50% | 99% / 100% | 0.95 / 1.01 | 0% / 1% | 39% / 33% | 0.04 / 0.09 |
| 0.75M | 30% / 60% | 100% / 100% | 1.02 / 0.98 | 0% / 3% | 44% / 37% | 0.32 / 0.06 |
| 1.0M | 73% / 62% | 99% / 99% | 0.98 / 0.99 | 9% / 4% | 84% / 68% | 0.87 / 0.73 |
| 1.2M | 64% / 82% | 100% / 100% | 1.00 / 0.99 | 28% / 36% | 95% / 95% | 0.92 / 0.92 |
| 1.5M | 74% / 83% | 100% / 100% | 1.02 / 0.99 | 42% / 53% | 98% / 97% | 0.95 / 0.95 |
| 2.0M | 81% / 88% | 100% / 100% | 0.98 / 0.99 | 69% / 81% | 100% / 100% | 1.02 / 1.00 |
| 2.5M | 84% / 88% | 100% / 100% | 0.99 / 0.99 | 77% / 80% | 99% / 100% | 0.96 / 0.99 |
| 3.0M (final) | 86% / 87% | 100% / 100% | 0.99 / 0.99 | 75% / 85% | 100% / 100% | 1.01 / 1.00 |

**The start.** The pretrained backbone looks from the first checkpoint (slope 0.87 / 0.97 after
0.25M samples). The random one is at chance at the first two checkpoints (0.25M and 0.5M), starts to
move at 0.75M (slope 0.32 with the trained weights, 0.06 with the averaged ones) and looks at 1.0M
(0.87 / 0.73). In both, the slope rises first and success follows, as in the patch-token runs ([How
much data, how much training](71-how-much-data-and-training.md)).

<!-- interactive: onset -->
*Figure: success, closed at the named object and slope against training samples for both VLAs.
The pretrained curves are high at the first checkpoint; the random ones rise the same way about
three quarters of a million samples later.*

**Averaged weights** were not reliably ahead: 43% against 5% at the pretrained run's first
checkpoint, 62% against 73% at its 1.0M one, 36% against 28% at the random run's 1.2M one. Evaluate
both.

### How far one seed per side goes

The runs end at 86% / 87% and 75% / 85%. The Wilson 95% intervals (a confidence interval for a
success rate that stays sensible near 0% and 100%) around 86/100 and 75/100 are 77.9-91.5% and
65.7-82.5% (arithmetic): they overlap. With averaged weights the two end 2 points apart. With
trained weights the gap is 11, but one run's evaluations swing by about that much between
neighbouring checkpoints (73% then 64% for the pretrained run at 1.0M and 1.2M). The end points do
not rank the backbones.

The start is firmer: at 0.5M samples the runs close at the named object 99% / 100% and 39% / 33% of
the time, a gap that holds over three checkpoints. The only seed comparison is for another model:
the patch-token policy's two seeds closed at the named object 65% (trained weights; that run saved
no averaged ones) and 83% / 72% of the time after 1.0M samples, both inside their transition. A seed
moved that policy's start by less than the 0.75M samples separating this pair. An earlier start is
likely; a second seed per side would confirm it.

### Against a policy with no language model

The patch-token policy `dp_vit_lang_reactive` (48.9M parameters, 12.6M of them its instruction
embedding table) has a small ViT trained from scratch and no language model. Trained with the fixed
table on the same demonstrations (trained weights; notebook, "The same run before and after the
fix"), it starts to look at about the same point as the random VLA, slightly later at 1.0M samples
(closed at the named object 65% against 84%, slope 0.58 against 0.87). Equal samples are not equal
training: the VLA trains at batch 32 with `lr: 3e-4`, the small policy at batch 64 with the preset's
5e-4, so the VLA has taken twice as many optimiser steps; the architectures differ too, and each
side is one seed. Size alone did not bring the start forward by much; within the pair, pretraining
did.

Neither VLA ends above it: it is at 90% after the same 3M samples and 93% after 5M. Earlier in
training the pretrained VLA is ahead (73% against 8% at 1.0M): the earlier start again. The field
notes' reading: with four known objects and one form of instruction, there is nothing for a language
model to add.

### The Qwen backbone

Pretrained Qwen3-VL-2B ([Building on a pretrained VLM](61-pretrained-backbones.md)). With one
conditioning vector and history (`vla_qwen.yaml`, 1500 demonstrations, 0.4M samples) it ended at 5%,
closing at the named object 71% of the time (slope 0.68, 83 mm off): it pointed at the right object
and did not grasp it.

With every token and no history (`vla_qwen_tokens_reactive.yaml`, which trains the last 4
language-model layers and the action head), the Qwen VLA was trained on 4000 demonstrations for 0.5M
samples. Notebook, "Qwen3-VL-2B backbone", selected rows:

| Samples | Success (trained / averaged) | Closed at the named object | Slope | Median miss to named |
|---|---|---|---|---|
| 0.10M | 6% / 12% | 71% / 80% | 0.76 / 0.83 | 94 / 73 mm |
| 0.20M | 30% / 36% | 98% / 99% | 0.96 / 1.01 | 27 / 25 mm |
| 0.5M | 57% / 65% | 100% / 100% | 1.00 / 0.99 | 16 / 13 mm |
| 0.5M (final) | 70% / 72% | 98% / 99% | 0.95 / 0.99 | 15 / 14 mm |

It looks after 0.1M samples. The last two evaluations, at the same sample count, are 13 points apart
with trained weights: one final number from one seed can move by that much. Its final misses are
twice the from-scratch VLA's 7 / 6 mm; whether its coarser image tokens (49 per image of 32 x 32
pixels, against 256 of 14 x 14) or the short run are the reason was not tested, nor whether a longer
run reaches the small policy.

### What each costs

Lab notebook, "Throughput", machine B; the hours are arithmetic:

| Model | Training speed (samples/s) | Its budget here |
|---|---|---|
| `dp_vit_lang_reactive`, 48.9M | about 1100 alone; 570-700 while evaluated | 3M in about 45 min |
| `vla_custom_tokens_reactive`, 130M | about 260; about 190 while evaluated | 3M in about 3.2 h |
| `vla_qwen_tokens_reactive`, 2.2B, batch 16 | about 62; about 35 while evaluated | 0.5M in about 2.2 h |

The from-scratch VLA trains at about a quarter of the small policy's speed, the Qwen one at about an
eighteenth. Inference latency was not measured for either every-token VLA.

## What it teaches

1. **A comparison needs both sides off the floor.** Check that the recipe can work at all before
   comparing initialisations.
2. **Compare by checkpoint, at equal samples.** Only the curve shows that the difference is in when
   looking starts (the worked example in The idea).
3. **Pretraining bought mainly an earlier start**, for two short stages. In a short run that
   decides the result; in a long one it mostly does not.
4. **Change one thing per pair.** The pair isolates the initialisation. The step from the first
   round to the second changed conditioning, history, data, budget and position table together, so
   "0% to 86% with every token" is not the effect of the conditioning alone.

## Where it goes wrong

**Both sides at chance.** Closed at the named object near 33% and slope near 0 on both runs: the
comparison is at the floor. Check history, conditioning and budget first.

**The stage-2 VLM ignores its images.** It once scored 1/300 on ordering because
`SimpleVLMProcessor` scaled images differently in training and at inference ([Train and inference
must see the same thing](93-train-inference-skew.md)). Run `custom_path.sh check` first.

**The wrong dataset under the right name.** `README.md` builds `pick_target_4k` with
`--random_start --disturb_prob 0.5`, and `make_dataset.sh` replaces a dataset of the same name. The pair was trained
on plain demonstrations.

**A prompt longer than `conditioning_len`** makes `conditioning_from_hidden_states` raise a
`ValueError`.

**A finished pair still needs probing.** Neither VLA was given the probes of [Probing a policy that
works](95-what-a-policy-looks-at.md). Both learned from one start pose with nothing moving, the
conditions under which the small policy learned to look once (same chapter; what changes that: [A
policy that corrects itself](103-case-self-correction.md)).

## Background and further reading

Background, not measured here:

- Mercat et al., *VLA Foundry: A Unified Framework for Training Vision-Language-Action Models*,
  2026, [arXiv:2604.19728](https://arxiv.org/abs/2604.19728). The framework this repository builds on;
  it reports a from-scratch model and a stronger one built on Qwen3-VL.
- Brohan et al., *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*,
  2023, [arXiv:2307.15818](https://arxiv.org/abs/2307.15818). Fine-tuned web-trained VLMs for robot
  control and reported better generalisation to new objects: the premise tested here.
- Black et al., *π0: A Vision-Language-Action Flow Model for General Robot Control*, 2024,
  [arXiv:2410.24164](https://arxiv.org/abs/2410.24164). A flow-matching action model built on a
  pretrained VLM.

Here the same four objects and one form of instruction appear in every episode, where a pretrained
backbone has least to add; [Toward real tasks](110-toward-real-tasks.md) describes scenes in which
to ask again.

## Check yourself

1. From `summarize.py`'s table for your pair, find each run's first checkpoint with slope above 0.5
   and its first with success above 50%. How many samples apart are the runs at each threshold?

   ```bash
   .venv/bin/python -m sim.pickup.summarize --task pick_target --runs vla_pretrained vla_scratch
   ```

2. Show that the `scratch` side takes no weights from stages 1 and 2.

   ```bash
   grep -n "backbone=" sim/pickup/scripts/custom_path.sh
   sed -n 27,71p vla_foundry/models/diffusion_policy/__init__.py
   ```

3. Feed `conditioning_from_hidden_states` two prompts of different length; check the shape and
   where the zero rows go.

   ```bash
   .venv/bin/python -c "
   import torch
   from vla_foundry.models.vision_language_backbones.base import conditioning_from_hidden_states
   hs = [torch.randn(2, 5, 3) for _ in range(4)]
   mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
   out = conditioning_from_hidden_states(hs, mask, 'all_tokens', conditioning_len=6)
   print(out.shape); print(out.abs().sum(-1))"
   ```
