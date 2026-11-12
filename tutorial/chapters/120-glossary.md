---
title: Glossary
part: 12 Appendix
order: 120
summary: Look up any term the chapters use, with its meaning in this repository and the chapter that introduces it.
status: draft
---

# Glossary

Every term the chapters define, in alphabetical order. Each entry gives the meaning the chapters
use and links the chapter that introduces the term; "also" names chapters that explain it further.
Numbers in an entry are properties of this repository's code and configs; measurements are in the
[lab notebook](../lab-notebook.md).

A few words are easy to confuse, and the chapters keep them apart:

- *The action token* is one special token appended to a VLM prompt; *action-step tokens* are the
  per-action tokens inside the action head. Neither is called a "probe token".
- A *probe* is always an evaluation (`GraspProbe`, or a probe of a policy such as frozen cameras).
- A *sample* is one training window drawn by the data loader; one output of the model is a *draw*.
- *Step* always says which: an *optimiser step*, a *control step*, a *flow step* or a *generation
  step*.
- *Open-loop* means predictions compared with recorded data and never executed. The stretch
  between two re-plans is not called open-loop.

## A

**Action.** The command applied for one control step. In the pick-up simulator it is 8 numbers: 7
joint position targets and a gripper command (1 = open, 0 = closed).
[40 What an Action Is](40-action-spaces.md).

**Action chunk.** See *chunk*.

**Action head.** The network that turns the conditioning into actions. Here it is a second,
non-causal transformer that receives the noise level, the conditioning, the joint state and a noisy
chunk, and predicts how to remove the noise. [03 The Model Families Today](03-the-landscape.md);
also [50](50-what-the-action-head-sees.md), [60](60-anatomy-of-a-vla.md).

**Action-step tokens.** The tokens of the action head's own sequence that stand for actions, one
per action of the chunk (15), preceded, in a policy with history, by the previous action. Not to be
confused with *the action token*. [03 The Model Families Today](03-the-landscape.md); also
[42](42-chunks-and-execution.md), [50](50-what-the-action-head-sees.md).

**Action token (the).** The single special token appended to a VLM prompt whose hidden states
condition the action head (`_action_token_id` in the backbone wrappers). Both wrappers insert it
directly after the prompt's last real token (`insert_after_last_real_token` in
`vla_foundry/models/vision_language_backbones/base.py`); it used to sit after the batch padding.
[03 The Model Families Today](03-the-landscape.md); also [10](10-tokens.md),
[50](50-what-the-action-head-sees.md), [60](60-anatomy-of-a-vla.md).

**AdamW.** The optimiser used here: Adam keeps running averages of each weight's gradient and of
its square and divides the first by the square root of the second; AdamW applies weight decay to
the weights directly. [13 Training a Language Model](13-training-a-language-model.md).

**All tokens (`conditioning: all_tokens`).** The setting of both VLM backbones that hands the
action head the hidden states of every image and text token instead of the action token's alone.
[50 What the Action Head Sees](50-what-the-action-head-sees.md); also
[60](60-anatomy-of-a-vla.md), [61](61-pretrained-backbones.md), [102](102-case-pretrained-backbones.md).

**Anchor.** The current frame of a training window, the one the window is cut around.
[42 Action Chunks and Closed-Loop Execution](42-chunks-and-execution.md); also
[70](70-demonstrations.md).

**As-served baseline.** A policy evaluated exactly as it is served, the reference a probe is read
against. [52 History and the Shortcut It Opens](52-history-and-shortcuts.md).

**Attention.** The operation that lets each position read from others: a query is scored against
every key with a dot product, and the values are mixed with the softmax of those scores as weights.
[11 The Transformer Block](11-the-transformer-block.md).

**Attention mask.** A tensor the shape of a padded batch of token ids, 1 for real tokens and 0 for
padding. [10 Text Becomes Tokens](10-tokens.md).

**Attention pooling.** One vector made from many tokens by an attention layer with a single query.
[21 One Vector or Many Tokens](21-pooled-vs-patch-tokens.md).

**Autocast.** PyTorch's mixed-precision context: operations on its 16-bit list run in bfloat16,
those on its float32 list in float32. A constant computed inside it is computed in bfloat16, which
is how the old position table lost positions. [12 Telling Tokens Where They Are](12-positions.md);
also [13](13-training-a-language-model.md).

**Autoregressive.** Producing each token from the tokens before it, one generation step at a time.
[14 Generating Text](14-generation.md); also [03](03-the-landscape.md).

**Averaged weights (EMA).** See *exponential moving average*.

## B

**Backbone.** The network that reads the observation and turns images and text into the
conditioning: a ViT, CLIP or a vision-language model. [03 The Model Families Today](03-the-landscape.md);
also [60](60-anatomy-of-a-vla.md).

**Behaviour cloning.** Supervised learning on demonstrations: a policy fitted to recorded pairs of
observation and expert action. [43 Imitation Learning and Why It Breaks](43-imitation-learning.md).

**bfloat16 (bf16).** A 16-bit float format with float32's exponent range and 8 significant bits:
whole numbers are exact up to 256, then only even ones, above 512 only multiples of 4.
[12 Telling Tokens Where They Are](12-positions.md); also [13](13-training-a-language-model.md).

**Budget.** A run's number of training samples; the learning-rate schedule is stretched over it.
[71 How Much Data, How Much Training](71-how-much-data-and-training.md).

**Byte-pair encoding (BPE).** Building a subword vocabulary by repeatedly merging the most frequent
adjacent pair of symbols into a new token. [10 Text Becomes Tokens](10-tokens.md).

## C

**Caption.** A text that describes an image; stage 2 trains the VLM on rendered scenes and
captions built from the simulator's state. [31 Training a VLM](31-training-a-vlm.md).

**Case study.** One task taken through every run made on it, in order, each step saying what was
believed, run, measured and changed. [100 Case Study: One Cube](100-case-pick-cube.md); also
[101](101-case-pick-target.md), [102](102-case-pretrained-backbones.md),
[103](103-case-self-correction.md).

**Causal.** Of a model: each position attends only to itself and earlier positions.
[10 Text Becomes Tokens](10-tokens.md).

**Causal mask.** What enforces causality: before the softmax, the score of any later key is set to
minus infinity. [11 The Transformer Block](11-the-transformer-block.md).

**Chat template.** A small program stored with a tokenizer that turns messages into one string with
the special tokens the model was trained with. [10 Text Becomes Tokens](10-tokens.md).

**Checkpoint.** The training state saved at one point: weights and position in the data
(`checkpoint_N.pt`), optimiser state (`optimizer_N.pt`) and, with averaged weights on,
`ema_N.pt`. A run resumes from it. When only the weights are meant, the chapters say *weights file*.
[01 Setup and a First Run](01-setup-and-first-run.md); also [80](80-the-training-loop.md),
[81](81-surviving-long-runs.md).

**Checkpoint interval.** The slice of the budget, in samples, after which a checkpoint is saved;
rounded to whole shards and batches. [80 The Training Loop](80-the-training-loop.md).

**Chunk.** The sequence of 15 future actions a policy predicts in one go, shape `[15, 8]`. The
client executes the first few (`--exec_horizon`, default 8) and asks again.
[01 Setup and a First Run](01-setup-and-first-run.md); also [42](42-chunks-and-execution.md).

**Class token.** A learned vector put in front of the patch sequence whose final state stands for
the whole image. [21 One Vector or Many Tokens](21-pooled-vs-patch-tokens.md).

**Closed at the named object.** The object nearest the gripper when it first closed was the one the
instruction named. Chance is 33% with three objects, 100% with one. Key
`closed_nearest_is_target_rate` in `results.json`; the lab notebook's column of the same name.
[01 Setup and a First Run](01-setup-and-first-run.md); also [92](92-measuring-failure.md).

**Closed-loop.** A policy running on its own outputs: each action changes what the cameras see
next. [42 Action Chunks and Closed-Loop Execution](42-chunks-and-execution.md).

**Closed-loop evaluation.** Running a policy in the simulator on unseen layouts and counting the
episodes that end in success. [01 Setup and a First Run](01-setup-and-first-run.md); also
[90](90-closed-loop-evaluation.md).

**Clutter, occlusion.** Objects that crowd the target or hide it from a camera. Not in `pick_target`,
where three objects stand at least 110 mm apart with nothing in front of them; the task
`pick_clutter` is not written. [110 Toward Real Tasks](110-toward-real-tasks.md).

**Compounding error.** In behaviour cloning, a small error moving the arm off the demonstrated
path, where the next observation is unlike the data and the next error larger.
[43 Imitation Learning and Why It Breaks](43-imitation-learning.md).

**Conditioning.** The tensor the backbone hands to the action head, `[batch, tokens, width]`. The
noise level and the joint state are separate inputs of the action head, not part of it.
[03 The Model Families Today](03-the-landscape.md); also [50](50-what-the-action-head-sees.md),
[60](60-anatomy-of-a-vla.md).

**`conditioning_len`.** A fixed prompt length for `all_tokens` conditioning, so that the action
head sees the same sequence length in a padded batch and when one prompt is served; a longer prompt
raises a `ValueError`. [10 Text Becomes Tokens](10-tokens.md); also
[50](50-what-the-action-head-sees.md).

**Confidence interval (95%).** A range produced by a procedure that contains the true rate in about
95% of repeated evaluations; here the Wilson interval. [90 Closed-Loop Evaluation](90-closed-loop-evaluation.md).

**Control.** In an experiment, a second run identical except in the one thing under test.
[62 The From-Scratch Path](62-the-from-scratch-path.md); also [102](102-case-pretrained-backbones.md).

**Control step.** One command to the arm: 50 ms in this simulator, which runs at 20 Hz. Episodes
end at 200 control steps. [01 Setup and a First Run](01-setup-and-first-run.md); also
[40](40-action-spaces.md).

**Copycat problem.** A policy given its own past actions that learns to repeat them instead of
reading the observation; one case of a *shortcut*.
[43 Imitation Learning and Why It Breaks](43-imitation-learning.md); also
[52](52-history-and-shortcuts.md).

**Counterfactual check.** A test that changes one input, holds the rest fixed and states in advance
what the output must do, such as a mismatched image for a VLM. [32 Testing a VLM](32-testing-a-vlm.md);
also [93](93-train-inference-skew.md).

**Covariate shift.** A change in the distribution of inputs between training and use while the
right output for each input stays the same. [43 Imitation Learning and Why It Breaks](43-imitation-learning.md).

**Cross-entropy.** The language-model loss: minus the log of the probability given to the token that
actually followed. [13 Training a Language Model](13-training-a-language-model.md).

## D

**Delta action.** An action given as a change from the previous command or state; the simulator here
uses absolute joint targets. [40 What an Action Is](40-action-spaces.md).

**Demonstration.** One recorded episode of the task done correctly; here by the scripted expert,
about 66 control steps each. [01 Setup and a First Run](01-setup-and-first-run.md); also
[70](70-demonstrations.md).

**Disturbance.** An evaluation (`eval.py --disturb reach` or `--disturb grasp`) or a demonstration
(`collect.py --disturb_prob`) in which the named object is moved once, during the reach or as the
gripper first closes; by 80 mm unless `--disturb_mm` says otherwise. The object is set down at the
first of up to 64 random directions that leaves room; if none does, the episode runs undisturbed.
For an object that keeps moving, see *drift*.
[01 Setup and a First Run](01-setup-and-first-run.md); also [43](43-imitation-learning.md),
[95](95-what-a-policy-looks-at.md), [103](103-case-self-correction.md).

**Draw.** One output of a generative model: an action chunk sampled from one starting noise, or a
token drawn with its probability. Not a *sample*.
[41 Generating Actions with Flow Matching](41-flow-matching.md); also [14](14-generation.md).

**Drift.** An evaluation (`eval.py --disturb drift`) or a share of demonstrations
(`collect.py --drift_prob`) in which the named object slides from the first control step along a
random path that turns away from the edge of the placement area and from the other objects, at
`--drift_cm_s` (default 2.0), and stands still while the gripper is commanded closed. In
`collect.py` the share applies to the episodes not drawn for a jump, and each drifting episode
draws its speed between 0.5 and 1.5 times `--drift_cm_s`. No result yet.
[110 Toward Real Tasks](110-toward-real-tasks.md); also [103](103-case-self-correction.md).

## E

**Early windows repeated.** A dataset in which the windows that start in frames 0-7 of every
demonstration appear 6 times instead of once (`boost_early_windows.py --max_frame 7 --repeat 5`).
The new dataset links the original shards and adds the copies; nothing is collected.
[70 Demonstrations](70-demonstrations.md); also [71](71-how-much-data-and-training.md),
[72](72-which-demonstrations-teach.md), [95](95-what-a-policy-looks-at.md).

**EGL.** The interface through which the simulator renders on a GPU without a display; its device
order is not `nvidia-smi`'s. [01 Setup and a First Run](01-setup-and-first-run.md); also
[81](81-surviving-long-runs.md).

**Embedding table.** A matrix `[vocabulary size, width]`, one row per token id. The instruction
table of `dp_vit_lang*` (49,280 x 256) adds 12.6M parameters.
[10 Text Becomes Tokens](10-tokens.md).

**Embodiment.** One robot body with its cameras and action space.
[02 From Language Models to Robot Policies](02-how-we-got-here.md).

**End-effector action.** An action that names the gripper's pose instead of joint angles; it needs
inverse kinematics. [40 What an Action Is](40-action-spaces.md).

**End-of-sequence token (EOS).** The special token that ends a text; generation can stop on it.
[14 Generating Text](14-generation.md).

**Episode.** One attempt at the task from a fresh layout, at most 200 control steps.
[90 Closed-Loop Evaluation](90-closed-loop-evaluation.md).

**Epoch.** One pass over a dataset's windows: 5M samples on `pick_target_4k` are 19.0 epochs.
[01 Setup and a First Run](01-setup-and-first-run.md); also [71](71-how-much-data-and-training.md).

**Every object named.** A dataset in which each layout is demonstrated once per object on the
table, each named in turn (`collect.py --targets_per_layout 3`, `pick_target_cf4k`): the scene is
identical and only the instruction differs, and 4000 episodes are about 1333 layouts.
[70 Demonstrations](70-demonstrations.md); also [51](51-language-conditioning.md),
[72](72-which-demonstrations-teach.md).

**Execution horizon.** The number of actions of a chunk executed before the policy is queried again
(`--exec_horizon`, default 8 of 15). [42 Action Chunks and Closed-Loop Execution](42-chunks-and-execution.md).

**Expert handover.** A probe in which the scripted expert drives the first K control steps
(`eval.py --expert_prefix K`), toward the named object or another one (`--prefix_target
distractor`), and the policy takes over. [51 The Instruction Has to Reach the Action Head](51-language-conditioning.md);
also [95](95-what-a-policy-looks-at.md).

**Exponential moving average (EMA), averaged weights.** A second copy of the weights that is never
trained directly: after every optimiser step each of its weights moves a small fraction toward the
trained one. Saved as `ema_N.pt`; the pickup policy configs switch it on (the two Qwen configs leave
it to `--ema.enabled True`), and `eval.sh` serves it by default. [80 The Training Loop](80-the-training-loop.md); also [01](01-setup-and-first-run.md),
[71](71-how-much-data-and-training.md).

## F

**Falsy.** Treated as false by Python; `0 or 1` is `1`, the bug behind `lowdim_past_timesteps or
default`. [93 Train and Inference Must See the Same Thing](93-train-inference-skew.md).

**Feed-forward network.** The sublayer of a transformer block that transforms each position on its
own. [11 The Transformer Block](11-the-transformer-block.md).

**Fine-tuning.** Continuing training from pretrained weights on a new task.
[02 From Language Models to Robot Policies](02-how-we-got-here.md).

**Floor.** A comparison is at the floor when both sides succeed near 0% and close at the named
object at chance: no difference between them can show.
[102 Case Study: A Backbone From Scratch Against a Pretrained One](102-case-pretrained-backbones.md).

**Flow matching.** How the action head here generates actions: a demonstrated chunk is mixed with
noise along a straight line, the network learns the direction from action to noise, and sampling
follows it back from pure noise in a few flow steps.
[41 Generating Actions with Flow Matching](41-flow-matching.md); also
[02](02-how-we-got-here.md).

**Flow step.** One step of the sampler from noise toward actions (`--flow_steps`, default 10). Use a
divisor of 1000. [01 Setup and a First Run](01-setup-and-first-run.md); also
[41](41-flow-matching.md).

**Flower trimming.** The real task this repository is heading for (Phase 4 of the roadmap): find
one flower among others that look like it, among leaves that move and hide it, and cut its stem in
the right place. [110 Toward Real Tasks](110-toward-real-tasks.md).

**Freeze.** Exclude a part's weights from training. [31 Training a VLM](31-training-a-vlm.md);
also [61](61-pretrained-backbones.md).

**From-scratch path.** Training a language model (stage 1), a VLM (stage 2) and a VLA (stage 3)
here from random weights, each stage starting from the last one's weights (`custom_path.sh`).
[62 The From-Scratch Path](62-the-from-scratch-path.md); also
[102](102-case-pretrained-backbones.md).

**Frozen cameras.** A probe that sends the first frame's images for the whole episode
(`eval.py --stale_images`), so that only the joint state changes.
[95 Probing a Policy That Works](95-what-a-policy-looks-at.md); also [21](21-pooled-vs-patch-tokens.md).

## G

**Gap from simulation.** Every difference between the simulator and a real robot: rendered images
under fixed lights against real cameras, simulated against real contact. Not measured here; the one
gap measured between two setups is the grasp tolerance on two simulator builds.
[110 Toward Real Tasks](110-toward-real-tasks.md).

**Generation step.** One pass of the text-generation loop: pick a token, append it, run the model
again. [14 Generating Text](14-generation.md).

**Gradient clipping.** Scaling all gradients down by one factor when their joint L2 norm exceeds a
limit. [13 Training a Language Model](13-training-a-language-model.md).

**`GraspProbe`.** The recorder in `sim/pickup/diagnostics.py` that reads the true positions of the
gripper and every object when the gripper closes, and counts the closes of an episode.
[92 Measuring How a Policy Fails](92-measuring-failure.md); also [01](01-setup-and-first-run.md).

**Greedy decoding.** Generating by taking the highest-scoring token at every generation step.
[14 Generating Text](14-generation.md).

## H

**Head (attention).** One of the equal slices of the query, key and value vectors, each with its
own attention pattern. [11 The Transformer Block](11-the-transformer-block.md).

**Held-out inputs.** Inputs not used in training. [32 Testing a VLM](32-testing-a-vlm.md).

**Hidden state.** The vector a block outputs at one position.
[30 Joining Vision and Language](30-joining-vision-and-language.md).

**History.** A policy with history receives the frames and states at $t-1$ and $t$ and its previous
action; a policy without history (the `*_reactive` configs) receives one frame per camera and the
current state. [01 Setup and a First Run](01-setup-and-first-run.md); also
[52](52-history-and-shortcuts.md).

**Hold command.** The command the robot is under after reset: the start joint angles with the
gripper open. It fills the previous-action slot of the first query when a policy is served.
[40 What an Action Is](40-action-spaces.md); also [52](52-history-and-shortcuts.md).

**Hold frame.** A frame of the robot at rest recorded at the start of every demonstration
(`collect.py --hold_steps 1`), so that the first windows hold what serving provides.
[52 History and the Shortcut It Opens](52-history-and-shortcuts.md); also
[70](70-demonstrations.md), [72](72-which-demonstrations-teach.md).

## I

**Id (token id).** A token's index in the vocabulary. [10 Text Becomes Tokens](10-tokens.md).

**Initialisation.** The values the weights hold before a run: random, or the result of pretraining.
[62 The From-Scratch Path](62-the-from-scratch-path.md); also [102](102-case-pretrained-backbones.md).

**Invariant.** A property that must hold for every input, such as "the conditioning of a sample
does not depend on the rest of its batch"; pinned by unit tests.
[93 Train and Inference Must See the Same Thing](93-train-inference-skew.md).

**Inverse kinematics (IK).** Finding joint angles that put the tool centre point at a wanted pose.
[40 What an Action Is](40-action-spaces.md); also [70](70-demonstrations.md).

## K

**Key-value cache (KV cache).** The keys and values of all tokens processed so far, kept per layer
so that generation processes one new token per generation step. [14 Generating Text](14-generation.md).

## L

**Language model.** A causal transformer that predicts the next token; here a 91.4M-parameter model
trained in stage 1. [13 Training a Language Model](13-training-a-language-model.md); also
[30](30-joining-vision-and-language.md).

**Language prior.** What a model can predict about text from the text alone.
[32 Testing a VLM](32-testing-a-vlm.md).

**Latency.** The time from an observation to the chunk computed from it.
[42 Action Chunks and Closed-Loop Execution](42-chunks-and-execution.md).

**LayerNorm.** A normalisation of each token's vector to zero mean and unit variance across its
features, then a learned gain and bias. [11 The Transformer Block](11-the-transformer-block.md).

**Layout.** Which objects are on the table, where they are, which one the instruction names, and the
arm's start pose; fixed by the seed. [01 Setup and a First Run](01-setup-and-first-run.md); also
[90](90-closed-loop-evaluation.md).

**Liveness signal.** Something visible from outside a run that changes only while it makes
progress; here the training log. [81 Surviving Long Runs](81-surviving-long-runs.md).

**Lockfile.** `uv.lock`, the exact version of every dependency. [81 Surviving Long Runs](81-surviving-long-runs.md).

**Logits.** The unnormalised scores, one per vocabulary entry, that a softmax turns into
probabilities. [11 The Transformer Block](11-the-transformer-block.md); also
[14](14-generation.md).

**Look-alikes (`pick_lookalike`).** Objects of one shape whose colours are a step apart. The task
`pick_lookalike` in `sim/pickup/env.py` puts 3 of a red, an orange, a pink and a dark red cube on
the table, so only the colour word of the instruction tells them apart. No result yet.
[110 Toward Real Tasks](110-toward-real-tasks.md).

**Looks once.** A policy that chooses its target from the first frames and then follows its own arm,
so that later images matter only for the last centimetres. Shown with frozen cameras, expert
handover and disturbances. [01 Setup and a First Run](01-setup-and-first-run.md); also
[95](95-what-a-policy-looks-at.md), [103](103-case-self-correction.md).

**Low-dimensional (lowdim).** The framework's name for the small arrays of a window, here the 8-D
state and the 8-D action. [42 Action Chunks and Closed-Loop Execution](42-chunks-and-execution.md).

## M

**Machine A, machine B.** The two computers of the measurements: machine A (one RTX 4090, 24 GB) and
machine B (L40S GPUs, 128 CPU cores). [00 The Route on One Page](00-the-route.md).

**Mean pooling.** One vector made from many by averaging them.
[21 One Vector or Many Tokens](21-pooled-vs-patch-tokens.md).

**Median miss.** The median distance, at the first close, from the gripper to the named object
(`median_xy_err_target_mm`); the expert's is 5 mm. [01 Setup and a First Run](01-setup-and-first-run.md);
also [71](71-how-much-data-and-training.md), [92](92-measuring-failure.md).

**Mixed precision.** Most arithmetic in a 16-bit format, the sensitive parts in float32.
[13 Training a Language Model](13-training-a-language-model.md); also [80](80-the-training-loop.md).

**Mode, multimodal.** A mode is one peak of a distribution; a multimodal distribution has several,
such as the actions for an observation that allows more than one correct motion.
[41 Generating Actions with Flow Matching](41-flow-matching.md).

## N

**Noise level.** How far a chunk has been mixed with noise, given to the action head as one token in
front of the conditioning (the *time token*); not a control step.
[03 The Model Families Today](03-the-landscape.md); also [21](21-pooled-vs-patch-tokens.md),
[50](50-what-the-action-head-sees.md).

**Normalisation (of actions and state).** Rescaling every channel by its mean and standard
deviation before the model sees it, and reversing that on the output.
[40 What an Action Is](40-action-spaces.md).

## O

**Offset (between positions).** The query's position minus the key's.
[12 Telling Tokens Where They Are](12-positions.md).

**Onset (of looking).** The point in training at which a policy starts to use the scene, read per
checkpoint from closed at the named object and the slope, which rise before success does; the
start of the *transition*. [72 Which Demonstrations Teach What](72-which-demonstrations-teach.md);
also [12](12-positions.md), [71](71-how-much-data-and-training.md),
[102](102-case-pretrained-backbones.md).

**Open-loop.** The model's predictions are compared with recorded data and never executed, as in
`open_loop_check.py` or the notebook's "First action chunk, open loop".
[43 Imitation Learning and Why It Breaks](43-imitation-learning.md); also
[91](91-loss-is-not-success.md).

**Optical axis.** The direction a camera looks along. [31 Training a VLM](31-training-a-vlm.md).

**Optimiser step.** One update of the weights on one batch: 5,000,000 samples at batch 64 are
78,125 optimiser steps. [13 Training a Language Model](13-training-a-language-model.md); also
[71](71-how-much-data-and-training.md), [80](80-the-training-loop.md).

## P

**p-value.** For the sign test on paired evaluations, the probability of a split at least as
lopsided as the one observed if each disagreement were a coin toss.
[90 Closed-Loop Evaluation](90-closed-loop-evaluation.md).

**Padding, padding side.** Pad ids appended to the short rows of a batch; the padding side is the
end that receives them. [10 Text Becomes Tokens](10-tokens.md).

**Patch, patch token.** One of the squares, `patch_size` pixels on a side, that a ViT cuts a frame
into, and the encoder's output vector for it. [20 Images Become Tokens](20-images-to-tokens.md);
also [21](21-pooled-vs-patch-tokens.md).

**Pixel shuffle.** Shortening the image part of a VLM sequence by joining each `r` x `r` block of
patch tokens into one wider token. [30 Joining Vision and Language](30-joining-vision-and-language.md).

**Placeholder token.** The `<image>` token, repeated in the prompt once per image token; the image
vectors are written over its embeddings. [30 Joining Vision and Language](30-joining-vision-and-language.md).

**Plateau.** The level at which success holds after the transition.
[71 How Much Data, How Much Training](71-how-much-data-and-training.md).

**Policy.** A function from observations to actions. [01 Setup and a First Run](01-setup-and-first-run.md).

**Pooled embedding.** A single vector that summarises all the patch tokens of an image.
[21 One Vector or Many Tokens](21-pooled-vs-patch-tokens.md).

**Position.** A token's index in the sequence, from 0. [12 Telling Tokens Where They Are](12-positions.md).

**Position embedding (learned).** In the ViT, a learned vector per patch position added to that
patch's token. [20 Images Become Tokens](20-images-to-tokens.md).

**Position table.** The table of rotary angles, cosines and sines a transformer builds for every
position up to `max_seq_len`. Until it was fixed it was computed in bfloat16 under autocast (641
distinct positions of 2048); it is now computed in float32, and
`positional_embedding_type: rotary_legacy_autocast` rebuilds the old one for weights trained with
it. [12 Telling Tokens Where They Are](12-positions.md); also [13](13-training-a-language-model.md).

**Pre-norm.** A block in which the norm sits in front of each sublayer on the branch, never on the
residual stream. [11 The Transformer Block](11-the-transformer-block.md).

**Preset.** A YAML fragment under `vla_foundry/config_presets/` that a config includes.
[80 The Training Loop](80-the-training-loop.md).

**Pretraining.** Training on one task so that the weights become the starting point for another.
[62 The From-Scratch Path](62-the-from-scratch-path.md); also [102](102-case-pretrained-backbones.md).

**Previous-action slot.** Row 0 of the action tensor of a policy with history: the previous action,
given clean and never scored. In a window that starts at the first frame it holds a copy of the
first action; when served, the hold command at the first query.
[52 History and the Shortcut It Opens](52-history-and-shortcuts.md).

**Probe.** An evaluation that changes one thing about an episode: frozen cameras
(`--stale_images`), expert handover (`--expert_prefix`), the expert's first command as the first
previous action (`--first_prev_action expert`), a disturbance (`--disturb`) or a random start pose
(`--random_start`). `GraspProbe` is the recorder that measures where the gripper closed.
[01 Setup and a First Run](01-setup-and-first-run.md); also [95](95-what-a-policy-looks-at.md).

**Projector.** The small trainable map from vision-encoder width to language-model width.
[30 Joining Vision and Language](30-joining-vision-and-language.md).

**Proprioception, state.** The robot's measurement of itself: 7 joint angles and the gripper
opening, fed to the policy as the joint state. [40 What an Action Is](40-action-spaces.md).

## R

**Random start.** Demonstrations or evaluations in which every episode starts from a random gripper
pose above the table instead of the home pose (`collect.py --random_start`, `eval.py
--random_start`). [70 Demonstrations](70-demonstrations.md); also [72](72-which-demonstrations-teach.md),
[95](95-what-a-policy-looks-at.md), [103](103-case-self-correction.md).

**Reactive expert (the expert that plans again).** `ScriptedExpert(reactive=True)`. Before every
action `_needs_new_plan` checks whether the named object lies more than 15 mm from where the plan
was made for it, or whether the fingers met on nothing during the lift. A move of up to 40 mm
(`TRACK_LIMIT`) while the gripper is not being closed shifts the remaining waypoints (`_track`,
counted in `tracks`); anything else makes it plan again (`_plan(again=True)`, counted in `replans`).
Switched on by `eval.py --expert --expert_reactive`, and for every episode by `collect.py
--disturb_prob` or `--drift_prob` above 0. [103 Case Study: A Policy That Corrects Itself](103-case-self-correction.md);
also [72](72-which-demonstrations-teach.md), [110](110-toward-real-tasks.md).

**Recoveries.** Demonstrations in which the named object is moved once and the expert plans again
(`collect.py --disturb_prob 0.5`, with `ScriptedExpert(reactive=True)`). Only episodes in which a
free place was found are moved: 1505 of 4000 in `pick_target_4k_recover`.
[70 Demonstrations](70-demonstrations.md); also [72](72-which-demonstrations-teach.md),
[95](95-what-a-policy-looks-at.md), [103](103-case-self-correction.md).

**Register tokens.** Extra learned tokens of some DINOv2 checkpoints; not patches.
[22 Pretrained Vision Encoders](22-pretrained-vision-encoders.md).

**Regression.** Training the output directly with mean squared error, which predicts the mean of
all targets for an input. [41 Generating Actions with Flow Matching](41-flow-matching.md).

**Re-plan.** A new query to the policy after the executed part of a chunk. Not the scripted
expert's `replans`, which counts new plans after the object moved (see *reactive expert*).
[42 Action Chunks and Closed-Loop Execution](42-chunks-and-execution.md).

**Residual stream.** The tensor `[batch, tokens, width]` that runs from embedding to output, each
block adding its update. [11 The Transformer Block](11-the-transformer-block.md).

**Retreat waypoint.** The waypoint the expert that plans again puts first when its gripper
set-point is more than 2 cm below the new approach height: straight up, gripper open, so the
fingers clear the objects before the arm moves sideways.
[103 Case Study: A Policy That Corrects Itself](103-case-self-correction.md).

**Retried rate.** The share of episodes in which the gripper closed twice or more, a second grasp
attempt (`grasp.retried_rate` in `results.json`; `closes >= 2` per episode). A retry is not a
recovery: it says nothing about success. [92 Measuring How a Policy Fails](92-measuring-failure.md);
also [95](95-what-a-policy-looks-at.md), [103](103-case-self-correction.md).

**Rollout.** One run of a policy in the simulator. [43 Imitation Learning and Why It Breaks](43-imitation-learning.md).

**Rotary position embedding (RoPE).** Queries and keys rotated in every attention layer by an angle
proportional to their position, so that a score depends on content and offset only.
[12 Telling Tokens Where They Are](12-positions.md).

## S

**Sample.** One training window drawn by the data loader; budgets are counted in samples. A draw from
the model is a *draw*. [01 Setup and a First Run](01-setup-and-first-run.md); also
[71](71-how-much-data-and-training.md).

**Sampler.** The procedure that turns noise into a chunk in a number of flow steps.
[41 Generating Actions with Flow Matching](41-flow-matching.md).

**Scripted expert.** A hand-written controller that reads the true object pose from the simulator
and executes a plan; it records the demonstrations and, through `eval.py --expert`, checks the task.
With `--expert_reactive` it plans again when the object has moved (see *reactive expert*).
[00 The Route on One Page](00-the-route.md); also [70](70-demonstrations.md),
[103](103-case-self-correction.md).

**Second set of layouts.** A further 100 unseen layouts (`--seed_start 1000100`), to re-evaluate a
chosen checkpoint. [90 Closed-Loop Evaluation](90-closed-loop-evaluation.md); also
[103](103-case-self-correction.md).

**Seed (layout seed), training seed.** The integer a layout is generated from (demonstrations from 0,
evaluation from 1,000,000), and, separately, the integer that seeds a training run
(`--data.seed`, `--hparams.seed`). [90 Closed-Loop Evaluation](90-closed-loop-evaluation.md).

**Serving path.** The code between a raw observation and the model call at run time, here
`Policy.infer` in `sim/pickup/policy_server.py`. A policy is *served* when it answers the
simulator's queries. [93 Train and Inference Must See the Same Thing](93-train-inference-skew.md);
also [52](52-history-and-shortcuts.md).

**Shard.** A tar file of 256 windows that the data loader reads as a stream.
[01 Setup and a First Run](01-setup-and-first-run.md); also [70](70-demonstrations.md).

**Shortcut.** A rule that fits the training data through an input that does not carry the same
information when the policy runs; the previous action is the one found here (causal confusion, the
copycat problem). [52 History and the Shortcut It Opens](52-history-and-shortcuts.md).

**Slope.** The least-squares slope of where the gripper first closed against where the named object
was: 1 when it closes where the object is, 0 when the scene makes no difference. It separates an
imprecise policy from one that is not looking. [01 Setup and a First Run](01-setup-and-first-run.md);
also [92](92-measuring-failure.md).

**Softmax.** Exponentiate, then divide by the total. [11 The Transformer Block](11-the-transformer-block.md).

**Special token.** A vocabulary entry that stands for structure, not text.
[10 Text Becomes Tokens](10-tokens.md).

**State token.** The token of the action head's sequence that carries the joint state, one per time
step. [50 What the Action Head Sees](50-what-the-action-head-sees.md).

**Success, success rate.** The named object held 8 cm above its rest height for 10 control steps
within 200; the rate is successes over episodes. [00 The Route on One Page](00-the-route.md); also
[90](90-closed-loop-evaluation.md).

**Supervisor.** A process that starts the trainer, watches the liveness signal and restarts from the
newest checkpoint: `train_resilient.sh`. [81 Surviving Long Runs](81-surviving-long-runs.md).

## T

**Teacher forcing.** Scoring a model on true text, each position predicted from the true tokens
before it. [32 Testing a VLM](32-testing-a-vlm.md).

**Teleoperated.** Driven by a person. [02 From Language Models to Robot Policies](02-how-we-got-here.md).

**Temperature, top-k, top-p.** Sampling settings: divide the logits by $T$; keep the $k$ best tokens;
keep the smallest set whose probability exceeds $p$. [14 Generating Text](14-generation.md).

**Text slots (`text_tokens`).** The instruction's token ids, each given a learned vector and placed
in front of the patch tokens of a patch-token policy (32 slots in the pickup configs).
[51 The Instruction Has to Reach the Action Head](51-language-conditioning.md); also
[50](50-what-the-action-head-sees.md).

**Time token.** See *noise level*.

**Token.** A piece of text treated as one unit; more generally one vector of a transformer's
sequence. [10 Text Becomes Tokens](10-tokens.md); also [20](20-images-to-tokens.md).

**Tolerance.** The placement error at which a grasp still succeeds, per object and simulator build.
[92 Measuring How a Policy Fails](92-measuring-failure.md); also [00](00-the-route.md),
[110](110-toward-real-tasks.md).

**Tool centre point (TCP).** The reference point between the fingertips.
[40 What an Action Is](40-action-spaces.md); also [92](92-measuring-failure.md).

**Train/inference skew.** Any difference between what reaches the model in training and what
reaches it when served for the same input. [32 Testing a VLM](32-testing-a-vlm.md); also
[93](93-train-inference-skew.md).

**Trained weights (raw).** A run's weights as the optimiser left them, as opposed to their
exponential moving average. Tables give "trained / averaged", or "raw / EMA".
[01 Setup and a First Run](01-setup-and-first-run.md); also [71](71-how-much-data-and-training.md).

**Transition.** The stretch of training in which the slope goes from near 0 to near 1: slope first,
then median miss, then success; its start is the *onset*. [71 How Much Data, How Much Training](71-how-much-data-and-training.md);
also [72](72-which-demonstrations-teach.md), [101](101-case-pick-target.md).

**Truncation, truncation side.** Cutting a row longer than `max_length`, and the end that is cut;
the SmolVLM2 tokenizer cuts on the left. [10 Text Becomes Tokens](10-tokens.md).

## U

**Unseen layouts.** Layouts from seeds the demonstrations never used (1,000,000 and up).
[00 The Route on One Page](00-the-route.md).

## V

**Vision encoder.** The network that turns an image into vectors.
[30 Joining Vision and Language](30-joining-vision-and-language.md); also
[22](22-pretrained-vision-encoders.md).

**Vision Transformer (ViT).** A transformer that cuts a frame into patches and makes each one a
token. [20 Images Become Tokens](20-images-to-tokens.md).

**Vision-language model (VLM).** A language model that also reads image tokens.
[03 The Model Families Today](03-the-landscape.md); also [30](30-joining-vision-and-language.md).

**Vision-language-action policy (VLA).** A policy that reads images and an instruction and outputs
actions; in the recipes here a VLM backbone with an action head.
[00 The Route on One Page](00-the-route.md); also [60](60-anatomy-of-a-vla.md).

**Vocabulary.** The fixed list of all tokens. [10 Text Becomes Tokens](10-tokens.md).

## W

**Warm-up.** The first part of the learning-rate schedule, a linear rise to the peak rate. Policy
runs then follow a cosine from 5e-4 down to 1e-5. [13 Training a Language Model](13-training-a-language-model.md);
also [80](80-the-training-loop.md).

**Waypoint.** One target of the scripted expert's plan for the tool centre point.
[70 Demonstrations](70-demonstrations.md).

**Weight decay.** A small pull of every weight toward zero. [13 Training a Language Model](13-training-a-language-model.md).

**Weight tying.** One matrix for the embedding table and the output projection.
[11 The Transformer Block](11-the-transformer-block.md).

**Weights file.** A file of weights alone, such as `checkpoint_N.pt` read for its weights or
`ema_N.pt`; `results.json` names the one served under `server`.
[01 Setup and a First Run](01-setup-and-first-run.md).

**Width.** The length of each token's vector. [10 Text Becomes Tokens](10-tokens.md).

**Wilson interval.** The confidence interval `eval.py` reports (`success_ci95`); unlike rate
plus or minus 1.96 standard errors it does not shrink to zero width at 0 successes.
[90 Closed-Loop Evaluation](90-closed-loop-evaluation.md).

**Window.** One training example: the observations at one frame plus the actions that follow. Here
it holds the images at `t-1` and `t` and the states and actions of 16 frames (1 past, the current
one, 14 future); every frame starts one. [01 Setup and a First Run](01-setup-and-first-run.md);
also [70](70-demonstrations.md).

**Works.** A policy works on a task when its closed-loop success rate on unseen layouts clears a
stated threshold, reported with a confidence interval. [00 The Route on One Page](00-the-route.md).

## Y

**Yaw.** Rotation about the vertical axis; for the gripper, of the direction its fingers close.
[92 Measuring How a Policy Fails](92-measuring-failure.md).
