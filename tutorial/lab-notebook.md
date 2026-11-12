# Lab notebook

Every number measured in this repository, in one place. Chapters cite this file; if a number is not
here, it was not measured here. All success rates are closed-loop on layouts the policy never saw
(evaluation seeds `1_000_000+`), one training seed per run, Wilson 95% intervals. Differences of a
few percent between single runs are noise.

Two machines were used. **Machine A** has one 24 GB GPU (RTX 4090). **Machine B** has 46 GB GPUs
(L40S) and 128 CPU cores. Machine A was unreliable; its runs died or hung several times and were
resumed from checkpoints.

## Tasks and data

| | |
|---|---|
| Robot | Franka Panda in MuJoCo, 20 Hz control |
| Observations | `front` and `wrist` RGB cameras at 224x224, 8-D state (7 joint angles, gripper opening in [0, 1]) |
| Actions | 8-D absolute targets (7 joint positions, gripper command 1 = open, 0 = closed) |
| `pick_cube` | one red cube at a random pose, "pick up the red cube" |
| `pick_target` | 3 of 4 objects (red cube, green can, blue hex nut, yellow T block) at random poses, "pick up the \<object\>" |
| Success | the named object held 8 cm above its rest height for 10 control steps (0.5 s); episodes end at 200 steps |
| Demonstrations | scripted expert, about 66 steps each, failures discarded; seeds `0..` |
| Training windows | 1 past + current + 14 future low-dimensional steps, images at t-1 and t; 1500 `pick_target` demos give 98,841 windows |

### Datasets

Every frame of a demonstration starts one training window, so windows = frames.

| Name | Episodes | Windows | What is in it |
|---|---|---|---|
| `pick_cube_1k` | 1000 | 65,391 | one red cube |
| `pick_target_1500` | 1500 | 98,841 | 1500 layouts, one instruction each |
| `pick_target_4k` | 4000 | 263,289 | 4000 layouts, one instruction each |
| `pick_target_cf` | 1500 | 98,616 | 500 layouts, each demonstrated three times, once per object |
| `pick_target_cf4k` | 4000 | 263,552 | about 1333 layouts, each demonstrated once per object |
| `pick_target_cf_early` | 1500 | 158,616 | `pick_target_cf` with the windows that start in frames 0-7 present 6 times |
| `pick_target_4k_early` | 4000 | 423,289 | `pick_target_4k` with the windows that start in frames 0-7 present 6 times |
| `pick_target_4k_recover` | 4000 | 293,541 | same seeds as `pick_target_4k`; in 1505 episodes the object is moved once (during the reach or as the gripper closes) and the expert plans again |
| `pick_target_4k_rs` | 4000 | 241,318 | same seeds; every episode starts from a random gripper pose above the table |
| `pick_target_4k_rs_recover` | 4000 | 275,550 | random start poses, and the object is moved once in 1456 episodes |
| `pick_target_4k_hold` | 4000 | 267,075 | same seeds; every episode begins with one frame of the robot at rest |

Training budgets in passes over the data: 1M samples on `pick_target_1500` are 10.1 epochs, 3M are
30.4; 5M samples on `pick_target_4k` are 19.0 epochs; 1.5M on `pick_cube_1k` are 22.9.

## The scripted expert

| Measurement | Result |
|---|---|
| Expert through the evaluation loop, `pick_target`, 100 layouts (machine B) | 100/100, gripper closes 5 mm (median) from the object, 98% within 10 mm |
| Same on machine A | 99/100 |
| Expert spot check, `pick_cube` | 49/50 |

## Grasp tolerance

Expert with a fixed sideways offset added to its grasp point, random direction, 320 episodes per
offset. Measured on both machines with the same seeds:

Machine B (MuJoCo 4.0.1), the machine all second-round results come from:

| Offset | Cube | T block | Can | Hex nut | All |
|---|---|---|---|---|---|
| 0 mm | 100% | 100% | 100% | 100% | 100% |
| 5 mm | 97% | 100% | 99% | 99% | 99% |
| 10 mm | 99% | 100% | 94% | 97% | 98% |
| 15 mm | 95% | 100% | 76% | 93% | 91% |
| 20 mm | 93% | 98% | 29% | 70% | 72% |
| 25 mm | 73% | 89% | 15% | 31% | 52% |
| 30 mm | 32% | 76% | 3% | 15% | 32% |

Machine A (an earlier MuJoCo 4.0 build; the exact version was not recorded):

| Offset | Cube | T block | Can | Hex nut | All |
|---|---|---|---|---|---|
| 0 mm | 100% | 100% | 94% | 89% | 96% |
| 5 mm | 97% | 100% | 87% | 83% | 92% |
| 10 mm | 99% | 100% | 60% | 61% | 80% |
| 15 mm | 95% | 100% | 35% | 37% | 67% |
| 20 mm | 93% | 98% | 18% | 25% | 58% |
| 25 mm | 73% | 89% | 11% | 21% | 49% |
| 30 mm | 32% | 76% | 0% | 10% | 30% |

The cube and T block columns are identical on the two machines. The can and hex nut, whose
collision shapes are cylinders, are much easier to grasp on machine B. The simulator build is the
one known difference between the two; it has not been isolated as the cause.

## `pick_cube` (machine A)

| Policy | Demos | Training samples (epochs) | Final loss | Success | 95% CI |
|---|---|---|---|---|---|
| Diffusion policy + CLIP ViT-B/32, one pooled vector per image and one for the text, 245M | 200 | 300k (23) | 0.040 | 2/50 = 4% | 1-14% |
| Diffusion policy + small ViT from scratch, every patch a token, 36M (`dp_vit`) | 200 | 300k (23) | 0.091 | 15/50 = 30% | 19-44% |
| same | 1000 | 500k (8) | 0.084 | 20/50 = 40% | 28-54% |
| same | 1000 | 1.5M (23) | 0.058 | 37/50 = 74% | 60-84% |

Inference settings on the last row's checkpoint, 50 episodes each: default (8 actions executed per
chunk, 10 flow steps) 74%; 4 actions per chunk 84%; 15 actions per chunk 78%; 4 flow steps 86%; 25
flow steps 84%. All intervals overlap.

The pooled-CLIP policy scored 0/20 on its own training layouts: it ran a plausible
reach-descend-close-lift a few centimetres from the cube.

## `pick_target`, first round (machine A)

| Policy | Demos | Samples (epochs) | Final loss | Success |
|---|---|---|---|---|
| `dp_vit`: ViT patch tokens, no language, 36M | 1500 | 1.0M (10) | 0.069 | 1/100 |
| `vla_qwen`: Qwen3-VL-2B, last 4 LLM layers + action head trained (306M of 2.2B), one conditioning vector | 1500 | 100k (1) | 0.087 | 0/50 |
| same run | 1500 | 200k (2) | 0.066 | 1/50 |
| same run | 1500 | 300k (3) | 0.039 | 6/50 |
| same run, final | 1500 | 400k (4) | 0.03-0.04 | 6/100 |
| `vla_custom`: 91M LLM -> 96M VLM -> 130M VLA, all trained here, one conditioning vector | 1500 | 1.0M (10) | ~0.03 | 0/100 |
| same architecture, VLM randomly initialised | 1500 | 1.0M (10) | ~0.03 | 1/100 |

The Qwen policy never lifted an object other than the named one in 150 episodes.

Where the gripper was when it first closed (same checkpoints, 100 episodes):

| Policy | Closed the gripper | Nearest object was the named one | Median miss to named | Median miss to nearest |
|---|---|---|---|---|
| `dp_vit`, no language | 100% | 32% | 185 mm | 110 mm |
| `vla_custom`, pretrained VLM | 100% | 33% | 175 mm | 89 mm |
| `vla_custom`, random-init VLM | 100% | 29% | 174 mm | 89 mm |

Chance for "nearest object was the named one" is 33%. All three closed at the right height (within
about 3 mm) around step 34-44, as the expert does.

## The from-scratch stages (machine A unless noted)

| Stage | Model | Data | Budget | Result |
|---|---|---|---|---|
| 1. Language model | 12 layers, 8 heads, 512 wide, vocabulary 49,280, 91.4M parameters | 61,000 DCLM-baseline web documents | 500k samples, batch 32, about 15 min at 550 samples/s | loss 6.3 -> 4.1, grammatical English. Machine B: final loss 4.15 |
| 2. VLM | stage 1 + ViT (224 px, patch 14, 256 wide, 6 layers) + linear projector, 96.5M | 40,000 captioned sim renders | 400k samples, about 15 min at 437 samples/s | loss 0.058; on 300 held-out scenes it lists the objects in left-to-right order 85% of the time (256/300) and names the leftmost object 95% (285/300) |
| 3. VLA | stage 2 + flow-matching action transformer, 130.0M, everything trainable | 1500 `pick_target` demos | 1.0M samples, about 1.7 h at 165 samples/s | 0/100 (see above) |

- Stage 1 ran out of memory at batch 128: the float32 logits for 128 x 256 tokens over a 49,280-word
  vocabulary are 6.5 GB per tensor.
- Before the pixel-range fix (below) the stage-2 model scored 1/300 on exact ordering and 70/300 on
  the leftmost object. A teacher-forced check on that model gave the same loss with the right and
  with a mismatched image when images were fed as PIL (1.93 vs 1.89), and 0.10 vs 1.18 when fed as
  the unscaled uint8 tensors it was trained on.

## `pick_target`, second round (machine B)

All policies in this section are the 36M-parameter one: a flow-matching action transformer on the patch
tokens of a small ViT trained from scratch. "Language" adds the instruction as tokens. "History" =
frames t-1 and t, states t-1 and t, and the previous action as inputs. "No history" (`*_reactive`) =
one frame per camera, the current state, no previous action. Every evaluation is 100 unseen layouts,
8 actions executed per chunk, 10 flow steps, unless stated. "raw" = the trained weights, "EMA" = their
running average, saved by the same run.

**Slope**: least-squares slope of where the gripper first closed against where the named object was
(both table axes pooled). 1: it closes where the object is. 0: where it closes does not depend on the
scene. **Closed at the named object**: the object nearest the gripper when it first closed was the
named one (chance 33%).

Runs marked "position fix" were trained after the position-table bug (see the bug table) was fixed;
the others before. Evaluations always use the code a run was trained with.

### Final checkpoints

| Policy | Demos | Samples | Success (raw / EMA) | Closed at the named object | Slope | Median miss to named | Best checkpoint |
|---|---|---|---|---|---|---|---|
| `dp_vit`: no language, history | 1500 | 1.0M | 0% / 0% | 31% / 37% | 0.11 / 0.17 | 207 / 198 mm | 3% (raw, 0.5M) |
| `dp_vit_lang`: language, history | 1500 | 1.0M | 3% / 2% | 35% / 29% | 0.13 / 0.04 | 158 / 176 mm | 3% (ema, 0.4M) |
| `dp_vit_reactive`: no language, no history | 1500 | 1.0M | 1% / 1% | 26% / 34% | 0.01 / 0.04 | 175 / 173 mm | 2% (ema, 0.1M) |
| `dp_vit_lang_reactive`: language, no history | 1500 | 1.0M | 3% / 5% | 49% / 41% | 0.29 / 0.18 | 127 / 139 mm | 5% (ema, 1.0M) |
| same | 1500 | 3.0M | 64% / 62% | 99% / 100% | 0.96 / 0.97 | 11 / 11 mm | 70% (ema, 3.0M) |
| same | 4000 | 5.0M | 83% / 85% | 100% / 100% | 0.98 / 0.98 | 7 / 7 mm | 86% (raw, 4.0M) |
| `dp_vit_lang`: language, history | 4000 | 3.0M (stopped) | 2% / 2% | 30% / 35% | 0.11 / 0.08 | 183 / 164 mm | 3% (raw, 2.0M) |

The three rows of `dp_vit_lang_reactive` differ only in the number of demonstrations and of training
samples. The learning-rate schedule (cosine to zero) is stretched over each run's own budget.

### 4000 demonstrations, no history (`dp_vit_lang_reactive_pt4k`)

| Samples | Success (raw / EMA) | Closed at the named object | Slope | Median miss to named |
|---|---|---|---|---|
| 0.5M | 0% / 0% | 34% / 26% | 0.02 / -0.06 | 166 / 167 mm |
| 1.0M | 0% / 1% | 35% / 32% | 0.11 / 0.14 | 138 / 154 mm |
| 1.5M | 4% / 2% | 61% / 49% | 0.47 / 0.37 | 102 / 104 mm |
| 2.0M | 4% / 36% | 95% / 98% | 0.81 / 0.96 | 32 / 18 mm |
| 2.5M | 56% / 70% | 98% / 100% | 0.94 / 0.98 | 14 / 13 mm |
| 3.0M | 81% / 75% | 100% / 99% | 0.98 / 0.97 | 9 / 9 mm |
| 3.5M | 83% / 82% | 100% / 100% | 0.99 / 0.99 | 9 / 8 mm |
| 4.0M | 86% / 80% | 100% / 100% | 1.00 / 0.98 | 8 / 8 mm |
| 4.5M | 81% / 84% | 100% / 100% | 0.99 / 0.99 | 7 / 7 mm |
| 5.0M | 81% / 82% | 100% / 100% | 0.98 / 0.99 | 8 / 8 mm |
| 5.0M (final) | 83% / 85% | 100% / 100% | 0.98 / 0.98 | 7 / 7 mm |


Per object at 5.0M (EMA): can 26/28, cube 20/24, hex 24/25, tblock 15/23. No episode lifted an object other than the named one.

A second set of 100 layouts (seeds 1,000,100-1,000,199), same checkpoint: 83/100.

### 1500 demonstrations, no history, 3M samples (`dp_vit_lang_reactive_pt1500_3M`)

| Samples | Success (raw / EMA) | Closed at the named object | Slope | Median miss to named |
|---|---|---|---|---|
| 0.3M | 1% / 0% | 31% / 31% | 0.00 / 0.02 | 197 / 175 mm |
| 0.6M | 1% / 0% | 36% / 26% | 0.06 / 0.03 | 180 / 176 mm |
| 0.9M | 1% / 0% | 35% / 36% | 0.07 / 0.06 | 202 / 164 mm |
| 1.2M | 3% / 0% | 56% / 38% | 0.52 / 0.14 | 99 / 152 mm |
| 1.5M | 21% / 18% | 94% / 87% | 0.95 / 0.88 | 29 / 46 mm |
| 1.8M | 44% / 43% | 98% / 97% | 1.00 / 0.95 | 17 / 17 mm |
| 2.1M | 59% / 50% | 99% / 100% | 0.98 / 0.97 | 12 / 12 mm |
| 2.4M | 60% / 67% | 100% / 100% | 0.97 / 0.97 | 11 / 11 mm |
| 2.7M | 63% / 62% | 100% / 100% | 0.98 / 0.97 | 11 / 11 mm |
| 3.0M | 62% / 70% | 100% / 100% | 0.99 / 0.99 | 10 / 11 mm |
| 3.0M (final) | 64% / 62% | 99% / 100% | 0.96 / 0.97 | 11 / 11 mm |


### 1500 demonstrations, no history, 1M samples (`dp_vit_lang_reactive_pt1500`)

| Samples | Success (raw / EMA) | Closed at the named object | Slope | Median miss to named |
|---|---|---|---|---|
| 0.2M | 0% / 0% | 29% / 30% | -0.01 / -0.00 | 177 / 167 mm |
| 0.4M | 1% / 0% | 37% / 31% | 0.11 / -0.05 | 174 / 163 mm |
| 0.6M | 1% / 1% | 33% / 38% | 0.10 / 0.09 | 162 / 159 mm |
| 0.8M | 2% / 4% | 45% / 44% | 0.36 / 0.24 | 121 / 141 mm |
| 1.0M | 0% / 1% | 49% / 46% | 0.19 / 0.19 | 134 / 131 mm |
| 1.0M (final) | 3% / 5% | 49% / 41% | 0.29 / 0.18 | 127 / 139 mm |


### 4000 demonstrations, with history (`dp_vit_lang_pt4k`)

| Samples | Success (raw / EMA) | Closed at the named object | Slope | Median miss to named |
|---|---|---|---|---|
| 0.5M | 0% / 0% | 30% / 34% | -0.06 / -0.05 | 240 / 178 mm |
| 1.0M | 1% / 1% | 31% / 32% | 0.03 / 0.04 | 185 / 187 mm |
| 1.5M | 1% / 1% | 33% / 33% | 0.10 / 0.07 | 184 / 173 mm |
| 2.0M | 3% / 1% | 34% / 26% | 0.11 / 0.06 | 169 / 177 mm |
| 2.5M | 1% / 1% | 32% / 30% | 0.06 / 0.03 | 183 / 186 mm |
| 3.0M | 2% / 2% | 30% / 35% | 0.11 / 0.08 | 183 / 164 mm |


The run was stopped at 3.0M samples.

### What the history policy depends on

Same checkpoint throughout (2.5M samples, EMA), 100 layouts each:

| Condition | Success | Closed at the named object | Median miss to named |
|---|---|---|---|
| as served: the previous action sent with the first query is the hold command | 1/100 | 30% | 186 mm |
| the previous action sent with the first query is the expert's first command | 22/100 | 97% | 25 mm |
| the expert drives the first 10 steps, then the policy takes over | 77/100 | 100% | 12 mm |

In a training window that starts at the first frame of a demonstration there is no previous action;
the slot is filled with a copy of the first action, which is the expert's first command. When the
policy is served, the slot holds the command the robot is actually under before it moves (hold).

### What the no-history policy depends on

`dp_vit_lang_reactive_pt4k` at 5.0M samples (EMA), 100 layouts each:

| Condition | Success | Closed at the named object | Median miss to named | |
|---|---|---|---|---|
| as trained | 85/100 | 100% | 7 mm | |
| the first frame's images are sent for the whole episode | 1/100 | 99% | 17 mm | |
| the expert drives the first 10 steps toward another object, then the policy takes over | 12/100 | 74% | 64 mm | closed at the object the expert was heading for: 16/100 |
| the expert drives the first 25 steps toward another object, then the policy takes over | 0/100 | 0% | 209 mm | closed at the object the expert was heading for: 100/100 |

### Disturbances

The named object is moved once per episode, 80 mm in a random direction unless stated, and turned by up to 0.5 rad (`eval.py --disturb`). `dp_vit_lang_reactive_pt4k` at 5.0M samples (EMA); it was trained on demonstrations in which nothing moves. Counts are over the episodes in which the object could be moved (a free place existed).

| The object is moved | Success | Gripper closed, median distance from the old place | from the new place | Second grasp attempt |
|---|---|---|---|---|
| at step 10 of the reach | 9/100 | 39 mm | 49 mm | 43/100 |
| at step 20 | 0/96 | 8 mm | 78 mm | 31/96 |
| at step 30 | 0/99 | 7 mm | 81 mm | 28/99 |
| at step 20, by 40 mm | 1/17 | 18 mm | 35 mm | 9/17 |
| at step 20, by 120 mm | 0/99 | 9 mm | 116 mm | 35/99 |
| at the moment the gripper first closes | 1/99 | - | - | 29/99 |

The expert closes the gripper around step 40. The scripted expert under the same disturbances, 40
layouts each: with its one plan per episode 1/40 (moved at step 15) and 0/40 (moved at the grasp);
planning again when the object has moved or the fingers met on nothing, 40/40 and 40/40, with a
second grasp attempt in every episode of the second case.

### Inference settings

`dp_vit_lang_reactive_pt4k` at 5.0M samples (EMA). Each cell is 100 layouts; where two numbers are given the second is the second set of 100 layouts.

| Flow steps (8 actions executed per chunk) | Success | Median latency per chunk |
|---|---|---|
| 1 | 61/100 | 21 ms |
| 2 | 89/100, 90/100 | 25 ms |
| 4 | 88/100, 86/100 | 32 ms |
| 10 | 85/100, 83/100 | 71 ms |
| 20 | 79/100 | 87 ms |

| Actions executed per chunk (10 flow steps) | Success | T block | With the same starting noise for every chunk of an episode |
|---|---|---|---|
| 2 | 76/100 | 8/23 | 39/100 |
| 4 | 84/100 | 11/23 | 55/100 |
| 8 | 85/100 | 15/23 | 80/100 |
| 15 (the whole chunk) | 84/100 | 16/23 | - |

Latencies were measured while the same GPU was training another model.

### The failures that remain

`dp_vit_lang_reactive_pt4k`, checkpoints from 3.0M samples on (raw and EMA, 12 evaluations of 100 layouts pooled):

| Object | Success | Median miss, successes | Median miss, failures | Median yaw error, successes | Median yaw error, failures | Median height above the grasp point at closing, failures |
|---|---|---|---|---|---|---|
| green can | 311/336 = 93% | 5 mm | 12 mm | - | - | 3 mm |
| blue hex nut | 283/300 = 94% | 7 mm | 30 mm | - | - | -0 mm |
| red cube | 236/288 = 82% | 8 mm | 16 mm | 20 deg | 32 deg | -1 mm |
| yellow T block | 153/276 = 55% | 9 mm | 15 mm | 35 deg | 64 deg | 9 mm |

Yaw error is measured against the grasp the expert uses (cube: any of four, T block: across the stem,
either way round). The can and the hex nut can be grasped at any yaw.

### Single cube (`pick_cube`, 1000 demonstrations, 1.5M samples)

No history (`dp_vit_lang_reactive_cube1k`):

| Samples | Success (raw / EMA) | Closed at the named object | Slope | Median miss to named |
|---|---|---|---|---|
| 0.3M | 13% / 54% | 100% / 100% | 0.96 / 0.94 | 40 / 16 mm |
| 0.6M | 89% / 84% | 100% / 100% | 0.98 / 0.99 | 11 / 9 mm |
| 0.9M | 77% / 94% | 100% / 100% | 1.00 / 0.99 | 10 / 7 mm |
| 1.2M | 96% / 94% | 100% / 100% | 1.00 / 0.99 | 6 / 7 mm |
| 1.5M | 91% / 91% | 100% / 100% | 0.99 / 1.00 | 7 / 6 mm |
| 1.5M (final) | 95% / 91% | 100% / 100% | 1.00 / 0.99 | 7 / 6 mm |


With history (`dp_vit_cube1k`, no language input; the task has one instruction):

| Samples | Success (raw / EMA) | Closed at the named object | Slope | Median miss to named |
|---|---|---|---|---|
| 0.3M | 11% / 38% | 100% / 100% | 0.93 / 1.00 | 40 / 24 mm |
| 0.6M | 57% / 45% | 100% / 100% | 1.04 / 1.04 | 16 / 16 mm |
| 0.9M | 73% / 56% | 100% / 100% | 1.04 / 1.04 | 13 / 15 mm |
| 1.2M | 65% / 68% | 100% / 100% | 1.04 / 1.04 | 12 / 14 mm |
| 1.5M | 66% / 69% | 100% / 100% | 1.03 / 1.04 | 12 / 13 mm |
| 1.5M (final) | 62% / 70% | 100% / 100% | 1.03 / 1.03 | 14 / 14 mm |


### Which demonstrations (1500 episodes, 1M samples)

`dp_vit_lang_reactive` throughout. "Every object named": 500 layouts, each demonstrated three times,
once per object. "Early windows repeated": the windows that start in frames 0-7 of each demonstration
are in the training set 6 times instead of once (45% of all windows).

| Data | Success (raw / EMA) | Closed at the named object | Slope | Median miss to named | Best checkpoint |
|---|---|---|---|---|---|
| one instruction per layout (1500 layouts) | 3% / 5% | 49% / 41% | 0.29 / 0.18 | 127 / 139 mm | 5% (ema, 1.0M) |
| every object named | 0% / 1% | 32% / 28% | 0.04 / -0.10 | 151 / 188 mm | 3% (raw, 0.9M) |
| every object named, early windows repeated | 8% / 15% | 52% / 58% | 0.27 / 0.32 | 113 / 110 mm | 15% (ema, 1.0M) |

### First action chunk, open loop

Open-loop test of the first action chunk (15 steps) from the first frame, 60 layouts,
`dp_vit_lang_reactive` at 0.9M samples:

| | Training layouts | Unseen layouts |
|---|---|---|
| Expert's chunk heads for the named object | 100% | 100% |
| Policy's chunk heads for the named object | 35% | 48% |
| Distance between the policy's and the expert's chunk endpoint (median) | 12.1 cm | 11.2 cm |
| Distance between two samples of the policy for the same input (median) | 10.9 cm | 10.8 cm |

The expert moves the gripper 30 cm in that chunk. The no-language policy gives the same picture
(32-42% heading for the named object, endpoints 12.9 cm from the expert's).

### From-scratch stages on machine B

Language model final loss 4.15. VLM final loss 0.058; held-out captioning check 249/300 = 83% exact
ordering and 279/300 = 93% leftmost object. Both were trained before the position fix; the language
model's sequences are 256 tokens long and are not affected by it, the VLM's are 320 tokens and are.

### Runs in progress when this was written

With the position fix: the 4000-demo policy again (the comparison with the run above), the same with
the early windows repeated, with every object named, with demonstrations that contain recoveries, with
random start poses, with both, a history policy on demonstrations that begin with a hold frame, the
single-cube controls, and the from-scratch VLM and VLA (pretrained and random-init backbone). Before
the fix: the Qwen VLA on 1500 demos.

## Throughput

| Model | Machine A (RTX 4090) | Machine B (L40S) |
|---|---|---|
| `dp_vit` / `dp_vit_lang` (4 images) | about 620 samples/s | about 540 samples/s |
| `dp_vit_reactive` / `dp_vit_lang_reactive` (2 images) | not run | about 1100 samples/s alone on a GPU; 570-700 while 4-5 policy servers evaluate on the same GPU |
| `vla_custom` (130M) | 165 samples/s | not yet run |
| `vla_qwen` (2.2B, 306M trainable, batch 16) | 49 samples/s | 45-48 samples/s |
| Language model stage (91M, batch 32, 256 tokens) | 550 samples/s | 520-545 samples/s |

Policy inference per chunk: 20-65 ms for the 36M policy, about 190 ms for the Qwen VLA (machine A).

## Operations

- Without a thread cap, each policy server on the 128-core machine started 256 threads and used
  14-22 cores; twelve of them pushed the load average to 570 and roughly halved training throughput
  (for example 1069 -> 410 samples/s). With `OMP_NUM_THREADS=4` a server has 12 threads and uses
  under one core.
- On a multi-GPU machine the EGL device order differed from `nvidia-smi`'s (two devices in the
  middle of EGL's list were GPUs 0 and 1). MuJoCo 4.0.1 orders EGL devices by
  `CUDA_VISIBLE_DEVICES`, and a render with that variable set to one GPU used that GPU only.
- `uv pip install -e .` after `uv sync` replaced the locked `draccus` 0.11.5 with 0.11.6.
- Collecting 1500 demonstrations took 70 s with 48 simulator processes; 4000 with recoveries took
  272 s with 40.
- Two evaluations started on the same port both talked to whichever policy server had bound it
  first, and the second one filed the other run's results under its own name: about 40 evaluations here.
  They were found by comparing the checkpoint path each result recorded with its directory, deleted
  and redone. No number in this file comes from them.

## Bugs found and fixed

| Problem | Effect | Fix |
|---|---|---|
| `SimpleVLMProcessor` scaled PIL and numpy images to [0, 1] but left uint8 tensors at 0-255; training fed tensors | the VLM ignored images at inference (captioning 0%) | scale uint8 tensors too; test added |
| Action token appended after the batch padding (`vlm_foundry_backbone`) | a short instruction is probed one position later, behind a padding token, in a training batch than when served alone | insert it after the last real token; test added. Effect on success not measured |
| `lowdim_past_timesteps or default` treats 0 as unset | with 0 past steps the proprioception input contains the state one step in the future | explicit `is None` check; test added |
| Both VLA recipes condition the action head on one vector | see the `pick_target` tables | `conditioning: all_tokens` for the from-scratch VLM backbone; effect not yet measured |
| A dead data-loader worker leaves the trainer waiting forever | runs hang at 0% GPU | `train_resilient.sh` restarts on exit or on a stalled log |
| The rotary position table was computed in bfloat16 under autocast, which keeps 8 significant bits | positions above 256 collapse onto even numbers, above 512 onto multiples of 4: 641 distinct positions out of 2048. In the no-history policy (425 tokens before the actions) 15 action tokens shared 8 positions; with history (about 820) 16 shared 4 | the table is computed in float32 and then cast; test added. Effect on success: comparison in progress |
| Evaluations shared a port (see Operations) | results filed under the wrong run | `eval.sh` takes a free port and waits for its own server; the client stops unless the server reports the checkpoint it asked for; `summarize.py` leaves out results whose recorded checkpoint does not match |
| The previous-action input of a window at the start of a demonstration is filled with a copy of the first action | the expert's first command, which already points at the target, is an input in training and absent when served (see "What the history policy depends on") | `collect.py --hold_steps 1` begins every demonstration with a frame of the robot at rest; test of the effect in progress |
