# Lab notebook

Every number measured in this repository, in one place. Chapters cite this file; if a number is not
here, it was not measured here. All success rates are closed-loop on layouts the policy never saw
(evaluation seeds `1_000_000+`), one training seed per run, Wilson 95% intervals. Differences of a
few percent between single runs are noise.

Two machines were used. **Machine A** has one 24 GB GPU (RTX 4090). **Machine B** has 46 GB GPUs
(L40S) and 128 CPU cores. Machine A was unreliable; its runs died or hung several times and were
resumed from checkpoints.

Every run up to and including the section "`pick_target`, second round" was trained with a position
table computed in half precision (see the bug table), on both machines. The runs trained with the
corrected table are in the section "After the position fix" and nowhere else.

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
| `pick_target_4k_rs_recover_early` | 4000 | 435,550 | `pick_target_4k_rs_recover` with the windows that start in frames 0-7 present 6 times |
| `pick_target_4k_hold` | 4000 | 267,075 | same seeds; every episode begins with one frame of the robot at rest |

How each was made (`sim/pickup/collect.py` options through `make_dataset.sh`): `*_cf*` with
`--targets_per_layout 3`; `*_recover` with `--disturb_prob 0.5`; `*_rs*` with `--random_start`;
`*_hold` with `--hold_steps 1`; `*_early` from the dataset named, with
`sim/pickup/boost_early_windows.py --max_frame 7 --repeat 5`.

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

Open loop, on recorded samples (`open_loop_check.py`): the served policies reproduce the expert's
action chunks to within 0.01-0.04 rad per joint; copying the previous action is off by 0.05-0.2 rad.
The pooled-CLIP policy has the lowest training loss and the lowest open-loop error of the four.

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

All policies in this section are the patch-token policy: a flow-matching action transformer on the patch
tokens of a small ViT trained from scratch, 36.3M parameters (`dp_vit`, `dp_vit_reactive`). "Language"
adds the instruction as tokens and, with their embedding table (49,280 x 256), 12.6M parameters: 48.9M
(`dp_vit_lang`, `dp_vit_lang_reactive`). "History" =
frames t-1 and t, states t-1 and t, and the previous action as inputs. "No history" (`*_reactive`) =
one frame per camera, the current state, no previous action. Every evaluation is 100 unseen layouts,
8 actions executed per chunk, 10 flow steps, unless stated. "raw" = the trained weights, "EMA" = their
running average, saved by the same run.

**Slope**: least-squares slope of where the gripper first closed against where the named object was
(both table axes pooled). 1: it closes where the object is. 0: where it closes does not depend on the
scene. **Closed at the named object**: the object nearest the gripper when it first closed was the
named one (chance 33%).

Runs marked "position fix" were trained after the position-table bug (see the bug table) was fixed;
the others before. Evaluations always use the code a run was trained with. Runs on machine B save 10
checkpoints over their budget (`PICKUP_NUM_CHECKPOINTS=10`) and one more at the end, which is why the
tables below have a row every tenth of the budget and a row marked "final".

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
samples. The learning-rate schedule (a cosine from 5e-4 down to 1e-5) is stretched over each run's own budget.

### Does the closing position depend on the scene?

All evaluations of a run pooled (22 evaluations of 100 layouts each; 12 for the last row).
Least-squares fit of where the gripper first closed against where the named object was, for each
table axis (x: away from the robot, y: sideways). A policy that closes where the named object is has
slope 1 and R² near 1.

| Run | Slope, x / y | R², x / y |
|---|---|---|
| `dp_vit_pt1500`: no language, history, 1M samples | 0.04 / 0.09 | 0.001 / 0.006 |
| `dp_vit_reactive_pt1500`: no language, no history, 1M samples | 0.00 / 0.01 | 0.000 / 0.000 |
| `dp_vit_lang_pt1500`: language, history, 1M samples | 0.05 / 0.06 | 0.002 / 0.003 |
| `dp_vit_lang_reactive_pt1500`: language, no history, 1M samples (its last checkpoints are at the start of the transition) | 0.03 / 0.15 | 0.001 / 0.021 |
| `dp_vit_lang_pt4k`: language, history, 4000 demonstrations, 3M samples | 0.08 / 0.04 | 0.004 / 0.002 |

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

The expert closes the gripper around step 40. The scripted expert under the same disturbances is
in the section after the position fix (100 layouts per condition): with its one plan per episode it
fails as the policy does; planning again it succeeds every time.

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
| 0.1M | 3% / 8% | 100% / 100% | 0.62 / 0.70 | 91 / 54 mm |
| 0.3M | 13% / 54% | 100% / 100% | 0.96 / 0.94 | 40 / 16 mm |
| 0.4M | 46% / 70% | 100% / 100% | 0.94 / 0.99 | 16 / 11 mm |
| 0.6M | 89% / 84% | 100% / 100% | 0.98 / 0.99 | 11 / 9 mm |
| 0.7M | 83% / 94% | 100% / 100% | 1.00 / 0.99 | 10 / 8 mm |
| 0.9M | 77% / 94% | 100% / 100% | 1.00 / 0.99 | 10 / 7 mm |
| 1.0M | 94% / 97% | 100% / 100% | 1.00 / 1.00 | 6 / 7 mm |
| 1.2M | 96% / 94% | 100% / 100% | 1.00 / 0.99 | 6 / 7 mm |
| 1.3M | 94% / 95% | 100% / 100% | 1.00 / 1.00 | 6 / 6 mm |
| 1.5M | 91% / 91% | 100% / 100% | 0.99 / 1.00 | 7 / 6 mm |
| 1.50M (final) | 95% / 91% | 100% / 100% | 1.00 / 0.99 | 7 / 6 mm |


With history (`dp_vit_cube1k`, no language input; the task has one instruction). The two runs differ
in more than history: this one has no instruction tokens and is fed pixels in 0-255 where the other
gets them scaled. Trained again after the position fix it ends at 96% / 98% (section "Single cube
with history, after the fix"), so what held it at 62-70% here was the position table, in which its 16
action-step tokens shared 4 positions, not history:

| Samples | Success (raw / EMA) | Closed at the named object | Slope | Median miss to named |
|---|---|---|---|---|
| 0.1M | 4% / 7% | 100% / 100% | 0.86 / 0.57 | 74 / 89 mm |
| 0.3M | 11% / 38% | 100% / 100% | 0.93 / 1.00 | 40 / 24 mm |
| 0.4M | 12% / 52% | 100% / 100% | 1.07 / 1.03 | 31 / 16 mm |
| 0.6M | 57% / 45% | 100% / 100% | 1.04 / 1.04 | 16 / 16 mm |
| 0.7M | 62% / 59% | 100% / 100% | 1.00 / 1.04 | 14 / 14 mm |
| 0.9M | 73% / 56% | 100% / 100% | 1.04 / 1.04 | 13 / 15 mm |
| 1.0M | 59% / 63% | 100% / 100% | 1.03 / 1.03 | 14 / 14 mm |
| 1.2M | 65% / 68% | 100% / 100% | 1.04 / 1.04 | 12 / 14 mm |
| 1.3M | 62% / 64% | 100% / 100% | 1.04 / 1.03 | 14 / 13 mm |
| 1.5M | 66% / 69% | 100% / 100% | 1.03 / 1.04 | 12 / 13 mm |
| 1.50M (final) | 62% / 70% | 100% / 100% | 1.03 / 1.03 | 14 / 14 mm |


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
ordering and 279/300 = 93% leftmost object. Both were trained before the position fix. The language
model's sequences are 256 tokens long: no two of its positions were merged (the old table rounded
their angles to 8 significant bits). The VLM's are 320 tokens long, and positions above 256 were merged
in pairs.
The VLM retrained after the fix (same data and budget): final loss 0.059, captioning check 267/300 =
89% exact ordering and 286/300 = 95% leftmost object.

## After the position fix (machine B)

Everything in this section was trained with the rotary position table computed in float32 (see the
bug table). Same policy as above unless stated (`dp_vit_lang_reactive`: patch tokens, instruction
tokens, no history), 4000 demonstrations, 5M samples, 100 unseen layouts per evaluation.

### The same run before and after the fix

Trained weights (the run after the fix did not save averaged weights). Before: `dp_vit_lang_reactive_pt4k`. After: `dp_vit_lang_reactive_pt4k_posfix`.

| Samples | Success before / after | Closed at the named object | Slope | Median miss to named |
|---|---|---|---|---|
| 0.5M | 0% / 0% | 34% / 36% | 0.02 / 0.03 | 166 / 177 mm |
| 1.0M | 0% / 8% | 35% / 65% | 0.11 / 0.58 | 138 / 80 mm |
| 1.5M | 4% / 58% | 61% / 98% | 0.47 / 0.94 | 102 / 14 mm |
| 2.0M | 4% / 40% | 95% / 98% | 0.81 / 0.97 | 32 / 16 mm |
| 2.5M | 56% / 77% | 98% / 99% | 0.94 / 0.90 | 14 / 10 mm |
| 3.0M | 81% / 90% | 100% / 99% | 0.98 / 0.96 | 9 / 7 mm |
| 3.5M | 83% / 88% | 100% / 100% | 0.99 / 0.98 | 9 / 6 mm |
| 4.0M | 86% / 94% | 100% / 100% | 1.00 / 0.99 | 8 / 5 mm |
| 4.5M | 81% / 92% | 100% / 100% | 0.99 / 0.99 | 7 / 6 mm |
| 5.0M | 83% / 93% | 100% / 99% | 0.98 / 0.98 | 7 / 6 mm |

Per object at the final checkpoint, before / after: can 26/28 / 25/28, cube 18/24 / 23/24, hex 24/25 / 25/25, tblock 15/23 / 20/23.

### What the demonstrations contain

Success with trained / averaged weights (a dash where a run did not save averaged weights), at fixed points of training.

| Demonstrations | 0.5M | 1.0M | 1.5M | 2.0M | 3.0M | 4.0M | 5.0M | Best checkpoint |
|---|---|---|---|---|---|---|---|---|
| one instruction per layout (`pick_target_4k`) | 0% / - | 8% / - | 58% / - | 40% / - | 90% / - | 94% / - | 93% / - | 94% (raw, 4.0M) |
| early windows repeated (`pick_target_4k_early`) | 1% / - | 5% / - | 76% / - | 73% / - | 96% / - | 96% / - | 96% / - | 97% (raw, 5.0M) |
| every object named (`pick_target_cf4k`) | 1% / 2% | 41% / 38% | 74% / 80% | 73% / 84% |  |  |  | 92% (ema, 2.5M), still training |
| recoveries (`pick_target_4k_recover`) | 0% / 12% | 38% / 79% | 79% / 80% | 85% / 87% | 83% / 85% | 86% / 88% | 89% / 84% | 92% (ema, 4.5M) |
| random starts (`pick_target_4k_rs`) | 0% / 1% | 56% / 76% | 57% / 79% | 71% / 84% | 94% / 98% | 98% / 98% | 98% / 98% | 99% (ema, 5.0M) |
| random starts and recoveries (`pick_target_4k_rs_recover`) | 12% / 54% | 68% / 86% | 86% / 91% | 93% / 84% | 90% / 93% |  |  | 93% (raw, 2.0M), still training |
| random starts, recoveries, early windows repeated (`pick_target_4k_rs_recover_early`) | 5% / 13% |  |  |  |  |  |  | 13% (ema, 0.5M), still training |

Closed at the named object and slope at the same points (trained weights, or averaged where only those exist):

| Demonstrations | 0.5M | 1.0M | 1.5M | 2.0M | 3.0M |
|---|---|---|---|---|---|
| one instruction per layout (`pick_target_4k`) | 36%, 0.03 | 65%, 0.58 | 98%, 0.94 | 98%, 0.97 | 99%, 0.96 |
| early windows repeated (`pick_target_4k_early`) | 58%, 0.49 | 99%, 0.93 | 100%, 1.00 | 100%, 1.02 | 99%, 1.01 |
| every object named (`pick_target_cf4k`) | 30%, -0.04 | 98%, 1.02 | 99%, 0.98 | 99%, 0.99 |  |
| recoveries (`pick_target_4k_recover`) | 33%, 0.11 | 81%, 0.96 | 100%, 0.97 | 100%, 1.00 | 100%, 0.99 |
| random starts (`pick_target_4k_rs`) | 46%, 0.18 | 91%, 0.81 | 96%, 0.94 | 87%, 0.84 | 98%, 0.95 |
| random starts and recoveries (`pick_target_4k_rs_recover`) | 50%, 1.00 | 100%, 1.02 | 100%, 1.00 | 100%, 1.00 | 100%, 0.99 |
| random starts, recoveries, early windows repeated (`pick_target_4k_rs_recover_early`) | 56%, 0.56 |  |  |  |  |

Success by object, pooled over each finished run's last three checkpoints (trained and, where saved, averaged weights):

| Demonstrations | Can | Cube | Hex nut | T block |
|---|---|---|---|---|
| before the fix, one instruction per layout | 156/168 = 93% | 122/144 = 85% | 144/150 = 96% | 74/138 = 54% |
| one instruction per layout | 76/84 = 90% | 68/72 = 94% | 74/75 = 99% | 60/69 = 87% |
| early windows repeated | 79/84 = 94% | 72/72 = 100% | 72/75 = 96% | 64/69 = 93% |
| recoveries | 130/168 = 77% | 141/144 = 98% | 144/150 = 96% | 109/138 = 79% |
| random starts | 168/168 = 100% | 144/144 = 100% | 149/150 = 99% | 127/138 = 92% |

### How each policy copes outside the conditions it was trained in

Final checkpoint of each run (averaged weights where the run saved them), 100 layouts per cell. The object is moved 80 mm; counts are over the episodes in which it could be moved.

| Demonstrations | Standard | Second set of layouts | Random start | Moved at step 10 | Moved at step 20 | Moved at step 30 | Moved at the grasp | Random start, moved at step 20 | Random start, moved at the grasp |
|---|---|---|---|---|---|---|---|---|---|
| before the fix, plain demonstrations | 85/100 | 83/100 | 6/100 | 9/100 | 0/96 | 0/99 | 1/99 (29 retried) | 0/98 | 0/98 (41 retried) |
| one instruction per layout | 93/100 | 96/100 | 9/100 | 12/100 | 0/97 | 0/100 | 0/100 (8 retried) | 0/99 | 0/99 (2 retried) |
| early windows repeated | 96/100 | 100/100 | 9/100 | 10/100 | 0/100 | 0/100 | 0/100 (3 retried) | 0/100 | 0/100 (7 retried) |
| recoveries | 84/100 | 93/100 | 68/100 | 88/100 | 79/96 | 85/100 | 78/99 (99 retried) | 70/100 | 65/81 (78 retried) |
| random starts | 98/100 | 97/100 | 97/100 | 94/100 | 50/98 | 0/100 | 0/100 (2 retried) | 15/99 | 0/100 (1 retried) |

The scripted expert under the same conditions, 100 layouts each:

| Expert | Standard | Random start | Moved at step 10 | Moved at step 20 | Moved at step 30 | Moved at the grasp | Random start, moved at step 20 | Random start, moved at the grasp |
|---|---|---|---|---|---|---|---|---|
| one plan per episode | 100/100 | 100/100 | 0/100 | 0/99 | 0/100 | 0/100 | | |
| plans again | 100/100 | 100/100 | 100/100 | 96/96 | 100/100 | 100/100 | 98/98 | 100/100 |

What the same policies depend on:

| Demonstrations | Cameras frozen on the first frame | Expert drives 10 steps toward another object | Expert drives 25 steps toward another object | 2 flow steps |
|---|---|---|---|---|
| before the fix, plain demonstrations | 1/100 (at the named object 99%) | 12/100 (closed at the other object 16/100) | 0/100 (closed at the other object 100/100) | 89/100 |
| one instruction per layout | 1/100 (at the named object 94%) | 18/100 (closed at the other object 24/100) | 0/100 (closed at the other object 100/100) | 94/100 |
| early windows repeated | 0/100 (at the named object 70%) | 16/100 (closed at the other object 28/100) | 0/100 (closed at the other object 100/100) | 77/100 |
| recoveries | 0/100 (at the named object 3%) | 79/100 (closed at the other object 1/97) | 56/100 (closed at the other object 28/80) | 84/100 |
| random starts | 5/100 (at the named object 98%) | 98/100 (closed at the other object 0/100) | 14/100 (closed at the other object 57/98) | 98/100 |

### History, on demonstrations that begin with a hold frame (`dp_vit_lang_pt4k_hold_posfix`)

| Samples | Success (raw / EMA) | Closed at the named object | Slope | Median miss to named |
|---|---|---|---|---|
| 0.5M | 0% / 2% | 35% / 41% | 0.09 / 0.09 | 148 / 152 mm |
| 1.0M | 2% / 1% | 34% / 30% | -0.06 / 0.03 | 156 / 163 mm |
| 1.5M | 1% / 1% | 34% / 34% | -0.02 / -0.04 | 159 / 157 mm |
| 2.0M | 1% / 1% | 41% / 32% | 0.11 / -0.06 | 141 / 152 mm |
| 2.5M | 2% / 2% | 43% / 38% | 0.12 / 0.06 | 149 / 151 mm |
| 3.0M | 1% / 0% | 31% / 39% | -0.11 / -0.02 | 181 / 168 mm |

Still training when this was written: the table holds the checkpoints evaluated so far.


### Single cube with history, after the fix (`dp_vit_cube1k_posfix`)

Before the fix the same configuration ended at 62% / 70% (table above).

| Samples | Success (raw / EMA) | Closed at the named object | Slope | Median miss to named |
|---|---|---|---|---|
| 0.15M | 1% / 2% | 91% / 89% | 0.22 / 0.08 | 117 / 164 mm |
| 0.30M | 30% / 37% | 99% / 100% | 0.93 / 0.96 | 27 / 23 mm |
| 0.45M | 29% / 65% | 100% / 100% | 1.01 / 0.98 | 23 / 15 mm |
| 0.60M | 52% / 86% | 100% / 100% | 0.98 / 1.00 | 20 / 12 mm |
| 0.75M | 89% / 87% | 100% / 100% | 0.99 / 0.99 | 9 / 10 mm |
| 0.89M | 92% / 91% | 100% / 100% | 0.98 / 0.99 | 9 / 8 mm |
| 1.0M | 93% / 93% | 100% / 100% | 0.99 / 0.99 | 9 / 8 mm |
| 1.2M | 96% / 97% | 100% / 100% | 0.99 / 0.98 | 7 / 7 mm |
| 1.3M | 98% / 97% | 100% / 100% | 0.99 / 0.99 | 7 / 6 mm |
| 1.5M | 99% / 98% | 100% / 100% | 0.99 / 1.00 | 7 / 6 mm |
| 1.5M (final) | 96% / 98% | 100% / 100% | 0.99 / 0.99 | 8 / 7 mm |


### Single cube without history, after the fix (`dp_vit_lang_reactive_cube1k_posfix`)

| Samples | Success (raw) | Closed at the named object | Slope | Median miss to named |
|---|---|---|---|---|
| 0.15M | 1% | 100% | 0.68 | 77 mm |
| 0.30M | 28% | 100% | 1.00 | 29 mm |
| 0.45M | 12% | 100% | 1.05 | 26 mm |
| 0.60M | 70% | 100% | 0.97 | 13 mm |
| 0.75M | 90% | 100% | 1.01 | 9 mm |
| 0.89M | 81% | 100% | 0.97 | 11 mm |
| 1.0M | 94% | 100% | 1.00 | 5 mm |
| 1.2M | 96% | 100% | 0.99 | 6 mm |
| 1.3M | 96% | 100% | 0.99 | 6 mm |
| 1.5M | 97% | 100% | 0.99 | 6 mm |
| 1.5M (final) | 91% | 100% | 0.99 | 6 mm |


### From-scratch VLA, every token, no history (`vla_custom_tokens_reactive`, 130M parameters, 3M samples)

Backbone pretrained on captions (stage 2, retrained with the fix):

| Samples | Success (raw / EMA) | Closed at the named object | Slope | Median miss to named |
|---|---|---|---|---|
| 0.25M | 5% / 43% | 81% / 97% | 0.87 / 0.97 | 64 / 21 mm |
| 0.5M | 31% / 50% | 99% / 100% | 0.95 / 1.01 | 29 / 19 mm |
| 0.75M | 30% / 60% | 100% / 100% | 1.02 / 0.98 | 22 / 15 mm |
| 1.0M | 73% / 62% | 99% / 99% | 0.98 / 0.99 | 13 / 13 mm |
| 1.2M | 64% / 82% | 100% / 100% | 1.00 / 0.99 | 13 / 10 mm |

Still training when this was written: the table holds the checkpoints evaluated so far.


The same architecture with a randomly initialised backbone:

| Samples | Success (raw / EMA) | Closed at the named object | Slope | Median miss to named |
|---|---|---|---|---|
| 0.25M | 0% / 0% | 33% / 28% | -0.09 / -0.04 | 177 / 184 mm |
| 0.5M | 0% / 1% | 39% / 33% | 0.04 / 0.09 | 178 / 159 mm |

Still training when this was written: the table holds the checkpoints evaluated so far.


### Qwen3-VL-2B backbone

One action-token vector, with history, 1500 demonstrations (`vla_qwen_pt1500`, trained before the fix; its action head has 20 tokens and was not affected):

| Samples | Success (raw) | Closed at the named object | Slope | Median miss to named |
|---|---|---|---|---|
| 0.10M | 1% | 33% | 0.04 | 175 mm |
| 0.20M | 3% | 60% | 0.53 | 105 mm |
| 0.30M | 2% | 60% | 0.62 | 98 mm |
| 0.40M | 5% | 71% | 0.68 | 83 mm |


Every token, no history, 4000 demonstrations (`vla_qwen_tokens_reactive_pt4k_posfix`):

| Samples | Success (raw / EMA) | Closed at the named object | Slope | Median miss to named |
|---|---|---|---|---|
| 0.05M | 1% / 1% | 40% / 35% | 0.26 / 0.08 | 159 / 149 mm |
| 0.10M | 6% / 12% | 71% / 80% | 0.76 / 0.83 | 94 / 73 mm |

Still training when this was written: the table holds the checkpoints evaluated so far.


### Runs in progress when this was written

Still training or being evaluated, with no final result here: every object named on 4000 demonstrations; random starts and recoveries together; the same with the early windows repeated; the history policy on demonstrations that begin with a hold frame; the plain run again with another seed, with and without the fix; the from-scratch VLA with a pretrained and with a randomly initialised backbone; the Qwen backbone with every token. Tables and rows above that say 'still training' show the checkpoints evaluated so far.

## Throughput

| Model | Machine A (RTX 4090) | Machine B (L40S) |
|---|---|---|
| `dp_vit` / `dp_vit_lang` (4 images) | about 620 samples/s | about 540 samples/s |
| `dp_vit_reactive` / `dp_vit_lang_reactive` (2 images) | not run | about 1100 samples/s alone on a GPU; 570-700 while 4-5 policy servers evaluate on the same GPU |
| `vla_custom` (130M) | 165 samples/s | not yet run |
| `vla_custom_tokens_reactive` (130M, every token, 2 images) | not run | about 260 samples/s; about 190 while its checkpoints are evaluated on the same GPU |
| `vla_qwen` (2.2B, 306M trainable, batch 16) | 49 samples/s | 45-48 samples/s |
| `vla_qwen_tokens_reactive` (2.2B, every token, 2 images, batch 16) | not run | about 62 samples/s; about 35 while its checkpoints are evaluated on the same GPU |
| Language model stage (91M, batch 32, 256 tokens) | 550 samples/s | 520-545 samples/s |

Wall time on machine B, one GPU per run, with every checkpoint evaluated on the same GPU as it appeared: the 5M-sample runs of
`dp_vit_lang_reactive` took 77 to 87 minutes each; the 1.5M-sample single-cube runs took 33 minutes
without history and 53 with.

Policy inference per chunk: 20-65 ms for the 36M policy (`dp_vit`), about 190 ms for the Qwen VLA (machine A).

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
| Action token appended after the batch padding (`vlm_foundry_backbone`, and the Hugging Face wrapper `vlm_backbone`) | a short instruction is probed one position later, behind a padding token, in a training batch than when served alone | insert it after the last real token; test added. Effect on success not measured |
| `lowdim_past_timesteps or default` treats 0 as unset | with 0 past steps the proprioception input contains the state one step in the future | explicit `is None` check; test added |
| Both VLA recipes condition the action head on one vector | see the `pick_target` tables | `conditioning: all_tokens` for both VLM backbones (`vla_custom_tokens_reactive.yaml`, `vla_qwen_tokens_reactive.yaml`); see the from-scratch VLA and the Qwen backbone in "After the position fix" |
| A dead data-loader worker leaves the trainer waiting forever | runs hang at 0% GPU | `train_resilient.sh` restarts on exit or on a stalled log |
| The rotary position table was computed in bfloat16 under autocast, which keeps 8 significant bits | positions above 256 collapse onto even numbers, above 512 onto multiples of 4: 641 distinct positions out of 2048. In the no-history policy (426 tokens before the actions: one for the noise level, 32 for the instruction, 392 patches, one for the joint state) the 15 action-step tokens, at positions 426-440, shared 8 positions; with history (about 820) 16 shared 4 | the table is computed in float32 and then cast; test added. Effect: see 'The same run before and after the fix' (one seed each; a second pair is running) and the single cube with history |
| Evaluations shared a port (see Operations) | results filed under the wrong run | `eval.sh` takes a free port and waits for its own server; the client stops unless the server reports the checkpoint it asked for; `summarize.py` leaves out results whose recorded checkpoint does not match |
| The previous-action input of a window at the start of a demonstration is filled with a copy of the first action | the expert's first command, which already points at the target, is an input in training and absent when served (see "What the history policy depends on") | `collect.py --hold_steps 1` begins every demonstration with a frame of the robot at rest; test of the effect in progress |
