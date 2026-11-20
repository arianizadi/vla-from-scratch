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

- Stage 1 trains on the end of each document: the tokenizer's published configuration truncates on
  the left, so of a document longer than 257 tokens the last 257 are kept.
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

Open-loop test of the first action chunk (15 steps) from the first frame of an episode; the chunk is
not executed. `dp_vit_lang_reactive_pt1500` (1500 demonstrations, the 1M-sample run) at 0.9M samples,
trained weights, 60 layouts per column: seeds 0-59, which are in its demonstrations, and seeds
1,000,000-1,000,059, which are not. The end point of a chunk is the gripper position that its last
joint target gives by forward kinematics. A chunk "heads for" the object whose direction from the
start pose is closest to the chunk's direction of travel in the table plane.

| | Training layouts | Unseen layouts |
|---|---|---|
| Expert's chunk heads for the named object | 100% | 100% |
| Policy's chunk heads for the named object | 35% | 48% |
| With another object of the layout named instead, the policy's chunk heads for that one | 42% | 43% |
| Distance between the policy's and the expert's chunk end point (median) | 12.1 cm | 11.2 cm |
| Distance between the policy's end points for the two instructions, one draw each (median) | 10.9 cm | 10.8 cm |

The expert moves the gripper 30 cm in that chunk. The policy without a language input
(`dp_vit_reactive_pt1500`, final checkpoint) gives the same picture: 32-42% heading for the named
object, end points 12.9 cm from the expert's.

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
| every object named (`pick_target_cf4k`) | 1% / 2% | 41% / 38% | 74% / 80% | 73% / 84% | 91% / 89% | 89% / 91% | 91% / 89% | 93% (raw, 3.5M) |
| recoveries (`pick_target_4k_recover`) | 0% / 12% | 38% / 79% | 79% / 80% | 85% / 87% | 83% / 85% | 86% / 88% | 89% / 84% | 92% (ema, 4.5M) |
| random starts (`pick_target_4k_rs`) | 0% / 1% | 56% / 76% | 57% / 79% | 71% / 84% | 94% / 98% | 98% / 98% | 98% / 98% | 99% (ema, 5.0M) |
| random starts and recoveries (`pick_target_4k_rs_recover`) | 12% / 54% | 68% / 86% | 86% / 91% | 93% / 84% | 90% / 93% | 93% / 89% | 94% / 94% | 94% (raw, 4.5M) |
| random starts, recoveries, early windows repeated (`pick_target_4k_rs_recover_early`) | 5% / 13% | 58% / 60% | 49% / 82% | 84% / 84% | 80% / 83% | 85% / 87% | 87% / 86% | 87% (ema, 2.5M) |

Closed at the named object and slope at the same points (trained weights, or averaged where only those exist):

| Demonstrations | 0.5M | 1.0M | 1.5M | 2.0M | 3.0M |
|---|---|---|---|---|---|
| one instruction per layout (`pick_target_4k`) | 36%, 0.03 | 65%, 0.58 | 98%, 0.94 | 98%, 0.97 | 99%, 0.96 |
| early windows repeated (`pick_target_4k_early`) | 58%, 0.49 | 99%, 0.93 | 100%, 1.00 | 100%, 1.02 | 99%, 1.01 |
| every object named (`pick_target_cf4k`) | 30%, -0.04 | 98%, 1.02 | 99%, 0.98 | 99%, 0.99 | 100%, 1.00 |
| recoveries (`pick_target_4k_recover`) | 33%, 0.11 | 81%, 0.96 | 100%, 0.97 | 100%, 1.00 | 100%, 0.99 |
| random starts (`pick_target_4k_rs`) | 46%, 0.18 | 91%, 0.81 | 96%, 0.94 | 87%, 0.84 | 98%, 0.95 |
| random starts and recoveries (`pick_target_4k_rs_recover`) | 50%, 1.00 | 100%, 1.02 | 100%, 1.00 | 100%, 1.00 | 100%, 0.99 |
| random starts, recoveries, early windows repeated (`pick_target_4k_rs_recover_early`) | 56%, 0.56 | 93%, 0.95 | 97%, 1.00 | 100%, 0.98 | 100%, 1.00 |

Success by object, pooled over each finished run's last three checkpoints (trained and, where saved, averaged weights):

| Demonstrations | Can | Cube | Hex nut | T block |
|---|---|---|---|---|
| before the fix, one instruction per layout | 156/168 = 93% | 122/144 = 85% | 144/150 = 96% | 74/138 = 54% |
| one instruction per layout | 76/84 = 90% | 68/72 = 94% | 74/75 = 99% | 60/69 = 87% |
| early windows repeated | 79/84 = 94% | 72/72 = 100% | 72/75 = 96% | 64/69 = 93% |
| every object named | 164/168 = 98% | 138/144 = 96% | 148/150 = 99% | 88/138 = 64% |
| recoveries | 130/168 = 77% | 141/144 = 98% | 144/150 = 96% | 109/138 = 79% |
| random starts | 168/168 = 100% | 144/144 = 100% | 149/150 = 99% | 127/138 = 92% |
| random starts and recoveries | 163/168 = 97% | 138/144 = 96% | 146/150 = 97% | 106/138 = 77% |
| random starts, recoveries, early windows repeated | 115/168 = 68% | 126/144 = 88% | 147/150 = 98% | 123/138 = 89% |

### How each policy copes outside the conditions it was trained in

Final checkpoint of each run (averaged weights where the run saved them), 100 layouts per cell. The object is moved 80 mm; counts are over the episodes in which it could be moved.

| Demonstrations | Standard | Second set of layouts | Random start | Moved at step 10 | Moved at step 20 | Moved at step 30 | Moved at the grasp | Random start, moved at step 20 | Random start, moved at the grasp |
|---|---|---|---|---|---|---|---|---|---|
| before the fix, plain demonstrations | 85/100 | 83/100 | 6/100 | 9/100 | 0/96 | 0/99 | 1/99 (29 retried) | 0/98 | 0/98 (41 retried) |
| one instruction per layout | 93/100 | 96/100 | 9/100 | 12/100 | 0/97 | 0/100 | 0/100 (8 retried) | 0/99 | 0/99 (2 retried) |
| early windows repeated | 96/100 | 100/100 | 9/100 | 10/100 | 0/100 | 0/100 | 0/100 (3 retried) | 0/100 | 0/100 (7 retried) |
| every object named | 89/100 | 93/100 | 10/100 | 18/100 | 0/96 | 0/100 | 0/100 (29 retried) | 0/99 | 0/99 (31 retried) |
| recoveries | 84/100 | 93/100 | 68/100 | 88/100 | 79/96 | 85/100 | 78/99 (99 retried) | 70/100 | 65/81 (78 retried) |
| random starts | 98/100 | 97/100 | 97/100 | 94/100 | 50/98 | 0/100 | 0/100 (2 retried) | 15/99 | 0/100 (1 retried) |
| random starts and recoveries | 94/100 | 87/100 | 91/100 | 92/100 | 89/98 | 94/100 | 89/100 (100 retried) | 90/98 | 89/100 (99 retried) |
| random starts, recoveries, early windows repeated | 86/100 | 74/100 | 85/100 | 73/99 | 76/97 | 81/100 | 77/100 (99 retried) | 75/97 | 68/100 (99 retried) |

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
| every object named | 4/100 (at the named object 86%) | 20/100 (closed at the other object 22/100) | 0/100 (closed at the other object 100/100) | 77/100 |
| recoveries | 0/100 (at the named object 3%) | 79/100 (closed at the other object 1/97) | 56/100 (closed at the other object 28/80) | 84/100 |
| random starts | 5/100 (at the named object 98%) | 98/100 (closed at the other object 0/100) | 14/100 (closed at the other object 57/98) | 98/100 |
| random starts and recoveries | 0/100 (at the named object 0%) | 89/100 (closed at the other object 0/99) | 89/100 (closed at the other object 3/100) | 94/100 |
| random starts, recoveries, early windows repeated | 0/100 (at the named object 1%) | 78/100 (closed at the other object 0/100) | 72/100 (closed at the other object 15/100) | 74/100 |

### A second seed, with and without the fix

The plain run again with other seeds for the data order and the initial weights (`--data.seed 43 --hparams.seed 43`): once with the position table in float32 (`dp_vit_lang_reactive_pt4k_seed43_posfix`) and once with the half-precision table of the runs before the fix (`--model.transformer.positional_embedding_type rotary_legacy_autocast`, `dp_vit_lang_reactive_pt4k_seed43_legacy`). Everything else is the code after the fix. Trained / averaged weights.

| Samples | Success, old table | Success, fixed table | Closed at the named object, old | fixed | Slope, old | fixed |
|---|---|---|---|---|---|---|
| 0.5M | 1% / 2% | 0% / 1% | 37% / 31% | 25% / 36% | 0.03 / -0.05 | -0.06 / 0.01 |
| 1.0M | 3% / 2% | 2% / 6% | 44% / 39% | 83% / 72% | 0.25 / 0.15 | 0.77 / 0.72 |
| 1.5M | 43% / 46% | 46% / 58% | 97% / 100% | 98% / 100% | 0.92 / 0.98 | 0.90 / 0.98 |
| 2.0M | 58% / 74% | 71% / 87% | 99% / 100% | 99% / 100% | 0.94 / 0.99 | 0.97 / 0.99 |
| 2.5M | 81% / 80% | 88% / 92% | 100% / 100% | 100% / 100% | 1.00 / 0.98 | 0.99 / 0.98 |
| 3.0M | 84% / 85% | 92% / 88% | 100% / 99% | 100% / 100% | 0.98 / 0.99 | 0.98 / 0.99 |
| 3.5M | 91% / 90% | 94% / 93% | 100% / 99% | 100% / 100% | 0.98 / 0.98 | 0.98 / 0.99 |
| 4.0M | 93% / 91% | 91% / 94% | 100% / 100% | 100% / 100% | 0.99 / 0.99 | 0.99 / 0.99 |
| 4.5M | 90% / 93% | 93% / 95% | 99% / 100% | 100% / 100% | 0.97 / 0.98 | 0.99 / 0.99 |
| 5.0M | 90% / 89% | 96% / 96% | 100% / 100% | 100% / 100% | 0.99 / 0.99 | 0.99 / 0.99 |

The four runs side by side, final checkpoint, trained weights (averaged weights in brackets where saved):

| Seeds | Old table | Fixed table |
|---|---|---|
| first pair | 83/100 (85/100) | 93/100 |
| second pair (seed 43) | 90/100 (89/100) | 96/100 (96/100) |

The first pair's old-table run was trained from the code as it was before the fix. The second pair was trained from one version of the code and differs in the table alone.


### History, on demonstrations that begin with a hold frame (`dp_vit_lang_pt4k_hold_posfix`)

| Samples | Success (raw / EMA) | Closed at the named object | Slope | Median miss to named |
|---|---|---|---|---|
| 0.5M | 0% / 2% | 35% / 41% | 0.09 / 0.09 | 148 / 152 mm |
| 1.0M | 2% / 1% | 34% / 30% | -0.06 / 0.03 | 156 / 163 mm |
| 1.5M | 1% / 1% | 34% / 34% | -0.02 / -0.04 | 159 / 157 mm |
| 2.0M | 1% / 1% | 41% / 32% | 0.11 / -0.06 | 141 / 152 mm |
| 2.5M | 2% / 2% | 43% / 38% | 0.12 / 0.06 | 149 / 151 mm |
| 3.0M | 1% / 0% | 31% / 39% | -0.11 / -0.02 | 181 / 168 mm |
| 3.5M | 0% / 1% | 36% / 35% | 0.07 / 0.05 | 148 / 143 mm |
| 4.0M | 2% / 4% | 37% / 32% | -0.04 / -0.00 | 174 / 160 mm |
| 4.5M | 0% / 1% | 28% / 35% | -0.07 / -0.01 | 175 / 170 mm |
| 5.0M | 0% / 0% | 30% / 35% | 0.10 / 0.06 | 160 / 159 mm |
| 5.0M (final) | 0% / 4% | 34% / 35% | 0.01 / 0.12 | 147 / 145 mm |


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
| 1.5M | 74% / 83% | 100% / 100% | 1.02 / 0.99 | 12 / 9 mm |
| 1.7M | 79% / 88% | 100% / 100% | 0.97 / 0.99 | 9 / 8 mm |
| 2.0M | 81% / 88% | 100% / 100% | 0.98 / 0.99 | 7 / 8 mm |
| 2.2M | 87% / 90% | 100% / 100% | 0.99 / 0.99 | 8 / 8 mm |
| 2.5M | 84% / 88% | 100% / 100% | 0.99 / 0.99 | 7 / 7 mm |
| 2.7M | 88% / 85% | 100% / 100% | 0.99 / 0.99 | 7 / 7 mm |
| 3.0M | 88% / 86% | 100% / 100% | 0.99 / 1.00 | 7 / 7 mm |
| 3.0M (final) | 86% / 87% | 100% / 100% | 0.99 / 0.99 | 7 / 6 mm |


The same architecture with a randomly initialised backbone:

| Samples | Success (raw / EMA) | Closed at the named object | Slope | Median miss to named |
|---|---|---|---|---|
| 0.25M | 0% / 0% | 33% / 28% | -0.09 / -0.04 | 177 / 184 mm |
| 0.5M | 0% / 1% | 39% / 33% | 0.04 / 0.09 | 178 / 159 mm |
| 0.75M | 0% / 3% | 44% / 37% | 0.32 / 0.06 | 160 / 151 mm |
| 1.0M | 9% / 4% | 84% / 68% | 0.87 / 0.73 | 60 / 77 mm |
| 1.2M | 28% / 36% | 95% / 95% | 0.92 / 0.92 | 28 / 22 mm |
| 1.5M | 42% / 53% | 98% / 97% | 0.95 / 0.95 | 17 / 17 mm |
| 1.7M | 66% / 59% | 100% / 99% | 0.99 / 1.01 | 11 / 15 mm |
| 2.0M | 69% / 81% | 100% / 100% | 1.02 / 1.00 | 13 / 11 mm |
| 2.2M | 83% / 75% | 100% / 100% | 1.01 / 0.99 | 9 / 9 mm |
| 2.5M | 77% / 80% | 99% / 100% | 0.96 / 0.99 | 9 / 9 mm |
| 2.7M | 78% / 77% | 99% / 100% | 0.99 / 1.00 | 10 / 8 mm |
| 3.0M | 79% / 85% | 100% / 100% | 1.00 / 0.99 | 10 / 9 mm |
| 3.0M (final) | 75% / 85% | 100% / 100% | 1.01 / 1.00 | 9 / 10 mm |


### Qwen3-VL-2B backbone

One action-token vector, with history, 1500 demonstrations, on machine B (`vla_qwen_pt1500`: a second run of the first-round recipe, whose machine A run is in the first-round table; trained before the fix, but its action head has 20 tokens and was not affected):

| Samples | Success (raw) | Closed at the named object | Slope | Median miss to named |
|---|---|---|---|---|
| 0.10M | 1% | 33% | 0.04 | 175 mm |
| 0.20M | 3% | 60% | 0.53 | 105 mm |
| 0.30M | 2% | 60% | 0.62 | 98 mm |
| 0.40M | 5% | 71% | 0.68 | 83 mm |


Every token, no history, 4000 demonstrations, 0.5M samples (`vla_qwen_tokens_reactive_pt4k_posfix`):

| Samples | Success (raw / EMA) | Closed at the named object | Slope | Median miss to named |
|---|---|---|---|---|
| 0.05M | 1% / 1% | 40% / 35% | 0.26 / 0.08 | 159 / 149 mm |
| 0.10M | 6% / 12% | 71% / 80% | 0.76 / 0.83 | 94 / 73 mm |
| 0.15M | 9% / 29% | 89% / 97% | 0.89 / 0.98 | 55 / 32 mm |
| 0.20M | 30% / 36% | 98% / 99% | 0.96 / 1.01 | 27 / 25 mm |
| 0.25M | 32% / 45% | 99% / 98% | 1.06 / 0.96 | 27 / 20 mm |
| 0.30M | 51% / 54% | 99% / 100% | 1.02 / 1.00 | 17 / 17 mm |
| 0.35M | 46% / 64% | 100% / 100% | 0.99 / 1.00 | 19 / 15 mm |
| 0.40M | 68% / 63% | 99% / 100% | 0.97 / 0.99 | 15 / 16 mm |
| 0.5M | 57% / 65% | 100% / 100% | 1.00 / 0.99 | 16 / 13 mm |
| 0.5M (final) | 70% / 72% | 98% / 99% | 0.95 / 0.99 | 15 / 14 mm |


## Toward flower trimming (machine B)

The ladder of `tutorial/roadmap.md` (Phase 4): each step adds one property of the final task. Same policy
and recipe as the runs above unless stated: `dp_vit_lang_reactive`, 4000 demonstrations with random start
poses and recoveries (`--random_start --disturb_prob 0.5`), 5M samples, 100 unseen layouts per
evaluation, final checkpoint, trained / averaged weights where both are given. "Closed at the named
object" is the share of first closes nearest the named object.

**Physics change (after the runs of steps 1-5 above).** An audit of every contact found the arm sinking
into solid things: with MuJoCo's default soft contacts (time constant 0.02 s) the position-controlled arm
pressed the hand 11 mm into the table and a finger 24 mm into a flower, and after a cut the top part slid
down through its own base stem (the two stem pieces were excluded from colliding with each other so that
they could overlap while whole). Now contacts are stiff (`o_solref` 0.004 s, the stiffest the 2 ms step
allows), a cut leaves the two pieces apart and colliding, and a test checks that a cut top rests on the stub
instead of sliding through it. The deepest contact is now 2-7 mm (fingers squeezing the cut pieces with the
jaws shut). The scripted expert, 100 layouts each: `pick_target` 100, `pick_clutter` 99, `grasp_stem` 91,
`cut_stem` 91, `cut_one` 100 (before: 100, 100, 88, 92, 100). The policy results above were measured with
the old physics.

### Step 1: a target that keeps moving (`--disturb drift`)

The named object slides on a winding path at a fixed speed whenever the gripper is open (it turns away
from the edge of the area and from the other objects). The demonstrations of
`dp_vit_lang_reactive_pt4k_rs_recover_drift` (dataset `pick_target_4k_rs_recover_drift`): 1237 of the
4000 slide while the expert follows the object, each at its own speed drawn between 1 and 3 cm/s (`--drift_cm_s 2`: 0.5 to 1.5 times it); 789 are moved during the reach and 672 at the
grasp; 1302 are undisturbed.

| Who | 1 cm/s | 2 cm/s | 4 cm/s |
|---|---|---|---|
| expert, one plan per episode | 86/100 | 47/100 | 14/100 |
| expert, follows the object (shifts its plan by moves up to 4 cm, plans again beyond) | 98/100 | 99/100 | 97/100 |
| policy trained on clean demonstrations (`dp_vit_lang_reactive_pt4k_posfix`) | 94/100 | 62/100 | 10/100 |
| policy trained on random starts and recoveries (`..._pt4k_rs_recover_posfix`) | 94/100 | 78/100 | 27/100 |
| policy trained with sliding objects as well (`..._pt4k_rs_recover_drift`) | 96/100 | 88/100 | 70/100 |

The sliding demonstrations cost nothing elsewhere. The same policy on the battery of the table "How each
policy copes outside the conditions it was trained in": standard 89/100 / 93/100, second set of layouts
92/100, random start 89/100, moved at step 10 / 20 / 30 94/100, 89/97, 90/100, moved at the grasp
94/100 (99 retried), random start and moved at step 20 90/97, random start and moved at the grasp 87/99;
2 flow steps 99/100; cameras frozen on the first frame 0/100; expert drives 10 / 25 steps toward another
object 90/100 / 91/100. Averaged weights, checkpoints 1 to 11: 15, 69, 76, 83, 89, 95, 88, 93, 87, 96, 93.

### Step 2: look-alikes (`pick_lookalike`)

Three cubes of one shape drawn from red, orange, pink and dark red; only the colour word of the
instruction tells them apart. The expert takes the named cube in 100/100 (one plan, and planning again).

| Policy | Look-alikes | Closed at the named cube |
|---|---|---|
| clean demonstrations, never shown look-alikes | 22/100 | 34% |
| random starts and recoveries, never shown look-alikes | 24/100 | 33% |
| trained on look-alikes (`dp_vit_lang_reactive_lookalike4k_rs_recover`, seed 42) | 35/100 / 28/100 | 43% |
| the same, other seeds (`..._seed43`: `--data.seed 43 --hparams.seed 43`) | 89/100 / 93/100 | |
| the same, a third pair of seeds (`..._seed44`) | 88/100 / 91/100 | |
| the same as seed 42 for 10M samples (`..._10m`) | 80/100 / 89/100 | 85% |
| trained on the clutter scenes of step 3, which include the same cubes | 86/100 | 100% |
| trained on the clutter scenes without the leaves (`dp_vit_lang_reactive_mixed4k_rs_recover`; on its own task 96/100 / 93/100) | 97/100 | |

Look-alikes are learnable from look-alike scenes alone, but when the policy starts to tell them apart
depends on the seeds. Averaged weights per checkpoint (about 0.45M samples apart; wrong cube lifted in
brackets):

| Run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed 42 | 28 (59) | 34 (65) | 30 (67) | 38 (68) | 54 (58) | 25 (67) | 22 (67) | 36 (56) | 33 (57) | 37 (52) | 28 (59) |
| seed 43 | 24 (52) | 83 (1) | 92 (0) | 93 (0) | 83 (0) | 87 (0) | 89 (0) | 92 (0) | 83 (0) | 90 (0) | 93 (0) |
| seed 44 | 24 (60) | 36 (64) | 30 (65) | 56 (44) | 88 (3) | 83 (0) | 90 (1) | 91 (0) | 91 (0) | 91 (0) | 91 (0) |
| seed 42, 10M samples (checkpoints 0.9M apart) | 32 (62) | 31 (63) | 17 (37) | 48 (56) | 55 (48) | 59 (47) | 81 (23) | 84 (21) | 88 (17) | 80 (19) | 89 (15) |
| clutter, no leaves (its own task) | 9 (29) | 68 (1) | 84 (0) | 90 (0) | 88 (0) | 95 (0) | 95 (0) | 94 (0) | 91 (0) | 91 (0) | 93 (0) |
| clutter with leaves (its own task) | 6 (17) | 75 (0) | 81 (0) | 87 (0) | 85 (0) | 83 (0) | 84 (0) | 82 (0) | 82 (0) | 88 (0) | 86 (0) |

The switch is sudden, as on `pick_target` (notes: "Looking at the scene arrives late, and all at once"):
from a wrong cube in about 60% of episodes to none within one checkpoint. With seed 43 it comes after
about 0.9M samples, with seed 44 after about 2M, with seed 42 not within 5M. In the two runs whose scenes
also hold other shapes it came by the second checkpoint. Seed 42 with twice the samples does learn, but
gradually rather than all at once: 89/100 at the end, a wrong cube still lifted in 15%. A look-alike run that has not switched by the
end of its budget looks exactly like one that cannot learn the task; a second seed is what tells them
apart.

What carries over: the policy trained on clutter without leaves picks the named look-alike in 97/100
and, never having seen a leaf, 74/100 on clutter with leaves; the look-alike policy with seed 43 (93/100
on its own task) manages 42/100 on clutter, with a wrong object lifted in 34%.

### Step 3: clutter and occlusion (`pick_clutter`)

Four objects from seven (the four shapes of `pick_target` and the three look-alike cubes) and three
upright leaves, each standing between an object and the front camera; the leaves are seen, not touched.
The expert takes the named object in 100/100.

| Policy | Clutter | Closed at the named object |
|---|---|---|
| random starts and recoveries, never shown clutter | 35/100 | 51% |
| trained on look-alikes (step 2) | 17/100 | 30% |
| trained with sliding objects (step 1) | 46/100 | 54% |
| trained on clutter (`dp_vit_lang_reactive_clutter4k_rs_recover`) | 87/100 / 86/100 | 100% |

The clutter policy, averaged weights, checkpoints 1 to 11: 6, 75, 81, 87, 85, 83, 84, 82, 82, 88, 86.
Under the battery: moved at step 20 86/100, moved at the grasp 78/100, random start 78/100; sliding
(not in its demonstrations) at 1 cm/s 82/100, at 4 cm/s 42/100. On the plain three-shape task 87/100, on
the look-alikes 86/100.

### Step 4: a stem to grasp from the side (`grasp_stem`)

Three flowers drawn from four (red, orange, pink and purple heads on green stems in heavy bases); the instruction is "grasp
the stem of the ... flower". Success: the gripper is commanded shut with the named stem between the
fingers and inside the fingertips' depth, a finger touching it, the gripper's centre inside the band
7-11 cm above the table and the stem upright, for 10 control steps in a row. The expert (horizontal
gripper, coming in across the table from the side with more room) succeeds in 88/100; in 10 it stalls
before reaching the stem and never closes, in 2 it closes and fails.

| Policy | Grasp the stem |
|---|---|
| random starts and recoveries on `pick_target`, never shown stems | 0/100 |
| trained on stems (`dp_vit_lang_reactive_stem4k_rs_recover`) | 4/100 / 6/100 |

Over its checkpoints 1 to 11 the stem policy never passes 10/100 (averaged weights: 0, 1, 0, 3, 5, 5, 2, 4, 5, 3, 6).
Where it fails (checkpoint 9, averaged weights; 30 layouts rolled out with every control
step recorded): in 16 it never closes within 3 cm of the named stem; in 13 the stem is held by the test
above at some step, but the hold lasts 3-9 steps before a finger loses the stem; 1 succeeds. Over the
closed steps within 3 cm of the named stem, no finger touches it 56% of the time, the stem is outside the
fingertips' depth 32% of the time, outside the jaws sideways 20%, the gripper outside the band 20%. The
first close is nearest the named stem in 60% of episodes (chance 33%): the flowers' heads are themselves
look-alikes (red, orange, pink). The evaluation now also reports `caught_rate`, the share of episodes in
which the named object was held (stems) or lifted at any step: for the final stem policy, 52/100 caught
and 6/100 held for 10 steps. Executing 4 actions per chunk instead of 8 does not help (1/100, caught 40/100).

Other seeds do not change it: `..._stem4k_rs_recover_seed43` ends at 2/100 / 3/100 (never above 4).
Twice the samples help slowly (`..._stem4k_rs_recover_10m`, averaged weights per checkpoint: 0, 0, 4, 5, 7,
4, 10, 5, 13, 8, 12; final 14/100 / 12/100). It closes in almost every episode and catches the named stem
in about 40%, but in another 40-47% it holds another stem, and its first close is nearest the named stem
in only 41-46%: the step-2 problem (telling look-alikes apart) is back, and not solved by 10M samples
here. Where it goes wrong shows when the expert drives the first K control steps
toward the named stem and the policy takes over (final checkpoint, averaged weights):

| Expert drives | Grasp the stem (10 steps) | Caught at some step | First close nearest the named stem | Cut the named stem (3 steps) |
|---|---|---|---|---|
| nothing | 6/100 | 52/100 | 60% | 9/100 (19 other stems cut) |
| 20 steps (1 s; the expert's close comes at about step 60) | 7/100 | 83/100 | 88% | |
| 35 steps | 10/100 | 83/100 | 90% | 58/100 (1 other) |
| 50 steps | 10/100 | 84/100 | 92% | 66/100 (1 other) |

One second of the expert is enough to make the policy catch the named stem most of the time, so the
first failure is at the very start: committing to a stem and a side. The expert takes the side whose
path in passes farthest from the other stems, so in a given layout either side is about as likely, and
the two sides need opposite turns of the wrist from the first step. A second failure is holding still:
even started well it keeps the stem for 10 steps in only a tenth of the episodes, though for 3 steps (a
cut) in about 60%.

`collect.py --expert_side preferred` makes the side follow from where the stem stands: in from the outer
edge of the table toward its middle, the other side only when that path passes within 4 cm of another
stem. The expert cuts 90/100 that way (92/100 with the clearance rule). A first version, always the side
away from the front camera, cut only 73/100: it failed on stems right of the robot's middle line (20 of
36).

### Step 5: a cut (`cut_stem`)

The same scene; the instruction is "cut the stem of the ... flower". Each flower is built as a base and
a top part (upper stem and head) held by a weld; when the test of step 4 holds on a stem for 3 control
steps in a row, the stem is cut at the gripper's height (clipped to the band) and the top part falls.
Cutting the named stem is a success; cutting any other ends the episode as a failure.

| Who | Cut the named stem | Cut another stem |
|---|---|---|
| expert (one plan, and planning again) | 92/100 | 0/100 |
| the stem policy of step 4, never shown a cut | 9/100 | 19/100 |
| trained on cuts (`dp_vit_lang_reactive_cut4k_rs_recover`) | 0/100 / 0/100 | 0/100 / 3/100 |
| trained on cuts coming in from the outer edge (`..._cut4k_rs_recover_outer`) | 3/100 / 2/100 | 2/100 / 2/100 |

The policies trained on cuts do worse than the stem policy they were meant to improve on: 0-7/100 over
their checkpoints with either side rule. The outer-edge one closes its gripper in 80-91% of episodes, its
first close is nearest the named stem in 59-72% (the stem policy: 60%), yet the named stem is caught at
any step in only 5-6%. Rolled out step by step (30 layouts, final checkpoint, averaged weights): in 96
closed control steps within 3 cm of the named stem, the stem lies between the jaws and inside the band,
but no finger touches it in 98% of them. The gripper is commanded shut for about three steps and then
opened again, before the fingers (about 4 steps from open to the stem) arrive. The demonstrations
explain it: a cut ends the episode 3 steps after the jaws reach the stem, so the shut gripper appears in
only the last 6-7 frames of each one (the stem demonstrations hold it for 13-14), and in no frame does
the scene show a shut gripper still waiting for a stem.

`collect.py --tail_steps N` keeps recording N frames after each success with the robot holding its last
command, so every cut demonstration ends with the jaws shut for 16-17 frames. To separate cutting from
choosing the stem, `cut_one` is the same cut with a single flower (the expert cuts 100/100 with either
side rule). Final checkpoint, trained / averaged weights; all demonstrations come in from the outer edge:

| Policy | Task | Cut the named stem | Cut another stem | First close nearest the named stem |
|---|---|---|---|---|
| one flower (`dp_vit_lang_reactive_cutone4k_rs_recover_outer`) | `cut_one` | 0/100 / 1/100 | | |
| the same, other seeds (`..._seed43`) | `cut_one` | 0/100 / 0/100 | | |
| one flower, jaws held shut 10 frames past the cut (`..._cutone4k_rs_recover_outer_tail10`) | `cut_one` | **98/100 / 97/100** | | |
| three flowers, the same tail (`..._cut4k_rs_recover_outer_tail10`) | `cut_stem` | 23/100 / 22/100 | 51/100 / 50/100 | 26% / 30% |
| the same, other seeds (`..._cut4k_rs_recover_outer_tail10_seed43`) | `cut_stem` | 21/100 / 20/100 | 28/100 / 31/100 | 33% / 31% |

Averaged weights per checkpoint, cut (another stem cut in brackets):

| Run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| one flower, no tail | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 |
| one flower, tail | 36 | 80 | 87 | 92 | 92 | 99 | 97 | 100 | 99 | 95 | 97 |
| three flowers, tail | 14 (18) | 21 (30) | 22 (29) | 15 (45) | 27 (44) | 18 (47) | 23 (54) | 20 (47) | 24 (46) | 25 (49) | 22 (50) |
| three flowers, tail, seed 43 | 7 (14) | 13 (32) | 11 (33) | 24 (30) | 16 (34) | 16 (33) | 18 (38) | 25 (29) | 22 (33) | 20 (24) | 20 (31) |

Ten frames of a held gripper are the whole difference between a policy that never cuts and one that cuts
97 times in 100, from its first checkpoint on (36 after about 0.45M samples). With three flowers the
cutting itself is solved too: once the named stem is between the jaws it is cut almost every time (caught
24, cut 22). What is left is choosing the stem: the first close is nearest the named stem in 26-33% of
episodes, which is chance with three flowers, and another stem is cut in 31-50%. The instruction's colour
word is not used at all, the step-2 problem in a scene where it did not solve itself (red, orange and pink
heads are look-alikes; purple is the only colour far from the others).

Videos, recorded with the new physics (see "Physics change" above), final checkpoint, averaged weights:
the one-flower policy, 15 episodes recorded, 14 cut, 5 shown (`flower-step5a-cut-one-stem.mp4`; with
the new physics it cuts 93/100); the three-flower 10M-budget policy, 43 episodes recorded, 33 cut, 5
shown (`flower-step5-cut-the-named-stem.mp4`; 79/100 with the new physics, another stem cut in 8). After
each cut a script opens the jaws and draws the arm back so that the top can be seen to fall.

**Choosing the stem: twice the training budget.** The same 4000 demonstrations with the tail, trained for 10M
samples (`dp_vit_lang_reactive_cut4k_rs_recover_outer_tail10_10m`; checkpoints about 0.9M samples apart),
cut the named stem in 89/100 / 87/100, another stem in 3/100 / 6/100, and its first close is nearest the
named stem in 90% (the scripted expert cuts 92/100). Averaged weights per checkpoint (another stem cut in
brackets): 21 (32), 19 (50), 26 (43), 65 (23), 76 (9), 80 (3), 79 (3), 84 (4), 84 (8), 82 (8), 87 (6). Up to
about 2.7M samples it is the 5M run again; by about 3.6M it has started to use the colour word, all at once,
the same late and sudden switch as the look-alike cubes of step 2. That is before the 5M run ended,
so the difference is not simply more samples: the learning rate follows a cosine over the planned budget
(`lr_scheduler: cosine`, peak 5e-4), and at 3.6M samples this run still had about 71% of its peak rate, the
5M run about 18%. One seed of each so far. By flower: red 23/24, pink 23/25,
orange 22/28, purple 19/23 (averaged weights). Other seeds of the same run
(averaged weights per checkpoint, another stem cut in brackets):

| Run | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| seed 42 | 21 (32) | 19 (50) | 26 (43) | 65 (23) | 76 (9) | 80 (3) | 79 (3) | 84 (4) | 84 (8) | 82 (8) | 87 (6) |
| seed 43 (`..._10m_seed43`) | 15 (25) | 19 (38) | 19 (38) | 21 (49) | 24 (45) | 31 (37) | 37 (41) | 35 (43) | 47 (38) | 34 (46) | 36 (35) |
| seed 44 (`..._10m_seed44`) | 13 (36) | 26 (38) | 17 (38) | 68 (10) | 84 (3) | 87 (3) | 92 (2) | 87 (0) | 92 (1) | 91 (1) | 88 (1) |
| 16000 demonstrations (`..._cut16k_rs_recover_outer_tail10_10m`) | 16 (30) | 23 (44) | 19 (48) | 22 (61) | 20 (57) | 21 (64) | 34 (57) | 29 (52) | 26 (56) | 32 (53) | 31 (56) |

Two seeds of three switch at the same checkpoint (about 3.6M samples) and end at 87-88/100 with another
stem cut in 1-6%; the third climbs slowly and ends at 36. From 16000 demonstrations the switch never comes
within 10M samples: four times the demonstrations at the same number of samples means each one is seen a
quarter as often. As with the look-alike cubes, a run that has not switched looks the same as one that
cannot; other seeds tell them apart.

### Step 6: one side shoot (`prune_one`), new physics

The pruning scene of the roadmap: a potted plant (main stem with a flower on top) on a motorised stand that
turns while one of its two push buttons is held down. In `prune_one` the plant has one side shoot, already
facing the robot; holding it with shut jaws 2-5 cm out from the main stem for 3 control steps cuts it off.
Touching the flower, pushing the main stem with more than 5 N, or cutting the main stem is damage and fails
the episode. The scripted expert succeeds in 100/100 with no damage. The demonstrations: 4000, random start
poses, the jaws held shut 10 frames past the cut (`--random_start --tail_steps 10`); 5M samples.

| Run | Pruned (trained / averaged) | Flower damaged | Damage: flower touched / main stem pushed (averaged) |
|---|---|---|---|
| seed 42 (`dp_vit_lang_reactive_prune_one4k_rs_tail10`) | 80/100 / 77/100 | 12 / 9 | 6 / 3 |
| seed 43 (`..._seed43`) | 89/100 / 84/100 | 6 / 6 | 6 / 0 |

Averaged weights per checkpoint (damage in brackets): seed 42: 14 (76), 19 (76), 37 (62), 45 (54), 61 (30),
69 (24), 78 (14), 78 (11), 74 (16), 81 (12), 77 (9); seed 43: 38 (36), 44 (29), 48 (31), 58 (20), 69 (14), 82 (4),
88 (7), 88 (7), 90 (5), 92 (5), 84 (6). Early on the policy damages the flower in most episodes (up to 76%);
damage falls as success rises, and what is left is mostly a touch of the flower head. A successful episode
takes about 120 control steps (6 s).

### Step 7: turning the plant (`prune_turn`), new physics

One side shoot at a random angle; when it points more than 25° away from the robot the plant has to be turned
first, by holding down the button that turns it the right way. The expert succeeds in 100/100 (211 control
steps on average). 4000 demonstrations as in step 6, 10M samples.

| Run | Pruned (trained / averaged) | Flower damaged (averaged) | Steps (success) | Degrees turned / needed | First press the short way |
|---|---|---|---|---|---|
| expert | 100/100 | 0 | 211 | 89 / 91 | 100% |
| seed 42 (`dp_vit_lang_reactive_prune_turn4k_rs_tail10_10m`) | 85/100 / 90/100 | 3 (flower touched) | 232 | 106 / 88 | 88% |
| seed 43 (`..._seed43`) | 81/100 / 83/100 | 10 (flower touched) | 226 | 95 / 85 | 87% |

Averaged weights per checkpoint (damage in brackets): seed 42: 31 (19), 46 (21), 74 (10), 83 (5), 80 (8), 85 (8),
89 (3), 82 (5), 87 (4), 90 (4), 90 (3); seed 43: 30 (16), 42 (24), 50 (21), 70 (9), 81 (9), 81 (12), 84 (5), 86 (4),
77 (9), 75 (9), 83 (10). The turning is purposeful, not random. At checkpoint 4 of seed 42 (83/100), rolled out on
60 other layouts with every button press recorded: when the shoot was out of reach it pressed a button in 43 of
46, never switched buttons, and at the cut the shoot pointed a median of 3° (at most 13°) from the robot; when the
shoot already faced the robot (within about 10°) it did not press. The first press went the short way in 34 of
43 (79%); the misses were mostly shoots that had to turn counter-clockwise (14 of 20 the short way, against 20 of
23 for clockwise), so the policy leaned toward one button. Going the long way still worked: it held the button
until the shoot came round. By the final checkpoint the short way is taken in 87-88%, and the policies take
about 10% longer than the expert and turn 10-20% further.

### Step 8: the full task (`prune`)

Two or three shoots at random angles: turn, cut, turn, cut. The expert succeeds in 100/100 in about 515 control
steps; an episode may last 1000. 2000 demonstrations, 10M samples, averaged weights per checkpoint (damage in
brackets): seed 42 (`dp_vit_lang_reactive_prune2k_rs_tail10_10m`): 0 (8), 0 (38), 0 (13), 2 (35), 9 (17), 22 (7),
35 (4), 37 (3), 41 (4), 39 (3), 34 (5); seed 43: 0 (24), 1 (19), 1 (20), 6 (4), 10 (8), 17 (3), 20 (9), 25 (8), 11 (7),
16 (12), 22 (10). Executing 4 or 16 actions per predicted chunk instead of 8 (seed 42, checkpoint 8, 37/100): 16
gives 39/100, 4 gives 18/100 with the flower damaged in 16. At checkpoint 5 the failures are mostly timeouts (74 and 82
of 100), not damage (17, 8). Rolled out on 20 layouts with every cut and press recorded, the policy cut at least
one shoot in 13 and two or more in 7; the first cut came 200-700 steps in and the next 300-450 steps later (the
expert needs about 515 for everything), and in the last 200 steps of the episodes that ran out of time the
gripper was mostly near neither an uncut shoot nor a button, and some turned the plant 450-560° in all. The same
checkpoint with twice the time (2000 steps) prunes 27 and 30 plants instead of 9 and 10, but 36 and 52 still run
out of time and the flower is damaged in more (37 and 18): it is slow and it wanders, and more time also means
more chances to touch the flower. The run planned for 20M samples (same demonstrations and seed,
`dp_vit_lang_reactive_prune2k_rs_tail10_20m`; checkpoints about 1.8M samples apart) prunes 29/100 at its
second checkpoint (about 3.6M samples), where the 10M runs were at 2-6: as in step 5, a learning rate that
decays over a longer budget learns faster per sample. Averaged weights per checkpoint (damage in
brackets): 0 (60), 29 (10), 67 (4), 65 (9), 69 (8), 76 (5), 79 (1), 73 (4), 79 (5), 80 (4), 79 (4); final
81/100 / 79/100 (trained / averaged), the flower touched in 3 and 4, the rest out of time; successful
episodes take about 640 control steps. The same budget from 4000 demonstrations (`..._prune4k_rs_tail10_20m_seed43`)
prunes 84/100 at checkpoint 6 (12 out of time, the flower damaged in 4; successful episodes take about 660
control steps); the video `flower-step8-prune.mp4` shows that checkpoint (11 episodes recorded, 7 pruned, 4
shown; quiet stretches at 2x; after the last cut a script opens the jaws and draws the arm back). At checkpoint 3 (67/100) the other 29
ran out of time and 4 touched the flower; successful episodes took 723 control steps on average (the expert
about 515).

All five 20M runs, 100 layouts per checkpoint, final checkpoint as trained / averaged weights, the best
averaged checkpoint, and the flower damaged at the final averaged checkpoint:

| Demonstrations | Seed | Final (trained / averaged) | Best averaged (checkpoint) | Flower damaged (final) |
|---|---|---|---|---|
| 2000 | 42 | 81/100 / 79/100 | 80/100 (10) | 4 |
| 2000 | 43 | 44/100 / 56/100 | 59/100 (10) | 2 |
| 2000 | 44 | 80/100 / 78/100 | 82/100 (9) | 7 |
| 4000 | 42 | 78/100 / 65/100 | 78/100 (7) | 2 |
| 4000 | 43 | 76/100 / 76/100 | 84/100 (6, 8) | 6 |

Averaged weights per checkpoint: 2000 demonstrations, seed 43: 0, 2, 1, 22, 36, 50, 46, 52, 54, 59, 56; seed 44:
0, 32, 63, 67, 76, 76, 74, 79, 82, 73, 78; 4000 demonstrations, seed 42: 0, 1, 25, 55, 63, 77, 78, 65, 69, 65, 65;
seed 43: 0, 20, 53, 55, 75, 84, 82, 84, 79, 80, 76. Twice the demonstrations do not help at this budget: the
final checkpoints of the 4000-demonstration runs (65-78) sit inside the range of the 2000-demonstration runs
(56-81). The seed matters more: seed 43 of the 2000-demonstration runs learns late (1 at checkpoint 3, where the
other two seeds are at 63-67) and never catches up. Between neighbouring checkpoints a run moves by up to 13, so a best
checkpoint picked from eleven is an optimistic number; the final checkpoints are the fair comparison. Once in
five runs the averaged weights end well below the trained ones (65 against 78). The failures are mostly running
out of time, as at 10M; the flower is damaged in 2-7.

Twice the budget again, 40M samples (2000 demonstrations, seed 42, `dp_vit_lang_reactive_prune2k_rs_tail10_40m`;
checkpoints about 3.6M samples apart), averaged weights per checkpoint (damage in brackets): 0 (57), 28 (16), 46
(11), 51 (7), 64 (6), 78 (6), 69 (8), 77 (5), 76 (11), 76 (8), 80 (2); final 75/100 / 80/100 (trained / averaged),
successful episodes take about 650 control steps. The same as at 20M (81/100 / 79/100, about 640 steps): past
20M samples a longer budget does not help, and the policy is no faster.

**Takeover demonstrations (DAgger-style), first attempt: worse.** Most failures are the policy wandering into
states the demonstrations never show, so a second dataset starts from such states: the final averaged
checkpoint of `..._prune2k_rs_tail10_20m` drives the first K control steps (K uniform in 0-700), then the
scripted expert takes over from wherever the arm is (`collect.py --policy_ports`; the expert first backs the
hand out of the plant and returns to the lane in front of it) and only its part is recorded, only when it
succeeds. Of 2581 attempts, 369 ended before the handover (the policy pruned the plant in 313 and damaged
it in 56); of the 2212 handovers the expert succeeded in 2033 (92%) and touched the flower in 160; the
2000 kept episodes were handed over at a median K of 296, with 0, 1 and 2 shoots already cut in 868, 841 and
291. Trained for 20M samples on these plus the 2000 ordinary demonstrations (`..._prune2k_takeover2k_20m`),
averaged weights per checkpoint (damage in brackets): seed 42: 0 (31), 0 (47), 1 (9), 0 (11), 0
(9), 0 (15), 0 (24), 0 (27), 1 (28), 0 (26), 0 (25); seed 43: 0, 32, 45, 49, 54, 48, 38, 30, 16, 22, 23 (final
damage 22); seed 44: 13, 40, 60, 73, 67, 53, 41, 32, 25, 25, 28 (final damage 22). Final 0/100 / 0/100, 27/100
/ 23/100 and 19/100 / 28/100 (trained / averaged), against 79, 56 and 78 without them: the runs rise, then
fall as training goes on, timing out more and touching the flower more (seed 44 at checkpoint 9: 25 pruned, 48
out of time, 26 flower touched). Two faults in the dataset: it had no tail (`--tail_steps 0`; the ordinary half
has 10), and the expert's back-out and lane return after a takeover are moves the ordinary expert never
makes (4% of the takeover frames, in 92% of its episodes), so near a shoot the two halves teach opposite
actions. Both are fixed in a second takeover dataset.

**Takeover demonstrations, corrected: no better on average.** The second dataset (`prune_2k_takeover2kv2`, same
policy and handover range, other layouts) has the tail of 10 steps and leaves out the frames of the back-out and
lane return: the recording starts once the expert is on its ordinary plan (1827 of the 2000 kept episodes began
with such moves; 32,062 steps were left out, a median of 14 per episode). Of 2615 attempts, 389 ended before the
handover (335 pruned, 54 flower touched) and the expert failed after 188 handovers (163 flower touched, 8 main
stem pushed, 17 out of time); the kept episodes were handed over at a median K of 300. Trained for 20M samples on
these plus the same 2000 ordinary demonstrations, averaged weights per checkpoint (damage in brackets):

| Seed | Averaged weights per checkpoint | Final (trained / averaged) | Damaged (final) | Without takeover demonstrations |
|---|---|---|---|---|
| 42 (`..._prune2k_takeover2kv2_20m`) | 0, 27, 69, 72, 81, 84, 77, 82, 85, 78, 83 | 79/100 / 83/100 | 7 | 81/100 / 79/100 |
| 43 (`..._seed43`) | 0, 1, 0, 1, 12, 16, 25, 24, 24, 19, 24 | 25/100 / 24/100 | 7 | 44/100 / 56/100 |
| 44 (`..._seed44`) | 0, 48, 69, 65, 59, 65, 75, 70, 60, 69, 68 | 73/100 / 68/100 | 16 | 80/100 / 78/100 |

No collapse this time, so the faults of the first dataset caused it (which of the two was not tested apart). But
the takeover demonstrations do not help either: one seed gains a little (85/100 at checkpoint 9 is the best of
any pruning checkpoint, 83/100 at the end), the other two lose more, and the seed that learns late learns later
still. Step 8 stands at about 80/100 on two seeds of three, with most failures out of time.

**Memory: a frame from one second ago.** The policies so far see one frame per camera, and rolled out they wander
between shoots. Three ways of giving the policy a memory, each trained like `..._prune2k_rs_tail10_20m` (2000
demonstrations, 20M samples), final checkpoint, trained / averaged weights, successful episodes' mean length:

| Memory | Run | Final (trained / averaged) | Flower damaged | Steps per success | Without memory, same seed |
|---|---|---|---|---|---|
| Frames t-1 and t, states t-1 and t, the previous action (`dp_vit_lang`) | `dp_vit_lang_prune2k_rs_tail10_20m` | 0 at checkpoints 1-3, stopped | 40, 6, 2 | - | 81/100 / 79/100 |
| Frames t-1 and t, no previous action (`dp_vit_lang_frames`) | `dp_vit_lang_frames_prune2k_rs_tail10_20m` | 53/100 / 35/100 | 1 | about 810 | 81/100 / 79/100 |
| Frames t-20 and t (1 s back), no previous action (`dp_vit_lang_frames_1s`), seed 42 | `dp_vit_lang_frames_1s_prune2k_rs_tail10_20m` | 77/100 / 70/100 | 2 | about 700 | 81/100 / 79/100 |
| The same, seed 43 | `..._seed43` | 86/100 / 89/100 | 4 | about 680 | 44/100 / 56/100 |
| The same, seed 44 | `..._seed44` | 93/100 / 90/100 | 1 | about 670 | 80/100 / 78/100 |

Averaged weights per checkpoint, 1 s memory: seed 42: 8, 32, 36, 56, 63, 65, 73, 73, 69, 76, 70; seed 43: 13, 50, 78,
77, 77, 84, 87, 79, 90, 80, 89; seed 44: 1, 37, 49, 74, 85, 87, 91, 91, 91, 93, 90. The previous action breaks the policy again (it repeats its last action, as with
`pick_target`), and the previous frame alone does worse than no memory. A frame from a second ago does not hurt the
seed that learned well without memory (70-77 against 79-81), lifts the seed that learned late from 56 to 89, and
lifts seed 44 from 78 to 90 (93 trained, 1 flower touched): over three seeds 70/89/90 against 79/56/78 without memory,
the best final scores of any pruning runs so far. It does not make the policy
faster (about 680-700 steps per success, as without memory, against the expert's 515): it fails less often, not
quicker. Over three seeds the memory helps on average (83 against 71) and on two of the three.

**Test situations and robustness.** `tutorial/test-situations.md` adds situations on top of any plant task
(`--scenario`): changed lighting, a knocked camera, another bench, distractor objects, wind, a bumped plate, a sticky
button. A "/test" half of each visual one is never used for training (grow lights, a camera moved 3-4 cm, the
greenhouse, other distractors). A bug found on the way: every leaf was drawn about 29 mm from its collision box and
turned on its side (`leaf_fix`, off by default so earlier results stay comparable). Final averaged weights, 100
layouts per situation (successes):

| Policy | plain | lighting (/test) | camera (/test) | bench (/test) | distractors (/test) | wind | bump | sticky |
|---|---|---|---|---|---|---|---|---|
| Step-8 baseline, old leaves (`dp_vit_lang_reactive_prune2k_rs_tail10_20m`) | 79 (74 with the leaves fixed) | 17 (0) | 4 (0) | 13 (0) | 71 (69) | 75 | 74 | 49 |
| Same recipe, demonstrations with the leaves fixed (`..._prune2k_leaffix_20m`) | 76 | 13 (0) | 3 (0) | 19 (0) | 56 (74) | 74 | 73 | 52 |
| Same, with a third of the demonstrations under varied lighting, camera, bench and distractors (`..._prune2k_leaffix_vis_20m`) | 5 | 2 (0) | 2 (1) | 6 (2) | 7 (11) | 4 | 7 | 4 |

The policies cope with physical disturbances (wind, a bump, distractors cost 1-20) and fail when what the camera
sees changes (lighting, camera, bench: 0-19). With the leaves drawn where they are, the policy learns sooner (60/100
at checkpoint 2 against 2-32 for the earlier runs) but ends at the same level (76). Mixing visual variation into a
third of the demonstrations at this budget stops the policy from learning the task at all (0-10 over all eleven
checkpoints), so it is not a way to robustness here; gentler variation or a longer budget would be the next try. The
sticky button costs about 25 points: the policies do not notice a press that did nothing.

**Paired re-rendering: no better.** `sim/pickup/rerender.py` replays the 2000 leaf-fixed demonstrations bit for bit
and draws each again in other looks (lighting, camera, bench), with states and actions unchanged, so a look can only
teach what does not matter: 6000 episodes, a third in the plain look. With the 1 s memory policy, 20M samples:

| Policy | plain | lighting (/test) | camera (/test) | bench (/test) | distractors (/test) | wind | bump | sticky |
|---|---|---|---|---|---|---|---|---|
| Plain demonstrations, seed 42 (`dp_vit_lang_frames_1s_prune2k_leaffix_20m`) | 87 | 3 (0) | 2 (2) | 2 (0) | 72 (85) | 82 | 79 | 38 |
| Plain demonstrations, seed 43 (`..._seed43`) | 80 | 2 (0) | 4 (2) | 0 (0) | 60 (79) | 78 | 76 | 38 |
| Paired, seeds 42 and 43 (`..._leaffix_paired_20m`, `..._seed43`), stopped at checkpoint 9 of 11 | 2, 16 | - | - | - | - | - | - | - |

The plain policies with memory are as brittle as the ones without (lighting, camera, bench: 0-4). The paired runs
learn the plain look per frame as well as the plain runs do, and the new looks far better (with the expert driving,
the predicted joints are off by 0.029 under changed lighting against 0.112 for the plain policy, and 0.025 against
0.025 in the plain look), but driving themselves they fail (87-90% of episodes never make the first cut; they shuttle
between the buttons and the stem and push the main stem in 16-23%). A check of the data found frames, actions,
cameras, image format and statistics all identical to the source. One bug was found on the way: the distractors look
was never drawn (a parked distractor resting on the floor counted as touching the trajectory), so those copies fell
through to lighting; fixed, not re-run. Like the visual-variation run above, variation that the frames learn well
still breaks the closed loop, so robustness needs another route than more looks in the demonstrations.

**A pretrained image encoder keeps half its success when the look changes.** The named-shoot policies of step 9
include two with a pretrained vision-language backbone (`vla_qwen_tokens_reactive_named4k_2500k`, seeds 42 and 43:
Qwen3-VL-2B, trained 2.5M samples). Their report card against the from-scratch ViT policy (`dp_vit_lang_reactive`), final
averaged weights, 100 layouts per cell, successes on `prune_named`:

| Policy | Samples | plain | lighting (/test) | camera (/test) | bench (/test) | Visual mean, share of plain kept |
|---|---|---|---|---|---|---|
| Qwen3-VL backbone, seed 42 | 2.5M | 21 | 14 (9) | 7 (11) | 10 (14) | 10.8, 52% |
| Qwen3-VL backbone, seed 43 | 2.5M | 22 | 13 (5) | 10 (8) | 14 (16) | 11.0, 50% |
| From-scratch ViT, seed 42 | 2.5M | 22 | 5 (0) | 4 (6) | 6 (1) | 3.7, 17% |
| From-scratch ViT, seed 42 (`dp_vit_lang_reactive_named4k_20m`) | 20M | 53 | 11 (1) | 9 (5) | 25 (19) | 11.7, 22% |
| From-scratch ViT, seed 43 | 20M | 62 | 9 (2) | 9 (5) | 22 (20) | 11.2, 18% |
| From-scratch ViT, seed 44 | 20M | 67 | 7 (1) | 9 (8) | 8 (8) | 6.8, 10% |

At the same plain score (21-22) the pretrained backbone solves 65-66 of the 600 visual episodes against 22 for the
from-scratch ViT, and under grow lights never seen in training (lighting/test) 9 and 5 against 0-2 for every
from-scratch run. The pretrained policies were only trained to 2.5M samples, so they do not yet do the task well: in
absolute visual successes they match the 20M from-scratch runs. Under test now, with the same demonstrations and
recipe as `dp_vit_lang_frames_1s_prune2k_leaffix_20m` (20M samples, seed 42): the small policy on a frozen pretrained
image encoder in place of its own ViT (the Qwen3-VL vision tower's 196 patch features per image, on plain and on
paired demonstrations; DINOv2-B with registers at 224 and at 448 pixels), and the from-scratch policy with image
augmentation (shift, brightness, contrast, white balance, colour overlays that keep the colour cues).

**A frozen DINOv2 encoder: first real robustness to lighting (interim, checkpoint 6 of 11).** The 1 s memory policy with
the frozen DINOv2-B (registers) patch tokens at 224 pixels in place of its own ViT (`dp_dinov2_lang_frames_1s_prune2k_leaffix_20m`,
seed 42), averaged weights at checkpoint 6, 100 layouts per cell:

| Policy | plain | lighting (/test) | camera (/test) | bench (/test) |
|---|---|---|---|---|
| From-scratch ViT, 1 s memory (`dp_vit_lang_frames_1s_prune2k_leaffix_20m`, final) | 87 | 3 (0) | 2 (2) | 2 (0) |
| Frozen DINOv2 at 224 px (checkpoint 6) | **92** | **48 (43)** | **20 (7)** | 1 (0) |

It learns faster than any from-scratch run (59, 77, 88, 88, 91, 92 over checkpoints 1-6, against 9, 50, 64, 82, 82, 87),
keeps about half its success under the training lights (flicker 22, grow lights 18, noon 6, dusk 2 of the 48) and 43 under
grow lights never seen in training, where every from-scratch policy scored 0-2. A moved camera costs most of it (20, and
7 for the larger held-out moves; the failures run out of time), and another bench all of it: the 199 lost episodes all
run out of time with nothing damaged, as if the policy does not engage at all. Interim results of the other runs: the
frozen Qwen3-VL tower learns on paired demonstrations (36 and 54 at checkpoints 1-2), where the from-scratch ViT never
did; augmentation as set stops the from-scratch policy from learning the task (7/100 at the end); DART-style noisy
demonstrations learn slowly (25 at checkpoint 5); the paired consistency loss had not started to learn at checkpoint 3.

**Final report cards: two seeds of the frozen DINOv2 policy, and two failures.** Final averaged weights (checkpoint
11), 100 layouts per cell:

| Policy | plain | lighting (/test) | camera (/test) | bench (/test) | distractors (/test) | wind | bump | sticky |
|---|---|---|---|---|---|---|---|---|
| From-scratch ViT, 1 s memory, seed 42 (reference) | 87 | 3 (0) | 2 (2) | 2 (0) | 72 (85) | 82 | 79 | 38 |
| Frozen DINOv2 at 224 px, seed 42 (`dp_dinov2_lang_frames_1s_prune2k_leaffix_20m`) | 92 | 53 (55) | 15 (5) | 0 (0) | 74 (82) | 89 | 89 | 50 |
| Frozen DINOv2 at 224 px, seed 43 (`..._seed43`) | 95 | 46 (47) | 10 (4) | 1 (0) | 85 (88) | 89 | 91 | 58 |
| Frozen DINOv3-B/16 at 224 px, seed 42 (`dp_dinov3_lang_frames_1s_prune2k_leaffix_20m`) | 91 | 43 (70) | 12 (6) | 0 (0) | 66 (69) | 91 | 89 | 49 |
| DART-style noisy demonstrations, from-scratch ViT (`dp_vit_lang_frames_1s_prune2k_leaffix_dart01_20m`) | 40 | 2 (0) | 3 (2) | 0 (0) | 16 (32) | 39 | 43 | 22 |
| Paired consistency loss, from-scratch ViT (`dp_vit_lang_frames_1s_twin_prune2k_leaffix_paired_20m`) | 0 | 0 (0) | 0 (1) | 0 (0) | 0 (0) | 2 | 1 | 0 |

The frozen DINOv2 encoder holds across two seeds: the best plain pruning of any policy (92 and 95), about half of it
kept under changed light (46-55, and 47-55 under lights never seen in training), and the physical situations better
than from scratch too (wind 89, bump 89-91, a sticky button 50-58 against 38). A moved camera (4-15) and another bench
(0-1) still defeat it: on the bench it does not engage at all. Noise injected into the expert's joints (0.01 rad) halves
plain success and buys no robustness; the consistency loss between a frame and its re-rendered twin keeps the
from-scratch policy from learning at all, as the paired data alone did.

**A frozen encoder on paired demonstrations: the first policy that holds when the camera or the bench changes (interim).**
The frozen Qwen3-VL vision tower trained on the paired demonstrations (the plain ones plus the same ones drawn again
under other lighting, camera poses and benches; `dp_tower_lang_frames_1s_prune2k_leaffix_paired_20m`, seed 42), final
averaged weights, set against the same tower on plain demonstrations (`dp_tower_lang_frames_1s_prune2k_leaffix_20m`,
complete card), 100 layouts per cell:

| Policy | plain | lighting (/test) | camera (/test) | bench (/test) | distractors (/test) | wind | bump | sticky |
|---|---|---|---|---|---|---|---|---|
| Frozen Qwen3-VL tower, plain demonstrations | 92 | 29 (1) | 10 (8) | 18 (14) | 73 (87) | 85 | 86 | 58 |
| Frozen Qwen3-VL tower, paired demonstrations | 80 | 60 (29) | 51 (53) | 37 (16) | 61 (59) | 84 | 78 | 56 |

On plain demonstrations the tower is as good a pruner as DINOv2 (92) but holds less of it under new lighting (29, and
1 under lights never seen, against DINOv2's 53 and 55). Trained on the paired demonstrations it gives up 12 in its own
room and gains everywhere the look changes: lighting 29 to 60, camera 10 to 51 (8 to 53 for the larger held-out moves),
bench 18 to 37. Distractors fall (73 and 87 to 61 and 59): the one look the paired demonstrations lack, since the
distractors look was never drawn in them (the placement bug fixed above). Every policy trained on plain demonstrations scored 0-15 under a moved camera and 0-18 on another
bench, and the from-scratch ViT never learned from the paired data at all: a frozen pretrained encoder is what lets the
paired looks be learned, and the looks are what carry over to the camera and the bench. Runs of the same with frozen
DINOv2 (seeds 42, 43, 44) and DINOv3 are under way.

**Aim drills.** Seven motor drills on the bare table (`tutorial/aim-range.md`: touch, touch along a pin, stop on,
follow, flick between, press, pinch; the expert solves each at 499-500/500), as practice for the arm. Co-trained with
pruning (75% pruning, 25% drills, `dp_vit_lang_reactive_prune2k_aim8k_co27m`): 22/100 on pruning (66 on the drills),
against 79 without drills. Pretrained on drills, then fine-tuned 20M on pruning
(`dp_vit_lang_reactive_prune2k_from_aim8k_20m`): 74/100, ahead early (23 at its first pruning checkpoint, where the
earlier runs were at 0) and level at the end. Under the situations: wind 69, bump 64, sticky 32, lighting 17, camera
3, against 75, 74, 49, 17, 4 without drills. The drills do not help pruning; avoidance drills (reaching into boxes,
through gaps, around a hot zone) are built and collected but not trained.

### Step 9: named shoots (`prune_named`)

The plant of `prune` (2-3 shoots at any angles); the instruction names one shoot by its height on the stem (lowest,
middle when there are three, highest), and the robot turns the plant if needed and cuts that shoot only. The episode
ends at that cut (450 steps at most); cutting another shoot ends it as a wrong shoot. Height was chosen because it does
not change as the plate turns and reads clearly in the front camera; shoot lengths differ by too little, and a coloured
tag is a few pixels at 224 px. 24 training phrasings (one word per position, two nouns, four sentence frames).
`prune_named_heldout` uses the same layouts and named shoots with phrasings never trained on, a third each: new words
for the position ("bottom", "uppermost", "nearest the pot", "between the other two"), new sentence frames ("snip ...",
"please trim away ... but spare the others"), or both with new nouns ("branch", "offshoot"). The scripted expert succeeds
in 99/100. 4000 demonstrations with random starts and a tail of 10 (`prune_named_4k_rs_tail10`).

Final checkpoint, averaged weights, 100 layouts each (successes; wrong shoot / damaged / out of time):

| Policy | Budget | Training phrasings | Held-out phrasings | Held out by kind (new words / new frame / both) |
|---|---|---|---|---|
| `dp_vit_lang_reactive` (49M), seed 42 | 20M | 53 (9 / 20 / 18) | 28 (12 / 30 / 30) | 11 / 14 / 3 |
| the same, seed 43 | 20M | 62 (7 / 5 / 26) | 36 (28 / 6 / 30) | 7 / 21 / 8 |
| the same, seed 44 | 20M | 67 (2 / 14 / 17) | 41 (22 / 14 / 23) | 9 / 24 / 8 |
| the same, seed 42 | 2.5M | 22 (3 / 41 / 34) | 20 (13 / 38 / 29) | 3 / 12 / 5 |
| `vla_qwen_tokens_reactive` (Qwen3-VL-2B), seed 42 | 2.5M | 21 (10 / 26 / 43) | 18 (16 / 24 / 42) | 4 / 12 / 2 |
| the same, seed 43 | 2.5M | 22 (12 / 12 / 54) | 14 (11 / 9 / 66) | 3 / 6 / 5 |

The custom 130M VLA (`vla_custom_tokens_reactive`, 10M) was stopped at checkpoint 6 of 10 (the stall detector of
`train_resilient.sh` killed it during a slow save; see Operations), at 5-22 on training phrasings, failing on the
motion (damage and time-outs), not the choice of shoot.

The small policy learns to read the instruction: at 20M, when it cuts a shoot it is the named one about 7 times in 8
with training phrasings (2-9 wrong against 53-67 right; chance is 1 in 2-3). Phrasings it never saw cost it a third of
its successes and multiply the wrong cuts (12-28). Each held-out kind is about a third of the episodes: a new
sentence frame around a known position word is followed nearly as well as a training phrasing (14-24 successes), a new
word for the position much less (7-11), and both new least (3-8), as expected of a word table learned from scratch,
which has no entry for "bottom" or "uppermost". At the budget Qwen could finish
in a day (2.5M samples, about 60 samples/s against 1100), the two are level (21-22 against 22 on training phrasings,
14-18 against 20 held out), and both are still learning the motion (damage and time-outs dominate), so this budget
cannot show whether the pretrained language model generalises better. Answering that needs Qwen at a budget where the
small policy already reads the instruction (it was at 30-40% with training phrasings by 4-6M samples).

## Reasoning in words (Phase 7, stage 1)

Design and code: `tutorial/reasoning-vla.md`. Four task kinds on the step-9 plant, each needing a short chain of
inference that a lookup from instruction words to a shoot cannot do (the best such lookup is right 42-65% of the time):
ordinal ("cut the second shoot from the top"), conditional ("if the flower is red, cut the lowest shoot, otherwise the
highest one"), exclusion ("cut every side shoot except the highest one") and hidden ("read the label in the pot and cut
the shoot it names"; the label often faces away, so the plant must be turned to read it and the answer remembered).
Held-out instructions have new sentence wordings, new combinations of trained words, or both. The scripted expert
solves every kind 99/100 and writes a five-field note at its decision points (see / ask / so / aim / next), whose
sub-goal names the next shoot. 8000 demonstrations with random starts and a tail of 10 (`reason_mix_8k_rs_tail10`).

Two forms, against an end-to-end baseline (100 or 200 layouts, final checkpoints):

| Policy | Training instructions | Held out | Wrong shoot (training / held out) |
|---|---|---|---|
| Baseline: `dp_vit_lang_reactive`, instruction only, 20M (`dp_vit_lang_reactive_reason8k_20m`) | 21/100 | 13/100 | 17% / 31% |
| Plain Qwen VLA, instruction only, 2.5M (`vla_qwen_tokens_reactive_reason8k_2500k`, checkpoint 6) | 23/100 | 23/100 | 36% / 31% |
| Two-level: Qwen3-VL-2B reasoner (`qwen_reasoner_mix8k_600k`, 600k samples) writes the note, the small policy reads its sub-goal (`dp_vit_lang_reactive_subgoal_reason8k_20m_seed43`, 20M) | **133/200** | 66/200 | 7% / 36% |
| The same with the seed-42 fast level | 74/200 | 39/200 | 9% / 30% |
| One model: Qwen writes the note, then its action head reads it (`vla_qwen_reason_reason8k_2500k`, 2.5M) | 14/100 | 13/100 | 4% / 19% |
| The one model acting on the expert's notes | 56/100 | - | 0% |
| The one model acting on the reasoner's notes (hybrid) | 17/100 | 10/100 | 7% / 25% |

Per kind, training instructions (baseline per-kind runs 100 each; two-level from its 200-episode mix):

| Kind | Baseline | Two-level (seed-43 fast level) | Two-level held out |
|---|---|---|---|
| ordinal | 22 | 32/49 | 6/49 |
| conditional | 17 | 47/57 | 35/57 |
| exclusion | 13 | 19/49 | 9/49 |
| hidden | 14 | 35/45 | 16/45 |

On hidden-label plants where the card is out of view when the robot cuts, the two-level policy solves 26/33 and the
baseline 1/11: the note carries what the camera no longer shows. **The text drives the action:** with the sub-goal
swapped for another shoot's, the fast level (seed 42) solves 5/200 and cuts the shoot the wrong note names in 50%; the
one model with swapped notes solves 0/100 and cuts the wrong shoot in 77%. The fast level alone, given the expert's
sub-goals, solves 75/100 (seed 43) and 86/200 (seed 42), so the reasoner costs about 9 points on training
instructions; the fast level's seed decides most of the gap between the two two-level rows. Weak points: held-out
instructions (the reasoner restates a new sentence wording correctly only 32-36% of the time, and on new
combinations restates correctly but works out the wrong shoot), and exclusion, where its sub-goal is right only 28% of
the time because it loses track of which shoots are cut. The one model acts well on good notes but writes poor ones,
and the reasoner's notes do not suit its head (its "aim" field is right 18% of the time, the sub-goal 54%). Continuing the reasoner for 900k more samples (from its 600k weights), with and without paraphrased
training instructions (half the samples get one of 6-8 hand-written paraphrases that use no held-out word), and the
seed-43 fast level, 200 layouts each:

| Reasoner | Training instructions | Held out | Wrong shoot (training / held out) |
|---|---|---|---|
| 600k (above) | 133/200 | 66/200 | 7% / 36% |
| + 900k without paraphrases (`qwen_reasoner_mix8k_600k_cont900k`) | **153/200** | 78/200 | 7% / 36% |
| + 900k with paraphrases (`qwen_reasoner_mix8k_600k_para900k`) | 129/200 | 87/200 | 10% / 38% |

More training alone raises both (153/200 is the best two-level result so far). Paraphrases buy 9 more held-out
successes and cost 24 on the training instructions; held-out instructions stay well below training ones either way,
and wrong cuts on them stay at 36-38%.

## Architecture v2: eyes, a written program, and memory (Phase 7, stage 2)

Design, diagnosis and every number in detail: `tutorial/architecture-v2.md`. The slow level of stage 1 is split in
three: **eyes** (a small ViT trained on the simulator's labels to output one record per part: position, height
order, colour, visible, cut), a **scene memory** that keeps those records across the episode (stable identities
through rotation, the label once read, which shoots are cut), and the **language model as downloaded** (Qwen3-VL-2B,
frozen) writing the instruction as a short typed program (`If`, `Rank`, `Middle`, `Except`, `ReadLabel`, `Cut`) under a
grammar constraint; code runs the program on the memory and hands the fast level the same sub-goal as before. The fast
level is unchanged (`dp_vit_lang_reactive_subgoal_reason8k_20m_seed43`, final averaged weights). 200 layouts per
split; differences under about 15 are within noise.

| Slow level | Training instructions | Held out |
|---|---|---|
| Stage 1: Qwen reasoner writes notes from pixels | 133/200 | 66/200 |
| Stage 1 reasoner given the true scene facts (`--see expert`) | 159/200 | 96/200 |
| v2, the simulator's scene, written program | 166/200 | 138/200 |
| v2, the eyes' scene, written program | 160/200 | 135/200 |
| **v2, the eyes' scene, written program, memory options on** (`--slow_unseen --slow_revoke 3 --slow_give_up 80`) | **169/200** | **148/200** |
| v2, the simulator's scene, the task's own program (ceiling of this fast level) | 169/200 | 154/200 |

What the pieces show:

- **Counting was the stage-1 failure, and the features had it.** The stage-1 reasoner wrote "2 shoots" for 105 of 108
  three-shoot plants. A small typed head on the frozen tower's own 49 tokens reads the three-shoot count in 91% of
  frames; the eyes read the count right in 97-98% of frames over three seeds (98.5-99% with a crop of the plant),
  after 35-44 minutes of training. A loss on what is seen, not on note text, is what was missing.
- **Fine-tuning hurt the language model's language.** As downloaded it writes a program with the right meaning for
  100/100 training and 92/100 held-out instructions; with the stage-1 reasoner's fine-tuned weights, 71 and 60. The
  remaining misses are mostly "leave only the middle shoot and cut the rest".
- **Closed-loop errors that frame accuracy did not show:** an inventory that never completes (a shoot the eyes are
  sure exists but never see: 4-6/200) and a cut seen that did not happen (the eyes were trained on expert frames,
  where shut jaws at a shoot always mean a cut: 3-12/200). The memory options (hold an unseen part, revoke a belief
  after 3 contradicting looks, give up after 80 steps) take v2 on the eyes from 143 to 162 on validation layouts.
- **Under perturbation** (notes / eyes / simulator scene): plate turned mid-task 111 / 126 / 146 of 200, a button
  whose first presses do nothing 83 / 101 / 110. Bounded recovery (retrying a step with no effect) did not help (171
  without, 157-159 with) and stays off.
- **The fast level is now the limit:** the v2 slow level on the eyes is within noise of the simulator's scene, and
  with a perfect slow level the fast level times out in 22/200, 5-8 of them with a correct sub-goal and an arm that
  never moves. On the weaker fast-level seed v2 solves 99/200 (simulator scene) against 74 for the stage-1 notes.
- **Another fast-level seed.** With the seed-44 fast level (`dp_vit_lang_reactive_subgoal_reason8k_20m_seed44`, 83/100
  with the expert's sub-goals) v2 on the eyes solves 169/200 and 145/200 held out (170 and 171 on the simulator's
  scene): the result does not hang on one fast level. With the 1 s memory fast level (seed 42) and the simulator's
  scene, 151/200 on evaluation layouts.
- **A reset for the fast level's wrong first move** (`eval.py --resets 2`). About 20 episodes in 200 start for a
  shoot that faces away, jaws set to cut, without turning the plant; three in four of those are lost, and the
  wrong-shoot cuts come from the same error (an arm that went for a shoot it cannot reach closes on one it can). The
  rule sends the arm back to its start pose when the jaws are set to cut a shoot more than 45 degrees out of reach and
  the plate stands, with a grace of four looks after each cut. All of v2 on the eyes, written program: **180/200 and
  165/200 held out** (single runs; 179 and 160 on average over repeats, see "The headline, reproduced") with the
  seed-43 fast level (169 and 148 without), 175 and 157 with seed 44 (169 and 145); with
  the task's own program 178 and 183, 171 and 174. Wrong-shoot cuts fall from 6-8 per run to 0-1. Single runs of
  200, with the thresholds tuned on seed 43's validation layouts. It does not help the 1 s memory fast level.
- **Eyes trained on the fast level's own frames** cut false "cut" reports from 0.79% to 0.11% of parts in view but
  do not change episodes solved (589 and 621 of 800 against 604 and 628), so the first eyes stay.
- **The stage-1 reasoner at 448 px** (`qwen_reasoner_hires_mix8k_600k_para500k`, continued 500k with paraphrases):
  141/200 and 100/200 held out with the seed-43 fast level, against 133 and 66 at 224 px, below v2 on both.
- **A fast level told its aim** (`dp_vit_lang_reactive_aim_reason8k_20m_seed43`: the sub-goal also says whether the
  shoot is in reach and which way to turn, `eval.py --aim`; trained on half the samples with the aim and half
  without). With the expert's sub-goals it learns faster (76/100 at 4M samples against 26) but ends level (74-82
  against 75-80). In v2 it fixes what it was built for (a button pressed in 99% of the starts that need one against
  85-91%, the short way round in 86-93% against 59-70%, no wrong-shoot cut, no failure to turn) and loses as many
  episodes a new way: 26-30 in 200 time out after turning the plate past the target and round again, or closing the
  jaws beside the shoot (4-9 for the sub-goal fast level). All of v2, eyes, written program: 163/200 and 152/200 held
  out (the sub-goal fast level without resets: 169 and 148); with resets 166 on validation layouts (173-179). The same
  weights not told the aim fall to 112 and 86: one model does not learn both wordings well. Saying "in reach" as soon
  as the shoot is within 25 degrees instead of when the plate rests gave 165-168 against 156-159, within noise. The
  headline configuration stays the sub-goal fast level with resets.
- **The 1 s memory fast level, second seed** (`dp_vit_lang_frames_1s_subgoal_reason8k_20m_seed43`): 135/200 on
  validation layouts and 118 and 105 on evaluation layouts (simulator's scene), 141 with resets, below the seed-43
  sub-goal fast level. A frame from a second earlier does not help a fast level that is strong without it.
- **The 1 s memory fast level told its aim** (`dp_vit_lang_frames_1s_aim_reason8k_20m`, seed 42, final averaged weights,
  200 episodes each), told / not told with the same weights: all of v2 on the eyes, written program, 157/200 and
  158/200 held out against 131 and 118; on the task's own program 171 and 175 against 135 and 125; simulator's scene,
  evaluation layouts 169 and 167 against 134 and 139, validation layouts 180 against 136; with up to two resets 174
  (3 resets in 3 episodes) against 141 (103 resets in 85 episodes). Told the aim it hardly needs a reset, but it does
  not beat the seed-43 sub-goal fast level with resets (180 and 165), so the headline stays.
- **The film** (`reasoning-v2`) now shows that headline configuration (v2 on the eyes, written program, memory options,
  seed-43 sub-goal fast level, up to two resets): five episodes, among them a reset followed by a turn and the cut, and
  a label read, turned away and remembered.
- **The headline, reproduced** (after architecture v3's step 7: the same configuration in v3's code gave 182 and 153).
  The run's own record and the code agree on every setting (the fast level's file, its averaged weights, evaluation
  layouts 1,000,000 to 1,000,199, 8 actions a chunk and 10 flow steps, the eyes' checkpoint, the writer's snapshot,
  prompt and greedy decoding, the memory's options, `--resets 2`), the written programs are the same on all 400
  layouts, and nothing changed since in the simulator, the wordings or v2's slow level. Repeated, 200 episodes each:

  | Code | training wordings | held-out wordings |
  |---|---|---|
  | the published run's (exported and run again) | 180 (published), 181 | 165 (published), 165, 162 |
  | v3's | 182, 178, 175 | 153, 161, 156, 161 |
  | both, every run | **179** on average (175 to 182) | **160** on average (153 to 165) |

  On held-out wordings no layout fails in every run of one code and succeeds in every run of the other, and with the sampler's noise seeded
  per episode (`--noise episode`) the two codes solve 72 and 74 of the same 100 held-out layouts, with only 28 of 100
  episodes alike to the step: rollouts are chaotic, and a run of 200 varies by about 6 either way. The published
  165 is a high draw, not a configuration since lost. **The numbers to quote are 179/200 and 160/200** (the mean of 5
  and 7 runs); the comparisons above, made against single runs, stand.

## Architecture v3: the language, the truth with attributes, the plan (steps 1 to 3)

Design: `tutorial/architecture-v3.md`; what was built and where it departs from the design is under "As built" there.
No GPU. Step 1: v3's language (`CutFrom`, `Then`, `Nothing`, `Take`, `Count`; `tie,any`, `sick,no`, `tag`) beside
v2's, which stays the default of the writer's constraint; stages, equivalence over the orders of cuts, and each
situation's own program (`program.from_scenario`, which selects `env.targets`, in order for `order`, on 300 layouts
of each situation). Step 2: the simulator's truth with a tie, sick leaves and a bud per part, "unreadable" where no
camera's rays reach the place, `+undrawn` for the paired control, and the census. Step 3: the memory's rules for
attributes, the plan over the worlds the unknown facts allow (`program.WorldPlan`), and the supervisor's look, home
move and cut gate, tested with scripted percepts; for v2's programs the decisions are v2's.

**The census (E14)**: the scripted expert on 1000 validation layouts per situation (seeds from 900,000), no images,
a look every 8 control steps; each shoot counts once per attribute (2490 shoots per situation). Readable means a
camera's rays reach the tie's place, a leaf, or the tip and the place of a bud. Starting at the home pose, as v3's
evaluations will start:

| Situation: attribute | Front or wrist, first look | before the first cut | never | Front camera, first look | before the first cut | never |
|---|---|---|---|---|---|---|
| ties: tie | 95.2% | 100.0% | 0.0% | 84.7% | 97.0% | 3.0% |
| sick: sick (sick) | 99.6% | 100.0% | 0.0% | 96.5% | 99.4% | 0.4% |
| sick: sick (healthy) | 100.0% | 100.0% | 0.0% | 97.2% | 98.6% | 1.4% |
| buds: bud (a bud) | 100.0% | 100.0% | 0.0% | 94.0% | 95.8% | 4.2% |
| buds: bud (no bud) | 99.8% | 100.0% | 0.0% | 93.4% | 99.7% | 0.3% |
| prune: tie | 95.1% | 100.0% | 0.0% | 85.0% | 92.9% | 0.9% |
| prune: sick | 100.0% | 100.0% | 0.0% | 97.1% | 99.6% | 0.4% |
| prune: bud | 99.9% | 100.0% | 0.0% | 93.6% | 99.7% | 0.3% |

From the demonstrations' random starts (the expert then solves 721 to 770 of 1000: a start in the plant ends an
episode early, and with it the looks it would have had):

| Situation: attribute | Front or wrist, first look | before the first cut | never | Front camera, first look | before the first cut | never |
|---|---|---|---|---|---|---|
| ties: tie | 87.1% | 96.4% | 3.6% | 81.6% | 91.4% | 8.6% |
| sick: sick (sick) | 95.3% | 98.0% | 2.0% | 93.0% | 96.8% | 3.1% |
| sick: sick (healthy) | 95.1% | 98.3% | 1.7% | 93.3% | 96.5% | 3.5% |
| buds: bud (a bud) | 93.5% | 98.5% | 1.5% | 90.4% | 94.5% | 5.5% |
| buds: bud (no bud) | 92.7% | 96.4% | 3.6% | 89.5% | 95.3% | 4.7% |
| prune: tie | 87.5% | 96.3% | 3.7% | 81.9% | 88.5% | 7.1% |
| prune: sick | 95.3% | 97.6% | 2.4% | 93.3% | 96.6% | 3.4% |
| prune: bud | 92.9% | 96.8% | 3.2% | 89.8% | 95.9% | 4.1% |

"Never" is never while the shoot was on the plant. What it says: from the home pose, with both cameras, 95% to 100%
of attributes can be read at the first look and all of them before the first cut, so on the simulator's scene the
plan's looks will be rare and rungs 3 and 4 will rest on reading, not on looking. The tie is the hardest (95%, its
place hidden by the stem or the flower). The front camera alone misses 3% to 15% at the first look and up to 4.2% for
the whole time a shoot is on the plant: the wrist camera, which sees the plant from above at the home pose, closes
that gap, so the eyes of step 6 need it. From random starts the arm is often in the way: 87% to 95% at the first look
with both cameras, and 1.5% to 3.7% never. The expert solved 995 to 997 of 1000 layouts of each situation from home.

### Step 4's runner, built and measured on CPU (no fast level yet)

`eval.py --slow_level v3` runs v3's slow level over a served fast level: the situation's own program
(`--program gold`), the simulator's scene or its whole state (`--scene truth|full`), a look held for or shown
(`--inspect`), the home move on a verdict, the cut gate at every control step (`--gate`; off, it only records), starts
clear of the plant (`--clear_start`), and a v3 section in results.json by situation, mode, shoots and cuts meant. Two
settings were measured with the scripted expert (`calibrate_v3.py`, validation layouts):

**Starts clear of the plant.** From the demonstrations' random starts the expert touches the flower in a quarter of the
episodes, from poses among the shoots and from poses behind the plant, whose way to the lane in front of it passes the
flower. A start is kept when the gripper is at least 5 cm in front of the stem's axis (toward the robot) and at least
the clearance from it, seen from above; otherwise it is drawn again from the same stream. Expert successes per 100
layouts:

| Clearance | prune | order | nothing_to_do | ties | sick | buds |
|---|---|---|---|---|---|---|
| 0.12 m | 98 | 97 | 98 | 98 | 98 | 98 |
| **0.15 m** (the setting) | 99 | 98 | 100 | 99 | 100 | 100 |
| 0.17 m | 99 | 98 | 99 | 99 | 100 | 99 |
| 0.15 m, 300 more layouts | 294 | 298 | 291 | 297 | 297 | 293 |

**The gate's calibration** (600 expert episodes, 821 cuts): open jaws that shut on a shoot are 2.6 to 4.0 cm from the
stem's axis and 0.3 to 1.8 cm (median 1.0) above where that shoot leaves the stem. With the jaws' height less 1.0 cm
the nearest shoot is the one cut in all 821, within 0.8 cm (the band is 1.75 cm, half the least spacing). Open jaws
that shut for anything else (on the way to a button) are at least 15.7 cm from the axis, so "among the shoots" is
within 8 cm of the stand's axis and below the flower. A closed gripper pressing a button is not shutting jaws.

### E13: v3 over the fast level as it is, on rungs 1 to 4

v3's slow level with each situation's own program (`--program gold`), on the simulator's scene (`--scene truth`: what
the cameras show) or its whole state (`--scene full`: every part and attribute from the first look), over the sub-goal
fast level as it is (`dp_vit_lang_reactive_subgoal_reason8k_20m`, final averaged weights, seed 43 and its repetition
seed 44), every episode from a start clear of the plant, up to two resets, a missing attribute held for
(`--inspect hold`: nothing turns the plate for it), the gate off unless marked. "Look undrawn" is the same episode
with the ties, yellow leaves or buds not drawn (the program and the truth's readings unchanged), so it measures what
seeing them costs the fast level. `sim/pickup/scripts/e13.sh`; 70 runs, 9,450 episodes.

**Validation layouts first** (seed 43; episodes solved of 100 layouts a cell, `order` 50 per mode), to fix the
setting:

| Situation (rung) | The scene | The whole state | The look undrawn | The scene, gate on |
|---|---|---|---|---|
| `prune` | 83 | 89 | | 83 |
| `order` (1), of 150 | 131 | 129 | | 126 |
| `nothing_to_do` (2) | 95 | 98 | | |
| `ties` (3) | **52** | 65 | 85 | |
| `sick` (4) | 91 | 91 | 88 | |
| `buds` (4) | 93 | 94 | 89 | |

**The decisions, taken there:**

- **The gate stays off.** Its rule (architecture-v3.md, rule f) was to keep it where the fast level cuts a shoot its
  sub-goal did not name in more than 2 episodes of 200. Without the gate it did so in none of 250 `prune` and `order`
  episodes (no wrong shoot, no cut out of order). What the gate would have vetoed was mostly jaws shutting between two
  parts or at a shoot already cut or cut in its turn later (`prune`: 86 of 95 such commands, in 22 episodes, 15 of
  them solved), and each veto asks for a reset: with the gate on the resets rise from 18 to 48 in `prune` and episodes
  solved do not (83 and 83; `order` 131 and 126 of 150).
- **The memory's rule for attributes stays "row"**: on the simulator's scene a reading is never wrong, so the rules
  cannot differ here; E17 compares them on the eyes.
- **Ties need the fast level fine-tuned before rung 3** (P8 refuted for ties): with the same layouts, targets and
  readings, drawing the ties takes the fast level from 85 to 52 of 100. The yellow leaves (91 against 88) and the buds
  (93 against 89) cost nothing that 100 episodes can show.

**Evaluation layouts** (episodes solved of 200 a cell; `order` 100 per mode, 300 in all), both fast levels, the
setting above, the gate run as well where E13 asks for it:

| Situation (rung) | Seed 43: the scene | the whole state | the look undrawn | gate on | Seed 44: the scene | the whole state | the look undrawn | gate on |
|---|---|---|---|---|---|---|---|---|
| `prune` | 159 | 167 | | 163 | 169 | 170 | | 157 |
| `order` (1), of 300 | 247 | 246 | | 244 | 245 | 248 | | 223 |
| `order`: from the top down, of 100 | 75 | 82 | | 74 | 80 | 79 | | 67 |
| `order`: from the bottom up, of 100 | 80 | 80 | | 82 | 78 | 77 | | 74 |
| `order`: a pair, of 100 | 92 | 84 | | 88 | 87 | 92 | | 82 |
| `nothing_to_do` (2) | 188 | 197 | | | 188 | 197 | | |
| `ties` (3) | **105** | 124 | 171 | | **60** | 71 | 169 | |
| `sick` (4) | 172 | 178 | 176 | | 165 | 165 | 177 | |
| `sick`, no shoot meant, of 62 | 60 | 60 | 60 | | 60 | 60 | 60 | |
| `sick`, one or more meant, of 138 | 112 | 118 | 116 | | 105 | 105 | 117 | |
| `buds` (4) | 183 | 182 | 186 | | 180 | 189 | 185 | |
| `buds`, no shoot meant, of 54 | 53 | 54 | 53 | | 53 | 54 | 53 | |
| `buds`, one or more meant, of 146 | 130 | 128 | 133 | | 127 | 135 | 132 | |

Why episodes were lost, from the episode records (the scene, both fast levels together):

| Cause | `prune` (400) | `order` (600) | `nothing_to_do` (400) | `ties` (400) | `sick` (400) | `buds` (400) |
|---|---|---|---|---|---|---|
| the fast level out of time on a cut | 43 | 72 | | 176 | 34 | 15 |
| the fast level touched the flower | 19 | 24 | | 9 | 9 | 12 |
| a shoot not meant cut, or cut out of order | | 6 | | 8 | 2 | 2 |
| held for a shoot never seen (the inventory never completes) | 10 | 6 | 18 | 10 | 10 | 8 |
| a look ran out ("cannot see the tie of ..."; where no shoot was meant it is scored solved, below) | | | | 32 | 4 | |
| the home move touched the flower | | | 6 | | 4 | |
| solved | 328 | 492 | 376 | 165 | 337 | 363 |

What E13 shows:

- **Rung 1 carries over: order costs no more than pruning.** 247 and 245 of 300 `order` episodes (82%) against 159
  and 169 of 200 `prune` episodes (80% to 85%), with a slow level that makes no error. P1 holds: `prune` is inside
  140 to 170, and `order` (165 and 163 per 200) is at the top of 120 to 160; chaining sub-goals is not the limit. From
  the top down is the hardest mode (75 and 80 of 100): the highest shoot, first, is next to the flower (11 of 200
  touched it, against 4 from the bottom up), and the fast level cut a lower shoot before the named one in 1 and 3
  episodes of 100 (with one in a pair, the only cuts out of order: 5 of 600).
- **Rung 2: nothing to do is said at once, and lost to two scripted things.** v3 says `Complete` before any sub-goal
  in 191 of 200 `nothing_to_do` episodes (P2's bar on the simulator's scene: 190) and solves 188 (the fast level is
  never asked, so both seeds give the same episodes); with the whole state it says so in 200 and solves 197. The rest:
  9 episodes in which a stub is never seen and the plan holds for it to the end (nothing turns the plate while it
  holds), and 3 in which the home move itself touches the flower (the straight line in joint space from a clear start
  to the home pose passes it; also 2 of 62 `sick` plants with no sick shoot). The reverse error: no `Complete` at the
  first look on a `prune` plant (0 of 400); 5 of 200 on `ties` and 1 of 200 on `buds`, all withdrawn when the shoot
  not yet seen came into view.
- **Rung 3: the ties break the fast level.** 105 and 60 of 200 against 171 and 169 on the same episodes with the ties
  undrawn: seeing a tie costs 66 and 109 episodes (P8 refuted for ties: more than 30). The episodes it loses run out of
  time with the plate turning: in 137 of the 176 of them it turned more than a full turn (2 of 10 with the ties
  undrawn), past the named shoot and round again. It also cut a shoot it was not told to in 8 of 400 episodes (21 of
  400 with the whole state; 1 undrawn). So the fast level is fine-tuned on the looks before rung 3 is reported, as P8
  says.
- **Rung 4: yellow leaves and buds cost the fast level nothing measurable.** `sick` 172 and 165 against 176 and 177
  undrawn; `buds` 183 and 180 against 186 and 185 (P8 holds for them). Plants with no shoot meant are solved in 60 of
  62 (`sick`) and 53 of 54 (`buds`) with both seeds; the cuts are what is lost (`sick` 112 and 105 of 138 plants with a
  shoot meant).
- **What looking costs** (the scene against the whole state): 16 of 200 `ties` layouts need a look at a tie the
  cameras do not show from the clear start, 4 `sick` and 7 `buds`; held for (no fast level can yet show a shoot),
  every one of those looks runs out and ends as `Failed`, "cannot see", as designed. Where no shoot was meant, the
  situations' success rule (no cut, the arm home) scores that `Failed` as solved: 7 `buds` and 2 `sick` episodes of
  200 with each fast level are in the table though v3 gave up on them (without them `buds` 176 and 173, `sick` 170
  and 163). Elsewhere the two oracles are within noise (`prune` 159 and 167, 169 and 170), and `nothing_to_do` loses
  the 9 stubs never seen.
- **The gate, on the evaluation layouts too**, takes away the cuts out of order (4 to 0 from the top down) and costs
  more than it saves: `order` 244 and 223 against 247 and 245, `prune` 163 and 157 against 159 and 169; it doubles the
  resets (`prune` 91 and 108 against 45 and 50) and from the top down the flower is touched more often (24 against
  11). It stays off. On `ties` the fast level's wrong cuts are above the gate's threshold (3 of 100 on validation
  layouts), but the gate was not run there; it is measured again on `ties` after the fine-tuning.
- **What the fast level costs per rung**, against the scripted expert from the same starts (98 to 100 of 100 on each
  situation, above): with the whole state, where the slow level has nothing to find, the fast level solves 84% to 85%
  of `prune`, 82% to 83% of `order`, 98.5% of `nothing_to_do` (all of the loss the home move), 62% and 36% of `ties`,
  89% and 83% of `sick`, 91% and 95% of `buds`. Most of what it loses is time, a cut not made before the episode
  ends, and the rest mostly the flower touched.
- **v2 from the new starts.** v2's slow level on `reason_mix` (the simulator's scene, the task's program, seed 43,
  training instructions, up to two resets) from starts clear of the plant: 170 of 200. From the home pose the same
  fast level solved 169 there without resets, 173 to 179 with them on validation layouts, and 178 with them and the
  eyes (the task's program; 180 with the written one, v2's headline): within about 8 episodes, so v2's behaviour holds
  from the new starts. It loses 20 episodes to a cut not made in time, 5 to a shoot never seen, 3 to the flower and 2
  to a wrong shoot.
- **A miscount found and fixed.** The v3 summary counted the cuts meant of a `sick` or `buds` plant with no shoot meant
  as every shoot on it, as for pruning, which put those episodes among the `Complete` said wrongly; the records'
  scenario fields hold the right number and every count above is from them (`slow_level.SlowLevel.situation` now
  gives 0 there).

70 runs, 9,450 episodes, on one L40S in 2.25 hours of wall time (4 to 24 runs sharing it; the runs' own wall times add
to 27.5 hours).

### After E13: the runner fixed, the ties diagnosed, the fast level fine-tuned on them

Design: architecture-v3.md, "After E13". Three fixes to the runner, each with its tests (v2's runs unchanged): the home
move goes back to the lane in front of the plant, up and across, not in a straight line in joint space
(`--home_move clear`, the default; `line` is E13's); a missing fact, an attribute or a shoot never seen, is looked for
by turning the plate with the expert's press of a button while the slow level looks at every control step
(`--inspect turn`), within the design's bound of one full turn; and an episode whose verdict is `Failed` is not solved,
whatever the situation's rule says (`rule_success` keeps the rule's verdict). Every run below starts clear of the
plant, on the simulator's scene with the task's program, up to two resets, the gate off, on evaluation layouts (seeds
from 1,000,000).

**The fixes, on the cells they touch** (the fast levels as they were; E13's runner, then the fixed one; E13 counted
with a `Failed` verdict as a failure in brackets):

| Situation | Seed 43: E13 | fixed | Seed 44: E13 | fixed | What changed |
|---|---|---|---|---|---|
| `nothing_to_do` | 188 | **200** | 188 | **200** | the 9 stubs never seen found by a turn (9 looks, none ran out); the 3 home moves that touched the flower clear |
| `sick` | 172 (170) | 175 | 165 (163) | 167 | no shoot meant: 62 of 62 with both (E13 60, 58 without the `Failed`); 9 looks, none ran out |
| `buds` | 183 (176) | 183 | 180 (173) | 183 | no shoot meant: 54 of 54 with both (E13 53, 46 without the `Failed`); 13 looks, none ran out |
| `ties` | 105 | 118 | 60 | 64 | the 16 ties hidden from the clear start read after a turn (22 looks, none ran out) |

No episode ends on `Failed` any more in these eight runs. What is left is the fast level's: on `sick` and `buds` time
on a cut (9 to 27 episodes) and the flower touched (3 to 6); on `ties`, the ties.

**The ties, diagnosed.** In 137 of the 176 `ties` episodes E13 lost, the plate turned more than a full turn. Two
questions: is it the ties as things in view, or the sub-goal ("cut the middle shoot") clashing with a shoot that looks
different? The sub-goal's text is the same whatever is drawn, so the runs vary only what is drawn, on E13's episodes
(E13's runner, `ties+undrawn_target`: every tie but the one on the shoot meant; `ties+undrawn_others`: only that one):

| What is drawn | Seed 43 | Seed 44 | lost with more than a full turn |
|---|---|---|---|
| every tie (E13) | 105 | 60 | 45 and 94 |
| only the tie on the shoot meant | 101 | | 53 |
| every tie but the one on the shoot meant | 158 | 170 | 5 and 1 |
| no tie (E13) | 171 | 169 | 1 and 1 |

On plants where every shoot has a tie (mode `colour`), ties on the other shoots alone cost 10 and 1 of 103 (81 and 88
against 91 and 89 undrawn); the tie on the shoot meant costs the rest. And it depends on its colour (E13, both seeds;
episodes solved by the colour of the tie on the shoot meant):

| Tie on the shoot meant | drawn | undrawn |
|---|---|---|
| red (0.85, 0.10, 0.10), the red flower's colour | 13 of 90 | 74 of 90 |
| white | 35 of 104 | 90 of 104 |
| blue | 42 of 98 | 82 of 98 |
| yellow, nearest the plant's own greens and yellows | 75 of 108 | 94 of 108 |

So it is visual and local: given a sub-goal for a shoot that carries a tie, the fast level does not take that shoot
for one in reach and turns the plate on past it, most of all when the tie is red. Not the instruction or the
sub-goal: they are the same text in every row.

**The fine-tune.** `ties_1500_rs_tail10`: 1500 demonstrations of `ties` kept of 2074 tried (the scripted expert from
random starts, a tail of 10, `--park_reset`, seeds from 300,000; 311,670 frames; collected and preprocessed in about 20
minutes with 48 processes, 51 GB of shards), its sub-goal column written as the expert's notes would write it
(`notes.SubgoalWriter`: the shoot the expert goes for, by its position on the plant as found; "stop" in the tail).
The seed-43 fast level fine-tuned from its averaged weights on `reason_mix_8k_rs_tail10` and `ties_1500_rs_tail10`,
0.7 and 0.3 of each batch, with `reason_mix_8k_rs_tail10`'s normalisation, 3M samples, a peak learning rate of 2e-4
with its cosine schedule (`dp_vit_lang_reactive_subgoal_reason8k_ties_ft3m_seed43`, final averaged weights; one hour
on an L40S that evaluations shared). Evaluated on the E13 grid with the fixed runner (`--inspect turn`, the home move
clear), and the fast level as it was under the same runner where step 1 had not re-run a cell:

| Situation (rung) | Seed 43 as it was: E13 | the runner fixed | **fine-tuned on ties** |
|---|---|---|---|
| `prune` | 159 | 164 | 161 |
| `order` (1), of 300 | 247 | 245 | 241 |
| `order`: from the top down, of 100 | 75 | 76 | 77 |
| `order`: from the bottom up, of 100 | 80 | 84 | 77 |
| `order`: a pair, of 100 | 92 | 85 | 87 |
| `nothing_to_do` (2) | 188 | 200 | 200 |
| `ties` (3) | 105 | 118 | **177** |
| `ties`, the look undrawn | 171 | 192 | 191 |
| `ties`, the whole state | 124 | | 179 |
| `sick` (4) | 172 | 175 | 176 |
| `buds` (4) | 183 | 183 | 182 |
| `reason_mix`, v2's slow level | 170 | (v2 unchanged) | 182 |

- **Rung 3 is open.** 177 of 200 `ties` episodes against 118 with the same runner, and 191 with the ties undrawn: a
  tie now costs the fast level 14 episodes in 200, inside P8's 15 (74 under the same runner before the fine-tune; 66
  and 109 in E13). By the colour of the tie on
  the shoot meant: red 41 of 45 (13 before), white 44 of 52, blue 45 of 49, yellow 47 of 54. One lost episode in 23
  turns more than a full turn (44 of 82 before). It cut a shoot not meant in 2 of 200 (8 before): at the gate's
  threshold (more than 2 in 200), so the gate stays off on `ties` too. With the whole state, 179.
- **Nothing regressed.** `order` 241 of 300 against 245 and 247, `sick` 176, `buds` 182, `nothing_to_do` 200, and v2 on
  `reason_mix` 182 of 200 against 170 for the same slow level over the fast level as it was (from the same clear
  starts; per kind 46 of 49 exclusion, 44 of 49 ordinal, 52 of 57 conditional, 40 of 45 hidden): fact 6 did not
  recur, with the old data at 0.7 of each batch, a lower learning rate and 3M samples. `prune` 161 of 200 against
  164 under the same runner. The fine-tuned fast level asks for more resets on `prune` and `order` (80 in 200 `prune`
  episodes against 51; 50, 55 and 44 per 100 `order` episodes by mode against 29, 35 and 27) and solves as many.
- **Single runs of 200.** The `order` modes move by up to 7 in 100 between E13 and the fixed runner, with the same
  fast level and fixes that change little there (one look in 300 episodes): the noise to read the rest against.

Runs: 27 (8 re-runs of the fixed cells, 3 of the diagnosis, 16 of the grid), 4,800 episodes, in
`results/pickup/e13b/`. Two L40S GPUs for 3.3 hours of wall time in all: one for 1.95 hours (the collection's
rendering, the fine-tune's hour, the re-runs and the diagnosis), the other for 1.4 hours (the grid, 15 runs at once);
the runs' own wall times add to 10.3 hours.

### Step 6: the eyes with attributes (E16), the shortcut probe, and v3 on the eyes

Design: architecture-v3.md, "Eyes: four more outputs per part" and "Step 6, as built". The eyes of v2 with three more
outputs per part slot, each with a class for "cannot be read in this view": `tie` (unreadable, none, red, blue,
yellow, white), `sick` and `bud` (unreadable, no, yes), one cross-entropy each, trained from scratch on every frame,
old and new, from the simulator's labels. Every run below is on validation layouts (seeds from 900,000) unless marked,
from the demonstrations' random starts.

**The data.** `scene_tend` (the design's `scene_tend_12k` without `tags`): the scripted expert on each situation from
seeds of its own, every fourth control step and the tail, every episode kept, both cameras and the plant crop, every
frame's truth with attributes: `ties+sized` 2000 episodes (91,348 frames), `sick` 2000 (118,768), `buds+sized` 2000
(120,641), `reason_mix` 3000 (158,821), `prune` 1000 (108,102), `order` 500 (52,645), `nothing_to_do` 500 (8,800);
with v2's `scene_reason_mix_6k` and `scene_prune_2k`, 19,000 episodes and 1,193,168 frames. `+sized` draws a fifth of
the episodes with each tie or bud at a scale of its own, half to twice its drawn size. Validation: `scene_val_tend`,
1100 episodes and 68,716 frames (`ties` and `buds` 300 each, the others 100). About 16 minutes with 64 processes.

**E16: the eyes alone** (final averaged weights; three seeds of the views chosen, front, wrist and the plant crop;
each attribute on its own situation's validation frames, over the parts where it can be read and at the size it is
drawn; "cannot be read" over the parts where it cannot; ECE, the expected calibration error of the probability the
eyes give their class, over every part):

| Eyes | tie (colour) | tie: cannot be read | sick | sick: cannot be read | bud | bud: cannot be read | counted and wrong | ECE: tie, sick, bud | old outputs: three-shoot count, cut, card |
|---|---|---|---|---|---|---|---|---|---|
| front and wrist (`eyes_v3_fw_3m`) | 91.6% (90.6%) | 79.4% | 95.4% | 96.7% | 93.1% | 96.8% | 2.1% to 4.6% | 0.041, 0.021, 0.033 | 99.1%, 99.8%, 99.0% |
| with the plant crop (`eyes_v3_fwp_3m`, seed 42) | 95.4% (95.0%) | 85.9% | 99.0% | 98.2% | 98.2% | 98.3% | 0.5% to 0.7% | 0.026, 0.005, 0.009 | 99.6%, 99.9%, 99.7% |
| the same, seed 43 | 95.6% (95.2%) | 86.9% | 99.2% | 98.3% | 98.7% | 98.4% | 0.2% to 0.4% | 0.033, 0.004, 0.008 | 99.4%, 99.9%, 99.9% |
| the same, seed 44 | 95.2% (94.5%) | 85.3% | 98.3% | 97.9% | 97.8% | 98.3% | 0.7% to 0.9% | 0.026, 0.007, 0.010 | 99.6%, 100%, 99.7% |
| v2's `eyes_vit_fwp_2m` | | | | | | | | | 99.2%, 99.9%, 100% |

"Counted and wrong": of the readings the memory counts (readable, at least 0.7 sure), the share that is wrong. Each run
took 35 to 75 minutes on a shared L40S (1500 to 2500 samples per second alone, 700 beside another run and evaluations).

- **The old outputs hold.** On `scene_val_reason_mix` the three-shoot count, cut and the card read 99.6%, 99.9% and
  99.7% (`eyes_v3_fwp_3m`) against 99.2%, 99.9% and 100% for `eyes_vit_fwp_2m`: within the half point step 6 asks.
- **P3 is half met.** Sick is read on 99% (P3: 97%) and a bud on 98% with the crop (97%) and 93% from the two
  cameras (85% to 93%); "cannot be read" on 98% for both. The tie falls short: its colour on 95% of the parts where
  it can be read (98% asked), and "cannot be read" on 86% (95% asked): a tie whose place the truth calls hidden is
  often partly in view, and its band is 6 pixels wide. The refutation P3 names (a bud under 90% with the crop) did
  not happen, so the narrower crop was not needed for the design; it was built for the small-object test below.
- **Calibrated where it counts.** The probability the eyes give is close to how often they are right (ECE 0.03 for
  the tie, under 0.01 for sick and bud), but not between 0.7 and 0.99, where the memory counts a reading: a tie said
  with 0.7 to 0.9 is right in 68% of the parts, with 0.9 to 0.99 in 89% (99.7% above 0.99, where 72% of them are).
  A higher `p_min` for the tie is a lever not yet tried.

**The small-object test.** Ties are drawn at `TIE_SCALE` 1.8 because at their true size they are 2 to 3 pixels wide in
the 224-pixel front image; a bud is about 5 by 10. Two questions: does the eyes' accuracy hold at the true size, and
which view reads how small. The eyes above were trained mostly at the drawn size, so their accuracy falls away from it
in both directions (the plant crop alone reads a tie's colour at 82% at its true size, 97% at 1.8, 85% at 3.6): that
measures the training data as much as the image. For a clean measure:

- `+wide`: every tie or bud of every episode drawn at a scale of its own, a quarter to twice its drawn size in 13 steps
  (ties 0.45 to 3.6 times their true size, buds 0.25 to 2): `scene_tend_ties_wide` and `scene_tend_buds_wide`, 2000
  episodes each (90,936 and 120,660 frames), validation 300 each, from the same seeds as `scene_tend`;
- the narrower crop of the design's third set of views (`--crop4`, `plant4`): the front camera at four times the size,
  about 0.85 mm per pixel, the 224-pixel window around the plant cut out;
- eyes that look through one view each, trained on the same frames (2M samples, tie and bud only), each learning
  "cannot be read" where its own view cannot read the attribute (the truth says which camera can; for the narrower
  crop, the front camera inside its window, which leaves out about a fifth of the buds' places);
- scored on the parts its view can read: a tie read right with its colour where there is one, a bud said "yes" where
  there is one.

Two seeds per view, averaged (2M samples each; validation `scene_val_ties_wide` and `scene_val_buds_wide`, 300 + 1000
episodes each from seeds 900,000 and 910,000, about 2,000 to 8,000 readable part-frames per scale and view; the
pixel sizes are the front camera's at 224, twice that in the plant crop and four times in the narrower crop):

| A tie's colour, by scale against its true size | 0.45 to 0.54 | 0.64 to 0.76 | **0.9 to 1.07** | 1.27 to 1.51 | 1.8 to 2.14 (drawn) | 2.55 to 3.6 | ECE |
|---|---|---|---|---|---|---|---|
| tie band in the 224-pixel front image | 1.5 to 1.8 px | 2.1 to 2.5 px | **3.0 to 3.6 px** | 4.2 to 5.0 px | 6.0 to 7.1 px | 8.5 to 12 px | |
| front, 224 | 37.2% | 73.1% | **83.2%** | 86.5% | 89.3% | 88.5% | 0.14 |
| wrist, 224 | 53.9% | 66.8% | **69.5%** | 73.7% | 71.5% | 69.1% | 0.03 |
| plant crop, 2x | 69.1% | 87.0% | **93.5%** | 93.4% | 94.2% | 94.1% | 0.07 |
| narrower crop, 4x | 74.1% | 90.9% | **94.8%** | 93.3% | 93.6% | 91.2% | 0.06 |

| A bud said "yes", by scale against its true size | 0.25 to 0.35 | 0.42 to 0.5 | 0.59 to 0.71 | **0.84 to 1.0** | 1.19 to 1.41 | 1.68 to 2.0 | ECE |
|---|---|---|---|---|---|---|---|
| bud in the 224-pixel front image (across) | 1.3 to 1.8 px | 2.1 to 2.5 px | 3.0 to 3.5 px | **4.2 to 5.0 px** | 5.9 to 7.1 px | 8.4 to 10 px | |
| front, 224 | 42.4% | 54.4% | 61.2% | **74.8%** | 84.3% | 90.4% | 0.14 |
| wrist, 224 | 70.0% | 72.1% | 74.4% | **77.7%** | 78.6% | 73.7% | 0.01 |
| plant crop, 2x | 40.2% | 66.4% | 82.3% | **94.7%** (97.3% at 1.0) | 95.1% | 97.6% | 0.09 |
| narrower crop, 4x | 38.3% | 77.8% | 89.5% | **95.6%** | 96.0% | 96.9% | 0.06 |

- **True size fails at 224 and works on the crop.** A true-size tie's colour (0.9 to 1.07 times, 3 to 3.6 pixels
  wide in the front image) is read on 83% of the parts the front camera can read, on 93.5% in the plant crop and on
  94.8% in the narrower crop; a true-size bud (1.0) on 78% from the front image, 96% to 97% in the crops.
- **The smallest scale read reliably** (95%, and as well at every larger scale): a bud from 0.84 times its true size
  in the narrower crop (95.0% there, 95.6% over 0.84 to 1.0), from about its true size in the plant crop (97.3% at 1.0,
  95.1% over 1.19 to 1.41), never from the front image (90% at twice its size) or the wrist camera. A tie at no scale
  in any view: the crops level off at 92% to 95% from 0.9 times its true size up (the narrower crop from 0.76, 92.7%,
  with a dip to 89% at 2.5), the front image at 85% to 90% from its true size up, so above the true size what limits
  the tie is not pixels. The same ceiling is in the main eyes (95% at the drawn size): it is the edge of "cannot be
  read", where a tie whose place the truth calls hidden or seen is half in view.
- **The wrist camera reads at about 70% whatever the size**: what its rays reach is mostly far off and from above,
  until the hand is at the shoot. It is in the eyes for the parts the front camera cannot see (the census), not for
  small things.
- **What the views cost**: the narrower crop gains on the plant crop only below the true size (a bud at 0.59 to 0.71:
  89.5% against 82.3%; a tie at 0.64 to 0.76: 90.9% against 87.0%) and leaves about a fifth of the buds' places out
  of its window. The plant crop is the view that the eyes need for true-size things; the narrower crop is the one
  for things smaller than these.

With the main runs' eyes trained on true-size ties (`scene_tend_ties` drawn again at 1.0, everything else as
above), the front and wrist cameras read a true-size tie's colour in 85% of the parts, and with the plant crop 93%
(`eyes_v3_fw_true_3m`, `eyes_v3_fwp_true_3m`, on the fixed-scale probe sets): the same picture with the views
together. A true-size tie is out of reach of 224 pixels, and with the crop within a point or two of the crop's
ceiling.

**The shortcut probe** (the simulator's scene, the task's program, evaluation layouts, 200 episodes each,
`eval.py --swap_target`: the slow level names another shoot than the one the situation means, chosen per seed). The
fine-tune's data mixed `colour` (every shoot tied) and `one` (only the shoot meant tied), so the fine-tuned fast
level might have learned "cut the tied shoot":

| | Mode `one`, sub-goal for an untied shoot | Mode `colour`, sub-goal for a shoot with another colour |
|---|---|---|
| fine-tuned on ties (seed 43): cut the shoot named | **176** | 177 |
| ... cut the tied shoot the situation meant | 2 | 4 |
| as it was (seed 43): cut the shoot named | 180 | 101 |
| ... cut the tied shoot the situation meant | 2 | 5 |

No shortcut: sent to an untied shoot on a plant with one tie, the fine-tuned level cuts the shoot named in 176 of 200
and the tied one in 2, as the level before the fine-tune does (180 and 2). In `colour` mode it cuts the shoot named in
177 (101 before: the ties on the shoot named stopped it, as diagnosed). So the fine-tune was not redone.

**Cut precision**: where the gripper's centre was, from the shoot's axis as grown, at the control step that cut it
(`cut_places`, evaluation layouts, the simulator's scene, the task's program):

| | cuts | across the shoot: median | mean | 90th percentile | along it from the stem: median |
|---|---|---|---|---|---|
| the scripted expert, `prune` | 501 | 5.9 mm | 6.2 mm | 9.5 mm | 34.6 mm |
| the scripted expert, `ties` | 197 | 5.8 mm | 6.0 mm | 9.7 mm | 34.2 mm |
| as it was, `prune` | 451 | 7.9 mm | 8.5 mm | 14.4 mm | 33.3 mm |
| fine-tuned on ties, `prune` | 437 | 8.2 mm | 8.9 mm | 15.6 mm | 33.5 mm |
| fine-tuned, `ties`, `one` mode swapped (untied shoot) | 176 | 7.8 mm | 8.3 mm | 13.9 mm | 34.3 mm |
| fine-tuned, `ties`, `colour` mode swapped (tied shoot) | 177 | 10.5 mm | 11.1 mm | 18.4 mm | 33.9 mm |
| fine-tuned, `ties` on the eyes | 173 | 9.8 mm | 10.6 mm | 17.6 mm | 33.8 mm |

Both fast levels cut about 2 mm further from the shoot's axis than the expert (whose jaws close 2.6 to 4.0 cm out
along it, as the gate's calibration found), and the fine-tune did not change that on `prune` (8.2 against 7.9). On a
shoot with a tie it closes about 2.5 mm further off (10.5 against 7.8): the tie, at `TIE_AT` on the shoot, is where
the jaws would go.

**v3 on the eyes** (`--scene eyes` with `eyes_v3_fwp_3m`, the task's program, the fine-tuned fast level, up to two
resets, the gate off, `--inspect turn`). On validation layouts first, to choose the views and the memory's rule:

| Situation | front and wrist, "row" | with the crop, "row" | "last" | "angles" | "sticky" |
|---|---|---|---|---|---|
| `prune` | 78 | 78 | | | |
| `order`, of 150 | 127 | 117 | | | |
| `nothing_to_do` | 99 | 99 | | | |
| `ties` | 77 | 85 | 81 | 81 | 87 |
| `sick` | 75 | 86 | 83 | 82 | 89 |
| `buds` | 68 | 89 | 77 | 74 | 82 |
| a wrong shoot cut or `Complete` said wrongly, ties+sick+buds | 43 | 19 | 27 | 13 | 15 |

- **The crop stays**: 260 of 300 on the three situations that read attributes against 220, and half the attribute
  errors in the memory (26 episodes against 81); `order` (no attribute) 117 against 127 of 150, inside the noise of
  50-episode cells.
- **The memory's rule stays "row"** (three readings in a row, as v2's): 260 of 300, "sticky" (a first value from a row,
  a change only from two angles) 258, "last" 241, "angles" (a row whose readings span 30 degrees) 237. "Angles"
  cuts the false completes from 13 to 3 but turns the plate for 392 looks against 42 and runs out of time in 46
  episodes;
  "sticky" halves the false completes and more than doubles the wrong cuts on `buds` (7 against 3). The
  rules were compared on these layouts before the evaluation layouts were run.

Then the evaluation layouts, against E13b (the same fast level and runner on the simulator's scene):

| Situation (rung) | simulator's scene (E13b) | **the eyes** | difference | attribute misread (episodes) | wrong shoot cut | `Complete` said wrongly |
|---|---|---|---|---|---|---|
| `prune` | 161 | 154 | -7 | 3 | 0 | 0 |
| `order` (1), of 300 | 241 | 239 | -2 | 6 | 4 | 2 |
| `order`: from the top down, of 100 | 77 | 78 | +1 | | | |
| `order`: from the bottom up, of 100 | 77 | 77 | 0 | | | |
| `order`: a pair, of 100 | 87 | 84 | -3 | | | |
| `nothing_to_do` (2) | 200 | 199 | -1 | 1 | 0 | 0 |
| `ties` (3) | 177 | 170 | -7 | 23 | 3 | 11 |
| `sick` (4) | 176 | 162 | -14 | 14 | 0 | 13 |
| `buds` (4) | 182 | 166 | **-16** | 19 | 9 | 4 |
| `reason_mix`, v2's slow level | 182 | 176 | -6 | | | 1 |
| `reason_mix`, v2's slow level with v2's eyes (`eyes_vit_fwp_2m`) | | 177 | | | | |

- **P5 holds on rungs 1 to 3 and misses rung 4 by one episode.** `buds` loses 16 (P5 allows 15), `sick` 14. The
  episodes lost to perception are the attribute's: on `buds` 9 wrong shoots cut, 2 false `Complete` and 2 out of time
  with a bud misread; on `sick` 10 `Complete` said with a sick shoot left; on `ties` 11 `Complete` said and 3 wrong
  shoots cut. The rest (out of time, the flower touched) is about as often as on the simulator's scene.
- **No regression where no attribute is read**: `prune`, `order` and `nothing_to_do` within 7, v2's `reason_mix`
  176 against 182 on the simulator's scene, and 177 with v2's own eyes: the new outputs cost the old task nothing.
- **What reads wrong**: the memory held a wrong attribute at some look in 7% to 12% of the episodes of the three
  situations (14 to 23 of 200), each for a median of 13 to 20 looks: the eyes are wrong in runs, not in single
  frames, which three readings in a row do not filter. On `buds` most of these start at the first look, from the
  clear start (median control step 27); on `ties` and `sick` after 120 to 165 control steps, with the arm at the
  plant, states the expert's frames show less: what step 7's frames from the fast level's own rollouts are for.

Runs: 18 perception datasets (the 14 of `scene_tend` and its validation in 16 minutes with 64 processes, the true-size
ties, the 4 wide sets of the small-object test in 18 minutes), 17 eyes runs (9 on everything, 8 one-view), 39
closed-loop runs and 5,400 episodes in `results/pickup/e16/`. Two L40S GPUs for 4.9 hours of wall time each (9.8
GPU-hours held), shared by the eyes' training, the collections' rendering and up to 9 evaluations at once.

### Step 5: the program writer for v3's language (E15)

Design: architecture-v3.md, "The program language" and "Step 5, as built". The frozen Qwen3-VL-2B of v2's writer
(not fine-tuned) writes v3 programs (`CutFrom`, `Then`, `Take`, `Count`, `Nothing`, `tie,any`, `sick,no`) under v3's
grammar constraint, greedily, with a description of the language construct by construct and sixteen worked examples
in pairs (`program_writer_v3.py`); v2's writer, its prompt and its numbers are unchanged. It is scored alone, without
images, on 100 tasks per situation and split drawn as the situations draw them (two or three shoots, the ties, sick
shoots and buds): `prune` and `nothing_to_do` (one instruction, the task's own), `order`, `ties`, `sick`, `buds` in
their training and held-out sentence frames, and the compositions of rung 5: a first set of 12 (24 instructions) that
supplies the examples' constructs, and a held-out set of 8 new combinations in frames of their own (16 instructions).
The task's own program is the one the closed loop runs (`program.from_scenario`'s). Flags as v2's: format (it parses),
exact (its text is the task's), meaning (equivalent in v3: the same sequences of cuts on every plant), resolved (the
same sequences of cuts on the plant drawn). Each distinct instruction is written once (the output depends on nothing
else).

**Two prompts.** The first (16 examples, set A) was run, its errors on training instructions read, and the description
and examples revised once (the second): "the middle one" had been written `Rank(2,top)`, "the highest first, then the
next one down" as two cuts, and sick and bud with the wrong value. Nothing was changed on the held-out errors. Meaning
of 100 (resolved in brackets where it differs):

| Writer | training frames (of 700) | held-out frames (of 400) | held-out compositions | prune, nothing_to_do | order train / held out | ties | sick | buds | compositions, first set |
|---|---|---|---|---|---|---|---|---|---|
| first prompt, A, 16, constrained | 560 (585) | 205 (209) | 67 (86) | 100, 100 | 56 (76) / 78 (79) | 100 / 78 (81) | 63 / 0 | 67 / 49 | 74 (79) |
| **second prompt, A, 16, constrained** | **580 (610)** | **215** | **56 (77)** | 100, 100 | 68 (88) / 88 | 100 / 78 | 66 / 0 | 67 / 49 | 79 (89) |
| second, A, 16, free | 580 (597) | 215 | 56 (77) | 100, 100 | 68 (75) / 88 | 100 / 78 | 66 / 0 | 67 / 49 | 79 (89) |
| second, A, 8 (first of each pair), constrained | 391 (441) | 201 (216) | 47 (68) | 0, 0 | 52 (73) / 74 (89) | 100 / 78 | 100 / 0 | 100 / 49 | 39 (68) |
| second, B, 16, constrained | 407 (436) | 234 (250) | 65 (79) | 0, 0 | 74 (81) / 83 | 100 / 49 (65) | 100 / 53 | 67 / 49 | 66 (88) |

Free writing parses in 1180 of 1200 (20 `order` tasks: a `Then` of three statements written flat, `Then(a,b,c)`, and
programs run together); constrained always. Asked twice (second prompt, A and B, the slow level acting only when the two
programs are equivalent), of 1200 tasks: the two agree on 721, of which 151 are both wrong; of the 479 where they
disagree, 281 had a right first program (blocked) and 198 a wrong one (stopped); the first writer alone runs 349 wrong
programs. Set B cannot hold the prune instruction (the sets are disjoint and it has one wording), and the writer then
writes `Cut(Filter(tie,any,All()))` for it, so asking twice blocks every `prune` and `nothing_to_do` episode. It is not
used. A probe that let the model think first (up to 512 tokens, then the program under the constraint) wrote 3 of the
first 9 distinct training instructions wrong, as without it, at one to two minutes per instruction on the shared GPU,
and was stopped.

- **P4 is refuted.** Meaning on 83% of training frames (95% asked), 54% of held-out frames (85% asked) and 56% of
  held-out compositions (75% asked); P4's refutation (under 70 held out in new frames) says the examples come first,
  and one revision raised the training frames by 20 and the held-out frames by 10.
- **What it gets wrong is polarity and order, not form.** "cut the sick shoots, keep the healthy ones" is written
  `sick,no` (34 of 100 `sick` training tasks), "keep the buds, cut the other shoots" `bud,yes` (33 of 100), though that
  sentence is one of the sixteen examples with its program; every held-out `sick` frame ("the unhealthy shoots have to
  go", "snip off the ailing branches and spare the healthy ones") is written `sick,no`; "the highest first, then the
  next one down" is two cuts. The 2B model copies the program of the example whose words are nearest, and the pairs,
  meant to teach the difference, put a near sentence with the other program beside each one.
- **The constraint changes almost nothing** (as in v2): the free writer writes the same programs where both parse.
  Sixteen examples beat eight by 189 training tasks, most of it the prune instruction, which is the sixteenth example.

**All of v3 with the written program** (the eyes `eyes_v3_fwp_3m`, the fast level fine-tuned on ties, the second
prompt, set A, 16 examples, constrained, once; evaluation layouts, 200 per cell, `order` 100 per mode; the same setting
as step 6's "v3 on the eyes", whose task-program column is copied here). Written: episodes whose program means the
task's (v3 equivalence, computed from the records):

| Situation | the task's program (step 6) | **the written program** | written right | solved with a wrong program |
|---|---|---|---|---|
| `prune` | 154 | **160** | 200 | 0 |
| `order`, of 300 | 239 | **215** | 226 | 33 |
| `order`: from the top down | 78 | 57 | 43 | 24 |
| `order`: from the bottom up | 77 | 78 | 100 | 0 |
| `order`: a pair | 84 | 80 | 83 | 9 |
| `nothing_to_do` | 199 | **199** | 200 | 0 |
| `ties` | 170 | **168** | 200 | 0 |
| `sick` | 162 | **125** | 149 | 1 |
| `buds` | 166 | **121** | 136 | 5 |
| held-out frames: `order`, of 300 | (239) | **220** | 283 | 0 |
| held-out frames: `ties` | (170) | **147** | 146 | 19 |
| held-out frames: `sick` | (162) | **2** | 0 | 2 |
| held-out frames: `buds` | (166) | **94** | 98 | 10 |

The task's program does not read the words, so its held-out-frame cells are its training-frame ones (in brackets).

- **P5 holds where the writer is right and fails where it is not.** `prune`, `nothing_to_do` and `ties` are within 6
  of the task's program (the writer is right on all of them); `order` loses 24 (one sentence frame of "from the top
  down" in three is written as two cuts; on a two-shoot plant that is the same and 24 of them are solved anyway),
  `sick` 37 and `buds` 45: the third of their training frames written with the wrong value cuts the healthy shoots or
  the buds (29 and 35 wrong cuts against 0 and 9). In held-out frames `sick` is lost whole (every program `sick,no`).
- **Where the program is right, the closed loop is the task program's**: among the episodes with a right program the
  share solved matches step 6's (`sick` 124 of 149, `buds` 116 of 136). What the writer costs is its own meaning
  errors, one sentence frame at a time, since an instruction is written the same way in every episode.

### Step 7: retraining on the fast level's own rollouts (the fast level and the eyes), and calibration

Design: architecture-v3.md, "Step 7, as built". Two retrainings from the fast level's own rollouts under v3, one for
each part whose errors E13b and step 6 measured, each tried on validation layouts first, then the E13 grid on
evaluation layouts. Every run below: the task's program, starts clear of the plant, up to two resets, `--inspect
turn`, the gate off.

**The fast level, on takeover demonstrations from its own rollouts.** The fast level fine-tuned on ties (seed 43)
drives under its sub-goals (the column the situations' demonstrations carry, `notes.SubgoalWriter`), from starts clear
of the plant, and the scripted expert takes over where it drifts (`collect.py --takeover_subgoals`): when its open jaws
come within 3.5 cm of the middle of the named shoot's cut zone (with probability 0.5 per shoot; the expert then
finishes that cut from where the hand is, without backing out, `ScriptedExpert.finish_cut`), when the plate has
carried the named shoot within 10 degrees of its reach and on past it by 25 degrees while still turning away (the
expert turns it back), or at a step drawn from 0 to 700, whichever comes first. Only the expert's part is recorded
(not its back-out), with a tail of 10, only when it succeeds; layouts from seeds 610,000 to 650,000:

| Situation | kept of attempts | frames | handed over at a cut | at an over-turn | at the drawn step | the expert failed after the handover |
|---|---|---|---|---|---|---|
| `prune` | 400 of 456 | 155,492 | 155 | 37 | 208 | 27 (16 flower) |
| `order` | 400 of 447 | 156,944 | 166 | 24 | 210 | 13 |
| `ties` | 400 of 561 | 48,607 | 138 | 40 | 222 | 15 (115 solved before any handover) |
| `sick` | 400 of 642 | 108,111 | 126 | 38 | 236 | 170 (165 flower: its way home from a takeover pose) |
| `buds` | 400 of 655 | 95,611 | 136 | 27 | 237 | 174 (164 flower) |
| all (`takeover_tend_2k`) | 2000 of 2761 | 564,765 | 721 | 166 | 1113 | |

The fast level was fine-tuned from the ties fine-tune's averaged weights on `reason_mix_8k_rs_tail10`,
`ties_1500_rs_tail10` and `takeover_tend_2k` (0.6, 0.15, 0.25 of each batch, the old normalisation), 3M samples at a
peak learning rate of 2e-4, as the ties fine-tune (`dp_vit_lang_reactive_subgoal_reason8k_ties_tk_ft3m_seed43`, final
averaged weights; one hour). The simulator's scene, the task's program:

| Situation | validation: ties fine-tune | validation: **takeover fine-tune** | evaluation: ties fine-tune (E13b) | evaluation: **takeover fine-tune** |
|---|---|---|---|---|
| `prune` | 79 | 85 | 161 | 163 |
| `order`, of 150 / of 300 | 118 | 124 | 241 | 244 |
| `nothing_to_do` | 100 | 100 | 200 | 200 |
| `ties` | 94 | 88 | 177 | 171 |
| `sick` | 87 | 90 | 176 | 178 |
| `buds` | 94 | 90 | 182 | 182 |
| `reason_mix`, v2's slow level | | | 182 | 175 |
| all situations (validation, of 650) | 572 | 577 | | |
| cut precision, across the shoot: median (validation) | 8.3 mm | **9.0 mm** | 8.2 mm (`prune`, step 6) | 8.7 mm (`prune`) |
| wrong shoots cut (validation; evaluation, all situations) | 4 | 3 | 6 | 6 |
| episodes lost with more than a full turn (validation) | 26 of 78 | 30 of 73 | | |

- **No gain, and the cut is no closer.** On validation layouts the takeover fine-tune solves 577 of 650 against 572,
  within noise, and on evaluation layouts every cell is within 7 of E13b's (`reason_mix` 175 against 182). The jaws
  close further from the shoot's axis, not nearer (median 9.0 mm against 8.3 on the same validation episodes; the
  expert's 5.9), and over-turning is as before (30 lost episodes with more than a full turn against 26). The fast
  level as it was stays. No control fine-tune on the old data alone was run, so whether the takeover data or a second
  fine-tune moved the cut out is not separated.
- **Where the handovers came.** Drift was found in 755 of 2761 attempts at a cut and 167 at an over-turn: an over-turn
  past 25 degrees happens in about one rollout in sixteen; the cuts handed over show the expert's last 25 to 40
  frames from the fast level's own approach, a small part of a batch (0.25 of it is takeover data, of which those
  frames are a fraction).

**The eyes, on the fast level's own rollouts** (the design's step 7). The fast level fine-tuned on ties ran under v3
on the eyes (`eyes_v3_fwp_3m`, the task's program) on `ties`, `sick` and `buds` from seeds 700,000 to 722,000 (layouts
used nowhere else) with `eval.py --record_scene`: 1998 episodes (702, 651, 651; 1670 solved), 190,756 frames, every
fourth control step with the plant crop and the simulator's truth. The eyes were trained again from scratch on every
frame of step 6 and these (1.38M frames; front, wrist and the plant crop, 3M samples, seed 42, as `eyes_v3_fwp_3m`:
`eyes_v3_fwp_roll_3m`). On the expert's validation frames they read as before (tie 95.1%, sick 98.5%, bud 97.8% on
their own situations, against 95.4%, 99.0%, 98.2%; the three-shoot count 99.7%, cut 99.99%, the card 98.8% on
`scene_val_tend_reason`); in closed loop they misread far less.

**Calibration.** One temperature per attribute, fitted on the validation frames (`eyes_calibrate.py`: least NLL,
"cannot be read" a class like the others): 1.23 (tie), 1.24 (sick), 1.19 (bud) for `eyes_v3_fwp_3m`, 1.25, 1.22, 1.22
for the new eyes. It lowers the expected calibration error over all parts (tie 0.009 to 0.003) and the readings the
memory counts that are wrong only from 0.30% to 0.24% (tie): over all frames the eyes were not far from calibrated,
and a misreading that lasts is said with confidence.

On validation layouts (the fast level fine-tuned on ties; 100 episodes a cell; "misread": episodes in which the memory
held a wrong attribute at some look):

| Eyes | `ties` | `sick` | `buds` | of 300 | wrong shoots cut | `Complete` said wrongly | misread |
|---|---|---|---|---|---|---|---|
| `eyes_v3_fwp_3m` (step 6) | 85 | 86 | 89 | 260 | 6 | 13 | 26 |
| the same, with its temperatures | 88 | 87 | 82 | 257 | 8 | 8 | 32 |
| **`eyes_v3_fwp_roll_3m`** | **92** | **92** | **91** | **275** | **0** | **3** | **10** |
| the same, with its temperatures | 92 | 85 | 88 | 265 | 1 | 4 | 11 |

The temperatures do not help in closed loop and are not used; the rollout eyes are. On evaluation layouts, with the
fast level fine-tuned on ties (the same as step 6's "v3 on the eyes" and E13b's):

| Situation | the simulator's scene (E13b) | the eyes of step 6 | **the rollout eyes** | misread: step 6 / rollout | wrong shoots | `Complete` said wrongly |
|---|---|---|---|---|---|---|
| `prune` | 161 | 154 | **157** | 3 / 0 | 0 / 0 | 0 / 0 |
| `order`, of 300 | 241 | 239 | **242** | 6 / 1 | 4 / 1 | 2 / 2 |
| `nothing_to_do` | 200 | 199 | **198** | 1 / 2 | 0 / 0 | 0 / 0 |
| `ties` | 177 | 170 | **179** | 23 / 6 | 3 / 3 | 11 / 1 |
| `sick` | 176 | 162 | **172** | 14 / 7 | 0 / 1 | 13 / 2 |
| `buds` | 182 | 166 | **180** | 19 / 9 | 9 / 2 | 4 / 1 |
| `reason_mix`, v2's slow level | 182 | 176 | **177** | | | |

- **Perception now costs at most 4 episodes in 200 on any situation** (`ties` 179, `sick` 172, `buds` 180 against
  177, 176 and 182 on the simulator's scene; 170, 162 and 166 with step 6's eyes; `prune` 157 against 161): P5
  holds on every rung with the task's program. The
  misreads that lasted 13 to 20 looks were states the expert's frames do not show; with 190,000 frames of the fast
  level's own states they fall by half to three quarters, the wrong cuts on `buds` from 9 to 2 and the false
  `Complete` on `ties` and `sick` from 24 to 3.
- **Nothing regressed**: `order`, `nothing_to_do` and v2's `reason_mix` (177 against 176) are as they were.
- **The wrong cuts on `buds` were the eyes', not the fast level's.** All 9 of step 6's came after the memory held a
  budding shoot as "no bud" (the record's `attributes_wrong`), and on the simulator's scene the fast level cuts a shoot
  not meant in at most 2 of 200 `buds` episodes.

Runs: 9 recording runs (1998 episodes), 5 takeover collections (2761 attempts), one fine-tune, one eyes run, two
temperature fits, 12 validation and 18 evaluation closed-loop runs for step 7 (about 5,800 episodes), 14 for step 5
(2,600 episodes) and the writer alone (8 configurations), in `results/pickup/step5/`, `results/pickup/e15/` and
`results/pickup/step7/`. GPUs 6 and 7 of an L40S machine were held for 2.5 and 2.2 hours of wall time (4.7 GPU-hours), shared by up to 16 runs at once.

### After step 7: the writer fine-tuned with LoRA, and the fast level on a frozen DINOv2 encoder

Design: architecture-v3.md, "After step 7, as built". The two parts that step 7 left weakest: the program writer
(meaning on 83% of training instructions, 54% of held-out frames, 56% of held-out compositions; `sick` 125 and `buds`
121 with written programs) and the fast level's cut (8.2 mm from the shoot's axis against the expert's 5.9, and 26 to 30
of 650 validation episodes lost turning the plate on past a full turn).

**The writer: thinking first, and LoRA on program writing.** Scored alone exactly as E15 (100 tasks per situation and
split, seed 0, the second prompt, examples A, 16, constrained; the base run reproduces E15's report row for row). The
writer of v2 and E15 is already the Qwen3-VL-2B **Thinking** checkpoint, made to answer at once with an empty think
block; the Instruct checkpoint is not in the machine's cache, and nothing was downloaded. So "thinking" is the same
checkpoint let to reason first (up to 2048 tokens, greedily or sampled at Qwen's 0.6 / top-p 0.95 / top-k 20 for
thinking, reasoning in batches of 16 instructions; only the program after `</think>` is held to the grammar). LoRA is
the same model with low-rank adapters (rank 16, alpha 32, on q, k, v, o, gate, up and down of the 28 language-model
layers, 17.4M parameters, the base frozen; `writer_tune.py`) trained on program writing alone, with the same prompt as
at test time: 2191 instructions written from templates and synonyms of every situation's training tasks (prune, the
three orders, ties of each colour and any tie, sick and healthy, with and without a bud, in both polarities: "keep the
healthy shoots and trim every other shoot"), the first set of compositions said in other words (never recombined), and
v2's training tasks in its frames and training paraphrases. Kept out, and tested (`check_excluded`,
`test_writer_tune.py`): every held-out instruction of v3 and v2, every word that only held-out instructions use (45,
from "snip", "branch", "spare", "unhealthy" to "to", "of", "each", "only") and every two-word sequence that only they
use, and every program equivalent to a held-out composition's (v3's eight and v2's five). Two epochs, learning rate
1e-4, batches of 8, 548 steps in 41 minutes on a shared GPU; the loss is on the program and the end of the turn only.

Meaning of 100 tasks (resolved on the plant drawn in brackets where it differs):

| Writer | training (of 700) | held-out frames (of 400) | held-out compositions | order train / held out | ties held out | sick train / held out | buds train / held out | compositions, first set | seconds per instruction |
|---|---|---|---|---|---|---|---|---|---|
| E15: as downloaded, no thinking | 580 (610) | 215 | 56 (77) | 68 (88) / 88 | 78 | 66 / 0 | 67 / 49 | 79 (89) | 0.7 |
| thinking first, greedy | 566 (586) | 181 (239) | 60 (75) | 77 (85) / 58 (73) | 72 | 66 / 0 (28) | 67 / 51 (66) | 56 (68) | 15.6 |
| thinking first, sampled | 386 (432) | 212 (238) | 67 (79) | 58 (84) / 89 (91) | 72 | 100 / 0 (9) | 68 (76) / 51 (66) | 60 (72) | 22.3 |
| **LoRA on program writing** | **700** | **382 (392)** | **64 (86)** | 100 / 82 (92) | **100** | **100 / 100** | **100 / 100** | 100 | 0.5 |

v2's test of its own writer (25 tasks per kind and split, v2's prompt with eight examples, v2's grammar): 100 and 92
of 100 as downloaded, **100 and 100 with the adapters** (the misses were "keep the middle one" in new wordings and an
"unless" read backwards). Nothing of v2 regressed.

- **Thinking does not fix polarity.** Greedy, 82 of 106 instructions ran out of 2048 tokens in a loop of "wait, no";
  sampled, 66 did, and the one prune instruction was written `Cut(Except(All(),ReadLabel()))`, so every `prune` and
  `nothing_to_do` task was lost. On held-out `sick` frames both still write `sick,no`. It costs 20 to 30 times the
  time and is not used.
- **The adapters fix polarity and the situations' held-out wordings**: every training task, and of the held-out frames
  `ties`, `sick` and `buds` 100 of 100 each (78, 0, 49 before), so "the unhealthy shoots have to go" and "spare the
  budding branches and snip the rest" are read right with none of their words or word pairs seen. What is left: one
  held-out order frame of two ("remove the middle branch before the lowest branch", 12 of 30 pair tasks against 18
  as downloaded: "before" is a word it never saw, and it writes one cut or `CutFrom`), and held-out compositions at 64
  (P4 asks 75): "two of the shoots have to go, whichever" is `Cut(All())`, "begin at the bottom and remove each branch
  that has no bud" loses its order, an "unless" and a "should" lose their condition. The training loss is near zero
  within the first epoch; these are what templates of other tasks cannot teach.
- **A verifier was not built.** The gap left on rungs 1 to 4 is one sentence frame; the compositions' gap is rung 5's,
  which no closed loop runs yet (step 9). A verifier that reads the program back in words would ask the same 2B model
  to judge the polarity it got wrong before.

**v3 on the eyes with the written program** (the rollout eyes `eyes_v3_fwp_roll_3m`, the fast level fine-tuned on
ties, evaluation layouts, 200 a cell, `order` 100 a mode; the adapters' writer against the writer as downloaded, run
again on these eyes where it writes any wrong program, and the task's program of step 7):

| Situation | the task's program (step 7) | as downloaded | **LoRA writer** | written right: as downloaded / LoRA | wrong shoots cut: as downloaded / LoRA |
|---|---|---|---|---|---|
| `prune` | 157 | (157) | **166** | 200 / 200 | 0 / 0 |
| `order`, of 300 | 242 | 224 | **257** | 226 / 300 | 4 / 0 |
| `order`: from the top down | 78 | 61 | 86 | 43 / 100 | 0 / 0 |
| `order`: from the bottom up | 79 | 75 | 84 | 100 / 100 | 0 / 0 |
| `order`: a pair | 85 | 88 | 87 | 83 / 100 | 4 / 0 |
| `nothing_to_do` | 198 | (198) | **198** | 200 / 200 | 0 / 0 |
| `ties` | 179 | (179) | **173** | 200 / 200 | (3) / 5 |
| `sick` | 172 | 130 | **168** | 149 / 200 | 30 / 3 |
| `buds` | 180 | 125 | **177** | 136 / 200 | 38 / 3 |
| held-out frames: `order`, of 300 | (242) | 230 | **213** | 283 / 253 | 15 / 19 |
| held-out frames: `order`, a pair | (85) | 72 | 62 | 83 / 53 | 14 / 18 |
| held-out frames: `ties` | (179) | 152 | **173** | 146 / 200 | 33 / 7 |
| held-out frames: `sick` | (172) | 3 | **171** | 0 / 200 | 128 / 2 |
| held-out frames: `buds` | (180) | 80 | **175** | 98 / 200 | 72 / 2 |

In brackets: cells where the writer as downloaded writes the task's program on every task, whose runs are the task
program's.

- **With the adapters the written program costs nothing measurable on rungs 1 to 4**: every cell is within 7 of the
  task's program, in training frames and in held-out ones, except the held-out pair frame. `sick` goes from 130 to 168
  and `buds` from 125 to 177; in held-out frames `sick` from 3 to 171, `buds` from 80 to 175 and `ties` from 152 to
  173. The wrong cuts that a wrong polarity made (30 and 38; 128 and 72 held out) are gone.
- **The held-out pair frame** costs 10 episodes against the writer as downloaded (62 against 72), the one place where
  it is worse: "remove the X branch before the Y branch" is half the time written as one cut.
- **These adapters are v3's writer from here on** (`--writer_lora`, `PICKUP_WRITER_LORA`).

**The fast level on the frozen DINOv2 encoder.** The recipe of the seed-43 sub-goal fast level
(`dp_vit_lang_reactive_subgoal_reason8k_20m`: one frame per camera, the sub-goal, 20M samples from the start, peak
learning rate 5e-4) with the frozen DINOv2-B (registers) patch tokens at 224 pixels in place of its own ViT
(`dp_dinov2_lang_reactive_subgoal`; the 1 s configuration of step 8 needs data preprocessed with the frames a second
back, which the reasoning data is not), on `reason_mix_8k_rs_tail10` and `ties_1500_rs_tail10` at 0.7 and 0.3 of each
batch as the ties fine-tune mixed them, the old normalisation; seeds 43 and 42
(`dp_dinov2_lang_reactive_subgoal_reason8k_ties_20m_seed43`, `_seed42`; final averaged weights). 690 samples a second
alone on an L40S, 440 beside the writer's runs: 9 hours each. Against the ViT fast level fine-tuned on ties (seed 43,
E13b and step 7), the task's program, every run as step 7's:

| | ViT, fine-tuned on ties | **DINOv2, seed 43** | DINOv2, seed 42 |
|---|---|---|---|
| validation layouts, the simulator's scene, of 650 | 572 | **611** | 574 |
| `prune` / `order` of 150 / `nothing_to_do` / `ties` / `sick` / `buds` (validation) | 79 / 118 / 100 / 94 / 87 / 94 | 91 / 135 / 100 / 98 / 93 / 94 | 82 / 121 / 100 / 94 / 84 / 93 |
| cut, across the shoot: median (90th percentile), validation | 8.3 (14.9) mm | 7.9 (13.8) mm | 8.6 (16.6) mm |
| episodes lost, and of them with more than a full turn (validation) | 78, 26 | 39, 7 | 76, 20 |
| **evaluation, the simulator's scene**: `prune` | 161 | **180** | 177 |
| `order`, of 300 | 241 | **281** | 259 |
| `nothing_to_do` | 200 | 200 | 200 |
| `ties` | 177 | **190** | 187 |
| `sick` | 176 | 172 | 161 |
| `buds` | 182 | **192** | 184 |
| v2's `reason_mix` (v2's slow level, clear starts) | 182 | 180 | 174 |
| cut: median, all situations | 8.2 mm (`prune`, step 6) | 8.3 mm | 8.8 mm |
| lost, of 1300; with more than a full turn | 163; 44 | 85; 19 | 132; 30 |
| **evaluation, the rollout eyes**: `prune` | 157 | **181** | 170 |
| `order`, of 300 | 242 | **262** | 249 |
| `nothing_to_do` | 198 | 198 | 198 |
| `ties` | 179 | **194** | 190 |
| `sick` | 172 | 169 | 156 |
| `buds` | 180 | **190** | 174 |
| v2's `reason_mix` | 177 | **193** | 180 |
| cut: median | 8.3 mm | 8.5 mm | 8.6 mm |
| lost, of 1300; with more than a full turn | 172; 54 | 106; 21 | 163; 37 |

v2 in its headline configuration (the eyes `eyes_vit_fwp_2m`, v2's written program, the memory's two options, up to two
resets; `reason_mix` in training and held-out wordings, 200 evaluation layouts), run in this code over each fast level:
v2's own fast level (the seed-43 ViT, not fine-tuned on ties) 182 and 153 (the published run: 180 and 165; over 5
and 7 repeats in both codes 179 and 160 on average, see "The headline, reproduced" in v2's section), DINOv2 seed 43
175 and 164, seed 42 169 and 156.

- **The cut is no closer.** The jaws close 7.9 to 8.8 mm from the shoot's axis on every set of layouts, as the ViT
  fast level's 8.2 to 8.3, against the expert's 5.9. Neither a pretrained encoder nor the takeover data of step 7 moves
  it, so it is not set by what the encoder lets the policy see; where it is set (the action head, the demonstrations'
  own spread of placements, the control rate) is not separated here.
- **Over-turning falls by half or more.** Episodes lost with the plate turned more than a full turn: 7 and 20 against
  26 on validation layouts, 19 and 30 against 44 of 1300 on the simulator's scene, 21 and 37 against 54 on the eyes.
  Seed 43 loses half as many episodes as the ViT fast level in all (85 against 163 on the scene).
- **Seed 43 is better on every rung but `sick`, seed 42 is close to the ViT's.** On the rollout eyes seed 43 solves
  1194 of 1300 situation episodes against 1128 (seed 42: 1137); `sick` is the one situation where both seeds are
  below the ViT's (169 and 156 against 172), with more lost turning. The two seeds differ by 57 episodes, as much as
  seed 43 differs from the ViT: the gain is not yet a property of the encoder alone.
- **v2's headline does not move** (175 and 164 against 182 and 153 in the same code): on `reason_mix` from v2's own
  starts the fast levels are within the seeds' spread; from v3's clear starts with v3's eyes seed 43 solves 193
  against 177.
- **The seed-43 DINOv2 fast level is v3's fast level from here on**, chosen on validation layouts (611 against 572 and
  574). Whether it keeps step 8's robustness to light on the reasoning tasks is not measured here.

**v3 whole, with both changes** (the adapters' writer, the rollout eyes, the seed-43 DINOv2 fast level; evaluation
layouts, 200 a cell, `order` 100 a mode), against step 5's written-program run (the writer as downloaded, step 6's
eyes, the ViT fast level fine-tuned on ties) and the task's program on the rollout eyes over the ViT (step 7):

| Situation | step 5: written | step 7: the task's program | **written, adapters, DINOv2** | written right | wrong shoots cut | lost with more than a full turn |
|---|---|---|---|---|---|---|
| `prune` | 160 | 157 | **186** | 200 | 0 | 1 |
| `order`, of 300 | 215 | 242 | **276** | 300 | 1 | 10 |
| `nothing_to_do` | 199 | 198 | **197** | 200 | 0 | 1 |
| `ties` | 168 | 179 | **194** | 200 | 1 | 0 |
| `sick` | 125 | 172 | **172** | 200 | 2 | 4 |
| `buds` | 121 | 180 | **189** | 200 | 1 | 4 |
| held-out frames: `order`, of 300 | 220 | (242) | **245** | 253 | 15 | 10 |
| held-out frames: `ties` | 147 | (179) | **195** | 200 | 0 | 0 |
| held-out frames: `sick` | 2 | (172) | **175** | 200 | 0 | 4 |
| held-out frames: `buds` | 94 | (180) | **187** | 200 | 3 | 0 |

- **v3 with a written program now beats the task's program of step 7 on every situation but `nothing_to_do` (197
  against 198) and `sick` (level)**, in training and in held-out frames: of 2600 episodes 2016 solved against 1451 with
  step 5's writer. All but one held-out frame are read right; what is left is that frame (the held-out pair, 66 of
  100, 14 wrong shoots) and the fast level's own losses.
- The cut is where it was (median 8.4 mm over 3098 cuts).

Runs: two fast levels (20M samples each, 9 hours each), one adapter training (41 minutes), the writer alone in four
configurations (v3) and two (v2), 2 validation and 4 evaluation grids of the fast levels (7,300 episodes), 6 v2
headline runs (1200), and 39 closed-loop runs with written programs (6,800 episodes), in
`results/pickup/v3fix/`. GPUs 6 and 7 of an L40S machine were held for 11.5 and 11.7 hours of wall time (23.2
GPU-hours), 18.1 of them training the two fast levels beside the rest.

### The cut, diagnosed: where the fast level's 8 mm come from

Every fast level tried closes its jaws 7.9 to 8.8 mm from the shoot's axis, the expert 5.9 mm, and neither a
pretrained encoder nor takeover data moved it. Diagnosed first, cheaply, with the DINOv2 seed-43 fast level
(`sim/pickup/cut_diag.py`: the simulator and the fast level in one process, `prune` on validation layouts from clear
starts; the scripted expert chooses the shoot and turns the plate; each cut measured as `cut_places` measures it, and
the offset also split in the hand's frame: along the closing jaws, which close vertically, and across them).

**(c) The expert's own cut is a consistent target.** Its set-point is always on the shoot's axis (0.2 to 0.7 mm in its
commands over the last 5 cm), at a point drawn from 31 to 39 mm out along it. Where its hand really is when the shoot
is cut (200 episodes, 488 cuts): 5.8 mm median, 6.2 mean, 9.8 at the 90th percentile, almost all of it along the
closing jaws (standard deviation 4.1 mm; 0.8 mm across them): the hand still settling toward the set-point when the
jaws shut (it closes once within 12 mm of it, and the cut comes 3 control steps later). Two runs of the expert agree
to 0.1 mm. So 5.9 mm is the floor this expert's demonstrations teach, and the right thing to compare with.

**(a) Perception or control: the expert drives, then hands the arm to the fast level** at a distance from the middle
of the shoot's cut zone, with the jaws open and the sub-goal of that shoot (`--mode handover`, 150 episodes each, the
first cut of each episode):

| Handed over | cuts | across the shoot: median | mean | 90th percentile | spread along the jaws (sd) | across the jaws (sd) |
|---|---|---|---|---|---|---|
| the expert alone | 488 | 5.8 mm | 6.2 | 9.8 | 4.1 | 0.8 |
| at 1.5 cm | 148 | 5.7 mm | 6.0 | 9.5 | 4.1 | 1.0 |
| at 3 cm | 147 | 5.8 mm | 6.1 | 9.5 | 4.0 | 1.5 |
| at 6 cm | 149 | 6.0 mm | 6.3 | 9.7 | 4.4 | 1.3 |
| at 10 cm | 149 | 6.4 mm | 6.7 | 10.4 | 4.8 | 1.4 |
| as soon as the shoot is chosen and in reach (two runs) | 147, 143 | 8.1, 7.4 mm | 8.3, 7.5 | 14.5, 13.6 | 7.2, 6.3 | 2.9, 2.5 |

- **From the expert's state 6 cm out the fast level cuts as well as the expert** (6.0 mm against 5.8), from 10 cm
  nearly (6.4). The last centimetres, the closing and what the 224-pixel cameras show there are not the limit.
- **The error is made on the fast level's own way in, before 10 cm, and never corrected.** Where its hand passes 6 cm
  from the cut zone, on its own approach: along the jaws (height) a standard deviation of 6.8 mm against the expert's
  4.2, across them 5.4 against 0.7, and its pointing 3.1 degrees from square across the shoot on average, 11.7 at the
  90th percentile, against 0.0. From there to the cut it keeps the height error (correlation 0.93, slope 0.93 between
  the offset at 6 cm and at the cut) and the angle, and takes out only a third of the sideways error. The expert's
  demonstrations never show a correction there: its line of approach is set 13 to 15 cm out and followed straight.
- **Open loop, at the expert's own states** (60 episodes), the fast level's chunks are where the expert's commands are
  to 2.5 to 3.5 mm over the 8 actions executed within 5 cm of the cut zone (4.1 from 5 to 8 cm), and the hand they lead
  to is 2.2 to 2.8 mm from the axis (the expert's 0.2 to 0.7); 8 chunks drawn at the same state lie 1.8 to 2.1 mm
  apart. Small next to the 6 to 7 mm the
  closed loop adds: the gap is compounding error on states the demonstrations do not cover.

**(b) Inference settings near the cut** (handed over as soon as the shoot is chosen, 150 episodes each): 8 actions
executed per chunk 8.1 and 7.4 mm (two runs; the noise between runs is about 0.7 mm); 4 actions 7.7; 2 actions 8.4 (4
episodes damaged); 8 chunks drawn per look and averaged 7.7; 4 actions and 8 chunks averaged 7.8; 30 flow steps instead
of 10 breaks the policy (13.4 mm, 68 of 150 episodes damaged; it was trained and always run with 10). No setting moves
the cut beyond the noise: the error is not in how the chunks are sampled or executed.

**(d) Resolution: the encoder sees the error; the policy does not act on it.** The expert approached along its line
moved by up to 10 mm up or down and sideways (300 `prune` episodes, 2191 frames where its hand passed 10, 6 and 3 cm
from the cut zone; `sim/pickup/cut_probe.py`), and a ridge regression on the frozen DINOv2-B patch tokens of one image,
fitted on four fifths of the episodes, read the offset of the shoot's axis in the hand's frame on the rest:

| View (224 pixels to the encoder) | along the jaws (height), sd 6.6 mm | across them, sd 2.5 mm | along the shoot, sd 6.3 mm |
|---|---|---|---|
| wrist | 1.6 mm at 10, 6 and 3 cm | 0.7 to 0.8 | 1.9 to 2.2 |
| wrist rendered at 448, its middle kept (twice the pixels) | 1.2 to 1.3 | 0.5 to 0.6 | 1.7 to 2.2 |
| front | 4.4 to 4.6 | 2.1 to 2.2 | 3.9 to 4.0 |

  The wrist image at 224 pixels already tells the height of the jaws against the shoot to 1.6 mm from 10 cm out, a
  quarter of the fast level's 6.8 mm error there; twice the pixels takes 0.3 to 0.4 mm off that. And handed the arm at
  10 cm on a line moved 6 mm up or down (150 episodes), the fast level keeps 93% of the offset to the cut (the slope
  of the offset at the cut on the offset at the handover; 0.94 when handed over at 1 cm, where no correction is
  possible): it does not correct an error its encoder can read. So no high-resolution crop was trained: the limit is
  what the policy learned to do with what it sees (its demonstrations never correct the line), not the pixels.

**The fix tried: demonstrations that show the correction (DART-style, quiet near the cut).** `collect.py
--action_noise 0.02 --action_noise_quiet_cm 3`: the expert's executed joint targets get a smooth random offset
(standard deviation 0.02 rad, correlation 0.9 a step), its own commands are recorded as the actions, and no fresh noise
is added within 3 cm of an uncut shoot's cut zone or with the jaws commanded shut (`ActionNoise.calm`), so the offset
dies away and the demonstrations show the hand brought back onto the line before the cut. The size was chosen with the
expert alone (80 `prune` episodes each):

| Expert with noise | solved | damaged | at 6 cm: sd along the jaws | pointing (mean, 90th pct) | its cut: median |
|---|---|---|---|---|---|
| none | 196 of 200 | 0 | 4.2 mm | 0.0, 0.0 degrees | 5.8 mm |
| 0.01 rad, everywhere | 63 | 17 | 6.4 | 0.7, 1.5 | 8.1 |
| 0.01 rad, quiet within 3 cm | 70 | 5 | 6.1 | 0.7, 1.4 | 6.3 |
| **0.02 rad, quiet within 3 cm** | 53 | 13 | **10.1** | 1.3, 2.5 | 6.9 |
| 0.02 rad, everywhere | 21 | 42 | 10.0 | 1.4, 2.7 | 8.7 |
| 0.03 rad, quiet within 3 cm | 37 | 20 | 14.4 | 2.0, 4.1 | 6.7 |
| 0.03 rad, everywhere | 2 | 58 | 14.5 | 1.9, 3.7 | 9.6 |

Noise of 0.02 rad pushes the hand off its line of approach by more than the fast level's own errors (10.1 mm at 6 cm
against 6.8), and quiet near the cut the expert still cuts at 6.9 mm: only successful episodes are kept, with the clean
commands as labels. 3000 `reason_mix` demonstrations from random starts (seeds from 400,000, 4758 attempted, 867,122
frames: `reason_mix_3k_rs_tail10_dart02q3`). The DINOv2 seed-43 fast level fine-tuned from its averaged weights for 3M
samples at a peak learning rate of 2e-4 on `reason_mix_8k_rs_tail10`, `ties_1500_rs_tail10` and these (0.45, 0.2,
0.35 of each batch, the old normalisation: `dp_dinov2_lang_reactive_subgoal_reason8k_ties_dart_ft3m_seed43`), and as
the control the same fine-tune without them (0.7, 0.3: `..._ctrl_ft3m_seed43`), which step 7 lacked.

**Result** (the DART fine-tune against the fast level as it is and the control fine-tune):

| | DINOv2 seed 43 | control fine-tune | **DART fine-tune** |
|---|---|---|---|
| handed over as soon as the shoot is chosen (150): cut, median (mean, 90th pct) | 8.1, 7.4 (8.3, 7.5; 14.5, 13.6) mm | 7.3 (7.7, 12.3) | 7.3 (7.4, 11.6) |
| a 6 mm height error handed over at 10 cm: kept at the cut | 93% | | 75% |
| validation layouts, the simulator's scene, of 650 | 611 | 614 | 608 |
| cut on validation: median (mean, 90th pct) | 7.9 (8.4, 13.8) mm | 7.8 (8.2, 13.6) | 7.6 (7.8, 12.6) |
| v3 whole (LoRA writer, rollout eyes, written programs; evaluation layouts, the 14 cells above), of 2600 | 2016 | | 1990 |
| ... cut over all its cuts: median (mean, 90th pct) | 8.4 (8.7, 14.2) mm | | 8.0 (8.1, 12.4) |
| v2's `reason_mix` (v2's slow level, clear starts): the simulator's scene / the rollout eyes | 180 / 193 | | 189 / 189 |

v3 whole, cell by cell for the DART fine-tune (against DINOv2 seed 43): `prune` 181 (186), `order` 264 of 300 (276),
`nothing_to_do` 198 (197), `ties` 195 (194), `sick` 171 (172), `buds` 184 (189); held-out frames `order` 251 (245),
`ties` 195 (195), `sick` 164 (175), `buds` 187 (187).

- **The demonstrations of the correction help a little and not enough.** The fine-tune now takes out a quarter of a
  height error from 10 cm (75% kept against 93%) and narrows the spread: the mean cut 0.4 to 0.6 mm nearer and the 90th
  percentile 1.2 to 1.8 mm nearer, on validation layouts and on all 3060 cuts of v3 whole. The median moves 0.3 to 0.4
  mm, the same as the control fine-tune moves it in the handover test, and stays 8.0 against the expert's 5.9.
- **No success is gained** (608 against 611 on validation, 1990 against 2016 of 2600 in v3 whole, within noise of
  14 cells; `reason_mix` level). By the rule set for this change (kept only if the cut is clearly closer and nothing is
  lost) **the DART fine-tune is not adopted**: v3's fast level stays the DINOv2 seed-43 one.
- **Inference settings and resolution are ruled out, and so is the encoder**: what the cut needs is a policy that
  corrects its line from what the wrist camera shows, which these demonstrations teach weakly at 35% of the batches for
  3M samples. A stronger dose (DART demonstrations as most of the data, noise that reaches nearer the cut, a 20M run
  from the start) is the next step; a scripted correction from the simulator's truth would only hide it.

Runs: 29 diagnosis runs (`cut_diag.py`: 3840 episodes; `cut_probe.py`: 300), one DART collection (3000 kept of 4758,
11 minutes on 48 processes), two fine-tunes (3M samples each, 84 and 79 minutes beside the rest), two validation
grids (1300 episodes), v3 whole over the DART fine-tune (2600) and `reason_mix` (400), in `results/pickup/cutdiag/`,
`results/pickup/v3fix/fast/` and `results/pickup/v3fix/writer_loop/lora_dart43/`; with the v2 reproduction below
(10 runs, 1800 episodes), GPUs 6 and 7 of an L40S machine were held for 3.4 hours of wall time each (6.8
GPU-hours).

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
| The rotary position table was computed in bfloat16 under autocast, which keeps 8 significant bits | positions above 256 collapse onto even numbers, above 512 onto multiples of 4: 641 distinct positions out of 2048. In the no-history policy (426 tokens before the actions: one for the noise level, 32 for the instruction, 392 patches, one for the joint state) the 15 action-step tokens, at positions 426-440, shared 8 positions; with history (about 820) 16 shared 4 | the table is computed in float32 and then cast; test added. Effect: see 'The same run before and after the fix' (and the second pair of seeds in 'A second seed, with and without the fix') and the single cube with history |
| Evaluations shared a port (see Operations) | results filed under the wrong run | `eval.sh` takes a free port and waits for its own server; the client stops unless the server reports the checkpoint it asked for; `summarize.py` leaves out results whose recorded checkpoint does not match |
| The previous-action input of a window at the start of a demonstration is filled with a copy of the first action | the expert's first command, which already points at the target, is an input in training and absent when served (see "What the history policy depends on") | `collect.py --hold_steps 1` begins every demonstration with a frame of the robot at rest; it did not help (see 'History, on demonstrations that begin with a hold frame') |
