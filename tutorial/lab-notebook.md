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

## The scripted expert

| Measurement | Result |
|---|---|
| Expert through the evaluation loop, `pick_target`, 100 layouts (machine B) | 100/100, gripper closes 5 mm (median) from the object, 98% within 10 mm |
| Same on machine A | 99/100 |
| Expert spot check, `pick_cube` | 49/50 |

## Grasp tolerance

Expert with a fixed sideways offset added to its grasp point, random direction, 320 episodes per
offset (machine A):

| Offset | Cube | T block | Can | Hex nut | All |
|---|---|---|---|---|---|
| 0 mm | 100% | 100% | 94% | 89% | 96% |
| 5 mm | 97% | 100% | 87% | 83% | 92% |
| 10 mm | 99% | 100% | 60% | 61% | 80% |
| 15 mm | 95% | 100% | 35% | 37% | 67% |
| 20 mm | 93% | 98% | 18% | 25% | 58% |
| 25 mm | 73% | 89% | 11% | 21% | 49% |
| 30 mm | 32% | 76% | 0% | 10% | 30% |

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

Same 1500 demos (98,841 windows), 1M samples, final checkpoint, 100 layouts. "History" = frames
t-1 and t, states t-1 and t, and the previous action as an input. "No history" = one frame, the
current state, no previous action.

| Policy | Success (raw / EMA weights) | Nearest object was the named one | Median miss to named | Median miss to nearest |
|---|---|---|---|---|
| `dp_vit_reactive`: no language, no history | 1% / 1% | 26% / 34% | 175 / 173 mm | 95 / 92 mm |
| `dp_vit_lang_reactive`: instruction tokens, no history | 3% / 5% | 49% / 41% | 127 / 139 mm | 71 / 75 mm |

By checkpoint, `dp_vit_lang_reactive` (raw / EMA): named-object rate 28% / 31% at 0.3M samples, 25%
/ 30% at 0.5M, 34% / 46% at 0.7M, 45% / 48% at 0.9M, 49% / 41% at 1.0M. The no-language policy stays
between 26% and 39% throughout.

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

Runs still in progress when this was written: the history variants, a 4000-demo / 5M-sample pair,
the Qwen VLA, two data variants (every layout demonstrated with each object named; the same with the
first frames of each episode up-weighted), and the from-scratch stages for an all-token VLA. Their
rows are in the generated table at the end of this file.

## Throughput

| Model | Machine A (RTX 4090) | Machine B (L40S) |
|---|---|---|
| `dp_vit` / `dp_vit_lang` (4 images) | about 620 samples/s | about 540 samples/s |
| `dp_vit_reactive` / `dp_vit_lang_reactive` (2 images) | not run | about 1100 samples/s |
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
- Collecting 1500 demonstrations took 70 s with 48 simulator processes.

## Bugs found and fixed

| Problem | Effect | Fix |
|---|---|---|
| `SimpleVLMProcessor` scaled PIL and numpy images to [0, 1] but left uint8 tensors at 0-255; training fed tensors | the VLM ignored images at inference (captioning 0%) | scale uint8 tensors too; test added |
| Action token appended after the batch padding (`vlm_foundry_backbone`) | a short instruction is probed one position later, behind a padding token, in a training batch than when served alone | insert it after the last real token; test added. Effect on success not measured |
| `lowdim_past_timesteps or default` treats 0 as unset | with 0 past steps the proprioception input contains the state one step in the future | explicit `is None` check; test added |
| Both VLA recipes condition the action head on one vector | see the `pick_target` tables | `conditioning: all_tokens` for the from-scratch VLM backbone; effect not yet measured |
| A dead data-loader worker leaves the trainer waiting forever | runs hang at 0% GPU | `train_resilient.sh` restarts on exit or on a stalled log |
