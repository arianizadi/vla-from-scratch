---
title: One Vector or Many Tokens
part: 2 Vision
order: 21
summary: Tell a pooled image embedding from a sequence of patch tokens, predict which one a manipulation policy needs, and find where this repository produces each.
status: draft
---

# One Vector or Many Tokens

A vision encoder can hand the rest of the model one vector for the whole image, or one vector for
every patch of it. The single vector is compact and says what is in the picture; the patch vectors
are many and also say where, which a policy needs to close a gripper within a centimetre or two of
an object. In the runs here patch tokens were necessary and not sufficient: a policy given one
vector per image lifted a single cube in 2 of 50 attempts, and on a harder task the same patch
tokens gave 0% or 85% depending on what else was right (93-98% once a position bug was fixed).

## The idea

[Images Become Tokens](20-images-to-tokens.md) showed how a vision transformer (ViT) cuts an image
into square patches and turns each into a vector. A **patch token** is the encoder's output vector
for one patch. A **pooled embedding** is a single vector that summarises all of them.

For one 224x224 frame from the `front` camera:

- The ViT in [`dp_vit.yaml`](../../sim/pickup/configs/dp_vit.yaml) uses 16-pixel patches, so the
  frame becomes a 14x14 grid and the encoder returns `[1, 196, 256]`: 196 tokens of width 256,
  50,176 numbers.
- CLIP is an image encoder pretrained to match images with their captions. The one in
  [`dp_clip.yaml`](../../sim/pickup/configs/dp_clip.yaml) returns `[1, 512]`: 512 numbers for the
  whole frame, with no axis for position.

The pooled output is 98 times smaller, but size is not the main difference. Pooled embeddings
usually come from encoders pretrained to classify images or to match them with captions: they are
trained to answer "what is in this image". A grid of patch tokens keeps the layout of the image in
its second axis, so a later layer can ask "what is at this place".

<!-- interactive: pooled-vs-tokens -->
*Figure: the same camera frame encoded two ways: 196 patch tokens reach the action head, or one
vector in which the cube's position is no longer a separate input.*

## How it works

### Three ways to make one vector

1. **Class token.** A learned extra vector is put in front of the patch sequence before the first
   transformer block. Attention lets it read from every patch, and its state after the last block
   represents the image. Background: the original ViT (Dosovitskiy, 2020) and CLIP's ViT variants
   (Radford, 2021) work this way, including the ViT-B/32 (base size, 32-pixel patches) used here.
2. **Mean pooling.** The patch tokens (or the cells of a convolutional feature map) are averaged.
3. **Attention pooling.** One extra attention layer with a single query produces a weighted average
   of the tokens, with weights that depend on the content.

All three remove the position axis: whatever location information survives is in the values of
one vector.

### What a pooled embedding is trained to keep

Background, from the CLIP paper (Radford, 2021): CLIP trains an image encoder and a text encoder
together on image-caption pairs. In a batch of $N$ pairs it raises the cosine similarity of the
$N$ matching image and text vectors and lowers it for the $N^2 - N$ mismatched ones. The image side
of that comparison is the pooled embedding.

The image vector is therefore rewarded for whatever tells one caption from another: which objects
are present, their colour, the kind of scene. "A red cube on a table" is equally true when the cube
sits three centimetres further left, so the loss gives no reward for telling those two images
apart. This is an argument about the objective, not a measurement of what CLIP's vector contains,
and it describes only where the encoder starts: `dp_clip` trains the CLIP weights with the policy
(`freeze_image_encoder` and `freeze_text_encoder` default to `False` in `CLIPHFParams`; no config
under `sim/pickup/configs/` sets them).

Three centimetres decide a grasp. Measured here: with its grasp point offset sideways by 30 mm,
the scripted expert, the hand-written controller behind the [demonstrations](70-demonstrations.md),
succeeds in 32% of the episodes that name the cube, against 93% at 20 mm
([Measuring How a Policy Fails](92-measuring-failure.md)).

### What patch tokens keep

Each patch token starts as a projection of one fixed square of pixels. Attention by itself does
not see a token's index ([Telling Tokens Where They Are](12-positions.md)); its place is readable
for two reasons. `ViTPatchEmbeddings` adds a learned position embedding for that square before the
first block, so the token's values carry it. And the action transformer (the action head, which
turns noise into actions) applies rotary position encoding along its sequence
(`positional_embedding_type: rotary`, the default). Its attention can then weight the few tokens
that cover the cube and read from them where it is.

The rotary table was faulty in every run in the tables below. Computed in bfloat16 under autocast
(mixed precision), it held 641 distinct positions out of 2048: positions above 256 collapsed onto
even numbers and above 512 onto multiples of 4. Every sequence with patch tokens here is longer
than 256 tokens; the pooled policy's 24 are not. `RotaryEmbedding._update_cos_sin_tables` in
`vla_foundry/positional_embedding.py` now computes the table in float32. Fixed, the 85% recipe below
ends at 93% (trained weights) against 83%. A second pair of runs with other seeds, differing in the
table alone, ends at 96% / 96% against 90% / 89%. In both pairs the policy starts to look earlier
([chapter 12](12-positions.md)).

### Shapes in the policies trained here

`dp_clip` and `dp_vit` use two cameras and two frames per camera (`image_indices: [-1, 0]`, control
steps t-1 and t): four images at a time. Both feed a flow-matching action transformer
([Generating Actions with Flow Matching](41-flow-matching.md)).
`DiffusionPolicy._build_transformer_input` in
[`vla_foundry/models/diffusion_policy/diffusion_policy.py`](../../vla_foundry/models/diffusion_policy/diffusion_policy.py)
builds its input sequence from four parts:

- one **time token** for the flow-matching time, that is, how noisy the actions being generated
  currently are (not the control step);
- the **conditioning** tokens from the backbone (the image and text encoder), which it reads but
  does not predict;
- the state tokens, the robot's state at control steps t-1 and t;
- the **action-step tokens**: the previous action, given without noise, and the **chunk** being
  generated, 15 actions from control step t on.

The previous frame, state and action are the policy's **history**.
[`dp_vit_lang_reactive.yaml`](../../sim/pickup/configs/dp_vit_lang_reactive.yaml), the config that
solves the harder task below, has none (`image_indices: [0]`, `lowdim_past_timesteps: 0`) and adds
the instruction as 32 token slots (`text_tokens: 32`).

| | `dp_clip` (pooled) | `dp_vit` (patch tokens) | `dp_vit_lang_reactive` |
|---|---|---|---|
| Conditioning | `[B, 5, 512]` (1 text + 4 images) | `[B, 784, 256]` (4 x 196) | `[B, 424, 256]` (32 text slots + 2 x 196) |
| Time + state + action-step tokens | 1 + 2 + 16 | 1 + 2 + 16 | 1 + 1 + 15 |
| Sequence length | 24 | 803 | 441 |
| Token pairs per attention layer | 576 | 644,809 | 194,481 |

### The cost of tokens

Self-attention compares every token with every other
([The Transformer Block](11-the-transformer-block.md)), so its compute grows with the square of the
sequence length, and so does its memory whenever the attention matrix is materialised. Going from
24 to 803 tokens is 33 times more tokens and about 1,100 times more token pairs per attention
layer. The number of patch tokens is cameras x frames per camera x (image size / patch size)².
When actions are generated the cost is paid once per flow step: `DiffusionPolicy.generate_actions`
computes the backbone output once and then runs the action transformer for every flow step (10 by
default in `sim/pickup/eval.py`).

These are sequence lengths, not measured speed; the pooled policy's throughput was not recorded.
On machine B (L40S GPUs, 128 CPU cores) the patch-token policies trained at about 540 samples/s
with four images (803 or 835 tokens) and about 1100 with two (409 or 441), alone on a GPU.

## In this repository

**The pooled path.** [`vla_foundry/models/diffusion_policy/clip_hf.py`](../../vla_foundry/models/diffusion_policy/clip_hf.py)
wraps the Hugging Face `CLIPModel`. In `CLIPHF.forward`, images go through
`self.model.vision_model(pixel_values).pooler_output` and `self.model.visual_projection`, and are
then reshaped from `[B*N, D]` to `[B, N, D]` and L2-normalised. The instruction becomes one vector
too: the text model's `pooler_output`, normalised and passed through `self.model.text_projection`.
Background: in `transformers`, the vision model's `pooler_output` is the class token's final state
after a layer norm; `CLIPHF.forward` never reads the patch tokens.
`CLIPBackboneWrapper._concatenate_for_conditioning` in
[`vla_foundry/models/vision_language_backbones/clip_hf_backbone.py`](../../vla_foundry/models/vision_language_backbones/clip_hf_backbone.py)
returns the text vector followed by the image vectors, `[B, 1+N, D]`.

**The token path.** `ViTBackboneWrapper.get_action_conditioning` in
[`vla_foundry/models/vision_language_backbones/vit_backbone.py`](../../vla_foundry/models/vision_language_backbones/vit_backbone.py)
calls the `ViT` of [`vla_foundry/models/vit.py`](../../vla_foundry/models/vit.py), which returns
`[B, N, num_patches, hidden_dim]`, and flattens cameras and patches into one sequence. With
`text_tokens` above 0, `_embed_text` turns the instruction's token ids into that many slots, placed
in front of the patch tokens ([Language conditioning](51-language-conditioning.md)). The ViT can
pool as well: with `cls_flag` set (`ViTParams` defaults it to `False`; no config under
`sim/pickup/configs/` sets it), `ViT.forward` keeps only the class token's state.

**Config keys.** `model.vision_language_backbone.type` selects the path: `clip_backbone` in
`dp_clip.yaml` (from the preset `vla_foundry/config_presets/models/diffusion_policy.yaml`),
`vit_backbone` in `dp_vit.yaml`, whose `data.processor: "none"` means that the instruction never
reaches the model.

To repeat this chapter's comparison on 200 demonstrations:

```bash
sim/pickup/scripts/make_dataset.sh pick_cube 200 pick_cube_200
for cfg in dp_clip dp_vit; do
  sim/pickup/scripts/train.sh $cfg pick_cube_200 ${cfg}_pick_cube_200 300000
  sim/pickup/scripts/eval.sh experiments/pickup/${cfg}_pick_cube_200 pick_cube 50
done
```

That is a comparison, not a recipe. The first real run the [README](../../README.md) recommends is
the same patch-token policy trained on demonstrations with random start poses and recoveries, which
reached 94% / 94% after the position fix:

```bash
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k --random_start --disturb_prob 0.5
sim/pickup/scripts/train_resilient.sh dp_vit_lang_reactive pick_target_4k my_run 5000000
sim/pickup/scripts/eval_checkpoints.sh experiments/pickup/my_run pick_target
```

## What we measured

### One cube: pooled against patch tokens

`pick_cube` (one red cube at a random pose), machine A (one RTX 4090, 24 GB), one training seed per
row, 50 closed-loop episodes (the policy drives the simulated arm, so its own actions decide what
it sees next) on layouts it never saw. A **layout** is what a seed fixes: the objects, their poses,
the one named and the arm's start pose. [Closed-Loop Evaluation](90-closed-loop-evaluation.md)
explains the Wilson 95% intervals.

| Policy | Demos | Training samples (epochs) | Final loss | Success | 95% CI |
|---|---|---|---|---|---|
| `dp_clip`: CLIP ViT-B/32, one pooled vector per image and one for the text, 245M | 200 | 300k (23) | 0.040 | 2/50 = 4% | 1-14% |
| `dp_vit`: small ViT from scratch, every patch a token, 36M | 200 | 300k (23) | 0.091 | 15/50 = 30% | 19-44% |

The intervals do not overlap: on the same demonstrations and samples the patch-token policy
grasped and the pooled one did not. The pooled-CLIP policy, which had the lower training
loss ([Loss Is Not Success](91-loss-is-not-success.md)), also scored 0/20 on its own training
layouts: it ran a plausible reach-descend-close-lift a few centimetres from the cube.

What the comparison does not show:

- A controlled ablation of pooling. The policies also differ in pretraining, patch size (32 against
  16), encoder and action-transformer size, text input and image preprocessing. CLIP's own patch
  tokens were not tested ([Pretrained Vision Encoders](22-pretrained-vision-encoders.md)).
- That one vector cannot carry position. CLIP was trained with the policy, and after 300k samples
  the policy still missed. A longer run, a frozen encoder and a direct test of what the vector
  holds were not tried, and the slope defined below was not recorded.
- A working policy. With 1000 demonstrations and 1.5M samples the patch-token recipe (with
  history) reached 37/50 = 74% (60-84%) on machine A and 62% / 70% on machine B (100 layouts; raw /
  EMA: trained weights / their running average). The position table, not history, held it there:
  trained again after the fix it ends at 96% / 98% ([chapter 52](52-history-and-shortcuts.md)).

### Three objects: the same tokens from 0% to 85%, before the position fix

`pick_target` puts three of four objects on the table and the instruction names one. Machine B, 100
unseen layouts per row, raw / EMA weights; in every row the action head receives the patch tokens
of the small ViT above.

| Policy | Demos | Training samples | Success | Closed at the named object | Slope |
|---|---|---|---|---|---|
| `dp_vit`: no language, history | 1500 | 1.0M | 0% / 0% | 31% / 37% | 0.11 / 0.17 |
| `dp_vit_lang`: language, history | 4000 | 3.0M (stopped) | 2% / 2% | 30% / 35% | 0.11 / 0.08 |
| `dp_vit_lang_reactive`: language, no history | 1500 | 1.0M | 3% / 5% | 49% / 41% | 0.29 / 0.18 |
| same | 1500 | 3.0M | 64% / 62% | 99% / 100% | 0.96 / 0.97 |
| same | 4000 | 5.0M | 83% / 85% | 100% / 100% | 0.98 / 0.98 |

**Closed at the named object**: the object nearest the gripper when it first closed was the named
one (chance 33%). **Slope**: the least-squares slope of where the gripper first closed against
where the named object was; 1 if it closes where the object is, 0 if where it closes does not
depend on the scene.

- The failing rows had every patch token and did not use them. With slopes of 0.29 and below they
  were neither near misses nor reaches between the objects: where the gripper closed depended
  little or not at all on the scene
  ([Measuring How a Policy Fails](92-measuring-failure.md)).
- So patch tokens were not sufficient. The policy also needed the instruction as tokens, no
  history, training that outlasts the late start of looking (the last row's run was at 4% / 36%
  after 2.0M samples) and, raising 83% to 93%, a position table in float32
  ([How much data and training](71-how-much-data-and-training.md); the whole route is the
  [`pick_target` case study](101-case-pick-target.md)).
- The history row reads the target from its previous action: at 2.5M samples it succeeded in
  77/100 layouts when the scripted expert drove the first 10 control steps, against 1/100 as served
  ([chapter 52](52-history-and-shortcuts.md)).
- That patch tokens were necessary is measured on `pick_cube` only. The first-round policies with
  one conditioning vector scored 0/100, 1/100 and 6/100 here (machine A), but with history and at
  most 1.0M samples on 1500 demonstrations, where the third row stood at 3% / 5%. Handed every token,
  with no history and 4000 demonstrations, the same VLAs reach 86% / 87% (from scratch, 3M samples)
  and 70% / 72% (Qwen, 0.5M), but history and demonstrations changed too
  ([chapter 50](50-what-the-action-head-sees.md)).

## Where it goes wrong

**The loss is low, the arm moves well, and it grasps air.** Conditioning without position is one
cause among several. First evaluate on training layouts: demonstrations use environment seeds from
0 upwards, so `--seed_start 0` replays them.

```bash
PICKUP_OUT=results/pickup/dp_clip_pick_cube_200/pick_cube_train \
  sim/pickup/scripts/eval.sh experiments/pickup/dp_clip_pick_cube_200 pick_cube 20 --seed_start 0
```

Near-zero success there too rules out generalisation. Then read how it failed: `eval.py` prints
how often the gripper closed at the named object and the median miss, and
`.venv/bin/python -m sim.pickup.summarize` the slope. A slope near 1 is an imprecise policy. A slope near 0
is a policy that is not looking: check that the conditioning is more than a few vectors, that the
policy is not given its previous action, and that training did not end before the slope moved.

**The same bottleneck under another name.** A vision-language-action model (VLA) can pass hundreds
of image tokens through its language model and still hand the action head one vector. Both VLA
recipes here do so by default: `_prepare_inputs_for_action_token` in `VLMHFBackboneWrapper` and in
`VLMFoundryBackboneWrapper` adds one special token, the action token, after the prompt, and only
its hidden states reach the action head, shape `[B, 1, lm_hidden_dim * num_vlm_layers_to_use]`:
the class-token method again. [Chapter 50](50-what-the-action-head-sees.md) has the wiring and the
`conditioning: all_tokens` alternative, measured above.

**The tokens arrive and are read once.** Sent the first frame's images for the whole episode
(`eval.py --stale_images`), the 85% policy still closed at the named object in 99% of episodes,
17 mm off (median) against 7 mm, and succeeded in 1/100. It chooses where to go from its first
look and then follows its own arm; the images matter again only for the last centimetres
([What a policy looks at](95-what-a-policy-looks-at.md)).

**Tokens that are too coarse.** Patch tokens help only as far as position survives the patch
embedding. One edge of the cube is about 10 to 15 pixels long, about one 16-pixel patch (estimated
in [Images Become Tokens](20-images-to-tokens.md)), so its position has to be read from how its
pixels split between neighbouring patches; the policies above learned to. Whether the 32-pixel
patches of CLIP ViT-B/32 would be enough was not measured. Before choosing a patch size, compute
the object's width in pixels.

**The sequence outgrows memory or time.** A camera, a frame of history or a finer patch grid
multiplies the token count, and attention cost grows with its square. No length check stops such a
run: `max_seq_len: 2048` in `dp_vit.yaml` only sizes the rotary tables built at construction
([chapter 12](12-positions.md)). Compute the sequence length first; otherwise expect an
out-of-memory error or a fall in samples/s.

## Background and further reading

- Dosovitskiy et al., "An Image is Worth 16x16 Words", 2020.
  [arXiv:2010.11929](https://arxiv.org/abs/2010.11929). Introduced the ViT: patches as tokens and a
  class token whose final state represents the image.
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (CLIP),
  2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020). Trained image and text encoders with
  a contrastive loss on pooled embeddings.
- Shridhar et al., "CLIPort: What and Where Pathways for Robotic Manipulation", 2021.
  [arXiv:2109.12098](https://arxiv.org/abs/2109.12098). Paired CLIP's semantic features ("what")
  with a spatially precise pathway ("where") for language-conditioned manipulation.
- Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", 2023.
  [arXiv:2303.04137](https://arxiv.org/abs/2303.04137). Its ResNet-18 image encoder replaces global
  average pooling with spatial softmax pooling to maintain spatial information.

## Check yourself

1. Compute by hand the conditioning shape `dp_vit` would produce with three cameras and one frame
   each, and the sequence length in the action transformer with the 19 time, state and action-step
   tokens unchanged (`[B, 588, 256]` and 607). Then confirm the four-image case in code (expected
   output `torch.Size([1, 784, 256])`):

   ```bash
   .venv/bin/python -c "
   import torch
   from vla_foundry.models.vision_language_backbones.vit_backbone import ViTBackboneWrapper
   from vla_foundry.params.model_params import ViTBackboneParams
   p = ViTBackboneParams(img_size=224, patch_size=16, hidden_dim=256, inter_dim=1024, n_heads=4, n_layers=6)
   out = ViTBackboneWrapper(p).get_action_conditioning(torch.zeros(1, 1, dtype=torch.long), torch.rand(4, 3, 224, 224))
   print(out.embeddings.shape)"
   ```

2. In `CLIPHF.forward`, find the line that discards the patch tokens. Name the attribute of the
   vision model's output that still holds them.
3. A policy reaches a training loss of 0.04 and 4% success. Write the evaluation command that
   separates "cannot generalise" from the other causes, and name the number that separates
   "imprecise" from "not looking".
