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
vector per image lifted a single cube in 2 of 50 attempts against 15 of 50 with patch tokens, and
on a harder task the same patch tokens failed or succeeded depending on what else was right.

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

### Shapes in the policies trained here

`dp_clip` and `dp_vit` use two cameras and two frames per camera (`image_indices: [-1, 0]`, control
steps t-1 and t): four images at a time. Both feed a flow-matching action transformer
([Generating Actions with Flow Matching](41-flow-matching.md)).
`DiffusionPolicy._build_transformer_input` in
[`vla_foundry/models/diffusion_policy/diffusion_policy.py`](../../vla_foundry/models/diffusion_policy/diffusion_policy.py)
concatenates its input sequence in this order:

1. one **noise-level token** for the flow-matching time, that is, how noisy the actions being
   generated currently are (not the control step);
2. the **conditioning**, the tokens the backbone (the image and text encoder) hands over, which the
   action head reads but does not predict;
3. the state tokens, the robot's joint state at control steps t-1 and t;
4. the **action-step tokens**: the previous action, given without noise, and the **chunk** being
   generated, 15 actions from control step t on.

The previous frame, state and action are the policy's **history**.
[`dp_vit_lang_reactive.yaml`](../../sim/pickup/configs/dp_vit_lang_reactive.yaml), the patch-token
config that later solves the three-object task, has none (`image_indices: [0]`,
`lowdim_past_timesteps: 0`) and adds the instruction as 32 token slots (`text_tokens: 32`).

| | `dp_clip` (pooled) | `dp_vit` (patch tokens) | `dp_vit_lang_reactive` |
|---|---|---|---|
| Conditioning | `[B, 5, 512]` (1 text + 4 images) | `[B, 784, 256]` (4 x 196) | `[B, 424, 256]` (32 text slots + 2 x 196) |
| Noise-level + state + action-step tokens | 1 + 2 + 16 | 1 + 2 + 16 | 1 + 1 + 15 |
| Sequence length | 24 | 803 | 441 |
| Positions of the action-step tokens | 8-23 | 787-802 | 426-440 |
| Token pairs per attention layer | 576 | 644,809 | 194,481 |

The fourth row follows from the order above: in `dp_vit_lang_reactive` the noise-level token is at
position 0, the conditioning at 1-424, the joint state at 425, so 426 tokens precede the action-step
tokens: the ones the head denoises sit after every patch.

### The cost of tokens

Self-attention compares every token with every other
([The Transformer Block](11-the-transformer-block.md)), so its compute grows with the square of the
sequence length, and so does its memory whenever the attention matrix is materialised. Going from
24 to 803 tokens is 33 times more tokens and about 1,100 times more token pairs per attention
layer. The number of patch tokens is cameras x frames per camera x (image size / patch size)².
When actions are generated the cost is paid once per flow step: `DiffusionPolicy.generate_actions`
computes the backbone output once and then runs the action transformer for every flow step (10 by
default, `eval.py --flow_steps`).

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
after a layer norm; `CLIPHF.forward` never reads the patch tokens, which the same output holds in
`last_hidden_state` (for ViT-B/32 at 224 pixels, 49 patch tokens plus the class token).
`CLIPBackboneWrapper._concatenate_for_conditioning` in
[`vla_foundry/models/vision_language_backbones/clip_hf_backbone.py`](../../vla_foundry/models/vision_language_backbones/clip_hf_backbone.py)
returns the text vector followed by the image vectors, `[B, 1+N, D]`.

**The token path.** `ViTBackboneWrapper.get_action_conditioning` in
[`vla_foundry/models/vision_language_backbones/vit_backbone.py`](../../vla_foundry/models/vision_language_backbones/vit_backbone.py)
calls the `ViT` of [`vla_foundry/models/vit.py`](../../vla_foundry/models/vit.py), which returns
`[B, N, num_patches, hidden_dim]`, and flattens cameras and patches into one sequence. With
`text_tokens` above 0, `_embed_text` turns the instruction's token ids into that many slots, placed
in front of the patch tokens ([chapter 51](51-language-conditioning.md)). The ViT can
pool as well: with `cls_flag` set (`ViTParams` defaults it to `False`; no config under
`sim/pickup/configs/` sets it), `ViTPatchEmbeddings` prepends a class token and `ViT.forward` keeps
only its final state, `x[:, 0]`: one flag turns the token path into a pooled one, everything else
equal.

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
the same patch-token policy with the instruction as tokens and no history, trained on
demonstrations with random start poses and recoveries
([Which Demonstrations Teach What](72-which-demonstrations-teach.md)):

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
  tokens were not tested ([Pretrained Vision Encoders](22-pretrained-vision-encoders.md)), nor was
  `dp_vit` with `cls_flag` set, the test that would change pooling alone.
- That one vector cannot carry position. CLIP was trained with the policy, and after 300k samples
  the policy still missed. A longer run, a frozen encoder and a direct test of what the vector
  holds were not tried, and the slope defined below was not recorded.
- A working policy. With 1000 demonstrations and 1.5M samples the patch-token recipe reached
  37/50 = 74% (60-84%).

### Three objects: patch tokens are not sufficient

`pick_target` puts three of four objects on the table and the instruction names one. On machine B
every policy of the second round of runs (all trained before the position fix of
[chapter 12](12-positions.md)) received the patch tokens of the small ViT above, and most failed. Two
numbers read how: **closed at the named object**, how often the object nearest the gripper when it
first closed was the named one (chance 33%), and the **slope**, the least-squares slope of where
the gripper first closed against where the named object was: 1 if it closes where the object is,
0 if where it closes does not depend on the scene.

The failing runs had slopes near 0 and closed at the named object at about chance: they had every
patch token, and where they closed did not depend on the scene
([Measuring How a Policy Fails](92-measuring-failure.md) reads those runs). The image tokens were
the same in every run; what separated failure from success was the rest of the recipe:

- the instruction as tokens ([The Instruction Has to Reach the Action Head](51-language-conditioning.md));
- no previous action as an input, which a policy otherwise reads the target from instead of the
  patches ([History and the Shortcut It Opens](52-history-and-shortcuts.md));
- training past the late, abrupt start of looking: with both of the above, 4000 demonstrations and
  5M samples gave 83% / 85% (trained weights / **averaged weights**: their running average,
  saved by the run)
  ([How Much Data, How Much Training](71-how-much-data-and-training.md)).

That patch tokens were necessary is measured on `pick_cube` only. No pooled image encoder was
trained on `pick_target`; the closest test is the VLA in "Where it goes wrong".

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
`.venv/bin/python -m sim.pickup.summarize` the slope. A slope near 1 is an imprecise policy. A
slope near 0 is a policy that is not looking: check that the conditioning is more than a few
vectors (print its shape, as in the first exercise below), that the policy is not given its
previous action, and that training did not end before the slope moved.

**The same bottleneck under another name.** A vision-language-action model (VLA) can pass hundreds
of image tokens through its language model and still hand the action head one vector. Both VLA
recipes here do so by default: `_prepare_inputs_for_action_token` in `VLMHFBackboneWrapper` and in
`VLMFoundryBackboneWrapper` places one special token, the action token, directly after the last
real token of the prompt, and only its hidden states reach the action head, shape
`[B, 1, lm_hidden_dim * num_vlm_layers_to_use]`: the class-token method again.
`conditioning: all_tokens` hands over every real token's states instead. On
`pick_target` the Qwen VLA went from 5% with one vector to 70% / 72% with every token, but history
and the number of demonstrations changed with it
([A Backbone From Scratch Against a Pretrained One](102-case-pretrained-backbones.md);
[chapter 50](50-what-the-action-head-sees.md) has the wiring).

**The tokens arrive and are read once.** Having every patch token is not reading them at every
step: a policy can choose its target from the first frames and then follow its own arm.
`eval.py --stale_images` keeps sending the first frame's images and shows whether later frames
matter ([Probing a Policy That Works](95-what-a-policy-looks-at.md)).

**Tokens that are too coarse.** Patch tokens help only as far as position survives the patch
embedding. One edge of the cube is about 10 to 15 pixels long, about one 16-pixel patch (estimated
in [Images Become Tokens](20-images-to-tokens.md)), so its position has to be read from how its
pixels split between neighbouring patches; the policies above learned to. A 32-pixel patch, as in
CLIP ViT-B/32, spans two to three cube widths, and a 224-pixel frame becomes a 7x7 grid; whether
that is enough was not measured. Before choosing a patch size, compute the object's width in
pixels.

**The sequence outgrows memory or time.** A camera, a frame of history or a finer patch grid
multiplies the token count, and attention cost grows with its square. No length check stops such a
run: `max_seq_len: 2048` in `dp_vit.yaml` only sizes the rotary tables built at construction
([chapter 12](12-positions.md)). Compute the sequence length first; otherwise expect an
out-of-memory error or a fall in samples/s.

**A longer sequence is a different test of the position code.** Adding tokens moves the
action-step tokens to higher positions. The rotary table of every run under "What we measured" was
computed in bfloat16 and was inexact above position 256 ([chapter 12](12-positions.md) has the
mechanism and the fix): the pooled policy's 24 tokens never reach that range, and
every patch-token sequence here crosses it. A comparison of pooled against patch tokens therefore
also compares two ranges of the position table. After changing the token count, compute where the
action-step tokens now sit.

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
   each, and the sequence length in the action transformer with the 19 noise-level, state and
   action-step tokens unchanged (`[B, 588, 256]` and 607). Then confirm the four-image case in code
   (expected output `torch.Size([1, 784, 256])`):

   ```bash
   .venv/bin/python -c "
   import torch
   from vla_foundry.models.vision_language_backbones.vit_backbone import ViTBackboneWrapper
   from vla_foundry.params.model_params import ViTBackboneParams
   p = ViTBackboneParams(img_size=224, patch_size=16, hidden_dim=256, inter_dim=1024, n_heads=4, n_layers=6)
   out = ViTBackboneWrapper(p).get_action_conditioning(torch.zeros(1, 1, dtype=torch.long), torch.rand(4, 3, 224, 224))
   print(out.embeddings.shape)"
   ```

2. In `CLIPHF.forward`, find the line that discards the patch tokens. Then see both outputs of a
   CLIP vision model of the ViT-B/32 shape, with random weights so nothing is downloaded (expected
   `torch.Size([1, 50, 768]) torch.Size([1, 768])`). Which of the 50 tokens is the class token, and
   how many patches does the grid have per side?

   ```bash
   .venv/bin/python -c "
   import torch
   from transformers import CLIPVisionConfig, CLIPVisionModel
   m = CLIPVisionModel(CLIPVisionConfig(image_size=224, patch_size=32)).eval()
   out = m(pixel_values=torch.rand(1, 3, 224, 224))
   print(out.last_hidden_state.shape, out.pooler_output.shape)"
   ```

3. Using the order in which `_build_transformer_input` concatenates its parts, work out the
   positions of the action-step tokens for `dp_vit_lang` (the same as `dp_vit` plus 32 instruction
   slots) and say whether any of them lies below position 256. (Answer: 819-834; none.)
4. A policy reaches a training loss of 0.04 and 4% success. Write the evaluation command that
   separates "cannot generalise" from the other causes, and name the number that separates
   "imprecise" from "not looking".
