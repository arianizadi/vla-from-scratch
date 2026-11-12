---
title: Pretrained Vision Encoders
part: 2 Vision
order: 22
summary: Choose between a from-scratch ViT, a pretrained encoder from the timm library and a VLM's own vision tower, knowing what CLIP, SigLIP and DINOv2 features carry and what each costs in tokens.
status: draft
---

# Pretrained Vision Encoders

A pretrained vision encoder, here always a Vision Transformer (ViT), has weights trained on a
large image collection before it sees any robot data. A policy that reuses one starts with
features that already separate objects and, if the encoder was trained with captions, tie them to
words, instead of learning that from a few thousand demonstrations. Which encoder, at what
resolution, frozen (weights fixed) or trained, decides how many tokens the policy pays for and
whether those tokens still say where things are.

## The idea

[Images to tokens](20-images-to-tokens.md) showed how a ViT cuts an image into square patches and
turns each patch into a token. The encoders in this chapter share that architecture and differ in
the training signal, which decides what the tokens are good for.

Background, from the CLIP, SigLIP and DINOv2 papers:

| Encoder | Training signal | What the loss is computed on |
|---|---|---|
| CLIP | contrastive, on image-text pairs from the web: each image must pick its caption among the captions in the batch, and each caption its image, with a softmax over the batch | one embedding per image, one per caption |
| SigLIP | the same pairing, but each image-text pair is scored on its own with a sigmoid, without normalising over the batch | one embedding per image, one per caption |
| DINOv2 | no text and no labels (self-supervised: the target is made from the images themselves). A student network sees cropped, partly masked views of an image and must match a teacher, a slowly updated average of the student's weights, on the whole image and on each masked patch | the image-level token and the masked patch tokens |

The last column matters. Worked example (background): CLIP ViT-B/32, the encoder that
`dp_clip.yaml` loads, turns a 224 px frame into 49 patch tokens plus one class token, an extra
learned token whose final state stands for the whole image. CLIP's loss reads only the projected
class token, 1 of 50 outputs. CLIP and SigLIP features are therefore aligned with language, but
their patch tokens are a by-product; no loss term asks a single patch to be right. A DINOv2 ViT
with 14 px patches turns the same frame into 256 patch tokens plus a class token, and its loss,
which uses no language, reads the class token and the patch tokens that were masked.

A grasp needs both. On `pick_target` the policy must find the object the instruction names
(semantics) and put the gripper on it (geometry). The lab notebook's grasp tolerance test gives the
scale of the second: with the scripted expert's grasp point shifted sideways by 10 mm, it succeeds
on the can 60% of the time and on the hex nut 61% on machine A (one RTX 4090, 24 GB); on machine B
(L40S GPUs, 128 CPU cores) 94% and 97%, dropping at 20 mm to 29% and 70%.

## How it works

### Semantics and geometry

Background. El Banani et al. (2024) trained depth and surface-normal decoders on frozen encoders:
DINOv2 gave accurate, detailed depth; CLIP and the other language-supervised models did poorly.
Tong et al. (2024) collected image pairs that CLIP embeds as nearly identical although they differ
visibly; multimodal language models, GPT-4V among them, often answered simple questions about them
wrongly, and adding DINOv2 features improved visual grounding (tying an answer to the right image
region).

On semantics the order reverses. In Prismatic VLMs (Karamcheti et al., 2024) CLIP and SigLIP were
better single encoders than DINOv2. Concatenating DINOv2 and SigLIP features for each patch gave
gains across the benchmarks, the largest on localisation (answering with where an object is); the
same fusion with CLIP gave no significant gain. OpenVLA (Kim et al., 2024) uses the fused DINOv2 and
SigLIP encoder for robot control.

### Resolution and the token budget

An image of side $s$ cut into patches of side $p$ gives $(s/p)^2$ tokens. Halving the patch size or
doubling the resolution multiplies the token count by four. The cameras here render 224 x 224
frames, which the configurations in `sim/pickup/configs/` turn into:

| Configuration | Patch | Tokens per image | Conditioning (what the backbone hands the action head) |
|---|---|---|---|
| `dp_vit.yaml`, from-scratch ViT | 16 px | 14 x 14 = 196, each 256 wide | every patch token: `[batch, 784, 256]` for the 4 images (2 cameras x 2 frames) |
| `dp_vit_lang_reactive.yaml`, the same ViT | 16 px | 196, each 256 wide | `[batch, 424, 256]`: 32 instruction tokens and the 392 patch tokens of 2 images (one frame per camera) |
| `custom_vlm.yaml`, from-scratch ViT | 14 px | 256, each 256 wide | not a policy; the tokens go to the language model |
| `dp_clip.yaml`, CLIP ViT-B/32 | 32 px | 49 plus a class token, inside the encoder | `[batch, 5, 512]`: 1 pooled vector per image and 1 for the instruction |
| `vla_qwen.yaml`, Qwen3-VL-2B | 16 px, then 2 x 2 merged | 196, then 49 | 1 vector, read at the action token (a special token appended to the prompt) |

One Qwen image token covers 32 x 32 pixels, four times the area of one `dp_vit` token.

Self-attention compares every token with every other, so its cost grows with the square of the
count: 784 x 784 = 614,656 pairs per layer for the image tokens of `dp_vit`, against
196 x 196 = 38,416 for the 4 x 49 that the language model of `vla_qwen` reads, 16 times fewer.

<!-- interactive: patchify -->
*Figure: a 224 x 224 camera frame cut into patches at a chosen patch size. Each patch becomes one
token; an object's size is shown in pixels and in patches.*

Background. PaliGemma (Beyer et al., 2024) publishes weights for 224, 448 and 896 px inputs, with
256, 1024 and 4096 image tokens. Prismatic found that 336 or 384 px improved VLM benchmarks.
OpenVLA found no difference between 224 and 384 px for robot control; 384 px took three times
longer to train.

### Frozen or fine-tuned

A frozen encoder receives no gradient, which saves gradient and optimiser memory and keeps the
pretrained features intact; a fine-tuned one can adapt to the task.

Background. LLaVA (Liu et al., 2023) kept its CLIP encoder frozen. Prismatic found that
fine-tuning the encoder during VLM training degraded performance, most on localisation. In
PaliGemma's ablation a frozen encoder made no difference after transfer but worsened validation
perplexity (how surprised the model is by held-out answers) on spatial tasks. OpenVLA found
fine-tuning the encoder "crucial" for policy performance. None of this was isolated here.

### How a VLM reuses an encoder

The VLMs cited here (LLaVA, Prismatic, PaliGemma, Qwen3-VL) discard the pooled vector and keep the
patch tokens, which a projector maps to the width of the language model's embeddings
([Joining vision and language](30-joining-vision-and-language.md)); Qwen3-VL (Bai et al., 2025)
first merges each 2 x 2 group into one. The projector and the language model were trained on this
tower's features, so a policy built on a pretrained VLM keeps its tower.

## In this repository

There are four ways to get image features. One goes through timm (PyTorch Image Models), a library
of image backbones with pretrained weights.

| Route | Config | Class |
|---|---|---|
| From-scratch ViT | `model.vit.type: vit`, or `vision_language_backbone.type: vit_backbone` | `ViT` in [`vla_foundry/models/vit.py`](../../vla_foundry/models/vit.py) |
| timm encoder | `model.vit.type: vit_hf` inside `model.type: vlm` | `ViTHF` in [`vla_foundry/models/vit_hf.py`](../../vla_foundry/models/vit_hf.py) |
| CLIP | `vision_language_backbone.type: clip_backbone` | `CLIPHF` in [`vla_foundry/models/diffusion_policy/clip_hf.py`](../../vla_foundry/models/diffusion_policy/clip_hf.py) |
| A VLM's own tower | `vision_language_backbone.type: vlm_backbone` | `VLMHF` in [`vla_foundry/models/vlm_hf.py`](../../vla_foundry/models/vlm_hf.py) |

**The from-scratch ViT.** This is the class from [Images to tokens](20-images-to-tokens.md).
`ViT.forward` applies a final `LayerNorm` and, with the default `cls_flag: false`, returns all
patch tokens, `[batch, num_patches, hidden_dim]`. It has no loader for published weights (no code
under `vla_foundry/` reads the `pretrained` field of `ViTParams`). It starts from `_init_weights`,
or from a checkpoint written by this repository, which `create_vlm` in `vla_foundry/models/vlm.py`
loads from `model.vit.resume_from_checkpoint`. `ViTBackboneWrapper` always builds this class, so the
`vit_backbone` of a policy cannot be a pretrained encoder. `text_tokens: 32`
(`dp_vit_lang_reactive.yaml`) adds the instruction as learned token embeddings, trained by the
action loss alone ([Language conditioning](51-language-conditioning.md)).

**The timm encoder.** `ViTHF.__init__` calls
`timm.create_model(model_params.hf_pretrained, num_classes=0, pretrained=load_pretrained)`.
`ViTHF.forward` returns `self.model.forward_intermediates(image)[0]`, which for timm's ViT models
is the final normalised token sequence, `[batch, tokens, width]`, including any class token or
register tokens (extra learned tokens that some DINOv2 models carry). `ViTHF` does not check the
architecture; only timm ViTs return this shape.

Three settings are not read from the timm model and must be set by hand: the encoder width
`model.vit.hidden_dim` (`ViTHFParams`, default 768), `model.vit.projector_pixel_shuffle_factor`
(default 1; how many tokens per side are merged before the projector, which above 1 requires a
square token count) and `data.img_num_tokens`, the tokens that reach the language model per image.

**CLIP.** `CLIPHF.forward` takes `vision_model(pixel_values).pooler_output` and applies
`visual_projection`; the patch tokens never leave the encoder. `CLIPBackboneWrapper` puts the
pooled text embedding in front of the 4 image vectors. Neither `dp_clip.yaml` nor its preset sets
`freeze_image_encoder` or `freeze_text_encoder`, so CLIP is trained with the action head.

**A VLM's tower.** `VLMHF` loads the whole model, vision tower included. Background: the Qwen3-VL
report states that the tower of the 2B model is a SigLIP-2 encoder (SigLIP2-Large, 300M), trained
further inside Qwen. In `vla_qwen.yaml`, `processor_kwargs.size` sets both pixel-count bounds to
50,176 (224 x 224), and `num_trainable_vlm_layers: 4` makes
`VLMHFBackboneWrapper._train_only_last_layers` freeze every VLM parameter except the last 4
language-model layers and the final norm
([Building on a pretrained VLM](61-pretrained-backbones.md)).

With `num_trainable_vlm_layers` unset (the default, `None`) the whole VLM, tower included, is
trained; with any number the tower is frozen. No key trains the tower while freezing the language
model. In a `vlm` trained here, `--model.vit.freeze true` freezes only the vision encoder.

To train the two `pick_cube` policies that the next section compares:

```bash
sim/pickup/scripts/make_dataset.sh pick_cube 200 pick_cube_200
sim/pickup/scripts/train.sh dp_clip pick_cube_200 dp_clip_pick_cube_200 300000
sim/pickup/scripts/train.sh dp_vit  pick_cube_200 dp_vit_pick_cube_200  300000
```

The recommended first real run uses the from-scratch ViT, on demonstrations with random start poses
and the object moved once in about a third of them. The config saves a running average of
the weights (EMA) next to the trained (raw) ones, and `eval_checkpoints.sh` evaluates both:

```bash
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k --random_start --disturb_prob 0.5
sim/pickup/scripts/train_resilient.sh dp_vit_lang_reactive pick_target_4k my_run 5000000
sim/pickup/scripts/eval_checkpoints.sh experiments/pickup/my_run pick_target
```

## What we measured

From the [lab notebook](../lab-notebook.md), one run per policy, closed-loop on unseen layouts;
"raw / EMA" are the trained and the averaged weights.

1. `pick_cube`, 200 demonstrations, 300k training samples (23 epochs), machine A: CLIP ViT-B/32
   with one pooled vector per image and one for the text (245M parameters) scored 2/50 = 4% (95% CI
   1-14%); the from-scratch ViT with every patch a token (`dp_vit`, 36M) scored 15/50 = 30%
   (19-44%).
2. `pick_target`, 1500 demonstrations, machine A: the `vla_qwen` policy (frozen tower, 306M of 2.2B
   parameters trained, one conditioning vector, history) scored 6/100 after 400k samples. A second
   run of the same recipe on machine B ended at 5% after 0.4M samples. It closed at the named
   object (the object nearest the gripper when it first closed) 71% of the time, a median 83 mm
   off (notebook, "Qwen3-VL-2B backbone").
3. The same Qwen backbone, tower still frozen, with `conditioning: all_tokens`, no history and 4000
   demonstrations (`vla_qwen_tokens_reactive.yaml`, machine B): 70% / 72% after 0.5M samples.
4. `pick_target`, 4000 demonstrations, 5M samples, machine B: `dp_vit_lang_reactive` (from-scratch
   ViT, instruction tokens, no history) scored 83% / 85%, a median 7 mm from the named object; it
   was at 0% / 1% after 1M samples ([How much data and training](71-how-much-data-and-training.md)).
   This run was trained before
   the position-table fix; the same recipe after the fix ends at 93% (trained weights), and a
   second seed at 96% / 96% (notebook, "After the position fix").
5. The from-scratch VLA with `conditioning: all_tokens`, no history, 4000 demonstrations, 3M
   samples, machine B: 86% / 87% with the stage-2 VLM pretrained on captions (its ViT included),
   75% / 85% with the same architecture from random weights. The
   pretrained one starts to look after 0.25M samples, the other after about 1.0M
   ([notes](../notes.md): "Pretraining a backbone: mainly an earlier start").

What these numbers show: on a one-object task, one pooled vector from a pretrained encoder lost to
the patch tokens of a from-scratch one ([Pooled vs patch tokens](21-pooled-vs-patch-tokens.md)).
With fixed cameras and four known objects, tokens trained from random weights by the action loss
alone found the named object. Pretraining on captions bought mainly an earlier start.

What they do not show:

- Whether CLIP's patch tokens would have worked. They never reached the action head, and the two
  policies also differ in size and in whether they receive text.
- Whether the frozen tower limited the first Qwen policy. The run of item 3 kept it frozen and
  reached 70% / 72%; conditioning, history and demonstrations changed together, so this does not
  isolate the tower, but the frozen tower did not prevent success. History alone keeps the
  from-scratch policy at 2% / 2% after 3M samples on 4000 demonstrations (trained before the
  position fix; after it, on demonstrations that begin with a hold frame, a history policy ends at
  0% / 4% after 5M samples; [History and shortcuts](52-history-and-shortcuts.md)).
- Whether a published encoder (CLIP, SigLIP, DINOv2) brings the start forward as the
  caption-pretrained backbone did: none was isolated. Nor were resolutions other than 224 px, or
  frozen against fine-tuned encoders.

## Choosing an encoder

| Situation | Choice | Basis |
|---|---|---|
| One simulator, fixed cameras, known objects | from-scratch `ViT`, all patch tokens (`dp_vit_lang_reactive.yaml`) | measured here on machine B, after the position fix: `pick_target` 93% (trained weights), 96% / 96% on a second seed, 98% / 98% with random start poses (4000 demonstrations, 5M samples); `pick_cube` 91% (1000 demonstrations, 1.5M samples) |
| The policy is built on a pretrained VLM | that VLM's own tower, every token to the action head (`conditioning: all_tokens`) | Qwen: 70% / 72% with every token, 5% with one vector |
| New objects, new scenes, real cameras | SigLIP, possibly fused with DINOv2, with patch tokens kept | background (Prismatic, OpenVLA); not measured here |

The third row is not a config change here. No route gives a published encoder's raw patch tokens
directly to an action head: `vit_backbone` cannot load pretrained weights, `clip_backbone` returns
only pooled vectors, and no code fuses two encoders. Both VLM backbones (`vlm_foundry_backbone`,
`vlm_backbone`) accept `conditioning: all_tokens`, which gives the language model's hidden states at
each image and text token, not the encoder's own. For a `vlm` on a timm encoder,
`VLMFoundryBackboneWrapper._prepare_inputs` resizes images only for a from-scratch ViT and passes a
`vit_hf` tower's images on unchanged; no run here used that.

## Where it goes wrong

| What you see | Why | Check and fix |
|---|---|---|
| `Unknown model` from timm | `hf_pretrained` holds a `transformers` repository id, such as the `google/siglip-so400m-patch14-384` in `docs/reference/models/vlm.md`; timm has its own names | use a timm name such as `vit_so400m_patch14_siglip_384`, or timm's `hf-hub:` prefix |
| a matrix shape error in `ModalityProjector` on the first batch | its linear layer takes `hidden_dim * projector_pixel_shuffle_factor**2` inputs, and `model.vit.hidden_dim` is still 768 | set `model.vit.hidden_dim` to the encoder's width |
| `AssertionError` in `VLM.forward` | the `simple_vlm` processor writes `data.img_num_tokens` placeholders per image, and the encoder returns another number; a class token or register tokens add to the patch count | print the encoder's output shape (exercise 2) and set `img_num_tokens` to its second dimension, divided by the shuffle factor squared |
| no error, but pretrained features are fed inputs they were not trained on | `SimpleVLMProcessor` normalises with mean 0.5 and std 0.5 per channel. timm's configurations list those values for SigLIP, ImageNet statistics for DINOv2 and a third set for OpenAI CLIP | compare with `timm.create_model(name).pretrained_cfg['mean']`. No config key sets them: `get_processor` (`vla_foundry/data/processor/__init__.py`) passes only `image_size`, so DINOv2 or OpenAI CLIP needs `image_mean` and `image_std` passed there in code |
| a size error on the first batch | the image is not the encoder's size; timm's DINOv2 models default to 518 px | set `data.image_size` to the encoder's size |

## Background and further reading

- Dosovitskiy et al., [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929), 2020.
  The ViT, a plain transformer over image patches; the architecture of every encoder here.
- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020),
  2021. CLIP: contrastive training on 400 million image-text pairs.
- Zhai et al., [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343),
  2023. SigLIP: a per-pair sigmoid loss in place of the batch softmax.
- Oquab et al., [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193),
  2023. Self-supervised image and patch objectives on 142 million curated images.
- Liu et al., [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485), 2023. LLaVA: a frozen
  CLIP encoder joined to a language model by a trained projection.
- Karamcheti et al., [Prismatic VLMs](https://arxiv.org/abs/2402.07865), 2024. Controlled
  comparison of encoders, fusion, resolution and freezing in VLMs.
- Tong et al., [Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs](https://arxiv.org/abs/2401.06209),
  2024. Image pairs that CLIP cannot tell apart.
- El Banani et al., [Probing the 3D Awareness of Visual Foundation Models](https://arxiv.org/abs/2404.08636),
  2024. Depth, surface normals and correspondence read from frozen encoders.
- Beyer et al., [PaliGemma: A versatile 3B VLM for transfer](https://arxiv.org/abs/2407.07726),
  2024. SigLIP encoder, linear projector, Gemma language model; a freezing ablation.
- Kim et al., [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246),
  2024. Fused DINOv2 and SigLIP encoder for robot control, fine-tuned; 224 against 384 px.
- Bai et al., [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631), 2025. A VLM whose
  SigLIP-2 patch features are compressed 2 x 2 before the language model; the backbone of
  `vla_qwen.yaml`.

## Check yourself

1. Compute the tokens per image for a 224 px frame at patch 14, 16 and 32, and after a 2 x 2 merge
   at patch 16.

2. Build a timm encoder with random weights (nothing is downloaded) and print what `ViTHF` returns.
   Then state the `model.vit.hidden_dim` and `data.img_num_tokens` a `vlm` config would need for it.

   ```bash
   .venv/bin/python -c "
   import torch
   from vla_foundry.models.vit_hf import ViTHF
   from vla_foundry.params.model_params import ViTHFParams
   m = ViTHF(ViTHFParams(hf_pretrained='vit_base_patch16_siglip_224'), load_pretrained=False)
   print(m(torch.zeros(1, 3, 224, 224)).shape)"
   ```

3. Run the freezing tests and read which parameters the second expects to stay trainable:

   ```bash
   .venv/bin/python -m pytest tests/essential/models/test_freeze.py tests/essential/models/test_vlm_backbone_partial_freeze.py -q
   ```
