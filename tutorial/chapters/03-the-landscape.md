---
title: The Model Families Today
part: 0 Orientation
order: 03
summary: Place any robot policy, published or in this repository, on a map of five design choices and say what follows from each choice.
status: draft
---

# The Model Families Today

A robot manipulation policy takes camera images, an instruction and the robot's own state, and
returns motor commands. The published systems that do this differ in five design choices, and these
choices, a map on which any policy can be placed, also separate the ten policy configs in this
repository. Which of them works here depended on two things off that map: history and training
budget.

## The idea

Every policy in this chapter has two parts. The **backbone** turns images and text into vectors. The
**action head** turns those vectors into actions. Most current systems predict an **action chunk**,
a short sequence of future actions produced in one call
([42-chunks-and-execution.md](42-chunks-and-execution.md)).

Five questions describe a system:

1. **How are actions produced?** As discrete tokens (integers from a fixed vocabulary,
   [10-tokens.md](10-tokens.md)) generated one after another, by diffusion or flow matching, by
   direct regression, or as compressed tokens.
2. **What is the backbone?** A small network trained from scratch on robot data, a pretrained
   vision encoder, or a pretrained vision-language model (VLM), a language model that also accepts
   image tokens.
3. **How does the head read the backbone?** One vector, every token, cross-attention, attention to
   the backbone's tokens in every layer, or no separate head.
4. **How much data, from how many robots?**
5. **What can be downloaded?**

In this repository's tasks a policy receives 224x224 images from a front and a wrist camera, an
instruction such as "pick up the red cube" and the 8-D state (7 joint angles and the gripper
opening). **With history** it receives the images and the state of control steps t-1 and t (4
images) and the previous action; **without history**, those of control step t only. The output is
a chunk of 15 actions, shape `[15, 8]` (7 joint position targets and a gripper command each), of
which 8 are executed before the policy is called again.

## How it works

Statements about published systems in this section are background from the cited papers and
code, not measured here.

### Choice 1: how actions are produced

Demonstrations often contain more than one valid action for the same observation, such as passing
an object on either side. A head that regresses one chunk under a squared-error loss is pulled
toward their average, which may be neither. The four families below represent a distribution over
chunks instead.

**Discrete tokens, generated autoregressively.** "Autoregressive" means each token is predicted
from all the tokens before it, as a language model produces text. RT-2 (Brohan, 2023) splits each
action dimension into 256 uniform bins and writes an action as a short string of integers that the
VLM emits like words; OpenVLA (Kim, 2024) uses the same bins. The cost is one forward pass per
token: the OpenVLA paper reports about 6 Hz on one RTX 4090.

**Diffusion or flow matching.** The head starts from noise and refines it into a chunk over several
steps (flow steps, in flow matching), conditioned on the observation
([41-flow-matching.md](41-flow-matching.md)). Diffusion Policy (Chi, 2023) applies this to
visuomotor control; Octo (Octo Model Team, 2024) puts a diffusion head on a transformer; pi0
(Black, 2024) uses flow matching, a chunk of 50 actions and 10 flow steps. The cost is one pass of
the head per flow step, while the observation is encoded once.

**Regression on chunks with a CVAE.** ACT (Zhao, 2023) predicts a chunk of 100 actions with an
L1 loss (mean absolute error). It is trained as a conditional variational autoencoder
(CVAE): an extra encoder compresses the demonstrated chunk into a latent variable, so the decoder
can account for differences between demonstrations of the same situation. At test time the latent
is set to zero, so the output is deterministic.

**Compressed tokens.** FAST (Pertsch, 2025) keeps autoregressive generation but shortens the token
sequence: each action dimension of a chunk goes through a discrete cosine transform, under which a
smooth trajectory has few large coefficients, and the rounded coefficients are compressed with
byte-pair encoding ([10-tokens.md](10-tokens.md)). The paper reports that pi0 trained with these
tokens (pi0-FAST) matches flow-matching pi0 with up to 5x less training time and is slower at
inference: about 750 ms for a one-second chunk, against typically within 100 ms on an RTX 4090.

### Choice 2: what the backbone is

**Trained from scratch on robot data**: Diffusion Policy, and Octo, which feeds its transformer
image patches (square blocks of pixels, each turned into one token,
[20-images-to-tokens.md](20-images-to-tokens.md)) and has only its text encoder pretrained.

**A pretrained vision or vision-text encoder**, with the rest trained on robot data: `dp_clip`
below, and ACT as released: its paper trains from scratch per task, but its code loads
ImageNet-pretrained weights into the ResNet18 image encoders
(`detr/models/backbone.py`, `pretrained=is_main_process()`).

**A pretrained VLM**, fine-tuned: RT-2, OpenVLA and pi0.

### Choice 3: how the head reads the backbone

| Reading | What reaches the head |
|---|---|
| One vector | a single summary vector; in Octo, the embedding of a learned readout token, an extra input token that attends to the observation and instruction tokens |
| All tokens | every image and text token, in the head's own input sequence |
| Cross-attention | the head queries the backbone's tokens through separate attention layers |
| Shared attention | the head's tokens (state, noisy actions) share a sequence with the backbone's tokens and attend to them in every layer; the backbone's tokens do not attend back |
| No separate head | actions are further output tokens of the VLM |

The distinction matters for a grasp because an object's position is spread over image patches. A
pooled vector merges a whole image into one vector
([21-pooled-vs-patch-tokens.md](21-pooled-vs-patch-tokens.md)), and a head that receives only that
vector depends on it having kept the position.

<!-- interactive: pooled-vs-tokens -->
*Figure: camera frames reach the action head as one pooled vector per image or as 196 patch tokens
per image, among which attention can pick out the object.*

### Choices 4 and 5: data and availability

Years are first arXiv submissions; an empty cell was not confirmed for this chapter. There is no
success column: each system was evaluated on its own robots, tasks and data.

| System | Actions | Backbone | Head reads | Training data | Released |
|---|---|---|---|---|---|
| ACT (2023) | L1 regression on a chunk of 100, CVAE training | ResNet18 and a transformer encoder-decoder, about 80M parameters; no language input | cross-attention | 50 demonstrations per task (100 for one task) | code (MIT) |
| Diffusion Policy (2023) | denoising diffusion over an action sequence | ResNet-18 without pretraining, trained end to end | FiLM, in which observation features scale and shift the head's activations (CNN variant), or cross-attention (transformer variant) | | code (MIT), configs and weights per experiment |
| RT-2 (2023) | 256 bins per dimension, autoregressive tokens | PaLI-X (5B, 55B) or PaLM-E (12B), co-fine-tuned (trained on robot data mixed with the original web data) | no separate head | web vision-language data plus robot data from 13 robots over 17 months | none announced in the paper |
| Octo (2024) | diffusion head, action chunk | transformer trained on robot data, 27M or 93M parameters; pretrained `t5-base` text encoder | one readout token | 800k trajectories from 25 [Open X-Embodiment](https://arxiv.org/abs/2310.08864) (2023) datasets | code and weights (MIT) |
| OpenVLA (2024) | 256 bins per dimension, autoregressive tokens | Prismatic VLM, 7B: SigLIP + DINOv2 + Llama 2 | no separate head | 970k Open X-Embodiment trajectories | code (MIT) and weights (derived from Llama 2, so subject to its licence) |
| pi0 (2024) | flow matching, chunk of 50 | PaliGemma (3B) plus a 300M action expert trained from scratch, 3.3B in total | shared attention | over 10,000 hours: the authors' data from 7 robot configurations and 68 tasks, plus Open X-Embodiment | code and base weights in `openpi` (Apache-2.0) |
| pi0-FAST (2025) | FAST tokens, autoregressive | the pi0 VLM, without the action expert | no separate head | 10k hours | FAST+ tokenizer; base weights in `openpi` |

What follows: ACT and Diffusion Policy train one policy per task or experiment; Octo, OpenVLA and
pi0 pool data from many robots and release weights to fine-tune.

## In this repository

All ten policy configs in [`sim/pickup/configs/`](../../sim/pickup/configs/) set
`model.type: diffusion_policy` and `use_flow_matching_scheduler: true`, directly or through a
preset, so on choice 1 they are the same: a flow-matching head that predicts 15 actions, the
current control step plus `lowdim_future_timesteps: 14`. `sim/pickup/eval.py` executes 8 of them by
default (`--exec_horizon`) and takes 10 flow steps (`--flow_steps`).

They differ on choices 2 and 3 through one key, `model.vision_language_backbone.type`, from which
`get_vision_language_backbone` in
[`vla_foundry/models/vision_language_backbones/__init__.py`](../../vla_foundry/models/vision_language_backbones/__init__.py)
picks one of four wrapper classes. Each wrapper's `get_action_conditioning` returns a
`VisionLanguageBackboneOutput` whose `embeddings` field, the **conditioning**, has shape
`[batch, tokens, width]`. For the six configs with history (shapes derived in
[50-what-the-action-head-sees.md](50-what-the-action-head-sees.md) and
[60-anatomy-of-a-vla.md](60-anatomy-of-a-vla.md)):

| Config | Backbone | Conditioning the head receives |
|---|---|---|
| [`dp_clip`](../../sim/pickup/configs/dp_clip.yaml) | `clip_backbone`, pretrained `openai/clip-vit-base-patch32` ([22-pretrained-vision-encoders.md](22-pretrained-vision-encoders.md)) | `[batch, 5, 512]`: one text vector and one pooled vector per image |
| [`dp_vit`](../../sim/pickup/configs/dp_vit.yaml) | `vit_backbone` from scratch (patch 16, 256 wide, 6 layers); `processor: "none"` drops the text | `[batch, 784, 256]`: 4 images x 196 patches |
| [`dp_vit_lang`](../../sim/pickup/configs/dp_vit_lang.yaml) | the same ViT, with `text_tokens: 32` ([51-language-conditioning.md](51-language-conditioning.md)) | `[batch, 816, 256]`: 32 text slots + 784 patches |
| [`vla_qwen`](../../sim/pickup/configs/vla_qwen.yaml) | `vlm_backbone`, pretrained `Qwen/Qwen3-VL-2B-Thinking` | `[batch, 1, 4 x lm_hidden_dim]` |
| [`vla_custom`](../../sim/pickup/configs/vla_custom.yaml) | `vlm_foundry_backbone`, the VLM trained here | `[batch, 1, 2048]` |
| [`vla_custom_tokens`](../../sim/pickup/configs/vla_custom_tokens.yaml) | the same, with `conditioning: all_tokens` | `[batch, 1040, 2048]`: every real token, padded in front to `conditioning_len` |

Four more configs have no history (`image_indices: [0]`, `lowdim_past_timesteps: 0`) and half the
images: `dp_vit_reactive` `[batch, 392, 256]`, `dp_vit_lang_reactive` `[batch, 424, 256]`,
`vla_custom_tokens_reactive` `[batch, 528, 2048]` and `vla_qwen_tokens_reactive`, Qwen with
`conditioning: all_tokens`, `[batch, 128, 4 x lm_hidden_dim]`.

`vla_qwen` and `vla_custom` read the VLM through **the action token**: both wrappers insert a single
special token directly after the prompt's last real token, not after the batch padding
(`insert_after_last_real_token` in `vla_foundry/models/vision_language_backbones/base.py`;
[93-train-inference-skew.md](93-train-inference-skew.md)), and concatenate its hidden states from
the last `num_vlm_layers_to_use: 4` layers. The VLM sees many image tokens (`img_num_tokens: 49` per
image for Qwen, 256 for the custom VLM); the head sees one vector.

The head is always the same class, `DiffusionPolicy` in
`vla_foundry/models/diffusion_policy/diffusion_policy.py`. `_build_transformer_input` puts a time
token (the noise level) in front of the conditioning and appends the state tokens and the
**action-step tokens**, one per action of the chunk, preceded with history by the previous action.
All ten configs set `is_causal: false` for the head, so every action-step token attends to every
conditioning token: the "all tokens" reading when the backbone returns many tokens, "one vector"
when it returns one. There is no cross-attention or shared attention: the backbone runs to
completion first.

On the map, the `dp_vit` configs are the from-scratch family with a generative head, `dp_clip` the
pretrained-encoder family, and `vla_qwen` a pretrained VLM with a continuous head, as pi0 is, but
read through one vector, as Octo's head is. Nothing in `dp_vit_lang_reactive`, which solves
`pick_target` (three objects, one named) here, is pretrained.

The recommended first run is `dp_vit_lang_reactive` on 4000 demonstrations with random start poses
and recoveries, for 5M **samples** (a sample is one training window drawn by the data loader: one
frame's observations plus the actions that follow); its config saves averaged (EMA) weights itself:

```bash
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k --random_start --disturb_prob 0.5
sim/pickup/scripts/train_resilient.sh dp_vit_lang_reactive pick_target_4k my_run 5000000
sim/pickup/scripts/eval_checkpoints.sh experiments/pickup/my_run pick_target
.venv/bin/python -m sim.pickup.summarize --task pick_target --runs my_run
```

The three `vla_custom*` configs also need
`--model.vision_language_backbone.resume_from_checkpoint <checkpoint_N.pt>`, a checkpoint of the
VLM ([62-the-from-scratch-path.md](62-the-from-scratch-path.md)).

## What we measured

All numbers are from the [lab notebook](../lab-notebook.md): closed-loop success (the policy's own
actions determine the next observation), one training seed per run, on unseen **layouts** (the
objects on the table, their places, the one the instruction names and the arm's start pose, all
fixed by the seed).

`pick_cube` on machine A (one RTX 4090, 24 GB), 50 episodes. With 200 demonstrations and 300k
samples, the 245M `dp_clip` policy succeeded in 2/50 = 4% (95% CI 1-14%) and the 36M `dp_vit` in
15/50 = 30% (19-44%) ([21-pooled-vs-patch-tokens.md](21-pooled-vs-patch-tokens.md)). The two also
differ in pretraining, size and language input, so this does not isolate pooling.

`pick_target`, 100 layouts per evaluation. Two things off the map, budget and history, decided as
much as the map's choices did; each result below is told in full in its own chapter:

- **Training budget.** After 1500 demonstrations and 1.0M samples, all four `dp_vit` variants, with
  and without language and history, were between 0% and 5%. Given 4000 demonstrations and 5M
  samples, `dp_vit_lang_reactive` (language, no history) ended at 83% / 85% (trained / averaged
  weights): a policy starts to use the scene late, then quickly
  ([71-how-much-data-and-training.md](71-how-much-data-and-training.md)). That run used the rotary
  position table computed in bfloat16, since fixed; the fixed table helps
  ([12-positions.md](12-positions.md)).
- **History.** The same config with history, on 4000 demonstrations, stayed at 2% / 2% after 3.0M
  samples: it read the target from its own previous action (trained before the position fix; what
  holds after it is in [52-history-and-shortcuts.md](52-history-and-shortcuts.md)).
- **The reading, choice 3.** Read through the action token, one vector, the VLAs stayed near zero:
  `vla_custom` 0/100 after 1.0M samples (machine A) and `vla_qwen` 5% after 0.40M samples (machine
  B, L40S GPUs, 128 CPU cores), both with history and 1500 demonstrations. Read through every token,
  with no history and 4000 demonstrations, the from-scratch VLA (`vla_custom_tokens_reactive`, 3M
  samples) ends at 86% / 87% and Qwen (`vla_qwen_tokens_reactive`, 0.5M samples) at 70% / 72%.
  History and the number of demonstrations changed with the reading, so this does not isolate it;
  what pretraining the backbone adds is in
  [102-case-pretrained-backbones.md](102-case-pretrained-backbones.md).

None of this ranks the families of the published map: one task, one seed per run, and no
autoregressive head was measured here. Success on unmoved layouts also does not say what a policy
does when the object moves during the reach; that is measured in
[95-what-a-policy-looks-at.md](95-what-a-policy-looks-at.md), and it depends on the demonstrations
([103-case-self-correction.md](103-case-self-correction.md)).

## Where it goes wrong

**A budget that ends before the policy starts to look.** What you see: every variant between 0%
and 5%, so every family looks equally bad. Why: a policy starts to use the scene late, then
quickly; the run that ended at 83% / 85% was at 0% / 1% after 1.0M samples. How to check: evaluate
every checkpoint (`sim/pickup/scripts/eval_checkpoints.sh`, then `sim.pickup.summarize`) and read
the **slope**, the least-squares slope of where the gripper first closed against where the named
object was: 1 if it closes where the object is, 0 if where it closes does not depend on the scene.
It rises before success does. What to do: compare two designs only at a budget at which a
comparable run has begun to look.

**A large backbone hiding a one-vector bottleneck.** What you see: a complete, plausible grasp in
the wrong place. Measured here on `vla_custom`: the gripper closed in every episode, at the named
object no more often than chance
([92-measuring-failure.md](92-measuring-failure.md); the run also had history and a short budget).
Why: the config name does not say the reading. Both VLM wrappers read
`model.vision_language_backbone.conditioning`, whose default in
[`vla_foundry/params/model_params.py`](../../vla_foundry/params/model_params.py) is
`action_token`; only the `*_tokens*` configs set `all_tokens`. How to check: grep the configs for
`conditioning:` and print `get_action_conditioning(...).embeddings.shape[1]`. What to do: pass
tokens (`conditioning: all_tokens`), as the from-scratch VLA above does
([50-what-the-action-head-sees.md](50-what-the-action-head-sees.md)).

**A prompt longer than `conditioning_len`.** What you see: `ValueError: A prompt has N tokens but
conditioning_len is M.` Why: with `all_tokens`, `conditioning_from_hidden_states` in
`vla_foundry/models/vision_language_backbones/base.py` pads the real tokens in front to a fixed
`conditioning_len` (1040 in `vla_custom_tokens`, 528 in `vla_custom_tokens_reactive`, 128 in
`vla_qwen_tokens_reactive`), so the conditioning's length does not depend on what else is in the
batch. A longer instruction or more images push a prompt past it. What to do: raise
`conditioning_len` before training, and evaluate a checkpoint with the value it was trained with,
since it sets where the state and action-step tokens sit in the head's sequence.

**Memory for a pretrained VLM.** Training all 2.2B parameters of `vla_qwen` needs about 36 GB by
the estimate in its config comment, so for a 24 GB GPU it sets `num_trainable_vlm_layers: 4` and
trains 306M ([61-pretrained-backbones.md](61-pretrained-backbones.md)).

## Background and further reading

- Zhao, *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, 2023.
  [arXiv:2304.13705](https://arxiv.org/abs/2304.13705). ACT: chunked regression trained as a CVAE.
  [Code](https://github.com/tonyzhaozh/act).
- Chi, *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, 2023.
  [arXiv:2303.04137](https://arxiv.org/abs/2303.04137). Actions generated by conditional denoising.
  [Code](https://github.com/real-stanford/diffusion_policy).
- Brohan, *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*, 2023.
  [arXiv:2307.15818](https://arxiv.org/abs/2307.15818). Actions as text tokens of a co-fine-tuned
  VLM.
- Octo Model Team, *Octo: An Open-Source Generalist Robot Policy*, 2024.
  [arXiv:2405.12213](https://arxiv.org/abs/2405.12213). A transformer with a diffusion head.
  [Code](https://github.com/octo-models/octo).
- Kim, *OpenVLA: An Open-Source Vision-Language-Action Model*, 2024.
  [arXiv:2406.09246](https://arxiv.org/abs/2406.09246). A 7B token-output VLA.
  [Code](https://github.com/openvla/openvla).
- Black, *π0: A Vision-Language-Action Flow Model for General Robot Control*, 2024.
  [arXiv:2410.24164](https://arxiv.org/abs/2410.24164). Flow matching on a pretrained VLM.
  [Code](https://github.com/Physical-Intelligence/openpi).
- Pertsch, *FAST: Efficient Action Tokenization for Vision-Language-Action Models*, 2025.
  [arXiv:2501.09747](https://arxiv.org/abs/2501.09747). A compressed action tokenizer.

How these systems followed from one another is in [02-how-we-got-here.md](02-how-we-got-here.md).

## Check yourself

1. List every config's backbone (`dp_clip` takes it from the preset) and name its reading, one
   vector or all tokens:

   ```bash
   grep -A 5 "vision_language_backbone:" sim/pickup/configs/*.yaml \
     vla_foundry/config_presets/models/diffusion_policy.yaml
   ```

2. Find where the two VLM wrappers place the action token and read its hidden state, and say why
   neither depends on padding:

   ```bash
   grep -n "insert_after_last_real_token\|conditioning_from_hidden_states" \
     vla_foundry/models/vision_language_backbones/*.py
   ```

3. List every difference between the 4% and the 30% `pick_cube` policy above and describe a run that
   would isolate pooling.
