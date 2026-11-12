---
title: What the Action Head Sees
part: 5 Conditioning the action head
order: 50
summary: Read off, for any config in this repository, the exact token sequence the action head receives, and tell whether the object's position can reach it.
status: draft
---

# What the Action Head Sees

The action head is the transformer that turns noise into a chunk of robot actions. It never sees
a camera frame or an instruction, only one sequence of tokens, part of which the backbone (the
model that encodes the images and the text) hands over: one vector in some configs here, more than
a thousand tokens in others. That decides whether the object's position can reach the part of the
model that moves the arm; whether a trained policy uses it is measured below.

## The idea

The **conditioning** is the tensor the backbone hands to the action head, `[batch, tokens, width]`
(here `[B, N, D]`): `N` tokens of width `D` for each of the `B` rows of a batch (training windows,
or one served observation). The noise level, the joint state and the noisy
actions are separate inputs. All four arrive in one sequence:

```text
[ time ][ conditioning ][ state ][ action-step tokens ]
   1          N             P              T
```

The **time token** says how noisy the actions are ([Flow matching](41-flow-matching.md)). The
**state tokens** hold the joint angles and the gripper opening, one token per control step (one
command to the arm; 20 per second). The **action-step tokens** are one per action of the
**chunk**, the 15 future actions the policy predicts in one go. A policy with **history** is also
given the previous camera frames, the previous state and its previous action: a second state
token, and a sixteenth action-step token in front of the chunk.

Two configs for `pick_target` (three objects on the table, an instruction that names one):

- [`dp_vit_lang_reactive.yaml`](../../sim/pickup/configs/dp_vit_lang_reactive.yaml), the policy
  that solves the task: `N` = 424 (32 text slots and 2 x 196 image patches, 16-pixel squares of a
  camera frame), so 1 + 424 + 1 + 15 = 441 tokens.
- [`vla_custom.yaml`](../../sim/pickup/configs/vla_custom.yaml), a VLA (a policy whose backbone is
  a vision-language model), with history: `N` = 1 (one vector summarising four images and the
  instruction), so 1 + 1 + 2 + 16 = 20 tokens.

In the first, the token of the image patch that contains the cube is in the sequence and an
action-step token can attend to it. In the second, the backbone has to write the cube's position
into one vector first.

<!-- interactive: sequence-layout -->
*Figure: the policy sequence as a row of cells: time token, conditioning, state tokens, action-step
tokens. Switching the config redraws the conditioning block as 1, 5, 424, 816 or 1040 cells.*

## How it works

### Building the sequence

`DiffusionPolicy._build_transformer_input` in
[`vla_foundry/models/diffusion_policy/diffusion_policy.py`](../../vla_foundry/models/diffusion_policy/diffusion_policy.py)
assembles the sequence. With the numbers of `dp_vit_lang_reactive.yaml` (backbone and action
transformer both 256 wide):

1. The backbone returns `backbone_embeddings`, `[B, 424, 256]`.
2. `self.time_encoding`, an embedding table with one row per noise level (1000), gives
   `time_embeddings`, `[B, 1, 256]`.
3. With the default `model.diffusion_step_conditioning: concat` the two are concatenated, time
   first, and `self.condition_encode`, one linear layer, maps each token to the transformer width:
   `[B, 425, 256]`.
4. The state `[B, 1, 8]` (`lowdim_past_timesteps + 1` control steps, here 1) goes through
   `self.proprioception_encode` and becomes one token.
5. The noisy chunk `[B, 15, 8]` goes through `self.action_encode` and becomes 15 action-step
   tokens.
6. `torch.cat(parts, dim=1)` joins them: `[B, 441, 256]`.

`DiffusionPolicy.forward` runs the transformer on this sequence and maps the last `T` positions of
its last layer's output with `self.output_layer` to `[B, 15, 8]`, a predicted direction per
action-step token. Every policy config under `sim/pickup/configs/` sets `is_causal: false` on the
action transformer, so each action-step token attends to every other
token ([The transformer block](11-the-transformer-block.md)). That attention is its only route to
the scene; rotary position encoding tells it where in the sequence a token sits
([Positions](12-positions.md)).

With history (`image_indices: [-1, 0]`, `lowdim_past_timesteps: 1`, as in `dp_vit_lang.yaml`) there
are four images, the state is `[B, 2, 8]` and the actions are `[B, 16, 8]`: the first row is the
previous action, passed in without noise (`past_mask` in `generate_actions`). The sequence is
`[B, 835, 256]`.

### What each backbone hands over

All four backbone wrappers return a `VisionLanguageBackboneOutput` whose `embeddings` tensor is the
conditioning. The four `*_reactive` configs have no history: two images (one per camera) and
1 + `N` + 1 + 15 tokens. The others have four images and 1 + `N` + 2 + 16 tokens.

| `vision_language_backbone.type` | Config (`.yaml`) | What `embeddings` holds | Shape | Sequence length |
|---|---|---|---|---|
| `clip_backbone` | `dp_clip` | one pooled vector for the text, one per image | `[B, 5, 512]` | 24 |
| `vit_backbone` | `dp_vit`, `dp_vit_reactive` | every patch of every image (196 each) | `[B, 784, 256]`, `[B, 392, 256]` | 803, 409 |
| `vit_backbone`, `text_tokens: 32` | `dp_vit_lang`, `dp_vit_lang_reactive` | 32 text slots, then every patch | `[B, 816, 256]`, `[B, 424, 256]` | 835, 441 |
| `vlm_foundry_backbone` | `vla_custom` | hidden states of the action token | `[B, 1, 2048]` | 20 |
| `vlm_foundry_backbone`, `conditioning: all_tokens` | `vla_custom_tokens`, `vla_custom_tokens_reactive` | hidden states of every prompt token | `[B, 1040, 2048]`, `[B, 528, 2048]` | 1059, 545 |
| `vlm_backbone` | `vla_qwen` | hidden states of the action token | `[B, 1, 4 x LM width]` | 20 |
| `vlm_backbone`, `conditioning: all_tokens` | `vla_qwen_tokens_reactive` | hidden states of every prompt token | `[B, 128, 4 x LM width]` | 145 |

(A hidden state is the vector a layer holds for a token.)

### The action token

A vision-language model (VLM) reads a prompt of image tokens followed by the instruction
([Joining vision and language](30-joining-vision-and-language.md)): for `vla_custom.yaml`, four
images at 256 tokens each and the instruction, each token with a 512-wide hidden state per layer.
The default, `conditioning: action_token`, reads the result out at one place:

1. One new token, the **action token**, is added to the VLM's vocabulary and inserted directly
   after the last real token of each prompt.
2. The VLM runs. The from-scratch language model is causal (`is_causal: true` in
   `vla_foundry/config_presets/models/transformer_100m.yaml`), so the action token, being last,
   can attend to every image and instruction token.
3. The hidden states at the action token's position are taken from the last
   `num_vlm_layers_to_use` layers (4) and concatenated: 4 x 512 = 2048 numbers. That vector,
   `[B, 1, 2048]`, is the whole conditioning.

With `conditioning: all_tokens` no token is added: the last four layers' hidden states at every
real prompt position go to the action head, right-aligned in `conditioning_len` slots, so that the
sequence is as long in a training batch as when one prompt is served.

The image states among them never depend on the instruction: `apply_chat_template`
(`vla_foundry/data/processor/__init__.py`) puts the images first and the language model is causal,
so "which object" and "where" meet only at the few text positions or in the action head. The
price: attention over 1059 x 1059 token pairs per layer instead of 20 x 20, at each of the 10 flow
steps (passes of the action head) per chunk.

### Why one vector can carry a position, and why that is hard to learn

Capacity is not the obstacle: three objects' positions are six numbers and the vector has 2048.
The difficulty is where the work happens and what trains it (an argument, not a measurement).

With patch tokens, position arrives laid out: each token starts as one 16 x 16 pixel cell,
`ViTPatchEmbeddings` (`vla_foundry/models/vit.py`) adds a learned embedding that says which cell,
and the cube changes a few tokens at known places.

With one vector, position must be in the *values*: the backbone has to turn "the cube is in these
patches" into coordinates, pick the named object, and encode the result precisely enough for a
grasp, taught only by the action loss passed back through that one token. On machine A (one RTX
4090, 24 GB), where the first-round single-vector VLAs ran, a 10 mm sideways offset takes the
scripted expert (the hand-written controller that recorded the demonstrations) from 94% to 60% on
the can; on machine B, from 100% to 94% ([Measuring failure](92-measuring-failure.md)).

## In this repository

**`clip_backbone`, `vit_backbone`.** `CLIPBackboneWrapper._concatenate_for_conditioning`
([`clip_hf_backbone.py`](../../vla_foundry/models/vision_language_backbones/clip_hf_backbone.py))
stacks the text vector and the image vectors. `ViTBackboneWrapper.get_action_conditioning`
([`vit_backbone.py`](../../vla_foundry/models/vision_language_backbones/vit_backbone.py)) flattens
cameras and patches into one sequence; with `text_tokens > 0`, `_embed_text` puts the prompt's
token ids in front of them as exactly that many embedded slots
([Language conditioning](51-language-conditioning.md)).

**`vlm_backbone` and `vlm_foundry_backbone`.** `VLMHFBackboneWrapper`
([`vlm_hf_backbone.py`](../../vla_foundry/models/vision_language_backbones/vlm_hf_backbone.py),
Hugging Face VLMs) and `VLMFoundryBackboneWrapper`
([`vlm_foundry_backbone.py`](../../vla_foundry/models/vision_language_backbones/vlm_foundry_backbone.py),
VLMs trained here) share two functions in
[`base.py`](../../vla_foundry/models/vision_language_backbones/base.py):

- `insert_after_last_real_token` puts the action token directly after each prompt's last real
  token, so that padding ends up behind it; `all_tokens` mode does not call it.
- `conditioning_from_hidden_states` builds the conditioning. `action_token`: the states at each
  row's last real token, the action token. `all_tokens`: the real tokens moved to the right end,
  the padding zeroed and, with `conditioning_len > 0`, zeros added in front (or slots cut from the
  front) to that length; a prompt with more real tokens raises a `ValueError`.

**Config keys.** The conditioning's shape is set under `model.vision_language_backbone`. For
patches plus text slots (from `dp_vit_lang_reactive.yaml`):

```yaml
type: vit_backbone
img_size: 224            # (224 / 16)^2 = 196 patches per image
patch_size: 16
hidden_dim: 256
text_tokens: 32          # default 0: image-only
text_vocab_size: 49280   # required when text_tokens > 0
```

For every token of the from-scratch VLM (`vla_custom_tokens_reactive.yaml`; `vla_custom.yaml` omits
the last two keys, leaving the default `conditioning: action_token`):

```yaml
type: vlm_foundry_backbone
num_vlm_layers_to_use: 4
conditioning: all_tokens
# 3 template tokens + 2 x 256 image tokens + the instruction with its 2 closing
# tokens (at most 8): 523, with 5 spare
conditioning_len: 528
```

The number of images, state tokens and action-step tokens follows from `data.camera_names`,
`data.image_indices` and `data.lowdim_past_timesteps`.

**Running it.** The policy of the table below, on plain demonstrations (the
[README](../../README.md) adds `--random_start --disturb_prob 0.5` to the first line, for a policy
that copes when the object moves); the config switches averaged weights on itself:

```bash
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k
sim/pickup/scripts/train_resilient.sh dp_vit_lang_reactive pick_target_4k my_run 5000000
sim/pickup/scripts/eval_checkpoints.sh experiments/pickup/my_run pick_target
.venv/bin/python -m sim.pickup.summarize --task pick_target --runs my_run
```

The all-token VLA needs a stage-2 VLM in `experiments/pickup/custom_vlm`
([The from-scratch path](62-the-from-scratch-path.md)). On 4000 demonstrations and 3M samples it
ends at 86% / 87% (trained / averaged weights) with the caption-pretrained backbone, 75% / 85% with
a randomly initialised one (`scratch`; [field notes](../notes.md), "Pretraining a backbone: mainly
an earlier start"):

```bash
sim/pickup/scripts/custom_path.sh vla pick_target_4k my_vla_tokens 3000000 pretrained vla_custom_tokens_reactive
```

## What we measured

All numbers are from the [lab notebook](../lab-notebook.md): closed-loop success (the policy drives
the simulated arm) on **layouts** the policy never saw (what a seed fixes: the objects, their poses,
the named one and the arm's start pose), one training seed per run. A **sample** is one training window (one frame's observations and the actions that follow) drawn by
the data loader.

**One vector, one object.** On `pick_cube` (machine A, 200 demonstrations, 300k samples, 50
episodes) one pooled vector per image from a pretrained CLIP encoder gave 2/50 = 4% and every
patch as a token 15/50 = 30%; the two policies differ in more than that
([One vector or many tokens](21-pooled-vs-patch-tokens.md)).

**One vector, three objects.** The two first-round VLAs with one conditioning vector
(`vla_custom`, VLM pretrained or randomly initialised; machine A, 1500 demonstrations, 1.0M
samples, with history) scored 0/100 and 1/100 on `pick_target`. They **closed at the named
object** (the object nearest the gripper when it first closed was the one the instruction named)
in 33% and 29% of episodes (chance 33%), a median 89 mm from the nearest object. These are neither
near misses nor reaches toward a point between the objects: where the gripper closed did not
depend on the scene ([field notes](../notes.md), "A failing policy ignores the scene; it does not
average"). `vla_qwen` (pretrained Qwen3-VL-2B, one vector, 400k samples) scored 6/100.

**Tokens, three objects.** `pick_target` on machine B (L40S GPUs, 128 CPU cores), 100 layouts
per row, the 48.9M-parameter patch-token policy with the instruction (12.6M of it the instruction
embedding table), every row trained with the old position table (below). Each pair is the trained
weights / their running average (raw / EMA, [The training loop](80-the-training-loop.md)).
**Slope** is the least-squares slope of where the gripper first closed against where the named
object was: 1 if it closes where the object is, 0 if where it closes does not depend on the scene.

| Policy | Demos | Samples | Success | Closed at the named object | Slope |
|---|---|---|---|---|---|
| `dp_vit_lang` (history) | 4000 | 3.0M (stopped) | 2% / 2% | 30% / 35% | 0.11 / 0.08 |
| `dp_vit_lang_reactive` (no history) | 1500 | 1.0M | 3% / 5% | 49% / 41% | 0.29 / 0.18 |
| same | 1500 | 3.0M | 64% / 62% | 99% / 100% | 0.96 / 0.97 |
| same | 4000 | 5.0M | 83% / 85% | 100% / 100% | 0.98 / 0.98 |

What this shows:

- **Tokens carry the place and the instruction.** In the last row the gripper closed at the named
  object in every episode, 7 mm (median) from it, against the expert's 5 mm. That run's
  remaining failures, mostly grasp angles on the T block (55%), were mainly the position bug: after the fix
  the T block is at 87-93% and the rest is spread thinly over the objects
  ([the `pick_target` case study](101-case-pick-target.md)).
- **Being in the sequence is not being used.** The 424 conditioning tokens were there from the
  first optimiser step, yet the last row's run stood at 0% / 1% after 1.0M samples (slope 0.11 /
  0.14) and 4% / 36% after 2.0M (slope 0.81 / 0.96). Rows two and three are one policy on one
  dataset: the 1.0M-sample run ended inside that transition
  ([How much data and training](71-how-much-data-and-training.md)).
- **The old position table delayed the start and lowered the end.** Computed in half precision, it held 641 distinct
  positions out of 2048, and the 15 action-step tokens of the no-history policy shared 8. Trained
  again with the fix, the last row's recipe ends at 93% (trained weights) and starts to look earlier
  (58% against 4% at 1.5M samples). A second pair of runs with other seeds, differing in the table
  alone, ends at 96% / 96% against 90% / 89% ([Positions](12-positions.md)).
- **An input that holds the answer displaces the image.** The history policy reads the target
  from its previous action. At 2.5M samples (EMA) it succeeded in 1/100 layouts as served, 22/100
  when the previous action sent with its first query was the expert's first command, and 77/100
  when the expert drove the first 10 control steps. In training, the previous-action slot of a
  window that starts at the first frame holds a copy of that command; when served, the hold command
  ([History and shortcuts](52-history-and-shortcuts.md)).
- **A policy that succeeds may read the images once.** Sent the first frame's images for the whole
  episode (`eval.py --stale_images`), the 85% policy still closed at the named object in 99% of
  episodes, 17 mm (median) from it, and succeeded in 1/100: after its first look the state token
  says where the arm is heading ([What a policy looks at](95-what-a-policy-looks-at.md)).

What it does not show:

- That the single vector caused the first-round failures, or that `all_tokens` alone fixes them.
  Those VLAs also had history and 1.0M samples on 1500 demonstrations, each of which alone kept a
  patch-token policy with the instruction at 2% to 5%. With every token, no history and 4000
  demonstrations the from-scratch VLA reaches 86% / 87% (3M samples) and the Qwen VLA 70% / 72%
  (0.5M samples), against 0-6% with one vector; no run changes only the conditioning.

## Where it goes wrong

**Low loss, a complete grasp motion, an empty gripper.** `vla_custom` ended at a loss of about
0.03. `sim/pickup/eval.py` prints how often the gripper closed at the named object and the
median miss; `sim/pickup/summarize.py` adds the slope. Near 1 with a miss of a few
centimetres, the policy is imprecise ([Measuring failure](92-measuring-failure.md)). Near 0, it is
not looking: check that the conditioning is more than a few vectors, that the policy is not given
its previous action, and that training did not end before the slope moved in a comparable run
(the evidence for each, with its limits, is above).

**The previous action in the sequence.** If a policy with history scores near zero, run
`eval.py --first_prev_action expert` and `--expert_prefix 10`: a policy that scores far better
with them than without reads its target from its own past. Start without
history. `collect.py --hold_steps 1`, a frame of the robot at rest at the start of every
demonstration, closes the leak at the first frame but did not help: trained on such demonstrations
with the fixed position table, the history policy ends at 0% / 4% after 5M samples, and where it
closes never depends on the scene. From the second step on the previous action still predicts the
next one ([History and shortcuts](52-history-and-shortcuts.md)).

**The action token behind padding.** Appended after the batch padding, it read a short
instruction one position later in a training batch than when served alone. Both wrappers now place
it after the last real token; the effect on success was not measured
([Train and inference must see the same thing](93-train-inference-skew.md)).

## Background and further reading

- Mercat et al., "VLA Foundry: A Unified Framework for Training Vision-Language-Action Models",
  2026. [arXiv:2604.19728](https://arxiv.org/abs/2604.19728). Its flow transformer is conditioned
  on the last N layers' hidden features (4 in its experiments) at what it calls an observation
  token: the action token here.
- Octo Model Team (Ghosh et al.), "Octo: An Open-Source Generalist Robot Policy", 2024.
  [arXiv:2405.12213](https://arxiv.org/abs/2405.12213). Its diffusion action head reads the embeddings of readout tokens, the same device, which attend to
  the observation and task tokens without being attended to by them.
- Black et al., "π0: A Vision-Language-Action Flow Model for General Robot Control", 2024.
  [arXiv:2410.24164](https://arxiv.org/abs/2410.24164). The other design: the tokens of the noisy
  chunk attend to the full input sequence through self-attention shared with the VLM. `all_tokens`
  here is looser: a separate action transformer attends to the VLM's last hidden states.

## Check yourself

1. `dp_vit_reactive.yaml` has no text slots, `image_indices: [0]` and `lowdim_past_timesteps: 0`.
   Compute `N`, `P`, `T` and the sequence length. (Answer: 392, 1, 15 and 409.)

2. Predict the two shapes and the six values this prints, then run it:

   ```bash
   .venv/bin/python - <<'EOF'
   import torch
   from vla_foundry.models.vision_language_backbones.base import conditioning_from_hidden_states

   mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])  # a short prompt next to a long one
   layers = [torch.randn(2, 5, 512) for _ in range(4)]      # hidden states of the last four layers
   one = conditioning_from_hidden_states(layers, mask, "action_token")
   every = conditioning_from_hidden_states(layers, mask, "all_tokens", conditioning_len=6)
   print(one.shape, every.shape)
   print((every[0].abs().sum(-1) == 0).tolist())            # zero slots of the short prompt
   EOF
   ```

3. Find the assertion that shows `action_token` mode returns exactly one vector, and the test that
   shows where the Hugging Face wrapper puts the action token in a padded batch:

   ```bash
   .venv/bin/python -m pytest tests/essential/models/test_vlm_foundry_backbone_conditioning.py \
       tests/essential/models/test_backbone_conditioning.py -q
   ```

4. A policy with history scores 2% after 3.0M samples, with a slope of 0.08. Name two changes
   to make before touching the backbone, and the two `eval.py` options that test whether it reads
   its target from its previous action.
