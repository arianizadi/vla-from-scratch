---
title: Telling Tokens Where They Are
part: 1 Language models
order: 12
summary: Explain how rotary position embeddings make attention depend on relative offsets, find where position enters each model in this repository, and check that a policy's tokens keep the same offsets in training and serving and a distinct position each.
status: draft
---

# Telling Tokens Where They Are

Attention compares tokens by content and has no notion of order, so a transformer has to be told
where each token is; this repository does it with rotary position embeddings in its transformer
(the language model, and the action head that produces actions) and a learned table in its vision
transformer (ViT). The action head reads one sequence: a token for the noise level, the
conditioning (the tensor `[batch, tokens, width]` from the backbone, the model that encodes the
instruction and the images), the joint state, and one action-step token per control step. A
policy needs the same offsets between tokens in training and serving and a distinct position for
every token, and both have been broken here without an error.

## The idea

Attention ([the transformer block](11-the-transformer-block.md)) scores a query from one token
against a key from every token with a dot product and mixes the values by the softmax of the
scores. Queries, keys and values are computed from a token's content alone, so with no mask and no
position information, shuffling the input tokens only shuffles the outputs: "red cube left of
green can" and "green can left of red cube" are the same input. So are the 196 patch tokens of a
camera frame ([images to tokens](20-images-to-tokens.md)) in any order, and the action-step tokens
of a chunk (the sequence of 15 future actions a policy predicts in one go), whose order is time. A
causal mask breaks the symmetry by itself, since the token at index $i$ can read exactly $i$
earlier tokens; the action head has no such mask.

A token's **position** is its index in the sequence, starting at 0. The **offset** from a key to a
query is the query's position minus the key's position; it is negative when the key comes later.
Three ways of supplying position are in common use (background; papers at the end):

| Scheme | Where position enters | A score depends on |
|---|---|---|
| Learned absolute | a trained row per position, added to the token embedding | both absolute positions |
| Sinusoidal | fixed sines and cosines of the position, added the same way | both absolute positions |
| Rotary (RoPE) | queries and keys rotated by an angle proportional to position, in every attention layer; nothing trained | content and the offset only |

With the first two, attending by offset has to be learned, and a learned table has no row beyond
the trained length.

### Rotary in two dimensions

Take an attention head whose queries and keys have two numbers each. Turn the query at position
$m$ by $m$ radians and the key at position $n$ by $n$ radians, and let the content be the same
everywhere: query $(1, 0)$, key $(1, 0)$.

| Query at | Rotated query | Key at | Rotated key | Dot product |
|---|---|---|---|---|
| 3 | (-0.990, 0.141) | 5 | (0.284, -0.959) | -0.281 - 0.135 = -0.416 |
| 13 | (0.907, 0.420) | 15 | (-0.760, 0.650) | -0.689 + 0.273 = -0.416 |
| 3 | (-0.990, 0.141) | 4 | (-0.654, -0.757) | 0.647 - 0.107 = 0.540 |

Rows one and two share the offset $-2$ and the score $\cos(-2)$; row three, at offset $-1$, scores
$\cos(-1)$. Rotation changes the angle between the two vectors by $m - n$ and not their lengths.

<!-- interactive: rotary -->
*Figure: a query and a key drawn as arrows in the plane, each turned by its position. Moving both
positions by the same amount leaves the score fixed; changing the offset between them changes it.*

## How it works

Background: this is the construction of Su et al. (2021). For a head whose vectors have $d$ numbers
(the head dimension, `hidden_dim / n_heads`, which must be even):

1. Split the query into $d/2$ pairs of numbers. Each pair is a point in a plane.
2. Give pair $i$ a frequency $\theta_i = 10000^{-2i/d}$ for $i = 0 \dots d/2 - 1$, the
   frequencies of the sinusoidal scheme.
3. For the token at position $m$, rotate pair $i$ by the angle $m\theta_i$:
   $(x_1, x_2) \to (x_1\cos m\theta_i - x_2\sin m\theta_i,\; x_2\cos m\theta_i + x_1\sin m\theta_i)$.
4. Do the same to the key. Leave the value alone.
5. Compute attention scores from the rotated queries and keys.

Write $R(a)$ for the 2x2 matrix that rotates by angle $a$. Since $R(a)^\top R(b) = R(b - a)$, each
pair contributes

$$\langle R(m\theta_i)\,q,\; R(n\theta_i)\,k \rangle = \langle q,\; R((n - m)\theta_i)\,k \rangle
= |q|\,|k|\cos\big(\varphi_q - \varphi_k + (m - n)\theta_i\big)$$

where $\varphi_q$ and $\varphi_k$ are the angles of the unrotated query pair and key pair. The
score is a sum of such terms over the pairs. Content sets $|q|$, $|k|$ and the two angles; position
enters only through $m - n$. Because of those angles the sign of the offset matters: a head can
prefer "two tokens earlier" over "two tokens later".

With $d = 64$, the head dimension in the pickup configs (512 / 8 or 256 / 4), pair 0 turns 1
radian per position and pair 31 about 0.000133, repeating after about 47,000 positions.

Two consequences matter below. Inserting tokens between two groups changes every offset across
the gap: with conditioning tokens `C1 C2 C3` followed by action-step tokens `A1 A2`, `A1` reads
`C3` at offset 1, and at offset 3 once two padding tokens follow `C3`. And two tokens given the
same angles are at offset 0: between them the shuffle symmetry of "The idea" is back.

## In this repository

**Rotary, in the transformer.** In
[`vla_foundry/positional_embedding.py`](../../vla_foundry/positional_embedding.py), `get_pos_embed`
reads `positional_embedding_type` and returns `RotaryWithCast` (`"rotary"`, the default in
`TransformerParams` in `vla_foundry/params/model_params.py`, which no pickup config overrides),
the same wrapped by `Float32Module` (`"rotary_float32"`), the same with the table of older
checkpoints (`"rotary_legacy_autocast"`, below), or `identity_with_cast` (`"none"`, no position
information). Its only caller is `CustomAttn.__init__` in
[`vla_foundry/models/transformer.py`](../../vla_foundry/models/transformer.py), so every layer has
its own rotary module, which `CustomAttn.forward` applies to queries and keys of shape
`[batch, tokens, heads, head_dim]` just before the attention function. `Transformer.forward` adds
no position vector to the embeddings: this is the only place position enters.

Inside `RotaryEmbedding` and the two functions it calls:

- `reset_parameters` sets the buffer `inv_freq` to `1 / 10000 ** (arange(0, dim, 2) / dim)`, the
  $\theta_i$ above. Nothing is trained.
- `_update_cos_sin_tables` tabulates $\cos m\theta_i$ and $\sin m\theta_i$ for `max_seq_len`
  positions (2048 in every pickup config), and again when a longer sequence arrives or the device
  or dtype of the keys changes. `max_seq_len` is therefore not a limit.
- `apply_rotary_pos_emb` returns `x * cos + rotate_half(x) * sin`, which pairs number `i` with
  number `i + head_dim / 2`, not with its neighbour as in the paper. The rotation is the same.
- `offset` is the number of tokens already in the key-value cache (earlier keys and values, kept
  while text is generated; [generation](14-generation.md)), so a token generated alone is still
  rotated by its true position.

**A learned table, in the ViT.** `ViTPatchEmbeddings` in
[`vla_foundry/models/vit.py`](../../vla_foundry/models/vit.py) adds `position_embedding`, a
parameter of shape `[1, num_patches, hidden_dim]` initialised with `torch.rand`, to the patch
embeddings once: 256 rows in the ViT of the from-scratch vision-language model (VLM;
`sim/pickup/configs/custom_vlm.yaml`, patch 14), 196 in the policy ViT (the `dp_vit*.yaml`
configs, patch 16). `ViT.forward` folds several images into the batch dimension, so all share the
table: it says where in an image a patch is, not which image it came from.

### The action head's sequence

`DiffusionPolicy._build_transformer_input` in
[`vla_foundry/models/diffusion_policy/diffusion_policy.py`](../../vla_foundry/models/diffusion_policy/diffusion_policy.py)
concatenates a time token carrying the noise level, the conditioning, the state tokens and the
action-step tokens. `sim/pickup/configs/dp_vit_lang_reactive.yaml` configures the policy that
solves `pick_target` ([the `pick_target` case study](101-case-pick-target.md)): two cameras with
one frame each, patch 16, `text_tokens: 32`, one state, and the 15 actions of the chunk
(`lowdim_past_timesteps: 0`; `lowdim_future_timesteps: 14` from the data preset).

```text
shape [1, 441, 256]

position   0      1 .. 32       33 .. 424           425     426 .. 440
token      time   text slots    patches (2 x 196)   state   action steps (15)
```

`dp_vit_lang.yaml` adds history ([History and shortcuts](52-history-and-shortcuts.md)): the
previous frame of each camera, the previous state, and the previous action as a first action-step
token, given without noise.

```text
shape [1, 835, 256]

position   0      1 .. 32       33 .. 816           817 818   819 .. 834
token      time   text slots    patches (4 x 196)   state     action steps (16)
```

This transformer has `is_causal: false` and is called without an attention mask, so every token
attends to every other. A patch token's content can carry its place within the image, from the
ViT's table; which image it belongs to, and which control step an action-step token stands for,
reach the action head only through rotary position.

### The position table under bfloat16

`_update_cos_sin_tables` builds the table again in the first forward pass, in the dtype of the
keys. Every pickup config trains with `precision: amp_bf16`: the forward pass runs under
**autocast**, PyTorch's mixed-precision switch, which executes many operations in **bfloat16**, a
16-bit float with 8 significant bits ([training a language model](13-training-a-language-model.md)).
Until it was fixed, the product $m\theta_i$ was one of them, so position, frequency and angle were
each rounded to 8 significant bits.

Eight significant bits hold every whole number up to 256, only the even ones up to 512, multiples
of 4 up to 1024 and multiples of 8 beyond: positions 427, 428 and 429 all became 428. Of the 2048
positions in the table, 641 distinct ones remained (lab notebook, "Bugs found and fixed"), and
tokens that shared one were rotated alike in every pair of every layer:

| Policy | Action-step tokens at | Positions under the old table |
|---|---|---|
| `dp_vit_lang_reactive` | 426 to 440 (15) | 8: the even numbers 426 to 440 |
| `dp_vit_lang` | 819 to 834 (16) | 4: 820, 824, 828 and 832 |

Patch tokens above position 256 shared rotations too, but their content carries the ViT's table.
Action-step tokens have no second source of order beyond their contents, the noisy actions.
Training and serving built the same table, so comparing the two paths
([train/inference skew](93-train-inference-skew.md)) could not have found it.

**The fix.** The angles are now computed in float32 inside `torch.autocast(..., enabled=False)`,
and only the finished cosines and sines are cast. `positional_embedding_type:
rotary_legacy_autocast` keeps the old computation for checkpoints trained before the fix, whose
weights were fitted to the old table; for a patch-token policy the key goes under
`model.transformer` in the run's saved `config.yaml`, which `sim/pickup/policy_server.py` reads.
Do not train a new run with it, except to measure what the old table costs.

**The test.** `tests/essential/models/test_rotary_precision.py` runs `RotaryWithCast(64, 2048)`
under bfloat16 autocast and requires 1024 distinct table rows for 1024 positions, and agreement
with a float32 reference to within 0.01. A third test, which needs a GPU, checks that the legacy
table still collapses.

### Why the conditioning has a fixed length

`RoboticsProcessor.process_inputs` (`vla_foundry/data/processor/robotics_processor.py`) wraps each
instruction with `apply_chat_template` and pads a batch to its longest prompt; the four
`pick_target` instructions give `dp_vit_lang_reactive` prompts of 12 or 13 tokens. The policy
server calls the same function on a batch of one, where nothing is padded. If the padded length
reached the action head, offsets would depend on the rest of the training batch.

`ViTBackboneWrapper._embed_text` in
[`vit_backbone.py`](../../vla_foundry/models/vision_language_backbones/vit_backbone.py) sorts each
prompt's real tokens to the front, pads or truncates to exactly `text_tokens` slots, adds a learned
`text_position`, applies a layer norm and zeroes the padding slots
([Language conditioning](51-language-conditioning.md)). Token `k` of the prompt is always at
sequence position `1 + k`. The chat template puts three tokens and one `<image>` placeholder per
image in front of the instruction and two closing tokens behind ([tokens](10-tokens.md)), so its
first word is in slot 5 with two images and in slot 7 with four.

With `conditioning: all_tokens` (the `vla_*_tokens*.yaml` configs),
`conditioning_from_hidden_states` in
[`base.py`](../../vla_foundry/models/vision_language_backbones/base.py) hands over the hidden
states of every prompt token of a VLM backbone, right-aligned in exactly `conditioning_len` slots
with zeros in front. With `conditioning_len: 528` (`vla_custom_tokens_reactive.yaml`, the
from-scratch VLM) the action head sees `[1, 545, 256]`: the prompt's last token always at 528,
state at 529, action-step tokens at 530 to 544. A longer prompt raises a `ValueError`. The count
is fixed as well as the alignment: without a mask, a zeroed slot is still a key that every token
attends to.

With the default `conditioning: action_token`, the hidden states of one special token added to the
prompt, the action token, are the whole conditioning; `insert_after_last_real_token` in the same
file places it after the last real token, not after the batch padding
([train/inference skew](93-train-inference-skew.md)).

## What we measured

No run in the [lab notebook](../lab-notebook.md) compares learned, sinusoidal and rotary schemes.
It compares the old table with the fixed one in two pairs of runs of `dp_vit_lang_reactive`
(4000 demonstrations, 5M samples, machine B (L40S GPUs, 128 CPU cores)). Success on 100 unseen
layouts, trained weights, with the averaged weights (their running average) in brackets where
saved.

| Seeds | Old table | Fixed table |
|---|---|---|
| first pair | 83/100 (85/100) | 93/100 |
| second pair (`--data.seed 43 --hparams.seed 43`) | 90/100 (89/100) | 96/100 (96/100) |

- **First pair.** Its old-table run predates the fix, so other code changes ride along with the
  table. Per object at the final checkpoint the cube and the T block gain most
  (cube 23/24 against 18/24, T block 20/23 against 15/23). The fixed table also makes the policy
  start to look earlier (where the gripper closes begins to follow the named object): it closed at
  the named object in 65% of layouts against 35% after 1.0M samples, and succeeded in 58% against
  4% after 1.5M.
- **Second pair.** Trained from one version of the code, once with `rotary` and once with
  `rotary_legacy_autocast`: it differs in the table alone. Again the fixed table makes the policy
  start to look earlier: after 1.0M samples the gripper closed at the named object in 83% of
  layouts against 44%.

With the trained weights the old table cost 10 points at the end for one pair and 6 for the other,
and in both it delayed the onset of looking (field notes, "Train/inference mismatches hide in
preprocessing"; the onset is the subject of
[how much data and training](71-how-much-data-and-training.md)).

- **Policies with history.** On the single cube (`pick_cube`, 1000 demonstrations, 1.5M samples,
  trained / averaged weights) the history policy went from 62% / 70% before the fix to 96% / 98%
  after it: there history costs nothing once the table is fixed. On three objects it fails for
  another reason ([History and shortcuts](52-history-and-shortcuts.md)).
- **The from-scratch language model and VLM** on machine B. The language model's 256-token
  sequences use positions 0 to 255, all of which bfloat16 holds exactly. The VLM's 320-token
  sequences were merged in pairs above 256; trained before the fix, it names a held-out scene's
  objects in left-to-right order in 249/300 cases (83%), and retrained with the fix (same data and
  budget) in 267/300 = 89% (final loss 0.059). Its use as a policy's backbone is in
  [pretrained backbones](102-case-pretrained-backbones.md).

## Where it goes wrong

**Tokens that share a position.** No error, and training agrees with serving. Check: build the
module under the precision you train with and count the distinct rows of its table, as
`test_rotary_precision.py` does. Compute such a table in float32 with autocast off.

**A checkpoint served on the other table.** A run trained before the fix has `rotary` (or no key,
which defaults to it) in its saved `config.yaml`, and `rotary` now builds the float32 table: the
server loads its weights onto a table they were not fitted to, without a warning. Check when a
checkpoint was trained and set the key under `model.transformer` to match. What this costs was not
measured.

**Padding between conditioning and action-step tokens.** No error and a normal training loss, but
the served policy receives inputs it was not trained on. Check: compare the conditioning of one
prompt alone and inside a padded batch, as the tests under "Check yourself" do.

**A prompt longer than the slots.** `_embed_text` silently keeps the first `text_tokens` tokens
(32 in the pickup configs): the two closing tokens are lost first, then the end of the
instruction. Template tokens and placeholders already take 7 slots with two images and 9 with
four.

**Sequences longer than those trained.** The angles are defined, but the weights were fitted on
shorter offsets. Train at the longer length: the VLM, trained on 320 tokens, reads about 523 or
1,035 as a policy's backbone, and the policy's training updates it.

**`positional_embedding_type: none` with `is_causal: false`.** Training runs, but the action head
receives no order: its images cannot be told apart and action-step tokens lose their control step.
In a causal model the mask carries order (background: Kazemnejad et al., 2023). Check this key in
the resolved config before training a non-causal model.

## Background and further reading

- Vaswani et al., "Attention Is All You Need", 2017, <https://arxiv.org/abs/1706.03762>. The
  transformer, with sinusoidal position vectors added to the embeddings; learned ones gave nearly
  identical results.
- Shaw et al., "Self-Attention with Relative Position Representations", 2018,
  <https://arxiv.org/abs/1803.02155>. A fourth scheme: learned vectors for the signed offset
  between tokens, added inside attention.
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021,
  <https://arxiv.org/abs/2104.09864>. Rotates queries and keys so that their inner product depends
  on relative position; cited in `RotaryEmbedding`'s docstring.
- Dosovitskiy et al., "An Image is Worth 16x16 Words", 2020, <https://arxiv.org/abs/2010.11929>.
  The vision transformer, with learned position embeddings added to patch embeddings.
- Kazemnejad et al., "The Impact of Positional Encoding on Length Generalization in Transformers",
  2023, <https://arxiv.org/abs/2305.19466>. Decoder-only models on sequences longer than those
  trained on: the common schemes generalised poorly and no position encoding did best.

## Check yourself

1. Predict the four numbers before running this, then explain why the third equals the first.

   ```bash
   .venv/bin/python - <<'EOF'
   import torch
   from vla_foundry.positional_embedding import RotaryEmbedding

   rot = RotaryEmbedding(dim_model=2, seq_len=16)    # one pair, 1 radian per position
   q = torch.tensor([1.0, 0.0]).expand(1, 16, 1, 2)  # [batch, tokens, heads, head_dim]
   k = torch.tensor([1.0, 0.0]).expand(1, 16, 1, 2)
   q, k = rot(q, k)
   scores = torch.einsum("bqhd,bkhd->qk", q, k)      # scores[query position, key position]
   print(scores[3, 5].item(), scores[13, 15].item(), scores[5, 3].item(), scores[3, 4].item())
   EOF
   ```

   Change the key to `[0.0, 1.0]` and run it again. `scores[3, 5]` and `scores[5, 3]` now differ in
   sign; say why.

2. Run the padding tests, then find the line of `_embed_text` that makes the input `[[0, 0, 8, 9]]`
   give the same conditioning as `[[8, 9]]`:

   ```bash
   .venv/bin/python -m pytest tests/essential/models/test_vit_backbone_text.py \
     tests/essential/models/test_vlm_foundry_backbone_conditioning.py -q
   ```

3. Predict the three lines this prints: the number of distinct positions, and the positions the
   action-step tokens of the two policies had under the old table. Then run the precision tests;
   without a GPU the third is skipped.

   ```bash
   .venv/bin/python - <<'EOF'
   import torch

   position = torch.arange(2048, dtype=torch.float32)
   rounded = position.to(torch.bfloat16).float()  # what the old table multiplied the frequencies by
   print(rounded.unique().numel())
   print(rounded[426:441].tolist())               # dp_vit_lang_reactive: 15 action-step tokens
   print(rounded[819:835].tolist())               # dp_vit_lang: 16 action-step tokens
   EOF
   .venv/bin/python -m pytest tests/essential/models/test_rotary_precision.py -q
   ```
