---
title: The Instruction Has to Reach the Action Head
part: 5 Conditioning the action head
order: 51
summary: Trace an instruction from a string to the tokens the action head attends to, and tell from two numbers of an evaluation and two tests whether a trained policy follows it.
status: draft
---

# The Instruction Has to Reach the Action Head

On `pick_target` three objects lie on the table and the instruction ("pick up the green can") is
the only input that says which one to lift. The action head, the transformer that turns
noise into a chunk (the 15 future actions a policy predicts in one go), can follow it only if it
arrives as tokens in the sequence that transformer reads.

## The idea

The action head reads a time token (the noise level), the **conditioning** (the tensor
`[batch, tokens, width]` handed over by the backbone, the network that encodes images and text),
the joint state, and one action-step token per action of the chunk
([What the action head sees](50-what-the-action-head-sees.md)). An instruction can change the
actions only through conditioning tokens. Two ways are implemented here:

- **Text slots.** Each token id of the prompt (the integers a tokenizer turns text into; the prompt
  is the instruction wrapped in a chat template) gets a learned vector, placed in front of the
  patch tokens of a small vision transformer (ViT), one token per 16-pixel square of a camera frame. No language model; words and patches first meet
  in the action head.
- **Through a vision-language model (VLM).** The instruction is part of a VLM's prompt, and the
  action head receives the VLM's hidden states (the vectors a layer holds for each token): those
  of one appended **action token**, or those of every prompt token.

A worked example: in
[`dp_vit_lang_reactive.yaml`](../../sim/pickup/configs/dp_vit_lang_reactive.yaml), the config that
solves `pick_target` here, the prompt for the red cube fills 12 of 32 slots:

```text
slot   0             1      2   3        4        5      6    7     8       9      10                  11   12-31
token  <|im_start|>  User   :   <image>  <image>  pick   Ġup  Ġthe  Ġred    Ġcube  <end_of_utterance>  Ċ    zeros
id     1             11126  42  49190    49190    18188  614  260   2382    20636  49279               198

"pick up the green can" changes two slots:                          Ġgreen  Ġcan
                                                                    2654    416
```

`Ġ` marks a leading space, `Ċ` a newline ([Text becomes tokens](10-tokens.md)). Behind the
slots come 2 x 196 patch tokens; with the time token, the state token and 15 action-step tokens
the action head reads 441 tokens. Whatever sends the arm to the can rather than the cube comes from
two of them, combined with the patches that show where the can is.

<!-- interactive: sequence-layout -->
*Figure: the 441 tokens as a row of cells. Changing the instruction recolours the slots that
differ; a VLM config shows its prompt with the action token behind it.*

Being in the sequence is not being used. After 1.0M training samples (a **sample** is one
**window**, the observations at one frame plus the actions that follow, drawn by the data loader)
on 4000 demonstrations (before the position fix, described under What we measured) this policy
closed at the named object no more often than chance; after 5.0M, every time.

## How it works

### Text slots next to the patches

For a batch of `B` windows, two cameras and no **history** (the previous frame, state and action
as extra inputs):

1. **Prompt.** `RoboticsProcessor.process_inputs`
   ([`robotics_processor.py`](../../vla_foundry/data/processor/robotics_processor.py)) wraps each
   instruction with `apply_chat_template` and tokenises the batch, padded to its longest prompt:
   `input_ids` and `attention_mask`, `[B, L]`. The cube and can prompts have 12 ids, the hex nut
   and T block prompts 13.
2. **Slots.** `ViTBackboneWrapper._embed_text`
   ([`vit_backbone.py`](../../vla_foundry/models/vision_language_backbones/vit_backbone.py)) moves
   each row's real ids to the front, pads or cuts the row to `text_tokens` = 32, looks the ids up
   in `text_embedding` (a table `[49280, 256]`), adds `text_position` (one learned vector per
   slot), applies the layer norm `text_norm` and zeroes the padding slots: `[B, 32, 256]`.
3. **Conditioning.** `get_action_conditioning` puts the slots in front of the patch tokens:
   `[B, 424, 256]`.
4. **Sequence.** `DiffusionPolicy._build_transformer_input` adds the time, state and action-step
   tokens: `[B, 441, 256]`. The action transformer is not causal (`is_causal: false`): every token
   attends to every other, so an action-step token can read the word slots and the patches.

Moving the real ids to the front makes the slots independent of padding: a 12-id prompt padded
in a training batch and served alone gives the same 32 vectors
([Train/inference skew](93-train-inference-skew.md)). No knowledge of language comes in: the four
instructions use 19 distinct ids, and the other rows of the table get no gradient, so an unseen
wording would select untrained rows (an inference from the code).

### Through a VLM

In [`vla_custom.yaml`](../../sim/pickup/configs/vla_custom.yaml) and
[`vla_qwen.yaml`](../../sim/pickup/configs/vla_qwen.yaml) the prompt goes through a VLM
([Joining vision and language](30-joining-vision-and-language.md)). The key `conditioning` of both
backbone wrappers (`vlm_foundry_backbone` for the from-scratch VLM, `vlm_backbone` for Hugging Face
models such as Qwen3-VL) sets what the action head receives; `conditioning_from_hidden_states` in
[`base.py`](../../vla_foundry/models/vision_language_backbones/base.py) builds it.

`action_token` (the default): the action token (`_action_token_id`, a new row of the VLM's
embedding table) is inserted directly after each prompt's last real token
(`insert_after_last_real_token`), so padding cannot move it. Its hidden states in the last four
layers, concatenated, are the whole conditioning: `[B, 1, 2048]` for the from-scratch VLM (512
wide). That VLM is causal (each position sees only itself and earlier ones), so this one vector, the only
position that has seen everything, has to carry both the instruction and where the named object is.

`all_tokens` (`vla_custom_tokens_reactive.yaml`, `vla_qwen_tokens_reactive.yaml`): the states of
every real prompt token, right-aligned and padded in front with zeros to `conditioning_len` (528
and 128 in those configs): `[B, 528, 2048]` for the from-scratch VLM. The images come first in the
prompt, so their states cannot see the instruction; the action head combines the two, as it does
with text slots.

### Closed at the named object, and the slope

`GraspProbe` in [`diagnostics.py`](../../sim/pickup/diagnostics.py) records where the gripper and
every object are at the first action that commands the gripper to close
([Measuring failure](92-measuring-failure.md)). Two summaries bear on the instruction:

- **Closed at the named object.** The share of episodes in which the object nearest the gripper
  at that moment was the named one: `grasp.closed_nearest_is_target_rate` in `results.json` (out
  of all episodes), "At named
  object" in [`summarize.py`](../../sim/pickup/summarize.py)'s table. `PickupEnv.reset` draws the
  target uniformly, independently of where the objects lie, so a policy that cannot read the
  instruction scores 33% on average.
- **Slope.** The least-squares slope of where the gripper closed against where the named object
  was, both table axes pooled (`close_slope` in `summarize.py`). 1: it closes where the object is.
  0: where it closes does not depend on the scene.

The instruction says which object, never where: the **layout** (which objects are on the table,
where, which one is named, and the arm's start pose; fixed by the seed) is random. So both numbers
measure the instruction and the image together, and the median miss (the table-plane distance from
the gripper to an object when it first closed) to the *nearest* object separates the two ways of
failing. Reference values from the layout rule alone (exercise 2;
arithmetic, not measurements):

| A policy that closes | Closed at the named object | Slope | Median miss to nearest |
|---|---|---|---|
| at the named object | 100% | 1 | 0 mm |
| at one of the three objects, at random: it looks, ignores the instruction | 33% | 0.20 | 0 mm |
| at a random place where objects can lie: it looks at nothing | 33% | 0.00 | 89 mm |

### Two tests of the instruction

A rate far above 33% shows the instruction is used; two tests go further.

1. **The same layout, another instruction.** `PickupEnv.reset(seed, target_index)`
   ([`env.py`](../../sim/pickup/env.py)) builds the layout of `seed` and names its object number
   `target_index`; the layout does not depend on the index. With the same starting noise
   (`noise_seed`) for the three episodes of a layout, a policy that ignores the instruction closes
   at about the same place every time; one that follows it closes at three different objects.
2. **A handover toward another object.** `eval.py --expert_prefix K --prefix_target distractor`
   lets the scripted expert (the hand-written controller whose episodes are the demonstrations)
   drive the first `K` control steps (one command to the arm; 20 per second) toward another object
   of the layout, then hands over. The instruction is unchanged; a policy that still reads it turns
   back. Run it only on a policy that passes the first test.

## In this repository

**Text slots.** `text_tokens` and `text_vocab_size` are fields of `ViTBackboneParams`
([`model_params.py`](../../vla_foundry/params/model_params.py)), both 0 by default. From
`dp_vit_lang_reactive.yaml`:

```yaml
model:
  vision_language_backbone:
    type: vit_backbone
    # ... the ViT's own keys
    text_tokens: 32          # slots; 0 gives an image-only backbone
    text_vocab_size: 49280   # rows of the embedding table = the tokenizer's vocabulary
data:
  processor: simple_vlm      # supplies the tokenizer
  img_num_tokens: 1          # one <image> placeholder per image in the prompt
```

The controls `dp_vit.yaml` and `dp_vit_reactive.yaml` have no text keys and `processor: "none"`,
whose `PassthroughProcessor` returns the id 0 for every instruction.

**Tests.** In `test_vit_backbone_text.py` a prompt padded on either side gives the tensor it gives
alone, and another id changes the text slots but not the patch tokens;
`test_backbone_conditioning.py` tests the two VLM modes.

```bash
.venv/bin/python -m pytest tests/essential/models/test_vit_backbone_text.py \
    tests/essential/models/test_backbone_conditioning.py -q
```

**Training and the two numbers.** The [README](../../README.md)'s first run without its two data
options (the plain demonstrations of `pick_target_4k`, under another name), and a
control without the instruction (not run here). The configs keep averaged weights (a running
average, EMA) next to the trained ones; every checkpoint (the
training state saved at one point) is evaluated with both.

```bash
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k_plain
# 10 checkpoints (the default is 6), as in the notebook's machine-B runs: one every 0.5M samples
PICKUP_NUM_CHECKPOINTS=10 sim/pickup/scripts/train_resilient.sh dp_vit_lang_reactive pick_target_4k_plain my_plain_run 5000000
PICKUP_NUM_CHECKPOINTS=10 sim/pickup/scripts/train_resilient.sh dp_vit_reactive pick_target_4k_plain my_control 5000000

# every checkpoint, trained and averaged weights, 100 unseen layouts each
sim/pickup/scripts/eval_checkpoints.sh experiments/pickup/my_plain_run pick_target
sim/pickup/scripts/eval_checkpoints.sh experiments/pickup/my_control pick_target

# one row per evaluation: success, "At named object", "Slope", the two misses
.venv/bin/python -m sim.pickup.summarize --task pick_target --runs my_plain_run my_control

# every object of a layout named in turn (collect.py passes target_index to reset)
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_cf4k --targets_per_layout 3
```

**The same layout, another instruction.** `eval.py` has no option for a target index, so this
script runs its loop in-process for each object of 20 unseen layouts. It was run against stand-ins
for the simulator and the policy only.

```bash
.venv/bin/python - experiments/pickup/my_plain_run <<'EOF'
import sys
from collections import Counter

from sim.pickup.diagnostics import GraspProbe
from sim.pickup.env import PickupEnv
from sim.pickup.eval import _jpeg
from sim.pickup.policy_server import Policy

policy = Policy(sys.argv[1], None, "cuda")  # newest checkpoint; averaged weights if the run saved them
env = PickupEnv("pick_target")
seeds = range(1_000_000, 1_000_020)
at_named, different = 0, Counter()
for seed in seeds:
    closed_at = []
    for index in range(3):  # the same layout, each of its three objects named in turn
        obs = env.reset(seed=seed, target_index=index)
        prev_obs, queue, probe, done = obs, [], GraspProbe(env), False
        while not done:
            if not queue:  # as in eval.py: a new chunk, of which 8 actions are executed
                images = {f"{cam}_t-1": _jpeg(prev_obs[cam]) for cam in env.camera_names}
                images.update({f"{cam}_t0": _jpeg(obs[cam]) for cam in env.camera_names})
                state = [prev_obs["state"].tolist(), obs["state"].tolist()]
                # noise_seed: every chunk of the layout's three episodes starts from the same noise
                actions = policy.infer(images, state, env.last_action, env.instruction, noise_seed=seed)
                queue = list(actions[:8])
            prev_obs = obs
            action = queue.pop(0)
            probe.update(action)
            obs, done, _ = env.step(action)
        nearest = probe.result()["close_nearest"]
        closed_at.append(nearest)
        at_named += int(nearest == env.target)
    different[len(set(closed_at) - {None})] += 1
print(f"closed at the named object: {at_named}/{3 * len(seeds)}")
print("layouts by the number of different objects the gripper closed at:", dict(sorted(different.items())))
EOF
```

The notebook's open-loop version ("First action chunk, open loop": 1500 demonstrations, 0.9M
samples, one draw per instruction): the first chunk heads for the named object in 35% / 48% of
layouts (training / unseen), and in 42% / 43% when another object is named; the policy did not yet
use the instruction.

**The handover.** `PICKUP_OUT` keeps the result apart from the plain evaluation; each episode in
`results.json` records `prefix_goal`, the object the expert headed for:

```bash
PICKUP_OUT=results/pickup/my_plain_run/pick_target/handover_25 sim/pickup/scripts/eval.sh \
    experiments/pickup/my_plain_run pick_target 100 --expert_prefix 25 --prefix_target distractor

.venv/bin/python - <<'EOF'
import json

r = json.load(open("results/pickup/my_plain_run/pick_target/handover_25/results.json"))
closed = [e for e in r["per_episode"] if e["close_nearest"]]  # the episodes in which the gripper closed
print("success:", r["successes"], "of", r["episodes"])
print("closed at the named object:", sum(e["close_nearest"] == e["target"] for e in closed), "of", len(closed))
print("closed at the expert's object:", sum(e["close_nearest"] == e["prefix_goal"] for e in closed), "of", len(closed))
EOF
```

## What we measured

Numbers from the [lab notebook](../lab-notebook.md): `pick_target`, 100 unseen layouts per
evaluation, one training seed per run, machine B (L40S GPUs, 128 CPU cores) unless stated. Pairs
are trained / averaged weights.

**With and without the instruction.** The patch-token policy without history (36.3M parameters,
48.9M with the instruction's embedding table), trained before the position fix
([Positions](12-positions.md)). Without the instruction (`dp_vit_reactive`, 1500 demonstrations,
1.0M samples) it closed at the named object at chance, 26% / 34% (slope 0.01 / 0.04). With it
(`dp_vit_lang_reactive`, 4000 demonstrations) it was still at chance after 1.0M samples and after
5.0M closed at the named object in every episode, with no episode lifting another object;
[How much data, how much training](71-how-much-data-and-training.md) follows that transition
checkpoint by checkpoint.

The failing policies of the notebook's first round (machine A, one RTX 4090, 24 GB; 1500
demonstrations, 1.0M samples, one conditioning vector for the VLAs) sit on the reference table's last line: `dp_vit` 32% at the named object, 110 mm (median)
from the nearest one; the two `vla_custom` VLAs 33% and 29%, 89 mm
([Measuring failure](92-measuring-failure.md); notes: "A failing policy ignores the scene; it does
not average").

<!-- interactive: grasp-map -->
*Figure: where the gripper first closed relative to the named object, one dot per episode, for
the policy without the instruction and for the policy with it, failing and working.*

**What the two numbers do not separate.** The rate counts the nearest object at the moment of
closing, so it rises before the gripper is close enough to grasp: after 2.0M samples the
4000-demonstration policy closes at the named object in 95% / 98% of episodes and still fails most
of them. The slope is pooled over both table axes and all four objects. Neither says *when* in the
episode the instruction was read; the handover test does.

**Reading the instruction earlier.** Of the changes that made the rate rise sooner with 4000
demonstrations, two bear on the instruction. With the position table fixed
([Positions](12-positions.md)) the 4000-demonstration recipe above (one instruction per layout)
closes at the named object in 65% of episodes after 1.0M samples, against 35% with the old table
(slope 0.58 against 0.11; notebook, "The same run before and after the fix", trained weights).
With one instruction per layout, remembering which object was picked in each training scene fits
the data as well as reading the instruction; naming every object of a layout in turn removes that
solution, and the rate after 1.0M samples is 98% (slope 1.02) against 65%
([Which demonstrations teach what](72-which-demonstrations-teach.md)). With 1500 demonstrations
and 1.0M samples, naming every object did not help (notebook, "Which demonstrations").

**The handover.** The 4000-demonstration policy after 5.0M samples reads the instruction at the
start and then follows its own arm: when the expert drives the first 25 control steps toward
another object, it closes at the expert's object in 100 of 100 episodes, although the instruction is in every query and it plans
again every 8 control steps (notebook, "What the no-history policy depends on";
[What a policy looks at](95-what-a-policy-looks-at.md)). Which demonstrations make it turn back is
the subject of [A policy that corrects itself](103-case-self-correction.md).

**Through a VLM, one action token.** `vla_qwen_pt1500` (notebook, "Qwen3-VL-2B backbone": history,
1500 demonstrations, trained weights; trained before the position fix, which did not affect its
20-token action head) closed at the named object in 71% of episodes after 0.40M samples, from 33%
after 0.10M: the instruction and a rough position got through one vector (slope 0.68), not a
precise one: it pointed at the object, 83 mm off, and rarely grasped it. With every token
(`conditioning: all_tokens`), no history and 4000 demonstrations, both VLAs work
([A backbone from scratch against a pretrained one](102-case-pretrained-backbones.md)). History and
the number of demonstrations changed with the conditioning, so the effect of the conditioning
alone was not measured (notes: "One pooled vector is not enough for grasping").

## Where it goes wrong

| What you see | Why | Check and fix |
|---|---|---|
| The rate stays near 33% at every checkpoint | No instruction in the input: `text_tokens` is 0 or `data.processor` is `"none"` (no error with `text_tokens: 32`) | Read both in `experiments/pickup/<run>/config.yaml` |
| The same, with `processor: simple_vlm` | `img_num_tokens` left at the preset's 64: the 32 slots fill with template and placeholders, the instruction is cut off silently | `img_num_tokens: 1`; longest prompt (13 ids, 15 with history) within `text_tokens` |
| Near 33%, slope near 0, although the slots differ | Too little training, or history: `dp_vit_lang`, 4000 demonstrations, 3.0M samples (before the position fix): 30% / 35% | Evaluate every checkpoint; start without history ([History and shortcuts](52-history-and-shortcuts.md)) |
| Well above 33%, slope well below 1, misses of centimetres | Reads the instruction, cannot place the object precisely (the Qwen run with one action token) | Spatial tokens: text slots next to patches, or `conditioning: all_tokens`; start without history |
| A reworded instruction ("grab the green can") followed less often than the training wording (expected, not measured here) | Its new ids select rows of `text_embedding` that never got a gradient (an inference from the code) | Compare the prompt's ids with those of the training instructions (exercise 1); train with the wordings you will serve |
| 100% at the named object, the wrong object after a handover | Follows the instruction only until its arm has a direction | The handover test; random start poses and recoveries in the data (`--random_start --disturb_prob 0.5`) |

## Background and further reading

- Brohan et al., "RT-1: Robotics Transformer for Real-World Control at Scale", 2022.
  [arXiv:2212.06817](https://arxiv.org/abs/2212.06817). Puts the instruction into the image
  encoder through FiLM layers ([arXiv:1709.07871](https://arxiv.org/abs/1709.07871)), which scale
  and shift features, instead of adding tokens.
- Octo Model Team (Ghosh et al.), "Octo: An Open-Source Generalist Robot Policy", 2024.
  [arXiv:2405.12213](https://arxiv.org/abs/2405.12213). Closest to the text slots: language
  tokens (16, from a pretrained t5-base encoder) and image patches in one transformer.
- Zhou et al., "LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action Models
  Beyond Memorization", 2025. [arXiv:2510.03827](https://arxiv.org/abs/2510.03827). Models above
  90% on a standard benchmark whose outputs stay the same when the instruction is corrupted.

## Check yourself

1. Predict which of the 32 slots differ between "pick up the red cube" and "pick up the blue hex
   nut", then run this (the first run downloads the tokenizer files). Then set
   `proc.image_seq_length = 64` and explain the last two lines.

   ```bash
   .venv/bin/python - <<'EOF'
   import torch
   from vla_foundry.data.processor import apply_chat_template
   from vla_foundry.models.vision_language_backbones.simple_vlm_processor import SimpleVLMProcessor
   from vla_foundry.models.vision_language_backbones.vit_backbone import ViTBackboneWrapper
   from vla_foundry.params.model_params import ViTBackboneParams

   proc = SimpleVLMProcessor()
   proc.image_seq_length = 1  # data.img_num_tokens
   backbone = ViTBackboneWrapper(
       ViTBackboneParams(
           img_size=224, patch_size=16, hidden_dim=256, inter_dim=1024, n_heads=4, n_layers=6,
           text_tokens=32, text_vocab_size=49280,
       )
   ).eval()
   names = ("red cube", "green can", "blue hex nut")
   prompts = [apply_chat_template(proc, 2, f"pick up the {name}") for name in names]  # 2 images
   batch = proc(None, prompts, padding=True)
   with torch.no_grad():
       slots = backbone._embed_text(batch["input_ids"], batch["attention_mask"])
   print(tuple(batch["input_ids"].shape), tuple(slots.shape))
   print("filled slots:", (slots.abs().sum(-1) > 0).sum(-1).tolist())
   for other in (1, 2):
       differ = (slots[0] - slots[other]).abs().sum(-1) > 0
       print(f"{names[0]} against {names[other]}: slots", differ.nonzero().flatten().tolist())
   EOF
   ```

2. Compute the reference table from the layout rule. A third of the middle line's closes are at
   the named object; why is its slope below one third?

   ```bash
   .venv/bin/python - <<'EOF'
   import numpy as np
   from sim.pickup.summarize import close_slope

   rng = np.random.default_rng(0)
   names = ["a", "b", "c"]
   episodes = {"the named object": [], "one of the three, at random": [], "a random place": []}
   for _ in range(20000):
       placed = []
       while len(placed) < 3:  # as PickupEnv.reset: uniform in SPAWN_X x SPAWN_Y, at least 11 cm apart
           xy = [rng.uniform(0.38, 0.62), rng.uniform(-0.22, 0.22)]
           if all(np.hypot(xy[0] - p[0], xy[1] - p[1]) > 0.11 for p in placed):
               placed.append(xy)
       objects = dict(zip(names, placed))
       target = names[rng.integers(3)]  # the instruction: drawn independently of the positions
       closes = {
           "the named object": objects[target],
           "one of the three, at random": objects[names[rng.integers(3)]],
           "a random place": [rng.uniform(0.38, 0.62), rng.uniform(-0.22, 0.22)],
       }
       for policy, tcp in closes.items():
           dist = {n: np.hypot(tcp[0] - xy[0], tcp[1] - xy[1]) for n, xy in objects.items()}
           episodes[policy].append(dict(target=target, close_tcp_xy=tcp, close_object_xy=objects, dist=dist))
   for policy, eps in episodes.items():
       at_named = np.mean([min(e["dist"], key=e["dist"].get) == e["target"] for e in eps])
       miss = 1000 * np.median([min(e["dist"].values()) for e in eps])
       slope = close_slope({"per_episode": eps})
       print(f"closes at {policy:28s} at the named object {at_named:4.0%}   slope {slope:5.2f}   miss to nearest {miss:3.0f} mm")
   EOF
   ```

3. `_embed_text` moves the real ids to the front with a stable `argsort`. Change `1 - mask` in that
   line to `0 * mask` (every id then keeps its place), run the command below, and undo the change.
   Which parametrised cases of `test_conditioning_does_not_depend_on_padding` fail, and why does
   `[[8, 9, 0, 0]]` still pass?

   ```bash
   .venv/bin/python -m pytest tests/essential/models/test_vit_backbone_text.py -q
   ```
