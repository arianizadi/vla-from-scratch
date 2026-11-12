---
title: From Language Models to Robot Policies
part: 0 Orientation
order: 02
summary: Name the published idea behind each part of a vision-language-action policy, the problem it solved, and what is still open.
status: draft
---

# From Language Models to Robot Policies

A vision-language-action policy (VLA) is a neural network that takes camera images, a text
instruction and the robot's joint state, and outputs robot actions. Each of its parts was invented
for a different problem: translation, image classification, captioning, steering a van.

## The idea

Three lines of research ran separately and merged around 2023: language models that predict the next
token (a word or a piece of a word) of text, vision models that turn an image into vectors a
language model can read, and robot policies trained to copy demonstrated actions.

One config in this repository contains all three:
[`sim/pickup/configs/vla_qwen.yaml`](../../sim/pickup/configs/vla_qwen.yaml), a policy for a
simulated Franka Panda arm, made of two networks. The backbone is a pretrained vision-language model
(VLM, a language model that also accepts images) that reads the images and the instruction. The
action head is a smaller network that turns the backbone's output, its conditioning, into actions.

| Part | In `vla_qwen.yaml` and the preset files it includes | Line of work |
|---|---|---|
| Backbone | `Qwen/Qwen3-VL-2B-Thinking`; 4 images enter its prompt as 49 tokens each | Transformer (2017), GPT-style pretraining (2018-2020), ViT (2020), VLMs (2022-2024) |
| Readout | the action token, one extra token appended to the prompt; its hidden states (the language model's vectors for it) in the last 4 layers, joined into one vector, condition the action head | VLM as backbone: RT-2 (2023); this readout: VLA Foundry (2026) |
| Action head | a 12-layer, 512-wide transformer in which every position attends to every other (no causal mask), trained with flow matching | Diffusion Policy (2023), π0 (2024) |
| Output | `[batch, 16, 8]`: 16 control steps of 8-D joint and gripper targets, 15 predicted (one chunk); `sim/pickup/eval.py` executes 8 (`--exec_horizon`), then asks again | action chunks (ACT, 2023), receding-horizon execution (Diffusion Policy, 2023) |
| Training data | demonstrations from a scripted expert | supervised image-to-action learning (ALVINN, 1988), behaviour cloning |

<!-- interactive: timeline -->
*Figure: the papers and systems named here on a time axis from 1988 to 2025, in three lanes
(language, vision, robot learning) that join at RT-2 in 2023.*

## How each step led to the next

Background: published work, none of it measured here.

### Sequences and attention (2014-2017)

Until 2017 the usual model for a sequence of tokens was a recurrent network, which reads one token
at a time and carries a state vector forward.

Sutskever 2014 translated with two recurrent networks: one encoded the source sentence into one
fixed-length vector, the other decoded the translation from it. Every sentence had to fit in that
vector.

Bahdanau 2014 removed that limit. For each word it produced, the decoder computed a weighted average
over all encoder states, with weights from a small learned network that scored each encoder state
against the decoder's state, normalised to sum to one. That operation is attention: a lookup over
a set of vectors, driven by their content.

Vaswani 2017 (the Transformer) dropped the recurrence and kept only attention: every token attends
to every other token in one step (in a next-token language model, only to earlier tokens: the causal
mask), so all positions are trained in parallel. That made much larger models practical. See
[the transformer block](11-the-transformer-block.md).

### Pretraining and scale (2018-2020)

Radford 2018 (GPT) trained a Transformer to predict the next token of unlabelled text, then
fine-tuned it (continued training from the pretrained weights) on each task. Brown 2020 (GPT-3)
showed that a large enough model performs new tasks from examples written in its prompt. A
pretrained language model became a reusable part that already knows about objects and instructions.
See
[training a language model](13-training-a-language-model.md).

### Images as tokens (2020-2021)

Dosovitskiy 2020 (the Vision Transformer, ViT) cut an image into 16x16-pixel patches, mapped each
patch to a vector, and fed the sequence to a standard Transformer. An image became a sequence of
tokens, the data type a language model consumes. See [images to tokens](20-images-to-tokens.md).

Radford 2021 (CLIP) trained an image encoder and a text encoder on 400 million image-text pairs to
predict which caption belongs to which image (contrastive pretraining). It gave image features that
line up with language without class labels. SigLIP (Zhai 2023) is a later encoder of the same kind.

CLIP's objective compares one pooled vector per image (the patch vectors reduced to one) with one
per caption, so that vector is trained to say what is in the picture, not where: see
[pooled against patch tokens](21-pooled-vs-patch-tokens.md).

### Image tokens inside a language model (2022-2024)

The open question for a VLM was the connector between a pretrained vision encoder and a pretrained
language model. Below, frozen means the weights are not updated; cross-attention is attention from
text tokens to image tokens.

| Model | Connector | What it showed |
|---|---|---|
| Flamingo (Alayrac 2022) | a resampler that outputs a fixed number of image tokens, plus new cross-attention layers between the layers of a frozen language model | both pretrained models can be reused unchanged |
| BLIP-2 (Li 2023) | a small Querying Transformer, the only trained part | a light bridge between frozen models is enough |
| LLaVA (Liu 2023) | one trainable linear layer from CLIP patch features into the language model's word-embedding space | the simplest connector works |
| PaliGemma (Beyer 2024) | a linear projection of SigLIP patch tokens into the Gemma-2B language model, the LLaVA design | a 3B VLM released to be fine-tuned on other tasks |

See [joining vision and language](30-joining-vision-and-language.md).

### Imitation before large models (1988-2011)

Behaviour cloning, the simplest form of imitation learning (training a policy from demonstrations),
is supervised learning: the input is an observation, the label is the action the expert took.

ALVINN (Pomerleau 1988) was a three-layer network that learned to map camera and laser range finder
images to a steering direction.

Ross and Bagnell showed in 2010 that a cloned policy's errors compound: a small error takes it to a
state the expert never visited, where it has no training data and errs more. Ross 2011 gave the
standard fix, DAgger: run the learned policy, have the expert label the states it reached, add them
to the dataset and retrain. It needs an expert on call during training. See
[imitation learning](43-imitation-learning.md).

### Scale, then a VLM as the backbone (2021-2023)

BC-Z (Jang 2021) cloned more than 100 tasks on a real robot, conditioned on an embedding of a
language instruction or human video, and performed some tasks it had no robot demonstrations for.

RT-1 (Brohan 2022) was a Transformer that takes images and an instruction and outputs action tokens:
each action dimension is rounded into one of 256 bins and the bin index is predicted like a word. It
was trained on over 130k real-robot episodes covering over 700 tasks. One model could absorb many
tasks; collecting the data became the bottleneck.

RT-2 (Brohan 2023) started from a pretrained VLM. It wrote robot actions as text tokens, trained the
VLM on robot trajectories and web-scale vision-language tasks together, and named the category
vision-language-action models. The robot data teaches motion; the web data supplies knowledge of
objects and words.

### Generative action heads (2023)

Demonstrations are multimodal: in the same situation two different actions can both be correct, for
example passing an obstacle on the left or the right. A network trained with a squared-error loss to
output one action learns their average, which is correct for neither.

Diffusion Policy (Chi 2023) generates a short sequence of actions by denoising conditioned on the
observation: starting from random noise, it refines it over several steps, which lets it represent
several modes. It executes the first part of the sequence and re-plans (receding-horizon control).
ACT (Zhao 2023), on low-cost two-arm ALOHA hardware, predicts a chunk of future actions at once with
a generative model (action chunking). A chunk means fewer decisions per episode, so fewer chances
for errors to compound. Flow matching (Lipman 2022) is a simpler way to train such a generator:
regress the velocity that carries noise to data. See [flow matching](41-flow-matching.md) and
[chunks and execution](42-chunks-and-execution.md).

### Shared data and generalist policies (2023-2025)

An embodiment is one robot body with its own cameras and action space. Until 2023 the convention was
a separate model and dataset for each robot. Open X-Embodiment (2023) pooled data from 22 robots
into one dataset in a standard format and reports positive transfer to several of them. Octo and
OpenVLA train on it. The policies that followed combine the earlier steps:

| Policy | Backbone | Action output |
|---|---|---|
| Octo (2024) | transformer trained on robot data | diffusion head |
| OpenVLA (Kim 2024) | Llama 2 with DINOv2 and SigLIP features | 256 token bins per action dimension |
| π0 (Black 2024) | PaliGemma | a flow-matching "action expert", chunks of 50 actions |
| GR00T N1 (Bjorck 2025) | a VLM ("System 2") | a diffusion transformer ("System 1") |
| SmolVLA (Shukor 2025) | SmolVLM-2 | a flow-matching action expert, chunks of 50 actions |

π0 joins the two robot-learning lines: the VLM backbone of RT-2 with the continuous generative head
of Diffusion Policy. The VLA recipes in this repository have that shape.

## In this repository

The training framework is VLA Foundry (Mercat 2026), which this repository forks. Three places
differ from the papers.

**Actions are not text tokens.** RT-2 and OpenVLA emit actions through the language model's
vocabulary. Here `VLMHFBackboneWrapper._prepare_inputs_for_action_token`
([`vlm_hf_backbone.py`](../../vla_foundry/models/vision_language_backbones/vlm_hf_backbone.py))
places the action token after the prompt's last real token, and `conditioning_from_hidden_states`
([`base.py`](../../vla_foundry/models/vision_language_backbones/base.py)) concatenates its hidden
states from the last `num_vlm_layers_to_use` layers into
`[batch, 1, hidden x num_vlm_layers_to_use]`, the conditioning of a separate action transformer.

**The "diffusion policy" is trained with flow matching.** Every policy config in
`sim/pickup/configs` sets `use_flow_matching_scheduler: true`, directly or through a preset, so
`create_noise_scheduler` (`vla_foundry/models/diffusion/__init__.py`) returns
`FlowMatchingScheduler`, and `DiffusionPolicyBatchHandler.prepare_inputs_and_targets`
(`vla_foundry/models/batch_handlers.py`) sets the target to `noise - actions`
([flow matching](41-flow-matching.md)).

**Demonstrations come from a scripted expert.** `_worker` in
[`sim/pickup/collect.py`](../../sim/pickup/collect.py) steps `ScriptedExpert.act()` until the
episode ends and drops failed episodes. No learned policy runs there, so the data has none of the
states DAgger collects. Two options put states off the expert's planned path into the data.
`--random_start` starts each episode from a random gripper pose (`random_start` in
[`sim/pickup/expert.py`](../../sim/pickup/expert.py)). `--disturb_prob 0.5` draws a move of the named
object for half of the episodes, during the reach or as the gripper closes (`Disturbance` in
[`sim/pickup/disturb.py`](../../sim/pickup/disturb.py); in `pick_target_4k_recover` the object was
moved in 1505 of 4000), and makes the expert `ScriptedExpert(reactive=True)`, which plans again
when the object has moved or a grasp closed on nothing. This is a scripted stand-in for DAgger: the
expert, not the learned policy, reached these states. The dataset that [README.md](../../README.md) recommends uses both:

```bash
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k --random_start --disturb_prob 0.5
```

The era is selected by `model.vision_language_backbone.type`, resolved by
`get_vision_language_backbone` in `vla_foundry/models/vision_language_backbones/__init__.py`:

| `type` | Conditioning | Used by (in `sim/pickup/configs/`) |
|---|---|---|
| `clip_backbone` | one pooled vector per image, one for the text | `dp_clip.yaml` |
| `vit_backbone` | every patch token (196 per image), plus 32 instruction-token slots when `text_tokens` is set | `dp_vit.yaml`, `dp_vit_lang.yaml` (4 images); `dp_vit_reactive.yaml`, `dp_vit_lang_reactive.yaml` (2 images) |
| `vlm_backbone` | one vector, or all tokens (`conditioning: all_tokens`), from a pretrained Hugging Face VLM | `vla_qwen.yaml`, `vla_qwen_tokens_reactive.yaml` |
| `vlm_foundry_backbone` | the same choice, from a VLM trained here | `vla_custom.yaml`, `vla_custom_tokens.yaml`, `vla_custom_tokens_reactive.yaml` |

## What we measured

From the [lab notebook](../lab-notebook.md): closed-loop success (the policy's own actions drive the
simulator and it sees the result) on unseen layouts, one training seed per run. Each result tests one
idea from the history above; the chapter linked has the full numbers.

**One vector is a bottleneck here too.** On `pick_cube` with 200 demonstrations on machine A (one RTX
4090, 24 GB), a diffusion policy given one pooled CLIP vector per image scored 4%, one given every
patch token of a small ViT trained from scratch 30%, though the two also differ in size and
pretraining ([pooled against patch tokens](21-pooled-vs-patch-tokens.md)). The pooled vector says what
is in the picture; a grasp needs where.

**No pretrained part is needed for this task.** `dp_vit_lang_reactive`, the same patch-token policy
with the instruction added as tokens ([language conditioning](51-language-conditioning.md)) and no
history, scored 83% / 85% (trained weights / their running average, EMA) on 100 `pick_target`
layouts (three of four known objects on the table, one of them named) after 5.0M samples on 4000
demonstrations, on machine B (L40S GPUs, 128 CPU cores) ([case study](101-case-pick-target.md)).
That run used a position table later found to be wrong ([positions](12-positions.md)), and it
started to look at the scene (where the gripper closed began to depend on where the named object
was) only late in training ([how much data and training](71-how-much-data-and-training.md)).

**Copying the expert invites shortcuts.** Given its previous action as an input as well, the same
policy on `pick_target` never starts to look: it reads the target from that action
([history and shortcuts](52-history-and-shortcuts.md)). In a smooth expert demonstration the
previous action predicts the next one well, so the shortcut is cheap.

**A pretrained VLM as backbone: mainly an earlier start.** This tests RT-2's bet. The first two
VLAs trained here (machine A, 1500 demonstrations) had history, one conditioning vector (the readout
in the first table) and too small a budget, and failed. With every token handed to the action head,
no history and 4000 demonstrations, the from-scratch VLA ends at 86% / 87% after 3M samples with its
VLM pretrained on captions and 75% / 85% with a randomly initialised one, which starts to look later
([pretrained backbones](102-case-pretrained-backbones.md)). With four known objects and one form of
instruction, a language model has little to add.

## Where it goes wrong: what is still hard

**Precision.** What you see: a policy reaches for the right object and closes the gripper beside it.
Each object forgives a different placement error, and the cube and the T block also need the right
grasp angle.
Measure the tolerance per object and per machine before blaming the model. After the position fix
the T block is still the hardest object (87% with clean demonstrations, 92% with random start
poses, last three checkpoints pooled), but the remaining failures are spread thinly over the
objects ([measuring failure](92-measuring-failure.md)).

**Data.** Text and images were already on the internet; robot actions are not: each demonstration
has to be teleoperated (driven by a person) or scripted. Here 1500 demonstrations took 70 s with 48
simulator processes, cheap only because the task can be scripted
([demonstrations](70-demonstrations.md)).

**Evaluation.** Language models are compared by their loss on held-out text; policies cannot be: on
`pick_cube` above, the pooled-vector policy had the lower final loss ([loss is not
success](91-loss-is-not-success.md)). Only closed-loop rollouts count, and they are noisy: 2 successes
in 50 has a Wilson 95% interval (a confidence interval for a proportion) of 1-14%. A rate near zero
does not say whether a policy is imprecise or not looking; the slope `sim/pickup/summarize.py`
prints, of where the gripper closed against where the named object was, does: near 1 for a policy
that is imprecise, near 0 for one that is not looking ([measuring failure](92-measuring-failure.md)).

**Recovery.** What you see: a policy trained on clean demonstrations that misses does not try again,
and does not follow an object moved during the reach
([what a policy looks at](95-what-a-policy-looks-at.md)). Why: Ross 2010; its data holds only the
expert's own path. How to check: `eval.py --disturb reach` or `--disturb grasp`. What to do: the two
`collect.py` options above; a policy trained on both copes with every disturbance tried
([self-correction](103-case-self-correction.md)). Noise-injected demonstrations are on the
[roadmap](../roadmap.md).

## Background and further reading

Language
- Sutskever, [Sequence to Sequence Learning](https://arxiv.org/abs/1409.3215), 2014: one vector between encoder and decoder.
- Bahdanau, [Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473), 2014: attention over all encoder states.
- Vaswani, [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017: the Transformer.
- Radford, [Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), 2018: next-token pretraining, then fine-tuning.
- Brown, [GPT-3](https://arxiv.org/abs/2005.14165), 2020: tasks specified in the prompt.

Vision
- Dosovitskiy, [ViT](https://arxiv.org/abs/2010.11929), 2020: image patches as Transformer tokens.
- Radford, [CLIP](https://arxiv.org/abs/2103.00020), 2021: contrastive image-text pretraining.
- Zhai, [SigLIP](https://arxiv.org/abs/2303.15343), 2023: the same with a per-pair loss.
- Alayrac, [Flamingo](https://arxiv.org/abs/2204.14198), 2022: frozen models bridged by cross-attention.
- Li, [BLIP-2](https://arxiv.org/abs/2301.12597), 2023: a Querying Transformer between frozen models.
- Liu, [LLaVA](https://arxiv.org/abs/2304.08485), 2023: a linear projector into the language model.
- Beyer, [PaliGemma](https://arxiv.org/abs/2407.07726), 2024: a 3B VLM built for transfer.

Robot learning
- Pomerleau, [ALVINN](https://proceedings.neurips.cc/paper/1988/hash/812b4ba287f5ee0bc9d43bbf5bbe87fb-Abstract.html), 1988: supervised learning of steering from images.
- Ross, [Efficient Reductions for Imitation Learning](https://proceedings.mlr.press/v9/ross10a.html), 2010: a cloned policy's regret grows quadratically with the horizon.
- Ross, [DAgger](https://arxiv.org/abs/1011.0686), 2011: the expert labels states the policy reaches.
- Jang, [BC-Z](https://arxiv.org/abs/2202.02005), CoRL 2021: language-conditioned cloning of 100+ tasks.
- Brohan, [RT-1](https://arxiv.org/abs/2212.06817), 2022: a Transformer policy on 130k real-robot episodes.
- Brohan, [RT-2](https://arxiv.org/abs/2307.15818), 2023: a VLM that outputs actions as tokens.
- Lipman, [Flow Matching](https://arxiv.org/abs/2210.02747), 2022: a generator trained by regressing a velocity field.
- Chi, [Diffusion Policy](https://arxiv.org/abs/2303.04137), 2023: action sequences by denoising, receding-horizon execution.
- Zhao, [ACT / ALOHA](https://arxiv.org/abs/2304.13705), 2023: action chunking on low-cost bimanual hardware.
- Open X-Embodiment Collaboration, [Open X-Embodiment](https://arxiv.org/abs/2310.08864), 2023: pooled data from 22 robots.
- Octo Model Team, [Octo](https://arxiv.org/abs/2405.12213), 2024: an open generalist policy, diffusion head.
- Kim, [OpenVLA](https://arxiv.org/abs/2406.09246), 2024: an open 7B VLA with tokenised actions.
- Black, [π0](https://arxiv.org/abs/2410.24164), 2024: a VLM with a flow-matching action expert.
- Bjorck, [GR00T N1](https://arxiv.org/abs/2503.14734), 2025: a dual-system VLA for humanoids.
- Shukor, [SmolVLA](https://arxiv.org/abs/2506.01844), 2025: a compact VLA for consumer hardware.
- Mercat, [VLA Foundry](https://arxiv.org/abs/2604.19728), 2026: the framework this repository forks.

## Check yourself

1. Print the Qwen VLA config, its model preset and that preset's transformer preset. For
   `vision_language_backbone.type`, `hf_pretrained`, `transformer`, `is_causal: false` and
   `use_flow_matching_scheduler`, name the reading-list paper behind the idea, and say which preset
   value `vla_qwen.yaml` overrides.

   ```bash
   cat sim/pickup/configs/vla_qwen.yaml vla_foundry/config_presets/models/vla_diffusion_paligemma2.yaml \
     vla_foundry/config_presets/models/transformer_100m.yaml
   ```

2. RT-2 writes actions as text tokens. From the two functions below, state what this repository
   does with the action token instead, and the shape of the result.

   ```bash
   grep -n -A17 "def insert_after_last_real_token\|def conditioning_from_hidden_states" \
     vla_foundry/models/vision_language_backbones/base.py
   ```

3. With the environment from [setup](01-setup-and-first-run.md), run the scripted expert on ten
   unseen layouts. State which of the four open problems above it avoids.

   ```bash
   .venv/bin/python -m sim.pickup.eval --task pick_target --episodes 10 --expert --out results/pickup/expert_check
   ```
