---
title: Training a VLM
part: 3 Vision-language models
order: 31
summary: Build caption data from a scene you control, run stage 2 on it, and say which tokens the loss scores, which weights were pretrained and what the captions can teach.
status: draft
---

# Training a VLM

A vision-language model (VLM) is trained like a language model, by predicting the next token,
except that part of its input sequence is an image, and only text positions are scored. Here that is stage 2 of the from-scratch path:
the stage-1 language model gains a vision transformer (ViT) and a projector and learns to caption
renders of the pick-up scene. Stage 3 turns the result into a robot policy, so what the captions
teach is what the policy's backbone knows about the scene before it sees an action.

## The idea

The model of [Joining vision and language](30-joining-vision-and-language.md) turns an image into
256 patch tokens and puts them in reserved positions of the token sequence. Training it needs
pairs of an image and a **caption**, a text that describes the image. The sequence is a short
prompt, the image positions, then the caption, and the loss is the next-token cross-entropy of
[Training a language model](13-training-a-language-model.md), taken on the text targets only: the
caption and four fixed template tokens.

One generated caption:

```text
robot arm over a table. from left to right: a red cube, a green can and a blue hex nut.
```

Most of these words follow from the words before them; a language model with no image can learn
`from left to right: a`. The next object cannot be read from the text: three of four objects are on
the table, ordered by where they are in the picture. Every scored token after the image sends a
gradient through the language model and the projector into the ViT, but for the template words
the loss falls without the image. For this token it falls only if the patch tokens carry which
object is where. The dataset decides which tokens have that property, and so what the vision side
is trained to see.

<!-- interactive: sequence-layout -->
*Figure: the stage-2 sequence as a row of 321 cells, from prompt to padding, with the loss
targets highlighted.*

## How it works

### One batch

With `sim/pickup/configs/custom_vlm.yaml` (batch 32, `seq_len: 320`, `img_num_tokens: 256`):

| Tensor | Shape | Content |
|---|---|---|
| `pixel_values` | `[32, 3, 224, 224]` | images scaled to [0, 1], then to [-1, 1] |
| ids from the data loader | `[32, 321]` | each row padded on the right or truncated on the left to `seq_len + 1` |
| inputs, targets | `[32, 320]` each | ids 0..319 and ids 1..320 |
| ViT output, then projector | `[32, 256, 256]`, `[32, 256, 512]` | (224 / 14)² = 256 patch tokens, 256 wide, then the language model's width |
| logits | `[32, 320, 49280]` | one score per vocabulary entry, per position |

One row of ids, wrapped by the tokenizer's chat template
([Text becomes tokens](10-tokens.md)); `c` is the caption's token count:

| Positions | Content | Counts as a target |
|---|---|---|
| 0 | `<\|im_start\|>` | no, targets start at position 1 |
| 1-2 | `User`, `:` | yes |
| 3-258 | 256 `<image>` placeholders | no, masked |
| 259 onward | caption, c tokens | yes |
| next two | `<end_of_utterance>`, newline | yes |
| the rest | padding, 60 - c tokens | no, masked |

The row scores c + 4 targets out of 320.

### The masked loss

With positions numbered from 0 as above, let $S$ be the set of (row, position) pairs $(b, t)$
whose target, id $t+1$ of row $b$, is neither the padding id nor the image-placeholder id. The
loss is

$$L = -\frac{1}{|S|}\sum_{(b,t)\in S} \log p_\theta\left(x_{b,t+1} \mid x_{b,0..t}\right)$$

For $t \ge 3$ the context $x_{b,0..t}$ includes image positions, whose embeddings are the projected
patch tokens of image $b$. Placeholders are masked because the model is not asked to generate an
image; padding because it is not text.

### Where the weights come from, and what to freeze

A **checkpoint** is the training state saved at one point: weights, optimiser state and position
in the data. Stage 2 takes only the weights of the last stage-1 checkpoint, and only into the
language model (91.4M parameters). The ViT (4,955,392 parameters) and the projector (one linear
layer 256 -> 512 without bias, 131,072) start random; those two counts follow from the
config's layer sizes, and the three sum to the notebook's 96.5M. At optimiser step 0 the
language model therefore receives three prompt tokens, then 256 vectors that are a random function
of the image, then text: the image information is there, in no form it can use yet.

To **freeze** a part is to exclude its weights from the update. Background: early on, the
gradients of a random part can overwrite what a pretrained neighbour knows; freezing that
neighbour for a while, or warming up its learning rate slowly, avoids it.

| Starting point | Choice | Reason |
|---|---|---|
| Pretrained encoder and language model. Background: LLaVA (Liu, 2023) | vision encoder frozen throughout; train only the projector, then the projector and the language model | the projector becomes a visual tokenizer for an already good language model |
| The same. Background: PaliGemma (Beyer, 2024) | nothing frozen, the image encoder's learning rate warmed up slowly; in its ablation, freezing the language model was worse, and freezing the image encoder worsened perplexity on spatial tasks | a captioning loss can teach the encoder spatial skills |
| Stage 2 here: random ViT and projector, a language model trained for about 15 minutes | nothing frozen | a frozen random ViT cannot learn to see, and the language model must learn a new kind of text |

Freezing the language model was not run here; exercise 2 shows the switch.

### Captions from a scene you control

A simulator knows the 3-D position of every object, so a caption that says where things are can
be computed. Background: CLEVR (Johnson, 2016) generated its questions the same way.
[`sim/pickup/caption_data.py`](../../sim/pickup/caption_data.py) does it in three steps.

1. **Make a scene.** For each seed from 2,000,000 the script picks `pick_target` with probability
   0.8, otherwise `pick_cube` (one red cube), resets the environment, and with probability 0.7
   advances the scripted expert by 0 to 69 control steps, so the arm appears in poses that occur
   during a pick. Both cameras are rendered: two image-caption pairs per scene.
2. **Project each object into the camera.** `project` subtracts the camera's position from the
   object's and rotates the result into the camera frame: $x$ right, $y$ up, the camera looking
   along $-z$ (its **optical axis**), with depth $d$ the distance along that axis. Then
   $u = f x / d$ and $v = -f y / d$, where `fovy` is the camera's vertical field of view and
   $f = 1/\tan(\text{fovy}/2)$ is the focal length in units of half the image height. The minus
   sign makes $v$ grow downward, like image rows. Both lie in [-1, 1] when the point is in view
   ($u$ because the 224 x 224 render is square). An object is named only if its centre has
   $d > 0$, $|u| < 0.92$ and $|v| < 0.92$.
3. **Fill a template.** The named objects are sorted by $u$, left to right, and one of four
   sentence forms is drawn.

Worked example, made-up coordinates. The `front` camera has `fovy=45`, so
$f = 1/\tan 22.5° = 2.414$. A cube 0.10 m right of the optical axis, 0.05 m below it and 1.0 m
away projects to $u = 0.241$, $v = 0.121$: column $(u+1)/2 \times 224 = 139$, row 126. The
thresholds at ±0.33 cut the image into bands at columns 75 and 149, so the cube is in the middle
band both ways: the form-0 example below.

| Form | Example, after the prefix `robot arm over a table` (front camera) or `view from the robot gripper` (wrist) | What the image must supply |
|---|---|---|
| 0, or one object in view | `the red cube is in the middle of the image, near the centre.` | which object, its cell in a 3 x 3 grid |
| 1 | `from left to right: a red cube, a green can and a blue hex nut.` | every object and the full horizontal order |
| 2 | `with a blue hex nut, a red cube and a green can. the red cube is leftmost.` | which objects are present (shuffled list), which is leftmost |
| 3 | `the red cube is to the left of the green can.` or the mirrored `right of` sentence | the horizontal order of one pair |

With nothing in view the caption ends `no objects are in view.`

### What these captions can and cannot teach

They can teach the object names used in the task instructions, tied to image regions, and
left-to-right order.

They cannot teach what they never state: depth, distance in metres, orientation, where the
gripper is relative to an object, or which object an instruction means (a policy learns that from
demonstrations, [Language conditioning](51-language-conditioning.md)). Position is given to a
third of the image, about 75 pixels. By arithmetic on the front camera's pose in
`sim/pickup/env.py` (1.05 m from the point it looks at), the image spans about 0.87 m
at that distance and one band about 0.29 m: about thirty times the 10 mm grasp-point error that
cuts the can's success from 94% to 60% on machine A (one RTX 4090, 24 GB; the notebook's
grasp-tolerance table).

Form 2 shows why wording matters. If every caption were a list in arbitrary order, like its
first sentence, the model could reach a low loss knowing only which objects are present, which one
pooled vector can carry ([One vector or many tokens](21-pooled-vs-patch-tokens.md)).

## In this repository

**Data.** `work` in `sim/pickup/caption_data.py` writes tar shards (a shard is one file of a
dataset split into many) holding `<key>.jpg` and `<key>.txt`, with the key
`<seed, 8 digits>_<camera>` and up to 1,024 pairs per shard, listed in `manifest.jsonl`.
`ImageCaptionPipeline.create_pipeline`
([`image_caption.py`](../../vla_foundry/data/pipelines/image_caption.py)) decodes each image to a
uint8 tensor, wraps the text with `apply_chat_template`, batches, and calls `SimpleVLMProcessor`
([`simple_vlm_processor.py`](../../vla_foundry/models/vision_language_backbones/simple_vlm_processor.py))
with `max_length=seq_len + 1`; it scales the images, expands `<image>` into 256 placeholders and
tokenises with `truncation=True`. The tokenizer pads on the right and truncates on the left
(`truncation_side: left` in its `tokenizer_config.json`): an over-long row loses its first ids.

**Targets and mask.** `VLMBatchHandler.prepare_inputs_and_targets`
([`vla_foundry/models/batch_handlers.py`](../../vla_foundry/models/batch_handlers.py)) calls
`sample_chunk` (`vla_foundry/data/sampler.py`) to split each row into inputs and targets, then
builds `mask = (targets == cfg.data.pad_token_id) | (targets == cfg.data.image_token_id)`.
`CrossEntropyLossWithZLoss.forward` ([`vla_foundry/losses.py`](../../vla_foundry/losses.py))
replaces masked targets with `ignore_index` (-100) before the mean.

**Weights.** If `model.transformer.resume_from_checkpoint` is set, `create_vlm`
([`vla_foundry/models/vlm.py`](../../vla_foundry/models/vlm.py)) loads the `state_dict` (the
weights) of that file into the transformer with `strict=True`, so stage 1 and `model.transformer`
must describe the same architecture. `ModelParams.freeze` (`vla_foundry/params/model_params.py`)
defaults to `False`; when set, `BaseModel._post_init` (`vla_foundry/models/base_model.py`) turns
off `requires_grad` and `create_optimizer` (`vla_foundry/optimizer.py`) skips those parameters.

**Running it.**

```bash
sim/pickup/scripts/custom_path.sh data    # text, caption scenes, held-out scenes
sim/pickup/scripts/custom_path.sh llm     # stage 1
sim/pickup/scripts/custom_path.sh vlm     # stage 2
sim/pickup/scripts/custom_path.sh check   # held-out captioning check
```

The `vlm` stage of [`custom_path.sh`](../../sim/pickup/scripts/custom_path.sh) passes the
highest-numbered weights file, `checkpoint_N.pt` in `experiments/pickup/custom_llm/checkpoints`, to
`vla_foundry/main.py` as
`--model.transformer.resume_from_checkpoint`. Checkpoints go to
`experiments/pickup/custom_vlm/checkpoints`. Budgets: `PICKUP_VLM_SAMPLES` (default 400000) and
`PICKUP_CAPTION_SCENES` (default 20000), which the `data` stage ignores once
`data/custom/sim_captions/manifest.jsonl` exists: delete that directory to regenerate.

In [`custom_vlm.yaml`](../../sim/pickup/configs/custom_vlm.yaml),
`data.allow_multiple_epochs: true` permits the 10 passes that 400,000 samples make over 40,000
pairs, and `data.img_num_tokens` must equal the ViT's 256 patch tokens.

## What we measured

The 96.5M model after 400k samples on 40,000 captioned sim renders,
checked on 300 held-out scenes (seeds the training set never used).

| | Final loss | Objects listed in left-to-right order | Leftmost object named |
|---|---|---|---|
| Machine A, about 15 min at 437 samples/s | 0.058 | 256/300 = 85% | 285/300 = 95% |
| Machine B (L40S GPUs, 128 CPU cores) | 0.058 | 249/300 = 83% | 279/300 = 93% |
| Machine B, after the position fix | 0.059 | 267/300 = 89% | 286/300 = 95% |

A loss of 0.058, a geometric-mean probability of $e^{-0.058} = 0.94$ for the correct token,
includes every template word, so it does not show that the image is used. The check does:
`sim/pickup/vlm_check.py` gives the model a held-out image and a form-1 caption cut after the
colon, and scores the completion ([Testing a VLM](32-testing-a-vlm.md)). A model that knew
which objects were present but not where would order three listed objects correctly 1 time in 6, and two 1 time in 2; 83% is above either.

Before a pixel-range bug was fixed the check gave 1/300 and 70/300 on the two scores: the
processor scaled inference images to [0, 1] but left the uint8 tensors of training at 0-255
([train/inference skew](93-train-inference-skew.md)).

**Positions.** The first two rows were trained before a second fix. Stage 2 runs in mixed
precision (`hparams.precision: amp_bf16`), and the table of rotary position angles was computed in
bfloat16, which keeps every whole number up to 256 but only the even ones up to 512 (641 distinct
positions out of 2048; [Telling tokens where they are](12-positions.md) has the mechanism). Stage 1's 256-token sequences were not affected; the VLM's 320 were, and in
the worst place: every caption token sits above position 258. By arithmetic, the first three
caption tokens (positions 259 to 261) all got the angles of position 260, and the example's 24
caption tokens stood on 12 positions. Trained again with the table fixed, the same VLM scores 89%
and 95% against 83% and 93% (machine B, one seed each: a few points may be noise). What the old
table cost a policy built on the VLM was not measured on its own.

**The policy.** The captioning scores alone do not show that a policy benefits. In the first
round (machine A) the stage-3 policy built on this VLM scored 0/100 on `pick_target`, where the
instruction names one of three objects, and closed at the named object (the object nearest the
gripper when it first closed was the named one) about as often as the same policy
on a randomly initialised VLM: 33% and 29%, where chance is 33%
([Measuring how a policy fails](92-measuring-failure.md)). That run had three handicaps, none of
them the captions: one conditioning vector reached the action head
([What the action head sees](50-what-the-action-head-sees.md)), the policy had history (the
previous frame, state and action as inputs; [History and shortcuts](52-history-and-shortcuts.md)),
and it had 1M samples on 1500 demonstrations, too few even for a patch-token policy with no VLM
([How much data and training](71-how-much-data-and-training.md)). With all three changed (every
token passed to the action head, no history, 4000 demonstrations, 3M samples; machine B, after the
fix), the policy on this VLM starts to look (closes at the named object far above chance) after
0.25M samples (81% / 97%, trained / averaged weights), the one on a randomly initialised VLM only
after about 1.0M (84% / 68%; 33% / 28% at 0.25M): what the captions buy is mainly an earlier start
([A backbone from scratch against a pretrained one](102-case-pretrained-backbones.md)).

That fits what the captions can teach (an interpretation, not isolated by a run): they tie names
to bands about 0.29 m wide, so the pretrained backbone tells the objects apart from the start, and the
factor of thirty down to a 10 mm grasp has to come from the demonstrations either way.

## Where it goes wrong

| What you see | Why | How to check, what to do |
|---|---|---|
| `AssertionError` in `VLM.forward` on the first batch | the start-up check, `TrainExperimentParams.check_asserts`, only requires `seq_len >= img_num_tokens`. A row is 3 + 256 + c + 2 ids and truncation is from the left: the three prompt tokens go first, without an error, then placeholders | keep `seq_len + 1 >= 3 + img_num_tokens + c + 2`; exercise 3, run on the longest caption, must print `<\|im_start\|>` |
| Low loss, wrong objects in the check | the scored tokens were predictable without position or without the image; or the check feeds the model other tensors than training did, as with the pixel range | compare the ordering score with chance and the loss with a mismatched image ([Testing a VLM](32-testing-a-vlm.md)); add forms that state position |
| Nothing: the loss falls and the check passes | until the position table was fixed, tokens above position 256 shared angles ("What we measured"); training and the check built the same table | `RotaryEmbedding._update_cos_sin_tables` in [`vla_foundry/positional_embedding.py`](../../vla_foundry/positional_embedding.py) now computes the angles in float32. A VLM trained before that was fitted to the old table: train it again, or run it with `positional_embedding_type: rotary_legacy_autocast` under `transformer` in its `config_model.yaml` |
| A caption can name an object the image does not show | `caption` tests only the projected centre: an object behind the arm is named, and one centred past 0.92 is not, though part of it shows | compare a caption (`tar -xOf data/custom/sim_captions/shard-000000.tar 02000000_front.txt`) with the `.jpg` of the same key; accept the label noise, or test visibility with a segmentation render (an object id per pixel) |

## Background and further reading

- Liu, *Visual Instruction Tuning* (LLaVA), 2023. <https://arxiv.org/abs/2304.08485>. A linear
  projector between a frozen vision encoder and a language model, trained in two stages.
- Beyer, *PaliGemma: A versatile 3B VLM for transfer*, 2024. <https://arxiv.org/abs/2407.07726>.
  Trains with nothing frozen and reports an ablation of which parts to freeze.
- Johnson, *CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual
  Reasoning*, 2016. <https://arxiv.org/abs/1612.06890>. Rendered scenes with questions generated
  from the renderer's ground truth.

## Check yourself

1. With `--data.seq_len 256` the start-up check passes, and left truncation keeps the last 257
   ids of a row: 231 placeholders for the caption in The idea (c = 24). Predict the error, then
   run:

   ```bash
   .venv/bin/python vla_foundry/main.py --config_path sim/pickup/configs/custom_vlm.yaml \
     --name seq_len_test --total_train_samples 3200 --num_checkpoints 1 --data.seq_len 256
   ```

2. Run the same command with `--name freeze_test --model.transformer.freeze true` in place of
   `--name seq_len_test --data.seq_len 256` and read the `Trainable parameters:` line that
   `vla_foundry/main.py` prints. It should equal the ViT plus the projector.

3. Print one processed row: its first token, then the counts of ids, image ids and padding ids.
   The number of scored targets is 320 minus the last two; compare it with c + 4.

   ```bash
   .venv/bin/python -c "
   from vla_foundry.data.processor import apply_chat_template
   from vla_foundry.models.vision_language_backbones.simple_vlm_processor import SimpleVLMProcessor
   proc = SimpleVLMProcessor(); proc.image_seq_length = 256
   text = apply_chat_template(proc, 1, 'robot arm over a table. no objects are in view.')
   ids = proc(None, [text], padding='max_length', max_length=321)['input_ids'][0]
   print(proc.tokenizer.convert_ids_to_tokens(ids[:1].tolist()), len(ids),
         int((ids == proc.image_token_id).sum()), int((ids == proc.tokenizer.pad_token_id).sum()))"
   ```
