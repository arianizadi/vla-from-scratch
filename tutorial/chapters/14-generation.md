---
title: Generating Text
part: 1 Language models
order: 14
summary: Turn logits into text with greedy decoding or sampling, explain what the key-value cache saves, and spot a prompt, padding or position-table mismatch in this repository's generate methods.
status: draft
---

# Generating Text

A language model's forward pass returns a row of scores for every position, and each row scores
every token in the vocabulary as the next token. Generation is the loop that turns those scores into
text: pick a token, append it, run the model again. One pass of the loop is a **generation step**.
The policies in this tutorial do not generate text when they act (the action head reads the
backbone's hidden states, [60-anatomy-of-a-vla.md](60-anatomy-of-a-vla.md)), but generation is how
the vision-language model under a policy is tested: `sim/pickup/vlm_check.py` asks the stage-2 model
to finish a caption of a scene it never saw and scores the answer.

## The idea

**Logits** are the unnormalised scores that the final linear layer produces. For the stage-1
language model the output has shape `[batch, tokens, 49280]`, one score per vocabulary entry at each
position. A model that predicts each token from the tokens before it is called **autoregressive**,
and for such a model only the last row matters when generating: `[batch, 49280]`, the scores for the
token that comes next. A softmax turns that row into probabilities.

Two choices decide which token is taken: how the row of probabilities is reshaped, and how a token
is picked from it. The example uses a made-up vocabulary of four tokens with logits 2.0, 1.0, 0.0
and -1.0. The first table row is the plain softmax:
$e^{2} / (e^{2} + e^{1} + e^{0} + e^{-1}) = 7.389 / 11.475 = 0.644$ for ` red`.

| Distribution the token is drawn from | ` red` | ` green` | ` blue` | ` yellow` |
|---|---|---|---|---|
| Softmax, temperature 1.0 | 0.644 | 0.237 | 0.087 | 0.032 |
| Temperature 0.5 | 0.865 | 0.117 | 0.016 | 0.002 |
| Temperature 2.0 | 0.455 | 0.276 | 0.167 | 0.102 |
| Temperature 1.0, top-k with k = 2 | 0.731 | 0.269 | 0 | 0 |
| Temperature 1.0, top-p with p = 0.9 | 0.665 | 0.245 | 0.090 | 0 |

- **Greedy decoding** takes the highest-scoring token, here ` red`, every time. The same prompt
  always gives the same text.
- **Sampling** draws a token at random, each with its probability. From the first row, ` red` is
  drawn 64.4% of the time and ` yellow` 3.2%. The three settings below reshape the row before the
  draw; none of them changes which token is the most probable, so none of them changes greedy
  decoding. Elsewhere in this tutorial a *sample* is a training example, so a token picked at random
  is called a draw here.
- **Temperature** divides the logits $z$ by a number $T$ before the softmax:
  $p = \mathrm{softmax}(z / T)$. Below 1 the distribution sharpens toward the top token; above 1 it
  flattens. As $T$ approaches 0, sampling becomes greedy decoding.
- **Top-k** keeps the $k$ highest-scoring tokens, discards the rest and renormalises.
- **Top-p**, also called nucleus sampling, sorts the tokens by probability and keeps the shortest
  prefix whose cumulative probability exceeds $p$. In the example 0.644 + 0.237 = 0.881 does not
  exceed 0.9, so the third token is kept and only ` yellow` is dropped. Top-k keeps a fixed number
  of tokens; top-p keeps few when the model is confident, many when it is not.

A setting matters beyond its one row because a drawn token becomes part of the input of every later
generation step: one unlikely choice changes the context for the rest of the text.

Background (Holtzman et al., cited below): the two extremes fail in opposite ways. Always taking the
most probable continuation gives bland, repetitive text. Sampling from the full distribution gives
text that loses coherence: with 49,280 vocabulary entries, thousands of tokens with a tiny
probability each add up to a real chance of drawing one of them, and top-k and top-p exist to cut
that unreliable tail off. Lower temperature, smaller $k$ and smaller $p$ move toward the first
failure; higher values toward the second. A check with one right answer, such as the captioning
check below, uses greedy decoding because it is repeatable.

## How it works

Two parts of the loop need explaining first: what is reused between generation steps, and when the
loop ends.

### The key-value cache

Attention at one position compares that position's query vector with the key vector of every
position it may attend to, and mixes their value vectors
([11-the-transformer-block.md](11-the-transformer-block.md)). Under a causal mask those are itself
and earlier positions, so appending a token changes nothing about the keys, values or outputs of the
tokens already there.

<!-- interactive: attention -->
*Figure: an attention matrix. With the causal mask on, every row uses only the columns at or before
its own position, so a new last row leaves all earlier rows unchanged.*

The **key-value cache** (KV cache) stores, for every layer, the keys and values of all tokens
processed so far. Each generation step then runs only the new token through the network: compute its
query, key and value, append the key and value to the store, and attend with one query row over all
stored keys.

Take the captioning check's front-camera prompt, 270 tokens (`<|im_start|>User:`,
256 image placeholders and the caption prefix), and 28 new tokens, the limit `vlm_check.py` uses.

| | Positions run through the 12 blocks |
|---|---|
| No cache: generation step i = 0..27 reprocesses all 270 + i tokens | 28 x 270 + (28 x 27) / 2 = 7,938 |
| Cache: the prompt once, then one token per generation step | 270 + 27 = 297 |

In general $n$ new tokens after a prompt of $P$ cost $nP + n(n-1)/2$ positions without the cache:
the prompt again at every generation step, plus a term that grows with the square of $n$. With the
cache the count is $P + n - 1$, linear, though each position still compares its query with every key
before it.

The price is memory: 12 layers x 2 (key and value) x 512 = 12,288 stored numbers per token for the
stage-1 model.

### Stopping

The loop ends when the model emits its **end-of-sequence token** (EOS, a special token that marks
the end of a text) or when a fixed number of new tokens has been produced. The first only works if
the training text contained that token where an answer ends.

## In this repository

| Method | File | Decoding | Cache |
|---|---|---|---|
| `Transformer.generate` | [`vla_foundry/models/transformer.py`](../../vla_foundry/models/transformer.py) | temperature, top-k, top-p; greedy when `temperature=0` | yes |
| `VLM.generate` | [`vla_foundry/models/vlm.py`](../../vla_foundry/models/vlm.py) | the same | yes; the image is encoded once |
| `VLMHF.generate` | [`vla_foundry/models/vlm_hf.py`](../../vla_foundry/models/vlm_hf.py) | greedy only (`torch.argmax`), no EOS stop | no; a full forward pass per token |

`Transformer.generate` and `VLM.generate` share their defaults: `max_new_tokens=20`,
`temperature=1.0`, `top_p=0.9`, `top_k=50`, `eos_token_id=None`, `use_cache=True`. A call with no
options therefore draws at random and never stops early. Both return the prompt and the new tokens
in one tensor; callers slice the prompt off.

Each generation step in both methods does the following, in this order:

1. Take `outputs.logits[:, -1, :]`, the scores at the last position.
2. If `temperature != 0`, divide the logits by it.
3. If `top_k > 0`, set every logit below the k-th largest to `-inf`.
4. If `top_p < 1.0`, sort, take the cumulative softmax and set to `-inf` every token after the one
   that crosses `top_p`.
5. If `temperature == 0`, take `torch.argmax`; otherwise `torch.multinomial` over the softmax.
6. Append the token to `generated` and a 1 to the attention mask.
7. Stop if `eos_token_id` is set and every row of the batch produced it at this step.

There are two ways to get greedy decoding. `temperature=0` reaches the `argmax` branch; the Jupyter
notebook
[`tutorials/llm_vlm_inference_tutorial.ipynb`](../../tutorials/llm_vlm_inference_tutorial.ipynb)
uses it. `top_k=1`, which `vlm_check.py` uses, keeps every logit that is not below the largest one;
when the largest is unique, the multinomial draw has one possible outcome. On an exact tie the two
differ: item 3 compares with a strict `<`, so tied tokens all survive and `torch.multinomial` picks
one of them at random, while `torch.argmax` returns one index without drawing. Ties are possible in
the check: it runs under `torch.autocast("cuda", dtype=torch.bfloat16)` and `custom_vlm.yaml` does
not set `cast_output_to_float32`, so its logits are bfloat16, a 16-bit float format with 8
significant bits (12.30 and 12.31 are the same bfloat16 value). Whether a tie occurs in the
300-scene check was not measured here.

**The cache.** `CustomAttn.forward` in `transformer.py` receives `past_key_value`, a `[keys, vals]`
pair. It projects the new tokens, applies the position encoding with `offset=past_length` so the new
token is rotated by its absolute index ([12-positions.md](12-positions.md)), concatenates the
stored keys (already rotated) and values in front with `torch.cat`, and returns the longer pair when
`use_cache` is true. `Transformer.forward` returns one pair per layer in
`TransformerOutput.past_key_values`. Once a cache exists, `Transformer.generate` feeds only
`generated[:, -1:]`.

**Images.** `VLM.generate` makes its first call through `VLM.forward`, which runs the image encoder
and writes the image embeddings into the `<image>` placeholder positions
([30-joining-vision-and-language.md](30-joining-vision-and-language.md)). Later generation steps
call `self.transformer` with the one new token and the cache, so the image encoder runs once per
prompt; with `use_cache=False` it runs at every step.

**The attention mask.** `Transformer.forward` discards `attention_mask` when the model is causal and
`use_cache` is false: in training, padding sits to the right and the causal mask already hides it
from real tokens. With `use_cache=True` the mask is kept, and `torch_attn` in
[`vla_foundry/attention.py`](../../vla_foundry/attention.py) combines it with a causal mask built by
`get_rectangular_causal_mask`. `apply_attention_mask_` applies it to `bias[..., :mask_length]`, so
entry i of the mask blocks key i. Item 6 appends a 1 per new token to keep the mask as long as the
key sequence (cached keys plus the new one); a mask that is too short would raise no error and leave
the last keys unmasked.

**The captioning check.** [`sim/pickup/vlm_check.py`](../../sim/pickup/vlm_check.py) reads held-out
captions that contain `from left to right:`, and for each one:

```python
prompt = apply_chat_template(proc, 1, text).split(MARK)[0] + MARK
inputs = proc([image], [prompt], padding=False)
out = model.generate(..., max_new_tokens=28, top_k=1, eos_token_id=proc.tokenizer.eos_token_id)
pred = proc.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
pred = pred.split(".")[0].strip()
```

`apply_chat_template` in `vla_foundry/data/processor/__init__.py` is the function the training
pipeline (`ImageCaptionPipeline.create_pipeline`) applies to every caption. `SimpleVLMProcessor` has
no chat template of its own, so the function falls through to the tokenizer's. Background: the
`HuggingFaceTB/SmolVLM2-256M-Video-Instruct` tokenizer on the Hugging Face Hub carries a template
and names `<end_of_utterance>` as its EOS token
([tokenizer_config.json](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct/blob/main/tokenizer_config.json)).
With it, a training caption and the check's prompt are:

```text
<|im_start|>User:<image>robot arm over a table. from left to right: a red cube, a green can and a blue hex nut.<end_of_utterance>
<|im_start|>User:<image>robot arm over a table. from left to right:
```

The template also appends a newline after `<end_of_utterance>`, which the block cannot show; a
tokenised training caption ends with the ids for `.`, `<end_of_utterance>`, newline, then padding.
The processor expands `<image>` into 256 placeholder tokens (`proc.image_seq_length = 256`, matching
`img_num_tokens: 256` in `sim/pickup/configs/custom_vlm.yaml`). The prediction is compared as a
string with the true list; a second score compares only the first object.

Run it with:

```bash
sim/pickup/scripts/custom_path.sh check
```

It needs a GPU, a trained `experiments/pickup/custom_vlm` and the held-out captions that
`custom_path.sh data` writes to `data/custom/sim_captions_heldout`. Decoding settings are arguments of
`generate`, not YAML keys. `vla_foundry/inference/inference_vlm.py` exposes two on the command line,
`--temperature` (default 0.7) and `--max-new-tokens` (default 100), but on the from-scratch VLM it
generates from a padded prompt (see "A padded prompt" below).

## What we measured

From the [lab notebook](../lab-notebook.md), `vlm_check.py` on 300 held-out scenes:

| Stage-2 VLM trained on | Objects in left-to-right order | Leftmost object |
|---|---|---|
| machine A (one RTX 4090, 24 GB) | 256/300 = 85% | 285/300 = 95% |
| machine B (L40S GPUs, 128 CPU cores) | 249/300 = 83% | 279/300 = 93% |
| machine A, before the pixel-range fix | 1/300 | 70/300 |
| machine B, retrained after the position fix | 267/300 = 89% | 286/300 = 95% |

Before the pixel-range fix the check fed images scaled to [0, 1] to a model trained on pixel values of 0-255
([93-train-inference-skew.md](93-train-inference-skew.md) tells it in full).

What this shows: a greedy captioning check with an exact-match score detects a model that ignores
its image. The two machines' models differ by 7 and 6 scenes, which the lab notebook treats as noise
between single runs. The last row was retrained with the corrected position table on the same data
and budget (lab notebook, "From-scratch stages on machine B"); the other rows used the old table.

Not measured: how temperature, top-k and top-p change these scores, and how long generation takes
with and without the cache.

## Where it goes wrong

**Repetition.** You see a word or phrase repeated until the token limit, the first failure under
"The idea". The test suite shows it: `test_inference_text` in
`tests/essential/inference/test_inference.py` decodes greedily from a 2-layer, 64-wide fixture model
and asserts that both prompts continue with twenty periods. The test does not separate the two
possible causes, the decoding rule and the tiny model; on your own model, generate the same prompt
with `temperature=0` and with the defaults and compare. Background: the usual remedies are sampling
with top-k or top-p, a lower `max_new_tokens`, or an EOS the model has learned to emit; none was
measured here, and neither `generate` method has a repetition penalty.

**The prompt is not formatted as in training.** Nothing fails: `generate` returns tokens for any
ids, but the model's next-token probabilities were fitted only after prefixes in the training
format. The lab notebook records no run with a mismatched prompt, so its symptom is not described
here. `vlm_check.py` guards against it by cutting its prompt out of a templated caption.
`inference_vlm.py` and the Jupyter notebook build a prompt from an empty text, which the template
closes with `<end_of_utterance>` and a newline; they remove those two tokens when the row is not
padded (next entry), so that the prompt ends where a training caption begins. Left in place, the two
tokens would ask the model to continue after its own end marker, where training had only padding,
which `VLMBatchHandler.prepare_inputs_and_targets` masks out of the loss. To check a prompt, print
it next to a training text:

```bash
.venv/bin/python -c "
from vla_foundry.data.processor import apply_chat_template
from vla_foundry.models.vision_language_backbones.simple_vlm_processor import SimpleVLMProcessor
proc = SimpleVLMProcessor()
mark = 'from left to right:'
full = apply_chat_template(proc, 1, 'robot arm over a table. from left to right: a red cube.')
empty = apply_chat_template(proc, 1, '')
print(repr(full))                        # a training text
print(repr(full.split(mark)[0] + mark))  # the prompt vlm_check.py builds
print(repr(empty))                       # what inference_vlm.py builds before stripping
print(proc(None, [empty])['input_ids'].shape)"
```

The second line must be a prefix of the first. The third ends in `<end_of_utterance>` and a newline,
the two tokens that have to be removed. The fourth belongs to the next entry.

**A padded prompt.** Both methods read the logits at position -1. If that position holds a padding
id, the first new token is predicted from padding and every new token is appended after it. This
does not need a batch: `SimpleVLMProcessor.__call__` defaults to `padding="max_length"` with
`max_length=2048`, so a single prompt comes back right-padded to 2,048 ids (the
`torch.Size([1, 2048])` the command above prints) unless the caller passes `padding=False`, as
`vlm_check.py` does. `inference_vlm.py` passes only `processor_kwargs` from the config,
`custom_vlm.yaml` sets none, and the strip tests the tokenised row (`input_ids[0, -2] == eos_id`),
which on a padded row holds a padding id. On the from-scratch VLM it therefore removes nothing and
generation starts from a padding position; use `vlm_check.py` for that model. To check any caller,
print `inputs["input_ids"].shape` and decode the last five ids before calling `generate`: the row
must end in the last prompt token. The shorter rows of a padded batch have the same problem. With
`use_cache=False`, `Transformer.forward` discards the mask, so every generated token also attends to
the padding in front of it, on whichever side the prompt was padded. Cached and uncached generation
can therefore differ on a padded batch; `test_kv_cache_transformer` compares them on one unpadded
prompt only.

**A model trained before the position fix.** Nothing fails. The rotary table used to keep
only the even positions between 256 and 512, so neighbouring positions shared a rotation
([12-positions.md](12-positions.md) has the mechanism). The check's prompt is at least 270 tokens,
so the tokens it generates sat where the table had collapsed, with or without the cache: `RotaryEmbedding.forward` slices the same table from
`offset=past_length`. Training and the check built the same table, so nothing looked wrong.
`RotaryEmbedding._update_cos_sin_tables` in
[`vla_foundry/positional_embedding.py`](../../vla_foundry/positional_embedding.py) now computes the
angles in float32, so a model trained with the old table meets rotations it was not trained with;
what that does to its scores was not measured here. To run such a model as it was trained, set
`positional_embedding_type: rotary_legacy_autocast` under `transformer` in the `config_model.yaml`
that `vlm_check.py` reads; never train a new model with that value.

**It never stops, or stops late.** With `eos_token_id=None` the loop always runs to
`max_new_tokens`. In a batch it stops only when every row emits EOS at the same generation step;
rows that finished earlier keep generating. `vlm_check.py` does not rely on EOS: it keeps the text
before the first period.

**Scores change between runs.** The defaults draw at random; a check needs `temperature=0` or
`top_k=1`.

## Background and further reading

- Fan et al., [Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833) (2018).
  Draws each token from the 10 most likely ones and reports that this works better for stories than
  beam search, a decoding method that keeps several partial texts at each generation step instead of
  one.
- Holtzman et al., [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
  (2019; ICLR 2020). Introduces nucleus (top-p) sampling, and is the source of the two failures
  described under "The idea"; it finds the first in beam search as well as in greedy decoding.

## Check yourself

1. For logits 3.0, 1.0, 0.0 compute by hand the probabilities at temperature 0.5 with top-k 2, then
   compare with the unfiltered distribution:

   ```bash
   .venv/bin/python -c "import torch; print(torch.softmax(torch.tensor([3.0, 1.0, 0.0]) / 0.5, -1))"
   ```

   Expected: 0.982 and 0.018 with top-k 2; 0.980, 0.018 and 0.002 unfiltered.

2. Run the cache tests and read `test_kv_cache_transformer`. State why cached and uncached greedy
   outputs must be identical, and which line of `Transformer.generate` selects the input in each
   case.

   ```bash
   .venv/bin/python -m pytest tests/essential/inference/test_inference.py -k kv_cache
   ```

3. `test_inference_text` pads `"hi"` on the right to the length of `"This is a batch"`. From which
   position does the first generated token of the `"hi"` row read its logits?
4. Run `sim/pickup/scripts/custom_path.sh check`. Then predict what changes if `top_k=1` in
   `sim/pickup/vlm_check.py` is replaced by `temperature=0` (consider what happens when two logits
   are exactly equal), and what changes if `eos_token_id` is removed, given the line
   `pred.split(".")[0]`. Run it again to test both predictions.
