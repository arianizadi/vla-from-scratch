---
title: The Training Loop
part: 8 Training
order: 80
summary: Follow a run from its config to its checkpoint files, read its log line, and choose which saved weights to evaluate.
status: draft
---

# The Training Loop

The training loop is the program that turns shards (tar files of training windows) into
checkpoints (the training state: weights, optimiser state and position in the data, saved at
intervals). Every model in this repository is trained by this one loop, and a
policy's closed-loop success depends on choices made inside it: how many samples, the
learning-rate schedule that budget implies, and which of the two saved sets of weights is
evaluated.

## The idea

A run is defined by a budget counted in **samples**. A sample is one training window drawn by the
data loader: for a policy, the observations at one frame of a demonstration plus the actions that
follow ([70-demonstrations.md](70-demonstrations.md)). Optimiser steps, epochs, checkpoints and
the learning-rate schedule are derived from that number.

Worked example, used throughout: `dp_vit_lang_reactive` on the plain `pick_target_4k` (4000
demonstrations made without `--random_start` or `--disturb_prob`, 263,289 windows), trained
before the position fix with `total_train_samples = 5_000_000` and batch 64, which the
[lab notebook](../lab-notebook.md) records at 83% / 85% success on `pick_target` (trained /
averaged weights, section 6). The [README](../../README.md)'s recipe adds
`--random_start --disturb_prob 0.5` to the dataset and reaches 94% / 94%. The example's budget is
5,000,000 // 64 = 78,125 **optimiser steps** (one update of the weights on one batch each) and
19.0 **epochs** (passes over the dataset). With 10 **checkpoints**, a **checkpoint interval**, the slice of the budget
after which one is written, is 500,000 samples.

## How it works

### 1. The config

`main()` in [`vla_foundry/main.py`](../../vla_foundry/main.py) starts with
`draccus.parse(config_class=TrainExperimentParams)`;
[draccus](https://github.com/marin-community/draccus) fills a Python dataclass from a YAML file
and from command-line arguments. Three layers decide each value, later ones winning: the
dataclass defaults; the YAML file given as `--config_path`, which can include **presets** (YAML
fragments under `vla_foundry/config_presets/`); and command-line arguments in dotted form.

For the example, `hparams.lr` is 1e-4 in `HyperParams` and 5e-4 in the preset
`vla_foundry/config_presets/hparams/diffusion_policy.yaml`. `ema.enabled` is false in `EMAParams`,
but the pickup policy configs set `ema: enabled: true`
([`sim/pickup/configs/dp_vit_lang_reactive.yaml`](../../sim/pickup/configs/dp_vit_lang_reactive.yaml)),
so every run made from them saves averaged weights. The budget is in no YAML file either:
[`sim/pickup/scripts/train.sh`](../../sim/pickup/scripts/train.sh) passes `--total_train_samples`
and `--num_checkpoints` (6 unless `PICKUP_NUM_CHECKPOINTS` is set; 10 in the example).

`main()` writes the resolved config to `config.yaml` in the run directory; the policy server (the
process that loads a checkpoint's weights and answers the simulator's requests for actions,
[90-closed-loop-evaluation.md](90-closed-loop-evaluation.md)) rebuilds the model from that file.

### 2. Data: shards and workers

A **shard** is a tar file of windows, read as a stream by the
[WebDataset](https://github.com/webdataset/webdataset) library: 256 windows per shard here
(`--samples_per_shard 256` in `make_dataset.sh`). `manifest.jsonl` lists the shards and their
sample counts.

For each checkpoint interval, `get_datastring_input` in
[`vla_foundry/data/dataloader.py`](../../vla_foundry/data/dataloader.py) takes shards from the
manifest, in a seeded shuffled order, until they hold at least `samples_per_checkpoint` samples
and number at least 8, one per worker; when the manifest runs out it reshuffles and continues
(`allow_multiple_epochs`). The list is then trimmed to a multiple of the worker count. Only the
order of shards is shuffled: preprocessing shuffled the windows once, before writing them, and a
shard's windows are read in that order every time (effect not measured:
[roadmap](../roadmap.md), fixes ledger).

`get_wds_dataloader` wraps the pipeline in a `wds.WebLoader` with `data.num_workers` worker
processes (8 in the `dp_vit*` configs). In `RoboticsPipeline.create_pipeline` each worker gets its
share of the shards, decodes the JPEGs, forms batches of 64 (`wds.batched(..., partial=False)`
drops its last incomplete batch) and runs `RoboticsProcessor`.

### 3. Checkpoint intervals

`main()` computes `total_steps = total_train_samples // global_batch_size` and loops: select
shards, build a loader, `train_one_checkpoint`, `save_checkpoint`. `train_one_checkpoint` returns
when its loader is exhausted or when `step` reaches `total_steps`.

An interval therefore ends when its shards end, not after exactly 500,000 samples. Arithmetic on
the code for the example, taking the 263,289 windows as 1,028 shards of 256 and one of 121: the
shards needed for 500,000 samples are trimmed to 1,952, which hold at most 499,712 samples, 7,808
optimiser steps, so ten intervals cover at most 78,080 of the 78,125. The loop then opens an
eleventh interval and saves again: the run ends with `checkpoint_11.pt`, and the notebook's table
of it has ten rows 0.5M samples apart and a final one. With `train.sh`'s default of 6, the same
code run on a manifest of these sizes gives 7 files.

The reverse happens when an interval starts within 8 shards of the end of the manifest and needs
less than one more pass: a whole reshuffled pass is appended, and a 300,000-sample run on the
1500-demonstration dataset (387 shards) asks for 6 checkpoints and writes 5.

### 4. One optimiser step

<!-- interactive: training-step -->
*Figure: one optimiser step: a batch goes through the model, the loss compares prediction and
target, the backward pass gives every weight a gradient, the optimiser moves the weights.*

One iteration of the loop in `train_one_checkpoint`
([`vla_foundry/train.py`](../../vla_foundry/train.py)):

1. `scheduler(step)` sets the learning rate for this optimiser step.
2. `next(data_iterator)` fetches a batch; the wait is recorded as `data_time`.
3. The batch handler builds inputs and targets: for a policy, `DiffusionPolicyBatchHandler`
   (`vla_foundry/models/batch_handlers.py`) draws noise shaped like the batch of action chunks,
   `[64, 15, 8]` in the example, and the target is `noise - actions`
   ([41-flow-matching.md](41-flow-matching.md)).
4. Forward pass and loss run inside `autocast()`, which `get_autocast`
   (`vla_foundry/precision.py`) sets to `dtype=torch.bfloat16` for `precision: amp_bf16`:
   operations run in bf16, a 16-bit float with float32's exponent range and 8 significant bits.
   **Mixed precision** means doing most of the arithmetic in a 16-bit format and keeping float32
   where rounding would hurt. The original recipe (Micikevicius et al.) keeps a float32 copy of
   the weights so that small updates are not rounded away. This repository does not: a
   single-process run stores its weights in the type `get_model_precision`
   (`vla_foundry/distributed.py`) returns, bf16 here, and by arithmetic on the format (not
   measured here) a change smaller than roughly 0.2 to 0.4% of its magnitude leaves a bf16 weight
   unchanged.
5. `local_loss.backward()`. With gradient accumulation, items 4 and 5 repeat on slices of the
   batch; in the pickup configs `accum_freq` is 1.
6. `clip_grad_norm_(model.parameters(), grad_clip_norm)`: if the L2 norm of the whole gradient
   exceeds `grad_clip_norm` (1.0 in the preset, off by default) the gradient is scaled down to it.
7. `optimizer.step()`. `create_optimizer` (`vla_foundry/optimizer.py`) builds AdamW (Loshchilov
   and Hutter), which divides a running average of each weight's gradient by the square root of a
   running average of its square (betas 0.9 and 0.95), so a weight moves by up to about the
   learning rate per update. Weight decay (0.01) shrinks each weight by `lr x 0.01` of its value
   per update, too little to change a bf16 weight (item 4); parameters whose name ends in `.bias`
   or contains `norm` are exempt.
8. `ema_model.step(...)` if the averaged copy of section 6 is enabled, then `step += 1` and a log
   line every `log_every_n_steps` (20) iterations.

### 5. The learning-rate schedule

`create_scheduler` in [`vla_foundry/scheduler.py`](../../vla_foundry/scheduler.py) returns a
function of the optimiser step number. The default `lr_scheduler: cosine` is a linear warm-up
(the rate rises from near zero to its base value over the first `warmup` optimiser steps)
followed by a cosine decay to a minimum. Background: the warm-up is there because AdamW's running
averages rest on very few batches at the start; the decay lets the weights settle late in
training. With $s$ the optimiser step, $w$ the warm-up length and $T$ the total:

$$\text{lr}(s) = \begin{cases} \text{lr}_\text{base}\,\dfrac{s+1}{w} & s < w \\[2mm] \text{lr}_\text{min} + \tfrac12\left(1+\cos\dfrac{\pi (s-w)}{T-w}\right)(\text{lr}_\text{base}-\text{lr}_\text{min}) & s \ge w \end{cases}$$

For the example, `lr` is 5e-4, `lr_cooldown_end` (the minimum) 1e-5, `warmup` 1000 (the default)
and $T = 78{,}125$: the rate is 5e-7 at $s=0$, 5e-4 at $s=999$, 2.55e-4 at $s=39{,}562$ (halfway
through the decay) and 1e-5 at $s=78{,}124$.

$T$ comes from the budget, so the budget sets how fast the rate falls as well as where training
stops: after 1.0M samples ($s=15{,}625$) the rate is 4.58e-4 in the example, 3.87e-4 under a 3M
budget, and 1e-5 under a 1M budget, whose run has ended.

### 6. EMA weights

An **exponential moving average (EMA)** of the weights is a second copy of the model that is never
trained directly. After every optimiser step each of its weights moves a small fraction toward the
trained weight:

$$\theta_\text{ema} \leftarrow d\,\theta_\text{ema} + (1-d)\,\theta$$

The trained ("raw") weights are pushed around by the noise of each batch; the average smooths that
noise out. An EMA with decay $d$ gives the weights from $k$ optimiser steps ago a share
$(1-d)\,d^k$, so it averages over about $1/(1-d)$ recent ones.

`EMAModel` in [`vla_foundry/models/ema.py`](../../vla_foundry/models/ema.py) (the default
`ema.type`) raises $d$ as training proceeds: `get_decay` returns
$1-(1+n/\texttt{inv\_gamma})^{-\texttt{power}}$, capped at `max_value`, where $n$ is
`optimization_step - update_after_step - 1`; the first two calls return 0, which copies the model.
With the defaults (`inv_gamma` 1, `power` 0.75, `max_value` 0.9999, `update_after_step` 0):

| $n$ (updates after the two copies) | Decay $d$ | $1/(1-d)$ in optimiser steps |
|---|---|---|
| 1 | 0.405 | about 2 |
| 31,248 (2.0M samples) | 0.99957 | about 2,350, or 150,000 samples |
| 78,123 (last update of the example) | 0.99979 | about 4,700, or 300,000 samples |

`main()` keeps the EMA copy in float32: a decay such as 0.9999 rounds to 1.0 in bf16, which would
stop every update.

## In this repository

The loop is `main()` (`vla_foundry/main.py`) and `train_one_checkpoint` (`vla_foundry/train.py`);
the config classes are in `vla_foundry/params/`. The example run (plain demonstrations, 10
intervals as on machine B):

```bash
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k
PICKUP_NUM_CHECKPOINTS=10 sim/pickup/scripts/train_resilient.sh \
  dp_vit_lang_reactive pick_target_4k my_run 5000000
ls experiments/pickup/my_run experiments/pickup/my_run/checkpoints
```

The README's `--random_start --disturb_prob 0.5` changes the window count (275,550 in the
notebook), not the file counts of section 3.

**Files.** Per interval, `save_checkpoint`
([`vla_foundry/file_utils.py`](../../vla_foundry/file_utils.py)) writes checkpoint N as
`checkpoint_N.pt` (the raw weights, `global_step`, `samples_seen`, the shard cursor and shuffle
seed), `optimizer_N.pt` (the AdamW state) and, with `ema.enabled`, `ema_N.pt` (the averaged
weights and their update count).

**Choosing weights.** `sim/pickup/policy_server.py` takes `--weights auto|ema|raw`; `auto` serves
the EMA weights file when the run's `config.yaml` has `ema.enabled`, otherwise the raw one.
`eval.sh` passes `--weights` only when `PICKUP_WEIGHTS` is set, so the default for an EMA run is
EMA. To evaluate the raw weights of the newest checkpoint:

```bash
PICKUP_WEIGHTS=raw PICKUP_OUT=results/pickup/my_run/pick_target/final_raw \
  sim/pickup/scripts/eval.sh experiments/pickup/my_run pick_target 100
```

Repeat with `ema` and `final_ema` (without `PICKUP_OUT` both write to one directory).
`eval_checkpoints.sh` does this for every checkpoint and both kinds of weights (`ckpt_<N>_raw`,
`ckpt_<N>_ema`), and with `PICKUP_WATCH=1` keeps doing so while the run trains. `summarize.py`
prints one row per evaluation, with the samples seen at that checkpoint and the slope ("What we
measured"):

```bash
PICKUP_WATCH=1 sim/pickup/scripts/eval_checkpoints.sh experiments/pickup/my_run pick_target
.venv/bin/python -m sim.pickup.summarize --task pick_target --runs my_run
```

**The log line.** `train.sh` sends the trainer's output to
`experiments/pickup/<run>.train.log`. Every 20 optimiser steps `Metrics.update_and_log_state`
([`vla_foundry/meters.py`](../../vla_foundry/meters.py)) writes:

```text
Train Checkpoint: <interval> [<samples so far>/<estimated samples in interval> (<%>)] Loss: <mean> Data (t): <s> Batch (t): <s>, <samples/s>/s, <samples/s>/s/gpu LR: <rate>
```

`Loss`, `Data (t)` and `Batch (t)` are means since the previous line; `Data (t)` is the wait for a
batch and `Batch (t)` the whole iteration, waiting included. If `Data (t)` is close to zero the
model sets the pace and more workers change nothing; if it is a large part of `Batch (t)` the
trainer is waiting for the CPU side. The loss is that of the raw weights; the EMA copy is never
run during training.

**Resuming.** `train.sh` with `--model.resume_from_checkpoint <path>/checkpoint_N.pt` continues
a run; `train_resilient.sh` is `train.sh` in a retry loop that does this by itself and keeps the
old log ([81-surviving-long-runs.md](81-surviving-long-runs.md)). `main()` restores the weights,
optimiser step, shard position, optimiser and EMA copy. If EMA is switched on for a run whose
checkpoints were written without it, there is no `ema_N.pt`: `main()` logs a warning and
`EMABaseModel.start_from` sets the average to the resumed weights; its update count starts at
zero, so the decay of section 6 warms up again. The schedule depends on the optimiser step and on
`total_train_samples`, so a run resumed with a different budget continues at a different learning
rate.

## What we measured

Throughput (lab notebook): `dp_vit` and `dp_vit_lang`, which take the previous frame, state and
action as well (history: four images per window), ran at about 620 samples/s on machine A (one
RTX 4090, 24 GB) and about 540 on machine B (L40S GPUs, 128 CPU cores); the no-history configs
(`*_reactive`, two images) at about 1100 alone on a machine B GPU and 570-700 while 4-5 policy
servers evaluated on the same GPU.

**The example run at every checkpoint** (machine B, 100 unseen layouts each, raw / EMA weights).
**Slope** is the least-squares slope of where the gripper first closed against where the named
object was: 1 if it closes where the object is, 0 if the scene makes no difference
([92-measuring-failure.md](92-measuring-failure.md)).

| Samples | Success | Slope |
|---|---|---|
| 0.5M | 0% / 0% | 0.02 / -0.06 |
| 1.0M | 0% / 1% | 0.11 / 0.14 |
| 1.5M | 4% / 2% | 0.47 / 0.37 |
| 2.0M | 4% / 36% | 0.81 / 0.96 |
| 2.5M | 56% / 70% | 0.94 / 0.98 |
| 3.0M | 81% / 75% | 0.98 / 0.97 |
| 3.5M | 83% / 82% | 0.99 / 0.99 |
| 4.0M | 86% / 80% | 1.00 / 0.98 |
| 4.5M | 81% / 84% | 0.99 / 0.99 |
| 5.0M | 81% / 82% | 0.98 / 0.99 |
| 5.0M (final) | 83% / 85% | 0.98 / 0.98 |

**The budget.** Success stays at 0% or 1% for the first million samples; between 1.5M and 2.5M the
policy goes from not using the scene to using it, the slope moving first
([field notes](../notes.md), "Looking at the scene arrives late, and all at once"). On 1500
demonstrations the same policy ended at 3% / 5% (slope 0.29 / 0.18) with a 1M budget, 10.1 epochs,
and at 64% / 62% (slope 0.96 / 0.97) with a 3M budget, 30.4 epochs. The 1M run had ended inside
the transition, its learning rate at the minimum, and read as a design that does not work. The two
budgets differ in samples and in schedule (section 5); no run here separates the two. See
[71-how-much-data-and-training.md](71-how-much-data-and-training.md) and
[101-case-pick-target.md](101-case-pick-target.md).

**Raw against EMA.** During the transition the EMA weights were far ahead: 36% against 4% at 2.0M
samples (median miss to the named object 18 mm against 32 mm), 70% against 56% at 2.5M. Before and
after it the two differ by at most 6 points, in either direction: equal within noise (field notes,
"Averaged weights and the sampler's settings"). Checkpoints 10
and 11, under 1% of the budget apart at the minimum learning rate, differ by 2 and 3 points.

**What this does not show.** Each run has one training seed. The EMA lead is not a rule: in the
1500-demonstration 3M run the two stayed within 9 points at every checkpoint, the raw weights
ahead at 1.5M (21% / 18%) and 2.1M (59% / 50%). The supported conclusion: save both, which costs
one extra copy of the model, and evaluate both. Nor is success on static layouts all a robot
needs: with the named object moved at control step 20 of the reach, the final EMA weights
succeeded in 0 of 96 episodes ([95-what-a-policy-looks-at.md](95-what-a-policy-looks-at.md)).

## Where it goes wrong

- **More or fewer checkpoint files than requested.** Expected (section 3).
  `grep "Samples in this checkpoint" experiments/pickup/<run>.train.log` prints what each
  interval selected.
- **Success is still 0% when the budget ends.** That says nothing about what comes next; a slope
  that has left zero does. Do not judge a design on a budget that ends before the transition of a
  comparable run.
- **Two evaluations of one run disagree.** One served raw weights, the other EMA. `results.json`
  records the served weights file under `server`; set `PICKUP_WEIGHTS` explicitly.
- **A result belongs to another run.** Two evaluations started on one port both talked to the
  policy server that had bound it first: about 40 results here were filed under the wrong run and
  redone. `eval.sh` now takes a free port, and `eval.py` stops unless the server reports the run
  it was started for ([90-closed-loop-evaluation.md](90-closed-loop-evaluation.md)).
- **No symptom: the loss falls, and training agrees with serving.** The rotary position table was
  computed in bf16 under the autocast of item 4, which left 641 distinct positions out of 2048. It
  is now computed in float32 and then cast ([12-positions.md](12-positions.md)). Every success
  rate in this chapter comes from a run trained before that fix. Trained again with the fixed
  table, the example recipe ends at 93% (trained weights). A second pair of runs with other
  seeds, differing in the table alone, ends at 96% / 96% with the fixed table and 90% / 89% with
  the old one; in both pairs the fixed table makes the policy start to look earlier (lab
  notebook, "After the position fix").
- **The log stops and GPU use is 0%.** A data-loader worker died and the trainer waits for it
  forever. `train_resilient.sh` kills and resumes a run whose log has not moved for 420 seconds.
- **`ValueError: Zero batches`.** An interval holds fewer samples than
  `global_batch_size x num_workers` (512 here).

## Background and further reading

- Loshchilov and Hutter, "Decoupled Weight Decay Regularization", 2017,
  [arXiv:1711.05101](https://arxiv.org/abs/1711.05101). Showed that L2 regularisation and weight
  decay differ under Adam, and decayed the weights directly (AdamW).
- Loshchilov and Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts", 2016,
  [arXiv:1608.03983](https://arxiv.org/abs/1608.03983). Introduced cosine annealing of the
  learning rate, there with restarts.
- Micikevicius et al., "Mixed Precision Training", 2017,
  [arXiv:1710.03740](https://arxiv.org/abs/1710.03740). Half-precision training with a float32
  copy of the weights and loss scaling.
- Chi et al., "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", 2023,
  [arXiv:2303.04137](https://arxiv.org/abs/2303.04137). Its public code evaluates EMA weights,
  with an `EMAModel` whose warm-up formula is the one in `get_decay` here.

## Check yourself

1. For 300,000 samples at batch 64 and 6 checkpoints, compute `total_steps` and
   `samples_per_checkpoint`. Train on the example's dataset and count the files:

   ```bash
   sim/pickup/scripts/train.sh dp_vit_lang_reactive pick_target_4k loop_test 300000
   ls experiments/pickup/loop_test/checkpoints/checkpoint_*.pt | wc -l
   grep "Samples in this checkpoint" experiments/pickup/loop_test.train.log
   ```

   By arithmetic on the code, expect 7: each interval keeps 192 shards, at most 49,152 samples
   and 768 optimiser steps, so six cover at most 4,608 of the 4,687.

2. Compute the learning rate after 1.0M samples under the example's budget, then under a 3M
   budget (change `T`), and compare with section 5:

   ```bash
   .venv/bin/python -c "import math; T=5_000_000//64; s=15_625; print(1e-5+0.5*(1+math.cos(math.pi*(s-1000)/(T-1000)))*4.9e-4)"
   ```

3. After the example run, list its evaluations. For each kind of weights, which is the first
   checkpoint with a slope above 0.5, and what is its success?

   ```bash
   .venv/bin/python -m sim.pickup.summarize --task pick_target --runs my_run
   ```
