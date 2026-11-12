---
title: How Much Data, How Much Training
part: 7 Data
order: 71
summary: Count a training budget in demonstrations, windows, samples and epochs, read a run's checkpoints for the point where the policy starts to use the scene, and choose a budget that does not end before it.
status: draft
---

# How Much Data, How Much Training

Two numbers are fixed before a policy is trained: how many demonstrations it learns from and how
many training samples it is given. Here one design ended at 3% / 5% success with one pair of
numbers and at 83% / 85% with another, and the short run's success rate gave no sign that more of
the same would work.

Each pair is **trained / averaged** weights: the weights the optimiser produced, and a running
(exponential moving) average of them saved by the same run as `ema_N.pt` (raw / EMA in the
notebook and `summarize.py`).

## The idea

A run is counted in four units.

- A **demonstration** is one recorded episode of the scripted expert, a hand-written controller
  ([Demonstrations](70-demonstrations.md)): about 66 control steps, 20 per second.
- A **window** is one training example: the observations at one frame plus the actions that
  follow. Every frame starts one, so windows = frames.
- A **sample** is one window drawn by the data loader. A run's **budget** is its number of
  samples.
- An **epoch** is one pass over a dataset's windows.

Worked example, the dataset of the runs below: 4000 demonstrations of `pick_target` (three objects
on the table, an instruction that names one) are 263,289 windows, and a budget of 5,000,000 samples
is 19.0 epochs. A batch holds 64 samples, so the budget is also
5,000,000 // 64 = 78,125 **optimiser steps**, each one update of the weights.

Demonstrations set how many **layouts** the policy has seen (a layout is what a seed fixes: the
objects, their places, the named one, the arm's start pose); samples, how long the optimiser works.

For the policy that solves `pick_target` here, success does not climb steadily. Before the
position fix (a bug that gave neighbouring tokens the same position; see below), on
demonstrations from one start pose in which nothing moves, the slope and the share closed at the
named object (defined below) stay at chance for a million samples or more, then rise to 1 and to
100% within about a million: the **transition**, in which the policy goes from not using the scene
to using it. Success follows last, then holds a level (the **plateau**). A budget that ends before
the transition gives a success rate near zero, one that ends inside it a rate far below the
design's plateau.

## How it works

### Counting the datasets

From the notebook's datasets table (options of `collect.py`); epochs are arithmetic.

| Dataset | Made with | Demonstrations | Windows | Budget (samples) | Epochs |
|---|---|---|---|---|---|
| `pick_cube_1k` | `--task pick_cube` | 1000 | 65,391 | 1.5M | 22.9 |
| `pick_target_1500` | `--task pick_target` | 1500 | 98,841 | 1.0M or 3.0M | 10.1 or 30.4 |
| `pick_target_4k` | `--task pick_target` | 4000 | 263,289 | 5.0M | 19.0 |
| `pick_target_4k_early` | `boost_early_windows.py` (frames 0-7 six times) | 4000 | 423,289 | 5.0M | 11.8 |
| `pick_target_4k_rs` | `--random_start` | 4000 | 241,318 | 5.0M | 20.7 |
| `pick_target_4k_recover` | `--disturb_prob 0.5` | 4000 | 293,541 | 5.0M | 17.0 |
| `pick_target_4k_rs_recover` | both | 4000 | 275,550 | 5.0M | 18.1 |

### The budget also sets the learning rate

`create_scheduler` in [`vla_foundry/scheduler.py`](../../vla_foundry/scheduler.py) divides the
budget by `global_batch_size` (64). With `lr_scheduler: cosine`, the default, `cosine_lr` raises the
learning rate over the first `warmup` optimiser steps (default 1000) to `lr` (5e-4 here), then
lowers it along a cosine to `lr_cooldown_end` (1e-5) at the run's last step
([The training loop](80-the-training-loop.md)). From `cosine_lr` (arithmetic):

| Budget | After 0.8M samples | 1.0M | 1.5M | 2.0M |
|---|---|---|---|---|
| 1M | 6.32e-5 | 1.00e-5 | ended | ended |
| 3M | 4.28e-4 | 3.87e-4 | 2.63e-4 | 1.37e-4 |
| 5M | 4.74e-4 | 4.58e-4 | 4.05e-4 | 3.36e-4 |

A 1M run is not the first million samples of a 3M run: after 0.8M samples its rate is 13% of the
peak, the 3M run's 86%.

### Four numbers per checkpoint

Each **checkpoint** (the training state saved at one point: weights, optimiser state, data
position) drives the simulated arm on 100 unseen layouts
([Closed-loop evaluation](90-closed-loop-evaluation.md)), and four numbers are kept
([Measuring failure](92-measuring-failure.md)): **success**; the share **closed at the named
object** (the object nearest the gripper when it first closed was the named one; chance 33% with
three objects); the **slope** of where the gripper first closed against where the named object
was (1: where the object is; 0: independent of the scene); and the **median miss** to the named
object (the expert's is 5 mm). The last three can move while success is zero.

## In this repository

The plain recipe of the runs below, with ten checkpoints evaluated while the run trains:

```bash
# demonstrations: the second argument (collect.py --episodes, the successful episodes to keep)
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k
# the README's recommended recipe adds --random_start --disturb_prob 0.5 (the notebook's
# pick_target_4k_rs_recover, 275,550 windows); see chapter 72 (coming)
# the other datasets of this chapter:
sim/pickup/scripts/make_dataset.sh pick_target 1500 pick_target_1500
.venv/bin/python -m sim.pickup.boost_early_windows --shards data/preprocessed/pick_target_4k/shards \
    --out data/preprocessed/pick_target_4k_early/shards --max_frame 7 --repeat 5

# budget: the fourth argument (--total_train_samples). PICKUP_NUM_CHECKPOINTS (default 6) sets
# --num_checkpoints; the config already writes the averaged weights (ema_N.pt) next to each checkpoint
PICKUP_NUM_CHECKPOINTS=10 sim/pickup/scripts/train_resilient.sh \
    dp_vit_lang_reactive pick_target_4k my_run 5000000 &

# every checkpoint, newest first, trained (raw) and averaged (ema) weights;
# PICKUP_WATCH=1 keeps waiting for new checkpoints until the training log reports the end of the run
PICKUP_WATCH=1 sim/pickup/scripts/eval_checkpoints.sh experiments/pickup/my_run pick_target 100 &

# at any time: one row per evaluated checkpoint (columns Samples, Success, At named object, Slope,
# Miss, named (mm)); then the loss next to the success of each checkpoint
.venv/bin/python -m sim.pickup.summarize --task pick_target --runs my_run
.venv/bin/python -m sim.pickup.monitor --runs my_run
```

As the code is now, this trains with the fixed position table (93% and 96% at the end, two seeds).
To train as the three runs below were (before the fix), add `--model.transformer.positional_embedding_type
rotary_legacy_autocast` to the training command; with another seed that table ended at 90% / 89%,
not 83% / 85%.

`train_resilient.sh` restarts a run that dies or hangs ([Surviving long
runs](81-surviving-long-runs.md)); [`summarize.py`](../../sim/pickup/summarize.py) tabulates the
evaluations. To count a dataset's windows:

```bash
.venv/bin/python - <<'EOF'
import json
name = "pick_target_4k"
budgets = (1_000_000, 3_000_000, 5_000_000)
manifest = f"data/preprocessed/{name}/shards/manifest.jsonl"
windows = sum(json.loads(line)["num_sequences"] for line in open(manifest) if line.strip())
print(windows, "windows")
for budget in budgets:
    print(f"{budget:,} samples: {budget / windows:.1f} epochs, {budget // 64:,} optimiser steps of 64")
EOF
```

<!-- interactive: loss-vs-success -->
*Figure: one run's training loss against samples, with the closed-loop success of each evaluated
checkpoint; compare where success rises with what the loss does there.*

The loss is no substitute: similar final losses gave 74% on `pick_cube` (0.058) and 1/100 on
`pick_target` (0.069) with `dp_vit` ([Loss is not success](91-loss-is-not-success.md)).

## What we measured

From the [lab notebook](../lab-notebook.md): one training seed per run, 100 unseen layouts per
row, machine B (L40S GPUs, 128 CPU cores). The policy is `dp_vit_lang_reactive` (48.9M
parameters, 12.6M of them the instruction's embedding table): patch and instruction tokens, no
history ([Language conditioning](51-language-conditioning.md)). The three runs below differ only in
demonstrations and samples and predate the position fix. Cells are trained / averaged; rows
covering several checkpoints give ranges.

**4000 demonstrations, 5M samples** (`dp_vit_lang_reactive_pt4k`):

| Samples | Success | Closed at the named object | Slope | Median miss (mm) |
|---|---|---|---|---|
| 1.0M | 0% / 1% | 35% / 32% | 0.11 / 0.14 | 138 / 154 |
| 1.5M | 4% / 2% | 61% / 49% | 0.47 / 0.37 | 102 / 104 |
| 2.0M | 4% / 36% | 95% / 98% | 0.81 / 0.96 | 32 / 18 |
| 2.5M | 56% / 70% | 98% / 100% | 0.94 / 0.98 | 14 / 13 |
| 3.0M to 5.0M, five checkpoints | 81-86% / 75-84% | 100% / 99-100% | 0.98-1.00 / 0.97-0.99 | 7-9 / 7-9 |
| 5.0M (final) | 83% / 85% | 100% / 100% | 0.98 / 0.98 | 7 / 7 |

**1500 demonstrations, 3M samples** (`dp_vit_lang_reactive_pt1500_3M`):

| Samples | Success | Closed at the named object | Slope | Median miss (mm) |
|---|---|---|---|---|
| 0.3M to 0.9M, three checkpoints | 1% / 0% | 31-36% / 26-36% | 0.00-0.07 / 0.02-0.06 | 180-202 / 164-176 |
| 1.2M | 3% / 0% | 56% / 38% | 0.52 / 0.14 | 99 / 152 |
| 1.5M | 21% / 18% | 94% / 87% | 0.95 / 0.88 | 29 / 46 |
| 1.8M and 2.1M | 44-59% / 43-50% | 98-99% / 97-100% | 0.98-1.00 / 0.95-0.97 | 12-17 / 12-17 |
| 2.4M to 3.0M, four checkpoints | 60-64% / 62-70% | 99-100% / 100% | 0.96-0.99 / 0.97-0.99 | 10-11 / 11 |

**1500 demonstrations, 1M samples** (`dp_vit_lang_reactive_pt1500`):

| Samples | Success | Closed at the named object | Slope | Median miss (mm) |
|---|---|---|---|---|
| 0.2M to 0.6M, three checkpoints | 0-1% / 0-1% | 29-37% / 30-38% | -0.01 to 0.11 / -0.05 to 0.09 | 162-177 / 159-167 |
| 0.8M | 2% / 4% | 45% / 44% | 0.36 / 0.24 | 121 / 141 |
| 1.0M (final) | 3% / 5% | 49% / 41% | 0.29 / 0.18 | 127 / 139 |

<!-- interactive: onset -->
*Figure: success, closed at the named object and the slope against samples for the three
runs; the last two rise before success, and the 1M run stops with both part way up.*

**The order of the columns.** The slope and the share closed at the named object move first,
together (from 1.5M with 4000 demonstrations, 1.2M with 1500), while success is at 0-4%. Success
comes last because a grasp forgives little: at 2.0M the trained weights still missed by a median
32 mm, and the expert with its grasp point moved 30 mm sideways succeeds in 32% of episodes
(notebook, "Grasp tolerance", machine B).

**The 1M run ended inside the transition.** Its slope left zero at 0.8M (0.36 / 0.24) and climbed
no further (0.29 / 0.18 at the end) as its rate fell from 6.32e-5 to 1e-5; the 3M run on the same
demonstrations, at 0.07 at 0.9M, reached 0.95 / 0.88 at 1.5M. Samples, schedule and chance are
confounded (one seed per run). Judged by success alone, the 1M run looked like a broken design
([field notes](../notes.md), "Looking at the scene arrives late, and all at once").

**Averaged weights.** With 4000 demonstrations they led during the transition (36% against 4% at
2.0M, 70% against 56% at 2.5M). In the 3M run on 1500 they did not: within 9 points of the
trained weights throughout.

**More demonstrations.**

- The plateau is higher, 83% / 85% against 64% / 62% at the end; from 3.0M on, 75-86% against
  60-70% from 2.4M on.
- The trained weights' slope first passed 0.3 at 1.2M in the 3M run on 1500 demonstrations (12.1
  epochs) and at 1.5M with 4000 (5.7 epochs), and passed 0.9 at 1.5M and 2.5M: a gap that a
  second seed with 4000 closes (below).

**The position fix moved the transition earlier.** The runs above used a rotary position table
computed in bfloat16, in which neighbouring action-step tokens shared a position
([Positions](12-positions.md)). Trained weights, before / after (the run with the corrected table,
`dp_vit_lang_reactive_pt4k_posfix`, saved no averaged weights):

| Samples | Success, before / after | Closed at the named object, before / after | Slope, before / after | Median miss (mm), before / after |
|---|---|---|---|---|
| 1.0M | 0% / 8% | 35% / 65% | 0.11 / 0.58 | 138 / 80 |
| 1.5M | 4% / 58% | 61% / 98% | 0.47 / 0.94 | 102 / 14 |
| 2.0M | 4% / 40% | 95% / 98% | 0.81 / 0.97 | 32 / 16 |
| 5.0M | 83% / 93% | 100% / 99% | 0.98 / 0.98 | 7 / 6 |

The slope passes 0.9 a million samples earlier. Success fell from 58% to 40% between 1.5M and
2.0M while the slope held (0.94, 0.97): do not judge a run, or rank two, from a single checkpoint.

A second pair with other seeds (`--data.seed 43 --hparams.seed 43`), differing in the table alone,
agrees: closed at the named object after 1.0M samples 83% / 72% with the fixed table against
44% / 39% with the old, and 96% / 96% against 90% / 89% at the end (notebook, "A second seed, with
and without the fix"). Its old-table run passed a slope of 0.9 at 1.5M, a million samples earlier
than the first: the crossing varies that much between seeds.

**What the demonstrations contain moves it too.** Same policy, 4000 episodes, 5M samples, after
the fix (notebook, "What the demonstrations contain"):

| Demonstrations | Success at 1.0M | Closed at the named object, slope at 1.0M | Final |
|---|---|---|---|
| one instruction per layout | 8% / - | 65%, 0.58 | 93% / - |
| early windows repeated | 5% / - | 99%, 0.93 | 96% / - |
| random start poses | 56% / 76% | 91%, 0.81 | 98% / 98% |
| the object moved, the expert recovers | 38% / 79% | 81%, 0.96 | 89% / 84% |
| every object named | 41% / 38% | 98%, 1.02 | 91% / 89% |
| random starts and recoveries | 68% / 86% | 100%, 1.02 | 94% / 94% |

The field notes read this as one effect: the more windows need the scene, the sooner the policy
looks. More in [the `pick_target` case study](101-case-pick-target.md).

**One object.** In the first round (machine A, one RTX 4090, 24 GB; final checkpoints, 50
episodes each), `dp_vit` with history on 1000 `pick_cube` demonstrations reached 40% (28-54%) after
500k samples and 74% (60-84%) after 1.5M. In the second round, before the fix and evaluated at
every checkpoint, the single-cube policy without history passed a slope of 0.9 by 0.3M samples
(0.96 / 0.94; 4.6 epochs) and ended at 95% / 91% after 1.5M: with one object, looking comes early
(notebook, "Single cube").

**Not shown:** how much of the higher plateau the demonstrations account for. The budgets differ
too, and the 1500-demonstration plateau lies in the last fifth of its schedule, where the rate is
low (step 5 below).

## Choosing a budget

1. **Find the closest comparable run** and note where its trained weights' slope reached 0.9:
   1.5M samples (1500 demonstrations, one seed); 2.5M and 1.5M (4000, two seeds) before the fix;
   1.5M (4000, two seeds) after. With no comparable run, give the first a generous budget (5M
   samples here) and evaluate every checkpoint; it becomes your reference.
2. **Budget at least what the comparable run had**, and no less than twice the sample count at
   which its slope reached 0.9. In the five long runs here success levelled off
   between 0.5M and 2.0M samples after that point (the two long runs above, both seed pairs). A
   shorter budget lowers the learning rate earlier, and the 1M run stopped climbing as its rate fell
   (one seed): do not shorten a budget because a crossing came early. A rule of thumb from five runs.
3. **Save ten checkpoints, not the default 6**, and evaluate both sets of weights at each.
4. **Read the slope before the success**, over the last two or three checkpoints together (the 1M
   run's slope went 0.36, 0.19, 0.29).
   - Slope within 0.15 of zero and closed at the named object 26-38% at every recent checkpoint:
     not started. A flat 0% says nothing about what comes next; the 85% run was at 0% / 1% after
     1.0M samples.
   - Slope at or above 0.3 at any recent checkpoint, or closed at the named object clearly above
     chance at several in a row (the 1M run: 45%, 49%, 49%): started. A budget that ends here was
     too short.
   - Slope near 1, closed at the named object near 100%: the miss decides; compare it with the
     grasp tolerance.
5. **Read the plateau where the rate is still high.** Success flat over three checkpoints in
   mid-schedule is a plateau (at 3.0M of 5M the rate is still 37% of its peak; arithmetic). Flat
   at the very end may be the schedule running out.
6. **Call a design failed** only when it is flat well past a comparable run's transition, then
   find out why ([History and shortcuts](52-history-and-shortcuts.md) has an example).

## Where it goes wrong

| What you see | Why | How to check | What to do |
|---|---|---|---|
| Success 0-5% at the end | The budget ended before or inside the transition | Step 4 on the last checkpoints | If started, train again with a budget of step 2 |
| Trained and averaged weights differ by tens of points | The checkpoint is inside the transition (4% / 36% at 2.0M) | Slope of both | Evaluate both at every checkpoint |
| Designs ranked after a short run | Short runs rank designs by when their transition starts, not where they end (three 1M-sample runs: 3% / 5%, 0% / 1%, 8% / 15%; notebook, "Which demonstrations") | Whether each slope passed 0.9 | Compare past the transition of the slowest |
| Two runs compared at equal samples, different budgets | The rate differs (13% against 86% of peak at 0.8M) | `cosine_lr` (Check yourself, 2) | Compare at equal fractions of the schedule, or rerun at one budget |
| Closed at the named object 100% from the first checkpoint | With one object, chance is 100% (single-cube tables) | Count the objects | Read the slope and the miss |

## Background and further reading

Background, not measured here:

- Power et al., *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets*, 2022
  ([arXiv:2201.02177](https://arxiv.org/abs/2201.02177)). Networks can generalise long after
  fitting their training data; here the layouts were not fitted first (notebook, "First action
  chunk, open loop").
- Polyak and Juditsky, *Acceleration of Stochastic Approximation by Averaging*, 1992
  ([SIAM J. Control Optim.](https://doi.org/10.1137/0330046)). Averaging the iterates of
  stochastic gradient descent speeds convergence; the average here is exponential.
- Lin et al., *Data Scaling Laws in Imitation Learning for Robotic Manipulation*, 2024
  ([arXiv:2410.18647](https://arxiv.org/abs/2410.18647)). Diverse environments and objects matter
  more than more demonstrations.
- Mandlekar et al., *What Matters in Learning from Offline Human Demonstrations for Robot
  Manipulation*, 2021 ([arXiv:2108.03298](https://arxiv.org/abs/2108.03298)). Results depend on
  when training is stopped.

## Check yourself

1. Collect 1000 `pick_cube` demonstrations and count their windows with the snippet above (set
   `name = "pick_cube_1k"` and add 1_500_000 to the budgets). How many epochs are 1.5M samples?
   The notebook has 22.9.

   ```bash
   sim/pickup/scripts/make_dataset.sh pick_cube 1000 pick_cube_1k
   ```

2. Print the learning rate of the three budgets with the repository's own function (a run that
   has ended prints its last rate). What was each run's rate when its slope first passed 0.3?

   ```bash
   .venv/bin/python - <<'EOF'
   from types import SimpleNamespace
   from vla_foundry.scheduler import cosine_lr

   for budget in (1_000_000, 3_000_000, 5_000_000):
       steps = budget // 64
       rate = cosine_lr(SimpleNamespace(param_groups=[{}]), 5e-4, 1000, steps, 1e-5, 0.0)
       at = (800_000, 1_000_000, 1_200_000, 1_500_000, 2_000_000)
       print(budget, [f"{rate(min(samples // 64, steps - 1)):.2e}" for samples in at])
   EOF
   ```

3. A run on a new task has 0% success at its first four checkpoints, with slopes of 0.05, 0.02,
   0.31 and 0.44, and its budget ends at the next one. What would success alone have told you,
   and what do you do?
