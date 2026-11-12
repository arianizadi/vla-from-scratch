---
title: "Case Study: Three Objects, From 0% to 98%"
part: 10 Case studies
order: 101
summary: Follow the debugging method through the whole history of `pick_target`, wrong turns included, and run the same checks, in the same order, on a policy of your own.
status: draft
---

# Case Study: Three Objects, From 0% to 98%

`pick_target` puts three objects on the table and names one in the instruction. Every first-round
policy failed it; the scripted expert (a hand-written controller in `sim/pickup/expert.py` that
reads the object poses from the simulator) did not. This chapter retraces how that became
a policy that lifts the named object in 98 of 100 unseen layouts.

## The idea

**The task.** `TASKS["pick_target"]` in [`sim/pickup/env.py`](../../sim/pickup/env.py) places 3 of
4 objects (red cube, green can, blue hex nut, yellow T block) and asks for one: "pick up the green
can". The seed fixes the **layout**: which objects are where, which one is named, the arm's start
pose. Success: the named object held 8 cm above its rest height for 10 **control steps** (commands
to the arm, 20 per second); episodes end after 200.

**Terms.** A **sample** is one **window** (the observations at one frame plus the actions that
follow) drawn by the data loader. **Raw / EMA**: a checkpoint's trained weights and their
exponential moving average (EMA, "averaged weights"). **History**: the previous frame, state and action as extra inputs; `*_reactive` configs have none.
**Conditioning**: the tensor the **backbone** (the part that turns images and instruction into
tokens) hands to the **action head** (the flow-matching transformer that predicts a **chunk**, the
next 15 actions, in one go), `[batch, tokens, width]`; the round-one VLAs hand over one row, the
hidden states of **the action token** appended to the prompt
([What the action head sees](50-what-the-action-head-sees.md)). **Closed at the named object**: the
object nearest the gripper when it first closed was the named one (chance: 33%). The **slope** of
where the gripper first closed against where the named object was is 1 if it closes at the object,
0 if the scene makes no difference.

Numbers: the [lab notebook](../lab-notebook.md), one seed per run, 100 unseen layouts per
evaluation. Machine A (one RTX 4090, 24 GB); machine B (L40S GPUs, 128 CPU cores).

## The route

### 1. First round: everything fails, the expert does not

**Believed.** That 1500 demonstrations would do: `dp_vit` had reached 74% on the single cube after
23 epochs (1.5M samples on 1000 demonstrations;
[Case study: one cube](100-case-pick-cube.md)). The first round gave `pick_target` 10 epochs
(1.0M samples on 1500).

**Measured.** Machine A, 1500 demonstrations (notebook, "`pick_target`, first round"):

| Policy | Samples (epochs) | Final loss | Success |
|---|---|---|---|
| `dp_vit`: patch tokens, no language, 36M | 1.0M (10) | 0.069 | 1/100 |
| `vla_qwen`: Qwen3-VL-2B, one conditioning row | 400k (4) | 0.03-0.04 | 6/100 |
| `vla_custom`: 130M VLA built here, one row | 1.0M (10) | ~0.03 | 0/100 |
| same, VLM randomly initialised | 1.0M (10) | ~0.03 | 1/100 |

The losses are low and say nothing about success ([Loss is not success](91-loss-is-not-success.md)).
Through the same evaluation loop the expert scored 99/100 on machine A and 100/100 on machine B,
closing a median 5 mm from the object.

### 2. Measuring how they fail

**Ran.** `GraspProbe` in [`sim/pickup/diagnostics.py`](../../sim/pickup/diagnostics.py) records
where the gripper and every object are at the first command to close.

**Measured.** `dp_vit` and both `vla_custom` closed the gripper in every episode, at the right
height, around control steps 34-44, as the expert does: they ran the whole motion, but closed at
the named object only at chance and 89-110 mm from the nearest object. That distance, from where
the gripper first closed to an object, is the **miss** (the notebook's "median miss" columns; the
per-policy table:
[Measuring failure](92-measuring-failure.md)).

<!-- interactive: grasp-map -->
*Figure: where the gripper first closed, one dot per episode, around the named object, for the
expert and a first-round policy.*

### 3. The first reading, and why it was wrong

**Believed.** That the policy reached for a point between the objects, an average of the reaches it
had seen.

**Measured.** Checked later, on the 1M-sample runs of section 5: one fit per table axis, all
evaluations of a run pooled (notebook, "Does the closing position depend on the scene?"):

| Run | Slope, x / y | R², x / y |
|---|---|---|
| no language, history | 0.04 / 0.09 | 0.001 / 0.006 |
| no language, no history | 0.00 / 0.01 | 0.000 / 0.000 |
| language, history | 0.05 / 0.06 | 0.002 / 0.003 |
| language, no history (starting to move, section 6) | 0.03 / 0.15 | 0.001 / 0.021 |

**Changed.** Closing at the mean of three independently placed objects would give a slope of about
1/3 against the named one. On one axis the mean is $\bar x = (x_1 + x_2 + x_3)/3$ with $x_1$ the
named object; the other two do not move with it, so
$\mathrm{cov}(\bar x, x_1)/\mathrm{var}(x_1) = 1/3$ (arithmetic, not a measurement; Check yourself
2 simulates it). These are at most 0.15, and R² (the share of the closing position's variation the
object's position explains) at most 0.021. The miss is the same against another episode's objects,
and the closing positions fill the area where objects are placed (field notes, "A failing policy
ignores the scene; it does not average"). The question became "why does it not look"
([Measuring failure](92-measuring-failure.md)).

### 4. Four candidate causes

| Candidate | How it was tested |
|---|---|
| No language input (`dp_vit.yaml` has no `text_tokens`) | needed by construction: without it, one object in three at best; section 5 adds it |
| One conditioning vector (the VLAs) | tested after the position fix (section 7) with every token handed over (`conditioning: all_tokens`): [Case study: pretrained backbones](102-case-pretrained-backbones.md). `dp_vit` had patch tokens, so not its cause |
| History: "keep going" predicts most actions | with and without, sections 5-6 |
| Budget and data: 1M samples on 1500 demonstrations | longer runs, more demonstrations, section 6 |

### 5. Second round: everything still fails at 1M samples

**Ran.** Machine B, the patch-token policy on the same demonstrations, with and without history
and the instruction as 32 tokens (48.9M parameters, 12.6M of them its embedding table, against
36.3M).

**Measured.** Notebook, "Final checkpoints":

| Policy, 1.0M samples | Success (raw / EMA) | Closed at the named object | Slope |
|---|---|---|---|
| `dp_vit`: no language, history | 0% / 0% | 31% / 37% | 0.11 / 0.17 |
| `dp_vit_lang`: language, history | 3% / 2% | 35% / 29% | 0.13 / 0.04 |
| `dp_vit_reactive`: no language, no history | 1% / 1% | 26% / 34% | 0.01 / 0.04 |
| `dp_vit_lang_reactive`: language, no history | 3% / 5% | 49% / 41% | 0.29 / 0.18 |

**Changed.** It read as "this design fails with 1500 demonstrations", but language without history
closed at the named object 41-49% of the time. The no-language runs were not trained
longer.

### 6. Training long enough

**Ran.** `dp_vit_lang_reactive` on 4000 demonstrations for 5M samples (19.0 epochs) and on the
same 1500 for 3M (30.4 epochs), every checkpoint evaluated (`eval_checkpoints.sh`).

**Measured.** The 4000-demonstration run was at 0% / 1% after 1.0M samples; at 2.0M its slope was
already 0.81 / 0.96 while success was 4% / 36%; it ended at 83% / 85% (notebook, "4000
demonstrations, no history"; the whole curve and the 1500-demonstration run:
[How much data and training](71-how-much-data-and-training.md)). It went from not using the scene
(slope 0.11 / 0.14 at 1.0M samples) to using it (0.94 / 0.98 at 2.5M), mostly between 1.5M and
2.5M; that stretch is the **transition**. That order is what to watch for
in a run of your own: the slope rises first, then the miss shrinks, then success follows.

**Why the 1M runs of section 5 could not show it.** The learning rate follows `cosine_lr` in
[`vla_foundry/scheduler.py`](../../vla_foundry/scheduler.py): 1000 warm-up optimiser steps (the
`warmup` default in `vla_foundry/params/hyper_params.py`), then a cosine from 5e-4 down to 1e-5
(`lr` and `lr_cooldown_end` in `vla_foundry/config_presets/hparams/diffusion_policy.yaml`).
`create_scheduler` stretches it over the run's own budget, `total_train_samples` divided by the
batch of 64. The same sample count therefore means a different learning rate in runs of different
length. Computed from that code (Check yourself 3), not measured:

| Samples seen | Learning rate, 1M-sample run | Learning rate, 5M-sample run |
|---|---|---|
| 0.5M | 2.8e-4 | 4.9e-4 |
| 1.0M | 1.0e-5 | 4.6e-4 |

The 1500-demonstration 1M run's last checkpoints sit at the start of the transition (notebook), with
the learning rate at its floor. Choose the budget before starting; resuming a finished short run
does not give the long one.

**History.** The twin with history, `dp_vit_lang`, on the same 4000 demonstrations ended at 2% / 2%
after 3.0M samples, against 81% / 75% without history at that point, and where it closed never
depended on the scene (pooled slope 0.08 / 0.04, R² 0.004 / 0.002; notebook, "Does the closing
position depend on the scene?"). It is not blind: it reads the target from its previous action, a
train/inference skew that two probes expose and a hold frame at the start of each demonstration did
not cure ([History and shortcuts](52-history-and-shortcuts.md), which also has what stands after
the position fix of section 7). History left the recipe.

<!-- interactive: onset -->
*Figure: success, closed at the named object and slope against training samples for these runs;
the history run stays flat.*

**Changed.** The recipe became 4000 demonstrations and 5M samples, against 64% / 62% for 1500
and 3M. Data and budget both differ: this ranks the recipes, not the effect of data.

### 7. Two detours

**Results under the wrong name.** Two evaluations sharing a local port both reached the policy
server bound to it first, filing about 40 results under the wrong run; they were redone. `eval.sh`
now takes a free port and passes `eval.py --expect_run`, which stops unless the server serves the
run (and checkpoint) asked for. Results written before that check are caught afterwards:
`served_checkpoint_matches` in [`sim/pickup/summarize.py`](../../sim/pickup/summarize.py) compares
the checkpoint the server recorded in `results.json` with the run, number and weights the
directory names, and `summarize` leaves out every evaluation where they differ.

**The position table in half precision.** The position table holds the rotary angles precomputed
for each token position. `RotaryEmbedding._update_cos_sin_tables` in
[`vla_foundry/positional_embedding.py`](../../vla_foundry/positional_embedding.py) computed it in
bfloat16 under autocast, which merged neighbouring positions, the action-step tokens among them
([Positions](12-positions.md) has the mechanism and the count). Training and serving shared the
table, so nothing looked wrong. It is now computed in float32;
`positional_embedding_type: rotary_legacy_autocast` reproduces the old one (command 8). With the
fix the same recipe ends at 93% (trained weights) against 83%, and starts to look earlier, mostly
between 1.0M and 1.5M samples instead of 1.5M and 2.5M; a second pair of seeds, differing in the
table alone, agrees. What sections 3-6 concluded still holds. Every run in sections 1-6 used the old
table, except the every-token VLAs (section 4) and the hold-frame history run (section 6).

### 8. What the probes showed: it looks once

**Believed.** That the 85% policy, without history, watches the scene.

**Measured.** Three probes of that checkpoint (EMA, before the fix) said otherwise
([What a policy looks at](95-what-a-policy-looks-at.md) has them probe by probe). With the cameras
frozen on the first frame it still closed at the named object, a little further off, and almost
never succeeded. When the expert drove the first 25 control steps toward another object, it closed
at that object. When the named object was moved 80 mm at control step 20, it closed at the old
place.

<!-- interactive: disturbance -->
*Figure: a top-down replay of a reach during which the named object is moved; the gripper closes
at the old place.*

It decides from its first look, then follows its own arm, whose straight reach from one home pose
gives the end away; later images only refine the last centimetres. The policy trained after the fix
does the same.

### 9. What changed that: the demonstrations

After the fix, same policy, 4000 demonstrations, 5M samples, only the data changed. Two collection
options add what plain demonstrations lack:

- **Random start poses** (`collect.py --random_start`): the arm's position no longer gives the goal
  away. Alone they reach 98% / 98%, the title's 98, but the policy watches only early in the reach.
- **Recoveries** (`collect.py --disturb_prob 0.5`): in 1505 of 4000 episodes the object is
  moved 5-10 cm once, during the reach or as the gripper closes (`Disturbance`), and the expert
  plans again. Alone they make the policy follow a late move and try again, at a cost on the
  unchanged task.
- **Both** end at 94% / 94% and hold 89-94% under every disturbance, also from a random start
  pose: the recipe `README.md` recommends.

One seed each. What each option does over training:
[Which demonstrations teach what](72-which-demonstrations-teach.md). The policies under every
disturbance, with the scripted expert as the reference:
[Case study: self-correction](103-case-self-correction.md).

## In this repository

```bash
# From the repository root. make_dataset.sh replaces a dataset of the same name.
# The policy: sim/pickup/configs/dp_vit_lang_reactive.yaml (text_tokens: 32, image_indices: [0],
# lowdim_past_timesteps: 0).

# 1. the expert through the evaluation loop
.venv/bin/python -m sim.pickup.eval --task pick_target --episodes 100 --expert \
    --out results/pickup/expert/pick_target

# 2. the plain run of sections 6-7, every checkpoint evaluated while it trains (EMA is on in the
#    config); the recommended recipe is the last dataset of command 6
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k
PICKUP_WATCH=1 sim/pickup/scripts/eval_checkpoints.sh experiments/pickup/pt4k pick_target &
PICKUP_NUM_CHECKPOINTS=10 sim/pickup/scripts/train_resilient.sh dp_vit_lang_reactive pick_target_4k pt4k 5000000
wait
.venv/bin/python -m sim.pickup.summarize --task pick_target --runs pt4k

# 3. where it closed, against the expert (<N>: a checkpoint number)
.venv/bin/python -m sim.pickup.grasp_map results/pickup/expert/pick_target/results.json \
    results/pickup/pt4k/pick_target/ckpt_<N>_ema/results.json --extent_cm 30

# 4. probes of the newest checkpoint (averaged weights; PICKUP_WEIGHTS=raw for the trained ones)
run=experiments/pickup/pt4k; out=results/pickup/pt4k/pick_target
PICKUP_OUT=$out/stale    sim/pickup/scripts/eval.sh $run pick_target 100 --stale_images
PICKUP_OUT=$out/handover sim/pickup/scripts/eval.sh $run pick_target 100 --expert_prefix 25 --prefix_target distractor
PICKUP_OUT=$out/reach20  sim/pickup/scripts/eval.sh $run pick_target 100 --disturb reach --disturb_step 20
PICKUP_OUT=$out/grasp    sim/pickup/scripts/eval.sh $run pick_target 100 --disturb grasp
PICKUP_OUT=$out/random   sim/pickup/scripts/eval.sh $run pick_target 100 --random_start
# other moments of the move: --disturb_step 10 and --disturb_step 30 in place of 20
# with history, also: --first_prev_action expert, and --expert_prefix 10

# 5. the reference: an expert that plans again, under the same disturbance
.venv/bin/python -m sim.pickup.eval --task pick_target --episodes 100 --expert --expert_reactive \
    --disturb reach --disturb_step 20 --out results/pickup/expert/pick_target_reach20

# 6. the datasets of section 9, each trained like command 2 and probed like command 4;
#    the last one is the recipe README.md recommends
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k_rs --random_start
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k_recover --disturb_prob 0.5
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k_rs_recover --random_start --disturb_prob 0.5
PICKUP_NUM_CHECKPOINTS=10 sim/pickup/scripts/train_resilient.sh dp_vit_lang_reactive pick_target_4k_rs pt4k_rs 5000000
PICKUP_NUM_CHECKPOINTS=10 sim/pickup/scripts/train_resilient.sh dp_vit_lang_reactive pick_target_4k_recover pt4k_recover 5000000
PICKUP_NUM_CHECKPOINTS=10 sim/pickup/scripts/train_resilient.sh dp_vit_lang_reactive pick_target_4k_rs_recover pt4k_rs_recover 5000000

# 7. the history policy after the fix (section 6), on demonstrations that begin with a hold frame
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k_hold --hold_steps 1
PICKUP_NUM_CHECKPOINTS=10 sim/pickup/scripts/train_resilient.sh dp_vit_lang pick_target_4k_hold pt4k_hold 5000000

# 8. the detour of section 7: the plain run again on the old half-precision position table
PICKUP_NUM_CHECKPOINTS=10 sim/pickup/scripts/train_resilient.sh dp_vit_lang_reactive pick_target_4k pt4k_legacy 5000000 \
    --model.transformer.positional_embedding_type rotary_legacy_autocast
```

## Where it goes wrong

| What you see | Why | What to do |
|---|---|---|
| 0% after 1M samples | the flat stretch before the transition | read the slope by checkpoint; judge only after a comparable run's transition |
| A short run and a long one disagree at the same sample count | the cosine is stretched over each run's own budget: at 1.0M samples the 1M run is at its floor, the 5M run near its peak | compare runs at the same share of their budget; choose the budget before starting |
| Low success, slope near 1, miss 1-3 cm | it looks but is imprecise | compare the miss with the grasp tolerance per object; train longer |
| Slope well below 1/3, R² near 0 | it is not averaging the objects, it is not looking | look for what it reads instead (section 6, section 8) |
| A history policy jumps with `--first_prev_action expert` | it reads the goal from its previous action | drop history |
| A result that matches another run's too well | two evaluations reached one policy server | evaluate through `eval.sh`; rerun what `summarize` leaves out |
| High success, 0% when the object moves | it looks once | random starts and recoveries; evaluate with `--disturb` |

## Next time, in this order

1. The expert through the evaluation loop, and the grasp tolerance (how far off the expert can
   close and still succeed, `sim/pickup/grasp_tolerance.py`), on the evaluating machine.
2. The config: the instruction as tokens rather than one vector? History?
3. One sample through the training and the serving path, the tensors compared (this catches the
   previous-action skew).
4. The position table checked directly (`tests/essential/models/test_rotary_precision.py`); item 3
   cannot see it, since both paths share it.
5. The budget chosen before the run, from a comparable run's transition; every checkpoint
   evaluated, raw and EMA; slope and miss read before success.
6. With history, the two probes of the previous action.
7. Once it works, the probes and disturbances with the expert alongside; then the wanted behaviour
   put into the demonstrations.

## Background and further reading

Background, not measured here:

- de Haan et al., "Causal Confusion in Imitation Learning", 2019.
  [arXiv:1905.11979](https://arxiv.org/abs/1905.11979). A cloned policy can get worse with more
  information.
- Wen et al., "Fighting Copycat Agents in Behavioral Cloning from Observation Histories", 2020.
  [arXiv:2010.14876](https://arxiv.org/abs/2010.14876). Given histories, a policy can learn to copy
  the expert's previous action.
- Laskey et al., "DART: Noise Injection for Robust Imitation Learning", 2017.
  [arXiv:1703.09327](https://arxiv.org/abs/1703.09327). Noise in the expert's demonstrations puts
  recoveries into them; section 9 moves the object instead.

## Check yourself

1. Compute the expert's slope from command 1 and compare it with the pooled slopes of sections 5
   and 6 (`close_slope` pools both axes; section 3 fits each axis separately).

   ```bash
   .venv/bin/python -c "import json; from sim.pickup.summarize import close_slope; print(close_slope(json.load(open('results/pickup/expert/pick_target/results.json'))))"
   ```

2. Simulate the three readings of section 3 on one table axis: a gripper that closes at the named
   object, one that closes at the mean of the three objects, and one that ignores the scene. Which
   slope does each give, and which did the first-round policies show?

   ```bash
   .venv/bin/python -c "
   import numpy as np
   rng = np.random.default_rng(0)
   x = rng.uniform(-0.1, 0.1, size=(10000, 3))  # three objects on one table axis; the first is named
   for how, close in [('at the named object', x[:, 0]), ('at the mean of the three', x.mean(1)), ('ignoring the scene', rng.uniform(-0.1, 0.1, 10000))]:
       print(f'{how}: slope {np.polyfit(x[:, 0], close, 1)[0]:.2f}')
   "
   ```

3. Reproduce the learning-rate table of section 6 from `cosine_lr`, then add a 3M-sample budget.
   At what learning rate does each run reach 1.5M samples?

   ```bash
   .venv/bin/python -c "
   import torch
   from vla_foundry.scheduler import cosine_lr
   opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0)
   for budget in (1_000_000, 5_000_000):
       lr = cosine_lr(opt, 5e-4, 1000, budget // 64, 1e-5, 0.0)
       print(budget, [f'{lr(s // 64):.1e}' for s in (500_000, 1_000_000, 1_500_000, 2_000_000) if s <= budget])
   "
   ```

4. After command 2, find the first checkpoints whose slope exceeds 0.5 and whose success exceeds
   50%. How far apart are they, against the notebook's "The same run before and after the fix"?
   Then fit each axis separately over all of the run's checkpoint evaluations pooled, as section 3
   does, with `_fit` from `sim/pickup/diagnostics.py`:

   ```bash
   .venv/bin/python -c "
   import glob, json
   from sim.pickup.diagnostics import _fit
   eps = [e for f in glob.glob('results/pickup/pt4k/pick_target/ckpt_*/results.json')
          for e in json.load(open(f))['per_episode'] if e.get('close_tcp_xy') and e.get('close_object_xy')]
   for axis in (0, 1):
       print('xy'[axis], _fit([e['close_object_xy'][e['target']][axis] for e in eps], [e['close_tcp_xy'][axis] for e in eps]))
   "
   ```

5. Run the probes of command 4 on your final checkpoint. Closing at the named object with frozen
   cameras, and near the old place after a move at control step 20, means it looks once.
