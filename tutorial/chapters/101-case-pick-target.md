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
height (within about 3 mm), around control steps 34-44, as the expert does. They ran the whole
motion, but a median 174-185 mm from the named object and 89-110 mm from the nearest, and closed at
the named one 29-33% of the time: chance.

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
1/3 against the named one (arithmetic, not a measurement). These are at most 0.15, and R² (the
share of the closing position's variation the object's position explains) at most 0.021. The
miss is the same against another episode's objects, and the closing positions fill the area where
objects are placed (field notes, "A failing policy ignores the scene; it does not average"). The
question became "why does it not look" ([Measuring failure](92-measuring-failure.md)).

### 4. Four candidate causes

| Candidate | How it was tested |
|---|---|
| No language input (`dp_vit.yaml` has no `text_tokens`) | needed by construction: without it, one object in three at best; section 5 adds it |
| One conditioning vector (the VLAs) | tested after the position fix with every token handed over (`conditioning: all_tokens`), no history, 4000 demonstrations: from-scratch VLA 86% / 87%, Qwen 70% / 72%; history, data, budget (3M and 0.5M samples) and the position table changed too (field notes, "One pooled vector is not enough for grasping"). `dp_vit` had patch tokens, so not its cause |
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
same 1500 for 3M (30.4 epochs), every checkpoint evaluated.

**Measured.** Notebook, "4000 demonstrations, no history":

| Samples | Success (raw / EMA) | Closed at the named object | Slope | Median miss |
|---|---|---|---|---|
| 1.0M | 0% / 1% | 35% / 32% | 0.11 / 0.14 | 138 / 154 mm |
| 1.5M | 4% / 2% | 61% / 49% | 0.47 / 0.37 | 102 / 104 mm |
| 2.0M | 4% / 36% | 95% / 98% | 0.81 / 0.96 | 32 / 18 mm |
| 5.0M (final) | 83% / 85% | 100% / 100% | 0.98 / 0.98 | 7 / 7 mm |

Between 1.5M and 2.5M comes the **transition**, the stretch of training in which the slope goes
from near 0 to near 1: slope first, then miss, then success. The 1500-demonstration run did the
same, earlier: 21% / 18% at 1.5M. The learning-rate schedule (a cosine from 5e-4 down to
1e-5) is stretched over each run's own budget, so the 1M run of section 5 reached the transition
with its learning rate already near zero. Choose the budget before starting; resuming a finished
short run does not give the long one ([How much data and training](71-how-much-data-and-training.md)).

History was tested with the twin `dp_vit_lang` on the same 4000 demonstrations: 2% / 2% after
3.0M samples (81% / 75% without history). Its 2.5M checkpoint (EMA) is not blind; the **hold
command** keeps the arm where it is, and the robot is under it before it moves
(notebook, "What the history policy depends on"):

| At the start of the episode | Success | Closed at the named object |
|---|---|---|
| as served: the previous action is the hold command | 1/100 | 30% |
| the previous action is the expert's first command | 22/100 | 97% |
| the expert drives the first 10 control steps | 77/100 | 100% |

It reads the target from its previous action. A window that starts at a demonstration's first frame
fills that slot with a copy of the expert's first command; served, it holds the hold command: a
train/inference skew ([History and shortcuts](52-history-and-shortcuts.md),
[Train/inference skew](93-train-inference-skew.md)).

These runs predate the position fix of section 7, which hit the history policy harder (16
action-step tokens on 4 positions); on the single cube, history costs nothing once the table is
fixed (96% / 98%). After the fix, a history policy trained on demonstrations that begin with one
frame of the robot at rest (`collect.py --hold_steps 1`, closing the first-frame leak) ends at
0% / 4% after 5M samples; where it closes never depends on the scene (slope -0.11 to 0.12), since
from the second step on the previous action still predicts the next one (field notes, "The previous
action as an input"). A history run on the plain demonstrations was not repeated after the fix.

<!-- interactive: onset -->
*Figure: success, closed at the named object and slope against training samples for these runs;
the history run stays flat.*

**Changed.** The recipe became 4000 demonstrations and 5M samples: 83% / 85% against 64% / 62%
with 1500 and 3M. Data and budget both differ: this ranks the recipes, not the effect of data.

### 7. Two detours

**Results under the wrong name.** Two evaluations sharing a local port both reached the policy
server bound to it first, filing about 40 results under the wrong run; they were redone. `eval.sh`
now takes a free port and passes `eval.py --expect_run`, which stops unless the server serves the
run (and checkpoint) asked for.

**The position table in half precision.** The position table holds the rotary angles precomputed
for each token position ([Positions](12-positions.md)). `RotaryEmbedding._update_cos_sin_tables` in
[`vla_foundry/positional_embedding.py`](../../vla_foundry/positional_embedding.py) computed it in
bfloat16 under autocast. With 8 significant bits, integers above 256 round to even
numbers, above 512 to multiples of 4 and above 1024 to multiples of 8: 641 distinct positions of 2048. Without history the 15
**action-step tokens** (one per predicted action) sit at positions 426-440, after 1 noise-level
token, 32 instruction tokens, 392 patch tokens and 1 state token: 8 distinct positions. With history
about 820 tokens come first and a 16th action-step token holds the previous action: 16 on 4
positions. Training and serving shared the table, so nothing looked wrong; every run in sections
1-6 used it, except the every-token VLAs (section 4) and the hold-frame and single-cube history
runs (section 6). It is now computed in float32; `positional_embedding_type: rotary_legacy_autocast`
reproduces the old one.

After the fix, same recipe, trained weights, before in brackets (notebook, "The same run before
and after the fix"): 8% at 1.0M samples (0%); 58% at 1.5M (4%); 93% at
5.0M (83%; T block 20/23 against 15/23). What sections 3-6 concluded still holds, but the
transition comes earlier, mostly between 1.0M and 1.5M instead of 1.5M and 2.5M. A second pair with
other seeds, differing in the table alone, agrees: closed at the named object after 1.0M samples 83%
against 44% (trained weights), and 96% / 96% at the end against 90% / 89% (notebook, "A second seed,
with and without the fix"). Final gaps (trained weights): 10 and 6 points.

### 8. What the probes showed: it looks once

**Believed.** That the 85% policy, without history, watches the scene.

**Measured.** Probes of the 85% checkpoint (before the fix, EMA)
([What a policy looks at](95-what-a-policy-looks-at.md); notebook, "What the no-history policy
depends on", "Disturbances"). Objects are moved 80 mm; the expert closes around control step 40:

| Probe | Success | Where it closed |
|---|---|---|
| cameras frozen on the first frame | 1/100 | at the named object in 99%, 17 mm off |
| expert drives 25 control steps toward another object | 0/100 | at that object in 100/100 |
| object moved at control step 20 | 0/96 | 8 mm from the old place, 78 mm from the new |

<!-- interactive: disturbance -->
*Figure: a top-down replay of a reach during which the named object is moved; the gripper closes
at the old place.*

With frozen cameras it misses by 17 mm instead of 7: later images only refine the last
centimetres. It decides from its first look, then follows its own arm, whose straight reach
from one home pose gives the end away. The policy trained after the fix does the same (frozen
cameras 1/100; the "clean" row of section 9).

### 9. What changed that: the demonstrations

After the fix, same policy, 4000 demonstrations, 5M samples, only the data changed (notebook, "What
the demonstrations contain"). Repeating the windows of the first 8 frames 6 times (the frames where
the scene alone says where to go) ends at 96%. Two changes add what plain demonstrations lack:

- **Random start poses** (`collect.py --random_start`): the arm's position no longer gives the goal
  away: 98% / 98%, within noise of the 96%.
- **Recoveries** (`collect.py --disturb_prob 0.5`): in 1505 of 4000 episodes the object is moved
  5-10 cm once, during the reach (control steps 5-29) or as the gripper closes (`Disturbance`), and
  the expert plans again. 89% / 84%.
- **Both** (`collect.py --random_start --disturb_prob 0.5`; the object is moved once in 1456
  episodes): 94% / 94%.

Final checkpoints, averaged weights where saved (notebook, "How each policy copes outside the
conditions it was trained in"):

| Trained on | Standard | Random start | Moved at control step 10 | 20 | 30 | Moved at the grasp |
|---|---|---|---|---|---|---|
| clean | 93/100 | 9/100 | 12/100 | 0/97 | 0/100 | 0/100 (8 retried) |
| random starts | 98/100 | 97/100 | 94/100 | 50/98 | 0/100 | 0/100 (2 retried) |
| recoveries | 84/100 | 68/100 | 88/100 | 79/96 | 85/100 | 78/99 (99 retried) |
| random starts and recoveries | 94/100 | 91/100 | 92/100 | 89/98 | 94/100 | 89/100 (100 retried) |

With one plan per episode the scripted expert fails every moved-object column; planning again, it
succeeds in all. Random starts make the policy watch early in the reach only; only recoveries in
the data make it follow a late move and try again.
Recoveries alone cost something on the unchanged task (84/100 here, 93/100 on a second set of
layouts, the can at 77% against 90% clean). With random start poses as well, 94/100 on the first set, as
high as clean; on the second set 87/100, below recoveries alone (93/100) and clean (96/100). It
holds 89-94% under every disturbance, also from a random start pose (field notes,
"Self-correction has to be in the data"): the recipe `README.md` recommends. One seed each; why
each helps: [What a policy looks at](95-what-a-policy-looks-at.md).

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
# the other columns of section 9: --disturb_step 10 and --disturb_step 30 in place of 20
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
```

## Where it goes wrong

| What you see | Why | What to do |
|---|---|---|
| 0% after 1M samples | the flat stretch before the transition | read the slope by checkpoint; judge only after a comparable run's transition |
| Low success, slope near 1, miss 1-3 cm | it looks but is imprecise | compare the miss with the grasp tolerance per object; train longer |
| A history policy jumps with `--first_prev_action expert` | it reads the goal from its previous action | drop history |
| High success, 0% when the object moves | it looks once | random starts and recoveries; evaluate with `--disturb` |

## Next time, in this order

1. The expert through the evaluation loop, and the grasp tolerance (how far off the expert can
   close and still succeed, `sim/pickup/grasp_tolerance.py`), on the evaluating machine.
2. The config: the instruction as tokens rather than one vector? History?
3. One sample through the training and the serving path, the tensors compared (this catches the
   previous-action skew).
4. The position table checked directly (`tests/essential/models/test_rotary_precision.py`); item 3
   cannot see it, since both paths share it.
5. Every checkpoint evaluated, raw and EMA; slope and miss read before success.
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

2. After command 2, find the first checkpoints whose slope exceeds 0.5 and whose success exceeds
   50%. How far apart are they, against sections 6 and 7?

3. Run the probes of command 4 on your final checkpoint. Closing at the named object with frozen
   cameras, and near the old place after a move at control step 20, means it looks once.
