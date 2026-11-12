---
title: Imitation Learning and Why It Breaks
part: 4 Actions and policies
order: 43
summary: Explain why a policy that copies demonstrations accurately can still fail when it runs, name the four causes, and find each one in this repository's expert, data and measurements.
status: draft
---

# Imitation Learning and Why It Breaks

Every policy in the lab notebook is trained by imitation: it is shown recordings of an expert doing
the task and trained to output what the expert output. That is ordinary supervised learning until
the policy is switched on. From then on its own outputs decide what it sees next, and four
problems appear that a held-out loss cannot show.

## The idea

**Behaviour cloning** is supervised learning on demonstrations. A **demonstration** is a recording
of an expert performing the task: at every control step, the observation (camera frames, joint
state) and the action the expert commanded. A **policy** is a function from observation to action;
behaviour cloning fits it to the recorded pairs.

The lab notebook lists 1500 `pick_target` demonstrations of about 66 control steps each, cut into
98,841 **windows**: one training example per recorded step, the observations at that step plus
the actions that follow. `pick_target` puts 3 of 4 objects on the table and names one in the
instruction; `pick_cube` has one red cube. A window for `dp_vit`, the 36M-parameter diffusion
policy with a small vision transformer and no language input, holds:

| Part | Shape | Content |
|---|---|---|
| Images | 4 frames of 224x224 | `front` and `wrist` cameras at control steps t-1 and t |
| State | `[2, 8]` | 7 joint angles and the gripper opening at t-1 and t |
| Previous action | `[1, 8]` | the command the expert sent at t-1 |
| Target | `[15, 8]` | the **chunk**: the commands the expert sent at t, t+1, ..., t+14 |

The frames and the state at t-1 and the previous action are the policy's **history**.

Supervised learning assumes test inputs come from the training distribution. Held-out expert windows
do; a running policy does not: the observation at control step t+1 results from the action at step
t, so its inputs come from the states *the policy* reaches, which are the expert's only until its
first error.

Running the policy in the simulator is a **rollout**, and success measured this way is
**closed-loop**. Comparing its output on recorded observations with the recorded actions is
**open-loop**: its outputs are never executed.

## How it breaks

### 1. The policy visits states its own mistakes create

**Covariate shift** is a change in the distribution of inputs between training and use, while the
correct output for each input stays the same. In behaviour cloning the policy causes the shift
itself. A small error moves the arm off the demonstrated path; the next observation is unlike the
data, so the next prediction is worse, and the arm moves further off. This is **compounding error**.

Background: [Ross, *Efficient Reductions*, 2010](https://proceedings.mlr.press/v9/ross10a.html)
bounded the extra cost by εT², for a policy that errs on a share ε of expert states over T control
steps. Arithmetic: with ε = 1% and T = 66 the bound is 0.01 x 66² = 43.56 steps, against 0.01 x 66 =
0.66 errors if errors did not compound.

### 2. The demonstrations never show recovery

Compounding error would be harmless if the policy knew how to return to the demonstrated path.
It learns that only from states off the path paired with the action that leads back, and a
competent expert does not produce such states: after the gripper closes beside the object, the
correction (open, move over, descend, close again) follows a state no clean demonstration contains.

### 3. Demonstrations disagree with each other

The same observation can be followed by different correct actions: reach for the left object or the
right one. Their distribution then has several peaks: it is **multimodal**. A model trained to
minimise squared error outputs the mean of the targets. Arithmetic: with two objects 20 cm apart and
demonstrations split evenly between them, the mean reach ends 10 cm from both, an action no
demonstration contains. The policies here are generative; the failing ones did not do this.

### 4. Inputs that predict the action without causing it

Robot motion is smooth, so the expert's command at control step t is almost its command at step t-1.
If the previous action is an input, "repeat it, and go a little further the same way" fits nearly
every training window without the cameras; only the first windows of a demonstration need the scene.
**Proprioception**, the robot's measurement of its own joints, is such an input too: when every
demonstration starts from nearly the same pose and reaches in a straight line, as here, one state a
few control steps in says where the reach ends.

The expert's motion was caused by what the expert saw; in a rollout the motion is the policy's
own, and a policy that leans on it continues what it is doing, right or wrong. Background:
[de Haan, *Causal Confusion*, 2019](https://arxiv.org/abs/1905.11979) named this *causal
confusion*; [Wen, *Fighting Copycat Agents*, 2020](https://arxiv.org/abs/2010.14876) called the
case of the previous action the *copycat problem*. Here that action is given to the policy
directly.

<!-- interactive: copycat -->
*Figure: a demonstration's joint target and gripper command against time, beside a line that
repeats the previous action; the gap is what that action does not predict.*

### The classical remedies

Background, not measured here.

| Remedy | What it changes | Aimed at |
|---|---|---|
| DAgger | Run the policy, have the expert label the states it reached, retrain on them, repeat. Needs an expert that can be queried in any state | 1, 2 |
| Noise injection (DART) | Perturb the expert's executed actions during collection and record the intended ones, so the data holds off-path states with corrections | 1, 2 |
| Action chunking (ACT) | Predict several future actions per decision, so an episode contains fewer decisions | 1 |
| Generative policy (Diffusion Policy) | Learn the distribution of actions and draw from it instead of regressing to its mean | 3 |
| History dropout (ChauffeurNet) | Hide the past-motion input for half of the training examples | 4 |

## In this repository

**By default the expert makes one clean pass.** In
[`sim/pickup/expert.py`](../../sim/pickup/expert.py), `ScriptedExpert._plan`, called from `reset`,
reads the target's position and builds four waypoints: above the grasp point, at it, the same point
with the gripper closed, then a lift. Its random draws ([Demonstrations](70-demonstrations.md)) vary
the plan (a mild problem 3), never take the arm off it. With the default `reactive=False`, `act`
only ever increases the waypoint index `self.wp`: it never reads the object's position again and
never reopens the gripper. With `reactive=True`, `_needs_new_plan` runs before every action and
`_plan` is called again when the named object lies more than 15 mm from where it was planned for, or
when the arm is lifting and the fingers have met on nothing.

**Collection records what it executes and keeps only successes.** In
[`sim/pickup/collect.py`](../../sim/pickup/collect.py), `_worker` stores the action from
`expert.act()` and passes the same action to `env.step`; that is where DART would add noise. An
episode whose final `info["success"]` is false is skipped. By default the data therefore contains no
missed grasp, nothing in a scene moves but the arm, and every reach starts near the home pose.

**The shortcut inputs exist by default.**
[`configs/preprocess.yaml`](../../sim/pickup/configs/preprocess.yaml) cuts windows with
`past_lowdim_steps: 1`. In `DiffusionPolicy.forward`
([`diffusion_policy.py`](../../vla_foundry/models/diffusion_policy/diffusion_policy.py)) the line
`torch.where(future_mask.unsqueeze(-1), noisy_action, actions)` leaves the past slot of the action
sequence as the recorded, noise-free expert action. In a rollout,
[`eval.py`](../../sim/pickup/eval.py) sends `env.last_action`, the command the simulator last
applied, and `Policy.infer` in [`policy_server.py`](../../sim/pickup/policy_server.py) writes it
into that slot.

The first control step differs. An episode's first window has no t-1, and `padding_strategy: copy`
fills the slot with a copy of the expert's first command, which already points at the target
(`forward` does not use `past_mask`); `eval.py` sends the hold command set by `PickupEnv.reset`,
the start joint angles with the gripper open ([Train/inference skew](93-train-inference-skew.md)).

**Which remedies are present.**

| Remedy | Here |
|---|---|
| Action chunking | 15 actions per chunk, of which `eval.py` executes `--exec_horizon` (default 8) ([Chunks and execution](42-chunks-and-execution.md)) |
| Generative policy | [Flow matching](41-flow-matching.md) |
| History removal | [`dp_vit_reactive.yaml`](../../sim/pickup/configs/dp_vit_reactive.yaml) and [`dp_vit_lang_reactive.yaml`](../../sim/pickup/configs/dp_vit_lang_reactive.yaml) set `lowdim_past_timesteps: 0` and `image_indices: [0]`: one frame per camera, the current state, no previous action (not dropout). The arm's position remains an input |
| Recovery demonstrations (problem 2) | `collect.py --disturb_prob P`: in about a share P of the episodes [`disturb.py`](../../sim/pickup/disturb.py) moves the named object once, during the reach or as the gripper closes, and the expert (`reactive=True`) plans again |
| Varied start poses (problem 4) | `collect.py --random_start`: every episode starts from a random gripper pose above the table |
| Hold frame (problem 4) | `collect.py --hold_steps 1`: every episode begins with a frame of the robot at rest, so the first window's previous action is the hold command |
| DAgger, DART | Not implemented; [`roadmap.md`](../roadmap.md) lists noise-injected demonstrations as open |

**The open-loop check.** [`sim/pickup/open_loop_check.py`](../../sim/pickup/open_loop_check.py)
gives the supervised-learning view of a policy, with or without history: its per-joint error against
the expert's actions on recorded windows (a policy with history is given the expert's previous
action), next to the error of repeating that previous action ([Loss is not
success](91-loss-is-not-success.md)).

```bash
.venv/bin/python -m sim.pickup.open_loop_check \
    --checkpoint_dir experiments/pickup/<run> \
    --shards data/preprocessed/<dataset>/shards --samples 64
```

## What we measured

All numbers are from the [lab notebook](../lab-notebook.md): first-round runs on machine A (one
RTX 4090, 24 GB), second-round runs on machine B (L40S GPUs, 128 CPU cores), one training seed per
run, on **layouts** the policy never saw (a layout: the objects, their poses, the named object
and the arm's start pose, fixed by the seed). Paired numbers: trained weights (raw) /
their running average (EMA). Runs not marked "after the position fix" used the
old rotary position table, computed in half precision ([Positions](12-positions.md)); after the
fix, the 83% / 85% recipe below ends at 93% (raw).

**The failing policies copied the motion and did not look.** On `pick_target` (machine A, 1500
demonstrations, 1.0M samples, 100 episodes each), `dp_vit` scored 1/100 and `vla_custom`, the
130M-parameter vision-language-action policy of [The from-scratch
path](62-the-from-scratch-path.md), 0/100. Both closed the gripper in 100% of episodes, at the right
height and time, as the expert does. They **closed at the named object** (the object nearest the
gripper when it first closed was the one the instruction named) in 32% and 33% of episodes (chance
is 33%), a median of 110 mm and 89 mm from the nearest object. `dp_vit`'s loss (0.069) did not show
it: on `pick_cube` (1000 demonstrations, 1.5M samples) it ended at 0.058 and scored 37/50 = 74%
([Loss is not success](91-loss-is-not-success.md)).

This is not problem 3. The [field notes](../notes.md) ("A failing policy ignores the scene; it does
not average") found that these policies did not reach between the objects: where they closed did not
depend on the scene. `sim/pickup/summarize.py` prints the **slope** of where the gripper first
closed against where the named object was: 1 if it closes where the object is, 0 if the place does
not depend on the scene ([Measuring failure](92-measuring-failure.md)). In the three runs that never
started to look (machine B, 1.0M samples on 1500 demonstrations, all evaluations pooled) it lay
between 0.00 and 0.09; the fourth, `dp_vit_lang_reactive`, ended at 0.29 / 0.18, at the start of its
transition.

**Looking at the scene arrived late.** `dp_vit_lang_reactive` is `dp_vit` with the instruction as
tokens ([Language conditioning](51-language-conditioning.md)) and without history. After 1.0M
samples on 1500 demonstrations it scored 3% / 5%; after 3.0M samples on the same demonstrations, 64%
/ 62% (slope 0.96 / 0.97); after 5.0M samples on 4000 demonstrations, 83% / 85% (0.98 / 0.98),
having been at 0% / 1% after 1.0M and at 4% / 36% after 2.0M. At 1.0M samples the policy that later
worked had barely started to look ([How much data and training](71-how-much-data-and-training.md),
[the `pick_target` case study](101-case-pick-target.md)).

**With history, the policy takes its target from the previous action.** `dp_vit_lang` is the same
policy with history. On the same 4000 demonstrations it was at 2% / 2% after 3.0M samples, where
the run was stopped; the run without history was then at 81% / 75%. It is not blind, as two
**probes** show: evaluations that change one thing about an episode. Its checkpoint at 2.5M
samples (EMA), 100 layouts each:

| At the start (`eval.py` option) | Success | Closed at the named object |
|---|---|---|
| as served: the first previous action is the hold command | 1/100 | 30% |
| `--first_prev_action expert`: it is the expert's first command | 22/100 | 97% |
| `--expert_prefix 10`: the expert drives the first 10 control steps | 77/100 | 100% |

In training the previous-action slot always holds an expert command, the first window included;
served, the first query carries the hold command, which says nothing about the target. The policy
learned to read where to go from its own past ([History and
shortcuts](52-history-and-shortcuts.md)). After the position fix, a history policy trained on
demonstrations that begin with a hold frame (`--hold_steps 1`) ends at 0% / 4% after 5M samples,
with slopes between -0.11 and 0.12: closing the leak at the first frame did not make it look. A
history run on plain demonstrations was not repeated.

**Without history, the policy looks once.** The 85% policy (`dp_vit_lang_reactive` at 5.0M
samples, EMA) closes at the named object in 100% of episodes, 7 mm (median) from it. Probed, 100
layouts per row (fewer where the object could not be moved):

| Probe (`eval.py` option) | Success | The gripper (distances are medians) |
|---|---|---|
| `--stale_images`: the first frame's images throughout | 1/100 | closed at the named object in 99%, 17 mm from it |
| `--expert_prefix 25 --prefix_target distractor`: the expert drives the first 25 control steps toward another object | 0/100 | closed at that other object in 100/100 |
| `--disturb reach --disturb_step 20`: the named object is moved 80 mm at control step 20 | 0/96 | closed 8 mm from where the object had been |
| `--disturb grasp`: it is moved as the gripper first closes | 1/99 | closed a second time in 29/99 |

The field notes' reading ("Success is not the same as looking"): it decides where to go from its
first look, then follows its own arm, whose joint state says where a straight reach from the home
pose ends. Nothing in the demonstrations contradicts that: nothing in them moves and no grasp fails
(problems 4 and 2) ([What a policy looks at](95-what-a-policy-looks-at.md)). The scripted expert
with its one plan fails this too (0/100 with the object moved at the grasp); planning again, it
succeeds 100/100.

**Deciding more often did not help.** Executing 2, 4, 8 or 15 actions of each chunk
(`--exec_horizon`) gave the 85% policy 76/100, 84/100, 85/100 and 84/100.

**Recoveries and varied starts in the data did.** The same policy after the position fix, 4000
demonstrations collected with the options above, 5M samples (notes: "Self-correction has to be in
the data"). With `--random_start`: 98% / 98%, 97% from a random start pose, 94% with the object
moved at control step 10, but 0% moved at step 30 or at the grasp. With `--disturb_prob 0.5`:
89% / 84%, 79-88% with the object moved, a second grasp attempt in 99 of 99, but 68% from a
random start pose. With both: 94% / 94%, and 89-94% under every disturbance, also from a random
start pose. A behaviour the demonstrations never show does not appear in the policy.

## Where it goes wrong

| What you see | Why | How to check | What to do |
|---|---|---|---|
| A complete reach-descend-close-lift at the wrong place | The policy has not started to look, or it reads its target from an input that differs when served | The Slope column of `.venv/bin/python -m sim.pickup.summarize`: near 0 it is not looking, near 1 it is imprecise | Evaluate every checkpoint and train past the point where a comparable run started to look. The recipe that worked: `dp_vit_lang_reactive`, 4000 demonstrations collected with `--random_start --disturb_prob 0.5`, 5M samples (the config saves averaged weights itself) |
| A policy with history stays near 0% | It takes the target from its previous action | `--first_prev_action expert` and `--expert_prefix 10`: far better than as served | Train without history. `collect.py --hold_steps 1` did not cure it (0% / 4% after 5M samples) |
| The gripper closes where a moved object was, or beside the object, and the arm lifts empty | No clean demonstration contains a moved object or a miss followed by a correction | `--disturb reach`, `--disturb grasp`; `grasp.retried_rate` in `results.json` counts second grasp attempts | Collect with `--random_start --disturb_prob 0.5`: 89-94% under every disturbance |

## Background and further reading

- Ross, *Efficient Reductions for Imitation Learning*, 2010
  ([PMLR](https://proceedings.mlr.press/v9/ross10a.html)). The εT² bound.
- Ross, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*,
  2011 ([arXiv:1011.0686](https://arxiv.org/abs/1011.0686)). DAgger.
- Laskey, *DART: Noise Injection for Robust Imitation Learning*, 2017
  ([arXiv:1703.09327](https://arxiv.org/abs/1703.09327)). Noise injected during collection.
- de Haan, *Causal Confusion in Imitation Learning*, 2019
  ([arXiv:1905.11979](https://arxiv.org/abs/1905.11979)). More inputs can make a cloned policy
  worse.
- Wen, *Fighting Copycat Agents in Behavioral Cloning from Observation Histories*, 2020
  ([arXiv:2010.14876](https://arxiv.org/abs/2010.14876)). The copycat problem.
- Bansal, *ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst*, 2018
  ([arXiv:1812.03079](https://arxiv.org/abs/1812.03079)). Perturbed demonstrations, past-motion
  dropout.
- Zhao, *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, 2023
  ([arXiv:2304.13705](https://arxiv.org/abs/2304.13705)). Action Chunking with Transformers (ACT).
- Chi, *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, 2023
  ([arXiv:2303.04137](https://arxiv.org/abs/2303.04137)). Generated actions for multimodal targets.

## Check yourself

1. Run the scripted expert with the object moved as its gripper closes, with one plan and
   planning again. Using `ScriptedExpert.act` and `_needs_new_plan`, state what the arm does after
   the failed close in each run.

   ```bash
   .venv/bin/python -m sim.pickup.eval --task pick_target --episodes 100 --expert \
       --disturb grasp --out results/pickup/expert/pick_target/grasp_one_plan
   .venv/bin/python -m sim.pickup.eval --task pick_target --episodes 100 --expert --expert_reactive \
       --disturb grasp --out results/pickup/expert/pick_target/grasp_plans_again
   ```

2. Open `sim/pickup/configs/dp_vit_lang_reactive.yaml` and name the two keys that remove the
   history. Which shortcut input of problem 4 do they leave in place?

3. Probe a policy trained with history (`dp_vit_lang`). Compare `successes` and
   `grasp.closed_nearest_is_target_rate` in the two `results.json` files with the evaluation as
   served, and say where the policy gets its target.

   ```bash
   PICKUP_OUT=results/pickup/<run>/pick_target/first_expert \
     sim/pickup/scripts/eval.sh experiments/pickup/<run> pick_target 100 --first_prev_action expert
   PICKUP_OUT=results/pickup/<run>/pick_target/prefix10 \
     sim/pickup/scripts/eval.sh experiments/pickup/<run> pick_target 100 --expert_prefix 10
   ```
