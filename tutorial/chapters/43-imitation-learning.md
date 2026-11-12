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
instruction; `pick_cube` has one red cube. A window for `dp_vit`, the 36.3M-parameter diffusion
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
demonstration contains. The policies here are generative, and the ones that failed did not fail
this way (What we measured, problem 3).

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
success](91-loss-is-not-success.md)). It prints three lines with one value per joint: the error
over the whole chunk, the error of the chunk's first action, and the copy baseline. `window_rows`
skips every window that is padded where the policy reads or has no action before the current step,
so the first window of each demonstration, the one where a served policy's previous action says
nothing about the target, is never in the comparison. A policy that only beats the baseline narrowly has learned little beyond
problem 4's shortcut.

```bash
.venv/bin/python -m sim.pickup.open_loop_check \
    --checkpoint_dir experiments/pickup/<run> \
    --shards data/preprocessed/<dataset>/shards --samples 64
```

## What we measured

All numbers are from the [lab notebook](../lab-notebook.md), one training seed per run, evaluated on
**layouts** the policy never saw (a layout: the objects, their poses, the named object and the arm's
start pose, fixed by the seed). First-round runs used machine A (one RTX 4090, 24 GB), the rest
machine B (L40S GPUs, 128 CPU cores). Until a fix partway through, the table of rotary position
angles was computed in bfloat16 under autocast and gave many positions the same angles (641 distinct
positions of 2048). Runs trained before and after that fix differ, and
[Positions](12-positions.md) gives the results of both. The results are sorted by the four problems.

**The expert being copied succeeds, with one plan.** Through the evaluation loop on `pick_target`
(machine B, 100 layouts) the scripted expert succeeds 100/100 and closes the gripper 5 mm (median)
from the object. The same expert under the disturbances of `eval.py`, 100 layouts per condition,
counted over the episodes in which the object could be moved 80 mm:

| Condition | One plan (`reactive=False`) | Plans again (`reactive=True`) |
|---|---|---|
| unchanged task | 100/100 | 100/100 |
| random start pose | 100/100 | 100/100 |
| object moved at control step 10 of the reach | 0/100 | 100/100 |
| moved at control step 20 | 0/99 | 96/96 |
| moved at control step 30 | 0/100 | 100/100 |
| moved as the gripper first closes | 0/100 | 100/100 |

The expert closes the gripper around control step 40, so a move at step 10 leaves it about 30 steps
to react. With one plan it never does: `act` does not read the object's position after `reset`.
The default demonstrations are the successful episodes of this expert, so problem 2 is built into
them: no window shows an object somewhere other than where the arm is heading, and none shows a
failed grasp.

**Problems 1 and 2 in the policy.** What the expert never does, the policy does not learn. The
policy that works here is `dp_vit_lang_reactive`: `dp_vit` plus the instruction as tokens, without
history, 48.9M parameters. Trained on 4000 demonstrations for 5M samples and evaluated at its final
checkpoint on 100 layouts per condition, it scores no more than 18/100 with the object moved when
the demonstrations are clean, before or after the position fix; trained on demonstrations collected
with `--random_start --disturb_prob 0.5`, it holds 89-94% under every disturbance
([Self-correction](103-case-self-correction.md); notes: "Self-correction has to be in the data"). Chunking does not stand in for the missing data: executing fewer actions of
each chunk did not help the clean-data policy ([Chunks and execution](42-chunks-and-execution.md)).

**Problem 3 did not occur.** The first-round policies that failed on machine A (`dp_vit`, and
`vla_custom`, the 130M VLA built from scratch here and conditioned on one vector; 1500
demonstrations, 1M samples, 0-1/100) ran the whole reach-descend-close-lift at the right height and
time, at a place that did not depend on where the named object was. The measure is the **slope** of
where the gripper first closed against where the named object was: 1 when it closes at the object,
0 when where it closes does not depend on the scene. They did not average two reaches; they had no
target ([Measuring failure](92-measuring-failure.md); notes: "A failing policy ignores the scene; it
does not average").

**Problem 4: how much the past gives away.** On `pick_cube` (machine A) the served policies
reproduced the expert's chunks on recorded windows to within 0.01-0.04 rad per joint, and repeating
the previous action came within 0.05-0.2 rad: most of the next action is already in the last one.
Arithmetic on the 98,841 windows of `pick_target_1500`: 1500 of them (1.5%) start at a
demonstration's first frame, and 8 x 1500 = 12,000 (12.1%) start in frames 0-7, the windows
`boost_early_windows.py --max_frame 7` repeats. The notes find that only these first moments need
the scene; in every later window the arm's own motion says where it is going. So few windows reward
looking, which the notes ("Looking at the scene arrives late, and all at once") give as the reason
the slope of a policy here stays near 0 for a long time and then rises quickly ([How much data and
training](71-how-much-data-and-training.md)).

Both shortcut inputs were caught. With the previous action as an input, `dp_vit_lang` (the same
policy with history, 48.9M parameters; trained on three objects before the position fix) reads its
target from that action: it fails as served and does far better when the first previous action is
the expert's or the expert drives the first control steps. After the fix, demonstrations that begin
with a hold frame did not cure it ([History and shortcuts](52-history-and-shortcuts.md)). Without
the previous action, the arm's joint state remains. The policy trained on clean demonstrations from the home pose
decides where to go at its first look and then follows its own arm: with its cameras frozen on the
first frame it still closes at the named object, and an object moved during the reach is missed
([What a policy looks at](95-what-a-policy-looks-at.md)). Random start poses take that guide away,
and the policy then watches the scene while it reaches ([Which demonstrations
teach](72-which-demonstrations-teach.md)).

## Where it goes wrong

| What you see | Why | How to check | What to do |
|---|---|---|---|
| A complete reach-descend-close-lift at the wrong place | The policy has not started to look, or it reads its target from an input that differs when served | The Slope column of `.venv/bin/python -m sim.pickup.summarize`: near 0 it is not looking, near 1 it is imprecise | Evaluate every checkpoint and train past the point where a comparable run started to look. The recipe that worked: `dp_vit_lang_reactive`, 4000 demonstrations collected with `--random_start --disturb_prob 0.5`, 5M samples (the config saves averaged weights itself) |
| Low open-loop error, low closed-loop success | Repeating the previous action already fits most windows, and the check skips the first window of each demonstration, where it does not | `open_loop_check.py`: the gap between the chunk error and the "repeat previous action" baseline | Select checkpoints by closed-loop evaluation; read the open-loop check only as a test that the model beats the baseline at all |
| A policy with history stays near 0% | It takes the target from its previous action | `--first_prev_action expert` and `--expert_prefix 10`: far better than as served | Train without history; `collect.py --hold_steps 1` did not cure it ([History and shortcuts](52-history-and-shortcuts.md)) |
| The gripper closes where a moved object was, or beside the object, and the arm lifts empty | No clean demonstration contains a moved object or a miss followed by a correction | `--disturb reach`, `--disturb grasp`; `grasp.retried_rate` in `results.json` counts second grasp attempts | Collect with `--random_start --disturb_prob 0.5` and evaluate with the same disturbances |

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
