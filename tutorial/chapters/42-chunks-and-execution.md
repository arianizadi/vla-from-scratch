---
title: Action Chunks and Closed-Loop Execution
part: 4 Actions and policies
order: 42
summary: Explain why a policy predicts 15 future actions and executes only the first few, trace one re-plan through eval.py and policy_server.py, and choose and test an execution horizon.
status: draft
---

# Action Chunks and Closed-Loop Execution

An **action chunk** is the sequence of future actions a policy predicts in one go, instead of one
action per observation: the policies trained on this repository's pickup simulator predict 15
actions (0.75 s at 20 Hz), execute the first 8 by default, read the cameras again and predict a new
chunk. A robot policy needs this because it runs **closed-loop**: each action changes what the
cameras see next, so small errors feed back into the input. Asking for a new chunk more often turns
out not to be the same as looking more often.

## The idea

The simplest loop is: observe, predict one action, execute it, repeat. At 20 Hz that is 20
decisions per second, and it has three costs.

1. Every decision is a chance to drift. Background: Zhao et al. (2023) argue that executing a whole
   chunk before the next decision divides the number of decisions in a task (the effective horizon)
   by the chunk length.
2. A generative policy draws its output from a distribution
   ([41-flow-matching.md](41-flow-matching.md)). If two motions are both valid, two draws for
   nearly the same observation can pick different ones; one chunk is a single coherent motion.
3. One forward pass per control step is expensive when the model is large.

A chunked policy predicts $H$ actions and executes the first $k$ of them; $k$ is the **execution
horizon**. The other $H - k$ are thrown away and a new chunk is predicted from a fresh observation
(a **re-plan**), so the number of decisions falls by a factor of $k$. Between two re-plans the
robot follows the chunk without looking.

In this repository's simulator (20 Hz control, $H = 15$, episodes of at most 200 control steps):

| Actions executed per chunk (`--exec_horizon`) | Time between observations | Chunks in 200 control steps |
|---|---|---|
| 1 | 0.05 s | 200 |
| 4 | 0.2 s | 50 |
| 8 (default) | 0.4 s | 25 |
| 15 | 0.75 s | 14 |

<!-- interactive: action-chunks -->
*Figure: a timeline of control steps. At each re-plan the policy draws 15 future actions; 8 are
executed and 7 discarded when the next chunk replaces them.*

## How it works

### The training window

A policy learns to predict chunks because each training example is a **window**: the observations
at one frame of a demonstration (one recorded control step) plus the actions that follow.
`sim/pickup/configs/preprocess.yaml` sets `past_lowdim_steps: 1` and `future_lowdim_steps: 14`, so
a window anchored at frame $t$ holds 16 **low-dimensional** rows (the 8-D state and the 8-D action,
as opposed to images), and `image_indices: [-1, 0]` stores each camera's image at $t-1$ and $t$.

```text
slot            0       1       2       3     ...    15
frame          t-1      t      t+1     t+2    ...   t+14
action         given    |------- predicted: 15 actions -------|
state          input   input    (not an input)
images         t-1      t
```

Slot 1 is the **anchor**, the current frame. `action[t]` is the command applied after observing
frame `t`, so slot 1 is the first action the policy has to produce.

Windows near the ends of a demonstration do not fit. In one of 66 frames (0 to 65):

- Anchor 0 has no frame $-1$. `max_padding_left: 1` allows one missing past row, and
  `padding_strategy: copy` fills it with a copy of frame 0: slot 0 then holds the first command of
  the expert, the scripted controller that produced the demonstration.
- Anchor 65 has no future. `max_padding_right: 14` allows all 14 future rows to be copies of frame
  65.

`future_mask` marks the real rows from the anchor onward. With these limits every frame is an
anchor (`stride: 1`), which is why 1500 demonstrations of `pick_target` (three objects on the
table, one named by the instruction), about 66 frames each, give 98,841 windows.

In training, `DiffusionPolicy.forward` (the model class of the `dp_vit*` configs) adds noise only
to the rows where `future_mask` is true, and the loss, `masked_mse_loss`, counts only those. Slot 0
therefore enters the model clean, as an input, and padded rows cost nothing.

### The run-time loop

`sim/pickup/eval.py` owns the simulator and a queue of actions; `sim/pickup/policy_server.py` owns
the model. One episode:

1. `env.reset(seed)` builds the seed's **layout** (objects, their places, the one the instruction
   names, the start pose) and returns the first observation; `prev_obs` is set to it and the queue
   is empty.
2. When the queue is empty, the client sends an `infer` message: both cameras' images from
   `prev_obs` and the current observation (JPEG bytes), the two states (`[2, 8]`), the last command
   the simulator applied (`env.last_action`, `[8]`), the instruction, the number of flow steps
   (`--flow_steps`, default 10, [41-flow-matching.md](41-flow-matching.md)) and a `noise_seed`:
   `None`, or the layout's seed with `--noise episode`.
3. `Policy.infer` builds a batch of one and runs it through `RoboticsProcessor.process_inputs` and
   `add_action_and_proprioception_fields`, the two calls the training pipeline makes. With
   `lowdim_past_timesteps: 1` the action tensor is `[1, 16, 8]`: slot 0 holds the previous action
   and `past_mask` marks it. With `lowdim_past_timesteps: 0` it is `[1, 15, 8]` and `prev_action`
   is unused.
4. `DiffusionPolicy.generate_actions` keeps slot 0 fixed, starts the other rows from noise and
   removes it in `num_flow_steps` flow steps (exact for divisors of 1000). The noise is new for
   every chunk; with a `noise_seed`, `Policy.infer` calls `torch.manual_seed` first and every chunk
   of an episode starts from the same noise. The server denormalises the result (back to joint
   angles and the gripper command, [40-action-spaces.md](40-action-spaces.md)) and returns
   `out[0, self.past:]`, shape `[15, 8]`: the actions for $t, t+1, \dots, t+14$.
5. The client keeps `reply["actions"][:exec_horizon]` as the queue. Each control step it stores the
   current observation as `prev_obs`, pops one action and calls `env.step`. When the queue is empty
   it sends the next `infer` message.

### The first control step

At the first control step there is no previous image, state or action. The client sends the first
observation twice and, as the previous action, the **hold command** the simulator is under after
reset: the start joint angles with the gripper open (`last_action` in `PickupEnv.reset`).

Training pads instead, and the two agree only partly. Image 0 twice and state 0 twice are what the
anchor-0 window contains, but its slot 0 is a copy of the expert's first command, which already
points at the target, where a served policy gets the hold command. For a policy that is given its
previous action this train/inference skew
([93-train-inference-skew.md](93-train-inference-skew.md)) decides where the gripper goes (below).

### Latency

**Latency** is the time between capturing an observation and having the chunk computed from it.
Here it costs nothing: `eval.py` waits for the reply before it calls `env.step`, so the simulated
world stands still. The lab notebook gives 20-65 ms per chunk for the 36M-parameter `dp_vit` and
about 190 ms for the Qwen VLA (`vla_qwen`, 2.2B parameters) on machine A (one RTX 4090, 24 GB). On
machine B (L40S GPUs, 128 CPU cores), on a GPU that was also training another model, the
48.9M-parameter `dp_vit_lang_reactive` measured below took a median of 71 ms at 10 flow steps and
25 ms at 2, which scored no lower (89/100 against 85/100; [41-flow-matching.md](41-flow-matching.md)).

On a real robot the world keeps moving, and a control step at 20 Hz is 50 ms: a robot waiting 190 ms
would stall for almost four control steps at every re-plan, and one that kept moving would get a
chunk whose first actions are out of date.

## In this repository

| What | Where |
|---|---|
| Windows and padding at preprocessing | [`preprocess.yaml`](../../sim/pickup/configs/preprocess.yaml); `LeRobotConverter.extract_sample_data` in [`converters/lerobot.py`](../../vla_foundry/data/preprocessing/robotics/converters/lerobot.py) |
| Cropping a stored window; choosing the state rows | `extract_robotics_fields` in [`pipelines/robotics.py`](../../vla_foundry/data/pipelines/robotics.py); `RoboticsProcessor.add_action_and_proprioception_fields` in [`robotics_processor.py`](../../vla_foundry/data/processor/robotics_processor.py) |
| Training forward pass; the sampler | `DiffusionPolicy.forward`, `generate_actions` in [`diffusion_policy.py`](../../vla_foundry/models/diffusion_policy/diffusion_policy.py) |
| Server; client, queue, `--exec_horizon`, `--noise` | `Policy.infer` in [`policy_server.py`](../../sim/pickup/policy_server.py); `main` in [`eval.py`](../../sim/pickup/eval.py) |

The preset `vla_foundry/config_presets/data/diffusion_policy.yaml` sets `lowdim_past_timesteps: 1` and
`lowdim_future_timesteps: 14`. `dp_vit` and `dp_vit_lang` keep that window (images and states at
$t-1$ and $t$, and the previous action): the policies **with history**. `dp_vit_reactive.yaml` and
`dp_vit_lang_reactive.yaml` set `lowdim_past_timesteps: 0` and `image_indices: [0]` (one image per
camera, the current state, no previous action): the policies **without history**, of which
`dp_vit_lang_reactive` solves `pick_target` ([101-case-pick-target.md](101-case-pick-target.md)).
The stored windows still have 16 rows with the anchor at index 1; `extract_robotics_fields` crops
them to rows 1 to 15, so the action tensor is `[batch, 15, 8]`.

The server reads `past` and `future` from the run's saved `config.yaml`, so one client serves both
kinds: it always sends two states and four images, and the server takes `state[-(self.past + 1):]`
and the images in `self.image_names`.

The protocol is three messages over `multiprocessing.connection` on `127.0.0.1`, as Python
dictionaries:

| Request | Reply |
|---|---|
| `{"cmd": "info"}` | `checkpoint`, `params_m`, `image_names`, `past`, `future` |
| `{"cmd": "infer", "images", "state", "prev_action", "instruction", "num_flow_steps", "noise_seed"}` | `{"actions": nested list [15, 8], "latency_s"}` |
| `{"cmd": "shutdown"}` | `{"ok": True}`, then the server exits |

`eval.sh` starts the server on a free port, runs the client and shuts the server down. The client
asks for `info` first and stops unless `checkpoint` belongs to the run it was started for
(`check_server`), so two evaluations on one port cannot file results under the wrong run. Extra
arguments go to `eval.py`; `PICKUP_OUT` keeps each setting's `results.json` apart:

```bash
PICKUP_OUT=results/pickup/dp_vit_lang_reactive_pt4k/pick_target/exec_4 \
  sim/pickup/scripts/eval.sh experiments/pickup/dp_vit_lang_reactive_pt4k pick_target 100 --exec_horizon 4
```

This is the first real run of [01-setup-and-first-run.md](01-setup-and-first-run.md):
`dp_vit_lang_reactive`, 4000 demonstrations, 5M samples; its config keeps averaged (EMA) weights
next to the trained (raw) ones, and `eval.sh` serves the EMA weights unless `PICKUP_WEIGHTS=raw`.

## What we measured

All numbers are from the [lab notebook](../lab-notebook.md).

**Actions executed per chunk.** `dp_vit_lang_reactive` after 5.0M samples on 4000 `pick_target`
demonstrations, EMA weights, machine B, 10 flow steps, 100 unseen layouts per cell, 23 of which
name the T block:

| Actions executed per chunk | Success | T block | Same starting noise for every chunk of an episode |
|---|---|---|---|
| 2 | 76/100 | 8/23 | 39/100 |
| 4 | 84/100 | 11/23 | 55/100 |
| 8 (default) | 85/100 | 15/23 | 80/100 |
| 15 (the whole chunk) | 84/100 | 16/23 | - |

- From 4 to 15 actions per chunk the count stays within one episode; the default scored 83/100 on a
  second set of 100 layouts, so one or two episodes are noise.
- Re-planning more often did not help: at 2 actions per chunk nine episodes are lost against the
  default, seven of them on the T block, the object this policy fails on most (grasps off in angle,
  mostly from the position bug: after the fix it reaches 87-93%;
  [101-case-pick-target.md](101-case-pick-target.md)).
- The same noise for every chunk, meant to make successive chunks choose alike, cost success, the
  more so the more chunks an episode has, for a cause not measured.

The sweep has one training seed, no row for $k = 1$, and layouts in which nothing moves, so it
cannot show what re-planning is for. An earlier sweep on `pick_cube` (`dp_vit`, with history, 1000
demonstrations, 1.5M samples, machine A, 50 layouts each) gave 84%, 74% and 78% at 4, 8 and 15
actions; all intervals overlap.

**Re-planning is not looking.** Three probes (evaluations with one condition changed) of the same
weights, 8 actions per chunk ([95-what-a-policy-looks-at.md](95-what-a-policy-looks-at.md)):

- With the first frame's images sent at every re-plan (`--stale_images`; the state stays current)
  success is 1/100, yet the policy still **closed at the named object** (the object nearest the
  gripper when it first closed was the one the instruction names) in 99% of episodes, a median of
  17 mm from it; as trained, 100% and 7 mm.
- With the named object moved 80 mm at control step 20 (`--disturb reach --disturb_step 20`) it
  succeeded in 0 of the 96 episodes in which the object could be moved, and closed a median of 8 mm
  from the old place. A policy that closes around control step 40, as the expert does, re-plans at
  control steps 24 and 32 from images that show the new place.
- With the expert driving the first 25 control steps toward another object
  (`--expert_prefix 25 --prefix_target distractor`) it closed at that object in 100/100 episodes.

The policy takes its goal from its first look and then from its own arm. No run combines a
disturbance with a shorter execution horizon. The demonstrations change this: trained on random
start poses with recoveries (after the position fix), a policy scores 89-94% under every
disturbance and, cameras frozen on the first frame, no longer finds the named object: it looks
throughout ([field notes](../notes.md), "Self-correction has to be in the data"; chapter 103,
coming).

**The previous action at the first control step.** `dp_vit_lang` (with history) on the same 4000
demonstrations (machine B) is at 2% / 2% (raw / EMA) after 3.0M samples. Its checkpoint at
2.5M samples (EMA) scored 1/100 as served, 22/100 with the expert's first command as the previous
action of the first query (`--first_prev_action expert`) and 77/100 with the expert driving the
first 10 control steps (`--expert_prefix 10`), closing at the named object in 30% (chance is 33%),
97% and 100% of episodes. It read the target from its previous action, which carries a decision
from chunk to chunk well enough to replace the cameras
([52-history-and-shortcuts.md](52-history-and-shortcuts.md)).

## Where it goes wrong

**The chunk starts at the wrong index.** In a 16-row output the first row is the previous action,
so `Policy.infer` returns `out[0, self.past:]`. Returning `out[0]` for a policy with history would
repeat the previous command at every re-plan and run every later action one control step late, with
valid shapes and no error. Check: print the first returned action next to `prev_action`; they
should differ whenever the arm is moving. `sim/pickup/open_loop_check.py` compares the returned
chunk with the stored actions from the anchor on (`window_rows`), with and without history.

**The wrong state for a policy without history.** The client sends `[previous, current]`; a policy
with `past = 0` must get the current one, so `Policy.infer` slices from the end. From the front the
slice has the same shape and is one control step old. Training once had the mirror image:
`lowdim_past_timesteps: 0` was read as unset and added the next step's state
([93-train-inference-skew.md](93-train-inference-skew.md)).

**A policy with history that stays near 0%.** Check: `--first_prev_action expert` and
`--expert_prefix 10`; a policy that scores far better with them leans on its own past. What to do:
start without history. `collect.py --hold_steps 1` begins every demonstration with a frame of the
robot at rest, so that the anchor-0 window holds what serving provides. It did not help: a history
policy trained on such demonstrations after the position fix ends at 0% / 4% after 5M samples, and
where it closes never depends on the scene, because from the second step on the previous action
still predicts the next one ([52-history-and-shortcuts.md](52-history-and-shortcuts.md)).

**Blaming the re-plan for a miss.** A gripper closing on empty table looks like successive chunks
heading for different objects. The failing policies here did not do that: where they closed did
not depend on the scene. The policy that works was one of them after 1.0M samples:
success 0% / 1% and a **slope** of 0.11 / 0.14 (of where the gripper first closed against where the
named object was: 1 if it closes where the object is, 0 if the scene makes no difference). After
5.0M samples, with the same settings, the slope is 0.98 / 0.98
([71-how-much-data-and-training.md](71-how-much-data-and-training.md)). Check: read the slope,
which `sim/pickup/summarize.py` prints, before changing `--exec_horizon`
([92-measuring-failure.md](92-measuring-failure.md)).

**Action-step tokens that share a position.** Each action of a chunk is one token, an action-step
token, in the transformer that generates it, and rotary position tells them apart. The position
table was computed in bfloat16: without history the 15 action-step tokens shared 8 positions, with
history 16 shared 4. Unless stated, every policy in this chapter was trained before the fix. After
it the same recipe ends at 93% against 83% (trained weights), and a second pair of seeds differing
in the table alone ends at 96% / 96% against 90% / 89%; in both the policy starts to look earlier
([12-positions.md](12-positions.md)).

## Background and further reading

- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*, 2023
  ([arXiv:2304.13705](https://arxiv.org/abs/2304.13705)). Introduced Action Chunking with
  Transformers (ACT): predict a chunk of joint targets at once to shorten the effective horizon.
  Its temporal ensembling queries the policy every control step and averages the overlapping
  predictions.
- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, 2023
  ([arXiv:2303.04137](https://arxiv.org/abs/2303.04137)). Names the three horizons used here:
  observation horizon, prediction horizon and action execution horizon, with re-planning after the
  execution horizon (receding horizon control). It reports that an execution horizon of 8 was best
  for most of its tasks and that too long a horizon costs reaction time.
- Liu et al., *Bidirectional Decoding: Improving Action Chunking via Guided Test-Time Sampling*,
  2024 ([arXiv:2408.17355](https://arxiv.org/abs/2408.17355)). At test time it draws several
  candidate chunks and keeps one, judged first by closeness to what the previous chunk had planned
  for the overlapping control steps: the part this repository's loop discards.
- Black et al., *Real-Time Execution of Action Chunking Flow Policies*, 2025
  ([arXiv:2506.07339](https://arxiv.org/abs/2506.07339)). Computes the next chunk while the current
  one executes, keeps the actions that will run meanwhile fixed and fills in the rest, so that
  consecutive chunks join smoothly despite the delay.

## Check yourself

1. A demonstration has 66 frames. For the window anchored at frame 60, work out how many of the 16
   rows are copies and which rows `future_mask` marks as true, then check. Repeat for anchor 0 and
   say what slot 0 holds there and what a served policy gets instead.

   ```bash
   .venv/bin/python -c "from vla_foundry.data.preprocessing.robotics.preprocess_masks import create_past_and_future_masks as m; print(m(60, 1, 14, 66))"
   ```

2. Evaluate one `pick_target` run with `--exec_horizon 2` and `8`, each with and without
   `--noise episode`, on the same 100 seeds (command above, one `PICKUP_OUT` per setting). Say
   whether the 95% intervals (`success_ci95` in `results.json`, a Wilson interval,
   [90-closed-loop-evaluation.md](90-closed-loop-evaluation.md)) overlap before drawing a
   conclusion, then convert `inference_latency_ms.p95` into control steps at 20 Hz.

3. Start a server (it needs a GPU; for the thread cap see
   [81-surviving-long-runs.md](81-surviving-long-runs.md)), wait for its `listening on` line and
   ask it what window its policy uses. From `past` and `image_names`, say which of the two states
   and four images the client sends are used.

   ```bash
   OMP_NUM_THREADS=4 .venv/bin/python -m sim.pickup.policy_server --checkpoint_dir experiments/pickup/dp_vit_lang_reactive_pt4k --port 6200 &
   .venv/bin/python -c "from multiprocessing.connection import Client; c = Client(('127.0.0.1', 6200), authkey=b'pickup-sim'); c.send({'cmd': 'info'}); print(c.recv()); c.send({'cmd': 'shutdown'}); print(c.recv())"
   ```
