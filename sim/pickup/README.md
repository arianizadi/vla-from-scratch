# Pickup sim: a MuJoCo training and evaluation environment for VLA Foundry

A self-contained loop for training VLA Foundry policies on a simulated robot and measuring them
closed-loop:

```
scripted expert in MuJoCo ──> LeRobot v2.0 dataset ──> Foundry shards ──> vla_foundry/main.py
                                                                               │
        closed-loop success rate <── simulator client <── socket ── policy server (checkpoint)
```

- **Robot**: Franka Panda (MuJoCo Menagerie) on a table, 20 Hz control.
- **Objects**: red cube, green can, blue hex nut, yellow T block, modelled in Blender
  (`assets/build_assets_blender.py`, `assets/pickup_assets.blend`) and exported as OBJ. Physics uses
  primitive collision shapes; the Blender meshes are the visuals.
- **Observations**: `front` and `wrist` RGB cameras (224x224) and an 8-D state
  (7 joint angles + gripper opening in [0, 1]).
- **Actions**: 8-D absolute targets (7 joint positions + gripper command, 1 = open, 0 = closed).
  `action[t]` is the command applied after observing frame `t`.

| Task | Scene | Instruction | Success |
|---|---|---|---|
| `pick_cube` | one cube, random pose | "pick up the red cube" | cube held 8 cm above the table for 0.5 s |
| `pick_target` | 3 of the 4 objects, random poses | "pick up the \<object\>" | the *named* object held 8 cm up for 0.5 s |

Episodes time out after 200 steps (10 s). Demonstrations use seeds `0..`; evaluation uses seeds
`1_000_000..`, so every evaluated layout is unseen.

## Layout

```
sim/pickup/
  env.py             PickupEnv: scene construction (MjSpec), reset/step, cameras, success test
  expert.py          scripted top-down grasp expert (Cartesian plan + damped-least-squares IK)
  collect.py         run the expert, write a LeRobot v2.0 dataset (inline JPEG parquet)
  policy_server.py   serve action chunks from a checkpoint over a local socket
  eval.py            closed-loop evaluation client; also runs the scripted expert (--expert)
  diagnostics.py     where the gripper was, relative to the objects, when it first closed
  grasp_tolerance.py grasp success against a controlled placement error, per object
  open_loop_check.py compare predicted chunks with expert actions on recorded samples
  caption_data.py    captioned scene renders for vision-language pretraining
  vlm_check.py       held-out captioning check for a trained VLM
  monitor.py         terminal dashboard: runs, loss, closed-loop success per checkpoint, GPUs
  summarize.py       all closed-loop results as one table (or JSON)
  text_data.py       web text for the language-model stage of the from-scratch path
  smoke_test.py      expert-only sanity check
  configs/           preprocessing + training configs
  scripts/           fetch_assets.sh, make_dataset.sh, train.sh, train_resilient.sh (auto-resume), eval.sh,
                     eval_checkpoints.sh (every checkpoint), custom_path.sh (LLM -> VLM -> VLA stages)
  assets/            Blender source, exported meshes; third_party/ (Panda model, fetched, not committed)
```

## Setup

One environment runs everything (simulator, preprocessing, training, policy server):

```bash
uv sync --group preprocessing --group sim
sim/pickup/scripts/fetch_assets.sh          # once: Panda model from MuJoCo Menagerie
```

Rendering is headless through EGL (`MUJOCO_GL=egl`, set by the scripts), so no display is needed.

The scripts read a few environment variables, documented in `scripts/_common.sh`:

| Variable | Use |
|---|---|
| `PICKUP_GPU` | GPU index (as `nvidia-smi` shows it) for compute and rendering. Lets several jobs run side by side on a multi-GPU machine |
| `PICKUP_WORKERS` | simulator processes and preprocessing CPUs for `make_dataset.sh` (default 8) |
| `PICKUP_CPU_THREADS` | CPU threads per process (default 4). Without a cap, numerical libraries start one thread per core, which saturates a many-core machine |
| `PICKUP_FOUNDRY_PYTHON`, `PICKUP_SIM_PYTHON`, `PICKUP_ROOT` | run the training side or the simulator side with a different interpreter, for example inside a container |

## Quickstart

All commands run from the repository root.

```bash
# 1. demonstrations -> shards   (1000 episodes)
sim/pickup/scripts/make_dataset.sh pick_cube 1000 pick_cube_1k

# 2. train                      (config, dataset, run name, total training samples)
sim/pickup/scripts/train.sh dp_vit pick_cube_1k dp_vit_pick_cube_1k 1500000

# 3. closed-loop evaluation     (experiment dir, task, episodes)
sim/pickup/scripts/eval.sh experiments/pickup/dp_vit_pick_cube_1k pick_cube 50
```

`train_resilient.sh` takes the same arguments as `train.sh` and restarts the run from its newest
checkpoint if it dies or stops making progress. Use it for anything longer than a few minutes.

`eval.sh` writes `results/pickup/<run>/<task>/results.json` and a few rollout videos. The results
hold the success rate with a Wilson 95% interval, a per-object breakdown, and grasp diagnostics:
whether the gripper closed, which object it was nearest to when it did, and how far it missed by.
Useful knobs: `--exec_horizon` (actions executed per predicted chunk, default 8), `--flow_steps`
(denoising steps, default 10), `--seed_start`, `--videos`. `PICKUP_WEIGHTS=raw|ema` picks the
trained or the averaged weights of a run trained with `--ema.enabled True`.

### Watching runs

`eval_checkpoints.sh` evaluates every checkpoint of a run (newest first, several at a time with
`PICKUP_EVAL_JOBS`), and with `PICKUP_WATCH=1` it follows a run that is still training. Started
next to `train_resilient.sh`, it turns a run into a curve of closed-loop success against training:

```bash
sim/pickup/scripts/train_resilient.sh dp_vit_lang pick_target_1500 my_run 1000000 --ema.enabled True &
PICKUP_WATCH=1 PICKUP_EVAL_JOBS=2 sim/pickup/scripts/eval_checkpoints.sh experiments/pickup/my_run pick_target 100 &
```

`monitor.py` is a terminal dashboard over all runs. It shows progress, the loss with its history,
throughput, time left and restarts, and next to them what the loss does not tell you: the
closed-loop success of each checkpoint, how often the gripper closed at the named object, and the
median miss. GPU load is underneath. It only reads logs and results, so it can be opened and closed
at any time, in another terminal or over ssh.

```bash
.venv/bin/python -m sim.pickup.monitor                         # live, Ctrl-C quits
.venv/bin/python -m sim.pickup.monitor --once --svg runs.svg   # one frame, saved as an image
.venv/bin/python -m sim.pickup.summarize --task pick_target    # the same results as one table
```

### Checks that need no trained policy

Three checks that do not need a trained policy, or do not need the simulator:

```bash
# the scripted expert through the same evaluation loop: is the task itself sound?
.venv/bin/python -m sim.pickup.eval --task pick_target --episodes 100 --expert --out results/pickup/expert/pick_target

# how much placement error does a grasp tolerate, per object?
.venv/bin/python -m sim.pickup.grasp_tolerance --out results/pickup/grasp_tolerance.json

# does a checkpoint reproduce recorded expert actions? (no simulator involved)
.venv/bin/python -m sim.pickup.open_loop_check \
    --checkpoint_dir experiments/pickup/<run> --shards data/preprocessed/<dataset>/shards
```

## How it plugs into Foundry

- `collect.py` writes exactly the LeRobot v2.0 layout Foundry's `--type lerobot` converter reads.
  Images are stored inline as JPEG, so the simulator renders at the training resolution and
  `configs/preprocess.yaml` sets `resize_images_size: null` (the converter cannot resize images
  that are already encoded).
- Samples are 16-step windows: 1 past + current + 14 future, with images at `t-1` and `t`.
  Foundry treats the first slot as a known past action and predicts the remaining 15.
- `policy_server.py` builds its model input with the same `RoboticsProcessor.process_inputs` and
  `add_action_and_proprioception_fields` calls as the training dataloader, then denormalizes the
  predicted chunk with the run's saved normalizer. It reads everything from the experiment
  directory (`config.yaml`, `config_processor.yaml`, `stats.json`, latest checkpoint or EMA).
- Data-loader workers hand batches to the trainer through shared memory (`/dev/shm`). With the
  passthrough processor a batch of 64 is ~150 MB. If shared memory is small (containers often get
  64 MB to 2 GB), too many workers or too much prefetch overflows it and the run dies without a
  traceback; lower `data.num_workers` and `data.prefetch_factor`.

## Results

See `RESULTS.md`.
