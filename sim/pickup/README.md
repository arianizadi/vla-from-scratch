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
  policy_server.py   serve action chunks from a Foundry checkpoint (runs in the Foundry env)
  eval.py            closed-loop evaluation client (runs in the simulator env)
  open_loop_check.py compare predicted chunks with expert actions on recorded samples
  smoke_test.py      expert-only sanity check
  configs/           preprocessing + training configs
  scripts/           fetch_assets.sh, make_dataset.sh, train.sh, eval.sh
  assets/            Blender source, exported meshes; third_party/ (Panda model, fetched, not committed)
```

Two Python environments are involved and never need to be merged:

- the **simulator** side (`env.py`, `expert.py`, `collect.py`, `eval.py`) needs only
  `mujoco numpy pillow pyarrow av`; for example a small separate virtual environment.
- the **Foundry** side (`policy_server.py`, preprocessing, training) runs in the Foundry
  environment; for example a container launcher.

Set `PICKUP_SIM_PYTHON` to point the scripts at a different simulator interpreter.

## Quickstart

All commands run from the repository root.

```bash
sim/pickup/scripts/fetch_assets.sh                      # once: Panda model from MuJoCo Menagerie

# 1. demonstrations -> shards   (1000 episodes: ~40 s to simulate, ~4 min to shard)
sim/pickup/scripts/make_dataset.sh pick_cube 1000 pick_cube_1k

# 2. train                      (config, dataset, run name, total training samples)
sim/pickup/scripts/train.sh dp_vit pick_cube_1k dp_vit_pick_cube_1k 1500000

# 3. closed-loop evaluation     (experiment dir, task, episodes)
sim/pickup/scripts/eval.sh experiments/pickup/dp_vit_pick_cube_1k pick_cube 50
```

`eval.sh` writes `results/pickup/<run>/<task>/results.json` (success rate with a Wilson 95%
interval, per-object breakdown, wrong-object rate, inference latency) and a few rollout videos.
Useful knobs: `--exec_horizon` (actions executed per predicted chunk, default 8), `--flow_steps`
(denoising steps, default 10), `--seed_start`, `--videos`.

To check a checkpoint without the simulator:

```bash
vla-foundry python -m sim.pickup.open_loop_check \
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
- A container launcher may give the container only 2 GB of shared memory. With the passthrough
  processor a batch of 64 is ~150 MB, so keep `data.num_workers` at 8; 16 workers overflows it and
  the run dies without a traceback.

## Results

See `RESULTS.md`.
