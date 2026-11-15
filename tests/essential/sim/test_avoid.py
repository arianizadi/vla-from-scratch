"""Avoidance drills (sim/pickup/avoid.py): reach past obstacles and a keep-out zone without touching them.

Needs MuJoCo, a GL context for the cameras and the Franka model (sim/pickup/scripts/fetch_assets.sh); skipped where
any of them is missing.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:  # sim/ is not an installed package
    sys.path.insert(0, str(REPOSITORY))

mujoco = pytest.importorskip("mujoco")

from sim.pickup import avoid  # noqa: E402
from sim.pickup.env import PANDA_XML, PickupEnv  # noqa: E402
from sim.pickup.expert import ScriptedExpert, random_start  # noqa: E402

SEEDS = (1_000_000, 1_000_001)


def _env(task):
    if not PANDA_XML.exists():
        pytest.skip("Franka model not fetched (sim/pickup/scripts/fetch_assets.sh)")
    try:
        return PickupEnv(task, image_size=64)
    except Exception as error:  # no GL context on this machine
        pytest.skip(f"cannot render: {error}")


def _run_expert(env, seed):
    expert = ScriptedExpert(env, seed=0)
    env.reset(seed)
    _, yaw = random_start(env, expert, seed)
    expert.reset(yaw=yaw)
    done, info, touched_at = False, {}, None
    while not done:
        _, done, info = env.step(expert.act())
        if touched_at is None and getattr(env, "touched", False):
            touched_at = env.steps
    return info, touched_at


@pytest.mark.parametrize("drill", avoid.DRILLS)
def test_the_expert_solves_every_drill_without_touching_anything(drill):
    env = _env(f"avoid_{drill}")
    assert type(ScriptedExpert(env)).__name__ == "AvoidExpert"
    for seed in SEEDS:
        info, touched_at = _run_expert(env, seed)
        assert info["success"], (drill, seed, info)
        assert info["contacts"] == 0 and not info["hit"] and not info["in_zone"]
        assert env.plan_clearance is not None and env.plan_clearance >= avoid.AVOID_MARGIN
        if drill == "turn":  # touching the dot is not enough: the hand has to come back out of the pocket
            assert touched_at is not None and env.steps > touched_at + 5


def test_a_seed_always_gives_the_same_layout_and_avoid_mix_gives_the_drills_own():
    env = _env("avoid_mix")
    drills = set()
    for seed in range(1_000_000, 1_000_024):
        env.reset(seed)
        slabs = env.model.body_pos[env.slab_body].round(6).tolist()
        first = (env.drill, env.instruction, slabs)
        env.reset(seed)
        assert first == (env.drill, env.instruction, env.model.body_pos[env.slab_body].round(6).tolist())
        drills.add(env.drill)
        if seed < 1_000_006:
            own = _env(f"avoid_{env.drill}")
            own.reset(seed)
            assert own.instruction == env.instruction
            assert own.model.body_pos[own.slab_body].round(6).tolist() == slabs
    assert drills == set(avoid.DRILLS)


def _press_into(env, point, steps=60):
    """Drive the gripper's set-point toward `point` (from where it is), a few millimetres a step."""
    expert = ScriptedExpert(env, seed=0)
    q = env.data.qpos[:7].copy()
    start = env.tcp_pos()
    done, info = False, {}
    for k in range(1, steps + 1):
        target = start + (point - start) * min(1.0, k / (steps * 0.6))
        q = expert.solve_ik(q, target, 0.0)
        _, done, info = env.step(np.concatenate([q, [0.0]]))
        if done:
            break
    return done, info


def test_touching_an_obstacle_fails_the_drill_and_is_counted():
    env = _env("avoid_wall")
    env.reset(SEEDS[0])
    wall = env.model.body_pos[env.slab_body[0]].copy()
    top = wall[2] + env.model.geom_size[env.slab_geom[0]][2]
    done, info = _press_into(env, np.array([wall[0], wall[1], top - 0.03]))
    assert done and not info["success"] and info["wrong_object"]
    assert info["hit"] and info["contacts"] > 0


def test_entering_the_hot_zone_fails_the_drill():
    env = _env("avoid_hot")
    env.reset(SEEDS[1])
    zone = env.model.body_pos[env.zone_body].copy()
    done, info = _press_into(env, zone)
    assert done and not info["success"] and info["in_zone"] and info["zone_mm"] < 0


def test_other_tasks_are_unchanged_by_the_avoidance_drills():
    for task in ("aim_touch", "prune"):
        env = _env(task)
        assert type(ScriptedExpert(env)).__name__ != "AvoidExpert"
        names = {env.model.body(i).name for i in range(env.model.nbody)}
        assert not any(n.startswith("avoid_") for n in names)
