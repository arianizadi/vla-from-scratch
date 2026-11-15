"""The aim range (sim/pickup/aim.py): short motor drills with a success rule and a time budget each.

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

from sim.pickup import aim  # noqa: E402
from sim.pickup.env import PANDA_XML, PRESS_TRAVEL, PickupEnv  # noqa: E402
from sim.pickup.expert import ScriptedExpert, random_start  # noqa: E402

SEEDS = (1_000_000, 1_000_001, 1_000_002)


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
    done, info = False, {}
    while not done:
        _, done, info = env.step(expert.act())
    return info


def _put_gripper(env, pos, R=None):
    """Move the arm (at rest) so the gripper's centre is at `pos`, pointing down unless R is given; returns the
    action that holds it there."""
    expert = ScriptedExpert(env, seed=0)
    q = expert.solve_ik(env.data.qpos[:7].copy(), np.asarray(pos, dtype=float), 0.0, iters=300, R_des=R)
    env.set_arm(q)
    return np.concatenate([q, [1.0]])


def _settle_dot_on_gripper(env, hold, steps=10):
    """Let the arm settle under `hold`, then put the named dot exactly where the gripper's centre is."""
    for _ in range(steps):
        env.data.ctrl[:7] = hold[:7]
        for _ in range(25):
            mujoco.mj_step(env.model, env.data)
    env.data.mocap_pos[env.dot_mocap[env.colour]] = env.tcp_pos()
    mujoco.mj_forward(env.model, env.data)


@pytest.mark.parametrize("drill", aim.DRILLS)
def test_the_expert_solves_every_drill(drill):
    env = _env(f"aim_{drill}")
    assert type(ScriptedExpert(env)).__name__ == "AimExpert"
    for seed in SEEDS:
        info = _run_expert(env, seed)
        assert info["success"], (drill, seed, info)
        assert info["drill"] == drill and not info["wrong_object"]
        assert env.steps <= aim.BUDGET[drill]


def test_a_seed_always_gives_the_same_episode_and_aim_mix_gives_the_drills_own_layouts():
    env = _env("aim_mix")
    drills = set()
    for seed in range(1_000_000, 1_000_060):
        env.reset(seed)
        first = (env.drill, env.instruction, {c: p.round(6).tolist() for c, p in env.dots.items()})
        env.reset(seed)
        assert first == (env.drill, env.instruction, {c: p.round(6).tolist() for c, p in env.dots.items()})
        drills.add(env.drill)
        if seed < 1_000_008:
            own = _env(f"aim_{env.drill}")
            own.reset(seed)
            assert own.instruction == env.instruction
            assert {c: p.round(6).tolist() for c, p in own.dots.items()} == first[2]
    assert drills == set(aim.DRILLS)
    # instructions name the drill and the target, and the named target is among the things in view
    env = _env("aim_touch")
    colours = set()
    for seed in SEEDS:
        env.reset(seed)
        assert env.instruction == f"touch the {env.colour} dot"
        assert len(env.dots) == 3 and env.colour in env.dots
        colours.add(env.colour)
        gaps = [np.linalg.norm(a - b) for a in env.dots.values() for b in env.dots.values() if a is not b]
        assert min(gaps) >= aim.DOT_GAP - 1e-9


def test_touch_needs_the_named_dot_for_a_few_steps():
    env = _env("aim_touch")
    env.reset(SEEDS[0])
    distractor = next(c for c in env.dots if c != env.colour)
    hold = _put_gripper(env, env.dot_pos(distractor))
    for _ in range(10):  # on another dot: nothing happens
        _, done, info = env.step(hold)
        assert not info["success"] and not done
    hold = _put_gripper(env, env.dot_pos())
    _settle_dot_on_gripper(env, hold)
    results = [env.step(hold)[2]["success"] for _ in range(aim.TOUCH_HOLD)]
    assert results == [False] * (aim.TOUCH_HOLD - 1) + [True]


def test_hold_needs_a_precise_stop():
    env = _env("aim_hold")
    env.reset(SEEDS[1])
    hold = _put_gripper(env, env.dot_pos())
    _settle_dot_on_gripper(env, hold)
    env.data.mocap_pos[env.dot_mocap[env.colour]] += [0.012, 0, 0]  # 12 mm off: near enough to touch, not to stop
    for _ in range(aim.HOLD_STEPS + 5):
        assert not env.step(hold)[2]["success"]
    env.data.mocap_pos[env.dot_mocap[env.colour]] -= [0.012, 0, 0]
    results = [env.step(hold)[2]["success"] for _ in range(aim.HOLD_STEPS)]
    assert results[-1] and not any(results[:-1])


def test_pose_needs_the_gripper_along_the_pin():
    env = _env("aim_pose")
    for seed in range(1_000_000, 1_000_020):  # a layout whose pin is far from vertical
        env.reset(seed)
        if np.degrees(np.arccos(-env.pin_dir[2])) > 45:
            break
    hold = _put_gripper(env, env.dot_pos())  # pointing down
    _settle_dot_on_gripper(env, hold)
    for _ in range(aim.TOUCH_HOLD + 3):
        _, _, info = env.step(hold)
        assert not info["success"] and info["angle_deg"] > np.degrees(aim.POSE_ANGLE)


def test_a_gripper_that_stays_put_loses_a_moving_dot():
    env = _env("aim_track")
    env.reset(SEEDS[0])
    hold = np.concatenate([env.data.qpos[:7], [1.0]])
    done, info = False, {}
    while not done:
        _, done, info = env.step(hold)
    assert env.steps == len(env.path) - 1 or env.steps == env.max_steps
    assert not info["success"] and info["within"] < aim.TRACK_SHARE
    moved = np.linalg.norm(np.diff(env.path, axis=0), axis=1).max() * 20
    assert moved <= aim.TRACK_SPEED[1] + 1e-6


def test_flick_lights_the_next_dot_after_each_hit():
    env = _env("aim_flick")
    env.reset(SEEDS[2])
    assert 3 <= len(env.flick_points) <= 5
    for k, point in enumerate(env.flick_points):
        assert np.allclose(env.dot_pos(), point)
        hold = _put_gripper(env, point)
        _settle_dot_on_gripper(env, hold)
        _, done, info = env.step(hold)
        assert info["hits"] == k + 1
    assert info["success"] and done


def test_pressing_the_other_button_is_a_failure():
    env = _env("aim_press")
    env.reset(SEEDS[0])
    other = env._others()[0]
    top = env.button_top(other)
    hold = _put_gripper(env, top + [0, 0, 0.03])
    hold[7] = 0.0
    expert = ScriptedExpert(env, seed=0)
    done, info, z = False, {}, 0.03
    while not done and z > -0.01:
        z -= 0.002
        action = np.concatenate([expert.solve_ik(hold[:7], top + [0, 0, z], 0.0), [0.0]])
        _, done, info = env.step(action)
    assert env.aim_button_travel(other) > PRESS_TRAVEL
    assert done and info["wrong_object"] and not info["success"]


def test_other_tasks_are_unchanged_by_the_aim_range():
    for task in ("pick_target", "prune"):
        env = _env(task)
        assert type(env) is PickupEnv and type(ScriptedExpert(env)) is ScriptedExpert
        names = {env.model.body(i).name for i in range(env.model.nbody)}
        assert not any(n.startswith("aim_") for n in names)
