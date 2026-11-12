"""Disturbances, the re-planning expert, random start poses and the look-alike task (sim/pickup/).

Needs MuJoCo, a GL context for the cameras and the Franka model (sim/pickup/scripts/fetch_assets.sh);
skipped where any of them is missing.
"""

import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:  # sim/ is not an installed package
    sys.path.insert(0, str(REPOSITORY))

pytest.importorskip("mujoco")

from sim.pickup.disturb import Disturbance  # noqa: E402
from sim.pickup.env import PANDA_XML, PickupEnv  # noqa: E402
from sim.pickup.expert import ScriptedExpert, random_start  # noqa: E402

SEEDS = (1_000_000, 1_000_001, 1_000_002)


@pytest.fixture(scope="module")
def env():
    if not PANDA_XML.exists():
        pytest.skip("Franka model not fetched (sim/pickup/scripts/fetch_assets.sh)")
    try:
        environment = PickupEnv("pick_target")
    except Exception as error:  # no GL context on this machine
        pytest.skip(f"cannot render: {error}")
    yield environment
    environment.close()


def rollout(env, seed, kind, reactive, speed=0.02):
    env.reset(seed=seed)
    expert = ScriptedExpert(env, seed=0, reactive=reactive)
    expert.reset()
    disturbance = Disturbance(env, kind, step=15, distance=0.08, seed=seed, speed=speed)
    actions, done, info = [], False, {}
    while not done:
        action = expert.act()
        actions.append(action)
        disturbance.update(action)
        _, done, info = env.step(action)
    return info["success"], actions, disturbance.record, expert.replans


def test_watching_changes_nothing_when_nothing_happens(env):
    """Without a disturbance the re-planning expert issues the same commands as the plain one."""
    for seed in SEEDS:
        ok_plain, plain, _, _ = rollout(env, seed, "none", reactive=False)
        ok_reactive, reactive, _, replans = rollout(env, seed, "none", reactive=True)
        assert ok_plain and ok_reactive and replans == 0
        assert len(plain) == len(reactive)
        assert all((a == b).all() for a, b in zip(plain, reactive, strict=True))


@pytest.mark.parametrize("kind", ["reach", "grasp"])
def test_only_the_replanning_expert_recovers(env, kind):
    moved = 0
    for seed in SEEDS:
        ok_plain, _, record, _ = rollout(env, seed, kind, reactive=False)
        if not record or record["to_xy"] is None:
            continue  # no free place for the object in this layout
        moved += 1
        ok_reactive, _, _, replans = rollout(env, seed, kind, reactive=True)
        assert not ok_plain, f"seed {seed}: the object moved {kind}, yet the one-plan expert succeeded"
        assert ok_reactive and replans >= 1, f"seed {seed}: the re-planning expert did not recover ({kind})"
    assert moved, "no episode was disturbed"


def test_the_object_lands_on_free_table(env):
    env.reset(seed=SEEDS[0])
    start = env.object_pos(env.target)[:2].copy()
    disturbance = Disturbance(env, "reach", step=0, distance=0.08, seed=3)
    assert disturbance.update([0] * 7 + [1.0])
    end = env.object_pos(env.target)[:2]
    assert abs(((end - start) ** 2).sum() ** 0.5 - 0.08) < 1e-6
    for other in env.active:
        if other != env.target:
            assert ((env.object_pos(other)[:2] - end) ** 2).sum() ** 0.5 >= 0.11
    assert not disturbance.update([0] * 7 + [1.0]), "one move per episode"


def test_random_start_keeps_the_layout_and_the_expert_still_succeeds(env):
    for seed in SEEDS:
        env.reset(seed=seed)
        layout = {name: env.object_pos(name).copy() for name in env.active}
        home = env.tcp_pos().copy()
        expert = ScriptedExpert(env, seed=0)
        _, yaw = random_start(env, expert, seed)
        assert all((env.object_pos(name) == layout[name]).all() for name in env.active)
        assert ((env.tcp_pos() - home) ** 2).sum() ** 0.5 > 0.02, "the gripper did not leave the home pose"
        assert env.tcp_pos()[2] > 0.2
        expert.reset(yaw=yaw)
        done, info = False, {}
        while not done:
            _, done, info = env.step(expert.act())
        assert info["success"]


def test_drift_slides_the_object_at_its_speed_and_stops_while_it_is_held(env):
    env.reset(seed=SEEDS[1])
    others = {n: env.object_pos(n).copy() for n in env.active if n != env.target}
    drift = Disturbance(env, "drift", seed=5, speed=0.02)
    start = env.object_pos(env.target)[:2].copy()
    for _ in range(40):  # two seconds with the gripper open
        assert drift.update([0] * 7 + [1.0])
    assert abs(drift.record["path_m"] - 40 * 0.02 / 20) < 1e-9
    travelled = ((env.object_pos(env.target)[:2] - start) ** 2).sum() ** 0.5
    assert 0.005 < travelled <= 0.04 + 1e-9, "a smooth path, no jump"
    end = env.object_pos(env.target)[:2]
    for name, pos in others.items():
        assert (env.object_pos(name) == pos).all(), "only the named object moves"
        assert ((pos[:2] - end) ** 2).sum() ** 0.5 >= 0.11
    assert not drift.update([0] * 7 + [0.0]), "held: no drift"
    assert drift.record["to_xy"] == [float(v) for v in end]


@pytest.mark.parametrize("speed", [0.01, 0.03])
def test_only_the_tracking_expert_follows_a_drifting_target(env, speed):
    for seed in SEEDS:
        ok_plain, _, record, _ = rollout(env, seed, "drift", reactive=False, speed=speed)
        assert record["path_m"] > 0.02
        ok_reactive, _, _, _ = rollout(env, seed, "drift", reactive=True, speed=speed)
        assert ok_reactive, f"seed {seed}: the watching expert lost a target drifting at {speed} m/s"
    # the one-plan expert closes where the object was; at 3 cm/s it has moved several cm by then
    if speed >= 0.03:
        assert not all(rollout(env, s, "drift", reactive=False, speed=speed)[0] for s in SEEDS)


def test_lookalike_layouts_hold_three_cubes_told_apart_by_colour_only():
    if not PANDA_XML.exists():
        pytest.skip("Franka model not fetched (sim/pickup/scripts/fetch_assets.sh)")
    try:
        env = PickupEnv("pick_lookalike")
    except Exception as error:
        pytest.skip(f"cannot render: {error}")
    try:
        labels = set()
        for seed in SEEDS:
            env.reset(seed=seed)
            assert len(env.active) == 3 and env.target in env.active
            assert all(n == "cube" or n.startswith("cube_") for n in env.active)
            labels.add(env.instruction)
            expert = ScriptedExpert(env, seed=0)
            expert.reset()
            done, info = False, {}
            while not done:
                _, done, info = env.step(expert.act())
            assert info["success"], f"seed {seed}: the expert failed on a look-alike layout"
        assert all(text.startswith("pick up the ") and text.endswith(" cube") for text in labels)
    finally:
        env.close()
