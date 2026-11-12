"""Pruning a potted plant on a motorised turntable (sim/pickup/, tasks prune_one, prune_turn, prune).

Needs MuJoCo, a GL context for the cameras and the Franka model (sim/pickup/scripts/fetch_assets.sh);
skipped where any of them is missing.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:  # sim/ is not an installed package
    sys.path.insert(0, str(REPOSITORY))

mujoco = pytest.importorskip("mujoco")

from sim.pickup.env import (  # noqa: E402
    BUTTONS,
    CUT_MID,
    CUT_ZONE,
    PANDA_XML,
    PRESS_TRAVEL,
    STEM_H,
    TURN_SPEED,
    PickupEnv,
)
from sim.pickup.expert import ScriptedExpert  # noqa: E402

SEEDS = (1_000_000, 1_000_001, 1_000_002)
PENETRATION_LIMIT = 0.008  # metres: deepest the arm may sink into the plant, its stand or the buttons


def _env(task):
    if not PANDA_XML.exists():
        pytest.skip("Franka model not fetched (sim/pickup/scripts/fetch_assets.sh)")
    try:
        return PickupEnv(task)
    except Exception as error:  # no GL context on this machine
        pytest.skip(f"cannot render: {error}")


def _plant_bodies(env):
    m = env.model
    names = ("stand", "plate", "plant", *(f"button_{b}" for b in BUTTONS), *env.shoot_names)
    return {m.body(n).id for n in names}


def _deepest(env, plant_bodies, robot):
    """Deepest current contact between the robot and the plant, its stand or the buttons (metres)."""
    m, d = env.model, env.data
    deepest = 0.0
    for i in range(d.ncon):
        b1, b2 = m.geom_bodyid[d.contact.geom1[i]], m.geom_bodyid[d.contact.geom2[i]]
        if (b1 in robot and b2 in plant_bodies) or (b2 in robot and b1 in plant_bodies):
            deepest = max(deepest, -float(d.contact.dist[i]))
    return deepest


def _push_button(env, name, force, steps):
    """Press a button straight down with `force` newtons for `steps` control steps (the arm holds still)."""
    body = env.model.body(f"button_{name}").id
    env.data.xfrc_applied[body, 2] = -force
    for _ in range(steps):
        env.step(env.last_action.copy())
    env.data.xfrc_applied[body] = 0


def test_buttons_turn_the_plate_their_way_and_only_while_pressed():
    env = _env("prune_turn")
    try:
        env.reset(seed=SEEDS[0])
        start = env.plate_angle()
        for _ in range(10):  # nothing pressed: the plate stands still
            env.step(env.last_action.copy())
        assert abs(env.plate_angle() - start) < 1e-3
        _push_button(env, "ccw", force=1.2, steps=10)  # a light touch: under PRESS_TRAVEL
        assert env.button_travel("ccw") < PRESS_TRAVEL
        assert abs(env.plate_angle() - start) < 1e-3, "a light touch turned the plate"
        for name, sign in (("ccw", 1), ("cw", -1)):
            before = env.plate_angle()
            _push_button(env, name, force=6.0, steps=20)  # one second held down
            turned = env.plate_angle() - before
            assert sign * turned > 0.8 * TURN_SPEED, f"{name}: turned {turned:.3f} rad in 1 s"
            assert sign * turned < 1.1 * TURN_SPEED
            for _ in range(3):  # let go: the spring lifts the button and the motor stops
                env.step(env.last_action.copy())
            assert env.button_travel(name) < PRESS_TRAVEL
            stopped = env.plate_angle()
            for _ in range(20):
                env.step(env.last_action.copy())
            assert abs(env.plate_angle() - stopped) < 0.01, f"{name}: the plate kept turning after release"
    finally:
        env.close()


@pytest.mark.parametrize("task", ["prune_one", "prune_turn", "prune"])
def test_the_expert_prunes_without_damage_or_passing_through(task):
    env = _env(task)
    try:
        plant_bodies, robot = _plant_bodies(env), set(env.robot_bodies.tolist())
        wins, pressed = 0, 0
        for seed in SEEDS:
            env.reset(seed=seed)
            assert env.instruction.endswith("off the stem, keep the flower")
            expert = ScriptedExpert(env, seed=0)
            expert.reset()
            done, info, deepest, turned = False, {}, 0.0, False
            start = env.plate_angle()
            while not done:
                _, done, info = env.step(expert.act())
                deepest = max(deepest, _deepest(env, plant_bodies, robot))
                turned = turned or abs(env.plate_angle() - start) > 0.3
            pressed += turned
            assert info["damaged"] is None, f"seed {seed}: {info['damaged']}"
            assert deepest < PENETRATION_LIMIT, f"seed {seed}: the arm sank {1000 * deepest:.1f} mm into the plant"
            wins += info["success"]
            if info["success"]:
                assert info["cut"] == sorted(env.active)
        assert wins >= 2, f"the expert pruned {wins} of {len(SEEDS)} plants"
        if task == "prune_one":
            assert pressed == 0, "prune_one needs no turning"
    finally:
        env.close()


def test_a_cut_shoot_falls_and_comes_to_rest_on_something_solid():
    env = _env("prune_one")
    try:
        env.reset(seed=SEEDS[0])
        name = env.active[0]
        expert = ScriptedExpert(env, seed=0)
        expert.reset()
        done = False
        while not done:
            _, done, info = env.step(expert.act())
        assert info["success"] and info["cut"] == [name]
        assert not env.data.eq_active[env.shoot_weld[name]], "the weld still holds the cut part"
        a, d = env.shoot_axis(name)
        s = env.shoots[name]["cut_s"]
        assert CUT_ZONE[0] <= s <= CUT_ZONE[1]
        cut_point = a + d * s
        stub, outer = env.stub_geoms[name][0], env.outer_geoms[name][0]
        # the two pieces no longer overlap: the stub's cap ends before the cut, the outer part's cap after it
        m = env.model
        stub_reach = 2 * m.geom_size[stub][1] + m.geom_size[stub][0]  # from the stem's axis, cap included
        outer_start = CUT_MID + m.geom_pos[outer][2] - m.geom_size[outer][1] - m.geom_size[outer][0]
        assert stub_reach < s < outer_start, (stub_reach, s, outer_start)
        for _ in range(80):  # the expert opens, backs out and waits; the part falls on the plate, pot or table
            env.step(expert.act())
        body = env.obj_body[name]
        assert env.data.xipos[body][2] < cut_point[2] - 0.05, "the cut part did not fall"
        assert env.data.xipos[body][2] > 0.0, "the cut part passed through the table"
        vadr = env.obj_vadr[name]
        assert np.linalg.norm(env.data.qvel[vadr : vadr + 3]) < 0.05, "the cut part is still moving"
        dd = env.data
        touching = [
            i
            for i in range(dd.ncon)
            if body in (m.geom_bodyid[dd.contact.geom1[i]], m.geom_bodyid[dd.contact.geom2[i]])
        ]
        assert touching, "the cut part rests on nothing"
        assert all(dd.contact.dist[i] > -0.004 for i in touching), "the cut part sank into what it lies on"
    finally:
        env.close()


def _drive(env, expert, waypoints, steps):
    """Run the expert's waypoint follower on a hand-made plan; returns the last info."""
    expert.waypoints, expert.wp, expert.dwell = waypoints, 0, 0
    info = {}
    for _ in range(steps):
        _, done, info = env.step(expert.act())
        if done:
            break
    return info


def test_touching_the_flower_ends_the_episode_as_a_failure():
    env = _env("prune_one")
    try:
        env.reset(seed=SEEDS[0])
        expert = ScriptedExpert(env, seed=0)
        expert.reset()
        base, up = env.stem_axis()
        head = base + up * STEM_H
        R = expert.R_down
        info = _drive(
            env,
            expert,
            [(head + [0, 0, 0.12], 0.0, 0.3, 0, R), (head + [0, 0, 0.0], 0.0, 0.1, 10**6, R)],
            200,
        )
        assert info["damaged"] == "flower touched"
        assert not info["success"] and info["wrong_object"]
        assert env.steps < 200, "the episode did not end"
    finally:
        env.close()


def test_cutting_the_main_stem_ends_the_episode_as_a_failure():
    env = _env("prune_one")
    try:
        env.reset(seed=SEEDS[0])
        expert = ScriptedExpert(env, seed=0)
        expert.reset()
        base, up = env.stem_axis()
        point = base + up * 0.05  # below the lowest shoot, above the pot's rim
        toward = np.array([0.0, 1.0, 0.0])  # across the table, the jaws closing horizontally on the upright stem
        R = expert.side_orientation(toward)
        pre = point - toward * 0.14
        info = _drive(
            env,
            expert,
            [
                (np.array([pre[0] - 0.12, pre[1], 0.30]), 1.0, 0.4, 0, R),
                (pre, 1.0, 0.25, 0, R),
                (point, 1.0, 0.10, 2, R),
                (point, 0.0, 0.0, 10**6, R),
            ],
            300,
        )
        assert info["damaged"] == "main stem cut", info
        assert not info["success"] and info["wrong_object"] and info["cut"] == []
    finally:
        env.close()


def test_other_tasks_have_no_plant():
    env = _env("cut_one")
    try:
        assert not env.plant_task and env.model.nu == 8
        assert all(env.model.body(i).name not in ("stand", "plant") for i in range(env.model.nbody))
    finally:
        env.close()
