"""The full-3D-reach situations (sim/pickup/scenarios_reach.py) and the arm they ask (sim/pickup/arm.py): the plate is
locked and the buttons gone, every layout is one the arm can cut, each situation draws what it says, the expert solves
them, and the expert's settling rule closes the jaws nearer the shoot. Without these scenarios nothing changes (the
expert's own tests and test_scenarios.py cover that).

They need MuJoCo, a GL context for the cameras and the Franka model (sim/pickup/scripts/fetch_assets.sh), and are
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

from sim.pickup import env as E  # noqa: E402
from sim.pickup import scenarios as S  # noqa: E402
from sim.pickup import scenarios_reach as R  # noqa: E402
from sim.pickup.arm import arm_for, segment_frame  # noqa: E402
from sim.pickup.env import PANDA_XML, PickupEnv  # noqa: E402
from sim.pickup.expert import ScriptedExpert, random_start  # noqa: E402

SEEDS = (1_000_000, 1_000_001)
REACH = ["locked", "heights", "pitched", "blocked"]


def _env(scenario=None, task="prune"):
    if not PANDA_XML.exists():
        pytest.skip("Franka model not fetched (sim/pickup/scripts/fetch_assets.sh)")
    try:
        return PickupEnv(task, scenario=scenario, leaf_fix=True, park_reset=True)
    except ValueError:
        raise
    except Exception as error:  # no GL context on this machine
        pytest.skip(f"cannot render: {error}")


def _run(env, seed, settle=False):
    env.reset(seed=seed)
    expert = ScriptedExpert(env, seed=0, settle=settle)
    _, yaw = random_start(env, expert, seed)
    expert.reset(yaw=yaw)
    done, info, cuts = False, {}, []
    while not done:
        before = set(env.cut)
        _, done, info = env.step(expert.act())
        cuts += [R.cut_error(env, n) for n in sorted(set(env.cut) - before)]
    return info, cuts


def test_the_names_parse_and_change_the_task_one_at_a_time():
    got = S.parse("R1")
    assert [s.name for s in got] == ["locked"] and got[0].changes_task
    assert {s.name for s in S.parse("heights,lighting:dusk")} == {"lighting", "heights"}
    with pytest.raises(ValueError):
        S.parse("locked,ties")  # two situations that set the task
    with pytest.raises(ValueError):
        _env("locked", task="prune_named")


@pytest.mark.parametrize("scenario", REACH)
def test_the_plate_is_locked_the_buttons_gone_and_every_shoot_can_be_cut(scenario):
    env = _env(scenario)
    try:
        env.reset(seed=SEEDS[0])
        assert {"reach3d", "locked"} <= env.flags
        m = env.model
        for b in E.BUTTONS:
            assert m.geom_contype[m.geom(f"button_{b}_col").id] == 0 and m.geom_group[m.geom(f"button_{b}_vis").id] == 3
        # the arm has a way in to every shoot, as the draw found
        assert env.scenario_info[scenario]["draws"] >= 1
        assert all(R.plan_cut(env, n) is not None for n in env.active)
        # pushing the plate hard does not turn it
        start = env.plate_angle()
        for _ in range(20):
            env.data.qfrc_applied[env.plate_vadr] = 2.0
            env.step(env.last_action.copy())
        assert abs(env.plate_angle() - start) < np.radians(0.5)
        env.data.qfrc_applied[:] = 0
    finally:
        env.close()


@pytest.mark.parametrize("scenario", REACH)
def test_the_draw_is_seeded_and_points_the_shoots_all_round(scenario):
    env = _env(scenario)
    try:
        first, azimuths = [], []
        for seed in range(6):
            env.reset(seed=seed)
            first.append(str(env.scenario_info[scenario]))
            azimuths += [env.shoot_azimuth(n) for n in env.active]
        again = []
        for seed in range(6):
            env.reset(seed=seed)
            again.append(str(env.scenario_info[scenario]))
        assert first == again and len(set(first)) == 6
        behind = [a for a in azimuths if abs(E._wrap_angle(a - E.REACH_AZIMUTH)) > np.radians(90)]
        assert behind, "no shoot pointed away from the robot in six layouts"
    finally:
        env.close()


def test_without_the_scenario_the_buttons_and_the_plate_are_as_before():
    plain, env = _env(), _env("locked@0.5")
    try:
        for seed in range(8):
            env.reset(seed=seed)
            if not env.scenario_info["locked"]["applied"]:
                plain.reset(seed=seed)
                hinge = env.model.joint("plate_hinge").dofadr[0]
                assert env.model.dof_damping[hinge] == plain.model.dof_damping[hinge]
                assert env.model.geom_contype[env.model.geom("button_ccw_col").id] == 1
                return
        pytest.fail("the scenario applied to all eight layouts at p=0.5")
    finally:
        plain.close()
        env.close()


def test_heights_grows_the_stem_and_spreads_the_shoots_over_it():
    env = _env("heights")
    try:
        scales, lows, highs = set(), [], []
        for seed in range(8):
            env.reset(seed=seed)
            v = env.scenario_info["heights"]
            k = v["stem_scale"]
            scales.add(k)
            assert env.stem_h == pytest.approx(k * E.STEM_H)
            assert env.model.geom_size[env.stem_geom][1] == pytest.approx(env.stem_h / 2)
            hs = [env.shoots[n]["origin"][2] - E.STEM_BASE for n in env.active]
            assert min(hs) >= R.LOWEST - 1e-9 and max(hs) <= env.stem_h - R.BELOW_FLOWER + 1e-9
            lows.append(min(hs))
            highs.append(max(hs) / env.stem_h)
        assert len(scales) >= 3 and min(lows) < 0.06 and max(highs) > 0.6
    finally:
        env.close()


def test_a_cut_outside_the_height_band_fails_the_episode():
    env = _env("heights")
    try:
        env.reset(seed=SEEDS[0])
        scenario = env.scenarios[0]
        name = env.active[0]
        a, d = env.shoot_axis(name)
        _, b2 = segment_frame(d)
        # the jaws' centre 15 mm above the shoot's axis (moved there by hand): outside the 10 mm band
        assert not scenario.cut_ok(_AtOffset(env, a + d * 0.035 + b2 * 0.015), name)
        assert scenario.cut_ok(_AtOffset(env, a + d * 0.035 + b2 * 0.004), name)
    finally:
        env.close()


class _AtOffset:
    """An environment whose gripper centre is at `tcp` (for cut_error), everything else as env's."""

    def __init__(self, env, tcp):
        self._env, self._tcp = env, np.asarray(tcp)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def tcp_pos(self):
        return self._tcp.copy()


def test_pitched_shoots_are_steep_and_the_jaws_pitch_with_them():
    env = _env("pitched")
    try:
        env.reset(seed=SEEDS[0])
        for n in env.active:
            _, d = env.shoot_axis(n)
            assert np.radians(39) <= abs(np.arcsin(d[2])) <= np.radians(61)
            plan = R.plan_cut(env, n)
            assert abs(d @ plan.R[:, 1]) < np.cos(np.radians(45))  # the jaws close across the shoot
    finally:
        env.close()


def test_a_blocking_leaf_lies_along_a_shoot_on_the_side_away_from_the_way_in_and_touching_it_damages():
    env = _env("blocked")
    try:
        env.reset(seed=SEEDS[0])
        scenario = env.scenarios[0]
        sides = env.scenario_info["blocked"]["leaf"]
        assert scenario.used and len(scenario.used) == sum(side != "none" for side in sides)
        blades = iter(scenario.used)
        for n, side in zip(env.active, sides, strict=True):
            if side == "none":
                continue
            blade = next(blades)
            a, d = env.shoot_axis(n)
            _, b2 = segment_frame(d)
            assert (env.data.geom_xpos[blade] - a) @ b2 * (1 if side == "above" else -1) > R.BLOCK_GAP[0]
            plan = R.plan_cut(env, n)
            assert plan.kind != "legacy"  # jaws closing up and down would meet the leaf
            assert plan.kind == ("below" if side == "above" else "above") or abs(np.sin(plan.phi)) > 0.3
        # a leaf moved onto the open jaws (the arm where the episode starts, well clear of the flower): damage
        blade = scenario.used[0]
        plant = env.plant_body
        Rp, pp = env.data.xmat[plant].reshape(3, 3), env.data.xpos[plant]
        env.model.geom_pos[blade] = Rp.T @ (env.tcp_pos() - pp)
        info = {}
        for _ in range(3):
            _, done, info = env.step(env.last_action.copy())
            if done:
                break
        assert info["damaged"] == "leaf touched" and not info["success"]
    finally:
        env.close()


@pytest.mark.parametrize("scenario", REACH)
def test_the_expert_solves_the_situation_from_random_starts(scenario):
    env = _env(scenario)
    try:
        wins = 0
        for seed in SEEDS:
            info, cuts = _run(env, seed)
            assert info["damaged"] is None, f"seed {seed}: {info['damaged']}"
            wins += info["success"]
            assert len(cuts) == len(info["cut"])
        assert wins >= 1
    finally:
        env.close()


def test_settling_closes_the_jaws_nearer_the_shoot_in_height():
    env = _env("locked")
    try:
        z0, z1 = [], []
        for seed in SEEDS:
            _, cuts = _run(env, seed)
            z0 += [abs(c["jaws_mm"]) for c in cuts]
            _, cuts = _run(env, seed, settle=True)
            z1 += [abs(c["jaws_mm"]) for c in cuts]
        assert np.mean(z1) < np.mean(z0)
    finally:
        env.close()


def test_the_cut_orientation_puts_the_shoot_across_the_jaws():
    env = _env("locked")
    try:
        env.reset(seed=SEEDS[0])
        arm = arm_for(env)
        rng = np.random.default_rng(0)
        for _ in range(20):
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            for phi in np.radians([0, 60, 150, 250]):
                for psi in np.radians([0, -30]):
                    R = arm.cut_orientation(d, phi, psi, arm.R_down)
                    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.linalg.det(R) > 0
                    assert abs(d @ R[:, 1]) < 1e-9  # the closing axis at right angles to the shoot
    finally:
        env.close()
