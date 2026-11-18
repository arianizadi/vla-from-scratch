"""Test situations (sim/pickup/scenarios.py): each scenario applies, keeps the task's layout, draws a seeded variation
and, where it changes the task, has a success rule the expert meets. Without a scenario nothing changes.

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

from sim.pickup import scenarios as S  # noqa: E402
from sim.pickup.env import PANDA_XML, PRESS_TRAVEL, TURN_SPEED, PickupEnv  # noqa: E402
from sim.pickup.expert import ScriptedExpert, random_start  # noqa: E402

SEEDS = (1_000_000, 1_000_001, 1_000_002)


def _env(task, scenario=None):
    if not PANDA_XML.exists():
        pytest.skip("Franka model not fetched (sim/pickup/scripts/fetch_assets.sh)")
    try:
        return PickupEnv(task, scenario=scenario)
    except ValueError:
        raise
    except Exception as error:  # no GL context on this machine
        pytest.skip(f"cannot render: {error}")


def _layout(env):
    """What the task drew for this episode: stand, plate, flower, shoots, instruction."""
    m = env.model
    return dict(
        stand=m.body_pos[env.stand_body][:2].round(6).tolist(),
        plate=round(env.plate_angle(), 6),
        flower=env.flower_colour,
        shoots={
            n: (s["origin"].round(6).tolist(), s["dir"].round(6).tolist(), round(s["length"], 6))
            for n, s in env.shoots.items()
        },
        arm=env.data.qpos[:7].round(6).tolist(),
    )


def _hold(env, steps):
    info = {}
    for _ in range(steps):
        _, done, info = env.step(env.last_action.copy())
        if done:
            break
    return info


def _run_expert(env, seed, start=False):
    obs, yaw = env.reset(seed=seed), 0.0
    expert = ScriptedExpert(env, seed=0)
    if start:
        obs, yaw = random_start(env, expert, seed)
    expert.reset(yaw=yaw)
    done, info = False, {}
    while not done:
        _, done, info = env.step(expert.act())
    return info


# ------------------------------------------------------------------------------------------------------ the spec
def test_parse_names_aliases_modes_and_probabilities():
    assert S.parse(None) == [] and S.parse("") == []
    got = S.parse("B1:dusk, wind@0.25,bump")
    assert [s.name for s in got] == ["lighting", "wind", "bump"]  # bump's hooks run last (priority)
    assert got[0].mode == "dusk" and got[1].prob == 0.25 and got[2].mode is None
    assert [s.spec() for s in got] == ["lighting:dusk", "wind@0.25", "bump"]
    for name in ("A5", "A8", "B2", "B7"):
        assert S.parse(name)[0].catalogue == name
    for bad in ("nope", "wind,wind", "lighting:midnight", "wind@0", "order,nothing_to_do", "order,order_heldout"):
        with pytest.raises((ValueError, AssertionError)):
            S.parse(bad)


def test_scenarios_that_need_a_plant_or_another_task_refuse():
    with pytest.raises(ValueError):
        _env("pick_cube", "wind")
    with pytest.raises(ValueError):
        _env("prune_named", "nothing_to_do")
    with pytest.raises(ValueError):
        _env("prune_turn", "order")  # one shoot: no order to give
    env = _env("pick_cube", "lighting,camera")  # the world-only scenarios apply to any task
    env.reset(seed=SEEDS[0])
    assert set(env.scenario_info) == {"lighting", "camera"}
    env.close()


# --------------------------------------------------------------------------------------------- defaults, layouts
def test_without_a_scenario_nothing_is_recorded_or_flagged():
    env = _env("prune")
    try:
        env.reset(seed=SEEDS[0])
        assert env.scenarios == [] and env.scenario_info == {} and env.flags == set() and env.events == set()
    finally:
        env.close()


ALL = ["lighting", "camera", "wind", "bump", "sticky", "order", "nothing_to_do"]
ASSETS = ["ties", "sick", "buds", "bench", "distractors"]


@pytest.mark.parametrize("scenario", ALL + ASSETS)
def test_a_scenario_keeps_the_layout_the_task_draws_for_the_seed(scenario):
    plain, env = _env("prune"), _env("prune", scenario)
    try:
        for seed in SEEDS:
            plain.reset(seed=seed)
            env.reset(seed=seed)
            assert _layout(env) == _layout(plain), f"{scenario} changed the layout of seed {seed}"
    finally:
        plain.close()
        env.close()


@pytest.mark.parametrize("scenario", ["lighting", "camera", "wind", "bump", "sticky", "order", *ASSETS])
def test_the_variation_is_seeded(scenario):
    env = _env("prune", scenario)
    try:

        def variation(seed):
            env.reset(seed=seed)
            return dict(env.scenario_info[scenario])

        first = [variation(s) for s in range(10)]
        again = [variation(s) for s in range(10)]
        assert first == again, "the same seed drew another variation"
        assert len({str(v) for v in first}) >= 4, "different seeds drew the same variation"
        if len(S.SCENARIOS[scenario].modes) > 1:
            assert len({v["mode"] for v in first}) > 1, "ten seeds drew one mode only"
        assert all(v["applied"] for v in first)
    finally:
        env.close()


def test_a_probability_applies_a_scenario_to_that_share_and_undoes_it_after():
    env, plain = _env("prune", "lighting:noon@0.5"), _env("prune")
    try:
        applied = 0
        for seed in range(40):
            obs = env.reset(seed=seed)
            if env.scenario_info["lighting"]["applied"]:
                applied += 1
            else:  # an episode it does not apply to has the plain task's lights and looks like it
                base = plain.reset(seed=seed)
                for field in ("light_pos", "light_dir", "light_diffuse", "light_ambient", "light_specular"):
                    assert np.array_equal(getattr(env.model, field), getattr(plain.model, field)), field
                assert np.array_equal(env.model.vis.headlight.ambient, plain.model.vis.headlight.ambient)
                # (the renderer itself differs by 1 in a few pixels from one render to the next)
                assert np.abs(obs["front"].astype(int) - base["front"]).max() <= 1
        assert 10 <= applied <= 30
    finally:
        env.close()
        plain.close()


# ----------------------------------------------------------------------------------------------- B situations
@pytest.mark.parametrize("mode", ["dusk", "noon", "grow"])
def test_lighting_changes_what_the_front_camera_sees(mode):
    env, plain = _env("prune", f"lighting:{mode}"), _env("prune")
    try:
        lit, base = env.reset(seed=SEEDS[0])["front"].astype(float), plain.reset(seed=SEEDS[0])["front"].astype(float)
        assert np.abs(lit - base).mean() > 5.0
        if mode == "grow":  # purple: little green left
            assert lit[..., 1].mean() < 0.8 * base[..., 1].mean()
    finally:
        env.close()
        plain.close()


def test_flickering_light_dips_now_and_then():
    env = _env("prune", "lighting:flicker")
    try:
        env.reset(seed=SEEDS[1])
        key = env.scenarios[0].key
        default = env.scenarios[0].defaults["light_diffuse"][key][0]
        levels = []
        for _ in range(80):
            env.step(env.last_action.copy())
            levels.append(env.model.light_diffuse[key][0] / default)
        assert min(levels) < 0.85 and max(levels) > 0.9
    finally:
        env.close()


def test_the_camera_is_knocked_a_few_centimetres_and_degrees():
    env, plain = _env("prune", "camera"), _env("prune")
    try:
        cam = env.model.camera("front").id
        for seed in SEEDS:
            obs, base = env.reset(seed=seed), plain.reset(seed=seed)
            v = env.scenario_info["camera"]
            moved = np.linalg.norm(env.model.cam_pos[cam] - plain.model.cam_pos[cam])
            assert 0.0099 < moved < 0.0401 and 1.0 <= v["angle_deg"] <= 5.0
            assert np.abs(obs["front"].astype(float) - base["front"]).mean() > 1.0
            assert np.array_equal(obs["wrist"], base["wrist"])  # the wrist camera is untouched
        env.close()
        env = _env("prune", "camera:moved_noise")
        a, b = env.reset(seed=SEEDS[0])["front"], env.reset(seed=SEEDS[0])["front"]
        assert np.abs(a.astype(int) - b).max() <= 1, "the noise is not seeded"  # (renders differ by 1 at most)
        assert env.scenario_info["camera"]["noise"] >= 3.0
    finally:
        env.close()
        plain.close()


@pytest.mark.parametrize("mode", ["sway", "gust"])
def test_wind_sways_the_shoots_and_their_outer_parts_follow(mode):
    env = _env("prune", f"wind:{mode}")
    try:
        env.reset(seed=SEEDS[0])
        wind = env.scenarios[0]
        if mode == "gust":  # skip ahead to the gust
            env.steps = int(wind.gust["t0"] * 20)
        name = env.active[0]
        start, outer0 = env.shoots[name]["dir"].copy(), env.object_pos(name)
        angles, moved = [], []
        for _ in range(40):
            env.step(env.last_action.copy())
            angles.append(np.degrees(np.arccos(np.clip(env.shoots[name]["dir"] @ wind.base[name], -1, 1))))
            moved.append(np.linalg.norm(env.object_pos(name) - outer0))
        limit = 11.5 if mode == "sway" else 31.0
        assert 1.0 < max(angles) < limit, f"swayed {max(angles):.1f} degrees"
        assert max(moved) > 0.001, "the outer part did not follow its shoot"
        assert env.damaged is None and not env.cut
        assert np.allclose(np.linalg.norm(start), 1.0)
    finally:
        env.close()


def test_a_bump_turns_the_plate_by_the_drawn_angle_and_says_so():
    env = _env("prune", "bump")
    try:
        for seed in SEEDS[:2]:
            env.reset(seed=seed)
            v = env.scenario_info["bump"]
            start, seen = env.plate_angle(), False
            for _ in range(v["due_step"] + 30):
                env.step(env.last_action.copy())  # the arm holds still at home, clear of the plant
                seen = seen or "bumped" in env.events
            turned = np.degrees(env.plate_angle() - start)
            assert seen and v["bumped_step"] == v["due_step"]
            assert abs(turned - v["angle_deg"]) < 8.0, f"turned {turned:.0f} of {v['angle_deg']:.0f} degrees"
            assert env.damaged is None
    finally:
        env.close()


def _push(env, name, steps, force=6.0):
    body = env.model.body(f"button_{name}").id
    env.data.xfrc_applied[body, 2] = -force
    for _ in range(steps):
        env.step(env.last_action.copy())
    env.data.xfrc_applied[body] = 0
    for _ in range(5):  # let go
        env.step(env.last_action.copy())


def test_a_dead_button_needs_its_extra_presses_and_a_slow_one_turns_slowly():
    env = _env("prune_turn", "sticky:dead")
    try:
        for seed in range(6):
            env.reset(seed=seed)
            v = env.scenario_info["sticky"]
            name = v["buttons"][0]
            sign = 1 if name == "ccw" else -1
            for press in range(v["dead_presses"]):
                before = env.plate_angle()
                _push(env, name, 10)
                assert abs(env.plate_angle() - before) < 0.01, f"dead press {press + 1} turned the plate"
            before = env.plate_angle()
            _push(env, name, 20)
            assert sign * (env.plate_angle() - before) > 0.7 * TURN_SPEED, "the press after the dead ones did nothing"
            assert "sticky" in env.flags
        env.close()
        env = _env("prune_turn", "sticky:slow")
        env.reset(seed=SEEDS[0])
        v = env.scenario_info["sticky"]
        name = v["buttons"][0]
        before = env.plate_angle()
        _push(env, name, 20)
        turned = abs(env.plate_angle() - before)
        assert turned < (v["speed_factor"] + 0.15) * TURN_SPEED and turned > 0.5 * v["speed_factor"] * TURN_SPEED
        assert env.button_travel(name) < PRESS_TRAVEL
    finally:
        env.close()


# ----------------------------------------------------------------------------------------------- A situations
def test_nothing_to_do_holding_still_at_home_succeeds_and_staying_away_fails():
    env = _env("prune", "nothing_to_do")
    try:
        env.reset(seed=SEEDS[0])
        assert env.cut == set(env.active) and len(env.active) >= 2  # every shoot cut already
        assert all(env.object_pos(n)[2] < -0.5 for n in env.active)  # the cut parts are gone
        info = _hold(env, 40)
        assert info["success"] and env.steps == S.HOME_HOLD, "holding still at the home pose is the right answer"
        obs = env.reset(seed=SEEDS[1])
        expert = ScriptedExpert(env, seed=0)
        obs, _ = random_start(env, expert, SEEDS[1])  # away from home, and it stays there
        info = _hold(env, S.NOTHING_MAX_STEPS + 5)
        assert not info["success"] and env.steps == S.NOTHING_MAX_STEPS and info["home_m"] > S.HOME_TOL
    finally:
        env.close()


def test_order_draws_an_instruction_of_its_family_and_cutting_out_of_order_fails():
    env = _env("prune", "order:top_down")
    try:
        env.reset(seed=SEEDS[0])
        assert env.instruction in S.ORDER_FRAMES["top_down"]
        assert env.targets == env.active[::-1] and env.target == env.active[-1]
        wrong_first = env.targets[-1]  # the lowest, which should come last
        env._cut_shoot(wrong_first)
        _, done, info = env.step(env.last_action.copy())
        assert done and not info["success"] and info["out_of_order"] and info["wrong_object"]
        env.close()
        env = _env("prune", "order_heldout:pair")
        env.reset(seed=SEEDS[1])
        assert env.instruction not in {
            f.format(a=a, b=b)
            for f in S.ORDER_FRAMES["pair"]
            for a in ("lowest", "middle", "highest")
            for b in ("lowest", "middle", "highest")
        }
        assert len(env.targets) == 2 and env.phrasing_kind == "heldout"
    finally:
        env.close()


# ---------------------------------------------------------------------------------------- situations with assets
def test_ties_mark_the_shoots_and_the_instruction_names_the_tied_one():
    env = _env("prune", "ties")
    try:
        m = env.model
        for seed in range(6):
            env.reset(seed=seed)
            v = env.scenario_info["ties"]
            shown = {n for n, g in env.scenarios[0].ties.items() if m.geom_group[g] == 2}
            assert shown == set(v["ties"]) and (len(shown) == 1) == (v["mode"] == "one")
            assert len(env.targets) == 1 and v["ties"][env.targets[0]] == v["colour"]
            if v["mode"] == "colour":
                assert v["colour"] in env.instruction and len(set(v["ties"].values())) == len(shown)
            for n, c in v["ties"].items():
                assert np.allclose(m.geom_rgba[env.scenarios[0].ties[n]], S.manifest()["assets"]["tie"]["colours"][c])
    finally:
        env.close()


def test_sick_shoots_look_sick_and_are_the_ones_to_cut():
    env, plain = _env("prune", "sick"), _env("prune")
    try:
        counts = set()
        for seed in range(12):
            env.reset(seed=seed)
            plain.reset(seed=seed)
            v = env.scenario_info["sick"]
            counts.add(v["sick"])
            assert sorted(env.targets) == sorted(v["looks"])
            for n in env.active:
                vis, base = env.leaf_geoms[n][0][0], plain.leaf_geoms[n][0][0]  # (the models' geom ids differ)
                same = np.array_equal(env.model.geom_rgba[vis], plain.model.geom_rgba[base])
                assert same == (n not in env.targets)
        assert {0, 1, 2} <= counts
    finally:
        env.close()
        plain.close()


def test_buds_sit_on_the_shoots_to_keep():
    env = _env("prune", "buds")
    try:
        m = env.model
        for seed in range(8):
            env.reset(seed=seed)
            budding = {n for n, (g, _, col) in env.scenarios[0].buds.items() if m.geom_group[g] == 2}
            assert budding == set(env.active) - set(env.targets) and len(budding) == env.scenario_info["buds"]["buds"]
            for n in budding:  # a solid bud at the tip, two leaves
                assert m.geom_contype[env.scenarios[0].buds[n][2]] == 2
                assert len(S._leaves_in_use(env, n)) == 2
    finally:
        env.close()


def test_the_bench_and_the_greenhouse_replace_the_table_and_the_dark_backdrop():
    env, plain = _env("prune", "bench:stained_greenhouse"), _env("prune")
    try:
        obs, base = env.reset(seed=SEEDS[0]), plain.reset(seed=SEEDS[0])
        b = env.scenarios[0]
        m = env.model
        assert m.geom_group[b.table] == 3 and m.geom_group[b.bench] == 2 and all(m.geom_group[b.backdrop] == 2)
        assert np.abs(obs["front"].astype(float) - base["front"]).mean() > 10
    finally:
        env.close()
        plain.close()


def test_distractors_stand_on_the_table_clear_of_the_stand_and_are_watched():
    env = _env("prune", "distractors")
    try:
        for seed in SEEDS:
            env.reset(seed=seed)
            d = env.scenarios[0]
            assert 2 <= len(d.placed) <= 4
            stand = env.model.body_pos[env.stand_body][:2]
            for n in d.placed:
                xyz = env.data.qpos[d.qadr[n] : d.qadr[n] + 3]
                assert abs(xyz[2]) < 0.01 and np.linalg.norm(xyz[:2] - stand) > 0.11 + d.radius[n]  # resting, clear
            info = _hold(env, 3)
            assert info["distractors_touched"] == [] and info["distractors_moved"] == []
    finally:
        env.close()


# --------------------------------------------------------------------------------------------- the expert
@pytest.mark.parametrize(
    "task, scenario",
    [
        ("prune", "nothing_to_do"),
        ("prune", "order"),
        ("prune_turn", "bump"),
        ("prune_turn", "sticky"),
        ("prune_turn", "wind"),
        ("prune", "ties"),
        ("prune", "sick"),
        ("prune", "buds"),
    ],
)
def test_the_expert_solves_the_situation(task, scenario):
    env = _env(task, scenario)
    try:
        wins = 0
        for seed in SEEDS:
            info = _run_expert(env, seed, start=scenario == "nothing_to_do")
            assert info["damaged"] is None, f"seed {seed}: {info['damaged']} ({env.scenario_info})"
            wins += info["success"]
            if scenario == "order" and info["success"]:
                assert info["cut_order"] == env.targets
        assert wins >= 2, f"the expert solved {wins} of {len(SEEDS)} ({scenario})"
    finally:
        env.close()


# ------------------------------------------------------------------------------------------------- held-out splits
@pytest.mark.parametrize("scenario", ["lighting", "camera", "bench", "distractors"])
def test_the_train_and_test_splits_do_not_overlap(scenario):
    train, test = _env("prune", f"{scenario}/train"), _env("prune", f"{scenario}/test")
    try:
        seen = {"train": [], "test": []}
        for seed in range(30):
            for split, env in (("train", train), ("test", test)):
                env.reset(seed=seed)
                v = env.scenario_info[scenario]
                assert v["split"] == split and v["mode"] in S.SCENARIOS[scenario].splits[split]
                seen[split].append(v)
        if scenario == "camera":
            for split, (lo, hi) in (("train", (0.0099, 0.0301)), ("test", (0.0299, 0.0401))):
                assert all(lo < np.linalg.norm(v["shift_m"]) < hi for v in seen[split])
            assert max(v["angle_deg"] for v in seen["train"]) <= 3.5 <= min(v["angle_deg"] for v in seen["test"])
        if scenario == "distractors":
            used = {s: {n for v in seen[s] for n in v["objects"]} for s in seen}
            assert not used["train"] & used["test"] and all("bee_on" in v for v in seen["test"])
            assert not any("bee_on" in v for v in seen["train"])
        if scenario in ("lighting", "bench"):
            assert not {v["mode"] for v in seen["train"]} & {v["mode"] for v in seen["test"]}
    finally:
        train.close()
        test.close()


# ------------------------------------------------------------------------------------------------- the leaf fix
def _drawn_leaf(env, vis):
    """World positions of a leaf geom's vertices as drawn."""
    m, d = env.model, env.data
    mid = m.geom_dataid[vis]
    v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
    return d.geom_xpos[vis] + v @ d.geom_xmat[vis].reshape(3, 3).T


def _pair(task):
    """The same task with and without the leaf fix."""
    _env(task).close()  # (skips where the simulator cannot run)
    return PickupEnv(task, leaf_fix=True), PickupEnv(task)


def test_with_the_leaf_fix_each_leaf_is_drawn_on_its_collision_box():
    env, old = _pair("prune")
    try:
        for seed in SEEDS:
            env.reset(seed=seed)
            old.reset(seed=seed)
            for n in env.active:
                for (vis, col), (ovis, ocol) in zip(env.leaf_geoms[n], old.leaf_geoms[n], strict=True):
                    if env.model.geom_group[vis] != 2:
                        continue
                    pts = _drawn_leaf(env, vis)
                    box, axes = env.data.geom_xpos[col], env.data.geom_xmat[col].reshape(3, 3)
                    local = (pts - box) @ axes  # in the box's frame
                    assert np.linalg.norm(pts.mean(0) - box) < 0.005  # the blade is centred on its box
                    # along it: the box (38 x 16 x 3 mm) covers the blade but for the last 9 mm of the petiole
                    assert local[:, 0].min() > -0.029 and local[:, 0].max() < 0.021
                    assert np.all(np.abs(local[:, 1]) < 0.011) and np.all(np.abs(local[:, 2]) < 0.007)
                    before = _drawn_leaf(old, ovis)
                    assert np.linalg.norm(before.mean(0) - old.data.geom_xpos[ocol]) > 0.02  # as it was drawn
                    assert np.array_equal(env.data.geom_xpos[col], old.data.geom_xpos[ocol])  # same collision box
    finally:
        env.close()
        old.close()


def test_the_leaf_fix_changes_what_is_seen_and_nothing_else():
    env, old = _pair("prune_turn")
    try:
        differ = False
        for e in (env, old):
            e.reset(seed=SEEDS[0])
        a, b = ScriptedExpert(env, seed=0), ScriptedExpert(old, seed=0)
        a.reset()
        b.reset()
        done = False
        while not done:
            act = a.act()
            assert np.array_equal(act, b.act())
            obs, done, info = env.step(act)
            obs_old, done_old, info_old = old.step(act)
            assert np.array_equal(env.data.qpos, old.data.qpos) and done == done_old and info == info_old
            differ = differ or not np.array_equal(obs["front"], obs_old["front"])
        assert differ and info["success"]
    finally:
        env.close()
        old.close()


# ------------------------------------------------------------------------------------------- paired re-rendering
def test_a_replay_reproduces_the_states_and_a_look_changes_only_the_images():
    from io import BytesIO

    from PIL import Image

    from sim.pickup.rerender import Replayer

    env = _env("prune_turn")
    env.close()
    info = dict(sim_task="prune_turn", leaf_fix=True, random_start=True)
    rec = PickupEnv("prune_turn", leaf_fix=True)
    try:
        seed = SEEDS[1]
        rec.reset(seed=seed)
        expert = ScriptedExpert(rec, seed=0)
        obs, yaw = random_start(rec, expert, seed)
        expert.reset(yaw=yaw)
        states, actions, fronts, done = [], [], [], False
        while not done:
            states.append(obs["state"])
            fronts.append(obs["front"])
            a = expert.act()
            actions.append(a)
            obs, done, _ = rec.step(a)
        replayer = Replayer(info)
        frames = replayer.trajectory(seed, np.array(states), np.array(actions))
        assert len(frames) == len(actions)
        with pytest.raises(RuntimeError):  # a state that is not reproduced is caught
            replayer.trajectory(seed, np.array(states) + 1e-3, np.array(actions))
        same, _ = replayer.render(seed, 7, "", frames[:40:10])  # no look: the source's own images
        for img, src in zip(same["front"], fronts[:40:10], strict=True):
            decoded = np.asarray(Image.open(BytesIO(img))).astype(int)
            assert np.abs(decoded - src).mean() < 2.0  # (JPEG)
        bench, variation = replayer.render(seed, 7, "bench/train", frames[:40:10])
        assert variation["bench"]["mode"] in ("wood", "stained")
        assert bench["front"] != same["front"]
        again, _ = replayer.render(seed, 7, "bench/train", frames[:40:10])
        for a, b in zip(again["front"], bench["front"], strict=True):  # the same look for the same scenario seed
            diff = np.abs(np.asarray(Image.open(BytesIO(a))).astype(int) - np.asarray(Image.open(BytesIO(b))))
            assert diff.mean() < 0.05  # (the renderer's own noise of a level in a few pixels, through JPEG)
        # the plain look at another resolution: the same scene, finer (here coarser, to keep the test quick)
        small, variation = Replayer(info, image_size=112).render(seed, 7, "plain", frames[:40:10])
        assert variation == {}
        for img, src in zip(small["front"], fronts[:40:10], strict=True):
            decoded = np.asarray(Image.open(BytesIO(img))).astype(int)
            assert decoded.shape == (112, 112, 3)
            shrunk = np.asarray(Image.fromarray(src).resize((112, 112), Image.BILINEAR)).astype(int)
            assert np.abs(decoded - shrunk).mean() < 6.0  # (the same scene; anti-aliasing and JPEG differ)
    finally:
        rec.close()


# ------------------------------------------------------------------------------------------- the parked shoot
def _model_arrays(env):
    m = env.model
    return {f: getattr(m, f).copy() for f in ("body_ipos", "geom_pos", "geom_size", "geom_contype", "eq_data")}


@pytest.mark.parametrize("park_reset", [False, True])
def test_with_park_reset_an_episode_does_not_depend_on_the_ones_before(park_reset):
    _env("prune").close()  # (skips where the simulator cannot run)
    used, fresh = PickupEnv("prune", park_reset=park_reset), PickupEnv("prune", park_reset=park_reset)
    try:
        three = next(s for s in range(50) if len(fresh.reset(seed=s) and fresh.active) == 3)
        two = next(s for s in range(50) if len(fresh.reset(seed=s) and fresh.active) == 2)
        expert = ScriptedExpert(used, seed=0)
        used.reset(seed=three)
        expert.reset()
        done = False
        while not done:  # an episode that uses the third shoot, cuts it, and leaves it on the floor or the plate
            _, done, _ = used.step(expert.act())
        used.reset(seed=two)
        fresh.reset(seed=two)
        same = all(
            np.array_equal(a, b)
            for a, b in zip(_model_arrays(used).values(), _model_arrays(fresh).values(), strict=True)
        )
        if not park_reset:  # the history the flag removes: the unused shoot keeps the last episode's shape
            assert not same
            return
        assert same
        experts = [ScriptedExpert(e, seed=0) for e in (used, fresh)]
        for x in experts:
            x.reset()
        done = False
        while not done:
            a, b = experts[0].act(), experts[1].act()
            assert np.array_equal(a, b)
            _, done, _ = used.step(a)
            fresh.step(b)
            assert np.array_equal(used.data.qpos, fresh.data.qpos)
    finally:
        used.close()
        fresh.close()
