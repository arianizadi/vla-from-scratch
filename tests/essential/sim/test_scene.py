"""Scene records, the simulator's truth for them, the plant crop, the scene memory and the perception dataset
(sim/pickup/scene.py, sim/pickup/collect_scene.py; architecture v2).

The record and memory tests need nothing but numpy: hand-made scenes stand in for what the eyes would report. The
others need MuJoCo, a GL context for the cameras and the Franka model (sim/pickup/scripts/fetch_assets.sh), and are
skipped where any of them is missing.
"""

import io
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:  # sim/ is not an installed package
    sys.path.insert(0, str(REPOSITORY))

from sim.pickup import reasoning as R  # noqa: E402
from sim.pickup import scene as S  # noqa: E402

SEEDS = (1_000_000, 1_000_001, 1_000_002)


def _part(h, b=0.0, cut=False, vis=True, p=0.95, p_cut=None):
    """A part as the eyes would report it."""
    return S.Part(h, b, cut, vis, p=p, p_cut=p_cut, p_visible=0.9 if vis else 0.1)


def _scene(*parts, flower="red", label="none", p_flower=0.95, p_label=0.95):
    return S.Scene(list(parts), flower, label, p_flower, p_label)


def _feed(memory, scene, times, step=0):
    """`times` updates with the same scene; what each returned."""
    return [memory.update(scene, step + k) for k in range(times)]


# ------------------------------------------------------------------------------------------------------- records
def test_truth_and_percept_records_round_trip_through_plain_dicts_and_json():
    truth = S.Scene(
        [
            S.Part(0.0834, -1.2051, False, True, views=(True, False)),
            S.Part(0.1203, 2.7, True, False, views=(False, False)),
        ],
        "purple",
        "none",
    )
    d = truth.to_dict()
    assert d == {
        "parts": [
            {"h": 0.0834, "b": -1.2051, "cut": False, "vis": [True, False]},
            {"h": 0.1203, "b": 2.7, "cut": True, "vis": [False, False]},
        ],
        "flower": "purple",
        "label": "none",
    }
    assert json.loads(truth.to_json()) == d
    assert S.Scene.from_dict(d) == truth and S.Scene.from_json(truth.to_json()) == truth

    percept = S.Scene(
        [
            S.Part(0.0834, -1.2051, False, True, p=0.97, p_cut=0.03, p_visible=0.91),
            S.Part(0.0, 0.0, False, False, p=0.02, p_cut=0.5, p_visible=0.01),
        ],
        "red",
        "unreadable",
        p_flower=0.88,
        p_label=0.6,
    )
    d = percept.to_dict()
    first = {"h": 0.0834, "b": -1.2051, "cut": False, "vis": True, "p": 0.97, "p_cut": 0.03, "p_vis": 0.91}
    assert d["parts"][0] == first and d["p_flower"] == 0.88 and d["p_label"] == 0.6
    assert S.Scene.from_dict(d) == percept and S.Scene.from_json(percept.to_json()) == percept

    # numbers are stored to 4 decimals; a truth part read back is visible when either camera shows it
    assert S.Part(0.083456, 0.123456, False, True, views=(False, True)).to_dict()["h"] == 0.0835
    back = S.Part.from_dict({"h": 0.1, "b": 0.0, "cut": False, "vis": [False, True]})
    assert back.visible and back.views == (False, True) and back.p == 1.0 and back.p_cut is None
    # an unknown flower and an attribute survive the trip; a label that is not a card state is refused
    odd = S.Scene([S.Part(0.1, 0.0, False, True, p=0.9, attributes={"tie": "red"})], None, "cyan", p_flower=0.3)
    assert S.Scene.from_json(odd.to_json()) == odd
    with pytest.raises(AssertionError):
        S.Scene([], "red", "green")


def test_the_card_states_are_no_card_an_unreadable_one_and_the_colours_that_name_a_shoot():
    cards = S.LABELS
    assert cards == ("none", "unreadable", *R.LABEL_MEANS)


def test_the_records_and_the_memory_need_no_simulator():
    code = "import sys, sim.pickup.scene; assert not {'mujoco', 'sim.pickup.env'} & set(sys.modules), sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=REPOSITORY, check=True)


# --------------------------------------------------------------------------------------------------------- memory
def test_unsure_or_unseen_parts_do_not_count():
    memory = S.SceneMemory(stable=2, p_min=0.7)
    _feed(memory, _scene(_part(0.08, p=0.5), _part(0.12, vis=False)), 5)
    assert memory.parts == [] and not memory.settled
    _feed(memory, _scene(_part(0.08, p=0.7), _part(0.12, vis=False)), 2, step=5)
    assert [r.height for r in memory.parts] == [pytest.approx(0.08)]


def test_a_part_is_added_after_stable_sightings_in_a_row():
    memory = S.SceneMemory(stable=3)
    seen, hidden = _scene(_part(0.10)), _scene(_part(0.10, vis=False))
    _feed(memory, seen, 2)
    memory.update(hidden, 2)  # the row is broken
    _feed(memory, seen, 2, step=3)
    assert memory.parts == []
    memory.update(seen, 5)
    (part,) = memory.parts
    assert part.id == 0 and part.sightings == 3 and part.seen_step == 5 and part.height == pytest.approx(0.10)


def test_identity_is_the_height_and_survives_the_plant_turning_and_a_part_out_of_view():
    memory = S.SceneMemory(stable=2)
    bearings = np.linspace(-3.0, 3.0, 40)  # the plate turns the plant round
    for k in range(40):
        lower = _part(0.08 + 0.004 * np.sin(k), bearings[k])  # the eyes read its height a few mm off
        upper = _part(0.125, -bearings[k], vis=not 10 <= k < 25)  # out of view for a while
        memory.update(_scene(lower, upper), k)
        if k >= 1:
            assert [r.id for r in memory.parts] == [0, 1], k
    low, high = memory.parts
    assert low.bearing == pytest.approx(bearings[-1]) and high.bearing == pytest.approx(-bearings[-1])
    assert low.sightings == 40 and high.sightings == 25 and high.seen_step == 39
    assert low.height == pytest.approx(0.08, abs=0.001) and high.height == pytest.approx(0.125)


def test_parts_a_shoot_spacing_apart_stay_apart_and_a_sighting_1cm_off_is_the_same_part():
    memory = S.SceneMemory(stable=2)
    _feed(memory, _scene(_part(0.100)), 2)
    _feed(memory, _scene(_part(0.100), _part(0.135)), 2, step=2)  # 3.5 cm apart: another part
    assert [r.id for r in memory.parts] == [0, 1]
    _feed(memory, _scene(_part(0.110), _part(0.145)), 4, step=4)  # both read 1 cm high
    assert [r.id for r in memory.parts] == [0, 1] and [r.sightings for r in memory.parts] == [8, 6]
    together = S.SceneMemory(stable=2)
    _feed(together, _scene(_part(0.100), _part(0.135)), 3)  # seen together from the start
    assert len(together.parts) == 2


def test_a_cut_needs_stable_cut_sightings_in_a_row_and_is_never_undone():
    memory = S.SceneMemory(stable=3)
    _feed(memory, _scene(_part(0.1)), 3)
    _feed(memory, _scene(_part(0.1, cut=True)), 2, step=3)
    memory.update(_scene(_part(0.1, cut=True, p_cut=0.5)), 5)  # not sure it is cut: the row is broken
    _feed(memory, _scene(_part(0.1, cut=True)), 2, step=6)
    assert not memory.parts[0].cut
    memory.update(_scene(_part(0.1, cut=True, vis=False)), 8)  # out of view: neither counts nor breaks the row
    assert not memory.parts[0].cut
    memory.update(_scene(_part(0.1, cut=True, p_cut=0.9)), 9)
    assert memory.parts[0].cut
    _feed(memory, _scene(_part(0.1, cut=False)), 5, step=10)
    assert memory.parts[0].cut


def test_a_part_that_leaves_the_view_is_not_marked_cut_and_stays_listed():
    memory = S.SceneMemory(stable=2)
    _feed(memory, _scene(_part(0.08), _part(0.12)), 2)
    _feed(memory, _scene(_part(0.08)), 20, step=2)  # the upper one hidden, or missed by the eyes
    assert len(memory.parts) == 2 and not any(r.cut for r in memory.parts)
    assert memory.text() == "2 shoots, red flower"


def test_the_card_is_debounced_and_its_colour_is_kept_once_read():
    memory = S.SceneMemory(stable=2)
    shoots = (_part(0.08), _part(0.115), _part(0.15))

    def card(label, p=0.95):
        return _scene(*shoots, flower="pink", label=label, p_label=p)

    _feed(memory, card("unreadable"), 2)
    assert memory.see_text() == "3 shoots, pink flower, label not seen"
    memory.update(card("cyan"), 2)
    assert memory.label_read is None and memory.label_now == "unreadable"
    memory.update(card("cyan"), 3)
    assert memory.see_text() == "3 shoots, pink flower, label cyan"
    _feed(memory, card("unreadable"), 2, step=4)  # the plate turned the card away
    assert memory.label_now == "unreadable" and memory.label_read == "cyan"
    assert memory.see_text() == "3 shoots, pink flower, label hidden, read cyan"
    _feed(memory, card("none"), 2, step=6)  # read once, a card it remains
    assert memory.see_text() == "3 shoots, pink flower, label hidden, read cyan"
    _feed(memory, card("white", p=0.5), 5, step=8)  # unsure readings change nothing
    memory.update(card("white"), 13)
    assert memory.label_read == "cyan"
    memory.update(card("white"), 14)
    assert memory.label_read == memory.label_now == "white"


def test_the_flower_is_known_after_stable_confident_percepts_and_changes_only_after_as_many_of_another():
    memory = S.SceneMemory(stable=3)
    _feed(memory, _scene(flower="red"), 2)
    assert memory.flower is None and memory.see_text() == "0 shoots, flower not seen"
    memory.update(_scene(flower="red"), 2)
    assert memory.flower == "red"
    _feed(memory, _scene(flower="purple"), 2, step=3)
    memory.update(_scene(flower="red"), 5)  # breaks the row
    _feed(memory, _scene(flower="purple"), 2, step=6)
    assert memory.flower == "red"
    _feed(memory, _scene(flower="purple", p_flower=0.4), 5, step=8)  # unsure: neither counts nor breaks the row
    _feed(memory, _scene(flower=None), 2, step=13)  # no colour given
    assert memory.flower == "red"
    memory.update(_scene(flower="purple"), 15)
    assert memory.flower == "purple"


def test_settled_waits_for_stable_updates_without_growth_among_those_that_count_a_part():
    memory = S.SceneMemory(stable=2)
    two = _scene(_part(0.08), _part(0.12))
    three = _scene(_part(0.08), _part(0.12), _part(0.155))
    nothing = _scene()  # the arm hides the plant
    states = []
    for scene in (nothing, two, two, two, nothing, two, three, three, three, nothing, three):
        memory.update(scene, len(states))
        states.append((len(memory.parts), memory.settled))
    assert states == [
        (0, False),  # nothing counted
        (0, False),  # two candidates
        (2, False),  # the inventory grew
        (2, False),  # one counted update without growth
        (2, False),  # an update that counts no part says nothing
        (2, True),  # two counted updates without growth
        (2, True),  # a third shoot is a candidate: the inventory has not grown
        (3, False),  # now it has
        (3, False),
        (3, False),
        (3, True),
    ]


def test_positions_are_named_over_the_whole_inventory_cut_or_not():
    memory = S.SceneMemory(stable=1, p_min=0.0)
    memory.update(_scene(_part(0.08, cut=True), _part(0.115), _part(0.15)), 0)
    low, mid, high = memory.parts
    assert [memory.position(r.id) for r in memory.parts] == ["lowest", "middle", "highest"]
    assert memory.by_position("lowest") is low and memory.by_position("middle") is mid
    assert memory.by_position("highest") is high
    assert memory.text() == "3 shoots, red flower; cut: the lowest"
    memory.update(_scene(_part(0.08, cut=True), _part(0.115, cut=True), _part(0.15)), 1)
    assert memory.position(high.id) == "highest" and memory.by_position("middle") is mid
    assert memory.text() == "3 shoots, red flower; cut: the lowest and the middle"
    two = S.SceneMemory(stable=1, p_min=0.0)
    two.update(_scene(_part(0.09), _part(0.14)), 0)
    assert two.by_position("middle") is None and two.position(two.parts[1].id) == "highest"
    with pytest.raises(KeyError):
        two.position(7)


def _see(task, flower, label=None):
    """The `see` field of the expert's note for these facts."""
    return R.note_fields(R.note_text(task, flower, label=label, aim=None))["see"]


def test_the_memory_writes_what_it_sees_as_the_notes_do():
    ordinal = R.Task("ordinal", "x", "train", 2, (0,), "second from the top", dict(k=2, end="top"))
    memory = S.SceneMemory(stable=1, p_min=0.0)
    memory.update(_scene(_part(0.09), _part(0.14), flower="orange"), 0)
    assert memory.see_text() == _see(ordinal, "orange") == "2 shoots, orange flower"

    hidden = R.Task("hidden", "x", "train", 3, (2,), "the shoot the label names", dict(label="cyan"))
    shoots = (_part(0.08), _part(0.115), _part(0.15))
    memory = S.SceneMemory(stable=1, p_min=0.0)
    for label, note_label in (
        ("unreadable", None),  # a card, facing away
        ("cyan", "visible:cyan"),
        ("unreadable", "hidden:cyan"),
        ("none", "hidden:cyan"),  # out of view altogether: still the card that was read
    ):
        memory.update(_scene(*shoots, flower="purple", label=label), 0)
        assert memory.see_text() == _see(hidden, "purple", note_label), label
    assert memory.text() == "3 shoots, purple flower, label hidden, read cyan"
    memory.update(_scene(_part(0.08, cut=True), *shoots[1:], flower="purple", label="unreadable"), 1)
    assert memory.text() == "3 shoots, purple flower, label hidden, read cyan; cut: the lowest"


def test_with_stable_1_and_no_threshold_one_update_with_the_truth_is_the_truth():
    truth = S.Scene(
        [
            S.Part(0.0712, 3.0, True, True, views=(True, False)),
            S.Part(0.1105, -0.4, False, True, views=(False, True)),
            S.Part(0.1530, 1.2, False, True, views=(True, True)),
        ],
        "pink",
        "yellow",
    )
    expected = [(p.height, p.bearing, p.cut) for p in truth.parts]
    for given in (truth, S.Scene.from_json(truth.to_json())):  # as scene_truth gives it, and as a dataset stores it
        memory = S.SceneMemory(stable=1, p_min=0.0)
        assert memory.update(given, 0)
        assert [(r.height, r.bearing, r.cut) for r in memory.parts] == expected
        assert memory.flower == "pink" and memory.label_now == memory.label_read == "yellow"
    memory = S.SceneMemory(stable=1, p_min=0.0)
    memory.update(S.Scene(truth.parts, "pink", "unreadable"), 0)
    assert memory.label_now == "unreadable" and memory.label_read is None


def test_update_says_when_a_fact_changed_and_only_then():
    memory = S.SceneMemory(stable=2)

    def scene(cut=False, turned=0.0, label="none"):
        return _scene(_part(0.08, 1.0 + turned, cut=cut), _part(0.12, -2.0 + turned), label=label)

    returns = [
        memory.update(scene(), 0),  # candidates and a first look at the flower: nothing known yet
        memory.update(scene(), 1),  # two parts, a red flower
        memory.update(scene(), 2),  # nothing new (not settled: the inventory grew one counted update ago)
        memory.update(scene(), 3),  # settled
        memory.update(scene(turned=0.5), 4),  # the plant turned: a bearing is not a fact
        memory.update(scene(cut=True), 5),  # one sighting of a cut
        memory.update(scene(cut=True), 6),  # the lowest is cut
        memory.update(scene(cut=True), 7),
        memory.update(scene(label="unreadable"), 8),
        memory.update(scene(label="unreadable"), 9),  # a card
    ]
    assert returns == [False, True, False, True, False, False, True, False, False, True]


def test_the_snapshot_is_plain_data():
    memory = S.SceneMemory(stable=1, p_min=0.0)
    memory.update(_scene(_part(np.float64(0.08), np.float64(1.0), cut=True), _part(0.12), label="cyan"), 4)
    snap = json.loads(json.dumps(memory.snapshot()))
    assert snap == memory.snapshot()
    assert snap["parts"][0] == dict(
        id=0, position="lowest", height=0.08, cut=True, bearing=1.0, sightings=1, seen_step=4, attributes={}
    )
    assert snap["text"] == "2 shoots, red flower, label cyan; cut: the lowest" and snap["settled"] is False


# ------------------------------------------------------------------------------------------------ simulator tests
def _env(task):
    pytest.importorskip("mujoco")
    from sim.pickup.env import PANDA_XML, PickupEnv

    if not PANDA_XML.exists():
        pytest.skip("Franka model not fetched (sim/pickup/scripts/fetch_assets.sh)")
    try:
        return PickupEnv(task)
    except Exception as error:  # no GL context on this machine
        pytest.skip(f"cannot render: {error}")


@pytest.mark.parametrize("task", ["reason_mix", "prune"])
def test_the_truth_lists_every_shoot_lowest_first_and_uncut_at_the_start(task):
    env = _env(task)
    from sim.pickup.env import SHOOT_GAP, SHOOT_HEIGHTS

    try:
        front = total = 0
        for seed in range(1_000_000, 1_000_010):
            env.reset(seed=seed)
            truth = S.scene_truth(env)
            heights = [p.height for p in truth.parts]
            assert len(truth.parts) == len(env.active) and np.all(np.diff(heights) >= SHOOT_GAP - 1e-9)
            assert all(SHOOT_HEIGHTS[0] - 1e-9 <= h <= SHOOT_HEIGHTS[1] + 1e-9 for h in heights)
            assert not any(p.cut for p in truth.parts) and truth.flower == env.flower_colour
            for p in truth.parts:
                assert len(p.views) == 2 and p.visible == any(p.views) and -np.pi <= p.bearing <= np.pi
            if env.label_colour is None:
                assert truth.label == "none"
            else:
                assert truth.label in (env.label_colour, "unreadable")
            assert S.Scene.from_json(truth.to_json()).to_dict() == truth.to_dict()
            front += sum(p.views[0] for p in truth.parts)
            total += len(truth.parts)
        assert front >= 0.7 * total, f"the front camera sees {front} of {total} shoots at the start"
    finally:
        env.close()


def test_the_bearing_says_what_the_notes_aim_says():
    """|bearing| <= PRUNE_SECTOR exactly when the note says the shoot it aims at faces the robot, and otherwise the
    bearing's sign is the button the note names (notes.NoteWriter.text)."""
    env = _env("reason_mix")
    from sim.pickup.expert import PRUNE_SECTOR
    from sim.pickup.notes import NoteWriter

    try:
        checked = 0
        for seed in range(1_000_000, 1_000_040):
            env.reset(seed=seed)
            writer = NoteWriter(env)
            writer.update()
            aim = R.note_fields(writer.note)["aim"]
            if not aim.startswith("the ") or aim.startswith("the label"):
                continue  # the card has to be shown first
            position = aim.split(",")[0].removeprefix("the ")
            part = S.scene_truth(env).parts[R.position_index(position, len(env.active))]
            facing = aim.endswith("faces me")
            assert (abs(part.bearing) <= PRUNE_SECTOR) == facing, (seed, aim, part.bearing)
            if not facing:
                button = R.BUTTON_NAMES[1 if part.bearing >= 0 else -1]
                assert aim.endswith(f"press {button}"), (seed, aim, part.bearing)
            checked += 1
        assert checked >= 20
    finally:
        env.close()


def test_after_pruning_every_part_is_cut_and_still_listed():
    env = _env("prune")
    from sim.pickup.expert import ScriptedExpert

    try:
        wins = stubs_seen = 0
        for seed in SEEDS:
            env.reset(seed=seed)
            before = S.scene_truth(env)
            expert = ScriptedExpert(env, seed=0)
            expert.reset()
            done, info = False, {}
            while not done:
                _, done, info = env.step(expert.act())
            if not info["success"]:
                continue
            wins += 1
            after = S.scene_truth(env)
            assert len(after.parts) == len(env.active) and all(p.cut for p in after.parts)
            assert [p.height for p in after.parts] == [p.height for p in before.parts]
            stubs_seen += sum(p.views[0] for p in after.parts)
        assert wins >= 1, "the expert pruned none of the plants"
        assert stubs_seen >= 1, "the front camera saw none of the stubs"
    finally:
        env.close()


# A part's pixels in a 224-pixel front image: a whole shoot (about 2 by 20 to 25 pixels, and its leaves) covers 50 or
# more, a stub 6 to 25. Two of the ray test's points in view is about a quarter of the part or more.
SHOOT_PIXELS, STUB_PIXELS = 15, 4


def _drawn_geoms(env, name):
    """The geoms the cameras draw of a part: its stub, and while attached its outer part's body."""
    m = env.model
    drawn = [env.stub_geoms[name][1]]
    if name not in env.cut:
        drawn += [g for g in range(m.ngeom) if m.geom_bodyid[g] == env.obj_body[name] and m.geom_group[g] <= 2]
    return drawn


def test_the_front_view_of_each_part_agrees_with_a_segmentation_render():
    env = _env("prune")
    import mujoco

    from sim.pickup.expert import ScriptedExpert

    seg = mujoco.Renderer(env.model, env.image_size, env.image_size)
    seg.enable_segmentation_rendering()
    geom_type = mujoco.mjtObj.mjOBJ_GEOM.value
    confusion = Counter()  # (the rays say seen, the pixels say seen)
    try:
        for seed in SEEDS[:2]:
            env.reset(seed=seed)
            expert = ScriptedExpert(env, seed=0)
            expert.reset()
            done = False
            while not done:
                if env.steps % 8 == 0:
                    truth = S.scene_truth(env)
                    seg.update_scene(env.data, camera="front")
                    ids = seg.render()
                    geoms = np.where(ids[..., 1] == geom_type, ids[..., 0], -1)
                    for name, part in zip(env.active, truth.parts, strict=True):
                        pixels = int(np.isin(geoms, _drawn_geoms(env, name)).sum())
                        confusion[(part.views[0], pixels >= (STUB_PIXELS if part.cut else SHOOT_PIXELS))] += 1
                _, done, _ = env.step(expert.act())
    finally:
        seg.close()
        env.close()
    total = sum(confusion.values())
    agree = confusion[(True, True)] + confusion[(False, False)]
    print(f"rays against pixels, (rays, pixels): {dict(confusion)}")
    assert total >= 60, f"only {total} part-frames"
    assert agree >= 0.85 * total, f"rays and segmentation agree on {agree} of {total}: {dict(confusion)}"


def test_the_card_is_a_colour_only_while_it_faces_the_front_camera():
    env = _env("reason_hidden")
    from sim.pickup.expert import ScriptedExpert

    try:
        read = facing = 0
        for seed in SEEDS:
            env.reset(seed=seed)
            expert = ScriptedExpert(env, seed=0)
            expert.reset()
            done = False
            while not done and env.steps < 400:
                if env.steps % 4 == 0:
                    label = S.scene_truth(env).label
                    assert label in (env.label_colour, "unreadable"), label
                    assert label == "unreadable" or env.label_visible()
                    read += label != "unreadable"
                    facing += env.label_visible()
                _, done, _ = env.step(expert.act())
        assert read > 0, "the card was never read"
        assert read >= 0.8 * facing, f"read in {read} of the {facing} frames in which it faced the camera"
    finally:
        env.close()


def test_the_plant_crop_is_a_larger_view_centred_on_the_stem():
    env = _env("prune")
    import mujoco

    crop = S.PlantCrop(env)
    seg = mujoco.Renderer(env.model, crop.size * crop.zoom, crop.size * crop.zoom)
    try:
        env.reset(seed=SEEDS[0])
        img = crop.render()
        assert img.shape == (224, 224, 3) and img.dtype == np.uint8
        assert not np.array_equal(img, env.render("front"))
        # the main stem runs down the middle of the window, as a segmentation render of the same view shows
        seg.enable_segmentation_rendering()
        seg.update_scene(env.data, camera="front")
        ids = seg.render()
        top, left = crop.window()
        window = ids[top : top + crop.size, left : left + crop.size]
        stem = (window[..., 0] == env.model.geom("stem_vis").id) & (window[..., 1] == mujoco.mjtObj.mjOBJ_GEOM.value)
        rows, cols = np.nonzero(stem)
        band = np.abs(rows - crop.size / 2) < 40
        assert band.sum() > 50, "the main stem is not near the middle of the crop"
        assert abs(np.median(cols[band]) - crop.size / 2) < 12, np.median(cols[band])
    finally:
        seg.close()
        crop.close()
        env.close()


def test_the_collector_writes_the_perception_dataset(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    from PIL import Image

    _env("reason_mix").close()  # skipped where the simulator or a GL context is missing
    out = tmp_path / "scene"
    command = [sys.executable, "-m", "sim.pickup.collect_scene", "--task", "reason_mix", "--episodes", "2"]
    command += ["--workers", "1", "--seed", "1000000", "--crop", "--out", str(out)]
    run = subprocess.run(command, cwd=REPOSITORY, capture_output=True, text=True, timeout=1800)
    assert run.returncode == 0, run.stderr[-4000:]
    assert run.stdout.strip().splitlines()[-1].startswith("wrote 2 episodes")

    meta = json.loads((out / "meta.json").read_text())
    expected = dict(task="reason_mix", episodes=2, stride=4, cameras=["front", "wrist", "plant"], image_size=224)
    expected.update(crop_zoom=2, driver="expert", random_start=True, scenario=None, seed=1_000_000)
    assert {k: meta[k] for k in expected} == expected
    episodes = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
    assert [e["seed"] for e in episodes] == [1_000_000, 1_000_001]
    assert meta["frames"] == sum(e["frames"] for e in episodes)
    for e in episodes:
        assert e["file"] == f"episodes/ep_{e['seed']:07d}.parquet" and e["task"] == "reason_mix"
        table = pq.read_table(out / e["file"])
        assert table.column_names == ["step", "front", "wrist", "plant", "scene"]
        assert table.num_rows == e["frames"] > 0 and str(table.schema.field("step").type) == "int32"
        steps = table.column("step").to_pylist()
        tail = meta["tail_steps"] if e["success"] else 0
        body = steps[: len(steps) - tail]
        assert len({s % meta["stride"] for s in body}) == 1 and all(s < e["steps"] for s in body)
        assert steps[len(body) :] == list(range(e["steps"], e["steps"] + tail))  # after a success, every frame
        for cell in table.column("scene").to_pylist():
            scene = S.Scene.from_json(cell)
            assert len(scene.parts) == e["shoots"] and all(len(p.views) == 2 for p in scene.parts)
        for camera in ("front", "wrist", "plant"):
            img = np.asarray(Image.open(io.BytesIO(table.column(camera)[0].as_py())))
            assert img.shape == (224, 224, 3), (camera, img.shape)
