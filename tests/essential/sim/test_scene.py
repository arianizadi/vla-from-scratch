"""Scene records, the simulator's truth for them, the plant crop, the scene memory and the perception dataset
(sim/pickup/scene.py, sim/pickup/collect_scene.py, eval.py --record_scene; architecture v2).

The record and memory tests need nothing but numpy: hand-made scenes stand in for what the eyes would report. The
others need MuJoCo, a GL context for the cameras and the Franka model (sim/pickup/scripts/fetch_assets.sh), the
dataset's also pyarrow (and torch, for the eyes' reader), and are skipped where any of them is missing.
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

from sim.pickup import program as P  # noqa: E402
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


def test_attributes_with_their_unreadable_class_and_probabilities_round_trip():
    truth = S.Part(
        0.1, 0.0, False, True, views=(True, False), attributes={"tie": "red", "sick": "no", "bud": "unreadable"}
    )
    assert (
        truth.to_dict()["attr"] == {"tie": "red", "sick": "no", "bud": S.UNREADABLE} and "p_attr" not in truth.to_dict()
    )
    percept = S.Part(0.1, 0.0, False, True, p=0.9, attributes={"bud": "yes"}, p_attributes={"bud": 0.812345})
    assert percept.to_dict()["p_attr"] == {"bud": 0.8123}
    for part in (truth, S.Part.from_dict(percept.to_dict())):
        assert S.Part.from_dict(part.to_dict()) == part
    # the truth's values are the program's, and "unreadable" is none of them
    for attr in S.TRUTH_ATTRIBUTES:
        assert S.NO_ATTRIBUTES[attr] in P.DOMAINS[attr] and S.UNREADABLE not in P.DOMAINS[attr]


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


def test_each_part_keeps_the_sighting_its_state_rests_on_and_the_one_that_marked_it_cut():
    memory = S.SceneMemory(stable=2, p_min=0.7)
    _feed(memory, _scene(_part(0.10, p=np.float64(0.912345))), 2)  # added at its second sighting, step 1
    (part,) = memory.parts
    assert part.evidence == dict(step=1, p=0.9123, p_cut=None, cut=False) and part.cut_evidence is None
    memory.update(_scene(_part(0.10, cut=True, p=0.8, p_cut=0.75)), 2)
    assert not part.cut and part.evidence == dict(step=2, p=0.8, p_cut=0.75, cut=True)
    memory.update(_scene(_part(0.10, cut=True, p=0.85, p_cut=0.9)), 3)  # the second in a row: it counts
    assert part.cut and part.cut_evidence == dict(step=3, p=0.85, p_cut=0.9, cut=True)
    memory.update(_scene(_part(0.10, cut=True, p=0.99, p_cut=0.95)), 4)
    memory.update(_scene(_part(0.10, vis=False)), 5)  # not counted: the evidence stays what it was
    assert part.evidence == dict(step=4, p=0.99, p_cut=0.95, cut=True)
    assert part.cut_evidence == dict(step=3, p=0.85, p_cut=0.9, cut=True)  # never overwritten
    snap = json.loads(json.dumps(memory.snapshot()))
    assert snap == memory.snapshot()
    assert snap["evidence"] == [dict(id=0, evidence=part.evidence, cut_evidence=part.cut_evidence)]
    assert "evidence" not in snap["parts"][0]  # each part's own entry is as it was


# ---------------------------------------------------------------------------------- memory: unseen parts, revoked cuts
def _with(scene, *parts):
    """`scene` with more parts reported (the eyes' other slots)."""
    return S.Scene([*scene.parts, *parts], scene.flower, scene.label, scene.p_flower, scene.p_label)


def test_without_unseen_a_part_the_eyes_are_sure_of_but_judge_no_camera_shows_changes_nothing():
    plain, told = S.SceneMemory(stable=2), S.SceneMemory(stable=2)
    ghost = _part(0.15, 0.7, p=1.0, vis=False)  # sure it is there, and that no camera shows it
    two, cut = _scene(_part(0.08), _part(0.115)), _scene(_part(0.08), _part(0.115, cut=True))
    for step, scene in enumerate([_scene(), two, two, two, cut, cut, _scene(), _scene(), two, two]):
        assert told.update(_with(scene, ghost), step) == plain.update(scene, step)
        assert told.snapshot() == plain.snapshot()
    assert len(told.parts) == 2 and told.parts[1].cut and told.settled


def test_without_unseen_a_sighting_2cm_off_a_remembered_part_still_founds_a_phantom():
    memory = S.SceneMemory(stable=2)
    _feed(memory, _scene(_part(0.10)), 2)
    _feed(memory, _scene(_part(0.12)), 2, step=2)  # the same shoot read 2 cm high: beyond `match`, so another part
    assert [r.height for r in memory.parts] == pytest.approx([0.10, 0.12])


def test_with_unseen_a_part_no_camera_shows_is_added_and_its_first_seen_sighting_replaces_its_height():
    memory = S.SceneMemory(stable=3, unseen=True)
    low = _part(0.08, 1.0)
    for step, (h, b) in enumerate([(0.128, 0.3), (0.13, 0.4), (0.132, 0.5)]):  # the eyes are sure, the cameras not
        memory.update(_scene(low, _part(h, b, p=1.0, vis=False)), step)
    lowest, part = memory.parts
    assert part.height == pytest.approx(0.13) and part.bearing == 0.5  # the mean height, the last bearing
    assert (part.sightings, part.seen_step, part.evidence, part.cut) == (0, -1, None, False)
    assert (lowest.sightings, lowest.seen_step) == (3, 2) and not memory.settled  # it grew at the third look
    memory.update(_scene(low, _part(0.121, -0.2)), 3)  # a camera shows it now, 9 mm lower
    assert part.height == pytest.approx(0.121) and part.bearing == -0.2
    assert (part.sightings, part.seen_step) == (1, 3) and part.evidence == dict(step=3, p=0.95, p_cut=None, cut=False)
    memory.update(_scene(low, _part(0.123, -0.2)), 4)
    assert part.height == pytest.approx(0.122) and not memory.settled  # the mean of the seen sightings alone
    assert memory.update(_scene(low, _part(0.13, 2.0, p=1.0, vis=False)), 5)  # three counted updates without growth
    assert memory.settled and part.height == pytest.approx(0.122) and part.bearing == -0.2 and part.sightings == 2
    assert memory.text() == "2 shoots, red flower"


def test_with_unseen_updates_that_count_only_unseen_parts_found_and_settle_the_inventory():
    for unseen in (False, True):
        memory = S.SceneMemory(stable=2, unseen=unseen)
        _feed(memory, _scene(_part(0.09, p=1.0, vis=False), _part(0.13, p=1.0, vis=False)), 4)
        assert (len(memory.parts), memory.settled) == ((2, True) if unseen else (0, False)), unseen


def test_with_unseen_an_unseen_sighting_neither_feeds_nor_breaks_a_cut_row_and_leaves_its_part_as_it_is():
    memory = S.SceneMemory(stable=3, unseen=True)
    _feed(memory, _scene(_part(0.10, 0.2)), 3)
    (part,) = memory.parts
    _feed(memory, _scene(_part(0.10, 0.2, cut=True)), 2, step=3)  # two seen sightings in a row that say cut
    _feed(memory, _scene(_part(0.105, 1.5, cut=True, p=1.0, vis=False)), 3, step=5)  # unseen, saying cut: no more
    _feed(memory, _scene(_part(0.095, -1.5, p=1.0, vis=False)), 3, step=8)  # unseen, saying not cut: no fewer
    assert not part.cut
    assert part.height == pytest.approx(0.10) and (part.bearing, part.sightings, part.seen_step) == (0.2, 5, 4)
    assert part.evidence == dict(step=4, p=0.95, p_cut=None, cut=True)
    memory.update(_scene(_part(0.10, 0.2, cut=True)), 11)  # the third seen sighting in a row that says cut
    assert part.cut and part.cut_evidence == dict(step=11, p=0.95, p_cut=None, cut=True)


def test_with_unseen_a_sighting_nearer_than_the_separation_to_a_remembered_part_founds_nothing_seen_or_unseen():
    for visible in (True, False):
        memory = S.SceneMemory(stable=2, unseen=True)
        _feed(memory, _scene(_part(0.075)), 2)
        _feed(memory, _scene(_part(0.075), _part(0.10, p=1.0, vis=visible)), 6, step=2)  # 2.5 cm above: ambiguous
        _feed(memory, _scene(_part(0.095, p=1.0, vis=visible)), 6, step=8)  # 2 cm above, alone: once a phantom
        assert [r.id for r in memory.parts] == [0], visible
        _feed(memory, _scene(_part(0.075), _part(0.115, p=1.0, vis=visible)), 2, step=14)  # 4 cm above: a shoot
        # 3.5 cm above that one, as near as two shoots get (0.15 - 0.115 < 0.035 in floats): a shoot too
        _feed(memory, _scene(_part(0.075), _part(0.115), _part(0.15, p=1.0, vis=visible)), 2, step=16)
        assert [r.height for r in memory.parts] == pytest.approx([0.075, 0.115, 0.15]), visible


def test_with_unseen_a_shoot_whose_height_is_read_a_few_mm_towards_its_neighbour_is_still_a_shoot():
    # Shoots are 3.5 cm apart at the nearest, and the separation leaves room for the error of a height: a sighting
    # 3 cm from a remembered part is another shoot (S.SEPARATE * match is 2.6 cm), one 2.5 cm from it is not
    assert S.SEPARATE * S.SceneMemory().match == pytest.approx(0.02625)
    for visible in (True, False):
        memory = S.SceneMemory(stable=2, unseen=True)
        _feed(memory, _scene(_part(0.115)), 2)
        _feed(memory, _scene(_part(0.115), _part(0.145, p=1.0, vis=visible)), 2, step=2)
        assert [r.height for r in memory.parts] == pytest.approx([0.115, 0.145]), visible


def test_with_unseen_a_part_no_camera_has_shown_takes_its_first_seen_sighting_from_further_than_match():
    memory = S.SceneMemory(stable=2, unseen=True)
    low = _part(0.08)
    _feed(memory, _scene(low, _part(0.15, 0.5, p=1.0, vis=False)), 2)  # founded where the eyes judged it to be
    _, part = memory.parts
    memory.update(_scene(low, _part(0.128, -0.2)), 2)  # shown 2.2 cm lower: beyond match, within the separation
    assert len(memory.parts) == 2 and part.height == pytest.approx(0.128) and part.bearing == -0.2
    assert (part.sightings, part.seen_step) == (1, 2)
    memory.update(_scene(low, _part(0.15, 0.5)), 3)  # once seen it is where it was seen: 2.2 cm from it is ambiguous
    assert len(memory.parts) == 2 and part.height == pytest.approx(0.128) and part.sightings == 1
    # a part that has been seen is never moved that far: the same sighting founds nothing and moves nothing
    seen = S.SceneMemory(stable=2, unseen=True)
    _feed(seen, _scene(low, _part(0.15, 0.5)), 2)
    _feed(seen, _scene(low, _part(0.128, -0.2)), 4, step=2)
    assert [r.height for r in seen.parts] == pytest.approx([0.08, 0.15])


def test_with_revoke_a_cut_is_unmarked_by_revoke_confident_sightings_in_a_row_of_the_part_not_cut():
    memory = S.SceneMemory(stable=3, revoke=3)
    _feed(memory, _scene(_part(0.10)), 3)
    (part,) = memory.parts
    _feed(memory, _scene(_part(0.10, cut=True, p_cut=0.9)), 3, step=3)  # the jaws closed beside it: it seems cut
    assert part.cut and part.cut_evidence == dict(step=5, p=0.95, p_cut=0.9, cut=True)
    assert _feed(memory, _scene(_part(0.10, p_cut=0.1)), 3, step=6) == [False, False, True]  # unmarked: a fact
    assert not part.cut and part.cut_evidence is None and part.revocations == 1
    assert part.evidence == dict(step=8, p=0.95, p_cut=0.1, cut=False)
    snap = json.loads(json.dumps(memory.snapshot()))
    assert snap["evidence"] == [dict(id=0, evidence=part.evidence, cut_evidence=None, revocations=1)]
    assert snap["parts"][0]["cut"] is False and "revocations" not in snap["parts"][0]
    _feed(memory, _scene(_part(0.10, cut=True, p_cut=0.9)), 2, step=9)  # its cut row started again
    assert not part.cut
    assert memory.update(_scene(_part(0.10, cut=True, p_cut=0.9)), 11)
    assert part.cut and part.cut_evidence["step"] == 11 and part.revocations == 1


def test_with_revoke_an_unsure_sighting_breaks_the_row_and_a_look_that_does_not_see_the_part_leaves_it():
    sure, unsure = _scene(_part(0.10, p_cut=0.1)), _scene(_part(0.10, p_cut=0.5))
    for unseen in (False, True):
        memory = S.SceneMemory(stable=2, revoke=3, unseen=unseen)
        _feed(memory, _scene(_part(0.10)), 2)
        _feed(memory, _scene(_part(0.10, cut=True)), 2, step=2)
        (part,) = memory.parts
        _feed(memory, sure, 2, step=4)
        memory.update(unsure, 6)  # it says not cut, but not surely enough: the row is broken
        _feed(memory, sure, 2, step=7)
        memory.update(_scene(_part(0.10, cut=True, p_cut=0.5)), 9)  # it says cut, unsure of it: broken again
        _feed(memory, sure, 2, step=10)
        memory.update(_scene(_part(0.10, p_cut=0.1, p=1.0, vis=False)), 12)  # no camera shows it (an unseen sighting)
        memory.update(_scene(), 13)  # the arm hides the plant
        assert part.cut and part.revocations == 0, unseen
        memory.update(sure, 14)
        assert not part.cut and part.revocations == 1, unseen


def test_with_revoke_the_truth_of_a_part_not_cut_counts_as_sure():
    def truth(cut):
        return S.Scene([S.Part(0.10, 0.0, cut, True, views=(True, False))], "red", "none")

    memory = S.SceneMemory(stable=1, p_min=0.0, revoke=2)
    memory.update(truth(False), 0)
    memory.update(truth(True), 1)
    (part,) = memory.parts
    assert part.cut
    assert _feed(memory, truth(False), 2, step=2) == [False, True]
    assert not part.cut and part.revocations == 1


def test_without_revoke_a_cut_is_never_unmarked_and_the_snapshot_is_as_it_was():
    memory = S.SceneMemory(stable=2)
    _feed(memory, _scene(_part(0.10)), 2)
    _feed(memory, _scene(_part(0.10, cut=True)), 2, step=2)
    assert not any(_feed(memory, _scene(_part(0.10, p_cut=0.0)), 20, step=4))  # sure it is not cut, twenty times
    (part,) = memory.parts
    assert part.cut and part.revocations == 0 and part.cut_evidence["step"] == 3
    assert set(memory.snapshot()["evidence"][0]) == {"id", "evidence", "cut_evidence"}


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


def test_the_narrower_crop_is_twice_the_plant_crop_around_the_same_place():
    """zoom 4 (collect_scene --crop4) renders larger than the model's offscreen buffer: the buffer is made larger. Its
    window shows the middle of the plant crop's window at twice the size."""
    env = _env("prune")
    crop, crop4 = S.PlantCrop(env), S.PlantCrop(env, zoom=4)
    try:
        assert env.model.vis.global_.offheight >= 896 and crop4.renderer.height == 896
        env.reset(seed=SEEDS[0])
        img, img4 = crop.render().astype(float), crop4.render().astype(float)
        assert img4.shape == (224, 224, 3)
        # the narrower crop's window, in the plant crop's pixels (half its render's), is about the middle of the plant
        # crop; that part of the plant crop scaled up twice looks like the narrower crop
        (top, left), (top4, left4) = crop.window(), crop4.window()
        r0, c0 = top4 // 2 - top, left4 // 2 - left
        assert abs(r0 - 56) <= 2 and abs(c0 - 56) <= 2, (r0, c0)
        middle = img[r0 : r0 + 112, c0 : c0 + 112].repeat(2, 0).repeat(2, 1)
        assert np.abs(middle - img4).mean() < 0.5 * np.abs(img - img4).mean()
    finally:
        crop.close()
        crop4.close()
        env.close()


def test_a_crop_window_is_a_third_attribute_view_that_reads_no_more_than_the_front_camera():
    env = _env("prune")
    env.close()
    from sim.pickup.env import PickupEnv

    env = PickupEnv("prune", scenario="buds")
    crop4 = S.PlantCrop(env, zoom=4)
    try:
        outside = 0
        for seed in range(900_000, 900_010):
            env.reset(seed=seed)
            plain = S.scene_truth(env, attributes=True, views=True)
            windowed = S.scene_truth(env, attributes=True, views=True, windows=(crop4,))
            for a, b in zip(plain.parts, windowed.parts, strict=True):
                assert a.attributes == b.attributes  # what can be read is the two cameras' as before
                for attr, views in b.attribute_views.items():
                    assert len(views) == 3 and views[:2] == a.attribute_views[attr] and (views[0] or not views[2])
                    outside += views[0] and not views[2]
        assert outside > 0, "the narrower crop's window shows every tip the front camera shows"
    finally:
        crop4.close()
        env.close()


def test_the_collector_writes_the_perception_dataset(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    from PIL import Image

    _env("reason_mix").close()  # skipped where the simulator or a GL context is missing
    out = tmp_path / "scene"
    command = [sys.executable, "-m", "sim.pickup.collect_scene", "--task", "reason_mix", "--episodes", "2"]
    command += ["--workers", "1", "--seed", "1000000", "--crop", "--crop4", "--out", str(out)]
    run = subprocess.run(command, cwd=REPOSITORY, capture_output=True, text=True, timeout=1800)
    assert run.returncode == 0, run.stderr[-4000:]
    assert run.stdout.strip().splitlines()[-1].startswith("wrote 2 episodes")

    meta = json.loads((out / "meta.json").read_text())
    expected = dict(task="reason_mix", episodes=2, stride=4, cameras=["front", "wrist", "plant", "plant4"])
    expected.update(
        image_size=224, crop_zoom=2, crop4_zoom=4, driver="expert", random_start=True, scenario=None, seed=1_000_000
    )
    assert {k: meta[k] for k in expected} == expected
    episodes = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
    assert [e["seed"] for e in episodes] == [1_000_000, 1_000_001]
    assert meta["frames"] == sum(e["frames"] for e in episodes)
    for e in episodes:
        assert e["file"] == f"episodes/ep_{e['seed']:07d}.parquet" and e["task"] == "reason_mix"
        table = pq.read_table(out / e["file"])
        assert table.column_names == ["step", "front", "wrist", "plant", "plant4", "scene"]
        assert table.num_rows == e["frames"] > 0 and str(table.schema.field("step").type) == "int32"
        steps = table.column("step").to_pylist()
        tail = meta["tail_steps"] if e["success"] else 0
        body = steps[: len(steps) - tail]
        assert len({s % meta["stride"] for s in body}) == 1 and all(s < e["steps"] for s in body)
        assert steps[len(body) :] == list(range(e["steps"], e["steps"] + tail))  # after a success, every frame
        for cell in table.column("scene").to_pylist():
            scene = S.Scene.from_json(cell)
            assert len(scene.parts) == e["shoots"] and all(len(p.views) == 2 for p in scene.parts)
        for camera in ("front", "wrist", "plant", "plant4"):
            img = np.asarray(Image.open(io.BytesIO(table.column(camera)[0].as_py())))
            assert img.shape == (224, 224, 3), (camera, img.shape)


def test_an_evaluation_records_the_perception_dataset_of_its_rollouts(tmp_path, monkeypatch):
    """eval.py --record_scene writes the collector's layout from closed-loop rollouts (the scripted expert drives, so no
    server is needed): the eyes' reader takes every frame; an episode's frames are its first step, every stride-th from
    its offset and its last; results.json is as without the option but for one key; a second run into it is refused."""
    pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("torch")  # the eyes' reader
    _env("reason_mix").close()  # skipped where the simulator or a GL context is missing
    from sim.pickup import collect_scene as C
    from sim.pickup import eval as E
    from sim.pickup.eyes import EyesConfig
    from sim.pickup.eyes_data import SceneFrames

    data = tmp_path / "data"

    def evaluate(out, *more):
        argv = ["--task", "reason_mix", "--expert", "--episodes", "2", "--videos", "0", *more]
        monkeypatch.setattr(sys, "argv", ["eval.py", "--out", str(out), *argv])
        E.main()
        return json.loads((out / "results.json").read_text())

    def timeless(results):
        return json.dumps({k: v for k, v in results.items() if k != "wall_time_s"}, sort_keys=True)

    plain = evaluate(tmp_path / "plain")
    results = evaluate(tmp_path / "recorded", "--record_scene", str(data))
    assert sorted(p.name for p in (tmp_path / "plain").iterdir()) == ["results.json"] and "record_scene" not in plain
    summary = results.pop("record_scene")
    assert timeless(results) == timeless(plain)  # the rollouts are those of an evaluation that records nothing

    meta = json.loads((data / "meta.json").read_text())
    episodes = [json.loads(line) for line in (data / "episodes.jsonl").read_text().splitlines()]
    assert summary == dict(dir=str(data), episodes=2, frames=meta["frames"])
    expected = dict(task="reason_mix", episodes=2, stride=4, cameras=["front", "wrist", "plant"], image_size=224)
    expected.update(source="eval", driver="expert", policy=plain["server"], slow_level=None, seed=1_000_000)
    assert {k: meta[k] for k in expected} == expected and meta["frames"] == sum(e["frames"] for e in episodes)
    runs = [(r["seed"], r["success"], r["steps"]) for r in plain["per_episode"]]
    assert [(e["seed"], e["success"], e["steps"]) for e in episodes] == runs
    assert any(e["success"] for e in episodes), "the expert succeeded in neither episode"
    frames = []  # (file, step) of every frame, in the listing's order
    for e in episodes:
        table = pq.read_table(data / e["file"])
        assert table.column_names == ["step", "front", "wrist", "plant", "scene"] and table.num_rows == e["frames"]
        steps = table.column("step").to_pylist()
        offset = C.stride_offset(e["seed"], 4)
        assert steps == sorted({0, e["steps"], *(t for t in range(e["steps"]) if t % 4 == offset)}), (e, steps)
        scenes = [S.Scene.from_json(cell) for cell in table.column("scene").to_pylist()]
        assert all(len(s.parts) == e["shoots"] and all(len(p.views) == 2 for p in s.parts) for s in scenes)
        assert not any(p.cut for p in scenes[0].parts)
        if e["success"]:  # the last frame shows the cut that ended the episode
            assert any(p.cut for p in scenes[-1].parts), e
        frames += [(str(data / e["file"]), step) for step in steps]

    reader = SceneFrames(data, EyesConfig(cameras=["front", "wrist", "plant"]), train=False)
    samples = list(reader)
    assert reader.frames == len(samples) == meta["frames"]
    assert [(s["episode"], s["step"]) for s in samples] == frames
    assert all(tuple(s["images"].shape) == (3, 3, 224, 224) for s in samples)

    listing = (data / "episodes.jsonl").read_text()
    with pytest.raises(AssertionError, match="already holds a perception dataset"):
        evaluate(tmp_path / "again", "--record_scene", str(data))
    assert (data / "episodes.jsonl").read_text() == listing


RECORD_ARGUMENTS = [
    (["--task", "pick_cube", "--record_scene", "new"], "it needs a plant task"),
    (["--task", "reason_mix", "--record_scene", "new", "--record_stride", "0"], "--record_stride must be 1 or more"),
    (["--task", "reason_mix", "--record_stride", "2"], "--record_stride is for --record_scene"),
    (["--task", "reason_mix", "--record_scene", "stopped"], "already holds a perception dataset"),
]


@pytest.mark.parametrize("argv, message", RECORD_ARGUMENTS)
def test_the_recording_arguments_are_checked_before_any_server_is_reached(argv, message, tmp_path, monkeypatch):
    """`stopped`: what a run that did not finish leaves, an episode's file without the listing, is a dataset too."""
    _env("reason_mix").close()  # skipped without MuJoCo, the Franka model or a GL context
    from sim.pickup import eval as E

    (tmp_path / "stopped" / "episodes").mkdir(parents=True)
    (tmp_path / "stopped" / "episodes" / "ep_1000000.parquet").write_bytes(b"")
    argv = [str(tmp_path / a) if a in ("new", "stopped") else a for a in argv]
    monkeypatch.setattr(sys, "argv", ["eval.py", "--out", str(tmp_path / "out"), *argv])
    with pytest.raises(AssertionError, match=message):
        E.main()


# ================================================================================ architecture v3: attributes in memory
def _attr_part(h, b=0.0, cut=False, vis=True, p=0.95, p_cut=None, p_attributes=None, **attributes):
    return S.Part(h, b, cut, vis, p=p, p_cut=p_cut, attributes=attributes, p_attributes=p_attributes or {})


def test_an_attribute_needs_stable_readings_in_a_row_and_a_look_that_cannot_read_it_neither_counts_nor_breaks_it():
    memory = S.SceneMemory(stable=3)
    looks = ["red", "red", S.UNREADABLE, "red", S.UNREADABLE, "blue", "blue", "unreadable", "blue"]
    known = []
    for step, tie in enumerate(looks):
        memory.update(_scene(_attr_part(0.1, tie=tie)), step)
        known.append(memory.parts[0].attributes.get("tie") if memory.parts else None)
    # red after its third reading; blue only after three of its own in a row, the unreadable looks skipped
    assert known == [None, None, None, "red", "red", "red", "red", "red", "blue"]
    assert [v for _, v, _ in memory.readings[(0, "tie")]] == ["red", "red", "red", "blue", "blue", "blue"]
    # an unsure reading does not count either, nor one of a part not seen
    memory = S.SceneMemory(stable=1)
    memory.update(_scene(_attr_part(0.1, bud="yes", p_attributes={"bud": 0.5})), 0)
    assert memory.parts[0].attributes == {}
    memory.update(_scene(_attr_part(0.1, vis=False, bud="yes")), 1)
    assert memory.parts[0].attributes == {}
    memory.update(_scene(_attr_part(0.1, bud="yes", p_attributes={"bud": 0.9})), 2)
    assert memory.parts[0].attributes == {"bud": "yes"}


def test_attributes_are_frozen_once_a_part_is_cut_and_a_stub_reads_nothing():
    memory = S.SceneMemory(stable=2)
    _feed(memory, _scene(_attr_part(0.1, tie="red")), 2)
    assert memory.parts[0].attributes == {"tie": "red"}
    _feed(memory, _scene(_attr_part(0.1, cut=True, tie="none")), 2, step=2)  # a stub, read as "no tie"
    assert memory.parts[0].cut and memory.parts[0].attributes == {"tie": "red"}
    _feed(memory, _scene(_attr_part(0.1, cut=True, tie="none")), 3, step=4)
    assert memory.parts[0].attributes == {"tie": "red"}
    # unmarked (revoke), the part reads again
    memory = S.SceneMemory(stable=1, revoke=2)
    memory.update(_scene(_attr_part(0.1, tie="red")), 0)
    memory.update(_scene(_attr_part(0.1, cut=True, tie="none")), 1)
    _feed(memory, _scene(_attr_part(0.1, tie="blue")), 2, step=2)
    assert not memory.parts[0].cut and memory.parts[0].attributes == {"tie": "blue"}


def test_the_last_look_and_two_angles_are_the_other_attribute_rules_and_every_reading_keeps_its_bearing():
    last = S.SceneMemory(stable=3, attribute_rule="last")
    _feed(last, _scene(_attr_part(0.1, sick="yes")), 3)
    last.update(_scene(_attr_part(0.1, sick="no")), 3)
    assert last.parts[0].attributes == {"sick": "no"}
    angles = S.SceneMemory(stable=2, attribute_rule="angles")
    for step, bearing in enumerate((0.0, 0.1, 0.2, 0.3)):  # four readings in a row within 17 degrees: one witness
        angles.update(_scene(_attr_part(0.1, b=bearing, sick="yes")), step)
    assert angles.parts[0].attributes == {}
    angles.update(_scene(_attr_part(0.1, b=0.6, sick="yes")), 4)  # 34 degrees from the first
    assert angles.parts[0].attributes == {"sick": "yes"}
    assert [b for _, _, b in angles.readings[(0, "sick")]] == [0.0, 0.1, 0.2, 0.3, 0.6]
    snapshot = angles.snapshot()
    assert snapshot["readings"] == [
        dict(id=0, attribute="sick", readings=[[s, "yes", b] for s, b in enumerate((0.0, 0.1, 0.2, 0.3, 0.6))])
    ]
    json.dumps(snapshot)
    assert "readings" not in S.SceneMemory().snapshot()  # without attributes, the snapshot is as it was
    with pytest.raises(AssertionError):
        S.SceneMemory(attribute_rule="majority")


def test_sticky_takes_the_first_value_from_a_row_and_changes_it_only_from_two_angles():
    sticky = S.SceneMemory(stable=2, attribute_rule="sticky")
    for step in range(2):  # the first value: a row, from one angle
        sticky.update(_scene(_attr_part(0.1, b=0.0, tie="white")), step)
    assert sticky.parts[0].attributes == {"tie": "white"}
    for step in range(2, 8):  # the hand in front of the tie: "none" from one angle, many times; it stays white
        sticky.update(_scene(_attr_part(0.1, b=0.05, tie="none")), step)
    assert sticky.parts[0].attributes == {"tie": "white"}
    sticky.update(_scene(_attr_part(0.1, b=0.7, tie="none")), 8)  # the same reading from 37 degrees further round
    assert sticky.parts[0].attributes == {"tie": "none"}
    row = S.SceneMemory(stable=2)  # the default rule changes it after any two readings in a row
    for step, tie in enumerate(("white", "white", "none", "none")):
        row.update(_scene(_attr_part(0.1, b=0.0, tie=tie)), step)
    assert row.parts[0].attributes == {"tie": "none"}


def test_with_witness_a_part_first_seen_whole_is_cut_only_where_the_jaws_have_shut():
    memory = S.SceneMemory(stable=1, witness=True)
    memory.update(_scene(_attr_part(0.08), _attr_part(0.115)), 0)
    memory.update(_scene(_attr_part(0.08, cut=True), _attr_part(0.115, cut=True)), 1)
    assert [p.cut for p in memory.parts] == [False, False]  # the hand was never there
    memory.jaws_shut(0.112, 2)  # the jaws shut at the upper shoot's height
    memory.update(_scene(_attr_part(0.08, cut=True), _attr_part(0.115, cut=True)), 3)
    assert [p.cut for p in memory.parts] == [False, True]
    # jaws shut before the part was first seen whole are no witness; a part found cut needs none
    memory = S.SceneMemory(stable=1, witness=True)
    memory.jaws_shut(0.08, 0)
    memory.update(_scene(_attr_part(0.08), _attr_part(0.115, cut=True)), 1)
    memory.update(_scene(_attr_part(0.08, cut=True), _attr_part(0.115, cut=True)), 2)
    assert [p.cut for p in memory.parts] == [False, True]
    without = S.SceneMemory(stable=1)
    without.update(_scene(_attr_part(0.08)), 0)
    without.update(_scene(_attr_part(0.08, cut=True)), 1)
    assert without.parts[0].cut  # the option is off by default
