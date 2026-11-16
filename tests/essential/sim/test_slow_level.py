"""The slow level of architecture v2 in closed-loop evaluation (sim/pickup/slow_level.py, slow_server.py, eval.py
--slow_level v2): percepts, the scene memory and a plan in one loop, with its holds, monitored transitions, bounded
recovery and false completions.

The Supervisor is fed scripted percepts of plants drawn by hand, the SlowLevel a stand-in for the simulator and for the
slow server; those tests need nothing but numpy (and Pillow for the images). The server's tests need torch. The rest
need MuJoCo, a GL context for the cameras and the Franka model (sim/pickup/scripts/fetch_assets.sh), and are skipped
where any of them is missing: the scripted expert drives whole episodes while the slow level watches, and eval.py's
new arguments are checked.
"""

import ast
import io
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:  # sim/ is not an installed package
    sys.path.insert(0, str(REPOSITORY))

from sim.pickup import reasoning as R  # noqa: E402
from sim.pickup import scene as S  # noqa: E402
from sim.pickup import slow_level as L  # noqa: E402

HEIGHTS = (0.08, 0.115, 0.15)  # metres above the stem's base, lowest first (the plants keep 3.5 cm between shoots)
HOLD = ("need_observation", None)


# ------------------------------------------------------------------------------------------- scripted percepts
@dataclass
class Plant:
    """A plant as the eyes would report it, which a test changes between looks: the shoots (by index, lowest first),
    those cut and those no camera shows, the flower (and whether the eyes are sure of it), the card (None: no card)
    and whether it faces the front camera."""

    heights: tuple = HEIGHTS
    bearings: list | None = None
    cut: set = field(default_factory=set)
    hidden: set = field(default_factory=set)
    flower: str = "red"
    sure_of_flower: bool = True
    card: str | None = None
    facing: bool = False

    def percept(self) -> S.Scene:
        bearings = self.bearings or [0.0] * len(self.heights)
        parts = [
            S.Part(h, b, i in self.cut, i not in self.hidden, p=0.95, p_cut=0.97 if i in self.cut else 0.02)
            for i, (h, b) in enumerate(zip(self.heights, bearings, strict=True))
        ]
        label = "none" if self.card is None else self.card if self.facing else "unreadable"
        return S.Scene(parts, self.flower, label, p_flower=0.95 if self.sure_of_flower else 0.4, p_label=0.95)

    def index(self, subgoal: str) -> int:
        """The shoot a sub-goal "cut the <position> shoot" names."""
        return R.position_index(R.subgoal_target(subgoal), len(self.heights))


class Run:
    """A Supervisor looking at a Plant every `every` control steps. With `obey`, the fast level does as it is told
    between looks: it cuts the shoot an Execute names at once."""

    def __init__(self, program, plant, every=1, obey=True, **kwargs):
        self.supervisor = L.Supervisor(program, **kwargs)
        self.plant, self.every, self.obey, self.step = plant, every, obey, 0
        self.decisions = []

    def look(self, n=1) -> list[L.Decision]:
        out = []
        for _ in range(n):
            decision = self.supervisor.step(self.plant.percept(), self.step)
            if self.obey and decision.kind == "execute":
                self.plant.cut.add(self.plant.index(decision.subgoal))
            self.step += self.every
            out.append(decision)
        self.decisions += out
        return out


def _pairs(decisions) -> list[tuple]:
    return [(d.kind, d.subgoal) for d in decisions]


def _runs(decisions) -> list[tuple]:
    """(kind, sub-goal, how many looks in a row)."""
    out = []
    for pair in _pairs(decisions):
        if out and out[-1][:2] == pair:
            out[-1] = (*pair, out[-1][2] + 1)
        else:
            out.append((*pair, 1))
    return out


# ------------------------------------------------------------------------------------ the four kinds, whole episodes
# With stable=3 the memory adds the parts at the third look and is settled at the sixth (three looks without growth);
# a cut counts at its third sighting in a row, the card's colour at its third reading.
def test_an_ordinal_task_holds_until_the_plant_is_settled_then_names_the_shoot_and_completes_once_the_cut_is_seen():
    run = Run("Cut(Rank(2,top))", Plant())
    run.look(12)
    assert _runs(run.decisions) == [(*HOLD, 5), ("execute", "cut the middle shoot", 3), ("complete", "stop", 4)]
    sup = run.supervisor
    assert sup.declared_complete == 8 and sup.hold_steps == 5 and sup.inventory_hold_steps == 0
    assert run.decisions[0].detail == "plant"
    # the cut was made after look 6: it was seen at looks 7, 8 and 9, and only the third made it count
    assert [p.cut for p in sup.memory.parts] == [False, True, False]


def test_a_conditional_task_holds_while_the_flower_is_not_known():
    plant = Plant(heights=HEIGHTS[:2], flower="orange", sure_of_flower=False)
    run = Run("If(FlowerIs(red),Cut(Rank(1,bottom)),Cut(Rank(1,top)))", plant)
    run.look(8)
    plant.sure_of_flower = True
    run.look(10)
    assert _runs(run.decisions) == [
        (*HOLD, 10),  # the inventory (5 looks), then the flower: three sure looks from look 9
        ("execute", "cut the highest shoot", 3),
        ("complete", "stop", 5),
    ]
    assert [d.detail for d in run.decisions[:10]] == ["plant"] * 5 + ["flower"] * 5


def test_an_exclusion_task_cuts_its_two_shoots_one_after_the_other_least_turning_first():
    plant = Plant(bearings=[1.0, -0.3, 0.2])  # every shoot but the highest: the middle needs the least turning
    run = Run("Cut(Except(All(),Rank(1,top)))", plant)
    run.look(15)
    assert _runs(run.decisions) == [
        (*HOLD, 5),
        ("execute", "cut the middle shoot", 3),
        ("execute", "cut the lowest shoot", 3),
        ("complete", "stop", 4),
    ]
    assert run.supervisor.declared_complete == 11


def test_a_label_task_shows_the_label_until_its_colour_is_seen_then_names_the_shoot_and_never_completes_before():
    plant = Plant(card="cyan")
    run = Run("Cut(ReadLabel())", plant)
    run.look(9)
    plant.facing = True  # the fast level has turned the card to the camera
    run.look(10)
    assert _runs(run.decisions) == [
        (*HOLD, 5),
        ("need_observation", "show the label", 6),  # facing away at looks 6 to 9, then two readings of the colour
        ("execute", "cut the highest shoot", 3),  # cyan names the highest
        ("complete", "stop", 5),
    ]
    assert run.decisions[5].detail == "label" and run.supervisor.memory.label_read == "cyan"

    # Every shoot cut before the card was read: nothing says which one was meant, so it is never complete
    plant = Plant(card="white", cut={0, 1, 2})
    run = Run("Cut(ReadLabel())", plant)
    decisions = run.look(20)
    assert _runs(decisions) == [(*HOLD, 5), ("need_observation", "show the label", 15)]
    assert run.supervisor.declared_complete is None


def test_a_cut_seen_on_a_shoot_that_was_to_stay_fails_and_says_stop():
    plant = Plant()
    run = Run("Cut(Rank(1,top))", plant, obey=False)
    run.look(6)
    assert run.decisions[-1] == L.Decision("execute", "cut the highest shoot")
    plant.cut.add(0)  # the fast level cut the lowest instead
    decisions = run.look(4)
    assert _pairs(decisions) == [("execute", "cut the highest shoot")] * 2 + [("failed", "stop")] * 2
    assert decisions[-1].detail == "a shoot that was to stay is cut"
    assert run.supervisor.record()["failure"] == "a shoot that was to stay is cut"


@pytest.mark.parametrize("heights", [(0.10,), (0.07, 0.10, 0.13, 0.16)])
def test_an_inventory_the_fast_level_cannot_name_is_held_not_planned(heights):
    run = Run("Cut(Rank(1,bottom))", Plant(heights=heights))
    decisions = run.look(12)
    assert all(d.kind == "need_observation" and d.subgoal is None and d.detail == "plant" for d in decisions)
    sup = run.supervisor
    assert len(sup.memory.parts) == len(heights) and sup.memory.settled
    assert sup.inventory_hold_steps == 7 and sup.hold_steps == 12  # held by the guard once settled, at look 6
    assert sup.plan.targets is None  # the plan was never asked to name a shoot


# -------------------------------------------------------------------------------------------- no progress
def _stuck(**kwargs) -> Run:
    """A plant that is settled and whose sub-goal is "cut the middle shoot", with a fast level that does nothing:
    looks every 8 control steps (one chunk), the first Execute at step 40."""
    run = Run("Cut(Rank(2,top))", Plant(), every=8, obey=False, **kwargs)
    run.look(6)
    assert run.decisions[-1] == L.Decision("execute", "cut the middle shoot") and run.step == 48
    return run


def test_without_recoveries_slowness_is_never_a_failure():
    run = _stuck()
    decisions = run.look(200)  # 1600 control steps with nothing changing
    assert all(d == L.Decision("execute", "cut the middle shoot") for d in decisions)
    assert run.supervisor.recoveries == 0 and not run.supervisor.done


def test_with_two_recoveries_a_stall_asks_for_a_recovery_twice_then_fails():
    run = _stuck(max_recoveries=2, patience=160)
    run.look(80)
    asked = [(8 * i, d) for i, d in enumerate(run.decisions) if d.recover]  # look i is at control step 8 i
    # 160 steps after the first Execute (step 40); the clock starts again when the fast level has the arm back (the
    # look after a recovery), so 168 steps later each time
    assert [step for step, _ in asked] == [200, 368]
    assert all(d == L.Decision("execute", "cut the middle shoot", recover=True, detail="no progress") for _, d in asked)
    sup = run.supervisor
    assert sup.recoveries == 2 and sup.recovery_steps == [200, 368]
    failed = next(i for i, d in enumerate(run.decisions) if d.kind == "failed")
    assert failed * 8 == 536 and run.decisions[failed] == L.Decision("failed", "stop", detail="no progress")
    assert all(d == run.decisions[failed] for d in run.decisions[failed:])  # final, whatever is seen after
    assert sup.record()["failure"] == "no progress" and sup.done


def test_a_turning_plant_is_progress_and_a_shaking_one_is_not():
    run = _stuck(max_recoveries=1, patience=160)
    for _ in range(60):  # the plant turns 2 degrees per chunk: 10 degrees add up within 48 steps
        run.plant.bearings = [b + np.radians(2) for b in (run.plant.bearings or [0.0] * 3)]
        run.look()
    assert not any(d.recover for d in run.decisions)
    run = _stuck(max_recoveries=1, patience=160)
    for k in range(60):  # it shakes by 6 degrees: nothing adds up
        run.plant.bearings = [np.radians(6) * (k % 2)] * 3
        run.look()
    assert sum(d.recover for d in run.decisions) == 1


def test_while_the_label_is_shown_a_turning_plant_is_progress_too():
    for turning, recoveries in ((True, 0), (False, 1)):
        run = Run("Cut(ReadLabel())", Plant(card="cyan"), every=8, max_recoveries=1, patience=160)
        for _ in range(40):  # the card stays out of view; the plant turns 3 degrees per chunk, or stands still
            if turning:
                run.plant.bearings = [b + np.radians(3) for b in (run.plant.bearings or [0.0] * 3)]
            run.look()
        assert run.decisions[-1].subgoal == "show the label" or run.decisions[-1].kind == "failed"
        assert run.supervisor.recoveries == recoveries, turning


def test_holding_and_stop_do_not_count_as_stalls():
    run = Run("Cut(Rank(2,top))", Plant(heights=(0.10,)), every=8, max_recoveries=1, patience=16)
    run.look(50)  # an inventory that cannot be named: held for 400 steps
    assert run.supervisor.recoveries == 0
    run = Run("Cut(Rank(2,top))", Plant(), every=8, max_recoveries=1, patience=32)
    run.look(50)  # the cut is seen 24 steps after it is asked for; then "stop" for 300 steps
    assert run.supervisor.recoveries == 0 and run.decisions[-1].kind == "complete"


# -------------------------------------------------------------------------------------------------- the end
def test_done_only_once_a_complete_has_stood_for_give_up_steps():
    run = Run("Cut(Rank(2,top))", Plant(), give_up=40)
    run.look(9)
    assert run.decisions[-1].kind == "complete" and run.supervisor.declared_complete == 8
    run.look(39)  # steps 9 to 47: it has stood 39 steps
    assert not run.supervisor.done
    run.look()  # step 48
    assert run.supervisor.done


def test_a_verdict_that_a_later_look_changes_is_not_done():
    # Failed: the plant seems to have two shoots, so it has no middle one; then the third comes into view
    plant = Plant(hidden={1})
    run = Run("Cut(Middle())", plant, give_up=40)
    run.look(30)  # failed since step 5
    assert run.decisions[-1] == L.Decision("failed", "stop", detail="no middle shoot on a plant with 2")
    plant.hidden.clear()
    run.look(8)  # steps 30 to 37: the third shoot is added (step 32), the plant settles again, the middle one is named
    assert run.decisions[-1] == L.Decision("execute", "cut the middle shoot") and not run.supervisor.done
    run.look(41)  # complete from step 38, and done 40 steps later
    assert run.supervisor.done
    kinds = [kind for _, kind, _ in run.supervisor.outcomes]
    assert kinds == ["need_observation", "failed", "need_observation", "execute", "complete"]
    assert run.supervisor.outcomes[-1][0] == run.supervisor.declared_complete == 38

    # Complete, then a shoot that was to stay is seen cut: Failed, and the clock starts again
    plant = Plant()
    run = Run("Cut(Rank(2,top))", plant, give_up=40)
    run.look(30)  # complete since step 8
    plant.cut.add(0)
    run.look(3)  # failed from step 32
    assert run.decisions[-1].kind == "failed" and not run.supervisor.done
    run.look(39)  # steps 33 to 71: 39 steps of the failure
    assert not run.supervisor.done
    run.look()
    assert run.supervisor.done


def test_text_that_is_not_a_program_fails_every_look():
    for source in ("Cut(Rank(2,", None):
        run = Run(source, Plant(), give_up=5)
        decisions = run.look(6)
        assert all(d == L.Decision("failed", "stop", detail="no program") for d in decisions)
        record = run.supervisor.record()
        assert record["program"] == source and record["failure"] == "no program" and run.supervisor.done


def test_the_record_is_plain_data_and_holds_the_outcome_changes():
    run = Run("Cut(Rank(2,top))", Plant(), every=8)
    run.look(12)
    record = run.supervisor.record()
    assert json.loads(json.dumps(record)) == record
    assert record["outcomes"] == [[0, "need_observation", "plant"], [40, "execute", "cut the middle shoot"]] + [
        [64, "complete", "stop"]
    ]
    assert record["program"] == "Cut(Rank(2,top))" and record["final"] == "complete" and record["failure"] is None
    assert record["declared_complete"] == 64 and record["looks"] == 12 and record["hold_steps"] == 5
    assert record["memory"]["text"] == "3 shoots, red flower; cut: the middle"


# ------------------------------------------------------------------------------------------------------ the ledger
SEEN_CUT = dict(p=0.95, p_cut=0.97, cut=True)  # what Plant.percept says of a cut part


def _ledger(run: Run) -> list[dict]:
    ledger = run.supervisor.record()["ledger"]
    assert json.loads(json.dumps(ledger)) == ledger
    return ledger


def _entry(sub_goal, part, opened, closed, outcome, evidence=None, recoveries=0) -> dict:
    position = R.subgoal_target(sub_goal)
    expected = "the label read" if position is None else f"the {position} shoot cut, no other shoot cut"
    return dict(
        sub_goal=sub_goal,
        part=part,
        position=position,
        expected=expected,
        opened=opened,
        closed=closed,
        outcome=outcome,
        evidence=evidence,
        recoveries=recoveries,
    )


def test_the_ledger_of_one_cut_holds_one_attempt_confirmed_by_the_sighting_that_showed_the_cut():
    run = Run("Cut(Rank(2,top))", Plant())
    run.look(12)
    # asked for at step 5, cut at once, seen cut at steps 6, 7 and 8: the third sighting is the evidence
    evidence = dict(step=8, **SEEN_CUT)
    assert _ledger(run) == [_entry("cut the middle shoot", 1, 5, 8, "confirmed", evidence)]
    assert run.supervisor.memory.parts[1].cut_evidence == evidence


def test_the_ledger_of_a_label_task_confirms_the_card_read_then_the_cut():
    plant = Plant(card="cyan")
    run = Run("Cut(ReadLabel())", plant)
    run.look(9)
    plant.facing = True
    run.look(10)
    assert _ledger(run) == [
        _entry("show the label", None, 5, 11, "confirmed", dict(step=11, label="cyan")),
        _entry("cut the highest shoot", 2, 11, 14, "confirmed", dict(step=14, **SEEN_CUT)),
    ]


def test_the_ledger_of_two_targets_holds_an_attempt_for_each_in_turn():
    run = Run("Cut(Except(All(),Rank(1,top)))", Plant(bearings=[1.0, -0.3, 0.2]))
    run.look(15)
    assert _ledger(run) == [
        _entry("cut the middle shoot", 1, 5, 8, "confirmed", dict(step=8, **SEEN_CUT)),
        _entry("cut the lowest shoot", 0, 8, 11, "confirmed", dict(step=11, **SEEN_CUT)),
    ]


def test_a_cut_that_is_commanded_but_never_seen_stays_open_and_is_not_confirmed():
    plant = Plant()
    run = Run("Cut(Rank(1,top))", plant, obey=False)
    run.look(6)
    plant.cut.add(2)  # the fast level cut the highest, out of every camera's view
    plant.hidden.add(2)
    run.look(30)
    assert _ledger(run) == [_entry("cut the highest shoot", 2, 5, None, "open")]
    assert run.decisions[-1] == L.Decision("execute", "cut the highest shoot")
    assert run.supervisor.declared_complete is None


def test_a_shoot_that_was_to_stay_seen_cut_violates_the_attempt():
    plant = Plant()
    run = Run("Cut(Rank(1,top))", plant, obey=False)
    run.look(6)
    plant.cut.update({0, 2})  # the highest, as asked, and the lowest too
    run.look(4)
    # seen at steps 6, 7 and 8: both count at step 8, and the cut not asked for makes it a violation
    evidence = dict(step=8, **SEEN_CUT, part=0)
    assert _ledger(run) == [_entry("cut the highest shoot", 2, 5, 8, "violated", evidence)]


def test_a_rebinding_supersedes_the_attempt():
    plant = Plant(hidden={1})  # the middle shoot is out of view at first: "second from the top" of two is the lowest
    run = Run("Cut(Rank(2,top))", plant, obey=False)
    run.look(8)
    plant.hidden.clear()  # it comes into view (step 8), is added (step 10) and the plant settles again (step 13)
    run.look(8)
    assert _ledger(run) == [
        _entry("cut the lowest shoot", 0, 5, 13, "superseded"),
        _entry("cut the middle shoot", 2, 13, None, "open"),  # the shoot found later has the next id
    ]
    assert run.supervisor.plan.rebinds == 1


def test_an_execute_says_what_its_shoot_needs_when_asked_to_and_a_changing_aim_starts_no_new_attempt():
    # the middle shoot needs most of half a turn counter-clockwise: press blue until it is within 25 degrees
    plant = Plant(bearings=[0.1, 2.0, -1.0])
    run = Run("Cut(Rank(2,top))", plant, obey=False, aim=True)
    first = run.look(8)[-1]
    assert (first.kind, first.subgoal, first.aim) == ("execute", "cut the middle shoot", "press blue")
    assert first.instruction == "cut the middle shoot, press blue"
    plant.bearings = [-1.4, 0.5, -2.5]  # being turned: still out of the sector
    assert run.look()[-1].instruction == "cut the middle shoot, press blue"
    plant.bearings = [-1.7, 0.2, -2.8]  # within it, and the plate still turning: the press stands, as in the notes
    assert run.look()[-1].instruction == "cut the middle shoot, press blue"
    plant.bearings = [-1.75, 0.15, -2.85]  # the plate has stopped (3 degrees since the last look)
    assert run.look()[-1].instruction == "cut the middle shoot, in reach"
    assert run.look()[-1].instruction == "cut the middle shoot, in reach"
    plant.bearings = [-2.4, -0.5, 2.9]  # turned too far: back the other way
    assert run.look()[-1].instruction == "cut the middle shoot, press orange"
    plant.bearings = [-2.1, -0.2, -3.1]
    assert run.look()[-1].instruction == "cut the middle shoot, press orange"  # turning back, not there yet
    # one attempt and one outcome through all of it: the sub-goal is the same, only what is said with it changed
    assert _ledger(run) == [_entry("cut the middle shoot", 1, 5, None, "open")]
    assert [o[1:] for o in run.supervisor.outcomes] == [
        ["need_observation", "plant"],
        ["execute", "cut the middle shoot"],
    ]
    # holds, the card and the end say nothing more than they did
    assert run.decisions[0].instruction is None and run.decisions[0].aim == ""
    label = Run("Cut(ReadLabel())", Plant(card="cyan"), aim=True)
    assert label.look(8)[-1].instruction == "show the label"
    done = Run("Cut(Rank(1,top))", Plant(), aim=True)
    assert [d.instruction for d in done.look(12)][-1] == "stop"
    # and without the option nothing is said: the fast level decides by itself, as before
    plain = Run("Cut(Rank(2,top))", Plant(bearings=[0.1, 2.0, -1.0]), obey=False)
    assert all(d.aim == "" and d.instruction == d.subgoal for d in plain.look(8))


def _jaws(supervisor, plant, jaws, first=0) -> list[L.Decision]:
    """One look per entry of `jaws` (the gripper's opening at that look), from control step `first`."""
    return [supervisor.step(plant.percept(), first + k, jaws=j) for k, j in enumerate(jaws)]


def test_jaws_set_to_cut_a_shoot_that_is_out_of_reach_ask_for_a_reset_as_often_as_allowed():
    plant = Plant(bearings=[2.0, 0.1, -1.0])  # the lowest shoot faces away
    sup = L.Supervisor("Cut(Rank(1,bottom))", resets=1)
    decisions = _jaws(sup, plant, [1.0] * 6 + [0.6, 0.6, 0.6])  # open while it settles, then set to cut
    assert [d.kind for d in decisions] == ["need_observation"] * 5 + ["execute"] * 4
    assert [d.recover for d in decisions] == [False] * 8 + [True] and decisions[-1].detail == "reset"
    assert decisions[-1].subgoal == "cut the lowest shoot"  # the sub-goal stands: the fast level is to choose again
    assert (sup.resets, sup.reset_steps) == (1, [8])
    more = _jaws(sup, plant, [0.6] * 6, first=9)  # the same again: no second reset was allowed
    assert not any(d.recover for d in more) and sup.resets == 1
    record = sup.record()
    assert (record["resets"], record["reset_steps"]) == (1, [8])
    assert [e["outcome"] for e in record["ledger"]] == ["open"]  # one attempt through it all

    def asked(jaws, bearings=None, resets=2):
        plant = Plant(bearings=[2.0, 0.1, -1.0])
        sup = L.Supervisor("Cut(Rank(1,bottom))", resets=resets)
        out = []
        for k, j in enumerate(jaws):
            if bearings is not None:
                plant.bearings = [bearings[k], 0.1, -1.0]
            out.append(sup.step(plant.percept(), k, jaws=j).recover)
        return [k for k, r in enumerate(out) if r]

    settle = [1.0] * 6
    assert asked(settle + [0.6] * 9) == [8, 11]  # twice, three looks each, and no third
    assert asked(settle + [0.0] * 9) == []  # shut to press a button: the right choice
    assert asked(settle + [0.6, 0.6, 0.0, 0.6, 0.6, 1.0, 0.6, 0.6]) == []  # never three looks in a row
    assert asked(settle + [None] * 9) == []  # the jaws are not known
    assert asked(settle + [0.6] * 9, resets=0) == []  # the rule is off unless asked for
    turning = [2.0] * 6 + [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4]  # the plate turns: 11 degrees a look
    assert asked(settle + [0.6] * 9, bearings=turning) == []
    stopped = [2.0] * 6 + [2.0, 1.8, 1.6, 1.4, 1.2, 1.2, 1.2, 1.2, 1.2]  # and stops short, 69 degrees away
    assert asked(settle + [0.6] * 9, bearings=stopped) == [12]
    near = [0.7] * 15  # 40 degrees from facing the robot: not out of reach by this rule
    assert asked(settle + [0.6] * 9, bearings=near) == []
    assert "resets" not in L.Supervisor("Cut(Rank(1,bottom))").record()  # as records were before the rule


def test_the_no_progress_rule_gives_up_the_attempt_with_its_recoveries_counted():
    run = _stuck(max_recoveries=2, patience=160)
    run.look(80)
    assert _ledger(run) == [_entry("cut the middle shoot", 1, 40, 536, "gave_up", recoveries=2)]


# ------------------------------------------------------------------------- the memory's options: revoke and unseen
def test_with_revoke_a_false_complete_is_withdrawn_once_the_shoot_is_seen_whole_and_done_waits_for_one_that_stands():
    plant = Plant()
    run = Run("Cut(Rank(1,top))", plant, obey=False, revoke=3, give_up=5)
    sup, done = run.supervisor, []

    def look(n):
        for _ in range(n):
            run.look()
            done.append(sup.done)

    look(6)  # held five looks, then "cut the highest shoot"
    plant.cut.add(2)  # the eyes say it is cut: the jaws closed beside it
    look(3)  # seen cut at steps 6, 7 and 8: Complete at 8
    plant.cut.discard(2)  # the jaws open: it is seen whole
    look(3)  # at steps 9, 10 and 11: the cut is unmarked at 11, and the plan says Execute again
    assert _runs(run.decisions) == [
        (*HOLD, 5),
        ("execute", "cut the highest shoot", 3),
        ("complete", "stop", 3),
        ("execute", "cut the highest shoot", 1),
    ]
    part = sup.memory.parts[2]
    assert sup.withdrawn == 1 and not part.cut and part.cut_evidence is None and part.revocations == 1
    plant.cut.add(2)  # now it is cut
    look(8)  # seen cut at steps 12, 13 and 14: Complete from 14, which stands its 5 steps at 19
    assert [d.kind for d in run.decisions[12:]] == ["execute"] * 2 + ["complete"] * 6
    assert done == [False] * 19 + [True]
    record = sup.record()
    assert record["declared_complete"] == 8 and record["verdicts_withdrawn"] == 1 and record["revocations"] == 1
    # the first attempt stays confirmed by the sighting that misled the memory; the Execute after it opened another
    assert _ledger(run) == [
        _entry("cut the highest shoot", 2, 5, 8, "confirmed", dict(step=8, **SEEN_CUT)),
        _entry("cut the highest shoot", 2, 11, 14, "confirmed", dict(step=14, **SEEN_CUT)),
    ]


def test_with_revoke_a_failure_for_a_shoot_that_was_to_stay_seen_cut_is_withdrawn_the_same_way():
    plant = Plant()
    run = Run("Cut(Rank(1,top))", plant, obey=False, revoke=3)
    run.look(6)
    plant.cut.add(0)  # the lowest, which is to stay, seems cut
    run.look(3)
    plant.cut.discard(0)
    run.look(3)
    assert _runs(run.decisions) == [
        (*HOLD, 5),
        ("execute", "cut the highest shoot", 3),
        ("failed", "stop", 3),
        ("execute", "cut the highest shoot", 1),
    ]
    assert run.decisions[8].detail == L.STAYED and run.supervisor.withdrawn == 1 and not run.supervisor.done
    assert _ledger(run) == [
        _entry("cut the highest shoot", 2, 5, 8, "violated", dict(step=8, **SEEN_CUT, part=0)),
        _entry("cut the highest shoot", 2, 11, None, "open"),
    ]


def test_without_revoke_a_false_complete_and_a_false_failure_stand_as_before():
    for seen_cut, kind in ((2, "complete"), (0, "failed")):
        plant = Plant()
        run = Run("Cut(Rank(1,top))", plant, obey=False, give_up=5)
        run.look(6)
        plant.cut.add(seen_cut)
        run.look(3)
        plant.cut.discard(seen_cut)  # seen whole from step 9: it changes nothing
        run.look(6)
        sup = run.supervisor
        assert [d.kind for d in run.decisions[8:]] == [kind] * 7, kind
        assert sup.withdrawn == 0 and sup.done and sup.memory.parts[seen_cut].cut, kind
        assert (sup.record()["verdicts_withdrawn"], sup.record()["revocations"]) == (0, 0), kind


def test_a_verdict_that_goes_is_withdrawn_and_one_that_changes_into_another_is_not():
    plant = Plant(hidden={1})
    run = Run("Cut(Middle())", plant)
    run.look(30)  # failed since step 5: a plant of two has no middle shoot
    plant.hidden.clear()
    run.look(8)  # the third shoot is added at step 32: no verdict while the plant settles again, then Execute
    assert [kind for _, kind, _ in run.supervisor.outcomes] == [
        "need_observation",
        "failed",
        "need_observation",
        "execute",
    ]
    assert run.supervisor.withdrawn == 1

    plant = Plant()
    run = Run("Cut(Rank(2,top))", plant)
    run.look(30)  # complete since step 8
    plant.cut.add(0)  # then a shoot that was to stay is seen cut: Failed from step 32
    run.look(5)
    assert [kind for _, kind, _ in run.supervisor.outcomes][-2:] == ["complete", "failed"]
    assert run.supervisor.withdrawn == 0


def test_with_unseen_a_shoot_the_eyes_are_sure_of_but_judge_no_camera_shows_ends_the_hold_on_an_inventory_of_one():
    def run(unseen):
        # Two shoots, the upper one shown a little by the wrist camera only: the eyes judge that no camera shows it
        run = Run("Cut(Rank(1,top))", Plant(heights=HEIGHTS[:2], hidden={1}), obey=False, unseen=unseen)
        run.look(20)
        return run

    held = run(unseen=False)  # as before: an inventory of one, held for as long as the image does not change
    assert _runs(held.decisions) == [(*HOLD, 20)] and held.supervisor.inventory_hold_steps == 15
    assert len(held.supervisor.memory.parts) == 1 and held.supervisor.memory.settled
    told = run(unseen=True)
    assert _runs(told.decisions) == [(*HOLD, 5), ("execute", "cut the highest shoot", 15)]
    memory = told.supervisor.memory
    assert memory.settled and [(part.sightings, part.seen_step) for part in memory.parts] == [(20, 19), (0, -1)]
    assert memory.parts[1].height == pytest.approx(HEIGHTS[1])  # the unseen sightings' height, never seen since


# ------------------------------------------------------------------------------------ the slow level, stand-ins
class StubEnv:
    """What the slow level reads of a PickupEnv: a plant task, its shoots (lowest first) and those cut, the flower and
    the card (and whether it faces the front camera), the reasoning task, the instruction and the control step."""

    def __init__(self, task: R.Task | None, flower="red", label=None, count=3):
        self.plant_task = True
        self.reason = None if task is None else (task.kind, "train")
        self.reason_task = task
        self.instruction = "cut the side shoots off the stem, keep the flower" if task is None else task.instruction
        self.active = [f"shoot{i}" for i in range(count if task is None else task.count)]
        self.cut: set[str] = set()
        self.flower_colour, self.label_colour, self.facing = flower, label, False
        self.steps = 0

    def name(self, subgoal: str) -> str:
        return self.active[R.position_index(R.subgoal_target(subgoal), len(self.active))]


def _truth(env: StubEnv, cut=()) -> S.Scene:
    """The truth of a StubEnv in scene.scene_truth's form; `cut`: shoots to report cut that are not."""
    parts = [
        S.Part(HEIGHTS[i], 0.0, name in env.cut or name in cut, True, views=(True, False))
        for i, name in enumerate(env.active)
    ]
    label = "none" if env.label_colour is None else env.label_colour if env.facing else "unreadable"
    return S.Scene(parts, env.flower_colour, label)


def _drive(slow: L.SlowLevel, env: StubEnv, looks: int, obs=None, history=None) -> list[L.Decision]:
    """Looks one control step apart; the fast level cuts the shoot an Execute names at once."""
    out = []
    for _ in range(looks):
        decision = slow.tick(obs, history)
        if decision.kind == "execute":
            env.cut.add(env.name(decision.subgoal))
        env.steps += 1
        out.append(decision)
    return out


class StubServer:
    """A connection to a slow server (send, recv): `replies` maps a request's cmd to its reply, or to a function of
    the request that gives it. The requests are kept."""

    def __init__(self, **replies):
        self.replies, self.requests = replies, []

    def send(self, request):
        self.requests.append(request)

    def recv(self):
        request = self.requests[-1]
        reply = self.replies.get(request["cmd"], {"error": f"no {request['cmd']} here"})
        return reply(request) if callable(reply) else reply


ORDINAL = R.Task("ordinal", "cut the first shoot from the top", "train", 3, (2,), "x", dict(k=1, end="top"))
HIDDEN = R.Task("hidden", "cut the shoot that the label in the pot names", "train", 3, (0,), "x", dict(label="white"))


def test_the_slow_level_needs_no_simulator_at_import():
    code = (
        "import sys, sim.pickup.slow_level; "
        "assert not {'mujoco', 'torch', 'PIL', 'sim.pickup.env'} & set(sys.modules), sorted(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPOSITORY, check=True)


def test_on_the_truth_with_the_task_s_program_a_seen_cut_completes_and_the_memory_agrees_with_the_simulator():
    env = StubEnv(ORDINAL)
    slow = L.SlowLevel(env, truth=_truth)
    assert slow.stable == 1 and slow.offsets == [0] and slow.crop is None
    slow.reset()
    decisions = _drive(slow, env, 6)
    # stable 1: the parts at the first look, settled at the second; the cut seen at the next look
    assert _runs(decisions) == [(*HOLD, 1), ("execute", "cut the highest shoot", 1), ("complete", "stop", 4)]
    record = slow.record({"success": True})
    assert json.loads(json.dumps(record)) == record
    assert record["gold"] == record["program"] == "Cut(Rank(1,top))" and record["program_meaning"] is True
    assert record["scene"] == "truth" and record["program_source"] == "gold" and record["written"] is None
    assert record["declared_complete"] == 2 and record["false_complete"] is False
    assert record["memory_right"] == {name: dict(share=1.0, ever_wrong=False) for name in L.CHECKS}
    assert record["see_ms"] is None and record["write_ms"] is None


def test_a_complete_the_simulator_does_not_confirm_is_a_false_completion():
    env = StubEnv(ORDINAL)
    slow = L.SlowLevel(env, truth=lambda env: _truth(env, cut={"shoot2"}))  # the highest seems cut and is not
    slow.reset()
    for _ in range(4):
        slow.tick(None)
        env.steps += 1
    record = slow.record({"success": False})
    assert record["declared_complete"] == 1 and record["final"] == "complete" and record["false_complete"] is True
    assert record["ended_false_complete"] is True  # and it is the last decision
    # the memory says the highest is cut and the simulator does not, at every look from the first
    assert record["memory_right"]["cut"] == dict(share=0.0, ever_wrong=True)
    assert record["memory_right"]["parts"] == dict(share=1.0, ever_wrong=False)


def test_a_misread_flower_or_card_is_in_the_record_and_one_not_read_yet_is_not_wrong():
    env = StubEnv(HIDDEN, flower="pink", label="white")
    slow = L.SlowLevel(env, truth=lambda env: S.Scene(_truth(env).parts, "red", _truth(env).label))
    slow.reset()
    _drive(slow, env, 3)  # the card faces away: not read, which is not wrong
    env.facing = True
    _drive(slow, env, 3)
    record = slow.record({"success": False})
    assert record["memory_right"]["flower"] == dict(share=0.0, ever_wrong=True)
    assert record["memory_right"]["label"] == dict(share=1.0, ever_wrong=False)
    assert env.cut == {"shoot0"}  # white names the lowest, read once the card faced the camera


def test_the_record_tells_the_memory_s_false_parts_and_cuts_from_its_lag_and_counts_what_was_unmarked_or_withdrawn():
    # The highest seems cut at the first two looks, then is seen whole: unmarked (revoke 2), then cut for real
    env = StubEnv(ORDINAL)
    slow = L.SlowLevel(env, truth=lambda env: _truth(env, cut={"shoot2"} if env.steps < 2 else ()), revoke=2)
    assert (slow.settings()["unseen"], slow.settings()["revoke"]) == (False, 2)
    slow.reset()
    decisions = _drive(slow, env, 8)
    assert _runs(decisions) == [
        (*HOLD, 1),
        ("complete", "stop", 2),
        ("execute", "cut the highest shoot", 1),
        ("complete", "stop", 4),
    ]
    record = slow.record({"success": True})
    assert json.loads(json.dumps(record)) == record
    assert record["memory_false"] == dict(parts=False, cut=True) and record["memory_right"]["cut"]["share"] == 5 / 8
    assert record["revocations"] == 1 and record["verdicts_withdrawn"] == 1
    assert record["memory"]["evidence"][2]["revocations"] == 1
    assert record["declared_complete"] == 1 and record["false_complete"] is False
    # had the episode failed, the withdrawn Complete would be a false completion, but not one the episode ended on
    for _ in range(2):
        env.steps += 1
        slow.tick(None)
    failed = slow.record({"success": False})
    assert failed["false_complete"] is True and failed["ended_false_complete"] is True  # Complete stands again
    env = StubEnv(ORDINAL)
    slow = L.SlowLevel(env, truth=lambda env: _truth(env, cut={"shoot2"} if env.steps < 2 else ()), revoke=2)
    slow.reset()
    for _ in range(5):  # Complete at looks 2 and 3, withdrawn at look 4: the episode stops there, the shoot uncut
        slow.tick(None)
        env.steps += 1
    stopped = slow.record({"success": False})
    assert stopped["final"] == "execute" and stopped["verdicts_withdrawn"] == 1
    assert stopped["false_complete"] is True and stopped["ended_false_complete"] is False
    out = L.summary([dict(success=False, v2=record) for record in (failed, stopped)])
    assert (out["false_complete"], out["ended_false_complete"]) == (2, 1)
    old = {key: value for key, value in stopped.items() if key != "ended_false_complete"}
    assert L.summary([dict(success=False, v2=old)])["ended_false_complete"] == 1  # as records were before

    # The middle shoot out of view for three looks, and with stable 2 a cut seen a look after it is made: the memory is
    # short of the plant and behind it, which is lag, not false
    def middle_late(env):
        scene = _truth(env)
        return scene if env.steps >= 3 else replace(scene, parts=[scene.parts[0], scene.parts[2]])

    env = StubEnv(ORDINAL)
    late = L.SlowLevel(env, truth=middle_late, stable=2)
    late.reset()
    decisions = _drive(late, env, 7)
    assert _runs(decisions) == [
        (*HOLD, 3),  # the two shoots in view are added at step 1
        ("execute", "cut the highest shoot", 1),  # settled at step 3, the middle one not added yet
        (*HOLD, 2),  # added at step 4: the plant settles again while the cut made at step 3 is seen
        ("complete", "stop", 1),
    ]
    record = late.record({"success": True})
    assert record["memory_right"]["parts"]["ever_wrong"] and record["memory_right"]["cut"]["ever_wrong"]
    assert record["memory_false"] == dict(parts=False, cut=False)
    assert record["revocations"] == 0 and "revocations" not in record["memory"]["evidence"][0]

    # A part the plant does not have is false
    env = StubEnv(ORDINAL)
    ghost = S.Part(0.19, 0.0, False, True, views=(True, False))
    phantom = L.SlowLevel(env, truth=lambda env: replace(_truth(env), parts=[*_truth(env).parts, ghost]))
    phantom.reset()
    _drive(phantom, env, 4)
    assert phantom.record({"success": False})["memory_false"] == dict(parts=True, cut=False)


def test_a_written_program_is_compared_with_the_task_s_by_meaning():
    def writer(program):
        return StubServer(
            info=dict(eyes=None, writer="stub", constrained=True),
            write=dict(program=program, complete=True, latency_s=0.25),
        )

    cases = [
        ("Cut(Rank(1,top))", True),
        ("Cut(Except(All(),Except(All(),Rank(1,top))))", True),  # other words, the same shoots on every plant
        ("Cut(Rank(1,bottom))", False),
    ]
    for written, meaning in cases:
        env = StubEnv(ORDINAL)
        server = writer(written)
        slow = L.SlowLevel(env, program="writer", server=server, truth=_truth)
        slow.reset()
        assert server.requests[-1] == {"cmd": "write", "instruction": ORDINAL.instruction}
        _drive(slow, env, 4)
        record = slow.record({"success": meaning})
        assert record["written"] == written and record["program_meaning"] is meaning and record["write_ms"] == 250.0
        assert record["gold"] == "Cut(Rank(1,top))" and record["program_source"] == "writer"

    env = StubEnv(ORDINAL)  # not a program: the slow level fails at every look and stops the fast level
    slow = L.SlowLevel(env, program="writer", server=writer("Cut(Rank(1,"), truth=_truth)
    slow.reset()
    assert all(d == L.Decision("failed", "stop", detail="no program") for d in _drive(slow, env, 3))
    record = slow.record({"success": False})
    assert record["program_meaning"] is False and record["program"] == "Cut(Rank(1," and env.cut == set()

    env = StubEnv(None)  # a task without a program of its own: the writer's is all there is
    slow = L.SlowLevel(env, program="writer", server=writer("Cut(All())"), truth=_truth)
    slow.reset()
    _drive(slow, env, 8)
    assert slow.record({"success": True})["program_meaning"] is None and env.cut == set(env.active)
    with pytest.raises(AssertionError, match="reason_"):
        L.SlowLevel(StubEnv(None), truth=_truth)  # the gold program needs a reasoning task


class Frames:
    """eval.FrameHistory's at(k): the observation k control steps before the newest."""

    def __init__(self, frames):
        self.frames = frames

    def at(self, k):
        return self.frames[max(0, len(self.frames) - 1 + k)]


def _shade(jpeg: bytes) -> float:
    from PIL import Image

    with Image.open(io.BytesIO(jpeg)) as image:
        return float(np.asarray(image.convert("RGB")).mean())


def test_the_eyes_get_the_images_their_keys_name_and_their_percepts_drive_the_memory():
    env = StubEnv(ORDINAL)
    shown = []

    def see(request):
        shown.append({key: _shade(jpeg) for key, jpeg in request["images"].items()})
        parts = [
            S.Part(HEIGHTS[i], 0.0, name in env.cut, True, p=0.9, p_cut=0.9 if name in env.cut else 0.1)
            for i, name in enumerate(env.active)
        ]
        return dict(scene=S.Scene(parts, "red", "none", 0.9, 0.9).to_dict(), latency_s=0.004)

    info = dict(eyes=dict(keys=["front_t0", "wrist_t-20"]), writer=None)
    server = StubServer(info=info, see=see)
    slow = L.SlowLevel(env, scene="eyes", server=server)
    assert slow.stable == 3 and slow.offsets == [-20, 0] and server.requests == [{"cmd": "info"}]
    slow.reset()

    def image(shade):
        return np.full((32, 32, 3), shade, np.uint8)

    obs = dict(front=image(10), wrist=image(60))
    with pytest.raises(ValueError, match="20 control steps back"):
        slow.tick(obs)
    history = Frames([dict(front=image(250), wrist=image(200))] + [dict(front=image(90), wrist=image(90))] * 20)
    decisions = _drive(slow, env, 12, obs, history)
    assert set(shown[0]) == {"front_t0", "wrist_t-20"}
    assert abs(shown[0]["front_t0"] - 10) < 3 and abs(shown[0]["wrist_t-20"] - 200) < 3  # now, and 20 steps back
    assert _runs(decisions) == [(*HOLD, 5), ("execute", "cut the highest shoot", 3), ("complete", "stop", 4)]
    assert slow.record({"success": True})["see_ms"] == 4.0


def test_the_eyes_scene_has_the_memory_s_two_options_on_unless_told_otherwise_and_the_truth_has_them_off():
    info = dict(eyes=dict(keys=["front_t0"]), writer=None)
    eyes = L.SlowLevel(StubEnv(ORDINAL), scene="eyes", server=StubServer(info=info))
    assert (eyes.stable, eyes.give_up, eyes.unseen, eyes.revoke) == (3, 80, True, 3)
    assert L.DEFAULTS["eyes"] == dict(stable=3, give_up=80, unseen=True, revoke=3)
    settings = eyes.settings()
    assert (settings["stable"], settings["give_up"], settings["unseen"], settings["revoke"]) == (3, 80, True, 3)
    eyes.reset()  # and they reach the rules and the memory
    memory = eyes.supervisor.memory
    assert (memory.stable, memory.unseen, memory.revoke, eyes.supervisor.give_up) == (3, True, 3, 80)
    truth = L.SlowLevel(StubEnv(ORDINAL), truth=_truth)
    assert (truth.stable, truth.give_up, truth.unseen, truth.revoke) == (1, 40, False, 0)
    # the eyes' settings before these defaults are still there for whoever names them
    old = L.SlowLevel(StubEnv(ORDINAL), scene="eyes", server=StubServer(info=info), give_up=40, unseen=False, revoke=0)
    assert (old.stable, old.give_up, old.unseen, old.revoke) == (3, 40, False, 0)
    told = L.SlowLevel(StubEnv(ORDINAL), truth=_truth, stable=2, give_up=10, unseen=True, revoke=1)
    assert (told.stable, told.give_up, told.unseen, told.revoke) == (2, 10, True, 1)


def test_the_slow_server_s_refusals_are_errors_here():
    with pytest.raises(AssertionError, match="no eyes"):
        L.SlowLevel(StubEnv(ORDINAL), scene="eyes", server=StubServer(info=dict(eyes=None, writer="w")))
    with pytest.raises(AssertionError, match="no program writer"):
        L.SlowLevel(StubEnv(ORDINAL), program="writer", server=StubServer(info=dict(eyes=None, writer=None)))
    with pytest.raises(AssertionError, match="slow_server"):
        L.SlowLevel(StubEnv(ORDINAL), scene="eyes")
    for key in ("plant_t-4", "side_t0", "front_t2", "front"):
        server = StubServer(info=dict(eyes=dict(keys=[key]), writer=None))
        with pytest.raises(ValueError, match=re.escape(repr(key))):
            L.SlowLevel(StubEnv(ORDINAL), scene="eyes", server=server)
    server = StubServer(info=dict(eyes=dict(keys=["front_t0"]), writer=None), see={"error": "out of memory"})
    slow = L.SlowLevel(StubEnv(ORDINAL), scene="eyes", server=server)
    slow.reset()
    with pytest.raises(RuntimeError, match="out of memory"):
        slow.tick(dict(front=np.zeros((8, 8, 3), np.uint8)))


# ----------------------------------------------------------------------------------------------- recovery
class StubExpert:
    """ScriptedExpert's take_over, preamble and act, with a preamble of `moves` commands (0: none; None: endless)."""

    def __init__(self, moves):
        self.moves, self.taken, self.preamble, self.left = moves, 0, False, 0

    def take_over(self):
        self.taken += 1
        self.left = self.moves
        self.preamble = self.moves is None or self.moves > 0

    def act(self):
        if self.moves is not None:
            self.left -= 1
            self.preamble = self.left > 0
        return np.full(8, 0.5, np.float32)


def test_a_recovery_makes_the_expert_s_preamble_and_ends_with_it():
    env = SimpleNamespace(data=SimpleNamespace(qpos=np.ones(9)))
    expert = StubExpert(3)
    recovery = L.Recovery(env, expert)
    assert expert.taken == 1 and recovery.path is None
    commands = []
    while not recovery.done:
        commands.append(recovery.act())
    assert len(commands) == 3
    with pytest.raises(AssertionError):
        recovery.act()
    endless = L.Recovery(env, StubExpert(None))
    for _ in range(L.PREAMBLE_LIMIT):
        endless.act()
    assert endless.done


def test_without_a_preamble_a_recovery_goes_home_in_a_straight_line_with_the_jaws_open():
    env = SimpleNamespace(data=SimpleNamespace(qpos=np.array([1.0] * 7 + [0.04, 0.04])))
    home = np.linspace(-1, 1, 7)
    recovery = L.Recovery(env, StubExpert(0), home=home)
    commands = []
    while not recovery.done:
        commands.append(recovery.act())
    assert len(commands) == L.HOME_STEPS == 40
    path = np.array(commands)
    assert np.allclose(path[:, :7], [1 + (home - 1) * (k + 1) / 40 for k in range(40)], atol=1e-6)
    assert np.allclose(path[-1, :7], home) and np.all(path[:, 7] == L.JAWS_OPEN)


def test_a_reset_s_recovery_goes_on_to_the_home_pose_after_the_preamble():
    env = SimpleNamespace(data=SimpleNamespace(qpos=np.array([1.0] * 7 + [0.02, 0.02])))
    home = np.linspace(-1, 1, 7)
    recovery = L.Recovery(env, StubExpert(3), home=home, then_home=True)
    commands = []
    while not recovery.done:
        if len(commands) == 3:  # the preamble is made: the line starts from where it left the arm
            env.data.qpos[:7] = 0.0
        commands.append(recovery.act())
    assert len(commands) == 3 + L.HOME_STEPS
    assert all(np.all(c == 0.5) for c in commands[:3])  # the expert's three
    path = np.array(commands[3:])
    assert np.allclose(path[:, :7], [home * (k + 1) / 40 for k in range(40)], atol=1e-6)
    assert np.allclose(path[-1, :7], home) and np.all(path[:, 7] == L.JAWS_OPEN)
    # without a preamble it is the plain line home, as any recovery from such a pose
    plain = L.Recovery(env, StubExpert(0), home=home, then_home=True)
    assert plain.path is not None and len(plain.path) == L.HOME_STEPS


# --------------------------------------------------------------------------------------------------- summary
def test_the_summary_counts_false_completions_failures_meanings_recoveries_and_the_memory():
    def episode(success, env, slow, looks):
        slow.reset()
        _drive(slow, env, looks)
        return dict(success=success, v2=slow.record(dict(success=success)))

    env = StubEnv(ORDINAL)
    right = episode(True, env, L.SlowLevel(env, truth=_truth), 4)
    env = StubEnv(ORDINAL)
    lying = episode(False, env, L.SlowLevel(env, truth=lambda env: _truth(env, cut={"shoot2"})), 4)
    env = StubEnv(ORDINAL)
    written = dict(program="nonsense", complete=False, latency_s=2)
    server = StubServer(info=dict(eyes=None, writer="stub"), write=written)
    garbled = episode(False, env, L.SlowLevel(env, program="writer", server=server, truth=_truth), 4)
    stuck = dict(right["v2"], recoveries=2, failure="no progress", declared_complete=None, false_complete=False)
    rescued = dict(right["v2"], recoveries=1)
    episodes = [right, lying, garbled, dict(success=False, v2=stuck), dict(success=True, v2=rescued), dict(success=0)]
    out = L.summary(episodes)
    assert json.loads(json.dumps(out)) == out
    assert out["episodes"] == 5 and out["successes"] == 2  # the last has no v2 record
    assert out["declared_complete"] == 3 and out["false_complete"] == 1
    assert out["failures"] == {"no program": 1, "no progress": 1}
    assert (out["programs_right_meaning"], out["programs_compared"]) == (4, 5)
    assert out["recoveries"] == dict(asked=3, episodes=2, successes=1)
    assert out["memory_ever_wrong"] == dict(parts=0, cut=1, label=0, flower=0)
    assert out["memory_right"]["cut"] == pytest.approx(0.8) and out["memory_right"]["parts"] == 1.0
    assert out["mean_hold_steps"] == pytest.approx((1 + 1 + 0 + 1 + 1) / 5) and out["write_ms"] == 2000.0
    assert L.summary([])["episodes"] == 0 and L.summary([])["mean_hold_steps"] is None


def test_the_summary_sums_the_ledgers_and_counts_the_episodes_that_aimed_at_one_part_twice():
    env = StubEnv(ORDINAL)
    slow = L.SlowLevel(env, truth=_truth)
    slow.reset()
    _drive(slow, env, 4)
    confirmed = slow.record({"success": True})  # the SlowLevel's record carries the Supervisor's ledger
    assert [entry["outcome"] for entry in confirmed["ledger"]] == ["confirmed"]

    plant = Plant(bearings=[0.5, 0.2, 0.0])  # every shoot but the highest: the middle first, it needs less turning
    turning = Run("Cut(Except(All(),Rank(1,top)))", plant, obey=False)
    turning.look(6)
    plant.bearings = [0.1, 0.4, 0.0]  # the plant turns: now the lowest needs less
    turning.look(2)
    plant.bearings = [0.5, 0.2, 0.0]  # and back: the middle again, in a second attempt
    turning.look(2)
    assert [(e["part"], e["outcome"]) for e in _ledger(turning)] == [(1, "superseded"), (0, "superseded"), (1, "open")]

    stuck = _stuck(max_recoveries=1, patience=160)
    stuck.look(80)
    wrong = Run("Cut(Rank(1,top))", Plant(), obey=False)
    wrong.look(6)
    wrong.plant.cut.add(0)
    wrong.look(4)
    episodes = [dict(success=True, v2=confirmed), dict(success=False, v2=dict(confirmed, ledger=[]))]
    episodes += [dict(success=False, v2=dict(confirmed, ledger=_ledger(run))) for run in (turning, stuck, wrong)]
    out = L.summary(episodes)
    assert out["ledger"] == dict(attempts=6, open=1, confirmed=1, violated=1, superseded=2, gave_up=1)
    assert out["repeated_targets"] == 1
    assert L.summary([])["ledger"] == dict(attempts=0, open=0, confirmed=0, violated=0, superseded=0, gave_up=0)


def test_the_summary_counts_false_memories_revocations_and_withdrawn_verdicts_and_reads_records_without_them():
    env = StubEnv(ORDINAL)
    slow = L.SlowLevel(env, truth=_truth)
    slow.reset()
    _drive(slow, env, 4)
    right = slow.record({"success": True})
    assert right["memory_false"] == dict(parts=False, cut=False)
    assert (right["revocations"], right["verdicts_withdrawn"]) == (0, 0)
    false_cut = dict(right, memory_false=dict(parts=False, cut=True), revocations=2, verdicts_withdrawn=1)
    phantom = dict(right, memory_false=dict(parts=True, cut=True), revocations=1, verdicts_withdrawn=3)
    new = ("memory_false", "revocations", "verdicts_withdrawn")
    old = {key: value for key, value in right.items() if key not in new}
    out = L.summary([dict(success=True, v2=record) for record in (right, false_cut, phantom, old)])
    assert json.loads(json.dumps(out)) == out
    assert out["memory_false"] == dict(parts=1, cut=2)
    assert out["revocations"] == dict(total=3, episodes=2) and out["verdicts_withdrawn"] == 4
    for records in ([], [old]):  # nothing, or records written before these were kept
        out = L.summary([dict(success=True, v2=record) for record in records])
        assert out["memory_false"] == dict(parts=0, cut=0)
        assert out["revocations"] == dict(total=0, episodes=0) and out["verdicts_withdrawn"] == 0


# -------------------------------------------------------------------------------------------- the slow server
class StubEyes:
    def __init__(self):
        from sim.pickup.eyes import EyesConfig

        self.config = EyesConfig(cameras=["front", "plant"], frames=[-20, 0])

    def see(self, images):
        missing = [key for key in self.config.keys if key not in images]
        if missing:
            raise ValueError(f"the eyes need the images {missing}")
        return dict(parts=[], flower="red", label="none", p_flower=0.9, p_label=0.9)


class StubWriter:
    constrained, examples = True, [("i", "p")] * 8

    def write(self, instruction):
        return dict(program="Cut(Rank(1,top))", complete=True, tokens=9, seconds=0.1)


def test_the_slow_server_answers_info_see_and_write_and_refuses_what_it_has_not_loaded():
    pytest.importorskip("torch")
    from sim.pickup.slow_server import SlowServer

    about = dict(eyes_run="runs/eyes", eyes_checkpoint=7, eyes_weights="ema", writer="some/model")
    server = SlowServer(StubEyes(), StubWriter(), about)
    info = server.handle({"cmd": "info"})
    assert info["eyes"]["keys"] == ["front_t-20", "front_t0", "plant_t-20", "plant_t0"]
    assert info["eyes"]["cameras"] == ["front", "plant"] and info["eyes_checkpoint"] == 7
    assert info["writer"] == "some/model" and info["constrained"] is True and info["shots"] == 8
    images = {key: b"" for key in info["eyes"]["keys"]}
    reply = server.handle({"cmd": "see", "images": images})
    assert reply["scene"]["flower"] == "red" and reply["latency_s"] >= 0
    reply = server.handle({"cmd": "write", "instruction": "cut the highest shoot"})
    assert reply["program"] == "Cut(Rank(1,top))" and reply["complete"] is True and "latency_s" in reply
    assert "front_t-20" in server.handle({"cmd": "see", "images": {}})["error"]  # a failure is an answer
    assert "unknown request" in server.handle({"cmd": "plan"})["error"]
    bare = SlowServer()
    assert bare.handle({"cmd": "info"}) == dict(
        eyes=None,
        eyes_run=None,
        eyes_checkpoint=None,
        eyes_weights=None,
        writer=None,
        writer_weights=None,
        constrained=False,
        shots=None,
    )
    assert "--eyes" in bare.handle({"cmd": "see", "images": {}})["error"]
    assert "--writer" in bare.handle({"cmd": "write", "instruction": "x"})["error"]


def test_the_slow_server_binds_announces_serves_and_shuts_down():
    pytest.importorskip("torch")
    from multiprocessing.connection import Client

    from sim.pickup.slow_server import AUTHKEY

    command = [sys.executable, "-m", "sim.pickup.slow_server", "--port", "0", "--device", "cpu"]
    process = subprocess.Popen(command, cwd=REPOSITORY, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        line = process.stdout.readline()
        match = re.fullmatch(r"\[slow_server\] listening on 127\.0\.0\.1:(\d+)\n", line)
        assert match, line
        with Client(("127.0.0.1", int(match.group(1))), authkey=AUTHKEY) as conn:
            conn.send({"cmd": "info"})
            assert conn.recv()["eyes"] is None
            conn.send({"cmd": "see", "images": {}})
            assert "no eyes" in conn.recv()["error"]
            conn.send({"cmd": "shutdown"})
            assert conn.recv() == {"ok": True}
        process.communicate(timeout=30)
        assert process.returncode == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def test_every_server_and_client_uses_the_same_authkey():
    pytest.importorskip("torch")
    from sim.pickup.policy_server import AUTHKEY as policy
    from sim.pickup.slow_server import AUTHKEY as slow

    # eval.py keeps its own copy (it cannot import the servers, which need torch); read it from the source
    client = re.search(r'^AUTHKEY = (b"[^"]*")', (REPOSITORY / "sim/pickup/eval.py").read_text(), re.M).group(1)
    assert policy == slow == ast.literal_eval(client)


# ---------------------------------------------------------------------------------------------- simulator tests
def _env(task):
    pytest.importorskip("mujoco")
    from sim.pickup.env import PANDA_XML, PickupEnv

    if not PANDA_XML.exists():
        pytest.skip("Franka model not fetched (sim/pickup/scripts/fetch_assets.sh)")
    try:
        return PickupEnv(task)
    except Exception as error:  # no GL context on this machine
        pytest.skip(f"cannot render: {error}")


SEEDS = range(1_000_000, 1_000_012)
TAIL = 100  # control steps the expert goes on driving after a success while the slow level watches (it backs out)


@pytest.mark.parametrize("task", ["reason_mix", "reason_mix_heldout"])
def test_on_the_truth_the_slow_level_follows_the_expert_through_whole_episodes(task):
    """The scripted expert drives; the slow level (the truth, the task's program) watches every control step.

    - While the expert's note says "show the label", the slow level says so too or holds, unless it has read the card
      already: the truth gives the card's colour once its face is in view, the note only once the plate stands still.
    - When it says Execute, it names a shoot the task means, and not one that is cut and was seen cut (a cut that no
      camera has shown yet cannot be in the memory).
    - When the episode succeeds, it declares Complete no earlier than the last cut and within stable + 2 looks of the
      last cut coming into view, by the end of the tail, and false_complete is False.

    A failure message says how many shoots the memory held: with stable 1 it settles after two looks, and a shoot that
    neither camera shows at the start (the front camera looks from across the table, past the main stem) is then
    missing from the inventory until it comes into view, which the second check would show as a wrong shoot named.
    """
    env = _env(task)
    from sim.pickup.expert import ScriptedExpert
    from sim.pickup.notes import NoteWriter

    looked = {}

    def truth(env):  # the percept of each look, kept for the checks
        looked["scene"] = S.scene_truth(env)
        return looked["scene"]

    slow = L.SlowLevel(env, scene="truth", program="gold", truth=truth)
    expert, writer = ScriptedExpert(env, seed=0), NoteWriter(env)
    same = looks = successes = 0
    try:
        for seed in SEEDS:
            obs = env.reset(seed=seed)
            expert.reset()
            writer.reset()
            slow.reset()
            names = {R.position_name(i, len(env.active)): name for i, name in enumerate(env.active)}
            cut_at, seen_cut = {}, {}  # shoot -> the step it was cut, the first look that showed it cut
            done, info, after = False, {}, 0
            while not done or (info["success"] and after < TAIL):
                after += done
                writer.update()
                decision = slow.tick(obs)
                for name, part in zip(env.active, looked["scene"].parts, strict=True):
                    if part.cut and part.visible:
                        seen_cut.setdefault(name, env.steps)
                memory = slow.supervisor.memory
                inventory = f"the memory holds {len(memory.parts)} of the {len(env.active)} shoots"
                where = (seed, env.steps, env.reason_kind, decision, writer.note, inventory, memory.text())
                note = R.subgoal(writer.note)
                if note == "show the label":
                    read = memory.label_read
                    assert decision.subgoal in ("show the label", None) or read == env.label_colour, where
                if decision.kind == "execute":
                    named = names.get(R.subgoal_target(decision.subgoal))
                    assert named in env.targets, where
                    assert named not in env.cut or named not in seen_cut, where
                same += decision.subgoal == note
                looks += 1
                before = set(env.cut)
                obs, finished, info = env.step(expert.act())
                for name in set(env.cut) - before:
                    cut_at[name] = env.steps
                done = done or finished
            record = slow.record(info)
            if not info["success"]:
                continue
            successes += 1
            unseen = [name for name in env.targets if name not in seen_cut]
            assert not unseen, (seed, "a cut never came into view", unseen)
            last_seen = max(seen_cut[name] for name in env.targets)
            declared = record["declared_complete"]
            assert declared is not None, (seed, record["outcomes"])
            assert max(cut_at[name] for name in env.targets) <= declared <= last_seen + slow.stable + 2, (seed, record)
            assert record["false_complete"] is False and record["final"] == "complete", (seed, record)
        print(f"{task}: the slow level's sub-goal is the note's at {same} of {looks} looks ({same / looks:.1%})")
        assert successes >= len(SEEDS) // 2, f"the expert succeeded in {successes} of {len(SEEDS)}"
    finally:
        env.close()


def test_the_aim_the_slow_level_says_is_the_aim_of_the_expert_s_first_note():
    """On the simulator's scene, the sub-goal with its aim that the slow level hands the fast level at the start of an
    episode is the one the expert's note gives (the text the fast level was trained on), for every aim there is."""
    env = _env("reason_mix")
    from sim.pickup.expert import PRUNE_SECTOR
    from sim.pickup.notes import NoteWriter

    assert pytest.approx(float(PRUNE_SECTOR)) == R.AIM_SECTOR
    slow, writer, said = L.SlowLevel(env, aim=True), NoteWriter(env), set()
    assert slow.settings()["aim"] is True and "aim" not in L.SlowLevel(env).settings()
    try:
        for seed in range(1_000_000, 1_000_060):
            obs = env.reset(seed=seed)
            if env.reason_kind == "hidden":  # the card first: its sub-goal has no aim
                continue
            writer.reset()
            writer.update()
            slow.reset()
            decision = slow.tick(obs)
            while decision.subgoal is None:
                obs, _, _ = env.step(env.last_action.copy())
                decision = slow.tick(obs)
            assert decision.instruction == R.aimed_subgoal(writer.note), (seed, writer.note)
            said.add(decision.aim)
        assert said == {"in reach", "press blue", "press orange"}
    finally:
        env.close()


V2_ARGUMENTS = [
    (["--task", "pick_cube", "--slow_level", "v2"], "it needs a plant task"),
    (["--task", "reason_mix", "--slow_level", "v2", "--expert"], "drop --expert"),
    (["--task", "reason_mix", "--slow_level", "v2", "--subgoals", "expert"], "it needs --subgoals none"),
    (["--task", "reason_mix", "--slow_level", "v2", "--notes", "reasoner"], "no reasoner"),
    (["--task", "prune", "--slow_level", "v2"], "use a reason_* task"),
    (["--task", "reason_mix", "--slow_level", "v2", "--scene", "eyes"], "give its --slow_port"),
    (["--task", "reason_mix", "--slow_level", "v2", "--program", "writer"], "give its --slow_port"),
    (["--task", "reason_mix", "--slow_level", "v2", "--recover", "-1"], "--recover must be 0 or more"),
    (["--task", "reason_mix", "--scene", "eyes"], "are for --slow_level v2"),
    (["--task", "reason_mix", "--aim"], "it needs --subgoals or --slow_level v2"),
    (["--task", "reason_mix", "--recover", "2"], "are for --slow_level v2"),
    (["--task", "reason_mix", "--resets", "2"], "are for --slow_level v2"),
    (["--task", "reason_mix", "--slow_level", "v2", "--resets", "-1"], "--resets must be 0 or more"),
    (["--task", "reason_mix", "--slow_level", "v2", "--slow_revoke", "-1"], "--slow_revoke must be 0 or more"),
    (["--task", "reason_mix", "--slow_level", "v2", "--slow_give_up", "-1"], "--slow_give_up 0 or more"),
    (["--task", "reason_mix", "--slow_unseen"], "are for --slow_level v2"),
    (["--task", "reason_mix", "--no-slow_unseen"], "are for --slow_level v2"),
    (["--task", "reason_mix", "--slow_revoke", "3"], "are for --slow_level v2"),
    (["--task", "reason_mix", "--slow_give_up", "80"], "are for --slow_level v2"),
]


@pytest.mark.parametrize("argv, message", V2_ARGUMENTS)
def test_the_v2_arguments_are_checked_before_any_server_is_reached(argv, message, tmp_path, monkeypatch):
    _env("prune").close()  # skipped without MuJoCo, the Franka model or a GL context
    from sim.pickup import eval as E

    monkeypatch.setattr(sys, "argv", ["eval.py", "--out", str(tmp_path), *argv])
    with pytest.raises(AssertionError, match=re.escape(message)):
        E.main()


def test_a_reasoning_task_with_a_test_situation_on_top_is_evaluated_and_recorded(tmp_path, monkeypatch):
    """A reasoning task and a scenario both record the cuts in order: an episode's record holds them once. The scripted
    expert drives, so no server is needed."""
    _env("reason_mix").close()  # skipped without MuJoCo, the Franka model or a GL context
    from sim.pickup import eval as E

    argv = ["--task", "reason_mix", "--scenario", "bump", "--expert", "--episodes", "1", "--videos", "0"]
    monkeypatch.setattr(sys, "argv", ["eval.py", "--out", str(tmp_path), *argv])
    E.main()
    results = json.loads((tmp_path / "results.json").read_text())
    (episode,) = results["per_episode"]
    assert results["scenario"]["spec"] == "bump" and episode["scenario"]["bump"]["applied"]
    assert episode["kind"] and isinstance(episode["cut_order"], list) and "notes_correct" in episode
