"""The instruction type "aim" (vla_foundry/data/pipelines/robotics.py): a sub-goal with what its shoot needs, written
from the expert's note in a frame's metadata, in the words a served policy is given (sim/pickup/reasoning.py)."""

import random

import numpy as np
import pytest

from sim.pickup import reasoning as R
from vla_foundry.data.pipelines.robotics import aim_instruction, extract_robotics_fields
from vla_foundry.params.data_params import RoboticsDataParams

NOTE = "see: 3 shoots, pink flower | ask: second from the top | so: of 3, second from the top is the middle"


def _notes():
    """The expert's note for every drawn task with every aim it can have."""
    rng = np.random.default_rng(0)
    for kind in R.KINDS:
        for split in ("train", "heldout"):
            for _ in range(40):
                count = int(rng.integers(2, 4))
                flower = R.FLOWERS[int(rng.integers(len(R.FLOWERS)))]
                task = R.draw(kind, split, count, flower, rng)
                label = None if kind != "hidden" else "visible:cyan"
                for position in [R.position_name(i, count) for i in range(count)]:
                    for aim in ((position, True, None), (position, False, "blue"), (position, False, "orange")):
                        yield R.note_text(task, flower, label=label, aim=aim)
                yield R.note_text(task, flower, label=label, aim=None)
                if kind == "hidden":
                    for button in ("blue", "orange"):
                        yield R.note_text(task, flower, label=None, aim=("label", False, button))


def test_the_pipeline_writes_a_note_s_aim_as_the_simulator_s_side_does():
    notes = list(_notes())
    assert len(notes) > 1000
    for note in notes:
        assert aim_instruction(note) == R.aimed_subgoal(note), note
    said = {aim_instruction(note) for note in notes}
    cuts = {f"cut the {p} shoot, {a}" for p in R.POSITIONS for a in ("in reach", "press blue", "press orange")}
    assert said == cuts | {"show the label", "stop"}


def test_the_aim_is_in_reach_within_the_sector_and_else_the_button_of_the_shorter_way():
    assert R.aim_of(0.0) == R.aim_of(R.AIM_SECTOR) == R.aim_of(-R.AIM_SECTOR) == "in reach"
    assert R.aim_of(R.AIM_SECTOR + 0.01) == R.aim_of(3.1) == "press blue"  # + counter-clockwise: the blue button
    assert R.aim_of(-R.AIM_SECTOR - 0.01) == R.aim_of(-3.1) == "press orange"
    assert R.aim_of(None) == ""
    assert R.aimed("cut the lowest shoot", "press blue") == "cut the lowest shoot, press blue"
    assert R.aimed("cut the lowest shoot", "") == "cut the lowest shoot"
    assert R.aimed("show the label", "press blue") == "show the label" and R.aimed("stop", "in reach") == "stop"
    assert R.aimed_subgoal("") == "" and aim_instruction("") == ""
    assert R.subgoal(f"{NOTE} | aim: the middle, faces me | next: cut the middle shoot") == "cut the middle shoot"


def _sample(metadata, instructions):
    lowdim = {"action": np.zeros((15, 8)), "state": np.zeros((15, 8)), "past_mask": np.array([1] + [0] * 14)}
    lowdim["future_mask"] = np.array([0] + [1] * 14)
    return {
        "s.lowdim.npz": lowdim,
        "s.language_instructions.json": instructions,
        "s.metadata.json": dict(anchor_relative_idx=1, **metadata),
    }


def _instruction(types, metadata, instructions):
    fields = extract_robotics_fields(
        _sample(metadata, dict(instructions)),
        language_instruction_types=types,
        action_fields=["action"],
        proprioception_fields=["state"],
    )
    return fields["language_instruction"]


def test_a_sample_s_instruction_of_type_aim_comes_from_the_note_in_its_metadata():
    note = f"{NOTE} | aim: the middle, faces away, press orange | next: cut the middle shoot"
    given = {"original": "cut the second shoot from the top", "subgoal": "cut the middle shoot"}
    assert _instruction(["aim"], dict(note=note), given) == "cut the middle shoot, press orange"
    assert _instruction(["subgoal"], dict(note=note), given) == "cut the middle shoot"  # as before
    assert _instruction(["original"], dict(note=note), given) == "cut the second shoot from the top"
    random.seed(0)  # both types: one of the two, each about half of the time
    drawn = [_instruction(["subgoal", "aim"], dict(note=note), given) for _ in range(200)]
    assert set(drawn) == {"cut the middle shoot", "cut the middle shoot, press orange"}
    assert 60 < drawn.count("cut the middle shoot") < 140
    # a frame without a note has no instruction of that type: the other types listed stand, or none
    assert _instruction(["subgoal", "aim"], {}, given) == "cut the middle shoot"
    assert _instruction(["aim"], {}, given) == ""
    assert given == {"original": "cut the second shoot from the top", "subgoal": "cut the middle shoot"}


def test_the_data_parameters_take_the_type_aim():
    # the types are checked before anything is read from a dataset: a wrong one is named, "aim" is not
    with pytest.raises(RuntimeError, match="Invalid language instruction types: {'aims'}"):
        RoboticsDataParams(language_instruction_types=["subgoal", "aims"])
    try:
        RoboticsDataParams(language_instruction_types=["subgoal", "aim"])
    except (RuntimeError, IndexError) as error:  # no dataset was given: it stops later, at the statistics
        assert "Invalid language instruction types" not in str(error)
