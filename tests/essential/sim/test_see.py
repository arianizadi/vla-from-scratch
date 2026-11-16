"""The `see` field of a note given from outside (policy_server.Policy.write_note, eval.py --see expert).

A reasoner that miscounts the shoots writes a wrong `see` field and reasons correctly from it. Handing it the field
and letting it write the rest measures its reasoning apart from what it sees (tutorial/architecture-v2.md, E0). The
backbone is replaced by a stub here: the tests are about when a note is written again and what it then says.
"""

import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:  # sim/ is not an installed package
    sys.path.insert(0, str(REPOSITORY))

pytest.importorskip("torch")

from sim.pickup.policy_server import Policy, rewrite_with_see  # noqa: E402

SAME = "same"
NOTE = "see: 2 shoots, red flower | ask: second from the top | so: of 2, second from the top is the lowest"
SEE = "3 shoots, red flower"


def _policy(free, rest="ask: second from the top | so: of 3, second from the top is the middle"):
    """A policy whose backbone writes `free` after "now:" and `rest` after a given beginning of a note."""
    policy = Policy.__new__(Policy)  # no checkpoint: only write_note is used
    policy.reasons = True
    starts = []

    def write(images, instruction, held, start=None):
        starts.append(start)
        return free if start is None else rest

    policy._write = write
    return policy, starts


def test_a_note_is_written_again_only_when_a_see_field_is_given_and_does_not_stand():
    assert not rewrite_with_see(NOTE, "", None) and not rewrite_with_see(SAME, NOTE, None)  # no field given
    assert rewrite_with_see(NOTE, "", SEE)  # a new note: its own see field may be wrong
    assert rewrite_with_see(SAME, NOTE, SEE)  # the note held was written for other facts
    assert not rewrite_with_see(SAME, f"see: {SEE} | ask: x", SEE)  # the note held already starts from them
    assert rewrite_with_see(SAME, None, SEE)  # nothing held yet


def test_without_a_see_field_the_backbone_writes_as_before():
    policy, starts = _policy(NOTE)
    assert policy.write_note({}, "cut it", None) == (NOTE, NOTE) and starts == [None]
    policy, starts = _policy(SAME)
    assert policy.write_note({}, "cut it", NOTE) == (SAME, NOTE) and starts == [None]
    policy, _ = _policy("")  # nothing written counts as "same"
    assert policy.write_note({}, "cut it", NOTE) == (SAME, NOTE)


def test_a_new_note_is_written_again_from_the_given_see_field():
    policy, starts = _policy(NOTE)
    now, held = policy.write_note({}, "cut it", None, SEE)
    assert starts == [None, f"see: {SEE} |"]  # the second pass starts after the field
    assert now == held == f"see: {SEE} | ask: second from the top | so: of 3, second from the top is the middle"


def test_a_note_held_for_the_same_facts_stands():
    policy, starts = _policy(SAME)
    standing = f"see: {SEE} | ask: second from the top"
    assert policy.write_note({}, "cut it", standing, SEE) == (SAME, standing) and starts == [None]


def test_a_note_held_for_other_facts_is_written_again_even_if_the_backbone_says_same():
    policy, starts = _policy(SAME)
    now, held = policy.write_note({}, "cut it", NOTE, SEE)
    assert starts == [None, f"see: {SEE} |"] and now == held and now.startswith(f"see: {SEE} | ask:")


def test_the_request_carries_the_see_field_only_when_one_is_given():
    pytest.importorskip("mujoco")
    import numpy as np

    from sim.pickup.eval import infer_request

    obs = dict(front=np.zeros((8, 8, 3), np.uint8), wrist=np.zeros((8, 8, 3), np.uint8), state=np.zeros(8, np.float32))
    assert "see" not in infer_request(obs, obs, np.zeros(8), "cut it", 10)
    assert infer_request(obs, obs, np.zeros(8), "cut it", 10, see=SEE)["see"] == SEE
