"""Rows of a stored window that the open-loop check hands to a policy (sim/pickup/open_loop_check.py).

The shards store one step before the current one. A policy with history reads it; a policy without
history must be given the current state and be compared from the current action on. The check used
to assume that the policy takes as many past steps as the shards store, and skipped every window of
a policy without history.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:  # sim/ is not an installed package
    sys.path.insert(0, str(REPOSITORY))

from sim.pickup.open_loop_check import window_rows  # noqa: E402
from vla_foundry.data.preprocessing.robotics.preprocess_masks import create_past_and_future_masks  # noqa: E402

STORED_PAST, STORED_FUTURE, EPISODE = 1, 14, 66  # the pickup datasets: 1 past + current + 14 future


def masks(index):
    return create_past_and_future_masks(index, STORED_PAST, STORED_FUTURE, EPISODE)


def test_policy_with_history_reads_the_stored_past_step():
    first, now, end, previous = window_rows(*masks(30), past=1, future=14)
    assert (first, now, end, previous) == (0, 1, 16, 0)


def test_policy_without_history_starts_at_the_current_step():
    first, now, end, previous = window_rows(*masks(30), past=0, future=14)
    assert (first, now, end) == (1, 1, 16)  # no past rows; 15 actions from the current one on
    assert previous == 0  # the action before the current step, for the "repeat it" baseline


def test_window_at_the_start_of_an_episode_has_no_previous_action():
    past_mask, future_mask = masks(0)  # the stored past row is padding
    assert window_rows(past_mask, future_mask, past=1, future=14) is None  # a history policy would read padding
    first, now, end, previous = window_rows(past_mask, future_mask, past=0, future=14)
    assert (first, now, end, previous) == (1, 1, 16, None)


def test_window_padded_at_the_end_of_an_episode_is_skipped():
    for past in (0, 1):
        assert window_rows(*masks(EPISODE - 3), past=past, future=14) is None
    # a policy that predicts fewer steps does not reach the padding
    assert window_rows(*masks(EPISODE - 3), past=0, future=2) == (1, 1, 4, 0)


def test_policy_that_needs_more_than_the_shards_store_is_an_error():
    with pytest.raises(ValueError, match="the shards store 1 past and 14 future"):
        window_rows(*masks(30), past=2, future=14)
    with pytest.raises(ValueError, match="the policy takes 0 and 20"):
        window_rows(*masks(30), past=0, future=20)


def test_rows_line_up_with_the_arrays():
    past_mask, future_mask = masks(30)
    action = np.arange(16)  # row i holds the action of step 29 + i
    first, now, end, previous = window_rows(past_mask, future_mask, past=0, future=14)
    assert action[now:end].tolist() == list(range(1, 16)) and action[previous] == 0
