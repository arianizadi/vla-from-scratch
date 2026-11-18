"""DART-style action noise for collected demonstrations (collect.py --action_noise, ActionNoise)."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:  # sim/ is not an installed package
    sys.path.insert(0, str(REPOSITORY))

pytest.importorskip("mujoco")
pytest.importorskip("pyarrow")

from sim.pickup.collect import ActionNoise  # noqa: E402


def test_no_noise_executes_the_expert_command():
    action = np.arange(8, dtype=np.float32)
    assert ActionNoise(3, 0.0).apply(action) is action


def test_noise_moves_the_arm_only_smoothly_and_repeatably():
    action = np.zeros(8, dtype=np.float32)
    a, b = ActionNoise(5, 0.02), ActionNoise(5, 0.02)
    runs = np.array([a.apply(action) for _ in range(20000)])
    assert np.array_equal(runs[:50], np.array([b.apply(action) for _ in range(50)]))  # from the seed
    assert runs.dtype == np.float32 and np.all(runs[:, 7] == 0)  # the gripper command is left alone
    std = runs[2000:, :7].std(axis=0)
    assert np.all(np.abs(std - 0.02) < 0.004)  # the stationary standard deviation
    lag1 = np.corrcoef(runs[2000:-1, 0], runs[2001:, 0])[0, 1]
    assert abs(lag1 - ActionNoise.CORR) < 0.03  # smooth from one step to the next
    assert not np.array_equal(runs[:50], np.array([ActionNoise(6, 0.02).apply(action) for _ in range(50)]))
