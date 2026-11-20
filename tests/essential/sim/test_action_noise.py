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


class _Plant:
    """Just enough of a pruning PickupEnv for ActionNoise.calm: one uncut shoot whose cut zone's middle is at x=1."""

    plant_task = True
    active = ["shoot_0"]
    cut: set = set()

    def __init__(self, tcp):
        self.tcp = np.asarray(tcp, dtype=float)

    def tcp_pos(self):
        return self.tcp

    def shoot_axis(self, name):
        from sim.pickup.env import CUT_MID

        return np.array([1.0 - CUT_MID, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])


def test_quiet_near_a_cut_lets_the_offset_die_away_and_keeps_the_stream():
    open_jaws = np.ones(8, dtype=np.float32)
    far, near = _Plant([0.5, 0.0, 0.0]), _Plant([1.0, 0.02, 0.0])
    a, b = ActionNoise(7, 0.02, quiet=0.03), ActionNoise(7, 0.02)
    for _ in range(100):  # far from the shoot: the same noise as without the option
        assert np.array_equal(a.apply(open_jaws, far), b.apply(open_jaws))
    start = a.offset.copy()
    for _ in range(10):
        a.apply(open_jaws, near)
        b.apply(open_jaws)
    assert np.allclose(a.offset, start * ActionNoise.CORR**10)  # no fresh noise: the offset decays
    shut = open_jaws.copy()
    shut[7] = 0.0
    assert a.calm(far, shut) and not a.calm(far, open_jaws)  # the jaws commanded shut count as near
    before_a, before_b = a.offset.copy(), b.offset.copy()
    a.apply(open_jaws, far)
    b.apply(open_jaws)
    # the stream went on drawing while quiet: the next fresh draws of the two are the same
    assert np.allclose(a.offset - ActionNoise.CORR * before_a, b.offset - ActionNoise.CORR * before_b)
