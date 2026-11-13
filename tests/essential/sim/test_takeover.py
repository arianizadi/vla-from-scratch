"""Takeover demonstrations (sim/pickup/collect.py --policy_ports): another controller drives the first steps of a
pruning episode, then the scripted expert takes over from wherever the arm is and finishes the task. A demonstration
starts only once the expert is on a plan of its ordinary kind.

Needs MuJoCo, a GL context for the cameras and the Franka model (sim/pickup/scripts/fetch_assets.sh);
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
pytest.importorskip("pyarrow")

from sim.pickup.collect import hand_over  # noqa: E402
from sim.pickup.env import PANDA_XML, PickupEnv  # noqa: E402
from sim.pickup.expert import ScriptedExpert  # noqa: E402

SEEDS = (1_000_000, 1_000_001, 1_000_002)


def _env(task):
    if not PANDA_XML.exists():
        pytest.skip("Franka model not fetched (sim/pickup/scripts/fetch_assets.sh)")
    try:
        return PickupEnv(task)
    except Exception as error:  # no GL context on this machine
        pytest.skip(f"cannot render: {error}")


def _finish(env, expert):
    """Run the expert to the end of the episode; returns the last info."""
    done, info = False, {}
    while not done:
        _, done, info = env.step(expert.act())
    return info


def _on_its_ordinary_plan(env, expert, preamble):
    """Where a demonstration starts: the expert is on a plan of its own kind (_plan_prune), not on the moves that only
    a takeover makes. Without a preamble the set-point is the arm's actual pose, not one left over from before the
    handover."""
    assert not expert.preamble
    assert expert.prune_target in env.active and len(expert.waypoints) >= 2
    if preamble == 0:
        R = env.data.site_xmat[env.tcp_id].reshape(3, 3)
        assert np.linalg.norm(expert.des_pos - env.tcp_pos()) < 1e-6
        assert np.allclose(expert.des_R, R)
        assert np.allclose(expert.q_cmd, env.data.qpos[:7])


def _stand_in(env, seed):
    """A stand-in for a learned policy: an expert with draws of its own, two actions per query."""
    pilot = ScriptedExpert(env, seed=seed)
    pilot.reset()
    return lambda prev_obs, obs: [pilot.act(), pilot.act()]


@pytest.mark.parametrize("steps", [150, 450])
def test_the_expert_finishes_pruning_from_where_another_controller_left_the_arm(steps):
    env = _env("prune")
    try:
        wins, cut_before, preambles = 0, 0, 0
        for seed in SEEDS:
            obs = env.reset(seed=seed)
            expert = ScriptedExpert(env, seed=0)
            budget = env.max_steps
            obs, ended, preamble = hand_over(env, expert, obs, _stand_in(env, seed=7), steps, budget)
            assert ended is None, f"seed {seed}: the episode ended before the demonstration began: {ended}"
            assert env.steps == steps + preamble and env.max_steps == steps + budget
            cut_before += len(env.cut)
            preambles += preamble > 0
            _on_its_ordinary_plan(env, expert, preamble)
            info = _finish(env, expert)
            assert info["damaged"] is None, f"seed {seed}: {info['damaged']}"
            wins += info["success"]
            if info["success"]:
                assert info["cut"] == sorted(env.active)
        assert wins >= 2, f"the expert finished {wins} of {len(SEEDS)} plants after {steps} steps of another controller"
        if steps == 450:
            assert cut_before > 0, "no shoot was cut before the handover: the test does not cover a task half done"
        if steps == 150:  # on the first plant the stand-in is backing out of a cut: among the shoots
            assert preambles > 0, "no handover needed a preamble: the test does not cover backing out of the plant"
    finally:
        env.close()


def test_the_expert_takes_over_from_a_pose_it_would_never_choose():
    """A fixed-action 'policy' swings the arm aside and up, leans and turns the wrist and half-closes the gripper;
    the expert brings the hand back onto the lane (a preamble, not recorded) and prunes the plant."""
    env = _env("prune")
    try:
        obs = env.reset(seed=SEEDS[0])
        pose = np.concatenate([env.data.qpos[:7] + [0.4, 0.0, 0.0, 0.0, 0.8, 0.0, 0.8], [0.3]]).astype(np.float32)
        tilts = []

        def fixed(prev_obs, obs):
            tilts.append(np.degrees(np.arccos(-env.data.site_xmat[env.tcp_id].reshape(3, 3)[2, 2])))
            return [pose]

        expert = ScriptedExpert(env, seed=0)
        obs, ended, preamble = hand_over(env, expert, obs, fixed, 60, env.max_steps)
        assert ended is None, ended
        assert tilts[-1] > 20, f"the gripper leans only {tilts[-1]:.0f} degrees from pointing down"
        assert preamble > 0, "the hand was brought back without a preamble"
        _on_its_ordinary_plan(env, expert, preamble)
        info = _finish(env, expert)
        assert info["success"] and info["damaged"] is None, info
    finally:
        env.close()
