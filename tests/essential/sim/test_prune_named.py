"""Named shoots (sim/pickup/, tasks prune_named and prune_named_heldout): the instruction names one side shoot by its
height on the main stem; cut that one only.

The phrasing tests need nothing but numpy. The others need MuJoCo, a GL context for the cameras and the Franka model
(sim/pickup/scripts/fetch_assets.sh), and are skipped where any of them is missing.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:  # sim/ is not an installed package
    sys.path.insert(0, str(REPOSITORY))

from sim.pickup import phrasings  # noqa: E402

SEEDS = (1_000_000, 1_000_001, 1_000_002)


def _words(text):
    return set(re.findall(r"[a-z]+", text.lower()))


def _all_training():
    return {t for p in phrasings.POSITIONS for t in phrasings.training(p)}


def _all_heldout():
    return {t for p in phrasings.POSITIONS for texts in phrasings.heldout(p).values() for t in texts}


def test_heldout_phrasings_are_never_training_phrasings():
    train, heldout = _all_training(), _all_heldout()
    assert len(train) == 24 and len(heldout) > 100
    assert not train & heldout
    # each held-out instruction has at least one word that no training instruction has
    vocabulary = set().union(*map(_words, train))
    for text in heldout:
        assert _words(text) - vocabulary, f"held-out {text!r} uses only training words"
    # a position's phrasings never name another position (no instruction is shared between positions)
    for a in phrasings.POSITIONS:
        for b in phrasings.POSITIONS:
            if a != b:
                assert not set(phrasings.training(a)) & set(phrasings.training(b))
                for kind in phrasings.KINDS:
                    assert not set(phrasings.heldout(a)[kind]) & set(phrasings.heldout(b)[kind])


def test_sampling_draws_from_the_right_set():
    rng = np.random.default_rng(0)
    kinds = set()
    for _ in range(300):
        position = phrasings.POSITIONS[int(rng.integers(3))]
        text, kind = phrasings.sample(position, "train", rng)
        assert kind == "train" and text in phrasings.training(position)
        text, kind = phrasings.sample(position, "heldout", rng)
        assert kind in phrasings.KINDS and text in phrasings.heldout(position)[kind]
        kinds.add(kind)
    assert kinds == set(phrasings.KINDS)
    assert [phrasings.position_name(i, 2) for i in range(2)] == ["lowest", "highest"]
    assert [phrasings.position_name(i, 3) for i in range(3)] == ["lowest", "middle", "highest"]


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


def _plant(env):
    """The plant's random draws: shoot geometry and plate angle."""
    return [
        (n, tuple(np.round(env.shoots[n]["origin"], 6)), tuple(np.round(env.shoots[n]["dir"], 6))) for n in env.active
    ], round(env.plate_angle(), 6)


def test_training_episodes_use_training_phrasings_and_heldout_ones_name_the_same_shoot():
    named, heldout, prune = _env("prune_named"), _env("prune_named_heldout"), _env("prune")
    try:
        train, kinds, positions = _all_training(), set(), set()
        for seed in range(40):
            named.reset(seed=seed)
            heldout.reset(seed=seed + 1_000_000)
            assert named.instruction in train and named.phrasing_kind == "train"
            assert named.target_label != "middle" or len(named.active) == 3
            positions.add(named.target_label)
            # the named shoot is the one at that height: shoots are ordered from the lowest up
            heights = [named.shoots[n]["origin"][2] for n in named.active]
            assert heights == sorted(heights)
            assert named.target_label == phrasings.position_name(named.active.index(named.target), len(named.active))
            # the held-out task: the same plant and the same named shoot as prune_named, a held-out phrasing
            named.reset(seed=seed + 1_000_000)
            assert _plant(heldout) == _plant(named)
            assert (heldout.target, heldout.target_label) == (named.target, named.target_label)
            assert heldout.instruction not in train
            assert heldout.instruction in phrasings.heldout(heldout.target_label)[heldout.phrasing_kind]
            kinds.add(heldout.phrasing_kind)
            # and the plant is the one prune draws for the same seed
            prune.reset(seed=seed + 1_000_000)
            assert _plant(prune) == _plant(named)
        assert positions == set(phrasings.POSITIONS)
        assert kinds == set(phrasings.KINDS)
    finally:
        for env in (named, heldout, prune):
            env.close()


def _run(env, expert, steer=None):
    """Run the expert to the end of the episode. `steer`: a shoot the expert is sent to cut instead of the named
    one (the instruction and the success test still name the named one)."""
    done, info = False, {}
    named = env.target
    while not done:
        if steer is not None:
            env.target = steer
        action = expert.act()
        env.target = named
        _, done, info = env.step(action)
    return info


def test_the_expert_cuts_only_the_named_shoot():
    env = _env("prune_named")
    from sim.pickup.expert import ScriptedExpert

    try:
        three = next(s for s in range(1_000_000, 1_000_100) if env.reset(seed=s) is not None and len(env.active) == 3)
        cases = [(seed, None) for seed in SEEDS] + [(three, k) for k in range(3)]
        wins = 0
        for seed, k in cases:
            env.reset(seed=seed, target_index=k)
            if k is not None:
                assert env.target_label == phrasings.POSITIONS[k]
            expert = ScriptedExpert(env, seed=0)
            expert.reset()
            info = _run(env, expert)
            assert info["damaged"] is None, f"seed {seed}: {info['damaged']}"
            assert not info["wrong_object"] and info["wrong_shoots"] == []
            wins += info["success"]
            if info["success"]:
                assert info["cut"] == [env.target]
                for other in env.active:  # the other shoots are still on the plant
                    if other != env.target:
                        assert env.data.eq_active[env.shoot_weld[other]], f"seed {seed}: {other} came off"
                assert env.steps < env.max_steps
        assert wins >= len(cases) - 1, f"the expert cut the named shoot in {wins} of {len(cases)}"
    finally:
        env.close()


def test_cutting_another_shoot_ends_the_episode_as_a_failure():
    env = _env("prune_named")
    from sim.pickup.expert import ScriptedExpert

    try:
        env.reset(seed=SEEDS[0])
        named, wrong = env.target, next(n for n in env.active if n != env.target)
        expert = ScriptedExpert(env, seed=0)
        env.target = wrong  # the expert plans for the other shoot from the start
        expert.reset()
        env.target = named
        info = _run(env, expert, steer=wrong)
        assert info["cut"] == [wrong] and info["wrong_shoots"] == [wrong]
        assert info["wrong_object"] and not info["success"] and info["damaged"] is None
        assert env.steps < env.max_steps, "the episode did not end at the wrong cut"
    finally:
        env.close()
