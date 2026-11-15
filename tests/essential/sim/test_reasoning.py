"""Reasoning tasks (sim/pickup/reasoning.py, notes.py; tasks reason_*): the shoot to cut follows from the instruction
and what is seen or was seen, not from the instruction's words alone.

The task and note tests need nothing but numpy. The others need MuJoCo, a GL context for the cameras and the Franka
model (sim/pickup/scripts/fetch_assets.sh), and are skipped where any of them is missing.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:  # sim/ is not an installed package
    sys.path.insert(0, str(REPOSITORY))

from sim.pickup import reasoning as R  # noqa: E402


def _draws(kind, split, n=600, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        count, flower = int(rng.integers(2, 4)), R.FLOWERS[int(rng.integers(4))]
        yield count, flower, R.draw(kind, split, count, flower, rng)


@pytest.mark.parametrize("kind", R.KINDS)
def test_targets_follow_from_the_instruction_and_the_plant(kind):
    for count, flower, task in _draws(kind, "train"):
        assert task.kind == kind and task.count == count and task.phrasing == "train"
        assert all(0 <= i < count for i in task.targets) and len(set(task.targets)) == len(task.targets)
        p = task.params
        if kind == "ordinal":
            assert task.targets == ((p["k"] - 1,) if p["end"] == "bottom" else (count - p["k"],))
        elif kind == "conditional":
            assert p["holds"] == (p["colour"] == flower)
            assert task.target_positions == ((p["yes"],) if p["holds"] else (p["no"],))
        elif kind == "exclusion":
            assert task.target_positions == tuple(
                R.position_name(i, count)
                for i in range(count)
                if i != {"lowest": 0, "middle": 1, "highest": count - 1}[p["keep"]]
            )
        else:
            assert task.target_positions == (R.LABEL_MEANS[p["label"]],)
            assert p["label"] != "yellow" or count == 3


def test_hand_worked_cases():
    assert R.ordinal_index(2, "top", 3) == 1 and R.ordinal_index(2, "top", 2) == 0  # the middle, the lowest
    assert R.ordinal_index(3, "bottom", 3) == 2 and R.ordinal_index(1, "bottom", 2) == 0
    assert R.position_index("highest", 2) == 1 and R.position_index("highest", 3) == 2
    with pytest.raises(AssertionError):
        R.position_index("middle", 2)


@pytest.mark.parametrize("kind", R.KINDS)
def test_heldout_instructions_are_never_training_instructions(kind):
    training = R.training_instructions(kind)
    drawn = {task.instruction for _, _, task in _draws(kind, "train")}
    assert drawn <= training
    heldout = R.all_heldout_instructions(kind)
    assert heldout and not heldout & training
    kinds = {task.phrasing for _, _, task in _draws(kind, "heldout")}
    assert kinds == ({"frame"} if kind == "hidden" else set(R.HELDOUT_KINDS))


def test_a_lookup_from_the_words_cannot_solve_any_kind():
    """However well a policy maps instructions to shoots, without the plant it is right at most this often."""
    ceilings = {kind: R.lookup_ceiling(kind) for kind in R.KINDS}
    assert all(c < 0.7 for c in ceilings.values()), ceilings
    assert ceilings["conditional"] < 0.56 and ceilings["exclusion"] < 0.56 and ceilings["hidden"] < 0.5


def test_held_out_compositions_are_new_combinations_of_trained_parts():
    c = R.compositions()
    words = {w for k, e in c["ordinal_train"] for w in (k, e)}
    for k, e in c["ordinal_heldout"]:  # the number and the end were each trained, never together
        assert k in words and e in words and (k, e) not in c["ordinal_train"]
    colours = {col for col, _ in c["conditional_train"]}
    branches = {y for _, y in c["conditional_train"]}
    for col, y in c["conditional_heldout"]:
        assert col in colours and y in branches and (col, y) not in c["conditional_train"]


def test_notes_have_five_fields_that_parse_back():
    rng = np.random.default_rng(3)
    task = R.draw("ordinal", "train", 3, "red", rng)
    while task.params != dict(k=2, end="top"):
        task = R.draw("ordinal", "train", 3, "red", rng)
    note = R.note_text(task, "red", aim=("middle", False, "blue"))
    assert note == (
        "see: 3 shoots, red flower | ask: second from the top | so: of 3, second from the top is the middle | "
        "aim: the middle, faces away, press blue | next: cut the middle shoot"
    )
    fields = R.note_fields(note)
    assert list(fields) == list(R.NOTE_FIELDS)
    assert R.subgoal(note) == "cut the middle shoot" and R.subgoal_target(R.subgoal(note)) == "middle"
    assert R.subgoal("garbled") == "" and R.subgoal_target("show the label") is None
    done = R.note_text(task, "red", cut=(1,), aim=None)
    assert R.subgoal(done) == "stop"


def test_the_hidden_kind_notes_remember_the_card():
    task = R.Task("hidden", "x", "train", 3, (2,), "the shoot the label names", dict(label="cyan"))
    unseen = R.note_text(task, "pink", label=None, aim=("label", False, "orange"))
    assert "label not seen" in unseen and R.subgoal(unseen) == "show the label"
    read = R.note_text(task, "pink", label="hidden:cyan", aim=("highest", True, None))
    assert "label hidden, read cyan" in read and "cyan names the highest" in read
    assert R.subgoal(read) == "cut the highest shoot"


def test_exclusion_notes_keep_track_of_the_shoots_done():
    task = R.Task("exclusion", "x", "train", 3, (0, 1), "all but the highest", dict(keep="highest"))
    note = R.note_text(task, "red", cut=(0,), aim=("middle", True, None))
    assert "so: cut the lowest and the middle; the lowest is cut" in note


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
    return [
        (n, tuple(np.round(env.shoots[n]["origin"], 6)), tuple(np.round(env.shoots[n]["dir"], 6))) for n in env.active
    ], round(env.plate_angle(), 6)


def test_the_plant_is_the_one_prune_draws_and_only_the_hidden_kind_shows_a_card():
    mix, heldout, prune = _env("reason_mix"), _env("reason_mix_heldout"), _env("prune")
    try:
        assert "label_col" not in [prune.model.geom(i).name for i in range(prune.model.ngeom)]  # prune is unchanged
        kinds = set()
        for seed in range(1_000_000, 1_000_040):
            for env in (mix, heldout, prune):
                env.reset(seed=seed)
            assert _plant(mix) == _plant(prune) == _plant(heldout)
            kinds.add(mix.reason_kind)
            assert mix.max_steps == R.MAX_STEPS[mix.reason_kind]
            assert mix.targets == [mix.active[i] for i in mix.reason_task.targets]
            face = mix.label_geoms[1]
            shown = mix.model.geom_group[face] == 2
            assert shown == (mix.reason_kind == "hidden") == (mix.label_colour is not None)
            assert heldout.phrasing_kind != "train" and mix.phrasing_kind == "train"
        assert kinds == set(R.KINDS)
    finally:
        for env in (mix, heldout, prune):
            env.close()


def test_the_card_faces_the_camera_after_the_turn_it_asks_for():
    env = _env("reason_hidden")
    try:
        env.reset(seed=1_000_000)
        needed = env.label_turn_needed()
        env.data.qpos[env.plate_qadr] += needed  # turn the plate by hand
        import mujoco

        mujoco.mj_forward(env.model, env.data)
        assert abs(env.label_turn_needed()) < 1e-6 and env.label_visible()
        env.data.qpos[env.plate_qadr] += np.pi
        mujoco.mj_forward(env.model, env.data)
        assert not env.label_visible()
    finally:
        env.close()


@pytest.mark.parametrize("kind", R.KINDS)
def test_the_expert_cuts_exactly_the_shoots_meant_and_writes_notes(kind):
    env = _env(f"reason_{kind}")
    from sim.pickup.expert import ScriptedExpert
    from sim.pickup.notes import NoteWriter

    try:
        wins = 0
        for seed in (1_000_000, 1_000_001, 1_000_002):
            env.reset(seed=seed)
            expert, writer = ScriptedExpert(env, seed=0), NoteWriter(env)
            expert.reset()
            done = False
            while not done:
                writer.update()
                _, done, info = env.step(expert.act())
            writer.update()
            assert info["damaged"] is None and info["wrong_shoots"] == []
            wins += info["success"]
            if info["success"]:
                assert sorted(info["cut"]) == sorted(env.targets)
                steps = [t for t, _ in writer.written]
                assert steps[0] == 0 and R.subgoal(writer.written[-1][1]) == "stop"
                first = R.subgoal(writer.written[0][1])
                assert first == "show the label" or R.subgoal_target(first) in env.reason_task.target_positions
                if kind == "hidden" and "label not seen" in writer.written[0][1]:  # it turned the plant to read
                    assert any("names the" in text for _, text in writer.written[1:])
        assert wins >= 2, f"the expert succeeded in {wins} of 3"
    finally:
        env.close()


def test_cutting_a_shoot_not_meant_fails_and_a_swapped_note_names_it():
    env = _env("reason_conditional")
    from sim.pickup.expert import ScriptedExpert
    from sim.pickup.notes import NoteWriter

    try:
        env.reset(seed=1_000_000)
        meant = list(env.targets)
        wrong = next(n for n in env.active if n not in meant)
        writer = NoteWriter(env)
        writer.update()
        swapped = writer.text(targets=[wrong])
        position = R.position_name(env.active.index(wrong), len(env.active))
        assert R.subgoal(swapped) == f"cut the {position} shoot" and R.subgoal(writer.note) != R.subgoal(swapped)
        expert = ScriptedExpert(env, seed=0)
        env.targets = [wrong]  # the expert plans for the other shoot
        expert.reset()
        done = False
        while not done:
            env.targets = [wrong]
            action = expert.act()
            env.targets = meant
            _, done, info = env.step(action)
        assert info["wrong_shoots"] == [wrong] and info["wrong_object"] and not info["success"]
        assert env.steps < env.max_steps, "the episode did not end at the wrong cut"
    finally:
        env.close()


def test_training_paraphrases_say_the_same_task_without_held_out_words():
    import json

    table = R.paraphrase_table()
    assert set(table) == set().union(*(R.training_instructions(k) for k in R.KINDS))
    forbidden = R.heldout_only_words()
    heldout = set().union(*(R.all_heldout_instructions(k) for k in R.KINDS))
    for instruction, alternatives in table.items():
        assert len(alternatives) >= 6
        for alt in alternatives:
            assert not R._words(alt) & forbidden, (alt, R._words(alt) & forbidden)
            assert alt not in heldout
        # the parts that decide the task (ordinal and end, colour and branches, the shoot kept) are kept
        keep = {w for w in R._words(instruction) if w in {*R.ORDINALS.values(), *R.ENDS, *R.FLOWERS, *R.POSITIONS}}
        assert all(keep <= R._words(alt) for alt in alternatives), instruction
    # the committed table is this one
    committed = json.loads((REPOSITORY / "sim" / "pickup" / "configs" / "reason_paraphrases.json").read_text())
    assert committed == table
