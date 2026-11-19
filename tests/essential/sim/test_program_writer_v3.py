"""The program writer for v3's language (sim/pickup/program_writer_v3.py): the worked examples are training instructions
in pairs and show every new construct, the two sets are disjoint, the prompt uses no word that only held-out
instructions use, the held-out compositions are new, the tasks carry their situation's own program, constrained writing
ends in a program of v3, and the scoring and asking twice count what they say.

The tokenizer and model are the toy ones of test_program_writer.py (nothing is downloaded). The situations' frames come
from scenarios.py, which needs MuJoCo.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:  # sim/ is not an installed package
    sys.path.insert(0, str(REPOSITORY))

from sim.pickup import program as P  # noqa: E402
from sim.pickup import program_writer_v3 as V  # noqa: E402

_spec = importlib.util.spec_from_file_location("_writer_tests", Path(__file__).with_name("test_program_writer.py"))
_toys = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_toys)

NEW = ("CutFrom(", "Then(", "Nothing()", "Take(", "Count(", "tie,any")
# Words of the description that only held-out instructions use: the constructs' own names
NAMES = {"middle", "nothing"}


def _programs(pairs):
    return [P.parse(written, "v3") for _, written in pairs]


@pytest.mark.parametrize("prompt", sorted(V.PROMPTS))
def test_the_examples_are_training_instructions_in_pairs_that_show_every_new_construct(prompt):
    train, heldout = V.training_instructions(), V.heldout_instructions()
    assert not train & heldout
    for which in V.EXAMPLES:
        pairs = V.examples(16, which, prompt)
        assert len(pairs) == 16 and V.examples(8, which, prompt) == pairs[::2]
        said = [s for s, _ in pairs]
        assert len(set(said)) == 16 and set(said) <= train and not set(said) & heldout
        programs = _programs(pairs)
        assert all(P.text(node) == written for node, (_, written) in zip(programs, pairs, strict=True))
        for first, second in zip(programs[::2], programs[1::2], strict=True):  # a pair: two programs that differ
            assert not P.equivalent_v3(first, second)
        shown = "".join(written for _, written in pairs)
        assert all(construct in shown for construct in NEW), which
    assert not {s for s, _ in V.examples(16, "A", prompt)} & {s for s, _ in V.examples(16, "B", prompt)}  # disjoint


@pytest.mark.parametrize("prompt", sorted(V.PROMPTS))
def test_an_example_from_a_situation_s_frame_has_that_situation_s_program(prompt):
    """The examples' programs are those the tasks carry for the same instruction."""
    known = {}
    rng = np.random.default_rng(0)
    for situation in V.SITUATIONS:
        for _ in range(600):
            task = V.draw(situation, "train", rng)
            assert known.setdefault(task.instruction, task.program) == task.program
    for which in V.EXAMPLES:
        for said, written in V.examples(16, which, prompt):
            if said in known:
                assert P.equivalent_v3(P.parse(known[said], "v3"), P.parse(written, "v3")), said
    assert len([s for s, _ in V.examples(16, "A", prompt) if s in known]) >= 9


@pytest.mark.parametrize("prompt", sorted(V.PROMPTS))
def test_the_prompt_uses_no_word_that_only_held_out_instructions_use(prompt):
    forbidden = V.held_out_only_words()
    assert {"snip", "branch", "ribbon"} <= forbidden  # the held-out frames' own words
    for which in V.EXAMPLES:
        text = V.PROMPTS[prompt][0] + " ".join(said for said, _ in V.examples(16, which, prompt))
        assert V.words(text) & forbidden <= NAMES, V.words(text) & forbidden


def test_held_out_compositions_are_new_and_the_first_set_is_training():
    first = [node for node, _ in V.COMPOSITIONS]
    shown = [P.parse(w, "v3") for p in V.PROMPTS for which in V.EXAMPLES for _, w in V.examples(16, which, p)]
    for node, frames in V.HELDOUT_COMPOSITIONS:
        assert P.in_language(node, "v3") and len(frames) == 2
        assert not any(P.equivalent_v3(node, other) for other in first + shown), P.text(node)
    assert {said for _, frames in V.COMPOSITIONS for said in frames} <= V.training_instructions()


def test_the_tasks_carry_their_situation_s_program_and_its_plant():
    tasks = V.draw_tasks(30, seed=1)
    groups = {(t.situation, t.split) for t in tasks}
    assert ("prune", "heldout") not in groups and ("compose", "heldout") in groups and len(groups) == 12
    for task in tasks:
        assert len(task.attributes) == task.count in (2, 3)
        flags = V.score(task.program, task)
        assert flags["format"] and flags["exact"] and flags["meaning"] and flags["resolved"], task
        if task.split == "heldout":
            assert task.instruction in V.heldout_instructions()
        else:
            assert task.instruction in V.training_instructions()
        if task.situation == "ties":
            tied = [a["tie"] for a in task.attributes if a["tie"] != "none"]
            assert len(tied) == (task.count if task.mode == "colour" else 1)
    assert V.draw_tasks(30, seed=1) == tasks


def test_scoring_tells_form_text_meaning_and_shoots_apart():
    task = V.TendTask("order", "train", "train", "top_down", "cut the side shoots from the top down", 3, (
        {"tie": "none", "sick": "no", "bud": "no"},) * 3, "CutFrom(top,All())")
    assert V.score("Cut(All())", task) == dict(V.score("Cut(All())", task), format=True, exact=False, meaning=False)
    assert not V.score("Cut(All())", task)["resolved"]  # any order against from the top down
    same = V.score("CutFrom(top,Except(All(),Nothing))", task)
    assert not same["format"] and same["got"] == "not a program"
    alias = V.score("CutFrom(top,Take(3,any,All()))", task)  # another text, the same cuts on every plant
    assert alias["format"] and not alias["exact"] and alias["meaning"] and alias["resolved"]
    sick = V.TendTask("sick", "train", "train", "sick", "prune every sick shoot", 2, (
        {"tie": "none", "sick": "no", "bud": "no"}, {"tie": "none", "sick": "no", "bud": "no"}), "Cut(Filter(sick,yes,All()))")
    wrong = V.score("Cut(Filter(bud,no,All()))", sick)  # cuts both of these healthy shoots, where none is meant
    assert wrong["format"] and not wrong["meaning"] and not wrong["resolved"]
    nothing = V.score("Nothing()", sick)  # right on this plant, wrong in meaning
    assert nothing["resolved"] and not nothing["meaning"]


def test_constrained_writing_ends_in_a_program_of_v3():
    tokenizer = _toys._toy_tokenizer()
    for seed in range(4):
        writer = V.writer(_toys._model(tokenizer, seed), tokenizer, shots=8, max_new_tokens=V.MAX_NEW_TOKENS)
        assert writer.language == "v3" and writer.prompt("cut two shoots").startswith(V.SYSTEM_V3)
        for said in ("cut two shoots", "snip the sick branches one by one, top to bottom"):
            out = writer.write(said)
            assert out["complete"] and P.text(P.parse(out["program"], "v3")) == out["program"], out


def test_each_instruction_is_written_once_and_asking_twice_counts_what_it_stops():
    tasks = V.draw_tasks(5, seed=2)
    calls = []

    def write(said):
        calls.append(said)
        return dict(program="Cut(All())", tokens=4, seconds=0.0)

    rows = V.evaluate(write, tasks)
    assert len(calls) == len({t.instruction for t in tasks}) and len(rows) == len(tasks)
    summary = V.summarize(rows)
    assert summary["all"]["n"] == len(tasks) and summary["all"]["format"] == len(tasks)
    assert summary["prune/train"]["meaning"] == 5  # Cut(All()) is pruning's program
    gold = V.evaluate(lambda said: dict(program=next(t.program for t in tasks if t.instruction == said), tokens=1,
                                        seconds=0.0), tasks)
    counts = V.twice(gold, rows)["all"]
    wrong = sum(not r["meaning"] for r in rows)
    assert counts["n"] == len(tasks) and counts["first_wrong"] == 0
    assert counts["blocked_right"] + counts["agree_right"] == len(tasks)  # the first (the task's own) is always right
    assert counts["agree_right"] == sum(r["meaning"] for r in rows)
    same = V.twice(rows, rows)["all"]
    assert same["agree"] == len(tasks) and same["agree_wrong"] == wrong
