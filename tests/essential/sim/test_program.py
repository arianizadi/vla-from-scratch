"""Typed instruction programs (sim/pickup/program.py): the program of every benchmark task selects the shoots the task
means, text and parse are inverses, the grammar acceptor for constrained decoding is exact, and a plan evaluated on a
scene memory says what to do next with stable part identities.

Pure Python: nothing here needs MuJoCo. The scene memory is a small fake with the fields a plan reads.
"""

import itertools
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:  # sim/ is not an installed package
    sys.path.insert(0, str(REPOSITORY))

from sim.pickup import program as P  # noqa: E402
from sim.pickup import reasoning as R  # noqa: E402

# The attribute values, and the selections that contain no other selection, written out here rather than taken from
# program.py, so that the tests hold the module to the grammar in its docstring.
ATTRIBUTE_VALUES = (
    ("tie", "red"),
    ("tie", "blue"),
    ("tie", "yellow"),
    ("tie", "white"),
    ("sick", "yes"),
    ("bud", "yes"),
    ("bud", "no"),
)
LEAVES = (P.All(), P.Middle(), P.ReadLabel(), *(P.Rank(k, end) for k in (1, 2, 3) for end in ("top", "bottom")))


def _draws(kind, split, n=600, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        count, flower = int(rng.integers(2, 4)), R.FLOWERS[int(rng.integers(4))]
        yield count, flower, R.draw(kind, split, count, flower, rng)


def _random_selection(rng, depth):
    """A random selection in which Except and Filter nest at most `depth` deep."""
    roll = rng.random()
    if depth > 0 and roll < 0.3:
        return P.Except(_random_selection(rng, depth - 1), _random_selection(rng, depth - 1))
    if depth > 0 and roll < 0.5:
        return P.Filter(*rng.choice(ATTRIBUTE_VALUES), _random_selection(rng, depth - 1))
    return rng.choice(LEAVES)


def _random_program(rng):
    if rng.random() < 0.4:
        return P.Cut(_random_selection(rng, 2))
    cond = P.FlowerIs(rng.choice(R.FLOWERS))
    return P.If(cond, P.Cut(_random_selection(rng, 2)), P.Cut(_random_selection(rng, 2)))


def _all_selections(depth):
    """Every selection in which Except and Filter nest at most `depth` deep."""
    if depth == 0:
        return list(LEAVES)
    inner = _all_selections(depth - 1)
    return [
        *LEAVES,
        *(P.Except(a, b) for a in inner for b in inner),
        *(P.Filter(attr, value, s) for attr, value in ATTRIBUTE_VALUES for s in inner),
    ]


def _parses(written):
    try:
        P.parse(written)
    except P.ProgramError:
        return False
    return True


def _corrupt(written, rng):
    """`written` with one character replaced, inserted, deleted, or swapped with the next."""
    i = rng.randrange(len(written))
    char = rng.choice(sorted(P.ALPHABET))
    how = rng.choice(("replace", "insert", "delete", "swap"))
    if how == "replace":
        return written[:i] + char + written[i + 1 :]
    if how == "insert":
        return written[:i] + char + written[i:]
    if how == "delete":
        return written[:i] + written[i + 1 :]
    return written[:i] + written[i + 1 : i + 2] + written[i : i + 1] + written[i + 2 :]


# ------------------------------------------------------------------------------------------- the programs of the tasks
@pytest.mark.parametrize("split", ("train", "heldout"))
@pytest.mark.parametrize("kind", R.KINDS)
def test_the_program_of_a_task_selects_the_shoots_the_task_means(kind, split):
    for count, flower, task in _draws(kind, split):
        program = P.from_task(task)
        label = task.params.get("label")  # the card's colour (hidden kind); the other kinds must not need it
        assert P.targets_on(program, count, flower, label) == task.target_positions, (task, P.text(program))
        if kind == "hidden":
            with pytest.raises(P.NeedsObservation) as need:
                P.targets_on(program, count, flower, None)
            assert need.value.what == "label"


def test_the_programs_of_the_four_kinds_written_out():
    def written(kind, **params):
        return P.text(P.from_task(R.Task(kind, "an instruction", "train", 3, (0,), "an ask", params)))

    assert written("ordinal", k=2, end="top") == "Cut(Rank(2,top))"
    assert written("ordinal", k=3, end="top") == "Cut(Rank(3,top))"
    conditional = written("conditional", colour="red", yes="lowest", no="highest", holds=True)
    assert conditional == "If(FlowerIs(red),Cut(Rank(1,bottom)),Cut(Rank(1,top)))"
    assert written("exclusion", keep="highest") == "Cut(Except(All(),Rank(1,top)))"
    assert written("exclusion", keep="middle") == "Cut(Except(All(),Middle()))"
    assert written("hidden", label="cyan") == "Cut(ReadLabel())"


def test_hand_worked_targets():
    second_from_top = P.parse("Cut(Rank(2,top))")
    assert P.targets_on(second_from_top, 3, "red", None) == ("middle",)
    assert P.targets_on(second_from_top, 2, "red", None) == ("lowest",)
    for no_such_shoot in ("Cut(Rank(3,top))", "Cut(Middle())", "Cut(Except(All(),Middle()))"):
        with pytest.raises(P.ProgramError):
            P.targets_on(P.parse(no_such_shoot), 2, "red", None)
    assert P.targets_on(P.parse("Cut(Except(All(),Rank(1,top)))"), 3, "pink", None) == ("lowest", "middle")
    branch = P.parse("If(FlowerIs(red),Cut(Rank(1,bottom)),Cut(Rank(1,top)))")
    assert P.targets_on(branch, 3, "red", None) == ("lowest",)
    assert P.targets_on(branch, 3, "orange", None) == ("highest",)
    with pytest.raises(P.NeedsObservation) as need:
        P.targets_on(branch, 3, None, None)
    assert need.value.what == "flower"
    card = P.parse("Cut(ReadLabel())")
    assert P.targets_on(card, 3, "red", "yellow") == ("middle",)
    assert P.targets_on(card, 2, "red", "cyan") == ("highest",)
    with pytest.raises(P.ProgramError):  # yellow names the middle shoot, and a plant with two has none
        P.targets_on(card, 2, "red", "yellow")
    with pytest.raises(P.NeedsObservation) as need:  # no plant here has attributes
        P.targets_on(P.parse("Cut(Filter(tie,red,All()))"), 3, "red", None)
    assert need.value.what == "tie"
    # a Filter over no shoot needs nothing; and an empty selection is a program with nothing to cut
    assert P.targets_on(P.parse("Cut(Filter(tie,red,Except(All(),All())))"), 3, "red", None) == ()
    # a selection alone reads as Cut(selection)
    assert P.targets_on(P.Rank(1, "top"), 2, "red", None) == ("highest",)


# ----------------------------------------------------------------------------------------------------------------- text
def test_canonical_text_parses_back_to_the_same_program():
    for kind in R.KINDS:
        for split in ("train", "heldout"):
            for _, _, task in _draws(kind, split, n=200):
                program = P.from_task(task)
                written = P.text(program)
                assert " " not in written and P.is_complete(written)
                assert P.parse(written) == program and P.text(P.parse(written)) == written
    rng = random.Random(0)
    for _ in range(500):
        program = _random_program(rng)
        assert P.parse(P.text(program)) == program
    # spaces between words are read past; the text written back has none
    spaced = " If( FlowerIs(red) , Cut(Rank(1, bottom)),Cut( Rank(1,top) ) ) "
    assert P.text(P.parse(spaced)) == "If(FlowerIs(red),Cut(Rank(1,bottom)),Cut(Rank(1,top)))"


MALFORMED = (
    "",
    "Cut",
    "Cut(",
    "Cut()",
    "Cut(All()",
    "Cut(All()))",
    "Cut(All())Cut(All())",
    "Cut(All()),",
    "cut(All())",
    "Cut(All)",
    "Cut((All()))",
    "Cut(All(),)",
    "Cut(,All())",
    "Cut(All() All())",
    "Cut(Shoot())",
    "Cut(Rank(2))",
    "Cut(Rank(2,top,top))",
    "Cut(Rank(0,top))",
    "Cut(Rank(4,top))",
    "Cut(Rank(02,top))",
    "Cut(Rank(two,top))",
    "Cut(Rank(2,up))",
    "Cut(Rank(top,2))",
    "Cut(Rank(2,t op))",
    "Cut(Middle(1))",
    "Cut(ReadLabel(white))",
    "Cut(Except(All()))",
    "Cut(Filter(tie,green,All()))",
    "Cut(Filter(sick,no,All()))",
    "Cut(Filter(colour,red,All()))",
    "Cut(Filter(tie,red))",
    "If(FlowerIs(blue),Cut(All()),Cut(All()))",
    "If(FlowerIs(Red),Cut(All()),Cut(All()))",
    "If(FlowerIs(red),Cut(All()))",
    "Cut(All());",
    "Cut(All_())",
    "Cut(Allé())",
    "Cut(Rank(²,top))",
    "Rank(2,top)",  # a selection alone is not a program
    "FlowerIs(red)",
    "red",
)
TYPE_ERRORS = (
    "Cut(FlowerIs(red))",
    "Cut(Cut(All()))",
    "If(All(),Cut(All()),Cut(All()))",
    "If(Rank(1,top),Cut(All()),Cut(All()))",
    "If(FlowerIs(red),All(),Cut(All()))",
    "If(FlowerIs(red),Cut(All()),Rank(1,top))",
    "Cut(Except(FlowerIs(red),All()))",
    "Cut(Except(All(),Cut(Middle())))",
    "Cut(Filter(tie,red,FlowerIs(red)))",
    "Cut(Filter(tie,red,Cut(All())))",
    "Cut(Except(All(),If(FlowerIs(red),Cut(All()),Cut(All()))))",
)
TOO_DEEP = (
    "Cut(Except(Except(Except(All(),Middle()),Middle()),Middle()))",
    "Cut(Except(All(),Except(All(),Except(All(),Middle()))))",
    "Cut(Filter(tie,red,Filter(tie,blue,Filter(bud,yes,All()))))",
    "Cut(Except(All(),Filter(bud,no,Except(All(),Middle()))))",
    "If(FlowerIs(red),Cut(All()),Cut(Except(Except(Filter(sick,yes,All()),Middle()),All())))",
    "If(FlowerIs(red),If(FlowerIs(pink),Cut(All()),Cut(Middle())),Cut(All()))",
    "If(FlowerIs(red),Cut(All()),If(FlowerIs(pink),Cut(All()),Cut(Middle())))",
)


def test_malformed_text_is_rejected():
    for written in MALFORMED:
        with pytest.raises(P.ProgramError):
            P.parse(written)
        assert not P.is_complete(written), written
    with pytest.raises(P.ProgramError):  # a ProgramError, not Python's recursion limit
        P.parse("Cut(" * 3000 + "All()" + ")" * 3000)


def test_type_errors_are_rejected_in_text_and_in_code():
    for written in TYPE_ERRORS:
        with pytest.raises(P.ProgramError):
            P.parse(written)
        assert not P.is_complete(written), written
    for build in (
        lambda: P.Cut(P.FlowerIs("red")),
        lambda: P.If(P.All(), P.Cut(P.All()), P.Cut(P.All())),
        lambda: P.If(P.FlowerIs("red"), P.All(), P.Cut(P.All())),
        lambda: P.Except(P.All(), "highest"),
        lambda: P.Rank(4, "top"),
        lambda: P.Rank(True, "top"),
        lambda: P.Rank(2.0, "top"),
        lambda: P.Rank("2", "top"),
        lambda: P.Filter("tie", "green", P.All()),
        lambda: P.FlowerIs("blue"),
    ):
        with pytest.raises(P.ProgramError):
            build()


def test_programs_beyond_the_depth_bounds_are_rejected():
    for written in TOO_DEEP:
        with pytest.raises(P.ProgramError):
            P.parse(written)
        assert not P.is_complete(written), written
    two_deep = P.Except(P.Except(P.All(), P.Middle()), P.Middle())
    # A node holds to v3's bound (three deep: tutorial/architecture-v3.md), and parse and the acceptor to v2's by
    # default (two deep)
    for too_deep in (P.Except(two_deep, P.Middle()), P.Filter("bud", "no", two_deep)):
        with pytest.raises(P.ProgramError):
            P.parse(P.text(P.Cut(too_deep)))
        assert not P.in_language(too_deep, "v2") and P.in_language(too_deep, "v3")
    three_deep = P.Except(two_deep, P.Middle())
    with pytest.raises(P.ProgramError):
        P.Except(three_deep, P.Middle())
    with pytest.raises(P.ProgramError):
        P.Filter("bud", "no", three_deep)
    with pytest.raises(P.ProgramError):
        P.If(P.FlowerIs("red"), P.If(P.FlowerIs("pink"), P.Cut(P.All()), P.Cut(P.All())), P.Cut(P.All()))
    # two deep is within the bounds
    for written in (
        "Cut(Except(Except(All(),Middle()),Middle()))",
        "Cut(Filter(tie,red,Filter(tie,blue,All())))",
        "Cut(Except(Filter(bud,no,All()),Filter(sick,yes,ReadLabel())))",
        "Cut(Filter(sick,yes,Except(Rank(1,top),ReadLabel())))",
        "If(FlowerIs(purple),Cut(Except(All(),Except(All(),Middle()))),Cut(Filter(tie,white,Except(All(),All()))))",
    ):
        assert P.text(P.parse(written)) == written and P.is_complete(written)


# ---------------------------------------------------------------------------------------------------------- equivalence
def test_equivalent_programs_have_the_same_result_on_every_plant():
    def same(a, b):
        return P.equivalent(P.parse(a), P.parse(b))

    for kind in R.KINDS:
        for _, _, task in _draws(kind, "heldout", n=50):
            assert P.equivalent(P.from_task(task), P.from_task(task))
    # on a plant with two shoots the second from the bottom is the highest, and there is no middle one
    assert not P.equivalent(P.Rank(2, "bottom"), P.Middle())
    # on a plant with three, every shoot but the highest includes the middle one
    assert not P.equivalent(P.Except(P.All(), P.Rank(1, "top")), P.Rank(1, "bottom"))
    # the third from the top is the lowest of three, and no shoot of two
    assert not same("Cut(Rank(3,top))", "Cut(Rank(1,bottom))")
    # needing the label is not the same as naming no shoot, and needing a tie not the same as needing a bud
    assert not same("Cut(ReadLabel())", "Cut(Middle())")
    assert not same("Cut(Filter(tie,red,All()))", "Cut(Filter(bud,no,All()))")
    # two spellings of one meaning: the complement of the complement, and a condition whose branches agree
    assert same("Cut(Rank(1,bottom))", "Cut(Except(All(),Except(All(),Rank(1,bottom))))")
    assert same("Cut(ReadLabel())", "Cut(Except(ReadLabel(),Except(All(),ReadLabel())))")
    assert same("If(FlowerIs(pink),Cut(Rank(1,top)),Cut(Rank(1,top)))", "Cut(Rank(1,top))")
    # the highest without the lowest is the highest on every plant here
    assert same("Cut(Except(Rank(1,top),Rank(1,bottom)))", "Cut(Rank(1,top))")
    # and the ordinal held-out compositions are lookups of training ones only where the count makes them so
    assert not same("Cut(Rank(2,bottom))", "Cut(Rank(2,top))")


def test_equivalent_agrees_with_the_results_compared_plant_by_plant():
    def result(node, count, flower, label):
        try:
            return P.targets_on(node, count, flower, label)
        except P.NeedsObservation as need:
            return need.what
        except P.ProgramError:
            return "error"

    plants = [(c, f, lab) for c in (2, 3) for f in R.FLOWERS for lab in (*R.label_options(c), None)]
    selections = _all_selections(1)  # 153 selections
    signatures = {s: tuple(result(s, *plant) for plant in plants) for s in selections}
    rng = random.Random(1)
    pairs = [(a, b) for a in selections[:40] for b in selections]
    for a, b in rng.sample(pairs, 1500):
        assert P.equivalent(a, b) == (signatures[a] == signatures[b])
    assert len(set(signatures.values())) < len(selections)  # some different spellings mean the same


# --------------------------------------------------------------------------------------------------------- the acceptor
def test_the_alphabet_is_the_characters_of_the_language():
    words = (
        "Cut If All Rank Middle Except Filter ReadLabel FlowerIs top bottom red orange pink purple "
        "tie blue yellow white sick yes bud no 123(),"
    )
    assert set(words.replace(" ", "")) == P.ALPHABET
    assert P.next_chars("") == {"C", "I"}
    assert P.next_chars("Cut(R") == {"a", "e"}  # Rank( or ReadLabel()
    assert P.next_chars("If(FlowerIs(p") == {"i", "u"}  # pink or purple
    assert P.next_chars("Cut(Rank(") == {"1", "2", "3"}
    assert P.next_chars("Cut(All())") == set() and P.next_chars("Cut(All()))") == set()


def test_every_prefix_of_a_program_allows_its_next_character():
    rng = random.Random(2)
    for _ in range(300):
        written = P.text(_random_program(rng))
        for i in range(len(written)):
            allowed = P.next_chars(written[:i])
            assert written[i] in allowed and allowed <= P.ALPHABET, (written, i)
            assert not P.is_complete(written[:i]), (written, i)  # no program begins another
        assert P.is_complete(written) and P.next_chars(written) == set()


def test_a_character_that_breaks_the_program_leaves_nothing_to_write():
    rng = random.Random(3)
    alphabet = sorted(P.ALPHABET)
    broken = 0
    for _ in range(600):
        written = P.text(_random_program(rng))
        i = rng.randrange(len(written))
        char = rng.choice(alphabet)
        if char in P.next_chars(written[:i]):
            continue
        broken += 1
        corrupted = written[:i] + char + written[i + 1 :]
        for end in range(i + 1, len(corrupted) + 1):
            assert P.next_chars(corrupted[:end]) == set() and not P.is_complete(corrupted[:end]), (corrupted, end)
        with pytest.raises(P.ProgramError):
            P.parse(corrupted)
    assert broken > 300


def test_any_walk_through_the_allowed_characters_ends_in_a_whole_program():
    rng = random.Random(4)
    kinds = set()
    for _ in range(400):
        written = ""
        while allowed := P.next_chars(written):
            assert allowed <= P.ALPHABET
            written += rng.choice(sorted(allowed))
        assert P.is_complete(written), written
        program = P.parse(written)
        assert P.text(program) == written
        kinds.add(type(program))
    assert kinds == {P.Cut, P.If}


def test_the_acceptor_and_the_parser_agree():
    rng = random.Random(5)
    agreed = {True: 0, False: 0}
    for _ in range(3000):
        corrupted = _corrupt(P.text(_random_program(rng)), rng)
        parses = _parses(corrupted)
        assert P.is_complete(corrupted) == parses, corrupted
        if parses:
            assert P.text(P.parse(corrupted)) == corrupted
        agreed[parses] += 1
    assert min(agreed.values()) > 100  # both kinds of string were tried


def test_every_cut_program_is_accepted():
    selections = _all_selections(2)
    assert len(selections) == 24_489  # the Cut programs of the language
    for sel in selections:
        written = P.text(P.Cut(sel))
        assert P.is_complete(written), written
    texts = {P.text(sel) for sel in selections}
    assert len(texts) == len(selections)  # each has one spelling


def test_the_acceptor_matches_a_regular_expression_of_the_grammar():
    """An independent check of exactness: the language is finite, so it is a regular expression, here written out from
    the grammar by hand. `regex` matches prefixes (partial=True), which `re` cannot."""
    regex = pytest.importorskip("regex")
    leaf = r"(?:All\(\)|Rank\([123],(?:top|bottom)\)|Middle\(\)|ReadLabel\(\))"
    pair = r"(?:tie,(?:red|blue|yellow|white)|sick,yes|bud,(?:yes|no))"

    def nest(inner):
        return rf"(?:{leaf}|Except\({inner},{inner}\)|Filter\({pair},{inner}\))"

    cut = rf"Cut\({nest(nest(leaf))}\)"
    language = regex.compile(rf"(?:{cut}|If\(FlowerIs\((?:red|orange|pink|purple)\),{cut},{cut}\))")
    rng = random.Random(6)
    tried = 0
    for _ in range(150):
        program = P.text(_random_program(rng))
        corrupted = _corrupt(program, rng)
        for written in {program[: rng.randrange(len(program) + 1)], corrupted, corrupted[: rng.randrange(40)]}:
            expected = {c for c in P.ALPHABET if language.fullmatch(written + c, partial=True)}
            assert P.next_chars(written) == expected, written
            assert P.is_complete(written) == bool(language.fullmatch(written)), written
            tried += 1
    assert tried > 300


# ------------------------------------------------------------------------------------------------------------ the plan
@dataclass
class Part:
    id: int
    height: float
    cut: bool = False
    bearing: float | None = 0.0
    attributes: dict = field(default_factory=dict)


@dataclass
class Memory:
    """What a plan reads of a scene memory (scene.py's SceneMemory): the inventory, lowest first, and the facts."""

    parts: list = field(default_factory=list)
    settled: bool = True
    flower: str | None = "red"
    label_read: str | None = None

    def discover(self, height, bearing=0.0):
        """A new part: the next id; the inventory stays lowest first."""
        part = Part(len(self.parts), height, bearing=bearing)
        self.parts = sorted([*self.parts, part], key=lambda p: p.height)
        return part

    def position(self, id):
        index = next(i for i, part in enumerate(self.parts) if part.id == id)
        return R.position_name(index, len(self.parts))


def _memory(count, bearings=None, **facts):
    memory = Memory(**facts)
    for i in range(count):
        memory.discover(0.08 + 0.03 * i, bearing=0.0 if bearings is None else bearings[i])
    return memory


def test_the_plan_holds_until_the_plant_is_settled():
    memory = _memory(3, settled=False)
    plan = P.Plan(P.parse("Cut(Rank(1,top))"))
    assert plan.step(memory) == P.NeedObservation("plant", None) and plan.targets is None
    memory.settled = True
    assert plan.step(memory) == P.Execute("cut the highest shoot", 2) and plan.targets == [2]


def test_the_label_task_shows_the_label_until_it_is_read_and_never_completes_before():
    memory = _memory(3)
    plan = P.Plan(P.parse("Cut(ReadLabel())"))
    assert plan.step(memory) == P.NeedObservation("label", "show the label")
    for part in memory.parts:  # every shoot happens to be cut: nothing says which one was meant, so not done
        part.cut = True
    assert plan.step(memory) == P.NeedObservation("label", "show the label")
    for part in memory.parts:
        part.cut = False
    memory.label_read = "cyan"
    assert plan.step(memory) == P.Execute("cut the highest shoot", 2)
    memory.parts[2].cut = True
    assert plan.step(memory) == P.Complete()
    memory.parts[0].cut = True
    assert plan.step(memory) == P.Failed("a shoot that was to stay is cut")


def test_a_label_rule_says_what_each_colour_names():
    memory = _memory(3, label_read="white")
    plan = P.Plan(P.parse("Cut(ReadLabel())"), label_rule={"white": "highest", "cyan": "lowest"})
    assert plan.step(memory) == P.Execute("cut the highest shoot", 2)
    memory.label_read = "yellow"  # a colour the rule does not have names no shoot
    assert isinstance(P.Plan(P.parse("Cut(ReadLabel())"), {"white": "lowest"}).step(memory), P.Failed)
    with pytest.raises(ValueError):
        P.Plan(P.parse("Cut(ReadLabel())"), label_rule={"white": "top"})


def test_cutting_a_shoot_that_was_to_stay_fails():
    memory = _memory(3)
    plan = P.Plan(P.parse("Cut(Except(All(),Middle()))"))
    assert plan.step(memory) == P.Execute("cut the lowest shoot", 0)
    memory.parts[1].cut = True
    assert plan.step(memory) == P.Failed("a shoot that was to stay is cut")


def test_a_shoot_the_plant_does_not_have_fails():
    for written in ("Cut(Rank(3,top))", "Cut(Middle())", "If(FlowerIs(red),Cut(Middle()),Cut(Rank(1,top)))"):
        outcome = P.Plan(P.parse(written)).step(_memory(2))
        assert isinstance(outcome, P.Failed) and "on a plant with 2" in outcome.reason, (written, outcome)
    # yellow names the middle shoot, which a plant with two does not have
    assert isinstance(P.Plan(P.parse("Cut(ReadLabel())")).step(_memory(2, label_read="yellow")), P.Failed)


def test_the_target_that_needs_the_least_turning_comes_first():
    memory = _memory(3, bearings=[1.2, -0.4, 0.0])
    plan = P.Plan(P.parse("Cut(Except(All(),Rank(1,top)))"))  # the lowest and the middle
    assert plan.step(memory) == P.Execute("cut the middle shoot", 1)
    memory.parts[1].cut = True
    assert plan.step(memory) == P.Execute("cut the lowest shoot", 0)
    memory.parts[0].cut = True
    assert plan.step(memory) == P.Complete()
    # ties go to the lowest; a bearing not known counts as half a turn (pi)
    for bearings, first in (
        ([0.7, -0.7, 0.0], 0),
        ([-0.7, 0.7, 0.0], 0),
        ([None, 3.0, 0.0], 1),
        ([None, None, 0.0], 0),
        ([None, -math.pi, 0.0], 0),
        ([2.0, None, 0.0], 0),
    ):
        outcome = P.Plan(P.parse("Cut(Except(All(),Rank(1,top)))")).step(_memory(3, bearings=bearings))
        assert outcome.part == first, (bearings, outcome)


def test_positions_refer_to_the_whole_inventory():
    memory = _memory(3)
    plan = P.Plan(P.parse("Cut(Rank(2,top))"))
    assert plan.step(memory) == P.Execute("cut the middle shoot", 1)
    memory.parts[1].cut = True
    # of the two shoots left the lowest is now second from the top: the plan does not move on to it
    assert plan.step(memory) == P.Complete() and plan.targets == [1]


def test_the_plan_binds_again_when_the_inventory_grows():
    memory = _memory(2)  # ids 0 (lowest) and 1 (highest)
    plan = P.Plan(P.parse("Cut(Rank(2,top))"))
    assert plan.step(memory) == P.Execute("cut the lowest shoot", 0) and plan.rebinds == 0
    memory.discover(0.095)  # a third shoot comes into view between the two: id 2, the middle one
    assert plan.step(memory) == P.Execute("cut the middle shoot", 2)
    assert plan.targets == [2] and plan.rebinds == 1
    assert plan.step(memory) == P.Execute("cut the middle shoot", 2) and plan.rebinds == 1  # no growth, no rebinding
    # binding again cannot undo a cut: one made on the plant as first seen is now a shoot that was to stay
    memory = _memory(2)
    plan = P.Plan(P.parse("Cut(Rank(2,top))"))
    plan.step(memory)
    memory.parts[0].cut = True
    memory.discover(0.095)
    assert plan.step(memory) == P.Failed("a shoot that was to stay is cut") and plan.rebinds == 1


def test_a_corrected_flower_binds_a_conditional_plan_again():
    memory = _memory(3, flower="red")
    plan = P.Plan(P.parse("If(FlowerIs(red),Cut(Rank(1,bottom)),Cut(Rank(1,top)))"))
    assert plan.step(memory) == P.Execute("cut the lowest shoot", 0) and plan.rebinds == 0
    memory.flower = "orange"  # the memory corrects a first wrong reading
    assert plan.step(memory) == P.Execute("cut the highest shoot", 2)
    assert plan.targets == [2] and plan.rebinds == 1
    assert plan.step(memory) == P.Execute("cut the highest shoot", 2) and plan.rebinds == 1  # nothing changed
    # a colour that takes the same branch still differs from the one read: bound again, to the same shoot
    memory.flower = "pink"
    assert plan.step(memory) == P.Execute("cut the highest shoot", 2) and plan.rebinds == 2


def test_a_cut_made_under_a_reading_since_corrected_fails():
    memory = _memory(3, flower="red")
    plan = P.Plan(P.parse("If(FlowerIs(red),Cut(Rank(1,bottom)),Cut(Rank(1,top)))"))
    assert plan.step(memory) == P.Execute("cut the lowest shoot", 0)
    memory.parts[0].cut = True
    assert plan.step(memory) == P.Complete()
    memory.flower = "orange"
    assert plan.step(memory) == P.Failed("a shoot that was to stay is cut")
    assert plan.targets == [2] and plan.rebinds == 1


def test_a_corrected_label_binds_the_label_plan_again():
    memory = _memory(3, label_read="white")
    plan = P.Plan(P.parse("Cut(ReadLabel())"))
    assert plan.step(memory) == P.Execute("cut the lowest shoot", 0) and plan.rebinds == 0
    memory.label_read = "cyan"
    assert plan.step(memory) == P.Execute("cut the highest shoot", 2)
    assert plan.targets == [2] and plan.rebinds == 1


def test_a_fact_the_evaluation_did_not_read_never_binds_the_plan_again():
    memory = _memory(3, flower="red", label_read="white")
    ordinal = P.Plan(P.parse("Cut(Rank(2,top))"))  # reads neither the flower nor the card
    card = P.Plan(P.parse("Cut(ReadLabel())"))  # reads the card, not the flower
    branch = P.Plan(P.parse("If(FlowerIs(red),Cut(Middle()),Cut(ReadLabel()))"))  # the card only if not red
    assert ordinal.step(memory) == P.Execute("cut the middle shoot", 1)
    assert card.step(memory) == P.Execute("cut the lowest shoot", 0)
    assert branch.step(memory) == P.Execute("cut the middle shoot", 1)
    memory.label_read = "cyan"
    memory.parts[0].attributes["tie"] = "red"
    assert ordinal.step(memory) == P.Execute("cut the middle shoot", 1) and ordinal.rebinds == 0
    assert branch.step(memory) == P.Execute("cut the middle shoot", 1) and branch.rebinds == 0  # took the red branch
    assert card.step(memory) == P.Execute("cut the highest shoot", 2) and card.rebinds == 1
    memory.flower = "orange"
    assert ordinal.step(memory) == P.Execute("cut the middle shoot", 1) and ordinal.rebinds == 0
    assert card.step(memory) == P.Execute("cut the highest shoot", 2) and card.rebinds == 1
    assert branch.step(memory) == P.Execute("cut the highest shoot", 2) and branch.rebinds == 1  # now reads the card
    memory.label_read = "white"  # and from now on watches it
    assert branch.step(memory) == P.Execute("cut the lowest shoot", 0) and branch.rebinds == 2
    # the inventory is always read: a shoot found later binds the plan again, where a new flower colour did not
    memory = _memory(2, flower="red")
    ordinal = P.Plan(P.parse("Cut(Rank(2,top))"))
    assert ordinal.step(memory) == P.Execute("cut the lowest shoot", 0)
    memory.flower = "orange"
    assert ordinal.step(memory) == P.Execute("cut the lowest shoot", 0) and ordinal.rebinds == 0
    memory.discover(0.095)
    assert ordinal.step(memory) == P.Execute("cut the middle shoot", 2) and ordinal.rebinds == 1


def test_a_filter_binds_again_only_when_a_shoot_it_looked_at_changes():
    memory = _memory(3)
    for part, tie in zip(memory.parts, ("blue", "red", "red"), strict=True):
        part.attributes["tie"] = tie
    plan = P.Plan(P.parse("Cut(Filter(tie,red,Except(All(),Rank(1,bottom))))"))  # looks at the middle and the highest
    assert plan.step(memory) == P.Execute("cut the middle shoot", 1) and plan.targets == [1, 2]
    memory.parts[0].attributes["tie"] = "red"  # the lowest: not looked at
    memory.parts[1].attributes["bud"] = "yes"  # another attribute: not read
    assert plan.step(memory) == P.Execute("cut the middle shoot", 1) and plan.rebinds == 0
    memory.parts[1].attributes["tie"] = "white"  # the middle one's tie was read wrong
    assert plan.step(memory) == P.Execute("cut the highest shoot", 2)
    assert plan.targets == [2] and plan.rebinds == 1


def test_binding_again_that_needs_a_fact_holds_and_keeps_the_old_targets():
    memory = _memory(3, flower="orange")
    plan = P.Plan(P.parse("If(FlowerIs(red),Cut(ReadLabel()),Cut(Rank(1,top)))"))
    assert plan.step(memory) == P.Execute("cut the highest shoot", 2)
    memory.flower = "red"  # corrected: the card decides now, and it has not been read
    assert plan.step(memory) == P.NeedObservation("label", "show the label")
    assert plan.targets == [2] and plan.rebinds == 0
    memory.label_read = "white"
    assert plan.step(memory) == P.Execute("cut the lowest shoot", 0)
    assert plan.targets == [0] and plan.rebinds == 1


def test_an_unknown_flower_holds_a_conditional_plan():
    memory = _memory(3, flower=None)
    plan = P.Plan(P.parse("If(FlowerIs(red),Cut(Rank(1,bottom)),Cut(Rank(1,top)))"))
    assert plan.step(memory) == P.NeedObservation("flower", None)
    memory.parts[0].cut = memory.parts[2].cut = True  # never Complete while the flower decides what was meant
    assert plan.step(memory) == P.NeedObservation("flower", None)
    memory.parts[0].cut = memory.parts[2].cut = False
    memory.flower = "orange"
    assert plan.step(memory) == P.Execute("cut the highest shoot", 2)


def test_a_filter_needs_its_attribute_seen_on_every_shoot_it_looks_at():
    memory = _memory(3)
    plan = P.Plan(P.parse("Cut(Filter(tie,red,All()))"))
    assert plan.step(memory) == P.NeedObservation("tie", None)
    memory.parts[0].attributes["tie"] = "blue"
    memory.parts[2].attributes["tie"] = "red"
    assert plan.step(memory) == P.NeedObservation("tie", None)  # the middle one's tie is not seen yet
    assert P.Plan(P.parse("Cut(Filter(tie,red,Rank(1,top)))")).step(memory) == P.Execute("cut the highest shoot", 2)
    memory.parts[1].attributes["tie"] = "yellow"
    assert plan.step(memory) == P.Execute("cut the highest shoot", 2)
    nothing = P.Plan(P.parse("Cut(Filter(tie,white,All()))"))  # no shoot has a white tie: nothing to do
    assert nothing.step(memory) == P.Complete() and nothing.targets == []


@pytest.mark.parametrize("kind", R.KINDS)
def test_a_plan_carried_out_cuts_exactly_the_shoots_the_task_means(kind):
    rng = random.Random(7)
    for split in ("train", "heldout"):
        for count, flower, task in _draws(kind, split, n=150):
            bearings = [rng.uniform(-math.pi, math.pi) for _ in range(count)]
            memory = _memory(count, bearings=bearings, flower=flower)
            plan = P.Plan(P.from_task(task))
            executed = 0
            for _ in range(10):
                outcome = plan.step(memory)
                if isinstance(outcome, P.NeedObservation):
                    assert kind == "hidden" and outcome == P.NeedObservation("label", "show the label")
                    memory.label_read = task.params["label"]  # the robot turned the plant and read the card
                elif isinstance(outcome, P.Execute):
                    part = next(p for p in memory.parts if p.id == outcome.part)
                    assert not part.cut and outcome.subgoal == f"cut the {memory.position(part.id)} shoot"
                    part.cut = True  # the fast level cut it and the memory saw it
                    executed += 1
                else:
                    break
            assert outcome == P.Complete(), (task, outcome)
            assert tuple(memory.position(p.id) for p in memory.parts if p.cut) == task.target_positions
            assert executed == len(task.targets) and plan.rebinds == 0


# ================================================================================================= v3's language
# tutorial/architecture-v3.md, "The program language": CutFrom, Then, Nothing, Take, Count, and the pairs tie,any,
# sick,no and tag,<colour>. v2's language stays the default of parse and of the acceptor (the tests above).
ATTRIBUTE_VALUES_V3 = (
    *ATTRIBUTE_VALUES,
    ("tie", "any"),
    ("sick", "no"),
    *(("tag", colour) for colour in ("red", "blue", "yellow", "white")),
)


def _random_selection_v3(rng, depth):
    roll = rng.random()
    if depth > 0 and roll < 0.2:
        return P.Except(_random_selection_v3(rng, depth - 1), _random_selection_v3(rng, depth - 1))
    if depth > 0 and roll < 0.35:
        return P.Filter(*rng.choice(ATTRIBUTE_VALUES_V3), _random_selection_v3(rng, depth - 1))
    if depth > 0 and roll < 0.5:
        return P.Take(rng.choice((1, 2, 3)), rng.choice(("top", "bottom", "any")), _random_selection_v3(rng, depth - 1))
    return rng.choice(LEAVES)


def _random_basic(rng):
    roll = rng.random()
    if roll < 0.45:
        return P.Cut(_random_selection_v3(rng, 3))
    if roll < 0.85:
        return P.CutFrom(rng.choice(("top", "bottom")), _random_selection_v3(rng, 3))
    return P.Nothing()


def _chain(rng, make):
    """One statement, or a Then of two or three, written to the right."""
    statements = [make(rng) for _ in range(rng.choice((1, 1, 2, 3)))]
    out = statements[-1]
    for statement in reversed(statements[:-1]):
        out = P.Then(statement, out)
    return out


def _random_atom(rng):
    if rng.random() < 0.6:
        return _random_basic(rng)
    if rng.random() < 0.5:
        cond = P.FlowerIs(rng.choice(R.FLOWERS))
    else:
        cond = P.Count(rng.choice(("exactly", "more", "fewer")), rng.choice((0, 1, 2, 3)), _random_selection_v3(rng, 3))
    return P.If(cond, _chain(rng, _random_basic), _chain(rng, _random_basic))


def _random_program_v3(rng):
    return _chain(rng, _random_atom)


def _parses_v3(written):
    try:
        P.parse(written, "v3")
    except P.ProgramError:
        return False
    return True


def test_v3_text_parses_back_to_the_same_program_and_the_acceptor_agrees_with_the_parser():
    rng = random.Random(10)
    agreed = {True: 0, False: 0}
    for _ in range(10_000):
        program = _random_program_v3(rng)
        written = P.text(program)
        assert P.parse(written, "v3") == program and P.is_complete(written, "v3"), written
        assert P.next_chars(written, "v3") == set()
        corrupted = _corrupt(written, rng)
        parses = _parses_v3(corrupted)
        assert P.is_complete(corrupted, "v3") == parses, corrupted
        if parses:
            assert P.text(P.parse(corrupted, "v3")) == corrupted
        agreed[parses] += 1
    assert min(agreed.values()) > 300  # both kinds of string were tried


def test_v3_every_prefix_allows_its_next_character_and_any_walk_ends_in_a_whole_program():
    rng = random.Random(11)
    alphabet = P.ALPHABETS["v3"]
    for _ in range(300):
        written = P.text(_random_program_v3(rng))
        for i in range(len(written)):
            allowed = P.next_chars(written[:i], "v3")
            assert written[i] in allowed and allowed <= alphabet, (written, i)
            assert not P.is_complete(written[:i], "v3"), (written, i)
    kinds = set()
    for _ in range(400):
        written = ""
        while allowed := P.next_chars(written, "v3"):
            written += rng.choice(sorted(allowed))
        program = P.parse(written, "v3")
        assert P.text(program) == written
        kinds.add(type(program))
    assert kinds == {P.Cut, P.If, P.CutFrom, P.Then, P.Nothing}


def test_v3_is_v2_and_more_and_v2_stays_the_default():
    rng = random.Random(12)
    for _ in range(500):
        written = P.text(_random_program(rng))
        assert P.parse(written, "v3") == P.parse(written) and P.is_complete(written, "v3")
    assert P.ALPHABETS["v3"] > P.ALPHABET and P.ALPHABETS["v2"] == P.ALPHABET
    assert P.next_chars("", "v3") == {"C", "I", "T", "N"} and P.next_chars("") == {"C", "I"}
    for written in (
        "CutFrom(top,All())",
        "Then(Cut(Rank(1,bottom)),Cut(Rank(1,top)))",
        "Nothing()",
        "Cut(Take(1,top,Filter(bud,no,All())))",
        "If(Count(more,1,Filter(sick,yes,All())),Cut(Take(1,bottom,Filter(sick,yes,All()))),Nothing())",
        "Cut(Filter(sick,no,All()))",
        "Cut(Filter(tie,any,All()))",
        "Cut(Filter(tag,red,All()))",
        "Cut(Except(Except(Except(All(),Middle()),Middle()),Middle()))",
        "If(FlowerIs(red),CutFrom(top,All()),Cut(All()))",
    ):
        program = P.parse(written, "v3")
        assert P.text(program) == written and P.is_complete(written, "v3")
        assert not P.is_complete(written) and not P.in_language(program, "v2")
        with pytest.raises(P.ProgramError):
            P.parse(written)


def test_v3_rules_keep_the_language_finite():
    for written in (
        "Then(Then(Cut(All()),Cut(All())),Cut(All()))",  # a Then nests to the right
        "Then(Cut(All()),Then(Cut(All()),Then(Cut(All()),Cut(All()))))",  # four statements
        "If(FlowerIs(red),If(FlowerIs(pink),Cut(All()),Nothing()),Nothing())",  # an If in an If
        "If(FlowerIs(red),Then(Cut(All()),If(FlowerIs(pink),Cut(All()),Nothing())),Nothing())",
        "Cut(Except(Except(Except(Except(All(),Middle()),Middle()),Middle()),Middle()))",  # four deep
        "Cut(Take(1,top,Take(1,top,Take(1,top,Take(1,top,All())))))",
        "Cut(Take(0,top,All()))",
        "Cut(Take(4,top,All()))",
        "Cut(Take(1,middle,All()))",
        "If(Count(exactly,4,All()),Nothing(),Nothing())",
        "If(Count(most,1,All()),Nothing(),Nothing())",
        "Cut(Count(more,1,All()))",
        "CutFrom(any,All())",
        "Cut(Filter(tag,any,All()))",
        "Cut(Filter(sick,maybe,All()))",
        "Then(Cut(All()))",
        "Nothing(All())",
    ):
        with pytest.raises(P.ProgramError):
            P.parse(written, "v3")
        assert not P.is_complete(written, "v3"), written
    # a Then of three, an If inside a Then, and a Then inside an If are all in the language
    for written in (
        "Then(Cut(Rank(1,top)),Then(Cut(Rank(1,bottom)),Nothing()))",
        "Then(If(FlowerIs(red),Cut(All()),Nothing()),CutFrom(bottom,All()))",
        "If(FlowerIs(red),Then(Cut(Middle()),CutFrom(top,All())),Nothing())",
    ):
        assert P.text(P.parse(written, "v3")) == written and P.is_complete(written, "v3")


def _stages(written, count, **facts):
    return P.stages_on(P.parse(written, "v3"), count, **facts)


def test_v3_stages_of_the_situations_programs_and_of_rung_5():
    assert _stages("CutFrom(top,All())", 3) == [(("highest",), ("middle",), ("lowest",))]
    assert _stages("CutFrom(bottom,All())", 2) == [(("lowest",), ("highest",))]
    assert _stages("Then(Cut(Rank(1,bottom)),Cut(Rank(1,top)))", 3) == [(("lowest",), ("highest",))]
    assert _stages("Cut(All())", 3) == [(("lowest", "middle", "highest"),)]
    ties = [{"tie": "none"}, {"tie": "red"}, {"tie": "none"}]
    assert _stages("Cut(Filter(tie,any,All()))", 3, attributes=ties) == [(("middle",),)]
    assert _stages("Cut(Filter(tie,red,All()))", 3, attributes=ties) == [(("middle",),)]
    assert _stages("Cut(Filter(tie,blue,All()))", 3, attributes=ties) == [()]  # nothing to cut
    sick = [{"sick": "yes", "bud": "no"}, {"sick": "no", "bud": "yes"}, {"sick": "yes", "bud": "no"}]
    assert _stages("Cut(Filter(sick,yes,All()))", 3, attributes=sick) == [(("lowest", "highest"),)]
    assert _stages("Cut(Filter(sick,no,All()))", 3, attributes=sick) == [(("middle",),)]
    assert _stages("CutFrom(top,Filter(sick,yes,All()))", 3, attributes=sick) == [(("highest",), ("lowest",))]
    assert _stages("Cut(Take(1,top,Filter(bud,no,All())))", 3, attributes=sick) == [(("highest",),)]
    assert _stages("Cut(Take(1,bottom,Filter(bud,no,All())))", 3, attributes=sick) == [(("lowest",),)]
    assert _stages("Nothing()", 3) == [()]
    # any two shoots: every choice is listed, and "leave exactly one" keeps any one
    assert _stages("Cut(Take(2,any,All()))", 3) == [
        (("lowest", "middle"),),
        (("lowest", "highest"),),
        (("middle", "highest"),),
    ]
    assert sorted(_stages("Cut(Except(All(),Take(1,any,All())))", 2)) == [(("highest",),), (("lowest",),)]
    assert _stages("Cut(Take(3,any,All()))", 2) == [(("lowest", "highest"),)]  # all of it if it has fewer
    counted = "If(Count(more,1,Filter(sick,yes,All())),Cut(Take(1,bottom,Filter(sick,yes,All()))),Nothing())"
    assert _stages(counted, 3, attributes=sick) == [(("lowest",),)]
    one_sick = [{"sick": "no"}, {"sick": "yes"}, {"sick": "no"}]
    assert _stages(counted, 3, attributes=one_sick) == [()]
    # a Then does not ask again for a shoot its first statement cut
    assert _stages("Then(Cut(Rank(1,top)),Cut(All()))", 3) == [(("highest",), ("lowest", "middle"))]
    assert _stages("Then(Cut(All()),CutFrom(top,All()))", 2) == [(("lowest", "highest"),)]
    # facts are read as in v2: a Filter needs its attribute on every shoot it looks at, and says which
    with pytest.raises(P.NeedsObservation) as need:
        _stages("Cut(Filter(bud,no,All()))", 3, attributes=[{"bud": "no"}, {}, {}])
    assert (need.value.what, need.value.index) == ("bud", 1)
    with pytest.raises(P.ProgramError):
        _stages("CutFrom(top,Middle())", 2)


def test_v3_equivalence_compares_the_sequences_of_cuts_on_plants_with_attributes():
    def same(a, b):
        return P.equivalent(P.parse(a, "v3"), P.parse(b, "v3"), "v3")

    assert not same("CutFrom(top,All())", "Cut(All())")
    # with no language named, v3's comparison for programs outside v2's language
    assert not P.equivalent(P.parse("CutFrom(top,All())", "v3"), P.parse("Cut(All())"))
    assert not same("CutFrom(top,All())", "CutFrom(bottom,All())")
    assert same("CutFrom(top,Rank(1,top))", "Cut(Rank(1,top))")  # one shoot: one order
    assert same("Then(Cut(Rank(1,bottom)),Cut(Rank(1,top)))", "Then(CutFrom(top,Rank(1,bottom)),Cut(Rank(1,top)))")
    assert not same("Then(Cut(Rank(1,bottom)),Cut(Rank(1,top)))", "CutFrom(bottom,Except(All(),Middle()))")
    assert same("Then(Cut(All()),Cut(Rank(1,top)))", "Cut(All())")
    assert same("Nothing()", "Cut(Except(All(),All()))")
    assert same("Cut(Filter(sick,no,All()))", "Cut(Except(All(),Filter(sick,yes,All())))")
    assert not same("Cut(Filter(tie,red,All()))", "Cut(Filter(tie,blue,All()))")  # v2's comparison has them equal
    assert P.equivalent(P.parse("Cut(Filter(tie,red,All()))"), P.parse("Cut(Filter(tie,blue,All()))"))
    assert not P.equivalent(P.parse("Cut(Filter(tie,red,All()))"), P.parse("Cut(Filter(tie,blue,All()))"), "v3")
    assert not same("Cut(Filter(tie,any,All()))", "Cut(Except(All(),Filter(tie,any,Except(All(),All()))))")
    assert not same("Cut(Take(1,top,Filter(bud,no,All())))", "Cut(Filter(bud,no,Rank(1,top)))")
    assert same("Cut(Take(1,top,All()))", "Cut(Rank(1,top))")
    assert same("Cut(Take(2,any,All()))", "Cut(Take(2,any,All()))")
    assert not same("Cut(Take(2,any,All()))", "Cut(Take(2,top,All()))")  # any allows more choices
    assert same(
        "If(Count(exactly,0,Filter(sick,yes,All())),Nothing(),Cut(Filter(sick,yes,All())))",
        "Cut(Filter(sick,yes,All()))",
    )
    assert not same("Cut(ReadLabel())", "Cut(Middle())")
    for kind in R.KINDS:  # v2's programs are equivalent to themselves either way
        for _, _, task in _draws(kind, "heldout", n=20):
            assert P.equivalent(P.from_task(task), P.from_task(task), "v3")


class _Scenario:
    def __init__(self, name, changes_task=True):
        self.name, self.changes_task = name, changes_task


class _Env:
    """What from_scenario reads of a PickupEnv."""

    def __init__(self, situation=None, **variation):
        self.task, self.reason, self.named = "prune", None, None
        self.scenarios = [] if situation is None else [_Scenario("wind", False), _Scenario(situation)]
        self.scenario_info = {} if situation is None else {situation: dict(applied=True, **variation), "wind": {}}


def test_the_program_of_each_situation_written_out():
    def written(situation=None, **variation):
        return P.text(P.from_scenario(_Env(situation, **variation)))

    assert written() == "Cut(All())"
    assert written("nothing_to_do", mode="pruned") == "Cut(All())"
    assert written("order", mode="top_down") == "CutFrom(top,All())"
    assert written("order_heldout", mode="bottom_up") == "CutFrom(bottom,All())"
    assert written("order", mode="pair", order="lowest,highest") == "Then(Cut(Rank(1,bottom)),Cut(Rank(1,top)))"
    assert written("order", mode="pair", order="middle,lowest") == "Then(Cut(Middle()),Cut(Rank(1,bottom)))"
    assert written("ties", mode="colour", colour="red") == "Cut(Filter(tie,red,All()))"
    assert written("ties_heldout", mode="one") == "Cut(Filter(tie,any,All()))"
    assert written("sick", mode="sick") == "Cut(Filter(sick,yes,All()))"
    assert written("buds", mode="buds") == "Cut(Filter(bud,no,All()))"
    env = _Env("nothing_to_do", mode="pruned")
    env.scenario_info["nothing_to_do"]["applied"] = False  # nothing_to_do@p in an episode it does not apply to
    assert P.text(P.from_scenario(env)) == "Cut(All())"
    env = _Env()
    env.named = "train"
    with pytest.raises(ValueError):
        P.from_scenario(env)
    env = _Env()
    env.reason, env.reason_task = (
        ("ordinal", "train"),
        R.Task("ordinal", "x", "train", 3, (1,), "x", dict(k=2, end="top")),
    )
    assert P.text(P.from_scenario(env)) == "Cut(Rank(2,top))"


# ================================================================================================ v3's plan (WorldPlan)
def _plan(written, memory, **kwargs):
    plan = P.WorldPlan(P.parse(written, "v3"), **kwargs)
    return plan, plan.step(memory)


def test_world_plan_keeps_an_order_and_fails_one_broken():
    memory = _memory(3)
    plan = P.WorldPlan(P.parse("CutFrom(top,All())", "v3"))
    for expected in (2, 1, 0):
        assert plan.step(memory) == P.Execute(f"cut the {memory.position(expected)} shoot", expected)
        memory.parts[expected].cut = True
    assert plan.step(memory) == P.Complete() and plan.stages == [[2], [1], [0]] and plan.targets == [0, 1, 2]
    memory = _memory(3, bearings=[0.0, 2.0, 2.0])  # the lowest needs no turning, and still waits for its turn
    plan = P.WorldPlan(P.parse("CutFrom(top,All())", "v3"))
    assert plan.step(memory) == P.Execute("cut the highest shoot", 2)
    memory.parts[0].cut = True  # the fast level cut the lowest first
    assert plan.step(memory) == P.Failed(P.OUT_OF_ORDER)
    # a pair: the lowest, then the highest; the middle stays
    memory = _memory(3)
    plan = P.WorldPlan(P.parse("Then(Cut(Rank(1,bottom)),Cut(Rank(1,top)))", "v3"))
    assert plan.step(memory) == P.Execute("cut the lowest shoot", 0)
    memory.parts[0].cut = True
    assert plan.step(memory) == P.Execute("cut the highest shoot", 2)
    memory.parts[1].cut = True
    assert plan.step(memory) == P.Failed(P.STAYED)


def test_world_plan_returns_to_a_stage_whose_cut_is_unmarked():
    memory = _memory(3)
    plan = P.WorldPlan(P.parse("CutFrom(bottom,All())", "v3"))
    assert plan.step(memory).part == 0
    memory.parts[0].cut = True
    assert plan.step(memory).part == 1
    memory.parts[0].cut = False  # the memory unmarks the cut (revoke): back to the first stage
    assert plan.step(memory) == P.Execute("cut the lowest shoot", 0)
    memory.parts[1].cut = True  # and the next stage's cut made meanwhile breaks the order
    assert plan.step(memory) == P.Failed(P.OUT_OF_ORDER)


def test_world_plan_on_a_plant_of_stubs_is_complete_with_nothing_to_do():
    memory = _memory(3)
    for part in memory.parts:
        part.cut = True
    assert _plan("Cut(All())", memory)[1] == P.Complete()
    healthy = _memory(2)
    for part in healthy.parts:
        part.attributes["sick"] = "no"
    assert _plan("Cut(Filter(sick,yes,All()))", healthy)[1] == P.Complete()
    assert _plan("Nothing()", healthy)[1] == P.Complete()


def test_world_plan_cuts_what_every_world_cuts_and_asks_only_for_what_decides():
    # the design's example: the lowest known sick, the middle unread, the highest healthy
    memory = _memory(3, bearings=[1.0, 0.0, 0.5])
    memory.parts[0].attributes["sick"] = "yes"
    memory.parts[2].attributes["sick"] = "no"
    plan, outcome = _plan("Cut(Filter(sick,yes,All()))", memory)
    assert outcome == P.Execute("cut the lowest shoot", 0) and plan.worlds == 2 and plan.stages is None
    memory.parts[0].cut = True
    assert plan.step(memory) == P.NeedObservation("sick", "show the middle shoot", 1)
    memory.parts[1].attributes["sick"] = "no"
    assert plan.step(memory) == P.Complete() and plan.stages == [[0]] and plan.worlds == 1
    # "the highest sick shoot": the two worlds cut different shoots, so it asks first
    memory = _memory(3)
    memory.parts[0].attributes["sick"] = "yes"
    memory.parts[2].attributes["sick"] = "no"
    assert _plan("Cut(Take(1,top,Filter(sick,yes,All())))", memory)[1] == P.NeedObservation(
        "sick", "show the middle shoot", 1
    )
    # a fact that cannot change what the program does is not asked for: the lowest's bud decides nothing here
    memory = _memory(3)
    memory.parts[1].attributes["bud"] = "no"
    memory.parts[2].attributes["bud"] = "yes"
    plan, outcome = _plan("Cut(Except(Filter(bud,no,Except(All(),Rank(1,bottom))),Rank(1,bottom)))", memory)
    assert outcome == P.Execute("cut the middle shoot", 1) and plan.worlds == 1
    # with the facts that decide unread, it asks for the one the evaluation read first
    memory = _memory(2)
    assert _plan("Cut(Filter(tie,red,All()))", memory)[1] == P.NeedObservation("tie", "show the lowest shoot", 0)
    # the flower and the card as in v2
    assert _plan("If(FlowerIs(red),Cut(Rank(1,top)),Cut(Rank(1,bottom)))", _memory(3, flower=None))[1] == (
        P.NeedObservation("flower", None)
    )
    assert _plan("Cut(ReadLabel())", _memory(3))[1] == P.NeedObservation("label", "show the label")
    # a condition whose branches agree needs no flower
    assert _plan("If(FlowerIs(red),Cut(Rank(1,top)),Cut(Rank(1,top)))", _memory(3, flower=None))[1].part == 2


def test_world_plan_never_takes_a_fact_it_cannot_get_for_a_no():
    memory = _memory(3)
    memory.parts[0].cut = True  # cut before its tie was read: its tie fell with it
    memory.parts[1].attributes["tie"] = "none"
    memory.parts[2].attributes["tie"] = "none"
    assert _plan("Cut(Filter(tie,red,All()))", memory)[1] == P.Failed("cannot see the tie of the lowest shoot")
    # one that can still be seen is asked for first
    memory.parts[2].attributes.pop("tie")
    assert _plan("Cut(Filter(tie,red,All()))", memory)[1] == P.NeedObservation("tie", "show the highest shoot", 2)


def test_world_plan_frozen_attributes_keep_a_cut_target_and_a_stub_read_as_no_tie_binds_nothing_again():
    memory = _memory(3)
    for part, tie in zip(memory.parts, ("blue", "red", "none"), strict=True):
        part.attributes["tie"] = tie
    plan = P.WorldPlan(P.parse("Cut(Filter(tie,red,All()))", "v3"))
    assert plan.step(memory) == P.Execute("cut the middle shoot", 1)
    memory.parts[1].cut = True  # the memory keeps its red tie (SceneMemory rule 8)
    assert plan.step(memory) == P.Complete() and plan.rebinds == 0
    memory.parts[1].attributes["tie"] = "none"  # without the rule: the stub read as "no tie"
    assert plan.step(memory) == P.Failed(P.STAYED) and plan.rebinds == 1


def test_world_plan_makes_the_choice_of_any_once_and_keeps_it():
    memory = _memory(3, bearings=[2.0, 0.1, -0.5])
    plan = P.WorldPlan(P.parse("Cut(Take(2,any,All()))", "v3"))
    assert plan.step(memory) == P.Execute("cut the middle shoot", 1)  # the two that need the least turning
    assert plan.targets == [1, 2]
    memory.parts[1].cut = True
    memory.parts[0].bearing, memory.parts[2].bearing = 0.0, 3.0  # the plate turned: the lowest now faces the robot
    assert plan.step(memory) == P.Execute("cut the highest shoot", 2)  # the choice stands: no new budget
    memory.parts[2].cut = True
    assert plan.step(memory) == P.Complete() and len(plan.choices) == 1
    # shoots to keep are those that need the most turning: "leave exactly one"
    memory = _memory(3, bearings=[2.0, 0.1, -0.5])
    plan = P.WorldPlan(P.parse("Cut(Except(All(),Take(1,any,All())))", "v3"))
    assert plan.step(memory) == P.Execute("cut the middle shoot", 1) and plan.targets == [1, 2]


def test_world_plan_holds_until_the_plant_is_settled_and_fails_a_shoot_the_plant_does_not_have():
    assert _plan("CutFrom(top,All())", _memory(3, settled=False))[1] == P.NeedObservation("plant", None)
    assert _plan("CutFrom(top,Middle())", _memory(2))[1] == P.Failed("no middle shoot on a plant with 2")
    with pytest.raises(ValueError):
        P.WorldPlan(P.parse("Cut(ReadLabel())"), label_rule={"white": "top"})


# ------------------------------------------------------------------------------------------------ the brute force
def _completions(program, memory, limit=4000):
    """Every assignment of the facts the program can read and the memory does not know, or None if more than
    `limit`."""
    reads_flower, reads_label, attrs = P._reads(program)
    facts = []
    if reads_flower and memory.flower is None:
        facts.append((("flower",), R.FLOWERS))
    if reads_label and memory.label_read is None:
        facts.append((("label",), tuple(R.LABEL_MEANS)))
    for part in memory.parts:
        for attr in sorted(attrs):
            if attr not in part.attributes:
                facts.append(((attr, part.id), P.DOMAINS[attr]))
    if math.prod(len(values) for _, values in facts) > limit:
        return None
    keys = [key for key, _ in facts]
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(v for _, v in facts))]


def _status_in(plan, memory, completion):
    """The world `completion` evaluated independently of WorldPlan's combination: failed, complete, or the parts of
    its open stage."""
    parts = list(memory.parts)
    attributes = [{**p.attributes, **{k[0]: v for k, v in completion.items() if k[1:] == (p.id,)}} for p in parts]
    plant = P._Shoots(
        len(parts),
        memory.flower or completion.get(("flower",)),
        memory.label_read or completion.get(("label",)),
        tuple(attributes),
        dict(R.LABEL_MEANS),
        plan._chooser(parts),
    )
    try:
        (stages,) = P._stage_lists(plan.program, plant)
    except P.ProgramError:
        return ("failed",)
    stages = [{parts[i].id for i in stage} for stage in stages]
    cut = {p.id for p in parts if p.cut}
    if cut - set().union(*stages):
        return ("failed",)
    for j, stage in enumerate(stages):
        if stage - cut:
            return ("failed",) if cut & set().union(*stages[j + 1 :]) else ("open", stage - cut)
    return ("complete",)


def test_world_plan_acts_only_on_what_every_completion_of_the_unknown_facts_agrees_on():
    rng = random.Random(20)
    checked = Counter()
    while sum(checked.values()) < 2000:
        program = _random_program_v3(rng)
        count = rng.choice((2, 3))
        memory = _memory(count, bearings=[rng.uniform(-3, 3) for _ in range(count)])
        memory.flower = None if rng.random() < 0.3 else rng.choice(R.FLOWERS)
        memory.label_read = None if rng.random() < 0.5 else rng.choice(R.label_options(count))
        for part in memory.parts:
            part.cut = rng.random() < 0.3
            for attr, values in P.DOMAINS.items():
                if rng.random() < 0.7:
                    part.attributes[attr] = rng.choice(values)
        completions = _completions(program, memory)
        if completions is None:
            continue
        plan = P.WorldPlan(program)
        outcome = plan.step(memory)
        statuses = [_status_in(plan, memory, c) for c in completions]
        if isinstance(outcome, P.Execute):
            assert all(s[0] == "open" and outcome.part in s[1] for s in statuses), (P.text(program), outcome)
        elif isinstance(outcome, P.Complete):
            assert all(s == ("complete",) for s in statuses), (P.text(program), outcome)
        elif isinstance(outcome, P.Failed) and not outcome.reason.startswith("cannot see"):
            assert all(s == ("failed",) for s in statuses), (P.text(program), outcome)
        else:  # a fact is asked for: the completions do not all agree on any of the three
            assert not all(s == ("complete",) for s in statuses) and not all(s == ("failed",) for s in statuses)
        checked[type(outcome).__name__] += 1
    assert min(checked.values()) > 50, checked  # every kind of outcome was met


# ------------------------------------------------------------------------------------------- v2's plan, the reference
@pytest.mark.parametrize("kind", R.KINDS)
def test_for_v2_s_programs_the_world_plan_decides_as_v2_s_plan_look_for_look(kind):
    """2000 scripted sequences in all: facts arrive late or are first read wrong, a shoot is found late, the plate
    turns, the fast level cuts the shoot it is told to (now and then another), and the memory unmarks a cut."""
    rng = random.Random(21 + R.KINDS.index(kind))
    for split in ("train", "heldout"):
        for count, flower, task in _draws(kind, split, n=250, seed=R.KINDS.index(kind)):
            program = P.from_task(task)
            v2, v3 = P.Plan(program), P.WorldPlan(program)
            late = count == 3 and rng.random() < 0.3  # the middle shoot is found after a few looks
            memory = _memory(2 if late else count, flower=None, settled=False)
            if late:
                memory.parts[1].height = 0.15
            wrong_flower = rng.choice([f for f in R.FLOWERS if f != flower])
            label = task.params.get("label")
            for look in range(14):
                if look == 1:
                    memory.settled = True
                if look == 2:
                    memory.flower = wrong_flower if rng.random() < 0.2 else flower
                if look == 4:
                    memory.flower = flower
                    if label is not None:
                        memory.label_read = label
                if late and look == 5:
                    memory.discover(0.115)
                for part in memory.parts:
                    part.bearing = rng.uniform(-math.pi, math.pi) if rng.random() < 0.5 else part.bearing
                a, b = v2.step(memory), v3.step(memory)
                assert a == b, (task, look, a, b)
                if isinstance(a, P.Execute) and rng.random() < 0.8:
                    victim = a.part if rng.random() < 0.9 else rng.choice(memory.parts).id
                    next(p for p in memory.parts if p.id == victim).cut = True
                elif rng.random() < 0.05:
                    cut = [p for p in memory.parts if p.cut]
                    if cut:
                        rng.choice(cut).cut = False
