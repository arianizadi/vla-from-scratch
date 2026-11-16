"""Typed instruction programs (sim/pickup/program.py): the program of every benchmark task selects the shoots the task
means, text and parse are inverses, the grammar acceptor for constrained decoding is exact, and a plan evaluated on a
scene memory says what to do next with stable part identities.

Pure Python: nothing here needs MuJoCo. The scene memory is a small fake with the fields a plan reads.
"""

import math
import random
import sys
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
    with pytest.raises(P.ProgramError):
        P.Except(two_deep, P.Middle())
    with pytest.raises(P.ProgramError):
        P.Filter("bud", "no", two_deep)
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
