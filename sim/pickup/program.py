"""Typed instruction programs: an instruction compiled into a small program, which is evaluated on the scene memory.

A reasoning task's instruction (reasoning.py) says which side shoots to cut in terms of the plant: "cut the second
shoot from the top" is the middle shoot of three and the lowest of two. A program is the instruction written in a
small typed language in which every word has one meaning, so that what it selects is computed, not recalled. The
programs of the four kinds of the benchmark (`from_task`):

  ordinal      Cut(Rank(2,top))                                          the second shoot from the top
  conditional  If(FlowerIs(red),Cut(Rank(1,bottom)),Cut(Rank(1,top)))   the lowest if the flower is red, else the
                                                                         highest
  exclusion    Cut(Except(All(),Rank(1,top)))                            every shoot but the highest
  hidden       Cut(ReadLabel())                                          the shoot the card in the pot names

The language (GRAMMAR; `text` writes a program in its canonical form, without spaces, and `parse` reads it back):

  stmt := Cut(sel) | If(cond,stmt,stmt)
  sel  := All() | Rank(k,end) | Middle() | Except(sel,sel) | Filter(attr,value,sel) | ReadLabel()
  cond := FlowerIs(colour)

with k 1, 2 or 3, end top or bottom, the four flower colours, and the attribute values of ATTRIBUTES (no task uses
those yet). A selection nests Except and Filter at most two deep and an If branch is a Cut, so the language is finite:
24,489 Cut programs and about 2.4 billion If programs. That is too many to list, so the acceptor for constrained
decoding (`next_chars`, `is_complete`) recognises prefixes with the grammar itself.

Compiled once, evaluated many times. Only reading the instruction needs a language model: it writes the program once,
held to the grammar while it writes. The rest is computed: counting from an end, taking a branch, the complement. A
combination no demonstration contains ("the third from the top") is then worked out like any other; the fine-tuned
reasoner of the first design learned that step as a table from thirty instructions and gets new combinations wrong
(tutorial/architecture-v2.md). What the program needs to know about the plant (the number of shoots, the flower's
colour, the card's colour) it reads from the scene memory (scene.py), which changes while the robot works: the card is
read, a third shoot comes into view, a shoot is cut. Evaluating the program again after each change costs nothing and
gives the same answer for the same facts; asking the model again would cost a forward pass and could give another.

Positions refer to the whole inventory. The scene memory keeps every side shoot it has seen, cut or not, lowest
first, and a program counts over all of them, as the expert's notes do: on a plant with three shoots "the second from
the top" is the middle one, and after the middle one is cut it is still the middle one, not the lower of the two that
are left. A plan (`Plan`) binds the program's selection to the memory's part ids, and binds it again only when a fact
that binding read has changed: the inventory has grown, or the memory has corrected a first wrong reading of the
flower's colour, the card's colour or an attribute the program looked at. A fact the evaluation did not read cannot
change what it selects, so it never makes a plan bind again.

At each step a plan says one of four things:

  Execute(subgoal, part)          cut this part: "cut the middle shoot", and the remembered part's id
  NeedObservation(what, subgoal)  a fact is missing: "plant" (the inventory is not settled yet), "flower", "label" or
                                  an attribute's name; subgoal "show the label" for the label, else None (hold)
  Complete()                      every shoot meant is cut: stop
  Failed(reason)                  the program names a shoot the plant does not have, or a shoot that was to stay is cut

v3's language (tutorial/architecture-v3.md, GRAMMAR_V3; parse, next_chars and is_complete with language "v3") adds
five constructs and three attribute pairs, and is v2's language and more:

  stmt := ... | CutFrom(end,sel) | Then(stmt,stmt) | Nothing()
  sel  := ... | Take(k,pick,sel)          pick: top, bottom, any
  cond := ... | Count(cmp,n,sel)          cmp: exactly, more, fewer; n 0 to 3
  attr,value: also tie,any, sick,no and tag,<colour>

Except, Filter and Take nest at most three deep, a Then holds at most three statements (written to the right), and an
If's branch is any statement with no If in it. A v3 program evaluates to stages, sets of shoots in the order in which
they are to be cut (`stages_on`); two programs are equivalent in v3 when they allow the same sequences of cuts on every
plant with attributes (`equivalent_v3`). The situations built on prune have programs of their own (`from_scenario`).
v2's language stays the default everywhere, so that v2's writer and its runs are what they were; a node holds to v3's
bounds, and `in_language` says whether a program is also one of v2.

This module is pure Python.
"""

import dataclasses
import functools
import itertools
import math
import re
from dataclasses import dataclass, field

from . import reasoning


# --------------------------------------------------------------------------------------------------------------- errors
class ProgramError(ValueError):
    """Text that is not a program, a program that breaks the language's rules, or a shoot the plant does not have."""


class NeedsObservation(Exception):
    """Evaluating a program needs a fact that is not known yet: `what` is "flower", "label" or an attribute's name;
    for an attribute, `index` is the shoot's (from the lowest up) where the evaluation says it (v3's evaluation)."""

    def __init__(self, what: str, index: int | None = None):
        super().__init__(f"the {what} is not known yet")
        self.what, self.index = what, index


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ProgramError(message)


# --------------------------------------------------------------------------------------------------------- the language
RANKS = tuple(reasoning.ORDINALS)  # (1, 2, 3): the k of Rank(k,end) and of Take(k,pick,sel)
ENDS = reasoning.ENDS  # ("top", "bottom")
COLOURS = reasoning.FLOWERS
# What a Filter can ask about a shoot, and the values it can name, in v2's language: no v2 task sets any. They are for
# the test situations with gardener's ties, sick shoots and buds (tutorial/test-situations.md).
ATTRIBUTES = {"tie": ("red", "blue", "yellow", "white"), "sick": ("yes",), "bud": ("yes", "no")}
_ATTRIBUTE_VALUES = tuple((attr, value) for attr, values in ATTRIBUTES.items() for value in values)
MAX_NESTING = 2  # how deep Except and Filter may nest in a selection: what keeps the language finite

# v3's language (tutorial/architecture-v3.md) adds five constructs and three attribute pairs. "tie,any" is a shoot with
# a tie of any colour; "tag" is the plate of the hidden-tag situation (rung 6, not built yet).
LANGUAGES = ("v2", "v3")
ATTRIBUTES_V3 = {
    "tie": ("red", "blue", "yellow", "white", "any"),
    "sick": ("yes", "no"),
    "bud": ("yes", "no"),
    "tag": ("red", "blue", "yellow", "white"),
}
_ATTRIBUTE_VALUES_V3 = tuple((attr, value) for attr, values in ATTRIBUTES_V3.items() for value in values)
# The values a shoot's attribute has: what the simulator's truth and the scene memory hold (scene.py). The eyes have one
# class more, "unreadable" (scene.UNREADABLE), which is never a value of the shoot.
DOMAINS = {
    "tie": ("none", "red", "blue", "yellow", "white"),
    "sick": ("no", "yes"),
    "bud": ("no", "yes"),
    "tag": ("none", "red", "blue", "yellow", "white"),
}
PICKS = ("top", "bottom", "any")  # Take: the k highest, the k lowest, or any k (the plan's choice)
COMPARISONS = ("exactly", "more", "fewer")  # Count: exactly n shoots, more than n, fewer than n
COUNTS = (0, 1, 2, 3)  # the n of Count
MAX_NESTING_V3 = 3  # how deep Except, Filter and Take may nest in a selection of v3
MAX_THEN = 3  # the statements a Then holds at most


@dataclass(frozen=True)
class All:
    """Every side shoot."""


@dataclass(frozen=True)
class Rank:
    """The k-th shoot counting from an end: Rank(1, "top") is the highest, Rank(1, "bottom") the lowest."""

    k: int
    end: str

    def __post_init__(self):
        k_ok = isinstance(self.k, int) and not isinstance(self.k, bool) and self.k in RANKS
        _require(k_ok, f"Rank counts 1, 2 or 3 shoots from an end, not {self.k!r}")
        _require(self.end in ENDS, f"Rank counts from the top or the bottom, not {self.end!r}")


@dataclass(frozen=True)
class Middle:
    """The middle shoot: the second of three. A plant with two has none."""


@dataclass(frozen=True)
class ReadLabel:
    """The shoot the card in the pot names by its colour (reasoning.LABEL_MEANS, or a plan's label_rule)."""


@dataclass(frozen=True)
class Except:
    """The shoots of `sel` without those of `without`."""

    sel: "Sel"
    without: "Sel"

    def __post_init__(self):
        for part in (self.sel, self.without):
            _require(isinstance(part, Sel), f"Except takes two selections, not {_shown(part)}")
        _check_nesting(self)


@dataclass(frozen=True)
class Filter:
    """The shoots of `sel` whose attribute `attr` has `value`: Filter("tie", "red", All()) is every shoot with a red
    tie. It needs the attribute seen on every shoot of `sel`."""

    attr: str
    value: str
    sel: "Sel"

    def __post_init__(self):
        pair_ok = (self.attr, self.value) in _ATTRIBUTE_VALUES_V3
        _require(pair_ok, f"Filter takes an attribute and a value of ATTRIBUTES_V3, not {self.attr!r}, {self.value!r}")
        _require(isinstance(self.sel, Sel), f"Filter takes a selection, not {_shown(self.sel)}")
        _check_nesting(self)


@dataclass(frozen=True)
class Take:
    """v3: k shoots of `sel`: the k highest (pick "top"), the k lowest ("bottom"), or any k ("any": the plan's choice,
    made once, WorldPlan); all of `sel` if it has fewer."""

    k: int
    pick: str
    sel: "Sel"

    def __post_init__(self):
        k_ok = isinstance(self.k, int) and not isinstance(self.k, bool) and self.k in RANKS
        _require(k_ok, f"Take takes 1, 2 or 3 shoots, not {self.k!r}")
        _require(self.pick in PICKS, f"Take picks {', '.join(PICKS)}, not {self.pick!r}")
        _require(isinstance(self.sel, Sel), f"Take takes a selection, not {_shown(self.sel)}")
        _check_nesting(self)


Sel = All | Rank | Middle | ReadLabel | Except | Filter | Take  # a selection: a set of side shoots


@dataclass(frozen=True)
class FlowerIs:
    """True when the flower has this colour."""

    colour: str

    def __post_init__(self):
        _require(self.colour in COLOURS, f"FlowerIs takes a flower colour ({', '.join(COLOURS)}), not {self.colour!r}")


@dataclass(frozen=True)
class Count:
    """v3: true when `sel` has exactly n shoots ("exactly"), more than n ("more") or fewer than n ("fewer")."""

    cmp: str
    n: int
    sel: Sel

    def __post_init__(self):
        _require(self.cmp in COMPARISONS, f"Count compares {', '.join(COMPARISONS)}, not {self.cmp!r}")
        n_ok = isinstance(self.n, int) and not isinstance(self.n, bool) and self.n in COUNTS
        _require(n_ok, f"Count compares with 0, 1, 2 or 3 shoots, not {self.n!r}")
        _require(isinstance(self.sel, Sel), f"Count takes a selection, not {_shown(self.sel)}")


Cond = FlowerIs | Count  # a condition


@dataclass(frozen=True)
class Cut:
    """Cut the shoots `sel` selects."""

    sel: Sel

    def __post_init__(self):
        _require(isinstance(self.sel, Sel), f"Cut takes a selection, not {_shown(self.sel)}")


@dataclass(frozen=True)
class CutFrom:
    """v3: cut the shoots `sel` selects one at a time, starting at that end: CutFrom(top,All()) is "from the top
    down"."""

    end: str
    sel: Sel

    def __post_init__(self):
        _require(self.end in ENDS, f"CutFrom starts at the top or the bottom, not {self.end!r}")
        _require(isinstance(self.sel, Sel), f"CutFrom takes a selection, not {_shown(self.sel)}")


@dataclass(frozen=True)
class Nothing:
    """v3: cut no shoot (the branch of an If that asks for nothing)."""


@dataclass(frozen=True)
class Then:
    """v3: carry out `first`, then `rest`; a shoot `first` has cut is not asked for again. A Then holds at most MAX_THEN
    statements, written to the right: Then(a,Then(b,c)), never Then(Then(a,b),c)."""

    first: "Stmt"
    rest: "Stmt"

    def __post_init__(self):
        for part in (self.first, self.rest):
            _require(isinstance(part, Stmt), f"Then takes two statements, not {_shown(part)}")
        _require(not isinstance(self.first, Then), f"a Then nests to the right, Then(a,Then(b,c)): {text(self)}")
        _require(_statements(self) <= MAX_THEN, f"a Then holds at most {MAX_THEN} statements: {text(self)}")


@dataclass(frozen=True)
class If:
    """`yes` if the condition holds, else `no`. v2: the condition is a FlowerIs and a branch is a Cut. v3: the condition
    is a FlowerIs or a Count, and a branch is any statement with no If in it: an If does not nest."""

    cond: Cond
    yes: "Stmt"
    no: "Stmt"

    def __post_init__(self):
        _require(isinstance(self.cond, Cond), f"an If tests FlowerIs(colour) or Count(...), not {_shown(self.cond)}")
        for branch in (self.yes, self.no):
            _require(isinstance(branch, Stmt), f"an If branch is a statement, not {_shown(branch)}")
            _require(not _has_if(branch), f"an If branch has no If in it: {text(branch)}")


Stmt = Cut | If | CutFrom | Then | Nothing  # a statement: a whole program
Node = Stmt | Sel | Cond  # a program or a part of one
_NODE_TYPES = {
    kind.__name__: kind
    for kind in (Cut, If, All, Rank, Middle, Except, Filter, ReadLabel, FlowerIs, CutFrom, Then, Nothing, Take, Count)
}
_V2_TYPES = (Cut, If, All, Rank, Middle, Except, Filter, ReadLabel, FlowerIs)


def _nesting(sel: Sel) -> int:
    """How deep Except, Filter and Take nest in a selection."""
    if isinstance(sel, Except):
        return 1 + max(_nesting(sel.sel), _nesting(sel.without))
    if isinstance(sel, Filter | Take):
        return 1 + _nesting(sel.sel)
    return 0


def _check_nesting(sel: Sel) -> None:
    _require(
        _nesting(sel) <= MAX_NESTING_V3, f"Except, Filter and Take nest at most {MAX_NESTING_V3} deep: {text(sel)}"
    )


def _statements(stmt: Stmt) -> int:
    """How many statements a Then chain holds (1 for any other statement)."""
    return _statements(stmt.first) + _statements(stmt.rest) if isinstance(stmt, Then) else 1


def _has_if(stmt: Stmt) -> bool:
    if isinstance(stmt, If):
        return True
    return isinstance(stmt, Then) and (_has_if(stmt.first) or _has_if(stmt.rest))


def _children(node: Node) -> list:
    return [getattr(node, f.name) for f in dataclasses.fields(node) if isinstance(getattr(node, f.name), Node)]


def _walk(node: Node):
    """Every node of a program, the program first."""
    yield node
    for child in _children(node):
        yield from _walk(child)


def _v2_nesting(sel) -> int:
    if isinstance(sel, Except):
        return 1 + max(_v2_nesting(sel.sel), _v2_nesting(sel.without))
    if isinstance(sel, Filter):
        return 1 + _v2_nesting(sel.sel)
    return 0


def language_error(node: Node, language: str = "v3") -> str | None:
    """Why a program (or a part of one) is not in `language`, or None when it is. v3's rules are the nodes' own (a node
    that breaks them cannot be made); v2's language is v3's without CutFrom, Then, Nothing, Take and Count, with the
    attribute pairs of ATTRIBUTES, selections nested at most MAX_NESTING deep and an If's branches Cuts."""
    assert language in LANGUAGES, f"language {language!r}: one of {LANGUAGES}"
    if language == "v3":
        return None
    for part in _walk(node):
        if not isinstance(part, _V2_TYPES):
            return f"{type(part).__name__} is not in v2's language"
        if isinstance(part, Filter) and (part.attr, part.value) not in _ATTRIBUTE_VALUES:
            return f"Filter takes an attribute and a value of ATTRIBUTES, not {part.attr!r}, {part.value!r}"
        if isinstance(part, Except | Filter) and _v2_nesting(part) > MAX_NESTING:
            return f"Except and Filter nest at most {MAX_NESTING} deep: {text(part)}"
        if isinstance(part, If) and not (isinstance(part.yes, Cut) and isinstance(part.no, Cut)):
            return f"an If branch is a Cut in v2's language: {text(part)}"
    return None


def in_language(node: Node, language: str = "v2") -> bool:
    """True when the program (or a part of one) is in `language` ("v2" or "v3")."""
    return language_error(node, language) is None


def _shown(value) -> str:
    return text(value) if isinstance(value, Node) else repr(value)


def text(node: Node) -> str:
    """The canonical text of a program or of a part of one: every node as Name(arguments), with no spaces."""
    if not isinstance(node, Node):
        raise TypeError(f"not a program or a part of one: {node!r}")
    arguments = (getattr(node, f.name) for f in dataclasses.fields(node))
    return f"{type(node).__name__}({','.join(text(a) if isinstance(a, Node) else str(a) for a in arguments)})"


# Tokens: a word, a single digit, a parenthesis or a comma. Spaces between tokens are skipped; any other character is
# a token of its own, which nothing accepts. A digit is a token by itself, so "02" is two tokens and not the number 2:
# without spaces, the canonical text is the only spelling parse accepts.
_TOKEN = re.compile(r"[A-Za-z]+|[0-9]|[(),]|\S")
# A program of v2 nests at most 5 deep (If, Cut, Except, Except, Rank), one of v3 at most 10 (Then, Then, If, Then,
# Then, Cut, Except, Except, Except, Rank): stop long before Python's stack does
_MAX_DEPTH = 12


def parse(text: str, language: str = "v2") -> Stmt:
    """The program a text writes, in the form `text` writes it; spaces between words are allowed. Raises ProgramError
    for text that is not a whole program of `language` ("v2", the default, or "v3"): bad syntax, an unknown word, a
    value out of range, a selection where a statement belongs (or the other way round), nesting beyond the bounds, a
    construct the language does not have."""
    assert language in LANGUAGES, f"language {language!r}: one of {LANGUAGES}"
    tokens = _TOKEN.findall(text)
    node, end = _read(tokens, 0, depth=1)
    if end < len(tokens):
        raise ProgramError(f"unexpected {tokens[end]!r} after the program")
    if language == "v2" and not isinstance(node, Cut | If):
        raise ProgramError(f"a program is a Cut or an If, not {type(node).__name__}")
    if not isinstance(node, Stmt):
        raise ProgramError(f"a program is a statement, not {type(node).__name__}")
    error = language_error(node, language)
    if error is not None:
        raise ProgramError(error)
    return node


def _read(tokens: list[str], i: int, depth: int) -> tuple[Node, int]:
    """The node that starts at token i, Name(argument,...), and the index of the token after it. An argument is a
    node, a word (top, red, tie) or a digit. The node's own checks reject what does not fit."""
    name = _token(tokens, i)
    if name not in _NODE_TYPES:
        raise ProgramError(f"expected one of {', '.join(_NODE_TYPES)}, not {name!r}")
    if depth > _MAX_DEPTH:
        raise ProgramError("nested deeper than any program")
    if _token(tokens, i + 1) != "(":
        raise ProgramError(f"expected '(' after {name}")
    i += 2
    arguments = []
    if _token(tokens, i) == ")":
        i += 1
    else:
        while True:
            token = _token(tokens, i)
            if i + 1 < len(tokens) and tokens[i + 1] == "(":
                argument, i = _read(tokens, i, depth + 1)
            elif re.fullmatch(r"[A-Za-z]+", token):
                argument, i = token, i + 1
            elif re.fullmatch(r"[0-9]", token):
                argument, i = int(token), i + 1
            else:
                raise ProgramError(f"unexpected {token!r} in {name}(...)")
            arguments.append(argument)
            separator = _token(tokens, i)
            i += 1
            if separator == ")":
                break
            if separator != ",":
                raise ProgramError(f"expected ',' or ')' in {name}(...), not {separator!r}")
    kind = _NODE_TYPES[name]
    wanted = len(dataclasses.fields(kind))
    if len(arguments) != wanted:
        raise ProgramError(f"{name} takes {wanted} argument{'' if wanted == 1 else 's'}, not {len(arguments)}")
    return kind(*arguments), i


def _token(tokens: list[str], i: int) -> str:
    if i >= len(tokens):
        raise ProgramError("the program ends too early")
    return tokens[i]


# -------------------------------------------------------------------------------------- programs of the benchmark tasks
# The positions the instructions name, as selections ("the lowest shoot", "the middle one", "the highest").
_NAMED = {"lowest": Rank(1, "bottom"), "middle": Middle(), "highest": Rank(1, "top")}


def from_task(task: reasoning.Task) -> Stmt:
    """The program of a benchmark task (any kind, either split): what its instruction means, whatever its words."""
    p = task.params
    if task.kind == "ordinal":
        return Cut(Rank(int(p["k"]), p["end"]))
    if task.kind == "conditional":
        return If(FlowerIs(p["colour"]), Cut(_NAMED[p["yes"]]), Cut(_NAMED[p["no"]]))
    if task.kind == "exclusion":
        return Cut(Except(All(), _NAMED[p["keep"]]))
    if task.kind == "hidden":
        return Cut(ReadLabel())  # the card's colour is not in the program: it is read off the plant
    raise ValueError(f"no program for a task of kind {task.kind!r}")


# The programs of the situations built on prune (scenarios.py), by the scenario that sets the task and its mode. An
# instruction that names a position names it on the whole plant, as the scenario's draw does.
_SITUATION_PROGRAMS = {
    ("order", "top_down"): CutFrom("top", All()),
    ("order", "bottom_up"): CutFrom("bottom", All()),
    ("ties", "one"): Cut(Filter("tie", "any", All())),
    ("sick", "sick"): Cut(Filter("sick", "yes", All())),
    ("buds", "buds"): Cut(Filter("bud", "no", All())),
    ("nothing_to_do", "pruned"): Cut(All()),
}


def from_scenario(env) -> Stmt:
    """The program of an episode as `env` (a PickupEnv after reset) has drawn it: a reasoning task's (from_task), else
    the one of the situation that sets the task (scenarios.py: order, nothing_to_do, ties, sick, buds and their held-out
    frames), else pruning's, "cut the side shoots off the stem, keep the flower": Cut(All()). Raises ValueError for a
    task with a named shoot (prune_named), which no program here writes."""
    if getattr(env, "reason", None):
        return from_task(env.reason_task)
    if getattr(env, "named", None):
        raise ValueError(f"no program for a task with a named shoot ({env.task})")
    for scenario in getattr(env, "scenarios", ()):
        variation = env.scenario_info.get(scenario.name, {})
        if not (scenario.changes_task and variation.get("applied")):
            continue
        name = scenario.name.removesuffix("_heldout")
        mode = variation["mode"]
        if (name, mode) in _SITUATION_PROGRAMS:
            return _SITUATION_PROGRAMS[(name, mode)]
        if (name, mode) == ("ties", "colour"):
            return Cut(Filter("tie", variation["colour"], All()))
        if (name, mode) == ("order", "pair"):
            first, second = variation["order"].split(",")
            return Then(Cut(_NAMED[first]), Cut(_NAMED[second]))
        raise ValueError(f"no program for the situation {scenario.name} in mode {mode!r}")
    return Cut(All())


# ----------------------------------------------------------------------------------------------------------- evaluation
@dataclass
class _Plant:
    """What an evaluation reads: the number of shoots in the inventory, the facts known (None: not yet), and each
    shoot's attributes, lowest first. `read` collects the facts the evaluation did read, with the values it read:
    "flower", "label", and (attribute, shoot index) for each shoot a Filter looked at. A plan watches those."""

    count: int
    flower: str | None
    label: str | None  # the card's colour, once read
    attributes: tuple[dict[str, str], ...]
    label_rule: dict[str, str]  # card colour -> the position it names
    read: dict = field(default_factory=dict)


def _shoots(program: Stmt, plant: _Plant) -> list[int]:
    """The shoots `program` cuts on `plant`, by index from the lowest up. Evaluation reads the program from left to
    right and stops at the first fact it lacks (NeedsObservation) or shoot the plant does not have (ProgramError);
    an If reads only the branch it takes."""
    if isinstance(program, If):
        if plant.flower is None:
            raise NeedsObservation("flower")
        plant.read["flower"] = plant.flower
        program = program.yes if plant.flower == program.cond.colour else program.no
    return sorted(_select(program.sel, plant))


def _select(sel: Sel, plant: _Plant) -> set[int]:
    n = plant.count
    if isinstance(sel, All):
        return set(range(n))
    if isinstance(sel, Rank):
        if sel.k > n:
            raise ProgramError(f"no {reasoning.ORDINALS[sel.k]} shoot from the {sel.end} on a plant with {n}")
        return {reasoning.ordinal_index(sel.k, sel.end, n)}
    if isinstance(sel, Middle):
        if n != 3:
            raise ProgramError(f"no middle shoot on a plant with {n}")
        return {1}
    if isinstance(sel, Except):
        return _select(sel.sel, plant) - _select(sel.without, plant)
    if isinstance(sel, Filter):
        chosen = _select(sel.sel, plant)
        if any(sel.attr not in plant.attributes[i] for i in chosen):
            raise NeedsObservation(sel.attr)
        plant.read.update({(sel.attr, i): plant.attributes[i][sel.attr] for i in chosen})
        return {i for i in chosen if plant.attributes[i][sel.attr] == sel.value}
    # ReadLabel: the card names a position, which selects as the instruction's own words would
    if plant.label is None:
        raise NeedsObservation("label")
    plant.read["label"] = plant.label
    if plant.label not in plant.label_rule:
        raise ProgramError(f"a {plant.label} label names no shoot")
    return _select(_NAMED[plant.label_rule[plant.label]], plant)


def _statement(node: Node) -> Stmt:
    """A program; a selection alone is read as Cut(selection)."""
    if isinstance(node, Stmt):
        return node
    if isinstance(node, Sel):
        return Cut(node)
    raise TypeError(f"not a program or a selection: {node!r}")


def targets_on(node: Node, count: int, flower: str | None, label: str | None) -> tuple[str, ...]:
    """The positions the program (or a selection alone) selects on a plant with `count` shoots (2 or 3), a flower of
    colour `flower` and a card of colour `label` (None: not read, or no card), lowest first. Raises ProgramError when
    it names a shoot the plant does not have, NeedsObservation when it needs a fact not given: the label, the flower,
    or an attribute (none is given here)."""
    plant = _Plant(count, flower, label, ({},) * count, reasoning.LABEL_MEANS)
    return tuple(reasoning.position_name(i, count) for i in _shoots(_statement(node), plant))


# Every plant of the benchmark, as far as a program can tell them apart: (count, flower, label).
_PLANTS = tuple(
    (count, flower, label)
    for count in (2, 3)
    for flower in reasoning.FLOWERS
    for label in (*reasoning.label_options(count), None)
)


def _outcome(node: Node, count: int, flower: str, label: str | None) -> tuple:
    try:
        return ("cut", targets_on(node, count, flower, label))
    except NeedsObservation as need:
        return ("needs", need.what)
    except ProgramError:
        return ("no such shoot",)


def equivalent(a: Node, b: Node, language: str | None = None) -> bool:
    """True when two programs (or selections) mean the same.

    v2's comparison (language "v2"): the same result on every plant of the benchmark: 2 and 3 shoots, every flower
    colour, every card colour the count allows and no card. A result is the positions selected, a shoot the plant does
    not have, or the fact needed. No plant here has attributes, so a Filter over any shoot needs one: two programs that
    differ only in the value a Filter keeps count as equivalent.

    v3's comparison (language "v3", equivalent_v3): plants with attributes, and the sequences of cuts each program
    allows. With language None, v2's when both programs are in v2's language (so that what v2's runs recorded stays
    what it was) and v3's otherwise."""
    if language is None:
        language = "v2" if in_language(a, "v2") and in_language(b, "v2") else "v3"
    if language == "v3":
        return equivalent_v3(a, b)
    return all(_outcome(a, *plant) == _outcome(b, *plant) for plant in _PLANTS)


# ------------------------------------------------------------------------------------------------ evaluation in v3
# A v3 program evaluates to *stages*: sets of shoots in the order in which they are to be cut. Cut(sel) is one stage,
# CutFrom one stage per shoot from its end, Then the stages of its first statement and then those of the rest without
# the shoots already asked for, Nothing none; empty stages are dropped. Inside a stage any order is allowed. A Take with
# "any" leaves a choice: evaluated alone (stages_on, equivalent_v3) every choice is listed, one list of stages each; a
# plan (WorldPlan) makes the choice itself, once. Positions and ranks count over the whole plant, cut or not, as in v2.
@dataclass
class _Shoots:
    """What a v3 evaluation reads: the number of shoots, the facts known (None: not yet), each shoot's attributes
    (lowest first), the card's rule, and `choose`, which makes the choice of a Take with "any": (path of the Take in
    the program, the shoots it takes from, k, keep) -> the possible choices. `keep` is true where the Take is inside the
    second selection of an odd number of Excepts: the shoots it takes are kept, not cut. None: every choice."""

    count: int
    flower: str | None = None
    label: str | None = None
    attributes: tuple = ()
    label_rule: dict = field(default_factory=lambda: dict(reasoning.LABEL_MEANS))
    choose: object = None


def _distinct(items) -> list:
    out = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def _matches(attr: str, wanted: str, value: str) -> bool:
    """A shoot whose attribute `attr` has `value` is one Filter(attr,wanted,...) keeps: "any" keeps every value but
    "none" (a tie of any colour)."""
    return value != "none" if wanted == "any" else value == wanted


def _alternatives(sel: Sel, plant: _Shoots, path: tuple = (), keep: bool = False) -> list[frozenset]:
    """The sets of shoots (indices from the lowest up) `sel` can select: one, or one per choice a Take leaves. Reads
    left to right and raises NeedsObservation at the first fact it lacks, ProgramError for a shoot the plant does not
    have."""
    n = plant.count
    if isinstance(sel, All):
        return [frozenset(range(n))]
    if isinstance(sel, Rank):
        if sel.k > n:
            raise ProgramError(f"no {reasoning.ORDINALS[sel.k]} shoot from the {sel.end} on a plant with {n}")
        return [frozenset({reasoning.ordinal_index(sel.k, sel.end, n)})]
    if isinstance(sel, Middle):
        if n != 3:
            raise ProgramError(f"no middle shoot on a plant with {n}")
        return [frozenset({1})]
    if isinstance(sel, ReadLabel):
        if plant.label is None:
            raise NeedsObservation("label")
        if plant.label not in plant.label_rule:
            raise ProgramError(f"a {plant.label} label names no shoot")
        return _alternatives(_NAMED[plant.label_rule[plant.label]], plant, path, keep)
    if isinstance(sel, Except):
        kept = _alternatives(sel.sel, plant, (*path, 0), keep)
        without = _alternatives(sel.without, plant, (*path, 1), not keep)
        return _distinct(a - b for a in kept for b in without)
    if isinstance(sel, Filter):
        out = []
        for chosen in _alternatives(sel.sel, plant, (*path, 2), keep):
            for i in sorted(chosen):
                if sel.attr not in plant.attributes[i]:
                    raise NeedsObservation(sel.attr, i)
            out.append(frozenset(i for i in chosen if _matches(sel.attr, sel.value, plant.attributes[i][sel.attr])))
        return _distinct(out)
    # Take
    out = []
    for chosen in _alternatives(sel.sel, plant, (*path, 2), keep):
        if len(chosen) <= sel.k:
            out.append(chosen)
        elif sel.pick != "any":
            ranked = sorted(chosen, reverse=sel.pick == "top")
            out.append(frozenset(ranked[: sel.k]))
        elif plant.choose is None:
            out += [frozenset(c) for c in itertools.combinations(sorted(chosen), sel.k)]
        else:
            out += plant.choose(path, chosen, sel.k, keep)
    return _distinct(out)


def _holds(cond: Cond, plant: _Shoots, path: tuple) -> list[bool]:
    """The values a condition can have (two where a Take's choice decides it)."""
    if isinstance(cond, FlowerIs):
        if plant.flower is None:
            raise NeedsObservation("flower")
        return [plant.flower == cond.colour]
    sizes = {len(chosen) for chosen in _alternatives(cond.sel, plant, (*path, 2))}
    compare = {"exactly": lambda s: s == cond.n, "more": lambda s: s > cond.n, "fewer": lambda s: s < cond.n}
    return sorted({compare[cond.cmp](size) for size in sizes}, reverse=True)


def _stage_lists(stmt: Stmt, plant: _Shoots, path: tuple = ()) -> list[tuple[frozenset, ...]]:
    """The lists of stages a statement can evaluate to (one per choice it leaves), each a tuple of non-empty sets."""
    if isinstance(stmt, Cut):
        return _distinct((chosen,) if chosen else () for chosen in _alternatives(stmt.sel, plant, (*path, 0)))
    if isinstance(stmt, CutFrom):
        return _distinct(
            tuple(frozenset({i}) for i in sorted(chosen, reverse=stmt.end == "top"))
            for chosen in _alternatives(stmt.sel, plant, (*path, 1))
        )
    if isinstance(stmt, Nothing):
        return [()]
    if isinstance(stmt, Then):
        out = []
        for first in _stage_lists(stmt.first, plant, (*path, 0)):
            done = frozenset().union(*first)
            for rest in _stage_lists(stmt.rest, plant, (*path, 1)):
                out.append(first + tuple(stage - done for stage in rest if stage - done))
        return _distinct(out)
    # If
    out = []
    for holds in _holds(stmt.cond, plant, (*path, 0)):
        out += _stage_lists(stmt.yes if holds else stmt.no, plant, (*path, 1 if holds else 2))
    return _distinct(out)


def stages_on(
    node: Node,
    count: int,
    flower: str | None = None,
    label: str | None = None,
    attributes: tuple | list | None = None,
    label_rule: dict | None = None,
) -> list[tuple[tuple[str, ...], ...]]:
    """The stages a program (of either language; a selection alone reads as Cut(selection)) evaluates to on a plant of
    `count` shoots, as positions ("lowest", "middle", "highest"), one list of stages per choice a Take with "any"
    leaves. `attributes`: per shoot, lowest first, a dict of the attributes known ({"tie": "red", "bud": "no"}). Raises
    ProgramError for a shoot the plant does not have, NeedsObservation for a fact not given."""
    rule = dict(reasoning.LABEL_MEANS if label_rule is None else label_rule)
    known = tuple(dict(a) for a in attributes) if attributes is not None else ({},) * count
    plant = _Shoots(count, flower, label, known, rule)
    return [
        tuple(tuple(reasoning.position_name(i, count) for i in sorted(stage)) for stage in stages)
        for stages in _stage_lists(_statement(node), plant)
    ]


def _sequences(stage_lists: list[tuple[frozenset, ...]]) -> frozenset[tuple[int, ...]]:
    """Every order of cuts the stages allow: the stages in turn, each stage's shoots in any order."""
    out = set()
    for stages in stage_lists:
        for orders in itertools.product(*(itertools.permutations(sorted(stage)) for stage in stages)):
            out.add(tuple(i for order in orders for i in order))
    return frozenset(out)


def _reads(node: Node) -> tuple[bool, bool, frozenset[str]]:
    """Whether a program can read the flower, the card, and which attributes."""
    nodes = list(_walk(node))
    return (
        any(isinstance(n, FlowerIs) for n in nodes),
        any(isinstance(n, ReadLabel) for n in nodes),
        frozenset(n.attr for n in nodes if isinstance(n, Filter)),
    )


def _plants_v3(a: Node, b: Node):
    """Every plant on which two programs can differ: 2 and 3 shoots; every flower colour if either reads it; every
    card colour the count allows and no card if either reads the card; every assignment of the attributes either
    reads."""
    flower_a, label_a, attrs_a = _reads(a)
    flower_b, label_b, attrs_b = _reads(b)
    attrs = sorted(attrs_a | attrs_b)
    flowers = reasoning.FLOWERS if flower_a or flower_b else (None,)
    shoot_values = [dict(zip(attrs, values, strict=True)) for values in itertools.product(*(DOMAINS[a] for a in attrs))]
    for count in (2, 3):
        labels = (*reasoning.label_options(count), None) if label_a or label_b else (None,)
        for flower, label in itertools.product(flowers, labels):
            for attributes in itertools.product(shoot_values, repeat=count):
                yield _Shoots(count, flower, label, attributes)


def _outcome_v3(node: Node, plant: _Shoots) -> tuple:
    try:
        return ("cuts", _sequences(_stage_lists(_statement(node), plant)))
    except NeedsObservation as need:
        return ("needs", need.what)
    except ProgramError:
        return ("no such shoot",)


def equivalent_v3(a: Node, b: Node) -> bool:
    """v3's equivalence: the same sequences of cuts allowed (or the same fact needed, or both name a shoot the plant
    does not have) on every plant either program can tell apart (_plants_v3): two or three shoots, every flower and
    card, every assignment of the attributes either reads. CutFrom(top,All()) and Cut(All()) differ; a Take with "any"
    equals another only if both allow the same choices."""
    a, b = _statement(a), _statement(b)
    return all(_outcome_v3(a, plant) == _outcome_v3(b, plant) for plant in _plants_v3(a, b))


# ------------------------------------------------------------------------------------ grammar, for constrained decoding
# The language with its bounds written in: <selD> is a selection in which Except and Filter nest at most D deep, and an
# If branch is a <cut>. Nonterminals are in angle brackets; every other symbol is a terminal, a fixed string. Nothing
# is recursive (the bounds unroll the recursion) and every nonterminal can be finished, so a prefix that the grammar
# can still continue always begins some program: that is what makes next_chars exact.
def _selection_rules(depth: int) -> list[tuple[str, ...]]:
    rules = [("All()",), ("Rank(", "<k>", ",", "<end>", ")"), ("Middle()",), ("ReadLabel()",)]
    if depth > 0:
        inner = f"<sel{depth - 1}>"
        rules += [("Except(", inner, ",", inner, ")"), ("Filter(", "<attr,value>", ",", inner, ")")]
    return rules


GRAMMAR: dict[str, list[tuple[str, ...]]] = {
    "<program>": [("<cut>",), ("If(", "<cond>", ",", "<cut>", ",", "<cut>", ")")],
    "<cut>": [("Cut(", f"<sel{MAX_NESTING}>", ")")],
    **{f"<sel{depth}>": _selection_rules(depth) for depth in range(MAX_NESTING + 1)},
    "<cond>": [("FlowerIs(", "<colour>", ")")],
    "<k>": [(str(k),) for k in RANKS],
    "<end>": [(end,) for end in ENDS],
    "<colour>": [(colour,) for colour in COLOURS],
    "<attr,value>": [(f"{attr},{value}",) for attr, value in _ATTRIBUTE_VALUES],
}


# v3's language in the same form. A Then is written to the right and holds at most three statements, an If's branch
# has no If in it, and Except, Filter and Take nest at most three deep. <atom> is a statement that is not a Then,
# <basic> one that is neither a Then nor an If.
def _selection_rules_v3(depth: int) -> list[tuple[str, ...]]:
    rules = [("All()",), ("Rank(", "<k>", ",", "<end>", ")"), ("Middle()",), ("ReadLabel()",)]
    if depth > 0:
        inner = f"<sel{depth - 1}>"
        rules += [
            ("Except(", inner, ",", inner, ")"),
            ("Filter(", "<attr,value>", ",", inner, ")"),
            ("Take(", "<k>", ",", "<pick>", ",", inner, ")"),
        ]
    return rules


_SEL_V3 = f"<sel{MAX_NESTING_V3}>"
GRAMMAR_V3: dict[str, list[tuple[str, ...]]] = {
    "<program>": [("<atom>",), ("Then(", "<atom>", ",", "<then>", ")")],
    "<then>": [("<atom>",), ("Then(", "<atom>", ",", "<atom>", ")")],
    "<atom>": [("<basic>",), ("If(", "<cond>", ",", "<branch>", ",", "<branch>", ")")],
    "<branch>": [("<basic>",), ("Then(", "<basic>", ",", "<branch_then>", ")")],
    "<branch_then>": [("<basic>",), ("Then(", "<basic>", ",", "<basic>", ")")],
    "<basic>": [("Cut(", _SEL_V3, ")"), ("CutFrom(", "<end>", ",", _SEL_V3, ")"), ("Nothing()",)],
    **{f"<sel{depth}>": _selection_rules_v3(depth) for depth in range(MAX_NESTING_V3 + 1)},
    "<cond>": [("FlowerIs(", "<colour>", ")"), ("Count(", "<cmp>", ",", "<n>", ",", _SEL_V3, ")")],
    "<k>": [(str(k),) for k in RANKS],
    "<n>": [(str(n),) for n in COUNTS],
    "<end>": [(end,) for end in ENDS],
    "<pick>": [(pick,) for pick in PICKS],
    "<cmp>": [(cmp,) for cmp in COMPARISONS],
    "<colour>": [(colour,) for colour in COLOURS],
    "<attr,value>": [(f"{attr},{value}",) for attr, value in _ATTRIBUTE_VALUES_V3],
}
GRAMMARS = {"v2": GRAMMAR, "v3": GRAMMAR_V3}


def _alphabet(grammar: dict) -> frozenset[str]:
    return frozenset(
        char
        for rules in grammar.values()
        for rule in rules
        for symbol in rule
        if not symbol.startswith("<")
        for char in symbol
    )


# Every character a program can contain: of v2's language (ALPHABET) and of each language (ALPHABETS).
ALPHABET: frozenset[str] = _alphabet(GRAMMAR)
ALPHABETS = {language: _alphabet(grammar) for language, grammar in GRAMMARS.items()}

# The acceptor reads a prefix one character at a time and keeps every configuration it can be in: the symbols still
# to be written, first symbol first, where a terminal that is partly written keeps only its rest. Several
# configurations are alive at once where terminals share their first letters ("Rank(" and "ReadLabel()", "pink" and
# "purple", the four "tie,..."); the next characters settle it. A prefix begins some program while any configuration
# is alive, and a text is a whole program when one configuration has nothing left to write. No program continues
# another (each ends with the parenthesis its first word opened), so after a whole program nothing is alive.
_START = frozenset({("<program>",)})


@functools.lru_cache(maxsize=1 << 16)
def _expanded(configurations: frozenset, language: str = "v2") -> frozenset:
    """The same configurations, each leading nonterminal replaced by its rules until a terminal leads. A
    configuration with nothing left to write (a whole program) has no next character and is dropped."""
    grammar = GRAMMARS[language]
    out, todo = set(), list(configurations)
    while todo:
        configuration = todo.pop()
        if configuration and configuration[0].startswith("<"):
            todo.extend(rule + configuration[1:] for rule in grammar[configuration[0]])
        elif configuration:
            out.add(configuration)
    return frozenset(out)


@functools.lru_cache(maxsize=1 << 16)
def _step(configurations: frozenset, char: str, language: str = "v2") -> frozenset:
    """The configurations after one more character."""
    out = set()
    for terminal, *rest in _expanded(configurations, language):
        if terminal[0] == char:
            out.add((terminal[1:], *rest) if len(terminal) > 1 else tuple(rest))
    return frozenset(out)


def _configurations(prefix: str, language: str = "v2") -> frozenset:
    # Both steps are cached: a decoder that adds one character at a time pays a dictionary lookup per character.
    assert language in GRAMMARS, f"language {language!r}: one of {LANGUAGES}"
    configurations = _START
    for char in prefix:
        if not configurations:
            break
        configurations = _step(configurations, char, language)
    return configurations


def is_complete(text: str, language: str = "v2") -> bool:
    """True when `text` is a whole program of `language` (v2's by default) within its bounds, in canonical form (no
    spaces)."""
    return () in _configurations(text, language)


def next_chars(prefix: str, language: str = "v2") -> set[str]:
    """Every character c such that prefix + c begins some program of `language` (v2's by default) within its bounds
    (canonical form). Empty for a prefix that no program begins with, and after a whole program."""
    return {configuration[0][0] for configuration in _expanded(_configurations(prefix, language), language)}


# ------------------------------------------------------------------------------------------------------------- the plan
@dataclass(frozen=True)
class Execute:
    """Cut a shoot: the sub-goal for the fast level ("cut the middle shoot") and the remembered part's id."""

    subgoal: str
    part: int


@dataclass(frozen=True)
class NeedObservation:
    """A fact the program needs is missing: "plant" (the inventory is not settled), "flower", "label" or an
    attribute's name. `subgoal`: "show the label" for the label (turn the plant until the card faces the front
    camera), None for the rest (hold). WorldPlan: for an attribute, `part` is the part's id and `subgoal` "show the
    <position> shoot" (turn the plant until that shoot faces the front camera)."""

    what: str
    subgoal: str | None
    part: int | None = None


@dataclass(frozen=True)
class Complete:
    """Every shoot meant is cut."""


@dataclass(frozen=True)
class Failed:
    """The program names a shoot the plant does not have, or a shoot that was to stay is cut."""

    reason: str


class Plan:
    """A program bound to the parts of a scene memory. `step(memory)` says what to do now; call it whenever the
    memory changes (or every chunk: it costs next to nothing).

    The memory is read, never written, and only through `parts` (the inventory, lowest first: objects with `id`,
    `cut`, `bearing` and `attributes`), `settled`, `flower`, `label_read` and `position(id)`, as scene.py's
    SceneMemory has them. What was cut is what the memory saw cut, never what a sub-goal asked for.

    When it binds. The first evaluation that succeeds binds the program's selection to part ids (`targets`), and the
    plan keeps the facts that evaluation read, with the values it read: the inventory always, the flower's colour if
    a FlowerIs was evaluated, the card's colour if a ReadLabel was, and the attribute of each shoot a Filter looked
    at. It binds again, counted in `rebinds`, when any of them differs in the memory: the inventory grew, or the
    memory corrected a first wrong reading. A shoot cut under the old reading that the new binding does not ask for
    is a shoot that was to stay, and the plan says Failed. If binding again does not succeed (a fact is missing, or
    the plant has no such shoot), the plan says so and keeps the old targets until it does.

    `label_rule`: card colour -> the position it names (default reasoning.LABEL_MEANS).
    """

    def __init__(self, program: Node, label_rule: dict[str, str] | None = None):
        self.program = _statement(program)
        self.label_rule = dict(reasoning.LABEL_MEANS if label_rule is None else label_rule)
        if not set(self.label_rule.values()) <= set(reasoning.POSITIONS):
            raise ValueError(f"a card names one of {reasoning.POSITIONS}: {self.label_rule}")
        self.targets: list[int] | None = None  # the part ids bound to the program's selection, lowest first
        self.rebinds = 0  # how often the binding was made again because a fact it read changed
        # The facts the binding read, as the memory said them then: "inventory" (the part ids), "flower", "label",
        # and (attribute, part id) for each shoot a Filter looked at.
        self._read: dict = {}

    def step(self, memory) -> Execute | NeedObservation | Complete | Failed:
        # 1. Positions are named over the whole plant: until the inventory is believed complete, they are unknown.
        if not memory.settled:
            return NeedObservation("plant", None)
        parts = list(memory.parts)
        # 2. Bind, or bind again when a fact the binding read has changed. The program is evaluated over every part,
        #    cut or not, and the result is kept as part ids, so the binding never moves to other shoots as shoots are
        #    cut. A fact the evaluation did not read cannot change its result, so it is not watched.
        if self.targets is None or self._stale(memory, parts):
            attributes = tuple(part.attributes for part in parts)
            plant = _Plant(len(parts), memory.flower, memory.label_read, attributes, self.label_rule)
            try:
                chosen = _shoots(self.program, plant)
            except NeedsObservation as need:
                return NeedObservation(need.what, "show the label" if need.what == "label" else None)
            except ProgramError as error:
                return Failed(str(error))
            if self.targets is not None:
                self.rebinds += 1
            self.targets = [parts[i].id for i in chosen]
            self._read = {"inventory": frozenset(part.id for part in parts)}
            for fact, value in plant.read.items():
                # a shoot's attribute is kept under the part's id: its index changes when a shoot is found below it
                key = (fact[0], parts[fact[1]].id) if isinstance(fact, tuple) else fact
                self._read[key] = value
        # 3. A cut the program did not ask for cannot be undone.
        if any(part.cut and part.id not in self.targets for part in parts):
            return Failed("a shoot that was to stay is cut")
        # 4. Done when the memory has seen every target cut. An unread label or an unknown flower never gets here:
        #    without the binding, nothing says which shoots were meant.
        left = [part for part in parts if part.id in self.targets and not part.cut]
        if not left:
            return Complete()
        # 5. The target that needs the least turning of the plate (a bearing not known counts as half a turn); min
        #    keeps the first of equals, the lowest.
        part = min(left, key=lambda part: math.pi if part.bearing is None else abs(part.bearing))
        return Execute(f"cut the {memory.position(part.id)} shoot", part.id)

    def _stale(self, memory, parts) -> bool:
        """True when a fact the binding read is no longer what the memory says."""
        now = {"inventory": frozenset(part.id for part in parts), "flower": memory.flower, "label": memory.label_read}
        attributes = {part.id: part.attributes for part in parts}
        for fact, then in self._read.items():
            if isinstance(fact, tuple):
                attr, part_id = fact
                value = attributes.get(part_id, {}).get(attr)
            else:
                value = now[fact]
            if value != then:
                return True
        return False


# ---------------------------------------------------------------------------------------------- the plan of v3
STAYED = "a shoot that was to stay is cut"  # Plan's reason, and WorldPlan's
OUT_OF_ORDER = "a shoot was cut out of its order"
PLACES = {"tie": "tie", "tag": "tag", "sick": "leaves", "bud": "tip"}  # where each attribute is read


def cannot_see(attr: str, position: str) -> str:
    """The reason of a Failed for a fact that cannot be had: "cannot see the leaves of the middle shoot"."""
    return f"cannot see the {PLACES.get(attr, attr)} of the {position} shoot"


def _turning(part) -> float:
    """How far the plate has to turn to bring a part to face the robot (a bearing not known: half a turn)."""
    return math.pi if part.bearing is None else abs(part.bearing)


class WorldPlan:
    """v3's plan (tutorial/architecture-v3.md, "The plan and the runner"): a program of either language evaluated on a
    scene memory in every *world* the unknown facts still allow. `step(memory)` says what to do now, as Plan does, and
    reads the memory the same way (`parts`, each with `id`, `cut`, `bearing` and `attributes`; `settled`, `flower`,
    `label_read`, `position(id)`).

    A world is one value for each fact the evaluation reads and the memory does not know: the flower's colour, the
    card's colour, an attribute of one shoot (DOMAINS). Only the facts the evaluation reads are enumerated, as it
    reads them, so a fact that does not change what the program selects is never one. In each world the program
    evaluates to stages (sets of parts in the order in which they are to be cut); the open stage is the first that has
    a part the memory has not seen cut. Then:

      every world has a shoot cut that was to stay, or one cut before an earlier stage was done, or names a shoot the
      plant does not have: Failed (STAYED, OUT_OF_ORDER, or the evaluation's own reason)
      every world has nothing left to cut: Complete (a plant of stubs, a selection that is empty)
      a part is in the open stage of every world: Execute the one of those that needs the least turning
      otherwise: NeedObservation of a fact on which the outcome depends ("flower"; "label" with "show the label"; an
      attribute with "show the <position> shoot" and the part's id), or Failed "cannot see the tie (leaves, tip) of
      the <position> shoot" (cannot_see) when every such fact is about a part already cut: its tie, leaves and bud
      fell with it

    A fact the outcome depends on: one whose value changes the outcomes the worlds allow (looked for in the order the
    evaluation read the facts); where no single fact does (two facts that decide together), the first fact read that
    can still be seen. The plan never acts on a guess and never takes a fact it does not have for a "no".

    Bound once: positions and ranks count over the plant as found (every part, cut or not), and the choice of a Take
    with "any" is made the first time its selection is evaluated and kept as part ids (`choices`): among shoots to cut
    it takes those that need the least turning, among shoots to keep (inside the second selection of an Except) those
    that need the most. A choice is made again only for another selection (the inventory grew, or another world).

    `stages`: the stages as part ids when every world agrees, else None; `targets` their parts, lowest first (None
    until then); `rebinds` how often the stages every world agreed on changed; `worlds` the worlds evaluated at the
    last step. With every fact the program reads known there is one world, and for v2's programs the decisions are
    v2's Plan's.
    """

    def __init__(self, program: Node, label_rule: dict[str, str] | None = None):
        self.program = _statement(program)
        self.label_rule = dict(reasoning.LABEL_MEANS if label_rule is None else label_rule)
        if not set(self.label_rule.values()) <= set(reasoning.POSITIONS):
            raise ValueError(f"a card names one of {reasoning.POSITIONS}: {self.label_rule}")
        self.stages: list[list[int]] | None = None
        self.targets: list[int] | None = None
        self.rebinds = 0
        self.worlds = 0
        self.choices: dict[tuple, frozenset] = {}  # (path, part ids taken from, k, keep) -> the part ids taken

    # ------------------------------------------------------------------------------------------------ evaluating
    def _domain(self, fact: tuple) -> tuple:
        if fact[0] == "flower":
            return reasoning.FLOWERS
        if fact[0] == "label":
            return tuple(self.label_rule)
        return DOMAINS[fact[0]]

    def _chooser(self, parts: list):
        index = {part.id: i for i, part in enumerate(parts)}

        def choose(path, chosen, k, keep):
            key = (path, frozenset(parts[i].id for i in chosen), k, keep)
            if key not in self.choices:
                sign = -1.0 if keep else 1.0
                ranked = sorted(chosen, key=lambda i: (sign * _turning(parts[i]), i))
                self.choices[key] = frozenset(parts[i].id for i in ranked[:k])
            return [frozenset(index[part_id] for part_id in self.choices[key])]

        return choose

    def _evaluate(self, memory, parts: list, assumed: dict) -> tuple:
        """("stages", stages as part ids) or ("error", reason) in the world `assumed` ({fact: value}); raises
        NeedsObservation with the fact's key for a fact neither the memory nor the world gives."""
        attributes = []
        for part in parts:
            known = dict(part.attributes)
            known.update({fact[0]: value for fact, value in assumed.items() if fact[1:] == (part.id,)})
            attributes.append(known)
        plant = _Shoots(
            len(parts),
            memory.flower if memory.flower is not None else assumed.get(("flower",)),
            memory.label_read if memory.label_read is not None else assumed.get(("label",)),
            tuple(attributes),
            self.label_rule,
            self._chooser(parts),
        )
        try:
            (stages,) = _stage_lists(self.program, plant)  # the plan's own choices: one list
        except ProgramError as error:
            return ("error", str(error))
        return ("stages", tuple(frozenset(parts[i].id for i in stage) for stage in stages))

    def _worlds(self, memory, parts: list) -> list[tuple[dict, tuple]]:
        """Every world the unknown facts allow, as far as the evaluation reads them: (the facts assumed, the result)."""
        out, todo = [], [{}]
        while todo:
            assumed = todo.pop(0)
            try:
                out.append((assumed, self._evaluate(memory, parts, assumed)))
            except NeedsObservation as need:
                fact = (need.what,) if need.index is None else (need.what, parts[need.index].id)
                todo[:0] = [{**assumed, fact: value} for value in self._domain(fact)]
        return out

    @staticmethod
    def _status(result: tuple, cut: frozenset) -> tuple:
        """("failed", reason), ("complete",) or ("open", the parts of the open stage not cut yet)."""
        if result[0] == "error":
            return ("failed", result[1])
        stages = result[1]
        if cut - frozenset().union(*stages):
            return ("failed", STAYED)
        for j, stage in enumerate(stages):
            if stage - cut:
                if cut & frozenset().union(*stages[j + 1 :]):
                    return ("failed", OUT_OF_ORDER)
                return ("open", stage - cut)
        return ("complete",)

    # ---------------------------------------------------------------------------------------------------- deciding
    def step(self, memory) -> Execute | NeedObservation | Complete | Failed:
        if not memory.settled:
            return NeedObservation("plant", None)
        parts = list(memory.parts)
        cut = frozenset(part.id for part in parts if part.cut)
        worlds = self._worlds(memory, parts)
        self.worlds = len(worlds)
        statuses = [self._status(result, cut) for _, result in worlds]
        results = {result for _, result in worlds}
        if len(results) == 1 and worlds[0][1][0] == "stages":
            stages = [sorted(stage, key=[part.id for part in parts].index) for stage in worlds[0][1][1]]
            if self.stages is not None and stages != self.stages:
                self.rebinds += 1
            self.stages = stages
            meant = {part_id for stage in stages for part_id in stage}
            self.targets = [part.id for part in parts if part.id in meant]
        kinds = {status[0] for status in statuses}
        if kinds == {"failed"}:
            return Failed(statuses[0][1])
        if kinds == {"complete"}:
            return Complete()
        if kinds == {"open"}:
            common = frozenset.intersection(*(status[1] for status in statuses))
            if common:
                part = min((part for part in parts if part.id in common), key=_turning)  # min keeps the lowest
                return Execute(f"cut the {memory.position(part.id)} shoot", part.id)
        return self._ask(memory, parts, worlds, statuses)

    def _ask(self, memory, parts, worlds, statuses):
        """The fact to observe, or Failed when the facts that decide can no longer be seen."""
        facts = list(dict.fromkeys(fact for assumed, _ in worlds for fact in assumed))
        cut = {part.id for part in parts if part.cut}

        def seeable(fact):
            return len(fact) == 1 or fact[1] not in cut

        def relevant(fact):
            outcomes = [
                frozenset(
                    status for (assumed, _), status in zip(worlds, statuses, strict=True) if assumed.get(fact, v) == v
                )
                for v in self._domain(fact)
            ]
            return len(set(outcomes)) > 1

        deciding = [fact for fact in facts if relevant(fact)] or facts
        fact = next((fact for fact in deciding if seeable(fact)), deciding[0])
        if fact == ("flower",):
            return NeedObservation("flower", None)
        if fact == ("label",):
            return NeedObservation("label", "show the label")
        attr, part_id = fact
        position = memory.position(part_id)
        if not seeable(fact):
            return Failed(cannot_see(attr, position))
        return NeedObservation(attr, f"show the {position} shoot", part_id)
