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

This module is pure Python.
"""

import dataclasses
import functools
import math
import re
from dataclasses import dataclass, field

from . import reasoning


# --------------------------------------------------------------------------------------------------------------- errors
class ProgramError(ValueError):
    """Text that is not a program, a program that breaks the language's rules, or a shoot the plant does not have."""


class NeedsObservation(Exception):
    """Evaluating a program needs a fact that is not known yet: `what` is "flower", "label" or an attribute's name."""

    def __init__(self, what: str):
        super().__init__(f"the {what} is not known yet")
        self.what = what


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ProgramError(message)


# --------------------------------------------------------------------------------------------------------- the language
RANKS = tuple(reasoning.ORDINALS)  # (1, 2, 3): the k of Rank(k,end)
ENDS = reasoning.ENDS  # ("top", "bottom")
COLOURS = reasoning.FLOWERS
# What a Filter can ask about a shoot, and the values it can name. No task sets any yet: they are for the test
# situations with gardener's ties, sick shoots and buds (tutorial/test-situations.md).
ATTRIBUTES = {"tie": ("red", "blue", "yellow", "white"), "sick": ("yes",), "bud": ("yes", "no")}
_ATTRIBUTE_VALUES = tuple((attr, value) for attr, values in ATTRIBUTES.items() for value in values)
MAX_NESTING = 2  # how deep Except and Filter may nest in a selection: what keeps the language finite


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
        _require(_nesting(self) <= MAX_NESTING, f"Except and Filter nest at most {MAX_NESTING} deep: {text(self)}")


@dataclass(frozen=True)
class Filter:
    """The shoots of `sel` whose attribute `attr` has `value`: Filter("tie", "red", All()) is every shoot with a red
    tie. It needs the attribute seen on every shoot of `sel`."""

    attr: str
    value: str
    sel: "Sel"

    def __post_init__(self):
        pair_ok = (self.attr, self.value) in _ATTRIBUTE_VALUES
        _require(pair_ok, f"Filter takes an attribute and a value of ATTRIBUTES, not {self.attr!r}, {self.value!r}")
        _require(isinstance(self.sel, Sel), f"Filter takes a selection, not {_shown(self.sel)}")
        _require(_nesting(self) <= MAX_NESTING, f"Except and Filter nest at most {MAX_NESTING} deep: {text(self)}")


Sel = All | Rank | Middle | ReadLabel | Except | Filter  # a selection: a set of side shoots


@dataclass(frozen=True)
class FlowerIs:
    """True when the flower has this colour."""

    colour: str

    def __post_init__(self):
        _require(self.colour in COLOURS, f"FlowerIs takes a flower colour ({', '.join(COLOURS)}), not {self.colour!r}")


@dataclass(frozen=True)
class Cut:
    """Cut the shoots `sel` selects."""

    sel: Sel

    def __post_init__(self):
        _require(isinstance(self.sel, Sel), f"Cut takes a selection, not {_shown(self.sel)}")


@dataclass(frozen=True)
class If:
    """`yes` if the condition holds, else `no`. A branch is a Cut: an If does not nest."""

    cond: FlowerIs
    yes: Cut
    no: Cut

    def __post_init__(self):
        _require(isinstance(self.cond, FlowerIs), f"an If tests FlowerIs(colour), not {_shown(self.cond)}")
        for branch in (self.yes, self.no):
            _require(isinstance(branch, Cut), f"an If branch is a Cut, not {_shown(branch)}")


Stmt = Cut | If  # a statement: a whole program
Node = Stmt | Sel | FlowerIs  # a program or a part of one
_NODE_TYPES = {kind.__name__: kind for kind in (Cut, If, All, Rank, Middle, Except, Filter, ReadLabel, FlowerIs)}


def _nesting(sel: Sel) -> int:
    """How deep Except and Filter nest in a selection."""
    if isinstance(sel, Except):
        return 1 + max(_nesting(sel.sel), _nesting(sel.without))
    if isinstance(sel, Filter):
        return 1 + _nesting(sel.sel)
    return 0


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
_MAX_DEPTH = 8  # a program nests at most 5 deep (If, Cut, Except, Except, Rank): stop long before Python's stack does


def parse(text: str) -> Stmt:
    """The program a text writes, in the form `text` writes it; spaces between words are allowed. Raises ProgramError
    for text that is not a whole program: bad syntax, an unknown word, a value out of range, a selection where a
    statement belongs (or the other way round), nesting beyond the bounds."""
    tokens = _TOKEN.findall(text)
    node, end = _read(tokens, 0, depth=1)
    if end < len(tokens):
        raise ProgramError(f"unexpected {tokens[end]!r} after the program")
    if not isinstance(node, Stmt):
        raise ProgramError(f"a program is a Cut or an If, not {type(node).__name__}")
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


def equivalent(a: Node, b: Node) -> bool:
    """True when two programs (or selections) have the same result on every plant of the benchmark: 2 and 3 shoots,
    every flower colour, every card colour the count allows and no card. A result is the positions selected, a
    shoot the plant does not have, or the fact needed. No plant here has attributes, so a Filter over any shoot
    needs one: two programs that differ only in the value a Filter keeps count as equivalent."""
    return all(_outcome(a, *plant) == _outcome(b, *plant) for plant in _PLANTS)


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

# Every character a program can contain.
ALPHABET: frozenset[str] = frozenset(
    char
    for rules in GRAMMAR.values()
    for rule in rules
    for symbol in rule
    if not symbol.startswith("<")
    for char in symbol
)

# The acceptor reads a prefix one character at a time and keeps every configuration it can be in: the symbols still
# to be written, first symbol first, where a terminal that is partly written keeps only its rest. Several
# configurations are alive at once where terminals share their first letters ("Rank(" and "ReadLabel()", "pink" and
# "purple", the four "tie,..."); the next characters settle it. A prefix begins some program while any configuration
# is alive, and a text is a whole program when one configuration has nothing left to write. No program continues
# another (each ends with the parenthesis its first word opened), so after a whole program nothing is alive.
_START = frozenset({("<program>",)})


@functools.lru_cache(maxsize=1 << 16)
def _expanded(configurations: frozenset) -> frozenset:
    """The same configurations, each leading nonterminal replaced by its rules until a terminal leads. A
    configuration with nothing left to write (a whole program) has no next character and is dropped."""
    out, todo = set(), list(configurations)
    while todo:
        configuration = todo.pop()
        if configuration and configuration[0].startswith("<"):
            todo.extend(rule + configuration[1:] for rule in GRAMMAR[configuration[0]])
        elif configuration:
            out.add(configuration)
    return frozenset(out)


@functools.lru_cache(maxsize=1 << 16)
def _step(configurations: frozenset, char: str) -> frozenset:
    """The configurations after one more character."""
    out = set()
    for terminal, *rest in _expanded(configurations):
        if terminal[0] == char:
            out.add((terminal[1:], *rest) if len(terminal) > 1 else tuple(rest))
    return frozenset(out)


def _configurations(prefix: str) -> frozenset:
    # Both steps are cached: a decoder that adds one character at a time pays a dictionary lookup per character.
    configurations = _START
    for char in prefix:
        if not configurations:
            break
        configurations = _step(configurations, char)
    return configurations


def is_complete(text: str) -> bool:
    """True when `text` is a whole program within the bounds, in canonical form (no spaces)."""
    return () in _configurations(text)


def next_chars(prefix: str) -> set[str]:
    """Every character c such that prefix + c begins some program within the bounds (canonical form). Empty for a
    prefix that no program begins with, and after a whole program."""
    return {configuration[0][0] for configuration in _expanded(_configurations(prefix))}


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
    camera), None for the rest (hold)."""

    what: str
    subgoal: str | None


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
