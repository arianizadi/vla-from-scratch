"""The program writer for v3's language (tutorial/architecture-v3.md, step 5 and E15).

The writer of v2 (program_writer.py) with v3's language: the same language model as downloaded (Qwen3-VL-2B, not
fine-tuned), greedy decoding held to the grammar (here v3's, program.GRAMMAR_V3), a description of the language
construct by construct (SYSTEM_V3) and worked examples from training instructions only. What v3 changes:

- **The language.** CutFrom, Then, Nothing, Take and Count, and the pairs tie,any and sick,no. The description of the new
  constructs uses no word that only a held-out instruction uses (tested), except the constructs' own names.
- **Sixteen worked examples, in pairs** that differ in a word or two and in the program ("from the top down" and "the
  first from the top"; "with a bud" and "without a bud"; "the sick shoots" and "the highest sick shoot"), chosen so that
  every new construct appears (EXAMPLES). Eight examples are the first of each pair.
- **Two disjoint sets** of examples, "A" and "B": the same pairs written with other training instructions. Asked twice
  (`--twice` here, slow_server.py --twice in the closed loop), the writer writes with both, and if the two programs are
  not equivalent the slow level does not act ("the instruction is read two ways").

The instructions it is evaluated on (`draw`): every situation built on prune, as the situation draws its instruction
(scenarios.py: the training sentence frames and the held-out ones, `<situation>_heldout`), and the compositions of rung
5: a first set (COMPOSITIONS) that supplies the examples' constructs, and a second set (HELDOUT_COMPOSITIONS) that
combines the same constructs in ways no example shows, in sentence frames of its own. A plant is drawn with each task
as the situation draws one (two or three shoots, the ties, sick shoots and buds), and the task's own program is the one
the situation runs (program.from_scenario's) or the composition's.

Scored as v2's writer is (`score`): format (it parses in v3's language), exact (its canonical text is the task's
program), meaning (equivalent in v3: the same sequences of cuts allowed on every plant, program.equivalent_v3) and
resolved (on the plant drawn, the same sequences of cuts allowed), by situation and split. The model's output depends
only on the instruction, so each distinct instruction is written once and its program scored on every task that
carries it.

  python -m sim.pickup.program_writer_v3 --model Qwen/Qwen3-VL-2B-Thinking --out writer_v3_A16.json
  python -m sim.pickup.program_writer_v3 --model ... --examples B --out writer_v3_B16.json
  python -m sim.pickup.program_writer_v3 --model ... --shots 8 --unconstrained --out free_A8.json
  python -m sim.pickup.program_writer_v3 --twice writer_v3_A16.json writer_v3_B16.json
"""

import argparse
import functools
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import program, reasoning
from .program import All, Count, Cut, CutFrom, Filter, If, Nothing, Rank, Take, Then
from .program_writer import ProgramWriter, load_lm, table

SPLITS = ("train", "heldout")
FLAGS = ("format", "exact", "meaning", "resolved")
SITUATIONS = ("prune", "nothing_to_do", "order", "ties", "sick", "buds", "compose")
# The longest program a v3 writer is let write: far above any here (the longest task program has 103 characters), and
# a token writes at least one character. A program cut short by it is not complete and fails `format`.
MAX_NEW_TOKENS = 400
PRUNE_INSTRUCTION = "cut the side shoots off the stem, keep the flower"  # env.TASKS["prune"]; nothing_to_do's too

# --------------------------------------------------------------------------------------------------------- the prompt
# Words: none that only held-out instructions use (held_out_only_words), but the constructs' names. Two versions: the
# first, and the second, revised on the first's errors on training instructions (E15): "the middle one" written as
# Rank(2,top), "the highest first, then the next one down" as two cuts, sick and bud read with the wrong value.
SYSTEM_FIRST = """You write programs for a robot that prunes a potted plant: a main stem with a flower on top, and two \
side shoots, sometimes three, lower down the stem. A program says which side shoots the robot must cut, and in what \
order. Reply with the program for the instruction alone, written without spaces.

Programs:
P: one statement.
Then(P1,P2): carry out P1, then P2. P2 can itself start with Then.
Statements:
Cut(S): cut the side shoots that the selection S picks, in any order.
CutFrom(E,S): cut the shoots in S one after another, starting with end E (top, bottom). CutFrom(top,All()) cuts every \
shoot from the top down.
Nothing(): cut no shoot.
If(Q,P1,P2): if the condition Q holds, carry out P1, otherwise P2.
Conditions:
FlowerIs(C): the flower is C (red, orange, pink, purple).
Count(M,N,S): the selection S has exactly N shoots (M is exactly), more than N (M is more), fewer than N (M is fewer). \
N is 0, 1, 2, 3.
Selections:
All(): every side shoot.
Rank(K,top): the K-th shoot counting from the top (K is 1, 2, 3). Rank(1,top) is the highest shoot.
Rank(K,bottom): the K-th shoot counting from the bottom. Rank(1,bottom) is the lowest shoot.
Middle(): the middle shoot, on a plant with three.
Except(S1,S2): the shoots in S1 except those in S2.
Filter(A,V,S): the shoots in S whose attribute A is V. The pairs A,V: tie,red; tie,blue; tie,yellow; tie,white; \
tie,any (a tie, any colour); sick,yes; sick,no (healthy); bud,yes; bud,no.
Take(K,top,S): the K highest shoots in S. Take(K,bottom,S): the K lowest shoots in S. Take(K,any,S): any K shoots in \
S. Take(1,top,Filter(sick,yes,All())) is the highest sick shoot.
ReadLabel(): the shoot that the label in the pot names."""


SYSTEM_V3 = """You write programs for a robot that prunes a potted plant: a main stem with a flower on top, and two \
side shoots, sometimes three, lower down the stem. A program says which side shoots the robot must cut, and in what \
order. Reply with the program for the instruction alone, written without spaces.

Programs:
P: one statement.
Then(P1,P2): carry out P1, then P2. P2 can itself start with Then. Then(Cut(Middle()),Cut(Rank(1,top))) cuts the \
middle shoot, then the highest one.
Statements:
Cut(S): cut the side shoots that the selection S picks, in any order.
CutFrom(E,S): cut every shoot in S one after another, starting with end E (top, bottom). CutFrom(top,All()) cuts every \
shoot from the top down: the highest first, then the next one down, and so on.
Nothing(): cut no shoot.
If(Q,P1,P2): if the condition Q holds, carry out P1, otherwise P2.
Conditions:
FlowerIs(C): the flower is C (red, orange, pink, purple).
Count(M,N,S): the selection S has exactly N shoots (M is exactly), more than N (M is more), fewer than N (M is fewer). \
N is 0, 1, 2, 3.
Selections:
All(): every side shoot.
Rank(K,top): the K-th shoot counting from the top (K is 1, 2, 3). Rank(1,top) is the highest shoot.
Rank(K,bottom): the K-th shoot counting from the bottom. Rank(1,bottom) is the lowest shoot.
Middle(): the middle shoot. Write Middle() for the middle shoot, never Rank(2,top).
Except(S1,S2): the shoots in S1 except those in S2.
Filter(A,V,S): the shoots in S whose attribute A is V. The pairs A,V: tie,red; tie,blue; tie,yellow; tie,white; \
tie,any (a tie, any colour); sick,yes (sick, diseased); sick,no (healthy); bud,yes (with a bud); bud,no (without a \
bud, no bud). A selection picks the shoots cut, never those kept: keep the healthy ones, cut the other shoots is \
Cut(Filter(sick,yes,All())).
Take(K,top,S): the K highest shoots in S. Take(K,bottom,S): the K lowest shoots in S. Take(K,any,S): any K shoots in \
S. Take(1,top,Filter(sick,yes,All())) is the highest sick shoot.
ReadLabel(): the shoot that the label in the pot names."""


def _sick(value="yes"):
    return Filter("sick", value, All())


def _bud(value):
    return Filter("bud", value, All())


def _tie(value):
    return Filter("tie", value, All())


LOWEST, HIGHEST = Rank(1, "bottom"), Rank(1, "top")

# The compositions of rung 5. A first set, whose instructions are training instructions (the examples use some of
# them), and a second set, held out: each a combination of parts that occur in training and that no example or training
# composition combines so, in sentence frames of its own.
COMPOSITIONS = (
    (Cut(Take(1, "bottom", _bud("yes"))), ("cut the lowest shoot that has a bud", "prune the lowest shoot with a bud")),
    (Cut(Take(1, "any", All())), ("cut one shoot", "prune any one side shoot")),
    (
        If(Count("exactly", 0, _sick()), Cut(LOWEST), Nothing()),
        (
            "if no shoot is sick, cut the lowest shoot, otherwise cut none",
            "cut the lowest shoot if no shoot is sick, else cut none",
        ),
    ),
    (Cut(_bud("yes")), ("cut every shoot with a bud", "prune the shoots that have a bud")),
    (Cut(_sick("no")), ("cut the healthy shoots, keep the sick ones", "prune every healthy shoot")),
    (Cut(Take(1, "top", _sick())), ("cut the highest sick shoot", "prune the highest shoot that is sick")),
    (
        CutFrom("bottom", _tie("any")),
        ("cut the tied shoots from the bottom up", "prune the shoots with a tie, starting with the lowest and working up"),
    ),
    (
        Then(Cut(_tie("blue")), Cut(HIGHEST)),
        ("cut the shoot with the blue tie, then the highest one", "first cut the shoot tied with blue, after that the highest one"),
    ),
    (
        If(Count("more", 1, _bud("yes")), Cut(HIGHEST), Nothing()),
        (
            "if more than one shoot has a bud, cut the highest shoot, otherwise cut none",
            "cut the highest shoot if more than one shoot has a bud, else cut none",
        ),
    ),
    (Cut(Take(2, "bottom", All())), ("cut the two lowest shoots", "prune the two lowest side shoots")),
    (
        If(Count("fewer", 2, _bud("yes")), Cut(LOWEST), Nothing()),
        (
            "if fewer than two shoots have a bud, cut the lowest shoot, otherwise cut none",
            "cut the lowest shoot if fewer than two shoots have a bud, else cut none",
        ),
    ),
    (
        If(Count("exactly", 1, _sick()), Cut(_sick()), Nothing()),
        (
            "if exactly one shoot is sick, cut the sick shoot, otherwise cut none",
            "cut the sick shoot if exactly one shoot is sick, else cut none",
        ),
    ),
)
HELDOUT_COMPOSITIONS = (
    (
        Cut(Take(1, "top", _bud("no"))),
        ("cut the highest shoot that has no bud", "of the shoots without a bud, the highest one has to go"),
    ),
    (
        CutFrom("top", _sick()),
        ("cut the sick shoots from the top down", "snip the sick branches one by one, top to bottom"),
    ),
    (Cut(program.Except(All(), Take(1, "any", All()))), ("leave exactly one shoot", "keep a single shoot and cut the others")),
    (Cut(Take(2, "any", All())), ("cut two shoots", "two of the shoots have to go, whichever")),
    (
        If(Count("more", 1, _sick()), Cut(Take(1, "bottom", _sick())), Nothing()),
        (
            "if more than one shoot is sick, cut only the lowest sick one; otherwise cut none",
            "should two or more shoots be sick, snip the lowest sick branch and nothing else",
        ),
    ),
    (
        CutFrom("bottom", _bud("no")),
        ("cut the shoots without a bud from the bottom up", "begin at the bottom and remove each branch that has no bud"),
    ),
    (
        Then(Cut(_tie("red")), Cut(LOWEST)),
        ("cut the shoot with the red tie, then the lowest one", "the shoot with the red tie goes first and the lowest shoot second"),
    ),
    (
        If(Count("fewer", 2, _sick()), Cut(_sick()), Cut(All())),
        (
            "if fewer than two shoots are sick, cut the sick ones, otherwise cut every shoot",
            "unless two or more shoots are sick, remove just the sick ones; if they are, remove them all",
        ),
    ),
)


def _t(node) -> str:
    return program.text(node)


# Worked examples, in pairs, two disjoint sets of training instructions (A and B). Each pair differs in a word or two
# and in its program; together they show every new construct. EXAMPLES_FIRST are the first version's.
EXAMPLES_FIRST = {
    "A": (
        ("cut the side shoots from the top down", CutFrom("top", All())),
        ("cut the first shoot from the top", Cut(HIGHEST)),
        ("cut the lowest shoot, then the highest one", Then(Cut(LOWEST), Cut(HIGHEST))),
        ("cut the side shoots from the bottom up", CutFrom("bottom", All())),
        ("cut the shoot with the red tie", Cut(_tie("red"))),
        ("cut the shoot with the tie", Cut(_tie("any"))),
        ("cut every shoot without a bud", Cut(_bud("no"))),
        ("cut every shoot with a bud", Cut(_bud("yes"))),
        ("cut the sick shoots, keep the healthy ones", Cut(_sick())),
        ("cut the highest sick shoot", Cut(Take(1, "top", _sick()))),
        ("cut every side shoot except the lowest one", Cut(program.Except(All(), LOWEST))),
        ("cut the side shoots off the stem, keep the flower", Cut(All())),
        ("cut one shoot", Cut(Take(1, "any", All()))),
        ("cut the lowest shoot that has a bud", Cut(Take(1, "bottom", _bud("yes")))),
        (
            "if no shoot is sick, cut the lowest shoot, otherwise cut none",
            If(Count("exactly", 0, _sick()), Cut(LOWEST), Nothing()),
        ),
        (
            "if the flower is pink, cut the lowest shoot, otherwise cut the highest one",
            If(program.FlowerIs("pink"), Cut(LOWEST), Cut(HIGHEST)),
        ),
    ),
    "B": (
        ("prune the shoots starting with the highest and working down", CutFrom("top", All())),
        ("prune the first side shoot counting from the top", Cut(HIGHEST)),
        ("first cut the highest shoot, after that the lowest one", Then(Cut(HIGHEST), Cut(LOWEST))),
        ("prune the shoots starting with the lowest and working up", CutFrom("bottom", All())),
        ("prune the shoot tied with blue", Cut(_tie("blue"))),
        ("prune the tied shoot", Cut(_tie("any"))),
        ("prune the shoots that have no bud", Cut(_bud("no"))),
        ("prune the shoots that have a bud", Cut(_bud("yes"))),
        ("prune every sick shoot", Cut(_sick())),
        ("prune every healthy shoot", Cut(_sick("no"))),
        ("prune all the shoots but the highest one", Cut(program.Except(All(), HIGHEST))),
        ("cut the tied shoots from the bottom up", CutFrom("bottom", _tie("any"))),
        ("prune any one side shoot", Cut(Take(1, "any", All()))),
        ("prune the lowest shoot with a bud", Cut(Take(1, "bottom", _bud("yes")))),
        (
            "cut the lowest shoot if no shoot is sick, else cut none",
            If(Count("exactly", 0, _sick()), Cut(LOWEST), Nothing()),
        ),
        ("cut the shoot with the blue tie, then the highest one", Then(Cut(_tie("blue")), Cut(HIGHEST))),
    ),
}


MIDDLE = program.Middle()
EXAMPLES = {
    "A": (
        ("cut the side shoots from the top down", CutFrom("top", All())),
        ("cut the lowest shoot first, then the next one up", CutFrom("bottom", All())),
        ("cut the middle shoot, then the highest one", Then(Cut(MIDDLE), Cut(HIGHEST))),
        ("cut the highest shoot, then the middle one", Then(Cut(HIGHEST), Cut(MIDDLE))),
        ("cut the shoot with the red tie", Cut(_tie("red"))),
        ("cut the shoot with the tie", Cut(_tie("any"))),
        ("keep the buds, cut the other shoots", Cut(_bud("no"))),
        ("cut every shoot with a bud", Cut(_bud("yes"))),
        ("remove the diseased shoots", Cut(_sick())),
        ("cut the healthy shoots, keep the sick ones", Cut(_sick("no"))),
        ("cut the highest sick shoot", Cut(Take(1, "top", _sick()))),
        ("cut the two lowest shoots", Cut(Take(2, "bottom", All()))),
        ("cut one shoot", Cut(Take(1, "any", All()))),
        ("cut every side shoot except the lowest one", Cut(program.Except(All(), LOWEST))),
        (
            "if no shoot is sick, cut the lowest shoot, otherwise cut none",
            If(Count("exactly", 0, _sick()), Cut(LOWEST), Nothing()),
        ),
        ("cut the side shoots off the stem, keep the flower", Cut(All())),
    ),
    "B": (
        ("prune the shoots starting with the highest and working down", CutFrom("top", All())),
        ("prune the shoots starting with the lowest and working up", CutFrom("bottom", All())),
        ("first cut the middle shoot, after that the lowest one", Then(Cut(MIDDLE), Cut(LOWEST))),
        ("start with the lowest shoot, then cut the middle shoot", Then(Cut(LOWEST), Cut(MIDDLE))),
        ("prune the shoot tied with blue", Cut(_tie("blue"))),
        ("prune the tied shoot", Cut(_tie("any"))),
        ("prune the shoots that have no bud", Cut(_bud("no"))),
        ("prune the shoots that have a bud", Cut(_bud("yes"))),
        ("prune every sick shoot", Cut(_sick())),
        ("prune every healthy shoot", Cut(_sick("no"))),
        ("prune the highest shoot that is sick", Cut(Take(1, "top", _sick()))),
        ("prune the two lowest side shoots", Cut(Take(2, "bottom", All()))),
        ("prune any one side shoot", Cut(Take(1, "any", All()))),
        ("prune all the shoots but the highest one", Cut(program.Except(All(), HIGHEST))),
        (
            "if fewer than two shoots have a bud, cut the lowest shoot, otherwise cut none",
            If(Count("fewer", 2, _bud("yes")), Cut(LOWEST), Nothing()),
        ),
        ("cut the shoot with the blue tie, then the highest one", Then(Cut(_tie("blue")), Cut(HIGHEST))),
    ),
}
PROMPTS = {"first": (SYSTEM_FIRST, EXAMPLES_FIRST), "second": (SYSTEM_V3, EXAMPLES)}


def examples(shots: int = 16, which: str = "A", prompt: str = "second") -> list[tuple[str, str]]:
    """The worked examples of set `which` (of the prompt's version): all sixteen, or (shots=8) the first of each
    pair."""
    chosen = PROMPTS[prompt][1]
    assert shots in (8, 16) and which in chosen, (shots, which)
    pairs = [(said, _t(node)) for said, node in chosen[which]]
    return pairs if shots == 16 else pairs[::2]


def writer(
    model,
    tokenizer,
    shots: int = 16,
    which: str = "A",
    constrained: bool = True,
    max_new_tokens=None,
    prompt="second",
    think: int = 0,
    think_temperature: float = 0.0,
):
    """A ProgramWriter for v3's language, with the description and the examples of set `which` of the prompt's
    version (PROMPTS: "second", the one used, or "first")."""
    return ProgramWriter(
        model,
        tokenizer,
        constrained=constrained,
        max_new_tokens=MAX_NEW_TOKENS if max_new_tokens is None else max_new_tokens,
        language="v3",
        system=PROMPTS[prompt][0],
        pairs=examples(shots, which, prompt),
        think=think,
        think_temperature=think_temperature,
    )


# ------------------------------------------------------------------------------------------------------------ the tasks
@dataclass(frozen=True)
class TendTask:
    """An instruction of a situation (or a composition), the plant it was drawn with and the task's own program."""

    situation: str
    split: str  # "train" or "heldout"
    phrasing: str  # "train"; held out: "frame" (a held-out sentence frame) or "composition" (rung 5's second set)
    mode: str
    instruction: str
    count: int
    attributes: tuple  # per shoot, lowest first: {"tie": ..., "sick": ..., "bud": ...}
    program: str  # the task's own, canonical text

    @property
    def node(self):
        return program.parse(self.program, "v3")


@functools.cache
def _frames() -> dict:
    """The situations' sentence frames, by situation and split (scenarios.py; imported here, it needs MuJoCo)."""
    from . import scenarios as S

    return {
        "order": {"train": S.ORDER_FRAMES, "heldout": S.ORDER_HELDOUT_FRAMES},
        "ties": {"train": S.TIE_FRAMES, "heldout": S.TIE_HELDOUT_FRAMES},
        "sick": {"train": {"sick": S.SICK_FRAMES}, "heldout": {"sick": S.SICK_HELDOUT_FRAMES}},
        "buds": {"train": {"buds": S.BUD_FRAMES}, "heldout": {"buds": S.BUD_HELDOUT_FRAMES}},
        "tie_colours": tuple(S.manifest()["assets"]["tie"]["colours"]),
    }


def _plain(count: int) -> list[dict]:
    return [{"tie": "none", "sick": "no", "bud": "no"} for _ in range(count)]


def _pick(rng, options):
    return options[int(rng.integers(len(options)))]


def draw(situation: str, split: str, rng) -> TendTask:
    """A task of `situation` from its training or held-out instructions, with a plant drawn as the situation draws it.
    prune and nothing_to_do have one instruction, the task's own, and no held-out one."""
    assert situation in SITUATIONS and split in SPLITS, (situation, split)
    count = int(rng.integers(2, 4))
    attributes = _plain(count)
    names = [reasoning.position_name(i, count) for i in range(count)]
    if situation in ("prune", "nothing_to_do"):
        assert split == "train", f"{situation} has no held-out instruction"
        node, mode, said, phrasing = Cut(All()), "pruned" if situation == "nothing_to_do" else "plain", PRUNE_INSTRUCTION, "train"
    elif situation == "compose":
        sets = HELDOUT_COMPOSITIONS if split == "heldout" else COMPOSITIONS
        node, frames = _pick(rng, sets)
        said, mode = _pick(rng, frames), _t(node)
        phrasing = "composition" if split == "heldout" else "train"
        for shoot in attributes:  # every attribute drawn at random: the compositions read any of them
            shoot["tie"] = _pick(rng, ("none", *_frames()["tie_colours"]))
            shoot["sick"] = _pick(rng, ("no", "yes"))
            shoot["bud"] = _pick(rng, ("no", "yes"))
    else:
        frames = _frames()[situation][split]
        phrasing = "frame" if split == "heldout" else "train"
        if situation == "order":
            mode = _pick(rng, ("top_down", "bottom_up", "pair"))
            frame = _pick(rng, frames[mode])
            if mode == "pair":
                i, j = (int(x) for x in rng.choice(count, size=2, replace=False))
                said = frame.format(a=names[i], b=names[j])
                node = Then(Cut(program._NAMED[names[i]]), Cut(program._NAMED[names[j]]))
            else:
                said, node = frame, program._SITUATION_PROGRAMS[("order", mode)]
        elif situation == "ties":
            mode = _pick(rng, ("colour", "one"))
            colours = _frames()["tie_colours"]
            target = int(rng.integers(count))
            tied = range(count) if mode == "colour" else [target]
            picked = [colours[k] for k in rng.choice(len(colours), size=len(tied), replace=False)]
            for i, colour in zip(tied, picked, strict=True):
                attributes[i]["tie"] = colour
            colour = attributes[target]["tie"]
            said = _pick(rng, frames[mode]).format(c=colour)
            node = Cut(_tie(colour if mode == "colour" else "any"))
        else:  # sick, buds: none to every shoot marked
            mode = situation
            attr = "sick" if situation == "sick" else "bud"
            marked = rng.choice(count, size=int(rng.integers(count + 1)), replace=False)
            for i in marked:
                attributes[int(i)][attr] = "yes"
            said = _pick(rng, frames[mode])
            node = Cut(_sick()) if situation == "sick" else Cut(_bud("no"))
    return TendTask(situation, split, phrasing, mode, said, count, tuple(attributes), _t(node))


def draw_tasks(n: int, seed: int = 0) -> list[TendTask]:
    """`n` tasks per situation and split (prune and nothing_to_do: training only)."""
    rng = np.random.default_rng([seed, 3])
    return [
        draw(situation, split, rng)
        for situation in SITUATIONS
        for split in SPLITS
        if split == "train" or situation not in ("prune", "nothing_to_do")
        for _ in range(n)
    ]


def training_instructions() -> set[str]:
    """Every training instruction of v3: the situations' training frames, the first set of compositions, and v2's."""
    frames = _frames()
    out = {PRUNE_INSTRUCTION}
    names = ("lowest", "middle", "highest")
    for mode, options in frames["order"]["train"].items():
        out |= {f.format(a=a, b=b) for f in options for a in names for b in names if a != b} if mode == "pair" else set(options)
    out |= {f.format(c=c) for fs in frames["ties"]["train"].values() for f in fs for c in frames["tie_colours"]}
    out |= set(frames["sick"]["train"]["sick"]) | set(frames["buds"]["train"]["buds"])
    out |= {said for _, frames_ in COMPOSITIONS for said in frames_}
    return out | set().union(*(reasoning.training_instructions(kind) for kind in reasoning.KINDS))


def heldout_instructions() -> set[str]:
    """Every held-out instruction of v3: the situations' held-out frames and the second set of compositions."""
    frames = _frames()
    out = set()
    names = ("lowest", "middle", "highest")
    for mode, options in frames["order"]["heldout"].items():
        out |= {f.format(a=a, b=b) for f in options for a in names for b in names if a != b} if mode == "pair" else set(options)
    out |= {f.format(c=c) for fs in frames["ties"]["heldout"].values() for f in fs for c in frames["tie_colours"]}
    out |= set(frames["sick"]["heldout"]["sick"]) | set(frames["buds"]["heldout"]["buds"])
    return out | {said for _, frames_ in HELDOUT_COMPOSITIONS for said in frames_}


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def held_out_only_words() -> set[str]:
    """Words that v3's held-out instructions use and no training instruction (v3's or v2's) does."""
    train = set().union(*(words(t) for t in training_instructions()))
    return set().union(*(words(t) for t in heldout_instructions())) - train


# ------------------------------------------------------------------------------------------------------------- scoring
def _outcome(node, task: TendTask):
    plant = program._Shoots(task.count, attributes=task.attributes)
    return program._outcome_v3(node, plant)


def score(written: str, task: TendTask) -> dict:
    """format (it parses in v3), exact (its canonical text is the task's), meaning (equivalent in v3) and resolved (the
    same sequences of cuts allowed on the plant drawn); `got`: the stages it evaluates to there, or why none."""
    gold = task.node
    try:
        node = program.parse(written, "v3")
    except program.ProgramError:
        return dict(format=False, exact=False, meaning=False, resolved=False, got="not a program")
    try:
        got = [list(map(list, stages)) for stages in program.stages_on(node, task.count, attributes=task.attributes)]
    except program.NeedsObservation as need:
        got = f"needs the {need.what}"
    except program.ProgramError as error:
        got = str(error)
    return dict(
        format=True,
        exact=program.text(node) == task.program,
        meaning=program.equivalent_v3(node, gold),
        resolved=_outcome(node, task) == _outcome(gold, task),
        got=got,
    )


def evaluate(write, tasks: Sequence[TendTask], progress: bool = False) -> list[dict]:
    """One row per task; `write(instruction)` (ProgramWriter.write) is called once per distinct instruction."""
    written: dict[str, dict] = {}
    rows = []
    for task in tasks:
        if task.instruction not in written:
            written[task.instruction] = write(task.instruction)
            if progress:
                out = written[task.instruction]
                print(f"{len(written)} {task.situation} {task.split}: {task.instruction} -> {out['program']}", flush=True)
        out = written[task.instruction]
        rows.append(
            dict(
                situation=task.situation,
                kind=task.situation,
                split=task.split,
                phrasing=task.phrasing,
                mode=task.mode,
                count=task.count,
                attributes=list(task.attributes),
                instruction=task.instruction,
                gold=task.program,
                written=out["program"],
                **score(out["program"], task),
                tokens=out["tokens"],
                seconds=out["seconds"],
                **({"thought": out["thought"]} if "thought" in out else {}),
            )
        )
    return rows


def _counts(rows: list[dict]) -> dict:
    out = {"n": len(rows), **{flag: sum(bool(row[flag]) for row in rows) for flag in FLAGS}}
    out["tokens"] = sum(row["tokens"] for row in rows) / len(rows)
    out["seconds"] = sum(row["seconds"] for row in rows) / len(rows)
    out["instructions"] = len({row["instruction"] for row in rows})
    return out


def summarize(rows: list[dict]) -> dict[str, dict]:
    """The flags counted over all rows, by split, by situation and split, and the held-out ones by phrasing."""
    groups = {"all": rows}
    for split in SPLITS:
        groups[split] = [row for row in rows if row["split"] == split]
    groups["heldout/frame"] = [row for row in rows if row["phrasing"] == "frame"]
    groups["heldout/composition"] = [row for row in rows if row["phrasing"] == "composition"]
    for situation in SITUATIONS:
        for split in SPLITS:
            groups[f"{situation}/{split}"] = [r for r in rows if (r["situation"], r["split"]) == (situation, split)]
    return {name: _counts(group) for name, group in groups.items() if group}


def twice(first: list[dict], second: list[dict]) -> dict:
    """Asking twice, from the rows of the same tasks written with two sets of examples: the slow level acts only when
    the two programs are equivalent. Counts, over all rows and by group: `agree` (it acts), `agree_right` (and the
    first program means the task's), `agree_wrong` (both programs agree and are wrong), `blocked_right` (the two
    disagree and the first was right: a right program blocked), `blocked_wrong` (disagree, first wrong: a wrong program
    stopped), and `first_wrong` (wrong programs the first writer alone would have run)."""
    assert [r["instruction"] for r in first] == [r["instruction"] for r in second], "not the same tasks"

    def agreeing(a: str, b: str) -> bool:
        try:
            return program.equivalent_v3(program.parse(a, "v3"), program.parse(b, "v3"))
        except program.ProgramError:
            return False

    cache: dict[tuple[str, str], bool] = {}
    rows = []
    for a, b in zip(first, second, strict=True):
        key = (a["written"], b["written"])
        if key not in cache:
            cache[key] = agreeing(*key)
        rows.append(dict(a, agree=cache[key], second_meaning=b["meaning"]))

    def counts(group):
        return dict(
            n=len(group),
            agree=sum(r["agree"] for r in group),
            agree_right=sum(r["agree"] and r["meaning"] for r in group),
            agree_wrong=sum(r["agree"] and not r["meaning"] for r in group),
            blocked_right=sum(not r["agree"] and r["meaning"] for r in group),
            blocked_wrong=sum(not r["agree"] and not r["meaning"] for r in group),
            first_wrong=sum(not r["meaning"] for r in group),
        )

    groups = {"all": rows, **{s: [r for r in rows if r["split"] == s] for s in SPLITS}}
    for situation in SITUATIONS:
        for split in SPLITS:
            group = [r for r in rows if (r["situation"], r["split"]) == (situation, split)]
            if group:
                groups[f"{situation}/{split}"] = group
    return {name: counts(group) for name, group in groups.items()}


# ---------------------------------------------------------------------------------------------------- the command line
def main(argv: Sequence[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(
        description="Write v3 programs for the situations' instructions and the compositions with a language model "
        "that is not fine-tuned; score format, exact text, meaning and the shoots resolved (program_writer_v3.py)."
    )
    ap.add_argument("--model", help="a Hugging Face model name or a local directory")
    ap.add_argument("--n", type=int, default=100, help="tasks per situation and split")
    ap.add_argument("--shots", type=int, choices=(8, 16), default=16, help="worked examples: 16 in pairs, or 8")
    ap.add_argument("--examples", choices=sorted(EXAMPLES), default="A", help="the set of worked examples")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="second", help="the prompt's version")
    ap.add_argument("--think", type=int, default=0, help="tokens of free reasoning before the program (0: none)")
    ap.add_argument("--think_temperature", type=float, default=0.0, help="the reasoning sampled at it (0: greedy)")
    ap.add_argument("--think_batch", type=int, default=16, help="instructions reasoned about at once")
    ap.add_argument("--lora", help="a directory of writer_tune.py's adapters, merged into the model")
    ap.add_argument("--seed", type=int, default=0, help="the tasks")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--unconstrained", action="store_true", help="the baseline: the model writes freely")
    ap.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--twice", nargs=2, metavar="REPORT", help="asking twice, from two reports of the same tasks")
    ap.add_argument("--out", help="the JSON report: the summary and every row")
    args = ap.parse_args(argv)
    if args.twice:
        first, second = (json.load(open(path)) for path in args.twice)
        report = dict(mode="twice", reports=args.twice, summary=twice(first["rows"], second["rows"]))
        for name, counts in report["summary"].items():
            print(name, counts, flush=True)
    else:
        assert args.model, "--model is needed to write"
        model, tokenizer = load_lm(args.model, device=args.device, lora=args.lora)
        chosen = writer(
            model, tokenizer, args.shots, args.examples, not args.unconstrained, args.max_new_tokens, args.prompt,
            args.think, args.think_temperature,
        )
        tasks = draw_tasks(args.n, args.seed)
        chosen.think_many([task.instruction for task in tasks], batch=args.think_batch, seed=args.seed)
        rows = evaluate(chosen.write, tasks, progress=True)
        summary = summarize(rows)
        print(table({k: {**v, **{f: v[f] for f in FLAGS}} for k, v in summary.items()}), flush=True)
        report = dict(
            mode="unconstrained" if args.unconstrained else "constrained",
            model=args.model,
            n=args.n,
            shots=args.shots,
            examples_set=args.examples,
            prompt=args.prompt,
            think=args.think,
            think_temperature=args.think_temperature,
            lora=args.lora,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            examples=[list(example) for example in chosen.examples],
            summary=summary,
            rows=rows,
        )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)
            f.write("\n")
    return report


if __name__ == "__main__":
    main()
