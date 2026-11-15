"""Reasoning tasks on the pruning plant (tasks reason_*): instructions that a lookup from words to a shoot cannot solve.

In `prune_named` the instruction names a shoot by its height ("the lowest shoot"): one word, one shoot. Here the shoot
to cut follows from the instruction and something about the plant that has to be seen, counted or remembered first.
There are four kinds (KINDS):

  ordinal      "cut the second shoot from the top": count the shoots, then count down. With three shoots that is the
               middle one, with two the lowest.
  conditional  "if the flower is red, cut the lowest shoot, otherwise cut the highest one": look at the flower, compare
               its colour with the one named, take the branch that holds.
  exclusion    "cut every side shoot except the highest one": the shoots to cut are the complement of the one named,
               one or two of them depending on the plant, and the robot has to keep track of which are done.
  hidden       "read the label in the pot and cut the shoot it names": a card in the pot names the shoot by its colour
               (LABEL_MEANS), and the card faces a random way. If it faces away from the front camera the robot first
               turns the plant to read it; turning the shoot into reach afterwards usually hides the card again, so
               what it said has to be remembered.

Every kind has training instructions and held-out ones (split "heldout"). A held-out instruction is a composition no
demonstration contains ("composition": e.g. "third from the top" when training has "third from the bottom" and
"second from the top"), a training composition in a sentence frame no demonstration uses ("frame"), or both ("both").
The hidden kind has only held-out frames.

This module is pure Python: it draws a task for a plant (number of shoots, flower colour) and writes the expert's
notes (`note_text`), short lines of reasoning with a fixed layout that a parser (`note_fields`) can check. The simulator
side is in env.py (the tasks) and notes.py (when a note is written).
"""

import itertools
from dataclasses import dataclass, field

POSITIONS = ("lowest", "middle", "highest")
KINDS = ("ordinal", "conditional", "exclusion", "hidden")
FLOWERS = ("red", "orange", "pink", "purple")  # env.FLOWER_COLOURS, in that order
# The card in the pot names a shoot by its colour. A fixed convention, like a sign: the demonstrations teach it.
LABEL_MEANS = {"white": "lowest", "yellow": "middle", "cyan": "highest"}
BUTTON_NAMES = {+1: "blue", -1: "orange"}  # the button that turns the plate counter-clockwise (+1) or clockwise (-1)
# Episode length limit per kind (control steps at 20 Hz). The expert needs about 250 steps for one cut with a turn,
# a look adds about 150, and exclusion cuts up to two shoots.
MAX_STEPS = {"ordinal": 450, "conditional": 450, "exclusion": 900, "hidden": 700}

ORDINALS = {1: "first", 2: "second", 3: "third"}
ENDS = ("top", "bottom")
# (k, end): the k-th shoot counting from that end. Held out: "second from the bottom" (training says "second" only
# from the top) and "third from the top" (training says "third" only from the bottom).
# Weights: "second from the top" is the training composition whose shoot depends on the count (the middle one of
# three, the lowest of two), so it is drawn most; the others are lookups once learned.
ORDINAL_TRAIN = {(1, "top"): 0.1, (2, "top"): 0.7, (1, "bottom"): 0.1, (3, "bottom"): 0.1}
ORDINAL_HELDOUT = ((2, "bottom"), (3, "top"))
ORDINAL_FRAMES = (
    "cut the {ord} shoot from the {end}",
    "prune the {ord} side shoot counting from the {end}",
    "remove the shoot that is {ord} from the {end}",
)
ORDINAL_HELDOUT_FRAMES = (
    "snip the {ord} branch from the {end}",
    "counting from the {end}, the {ord} shoot has to go",
)

# (colour named, shoot if it is that colour); the other branch is the other end. Held out: two of the eight.
CONDITIONAL_HELDOUT = (("red", "highest"), ("purple", "lowest"))
CONDITIONAL_FRAMES = (
    "if the flower is {colour}, cut the {yes} shoot, otherwise cut the {no} one",
    "cut the {yes} shoot if the flower is {colour}, else the {no} one",
)
CONDITIONAL_HELDOUT_FRAMES = (
    "should the flower be {colour}, prune the {yes} shoot; if not, prune the {no} one",
    "unless the flower is {colour}, remove the {no} shoot; if it is, remove the {yes} one",
)

EXCLUSION_TRAIN = ("lowest", "highest")
EXCLUSION_HELDOUT = ("middle",)  # three-shoot plants only: cut the two ends
EXCLUSION_FRAMES = (
    "cut every side shoot except the {keep} one",
    "prune all the shoots but the {keep} one",
)
EXCLUSION_HELDOUT_FRAMES = (
    "leave only the {keep} shoot and cut the rest",
    "remove all side shoots apart from the {keep} one",
)

HIDDEN_FRAMES = (
    "read the label in the pot and cut the shoot it names",
    "cut the shoot that the label in the pot names",
)
HIDDEN_HELDOUT_FRAMES = (
    "check the tag in the pot, then prune the shoot it points to",
    "the card in the pot says which shoot to remove: remove that one",
)
HELDOUT_KINDS = ("composition", "frame", "both")


@dataclass(frozen=True)
class Task:
    kind: str
    instruction: str
    phrasing: str  # "train", or the kind of held-out instruction (HELDOUT_KINDS)
    count: int  # shoots on the plant
    targets: tuple[int, ...]  # shoots to cut, by index from the lowest up
    ask: str  # the instruction restated in a fixed form (the notes' "ask" field)
    params: dict = field(default_factory=dict)

    @property
    def target_positions(self) -> tuple[str, ...]:
        return tuple(position_name(i, self.count) for i in self.targets)


def position_name(index: int, count: int) -> str:
    """Name of shoot `index` (0 = the lowest) on a plant with `count` shoots (as phrasings.position_name)."""
    assert 0 <= index < count and 2 <= count <= 3, (index, count)
    return "lowest" if index == 0 else "highest" if index == count - 1 else "middle"


def position_index(name: str, count: int) -> int:
    index = {"lowest": 0, "middle": 1, "highest": count - 1}[name]
    assert name != "middle" or count == 3, f"no middle shoot on a plant with {count}"
    return index


def ordinal_index(k: int, end: str, count: int) -> int:
    """The k-th shoot from `end` (top or bottom), by index from the lowest up."""
    assert 1 <= k <= count, (k, count)
    return k - 1 if end == "bottom" else count - k


def label_options(count: int) -> list[str]:
    """Card colours that name a shoot this plant has (yellow names the middle one)."""
    return [c for c, p in LABEL_MEANS.items() if p != "middle" or count == 3]


def _pick(rng, options):
    return options[int(rng.integers(len(options)))]


def _heldout_kind(rng, has_composition: bool) -> str:
    return _pick(rng, HELDOUT_KINDS) if has_composition else "frame"


def draw(kind: str, split: str, count: int, flower: str, rng) -> Task:
    """A task of `kind` for a plant with `count` shoots and a `flower`-coloured flower, from the training ('train') or
    held-out ('heldout') instructions. Every random choice comes from `rng`."""
    assert kind in KINDS and split in ("train", "heldout") and count in (2, 3) and flower in FLOWERS
    phrasing = "train" if split == "train" else _heldout_kind(rng, kind != "hidden")
    new_composition = phrasing in ("composition", "both")
    new_frame = phrasing in ("frame", "both")
    if kind == "ordinal":
        if new_composition:
            options = [c for c in ORDINAL_HELDOUT if c[0] <= count]
            k, end = _pick(rng, options)
        else:
            options = [c for c in ORDINAL_TRAIN if c[0] <= count]
            weights = [ORDINAL_TRAIN[c] for c in options]
            k, end = options[int(rng.choice(len(options), p=[w / sum(weights) for w in weights]))]
        frame = _pick(rng, ORDINAL_HELDOUT_FRAMES if new_frame else ORDINAL_FRAMES)
        return Task(
            kind,
            frame.format(ord=ORDINALS[k], end=end),
            phrasing,
            count,
            (ordinal_index(k, end, count),),
            f"{ORDINALS[k]} from the {end}",
            dict(k=k, end=end),
        )
    if kind == "conditional":
        pairs = [(c, y) for c in FLOWERS for y in ("lowest", "highest")]
        options = list(CONDITIONAL_HELDOUT) if new_composition else [p for p in pairs if p not in CONDITIONAL_HELDOUT]
        holds = bool(rng.random() < 0.5)  # the condition holds in half of the episodes
        # the colour named: the flower's if the condition holds, another otherwise (the held-out compositions name
        # only two colours: for another flower the condition cannot hold)
        colours = sorted({c for c, _ in options if (c == flower) == holds})
        if not colours:
            holds = not holds
            colours = sorted({c for c, _ in options if (c == flower) == holds})
        colour = _pick(rng, colours)
        yes = _pick(rng, sorted({y for c, y in options if c == colour}))
        no = "lowest" if yes == "highest" else "highest"
        frame = _pick(rng, CONDITIONAL_HELDOUT_FRAMES if new_frame else CONDITIONAL_FRAMES)
        return Task(
            kind,
            frame.format(colour=colour, yes=yes, no=no),
            phrasing,
            count,
            (position_index(yes if holds else no, count),),
            f"if {colour} the {yes}, else the {no}",
            dict(colour=colour, yes=yes, no=no, holds=holds),
        )
    if kind == "exclusion":
        if new_composition and count < 3:  # the held-out composition needs a middle shoot
            phrasing = "frame"
            new_composition, new_frame = False, True
        keep = _pick(rng, EXCLUSION_HELDOUT if new_composition else EXCLUSION_TRAIN)
        kept = position_index(keep, count)
        frame = _pick(rng, EXCLUSION_HELDOUT_FRAMES if new_frame else EXCLUSION_FRAMES)
        return Task(
            kind,
            frame.format(keep=keep),
            phrasing,
            count,
            tuple(i for i in range(count) if i != kept),
            f"all but the {keep}",
            dict(keep=keep),
        )
    label = _pick(rng, label_options(count))
    frame = _pick(rng, HIDDEN_HELDOUT_FRAMES if new_frame else HIDDEN_FRAMES)
    return Task(
        kind,
        frame,
        phrasing,
        count,
        (position_index(LABEL_MEANS[label], count),),
        "the shoot the label names",
        dict(label=label),
    )


def training_instructions(kind: str) -> set[str]:
    """Every instruction a demonstration of `kind` can carry."""
    if kind == "ordinal":
        return {f.format(ord=ORDINALS[k], end=e) for f in ORDINAL_FRAMES for k, e in ORDINAL_TRAIN}
    if kind == "conditional":
        pairs = [(c, y) for c in FLOWERS for y in ("lowest", "highest") if (c, y) not in CONDITIONAL_HELDOUT]
        return {
            f.format(colour=c, yes=y, no="lowest" if y == "highest" else "highest")
            for f in CONDITIONAL_FRAMES
            for c, y in pairs
        }
    if kind == "exclusion":
        return {f.format(keep=k) for f in EXCLUSION_FRAMES for k in EXCLUSION_TRAIN}
    return set(HIDDEN_FRAMES)


# ---------------------------------------------------------------------------------------------------------------- notes
# A note is one line of five fields separated by " | ", each "<name>: <value>":
#   see   what is seen: the number of shoots, the flower's colour, the label (hidden kind)
#   ask   the instruction in a fixed form
#   so    the inference: which shoot(s) that means, and (exclusion) which are done
#   aim   the shoot (or the label) to bring in front of the robot next: whether it faces the robot, else the button
#   next  the current sub-goal: "cut the <position> shoot", "show the label" or "stop"
NOTE_FIELDS = ("see", "ask", "so", "aim", "next")


def _join(names):
    names = [f"the {n}" for n in names]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def note_text(task: Task, flower: str, cut: tuple[int, ...] = (), label: str | None = None, aim=None) -> str:
    """The expert's note for `task` on a plant with a `flower`-coloured flower.

    `cut`: shoots already cut (indices). `label` (hidden kind): None while the card has not been read, else
    "visible:<colour>" or "hidden:<colour>" (read earlier, out of view now). `aim`: None when nothing is left to do,
    else (what, facing, button): what is 'label' or a position, facing is True when it faces the robot, button
    'blue' / 'orange' (the one that turns it toward the robot the shorter way) when it does not.
    """
    see = f"{task.count} shoots, {flower} flower"
    known = None
    if task.kind == "hidden":
        if label is None:
            see += ", label not seen"
        else:
            where, known = label.split(":")
            see += f", label {known}" if where == "visible" else f", label hidden, read {known}"
    positions = [position_name(i, task.count) for i in task.targets]
    if task.kind == "ordinal":
        p = task.params
        so = f"of {task.count}, {ORDINALS[p['k']]} from the {p['end']} is the {positions[0]}"
    elif task.kind == "conditional":
        p = task.params
        so = f"{flower} is {'' if p['holds'] else 'not '}{p['colour']}: the {positions[0]}"
    elif task.kind == "exclusion":
        so = f"cut {_join(positions)}"
        done = [position_name(i, task.count) for i in task.targets if i in cut]
        if done:
            so += f"; {_join(done)} {'is' if len(done) == 1 else 'are'} cut"
    else:
        so = "read the label first" if known is None else f"{known} names the {LABEL_MEANS[known]}"
    if aim is None:
        aim_text, goal = "nothing left", "stop"
    else:
        what, facing, button = aim
        subject = "the label" if what == "label" else f"the {what}"
        aim_text = f"{subject}, faces me" if facing else f"{subject}, faces away, press {button}"
        goal = "show the label" if what == "label" else f"cut the {what} shoot"
    values = dict(see=see, ask=task.ask, so=so, aim=aim_text, next=goal)
    return " | ".join(f"{k}: {values[k]}" for k in NOTE_FIELDS)


def note_fields(text: str) -> dict[str, str]:
    """The fields of a note (missing ones are absent). Tolerant of what a model might write."""
    out = {}
    for part in text.split("|"):
        name, sep, value = part.strip().partition(":")
        if sep and name.strip() in NOTE_FIELDS:
            out[name.strip()] = value.strip()
    return out


def subgoal(text: str) -> str:
    """The note's sub-goal (its "next" field), or "" if it has none."""
    return note_fields(text).get("next", "")


def subgoal_target(goal: str) -> str | None:
    """The position a sub-goal "cut the <position> shoot" names, else None."""
    words = goal.split()
    return words[2] if len(words) == 4 and words[0] == "cut" and words[2] in POSITIONS else None


# ------------------------------------------------------------------------------------------------- what a lookup gets
def lookup_ceiling(kind: str, split: str = "train", draws: int = 4000, seed: int = 0) -> float:
    """The best success a policy could reach that maps the instruction alone to the shoots to cut, whatever it sees:
    over many drawn tasks (plants with 2 or 3 shoots and any flower, as env.py draws them), the share whose target
    positions are the most common ones for their instruction. 1.0 would mean that the words alone decide."""
    import numpy as np

    rng = np.random.default_rng(seed)
    by_instruction: dict[str, dict[tuple, int]] = {}
    for _ in range(draws):
        count, flower = int(rng.integers(2, 4)), FLOWERS[int(rng.integers(len(FLOWERS)))]
        task = draw(kind, split, count, flower, rng)
        counts = by_instruction.setdefault(task.instruction, {})
        counts[task.target_positions] = counts.get(task.target_positions, 0) + 1
    return sum(max(c.values()) for c in by_instruction.values()) / draws


def all_heldout_instructions(kind: str, draws: int = 3000, seed: int = 1) -> set[str]:
    """Held-out instructions of `kind`, as drawn (for tests)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    out = set()
    for _ in range(draws):
        count, flower = int(rng.integers(2, 4)), FLOWERS[int(rng.integers(len(FLOWERS)))]
        out.add(draw(kind, "heldout", count, flower, rng).instruction)
    return out


def compositions() -> dict[str, list]:
    """The compositions each kind can draw, training and held out (for the docs and tests)."""
    pairs = [(c, y) for c, y in itertools.product(FLOWERS, ("lowest", "highest"))]
    return dict(
        ordinal_train=list(ORDINAL_TRAIN),
        ordinal_heldout=list(ORDINAL_HELDOUT),
        conditional_train=[p for p in pairs if p not in CONDITIONAL_HELDOUT],
        conditional_heldout=list(CONDITIONAL_HELDOUT),
        exclusion_train=list(EXCLUSION_TRAIN),
        exclusion_heldout=list(EXCLUSION_HELDOUT),
    )


# ------------------------------------------------------------------------------------------- training paraphrases
# More ways to say each training instruction, for training only (data.instruction_paraphrases): the same task, other
# words and sentence shapes. They never use a word that only held-out instructions use (heldout_only_words; tested),
# and never a held-out composition, so the held-out evaluation stays unseen. Their point is the reasoner's: it learns to
# turn any wording into the fixed "ask" of its note.
ORDINAL_PARAPHRASES = (
    "please cut the {ord} shoot from the {end}",
    "the {ord} shoot from the {end} needs cutting",
    "take off the {ord} side shoot from the {end}",
    "cut whichever shoot is {ord} from the {end}",
    "prune away the {ord} shoot from the {end}",
    "find the {ord} shoot from the {end} and cut it",
    "cut off the side shoot that is {ord} from the {end}",
    "remove the {ord} side shoot from the {end} of the stem",
)
CONDITIONAL_PARAPHRASES = (
    "if the flower is {colour}, prune the {yes} shoot, otherwise prune the {no} one",
    "if the flower looks {colour}, remove the {yes} shoot, else remove the {no} one",
    "with a {colour} flower cut the {yes} shoot, with any other flower cut the {no} one",
    "a {colour} flower means the {yes} shoot and any other flower means the {no} one: cut that shoot",
    "cut the {yes} side shoot if the flower is {colour}; otherwise cut the {no} side shoot",
    "when the flower is {colour} cut the {yes} shoot, otherwise the {no} one",
)
EXCLUSION_PARAPHRASES = (
    "cut all side shoots except the {keep} one",
    "prune every shoot other than the {keep} one",
    "keep the {keep} shoot and cut every other side shoot",
    "remove every shoot except the {keep} one",
    "the {keep} shoot stays; cut the other side shoots",
    "cut off each side shoot but the {keep} one",
)
HIDDEN_PARAPHRASES = (
    "look at the label in the pot and cut the shoot it names",
    "the label in the pot names a shoot: cut it",
    "cut the side shoot named by the label in the pot",
    "prune the shoot the label in the pot names",
    "find the label in the pot and remove the shoot it names",
    "read the pot label and cut the shoot it names",
)


def _words(text):
    import re

    return set(re.findall(r"[a-z]+", text.lower()))


def heldout_only_words() -> set[str]:
    """Words that held-out instructions use and training instructions do not."""
    train = set().union(*(_words(t) for k in KINDS for t in training_instructions(k)))
    heldout = set().union(*(_words(t) for k in KINDS for t in all_heldout_instructions(k)))
    return heldout - train


def paraphrase_table() -> dict[str, list[str]]:
    """Every training instruction -> other training-only ways to say it (same task)."""
    table: dict[str, list[str]] = {}
    for k, e in ORDINAL_TRAIN:
        alts = [f.format(ord=ORDINALS[k], end=e) for f in ORDINAL_PARAPHRASES]
        for f in ORDINAL_FRAMES:
            table[f.format(ord=ORDINALS[k], end=e)] = alts
    for c, y in compositions()["conditional_train"]:
        n = "lowest" if y == "highest" else "highest"
        alts = [f.format(colour=c, yes=y, no=n) for f in CONDITIONAL_PARAPHRASES]
        for f in CONDITIONAL_FRAMES:
            table[f.format(colour=c, yes=y, no=n)] = alts
    for keep in EXCLUSION_TRAIN:
        alts = [f.format(keep=keep) for f in EXCLUSION_PARAPHRASES]
        for f in EXCLUSION_FRAMES:
            table[f.format(keep=keep)] = alts
    for f in HIDDEN_FRAMES:
        table[f] = list(HIDDEN_PARAPHRASES)
    return table


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="write the training paraphrase table (data.instruction_paraphrases)")
    ap.add_argument("out")
    path = ap.parse_args().out
    with open(path, "w") as f:
        json.dump(paraphrase_table(), f, indent=1, sort_keys=True)
        f.write("\n")
