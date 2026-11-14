"""Instructions that name one side shoot of the pruning plant (tasks prune_named and prune_named_heldout).

A shoot is named by where it leaves the main stem: the lowest, the middle one (only when there are three) or the
highest. Height on the stem can be seen from any side and does not change as the plant turns, whereas "the left one"
or "the one facing you" would.

Demonstrations use only the TRAIN phrasings: one word for each position, two nouns and four sentence frames (24
instructions). Evaluation on held-out phrasings asks whether a policy understands an instruction it never saw. There
are three kinds of held-out phrasing (KINDS):

  synonym  an unseen way of saying the position, in a training frame: "prune the bottom shoot",
           "remove the side shoot nearest the pot"
  frame    the training word and noun in an unseen sentence: "snip the lowest shoot", "the highest side shoot has to go"
  all      unseen frame, unseen position words and an unseen noun: "the uppermost branch has to go"

Every held-out instruction contains at least one word that no training instruction contains (tests check it). A policy
that reads instructions through a word table learned from its demonstrations has never trained those words'
embeddings; a policy built on a pretrained language model may know that "bottom" means "lowest".
"""

import itertools

POSITIONS = ("lowest", "middle", "highest")
KINDS = ("synonym", "frame", "all")

TRAIN_NOUNS = ("side shoot", "shoot")
TRAIN_DESCRIPTIONS = {
    "lowest": ("the lowest {noun}",),
    "middle": ("the middle {noun}",),
    "highest": ("the highest {noun}",),
}
TRAIN_FRAMES = (
    "cut {shoot} off the stem, keep the flower",
    "prune {shoot}",
    "cut off {shoot} and leave the rest",
    "remove {shoot}",
)

HELDOUT_NOUNS = ("branch", "offshoot")
HELDOUT_DESCRIPTIONS = {
    "lowest": ("the bottom {noun}", "the lowermost {noun}", "the {noun} nearest the pot"),
    "middle": ("the centre {noun}", "the {noun} between the other two"),
    "highest": ("the top {noun}", "the uppermost {noun}", "the {noun} nearest the flower"),
}
HELDOUT_FRAMES = (
    "snip {shoot}",
    "please trim away {shoot} but spare the others",
    "{shoot} has to go",
    "take {shoot} off the plant",
)


def position_name(index: int, count: int) -> str:
    """Name of shoot `index` (0 = the lowest) on a plant with `count` shoots."""
    assert 0 <= index < count and 2 <= count <= 3, (index, count)
    return "lowest" if index == 0 else "highest" if index == count - 1 else "middle"


def _compose(frames, descriptions, nouns):
    return [f.format(shoot=d.format(noun=n)) for f, d, n in itertools.product(frames, descriptions, nouns)]


def training(position: str) -> list[str]:
    """The training instructions that name `position`."""
    return _compose(TRAIN_FRAMES, TRAIN_DESCRIPTIONS[position], TRAIN_NOUNS)


def heldout(position: str) -> dict[str, list[str]]:
    """The held-out instructions that name `position`, by kind."""
    return dict(
        synonym=_compose(TRAIN_FRAMES, HELDOUT_DESCRIPTIONS[position], TRAIN_NOUNS),
        frame=_compose(HELDOUT_FRAMES, TRAIN_DESCRIPTIONS[position], TRAIN_NOUNS),
        all=_compose(HELDOUT_FRAMES, HELDOUT_DESCRIPTIONS[position], HELDOUT_NOUNS),
    )


def sample(position: str, which: str, rng) -> tuple[str, str]:
    """An instruction naming `position`: (text, kind). `which` is 'train' (kind 'train') or 'heldout' (one of KINDS,
    each equally likely, then one phrasing of that kind)."""
    if which == "train":
        options, kind = training(position), "train"
    elif which == "heldout":
        kind = KINDS[int(rng.integers(len(KINDS)))]
        options = heldout(position)[kind]
    else:
        raise ValueError(f"phrasings: unknown set {which!r}")
    return options[int(rng.integers(len(options)))], kind
