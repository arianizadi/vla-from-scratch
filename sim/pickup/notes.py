"""The expert's notes on the reasoning tasks: when one is written, and what it says (format: reasoning.note_text).

A note is written at the moments where a decision is made or its basis changes:

  - the start of the episode;
  - a shoot was cut;
  - the card in the pot was read (hidden kind): it faces the front camera while the plant stands still;
  - a turn of the plant has ended: a button was held down and now none is, and the plate has stopped.

What a note says is a function of what the robot can see at that moment (the plant, the shoots cut, the card if it
faces the camera) and of what it remembers (the card's colour once read). Between two such moments the note stands.
The same writer serves the demonstrations (collect.py), where the expert drives, and evaluation (eval.py), where a
policy drives and the writer's notes are the correct ones to compare with or to hand to the policy.
"""

import dataclasses

import numpy as np

from . import reasoning
from .env import BUTTONS, PRESS_TRAVEL, REACH_AZIMUTH, _wrap_angle
from .expert import PRUNE_SECTOR

PLATE_STILL = 0.05  # rad/s: the plate counts as standing still below this


class NoteWriter:
    def __init__(self, env):
        assert env.reason, f"{env.task} is not a reasoning task"
        self.env = env
        self.reset()

    def reset(self):
        """Call after env.reset."""
        self.label = None  # the card's colour once read
        self.note = ""
        self.written: list[tuple[int, str]] = []  # (control step, note)
        self._cut = frozenset()
        self._pressed = False

    def update(self) -> bool:
        """Look at the scene now (before the next action); True if a new note was written."""
        env = self.env
        pressed = any(env.button_travel(b) > PRESS_TRAVEL for b in BUTTONS)
        still = not pressed and abs(env.plate_speed()) < PLATE_STILL
        self._pressed |= pressed
        event = not self.written
        if frozenset(env.cut) != self._cut:
            self._cut, event = frozenset(env.cut), True
        if env.label_colour is not None and self.label is None and still and env.label_visible():
            self.label, event = env.label_colour, True
        if self._pressed and still:
            self._pressed, event = False, True
        if not event:
            return False
        text = self.text()
        if text == self.note:
            return False
        self.note = text
        self.written.append((env.steps, text))
        return True

    def text(self, targets=None) -> str:
        """The note for the scene now. `targets`: shoots to treat as the ones meant instead of the task's (to write a
        wrong note on purpose, for the test that the text matters)."""
        env = self.env
        task = env.reason_task
        if targets is not None:
            task = dataclasses.replace(task, targets=tuple(env.active.index(n) for n in targets))
            targets = list(targets)
        else:
            targets = env.targets
        cut = tuple(env.active.index(n) for n in env.cut if n in env.active)
        label = None
        if self.label is not None:
            colour = self.label
            if task is not env.reason_task:  # a note about other shoots: the card read as naming the first of them
                position = reasoning.position_name(task.targets[0], task.count)
                colour = next(c for c, p in reasoning.LABEL_MEANS.items() if p == position)
            label = ("visible:" if env.label_visible() else "hidden:") + colour
        remaining = [n for n in targets if n not in env.cut]
        if not remaining:
            aim = None
        elif env.label_colour is not None and self.label is None:
            aim = ("label", False, reasoning.BUTTON_NAMES[_sign(env.label_turn_needed())])
        else:
            turns = {n: float(_wrap_angle(REACH_AZIMUTH - env.shoot_azimuth(n))) for n in remaining}
            name = min(remaining, key=lambda n: abs(turns[n]))
            position = reasoning.position_name(env.active.index(name), len(env.active))
            facing = abs(turns[name]) <= PRUNE_SECTOR
            aim = (position, facing, None if facing else reasoning.BUTTON_NAMES[_sign(turns[name])])
        return reasoning.note_text(task, env.flower_colour, cut, label, aim)


def _sign(x):
    return 1 if x >= 0 else -1


def note_columns(step_notes: list[str]) -> dict:
    """Per-frame bookkeeping for one episode's notes (`step_notes`: the note in force at each frame): the note, the one
    before it ('' before the first), the frames since the current one was written, and its sub-goal."""
    prev, age, last, before = [], [], None, ""
    for i, note in enumerate(step_notes):
        if note != last:
            before, last, since = (last or ""), note, i
        prev.append(before)
        age.append(i - since)
    return dict(
        note=step_notes,
        note_prev=prev,
        note_age=np.asarray(age, dtype=np.int64),
        instruction_subgoal=[reasoning.subgoal(n) for n in step_notes],
    )
