"""Notes in the prompt of a reason-then-act policy (data.notes; model type reason_then_act).

The text after the images is

    <instruction>
    note: <the note the policy holds, "none" at the start>
    now: <the note it writes now, or "same" if the one it holds still stands>

followed by the end of the turn (the chat template's end-of-turn token, or the tokenizer's end-of-sequence token for a
processor without a template). The language-model loss is taken on the tokens after "now:" up to and including that
end token, which teaches the backbone where a note ends.

Training samples come from demonstrations whose frames carry the expert's notes (sim/pickup/notes.py: the note in force,
the one before it and its age in frames). A served policy writes a note every few control steps, so the note it holds
can be a few steps old: for each training sample the held note is the one in force `d` frames earlier, d drawn
uniformly from 1..interval. Where a new note was written within the last d frames the target is that note; otherwise
it is "same".
"""

import os
import random

import torch

NOTE_MARK = "\nnow:"
SAME = "same"
_RNG: dict[int, random.Random] = {}  # one generator per process: data-loader workers are forked


def _rng():
    pid = os.getpid()
    if pid not in _RNG:
        _RNG[pid] = random.Random(int.from_bytes(os.urandom(8), "little"))
    return _RNG[pid]


def note_prompt(instruction: str, held: str, now: str | None = None) -> str:
    """The text after the images; ends at "now:" when `now` is None (a prompt to write the note from)."""
    text = f"{instruction}\nnote: {held or 'none'}{NOTE_MARK}"
    return text if now is None else f"{text} {now}"


def training_pair(metadata: dict, interval: int, rng=None) -> tuple[str, str]:
    """(held note, note to write now) for a training sample whose frame metadata has note, note_prev and note_age."""
    rng = rng or _rng()
    note, before = str(metadata["note"]), str(metadata.get("note_prev", ""))
    age = int(float(metadata.get("note_age", 0)))
    d = rng.randint(1, max(1, interval))
    held = note if d <= age else before
    return held, (SAME if held == note else note)


class NoteTokens:
    """Where a note ends for a given processor, and which tokens of a processed batch the note loss is taken on."""

    def __init__(self, processor, apply_chat_template):
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)
        self.apply_chat_template = apply_chat_template
        probe = apply_chat_template(processor, 0, "PROBE")
        tail = probe[probe.rindex("PROBE") + len("PROBE") :]
        if tail.strip():  # the template closes the turn: its first token ends the note
            self.stop_text, self.append_stop = tail, False
        else:  # no template: end the note with the end-of-sequence token
            self.stop_text, self.append_stop = self.tokenizer.eos_token, True
            if self.tokenizer.eos_token_id is None:
                raise ValueError("notes: the processor has neither a chat template nor an end-of-sequence token")
        self._stop_id = None

    @property
    def stop_id(self) -> int:
        """The token that ends a note. Found on first use, not when constructed: a fast tokenizer that encodes in the
        parent process before the data loader forks its workers can deadlock them."""
        if self._stop_id is None:
            self._stop_id = self.tokenizer(self.stop_text, add_special_tokens=False)["input_ids"][0]
        return self._stop_id

    def text(self, num_images: int, instruction: str, held: str, now: str | None) -> str:
        """The full prompt text for the processor (chat template applied); with `now` None, cut where the note to be
        written begins."""
        if now is None:
            full = self.apply_chat_template(self.processor, num_images, note_prompt(instruction, held, "\x00"))
            return full[: full.rindex(NOTE_MARK) + len(NOTE_MARK)]
        full = self.apply_chat_template(self.processor, num_images, note_prompt(instruction, held, now))
        return full + self.stop_text if self.append_stop else full

    def labels(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None, notes_now: list[str]):
        """[B, L] bool: the tokens of each row's current note and the end token after it."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        labels = torch.zeros_like(input_ids, dtype=torch.bool)
        for i, now in enumerate(notes_now):
            real = attention_mask[i].bool()
            stops = ((input_ids[i] == self.stop_id) & real).nonzero().flatten()
            if len(stops) == 0:
                raise ValueError(f"notes: no end token in row {i} (truncated? raise max_text_seq_len)")
            end = int(stops[-1])
            n = len(self.tokenizer(" " + now, add_special_tokens=False)["input_ids"])
            labels[i, end - n : end + 1] = True
            written = self.tokenizer.decode(input_ids[i, end - n : end]).strip()
            if written != now.strip():
                raise ValueError(f"notes: the note's tokens do not line up: {written!r} != {now!r}")
        return labels
