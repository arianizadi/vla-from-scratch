"""Notes in the prompt of a reason-then-act policy (vla_foundry/data/processor/notes.py).

A character-level tokenizer stands in for a real one (the real ones are checked on the training machine): every
character is a token, and the special tokens below are one token each.
"""

import random

import pytest
import torch

from vla_foundry.data.processor.notes import NOTE_MARK, SAME, NoteTokens, note_prompt, training_pair

SPECIAL = {"<eos>": 1, "<end>": 2, "<pad>": 0}


class _CharTokenizer:
    eos_token, eos_token_id, pad_token_id = "<eos>", 1, 0

    def _ids(self, text):
        ids, i = [], 0
        while i < len(text):
            special = next((s for s in SPECIAL if text.startswith(s, i)), None)
            if special:
                ids.append(SPECIAL[special])
                i += len(special)
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": self._ids(text)}

    def decode(self, ids):
        names = {v: k for k, v in SPECIAL.items()}
        return "".join(names.get(int(i), chr(int(i))) for i in ids)

    def batch(self, texts):
        """Right-padded ids and mask, as a processor would return them."""
        rows = [self._ids(t) for t in texts]
        width = max(map(len, rows))
        ids = torch.tensor([r + [0] * (width - len(r)) for r in rows])
        mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows])
        return ids, mask


class _Processor:
    def __init__(self, template):
        self.tokenizer = _CharTokenizer()
        self.template = template


def _template(processor, num_images, text):
    """A chat template that closes the turn with <end>, or none (a plain image prefix)."""
    prefix = "<img>" * num_images
    return f"<user>{prefix}{text}<end>\n" if processor.template else prefix + text


class _Rng:
    def __init__(self, d):
        self.d = d

    def randint(self, a, b):
        assert a == 1 and b == 8
        return self.d


def test_the_held_note_is_the_one_in_force_d_frames_ago():
    meta = dict(note="B", note_prev="A", note_age="3")  # B was written 3 frames ago
    for d in (1, 2, 3):  # B was already written d frames ago: it stands
        assert training_pair(meta, 8, _Rng(d)) == ("B", SAME)
    for d in (4, 8):  # d frames ago A was in force: B is to be written now
        assert training_pair(meta, 8, _Rng(d)) == ("A", "B")
    start = dict(note="A", note_prev="", note_age="0")  # the first note, at the first frame
    assert training_pair(start, 8, _Rng(1)) == ("", "A")
    held, now = training_pair(dict(note="A", note_prev="", note_age=5), 8)  # default generator; ints accepted
    assert (held, now) in (("A", SAME), ("", "A"))


def test_the_prompt_ends_where_the_note_is_written():
    assert note_prompt("cut it", "") == "cut it\nnote: none" + NOTE_MARK
    assert note_prompt("cut it", "x | y", "same") == "cut it\nnote: x | y\nnow: same"


@pytest.mark.parametrize("template", [True, False])
def test_labels_cover_the_note_and_the_end_token(template):
    processor = _Processor(template)
    notes = NoteTokens(processor, _template)
    assert notes._stop_id is None  # nothing is encoded until it is needed (see NoteTokens.stop_id)
    assert notes.stop_id == (SPECIAL["<end>"] if template else SPECIAL["<eos>"])
    rows = [("prune it", "", "see: 2 shoots | next: cut the lowest shoot"), ("cut", "old note", SAME)]
    texts = [notes.text(2, *row) for row in rows]
    ids, mask = processor.tokenizer.batch(texts)
    labels = notes.labels(ids, mask, [now for *_, now in rows])
    for i, (_, _, now) in enumerate(rows):
        span = ids[i][labels[i]].tolist()
        assert processor.tokenizer.decode(span) == " " + now + ("<end>" if template else "<eos>")
    # a prompt to write from stops right after the mark, before the template closes the turn
    prompt = notes.text(2, "prune it", "held", None)
    assert prompt.endswith(NOTE_MARK) and "<end>" not in prompt and "<eos>" not in prompt
    assert prompt == notes.text(2, "prune it", "held", "x")[: len(prompt)]


def test_a_note_that_does_not_line_up_is_an_error():
    processor = _Processor(True)
    notes = NoteTokens(processor, _template)
    ids, mask = processor.tokenizer.batch([notes.text(0, "a", "", "the note")])
    with pytest.raises(ValueError, match="line up"):
        notes.labels(ids, mask, ["another note!"])
    with pytest.raises(ValueError, match="no end token"):
        notes.labels(ids[:, :-3], mask[:, :-3], ["the note"])


def test_each_process_draws_its_own_intervals():
    seen = {training_pair(dict(note="B", note_prev="A", note_age=4), 8)[1] for _ in range(200)}
    assert seen == {SAME, "B"}
    random.seed(0)  # the module's generator does not depend on the global one


def test_paraphrases_replace_a_training_instruction_with_the_given_probability(tmp_path):
    import json

    from vla_foundry.data.processor.notes import paraphrase

    table = tmp_path / "paraphrases.json"
    table.write_text(json.dumps({"cut it": ["prune it", "remove it"]}))
    rng = random.Random(0)
    drawn = [paraphrase("cut it", str(table), 0.5, rng) for _ in range(400)]
    assert set(drawn) == {"cut it", "prune it", "remove it"} and 150 < drawn.count("cut it") < 250
    assert paraphrase("not listed", str(table), 1.0, rng) == "not listed"
    assert paraphrase("cut it", None, 1.0, rng) == "cut it" and paraphrase("cut it", str(table), 0.0, rng) == "cut it"
