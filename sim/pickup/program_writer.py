"""The program writer: a pretrained language model, not fine-tuned, compiles an instruction into a typed program.

An instruction of a reasoning task ("cut the second shoot from the top", "unless the flower is red, remove the lowest
shoot; if it is, remove the highest one") is written once as a program of program.py (`Cut(Rank(2,top))`,
`If(FlowerIs(red),Cut(Rank(1,top)),Cut(Rank(1,bottom)))`), and the program is then evaluated on the scene memory
whenever the memory changes. Reading the words is the one step that needs a language model. The model is shown a
description of the language, construct by construct (SYSTEM), and a few worked examples that use training instructions
only (`examples`), and writes the program for the instruction it is given (`ProgramWriter`).

Why the model is not fine-tuned. The reasoner of the first design was fine-tuned on thirty instruction strings and
learned them as a table: it restates 107 of 200 held-out instructions correctly, and none of the 45 about the label.
The same language model as downloaded, shown eight worked examples, restated 32 of 40 held-out instructions in a probe
(tutorial/architecture-v2.md, section 2). So the model is used for what it could do before any training here: it
restates. Counting from an end, taking a branch and the complement are computed by the program.

What the grammar constraint guarantees is the form. While the model writes, every token that would take the text off
every program is masked out (`allowed_tokens`, through generate's prefix_allowed_tokens_fn), and the end of the
sequence is allowed only once the text is a whole program. This is exact, not a filter of likely mistakes: the walk
down a trie of the vocabulary's tokens (`TokenTrie`) follows a character only where the grammar's own acceptor
(program.next_chars) allows it, so a token is allowed exactly when the text stays the beginning of some program.
Whatever the model prefers, the output is a program, unless the limit on new tokens comes first. A token adds at
least one character and the longest program of the language has 198 (LONGEST_PROGRAM; the benchmark's longest has
57), so with 199 new tokens, the command line's default, it cannot come first. What the constraint cannot give is the
meaning: `Cut(Rank(2,top))` for "the second shoot from the bottom" is well formed and wrong.

The evaluation (`main`) measures the two apart, on tasks drawn as the benchmark draws them:

  format    the output parses as a program. Always so when constrained; it is measured for the baseline that writes
            freely (--unconstrained), the same model with the same prompt
  exact     its canonical text is the task's program (program.from_task)
  meaning   it is equivalent to the task's program: the same result on every plant (program.equivalent)
  resolved  on the plant drawn, it selects the shoots the task means (program.targets_on)

by kind and split, and by kind of held-out instruction. --facts runs the comparison of the design's second probe
instead: the same model and worked examples, given accurate facts about the plant as text, name the shoots to cut in
one step, with no program (`FactsAnswerer`).

  python -m sim.pickup.program_writer --model Qwen/Qwen3-VL-2B-Thinking --out writer.json
  python -m sim.pickup.program_writer --model Qwen/Qwen3-VL-2B-Thinking --unconstrained --out free.json
  python -m sim.pickup.program_writer --model Qwen/Qwen3-VL-2B-Thinking --facts --out facts.json

--weights CHECKPOINT loads the language model of a reason_then_act training checkpoint over the downloaded weights,
so that the same tests run on a fine-tuned copy of the model (`load_weights`).
"""

import argparse
import functools
import json
import pickle
import re
import time
from collections.abc import Callable, Sequence

import numpy as np
import torch

from . import program, reasoning

SPLITS = ("train", "heldout")
FLAGS = ("format", "exact", "meaning", "resolved")
# The characters of the longest program of the language: an If whose branches select with Except nested two deep over
# the longest leaf, Rank(1,bottom) (tested). Every token writes at least one character, so a constrained writer given
# one token more than this always ends with a whole program.
LONGEST_PROGRAM = 198

# ----------------------------------------------------------------------------------------------------------- the prompt
# The language, construct by construct. The words are chosen with care: none is a word that only held-out instructions
# use (reasoning.heldout_only_words: "tag", "card", "unless", "only", "to", ...), except "middle", the name of
# Middle(). A prompt that said "the label, tag or card" would teach the held-out wording it is meant to be tested on.
SYSTEM = """You write programs for a robot that prunes a potted plant: a main stem with a flower on top, and two or \
three side shoots at different heights. A program names the side shoots the robot must cut. Reply with the program \
for the instruction, written without spaces, and nothing else.

The language, part by part. A program is a Cut or an If:
Cut(S): cut the side shoots that the selection S picks.
If(FlowerIs(C),Cut(S1),Cut(S2)): if the flower is of colour C, cut S1, otherwise cut S2. C is red, orange, pink or \
purple.
A selection is one of these:
All(): every side shoot.
Rank(K,top): the K-th shoot counting from the top (K is 1, 2 or 3). Rank(1,top) is the highest shoot.
Rank(K,bottom): the K-th shoot counting from the bottom. Rank(1,bottom) is the lowest shoot.
Middle(): the middle shoot of three.
Except(S1,S2): the shoots of S1 except those of S2. Except(All(),Rank(1,bottom)) is every shoot except the lowest.
ReadLabel(): the shoot that the label in the pot names.
Filter(A,V,S): the shoots of S whose attribute A is V. The pairs A,V are tie,red; tie,blue; tie,yellow; tie,white; \
sick,yes; bud,yes; bud,no."""

# The comparison (FactsAnswerer): the same plant described in words, the shoots named in one step.
FACTS_SYSTEM = """You plan for a robot that prunes a potted plant: a main stem with a flower on top, and two or three \
side shoots at different heights. You get a description of the plant and an instruction. Work out the side shoots \
that the instruction means, and reply with one line: 'cut: ' and their names as the description lists them, \
separated by commas."""

# The words that differ between two compositions of a kind; what is left of an instruction without them is its frame.
_COMPOSITION_WORDS = (*reasoning.ORDINALS.values(), *reasoning.ENDS, *reasoning.FLOWERS, *reasoning.POSITIONS)
_COMPOSITION = re.compile(r"\b(?:" + "|".join(_COMPOSITION_WORDS) + r")\b")


def _plant(rng) -> tuple[int, str]:
    """A plant as reasoning.lookup_ceiling draws one: 2 or 3 shoots, any flower."""
    return int(rng.integers(2, 4)), reasoning.FLOWERS[int(rng.integers(len(reasoning.FLOWERS)))]


def _program_of(task: reasoning.Task) -> str:
    return program.text(program.from_task(task))


def _frame(instruction: str) -> str:
    return _COMPOSITION.sub("_", instruction)


@functools.cache
def _heldout_instructions() -> frozenset[str]:
    return frozenset().union(*(reasoning.all_heldout_instructions(kind) for kind in reasoning.KINDS))


def _worked_tasks(shots: int, seed: int) -> list[tuple[reasoning.Task, str]]:
    """The tasks of the worked examples, each with the flower of the plant it was drawn for. The kinds take turns;
    within a kind, each example is the first drawn whose instruction, sentence frame and program the examples before it
    do not show (or as many of the three as the kind allows: the label task has one program)."""
    rng = np.random.default_rng(seed)
    chosen: list[tuple[reasoning.Task, str]] = []
    for i in range(shots):
        kind = reasoning.KINDS[i % len(reasoning.KINDS)]
        shown = [task for task, _ in chosen]
        same_kind = [task for task in shown if task.kind == kind]
        best, best_news = None, -1
        for _ in range(200):  # a kind has at most 12 training instructions: 200 draws see each of them
            count, flower = _plant(rng)
            task = reasoning.draw(kind, "train", count, flower, rng)
            news = (
                4 * all(task.instruction != other.instruction for other in shown)
                + 2 * all(_frame(task.instruction) != _frame(other.instruction) for other in same_kind)
                + all(_program_of(task) != _program_of(other) for other in same_kind)
            )
            if news > best_news:
                best, best_news = (task, flower), news
            if news == 7:
                break
        chosen.append(best)
    heldout = _heldout_instructions()
    for task, _ in chosen:  # drawn from the training split, so this holds by construction; the held-out test needs it
        assert task.instruction in reasoning.training_instructions(task.kind), task.instruction
        assert task.instruction not in heldout, task.instruction
    return chosen


def examples(shots: int = 8, seed: int = 0) -> list[tuple[str, str]]:
    """Worked examples for the prompt: (instruction, program text) pairs, from training instructions only, the four
    kinds in turn and different sentence frames and compositions within a kind. The same seed gives the same ones."""
    return [(task.instruction, _program_of(task)) for task, _ in _worked_tasks(shots, seed)]


def facts_text(count: int, flower: str, label: str | None) -> str:
    """Accurate facts about a plant, in words: the shoots from the bottom up, the flower's colour, the card in the pot
    (None: there is none) and what its colours mean."""
    shoots = ", ".join(reasoning.position_name(i, count) for i in range(count))
    meanings = reasoning.LABEL_MEANS.items()
    rule = ", ".join(f"a {colour} label names the {position} shoot" for colour, position in meanings)
    return (
        f"Side shoots, from the bottom of the stem up: {shoots} ({count} shoots).\n"
        f"Flower: {flower}.\n"
        f"Label in the pot: {label or 'none'}.\n"
        f"Label rule: {rule}."
    )


def _facts_question(task: reasoning.Task, flower: str) -> str:
    return f"{facts_text(task.count, flower, task.params.get('label'))}\nInstruction: {task.instruction}"


def answer_text(positions: Sequence[str]) -> str:
    """A one-step answer as the facts test expects it: "cut: lowest, highest"."""
    return "cut: " + ", ".join(positions)


def _prompt(tokenizer, system: str, turns: Sequence[tuple[str, str]], question: str, no_think: bool) -> str:
    """The description, the worked examples and the question, in the tokenizer's chat template when it has one (a
    system turn, then a user turn and the assistant's answer per example, then the question as the last user turn),
    else as plain text in the same order, ending with the question and a line break."""
    if not getattr(tokenizer, "chat_template", None):
        shown = "".join(f"{asked}\n{answer}\n\n" for asked, answer in turns)
        return f"{system}\n\n{shown}{question}\n"
    messages = [{"role": "system", "content": system}]
    for asked, answer in turns:
        messages += [{"role": "user", "content": asked}, {"role": "assistant", "content": answer}]
    messages.append({"role": "user", "content": question})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if no_think:
        # A "thinking" model writes its reasoning between <think> and </think> before it answers; an empty block makes
        # it answer at once. Some templates open the block themselves.
        text += "\n</think>\n\n" if text.endswith("<think>\n") else "<think>\n\n</think>\n\n"
    return text


# ------------------------------------------------------------------------------------------------------- the constraint
def token_texts(tokenizer) -> dict[int, str]:
    """What each token adds to the decoded text, by id (special tokens left out).

    A token is decoded after another one and its text is the difference. Decoding it alone can hide a leading space: a
    byte-level BPE vocabulary writes the space into the token (" the" is "Ġthe"), a SentencePiece one marks it with
    "▁", and a decoder may drop a space at the start of a text. Read in context, " the" is " the", which is not
    program text. A token that is part of a character of several bytes decodes to a replacement character."""
    anchor = tokenizer.encode("a", add_special_tokens=False)
    decode = functools.partial(tokenizer.decode, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    before = decode(anchor)
    special = set(tokenizer.all_special_ids)
    texts = {}
    for token_id in range(len(tokenizer)):
        if token_id not in special:
            after = decode([*anchor, token_id])
            if after.startswith(before):
                texts[token_id] = after[len(before) :]
    return texts


class _Node:
    __slots__ = ("children", "ids")

    def __init__(self):
        self.children: dict[str, _Node] = {}
        self.ids: list[int] = []  # the tokens whose text ends here


class TokenTrie:
    """A character trie over the tokens of a vocabulary that can be part of a program: those whose text (token_texts:
    what the token adds to the decoded text) is not empty and made only of program.ALPHABET characters (or of
    `alphabet`'s: program.ALPHABETS["v3"] for v3's language). A token with a leading space is not one: programs have no
    spaces. `texts`: those tokens' texts, by id."""

    def __init__(self, texts: dict[int, str], alphabet: frozenset[str] | None = None):
        alphabet = program.ALPHABET if alphabet is None else alphabet
        self.texts = {i: t for i, t in texts.items() if t and set(t) <= alphabet}
        self.root = _Node()
        for token_id, text in self.texts.items():
            node = self.root
            for char in text:
                node = node.children.setdefault(char, _Node())
            node.ids.append(token_id)


def allowed_tokens(trie: TokenTrie, written: str, eos_ids: Sequence[int], language: str = "v2") -> list[int]:
    """The tokens that may follow `written`: each token whose text keeps it the beginning of some program (of
    `language`), and the end-of-sequence tokens if and only if `written` is a whole program.

    Exact: the walk down the trie follows a character only if program.next_chars allows it after the written text
    and the path so far, and collects the tokens that end on the way, so a token is collected exactly when each of
    its characters is allowed in turn. Never empty, since generate needs a choice: where nothing is allowed (after an
    invalid text), the end-of-sequence tokens."""
    allowed = set()
    stack = [(trie.root, written)]
    while stack:
        node, text = stack.pop()
        allowed.update(node.ids)
        if node.children:
            for char in program.next_chars(text, language).intersection(node.children):
                stack.append((node.children[char], text + char))
    if program.is_complete(written, language):
        allowed.update(eos_ids)
    return sorted(allowed) if allowed else list(eos_ids)


# -------------------------------------------------------------------------------------------------------------- writing
def _end_ids(model, tokenizer) -> list[int]:
    """The tokens that end a sequence: the tokenizer's and the model's (a chat model ends its turn with its own token,
    Qwen with <|im_end|>)."""
    ids = set()
    for value in (tokenizer.eos_token_id, getattr(getattr(model, "generation_config", None), "eos_token_id", None)):
        if isinstance(value, int):
            ids.add(value)
        elif value is not None:
            ids.update(int(v) for v in value)
    ids = sorted(i for i in ids if 0 <= i < len(tokenizer))
    if not ids:
        raise ValueError("neither the tokenizer nor the model names an end-of-sequence token")
    return ids


def _generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    stop_ids: list[int],
    allowed: Callable[[list[int]], list[int]] | None = None,
    temperature: float = 0.0,
) -> tuple[list[int], float]:
    """Greedy decoding after `prompt`: the new token ids (the one that stopped it included) and the seconds taken. It
    stops at a token of `stop_ids` or after max_new_tokens. `allowed`, when given, maps the new ids so far to the ids
    that may come next."""
    # A chat template writes the special tokens the model expects; plain text gets the tokenizer's own (a BOS).
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=not getattr(tokenizer, "chat_template", None))
    input_ids = inputs["input_ids"].to(model.device)
    start = input_ids.shape[1]

    def constraint(batch_id: int, ids: torch.Tensor) -> list[int]:
        return allowed(ids[start:].tolist())

    began = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"].to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            num_beams=1,
            # a model's sampling settings (Qwen's) do not apply to greedy decoding; sampled: Qwen's for thinking
            temperature=temperature if temperature > 0 else None,
            top_p=0.95 if temperature > 0 else None,
            top_k=20 if temperature > 0 else None,
            eos_token_id=stop_ids,
            pad_token_id=stop_ids[0] if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
            prefix_allowed_tokens_fn=None if allowed is None else constraint,
        )
    return out[0, start:].tolist(), time.perf_counter() - began


class _Prompted:
    """A language model and its tokenizer, prompted with a description and worked examples, decoding greedily."""

    def __init__(self, model, tokenizer, no_think: bool, max_new_tokens: int):
        self.model, self.tokenizer = model, tokenizer
        self.max_new_tokens = max_new_tokens
        self.no_think = no_think and "<think>" in tokenizer.get_vocab()  # only a thinking model has the block
        self.texts = token_texts(tokenizer)
        self.eos_ids = _end_ids(model, tokenizer)
        # Free writing ends with the sequence or with the line: the answer is one line
        self.line_ids = sorted({*self.eos_ids, *(i for i, text in self.texts.items() if "\n" in text)})

    def _free(self, prompt: str) -> tuple[str, list[int], float]:
        """The first line written freely after `prompt`, the new ids and the seconds taken."""
        new, seconds = _generate(self.model, self.tokenizer, prompt, self.max_new_tokens, self.line_ids)
        reply = self.tokenizer.decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return reply.split("\n", 1)[0].strip(), new, seconds


def _parses(written: str, language: str = "v2") -> bool:
    try:
        program.parse(written, language)
    except program.ProgramError:
        return False
    return True


class ProgramWriter(_Prompted):
    """Writes the program of an instruction with a language model that is not fine-tuned (see the module's docstring).

    shots, seed     the worked examples (`examples`)
    constrained     held to the grammar while it writes (allowed_tokens); False: the model writes freely, greedily,
                    until the end of the sequence or of the line, and the text may not be a program (the baseline)
    no_think        make a thinking model answer at once (an empty <think></think> block); nothing for other models
    max_new_tokens  the limit on new tokens, the one that ends the writing included. A constrained program cut short
                    by it is not complete; with LONGEST_PROGRAM + 1 that cannot happen
    language        the language written and held to: "v2" (SYSTEM, `examples`) or "v3" (program_writer_v3: its
                    description and worked examples, passed as `system` and `pairs`)
    system, pairs   the description and the worked examples ((instruction, program) pairs) in place of v2's
    think           a thinking model first writes its reasoning freely, greedily, up to this many tokens or its own
                    </think> (closed for it at the limit), and then the program, constrained as without it. 0 (the
                    default): no_think's empty block, as v2's writer. The worked examples' answers stay programs
    think_temperature  0 (the default): the reasoning greedily; above 0, sampled at this temperature (top-p 0.95,
                    top-k 20, Qwen's settings for thinking). The program is always written greedily
    """

    def __init__(
        self,
        model,
        tokenizer,
        shots: int = 8,
        seed: int = 0,
        constrained: bool = True,
        no_think: bool = True,
        max_new_tokens: int = 96,
        language: str = "v2",
        system: str | None = None,
        pairs: Sequence[tuple[str, str]] | None = None,
        think: int = 0,
        think_temperature: float = 0.0,
    ):
        super().__init__(model, tokenizer, no_think and not think, max_new_tokens)
        self.think = int(think) if "</think>" in tokenizer.get_vocab() else 0
        self.think_temperature = float(think_temperature)
        self._thoughts: dict[str, tuple[str, float]] = {}  # instruction -> (reasoning, seconds), from think_many
        assert language in program.LANGUAGES, language
        self.language = language
        self.system = SYSTEM if system is None else system
        self.examples = examples(shots, seed) if pairs is None else [tuple(pair) for pair in pairs]
        self.constrained = constrained
        self.trie = TokenTrie(self.texts, program.ALPHABETS[language])

    def prompt(self, instruction: str) -> str:
        turns = [(f"Instruction: {said}", written) for said, written in self.examples]
        return _prompt(self.tokenizer, self.system, turns, f"Instruction: {instruction}", self.no_think)

    def write(self, instruction: str) -> dict:
        """{"program": the text written, "complete": it is a whole program (it parses), "tokens": the new tokens, the
        one that ended the writing included, "seconds": the time the model took}."""
        if not self.constrained:
            written, new, seconds = self._free(self.prompt(instruction))
            return dict(program=written, complete=_parses(written, self.language), tokens=len(new), seconds=seconds)

        def allowed(new: list[int]) -> list[int]:
            # every new token so far came from the trie: the end of the sequence ends the writing
            written = "".join(self.trie.texts[i] for i in new)
            return allowed_tokens(self.trie, written, self.eos_ids, self.language)

        prompt = self.prompt(instruction)
        thought, thought_s = "", 0.0
        if self.think and instruction in self._thoughts:
            thought, thought_s = self._thoughts[instruction]
            prompt = f"{self._think_prompt(prompt)}{thought}\n</think>\n\n"
        elif self.think:
            prompt, thought, thought_s = self._thought(prompt)
        new, seconds = _generate(self.model, self.tokenizer, prompt, self.max_new_tokens, self.eos_ids, allowed)
        written = "".join(self.trie.texts[i] for i in new if i not in self.eos_ids)
        out = dict(program=written, complete=_parses(written, self.language), tokens=len(new), seconds=seconds + thought_s)
        return dict(out, thought=thought) if self.think else out

    @staticmethod
    def _think_prompt(prompt: str) -> str:
        return prompt if prompt.endswith("<think>\n") else prompt + "<think>\n"

    def _thought(self, prompt: str) -> tuple[str, str, float]:
        """(the prompt with the model's reasoning and a closed think block after it, the reasoning, the seconds)."""
        prompt = self._think_prompt(prompt)
        end = self.tokenizer.convert_tokens_to_ids("</think>")
        new, seconds = _generate(
            self.model, self.tokenizer, prompt, self.think, [end, *self.eos_ids], temperature=self.think_temperature
        )
        thought = self.tokenizer.decode([i for i in new if i != end], skip_special_tokens=True)
        thought = thought.split("</think>")[0].rstrip()
        return f"{prompt}{thought}\n</think>\n\n", thought, seconds

    def think_many(self, instructions: Sequence[str], batch: int = 16, seed: int = 0) -> None:
        """Write the reasoning for many instructions at once, `batch` at a time (left-padded), and keep it for `write`:
        the same text as one at a time up to the numerics of batching. With think_temperature > 0 each batch samples
        from its own seed (seed + the batch's number). Nothing for a writer that does not think."""
        if not self.think:
            return
        todo = [said for said in dict.fromkeys(instructions) if said not in self._thoughts]
        end = self.tokenizer.convert_tokens_to_ids("</think>")
        side, self.tokenizer.padding_side = self.tokenizer.padding_side, "left"
        try:
            for start in range(0, len(todo), batch):
                part = todo[start : start + batch]
                prompts = [self._think_prompt(self.prompt(said)) for said in part]
                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
                sampled = self.think_temperature > 0
                if sampled:
                    torch.manual_seed(seed + start // batch)
                began = time.perf_counter()
                with torch.no_grad():
                    out = self.model.generate(
                        input_ids=inputs["input_ids"].to(self.model.device),
                        attention_mask=inputs["attention_mask"].to(self.model.device),
                        max_new_tokens=self.think,
                        do_sample=sampled,
                        temperature=self.think_temperature if sampled else None,
                        top_p=0.95 if sampled else None,
                        top_k=20 if sampled else None,
                        num_beams=1,
                        eos_token_id=[end, *self.eos_ids],
                        pad_token_id=self.eos_ids[0] if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id,
                    )
                seconds = (time.perf_counter() - began) / len(part)
                width = inputs["input_ids"].shape[1]
                for said, row in zip(part, out[:, width:].tolist(), strict=True):
                    thought = self.tokenizer.decode([i for i in row if i != end], skip_special_tokens=True)
                    self._thoughts[said] = (thought.split("</think>")[0].rstrip(), seconds)
                print(f"[program_writer] reasoned {start + len(part)} of {len(todo)} instructions", flush=True)
        finally:
            self.tokenizer.padding_side = side


class FactsAnswerer(_Prompted):
    """The comparison of the design's second probe: the same model and the same worked examples, but the model reads
    accurate facts about the plant (facts_text) and names the shoots to cut in one line ("cut: lowest, highest"):
    one step, no program, no constraint. Its worked examples are the program writer's, each with the plant it was
    drawn for."""

    def __init__(
        self, model, tokenizer, shots: int = 8, seed: int = 0, no_think: bool = True, max_new_tokens: int = 96
    ):
        super().__init__(model, tokenizer, no_think, max_new_tokens)
        self.examples = [
            (_facts_question(task, flower), answer_text(task.target_positions))
            for task, flower in _worked_tasks(shots, seed)
        ]

    def prompt(self, task: reasoning.Task, flower: str) -> str:
        return _prompt(self.tokenizer, FACTS_SYSTEM, self.examples, _facts_question(task, flower), self.no_think)

    def answer(self, task: reasoning.Task, flower: str) -> dict:
        """{"reply": the line written, "tokens": the new tokens, "seconds": the time the model took}."""
        reply, new, seconds = self._free(self.prompt(task, flower))
        return dict(reply=reply, tokens=len(new), seconds=seconds)


# -------------------------------------------------------------------------------------------------------------- loading
# In a training checkpoint of a reason_then_act model, the Hugging Face model's own tensors are under this prefix (the
# policy's vision_language_backbone wraps a VLMHF in `_model`, and the VLMHF holds the Hugging Face model as `model`).
BACKBONE_PREFIX = "vision_language_backbone._model.model."


def load_weights(model, path: str) -> dict:
    """Load the language model of a reason_then_act training checkpoint into `model`, the Hugging Face model it was
    trained from: the tensors under BACKBONE_PREFIX, by their names after it (strict=False: the action head and
    anything else is ignored, and a tensor of another shape is skipped). Returns and prints the counts. Raises
    ValueError when no tensor matches: the test would otherwise run on the downloaded weights unnoticed."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # A checkpoint of this repository's own training that holds more than tensors and plain values: read it as
        # vla_foundry reads its checkpoints (file_utils.pt_load).
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint
    # vla_foundry's checkpoint_N.pt holds the weights under "state_dict" and its ema_N.pt under "ema_state_dict"
    for key in ("state_dict", "ema_state_dict", "model"):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get(key), dict):
            state = checkpoint[key]
            break
    own = model.state_dict()
    found, mismatched, unexpected = {}, 0, 0
    for name, tensor in state.items():
        while name.startswith(("module.", "_orig_mod.")):  # wrappers (DDP, torch.compile), as unwrap_state_dict
            name = name.split(".", 1)[1]
        if not name.startswith(BACKBONE_PREFIX):
            continue
        name = name[len(BACKBONE_PREFIX) :]
        if name not in own:
            unexpected += 1
        elif tuple(own[name].shape) != tuple(tensor.shape):
            mismatched += 1
        else:
            found[name] = tensor
    if not found:
        raise ValueError(f"no tensor of {path} under {BACKBONE_PREFIX!r} belongs to {type(model).__name__}")
    model.load_state_dict(found, strict=False)
    report = dict(
        matched=len(found), of=len(own), kept=len(own) - len(found), mismatched=mismatched, unexpected=unexpected
    )
    print(
        f"[program_writer] loaded {report['matched']} of the model's {report['of']} tensors from {path} "
        f"({report['kept']} kept as they were, {mismatched} of another shape skipped, {unexpected} not in the model)",
        flush=True,
    )
    return report


def load_lm(model: str, weights: str | None = None, device: str = "cuda", dtype=None, lora: str | None = None):
    """(model, tokenizer): a language model by Hugging Face name or local directory, in evaluation mode on `device`. A
    vision-language model (its config has a vision_config, as Qwen3-VL's) is loaded whole and used with text only.
    `weights`: a reason_then_act training checkpoint whose language model replaces the downloaded weights
    (load_weights). `lora`: a directory of writer_tune.py's adapters, merged into the weights. `dtype`: bfloat16 on a
    GPU and float32 on the CPU unless given."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    if dtype is None:
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    vision = getattr(config, "vision_config", None) is not None
    loader = AutoModelForImageTextToText if vision else AutoModelForCausalLM
    lm = loader.from_pretrained(model, dtype=dtype, trust_remote_code=True)
    if vision:
        processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", processor)
        if not getattr(tokenizer, "chat_template", None) and getattr(processor, "chat_template", None):
            tokenizer.chat_template = processor.chat_template
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    if weights is not None:
        load_weights(lm, weights)
    lm = lm.to(device)
    if lora is not None:
        from .writer_tune import apply_lora

        apply_lora(lm, lora)
    return lm.eval(), tokenizer


# ------------------------------------------------------------------------------------------------------- the evaluation
def draw_tasks(n: int, seed: int = 0) -> list[tuple[reasoning.Task, str]]:
    """`n` tasks per kind and split, each with the flower of the plant it was drawn for (2 or 3 shoots, any flower, as
    reasoning.lookup_ceiling draws them). The generator is seeded apart from the worked examples' (`examples` with the
    same seed), so that the tasks do not begin with the examples' own draws."""
    rng = np.random.default_rng([seed, 1])
    tasks = []
    for kind in reasoning.KINDS:
        for split in SPLITS:
            for _ in range(n):
                count, flower = _plant(rng)
                tasks.append((reasoning.draw(kind, split, count, flower, rng), flower))
    return tasks


def score(written: str, task: reasoning.Task, flower: str) -> dict:
    """The four flags of a written program, measured apart: `format` (it parses), `exact` (its canonical text is the
    task's program), `meaning` (equivalent to the task's program) and `resolved` (on the task's plant it selects the
    task's shoots; the card's colour is given for the label task); and `got`, the positions it selects there or why it
    selects none."""
    gold = program.from_task(task)
    try:
        node = program.parse(written)
    except program.ProgramError:
        return dict(format=False, exact=False, meaning=False, resolved=False, got="not a program")
    try:
        got = list(program.targets_on(node, task.count, flower, task.params.get("label")))
    except program.NeedsObservation as need:
        got = f"needs the {need.what}"
    except program.ProgramError as error:
        got = str(error)
    return dict(
        format=True,
        exact=program.text(node) == program.text(gold),
        meaning=program.equivalent(node, gold),
        resolved=got == list(task.target_positions),
        got=got,
    )


_ANSWER = re.compile(r"cut: *(lowest|middle|highest)( *, *(lowest|middle|highest))*")


def score_facts(reply: str, task: reasoning.Task) -> dict:
    """The flags of a one-step answer: `format` (the line is "cut: " and names of shoots separated by commas) and
    `resolved` (the shoots named in its last "cut:" are the task's; read as tolerantly as the probe did). It writes
    no program, so `exact` and `meaning` are not measured (None)."""
    said = re.findall(r"cut\s*:\s*([^\n]*)", reply)
    named = [p for p in reasoning.POSITIONS if said and re.search(rf"\b{p}\b", said[-1])]
    return dict(
        format=_ANSWER.fullmatch(reply.strip()) is not None,
        exact=None,
        meaning=None,
        resolved=named == list(task.target_positions),
        got=named,
    )


def _row(task: reasoning.Task, flower: str, gold: str, written: str, flags: dict, out: dict) -> dict:
    return dict(
        kind=task.kind,
        split="train" if task.phrasing == "train" else "heldout",
        phrasing=task.phrasing,
        count=task.count,
        flower=flower,
        label=task.params.get("label"),
        instruction=task.instruction,
        targets=list(task.target_positions),
        gold=gold,
        written=written,
        **flags,
        tokens=out["tokens"],
        seconds=out["seconds"],
    )


def _shown(row: dict, done: int, total: int) -> str:
    flags = " ".join(f"{flag}={'-' if row[flag] is None else int(row[flag])}" for flag in FLAGS)
    where = f"{row['kind']} {row['split']} {row['phrasing']}"
    return f"{done}/{total} {where} {flags} | {row['instruction']} -> {row['written']}"


def evaluate(writer, tasks: Sequence[tuple[reasoning.Task, str]], progress: bool = False) -> list[dict]:
    """One row per task: the program `writer.write` writes for its instruction, scored against the task's (score)."""
    rows = []
    for task, flower in tasks:
        out = writer.write(task.instruction)
        rows.append(_row(task, flower, _program_of(task), out["program"], score(out["program"], task, flower), out))
        if progress:
            print(_shown(rows[-1], len(rows), len(tasks)), flush=True)
    return rows


def evaluate_facts(answerer, tasks: Sequence[tuple[reasoning.Task, str]], progress: bool = False) -> list[dict]:
    """One row per task: the shoots `answerer.answer` names given the facts of the task's plant (score_facts)."""
    rows = []
    for task, flower in tasks:
        out = answerer.answer(task, flower)
        gold = answer_text(task.target_positions)
        rows.append(_row(task, flower, gold, out["reply"], score_facts(out["reply"], task), out))
        if progress:
            print(_shown(rows[-1], len(rows), len(tasks)), flush=True)
    return rows


def _counts(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for flag in FLAGS:
        measured = [row[flag] for row in rows if row[flag] is not None]
        out[flag] = sum(measured) if measured else None
    out["tokens"] = sum(row["tokens"] for row in rows) / len(rows)
    out["seconds"] = sum(row["seconds"] for row in rows) / len(rows)
    return out


def summarize(rows: list[dict]) -> dict[str, dict]:
    """How many rows have each flag, with the mean tokens and seconds, over all rows ("all"), by split ("heldout"), by
    kind and split ("ordinal/heldout"), by kind of held-out instruction ("heldout/frame") and by kind and kind of
    held-out instruction ("ordinal/heldout/frame"). A flag the test does not measure counts None."""
    groups = {"all": rows}
    for split in SPLITS:
        groups[split] = [row for row in rows if row["split"] == split]
    for kind in reasoning.KINDS:
        for split in SPLITS:
            groups[f"{kind}/{split}"] = [row for row in rows if (row["kind"], row["split"]) == (kind, split)]
    for phrasing in reasoning.HELDOUT_KINDS:
        groups[f"heldout/{phrasing}"] = [row for row in rows if row["phrasing"] == phrasing]
    for kind in reasoning.KINDS:
        for phrasing in reasoning.HELDOUT_KINDS:
            groups[f"{kind}/heldout/{phrasing}"] = [
                row for row in rows if (row["kind"], row["phrasing"]) == (kind, phrasing)
            ]
    return {name: _counts(group) for name, group in groups.items() if group}


def table(summary: dict[str, dict]) -> str:
    """The summary as a small table: each flag as "count/n", "-" where it is not measured."""

    def cell(counts: dict, flag: str) -> str:
        return "-" if counts[flag] is None else f"{counts[flag]}/{counts['n']}"

    width = max(len(name) for name in summary)
    lines = [f"{'':{width}}  {'n':>4}" + "".join(f"  {flag:>8}" for flag in FLAGS) + f"  {'tokens':>6}  {'seconds':>7}"]
    for name, counts in summary.items():
        cells = "".join(f"  {cell(counts, flag):>8}" for flag in FLAGS)
        lines.append(f"{name:{width}}  {counts['n']:>4}{cells}  {counts['tokens']:>6.1f}  {counts['seconds']:>7.2f}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(
        description="Write the programs of drawn reasoning tasks with a language model that is not fine-tuned, and "
        "score their format, exact text, meaning and the shoots they resolve to (sim/pickup/program_writer.py)."
    )
    ap.add_argument("--model", required=True, help="a Hugging Face model name or a local directory")
    ap.add_argument("--weights", help="a reason_then_act training checkpoint whose language model is loaded over it")
    ap.add_argument("--lora", help="a directory of writer_tune.py's adapters, merged into the model")
    ap.add_argument("--n", type=int, default=25, help="tasks per kind and split")
    ap.add_argument("--shots", type=int, default=8, help="worked examples in the prompt")
    ap.add_argument("--seed", type=int, default=0, help="the worked examples and the tasks")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=LONGEST_PROGRAM + 1,
        help="the limit on new tokens. The default leaves room for any program and the end token: a token writes at "
        "least one character, so a constrained program always finishes. A model that stops early costs no more",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--unconstrained", action="store_true", help="the baseline: the model writes freely")
    mode.add_argument("--facts", action="store_true", help="the comparison: facts in words, shoots named in one step")
    ap.add_argument("--out", help="the JSON report: the summary and every row")
    args = ap.parse_args(argv)

    model, tokenizer = load_lm(args.model, device=args.device, lora=args.lora)
    loaded = None if args.weights is None else load_weights(model, args.weights)
    tasks = draw_tasks(args.n, args.seed)
    if args.facts:
        answerer = FactsAnswerer(model, tokenizer, args.shots, args.seed, max_new_tokens=args.max_new_tokens)
        shown, rows = answerer.examples, evaluate_facts(answerer, tasks, progress=True)
    else:
        writer = ProgramWriter(
            model,
            tokenizer,
            args.shots,
            args.seed,
            constrained=not args.unconstrained,
            max_new_tokens=args.max_new_tokens,
        )
        shown, rows = writer.examples, evaluate(writer, tasks, progress=True)
    summary = summarize(rows)
    print(table(summary), flush=True)
    report = dict(
        mode="facts" if args.facts else "unconstrained" if args.unconstrained else "constrained",
        model=args.model,
        weights=args.weights,
        lora=args.lora,
        loaded=loaded,
        n=args.n,
        shots=args.shots,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        examples=[list(example) for example in shown],
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
