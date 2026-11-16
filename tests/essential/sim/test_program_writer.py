"""The program writer (sim/pickup/program_writer.py): the grammar constraint over a tokenizer's vocabulary is exact,
constrained writing always ends in a whole program and free writing does not, the worked examples and the prompts use
training wording only, a training checkpoint's language model loads by name, and the evaluation scores format, exact
text, meaning and the shoots resolved apart.

Nothing is downloaded: the tokenizer is a small byte-level BPE built here (every byte a token, some longer tokens of
program text, some that never are) and the model a tiny GPT-2 with random weights. Nothing here needs MuJoCo.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:  # sim/ is not an installed package
    sys.path.insert(0, str(REPOSITORY))

from sim.pickup import program as P  # noqa: E402
from sim.pickup import program_writer as W  # noqa: E402
from sim.pickup import reasoning as R  # noqa: E402

EOS = "<|endoftext|>"
# Tokens longer than a character, in the byte-level alphabet (a space is "Ġ", and "é" is the two bytes "Ã©"). BPE builds
# each from its prefixes, so every prefix is a token too.
PROGRAM_PIECES = "Cut( Rank( ReadLabel() top bottom If( FlowerIs( All() Middle() Except( Filter( red pink purple"
PROGRAM_PIECES += " tie, )) ),Cut( 2,top"
NEVER_NEXT = "the ok Cutting"  # program characters that no program continues with, or only up to a point
NOT_PROGRAM = "Ġthe ĠCut( (? zz Ã©"  # a leading space, characters outside the alphabet
# Room for the longest program, a character per token, and the end (test_the_longest_program_is_as_long_as_stated)
LIMIT = W.LONGEST_PROGRAM + 1
INSTRUCTIONS = (
    "cut the third shoot from the top",
    "unless the flower is red, remove the lowest shoot; if it is, remove the highest one",
    "leave only the middle shoot and cut the rest",
    "check the tag in the pot, then prune the shoot it points to",
)


def _toy_tokenizer():
    """A byte-level BPE tokenizer like GPT-2's or Qwen's, small: the 256 byte characters, the longer tokens above, and
    an end-of-sequence token. Built afresh each time: tests change it."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    merges = []
    for word in f"{PROGRAM_PIECES} {NEVER_NEXT} {NOT_PROGRAM}".split():
        for end in range(2, len(word) + 1):
            if word[:end] not in vocab:
                vocab[word[:end]] = len(vocab)
                merges.append((word[: end - 1], word[end - 1]))
    backend = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token=EOS)


def _model(tokenizer, seed=0):
    """A tiny GPT-2 with random weights over the tokenizer's vocabulary. Its input and output embeddings are separate:
    tied, a random model mostly repeats the last token it read, which is the line break that ends the prompt."""
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(seed)
    end = tokenizer.eos_token_id
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=4096,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=end,
        eos_token_id=end,
        tie_word_embeddings=False,
    )
    return GPT2LMHeadModel(config).eval()


def _true_texts(tokenizer) -> dict[int, str]:
    """What each token of the toy vocabulary writes, worked out from its bytes (not by decoding, as the module does);
    the end-of-sequence token writes nothing."""
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

    byte_of = {char: byte for byte, char in bytes_to_unicode().items()}
    return {
        token_id: bytes(byte_of[char] for char in token).decode("utf-8", errors="replace")
        for token, token_id in tokenizer.get_vocab().items()
        if token != EOS
    }


def _brute_force(texts, end, written) -> set[int]:
    """The tokens allowed after `written`, from the definition: each character of the token's text allowed in turn
    (`texts`: every token's text, `end`: the end-of-sequence token)."""
    allowed = {
        token_id
        for token_id, text in texts.items()
        if text and all(text[i] in P.next_chars(written + text[:i]) for i in range(len(text)))
    }
    if P.is_complete(written):
        allowed.add(end)
    return allowed or {end}


def _random_prefixes(n, seed):
    """Beginnings of programs, from random walks through the allowed characters, and the whole programs they end in."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        written, stop = "", rng.randrange(80)
        while len(written) < stop and (allowed := P.next_chars(written)):
            written += rng.choice(sorted(allowed))
        out.append(written)
    return out


def _save(model, tokenizer, directory):
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return str(directory)


# ------------------------------------------------------------------------------------------------------- the constraint
def test_a_token_s_text_is_what_it_adds_to_the_decoded_text():
    tokenizer = _toy_tokenizer()
    vocab = tokenizer.get_vocab()
    texts = W.token_texts(tokenizer)
    assert texts == _true_texts(tokenizer)
    assert texts[vocab["Ġthe"]] == " the" and texts[vocab["Ã©"]] == "é" and texts[vocab["Ã"]] == "�"
    assert texts[vocab["Cut("]] == "Cut(" and tokenizer.eos_token_id not in texts


def test_a_leading_space_counts_where_a_decoder_drops_it_at_the_start_of_a_text():
    """A SentencePiece-style vocabulary marks a leading space with "▁", and its decoder drops the space at the start of
    a text: decoded alone, "▁Cut(" reads "Cut(". After another token it reads " Cut(", which is not program text."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {name: i for i, name in enumerate(("<unk>", "</s>", "▁", "a", "▁a", "Cut(", "▁Cut(", "All()", ")"))}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Metaspace()
    backend.decoder = decoders.Metaspace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="</s>", unk_token="<unk>")
    assert tokenizer.decode([vocab["▁Cut("]]) == "Cut("
    texts = W.token_texts(tokenizer)
    assert texts[vocab["▁Cut("]] == " Cut(" and texts[vocab["Cut("]] == "Cut(" and texts[vocab["▁"]] == " "
    assert set(W.TokenTrie(texts).texts) == {vocab[name] for name in ("a", "Cut(", "All()", ")")}


def test_the_trie_holds_exactly_the_tokens_made_of_program_characters():
    tokenizer = _toy_tokenizer()
    trie = W.TokenTrie(W.token_texts(tokenizer))
    expected = {i: text for i, text in _true_texts(tokenizer).items() if text and set(text) <= P.ALPHABET}
    assert trie.texts == expected
    names = {token for token, token_id in tokenizer.get_vocab().items() if token_id in trie.texts}
    assert {"C", "(", "Cut(", "ReadLabel()", "),Cut(", "the", "Cutting"} <= names
    assert not {"Ġ", "Ġthe", "ĠCut(", "Ċ", "(?", "zz", "Ã©", EOS} & names
    held, todo = [], [trie.root]
    while todo:  # every token sits at the end of the path that spells its text, and nowhere else
        node = todo.pop()
        held += node.ids
        todo += node.children.values()
    assert sorted(held) == sorted(trie.texts)
    for token_id, text in trie.texts.items():
        node = trie.root
        for char in text:
            node = node.children[char]
        assert token_id in node.ids


def test_allowed_tokens_are_exactly_those_after_which_a_program_is_still_possible():
    tokenizer = _toy_tokenizer()
    trie = W.TokenTrie(W.token_texts(tokenizer))
    texts, end = _true_texts(tokenizer), tokenizer.eos_token_id
    named = ["", "C", "Cut(", "Cut(R", "Cut(Ra", "Cut(Rank(2,", "If(FlowerIs(p", "If(FlowerIs(red),Cut(All()"]
    named += ["Cut(All())", "If(FlowerIs(red),Cut(All()),Cut(Middle()))", "Cut)", "cut(", "Cut( "]
    for written in named + _random_prefixes(300, seed=0):
        allowed = W.allowed_tokens(trie, written, [end])
        assert allowed and len(allowed) == len(set(allowed))
        assert set(allowed) == _brute_force(texts, end, written), written


def test_allowed_tokens_at_the_start_in_a_word_where_words_share_a_beginning_and_at_the_end():
    tokenizer = _toy_tokenizer()
    trie = W.TokenTrie(W.token_texts(tokenizer))
    token = {token_id: name for name, token_id in tokenizer.get_vocab().items()}

    def allowed(written):
        return {token[token_id] for token_id in W.allowed_tokens(trie, written, [tokenizer.eos_token_id])}

    def prefixes(*words):
        return {word[:end] for word in words for end in range(1, len(word) + 1)}

    assert allowed("") == prefixes("Cut(", "If(")  # not "Cutt": no program goes on so
    # a selection: every token that begins one of the six; "Fl" begins FlowerIs(, which is no selection
    assert allowed("Cut(") == prefixes("All()", "Rank(", "ReadLabel()", "Middle()", "Except(", "Filter(")
    assert allowed("Cut(R") == {"a", "e"}  # Rank( or ReadLabel(): the next letter decides
    assert allowed("Cut(Ra") == {"n"}  # in the middle of a word, one way on
    assert allowed("If(FlowerIs(p") == {"i", "u"}  # pink or purple
    assert allowed("Cut(Rank(2,") == prefixes("top", "bottom")  # not "th" of "the"
    # a token may close one part and begin the next: "),Cut(" ends the first branch and begins the second
    assert allowed("If(FlowerIs(red),Cut(All()") == {")", "),", "),C", "),Cu", "),Cut", "),Cut("}
    assert allowed("Cut(All())") == {EOS}  # a whole program: only the end
    assert allowed("Cut)") == {EOS}  # no program begins so: the end, never an empty list


# -------------------------------------------------------------------------------------------------------------- writing
def test_constrained_writing_always_ends_in_a_whole_program():
    tokenizer = _toy_tokenizer()
    written = set()
    for seed in range(6):
        writer = W.ProgramWriter(_model(tokenizer, seed), tokenizer, shots=2, max_new_tokens=LIMIT)
        for instruction in INSTRUCTIONS:
            out = writer.write(instruction)
            assert out["complete"] and P.text(P.parse(out["program"])) == out["program"], out
            assert 1 <= out["tokens"] <= LIMIT and out["seconds"] >= 0
            written.add(out["program"])
    assert len(written) > 1  # random weights take different paths through the grammar


def test_a_constrained_writer_stops_at_its_limit_on_new_tokens():
    tokenizer = _toy_tokenizer()
    for seed in range(3):
        writer = W.ProgramWriter(_model(tokenizer, seed), tokenizer, shots=2, max_new_tokens=3)
        for instruction in INSTRUCTIONS:
            out = writer.write(instruction)
            assert 1 <= out["tokens"] <= 3
            # cut short or not, what it wrote begins a program
            assert out["complete"] or P.next_chars(out["program"]), out


def test_free_writing_by_a_random_model_is_rarely_a_program():
    tokenizer = _toy_tokenizer()
    outs = [
        W.ProgramWriter(_model(tokenizer, seed), tokenizer, shots=2, constrained=False, max_new_tokens=40).write(said)
        for seed in range(6)
        for said in INSTRUCTIONS
    ]
    assert all(1 <= out["tokens"] <= 40 and "\n" not in out["program"] for out in outs)
    assert sum(out["complete"] for out in outs) <= len(outs) // 4  # constrained: every one (the test above)


def test_the_longest_program_is_as_long_as_stated():
    """The longest selection, over every selection within the depth bounds, and so the longest program."""
    leaves = [P.All(), P.Middle(), P.ReadLabel(), *(P.Rank(k, end) for k in (1, 2, 3) for end in ("top", "bottom"))]
    pairs = [(attr, value) for attr, values in P.ATTRIBUTES.items() for value in values]
    selections = leaves
    for _ in range(P.MAX_NESTING):
        inner = selections
        selections = [*leaves, *(P.Except(a, b) for a in inner for b in inner)]
        selections += [P.Filter(attr, value, s) for attr, value in pairs for s in inner]
    longest = max(selections, key=lambda s: len(P.text(s)))
    colour = max(R.FLOWERS, key=len)
    whole = P.text(P.If(P.FlowerIs(colour), P.Cut(longest), P.Cut(longest)))
    assert len(whole) == W.LONGEST_PROGRAM and P.is_complete(whole)


# ---------------------------------------------------------------------------------------------------------- the prompts
def test_worked_examples_are_training_instructions_of_every_kind():
    pairs = W.examples(8, seed=0)
    assert pairs == W.examples(8, seed=0) and len(pairs) == 8
    assert any(W.examples(8, seed) != pairs for seed in (1, 2, 3))
    program_of = {}  # every training instruction and its program
    for kind in R.KINDS:
        rng = np.random.default_rng(0)
        for _ in range(400):
            task = R.draw(kind, "train", int(rng.integers(2, 4)), R.FLOWERS[int(rng.integers(4))], rng)
            program_of[task.instruction] = P.text(P.from_task(task))
    assert set(pairs) <= set(program_of.items())
    kinds = [next(kind for kind in R.KINDS if said in R.training_instructions(kind)) for said, _ in pairs]
    assert kinds == [*R.KINDS, *R.KINDS]
    heldout = set().union(*(R.all_heldout_instructions(kind) for kind in R.KINDS))
    assert len({said for said, _ in pairs}) == 8 and not {said for said, _ in pairs} & heldout
    for kind in ("ordinal", "conditional", "exclusion"):  # two compositions each (the label task has one program)
        assert len({written for (_, written), k in zip(pairs, kinds, strict=True) if k == kind}) == 2


def test_the_prompt_describes_the_language_then_every_example_then_the_instruction():
    tokenizer = _toy_tokenizer()
    model = _model(tokenizer)
    for construct in ("Cut(S)", "If(FlowerIs(C)", "All()", "Rank(K,top)", "Rank(K,bottom)", "Middle()", "Except("):
        assert construct in W.SYSTEM
    assert "ReadLabel()" in W.SYSTEM and "Filter(A,V,S)" in W.SYSTEM
    instruction = "snip the third branch from the top"
    # no chat template: plain text, ending with the instruction and a line break for the program
    writer = W.ProgramWriter(model, tokenizer)
    prompt = writer.prompt(instruction)
    assert prompt.startswith(W.SYSTEM) and prompt.endswith(f"\nInstruction: {instruction}\n")
    at = 0
    for said, written in writer.examples:
        at = prompt.index(f"\nInstruction: {said}\n{written}\n", at) + 1
    assert prompt.index(f"Instruction: {instruction}") > at
    # a chat template: a system turn, a user turn and the answer per example, the instruction last
    tokenizer.chat_template = (
        "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}"
    )
    shown = "".join(f"<|user|>Instruction: {s}<|end|><|assistant|>{w}<|end|>" for s, w in writer.examples)
    expected = f"<|system|>{W.SYSTEM}<|end|>{shown}<|user|>Instruction: {instruction}<|end|><|assistant|>"
    assert W.ProgramWriter(model, tokenizer).prompt(instruction) == expected
    # a thinking model is made to answer at once, with one empty block whether or not the template opens one
    tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
    assert W.ProgramWriter(model, tokenizer).prompt(instruction) == expected + "<think>\n\n</think>\n\n"
    assert W.ProgramWriter(model, tokenizer, no_think=False).prompt(instruction) == expected
    opened = "<|assistant|><think>\n{% endif %}"
    tokenizer.chat_template = tokenizer.chat_template.replace("<|assistant|>{% endif %}", opened)
    assert W.ProgramWriter(model, tokenizer).prompt(instruction) == expected + "<think>\n\n</think>\n\n"


def test_the_prompts_use_no_word_that_only_held_out_instructions_use():
    """Except "middle", the name of Middle() and of a shoot: no training instruction says it."""
    forbidden = R.heldout_only_words()
    tokenizer = _toy_tokenizer()
    model = _model(tokenizer)
    task = R.Task("ordinal", "cut the second shoot from the top", "train", 3, (1,), "second from the top", {})
    prompts = (
        W.ProgramWriter(model, tokenizer).prompt(task.instruction),
        W.FactsAnswerer(model, tokenizer).prompt(task, "red"),
    )
    for text in (W.SYSTEM, W.FACTS_SYSTEM, *prompts):
        assert R._words(text) & forbidden <= {"middle"}, R._words(text) & forbidden


def test_the_facts_are_accurate_and_the_examples_answer_for_their_own_plants():
    assert W.facts_text(2, "red", None) == (
        "Side shoots, from the bottom of the stem up: lowest, highest (2 shoots).\n"
        "Flower: red.\n"
        "Label in the pot: none.\n"
        "Label rule: a white label names the lowest shoot, a yellow label names the middle shoot, a cyan label names "
        "the highest shoot."
    )
    assert "lowest, middle, highest (3 shoots)" in W.facts_text(3, "pink", "cyan")
    assert "Label in the pot: cyan." in W.facts_text(3, "pink", "cyan")
    tokenizer = _toy_tokenizer()
    answerer = W.FactsAnswerer(_model(tokenizer), tokenizer)
    said = [question.rsplit("\nInstruction: ", 1)[1] for question, _ in answerer.examples]
    assert said == [instruction for instruction, _ in W.examples()]  # the program writer's examples
    program_of = dict(W.examples())
    for question, answer in answerer.examples:  # each answer is right for the plant its facts describe
        count = 3 if "(3 shoots)" in question else 2
        flower = question.split("Flower: ", 1)[1].split(".", 1)[0]
        label = question.split("Label in the pot: ", 1)[1].split(".", 1)[0]
        label = None if label == "none" else label
        written = program_of[question.rsplit("\nInstruction: ", 1)[1]]
        assert answer == W.answer_text(P.targets_on(P.parse(written), count, flower, label))
    task = R.Task("hidden", INSTRUCTIONS[3], "frame", 3, (2,), "the shoot the label names", {"label": "cyan"})
    prompt = answerer.prompt(task, "pink")
    assert prompt.endswith(f"\n{W.facts_text(3, 'pink', 'cyan')}\nInstruction: {INSTRUCTIONS[3]}\n")
    assert all(f"{question}\n{answer}\n" in prompt for question, answer in answerer.examples)


# -------------------------------------------------------------------------------------------------------------- loading
def _checkpoint(trained, **extra):
    """A reason_then_act training checkpoint's state dict: the language model under its prefix, and more."""
    state = {W.BACKBONE_PREFIX + name: tensor.clone() for name, tensor in trained.state_dict().items()}
    state["action_head.out.weight"] = torch.zeros(3)  # not the language model's: ignored
    state.update(extra)
    return state


def test_load_lm_puts_a_training_checkpoint_s_language_model_in_place(tmp_path, capsys):
    tokenizer = _toy_tokenizer()
    downloaded, trained = _model(tokenizer, 0), _model(tokenizer, 1)
    directory = _save(downloaded, tokenizer, tmp_path / "lm")
    extra = {
        W.BACKBONE_PREFIX + "transformer.wpe.weight": torch.zeros(5, 5),  # another shape: skipped
        W.BACKBONE_PREFIX + "transformer.no_such.weight": torch.zeros(3),  # not in the model
    }
    torch.save({"checkpoint_num": 7, "state_dict": _checkpoint(trained, **extra), "global_step": 70}, tmp_path / "c.pt")
    model, loaded_tokenizer = W.load_lm(directory, weights=str(tmp_path / "c.pt"), device="cpu")
    own, before = trained.state_dict(), downloaded.state_dict()
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name] if name == "transformer.wpe.weight" else own[name]), name
    assert not torch.equal(own["transformer.h.0.attn.c_attn.weight"], before["transformer.h.0.attn.c_attn.weight"])
    assert not model.training and len(loaded_tokenizer) == len(tokenizer)
    printed = capsys.readouterr().out
    assert f"loaded {len(own) - 1} of the model's {len(own)} tensors" in printed
    assert "1 kept as they were, 1 of another shape skipped, 1 not in the model" in printed
    # without weights: the model as saved
    plain, _ = W.load_lm(directory, device="cpu")
    assert all(torch.equal(tensor, before[name]) for name, tensor in plain.state_dict().items())


@pytest.mark.parametrize("nesting", ["state_dict", "ema_state_dict", "model", None])
def test_load_weights_reads_training_and_averaged_checkpoints(tmp_path, nesting):
    tokenizer = _toy_tokenizer()
    trained, model = _model(tokenizer, 1), _model(tokenizer, 0)
    state = _checkpoint(trained)
    if nesting is None:  # a bare state dict, saved from a wrapped model (DDP)
        torch.save({f"module.{name}": tensor for name, tensor in state.items()}, tmp_path / "c.pt")
    else:
        torch.save({nesting: state, "global_step": 3, "samples_seen": np.int64(30)}, tmp_path / "c.pt")
    report = W.load_weights(model, str(tmp_path / "c.pt"))
    n = len(trained.state_dict())
    assert report == dict(matched=n, of=n, kept=0, mismatched=0, unexpected=0)
    assert all(torch.equal(tensor, trained.state_dict()[name]) for name, tensor in model.state_dict().items())


def test_a_checkpoint_without_the_language_model_is_refused(tmp_path):
    tokenizer = _toy_tokenizer()
    torch.save({"state_dict": {"action_head.out.weight": torch.zeros(3)}}, tmp_path / "c.pt")
    with pytest.raises(ValueError, match="no tensor"):
        W.load_weights(_model(tokenizer), str(tmp_path / "c.pt"))


# ------------------------------------------------------------------------------------------------------- the evaluation
def _ordinal(count):
    """The second shoot from the top, on a plant with `count` shoots: the middle of three, the lowest of two."""
    said = "cut the second shoot from the top"
    return R.Task("ordinal", said, "train", count, (count - 2,), "second from the top", dict(k=2, end="top"))


def test_a_written_program_is_scored_on_form_text_meaning_and_shoots_apart():
    def flags(written, task=None, flower="red"):
        scored = W.score(written, task or _ordinal(3), flower)
        return tuple(scored[flag] for flag in W.FLAGS), scored["got"]

    assert flags("Cut(Rank(2,top))") == ((True, True, True, True), ["middle"])
    assert flags("Cut(Rank(2, top))") == ((True, True, True, True), ["middle"])  # spaces: the same program
    # the same meaning, spelled otherwise
    assert flags("Cut(Except(Rank(2,top),Except(All(),All())))") == ((True, False, True, True), ["middle"])
    # on three shoots the second from the bottom is the middle one too, but on two it is the highest
    assert flags("Cut(Rank(2,bottom))") == ((True, False, False, True), ["middle"])
    assert flags("Cut(Rank(2,bottom))", _ordinal(2)) == ((True, False, False, False), ["highest"])
    assert flags("Cut(Middle())", _ordinal(2)) == ((True, False, False, False), "no middle shoot on a plant with 2")
    assert flags("Cut(ReadLabel())") == ((True, False, False, False), "needs the label")  # this plant has no card
    for malformed in ("Cut(Rank(2,top)", "cut the middle shoot", ""):
        assert flags(malformed) == ((False, False, False, False), "not a program")
    said = "read the label in the pot and cut the shoot it names"
    card = R.Task("hidden", said, "train", 3, (2,), "the shoot the label names", dict(label="cyan"))
    assert flags("Cut(ReadLabel())", card, "pink") == ((True, True, True, True), ["highest"])
    branch = R.draw("conditional", "train", 3, "red", np.random.default_rng(0))
    swapped = P.from_task(branch)
    swapped = P.text(P.If(swapped.cond, swapped.no, swapped.yes))
    assert flags(swapped, branch) == ((True, False, False, False), [R.POSITIONS[2 - branch.targets[0]]])


class _Stub:
    """Writes each task's program, or a mistake by kind: the program with spaces (ordinal: right), the branches swapped
    (conditional: wrong), the same meaning spelled otherwise (exclusion), a parenthesis short (hidden: no program)."""

    def __init__(self, tasks):
        self.task_of = {task.instruction: task for task, _ in tasks}  # the instruction decides the program

    def write(self, instruction):
        task = self.task_of[instruction]
        gold = P.from_task(task)
        if task.kind == "ordinal":
            written = P.text(gold).replace(",", ", ")
        elif task.kind == "conditional":
            written = P.text(P.If(gold.cond, gold.no, gold.yes))
        elif task.kind == "exclusion":
            written = P.text(P.Cut(P.Except(gold.sel, P.Except(P.All(), P.All()))))
        else:
            written = P.text(gold)[:-1]
        return dict(program=written, complete=task.kind != "hidden", tokens=len(written), seconds=0.5)


def test_the_evaluation_counts_each_flag_by_kind_split_and_held_out_phrasing():
    tasks = W.draw_tasks(5, seed=0)
    assert len(tasks) == 40 and tasks == W.draw_tasks(5, seed=0)
    rows = W.evaluate(_Stub(tasks), tasks)
    assert [row["instruction"] for row in rows] == [task.instruction for task, _ in tasks]
    for row, (task, flower) in zip(rows, tasks, strict=True):
        assert row["gold"] == P.text(P.from_task(task)) and row["flower"] == flower
        assert row["targets"] == list(task.target_positions) and row["tokens"] == len(row["written"])
    summary = W.summarize(rows)
    expected = {  # format, exact, meaning, resolved of every row of a kind
        "ordinal": (1, 1, 1, 1),
        "conditional": (1, 0, 0, 0),
        "exclusion": (1, 0, 1, 1),
        "hidden": (0, 0, 0, 0),
    }
    for kind, ones in expected.items():
        for split in W.SPLITS:
            counts = summary[f"{kind}/{split}"]
            assert counts["n"] == 5 and tuple(counts[flag] for flag in W.FLAGS) == tuple(5 * x for x in ones)
    assert tuple(summary["all"][flag] for flag in W.FLAGS) == (30, 10, 20, 20) and summary["all"]["n"] == 40
    assert summary["train"]["n"] == summary["heldout"]["n"] == 20
    assert summary["all"]["seconds"] == pytest.approx(0.5)
    assert summary["all"]["tokens"] == pytest.approx(np.mean([len(row["written"]) for row in rows]))
    for phrasing in R.HELDOUT_KINDS:  # by kind of held-out instruction, alone and by kind
        held = [row for row in rows if row["phrasing"] == phrasing]
        assert summary.get(f"heldout/{phrasing}", {"n": 0})["n"] == len(held)
        for kind in R.KINDS:
            n = sum(row["kind"] == kind for row in held)
            assert summary.get(f"{kind}/heldout/{phrasing}", {"n": 0})["n"] == n
    assert "hidden/heldout/frame" in summary and "hidden/heldout/composition" not in summary
    printed = W.table(summary)
    assert "ordinal/heldout" in printed and "30/40" in printed


def test_one_step_answers_are_scored_on_form_and_shoots():
    said = "cut every side shoot except the highest one"
    task = R.Task("exclusion", said, "train", 3, (0, 1), "all but the highest", dict(keep="highest"))

    def flags(reply):
        scored = W.score_facts(reply, task)
        return tuple(scored[flag] for flag in W.FLAGS)

    assert flags("cut: lowest, middle") == (True, None, None, True)
    assert flags("cut: middle,lowest") == (True, None, None, True)  # the order does not matter
    assert flags("cut: lowest") == (True, None, None, False)
    assert flags("cut: the lowest and the middle one") == (False, None, None, True)  # right shoots, loose form
    assert flags("the lowest and the middle") == (False, None, None, False)

    class Answers:
        def answer(self, task, flower):
            right = task.kind in ("ordinal", "hidden")
            reply = W.answer_text(task.target_positions if right else ["middle"] * 2)
            return dict(reply=reply, tokens=3, seconds=0.1)

    tasks = W.draw_tasks(2, seed=1)
    rows = W.evaluate_facts(Answers(), tasks)
    summary = W.summarize(rows)
    assert summary["all"]["exact"] is None and summary["all"]["meaning"] is None
    assert summary["all"]["format"] == 16 and summary["all"]["resolved"] == 8
    assert rows[0]["gold"] == W.answer_text(tasks[0][0].target_positions)
    line = next(line for line in W.table(summary).splitlines() if line.startswith("all "))
    assert line.split()[1:6] == ["16", "16/16", "-", "-", "8/16"]  # exact and meaning are not measured


def test_the_command_line_runs_each_test_and_writes_a_report_with_every_row(tmp_path, capsys):
    tokenizer = _toy_tokenizer()
    directory = _save(_model(tokenizer, 0), tokenizer, tmp_path / "lm")
    common = ["--model", directory, "--device", "cpu", "--n", "1", "--shots", "2"]
    report = W.main([*common, "--out", str(tmp_path / "writer.json")])
    assert report["mode"] == "constrained" and report["loaded"] is None and len(report["rows"]) == 8
    assert report["max_new_tokens"] == LIMIT and report["summary"]["all"]["format"] == 8  # always a program
    saved = json.loads((tmp_path / "writer.json").read_text())
    assert saved["summary"] == report["summary"] and saved["examples"] == [list(e) for e in W.examples(2)]
    for row in saved["rows"]:
        assert {"instruction", "gold", "written", *W.FLAGS, "kind", "split", "phrasing", "tokens"} <= set(row)
    assert "ordinal/heldout" in capsys.readouterr().out
    free = W.main([*common, "--unconstrained"])
    assert free["mode"] == "unconstrained"
    assert [row["instruction"] for row in free["rows"]] == [row["instruction"] for row in report["rows"]]
    facts = W.main([*common, "--facts"])
    assert facts["mode"] == "facts" and facts["summary"]["all"]["exact"] is None and len(facts["rows"]) == 8
    trained = _model(tokenizer, 1)
    torch.save({"state_dict": _checkpoint(trained)}, tmp_path / "c.pt")
    tuned = W.main([*common, "--weights", str(tmp_path / "c.pt")])
    assert tuned["loaded"]["matched"] == len(trained.state_dict()) and tuned["weights"] == str(tmp_path / "c.pt")
    with pytest.raises(SystemExit):  # one test at a time
        W.main([*common, "--facts", "--unconstrained"])
