"""The program writer fine-tuned with LoRA on program writing alone (architecture v3, after step 7).

The writer of step 5 (program_writer_v3.py: Qwen3-VL-2B as downloaded, v3's description and sixteen worked examples)
gets the meaning of 83% of training instructions right, 54% of held-out sentence frames and 56% of held-out
compositions; its errors are polarity and order ("keep the healthy, cut the sick" written as cut the healthy). Here
the same model, with the same prompt, is trained on many more ways of saying the training tasks, with low-rank
adapters (LoRA) on the language model's linear layers and the base weights frozen, so that what it knew of the language
stays (architecture v2 found that fine-tuning the whole model on thirty instructions cost it language).

The training set (`paraphrases`): templates and synonyms for every situation's training tasks (prune, the three
orders, the ties, sick, buds and their opposite polarity), the first set of rung 5's compositions and v2's training
tasks (ordinal, conditional, exclusion, the label), each written as its task's program. Strictly out of it
(`check_excluded`, tested): every held-out instruction (v3's and v2's), every word and every two-word sequence that
only held-out instructions use, and every program equivalent to a held-out composition's (v3's second set and v2's
held-out compositions). Templates are not recombined across tasks: a composition is said in other words, never built
from parts of others, so the held-out compositions stay new combinations.

The adapters (`LoRALinear`, no library): W x + (alpha / r) B A x on q, k, v, o, gate, up and down of every layer of the
language model (not the vision tower), A Gaussian and B zero at the start, trained in float32 on bfloat16 weights. The
loss is on the program and the end of the turn only. Saved as the adapters alone (`lora.pt`, `lora.json`), and merged
into the weights when loaded (`apply_lora`, program_writer.load_lm(lora=...)), so writing is as fast as before.

  python -m sim.pickup.writer_tune data --out paraphrases.json
  python -m sim.pickup.writer_tune train --model <Qwen3-VL-2B> --data paraphrases.json --out <dir>
  python -m sim.pickup.program_writer_v3 --model <Qwen3-VL-2B> --lora <dir> --out writer_lora.json
"""

import argparse
import itertools
import json
import math
import re
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import program, reasoning
from . import program_writer_v3 as V
from .program import All, Count, Cut, CutFrom, Except, Filter, FlowerIs, If, Nothing, Rank, ReadLabel, Take, Then

# -------------------------------------------------------------------------------------------------- what stays held out
_WORD = re.compile(r"[a-z]+")


def _seq(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _bigrams(text: str) -> set[tuple[str, str]]:
    w = _seq(text)
    return set(zip(w, w[1:], strict=False))


def heldout_instructions() -> set[str]:
    """v3's held-out instructions (the situations' held-out frames and the second set of compositions) and v2's."""
    v2 = set().union(*(reasoning.all_heldout_instructions(kind) for kind in reasoning.KINDS))
    return V.heldout_instructions() | v2


def training_instructions() -> set[str]:
    """v3's training instructions (with v2's): what the writer may already be shown."""
    return V.training_instructions()


def forbidden_words() -> set[str]:
    """Words that held-out instructions (v3's or v2's) use and no training instruction does."""
    train = set().union(*(set(_seq(t)) for t in training_instructions()))
    return set().union(*(set(_seq(t)) for t in heldout_instructions())) - train


def forbidden_bigrams() -> set[tuple[str, str]]:
    """Two-word sequences that held-out instructions use and no training instruction does."""
    train = set().union(*(_bigrams(t) for t in training_instructions()))
    return set().union(*(_bigrams(t) for t in heldout_instructions())) - train


def heldout_programs() -> list:
    """The held-out compositions' programs: v3's second set and v2's held-out ordinals, conditionals and exclusion."""
    named = program._NAMED
    v2 = [Cut(Rank(k, end)) for k, end in reasoning.ORDINAL_HELDOUT]
    for colour, yes in reasoning.CONDITIONAL_HELDOUT:
        no = "lowest" if yes == "highest" else "highest"
        v2.append(If(FlowerIs(colour), Cut(named[yes]), Cut(named[no])))
    v2 += [Cut(Except(All(), named[keep])) for keep in reasoning.EXCLUSION_HELDOUT]
    return [node for node, _ in V.HELDOUT_COMPOSITIONS] + v2


# --------------------------------------------------------------------------------------------------------- templates
# Words: none of forbidden_words() (no "to", "of", "by", "each", "only", "leave", "branch", "snip", "spare", ...).
VERBS = ("cut", "prune", "remove", "trim", "clip", "cut off", "take off")
SICK = ("sick", "diseased", "infected", "ill")
HEALTHY = ("healthy", "sound", "disease-free")
WITH_BUD = ("with a bud", "that has a bud", "bearing a bud")  # after one shoot
NO_BUD = ("without a bud", "with no bud", "bearing no bud")
WITH_BUDS = ("with a bud", "that have a bud", "bearing a bud", "with buds")  # after shoots
NO_BUDS = ("without a bud", "that have no bud", "with no bud", "bearing no bud", "without buds")
NAMES = ("lowest", "middle", "highest")
SHOOT = ("shoot", "side shoot")

LOWEST, HIGHEST, MIDDLE = Rank(1, "bottom"), Rank(1, "top"), program.Middle()


def _sick(value="yes"):
    return Filter("sick", value, All())


def _bud(value):
    return Filter("bud", value, All())


def _tie(value):
    return Filter("tie", value, All())


def _fill(templates: Sequence[str], **slots: Sequence[str]) -> list[str]:
    """Every template with every combination of the slots it uses."""
    out = []
    for template in templates:
        used = [name for name in slots if "{" + name + "}" in template]
        for values in itertools.product(*(slots[name] for name in used)):
            out.append(template.format(**dict(zip(used, values, strict=True))))
    return out


def _families() -> list[tuple[str, object, list[str]]]:
    """(family, program, instructions): the training tasks said many ways. Instructions may still hold a forbidden
    word or bigram here; `paraphrases` drops those."""
    colours = V._frames()["tie_colours"]
    fam: list[tuple[str, object, list[str]]] = []

    def add(name, node, templates, **slots):
        fam.append((name, node, _fill(templates, **slots)))

    # pruning, and nothing_to_do's instruction: every side shoot
    add("prune", Cut(All()), (
        "{v} the side shoots off the stem, keep the flower", "{v} every side shoot, keep the flower",
        "{v} all the side shoots", "{v} every side shoot off the stem", "{v} all side shoots but keep the flower",
        "{v} every shoot off the stem and keep the flower", "keep the flower, {v} all the side shoots",
    ), v=VERBS)
    # order
    add("order_top_down", CutFrom("top", All()), (
        "{v} the side shoots from the top down", "{v} the shoots from the highest down",
        "{v} the highest shoot first, then the next one down", "{v} the shoots starting with the highest and working down",
        "{v} every side shoot in turn, the highest first", "starting with the highest shoot, {v} every side shoot in turn",
        "{v} the side shoots in order from the top down", "{v} all the shoots in turn, starting at the top and working down",
        "{v} the highest shoot, then the next one down, and so on", "{v} the shoots one after another from the top down",
        "{v} the side shoots in order, highest first, lowest last",
    ), v=VERBS)
    add("order_bottom_up", CutFrom("bottom", All()), (
        "{v} the side shoots from the bottom up", "{v} the shoots from the lowest up",
        "{v} the lowest shoot first, then the next one up", "{v} the shoots starting with the lowest and working up",
        "{v} every side shoot in turn, the lowest first", "starting with the lowest shoot, {v} every side shoot in turn",
        "{v} the side shoots in order from the bottom up", "{v} all the shoots in turn, starting at the bottom and working up",
        "{v} the lowest shoot, then the next one up, and so on", "{v} the shoots one after another from the bottom up",
        "{v} the side shoots in order, lowest first, highest last",
    ), v=VERBS)
    for a, b in itertools.permutations(NAMES, 2):
        add("order_pair", Then(Cut(program._NAMED[a]), Cut(program._NAMED[b])), (
            "{v} the {a} shoot, then the {b} one", "first {v} the {a} shoot, after that the {b} one",
            "start with the {a} shoot, then {v} the {b} shoot", "{v} the {a} shoot first and the {b} shoot next",
            "{v} the {a} side shoot, and after that the {b} side shoot", "{v} the {a} shoot first, then the {b} one",
            "first the {a} shoot, then the {b} shoot: {v} both in that order",
        ), v=VERBS, a=(a,), b=(b,))
    # ties
    for c in colours:
        add("ties_colour", Cut(_tie(c)), (
            "{v} the shoot with the {c} tie", "{v} the shoot tied with {c}", "{v} the shoot that has a {c} tie",
            "{v} the shoot with a {c} tie on it", "find the {c} tie and {v} its shoot", "{v} the {s} with the {c} tie",
            "{v} the {s} tied with {c}",
        ), v=VERBS, c=(c,), s=SHOOT)
    add("ties_one", Cut(_tie("any")), (
        "{v} the shoot with the tie", "{v} the tied shoot", "{v} the shoot that has a tie", "{v} the shoot with a tie on it",
        "{v} the tied {s}", "{v} the {s} with a tie", "find the tie and {v} its shoot", "{v} every tied shoot",
        "{v} the shoots that have a tie",
    ), v=VERBS, s=SHOOT)
    # sick, and its other polarity
    add("sick_yes", Cut(_sick()), (
        "{v} the {S} shoots", "{v} every {S} shoot", "{v} the {S} shoots, keep the {H} ones",
        "keep the {H} shoots, {v} the {S} ones", "keep the {H} shoots and {v} every other shoot",
        "{v} every shoot except the {H} ones", "{v} all the shoots but the {H} ones", "the {S} shoots need cutting",
        "{v} any shoot that looks {S}", "keep every {H} shoot, {v} the other shoots", "{v} the {S} {s}s and keep the {H} ones",
        "{v} the {S} ones, keep the {H} shoots",
    ), v=VERBS, S=SICK, H=HEALTHY, s=SHOOT)
    add("sick_no", Cut(_sick("no")), (
        "{v} the {H} shoots", "{v} every {H} shoot", "{v} the {H} shoots, keep the {S} ones",
        "keep the {S} shoots, {v} the {H} ones", "keep the {S} shoots and {v} every other shoot",
        "{v} every shoot except the {S} ones", "{v} all the shoots but the {S} ones", "the {H} shoots need cutting",
        "{v} any shoot that looks {H}", "keep every {S} shoot, {v} the other shoots", "{v} the {H} {s}s and keep the {S} ones",
        "{v} the {H} ones, keep the {S} shoots",
    ), v=VERBS, S=SICK, H=HEALTHY, s=SHOOT)
    # buds, and their other polarity ({N}, {B}: after one shoot; {Np}, {Bp}: after shoots)
    add("bud_no", Cut(_bud("no")), (
        "{v} every shoot {N}", "{v} the shoots {Np}", "keep the buds, {v} the other shoots",
        "keep the shoots {Bp}, {v} the other shoots", "keep every shoot {B} and {v} every other shoot",
        "{v} every shoot except the ones {Bp}", "{v} all the shoots but the ones {Bp}", "the shoots {Np} need cutting",
        "keep the buds and {v} every other shoot", "{v} every {s} {N}",
    ), v=VERBS, N=NO_BUD, B=WITH_BUD, Np=NO_BUDS, Bp=WITH_BUDS, s=SHOOT)
    add("bud_yes", Cut(_bud("yes")), (
        "{v} every shoot {B}", "{v} the shoots {Bp}", "keep the shoots {Np}, {v} the other shoots",
        "keep every shoot {N} and {v} every other shoot", "{v} every shoot except the ones {Np}",
        "{v} all the shoots but the ones {Np}", "the shoots {Bp} need cutting", "{v} every {s} {B}",
    ), v=VERBS, N=NO_BUD, B=WITH_BUD, Np=NO_BUDS, Bp=WITH_BUDS, s=SHOOT)
    # the first set of rung 5's compositions, each in other words (never recombined)
    add("compose", Cut(Take(1, "bottom", _bud("yes"))), (
        "{v} the lowest shoot {B}", "among the shoots {Bp}, {v} the lowest one", "{v} the lowest {s} {B}",
        "find the shoots {Bp} and {v} the lowest one",
    ), v=VERBS, B=WITH_BUD, Bp=WITH_BUDS, s=SHOOT)
    add("compose", Cut(Take(1, "any", All())), (
        "{v} one shoot", "{v} any one side shoot", "{v} one side shoot, any one", "pick any shoot and {v} it",
        "{v} one shoot, any one will do", "{v} any one shoot",
    ), v=VERBS)
    add("compose", If(Count("exactly", 0, _sick()), Cut(LOWEST), Nothing()), (
        "if no shoot is {S}, {v} the lowest shoot, otherwise {v} none",
        "{v} the lowest shoot if no shoot is {S}, else {v} none",
        "if every shoot is {H}, {v} the lowest shoot, otherwise {v} none",
        "when no shoot is {S}, {v} the lowest shoot; otherwise {v} none",
    ), v=VERBS, S=SICK, H=HEALTHY)
    add("compose", Cut(Take(1, "top", _sick())), (
        "{v} the highest {S} shoot", "{v} the highest shoot that is {S}", "among the {S} shoots, {v} the highest one",
        "find the {S} shoots and {v} the highest one",
    ), v=VERBS, S=SICK)
    add("compose", CutFrom("bottom", _tie("any")), (
        "{v} the tied shoots from the bottom up", "{v} the shoots with a tie, starting with the lowest and working up",
        "{v} every tied shoot in turn, the lowest first", "{v} the shoots that have a tie from the bottom up",
    ), v=VERBS)
    add("compose", Then(Cut(_tie("blue")), Cut(HIGHEST)), (
        "{v} the shoot with the blue tie, then the highest one",
        "first {v} the shoot tied with blue, after that the highest one",
        "{v} the shoot that has a blue tie first, then the highest shoot",
    ), v=VERBS)
    add("compose", If(Count("more", 1, _bud("yes")), Cut(HIGHEST), Nothing()), (
        "if more than one shoot has a bud, {v} the highest shoot, otherwise {v} none",
        "{v} the highest shoot if more than one shoot has a bud, else {v} none",
        "when more than one shoot has a bud, {v} the highest shoot; otherwise {v} none",
    ), v=VERBS)
    add("compose", Cut(Take(2, "bottom", All())), (
        "{v} the two lowest shoots", "{v} the lowest two side shoots", "{v} the bottom two shoots",
        "{v} the two lowest side shoots",
    ), v=VERBS)
    add("compose", If(Count("fewer", 2, _bud("yes")), Cut(LOWEST), Nothing()), (
        "if fewer than two shoots have a bud, {v} the lowest shoot, otherwise {v} none",
        "{v} the lowest shoot if fewer than two shoots have a bud, else {v} none",
        "when fewer than two shoots have a bud, {v} the lowest shoot; otherwise {v} none",
    ), v=VERBS)
    add("compose", If(Count("exactly", 1, _sick()), Cut(_sick()), Nothing()), (
        "if exactly one shoot is {S}, {v} the {S} shoot, otherwise {v} none",
        "{v} the {S} shoot if exactly one shoot is {S}, else {v} none",
    ), v=VERBS, S=SICK)
    # v2's training tasks, in their frames and v2's training paraphrases (reasoning.py)
    named = program._NAMED
    for k, end in reasoning.ORDINAL_TRAIN:
        texts = [f.format(ord=reasoning.ORDINALS[k], end=end) for f in (*reasoning.ORDINAL_FRAMES, *reasoning.ORDINAL_PARAPHRASES)]
        fam.append(("v2_ordinal", Cut(Rank(k, end)), texts))
    for colour, yes in reasoning.compositions()["conditional_train"]:
        no = "lowest" if yes == "highest" else "highest"
        frames = (*reasoning.CONDITIONAL_FRAMES, *reasoning.CONDITIONAL_PARAPHRASES)
        fam.append(("v2_conditional", If(FlowerIs(colour), Cut(named[yes]), Cut(named[no])),
                    [f.format(colour=colour, yes=yes, no=no) for f in frames]))
    for keep in reasoning.EXCLUSION_TRAIN:
        frames = (*reasoning.EXCLUSION_FRAMES, *reasoning.EXCLUSION_PARAPHRASES)
        fam.append(("v2_exclusion", Cut(Except(All(), named[keep])), [f.format(keep=keep) for f in frames]))
    fam.append(("v2_hidden", Cut(ReadLabel()), [*reasoning.HIDDEN_FRAMES, *reasoning.HIDDEN_PARAPHRASES]))
    return fam


def _clean(text: str, words: set[str], bigrams: set[tuple[str, str]], heldout: set[str]) -> bool:
    return text not in heldout and not set(_seq(text)) & words and not _bigrams(text) & bigrams


def paraphrases(per_program: int = 300, seed: int = 0) -> list[dict]:
    """The training set: {"instruction", "program", "family"} rows, at most `per_program` instructions drawn for each
    program of each family (every one where there are fewer), each kept only if it holds no held-out instruction, word
    or bigram and its program is not equivalent to a held-out composition's. Deterministic for a seed."""
    words, bigrams, heldout = forbidden_words(), forbidden_bigrams(), heldout_instructions()
    banned = heldout_programs()
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, str], set[str]] = {}
    for family, node, texts in _families():
        assert program.in_language(node, "v3"), program.text(node)
        assert not any(program.equivalent_v3(node, other) for other in banned), program.text(node)
        kept = {t for t in texts if _clean(t, words, bigrams, heldout)}
        groups.setdefault((family, program.text(node)), set()).update(kept)
    rows, seen = [], set()
    for (family, written), texts in sorted(groups.items()):
        said_all = sorted(texts)
        if len(said_all) > per_program:
            said_all = [said_all[i] for i in sorted(rng.choice(len(said_all), size=per_program, replace=False))]
        for said in said_all:
            if said in seen:  # one instruction, one program
                continue
            seen.add(said)
            rows.append(dict(instruction=said, program=written, family=family))
    return rows


def check_excluded(rows: Sequence[dict]) -> dict:
    """What `paraphrases` must hold, counted (all zero, or it raises): held-out instructions, forbidden words and
    bigrams in the instructions, programs equivalent to a held-out composition's, and an instruction with two
    programs."""
    words, bigrams, heldout = forbidden_words(), forbidden_bigrams(), heldout_instructions()
    banned = heldout_programs()
    programs = {r["program"] for r in rows}
    out = dict(
        rows=len(rows),
        heldout_instructions=sum(r["instruction"] in heldout for r in rows),
        forbidden_words=sum(bool(set(_seq(r["instruction"])) & words) for r in rows),
        forbidden_bigrams=sum(bool(_bigrams(r["instruction"]) & bigrams) for r in rows),
        heldout_programs=sum(
            any(program.equivalent_v3(program.parse(p, "v3"), other) for other in banned) for p in programs
        ),
        two_programs=len(rows) - len({r["instruction"] for r in rows}),
    )
    bad = {k: v for k, v in out.items() if k != "rows" and v}
    if bad:
        raise AssertionError(f"held-out material in the training set: {bad}")
    return out


# --------------------------------------------------------------------------------------------------------------- LoRA
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def lora_targets(model) -> list[str]:
    """The names of the language model's linear layers that get adapters: TARGETS in every decoder layer, not the
    vision tower's."""
    from torch import nn

    names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in TARGETS and "visual" not in name:
            names.append(name)
    return names


def _lora_class():
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        """W x + (alpha / r) B A x, W frozen; A, B in float32."""

        def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
            super().__init__()
            self.base, self.r, self.scale = base, r, alpha / r
            self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
            self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            delta = self.drop(x).float() @ self.A.t() @ self.B.t()
            return self.base(x) + (self.scale * delta).to(x.dtype)

    return LoRALinear


def add_lora(model, r: int = 16, alpha: float = 32.0, dropout: float = 0.05) -> list[str]:
    """Wrap every target layer of `model` in an adapter; freeze everything else. Returns the names wrapped."""
    LoRALinear = _lora_class()
    for p in model.parameters():
        p.requires_grad_(False)
    names = lora_targets(model)
    for name in names:
        parent_name, child = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child, LoRALinear(getattr(parent, child), r, alpha, dropout))
    return names


def lora_state(model) -> dict:
    """{layer name: (A, B)} of the wrapped layers, on the CPU."""
    out = {}
    for name, module in model.named_modules():
        if hasattr(module, "A") and hasattr(module, "B") and hasattr(module, "base"):
            out[name] = (module.A.detach().cpu(), module.B.detach().cpu())
    return out


def save_lora(model, out: str, config: dict) -> None:
    import torch

    Path(out).mkdir(parents=True, exist_ok=True)
    torch.save(lora_state(model), Path(out) / "lora.pt")
    (Path(out) / "lora.json").write_text(json.dumps(config, indent=1) + "\n")


def apply_lora(model, path: str) -> dict:
    """Merge the adapters saved in `path` (a directory with lora.pt and lora.json) into `model`'s weights:
    W <- W + (alpha / r) B A. Raises if a layer is missing or the shapes differ."""
    import torch

    config = json.loads((Path(path) / "lora.json").read_text())
    state = torch.load(Path(path) / "lora.pt", map_location="cpu", weights_only=True)
    scale = float(config["alpha"]) / int(config["r"])
    with torch.no_grad():
        for name, (A, B) in state.items():
            layer = model.get_submodule(name)
            delta = (B.float() @ A.float()) * scale
            if tuple(delta.shape) != tuple(layer.weight.shape):
                raise ValueError(f"{name}: an adapter of {tuple(delta.shape)} for a weight of {tuple(layer.weight.shape)}")
            layer.weight += delta.to(layer.weight.device, layer.weight.dtype)
    print(f"[writer_tune] merged {len(state)} adapters from {path} (r {config['r']}, alpha {config['alpha']})", flush=True)
    return dict(config, merged=len(state))


# ----------------------------------------------------------------------------------------------------------- training
def _encode(writer, tokenizer, said: str, written: str) -> tuple[list[int], list[int]]:
    """(ids, labels) of one example: the writer's own prompt for the instruction, then the program and the end of the
    turn; labels -100 on the prompt."""
    prompt = writer.prompt(said)
    p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a = tokenizer(written + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return p + a, [-100] * len(p) + a


def train(args) -> dict:
    import torch

    from .program_writer import load_lm

    rows = json.loads(Path(args.data).read_text())["rows"]
    check_excluded(rows)
    model, tokenizer = load_lm(args.model, device=args.device)
    writer = V.writer(model, tokenizer, 16, "A")  # the prompt the writer is used with: the second prompt, set A, 16
    examples = [_encode(writer, tokenizer, r["instruction"], r["program"]) for r in rows]
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model.config.use_cache = False
    names = add_lora(model, args.r, args.alpha, args.dropout)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[writer_tune] {len(rows)} examples, {len(names)} adapters, {sum(p.numel() for p in params):,} parameters", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = math.ceil(len(examples) * args.epochs / args.batch)
    warm = max(1, steps // 20)

    def lr_at(step):
        if step < warm:
            return args.lr * (step + 1) / warm
        return args.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))

    order = np.concatenate([rng.permutation(len(examples)) for _ in range(math.ceil(args.epochs))])
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    began, log = time.time(), []
    for step in range(steps):
        batch = [examples[i] for i in order[step * args.batch : (step + 1) * args.batch]]
        width = max(len(ids) for ids, _ in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            ids[i, : len(x)], labels[i, : len(y)], mask[i, : len(x)] = torch.tensor(x), torch.tensor(y), 1
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.model(input_ids=ids.to(args.device), attention_mask=mask.to(args.device)).last_hidden_state
            # the logits only where a label is (the program): the vocabulary is 152k wide, the prompt 1300 tokens long
            target = labels[:, 1:].to(args.device)
            where = target != -100
            logits = model.lm_head(hidden[:, :-1][where])
            loss = torch.nn.functional.cross_entropy(logits.float(), target[where])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == steps - 1:
            log.append(dict(step=step, loss=float(loss), lr=lr_at(step), seconds=time.time() - began))
            print(f"[writer_tune] step {step}/{steps} loss {float(loss):.4f} lr {lr_at(step):.2e} {time.time() - began:.0f}s", flush=True)
    config = dict(r=args.r, alpha=args.alpha, dropout=args.dropout, lr=args.lr, epochs=args.epochs, batch=args.batch,
                  seed=args.seed, steps=steps, examples=len(rows), targets=list(TARGETS), data=args.data,
                  prompt="second, examples A, 16", seconds=time.time() - began, log=log)
    save_lora(model, args.out, config)
    return config


# ---------------------------------------------------------------------------------------------------- the command line
def main(argv: Sequence[str] | None = None):
    ap = argparse.ArgumentParser(description="LoRA fine-tuning of v3's program writer on program writing alone")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data", help="write the training paraphrases (and check what stays held out)")
    d.add_argument("--per_program", type=int, default=300)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out", required=True)
    t = sub.add_parser("train", help="train the adapters")
    t.add_argument("--model", required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--r", type=int, default=16)
    t.add_argument("--alpha", type=float, default=32.0)
    t.add_argument("--dropout", type=float, default=0.05)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--epochs", type=float, default=2.0)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if args.cmd == "data":
        rows = paraphrases(args.per_program, args.seed)
        checks = check_excluded(rows)
        families = {}
        for r in rows:
            families[r["family"]] = families.get(r["family"], 0) + 1
        print(json.dumps(dict(checks=checks, families=families), indent=1), flush=True)
        Path(args.out).write_text(json.dumps(dict(checks=checks, families=families, rows=rows), indent=1) + "\n")
    else:
        train(args)


if __name__ == "__main__":
    main()
