"""The LoRA-tuned program writer (sim/pickup/writer_tune.py): its training paraphrases hold no held-out instruction,
no word or two-word sequence that only held-out instructions use and no program equivalent to a held-out
composition's; every family of training tasks is there in both polarities; the adapters start as the identity, train
only themselves, and merged into the weights give what the wrapped layers gave.

The situations' frames come from scenarios.py, which needs MuJoCo.
"""

import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:  # sim/ is not an installed package
    sys.path.insert(0, str(REPOSITORY))

from sim.pickup import program as P  # noqa: E402
from sim.pickup import program_writer_v3 as V  # noqa: E402
from sim.pickup import writer_tune as T  # noqa: E402


@pytest.fixture(scope="module")
def rows():
    return T.paraphrases()


def test_nothing_held_out_is_in_the_training_paraphrases(rows):
    checks = T.check_excluded(rows)
    assert checks["rows"] == len(rows) > 2000
    heldout = T.heldout_instructions()
    assert set(V.heldout_instructions()) <= heldout and len(heldout) > len(V.heldout_instructions())
    words, bigrams = T.forbidden_words(), T.forbidden_bigrams()
    assert {"snip", "branch", "ribbon", "unhealthy", "spare", "budding", "card", "tag"} <= words
    for row in rows:
        said = row["instruction"]
        assert said not in heldout and not set(T._seq(said)) & words and not T._bigrams(said) & bigrams, said
    banned = T.heldout_programs()
    assert len(banned) == len(V.HELDOUT_COMPOSITIONS) + 5
    for written in {row["program"] for row in rows}:
        node = P.parse(written, "v3")
        assert not any(P.equivalent_v3(node, other) for other in banned), written


def test_check_excluded_catches_a_held_out_instruction_word_or_program(rows):
    said, node = V.HELDOUT_COMPOSITIONS[0][1][0], V.HELDOUT_COMPOSITIONS[0][0]
    for bad in (
        dict(instruction=said, program="Cut(All())", family="x"),
        dict(instruction="snip every side shoot", program="Cut(All())", family="x"),
        dict(instruction="cut the highest bare shoot", program=P.text(node), family="x"),
    ):
        with pytest.raises(AssertionError, match="held-out"):
            T.check_excluded([*rows[:5], bad])


def test_every_training_task_is_there_and_both_polarities(rows):
    programs = {row["program"] for row in rows}
    for node, _ in V.COMPOSITIONS:  # the first set of compositions, each with its own program
        assert P.text(node) in programs
    for written in ("Cut(All())", "CutFrom(top,All())", "CutFrom(bottom,All())", "Cut(Filter(tie,any,All()))",
                    "Cut(Filter(sick,yes,All()))", "Cut(Filter(sick,no,All()))", "Cut(Filter(bud,no,All()))",
                    "Cut(Filter(bud,yes,All()))", "Cut(ReadLabel())", "Then(Cut(Middle()),Cut(Rank(1,top)))"):
        assert written in programs, written
    families = {row["family"] for row in rows}
    assert {"sick_yes", "sick_no", "bud_yes", "bud_no", "order_top_down", "order_pair", "compose", "v2_ordinal"} <= families
    keep = [row for row in rows if row["instruction"].startswith("keep the healthy")]
    assert keep and all(row["program"] == "Cut(Filter(sick,yes,All()))" for row in keep)
    assert T.paraphrases() == rows  # deterministic


def test_adapters_start_as_the_identity_train_alone_and_merge_into_the_weights(tmp_path):
    torch = pytest.importorskip("torch")
    from torch import nn

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, self.down_proj, self.other = nn.Linear(6, 6), nn.Linear(6, 4), nn.Linear(4, 4)

        def forward(self, x):
            return self.other(self.down_proj(self.q_proj(x)))

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block(), Block()])
            self.visual = nn.Module()
            self.visual.q_proj = nn.Linear(3, 3)

        def forward(self, x):
            return sum(layer(x) for layer in self.layers)

    torch.manual_seed(0)
    model, x = Toy(), torch.randn(5, 6)
    before = model(x).detach()
    names = T.add_lora(model, r=2, alpha=4.0, dropout=0.0)
    assert sorted(names) == ["layers.0.down_proj", "layers.0.q_proj", "layers.1.down_proj", "layers.1.q_proj"]
    assert torch.allclose(model(x), before)  # B starts at zero
    trained = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trained and all(n.endswith((".A", ".B")) for n in trained)
    for module in model.modules():
        if hasattr(module, "B") and hasattr(module, "base"):
            module.B.data.normal_()
    tuned = model(x).detach()
    assert not torch.allclose(tuned, before)
    T.save_lora(model, str(tmp_path), dict(r=2, alpha=4.0))
    fresh = Toy()
    fresh.load_state_dict({k.replace(".base", ""): v for k, v in model.state_dict().items() if not k.endswith((".A", ".B"))})
    assert torch.allclose(fresh(x), before)
    T.apply_lora(fresh, str(tmp_path))
    assert torch.allclose(fresh(x), tuned, atol=1e-5)
