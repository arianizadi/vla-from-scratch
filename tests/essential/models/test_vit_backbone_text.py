"""Language conditioning of the ViT backbone (text_tokens > 0)."""

import pytest
import torch

from vla_foundry.models.vision_language_backbones.vit_backbone import ViTBackboneWrapper
from vla_foundry.params.model_params import ViTBackboneParams

VIT = dict(img_size=32, patch_size=16, hidden_dim=32, inter_dim=64, n_heads=2, n_layers=1)


def make(text_tokens):
    torch.manual_seed(0)
    return ViTBackboneWrapper(ViTBackboneParams(**VIT, text_tokens=text_tokens, text_vocab_size=50)).eval()


def test_image_only_backbone_is_unchanged():
    backbone = make(0)
    assert not any(name.startswith("text_") for name in backbone.state_dict())
    out = backbone.get_action_conditioning(torch.zeros(2, 1, dtype=torch.long), torch.rand(4, 3, 32, 32))
    assert out.embeddings.shape == (2, 2 * 4, 32)  # 2 images x 4 patches


def test_text_tokens_are_prepended_at_fixed_length():
    backbone = make(6)
    ids = torch.tensor([[5, 6, 7], [8, 9, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    out = backbone.get_action_conditioning(ids, torch.rand(4, 3, 32, 32), attention_mask=mask)
    assert out.embeddings.shape == (2, 6 + 2 * 4, 32)
    assert torch.count_nonzero(out.embeddings[1, 2:6]) == 0  # padding slots carry nothing
    assert torch.count_nonzero(out.embeddings[1, :2]) > 0


@pytest.mark.parametrize("padded", [[[8, 9, 0, 0]], [[0, 0, 8, 9]], [[0, 8, 9, 0, 0, 0, 0, 0]]])
def test_conditioning_does_not_depend_on_padding(padded):
    """An instruction served alone must look the same as when it was padded inside a training batch."""
    backbone = make(6)
    pixels = torch.rand(2, 3, 32, 32)
    alone = backbone.get_action_conditioning(
        torch.tensor([[8, 9]]), pixels, attention_mask=torch.tensor([[1, 1]])
    ).embeddings
    ids = torch.tensor(padded)
    batched = backbone.get_action_conditioning(ids, pixels, attention_mask=(ids != 0).long()).embeddings
    assert torch.allclose(alone, batched)


def test_instruction_changes_the_conditioning():
    backbone = make(6)
    pixels = torch.rand(2, 3, 32, 32)
    a = backbone.get_action_conditioning(torch.tensor([[5, 6]]), pixels, attention_mask=torch.ones(1, 2)).embeddings
    b = backbone.get_action_conditioning(torch.tensor([[5, 7]]), pixels, attention_mask=torch.ones(1, 2)).embeddings
    assert not torch.allclose(a[:, :6], b[:, :6])
    assert torch.allclose(a[:, 6:], b[:, 6:])
