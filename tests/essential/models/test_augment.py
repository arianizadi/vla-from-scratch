"""Image augmentation of the ViT backbone while training (augment.py)."""

import torch

from vla_foundry.models.vision_language_backbones.augment import augment_images
from vla_foundry.models.vision_language_backbones.vit_backbone import ViTBackboneWrapper
from vla_foundry.params.model_params import ViTBackboneParams


def test_shape_range_and_change():
    torch.manual_seed(0)
    x = torch.rand(3, 4, 3, 64, 64) * 2 - 1
    y = augment_images(x)
    assert y.shape == x.shape and y.dtype == x.dtype
    assert y.min() >= -1 and y.max() <= 1
    assert not torch.allclose(x, y)


def test_one_change_per_sample_for_all_of_its_images():
    """Identical images of one sample stay identical (same shift, light and overlay); other samples differ."""
    torch.manual_seed(1)
    one = torch.rand(1, 1, 3, 64, 64) * 2 - 1
    y = augment_images(one.expand(2, 4, 3, 64, 64).contiguous())
    assert torch.allclose(y[0, 0], y[0, 3]) and torch.allclose(y[1, 1], y[1, 2])
    assert not torch.allclose(y[0, 0], y[1, 0])


def test_shift_moves_the_content():
    torch.manual_seed(2)
    x = torch.full((8, 1, 3, 224, 224), -1.0)
    x[..., 100:124, 100:124] = 1.0  # a bright square
    y = augment_images(x)
    # the square stays above 0.35 in 0..1 whatever the light, the background below (an overlay adds at most 0.25)
    centres = [(y[i, 0].mean(0) > -0.3).nonzero().float().mean(0) for i in range(8)]
    offsets = torch.stack(centres) - 111.5
    assert offsets.abs().max() <= 8.5 and offsets.abs().max() > 0


def test_backbone_augments_only_while_training():
    torch.manual_seed(0)
    params = ViTBackboneParams(
        img_size=32, patch_size=16, hidden_dim=32, inter_dim=64, n_heads=2, n_layers=1, augment=True
    )
    backbone = ViTBackboneWrapper(params)
    ids, pixels = torch.zeros(2, 1, dtype=torch.long), torch.rand(4, 3, 32, 32) * 2 - 1
    backbone.eval()
    a = backbone.get_action_conditioning(ids, pixels).embeddings
    assert torch.allclose(a, backbone.get_action_conditioning(ids, pixels).embeddings)
    backbone.train()
    b = backbone.get_action_conditioning(ids, pixels).embeddings
    assert not torch.allclose(a, b)


def test_half_strength_without_overlay_keeps_to_its_ranges():
    """strength 0.5, no overlay: a flat grey image stays within brightness x 0.9-1.1 and white balance x 0.95-1.05."""
    torch.manual_seed(3)
    x = torch.zeros(64, 1, 3, 32, 32)  # 0.5 in 0..1
    y = augment_images(x, strength=0.5, overlay=False) * 0.5 + 0.5
    assert y.min() >= 0.5 * 0.9 * 0.95 - 1e-6 and y.max() <= 0.5 * 1.1 * 1.05 + 1e-6
    assert torch.allclose(y, y[..., :1, :1].expand_as(y))  # flat stays flat: no overlay field
