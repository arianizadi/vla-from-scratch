"""The ViT backbone with a frozen pretrained vision tower in place of its ViT (frozen_tower: a Qwen-VL kind model).

Nothing is downloaded: the model is a tiny Qwen3-VL with random weights saved into the test's directory with an image
processor of its own (as in tests/essential/sim/test_eyes_tower.py). The tokens are checked against the tower's last
block output normalised by the merger's own norm, computed from the image processor's own pixel_values.
"""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from vla_foundry.models.vision_language_backbones.frozen_tower import FrozenTower
from vla_foundry.models.vision_language_backbones.vit_backbone import ViTBackboneWrapper
from vla_foundry.params.model_params import ViTBackboneParams

transformers = pytest.importorskip("transformers")

MEAN = [0.48, 0.46, 0.41]  # differ by channel, so that channels out of order would show
STD = [0.27, 0.26, 0.28]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("tiny_qwen3_vl")
    torch.manual_seed(0)
    vision = dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=24)
    vision.update(patch_size=16, temporal_patch_size=2, spatial_merge_size=2, num_position_embeddings=64)
    vision.update(deepstack_visual_indexes=[0])
    rope = {"rope_type": "default", "mrope_section": [2, 2, 2], "mrope_interleaved": True}
    text = dict(vocab_size=64, hidden_size=24, intermediate_size=48, num_hidden_layers=1, num_attention_heads=2)
    text.update(num_key_value_heads=1, head_dim=12, rope_scaling=rope)
    tokens = dict(image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
    config = transformers.Qwen3VLConfig(vision_config=vision, text_config=text, **tokens)
    transformers.Qwen3VLForConditionalGeneration(config).save_pretrained(path)
    cut = dict(patch_size=16, temporal_patch_size=2, merge_size=2)
    transformers.Qwen2VLImageProcessor(image_mean=MEAN, image_std=STD, **cut).save_pretrained(path)
    return path


def make(tiny, text_tokens=4):
    torch.manual_seed(0)
    params = ViTBackboneParams(
        img_size=64, patch_size=16, hidden_dim=16, inter_dim=32, n_heads=2, n_layers=1,
        text_tokens=text_tokens, text_vocab_size=50, frozen_tower=str(tiny),
    )  # fmt: skip
    return ViTBackboneWrapper(params).eval()


def _pictures(n, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (n, 64, 64, 3), dtype=np.uint8)


def _processed(pictures):
    """As the simple_vlm processor writes them: (x - 0.5) / 0.5 for x in 0..1, [N, 3, H, W]."""
    return (torch.from_numpy(pictures).permute(0, 3, 1, 2).float() / 255 - 0.5) / 0.5


def test_tokens_are_the_towers_normalised_patches(tiny):
    """The tokens equal the merger norm's output on the model's own image processor's pixel_values."""
    tower = make(tiny)._model
    pictures = _pictures(2)
    processor = transformers.AutoImageProcessor.from_pretrained(tiny)
    inputs = processor(images=[Image.fromarray(p) for p in pictures], return_tensors="pt", do_resize=False)
    pixels, grid = tower.pixel_values(_processed(pictures) * 0.5 + 0.5)
    assert torch.equal(grid, inputs["image_grid_thw"])
    assert torch.allclose(pixels, inputs["pixel_values"], atol=1e-5)
    ours = tower.tokens(_processed(pictures))  # with the attention batched over the images
    for block in tower.tower.blocks:
        assert "forward" in vars(block.attn)
        del block.attn.forward  # the model's own attention, one image at a time
    kept = []
    hook = tower.tower.merger.norm.register_forward_hook(lambda _, args, out: kept.append(out))
    with torch.no_grad():
        tower.tower(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
    hook.remove()
    assert ours.shape == (2, 16, 32)  # 4 x 4 patches of 16 pixels, the tower's width
    assert torch.allclose(ours, kept[0].reshape(2, 16, 32), atol=1e-4)


def test_conditioning_shape_and_text(tiny):
    backbone = make(tiny)
    ids, mask = torch.tensor([[5, 6, 7], [8, 9, 0]]), torch.tensor([[1, 1, 1], [1, 1, 0]])
    pixels = _processed(_pictures(4))  # 2 samples x 2 images, flattened as the processors give them
    out = backbone.get_action_conditioning(ids, pixels, attention_mask=mask).embeddings
    assert out.shape == (2, 4 + 2 * 16, 16)
    assert backbone.get_conditioning_embeddings_dim() == 16
    # the second sample's images give the second sample's tokens
    alone = backbone.get_action_conditioning(ids[1:], pixels[2:], attention_mask=mask[1:]).embeddings
    assert torch.allclose(out[1:], alone, atol=1e-5)


def test_tower_is_frozen_and_left_out_of_checkpoints(tiny):
    backbone = make(tiny)
    backbone.train()
    assert not backbone._model.tower.training  # read as it was trained, whatever the policy's mode
    trained = {name for name, p in backbone.named_parameters() if p.requires_grad}
    assert trained and not any(".tower." in name for name in trained)
    state = backbone.state_dict()
    assert not any(".tower." in key for key in state)
    assert any(key.startswith("_model.proj.") for key in state)
    # a copy (the running average) keeps the hook; loading a checkpoint without the tower keeps the tower's weights
    assert not any(".tower." in key for key in copy.deepcopy(backbone).state_dict())
    other = make(tiny)
    with torch.no_grad():
        for p in other._model.proj.parameters():
            p.add_(1.0)
    other.load_state_dict(state, strict=True)
    pixels = _processed(_pictures(2))
    ids = torch.tensor([[5, 6]])
    a = backbone.eval().get_action_conditioning(ids, pixels).embeddings
    b = other.eval().get_action_conditioning(ids, pixels).embeddings
    assert torch.allclose(a, b)


def test_gradients_reach_the_projection_not_the_tower(tiny):
    backbone = make(tiny).train()
    out = backbone.get_action_conditioning(torch.tensor([[5, 6]]), _processed(_pictures(2))).embeddings
    out.square().mean().backward()
    assert backbone._model.proj.weight.grad is not None
    assert all(p.grad is None for p in backbone._model.tower.parameters())


def test_batched_attention_survives_a_copy(tiny):
    """The running average is a deep copy: its attention is batched too, and bound to its own modules."""
    backbone = make(tiny)
    twin = copy.deepcopy(backbone)
    for mine, theirs in zip(backbone._model.tower.blocks, twin._model.tower.blocks, strict=True):
        assert vars(theirs.attn)["forward"].__self__ is theirs.attn is not mine.attn


def test_not_a_tower(tiny):
    with pytest.raises(ValueError, match="Qwen-VL kind"):
        FrozenTower("x", 16, 64, tower=torch.nn.Linear(2, 2), mean=MEAN, std=STD)


@pytest.fixture(scope="module")
def tiny_dino(tmp_path_factory) -> Path:
    """A tiny random DINOv2 with registers (patches of 14, 4 registers), with an image processor of its own."""
    path = tmp_path_factory.mktemp("tiny_dinov2_reg")
    torch.manual_seed(0)
    config = transformers.Dinov2WithRegistersConfig(
        hidden_size=24, num_hidden_layers=2, num_attention_heads=2, intermediate_size=48,
        patch_size=14, image_size=56, num_register_tokens=4,
    )  # fmt: skip
    transformers.Dinov2WithRegistersModel(config).save_pretrained(path)
    transformers.BitImageProcessor(image_mean=MEAN, image_std=STD).save_pretrained(path)
    return path


def make_dino(tiny_dino, pool):
    torch.manual_seed(0)
    params = ViTBackboneParams(
        img_size=56, patch_size=16, hidden_dim=16, inter_dim=32, n_heads=2, n_layers=1, text_tokens=4,
        text_vocab_size=50, frozen_tower=str(tiny_dino), frozen_tower_pool=pool,
    )  # fmt: skip
    return ViTBackboneWrapper(params).eval()


@pytest.mark.parametrize("pool", [1, 2])
def test_dino_tokens_are_its_pooled_patches_without_class_and_registers(tiny_dino, pool):
    backbone = make_dino(tiny_dino, pool)
    pictures = np.random.default_rng(1).integers(0, 256, (2, 56, 56, 3), dtype=np.uint8)
    x01 = torch.from_numpy(pictures).permute(0, 3, 1, 2).float() / 255
    with torch.no_grad():
        hidden = backbone._model.tower(
            pixel_values=(x01 - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
        ).last_hidden_state
    assert hidden.shape[1] == 1 + 4 + 16  # class token, registers, 4 x 4 patches
    grid = hidden[:, 5:].reshape(2, 4, 4, 24)
    if pool == 2:
        grid = grid.reshape(2, 2, 2, 2, 2, 24).mean(dim=(2, 4))
    ours = backbone._model.tokens((x01 - 0.5) / 0.5)
    assert ours.shape == (2, 16 // pool**2, 24)
    assert torch.allclose(ours, grid.reshape(2, -1, 24), atol=1e-5)
    out = backbone.get_action_conditioning(torch.tensor([[5, 6]]), (x01 - 0.5) / 0.5).embeddings
    assert out.shape == (1, 4 + 2 * 16 // pool**2, 16)


def test_dino_is_frozen_and_left_out_of_checkpoints(tiny_dino):
    backbone = make_dino(tiny_dino, 2).train()
    assert not backbone._model.tower.training
    assert not any(".tower." in key for key in backbone.state_dict())
    assert not any(p.requires_grad for p in backbone._model.tower.parameters())
    other = make_dino(tiny_dino, 2)
    other.load_state_dict(backbone.state_dict(), strict=True)


def test_the_qwen_tower_is_not_pooled(tiny):
    with pytest.raises(ValueError, match="not pooled"):
        ViTBackboneWrapper(ViTBackboneParams(img_size=64, hidden_dim=16, frozen_tower=str(tiny), frozen_tower_pool=2))


@pytest.fixture(scope="module")
def tiny_dino3(tmp_path_factory) -> Path:
    """A tiny random DINOv3 (patches of 16, 4 registers, rotary positions), with an image processor of its own."""
    path = tmp_path_factory.mktemp("tiny_dinov3")
    torch.manual_seed(0)
    config = transformers.DINOv3ViTConfig(
        hidden_size=24, num_hidden_layers=2, num_attention_heads=2, intermediate_size=48, patch_size=16,
        num_register_tokens=4,
    )  # fmt: skip
    transformers.DINOv3ViTModel(config).save_pretrained(path)
    transformers.BitImageProcessor(image_mean=MEAN, image_std=STD).save_pretrained(path)
    return path


def test_dino3_tokens_are_its_pooled_patches_without_class_and_registers(tiny_dino3):
    torch.manual_seed(0)
    params = ViTBackboneParams(
        img_size=64, hidden_dim=16, text_tokens=4, text_vocab_size=50, frozen_tower=str(tiny_dino3), frozen_tower_pool=2
    )
    backbone = ViTBackboneWrapper(params).eval()
    x01 = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        normed = (x01 - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
        hidden = backbone._model.tower(pixel_values=normed).last_hidden_state
    assert hidden.shape[1] == 1 + 4 + 16  # class token, registers, 4 x 4 patches
    grid = hidden[:, 5:].reshape(2, 2, 2, 2, 2, 24).mean(dim=(2, 4)).reshape(2, 4, 24)
    assert torch.allclose(backbone._model.tokens((x01 - 0.5) / 0.5), grid, atol=1e-5)
    assert not any(".tower." in key for key in backbone.state_dict())
