"""Conditioning modes of the VLM Foundry backbone and their independence from batch padding."""

from types import SimpleNamespace

import pytest
import torch
import yaml

from vla_foundry.models.vision_language_backbones.vlm_foundry_backbone import VLMFoundryBackboneWrapper
from vla_foundry.params.model_params import VLMFoundryBackboneParams

IMAGE_TOKEN, PAD = 7, 0
VLM_CONFIG = dict(
    type="vlm",
    image_token_id=IMAGE_TOKEN,
    transformer=dict(type="transformer", hidden_dim=32, n_heads=2, n_layers=3, vocab_size=20, max_seq_len=64),
    vit=dict(type="vit", img_size=28, patch_size=14, hidden_dim=16, inter_dim=32, n_heads=2, n_layers=1),
)
PATCHES = 4  # one 28 px image, 14 px patches


def make(tmp_path, **kwargs):
    (tmp_path / "config_model.yaml").write_text(yaml.safe_dump(VLM_CONFIG))
    torch.manual_seed(0)
    params = VLMFoundryBackboneParams(vlm_experiment_dir=str(tmp_path), num_vlm_layers_to_use=2, **kwargs)
    return VLMFoundryBackboneWrapper(params, load_pretrained=False).eval()


def prompt(text_ids, pad_to=None):
    ids = [IMAGE_TOKEN] * PATCHES + list(text_ids)
    mask = [1] * len(ids)
    if pad_to is not None:
        ids, mask = ids + [PAD] * (pad_to - len(ids)), mask + [0] * (pad_to - len(mask))
    return torch.tensor([ids]), torch.tensor([mask])


@pytest.mark.parametrize("conditioning", ["action_token", "all_tokens"])
def test_conditioning_does_not_depend_on_padding(tmp_path, conditioning):
    """A prompt served alone must be probed exactly as when it was padded inside a training batch."""
    backbone = make(tmp_path, conditioning=conditioning, conditioning_len=10 if conditioning == "all_tokens" else 0)
    pixels = torch.rand(1, 3, 28, 28)
    ids, mask = prompt([3, 4], pad_to=9)
    padded = backbone.get_action_conditioning(ids, pixels, attention_mask=mask).embeddings
    alone = backbone.get_action_conditioning(prompt([3, 4])[0], pixels, attention_mask=prompt([3, 4])[1]).embeddings
    assert alone.shape == padded.shape
    assert torch.allclose(alone, padded, atol=1e-5)


def test_action_token_gives_one_vector(tmp_path):
    backbone = make(tmp_path)
    ids, mask = prompt([3, 4])
    out = backbone.get_action_conditioning(ids, torch.rand(1, 3, 28, 28), attention_mask=mask).embeddings
    assert out.shape == (1, 1, 2 * 32)


def test_all_tokens_gives_every_token_at_fixed_length(tmp_path):
    backbone = make(tmp_path, conditioning="all_tokens", conditioning_len=10)
    assert backbone._model.get_input_embeddings().weight.shape[0] == 20  # no action token added
    ids, mask = prompt([3, 4])
    out = backbone.get_action_conditioning(ids, torch.rand(1, 3, 28, 28), attention_mask=mask).embeddings
    assert out.shape == (1, 10, 2 * 32)
    assert torch.count_nonzero(out[0, :4]) == 0  # 10 slots - 6 real tokens = 4 leading zeros
    assert all(torch.count_nonzero(out[0, i]) > 0 for i in range(4, 10))


def test_all_tokens_rejects_prompts_longer_than_conditioning_len(tmp_path):
    backbone = make(tmp_path, conditioning="all_tokens", conditioning_len=5)
    ids, mask = prompt([3, 4])
    with pytest.raises(ValueError):
        backbone.get_action_conditioning(ids, torch.rand(1, 3, 28, 28), attention_mask=mask)


def test_images_are_resized_to_the_size_of_a_from_scratch_vit(tmp_path):
    backbone = make(tmp_path)
    ids, mask = prompt([3, 4])
    _, pixels, *_ = backbone._prepare_inputs(ids, torch.rand(1, 2, 3, 56, 56), attention_mask=mask)
    assert pixels.shape == (2, 3, 28, 28)


def test_a_vision_tower_without_img_size_gets_the_images_as_they_are(tmp_path):
    """A pretrained tower (vit_hf) has no img_size; reading it used to raise AttributeError."""
    backbone = make(tmp_path)
    backbone._model.model_params = SimpleNamespace(vit=SimpleNamespace())
    ids, mask = prompt([3, 4])
    _, pixels, *_ = backbone._prepare_inputs(ids, torch.rand(1, 2, 3, 56, 56), attention_mask=mask)
    assert pixels.shape == (2, 3, 56, 56)
