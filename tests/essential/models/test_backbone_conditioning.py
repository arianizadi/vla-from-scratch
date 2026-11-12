"""What a VLM backbone hands to the action head, and its independence from batch padding.

Covers the helpers shared by the two VLM backbones (base.py) and their wiring in the wrapper around
pretrained Hugging Face models, with a stand-in for the model itself.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vla_foundry.models.vision_language_backbones.base import (
    conditioning_from_hidden_states,
    insert_after_last_real_token,
)
from vla_foundry.models.vision_language_backbones.vlm_hf_backbone import VLMHFBackboneWrapper

ACTION, PAD = 99, 0


def test_token_goes_after_the_last_real_token_whatever_the_padding_side():
    ids = torch.tensor([[5, 6, 7, PAD, PAD], [PAD, PAD, PAD, 8, 9], [1, 2, 3, 4, 5]])
    mask = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 0, 1, 1], [1, 1, 1, 1, 1]])
    new_ids, new_mask = insert_after_last_real_token(ids, mask, ACTION)
    assert new_ids.tolist() == [[5, 6, 7, ACTION, PAD, PAD], [8, 9, ACTION, PAD, PAD, PAD], [1, 2, 3, 4, 5, ACTION]]
    assert new_mask.tolist() == [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]]


def test_without_a_mask_every_token_is_real():
    new_ids, new_mask = insert_after_last_real_token(torch.tensor([[5, 6]]), None, ACTION)
    assert new_ids.tolist() == [[5, 6, ACTION]] and new_mask.tolist() == [[1, 1, 1]]


def states(mask, layers=2, width=3):
    """Hidden states in which token j of row i holds the value 10 * i + j in every layer and channel."""
    batch, tokens = mask.shape
    base = (10 * torch.arange(batch)[:, None] + torch.arange(tokens)[None, :]).float()
    return [base[:, :, None].repeat(1, 1, width) + layer for layer in range(layers)]


def test_action_token_reads_the_last_real_token():
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    out = conditioning_from_hidden_states(states(mask), mask, "action_token")
    assert out.shape == (2, 1, 6)  # one vector per row, two layers of width 3 side by side
    assert out[0, 0].tolist() == [2, 2, 2, 3, 3, 3]  # row 0: token 2, layers 0 and 1
    assert out[1, 0].tolist() == [14, 14, 14, 15, 15, 15]  # row 1: token 4


def test_all_tokens_are_right_aligned_with_padding_zeroed():
    mask = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 1, 1, 1]])  # right-padded row, left-padded row
    out = conditioning_from_hidden_states(states(mask, layers=1, width=1), mask, "all_tokens")
    assert out[:, :, 0].tolist() == [[0, 0, 0, 1, 2], [0, 0, 12, 13, 14]]


def test_conditioning_len_pads_in_front_cuts_in_front_and_refuses_to_cut_real_tokens():
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    hidden = states(mask, layers=1, width=1)
    longer = conditioning_from_hidden_states(hidden, mask, "all_tokens", conditioning_len=7)
    assert longer[:, :, 0].tolist() == [[0, 0, 0, 0, 0, 1, 2]]
    shorter = conditioning_from_hidden_states(hidden, mask, "all_tokens", conditioning_len=3)
    assert shorter[:, :, 0].tolist() == [[0, 1, 2]]
    with pytest.raises(ValueError, match="3 tokens but conditioning_len is 2"):
        conditioning_from_hidden_states(hidden, mask, "all_tokens", conditioning_len=2)


def test_a_prompt_is_read_the_same_alone_and_padded_in_a_batch():
    alone = torch.tensor([[1, 1, 1]])
    padded = torch.tensor([[1, 1, 1, 0, 0, 0]])
    for mode, length in (("action_token", 0), ("all_tokens", 8)):
        a = conditioning_from_hidden_states(states(alone), alone, mode, length)
        b = conditioning_from_hidden_states(states(padded), padded, mode, length)
        assert a.shape == b.shape and torch.equal(a, b), mode


def test_unknown_mode_is_an_error():
    with pytest.raises(ValueError, match="conditioning must be"):
        conditioning_from_hidden_states(states(torch.ones(1, 2)), torch.ones(1, 2), "pooled")


class _FakeVLM(nn.Module):
    """Stands in for the wrapped model: each token's hidden state is its id, in every layer."""

    lm_hidden_dim = 2

    def forward(self, input_ids, pixel_values, attention_mask, attention_mask_images, output_hidden_states=False):
        assert output_hidden_states
        self.seen = (input_ids.clone(), attention_mask.clone())
        hidden = input_ids.float()[:, :, None].repeat(1, 1, self.lm_hidden_dim)
        return SimpleNamespace(hidden_states=[hidden, hidden + 0.5, hidden + 1.0])


def _wrapper(conditioning, conditioning_len=0):
    wrapper = object.__new__(VLMHFBackboneWrapper)
    nn.Module.__init__(wrapper)
    wrapper._model = _FakeVLM()
    wrapper._vlm_model_type = "fake"
    wrapper._action_token_id = ACTION
    wrapper.num_vlm_layers_to_use = 2
    wrapper.conditioning = conditioning
    wrapper.conditioning_len = conditioning_len
    return wrapper


def test_pretrained_backbone_reads_the_action_token_behind_the_last_real_token():
    wrapper = _wrapper("action_token")
    ids = torch.tensor([[5, 6, PAD, PAD], [1, 2, 3, 4]])
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    out = wrapper.get_action_conditioning(ids, None, attention_mask=mask).embeddings
    seen_ids, seen_mask = wrapper._model.seen
    assert seen_ids.tolist() == [[5, 6, ACTION, PAD, PAD], [1, 2, 3, 4, ACTION]]
    assert seen_mask.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    assert out.shape == (2, 1, 4)  # last two layers of width 2
    assert out[:, 0, 0].tolist() == [ACTION + 0.5, ACTION + 0.5]  # the action token's state in both rows
    assert wrapper.get_conditioning_embeddings_dim() == 4


def test_pretrained_backbone_can_hand_over_every_token():
    wrapper = _wrapper("all_tokens", conditioning_len=6)
    ids = torch.tensor([[5, 6, PAD, PAD], [1, 2, 3, 4]])
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    out = wrapper.get_action_conditioning(ids, None, attention_mask=mask).embeddings
    assert wrapper._model.seen[0].tolist() == ids.tolist()  # no action token is added
    assert out.shape == (2, 6, 4)
    assert out[:, :, 0].tolist() == [[0, 0, 0, 0, 5.5, 6.5], [0, 0, 1.5, 2.5, 3.5, 4.5]]
