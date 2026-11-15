"""Reason, then act (model type reason_then_act): a diffusion policy whose backbone also writes a note.

A fake vision-language backbone stands in for Qwen or the custom VLM: its logits put all their weight on a token chosen
by the test, and its hidden states are the token ids, so what the action head is conditioned on can be read off.
"""

from types import SimpleNamespace

import torch
from torch import nn

from vla_foundry.models.batch_handlers import ReasonThenActBatchHandler
from vla_foundry.models.diffusion import create_noise_scheduler
from vla_foundry.models.diffusion_policy.reason_then_act import ReasonThenActPolicy, note_lm_loss
from vla_foundry.models.registry import create_model
from vla_foundry.models.vision_language_backbones.base import conditioning_from_hidden_states
from vla_foundry.params.model_params import ReasonThenActParams
from vla_foundry.params.train_experiment_params import load_params_from_yaml

VOCAB, STOP = 20, 19


class _FakeVLM(nn.Module):
    """Logits: the next token is always `next_token(sequence)`; hidden states: each token's id."""

    def __init__(self, next_token):
        super().__init__()
        self.next_token = next_token
        self.scale = nn.Parameter(torch.ones(()))  # something to train

    def forward(self, input_ids, pixel_values=None, attention_mask=None, attention_mask_images=None, **kwargs):
        batch, length = input_ids.shape
        logits = torch.full((batch, length, VOCAB), -10.0)
        for b in range(batch):
            for t in range(length):
                logits[b, t, self.next_token(input_ids[b, : t + 1].tolist())] = 10.0
        hidden = input_ids.float()[:, :, None].repeat(1, 1, 4) * self.scale
        return SimpleNamespace(logits=logits * self.scale, hidden_states=[hidden, hidden + 1])


class _FakeBackbone(nn.Module):
    conditioning = "all_tokens"

    def __init__(self, next_token):
        super().__init__()
        self._model = _FakeVLM(next_token)

    def get_conditioning_embeddings_dim(self):
        return 8

    def _prepare_inputs(self, input_ids, pixel_values, attention_mask=None, attention_mask_images=None, **kwargs):
        return input_ids, pixel_values, attention_mask, attention_mask_images, kwargs

    def _extract_action_conditioning(self, outputs, attention_mask=None):
        embeddings = conditioning_from_hidden_states(outputs.hidden_states, attention_mask, "all_tokens", 12)
        return SimpleNamespace(embeddings=embeddings)


def _policy(next_token=lambda seq: STOP, action_loss_weight=1.0):
    params = load_params_from_yaml(
        ReasonThenActParams, "tests/essential/params/dummy_configs/dummy_diffusion_policy_config.yaml"
    )
    object.__setattr__(params, "action_loss_weight", action_loss_weight)
    return ReasonThenActPolicy(
        params, _FakeBackbone(next_token), create_model(params.transformer), create_noise_scheduler(params)
    )


def test_the_note_loss_is_taken_on_the_labelled_tokens_from_the_position_before_each():
    ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 0, 0]])
    logits = torch.full((2, 5, 6), -20.0)
    logits[0, 1, 3] = logits[0, 2, 4] = 20.0  # row 0 predicts its tokens 3 and 4 perfectly
    logits[1, 1, 0] = 20.0  # row 1 predicts its token 3 wrongly
    labels = torch.zeros(2, 5, dtype=torch.bool)
    labels[0, 2:4] = True
    loss, n = note_lm_loss(logits, ids, labels)
    assert n == 2 and loss.item() < 1e-3
    labels[1, 2] = True
    loss, n = note_lm_loss(logits, ids, labels)
    assert n == 3 and loss.item() > 10  # the wrong prediction dominates the mean
    zero, n = note_lm_loss(logits.requires_grad_(), ids, torch.zeros_like(labels))
    assert n == 0 and zero.item() == 0 and zero.requires_grad


def test_forward_returns_the_action_prediction_and_the_note_loss():
    policy = _policy()
    batch, steps = 2, 16
    ids = torch.tensor([[3, 4, 5, 6, STOP], [3, 4, 5, STOP, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
    labels = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 0, 1, 0]], dtype=torch.bool)
    out = policy(
        input_ids=ids,
        pixel_values=None,
        attention_mask=mask,
        attention_mask_images=None,
        actions=torch.randn(batch, steps, 7),
        noise=torch.randn(batch, steps, 7),
        past_mask=torch.zeros(batch, steps, dtype=torch.bool),
        future_mask=torch.ones(batch, steps, dtype=torch.bool),
        note_labels=labels,
        metadata=[{}, {}],  # extra batch keys are passed through, as in training
    )
    assert out.direction.shape == (batch, steps, 7)
    assert out.lm_tokens == 3
    # the fake backbone always predicts STOP: right for the two STOP labels, wrong for token 6
    assert 5 < out.lm_loss.item() < 20
    handler = ReasonThenActBatchHandler()
    cfg = SimpleNamespace(model=SimpleNamespace(lm_loss_weight=0.5, action_loss_weight=2.0))
    targets = torch.zeros(batch, steps, 7)
    total = handler.compute_loss(out, targets, lambda input, target, mask: (input - target).pow(2).mean(), cfg)
    expected = 0.5 * out.lm_loss + 2.0 * out.direction.pow(2).mean()
    assert torch.allclose(total, expected)
    total.backward()
    assert policy.vision_language_backbone._model.scale.grad is not None


def test_a_reasoner_only_skips_the_action_head():
    policy = _policy(action_loss_weight=0.0)
    out = policy(
        input_ids=torch.tensor([[3, STOP]]),
        pixel_values=None,
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        attention_mask_images=None,
        actions=torch.randn(1, 16, 7),
        noise=torch.randn(1, 16, 7),
        past_mask=torch.zeros(1, 16, dtype=torch.bool),
        future_mask=torch.ones(1, 16, dtype=torch.bool),
        note_labels=torch.tensor([[False, True]]),
    )
    assert out.direction is None and out.lm_tokens == 1
    cfg = SimpleNamespace(model=SimpleNamespace(lm_loss_weight=1.0, action_loss_weight=0.0))
    assert torch.allclose(ReasonThenActBatchHandler().compute_loss(out, None, None, cfg), out.lm_loss)


def test_generate_note_writes_greedily_until_the_stop_token():
    # counts up from the last token, and stops after 7
    policy = _policy(next_token=lambda seq: STOP if seq[-1] >= 7 else seq[-1] + 1)
    assert policy.generate_note(torch.tensor([[2, 4]]), None, stop_id=STOP) == [5, 6, 7]
    assert policy.generate_note(torch.tensor([[2, 4]]), None, stop_id=STOP, max_new_tokens=2) == [5, 6]


def test_a_backbone_that_hands_over_one_vector_is_refused():
    params = load_params_from_yaml(
        ReasonThenActParams, "tests/essential/params/dummy_configs/dummy_diffusion_policy_config.yaml"
    )
    backbone = _FakeBackbone(lambda seq: STOP)
    backbone.conditioning = "action_token"
    try:
        ReasonThenActPolicy(params, backbone, create_model(params.transformer), create_noise_scheduler(params))
    except ValueError as error:
        assert "all_tokens" in str(error)
    else:
        raise AssertionError("expected a ValueError")


def test_generate_note_uses_the_backbones_own_cached_generation_when_it_has_one():
    policy = _policy(next_token=lambda seq: STOP if seq[-1] >= 7 else seq[-1] + 1)
    inner = policy.vision_language_backbone._model
    calls = []

    def generate(ids, pixels, mask, max_new_tokens, temperature, eos_token_id, use_cache):
        calls.append((max_new_tokens, temperature, eos_token_id))
        return torch.cat([ids, torch.tensor([[5, 6, 7, STOP, 3]])], dim=1)  # stops at STOP; what follows is ignored

    inner.generate, inner.transformer = generate, None  # looks like our VLM
    assert policy.generate_note(torch.tensor([[2, 4]]), None, stop_id=STOP, max_new_tokens=9) == [5, 6, 7]
    assert calls == [(9, 0, STOP)]
    assert policy.generate_note(torch.tensor([[2, 4]]), None, stop_id=STOP, use_cache=False) == [5, 6, 7]
    assert len(calls) == 1  # without the cache, the loop
