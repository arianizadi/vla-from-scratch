"""Things the training step no longer pays for must leave what is computed untouched.

Three places stopped doing work nobody used, or stopped asking the GPU for a value in the middle of a step:
the action head no longer projects its tokens onto the vocabulary, the masked MSE loss no longer branches in
Python on the mask's sum, and the weight average keeps its step counter on the CPU. Each test pins the result
to the bits of the old computation.
"""

import copy

import pytest
import torch
import torch.nn.functional as F

from vla_foundry.losses import masked_mse_loss
from vla_foundry.models import create_model
from vla_foundry.models.ema import EMAModel
from vla_foundry.models.transformer import Transformer
from vla_foundry.params.model_params import DiffusionPolicyParams, TransformerParams, ViTBackboneParams

BATCH, STEPS, ACTION_DIM = 3, 5, 4


def tiny_policy():
    torch.manual_seed(0)
    params = DiffusionPolicyParams(
        vision_language_backbone=ViTBackboneParams(
            img_size=32,
            patch_size=16,
            hidden_dim=32,
            inter_dim=64,
            n_heads=2,
            n_layers=1,
            text_tokens=4,
            text_vocab_size=50,
        ),
        transformer=TransformerParams(hidden_dim=32, n_heads=2, n_layers=2, vocab_size=64, is_causal=False),
        use_flow_matching_scheduler=True,
        action_dim=ACTION_DIM,
        proprioception_dim=ACTION_DIM,
    )
    return create_model(params)


def policy_inputs():
    generator = torch.Generator().manual_seed(1)
    return dict(
        input_ids=torch.randint(1, 50, (BATCH, 6), generator=generator),
        pixel_values=torch.rand(BATCH * 2, 3, 32, 32, generator=generator),
        attention_mask=torch.ones(BATCH, 6, dtype=torch.long),
        attention_mask_images=None,
        actions=torch.randn(BATCH, STEPS, ACTION_DIM, generator=generator),
        noise=torch.randn(BATCH, STEPS, ACTION_DIM, generator=generator),
        past_mask=torch.zeros(BATCH, STEPS, dtype=torch.bool),
        future_mask=torch.ones(BATCH, STEPS, dtype=torch.bool),
        proprioception=torch.randn(BATCH, 1, ACTION_DIM, generator=generator),
    )


def loss_and_gradients(policy, head_kwargs):
    """One training step's loss and gradients with the action head called the given way."""
    policy._head_kwargs = head_kwargs
    policy.zero_grad()
    inputs = policy_inputs()
    torch.manual_seed(2)  # the diffusion time steps are drawn inside forward
    predicted = policy(**inputs)
    loss = masked_mse_loss(predicted, inputs["noise"] - inputs["actions"], mask=inputs["future_mask"])
    loss.backward()
    return loss.detach().clone(), {name: p.grad.clone() for name, p in policy.named_parameters() if p.grad is not None}


class TestTransformerWithoutLogits:
    def test_hidden_states_are_the_same_and_logits_are_not_computed(self):
        torch.manual_seed(0)
        transformer = Transformer(TransformerParams(hidden_dim=32, n_heads=2, n_layers=2, vocab_size=64))
        embeds = torch.randn(2, 7, 32)
        with_logits = transformer(inputs_embeds=embeds, output_hidden_states=True)
        without = transformer(inputs_embeds=embeds, output_hidden_states=True, compute_logits=False)
        assert with_logits.logits.shape == (2, 7, 64)
        assert without.logits is None
        assert len(without.hidden_states) == 2
        for a, b in zip(with_logits.hidden_states, without.hidden_states, strict=True):
            assert torch.equal(a, b)

    def test_logits_are_computed_by_default(self):
        transformer = Transformer(TransformerParams(hidden_dim=32, n_heads=2, n_layers=1, vocab_size=64))
        assert transformer(input_ids=torch.randint(0, 64, (2, 5))).logits.shape == (2, 5, 64)


class TestActionHeadSkipsTheVocabulary:
    def test_our_transformer_is_told_to_skip_the_logits(self):
        assert tiny_policy()._head_kwargs == {"compute_logits": False}

    def test_loss_and_gradients_match_the_old_call_bit_for_bit(self):
        policy = tiny_policy()
        old_loss, old_grads = loss_and_gradients(policy, {})  # the old call: logits computed, then dropped
        new_loss, new_grads = loss_and_gradients(policy, {"compute_logits": False})
        assert torch.equal(old_loss, new_loss)
        assert old_grads.keys() == new_grads.keys()
        for name in old_grads:
            assert torch.equal(old_grads[name], new_grads[name]), name
        # The vocabulary projection (and the final norm in front of it) never had a gradient: the action head reads
        # the hidden states in front of them. So the optimizer never moved them, before or after.
        assert not any(name.startswith(("transformer.output.", "transformer.norm.")) for name in old_grads)
        assert any(name.startswith("transformer.layers.") for name in old_grads)

    def test_generated_actions_match_the_old_call_bit_for_bit(self):
        policy = tiny_policy().eval()
        inputs = policy_inputs()
        kwargs = dict(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            actions=inputs["actions"],
            attention_mask=inputs["attention_mask"],
            num_inference_steps=4,
            proprioception=inputs["proprioception"],
        )
        policy._head_kwargs = {}
        torch.manual_seed(3)
        old = policy.generate_actions(**kwargs)
        policy._head_kwargs = {"compute_logits": False}
        torch.manual_seed(3)
        new = policy.generate_actions(**kwargs)
        assert torch.equal(old, new)


class TestMaskedMSEWithoutReadback:
    @pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float32])
    def test_same_bits_as_the_weighted_mean(self, mask_dtype):
        generator = torch.Generator().manual_seed(0)
        predicted = torch.randn(4, 6, 3, generator=generator)
        target = torch.randn(4, 6, 3, generator=generator)
        mask = (torch.rand(4, 6, generator=generator) > 0.3).to(mask_dtype)
        expected = F.mse_loss(predicted, target, weight=mask.unsqueeze(-1).expand_as(predicted))
        assert torch.equal(masked_mse_loss(predicted, target, mask=mask), expected)

    def test_same_bits_without_a_mask(self):
        generator = torch.Generator().manual_seed(0)
        predicted = torch.randn(4, 6, 3, generator=generator)
        target = torch.randn(4, 6, 3, generator=generator)
        expected = F.mse_loss(predicted, target, weight=torch.ones_like(predicted))
        assert torch.equal(masked_mse_loss(predicted, target), expected)

    def test_same_bits_under_bf16_autocast(self):
        generator = torch.Generator().manual_seed(0)
        predicted = torch.randn(4, 6, 3, generator=generator).to(torch.bfloat16)
        target = torch.randn(4, 6, 3, generator=generator)
        mask = torch.rand(4, 6, generator=generator) > 0.3
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            expected = F.mse_loss(predicted, target, weight=mask.unsqueeze(-1).expand_as(predicted))
            got = masked_mse_loss(predicted, target, mask=mask)
        assert got.dtype == expected.dtype
        assert torch.equal(got, expected)

    def test_empty_mask_is_a_zero_that_can_be_backpropagated(self):
        predicted = torch.randn(2, 5, 3, requires_grad=True)
        loss = masked_mse_loss(predicted, torch.randn(2, 5, 3), mask=torch.zeros(2, 5, dtype=torch.bool))
        assert loss.item() == 0.0
        loss.backward()
        assert torch.count_nonzero(predicted.grad) == 0


class TestEmaStepCounterStaysPut:
    def make(self):
        torch.manual_seed(0)
        model = torch.nn.Linear(4, 3)
        return model, EMAModel(model)

    def test_moving_the_average_leaves_the_counter(self):
        _, ema = self.make()
        counter = ema.optimization_step
        weight = ema.model.weight
        ema._apply(lambda t: t.clone())  # what .to(device) does: every tensor is replaced by a moved copy
        assert ema.optimization_step is counter
        assert ema.model.weight.data is not weight.data
        assert "optimization_step" in ema.state_dict()

    def test_cast_keeps_the_counter_and_the_updates(self):
        model, ema = self.make()
        reference = copy.deepcopy(ema)
        ema = ema.to("cpu", dtype=torch.float64)
        assert ema.optimization_step.dtype == torch.long
        for _ in range(3):
            with torch.no_grad():
                model.weight.add_(0.1)
            ema.step(model)
            reference.step(model)
        assert ema.optimization_step.item() == reference.optimization_step.item() == 3
        assert ema.get_decay(3) == reference.get_decay(3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    def test_counter_is_on_the_cpu_next_to_a_gpu_average(self):
        model, ema = self.make()
        model = model.to("cuda", dtype=torch.bfloat16)
        ema = ema.to("cuda", dtype=torch.float32)
        assert ema.optimization_step.device.type == "cpu"
        assert ema.model.weight.device.type == "cuda"
        ema.step(model)
        ema.step(model)
        assert ema.optimization_step.item() == 2
