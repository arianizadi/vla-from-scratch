"""The paired consistency loss (model.twin_consistency_weight, data.twin_suffix): a frame and its twin, the same frame
in another look, denoised from the same noise at the same flow time; the flow loss on both and the squared difference
of their predicted velocities."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vla_foundry.losses import masked_mse_loss
from vla_foundry.models import create_model
from vla_foundry.models.batch_handlers import DiffusionPolicyBatchHandler
from vla_foundry.models.diffusion_policy.diffusion_policy import twin_batch
from vla_foundry.params.model_params import DiffusionPolicyParams, TransformerParams, ViTBackboneParams

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:  # sim/ is not an installed package
    sys.path.insert(0, str(REPOSITORY))

B, STEPS, ACTION_DIM, IMAGES = 3, 5, 4, 2  # 2 own images per sample (and 2 of its twin's)


def tiny_policy(weight):
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
        ),  # fmt: skip
        transformer=TransformerParams(hidden_dim=32, n_heads=2, n_layers=2, vocab_size=64, is_causal=False),
        use_flow_matching_scheduler=True,
        action_dim=ACTION_DIM,
        proprioception_dim=ACTION_DIM,
        twin_consistency_weight=weight,
    )
    return create_model(params)


def inputs(own, twin):
    """own, twin: [B, IMAGES, 3, 32, 32]; the processor's layout, each sample's own images then its twin's, flat."""
    g = torch.Generator().manual_seed(1)
    return dict(
        input_ids=torch.randint(1, 50, (B, 6), generator=g),
        pixel_values=torch.cat([own, twin], dim=1).flatten(0, 1),
        attention_mask=torch.ones(B, 6, dtype=torch.long),
        attention_mask_images=None,
        actions=torch.randn(B, STEPS, ACTION_DIM, generator=g),
        noise=torch.randn(B, STEPS, ACTION_DIM, generator=g),
        past_mask=torch.zeros(B, STEPS, dtype=torch.bool),
        future_mask=torch.ones(B, STEPS, dtype=torch.bool),
        proprioception=torch.randn(B, 1, ACTION_DIM, generator=g),
    )


def test_twin_batch_puts_the_twins_after_the_samples():
    own, twin = torch.rand(B, IMAGES, 3, 8, 8), torch.rand(B, IMAGES, 3, 8, 8)
    ids = torch.arange(B)[:, None]
    for pixels in (torch.cat([own, twin], 1), torch.cat([own, twin], 1).flatten(0, 1)):
        new_ids, new_pixels, mask, _ = twin_batch(ids, pixels, torch.ones(B, 1), None)
        expected = torch.cat([own, twin])
        assert torch.equal(new_ids[:, 0], torch.cat([torch.arange(B)] * 2)) and mask.shape == (2 * B, 1)
        assert torch.equal(new_pixels if pixels.ndim == 5 else new_pixels.unflatten(0, (2 * B, IMAGES)), expected)
    with pytest.raises(ValueError, match="twin"):
        twin_batch(ids, torch.rand(B * 3, 3, 8, 8), None, None)


def test_identical_twins_predict_the_same_velocity_at_the_same_time_and_noise():
    """With the twin's images equal to the sample's own, the two halves of the batch are the same computation."""
    policy = tiny_policy(0.5).train()
    own = torch.rand(B, IMAGES, 3, 32, 32)
    out = policy(**inputs(own, own))
    assert out.shape == (2 * B, STEPS, ACTION_DIM)
    assert torch.allclose(out[:B], out[B:], atol=1e-5)
    different = policy(**inputs(own, torch.rand(B, IMAGES, 3, 32, 32)))
    assert not torch.allclose(different[:B], different[B:])


def test_a_served_policy_is_given_its_own_images_only():
    policy = tiny_policy(0.5).eval()
    x = inputs(torch.rand(B, IMAGES, 3, 32, 32), torch.rand(B, IMAGES, 3, 32, 32))
    x["pixel_values"] = x["pixel_values"].unflatten(0, (B, 2 * IMAGES))[:, :IMAGES].flatten(0, 1)
    assert policy(**x).shape == (B, STEPS, ACTION_DIM)


def test_loss_is_the_mean_flow_loss_plus_the_weighted_difference():
    handler, weight = DiffusionPolicyBatchHandler(), 0.3
    cfg = SimpleNamespace(model=SimpleNamespace(twin_consistency_weight=weight))
    targets, mask = torch.randn(B, STEPS, ACTION_DIM), torch.ones(B, STEPS, dtype=torch.bool)
    own, twin = torch.randn(B, STEPS, ACTION_DIM), torch.randn(B, STEPS, ACTION_DIM)
    loss = handler.compute_loss(torch.cat([own, twin]), targets, masked_mse_loss, cfg, mask=mask)
    flow = (masked_mse_loss(own, targets, mask) + masked_mse_loss(twin, targets, mask)) / 2
    assert torch.allclose(loss, flow + weight * masked_mse_loss(own, twin, mask))
    off = SimpleNamespace(model=SimpleNamespace(twin_consistency_weight=0.0))
    assert torch.allclose(
        handler.compute_loss(own, targets, masked_mse_loss, off, mask=mask), masked_mse_loss(own, targets, mask)
    )


def test_the_consistency_term_trains_the_encoder():
    policy = tiny_policy(1.0).train()
    x = inputs(torch.rand(B, IMAGES, 3, 32, 32), torch.rand(B, IMAGES, 3, 32, 32))
    out = policy(**x)
    own, twin = out.chunk(2)
    masked_mse_loss(own, twin, x["future_mask"]).backward()
    vit = policy.vision_language_backbone._model
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in vit.parameters())


def test_the_processor_appends_the_twins_images_in_training_only():
    from vla_foundry.data.processor.robotics_processor import RoboticsProcessor

    seen = []
    processor = RoboticsProcessor.__new__(RoboticsProcessor)
    processor.data_params = SimpleNamespace(
        twin_suffix="_twin", image_indices=[-20, 0], camera_names=["front", "wrist"], pad_missing_images=False,
        instruction_paraphrases=None, paraphrase_prob=0.0, normalization=SimpleNamespace(enabled=False),
    )  # fmt: skip
    processor.note_tokens, processor.normalizer, processor.processor_kwargs = None, None, {}
    processor.vlm_processor = lambda images, text, **kw: (seen.append(images), {"input_ids": torch.zeros(1, 1)})[1]
    names = ["front_t-20", "wrist_t-20", "front_t0", "wrist_t0"]
    image = {n: np.full((2, 2, 3), i) for i, n in enumerate(names)}
    image.update({n.replace("_t", "_twin_t"): np.full((2, 2, 3), 10 + i) for i, n in enumerate(names)})
    batch = dict(images=[image], language_instruction=["cut"], lowdim=[{"action": [0.0]}])
    import vla_foundry.data.processor.robotics_processor as rp

    original = rp.apply_chat_template
    rp.apply_chat_template = lambda proc, n, text: text
    try:
        processor.process_inputs(dict(batch, metadata=[{}]), names)  # training: the twin's after the sample's own
        processor.process_inputs(batch, names)  # served
    finally:
        rp.apply_chat_template = original
    assert [int(i[0, 0, 0]) for i in seen[0][0]] == [0, 1, 2, 3, 10, 11, 12, 13]
    assert [int(i[0, 0, 0]) for i in seen[1][0]] == [0, 1, 2, 3]


def test_twins_dataset_pairs_each_copy_with_its_source():
    pa = pytest.importorskip("pyarrow")  # (the dataset tools need it; skipped where it is missing)
    from sim.pickup.twins import pair_episodes, twin_table

    entries = [
        dict(episode_index=0, look="source", source_episode=0),
        dict(episode_index=1, look="source", source_episode=1),
        dict(episode_index=2, look="bench/train", source_episode=1),
        dict(episode_index=3, look="camera/train", source_episode=0),
    ]
    assert pair_episodes(entries) == [(2, 1), (3, 0)]

    def table(episode, look):
        n = 3
        return pa.table(
            {
                "front": pa.array([{"bytes": f"{look}f{i}".encode(), "path": None} for i in range(n)]),
                "wrist": pa.array([{"bytes": f"{look}w{i}".encode(), "path": None} for i in range(n)]),
                "state": pa.array([[float(i)] * 8 for i in range(n)]),
                "action": pa.array([[float(i) + 1] * 8 for i in range(n)]),
                "episode_index": pa.array(np.full(n, episode, dtype=np.int64)),
                "index": pa.array(np.arange(n, dtype=np.int64)),
            }
        )

    out = twin_table(table(2, "look"), table(1, "plain"), episode_index=0, first_index=7)
    assert out["front"][0].as_py()["bytes"] == b"lookf0" and out["front_twin"][2].as_py()["bytes"] == b"plainf2"
    assert out["wrist_twin"][1].as_py()["bytes"] == b"plainw1"
    assert out["episode_index"].to_pylist() == [0, 0, 0] and out["index"].to_pylist() == [7, 8, 9]
    moved = table(1, "plain").set_column(2, "state", pa.array([[9.0] * 8] * 3))
    with pytest.raises(ValueError, match="state"):
        twin_table(table(2, "look"), moved, 0, 0)
