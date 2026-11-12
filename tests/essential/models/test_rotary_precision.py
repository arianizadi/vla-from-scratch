"""Rotary positions must stay distinct when the model runs under bfloat16 autocast."""

import pytest
import torch

from vla_foundry.positional_embedding import RotaryWithCast

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def table(device, seq_len=1024, **kwargs):
    pe = RotaryWithCast(64, 2048, **kwargs).to(device)
    q = torch.randn(1, seq_len, 4, 64, device=device)
    with torch.autocast(device, dtype=torch.bfloat16):
        qb = q.to(torch.bfloat16)
        out_q, out_k, _ = pe(qb, qb, qb)
    assert out_q.dtype == torch.bfloat16 and out_k.dtype == torch.bfloat16
    return pe._cos_cached.float()[0, :seq_len, 0], pe._sin_cached.float()[0, :seq_len, 0]


@pytest.mark.parametrize("device", DEVICES)
def test_every_position_gets_its_own_rotation(device):
    cos, sin = table(device)
    rows = torch.cat([cos, sin], dim=-1)
    assert torch.unique(rows, dim=0).shape[0] == rows.shape[0]


@pytest.mark.parametrize("device", DEVICES)
def test_table_matches_float32_reference(device):
    cos, sin = table(device)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, 64, 2).float() / 64))
    angles = torch.outer(torch.arange(1024, dtype=torch.float32), inv_freq)
    reference = torch.cat([angles, angles], dim=-1)
    # bfloat16 keeps about 3 significant digits of each cos / sin value; the angle itself must be exact.
    assert torch.allclose(cos.cpu(), reference.cos(), atol=1e-2)
    assert torch.allclose(sin.cpu(), reference.sin(), atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the collapse needs GPU autocast")
def test_legacy_table_reproduces_the_old_collapse():
    cos, sin = table("cuda", seq_len=2048, legacy_autocast_table=True)
    assert torch.unique(torch.cat([cos, sin], dim=-1), dim=0).shape[0] < 2048
