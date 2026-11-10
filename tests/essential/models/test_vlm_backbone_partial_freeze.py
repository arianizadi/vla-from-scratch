"""Tests for VLMHFBackboneWrapper.num_trainable_vlm_layers (train only the last N language-model layers)."""

from types import SimpleNamespace

import pytest
from torch import nn

from vla_foundry.models.vision_language_backbones.vlm_hf_backbone import VLMHFBackboneWrapper


class _FakeLanguageModel(nn.Module):
    def __init__(self, num_layers=6, dim=8):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, dim)
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)


class _FakeVLM(nn.Module):
    """Mimics the HF layout: <wrapper>.model.model.{visual, language_model}."""

    def __init__(self, language_model_attr="language_model"):
        super().__init__()
        inner = nn.Module()
        inner.visual = nn.Linear(8, 8)
        setattr(inner, language_model_attr, _FakeLanguageModel())
        self.model = nn.Module()
        self.model.model = inner
        self.model.lm_head = nn.Linear(8, 16)


def _trainable_names(vlm):
    return {name for name, param in vlm.named_parameters() if param.requires_grad}


@pytest.mark.parametrize("language_model_attr", ["language_model", "text_model"])
def test_only_last_layers_and_final_norm_are_trainable(language_model_attr):
    vlm = _FakeVLM(language_model_attr)
    VLMHFBackboneWrapper._train_only_last_layers(SimpleNamespace(_model=vlm), 2)

    prefix = f"model.model.{language_model_attr}"
    expected = {f"{prefix}.layers.{i}.{p}" for i in (4, 5) for p in ("weight", "bias")}
    expected |= {f"{prefix}.norm.weight", f"{prefix}.norm.bias"}
    assert _trainable_names(vlm) == expected


def test_zero_trainable_layers_freezes_everything():
    vlm = _FakeVLM()
    VLMHFBackboneWrapper._train_only_last_layers(SimpleNamespace(_model=vlm), 0)
    assert _trainable_names(vlm) == set()


def test_out_of_range_layer_count_raises():
    with pytest.raises(ValueError, match="num_trainable_vlm_layers"):
        VLMHFBackboneWrapper._train_only_last_layers(SimpleNamespace(_model=_FakeVLM()), 7)


def test_missing_language_model_raises():
    vlm = nn.Module()
    vlm.model = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="no language-model layer stack"):
        VLMHFBackboneWrapper._train_only_last_layers(SimpleNamespace(_model=vlm), 1)
