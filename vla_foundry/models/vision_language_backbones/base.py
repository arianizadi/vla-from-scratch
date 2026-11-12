"""Base class for backbone wrappers used in action policies."""

from abc import abstractmethod
from typing import Any

import torch
from transformers.utils import ModelOutput

from vla_foundry.models.base_model import BaseModel
from vla_foundry.models.model_outputs.backbone_output import VisionLanguageBackboneOutput
from vla_foundry.models.registry import create_model
from vla_foundry.params.model_params import BackboneParams


class BaseBackboneWrapper(BaseModel):
    """Base wrapper that adapts a vision-language model for action policy conditioning.

    Subclasses create the underlying model internally and implement
    extraction of conditioning embeddings from model outputs.
    """

    def __init__(self, backbone_params: BackboneParams, load_pretrained: bool = True):
        super().__init__(backbone_params)
        self._model = create_model(backbone_params, load_pretrained)

    @abstractmethod
    def get_conditioning_embeddings_dim(self) -> int:
        """Get the output conditioning embeddings dimension."""
        raise NotImplementedError

    def get_action_conditioning(
        self,
        input_ids: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        attention_mask_images: torch.Tensor | None = None,
        **kwargs,
    ) -> VisionLanguageBackboneOutput:
        """Get embeddings for conditioning action policies.

        Args:
            input_ids: Text token IDs [B, seq_len]
            pixel_values: Image tensors (format depends on model)
            attention_mask: Text attention mask
            attention_mask_images: Image attention mask
            **kwargs: Model-specific args

        Returns:
            VisionLanguageBackboneOutput with embeddings
        """
        (input_ids, pixel_values, attention_mask, attention_mask_images, model_kwargs) = self._prepare_inputs(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            attention_mask_images=attention_mask_images,
            **kwargs,
        )

        outputs = self._model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            attention_mask_images=attention_mask_images,
            **model_kwargs,
        )

        return self._extract_action_conditioning(outputs=outputs)

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        attention_mask_images: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        dict[str, Any],
    ]:
        """Prepare inputs for conditioning, if necessary."""
        return input_ids, pixel_values, attention_mask, attention_mask_images, kwargs

    @abstractmethod
    def _extract_action_conditioning(self, outputs: ModelOutput) -> VisionLanguageBackboneOutput:
        """Extract conditioning embeddings from model outputs."""
        raise NotImplementedError


def insert_after_last_real_token(input_ids, attention_mask, token_id):
    """Put one extra token directly after each row's last real token; padding moves to the end.

    A batch is padded to its longest prompt. Appending the token after the padding would put it at a
    position, and behind padding tokens, that depend on what else is in the batch: a short instruction
    would be read out differently in a training batch than when served alone. Returns the new
    (input_ids, attention_mask), one column longer.
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    extra = input_ids.new_full((input_ids.shape[0], 1), token_id)
    input_ids = torch.cat([input_ids, extra], dim=1)
    attention_mask = torch.cat([attention_mask, torch.ones_like(extra).to(attention_mask.dtype)], dim=1)
    order = torch.argsort((attention_mask == 0).to(torch.int8), dim=1, stable=True)  # real tokens first
    return torch.gather(input_ids, 1, order), torch.gather(attention_mask, 1, order)


def conditioning_from_hidden_states(hidden_states, attention_mask, conditioning, conditioning_len=0):
    """Turn the last layers' hidden states of a prompt into the conditioning of an action head.

    hidden_states: the layers to use, each [batch, tokens, width]; they are concatenated along width.
    conditioning "action_token": the states of each row's last real token, [batch, 1, width * layers].
    conditioning "all_tokens": the states of every real token, right-aligned (so that their distance to
        what follows them in the action head does not depend on padding), padding zeroed and moved to
        the front, [batch, tokens or conditioning_len, width * layers]. With conditioning_len > 0 the
        sequence is padded in front (or cut in front) to that many tokens, so that its length does not
        depend on what else is in the batch.
    """
    if attention_mask is None:
        attention_mask = torch.ones(hidden_states[0].shape[:2], dtype=torch.long, device=hidden_states[0].device)
    rows = torch.arange(attention_mask.shape[0], device=attention_mask.device)
    real = attention_mask != 0
    if conditioning == "action_token":
        index = real.sum(dim=1) - 1
        return torch.cat([hs[rows, index] for hs in hidden_states], dim=-1).unsqueeze(1)
    if conditioning != "all_tokens":
        raise ValueError(f"conditioning must be 'action_token' or 'all_tokens', got {conditioning!r}")
    order = torch.argsort(real.to(torch.int8), dim=1, stable=True)
    real = torch.gather(real, 1, order)
    embeddings = torch.cat([hs[rows[:, None], order] for hs in hidden_states], dim=-1)
    embeddings = embeddings * real.unsqueeze(-1).to(embeddings.dtype)
    if conditioning_len > 0:
        longest = int(real.sum(dim=1).max())
        if longest > conditioning_len:
            raise ValueError(f"A prompt has {longest} tokens but conditioning_len is {conditioning_len}.")
        length = embeddings.shape[1]
        if length < conditioning_len:
            embeddings = torch.nn.functional.pad(embeddings, (0, 0, conditioning_len - length, 0))
        else:
            embeddings = embeddings[:, length - conditioning_len :]
    return embeddings
