"""ViT backbone wrapper for action policy conditioning.

Wraps a standalone ViT and provides patch embeddings as conditioning tokens for the diffusion
transformer. Optionally (``text_tokens > 0``) the instruction is added as learned token embeddings,
which makes this a language-conditioned policy that still hands every image patch to the action head.
"""

import einops
import torch

from vla_foundry.models.base_model import BaseModel
from vla_foundry.models.model_outputs.backbone_output import VisionLanguageBackboneOutput
from vla_foundry.models.vision_language_backbones.base import BaseBackboneWrapper
from vla_foundry.models.vit import ViT
from vla_foundry.params.model_params import ViTBackboneParams


class ViTBackboneWrapper(BaseBackboneWrapper):
    """Wraps a ViT and provides patch embeddings for action policy conditioning."""

    def __init__(self, backbone_params: ViTBackboneParams, load_pretrained: bool = True):
        # Skip BaseBackboneWrapper.__init__ which calls create_model with the wrong type.
        # Instead, initialize BaseModel directly and create the ViT ourselves.
        BaseModel.__init__(self, backbone_params)
        self._model = ViT(backbone_params)
        self.text_tokens = backbone_params.text_tokens
        if self.text_tokens > 0:
            if backbone_params.text_vocab_size <= 0:
                raise ValueError("vit_backbone: text_tokens > 0 requires text_vocab_size (the tokenizer's vocabulary).")
            dim = backbone_params.hidden_dim
            self.text_embedding = torch.nn.Embedding(backbone_params.text_vocab_size, dim)
            self.text_position = torch.nn.Parameter(torch.zeros(1, self.text_tokens, dim))
            # The ViT output is layer-normed; bring the text tokens to the same scale.
            self.text_norm = torch.nn.LayerNorm(dim, eps=backbone_params.ln_eps)
            torch.nn.init.normal_(self.text_embedding.weight, std=0.02)
            torch.nn.init.normal_(self.text_position, std=0.02)

    def get_conditioning_embeddings_dim(self) -> int:
        return self._model.model_params.hidden_dim

    def _embed_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """[B, L] token ids -> [B, text_tokens, D], independent of how the batch was padded.

        Tokenizers differ in padding side and a batch is padded to its longest instruction, so the
        same instruction can arrive at different positions in training (batched) and at inference
        (alone). Valid tokens are moved to the front and padding slots are zeroed, which makes the
        conditioning for a sample a function of that sample only.
        """
        mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask.to(input_ids.dtype)
        order = torch.argsort(1 - mask, dim=1, stable=True)  # valid tokens first, original order kept
        ids, mask = torch.gather(input_ids, 1, order), torch.gather(mask, 1, order)
        missing = self.text_tokens - ids.shape[1]
        if missing > 0:
            ids = torch.nn.functional.pad(ids, (0, missing))
            mask = torch.nn.functional.pad(mask, (0, missing))
        ids, mask = ids[:, : self.text_tokens], mask[:, : self.text_tokens]
        ids = ids * mask  # padding ids may be outside the embedding table
        text = self.text_norm(self.text_embedding(ids) + self.text_position)
        return text * mask.unsqueeze(-1).to(text.dtype)

    def get_action_conditioning(
        self,
        input_ids: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        attention_mask_images: torch.Tensor | None = None,
        **kwargs,
    ) -> VisionLanguageBackboneOutput:
        # pixel_values may arrive as [B, N, C, H, W] (5D) or [B*N, C, H, W] (4D)
        # depending on the upstream processor. CLIP/PaliGemma/PassthroughProcessor all
        # flatten to 4D. Reshape to 5D so ViT.forward returns [B, N, num_patches, D]
        # and the per-batch sequence stays correctly grouped.
        if pixel_values.ndim == 4 and input_ids is not None:
            batch_size = input_ids.shape[0]
            num_images = pixel_values.shape[0] // batch_size
            if num_images * batch_size != pixel_values.shape[0]:
                raise ValueError(
                    f"pixel_values batch size ({pixel_values.shape[0]}) is not divisible by "
                    f"input_ids batch size ({batch_size}); cannot infer images-per-sample."
                )
            pixel_values = pixel_values.reshape(batch_size, num_images, *pixel_values.shape[1:])
        embeddings = self._model(pixel_values)  # [B, N, num_patches, hidden_dim]
        # Flatten camera and patch dims into a single sequence
        if embeddings.ndim == 4:
            embeddings = einops.rearrange(embeddings, "b n t d -> b (n t) d")
        if self.text_tokens > 0:
            text = self._embed_text(input_ids, attention_mask).to(embeddings.dtype)
            embeddings = torch.cat([text, embeddings], dim=1)  # [B, text_tokens + N * num_patches, D]
        return VisionLanguageBackboneOutput(embeddings=embeddings)

    def _extract_action_conditioning(self, outputs) -> VisionLanguageBackboneOutput:
        # Not used — we override get_action_conditioning directly
        raise NotImplementedError
