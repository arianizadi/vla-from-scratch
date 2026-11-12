"""VLM backbone wrapper for action policy conditioning."""

import logging

import torch
from torch import nn

from vla_foundry.models.model_outputs.backbone_output import VisionLanguageBackboneOutput
from vla_foundry.models.vision_language_backbones.base import (
    BaseBackboneWrapper,
    conditioning_from_hidden_states,
    insert_after_last_real_token,
)
from vla_foundry.params.model_params import VLMBackboneParams


class VLMHFBackboneWrapper(BaseBackboneWrapper):
    """VLM backbone wrapper for action policy conditioning.

    This class wraps VLMHF and provides the conditioning embedding interface, while keeping VLMHF free
    from conditioning-specific logic. Two conditioning modes (`backbone_params.conditioning`):

    - ``action_token``: an action token is inserted after the prompt and its hidden states from the
      last layers are the conditioning, i.e. the action head receives a single vector.
    - ``all_tokens``: the last layers' hidden states of every image and text token are the
      conditioning, so the action head can attend to individual image tokens.
    """

    def __init__(self, backbone_params: VLMBackboneParams, load_pretrained: bool = True):
        """Initialize VLM backbone for conditioning.

        Args:
            backbone_params: Configuration parameters for the VLM backbone.
            load_pretrained: If True, download pretrained weights.
        """
        super().__init__(backbone_params, load_pretrained)
        self.num_vlm_layers_to_use = backbone_params.num_vlm_layers_to_use
        self.conditioning = backbone_params.conditioning
        self.conditioning_len = backbone_params.conditioning_len
        if self.conditioning not in ("action_token", "all_tokens"):
            raise ValueError(f"conditioning must be 'action_token' or 'all_tokens', got {self.conditioning!r}")

        # Detect model type
        self._vlm_model_type = getattr(self._model.model.config, "model_type", "").lower()

        if self.conditioning == "action_token":
            # Add action token to vocabulary
            self._action_token_id = self._model.resize_token_embeddings()

            # Initialize the new token's embedding as mean of existing embeddings
            with torch.no_grad():
                embeddings = self._model.model.get_input_embeddings()
                mean_emb = embeddings.weight[:-1].mean(dim=0)
                embeddings.weight[-1] = mean_emb

        if backbone_params.num_trainable_vlm_layers is not None:
            self._train_only_last_layers(backbone_params.num_trainable_vlm_layers)

    def _train_only_last_layers(self, num_layers: int) -> None:
        """Freeze the VLM except the last `num_layers` language-model layers and the final norm."""
        language_model, layers = None, None
        for name, module in self._model.model.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            if leaf in ("language_model", "text_model") and isinstance(getattr(module, "layers", None), nn.ModuleList):
                language_model, layers = module, module.layers
                break
        if layers is None:
            raise ValueError(
                f"num_trainable_vlm_layers is set but no language-model layer stack was found in "
                f"{type(self._model.model).__name__}."
            )
        if not 0 <= num_layers <= len(layers):
            raise ValueError(f"num_trainable_vlm_layers must be in [0, {len(layers)}], got {num_layers}.")

        for param in self._model.parameters():
            param.requires_grad = False
        trainable = list(layers[len(layers) - num_layers :]) if num_layers > 0 else []
        final_norm = getattr(language_model, "norm", None)
        if num_layers > 0 and isinstance(final_norm, nn.Module):
            trainable.append(final_norm)
        for module in trainable:
            for param in module.parameters():
                param.requires_grad = True
        n_train = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self._model.parameters())
        logging.info(
            f"VLM backbone: training last {num_layers}/{len(layers)} language-model layers "
            f"({n_train / 1e6:.0f}M of {n_total / 1e6:.0f}M parameters)."
        )

    def get_conditioning_embeddings_dim(self) -> int:
        """Return dimension for condition embeddings (with layers concatenated)."""
        return self._model.lm_hidden_dim * self.num_vlm_layers_to_use

    @property
    def _is_qwen_style(self) -> bool:
        return "qwen" in self._vlm_model_type

    @property
    def _is_paligemma_style(self) -> bool:
        return "paligemma" in self._vlm_model_type

    def _validate_inputs(
        self,
        input_ids: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        attention_mask_images: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        """Validate VLM backbone inputs.

        Ensures required arguments are present and consistent (e.g., Qwen image_grid_thw
        matches pixel_values patches). Raises ValueError if inputs are invalid.
        """
        if self._is_qwen_style and "image_grid_thw" not in kwargs:
            raise ValueError(
                f"Qwen model requires image_grid_thw but it was not provided. "
                f"pixel_values shape: {pixel_values.shape}. "
                f"Ensure the processor passes image_grid_thw through the data pipeline."
            )

    def _prepare_inputs_for_action_token(self, input_ids, attention_mask):
        """Insert the action token directly after each sample's last real token."""
        return insert_after_last_real_token(input_ids, attention_mask, self._action_token_id)

    def _prepare_inputs(
        self,
        input_ids: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        attention_mask_images: torch.Tensor | None = None,
        **kwargs,
    ):
        self._validate_inputs(input_ids, pixel_values, attention_mask, attention_mask_images, **kwargs)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if self.conditioning == "action_token":
            input_ids, attention_mask = self._prepare_inputs_for_action_token(input_ids, attention_mask)

        # The conditioning is read from the hidden states
        kwargs["output_hidden_states"] = True

        return input_ids, pixel_values, attention_mask, attention_mask_images, kwargs

    def get_action_conditioning(
        self,
        input_ids: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        attention_mask_images: torch.Tensor | None = None,
        **kwargs,
    ) -> VisionLanguageBackboneOutput:
        input_ids, pixel_values, attention_mask, attention_mask_images, model_kwargs = self._prepare_inputs(
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
        # The mask (with the action token already inserted) says which positions are real tokens.
        return self._extract_action_conditioning(outputs, attention_mask)

    def _extract_action_conditioning(self, outputs, attention_mask=None) -> VisionLanguageBackboneOutput:
        embeddings = conditioning_from_hidden_states(
            outputs.hidden_states[-self.num_vlm_layers_to_use :],
            attention_mask,
            self.conditioning,
            self.conditioning_len,
        )
        return VisionLanguageBackboneOutput(embeddings=embeddings)
