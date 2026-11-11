"""VLM Foundry backbone wrapper for action policy conditioning."""

import torch

from vla_foundry.models.model_outputs.backbone_output import VisionLanguageBackboneOutput
from vla_foundry.models.vision_language_backbones.base import BaseBackboneWrapper
from vla_foundry.params.model_params import VLMFoundryBackboneParams


class VLMFoundryBackboneWrapper(BaseBackboneWrapper):
    """Backbone wrapper for VLMs trained with VLA Foundry.

    Loads a VLM from an experiment directory (config_model.yaml + checkpoint) and turns its hidden
    states into conditioning for the action head, in one of two ways (``conditioning``):

    - ``action_token``: an action token is appended to the prompt and its hidden states from the
      last layers are the conditioning, i.e. the action head receives a single vector.
    - ``all_tokens``: the last layers' hidden states of every image and text token are the
      conditioning, so the action head can attend to individual image patches.
    """

    def __init__(self, backbone_params: VLMFoundryBackboneParams, load_pretrained: bool = True):
        super().__init__(backbone_params, load_pretrained)
        self.num_vlm_layers_to_use = backbone_params.num_vlm_layers_to_use
        self.conditioning = backbone_params.conditioning
        self.conditioning_len = backbone_params.conditioning_len
        if self.conditioning not in ("action_token", "all_tokens"):
            raise ValueError(f"conditioning must be 'action_token' or 'all_tokens', got {self.conditioning!r}")
        if self.conditioning == "all_tokens":
            return

        # Add action token to vocabulary by extending the embedding table by 1.
        # Note: processor/tokenizer isn't saved with the checkpoint, so on every load the base
        # VLM is re-instantiated at its original vocab size and the action-token row must be
        # re-created here; weights for that row are then restored by load_model_checkpoint.
        base_vocab_size = self._model.resize_token_embeddings()
        self._action_token_id = base_vocab_size  # new token gets the next available index
        new_vocab_size = self._model.resize_token_embeddings(base_vocab_size + 1)

        embeddings = self._model.get_input_embeddings()
        if new_vocab_size != base_vocab_size + 1 or embeddings.weight.shape[0] != base_vocab_size + 1:
            raise RuntimeError(
                f"Action-token extension failed: expected vocab size {base_vocab_size + 1}, got "
                f"resize return {new_vocab_size} and embedding rows {embeddings.weight.shape[0]}. "
                "Check resize_token_embeddings semantics in the underlying transformer."
            )

        # Initialize new token embedding as mean of existing embeddings
        with torch.no_grad():
            mean_emb = embeddings.weight[:-1].mean(dim=0)
            embeddings.weight[-1] = mean_emb

    def get_conditioning_embeddings_dim(self) -> int:
        return self._model.lm_hidden_dim * self.num_vlm_layers_to_use

    def _prepare_inputs_for_action_token(self, input_ids, attention_mask):
        """Insert the action token directly after each sample's last real token.

        A batch is padded to its longest prompt, so appending the token after the padding would put
        it at a position (and behind padding tokens) that depends on what else is in the batch: a
        short instruction would be probed differently in a training batch than when served alone.
        """
        action_tokens = input_ids.new_full((input_ids.shape[0], 1), self._action_token_id)
        input_ids = torch.cat([input_ids, action_tokens], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(action_tokens).to(attention_mask.dtype)], dim=1)
        order = torch.argsort((attention_mask == 0).to(torch.int8), dim=1, stable=True)  # real tokens first
        return torch.gather(input_ids, 1, order), torch.gather(attention_mask, 1, order)

    def _prepare_inputs(
        self,
        input_ids,
        pixel_values,
        attention_mask=None,
        attention_mask_images=None,
        **kwargs,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if self.conditioning == "action_token":
            input_ids, attention_mask = self._prepare_inputs_for_action_token(input_ids, attention_mask)
        kwargs["output_hidden_states"] = True
        if pixel_values is not None:
            # The data processor (SmolVLM2) upscales images to 512×512 tiles, but the custom
            # ViT was trained at a different resolution. Resize to the ViT's native img_size.
            vit_img_size = self._model.model_params.vit.img_size
            if pixel_values.shape[-1] != vit_img_size or pixel_values.shape[-2] != vit_img_size:
                orig_shape = pixel_values.shape
                pv = pixel_values.reshape(-1, *orig_shape[-3:]).float()
                pv = torch.nn.functional.interpolate(
                    pv, size=(vit_img_size, vit_img_size), mode="bilinear", align_corners=False
                ).to(pixel_values.dtype)
                pixel_values = pv.reshape(*orig_shape[:-2], vit_img_size, vit_img_size)
            if pixel_values.dim() == 5:
                batch, num_images, c, h, w = pixel_values.shape
                pixel_values = pixel_values.reshape(batch * num_images, c, h, w)
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
        return self._extract_action_conditioning(outputs, attention_mask)

    def _extract_action_conditioning(self, outputs, attention_mask=None) -> VisionLanguageBackboneOutput:
        hidden_states = outputs.hidden_states[-self.num_vlm_layers_to_use :]
        if attention_mask is None:
            attention_mask = torch.ones(hidden_states[0].shape[:2], dtype=torch.long, device=hidden_states[0].device)
        rows = torch.arange(attention_mask.shape[0], device=attention_mask.device)
        real = attention_mask != 0
        if self.conditioning == "action_token":
            # The action token is each sample's last real token (see _prepare_inputs_for_action_token).
            index = real.sum(dim=1) - 1
            embeddings = torch.cat([hs[rows, index] for hs in hidden_states], dim=-1).unsqueeze(1)
            return VisionLanguageBackboneOutput(embeddings=embeddings)

        # all_tokens: real tokens right-aligned (so their distance to the action tokens that follow in
        # the action head does not depend on padding), padding slots zeroed and moved to the front.
        order = torch.argsort(real.to(torch.int8), dim=1, stable=True)
        real = torch.gather(real, 1, order)
        embeddings = torch.cat([hs[rows[:, None], order] for hs in hidden_states], dim=-1)
        embeddings = embeddings * real.unsqueeze(-1).to(embeddings.dtype)
        if self.conditioning_len > 0:
            length = embeddings.shape[1]
            if int(real.sum(dim=1).max()) > self.conditioning_len:
                raise ValueError(
                    f"A prompt has {int(real.sum(dim=1).max())} tokens but conditioning_len is {self.conditioning_len}."
                )
            if length < self.conditioning_len:
                embeddings = torch.nn.functional.pad(embeddings, (0, 0, self.conditioning_len - length, 0))
            else:
                embeddings = embeddings[:, length - self.conditioning_len :]
        return VisionLanguageBackboneOutput(embeddings=embeddings)
