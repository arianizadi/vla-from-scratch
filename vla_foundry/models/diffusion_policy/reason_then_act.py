"""Reason, then act: a diffusion policy whose vision-language backbone also writes a short note.

The prompt is the images, the instruction, the note the policy wrote last ("note: ...") and the note it writes now
("now: ...", or "now: same" when the last one still stands); see vla_foundry/data/processor/notes.py. In training the
current note is the expert's (teacher forcing) and a language-model loss on its tokens trains the backbone to write it.
When served, the backbone writes it (`generate_note`) and the action head then reads the whole sequence, note included,
through the every-token conditioning: the hidden states of every token of the last layers.

With action_loss_weight 0 the action head is not trained and this is a reasoner only, the slow level of a two-level
policy whose fast level is an ordinary policy that reads the note's sub-goal as its instruction.
"""

import torch
import torch.nn.functional as F

from vla_foundry.models.diffusion_policy.diffusion_policy import DiffusionPolicy


class ReasonThenActOutput:
    """`direction`: the action head's prediction (None for a reasoner only); `lm_loss`: mean cross-entropy over the
    note's tokens; `lm_tokens`: how many tokens it was taken over."""

    def __init__(self, direction, lm_loss, lm_tokens):
        self.direction, self.lm_loss, self.lm_tokens = direction, lm_loss, lm_tokens


def note_lm_loss(logits: torch.Tensor, input_ids: torch.Tensor, note_labels: torch.Tensor | None):
    """Cross-entropy of each labelled token (note_labels True) predicted from the position before it, averaged over
    the labelled tokens of the batch. Returns (loss, number of tokens); a zero that keeps the graph when none."""
    if note_labels is None or not bool(note_labels[:, 1:].any()):
        return logits.sum() * 0.0, 0
    labels = note_labels.clone()
    labels[:, 0] = False
    rows, cols = labels.nonzero(as_tuple=True)
    predicted = logits[rows, cols - 1].float()
    return F.cross_entropy(predicted, input_ids[rows, cols]), int(rows.numel())


class ReasonThenActPolicy(DiffusionPolicy):
    def __init__(self, model_params, vision_language_backbone, transformer, noise_scheduler):
        super().__init__(model_params, vision_language_backbone, transformer, noise_scheduler)
        conditioning = getattr(vision_language_backbone, "conditioning", None)
        if conditioning != "all_tokens":
            raise ValueError(
                "reason_then_act needs a backbone that predicts tokens with conditioning: all_tokens "
                f"(vlm_backbone or vlm_foundry_backbone), got {type(vision_language_backbone).__name__} with "
                f"conditioning {conditioning!r}"
            )
        if self.num_action_head_repeats not in (None, 1):
            raise ValueError("reason_then_act does not support num_action_head_repeats")
        self.lm_loss_weight = model_params.lm_loss_weight
        self.action_loss_weight = model_params.action_loss_weight
        self.max_note_tokens = model_params.max_note_tokens

    def _run_backbone(self, input_ids, pixel_values, attention_mask, attention_mask_images, **kwargs):
        """The backbone's forward pass: (its raw outputs, with logits and hidden states; the action conditioning)."""
        backbone = self.vision_language_backbone
        ids, pixels, mask, mask_images, model_kwargs = backbone._prepare_inputs(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            attention_mask_images=attention_mask_images,
            **kwargs,
        )
        outputs = backbone._model(
            input_ids=ids,
            pixel_values=pixels,
            attention_mask=mask,
            attention_mask_images=mask_images,
            **model_kwargs,
        )
        return outputs, backbone._extract_action_conditioning(outputs, mask).embeddings

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask,
        attention_mask_images,
        actions,
        noise,
        past_mask,
        future_mask,
        proprioception=None,
        note_labels=None,
        **kwargs,
    ):
        outputs, backbone_embeddings = self._run_backbone(
            input_ids, pixel_values, attention_mask, attention_mask_images, **kwargs
        )
        lm_loss, lm_tokens = note_lm_loss(outputs.logits, input_ids, note_labels)
        if self.action_loss_weight == 0:
            return ReasonThenActOutput(None, lm_loss, lm_tokens)

        # The action head, as in DiffusionPolicy.forward.
        timesteps = torch.randint(0, self.scheduler.num_timesteps, (actions.shape[0],)).to(actions.device)
        noisy_action = self.scheduler.add_noise(actions, noise, timesteps, mask=future_mask)
        noisy_action = torch.where(future_mask.unsqueeze(-1), noisy_action, actions)
        if self.input_noise_std > 0:
            noisy_action = noisy_action + torch.randn_like(noisy_action) * self.input_noise_std
        noisy_action = self.action_encode(noisy_action)
        time_embeddings = self.time_encoding(timesteps).unsqueeze(1)
        proprio_embeddings = None
        if self.proprioception_encode is not None and proprioception is not None:
            proprio_embeddings = self.proprioception_encode(proprioception)
            if self.input_noise_std > 0:
                proprio_embeddings = proprio_embeddings + torch.randn_like(proprio_embeddings) * self.input_noise_std
        transformer_input = self._build_transformer_input(
            backbone_embeddings=backbone_embeddings,
            time_embeddings=time_embeddings,
            noisy_action=noisy_action,
            proprio_embeddings=proprio_embeddings,
        )
        transformer_output = self.transformer(
            inputs_embeds=transformer_input, output_hidden_states=True, use_cache=False, **self._head_kwargs
        )
        direction = self.output_layer(transformer_output.hidden_states[-1][:, -noise.shape[1] :, :])
        return ReasonThenActOutput(direction, lm_loss, lm_tokens)

    @torch.no_grad()
    def generate_note(
        self,
        input_ids,
        pixel_values,
        attention_mask=None,
        attention_mask_images=None,
        stop_id=None,
        max_new_tokens=None,
        use_cache=True,
        **kwargs,
    ) -> list[int]:
        """Greedy decoding for one prompt (batch of one) that ends where the note begins ("now:"): the tokens the
        backbone writes until `stop_id` (not included) or `max_new_tokens`. With `use_cache` the backbone's own
        generation with a key-value cache is used where it has one (a Hugging Face model, our VLM); otherwise, or
        without it, every step runs the whole sequence again."""
        if input_ids.shape[0] != 1:
            raise ValueError("generate_note takes one prompt at a time")
        max_new_tokens = max_new_tokens or self.max_note_tokens
        mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        if use_cache:
            written = self._generate_cached(
                input_ids, pixel_values, mask, attention_mask_images, stop_id, max_new_tokens, **kwargs
            )
            if written is not None:
                return written
        ids, written = input_ids, []
        for _ in range(max_new_tokens):
            outputs, _ = self._run_backbone(ids, pixel_values, mask, attention_mask_images, **kwargs)
            token = outputs.logits[:, -1].argmax(dim=-1)
            if stop_id is not None and int(token) == stop_id:
                break
            written.append(int(token))
            ids = torch.cat([ids, token[:, None].to(ids.dtype)], dim=1)
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
        return written

    def _generate_cached(self, input_ids, pixel_values, mask, mask_images, stop_id, max_new_tokens, **kwargs):
        """The backbone model's own greedy generation, or None if it has none."""
        backbone = self.vision_language_backbone
        ids, pixels, mask, mask_images, model_kwargs = backbone._prepare_inputs(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=mask,
            attention_mask_images=mask_images,
            **kwargs,
        )
        model_kwargs.pop("output_hidden_states", None)
        inner = backbone._model
        hf = getattr(inner, "model", None)
        if hf is not None and hasattr(hf, "generate") and hasattr(hf, "config"):  # a Hugging Face model (VLMHF)
            out = hf.generate(
                input_ids=ids,
                attention_mask=mask,
                pixel_values=pixels,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=stop_id,
                pad_token_id=stop_id,
                **model_kwargs,
            )
        elif hasattr(inner, "generate") and hasattr(inner, "transformer"):  # our VLM
            out = inner.generate(
                ids, pixels, mask, max_new_tokens=max_new_tokens, temperature=0, eos_token_id=stop_id, use_cache=True
            )
        else:
            return None
        written = []
        for token in out[0, ids.shape[1] :].tolist():
            if stop_id is not None and token == stop_id:
                break
            written.append(int(token))
        return written
