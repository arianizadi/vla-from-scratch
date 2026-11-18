"""A frozen pretrained image encoder in place of the ViT backbone's own ViT: the vision tower of a vision-language model
of the Qwen-VL kind (with Qwen3-VL-2B, the tower the Qwen policies and the reasoner look through).

What it is for. The small policies, whose ViT is trained from scratch on a few thousand demonstrations, fail when what
the camera sees changes (lighting, camera pose, bench: 0-4/100 against 80-87 in the plain look). A tower trained on a
very large set of images, kept frozen, cannot learn the look of the demonstrations; the question is whether the policy
built on it loses less under such changes. Everything else of the policy stays as it is: the instruction's tokens,
the action head, the training recipe.

The tokens are the patch features before the tower's merger, one per patch of 16 pixels (196 per 224-pixel image, as
many as the ViT gives), each normalised by the merger's own LayerNorm: the last block's output itself has a few
channels in the thousands (sim/pickup/eyes.py, the tower probe, which reads the same tokens). They are projected to
the backbone's width and layer-normed, as the ViT's output is.

The images arrive as the simple_vlm processor writes them, (x - 0.5) / 0.5 for x in 0..1; they are taken back to 0..1
and prepared on the device as the model's own image processor prepares them (its mean and std; the image repeated
over the tower's temporal patch; the patches in its order, each 2 x 2 group one after the other; grid_thw (1, S / p,
S / p) per image). The image is not resized.

Frozen: the tower runs under torch.no_grad(), in its weights' own type (not under autocast), with its attention batched
over the images (_attention_by_image), its blocks compiled while training on a GPU, and stays in eval mode whatever
the policy's mode. A checkpoint holds the projection and its norm, not the tower: whenever the policy is built the
tower is loaded from its name, and loading a state dict keeps the tower's weights where the state dict does not carry
them.
"""

import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROCESSOR_MEAN = 0.5  # the simple_vlm processor's normalisation (SimpleVLMProcessor's defaults)
PROCESSOR_STD = 0.5


def vision_tower(model: nn.Module) -> nn.Module | None:
    """The vision tower of a vision-language model of the Qwen-VL kind, or None: a module `visual` with transformer
    `blocks` and a `merger` that normalises each patch (`merger.norm`) before packing neighbouring patches."""
    tower = getattr(model, "visual", None)
    if tower is None:
        tower = getattr(getattr(model, "model", None), "visual", None)
    hf, merger = getattr(tower, "config", None), getattr(tower, "merger", None)
    sizes = ("patch_size", "temporal_patch_size", "spatial_merge_size", "hidden_size")
    fits = (
        isinstance(tower, nn.Module)
        and isinstance(getattr(tower, "blocks", None), nn.ModuleList)
        and isinstance(getattr(merger, "norm", None), nn.LayerNorm)
        and not getattr(merger, "use_postshuffle_norm", False)
        and all(isinstance(getattr(hf, size, None), int) for size in sizes)
    )
    return tower if fits else None


def _attention_by_image(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs):
    """The tower's attention with every image the same size, as one batched call: the model's own code (transformers
    4.57, without flash attention) attends within each image in a Python loop, one call per image and block, which
    leaves the GPU waiting on the CPU. Same arithmetic; images of different sizes go to the model's own code."""
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    if bool((lengths != lengths[0]).any()) or position_embeddings is None:
        return type(self).forward(self, hidden_states, cu_seqlens, rotary_pos_emb, position_embeddings, **kwargs)
    s, n, length = hidden_states.shape[0], lengths.numel(), int(lengths[0])
    q, k, v = self.qkv(hidden_states).reshape(s, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    q, k = sys.modules[type(self).__module__].apply_rotary_pos_emb_vision(q, k, *position_embeddings)
    q, k, v = (t.reshape(n, length, self.num_heads, -1).transpose(1, 2) for t in (q, k, v))
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=self.scaling)
    return self.proj(out.transpose(1, 2).reshape(s, -1))


def _batch_attention(tower: nn.Module) -> None:
    """Put _attention_by_image in place of each block's attention forward where the attention is of the kind it was
    written for (Qwen3-VL's vision attention: qkv, proj, num_heads, scaling, and apply_rotary_pos_emb_vision in its
    module)."""
    for block in tower.blocks:
        attn = getattr(block, "attn", None)
        fits = all(hasattr(attn, a) for a in ("qkv", "proj", "num_heads", "scaling")) and hasattr(
            sys.modules.get(type(attn).__module__), "apply_rotary_pos_emb_vision"
        )
        if fits:
            attn.forward = types.MethodType(_attention_by_image, attn)


def _without_tower(module: nn.Module, state_dict: dict, prefix: str, local_metadata: dict) -> None:
    """state_dict post-hook: the frozen tower's weights are left out of a checkpoint. A plain function, so that a copy
    of the module (the running average of the weights) keeps it."""
    for key in module.frozen:
        state_dict.pop(prefix + key, None)


class FrozenTower(nn.Module):
    """Images [N, 3, S, S] as the simple_vlm processor writes them -> tokens [N, (S / p)^2, out_dim]."""

    def __init__(self, name: str, out_dim: int, image_size: int, tower: nn.Module | None = None, mean=None, std=None):
        """name: the vision-language model (a hub name already in the local cache, or a local directory). tower, mean
        and std: give them to skip loading (tests)."""
        super().__init__()
        if tower is None:
            from transformers import AutoImageProcessor, AutoModelForImageTextToText

            local = Path(name).is_dir()
            model = AutoModelForImageTextToText.from_pretrained(name, local_files_only=local, dtype="auto")
            tower = vision_tower(model)
            if tower is None:
                raise ValueError(
                    f"frozen_tower: {name} ({type(model).__name__}) has no vision tower of the Qwen-VL kind"
                )
            del model  # the language model is not kept
            processor = AutoImageProcessor.from_pretrained(name, local_files_only=local)
            cut = tuple(getattr(processor, k, None) for k in ("patch_size", "temporal_patch_size", "merge_size"))
            hf = tower.config
            if cut != (hf.patch_size, hf.temporal_patch_size, hf.spatial_merge_size):
                raise ValueError(f"frozen_tower: the image processor of {name} cuts patches {cut}, unlike its tower")
            if not (
                processor.do_rescale and processor.do_normalize and math.isclose(processor.rescale_factor, 1 / 255)
            ):
                raise ValueError(f"frozen_tower: the image processor of {name} does not scale by 1/255 and normalise")
            mean, std = processor.image_mean, processor.image_std
        elif vision_tower(nn.ModuleDict({"visual": tower})) is None:
            raise ValueError("frozen_tower: not a vision tower of the Qwen-VL kind")
        hf = tower.config
        self.patch, self.temporal, self.merge = hf.patch_size, hf.temporal_patch_size, hf.spatial_merge_size
        if image_size % (self.patch * self.merge):
            raise ValueError(
                f"frozen_tower: {image_size}-pixel images are not whole groups of {self.merge} x {self.merge} patches"
            )
        self.count = (image_size // self.patch) ** 2
        for buffer, value in (("mean", mean), ("std", std)):  # not in a checkpoint: they come with the tower
            value = np.broadcast_to(np.asarray(value, dtype=np.float32), 3)
            self.register_buffer(buffer, torch.tensor(value).view(1, 3, 1, 1), persistent=False)
        self.tower = tower.float().eval().requires_grad_(False)
        _batch_attention(self.tower)
        self.compiled = False
        self.frozen = [f"tower.{key}" for key in self.tower.state_dict()]
        self.register_state_dict_post_hook(_without_tower)
        self.proj = nn.Linear(hf.hidden_size, out_dim)
        self.norm = nn.LayerNorm(out_dim, eps=1e-6)

    def train(self, mode: bool = True) -> "FrozenTower":
        super().train(mode)
        self.tower.eval()
        return self

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        tower = self.tower.state_dict(keep_vars=True)
        for key in self.frozen:  # a checkpoint leaves the tower out: keep the weights loaded from its name
            state_dict.setdefault(prefix + key, tower[key.removeprefix("tower.")])
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def pixel_values(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x [N, 3, S, S] in 0..1 -> the model's pixel_values [N * (S / p)^2, 3 * t * p * p] and grid_thw [N, 3]."""
        n, channels, size = x.shape[0], x.shape[1], x.shape[-1]
        p, t, m = self.patch, self.temporal, self.merge
        side = size // p
        x = ((x - self.mean.to(x.dtype)) / self.std.to(x.dtype))[:, None].expand(n, t, channels, size, size)
        x = x.reshape(n, t, channels, side // m, m, p, side // m, m, p).permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return x.reshape(n * side * side, channels * t * p * p), torch.tensor([[1, side, side]] * n)

    def tokens(self, images: torch.Tensor) -> torch.Tensor:
        """The tower's normalised patch features [N, P, D] of processor-normalised images [N, 3, S, S]."""
        x = images * PROCESSOR_STD + PROCESSOR_MEAN
        dtype = next(self.tower.parameters()).dtype
        if self.training and x.is_cuda and not self.compiled:
            # Training only (a policy server stays eager): each block compiled, 1.6x the images per second with
            # Qwen3-VL-2B in bfloat16 on machine A, as close to the float32 tower as the eager blocks in bfloat16 (5% of
            # each token's norm). The tower is most of the cost of a training step.
            for block in self.tower.blocks:
                block.compile()
            self.compiled = True
        kept = []
        hook = self.tower.merger.norm.register_forward_hook(lambda _, args, out: kept.append(out))
        try:
            # in the weights' own type: under mixed precision autocast would run every norm and every rotary step in
            # float32 and cast back and forth, at half the speed (the policy trains with bfloat16 weights)
            with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
                pixels, grid = self.pixel_values(x.to(dtype))
                self.tower(pixels, grid_thw=grid)
        finally:
            hook.remove()
        if len(kept) != 1:
            raise RuntimeError(f"frozen_tower: the merger's norm ran {len(kept)} times in one forward, not once")
        return kept[0].reshape(images.shape[0], self.count, kept[0].shape[-1])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.tokens(images)
        return self.norm(self.proj(tokens.to(self.proj.weight.dtype)))


class FrozenViT(FrozenTower):
    """A frozen self-supervised vision transformer of the DINOv2 kind (DINOv2 with registers): images [N, 3, S, S] as
    the simple_vlm processor writes them -> [N, (S / p / pool)^2, out_dim].

    The model's last hidden states (its final LayerNorm applied) are read; the class token and the register tokens
    take part in the forward pass but are dropped from the spatial grid, which keeps the patch tokens only. pool > 1
    averages pool x pool neighbouring patch tokens into one (at 448 pixels and patches of 14, 32 x 32 patches become
    16 x 16 tokens with pool 2), to keep the action head's sequence near the ViT's length. The images are not
    resized: the processor gives them at the config's image_size, and the model interpolates its position embeddings
    to that size.
    Frozen, checkpointed and compiled as FrozenTower (only the projection and its norm are trained and saved)."""

    def __init__(
        self,
        name: str,
        out_dim: int,
        image_size: int,
        pool: int = 1,
        model: nn.Module | None = None,
        mean=None,
        std=None,
    ):
        nn.Module.__init__(self)
        if model is None:
            from transformers import AutoImageProcessor, AutoModel

            local = Path(name).is_dir()
            model = AutoModel.from_pretrained(name, local_files_only=local, dtype="auto")
            processor = AutoImageProcessor.from_pretrained(name, local_files_only=local)
            if not (
                processor.do_rescale and processor.do_normalize and math.isclose(processor.rescale_factor, 1 / 255)
            ):
                raise ValueError(f"frozen_tower: the image processor of {name} does not scale by 1/255 and normalise")
            mean, std = processor.image_mean, processor.image_std
        hf = model.config
        if not all(isinstance(getattr(hf, k, None), int) for k in ("patch_size", "hidden_size")):
            raise ValueError(f"frozen_tower: {name} ({type(model).__name__}) is not a vision transformer with patches")
        self.patch, self.pool = hf.patch_size, pool
        if image_size % (self.patch * pool):
            raise ValueError(
                f"frozen_tower: {image_size}-pixel images are not whole {pool} x {pool} groups of "
                f"{self.patch}-pixel patches"
            )
        self.side = image_size // self.patch
        self.count = (self.side // pool) ** 2
        for buffer, value in (("mean", mean), ("std", std)):
            value = np.broadcast_to(np.asarray(value, dtype=np.float32), 3)
            self.register_buffer(buffer, torch.tensor(value).view(1, 3, 1, 1), persistent=False)
        self.tower = model.float().eval().requires_grad_(False)
        self.compiled = False
        self.frozen = [f"tower.{key}" for key in self.tower.state_dict()]
        self.register_state_dict_post_hook(_without_tower)
        self.proj = nn.Linear(hf.hidden_size, out_dim)
        self.norm = nn.LayerNorm(out_dim, eps=1e-6)

    def tokens(self, images: torch.Tensor) -> torch.Tensor:
        """The model's patch tokens [N, (S / p / pool)^2, D] of processor-normalised images [N, 3, S, S]."""
        x = images * PROCESSOR_STD + PROCESSOR_MEAN
        dtype = next(self.tower.parameters()).dtype
        if self.training and x.is_cuda and not self.compiled:  # as FrozenTower: training only, layer by layer
            for layer in self.tower.encoder.layer:
                layer.compile()
            self.compiled = True
        with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
            x = ((x - self.mean.to(x.dtype)) / self.std.to(x.dtype)).to(dtype)
            hidden = self.tower(pixel_values=x).last_hidden_state
        patches = hidden[:, hidden.shape[1] - self.side**2 :]  # class token and registers come first
        if self.pool > 1:
            grid = patches.unflatten(1, (self.side, self.side)).permute(0, 3, 1, 2)
            patches = torch.nn.functional.avg_pool2d(grid, self.pool).flatten(2).transpose(1, 2)
        return patches


def frozen_encoder(name: str, out_dim: int, image_size: int, pool: int = 1) -> FrozenTower:
    """The frozen encoder for a model name: the vision tower of a vision-language model of the Qwen-VL kind
    (FrozenTower), or a vision transformer of the DINOv2 kind (FrozenViT)."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(name, local_files_only=Path(name).is_dir())
    if hasattr(config, "vision_config"):
        if pool != 1:
            raise ValueError("frozen_tower_pool: the Qwen-VL tower's patch tokens are not pooled")
        return FrozenTower(name, out_dim, image_size)
    return FrozenViT(name, out_dim, image_size, pool)
