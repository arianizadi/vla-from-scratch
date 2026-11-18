"""The eyes of architecture v2 (tutorial/architecture-v2.md): a small vision network that reads the scene from the
camera images and writes it as a scene record (scene.py): one record per side shoot of the plant, the flower's colour
and the card in the pot.

The reasoning policy of the first design looks at the plant through a vision tower that stayed frozen while the rest
of it was trained, and it writes "2 shoots" for 105 of 108 plants that have three (architecture-v2.md, section 1).
Here seeing is learned on its own, as a supervised problem: the simulator knows what is on the stand in every frame
(scene.scene_truth), in any state, also in states that only a policy reaches, and collect_scene.py stores that truth
beside the images.

The network (Eyes). The images (front and wrist by default; optionally the plant crop, and each camera as it was a
moment earlier) go through a vision transformer that keeps every patch token: a shoot is about 2 pixels wide in the
front image, and a summary of the image in a few tokens can lose it. A learned embedding per view says which camera
and which moment a token comes from. Learned queries then read the tokens through a small transformer decoder: one
query per part slot, one for the flower and one for the card. A part slot gives six numbers: whether there is a part,
whether it is cut, whether a camera shows it, its height on the main stem and the sine and cosine of its bearing. The
slots are filled from the lowest part up.

Why part records and not a count. How many shoots the plant has, and which of them is the lowest, the middle or the
highest, follow from the records; no head predicts them on its own. A count would not be enough for what comes after.
The scene memory knows a part by its height on the stem (it does not change when the plate turns, and the stub of a
cut shoot stays at that height): it matches what it is shown to what it remembers by height and marks a part cut when
that part is seen cut, and the plan names shoots by where they are on the plant. So the eyes have to say where each
part is, and the count follows. A count head could be right while the parts it counted are not.

Why every weight is trained. The tower of the first design was frozen and handed on 49 tokens per image; the models in
this repository whose vision is trained read the count from the same images. The default backbone is the repository's
own vision transformer (vla_foundry/models/vit.py, as in the small policies), trained from scratch with everything
else. A pretrained vision model from Hugging Face can take its place for comparison (backbone "hf"), frozen or with its
last layers trained.

The tower probe (backbone "tower") is a diagnostic, not a design: the decoder and heads of the eyes on the frozen
vision tower of the first design's reasoner (Qwen3-VL), trained on the same labels and data. Whether they read the
count from the tower's patch features, and whether still after the merger that packs four patches into each token the
language model is given, says where the frozen model loses it (_TowerBackbone).

Hidden parts. A part is trained on whether a camera shows it or not: the arm, a leaf or the stem hides a shoot for a
while, and the eyes learn what is probably there. `visible` says when they see the part rather than guess it, and the
memory counts only parts that are visible and confident.

Attributes (architecture v3, tutorial/architecture-v3.md: "Eyes: four more outputs per part"). With
EyesConfig.attributes (tie, sick, bud) every part slot also classifies each attribute of its part, with a class of its
own for "cannot be read in this view" (scene.UNREADABLE) beside the values a program can ask for (program.DOMAINS):
the truth's own, which says unreadable where no camera's rays reach where the attribute is to be seen, and for every
attribute of a cut part. A slot without a part, and a part whose truth has no attributes (a dataset collected before
them), leaves the attribute out of the loss. The truth decides readable by the front and the wrist cameras together:
eyes meant for the scene memory look through both (the plant crop is the front camera's), and eyes that look through
one camera (a probe of what that camera can read) are scored only where it can read (eyes_report, with the truth's
attribute_views). Without attributes the eyes are v2's, weight for weight.

What the percept is for. Eyes.decode writes the outputs in the percept form of scene.py: every slot, the lowest first,
with how sure the eyes are of each field (Scene.from_dict reads it). SceneMemory takes it from there. Eyes.see does
the whole step for one request, from the JPEG images a client sends (keys as eval.infer_request writes them) to the
percept.

The dataset reader is eyes_data.py, the trainer train_eyes.py.
"""

import contextlib
import dataclasses
import inspect
import io
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from .program import DOMAINS
from .reasoning import FLOWERS
from .scene import LABELS, TRUTH_ATTRIBUTES, UNREADABLE

PART_FIELDS = ("exists", "cut", "visible", "height", "sin", "cos")  # the six numbers of a part slot
# Architecture v3: the classes of each attribute a part slot can read, "cannot be read in this view" first
ATTRIBUTE_CLASSES = {name: (UNREADABLE, *DOMAINS[name]) for name in TRUTH_ATTRIBUTES}
# targets_from_scene's keys: v2's, then one class index per slot for each attribute
TARGETS = ("exists", "cut", "visible", "height", "bearing", "flower", "label", *TRUTH_ATTRIBUTES)
# The loss is a weighted sum of one term per field. The height weighs most: it is a part's identity (the memory matches
# parts by it, to within 17.5 mm), and its term, an error in 0..1 units, is small next to a cross-entropy.
LOSS_WEIGHTS = {
    "exists": 1.0,
    "cut": 1.0,
    "visible": 1.0,
    "height": 5.0,
    "bearing": 1.0,
    "flower": 1.0,
    "label": 1.0,
    **dict.fromkeys(TRUTH_ATTRIBUTES, 1.0),  # only the attributes the eyes read (EyesConfig.attributes)
}
P_MIN = 0.7  # evaluate: a reading the scene memory counts is one at least this sure (SceneMemory's p_min)
HEIGHT_BETA = 0.02  # smooth L1 turns from the squared error to the absolute error here, in 0..1 units: about 2.6 mm
NO_CLASS = -100  # a flower target that is left out of the loss: a scene whose flower is not known
# Which of the truth's views (front, wrist) shows what each camera shows: the plant crop is part of the front image
CAMERA_VIEWS = {"front": 0, "wrist": 1, "plant": 0}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CONFIG_FILE = "eyes_config.json"
TOWER_TOKENS = ("patches", "merged")  # the tower probe: before the tower's merger, or after it


def _r(x) -> float:
    """A plain float rounded to 4 decimals, as the scene's dict form stores numbers."""
    return round(float(x), 4)


# ------------------------------------------------------------------------------------------------------ config
@dataclass
class EyesConfig:
    """What the eyes look at and how they are built. A run directory keeps it as eyes_config.json (to_json)."""

    cameras: list[str] = field(default_factory=lambda: ["front", "wrist"])  # "plant": the plant crop (PlantCrop)
    frames: list[int] = field(default_factory=lambda: [0])  # offsets in control steps: [-20, 0] adds 1 s earlier
    image_size: int = 224
    slots: int = 3  # part slots: the most side shoots a plant has
    # "vit": the repository's ViT, trained from scratch; "hf": a Hugging Face vision model; "tower": the vision tower of
    # a vision-language model of the Qwen-VL kind (the probe of _TowerBackbone)
    backbone: str = "vit"
    patch_size: int = 16  # vit
    n_layers: int = 6  # vit
    hidden_dim: int = 256  # the width of the tokens: the ViT's and the decoder's (hf and tower tokens are projected)
    inter_dim: int = 1024  # the width of the MLPs, in the ViT and in the decoder
    n_heads: int = 4  # attention heads, in the ViT and in the decoder
    hf_name: str | None = None  # hf: a model name on the Hugging Face hub or a local directory
    hf_trainable_layers: int = 0  # hf: 0 = frozen; n = its last n encoder layers and its final norm are trained
    tower_name: str | None = None  # tower: a vision-language model's name on the hub or a local directory
    tower_tokens: str = "patches"  # tower: "patches", its last block's output, normalised; "merged", its merger's
    tower_trainable_layers: int = 0  # tower: 0 = frozen; n = its last n blocks are trained, and its merger if "merged"
    decoder_layers: int = 2
    height_range: tuple[float, float] = (0.05, 0.18)  # metres; heights are learned as 0..1 over this range
    loss_weights: dict[str, float] = field(default_factory=lambda: dict(LOSS_WEIGHTS))
    attributes: list[str] = field(default_factory=list)  # architecture v3: of TRUTH_ATTRIBUTES; [] are v2's eyes

    def __post_init__(self):
        self.cameras = [str(c) for c in self.cameras]
        self.frames = [int(k) for k in self.frames]
        self.height_range = (float(self.height_range[0]), float(self.height_range[1]))
        self.attributes = [str(a) for a in self.attributes]
        if set(self.attributes) - set(TRUTH_ATTRIBUTES) or len(set(self.attributes)) != len(self.attributes):
            raise ValueError(f"attributes {self.attributes}: each once, of {list(TRUTH_ATTRIBUTES)}")

        unknown = set(self.loss_weights) - set(LOSS_WEIGHTS)
        if unknown:
            raise ValueError(f"no loss terms {sorted(unknown)}; the terms are {list(LOSS_WEIGHTS)}")
        self.loss_weights = {k: float(self.loss_weights.get(k, w)) for k, w in LOSS_WEIGHTS.items()}
        if not self.cameras or len(set(self.cameras)) != len(self.cameras):
            raise ValueError(f"cameras {self.cameras}: at least one, each once")
        if not self.frames or len(set(self.frames)) != len(self.frames) or max(self.frames) > 0:
            raise ValueError(f"frames {self.frames}: offsets of 0 or less (the past), at least one, each once")
        if self.backbone not in ("vit", "hf", "tower"):
            raise ValueError(f"backbone {self.backbone!r}: 'vit', 'hf' or 'tower'")
        if self.backbone == "hf" and not self.hf_name:
            raise ValueError("the hf backbone needs hf_name (a model name or a local directory)")
        if self.backbone == "tower" and not self.tower_name:
            raise ValueError("the tower backbone needs tower_name (a model name or a local directory)")
        if self.tower_tokens not in TOWER_TOKENS:
            raise ValueError(f"tower_tokens {self.tower_tokens!r}: one of {TOWER_TOKENS}")
        if min(self.hf_trainable_layers, self.tower_trainable_layers) < 0 or self.slots < 1 or self.decoder_layers < 1:
            raise ValueError("hf_ and tower_trainable_layers must be 0 or more, slots and decoder_layers 1 or more")
        if not self.height_range[0] < self.height_range[1]:
            raise ValueError(f"height_range {self.height_range}: low, high")

    @property
    def views(self) -> list[tuple[str, int]]:
        """(camera, frame offset) of each image, in the order the model takes them: cameras outer, frames inner."""
        return [(c, k) for c in self.cameras for k in self.frames]

    @property
    def keys(self) -> list[str]:
        """The request key of each view, as eval.infer_request names images: "front_t0", "front_t-20", ..."""
        return [f"{c}_t{k}" for c, k in self.views]

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "EyesConfig":
        return cls(**json.loads(text))


# ----------------------------------------------------------------------------------------------------- targets
def targets_from_scene(scene: dict, config: EyesConfig) -> dict[str, torch.Tensor]:
    """The training targets of one frame, from its truth dict (scene.py's plain form; a percept dict is read too).

    The parts, sorted by height, fill the slots from the lowest up; a slot without a part has exists 0 and zeros
    elsewhere. Every part is a target whether or not a camera shows it. `visible` is whether one of the cameras the
    eyes look at shows it (the plant crop counts as the front camera): with the front and wrist cameras, any(vis).
    Shapes: exists, cut, visible, height [slots] (height in 0..1 over config.height_range), bearing [slots, 2] (sine,
    cosine), flower and label [] (class indices; a flower that is not known is NO_CLASS), and per attribute of
    TRUTH_ATTRIBUTES [slots] (an index of ATTRIBUTE_CLASSES; NO_CLASS for a slot without a part and for a part whose
    truth has no value for it)."""
    parts = sorted(scene.get("parts", []), key=lambda p: float(p["h"]))
    k = config.slots
    if len(parts) > k:
        raise ValueError(f"a scene with {len(parts)} parts does not fit in {k} slots")
    shown_by = sorted({CAMERA_VIEWS[c] for c in config.cameras if c in CAMERA_VIEWS})
    lo, hi = config.height_range
    exists, cut, visible, height = [0.0] * k, [0.0] * k, [0.0] * k, [0.0] * k
    bearing = [[0.0, 0.0] for _ in range(k)]
    attributes = {name: [NO_CLASS] * k for name in TRUTH_ATTRIBUTES}
    for i, part in enumerate(parts):
        for name, value in part.get("attr", {}).items():
            if name in attributes:
                attributes[name][i] = ATTRIBUTE_CLASSES[name].index(value)
        vis = part.get("vis", False)
        exists[i] = 1.0
        cut[i] = float(bool(part["cut"]))
        visible[i] = float(any(vis[j] for j in shown_by) if isinstance(vis, (list, tuple)) else bool(vis))
        height[i] = min(max((float(part["h"]) - lo) / (hi - lo), 0.0), 1.0)
        bearing[i] = [math.sin(float(part["b"])), math.cos(float(part["b"]))]
    flower = scene.get("flower")
    return dict(
        exists=torch.tensor(exists),
        cut=torch.tensor(cut),
        visible=torch.tensor(visible),
        height=torch.tensor(height),
        bearing=torch.tensor(bearing),
        flower=torch.tensor(NO_CLASS if flower is None else FLOWERS.index(flower)),
        label=torch.tensor(LABELS.index(scene.get("label", "none"))),
        **{name: torch.tensor(values) for name, values in attributes.items()},
    )


# --------------------------------------------------------------------------------------------------- backbones
class _ViTBackbone(nn.Module):
    """The repository's vision transformer (vla_foundry/models/vit.py), trained from scratch: [N, 3, S, S] in 0..1 to
    [N, P, D], every patch token kept."""

    def __init__(self, config: EyesConfig):
        super().__init__()
        from vla_foundry.models.vit import ViT  # imported here: the data-loader workers need none of vla_foundry
        from vla_foundry.params.model_params import ViTParams

        params = ViTParams(
            img_size=config.image_size,
            patch_size=config.patch_size,
            hidden_dim=config.hidden_dim,
            inter_dim=config.inter_dim,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
        )
        self.vit = ViT(params)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit((x - self.mean) / self.std)


def _image_stats(name: str, local: bool) -> tuple[list[float], list[float]]:
    """The image mean and std a Hugging Face model was trained with (its image processor's), else ImageNet's."""
    from transformers import AutoImageProcessor

    try:
        processor = AutoImageProcessor.from_pretrained(name, local_files_only=local)
        mean, std = processor.image_mean, processor.image_std
    except Exception:  # no image processor saved with the model
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    return [float(v) for v in np.broadcast_to(mean, 3)], [float(v) for v in np.broadcast_to(std, 3)]


def _first_int(value) -> int | None:
    """A size from a model's configuration, which some models give as [height, width]."""
    if value is None:
        return None
    return int(value[0]) if isinstance(value, (list, tuple)) else int(value)


def _encoder_layers(model: nn.Module) -> nn.ModuleList | None:
    """The list of a transformer's encoder layers, wherever Hugging Face models keep it."""
    for path in ("encoder.layer", "encoder.layers", "layers", "layer", "blocks"):
        module = model
        for name in path.split("."):
            module = getattr(module, name, None)
        if isinstance(module, nn.ModuleList):
            return module
    return None


def _final_norm(model: nn.Module) -> nn.Module | None:
    """The norm after a transformer's last encoder layer."""
    for name in ("layernorm", "post_layernorm", "norm", "final_layernorm", "ln_post", "layer_norm"):
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            return module
    return None


class _HFBackbone(nn.Module):
    """A Hugging Face vision model: [N, 3, S, S] in 0..1 to its patch tokens [N, P, D], projected to the eyes' width.

    The model's own image mean and std are used (buffers, so a checkpoint carries them). Its class token, and any
    other token in front of the patches, is dropped. All its weights are frozen except, with hf_trainable_layers n,
    its last n encoder layers and its final norm. `pretrained` False builds the model from its configuration only,
    for weights that a checkpoint brings."""

    def __init__(self, config: EyesConfig, pretrained: bool = True):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        name = config.hf_name
        local = Path(name).is_dir()  # a local directory is read as it is, never looked up on the hub
        # float32 whatever the model was saved in: training runs it under autocast, the average is kept in float32
        if pretrained:
            model = AutoModel.from_pretrained(name, local_files_only=local, dtype=torch.float32)
        else:
            hf_config = AutoConfig.from_pretrained(name, local_files_only=local)
            model = AutoModel.from_config(hf_config, dtype=torch.float32)
        self.model = getattr(model, "vision_model", model)  # a vision-language model: its vision part
        hf = self.model.config
        width = getattr(hf, "hidden_size", None) or hf.hidden_sizes[-1]
        self.proj = nn.Linear(width, config.hidden_dim)
        mean, std = _image_stats(name, local) if pretrained else (IMAGENET_MEAN, IMAGENET_STD)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        patch = _first_int(getattr(hf, "patch_size", None))
        self.patches = (config.image_size // patch) ** 2 if patch else None
        # images of another size than the model was trained on: its position embeddings interpolated, where it can
        native = _first_int(getattr(hf, "image_size", None))
        accepts = "interpolate_pos_encoding" in inspect.signature(self.model.forward).parameters
        self.kwargs = {"interpolate_pos_encoding": True} if accepts and native not in (None, config.image_size) else {}

        self.model.requires_grad_(False)
        n = config.hf_trainable_layers
        if n > 0:
            layers = _encoder_layers(self.model)
            if layers is None:
                raise ValueError(f"cannot find the encoder layers of {name} to train its last {n}")
            for layer in list(layers)[-n:]:
                layer.requires_grad_(True)
            norm = _final_norm(self.model)
            if norm is not None:
                norm.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.model(pixel_values=(x - self.mean) / self.std, **self.kwargs).last_hidden_state
        if tokens.ndim == 4:  # a convolutional model: a feature map [N, C, h, w]
            tokens = tokens.flatten(2).transpose(1, 2)
        elif self.patches is not None:  # the patch tokens come last, after the class token (and registers)
            tokens = tokens[:, tokens.shape[1] - self.patches :]
        return self.proj(tokens)


def _vision_tower(model: nn.Module) -> nn.Module | None:
    """The vision tower of a vision-language model of the Qwen-VL kind, or None for another kind: a module `visual`
    whose transformer `blocks` are followed by a `merger` that normalises each patch (`merger.norm`, a LayerNorm) and
    packs spatial_merge_size x spatial_merge_size neighbouring patches into one token for the language model, with a
    configuration that gives its patch sizes and its widths before the merger (hidden_size) and after it
    (out_hidden_size)."""
    tower = getattr(model, "visual", None)
    if tower is None:
        tower = getattr(getattr(model, "model", None), "visual", None)
    hf, merger = getattr(tower, "config", None), getattr(tower, "merger", None)
    sizes = ("patch_size", "temporal_patch_size", "spatial_merge_size", "hidden_size", "out_hidden_size")
    fits = (
        isinstance(tower, nn.Module)
        and isinstance(getattr(tower, "blocks", None), nn.ModuleList)
        and isinstance(getattr(merger, "norm", None), nn.LayerNorm)
        and not getattr(merger, "use_postshuffle_norm", False)  # the norm is over each patch, before the packing
        and all(isinstance(getattr(hf, size, None), int) for size in sizes)
    )
    return tower if fits else None


def _without_frozen(module: nn.Module, state_dict: dict, prefix: str, local_metadata: dict) -> None:
    """The tower backbone's state_dict post-hook: the weights it keeps frozen (module.frozen) are left out. A plain
    function of the module rather than a bound method, so that a copy of the eyes (the running average) keeps it."""
    for key in module.frozen:
        state_dict.pop(prefix + key, None)


class _TowerBackbone(nn.Module):
    """The vision tower of a vision-language model of the Qwen-VL kind, frozen: [N, 3, S, S] in 0..1 to [N, P, D],
    projected linearly to the eyes' width. With Qwen3-VL-2B it is the tower the first design's reasoner looks through.

    What it is for. That reasoner writes "2 shoots" for 105 of 108 plants with three (architecture-v2.md, section 1).
    Its tower is frozen: either its patch features do not carry the count, or the merger that packs each 2 x 2 group of
    patches into one token for the language model loses it, or the language model does not read it. The eyes' decoder
    and heads, trained on the frozen tower's tokens with the eyes' labels and data, tell the first two apart:

      tower_tokens "patches": the tower's last block's output as the merger reads it, each patch normalised by the
                              merger's own LayerNorm, before the merger packs each 2 x 2 group into one token: one
                              token per patch of 16 pixels (196 per 224-pixel image);
      tower_tokens "merged":  the merger's output, the tokens the language model is given in the place of the image (49
                              per 224-pixel image; the model's own get_image_features). Qwen3-VL also adds features of
                              three earlier blocks, merged by mergers of their own ("deepstack"), to the hidden states
                              of its first language-model layers; those are not among them.

    So the two differ by the packing alone. The last block's output itself is not what the merger reads: in
    Qwen3-VL-2B two of its 1024 channels reach about 2000 and vary from patch to patch (token norms from about 70 to
    2900; normalised, 5 to 46; the merged tokens' at most 60). Projected as it is, a few tokens would take all of the
    decoder's attention at the start of training, a handicap the merged tokens do not have.

    The model is loaded whole (AutoModelForImageTextToText) and only its tower is kept: the language model goes. The
    images are prepared here, on their device, as the model's image processor prepares them (pixel_values): its mean
    and std; the image repeated over the tower's temporal patch (to the tower an image is a video of one frame, twice);
    the patches in its order, each 2 x 2 group of neighbouring patches one after the other (which lets the merger pack
    them by a reshape), and each patch flattened as (channel, frame, row, column); grid_thw (1, S / 16, S / 16) per
    image. The image is not resized: the reasoner was given its 224-pixel images whole (qwen_reasoner.yaml bounds the
    processor's pixel count at 224 x 224), where Qwen3-VL's own bounds would scale them up to 256 pixels.

    The patch tokens are taken by a forward hook on the merger's norm, for the length of the tower's own forward: the
    patch embedding, the position embeddings and the attention within each image are the model's code, not a copy of
    it. The decoder does not care about the order of the tokens; they stay in the tower's.

    Frozen (tower_trainable_layers 0), the tower runs under torch.no_grad(). With n, its last n blocks are trained,
    and its merger with "merged"; the blocks before them hold no weight that needs a gradient, so autograd records
    nothing there. The tower stays in eval mode whatever the eyes' mode: it is read as it was trained.

    A checkpoint holds what is trained, the projection and any blocks (and merger) trained, not the frozen weights:
    whenever the eyes are built, also by load_eyes, the tower is loaded from tower_name, and loading a state dict
    keeps those weights where it does not carry them."""

    def __init__(self, config: EyesConfig):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModelForImageTextToText

        name = config.tower_name
        local = Path(name).is_dir()  # a local directory is read as it is, never looked up on the hub
        try:
            # the weights in the type they were saved in (Qwen3-VL: bfloat16): the language model is let go below, and
            # the tower turned into float32, exactly
            model = AutoModelForImageTextToText.from_pretrained(name, local_files_only=local, dtype="auto")
        except ValueError as error:  # a model of no vision-language class: the message lists every class it knows
            reason = str(error).split("Model type should be")[0].strip()
            raise ValueError(f"tower: {name} does not load as a vision-language model ({reason})") from error
        tower = _vision_tower(model)
        if tower is None:
            raise ValueError(
                f"tower: {name} ({type(model).__name__}) has no vision tower of the Qwen-VL kind: a module `visual` "
                "with transformer `blocks` and a `merger` that normalises each patch (`merger.norm`) and packs "
                "neighbouring patches into the language model's tokens, and patch_size, temporal_patch_size, "
                "spatial_merge_size, hidden_size and out_hidden_size in its configuration"
            )
        del model  # the language model is not kept
        self.tower = tower.float().eval()
        hf = tower.config
        self.patch, self.temporal, self.merge = hf.patch_size, hf.temporal_patch_size, hf.spatial_merge_size
        self.merged = config.tower_tokens == "merged"
        if config.image_size % (self.patch * self.merge):
            raise ValueError(
                f"tower: images of {config.image_size} pixels are not whole {self.merge} x {self.merge} groups of "
                f"patches of {self.patch} pixels: use a multiple of {self.patch * self.merge}"
            )
        side = config.image_size // self.patch
        self.count = side * side // (self.merge**2 if self.merged else 1)  # tokens per image

        # the image processor's numbers: its mean and std, and the patch sizes it cuts images into for this tower
        processor = AutoImageProcessor.from_pretrained(name, local_files_only=local)
        cut = tuple(getattr(processor, k, None) for k in ("patch_size", "temporal_patch_size", "merge_size"))
        if cut != (self.patch, self.temporal, self.merge):
            raise ValueError(
                f"tower: the image processor of {name} cuts patches (size, frames, merge) {cut}, its tower takes "
                f"{(self.patch, self.temporal, self.merge)}"
            )
        if not (processor.do_rescale and processor.do_normalize and math.isclose(processor.rescale_factor, 1 / 255)):
            raise ValueError(f"tower: the image processor of {name} does not scale pixels by 1/255 and normalise them")
        # not in a checkpoint: they come from tower_name, as the frozen weights do
        for buffer, value in (("mean", processor.image_mean), ("std", processor.image_std)):
            value = np.broadcast_to(np.asarray(value, dtype=np.float32), 3)
            self.register_buffer(buffer, torch.tensor(value).view(1, 3, 1, 1), persistent=False)

        self.tower.requires_grad_(False)
        self.frozen_all = config.tower_trainable_layers == 0
        n = config.tower_trainable_layers
        if n > len(self.tower.blocks):
            raise ValueError(f"tower_trainable_layers {n}: the tower of {name} has {len(self.tower.blocks)} blocks")
        trained = list(self.tower.blocks)[-n:] if n else []
        if n and self.merged:
            trained.append(self.tower.merger)
        for module in trained:
            module.requires_grad_(True)
        # Fixed here, not read from requires_grad when saving: the running average has requires_grad off everywhere
        names = {key for key, p in self.tower.named_parameters() if p.requires_grad}
        self.frozen = [f"tower.{key}" for key in self.tower.state_dict() if key not in names]
        self.register_state_dict_post_hook(_without_frozen)
        self.proj = nn.Linear(hf.out_hidden_size if self.merged else hf.hidden_size, config.hidden_dim)

    def train(self, mode: bool = True) -> "_TowerBackbone":
        super().train(mode)
        self.tower.eval()
        return self

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # A checkpoint leaves out the frozen weights (_without_frozen): they stay the ones loaded from tower_name. They
        # are put in where missing, before the tower's own modules read the state dict.
        tower = self.tower.state_dict(keep_vars=True)
        for key in self.frozen:
            state_dict.setdefault(prefix + key, tower[key.removeprefix("tower.")])
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def pixel_values(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Images [N, 3, S, S] in 0..1 as the model's image processor writes them: pixel_values [N * (S / p)^2,
        3 * t * p * p] (p pixels a patch, t frames) and grid_thw [N, 3], on the CPU as the processor gives it (the
        tower reads it with Python loops)."""
        n, channels, size = x.shape[0], x.shape[1], x.shape[-1]
        p, t, m = self.patch, self.temporal, self.merge
        side = size // p
        x = ((x.float() - self.mean) / self.std)[:, None].expand(n, t, channels, size, size)  # one frame, t times
        # [N, frame, channel, block row, row in block, pixel row, block column, column in block, pixel column] to
        # [N, block row, block column, row in block, column in block, channel, frame, pixel row, pixel column]
        x = x.reshape(n, t, channels, side // m, m, p, side // m, m, p).permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        return x.reshape(n * side * side, channels * t * p * p), torch.tensor([[1, side, side]] * n)

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """The tower's own tokens of images [N, 3, S, S] in 0..1, before the projection: [N, P, D], D its width
        before the merger (patches) or after it (merged)."""
        pixels, grid = self.pixel_values(x)
        kept = []  # the patches as the merger reads them: its norm's output
        hook = None
        if not self.merged:
            hook = self.tower.merger.norm.register_forward_hook(lambda _, args, out: kept.append(out))
        try:
            with torch.no_grad() if self.frozen_all else contextlib.nullcontext():
                out = self.tower(pixels, grid_thw=grid)
        finally:
            if hook is not None:
                hook.remove()
        if self.merged:
            tokens = out[0] if isinstance(out, (tuple, list)) else out  # Qwen3-VL: (merged, deepstack features)
        elif len(kept) == 1:
            tokens = kept[0]
        else:
            raise RuntimeError(f"tower: the merger's norm ran {len(kept)} times in one forward of the tower, not once")
        return tokens.reshape(x.shape[0], self.count, tokens.shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.tokens(x))


# ------------------------------------------------------------------------------------------------------- model
class Eyes(nn.Module):
    """Images to a scene: forward gives raw outputs, loss trains them, decode reads them as percepts, see serves one
    request. `pretrained` (hf backbone only) False skips the model's pretrained weights, for load_eyes; the tower
    backbone always loads its tower, whose frozen weights no checkpoint holds."""

    def __init__(self, config: EyesConfig | None = None, pretrained: bool = True):
        super().__init__()
        self.config = config = config or EyesConfig()
        width = config.hidden_dim
        if config.backbone == "vit":
            self.backbone = _ViTBackbone(config)
        elif config.backbone == "hf":
            self.backbone = _HFBackbone(config, pretrained)
        else:
            self.backbone = _TowerBackbone(config)
        self.view_embedding = nn.Parameter(torch.randn(len(config.views), width) * 0.02)
        self.queries = nn.Parameter(torch.randn(config.slots + 2, width) * 0.02)  # the slots, the flower, the card
        layer = nn.TransformerDecoderLayer(
            width, config.n_heads, config.inter_dim, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, config.decoder_layers, norm=nn.LayerNorm(width))
        self.part_head = nn.Linear(width, len(PART_FIELDS))
        self.flower_head = nn.Linear(width, len(FLOWERS))
        self.label_head = nn.Linear(width, len(LABELS))
        if config.attributes:  # v3; v2's eyes have no such module, and their checkpoints load as they were
            self.attribute_heads = nn.ModuleDict(
                {name: nn.Linear(width, len(ATTRIBUTE_CLASSES[name])) for name in config.attributes}
            )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """images [B, V, 3, S, S]: float in 0..1, or uint8 (scaled here, so that a batch can travel to the GPU as
        bytes), V = len(config.views) in that order. Returns logits for exists, cut, visible [B, slots], height in
        0..1 [B, slots], the bearing's (sine, cosine) [B, slots, 2], logits for flower [B, 4] and label [B, 5], and
        for each attribute of config.attributes logits [B, slots, len(ATTRIBUTE_CLASSES[name])] under its name."""
        config = self.config
        size, views = config.image_size, len(config.views)
        if images.ndim != 5 or tuple(images.shape[1:]) != (views, 3, size, size):
            raise ValueError(f"images {tuple(images.shape)}: expected [B, {views}, 3, {size}, {size}]")
        b, v = images.shape[:2]
        if images.dtype == torch.uint8:
            images = images.float() / 255
        tokens = self.backbone(images.flatten(0, 1))  # [B * V, P, D]
        tokens = (tokens.unflatten(0, (b, v)) + self.view_embedding[:, None]).flatten(1, 2)  # [B, V * P, D]
        h = self.decoder(self.queries.expand(b, -1, -1), tokens)
        k = config.slots
        # the heads in float32 whatever the autocast: a height in bfloat16 is good to a few tenths of a millimetre
        with torch.autocast(h.device.type, enabled=False):
            h = h.float()
            parts = self.part_head(h[:, :k])
            flower, label = self.flower_head(h[:, k]), self.label_head(h[:, k + 1])
            attributes = {name: self.attribute_heads[name](h[:, :k]) for name in config.attributes}
        return dict(
            exists=parts[..., 0],
            cut=parts[..., 1],
            visible=parts[..., 2],
            height=torch.sigmoid(parts[..., 3]),
            bearing=parts[..., 4:6],
            flower=flower,
            label=label,
            **attributes,
        )

    # ---------------------------------------------------------------------------------------------- the loss
    def loss_terms(self, outputs: dict, targets: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """The weighted loss and its terms, unweighted, as tensors (no wait for the GPU; loss gives plain floats).

        Binary cross-entropy for exists over every slot, and for cut and visible over the slots that hold a part;
        smooth L1 for the height (0..1) and the squared error of (sine, cosine) for the bearing, over the slots that
        hold a part, the bearing only where the part is not cut (a stub's bearing is not asked for); cross-entropy
        for the flower and the label, and for each attribute of config.attributes over the slots whose part has a
        value for it (UNREADABLE among them)."""
        device = outputs["exists"].device
        t = {k: targets[k].to(device) for k in TARGETS}
        out = {k: v.float() for k, v in outputs.items()}
        exists, uncut = t["exists"], t["exists"] * (1 - t["cut"])
        parts, attached = exists.sum().clamp(min=1), uncut.sum().clamp(min=1)

        def bce(name):
            return F.binary_cross_entropy_with_logits(out[name], t[name], reduction="none")

        height = F.smooth_l1_loss(out["height"], t["height"], reduction="none", beta=HEIGHT_BETA)
        bearing = F.mse_loss(out["bearing"], t["bearing"], reduction="none").mean(-1)
        known = (t["flower"] != NO_CLASS).sum().clamp(min=1)
        terms = dict(
            exists=bce("exists").mean(),
            cut=(bce("cut") * exists).sum() / parts,
            visible=(bce("visible") * exists).sum() / parts,
            height=(height * exists).sum() / parts,
            bearing=(bearing * uncut).sum() / attached,
            flower=F.cross_entropy(out["flower"], t["flower"], ignore_index=NO_CLASS, reduction="sum") / known,
            label=F.cross_entropy(out["label"], t["label"]),
        )
        for name in self.config.attributes:
            known = (t[name] != NO_CLASS).sum().clamp(min=1)
            logits = out[name].flatten(0, 1)
            terms[name] = F.cross_entropy(logits, t[name].flatten(), ignore_index=NO_CLASS, reduction="sum") / known
        weights = self.config.loss_weights
        return sum(weights[name] * term for name, term in terms.items()), terms

    def loss(self, outputs: dict, targets: dict) -> tuple[torch.Tensor, dict[str, float]]:
        """(the weighted loss, to train on; its terms, unweighted, as floats)."""
        total, terms = self.loss_terms(outputs, targets)
        values = torch.stack([term.detach() for term in terms.values()]).tolist()
        return total, dict(zip(terms, values, strict=True))

    # ------------------------------------------------------------------------------------------- the percept
    @torch.no_grad()
    def decode(self, outputs: dict) -> list[dict]:
        """One percept dict per batch row (scene.py's percept form, which Scene.from_dict reads): every slot, the
        lowest first, each {"h" (metres), "b", "cut", "vis", "p", "p_cut", "p_vis"} with cut and vis at p >= 0.5 (and
        with attributes "attr", the most probable class of each, and "p_attr", its probability), then the most
        probable flower and card state and their probabilities."""
        lo, hi = self.config.height_range
        p, p_cut, p_vis = (torch.sigmoid(outputs[k].float()).cpu().tolist() for k in ("exists", "cut", "visible"))
        heights = (lo + outputs["height"].float() * (hi - lo)).cpu().tolist()
        bearing = outputs["bearing"].float()
        bearings = torch.atan2(bearing[..., 0], bearing[..., 1]).cpu().tolist()
        p_flower, flower = (x.cpu().tolist() for x in torch.softmax(outputs["flower"].float(), -1).max(-1))
        p_label, label = (x.cpu().tolist() for x in torch.softmax(outputs["label"].float(), -1).max(-1))
        read = {  # attribute -> (probabilities [B][slots], classes [B][slots])
            name: tuple(x.cpu().tolist() for x in torch.softmax(outputs[name].float(), -1).max(-1))
            for name in self.config.attributes
        }
        percepts = []
        for i in range(len(p)):
            parts = [
                {
                    "h": _r(heights[i][k]),
                    "b": _r(bearings[i][k]),
                    "cut": p_cut[i][k] >= 0.5,
                    "vis": p_vis[i][k] >= 0.5,
                    "p": _r(p[i][k]),
                    "p_cut": _r(p_cut[i][k]),
                    "p_vis": _r(p_vis[i][k]),
                }
                for k in range(self.config.slots)
            ]
            if read:
                for k, part in enumerate(parts):
                    part["attr"] = {name: ATTRIBUTE_CLASSES[name][c[i][k]] for name, (_, c) in read.items()}
                    part["p_attr"] = {name: _r(probs[i][k]) for name, (probs, _) in read.items()}
            percepts.append(
                {
                    "parts": parts,
                    "flower": FLOWERS[flower[i]],
                    "label": LABELS[label[i]],
                    "p_flower": _r(p_flower[i]),
                    "p_label": _r(p_label[i]),
                }
            )
        return percepts

    @torch.no_grad()
    def see(self, images: dict[str, bytes]) -> dict:
        """The percept of one request: `images` maps "<camera>_t<offset>" to JPEG bytes (eval.infer_request's keys);
        every view of the config must be there (a missing one is a ValueError that names it), others are ignored."""
        missing = [key for key in self.config.keys if key not in images]
        if missing:
            raise ValueError(f"the eyes need the images {missing}; the request has {sorted(images)}")
        views = [decode_jpeg(images[key], self.config.image_size) for key in self.config.keys]
        batch = torch.from_numpy(np.stack(views))[None].to(self.queries.device)
        training = self.training
        self.eval()
        try:
            return self.decode(self(batch))[0]
        finally:
            self.train(training)


def decode_jpeg(data: bytes, size: int | None = None) -> np.ndarray:
    """JPEG bytes to uint8 [3, H, W] (RGB), resized to size x size when it is another size."""
    with Image.open(io.BytesIO(data)) as image:
        image = image.convert("RGB")
        if size is not None and image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.BILINEAR)
        return np.ascontiguousarray(np.asarray(image).transpose(2, 0, 1))


# -------------------------------------------------------------------------------------------------- evaluation
def _ratio(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


@torch.no_grad()
def evaluate(model: Eyes, batches: Iterable[dict]) -> dict:
    """How well `model` reads a validation set (batches as eyes_data.collate makes them), the trained model's own
    thresholds (p >= 0.5) throughout:

      frames                 frames seen
      count_acc              the number of slots with p >= 0.5 is the number of parts; count_acc_<n> over the frames
                             whose plant has n parts (frames_<n> of them; None where there are none)
      attached_acc           the same for the parts that are attached: existing and not cut
      exists_acc             per part on the plant: reported (p >= 0.5); absent_acc per slot without one: not reported
      cut_acc, visible_acc   per part on the plant
      height_err_mm          the mean height error over the parts a camera shows; bearing_err_deg the same for the
                             bearing, over the parts a camera shows that are not cut
      flower_acc             over the frames whose flower is known
      label_acc              over all frames; label_acc_card over the frames whose plant has a card (frames_card)

    and for each attribute A of config.attributes, over the parts whose truth has a value for it:

      A_readable             parts it can be read on (the truth is not UNREADABLE); A_unreadable, parts it cannot
      A_acc                  over the parts it can be read on: the most probable class is the truth (UNREADABLE is
                             wrong there); A_acc_<value> over those whose truth is <value>
      A_missed               over the parts it can be read on: called UNREADABLE
      A_unreadable_acc       over the parts it cannot be read on: called UNREADABLE; A_false_read, called readable
      A_counted              over the parts it can be read on: a reading the memory counts (readable, at least P_MIN
                             sure); A_counted_wrong, of those, the share with a value other than the truth's
      A_false_read_counted   over the parts it cannot be read on: called readable at least P_MIN sure (the truth has
                             no value there to judge the reading by)
    """
    config = model.config
    k = config.slots
    lo, hi = config.height_range
    device = next(model.parameters()).device
    totals: dict[str, float] = {}
    sums: dict[str, torch.Tensor] = {}  # one batch's, summed on the device and brought over together

    def add(name, value):
        sums[name] = value.sum().float()

    training = model.training
    model.eval()
    try:
        for batch in batches:
            sums.clear()
            out = model(batch["images"].to(device))
            t = {key: batch[key].to(device) for key in TARGETS}
            exists, cut, visible = t["exists"] > 0.5, t["cut"] > 0.5, t["visible"] > 0.5
            said = torch.sigmoid(out["exists"]) >= 0.5
            said_cut = torch.sigmoid(out["cut"]) >= 0.5
            said_visible = torch.sigmoid(out["visible"]) >= 0.5
            count, attached = exists.sum(1), (exists & ~cut).sum(1)
            count_right = said.sum(1) == count
            add("frames", torch.ones_like(count))
            add("count", count_right)
            for n in range(k + 1):
                add(f"frames_{n}", count == n)
                add(f"count_{n}", count_right & (count == n))
            add("attached", (said & ~said_cut).sum(1) == attached)
            add("parts", exists)
            add("exists", said & exists)
            add("empty", ~exists)
            add("absent", ~said & ~exists)
            add("cut", (said_cut == cut) & exists)
            add("visible_right", (said_visible == visible) & exists)
            seen = exists & visible
            add("seen", seen)
            add("height_mm", (out["height"].float() - t["height"]).abs() * (hi - lo) * 1000 * seen)
            seen_uncut = seen & ~cut
            turn = torch.atan2(out["bearing"][..., 0], out["bearing"][..., 1]) - torch.atan2(
                t["bearing"][..., 0], t["bearing"][..., 1]
            )
            turn = torch.remainder(turn + math.pi, 2 * math.pi) - math.pi
            add("seen_uncut", seen_uncut)
            add("bearing_deg", torch.rad2deg(turn.abs()) * seen_uncut)
            known = t["flower"] != NO_CLASS
            add("flowers", known)
            add("flower", (out["flower"].argmax(-1) == t["flower"]) & known)
            label_right = out["label"].argmax(-1) == t["label"]
            card = t["label"] != LABELS.index("none")
            add("label", label_right)
            add("cards", card)
            add("label_card", label_right & card)
            for name in config.attributes:
                truth = t[name]
                probs, said = torch.softmax(out[name].float(), -1).max(-1)
                labelled, unreadable = truth != NO_CLASS, truth == 0  # class 0 is UNREADABLE
                readable = labelled & ~unreadable
                called = said != 0
                counted = called & (probs >= P_MIN)
                add(f"{name}_readable", readable)
                add(f"{name}_unreadable", unreadable)
                add(f"{name}_right", (said == truth) & readable)
                add(f"{name}_missed", ~called & readable)
                add(f"{name}_unread_right", ~called & unreadable)
                add(f"{name}_counted", counted & readable)
                add(f"{name}_counted_wrong", counted & readable & (said != truth))
                add(f"{name}_false_counted", counted & unreadable)
                for c in range(1, len(ATTRIBUTE_CLASSES[name])):
                    add(f"{name}_of_{c}", truth == c)
                    add(f"{name}_right_{c}", (said == c) & (truth == c))
            for name, value in zip(sums, torch.stack(list(sums.values())).tolist(), strict=True):
                totals[name] = totals.get(name, 0.0) + value
    finally:
        model.train(training)

    def total(name):
        return totals.get(name, 0.0)

    metrics = {"frames": int(total("frames")), "count_acc": _ratio(total("count"), total("frames"))}
    for n in range(k + 1):
        metrics[f"count_acc_{n}"] = _ratio(total(f"count_{n}"), total(f"frames_{n}"))
    for n in range(k + 1):
        metrics[f"frames_{n}"] = int(total(f"frames_{n}"))
    metrics.update(
        attached_acc=_ratio(total("attached"), total("frames")),
        parts=int(total("parts")),
        exists_acc=_ratio(total("exists"), total("parts")),
        absent_acc=_ratio(total("absent"), total("empty")),
        cut_acc=_ratio(total("cut"), total("parts")),
        visible_acc=_ratio(total("visible_right"), total("parts")),
        height_err_mm=_ratio(total("height_mm"), total("seen")),
        bearing_err_deg=_ratio(total("bearing_deg"), total("seen_uncut")),
        flower_acc=_ratio(total("flower"), total("flowers")),
        label_acc=_ratio(total("label"), total("frames")),
        label_acc_card=_ratio(total("label_card"), total("cards")),
        frames_card=int(total("cards")),
    )
    for name in config.attributes:
        readable, unreadable = total(f"{name}_readable"), total(f"{name}_unreadable")
        metrics.update(
            {
                f"{name}_readable": int(readable),
                f"{name}_unreadable": int(unreadable),
                f"{name}_acc": _ratio(total(f"{name}_right"), readable),
                f"{name}_missed": _ratio(total(f"{name}_missed"), readable),
                f"{name}_unreadable_acc": _ratio(total(f"{name}_unread_right"), unreadable),
                f"{name}_false_read": _ratio(unreadable - total(f"{name}_unread_right"), unreadable),
                f"{name}_counted": _ratio(total(f"{name}_counted"), readable),
                f"{name}_counted_wrong": _ratio(total(f"{name}_counted_wrong"), total(f"{name}_counted")),
                f"{name}_false_read_counted": _ratio(total(f"{name}_false_counted"), unreadable),
            }
        )
        for c, value in enumerate(ATTRIBUTE_CLASSES[name][1:], start=1):
            metrics[f"{name}_acc_{value}"] = _ratio(total(f"{name}_right_{c}"), total(f"{name}_of_{c}"))
    return metrics


# ------------------------------------------------------------------------------------------------- checkpoints
def checkpoints(run_dir: str | Path) -> dict[int, Path]:
    """A run's checkpoints by number: checkpoints/checkpoint_<k>.pt."""
    found = {}
    for path in (Path(run_dir) / "checkpoints").glob("checkpoint_*.pt"):
        match = re.fullmatch(r"checkpoint_(\d+)\.pt", path.name)
        if match:
            found[int(match.group(1))] = path
    return dict(sorted(found.items()))


def save(run_dir: str | Path, k: int, model: Eyes, ema: nn.Module | None = None, **state) -> Path:
    """Write checkpoint k of a run: checkpoints/checkpoint_<k>.pt with "model" (the trained weights), "ema" (the
    averaged ones; the trained ones again without an average) and whatever else `state` holds (the trainer's optimizer
    and counters), and eyes_config.json beside it. The file is written under another name and then renamed, so that a
    run stopped while saving leaves no checkpoint cut short."""
    run = Path(run_dir)
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run / CONFIG_FILE).write_text(model.config.to_json())
    path = run / "checkpoints" / f"checkpoint_{k}.pt"
    partial = path.with_name(path.name + ".partial")
    data = dict(checkpoint=k, config=dataclasses.asdict(model.config), model=model.state_dict())
    data["ema"] = (ema if ema is not None else model).state_dict()
    torch.save({**data, **state}, partial)
    partial.replace(path)
    return path


def load_eyes(run_dir: str | Path, checkpoint: int | None = None, weights: str = "ema", device: str = "cpu") -> Eyes:
    """The eyes of a run (train_eyes.py), in eval mode: checkpoint `checkpoint` (the latest by default), its averaged
    weights ("ema", as the policies are evaluated) or the trained ones ("model"). The tower backbone's frozen weights
    are loaded from its tower_name again (the checkpoint holds the rest)."""
    if weights not in ("ema", "model"):
        raise ValueError(f"weights {weights!r}: 'ema' or 'model'")
    run = Path(run_dir)
    found = checkpoints(run)
    if not found:
        raise FileNotFoundError(f"no checkpoints/checkpoint_<k>.pt in {run}")
    k = max(found) if checkpoint is None else checkpoint
    if k not in found:
        raise FileNotFoundError(f"no checkpoint {k} in {run} (there are {list(found)})")
    config = EyesConfig.from_json((run / CONFIG_FILE).read_text())
    model = Eyes(config, pretrained=False)
    state = torch.load(found[k], map_location="cpu", weights_only=True)
    model.load_state_dict(state[weights])
    return model.to(device).eval()
