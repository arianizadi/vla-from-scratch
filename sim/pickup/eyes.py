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

Hidden parts. A part is trained on whether a camera shows it or not: the arm, a leaf or the stem hides a shoot for a
while, and the eyes learn what is probably there. `visible` says when they see the part rather than guess it, and the
memory counts only parts that are visible and confident.

What the percept is for. Eyes.decode writes the outputs in the percept form of scene.py: every slot, the lowest first,
with how sure the eyes are of each field (Scene.from_dict reads it). SceneMemory takes it from there. Eyes.see does
the whole step for one request, from the JPEG images a client sends (keys as eval.infer_request writes them) to the
percept.

The dataset reader is eyes_data.py, the trainer train_eyes.py.
"""

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

from .reasoning import FLOWERS
from .scene import LABELS

PART_FIELDS = ("exists", "cut", "visible", "height", "sin", "cos")  # the six numbers of a part slot
TARGETS = ("exists", "cut", "visible", "height", "bearing", "flower", "label")  # targets_from_scene's keys
# The loss is a weighted sum of one term per field. The height weighs most: it is a part's identity (the memory matches
# parts by it, to within 17.5 mm), and its term, an error in 0..1 units, is small next to a cross-entropy.
LOSS_WEIGHTS = {"exists": 1.0, "cut": 1.0, "visible": 1.0, "height": 5.0, "bearing": 1.0, "flower": 1.0, "label": 1.0}
HEIGHT_BETA = 0.02  # smooth L1 turns from the squared error to the absolute error here, in 0..1 units: about 2.6 mm
NO_CLASS = -100  # a flower target that is left out of the loss: a scene whose flower is not known
# Which of the truth's views (front, wrist) shows what each camera shows: the plant crop is part of the front image
CAMERA_VIEWS = {"front": 0, "wrist": 1, "plant": 0}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CONFIG_FILE = "eyes_config.json"


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
    backbone: str = "vit"  # "vit": the repository's ViT, trained from scratch; "hf": a Hugging Face vision model
    patch_size: int = 16  # vit
    n_layers: int = 6  # vit
    hidden_dim: int = 256  # the width of the tokens: the ViT's and the decoder's (an hf model's are projected to it)
    inter_dim: int = 1024  # the width of the MLPs, in the ViT and in the decoder
    n_heads: int = 4  # attention heads, in the ViT and in the decoder
    hf_name: str | None = None  # hf: a model name on the Hugging Face hub or a local directory
    hf_trainable_layers: int = 0  # hf: 0 = frozen; n = its last n encoder layers and its final norm are trained
    decoder_layers: int = 2
    height_range: tuple[float, float] = (0.05, 0.18)  # metres; heights are learned as 0..1 over this range
    loss_weights: dict[str, float] = field(default_factory=lambda: dict(LOSS_WEIGHTS))

    def __post_init__(self):
        self.cameras = [str(c) for c in self.cameras]
        self.frames = [int(k) for k in self.frames]
        self.height_range = (float(self.height_range[0]), float(self.height_range[1]))
        unknown = set(self.loss_weights) - set(LOSS_WEIGHTS)
        if unknown:
            raise ValueError(f"no loss terms {sorted(unknown)}; the terms are {list(LOSS_WEIGHTS)}")
        self.loss_weights = {k: float(self.loss_weights.get(k, w)) for k, w in LOSS_WEIGHTS.items()}
        if not self.cameras or len(set(self.cameras)) != len(self.cameras):
            raise ValueError(f"cameras {self.cameras}: at least one, each once")
        if not self.frames or len(set(self.frames)) != len(self.frames) or max(self.frames) > 0:
            raise ValueError(f"frames {self.frames}: offsets of 0 or less (the past), at least one, each once")
        if self.backbone not in ("vit", "hf"):
            raise ValueError(f"backbone {self.backbone!r}: 'vit' or 'hf'")
        if self.backbone == "hf" and not self.hf_name:
            raise ValueError("the hf backbone needs hf_name (a model name or a local directory)")
        if self.hf_trainable_layers < 0 or self.slots < 1 or self.decoder_layers < 1:
            raise ValueError("hf_trainable_layers must be 0 or more, slots and decoder_layers 1 or more")
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
    cosine), flower and label [] (class indices; a flower that is not known is NO_CLASS)."""
    parts = sorted(scene.get("parts", []), key=lambda p: float(p["h"]))
    k = config.slots
    if len(parts) > k:
        raise ValueError(f"a scene with {len(parts)} parts does not fit in {k} slots")
    shown_by = sorted({CAMERA_VIEWS[c] for c in config.cameras if c in CAMERA_VIEWS})
    lo, hi = config.height_range
    exists, cut, visible, height = [0.0] * k, [0.0] * k, [0.0] * k, [0.0] * k
    bearing = [[0.0, 0.0] for _ in range(k)]
    for i, part in enumerate(parts):
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


# ------------------------------------------------------------------------------------------------------- model
class Eyes(nn.Module):
    """Images to a scene: forward gives raw outputs, loss trains them, decode reads them as percepts, see serves one
    request. `pretrained` (hf backbone only) False skips the model's pretrained weights, for load_eyes."""

    def __init__(self, config: EyesConfig | None = None, pretrained: bool = True):
        super().__init__()
        self.config = config = config or EyesConfig()
        width = config.hidden_dim
        self.backbone = _ViTBackbone(config) if config.backbone == "vit" else _HFBackbone(config, pretrained)
        self.view_embedding = nn.Parameter(torch.randn(len(config.views), width) * 0.02)
        self.queries = nn.Parameter(torch.randn(config.slots + 2, width) * 0.02)  # the slots, the flower, the card
        layer = nn.TransformerDecoderLayer(
            width, config.n_heads, config.inter_dim, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, config.decoder_layers, norm=nn.LayerNorm(width))
        self.part_head = nn.Linear(width, len(PART_FIELDS))
        self.flower_head = nn.Linear(width, len(FLOWERS))
        self.label_head = nn.Linear(width, len(LABELS))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """images [B, V, 3, S, S]: float in 0..1, or uint8 (scaled here, so that a batch can travel to the GPU as
        bytes), V = len(config.views) in that order. Returns logits for exists, cut, visible [B, slots], height in
        0..1 [B, slots], the bearing's (sine, cosine) [B, slots, 2], and logits for flower [B, 4] and label [B, 5]."""
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
        return dict(
            exists=parts[..., 0],
            cut=parts[..., 1],
            visible=parts[..., 2],
            height=torch.sigmoid(parts[..., 3]),
            bearing=parts[..., 4:6],
            flower=flower,
            label=label,
        )

    # ---------------------------------------------------------------------------------------------- the loss
    def loss_terms(self, outputs: dict, targets: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """The weighted loss and its terms, unweighted, as tensors (no wait for the GPU; loss gives plain floats).

        Binary cross-entropy for exists over every slot, and for cut and visible over the slots that hold a part;
        smooth L1 for the height (0..1) and the squared error of (sine, cosine) for the bearing, over the slots that
        hold a part, the bearing only where the part is not cut (a stub's bearing is not asked for); cross-entropy
        for the flower and the label."""
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
        lowest first, each {"h" (metres), "b", "cut", "vis", "p", "p_cut", "p_vis"} with cut and vis at p >= 0.5, then
        the most probable flower and card state and their probabilities."""
        lo, hi = self.config.height_range
        p, p_cut, p_vis = (torch.sigmoid(outputs[k].float()).cpu().tolist() for k in ("exists", "cut", "visible"))
        heights = (lo + outputs["height"].float() * (hi - lo)).cpu().tolist()
        bearing = outputs["bearing"].float()
        bearings = torch.atan2(bearing[..., 0], bearing[..., 1]).cpu().tolist()
        p_flower, flower = (x.cpu().tolist() for x in torch.softmax(outputs["flower"].float(), -1).max(-1))
        p_label, label = (x.cpu().tolist() for x in torch.softmax(outputs["label"].float(), -1).max(-1))
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
    weights ("ema", as the policies are evaluated) or the trained ones ("model")."""
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
