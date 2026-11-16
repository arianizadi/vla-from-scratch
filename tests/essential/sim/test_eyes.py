"""The eyes of architecture v2 (sim/pickup/eyes.py, eyes_data.py, train_eyes.py): a small vision model that turns the
camera images into a scene record, its targets, loss, percepts, evaluation, checkpoints and serving entry point, the
reader of the perception dataset and the trainer.

Everything runs on a CPU with small models (32-pixel images), except the trainer, which builds the default model. The
Hugging Face backbone is a tiny random model saved into the test's directory: nothing is downloaded. The reader and the
trainer need pyarrow to read the dataset and are skipped where it is missing.
"""

import copy
import io
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:  # sim/ is not an installed package
    sys.path.insert(0, str(REPOSITORY))

from sim.pickup import eyes as E  # noqa: E402
from sim.pickup import scene as S  # noqa: E402
from sim.pickup.eyes_data import SceneFrames, augment, collate  # noqa: E402
from sim.pickup.reasoning import FLOWERS  # noqa: E402

# A plant as the truth dict stores it: the parts out of order, the upper one cut and seen by the wrist camera only, the
# lower one hidden from both cameras
TRUTH = {
    "parts": [
        {"h": 0.1203, "b": 2.7, "cut": True, "vis": [False, True]},
        {"h": 0.0834, "b": -1.2051, "cut": False, "vis": [False, False]},
    ],
    "flower": "purple",
    "label": "unreadable",
}


def _config(**changes) -> E.EyesConfig:
    """A small model: 32-pixel images in patches of 8 (16 tokens per view), 32 wide."""
    small = dict(image_size=32, patch_size=8, hidden_dim=32, inter_dim=64, n_heads=2, n_layers=2, decoder_layers=1)
    return E.EyesConfig(**{**small, **changes})


def _stack(targets: list[dict]) -> dict:
    return {key: torch.stack([t[key] for t in targets]) for key in targets[0]}


def _jpeg(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _drawn_scenes(n: int, seed: int = 1) -> list[dict]:
    """Truth dicts with 0 to 3 parts, every flower and card state, some parts cut and some hidden."""
    rng = np.random.default_rng(seed)
    scenes = []
    for i in range(n):
        parts = []
        for h in np.sort(rng.uniform(0.07, 0.155, size=(0, 1, 2, 3, 2, 3, 3, 2)[i % 8])):
            b, cut = round(float(rng.uniform(-3, 3)), 4), bool(rng.random() < 0.3)
            vis = [bool(rng.random() < 0.7), bool(rng.random() < 0.5)]
            parts.append({"h": round(float(h), 4), "b": b, "cut": cut, "vis": vis})
        scenes.append({"parts": parts, "flower": FLOWERS[i % 4], "label": S.LABELS[i % 5]})
    return scenes


# -------------------------------------------------------------------------------------------- config and targets
def test_targets_fill_the_slots_from_the_lowest_part_up():
    config = _config()
    t = E.targets_from_scene(TRUTH, config)
    lo, hi = config.height_range
    assert t["exists"].tolist() == [1, 1, 0]
    assert t["height"].tolist() == pytest.approx([(0.0834 - lo) / (hi - lo), (0.1203 - lo) / (hi - lo), 0.0])
    assert t["cut"].tolist() == [0, 1, 0]  # the upper part, now in the second slot, is cut
    assert t["visible"].tolist() == [0, 1, 0]  # the lower one is hidden; any camera that shows a part makes it visible
    expected = [math.sin(-1.2051), math.cos(-1.2051), math.sin(2.7), math.cos(2.7), 0.0, 0.0]
    assert t["bearing"].flatten().tolist() == pytest.approx(expected, abs=1e-6)
    assert int(t["flower"]) == FLOWERS.index("purple") and int(t["label"]) == S.LABELS.index("unreadable")
    assert {k: tuple(v.shape) for k, v in t.items()} == dict(
        exists=(3,), cut=(3,), visible=(3,), height=(3,), bearing=(3, 2), flower=(), label=()
    )
    # eyes that look through the front camera alone do not see what only the wrist camera shows
    assert E.targets_from_scene(TRUTH, _config(cameras=["front"]))["visible"].tolist() == [0, 0, 0]
    # a flower that is not known is left out of the loss; a plant with more parts than slots is refused
    assert int(E.targets_from_scene({**TRUTH, "flower": None}, config)["flower"]) == E.NO_CLASS
    with pytest.raises(ValueError, match="slots"):
        E.targets_from_scene(TRUTH, _config(slots=1))


def test_the_config_round_trips_through_json_and_refuses_what_cannot_be_built():
    config = E.EyesConfig(cameras=["front", "wrist", "plant"], frames=[-20, 0], loss_weights={"height": 8.0})
    back = E.EyesConfig.from_json(config.to_json())
    assert back == config and back.height_range == (0.05, 0.18) and back.loss_weights["height"] == 8.0
    assert back.loss_weights["flower"] == E.LOSS_WEIGHTS["flower"]
    assert config.keys == ["front_t-20", "front_t0", "wrist_t-20", "wrist_t0", "plant_t-20", "plant_t0"]
    for bad in (dict(frames=[5]), dict(backbone="hf"), dict(backbone="cnn"), dict(loss_weights={"count": 1.0})):
        with pytest.raises(ValueError):
            E.EyesConfig(**bad)


# ------------------------------------------------------------------------------------------------------ model
@pytest.mark.parametrize(
    "changes, views",
    [(dict(), 2), (dict(cameras=["front", "wrist", "plant"]), 3), (dict(frames=[-20, 0]), 4)],
    ids=["front-wrist", "with-the-plant-crop", "two-frames"],
)
def test_forward_gives_every_output_for_each_set_of_views(changes, views):
    torch.manual_seed(0)
    config = _config(**changes)
    model = E.Eyes(config)
    assert len(config.views) == views
    images = torch.randint(0, 256, (2, views, 3, 32, 32), dtype=torch.uint8)
    out = model(images.float() / 255)
    shapes = {k: tuple(v.shape) for k, v in out.items()}
    slots = dict(exists=(2, 3), cut=(2, 3), visible=(2, 3), height=(2, 3), bearing=(2, 3, 2))
    assert shapes == dict(**slots, flower=(2, 4), label=(2, 5))
    assert ((out["height"] > 0) & (out["height"] < 1)).all()
    # bytes are scaled on the way in: the same outputs as floats in 0..1
    assert torch.allclose(model(images)["exists"], out["exists"], atol=1e-5)
    with pytest.raises(ValueError, match="expected"):
        model(torch.rand(2, views + 1, 3, 32, 32))


@pytest.fixture(scope="module")
def overfit():
    """A small model trained on eight random images, each with fixed targets, until it knows them."""
    torch.manual_seed(0)
    config = _config()
    model = E.Eyes(config)
    images = torch.rand(8, 2, 3, 32, 32)
    scenes = _drawn_scenes(8)
    targets = _stack([E.targets_from_scene(s, config) for s in scenes])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(200):
        loss, _ = model.loss(model(images), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
    return model.eval(), images, scenes, targets, losses


def test_the_loss_falls_by_more_than_half_when_the_model_overfits_eight_images(overfit):
    model, images, _, targets, losses = overfit
    assert losses[-1] < 0.5 * losses[0], (losses[0], losses[-1])
    # the loss is the weighted sum of its terms, which loss gives unweighted
    total, terms = model.loss(model(images), targets)
    assert set(terms) == set(E.LOSS_WEIGHTS) and all(isinstance(v, float) for v in terms.values())
    assert float(total) == pytest.approx(sum(E.LOSS_WEIGHTS[k] * v for k, v in terms.items()), rel=1e-5)


def test_after_the_overfit_decode_gives_percepts_that_count_the_parts(overfit):
    model, images, scenes, _, _ = overfit
    percepts = model.decode(model(images))
    assert len(percepts) == len(scenes)
    for percept, truth in zip(percepts, scenes, strict=True):
        read = S.Scene.from_dict(percept)
        assert read.to_dict() == percept  # the percept form, nothing lost or added
        assert len(read.parts) == 3  # every slot, with its probabilities
        assert sum(p.p >= 0.5 for p in read.parts) == len(truth["parts"]), (percept, truth)
        for part, true in zip(read.parts, truth["parts"], strict=False):  # the slots that hold a part, lowest first
            assert part.height == pytest.approx(true["h"], abs=0.003)  # back in metres
            assert part.cut == true["cut"] and part.visible == any(true["vis"])
        assert (read.flower, read.label) == (truth["flower"], truth["label"])
        # the scene memory takes the percept as it is: it counts the parts the eyes see and are sure of
        memory = S.SceneMemory(stable=1, p_min=0.5)
        memory.update(read, 0)
        assert len(memory.parts) == sum(p.p >= 0.5 and p.visible for p in read.parts)


class _Fixed(torch.nn.Module):
    """Stands in for the eyes in evaluate: frame i's outputs whatever its images, which are filled with i."""

    def __init__(self, config: E.EyesConfig, outputs: dict):
        super().__init__()
        self.config, self.outputs = config, outputs
        self.anchor = torch.nn.Parameter(torch.zeros(1))  # evaluate finds the device by the parameters

    def forward(self, images):
        frames = images[:, 0, 0, 0, 0].long()
        return {k: v[frames] for k, v in self.outputs.items()}


def test_evaluate_on_a_hand_made_case():
    config = _config()
    lo, hi = config.height_range
    truths = [
        {  # three parts: the middle one cut, the highest hidden; a card
            "parts": [
                {"h": 0.08, "b": 0.0, "cut": False, "vis": [True, False]},
                {"h": 0.115, "b": 1.0, "cut": True, "vis": [False, True]},
                {"h": 0.15, "b": -2.0, "cut": False, "vis": [False, False]},
            ],
            "flower": "red",
            "label": "cyan",
        },
        {  # two parts; no card
            "parts": [
                {"h": 0.09, "b": 3.0, "cut": False, "vis": [True, True]},
                {"h": 0.14, "b": -3.0, "cut": False, "vis": [True, False]},
            ],
            "flower": "pink",
            "label": "none",
        },
    ]

    def logits(*says):
        return [10.0 if s else -10.0 for s in says]

    def turn(*angles):
        return [[math.sin(a), math.cos(a)] for a in angles]

    outputs = dict(
        exists=torch.tensor([logits(1, 1, 1), logits(1, 1, 1)]),  # frame 1: a third part that is not there
        cut=torch.tensor([logits(0, 1, 0), logits(0, 0, 0)]),
        visible=torch.tensor([logits(1, 1, 1), logits(1, 1, 0)]),  # frame 0: the hidden part said visible
        height=(torch.tensor([[0.082, 0.111, 0.16], [0.091, 0.137, 0.10]]) - lo) / (hi - lo),  # 2, 4; 1, 3 mm off
        bearing=torch.tensor([turn(math.radians(10), 1.0, 0.0), turn(-3.0, -3.0, 0.0)]),  # 10 deg; across +-pi; 0
        flower=torch.tensor([[9.0, 0, 0, 0], [0, 9.0, 0, 0]]),  # red (right), orange (wrong)
        label=torch.tensor([[0, 9.0, 0, 0, 0], [9.0, 0, 0, 0, 0]]),  # unreadable (wrong), none (right)
    )
    model = _Fixed(config, outputs)
    # one frame per batch, so that the sums run over batches; its images (as bytes) hold its number
    samples = [{"images": torch.full((2, 3, 32, 32), i, dtype=torch.uint8), "scene": t} for i, t in enumerate(truths)]
    m = E.evaluate(model, [collate([s], config, as_float=False) for s in samples])
    assert model.training  # evaluate puts the model back as it found it
    across = math.degrees(2 * math.pi - 6.0)  # from 3.0 rad to -3.0 rad the short way round
    expected = dict(
        frames=2,
        count_acc=0.5,
        count_acc_0=None,
        count_acc_1=None,
        count_acc_2=0.0,
        count_acc_3=1.0,
        frames_0=0,
        frames_1=0,
        frames_2=1,
        frames_3=1,
        attached_acc=0.5,
        parts=5,
        exists_acc=1.0,
        absent_acc=0.0,
        cut_acc=1.0,
        visible_acc=0.8,
        height_err_mm=2.5,
        bearing_err_deg=(10 + across + 0) / 3,
        flower_acc=0.5,
        label_acc=0.5,
        label_acc_card=0.0,
        frames_card=1,
    )
    assert set(m) == set(expected)
    for key, value in expected.items():
        assert m[key] == (None if value is None else pytest.approx(value, abs=1e-4)), key
    assert json.loads(json.dumps(m)) == m  # plain numbers, for metrics.jsonl


# -------------------------------------------------------------------------------------------- checkpoints, serving
def test_save_then_load_gives_the_same_outputs_for_trained_and_averaged_weights(tmp_path):
    torch.manual_seed(0)
    config = _config()
    images = torch.rand(3, 2, 3, 32, 32)
    runs = []
    for k in (1, 2):
        model = E.Eyes(config)
        average = copy.deepcopy(model)
        with torch.no_grad():
            for p in average.parameters():
                p.add_(0.05 * torch.randn_like(p))
        E.save(tmp_path, k, model, average, step=10 * k)
        runs.append((model.eval(), average.eval()))
    assert E.EyesConfig.from_json((tmp_path / E.CONFIG_FILE).read_text()) == config
    assert list(E.checkpoints(tmp_path)) == [1, 2] and not list(tmp_path.glob("checkpoints/*.partial"))

    def same(a, b):
        out_a, out_b = a(images), b(images)
        return all(torch.equal(out_a[k], out_b[k]) for k in out_a)

    latest = E.load_eyes(tmp_path)  # the latest checkpoint, averaged weights
    assert not latest.training and same(latest, runs[1][1]) and not same(latest, runs[1][0])
    assert same(E.load_eyes(tmp_path, weights="model"), runs[1][0])
    assert same(E.load_eyes(tmp_path, checkpoint=1, weights="model"), runs[0][0])
    assert same(E.load_eyes(tmp_path, checkpoint=1), runs[0][1])
    with pytest.raises(ValueError):
        E.load_eyes(tmp_path, weights="best")
    with pytest.raises(FileNotFoundError):
        E.load_eyes(tmp_path, checkpoint=3)


def _tiny_hf_model(path: Path) -> Path:
    """A random two-layer ViT saved as a Hugging Face model, with an image processor of its own."""
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.ViTConfig(
        image_size=32, patch_size=8, hidden_size=24, num_hidden_layers=2, num_attention_heads=2, intermediate_size=48
    )
    transformers.ViTModel(config).save_pretrained(path)
    transformers.ViTImageProcessor(image_mean=[0.5] * 3, image_std=[0.25] * 3).save_pretrained(path)
    return path


def test_the_hf_backbone_is_frozen_or_trains_only_its_last_layer_and_norm(tmp_path):
    path = _tiny_hf_model(tmp_path / "tiny_vit")
    frozen = E.Eyes(_config(backbone="hf", hf_name=str(path)))
    backbone = frozen.backbone
    assert not any(p.requires_grad for p in backbone.model.parameters())
    assert all(p.requires_grad for p in backbone.proj.parameters())  # the projection to the eyes' width is trained
    assert backbone.mean.flatten().tolist() == [0.5] * 3 and backbone.std.flatten().tolist() == [0.25] * 3
    assert tuple(backbone(torch.rand(5, 3, 32, 32)).shape) == (5, 16, 32)  # 4 x 4 patches, no class token
    images = torch.rand(2, 2, 3, 32, 32)
    assert tuple(frozen(images)["exists"].shape) == (2, 3)

    last = E.Eyes(_config(backbone="hf", hf_name=str(path), hf_trainable_layers=1))
    named = dict(last.backbone.model.named_parameters())
    trained = {name for name, p in named.items() if p.requires_grad}
    assert any(n.startswith("encoder.layer.1.") for n in trained) and any(n.startswith("layernorm.") for n in trained)
    assert all(n.startswith(("encoder.layer.1.", "layernorm.")) for n in trained), trained
    last(images)["height"].sum().backward()
    assert {name for name, p in named.items() if p.grad is not None} == trained

    # a checkpoint carries the backbone's weights and its image statistics; loading builds it from its config alone
    E.save(tmp_path / "run", 1, last)
    loaded = E.load_eyes(tmp_path / "run", weights="model")
    assert torch.equal(loaded(images)["height"], last.eval()(images)["height"])


class _Recording(E.Eyes):
    """Eyes that keep the images they were last given."""

    def forward(self, images):
        self.seen = images.clone()
        return super().forward(images)


def test_see_reads_the_views_it_was_built_for_and_names_a_missing_one():
    torch.manual_seed(0)
    config = _config(frames=[-20, 0])
    model = _Recording(config).eval()
    rng = np.random.default_rng(0)
    jpegs = {key: _jpeg(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)) for key in config.keys}
    jpegs["plant_t0"] = _jpeg(np.zeros((48, 48, 3), dtype=np.uint8))  # not one of its views: ignored
    percept = model.see(jpegs)
    assert config.keys == ["front_t-20", "front_t0", "wrist_t-20", "wrist_t0"]
    for v, key in enumerate(config.keys):  # each image in its place: cameras outer, frames inner
        assert torch.equal(model.seen[0, v], torch.from_numpy(E.decode_jpeg(jpegs[key], 32))), key
    assert percept == model.decode(model(model.seen))[0]
    assert S.Scene.from_dict(percept).to_dict() == percept and len(percept["parts"]) == 3
    # an image of another size is brought to the model's
    jpegs["front_t0"] = _jpeg(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    assert len(model.see(jpegs)["parts"]) == 3
    with pytest.raises(ValueError, match="wrist_t-20"):
        model.see({k: v for k, v in jpegs.items() if k != "wrist_t-20"})


# ------------------------------------------------------------------------------------------------- batches
def test_collate_makes_images_in_0_1_and_stacks_the_targets_and_augmentation_is_mild():
    config = _config(cameras=["front", "wrist", "plant"])
    scenes = _drawn_scenes(5)
    rng = np.random.default_rng(0)
    images = torch.from_numpy(rng.integers(0, 256, (5, 3, 3, 32, 32), dtype=np.uint8))
    samples = [dict(images=images[i], scene=s, episode="ep", step=4 * i) for i, s in enumerate(scenes)]
    batch = collate(samples, config)
    assert batch["images"].dtype == torch.float32 and tuple(batch["images"].shape) == (5, 3, 3, 32, 32)
    assert batch["images"].min() >= 0 and batch["images"].max() <= 1
    assert tuple(batch["exists"].shape) == (5, 3) and tuple(batch["bearing"].shape) == (5, 3, 2)
    assert tuple(batch["flower"].shape) == (5,) and batch["step"].tolist() == [0, 4, 8, 12, 16]
    assert batch["exists"].sum(1).tolist() == [len(s["parts"]) for s in scenes]
    raw = collate(samples, config, as_float=False)  # bytes, scaled by the model on its device
    assert raw["images"].dtype == torch.uint8 and torch.equal(raw["images"].float() / 255, batch["images"])

    views = images[0].numpy()
    changed = augment(views, np.random.default_rng(1))
    assert changed.shape == views.shape and changed.dtype == np.uint8 and not np.array_equal(changed, views)
    for before, after in zip(views, changed, strict=True):
        assert abs(float(after.mean()) - float(before.mean())) < 0.2 * float(before.mean())
    flat = augment(np.full((2, 3, 32, 32), 100, dtype=np.uint8), np.random.default_rng(2))
    assert all(len(np.unique(view)) == 1 and 85 <= view.flat[0] <= 115 for view in flat)  # shifted, still flat


def test_the_reader_refuses_cameras_the_dataset_does_not_have(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"cameras": ["front", "wrist"]}))
    (tmp_path / "episodes.jsonl").write_text(json.dumps({"file": "episodes/ep_0000001.parquet", "frames": 3}) + "\n")
    assert SceneFrames(tmp_path, _config()).frames == 3
    with pytest.raises(ValueError, match="plant"):
        SceneFrames(tmp_path, _config(cameras=["front", "plant"]))


# ------------------------------------------------------------------------------------- the dataset (pyarrow)
# Two episodes as collect_scene.py records them: every 4th step, and after a success every step. Each image is one
# colour: red says the row, green the camera, blue the episode.
EPISODES = {12: [2, 6, 10, 14, 18, 19, 20, 21], 13: [1, 5, 9, 13, 17]}
CAMERA_GREEN = {"front": 0, "wrist": 100, "plant": 200}


def _frame_scene(seed: int, row: int) -> dict:
    parts = [
        {"h": 0.08, "b": 0.1 * row, "cut": False, "vis": [True, False]},
        {"h": 0.115, "b": -0.1 * row, "cut": row >= 4, "vis": [True, True]},
    ]
    if seed == 12:
        parts.append({"h": 0.15, "b": 2.0, "cut": False, "vis": [False, False]})
    return {"parts": parts, "flower": FLOWERS[row % 4], "label": "none" if seed == 13 else "yellow"}


def _write_dataset(root: Path) -> Path:
    """A perception dataset in the layout of collect_scene.py (tutorial-sized: 13 frames)."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    (root / "episodes").mkdir(parents=True)
    entries = []
    for e, (seed, steps) in enumerate(EPISODES.items()):
        columns = {"step": pa.array(steps, type=pa.int32())}
        for camera, green in CAMERA_GREEN.items():
            frames = [np.full((224, 224, 3), (10 + 20 * r, green, 60 * e), dtype=np.uint8) for r in range(len(steps))]
            columns[camera] = pa.array([_jpeg(f) for f in frames], type=pa.binary())
        scenes = [json.dumps(_frame_scene(seed, r)) for r in range(len(steps))]
        columns["scene"] = pa.array(scenes, type=pa.string())
        file = f"episodes/ep_{seed:07d}.parquet"
        pq.write_table(pa.table(columns), root / file)
        shoots = len(_frame_scene(seed, 0)["parts"])
        entries.append(dict(file=file, seed=seed, task="reason_mix", frames=len(steps), steps=steps[-1], shoots=shoots))
    (root / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    meta = dict(task="reason_mix", episodes=2, frames=13, stride=4, cameras=[*CAMERA_GREEN], image_size=224)
    meta.update(crop_zoom=2, driver="expert", random_start=True, scenario=None, seed=12)
    (root / "meta.json").write_text(json.dumps(meta))
    return root


def _what_it_shows(view: torch.Tensor) -> tuple[int, str, int]:
    """(row, camera, episode) of one of the dataset's one-colour images."""
    r, g, b = (float(c) for c in view.float().mean(dim=(1, 2)))
    camera = min(CAMERA_GREEN, key=lambda c: abs(CAMERA_GREEN[c] - g))
    return round((r - 10) / 20), camera, round(b / 60)


def _all_frames(root: Path) -> list[tuple[str, int]]:
    return [(str(root / f"episodes/ep_{seed:07d}.parquet"), step) for seed, steps in EPISODES.items() for step in steps]


def test_the_reader_gives_every_frame_with_its_images_and_truth(tmp_path):
    root = _write_dataset(tmp_path / "data")
    config = E.EyesConfig(cameras=["front", "wrist", "plant"])
    frames = SceneFrames(root, config, train=False)
    assert frames.frames == 13 and len(frames.episodes) == 2
    samples = list(frames)
    assert [(s["episode"], s["step"]) for s in samples] == _all_frames(root)  # one pass, in order
    for s in samples:
        assert s["images"].dtype == torch.uint8 and tuple(s["images"].shape) == (3, 3, 224, 224)
        seed = int(Path(s["episode"]).stem.split("_")[1])
        row = EPISODES[seed].index(s["step"])
        assert s["scene"] == _frame_scene(seed, row)
        shown = [_what_it_shows(view) for view in s["images"]]
        assert shown == [(row, camera, list(EPISODES).index(seed)) for camera in CAMERA_GREEN]
    batch = collate(samples[:4], config)
    assert tuple(batch["images"].shape) == (4, 3, 3, 224, 224) and tuple(batch["exists"].shape) == (4, 3)

    # training: endless, shuffled across the episodes, augmentation on
    training = SceneFrames(root, config, train=True, seed=1, augment=True)
    drawn = list(itertools.islice(iter(training), 40))
    assert len(drawn) == 40 and all(tuple(s["images"].shape) == (3, 3, 224, 224) for s in drawn)
    first = [(s["episode"], s["step"]) for s in drawn[:13]]
    assert first != _all_frames(root) and len({e for e, _ in first[:6]}) == 2


def _frame_names(samples: list[dict]) -> list[tuple[str, int]]:
    """A batch of plain data, which frames these are. Tensors from workers that are spawned (macOS) would go through
    torch's shared-memory manager, which outlives the test and keeps the interpreter from exiting."""
    return [(s["episode"], s["step"]) for s in samples]


def test_a_validation_pass_over_two_workers_gives_every_frame_exactly_once(tmp_path):
    root = _write_dataset(tmp_path / "data")
    frames = SceneFrames(root, E.EyesConfig(), train=False)
    loader = torch.utils.data.DataLoader(frames, batch_size=3, num_workers=2, collate_fn=_frame_names)
    seen = [frame for batch in loader for frame in batch]
    assert len(seen) == len(set(seen)) == 13 and sorted(seen) == sorted(_all_frames(root))


def test_a_frame_offset_shows_the_last_frame_at_or_before_it(tmp_path):
    root = _write_dataset(tmp_path / "data")
    config = E.EyesConfig(cameras=["front", "wrist"], frames=[-8, 0])
    kinds = set()
    for s in SceneFrames(root, config, train=False):
        steps = EPISODES[int(Path(s["episode"]).stem.split("_")[1])]
        row = steps.index(s["step"])
        earlier = [i for i, step in enumerate(steps) if step <= s["step"] - 8]
        before = earlier[-1] if earlier else 0  # nothing that early: the episode's first frame
        shown = [_what_it_shows(view)[:2] for view in s["images"]]
        assert shown == [(before, "front"), (row, "front"), (before, "wrist"), (row, "wrist")], (s["step"], shown)
        kinds.add("first" if not earlier else "exact" if steps[before] == s["step"] - 8 else "between")
    assert kinds == {"first", "exact", "between"}


def test_the_trainer_runs_writes_its_files_and_ends_with_the_sentence_the_scripts_wait_for(tmp_path, capsys):
    from sim.pickup import train_eyes
    from sim.pickup.runstate import PROGRESS, SAMPLES

    root = _write_dataset(tmp_path / "data")
    out = tmp_path / "run"
    argv = ["--data", str(root), "--val", str(root), "--out", str(out), "--samples", "64", "--batch", "16"]
    argv += ["--workers", "0", "--checkpoints", "2", "--device", "cpu", "--log_every", "1"]
    train_eyes.main(argv)
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1] == "Model has seen the desired number of samples"
    progress = [line for line in lines if line.startswith("Train Checkpoint: ")]
    assert len(progress) == 4 and all(PROGRESS.search(line) for line in progress)  # the run monitors read them
    assert [SAMPLES.search(line).groups() for line in lines if SAMPLES.search(line)] == [("0", "64"), ("32", "64")]

    assert E.EyesConfig.from_json((out / "eyes_config.json").read_text()) == E.EyesConfig()
    assert json.loads((out / "args.json").read_text())["samples"] == 64
    assert list(E.checkpoints(out)) == [1, 2]
    state = torch.load(out / "checkpoints" / "checkpoint_2.pt", map_location="cpu", weights_only=True)
    assert state["step"] == 4 and state["samples"] == 64 and {"model", "ema", "optimizer"} <= set(state)
    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert [r["checkpoint"] for r in records] == [1, 2] and [r["samples"] for r in records] == [32, 64]
    for r in records:
        assert r["model"]["frames"] == r["ema"]["frames"] == 13 and r["ema"]["frames_3"] == 8
        assert 0 <= r["ema"]["count_acc"] <= 1

    eyes = E.load_eyes(out)
    jpeg = _jpeg(np.full((224, 224, 3), 128, dtype=np.uint8))
    assert len(eyes.see({"front_t0": jpeg, "wrist_t0": jpeg})["parts"]) == 3

    # a finished run that is resumed has nothing left to do; one that is started again is not overwritten
    train_eyes.main([*argv, "--resume"])
    assert capsys.readouterr().out.strip().splitlines()[-1] == "Model has seen the desired number of samples"
    assert list(E.checkpoints(out)) == [1, 2] and len((out / "metrics.jsonl").read_text().splitlines()) == 2
    with pytest.raises(SystemExit):
        train_eyes.main(argv)
    # resumed from an earlier checkpoint, a run goes on from there and measures the later checkpoints again, once
    (out / "checkpoints" / "checkpoint_2.pt").unlink()
    train_eyes.main([*argv, "--resume"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1] == "Model has seen the desired number of samples"
    assert any(line.startswith("resumed from checkpoint 1: step 2") for line in lines)
    assert list(E.checkpoints(out)) == [1, 2]
    assert [json.loads(line)["checkpoint"] for line in (out / "metrics.jsonl").read_text().splitlines()] == [1, 2]
