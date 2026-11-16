"""The tower probe of the eyes (sim/pickup/eyes.py, backbone "tower"): the eyes' decoder and heads on the frozen vision
tower of a vision-language model of the Qwen-VL kind, before its merger ("patches") or after it ("merged").

Nothing is downloaded: the model is a tiny Qwen3-VL with random weights (two vision blocks 32 wide, patches of 16 pixels
merged 2 x 2, a language model of one layer), saved into the test's directory with an image processor of its own. The
preprocessing on the device is checked against that image processor's pixel_values, the merged tokens against the
model's own get_image_features and the patch tokens against the output of its last block, normalised by the merger's
own norm. The trainer test needs pyarrow to read its dataset and is skipped where it is missing.
"""

import copy
import io
import json
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:  # sim/ is not an installed package
    sys.path.insert(0, str(REPOSITORY))

from sim.pickup import eyes as E  # noqa: E402

transformers = pytest.importorskip("transformers")

# The tiny model's image processor: a mean and a std that differ by channel, so that channels out of order would show
MEAN = [0.48, 0.46, 0.41]
STD = [0.27, 0.26, 0.28]
TRUTH = {"parts": [{"h": 0.09, "b": 0.3, "cut": False, "vis": [True, False]}], "flower": "red", "label": "none"}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> Path:
    """A tiny random Qwen3-VL saved as a Hugging Face model, with an image processor of the kind Qwen3-VL has."""
    path = tmp_path_factory.mktemp("tiny_qwen3_vl")
    torch.manual_seed(0)
    vision = dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=24)
    vision.update(patch_size=16, temporal_patch_size=2, spatial_merge_size=2, num_position_embeddings=64)
    vision.update(deepstack_visual_indexes=[0])
    rope = {"rope_type": "default", "mrope_section": [2, 2, 2], "mrope_interleaved": True}
    text = dict(vocab_size=64, hidden_size=24, intermediate_size=48, num_hidden_layers=1, num_attention_heads=2)
    text.update(num_key_value_heads=1, head_dim=12, rope_scaling=rope)
    tokens = dict(image_token_id=60, video_token_id=61, vision_start_token_id=62, vision_end_token_id=63)
    config = transformers.Qwen3VLConfig(vision_config=vision, text_config=text, **tokens)
    transformers.Qwen3VLForConditionalGeneration(config).save_pretrained(path)
    cut = dict(patch_size=16, temporal_patch_size=2, merge_size=2)
    transformers.Qwen2VLImageProcessor(image_mean=MEAN, image_std=STD, **cut).save_pretrained(path)
    return path


def _config(tiny: Path, **changes) -> E.EyesConfig:
    """Small eyes on the tiny tower: 64-pixel images (4 x 4 patches, 2 x 2 merged tokens), 32 wide."""
    small = dict(image_size=64, hidden_dim=32, inter_dim=64, n_heads=2, decoder_layers=1)
    return E.EyesConfig(**{**small, "backbone": "tower", "tower_name": str(tiny), **changes})


def _targets(n: int) -> dict:
    one = E.targets_from_scene(TRUTH, E.EyesConfig())
    return {key: torch.stack([value] * n) for key, value in one.items()}


def _pictures(n: int, size: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (n, size, size, 3), dtype=np.uint8)


def _as_input(pictures: np.ndarray) -> torch.Tensor:
    """uint8 [N, H, W, 3] as the eyes take images: float [N, 3, H, W] in 0..1."""
    return torch.from_numpy(pictures).permute(0, 3, 1, 2).float() / 255


def _jpeg(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


# ------------------------------------------------------------------------------------------------ config, tokens
def test_an_eyes_config_written_before_the_tower_fields_reads_as_before_and_a_tower_config_round_trips():
    old = json.loads(E.EyesConfig().to_json())
    for key in ("tower_name", "tower_tokens", "tower_trainable_layers"):
        del old[key]
    assert E.EyesConfig.from_json(json.dumps(old)) == E.EyesConfig()
    config = E.EyesConfig(backbone="tower", tower_name="a/model", tower_tokens="merged", tower_trainable_layers=2)
    assert E.EyesConfig.from_json(config.to_json()) == config
    bad = (dict(tower_name=None), dict(tower_tokens="cls"), dict(tower_trainable_layers=-1))
    for changes in bad:
        with pytest.raises(ValueError):
            E.EyesConfig(**{"backbone": "tower", "tower_name": "a/model", **changes})


@pytest.mark.parametrize("kind, count, width", [("patches", 196, 32), ("merged", 49, 24)])
def test_a_224_pixel_image_gives_196_patch_tokens_or_49_merged_ones(tiny, kind, count, width):
    torch.manual_seed(0)
    eyes = E.Eyes(_config(tiny, image_size=224, tower_tokens=kind)).eval()
    images = torch.rand(2, 2, 3, 224, 224)  # [B, V, 3, S, S]
    assert tuple(eyes.backbone.tokens(images[0]).shape) == (2, count, width)  # the tower's own width
    assert tuple(eyes.backbone(images[0]).shape) == (2, count, 32)  # projected to the eyes' width
    memory = []
    eyes.decoder.register_forward_pre_hook(lambda _, args: memory.append(args[1]))
    out = eyes(images)
    assert tuple(memory[0].shape) == (2, 2 * count, 32)  # the decoder reads [B, V * P, D]: every view's tokens
    assert tuple(out["exists"].shape) == (2, 3) and tuple(out["label"].shape) == (2, 5)


# ---------------------------------------------------------------------------------- the model's own processing
@pytest.mark.parametrize("fast", [False, True], ids=["processor", "fast-processor"])
def test_the_preprocessing_on_the_device_writes_the_image_processors_pixel_values(tiny, fast):
    if fast:
        pytest.importorskip("torchvision")
    pictures = _pictures(3, 96)  # 6 x 6 patches each
    processor = transformers.AutoImageProcessor.from_pretrained(tiny, use_fast=fast)
    whole = {"shortest_edge": 96 * 96, "longest_edge": 96 * 96}  # as qwen_reasoner.yaml keeps its images whole
    expected = processor(images=[Image.fromarray(p) for p in pictures], size=whole, return_tensors="pt")
    backbone = E.Eyes(_config(tiny, image_size=96)).backbone
    pixels, grid = backbone.pixel_values(_as_input(pictures))
    assert grid.tolist() == expected["image_grid_thw"].tolist() == [[1, 6, 6]] * 3
    assert tuple(pixels.shape) == tuple(expected["pixel_values"].shape) == (3 * 36, 3 * 2 * 16 * 16)
    assert torch.allclose(pixels, expected["pixel_values"], rtol=0, atol=1e-6)
    assert backbone.mean.flatten().tolist() == pytest.approx(MEAN)  # the processor's, channel by channel
    assert backbone.std.flatten().tolist() == pytest.approx(STD)


def test_merged_tokens_are_the_models_image_features_and_patch_tokens_its_last_blocks_output_normalised(tiny):
    model = transformers.AutoModelForImageTextToText.from_pretrained(tiny).eval()
    processor = transformers.AutoImageProcessor.from_pretrained(tiny, use_fast=False)
    pictures = _pictures(3, 64, seed=1)
    whole = {"shortest_edge": 64 * 64, "longest_edge": 64 * 64}
    inputs = processor(images=list(pictures), size=whole, return_tensors="pt")
    last = []
    model.visual.blocks[-1].register_forward_hook(lambda _, args, out: last.append(out))
    with torch.no_grad():  # one tensor per image, and the deepstack features
        merged, _ = model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
        patches = model.visual.merger.norm(last[0])  # each patch as the merger reads it, before it packs them
    expected = {"merged": torch.cat(merged), "patches": patches}
    for kind, count in (("merged", 4), ("patches", 16)):
        tokens = E.Eyes(_config(tiny, tower_tokens=kind)).backbone.tokens(_as_input(pictures))
        assert tokens.shape[:2] == (3, count)
        assert torch.allclose(tokens.flatten(0, 1), expected[kind], rtol=0, atol=1e-5), kind


# --------------------------------------------------------------------------------------------- what is trained
@pytest.mark.parametrize("trained", [0, 1], ids=["frozen", "last-block"])
@pytest.mark.parametrize("kind", E.TOWER_TOKENS)
def test_only_the_tower_layers_asked_for_get_gradients_and_the_tower_stays_in_eval_mode(tiny, kind, trained):
    torch.manual_seed(0)
    eyes = E.Eyes(_config(tiny, tower_tokens=kind, tower_trainable_layers=trained)).train()
    backbone, tower = eyes.backbone, eyes.backbone.tower
    assert eyes.training and backbone.training and not tower.training
    images = torch.rand(2, 2, 3, 64, 64)
    loss, _ = eyes.loss(eyes(images), _targets(2))
    loss.backward()
    expected = set()
    if trained:  # the last block, and the merger where its output is what the eyes read
        prefixes = ("blocks.1.", "merger.") if kind == "merged" else ("blocks.1.",)
        expected = {name for name, _ in tower.named_parameters() if name.startswith(prefixes)}
        assert any(name.startswith("blocks.1.") for name in expected)
    assert {name for name, p in tower.named_parameters() if p.grad is not None} == expected
    assert {name for name, p in tower.named_parameters() if p.requires_grad} == expected
    learning = [backbone.proj, eyes.decoder, eyes.part_head, eyes.flower_head, eyes.label_head]
    assert all(p.grad is not None for module in learning for p in module.parameters())  # the eyes' own parts learn
    # frozen, the tower runs without autograd recording anything
    assert backbone.tokens(images[0]).requires_grad == bool(trained)


def test_only_the_vision_tower_is_kept_and_the_language_model_is_let_go(tiny, monkeypatch):
    model = transformers.AutoModelForImageTextToText.from_pretrained(tiny)
    tower = sum(p.numel() for p in model.visual.parameters())
    language = sum(p.numel() for p in model.parameters()) - tower
    loaded = []
    load = transformers.AutoModelForImageTextToText.from_pretrained

    def watched(*args, **kwargs):
        whole = load(*args, **kwargs)
        loaded.append(weakref.ref(whole))
        return whole

    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", watched)
    eyes = E.Eyes(_config(tiny))
    assert len(loaded) == 1 and loaded[0]() is None  # the whole model is gone; its tower lives on in the eyes
    # the eyes' parameters: the tower's, the projection's and the decoder's and heads' as the default eyes have them
    vit = E.Eyes(_config(tiny, backbone="vit", patch_size=16, n_layers=1))
    own = sum(p.numel() for p in vit.parameters()) - sum(p.numel() for p in vit.backbone.parameters())
    projection = (32 + 1) * 32  # patch tokens 32 wide, to the eyes' 32
    assert language > 0 and sum(p.numel() for p in eyes.parameters()) == tower + projection + own
    assert not any("language_model" in name or "lm_head" in name for name, _ in eyes.named_modules())


# ------------------------------------------------------------------------------------------ checkpoints, serving
@pytest.mark.parametrize("kind, trained", [("patches", 0), ("merged", 1)])
def test_a_checkpoint_holds_only_what_is_trained_and_load_eyes_builds_the_tower_again(tiny, tmp_path, kind, trained):
    torch.manual_seed(0)
    eyes = E.Eyes(_config(tiny, tower_tokens=kind, tower_trainable_layers=trained))
    average = copy.deepcopy(eyes)
    with torch.no_grad():  # trained weights unlike the tower's own, and an average unlike them both
        for model in (eyes, average):
            for p in model.parameters():
                if p.requires_grad:
                    p.add_(0.05 * torch.randn_like(p))
    average.requires_grad_(False)  # as the running average is kept (vla_foundry/models/ema.py)
    path = E.save(tmp_path, 1, eyes, average)

    state = torch.load(path, map_location="cpu", weights_only=True)
    names = {name for name, p in eyes.named_parameters() if p.requires_grad}
    assert set(state["model"]) == set(state["ema"]) == names  # what is trained, and nothing frozen
    in_tower = {key.removeprefix("backbone.tower.") for key in state["model"] if key.startswith("backbone.tower.")}
    expected = {name for name, _ in eyes.backbone.tower.named_parameters() if name.startswith(("blocks.1.", "merger."))}
    assert in_tower == (expected if trained else set())

    images = torch.rand(2, 2, 3, 64, 64)

    def same(a, b):
        out_a, out_b = a(images), b(images)
        return all(torch.equal(out_a[k], out_b[k]) for k in out_a)

    # compared with requires_grad as the average has it: the eyes' outputs move in the 7th digit with it, whatever the
    # backbone (torch's attention takes another path)
    loaded = E.load_eyes(tmp_path).requires_grad_(False)
    assert same(loaded, average.eval()) and same(E.load_eyes(tmp_path, weights="model"), eyes.eval())
    model = transformers.AutoModelForImageTextToText.from_pretrained(tiny)
    for name, p in loaded.backbone.tower.named_parameters():  # the frozen weights: the model's own, loaded again
        if f"backbone.tower.{name}" not in names:
            assert torch.equal(p, model.visual.get_parameter(name)), name
    # only the frozen weights may be missing from a state dict
    del state["model"]["backbone.proj.weight"]
    with pytest.raises(RuntimeError, match="backbone.proj.weight"):
        loaded.load_state_dict(state["model"])
    jpeg = _jpeg(_pictures(1, 64)[0])
    assert len(loaded.see({"front_t0": jpeg, "wrist_t0": jpeg})["parts"]) == 3


# ------------------------------------------------------------------------------------------------------ errors
def test_a_model_of_another_kind_an_image_size_or_a_number_of_layers_that_does_not_fit_is_a_clear_error(tiny, tmp_path):
    vit = transformers.ViTConfig(
        image_size=32, patch_size=8, hidden_size=24, num_hidden_layers=1, num_attention_heads=2, intermediate_size=48
    )
    transformers.ViTModel(vit).save_pretrained(tmp_path / "vit")
    with pytest.raises(ValueError, match="does not load as a vision-language model") as refused:
        E.Eyes(_config(tiny, tower_name=str(tmp_path / "vit")))
    assert "Model type should be" not in str(refused.value)  # not the list of every class there is
    # a vision-language model with a tower of another kind: CLIP's, whose tokens go to the language model unmerged
    clip = transformers.CLIPVisionConfig(
        image_size=32, patch_size=8, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2
    )
    llama = transformers.LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2
    )
    llava = transformers.LlavaConfig(vision_config=clip, text_config=llama, image_token_index=63)
    transformers.LlavaForConditionalGeneration(llava).save_pretrained(tmp_path / "llava")
    with pytest.raises(ValueError, match="no vision tower of the Qwen-VL kind"):
        E.Eyes(_config(tiny, tower_name=str(tmp_path / "llava")))
    with pytest.raises(ValueError, match="multiple of 32"):
        E.Eyes(_config(tiny, image_size=48))
    with pytest.raises(ValueError, match="has 2 blocks"):
        E.Eyes(_config(tiny, tower_trainable_layers=3))


# ----------------------------------------------------------------------------------------------------- trainer
def test_the_trainer_takes_the_tower_flags(tiny):
    from sim.pickup import train_eyes

    needed = ["--data", "d", "--val", "v", "--out", "o", "--samples", "8"]

    def tower(*flags):
        args = train_eyes.parse_args([*needed, *flags])
        return args.backbone, args.tower_name, args.tower_tokens, args.tower_trainable_layers

    asked = ("--backbone", "tower", "--tower_name", str(tiny), "--tower_tokens", "merged")
    assert tower(*asked) == ("tower", str(tiny), "merged", 0)
    assert tower("--tower_trainable_layers", "2") == ("vit", None, "patches", 2)
    with pytest.raises(SystemExit):
        train_eyes.parse_args([*needed, "--tower_tokens", "cls"])


def _write_dataset(root: Path) -> Path:
    """A perception dataset in the layout of collect_scene.py, as small as it gets: one episode of four frames."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    (root / "episodes").mkdir(parents=True)
    steps = [0, 4, 8, 12]
    columns = {"step": pa.array(steps, type=pa.int32())}
    for seed, camera in enumerate(("front", "wrist")):
        columns[camera] = pa.array([_jpeg(p) for p in _pictures(len(steps), 224, seed)], type=pa.binary())
    columns["scene"] = pa.array([json.dumps(TRUTH)] * len(steps), type=pa.string())
    pq.write_table(pa.table(columns), root / "episodes" / "ep_0000001.parquet")
    entry = dict(file="episodes/ep_0000001.parquet", seed=1, task="prune", frames=4, steps=12, shoots=1)
    (root / "episodes.jsonl").write_text(json.dumps(entry) + "\n")
    meta = dict(task="prune", episodes=1, frames=4, stride=4, cameras=["front", "wrist"], image_size=224)
    (root / "meta.json").write_text(json.dumps(meta))
    return root


def test_the_trainer_trains_a_probe_whose_checkpoints_hold_no_frozen_weight(tiny, tmp_path, capsys):
    from sim.pickup import train_eyes

    root = _write_dataset(tmp_path / "data")
    out = tmp_path / "run"
    argv = ["--data", str(root), "--val", str(root), "--out", str(out), "--samples", "8", "--batch", "4"]
    argv += ["--workers", "0", "--checkpoints", "1", "--device", "cpu", "--log_every", "1", "--backbone", "tower"]
    argv += ["--tower_name", str(tiny), "--tower_tokens", "merged", "--tower_trainable_layers", "1"]
    train_eyes.main(argv)
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1] == train_eyes.FINISHED and any(line.startswith("eyes: ") for line in lines)
    config = E.EyesConfig.from_json((out / E.CONFIG_FILE).read_text())
    assert (config.backbone, config.tower_name, config.tower_tokens, config.tower_trainable_layers) == (
        "tower",
        str(tiny),
        "merged",
        1,
    )
    state = torch.load(out / "checkpoints" / "checkpoint_1.pt", map_location="cpu", weights_only=True)
    tower = {key for key in state["model"] if key.startswith("backbone.tower.")}
    assert tower and all(key.startswith(("backbone.tower.blocks.1.", "backbone.tower.merger.")) for key in tower)
    assert set(state["ema"]) == set(state["model"])
    record = json.loads((out / "metrics.jsonl").read_text().splitlines()[0])
    assert record["model"]["frames"] == record["ema"]["frames"] == 4
    # a finished run resumed loads its weights, average and optimizer, and has nothing left to do
    train_eyes.main([*argv, "--resume"])
    assert capsys.readouterr().out.strip().splitlines()[-1] == train_eyes.FINISHED
    jpeg = _jpeg(_pictures(1, 224)[0])
    assert len(E.load_eyes(out).see({"front_t0": jpeg, "wrist_t0": jpeg})["parts"]) == 3
