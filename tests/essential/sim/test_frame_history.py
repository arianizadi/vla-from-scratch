"""Policies that look back further than one step (configs/dp_vit_lang_frames_1s.yaml): the frames eval.py sends to
the policy server must be the ones the training windows hold for the same step.

The training side is the preprocessing converter itself (LeRobotConverter.extract_sample_data, which picks each
window's frames), run on a synthetic episode whose every frame records its own step; the serving side is
eval.py's FrameHistory and infer_request, given the same episode one observation at a time. Neither is restated
here, so a mistake on one side shows up as a mismatch instead of being shared by both.

Needs MuJoCo (eval.py imports the simulator) and ray (the converter module imports it); skipped without them.
"""

import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:  # sim/ is not an installed package
    sys.path.insert(0, str(REPOSITORY))

pytest.importorskip("mujoco")
pytest.importorskip("ray")
Image = pytest.importorskip("PIL.Image")

from sim.pickup.env import PickupEnv  # noqa: E402
from sim.pickup.eval import FrameHistory, image_offsets, infer_request  # noqa: E402
from vla_foundry.data.preprocessing.robotics.converters.lerobot import LeRobotConverter  # noqa: E402
from vla_foundry.data.preprocessing.robotics.preprocess_masks import PaddingStrategy  # noqa: E402

CAMERAS = tuple(PickupEnv.camera_names)
LENGTH = 60  # control steps in the synthetic episode


def _shade(step, camera):
    """A grey level that tells the step and the camera apart after JPEG coding."""
    return (4 * step + (0 if camera == CAMERAS[0] else 2)) % 256


def _observation(step):
    obs = {cam: np.full((32, 32, 3), _shade(step, cam), dtype=np.uint8) for cam in CAMERAS}
    obs["state"] = np.full(8, float(step), dtype=np.float32)
    return obs


def _step_of(jpeg, camera):
    level = float(np.asarray(Image.open(io.BytesIO(jpeg))).mean())
    steps = [s for s in range(LENGTH) if abs(_shade(s, camera) - level) < 1.5]
    assert len(steps) == 1, (level, steps)
    return steps[0]


def _converter(image_indices):
    """The LeRobot converter with only what extract_sample_data reads (no dataset on disk, no ray actors)."""
    converter = LeRobotConverter.__new__(LeRobotConverter)
    converter.cfg = SimpleNamespace(
        past_lowdim_steps=1,
        future_lowdim_steps=14,
        max_padding_left=1,
        max_padding_right=14,
        filter_still_samples=False,
        image_indices=list(image_indices),
    )
    converter.image_indices = sorted(image_indices)
    converter.pad_fn = PaddingStrategy.get_pad_fn("copy")
    converter.entries = {0: {"tasks": ["prune the plant"]}}
    return converter


class _Counter:
    def __getattr__(self, name):
        return SimpleNamespace(remote=lambda *args, **kwargs: None)


def _training_frames(converter, anchor):
    """{image name: step of the frame} in the training window anchored at `anchor`."""
    camera_data = {cam: list(range(LENGTH)) for cam in CAMERAS}  # each "frame" is its own step
    lowdim = {"state": np.zeros((LENGTH, 8)), "action": np.zeros((LENGTH, 8))}
    images, *_ = converter.extract_sample_data(
        anchor, "episode", LENGTH, camera_data, lowdim, None, None, {"episode_index": np.zeros(LENGTH, int)}, None,
        _Counter(),
    )  # fmt: skip
    return images


@pytest.mark.parametrize("image_indices", [[-20, 0], [-1, 0], [0]])
def test_served_frames_match_training_windows(image_indices):
    converter = _converter(image_indices)
    names = [f"{cam}_t{k}" for k in image_indices for cam in CAMERAS]  # as RoboticsDataParams names them
    history = FrameHistory(_observation(0), image_offsets(names))
    for step in range(LENGTH):
        if step > 0:
            history.push(_observation(step))
        obs = history.at(0)
        request = infer_request(history.at(-1), obs, np.zeros(8), "prune the plant", 10, history=history)
        trained = _training_frames(converter, step)
        assert set(names) == set(trained) <= set(request["images"])
        for name, frame_step in trained.items():
            camera = name.rsplit("_t", 1)[0]
            assert _step_of(request["images"][name], camera) == frame_step, (step, name)
        # joint states at t-1 and t, as the training window's first two state rows (padded by copying at the start)
        assert request["state"] == [[float(max(step - 1, 0))] * 8, [float(step)] * 8]


def test_without_history_the_request_is_unchanged():
    """Callers that pass only the previous and the current observation (collect.py) still send t-1 and t."""
    prev, obs = _observation(3), _observation(4)
    request = infer_request(prev, obs, np.zeros(8), "prune the plant", 10)
    assert sorted(request["images"]) == sorted(f"{cam}_t{k}" for cam in CAMERAS for k in (-1, 0))
    for cam in CAMERAS:
        assert _step_of(request["images"][f"{cam}_t-1"], cam) == 3
        assert _step_of(request["images"][f"{cam}_t0"], cam) == 4


def test_stale_images_show_the_given_frames_at_every_offset():
    history = FrameHistory(_observation(0), [-20, 0])
    for step in range(1, 30):
        history.push(_observation(step))
    first = _observation(0)
    request = infer_request(
        history.at(-1), history.at(0), np.zeros(8), "x", 10, images_from=(first, first), history=history
    )
    assert all(_step_of(jpeg, name.rsplit("_t", 1)[0]) == 0 for name, jpeg in request["images"].items())


def test_history_keeps_only_what_it_needs():
    history = FrameHistory(_observation(0), [-20, 0])
    for step in range(1, 100):
        history.push(_observation(step))
    assert len(history.frames) == 21
    assert history.at(-20)["state"][0] == 79 and history.at(-1)["state"][0] == 98
    with pytest.raises(ValueError):
        FrameHistory(_observation(0), [0, 1])
