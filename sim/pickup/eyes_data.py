"""Reading the perception dataset for the eyes (eyes.py): camera images and the scene's truth, frame by frame, as
collect_scene.py writes them (one parquet file per episode, listed in episodes.jsonl).

SceneFrames is an iterable dataset over one or more such directories. For training it never ends: each data-loader
worker takes its own share of the episode files, goes through them in a new order on every pass, reads a file whole
and passes its rows through a shuffle buffer, so that a batch holds frames of many episodes rather than a run of
neighbouring frames of one. For validation it makes one pass in a fixed order (episodes.jsonl's), the workers splitting
the files between them, so that every frame comes exactly once.

A sample is what the eyes look at, uint8 images [V, 3, H, W] (cameras outer, frame offsets inner, as
EyesConfig.views), and the truth, the scene dict of scene.py (plus the episode file and the step, to tell frames
apart). A frame offset k < 0 shows a camera as it was -k control steps earlier (-20: one second): the last recorded
frame of the episode at or before that step, or the episode's first frame. Frames are recorded every few steps, so an
offset usually falls between two of them, and the one before is what a policy would have had. The JPEGs are decoded in
the data-loader workers, as samples leave the buffer. collate turns a list of samples into a batch: the images and the
targets of eyes.targets_from_scene.

Augmentation (augment=True, training only, off by default) is mild on purpose: each view's brightness and contrast
change by up to 15% and the image shifts by up to 4 pixels, which leaves the colours of the flower and the card
apart and every part where it is relative to the plant.

pyarrow is needed only to read the episode files, and is imported there.
"""

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .eyes import EyesConfig, decode_jpeg, targets_from_scene

SHUFFLE_BUFFER = 3000  # rows per worker; a row holds its JPEGs (about 50 kB), not decoded images
BRIGHTNESS = 0.15  # augmentation: a view's brightness is scaled by a factor in 1 +- this
CONTRAST = 0.15  # and its contrast about its mean
SHIFT = 4  # pixels, in each direction


def read_episode(path: Path, columns: list[str]) -> dict[str, list]:
    """The columns of one episode file, as lists (JPEGs as bytes, the scene as JSON text)."""
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=columns)
    return {c: table.column(c).to_pylist() for c in columns}


def augment(images: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mild changes to each view of uint8 [V, 3, H, W]: brightness, contrast, a shift of a few pixels (the edge
    pixels repeated where the image moved away)."""
    out = np.empty_like(images)
    height, width = images.shape[-2:]
    for v, view in enumerate(images):
        x = view.astype(np.float32)
        mean = x.mean()
        x = (x - mean) * rng.uniform(1 - CONTRAST, 1 + CONTRAST) + mean * rng.uniform(1 - BRIGHTNESS, 1 + BRIGHTNESS)
        dy, dx = (SHIFT + int(d) for d in rng.integers(-SHIFT, SHIFT + 1, size=2))
        x = np.pad(x, ((0, 0), (SHIFT, SHIFT), (SHIFT, SHIFT)), mode="edge")[:, dy : dy + height, dx : dx + width]
        out[v] = np.clip(np.rint(x), 0, 255).astype(np.uint8)
    return out


def _shuffled(items: Iterator, size: int, rng: np.random.Generator) -> Iterator:
    """`items` through a shuffle buffer of `size`: each item read takes the place of a random one, which goes out."""
    buffer = []
    for item in items:
        if len(buffer) < size:
            buffer.append(item)
            continue
        i = int(rng.integers(size))
        out, buffer[i] = buffer[i], item
        yield out
    for i in rng.permutation(len(buffer)):  # a stream that ends (training streams do not)
        yield buffer[i]


class SceneFrames(IterableDataset):
    """Frames of one or more perception datasets (collect_scene.py) as samples for the eyes of `config`.

    roots: dataset directories. train: endless and shuffled (see the module's docstring), else one pass in order.
    seed: the order of the files and the buffer (with the worker's number). augment: training only."""

    def __init__(
        self,
        roots: str | Path | Sequence[str | Path],
        config: EyesConfig,
        train: bool = True,
        seed: int = 0,
        shuffle_buffer: int = SHUFFLE_BUFFER,
        augment: bool = False,
    ):
        super().__init__()
        self.config, self.train, self.seed = config, train, seed
        self.shuffle_buffer, self.augment = shuffle_buffer, augment and train
        roots = [roots] if isinstance(roots, (str, Path)) else list(roots)
        self.episodes: list[tuple[str, int | None]] = []  # (file, frames), in episodes.jsonl's order, root by root
        for root in map(Path, roots):
            meta = root / "meta.json"
            if meta.exists():
                cameras = json.loads(meta.read_text()).get("cameras", [])
                missing = [c for c in config.cameras if c not in cameras]
                if missing:
                    raise ValueError(f"{root} has no images of {missing} (its cameras: {cameras})")
            listing = root / "episodes.jsonl"
            if listing.exists():
                entries = [json.loads(line) for line in listing.read_text().splitlines() if line.strip()]
                self.episodes += [(str(root / e["file"]), e.get("frames")) for e in entries]
            else:  # a directory without its listing: every episode file in it, by name
                self.episodes += [(str(p), None) for p in sorted((root / "episodes").glob("*.parquet"))]
        if not self.episodes:
            raise ValueError(f"no episodes in {[str(r) for r in roots]}")

    @property
    def frames(self) -> int | None:
        """How many frames the datasets hold (from episodes.jsonl), or None where that is not listed."""
        counts = [n for _, n in self.episodes]
        return None if None in counts else sum(counts)

    def __iter__(self) -> Iterator[dict]:
        info = get_worker_info()
        return self.iterate(info.id, info.num_workers) if info is not None else self.iterate(0, 1)

    def iterate(self, worker: int = 0, workers: int = 1) -> Iterator[dict]:
        """The samples of data-loader worker `worker` of `workers`."""
        if not self.train:  # every file once, by one worker, in order
            for path, _ in self.episodes[worker::workers]:
                for item in self._items(path):
                    yield self._sample(item, None)
            return
        share = self.episodes[worker::workers] if len(self.episodes) >= workers else self.episodes
        rng = np.random.default_rng([self.seed, worker])
        rows = [n for _, n in share]
        size = self.shuffle_buffer if None in rows else min(self.shuffle_buffer, sum(rows))
        for item in _shuffled(self._passes([path for path, _ in share], rng), max(size, 1), rng):
            yield self._sample(item, rng)

    def _passes(self, paths: list[str], rng: np.random.Generator) -> Iterator[tuple]:
        """The rows of `paths`, file by file in a new order on every pass, for ever."""
        while True:
            rows = 0
            for i in rng.permutation(len(paths)):
                for item in self._items(paths[i]):
                    rows += 1
                    yield item
            if rows == 0:
                raise ValueError(f"the episode files {paths[:3]}... hold no frames")

    def _items(self, path: str) -> Iterator[tuple]:
        """The rows of one episode file as (file, step, the JPEGs of every view, the scene's JSON)."""
        cameras, offsets = self.config.cameras, self.config.frames
        data = read_episode(Path(path), ["step", *cameras, "scene"])
        steps = np.asarray(data["step"], dtype=np.int64)
        if np.any(np.diff(steps) < 0):
            raise ValueError(f"{path}: the steps are not in order")
        rows = np.arange(len(steps))
        shown = {}  # for each offset, the row each row shows: itself for 0, else the last one at or before step + k
        for k in offsets:
            before = np.searchsorted(steps, steps + k, side="right") - 1
            shown[k] = rows if k == 0 else np.maximum(before, 0)  # nothing recorded that early: the first row
        for r in range(len(steps)):
            jpegs = tuple(data[c][shown[k][r]] for c in cameras for k in offsets)
            yield path, int(steps[r]), jpegs, data["scene"][r]

    def _sample(self, item: tuple, rng: np.random.Generator | None) -> dict:
        path, step, jpegs, scene = item
        images = np.stack([decode_jpeg(j, self.config.image_size) for j in jpegs])
        if self.augment and rng is not None:
            images = augment(images, rng)
        return {"images": torch.from_numpy(images), "scene": json.loads(scene), "episode": path, "step": step}


def collate(samples: list[dict], config: EyesConfig, as_float: bool = True) -> dict:
    """A batch from samples: "images" [B, V, 3, H, W], float in 0..1 (as_float False: uint8, which Eyes.forward
    scales on its device, a quarter of the bytes to move), the targets of eyes.targets_from_scene stacked, and "step"
    and "episode" when the samples carry them."""
    images = torch.stack([s["images"] for s in samples])
    batch = {"images": images.float().div_(255) if as_float else images}
    targets = [targets_from_scene(s["scene"], config) for s in samples]
    batch.update({key: torch.stack([t[key] for t in targets]) for key in targets[0]})
    if all("step" in s for s in samples):
        batch["step"] = torch.tensor([s["step"] for s in samples])
        batch["episode"] = [s["episode"] for s in samples]
    return batch
