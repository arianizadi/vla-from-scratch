"""Render captioned images of the pickup scene for VLM pretraining (stage 2 of the from-scratch path).

Each sample is one camera image plus a templated caption that names the visible objects and where
they are in the image, so a VLM trained on it has to tie the object words used in the task
instructions ("red cube", "green can", ...) to image regions. Scenes use seeds from 2_000_000, which
overlap neither the demonstration seeds (0..) nor the evaluation seeds (1_000_000..). The arm is
advanced a random number of expert steps so the images cover the poses seen during a pick.

Output is the WebDataset layout Foundry's `image_caption` pipeline reads: <key>.jpg + <key>.txt in tar
shards and a manifest.jsonl.

    python -m sim.pickup.caption_data --out data/custom/sim_captions --scenes 20000 --workers 16
"""

import argparse
import io
import json
import multiprocessing as mp
import os
import tarfile

import numpy as np
from PIL import Image

from .env import OBJECTS, PickupEnv
from .expert import ScriptedExpert

SEED0 = 2_000_000
SIDES = ("left side", "middle", "right side")


def project(env, camera, pos):
    """Normalised image coordinates (u right, v down, both in [-1, 1] when in view) and depth."""
    cam = env.model.camera(camera).id
    rel = env.data.cam_xmat[cam].reshape(3, 3).T @ (pos - env.data.cam_xpos[cam])
    depth = -rel[2]
    f = 1.0 / np.tan(np.deg2rad(env.model.cam_fovy[cam]) / 2)
    return f * rel[0] / max(depth, 1e-6), -f * rel[1] / max(depth, 1e-6), depth


def join(names):
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def caption(env, camera, rng):
    seen = []
    for name in env.active:
        u, v, depth = project(env, camera, env.object_pos(name))
        if depth > 0 and abs(u) < 0.92 and abs(v) < 0.92:
            seen.append((u, v, OBJECTS[name]["label"]))
    where = "robot arm over a table" if camera == "front" else "view from the robot gripper"
    if not seen:
        return f"{where}. no objects are in view."
    seen.sort()
    labels = [f"a {s[2]}" for s in seen]
    kind = rng.integers(4)
    if kind == 0 or len(seen) == 1:
        u, v, label = seen[rng.integers(len(seen))]
        side = SIDES[int(np.digitize(u, [-0.33, 0.33]))]
        height = "top" if v < -0.33 else "bottom" if v > 0.33 else "centre"
        return f"{where}. the {label} is in the {side} of the image, near the {height}."
    if kind == 1:
        return f"{where}. from left to right: {join(labels)}."
    if kind == 2:
        order = list(rng.permutation(len(labels)))
        return f"{where} with {join([labels[i] for i in order])}. the {seen[0][2]} is leftmost."
    i, j = sorted(rng.choice(len(seen), size=2, replace=False))
    a, b = (seen[i][2], seen[j][2])
    return f"{where}. the {a} is to the left of the {b}." if rng.random() < 0.5 else (
        f"{where}. the {b} is to the right of the {a}."
    )


def work(args):
    worker, seeds, out, per_shard = args
    env = PickupEnv(task="pick_target")
    expert = ScriptedExpert(env, seed=worker)
    entries, samples = [], []

    def flush():
        stem = f"shard-{worker:02d}{len(entries):04d}"
        with tarfile.open(os.path.join(out, stem + ".tar"), "w") as tar:
            for key, jpg, txt in samples:
                for name, payload in ((f"{key}.jpg", jpg), (f"{key}.txt", txt.encode())):
                    info = tarfile.TarInfo(name=name)
                    info.size = len(payload)
                    tar.addfile(info, io.BytesIO(payload))
        entries.append({"shard": stem, "num_sequences": len(samples)})
        samples.clear()

    for seed in seeds:
        rng = np.random.default_rng(seed)
        env.task = "pick_target" if rng.random() < 0.8 else "pick_cube"
        env.reset(seed=seed)
        expert.reset()
        for _ in range(int(rng.integers(0, 70)) if rng.random() < 0.7 else 0):
            _, done, _ = env.step(expert.act())
            if done:
                break
        for camera in env.camera_names:
            buf = io.BytesIO()
            Image.fromarray(env.render(camera)).save(buf, format="JPEG", quality=92)
            samples.append((f"{seed:08d}_{camera}", buf.getvalue(), caption(env, camera, rng)))
        if len(samples) >= per_shard:
            flush()
    if samples:
        flush()
    env.close()
    return entries


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--scenes", type=int, default=20000)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed0", type=int, default=SEED0, help="first scene seed (use another range for a held-out set)")
    p.add_argument("--samples_per_shard", type=int, default=1024)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    seeds = np.arange(a.seed0, a.seed0 + a.scenes)
    jobs = [(w, seeds[w :: a.workers].tolist(), a.out, a.samples_per_shard) for w in range(a.workers)]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        entries = [e for chunk in pool.map(work, jobs) for e in chunk]
    with open(os.path.join(a.out, "manifest.jsonl"), "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    print(f"{sum(e['num_sequences'] for e in entries)} image-caption pairs in {len(entries)} shards -> {a.out}")
