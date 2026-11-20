"""Can the fast level's encoder see how far the jaws are off a shoot on the way in? (tutorial/lab-notebook.md, "The
cut, diagnosed", (d)). The scripted expert approaches each shoot of `prune` layouts along its line moved by a random
offset (up or down and sideways, up to --shift_mm), and where its hand passes 10, 6 and 3 cm from the cut zone the
cameras are recorded with the offset of the shoot's axis in the hand's frame. Then the frozen DINOv2-B (registers)
encoder of the DINOv2 fast level reads each image, and a ridge regression on its patch tokens, fitted on four fifths
of the episodes, predicts the offset on the rest: the error is what the encoder lets a policy know at best (a linear
read-out of its tokens).

  python -m sim.pickup.cut_probe collect --episodes 300 --out probe.npz
  python -m sim.pickup.cut_probe fit probe.npz [more.npz ...]

Views: the wrist and front cameras at 224 pixels as the fast level sees them, and the wrist camera rendered at 448
pixels with its central 224 kept (twice the pixels on what is ahead of the jaws).
"""

import argparse
import time

import numpy as np

from .cut_diag import cut_offset, zone_distance

DISTANCES_MM = (100, 60, 30)
VIEWS = ("wrist", "wrist_x2", "front")
ENCODER = "facebook/dinov2-with-registers-base"


def collect(args):
    from .env import PickupEnv
    from .expert import PRUNE_OPEN, PRUNE_SECTOR, ScriptedExpert, clear_start

    env = PickupEnv("prune")
    starter = ScriptedExpert(env, seed=0)
    images = {v: [] for v in VIEWS}
    labels, meta = [], []
    t0 = time.time()
    for ep in range(args.episodes):
        seed = args.seed_start + ep
        env.reset(seed=seed)
        obs, yaw = clear_start(env, starter, seed)
        expert = ScriptedExpert(env, seed=seed)
        expert.reset(yaw=yaw)
        rng = np.random.default_rng([seed, 9])
        shifted, crossed, done = set(), set(), False
        while not done:
            name = getattr(expert, "prune_target", None)
            live = name is not None and name not in env.cut
            in_reach = live and abs(expert._turn_needed(name)) <= PRUNE_SECTOR and abs(env.plate_speed()) < 0.05
            cutting = any(w[1] == PRUNE_OPEN for w in expert.waypoints)
            if in_reach and cutting and name not in shifted:  # move this approach's line by a random offset
                shifted.add(name)
                a, d = env.shoot_axis(name)
                u = d[:2] / np.linalg.norm(d[:2])
                side = np.array([u[0], u[1], 0.0])  # sideways for a hand that comes in square across the shoot
                shift = rng.uniform(-1, 1, 2) * args.shift_mm / 1000
                move = np.array([0.0, 0.0, shift[0]]) + side * shift[1]
                expert.waypoints = [
                    (w[0] + move, *w[1:]) if i >= expert.wp else w for i, w in enumerate(expert.waypoints)
                ]
            if live and in_reach and name in shifted and float(obs["state"][7]) > 0.3:
                dist = 1000 * zone_distance(env, name)
                for mm in DISTANCES_MM:
                    if dist < mm and (name, mm) not in crossed:
                        crossed.add((name, mm))
                        off = cut_offset(env, name)
                        images["wrist"].append(obs["wrist"])
                        images["front"].append(obs["front"])
                        big = env.render("wrist", 448)
                        images["wrist_x2"].append(big[112:336, 112:336])
                        labels.append([off["hand_x_mm"], off["hand_y_mm"], off["along_mm"]])
                        meta.append([seed, mm])
            obs, done, _ = env.step(expert.act())
        if (ep + 1) % 20 == 0:
            print(f"[cut_probe] {ep + 1}/{args.episodes} episodes, {len(labels)} frames, {time.time() - t0:.0f} s")
    np.savez_compressed(
        args.out, labels=np.array(labels), meta=np.array(meta), **{v: np.array(x) for v, x in images.items()}
    )
    print(f"[cut_probe] wrote {len(labels)} frames to {args.out}")


def features(images, batch=64):
    """Patch tokens of the frozen encoder, [N, 256, 768] for 224-pixel images (class token and registers dropped), as
    float16 on the GPU (large arrays on the host were slow to allocate on the machine used)."""
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(ENCODER, local_files_only=True).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[:, None, None]
    skip = 1 + getattr(model.config, "num_register_tokens", 0)
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = torch.from_numpy(images[i : i + batch]).cuda().permute(0, 3, 1, 2).float() / 255
            out.append(model(pixel_values=(x - mean) / std).last_hidden_state[:, skip:].half())
    return torch.cat(out)


def ridge_cv(X, Y, groups, lams=(1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """Kernel ridge on X [N, D] (a GPU tensor) for targets Y [N, k]; five folds by episode; the best lambda on the
    folds' mean. Returns the root mean square error of the held-out predictions per target, and the lambda."""
    X = X.float()
    X = (X - X.mean(0)) / (X.std(0) + 1e-3)
    K = (X @ X.T / X.shape[1]).double().cpu().numpy()
    f = np.unique(groups, return_inverse=True)[1] % 5
    best = None
    for lam in lams:
        pred = np.zeros_like(Y)
        for k in range(5):
            tr, te = f != k, f == k
            mu = Y[tr].mean(0)
            alpha = np.linalg.solve(K[np.ix_(tr, tr)] + lam * np.eye(tr.sum()), Y[tr] - mu)
            pred[te] = K[np.ix_(te, tr)] @ alpha + mu
        rmse = np.sqrt(((pred - Y) ** 2).mean(0))
        if best is None or rmse.mean() < best[0].mean():
            best = (rmse, lam)
    return best


def fit(args):
    import torch

    parts = [np.load(path) for path in args.data]
    Y = np.concatenate([p["labels"] for p in parts]).astype(np.float64)
    meta = np.concatenate([p["meta"] for p in parts])
    print(f"{len(Y)} frames", flush=True)
    feats = {v: features(np.concatenate([p[v] for p in parts])) for v in VIEWS}
    for mm in DISTANCES_MM:
        sel = meta[:, 1] == mm
        index = torch.from_numpy(np.flatnonzero(sel)).cuda()
        for name, views in (
            ("wrist", ["wrist"]),
            ("wrist x2", ["wrist_x2"]),
            ("front", ["front"]),
            ("wrist and front", ["wrist", "front"]),
        ):
            X = torch.cat([feats[v][index].flatten(1) for v in views], dim=1)
            rmse, lam = ridge_cv(X, Y[sel], meta[sel, 0])
            sd = Y[sel].std(0)
            print(
                f"  {mm:3d} mm, {name:16s}: error (sd) across the jaws {rmse[0]:.1f} ({sd[0]:.1f}) mm, along the jaws "
                f"{rmse[1]:.1f} ({sd[1]:.1f}), along the shoot {rmse[2]:.1f} ({sd[2]:.1f}); lambda {lam:g}",
                flush=True,
            )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--episodes", type=int, default=300)
    c.add_argument("--seed_start", type=int, default=500000)
    c.add_argument("--shift_mm", type=float, default=10.0)
    c.add_argument("--out", required=True)
    f = sub.add_parser("fit")
    f.add_argument("data", nargs="+")
    args = ap.parse_args()
    collect(args) if args.cmd == "collect" else fit(args)


if __name__ == "__main__":
    main()
