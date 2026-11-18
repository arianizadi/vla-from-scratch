"""How well trained eyes (eyes.py) read the attributes of architecture v3, per validation set and by drawn size (E16 of
tutorial/architecture-v3.md):

    python -m sim.pickup.eyes_report --run experiments/pickup/<eyes run> \\
        --val data/scene/scene_val_tend_ties data/scene/scene_val_tend_buds ... --out <report.json>

For every validation set (a perception dataset of collect_scene.py) the report has eyes.evaluate's metrics (v2's
outputs and every attribute's), and, per attribute, the same counts by the scale its look was drawn at where the
dataset was collected with `+sized` (scenarios.SIZES; the variation's `sizes` list each shoot's scale, lowest shoot
first, the order of the truth's parts and of the eyes' slots; `+size<x>` draws every look at x). A size is given
against the thing's true size (true_scale: ties are drawn at TIE_SCALE) and in pixels, as the look's nominal size in
the front camera's 224-pixel image at that scale (NOMINAL_PIXELS: the design's table, a calculation from the camera and
the meshes, not a measurement of each frame). Where the dataset was collected with --attribute_views, the parts are
also tallied by the camera that can read them: those the eyes' own cameras can read (own_cameras: the probe of eyes
that look through one camera), and those the front camera alone, the wrist camera alone, or both can read.

Per part, the counts are those of eyes.evaluate: where the truth says the attribute can be read, the share read right
(acc; "cannot be read" is wrong there) and the share called unreadable (missed); where it cannot be read, the share
called unreadable (unreadable_acc); and the readings the scene memory counts (readable and at least eyes.P_MIN sure),
with the share of them that are wrong. Rows are also split by whether the part is attached or cut, and where sizes
were drawn, the parts whose look is at its own size are tallied apart (own_size).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .eyes import ATTRIBUTE_CLASSES, NO_CLASS, P_MIN, evaluate, load_eyes
from .eyes_data import SceneFrames, collate
from .scenarios import TIE_SCALE

# The look's size in the front camera's 224-pixel image at the scale it is drawn at (tutorial/architecture-v3.md, fact
# 3): a tie's band, drawn at TIE_SCALE times its modelled size, about 6 pixels; a bud about 5 across seen from the side
NOMINAL_PIXELS = {"tie": 6.0, "bud": 5.0}
# The scale it is drawn at, against the thing's true (modelled) size
DRAWN_AT = {"tie": TIE_SCALE, "bud": 1.0}
# Which of the truth's cameras (attribute_views: front, wrist) each of the eyes' views is
CAMERA_OF = {"front": 0, "plant": 0, "wrist": 1}


def _sizes(root: Path) -> dict[str, list[float]]:
    """episode file -> the drawn scale of each shoot's look, lowest shoot first, for episodes that record them."""
    out = {}
    listing = root / "episodes.jsonl"
    for line in listing.read_text().splitlines() if listing.exists() else []:
        if not line.strip():
            continue
        entry = json.loads(line)
        for variation in (entry.get("scenario") or {}).values():
            if isinstance(variation, dict) and "sizes" in variation:
                out[str(root / entry["file"])] = [float(s) for s in variation["sizes"]]
    return out


@torch.no_grad()
def part_rows(model, batches, sizes: dict[str, list[float]]) -> list[dict]:
    """One row per part slot that holds a part with a value for an attribute: the attribute, the truth's class and the
    class said with its probability, whether the part is cut, its look's drawn scale (None where not recorded), and
    which cameras can read it (`views`, front and wrist, where the truth has attribute_views; else None)."""
    device = next(model.parameters()).device
    rows = []
    model.eval()
    for batch in batches:
        out = model(batch["images"].to(device))
        parts = [sorted(scene.get("parts", []), key=lambda p: float(p["h"])) for scene in batch["scenes"]]
        for name in model.config.attributes:
            probs, said = (x.cpu() for x in torch.softmax(out[name].float(), -1).max(-1))
            truth = batch[name]
            for i in range(truth.shape[0]):
                scales = sizes.get(batch["episode"][i])
                for k in range(truth.shape[1]):
                    if int(truth[i, k]) == NO_CLASS:
                        continue
                    rows.append(
                        dict(
                            attribute=name,
                            truth=int(truth[i, k]),
                            said=int(said[i, k]),
                            p=float(probs[i, k]),
                            cut=bool(batch["cut"][i, k] > 0.5),
                            size=None if scales is None or k >= len(scales) else scales[k],
                            views=parts[i][k].get("attr_vis", {}).get(name),
                        )
                    )
    return rows


def tally(rows: list[dict]) -> dict:
    """The counts of eyes.evaluate's attribute metrics over rows of one attribute."""
    readable = [r for r in rows if r["truth"] != 0]
    unreadable = [r for r in rows if r["truth"] == 0]
    counted = [r for r in readable if r["said"] != 0 and r["p"] >= P_MIN]

    def share(part, whole):
        return None if not whole else len(part) / len(whole)

    out = dict(
        readable=len(readable),
        unreadable=len(unreadable),
        acc=share([r for r in readable if r["said"] == r["truth"]], readable),
        missed=share([r for r in readable if r["said"] == 0], readable),
        unreadable_acc=share([r for r in unreadable if r["said"] == 0], unreadable),
        counted=share(counted, readable),
        counted_wrong=share([r for r in counted if r["said"] != r["truth"]], counted),
        false_read_counted=share([r for r in unreadable if r["said"] != 0 and r["p"] >= P_MIN], unreadable),
    )
    if rows:
        name = rows[0]["attribute"]
        classes = ATTRIBUTE_CLASSES[name]
        for c, value in enumerate(classes[1:], start=1):
            of = [r for r in readable if r["truth"] == c]
            out[f"acc_{value}"] = share([r for r in of if r["said"] == c], of)
        if name == "tie":  # class 1 is "none": whether there is a tie, and which colour where there is one
            out["presence_acc"] = share(
                [r for r in readable if r["said"] != 0 and (r["said"] == 1) == (r["truth"] == 1)], readable
            )
            tied = [r for r in readable if r["truth"] >= 2]
            out["colour_acc"] = share([r for r in tied if r["said"] == r["truth"]], tied)
            out["tied"] = len(tied)
    return out


def report(model, root: Path, batch: int = 128, workers: int = 8) -> dict:
    """The metrics of one validation set, and per attribute its rows tallied in all, attached and cut, and by size."""
    config = model.config
    data = SceneFrames(root, config, train=False)

    def batches_of(samples):
        out = collate(samples, config, as_float=False)
        out["scenes"] = [s["scene"] for s in samples]
        return out

    loader = DataLoader(data, batch_size=batch, num_workers=min(workers, len(data.episodes)), collate_fn=batches_of)
    metrics = evaluate(model, loader)
    rows = part_rows(model, loader, _sizes(root))
    own = sorted({CAMERA_OF[c] for c in config.cameras if c in CAMERA_OF})  # the truth's cameras these eyes look by
    attributes = {}
    for name in config.attributes:
        mine = [r for r in rows if r["attribute"] == name]
        attached = [r for r in mine if not r["cut"]]
        entry = dict(all=tally(mine), attached=tally(attached))
        if any(r["views"] is not None for r in attached):
            # where the truth says which camera can read it: the parts these eyes' own cameras can read, and the parts
            # read by the front camera only, by the wrist camera only, and by both
            def by(test, rows=attached):
                return tally([r for r in rows if r["views"] is not None and r["truth"] != 0 and test(r["views"])])

            entry["own_cameras"] = by(lambda v: any(v[j] for j in own))
            entry["front_only"] = by(lambda v: v[0] and not v[1])
            entry["wrist_only"] = by(lambda v: v[1] and not v[0])
            entry["both"] = by(lambda v: v[0] and v[1])
        by_size = defaultdict(list)
        for r in attached:
            if r["size"] is not None:
                by_size[round(r["size"], 4)].append(r)
        if by_size:
            # the look at its own size, cut parts included: what the eyes are shown outside these frames
            entry["own_size"] = tally([r for r in mine if r["size"] is None or r["size"] == 1.0])
            entry["by_size"] = {}
            for size, group in sorted(by_size.items()):
                seen = [r for r in group if r["views"] is not None and any(r["views"][j] for j in own)]
                entry["by_size"][str(size)] = dict(
                    tally(group),
                    true_scale=round(size * DRAWN_AT.get(name, 1.0), 3),
                    pixels=round(NOMINAL_PIXELS.get(name, 0.0) * size, 1),
                    **({"own_cameras": tally(seen)} if seen else {}),
                )
        attributes[name] = entry
    return dict(dataset=str(root), metrics=metrics, attributes=attributes)


def main(argv=None):
    ap = argparse.ArgumentParser(description="the eyes' attributes per validation set and by drawn size")
    ap.add_argument("--run", required=True, help="a train_eyes.py run directory")
    ap.add_argument("--val", nargs="+", required=True, help="validation perception datasets, each reported apart")
    ap.add_argument("--out", required=True, help="the report, JSON")
    ap.add_argument("--checkpoint", type=int, default=None, help="default: the latest")
    ap.add_argument("--weights", choices=("ema", "model"), default="ema")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    model = load_eyes(args.run, args.checkpoint, args.weights, args.device)
    out = dict(run=args.run, checkpoint=args.checkpoint, weights=args.weights, sets=[])
    for root in args.val:
        entry = report(model, Path(root), args.batch, args.workers)
        out["sets"].append(entry)
        line = ", ".join(
            f"{name} {100 * (e['all']['acc'] or 0):.1f}% (unreadable {100 * (e['all']['unreadable_acc'] or 0):.1f}%)"
            for name, e in entry["attributes"].items()
        )
        print(f"{root}: {line}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
