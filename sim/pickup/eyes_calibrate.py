"""Temperature scaling of the eyes' attribute outputs (architecture v3, step 7).

The eyes of E16 are close to calibrated over all their readings but over-sure between 0.7 and 0.99, where the scene
memory counts a reading (a tie said with 0.7 to 0.9 is right on 68% of the parts). One temperature per attribute,
fitted on validation frames after training, divides that attribute's logits before the softmax: the class said does
not change, only how sure the eyes say they are, and so which readings reach the memory's threshold (p_min).

    python -m sim.pickup.eyes_calibrate --run <eyes run> --val <validation sets...> --out <calibration.json>

The temperature minimises the negative log-likelihood of the truth's class over every part slot with a value for the
attribute ("cannot be read" being a class like the others), searched on a grid of log-temperatures and then refined.
The file holds the temperatures and, per attribute, before and after: the NLL, the expected calibration error and the
reliability table (eyes_report.calibration), and the readings the memory would count at p_min (readable, said with at
least p_min) with the share of them wrong. eyes.load_eyes(..., calibration=<file>) and slow_server.py
--eyes_calibration apply it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .eyes import NO_CLASS, P_MIN, load_eyes
from .eyes_data import SceneFrames, collate
from .eyes_report import calibration

GRID = np.exp(np.linspace(np.log(0.5), np.log(8.0), 121))


@torch.no_grad()
def logits_of(model, roots, batch: int = 128, workers: int = 8) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """attribute -> (logits [N, classes], truth [N]) over every part slot of the validation sets with a value for it."""
    config = model.config
    device = next(model.parameters()).device
    found = {name: ([], []) for name in config.attributes}
    model.eval()
    for root in roots:
        data = SceneFrames(Path(root), config, train=False)
        loader = DataLoader(
            data,
            batch_size=batch,
            num_workers=min(workers, len(data.episodes)),
            collate_fn=lambda samples: collate(samples, config, as_float=False),
        )
        for batch_ in loader:
            out = model(batch_["images"].to(device))
            for name in config.attributes:
                truth = batch_[name].reshape(-1)
                keep = truth != NO_CLASS
                found[name][0].append(out[name].float().reshape(-1, out[name].shape[-1])[keep.to(device)].cpu())
                found[name][1].append(truth[keep])
    return {name: (torch.cat(z), torch.cat(t)) for name, (z, t) in found.items() if z}


def nll(logits: torch.Tensor, truth: torch.Tensor, temperature: float) -> float:
    return float(torch.nn.functional.cross_entropy(logits / temperature, truth))


def fit(logits: torch.Tensor, truth: torch.Tensor) -> float:
    """The temperature with the least NLL: the best of GRID, then a finer search around it."""
    best = min(GRID, key=lambda t: nll(logits, truth, t))
    fine = np.exp(np.linspace(np.log(best) - 0.05, np.log(best) + 0.05, 41))
    return float(min(fine, key=lambda t: nll(logits, truth, t)))


def summary(logits: torch.Tensor, truth: torch.Tensor, temperature: float, p_min: float = P_MIN) -> dict:
    """NLL, calibration (eyes_report.calibration over the rows) and the readings the memory counts at p_min."""
    probs, said = torch.softmax(logits / temperature, -1).max(-1)
    rows = [
        dict(truth=int(t), said=int(s), p=float(p)) for t, s, p in zip(truth.tolist(), said.tolist(), probs.tolist())
    ]
    readable = [r for r in rows if r["truth"] != 0]
    counted = [r for r in readable if r["said"] != 0 and r["p"] >= p_min]
    return dict(
        temperature=round(temperature, 4),
        nll=round(nll(logits, truth, temperature), 5),
        calibration=calibration(rows),
        counted=len(counted) / max(len(readable), 1),
        counted_wrong=sum(r["said"] != r["truth"] for r in counted) / max(len(counted), 1),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="fit one temperature per attribute of the eyes on validation frames")
    ap.add_argument("--run", required=True, help="a train_eyes.py run directory")
    ap.add_argument("--val", nargs="+", required=True, help="validation perception datasets")
    ap.add_argument("--out", required=True, help="the calibration file, JSON")
    ap.add_argument("--checkpoint", type=int, default=None)
    ap.add_argument("--weights", choices=("ema", "model"), default="ema")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    model = load_eyes(args.run, args.checkpoint, args.weights, args.device)
    found = logits_of(model, args.val, args.batch, args.workers)
    out = dict(run=args.run, checkpoint=args.checkpoint, weights=args.weights, val=args.val, temperature={}, fit={})
    for name, (logits, truth) in found.items():
        t = fit(logits, truth)
        out["temperature"][name] = round(t, 4)
        out["fit"][name] = dict(parts=len(truth), before=summary(logits, truth, 1.0), after=summary(logits, truth, t))
        b, a = out["fit"][name]["before"], out["fit"][name]["after"]
        print(
            f"{name}: T {t:.3f}, NLL {b['nll']:.4f} -> {a['nll']:.4f}, ECE {b['calibration']['ece']} -> "
            f"{a['calibration']['ece']}, counted wrong {100 * b['counted_wrong']:.2f}% -> {100 * a['counted_wrong']:.2f}% "
            f"(counted {100 * b['counted']:.1f}% -> {100 * a['counted']:.1f}%)",
            flush=True,
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
