"""Give the first moments of every episode more weight in a preprocessed dataset.

Which object to go for is decided in the first frames of a demonstration. After that the arm is
already on its way and its own position gives the answer away, so only a small share of the training
windows needs the instruction at all. This tool builds a copy of a dataset in which the windows
anchored at the first frames appear several times:

    python -m sim.pickup.boost_early_windows --shards data/preprocessed/pick_target_1500/shards \\
        --out data/preprocessed/pick_target_1500_early/shards --max_frame 7 --repeat 5

The new directory links to the original shards and adds shards that hold the repeated windows; its
manifest lists both. Statistics and preprocessing metadata are copied unchanged, so the result is
used like any other dataset.
"""

import argparse
import io
import json
import os
import random
import shutil
import tarfile
from pathlib import Path


def early_samples(shard_path, max_frame):
    """Samples (lists of (suffix, bytes)) whose anchor frame is at most `max_frame`."""
    out, current, key = [], [], None

    def flush():
        meta = next((data for suffix, data in current if suffix == "metadata.json"), None)
        if meta is not None and int(json.loads(meta)["frame_index"]) <= max_frame:
            out.append(list(current))

    with tarfile.open(shard_path) as tar:
        for member in tar:
            k, suffix = member.name.split(".", 1)
            if key is not None and k != key:
                flush()
                current.clear()
            key = k
            current.append((suffix, tar.extractfile(member).read()))
        if current:
            flush()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", required=True, help="shards directory of an existing dataset")
    p.add_argument("--out", required=True, help="shards directory of the new dataset")
    p.add_argument("--max_frame", type=int, default=7, help="windows anchored at frames 0..max_frame are repeated")
    p.add_argument("--repeat", type=int, default=5, help="extra copies of each of those windows")
    p.add_argument("--samples_per_shard", type=int, default=256)
    args = p.parse_args()

    src, out = Path(args.shards).resolve(), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    with open(src / "manifest.jsonl") as f:
        manifest = [json.loads(line) for line in f if line.strip()]

    selected = []
    for entry in manifest:
        os.symlink(src / f"{entry['shard']}.tar", out / f"{entry['shard']}.tar")
        selected.extend(early_samples(src / f"{entry['shard']}.tar", args.max_frame))

    copies = [(sample, r) for sample in selected for r in range(args.repeat)]
    random.Random(0).shuffle(copies)
    extra = []
    for shard, start in enumerate(range(0, len(copies), args.samples_per_shard)):
        stem = f"early_{shard:06d}"
        batch = copies[start : start + args.samples_per_shard]
        with tarfile.open(out / f"{stem}.tar", "w") as tar:
            for i, (sample, r) in enumerate(batch):
                key = f"early-{start + i:08d}-r{r}"  # unique: the same window appears several times
                for suffix, data in sample:
                    info = tarfile.TarInfo(name=f"{key}.{suffix}")
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
        extra.append({"shard": stem, "num_sequences": len(batch)})

    with open(out / "manifest.jsonl", "w") as f:
        f.writelines(json.dumps(entry) + "\n" for entry in manifest + extra)
    for name in ("stats.json", "preprocessing_config.yaml", "processing_metadata.json"):
        if (src / name).exists():
            shutil.copy(src / name, out / name)

    total = sum(e["num_sequences"] for e in manifest)
    print(
        f"{len(selected)} of {total} windows are anchored at frames 0..{args.max_frame}; with {args.repeat} extra "
        f"copies each they are {(len(selected) * (args.repeat + 1)) / (total + len(copies)):.0%} of the new dataset "
        f"({total + len(copies)} windows) -> {out}"
    )


if __name__ == "__main__":
    main()
