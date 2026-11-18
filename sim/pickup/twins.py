"""Twins: a paired re-rendered dataset (rerender.py) turned into one in which every frame carries its twin, the same
frame in another look, for the paired consistency loss (model.twin_consistency_weight, data.twin_suffix).

Each re-rendered copy of a demonstration becomes one episode: its own cameras (front, wrist) in its look, and the
source episode's cameras in the plain look as front_twin and wrist_twin; the states and actions, identical in the two
by construction, are checked and kept once. A dataset with the source included and two copies of each source episode
gives two episodes per source episode, so the plain frames appear twice and each look once.

  python -m sim.pickup.twins --src data/lerobot/prune_2k_rs_tail10_leaffix_paired \\
      --out data/lerobot/prune_2k_rs_tail10_leaffix_twins --workers 32
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAMERAS = ("front", "wrist")
SUFFIX = "_twin"


def _episode(path: Path, index: int) -> Path:
    return path / "data" / "chunk-000" / f"episode_{index:06d}.parquet"


def pair_episodes(entries: list[dict]) -> list[tuple[int, int]]:
    """(copy, source) episode indices: every re-rendered copy with the episode of its source in the plain look."""
    sources = {e["source_episode"]: e["episode_index"] for e in entries if e["look"] == "source"}
    pairs = []
    for e in entries:
        if e["look"] == "source":
            continue
        if e["source_episode"] not in sources:
            raise ValueError(f"episode {e['episode_index']}: its source episode {e['source_episode']} is not included")
        pairs.append((e["episode_index"], sources[e["source_episode"]]))
    return pairs


def twin_table(copy: pa.Table, source: pa.Table, episode_index: int, first_index: int) -> pa.Table:
    """The copy's table with the source's camera images as <camera>_twin, renumbered as episode episode_index."""
    for column in ("state", "action"):
        if not np.array_equal(np.array(copy[column].to_pylist()), np.array(source[column].to_pylist())):
            raise ValueError(f"episode {episode_index}: the copy's {column} differs from its source's")
    n = copy.num_rows
    table = copy
    for c in CAMERAS:
        table = table.append_column(c + SUFFIX, source[c])
    for name, values in (
        ("episode_index", np.full(n, episode_index, dtype=np.int64)),
        ("index", np.arange(first_index, first_index + n, dtype=np.int64)),
    ):
        table = table.set_column(table.schema.get_field_index(name), name, pa.array(values))
    return table


def _write(job):
    src, out, copy, source, episode_index, first_index = job
    table = twin_table(
        pq.read_table(_episode(src, copy)), pq.read_table(_episode(src, source)), episode_index, first_index
    )
    pq.write_table(table, _episode(out, episode_index))
    return episode_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="a dataset written by rerender.py with --include_source")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    info = json.loads((src / "meta" / "info.json").read_text())
    entries = [json.loads(line) for line in (src / "meta" / "episodes.jsonl").read_text().splitlines()]
    by_index = {e["episode_index"]: e for e in entries}
    pairs = pair_episodes(entries)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    jobs, episodes, first = [], [], 0
    for i, (copy, source) in enumerate(pairs):
        e = by_index[copy]
        jobs.append((src, out, copy, source, i, first))
        episodes.append(dict(e, episode_index=i, twin_look="source", twin_of=source, copy_of=copy))
        first += e["length"]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(_write, jobs)):
            if (k + 1) % 500 == 0:
                print(f"[twins] {k + 1}/{len(jobs)} episodes", flush=True)
    with open(out / "meta" / "episodes.jsonl", "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in episodes)
    (out / "meta" / "tasks.jsonl").write_text((src / "meta" / "tasks.jsonl").read_text())
    features = dict(info["features"])
    for c in CAMERAS:
        features[c + SUFFIX] = dict(features[c])
    meta = dict(info, features=features, total_episodes=len(episodes), total_frames=first)
    meta["twins"] = dict(source=src.name, suffix=SUFFIX, twin_look="source", episodes=len(episodes))
    (out / "meta" / "info.json").write_text(json.dumps(meta, indent=2))
    print(f"[twins] wrote {len(episodes)} episodes / {first} frames to {out}")


if __name__ == "__main__":
    main()
