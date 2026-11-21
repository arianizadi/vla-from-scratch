"""Paired re-rendering (rerender.py) of a collection with DART noise and per-frame notes: the copies keep the recorded
states, actions and notes bit for bit, and only the images change."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
pytest.importorskip("mujoco")
pq = pytest.importorskip("pyarrow.parquet")


def _run(*args):
    subprocess.run([sys.executable, "-m", *args], cwd=REPOSITORY, check=True, capture_output=True)


def _episodes(path):
    meta = [json.loads(line) for line in (path / "meta" / "episodes.jsonl").read_text().splitlines()]
    tables = [pq.read_table(path / "data" / "chunk-000" / f"episode_{e['episode_index']:06d}.parquet") for e in meta]
    return meta, tables


def test_noisy_reasoning_collection_is_redrawn_with_its_notes(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _run(
        "sim.pickup.collect",
        "--task",
        "reason_mix",
        "--episodes",
        "2",
        "--workers",
        "1",
        "--random_start",
        "--tail_steps",
        "2",
        "--seed",
        "123",
        "--action_noise",
        "0.02",
        "--action_noise_quiet_cm",
        "3",
        "--out",
        str(src),
    )
    _run(
        "sim.pickup.rerender",
        "--src",
        str(src),
        "--out",
        str(out),
        "--copies",
        "1",
        "--include_source",
        "--collect_workers",
        "1",
        "--workers",
        "1",
        "--looks",
        "lighting:dusk|noon/train",
    )
    smeta, stables = _episodes(src)
    ometa, otables = _episodes(out)
    assert len(ometa) == 2 * len(smeta)
    assert [e["look"] for e in ometa] == ["source"] * len(smeta) + ["lighting:dusk|noon/train"] * len(smeta)
    for e, table in zip(ometa[len(smeta) :], otables[len(smeta) :], strict=True):
        source = stables[e["source_episode"]]
        for name in ("state", "action", "note", "instruction_subgoal"):
            assert table[name].to_pylist() == source[name].to_pylist(), name
        assert table["front"][0].as_py()["bytes"] != source["front"][0].as_py()["bytes"]
