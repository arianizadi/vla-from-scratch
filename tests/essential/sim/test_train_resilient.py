"""One trainer per run, and a clean release of the run (sim/pickup/scripts/train_resilient.sh).

The script is copied into a temporary tree next to a stand-in for train.sh, so nothing is trained.
It needs bash, setsid and GNU stat; the tests are skipped where one of them is missing.
"""

import os
import shutil
import signal
import subprocess
import time
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "sim" / "pickup" / "scripts" / "train_resilient.sh"
FINISHED = "Model has seen the desired number of samples"

# What train.sh does as far as train_resilient.sh can tell: it writes <run>.train.log and exits.
# It also records its arguments (the checkpoint it resumes from), and with WRITE_TICKS keeps appending to a file in the
# run's checkpoints directory for that many half seconds, as a slow checkpoint save does, without logging anything.
STAND_IN = f"""#!/usr/bin/env bash
name=$3
echo "$@" > "experiments/pickup/$name.args"
echo "attempt started" > "experiments/pickup/$name.train.log"
for i in $(seq 1 "${{WRITE_TICKS:-0}}"); do
  mkdir -p "experiments/pickup/$name/checkpoints"
  echo x >> "experiments/pickup/$name/checkpoints/checkpoint_9.pt.part"
  sleep 0.5
done
sleep "${{TRAIN_SECONDS:-0}}"
echo "{FINISHED}" >> "experiments/pickup/$name.train.log"
"""


def _gnu_stat():
    try:
        return subprocess.run(["stat", "-c", "%Y", "."], capture_output=True).returncode == 0
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not (shutil.which("bash") and shutil.which("setsid") and _gnu_stat()),
    reason="train_resilient.sh needs bash, setsid and GNU stat",
)


@pytest.fixture
def tree(tmp_path):
    scripts = tmp_path / "sim" / "pickup" / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy(SCRIPT, scripts / "train_resilient.sh")
    stand_in = scripts / "train.sh"
    stand_in.write_text(STAND_IN)
    stand_in.chmod(0o755)
    return tmp_path


def launch(tree, name, train_seconds=0, **env):
    return subprocess.Popen(
        ["bash", str(tree / "sim" / "pickup" / "scripts" / "train_resilient.sh"), "config", "dataset", name, "100"],
        env={**os.environ, "TRAIN_SECONDS": str(train_seconds), "PICKUP_POLL_SECONDS": "1", **env},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def wait_for(path, seconds=10):
    deadline = time.monotonic() + seconds
    while not path.exists():
        assert time.monotonic() < deadline, f"{path.name} did not appear"
        time.sleep(0.05)


def test_a_finished_run_releases_its_name(tree):
    run = launch(tree, "one")
    assert run.wait(timeout=30) == 0
    experiments = tree / "experiments" / "pickup"
    assert FINISHED in (experiments / "one.train.log").read_text()
    assert not (experiments / "one.pid").exists()


def test_a_second_copy_of_a_live_run_is_refused(tree):
    experiments = tree / "experiments" / "pickup"
    first = launch(tree, "one", train_seconds=4)
    wait_for(experiments / "one.train.log")
    second = launch(tree, "one")
    assert second.wait(timeout=30) == 2
    assert "already being trained" in second.stderr.read()
    # The refused copy touched nothing: the live log was not set aside and no second attempt began.
    assert not list(experiments.glob("one.train.log.part*"))
    assert first.wait(timeout=30) == 0
    assert (experiments / "one.train.log").read_text().count("attempt started") == 1
    assert not (experiments / "one.pid").exists()


def test_another_name_runs_alongside(tree):
    first = launch(tree, "one", train_seconds=3)
    wait_for(tree / "experiments" / "pickup" / "one.train.log")
    other = launch(tree, "two")
    assert other.wait(timeout=30) == 0
    assert first.wait(timeout=30) == 0


def test_a_pid_file_left_by_a_killed_copy_does_not_block(tree):
    experiments = tree / "experiments" / "pickup"
    experiments.mkdir(parents=True)
    gone = subprocess.Popen(["true"])
    gone.wait()
    (experiments / "one.pid").write_text(f"{gone.pid}\n")
    run = launch(tree, "one")
    assert run.wait(timeout=30) == 0
    assert FINISHED in (experiments / "one.train.log").read_text()


def test_stopping_marks_the_run_and_releases_its_name(tree):
    experiments = tree / "experiments" / "pickup"
    run = launch(tree, "one", train_seconds=60)
    wait_for(experiments / "one.train.log")
    run.send_signal(signal.SIGTERM)
    assert run.wait(timeout=30) == 130
    assert (experiments / "one.stopped").read_text().startswith("stopped by hand")
    assert not (experiments / "one.pid").exists()
    assert FINISHED not in (experiments / "one.train.log").read_text()


def _archive(path, cut=False):
    """A small zip archive, like the files torch.save writes; `cut`: only its first half, as a save killed midway."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data.pkl", os.urandom(4096))
    if cut:
        path.write_bytes(path.read_bytes()[: path.stat().st_size // 2])


def test_a_checkpoint_cut_short_is_set_aside_and_the_run_resumes_from_the_one_before(tree):
    experiments = tree / "experiments" / "pickup"
    checkpoints = experiments / "one" / "checkpoints"
    for n in (1, 2):
        for kind in ("checkpoint", "optimizer", "ema"):
            _archive(checkpoints / f"{kind}_{n}.pt", cut=(n, kind) == (2, "ema"))
    results = tree / "results" / "pickup" / "one" / "task"
    for n in (1, 2):
        (results / f"ckpt_{n}_raw").mkdir(parents=True)
        (results / f"ckpt_{n}_raw" / "results.json").write_text("{}")
    run = launch(tree, "one")
    assert run.wait(timeout=30) == 0
    assert (
        "--model.resume_from_checkpoint experiments/pickup/one/checkpoints/checkpoint_1.pt"
        in (experiments / "one.args").read_text()
    )
    assert sorted(p.name for p in checkpoints.glob("*.pt")) == ["checkpoint_1.pt", "ema_1.pt", "optimizer_1.pt"]
    (aside,) = (checkpoints / "incomplete").iterdir()
    assert sorted(p.name for p in aside.iterdir()) == ["checkpoint_2.pt", "ema_2.pt", "optimizer_2.pt"]
    # the evaluation of the checkpoint set aside goes with it, so that the rewritten one is evaluated again
    assert (results / "ckpt_1_raw").exists() and not (results / "ckpt_2_raw").exists()
    assert len(list(results.glob("ckpt_2_raw.incomplete_*"))) == 1
    assert "checkpoint 2 of one has a file cut short" in (experiments / "one.retries.log").read_text()


def test_a_run_without_averaged_weights_resumes_from_its_newest_checkpoint(tree):
    checkpoints = tree / "experiments" / "pickup" / "one" / "checkpoints"
    for n in (1, 2):
        _archive(checkpoints / f"checkpoint_{n}.pt")
        _archive(checkpoints / f"optimizer_{n}.pt")
    run = launch(tree, "one")
    assert run.wait(timeout=30) == 0
    assert "checkpoints/checkpoint_2.pt" in (tree / "experiments" / "pickup" / "one.args").read_text()
    assert not (checkpoints / "incomplete").exists()


def test_a_checkpoint_being_written_is_not_a_stall(tree):
    experiments = tree / "experiments" / "pickup"
    run = launch(tree, "one", WRITE_TICKS="10", PICKUP_STALL_SECONDS="2")  # 5 s of saving, no log line
    assert run.wait(timeout=60) == 0
    assert FINISHED in (experiments / "one.train.log").read_text()
    assert "stalled" not in (
        (experiments / "one.retries.log").read_text() if (experiments / "one.retries.log").exists() else ""
    )


def test_a_run_that_does_nothing_is_still_a_stall(tree):
    experiments = tree / "experiments" / "pickup"
    run = launch(tree, "one", train_seconds=8, PICKUP_STALL_SECONDS="2", PICKUP_MAX_ATTEMPTS="1")
    assert run.wait(timeout=60) == 1
    assert "attempt 1 of one stalled" in (experiments / "one.retries.log").read_text()
