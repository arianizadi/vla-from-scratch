"""One trainer per run, and a clean release of the run (sim/pickup/scripts/train_resilient.sh).

The script is copied into a temporary tree next to a stand-in for train.sh, so nothing is trained.
It needs bash, setsid and GNU stat; the tests are skipped where one of them is missing.
"""

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "sim" / "pickup" / "scripts" / "train_resilient.sh"
FINISHED = "Model has seen the desired number of samples"

# What train.sh does as far as train_resilient.sh can tell: it writes <run>.train.log and exits.
STAND_IN = f"""#!/usr/bin/env bash
name=$3
echo "attempt started" > "experiments/pickup/$name.train.log"
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


def launch(tree, name, train_seconds=0):
    return subprocess.Popen(
        ["bash", str(tree / "sim" / "pickup" / "scripts" / "train_resilient.sh"), "config", "dataset", name, "100"],
        env={**os.environ, "TRAIN_SECONDS": str(train_seconds), "PICKUP_POLL_SECONDS": "1"},
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
