"""Tests for the training monitor's data layer (runstate.py) and its HTTP server (monitor_server.py).

They need the standard library and pytest only: a small fake experiments / results tree stands in
for real training logs and evaluations.
"""

import contextlib
import http.client
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
if str(REPOSITORY) not in sys.path:  # sim/ is not an installed package
    sys.path.insert(0, str(REPOSITORY))

from sim.pickup import monitor_server, runstate  # noqa: E402

SECRET = b"TOP SECRET"
VIDEO = bytes(range(256))
GPUS = [dict(index=0, name="Test GPU", util=87.0, mem_used_mb=1024.0, mem_total_mb=24564.0, temp_c=61.0)]
RUN_KEYS = {
    "name",
    "about",
    "state",
    "note",
    "samples_seen",
    "samples_total",
    "loss",
    "learning_rate",
    "samples_per_second",
    "seconds_left",
    "restarts",
    "checkpoints",
    "seconds_since_log",
    "loss_curve",
    "evaluations",
    "evaluations_running",
}
EVALUATION_KEYS = {
    "checkpoint",
    "weights",
    "task",
    "samples",
    "successes",
    "episodes",
    "success_rate",
    "ci95",
    "at_named_object",
    "median_miss_named_mm",
    "median_miss_nearest_mm",
    "within_10mm",
    "within_20mm",
    "close_slope",
    "retried_rate",
    "wrong_object_rate",
    "per_target",
}


# ---- a fake tree --------------------------------------------------------------------------------


def samples_line(seen, total):
    return f"2000-01-01,00:00:00 | INFO | Samples: {seen} / {total}"


def progress_line(checkpoint, done, of, loss, rate=100.0, lr=0.0001):
    return (
        f"2000-01-01,00:00:00 | INFO | Train Checkpoint: {checkpoint} [{done:>6}/{of} ({100 * done // of}%)] "
        f"Loss: {loss} Data (t): 0.001 Batch (t): 0.100, {rate:.2f}/s, {rate:.2f}/s/gpu LR: {lr:.6f} "
    )


FINISHED_RUN = [
    samples_line(0, 3000),
    progress_line(0, 500, 1000, "2.000"),
    progress_line(0, 1000, 1000, "1.500"),
    "Saving checkpoint_1 ...",
    samples_line(1000, 3000),
    progress_line(1, 500, 1000, "1.200"),
    progress_line(1, 1000, 1000, "1.000"),
    samples_line(2000, 3000),
    progress_line(2, 500, 1000, "0.800"),
    progress_line(2, 1000, 1000, "0.500", rate=250.0, lr=0.00005),
    "2000-01-01,00:00:00 | INFO | Model has seen the desired number of samples. Ending training.",
]
RUNNING_RUN = [samples_line(0, 4000), progress_line(0, 400, 1000, "3.000"), progress_line(0, 800, 1000, "nan")]


def results_json(successes, episodes, checkpoint, miss_mm):
    """What eval.py writes, cut down to two episodes."""
    episode = dict(
        seed=1_000_000,
        target="cube",
        instruction="pick up the red cube",
        success=True,
        steps=60,
        wrong_object=False,
        close_step=30,
        close_nearest="cube",
        close_xy_err_target=0.004,
        close_xy_err_nearest=0.004,
        close_tcp_xy=[0.5, 0.0],
        close_object_xy={"cube": [0.5, 0.004], "can": [0.6, 0.2]},
    )
    never_closed = dict(episode, seed=1_000_001, success=False, close_step=None, close_nearest=None)
    never_closed.update(close_xy_err_target=None, close_xy_err_nearest=None, close_tcp_xy=None, close_object_xy=None)
    return dict(
        task="pick_target",
        episodes=episodes,
        successes=successes,
        success_rate=successes / episodes,
        success_ci95=[0.1, 0.9],
        wrong_object_rate=0.0,
        per_target={"cube": dict(successes=successes, episodes=episodes, success_rate=successes / episodes)},
        grasp=dict(
            closed_nearest_is_target_rate=0.5,
            median_xy_err_target_mm=miss_mm,
            median_xy_err_nearest_mm=4.0,
            within_10mm_of_target_rate=0.5,
            within_20mm_of_target_rate=0.5,
        ),
        server=dict(checkpoint=checkpoint, params_m=1.0),
        per_episode=[episode, never_closed],
    )


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(runstate, "gpus", lambda: [dict(g) for g in GPUS])  # no nvidia-smi in tests
    experiments, results, ui = tmp_path / "experiments", tmp_path / "results", tmp_path / "ui"
    experiments.mkdir()
    (experiments / "alpha.train.log").write_text("\n".join(FINISHED_RUN) + "\n")
    (experiments / "beta.train.log").write_text("\n".join(RUNNING_RUN) + "\n")
    for variant, content in {
        "ckpt_1_raw": results_json(1, 4, str(experiments / "alpha" / "checkpoints" / "checkpoint_1.pt"), 12.5),
        "ckpt_2_ema": results_json(3, 4, "experiments/alpha/checkpoints/ema_2.pt", float("nan")),
    }.items():
        directory = results / "alpha" / "pick_target" / variant
        directory.mkdir(parents=True)
        (directory / "results.json").write_text(json.dumps(content))
    (results / "alpha" / "pick_target" / "ckpt_3_raw").mkdir()  # claimed, still being evaluated
    (results / "alpha" / "pick_target" / "ckpt_1_raw" / "rollout_000_cube_ok.mp4").write_bytes(VIDEO)
    (results / "alpha" / "pick_target" / "ckpt_1_raw" / ".hidden.mp4").write_bytes(SECRET)
    (ui / "sub").mkdir(parents=True)
    (ui / "index.html").write_text("<!doctype html><title>monitor</title>")
    (ui / "app.js").write_text("export const ready = true;")
    (ui / "style.css").write_text("body { margin: 0 }")
    (ui / "sub" / "index.html").write_text("<!doctype html><title>sub</title>")
    (tmp_path / "secret.txt").write_bytes(SECRET)
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "clip.mp4").write_bytes(SECRET)
    return SimpleNamespace(root=tmp_path, experiments=experiments, results=results, ui=ui)


@contextlib.contextmanager
def serving(tree, token=None):
    state = monitor_server.State(str(tree.experiments), str(tree.results), interval=0.05)
    state.start()
    server = monitor_server.MonitorServer(("127.0.0.1", 0), state, token=token, ui=str(tree.ui))
    thread = threading.Thread(target=server.serve_forever, kwargs=dict(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.close()
        thread.join(timeout=5)


@pytest.fixture
def server(tree):
    with serving(tree) as running:
        yield running


def get(server, path, headers=None, method="GET"):
    """(status, headers with lower-case names, body) of one request, sent exactly as written."""
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    try:
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        return response.status, {k.lower(): v for k, v in response.getheaders()}, response.read()
    finally:
        connection.close()


def strict_json(body):
    """Parse as a browser does: NaN and Infinity are errors."""

    def refuse(constant):
        raise ValueError(f"not JSON: {constant}")

    return json.loads(body, parse_constant=refuse)


def wait_until(condition, seconds=5.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


# ---- runstate: logs and results -----------------------------------------------------------------


def test_run_parses_a_finished_log(tree):
    run = runstate.Run(str(tree.experiments), "alpha")
    run.update()
    assert run.curve == [(500, 2.0), (1000, 1.5), (1500, 1.2), (2000, 1.0), (2500, 0.8), (3000, 0.5)]
    assert run.losses == [2.0, 1.5, 1.2, 1.0, 0.8, 0.5]
    assert (run.total, run.seen, run.loss, run.rate) == (3000, 3000, 0.5, 250.0)
    assert run.lr == pytest.approx(5e-5)
    assert run.finished and run.state == "done"
    assert run.samples_at_checkpoint() == {1: 1000, 2: 2000, 3: 3000}
    assert run.downsampled_curve() == [[s, loss] for s, loss in run.curve]


def test_run_in_progress_then_stalled_then_resumed(tree):
    log = tree.experiments / "beta.train.log"
    run = runstate.Run(str(tree.experiments), "beta")
    run.update()
    assert (run.seen, run.total, run.finished, run.state) == (800, 4000, False, "training")
    assert run.loss != run.loss  # the last logged loss was nan
    assert run.downsampled_curve() == [[400, 3.0]]  # nan is left out of the curve

    with open(log, "a") as f:  # only what was appended is read on the next update
        f.write(progress_line(0, 1000, 1000, "2.500") + "\n" + samples_line(1000, 4000) + "\n")
    run.update()
    assert len(run.curve) == 3 and run.curve[-1] == (1000, 2.5)
    assert (run.seen, run.loss) == (1000, 2.5)

    os.utime(log, (time.time() - 3600, time.time() - 3600))
    assert run.state == "stalled"
    assert runstate.Run(str(tree.experiments), "never_started").state == "waiting"


def test_a_run_can_say_what_it_is_for(tree):
    run = runstate.Run(str(tree.experiments), "alpha")
    run.update()
    assert run.about is None
    (tree.experiments / "alpha.about").write_text("4000 demonstrations, no history\nmore\n")
    assert run.about == "4000 demonstrations, no history"
    shown = runstate.snapshot([run], runstate.Evals(str(tree.results)))["runs"][0]
    assert shown["about"] == "4000 demonstrations, no history"
    (tree.experiments / "alpha.about").write_text("\n")
    assert run.about is None


def test_a_run_ended_on_purpose_is_stopped_not_stalled(tree):
    log = tree.experiments / "beta.train.log"
    marker = tree.experiments / "beta.stopped"
    run = runstate.Run(str(tree.experiments), "beta")
    run.update()
    hour_ago = time.time() - 3600
    os.utime(log, (hour_ago, hour_ago))
    assert (run.state, run.stopped_note) == ("stalled", None)

    reason = "stopped at 800 samples: its GPU was needed"
    marker.write_text(reason + "\nsecond line\n")
    assert (run.state, run.stopped_note) == ("stopped", reason)
    shown = runstate.snapshot([run], runstate.Evals(str(tree.results)))["runs"][0]
    assert (shown["state"], shown["note"]) == ("stopped", reason)

    marker.write_text("")  # a marker without a reason still counts
    assert (run.state, run.stopped_note) == ("stopped", "stopped")

    # The run is taken up again: its log is newer than the marker, which then no longer counts.
    os.utime(marker, (hour_ago - 600, hour_ago - 600))
    assert (run.state, run.stopped_note) == ("stalled", None)
    os.utime(log, None)
    assert run.state == "training"

    # A finished run is done whatever else is lying around.
    done = runstate.Run(str(tree.experiments), "alpha")
    done.update()
    (tree.experiments / "alpha.stopped").write_text("left over\n")
    assert done.state == "done"
    assert runstate.snapshot([done], runstate.Evals(str(tree.results)))["runs"][0]["note"] is None


def test_run_reads_earlier_attempts_first(tree):
    (tree.experiments / "gamma.train.log.part1").write_text(
        "\n".join([samples_line(0, 2000), progress_line(0, 1000, 1000, "2.000"), samples_line(1000, 2000)]) + "\n"
    )
    (tree.experiments / "gamma.train.log").write_text(
        "\n".join([samples_line(1000, 2000), progress_line(1, 600, 1000, "1.000")]) + "\n"
    )
    names = [run.name for run in runstate.discover(str(tree.experiments), {})]
    assert names == ["alpha", "beta", "gamma"]
    run = runstate.Run(str(tree.experiments), "gamma")
    run.update()
    assert run.curve == [(1000, 2.0), (1600, 1.0)]
    assert run.samples_at_checkpoint() == {1: 1000}  # checkpoint 2 is not written yet: its samples are not known


def test_run_waits_for_the_end_of_a_line(tree):
    log = tree.experiments / "beta.train.log"
    run = runstate.Run(str(tree.experiments), "beta")
    run.update()
    line = progress_line(0, 1000, 1000, "2.500", lr=0.00018)
    cut = line.index("LR: ") + len("LR: 0.")
    with open(log, "a") as f:  # the trainer is in the middle of writing this line
        f.write(line[:cut])
    run.update()
    assert len(run.curve) == 2 and run.seen == 800  # nothing is read from half a line
    with open(log, "a") as f:
        f.write(line[cut:] + "\n")
    run.update()
    assert run.curve[-1] == (1000, 2.5) and run.lr == pytest.approx(0.00018)

    with open(log, "a") as f:  # a log that ends without a line break is still read to its end
        f.write(progress_line(0, 1000, 1000, "2.250"))
    run.update()
    run.update()
    assert run.curve[-1] == (1000, 2.25)


def test_run_survives_lines_that_cannot_be_read(tree):
    (tree.experiments / "torn.train.log").write_bytes(
        b"\xff\xfe\x00 not text\n"
        + samples_line(0, 2000).encode()
        + b"\n"
        + progress_line(0, 500, 1000, "1.000").encode().replace(b"LR: 0.000100", b"LR: 1.8e-")
        + b"\n"
        + progress_line(0, 600, 1000, "0.900").encode().replace(b"Loss: 0.900", b"Loss: .")
        + b"\n"
        + progress_line(0, 700, 1000, "0.800").encode()
        + b"\n"
    )
    run = runstate.Run(str(tree.experiments), "torn")
    run.update()
    assert run.curve == [(700, 0.8)] and (run.seen, run.total) == (700, 2000)


def test_a_replaced_log_is_not_counted_twice(tree):
    log = tree.experiments / "beta.train.log"
    run = runstate.Run(str(tree.experiments), "beta")
    run.update()
    assert len(run.curve) == 2

    # A copy of the log, further along, takes the place of the file (as rsync does).
    longer = RUNNING_RUN + [progress_line(0, 1000, 1000, "2.500"), samples_line(1000, 4000)]
    (tree.experiments / "copy").write_text("\n".join(longer) + "\n")
    os.replace(tree.experiments / "copy", log)
    run.update()
    assert [samples for samples, _ in run.curve] == [400, 800, 1000] and run.seen == 1000

    # A new attempt: the log so far becomes a .part file and a new log starts.
    os.replace(log, tree.experiments / "beta.train.log.part1")
    run.update()  # between the two, there is no live log
    assert run.state == "waiting" and len(run.curve) == 3
    log.write_text("\n".join([samples_line(1000, 4000), progress_line(1, 300, 1000, "2.000")]) + "\n")
    run.update()
    assert [samples for samples, _ in run.curve] == [400, 800, 1000, 1300] and run.seen == 1300


@pytest.mark.parametrize(
    "content",
    [
        "{}",  # JSON, but not an evaluation
        "[1, 2, 3]",
        "null",
        '{"successes": 1, "episodes": 4, "success_rate": NaN, "success_ci95": [0.1, 0.9]}',
        '{"successes": 1, "episodes": 4, "success_rate": 0.25, "success_ci95": null}',
        '{"successes": 1, "episodes": 4, "success_rate": 0.25, "success_ci95": [0.1, 0.9], "per_target": {"cube": 3}}',
        json.dumps(results_json(1, 4, "", 12.5))[:200],  # eval.py is in the middle of writing it
        "",
    ],
)
def test_results_that_are_not_evaluations_are_left_out(tree, content):
    (tree.results / "alpha" / "pick_target" / "ckpt_3_raw" / "results.json").write_text(content)
    rows, running = runstate.Evals(str(tree.results)).of("alpha")
    assert [(checkpoint, weights) for checkpoint, weights, _, _ in rows] == [(1, "raw"), (2, "ema")] and running == 0
    data = runstate.snapshot(runstate.discover(str(tree.experiments), {}), runstate.Evals(str(tree.results)))
    assert [e["checkpoint"] for e in data["runs"][0]["evaluations"]] == [1, 2]


def test_episodes_that_cannot_be_read_hide_one_evaluation_only(tree):
    content = results_json(2, 4, "", 12.5)
    content["per_episode"] = None
    (tree.results / "alpha" / "pick_target" / "ckpt_3_raw" / "results.json").write_text(json.dumps(content))
    data = runstate.snapshot(runstate.discover(str(tree.experiments), {}), runstate.Evals(str(tree.results)))
    assert [e["checkpoint"] for e in data["runs"][0]["evaluations"]] == [1, 2]
    assert [run["name"] for run in data["runs"]] == ["alpha", "beta"]


def test_long_curves_are_averaged_down(tree):
    lines = [samples_line(0, 100_000)] + [
        progress_line(0, 100 * i, 100_000, f"{1 + i / 1000:.3f}") for i in range(1, 1001)
    ]
    (tree.experiments / "long.train.log").write_text("\n".join(lines) + "\n")
    run = runstate.Run(str(tree.experiments), "long")
    run.update()
    curve = run.downsampled_curve(50)
    assert len(curve) == 50 and curve[-1][0] == 100_000
    assert curve == sorted(curve) and 1.0 < curve[0][1] < curve[-1][1] <= 2.0


def test_evaluations_of_a_run(tree):
    rows, running = runstate.Evals(str(tree.results)).of("alpha")
    assert [(checkpoint, weights, task) for checkpoint, weights, task, _ in rows] == [
        (1, "raw", "pick_target"),
        (2, "ema", "pick_target"),
    ]
    assert rows[0][3]["successes"] == 1 and running == 1
    assert runstate.Evals(str(tree.results), "another_task").of("alpha") == ([], 0)
    assert runstate.Evals(str(tree.results)).of("beta") == ([], 0)


def test_snapshot_shape(tree):
    data = runstate.snapshot(runstate.discover(str(tree.experiments), {}), runstate.Evals(str(tree.results)))
    assert set(data) == {"time", "runs", "gpus"} and data["gpus"] == GPUS
    alpha, beta = data["runs"]
    assert set(alpha) == RUN_KEYS and set(beta) == RUN_KEYS
    assert (alpha["name"], alpha["state"], alpha["samples_seen"], alpha["samples_total"]) == (
        "alpha",
        "done",
        3000,
        3000,
    )
    assert alpha["loss_curve"][0] == [500, 2.0] and len(alpha["loss_curve"]) == 6
    assert alpha["samples_per_second"] is None and alpha["seconds_left"] is None  # only while training
    assert alpha["evaluations_running"] == 1
    first, second = alpha["evaluations"]
    assert set(first) == EVALUATION_KEYS
    assert (first["checkpoint"], first["weights"], first["samples"], first["success_rate"]) == (1, "raw", 1000, 0.25)
    assert (second["checkpoint"], second["weights"], second["samples"], second["successes"]) == (2, "ema", 2000, 3)
    assert first["per_target"] == {"cube": dict(successes=1, episodes=4)}
    assert (beta["state"], beta["samples_seen"], beta["evaluations"]) == ("training", 800, [])
    assert beta["samples_per_second"] == 100.0 and beta["seconds_left"] == pytest.approx(32.0)


# ---- the server: pieces --------------------------------------------------------------------------


def test_scrub_makes_strict_and_relative(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    here = os.getcwd()
    dirty = dict(
        a=float("nan"),
        b=[float("inf"), 1.5, None, True, 3],
        inside=os.path.join(here, "experiments", "run", "ema_2.pt"),
        outside="/somewhere/else/checkpoint_7.pt",
        home="~/data/checkpoint_8.pt",
        windows="C:\\data\\checkpoint_9.pt",
        text="pick up the red cube",
        relative="experiments/run/ema_2.pt",
    )
    assert monitor_server.scrub(dirty) == dict(
        a=None,
        b=[None, 1.5, None, True, 3],
        inside=os.path.join("experiments", "run", "ema_2.pt"),
        outside="checkpoint_7.pt",
        home="checkpoint_8.pt",
        windows="checkpoint_9.pt",
        text="pick up the red cube",
        relative="experiments/run/ema_2.pt",
    )


@pytest.mark.parametrize(
    "header, expected",
    [
        (None, None),
        ("bytes=0-", (0, 255)),
        ("bytes=2-5", (2, 5)),
        ("bytes=250-999", (250, 255)),
        ("bytes=-6", (250, 255)),
        ("bytes=-999", (0, 255)),
        ("bytes=0-1,4-5", None),
        ("items=0-5", None),
        ("bytes=-", None),
    ],
)
def test_byte_range(header, expected):
    assert monitor_server.byte_range(header, 256) == expected


@pytest.mark.parametrize("header", ["bytes=256-", "bytes=300-400", "bytes=9-3", "bytes=-0"])
def test_byte_range_outside_the_file(header):
    with pytest.raises(ValueError):
        monitor_server.byte_range(header, 256)


def test_token_is_required_off_loopback():
    for host in ("127.0.0.1", "127.0.0.5", "localhost", "::1"):
        assert monitor_server.access_token(host, None) is None
        assert monitor_server.access_token(host, "chosen") == "chosen"
    for host in ("0.0.0.0", "::", "", "192.168.1.20", "training-machine"):
        generated = monitor_server.access_token(host, None)
        assert generated and len(generated) >= 24 and generated != monitor_server.access_token(host, None)
        assert monitor_server.access_token(host, "chosen") == "chosen"


# ---- the server: routes --------------------------------------------------------------------------


def test_state(server, tree):
    status, headers, body = get(server, "/api/state")
    assert status == 200 and headers["content-type"] == "application/json"
    assert headers["cache-control"] == "no-store"
    data = strict_json(body)
    assert set(data) == {"time", "runs", "gpus", "interval", "error"}
    assert data["error"] is None and data["gpus"] == GPUS
    assert [run["name"] for run in data["runs"]] == ["alpha", "beta"]
    assert all(set(run) == RUN_KEYS for run in data["runs"])
    assert data["runs"][1]["loss"] is None  # nan in the log
    assert data["runs"][0]["evaluations"][1]["median_miss_named_mm"] is None  # NaN in a results file
    assert str(tree.root).encode() not in body and b"TOP SECRET" not in body
    assert get(server, "/api/state", method="HEAD")[2] == b""


def test_state_follows_the_files(server, tree):
    (tree.experiments / "delta.train.log").write_text("\n".join(RUNNING_RUN) + "\n")

    def names():
        return [run["name"] for run in strict_json(get(server, "/api/state")[2])["runs"]]

    assert wait_until(lambda: names() == ["alpha", "beta", "delta"])


def test_stream_sends_an_event_per_refresh_and_ends_with_the_client(server):
    def handlers():
        return [t for t in threading.enumerate() if "process_request_thread" in t.name]

    assert wait_until(lambda: not handlers())  # connections of earlier tests are gone
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    connection.request("GET", "/api/stream")
    response = connection.getresponse()
    assert response.status == 200 and response.getheader("Content-Type").startswith("text/event-stream")
    events, identifiers = [], []
    while len(events) < 3:
        line = response.readline()
        assert line, "the stream ended early"
        if line.startswith(b"data: "):
            events.append(strict_json(line[len(b"data: ") :]))
        elif line.startswith(b"id: "):
            identifiers.append(int(line[4:]))
    assert all([run["name"] for run in event["runs"]] == ["alpha", "beta"] for event in events)
    assert identifiers == sorted(set(identifiers)) and events[0]["time"] < events[2]["time"]
    assert len(handlers()) == 1
    response.close()
    connection.close()
    assert wait_until(lambda: not handlers()), "the stream outlived its client"
    assert get(server, "/api/state")[0] == 200


def test_stream_ends_when_the_server_stops(tree):
    with serving(tree) as running:
        connection = http.client.HTTPConnection("127.0.0.1", running.server_port, timeout=10)
        connection.request("GET", "/api/stream")
        response = connection.getresponse()
        while not response.readline().startswith(b"data: "):
            pass
    try:
        assert response.read()[-2:] in (b"\n\n", b"")  # whole events, then the end of the stream
    finally:
        connection.close()


def test_evaluation(server, tree):
    query = urlencode(dict(run="alpha", task="pick_target", variant="ckpt_1_raw"))
    status, headers, body = get(server, f"/api/evaluation?{query}")
    assert status == 200 and headers["content-type"] == "application/json"
    results = strict_json(body)
    assert results["successes"] == 1 and len(results["per_episode"]) == 2
    assert results["per_episode"][0]["close_tcp_xy"] == [0.5, 0.0]
    assert results["per_episode"][0]["close_object_xy"]["cube"] == [0.5, 0.004]
    assert results["per_episode"][1]["close_tcp_xy"] is None
    assert results["server"]["checkpoint"] == "checkpoint_1.pt"  # was an absolute path
    assert str(tree.root).encode() not in body

    query = urlencode(dict(run="alpha", task="pick_target", variant="ckpt_2_ema"))
    results = strict_json(get(server, f"/api/evaluation?{query}")[2])
    assert results["grasp"]["median_xy_err_target_mm"] is None  # NaN in the file
    assert results["server"]["checkpoint"] == "experiments/alpha/checkpoints/ema_2.pt"


@pytest.mark.parametrize(
    "query",
    [
        dict(),
        dict(run="alpha"),
        dict(run="", task="pick_target", variant="ckpt_1_raw"),
        dict(run="nope", task="pick_target", variant="ckpt_1_raw"),
        dict(run="alpha", task="nope", variant="ckpt_1_raw"),
        dict(run="alpha", task="pick_target", variant="ckpt_9_raw"),  # well formed, not on disk
        dict(run="alpha", task="pick_target", variant="ckpt_1"),
        dict(run="..", task="results", variant="ckpt_1_raw"),
        dict(run="../results/alpha", task="pick_target", variant="ckpt_1_raw"),
        dict(run="alpha/pick_target", task="ckpt_1_raw"),
        dict(run="alpha", task="..", variant=""),
        dict(run="alpha", task="pick_target", variant="../ckpt_1_raw"),
        dict(run="alpha", task="pick_target", variant="ckpt_1_raw/.."),
        dict(run="alpha", task="pick_target/ckpt_1_raw", variant=""),
        dict(run=".", task="alpha", variant=""),
        dict(run="alpha\0", task="pick_target", variant="ckpt_1_raw"),
    ],
)
def test_evaluation_names_must_exist_on_disk(server, query):
    for route in ("/api/evaluation", "/api/videos"):
        status, _, body = get(server, f"{route}?{urlencode(query)}")
        assert status == 404, (route, query)
        assert strict_json(body) == {"error": "no such evaluation"}


def test_evaluation_without_results_yet(server):
    for query in (
        dict(run="alpha", task="pick_target", variant="ckpt_3_raw"),  # claimed, still running
        dict(run="alpha", task="pick_target"),  # nothing was evaluated without naming a checkpoint
    ):
        assert get(server, f"/api/evaluation?{urlencode(query)}")[0] == 404
        status, _, body = get(server, f"/api/videos?{urlencode(query)}")
        assert (status, strict_json(body)) == (200, [])


def test_evaluation_by_absolute_path_is_refused(server, tree):
    query = urlencode(dict(run=str(tree.results / "alpha"), task="pick_target", variant="ckpt_1_raw"))
    assert get(server, f"/api/evaluation?{query}")[0] == 404


def test_tolerance(server, tree):
    status, _, body = get(server, "/api/tolerance")
    assert status == 404 and "error" in strict_json(body)
    table = dict(task="pick_target", episodes_per_offset=8, by_offset_mm={"0": {"all": dict(successes=8, episodes=8)}})
    (tree.results / "grasp_tolerance.json").write_text(json.dumps(table))
    status, _, body = get(server, "/api/tolerance")
    assert status == 200 and strict_json(body) == table


def test_videos_and_ranges(server):
    query = urlencode(dict(run="alpha", task="pick_target", variant="ckpt_1_raw"))
    status, _, body = get(server, f"/api/videos?{query}")
    assert status == 200
    assert strict_json(body) == [
        dict(
            name="rollout_000_cube_ok.mp4",
            url="/media/alpha/pick_target/ckpt_1_raw/rollout_000_cube_ok.mp4",
            bytes=len(VIDEO),
            episode=0,
            target="cube",
            success=True,
        )
    ]
    query = urlencode(dict(run="alpha", task="pick_target", variant="ckpt_2_ema"))
    assert strict_json(get(server, f"/api/videos?{query}")[2]) == []

    url = "/media/alpha/pick_target/ckpt_1_raw/rollout_000_cube_ok.mp4"
    status, headers, body = get(server, url)
    assert (status, body) == (200, VIDEO)
    assert headers["content-type"] == "video/mp4" and headers["accept-ranges"] == "bytes"
    status, headers, body = get(server, url, {"Range": "bytes=2-5"})
    assert (status, body, headers["content-range"]) == (206, VIDEO[2:6], "bytes 2-5/256")
    assert headers["content-length"] == "4"
    status, headers, body = get(server, url, {"Range": "bytes=250-"})
    assert (status, body, headers["content-range"]) == (206, VIDEO[250:], "bytes 250-255/256")
    status, headers, body = get(server, url, {"Range": "bytes=-6"})
    assert (status, body, headers["content-range"]) == (206, VIDEO[-6:], "bytes 250-255/256")
    status, headers, body = get(server, url, {"Range": "bytes=300-"})
    assert (status, body, headers["content-range"]) == (416, b"", "bytes */256")
    status, headers, body = get(server, url, method="HEAD")
    assert (status, body, headers["content-length"]) == (200, b"", "256")


def test_static_files(server):
    status, headers, body = get(server, "/")
    assert status == 200 and body == b"<!doctype html><title>monitor</title>"
    assert headers["content-type"] == "text/html; charset=utf-8"
    assert "default-src 'self'" in headers["content-security-policy"]
    assert get(server, "/index.html")[2] == body
    assert get(server, "/?anything=1")[2] == body
    status, headers, body = get(server, "/app.js")
    assert (status, headers["content-type"]) == (200, "text/javascript; charset=utf-8")
    assert body == b"export const ready = true;"
    assert get(server, "/style.css")[1]["content-type"] == "text/css; charset=utf-8"
    assert get(server, "/sub/")[2] == b"<!doctype html><title>sub</title>"
    for missing in ("/missing.js", "/sub", "/api/missing", "/index.html/"):
        assert get(server, missing)[0] == 404
    assert get(server, "/api/state", method="POST")[0] == 501


def test_the_packaged_ui_directory_is_next_to_the_server():
    assert Path(monitor_server.UI_DIRECTORY) == REPOSITORY / "sim" / "pickup" / "monitor_ui"


@pytest.mark.parametrize(
    "path",
    [
        "/../secret.txt",
        "/%2e%2e/secret.txt",
        "/..%2fsecret.txt",
        "/%2e%2e%2fsecret.txt",
        "/%252e%252e/secret.txt",
        "/sub/../../secret.txt",
        "/sub/%2e%2e/%2e%2e/secret.txt",
        "/..\\secret.txt",
        "/..%5csecret.txt",
        "//secret.txt",
        "/./../secret.txt",
        "/%00/../secret.txt",
        "/media/../secret.txt",
        "/media/%2e%2e/secret.txt",
        "/media/..%2fsecret.txt",
        "/media/../outside/clip.mp4",
        "/media/%2e%2e/outside/clip.mp4",
        "/media/..%2foutside%2fclip.mp4",
        "/media/../experiments/alpha.train.log",
        "/media/alpha/pick_target/ckpt_1_raw/results.json",  # there, but not a video
        "/media/alpha/pick_target/ckpt_1_raw/.hidden.mp4",
        "/media/alpha/pick_target/ckpt_1_raw/../ckpt_1_raw/rollout_000_cube_ok.mp4",
        "/media/alpha//pick_target/ckpt_1_raw/rollout_000_cube_ok.mp4",
        "/media/",
        "/api/../secret.txt",
        "/api/state/../../secret.txt",
    ],
)
def test_traversal_is_refused(server, path):
    status, _, body = get(server, path)
    assert status == 404, path
    assert SECRET not in body


def test_absolute_paths_are_refused(server, tree):
    for path in (f"/{tree.root}/secret.txt", f"/media/{tree.root}/outside/clip.mp4", f"//{tree.root}/secret.txt"):
        status, _, body = get(server, path)
        assert status == 404 and SECRET not in body


def test_symbolic_links_out_of_the_directories_are_refused(server, tree):
    try:
        os.symlink(tree.root / "outside", tree.results / "linked")
        os.symlink(tree.root / "secret.txt", tree.ui / "linked.txt")
        os.symlink(tree.root / "outside" / "clip.mp4", tree.results / "alpha" / "pick_target" / "ckpt_1_raw" / "x.mp4")
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are not available")
    for path in ("/media/linked/clip.mp4", "/linked.txt", "/media/alpha/pick_target/ckpt_1_raw/x.mp4"):
        status, _, body = get(server, path)
        assert status == 404 and SECRET not in body
    query = urlencode(dict(run="alpha", task="pick_target", variant="ckpt_1_raw"))
    listed = strict_json(get(server, f"/api/videos?{query}")[2])
    assert [video["name"] for video in listed] == ["rollout_000_cube_ok.mp4"]  # the list offers what /media serves


def test_only_requests_addressed_to_this_machine_are_answered_without_a_token(server):
    for host in ("attacker.example", "attacker.example:8765", "192.168.1.20:8765", "localhost.attacker.example"):
        status, _, body = get(server, "/api/state", {"Host": host})
        assert status == 403 and b"runs" not in body
    for host in ("localhost", "localhost:1234", "127.0.0.1:8765", "[::1]:8765"):
        assert get(server, "/api/state", {"Host": host})[0] == 200


def test_token_is_enforced(tree):
    with serving(tree, token="s3cret-token") as server:
        video = "/media/alpha/pick_target/ckpt_1_raw/rollout_000_cube_ok.mp4"
        for path in ("/", "/app.js", "/api/state", "/api/stream", "/api/tolerance", video, "/api/state?token="):
            status, _, body = get(server, path)
            assert status == 401, path
            assert b"alpha" not in body and b"monitor</title>" not in body
        for wrong in ("nope", "s3cret-toke", "s3cret-token2"):
            assert get(server, f"/api/state?token={wrong}")[0] == 401
        assert get(server, "/api/evaluation?run=alpha&task=pick_target&variant=ckpt_1_raw")[0] == 401

        status, headers, body = get(server, "/?token=s3cret-token")
        assert status == 200 and body.startswith(b"<!doctype html>")
        cookie = headers["set-cookie"].split(";")[0]
        assert "HttpOnly" in headers["set-cookie"] and cookie.endswith("=s3cret-token")

        status, headers, body = get(server, "/api/state", {"Cookie": f"other=1; {cookie}"})
        assert status == 200 and "set-cookie" not in headers
        assert b"s3cret-token" not in body
        assert get(server, "/app.js", {"Cookie": cookie})[0] == 200
        assert get(server, "/api/state", {"Cookie": cookie + "x"})[0] == 401
        assert get(server, "/api/state", {"Cookie": "monitor_token_1=s3cret-token"})[0] == 401
        # With a token the address in the request does not matter: the token is the check.
        assert get(server, "/api/state?token=s3cret-token", {"Host": "training-machine:8765"})[0] == 200


# ---- the command line -----------------------------------------------------------------------------


@pytest.mark.skipif(os.name == "nt", reason="signals")
@pytest.mark.parametrize("extra, address", [([], "/"), (["--token", "abc"], "/?token=abc")])
def test_command_line_starts_serves_and_stops(tree, extra, address):
    command = [sys.executable, "-m", "sim.pickup.monitor_server", "--port", "0", "--interval", "0.2"]
    command += ["--experiments", str(tree.experiments), "--results", str(tree.results), *extra]
    process = subprocess.Popen(command, cwd=REPOSITORY, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        line = process.stdout.readline()
        match = re.search(r"http://127\.0\.0\.1:(\d+)(/\S*)", line)
        assert match and match.group(2) == address, line
        fake = SimpleNamespace(server_port=int(match.group(1)))
        status, _, body = get(fake, "/api/state" + address[1:])
        assert status == 200
        assert [run["name"] for run in strict_json(body)["runs"]] == ["alpha", "beta"]
        if extra:
            assert get(fake, "/api/state")[0] == 401
        process.send_signal(signal.SIGTERM)
        out, _ = process.communicate(timeout=10)
        assert process.returncode == 0 and "stopped" in out
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
