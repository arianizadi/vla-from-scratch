"""State of training runs and their evaluations, read from log and result files.

Shared by the terminal dashboard (monitor.py), the HTTP server behind the graphical one
(monitor_server.py) and the exporter (export.py). Nothing here imports a UI library or touches a
running job: a run is whatever its `<name>.train.log` and its results directory say.
"""

import glob
import json
import os
import re
import subprocess
import time

from .summarize import close_slope, served_checkpoint_matches

PROGRESS = re.compile(
    r"Train Checkpoint: (\d+) \[\s*(\d+)/(\d+) \(\s*\d+%\)\] Loss: ([\d.]+|nan).*? ([\d.]+)/s,.*?LR: ([\d.e+-]+)"
)
SAMPLES = re.compile(r"Samples: (\d+) / (\d+)")
FINISHED = "Model has seen the desired number of samples"
STATES = ("training", "done", "stalled", "waiting")


class Run:
    """Incrementally parsed training log of one run (earlier attempts, then the live log)."""

    def __init__(self, experiments, name):
        self.experiments, self.name = experiments, name
        self.live = os.path.join(experiments, f"{name}.train.log")
        self.offset = 0
        self.inode = None
        self.base = self.in_chunk = self.total = 0
        self.loss = self.rate = self.lr = None
        self.curve = []  # (samples seen, loss), one point per logged step
        self.finished = False
        for part in sorted(glob.glob(self.live + ".part*")):  # attempts that died and were resumed
            with open(part, errors="ignore") as f:
                self._parse(f.read())
        self.finished = False

    def _parse(self, text):
        for line in text.splitlines():
            m = PROGRESS.search(line)
            if m:
                self.in_chunk = int(m.group(2))
                self.loss = float(m.group(4))
                self.rate = float(m.group(5))
                self.lr = float(m.group(6))
                self.curve.append((self.base + self.in_chunk, self.loss))
                continue
            m = SAMPLES.search(line)
            if m:
                self.base, self.total, self.in_chunk = int(m.group(1)), int(m.group(2)), 0
            elif FINISHED in line:
                self.finished = True

    def update(self):
        try:
            st = os.stat(self.live)
        except FileNotFoundError:
            return
        if st.st_ino != self.inode or st.st_size < self.offset:  # a new attempt started a new log
            self.inode, self.offset = st.st_ino, 0
        with open(self.live, errors="ignore") as f:
            f.seek(self.offset)
            text = f.read()
            self.offset = f.tell()
        self._parse(text)
        if len(self.curve) > 40000:
            self.curve = self.curve[::2]

    @property
    def losses(self):
        return [loss for _, loss in self.curve]

    @property
    def seen(self):
        return self.total if self.finished and self.total else min(self.base + self.in_chunk, self.total or 10**18)

    @property
    def age(self):
        try:
            return time.time() - os.stat(self.live).st_mtime
        except FileNotFoundError:
            return None

    @property
    def state(self):
        if self.finished:
            return "done"
        if self.age is None:
            return "waiting"
        return "training" if self.age < 180 else "stalled"

    @property
    def restarts(self):
        try:
            with open(os.path.join(self.experiments, f"{self.name}.retries.log")) as f:
                return sum(1 for line in f if "ended" in line or "stalled" in line)
        except FileNotFoundError:
            return 0

    @property
    def checkpoints(self):
        return len(glob.glob(os.path.join(self.experiments, self.name, "checkpoints", "checkpoint_*.pt")))

    def samples_at_checkpoint(self):
        """{checkpoint number: samples seen when it was written}, from the "Samples: X / Y" log lines."""
        seen = set()
        for log in glob.glob(self.live + "*"):
            with open(log, errors="ignore") as f:
                seen.update(int(m.group(1)) for m in SAMPLES.finditer(f.read()))
        ordered = sorted(seen)  # ordered[n] = samples at the start of chunk n = when checkpoint n was written
        out = {n: s for n, s in enumerate(ordered) if n > 0}
        if self.total:
            out[len(ordered)] = self.total  # the final checkpoint has no following chunk to log its count
        return out

    def downsampled_curve(self, points=300):
        """The loss curve averaged into at most `points` buckets: [[samples, loss], ...]."""
        curve = [(s, loss) for s, loss in self.curve if loss == loss]
        if len(curve) <= points:
            return [[s, round(loss, 5)] for s, loss in curve]
        step = len(curve) / points
        out = []
        for i in range(points):
            bucket = curve[int(i * step) : max(int((i + 1) * step), int(i * step) + 1)]
            out.append([bucket[-1][0], round(sum(loss for _, loss in bucket) / len(bucket), 5)])
        return out


class Evals:
    """Closed-loop results of a run's checkpoints, cached by file modification time."""

    def __init__(self, results, task=None):
        self.results, self.task, self.cache = results, task, {}

    def load(self, path):
        mtime = os.path.getmtime(path)
        if self.cache.get(path, (None,))[0] != mtime:
            with open(path) as f:
                self.cache[path] = (mtime, json.load(f))
        return self.cache[path][1]

    def of(self, name):
        """([(checkpoint, "raw" | "ema", task, results dict), ...] sorted, number still running)."""
        rows, running = [], 0
        for d in glob.glob(os.path.join(self.results, name, self.task or "*", "ckpt_*")):
            m = re.fullmatch(r"ckpt_(\d+)_(raw|ema)", os.path.basename(d))
            path = os.path.join(d, "results.json")
            if not m:
                continue
            if not os.path.exists(path):
                running += 1
                continue
            try:
                results = self.load(path)
            except (OSError, json.JSONDecodeError):
                continue
            if not served_checkpoint_matches(results, name, int(m.group(1)), m.group(2)):
                continue  # answered by another run's policy server (two evaluations on one port)
            rows.append((int(m.group(1)), m.group(2), os.path.basename(os.path.dirname(d)), results))
        return sorted(rows, key=lambda r: (r[2], r[0], r[1])), running

    def probes(self, name):
        """Evaluations of a run under other conditions than a plain checkpoint evaluation (sampler
        settings, interventions, disturbances): [(variant, task, results dict), ...] sorted."""
        rows = []
        for path in glob.glob(os.path.join(self.results, name, self.task or "*", "*", "results.json")):
            variant = os.path.basename(os.path.dirname(path))
            if re.fullmatch(r"ckpt_(\d+)_(raw|ema)", variant):
                continue
            try:
                rows.append((variant, os.path.basename(os.path.dirname(os.path.dirname(path))), self.load(path)))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(rows, key=lambda r: (r[1], r[0]))


def gpus():
    """[{index, name, util, mem_used_mb, mem_total_mb, temp_c}, ...] from nvidia-smi, or [] without it."""
    query = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=8,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        try:
            index, name, util, used, total, temp = parts
            rows.append(
                dict(
                    index=int(index),
                    name=name.replace("NVIDIA ", ""),
                    util=float(util),
                    mem_used_mb=float(used),
                    mem_total_mb=float(total),
                    temp_c=float(temp),
                )
            )
        except ValueError:
            continue
    return rows


def discover(experiments, known):
    """Find runs by their training logs; `known` ({name: Run}) is updated in place and returned sorted."""
    names = set()
    for path in glob.glob(os.path.join(experiments, "*.train.log*")):
        names.add(os.path.basename(path).split(".train.log")[0])
    for name in sorted(names - set(known)):
        known[name] = Run(experiments, name)
    return [known[n] for n in sorted(known)]


def evaluation_row(checkpoint, weights, task, results, samples=None):
    """The fields of one evaluation that a table or a chart needs."""
    grasp = results.get("grasp") or {}
    return dict(
        checkpoint=checkpoint,
        weights=weights,
        task=task,
        samples=samples,
        successes=results["successes"],
        episodes=results["episodes"],
        success_rate=results["success_rate"],
        ci95=results["success_ci95"],
        at_named_object=grasp.get("closed_nearest_is_target_rate"),
        median_miss_named_mm=grasp.get("median_xy_err_target_mm"),
        median_miss_nearest_mm=grasp.get("median_xy_err_nearest_mm"),
        within_10mm=grasp.get("within_10mm_of_target_rate"),
        within_20mm=grasp.get("within_20mm_of_target_rate"),
        close_slope=close_slope(results),
        retried_rate=grasp.get("retried_rate"),
        wrong_object_rate=results.get("wrong_object_rate"),
        per_target={
            k: dict(successes=v["successes"], episodes=v["episodes"]) for k, v in results.get("per_target", {}).items()
        },
    )


def snapshot(runs, evals, curve_points=300):
    """Everything about the given runs as plain data (what the server sends and the exporter writes)."""
    out = []
    for run in runs:
        run.update()
        rows, running = evals.of(run.name)
        samples = run.samples_at_checkpoint() if rows else {}
        left = (run.total - run.seen) / run.rate if run.state == "training" and run.rate else None
        out.append(
            dict(
                name=run.name,
                state=run.state,
                samples_seen=run.seen,
                samples_total=run.total,
                loss=run.loss,
                learning_rate=run.lr,
                samples_per_second=run.rate if run.state == "training" else None,
                seconds_left=left,
                restarts=run.restarts,
                checkpoints=run.checkpoints,
                seconds_since_log=run.age,
                loss_curve=run.downsampled_curve(curve_points),
                evaluations=[evaluation_row(n, w, task, r, samples.get(n)) for n, w, task, r in rows],
                evaluations_running=running,
            )
        )
    return dict(time=time.time(), runs=out, gpus=gpus())
