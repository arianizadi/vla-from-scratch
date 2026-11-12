"""Terminal dashboard for training runs and their closed-loop evaluations.

Shows, for every run under an experiments directory: progress, loss (with its history), throughput,
time left, restarts, and next to it what the loss cannot tell you: closed-loop success per
checkpoint, how often the gripper closed at the named object, and how far it missed. GPU load is
shown underneath. It only reads log and result files, so it can run next to training, in another
terminal or over ssh, and be closed and reopened at any time.

    python -m sim.pickup.monitor                       # live view, Ctrl-C to quit
    python -m sim.pickup.monitor --task pick_cube
    python -m sim.pickup.monitor --once                # print one frame and exit
    python -m sim.pickup.monitor --once --svg runs.svg # save the frame as an image
"""

import argparse
import glob
import json
import os
import re
import subprocess
import time
from datetime import UTC, datetime

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

SPARK = "▁▂▃▄▅▆▇█"
STATE_STYLE = {"training": "bold green", "done": "bold cyan", "stalled": "bold yellow", "waiting": "dim"}
STATE_MARK = {"training": "●", "done": "✔", "stalled": "▲", "waiting": "○"}
PROGRESS = re.compile(
    r"Train Checkpoint: (\d+) \[\s*(\d+)/(\d+) \(\s*\d+%\)\] Loss: ([\d.]+|nan).*? ([\d.]+)/s,.*?LR: ([\d.e+-]+)"
)
SAMPLES = re.compile(r"Samples: (\d+) / (\d+)")
FINISHED = "Model has seen the desired number of samples"


def spark(values, width):
    """Unicode sparkline of `values`, averaged down to at most `width` characters."""
    values = [v for v in values if v is not None and v == v]
    if not values:
        return ""
    if len(values) > width:
        step = len(values) / width
        buckets = [values[int(i * step) : max(int((i + 1) * step), int(i * step) + 1)] for i in range(width)]
        values = [sum(b) / len(b) for b in buckets]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return "".join(SPARK[min(len(SPARK) - 1, int((v - lo) / span * (len(SPARK) - 1) + 0.5))] for v in values)


def bar(fraction, width, style):
    fraction = min(max(fraction, 0.0), 1.0)
    filled = int(round(fraction * width))
    return Text("█" * filled, style=style) + Text("░" * (width - filled), style="grey35")


def duration(seconds):
    if seconds is None or seconds != seconds or seconds < 0:
        return "-"
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


class Run:
    """Incrementally parsed training log of one run (earlier attempts, then the live log)."""

    def __init__(self, experiments, name):
        self.experiments, self.name = experiments, name
        self.live = os.path.join(experiments, f"{name}.train.log")
        self.offset = 0
        self.inode = None
        self.base = self.in_chunk = self.total = 0
        self.loss = self.rate = None
        self.losses = []
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
                self.losses.append(self.loss)
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
        if len(self.losses) > 20000:
            self.losses = self.losses[::2]

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


class Evals:
    """Closed-loop results of a run's checkpoints, cached by file modification time."""

    def __init__(self, results, task):
        self.results, self.task, self.cache = results, task, {}

    def of(self, name):
        rows, running = [], 0
        for d in glob.glob(os.path.join(self.results, name, self.task or "*", "ckpt_*")):
            m = re.fullmatch(r"ckpt_(\d+)_(raw|ema)", os.path.basename(d))
            path = os.path.join(d, "results.json")
            if not m:
                continue
            if not os.path.exists(path):
                running += 1
                continue
            mtime = os.path.getmtime(path)
            if self.cache.get(path, (None,))[0] != mtime:
                try:
                    with open(path) as f:
                        self.cache[path] = (mtime, json.load(f))
                except (OSError, json.JSONDecodeError):
                    continue
            rows.append((int(m.group(1)), m.group(2), self.cache[path][1]))
        return sorted(rows, key=lambda r: (r[0], r[1])), running


def gpus():
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
        if len(parts) == 6:
            try:
                rows.append(
                    (int(parts[0]), parts[1], float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
                )
            except ValueError:
                continue
    return rows


def runs_table(runs, evals, width):
    wide = width >= 150
    table = Table(box=box.SIMPLE_HEAD, expand=True, pad_edge=False, header_style="bold")
    table.add_column("run", no_wrap=True, min_width=min(34, max([len(r.name) for r in runs] + [3]) + 6))
    table.add_column("progress", no_wrap=True, min_width=(14 if wide else 8) + 18)
    table.add_column("loss", no_wrap=True, min_width=6 + (14 if wide else 0))
    table.add_column("speed", justify="right", no_wrap=True, min_width=7)
    table.add_column("left", justify="right", no_wrap=True, min_width=6)
    table.add_column("closed-loop success by checkpoint", no_wrap=True, ratio=1)
    table.add_column("best", justify="right", no_wrap=True, min_width=12)
    table.add_column("at named", justify="right", no_wrap=True, min_width=8)
    table.add_column("miss", justify="right", no_wrap=True, min_width=6)
    for run in runs:
        state = run.state
        name = Text(f"{STATE_MARK[state]} ", style=STATE_STYLE[state]) + Text(run.name, style="bold")
        if run.restarts:
            name += Text(f"  ↻{run.restarts}", style="yellow")
        fraction = run.seen / run.total if run.total else 0.0
        progress = bar(fraction, 14 if wide else 8, STATE_STYLE[state].replace("bold ", ""))
        progress += Text(f" {100 * fraction:3.0f}%", style="bold")
        if run.total:
            progress += Text(f"  {run.seen / 1e6:.2f}/{run.total / 1e6:.2f}M", style="dim")
        loss = Text("-" if run.loss is None else f"{run.loss:.3f}")
        if wide:
            loss += Text(" " + spark(run.losses[len(run.losses) // 20 :], 13), style="magenta")
        speed = "-" if run.rate is None or state != "training" else f"{run.rate:,.0f}/s"
        left = duration((run.total - run.seen) / run.rate) if state == "training" and run.rate else "-"

        rows, running = evals.of(run.name)
        curve, best, named, miss = Text(""), Text("-", style="dim"), "-", "-"
        if rows:
            for kind in sorted({w for _, w, _ in rows}):
                rates = [r["success_rate"] for _, w, r in rows if w == kind]
                curve += Text(f"{kind} ", style="dim")
                curve += Text(spark([0.0, 1.0] + rates, 99)[2:], style="green") + Text("  ")
            n, kind, top = max(rows, key=lambda r: (r[2]["success_rate"], r[0]))
            best = Text(
                f"{100 * top['success_rate']:.0f}%", style="bold green" if top["success_rate"] >= 0.5 else "bold"
            )
            best += Text(f" #{n} {kind}", style="dim")
            grasp = top.get("grasp") or {}
            if grasp.get("closed_nearest_is_target_rate") is not None:
                named = f"{100 * grasp['closed_nearest_is_target_rate']:.0f}%"
            if grasp.get("median_xy_err_target_mm") is not None:
                miss = f"{grasp['median_xy_err_target_mm']:.0f} mm"
        if running:
            curve += Text(f"⟳ {running} running", style="yellow")
        elif not rows:
            curve = Text("no evaluations yet", style="dim")
        table.add_row(name, progress, loss, speed, left, curve, best, named, miss)
    return table


def gpu_table(rows, width):
    columns = 2 if width >= 120 and len(rows) > 4 else 1
    table = Table.grid(expand=True, padding=(0, 3))
    for _ in range(columns):
        table.add_column(ratio=1)
    cells = []
    for index, name, util, used, total, temp in rows:
        style = "green" if util >= 50 else ("yellow" if util >= 5 else "grey50")  # busy is the good state
        cell = Text(f"GPU {index} ", style="bold") + bar(util / 100, 14, style) + Text(f" {util:3.0f}%  ")
        cell += bar(used / total if total else 0, 10, "blue") + Text(f" {used / 1024:4.1f}/{total / 1024:.0f} GB  ")
        cell += Text(f"{temp:.0f}°C  ", style="red" if temp >= 85 else "dim") + Text(
            name.replace("NVIDIA ", ""), style="dim"
        )
        cells.append(cell)
    per_column = -(-len(cells) // columns)
    for r in range(per_column):
        table.add_row(
            *[cells[c * per_column + r] if c * per_column + r < len(cells) else Text("") for c in range(columns)]
        )
    return table


def frame(runs, evals, width, experiments):
    for run in runs:
        run.update()
    counts = {s: sum(r.state == s for r in runs) for s in STATE_MARK}
    title = Text("training monitor", style="bold") + Text(f"  {experiments}  ", style="dim")
    title += Text(datetime.now(UTC).strftime("%H:%M:%S UTC"), style="dim")
    summary = Text("   ").join(
        Text(f"{STATE_MARK[s]} {counts[s]} {s}", style=STATE_STYLE[s]) for s in STATE_MARK if counts[s]
    )
    parts = [Panel(runs_table(runs, evals, width), title=title, subtitle=summary, border_style="grey50")]
    gpu_rows = gpus()
    if gpu_rows:
        parts.append(Panel(gpu_table(gpu_rows, width), title=Text("GPUs", style="bold"), border_style="grey50"))
    legend = "success is closed-loop on unseen layouts · at named = gripper closed at the object the instruction named"
    parts.append(Text(legend + " · miss = median distance to it · Ctrl-C quits", style="dim"))
    return Group(*parts)


def discover(experiments, known):
    names = set()
    for path in glob.glob(os.path.join(experiments, "*.train.log*")):
        names.add(os.path.basename(path).split(".train.log")[0])
    for name in sorted(names - set(known)):
        known[name] = Run(experiments, name)
    return [known[n] for n in sorted(known)]


def main():
    p = argparse.ArgumentParser(description="Live terminal view of training runs and their closed-loop results.")
    p.add_argument("--experiments", default="experiments/pickup")
    p.add_argument("--results", default="results/pickup")
    p.add_argument("--task", default=None, help="only show evaluations of this task (default: all)")
    p.add_argument("--runs", nargs="*", default=None, help="only these runs (substring match)")
    p.add_argument("--interval", type=float, default=2.0, help="seconds between refreshes")
    p.add_argument("--once", action="store_true", help="print one frame and exit")
    p.add_argument("--svg", default=None, help="with --once: also save the frame as an SVG image")
    p.add_argument("--width", type=int, default=None, help="render width in characters (default: the terminal's)")
    args = p.parse_args()

    known = {}
    evals = Evals(args.results, args.task)

    def current():
        runs = discover(args.experiments, known)
        if args.runs:
            runs = [r for r in runs if any(s in r.name for s in args.runs)]
        return runs

    if args.once:
        console = Console(record=bool(args.svg), width=args.width, force_terminal=bool(args.svg) or None)
        console.print(frame(current(), evals, console.width, args.experiments))
        if args.svg:
            console.save_svg(args.svg, title="vla-from-scratch")
        return

    console = Console(width=args.width)
    with Live(console=console, screen=True, auto_refresh=False) as live:
        try:
            while True:
                live.update(frame(current(), evals, console.width, args.experiments), refresh=True)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
