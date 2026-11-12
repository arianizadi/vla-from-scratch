"""Top-down picture, in the terminal, of where a policy closed its gripper.

Each dot is one evaluation episode: the gripper's position at the moment it first closed, drawn
relative to an object at the centre. Two maps are shown side by side:

- relative to the object the instruction named: a tight cluster at the centre is a policy that goes
  to the right object; the rings are 1 cm and 2 cm, about what a grasp tolerates.
- relative to whichever object was nearest: if this cluster is tight while the first map is
  scattered, the policy grasps well but picks objects at random. If both are scattered, it is not
  aiming at any object.

    python -m sim.pickup.grasp_map results/pickup/<run>/<task>/results.json
    python -m sim.pickup.grasp_map results/pickup/<run>/<task>/ckpt_*_ema/results.json --extent_cm 6
"""

import argparse
import json
import math
import statistics

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

BRAILLE = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))  # dot bit by [row][column] in a cell
RANK = {"grey42": 0, "red": 1, "green": 2, "bold white": 3}


class Canvas:
    """Braille dot canvas: 2 x 4 dots per character, so dots are square when cols = 2 * rows."""

    def __init__(self, cols, rows):
        self.cols, self.rows = cols, rows
        self.bits = [[0] * cols for _ in range(rows)]
        self.style = [[None] * cols for _ in range(rows)]

    def dot(self, x, y, style):
        """x, y in [-1, 1], y up. Points outside are drawn on the border."""
        px = round((min(max(x, -1.0), 1.0) * 0.5 + 0.5) * (2 * self.cols - 1))
        py = round((0.5 - min(max(y, -1.0), 1.0) * 0.5) * (4 * self.rows - 1))
        col, row = px // 2, py // 4
        self.bits[row][col] |= BRAILLE[py % 4][px % 2]
        if self.style[row][col] is None or RANK[style] > RANK[self.style[row][col]]:
            self.style[row][col] = style

    def circle(self, radius, style):
        for i in range(max(24, int(radius * 400))):
            a = 2 * math.pi * i / max(24, int(radius * 400))
            if radius * max(abs(math.cos(a)), abs(math.sin(a))) <= 1:
                self.dot(radius * math.cos(a), radius * math.sin(a), style)

    def render(self):
        text = Text()
        for row in range(self.rows):
            for col in range(self.cols):
                bits = self.bits[row][col]
                text.append(chr(0x2800 + bits) if bits else " ", style=self.style[row][col])
            text.append("\n")
        return text


def offsets(episodes, reference):
    """(dx, dy, success) of the gripper relative to the named or the nearest object, per episode."""
    out = []
    for e in episodes:
        tcp, objects = e.get("close_tcp_xy"), e.get("close_object_xy")
        if not tcp or not objects:
            continue
        name = e["target"] if reference == "named" else e["close_nearest"]
        out.append((tcp[0] - objects[name][0], tcp[1] - objects[name][1], bool(e["success"])))
    return out


def draw(points, extent, rows):
    canvas = Canvas(2 * rows, rows)
    for ring in (0.01, 0.02):
        canvas.circle(ring / extent, "grey42")
    for dx, dy, ok in sorted(points, key=lambda p: p[2]):  # successes on top
        canvas.dot(dx / extent, dy / extent, "green" if ok else "red")
    canvas.dot(0.0, 0.0, "bold white")
    return canvas.render()


def describe(points):
    if not points:
        return Text("no data", style="dim")
    dist = [math.hypot(dx, dy) for dx, dy, _ in points]
    text = Text(f"median {1000 * statistics.median(dist):.0f} mm", style="bold")
    for limit in (0.01, 0.02):
        text.append(f"   within {1000 * limit:.0f} mm: {100 * sum(d <= limit for d in dist) / len(dist):.0f}%")
    return text


def panel(path, extent_cm, rows):
    with open(path) as f:
        results = json.load(f)
    episodes = results.get("per_episode", [])
    closed = [e for e in episodes if e.get("close_tcp_xy")]
    title = Text(path, style="bold")
    if not closed:
        note = "No gripper positions in this file. Re-run the evaluation with the current eval.py."
        if episodes and all(e.get("close_step") is None for e in episodes if "close_step" in e):
            note = "The gripper never closed in these episodes."
        return Panel(Text(note, style="yellow"), title=title, border_style="grey50")
    named, nearest = offsets(closed, "named"), offsets(closed, "nearest")
    if extent_cm is None:  # wide enough for 90% of the misses to the named object, at least 5 cm
        dist = sorted(math.hypot(dx, dy) for dx, dy, _ in named)
        extent_cm = min(30.0, max(5.0, 110.0 * dist[int(0.9 * (len(dist) - 1))]))
    extent = extent_cm / 100.0
    grid = Table.grid(padding=(0, 4))
    grid.add_column()
    grid.add_column()
    grid.add_row(
        Text("relative to the named object", style="bold"), Text("relative to the nearest object", style="bold")
    )
    grid.add_row(draw(named, extent, rows), draw(nearest, extent, rows))
    grid.add_row(describe(named), describe(nearest))
    n, k = len(episodes), sum(bool(e["success"]) for e in episodes)
    at_named = sum(e["close_nearest"] == e["target"] for e in closed)
    head = Text(f"success {k}/{n} = {100 * k / n:.0f}%", style="bold green" if k / n >= 0.5 else "bold")
    head.append(f"   gripper closed in {len(closed)}/{n}", style="default")
    head.append(f"   nearest object was the named one in {at_named}/{len(closed)}", style="default")
    legend = Text("each map spans ", style="dim")
    legend.append(f"±{extent_cm:.0f} cm", style="bold")
    legend.append(" · rings at 1 cm and 2 cm · ", style="dim")
    legend.append("● success", style="green")
    legend.append("  ")
    legend.append("● failure", style="red")
    legend.append(" · dots on the border are further out", style="dim")
    return Panel(Group(head, Text(""), grid, legend), title=title, border_style="grey50")


def main():
    p = argparse.ArgumentParser(description="Top-down terminal map of where the gripper closed.")
    p.add_argument("results", nargs="+", help="results.json files written by eval.py")
    p.add_argument("--extent_cm", type=float, default=None, help="half-width of each map (default: fit the data)")
    p.add_argument("--rows", type=int, default=16, help="map height in terminal rows")
    p.add_argument("--svg", default=None, help="also save the output as an SVG image")
    args = p.parse_args()
    console = Console(record=bool(args.svg), force_terminal=bool(args.svg) or None)
    for path in args.results:
        console.print(panel(path, args.extent_cm, args.rows))
    if args.svg:
        console.save_svg(args.svg, title="where the gripper closed")


if __name__ == "__main__":
    main()
