// The grasp map: where the gripper was, seen from above, at the moment it first closed.
//
// Two maps of the same episodes. On the left the centre is the object the instruction named: a
// tight cluster inside the rings is a policy that goes to the right object. On the right the centre
// is whichever object was nearest: tight there but scattered on the left means the policy grasps
// well and picks objects at random; scattered on both means it is not aiming at any object.
// Points keep their identity (the episode seed) between evaluations and glide to their new place.

import { h, s, attr, attrs, setText, setClass, reconcile, animate, onWidth } from "./dom.js";
import { objectGlyph, objectLabel } from "./icons.js";
import { fmtMm, fmtPct, isNum } from "./format.js";
import { median, quantile } from "./scale.js";

const STOPS = [2, 3, 5, 8, 10, 15, 20, 30, 40]; // half-width of a map, in cm
const RINGS = [1, 2]; // cm: about what a grasp tolerates
const FAR_RINGS = [5, 10, 20]; // cm: fainter, to read a miss off the map
const PAD = 12;
const REACH = 18; // px: how close the pointer has to be to a point

/** Episodes as points: offsets (cm) of the gripper from the named and from the nearest object. */
function toPoints(episodes) {
  const out = [];
  for (const e of episodes) {
    const tcp = e.close_tcp_xy;
    const objects = e.close_object_xy;
    if (!Array.isArray(tcp) || !objects || !isNum(tcp[0]) || !isNum(tcp[1])) continue;
    const offset = (name) => {
      const at = objects[name];
      return Array.isArray(at) && isNum(at[0]) && isNum(at[1])
        ? [100 * (tcp[0] - at[0]), 100 * (tcp[1] - at[1])]
        : null;
    };
    out.push({
      seed: e.seed,
      key: e.seed == null ? `#${out.length}` : e.seed,
      target: e.target,
      nearest: e.close_nearest,
      success: !!e.success,
      wrong: !!e.wrong_object,
      step: e.close_step,
      steps: e.steps,
      offsets: [offset(e.target), offset(e.close_nearest)],
    });
  }
  return out;
}

function reasonWithoutPoints(episodes) {
  if (!episodes.length) return "This evaluation has no per-episode data.";
  if (episodes.some((e) => "close_step" in e) && episodes.every((e) => e.close_step == null))
    return "The gripper never closed in these episodes.";
  return "This evaluation has no gripper positions: it was written by an older eval.py. Run the evaluation again to see where the gripper closed.";
}

function autoExtent(points) {
  const distances = points
    .map((p) => p.offsets[0] && Math.hypot(p.offsets[0][0], p.offsets[0][1]))
    .filter(isNum)
    .sort((a, b) => a - b);
  if (!distances.length) return 10;
  const wanted = Math.min(30, Math.max(5, 1.1 * quantile(distances, 0.9))); // 90% of the misses fit
  return STOPS.find((stop) => stop >= wanted) || STOPS[STOPS.length - 1];
}

function createPlot(index, title, caption, events) {
  const frame = s("rect", { class: "gm-frame", rx: 3 });
  const axes = s("path", { class: "gm-axes" });
  const rings = s("g");
  const extentLabel = s("text", { class: "gm-scale", "text-anchor": "end" });
  const points = s("g", { class: "gm-points" });
  const focus = s("circle", { class: "gm-focus", r: 9, display: "none" });
  const overlay = s("rect", { class: "overlay", fill: "transparent" });
  const svg = s(
    "svg",
    { class: "gm-svg", role: "group", tabindex: "0", "aria-label": `${title}. Arrow keys step through the episodes.` },
    frame,
    axes,
    rings,
    extentLabel,
    points,
    focus,
    overlay,
  );
  const stats = h("p", { class: "gm-stats" });
  const el = h(
    "figure",
    { class: "gm" },
    h("figcaption", null, h("b", { text: title }), h("span", { text: caption })),
    svg,
    stats,
  );
  overlay.addEventListener("pointermove", (event) => {
    const box = svg.getBoundingClientRect();
    events.point(index, event.clientX - box.left, event.clientY - box.top);
  });
  overlay.addEventListener("pointerdown", (event) => {
    const box = svg.getBoundingClientRect();
    events.point(index, event.clientX - box.left, event.clientY - box.top);
  });
  overlay.addEventListener("pointerleave", () => events.leave());
  svg.addEventListener("keydown", (event) => events.key(index, event));
  svg.addEventListener("blur", () => events.leave());
  return { el, svg, frame, axes, rings, extentLabel, points, focus, overlay, stats, size: 0, at: new Map() };
}

export function createGrasp() {
  let points = [];
  let size = 0;
  let auto = true;
  let extent = 10;
  let only = null; // an object name: show the episodes that named it
  let active = null; // {key, plot}
  let cancel = null;

  const events = {
    point(index, x, y) {
      let best = null;
      let bestDistance = REACH;
      for (const p of points) {
        if (only && p.target !== only) continue;
        const at = plots[index].at.get(p.key);
        if (!at) continue;
        const d = Math.hypot(at[0] - x, at[1] - y);
        if (d < bestDistance) {
          bestDistance = d;
          best = p;
        }
      }
      setActive(best ? { key: best.key, plot: index } : null);
    },
    leave: () => setActive(null),
    key(index, event) {
      const list = points.filter((p) => (!only || p.target === only) && plots[index].at.has(p.key));
      if (!list.length) return;
      const now = active ? list.findIndex((p) => p.key === active.key) : -1;
      let next;
      if (event.key === "ArrowRight" || event.key === "ArrowDown") next = (now + 1) % list.length;
      else if (event.key === "ArrowLeft" || event.key === "ArrowUp") next = (now - 1 + list.length) % list.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = list.length - 1;
      else if (event.key === "Escape") return setActive(null);
      else return undefined;
      event.preventDefault();
      return setActive({ key: list[next].key, plot: index });
    },
  };

  const plots = [
    createPlot(
      0,
      "From the named object",
      "A tight cluster inside the rings means the gripper goes to the object it was asked for.",
      events,
    ),
    createPlot(
      1,
      "From the nearest object",
      "Tight here but scattered around the named object: it grasps well, at the wrong object.",
      events,
    ),
  ];

  const range = h("input", {
    class: "range",
    type: "range",
    min: 0,
    max: STOPS.length - 1,
    step: 1,
    "aria-label": "Half-width of the maps",
  });
  const extentValue = h("output", { class: "num gm-extent-value" });
  const autoButton = h("button", {
    class: "toggle",
    type: "button",
    text: "Fit",
    "aria-pressed": "true",
    title: "Choose the width that holds 90% of the misses",
  });
  const filter = h("div", { class: "seg gm-filter", role: "group", "aria-label": "Show episodes that named" });
  const controls = h(
    "div",
    { class: "gm-controls" },
    h("label", { class: "gm-extent" }, h("span", { class: "control-label", text: "Width" }), range, extentValue),
    autoButton,
    filter,
  );
  const legend = h(
    "p",
    { class: "gm-legend" },
    h(
      "span",
      { class: "legend-key" },
      s(
        "svg",
        { class: "gm-key", viewBox: "-6 -6 12 12", "aria-hidden": "true" },
        s("circle", { class: "ok-mark", r: 4 }),
      ),
      "success",
    ),
    h(
      "span",
      { class: "legend-key" },
      s(
        "svg",
        { class: "gm-key", viewBox: "-6 -6 12 12", "aria-hidden": "true" },
        s("path", { class: "fail-mark", d: "M-3.6,-3.6L3.6,3.6M-3.6,3.6L3.6,-3.6" }),
      ),
      "failure",
    ),
    h("span", { class: "gm-legend-note", text: "Rings at 1 cm and 2 cm. Marks on the frame are further out." }),
  );
  const note = h("p", { class: "gm-note", hidden: true });
  const tip = h("div", { class: "tip", role: "status", hidden: true });
  const maps = h("div", { class: "gm-maps" }, plots[0].el, plots[1].el);
  const el = h("div", { class: "grasp" }, controls, maps, legend, note, tip);

  function place(plot, offset) {
    if (!offset) return null;
    let [dx, dy] = offset;
    const reach = Math.max(Math.abs(dx), Math.abs(dy));
    const out = reach > extent;
    if (out) {
      dx *= extent / reach; // onto the frame, in the direction of the miss
      dy *= extent / reach;
    }
    const half = plot.size / 2 - PAD;
    return [plot.size / 2 + (dx / extent) * half, plot.size / 2 - (dy / extent) * half, out];
  }

  function layout(glide) {
    if (!size) return;
    const targets = [];
    plots.forEach((plot, index) => {
      plot.size = size;
      const c = size / 2;
      const half = c - PAD;
      attrs(plot.svg, { width: size, height: size, viewBox: `0 0 ${size} ${size}` });
      attrs(plot.frame, { x: PAD, y: PAD, width: 2 * half, height: 2 * half });
      attrs(plot.overlay, { x: 0, y: 0, width: size, height: size });
      attr(plot.axes, "d", `M${PAD},${c}H${size - PAD}M${c},${PAD}V${size - PAD}`);
      attrs(plot.extentLabel, { x: size - PAD - 6, y: size - PAD - 7 });
      setText(plot.extentLabel, `±${extent} cm`);
      reconcile(
        plot.rings,
        [...RINGS, ...FAR_RINGS.filter((r) => r < extent)],
        (r) => r,
        () => s("g", null, s("circle", { class: "gm-ring" }), s("text", { class: "gm-ring-label" })),
        (node, r) => {
          const radius = (r / extent) * half;
          setClass(node, "far", r > RINGS[RINGS.length - 1]);
          attrs(node.firstChild, { cx: c, cy: c, r: radius.toFixed(1) });
          attrs(node.lastChild, { x: (c + radius * 0.72 + 3).toFixed(1), y: (c - radius * 0.72 - 3).toFixed(1) });
          setText(node.lastChild, radius >= 44 ? `${r} cm` : "");
        },
      );
      const drawable = points.filter((p) => p.offsets[index]).sort((a, b) => a.success - b.success); // successes on top
      plot.at = new Map();
      reconcile(
        plot.points,
        drawable,
        (p) => p.key,
        () =>
          s(
            "g",
            { class: "pt" },
            s("circle", { class: "ok-mark", r: 4 }),
            s("path", { class: "fail-mark", d: "M-3.6,-3.6L3.6,3.6M-3.6,3.6L3.6,-3.6" }),
          ),
        (node, p, i, fresh) => {
          const [x, y, out] = place(plot, p.offsets[index]);
          plot.at.set(p.key, [x, y]);
          setClass(node, "ok", p.success);
          setClass(node, "fail", !p.success);
          setClass(node, "out", out);
          setClass(node, "muted", !!only && p.target !== only);
          if (fresh || !node.__at || !glide) {
            node.__at = [x, y];
            attr(node, "transform", `translate(${x.toFixed(1)},${y.toFixed(1)})`);
            if (fresh && glide) node.classList.add("appear");
          } else if (node.__at[0] !== x || node.__at[1] !== y) {
            targets.push({ node, from: node.__at.slice(), to: [x, y] });
          }
        },
      );
      const shown = drawable.filter((p) => !only || p.target === only);
      const distances = shown.map((p) => Math.hypot(p.offsets[index][0], p.offsets[index][1]));
      const within = (cm) =>
        distances.length ? fmtPct(distances.filter((d) => d <= cm).length / distances.length) : "";
      const stats = distances.length
        ? [
            ["median", fmtMm(10 * median(distances))],
            ["within 1 cm", within(1)],
            ["within 2 cm", within(2)],
          ]
        : [];
      reconcile(
        plot.stats,
        stats,
        (item) => item[0],
        (item) => h("span", null, `${item[0]} `, h("b", { class: "num" })),
        (node, item) => setText(node.lastChild, item[1]),
      );
    });
    if (cancel) cancel();
    cancel = null;
    if (targets.length) {
      cancel = animate(
        560,
        (t) => {
          for (const m of targets) {
            m.node.__at = [m.from[0] + (m.to[0] - m.from[0]) * t, m.from[1] + (m.to[1] - m.from[1]) * t];
            m.node.setAttribute("transform", `translate(${m.node.__at[0].toFixed(1)},${m.node.__at[1].toFixed(1)})`);
          }
        },
        () => {
          cancel = null;
        },
      );
    }
    drawActive();
  }

  function setActive(next) {
    if ((active && active.key) === (next && next.key) && (active && active.plot) === (next && next.plot)) return;
    active = next;
    drawActive();
  }

  function drawActive() {
    const p = active && points.find((q) => q.key === active.key);
    plots.forEach((plot) => {
      const at = p && plot.at.get(p.key);
      if (at) attrs(plot.focus, { cx: at[0].toFixed(1), cy: at[1].toFixed(1), display: null });
      else attr(plot.focus, "display", "none");
    });
    if (!p) {
      tip.hidden = true;
      return;
    }
    const miss = (offset) => (offset ? fmtMm(10 * Math.hypot(offset[0], offset[1])) : "unknown");
    tip.replaceChildren(
      h(
        "div",
        { class: `tip-head ${p.success ? "ok" : "fail"}` },
        h("span", { class: "tip-object" }, objectGlyph(p.target), objectLabel(p.target)),
        h("b", { text: p.success ? "success" : p.wrong ? "lifted the wrong object" : "failure" }),
      ),
      h(
        "dl",
        null,
        h("dt", { text: "Seed" }),
        h("dd", { class: "num", text: p.seed == null ? "unknown" : p.seed }),
        h("dt", { text: "Nearest object" }),
        h(
          "dd",
          null,
          p.nearest ? h("span", { class: "tip-object" }, objectGlyph(p.nearest), objectLabel(p.nearest)) : "unknown",
        ),
        h("dt", { text: "Miss to named" }),
        h("dd", { class: "num", text: miss(p.offsets[0]) }),
        h("dt", { text: "Miss to nearest" }),
        h("dd", { class: "num", text: miss(p.offsets[1]) }),
        h("dt", { text: "Closed at step" }),
        h("dd", {
          class: "num",
          text: isNum(p.step) ? `${p.step}${isNum(p.steps) ? ` of ${p.steps}` : ""}` : "unknown",
        }),
      ),
    );
    tip.hidden = false;
    const plot = plots[active.plot];
    const at = plot.at.get(p.key);
    const root = el.getBoundingClientRect();
    const box = plot.svg.getBoundingClientRect();
    const x = box.left - root.left + at[0];
    const y = box.top - root.top + at[1];
    const width = tip.offsetWidth;
    const height = tip.offsetHeight;
    const left = x + 16 + width > root.width ? x - 16 - width : x + 16;
    tip.style.left = `${Math.max(0, Math.min(root.width - width, left)).toFixed(0)}px`;
    tip.style.top = `${Math.max(0, y - height / 2).toFixed(0)}px`;
  }

  function drawControls() {
    range.value = String(Math.max(0, STOPS.indexOf(extent)));
    setText(extentValue, `±${extent} cm`);
    range.setAttribute("aria-valuetext", `${extent} centimetres either side`);
    autoButton.setAttribute("aria-pressed", String(auto));
    const names = [...new Set(points.map((p) => p.target))].filter((name) => name != null).sort();
    filter.hidden = names.length < 2;
    reconcile(
      filter,
      names.length > 1 ? [null, ...names] : [],
      (name) => name || "*",
      (name) => {
        const button = h(
          "button",
          { type: "button", class: "seg-button" },
          name ? [objectGlyph(name), h("span", { class: "sr-only", text: objectLabel(name) })] : "All",
        );
        if (name) button.title = `Episodes that named the ${objectLabel(name)}`;
        button.addEventListener("click", () => {
          only = name;
          setActive(null);
          drawControls();
          layout(false);
        });
        return button;
      },
      (button, name) => attr(button, "aria-pressed", String(only === name)),
    );
  }

  range.addEventListener("input", () => {
    auto = false;
    extent = STOPS[Number(range.value)] || extent;
    drawControls();
    layout(true);
  });
  autoButton.addEventListener("click", () => {
    auto = !auto;
    if (auto) extent = autoExtent(points);
    drawControls();
    layout(true);
  });

  const stopWatching = onWidth(maps, (width) => {
    const columns = width >= 460 ? 2 : 1;
    size = Math.max(160, Math.min(440, Math.floor((width - (columns - 1) * 16) / columns)));
    maps.dataset.columns = String(columns);
    layout(false);
  });

  return {
    el,
    /** episodes: the per_episode list of a results.json; null while it is being fetched. */
    set({ episodes, loading, message }) {
      setClass(el, "loading", !!loading);
      if (message || !episodes) {
        note.hidden = !message;
        setText(note, message || "");
        if (message) {
          points = [];
          setActive(null);
          maps.hidden = controls.hidden = legend.hidden = true;
        }
        return;
      }
      const next = toPoints(episodes);
      const empty = !next.length;
      note.hidden = !empty;
      setText(note, empty ? reasonWithoutPoints(episodes) : "");
      maps.hidden = controls.hidden = legend.hidden = empty;
      const had = points.length > 0;
      points = next;
      if (only && !points.some((p) => p.target === only)) only = null;
      if (auto) extent = autoExtent(points);
      setActive(null);
      drawControls();
      layout(had);
    },
    destroy() {
      stopWatching();
      if (cancel) cancel();
    },
  };
}
