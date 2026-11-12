// Hand-written SVG charts: a line chart that updates in place and shares its cursor with the
// charts stacked above and below it, and the sparkline of the run cards.

import { h, s, attr, attrs, setText, setClass, reconcile, replay, onWidth, uid, reducedMotion } from "./dom.js";
import { scaleLinear, scaleLog, linearTicks, logTicks, logDomain, niceCeil, nearestIndex, lttb } from "./scale.js";
import { DASH, isNum } from "./format.js";

const LEFT = 46; // room for the y tick labels
const RIGHT = 12;

/** Charts in one group show the cursor at the same x, wherever the pointer is. */
export function createGroup(onHover) {
  const charts = new Set();
  return {
    add: (chart) => charts.add(chart),
    remove: (chart) => charts.delete(chart),
    hover(x) {
      for (const chart of charts) chart.showHover(x);
      if (onHover) onHover(x);
    },
  };
}

function linePath(points, sx, sy, dodge) {
  let d = "";
  for (let i = 0; i < points.length; i++) {
    d += `${i ? "L" : "M"}${(sx(points[i].x) + dodge).toFixed(1)},${sy(points[i].y).toFixed(1)}`;
  }
  return d;
}

/**
 * A chart of one or more series against a shared x.
 *
 * options: title, note, height (of the plot), xAxis (draw the x tick labels), xFormat, yFormat,
 *   mode ("linear" | "log"), domain(values, mode) -> [lo, hi] | null, minTop and maxTop (bounds on
 *   the top of a linear axis that starts at zero), context(x | null, items)
 *   -> text before the read-out, quiet (no read-out: the owner shows the values), group, onPick(x),
 *   controls (an element for the header), label.
 * set(data): series [{id, label, color, points [{x, y, lo, hi, key}], line, dots ("filled" | "hollow"),
 *   whiskers, dodge, dashed, width, dim, hidden, value(point) -> text, readout}], xDomain [lo, hi],
 *   marks [{x, label}], selectedX, refLines [{y, label}] (dotted, for a threshold), xAxis (overrides
 *   the option).
 */
export function createChart(options) {
  const o = { height: 150, mode: "linear", xFormat: String, yFormat: String, ...options };
  let mode = o.mode;
  let data = { series: [], xDomain: null, marks: [], selectedX: null, refLines: [] };
  let width = 0;
  let hoverX = null;
  let sx = scaleLinear(0, 1, 0, 1);
  let sy = scaleLinear(0, 1, 0, 1);
  let frame = { top: 8, bottom: 0, x0: LEFT, x1: LEFT };
  let stops = [];
  const drawn = new Map(); // series id -> the points that can be drawn in the current mode
  const lastX = new Map(); // series id -> x of its newest point, to notice a new one
  let entered = false;

  const context = h("span", { class: "ro-context" });
  const items = h("span", { class: "ro-items" });
  const readout = h("div", { class: "chart-readout", "aria-live": "off" }, items, context);
  const note = h("span", { class: "chart-note", text: o.note || "" });
  const head = h(
    "div",
    { class: "chart-head" },
    h("div", { class: "chart-heading" }, h("h3", { class: "chart-title", text: o.title }), note, o.controls || null),
    o.quiet ? null : readout,
  );

  const clipId = uid("clip");
  const clipRect = s("rect");
  const gY = s("g", { class: "axis-y" });
  const gX = s("g", { class: "axis-x" });
  const gRef = s("g", { class: "refs" });
  const gMarks = s("g", { class: "marks" });
  const selected = s("line", { class: "selected-line", display: "none" });
  const gSeries = s("g", { class: "series-layer", "clip-path": `url(#${clipId})` });
  const cross = s("line", { class: "crosshair", display: "none" });
  const gHover = s("g", { class: "hover-dots" });
  const overlay = s("rect", { class: "overlay", fill: "transparent" });
  const svg = s(
    "svg",
    { class: "chart-svg", role: "group", "aria-label": o.label || o.title, tabindex: "0" },
    s("defs", null, s("clipPath", { id: clipId }, clipRect)),
    gY,
    gX,
    gRef,
    gMarks,
    selected,
    gSeries,
    cross,
    gHover,
    overlay,
  );
  const el = h("section", { class: "chart" }, head, svg);

  const visible = () => data.series.filter((sr) => !sr.hidden);

  function yDomain() {
    let lo = Infinity;
    let hi = -Infinity;
    const values = [];
    for (const sr of visible()) {
      for (const p of drawn.get(sr.id) || []) {
        values.push(p.y);
        const top = sr.whiskers && isNum(p.hi) ? Math.max(p.y, p.hi) : p.y;
        if (p.y < lo) lo = p.y;
        if (top > hi) hi = top;
      }
    }
    const custom = o.domain ? o.domain(values, mode) : null;
    if (custom) return custom;
    if (mode === "log") return lo > 0 && hi > 0 ? logDomain(lo, hi) : [0.1, 1];
    for (const ref of data.refLines || []) if (ref.y > hi) hi = ref.y;
    if (!(hi > 0)) hi = 0;
    const top = niceCeil(Math.max(hi * 1.04, o.minTop || 0) || 1, Math.max(2, Math.round(o.height / 44)));
    return [0, o.maxTop ? Math.min(top, o.maxTop) : top];
  }

  function render() {
    if (!width) return;
    const series = visible();
    drawn.clear();
    const xs = new Set();
    for (const sr of data.series) {
      const points = (sr.points || []).filter((p) => isNum(p.x) && isNum(p.y) && (mode !== "log" || p.y > 0));
      drawn.set(sr.id, points);
      if (!sr.hidden) for (const p of points) xs.add(p.x);
    }
    stops = [...xs].sort((a, b) => a - b);

    const hasMarks = !!(data.marks && data.marks.length);
    const top = hasMarks ? 20 : 8;
    const bottom = top + o.height;
    const xAxis = data.xAxis == null ? !!o.xAxis : !!data.xAxis;
    const total = bottom + (xAxis ? 22 : 6);
    const x0 = LEFT;
    const x1 = Math.max(LEFT + 20, width - RIGHT);
    frame = { top, bottom, x0, x1 };
    attrs(svg, { width, height: total, viewBox: `0 0 ${width} ${total}` });
    attrs(clipRect, { x: x0 - 6, y: top - 6, width: x1 - x0 + 12, height: o.height + 12 });
    attrs(overlay, { x: x0, y: top, width: x1 - x0, height: o.height });

    let [dx0, dx1] = data.xDomain || [stops[0] || 0, stops[stops.length - 1] || 1];
    if (!(dx1 > dx0)) dx1 = dx0 + 1;
    sx = scaleLinear(dx0, dx1, x0, x1);
    const [dy0, dy1] = yDomain();
    sy = mode === "log" ? scaleLog(dy0, dy1, bottom, top) : scaleLinear(dy0, dy1, bottom, top);

    const yTicks = mode === "log" ? logTicks(dy0, dy1) : linearTicks(dy0, dy1, Math.max(2, Math.round(o.height / 44)));
    reconcile(
      gY,
      yTicks,
      (v) => v,
      () =>
        s("g", null, s("line", { class: "grid" }), s("text", { class: "tick", "text-anchor": "end", dy: "0.32em" })),
      (node, v) => {
        const y = Math.round(sy(v)) + 0.5;
        attrs(node.firstChild, { x1: x0, x2: x1, y1: y, y2: y });
        setClass(node.firstChild, "base", mode !== "log" && v === dy0);
        attrs(node.lastChild, { x: x0 - 8, y });
        setText(node.lastChild, o.yFormat(v));
      },
    );

    const xTicks = linearTicks(dx0, dx1, Math.max(2, Math.floor((x1 - x0) / 84)));
    reconcile(
      gX,
      xAxis ? xTicks : [],
      (v) => v,
      () => s("g", null, s("line", { class: "tick-line" }), s("text", { class: "tick", "text-anchor": "middle" })),
      (node, v) => {
        const x = Math.round(sx(v)) + 0.5;
        attrs(node.firstChild, { x1: x, x2: x, y1: bottom, y2: bottom + 4 });
        attrs(node.lastChild, { x, y: bottom + 16 });
        setText(node.lastChild, o.xFormat(v));
      },
    );

    reconcile(
      gRef,
      (data.refLines || []).filter((r) => isNum(r.y) && r.y >= dy0 && r.y <= dy1),
      (r) => r.y,
      () => s("g", { class: "ref" }, s("line"), s("text", { "text-anchor": "end" })),
      (node, r) => {
        const y = Math.round(sy(r.y)) + 0.5;
        attrs(node.firstChild, { x1: x0, x2: x1, y1: y, y2: y });
        attrs(node.lastChild, { x: x1 - 2, y: y - 4 });
        setText(node.lastChild, r.label || "");
      },
    );

    // Labels that would run into each other are left out; the selected mark is placed first.
    const taken = [];
    const labelled = new Set();
    const marks = hasMarks ? data.marks : [];
    for (const m of [...marks.filter((k) => k.x === data.selectedX), ...marks.filter((k) => k.x !== data.selectedX)]) {
      const x = sx(m.x);
      const half = 3.2 * String(m.label).length + 4;
      if (taken.some(([a, b]) => x + half > a && x - half < b)) continue;
      taken.push([x - half, x + half]);
      labelled.add(m.x);
    }
    reconcile(
      gMarks,
      marks,
      (m) => m.label,
      () => s("g", { class: "mark" }, s("line"), s("text", { "text-anchor": "middle" })),
      (node, m) => {
        const x = Math.round(sx(m.x)) + 0.5;
        attrs(node.firstChild, { x1: x, x2: x, y1: top - 5, y2: top + 4 });
        attrs(node.lastChild, { x, y: top - 8 });
        setText(node.lastChild, labelled.has(m.x) ? m.label : "");
        setClass(node, "picked", m.x === data.selectedX);
      },
    );

    const pick = data.selectedX;
    if (isNum(pick) && pick >= dx0 && pick <= dx1) {
      const x = Math.round(sx(pick)) + 0.5;
      attrs(selected, { x1: x, x2: x, y1: top - (hasMarks ? 5 : 0), y2: bottom, display: null });
    } else {
      attr(selected, "display", "none");
    }

    const plotWidth = x1 - x0;
    reconcile(
      gSeries,
      series,
      (sr) => sr.id,
      () =>
        s(
          "g",
          { class: "series" },
          s("g", { class: "whiskers" }),
          s("path", { class: "line", fill: "none" }),
          s("g", { class: "dots" }),
          s("circle", { class: "pulse", r: 4 }),
        ),
      (node, sr, index, fresh) => {
        const [whiskers, line, dots, pulse] = node.children;
        const dodge = sr.dodge || 0;
        const points = drawn.get(sr.id);
        const thin = points.length > 2 * plotWidth ? lttb(points, Math.max(200, Math.round(1.5 * plotWidth))) : points;
        node.style.setProperty("--c", sr.color);
        setClass(node, "dim", sr.dim);
        attrs(line, {
          d: sr.line !== false && thin.length > 1 ? linePath(thin, sx, sy, dodge) : "",
          "stroke-width": sr.width || 2,
          "stroke-dasharray": sr.dashed ? "5 4" : null,
          pathLength: sr.dashed ? null : 1,
        });
        if (fresh && !entered && !sr.dashed && !reducedMotion()) {
          line.classList.add("draw-in");
          line.addEventListener("animationend", () => line.classList.remove("draw-in"), { once: true });
        }
        reconcile(
          whiskers,
          sr.whiskers ? points.filter((p) => isNum(p.lo) && isNum(p.hi)) : [],
          (p) => (p.key == null ? p.x : p.key),
          () => s("path", { class: "whisker", fill: "none" }),
          (mark, p) => {
            const x = (sx(p.x) + dodge).toFixed(1);
            const cap = (sx(p.x) + dodge - 3).toFixed(1);
            const a = sy(p.lo).toFixed(1);
            const b = sy(p.hi).toFixed(1);
            attr(mark, "d", `M${x},${a}V${b}M${cap},${a}h6M${cap},${b}h6`);
          },
        );
        reconcile(
          dots,
          sr.dots ? points : [],
          (p) => (p.key == null ? p.x : p.key),
          () => s("circle", { class: "dot", r: 3.5 }),
          (dot, p) => {
            attrs(dot, { cx: (sx(p.x) + dodge).toFixed(1), cy: sy(p.y).toFixed(1) });
            setClass(dot, "hollow", sr.dots === "hollow");
            setClass(dot, "picked", p.x === data.selectedX);
          },
        );
        const last = points[points.length - 1];
        if (last) {
          const before = lastX.get(sr.id);
          if (before !== undefined && last.x > before) {
            attrs(pulse, { cx: (sx(last.x) + dodge).toFixed(1), cy: sy(last.y).toFixed(1) });
            replay(pulse, "go");
          }
          lastX.set(sr.id, last.x);
        }
      },
    );
    if (series.some((sr) => (drawn.get(sr.id) || []).length)) entered = true;
    drawHover();
  }

  function drawHover() {
    const hovering = hoverX != null;
    if (hovering) {
      const x = Math.round(sx(hoverX)) + 0.5;
      attrs(cross, { x1: x, x2: x, y1: frame.top, y2: frame.bottom, display: null });
    } else {
      attr(cross, "display", "none");
    }
    const active = visible().map((sr) => {
      const points = drawn.get(sr.id) || [];
      const index = hovering ? nearestIndex(points, hoverX) : points.length - 1;
      return { sr, point: points[index] || null };
    });
    reconcile(
      gHover,
      hovering ? active.filter((it) => it.point) : [],
      (it) => it.sr.id,
      () => s("circle", { class: "hover-dot", r: 4.5 }),
      (dot, it) => {
        dot.style.setProperty("--c", it.sr.color);
        attrs(dot, { cx: (sx(it.point.x) + (it.sr.dodge || 0)).toFixed(1), cy: sy(it.point.y).toFixed(1) });
      },
    );
    setText(context, o.context ? o.context(hovering ? hoverX : null, active) : "");
    reconcile(
      items,
      o.quiet ? [] : active.filter((it) => it.sr.readout !== false),
      (it) => it.sr.id,
      () =>
        h(
          "span",
          { class: "ro" },
          h("i", { class: "key" }),
          h("span", { class: "ro-label" }),
          h("b", { class: "ro-value num" }),
        ),
      (node, it) => {
        node.style.setProperty("--c", it.sr.color);
        setClass(node.children[0], "dashed", it.sr.dashed);
        setClass(node.children[0], "hollow", it.sr.dots === "hollow");
        setText(node.children[1], it.sr.label);
        setText(node.children[2], it.point ? (it.sr.value ? it.sr.value(it.point) : o.yFormat(it.point.y)) : DASH);
      },
    );
  }

  function nearestStop(x) {
    if (!stops.length) return null;
    let lo = 0;
    let hi = stops.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (stops[mid] < x) lo = mid;
      else hi = mid;
    }
    return Math.abs(stops[lo] - x) <= Math.abs(stops[hi] - x) ? lo : hi;
  }

  const chart = {
    el,
    set(next) {
      data = { ...data, ...next };
      render();
    },
    setMode(next) {
      if (next === mode) return;
      mode = next;
      render();
    },
    setNote(text) {
      setText(note, text);
    },
    showHover(x) {
      if (x === hoverX) return;
      hoverX = x;
      drawHover();
    },
    destroy() {
      stopWatching();
      if (o.group) o.group.remove(chart);
    },
  };

  const hover = (x) => (o.group ? o.group.hover(x) : chart.showHover(x));
  const stopAt = (event) => {
    const index = nearestStop(sx.invert(event.clientX - svg.getBoundingClientRect().left));
    return index == null ? null : stops[index];
  };
  overlay.addEventListener("pointermove", (event) => hover(stopAt(event)));
  overlay.addEventListener("pointerdown", (event) => hover(stopAt(event)));
  overlay.addEventListener("pointerleave", () => hover(null));
  overlay.addEventListener("click", (event) => {
    const x = stopAt(event);
    if (x != null && o.onPick) o.onPick(x);
  });
  svg.addEventListener("keydown", (event) => {
    if (!stops.length) return;
    const at = hoverX == null ? null : nearestStop(hoverX);
    const jump = event.shiftKey ? Math.max(1, Math.round(stops.length / 10)) : 1;
    let index;
    if (event.key === "ArrowLeft") index = at == null ? stops.length - 1 : Math.max(0, at - jump);
    else if (event.key === "ArrowRight") index = at == null ? stops.length - 1 : Math.min(stops.length - 1, at + jump);
    else if (event.key === "Home") index = 0;
    else if (event.key === "End") index = stops.length - 1;
    else if (event.key === "Escape") return hover(null);
    else if ((event.key === "Enter" || event.key === " ") && hoverX != null && o.onPick) {
      event.preventDefault();
      return o.onPick(hoverX);
    } else return undefined;
    event.preventDefault();
    return hover(stops[index]);
  });
  svg.addEventListener("focus", () => readout.setAttribute("aria-live", "polite"));
  svg.addEventListener("blur", () => {
    readout.setAttribute("aria-live", "off");
    hover(null);
  });

  if (o.group) o.group.add(chart);
  const stopWatching = onWidth(svg, (w) => {
    width = w;
    render();
  });
  return chart;
}

/**
 * A sparkline that stretches with its container. set([{color, points [[x, y]], dots, width}],
 * {yMin, yMax, log}); the last point of each series carries a dot. `baseline` draws the line of
 * yMin, for series that are read against zero.
 */
export function createSpark(label, height = 28, baseline = false) {
  const lines = s("g");
  const svg = s(
    "svg",
    {
      class: "spark",
      viewBox: `0 0 100 ${height}`,
      width: "100%",
      height,
      preserveAspectRatio: "none",
      role: "img",
      "aria-label": label,
    },
    baseline ? s("path", { class: "spark-base", d: `M0,${height - 4}H100` }) : null,
    lines,
  );
  return {
    el: svg,
    set(series, scale = {}) {
      let x0 = Infinity;
      let x1 = -Infinity;
      let y0 = Infinity;
      let y1 = -Infinity;
      const log = !!scale.log;
      const clean = series.map((sr) => ({
        ...sr,
        points: sr.points.filter((p) => isNum(p[0]) && isNum(p[1]) && (!log || p[1] > 0)),
      }));
      for (const sr of clean) {
        for (const [x, y] of sr.points) {
          if (x < x0) x0 = x;
          if (x > x1) x1 = x;
          if (y < y0) y0 = y;
          if (y > y1) y1 = y;
        }
      }
      if (isNum(scale.yMin)) y0 = scale.yMin;
      if (isNum(scale.yMax)) y1 = scale.yMax;
      if (!(x1 > x0)) x1 = x0 + 1;
      if (!(y1 > y0)) y1 = y0 + (log ? y0 : 1) || 1;
      const sx = scaleLinear(x0, x1, 3, 97);
      const sy = log ? scaleLog(y0, y1, height - 4, 4) : scaleLinear(y0, y1, height - 4, 4);
      reconcile(
        lines,
        clean,
        (sr, i) => sr.id || i,
        () =>
          s(
            "g",
            null,
            s("path", { class: "spark-line", fill: "none" }),
            s("path", { class: "spark-dots", fill: "none" }),
          ),
        (node, sr) => {
          node.style.setProperty("--c", sr.color);
          const at = sr.points.map(([x, y]) => `${sx(x).toFixed(2)},${sy(y).toFixed(2)}`);
          attr(node.firstChild, "d", at.length > 1 ? `M${at.join("L")}` : "");
          attr(node.firstChild, "stroke-width", sr.width || 1.5);
          const dotted = sr.dots ? at : at.slice(-1);
          attr(node.lastChild, "d", dotted.map((p) => `M${p}h0`).join(""));
          attr(node.lastChild, "stroke-width", sr.dots ? 4 : 5);
        },
      );
    },
  };
}
