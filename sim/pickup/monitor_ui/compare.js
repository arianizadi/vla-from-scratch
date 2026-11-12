// Several runs on shared axes: closed-loop success, how often the gripper closed at the named
// object, and the loss. The legend is also the read-out, and its rows switch series on and off.

import { h, setText, setClass, attr, reconcile, memory } from "./dom.js";
import { store } from "./store.js";
import { replaceParams, runHref } from "./route.js";
import { createChart, createGroup } from "./charts.js";
import { segmented } from "./controls.js";
import { stateIcon, STATE_LABEL } from "./icons.js";
import {
  summaryOf,
  checkpointPositions,
  lossPoints,
  evalSeries,
  intervalOf,
  signature,
  WEIGHT_LABEL,
} from "./model.js";
import { fmtSamples, fmtAxisSamples, fmtAxisNumber, fmtLoss, fmtPct, DASH } from "./format.js";
import { nearestIndex } from "./scale.js";

const SLOTS = 8; // the chart colours that stay distinguishable from each other
const pctTick = (v) => `${+(100 * v).toFixed(1)}%`;

export function createCompare(route) {
  let slots = new Array(SLOTS).fill(null); // run names; the index is the colour, kept while a run stays picked
  let fromLink = false;
  let weights = "ema";
  let lossMode = memory.get("loss-scale", "log") === "linear" ? "linear" : "log";
  const off = new Set(); // runs switched off in the legend
  let emphasis = null;
  let hoverX = null;
  let drawn = "";
  let prepared = []; // [{name, slot, color, loss, success, named}] of the picked runs that exist

  /** Take the picked runs and the weights from the parameters of the hash. */
  function read(params) {
    weights = params.get("w") === "raw" ? "raw" : "ema";
    slots = new Array(SLOTS).fill(null);
    fromLink = params.has("runs");
    (params.get("runs") || "")
      .split(",")
      .filter(Boolean)
      .slice(0, SLOTS)
      .forEach((name, i) => {
        slots[i] = name;
      });
  }
  read(route.params);

  const group = createGroup((x) => {
    hoverX = x;
    drawValues();
  });
  const charts = {
    success: createChart({
      title: "Closed-loop success",
      label: "Closed-loop success per checkpoint of the picked runs.",
      height: 180,
      minTop: 0.1,
      maxTop: 1,
      yFormat: pctTick,
      quiet: true,
      group,
    }),
    named: createChart({
      title: "Closed at the named object",
      label: "How often the gripper closed nearest the named object, for the picked runs.",
      height: 140,
      minTop: 0.5,
      maxTop: 1,
      yFormat: pctTick,
      quiet: true,
      group,
    }),
    loss: createChart({
      title: "Loss",
      label: "Training loss of the picked runs against samples seen.",
      height: 180,
      mode: lossMode,
      yFormat: fmtAxisNumber,
      xAxis: true,
      xFormat: fmtAxisSamples,
      quiet: true,
      group,
    }),
  };

  const weightsPicker = segmented({
    label: "Weights of the evaluations",
    options: [
      { id: "ema", label: "EMA weights" },
      { id: "raw", label: "raw weights" },
    ],
    value: weights,
    onChange(id) {
      weights = id;
      save();
      update();
    },
  });
  const scale = segmented({
    label: "Scale of the loss axis",
    options: [
      { id: "log", label: "log loss" },
      { id: "linear", label: "linear loss" },
    ],
    value: lossMode,
    onChange(id) {
      lossMode = id;
      memory.set("loss-scale", id);
      charts.loss.setMode(id);
    },
  });

  const picker = h("div", { class: "picker", role: "group", "aria-label": "Runs to compare" });
  const pickerNote = h("p", { class: "section-note picker-note" });
  const context = h("span", { class: "legend-context" });
  const legendBody = h("tbody");
  const legend = h(
    "table",
    { class: "legend-table" },
    h("caption", {
      class: "sr-only",
      text: "The picked runs and their values at the cursor. A run's button shows or hides its curves.",
    }),
    h(
      "thead",
      null,
      h(
        "tr",
        null,
        h("th", { scope: "col" }, context),
        h("th", { scope: "col", class: "numeric", text: "Success" }),
        h("th", { scope: "col", class: "numeric", text: "At named object" }),
        h("th", { scope: "col", class: "numeric", text: "Loss" }),
      ),
    ),
    legendBody,
  );
  const empty = h("div", { class: "empty", hidden: true });
  const plots = h(
    "div",
    { class: "compare-plots" },
    charts.success.el,
    charts.named.el,
    charts.loss.el,
    h("p", { class: "axis-caption", text: "Training samples seen." }),
  );
  const el = h(
    "div",
    { class: "view compare" },
    h(
      "section",
      { class: "panel", "aria-label": "Runs to compare" },
      h(
        "div",
        { class: "panel-head" },
        h("h1", { class: "panel-title", text: "Compare runs" }),
        weightsPicker.el,
        scale.el,
      ),
      picker,
      pickerNote,
    ),
    h(
      "section",
      { class: "panel compare-main", "aria-label": "Curves of the picked runs" },
      h("div", { class: "table-scroll" }, legend),
      plots,
    ),
    empty,
  );

  function save() {
    replaceParams({ runs: slots.filter(Boolean).join(","), w: weights });
  }

  function toggle(name) {
    const at = slots.indexOf(name);
    if (at >= 0) {
      slots[at] = null;
      off.delete(name);
    } else {
      const free = slots.indexOf(null);
      if (free < 0) return;
      slots[free] = name;
    }
    fromLink = true;
    save();
    update();
  }

  function drawValues() {
    const hovering = hoverX != null;
    setText(context, hovering ? `at ${fmtSamples(hoverX)} samples` : "latest values");
    reconcile(
      legendBody,
      prepared,
      (p) => p.name,
      (p) => {
        const button = h(
          "button",
          { type: "button", class: "legend-toggle" },
          h("i", { class: "key" }),
          h("span", { class: "id", text: p.name }),
        );
        button.addEventListener("click", () => {
          if (off.has(p.name)) off.delete(p.name);
          else off.add(p.name);
          update();
        });
        const open = h("a", { class: "row-link", href: runHref(p.name), text: "open", "aria-label": `Open ${p.name}` });
        const row = h(
          "tr",
          null,
          h("th", { scope: "row" }, button, h("span", { class: "only-note", hidden: true }), open),
          h("td", { class: "numeric" }),
          h("td", { class: "numeric" }),
          h("td", { class: "numeric" }),
        );
        const lift = (name) => {
          if (emphasis === name) return;
          emphasis = name;
          update();
        };
        row.addEventListener("pointerenter", () => lift(p.name));
        row.addEventListener("pointerleave", () => lift(null));
        button.addEventListener("focus", () => lift(p.name));
        button.addEventListener("blur", () => lift(null));
        return row;
      },
      (row, p) => {
        const button = row.firstChild.firstChild;
        const shown = !off.has(p.name);
        button.style.setProperty("--c", p.color);
        attr(button, "aria-pressed", String(shown));
        attr(button, "title", shown ? "Hide this run's curves" : "Show this run's curves");
        setClass(row, "off", !shown);
        const only = row.firstChild.children[1];
        only.hidden = !p.only;
        setText(only, p.only ? `${WEIGHT_LABEL[p.only]} only` : "");
        attr(
          only,
          "title",
          p.only
            ? `This run has no evaluations with ${WEIGHT_LABEL[weights]} weights: these are its ${WEIGHT_LABEL[p.only]} ones`
            : null,
        );
        const at = (points) => {
          if (!points.length) return null;
          return points[hovering ? nearestIndex(points, hoverX) : points.length - 1];
        };
        const evaluation = (cell, points, format) => {
          const point = at(points);
          cell.replaceChildren(
            point ? h("b", { class: "num", text: format(point) }) : DASH,
            point ? h("span", { class: "cell-note", text: `ckpt ${point.evaluation.checkpoint}` }) : "",
          );
        };
        evaluation(row.children[1], p.success, (point) => fmtPct(point.y));
        evaluation(row.children[2], p.named, (point) => fmtPct(point.y));
        const lossPoint = at(p.loss);
        const beyond = hovering && lossPoint && hoverX > p.loss[p.loss.length - 1].x * 1.02;
        row.children[3].replaceChildren(
          lossPoint && !beyond ? h("b", { class: "num", text: fmtLoss(lossPoint.y) }) : DASH,
        );
      },
    );
  }

  function update() {
    const snapshot = store.snapshot;
    if (!snapshot) {
      empty.hidden = false;
      empty.replaceChildren(h("p", { class: "empty-title", text: "Connecting to the monitor" }));
      el.children[0].hidden = el.children[1].hidden = true;
      return;
    }
    const runs = snapshot.runs;
    el.children[0].hidden = !runs.length;
    if (!fromLink && runs.length) {
      // Nothing chosen yet: start with the runs that have the best closed-loop results.
      const ranked = runs
        .map((run) => ({ run, best: summaryOf(run).best }))
        .sort(
          (a, b) =>
            (b.best ? b.best.success_rate : -1) - (a.best ? a.best.success_rate : -1) ||
            (a.run.name < b.run.name ? -1 : 1),
        );
      slots = new Array(SLOTS).fill(null);
      ranked.slice(0, 4).forEach((item, i) => {
        slots[i] = item.run.name;
      });
      fromLink = true;
    }
    const byName = new Map(runs.map((run) => [run.name, run]));
    const full = !slots.includes(null);

    reconcile(
      picker,
      runs.slice().sort((a, b) => (a.name < b.name ? -1 : 1)),
      (run) => run.name,
      (run) => {
        const button = h(
          "button",
          { type: "button", class: "pick" },
          h("i", { class: "key" }),
          h("span", { class: "pick-state" }),
          h("span", { class: "id", text: run.name }),
          h("span", { class: "pick-best num" }),
        );
        button.addEventListener("click", () => toggle(run.name));
        return button;
      },
      (button, run) => {
        const slot = slots.indexOf(run.name);
        const best = summaryOf(run).best;
        attr(button, "aria-pressed", String(slot >= 0));
        button.disabled = slot < 0 && full;
        button.style.setProperty("--c", slot >= 0 ? `var(--cat-${slot + 1})` : "transparent");
        if (button.dataset.state !== run.state) {
          button.dataset.state = run.state;
          button.children[1].replaceChildren(stateIcon(run.state));
        }
        setText(button.children[3], best ? fmtPct(best.success_rate) : "");
        attr(
          button,
          "title",
          `${run.name}, ${STATE_LABEL[run.state]}${best ? `, best success ${fmtPct(best.success_rate)}` : ", no evaluations yet"}`,
        );
      },
    );
    setText(
      pickerNote,
      full
        ? `${SLOTS} runs is as many as the chart colours can keep apart. Unpick one to add another.`
        : "The figure after a name is the best closed-loop success of that run.",
    );

    const picked = slots
      .map((name, slot) => ({ name, slot, run: name ? byName.get(name) : null }))
      .filter((p) => p.run);
    const mark = [
      picked.map((p) => `${p.slot}:${signature(p.run)}`).join("§"),
      weights,
      [...off].join(","),
      emphasis,
    ].join("|");
    if (mark !== drawn) {
      drawn = mark;
      prepared = picked.map((p) => {
        const summary = summaryOf(p.run);
        const positions = checkpointPositions(p.run, summary.evaluations);
        // A run that was evaluated with one kind of weights only is drawn with those, and says so.
        const has = (kind) => summary.evaluations.some((e) => e.weights === kind);
        const other = weights === "ema" ? "raw" : "ema";
        const shown = !has(weights) && has(other) ? other : weights;
        return {
          name: p.name,
          slot: p.slot,
          color: `var(--cat-${p.slot + 1})`,
          only: shown === weights ? null : shown,
          loss: lossPoints(p.run),
          success: evalSeries(summary.evaluations, positions, shown, (e) => e.success_rate).map((point) => {
            const interval = intervalOf(point.evaluation);
            return interval ? { ...point, lo: interval[0], hi: interval[1] } : point;
          }),
          named: evalSeries(summary.evaluations, positions, shown, (e) => e.at_named_object),
        };
      });
      let end = 0;
      for (const p of prepared) {
        if (off.has(p.name)) continue;
        for (const points of [p.loss, p.success, p.named])
          if (points.length && points[points.length - 1].x > end) end = points[points.length - 1].x;
      }
      const xDomain = [0, (end || 1) * 1.02];
      const series = (key, more) =>
        prepared.map((p) => ({
          id: p.name,
          label: p.name,
          color: p.color,
          points: p[key],
          hidden: off.has(p.name),
          dim: emphasis != null && emphasis !== p.name,
          ...more(p),
        }));
      charts.success.set({
        xDomain,
        series: series("success", (p) => ({
          dots: p.only ? "hollow" : "filled",
          whiskers: emphasis === p.name || prepared.length === 1,
        })),
      });
      charts.named.set({ xDomain, series: series("named", (p) => ({ dots: p.only ? "hollow" : "filled" })) });
      charts.loss.set({ xDomain, series: series("loss", () => ({ width: 1.6 })) });
    }
    drawValues();

    const some = prepared.length > 0;
    el.children[1].hidden = !some;
    empty.hidden = some;
    if (!runs.length) {
      empty.replaceChildren(
        h("p", { class: "empty-title", text: "No runs yet" }),
        h("p", { text: "Runs can be compared here once their training logs exist." }),
      );
    } else if (!some) {
      empty.replaceChildren(
        h("p", { class: "empty-title", text: "Pick runs to compare" }),
        h("p", {
          text: `Their success, their rate of closing at the named object and their loss are drawn on shared axes, for ${WEIGHT_LABEL[weights]} weights.`,
        }),
      );
    }
  }

  return {
    el,
    update,
    /** The hash changed while this view was open (a pasted link, the back button). */
    route(next) {
      read(next.params);
      off.clear();
      weightsPicker.set(weights);
      update();
    },
    destroy() {
      for (const chart of Object.values(charts)) chart.destroy();
    },
  };
}
