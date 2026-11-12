// One run: the loss and the closed-loop results on one shared axis of training samples, next to
// the inspector of a single checkpoint, above the table of all its evaluations.

import { h, setText, attr, replay, tweenNumber, memory } from "./dom.js";
import { store, runByName } from "./store.js";
import { replaceParams } from "./route.js";
import { createChart, createGroup } from "./charts.js";
import { createInspector } from "./inspector.js";
import { createEvalTable } from "./evaltable.js";
import { segmented } from "./controls.js";
import { stateIcon, backIcon, restartIcon, STATE_LABEL } from "./icons.js";
import {
  tasksOf,
  evalsOf,
  bestOf,
  checkpointPositions,
  lossPoints,
  evalSeries,
  intervalOf,
  progressOf,
  signature,
  WEIGHTS,
  WEIGHT_LABEL,
} from "./model.js";
import {
  fmtSamples,
  fmtAxisSamples,
  fmtAxisNumber,
  fmtLoss,
  fmtPct,
  fmtMm,
  fmtRate,
  fmtSpan,
  fmtLearningRate,
  fmtInt,
  isNum,
  DASH,
} from "./format.js";
import { niceCeil, quantile } from "./scale.js";

const pctTick = (v) => `${+(100 * v).toFixed(1)}%`;

function fact(label) {
  const value = h("dd", { class: "num" });
  const sub = h("dd", { class: "fact-sub" });
  return { el: h("div", { class: "fact" }, h("dt", { text: label }), value, sub), value, sub };
}

export function createDetail(route) {
  const name = route.run;
  const group = createGroup();
  let task = null;
  let selection = null;
  let explicit = false; // the reader chose the evaluation (or the link did); otherwise follow the best one
  let chance = null; // the rate of closing at the named object by picking an object at random
  let drawn = "";
  let shownState = null;
  let contextCheckpoint = null;
  let lossMode = memory.get("loss-scale", "log") === "linear" ? "linear" : "log";

  /** Take the task and the inspected evaluation from the parameters of the hash. */
  function read(params) {
    task = params.get("task");
    explicit = params.has("ckpt") && Number.isFinite(Number(params.get("ckpt")));
    selection = explicit
      ? { checkpoint: Number(params.get("ckpt")), weights: params.get("w") === "raw" ? "raw" : "ema" }
      : null;
  }
  read(route.params);

  // ---- header ----------------------------------------------------------------------------------
  const chip = h("span", { class: "chip" });
  const restarts = h("span", { class: "restarts", hidden: true }, restartIcon(), h("b", { class: "num" }));
  const status = h("span", { class: "detail-status" });
  const bar = h("i");
  const facts = {
    progress: fact("Progress"),
    loss: fact("Loss"),
    rate: fact("Learning rate"),
    speed: fact("Speed"),
    left: fact("Time left"),
    evaluations: fact("Evaluations"),
  };
  const head = h(
    "header",
    { class: "detail-head" },
    h("a", { class: "back", href: "#/" }, backIcon(), "Runs"),
    h("div", { class: "detail-title" }, h("h1", { class: "detail-name id", text: name }), chip, restarts, status),
    h("div", { class: "detail-progress", "aria-hidden": "true" }, bar),
    h(
      "dl",
      { class: "facts" },
      Object.values(facts).map((f) => f.el),
    ),
  );

  // ---- charts ----------------------------------------------------------------------------------
  const scale = segmented({
    label: "Scale of the loss axis",
    options: [
      { id: "log", label: "log" },
      { id: "linear", label: "linear" },
    ],
    value: lossMode,
    onChange(id) {
      lossMode = id;
      memory.set("loss-scale", id);
      charts.loss.setMode(id);
    },
  });
  const taskPicker = segmented({
    label: "Task",
    options: [],
    value: task,
    onChange(id) {
      task = id;
      explicit = false;
      selection = null;
      replaceParams({ task: id, ckpt: null, w: null });
      update();
    },
  });

  const evalContext = (x, items) => {
    const points = items.map((it) => it.point).filter(Boolean);
    if (!points.length) return "";
    const anchor =
      x == null
        ? points.reduce((a, b) => (b.x > a.x ? b : a))
        : points.reduce((a, b) => (Math.abs(b.x - x) < Math.abs(a.x - x) ? b : a));
    contextCheckpoint = anchor.evaluation.checkpoint;
    return x == null ? `latest: checkpoint ${contextCheckpoint}` : `checkpoint ${contextCheckpoint}`;
  };
  const elsewhere = (p) =>
    p.evaluation.checkpoint === contextCheckpoint ? "" : ` at checkpoint ${p.evaluation.checkpoint}`;

  const charts = {
    loss: createChart({
      title: "Loss",
      label: "Training loss against samples seen. Arrow keys move the cursor, Enter inspects the nearest checkpoint.",
      height: 190,
      mode: lossMode,
      controls: scale.el,
      xFormat: fmtAxisSamples,
      yFormat: fmtAxisNumber,
      group,
      onPick: (x) => pickNear(x),
      context: (x, items) => {
        const point = items[0] && items[0].point;
        return point ? (x == null ? `latest: ${fmtSamples(point.x)} samples` : `${fmtSamples(point.x)} samples`) : "";
      },
      domain: (values, mode) => {
        if (mode !== "linear" || values.length < 20) return null;
        // Scale to the bulk of the curve; the first steps would flatten everything after them.
        const sorted = values.slice().sort((a, b) => a - b);
        return [0, niceCeil(quantile(sorted, 0.95) * 1.15, 3)];
      },
    }),
    success: createChart({
      title: "Closed-loop success",
      note: "with 95% intervals",
      label: "Closed-loop success rate per checkpoint for raw and EMA weights, with 95% intervals.",
      height: 170,
      minTop: 0.1,
      maxTop: 1,
      yFormat: pctTick,
      group,
      onPick: (x) => pickNear(x),
      context: evalContext,
    }),
    named: createChart({
      title: "Closed at the named object",
      label: "How often the gripper closed nearest the object the instruction named, per checkpoint.",
      height: 130,
      minTop: 0.5,
      maxTop: 1,
      yFormat: pctTick,
      group,
      onPick: (x) => pickNear(x),
      context: evalContext,
    }),
    miss: createChart({
      title: "Median miss, mm",
      note: "solid to the named object, dashed to the nearest; dotted line at 2 cm",
      label: "Median distance between the gripper and the named object when it closed, per checkpoint.",
      height: 130,
      xAxis: true,
      xFormat: fmtAxisSamples,
      yFormat: (v) => `${fmtAxisNumber(v)}`,
      group,
      onPick: (x) => pickNear(x),
      context: evalContext,
    }),
  };
  const noEvaluations = h("p", { class: "panel-empty", hidden: true });
  const caption = h("p", { class: "axis-caption" });
  const curves = h(
    "section",
    { class: "panel curves", "aria-label": "Training and closed-loop curves" },
    h(
      "div",
      { class: "panel-head" },
      h("h2", { class: "panel-title", text: "Training and closed loop" }),
      taskPicker.el,
    ),
    charts.loss.el,
    charts.success.el,
    charts.named.el,
    charts.miss.el,
    caption,
    noEvaluations,
  );

  // ---- inspector and table ---------------------------------------------------------------------
  function choose(next) {
    selection = next;
    explicit = true;
    replaceParams({ ckpt: next.checkpoint, w: next.weights });
    drawn = "";
    update();
  }
  const inspector = createInspector({
    onPick: choose,
    onScene(count) {
      const next = count > 1 ? 1 / count : null;
      if (next === chance) return;
      chance = next;
      drawn = "";
      update();
    },
  });
  const table = createEvalTable({ onPick: choose });
  const tablePanel = h(
    "section",
    { class: "panel", "aria-label": "Evaluations" },
    h(
      "div",
      { class: "panel-head" },
      h("h2", { class: "panel-title", text: "Evaluations" }),
      h("span", { class: "section-note", text: "closed-loop, on layouts the policy has not seen" }),
    ),
    table.el,
  );

  const missing = h("div", { class: "empty", hidden: true });
  const body = h("div", { class: "detail-body" }, h("div", { class: "detail-grid" }, curves, inspector.el), tablePanel);
  const el = h("div", { class: "view detail" }, head, body, missing);

  let lastEvaluations = [];
  let lastPositions = new Map();

  function pickNear(x) {
    let best = null;
    for (const [checkpoint, at] of lastPositions) {
      if (!best || Math.abs(at.x - x) < Math.abs(best.x - x)) best = { checkpoint, x: at.x };
    }
    if (!best) return;
    const has = (w) => lastEvaluations.some((e) => e.checkpoint === best.checkpoint && e.weights === w);
    const wanted = selection ? selection.weights : "ema";
    choose({ checkpoint: best.checkpoint, weights: has(wanted) ? wanted : WEIGHTS.filter(has).pop() });
  }

  function drawHead(run) {
    if (shownState !== run.state) {
      if (shownState) replay(chip, "changed");
      shownState = run.state;
      head.dataset.state = run.state;
      chip.replaceChildren(stateIcon(run.state), STATE_LABEL[run.state] || run.state);
    }
    restarts.hidden = !(run.restarts > 0);
    setText(restarts.lastChild, run.restarts);
    attr(restarts, "title", `restarted ${run.restarts} time${run.restarts === 1 ? "" : "s"}`);
    setText(
      status,
      run.state === "stalled" && isNum(run.seconds_since_log)
        ? `no log output for ${fmtSpan(run.seconds_since_log)}`
        : run.state === "waiting"
          ? "waiting for the training log"
          : isNum(run.seconds_since_log)
            ? `last log line ${fmtSpan(run.seconds_since_log)} ago`
            : "",
    );
    const fraction = progressOf(run);
    bar.style.width = `${(100 * fraction).toFixed(2)}%`;
    tweenNumber(facts.progress.value, run.samples_total ? fraction : null, (v) => (v == null ? DASH : fmtPct(v, 1)));
    setText(
      facts.progress.sub,
      run.samples_total
        ? `${fmtInt(run.samples_seen)} of ${fmtInt(run.samples_total)} samples`
        : "length not logged yet",
    );
    const lossIsNaN = run.loss == null && (run.loss_curve || []).length > 0;
    if (lossIsNaN) tweenNumber(facts.loss.value, null, () => "NaN");
    else tweenNumber(facts.loss.value, run.loss, fmtLoss);
    facts.loss.value.classList.toggle("alarm", lossIsNaN);
    setText(
      facts.loss.sub,
      lossIsNaN
        ? "the last logged loss is not a number"
        : run.loss == null
          ? "no step logged yet"
          : "at the last logged step",
    );
    setText(facts.rate.value, fmtLearningRate(run.learning_rate));
    setText(facts.speed.value, run.state === "training" ? fmtRate(run.samples_per_second) : DASH);
    setText(facts.speed.sub, run.state === "training" ? "samples per second" : "not training");
    setText(
      facts.left.value,
      run.state === "training" ? fmtSpan(run.seconds_left) : run.state === "done" ? "finished" : DASH,
    );
    setText(
      facts.left.sub,
      run.checkpoints > 0 ? `${run.checkpoints} checkpoint${run.checkpoints === 1 ? "" : "s"} on disk` : "",
    );
    setText(facts.evaluations.value, (run.evaluations || []).length);
    setText(facts.evaluations.sub, run.evaluations_running > 0 ? `${run.evaluations_running} more running` : "");
  }

  function update() {
    const run = store.snapshot ? runByName(name) : null;
    head.hidden = body.hidden = !run;
    missing.hidden = !!run;
    if (!run) {
      if (!store.snapshot) {
        missing.replaceChildren(h("p", { class: "empty-title", text: "Connecting to the monitor" }));
      } else {
        missing.replaceChildren(
          h("p", { class: "empty-title", text: `There is no run named ${name}` }),
          h("p", { text: "Its training log may have been moved or deleted." }),
          h("a", { class: "button", href: "#/", text: "Show all runs" }),
        );
      }
      return;
    }
    drawHead(run);

    const tasks = tasksOf(run);
    if (!tasks.includes(task)) task = tasks[0] || null;
    const evaluations = task ? evalsOf(run, task) : [];
    const has = (s) => s && evaluations.some((e) => e.checkpoint === s.checkpoint && e.weights === s.weights);
    if (!has(selection) || !explicit) {
      const other = selection &&
        explicit && { checkpoint: selection.checkpoint, weights: selection.weights === "raw" ? "ema" : "raw" };
      const asked = explicit;
      if (has(other)) {
        selection = other;
      } else {
        const best = bestOf(evaluations) || evaluations[evaluations.length - 1] || null;
        selection = best ? { checkpoint: best.checkpoint, weights: best.weights } : null;
        explicit = false;
      }
      // The link named an evaluation that is not there: the address follows what is shown instead.
      if (asked)
        replaceParams(explicit ? { ckpt: selection.checkpoint, w: selection.weights } : { ckpt: null, w: null });
    }
    const mark = [signature(run), task, selection && selection.checkpoint, selection && selection.weights, chance].join(
      "|",
    );
    if (mark === drawn) {
      inspector.retry(); // results that could not be fetched while the server was away
      return;
    }
    drawn = mark;

    const positions = checkpointPositions(run, evaluations);
    lastEvaluations = evaluations;
    lastPositions = positions;
    const loss = lossPoints(run);
    let end = Math.max(run.samples_seen || 0, loss.length ? loss[loss.length - 1].x : 0);
    for (const at of positions.values()) if (at.x > end) end = at.x;
    const xDomain = [0, (end || 1) * 1.02];
    const selectedX = selection && positions.get(selection.checkpoint) ? positions.get(selection.checkpoint).x : null;
    const some = evaluations.length > 0;

    taskPicker.el.hidden = tasks.length < 2;
    taskPicker.options(tasks.map((t) => ({ id: t, label: t })));
    taskPicker.set(task);
    charts.success.el.hidden = charts.named.el.hidden = charts.miss.el.hidden = !some;
    inspector.el.hidden = tablePanel.hidden = !some;
    curves.classList.toggle("alone", !some);
    noEvaluations.hidden = some;
    if (!some) {
      setText(
        noEvaluations,
        run.evaluations_running > 0
          ? "The first evaluation of this run is in progress. Its closed-loop results will be drawn here, under the loss."
          : "No checkpoint of this run has been evaluated yet. Closed-loop success, and how the policy fails, are drawn here once one is (sim/pickup/scripts/eval_checkpoints.sh).",
      );
    }

    setText(
      caption,
      some
        ? "Training samples seen. The numbers above the loss are evaluated checkpoints; click a chart to inspect the nearest one."
        : "Training samples seen.",
    );
    charts.loss.set({
      xDomain,
      selectedX,
      xAxis: !some,
      marks: [...positions]
        .map(([checkpoint, at]) => ({ x: at.x, label: String(checkpoint) }))
        .sort((a, b) => a.x - b.x),
      series: [
        { id: "loss", label: "loss", color: "var(--ink)", width: 1.6, points: loss, value: (p) => fmtLoss(p.y) },
      ],
    });
    if (!some) return;

    const series = (valueOf, more) =>
      WEIGHTS.map((w) => ({
        id: w,
        label: WEIGHT_LABEL[w],
        color: `var(--${w})`,
        dots: w === "raw" ? "hollow" : "filled",
        dodge: w === "raw" ? -3 : 3,
        points: evalSeries(evaluations, positions, w, valueOf),
        ...more,
      }));
    charts.success.set({
      xDomain,
      selectedX,
      series: series((e) => e.success_rate, {
        whiskers: true,
        value: (p) => {
          const interval = intervalOf(p.evaluation);
          return `${fmtPct(p.y)}${interval ? ` (${(100 * interval[0]).toFixed(0)}${DASH}${(100 * interval[1]).toFixed(0)})` : ""}${elsewhere(p)}`;
        },
      }).map((sr) => ({
        ...sr,
        points: sr.points.map((p) => {
          const interval = intervalOf(p.evaluation);
          return interval ? { ...p, lo: interval[0], hi: interval[1] } : p;
        }),
      })),
    });
    charts.named.setNote(
      chance ? `dotted line: picking one of ${Math.round(1 / chance)} objects at random, ${fmtPct(chance)}` : "",
    );
    charts.named.set({
      xDomain,
      selectedX,
      refLines: chance ? [{ y: chance }] : [],
      series: series((e) => e.at_named_object, { value: (p) => `${fmtPct(p.y)}${elsewhere(p)}` }),
    });
    charts.miss.set({
      xDomain,
      selectedX,
      refLines: [{ y: 20 }],
      series: [
        ...series((e) => e.median_miss_named_mm, {
          value: (p) =>
            `${fmtMm(p.y)}${isNum(p.evaluation.median_miss_nearest_mm) ? `, nearest ${fmtMm(p.evaluation.median_miss_nearest_mm)}` : ""}${elsewhere(p)}`,
        }),
        ...series((e) => e.median_miss_nearest_mm, { dashed: true, width: 1.4, dots: false, readout: false }).map(
          (sr) => ({ ...sr, id: `${sr.id}-nearest` }),
        ),
      ],
    });
    inspector.set({ runName: name, task, evaluations, positions, selection });
    table.set({ evaluations, positions, selection });
  }

  return {
    el,
    update,
    /** The hash changed while this run was open (a pasted link, the back button). */
    route(next) {
      read(next.params);
      drawn = "";
      update();
    },
    destroy() {
      for (const chart of Object.values(charts)) chart.destroy();
      inspector.destroy();
    },
  };
}
