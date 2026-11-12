// All runs at a glance: a card per run, the GPUs when the server reports any, sort and filter.

import { h, setText, setClass, attr, reconcile, replay, tweenNumber, memory } from "./dom.js";
import { store } from "./store.js";
import { runHref } from "./route.js";
import { createSpark } from "./charts.js";
import { stateIcon, restartIcon, spinnerIcon, warningIcon, STATE_LABEL } from "./icons.js";
import { summaryOf, signature, progressOf, intervalOf, lossPoints, STATE_ORDER, WEIGHT_LABEL } from "./model.js";
import { fmtSamples, fmtLoss, fmtPct, fmtMm, fmtRate, fmtSpan, isNum, DASH } from "./format.js";

const SORTS = {
  state: (a, b) => STATE_ORDER[a.run.state] - STATE_ORDER[b.run.state] || byName(a, b),
  name: (a, b) => byName(a, b),
  success: (a, b) => rate(b) - rate(a) || byName(a, b),
};

function byName(a, b) {
  return a.run.name < b.run.name ? -1 : a.run.name > b.run.name ? 1 : 0;
}

function rate(item) {
  return item.summary.best ? item.summary.best.success_rate : -1;
}

function figure(label, extra) {
  const value = h("span", { class: "fig-value num" });
  const sub = h("span", { class: "fig-sub" });
  const el = h("div", { class: `fig ${extra || ""}` }, h("span", { class: "fig-label", text: label }), value, sub);
  return { el, value, sub };
}

function createCard(name) {
  const bar = h("i");
  const chip = h("span", { class: "chip" });
  const restarts = h("span", { class: "restarts", hidden: true }, restartIcon(), h("b", { class: "num" }));
  const status = h("span", { class: "card-status" });
  const progress = figure("Progress");
  const loss = figure("Loss", "fig-loss");
  const lossSpark = createSpark("Loss over training");
  loss.el.appendChild(lossSpark.el);
  const best = figure("Best success");
  const named = figure("At named object");
  const miss = figure("Median miss");
  const sparkLegend = h(
    "span",
    { class: "spark-legend" },
    h("span", { class: "fig-label", text: "Success by checkpoint" }),
    h("span", { class: "legend-key", style: { "--c": "var(--raw)" } }, h("i", { class: "key hollow" }), "raw"),
    h("span", { class: "legend-key", style: { "--c": "var(--ema)" } }, h("i", { class: "key" }), "EMA"),
  );
  const successSpark = createSpark("Closed-loop success by checkpoint, raw and EMA weights", 38, true);
  const running = h("span", { class: "running", hidden: true }, spinnerIcon(), h("span"));
  const closed = h(
    "div",
    { class: "card-closed" },
    h("div", { class: "card-figs three" }, best.el, named.el, miss.el),
    h("div", { class: "card-spark" }, sparkLegend, successSpark.el),
  );
  const none = h("p", { class: "card-none", hidden: true });
  const el = h(
    "a",
    { class: "card", href: runHref(name) },
    h("div", { class: "card-progress", "aria-hidden": "true" }, bar),
    h("div", { class: "card-head" }, chip, restarts, status),
    h("h2", { class: "card-name id", text: name }),
    h("div", { class: "card-figs two" }, progress.el, loss.el),
    closed,
    none,
    running,
  );
  el.__refs = {
    bar,
    chip,
    restarts,
    status,
    progress,
    loss,
    lossSpark,
    best,
    named,
    miss,
    successSpark,
    closed,
    none,
    running,
    state: null,
    signature: null,
    scale: null,
  };
  return el;
}

function statusText(run) {
  if (run.state === "training") {
    const parts = [];
    if (isNum(run.samples_per_second)) parts.push(`${fmtRate(run.samples_per_second)}`);
    if (isNum(run.seconds_left)) parts.push(`${fmtSpan(run.seconds_left)} left`);
    return parts.join(", ") || "running";
  }
  if (run.state === "stalled")
    return isNum(run.seconds_since_log) ? `no log output for ${fmtSpan(run.seconds_since_log)}` : "no log output";
  if (run.state === "stopped") return run.note || "stopped on purpose";
  if (run.state === "done") return "finished";
  return "waiting for the training log";
}

function updateCard(el, item, successTop) {
  const { run, summary } = item;
  const r = el.__refs;
  setText(r.status, statusText(run));
  const transition = store.transitions.get(run.name);
  if (transition && transition.serial === store.serial && r.state && r.state !== run.state) replay(el, "changed");
  if (r.state !== run.state) {
    r.state = run.state;
    el.dataset.state = run.state;
    r.chip.replaceChildren(stateIcon(run.state), STATE_LABEL[run.state] || run.state);
  }
  const mark = `${signature(run)}|${successTop}`;
  if (r.signature === mark) return;
  r.signature = mark;

  const fraction = progressOf(run);
  r.bar.style.width = `${(100 * fraction).toFixed(2)}%`;
  r.restarts.hidden = !(run.restarts > 0);
  setText(r.restarts.lastChild, run.restarts);
  attr(r.restarts, "title", `restarted ${run.restarts} time${run.restarts === 1 ? "" : "s"}`);
  tweenNumber(r.progress.value, run.samples_total ? fraction : null, (v) => (v == null ? DASH : fmtPct(v)));
  setText(
    r.progress.sub,
    run.samples_total
      ? `${fmtSamples(run.samples_seen)} of ${fmtSamples(run.samples_total)} samples`
      : "length not logged yet",
  );
  const lossIsNaN = run.loss == null && (run.loss_curve || []).length > 0; // steps were logged, the last one without a number
  if (lossIsNaN) tweenNumber(r.loss.value, null, () => "NaN");
  else tweenNumber(r.loss.value, run.loss, fmtLoss);
  setClass(r.loss.value, "alarm", lossIsNaN);
  const curve = lossPoints(run);
  const tail = curve.slice(Math.floor(curve.length / 20)).map((p) => [p.x, p.y]); // without the first steps, which dwarf the rest
  r.lossSpark.set([{ id: "loss", color: "var(--ink-2)", points: tail }], { log: true });

  const { best, evaluations, checkpoints } = summary;
  r.closed.hidden = !best;
  r.none.hidden = !!best;
  if (best) {
    const interval = intervalOf(best);
    tweenNumber(r.best.value, best.success_rate, (v) => fmtPct(v));
    setText(r.best.sub, `checkpoint ${best.checkpoint}, ${WEIGHT_LABEL[best.weights]}`);
    attr(
      r.best.el,
      "title",
      interval ? `95% interval ${(100 * interval[0]).toFixed(0)}${DASH}${(100 * interval[1]).toFixed(0)}%` : null,
    );
    tweenNumber(r.named.value, best.at_named_object, (v) => fmtPct(v));
    attr(r.named.el, "title", "How often the gripper closed nearest the object the instruction named");
    tweenNumber(r.miss.value, best.median_miss_named_mm, fmtMm);
    setText(r.miss.sub, isNum(best.median_miss_nearest_mm) ? `nearest ${fmtMm(best.median_miss_nearest_mm)}` : "");
    const index = new Map(checkpoints.map((c, i) => [c, i]));
    const line = (weights) =>
      evaluations
        .filter((e) => e.weights === weights && isNum(e.success_rate))
        .map((e) => [index.get(e.checkpoint), e.success_rate]);
    r.successSpark.set(
      [
        { id: "raw", color: "var(--raw)", points: line("raw"), dots: true },
        { id: "ema", color: "var(--ema)", points: line("ema"), dots: true },
      ],
      { yMin: 0, yMax: successTop },
    );
  } else {
    setText(r.none, run.evaluations_running ? "First evaluation in progress" : "No evaluations yet");
  }
  r.running.hidden = !(best && run.evaluations_running > 0);
  setText(
    r.running.lastChild,
    `${run.evaluations_running} evaluation${run.evaluations_running === 1 ? "" : "s"} running`,
  );

  const words = [
    run.name,
    STATE_LABEL[run.state],
    run.samples_total ? `${fmtPct(fraction)} of ${fmtSamples(run.samples_total)} samples` : "",
    `loss ${fmtLoss(run.loss)}`,
  ];
  if (best)
    words.push(
      `best success ${fmtPct(best.success_rate)} at checkpoint ${best.checkpoint} ${WEIGHT_LABEL[best.weights]}`,
    );
  attr(el, "aria-label", words.filter(Boolean).join(", "));
}

function meter(label, area) {
  const fill = h("i");
  const value = h("span", { class: "meter-value num" });
  const el = h(
    "div",
    { class: `meter meter-${area}` },
    h("span", { class: "meter-label", text: label }),
    h("div", { class: "meter-track" }, fill),
    value,
  );
  return { el, fill, value };
}

function createGpu() {
  const name = h("span", { class: "gpu-name" });
  const util = meter("Load", "load");
  const mem = meter("Memory", "memory");
  const temp = h("span", { class: "gpu-temp num" });
  const spark = createSpark("GPU load over the last minutes", 26);
  const el = h(
    "div",
    { class: "gpu" },
    h("div", { class: "gpu-id" }, h("b", { class: "num" }), name),
    util.el,
    mem.el,
    h("div", { class: "gpu-trend" }, spark.el),
    temp,
  );
  el.__refs = { index: el.querySelector("b"), name, util, mem, temp, spark };
  return el;
}

function updateGpu(el, gpu) {
  const r = el.__refs;
  setText(r.index, `GPU ${gpu.index}`);
  setText(r.name, gpu.name || "");
  const util = isNum(gpu.util) ? Math.min(100, Math.max(0, gpu.util)) : 0;
  r.util.fill.style.width = `${util}%`;
  r.util.el.dataset.level = util >= 50 ? "busy" : util >= 5 ? "low" : "idle"; // busy is the good state
  tweenNumber(r.util.value, gpu.util, (v) => (v == null ? DASH : `${Math.round(v)}%`), 400);
  const share = gpu.mem_total_mb ? gpu.mem_used_mb / gpu.mem_total_mb : 0;
  r.mem.fill.style.width = `${(100 * Math.min(1, Math.max(0, share))).toFixed(1)}%`;
  setText(
    r.mem.value,
    isNum(gpu.mem_used_mb) && isNum(gpu.mem_total_mb)
      ? `${(gpu.mem_used_mb / 1024).toFixed(1)} / ${(gpu.mem_total_mb / 1024).toFixed(0)} GB`
      : DASH,
  );
  const hot = isNum(gpu.temp_c) && gpu.temp_c >= 85;
  setClass(r.temp, "hot", hot);
  r.temp.replaceChildren(hot ? warningIcon() : "", isNum(gpu.temp_c) ? `${Math.round(gpu.temp_c)} °C` : DASH);
  const history = store.gpuHistory.get(gpu.index) || [];
  r.spark.set([{ id: "util", color: "var(--good)", points: history.map((p, i) => [i, p.util]) }], {
    yMin: 0,
    yMax: 100,
  });
  attr(
    el,
    "aria-label",
    `GPU ${gpu.index} ${gpu.name || ""}: load ${Math.round(util)}%, memory ${Math.round(100 * share)}%, ${isNum(gpu.temp_c) ? Math.round(gpu.temp_c) : "unknown"} degrees`,
  );
}

export function createOverview() {
  let sort = memory.get("sort", "state");
  if (!SORTS[sort]) sort = "state";
  let query = "";

  const gpus = h("section", { class: "gpus", "aria-label": "GPUs", hidden: true });
  const filter = h("input", {
    class: "input",
    type: "search",
    placeholder: "Filter runs",
    "aria-label": "Filter runs by name",
    autocomplete: "off",
    spellcheck: "false",
  });
  const select = h(
    "select",
    { class: "select", "aria-label": "Sort runs" },
    h("option", { value: "state", text: "Sort by state" }),
    h("option", { value: "name", text: "Sort by name" }),
    h("option", { value: "success", text: "Sort by best success" }),
  );
  select.value = sort;
  const shown = h("span", { class: "toolbar-count", "aria-live": "polite" });
  const toolbar = h("div", { class: "toolbar" }, filter, select, shown);
  const cards = h("div", { class: "cards" });
  const empty = h("div", { class: "empty", hidden: true });
  const el = h(
    "div",
    { class: "view overview" },
    h("h1", { class: "sr-only", text: "Runs" }),
    gpus,
    toolbar,
    cards,
    empty,
  );

  function update() {
    const snapshot = store.snapshot;
    if (!snapshot) {
      toolbar.hidden = true;
      empty.hidden = false;
      empty.replaceChildren(
        h("p", {
          class: "empty-title",
          text: store.connection === "unauthorized" ? "Waiting for an access token" : "Connecting to the monitor",
        }),
      );
      return;
    }
    const list = Array.isArray(snapshot.gpus) ? snapshot.gpus : [];
    gpus.hidden = !list.length;
    reconcile(gpus, list, (gpu) => gpu.index, createGpu, updateGpu);

    const all = snapshot.runs.map((run) => ({ run, summary: summaryOf(run) }));
    let top = 0.02; // one scale for every card, so the small charts can be compared
    for (const item of all)
      for (const e of item.summary.evaluations) if (isNum(e.success_rate) && e.success_rate > top) top = e.success_rate;
    top *= 1.15;
    const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
    const matching = all
      .filter((item) => terms.every((term) => item.run.name.toLowerCase().includes(term)))
      .sort(SORTS[sort]);
    reconcile(
      cards,
      matching,
      (item) => item.run.name,
      (item) => createCard(item.run.name),
      (node, item) => updateCard(node, item, top),
    );

    toolbar.hidden = !all.length;
    setText(
      shown,
      terms.length ? `${matching.length} of ${all.length} runs` : `${all.length} run${all.length === 1 ? "" : "s"}`,
    );
    empty.hidden = matching.length > 0;
    if (!all.length) {
      empty.replaceChildren(
        h("p", { class: "empty-title", text: "No runs yet" }),
        h("p", {
          text: "A run appears here as soon as its training log exists in the experiments directory the server reads.",
        }),
        h("p", {
          class: "empty-code id",
          text: "sim/pickup/scripts/train_resilient.sh <config> <dataset> <run-name> <samples>",
        }),
      );
    } else if (!matching.length) {
      const clear = h("button", { class: "button", type: "button", text: "Show all runs" });
      clear.addEventListener("click", () => {
        filter.value = "";
        query = "";
        update();
        filter.focus();
      });
      empty.replaceChildren(h("p", { class: "empty-title", text: `No run matches “${query.trim()}”` }), clear);
    }
  }

  filter.addEventListener("input", () => {
    query = filter.value;
    update();
  });
  select.addEventListener("change", () => {
    sort = select.value;
    memory.set("sort", sort);
    update();
  });

  return { el, update, destroy() {} };
}
