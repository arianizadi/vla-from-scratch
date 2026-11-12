// Every evaluation of a run as a sortable table: the numbers behind the charts, reachable without
// hovering. Choosing a row selects that evaluation everywhere on the page.

import { h, setText, setClass, attr, reconcile } from "./dom.js";
import { sortIcon } from "./icons.js";
import { intervalOf, WEIGHT_LABEL } from "./model.js";
import { fmtPct, fmtMm, fmtSamples, isNum, DASH } from "./format.js";
import { niceCeil } from "./scale.js";

const COLUMNS = [
  { key: "checkpoint", label: "Checkpoint", value: (e) => e.checkpoint, first: 1 },
  { key: "weights", label: "Weights", value: (e) => e.weights, first: 1, text: true },
  { key: "samples", label: "Samples", value: (e, x) => x, first: 1 },
  { key: "success", label: "Success", value: (e) => e.success_rate, first: -1, wide: true },
  { key: "named", label: "At named object", value: (e) => e.at_named_object, first: -1 },
  { key: "miss", label: "Miss, named", value: (e) => e.median_miss_named_mm, first: 1 },
  { key: "nearest", label: "Miss, nearest", value: (e) => e.median_miss_nearest_mm, first: 1 },
  { key: "w10", label: "Within 1 cm", value: (e) => e.within_10mm, first: -1 },
  { key: "w20", label: "Within 2 cm", value: (e) => e.within_20mm, first: -1 },
  { key: "wrong", label: "Wrong object", value: (e) => e.wrong_object_rate, first: -1 },
];

export function createEvalTable({ onPick }) {
  let sort = { key: "checkpoint", direction: 1 };
  let state = null; // {evaluations, positions, selection}

  const headRow = h("tr");
  const body = h("tbody");
  const table = h(
    "table",
    { class: "table" },
    h("caption", {
      class: "sr-only",
      text: "Evaluations of this run. Column headers sort; a row selects that evaluation.",
    }),
    h("thead", null, headRow),
    body,
  );
  const el = h(
    "div",
    { class: "table-scroll", tabindex: "0", role: "region", "aria-label": "Evaluations table" },
    table,
  );

  for (const column of COLUMNS) {
    const button = h("button", { type: "button", class: "th-button" }, h("span", { text: column.label }), sortIcon());
    button.addEventListener("click", () => {
      sort =
        sort.key === column.key
          ? { key: column.key, direction: -sort.direction }
          : { key: column.key, direction: column.first };
      render();
    });
    headRow.appendChild(h("th", { scope: "col", class: column.text ? "" : "numeric", "data-key": column.key }, button));
  }

  function createRow() {
    const pickButton = h("button", { type: "button", class: "row-pick num" });
    const range = h("i", { class: "ci-range" });
    const point = h("i", { class: "ci-point" });
    const cells = {
      weights: h("td", null, h("span", { class: "weights-tag" }, h("i", { class: "key" }), h("span"))),
      samples: h("td", { class: "numeric" }),
      success: h(
        "td",
        { class: "numeric success-cell" },
        h("span", { class: "success-count" }),
        h("b", { class: "success-rate" }),
        h("span", { class: "ci", "aria-hidden": "true" }, range, point),
        h("span", { class: "success-interval" }),
      ),
      named: h("td", { class: "numeric" }),
      miss: h("td", { class: "numeric" }),
      nearest: h("td", { class: "numeric" }),
      w10: h("td", { class: "numeric" }),
      w20: h("td", { class: "numeric" }),
      wrong: h("td", { class: "numeric" }),
    };
    const row = h("tr", null, h("th", { scope: "row", class: "numeric" }, pickButton), Object.values(cells));
    row.__refs = { pickButton, range, point, cells };
    row.addEventListener(
      "click",
      () => row.__evaluation && onPick({ checkpoint: row.__evaluation.checkpoint, weights: row.__evaluation.weights }),
    );
    return row;
  }

  function render() {
    if (!state) return;
    const { evaluations, positions, selection } = state;
    const column = COLUMNS.find((c) => c.key === sort.key);
    const xOf = (e) => (positions.get(e.checkpoint) || {}).x;
    const rows = evaluations.slice().sort((a, b) => {
      const va = column.value(a, xOf(a));
      const vb = column.value(b, xOf(b));
      const missingA = va == null || (typeof va === "number" && !isNum(va));
      const missingB = vb == null || (typeof vb === "number" && !isNum(vb));
      if (missingA !== missingB) return missingA ? 1 : -1; // rows without the value go last either way
      const order = missingA || va === vb ? 0 : va < vb ? -1 : 1;
      return order * sort.direction || a.checkpoint - b.checkpoint || (a.weights < b.weights ? -1 : 1);
    });
    for (const th of headRow.children) {
      const on = th.dataset.key === sort.key;
      attr(th, "aria-sort", on ? (sort.direction > 0 ? "ascending" : "descending") : "none");
    }
    let top = 0.1;
    for (const e of evaluations) {
      const interval = intervalOf(e);
      if (interval && interval[1] > top) top = interval[1];
    }
    top = niceCeil(top, 2);
    reconcile(
      body,
      rows,
      (e) => `${e.checkpoint}/${e.weights}`,
      createRow,
      (row, e) => {
        const r = row.__refs;
        row.__evaluation = e;
        const selected = selection && e.checkpoint === selection.checkpoint && e.weights === selection.weights;
        setClass(row, "selected", selected);
        attr(row, "aria-current", selected ? "true" : null);
        setText(r.pickButton, e.checkpoint);
        attr(r.pickButton, "aria-label", `Checkpoint ${e.checkpoint}, ${WEIGHT_LABEL[e.weights]} weights: inspect`);
        r.cells.weights.firstChild.style.setProperty("--c", `var(--${e.weights})`);
        setClass(r.cells.weights.firstChild.firstChild, "hollow", e.weights === "raw");
        setText(r.cells.weights.firstChild.lastChild, WEIGHT_LABEL[e.weights] || e.weights);
        const at = positions.get(e.checkpoint);
        setText(r.cells.samples, at ? `${at.estimated ? "~" : ""}${fmtSamples(at.x)}` : DASH);
        const interval = intervalOf(e);
        const success = r.cells.success;
        setText(success.children[0], isNum(e.successes) ? `${e.successes}/${e.episodes}` : "");
        setText(success.children[1], fmtPct(e.success_rate));
        setText(
          success.children[3],
          interval ? `${(100 * interval[0]).toFixed(0)}${DASH}${(100 * interval[1]).toFixed(0)}%` : "",
        );
        success.children[2].style.setProperty("--c", `var(--${e.weights})`);
        if (interval) {
          r.range.style.left = `${(100 * Math.min(1, interval[0] / top)).toFixed(1)}%`;
          r.range.style.width = `${(100 * Math.min(1, (interval[1] - interval[0]) / top)).toFixed(1)}%`;
        }
        r.point.style.left = `${(100 * Math.min(1, (e.success_rate || 0) / top)).toFixed(1)}%`;
        setText(r.cells.named, fmtPct(e.at_named_object));
        setText(r.cells.miss, fmtMm(e.median_miss_named_mm));
        setText(r.cells.nearest, fmtMm(e.median_miss_nearest_mm));
        setText(r.cells.w10, fmtPct(e.within_10mm));
        setText(r.cells.w20, fmtPct(e.within_20mm));
        setText(r.cells.wrong, fmtPct(e.wrong_object_rate));
      },
    );
  }

  return {
    el,
    set(next) {
      state = next;
      render();
    },
  };
}
