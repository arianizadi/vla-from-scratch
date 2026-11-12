// Inline icons. Each is a 16 x 16 drawing that takes the colour of the text around it.

import { s } from "./dom.js";

function icon(name, ...shapes) {
  return s(
    "svg",
    { class: `icon icon-${name}`, viewBox: "0 0 16 16", "aria-hidden": "true", focusable: "false" },
    shapes,
  );
}

const stroke = {
  fill: "none",
  stroke: "currentColor",
  "stroke-width": 1.7,
  "stroke-linecap": "round",
  "stroke-linejoin": "round",
};

/** The mark of a run's state; the same four marks as the terminal dashboard. */
export function stateIcon(state) {
  if (state === "training") {
    return icon(
      "training",
      s("circle", { class: "halo", cx: 8, cy: 8, r: 4 }),
      s("circle", { cx: 8, cy: 8, r: 4, fill: "currentColor" }),
    );
  }
  if (state === "done") return icon("done", s("path", { d: "M3 8.6l3.2 3.2L13 4.6", ...stroke, "stroke-width": 2 }));
  if (state === "stalled")
    return icon(
      "stalled",
      s("path", {
        d: "M8 2.4l6.2 11H1.8z",
        fill: "currentColor",
        stroke: "currentColor",
        "stroke-width": 1,
        "stroke-linejoin": "round",
      }),
    );
  return icon("waiting", s("circle", { cx: 8, cy: 8, r: 4.2, ...stroke }));
}

export const STATE_LABEL = { training: "training", done: "done", stalled: "stalled", waiting: "waiting" };

const OBJECT_SHAPES = {
  cube: () => s("rect", { x: 3, y: 3, width: 10, height: 10, rx: 1.6, fill: "currentColor" }),
  can: () => s("circle", { cx: 8, cy: 8, r: 5.4, fill: "currentColor" }),
  hex: () =>
    s("path", {
      d: "M8 1.8l5.4 3.1v6.2L8 14.2l-5.4-3.1V4.9zM8 5.6a2.4 2.4 0 100 4.8 2.4 2.4 0 000-4.8z",
      fill: "currentColor",
      "fill-rule": "evenodd",
    }),
  tblock: () => s("path", { d: "M2 3h12v4.2H10.1V13.4H5.9V7.2H2z", fill: "currentColor" }),
};

const OBJECT_LABELS = { cube: "cube", can: "can", hex: "hex nut", tblock: "T block" };

/** The outline of an object as seen from above; a diamond for names this page does not know. */
export function objectGlyph(name) {
  const shape = OBJECT_SHAPES[name];
  return icon(`object`, shape ? shape() : s("path", { d: "M8 2l6 6-6 6-6-6z", fill: "currentColor" }));
}

export function objectLabel(name) {
  return OBJECT_LABELS[name] || String(name).replace(/_/g, " ");
}

export function sunIcon() {
  return icon(
    "sun",
    s("circle", { cx: 8, cy: 8, r: 3, ...stroke }),
    s("path", {
      d: "M8 1.2v1.6M8 13.2v1.6M1.2 8h1.6M13.2 8h1.6M3.2 3.2l1.1 1.1M11.7 11.7l1.1 1.1M3.2 12.8l1.1-1.1M11.7 4.3l1.1-1.1",
      ...stroke,
      "stroke-width": 1.4,
    }),
  );
}

export function moonIcon() {
  return icon("moon", s("path", { d: "M13.4 9.6A5.6 5.6 0 016.4 2.6a5.6 5.6 0 107 7z", ...stroke }));
}

export function playIcon() {
  return icon("play", s("path", { d: "M4.5 2.8v10.4L13 8z", fill: "currentColor" }));
}

export function pauseIcon() {
  return icon("pause", s("path", { d: "M4 3h3v10H4zM9 3h3v10H9z", fill: "currentColor" }));
}

export function backIcon() {
  return icon("back", s("path", { d: "M9.8 3L4.8 8l5 5", ...stroke, "stroke-width": 1.9 }));
}

export function restartIcon() {
  return icon("restart", s("path", { d: "M13 8a5 5 0 11-1.6-3.7M13 2.6v2.6h-2.6", ...stroke, "stroke-width": 1.6 }));
}

export function spinnerIcon() {
  return icon(
    "spinner",
    s("circle", {
      cx: 8,
      cy: 8,
      r: 5.2,
      fill: "none",
      stroke: "currentColor",
      "stroke-width": 1.8,
      "stroke-dasharray": "20 13",
      "stroke-linecap": "round",
    }),
  );
}

export function warningIcon() {
  return icon(
    "warning",
    s("path", { d: "M8 2.2l6.4 11.2H1.6z", ...stroke, "stroke-width": 1.5 }),
    s("path", { d: "M8 6.4v3.3M8 11.6v.2", ...stroke }),
  );
}

export function sortIcon() {
  return icon(
    "sort",
    s("path", { class: "up", d: "M4.5 6.5L8 3l3.5 3.5z", fill: "currentColor" }),
    s("path", { class: "down", d: "M4.5 9.5L8 13l3.5-3.5z", fill: "currentColor" }),
  );
}
