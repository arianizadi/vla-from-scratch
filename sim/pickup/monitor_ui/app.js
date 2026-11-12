// Entry point: the header (connection, counts by state, theme), the router, and the view on screen.

import { h, setText, setClass, attr, reconcile, memory } from "./dom.js";
import { store, subscribe, start } from "./store.js";
import { parseRoute, hideToken } from "./route.js";
import { stateIcon, sunIcon, moonIcon, warningIcon, STATE_LABEL } from "./icons.js";
import { fmtClock, fmtSpan } from "./format.js";
import { createOverview } from "./overview.js";
import { createDetail } from "./detail.js";
import { createCompare } from "./compare.js";

const STATES = ["training", "stalled", "waiting", "done"];
const CONNECTION_TEXT = {
  connecting: "Connecting",
  live: "Live",
  polling: "Live, polling",
  reconnecting: "Reconnecting",
  unauthorized: "Access token needed",
};

const main = document.getElementById("main");
const counts = document.getElementById("counts");
const connection = document.getElementById("conn");
const connectionText = connection.querySelector(".conn-text");
const connectionTime = connection.querySelector(".conn-time");
const banner = document.getElementById("banner");
const themeButton = document.getElementById("theme");
const navLinks = [...document.querySelectorAll(".nav a")];

// ---- theme: dark unless the reader chose light ---------------------------------------------------

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  themeButton.replaceChildren(theme === "dark" ? sunIcon() : moonIcon());
  const other = theme === "dark" ? "light" : "dark";
  themeButton.setAttribute("aria-label", `Switch to the ${other} theme`);
  themeButton.title = `Switch to the ${other} theme`;
}

themeButton.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  memory.set("theme", next);
  applyTheme(next);
});
applyTheme(document.documentElement.dataset.theme === "light" ? "light" : "dark");

// ---- header --------------------------------------------------------------------------------------

function renderCounts() {
  const runs = store.snapshot ? store.snapshot.runs : [];
  const present = STATES.map((state) => ({ state, count: runs.filter((run) => run.state === state).length })).filter(
    (c) => c.count,
  );
  reconcile(
    counts,
    present,
    (c) => c.state,
    (c) =>
      h(
        "span",
        { class: `count state-${c.state}` },
        stateIcon(c.state),
        h("b", { class: "num" }),
        h("span", { class: "count-label", text: STATE_LABEL[c.state] }),
      ),
    (node, c) => {
      setText(node.children[1], c.count);
      attr(node, "title", `${c.count} ${STATE_LABEL[c.state]}`);
    },
  );
  attr(
    counts,
    "aria-label",
    present.length ? `Runs: ${present.map((c) => `${c.count} ${STATE_LABEL[c.state]}`).join(", ")}` : "No runs",
  );
}

function renderConnection() {
  const state = store.connection;
  attr(connection, "data-state", state);
  setText(connectionText, CONNECTION_TEXT[state] || state);
  if (!store.receivedAt) {
    setText(connectionTime, "");
    return;
  }
  const age = (Date.now() - store.receivedAt) / 1000;
  const stale = age > 10 && state !== "live" && state !== "polling";
  setText(
    connectionTime,
    stale ? `last update ${fmtClock(store.receivedAt)}, ${fmtSpan(age)} ago` : `updated ${fmtClock(store.receivedAt)}`,
  );
}

function renderBanner() {
  let message = "";
  if (store.connection === "unauthorized") {
    message =
      "This monitor asks for an access token. Open the address that the server printed when it started: it ends in ?token=…";
  } else if (store.snapshot && store.snapshot.error) {
    message = `The server could not read the runs just now (${store.snapshot.error}). This is the last state it read.`;
  }
  if (!message) {
    banner.hidden = true;
    return;
  }
  if (banner.dataset.message !== message) {
    banner.dataset.message = message;
    banner.replaceChildren(warningIcon(), h("span", { text: message }));
  }
  banner.hidden = false;
}

// ---- router --------------------------------------------------------------------------------------

let current = null;
const scrollOf = new Map();

function show() {
  const route = parseRoute();
  const key = route.view === "detail" ? `detail:${route.run}` : route.view;
  if (!current || current.key !== key) {
    if (current) {
      scrollOf.set(current.key, window.scrollY);
      current.view.destroy();
    }
    const view =
      route.view === "detail"
        ? createDetail(route)
        : route.view === "compare"
          ? createCompare(route)
          : createOverview(route);
    main.replaceChildren(view.el);
    current = { key, view };
    view.update();
    window.scrollTo(0, scrollOf.get(key) || 0);
  } else if (current.view.route) {
    current.view.route(route);
  }
  for (const link of navLinks) {
    // A run's page belongs to "Runs" without being that page.
    const here =
      link.dataset.view === route.view
        ? "page"
        : link.dataset.view === "overview" && route.view === "detail"
          ? "true"
          : null;
    attr(link, "aria-current", here);
  }

  document.title =
    route.view === "detail"
      ? `${route.run} – Training monitor`
      : route.view === "compare"
        ? "Compare – Training monitor"
        : "Training monitor";
}

subscribe((what) => {
  if (what === "data") {
    renderCounts();
    if (current) current.view.update();
  }
  renderConnection();
  renderBanner();
});

window.addEventListener("hashchange", show);
window.setInterval(renderConnection, 1000);

hideToken();
show();
renderCounts();
renderConnection();
setClass(document.documentElement, "ready", true);
start();
