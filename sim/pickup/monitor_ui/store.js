// The client-side store: the latest snapshot of all runs, kept current over /api/stream, with a
// fallback to polling /api/state while the stream is down. Views subscribe and patch themselves.

const listeners = new Set();

export const store = {
  snapshot: null, // the last payload of /api/state
  receivedAt: 0, // when it arrived, on this computer's clock (ms)
  connection: "connecting", // connecting | live | polling | reconnecting | unauthorized
  serial: 0, // counts snapshots
  transitions: new Map(), // run name -> {from, to, serial}: state changes, so a view can point at them
  gpuHistory: new Map(), // GPU index -> [{t, util, mem}], the last few minutes as seen by this page
};

const GPU_HISTORY = 150;

/** fn("data" | "connection") is called after every change. Returns a function that unsubscribes. */
export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function notify(what) {
  for (const fn of [...listeners]) {
    try {
      fn(what);
    } catch (error) {
      console.error(error);
    }
  }
}

function setConnection(value) {
  if (store.connection === value) return;
  store.connection = value;
  notify("connection");
}

export function runByName(name) {
  const runs = store.snapshot ? store.snapshot.runs : [];
  return runs.find((run) => run.name === name) || null;
}

// ---- receiving -------------------------------------------------------------------------------

let source = null;
let pollTimer = 0;
let retryTimer = 0;
let failures = 0;
let lastMessage = 0;
let pollBusy = false;

function refreshInterval() {
  const seconds = store.snapshot && store.snapshot.interval > 0 ? store.snapshot.interval : 2;
  return Math.max(500, 1000 * seconds);
}

function receive(snapshot, via) {
  if (!snapshot || !Array.isArray(snapshot.runs)) return;
  const before = new Map((store.snapshot ? store.snapshot.runs : []).map((run) => [run.name, run.state]));
  store.serial += 1;
  for (const run of snapshot.runs) {
    const from = before.get(run.name);
    if (store.snapshot && from !== run.state)
      store.transitions.set(run.name, { from: from || null, to: run.state, serial: store.serial });
  }
  const now = Date.now();
  for (const gpu of Array.isArray(snapshot.gpus) ? snapshot.gpus : []) {
    const history = store.gpuHistory.get(gpu.index) || [];
    history.push({ t: now, util: gpu.util, mem: gpu.mem_total_mb ? gpu.mem_used_mb / gpu.mem_total_mb : null });
    if (history.length > GPU_HISTORY) history.splice(0, history.length - GPU_HISTORY);
    store.gpuHistory.set(gpu.index, history);
  }
  store.snapshot = snapshot;
  store.receivedAt = now;
  lastMessage = now;
  setConnection(via);
  notify("data");
}

function lost() {
  if (store.connection === "unauthorized") return;
  setConnection(store.snapshot ? "reconnecting" : "connecting");
}

function openStream() {
  clearTimeout(retryTimer);
  if (!("EventSource" in window)) {
    startPolling();
    return;
  }
  if (source) source.close();
  const stream = new EventSource("api/stream");
  source = stream;
  stream.onmessage = (event) => {
    let snapshot;
    try {
      snapshot = JSON.parse(event.data);
    } catch {
      return;
    }
    failures = 0;
    stopPolling();
    receive(snapshot, "live");
  };
  stream.onerror = () => {
    if (source !== stream) return;
    failures += 1;
    if (Date.now() - lastMessage > 2 * refreshInterval()) lost();
    if (stream.readyState === 2) {
      // Closed for good (the server answered with an error): the browser will not retry, so we do.
      stream.close();
      source = null;
      retryTimer = setTimeout(openStream, Math.min(10000, 1000 * Math.pow(2, Math.min(failures, 4))));
    }
    startPolling(); // until the stream is back, ask for the state ourselves
  };
}

function startPolling() {
  if (pollTimer) return;
  pollTimer = setInterval(poll, refreshInterval());
  poll();
}

function stopPolling() {
  clearInterval(pollTimer);
  pollTimer = 0;
}

async function poll() {
  if (pollBusy) return;
  pollBusy = true;
  try {
    const response = await fetch("api/state", { cache: "no-store" });
    if (response.status === 401 || response.status === 403) {
      setConnection("unauthorized");
      return;
    }
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    receive(await response.json(), "polling");
  } catch {
    lost();
  } finally {
    pollBusy = false;
  }
}

function watch() {
  const quiet = Date.now() - lastMessage;
  if (!store.snapshot || quiet < Math.max(4 * refreshInterval(), 8000)) return;
  lost();
  startPolling();
  if (source && source.readyState === 1 && quiet > 20000) openStream(); // open but silent: start over
}

export function start() {
  openStream();
  setInterval(watch, 1000);
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && Date.now() - lastMessage > 2 * refreshInterval()) poll();
  });
}

// ---- single evaluations and their videos -------------------------------------------------------

const CACHE_LIMIT = 48;
const cache = new Map();

async function fetchJson(url, attempts = 4) {
  for (let attempt = 0; ; attempt++) {
    const response = await fetch(url, { cache: "no-store" });
    if (response.ok) return response.json();
    if (response.status === 503 && attempt < attempts) {
      await new Promise((resolve) => setTimeout(resolve, 1500)); // the file is being written
      continue;
    }
    let message = `HTTP ${response.status}`;
    try {
      message = (await response.json()).error || message;
    } catch {
      /* not JSON */
    }
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
}

function cached(key, load) {
  let entry = cache.get(key);
  if (entry) {
    cache.delete(key);
    cache.set(key, entry); // most recently used last
    return entry;
  }
  entry = { promise: null, value: undefined };
  entry.promise = load().then((value) => {
    entry.value = value;
    return value;
  });
  entry.promise.catch(() => cache.delete(key)); // a failure is asked for again next time
  cache.set(key, entry);
  while (cache.size > CACHE_LIMIT) cache.delete(cache.keys().next().value);
  return entry;
}

function query(run, task, variant) {
  return `run=${encodeURIComponent(run)}&task=${encodeURIComponent(task)}&variant=${encodeURIComponent(variant || "")}`;
}

/**
 * One results.json with its episodes. `stamp` is anything that changes when the file does (the
 * success count of its row in the snapshot), so a re-run evaluation is fetched again.
 */
export function loadEvaluation(run, task, variant, stamp = "") {
  return cached(`evaluation/${run}/${task}/${variant}/${stamp}`, () =>
    fetchJson(`api/evaluation?${query(run, task, variant)}`),
  ).promise;
}

export function loadVideos(run, task, variant, stamp = "") {
  return cached(`videos/${run}/${task}/${variant}/${stamp}`, async () => {
    try {
      const list = await fetchJson(`api/videos?${query(run, task, variant)}`, 0);
      return Array.isArray(list) ? list : [];
    } catch (error) {
      if (error.status === 404) return [];
      throw error;
    }
  }).promise;
}
