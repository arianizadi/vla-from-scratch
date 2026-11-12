// The view and its selection live in the URL hash, so a view can be linked and survives a reload:
//   #/                      all runs
//   #/run/<name>?ckpt=7&w=ema   one run, with the evaluation that is inspected
//   #/compare?runs=a,b&w=ema    several runs on shared axes

function split() {
  const raw = window.location.hash.replace(/^#\/?/, "");
  const cut = raw.indexOf("?");
  return { path: cut < 0 ? raw : raw.slice(0, cut), params: new URLSearchParams(cut < 0 ? "" : raw.slice(cut + 1)) };
}

function decode(part) {
  try {
    return decodeURIComponent(part);
  } catch {
    return part;
  }
}

export function parseRoute() {
  const { path, params } = split();
  const parts = path.split("/").filter(Boolean).map(decode);
  if (parts[0] === "run" && parts[1]) return { view: "detail", run: parts[1], params };
  if (parts[0] === "compare") return { view: "compare", params };
  return { view: "overview", params };
}

export function runHref(name) {
  return `#/run/${encodeURIComponent(name)}`;
}

/** Change parameters of the current view without adding a step to the browser history. */
export function replaceParams(updates) {
  const { path, params } = split();
  for (const [key, value] of Object.entries(updates)) {
    if (value == null || value === "") params.delete(key);
    else params.set(key, String(value));
  }
  const query = params.toString().replace(/%2C/gi, ","); // lists of run names stay readable
  const hash = `#/${path}${query ? `?${query}` : ""}`;
  if (hash !== window.location.hash)
    window.history.replaceState(null, "", window.location.pathname + window.location.search + hash);
}

/** The server takes its access token from ?token= once and keeps it in a cookie; drop it from the address bar. */
export function hideToken() {
  const search = new URLSearchParams(window.location.search);
  if (!search.has("token")) return;
  search.delete("token");
  const query = search.toString();
  window.history.replaceState(null, "", window.location.pathname + (query ? `?${query}` : "") + window.location.hash);
}
