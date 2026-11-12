"""HTTP server behind the graphical training monitor.

Serves the browser UI in monitor_ui/ and, as JSON, what runstate.py reads from the training logs
and the evaluation results. One background thread refreshes a cached snapshot of all runs; no
request parses a log, so any number of open browsers cost the training machine the same. Standard
library only (Python 3.10 or newer), so it runs next to a training job without its environment.

    python -m sim.pickup.monitor_server                   # http://127.0.0.1:8765/, this machine only
    python -m sim.pickup.monitor_server --host 0.0.0.0    # on the network: prints a URL with a token

Routes (all GET):
    /                    the UI: index.html and the other files of monitor_ui/
    /api/state           the cached snapshot of all runs and GPUs
    /api/stream          the same snapshot as Server-Sent Events, one event per refresh
    /api/evaluation      ?run=&task=&variant=ckpt_<N>_<raw|ema>  one results.json, with its episodes
    /api/videos          ?run=&task=&variant=                    the rollout videos of that evaluation
    /api/tolerance       results/pickup/grasp_tolerance.json, when grasp_tolerance.py has written it
    /media/<path>        a rollout video, by its path under the results directory (supports Range)

Access. Without a token the server only listens on a loopback address and only answers requests
addressed to one, which is what an ssh port forward produces. On any other address a token is
required: it is accepted as ?token= in the URL, and from then on as a cookie.
"""

import argparse
import contextlib
import hmac
import ipaddress
import json
import math
import os
import re
import secrets
import signal
import socket
import socketserver
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote

from .runstate import Evals, discover, snapshot

UI_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor_ui")
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".map": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".txt": "text/plain; charset=utf-8",
}
MEDIA_TYPES = {".mp4": "video/mp4"}  # the only files of the results directory that /media serves
VARIANT = re.compile(r"ckpt_\d+_(raw|ema)")
ROLLOUT = re.compile(r"rollout_(\d+)_(.+)_(ok|fail)\.mp4")  # as eval.py names them
RANGE = re.compile(r"bytes=(\d*)-(\d*)")
ABSOLUTE_PATH = re.compile(r"/|~/|[A-Za-z]:[\\/]")
HEARTBEAT_SECONDS = 15.0
# The UI is self-contained, and the browser is told to hold it to that: nothing is loaded from, or
# sent to, any other origin.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; font-src 'self' data:; media-src 'self' blob:; connect-src 'self'; "
    "object-src 'none'; base-uri 'self'; form-action 'self'"
)


def scrub(value):
    """Data read from logs and result files, made safe to hand out as strict JSON.

    NaN and infinity become null (browsers reject them), and an absolute path, which would tell the
    reader about the machine and its user, is cut down to a relative one.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _relative(value) if ABSOLUTE_PATH.match(value) else value
    if isinstance(value, dict):
        return {str(k): scrub(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [scrub(v) for v in value]
    return value


def _relative(path: str) -> str:
    """An absolute path as it looks from the working directory, or just its last component."""
    here = os.getcwd()
    if os.path.normpath(path).startswith(here + os.sep):
        return os.path.relpath(path, here)
    return re.split(r"[\\/]", path.rstrip("\\/"))[-1]


def encode(value) -> bytes:
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode()


def child(directory: str, name: str) -> str | None:
    """`directory/name` if `name` is an entry of `directory`, else None.

    Every name that arrives in a request goes through here, so a request can only ever reach what
    a directory listing would show: no "..", no separators, no hidden files.
    """
    if not name or name.startswith(".") or "/" in name or "\\" in name or "\0" in name:
        return None
    try:
        return os.path.join(directory, name) if name in os.listdir(directory) else None
    except OSError:
        return None


def inside(path: str, root: str) -> bool:
    """Whether `path`, with symbolic links resolved, is still under `root`."""
    root, path = os.path.realpath(root), os.path.realpath(path)
    return os.path.commonpath([root, path]) == root


def descend(root: str, names) -> str | None:
    """Follow `names` down from `root` one listed entry at a time; None if any step is not there."""
    path = root
    for name in names:
        path = child(path, name)
        if path is None:
            return None
    return path if inside(path, root) else None


def byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    """(first, last) byte asked for by a Range header, or None to send the whole file.

    Raises ValueError for a range that lies outside the file.
    """
    m = RANGE.fullmatch(header.strip()) if header else None
    if not m or not (m.group(1) or m.group(2)):
        return None  # absent, malformed or several ranges: ignoring the header is allowed
    if m.group(1):
        first = int(m.group(1))
        last = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
    else:  # "bytes=-N": the last N bytes
        first, last = max(size - int(m.group(2)), 0), size - 1
        if int(m.group(2)) == 0:
            raise ValueError("empty range")
    if first >= size or first > last:
        raise ValueError("range outside the file")
    return first, last


def is_loopback(host: str | None) -> bool:
    """Whether `host` (an address or a name, without port) can only be this machine."""
    if not host:
        return False
    host = host.strip("[]").lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def host_of(header: str | None) -> str | None:
    """The host part of a Host header ("name:port", "[v6]:port")."""
    if not header:
        return None
    header = header.strip()
    if header.startswith("["):
        return header[1:].split("]")[0]
    return header.rsplit(":", 1)[0] if header.count(":") == 1 else header


def access_token(host: str, token: str | None) -> str | None:
    """The token the server will ask for: the given one, a new one off loopback, else none."""
    if token:
        return token
    return None if is_loopback(host) else secrets.token_urlsafe(24)


class State:
    """The cached snapshot of all runs, refreshed by one background thread."""

    def __init__(self, experiments: str, results: str, interval: float = 2.0):
        self.experiments, self.results, self.interval = experiments, results, interval
        self.evals = Evals(results)
        self.known = {}
        self.data = dict(time=time.time(), runs=[], gpus=[], interval=interval, error=None)
        self.payload = encode(self.data)
        self.version = 0
        self.changed = threading.Condition()
        self.stopped = threading.Event()
        self.thread = None
        self.last_error = None

    def refresh(self):
        try:
            data = scrub(snapshot(discover(self.experiments, self.known), self.evals))
            data.update(interval=self.interval, error=None)
            self.last_error = None
        except Exception as error:  # a file changed under us, a result in an unknown format: keep the last state
            data = dict(self.data, error=f"refresh failed ({type(error).__name__})")
            if self.last_error != repr(error):
                self.last_error = repr(error)
                traceback.print_exc()
        payload = encode(data)
        with self.changed:
            self.data, self.payload, self.version = data, payload, self.version + 1
            self.changed.notify_all()

    def start(self):
        self.refresh()
        self.thread = threading.Thread(target=self._loop, name="monitor-refresh", daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.stopped.wait(self.interval):
            self.refresh()

    def stop(self):
        self.stopped.set()
        with self.changed:
            self.changed.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def wait(self, seen: int, timeout: float) -> tuple[int, bytes]:
        """Block until the snapshot is newer than version `seen`, or `timeout`, or shutdown."""
        with self.changed:
            self.changed.wait_for(lambda: self.version != seen or self.stopped.is_set(), timeout)
            return self.version, self.payload

    def evaluation_directory(self, run: str, task: str, variant: str) -> str | None:
        """The directory of one evaluation, if all three names exist on disk.

        An empty `variant` is the evaluation written directly into `<run>/<task>/` (eval.sh without
        a named checkpoint).
        """
        if variant and not VARIANT.fullmatch(variant):
            return None
        path = descend(self.results, [run, task, variant] if variant else [run, task])
        return path if path is not None and os.path.isdir(path) else None

    def media_file(self, relative: str) -> str | None:
        """A video under the results directory, by its relative path."""
        if os.path.splitext(relative)[1].lower() not in MEDIA_TYPES:
            return None
        path = descend(self.results, relative.split("/"))
        return path if path is not None and os.path.isfile(path) else None

    def videos(self, run: str, task: str, variant: str) -> list[dict] | None:
        directory = self.evaluation_directory(run, task, variant)
        if directory is None:
            return None
        out = []
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            if name.startswith(".") or os.path.splitext(name)[1].lower() not in MEDIA_TYPES:
                continue
            try:  # only what /media will serve: a file, and not a link that leads out of the results
                size = os.path.getsize(path) if os.path.isfile(path) and inside(path, self.results) else None
            except OSError:  # deleted since it was listed
                size = None
            if size is None:
                continue
            m = ROLLOUT.fullmatch(name)
            out.append(
                dict(
                    name=name,
                    url="/media/" + quote("/".join(p for p in (run, task, variant, name) if p)),
                    bytes=size,
                    episode=int(m.group(1)) if m else None,
                    target=m.group(2) if m else None,
                    success=m.group(3) == "ok" if m else None,
                )
            )
        return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 60  # seconds an idle or stuck connection may hold its thread

    def version_string(self):
        return "pickup-monitor"  # not the default, which names the Python version

    def do_GET(self):
        self._serve()

    def do_HEAD(self):
        self._serve()

    def _serve(self):
        self.new_cookie = False
        try:
            path, _, query = self.path.partition("?")
            query = {k: v[-1] for k, v in parse_qs(query, keep_blank_values=True).items()}
            if self._allowed(query):
                self._route(unquote(path), query)
        except (ConnectionError, TimeoutError):  # the browser went away
            self.close_connection = True
        except Exception:
            traceback.print_exc()
            self.close_connection = True
            with contextlib.suppress(Exception):
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, dict(error="internal error"))

    # ---- access ----------------------------------------------------------------------------

    @property
    def _cookie_name(self) -> str:
        return f"monitor_token_{self.server.server_port}"  # cookies ignore ports: one name per server

    def _allowed(self, query) -> bool:
        token = self.server.token
        if token is None:
            # No token means loopback only: the request must come from this machine and be addressed
            # to it. The Host header is what keeps a web page that points its own domain at 127.0.0.1
            # (DNS rebinding) from reading the state.
            host = self.headers.get("Host")
            if is_loopback(self.client_address[0]) and (host is None or is_loopback(host_of(host))):
                return True
            self._text(
                HTTPStatus.FORBIDDEN,
                "This monitor only answers requests addressed to localhost. Reach it through an ssh port "
                "forward, or start it with --host and open the URL with the token that it prints.\n",
            )
            return False
        if "token" in query and hmac.compare_digest(query["token"].encode(), token.encode()):
            self.new_cookie = True
            return True
        for part in self.headers.get("Cookie", "").split(";"):
            name, _, value = part.strip().partition("=")
            if name == self._cookie_name and hmac.compare_digest(unquote(value).encode(), token.encode()):
                return True
        self._text(
            HTTPStatus.UNAUTHORIZED,
            "A token is required. Open the URL that the server printed when it started: it ends in ?token=...\n",
        )
        return False

    # ---- routes ----------------------------------------------------------------------------

    def _route(self, path: str, query):
        state = self.server.state
        if path == "/api/state":
            self._send(HTTPStatus.OK, state.payload, "application/json")
        elif path == "/api/stream":
            self._stream()
        elif path == "/api/evaluation":
            directory = state.evaluation_directory(
                query.get("run", ""), query.get("task", ""), query.get("variant", "")
            )
            self._json_file(directory and os.path.join(directory, "results.json"), "no such evaluation")
        elif path == "/api/videos":
            videos = state.videos(query.get("run", ""), query.get("task", ""), query.get("variant", ""))
            if videos is None:
                self._json(HTTPStatus.NOT_FOUND, dict(error="no such evaluation"))
            else:
                self._json(HTTPStatus.OK, videos)
        elif path == "/api/tolerance":
            self._json_file(child(state.results, "grasp_tolerance.json"), "no grasp tolerance results")
        elif path.startswith("/api/"):
            self._json(HTTPStatus.NOT_FOUND, dict(error="no such route"))
        elif path.startswith("/media/"):
            self._media(state.media_file(path[len("/media/") :]))
        else:
            self._static(path)

    def _json_file(self, path: str | None, missing: str):
        if not path or not os.path.isfile(path):
            return self._json(HTTPStatus.NOT_FOUND, dict(error=missing))
        try:
            content = self.server.state.evals.load(path)
        except (OSError, ValueError):  # being written right now
            return self._json(HTTPStatus.SERVICE_UNAVAILABLE, dict(error="results are not readable yet"))
        self._json(HTTPStatus.OK, scrub(content))  # files written by other programs: clean them on the way out

    def _static(self, path: str):
        relative = path.lstrip("/")
        if not relative or relative.endswith("/"):
            relative += "index.html"
        file = descend(self.server.ui, relative.split("/"))
        if file is None or not os.path.isfile(file):
            missing_ui = relative == "index.html" and "The UI files (monitor_ui/) are missing; /api/state answers.\n"
            return self._text(HTTPStatus.NOT_FOUND, missing_ui or "Not found.\n")
        with open(file, "rb") as f:
            body = f.read()
        content_type = CONTENT_TYPES.get(os.path.splitext(file)[1].lower(), "application/octet-stream")
        self._send(HTTPStatus.OK, body, content_type, {"Cache-Control": "no-cache"})

    def _media(self, file: str | None):
        if file is None:
            return self._text(HTTPStatus.NOT_FOUND, "Not found.\n")
        try:
            with open(file, "rb") as f:
                self._ranged(f, MEDIA_TYPES[os.path.splitext(file)[1].lower()])
        except FileNotFoundError:  # deleted since it was found
            self._text(HTTPStatus.NOT_FOUND, "Not found.\n")

    def _ranged(self, f, content_type: str):
        """An open file, or the part of it that the Range header asks for."""
        size = os.fstat(f.fileno()).st_size
        headers = {"Accept-Ranges": "bytes", "Cache-Control": "no-cache"}
        try:
            span = byte_range(self.headers.get("Range"), size)
        except ValueError:
            headers["Content-Range"] = f"bytes */{size}"
            return self._send(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, b"", "text/plain", headers)
        first, last = span or (0, size - 1)
        if span:
            headers["Content-Range"] = f"bytes {first}-{last}/{size}"
        self._head(HTTPStatus.PARTIAL_CONTENT if span else HTTPStatus.OK, content_type, last - first + 1, headers)
        if self.command == "HEAD":
            return
        f.seek(first)
        left = last - first + 1
        while left > 0:
            block = f.read(min(left, 1 << 16))
            if not block:  # the file shrank: the promised length cannot be kept
                self.close_connection = True
                return
            self.wfile.write(block)
            left -= len(block)

    def _stream(self):
        """Server-Sent Events: the snapshot now, then again after every refresh."""
        self._head(HTTPStatus.OK, "text/event-stream; charset=utf-8", None, {"X-Accel-Buffering": "no"})
        if self.command == "HEAD":
            return
        state = self.server.state
        self.wfile.write(b"retry: 2000\n\n")
        seen = -1
        while not state.stopped.is_set():
            version, payload = state.wait(seen, HEARTBEAT_SECONDS)
            if version != seen:
                seen = version
                self.wfile.write(b"id: %d\ndata: %s\n\n" % (version, payload))
            else:
                self.wfile.write(b": waiting\n\n")  # a comment: keeps the connection open, shows up nowhere

    # ---- responses -------------------------------------------------------------------------

    def _head(self, status, content_type: str, length: int | None, headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if length is None:
            self.send_header("Connection", "close")  # a body without a length ends with the connection
        else:
            self.send_header("Content-Length", str(length))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        if content_type.startswith("text/html"):
            self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        if self.new_cookie:
            value = quote(self.server.token, safe="")
            self.send_header("Set-Cookie", f"{self._cookie_name}={value}; Path=/; HttpOnly; SameSite=Lax")
        for name, value in {"Cache-Control": "no-store", **(headers or {})}.items():
            self.send_header(name, value)
        self.end_headers()

    def _send(self, status, body: bytes, content_type: str, headers=None):
        self._head(status, content_type, len(body), headers)
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status, value):
        self._send(status, encode(value), "application/json")

    def _text(self, status, text: str):
        self._send(status, text.encode(), "text/plain; charset=utf-8")

    def log_message(self, format, *args):
        if self.server.verbose:
            line = re.sub(r"token=[^&\s\"]+", "token=...", format % args)
            sys.stderr.write(f"{self.log_date_time_string()} {self.address_string()} {line}\n")


class MonitorServer(ThreadingHTTPServer):
    """The HTTP server. `state.start()` must have been called; `close()` ends both."""

    daemon_threads = True

    def __init__(self, address, state: State, token: str | None = None, ui: str = UI_DIRECTORY, verbose=False):
        self.state, self.token, self.ui, self.verbose = state, token, ui, verbose
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, Handler)

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)  # not HTTPServer's: it looks the machine's name up, slowly
        self.server_name, self.server_port = self.server_address[:2]

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], ConnectionError | TimeoutError):  # a closed tab is not an error
            super().handle_error(request, client_address)

    def close(self):
        self.state.stop()  # ends the event streams
        self.shutdown()
        self.server_close()


def _address_to_open(host: str) -> str:
    """What to put in the printed URL for the address the server listens on."""
    if host in ("", "0.0.0.0", "::"):  # every interface: name the one that other machines reach
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("192.0.2.1", 9))  # no packet is sent; this only asks the routing table
                host = s.getsockname()[0]
        except OSError:
            return "<address-of-this-machine>"
    return f"[{host}]" if ":" in host else host


def _watch_parent(stop: threading.Event):
    parent = os.getppid()
    while not stop.wait(1.0):
        if os.getppid() != parent:
            stop.set()


def _say(text: str):
    with contextlib.suppress(OSError, ValueError):  # the terminal may already be gone
        print(text, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description="Web server of the graphical training monitor.")
    p.add_argument("--host", default="127.0.0.1", help="address to listen on (default: this machine only)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--experiments", default="experiments/pickup")
    p.add_argument("--results", default="results/pickup")
    p.add_argument("--interval", type=float, default=2.0, help="seconds between refreshes of the state")
    p.add_argument("--token", default=None, help="require this token (one is generated when --host is not loopback)")
    p.add_argument("--verbose", action="store_true", help="log every request")
    p.add_argument(
        "--exit-with-parent",
        action="store_true",
        help="stop when the process that started the server ends (scripts/monitor_remote.sh: with the ssh session)",
    )
    args = p.parse_args(argv)

    token = access_token(args.host, args.token)
    state = State(args.experiments, args.results, max(args.interval, 0.2))
    try:
        server = MonitorServer((args.host, args.port), state, token, verbose=args.verbose)
    except OSError as error:
        sys.exit(f"monitor: cannot listen on {args.host} port {args.port} ({error.strerror}); try another --port")

    stop = threading.Event()
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        number = getattr(signal, name, None)
        # A signal that is ignored stays ignored (nohup, a background job), unless the server belongs
        # to the session that started it: then Ctrl-C and a hangup must always end it.
        if number is not None and (args.exit_with_parent or signal.getsignal(number) is not signal.SIG_IGN):
            signal.signal(number, lambda *_: stop.set())
    state.start()
    if args.exit_with_parent:
        threading.Thread(target=_watch_parent, args=(stop,), daemon=True).start()
    threading.Thread(target=server.serve_forever, name="monitor-http", daemon=True).start()

    url = f"http://{_address_to_open(args.host)}:{server.server_port}/" + (f"?token={quote(token)}" if token else "")
    _say(f"monitor: open {url}   (reading {args.experiments} and {args.results}; Ctrl-C stops)")
    if not os.path.isfile(os.path.join(server.ui, "index.html")):
        print("monitor: the UI files (monitor_ui/index.html) are missing; only /api/ will answer", file=sys.stderr)
    try:
        stop.wait()
    finally:
        server.close()
        _say("monitor: stopped")


if __name__ == "__main__":
    main()
