"""The slow server of architecture v2 (tutorial/architecture-v2.md): serves the eyes (eyes.py) and the program writer
(program_writer.py) to the slow level of a closed-loop evaluation (slow_level.py, eval.py --slow_level v2) over a
local socket.

The simulator runs in its own process and environment, as with the policy server (policy_server.py), and asks this
server for what needs torch and a GPU: a percept of the images it sends, at every look, and the program of an
instruction, once per episode. The protocol is the policy server's: multiprocessing.connection with the same authkey,
one request and one reply at a time. The port is bound before anything is loaded, so that a port already taken fails at
once; "listening on" is printed once everything is loaded (the scripts wait for it).

  python -m sim.pickup.slow_server --port 6300 --eyes <eyes run dir> --writer Qwen/Qwen3-VL-2B-Thinking

Requests, as dicts with "cmd":

  info      -> {"eyes": the eyes' config as a dict with its image keys under "keys" (None without eyes), "eyes_run",
                "eyes_checkpoint", "eyes_weights", "writer": the language model's name (None without one),
                "writer_weights", "constrained", "shots"}
  see       {"images": {"front_t0": JPEG bytes, ...}} -> {"scene": the percept (scene.py's percept form), "latency_s"}
  write     {"instruction": "..."} -> {"program": the text written, "complete": it parses, "latency_s"}
  shutdown  -> {"ok": True}, and the server exits

A request for a part that was not loaded, and one that fails, is answered with {"error": "..."}, and the server goes
on serving. Either part can be left out: the eyes alone for the learned scene with the task's own program, the writer
alone for written programs on the simulator's truth.
"""

import argparse
import dataclasses
import time
import traceback
from multiprocessing.connection import Listener

from .eyes import checkpoints, load_eyes
from .program_writer import LONGEST_PROGRAM, ProgramWriter, load_lm

AUTHKEY = b"pickup-sim"  # policy_server.AUTHKEY (not imported: that module loads the policy's code)


class SlowServer:
    """Answers one request at a time (`handle`). `eyes`: an Eyes in eval mode (eyes.load_eyes), or None; `writer`: a
    ProgramWriter, or None. `about`: what the info request reports besides the eyes' config (the run, the checkpoint,
    the model's name, ...)."""

    def __init__(self, eyes=None, writer=None, about: dict | None = None):
        self.eyes, self.writer = eyes, writer
        self.about = dict(about or {})

    def info(self) -> dict:
        eyes = None
        if self.eyes is not None:
            config = self.eyes.config
            eyes = dict(dataclasses.asdict(config), keys=list(config.keys))
        return dict(
            eyes=eyes,
            eyes_run=self.about.get("eyes_run"),
            eyes_checkpoint=self.about.get("eyes_checkpoint"),
            eyes_weights=self.about.get("eyes_weights"),
            writer=self.about.get("writer") if self.writer is not None else None,
            writer_weights=self.about.get("writer_weights"),
            constrained=bool(self.writer.constrained) if self.writer is not None else False,
            shots=len(self.writer.examples) if self.writer is not None else None,
        )

    def handle(self, request: dict) -> dict:
        """The reply to one request (shutdown is the serving loop's)."""
        cmd = request.get("cmd")
        if cmd == "info":
            return self.info()
        if cmd not in ("see", "write"):
            return {"error": f"unknown request {cmd!r}: info, see, write or shutdown"}
        if cmd == "see" and self.eyes is None:
            return {"error": "no eyes loaded: start the slow server with --eyes"}
        if cmd == "write" and self.writer is None:
            return {"error": "no program writer loaded: start the slow server with --writer"}
        began = time.perf_counter()
        try:
            if cmd == "see":
                return {"scene": self.eyes.see(request["images"]), "latency_s": time.perf_counter() - began}
            out = self.writer.write(str(request["instruction"]))
            return {
                "program": out["program"],
                "complete": bool(out["complete"]),
                "latency_s": time.perf_counter() - began,
            }
        except Exception as error:  # the client stops with the message; the server stays up for the next request
            traceback.print_exc()
            return {"error": f"{cmd}: {type(error).__name__}: {error}"}


def serve(listener: Listener, server: SlowServer):
    """Answer requests until a shutdown request, one connection after another."""
    while True:
        with listener.accept() as conn:
            while True:
                try:
                    request = conn.recv()
                except EOFError:
                    break
                if request.get("cmd") == "shutdown":
                    conn.send({"ok": True})
                    return
                conn.send(server.handle(request))


def load(args) -> SlowServer:
    """The parts the command line asks for."""
    eyes = writer = None
    about = {}
    if args.eyes:
        # the averaged weights are an Eyes checkpoint's "ema" entry, the trained ones its "model" entry
        eyes = load_eyes(args.eyes, args.eyes_checkpoint, "ema" if args.eyes_weights == "ema" else "model", args.device)
        number = args.eyes_checkpoint if args.eyes_checkpoint is not None else max(checkpoints(args.eyes))
        about.update(eyes_run=args.eyes, eyes_checkpoint=number, eyes_weights=args.eyes_weights)
        print(f"[slow_server] eyes: {args.eyes}, checkpoint {number} ({args.eyes_weights})", flush=True)
    if args.writer:
        model, tokenizer = load_lm(args.writer, weights=args.writer_weights, device=args.device)
        # one token more than the longest program: a constrained program always ends whole (program_writer.py)
        writer = ProgramWriter(
            model,
            tokenizer,
            shots=args.shots,
            constrained=not args.unconstrained,
            max_new_tokens=LONGEST_PROGRAM + 1,
        )
        about.update(writer=args.writer, writer_weights=args.writer_weights)
        kind = "free writing" if args.unconstrained else "held to the grammar"
        print(f"[slow_server] writer: {args.writer}, {args.shots} worked examples, {kind}", flush=True)
    return SlowServer(eyes, writer, about)


def main(argv=None):
    ap = argparse.ArgumentParser(description="serve the eyes and the program writer to eval.py --slow_level v2")
    ap.add_argument("--port", type=int, required=True, help="0: any free port (the one bound is printed)")
    ap.add_argument("--eyes", default=None, help="a run directory of train_eyes.py")
    ap.add_argument("--eyes_checkpoint", type=int, default=None, help="its checkpoint number (default: the latest)")
    ap.add_argument("--eyes_weights", choices=["ema", "raw"], default="ema", help="averaged or trained weights")
    ap.add_argument("--writer", default=None, help="a language model: a Hugging Face name or a local directory")
    ap.add_argument(
        "--writer_weights", default=None, help="a reason_then_act checkpoint whose language model is loaded over it"
    )
    ap.add_argument("--shots", type=int, default=8, help="worked examples in the writer's prompt")
    ap.add_argument("--unconstrained", action="store_true", help="the writer writes freely (the baseline)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    # Bind before loading: if the port is taken this fails at once, not after a client has reached whoever holds it
    with Listener(("127.0.0.1", args.port), authkey=AUTHKEY) as listener:
        server = load(args)
        port = listener.address[1]
        print(f"[slow_server] listening on 127.0.0.1:{port}", flush=True)
        serve(listener, server)


if __name__ == "__main__":
    main()
