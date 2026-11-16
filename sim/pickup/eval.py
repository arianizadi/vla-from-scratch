"""Closed-loop evaluation of a policy served by `policy_server.py` in the pickup sim.

Evaluation seeds start at 1_000_000 and are disjoint from the demonstration seeds, so every
evaluated layout is unseen. Writes results.json (+ rollout videos) to --out.

  python -m sim.pickup.eval --task pick_cube --episodes 50 --port 6200 --out results/pickup/<run>

Reasoning tasks (reason_*) are reported by kind, phrasing and number of shoots, with the correct notes (notes.py)
written alongside every rollout. Three ways of running a policy on them:

  plain          the policy reads the instruction (as for every other task)
  reason-then-act  the server's policy writes a note every chunk and acts on it (model type reason_then_act);
                 --notes expert hands it the correct note instead, --notes swap a note about another shoot
  two-level      --subgoals: the served policy reads a sub-goal ("cut the middle shoot") as its instruction, from a
                 reasoner served on --reasoner_port (reasoner), from the correct notes (expert), or about another
                 shoot (swap)

The swap modes are the test that the text matters: if the policy cuts the shoot the note names rather than the one the
instruction means, it acts on the text. With --expert and --reasoner_port, the expert drives and the reasoner's notes
are compared with the correct ones at every chunk; --reason_check does the same with the served policy as the
reasoner (to evaluate a reasoner's checkpoints with eval.sh / eval_checkpoints.sh).

--see expert hands whoever writes the notes (a reasoner, or a policy that writes its own) the `see` field of the
correct note, what is on the stand, and lets it write the rest: how well it reasons when it sees correctly
(tutorial/architecture-v2.md, experiment E0).

--slow_level v2 puts the slow level of architecture v2 (slow_level.py) over the served policy, whose instruction is
then the slow level's sub-goal. Its scene memory takes the simulator's truth (--scene truth) or the percepts of the
eyes (--scene eyes), and it carries out the task's own program (--program gold) or the one a language model writes
(--program writer); the eyes and the writer are served by slow_server.py on --slow_port. While a fact is missing it
holds the arm still; with --recover N it hands the arm to a scripted move when nothing progresses, at most N times per
episode; the episode ends once its verdict (complete or failed) has stood for a while. Each episode records what it
decided (per_episode[i]["v2"]), and results.json counts its false completions apart from the successes.

--scenario applies test situations (scenarios.py: lighting, camera, wind, bump, sticky, nothing_to_do, order, ...) on
top of the task; each episode records what its scenarios drew, and results.json counts the outcomes by variation.
Episodes of the same seeds with and without a scenario have the same layouts.

The aim range's drills (aim_*, aim.py) are reported per drill: success, wrong targets, the steps of successful episodes
and how close the gripper came (aim_each takes the drills in turn, for equal numbers of each).
"""

import argparse
import io
import json
import re
import sys
import time
from collections import defaultdict, deque
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np
from PIL import Image

from . import reasoning
from .diagnostics import GraspProbe, summarize
from .disturb import Disturbance
from .env import BUTTONS, CONTROL_HZ, PRESS_TRAVEL, REACH_AZIMUTH, PickupEnv

AUTHKEY = b"pickup-sim"  # must match policy_server.AUTHKEY (not imported: that module needs torch)
EVAL_SEED_START = 1_000_000


def write_json(path, data, **kwargs):
    """Write through a temporary file, so that a monitor reading `path` never sees half of it."""
    partial = path.with_name(path.name + ".partial")
    with open(partial, "w") as f:
        json.dump(data, f, **kwargs)
    partial.replace(path)


def _jpeg(img, quality=95):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def connect(port, timeout_s=600):
    t0 = time.time()
    while True:
        try:
            return Client(("127.0.0.1", port), authkey=AUTHKEY)
        except (ConnectionRefusedError, OSError):
            if time.time() - t0 > timeout_s:
                raise
            time.sleep(1.0)


def image_offsets(image_names):
    """The frame offsets in a policy's image names ("front_t-20" -> -20), as the policy server reports them."""
    return sorted({int(name.rsplit("_t", 1)[1]) for name in image_names})


class FrameHistory:
    """The observations of an episode that a policy's images can reach back to, newest last.

    at(k) is the observation k control steps before the newest (k <= 0). An offset that reaches back before the first
    observation of the episode gives the first one: the training windows do the same (the preprocessing clips the
    frame index at the start of the episode). Offsets -1 and 0 are always kept, so that every request carries the
    images and joint states at t-1 and t whatever else the policy takes.
    """

    def __init__(self, first_obs, offsets=(-1, 0)):
        self.offsets = sorted({int(k) for k in offsets} | {-1, 0})
        if self.offsets[-1] > 0:
            raise ValueError(f"image offsets must be <= 0 (frames from the future): {self.offsets}")
        self.frames = deque([first_obs], maxlen=1 - self.offsets[0])

    def push(self, obs):
        self.frames.append(obs)

    def at(self, k):
        return self.frames[max(0, len(self.frames) - 1 + k)]


def infer_request(
    prev_obs,
    obs,
    prev_action,
    instruction,
    flow_steps,
    noise_seed=None,
    images_from=None,
    history=None,
    note_held=None,
    note_now=None,
    see=None,
):
    """The message that asks the policy server for an action chunk: both cameras' images, the joint states at t-1
    and t, the command the robot is under and the instruction. The images are those of `prev_obs` and `obs` (t-1 and
    t), or, with `history` (a FrameHistory whose newest observation is `obs`), one per offset it keeps, for policies
    that look further back. `images_from`: a pair of observations to take the images from instead (to show the policy
    stale images): the second for t, the first for every earlier offset. `note_held` / `note_now`: for a policy that
    writes notes, the note the client holds and a note to use instead of writing one; `see`: the first field of the
    note it is to write, given from outside (policy_server.py)."""
    if history is None:
        history = FrameHistory(prev_obs)
        history.push(obs)
    shown = history
    if images_from is not None:
        shown = FrameHistory(images_from[0], history.offsets)
        shown.push(images_from[1])
    images = {}
    for k in history.offsets:
        frame = shown.at(k)
        for cam in PickupEnv.camera_names:
            images[f"{cam}_t{k}"] = _jpeg(frame[cam])
    return dict(
        cmd="infer",
        images=images,
        state=[prev_obs["state"].tolist(), obs["state"].tolist()],
        prev_action=np.asarray(prev_action).tolist(),
        instruction=instruction,
        num_flow_steps=flow_steps,
        noise_seed=noise_seed,
        **({"note_held": note_held} if note_held is not None else {}),
        **({"note_now": note_now} if note_now is not None else {}),
        **({"see": see} if see is not None else {}),
    )


def write_video(path, frames, fps=20):
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
        for f in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def trace_step(env, obs, action):
    """One control step of a rollout as plain numbers, recorded before `action` is applied."""

    def r(values):
        return [round(float(v), 4) for v in values]

    return dict(
        tcp=r(env.tcp_pos()),
        yaw=round(env.tcp_yaw(), 4),
        grip=round(float(obs["state"][7]), 4),
        q=r(obs["state"][:7]),
        action=r(action),
        objects={n: r([*env.object_pos(n), env.object_yaw(n)]) for n in env.active},
    )


def check_server(info, port, run, number, weights):
    """Stop if the server on `port` is not serving the checkpoint this evaluation is for.

    Two evaluations started on the same port both reach whichever server bound it first. Without this
    check the second one silently evaluates the other run's policy and files the result under its own
    name.
    """
    served = str(info.get("checkpoint", "")).replace("\\", "/")
    parts = served.split("/")
    m = re.fullmatch(r"(ema|checkpoint)_(\d+)\.pt", parts[-1])
    problems = []
    if run is not None and (len(parts) < 3 or parts[-3] != run):
        problems.append(f"run '{run}'")
    if number is not None and (m is None or int(m.group(2)) != number):
        problems.append(f"checkpoint {number}")
    if weights is not None and (m is None or (m.group(1) == "ema") != (weights == "ema")):
        problems.append(f"{weights} weights")
    if problems:
        sys.exit(
            f"[eval] the policy server on port {port} is serving {served}, not {', '.join(problems)}. "
            "Another evaluation is using this port; give each one its own (eval.sh picks a free port "
            "unless PICKUP_PORT is set)."
        )


def _wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _turning_summary(episodes):
    """Pruning: how efficiently the plant was turned. For one-shoot tasks, `needed` is the turn that brings the shoot
    to point at the robot the short way; a first press is `short_way` when it turns that way."""
    eps = [e for e in episodes if e.get("turned_deg") is not None]
    if not eps:
        return None
    pressed = [e for e in eps if e["first_press"] is not None and abs(e["start_offset_deg"]) > 25]
    ok = [e for e in eps if e["success"]]
    return dict(
        mean_turned_deg_success=float(np.mean([e["turned_deg"] for e in ok])) if ok else None,
        mean_needed_deg_success=float(np.mean([abs(e["start_offset_deg"]) for e in ok])) if ok else None,
        first_press_short_way=sum(e["first_press"] == np.sign(e["start_offset_deg"]) for e in pressed) / len(pressed)
        if pressed
        else None,
        pressed_when_out_of_reach=len(pressed) / max(1, sum(abs(e["start_offset_deg"]) > 25 for e in eps)),
    )


def _named_summary(episodes):
    """Named shoots: the named shoot cut cleanly (success), another shoot cut (wrong shoot), the plant damaged and out
    of time, as counts, overall and by phrasing kind ('train', or a kind of held-out phrasing), by named position and
    by the number of shoots on the plant."""

    def counts(eps):
        return dict(
            episodes=len(eps),
            successes=sum(e["success"] for e in eps),
            wrong_shoot=sum(bool(e["wrong_shoots"]) for e in eps),
            damaged=sum(e["damaged"] is not None for e in eps),
            timeout=sum(not e["success"] and not e["wrong_shoots"] and e["damaged"] is None for e in eps),
        )

    def by(field):
        return {str(v): counts([e for e in episodes if e[field] == v]) for v in sorted({e[field] for e in episodes})}

    n = len(episodes)
    total = counts(episodes)
    return dict(
        **total,
        wrong_shoot_rate=total["wrong_shoot"] / n,
        damaged_rate=total["damaged"] / n,
        by_phrasing_kind=by("phrasing_kind"),
        by_position=by("named"),
        by_shoots=by("shoots"),
    )


def _aim_summary(episodes):
    """Aim range: success, wrong targets, time and accuracy per drill (aim.summary)."""
    from .aim import summary

    return summary(episodes)


NOTE_SAME = "same"  # what a policy writes when the note it holds still stands (vla_foundry/data/processor/notes.py)


def note_agreement(written: str, correct: str) -> dict:
    """Which fields of a written note match the correct one (exact text after normalising spaces)."""
    a, b = reasoning.note_fields(written), reasoning.note_fields(correct)
    return {k: " ".join(a.get(k, "").split()) == " ".join(b[k].split()) for k in reasoning.NOTE_FIELDS if k in b}


def swap_shoot(env, seed):
    """For the test that the text matters: a shoot the instruction does not mean, chosen per seed."""
    others = [n for n in env.active if n not in env.targets]
    rng = np.random.default_rng([seed, 41])
    return others[int(rng.integers(len(others)))] if others else None


def _reason_summary(episodes):
    """Reasoning tasks: success, a shoot cut that was not meant, damage and running out of time, overall and by kind, by
    phrasing (train or a kind of held-out instruction), by number of shoots, and for the hidden kind by whether the card
    is out of view once the shoot faces the robot. With notes: how often each field of the policy's notes was right.
    With a swap: how often the policy cut the shoot the note named, and how often one the instruction meant."""

    def counts(eps):
        out = dict(
            episodes=len(eps),
            successes=sum(e["success"] for e in eps),
            wrong_shoot=sum(bool(e["wrong_shoots"]) for e in eps),
            damaged=sum(e["damaged"] is not None for e in eps),
            timeout=sum(not e["success"] and not e["wrong_shoots"] and e["damaged"] is None for e in eps),
        )
        swapped = [e for e in eps if e.get("swap_shoot")]
        if swapped:
            out["cut_note_shoot"] = sum(e["swap_shoot"] in e["cut_order"] for e in swapped)
            out["cut_meant_shoot"] = sum(any(c in e["targets_named"] for c in e["cut_order"]) for e in swapped)
        return out

    def by(field):
        values = sorted({str(e[field]) for e in episodes})
        return {v: counts([e for e in episodes if str(e[field]) == v]) for v in values}

    hidden = [e for e in episodes if e["kind"] == "hidden"]
    agreement = [a for e in episodes for a in e.get("note_agreement", [])]
    return dict(
        **counts(episodes),
        by_kind=by("kind"),
        by_phrasing=by("phrasing"),
        by_shoots=by("shoots"),
        hidden_by_card_at_reach={
            "out of view": counts([e for e in hidden if e["label_hidden_at_reach"]]),
            "in view": counts([e for e in hidden if not e["label_hidden_at_reach"]]),
        }
        if hidden
        else None,
        notes_right={k: float(np.mean([a[k] for a in agreement if k in a])) for k in reasoning.NOTE_FIELDS}
        if agreement
        else None,
        notes_written=sum(len(e.get("notes_written", [])) for e in episodes),
    )


def _scenario_summary(env, episodes):
    """Scenarios: success, damage, a wrong shoot (or a wrong order) and running out of time, overall and by the mode
    each scenario drew (or 'not applied')."""

    def counts(eps):
        return dict(
            episodes=len(eps),
            successes=sum(e["success"] for e in eps),
            damaged=sum(e["damaged"] is not None for e in eps),
            wrong=sum(e["wrong_object"] and e["damaged"] is None for e in eps),
            timeout=sum(not e["success"] and not e["wrong_object"] and e["damaged"] is None for e in eps),
            **(
                dict(
                    distractors_touched=sum(bool(e["distractors_touched"]) for e in eps),
                    distractors_moved=sum(bool(e["distractors_moved"]) for e in eps),
                )
                if eps and "distractors_touched" in eps[0]
                else {}
            ),
        )

    by_mode = {}
    for s in env.scenarios:
        modes = [e["scenario"][s.name].get("mode", "not applied") for e in episodes]
        by_mode[s.name] = {
            m: counts([e for e, x in zip(episodes, modes, strict=True) if x == m]) for m in sorted(set(modes))
        }
    return dict(spec=",".join(s.spec() for s in env.scenarios), **counts(episodes), by_mode=by_mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pick_cube")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed_start", type=int, default=EVAL_SEED_START)
    ap.add_argument("--port", type=int, default=6200)
    ap.add_argument("--exec_horizon", type=int, default=8, help="actions executed per predicted chunk")
    ap.add_argument("--flow_steps", type=int, default=10)
    ap.add_argument(
        "--noise",
        choices=["fresh", "episode"],
        default="fresh",
        help="noise the sampler starts each chunk from: fresh every time, or the same for a whole episode",
    )
    ap.add_argument("--max_steps", type=int, default=None, help="episode length limit (default: the task's own)")
    ap.add_argument("--videos", type=int, default=5, help="number of rollouts to save as mp4")
    ap.add_argument(
        "--traces",
        type=int,
        default=0,
        help="number of rollouts to save as JSON (per control step: joint angles, gripper, gripper position, "
        "object poses), for replays and plots",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--shutdown_server", action="store_true")
    ap.add_argument("--expert", action="store_true", help="run the scripted expert instead of a served policy")
    ap.add_argument("--expect_run", default=None, help="stop unless the server serves a checkpoint of this run")
    ap.add_argument("--expect_number", type=int, default=None, help="... and this checkpoint number")
    ap.add_argument("--expect_weights", choices=["raw", "ema"], default=None, help="... and this kind of weights")
    # Interventions for finding out what a policy's choices depend on.
    ap.add_argument("--expert_prefix", type=int, default=0, help="the scripted expert drives the first K steps")
    ap.add_argument(
        "--prefix_target",
        choices=["named", "distractor"],
        default="named",
        help="which object the expert prefix heads for: the named one, or another object in the layout",
    )
    ap.add_argument("--stale_images", action="store_true", help="keep sending the first frame's images")
    ap.add_argument(
        "--first_prev_action",
        choices=["hold", "expert"],
        default="hold",
        help="previous action sent with the first query: the hold command the robot is really under, or the "
        "expert's first command (what a training window padded by copying contains)",
    )
    # Disturbances: how does the policy cope when the scene changes under it? (disturb.py)
    ap.add_argument(
        "--disturb",
        choices=["none", "reach", "grasp", "drift"],
        default="none",
        help="move the named object once per episode: 'reach' at --disturb_step, while the gripper is on its way; "
        "'grasp' at the moment the gripper first closes, so that succeeding takes a second attempt; 'drift': the "
        "object keeps sliding on a smooth random path at --drift_cm_s until it is held",
    )
    ap.add_argument("--disturb_step", type=int, default=15)
    ap.add_argument("--disturb_mm", type=float, default=80.0, help="how far the object is moved")
    ap.add_argument(
        "--drift_cm_s",
        type=float,
        default=2.0,
        help="with --disturb drift: how fast the named object slides (cm/s), from the first step until it is held",
    )
    ap.add_argument(
        "--expert_side",
        choices=["clearance", "preferred"],
        default="clearance",
        help="with --expert, stems: which side the expert approaches from (see ScriptedExpert)",
    )
    ap.add_argument(
        "--expert_reactive",
        action="store_true",
        help="with --expert: the expert plans again when the object has moved or a grasp closed on nothing",
    )
    ap.add_argument(
        "--random_start",
        action="store_true",
        help="start every episode from a random gripper pose above the table instead of the home pose",
    )
    # Reasoning tasks (see the module's docstring)
    ap.add_argument(
        "--notes",
        choices=["model", "expert", "swap", "reasoner"],
        default="model",
        help="a policy that writes notes: let it write them (model), hand it the correct ones (expert), notes about "
        "a shoot the instruction does not mean (swap), or the notes of the reasoner on --reasoner_port (reasoner: that "
        "reasoner thinks, the served policy acts)",
    )
    ap.add_argument(
        "--subgoals",
        choices=["none", "reasoner", "expert", "swap"],
        default="none",
        help="two-level: the served policy's instruction is a sub-goal, from the reasoner on --reasoner_port, from the "
        "correct notes, or about a shoot the instruction does not mean",
    )
    ap.add_argument(
        "--see",
        choices=["model", "expert"],
        default="model",
        help="who fills the `see` field of the notes (what is on the stand): the model that writes them, or the "
        "correct note (expert), after which the model writes the rest of each note",
    )
    ap.add_argument("--reasoner_port", type=int, default=None, help="port of a policy_server.py serving a reasoner")
    ap.add_argument(
        "--expect_reasoner_run", default=None, help="stop unless the reasoner serves a checkpoint of this run"
    )
    ap.add_argument(
        "--reason_check",
        action="store_true",
        help="the served policy is a reasoner: the expert drives and the server writes notes every chunk, compared "
        "with the correct ones",
    )
    ap.add_argument(
        "--scenario",
        default=None,
        help="test situations on top of the task (scenarios.py), comma-separated name[:mode][@p], e.g. 'wind' or "
        "'lighting:dusk,bump'",
    )
    ap.add_argument(
        "--leaf_fix",
        action="store_true",
        help="draw the plant's leaves where their collision boxes are (as in data from collect.py --leaf_fix)",
    )
    # Architecture v2 (see the module's docstring and slow_level.py)
    ap.add_argument(
        "--slow_level",
        choices=["none", "v2"],
        default="none",
        help="v2: the slow level of architecture v2 gives the served policy its sub-goals as its instruction",
    )
    ap.add_argument(
        "--scene",
        choices=["truth", "eyes"],
        default="truth",
        help="v2: what its scene memory takes, the simulator's truth or the percepts of the eyes on --slow_port",
    )
    ap.add_argument(
        "--program",
        choices=["gold", "writer"],
        default="gold",
        help="v2: the task's own program, or the one the language model on --slow_port writes for the instruction",
    )
    ap.add_argument("--slow_port", type=int, default=None, help="v2: port of a slow_server.py (eyes, writer)")
    ap.add_argument(
        "--slow_stable", type=int, default=None, help="v2: the memory's rows (default 1 for the truth, 3 for the eyes)"
    )
    ap.add_argument("--slow_p_min", type=float, default=0.7, help="v2: the confidence a percept needs to count")
    ap.add_argument(
        "--patience", type=int, default=160, help="v2: control steps without progress before a recovery (--recover)"
    )
    ap.add_argument(
        "--recover",
        type=int,
        default=0,
        help="v2: the most recoveries per episode (a scripted move out of the plant); after them the slow level gives "
        "up. 0: none, and no limit on slowness but the episode's own",
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = PickupEnv(args.task, max_steps=args.max_steps, scenario=args.scenario, leaf_fix=args.leaf_fix)
    if args.slow_level == "v2":  # checked before any server is reached
        assert env.plant_task, "--slow_level v2 reads a plant: it needs a plant task"
        assert not args.expert, "--slow_level v2 gives its sub-goals to a served policy: drop --expert"
        assert args.subgoals == "none", "--slow_level v2 is the slow level: it needs --subgoals none"
        assert args.notes != "reasoner" and not args.reason_check, "--slow_level v2 is the slow level: no reasoner"
        assert args.program == "writer" or env.reason, "--program gold is a reasoning task's own: use a reason_* task"
        assert (args.scene, args.program) == ("truth", "gold") or args.slow_port is not None, (
            "--scene eyes and --program writer are served by slow_server.py: give its --slow_port"
        )
        assert args.recover >= 0 and args.patience >= 1, "--recover must be 0 or more and --patience 1 or more"
    else:
        assert (args.scene, args.program, args.slow_port, args.recover) == ("truth", "gold", None, 0), (
            "--scene, --program, --slow_port and --recover are for --slow_level v2"
        )
    writer = None
    if env.reason:
        from .notes import NoteWriter

        writer = NoteWriter(env)
    reasoner = reasoner_info = None
    if "reasoner" in (args.subgoals, args.notes) or (args.expert and args.reasoner_port is not None):
        assert env.reason and args.reasoner_port is not None, "a reasoner needs a reasoning task and --reasoner_port"
        reasoner = connect(args.reasoner_port)
        reasoner.send({"cmd": "info"})
        reasoner_info = reasoner.recv()
        check_server(reasoner_info, args.reasoner_port, args.expect_reasoner_run, None, None)
        assert reasoner_info.get("reasons"), "the server on --reasoner_port does not write notes"
    assert args.subgoals == "none" or env.reason, "--subgoals needs a reasoning task"
    expert = conn = None
    if args.expert:
        from .expert import ScriptedExpert

        expert = ScriptedExpert(env, seed=0, reactive=args.expert_reactive, side=args.expert_side)
        server_info = dict(policy="scripted expert", reactive=args.expert_reactive, side=args.expert_side)
    else:
        conn = connect(args.port)
        conn.send({"cmd": "info"})
        server_info = conn.recv()
        check_server(server_info, args.port, args.expect_run, args.expect_number, args.expect_weights)
    # How far back the policy's images reach (t-1 and t unless its config takes older frames).
    offsets = image_offsets(server_info.get("image_names", [])) if conn is not None else [-1, 0]
    rta = conn is not None and bool(server_info.get("reasons"))  # the served policy writes notes itself
    if args.reason_check:
        assert rta and env.reason, "--reason_check needs a server that writes notes and a reasoning task"
        from .expert import ScriptedExpert

        reasoner, reasoner_info = conn, server_info
        expert = ScriptedExpert(env, seed=0, reactive=args.expert_reactive, side=args.expert_side)
    assert args.notes == "model" or rta, "--notes expert / swap need a policy that writes notes"
    assert args.see == "model" or (env.reason and (reasoner is not None or rta)), (
        "--see expert needs a reasoning task and something that writes notes (a reasoner or a policy that does)"
    )
    pilot = None
    if expert is None and (args.expert_prefix > 0 or args.first_prev_action == "expert"):
        from .expert import ScriptedExpert

        pilot = ScriptedExpert(env, seed=0)
    starter = None
    if args.random_start:
        from .expert import ScriptedExpert, random_start

        starter = ScriptedExpert(env, seed=0)  # for its inverse kinematics only
    slow = recovery = None  # --slow_level v2: the slow level, and a recovery in progress (slow_level.Recovery)
    if args.slow_level == "v2":
        from . import slow_level

        served = args.scene == "eyes" or args.program == "writer"
        slow = slow_level.SlowLevel(
            env,
            scene=args.scene,
            program=args.program,
            server=connect(args.slow_port) if served else None,
            stable=args.slow_stable,
            p_min=args.slow_p_min,
            patience=args.patience,
            max_recoveries=args.recover,
        )

    episodes, latencies = [], []
    per_target = defaultdict(lambda: [0, 0])
    t_start = time.time()
    for ep in range(args.episodes):
        seed = args.seed_start + ep
        obs, start_yaw = env.reset(seed=seed), 0.0
        if starter is not None:
            obs, start_yaw = random_start(env, starter, seed)
        history, queue, frames, trace = FrameHistory(obs, offsets), [], [], []
        if slow is not None:  # v2: the episode's program and an empty memory; its own frames, for eyes that look back
            slow.reset()
            slow_history, recovery = FrameHistory(obs, slow.offsets), None
        wrong_object = False  # a non-target object was lifted at any point in the episode
        caught = False  # the named object was lifted (stems: held) at some step, if only briefly
        probe = GraspProbe(env)
        disturbance = Disturbance(
            env, args.disturb, args.disturb_step, args.disturb_mm / 1000, seed=seed, speed=args.drift_cm_s / 100
        )
        if expert is not None:
            expert.reset(yaw=start_yaw)
        prefix_goal = None
        if pilot is not None:  # plan the prefix toward the chosen object; the instruction is unchanged
            named = env.target
            if args.prefix_target == "distractor":
                env.target = next(n for n in env.active if n != named)
            prefix_goal = env.target
            pilot.reset(yaw=start_yaw)
            env.target = named
        first_obs, first_query = obs, True
        done, info = False, {}
        # pruning efficiency: how far the plate turned, against how far the first shoot had to turn into reach
        turn = None
        if env.plant_task:
            # the shoot to cut first: the lowest in prune_one / prune_turn / prune, the named one in prune_named
            off0 = float(_wrap_pi(REACH_AZIMUTH - env.shoot_azimuth(env.target)))
            turn = dict(plate=env.plate_angle(), turned=0.0, offset0=off0, first_press=None)
        # reasoning tasks: the correct notes, the note the policy (or the reasoner) holds, a shoot to swap in, the
        # sub-goal handed to a two-level policy's fast level, the notes written and how they compare, the cuts in order
        held, subgoal, written, agreement, cut_order = "", None, [], [], []
        reasoned = ("", NOTE_SAME)  # --notes reasoner: the note the reasoner held and what it wrote at the last chunk
        swap = None
        if writer is not None:
            writer.reset()
            if "swap" in (args.notes, args.subgoals):
                swap = swap_shoot(env, seed)
        while not done:
            if writer is not None:
                writer.update()
            # --see expert: what the correct note says is on the stand, for whoever writes the notes
            see = reasoning.note_fields(writer.note).get("see", "") if args.see == "expert" else None
            fresh = not queue and expert is None and not (pilot is not None and env.steps < args.expert_prefix)
            if reasoner is not None and (fresh or (expert is not None and env.steps % args.exec_horizon == 0)):
                request = infer_request(history.at(-1), obs, env.last_action, env.instruction, 0, history=history)
                reasoner.send({**request, "cmd": "reason", "note_held": held, **({} if see is None else {"see": see})})
                reply = reasoner.recv()
                reasoned = (held, reply["note_now"])  # what the reasoner held and what it wrote (for --notes reasoner)
                if reply["note_now"] != NOTE_SAME:
                    written.append([env.steps, reply["note_now"]])
                held = reply["note"]
                agreement.append(note_agreement(held, writer.note))
                subgoal = reasoning.subgoal(held) or subgoal
            if slow is not None and fresh and recovery is None:  # v2: it looks where the policy needs a new chunk
                decision = slow.tick(obs, slow_history)
                if decision.recover:  # nothing has progressed for long: a scripted move first, then the policy
                    recovery = slow_level.Recovery(env)
                elif decision.subgoal is None:  # hold the command the robot is under for one step, then look again
                    queue = [env.last_action.copy()]
            if expert is not None:
                queue = [expert.act()]
            elif pilot is not None and env.steps < args.expert_prefix:
                queue = [pilot.act()]
            elif recovery is not None:  # v2: a recovery's scripted move drives until it is over
                queue = [recovery.act()]
                recovery = None if recovery.done else recovery
            elif not queue:
                prev_action = env.last_action
                if first_query and args.first_prev_action == "expert":
                    prev_action = pilot.act()
                first_query = False
                instruction, note_now, rta_held = env.instruction, None, held
                if args.subgoals in ("expert", "swap"):
                    subgoal = reasoning.subgoal(writer.note if swap is None else writer.text(targets=[swap]))
                if args.subgoals != "none":
                    instruction = subgoal or "stop"
                elif rta and args.notes == "reasoner":
                    rta_held, note_now = reasoned  # the policy acts on the reasoner's note
                elif rta and args.notes != "model":
                    correct = writer.note if args.notes == "expert" else writer.text(targets=[swap])
                    note_now = NOTE_SAME if correct == held else correct
                if slow is not None:  # v2: the slow level's sub-goal, from the look just taken
                    instruction = decision.subgoal
                conn.send(
                    infer_request(
                        history.at(-1),
                        obs,
                        prev_action,
                        instruction,
                        args.flow_steps,
                        noise_seed=seed if args.noise == "episode" else None,
                        images_from=(first_obs, first_obs) if args.stale_images else None,
                        history=history,
                        note_held=rta_held if rta else None,
                        note_now=note_now,
                        see=see if rta and note_now is None else None,
                    )
                )
                reply = conn.recv()
                latencies.append(reply["latency_s"])
                queue = list(reply["actions"][: args.exec_horizon])
                if rta:
                    if reply["note_now"] != NOTE_SAME and args.notes != "reasoner":  # a reasoner's are recorded above
                        written.append([env.steps, reply["note_now"]])
                    held = reply["note"]
                    if args.notes == "model":
                        agreement.append(note_agreement(held, writer.note))
            if ep < args.videos:
                frames.append(np.concatenate([obs["front"], obs["wrist"]], axis=1))
            action = queue.pop(0)
            if ep < args.traces:
                trace.append(trace_step(env, obs, action))
            probe.update(action)  # before the disturbance: the first close is judged on the scene it was aimed at
            disturbance.update(action)
            cut_before = set(env.cut) if env.plant_task else None
            obs, done, info = env.step(action)
            history.push(obs)
            if slow is not None:  # v2: its frames; and the episode ends once its verdict has stood (slow.done)
                slow_history.push(obs)
                done = done or slow.done
            if cut_before is not None:
                cut_order += sorted(set(env.cut) - cut_before)
            if turn is not None:
                angle = env.plate_angle()
                turn["turned"] += abs(angle - turn["plate"])
                turn["plate"] = angle
                if turn["first_press"] is None:
                    down = [b for b in BUTTONS if env.button_travel(b) > PRESS_TRAVEL]
                    turn["first_press"] = BUTTONS[down[0]][1] if down else None
            wrong_object = wrong_object or bool(info["wrong_object"])
            caught = caught or env.target in info["lifted"]
        episodes.append(
            dict(
                seed=seed,
                target=env.target,
                instruction=env.instruction,
                success=bool(info["success"]),
                steps=env.steps,
                wrong_object=wrong_object,
                damaged=info.get("damaged"),  # pruning: what was damaged, if anything
                # pruning: degrees the plate turned; how far the first shoot pointed from the robot at the start; the
                # turning direction of the first button press (+1 counter-clockwise, which brings a shoot with a
                # positive offset toward the robot the short way)
                turned_deg=round(float(np.degrees(turn["turned"])), 1) if turn else None,
                start_offset_deg=round(float(np.degrees(turn["offset0"])), 1) if turn else None,
                first_press=turn["first_press"] if turn else None,
                caught=caught,
                prefix_goal=prefix_goal,
                disturbed=disturbance.record,
                **probe.result(),
                # named shoots: which shoot was named, how (training phrasing or a kind of held-out one), and among how
                # many shoots; the shoots cut that were not named
                **(
                    dict(
                        named=env.target_label,
                        phrasing_kind=env.phrasing_kind,
                        shoots=len(env.active),
                        wrong_shoots=info.get("wrong_shoots", []),
                    )
                    if env.named
                    else {}
                ),
                # aim range: the drill, the colour named, and how close the gripper came (aim.AimEnv.aim_record)
                **(dict(aim=env.aim_record(info)) if getattr(env, "aim", None) else {}),
                # reasoning tasks: what was asked, what was cut in which order, the notes the policy wrote (control
                # step, text), how its held note compared with the correct one at every chunk, the shoot of a swapped
                # note
                **(
                    dict(
                        kind=env.reason_kind,
                        phrasing=env.phrasing_kind,
                        shoots=len(env.active),
                        targets_named=list(env.targets),
                        target_positions=list(env.reason_task.target_positions),
                        label=env.label_colour,
                        label_hidden_at_reach=bool(env.label_hidden_at_reach()) if env.label_colour else None,
                        wrong_shoots=info.get("wrong_shoots", []),
                        cut_order=cut_order,
                        notes_correct=[[int(t), n] for t, n in writer.written],
                        notes_written=written,
                        note_agreement=agreement,
                        swap_shoot=swap,
                    )
                    if env.reason
                    else {}
                ),
                # scenarios: what each drew this episode, the cuts in order, and the outcome details they add
                **(
                    dict(
                        scenario=env.scenario_info,
                        cut_order=cut_order,
                        **{
                            k: info[k]
                            for k in (
                                "out_of_order",
                                "home_m",
                                "wrong_shoots",
                                "distractors_touched",
                                "distractors_moved",
                            )
                            if k in info and not (k == "wrong_shoots" and (env.named or env.reason))  # (above)
                        },
                    )
                    if env.scenarios
                    else {}
                ),
                # architecture v2: what the slow level decided, and how its memory compared with the simulator
                **(dict(v2=slow.record(info)) if slow is not None else {}),
            )
        )
        key = env.target_label or env.target  # named shoots: by position (shoot names follow height, not position)
        per_target[key][0] += int(info["success"])
        per_target[key][1] += 1
        if ep < args.videos:
            write_video(out / f"rollout_{ep:03d}_{env.target}_{'ok' if info['success'] else 'fail'}.mp4", frames)
        if ep < args.traces:
            write_json(
                out / f"trace_{ep:03d}_{env.target}_{'ok' if info['success'] else 'fail'}.json",
                dict(
                    seed=seed,
                    target=env.target,
                    instruction=env.instruction,
                    objects=list(env.active),
                    control_hz=CONTROL_HZ,
                    success=bool(info["success"]),
                    close_step=episodes[-1]["close_step"],
                    disturbed=disturbance.record,
                    steps=trace,
                ),
            )
        done_n = len(episodes)
        print(
            f"[eval] {done_n}/{args.episodes} seed={seed} target={env.target} success={info['success']} "
            f"steps={env.steps}  running={sum(e['success'] for e in episodes) / done_n:.2f}",
            flush=True,
        )

    n = len(episodes)
    k = sum(e["success"] for e in episodes)
    p = k / n
    z = 1.96  # Wilson 95% interval
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    results = dict(
        task=args.task,
        episodes=n,
        successes=k,
        success_rate=p,
        success_ci95=[max(0.0, centre - half), min(1.0, centre + half)],
        mean_steps_success=float(np.mean([e["steps"] for e in episodes if e["success"]])) if k else None,
        wrong_object_rate=sum(e["wrong_object"] for e in episodes) / n,
        turning=_turning_summary(episodes),
        caught_rate=sum(e["caught"] for e in episodes) / n,
        per_target={t: dict(successes=s, episodes=m, success_rate=s / m) for t, (s, m) in sorted(per_target.items())},
        grasp=summarize(episodes),
        inference_latency_ms=dict(
            mean=1000 * float(np.mean(latencies)),
            p50=1000 * float(np.percentile(latencies, 50)),
            p95=1000 * float(np.percentile(latencies, 95)),
        )
        if latencies
        else None,
        exec_horizon=args.exec_horizon,
        flow_steps=args.flow_steps,
        noise=args.noise,
        random_start=args.random_start,
        interventions=dict(
            expert_prefix=args.expert_prefix,
            prefix_target=args.prefix_target if args.expert_prefix else None,
            stale_images=args.stale_images,
            first_prev_action=args.first_prev_action,
            disturb=args.disturb,
            disturb_step=args.disturb_step if args.disturb == "reach" else None,
            disturb_mm=args.disturb_mm if args.disturb in ("reach", "grasp") else None,
            drift_cm_s=args.drift_cm_s if args.disturb == "drift" else None,
            disturbed_rate=sum(bool(e["disturbed"] and e["disturbed"]["to_xy"]) for e in episodes) / n,
        ),
        seed_start=args.seed_start,
        wall_time_s=time.time() - t_start,
        server=server_info,
        **(dict(named=_named_summary(episodes)) if env.named else {}),
        **(dict(scenario=_scenario_summary(env, episodes)) if env.scenarios else {}),
        **(dict(leaf_fix=True) if args.leaf_fix else {}),
        **(dict(aim=_aim_summary(episodes)) if getattr(env, "aim", None) else {}),
        **(
            dict(
                reasoning=_reason_summary(episodes),
                notes=args.notes if rta else None,
                subgoals=args.subgoals,
                see=args.see,
                reasoner=reasoner_info,
            )
            if env.reason
            else {}
        ),
        **(dict(v2=dict(settings=slow.settings(), **slow_level.summary(episodes))) if slow is not None else {}),
        per_episode=episodes,
    )
    write_json(out / "results.json", results, indent=2)
    g = results["grasp"]
    print(
        f"[eval] {args.task}: {k}/{n} = {p:.1%} "
        f"(95% CI {results['success_ci95'][0]:.1%}-{results['success_ci95'][1]:.1%}) -> {out / 'results.json'}"
    )
    if g["median_xy_err_target_mm"] is not None:
        print(
            f"[eval] gripper closed in {g['closed_gripper_rate']:.0%} of episodes, "
            f"nearest object was the named one in {g['closed_nearest_is_target_rate']:.0%}; "
            f"median miss {g['median_xy_err_target_mm']:.0f} mm to the named object, "
            f"{g['median_xy_err_nearest_mm']:.0f} mm to the nearest object"
        )
    if reasoner is not None and reasoner is not conn:
        if args.shutdown_server:
            reasoner.send({"cmd": "shutdown"})
            reasoner.recv()
        reasoner.close()
    if conn is not None:
        if args.shutdown_server:
            conn.send({"cmd": "shutdown"})
            conn.recv()
        conn.close()
    if slow is not None:
        slow.close()
        if slow.server is not None:
            if args.shutdown_server:
                slow.server.send({"cmd": "shutdown"})
                slow.server.recv()
            slow.server.close()
    env.close()


if __name__ == "__main__":
    main()
