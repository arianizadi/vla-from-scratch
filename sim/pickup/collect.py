"""Collect scripted-expert demonstrations and write them as a LeRobot v2.0 dataset.

The layout (meta/info.json, meta/episodes.jsonl, meta/tasks.jsonl, data/chunk-000/episode_*.parquet
with inline JPEG image columns) is what vla_foundry's `--type lerobot` preprocessing converter reads.

Per-frame columns: front, wrist (JPEG bytes), state [8], action [8], plus LeRobot bookkeeping.
action[t] is the command applied after observing frame t.

  python -m sim.pickup.collect --task pick_cube --episodes 200 --out data/lerobot/pick_cube

Reasoning tasks (reason_*) also get the expert's notes (notes.py) as per-frame columns: note (the note in force),
note_prev (the one before it), note_age (frames since it was written) and instruction_subgoal (its "next" field, which
preprocessing offers as the language instruction type "subgoal"). A situation on a plant task that changes the task
(--scenario ties, sick, buds, order, nothing_to_do) gets the column instruction_subgoal alone, from the shoot the expert
goes for (notes.SubgoalWriter), so that the fast level of architecture v3 can be trained on it.

With --policy_ports, a trained policy (served by policy_server.py, one server per worker) drives the first K steps
of every episode and the scripted expert takes over from wherever the arm is and finishes the task; only the
expert's part is recorded (scripts/make_takeover_dataset.sh starts the servers).

Architecture v3's retraining on the fast level's own rollouts (tutorial/architecture-v3.md, step 7): with
--takeover_subgoals the served policy is a sub-goal fast level and is given the sub-goal of the scene now (the column
notes.SubgoalWriter writes; --subgoal_column writes it on a plain plant task too), and the handover comes at the first
place the fast level drifts (Drift) or at K, whichever is first: --takeover_overturn DEG when the plate has carried the
shoot named past its reach and on by DEG while still turning (the expert turns it back), --takeover_cut P, with
probability P, when the open jaws come within CUT_NEAR of the shoot named (the expert finishes that cut from where the
hand is, without backing out: ScriptedExpert.finish_cut). --clear_start starts every episode clear of the plant, as v3's
evaluations do.
"""

import argparse
import io
import json
import multiprocessing as mp
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from . import reasoning
from .disturb import Disturbance
from .env import CONTROL_HZ, CUT_MID, REACH_AZIMUTH, TASKS, PickupEnv
from .eval import connect, infer_request
from .expert import ScriptedExpert, clear_start, random_start
from .notes import NoteWriter, SubgoalWriter, note_columns

CAMERAS = ("front", "wrist")
NOTE_COLUMNS = ("note", "note_prev", "note_age", "instruction_subgoal")


def _jpeg(img, quality=95):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


CUT_NEAR = 0.035  # metres from the middle of a shoot's cut zone at which an approach counts as one (Drift)
OVERTURN_NEAR = np.radians(10)  # a shoot that came this near its reach direction has been turned to


def _wrap(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


class Drift:
    """Where a sub-goal fast level drifts, watched at every control step of a takeover (collect.py --takeover_cut,
    --takeover_overturn). `subgoal()` gives the sub-goal in force ("cut the middle shoot"); `check(obs)` returns None or
    (reason, shoot):

      "overturn"  the shoot named came within OVERTURN_NEAR of its reach direction (REACH_AZIMUTH) and the plate has
                  carried it on, past it by more than `overturn` radians, still turning away from it
      "cut"       the open jaws (more than 0.3) came within CUT_NEAR of the middle of the shoot's cut zone; decided once
                  per shoot, with probability `cut_prob` (from the episode's seed)

    The reason and shoot of the handover are kept (`reason`, `shoot`)."""

    def __init__(self, env, subgoal, seed: int, cut_prob: float = 0.0, overturn: float = 0.0):
        self.env, self.subgoal = env, subgoal
        self.rng = np.random.default_rng([seed, 23])
        self.cut_prob, self.overturn = cut_prob, overturn
        self.decided: set[str] = set()  # shoots whose approach was decided
        self.closest: dict[str, float] = {}  # shoot -> its least turn from the reach direction so far
        self.reason: str | None = None
        self.shoot: str | None = None

    def target(self) -> str | None:
        goal = self.subgoal()
        env = self.env
        for i, name in enumerate(env.active):
            if goal == f"cut the {reasoning.position_name(i, len(env.active))} shoot":
                return name
        return None

    def check(self, obs) -> tuple[str, str] | None:
        env = self.env
        name = self.target()
        if name is None or name in env.cut:
            return None
        if self.overturn > 0:
            turn = float(_wrap(REACH_AZIMUTH - env.shoot_azimuth(name)))
            speed = env.plate_speed()
            came = self.closest.get(name, np.inf) < OVERTURN_NEAR
            self.closest[name] = min(self.closest.get(name, np.inf), abs(turn))
            if came and abs(turn) > self.overturn and turn * speed < 0 and abs(speed) > 0.1:
                return self._found("overturn", name)
        if self.cut_prob > 0 and name not in self.decided:
            a, d = env.shoot_axis(name)
            if np.linalg.norm(env.tcp_pos() - (a + d * 0.035)) < CUT_NEAR and float(obs["state"][7]) > 0.3:
                self.decided.add(name)
                if self.rng.random() < self.cut_prob:
                    return self._found("cut", name)
        return None

    def _found(self, reason, name):
        self.reason, self.shoot = reason, name
        return reason, name


def hand_over(env, expert, obs, policy, steps, budget, drift: Drift | None = None):
    """Let `policy` drive the first `steps` control steps of an episode, then hand the arm to `expert` where it is
    (ScriptedExpert.take_over), with `budget` more steps to finish in, and let the expert make the moves that only a
    takeover needs (its preamble: backing out of the plant, a move back onto the lane). `policy(prev_obs, obs)`
    returns the next actions to execute. Returns (observation where the expert's ordinary plan begins, None,
    preamble steps), or (None, info, None) if the episode ended before the handover, (None, info, preamble steps)
    if it ended during the preamble. A demonstration starts at the observation returned: the preamble's frames are
    not recorded, since the expert's own demonstrations act differently in such states.

    `drift` (a Drift): the handover comes earlier, at the first control step at which it finds the fast level drifting
    (its `reason` and `shoot` say why); for "cut" the expert finishes that cut from where the hand is
    (ScriptedExpert.finish_cut), with no preamble."""
    prev_obs, queue = obs, []
    for _ in range(steps):
        if drift is not None and drift.check(obs) is not None:
            break
        if not queue:
            queue = list(policy(prev_obs, obs))
        prev_obs = obs
        obs, done, info = env.step(queue.pop(0))
        if done:
            return None, info, None
    env.max_steps = env.steps + budget
    if drift is not None and drift.reason == "cut":
        expert.finish_cut(drift.shoot)
    else:
        expert.take_over()
    preamble = 0
    while expert.preamble:
        obs, done, info = env.step(expert.act())
        preamble += 1
        if done:
            return None, info, preamble
    return obs, None, preamble


def served_policy(conn, env, exec_horizon=8, flow_steps=10, instruction=None):
    """The policy behind a policy_server.py connection, queried as eval.py does (fresh noise for every chunk).
    `instruction()`: the text it is given (a sub-goal fast level's sub-goal); env.instruction by default."""

    def policy(prev_obs, obs):
        said = env.instruction if instruction is None else instruction()
        conn.send(infer_request(prev_obs, obs, env.last_action, said, flow_steps))
        return conn.recv()["actions"][:exec_horizon]

    return policy


def _worker(args):
    (
        task,
        jobs,
        out_dir,
        worker_id,
        disturb_prob,
        anywhere,
        hold_steps,
        drift_prob,
        drift_cm_s,
        side,
        tail_steps,
        scenario,
        leaf_fix,
        park_reset,
        action_noise,
        subgoal_column,
        clear,
        takeover,
    ) = args  # takeover: None, or this worker's policy server and the range the handover step K is drawn from
    env = PickupEnv(task, scenario=scenario, leaf_fix=leaf_fix, park_reset=park_reset)
    expert = ScriptedExpert(env, seed=worker_id, reactive=disturb_prob > 0 or drift_prob > 0, side=side)
    writer = NoteWriter(env) if env.reason else None
    if writer is None and env.plant_task and (subgoal_column or any(s.changes_task for s in env.scenarios)):
        writer = SubgoalWriter(env)  # a situation (or --subgoal_column): the sub-goal column alone
    budget = env.max_steps
    conn = policy = None
    if takeover is not None:
        conn = connect(takeover["port"])

        def subgoal():  # the sub-goal in force now, for a sub-goal fast level (--takeover_subgoals)
            writer.update()
            return writer.note

        said = subgoal if takeover.get("subgoals") else None
        assert said is None or writer is not None, "--takeover_subgoals needs a sub-goal column (--subgoal_column)"
        policy = served_policy(conn, env, takeover["exec_horizon"], takeover["flow_steps"], said)
    results = []
    for seed, target_index in jobs:
        env.max_steps = budget
        obs, start_yaw = env.reset(seed=seed, target_index=target_index), 0.0
        if clear:
            obs, start_yaw = clear_start(env, expert, seed)
        elif anywhere:
            obs, start_yaw = random_start(env, expert, seed)
        handover = {}
        if takeover is None:
            expert.reset(yaw=start_yaw)
        else:
            lo, hi = takeover["steps"]
            k = int(np.random.default_rng([seed, 17]).integers(lo, hi + 1))  # its own stream, as random_start's
            drift = None
            if takeover.get("cut", 0) > 0 or takeover.get("overturn", 0) > 0:
                writer.reset()
                drift = Drift(env, subgoal, seed, takeover.get("cut", 0.0), np.radians(takeover.get("overturn", 0.0)))
            obs, ended, preamble = hand_over(env, expert, obs, policy, k, budget, drift)
            reason = {} if drift is None else dict(handover_reason=drift.reason or "step")
            if obs is None:  # ended before the handover (pruned everything, damaged the plant) or in the preamble
                key = "before_handover" if preamble is None else "in_preamble"
                results.append(dict(seed=seed, success=False, takeover_step=k, **reason, **{key: _ending(ended)}))
                continue
            handover = dict(takeover_step=k, cut_at_handover=len(env.cut), preamble_steps=preamble, **reason)
            handover["handover_at"] = int(env.steps)
        # Per-episode draw (from the seed, so a dataset can be rebuilt): is this episode disturbed, when
        # and by how much. Half of the moves come during the reach, half as the gripper closes.
        rng = np.random.default_rng([seed, 7])
        kind = str(rng.choice(["reach", "grasp"])) if rng.random() < disturb_prob else "none"
        step, distance = int(rng.integers(5, 30)), float(rng.uniform(0.05, 0.10))
        speed = 0.0
        if drift_prob > 0:  # drawn after the others, so that datasets made without drift are unchanged
            drifts, speed = rng.random() < drift_prob, float(rng.uniform(0.5, 1.5)) * drift_cm_s / 100
            if kind == "none" and drifts:
                kind = "drift"
        disturbance = Disturbance(env, kind, step=step, distance=distance, seed=seed, speed=speed)
        noise = ActionNoise(seed, *action_noise)
        frames = {c: [] for c in CAMERAS}
        states, actions, step_notes = [], [], []
        if writer is not None:
            writer.reset()  # after a takeover too: the column starts from the scene at the handover
        done, info = False, {}
        while not done:
            if writer is not None:  # the note in force at this frame, written before the expert acts on it
                writer.update()
                step_notes.append(writer.note)
            # The first hold_steps frames record the robot at rest under its hold command.
            action = env.last_action.copy() if len(actions) < hold_steps else expert.act()
            for c in CAMERAS:
                frames[c].append(_jpeg(obs[c]))
            states.append(obs["state"])
            actions.append(action)  # the expert's own command is the label, also when another one is executed
            disturbance.update(action)
            obs, done, info = env.step(noise.apply(action, env) if len(actions) > hold_steps else action)
        if not info["success"]:
            results.append(dict(seed=seed, success=False, **handover, after_handover=_ending(info)))
            continue
        for _ in range(tail_steps):  # after the success: the robot holds its last command (for a cut, jaws shut)
            if writer is not None:
                writer.update()
                step_notes.append(writer.note)
            action = env.last_action.copy()
            for c in CAMERAS:
                frames[c].append(_jpeg(obs[c]))
            states.append(obs["state"])
            actions.append(action)
            obs, _, _ = env.step(action)
        tmp = Path(out_dir) / "tmp" / f"seed_{seed:07d}_{target_index or 0}.parquet"
        table = pa.table(
            {
                **{c: pa.array([{"bytes": b, "path": None} for b in frames[c]]) for c in CAMERAS},
                "state": pa.array([s.tolist() for s in states], type=pa.list_(pa.float32())),
                "action": pa.array([a.tolist() for a in actions], type=pa.list_(pa.float32())),
                **(_note_columns(writer, step_notes) if writer is not None else {}),
            }
        )
        pq.write_table(table, tmp)
        results.append(
            dict(
                seed=seed,
                target_index=target_index or 0,
                success=True,
                length=len(actions),
                instruction=env.instruction,
                path=str(tmp),
                disturbed=kind if disturbance.record and disturbance.record["to_xy"] else "none",
                replans=expert.replans,
                tracks=expert.tracks,
                **handover,
                **({"named": env.target_label} if env.named else {}),  # named shoots: lowest / middle / highest
                **({"scenario": env.scenario_info} if env.scenarios else {}),  # what each scenario drew
                **(_reason_record(env, writer) if env.reason else {}),
                **(
                    {"subgoals": [[int(t), text] for t, text in writer.written]}
                    if writer is not None and not env.reason
                    else {}
                ),
            )
        )
    if conn is not None:
        conn.close()
    env.close()
    return results


class ActionNoise:
    """DART-style noise on the commands the expert executes (--action_noise): the arm's seven joint targets get a
    smooth random offset (an AR(1) process with correlation CORR from one control step to the next and a stationary
    standard deviation of `sigma` radians); the gripper command is left alone. The arm is pushed off the expert's path
    and the recorded label stays the expert's own command, so the demonstrations show the way back. Drawn from the
    episode's seed, so a dataset can be rebuilt."""

    CORR = 0.9
    ARM = 7

    def __init__(self, seed: int, sigma: float, quiet: float = 0.0):
        self.sigma, self.rng, self.offset = sigma, np.random.default_rng([seed, 13]), np.zeros(self.ARM)
        self.quiet = quiet

    def calm(self, env, action) -> bool:
        """--action_noise_quiet_cm: no fresh noise (the offset dies away, by CORR a step) while the jaws are commanded
        shut or the gripper's centre is within `quiet` metres of the middle of an uncut shoot's cut zone, so that the
        expert still cuts where it aims and the frames before show the arm brought back onto its line of approach."""
        if self.quiet <= 0 or not env.plant_task:
            return False
        if action[7] < 0.5:
            return True
        tcp = env.tcp_pos()
        for name in env.active:
            if name not in env.cut:
                a, d = env.shoot_axis(name)
                if np.linalg.norm(tcp - (a + d * CUT_MID)) < self.quiet:
                    return True
        return False

    def apply(self, action: np.ndarray, env=None) -> np.ndarray:
        if self.sigma <= 0:
            return action
        fresh = self.rng.normal(0.0, self.sigma * np.sqrt(1 - self.CORR**2), self.ARM)
        if env is not None and self.calm(env, action):
            fresh = np.zeros(self.ARM)  # drawn all the same, so the stream does not depend on where the arm is
        self.offset = self.CORR * self.offset + fresh
        executed = np.array(action, dtype=np.float64, copy=True)
        executed[: self.ARM] += self.offset
        return executed.astype(np.asarray(action).dtype)


def _note_columns(writer, step_notes):
    """The per-frame columns of an episode's notes (a reasoning task), or of its sub-goals alone (a situation)."""
    if isinstance(writer, SubgoalWriter):
        return {"instruction_subgoal": pa.array(step_notes)}
    return {k: pa.array(v) for k, v in note_columns(step_notes).items()}


def _reason_record(env, writer):
    """Reasoning tasks: what an episode asked for and the notes the expert wrote (control step, text)."""
    return dict(
        kind=env.reason_kind,
        phrasing=env.phrasing_kind,
        shoots=len(env.active),
        targets=list(env.reason_task.target_positions),
        label=env.label_colour,
        label_hidden_at_reach=bool(env.label_hidden_at_reach()) if env.label_colour else None,
        notes=[[int(t), text] for t, text in writer.written],
    )


def _ending(info):
    """How an episode ended: 'success', what was damaged, or 'time' (or 'wrong object' for the pick tasks)."""
    if info["success"]:
        return "success"
    return info.get("damaged") or ("wrong object" if info.get("wrong_object") else "time")


def _included(path, task):
    """The episodes of an earlier dataset written by this script, as entries to copy into the new one."""
    src = Path(path)
    info = json.loads((src / "meta" / "info.json").read_text())
    assert info["sim_task"] == task, f"{src} holds {info['sim_task']} demonstrations, not {task}"
    entries = [json.loads(line) for line in (src / "meta" / "episodes.jsonl").read_text().splitlines()]
    return [
        dict(
            e,
            instruction=e["tasks"][0],
            path=str(src / "data" / "chunk-000" / f"episode_{e['episode_index']:06d}.parquet"),
            source=src.name,
        )
        for e in entries
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="pick_cube")
    ap.add_argument("--episodes", type=int, default=200, help="number of successful episodes to keep")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0, help="first environment seed (training seeds are < 1_000_000)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--targets_per_layout",
        type=int,
        default=1,
        help="demonstrate this many different targets in every layout (pick_target: up to 3). With 1, each "
        "layout appears with a single instruction, and nothing in the data shows that the instruction matters.",
    )
    ap.add_argument(
        "--disturb_prob",
        type=float,
        default=0.0,
        help="share of episodes in which the named object is moved once, during the reach or as the gripper "
        "closes, and the expert recovers (plans again, retries the grasp). 0: every demonstration goes to plan, "
        "so the data holds no example of a correction.",
    )
    ap.add_argument(
        "--drift_prob",
        type=float,
        default=0.0,
        help="share of the episodes not otherwise disturbed in which the named object keeps sliding on a smooth "
        "random path until it is held, and the expert follows it",
    )
    ap.add_argument(
        "--expert_side",
        choices=["clearance", "preferred"],
        default="clearance",
        help="stems: which side the expert approaches from (see ScriptedExpert)",
    )
    ap.add_argument(
        "--drift_cm_s",
        type=float,
        default=2.0,
        help="typical drift speed in cm/s; each drifting episode draws its speed between 0.5 and 1.5 times this",
    )
    ap.add_argument(
        "--random_start",
        action="store_true",
        help="start every episode from a random gripper pose above the table instead of the home pose. From "
        "one fixed start, the arm's own position gives the goal away after a few steps of any reach.",
    )
    ap.add_argument(
        "--hold_steps",
        type=int,
        default=0,
        help="begin every episode with this many frames of the robot holding still. A policy that is given its "
        "previous action is served the hold command before its first step; without such frames, the first "
        "training window fills that slot with a copy of the expert's first command, which already points at the "
        "target.",
    )
    ap.add_argument(
        "--tail_steps",
        type=int,
        default=0,
        help="after each success, record this many more frames of the robot holding its last command. A cut "
        "succeeds 3 steps after the jaws reach the stem, so without a tail the shut gripper appears in only the "
        "last few frames of each demonstration, and a policy trained on them opens again before the jaws arrive.",
    )
    ap.add_argument(
        "--policy_ports",
        type=int,
        nargs="+",
        default=None,
        help="takeover demonstrations: one policy_server.py port per worker. A trained policy drives the first K "
        "steps of each episode, then the expert takes over from wherever the arm is and finishes the task. Only "
        "the expert's part is recorded, and only when it succeeds: demonstrations that start from states the policy "
        "reaches, which demonstrations from the expert's own starts rarely visit (DAgger-style data).",
    )
    ap.add_argument(
        "--takeover_steps",
        type=int,
        nargs=2,
        default=(0, 700),
        metavar=("LO", "HI"),
        help="with --policy_ports: K is drawn uniformly from LO..HI (inclusive) for each episode. The expert then "
        "has the task's whole step limit from the handover on.",
    )
    ap.add_argument(
        "--takeover_subgoals",
        action="store_true",
        help="with --policy_ports: the policy is a sub-goal fast level, given the sub-goal column's text (the "
        "situation's, or --subgoal_column's) as its instruction",
    )
    ap.add_argument(
        "--takeover_cut",
        type=float,
        default=0.0,
        help="with --takeover_subgoals: hand over, with this probability per shoot, when the open jaws come near the "
        "shoot named; the expert finishes the cut from there (Drift, ScriptedExpert.finish_cut)",
    )
    ap.add_argument(
        "--takeover_overturn",
        type=float,
        default=0.0,
        help="with --takeover_subgoals: hand over when the plate has carried the shoot named past its reach by this "
        "many degrees, still turning away (Drift); 0: off",
    )
    ap.add_argument(
        "--subgoal_column",
        action="store_true",
        help="write the sub-goal column (notes.SubgoalWriter) on a plant task without a situation too",
    )
    ap.add_argument(
        "--clear_start",
        action="store_true",
        help="start every episode from random_start's pose kept clear of the plant (expert.clear_start), as "
        "architecture v3's evaluations do",
    )
    ap.add_argument("--exec_horizon", type=int, default=8, help="with --policy_ports: actions executed per chunk")
    ap.add_argument("--flow_steps", type=int, default=10, help="with --policy_ports: sampler steps per chunk")
    ap.add_argument(
        "--scenario",
        default=None,
        help="test situations on top of the task (scenarios.py), comma-separated name[:mode][@p]; each episode "
        "records what they drew (meta/episodes.jsonl)",
    )
    ap.add_argument(
        "--leaf_fix",
        action="store_true",
        help="draw the plant's leaves where their collision boxes are (PickupEnv(leaf_fix=True)); policies trained on "
        "such data should be evaluated with eval.py --leaf_fix",
    )
    ap.add_argument(
        "--action_noise",
        type=float,
        default=0.0,
        help="DART-style demonstrations: the arm's executed joint targets get smooth random offsets of this standard "
        "deviation (radians; ActionNoise) while the expert's own commands are recorded as the actions, so the data "
        "shows recoveries from states a policy's own small errors lead to. 0: none",
    )
    ap.add_argument(
        "--action_noise_quiet_cm",
        type=float,
        default=0.0,
        help="with --action_noise, pruning: no fresh noise while the jaws are commanded shut or the gripper's centre "
        "is within this many cm of the middle of an uncut shoot's cut zone (ActionNoise.calm). 0: noise everywhere",
    )
    ap.add_argument(
        "--park_reset",
        action="store_true",
        help="put an unused side shoot back as built at every reset (PickupEnv(park_reset=True)), so that an episode "
        "does not depend on the ones a worker ran before it; for every new dataset",
    )
    ap.add_argument(
        "--include",
        nargs="+",
        default=[],
        help="datasets written by this script whose episodes come first in the new one, unchanged: new "
        "demonstrations and earlier ones preprocessed together, so that one set of normalisation statistics "
        "covers both",
    )
    args = ap.parse_args()
    assert TASKS[args.task].get("named") != "heldout", f"{args.task}: held-out phrasings are for evaluation only"
    assert TASKS[args.task].get("reason", ("", ""))[1] != "heldout", (
        f"{args.task}: held-out tasks are for evaluation only"
    )
    assert "order_heldout" not in (args.scenario or ""), "order_heldout is for evaluation only"
    num_objects = TASKS[args.task]["num_objects"]
    assert 1 <= args.targets_per_layout <= num_objects, f"{args.task} has {num_objects} objects per layout"
    per_layout = args.targets_per_layout
    if args.policy_ports is not None:
        assert len(args.policy_ports) == args.workers, "--policy_ports needs one port per worker"
    assert args.takeover_subgoals or not (args.takeover_cut or args.takeover_overturn), (
        "--takeover_cut and --takeover_overturn are for --takeover_subgoals"
    )
    assert not (args.random_start and args.clear_start), "--random_start or --clear_start, not both"
    included = [e for path in args.include for e in _included(path, args.task)]

    out = Path(args.out)
    (out / "tmp").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    servers = None
    if args.policy_ports is not None:  # what is being served, for the record (each server takes one client at a time)
        servers = set()
        for port in args.policy_ports:
            conn = connect(port)
            conn.send({"cmd": "info"})
            servers.add(conn.recv()["checkpoint"])
            conn.close()
        assert len(servers) == 1, f"the policy servers serve different checkpoints: {servers}"
    kept, attempted, successes, next_seed, endings = [], 0, 0, args.seed, []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        while len(kept) < args.episodes:
            need = args.episodes - len(kept)
            seeds = list(range(next_seed, next_seed + max(-(-(need + need // 10) // per_layout), args.workers)))
            next_seed = seeds[-1] + 1
            # With one target per layout the seed's own choice is kept (target_index None).
            jobs = [(seed, i if per_layout > 1 else None) for seed in seeds for i in range(per_layout)]
            extra = (
                args.disturb_prob,
                args.random_start,
                args.hold_steps,
                args.drift_prob,
                args.drift_cm_s,
                args.expert_side,
                args.tail_steps,
                args.scenario,
                args.leaf_fix,
                args.park_reset,
                (args.action_noise, args.action_noise_quiet_cm / 100),
                args.subgoal_column,
                args.clear_start,
            )
            chunks = [
                (args.task, jobs[i :: args.workers], str(out), i, *extra, _takeover(args, i))
                for i in range(args.workers)
            ]
            for res in pool.map(_worker, chunks):
                attempted += len(res)
                successes += sum(r["success"] for r in res)
                kept.extend(r for r in res if r["success"])
                endings.extend(r for r in res if not r["success"])
    kept = sorted(kept, key=lambda r: (r["seed"], r["target_index"]))[: args.episodes]
    new = len(kept)
    kept = included + kept

    tasks, episodes, index = {}, [], 0
    for ep_idx, r in enumerate(kept):
        task_idx = tasks.setdefault(r["instruction"], len(tasks))
        columns = [*CAMERAS, "state", "action"]  # without an included one's bookkeeping
        names = pq.read_schema(r["path"]).names
        if "note" in names:
            columns += NOTE_COLUMNS
        elif "instruction_subgoal" in names:  # a situation's sub-goals (notes.SubgoalWriter)
            columns.append("instruction_subgoal")
        table = pq.read_table(r["path"], columns=columns)
        n = table.num_rows
        table = table.append_column("timestamp", pa.array(np.arange(n, dtype=np.float32) / CONTROL_HZ))
        table = table.append_column("frame_index", pa.array(np.arange(n, dtype=np.int64)))
        table = table.append_column("episode_index", pa.array(np.full(n, ep_idx, dtype=np.int64)))
        table = table.append_column("index", pa.array(np.arange(index, index + n, dtype=np.int64)))
        table = table.append_column("task_index", pa.array(np.full(n, task_idx, dtype=np.int64)))
        pq.write_table(table, out / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet")
        episodes.append(
            dict(
                episode_index=ep_idx,
                tasks=[r["instruction"]],
                length=n,
                seed=r["seed"],
                disturbed=r["disturbed"],
                replans=r["replans"],
                tracks=r["tracks"],
                **{
                    k: r[k]
                    for k in (
                        "takeover_step",
                        "cut_at_handover",
                        "preamble_steps",
                        "handover_reason",
                        "handover_at",
                        "named",
                        "scenario",
                    )
                    if k in r
                },
                **{
                    k: r[k]
                    for k in (
                        "kind",
                        "phrasing",
                        "shoots",
                        "targets",
                        "label",
                        "label_hidden_at_reach",
                        "notes",
                        "subgoals",
                    )
                    if k in r
                },
                **({"source": r.get("source", out.name)} if args.include else {}),
            )
        )
        index += n
    for f in (out / "tmp").glob("*.parquet"):
        f.unlink()
    (out / "tmp").rmdir()

    with open(out / "meta" / "episodes.jsonl", "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in episodes)
    with open(out / "meta" / "tasks.jsonl", "w") as f:
        f.writelines(json.dumps(dict(task_index=i, task=t)) + "\n" for t, i in tasks.items())
    info = dict(
        codebase_version="v2.0",
        robot_type="franka_panda_mujoco",
        sim_task=args.task,
        fps=CONTROL_HZ,
        total_episodes=len(episodes),
        total_frames=index,
        total_tasks=len(tasks),
        expert_success_rate=successes / attempted if attempted else None,
        expert_attempted=attempted,
        targets_per_layout=per_layout,
        disturb_prob=args.disturb_prob,
        random_start=args.random_start,
        **({"clear_start": True} if args.clear_start else {}),
        **({"subgoal_column": True} if args.subgoal_column else {}),
        hold_steps=args.hold_steps,
        disturbed_episodes=sum(r["disturbed"] != "none" for r in kept[len(included) :]),
        drift_prob=args.drift_prob,
        drift_cm_s=args.drift_cm_s if args.drift_prob > 0 else None,
        drifting_episodes=sum(r["disturbed"] == "drift" for r in kept[len(included) :]),
        expert_side=args.expert_side,
        tail_steps=args.tail_steps,
        **({"scenario": args.scenario} if args.scenario else {}),
        **({"leaf_fix": True} if args.leaf_fix else {}),
        **({"park_reset": True} if args.park_reset else {}),
        **(
            {
                "action_noise": args.action_noise,
                "action_noise_corr": ActionNoise.CORR,
                "action_noise_quiet_cm": args.action_noise_quiet_cm,
            }
            if args.action_noise
            else {}
        ),
        takeover=_takeover_summary(args, servers, attempted, kept[len(included) :], endings),
        included=[
            dict(dataset=Path(p).name, episodes=sum(e["source"] == Path(p).name for e in included))
            for p in args.include
        ],
        new_episodes=new,
        data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        features={
            "front": dict(dtype="image", shape=[224, 224, 3]),
            "wrist": dict(dtype="image", shape=[224, 224, 3]),
            "state": dict(dtype="float32", shape=[8], names=["q1", "q2", "q3", "q4", "q5", "q6", "q7", "gripper_open"]),
            "action": dict(
                dtype="float32", shape=[8], names=["q1", "q2", "q3", "q4", "q5", "q6", "q7", "gripper_open"]
            ),
        },
    )
    with open(out / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(
        f"wrote {len(episodes)} episodes / {index} frames to {out} "
        f"({len(included)} included; expert attempted {attempted}, {time.time() - t0:.0f}s)"
    )


def _takeover(args, worker):
    if args.policy_ports is None:
        return None
    return dict(
        port=args.policy_ports[worker],
        steps=tuple(args.takeover_steps),
        exec_horizon=args.exec_horizon,
        flow_steps=args.flow_steps,
        subgoals=args.takeover_subgoals,
        cut=args.takeover_cut,
        overturn=args.takeover_overturn,
    )


def _takeover_summary(args, servers, attempted, kept, failed):
    """Takeover collection: what was served, how the attempts ended, at which step the kept ones were handed over
    (counts per 100 steps) with how many shoots or objects were already cut, and how many began with a preamble."""
    if servers is None:
        return None
    lo, hi = args.takeover_steps
    steps = np.array([r["takeover_step"] for r in kept])
    return dict(
        policy=sorted(servers)[0],
        takeover_steps=[lo, hi],
        exec_horizon=args.exec_horizon,
        flow_steps=args.flow_steps,
        attempts=attempted,
        ended_before_handover=dict(
            sorted(Counter(r["before_handover"] for r in failed if "before_handover" in r).items())
        ),
        ended_in_preamble=dict(sorted(Counter(r["in_preamble"] for r in failed if "in_preamble" in r).items())),
        expert_failed_after_handover=dict(
            sorted(Counter(r["after_handover"] for r in failed if "after_handover" in r).items())
        ),
        kept_by_step={f"{a}-{a + 99}": int(((steps >= a) & (steps < a + 100)).sum()) for a in range(lo, hi + 1, 100)},
        kept_cut_at_handover=dict(sorted(Counter(r["cut_at_handover"] for r in kept).items())),
        kept_after_a_preamble=sum(r["preamble_steps"] > 0 for r in kept),
        preamble_steps_not_recorded=sum(r["preamble_steps"] for r in kept),
        **(
            dict(
                subgoals=True,
                cut=args.takeover_cut,
                overturn_deg=args.takeover_overturn,
                kept_by_reason=dict(sorted(Counter(r.get("handover_reason", "step") for r in kept).items())),
                attempts_by_reason=dict(
                    sorted(Counter(r.get("handover_reason", "step") for r in [*kept, *failed]).items())
                ),
            )
            if args.takeover_subgoals
            else {}
        ),
    )


if __name__ == "__main__":
    main()
