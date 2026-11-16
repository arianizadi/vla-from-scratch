"""The slow level of architecture v2 (tutorial/architecture-v2.md) in closed-loop evaluation: the loop that turns what
is seen into the fast level's sub-goals.

The parts are built elsewhere. What is on the stand comes as a scene record (scene.py), from the simulator's truth or
from the eyes (eyes.py); the scene memory keeps the plant's inventory over the episode (scene.SceneMemory); a typed
program says which shoots to cut, the task's own or one that a language model wrote (program.py, program_writer.py),
and a plan evaluates it on the memory (program.Plan). This module runs them together, one look at a time:

  percept -> the memory takes it -> the plan says Execute, NeedObservation, Complete or Failed -> a decision: the fast
  level's sub-goal ("cut the middle shoot", "show the label", "stop"), or None: hold still for a step and look again

and adds what the loop needs and no single part has:

  Monitored transitions. What was cut is what the memory saw cut, never what a sub-goal asked for: Complete comes
  only once every shoot meant has been seen cut (`stable` looks in a row), and a shoot seen cut that was to stay makes
  the plan say Failed. The step at which the plan first said Complete is kept (declared_complete).
  Holding. While a fact is missing (the inventory is not settled, the flower's colour is not known) the arm holds the
  command it is under for one control step and the slow level looks again. An inventory of fewer than two or more than
  three parts is held the same way: the fast level knows the shoots of such a plant by no name.
  Bounded recovery. A sub-goal that stands for `patience` control steps with nothing changing (the parts seen cut, the
  card read, the sub-goal, the target's bearing by more than 10 degrees, or while the card is to be shown any part's)
  is a stall. Up to `max_recoveries` times the arm is then handed to a scripted move (Recovery) that backs it out of
  the plant; after that the slow level gives up (Failed, "no progress"). With max_recoveries 0 there is no such rule:
  the episode's own step limit applies.
  False success, measured apart. Complete is the slow level's own verdict; the simulator knows whether the shoots
  meant are cut. A Complete the simulator does not confirm is a false completion and is counted apart from successes.
  An end. Once Complete or Failed has stood for `give_up` control steps nothing more will happen, and the episode can
  end.

Supervisor holds these rules without a simulator, so that they can be tested with scripted percepts. SlowLevel is what
eval.py --slow_level v2 drives: it takes the percepts from the simulator or from the slow server (slow_server.py), and
at every look it compares the memory with the simulator, for the record only (nothing of it is fed back). summary
counts the episodes' records for results.json.

This module needs numpy at import and nothing else of the kind: the simulator, the JPEG encoder and the scripted expert
are imported where they are used.
"""

import io
import math
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from .program import Complete, Execute, NeedObservation, Plan, ProgramError, equivalent, from_task, parse
from .program import text as program_text
from .scene import PlantCrop, Scene, SceneMemory, _position_name, scene_truth

KINDS = ("execute", "need_observation", "complete", "failed")  # what a decision can be
SCENES = ("truth", "eyes")  # where the percepts come from
PROGRAMS = ("gold", "writer")  # where the program comes from: the task's own, or the language model's
STOP = "stop"  # the sub-goal of Complete and Failed
TURN = math.radians(10)  # the target's bearing moving by more than this is progress: the plant is being turned
CHECKS = ("parts", "cut", "label", "flower")  # the memory against the simulator, at every look (memory_right)
CAMERAS = ("front", "wrist", "plant")  # what the eyes can ask for: the two cameras and the plant crop
HOME_STEPS = 40  # a recovery without a preamble: control steps of the straight line in joint space to the home pose
JAWS_OPEN = 1.0  # the gripper's command at the start of every episode (env.reset)
PREAMBLE_LIMIT = 200  # control steps after which a recovery ends whatever the expert is doing (its preambles take
# at most about 100; a plate turned by hand during one starts it again)


@dataclass(frozen=True)
class Decision:
    """What the slow level says at one look."""

    kind: str  # one of KINDS
    subgoal: str | None  # the fast level's instruction from now on; None: hold still for this step and look again
    recover: bool = False  # first hand the arm to a scripted recovery (Recovery): nothing has progressed for long
    detail: str = ""  # need_observation: what is missing; failed: why; with a recovery: "no progress"


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _compile(source) -> tuple[str | None, Plan | None]:
    """(the program's text, a plan for it) from a program, the text of one, or None. Text that is not a program keeps
    its text and has no plan."""
    if source is None:
        return None, None
    if isinstance(source, str):
        try:
            source = parse(source)
        except ProgramError:
            return source, None
    return program_text(source), Plan(source)


# --------------------------------------------------------------------------------------------------------- the rules
class Supervisor:
    """The slow level's rules without a simulator: percepts in, decisions out (see the module's docstring).

    program         what to carry out: a program (program.py), its text, or None. Text that does not parse, and None,
                    make every decision Failed ("no program")
    stable, p_min   the scene memory's (scene.SceneMemory)
    patience        control steps a sub-goal may stand with nothing progressing before a recovery is asked for
    max_recoveries  how many recoveries may be asked for; then Failed ("no progress"). 0: no such rule at all
    give_up         control steps a Complete or a Failed must stand before `done`

    `step(percept, step)` takes one look; `step` is the control step, and every clock above counts control steps. The
    runner holds the arm for one control step per hold, so `hold_steps` counts those steps.
    """

    def __init__(self, program, stable=3, p_min=0.7, patience=160, max_recoveries=0, give_up=40):
        assert patience >= 1 and max_recoveries >= 0 and give_up >= 0, (patience, max_recoveries, give_up)
        self.memory = SceneMemory(stable=stable, p_min=p_min)
        self.text, self.plan = _compile(program)
        self.patience, self.max_recoveries, self.give_up = patience, max_recoveries, give_up
        self.decision: Decision | None = None  # the last one
        self.outcomes: list[list] = []  # [step, kind, sub-goal or detail], whenever that changes
        self.looks = 0
        self.hold_steps = 0  # looks that held the arm
        self.inventory_hold_steps = 0  # of those, held because the inventory had fewer than 2 or more than 3 parts
        self.recoveries = 0
        self.recovery_steps: list[int] = []
        self.declared_complete: int | None = None  # the control step at which the plan first said Complete
        self._step = 0  # the step of the last look
        # Progress (rule d): the state it is measured against, the bearings watched then, and since when it stands
        self._state: tuple | None = None
        self._bearings: dict[int, float] = {}
        self._progress_at = 0
        self._resume = False  # a recovery was asked for: its clock starts again when the fast level has the arm
        self._given_up = False
        # The verdict (Complete or Failed, with its detail) that stands now, and since which step
        self._verdict: tuple | None = None
        self._verdict_since = 0

    def step(self, percept: Scene, step: int) -> Decision:
        """Take one look at control step `step`: the memory takes the percept, the plan is stepped, the rules apply."""
        step = int(step)
        self.looks += 1
        self._step = step
        self.memory.update(percept, step)
        if self._resume:  # the first look after a recovery: the arm is the fast level's again, from now on
            self._progress_at, self._resume = step, False
        decision, target = self._decide(step)
        decision = self._watch(decision, target, step)
        self._keep(decision, step)
        self.decision = decision
        return decision

    def _decide(self, step: int) -> tuple[Decision, int | None]:
        """What the plan says for the memory now (rules a to c), and the part an Execute names."""
        if self.plan is None:
            return Decision("failed", STOP, detail="no program"), None
        if self._given_up:  # rule d gave up: final
            return Decision("failed", STOP, detail="no progress"), None
        memory = self.memory
        if memory.settled and not 2 <= len(memory.parts) <= 3:
            # Rule a: the fast level names shoots as the lowest, the middle or the highest of two or three
            # (reasoning.position_name). Hold, and do not let the plan name a shoot it could not mean.
            self.inventory_hold_steps += 1
            return Decision("need_observation", None, detail="plant"), None
        outcome = self.plan.step(memory)
        if isinstance(outcome, Execute):
            return Decision("execute", outcome.subgoal), outcome.part
        if isinstance(outcome, NeedObservation):  # "show the label" for the card, else hold
            return Decision("need_observation", outcome.subgoal, detail=outcome.what), None
        if isinstance(outcome, Complete):  # every shoot meant was seen cut: the memory's rows say so, not a sub-goal
            if self.declared_complete is None:
                self.declared_complete = step
            return Decision("complete", STOP), None
        return Decision("failed", STOP, detail=outcome.reason), None

    def _watch(self, decision: Decision, target: int | None, step: int) -> Decision:
        """Rule d: progress, and a recovery or a failure when there has been none for `patience` control steps."""
        memory = self.memory
        cut = frozenset(part.id for part in memory.parts if part.cut)
        state = (cut, memory.label_read, decision.subgoal, target)
        # A plant that is being turned is progress although nothing else changes. The bearings are compared with those
        # at the last progress, so that a slow turn adds up: the target's, or with no target (showing the label, where
        # turning the plant is the work) every part's, as the plate turns them all
        bearings = {part.id: part.bearing for part in memory.parts if target is None or part.id == target}
        turned = any(abs(_wrap(b - self._bearings[i])) > TURN for i, b in bearings.items() if i in self._bearings)
        if state != self._state or turned:
            self._state, self._bearings, self._progress_at = state, bearings, step
        if self.max_recoveries == 0 or decision.subgoal in (None, STOP) or step - self._progress_at < self.patience:
            return decision
        if self.recoveries < self.max_recoveries:
            self.recoveries += 1
            self.recovery_steps.append(step)
            self._progress_at, self._resume = step, True
            return replace(decision, recover=True, detail="no progress")
        self._given_up = True
        return Decision("failed", STOP, detail="no progress")

    def _keep(self, decision: Decision, step: int):
        """The counters, the outcome changes and the clock of `done`."""
        if decision.subgoal is None:
            self.hold_steps += 1
        shown = decision.detail if decision.kind == "failed" or decision.subgoal is None else decision.subgoal
        if not self.outcomes or self.outcomes[-1][1:] != [decision.kind, shown]:
            self.outcomes.append([step, decision.kind, shown])
        verdict = (decision.kind, decision.detail) if decision.kind in ("complete", "failed") else None
        if verdict != self._verdict:  # a verdict that changes, or goes, starts the clock again
            self._verdict, self._verdict_since = verdict, step

    @property
    def done(self) -> bool:
        """Complete or Failed has stood for `give_up` control steps in a row: nothing more will happen."""
        return self._verdict is not None and self._step - self._verdict_since >= self.give_up

    def record(self) -> dict:
        """Plain types, for the episode's record: the program run, the last decision's kind and its failure (if it
        failed), the step of the first Complete, the outcome changes, the counters and the memory's snapshot."""
        last = self.decision
        return dict(
            program=self.text,
            final=None if last is None else last.kind,
            failure=last.detail if last is not None and last.kind == "failed" else None,
            declared_complete=self.declared_complete,
            outcomes=[list(outcome) for outcome in self.outcomes],
            looks=self.looks,
            hold_steps=self.hold_steps,
            inventory_hold_steps=self.inventory_hold_steps,
            recoveries=self.recoveries,
            recovery_steps=list(self.recovery_steps),
            rebinds=0 if self.plan is None else int(self.plan.rebinds),
            memory=self.memory.snapshot(),
        )


# ---------------------------------------------------------------------------------------------- in the simulator
def _jpeg(image, quality: int = 95) -> bytes:
    """JPEG bytes of an RGB image, as eval.py sends images and collect_scene.py stores them."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(image)).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _view(key: str) -> tuple[str, int]:
    """("front", -20) from an image's request key "front_t-20" (eval.infer_request's names, EyesConfig.keys)."""
    camera, sep, offset = key.rpartition("_t")
    try:
        k = int(offset)
    except ValueError:
        k = 1
    if not sep or camera not in CAMERAS or k > 0:
        raise ValueError(f"the eyes ask for {key!r}: images are <camera>_t<offset>, a camera of {CAMERAS}, offset <= 0")
    if camera == "plant" and k != 0:
        raise ValueError(f"the eyes ask for {key!r}: the plant crop is rendered for the scene now only (offset 0)")
    return camera, k


def _median_ms(seconds: list[float]) -> float | None:
    return None if not seconds else round(1000 * float(np.median(seconds)), 3)


class SlowLevel:
    """The slow level in the simulator (eval.py --slow_level v2): a Supervisor fed with the percepts of a PickupEnv.

    env             a PickupEnv of a plant task
    scene           "truth": the percepts are the simulator's truth (scene.scene_truth); "eyes": the slow server's eyes
                    read them from the camera images
    program         "gold": the reasoning task's own program (program.from_task); "writer": the program the slow
                    server's language model writes for the instruction, once per episode
    server          an open connection to a slow server (slow_server.py); needed for the eyes and for the writer
    stable          the memory's rows; by default 1 for the truth (it is exact) and 3 for the eyes
    p_min, patience, max_recoveries, give_up: the Supervisor's
    truth           what gives the simulator's truth (default scene.scene_truth); a stand-in for tests

    reset() after env.reset (and any random start); tick(obs, history) at every look; record(info) at the end of an
    episode; close() at the end of the evaluation. `offsets` are the frame offsets the eyes look at: a `history` (an
    eval.FrameHistory) passed to tick must reach that far back.
    """

    def __init__(
        self,
        env,
        scene: str = "truth",
        program: str = "gold",
        server=None,
        stable: int | None = None,
        p_min: float = 0.7,
        patience: int = 160,
        max_recoveries: int = 0,
        give_up: int = 40,
        truth=None,
    ):
        assert scene in SCENES, f"scene {scene!r}: one of {SCENES}"
        assert program in PROGRAMS, f"program {program!r}: one of {PROGRAMS}"
        assert env.plant_task, "the slow level reads a plant: it needs a plant task"
        assert program == "writer" or env.reason, "the gold program is a reasoning task's own: it needs a reason_* task"
        self.env, self.scene, self.source, self.server = env, scene, program, server
        self.stable = (1 if scene == "truth" else 3) if stable is None else stable
        self.p_min, self.patience, self.max_recoveries, self.give_up = p_min, patience, max_recoveries, give_up
        self.truth = scene_truth if truth is None else truth
        self.info: dict | None = None  # the slow server's
        if scene == "eyes" or program == "writer":
            assert server is not None, "the eyes and the program writer are served: pass a connection to slow_server.py"
            self.info = self._ask({"cmd": "info"})
            assert scene != "eyes" or self.info.get("eyes"), "the slow server has no eyes (start it with --eyes)"
            assert program != "writer" or self.info.get("writer"), "the slow server has no program writer (--writer)"
        # The images the eyes take: (request key, camera, frame offset); the plant crop is rendered here, made once
        self.views = [(key, *_view(key)) for key in self.info["eyes"]["keys"]] if scene == "eyes" else []
        self.crop = PlantCrop(env) if any(camera == "plant" for _, camera, _ in self.views) else None
        self.supervisor: Supervisor | None = None
        self.gold = None  # the task's own program, when the task has one
        self.written: str | None = None  # what the writer wrote this episode
        self.meaning: bool | None = None  # the program run means what the task's does (program.equivalent)
        self._see_s: list[float] = []
        self._write_s: list[float] = []
        self._looked = 0
        self._right = dict.fromkeys(CHECKS, 0)
        self._wrong = dict.fromkeys(CHECKS, False)

    @property
    def offsets(self) -> list[int]:
        """The frame offsets (control steps, 0 = now) of the images the slow level looks at."""
        return sorted({0, *(offset for _, _, offset in self.views)})

    @property
    def decision(self) -> Decision | None:
        return None if self.supervisor is None else self.supervisor.decision

    @property
    def done(self) -> bool:
        """The verdict has stood for give_up control steps: the episode can end (Supervisor.done)."""
        return self.supervisor is not None and self.supervisor.done

    def _ask(self, request: dict) -> dict:
        self.server.send(request)
        reply = self.server.recv()
        if "error" in reply:
            raise RuntimeError(f"the slow server could not answer {request['cmd']!r}: {reply['error']}")
        return reply

    def reset(self):
        """A new episode: its program and an empty memory. Call after env.reset and any random start."""
        env = self.env
        task = getattr(env, "reason_task", None) if env.reason else None
        self.gold = None if task is None else from_task(task)
        self.written, self._see_s, self._write_s = None, [], []
        if self.source == "gold":
            assert self.gold is not None, "no reasoning task drawn yet: reset the slow level after env.reset"
            given = self.gold
        else:
            reply = self._ask({"cmd": "write", "instruction": env.instruction})
            self._write_s.append(float(reply["latency_s"]))
            given = self.written = str(reply["program"])
        self.supervisor = Supervisor(given, self.stable, self.p_min, self.patience, self.max_recoveries, self.give_up)
        plan = self.supervisor.plan
        self.meaning = None if self.gold is None else plan is not None and equivalent(plan.program, self.gold)
        self._looked = 0
        self._right = dict.fromkeys(CHECKS, 0)
        self._wrong = dict.fromkeys(CHECKS, False)

    def tick(self, obs: dict, history=None) -> Decision:
        """One look at the scene now (`obs`: the newest observation; `history`: frames to reach back to, for eyes that
        look at earlier ones). Returns the decision, and compares the memory with the simulator for the record."""
        assert self.supervisor is not None, "reset() the slow level after env.reset, before the first look"
        percept = self.truth(self.env) if self.scene == "truth" else self._see(obs, history)
        decision = self.supervisor.step(percept, int(self.env.steps))
        self._compare()
        return decision

    def _see(self, obs: dict, history) -> Scene:
        """The eyes' percept: the images under the keys the server listed, as JPEG bytes."""
        images = {}
        for key, camera, offset in self.views:
            if camera == "plant":
                image = self.crop.render()
            elif offset == 0:
                image = obs[camera]
            elif history is None:
                raise ValueError(f"the eyes look {-offset} control steps back ({key}): pass the frame history")
            else:
                image = history.at(offset)[camera]
            images[key] = _jpeg(image)
        reply = self._ask({"cmd": "see", "images": images})
        self._see_s.append(float(reply["latency_s"]))
        return Scene.from_dict(reply["scene"])

    def _compare(self):
        """The memory against the simulator (CHECKS), for the record only. A flower or a card colour not known yet is
        not wrong; the shoots cut are compared as positions on each one's own plant."""
        env, memory = self.env, self.supervisor.memory
        count = len(env.active)
        cut = {_position_name(env.active.index(name), count) for name in env.cut if name in env.active}
        right = dict(
            parts=len(memory.parts) == count,
            cut={memory.position(part.id) for part in memory.parts if part.cut} == cut,
            label=memory.label_read is None or memory.label_read == env.label_colour,
            flower=memory.flower is None or memory.flower == env.flower_colour,
        )
        self._looked += 1
        for name, ok in right.items():
            self._right[name] += int(ok)
            self._wrong[name] = self._wrong[name] or not ok

    def record(self, info: dict) -> dict:
        """The episode's record, plain types (`info`: the simulator's last). The Supervisor's record and: where the
        percepts and the program came from, the task's own program (gold) and what the writer wrote, whether the
        program run means the same as the task's, false_complete (Complete was declared and the simulator does not
        count the episode a success), memory_right (per check, the share of looks at which the memory agreed with the
        simulator and whether it ever disagreed), and the median latencies of the server's see and write in ms."""
        record = self.supervisor.record()
        shares = {name: self._right[name] / self._looked if self._looked else None for name in CHECKS}
        return dict(
            scene=self.scene,
            program_source=self.source,
            gold=None if self.gold is None else program_text(self.gold),
            written=self.written,
            program_meaning=self.meaning,
            false_complete=record["declared_complete"] is not None and not bool(info.get("success")),
            memory_right={name: dict(share=shares[name], ever_wrong=self._wrong[name]) for name in CHECKS},
            see_ms=_median_ms(self._see_s),
            write_ms=_median_ms(self._write_s),
            **record,
        )

    def settings(self) -> dict:
        """How the slow level was set up, for results.json."""
        return dict(
            scene=self.scene,
            program=self.source,
            stable=self.stable,
            p_min=self.p_min,
            patience=self.patience,
            max_recoveries=self.max_recoveries,
            give_up=self.give_up,
            server=self.info,
        )

    def close(self):
        if self.crop is not None:
            self.crop.close()
            self.crop = None


# ----------------------------------------------------------------------------------------------------- recovery
class Recovery:
    """The scripted move of a recovery (eval.py, when a decision asks for one).

    The arm is handed to the scripted expert (expert.ScriptedExpert.take_over), which plans from where the arm is. From
    some poses its plan begins with moves it makes nowhere else, its preamble: open the jaws and back the hand out of
    the plant, or move back above the table in front of it. Those moves are made, and then the fast level has the arm
    again. From a pose that needs no preamble the joints go to the home pose instead, in a straight line in joint space
    over `steps` control steps with the jaws open: the pose every episode starts from.

    act() gives the next command (8 numbers: 7 joint targets and the gripper); done says the move is over. `expert`
    and `home` are for tests (by default a new ScriptedExpert(env, seed=0) and env.HOME_Q).
    """

    def __init__(self, env, expert=None, home=None, steps: int = HOME_STEPS):
        if expert is None:
            from .expert import ScriptedExpert  # the simulator: imported only when a recovery is made

            expert = ScriptedExpert(env, seed=0)
        expert.take_over()
        self.expert, self.made, self.path = expert, 0, None
        if not expert.preamble:
            if home is None:
                from .env import HOME_Q

                home = HOME_Q
            start, home = np.asarray(env.data.qpos[:7], dtype=float), np.asarray(home, dtype=float)
            self.path = [
                np.append(start + (home - start) * (k + 1) / steps, JAWS_OPEN).astype(np.float32) for k in range(steps)
            ]

    @property
    def done(self) -> bool:
        if self.path is not None:
            return self.made >= len(self.path)
        return not self.expert.preamble or self.made >= PREAMBLE_LIMIT

    def act(self) -> np.ndarray:
        assert not self.done, "the recovery is over"
        self.made += 1
        return self.expert.act() if self.path is None else self.path[self.made - 1]


# ------------------------------------------------------------------------------------------------------ summary
def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def summary(episodes: list[dict]) -> dict:
    """results.json's v2 section, from eval.py's per-episode records (each with "success" and "v2", SlowLevel.record):
    successes; episodes in which Complete was declared, and false completions among them; failures by the reason of
    the last decision; programs with the right meaning (of those that have a task's program to compare with);
    recoveries (asked for, episodes with one, successes among those); the mean hold steps; per check of the memory
    against the simulator, the mean of its share of looks right and the number of episodes in which it was ever wrong;
    the median latencies of the server's see and write."""
    rows = [(bool(e["success"]), e["v2"]) for e in episodes if e.get("v2") is not None]
    compared = [r["program_meaning"] for _, r in rows if r["program_meaning"] is not None]
    rescued = [ok for ok, r in rows if r["recoveries"]]
    failures = Counter(r["failure"] for _, r in rows if r["failure"] is not None)
    see = [r["see_ms"] for _, r in rows if r["see_ms"] is not None]
    write = [r["write_ms"] for _, r in rows if r["write_ms"] is not None]
    return dict(
        episodes=len(rows),
        successes=sum(ok for ok, _ in rows),
        declared_complete=sum(r["declared_complete"] is not None for _, r in rows),
        false_complete=sum(bool(r["false_complete"]) for _, r in rows),
        failures=dict(sorted(failures.items())),
        programs_right_meaning=sum(compared),
        programs_compared=len(compared),
        recoveries=dict(asked=sum(r["recoveries"] for _, r in rows), episodes=len(rescued), successes=sum(rescued)),
        mean_hold_steps=_mean(r["hold_steps"] for _, r in rows),
        mean_inventory_hold_steps=_mean(r["inventory_hold_steps"] for _, r in rows),
        memory_right={name: _mean(r["memory_right"][name]["share"] for _, r in rows) for name in CHECKS},
        memory_ever_wrong={name: sum(r["memory_right"][name]["ever_wrong"] for _, r in rows) for name in CHECKS},
        see_ms=float(np.median(see)) if see else None,
        write_ms=float(np.median(write)) if write else None,
    )
