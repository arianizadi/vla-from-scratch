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
  the plan say Failed. The step at which the plan first said Complete is kept (declared_complete). The plan is stepped
  at every look, so a verdict goes when its grounds do (a shoot found later, a cut the memory unmarks: SceneMemory's
  revoke); one that gives way to no verdict at the next look is counted as withdrawn (verdicts_withdrawn).
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
  A ledger of attempts. Each sub-goal the fast level is given (the cut of one part, or showing the label) is an
  attempt, with what it is meant to bring about, and only what the memory shows closes it: confirmed (the part seen
  cut, the card read, with the sighting that shows it), violated (a shoot that was to stay seen cut), superseded (the
  plan moved on without the effect seen) or given up (the no-progress rule). An attempt still open at the end was
  never seen to work, whatever the fast level was told.

Architecture v3 (tutorial/architecture-v3.md; Supervisor with v3, on by itself for a program outside v2's language)
adds: a plan over the worlds the unknown facts allow, in stages (program.WorldPlan); a look at one shoot whose fact is
missing ("show the middle shoot", or a hold), bounded by one full turn of the plate; the home move on a verdict
(Decision.home), with its own attempt in the ledger; and rule f, the cut gate (cut_gate, Supervisor.check_cut), which
the runner asks at every control step.

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

from .program import (
    OUT_OF_ORDER,
    Complete,
    Execute,
    NeedObservation,
    Plan,
    ProgramError,
    WorldPlan,
    cannot_see,
    equivalent,
    from_scenario,
    from_task,
    in_language,
    parse,
)
from .program import text as program_text
from .reasoning import aim_of, aimed, subgoal_target
from .scene import PlantCrop, Scene, SceneMemory, _position_name, scene_truth

KINDS = ("execute", "need_observation", "complete", "failed")  # what a decision can be
SCENES = ("truth", "eyes", "full")  # where the percepts come from (full: the simulator's whole state, v3)
PROGRAMS = ("gold", "writer")  # where the program comes from: the task's own, or the language model's
STOP = "stop"  # the sub-goal of Complete and Failed
SHOW = "show the label"  # the sub-goal of the card not read yet
STAYED = "a shoot that was to stay is cut"  # program.Plan's reason for a cut the program did not ask for
OUTCOMES = ("open", "confirmed", "violated", "superseded", "gave_up")  # of an attempt in the ledger
TURN = math.radians(10)  # the target's bearing moving by more than this is progress: the plant is being turned
# The reset (Supervisor, rule e). The fast level opens its jaws to the cutting width on its way to a shoot and shuts
# them to press a button, so the jaws say which of the two it has chosen. Given a shoot that faces away, it sometimes
# goes for the shoot without turning the plant, and stands there (tutorial/architecture-v2.md, the first move)
FAR = math.radians(45)  # a shoot further than this from facing the robot cannot be cut where it is
JAWS_CUT = (0.45, 0.75)  # the jaws' opening (1 open, 0 shut) while they are set to cut
RESET_LOOKS = 3  # looks in a row with the jaws set to cut, the target that far and the plate still
RESET_GRACE = 4  # looks not counted once the target has passed from one shoot to another: the hand is coming back
#                  from the shoot it cut with its jaws still at the cutting width, and has not chosen yet. Without it
#                  the rule fires after the first cut of nearly every task of two cuts, 4 times in 5 for nothing
STILL = math.radians(
    6
)  # a bearing that moved less than this since the last look: the plate stands (it turns 16 a look)
# What SlowLevel takes where nothing is given, by where its percepts come from. The simulator's truth is exact: one look
# counts, nothing is unseen and no cut needs taking back. The eyes err (tutorial/architecture-v2.md, "First results"):
# three looks in a row, a part they are sure of counts although they judge no camera to show it, a cut is unmarked
# after three confident sightings of the part whole, and a verdict stands twice as long, so that the arm has left the
# plant and the eyes have looked again before an episode ends on it. The settings before these were the truth's with
# stable 3: stable=3, give_up=40, unseen=False, revoke=0
DEFAULTS = dict(
    truth=dict(stable=1, give_up=40, unseen=False, revoke=0),
    eyes=dict(stable=3, give_up=80, unseen=True, revoke=3),
    full=dict(stable=1, give_up=40, unseen=False, revoke=0),
)
# The resets SlowLevel allows where none are given (rule e), whatever its percepts come from: a fast level that is
# given the sub-goal alone goes for a shoot it cannot reach in about one episode in ten, and with up to two resets v2
# solved 180 and 165 of 200 against 169 and 148 (tutorial/architecture-v2.md, "First results"). Not with `aim`: a fast
# level told what its shoot needs does not make that error (one reset asked for in 200 episodes, 166 solved against
# 159 and 156 without the rule). The runs made before this default had resets=0
RESETS = 2
CHECKS = ("parts", "cut", "label", "flower")  # the memory against the simulator, at every look (memory_right)
# The memory's errors that no lag explains (memory_false): more parts than the plant has, a cut the plant does not have
FALSE_CHECKS = ("parts", "cut")
CAMERAS = ("front", "wrist", "plant")  # what the eyes can ask for: the two cameras and the plant crop
HOME_STEPS = 40  # a recovery without a preamble: control steps of the straight line in joint space to the home pose
JAWS_OPEN = 1.0  # the gripper's command at the start of every episode (env.reset)
PREAMBLE_LIMIT = 200  # control steps after which a recovery ends whatever the expert is doing (its preambles take
# at most about 100; a plate turned by hand during one starts it again)
# Architecture v3 (tutorial/architecture-v3.md, "The plan and the runner"; Supervisor with v3)
INSPECT = ("hold", "show")  # what a missing attribute of one shoot asks of the fast level: hold, or "show the X shoot"
LOOK_TURN = 2 * math.pi  # a look for one fact is bounded by one full turn of the plate from when the need arose,
LOOK_STEPS = 360  # and, where the plate does not turn (a hold), by this many control steps: two turns at the buttons'
#                   speed (0.7 rad/s)
HOME = "home"  # the ledger's entry for the home move that follows a verdict
JAWS_CUTTING = sum(JAWS_CUT) / 2  # rule f: the jaws a veto leaves at the cutting width
GATE_BAND = 0.0175  # rule f: metres; the jaws are at the part whose height is nearest, within half the least spacing
FACING = FAR  # rule f: the part authorised faces the robot within this
# Where the jaws are, for the gate (SlowLevel.check_cut): the robot's own kinematics and two constants of the setup.
# Measured with the scripted expert on 600 validation layouts (calibrate_v3.py --gate; the lab notebook has the table):
# open jaws that shut on a shoot (821 cuts) are 2.6 to 4.0 cm from the stem's axis and 0.3 to 1.8 cm (median 1.0)
# above where that shoot leaves the stem, so that the height less GATE_OFFSET is within 0.8 cm of the shoot's own and
# nearest to it in every one; open jaws that shut for anything else (to press a button) are at least 15.7 cm away
GATE_OFFSET = 0.010  # metres: the jaws' height above the stem's base, less this, is the height of the shoot they hold
GATE_AXIS = 0.08  # metres: jaws this near the stem's axis (seen from above) and below the flower are among the shoots
GATE_OPEN = 0.3  # jaws open more than this (1 open, 0 shut) are shut by a command below the cutting width
HOME_JOINTS = 0.05  # radians: the arm is at the home pose when every joint is this near it


@dataclass(frozen=True)
class Decision:
    """What the slow level says at one look."""

    kind: str  # one of KINDS
    subgoal: str | None  # the fast level's instruction from now on; None: hold still for this step and look again
    recover: bool = False  # first hand the arm to a scripted recovery (Recovery): nothing has progressed for long
    detail: str = ""  # need_observation: what is missing; failed: why; with a recovery: "no progress"
    aim: str = ""  # execute, where the slow level says it (Supervisor aim): what the shoot needs, reasoning.aim_of
    home: bool = False  # v3: a verdict; the runner takes the arm to the home pose (Recovery then_home) and holds it

    @property
    def instruction(self) -> str | None:
        """What the fast level is told: the sub-goal, with its aim where there is one ("cut the lowest shoot, press
        blue"). The sub-goal itself stays as it is, so that an aim that changes while a shoot is turned into reach
        starts no new attempt."""
        return self.subgoal if self.subgoal is None else aimed(self.subgoal, self.aim)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _compile(source, v3: bool | None = None) -> tuple[str | None, Plan | WorldPlan | None, bool]:
    """(the program's text, a plan for it, whether it is run as v3) from a program, the text of one, or None. Text that
    is not a program keeps its text and has no plan. v3 None: a program outside v2's language is run as v3, and text is
    read in v2's language; v3 True: text is read in v3's language and the plan is a WorldPlan."""
    if source is None:
        return None, None, bool(v3)
    if isinstance(source, str):
        try:
            source = parse(source, "v3" if v3 else "v2")
        except ProgramError:
            return source, None, bool(v3)
    if v3 is None:
        v3 = not in_language(source, "v2")
    return program_text(source), WorldPlan(source) if v3 else Plan(source), v3


def cut_gate(jaws: float, height: float | None, among: bool, parts, authorised: int | None) -> tuple[bool, int | None]:
    """Rule f, the cut gate, as a function: may a command that sets the jaws to `jaws` (1 open, 0 shut) be carried out
    with the gripper at `height` (metres above the stem's base) and among the shoots (`among`: near the stand's axis
    and in the band of heights where shoots grow, from the robot's own kinematics)? `parts`: the memory's inventory
    (`id`, `height`, `bearing`); `authorised`: the part the plan has authorised (the target of the Execute in force),
    or None. Returns (allowed, the part at that height, or None). A command that does not shut the jaws below the
    cutting width, or one away from the shoots, is always allowed. One that shuts them among the shoots is allowed only
    at the part authorised (the nearest part by height, within GATE_BAND) while it faces the robot (FACING)."""
    if not among or height is None or jaws >= JAWS_CUT[0]:
        return True, None
    near = min(parts, key=lambda part: abs(part.height - height), default=None)
    at = near.id if near is not None and abs(near.height - height) <= GATE_BAND else None
    faces = near is not None and near.bearing is not None and abs(near.bearing) <= FACING
    return at is not None and at == authorised and faces, at


# --------------------------------------------------------------------------------------------------------- the rules
class Supervisor:
    """The slow level's rules without a simulator: percepts in, decisions out (see the module's docstring).

    program         what to carry out: a program (program.py), its text, or None. Text that does not parse, and None,
                    make every decision Failed ("no program")
    stable, p_min   the scene memory's (scene.SceneMemory)
    unseen, revoke  the scene memory's options (scene.SceneMemory, rules 1 to 3), off by default
    patience        control steps a sub-goal may stand with nothing progressing before a recovery is asked for
    max_recoveries  how many recoveries may be asked for; then Failed ("no progress"). 0: no such rule at all
    give_up         control steps a Complete or a Failed must stand before `done`
    resets          how many resets may be asked for (rule e). When the target of an Execute is further than FAR
                    from facing the robot, the plate has not turned and the jaws are set to cut (JAWS_CUT; `jaws`
                    of step) for RESET_LOOKS looks in a row, the fast level has gone for a shoot it cannot reach:
                    the decision asks for a recovery that ends at the start pose (detail "reset"), where the fast
                    level chooses again. The first RESET_GRACE looks at a target that follows another do not count.
                    0: no such rule
    aim             an Execute also says what its shoot needs, from the part's remembered bearing (Decision.aim:
                    "in reach", "press blue", "press orange"), for a fast level trained to be told
                    (dp_vit_lang_reactive_aim.yaml). Off by default: the fast level decides by itself. A "press"
                    stands until the plate stands still (STILL), as the note it was written from stood until the
                    turn had ended: a shoot that comes within reach while the plate still turns is not yet "in reach"
    v3              architecture v3 (tutorial/architecture-v3.md): the plan is a WorldPlan (the program evaluated in
                    every world the unknown facts allow, in stages), a verdict asks for the home move (Decision.home),
                    and a look for an attribute of one shoot is bounded. None (the default): v3 for a program outside
                    v2's language, v2 otherwise, so that v2's runs are what they were; text is then read as v2's
    inspect         v3: an attribute of one shoot that is missing asks the fast level to "show the <position> shoot"
                    ("show") or holds ("hold", the default: the fast level has no such skill yet). Either way the look
                    is bounded by one full turn of the plate (LOOK_TURN, from the memory's bearings) or LOOK_STEPS
                    control steps from when the need arose; then Failed, "cannot see the tie of the middle shoot",
                    for good: a look that fails is never a "no"
    gate            rule f, the cut gate (`check_cut`, at every control step): a command that would shut the jaws
                    among the shoots anywhere but at the part authorised, facing the robot, leaves them at the cutting
                    width, is recorded as a veto and asks for a reset (as rule e does, within `resets`). Off: the same
                    is recorded as a veto that would have been, and nothing is changed
    attribute_rule, witness: the scene memory's (SceneMemory rules 8 and 9)

    `step(percept, step)` takes one look; `step` is the control step, and every clock above counts control steps. The
    runner holds the arm for one control step per hold, so `hold_steps` counts those steps. `withdrawn` counts the
    looks at which a verdict (Complete or Failed) that stood at the look before gave way to no verdict.

    `ledger`: one entry per attempt (see the module's docstring), plain data: {"sub_goal", "part" (the target's id, None
    for the label), "position", "expected" (what the sub-goal is meant to bring about, in words), "opened", "closed"
    (control steps; closed None while open), "outcome" (OUTCOMES), "evidence", "recoveries" (asked for during it)}.
    An attempt opens when a decision first gives a sub-goal other than "stop" that differs from the open attempt's, or
    names another part. A hold leaves it open. The evidence of a confirmed cut is the part's cut_evidence
    (scene.Remembered), of the card {"step", "label"}: the look at which the memory read it; of a violation, the
    cut_evidence of the shoot that was to stay, with its id ("part"). An entry once closed stays as it is: a confirmed
    cut that the memory unmarks later (revoke) is still confirmed there, and the plan's next Execute of that part opens
    another attempt.
    """

    def __init__(
        self,
        program,
        stable=3,
        p_min=0.7,
        patience=160,
        max_recoveries=0,
        give_up=40,
        unseen=False,
        revoke=0,
        aim=False,
        resets=0,
        v3=None,
        inspect="hold",
        gate=False,
        attribute_rule="row",
        witness=False,
    ):
        assert patience >= 1 and max_recoveries >= 0 and give_up >= 0, (patience, max_recoveries, give_up)
        assert resets >= 0, resets
        assert inspect in INSPECT, f"inspect {inspect!r}: one of {INSPECT}"
        self.memory = SceneMemory(
            stable=stable, p_min=p_min, unseen=unseen, revoke=revoke, attribute_rule=attribute_rule, witness=witness
        )
        self.text, self.plan, self.v3 = _compile(program, v3)
        self.inspect, self.gate_on = inspect, bool(gate)
        # v3: the look in progress ({"fact", "since", "turned"}), the bound's verdict once it ran out, the plate's turn
        self._look: dict | None = None
        self._look_failed: str | None = None
        self._seen_bearings: dict[int, float] = {}  # part id -> bearing, of the parts seen at the last look
        self.looks_asked = 0  # the looks for a fact of one shoot that began
        self.look_turn = 0.0  # radians the plate turned while one was going on
        self.attribute_hold_steps = 0  # looks that held for an attribute
        self.complete_without_action: bool | None = None  # Complete declared before any sub-goal was given
        self.home_reached: int | None = None  # the control step at which the arm was first reported home on a verdict
        # rule f: the part authorised now, the vetoes ({"step", "at", "authorised", "carried_out"}), a reset asked
        self._authorised: int | None = None
        self._vetoing = False
        self.vetoes: list[dict] = []
        self._veto_reset = False
        self.patience, self.max_recoveries, self.give_up = patience, max_recoveries, give_up
        self.aim = bool(aim)
        self._aimed: dict[int, tuple] = {}  # part id -> (the aim last said of it, its bearing then)
        self.max_resets, self.resets = resets, 0
        self.reset_steps: list[int] = []
        self._cut_row = 0  # rule e: looks in a row with the jaws set to cut and the target out of reach
        self._cut_from: tuple | None = None  # the target of that row and its bearing when the row began
        self._reset_target: int | None = None  # the last target an Execute named
        self._reset_grace = 0  # looks still not counted, after the target passed from one shoot to another
        self.decision: Decision | None = None  # the last one
        self.outcomes: list[list] = []  # [step, kind, sub-goal or detail], whenever that changes
        self.looks = 0
        self.hold_steps = 0  # looks that held the arm
        self.inventory_hold_steps = 0  # of those, held because the inventory had fewer than 2 or more than 3 parts
        self.recoveries = 0
        self.recovery_steps: list[int] = []
        self.declared_complete: int | None = None  # the control step at which the plan first said Complete
        self.withdrawn = 0  # verdicts that gave way to no verdict at the next look
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
        self.ledger: list[dict] = []
        self._attempt: dict | None = None  # the open one, an entry of the ledger
        self._label_step: int | None = None  # the look at which the memory's label_read took its value

    def step(self, percept: Scene, step: int, jaws: float | None = None, home: bool | None = None) -> Decision:
        """Take one look at control step `step`: the memory takes the percept, the plan is stepped, the rules apply.
        `jaws`: how far the gripper is open now (1 open, 0 shut), for rule e; None where that is not known. `home`
        (v3): whether the arm is at the home pose now, from its own joint angles; None where that is not known."""
        step = int(step)
        self.looks += 1
        self._step = step
        read = self.memory.label_read
        self.memory.update(percept, step)
        if self.memory.label_read != read:
            self._label_step = step
        if self._resume:  # the first look after a recovery: the arm is the fast level's again, from now on
            self._progress_at, self._resume = step, False
        turned = self._plate_turn(step)
        decision, target = self._decide(step, turned)
        decision = self._watch(decision, target, step)
        decision = self._reset(decision, target, jaws, step)
        decision = self._vetoed(decision, step)
        self._keep(decision, step)
        self._account(decision, target, step, home)
        self._authorised = target if decision.kind == "execute" else None  # rule f
        self.decision = decision
        return decision

    # --------------------------------------------------------------------------------------------------------- v3
    def _plate_turn(self, step: int) -> float:
        """How far the plate turned since the last look, from the memory's bearings of the parts seen at both (their
        median change); 0 when no part was."""
        seen = {part.id: part.bearing for part in self.memory.parts if part.seen_step == step}
        changes = [_wrap(b - self._seen_bearings[i]) for i, b in seen.items() if i in self._seen_bearings]
        self._seen_bearings = seen
        return float(np.median(changes)) if changes else 0.0

    def _inspect(self, outcome: NeedObservation, step: int, turned: float) -> Decision:
        """v3: a fact about one shoot is missing. The look is bounded; past the bound, Failed for good."""
        fact = (outcome.what, outcome.part)
        if self._look is None or self._look["fact"] != fact:
            self._look = dict(fact=fact, since=step, turned=0.0)
            self.looks_asked += 1
        else:
            self._look["turned"] += abs(turned)
            self.look_turn += abs(turned)
        if self._look["turned"] >= LOOK_TURN or step - self._look["since"] >= LOOK_STEPS:
            self._look_failed = cannot_see(outcome.what, self.memory.position(outcome.part))
            return Decision("failed", STOP, detail=self._look_failed, home=True)
        subgoal = outcome.subgoal if self.inspect == "show" else None
        if subgoal is None:
            self.attribute_hold_steps += 1
        return Decision("need_observation", subgoal, detail=outcome.what)

    def check_cut(self, jaws: float, height: float | None, among: bool, step: int) -> float:
        """Rule f, at every control step: the jaws' command to carry out in place of `jaws` (cut_gate). A command
        that shuts the jaws among the shoots is recorded for the memory's witness rule when it is carried out; one
        that the gate refuses is a veto, recorded once per run of control steps that it lasts, and with the gate on
        it leaves the jaws at the cutting width and asks for a reset at the next look."""
        allowed, at = cut_gate(jaws, height, among, self.memory.parts, self._authorised)
        shutting = among and height is not None and jaws < JAWS_CUT[0]
        if allowed:
            self._vetoing = False
            if shutting:
                self.memory.jaws_shut(height, step)
            return jaws
        if not self._vetoing:
            self.vetoes.append(dict(step=int(step), at=at, authorised=self._authorised, carried_out=not self.gate_on))
            self._veto_reset = self.gate_on
        self._vetoing = True
        if not self.gate_on:
            self.memory.jaws_shut(height, step)
            return jaws
        return JAWS_CUTTING

    def _vetoed(self, decision: Decision, step: int) -> Decision:
        """Rule f: a veto since the last look asks for a reset, within the resets allowed."""
        asked, self._veto_reset = self._veto_reset, False
        if not asked or decision.home or decision.recover or self.resets >= self.max_resets:
            return decision
        self.resets += 1
        self.reset_steps.append(step)
        self._cut_row, self._cut_from = 0, None
        self._progress_at, self._resume = step, True
        return replace(decision, recover=True, detail="veto")

    def _decide(self, step: int, turned: float = 0.0) -> tuple[Decision, int | None]:
        """What the plan says for the memory now (rules a to c), and the part an Execute (or a look) names."""
        if self.plan is None:
            return Decision("failed", STOP, detail="no program", home=self.v3), None
        if self._given_up:  # rule d gave up: final
            return Decision("failed", STOP, detail="no progress", home=self.v3), None
        if self._look_failed is not None:  # v3: a look ran out of its bound: final
            return Decision("failed", STOP, detail=self._look_failed, home=True), None
        memory = self.memory
        if memory.settled and not 2 <= len(memory.parts) <= 3:
            # Rule a: the fast level names shoots as the lowest, the middle or the highest of two or three
            # (reasoning.position_name). Hold, and do not let the plan name a shoot it could not mean.
            self.inventory_hold_steps += 1
            return Decision("need_observation", None, detail="plant"), None
        outcome = self.plan.step(memory)
        if not (isinstance(outcome, NeedObservation) and outcome.part is not None):
            self._look = None  # v3: no look in progress
        if isinstance(outcome, Execute):
            aim = self._aim(outcome.part) if self.aim else ""
            return Decision("execute", outcome.subgoal, aim=aim), outcome.part
        if isinstance(outcome, NeedObservation) and outcome.part is not None:  # v3: an attribute of one shoot
            decision = self._inspect(outcome, step, turned)
            return decision, outcome.part if decision.subgoal is not None else None
        if isinstance(outcome, NeedObservation):  # "show the label" for the card, else hold
            return Decision("need_observation", outcome.subgoal, detail=outcome.what), None
        if isinstance(outcome, Complete):  # every shoot meant was seen cut: the memory's rows say so, not a sub-goal
            if self.declared_complete is None:
                self.declared_complete = step
                self.complete_without_action = not self.ledger
            return Decision("complete", STOP, home=self.v3), None
        return Decision("failed", STOP, detail=outcome.reason, home=self.v3), None

    def _aim(self, target: int) -> str:
        """What the target needs, from where the memory last saw it point; a "press" is kept while the plate turns."""
        bearing = next((part.bearing for part in self.memory.parts if part.id == target), None)
        aim = aim_of(bearing)
        said, then = self._aimed.get(target, ("", None))
        if aim == "in reach" and said.startswith("press") and then is not None and abs(_wrap(bearing - then)) > STILL:
            aim = said
        self._aimed[target] = (aim, bearing)
        return aim

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

    def _reset(self, decision: Decision, target: int | None, jaws: float | None, step: int) -> Decision:
        """Rule e: the jaws are set to cut a shoot that is out of reach, and the plate is not turning."""
        if target is not None and target != self._reset_target:
            if self._reset_target is not None:  # after another shoot: the hand is still coming back from it
                self._reset_grace = RESET_GRACE
            self._reset_target = target
        waiting, self._reset_grace = self._reset_grace > 0, max(0, self._reset_grace - 1)
        part = next((p for p in self.memory.parts if p.id == target), None)
        bearing = None if part is None else part.bearing
        wrong = (
            decision.kind == "execute"
            and not waiting
            and not decision.recover
            and jaws is not None
            and JAWS_CUT[0] <= jaws <= JAWS_CUT[1]
            and bearing is not None
            and abs(bearing) > FAR
        )
        if self.max_resets == 0 or not wrong:
            self._cut_row, self._cut_from = 0, None
            return decision
        if self._cut_from is None or self._cut_from[0] != target or abs(_wrap(bearing - self._cut_from[1])) > TURN:
            self._cut_row, self._cut_from = 0, (target, bearing)  # another target, or the plate is turning after all
        self._cut_row += 1
        if self._cut_row < RESET_LOOKS or self.resets >= self.max_resets:
            return decision
        self.resets += 1
        self.reset_steps.append(step)
        self._cut_row, self._cut_from = 0, None
        self._progress_at, self._resume = step, True  # as after a recovery: rule d's clock starts again
        return replace(decision, recover=True, detail="reset")

    def _keep(self, decision: Decision, step: int):
        """The counters, the outcome changes and the clock of `done`."""
        if decision.subgoal is None:
            self.hold_steps += 1
        shown = decision.detail if decision.kind == "failed" or decision.subgoal is None else decision.subgoal
        if not self.outcomes or self.outcomes[-1][1:] != [decision.kind, shown]:
            self.outcomes.append([step, decision.kind, shown])
        verdict = (decision.kind, decision.detail) if decision.kind in ("complete", "failed") else None
        if self._verdict is not None and verdict is None:  # withdrawn; one that changes into another is not
            self.withdrawn += 1
        if verdict != self._verdict:  # a verdict that changes, or goes, starts the clock again
            self._verdict, self._verdict_since = verdict, step

    def _account(self, decision: Decision, target: int | None, step: int, home: bool | None = None):
        """The ledger: the open attempt is closed by what the memory shows, by the plan moving on or by rule d, and a
        new one opens when the decision gives the fast level a new sub-goal. Recording only: no decision reads it.
        v3 adds two kinds of attempt: a look at one shoot (closed by the fact read, or given up when the look's bound
        runs out) and the home move after a verdict (closed by the arm reported at the home pose)."""
        memory, attempt = self.memory, self._attempt
        violated = decision.kind == "failed" and decision.detail in (STAYED, OUT_OF_ORDER)
        if attempt is not None and attempt.get("fact") is not None:  # v3: a look at one shoot
            part = next((p for p in memory.parts if p.id == attempt["part"]), None)
            if part is not None and attempt["fact"] in part.attributes:
                self._close("confirmed", step, dict(step=step, value=part.attributes[attempt["fact"]]))
            elif decision.kind == "failed" and decision.detail.startswith("cannot see"):
                self._close("gave_up", step, None)
        elif attempt is not None and attempt["sub_goal"] == HOME:  # v3: the home move
            if not decision.home:
                self._close("superseded", step, None)  # the verdict was withdrawn
            elif home:
                self.home_reached = step if self.home_reached is None else self.home_reached
                self._close("confirmed", step, dict(step=step))
            if decision.home:
                return
        elif attempt is not None and attempt["part"] is not None:
            # A cut is meant to leave its part cut and no other: a shoot that was to stay seen cut at the same look
            # makes it a violation, even with its own part cut
            part = next((p for p in memory.parts if p.id == attempt["part"]), None)
            if violated:
                self._close("violated", step, self._stayed())
            elif part is not None and part.cut:
                self._close("confirmed", step, dict(part.cut_evidence))
        elif attempt is not None and attempt["sub_goal"] == SHOW and memory.label_read is not None:
            self._close("confirmed", step, dict(step=self._label_step, label=memory.label_read))
        elif attempt is not None and violated:
            self._close("violated", step, self._stayed())
        if self._attempt is not None and decision.kind == "failed" and decision.detail == "no progress":
            self._close("gave_up", step, None)
        subgoal = decision.subgoal
        if subgoal is None:  # a hold: the attempt goes on
            return
        if subgoal == STOP:  # Complete or Failed with the open attempt's effect not seen: the plan has moved on
            if self._attempt is not None:
                self._close("superseded", step, None)
            if decision.home and not (self.decision is not None and self.decision.home):  # v3: a verdict begins
                self._open(step, sub_goal=HOME, part=None, position=None, expected="the arm at the home pose")
                if home:
                    self.home_reached = step if self.home_reached is None else self.home_reached
                    self._close("confirmed", step, dict(step=step))
            return
        attempt = self._attempt
        if attempt is not None and (attempt["sub_goal"], attempt["part"]) == (subgoal, target):
            attempt["recoveries"] += int(decision.recover)
            return
        if attempt is not None:
            self._close("superseded", step, None)
        position = subgoal_target(subgoal)
        if decision.kind == "need_observation" and target is not None:  # v3: "show the <position> shoot"
            position = memory.position(target)
            self._open(
                step,
                decision.recover,
                sub_goal=subgoal,
                part=target,
                position=position,
                expected=f"the {decision.detail} of the {position} shoot read",
                fact=decision.detail,
            )
            return
        if position is not None:
            expected = f"the {position} shoot cut, no other shoot cut"
        else:
            expected = "the label read" if subgoal == SHOW else subgoal
        self._attempt = dict(
            sub_goal=subgoal,
            part=target,
            position=position,
            expected=expected,
            opened=step,
            closed=None,
            outcome="open",
            evidence=None,
            recoveries=int(decision.recover),
        )
        self.ledger.append(self._attempt)

    def _close(self, outcome: str, step: int, evidence: dict | None):
        self._attempt.update(outcome=outcome, closed=step, evidence=evidence)
        self._attempt = None

    def _open(self, step: int, recover: bool = False, **entry):
        self._attempt = dict(entry, opened=step, closed=None, outcome="open", evidence=None, recoveries=int(recover))
        self.ledger.append(self._attempt)

    def _stayed(self) -> dict | None:
        """The cut_evidence of the shoot that was to stay and was seen cut (the last such), with its id."""
        meant = set(self.plan.targets or ()) if self.plan is not None else set()
        stayed = [p for p in self.memory.parts if p.cut and p.id not in meant and p.cut_evidence is not None]
        if not stayed:
            return None
        part = max(stayed, key=lambda p: p.cut_evidence["step"])
        return dict(part.cut_evidence, part=int(part.id))

    @property
    def done(self) -> bool:
        """Complete or Failed has stood for `give_up` control steps in a row: nothing more will happen."""
        return self._verdict is not None and self._step - self._verdict_since >= self.give_up

    def record(self) -> dict:
        """Plain types, for the episode's record: the program run, the last decision's kind and its failure (if it
        failed), the step of the first Complete, the verdicts withdrawn, the outcome changes, the counters, the cuts
        the memory unmarked (revocations, over every part), the ledger (an attempt still unresolved is "open") and the
        memory's snapshot."""
        last = self.decision
        return dict(
            program=self.text,
            final=None if last is None else last.kind,
            failure=last.detail if last is not None and last.kind == "failed" else None,
            declared_complete=self.declared_complete,
            verdicts_withdrawn=self.withdrawn,
            outcomes=[list(outcome) for outcome in self.outcomes],
            looks=self.looks,
            hold_steps=self.hold_steps,
            inventory_hold_steps=self.inventory_hold_steps,
            recoveries=self.recoveries,
            recovery_steps=list(self.recovery_steps),
            **(dict(resets=self.resets, reset_steps=list(self.reset_steps)) if self.max_resets else {}),
            rebinds=0 if self.plan is None else int(self.plan.rebinds),
            revocations=sum(int(part.revocations) for part in self.memory.parts),
            ledger=[
                dict(entry, evidence=None if entry["evidence"] is None else dict(entry["evidence"]))
                for entry in self.ledger
            ],
            memory=self.memory.snapshot(),
            **(dict(v3=self._record_v3()) if self.v3 else {}),
        )

    def _record_v3(self) -> dict:
        """v3's counters (tutorial/architecture-v3.md, "What is reported")."""
        plan = self.plan if isinstance(self.plan, WorldPlan) else None
        return dict(
            inspect=self.inspect,
            gate=self.gate_on,
            stages=None if plan is None or plan.stages is None else [list(stage) for stage in plan.stages],
            worlds=None if plan is None else int(plan.worlds),
            choices=None if plan is None else len(plan.choices),
            complete_without_action=self.complete_without_action,
            looks_asked=self.looks_asked,
            look_turn=round(self.look_turn, 4),
            look_failed=self._look_failed,
            attribute_hold_steps=self.attribute_hold_steps,
            ended_holding_for_attribute=self._look is not None
            and self.decision is not None
            and self.decision.kind == "need_observation",
            vetoes=[dict(v) for v in self.vetoes],
            home_reached=self.home_reached,
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
    stable, give_up, unseen, revoke: the Supervisor's; where one is None it is DEFAULTS[scene]'s (for the truth 1, 40,
                    off and 0; for the eyes 3, 80, on and 3)
    p_min, patience, max_recoveries, aim: the Supervisor's
    resets          the Supervisor's; None: RESETS (2), or 0 with `aim`. 0: the rule is off
    truth           what gives the simulator's truth (default scene.scene_truth); a stand-in for tests
    v3              architecture v3 (eval.py --slow_level v3): the Supervisor's v3 rules; "gold" is then the program of
                    the situation the episode draws (program.from_scenario), on prune and its situations as on a
                    reasoning task; the scene may be "full" (the simulator's whole state: scene_truth full=True);
                    `tick` tells the supervisor whether the arm is at the home pose, and `check_cut` is the cut gate
    inspect, gate, attribute_rule, witness: the Supervisor's (v3)

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
        give_up: int | None = None,
        unseen: bool | None = None,
        revoke: int | None = None,
        aim: bool = False,
        resets: int | None = None,
        truth=None,
        v3: bool = False,
        inspect: str = "hold",
        gate: bool = False,
        attribute_rule: str = "row",
        witness: bool = False,
    ):
        assert scene in SCENES, f"scene {scene!r}: one of {SCENES}"
        assert program in PROGRAMS, f"program {program!r}: one of {PROGRAMS}"
        assert env.plant_task, "the slow level reads a plant: it needs a plant task"
        assert program == "writer" or env.reason or v3, (
            "the gold program is a reasoning task's own: it needs a reason_* task (or v3, for a situation's)"
        )
        assert scene != "full" or v3, "the whole state (scene full) is v3's oracle"
        self.v3, self.inspect, self.gate = bool(v3), inspect, bool(gate)
        self.attribute_rule, self.witness = attribute_rule, bool(witness)
        self._stem: tuple | None = None  # (where the stand's axis is, how high the stem's base): the gate's constants
        self.env, self.scene, self.source, self.server = env, scene, program, server
        given = dict(stable=stable, give_up=give_up, unseen=unseen, revoke=revoke)
        chosen = {name: DEFAULTS[scene][name] if value is None else value for name, value in given.items()}
        self.stable, self.give_up, self.unseen, self.revoke = (chosen[name] for name in given)
        self.p_min, self.patience, self.max_recoveries, self.aim = p_min, patience, max_recoveries, bool(aim)
        self.resets = (0 if self.aim else RESETS) if resets is None else resets
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
        self._false = dict.fromkeys(FALSE_CHECKS, False)

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
        if self.v3:
            self.gold = from_scenario(env)  # a reasoning task's own, or the situation's
            self._stem = self._setup()
        self.written, self._see_s, self._write_s = None, [], []
        if self.source == "gold":
            assert self.gold is not None, "no reasoning task drawn yet: reset the slow level after env.reset"
            given = self.gold
        else:
            reply = self._ask({"cmd": "write", "instruction": env.instruction})
            self._write_s.append(float(reply["latency_s"]))
            given = self.written = str(reply["program"])
        self.supervisor = Supervisor(
            given,
            self.stable,
            self.p_min,
            self.patience,
            self.max_recoveries,
            self.give_up,
            unseen=self.unseen,
            revoke=self.revoke,
            aim=self.aim,
            resets=self.resets,
            **(
                dict(
                    v3=True,
                    inspect=self.inspect,
                    gate=self.gate,
                    attribute_rule=self.attribute_rule,
                    witness=self.witness,
                )
                if self.v3
                else {}
            ),
        )
        plan = self.supervisor.plan
        self.meaning = None if self.gold is None else plan is not None and equivalent(plan.program, self.gold)
        self._looked = 0
        self._right = dict.fromkeys(CHECKS, 0)
        self._wrong = dict.fromkeys(CHECKS, False)
        self._false = dict.fromkeys(FALSE_CHECKS, False)

    def tick(self, obs: dict, history=None) -> Decision:
        """One look at the scene now (`obs`: the newest observation; `history`: frames to reach back to, for eyes that
        look at earlier ones). Returns the decision, and compares the memory with the simulator for the record."""
        assert self.supervisor is not None, "reset() the slow level after env.reset, before the first look"
        if self.scene == "eyes":
            percept = self._see(obs, history)
        else:
            percept = self.truth(self.env, full=True) if self.scene == "full" else self.truth(self.env)
        state = None if obs is None else obs.get("state")  # the robot's own: seven joints and the jaws' opening
        jaws = None if state is None else float(state[7])
        home = self.at_home(state) if self.v3 else None
        decision = self.supervisor.step(percept, int(self.env.steps), jaws=jaws, home=home)
        self._compare()
        return decision

    # ------------------------------------------------------------------------------------------- v3: the runner
    def _setup(self) -> tuple:
        """The gate's two constants of the setup: where the stand's axis is (as built, without the episode's jitter)
        and how high the main stem's base is."""
        from .env import STAND_XY

        base, _ = self.env.stem_axis()
        return np.array(STAND_XY, dtype=float), float(base[2])

    @staticmethod
    def at_home(state) -> bool | None:
        """The arm at the home pose, from its own joint angles (None where they are not known)."""
        if state is None:
            return None
        from .env import HOME_Q

        return bool(np.max(np.abs(np.asarray(state[:7], dtype=float) - HOME_Q)) <= HOME_JOINTS)

    def jaws_place(self) -> tuple[float, bool]:
        """(the height on the stem the jaws are at, whether they are among the shoots), from the gripper's position
        (the robot's forward kinematics) and the setup's constants: the jaws' height above the stem's base less
        GATE_OFFSET, and among the shoots within GATE_AXIS of the stand's axis and below the top of the flower."""
        from .env import STEM_H

        axis, base = self._stem if self._stem is not None else self._setup()
        tcp = self.env.tcp_pos()
        height = float(tcp[2] - base)
        among = float(np.linalg.norm(tcp[:2] - axis)) < GATE_AXIS and height < STEM_H
        return height - GATE_OFFSET, among

    def check_cut(self, action, state=None) -> np.ndarray:
        """Rule f at one control step: `action` with the jaws' command the gate allows (Supervisor.check_cut). The
        jaws count as shutting when the command is below the cutting width and they are open (`state`, the robot's
        own, more than GATE_OPEN; open where it is not given)."""
        action = np.array(action, dtype=np.float32)
        height, among = self.jaws_place()
        jaws_open = state is None or float(state[7]) > GATE_OPEN
        command = self.supervisor.check_cut(float(action[7]), height, among and jaws_open, int(self.env.steps))
        action[7] = command
        return action

    def situation(self) -> dict:
        """What the episode asks, for results.json's v3 section: the situation (the scenario that sets the task, else
        the task), its mode, the shoots on the plant and the cuts meant."""
        env = self.env
        name, mode, chooses = env.task, None, False
        for scenario in env.scenarios:
            variation = env.scenario_info.get(scenario.name, {})
            if scenario.changes_task and variation.get("applied"):
                name, mode = scenario.name, variation.get("mode")
                # ties, sick, buds ("targets") and order ("ordered"): env.targets is the shoots meant, perhaps none
                chooses = bool(env.flags & {"targets", "ordered"})
        if env.reason or chooses:
            meant = len(env.targets)
        elif name.startswith("nothing_to_do"):
            meant = 0
        else:
            meant = len(env.targets) if env.targets else len(env.active)
        undrawn = any(s.undrawn for s in env.scenarios)
        return dict(situation=name, mode=mode, shoots=len(env.active), cuts_meant=meant, undrawn=undrawn)

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
        not wrong; the shoots cut are compared as positions on each one's own plant. FALSE_CHECKS are the errors that
        are not mere lag: more parts than the plant has, a position marked cut that the simulator does not have cut."""
        env, memory = self.env, self.supervisor.memory
        count = len(env.active)
        cut = {_position_name(env.active.index(name), count) for name in env.cut if name in env.active}
        marked = {memory.position(part.id) for part in memory.parts if part.cut}
        right = dict(
            parts=len(memory.parts) == count,
            cut=marked == cut,
            label=memory.label_read is None or memory.label_read == env.label_colour,
            flower=memory.flower is None or memory.flower == env.flower_colour,
        )
        self._looked += 1
        for name, ok in right.items():
            self._right[name] += int(ok)
            self._wrong[name] = self._wrong[name] or not ok
        # positions are named over each side's own count: a cut can be judged only when the counts agree
        errors = dict(parts=len(memory.parts) > count, cut=len(memory.parts) == count and not marked <= cut)
        for name, made in errors.items():
            self._false[name] = self._false[name] or made

    def record(self, info: dict) -> dict:
        """The episode's record, plain types (`info`: the simulator's last). The Supervisor's record and: where the
        percepts and the program came from, the task's own program (gold) and what the writer wrote, whether the
        program run means the same as the task's, false_complete (Complete was declared at some look and the simulator
        does not count the episode a success), ended_false_complete (the same with Complete as the last decision: one
        that was withdrawn does not count), memory_right (per check, the share of looks at which the memory agreed
        with the simulator and whether it ever disagreed), memory_false (per FALSE_CHECKS, whether the memory ever made
        that error) and the median latencies of the server's see and write in ms."""
        record = self.supervisor.record()
        shares = {name: self._right[name] / self._looked if self._looked else None for name in CHECKS}
        return dict(
            scene=self.scene,
            program_source=self.source,
            gold=None if self.gold is None else program_text(self.gold),
            written=self.written,
            program_meaning=self.meaning,
            false_complete=record["declared_complete"] is not None and not bool(info.get("success")),
            ended_false_complete=record["final"] == "complete" and not bool(info.get("success")),
            memory_right={name: dict(share=shares[name], ever_wrong=self._wrong[name]) for name in CHECKS},
            memory_false=dict(self._false),
            see_ms=_median_ms(self._see_s),
            write_ms=_median_ms(self._write_s),
            **record,
            **(dict(situation=self.situation()) if self.v3 else {}),
        )

    def settings(self) -> dict:
        """How the slow level was set up, for results.json."""
        return dict(
            scene=self.scene,
            program=self.source,
            stable=self.stable,
            p_min=self.p_min,
            unseen=self.unseen,
            revoke=self.revoke,
            patience=self.patience,
            max_recoveries=self.max_recoveries,
            give_up=self.give_up,
            **(dict(aim=True) if self.aim else {}),  # absent where it is off, as it was before there was one
            **(dict(resets=self.resets) if self.resets else {}),
            **(
                dict(
                    v3=True,
                    inspect=self.inspect,
                    gate=self.gate,
                    attribute_rule=self.attribute_rule,
                    witness=self.witness,
                    gate_offset=GATE_OFFSET,
                    gate_axis=GATE_AXIS,
                )
                if self.v3
                else {}
            ),
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

    `then_home` (a reset: Supervisor, rule e): after a preamble the joints go on to the home pose as well, so that the
    move always ends where an episode starts, jaws open.

    act() gives the next command (8 numbers: 7 joint targets and the gripper); done says the move is over. `expert`
    and `home` are for tests (by default a new ScriptedExpert(env, seed=0) and env.HOME_Q).
    """

    def __init__(self, env, expert=None, home=None, steps: int = HOME_STEPS, then_home: bool = False):
        if expert is None:
            from .expert import ScriptedExpert  # the simulator: imported only when a recovery is made

            expert = ScriptedExpert(env, seed=0)
        expert.take_over()
        self.env, self.expert, self.made, self.path = env, expert, 0, None
        self._home, self._steps, self._line_from, self.then_home = home, steps, 0, then_home
        if not expert.preamble:
            self._line()

    def _line(self):
        """The straight line in joint space from where the arm is to the home pose, jaws open."""
        home = self._home
        if home is None:
            from .env import HOME_Q

            home = HOME_Q
        start, home = np.asarray(self.env.data.qpos[:7], dtype=float), np.asarray(home, dtype=float)
        steps = self._steps
        self.path = [
            np.append(start + (home - start) * (k + 1) / steps, JAWS_OPEN).astype(np.float32) for k in range(steps)
        ]
        self._line_from = self.made

    def _preamble_over(self) -> bool:
        return not self.expert.preamble or self.made >= PREAMBLE_LIMIT

    @property
    def done(self) -> bool:
        if self.path is not None:
            return self.made - self._line_from >= len(self.path)
        return self._preamble_over() and not self.then_home  # then_home: the line to the home pose is still to come

    def act(self) -> np.ndarray:
        assert not self.done, "the recovery is over"
        if self.path is None and self._preamble_over():  # then_home: the preamble is made, now home
            self._line()
        self.made += 1
        return self.expert.act() if self.path is None else self.path[self.made - self._line_from - 1]


# ------------------------------------------------------------------------------------------------------ summary
def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def summary(episodes: list[dict]) -> dict:
    """results.json's v2 section, from eval.py's per-episode records (each with "success" and "v2", SlowLevel.record):
    successes; episodes in which Complete was declared, false completions among them, and those that ended on one
    (records written before that was kept count every false completion); failures by the reason of the last decision;
    programs with the right meaning (of those that have a task's program to compare with); recoveries (asked for,
    episodes with one, successes among those); the mean hold steps; per check of the memory
    against the simulator, the mean of its share of looks right and the number of episodes in which it was ever wrong,
    and per FALSE_CHECKS the number of episodes in which it made that error; the cuts the memory unmarked (in all, and
    the episodes with one) and the verdicts withdrawn; the median latencies of the server's see and write; the ledgers'
    attempts by outcome, summed over the episodes, and repeated_targets, the episodes in which one part was the target
    of more than one attempt. Records written before memory_false, revocations and verdicts_withdrawn count as none."""
    rows = [(bool(e["success"]), e["v2"]) for e in episodes if e.get("v2") is not None]
    attempts = [entry for _, r in rows for entry in r.get("ledger", [])]
    targeted = [Counter(entry["part"] for entry in r.get("ledger", []) if entry["part"] is not None) for _, r in rows]
    compared = [r["program_meaning"] for _, r in rows if r["program_meaning"] is not None]
    rescued = [ok for ok, r in rows if r["recoveries"]]
    reset = [(ok, r) for ok, r in rows if r.get("resets")]  # rule e, in runs that have it
    failures = Counter(r["failure"] for _, r in rows if r["failure"] is not None)
    see = [r["see_ms"] for _, r in rows if r["see_ms"] is not None]
    write = [r["write_ms"] for _, r in rows if r["write_ms"] is not None]
    revoked = [r.get("revocations", 0) for _, r in rows]
    return dict(
        episodes=len(rows),
        successes=sum(ok for ok, _ in rows),
        declared_complete=sum(r["declared_complete"] is not None for _, r in rows),
        false_complete=sum(bool(r["false_complete"]) for _, r in rows),
        ended_false_complete=sum(bool(r.get("ended_false_complete", r["false_complete"])) for _, r in rows),
        failures=dict(sorted(failures.items())),
        programs_right_meaning=sum(compared),
        programs_compared=len(compared),
        recoveries=dict(asked=sum(r["recoveries"] for _, r in rows), episodes=len(rescued), successes=sum(rescued)),
        **(
            dict(
                resets=dict(
                    asked=sum(r["resets"] for _, r in reset), episodes=len(reset), successes=sum(ok for ok, _ in reset)
                )
            )
            if any("resets" in r for _, r in rows)
            else {}
        ),
        mean_hold_steps=_mean(r["hold_steps"] for _, r in rows),
        mean_inventory_hold_steps=_mean(r["inventory_hold_steps"] for _, r in rows),
        memory_right={name: _mean(r["memory_right"][name]["share"] for _, r in rows) for name in CHECKS},
        memory_ever_wrong={name: sum(r["memory_right"][name]["ever_wrong"] for _, r in rows) for name in CHECKS},
        memory_false={name: sum(bool(r.get("memory_false", {}).get(name)) for _, r in rows) for name in FALSE_CHECKS},
        revocations=dict(total=sum(revoked), episodes=sum(n > 0 for n in revoked)),
        verdicts_withdrawn=sum(r.get("verdicts_withdrawn", 0) for _, r in rows),
        see_ms=float(np.median(see)) if see else None,
        write_ms=float(np.median(write)) if write else None,
        ledger=dict(
            attempts=len(attempts), **{outcome: sum(e["outcome"] == outcome for e in attempts) for outcome in OUTCOMES}
        ),
        repeated_targets=sum(any(n > 1 for n in parts.values()) for parts in targeted),
    )


def summary_v3(episodes: list[dict]) -> dict:
    """results.json's v3 section (tutorial/architecture-v3.md, "What is reported"), from eval.py's per-episode records
    of --slow_level v3: in all, and by situation, by situation and mode, by the shoots on the plant and by the cuts
    meant: episodes, successes, wrong cuts (a shoot not meant, or out of order, without damage) and the episodes cut out
    of order, damage, false completions, Complete said before any sub-goal (where nothing was to be cut, and where
    something was: the reverse error), the home pose reached on a verdict, looks asked and their turn, looks that ran
    out of their bound, episodes that ended holding for an attribute, vetoes of the gate (episodes and in all) and
    resets, and the control steps per solved episode."""
    rows = [(e, e["v2"]) for e in episodes if e.get("v2") is not None and "v3" in e["v2"]]

    def counts(group):
        solved = [e for e, _ in group if e["success"]]
        return dict(
            episodes=len(group),
            successes=len(solved),
            wrong_cuts=sum(bool(e["wrong_object"]) and e.get("damaged") is None for e, _ in group),
            out_of_order=sum(bool(e.get("out_of_order")) for e, _ in group),
            damaged=sum(e.get("damaged") is not None for e, _ in group),
            false_complete=sum(bool(r["false_complete"]) for _, r in group),
            complete_without_action=sum(
                bool(r["v3"]["complete_without_action"]) and r["situation"]["cuts_meant"] == 0 for _, r in group
            ),
            complete_without_action_wrongly=sum(
                bool(r["v3"]["complete_without_action"]) and r["situation"]["cuts_meant"] > 0 for _, r in group
            ),
            home_reached=sum(r["v3"]["home_reached"] is not None for _, r in group),
            looks_asked=sum(r["v3"]["looks_asked"] for _, r in group),
            look_turn=round(sum(r["v3"]["look_turn"] for _, r in group), 4),
            look_failed=sum(r["v3"]["look_failed"] is not None for _, r in group),
            ended_holding_for_attribute=sum(bool(r["v3"]["ended_holding_for_attribute"]) for _, r in group),
            veto_episodes=sum(bool(r["v3"]["vetoes"]) for _, r in group),
            vetoes=sum(len(r["v3"]["vetoes"]) for _, r in group),
            resets=sum(r.get("resets", 0) for _, r in group),
            mean_steps_solved=_mean(e["steps"] for e in solved),
        )

    def by(key):
        groups: dict = {}
        for row in rows:
            groups.setdefault(str(key(row[1]["situation"])), []).append(row)
        return {name: counts(group) for name, group in sorted(groups.items())}

    return dict(
        **counts(rows),
        by_situation=by(lambda s: s["situation"]),
        by_mode=by(lambda s: f"{s['situation']}:{s['mode']}"),
        by_shoots=by(lambda s: s["shoots"]),
        by_cuts_meant=by(lambda s: s["cuts_meant"]),
    )
