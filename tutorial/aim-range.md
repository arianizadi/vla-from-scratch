# The aim range: motor drills for the arm

Every task in this repo so far teaches arm control only as a side effect of one job: pick up the named object, cut
the side shoots. On the full pruning task (`prune`, lab notebook step 8) the small policy prunes about 80 plants in
100 on two seeds of three, and what it gets wrong is mostly not precision but speed: it runs out of time, and its
successful episodes take about 640 control steps where the scripted expert needs 515. A human who wants to aim faster
and more precisely in a game trains on an aim trainer: short drills that do nothing but move the cursor to a target,
follow it, flick between targets and stop on them, apart from any game. This document describes the same idea for the
arm (`sim/pickup/aim.py`) and the experiments that test whether it helps. It covers the design; measured results go in
the [lab notebook](lab-notebook.md).

The drills are also meant for the fast half of a two-level policy ([reasoning VLA](reasoning-vla.md)): a slow
reasoner names a sub-goal ("go there", "press that button", "cut here") and a fast controller carries it out. So every
drill's instruction names the drill and the target, in the form a reasoner would hand over, and other targets of other
colours stand around, so that the instruction has to be read rather than ignored.

## The drills

Seven drills on the bare table, built from primitives (glowing spheres, thin pins, two push buttons on housings, two
small cubes). Each is a short episode with a success rule and a time budget (control steps at 20 Hz):

| Task | Instruction | Success | Budget | Expert |
|---|---|---|---|---|
| `aim_touch` | "touch the red dot" | the gripper's centre within 15 mm of the named dot (one of three in view, anywhere in the workspace) for 3 steps in a row | 160 | 500/500, 18 steps |
| `aim_pose` | "touch the red dot along its pin" | the same, with the gripper pointing along the dot's pin (within 20 degrees); pins range from straight down to level, roughly across the line from the robot's base | 200 | 500/500, 44 steps |
| `aim_hold` | "stop on the red dot" | within 8 mm for 15 steps in a row: a precise stop, no overshoot | 200 | 500/500, 40 steps |
| `aim_track` | "follow the red dot" | the dot moves on a smooth sway (two sine waves per axis) or a jerky path (straight runs that change direction at random), top speed 2-5 cm/s; after 30 steps to catch it, within 30 mm on at least 80% of the steps | 140-200 (the path) | 500/500, 171 steps |
| `aim_flick` | "touch the red dots one after another" | 3-5 dots light up one at a time, each where the last was hit (within 20 mm), at least 12 cm apart | 320 | 500/500, 62 steps |
| `aim_press` | "press the blue button" | of two buttons placed anywhere on the table, the named one pushed past 3 mm for 5 steps in a row; pushing the other is a failure | 240 | 500/500, 42 steps |
| `aim_pinch` | "pinch the small yellow cube" | of two 20 mm cubes, the named one lifted 3 cm for 5 steps; lifting the other is a failure | 240 | 498/500, 53 steps |

The expert column: the scripted expert's successes on 500 evaluation layouts per drill, from random start poses, and
the mean length of its episodes. `aim_mix` draws one drill per episode (the training data) and `aim_each` takes them
in turn by seed (evaluation: equal numbers of each). Every episode starts from a random gripper pose above the table
when collected or evaluated with `--random_start`, which every drill dataset and evaluation here uses: from one fixed
start, the arm's own position would give the goal away after a few steps.

`eval.py` reports the drills one by one: success, wrong targets (a button or cube not named), the mean steps of
successful episodes, and how close the gripper came (the distance to the dot at the end, or the mean tracking error, or
the angle to the pin).

**What the drills are for, one by one.** Touch is "go there". Pose is "go there, arriving this way", the move a cut
needs (the jaws come in level from the side). Hold is the precise stop before closing the jaws. Track is a target that
keeps moving, as a shoot sways. Flick is several moves in a row without a pause between them: where the pruning
policy loses its time, between one cut and the next. Press is the turntable's button. Pinch is a small grasp, where a
few millimetres decide.

**How the expert solves them.** It is the pruning expert's waypoint follower: straight lines at a steady speed, with a
speed factor drawn per episode (0.7-1.3 times 0.4 m/s), so the demonstrations show the same drill done faster and
slower. Two details mattered. Held still under its position servos the arm sags about 8 mm below its set-point, which
fails the precise stop: the drills' expert integrates what is left of the error while its set-point stands still (13
of 50 stops failed without it, none of 500 with it). And a pin pointing back toward the robot, or out away from it,
asks for a hand pose the arm cannot reach: those directions are left out of the drill (5 of 50 failed before).

**Nothing else changes.** The drills' objects are built only into the aim tasks' model. For every other task the
compiled model, the state after reset and the expert's actions and joint positions at every step of whole episodes
are identical with and without the aim range (23 tasks; 46 expert episodes over ten of them, from the home pose and
from random starts).

## The dataset

`aim_mix_8k_rs_tail10`: 8000 demonstrations of `aim_mix` from random starts, with a tail of 10 frames after each
success (as the pruning data), 464,537 frames; the expert succeeded in 8797 of 8800 attempts.

```
sim/pickup/scripts/make_dataset.sh aim_mix 8000 aim_mix_8k_rs_tail10 --random_start --tail_steps 10
```

Why this size: the drills are short (18-171 steps), so the number of episodes sets how many layouts each drill is seen
in, and the frames set what it costs. Drawn equally, the tracking drill (171 steps) would take 39% of the frames and
touching (18 steps) 4%; `aim.MIX_WEIGHTS` draws touching four times as often as tracking, which gives every drill 11-19%
of the frames and each drill 491-1840 layouts. That is in the range where the single pick-up skills of earlier
chapters generalised to new layouts (1000-4000 demonstrations), and at 465k frames the dataset is under half the size
of the pruning dataset it is mixed with (2000 demonstrations, 1.10M frames), so it can be seen many times over in a
fraction of the pruning run's budget. Collecting it took six minutes.

## The experiments

The policy is the step-8 baseline's (`dp_vit_lang_reactive`: one frame per camera, the joint state, the instruction;
no history), and so are its pruning demonstrations (`prune_2k_rs_tail10`) and its 20M samples of pruning. The
baseline's final averaged weights prune 79, 56 and 78 plants of 100 on three seeds. Both designs below see the same
20M samples of pruning and 6.7M of drills (a third as many again), so they differ only in when the drills come:

- **(a) Co-training** (`..._prune2k_aim8k_co27m`): every batch draws 75% pruning, 25% drills, for 26.7M samples. Why
  25%: enough drill samples (6.7M, about 14 passes over the drill data) for the drills to be learned, while pruning
  keeps the baseline's 20M samples; more would take samples from the task being measured, fewer would leave the drills
  barely seen.
- **(b) Pretraining, then fine-tuning** (`..._aim8k_pre6700k`, then `..._prune2k_from_aim8k_20m`): 6.7M samples of
  drills alone, then the trained weights fine-tuned on pruning alone for 20M samples with a fresh learning-rate
  schedule (`--model.resume_from_checkpoint <checkpoint> --model.resume_weights_only True`).

States and actions are normalised with statistics merged over both datasets in all three runs (`train.sh` with
`prune_2k_rs_tail10+aim_mix_8k_rs_tail10` and `PICKUP_DATA_WEIGHTS=0.75,0.25` for (a), `PICKUP_STATS_DATASETS` for
(b)): a policy pretrained under one normalisation and fine-tuned under another would start from actions that mean
something else. Checkpoint k of either design has seen about 2k M samples of pruning, as checkpoint k of the
baseline has, so the three can be compared checkpoint by checkpoint.

**What is measured.** On `prune` (100 layouts per checkpoint, as the baseline): success, flower or stem damaged, out of
time, and the mean steps of successful episodes (is it faster than 640?). On `aim_each` (140 episodes per checkpoint, 20
per drill, random starts): each drill's success, time and accuracy, during pretraining (are the drills learned?) and
after fine-tuning (are they kept?). And on the test situations of [test-situations.md](test-situations.md) (wind, a
bumped turntable, a knocked camera, ...), the final checkpoints against the baseline's: does general arm control make
the policy steadier where its pruning demonstrations never went?
