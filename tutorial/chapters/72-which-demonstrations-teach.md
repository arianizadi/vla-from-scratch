---
title: Which Demonstrations Teach What
part: 7 Data
order: 72
summary: Decide which windows of your demonstrations need the scene, choose collect.py options and window weighting accordingly, and read from a run's checkpoints whether the change made the policy look earlier, end higher, or neither.
status: draft
---

# Which Demonstrations Teach What

A policy trained by imitation learns only what its demonstrations make necessary. Here the same
policy, on the same number of episodes and the same training budget, starts to use the scene up to a
million samples earlier and ends between 84% and 98%, depending only on what the episodes contain.

## The idea

A **window** is one training example: the observations at one frame of a demonstration plus the 15
actions that follow (the **chunk** the policy learns to predict;
[Demonstrations](70-demonstrations.md)). The policy here, `dp_vit_lang_reactive` (48.9M
parameters; [Language conditioning](51-language-conditioning.md)), sees per window one image from
each camera, the instruction as 32 tokens and the current joint state, and not its previous action
([History and shortcuts](52-history-and-shortcuts.md)).

The rule: **a window that can be imitated without looking at the scene teaches nothing about
looking.** Its loss can be brought down from the joint state alone, so it gives no reason to read
the image.

Worked example. A plain `pick_target` demonstration starts at the home pose (`HOME_Q` plus 0.03 rad
of noise per joint), three objects on the table, one named; 263,289 windows over 4000 episodes is
65.8 frames per episode (arithmetic on the notebook's datasets table).

- **The window at frame 0.** The arm is where it is in every other layout. Which way the chunk goes
  depends on which object is named and where it lies: this window needs the instruction and the
  image.
- **A window at frame 20.** The arm has moved in a straight line from that same home pose toward
  the point above the named object (`ScriptedExpert._plan` in
  [`expert.py`](../../sim/pickup/expert.py)). The joint state alone now says which line it is on,
  and so roughly where it ends: the choice of object is already in the state.

Only the first few windows of each episode force the policy to look in order to choose the object;
after that the image is needed only for the last centimetres. Every option below changes that share.

The **onset** of looking is the point in training where the policy starts to use the scene, read
from two numbers per checkpoint ([Measuring failure](92-measuring-failure.md)): **closed at the
named object** (the object nearest the gripper when it first closed was the named one; chance 33%)
and the **slope**: a least-squares fit of where the gripper first closed against where the named
object was, x and y pooled, over the episodes in which it closed (`close_slope` in
[`summarize.py`](../../sim/pickup/summarize.py); 1: it closes where the object is; 0: where it
closes does not depend on the scene). The first is a share of all episodes, so one in which the
gripper never closes counts against it; the slope ignores such episodes. Both rise before
success does ([How much data and training](71-how-much-data-and-training.md)).

## How it works

Four options of [`collect.py`](../../sim/pickup/collect.py), passed through `make_dataset.sh`, and
one tool that reweights a finished dataset. Windows are from the notebook's datasets table, the
per-episode figures arithmetic on them:

| Change | Dataset | Windows (4000 episodes) |
|---|---|---|
| none | `pick_target_4k` | 263,289 (65.8 per episode) |
| `--targets_per_layout 3` | `pick_target_cf4k` | 263,552 |
| `--random_start` | `pick_target_4k_rs` | 241,318 (60.3) |
| `--disturb_prob 0.5` | `pick_target_4k_recover` | 293,541 (73.4) |
| `--random_start --disturb_prob 0.5` | `pick_target_4k_rs_recover` | 275,550 (68.9) |
| `boost_early_windows.py --max_frame 7 --repeat 5` | `pick_target_4k_early`, `pick_target_4k_rs_recover_early` | 423,289, 435,550 |
| `--hold_steps 1` | `pick_target_4k_hold` | 267,075 |

**Every object named.** `main` makes one job `(seed, i)` per target index `i` below
`--targets_per_layout`. `PickupEnv.reset` draws the seed's own target and then replaces it, so the
scene is identical in all three and the instruction is the only difference between the inputs of
the three windows at frame 0. Episodes are sorted by `(seed, target_index)` and the first `--episodes`
successful ones kept: 4000 episodes are about 1333 layouts, not 4000.

**Random start poses.** `random_start` in `expert.py` draws a gripper position inside the box
`START_X`, `START_Y`, `START_Z` and a yaw, from its own random stream (`[seed, 13]`, so the layout
is unchanged), solves inverse kinematics (the joint angles that put the gripper there) and places
the arm at rest. A joint
state in mid-reach is now consistent with many start points and so with many goals: every window,
not only the first, needs the image.

**Recoveries.** With `--disturb_prob` above 0 the expert has `reactive=True`: before every action
it checks the named object. A move of more than 15 mm but at most 40 mm (`TRACK_LIMIT`), outside
the grasp, shifts the plan with it; a larger move, one during the grasp, or fingers that met on
nothing during the lift make it plan again. The moves here are 5 to 10 cm, so each one causes a new
plan. Each episode draws from `[seed, 7]` whether it is disturbed, whether during the reach
(control step 5 to 29) or as the gripper closes, and how far. If
[`disturb.py`](../../sim/pickup/disturb.py) finds no free place in 64 random directions, the episode
runs undisturbed (`"disturbed": "none"`). With failed episodes dropped too, `--disturb_prob 0.5`
gave 1505 disturbed episodes of 4000, and 1456 with random starts, not 2000. The windows after a
move are the new ones: the arm heads for the old place, the expert's actions turn toward the new
one, and only the image says why.

**Early windows repeated.** [`boost_early_windows.py`](../../sim/pickup/boost_early_windows.py)
collects nothing. It links the original shards into a new directory and adds `--repeat` copies of
every window whose `frame_index` is at most `--max_frame`; `stats.json` is copied, so normalisation
is unchanged. On `pick_target_4k` that adds 160,000 windows (32,000 early windows, 8 per episode,
times 5), and the early windows go from about 12% of the data to 45% (arithmetic). The price, at a
fixed 5M samples: every other window is seen 11.8 times on average instead of 19.0; for
`pick_target_4k_rs_recover`, 11.5 instead of 18.1.

**The hold frame.** For the first `--hold_steps` frames `_worker` records the robot at rest under
`env.last_action`, the hold command: each episode gains a frame whose action is "stay", so for a
policy with history the first previous action is what serving provides, not the expert's first
command, which points at the target. It did not make such a policy look
([History and shortcuts](52-history-and-shortcuts.md)) and is not compared below.

## In this repository

From the repository root:

```bash
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_cf4k --targets_per_layout 3
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k_rs --random_start
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k_recover --disturb_prob 0.5
sim/pickup/scripts/make_dataset.sh pick_target 4000 pick_target_4k_rs_recover --random_start --disturb_prob 0.5

for base in pick_target_4k pick_target_4k_rs_recover; do
  .venv/bin/python -m sim.pickup.boost_early_windows --shards data/preprocessed/$base/shards \
      --out data/preprocessed/${base}_early/shards --max_frame 7 --repeat 5
done
```

The fifth line is the recipe [README](../../README.md) recommends. README calls the result
`pick_target_4k`, which here is the plain dataset, and `make_dataset.sh` replaces a dataset of the
same name, so keep the names used here. To see what a dataset holds before training on it:

```bash
.venv/bin/python - <<'EOF'
import json
from collections import Counter

name = "pick_target_4k_rs_recover"
info = json.load(open(f"data/lerobot/{name}/meta/info.json"))
print({k: info[k] for k in ("total_episodes", "total_frames", "disturbed_episodes",
                            "random_start", "disturb_prob", "targets_per_layout", "expert_success_rate")})
episodes = [json.loads(line) for line in open(f"data/lerobot/{name}/meta/episodes.jsonl")]
print(Counter(e["disturbed"] for e in episodes))
print("episodes with at least one new plan:", sum(e["replans"] > 0 for e in episodes))
EOF
```

Train each like the plain run, evaluating checkpoints as they appear:

```bash
PICKUP_NUM_CHECKPOINTS=10 sim/pickup/scripts/train_resilient.sh \
    dp_vit_lang_reactive pick_target_4k_rs my_rs 5000000 &
PICKUP_WATCH=1 sim/pickup/scripts/eval_checkpoints.sh experiments/pickup/my_rs pick_target 100 &
```

Once evaluations have been written (during the run or after it), print the table:

```bash
.venv/bin/python -m sim.pickup.summarize --task pick_target --runs my_rs
```

[`dp_vit_lang_reactive.yaml`](../../sim/pickup/configs/dp_vit_lang_reactive.yaml) switches
averaged weights on itself, and `eval_checkpoints.sh` evaluates the trained and the averaged weights
of each checkpoint.

## What we measured

From the [lab notebook](../lab-notebook.md), "After the position fix (machine B)", table "What the
demonstrations contain": `dp_vit_lang_reactive`, 4000 episodes, 5M samples, the position table
fixed ([Positions](12-positions.md)), one training seed per run, 100 unseen layouts per evaluation,
machine B (L40S GPUs, 128 CPU cores). Cells are trained / averaged weights (a running average of
the weights, saved next to the trained ones and evaluated separately); a dash where a run saved
none.

**Success at fixed points of training:**

| Demonstrations | 0.5M | 1.0M | 1.5M | 2.0M | 3.0M | 4.0M | 5.0M |
|---|---|---|---|---|---|---|---|
| one instruction per layout | 0% / - | 8% / - | 58% / - | 40% / - | 90% / - | 94% / - | 93% / - |
| early windows repeated | 1% / - | 5% / - | 76% / - | 73% / - | 96% / - | 96% / - | 96% / - |
| every object named | 1% / 2% | 41% / 38% | 74% / 80% | 73% / 84% | 91% / 89% | 89% / 91% | 91% / 89% |
| recoveries | 0% / 12% | 38% / 79% | 79% / 80% | 85% / 87% | 83% / 85% | 86% / 88% | 89% / 84% |
| random starts | 0% / 1% | 56% / 76% | 57% / 79% | 71% / 84% | 94% / 98% | 98% / 98% | 98% / 98% |
| random starts and recoveries | 12% / 54% | 68% / 86% | 86% / 91% | 93% / 84% | 90% / 93% | 93% / 89% | 94% / 94% |
| the same, early windows repeated | 5% / 13% | 58% / 60% | 49% / 82% | 84% / 84% | 80% / 83% | 85% / 87% | 87% / 86% |

**Closed at the named object, slope** (trained weights, or averaged where only those exist):

| Demonstrations | 0.5M | 1.0M | 1.5M | 2.0M | 3.0M |
|---|---|---|---|---|---|
| one instruction per layout | 36%, 0.03 | 65%, 0.58 | 98%, 0.94 | 98%, 0.97 | 99%, 0.96 |
| early windows repeated | 58%, 0.49 | 99%, 0.93 | 100%, 1.00 | 100%, 1.02 | 99%, 1.01 |
| every object named | 30%, -0.04 | 98%, 1.02 | 99%, 0.98 | 99%, 0.99 | 100%, 1.00 |
| recoveries | 33%, 0.11 | 81%, 0.96 | 100%, 0.97 | 100%, 1.00 | 100%, 0.99 |
| random starts | 46%, 0.18 | 91%, 0.81 | 96%, 0.94 | 87%, 0.84 | 98%, 0.95 |
| random starts and recoveries | 50%, 1.00 | 100%, 1.02 | 100%, 1.00 | 100%, 1.00 | 100%, 0.99 |
| the same, early windows repeated | 56%, 0.56 | 93%, 0.95 | 97%, 1.00 | 100%, 0.98 | 100%, 1.00 |

Figures by object below are from the notebook's table pooled over each run's last three
checkpoints, both sets of weights.

<!-- interactive: onset -->
*Figure: success, closed at the named object and the slope against training samples for the seven
runs side by side. The plain run's slope is 0.58 after 1.0M samples; most others are near 1 by
then, random starts with recoveries already at 0.5M.*

**How much noise to expect.** The plain recipe trained again with other seeds and the same fixed
table closed at the named object 83% / 72% of the time after 1.0M samples (slope 0.77 / 0.72) and
ended at 96% / 96% (notebook, "A second seed, with and without the fix"); the first run had 65%
(0.58) and 93%. Two seeds are not a noise estimate, but the second seed moved "closed at the named
object" at 1.0M by 18 points and the end by 3; read smaller differences between single runs as
noise.

**Which options moved the onset.**

- *Clearly earlier:* early windows repeated (58%, slope 0.49 already at 0.5M; 99% at 1.0M), random
  starts with recoveries (slope 1.00 at 0.5M against 0.03; 100% at 1.0M), and the same with early
  windows repeated (56%, 0.56 at 0.5M), though its slope there is behind the run without the
  copies. Random starts with recoveries has a slope of 1.00 at 0.5M while it closes at the named
  object in only half the episodes: the two numbers count different episodes (see above).
- *Earlier by less:* every object named (98%, 1.02 at 1.0M) and random starts (91%, 0.81), above
  both plain seeds at 1.0M; at 0.5M, every object named is no further on than the plain run.
- *Earlier, less sharply on one measure:* recoveries alone. At 1.0M its slope is 0.96 (plain seeds
  0.58 and 0.77) and it succeeds 38% / 79% (plain 8%, 2% / 6%), but it closes at the named object
  81% of the time, inside the plain seeds' 65-83%.

The more windows cannot be answered without looking, the sooner the policy looks (field notes,
'What the demonstrations contain' and 'Looking at the scene arrives late, and all at once').

Looking early is not succeeding early: with early windows repeated the policy closed at the named
object in 99% of episodes at 1.0M and succeeded in 5%.

**Which option moved the final score.** Random starts gave the best end, 98% / 98% (by object
100%, 100%, 99%, 92%: 588 of 600 episodes);
early windows repeated ended at 96% against 93%. Both are above the plain run, but the plain recipe
with other seeds also ended at 96%: against that, random starts gained 2 points and early windows
none, one seed each. Two changes moved it down: recoveries alone (89% / 84%, the can at 77%), and early windows on
top of random starts and recoveries (87% / 86% against 94% / 94%, the can at 68% against 97%).

**Why one tool helped one dataset and hurt another.** In the plain dataset the early windows are
the only ones that need the scene; the later windows, which the arm's state answers, lose little by
being seen less often. With random starts and recoveries every window needs the scene already; the
copies add nothing the policy lacked and take samples from the later windows, the ones with the
corrections.
Plausible, not shown: one seed per run, and a 7-point gap measured once.

**Every object named.** It made the policy look early and ended no higher (91% / 89%), with the T
block at 64% (88/138). The plain data had no instruction
problem to fix: with one instruction per layout the policy closed at the named object in 99% of
episodes at 3.0M.

**What the table does not show.** Most of what these options buy: with the object moved at control
step 20, clean demonstrations give 0 of 97, random starts and recoveries 89-94% under every
disturbance ([Case study: self-correction](103-case-self-correction.md); the probes:
[What a policy looks at](95-what-a-policy-looks-at.md)).

## Designing the demonstrations for your own task

1. **Walk through one demonstration window by window**: could the next chunk be predicted from the
   joint state (and previous action, if given) alone? If only a few percent could not, expect a
   late onset.
2. **Vary what gives the answer away.** Here it was the start pose; `--random_start` cost nothing on
   the unchanged task. Elsewhere it may be a fixed object order or a constant approach height.
3. **Put the behaviour you want into the data**: an expert that plans again, and disturbances in a
   share of episodes. Check the count in `meta/info.json`, not the probability you asked for.
4. **Reweight only clean data.** `boost_early_windows.py` needs no new collection and helped where
   the early windows were the only ones that needed the scene. On varied data it lowered the score
   here (87% / 86% against 94% / 94%, one seed); if you stack it, compare against the same data
   without the copies.
5. **Know how many layouts you have.** Naming every object meant a third as many layouts and ended
   no higher; do it only if the policy is shown not to follow the instruction.
6. **Compare at the same checkpoints**, both sets of weights, against a second seed of the
   baseline, and under the disturbances the data was meant to teach.

## Where it goes wrong

| What you see | Why | What to do |
|---|---|---|
| Fewer disturbed episodes than `--disturb_prob` asks for | no free place for the move, or the episode failed and was dropped | read `disturbed_episodes` in `meta/info.json`; raise the share if needed |
| A new dataset "wins" by 3 points | one seed per run; the plain recipe ended at 93% and 96% | run a second seed of the baseline |
| Looking early, succeeding late | the slope rises before the miss shrinks | read the miss in `summarize.py`; compare at the end |
| The new policy wants other inference settings | two flow steps equalled ten on four of these seven policies and cost 12-19 points on the three with early windows repeated or every object named (77% against 96%, 77% against 89%, 74% against 86%; notebook, "What the same policies depend on" against the Standard column of the table before it) | sweep `--flow_steps` again on each new policy |
| A boosted dataset fails to load, or its windows no longer match its statistics | its shards are symlinks to the base dataset, which `make_dataset.sh` deletes and rebuilds; `boost_early_windows.py` also refuses an existing `--out` | rebuild the base, delete the boosted folder, run `boost_early_windows.py` again |

## Background and further reading

Background, not measured here:

- Ross, Gordon and Bagnell, *A Reduction of Imitation Learning and Structured Prediction to
  No-Regret Online Learning*, AISTATS 2011 ([arXiv:1011.0686](https://arxiv.org/abs/1011.0686)).
  DAgger: the expert labels the states the learner reaches; not implemented here.
- Laskey, Lee, Fox, Dragan and Goldberg, *DART: Noise Injection for Robust Imitation Learning*, 2017
  ([arXiv:1703.09327](https://arxiv.org/abs/1703.09327)). Perturbs the supervisor so the data shows
  recoveries; `collect.py` perturbs the scene instead.
- de Haan, Jayaraman and Levine, *Causal Confusion in Imitation Learning*, 2019
  ([arXiv:1905.11979](https://arxiv.org/abs/1905.11979)). A cloned policy can rely on a cue that
  predicts the expert's action without causing it; here, the arm's own position.
- Lin et al., *Data Scaling Laws in Imitation Learning for Robotic Manipulation*, 2024
  ([arXiv:2410.18647](https://arxiv.org/abs/2410.18647)). Generalisation grew with the number of
  environments and objects more than with the number of demonstrations.

## Check yourself

1. Build 50 episodes with and without `--random_start` and compare their total frames. Why might
   episodes from a random start be shorter (60.3 against 65.8 frames per episode in the notebook)?

   ```bash
   sim/pickup/scripts/make_dataset.sh pick_target 50 pt50
   sim/pickup/scripts/make_dataset.sh pick_target 50 pt50_rs --random_start
   grep total_frames data/lerobot/pt50/meta/info.json data/lerobot/pt50_rs/meta/info.json
   ```

2. Predict the share `boost_early_windows.py` will print for `pt50` with `--max_frame 3 --repeat 10`,
   then run it (`--shards data/preprocessed/pt50/shards --out data/preprocessed/pt50_early/shards`).
3. In a dataset made with `--disturb_prob 0.5`, pick one episode marked `"reach"` and one marked
   `"grasp"` in `meta/episodes.jsonl`. Which of their windows could not be answered from the joint
   state alone?

   ```bash
   sim/pickup/scripts/make_dataset.sh pick_target 50 pt50_recover --disturb_prob 0.5
   .venv/bin/python -c "import json; e=[json.loads(l) for l in open('data/lerobot/pt50_recover/meta/episodes.jsonl')]; print([(x['episode_index'], x['disturbed'], x['length']) for x in e if x['disturbed']!='none'][:5])"
   ```
