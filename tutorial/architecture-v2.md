# Architecture v2: small eyes, a scene memory, and a reasoner that reads text

The two-level reasoning policy of [reasoning-vla.md](reasoning-vla.md) solves 133 of 200 reasoning episodes when
the instruction is worded as in training and 66 of 200 when it is not. This document works out from the result files
what limits it, decides whether a new architecture is the right answer, and specifies one. It is a design: nothing in
it is built. A number is a measured one where a run is named; predictions are marked as predictions.

## In short

**The idea.** Split the slow level (the part that reasons) into three parts, each trained on what its job needs.
Small *eyes*, a vision transformer in which every weight is trained, turn the camera images into a typed description
of the scene: how many shoots are on the plant, the flower's colour, what the label says. A *scene memory* without
weights keeps what was seen and what was cut. A pretrained language model then reasons over the instruction and that
description as text, and never sees a pixel. The fast level that moves the arm stays as it is.

**Why.** The reasoner now in use writes "2 shoots" for 105 of 108 three-shoot plants, and that one error is in 56 of
the 67 failures on training instructions. Its vision tower is frozen and hands it 49 tokens per image; models here
whose vision is trained read the count from the same images. On held-out instructions its first sub-goal is wrong in
126 of 200 episodes: it restates 107 of the 200 instructions correctly, where the same language model, not fine-tuned
and shown eight worked examples, restated 32 of 40 in a probe, and it works out new combinations wrongly even when it
has restated them correctly.

**The parts**, by what they are in general:

1. a scene-state estimator: a small supervised vision network whose outputs are typed fields, not text;
2. a scene memory: explicit state, updated by rules;
3. a text-only reasoner: a pretrained language model that restates the instruction, and a resolver that computes
   what can be computed;
4. an outcome check: the plan is made again whenever the scene memory changes;
5. a reactive low-level policy that reads a sub-goal (unchanged).

**The first test needs no training** (E0 below): hand the existing reasoner the true count and measure again.

## What limits the reasoning policy now

Totals are in the [lab notebook](lab-notebook.md). The breakdowns in this section were computed for this document
from the per-episode records of the result files named (`results/pickup/<run>/<evaluation>/results.json`) and are not
in the notebook yet. Short names for the evaluations used:

| Name | Evaluation | Episodes |
|---|---|---|
| T | the two-level policy, training instructions: `dp_vit_lang_reactive_subgoal_reason8k_20m_seed43/final_reason_mix_two_level_ema` (reasoner `qwen_reasoner_mix8k_600k`) | 200 |
| H | the same, held-out instructions: `.../final_reason_mix_heldout_two_level_ema` | 200 |
| N | the reasoner's notes while the expert drives: `qwen_reasoner_mix8k_600k/reason_mix_notes/ckpt_<n>_raw` | 50 per checkpoint |
| F43, F42 | the fast level given the expert's sub-goals: `..._subgoal_reason8k_20m_seed43/reason_mix_subgoals_expert/ckpt_11_ema`, `..._subgoal_reason8k_20m/final_reason_mix_expert_ema` | 100, 200 |

### 1. The reasoner does not see how many shoots the plant has

In T the policy prunes 82 of 92 plants with two shoots and 51 of 108 with three.

| What the first note of an episode says (T) | Right |
|---|---|
| the instruction, restated (`ask`) | 200 of 200 |
| the flower's colour | 200 of 200 |
| the label's colour, in some note of a hidden-label episode | 45 of 45 |
| the number of shoots, plants with two | 92 of 92 |
| the number of shoots, plants with three | 3 of 108 (105 say "2 shoots") |

The number it writes does not depend on the plant. In N, at ten of the eleven checkpoints of the run it writes
"2 shoots" for at least 48 of the 50 plants (29 of them have three), and at checkpoint 5 it writes "3 shoots" for all
50. What follows from the count goes wrong with it:

| Shoots to cut (T) | Solved |
|---|---|
| the highest | 56 of 60 |
| the lowest | 67 of 82 |
| the middle one | 5 of 23 |
| two shoots, the middle one among them | 5 of 35 |

The inference follows the count the reasoner wrote, not the plant. At checkpoint 5 of N, where every plant is
"3 shoots", the `so` field of the first note is right for all 19 three-shoot plants with an ordinal or exclusion
instruction ("of 3, second from the top is the middle") and wrong for all 14 two-shoot ones (the same sentence, on a
plant with no middle shoot). At the final checkpoint, where every plant is "2 shoots", it is the other way round. So
on training instructions the reasoning is in place and a correct count is what is missing.

Of the 67 failures in T, 56 have the wrong count in their first note. Where the sub-goal the reasoner held was right
in at least 95% of the chunks of an episode, 122 of 140 episodes succeed; in the other 60, 11 do. Of the 47 episodes
that ran out of time, 27 had the reasoner write `stop` with a shoot still to cut: it had planned for a plant with two
shoots.

Why it cannot see the count (in this repository): `qwen_reasoner.yaml` trains the last 8 language-model layers
(`num_trainable_vlm_layers: 8`), and `VLMHFBackboneWrapper._train_only_last_layers` freezes everything else, the
vision tower included. The tower hands on 49 tokens per image, one per 32 x 32 pixels. The front camera
(`sim/pickup/env.py`: 45 degrees of view, about 0.9 m from the plant) shows about 3.4 mm per pixel at 224 pixels, so
a shoot, 6 mm thick and 7 to 9 cm long, is at most about 2 pixels by 25, and the plant covers about 2 by 3 of the 7
by 7 tokens (a calculation from the camera's position, not a measurement).

It is not that 224 pixels cannot show it. The one-model policy on the same backbone, whose training also carries the
action loss and ran four times as long (`vla_qwen_reason_reason8k_2500k`, final checkpoint, 50 episodes), has the
count of three-shoot plants right in 21 of 29 first notes and of two-shoot plants in 17 of 21. Our own 130M model,
whose vision transformer keeps 256 tokens per image and is trained with everything else
(`vla_custom_reason_reason8k_10m`), has three-shoot plants right in 24, 29 and 25 of 29 at its checkpoints 5 to 7
(that run is still training; these are not final numbers). And the small policies cut the right shoot from the same
images.

### 2. New wording and new combinations

In H the first sub-goal is right in 74 of 200 episodes. The policy succeeds in 51 of those 74 and in 15 of the other
126.

| Held-out instructions (H), first note | Instruction restated right (`ask`) | Sub-goal right |
|---|---|---|
| ordinal, new combination ("third from the top") or both new | 33 of 33 | 0 of 33 |
| ordinal, new wording only | 16 of 16 | 8 of 16 |
| conditional, new combination only | 17 of 17 | 17 of 17 |
| conditional, new wording (alone or with a new combination) | 8 of 40 | 23 of 40 |
| exclusion, new combination only (keep the middle one) | 12 of 12 | 9 of 12 |
| exclusion, new wording (alone or with a new combination) | 21 of 37 | 14 of 37 |
| hidden label, new wording ("check the tag in the pot ...") | 0 of 45 | 3 of 45 |

Two different failures. A new wording is not recognised: with "tag" or "card" for "label" the reasoner never treats
the task as a label task, and the label's colour appears in no note of those 45 episodes. And a new combination of
known words is restated correctly but worked out wrongly: "third from the top" becomes `ask: third from the top` and
then the wrong shoot, 33 times of 33. The inference was learned as a table from (request, count) to shoot, not as
counting. The reasoner was fine-tuned on 30 instruction strings (`reasoning.training_instructions`: 12 ordinal, 12
conditional, 4 exclusion, 2 for the label).

Two probes were run for this document with the language model of Qwen3-VL-2B as downloaded: no fine-tuning, no
images, on a CPU, 10 drawn instructions per kind and split, so the samples are small.

- *Restating.* Given eight worked examples that use training instructions only, the model was asked to restate an
  instruction in the notes' fixed form. With training instructions it restated 40 of 40 exactly; with held-out ones
  32 of 40 (ordinal 10 of 10, conditional 7, exclusion 8, hidden label 7), and a resolver in code, given the true
  plant, then named the right shoots for exactly those 72. Six of the eight misses have the right meaning in a form
  the resolver does not know ("the shoot the tag points to"; "unless red the lowest, else the highest"); two are wrong
  ("leave only the highest shoot and cut the rest" became "all but the lowest"). The fine-tuned reasoner restates 107
  of the 200 held-out instructions of H correctly, and none of the 45 about the label.
- *Answering in one step.* Given a correct description of the plant in words and eight worked examples, the model was
  asked for the shoots to cut, without writing any reasoning first: 33 of 40 right with training instructions, 28 of
  40 with held-out ones (ordinal 12 of 20, conditional 12 of 20, exclusion 18 of 20, hidden label 19 of 20).

So the pretrained model reads a wording it was never shown far better than the fine-tuned reasoner does, and it is
unreliable at the counting and branching when it must answer in one step. Both point the same way: let the language
model restate, and compute the rest.

### 3. The fast level

Given the expert's sub-goals, the fast level prunes 75 of 100 plants with one training seed (F43; 86 of 100 with its
trained weights instead of the averaged ones) and 86 of 200 with another (F42). In F42, 95 episodes run out of time,
82 of them without a single cut, after a median of 7 closings of the jaws: it turns the plant, reaches and closes
beside the shoot. No slow level can score above its fast level. The lever found so far is not an architecture: a
frame from one second earlier as memory took `prune` from 79 and 56 to 70 and 89 on two seeds (lab notebook, step 8),
and fast levels with that memory are training.

### 4. Memory, and checking what happened

The note works as memory where it is right: with the label out of view at the cut, the two-level policy solves 26 of
33 and the baseline 1 of 11 (lab notebook). What is missing is a check of outcomes. Nothing in any policy here
compares what an action should have done with what happened: with a button whose first presses do nothing, the
pruning policies fall from 79 and 76 to 49 and 52 (lab notebook, test situations).

### 5. The hand-over between the levels

A sub-goal in words is followed: with the sub-goal swapped for another shoot's, the fast level cuts that shoot in 50%
of episodes and solves 5 of 200 (lab notebook). The form in which one Qwen model writes the note and its own action
head reads it solves 14 of 100 on its own notes and 56 of 100 on the expert's, with 41 episodes out of time even
then: at the budget it can be given, its action side is the weaker part. The text hand-over is not what limits the
system.

### 6. A changed view, and 7. the cost of training the large model

Every policy falls to 0-19 of 100 when the lighting, the camera or the bench changes (lab notebook). That is a
limit of the fast level's vision, separate from reasoning. The large model trains at about 60 samples per second
against 1100 for the small policy, and at about 20 with images at 448 pixels (the run in progress). The reasoner
never learned the count in 600k samples; the 130M model, at about 130 samples per second with its checkpoints
evaluated beside it, had it after 5M to 7M.

### Ranking

A two-level policy succeeds when the slow level names the right shoot and the fast level then cuts it. On training
instructions the first factor is 140 of 200 (the sub-goal right through the episode) and the second 122 of 140. On
held-out instructions the first is 74 of 200.

| Rank | Limit | What it costs now | Lever |
|---|---|---|---|
| 1 | what the slow level sees (the count) | 56 of 67 failures in T; three-shoot plants at 47% against 89% | eyes that are trained |
| 2 | new wording, new combinations | the first sub-goal wrong in 126 of 200 in H | keep the language model's own reading; compute what can be computed |
| 3 | the fast level | 13% to 25% of episodes fail with a correct sub-goal; with the other finished seed, 57% | memory (under test); not this design |
| 4 | no check of outcomes | 25 to 30 points with a sticky button | a check, and data for the fast level |
| 5 | a changed view | nearly everything | pretrained or varied vision; not this design |
| 6 | the hand-over in words | little | none needed |
| 7 | training the large model | hours per run, and it still does not see | do not train it on pixels |

## Is a new architecture the right lever?

For the slow level, yes, and a modest one. Its largest loss is what it sees, and the cheapest proof of that needs no
new model (E0). More training of the same reasoner does not fix it (eleven checkpoints, one constant). The run with
images at 448 pixels tests more tokens from the same frozen tower, at a third of the speed; whatever it shows, the
held-out failures of section 2 remain. For the fast level, no: its lever is memory and it is being tested. For the
one-model form, no: it has the same eyes and a weaker action side.

## The hypothesis

The slow level fails because one large model is asked to see, to remember and to reason, with its vision frozen and
its language layers fine-tuned on thirty sentences. Each job has a cheaper and better-suited learner: seeing is a
supervised problem with exact labels from the simulator, which a small network learns at a thousand samples per
second; remembering three facts is bookkeeping; reading an instruction is what the language model could already do
before it was fine-tuned. Predictions, each with the result that would refute it:

| | Prediction | Refuted if |
|---|---|---|
| P1 | Given the true `see` field, the existing reasoner and fast level (T's) solve 150 to 165 of 200 with training instructions (133 now), three-shoot plants at least 68 of 108 (51 now); held-out 72 to 90 of 200 (66 now) | fewer than 143 of 200 with training instructions: the count is then not the main loss |
| P2 | Eyes trained on the simulator's labels read the count of unseen three-shoot plants in at least 97% of chunks, the flower's colour in 99%, the label in 95%, from 224-pixel views | under 90% for three-shoot plants with the plant crop as well |
| P3 | With the eyes and the memory in place of the truth, the result of P1 holds to within 8 episodes | a larger drop: the memory's rules or the eyes on the policy's own states are the problem |
| P4 | Given the true scene as text, the reasoner names the right first sub-goal for at least 98% of training instructions and 85% of held-out ones (the reasoner now: 159 and 74 of 200, from pixels) | under 70% held-out for every variant of the reasoner |
| P5 | The whole of v2 with T's fast level solves at least 150 of 200 with training instructions and 110 of 200 held-out | under 143 or under 85 |
| P6 | Training the slow level takes under one GPU-hour (3 to 7 now per reasoner run) | |

The ceiling is the fast level: about 150 to 170 of 200 with the one used in T.

## The architecture

```
 every chunk (8 control steps, 0.4 s)                      only when the scene memory changes
                                                           (the expert writes about 2 notes per episode)
 front, wrist     +-------------------------+ fields and  +--------------------+ scene    +---------------------------+
 [+ plant crop] ->| 1 EYES                  |------------>| 2 SCENE MEMORY     |--------->| 3 REASONER, text only     |
                  | small ViT, every weight | confidences | rules, no weights: | as text  | instruction + scene text  |
                  | trained, patch tokens   |             | count, shoots cut, |          |  -> ask  (language model) |
                  | not merged; one head    |             | label once read    |<---------|  -> so, next  (resolver,  |
                  | per field of the scene  |             +---------+----------+ the note |     or the model itself)  |
                  +-------------------------+                       |                     +---------------------------+
                                                                    |
                    4 CHECK: plan again at every change of the memory; "stop" only when no shoot is
                    left to cut; count the steps a sub-goal stands with nothing changing
                                                                    |
                                                                    v  sub-goal: "cut the middle shoot",
                                                                       "show the label" or "stop"
 front, wrist (now and 1 s ago), joint angles --------->+--------------------------+
                                                        | 5 FAST LEVEL (unchanged) |---> 15 joint targets, 8 executed
                                                        +--------------------------+
```

**1. Eyes.** The vision transformer of the small policy (224 pixels, patches of 16, 6 layers, 256 wide, 196 tokens
per image, none merged), over the front and wrist images, followed by a few learned query vectors that attend to the
patch tokens and one classifier per field. The scene state is a short list of typed fields declared in one place:

| Field | Values | Truth in the simulator |
|---|---|---|
| `shoots` | 0 to 3 attached now | `env.active` without `env.cut` |
| `flower` | red, orange, pink, purple | `env.flower_colour` |
| `label` | none, not readable, white, yellow, cyan | `env.label_colour`, `env.label_visible()` |
| per attached shoot, lowest first (optional) | height on the stem in bins; pointing at the robot, to its left or to its right | `env.shoots`, `env.shoot_azimuth` |

The loss is a cross-entropy per field. The labels cost nothing and are valid in every state, also in states only a
policy reaches: unlike action labels, they need no expert. Training frames are the demonstrations' (the states a
rollout visits, with the arm in the way) and single frames of freshly drawn plants (more plants per frame). An
optional third image, the *plant crop*, is the front camera's view narrowed onto the plant, what a crop of a
higher-resolution camera would give: the plant about twice as large. Section 1 says the count does not need it; the
situations of [test-situations.md](test-situations.md) with ties, buds and sick leaves of 2 to 6 pixels will.

**2. Scene memory.** Code, not a network. A field's value counts once the eyes have given it with high confidence for
a few chunks in a row. Then: the plant's number of shoots is the largest count seen; a label's colour, once read, is
kept; when the count of attached shoots falls by one, the shoot named by the current sub-goal is marked cut (the
optional heights confirm which one went). The memory prints itself as the notes' `see` field with the cut shoots
added: `3 shoots, purple flower, label hidden, read cyan; cut: the lowest`.

**3. Reasoner.** It reads text only: the instruction, the scene memory's line, the note it wrote last. It writes the
note's remaining fields, `ask` (the instruction in a fixed form), `so` (the inference) and `next` (the sub-goal, from
the closed set the fast level was trained on). The `aim` field of the first design is dropped: no model wrote it
right in more than 41% of chunks (18% to 41%), and the fast level does not read it. Three ways to fill the reasoner,
behind one interface, to be compared in E3:

- *parse, then compute* (the default): the pretrained language model, not fine-tuned, with a few worked examples from
  training instructions, writes `ask`; a resolver in code works out `so` and `next` from `ask` and the scene memory
  with the rules of `sim/pickup/reasoning.py`. An `ask` the resolver does not know falls through to the model's own
  answer. Where several shoots are meant, it takes the one that needs the least turning if the eyes report which way
  the shoots point (as the expert does), otherwise the lowest first.
- *text fine-tuning*: the same language model trained on the notes as text (no images, so many times faster than
  the reasoner now), with paraphrases.
- *the model alone*: the pretrained model writes its reasoning before the answer, no training and no resolver. Not
  probed yet: on a CPU one such answer took five minutes.

**4. Check.** The plan is made again whenever the scene memory changes, so a cut that happened is seen and a cut
that did not is not assumed. `stop` is given only when every shoot meant is marked cut. A sub-goal that stands for
longer than a set number of steps without any change in the memory is counted as a stall and reported.

**5. Fast level.** As it is: the small flow-matching policy that reads the sub-goal
(`dp_vit_lang_reactive_subgoal.yaml`, or `dp_vit_lang_frames_1s_subgoal.yaml` once those runs are in).

**What stays the same.** The tasks, the held-out split, the note format and its parser, the swap test, the fast
level and its dataset, and every existing config. The new parts are added beside them.

**Two steps.** The first keeps the reasoner now in use and gives it the `see` field, from the truth (E0) and then
from the eyes and the memory (E2). It is the smallest change, and by the argument of section 1 it should be enough
for training instructions. The second puts the text-only reasoner in its place (E3, E4); that is the part aimed at
held-out instructions.

## Why this should fix what was diagnosed

| Diagnosed | Part | Mechanism | Expected |
|---|---|---|---|
| the count is a constant (section 1) | eyes | trained vision at full token resolution, a direct loss on the count instead of a language-model loss through frozen features | three-shoot plants from 47% toward the fast level's 70% to 87% |
| `stop` with a shoot still to cut (27 time-outs) | memory, check | `stop` follows from the memory's list, not from a note | those episodes continue |
| new wording not recognised (section 2) | reasoner | the language model is not fine-tuned on thirty sentences; it keeps what it knew | the probe: 32 of 40 held-out instructions restated exactly and 38 in meaning; the fine-tuned reasoner: 107 of 200 |
| new combinations worked out wrongly (0 of 33) | resolver | counting from an end is computed, not recalled | right whenever `ask` and the count are |
| hours per run for a reasoner that does not see | all | the eyes train at the small policy's rate; the language model is not trained on pixels at all | under one GPU-hour |

What it does not fix: the fast level's ceiling and its dependence on the seed, the sticky button (the fast level has
no skill for pressing again), and the changed views.

## What it costs

| | Now (two-level, v1) | v2 |
|---|---|---|
| Slow level, training | 600k samples at about 60 per second: about 3 GPU-hours; 500k at 448 pixels: about 7 | eyes: about 2M samples at the small policy's rate, about half an hour (an estimate until measured); reasoner: none, or minutes on text |
| Data | the demonstrations with notes | the same demonstrations collected again with a `scene` column (and the plant crop), plus single frames of drawn plants: about an hour of simulation |
| Slow level, per chunk | the large model every chunk: about 80 ms on average on a shared GPU (200 ms per chunk in T against 116 to 125 ms for the fast level alone) | the eyes every chunk (a few ms, to be measured); the language model about twice per episode |
| Code | | a model type for the eyes and its loss, the scene column, the memory, the reasoner and resolver, serving and evaluation: on the order of 1500 lines with tests |
| What is given up | one model that sees and reasons | the scene's fields are written by hand: a new attribute needs a field and labels. In the simulator labels are free; on a real robot they are not |

With the resolver, held-out *combinations* stop being a test of the model: they are right by construction. Held-out
*wording* stays a test (of the parse), and so does everything the eyes must see.

## The experiments

Baselines, all existing and final: the end-to-end small policy (21 of 100, 13 of 100 held out), the two-level policy
(T: 133 of 200; H: 66 of 200), the fast level with the expert's sub-goals (F43: 75 of 100, the ceiling), and, when
they finish, the reasoners continued with paraphrases and at 448 pixels. Every comparison uses the same 200
evaluation layouts per split (seeds from 1,000,000) and the same fast level (T's). The held-out split is the
benchmark's: `reason_mix_heldout`, whose wordings and combinations no training example, worked example or paraphrase
uses. Thresholds of the memory are set on layouts with seeds from 900,000, which are in neither the demonstrations
nor the evaluation. Every learned part is trained with three seeds.

| | Experiment | Tests | Cost |
|---|---|---|---|
| E0 | The existing reasoner with the `see` field of each new note set to the truth (a few lines in `eval.py` and `policy_server.py`), T and H again | P1: is the count the main loss? Decides whether to go on | 1 GPU-hour, no training |
| E1 | The eyes: accuracy per field on unseen layouts, while the expert drives and while the fast level drives; with and without the plant crop; one frame against two (now and 1 s ago) | P2 | 3 GPU-hours |
| E2 | Eyes and memory in place of the truth of E0 | P3 | 1 GPU-hour |
| E3 | The reasoner alone on the true scene as text, 400 drawn tasks per split: the three ways of filling it, the first sub-goal by kind and by kind of held-out instruction | P4; picks the reasoner | under 1 GPU-hour |
| E4 | All of v2: T and H again, by kind, number of shoots and kind of held-out instruction; the scene line swapped for a wrong one (does the plan follow what the eyes report?); hidden label with the card out of view | P5 | 3 GPU-hours |
| E5 | The eyes under changed lighting, camera, bench and distractors (`/train` and `/test` parts), trained with and without that variation | is seeing easier to make robust than acting? No prediction | 2 GPU-hours |
| E6 | Kinds no policy was trained on, with the fast level's existing skills only: "cut from the top down" (the `order` situation), "leave exactly one shoot" | does the reasoning carry to new kinds without new demonstrations? | 2 GPU-hours |
| E7 | Optional, for the fast level: its vision transformer started from the eyes' weights, three seeds | do all seeds start to look early, and does the spread between seeds shrink? | 15 GPU-hours |

E0 comes first and alone. If it refutes P1, the design stops there: the diagnosis of section 1 is then wrong.

## Where it can go wrong

- **A cut seen that did not happen.** The arm hides shoots while it works. If the eyes report one shoot fewer for
  long enough, the memory marks a shoot cut. The eyes are trained on frames with the arm in the way and labelled with
  what is there, not with what is visible; E1 reports the count separately for frames in which the arm is at the
  plant. The fall-back is a learned update: the eyes also read the memory's last line.
- **Eyes trained on the expert's states, used on the policy's.** Measured in E1; the cure is to add frames from
  policy rollouts, which is cheap because the labels come from the simulator.
- **The parse.** In the probe 6 of 40 held-out instructions were restated with the right meaning in a form the
  resolver does not know, and 2 wrongly. The first kind should go when the model may only choose among the fixed forms
  (it scores each form instead of writing freely); E3 measures that, on more than 40 instructions.
- **The resolver's reach.** It computes the four kinds of the benchmark. For anything else the language model is on
  its own, and E6 measures how far that goes.
- **The fast level.** It caps everything, and nothing here improves it.

## What is out of scope

The fast level's architecture, speed and seeds, apart from E7. Visual robustness of the fast level. A skill for
pressing a button again. The one-model form. Passing the fast level a pointer (image coordinates of the shoot) in
place of words: the eyes could supply it, and it becomes the next step if results by named shoot show the fast level
confusing shoots when the sub-goal is right (in T that happens in 2 of 140 such episodes). Real images.

## Alternatives considered

- **More tokens for the same reasoner** (448 pixels; training now). The cheapest change if it works, at a third of
  the training speed. It leaves section 2 as it is.
- **Unfreezing the large model's vision.** It may well learn the count. At 20 to 60 samples per second, and given
  that the count appeared in the models that learned it only after 1M to 5M samples, each attempt is a day of one GPU.
- **A verifier alone**, added to the two-level policy. It would catch the early `stop`, not the wrong shoot, and it
  needs the same eyes to see what happened.
- **One model.** Measured: 14 of 100.

## Prior work to check

Nothing in this document rests on a published result; the reasoning is from the measurements above. The first
design's references (embodied chain-of-thought, Hi Robot) are described in [reasoning-vla.md](reasoning-vla.md) from
their abstracts. Before building, the following should be searched for, since each part of this design is likely to
have published relatives:

- hierarchical vision-language-action model, high-level VLM, low-level policy, language sub-goal
- language-model planner over a symbolic scene description with success detection, closed loop
- object-centric scene state estimation for manipulation, typed slots, state estimator feeding a planner
- VLM counting failure, small objects, visual token merging, input resolution, high-resolution crop or zoom
- frozen vision encoder, fine-tuned VLM, fine-grained perception, unfreeze vision tower
- language to program, neuro-symbolic instruction following for robots, program executed over a scene graph
- decoupling perception from reasoning: perception module as text for a frozen language model
- VLM fine-tuning forgets language generalisation, narrow instruction set, paraphrase augmentation
- failure detection and re-planning in robot manipulation, monitor, verifier, retry
- explicit task memory or state tracking for long-horizon robot policies

## Where it would live

| What | Where |
|---|---|
| Scene fields, truth from the simulator, the memory, the scene as text | `sim/pickup/scene.py` |
| The `scene` column, the plant crop | `sim/pickup/collect.py --scene`, `sim/pickup/env.py` (a third camera, off by default) |
| The eyes: model type, heads, loss | `vla_foundry/models/scene_state.py`, a batch handler, `sim/pickup/configs/eyes_vit.yaml` |
| The reasoner: prompt, worked examples, resolver | `sim/pickup/reasoner.py` |
| Serving: `see` and `plan` requests | `sim/pickup/policy_server.py` |
| Evaluation: E0's forced `see`, the v2 slow level | `sim/pickup/eval.py`, `sim/pickup/scripts/eval_v2.sh` |
| Tests | `tests/essential/sim/test_scene.py`, `tests/essential/models/test_scene_state.py`, `tests/essential/sim/test_reasoner.py` |
