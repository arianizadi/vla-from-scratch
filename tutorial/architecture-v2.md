# Architecture v2: small eyes, a scene memory, and a program for the instruction

The two-level reasoning policy of [reasoning-vla.md](reasoning-vla.md) solves 133 of 200 reasoning episodes when the
instruction is worded as in training and 66 of 200 when it is not. This document works out from the result files what
limits it, decides whether a new architecture is the right answer, specifies one, and reports its first results. The
diagnosis and the predictions were written before anything was built, and where a later measurement corrected them the
correction is marked. A number is a measured one where a run is named; predictions are marked as predictions.

## In short

**The idea.** Split the slow level (the part that reasons) into parts, each trained on what its job needs. Small
*eyes*, a vision transformer in which every weight is trained, turn the camera images into one typed record per part
of the plant: where a shoot leaves the stem, which way it points, whether it is cut. A *scene memory* without weights
keeps what was seen and what was cut. A pretrained language model, not fine-tuned, compiles the instruction once into
a small typed *program*; code evaluates the program on the memory at every look and hands the fast level its sub-goal.
The fast level that moves the arm stays as it is.

**Why.** The reasoner now in use writes "2 shoots" for 105 of 108 three-shoot plants, and that one error is in 56 of
the 67 failures on training instructions. The count is in what its frozen vision hands it (a small head trained on
those features reads it in 91% of frames); the reasoner was never taught to read it, because its only loss is on the
text of its notes. On held-out instructions its first sub-goal is wrong in 126 of 200 episodes: it restates 107 of the
200 instructions correctly, where the same language model, not fine-tuned and shown eight worked examples, restated 32
of 40 in a probe, and it works out new combinations wrongly even when it has restated them correctly.

**The parts**, by what they are in general:

1. a scene-state estimator: a small supervised vision network that predicts one typed record per object part, not text
   and not a count;
2. a scene memory: explicit state with stable part identities, updated by rules from what is seen;
3. a program writer: a pretrained language model that compiles the instruction into a typed program under a grammar
   constraint;
4. a plan: the program evaluated by code on the memory, with four outcomes (execute, an observation is needed,
   complete, failed), transitions that are watched and a bounded recovery;
5. a reactive low-level policy that reads a sub-goal (unchanged).

**Where it stands.** The five parts are in the repository with their tests. Episodes solved of 200, the fast level the
same in every row:

| Slow level | Training instructions | Held out |
|---|---|---|
| the first design (notes written from pixels by a fine-tuned vision-language model) | 133 | 66 |
| v2, its scene from the simulator | 166 | 138 |
| v2, its scene from the eyes (all of v2), the memory's rules as designed | 160 | 135 |
| v2, its scene from the eyes, the memory's two options on | 169 | 148 |
| the same, with a reset when the fast level has gone for a shoot it cannot reach | 180 | 165 |

One run of 200 episodes per cell, with the written program and the eyes that also take the plant crop; a difference of
fewer than about 15 episodes between two such runs can be chance. With a slow level that makes no error the same fast
level solves 169 and 154: what a slow level can add by seeing and reasoning rightly ends there. The two options answer
two errors that the closed loop showed and single frames did not (the eyes are right in 97% to 99% of frames): an
inventory that never completes, and a cut seen that did not happen. The reset answers the fast level's own most
frequent error: told to cut a shoot that faces away, it goes for the shoot without turning the plant in about one
episode in ten, and three in four of those are lost unless the arm is taken back to its start pose to choose again.
With a second seed of the fast level the last two rows are 169 and 145, and 175 and 157. Everything is under "First
results".

## What limits the reasoning policy now

Totals are in the [lab notebook](lab-notebook.md). The breakdowns in this section were computed for this document from
the per-episode records of the result files named (`results/pickup/<run>/<evaluation>/results.json`) and are not in
the notebook yet. Short names for the evaluations used:

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

The number it writes does not depend on the plant. In N, at ten of the eleven checkpoints of the run it writes "2
shoots" for at least 48 of the 50 plants (29 of them have three), and at checkpoint 5 it writes "3 shoots" for all
50. What follows from the count goes wrong with it:

| Shoots to cut (T) | Solved |
|---|---|
| the highest | 56 of 60 |
| the lowest | 67 of 82 |
| the middle one | 5 of 23 |
| two shoots, the middle one among them | 5 of 35 |

The inference follows the count the reasoner wrote, not the plant. At checkpoint 5 of N, where every plant is "3
shoots", the `so` field of the first note is right for all 19 three-shoot plants with an ordinal or exclusion
instruction ("of 3, second from the top is the middle") and wrong for all 14 two-shoot ones (the same sentence, on a
plant with no middle shoot). At the final checkpoint, where every plant is "2 shoots", it is the other way round. So
on training instructions the reasoning is in place and a correct count is what is missing.

Of the 67 failures in T, 56 have the wrong count in their first note. Where the sub-goal the reasoner held was right
in at least 95% of the chunks of an episode, 122 of 140 episodes succeed; in the other 60, 11 do. Of the 47 episodes
that ran out of time, 27 had the reasoner write `stop` with a shoot still to cut: it had planned for a plant with two
shoots.

The explanation given before anything was built (corrected below): `qwen_reasoner.yaml` trains the last 8
language-model layers (`num_trainable_vlm_layers: 8`), and `VLMHFBackboneWrapper._train_only_last_layers` freezes
everything else, the vision tower included. The tower hands on 49 tokens per image, one per 32 x 32 pixels. The front
camera (`sim/pickup/env.py`: 45 degrees of view, about 0.9 m from the plant) shows about 3.4 mm per pixel at 224
pixels, so a shoot, 6 mm thick and 7 to 9 cm long, is at most about 2 pixels by 25, and the plant covers about 2 by 3
of the 7 by 7 tokens (a calculation from the camera's position, not a measurement).

It is not that 224 pixels cannot show it. The one-model policy on the same backbone, whose training also carries the
action loss and ran four times as long (`vla_qwen_reason_reason8k_2500k`, final checkpoint, 50 episodes), has the
count of three-shoot plants right in 21 of 29 first notes and of two-shoot plants in 17 of 21. Our own 130M model,
whose vision transformer keeps 256 tokens per image and is trained with everything else
(`vla_custom_reason_reason8k_10m`), has three-shoot plants right in 24, 29 and 25 of 29 at its checkpoints 5 to 7
(that run is still training; these are not final numbers). And the small policies cut the right shoot from the same
images.

**Corrected by a measurement made since** (the tower probes, under "First results"). The frozen tower is not where the
count is lost. A small decoder with typed heads, trained on the simulator's labels over the tower's frozen output,
reads the count of three-shoot plants in 91% of frames from the very 49 tokens per image that the language model is
given, and in 92% from the 196 patch features before they are merged. The count is in the features. The reasoner,
trained with its last 8 layers through a loss on the text of its notes, did not learn to read it in 600k samples (and
reads it for 128 of 200 plants after 1.5M). What was missing is a loss on what is to be seen, which is what the eyes
of this design have.

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

Two probes were run for this document with the language model of Qwen3-VL-2B as downloaded: no fine-tuning, no images,
on a CPU, 10 drawn instructions per kind and split, so the samples are small.

- *Restating.* Given eight worked examples that use training instructions only, the model was asked to restate an
  instruction in the notes' fixed form. With training instructions it restated 40 of 40 exactly; with held-out ones 32
  of 40 (ordinal 10 of 10, conditional 7, exclusion 8, hidden label 7), and a resolver in code, given the true plant,
  then named the right shoots for exactly those 72. Six of the eight misses have the right meaning in a form the
  resolver does not know ("the shoot the tag points to"; "unless red the lowest, else the highest"); two are wrong
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
beside the shoot. No slow level can score above its fast level. The lever found so far is not an architecture: a frame
from one second earlier as memory took `prune` from 79 and 56 to 70 and 89 on two seeds (lab notebook, step 8), and
fast levels with that memory are training.

### 4. Memory, and checking what happened

The note works as memory where it is right: with the label out of view at the cut, the two-level policy solves 26 of
33 and the baseline 1 of 11 (lab notebook). What is missing is a check of outcomes. Nothing in any policy here
compares what an action should have done with what happened: with a button whose first presses do nothing, the pruning
policies fall from 79 and 76 to 49 and 52 (lab notebook, test situations).

### 5. The hand-over between the levels

A sub-goal in words is followed: with the sub-goal swapped for another shoot's, the fast level cuts that shoot in 50%
of episodes and solves 5 of 200 (lab notebook). The form in which one Qwen model writes the note and its own action
head reads it solves 14 of 100 on its own notes and 56 of 100 on the expert's, with 41 episodes out of time even then:
at the budget it can be given, its action side is the weaker part. The text hand-over is not what limits the system.

### 6. A changed view, and 7. the cost of training the large model

Every policy falls to 0-19 of 100 when the lighting, the camera or the bench changes (lab notebook). That is a limit
of the fast level's vision, separate from reasoning. The large model trains at about 60 samples per second against
1100 for the small policy, and at about 20 with images at 448 pixels (that run has since ended: 141 and 100 of 200).
The reasoner never learned the count in 600k samples; the 130M model, at about 130 samples per second with its
checkpoints evaluated beside it, had it after 5M to 7M.

### Ranking

A two-level policy succeeds when the slow level names the right shoot and the fast level then cuts it. On training
instructions the first factor is 140 of 200 (the sub-goal right through the episode) and the second 122 of 140. On
held-out instructions the first is 74 of 200.

| Rank | Limit | What it costs now | Lever |
|---|---|---|---|
| 1 | what the slow level sees (the count) | 56 of 67 failures in T; three-shoot plants at 47% against 89% | eyes trained with a loss on each part |
| 2 | new wording, new combinations | the first sub-goal wrong in 126 of 200 in H | keep the language model's own reading; compute what can be computed |
| 3 | the fast level | 13% to 25% of episodes fail with a correct sub-goal; with the other finished seed, 57% | memory (under test); not this design |
| 4 | no check of outcomes | 25 to 30 points with a sticky button | a check, and data for the fast level |
| 5 | a changed view | nearly everything | pretrained or varied vision; not this design |
| 6 | the hand-over in words | little | none needed |
| 7 | training the large model | hours per run, and it still does not see | do not train it on pixels |

## Is a new architecture the right lever?

For the slow level, yes, and a modest one. Its largest loss is what it sees, and the cheapest proof of that needs no
new model (E0). More training of the same reasoner helps slowly: to 600k samples the count it writes is a constant
(eleven checkpoints); continued to 1.5M samples (`qwen_reasoner_mix8k_600k_cont900k`) its first note has the right
count for 128 of 200 plants and the two-level policy reaches 153 of 200 (lab notebook). The run with images at 448
pixels (more tokens from the same frozen tower, at a third of the speed; 600k samples and 500k more with paraphrases)
ends at 141 and 100 of 200 (`qwen_reasoner_hires_mix8k_600k_para500k`), its first note with the right count for 133
and 134 of 200 plants: no better at the count than the reasoners continued at 224 pixels (128 and 133; 118 and 124),
as the tower probes below lead one to expect, since the count is already in the 49 tokens. The held-out failures of
section 2 remain: 78, 87 and 100 of 200 for the three, against 148 for v2. For the fast level, no: its lever is memory
and it is being tested. For the one-model form, no: it has the same eyes and a weaker action side.

## The hypothesis

The slow level fails because one large model is asked to see, to remember and to reason, with its vision frozen and
its language layers fine-tuned on thirty sentences. Each job has a cheaper and better-suited learner: seeing is a
supervised problem with exact labels from the simulator, which a small network learns at a thousand samples per
second; remembering three facts is bookkeeping; reading an instruction is what the language model could already do
before it was fine-tuned. The predictions were written before anything was built; the last column is what has been
measured since ("First results" below has the details).

| | Prediction | Refuted if | Measured |
|---|---|---|---|
| P1 | Given the true `see` field, the existing reasoner and fast level (T's) solve 150 to 165 of 200 with training instructions (133 now), three-shoot plants at least 68 of 108 (51 now); held-out 72 to 90 of 200 (66 now) | fewer than 143 of 200 with training instructions: the count is then not the main loss | 159 of 200; three-shoot plants 80 of 108; held-out 96 of 200 |
| P2 | Eyes trained on the simulator's labels read the count of unseen three-shoot plants in at least 97% of chunks, the flower's colour in 99%, the label in 95%, from 224-pixel views | under 90% for three-shoot plants with the plant crop as well | without the plant crop 97.4% of the frames of three-shoot plants, the flower 99.8%, the label 99.6% (96.7% of the frames whose plant has a card); with it 99.1%, 99.8% and 100% |
| P3 | With the eyes and the memory in place of the truth, the result of P1 holds to within 8 episodes | a larger drop: the memory's rules or the eyes on the policy's own states are the problem | with the memory as designed, from 17 fewer to 3 more than the same program on the truth over eight runs (training instructions 152 to 160 against 169 and 166, held-out 135 to 157 against 154 and 138); within 8 in five of the eight. With the memory's two options the eight runs lie between 9 fewer and 10 more (628 and 641 of 800 over the four runs with and without the plant crop, against 627 on the truth) |
| P4 | Given the true scene as text, the reasoner names the right first sub-goal for at least 98% of training instructions and 85% of held-out ones (the reasoner now: 159 and 74 of 200, from pixels) | under 70% held-out for every variant of the reasoner | the written program has the right meaning for 100 of 100 training and 92 of 100 held-out instructions |
| P5 | The whole of v2 with T's fast level solves at least 150 of 200 with training instructions and 110 of 200 held-out | under 143 or under 85 | with the simulator's scene: 166 and 138; with the eyes: 160 and 135 (157 and 139 without the plant crop); with the memory's two options 169 and 148; with the reset as well 180 and 165 |
| P6 | Training the slow level takes under one GPU-hour (3 to 7 now per reasoner run) | | 21 minutes of data collection and 35 minutes of training for the eyes (44 with the plant crop); the language model is not trained |

The ceiling is the fast level: about 150 to 170 of 200 with the one used in T. (Measured later: that is the ceiling of
what seeing and reasoning rightly can add. The runner can also repair one error of the fast level's own from outside,
its wrong first move, and with that reset all of v2 solves 180 and 165: "First results".)

One assumption of the hypothesis did not survive: the frozen vision was not the obstacle. The count is in the features
the frozen tower hands on (the tower probes under "First results"); what the reasoner lacked was a loss on what it was
to see. The split stands for the reason its second sentence gives: each job gets the learner, and the loss, that fits
it.

## What the search for prior work changed

A search of the literature (the sources are under "Prior work") found every ingredient of the first design in
published systems and nothing that could be taken over as it is for this plant. It changed the design in five places,
all built:

1. **Part records, not a count.** The eyes predict one record per part (its height on the stem, the way it points, cut
   or not, visible or not, each with a confidence). The number of shoots and their order are derived from the records.
   A count head can be right for the wrong reason; a record has to find the part.
2. **A typed program, not another note.** The instruction is compiled once into a small program (`If`, `Rank`,
   `Middle`, `Filter`, `Except`, `ReadLabel`, `Cut`) that code evaluates again whenever the memory changes. A language
   model writes it under a grammar constraint, so its form is always valid; whether it means what the instruction says
   is measured separately.
3. **Unknowns are outcomes.** Evaluating the program gives one of four results: `Execute` (a sub-goal),
   `NeedObservation` (a fact is missing: hold, or show the label), `Complete`, `Failed`. An unread label can never
   give `Complete`.
4. **Stable identities.** A part is known by its height on the stem, which does not change when the plant turns. "The
   second shoot from the top" refers to the plant as it was found, cut shoots included, as the expert's notes name
   positions; the program's selection is bound to part ids, so that after the cut it is that part which counts as
   done, and no new "second from the top" is chosen among what is left. The binding is made again only when a fact it
   read has changed (a shoot discovered later, a colour corrected). What the label said stays in the memory after the
   card has turned away.
5. **Transitions are watched.** A part counts as cut when it has been seen cut, never because a cut was commanded. A
   sub-goal that makes no progress for a while (nothing cut, nothing read, the plant not turning) can be answered with
   a bounded number of recoveries (a scripted move out of the plant), after which the slow level gives up. A
   `Complete` that the simulator does not confirm is counted as a false completion, apart from the successes.

It also proposed four comparisons, each of which isolates one suspect:

| Comparison | Isolates | Here |
|---|---|---|
| the language model as downloaded against the fine-tuned one, on the same text tests | language lost in fine-tuning, or never there | "First results", the writer alone |
| the images as they are against a crop or a higher resolution | missing visual detail | the eyes with and without the plant crop |
| typed probes on the frozen tower before its merger, after it, and the generated notes | encoder, connector or decoding | the tower probes |
| the hierarchy on the simulator's state against the hierarchy on learned state, same controller | reasoning and control against perception and tracking | E0, and v2 with the simulator's scene against v2 with the eyes |

## The architecture

```
 every look (whenever the fast level needs a new chunk: 8 control steps, 0.4 s)
 front, wrist     +-------------------------+ a record  +--------------------+  facts   +---------------------------+
 [+ plant crop] ->| 1 EYES                  | per part  | 2 SCENE MEMORY     |--------->| 4 PLAN                    |
                  | small ViT, every weight |---------->| rules, no weights: |          | the program, evaluated on |
                  | trained: height on the  | with its  | parts by height,   |          | the memory at every look: |
                  | stem, bearing, cut,     | confid-   | cut once seen cut, |          | Execute, NeedObservation, |
                  | visible; flower; label  | ence      | label once read    |          | Complete or Failed        |
                  +-------------------------+           +--------------------+          +-------------+-------------+
                                                                                                      ^   |
 instruction ---> 3 PROGRAM WRITER, once per episode: a pretrained language model, not fine-tuned,   |   |
                  writes Cut(Rank(2,top)) under a grammar constraint ---------------------------------+   |
                                                                                                          v
                             sub-goal: "cut the middle shoot", "show the label" or "stop"; or hold; or recover
 front, wrist, joint angles --------------------------->+--------------------------+
                                                        | 5 FAST LEVEL (unchanged) |---> 15 joint targets, 8 executed
                                                        +--------------------------+
```

**1. Eyes** (`sim/pickup/eyes.py`). The vision transformer of the small policy (224 pixels, patches of 16, 6 layers,
256 wide, 196 tokens per image, none merged) over the front and wrist images, a learned embedding per view, then five
learned queries through a two-layer decoder that attends to the patch tokens: one per part slot (three, the most
shoots a plant has), one for the flower, one for the label. 7.1M parameters. A part slot gives six numbers:

| Output | Meaning | Truth in the simulator (`scene.scene_truth`) |
|---|---|---|
| exists | the plant has this part | `env.active`, lowest first |
| height | where it leaves the main stem, in metres above the stem's base | `env.shoots[name]["origin"]` |
| bearing (sin, cos) | the turn of the plate that brings it to face the robot | `REACH_AZIMUTH - env.shoot_azimuth(name)` |
| cut | only its stub is left | `env.cut` |
| visible | a camera shows it now | rays cast from each camera to points along the part |
| flower (4 colours), label (none, not readable, white, yellow, cyan) | | `env.flower_colour`, `env.label_colour`, `env.label_visible()` |

The loss is a cross-entropy or a regression per output. The labels cost nothing and are valid in every state, also in
states only a policy reaches: unlike action labels, they need no expert. The training frames come from the scripted
expert's rollouts with every episode kept, failed ones too (`sim/pickup/collect_scene.py`). An optional third image,
the *plant crop*, is the window around the plant of the front camera rendered at twice the size: what a crop of a
higher-resolution camera gives. No camera is added to the model. Two other backbones can stand in for the small
transformer under the same heads and labels: a pretrained vision model, and the vision tower of a vision-language
model read before or after its token merger (the tower probe), each frozen or with its last blocks trained.

**2. Scene memory** (`sim/pickup/scene.py`). Code, not a network. Only confident evidence counts: a part must be
reported as existing and visible with a probability of at least 0.7. A part is matched to a remembered one by its
height (within 1.75 cm; two shoots are at least 3.5 cm apart); one seen in three looks in a row that matches none
becomes a new remembered part with the next id; parts are never removed. `cut` turns true after three looks in a row
that show it cut. The flower's colour and the label are debounced the same way; the colour a label showed is kept when
the card turns away. The inventory counts as settled when it has not grown for three looks. With the simulator's truth
as the percept the same memory is used, with one look in place of three: an oracle that still has to see each part.

Two options were added after the first closed-loop runs showed what the eyes get wrong ("First results"). They are on
by default where the percepts come from the eyes, with a verdict that must stand four seconds in place of two before
an episode ends on it (`slow_level.DEFAULTS`), and off for the simulator's truth, which is exact. The runs made before
them are repeated with `--no-slow_unseen --slow_revoke 0 --slow_give_up 40`. `unseen`: a part the eyes are sure exists
counts for the inventory although they judge that no camera shows it; such a sighting marks nothing cut, and a
sighting between 1.75 and 2.6 cm from a remembered part is dropped as ambiguous instead of founding a second part.
`revoke`: a part marked cut is unmarked when the eyes then report it, three looks in a row and with confidence, as not
cut. The plan is evaluated at every look, so a `Complete` or a `Failed` that rested on that cut is withdrawn with it.

**3. Program writer** (`sim/pickup/program.py`, `sim/pickup/program_writer.py`). The language:

```
stmt := Cut(sel) | If(cond,stmt,stmt)
sel  := All() | Rank(k,end) | Middle() | Except(sel,sel) | Filter(attr,value,sel) | ReadLabel()
cond := FlowerIs(colour)
```

"Cut the second shoot from the top" is `Cut(Rank(2,top))`; "if the flower is red, cut the lowest shoot, otherwise the
highest one" is `If(FlowerIs(red),Cut(Rank(1,bottom)),Cut(Rank(1,top)))`; "cut every side shoot except the highest
one" is `Cut(Except(All(),Rank(1,top)))`; "read the label in the pot and cut the shoot it names" is
`Cut(ReadLabel())`. `Filter` selects by an attribute (a tie's colour, a sick shoot, a bud) that no task here sets yet.
The writer is the language model of Qwen3-VL-2B as downloaded, given a description of the language and eight worked
examples made from training instructions. It writes greedily, and at every step only the tokens that keep the text a
prefix of some valid program are allowed: an acceptor for the grammar is walked over a trie of the vocabulary, through
the `prefix_allowed_tokens_fn` hook of `generate`. No other library is needed. The constraint guarantees the form. It
cannot guarantee the meaning, which is scored by comparing the written program with the task's own on every plant (two
or three shoots, every flower, every label).

**4. Plan** (`program.Plan`, `sim/pickup/slow_level.py`). At every look the program is evaluated on the memory:

- the inventory is not settled, or the flower's colour is needed and not known: `NeedObservation`, and the arm holds
  still for one step and looks again;
- the label is needed and not read: `NeedObservation` with the sub-goal `show the label`;
- a shoot the program names does not exist (the third from the top of two): `Failed`;
- a shoot that was to stay has been seen cut: `Failed`;
- every shoot meant has been seen cut: `Complete`, and the sub-goal is `stop`;
- otherwise `Execute`: `cut the <position> shoot` for the uncut target that needs the least turning.

The runner (`eval.py --slow_level v2`) hands the sub-goal to the fast level as its instruction, counts the control
steps without progress, starts a recovery when asked to (`--recover N`, off by default), and ends the episode once
`Complete` or `Failed` has stood for two seconds (four with the eyes). Two things were added for the fast level's
first move ("First results"): with `--aim` the sub-goal says what its shoot needs ("cut the lowest shoot, press
blue"), for a fast level trained to be told; and with `--resets N` the arm is taken back to the start pose when the
jaws are set to cut a shoot that is more than 45 degrees out of reach and the plate is not turning, three looks in a
row, so that the fast level chooses again (a target that follows another is given four looks first: the hand is still
coming back from the shoot it cut). Each episode records the program, every change of outcome, the recoveries, whether
a `Complete` was false, how the memory compared with the simulator at every look, and a ledger: one entry per attempt
(the cut of one part, or showing the label) with what it was meant to bring about, closed only by what the memory then
shows (confirmed, with the look that showed it; violated; superseded; given up) and open otherwise.

**5. Fast level.** As it is: the small flow-matching policy that reads the sub-goal
(`dp_vit_lang_reactive_subgoal.yaml`, or `dp_vit_lang_frames_1s_subgoal.yaml` once those runs are in).

**What stays the same.** The tasks, the held-out split, the first design's notes and reasoner, the swap test, the fast
level and its dataset, and every existing config. The new parts are added beside them.

## First results

Not in the lab notebook yet. Every closed-loop number is 200 evaluation layouts with the fast level of T
(`dp_vit_lang_reactive_subgoal_reason8k_20m_seed43`, final averaged weights), so the rows differ in the slow level
only. One run each, and one run of 200 episodes does not resolve small differences. The fast level draws fresh noise
in every run: about 30 of the 200 episodes change their outcome in each direction between two runs, and at a success
rate of 80% the difference between two runs has a standard deviation of about 8 episodes. Runs that differ only in who
wrote the same programs differ here by 3, 8 and 5 episodes (and by 11 with the weaker fast level). A difference of
fewer than about 15 episodes is not a finding.

| Slow level | Training instructions | Held out |
|---|---|---|
| the first design: the reasoner writes notes from pixels (T, H) | 133 | 66 |
| the same reasoner continued to 1.5M samples, without and with paraphrases (lab notebook) | 153, 129 | 78, 87 |
| the same reasoner with images at 448 pixels (600k samples, then 500k with paraphrases) | 141 | 100 |
| **E0**: the same reasoner (600k) handed the true `see` field | 159 | 96 |
| **v2, the simulator's scene, the task's own program** | 169 | 154 |
| **v2, the simulator's scene, the written program** | 166 | 138 |
| **v2, the eyes' scene, the task's own program** | 152 | 157 |
| **v2, the eyes' scene, the written program** (all of v2) | 160 | 135 |
| v2, the eyes without the plant crop: the task's program, the written one | 152, 157 | 147, 139 |
| **all of v2 with the memory's two options**: the task's program, the written one | 160, 169 | 151, 148 |
| the same without the plant crop | 163, 174 | 159, 145 |
| the fast level given the expert's sub-goals (F43, 100 layouts) | 75 of 100 | |

**E0.** With the true `see` field the reasoning of the first design is in place on training instructions: its `so`
field is right in 94% of chunks (56% before), three-shoot plants go from 51 to 80 of 108, a middle shoot to cut from 5
to 19 of 23. On held-out instructions the count was not the main loss: new combinations of an ordinal are still worked
out wrongly (6 of 33 solved), and "keep the middle one" never (0 of 27). The held-out label task goes from 16 to 38 of
45, because the `see` field says that there is a label.

**The written program.** On 200 drawn instructions, the language model as downloaded writes a program with the right
meaning for 100 of 100 training instructions and 92 of 100 held-out ones (ordinal 25 of 25, conditional 23, exclusion
19, label 25), in 12 tokens and 0.3 s on average. Without the grammar constraint its programs are the same and all
well formed: with eight examples the constraint is a guarantee, not a crutch. All eight misses are held-out wordings:
"keep the middle one" in either new wording becomes `Cut(Middle())`, the shoot to keep in place of the others (4 of
4); "leave only the lowest shoot and cut the rest" keeps the wrong end (2); an "unless" has its branches the wrong way
round (2). With the fine-tuned reasoner's weights loaded over the same model, the constrained writer is right for 71
and 60 of 100, and without the constraint it writes no program at all: fine-tuning on thirty instructions cost the
model language it had. Asked for the shoots in one step from accurate facts in words, the model as downloaded is right
for 86 and 70 of 100 (fine-tuned: 48 and 29): it reads better than it computes, which is the case for letting code
compute.

**v2 on the simulator's scene.** With the task's own program the slow level makes no error, so 169 and 154 are the
fast level's own ceiling on these layouts (22 and 37 episodes out of time, 7 and 6 with a wrong shoot cut although the
sub-goal named the right one). With the written program, held-out instructions lose 16: all 15 episodes that keep the
middle shoot in a new wording are compiled to `Cut(Middle())` (none solved, 11 with the task's own program), and three
other programs are wrong (182 of 200 mean what the task's does). No `Complete` was false, and the memory agreed with
the simulator at every look, as it must with the truth as its percept. What was diagnosed is gone where the program is
right: held-out ordinals in a new combination go from 0 of 33 (6 with the true `see` field) to 27 and 23, "keep the
middle one" from 0 of 27 to 20 with the task's program, three-shoot plants with training instructions from 51 of 108
to 88 and 87.

**The eyes.** Trained on 534,043 frames of 8000 expert episodes (`scene_reason_mix_6k`, `scene_prune_2k`), 2M samples,
validated on 21,742 frames of 300 episodes from layouts used nowhere else (seeds from 900,000). Final averaged
weights, one seed each:

| Eyes | Count right: two shoots | three shoots | Parts still attached | Cut | Height | Bearing | Flower | Label, frames with a card |
|---|---|---|---|---|---|---|---|---|
| front and wrist (`eyes_vit_fw_2m`) | 97.0% | 97.4% | 97.8% | 98.9% | 2.7 mm | 7.2 degrees | 99.8% | 96.7% |
| with the plant crop (`eyes_vit_fwp_2m`) | 98.6% | 99.1% | 99.6% | 99.8% | 1.2 mm | 3.4 degrees | 99.8% | 100% |

The count is per frame (as many parts reported as the plant has), "still attached" the same for the parts not cut; cut
is per part; the errors are means over the parts a camera shows. From the same 224-pixel images out of which the
reasoner wrote the right count for 3 of 108 three-shoot plants, trained eyes read it in 97% of frames. The crop (the
same camera at twice the resolution around the plant) halves the errors of height and bearing and takes the count from
97% to 99%: there is detail the 224-pixel view loses, and it is the smaller part of the problem. The count comes late:
right in 57% of frames after 200k samples, 80% after 400k, 95% after 1M (with the crop 68%, 96%, 98%). The two runs
trained side by side on one GPU in 35 and 44 minutes; a look takes about 8 ms on a GPU that is not shared. Three seeds
read the count in 97.2%, 97.6% and 98.1% of frames without the crop and 98.9%, 99.0% and 98.5% with it.

**v2 on the eyes' scene.** With the eyes in place of the simulator's truth, and the memory's rules as designed (three
looks in a row, confidence 0.7), all of v2 solves 160 of 200 episodes with training instructions and 135 held out: 27
more than the first design with training instructions, and twice as many held out. With training instructions the
first design comes level when its reasoner is trained two and a half times as long (153); held out it does not (78,
and 87 with paraphrases). Without the plant crop v2 solves 157 and 139. Against the same program on the simulator's
scene the eight runs lie between 17 fewer and 3 more, which one run each does not resolve. What the eyes themselves
cost can be counted from the episode records instead, and it is two things, neither of them the accuracy on single
frames:

- *An inventory that never completes*: 4 of 200 episodes in every run with the crop, 6 without. On some two-shoot
  plants one shoot is shown only by the wrist camera, and little of it. The eyes report it with probability 1.00 and
  its height to within a centimetre, but judge it not visible (0.07 to 0.40), and the memory counted only what the
  eyes called visible. The inventory stays at one part, the plan cannot name a position, and the arm holds still for
  the whole episode: the picture never changes, so nothing ever will.
- *A cut seen that did not happen*: 3 to 12 of 200 episodes. With the jaws closed beside a shoot the eyes report it
  cut, three looks in a row. In the expert's frames they were trained on, closed jaws at a shoot mean a cut; the fast
  level misses, and the expert almost never does. The memory then marked the part cut for good. If it was the last
  target the plan said `Complete` (3 to 8 episodes per run ended on a false completion); if it was a shoot to keep,
  `Failed` (0 to 8).

Without the crop the eyes also found a part that is not there in 2 to 5 episodes of 200 (a height read 2 cm off for
three looks); with it, in none. In all, 7 to 17 of the 200 episodes of a run with the crop failed with one of these
errors in its record, and 18 to 22 without it. Two-shoot plants carry the loss (64 and 74 of 92 with training
instructions against 81 and 79 on the simulator's scene); three-shoot plants, where the first design lost, are level
with the truth (88 and 86 of 108 against 88 and 87; 51 before).

**The memory's two options.** Both errors come from the memory's rules, so both were answered there (`unseen` and
`revoke`, described under the scene memory). Five variants on 200 validation layouts (seeds from 900,000; the task's
program, the eyes with the crop):

| Memory | Solved | Ended holding for the plant | Ended on a false `Complete` | Ended on `Failed`, a shoot to keep seen cut | Episodes with a cut unmarked |
|---|---|---|---|---|---|
| as designed | 143 | 3 | 10 | 3 |  |
| `unseen` | 161 | 0 | 3 | 4 |  |
| `revoke` after 3 looks | 147 | 4 | 4 | 1 | 5 |
| both | 162 | 0 | 2 | 0 | 2 |
| both, `revoke` after 2 looks | 162 | 0 | 3 | 0 | 5 |

The two runs without `unseen` solve 143 and 147, the three with it 161 to 162, and v2 on the simulator's scene 171 on
these layouts. `unseen` removes the episodes that end holding for the plant. The verdicts that rest on a cut that did
not happen end 13 episodes as designed, 5 with `revoke` alone and 2 to 3 with both. The rest of what `unseen` gains is
not in the records (the sub-goals given are the same) and may be chance: without it the arm never moved in 13 and 10
episodes, with it in 5 to 8. With both on (`--slow_unseen --slow_revoke 3 --slow_give_up 80`, the setting fixed
beforehand and since then the default for the eyes' scene), on the evaluation layouts:

| Eyes, memory | Training instructions: the task's program, the written one | Held out |
|---|---|---|
| with the crop, as designed | 152, 160 | 157, 135 |
| with the crop, both options | 160, 169 | 151, 148 |
| without the crop, as designed | 152, 157 | 147, 139 |
| without the crop, both options | 163, 174 | 159, 145 |
| the simulator's scene | 169, 166 | 154, 138 |

Over the four runs of a row that is 604 of 800 before and 628 after with the crop, 595 and 641 without it, and 627 on
the simulator's scene: with the two options the eyes cost nothing that these runs can resolve, and all of v2 solves
169 and 148 of 200. No episode ended holding for the plant with the crop (4 before) and one without it (6); 0 to 4
ended on a false `Complete` (3 to 8), 1 to 4 on a `Failed` for a shoot to keep (0 to 8); the failed episodes with an
error of the eyes in their record went from 7 to 17 down to 2 to 6 with the crop, and from 18 to 22 down to 3 to 7
without it. A cut was unmarked in 2 to 8 episodes of a run, 37 in all, and 15 of those then succeeded; most of the
others ran out of time in place of ending on a false verdict. The cause is still there (the memory held a cut that had
not happened at some look in 3 to 12 episodes of a run), and the cure at its source is E8. Without the crop the eyes
still found a part that is not there in 2 to 4 episodes; the plant crop is otherwise not needed.

**The tower probes.** The eyes' decoder and heads were trained on the frozen vision tower of the first design's
reasoner (Qwen3-VL-2B) with the same data and labels, for 400k samples, reading either the 196 patch features per
image before the tower's merger or the 49 merged tokens that the language model is given (`eyes_tower_patches_400k`,
`eyes_tower_merged_400k`):

| Features under the same heads | Count right | three shoots | Cut | Height | Bearing | Label, frames with a card |
|---|---|---|---|---|---|---|
| the frozen tower, 49 merged tokens per image | 88.4% | 90.7% | 94.3% | 5.6 mm | 20 degrees | 81.9% |
| the frozen tower, 196 patch features per image | 90.0% | 92.1% | 94.4% | 5.2 mm | 20 degrees | 86.9% |
| the tower with its last four blocks trained, patch features, after 400k samples | 97.0% | 97.8% | 99.2% | 3.4 mm | 6 degrees | 96.3% |
| the small transformer from scratch, after the same 400k samples | 79.7% | 88.4% | 91.5% | 6.4 mm | 26 degrees | 31.8% |
| the small transformer from scratch, after 2M samples | 97.2% | 97.4% | 98.9% | 2.7 mm | 7 degrees | 96.7% |

The count is in the frozen features, and the merger costs it under two points. So the first design did not miss the
count because its tower was frozen or because 49 tokens are too few, as section 1 assumed. The reasoner wrote "3
shoots" for 3 of 108 three-shoot plants from tokens out of which a small head reads the count in 91% of frames: it did
not learn to read it through a loss on the text of its notes. Sample for sample the frozen tower is the better start
(90% against 80% after 400k). By the clock it is not: 117 samples per second with its checkpoints' evaluations against
about 960, and after 35 minutes the small transformer is at 97%. With its last four blocks trained beside the heads
(one learning rate of 1e-4 for everything, not tuned) the tower reads the count in 97.0% of frames after 400k samples,
at about 150 samples per second: a pretrained backbone that may adapt needs about a fifth of the samples of the small
transformer for the same count and about twice its time, and frozen it stays about seven points behind. For this plant
the small transformer is enough; where labelled frames are scarce, as on a real robot, the pretrained tower is the
better start.

**Under perturbation: notes, a memory from the eyes, a memory from the simulator.** The same 200 layouts and fast
level with the task's own program, undisturbed and with a test situation on top (`eval.py --scenario`): someone turns
the plate by 30 to 120 degrees mid-task (`bump`), or a button's first press does nothing or its motor runs slowly
(`sticky`). The memory's rules are as designed here (these runs were started before the options existed). Solved of
200; in brackets the episodes out of time, with a wrong shoot cut, and with the flower touched:

| Slow level | Undisturbed | The plate turned | A button that sticks |
|---|---|---|---|
| notes alone (the first design) | 133 (47, 14, 6) | 111 (74, 12, 3) | 83 (101, 13, 3) |
| the memory, fed by the eyes | 152 (39, 6, 3) | 126 (61, 10, 3) | 101 (95, 3, 1) |
| the memory, fed by the simulator | 169 (22, 7, 2) | 146 (43, 10, 1) | 110 (86, 3, 1) |

The turn costs every slow level about the same, 22 to 26 episodes, and what it costs is the fast level's time. The
memory keeps its lead over the notes and does not add to it. A sticking button costs more (50, 51 and 59 episodes):
with a button whose first press does nothing the memory fed by the simulator solves 31 of 98 episodes, because the
fast level does not press again, and no slow level here can make it. The plan picks, at every look, the target that
needs the least turning, and in 4 to 12 of the 200 episodes of these runs it went back to a shoot it had left; the
notes never did (undisturbed, they name one shoot per episode, also where two are to be cut). Whether holding on to a
target once chosen would save time is not measured.

**The fast level is the ceiling.** With the other finished seed of the fast level (F42: 86 of 200 given the expert's
sub-goals; 74 and 39 under the first design's notes), v2 on the simulator's scene solves 99 and 88 of 200 with
training instructions (the task's program, the written one) and 72 and 91 held out. A slow level without errors lifts
a weak fast level to what the expert's sub-goals give it and no further: 87 and 94 of its episodes with the task's
program end out of time. The bounded recovery does not help. On 200 validation layouts with the stronger fast level
and the simulator's scene, v2 solves 171 without a recovery and 159 and 157 with up to two (asked for after 160 or 240
control steps without progress; 60 and 38 recoveries in 38 and 28 episodes, of which 5 and 1 then succeeded; 9 and 5
episodes were given up), with more wrong shoots cut (5 and 8 against 2). A scripted move out of the plant does not
turn time-outs into successes here. It stays off.

**Where the fast level loses: the first move.** With a correct sub-goal from the first second, 5 to 8 of the 200
episodes of every run on the simulator's scene end without a button pressed or the jaws closed: 8 of the 22 episodes
out of time with the task's program. Traces of such episodes (four validation layouts, eight repetitions each) show an
arm that is not idle. In its first second it sets its jaws to the cutting width, goes to the plant and waits there for
a shoot that faces away. In the other repetitions of the same layouts it shuts its jaws, presses a button and cuts.
Over the 32 runs of v2 with this fast level it happens in 139 of the 2047 episodes whose first sub-goal is "cut the
lowest shoot" (6.8%) and in about 1% of those for the middle and the highest shoot; in 1 of the 940 episodes whose
shoot starts within 30 degrees of reach; and twice as often when the plate has to turn counter-clockwise as when
clockwise (11.8% against 5.8%, the lowest shoot more than 45 degrees away).

A probe asked the fast level for its first chunk 24 times in each of 16 validation layouts whose lowest shoot is 64 to
179 degrees from reach. 89 of the 384 chunks (23%) set the jaws to cut, the others shut them to press; 18 of 128
rollouts from those states stood at the plant after eight seconds. Asked again where they stood, all 18 answered with
the jaws set to cut, 24 times of 24: the first choice is not revisited. Put back at the home pose with open jaws, they
answered as at the start (27% to cut). So "turn the plant first, or go" is drawn once, with the odds of the first
chunk, and a wrong draw costs the episode.

The same judgment decides the way round. Over six runs (1200 episodes), 768 began with the first shoot out of reach:
462 with the button of the shorter way (95% solved, 295 control steps on average), 208 with the other (88%, 369
steps), 98 with no press at all (11 solved). Were every such start like the first group, the fast level would solve
about 8 more episodes in 100. The slow level knows the part's bearing, and the sub-goal did not say it: that is the
first item of the fast-level track below.

**A reset for a wrong first move (E11).** The jaws say which of the two the fast level has chosen: it shuts them to
press a button and sets them to the cutting width on its way to a shoot. Rule e of the runner (`eval.py --resets N`,
off by default): the jaws at the cutting width, the target more than 45 degrees from reach and the plate standing,
three looks in a row, and the arm is taken back to the start pose with open jaws, where the fast level chooses again;
at most N times an episode.

To see what the rule fires on, two runs of the 200 validation layouts were traced without it (162 and 166 solved) and
the rule was run over the traces. It would have fired in 55 and 46 episodes, and they hold 59 of the 72 episodes not
solved:

| Where the rule fires | Episodes of 400 | Solved unhelped | Solved with the reset as first built (three runs of 200) |
|---|---|---|---|
| the first move of a task of one cut (before step 60) | 40 | 10 | 58 of 66 |
| later in a task of one cut, after a turn that ended short or went the long way round | 21 | 0 | 16 of 37 |
| a task of two cuts, after the first cut, while the hand comes back with its jaws still at the cutting width | 40 | 32 | 49 of 60 |

So the wrong first move costs more than the 5 to 8 episodes counted above, which were the arms that never pressed or
closed: about 20 episodes in 200 start for a shoot that faces away, three in four of them are lost unhelped, and a
reset saves most. The third row is a false alarm: the rule fired after the first cut of nearly every task of two cuts
and gained nothing there. The first four looks at a target that follows another are therefore not counted
(`RESET_GRACE`); over the traces the rule then fires in 8 of those 40 episodes and in every other as before.

On the validation layouts, with the simulator's scene and the task's program, 200 episodes a run: 171, 161, 162 and
166 without the reset; 178 and 173 with up to two, as first built; 173 and 179 with the grace. With the eyes and both
options: 162 without, 180 as first built, 174 with the grace. With the grace a reset was asked for in 34, 41 and 32
episodes of 200 (54, 63 and 46 as first built), and 17 of those 107 were tasks of two cuts (60 of 163 before). The
grace takes away most of the resets after a first cut and changes nothing that 200 episodes can show (526 of 600 with
it, 531 of 600 as first built, 494 of 600 in the three matching runs without the rule). The rule was tuned on this
fast level. Under the seed-44 fast level, which it had not seen, the simulator's scene gives 170 without it and 182
with it, and the eyes 180 (both before the grace).

**Other fast levels under v2 (E12).** The slow level stays and the fast level changes. A fast level is judged first on
the validation layouts under a slow level without errors (the simulator's scene, the task's program): nothing of the
evaluation layouts is in that number, and nothing of the eyes. Episodes solved of 200:

| Fast level | Validation layouts, the simulator's scene | the same with up to two resets | Validation, the eyes with both options and resets | Evaluation layouts, the simulator's scene: training instructions, held out | All of v2 (the first eyes, both options, no resets), the task's program | the written program |
|---|---|---|---|---|---|---|
| the sub-goal fast level, seed 43 (the one used above) | 171, 161, 162, 166 | 173, 179 | 174 | 169, 154 | 160, 151 | 169, 148 |
| the same, seed 44 | 170 | 182 (no grace) | 180 (no grace) | 170, 171 | 167, 155 | 169, 145 |
| with a memory of one second, seed 42 | 157 | 157 | 164 | 151, 143 | 136, 137 | 147, 131 |
| the sub-goal fast level, seed 42 (the weak one) | | | | 99, 72 | | |

Two seeds of the sub-goal fast level carry v2 equally, on the validation layouts and on the evaluation's; the third is
the weak one of "The fast level is the ceiling". A memory of one second lifts that seed from 99 and 72 to 151 and 143
under a slow level without errors, and it stays below the two strong seeds that have no memory, with more wrong shoots
cut (10 to 16 episodes in 200 against 2 to 7). The reset does not help it: 157 with and without. It is asked for later
in a task of one cut far more often than with seed 43 (43 episodes of 200 against 11 and 12, 14 of them solved); wrong
shoots fall from 10 to 1 and episodes out of time rise from 29 to 40. Whether a memory helps a seed that is already
strong is the second run with one, still in training. Between seeds 43 and 44 the validation layouts do not decide
(165 on average over four runs against 170 in one; 176 against 182 with resets), so v2 stays on seed 43, which every
other measurement here was made with, and seed 44 is its repetition: the seed of the fast level moves the result more
than anything the slow level still gets wrong.

**With the reset, once on the evaluation layouts.** The memory's two options, the reset and its grace, the eyes and
the fast level were all settled on the validation layouts. All of v2 with the first eyes, both options and up to two
resets (`--resets 2`), 200 episodes a cell:

| Fast level | Training instructions: the task's program, the written one | Held out |
|---|---|---|
| seed 43, without resets (above) | 160, 169 | 151, 148 |
| seed 43, up to two resets | 178, 180 | 183, 165 |
| seed 44, without resets | 167, 169 | 155, 145 |
| seed 44, up to two resets | 171, 175 | 174, 157 |
| the simulator's scene, without resets, seed 43 | 169, 166 | 154, 138 |
| the simulator's scene, without resets, seed 44 (the task's program only) | 170 | 171 |

A reset was asked for in 33 to 37 episodes of 200 with seed 43 and 25 to 32 with seed 44. Every cell rises: 706 of 800
against 628 with seed 43, 677 against 636 with seed 44. With the reset v2 passes what the same fast level solves under
a slow level without errors and without resets (169 and 154), because what the reset repairs is the fast level's error
and not the slow level's. It also takes away the wrong shoots: with the task's program and seed 43 a wrong shoot was
cut in 7 and 8 episodes of 200 before and in 0 and 1 now. A fast level that has gone for a shoot it cannot reach ends
by closing on one it can. What is left with the written program on held-out instructions is the writer's: 17 episodes
with a wrong shoot cut, the instructions it reads as "keep the middle" (a known limit of the writer, left as it is).
With seed 44 the first eyes, which never saw that fast level's states, still end 7 and 8 episodes of 200 on a false
`Complete` with training instructions.

**Eyes trained on the fast level's own frames (E8).** `eval.py --record_scene` recorded 2000 rollouts of the fast
level under v2 on the simulator's scene (new layouts; 178,840 frames) and 200 more for validation (18,052 frames),
each frame with the simulator's truth. `eyes_vit_fwp_pol_2m` was trained on the expert's frames and these, 2M samples.
On the fast level's validation frames the first eyes report a whole part in view as cut, with confidence, 337 times in
42,598 (0.79%, against 0.11% on the expert's frames); the new eyes 45 times (0.11%), and they are as good as the first
on the expert's frames (the count 98.9%, 32 such reports in 40,948). In closed loop, the evaluation layouts:

| Eyes, memory | Training instructions: the task's program, the written one | Held out |
|---|---|---|
| the first eyes, as designed | 152, 160 | 157, 135 |
| trained also on the fast level's frames, as designed | 160, 146 | 147, 136 |
| the first eyes, both options | 160, 169 | 151, 148 |
| trained also on the fast level's frames, both options | 166, 162 | 157, 136 |
| the simulator's scene | 169, 166 | 154, 138 |

With the memory as designed, 0 to 2 episodes of 200 ended on a false `Complete` (3 to 6 with the first eyes) and 0 to
2 on a `Failed` for a shoot to keep (0 to 7); the memory ended with a cut that had not happened in 0 to 4 (3 to 10).
With both options: 0 to 1 and 0 to 1 (0 to 4 and 1 to 3), and 0 episodes with a cut unmarked (2 to 8). The new eyes do
what they were trained for: the cut that did not happen is nearly gone, and with both options the memory no longer has
a cut to take back. The episodes solved do not rise (589 and 621 of 800 against 604 and 628, inside the spread of such
runs): the first eyes lost few episodes to that error, and the two options had recovered most of them. What v2 still
loses is the fast level's. Which eyes to go on with was decided on the validation layouts, with both options and up to
two resets: 180 with the first eyes and 179 with the new ones under the seed-43 fast level (162 and 164 without
resets), 180 and 172 under seed 44, whose frames neither was trained on. Nothing in the episodes solved favours the
new eyes, so the first stay. One thing does speak for them where the fast level changes: under seed 44 the first eyes
end 3 of 200 validation episodes on a false `Complete` and the new eyes 0 (7 and 8 on the evaluation layouts with the
first eyes, above). The recorder and the recipe are in the repository for a fast level whose states the eyes read
worse.

## What it fixed of what was diagnosed

| Diagnosed | Part | Mechanism | Measured |
|---|---|---|---|
| the count is a constant (section 1) | eyes | a loss on each part, in place of a loss on the text of a note | the count of three-shoot plants right in 97% to 99% of frames; three-shoot plants solved with training instructions: 51 of 108 before, 88 and 86 on the eyes' scene (the task's program, the written one) |
| `stop` with a shoot still to cut (27 time-outs) | memory, plan | `stop` follows from the memory's inventory, not from a note | on the simulator's scene no `Complete` is false; on the eyes' scene 3 to 8 of 200 episodes ended on a false one with the rules as designed, 0 to 4 with `revoke` |
| new wording not recognised (section 2) | program writer | the language model is not fine-tuned on thirty sentences; it keeps what it knew | 92 of 100 held-out instructions compiled to the right program (the fine-tuned reasoner restated 107 of 200); the held-out label task 38 of 45 on the simulator's scene and 31 on the eyes' (16 before) |
| new combinations worked out wrongly (0 of 33) | plan | counting from an end is computed, not recalled | held-out ordinals in a new combination: 23 to 31 of 33 over the six runs with a written or the task's program |
| hours per run for a reasoner that does not see | all | the eyes train at the small policy's rate; the language model is not trained at all | 35 to 44 minutes for the eyes |

What it does not fix: the fast level's ceiling and its dependence on the seed (the reset, added later, takes back the
wrong first move, about a third of what that ceiling loses), the sticky button (the fast level has no skill for
pressing again), the changed views, and the eight held-out instructions in a hundred that the writer compiles to a
program with another meaning.

## What it costs

| | The first design (two-level) | v2 |
|---|---|---|
| Slow level, training | 600k samples at about 60 per second: about 3 GPU-hours; 500k at 448 pixels: about 7 | the eyes: 2M samples at about 960 per second (750 with the crop), the two runs side by side on one GPU: 35 and 44 minutes; the language model: none |
| Data | the demonstrations with notes | 8000 expert episodes recorded again as frames with part records (every fourth step), 534,043 frames with the crop: 21 minutes with 48 simulator processes |
| Slow level, per look | the large model every chunk: about 80 ms on average on a shared GPU (200 ms per chunk in T against 116 to 125 ms for the fast level alone) | the eyes every look: about 8 ms (55 ms on a GPU shared with other work); the language model once per episode, 0.2 to 0.3 s |
| Code | | about 4300 lines in ten new files and 150 added to `eval.py`, with about 4100 lines of tests |
| What is given up | one model that sees and reasons | the scene's fields are written by hand: a new attribute needs a field and labels. In the simulator labels are free; on a real robot they are not. The program language is closed: an instruction outside it cannot be compiled |

With the program evaluated by code, held-out *combinations* stop being a test of the model: they are right by
construction once the program is. Held-out *wording* stays a test (of the writer), and so does everything the eyes
must see. Results with the simulator's scene and results with the eyes' scene are reported apart for that reason.

## The experiments

Baselines, all existing and final: the end-to-end small policy (21 of 100, 13 of 100 held out), the two-level policy
(T: 133 of 200; H: 66 of 200), the fast level with the expert's sub-goals (F43: 75 of 100, the ceiling), and the
reasoners continued with paraphrases and at 448 pixels. Every comparison uses the same 200 evaluation layouts per
split (seeds from 1,000,000) and the same fast level (T's). The held-out split is the benchmark's:
`reason_mix_heldout`, whose wordings and combinations no training example, worked example or paraphrase uses. The eyes
are validated on layouts with seeds from 900,000, which are in neither the demonstrations nor the evaluation. The
memory's options answer errors that were found in the evaluation runs themselves: their variants were compared on
those validation layouts, and one setting, fixed before any run with it, was run on the evaluation layouts.

| | Experiment | Tests | State |
|---|---|---|---|
| E0 | The existing reasoner with the `see` field of each new note set to the truth (`eval.py --see expert`), T and H again | P1: is the count the main loss? | done: 159 and 96 |
| E1 | The eyes: accuracy per output on unseen layouts; with and without the plant crop | P2; images as they are against a crop | done: the count of three-shoot plants in 97.4% of frames, 99.1% with the crop |
| E2 | The tower probes: the eyes' heads on the frozen Qwen tower, before and after its merger; with its last four blocks trained | where the first design loses the count; frozen against partly unfrozen | done: 90.7% and 92.1% of three-shoot frames from the frozen tower, 97.8% with four blocks trained |
| E3 | The program writer alone: constrained and free, as downloaded and with the fine-tuned weights, and one-step answers from facts | P4; language lost in fine-tuning or never there | done: 100 and 92 of 100 programs right |
| E4 | All of v2, T and H: the simulator's scene and the eyes' scene, the task's program and the written one | P3, P5; reasoning and control against perception | done: 166 and 138; 160 and 135 |
| E5 | The memory's two options (`unseen`, `revoke`): variants on validation layouts, one setting on T and H | do they remove the inventory that never completes and the cut that did not happen? | done: 143 to 162 of 200 on validation layouts; all of v2 169 and 148 |
| E6 | The three slow levels under perturbation: the plate turned mid-task, a button that sticks | does an explicit memory hold up when the scene changes under the policy? | done: the plate turned 111, 126, 146 (notes, eyes, simulator); the button 83, 101, 110 |
| E7 | v2 over the weak seed of the fast level; the bounded recovery (`--recover 2`) on validation layouts | how much of v2 is the fast level; does a retry turn time-outs into successes? | done: the weak seed 99 and 88, 72 and 91; the recovery 171 without, 159 and 157 with: it stays off |
| E8 | Three seeds of the eyes; frames from the fast level's own rollouts added to their training data | how much the eyes vary; the cure at the source for the cut that did not happen | three seeds: done (97.2%, 97.6%, 98.1%; with the crop 98.9%, 99.0%, 98.5%); the policy's frames: done (`eyes_vit_fwp_pol_2m`: the cut that did not happen 0.79% to 0.11% of parts in view on those frames; episodes solved unchanged) |
| E11 | The fast level's first move: traces and a probe of its first chunk; a reset to the start pose; the aim in the sub-goal (two fast levels, each served with and without it) | where the fast level loses with a correct sub-goal, and what telling it the aim buys | the traces, the probe and the reset: done (validation layouts: 161 to 171 without the reset, 173 to 179 with); the aim: two fast levels in training |
| E12 | v2 over the other fast levels (a third seed; two with a memory of one second) | which fast level carries v2 best; how much the seed matters under a slow level without errors | done for seed 44 (170 on the validation layouts against 161 to 171) and for the first with a memory (157); the second with a memory is in training |
| E9 | Kinds no policy was trained on, with the fast level's existing skills only: "cut from the top down" (the `order` situation), "leave exactly one shoot" | does the reasoning carry to new kinds without new demonstrations? | later |
| E10 | Optional, for the fast level: its vision transformer started from the eyes' weights, three seeds | do all seeds start to look early, and does the spread between seeds shrink? | later |

How the eyes hold up under changed lighting, cameras and benches is the subject of a separate experiment on visual
robustness and is left out here.

## The fast level: a second track

Everything above leaves the fast level's weights as they are, and they are the ceiling: with a perfect slow level 31
to 46 of 200 episodes still fail, most of them out of time. The reset takes back a third to a half of those from
outside; what follows is about the fast level itself. With the weaker of its two finished seeds, 101 to 128 fail. What
a correct sub-goal still leaves to the fast level's own judgment costs most: whether the plant must be turned first,
and which way ("First results", the first move). Four reviews of recent work on control (model-based training of
whole-body controllers, video world-action models, memory for long tasks) were read for what they offer this arm. None
is a drop-in: their controllers read numerical state, not images, and none cuts anything. What is taken from them is a
set of experiments for after the slow level. The aim of item 1 is built and its fast levels are in training; the
ledger of item 4 is built as a record.

**Adopted, in this order:**

1. **The aim in the sub-goal** (built; its two fast levels are in training:
   `dp_vit_lang_reactive_aim_reason8k_20m_seed43` and, with a memory of one second,
   `dp_vit_lang_frames_1s_aim_reason8k_20m`). The fast level's most frequent error with a correct sub-goal is its
   first move ("First results", the first move): told "cut the lowest shoot", it has to judge by itself whether the
   shoot is in reach or the plant must be turned first, and which way. The plan knows the part's bearing, and the
   expert's notes carry the same fact in their `aim` field, which the sub-goal fast level was never shown. So the
   sub-goal now can carry it: "cut the lowest shoot, in reach", "cut the lowest shoot, press blue". The instruction
   type `aim` writes that text from the note in each frame's metadata, so the demonstrations and their preprocessed
   shards are used as they are; `dp_vit_lang_reactive_aim.yaml` trains on the sub-goal with the aim and without it,
   half each, so that one set of weights can be served both ways (`eval.py --aim`) and what the aim buys is measured
   without a second seed in the way. The plan says "in reach" within 25 degrees of facing the robot (the expert's own
   sector) and names the button of the shorter way otherwise.
2. **Numbers beside the words.** The same for the part's height, and for the phase (turn, approach, cut, retreat), as
   numbers the fast level reads beside its instruction, with one target kept active until its check passes (the plan
   already does). After the aim, and only if it leaves something: the wrong shoots cut under a correct sub-goal (7 and
   8 episodes of 200) turned out to be cut by an arm that had gone for a shoot it could not reach, and with the reset
   they are 0 and 1.
3. **A state-based primitive for one shoot** (approach, align, close, retract), learned as a correction to the
   scripted expert's reference move and started from the demonstrations; then imitation alone against imitation plus
   reinforcement learning through a learned model of the arm's dynamics over short imagined stretches. Five seeds
   each, first on the simulator's geometry and then on the eyes', with the scripted controller as the baseline.
   Measured: time to a safe cut, path length, smoothness, contacts with the flower, missed cuts, recovery from
   perturbed starts, on plants not trained on. The cut itself is never imagined: the imagined stretch ends at the cut,
   the simulator makes it, the eyes check it. What the task needs to know (which button, the plate's angle, which
   shoot, what is cut) stays in code. Findings from the controller work that carry over as things to keep, each
   reported for its own setting: a critic that values what lies beyond a short imagined stretch (without it every clip
   failed), actions given relative to a reference (position error 28 to 25 mm), imagined stretches started from
   perturbed states (fewer failures under pushes), penalties on actions and joints (training is unstable without), and
   judging a learned model by the closed loop and not by its fit (a model with 34% lower loss gave no better control).
   One report of the same method on an arm describes policies that hover or hold on for ever until the reward and the
   ending of an episode are changed: the failure to expect here.
4. **The memory as a ledger of outcomes** (built, as a record: `slow_level.Supervisor.ledger`). For each attempt: what
   was aimed at, what it was meant to bring about, what was then seen, and the look that supports it. A cut is entered
   only on evidence, and an attempt without an outcome stays open. What is still missing is the comparison made
   beforehand: the expected effect of a sub-goal in numbers ("this part cut, the flower untouched, the plate turned by
   about so much") checked against the state afterwards. The comparison is on state, not on pixels: a predicted future
   that looks right need not be one the action brings about. The experiment that decides whether an explicit ledger is
   enough holds the fast level fixed and compares notes alone (the first design), the ledger from the eyes, and the
   ledger from the simulator, on the same plants, undisturbed and disturbed: the table of "First results" and its
   paragraph on perturbation.
5. **Corrections after the policy's own failures.** The scripted expert labels the correction after a missed cut, a
   press that did nothing or an overshoot, with the failure kept as the context. The takeover demonstrations of step 8
   dropped the policy's part and did not help; these keep it, and recovery is evaluated apart from success.
6. **One cheap ablation: an auxiliary loss on the future.** Beside the action loss the fast level predicts the part
   records, or its own features, a short time ahead. Its size and its speed when served do not change. Same seeds with
   and without.

Also kept from the reviews, as practice: starts sampled toward the states before failures while keeping uniform
starts, and an audit of the expert's own trajectories for pauses and detours before more imitation of them.

**Not adopted now.** Fine-tuning the image policy on a task reward, and a full model-based actor-critic for it: the
learned model would have to predict images or features, and a smooth model of a plant whose shoots come off is the
part most likely to be wrong. Video world-action models: the published ones are far larger than this hardware and
budget, and a generated video can change which shoot is which. A learned memory branch in the fast level (recurrent or
test-time-trained): only if the explicit memory leaves a measured gap, since both the previous frame and the previous
action as inputs made this policy worse. Pipelines across several GPUs and simulator gradients: nothing measured here
says the simulator is the limit.

## Where it went wrong, and where it still can

- **A cut seen that did not happen** (measured: 3 to 12 of 200 episodes). Foreseen, and the guard was not enough:
  three looks in a row and a confidence of 0.7 do not help when the eyes are wrong in the same way at every look. The
  cause is the next point. `revoke` lets the memory take the cut back once the arm has left and the shoot is seen
  whole. With both options on, 0 to 4 episodes of 200 ended on a false completion (3 to 8 before) and 1 to 4 on a
  `Failed` for a shoot to keep (0 to 8); in 3 to 12 episodes of a run the memory still held such a cut at some look.
- **Eyes trained on the expert's states, used on the policy's.** The expert closes its jaws on a shoot and the shoot
  comes off; the fast level closes them beside it. The cure at the source is to add frames from the fast level's own
  rollouts, which is cheap because the labels come from the simulator (E8). `eval.py --record_scene` records them, and
  eyes trained on them as well (`eyes_vit_fwp_pol_2m`) are in training as this is written; the result is not in.
- **An inventory that never completes** (measured: 4 to 6 of 200). Not foreseen: holding still was the answer to a
  missing fact, and it cannot bring a fact that the picture does not hold. `unseen` takes the eyes' word that the part
  exists. A hold that asks for a view to change (turn the plate) would be the general answer, and the fast level has
  no such skill outside the label task.
- **The writer.** 8 of 100 held-out instructions compile to a program that means something else, and the grammar
  constraint cannot catch that. No training instruction says "middle", and no worked example keeps a middle shoot:
  examples that cover each construct in each role are the first thing to try, and asking for the program twice with
  different examples and holding when the two disagree is the cheapest check. Neither is built.
- **The language's reach.** The program computes what its seven constructs express. An instruction outside them cannot
  be compiled, and E9 measures how far they go.
- **The fast level.** It caps everything, and nothing built here improves it.
- **One run each.** Every closed-loop number here is one run of 200 episodes, and the difference between two such runs
  has a standard deviation of about 8. The differences this document rests on are larger (133 to 160, 66 to 135); the
  ones between variants of v2 mostly are not.

## What is out of scope

The fast level's architecture, speed and seeds, apart from the track above. Visual robustness. The one-model form.
Real images.

## Alternatives considered

- **More tokens for the same reasoner** (448 pixels, 196 tokens per image), or its vision unfrozen. Neither is what it
  lacks: the 49 tokens it has carry the count (the tower probes). Both leave section 2 as it is. The reasoner at 448
  pixels: 141 and 100 of 200 with the same fast level (133 and 66 at 224 pixels; 153 and 78, 129 and 87 with its two
  continuations). The one-model line ends there.
- **A loss on the count for the reasoner itself**: the eyes' heads on its vision tokens, trained beside the loss on
  its notes. The smallest change to the first design that the tower probes suggest, and the one to try if one model is
  to do everything. It leaves section 2 as it is, and it is not built.
- **A verifier alone**, added to the two-level policy. It would catch the early `stop`, not the wrong shoot, and it
  needs the same eyes to see what happened.
- **One model.** Measured: 14 of 100.
- **A pretrained vision backbone for the eyes** (a self-supervised transformer with 14-pixel patches) and a small
  text-only language model for the writer. Both are supported by a config field; neither is on the machines used here,
  so the eyes are trained from scratch and the writer is the language model of the vision-language model already in
  use. The comparison of a pretrained backbone, frozen and partly unfrozen, with the one trained from scratch was made
  with the pretrained tower that is on the machine (the tower probes): frozen it reads the count in 90% of frames
  after 400k samples, with its last four blocks trained in 97.0% after 400k samples, the small transformer from
  scratch in 80% after 400k and 97% after 2M.

## Prior work

The ingredients of this design are established; the combination was built here because nothing found could be used as
it is. Each source below was opened and its abstract (or the page named) read for this document; nothing from any of
them was reproduced here, and no code or weights of theirs are used.

*Seeing, parsing and executing as separate parts.*

- Yi et al., "Neural-Symbolic VQA: Disentangling Reasoning from Vision and Language Understanding", NeurIPS 2018
  ([arXiv:1810.02338](https://arxiv.org/abs/1810.02338)): a structural scene representation is recovered from the
  image and a program trace from the question, and the program is executed on the scene representation; 99.8% on
  CLEVR. The closest relative of the slow level here, for answering questions about images, not for acting.
- Shin et al., "Constrained Language Models Yield Few-Shot Semantic Parsers", EMNLP 2021
  ([arXiv:2104.08768](https://arxiv.org/abs/2104.08768)): a pretrained language model paraphrases an input into a
  controlled sublanguage that maps to a meaning representation. The program writer does this with a typed program as
  the sublanguage.
- Liang et al., "Code as Policies: Language Model Programs for Embodied Control", 2022
  ([arXiv:2209.07753](https://arxiv.org/abs/2209.07753)): a language model writes policy code that processes
  perception outputs and parameterises control primitives. Here the model writes far less: a program in a closed
  language whose interpreter is fixed and tested.
- Carion et al., "End-to-End Object Detection with Transformers", 2020
  ([arXiv:2005.12872](https://arxiv.org/abs/2005.12872)): detection as direct set prediction with learned object
  queries and bipartite matching. The eyes use learned queries; with at most three parts ordered by height, sorted
  targets take the place of the matching.

*A slow level over a fast one, with feedback.*

- Huang et al., "Inner Monologue: Embodied Reasoning through Planning with Language Models", 2022
  ([arXiv:2207.05608](https://arxiv.org/abs/2207.05608)): success detection, scene description and human interaction
  fed back to a language-model planner in a closed loop. There the reasoning stays in the language model; here it is a
  program evaluated by code.
- Shi et al., "Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models", 2025
  ([arXiv:2502.19417](https://arxiv.org/abs/2502.19417)): a vision-language model reasons over the prompt and feedback
  to choose the next step, which low-level actions carry out. The first design here has that shape.
- Guo et al., "DoReMi: Grounding Language Model by Detecting and Recovering from Plan-Execution Misalignment", 2023
  ([arXiv:2307.00329](https://arxiv.org/abs/2307.00329)): a language model generates constraints that indicate
  misalignment, a vision-language model checks them continuously during execution, and recovery follows a violation.
  The check here is on typed state from trained eyes.
- Rana et al., "SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning", 2023
  ([arXiv:2307.06135](https://arxiv.org/abs/2307.06135)): planning over a scene graph, with the plan refined by
  feedback from a scene-graph simulator.
- Yang et al., "EPoG: Integrated Exploration and Sequential Manipulation on Scene Graph with LLM-based Situated
  Replanning", ICRA 2026 ([project page](https://noietch.github.io/projects/epog/)): a belief graph of known and
  unknown objects updated from observations, and replanning when execution departs from the plan. The unread label is
  this design's one unknown.
- Hu et al., "What Matters in Orchestrating Robot Policies: A Systematic Study of Hierarchical VLA Agents", 2026
  ([arXiv:2606.10267](https://arxiv.org/abs/2606.10267)): a benchmark of the design choices of such hierarchies
  (planner, controller, the switch between them, how observations and memory reach the planner), in simulation and on
  a real robot.

*Memory.*

- Ji et al., "HiMe: Hierarchical Embodied Memory for Long-Horizon Vision-Language-Action Control", 2026
  ([arXiv:2607.03449](https://arxiv.org/abs/2607.03449)): an executor, a sentry for working memory and a planner, with
  a memory that supports adding, updating and deleting, for tasks that the current observation does not determine.
- Torne et al., "MEM: Multi-Scale Embodied Memory for Vision Language Action Models", 2026
  ([arXiv:2603.03596](https://arxiv.org/abs/2603.03596)): a short memory of video beside a long memory in text.
- Sobol Mark et al., "BPP: Long-Context Robot Imitation Learning by Focusing on Key History Frames", 2026
  ([arXiv:2602.15010](https://arxiv.org/abs/2602.15010)): conditioning on past observations naively fails through
  spurious correlations; a few task-relevant key frames are kept instead. The same failure was measured here with the
  previous frame and the previous action.
- Jiang et al., "RoboTTT: Context Scaling for Robot Policies", 2026
  ([arXiv:2607.15275](https://arxiv.org/abs/2607.15275)): a memory in fast weights updated during inference. The
  learned alternative to an explicit memory, not adopted (see the fast-level track).

*Seeing small things.*

- Paiss et al., "Teaching CLIP to Count to Ten", 2023 ([arXiv:2302.12066](https://arxiv.org/abs/2302.12066)): a loss
  aimed at counting improves counting in a pretrained vision-language model: counting does not come by itself.
- Zhang et al., "MLLMs Know Where to Look: Training-free Perception of Small Visual Details with Multimodal LLMs",
  ICLR 2025 ([arXiv:2502.17422](https://arxiv.org/abs/2502.17422)): the accuracy of multimodal language models is very
  sensitive to the size of the thing asked about. A shoot here is 2 pixels thick.

*For the fast-level track.*

- Liao et al., "SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model", 2026
  ([arXiv:2610.28480](https://arxiv.org/abs/2610.28480)): a differentiable model of the dynamics fitted to logged
  simulator transitions, and a tracking policy improved by backpropagation through short imagined rollouts. For
  humanoid motion tracking; the abstract says nothing of objects.
- The ablation study of the NDM Gym benchmark
  ([page](https://github.com/Frank-ZY-Dou/ndm-gym/blob/main/benchmarks/g1_lafan1_tracking/ablations.md)): the source
  of the five findings quoted in the track (critic, reference-relative actions, perturbed starts, penalties, model fit
  against control), each for its own configuration.
- `supermimic_manipulation` ([repository](https://github.com/emopointer/supermimic_manipulation)): the method on a
  simulated arm with low-dimensional state, a bank of validated start states, counters kept analytically, and the
  hovering and holding failures with the changes that removed them.
- Zheng et al., "FLARE: Robot Learning with Implicit World Modeling", 2025
  ([arXiv:2505.15659](https://arxiv.org/abs/2505.15659)), and Wu et al., "Unleashing Large-Scale Video Generative
  Pre-training for Visual Robot Manipulation" (GR-1), 2023 ([arXiv:2312.13139](https://arxiv.org/abs/2312.13139)): a
  policy that also aligns its features with, or predicts, future observations. The source of the auxiliary-loss
  ablation.
- Ahmed et al., "Seeing Through the Illusion: Test-Time Scaling for World Action Models", 2026
  ([arXiv:2611.03585](https://arxiv.org/abs/2611.03585)): a plausible predicted future need not be realised by the
  action generated with it. The reason the outcome check here compares states.

The first design's own references (embodied chain-of-thought, Hi Robot) are in [reasoning-vla.md](reasoning-vla.md).

## Where it lives

| What | Where |
|---|---|
| Part records, their truth and visibility, the plant crop, the scene memory | `sim/pickup/scene.py` |
| The perception dataset | `sim/pickup/collect_scene.py` (the expert drives); `sim/pickup/eval.py --record_scene` (whoever drives the evaluation) |
| The eyes: model, loss, metrics, serving; the tower probe | `sim/pickup/eyes.py`, `sim/pickup/eyes_data.py`, `sim/pickup/train_eyes.py` |
| Programs: the language, the acceptor, the plan and its outcomes | `sim/pickup/program.py` |
| The program writer and its evaluation | `sim/pickup/program_writer.py` |
| The slow level in the loop, the recovery, the ledger, the episode record | `sim/pickup/slow_level.py`, `sim/pickup/eval.py --slow_level v2` (`--scene`, `--program`, `--slow_unseen`, `--slow_revoke`, `--slow_give_up`, `--aim`, `--resets`, `--recover`) |
| Serving the eyes and the writer; starting everything | `sim/pickup/slow_server.py`, `sim/pickup/scripts/eval_v2.sh` |
| E0 | `sim/pickup/eval.py --see expert`, `sim/pickup/policy_server.py` |
| The aim in the sub-goal: its text, the instruction type, the fast levels' configs | `sim/pickup/reasoning.py` (`aim_of`, `aimed_subgoal`), `vla_foundry/data/pipelines/robotics.py` (`aim_instruction`), `sim/pickup/configs/dp_vit_lang_reactive_aim.yaml`, `dp_vit_lang_frames_1s_aim.yaml` |
| Tests | `tests/essential/sim/test_scene.py`, `test_program.py`, `test_program_writer.py`, `test_eyes.py`, `test_eyes_tower.py`, `test_slow_level.py`, `test_see.py`; `tests/essential/data/test_aim_instruction.py` |

```
python -m sim.pickup.collect_scene --task reason_mix --episodes 6000 --crop --out data/scene/scene_reason_mix_6k
python -m sim.pickup.train_eyes --data data/scene/scene_reason_mix_6k --val data/scene/scene_val_reason_mix \
    --cameras front wrist plant --out experiments/pickup/eyes_vit_fwp_2m --samples 2000000
python -m sim.pickup.program_writer --model Qwen/Qwen3-VL-2B-Thinking --out writer.json
PICKUP_SCENE=eyes PICKUP_EYES=experiments/pickup/eyes_vit_fwp_2m PICKUP_PROGRAM=writer \
    PICKUP_WRITER=Qwen/Qwen3-VL-2B-Thinking sim/pickup/scripts/eval_v2.sh experiments/pickup/<fast run> reason_mix 200
```
