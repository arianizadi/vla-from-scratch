# Architecture v3: harder reasoning on the plant, and what v2 needs for it

[Architecture v2](architecture-v2.md) solves 180 of 200 reasoning episodes when the instruction is worded as in training
and 165 of 200 when it is not, and what it still loses is the fast level's. This document designs the next step:
reasoning that v2's four task kinds do not ask for (telling a sick shoot from a healthy one, not acting when nothing is
to be done, cutting in a given order, turning the plant to look before deciding, counting), and the smallest change to
each of v2's five parts that carries it. It is a design and a plan of work for implementation agents. Steps 1 to 3
of the plan are built ("As built: steps 1 to 3", after the plan); nothing else is. A number is v2's own (from the
[lab notebook](lab-notebook.md) or from architecture-v2.md) unless it is marked as computed for this document, and
every expectation is marked as a prediction.

## In short

**The idea.** Keep v2's split and give each part the least that the new tasks need. The eyes get four more outputs per
part (a tie's colour, sick, a bud, a tag's colour), each with a class of its own for "cannot be read in this view". The
scene memory keeps them per part and freezes them when the part is cut. The program language gets five constructs: cut
in an order, one statement after another, take some shoots of a selection, compare a count, cut nothing. Stopping and
looking are not constructs. A program that selects nothing still on the plant is complete, and a fact about one shoot
that the program needs and the memory does not have is an outcome, which turns the plant to show that shoot. The fast
level stays as it is for every task whose sub-goals it already has and is trained once more, late, for one new sub-goal
("show the lowest shoot").

**The ladder.** Six rungs, each adding one kind of reasoning, ordered so that the first two can run before anything
is trained:

| Rung | Situation | Reasoning | What is new |
|---|---|---|---|
| 1 | `order`: "cut the side shoots from the top down" | a sequence, and memory of what is done | two constructs, stages in the plan |
| 2 | `nothing_to_do`: an already pruned plant | knowing when not to act | complete with no action; the arm goes home |
| 3 | `ties`: "cut the shoot with the red tie" | binding a word to something seen on one shoot | one output of the eyes |
| 4 | `sick`, `buds`: "cut only the sick shoots" | diagnosis; none to all of the shoots are meant | two outputs of the eyes |
| 5 | compositions and counting, none demonstrated: "cut the highest shoot that has no bud", "leave exactly one shoot" | computing what no example showed | three constructs; tasks defined by a program |
| 6 | the hidden tag: a tag readable from one side only | looking before deciding, and remembering what was seen | a look at one shoot; one sub-goal for the fast level |

**The plan.** Twelve steps, each sized for one implementation agent with its tests. The first three (the language, the
simulator's truth with attributes, the memory and the plan) need no GPU. The fourth runs v3 on the simulator's scene
with the fast level as it is, which shows in a day what the fast level can do on each rung and what the new looks cost
it. In all about 53 GPU-hours on L40S GPUs (48 GB), about 22 of them training; the rest is closed-loop evaluation.

**What the search for prior work found.** The ingredients exist and the combination was not found: no published system
was found in which a small language model, not fine-tuned, writes a typed program under a grammar constraint that code
executes over a learned memory of parts, with unknown facts as outcomes, on top of a learned sub-goal policy. Nothing
can be taken over as code. What others tested changed the design in eight places ("What the search for prior work
changed").

## What the new tasks ask, and what is already there

The situations are in [test-situations.md](test-situations.md). Built, and solved by the scripted expert in 100 of 100
layouts each: `ties` (A1), `sick` (A3), `buds` (A4), `nothing_to_do` (A5) and `order` (A8). Designed and not built: the
tie behind the plant (A2), the legend (A6), which pot (A7), counting (A9), most leaves (A10), balance (A12). No policy
was trained or tested on any of them. Six facts about the code and the data shape the design; the second, third and
fourth were worked out for this document and are not in the notebook.

**1. Half of the slow level's part is there, and was never run.** The language has `Filter(attr,value,sel)` with the
attributes `tie`, `sick` and `bud` (`program.ATTRIBUTES`), a part record has an `attributes` field (`scene.Part`), the
memory debounces attributes like every other fact (`SceneMemory._observe`), and a plan binds again when an attribute it
read changes (`Plan._stale`). Nothing sets an attribute: `scene.scene_truth` writes none, the eyes have no output for
one, and `slow_level.SlowLevel` refuses the task's own program on any task that is not `reason_*`.

**2. A part can be out of view from the first look.** In the eyes' validation frames (`scene_val_reason_mix`: the
expert driving from random start poses, 11,127 frames), the first frame of the 184 episodes that have one hides 16 of
455 shoots from the front camera (in 15 episodes) and 6 from both cameras (in 5). v2 answers a missing fact by holding
still, which works when the fact comes by itself and never otherwise: its inventory that never completes (4 to 6 of
200 episodes) is that case. An attribute is a smaller thing than a shoot and there are more of them to miss, so in v3
looking has to be something the plan can ask for, before the rung that is about looking.

**3. How large the things to see are** (a calculation from the camera's position and the meshes' sizes in
`assets/meshes/situations/manifest.json`, not a measurement). The front camera shows about 3.4 mm per pixel at the
plant, the plant crop of v2's eyes (the same camera at twice the resolution) 1.7 mm, and the wrist camera about 1 mm
from the 13 to 15 cm at which the hand comes in for a cut.

| What | Size | Front image, 224 pixels | Plant crop |
|---|---|---|---|
| a tie, drawn 1.8 times its modelled size | a band of about 20 mm with thinner tails | about 6 pixels (the catalogue's figure) | about 12 |
| a sick shoot's leaf: yellowish, (0.56, 0.60, 0.18) against (0.18, 0.50, 0.13) for a healthy one | a blade of 48 by 19 mm | 14 by 6 | 28 by 12 |
| the brown spots and edges on it | 3 to 6 mm | 1 to 2 | 2 to 4 |
| a bud at the tip of a shoot | 16 mm across, 33 mm long with its collar | 5 by 10 seen from the side; the catalogue gives 2 to 4 for what a view shows | 10 by 20 |

So "sick" as it is built is a colour over about 80 pixels (every leaf of a sick shoot is recoloured,
`scenarios.Sick.make_sick`), and its spots are not needed to read it. The bud is the small one. A shoot is 2 pixels
thick, and v2's eyes place it to 2.7 mm from this image and to 1.2 mm with the crop, so a thing of a few pixels is
within what these eyes learn; whether a bud is, is measured (E16) and not assumed.

**4. The success rules reward standing still at the home pose, and a random start is not yet the cure.**
`nothing_to_do`, and `sick` and `buds` when no shoot is meant, count an episode solved when the gripper has been within
5 cm of the home pose for one second (`scenarios.HOME_HOLD`). Evaluations start at the home pose unless told otherwise,
so a policy that hesitates for a second solves them. A random start pose (`eval.py --random_start`, as the
demonstrations have) removes that, but it can lie in the plant: with the expert driving from such poses, 1611 of 6000
`reason_mix` episodes and 575 of 2000 `prune` episodes end with the flower touched, every one of them within the first
two seconds (the episode listings of `scene_reason_mix_6k` and `scene_prune_2k`). The demonstrations drop those
episodes; an evaluation cannot. Every v3 evaluation of a situation therefore starts from a pose drawn as
`expert.random_start` draws it and kept clear of the plant, a new option whose distance is set where the expert,
starting there, solves at least 97 of 100 layouts.

**5. The built situations are scenarios on `prune`, not reasoning tasks.** They set the instruction, the shoots meant
(`env.targets`) and flags for the expert (`scenarios.ChooseShoots`, `Order`, `NothingToDo`). They have no program of
their own, no notes and no sub-goal column in their demonstrations. v3 uses them on `prune` (two or three shoots),
because the slow level names shoots by position.

**6. The fast level does not take new data well.** Objects added to the table cost the pruning policy 8 to 10 episodes
in 100 (79 to 71 and 69), a small price for a local change. But mixing other data into its training has broken it
twice: with a third of the demonstrations in varied looks it solved 5 of 100, and trained together with motor drills
22 of 100 against 79. Started from drill-trained weights and then trained on pruning alone it reached 74. So v3 changes
the fast level's training once, as late as it can, starting from the trained weights, and checks the old tasks again.

## The ladder

Each rung adds one kind of reasoning to the rungs below it and is reported apart.

| Rung | Situation and example | What a policy that does not reason does | Eyes | Language and plan | Fast level |
|---|---|---|---|---|---|
| 1 | `order` (A8): "cut the side shoots from the top down"; "cut the lowest shoot, then the highest one" | cuts the nearest shoot first; an episode ends at the first cut out of order | as v2 | `CutFrom`, `Then`; stages | as it is |
| 2 | `nothing_to_do` (A5): a plant of stubs, the task's own instruction | reaches for a stub, or stands still away from home | reads stubs on a plant never seen whole | complete with no action; the home move | as it is |
| 3 | `ties` (A1): "cut the shoot with the red tie"; "prune the tied shoot" | cuts by position | `tie` | `Filter` (there), one value added | as it is, if the ties in view do not disturb it |
| 4 | `sick` (A3), `buds` (A4): "cut the sick shoots, keep the healthy ones"; "cut every shoot without a bud" | cuts every shoot, or the usual number | `sick`, `bud` | `Filter`; nothing meant ends as rung 2 | as it is, if the looks do not disturb it |
| 5 | `compose` (A9 and compositions): "cut the highest shoot that has no bud", "cut the sick shoots from the top down", "leave exactly one shoot", "if more than one shoot is sick, cut only the lowest sick one" | has no demonstration of any of them | as rungs 3 and 4 | `Take`, `Count`, `Nothing`; the task is a program | as it is |
| 6 | `tags` (A2): every shoot has a tag that shows its colour to one side; "cut the shoot with the red tag" | cuts a shoot whose tag it has not seen | `tag` | a look at one shoot, bounded | one new sub-goal: trained once more |

Why this order. Rungs 1 and 2 need no new output of the eyes and can run before anything is trained: they are E9 of
architecture-v2.md, the kinds no policy was trained on, and they show at once whether the plan and the fast level
carry over. Rungs 3 and 4 add
perception one attribute at a time, from the largest thing to see to the smallest. Rung 5 costs code only, and it is
the rung on which an executed program should differ most from a policy that learned from examples, because nothing on
it was demonstrated. Rung 6 is the only one that needs a skill the fast level does not have, so it comes last, with
the one retraining. The plan's side of looking is built with rung 3 (fact 2), and until the fast level has the skill a
missing fact about a shoot is held for, as in v2, and counted.

Rung 5 has no demonstrations at all, and a split of its own for the writer. A first set of compositions supplies the
worked examples in which `Take`, `Count` and `Nothing` appear ("cut the lowest shoot that has a bud", "cut one shoot",
"if no shoot is sick, cut the lowest shoot, otherwise cut none"). The episodes are drawn from a second set, which
combines the same constructs in ways no example shows, in sentence frames of its own: the examples in the table above.
As in v2's split, a held-out composition is a new combination of parts that each occur in training.

The hidden tag follows the catalogue's A2 in what it tests and differs in how it is drawn. A tie is a band round the
shoot and shows from every side, so it cannot be hidden by turning. A tag is a small plate at the tie's place (18 by 14
mm, MuJoCo primitives, no new mesh), coloured on the face that looks outward along the shoot and brown on its back,
readable when that face is within 50 degrees of a camera's direction (the card's rule, `env.LABEL_VIEW`). Shoots stand
at least 100 degrees apart, so the front camera reads one tag at a time; and a shoot turned into reach points 137
degrees away from that camera, so the tag of the shoot being cut is out of its view. This is v2's hidden card, once per
shoot.

Designed and not in this plan: a legend for the card's colours (below, "Not in this plan"), most leaves (A10), which
pot (A7), balance (A12) and the legend for the buttons (A6).

## Predictions

Written before anything is built, as in v2. Single runs of 200 episodes differ by about 8 episodes by chance, so every
threshold leaves that room.

| | Prediction | Refuted if |
|---|---|---|
| P1 | With the simulator's scene and the task's program, the fast level as it is (seed 43, resets on) solves 140 to 170 of 200 `prune` episodes and 120 to 160 of 200 `order` episodes: about nine cuts in ten succeed, and an episode has two or three | `order` under 110: chaining sub-goals is then the limit, and the fast level's retraining comes before everything else |
| P2 | On `nothing_to_do` all of v3 says `Complete` without having given a sub-goal in at least 190 of 200 episodes, and says it at its first look on a plant that has a shoot to cut in at most 2 of 200 `prune` episodes | under 170: the eyes do not read a plant of stubs alone, and such frames go into their data |
| P3 | The eyes read a tie's colour on at least 98% of the parts where it can be read, sick on 97%, a bud on 85% to 93% from the two cameras and on at least 97% with the plant crop, and "cannot be read" on 95% | a bud under 90% with the crop: a narrower crop at four times the resolution is built |
| P4 | The writer's program has the right meaning for at least 95 of 100 instructions of the new kinds in training sentence frames, 85 of 100 in held-out frames and 75 of 100 held-out compositions | under 70 held out in new frames: the worked examples are the first thing to change, before any other part |
| P5 | All of v3 (the eyes, the written program) is within 15 episodes of the simulator's scene with the task's program on each of rungs 1 to 4 | a larger gap on a rung: its errors are counted from the episode records before anything is added |
| P6 | The end-to-end policy trained on the same demonstrations solves at least 80 of 100 `nothing_to_do` episodes, 35 to 60 of `ties`, 15 to 40 of `sick` and `buds`, 25 to 50 of `order`, and fewer in held-out sentence frames | within 10 points of v3 on rungs 1, 3 and 4: those rungs then do not test what they were chosen for |
| P7 | On compositions none of which was demonstrated, all of v3 solves at least 50 of 100 and the end-to-end policy at most 15 | v3 under 35 |
| P8 | Ties and buds in view cost the fast level at most 15 episodes in 200 against the same plants without them; yellow leaves may cost more | a look costs more than 30: the fast level is fine-tuned on the looks before rungs 3 and 4 are reported |
| P9 | Fine-tuned once on the old demonstrations and the new ones, the fast level stays within 15 episodes of what it was on `reason_mix`, and "show the lowest shoot" ends with that shoot facing the camera in at least 90 of 100 | a loss of more than 30 on `reason_mix`: the run from scratch decides, and if that fails too rung 6 is reported with the plan holding |
| P10 | With the look, v3 solves at least 60 of 100 `tags` episodes; holding in its place, at most 35 | |

## What the search for prior work changed

A search of the literature was made before the design was fixed (the sources are under "Prior work"; each was opened
and its abstract read). It found every ingredient in published systems and the combination in none. The nearest systems
each have a part of it: a language model's robot program verified before it runs, with probes for the observations it
lacks, over programmed skills; a symbolic plan with monitored stages over a learned policy that acts in chunks, its
parser not a language model; objects with attributes and a parsed program for manipulation, with a trained parser and
no memory of a scene that changes. It changed the design in eight places:

1. **A gate where the cut is made.** The slow level looks every 0.4 s and a cut takes three control steps (0.15 s) of
   shut jaws on a shoot, so a look can come too late. The runner therefore checks every command that would shut the
   jaws among the shoots against the part the plan has authorised (rule f below). Permission to act sits outside the
   policy that acts: tested policies of this kind moved when given no task at all, and a prompt did not stop them.
2. **Two oracles for seeing, not one.** The simulator's truth of what the cameras show now (v2's `--scene truth`)
   removes errors of recognition. The simulator's whole state, hidden facts included (`--scene full`, new), also
   removes the need to look. The difference between the two is what looking costs.
3. **"No" needs the place seen.** A shoot has no bud only if its tip was in view. Each attribute therefore has
   "cannot be read" as a class of its own, a look that fails is never a "no", and a look that runs out of its bound
   ends as `Failed`, not as nothing to do.
4. **Bound once, and compared as traces.** What a program selects is bound to part ids over the plant as it was found:
   positions (v2), and now also which shoots a "cut two shoots" took and where a rank inside a selection falls.
   Evaluating again after a cut never hands out a new budget. Programs are compared by the order of their cuts, not
   only by the final set.
5. **Looks from one angle are one witness.** More observations can sharpen a wrong belief. The memory records the
   plate's angle with every reading, and the experiment on memory compares three rules: the last look, looks in a row
   (v2), and looks from two angles.
6. **Worked examples in pairs.** Examples that cover the structures the output needs help a language model compose,
   and four of v2's eight misses were a construct in a role that none of its examples showed (a middle shoot to keep).
   The examples come in pairs that differ in one word and in the program ("except the highest", "only the highest").
   Asking twice is judged by how often two programs that agree are both wrong, not by how often they agree.
7. **The legend splits in two.** A legend that says what a colour means changes what the task is, and the plan can
   read it. A legend that says what a button does changes what an action does, and the fast level has to be told. Only
   the first is near.
8. **The success rules were read before the plan.** Benchmark checkers have reversed published rankings. Reading these
   found fact 4 above.

It also confirmed one choice: evaluating a program over what is definitely and what is possibly selected is the
established simple form of planning with facts not yet known, and full planning over beliefs is more than three shoots
need.

## The architecture

```
 every look (whenever the fast level needs a new chunk: 8 control steps, 0.4 s)
 front, wrist,   +-------------------------+ a record  +----------------------+  facts   +---------------------------+
 plant crop ---->| 1 EYES                  | per part  | 2 SCENE MEMORY       |--------->| 4 PLAN                    |
                 | as v2, and per part:    |---------->| as v2, and per part: |          | the program, evaluated in |
                 | + tie, sick, bud, tag,  | with its  | + attributes: kept   |          | every world the unknown   |
                 |   each with "cannot be  | confid-   |   when out of view,  |          | facts allow:              |
                 |   read in this view"    | ence      |   frozen at the cut  |          | + stages, in order        |
                 +-------------------------+           +----------------------+          | + a look at one shoot     |
                                                                                         | + complete with no action |
                                                                                         +-------------+-------------+
 instruction --> 3 PROGRAM WRITER, once per episode: the same language model, not                  ^   |
                 fine-tuned; + CutFrom, Then, Take, Count, Nothing; asked twice -------------------+   |
                                                                                                       v
          sub-goal: "cut the middle shoot", "show the label", + "show the middle shoot"; or hold; or reset;
          + or home; + every closing of the jaws among the shoots checked against the part authorised
 front, wrist, joint angles ---------------->+--------------------------------------+
                                             | 5 FAST LEVEL (as it is; trained once |---> 15 joint targets, 8 executed
                                             | more, late, for "show the ... shoot")|
                                             +--------------------------------------+
```

### 1. Eyes: four more outputs per part

A part slot gives its six numbers as in v2 and four classifications:

| Output | Classes | Truth in the simulator | "Cannot be read" when no camera shows |
|---|---|---|---|
| `tie` | unreadable, none, red, blue, yellow, white | `scenarios.Ties`: which tie is drawn, and its colour | the tie's place on the shoot (`TIE_AT`) |
| `sick` | unreadable, no, yes | `scenarios.Sick`: the shoots it made sick | any leaf of the shoot |
| `bud` | unreadable, no, yes | `scenarios.Buds`: the shoots that carry a bud | the shoot's tip |
| `tag` (rung 6) | unreadable, none, red, blue, yellow, white | the new `tags` scenario | the tag's face: it must look within 50 degrees of a camera |

The truth comes from the scenario's own draw, through one new hook per scenario that lists each shoot's attributes and
the points where each is to be seen. Whether a camera shows those points is decided as `scene_truth` decides whether it
shows a part: by rays from the camera, without rendering. Every attribute of a cut part is unreadable: its tie, leaves,
bud and tag fell with it. A plant without a scenario has no tie, no sick shoot and no bud, which is a valid label, so
every frame of every dataset trains every output.

Why a class and not a low confidence. The eyes of v2 report a part with probability 1.00 and judge it not visible;
the two are different facts and both are needed. "No bud" from a tip that is in view and "no bud" from a tip behind the
stem must not look alike to the memory, and a class trained on the simulator's rays keeps them apart. v2's card has
the same class and is read in 96.7% of frames (100% with the crop).

The loss adds one cross-entropy per attribute and part slot. The model is otherwise v2's (`sim/pickup/eyes.py`: the
6-layer transformer, five queries, two decoder layers), with the plant crop as a third image, and it is trained from
scratch on all frames, old and new: an eyes run costs about an hour, and starting over avoids outputs trained on
different data.

Whether the resolution is enough is an experiment (E16) with three sets of views, built only as far as needed:

1. front and wrist at 224 pixels: what the fast level also sees;
2. with the plant crop at twice the resolution (built, `scene.PlantCrop`);
3. with a narrower crop at four times the resolution (0.85 mm per pixel, 224 pixels across 19 cm, enough for the
   shoots). The plant's model has an offscreen buffer of 960 by 540, so this needs a larger buffer or the front
   camera's field of view narrowed for one render. A crop is cut from a larger render, never enlarged from the
   224-pixel image, which would add nothing.

The eyes are scored per attribute on the parts where it can be read, by how often they call an unreadable part
readable and the reverse, and by the size of the thing in pixels: a fifth of the frames of `ties` and `buds` draw the
tie or the bud at a random scale between half and twice its size, which gives accuracy against size from one model.
The wrist camera is in every set of views. It sees the shoot being cut at about 1 mm per pixel, but only once the hand
is there, after the decision; a pause at the shoot to look again before the jaws shut would need a skill the fast level
does not have, and is not in this plan.

As in v2 (E8), frames from the fast level's own rollouts are added once the first closed-loop runs show what the eyes
get wrong on the states a policy reaches. One such error can be named in advance: a cut part lies on the plate or the
pot with its tie, its leaves and its bud, next to shoots that are still on the plant.

### 2. Scene memory: attributes per part

Three rules are added to `scene.SceneMemory`; nothing of v2's changes.

- **A reading counts** when the part is seen with confidence (rule 1 of v2), the class is not "cannot be read" and its
  probability is at least `p_min`. An attribute is known after `stable` readings in a row with the same value, and
  changes only after `stable` readings in a row of another. Looks that cannot read it neither count nor break the row:
  what a tag showed is kept when it has turned away, as what the card showed is.
- **Frozen at the cut.** Once a part is marked cut its attributes no longer change. Without this rule the stub of the
  shoot that had the red tie would be read as "no tie", the plan would bind again, find the cut shoot outside its new
  selection and say `Failed`.
- **The angle is kept.** Every reading is stored with the plate's angle at that look (from the part's own bearing). An
  option, off at first, asks for readings from two angles at least 30 degrees apart before a cut may rest on an
  attribute. In most episodes the plate has to turn anyway (the first shoot is out of reach in 768 of 1200 episodes of
  v2), so the second angle is often free.

A second option concerns what counts as a cut: a part first seen whole may be marked cut only if the jaws have shut at
its height since (the gate of rule f keeps that record). It cannot remove v2's cut that did not happen, where the jaws
did shut beside the shoot, but it removes a cut reported of a part the hand was never at. A part found cut at its
first sighting, as on a pruned plant, needs no such witness.

### 3. The program language

v2's language, with what v3 adds marked:

```
stmt  := Cut(sel) | If(cond,stmt,stmt)
       | CutFrom(end,sel) | Then(stmt,stmt) | Nothing()                                    new
sel   := All() | Rank(k,end) | Middle() | Except(sel,sel) | Filter(attr,value,sel) | ReadLabel()
       | Take(k,pick,sel)                                                                  new
cond  := FlowerIs(colour)
       | Count(cmp,n,sel)                                                                  new
end   := top | bottom          pick := top | bottom | any          cmp := exactly | more | fewer
k     := 1 | 2 | 3             n := 0 | 1 | 2 | 3
attr,value := tie,red | tie,blue | tie,yellow | tie,white | tie,any | sick,yes | sick,no | bud,yes | bud,no
            | tag,red | tag,blue | tag,yellow | tag,white                    (tie,any, sick,no and tag are new)
```

| Construct | Meaning |
|---|---|
| `CutFrom(end,sel)` | cut the shoots of `sel` one at a time, starting at that end: `CutFrom(top,All())` is "from the top down" |
| `Then(s1,s2)` | carry out `s1`, then `s2`; a shoot `s1` has cut is not asked for again |
| `Nothing()` | cut no shoot: the branch of an `If` that asks for nothing |
| `Take(k,pick,sel)` | k shoots of `sel`: the k highest, the k lowest, or any k (all of `sel` if it has fewer) |
| `Count(cmp,n,sel)` | true when `sel` has exactly n shoots, more than n, or fewer than n |

The bounds keep the language finite, as in v2: `Except`, `Filter` and `Take` nest at most three deep in a selection
(two in v2), a `Then` holds at most three statements, and the branch of an `If` is any statement but an `If`.

Programs of the situations as built, and of rung 5:

| Instruction | Program |
|---|---|
| cut the side shoots from the top down | `CutFrom(top,All())` |
| cut the lowest shoot, then the highest one | `Then(Cut(Rank(1,bottom)),Cut(Rank(1,top)))` |
| cut the side shoots off the stem, keep the flower (on a pruned plant too) | `Cut(All())` |
| cut the shoot with the red tie | `Cut(Filter(tie,red,All()))` |
| prune the tied shoot | `Cut(Filter(tie,any,All()))` |
| cut the sick shoots, keep the healthy ones | `Cut(Filter(sick,yes,All()))` |
| cut every shoot without a bud | `Cut(Filter(bud,no,All()))` |
| cut the shoot with the red tag | `Cut(Filter(tag,red,All()))` |
| cut the highest shoot that has no bud | `Cut(Take(1,top,Filter(bud,no,All())))` |
| cut the sick shoots from the top down | `CutFrom(top,Filter(sick,yes,All()))` |
| leave exactly one shoot | `Cut(Except(All(),Take(1,any,All())))` |
| cut two shoots | `Cut(Take(2,any,All()))` |
| if more than one shoot is sick, cut only the lowest sick one; otherwise cut none | `If(Count(more,1,Filter(sick,yes,All())),Cut(Take(1,bottom,Filter(sick,yes,All()))),Nothing())` |

**What is not a construct.** There is no `Stop` and no `Look`. An instruction does not say either; both follow from
evaluating the program on the plant. "Cut the side shoots" on a plant of stubs selects shoots that are all cut, "cut
the sick shoots" on a healthy plant selects none, and in both the plan says `Complete` at its first look with every
fact it needs. A look follows from a fact that is missing. This is v2's rule that unknowns are outcomes, applied to one
shoot at a time, and it is why five constructs are enough.

**What a selection ranges over** is fixed once, as in v2: the plant as it was found, every part, cut or not. "The
highest shoot" of a plant whose highest shoot is a stub is that stub, and the program is complete. `Take` with `top`
or `bottom` ranks inside its selection. `Take` with `any` is the plan's choice, made once when the program is bound:
among shoots to cut it takes those that need the least turning, and inside an `Except`, where they are the shoots to
keep, those that need the most. The choice is kept as part ids, so a "cut two shoots" that has cut one does not choose
two again.

**How the language model writes it.** As in v2 (`sim/pickup/program_writer.py`): the language model of Qwen3-VL-2B as
downloaded, a description of the language construct by construct, worked examples from training sentence frames only,
greedy decoding in which only the tokens that keep the text a prefix of some program are allowed. The acceptor is
generated from the grammar table (`program.GRAMMAR`), so the new constructs need rules in the table and no new
mechanism. Three things change:

- *Sixteen worked examples in place of eight*, every construct in every role it can take, and in pairs that differ in
  one word and in the program: "every shoot except the highest" and "only the highest"; "from the top down" and "the
  top one"; "without a bud" and "with a bud"; "the sick shoots" and "the lowest sick shoot". v2's writer compiled "keep
  the middle one" to `Cut(Middle())`, and none of its eight examples kept a middle shoot.
- *Asked twice.* The writer is run with two disjoint sets of examples. If the two programs are not equivalent the
  slow level does not act (`Failed`, "the instruction is read two ways"). Measured: how many wrong programs this stops,
  how many right ones it blocks, and how often two programs that agree are both wrong.
- *The constraint is measured again.* With eight examples v2's writer wrote the same programs with and without the
  constraint. A larger language gives the constraint more to do, and masking tokens is known to shift a model's
  preferences among the outputs it still allows, so both are run again.

**Evaluating the writer alone**, as in v2 and without any image: format (it parses; always so under the constraint),
exact (its text is the task's program), meaning (equivalent to the task's program) and resolved (right on the plant
drawn), by kind and by split: training frames, held-out frames (`order_heldout`, `ties_heldout`, `sick_heldout`,
`buds_heldout` exist), and held-out compositions (rung 5). Equivalence is decided on every plant the benchmark can
draw: two or three shoots, every flower and card, and every assignment of the attributes that either program reads.
Two programs are equivalent when they allow the same sequences of cuts on each of them, so `CutFrom(top,All())` and
`Cut(All())` differ, and a `Take` with `any` equals another only if both allow the same choices.

### 4. The plan and the runner

**Worlds.** A fact a program reads can be unknown: the flower's colour, the card, an attribute of one shoot. v2 stops
at the first unknown fact. v3 evaluates the program in every *world* the unknown facts still allow, a world being one
value for each of them (three shoots and one attribute with five values give at most 125), and compares the results:

- a shoot that every world cuts next is cut now: `Execute`;
- if every world has nothing left to cut: `Complete`;
- if every world names a shoot the plant does not have, or has a shoot cut that was to stay or out of its order:
  `Failed`;
- otherwise the worlds differ, and the plan needs one of the facts on which the result depends: `NeedObservation`,
  with the fact and the part.

On a plant whose lowest shoot is known to be sick, whose middle one cannot be read and whose highest is healthy, "cut
the sick shoots" cuts the lowest now: it is cut in both worlds. The turn that brings it into reach shows the middle
shoot from other angles, and only if its leaves are still unread afterwards does the plan ask for a look. With "cut
the highest sick shoot" the plan asks first, because the two worlds cut different shoots. The plan never acts on a
guess, never looks for a fact that cannot change what it does, and never takes a fact it could not get for a "no".

**Stages.** A program evaluates to stages: sets of parts in the order in which they are to be cut. `Cut(sel)` is one
stage, `CutFrom` one stage per shoot, `Then` the stages of its first statement and then of its second. Inside a stage
the plan takes the shoot that needs the least turning, as v2 does; a stage is open until the memory has seen its parts
cut. A part seen cut that belongs to a later stage is `Failed`, "cut out of order". If the memory unmarks a cut (its
`revoke` option), the plan is back at that part's stage at the next look, because it is evaluated at every look. In v2
a cut unmarked late cost time. Here it costs the episode if the next stage's cut has been made meanwhile, and that is
counted (`out_of_order` episodes whose record holds a cut unmarked).

**The decisions of a look**, with what is new:

| Outcome | Sub-goal | New in v3 |
|---|---|---|
| `Execute` | `cut the <position> shoot` | the part is cut next in every world and belongs to the open stage; it becomes the part authorised (rule f) |
| `NeedObservation`: the plant, the flower | none: hold | |
| `NeedObservation`: the card | "show the label" | |
| `NeedObservation`: an attribute of a part | `show the <position> shoot`, or hold where the fast level has no such skill (`--inspect hold`) | new; bounded by one full turn of the plate from when the need arose, then `Failed`, "cannot see the tag of the middle shoot" |
| `Complete` | none: the home move | said before any sub-goal was given, it is recorded as `complete_without_action` |
| `Failed` | none: the home move | new reasons: cut out of order; cannot see; the instruction is read two ways |

The fast level always hears a position. It is never told of a tie or a bud: the slow level turns any reference into
"the middle shoot". That is what lets rung 5 run without a demonstration.

**The home move.** On a verdict the runner takes the arm to the home pose with the scripted move that the reset of v2
already makes (`slow_level.Recovery` with `then_home`) and holds it there. Going home is a fixed joint configuration
reached by a controller, not a skill to learn, and it is checked on the robot's own joint angles. If the verdict is
withdrawn, the fast level starts again from the home pose, as after a reset.

**Rule f: the cut gate.** Rules a to e are v2's (`slow_level.Supervisor`). At every control step, not every look, the
runner checks a command that would shut the jaws while the gripper is among the shoots (near the stand's axis and in
the band of heights where shoots grow, from the robot's own kinematics). The memory knows each part's height, and
shoots leave the stem at least 3.5 cm apart, so the height of the jaws says which part they are at. If that is not the
part authorised, if the part authorised does not face the robot, or if no part is authorised (during a hold, a look or
after a verdict), the jaws stay at the cutting width and the episode records a veto; a veto asks for a reset. The gate
uses the joint angles, the memory and two constants of the setup (where the stand is, how high the stem's base), and
nothing of the simulator's state. It cannot stop a cut the slow level wrongly authorised (a wrong program, an attribute
misread). It stops the cut the fast level makes on a shoot the sub-goal did not name: 7 and 8 episodes of 200 in v2
without resets, 0 and 1 with them. It is built with the runner, reported on and off, and left on only where the first
runs (E13) show more than 2 such cuts in 200.

**Checks of outcomes.** Every sub-goal has an effect it is given for, and only the memory closes it (v2's ledger):

| Sub-goal | Given when | Effect expected | If it does not come |
|---|---|---|---|
| cut the X shoot | X is cut next in every world, its stage is open | X seen cut, no other part newly cut | the reset (rule e), a veto (rule f); another part cut: `Failed`; the cut unmarked: the plan returns to X |
| show the X shoot | a fact about X is missing and the result depends on it | that fact known | after one full turn: `Failed`, never a "no" |
| show the label | as v2 | the card read | as v2 |
| home | a verdict | the joints at the home pose | none needed |

**Two oracles.** `--scene truth` is the simulator's truth of what the cameras show: an attribute behind the stem is
unreadable in it too. `--scene full` is new: every attribute of every part from the first look. With the task's own
program both give the fast level's ceiling, and they differ by what looking costs.

### 5. The fast level

In short: it is used unchanged on rungs 1, 2 and 5; whether the looks of rungs 3 and 4 need it trained again is
measured first; rung 6 needs one new kind of sub-goal, with demonstrations and one more training. Returning home is
not a sub-goal.

Its sub-goals in v3: "cut the lowest, middle or highest shoot" and "show the label" as before, and three new ones,
"show the lowest, middle or highest shoot": turn the plate until that shoot faces the front camera. There is no "go
home", and "stop" is no longer given: on a verdict the runner takes the arm home.

- **Rungs 1, 2 and 5** use the old sub-goals on plants that look as they always did. The fast level is used as it is
  (`dp_vit_lang_reactive_subgoal_reason8k_20m_seed43`, final averaged weights, with seed 44 as its repetition).
- **Rungs 3 and 4** use the old sub-goals on plants that look different. What a look costs is measured before anything
  is trained, with a paired control: the same layouts and targets with the ties, the yellow leaves or the buds left
  undrawn (an option of the scenario; the simulator's scene does not need to see them). Up to 15 episodes in 200 the
  fast level stays; beyond that it is fine-tuned on the looks (P8).
- **Rung 6** needs the new sub-goal, so new demonstrations and one more training.

The retraining, once, in step 11: fine-tuned from the trained weights of seeds 43 and 44 on the old demonstrations and
the new ones together (`train.sh` takes datasets joined by `+`, their weights, and `PICKUP_STATS_DATASETS` to keep the
old normalisation), 5M samples each; and one run from scratch on the same data, 20M samples, as the comparison. Fact 6
is the reason for the order: the fine-tuned fast level is judged first on `reason_mix` under a slow level without
errors, where it has to stay within 15 episodes of what it was, and only then on the new tasks.

The fast level stays the ceiling. An `order` episode chains two or three sub-goals, each of which fails about one time
in ten, so rung 1 is expected at 120 to 160 of 200 with a slow level that makes no error (P1). The fast-level track of
architecture-v2.md is where that ceiling is worked on; nothing here raises it.

**What stays the same.** The reasoning tasks of v2, their programs and held-out split; every rule of v2's memory, plan
and supervisor for a program of v2 whose facts are known; the reset; the fast level's weights and every existing
config. The eyes are trained again with more outputs and are held to v2's accuracy on the old ones. Every run of
architecture-v2.md can still be repeated with the settings it names.

## Data to collect

Collection runs in simulator processes on CPU cores and renders as the existing collections do. Times are scaled from
v2's (8000 expert episodes as 534,043 frames in 21 minutes with 48 processes) and are estimates. Every dataset keeps
the leaves as the existing policies and eyes saw them (`leaf_fix` off); the demonstrations are collected with
`collect.py --park_reset`, as test-situations.md asks of new datasets.

| Dataset | What | Episodes | Size (estimate) | Time (estimate) | For |
|---|---|---|---|---|---|
| census | the expert on each situation, truth only, no images | 1000 per situation | a few MB | minutes | step 2: how often each attribute can be read |
| `scene_tend_12k` | perception frames (`collect_scene.py --crop`, every fourth step, tails): `reason_mix` 3000, `prune` 1000, `ties` 2000, `sick` 2000, `buds` 2000, `order` 500, `nothing_to_do` 500, and `tags` 1000 at rung 6 | 12,000 | about 1M frames, 35 GB | 45 minutes with 48 processes | the eyes |
| `scene_val_tend` | the same from layouts used nowhere else (seeds from 900,000), 100 per task | 800 | about 60,000 frames | 3 minutes | the eyes' validation |
| `scene_policy_tend_2k` | frames of the fast level's own rollouts under v3 (`eval.py --record_scene`) | 2000 | about 180,000 frames | ten evaluation runs | the eyes, after the first closed-loop runs |
| `tend_mix_6k` | demonstrations with sub-goals, random starts, a tail of 10: `order` 1500, `ties` 1500, `sick` 1250, `buds` 1250, `nothing_to_do` 500 | 6000 kept of about 8600 tried | about 2.5M windows | 1 hour, then preprocessing | the end-to-end baseline; the fast level's retraining |
| `tags_2k` | demonstrations of the hidden tag, with "show the ... shoot" sub-goals | 2000 kept | about 1M windows | 25 minutes, then preprocessing | the fast level's retraining |
| writer's test | instructions drawn as the situations draw them, 100 per kind and split; no images | | | seconds | the writer alone |

The sub-goal of a frame is written by the expert, which knows what it is doing (a new `ScriptedExpert.subgoal`: the
cut of a shoot by its position, a look at a shoot, or "stop" in the tail), and a test holds it to what the plan says
on the simulator's truth for the same frames. Held-out sentence frames and the compositions of rung 5 are never
collected (`collect.py` refuses them, as it refuses `order_heldout` now).

## Evaluation

Layouts as in v2: 200 evaluation layouts per cell (seeds from 1,000,000), variants compared on validation layouts
(seeds from 900,000) with one setting fixed before the evaluation layouts are touched. Every situation starts away
from the home pose and clear of the plant (fact 4); `reason_mix` is run once from such starts too, so that the new
numbers can be set beside v2's, which started at the home pose. A difference under about 15 episodes between two runs
of 200 is not a finding.

**Systems compared**, on every rung and by task kind:

| System | What it isolates |
|---|---|
| the end-to-end small policy (`dp_vit_lang_reactive`), trained on the same demonstrations, reading the instruction | what a policy of this size learns without a slow level |
| v3, the whole state, the task's program (`--scene full --program gold`) | the fast level's ceiling with nothing to look for |
| v3, the simulator's scene, the task's program | what looking costs |
| v3, the simulator's scene, the written program | what the writer costs |
| v3, the eyes' scene, the task's program | what perception costs |
| all of v3 (the eyes, the written program) | |
| all of v3 without resets, and without the gate | how much rests on the runner's scripted moves |

v2 as built is not a row: its language cannot say an order, and its eyes have no output for an attribute. The first
design (a fine-tuned vision-language model that writes notes) is not trained on the new tasks; it would need notes for
them and three to seven GPU-hours a run, and v2 has settled that comparison.

**What is reported**, beside episodes solved:

- by kind, by mode (`order`: top down, bottom up, a pair), by the number of shoots and of cuts meant, and by split:
  training frames, held-out frames, held-out compositions;
- wrong cuts apart from everything else, per episode and per cut made, each with its cause from the record: the
  program's meaning, an attribute misread, the order, or the fast level cutting a shoot the sub-goal did not name;
- episodes with nothing to cut: `Complete` said with no sub-goal given, no cut and no touch of the plant, and the home
  pose reached; and the reverse error, `Complete` said on a plant that had a shoot to cut;
- false `Complete` among the completions declared and among all episodes; false `Failed`; verdicts withdrawn;
- looking: looks asked for, the plate's turn spent on them against the least that the whole state says was needed,
  episodes that ended holding for an attribute, episodes that ran out of the bound;
- the memory against the simulator at every look, per attribute: right, not yet known, wrong; and wrong at the look
  that authorised a cut;
- vetoes of the gate, resets, and episodes solved after either;
- control steps per solved episode.

**The experiments**, numbered on from v2's. E9 of v2 (kinds no policy was trained on) is E13 here.

| | Experiment | Tests | Step |
|---|---|---|---|
| E13 | v3 on the simulator's scene with the task's program and the fast level as it is: `prune`, `order`, `nothing_to_do`, `ties`, `sick`, `buds`; each look drawn and not drawn; the whole state against the simulator's scene | P1, P8: the fast level's ceiling per rung; what a look costs it; what looking costs | 4 |
| E14 | The census: how often each attribute can be read at the first look, at some look before the first cut, and never | fact 2: how much looking rungs 3 and 4 need | 2 |
| E15 | The writer alone on the new kinds: constrained and free, by kind and split; eight examples against sixteen in pairs; asked twice | P4 | 5 |
| E16 | The eyes: accuracy per attribute, by views (two cameras; with the crop; a narrower crop if needed) and by size in pixels; three seeds | P3 | 6 |
| E17 | All of v3 on rungs 1 to 4: the four combinations of scene and program; the closed-loop errors counted from the records; the memory with the last look, looks in a row, and two angles | P2, P5 | 7 |
| E18 | The end-to-end baseline on the new demonstrations | P6 | 8 |
| E19 | Compositions and counting, none demonstrated: v3 and the baseline | P7 | 9 |
| E20 | The fast level trained once more: fine-tuned against from scratch; the old tasks again; "show" alone | P9 | 11 |
| E21 | The hidden tag: with the look and holding in its place; the whole state against the simulator's scene | P10 | 12 |

## Implementation plan

Twelve steps, each for one agent in its own worktree, each ending with tests that pass and, where it has runs, a table
in the lab notebook. GPU-hours are for one L40S (48 GB). A training run is counted alone on its GPU: the small policy
at about 1100 samples per second (20M samples with its checkpoints evaluated: about 5.5 hours), the eyes at about 750
with the crop (3M samples: about 1.1 hours). An evaluation of 200 episodes is counted as 0.2 GPU-hours: in v2's result
files such a run took 16 to 44 minutes, 26 at the median (`wall_time_s` of the 53 runs of the seed-43 fast level under
v2; computed for this document), with its servers on a GPU that other work shares.

| Step | What | After | GPU-hours |
|---|---|---|---|
| 1 | The language | | 0 |
| 2 | The simulator's truth with attributes; the census (E14) | | 0 |
| 3 | The memory, the plan and the supervisor | 1, 2 | 0 |
| 4 | The runner on the situations; the fast level's ceiling per rung (E13) | 3 | 5.5 |
| 5 | The writer (E15) | 1 | 0.5 |
| 6 | The eyes (E16) | 2 | 5 |
| 7 | All of v3 on rungs 1 to 4 (E17) | 4, 5, 6 | 10 |
| 8 | Demonstrations with sub-goals; the end-to-end baseline (E18) | 2, 3 | 7.5 |
| 9 | Compositions and counting (E19) | 7, 8 | 1.5 |
| 10 | The hidden tag: the situation and its expert | 2 | 0 |
| 11 | The fast level, once more (E20) | 6, 8, 10 | 15.5 |
| 12 | Rung 6 and the final table (E21); the notebook | 7, 9, 11 | 7 |
| | | | about 53 |

Steps 1 and 2 can start together; 5, 6 and 10 can run beside 3 and 4, and 8 beside 4.

**Step 1. The language.** `sim/pickup/program.py`, `tests/essential/sim/test_program.py`.
*Build:* the nodes `CutFrom`, `Then`, `Nothing`, `Take`, `Count`; the pairs `tie,any`, `sick,no` and `tag`; their rules
in `GRAMMAR` with the bounds; evaluation on a fully known plant to stages, with the choices an `any` leaves;
`equivalent` over plants with attributes, comparing the sequences of cuts allowed; `from_scenario(env)`, the program of
a `prune`, `order`, `nothing_to_do`, `ties`, `sick` or `buds` episode from what the scenario drew.
*Done when:* v2's tests pass unchanged; `parse(text(p)) == p` and the acceptor agrees with `parse` on 10,000 random
programs and on texts made by changing one character of them; on 300 layouts of each situation the program of
`from_scenario` selects `env.targets`, in order for `order`; `equivalent` tells `CutFrom(top,All())` from `Cut(All())`.

**Step 2. The truth with attributes, and the census.** `sim/pickup/scenarios.py`, `scene.py`, a new `census.py`.
*Build:* a hook per scenario that gives each shoot's attributes and the points where each is seen; `scene_truth`
writes them per part, "cannot be read" where no camera shows the points and on cut parts; the option of `ties`, `sick`
and `buds` that leaves the look undrawn; `census.py`, which lets the expert drive 1000 layouts of a situation without
rendering and counts, per attribute, readable at the first look, at some look before the first cut, never.
*Done when:* the truth's attributes equal the scenario's draw on 200 layouts of each; without a scenario `scene_truth`
gives the same JSON as before on 50 seeds; the datasets on disk still load; E14's table is in the notebook.

**Step 3. The memory, the plan and the supervisor.** `scene.py` (`SceneMemory`), `program.py` (`Plan`),
`slow_level.py` (`Supervisor`), their three test files. No simulator: scripted percepts, as v2's tests.
*Build:* the memory's three rules and two options; the plan over worlds, with stages, `NeedObservation` for a part
and its sub-goal, the new reasons of `Failed`; the supervisor's bound on a look, `--inspect hold|show`, the home
request on a verdict, the gate's rule as a function of (jaws' command, gripper's height, the part authorised), the
ledger's entries for a look and for home, the counters.
*Done when:* a scripted test for each rule (an order kept and broken; a cut unmarked returns the plan to its stage; a
plant of stubs is complete at the first settled look; an unknown attribute asks for its part, and a shoot cut in every
world goes first; one full turn ends in `Failed`; a stub read as "no tie" does not bind again); and a brute-force test:
for 2000 random programs and random partial knowledge, an `Execute` names a shoot that is cut next in every completion
of the unknown facts, and `Complete` is said only where every completion is complete. For the programs of v2 the
decisions equal those of v2's plan, kept in the test as the reference, look for look on 2000 scripted sequences of
percepts.

**Step 4. The runner on the situations; E13.** `sim/pickup/eval.py`, `slow_level.py` (`SlowLevel`),
`scripts/eval_v2.sh`, `summarize.py`.
*Build:* `--program gold` on a situation (`from_scenario`); `--scene full`; start poses clear of the plant; the home
move; the gate at every control step; `results.json` by situation, mode, shoots and cuts meant, with the counters of
"What is reported".
*Runs:* seed 43 on validation layouts, then seeds 43 and 44 on evaluation layouts: `prune`, `order` (300 episodes, 100
per mode), `nothing_to_do`, `ties`, `sick`, `buds`; each look undrawn, and each in the whole state; the gate on and
off for `prune` and `order`; `reason_mix` once from the new start poses.
*Done when:* the expert solves at least 97 of 100 layouts of each situation from the new start poses; E13's table is in
the notebook, with the decisions it makes written down: whether a look needs the fine-tuning before rung 3 (P8),
whether the gate stays on, and P1.

**Step 5. The writer; E15.** `sim/pickup/program_writer.py`, its test.
*Build:* the two sets of compositions of rung 5 with their sentence frames (pure Python, beside the kinds of
`reasoning.py`); the description of the new constructs (in words that no held-out frame uses, as v2 holds its prompt
to); sixteen worked examples in pairs, chosen so that every construct appears in every role; `--twice`; the evaluation
over the new kinds and the three splits.
*Done when:* no example uses a held-out frame or composition (tested, as in v2); the result files give format, exact,
meaning and resolved by kind and split for the constrained and the free writer, for eight and sixteen examples, and
the three counts of asking twice.

**Step 6. The eyes; E16.** `sim/pickup/eyes.py`, `eyes_data.py`, `train_eyes.py`, `collect_scene.py`, their tests.
*Build:* the four outputs, their loss terms and metrics; the percept with a probability per attribute; the drawn size
varied in a fifth of the `ties` and `buds` frames; `scene_tend_12k` without `tags`, and `scene_val_tend`.
*Runs:* two cameras, and with the crop (1.1 GPU-hours each); the narrower crop only if P3 asks for it; two more seeds
of the views chosen.
*Done when:* E16's table is in the notebook; the old outputs are within half a point of `eyes_vit_fwp_2m` on
`scene_val_reason_mix` (the count of three-shoot plants, cut, the card).

**Step 7. All of v3 on rungs 1 to 4; E17.** No new code beyond what the runs show to be missing.
*Runs:* the eyes with the task's program on validation layouts; the errors counted from the episode records, as v2
found its two; the memory's three rules compared there; 2000 rollouts recorded and the eyes trained again with them if
a counted error calls for it; then one setting on the evaluation layouts: the five situations in the four combinations
of scene and program, and `order`, `ties`, `sick` and `buds` in held-out frames.
*Done when:* E17's tables are in the notebook, P2 and P5 answered, and each closed-loop error has a count.

**Step 8. Demonstrations with sub-goals; the end-to-end baseline; E18.** `sim/pickup/expert.py`, `collect.py`,
`scripts/make_dataset.sh`.
*Build:* `ScriptedExpert.subgoal`; the sub-goal column for situations; `tend_mix_6k`.
*Runs:* `dp_vit_lang_reactive` on `tend_mix_6k`, reading the instruction, 20M samples; its final weights on each
situation, in training and held-out frames.
*Done when:* the sub-goal column equals what the plan says on the simulator's truth in 200 episodes; E18's table is in
the notebook. A second seed costs 5.5 GPU-hours more.

**Step 9. Compositions and counting; E19.** `sim/pickup/scenarios.py`, `program.py`.
*Build:* the scenario `compose`: it dresses a plant with ties, sick leaves and buds together, draws a program from
the held-out set of compositions (step 5) and its instruction from their sentence frames, and takes its success rule
from the program: the cuts made are a sequence the program allows on the true plant, with no damage; nothing to cut
ends at the home pose. `collect.py` refuses it.
*Done when:* the expert, given the program's targets, solves 97 of 100; on every three-shoot plant the rule accepts
every sequence the program allows and no other (brute force); E19's table is in the notebook.

**Step 10. The hidden tag.** `sim/pickup/scenarios.py` (`tags`, `tags_heldout`), `expert.py`, `scene.py`.
*Build:* the plate on each shoot and its colours; when its face can be read; the instruction and its held-out frames;
the expert's look: it turns the plant to the nearest tag it has not read until it has read the one named, then cuts;
the sub-goals of its frames.
*Done when:* the expert solves 97 of 100 in training and held-out frames; the tag named cannot be read at the first
look in at least 60 of 100 layouts and at the cut in at least 90, or the situation does not test what it is for.

**Step 11. The fast level, once more; E20.** Runs with the fast level's own config
(`dp_vit_lang_reactive_subgoal.yaml`); no new code.
*Runs:* `tags_2k`; the fast level fine-tuned from seeds 43 and 44 on `reason_mix_8k_rs_tail10+tend_mix_6k+tags_2k`, 5M
samples each (1.4 GPU-hours each), and from scratch, 20M (5.5); the end-to-end baseline of step 8 fine-tuned on
`tend_mix_6k+tags_2k`, 5M (1.4), so that rung 6 has its baseline; the eyes with `tag` (1.1); each fast level under a
slow level without errors on validation layouts: `reason_mix`, `prune`, the situations, and "show" alone.
*Done when:* E20's table is in the notebook and one fast level is chosen on the validation layouts, by the rule of P9.

**Step 12. Rung 6 and the final table; E21.**
*Runs:* `tags` with the look and with holding, in the whole state and in the simulator's scene; then every rung once
on the evaluation layouts with the setting fixed beforehand, both seeds, training and held-out frames.
*Done when:* the notebook has a section for v3, this document has its "First results" with every prediction answered,
and the roadmap says where it stands.

## As built: steps 1 to 3

Steps 1 to 3 are in `sim/pickup/program.py`, `scene.py`, `scenarios.py`, `census.py` and `slow_level.py`, with their
tests. Nothing of v2 changes by default: v2's language stays the default of `parse`, the acceptor and `equivalent`, so
v2's writer is constrained as before; `program.Plan` is v2's plan and stays what a v2 program runs with; the truth of a
scene without a situation that draws attributes is byte for byte what it was (checked on 300 frames of 100 episodes);
a v2 record has no new field. The census (E14) is in the [lab notebook](lab-notebook.md). Where this design left a
choice open, or could not be followed as written, the simplest faithful option was taken:

- **Two languages.** v3's language is a separate grammar (`GRAMMAR_V3`, `parse(text, "v3")`, `next_chars(prefix,
  "v3")`) and v2's is kept for v2's writer, whose output a larger grammar would change. A node holds to v3's bounds
  (three deep), so the one test of v2 that built a three-deep selection and expected it refused now checks that v2's
  `parse` refuses it; every other test of v2 passes unchanged.
- **Then** takes two statements and nests to the right, `Then(a,Then(b,c))`, never `Then(Then(a,b),c)`, so that a
  sequence has one spelling. "The branch of an If is any statement but an If" is read as "has no If anywhere in it".
  `tie,any` is a tie of any colour; the `tag` pairs are in the language, but the truth does not write a tag until the
  hidden-tag situation exists (step 10).
- **The plan of v3 is a class of its own**, `program.WorldPlan`; the supervisor uses it for a program outside v2's
  language, or when told (`v3=True`). "A shoot that every world cuts next" is a part in the open stage of every world
  (inside a stage the order is free), the one of them that needs the least turning. A fact is asked for when its value
  changes the outcomes the worlds allow; where only two facts together decide, the first one read that can be seen.
  For v2's programs the decisions are v2's plan's (2000 scripted sequences of the plan, 200 through the memory and the
  supervisor), but where v2 held for a fact that cannot change the outcome: there the world plan acts. `rebinds` counts
  changes of the stages every world agrees on.
- **A fact that cannot be had.** An attribute of a part already cut that was never read can no longer be seen; if it
  decides, the plan says `Failed` ("cannot see the tie of the lowest shoot") at once, not after a look.
- **The look's bound** is one full turn of the plate, measured from the memory's bearings (the median change of the
  parts seen at two looks in a row), or 360 control steps (two turns at the buttons' speed) where the plate does not
  turn, as under `--inspect hold`, which would otherwise wait for ever. A look that ran out is final.
- **The verdict** keeps the sub-goal "stop" inside the supervisor and sets `Decision.home`; the runner (step 4) takes
  the arm home on it in place of telling the fast level "stop". The home move has its attempt in the ledger, confirmed
  when the runner reports the arm at the home pose (`step(..., home=True)`).
- **The gate** is `slow_level.cut_gate(jaws, height, among, parts, authorised)`: a command below the cutting width
  among the shoots is allowed only where the nearest remembered part by height (within 1.75 cm) is the part authorised
  and faces the robot within 45 degrees. A veto is counted once per run of control steps and asks for a reset within
  the resets allowed (shared with rule e). The memory's heights are where a shoot leaves the stem, and the jaws close
  2 to 5 cm out along a shoot that rises, so the height the runner passes has to be mapped onto the stem from the
  kinematics; step 4 measures that mapping before it turns the gate on.
- **The memory**: a sighting that says the part is cut reads none of its attributes (a stub has none), and an
  UNREADABLE class or a probability under `p_min` neither counts nor breaks a row. The three rules of E17 are
  `attribute_rule` "last", "row" (the default) and "angles" (a row whose readings span 30 degrees of bearing).
- **Where an attribute is seen** (`scene.attribute_views`): a tie's place is three points on the shoot's axis around
  `TIE_AT` (two must be seen), a bud's three points out along the axis from the tip, where a bud stands or would stand
  (two must be seen), sick leaves the centre of any leaf as drawn. A leaf is the shoot's own only for its own sickness:
  in front of the tip it hides a bud as anything else would. The truth writes attributes where a situation that draws
  them applies (or when asked: `scene_truth(env, attributes=True)`); `scene_truth(env, full=True)` is the second
  oracle (every part shown, every attribute and the card's colour given).
- **Undrawn** is a spec option, `ties+undrawn`, `sick+undrawn`, `buds+undrawn`: the episode, its draw and its targets
  are the drawn one's; a bud undrawn is neither seen nor touched, and its shoot keeps the two leaves posed as with a
  bud.
- **The census** starts at the home pose (clear of the plant, as v3's evaluations will start; the new start option is
  step 4's) on validation layouts, and once more from the demonstrations' random starts.

## Risks

- **The fast level.** It caps every rung, more so where sub-goals are chained (rung 1) and where a look comes before
  the cut (rung 6). The new looks may disturb it, and its training has not survived mixed data twice (fact 6). The
  plan measures before it trains, trains once, and keeps the old weights; if the retraining fails, rungs 1 to 5 stand
  on the old fast level and rung 6 is reported with the plan holding.
- **Errors that single frames do not show.** v2's eyes were right in 97% to 99% of frames and lost episodes to two
  errors that only the closed loop showed. Here a cut part lies next to the plant with its tie, leaves and bud; a bud
  shoot always has two leaves and a shoot with one never has a bud, so the eyes can learn the leaves and not the bud;
  and the leaves are drawn as the old policies saw them, about 29 mm from where they are, which may hide a bud. The
  census and the first closed-loop runs are placed before any tuning for that reason.
- **A wrong cut cannot be taken back.** A wrong program or an attribute misread authorises a cut that no gate stops.
  The guards are those the design has: an attribute needs readings in a row, optionally from two angles; a "no" needs
  the place seen; the writer is asked twice. Each is measured by what it stops and what it blocks.
- **The writer on a larger language.** More constructs are more ways to write a well-formed program with another
  meaning, and the prompt is about twice as long. `Take` against `Rank`, `any` against `top`, and which statement of an
  `Except` is kept are the expected errors.
- **Order and a cut that did not happen.** A cut marked falsely moves the plan to the next stage. If it is unmarked
  after the next cut, the order is broken for good.
- **The benchmark's reach.** A plant has at most three shoots, so counting stays small. The compositions are, by
  construction, those the language can say; an instruction outside it cannot be compiled, as in v2. Attributes are
  written by hand, and each needs a field and labels, which are free only in a simulator.
- **Scripted moves.** The reset, the home move and the gate are code, and the first two use the scripted expert. Every
  result is reported with and without them.
- **One run each.** As in v2. Where a decision rests on a difference under 15 episodes, the run is repeated.

## Not in this plan

- **A legend for the card's colours** (the near half of A6). A card on the table shows, per episode, which colour
  names which position. The plan already takes that rule as a table (`Plan(label_rule=...)`); the eyes would read it as
  one more output and `ReadLabel()` would wait for it. The test is to swap the legend and hold the rest of the scene:
  the cut has to change. No change to the fast level. The first rung after these six.
- **The legend for the buttons** (A6 as catalogued). It changes what a press does, so the fast level has to be told
  which button. The fast level told its aim exists and lost as many episodes as it gained (architecture-v2.md).
- **Most leaves or longest (A10), which pot (A7), balance (A12).** They need a number per part and its maximum, a
  wider bench and names for plants, and a notion of side that turns with the plate.
- **Uniqueness.** "The shoot with the red tie" presupposes one. A construct that fails when the selection has two
  would catch an eyes' error before the cut, at the price of one more way for the writer to be wrong. Not in the first
  language; the first thing to add if two-of-a-kind misreadings are counted.
- **A language model asked about each shoot** in place of trained outputs. It would read an attribute the eyes have no
  field for, which a real robot will need. Here labels are free, and in v2's tower probes a frozen pretrained tower
  under trained heads stayed about seven points behind the small transformer on the count.
- **Free code, a planner, or beliefs with probabilities** in place of the interpreter: with three shoots the worlds can
  be listed.
- **A learned gaze, a completion head, a learned memory.** Each has a computed counterpart here that can be tested
  without a model.
- Visual robustness, the fast level's own track, and real images, as in v2.

## Prior work

The ingredients are established; the combination was not found, and nothing below could be used as it is for this
plant. Each source was opened and its abstract (or the page named) read for this document. Nothing from any of them was
reproduced here, and no code or weights of theirs are used. The sources of v2's design are in architecture-v2.md and
are not repeated.

*The split, as others have built it.*

- Ahn et al., "Towards Reliable Code-as-Policies: A Neuro-Symbolic Framework for Embodied Task Planning", NeurIPS 2025
  ([arXiv:2510.21302](https://arxiv.org/abs/2510.21302)): symbolic verification of a language model's robot code, and
  exploratory code that interacts with the environment to collect the observations it lacks; a 46.2% higher success
  rate than Code-as-Policies baselines. The nearest to "a missing fact asks for a look"; the program there is free code
  and the skills are programmed.
- Zhu et al., "NS-VLA: Towards Neuro-Symbolic Vision-Language-Action Models", NeurIPS 2026
  ([arXiv:2603.09542](https://arxiv.org/abs/2603.09542)): the instruction is parsed into a plan of primitives tracked
  by a stage pointer, and a policy that acts in chunks is conditioned on the active sub-instruction; 76.5% and 51.8%
  on LIBERO and LIBERO-Plus against 71.2% and 45.4% for its backbone. The stages here have that shape; there the
  pointer is fitted to demonstrations, here a stage closes on a cut the memory has seen.
- Tziafas et al., "Enhancing Interpretability and Interactivity in Robot Manipulation: A Neurosymbolic Approach",
  International Journal of Robotics Research ([arXiv:2210.00858](https://arxiv.org/abs/2210.00858)): a parser turns a
  query into a program of primitives, some symbolic (counting) and some neural (grounding); 80.2% in interactive
  object picking. A trained parser, objects on a table, no memory of a scene that changes.
- Mao et al., "The Neuro-Symbolic Concept Learner", ICLR 2019 ([arXiv:1904.12584](https://arxiv.org/abs/1904.12584)):
  an object-based scene representation, and sentences translated into symbolic programs that are executed on it. With
  the neural-symbolic question answering of v2's list, where attributes per object and a program come from.
- Wang et al., "Programmatically Grounded, Compositionally Generalizable Robotic Manipulation", ICLR 2023
  ([project page](https://progport.github.io/)): a semantic parser turns the instruction into a program of modules
  that connect vision and action; tested on new objects and on combinations of concepts learned apart.

*Facts not yet known, and looking.*

- Petrick and Bacchus, "A Knowledge-Based Approach to Planning with Incomplete Information and Sensing", AIPS 2002
  ([paper](https://cdn.aaai.org/AIPS/2002/AIPS02-022.pdf)): the planner's incomplete knowledge is represented as such,
  and actions, sensing among them, are modelled by how they change what is known. The plan's worlds are the smallest
  form of this.
- Garrett et al., "Online Replanning in Belief Space for Partially Observable Task and Motion Problems", ICRA 2020
  ([arXiv:1911.04577](https://arxiv.org/abs/1911.04577)): observation actions chosen by planning over beliefs, with
  the rest of the old plan reused so that progress is not undone. More than three shoots need; what is kept is that a
  look is an action and that what is done stays done.
- Bohg et al., "Interactive Perception: Leveraging Action in Perception and Perception in Action", IEEE Transactions
  on Robotics 2017 ([arXiv:1604.03670](https://arxiv.org/abs/1604.03670)): a survey of acting on the world in order to
  perceive it. Turning the plate is such an action.
- Liu et al., "RoboQuest: Generalist Physical Agents that Search, Inspect and Test", 2026
  ([arXiv:2610.10388](https://arxiv.org/abs/2610.10388)): ten tasks in which information has to be gathered by
  interaction; the best agent succeeds in 23% of the episodes, and agents often stop exploring too early. The source of
  counting looks taken against looks needed.
- Chen et al., "FoveaVLA: Foveation Actions Replace the Wrist Camera in VLAs", 2026
  ([arXiv:2610.20064](https://arxiv.org/abs/2610.20064)): the policy predicts the region to look at next and is given
  high-resolution crops with the full views; 70.8% and 77.1% against 52.1% for the baseline with a wrist camera. The
  crop here is likewise a higher-resolution view; where to look is computed, not learned.

*When to act and when not.*

- Jang et al., "VLA Policies Do Not Know When Not to Act: Unintended Task Execution and Start Suppression", 2026
  ([arXiv:2610.41609](https://arxiv.org/abs/2610.41609)): six policy checkpoints all moved on an empty instruction; a
  prompt prefix stopped none of them, and a gate outside the policy held five of six still. The reason permission to
  act is outside the fast level here.
- Zhou et al., "Code-as-Monitor: Constraint-aware Visual Programming for Reactive and Proactive Robotic Failure
  Detection", CVPR 2025 ([arXiv:2412.04455](https://arxiv.org/abs/2412.04455)): failures detected by constraints that
  generated code evaluates in real time; a 28.7% higher success rate under severe disturbances. The cut gate is a
  monitor of that kind, written by hand and checked at every control step.
- Abouagour and Min, "Audit Before You Commit: Locating Belief Failures in Active Identification for One-Shot
  Manipulation", 2026 ([arXiv:2609.30608](https://arxiv.org/abs/2609.30608)): before one irreversible action, more
  probes sharpened the belief while the truth left its support in 16.9% of episodes. The reason looks from one angle
  count as one witness.
- Yang et al., "SeqVLA: Sequential Task Execution for Long-Horizon Manipulation with Completion-Aware
  Vision-Language-Action Model", 2025 ([arXiv:2509.14138](https://arxiv.org/abs/2509.14138)): a head that predicts
  that a sub-task is complete, added to a policy. The learned alternative to a completion computed from the memory.

*The writer.*

- Park et al., "Grammar-Aligned Decoding", NeurIPS 2024 ([arXiv:2405.21047](https://arxiv.org/abs/2405.21047)):
  masking by a grammar can distort the model's distribution over the outputs the grammar allows. The reason the
  constrained and the free writer are compared again.
- Levy et al., "Diverse Demonstrations Improve In-context Compositional Generalization", ACL 2023
  ([paper](https://aclanthology.org/2023.acl-long.78/)): demonstrations chosen to cover the structures the output
  needs. The rule for the worked examples.
- Bogin et al., "Leveraging Code to Improve In-Context Learning for Semantic Parsing", NAACL 2024
  ([paper](https://aclanthology.org/2024.naacl-long.279/)): a general-purpose programming language as the target in
  place of a domain-specific one, and a structured description of the domain in the prompt; 7.9% to 66.5% on a
  compositional split. The programs here are nested calls with word names and the prompt describes the language
  construct by construct; the language itself stays small and closed, because its interpreter has to be tested.
- Ahmed et al., "Parsed, Plausible, Wrong: Output Format Controls Silent Errors in Small-Model Semantic Parsing", 2026
  ([arXiv:2611.17169](https://arxiv.org/abs/2611.17169)): with small fine-tuned parsers, a target written in ordinary
  words gave the highest accuracy and fewer well-formed wrong outputs, and a shorter target alone did not help. The
  reason the constructs are words.

*Benchmarks and how to evaluate.*

- Haresh et al., "ClevrSkills: Compositional Language and Visual Reasoning in Robotics", NeurIPS 2024
  ([arXiv:2411.09052](https://arxiv.org/abs/2411.09052)): three levels of compositional difficulty; the
  vision-language baselines fail to compose skills they were taught.
- Jiang et al., "VIMA: General Robot Manipulation with Multimodal Prompts", ICML 2023
  ([arXiv:2210.03094](https://arxiv.org/abs/2210.03094)): a protocol with four levels of generalisation.
- Li et al., "Embodied Agent Interface: Benchmarking LLMs for Embodied Decision Making", NeurIPS 2024
  ([arXiv:2410.07166](https://arxiv.org/abs/2410.07166)): modules evaluated apart, with errors by category in place of
  a final success rate alone.
- Chen et al., "Faster and Better? Benchmark Bugs and Design Limitations Distort the Evaluation of
  Vision-Language-Action Acceleration", 2026 ([arXiv:2609.37771](https://arxiv.org/abs/2609.37771)): 22 bugs and 4
  design limitations in seven benchmarks, whose fixes can reverse rankings.
- Zhang et al., "LegendBench: A Diagnostic Benchmark for Legend Understanding with Counterfactual Interventions", 2026
  ([arXiv:2609.24172](https://arxiv.org/abs/2609.24172)): legends in charts, tested by changing the legend and holding
  the chart. The test for the legend rung.
- Liu et al., "Automated Behavior Planning for Fruit Tree Pruning via Redundant Robot Manipulators", 2025
  ([arXiv:2510.12509](https://arxiv.org/abs/2510.12509)): perception, modelling and planning for pruning on a real
  robot. Its subject is the part this design leaves to the fast level: reaching the branch.

## Where it will live

| What | Where |
|---|---|
| The new constructs, stages, worlds, the situations' own programs | `sim/pickup/program.py` |
| Attributes in the truth and what can be read; the memory's rules for them | `sim/pickup/scene.py`, `sim/pickup/scenarios.py` |
| The census | `sim/pickup/census.py` (new) |
| The looks undrawn; `compose`; `tags` | `sim/pickup/scenarios.py` |
| The eyes' four outputs | `sim/pickup/eyes.py`, `eyes_data.py`, `train_eyes.py` |
| The writer's prompt, examples, asking twice | `sim/pickup/program_writer.py` |
| The look, the home move, the gate, the counters | `sim/pickup/slow_level.py`, `sim/pickup/eval.py` (`--scene full`, `--inspect`, `--gate`) |
| The expert's look and its sub-goals; the sub-goal column | `sim/pickup/expert.py`, `sim/pickup/collect.py` |
| Tests | `tests/essential/sim/test_program.py`, `test_scene.py`, `test_slow_level.py`, `test_eyes.py`, `test_program_writer.py`, `test_scenarios.py` |
