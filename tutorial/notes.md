# Field notes: what worked, what didn't

A running record of training robot policies on the pickup sim, written so that someone else can
skip the dead ends. Each entry says what we believed, what we measured, and what to do about it.
Numbers are closed-loop success on 100 layouts the policy never saw unless stated otherwise; one
training seed per run, so treat differences of a few percent as noise.

Status legend: **confirmed** (measured here), **open** (hypothesis, experiment queued).

## What worked

| What | Evidence | Entry |
|---|---|---|
| Running the scripted expert through the evaluation before training anything | 99-100/100 on the task where the first policies scored 0-6% | 1 |
| Recording where the gripper closed, not only whether the episode succeeded | It showed that the failing policies were not looking at all (slope under 0.1 where 1 means it goes to the object), which redirected the work | 3, 6 |
| Patch tokens instead of one pooled vector | 30% against 4% on the single cube | 5 |
| No history: one frame, no previous action | 75-81% against 2% after 3M samples | 8 |
| Training past the stretch where nothing seems to happen | 0% after 1M samples, 85% after 5M, same policy and data | 7 |
| More demonstrations | 85% with 4000 against 64% with 1500 | 7 |
| Keeping a running average of the weights | 36% against 4% in the middle of the transition | 12 |
| Computing the position table in float32 | 93% against 83%; on the single cube with history 96-98% against 62-70% | 14 |
| Repeating the windows from the first frames of every demonstration | 96-97%, and the policy starts to look about half a million samples earlier | 10 |
| Random start poses in the demonstrations | 98% from the home pose and 97% from random poses; follows an object moved early in the reach (94%) | 10, 11 |
| Demonstrations in which the object is moved and the expert recovers | The policy follows a moved object (79-88%) and retries a missed grasp (78 of 99 episodes) | 11 |
| Random start poses and recoveries in the same demonstrations | 94% on the unchanged task and 89-94% with the object moved at any point of the reach or as the gripper closes, also from a random start pose | 11 |
| Two sampling steps instead of ten | Equal within noise on five of seven policies (84-98%) at a third of the latency; worse on the other two (77% against 96%, 77% against 89%) | 12 |
| A VLM pretrained on captions as the backbone, every token handed to the action head | In progress. At the first checkpoint (0.25M samples) 43%, where the same model without pretraining is at 0% | 16 |

## What did not work

| What | Evidence | Entry |
|---|---|---|
| One pooled image vector for the action head | 4% on the single cube | 5 |
| One vector read from a VLM (the framework's VLA recipe), from-scratch or Qwen | 0-6% | 5 |
| The previous action as an input | 2% after 3M samples: the policy reads its goal there instead of in the image | 8 |
| A hold frame at the start of every demonstration, as the cure for that | 1-2% after 2.5M samples (still training) | 8 |
| Judging by the training loss | The loss is flat while success goes from 0% to 80% | 2, 7 |
| Stopping at 1M samples | 3-5% for a recipe that reaches 64% at 3M on the same data | 7 |
| Naming every object of a layout in turn | 0-1% with 1500 demonstrations and 1M samples. With 4000 and 5M it starts to look early and ends at 91% / 89%, no better than one instruction per layout, and as blind to a moved object | 10, 11 |
| Planning again more often | 76% when 2 actions of each chunk are executed, 85% with 8 | 12 |
| The same starting noise for every chunk of an episode | 39-80%, worse at every setting | 12 |
| Trusting a policy outside the conditions of its demonstrations | Policies at 89-96% score 0% when the object is moved late in the reach and 9-10% from another start pose | 9, 11 |

## What we got wrong

| What we believed or did | What was the case | Entry |
|---|---|---|
| "The failing policy reaches for a point between the objects" | Where it closed did not depend on the scene at all | 6 |
| "This design does not work with 1500 demonstrations" | The run had ended inside the transition; with three times the samples it reaches 64% | 7 |
| "What still fails is the grasp angle on the T block" | Most of it was the position bug: 55% before the fix, 87-93% after | 13, 14 |
| Positions computed in half precision in every run of the first two rounds | 641 distinct positions out of 2048; nothing looked wrong because training and serving agreed | 14 |
| About 40 evaluations filed under the wrong run | Two evaluations on one port both talked to the same policy server | 17 |
| The grasp tolerance measured on one simulator build and applied to another | At a 10 mm offset the can is grasped 60% of the time on one build and 94% on the other | 4 |
| Runs relaunched without averaged weights; a second trainer started in a live run's directory; two dataset builds sharing one Ray instance | Each cost between a few minutes and half an hour | 17 |

## The short version

1. Check the environment with a scripted expert before training anything.
2. Judge policies by closed-loop success only. Training loss and open-loop error do not predict it.
3. When a policy fails, measure *how* it fails before changing anything: record where the gripper
   closed and where the objects were.
4. Know how much error the task tolerates.
5. Give the action head spatial tokens, not one pooled vector.
6. A policy that ignores the scene may only be under-trained. Looking at the scene arrives late
   and all at once; evaluate every checkpoint and train past the point where it would appear.
7. Do not give the policy its previous action without testing what it does with it. Here it used it
   in place of the cameras.
8. A policy that succeeds may still look at the scene only once. Test it with something that moves.
9. A policy learns to look where its demonstrations make looking necessary. Vary the start pose,
   and put recoveries in the data if you want recoveries in the behaviour.
10. Save averaged weights and evaluate both sets. Sweep the sampler's settings: fewer steps were enough.
11. Feed inference through exactly the code path used in training, and test that it does.
12. Checkpoint often and auto-resume; long runs die. Make every evaluation check which checkpoint
    it is talking to.

## 1. Verify the task with an expert first  (confirmed)

**What we did.** `sim/pickup/eval.py --expert` runs the scripted grasp expert through the same
evaluation loop, seeds and success test as a learned policy.

**Result.** 99/100 on `pick_target` on one machine and 100/100 on the other, closing the gripper
5 mm (median) from the object.

**Why it matters.** Every policy scored 0-6% on this task. Without the expert number you cannot tell
a broken simulator, an impossible task or a wrong success test from a policy that has not learned.

## 2. Loss and open-loop error do not predict success  (confirmed)

| Policy | Final training loss | Closed-loop success |
|---|---|---|
| ViT policy, `pick_cube`, 1000 demos | 0.058 | 74% |
| ViT policy, `pick_target` | 0.069 | 1% |
| Custom VLA, `pick_target` | ~0.03 | 0% |
| Qwen VLA, `pick_target` | 0.03-0.04 | 6% |

The 1% and the 74% policy reproduce recorded expert actions equally well (0.01-0.04 rad per joint).

**Why.** In a demonstration, the previous action and the arm's own motion predict the next action
almost perfectly from the second step on. A model can therefore reach a low loss while learning
very little about where the object is. Only the first moments of an episode need the scene, and
they are a small share of the training samples.

**What to do.** Evaluate closed-loop from the first checkpoint. Budget evaluation time like
training time.

## 3. Measure how it fails  (confirmed)

Success rate alone said "0-1%". `sim/pickup/diagnostics.py` records where the gripper is, relative
to every object, at the moment it first closes:

| Policy (`pick_target`) | Closes the gripper | Nearest object is the named one | Median miss to named object | Median miss to nearest object |
|---|---|---|---|---|
| Scripted expert | 100% | 100% | 5 mm | 5 mm |
| ViT policy, no language input | 100% | 32% | 185 mm | 110 mm |
| Custom VLA (pretrained VLM) | 100% | 33% | 175 mm | 89 mm |
| Custom VLA (random-init VLM) | 100% | 29% | 174 mm | 89 mm |

Three objects are on the table, so 33% is chance. All three policies run the whole
reach-descend-close-lift motion at the right height, at a place that is about 10 cm from *any*
object. They did not learn a slightly imprecise grasp; they learned the motion and not the target.

**Why it matters.** "Near miss, train longer" and "never looked at the scene" need different fixes.
The rollout videos looked like near misses; the measurements say otherwise.

## 4. Know the tolerance of the task  (confirmed)

`sim/pickup/grasp_tolerance.py` adds a fixed sideways offset to the expert's grasp point:

| Offset | Cube | T block | Can | Hex nut |
|---|---|---|---|---|
| 0 mm | 100% | 100% | 94% | 89% |
| 10 mm | 99% | 100% | 60% | 61% |
| 20 mm | 93% | 98% | 18% | 25% |
| 30 mm | 32% | 76% | 0% | 10% |

The cube and T block forgive 2 cm. The two round objects need 1 cm or better on the machine this
table is from. On a second machine with a newer simulator build the same test gave 94% for the can
and 97% for the hex nut at 10 mm (29% and 70% at 20 mm): measure it where you evaluate. A policy that is
"usually within 2 cm" would score about 95% on cubes and about 20% on cans.

## 5. One pooled vector is not enough for grasping  (confirmed on `pick_cube`, open on `pick_target`)

| Conditioning the action head receives | Task | Success |
|---|---|---|
| CLIP: one vector per image + one for the text | `pick_cube`, 200 demos | 4% |
| ViT: every 16x16 image patch as a token | `pick_cube`, 200 demos | 30% |

VLA Foundry's VLA recipe (`vlm_backbone`, `vlm_foundry_backbone`) appends an action token to the
prompt and passes only that token's hidden states to the action head: again one vector. Both
from-scratch VLAs built that way reached the named object at chance (section 3).

**Fix in this fork.** Both VLM backbones (`vlm_foundry_backbone` for the from-scratch VLM,
`vlm_backbone` for Hugging Face models) have `conditioning: all_tokens`, which passes the hidden
states of every image and text token. `vit_backbone` has `text_tokens`, which adds the instruction
to a patch-token policy.

## 6. A failing policy ignores the scene; it does not average  (confirmed)

The ViT policy has no language input. On `pick_cube` (one object) it reaches 74%. On `pick_target`
(three objects) it closes 11 cm from the nearest one. The first reading was that it reaches for a
point between the objects. The per-episode records of the 1500-demonstration runs say otherwise:

- the closing positions fill the area where objects can spawn evenly; they do not gather between
  the three objects of their own scene;
- regressing the closing position on the position of the named object gives slopes between 0.00
  and 0.09 and an R² of at most 0.006 in the three runs that never started to look (a policy that
  goes to the named object has slope 1);
- the miss does not change when each episode's closing position is compared with the objects of a
  different episode.

So the first action chunk is a draw from all the reaches in the training data, unrelated to the
scene in front of the policy, and later chunks continue that motion. "About 10 cm from the nearest
object" is what a random point in the spawn area gives when three objects are on the table. The
same holds for the policies with a language input that failed in section 3: they had the
instruction and still did not use the image.

**What to do.** Record where the gripper closes and where every object was, per episode. The
slope of one against the other separates "imprecise" (slope near 1, scattered) from "not looking"
(slope near 0) in one number. `sim/pickup/eval.py` writes both and the summary reports the slope.

## 7. Looking at the scene arrives late, and all at once  (confirmed)

The policy that works here is the patch-token policy with the instruction as extra tokens and no
history (`dp_vit_lang_reactive`, 49M parameters). After 1M training samples on 1500 demonstrations it
scored 3-5% and closed the gripper at the named object 41-49% of the time (chance is 33%). That looked
like a design that does not work. The same policy on 4000 demonstrations, evaluated at every
checkpoint (trained weights / averaged weights):

| Training samples | Success | Closed at the named object | Slope | Median miss |
|---|---|---|---|---|
| 0.5M | 0% / 0% | 34% / 26% | 0.02 / -0.06 | 166 / 167 mm |
| 1.0M | 0% / 1% | 35% / 32% | 0.11 / 0.14 | 138 / 154 mm |
| 1.5M | 4% / 2% | 61% / 49% | 0.47 / 0.37 | 102 / 104 mm |
| 2.0M | 4% / 36% | 95% / 98% | 0.81 / 0.96 | 32 / 18 mm |
| 2.5M | 56% / 70% | 98% / 100% | 0.94 / 0.98 | 14 / 13 mm |
| 3.0M | 81% / 75% | 100% / 99% | 0.98 / 0.97 | 9 / 9 mm |
| 5.0M | 83% / 85% | 100% / 100% | 0.98 / 0.98 | 7 / 7 mm |

For 1M samples nothing happens. Between 1.5M and 2.5M the policy goes from not using the scene to
using it. The slope moves first, then the miss shrinks, then success follows.

The 1500 demonstrations were enough too. Trained for 3M samples instead of 1M, the same policy does
the same thing, earlier: 0-3% up to 1.2M, 18-21% at 1.5M, 43-44% at 1.8M, 62-70% at 3M. The 1M run
had ended inside the transition, with its learning rate already near zero.

**What to do.** Evaluate every checkpoint and plot the slope next to success. A flat 0% says nothing
about what comes next; a slope that has started to move does. Do not judge a design on a budget that
ends before the transition of a comparable run.

**Why so late (tested).** In a demonstration only the first moments need the image and the
instruction; after that the arm's own position says where it is going (section 9). So few training
windows reward looking. Every change that made more windows depend on the scene brought the
transition forward (section 10): after 1.0M samples the plain run closes at the named object 65% of
the time, with the windows of the first frames repeated 99%, with random start poses 91%, with
recovery demonstrations 81%.

## 8. The previous action as an input  (confirmed)

Many policies are given a short history: the previous frame, the previous state and the action they
just executed. Here that turned a working design into one that never starts.

| `pick_target`, 4000 demonstrations | 1.0M | 2.0M | 3.0M |
|---|---|---|---|
| no history | 0-1% | 4-36% | 75-81% |
| with history | 1% | 1-3% | 2% |

On the single cube, where there is nothing to choose, history costs nothing. The first comparison there
(62% / 70% with history against 95% / 91% without) looked like a cost, but the two runs also differed in
their language input and pixel range, and the gap was the position bug of section 14: trained again
with the table fixed, the history policy ends at 96% / 98%.

The history policy is not blind. The same checkpoint (2.5M samples):

| What it is given at the start | Success | Closed at the named object |
|---|---|---|
| as served: the previous action is the hold command | 1% | 30% |
| the previous action is the expert's first command | 22% | 97% |
| the expert drives the first 10 steps | 77% | 100% |

One expert action is enough for it to find the named object 97% of the time. It learned to read the
target from the previous action, and never needed the instruction or the image for that.

The three-object runs with history were trained before the position fix, which hit them harder than
the runs without (16 action-step tokens on 4 positions against 15 on 8). The fix does not rescue them:
the history policy trained after it, on demonstrations that begin with a hold frame, is as blind (see
below). A history run on the plain demonstrations has not been repeated after the fix.

**Why.** From the second step of a demonstration on, the previous action is the best predictor of
the next one. And in the windows that start at the first frame, where there is no previous action,
the framework fills the slot with a copy of the first action: the expert's first command, which
already points at the target. In training the answer is always in the input. When the policy is
served, the slot holds what the robot is really doing before it moves, which is nothing.

**What to do.** Start without history. If you add it, run the two probes above
(`eval.py --first_prev_action expert`, `--expert_prefix 10`): a policy that scores far better with
them than without is leaning on its own past. `collect.py --hold_steps 1` begins every demonstration
with a frame of the robot at rest, so that the first windows contain what serving will provide. It
has not helped so far: a history policy trained on such demonstrations is at 1-2% after 2.5M samples
(still training). The leak at the first frame is then closed, but from the second step on the
previous action still predicts the next one, and that was enough to keep the policy from looking.

## 9. Success is not the same as looking  (confirmed)

The 85% policy has no history, so every chunk it plans starts from the current camera images. It
still does not use them the way one would hope:

| Test (same checkpoint) | Success | Where it closed the gripper |
|---|---|---|
| as trained | 85% | at the named object in 100% of episodes, 7 mm off |
| shown the first frame's images for the whole episode | 1% | at the named object in 99%, 17 mm off |
| the expert drives the first 25 steps toward another object | 0% | at that other object in 100% |
| the named object is moved 8 cm at step 20 of the reach | 0/96 | 8 mm from where the object was |
| the named object is moved 8 cm at step 10 | 9% | 39 mm from the old place, 49 mm from the new |
| the object is moved as the gripper closes | 1/99 | - |

It decides where to go from the first look and then follows its own arm: from a fixed home pose every
reach is a straight line, so a few steps in, the joint angles say where the line ends. The images
matter again only for the last centimetres. Nothing in the demonstrations ever contradicts this
strategy, because nothing in them ever moves and no grasp ever fails.

**Why it matters.** The success rate on a static benchmark cannot show this. A robot that has to
work around things that move, or to try again after a miss, needs exactly the behaviour this policy
lacks.

**What to do.** Add a disturbance to the evaluation (`eval.py --disturb reach` and `--disturb grasp`)
before trusting a policy outside the conditions it was trained in, and change the demonstrations
(sections 10 and 11).

## 10. What the demonstrations contain  (confirmed)

The policy, the number of demonstrations (4000) and the budget (5M samples) stay the same; only what
the demonstrations show changes. All runs after the position fix; success with the trained weights,
or trained / averaged where a run saved both:

| Demonstrations | After 1.0M samples | After 3.0M | Final |
|---|---|---|---|
| one clean pass per layout, from the home pose | 8% | 90% | 93% |
| the same, windows of the first 8 frames repeated 5 extra times | 5% | 96% | 96% |
| every object of a layout named in turn (about 1333 layouts, three demonstrations each) | 41% / 38% | 91% / 89% | 91% / 89% |
| every episode starts from a random gripper pose | 56% / 76% | 94% / 98% | 98% / 98% |
| in 1505 of 4000 episodes the object is moved once and the expert recovers | 38% / 79% | 83% / 85% | 89% / 84% |
| random start poses, and the object is moved once in 1456 episodes | 68% / 86% | 90% / 93% | 94% / 94% |

After one million samples the plain run closes at the named object 65% of the time. With the early
windows repeated it is 99%, with every object named 98%, with random starts 91%, with recoveries 81%,
with random starts and recoveries 100%: the policy starts to look sooner whenever more of its
training windows cannot be answered without looking. Random starts give the best result on the
unchanged task.

Naming every object of a layout in turn shows the policy that the instruction matters, and it does
start to look early. It ends no higher than one instruction per layout (the T block at 64%), and it
is a third as many layouts for the same number of episodes.

The run that adds repeated early windows to random starts and recoveries is in progress.

**What to do.** Before collecting more demonstrations, ask which of them can be imitated without
looking at the scene. From one fixed start pose with nothing ever moving, almost all of them can.
`collect.py --random_start` costs nothing; `boost_early_windows.py` needs no new data at all.

## 11. Self-correction has to be in the data  (confirmed)

The same policies, each at its final checkpoint, under conditions their demonstrations may or may
not have contained. 100 layouts per cell; the object is moved 80 mm:

| Trained on | Unchanged task | Random start pose | Object moved at step 10 / 20 / 30 of the reach | Object moved as the gripper closes |
|---|---|---|---|---|
| clean demonstrations | 93% | 9% | 12% / 0% / 0% | 0%, a second attempt in 8 of 100 |
| clean, early windows repeated | 96% | 9% | 10% / 0% / 0% | 0% |
| clean, every object named | 89% | 10% | 18% / 0% / 0% | 0% |
| random start poses | 98% | 97% | 94% / 51% / 0% | 0% |
| recoveries | 84% | 68% | 88% / 82% / 85% | 79%, a second attempt in 99 of 99 |
| random start poses and recoveries | 94% | 91% | 92% / 91% / 94% | 89%, a second attempt in 100 of 100 |
| scripted expert that plans again | 100% | 100% | 100% / 100% / 100% | 100% |

Each ingredient buys exactly what it shows. Random start poses make the policy watch the scene while
it reaches, so it follows an object that moves early, but nothing in its data ever went wrong late,
and it neither follows a late move nor tries again. Demonstrations with recoveries give both. They
also make the policy use its cameras throughout: frozen on the first frame it finds the named object
3% of the time, where the policy trained on clean demonstrations still finds it 94% of the time. And
handed over after the expert has driven 25 steps toward the wrong object, it turns back and succeeds
in 56 of 100 episodes; the clean-data policy closes on the wrong object every time.

The recovery policy pays on the unchanged task (84-89% against 93-98%, the can at 77%).

Both ingredients in the same demonstrations give both behaviours and take most of that cost back:
94% on the unchanged task, 91% from a random start pose, 89-94% wherever in the reach the object is
moved, 89% when it is moved as the gripper closes, and 90 of 98 from a random start pose with the
object moved on the way. With its cameras frozen on the first frame this policy finds the named object
in none of 100 episodes; handed over after the expert has driven 25 steps toward the wrong object, it
turns back and succeeds in 89. It looks all the way. What it has not regained is the T block (77%,
against 92% with random starts alone).

**What to do.** Write the expert so that it can recover (`ScriptedExpert(reactive=True)` plans again
when the object has moved or its fingers closed on nothing), disturb a share of the demonstrations and
vary where they start (`collect.py --disturb_prob 0.5 --random_start`), and evaluate with the same
disturbances. A behaviour that never
occurs in the demonstrations does not appear in the policy, whatever its success rate.

## 12. Averaged weights and the sampler's settings  (confirmed)

**Averaged weights.** Runs keep a running average of their weights (EMA) next to the trained ones.
During the transition the average is far ahead: 36% against 4% at 2.0M samples, 70% against 56% at
2.5M. Before it and after it the two are equal within noise. It is not a rule: in the 3M-sample run on
1500 demonstrations the two stay within 9 points of each other and the trained weights are the ones
ahead, at 9 of its 11 checkpoints. Saving both costs one extra copy of the
model.

**Flow steps.** The sampler turns noise into an action chunk in a number of steps. On the 85%
policy, 100 layouts per setting (a second 100 in brackets): 1 step 61%; 2 steps 89% (90%); 4 steps
88% (86%); 10 steps, the default, 85% (83%); 20 steps 79%. Two steps cost 25 ms per chunk, ten cost
71 ms. On the policies trained after the position fix, 2 steps against 10 give 94% against 93%, 98%
against 98%, 84% against 84% and 94% against 94%, but 77% against 96% for the policy trained with
repeated early windows and 77% against 89% for the one trained with every object named: measure it on
the policy you will run.

**Actions executed per chunk.** The policy predicts 15 actions. Executing 2 of them before planning
again gives 76%, 4 gives 84%, 8 gives 85%, all 15 gives 84%. Planning more often does not help, and
on the T block it hurts (8/23 at 2 actions, 15/23 at 8).

**Tried, did not help.** Starting every chunk of an episode from the same noise, so that successive
chunks would make the same choices: 80% at 8 actions per chunk, 55% at 4, 39% at 2.

## 13. What still fails  (confirmed)

Before the position fix the failures were angle errors on the two objects that cannot be grasped at
any angle: the T block at 55% and the cube at 82% (can 93%, hex nut 94%, pooled over the checkpoints
from 3M samples on), with a median of 64 degrees off on the failed T-block grasps. The failures were
spread over all object orientations and did not cluster where the expert's choice between two
equivalent grasps flips, so they were not the expert's ambiguity.

Most of that was the position bug of section 14. After the fix, pooled over the last three
checkpoints: T block 87%, cube 94%, can 90%, hex nut 99% with clean demonstrations; T block 92%, the
other three 99-100% with random start poses. What remains is spread thinly over the objects.

## 14. Train/inference mismatches hide in preprocessing  (confirmed)

Five found so far, none of which raised an error:

- **Pixel range.** `SimpleVLMProcessor` scaled PIL and numpy images to [0, 1] but left uint8 tensors
  at 0-255. Training fed tensors, inference fed PIL. The VLM scored 0% on a captioning check that it
  passes at 85% after the fix.
- **Padding moves the probe.** The action token was appended after the batch padding. In training a
  short instruction sits in a batch with longer ones, so its action token is one position later
  (behind a padding token) than when the same instruction is served alone. Fixed in both VLM
  backbones: the token goes directly after the last real token. Effect on success not measured.
- **Zero means "unset".** `lowdim_past_timesteps or default` treats 0 past steps as missing and
  falls back to 1, which hands the policy the state one step in the future during training. Found by
  reading the code before running such a config. Fixed, with a test.

- **The previous action at the first frame.** Section 8: the slot is filled with the expert's first
  command in training and with the hold command when served.
- **Positions computed in half precision.** The table of rotary position angles was built under
  bfloat16 autocast, which keeps 8 significant bits. Positions above 256 collapsed onto even numbers
  and above 512 onto multiples of 4: 641 distinct positions out of 2048. In the no-history policy 15
  action tokens shared 8 positions; with history, 16 shared 4. Training and serving agreed with each
  other, so nothing looked wrong. The table is now computed in float32. The same run after the fix
  starts to look earlier (58% against 4% after 1.5M samples) and ends higher (93% against 83%, the
  T block at 20/23 against 15/23); the policy with history on the single cube goes from 62-70% to
  96-98%. One seed each; a second pair is running.

**What to do.** For every model, run a check that pushes the same sample through the training
pipeline and through the serving path and compares the tensors the model receives.

## 15. Long runs die  (confirmed)

One of the machines used was unreliable: stage-3 runs died or hung three times. A dead data-loader
worker does not crash the trainer; it waits forever at 0% GPU. `scripts/train_resilient.sh` restarts
from the newest checkpoint when a run exits or when its log stops moving for 7 minutes.

## 16. Pretraining a small VLM on captions  (in progress)

The from-scratch path (91M LLM on web text, then a VLM on 40k captioned sim renders, then a VLA)
gave 0/100 with pretraining and 1/100 without in the first round. Both policies conditioned the
action head on a single vector (section 5), had history (section 8) and 1M samples on 1500
demonstrations (section 7), so neither could have found the object and the comparison said nothing
about pretraining.

It is being re-run with every token passed to the action head, no history and 4000 demonstrations.
The runs are not finished; their checkpoints so far are in the lab notebook. The first of them
already differ: after 0.25M samples the VLA on the pretrained VLM closes at the named object in 97%
of episodes and succeeds in 43% (averaged weights), and the same architecture with a randomly
initialised backbone is at chance, 0%.

The same change is being tried on the pretrained Qwen backbone, which with one conditioning vector
ended at 5% (closing at the named object 71% of the time, 83 mm off: it points, it does not grasp).

## 17. Operating many runs on one machine  (confirmed)

Things that cost time here and have nothing to do with models:

- **Cap CPU threads per process.** Numerical libraries start one thread per core. On a 128-core
  machine each policy server started 256 threads and used 14-22 cores doing nothing useful;
  twelve of them drove the load average to 570 and halved training throughput on every GPU. With
  `OMP_NUM_THREADS=4` (and the MKL / OpenBLAS equivalents) each server uses under one core. The
  scripts now set this (`PICKUP_CPU_THREADS`).
- **Pin the GPU for rendering as well as for compute.** `CUDA_VISIBLE_DEVICES` does not restrict
  EGL, and EGL's device order is not `nvidia-smi`'s: on one multi-GPU machine two devices in the
  middle of EGL's list were GPUs 0 and 1. MuJoCo 4 orders EGL devices by `CUDA_VISIBLE_DEVICES`, so setting that one variable
  per job pins both, provided `MUJOCO_EGL_DEVICE_ID` is left unset. Check with `nvidia-smi` while a
  render is running rather than trusting an index.
- **Start long jobs detached.** A job started inside an ssh session dies with the connection.
- **Do not edit a shell script while it is running.** Bash reads scripts as it goes; an in-place
  edit made a running evaluation fail with a syntax error halfway through.
- **Follow the lockfile.** The framework's install notes said `uv sync` and then
  `uv pip install -e .`. The second command re-resolves outside the lockfile and silently replaced a
  pinned dependency with a newer release. `uv sync` alone already installs the project.
- **Give every evaluation its own port, and make it check what answers.** Two evaluations started
  on one port both talked to the policy server that had bound it first. The second one ran to the
  end and filed the other run's success rate under its own name; about 40 results here were wrong
  that way before one of them looked too good. `eval.sh` now takes a free port, and the client stops
  unless the server reports the run and checkpoint it asked for.
- **One trainer per run directory.** A queue script that reused a name started a second trainer in
  the directory of a live run. It was caught before it wrote a checkpoint. `train_resilient.sh` now
  holds `<run>.pid` while it runs, and a second copy started with the same name exits.
- **Switch averaged weights on in the config, not on the command line.** Runs relaunched by hand
  came up without them, and a run cannot gain them later unless resuming allows it (it does now).
- **Give every dataset build its own Ray instance.** Two builds started together shared one, and the
  second died with an error about `num_cpus` (`make_dataset.sh` now passes `--ray_address local`).
- **Say why a run was stopped.** A run ended on purpose looks exactly like one that hangs. The
  monitors show `<run>.stopped` as "stopped" with its text; `train_resilient.sh` writes it when it is
  interrupted.
- **Evaluate while training.** Evaluating every checkpoint as it appears costs little next to
  training and turns one run into a curve; a single final number hides whether the policy was
  still improving, had peaked, or never moved.

## Open questions being tested

- Does repeating the early windows add anything to random start poses and recoveries together?
- How much does one run vary? The plain recipe is being repeated with another seed, with and
  without the position fix.
- How far does the from-scratch VLA get on its pretrained VLM, against the same model without
  pretraining and against the Qwen backbone with every token?
- Can a policy with history be made to look? The hold frame alone was not enough.
- Disturbances here are a single jump of the object. Does the recovery policy cope with an object
  that keeps moving, and with look-alike objects next to the named one?
