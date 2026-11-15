# A reasoning VLA: reason, then act

The policies in this repo map pixels, joint angles and an instruction to joint targets in one pass. On the pruning
plant they learned to turn the plant, reach a shoot and cut it, and with named shoots (step 9) to read which shoot
the instruction means. This document describes the next step (roadmap Phase 7): tasks where the shoot to cut has to be
worked out, the text the scripted expert writes while it works it out, and a policy that writes such text before it
acts. It covers the design. Measured results go in the [lab notebook](lab-notebook.md).

## The benchmark: reasoning on the pruning plant

The plant, the stand with its two buttons, the cut rules, the damage rules and the wrong-shoot accounting are those of
`prune_named` (2-3 side shoots at random angles, the same plant as `prune` for the same seed). What changes is the
instruction. In `prune_named` a word names a shoot ("the lowest"), so a policy can learn a lookup from the word to a
height on the stem. In the reasoning tasks the shoot follows from the instruction **and** something about the plant
that has to be seen, counted or remembered first. Four kinds (`sim/pickup/reasoning.py`):

| Kind | Example | The reasoning step | Training compositions | Held-out compositions |
|---|---|---|---|---|
| ordinal | "cut the second shoot from the top" | count the shoots, then count down: with three it is the middle one, with two the lowest | 1st / 2nd from the top, 1st / 3rd from the bottom | 2nd from the bottom, 3rd from the top |
| conditional | "if the flower is red, cut the lowest shoot, otherwise cut the highest one" | look at the flower, compare its colour with the one named, take the branch that holds (it holds in half the episodes) | 6 of the 8 (colour, branch) pairs | (red, highest), (purple, lowest) |
| exclusion | "cut every side shoot except the highest one" | the complement: one or two shoots depending on the plant, in any order, keeping track of which are done | keep the lowest / the highest | keep the middle (three-shoot plants) |
| hidden | "read the label in the pot and cut the shoot it names" | a card stuck in the pot names the shoot by its colour (white the lowest, yellow the middle, cyan the highest) and faces a random way. If it faces away from the front camera the robot first turns the plant to read it; turning the shoot into reach afterwards usually hides the card again, so what it said has to be remembered | | only new sentence frames |

Each kind has training sentence frames and held-out ones. A held-out episode (`reason_<kind>_heldout`,
`reason_mix_heldout`) draws a held-out composition in a training frame, a training composition in a held-out frame,
or both (the hidden kind: held-out frames only). No held-out instruction occurs in training (tested).

**Why a lookup cannot do it.** `reasoning.lookup_ceiling` draws many tasks and asks how often the best mapping from the
instruction's words to the shoots to cut is right, whatever the plant: about 0.65 for ordinal (where "first from the
top" is a lookup once learned, but "second from the top" depends on the count and is drawn most), 0.51 for
conditional and exclusion, 0.42 for hidden. A policy that does not look at the flower, count the shoots or read the
card is capped there. For the hidden kind, `env.label_hidden_at_reach()` says whether the card is out of view once the
shoot to cut faces the robot (in about two thirds of the layouts): there a policy that does not remember what it read
has nothing to go on but where the plant was turned.

**Tasks.** `reason_ordinal`, `reason_conditional`, `reason_exclusion`, `reason_hidden` and their `_heldout` versions;
`reason_mix` draws the kind per episode (the training data) and `reason_mix_heldout` likewise. Episode limits: 450
control steps for ordinal and conditional, 700 for hidden, 900 for exclusion. Cutting a shoot not meant ends the
episode as a failure (`wrong_shoots`), as does damage. `eval.py` reports every reasoning task by kind, by phrasing
(training or a kind of held-out instruction), by number of shoots, and for the hidden kind by whether the card is out
of view at the cut.

**The scripted expert** cuts exactly the shoots meant: it takes the targets the task says, the nearest first, and for
the hidden kind turns the plant to read the card before it knows them. From the home pose it succeeds in 99 of 100
layouts of every kind, training and held-out instructions alike (one plant times out in every kind). With random start
poses, as the demonstrations are collected, about 70% of its attempts succeed, as for `prune` and `prune_named`; only
successes are kept.

## Text in the demonstrations: the expert's notes

The expert writes a note at the moments where a decision is made or its basis changes (`sim/pickup/notes.py`): at
the start, when a shoot is cut, when the card is read (it faces the front camera while the plant stands still), and
when a turn of the plant has ended. A note is a function of what the robot can see at that moment and what it
remembers (the card's colour, once read); between those moments it stands. One line, five fields:

| Field | Says | Example |
|---|---|---|
| see | number of shoots, the flower's colour, the card | `3 shoots, purple flower, label hidden, read cyan` |
| ask | the instruction in a fixed form | `the shoot the label names` |
| so | the inference, and which shoots are done | `cyan names the highest` |
| aim | the shoot (or the card) to bring in front of the robot: whether it faces the robot, else which button turns it the short way | `the highest, faces away, press orange` |
| next | the sub-goal | `cut the highest shoot` (or `show the label`, `stop`) |

One episode (`reason_hidden`, seed 1000000, three shoots; the card faces away at the start), notes with the control
step at which each was written:

```
[   0] see: 3 shoots, purple flower, label not seen | ask: the shoot the label names | so: read the label first | aim: the label, faces away, press orange | next: show the label
[ 172] see: 3 shoots, purple flower, label cyan | ask: the shoot the label names | so: cyan names the highest | aim: the highest, faces away, press orange | next: cut the highest shoot
[ 278] see: 3 shoots, purple flower, label hidden, read cyan | ask: the shoot the label names | so: cyan names the highest | aim: the highest, faces me | next: cut the highest shoot
[ 390] see: 3 shoots, purple flower, label hidden, read cyan | ask: the shoot the label names | so: cyan names the highest | aim: nothing left | next: stop
```

At step 278 the card is out of view; the note still carries what it said. The notes are regular enough to check:
`reasoning.note_fields` parses them, and evaluation compares a policy's notes with the correct ones field by field.

**Storage.** `collect.py` adds four per-frame columns to the episodes of reasoning tasks: `note` (the note in force),
`note_prev` (the one before it), `note_age` (frames since it was written) and `instruction_subgoal` (its `next` field).
Preprocessing copies per-frame columns into each sample's metadata, and the LeRobot converter now offers any column
`instruction_<type>` as a language-instruction type, so a config can train on the sub-goal (`language_instruction_types:
[subgoal]`). `meta/episodes.jsonl` also lists each episode's kind, targets, card and notes.

## The model

### One model: reason, then act (`reason_then_act`)

A diffusion policy on a vision-language backbone that predicts tokens (Qwen3-VL-2B, `vla_qwen_reason.yaml`, or our
own 130M VLM, `vla_custom_reason.yaml`), with the every-token conditioning of the earlier VLAs. The text after the
images is

```
<instruction>
note: <the note the policy holds, "none" at the start>
now: <the note it writes now, or "same">
```

and a language-model loss on the tokens after `now:` (and the end-of-turn token that closes them) is added to the
flow-matching loss on the actions (`vla_foundry/models/diffusion_policy/reason_then_act.py`). In training the note is
the expert's (teacher forcing). The action head reads the hidden states of every token of the last layers, the note's
included. When served, the backbone writes the note greedily every chunk (8 control steps), then the actions are drawn
from the whole sequence; the note it wrote is the one it holds next time.

Two details make training match serving. A served policy holds a note that can be a few steps old, so for each
training sample the held note is the one in force `d` frames earlier, `d` uniform in 1..8; if the expert wrote a new
note in between, the target is that note, otherwise `same` (`vla_foundry/data/processor/notes.py`). Most targets are
`same` (two tokens) and the loss is the mean over tokens, so the few frames where a note is written carry most of it.
And `same` makes writing cheap when nothing has changed: a call costs two or three forward passes, a full note about
fifty.

The held note is the policy's memory. A policy that reads only the current images (the reactive policies here) cannot
know, once the card is out of view, what it said; the note can.

### Two levels

The same model with the action loss switched off (`qwen_reasoner.yaml`, `action_loss_weight: 0`, last 8 language-model
layers trained) is a reasoner only. A fast policy, the small `dp_vit_lang_reactive` reading the note's sub-goal instead
of the instruction (`dp_vit_lang_reactive_subgoal.yaml`), carries the sub-goal out: "cut the middle shoot" is the
named-shoot task of step 9 with a fixed phrasing, and "show the label" is a turn of the plant. When served
(`scripts/eval_two_level.sh`), the reasoner writes a note every chunk and the fast policy acts on its sub-goal.

**Which first.** The two-level form is the faster path to a working system, and the first runs build it alongside
the one-model form:

- The fast level is a skill the small policy already has (49-71 of 100 named shoots cut at step 9 with training
  phrasings), trained at about 1100 samples per second; the reasoning-heavy part is a short text over two images,
  which is what a pretrained VLM is for. The one-model Qwen VLA trains at about 60 samples per second and was still at
  8-14 of 100 on named shoots after 0.5M samples.
- Each level can be measured alone: the fast level with the correct sub-goals (an upper bound for the pair), the
  reasoner on the expert's own rollouts (are its notes right?). A failure of the pair can then be put on one level.
- The fast level's instruction comes from a closed set of sub-goals, so the from-scratch word table is no handicap:
  the reasoner turns any phrasing into the same few words.

The one-model form is the one that tests whether the action head itself uses the reasoning, and it is the general
form; it is trained too, at the slower rate.

### Is the text read? The swap test

A note the actions ignore is narration. `eval.py --notes swap` (one model) and `--subgoals swap` (two levels) hand the
policy correct-looking notes about a shoot the instruction does not mean, and report how often it cuts the shoot the
note names and how often one the instruction means. `--notes expert` and `--subgoals expert` hand it the correct
notes: what the acting part can do when the reasoning is right.

### Baselines

The plain small policy (`dp_vit_lang_reactive`, the instruction as text) and the plain Qwen VLA
(`vla_qwen_tokens_reactive`) trained on the same demonstrations, evaluated on the same tasks: where an end-to-end policy
without notes stands on each kind, and on the held-out instructions.

## Background

- Embodied chain-of-thought (Zawalski et al., 2024, "Robotic Control via Embodied Chain-of-Thought Reasoning"): OpenVLA
  fine-tuned to write several reasoning steps before each action (plan, sub-task, motion, and grounded features such as
  object bounding boxes and the end-effector position); the abstract reports 28% more absolute success on hard
  generalisation tasks, with no extra robot data. The one-model form here does the same with a short, fixed note and a
  diffusion action head conditioned on every token, and writes it only when something has changed.
- Hi Robot (Shi et al., 2025, "Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action
  Models"): a high-level VLM reasons over the prompt and feedback and passes the next step to a low-level policy. The
  two-level form here has the same split, with a closed set of sub-goals.

Both are described from their abstracts; nothing from them was reproduced here.

## Where it lives

| What | Where |
|---|---|
| Kinds, instructions, held-out splits, note format, lookup ceiling | `sim/pickup/reasoning.py` |
| When notes are written | `sim/pickup/notes.py` |
| Tasks, the card | `sim/pickup/env.py` (`reason_*`) |
| Expert: targets, reading the card | `sim/pickup/expert.py` |
| Notes in the demonstrations | `sim/pickup/collect.py` |
| Notes in the prompt, the loss's tokens | `vla_foundry/data/processor/notes.py` |
| The model | `vla_foundry/models/diffusion_policy/reason_then_act.py` |
| Serving notes, the reasoner | `sim/pickup/policy_server.py` |
| Evaluation, two levels, swap test, reasoner check | `sim/pickup/eval.py`, `sim/pickup/scripts/eval_two_level.sh` |
| Configs | `vla_qwen_reason.yaml`, `qwen_reasoner.yaml`, `vla_custom_reason.yaml`, `dp_vit_lang_reactive_subgoal.yaml` |
| Tests | `tests/essential/sim/test_reasoning.py`, `tests/essential/data/test_notes.py`, `tests/essential/models/test_reason_then_act.py` |

Collect and train as for any other task, for example:

```
sim/pickup/scripts/make_dataset.sh reason_mix 8000 reason_mix_8k --random_start --tail_steps 10
sim/pickup/scripts/train.sh qwen_reasoner reason_mix_8k qwen_reasoner_mix8k 1000000
sim/pickup/scripts/train.sh dp_vit_lang_reactive_subgoal reason_mix_8k fast_mix8k 20000000
sim/pickup/scripts/eval_two_level.sh experiments/pickup/fast_mix8k experiments/pickup/qwen_reasoner_mix8k reason_mix 100
```
