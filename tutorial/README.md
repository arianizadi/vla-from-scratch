# VLA from scratch: the tutorial

This tutorial teaches the models behind robot manipulation from the ground up: language models,
vision-language models (VLMs), vision-language-action policies (VLAs), and how to train, evaluate
and debug them. It is written for someone who programs in Python and has trained a small neural
network, and who has not built a language model, a VLM or a robot policy. Every chapter explains
one component, points at the code in this repository that implements it, reports what was measured
here, and lists the ways it fails. The running example is a MuJoCo pick-up simulator
([`sim/pickup`](../sim/pickup)) with two tasks: `pick_cube`, one red cube, and `pick_target`, three
of four objects with an instruction that names one. Every measured number comes from
[lab-notebook.md](lab-notebook.md), the runs that failed included.

Where it stands: `pick_target` is solved in simulation. The patch-token policy with the instruction
as tokens and no history (`dp_vit_lang_reactive`), trained on 4000 demonstrations for 5M samples,
reaches 93% on clean demonstrations and 98% with random start poses in them; with random start
poses and recoveries it scores 94% and holds 89-94% when the object is moved during the reach or as
the gripper closes. All chapters have `status: draft`. Where a chapter and the lab notebook disagree, the
notebook is right.

## Chapters

### Part 0: Orientation

| # | Chapter | After reading you can |
|---|---|---|
| 00 | [The Route on One Page](chapters/00-the-route.md) | Follow the ten-step order of work that takes a robot policy to a measured closed-loop success rate, and say where this repository stands on it. |
| 01 | [Setup and a First Run](chapters/01-setup-and-first-run.md) | Install the repository, run the whole loop once on the single-cube task as a smoke test, then start the three-object run (93% here after the position fix; 83% / 85% before it) and read its success curve and grasp map. |
| 02 | [From Language Models to Robot Policies](chapters/02-how-we-got-here.md) | Name the published idea behind each part of a vision-language-action policy, the problem it solved, and what is still open. |
| 03 | [The Model Families Today](chapters/03-the-landscape.md) | Place any robot policy, published or in this repository, on a map of five design choices and say what follows from each choice. |

### Part 1: Language models

| # | Chapter | After reading you can |
|---|---|---|
| 10 | [Text Becomes Tokens](chapters/10-tokens.md) | Trace an instruction from a Python string to the padded batch of token ids a model receives, and check where that conversion differs between training and serving. |
| 11 | [The Transformer Block](chapters/11-the-transformer-block.md) | Trace token ids to next-token logits through this repository's transformer, tensor by tensor, and derive its 91.4M parameters from the config. |
| 12 | [Telling Tokens Where They Are](chapters/12-positions.md) | Explain how rotary position embeddings make attention depend on relative offsets, find where position enters each model in this repository, and check that a policy's tokens keep the same offsets in training and serving and a distinct position each. |
| 13 | [Training a Language Model](chapters/13-training-a-language-model.md) | Run stage 1 of the from-scratch path and explain its loss, learning rate, memory use and throughput with your own arithmetic. |
| 14 | [Generating Text](chapters/14-generation.md) | Turn logits into text with greedy decoding or sampling, explain what the key-value cache saves, and spot a prompt, padding or position-table mismatch in this repository's generate methods. |

### Part 2: Vision

| # | Chapter | After reading you can |
|---|---|---|
| 20 | [Images Become Tokens](chapters/20-images-to-tokens.md) | Turn a camera frame into patch tokens with a Vision Transformer, count what it costs, and check that training and inference use the same pixel range. |
| 21 | [One Vector or Many Tokens](chapters/21-pooled-vs-patch-tokens.md) | Tell a pooled image embedding from a sequence of patch tokens, predict which one a manipulation policy needs, and find where this repository produces each. |
| 22 | [Pretrained Vision Encoders](chapters/22-pretrained-vision-encoders.md) | Choose between a from-scratch ViT, a pretrained encoder from the timm library and a VLM's own vision tower, knowing what CLIP, SigLIP and DINOv2 features carry and what each costs in tokens. |

### Part 3: Vision-language models

| # | Chapter | After reading you can |
|---|---|---|
| 30 | [Joining Vision and Language](chapters/30-joining-vision-and-language.md) | Trace an image and an instruction through the VLM's forward pass, write out the token sequence for one image and for four, and say which positions can attend to which. |
| 31 | [Training a VLM](chapters/31-training-a-vlm.md) | Build caption data from a scene you control, run stage 2 on it, and say which tokens the loss scores, which weights were pretrained and what the captions can teach. |
| 32 | [Testing a VLM](chapters/32-testing-a-vlm.md) | Find out whether a vision-language model uses its images with a behavioural check, a counterfactual-image check and a both-paths check, and read the three together to locate a fault. |

### Part 4: Actions and policies

| # | Chapter | After reading you can |
|---|---|---|
| 40 | [What an Action Is](chapters/40-action-spaces.md) | Read an 8-number action of the pick-up simulator, convert a joint error in radians into millimetres at the gripper, and explain how actions are normalised for training. |
| 41 | [Generating Actions with Flow Matching](chapters/41-flow-matching.md) | Explain why regression averages valid actions into an invalid one, derive this repository's flow-matching objective from its code, and choose the number of flow steps from a measured sweep. |
| 42 | [Action Chunks and Closed-Loop Execution](chapters/42-chunks-and-execution.md) | Explain why a policy predicts 15 future actions and executes only the first few, trace one re-plan through eval.py and policy_server.py, and choose and test an execution horizon. |
| 43 | [Imitation Learning and Why It Breaks](chapters/43-imitation-learning.md) | Explain why a policy that copies demonstrations accurately can still fail when it runs, name the four causes, and find each one in this repository's expert, data and measurements. |

### Part 5: Conditioning the action head

| # | Chapter | After reading you can |
|---|---|---|
| 50 | [What the Action Head Sees](chapters/50-what-the-action-head-sees.md) | Read off, for any config in this repository, the exact token sequence the action head receives, and tell whether the object's position can reach it. |
| 51 | [The Instruction Has to Reach the Action Head](chapters/51-language-conditioning.md) | Trace an instruction from a string to the tokens the action head attends to, and tell from two numbers of an evaluation and two tests whether a trained policy follows it. |
| 52 | [History and the Shortcut It Opens](chapters/52-history-and-shortcuts.md) | Tell which history inputs a config gives a policy and how they are filled in training and when served, explain how the previous action lets a policy fit demonstrations without looking at the scene, and test a trained policy for that shortcut with two probes and the as-served baseline. |

### Part 6: Vision-language-action models

| # | Chapter | After reading you can |
|---|---|---|
| 60 | [Anatomy of a VLA](chapters/60-anatomy-of-a-vla.md) | Trace one training pass and one sampling pass of a VLM-backed flow-matching policy with shapes, and say which parameters each config in this repository trains. |
| 61 | [Building on a Pretrained VLM](chapters/61-pretrained-backbones.md) | Configure a pretrained Hugging Face VLM as a policy backbone, budget its image tokens and GPU memory, train only its last layers, and read the resulting numbers without over-claiming. |
| 62 | [The From-Scratch Path](chapters/62-the-from-scratch-path.md) | Train a language model, a VLM and an action policy in sequence with custom_path.sh, pass weights between stages, and judge what a pretrained-versus-random control can show. |

### Part 7: Data

| # | Chapter | After reading you can |
|---|---|---|
| 70 | [Demonstrations](chapters/70-demonstrations.md) | Generate a demonstration dataset with the scripted expert, choose what its episodes contain, say what its files and arrays hold, and open a training shard to check it. |
| 71 | [How Much Data, How Much Training](chapters/71-how-much-data-and-training.md) | Count a training budget in demonstrations, windows, samples and epochs, read a run's checkpoints for the point where the policy starts to use the scene, and choose a budget that does not end before it. |

### Part 8: Training

| # | Chapter | After reading you can |
|---|---|---|
| 80 | [The Training Loop](chapters/80-the-training-loop.md) | Follow a run from its config to its checkpoint files, read its log line, and choose which saved weights to evaluate. |
| 81 | [Surviving Long Runs](chapters/81-surviving-long-runs.md) | Start a multi-hour training run so that it resumes by itself after a crash or a hang, leaves the CPUs and GPUs of a shared machine to its neighbours, and can be judged healthy from the dashboard. |

### Part 9: Evaluation and debugging

| # | Chapter | After reading you can |
|---|---|---|
| 90 | [Closed-Loop Evaluation](chapters/90-closed-loop-evaluation.md) | Run a policy on unseen layouts, read results.json, turn a training run into a success curve, and tell a real difference between two success rates from noise. |
| 91 | [Loss Is Not Success](chapters/91-loss-is-not-success.md) | Explain why a low training loss can come with a 0% success rate and why the open-loop action error ranks a 4% policy above a 74% one, use the open-loop check for the one thing it can show, and evaluate closed-loop from the first checkpoint. |
| 92 | [Measuring How a Policy Fails](chapters/92-measuring-failure.md) | Record where the gripper is when it first closes, and use that, the scripted expert's numbers and the task's tolerance to tell which kind of failure a low success rate hides. |
| 93 | [Train and Inference Must See the Same Thing](chapters/93-train-inference-skew.md) | Find differences between the tensors a model is trained on and the tensors it is served by comparing one window through both paths, and keep them out with counterfactual checks and unit tests. |
| 94 | [A Debugging Flowchart](chapters/94-debugging-flowchart.md) | Find the cause of a low closed-loop success rate with seven checks in a fixed order, and test a policy that succeeds with an eighth, an object that is moved. |
| 95 | [Probing a Policy That Works](chapters/95-what-a-policy-looks-at.md) | Find out what a trained policy's actions depend on by freezing its cameras, letting the expert drive its first steps, moving the object and changing the start pose, and read each result against the scripted expert under the same change. |

### Part 10: Case studies

| # | Chapter | After reading you can |
|---|---|---|
| 100 | [Case Study: One Cube](chapters/100-case-pick-cube.md) | Repeat, run by run, the route that took the single-cube task from a scripted expert to policies above 90%, and say what each run tested, what it measured and what it changed. |
| 101 | [Case Study: Three Objects, From 0% to 98%](chapters/101-case-pick-target.md) | Follow the debugging method through the whole history of `pick_target`, wrong turns included, and run the same checks, in the same order, on a policy of your own. |

### Part 12: Appendix

| # | Chapter | After reading you can |
|---|---|---|
| 120 | [Glossary](chapters/120-glossary.md) | Look up any term the chapters use, with its meaning in this repository and the chapter that introduces it. |
| 121 | [Edge Cases and Failures](chapters/121-edge-cases.md) | Look up what you see, find why it happens and how to check it, and follow the link to the chapters that explain it. |

### Still to come

Their results are in the lab notebook and the [field notes](notes.md); the chapters are not written.

| # | Chapter |
|---|---|
| 72 | Which demonstrations teach what: random start poses, recoveries, every object named, early windows repeated |
| 102 | The from-scratch path against a pretrained backbone, with every token handed to the action head |
| 103 | Self-correction: disturbances in the evaluation and recoveries in the data |
| 110 | Toward real tasks: clutter, look-alike objects, moving targets, cutting |

## Three ways through

Chapters link to each other and to the code, so they can be read out of order with the
[glossary](chapters/120-glossary.md) open. Read in numerical order for the whole thing, running the
"Check yourself" exercises with the repository open next to the chapter.

**Train a policy this week.** The language and vision parts can wait.

1. [00 The Route on One Page](chapters/00-the-route.md) and
   [01 Setup and a First Run](chapters/01-setup-and-first-run.md): the order of work, a smoke test
   on one cube, and the recommended three-object run.
2. [70 Demonstrations](chapters/70-demonstrations.md) and
   [71 How Much Data, How Much Training](chapters/71-how-much-data-and-training.md): what to put in
   the demonstrations, and a budget that does not end before the policy starts to look.
3. [50 What the Action Head Sees](chapters/50-what-the-action-head-sees.md) and
   [52 History and the Shortcut It Opens](chapters/52-history-and-shortcuts.md): give the action
   head spatial tokens, and start without history.
4. [80 The Training Loop](chapters/80-the-training-loop.md) and
   [81 Surviving Long Runs](chapters/81-surviving-long-runs.md): a run, its checkpoints and
   averaged weights, and keeping it alive.
5. [90 Closed-Loop Evaluation](chapters/90-closed-loop-evaluation.md) and
   [95 Probing a Policy That Works](chapters/95-what-a-policy-looks-at.md): a success curve over
   every checkpoint, then the probes that show what the policy looks at.
6. [101 Case Study: Three Objects, From 0% to 98%](chapters/101-case-pick-target.md): the whole
   route on one task, wrong turns included.

**Understand the models from the ground up.** In order:

1. [02 From Language Models to Robot Policies](chapters/02-how-we-got-here.md) and
   [03 The Model Families Today](chapters/03-the-landscape.md): where each idea came from and the
   map of design choices.
2. Part 1, [10](chapters/10-tokens.md) to [14](chapters/14-generation.md): tokens, the transformer
   block, positions, training a language model and generating text.
3. Part 2, [20](chapters/20-images-to-tokens.md) to [22](chapters/22-pretrained-vision-encoders.md):
   images as tokens, one vector against many, pretrained encoders.
4. Part 3, [30](chapters/30-joining-vision-and-language.md) to [32](chapters/32-testing-a-vlm.md):
   joining vision and language, training a VLM and testing that it uses its images.
5. Part 4, [40](chapters/40-action-spaces.md) to [43](chapters/43-imitation-learning.md): actions,
   flow matching, chunks and imitation learning.
6. Parts 5 and 6, [50](chapters/50-what-the-action-head-sees.md) to
   [62](chapters/62-the-from-scratch-path.md): what the action head receives, and the whole VLA,
   from a pretrained backbone or built stage by stage here.

**Debug a policy that does not work.** Start from the symptom.

1. [94 A Debugging Flowchart](chapters/94-debugging-flowchart.md): the checks in a fixed order,
   from the scripted expert to an object that is moved.
2. [92 Measuring How a Policy Fails](chapters/92-measuring-failure.md): where the gripper closed,
   the slope, and the tolerance of the task.
3. [91 Loss Is Not Success](chapters/91-loss-is-not-success.md) and
   [71 How Much Data, How Much Training](chapters/71-how-much-data-and-training.md): a flat 0% may
   be a run that has not reached its transition.
4. [52 History and the Shortcut It Opens](chapters/52-history-and-shortcuts.md) and
   [93 Train and Inference Must See the Same Thing](chapters/93-train-inference-skew.md): an input
   that holds the answer in training, and one window through both paths.
5. [12 Telling Tokens Where They Are](chapters/12-positions.md): a fault that training and serving
   share, the position table.
6. [95 Probing a Policy That Works](chapters/95-what-a-policy-looks-at.md): when it succeeds and
   still fails as soon as something moves.
7. [Edge Cases and Failures](chapters/121-edge-cases.md): look the symptom up, then follow the link
   to the chapter that explains it.

## Other files

| File | What it is |
|---|---|
| [lab-notebook.md](lab-notebook.md) | Every number measured in this repository, with the machine, data and budget it was measured on, and the bugs found. Chapters cite it; a number that is not there was not measured here. |
| [notes.md](notes.md) | Field notes: what worked, what did not, what we got wrong, and what to do about it. Chapters cite its entries by title. |
| [roadmap.md](roadmap.md) | Where the repository is going: the phases, the planned chapters, and the ledger of framework fixes. |
| [STYLE.md](STYLE.md) | How the chapters are written: reader, voice, the three kinds of statement, chapter structure and the list of interactive figures. |
