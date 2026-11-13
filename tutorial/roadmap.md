# Roadmap

Where this repo is going, and what is already measured. Findings are in [notes.md](notes.md).

## Goal

1. A language-conditioned policy that reliably solves `pick_target` in the sim, with the reasons
   understood.
2. From there toward a harder target: cutting flowers among moving, look-alike foliage, with a
   policy that corrects its own mistakes.
3. A deep, chaptered tutorial on LLMs, VLMs and VLAs: every component explained from the ground up,
   what works, what doesn't, the edge cases, each claim backed by a run here.
4. A site with animated explanations of the models and of training.

## Where things stand (2026-11-12)

**Results (100 unseen layouts each; details in the [lab notebook](lab-notebook.md))**

| Policy | `pick_cube` | `pick_target` |
|---|---|---|
| Scripted expert | 49/50 | 99-100/100 |
| Patch tokens + instruction, no history, 4000 demos, 5M samples (`dp_vit_lang_reactive`), after the position fix | 91-97% (1000 demos) | **93%** |
| Same, early windows repeated | | **96%** |
| Same, random start poses in the demonstrations | | **98%** |
| Same, demonstrations with recoveries | | 84-89%; follows a moved object (79-88%) and retries a missed grasp (78 of 99) |
| Same, random start poses and recoveries | | **94%**, and 89-94% with the object moved at any point, also from a random start pose |
| Same, a second seed | | 96% with the fixed position table, 90% with the old one |
| From-scratch VLA, every token, no history, 3M samples | not run | 86% / 87% with the pretrained VLM, 75% / 85% with a randomly initialised one |
| Qwen3-VL-2B VLA, every token, no history, 0.5M samples | not run | 70% / 72% |
| Same recipe before the position fix | 91-97% | 83-85% |
| Same, 1500 demos, 3M samples (before the fix) | | 64% |
| Same, 1500 demos, 1M samples (before the fix) | | 5% |
| Patch tokens + instruction, with history | 96-98% after the fix, 62-70% before | 2% (4000 demos, 3M samples, before the fix) |
| Qwen3-VL-2B VLA, one conditioning vector, 1500 demos | not run | 5-6% |
| Custom LLM -> VLM -> VLA, one conditioning vector, with history | not run | 0% (1% with a random-init VLM) |

**What was learned on the way** (field notes)

- The first-round policies were not imprecise; where they closed the gripper did not depend on the
  scene. One conditioning vector was one cause. The previous action as an input was another. A
  training budget that ended before the policy started to look was the third. A position table
  computed in half precision was the fourth.
- Looking at the scene arrives late and all at once: 0% at 1M samples, 36% at 2M, 70% at 2.5M.
- A policy trained on clean demonstrations from one start pose looks once and then follows its own
  arm, whatever its score. What the demonstrations contain decides what it looks at: random start
  poses make it watch during the reach, recoveries make it correct itself.

## Phase 1: make `pick_target` work  (done)

Steps taken: instruction tokens next to patch tokens; no history; 4000 demonstrations and 5M
samples; averaged weights; a sweep of the sampler's settings; the position-table fix; random start
poses and repeated early windows in the data; a second seed; every object of a layout named in turn
(no gain).

## Phase 2: self-correction  (works in simulation)

- [x] Evaluation with disturbances: the object is moved during the reach or as the gripper closes
      (`eval.py --disturb`), with the count of second grasp attempts.
- [x] An expert that plans again when the object has moved or a grasp closed on nothing, and
      demonstrations recorded with it (`collect.py --disturb_prob`).
- [x] Random start poses (`--random_start`), so that the arm's position alone does not give the goal away.
- [x] Policies trained on each, evaluated with and without disturbances: recoveries give
      self-correction (79-88% with a moved object, where clean-data policies score 0-12%); random
      starts give the best plain score (98%) and robustness to the start pose.
- [x] The two together: 94% on the unchanged task and 89-94% under every disturbance tried. Adding
      repeated early windows made it worse (87% / 86%).
- [ ] A target that keeps moving, not one that jumps once; look-alike objects next to the named one.
- [ ] Noise-injected demonstrations: execute a perturbed action, record the expert's clean one.

## Phase 3: redo the custom LLM -> VLM -> VLA comparison  (done)

The first comparison (0% vs 1%) said nothing about pretraining because neither policy could
localise. Re-run with every token passed to the action head, no history and 4000 demonstrations
(`vla_custom_tokens_reactive.yaml`), the pretrained VLM starts to look after 0.25M samples and the
randomly initialised one after about 1.0M; they end close, at 86% / 87% and 75% / 85% (trained weights 11 points apart, averaged weights 2). The same conditioning for pretrained Hugging
Face backbones (`vla_qwen_tokens_reactive.yaml`) takes the Qwen VLA from 5% to 70% / 72% in 0.5M
samples. Not done: a longer Qwen run, and a run that changes the conditioning alone.

## Phase 4: toward flower trimming  (started 2026-11-12)

**The final goal.** Pruning: a potted plant with a main stem topped by a flower and side shoots
branching off it. The robot cuts every side shoot off the main stem and leaves the main stem and its
flower; damaging the flower (touching it, pushing on the main stem, cutting the main stem) is a failure.
The plant stands on a motorised rotating stand with two push buttons, one for each direction; the arm
presses a button to turn the plant until a shoot on the far side is in reach. Later: leaves in the way,
a swaying plant, instructions that name a particular shoot. Steps 1-5 below were the way there with
simpler scenes; steps 6-8 are the pruning tasks themselves (models built in Blender,
`sim/pickup/assets/build_plant_blender.py`).

Each step adds one property to the previous task and asks the same three questions: can the scripted
expert do it, does the best policy so far cope without new data, and what must the demonstrations
contain for a policy to learn it. Every step reports the probes of chapter 95, not only a success
rate. Each breakthrough gets a short video.

| Step | What is added | State |
|---|---|---|
| 1 | **A target that keeps moving** (`--disturb drift`): the named object slides on a smooth random path, as a stem sways | **Works.** A policy trained with objects sliding at 1-3 cm/s in a third of its demonstrations picks up 96 / 88 / 70 of 100 at 1 / 2 / 4 cm/s (best earlier policy 94 / 78 / 27), and loses nothing elsewhere |
| 2 | **Look-alikes** (`pick_lookalike`): cubes a small colour step apart, as flowers of one plant are | **Works, with a caveat.** Learnable from look-alike scenes alone, but when the policy starts telling them apart depends on the seed: 93% and 91% for two seeds (after about 0.9M and 2M samples), 28% for a third within 5M (89% with 10M samples, gradually). With other shapes in the scene it comes early every time: the clutter policies pick the named look-alike in 86% and 97% |
| 3 | **Clutter and occlusion** (`pick_clutter`): four objects and upright leaves that hide them from the front camera | **Works.** 86-87% with leaves hiding the objects from the front camera (best earlier policy 35%); also 86% on the look-alikes |
| 4 | **A stem to grasp from the side** (`grasp_stem`): flowers on stems standing in heavy bases; close on the named stem inside a height band. The flower on top blocks a grasp from above, so the expert turns the gripper horizontal and comes in across the table (pointing straight out from the base is out of reach) | **Open.** Expert 88%; policies 0-6% at 5M samples on two seeds, 12-14% at 10M (the named stem caught in about 40%, another stem held in 40-47%: the flowers' heads are look-alikes again). Diagnosed: with one second of expert driving at the start the policy catches the named stem in 83% (alone 52%): it fails to commit to a stem and an approach side, which the expert chose by fine distances to the other stems. Holding still for 10 steps is a second, smaller problem |
| 5 | **A cut** (`cut_stem`): closing on the named stem inside the band for 3 control steps cuts it there and the part above falls; cutting another stem fails | **Works, on two seeds of three.** Expert 92%. Two changes were needed. The demonstrations must hold the jaws shut 10 frames past the cut (without it the cut ended each demonstration 3 steps after the jaws reached the stem and the policy opened again before its fingers arrived: one flower 1%, with it 97%). And the policy must train long enough to use the colour word: with three flowers it chose the stem at chance after 5M samples (20-22%, another stem cut in 31-50%) and switched suddenly at about 3.6M of 10M samples, ending at 87-89% with another stem cut in 3-6%; a second seed switched at the same point (88%), a third only climbed to 36%. 16000 demonstrations at the same budget never switched |
| 6 | **One side shoot** (`prune_one`): the plant on its stand, one shoot facing the robot; cut it off, keep the flower | **Works.** Expert 100%. Policies 77-84% on two seeds (5M samples, jaws held shut past the cut in the demonstrations); the flower is damaged in 6-9%, mostly touched |
| 7 | **Turning the plant** (`prune_turn`): one shoot at any angle; press a button to turn it into reach, then cut | **Works.** Expert 100%. Policies 83-90% on two seeds (10M samples), flower damaged in 3-10%. The turning is purposeful: it presses only when the shoot is out of reach, never switches buttons, stops with the shoot a median 3° from the robot, takes the short way in 87-88% and about 10% longer than the expert |
| 8 | **Pruning** (`prune`): 2-3 shoots at any angles, turn and cut each, the flower untouched | **In progress.** Expert 100% (about 515 control steps). Policies at 25-37% after 8 of 11 checkpoints (10M samples) and climbing; failures are mostly running out of time: it cuts and turns but is slow and wanders between shoots (twice the time triples the successes). A 20M-sample budget learns faster; runs with 20M samples, and with 4000 demonstrations, in progress |
| 9 | **Named shoots, then leaves in the way and swaying**: instructions that name which shoot to cut ("the lowest", "the one on the left", several phrasings), then leaves that hide the cut point and spring back, and a plant that sways | To do, after step 8 works. Named shoots are where language starts to matter: the best policy so far reads the instruction through a word table learned from scratch, so it only knows words that vary in its demonstrations (in steps 6-8 the instruction never varies and carries no information). Step 9 compares it with a VLA on a pretrained language backbone, tested on phrasings it never saw |

Alongside, once the scenes need it: a pretrained vision encoder with patch tokens, a higher resolution
at the wrist, and a policy that can say in words what it is about to cut (Phase 7).

## Phase 5: the tutorial (chapters and docs)

The repo trains every stage from random weights on one GPU (a 91M LLM, a VLM on top of it, a VLA on
top of that) next to a pretrained Qwen VLA. That is the spine: each chapter explains one component,
points at the code that implements it, shows a run that measures it, and lists the edge cases.

**Chapter template.** What it is and why it exists. How it works, with a figure or animation. Where
it lives in the code. An experiment with numbers. Edge cases and failure signs. A check the reader
can run.

| Part | Chapters |
|---|---|
| 0. Orientation | What "works" means; the route on one page; setup and repo map |
| 1. Language models | Tokens and tokenizers (BPE, special tokens, chat templates, padding side). The transformer block (embeddings, attention, causal mask, MLP, residual stream, norms). Positions (absolute vs rotary). Training an LLM (next-token loss, batching, warmup, AdamW, clipping, mixed precision, memory arithmetic). Generation (greedy, top-k/p, KV cache) |
| 2. Vision | Images to tokens (patches, ViT, position embeddings). Resolution against object size in pixels. Pooled vector vs patch tokens. Pretrained encoders, freezing |
| 3. Vision-language models | Joining vision to language (projector, image placeholder tokens, what causal attention lets each token see). Training a VLM (caption data, loss masking, weight transfer from the LLM). Building captions from a simulator. Testing a VLM |
| 4. Actions | Action spaces (joint targets, end-effector, gripper, control rate, chunks, normalisation). Regression vs diffusion vs flow matching; mode averaging. Sampling steps and execution horizon |
| 5. Conditioning the action head | One pooled vector vs all tokens. The action-token probe. Proprioception. History and the "keep going" shortcut |
| 6. VLAs | Anatomy (backbone + action head). The framework's recipes. Pretrained backbones, token budgets, what fits in 24 GB. The from-scratch LLM -> VLM -> VLA path |
| 7. Data | Demonstrations and scripted experts. Dataset format, windows, padding, statistics. How many demos and epochs. Recovery data: noise, retries, disturbances |
| 8. Training | The loop: samples, checkpoints, schedules, precision, EMA. Surviving long runs: crashes, hangs, auto-resume, shared memory |
| 9. Evaluation and debugging | Closed-loop evaluation, seeds, confidence intervals. Why loss and open-loop error mislead. Measuring failures. Task tolerance. Train/inference skew and how to test for it. A symptom -> check -> cause flowchart |
| 10. Case studies | `pick_cube`. `pick_target` from 0% upward. Custom path vs Qwen. Self-correction |
| 11. Toward real tasks | Clutter, look-alikes, moving targets, cutting, sim to real |
| Appendix | Glossary, config reference, reproduction commands, edge-case index, hardware notes |

Rules for the writing: measured claims cite the run; background that was not measured here is
labelled as background; every chapter ends with its edge cases; a lab notebook lists every run,
failures included.

## Phase 6: the site

Next.js, Tailwind, MDX for the chapters, Framer Motion and GSAP for diagrams, SVG for figures,
Three.js for the arm. Animated explanations, driven by real data from the runs where possible:

- a tokenizer stepping through a sentence; padding shifting a token's position
- attention over a sequence, causal mask on and off; rotary positions
- one training step; a loss curve drawing itself next to the closed-loop success of the same checkpoints
- an image becoming patches becoming tokens; a pooled vector vs all tokens reaching the action head
- flow matching turning noise into an action chunk, with the step count as a slider
- the "keep going" shortcut: what the policy can predict without looking
- a top-down map of where each policy closed the gripper; the tolerance curve with an offset slider
- 3D replay of rollouts from logged joint angles

## Phase 7: what the policy intends, and reasoning in words  (planned, not started)

The policies here go from pixels and joint angles to joint targets in one pass. They do not reason
in words, and the instruction has one job: to say which object. Two kinds of work would open them up.

**Showing what a policy intends** (no retraining)

- [ ] The plan. Draw the 15 actions of every predicted chunk as the path the gripper is about to
      take, with the step at which it means to close, in the rollout videos and on the site. When
      the object is moved, the next plan should bend to the new place.
- [ ] The uncertainty. Several draws for the same observation as a fan of paths, and one plan forming
      from noise over the flow steps.
- [ ] What it reads. Attention from the action-step tokens to the image patches and the instruction
      tokens, as a heat map over the two camera images. To be checked against the probes that change
      the input (frozen cameras, expert handover): an attention map is not evidence by itself.

**Reasoning in words** (retraining)

- [ ] Text in the demonstrations. The scripted expert knows its phase (reach, descend, close, lift,
      plan again) and where every object is, so `collect.py` can write a line of text per frame.
- [ ] Reason, then act. The backbone writes a short text before each chunk (the object, where it is,
      the phase) and the action head reads those tokens with the rest: the every-token conditioning
      of the VLM backbones already hands it the whole sequence. Background, to be checked when the
      chapter is written: embodied chain-of-thought (Zawalski et al., 2024).
- [ ] Two levels. A vision-language model names the next sub-task at a low rate and the reactive
      policy carries it out. This is the form that fits trimming: choosing what to cut is the
      reasoning, following a stem that moves is the control.
- [ ] The test that the text matters. Swap in the text of another object or phase and measure
      whether the action follows it, as `--prefix_target distractor` does for motion. A text the
      action ignores is a narration.
- [ ] The cost: latency per chunk with and without the text, and success at the same budget.

## Framework fixes ledger

Problems found in VLA Foundry (or in our own code) while doing this, and their state here.

| # | Problem | Effect | Status |
|---|---|---|---|
| 1 | `SimpleVLMProcessor` left uint8 tensors at 0-255 while scaling PIL / numpy images | VLM ignored images at inference (captioning 0%) | Fixed, tested |
| 2 | Action token appended after batch padding (`vlm_foundry_backbone`) | A short instruction is probed at a different position in training than when served | Fixed, tested |
| 3 | Same in `vlm_hf_backbone` (Qwen, PaliGemma) | Same | Fixed, tested (shared helper in `base.py`) |
| 4 | `lowdim_past_timesteps or default` treats 0 as unset | Policy is fed the next state during training | Fixed, tested |
| 5 | VLA recipes pass one pooled vector to the action head | No localisation on `pick_target` | `conditioning: all_tokens` added for both VLM backbones. With it (and no history, 4000 demonstrations): from-scratch VLA 86% / 87%, Qwen 70% / 72% |
| 6 | A dead data-loader worker makes training wait forever | Runs hang at 0% GPU | Worked around by `train_resilient.sh`; proper fix is a loader timeout |
| 7 | Data preset's PaliGemma-style `size` breaks the Qwen processor | Every batch fails | Worked around in `vla_qwen.yaml` |
| 8 | Full fine-tuning of a 2B VLM does not fit in 24 GB | Out of memory | `num_trainable_vlm_layers` added |
| 9 | `PassthroughProcessor` feeds 0-255 pixels to a from-scratch ViT | Works, but badly conditioned inputs | To test |
| 10 | Tutorial notebook says LLM -> VLM weight transfer is not automated | Misleading; `transformer.resume_from_checkpoint` works | Fix in docs |
| 11 | `collect.py` reported kept / attempted as the expert success rate | Number was meaningless | Fixed |
| 12 | Rotary position table computed in bfloat16 under autocast | 641 distinct positions out of 2048; neighbouring action tokens shared a position | Fixed, tested; `rotary_legacy_autocast` reproduces the old table bit for bit. Same run: 83% before, 93% after; a second pair of seeds 90% and 96%; single cube with history 62-70% before, 96-98% after |
| 13 | Evaluations on one port reached whichever policy server bound it first | Results filed under the wrong run | Fixed: free ports, the client checks the served checkpoint, mislabelled results are left out of summaries |
| 14 | The previous-action slot of a window at the start of an episode is a copy of the first action | The expert's first command is an input in training and absent when served | `collect.py --hold_steps` as a data-side fix did not cure a history policy (0% / 4% after 5M samples). To do in the model: slots that are padding are fed to the action head as clean values in training (`torch.where(future_mask, noisy, actions)`) and as noise by the framework's own inference adapter; an absent previous action should be absent in both |
| 15 | Preprocessing joined any Ray instance it found and passed `num_cpus` to it | Two dataset builds at once: the second dies | Fixed: `--ray_address local`, and no `num_cpus` when joining |
| 16 | Resuming a run with EMA switched on failed if its checkpoints were written without EMA | A run could not gain averaged weights after its start | Fixed, tested: the average starts from the resumed weights. The pickup configs now switch EMA on |
| 17 | Install notes said `uv sync` then `uv pip install -e .` | The second command re-resolves outside the lockfile | Fixed in the docs; `uv sync` alone installs the project |
| 18 | Tests used `/nonexistent` as a missing path | That directory exists, unreadable, on some systems (PermissionError) | Fixed: they use pytest's `tmp_path` |
| 19 | Windows are shuffled by shard, not by sample | Not measured | To test |
| 20 | A queue or wrapper that reuses a run name starts a second trainer in a live run's directory | Two trainers write the same checkpoints | Fixed, tested: `train_resilient.sh` holds `<run>.pid` while it runs and a second copy with the same name exits |
| 21 | `eval.py` wrote `results.json` and the trace files in place | A monitor reading them could see half a file | Fixed: written to a temporary file and renamed |
| 22 | `inference_vlm.py` chose its checkpoint by sorting the file names as text | With ten or more checkpoints it loaded `checkpoint_9.pt`, not the newest | Fixed, tested: it uses `get_latest_checkpoint` |
| 23 | `vlm_foundry_backbone` read `vit.img_size`, which a pretrained vision tower (`vit_hf`) does not have | AttributeError as soon as such a VLM is used as a policy backbone | Fixed, tested: the images are passed on as the tower's processor made them. The rest of that route has not been run here |
| 24 | The stage-1 tokenizer truncates on the left (its published configuration) | The language model is trained on the last 257 tokens of every longer document, not the first | Recorded, not changed; effect not measured |
| 25 | A Hugging Face backbone's rotary frequencies are a non-persistent buffer: training casts it to bfloat16 with the model, the policy server builds the model in float32 | The backbone may see slightly different position angles in training and when served | To test (read from the code, not measured) |
| 26 | `open_loop_check.py` assumed the policy takes as many past steps as the shards store | For a policy without history it skipped every window and ended in a TypeError | Fixed, tested: the rows are taken from the window's masks, for either kind of policy |

## Caveats

- One training seed per run; small differences are noise.
- One of the machines used was unreliable; long runs died or hung and were resumed automatically.
  Important results are re-run.
- Everything is simulation with rendered images; nothing here has touched a real camera yet.
