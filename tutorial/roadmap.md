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

## Where things stand (2026-11-11)

**Results (100 unseen layouts each)**

| Policy | `pick_cube` | `pick_target` |
|---|---|---|
| Scripted expert | 49/50 | 99/100 |
| ViT patch tokens, no language | 74% | 1% |
| Qwen3-VL-2B VLA | not run | 6% |
| Custom LLM -> VLM -> VLA | not run | 0% |
| Same VLA, random-init VLM | not run | 1% |

**Diagnosis**

- The simulator and scoring are fine: the expert scores 99/100 on the same layouts.
- The three from-scratch policies run the whole grasp motion but close the gripper about 10 cm from
  the nearest object, and reach the named object 29-33% of the time. Chance is 33%. They learned the
  motion, not the target.
- Two structural causes. The only architecture here that grasps (patch-token ViT) has no language
  input, so with three objects it heads between them. The architectures with language (both VLA
  recipes) give the action head a single pooled vector, the same bottleneck that made the pooled
  CLIP policy fail on the single cube (4% vs 30%).
- Likely third cause, untested: the policy is given its previous action, so after the first step
  "keep going" predicts the expert without looking at the scene.
- Grasp tolerance: cube and T block forgive 2 cm, can and hex nut need 1 cm.

## Phase 1: make `pick_target` work

One change per step, same 100 evaluation layouts, and for every run four numbers: success, how often
the gripper closes at the named object, the median miss, and per-object success. Raw and EMA weights
are both evaluated from the same run.

| Step | Change | Config | GPU time | Question it answers |
|---|---|---|---|---|
| 1 | Instruction tokens + patch tokens | `dp_vit_lang` | 30 min | Does the policy start choosing the named object? |
| 2 | Same, no history (one frame, no previous action) | `dp_vit_lang_reactive` | 20-30 min | Was "keep doing what I was doing" hiding the scene from the policy? |
| 3 | Grasp diagnostics for the Qwen VLA | existing checkpoint | 10 min | Fills the last row of the failure table |
| 4 | Winner of 1-2 on 4000 demos, about 5M samples | `pick_target_4k` | 2.5 h | How far do data and training budget go? |
| 5 | Inference settings on the best checkpoint | none | 15 min | Free gains |

Targets, to be judged against the data: after steps 1-2 the named-object rate far above 33% (aiming
for over 80%) even if success is still modest; after step 4, 80% success overall with the can and
hex nut above 60%.

If neither step 1 nor step 2 gets the policy to choose the named object: oversample the first
windows of each episode, or condition on a pretrained text-and-image encoder with patch tokens.

## Phase 2: self-correction

The demonstrations are single clean passes, so no policy here has seen a miss followed by a retry.

- [ ] Noise-injected demonstrations: execute a perturbed action, record the expert's clean one, and
      switch the noise off near the grasp.
- [ ] Retry demonstrations: start from a failed grasp and record the expert reopening and regrasping.
- [ ] Moving target: nudge the object during the approach; the expert re-plans every step.
- [ ] Evaluation with disturbances: push the object mid-episode and count recoveries.

## Phase 3: redo the custom LLM -> VLM -> VLA comparison

The first comparison (0% vs 1%) says nothing about pretraining because neither policy could
localise. Rerun both with `vla_custom_tokens.yaml`: pretrained VLM vs random-init VLM, 1M samples
each. Optional: the same change for the Qwen backbone.

## Phase 4: toward flower trimming

- [ ] A cluttered scene with look-alike distractors and partial occlusion.
- [ ] Pretrained vision encoder with patch tokens, higher resolution at the wrist.
- [ ] A cutting action in place of a grasp, with a success test that penalises cutting the wrong stem.

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

## Framework fixes ledger

Problems found in VLA Foundry (or in our own code) while doing this, and their state here.

| # | Problem | Effect | Status |
|---|---|---|---|
| 1 | `SimpleVLMProcessor` left uint8 tensors at 0-255 while scaling PIL / numpy images | VLM ignored images at inference (captioning 0%) | Fixed, tested |
| 2 | Action token appended after batch padding (`vlm_foundry_backbone`) | A short instruction is probed at a different position in training than when served | Fixed, tested |
| 3 | Same in `vlm_hf_backbone` (Qwen, PaliGemma) | Same | To do |
| 4 | `lowdim_past_timesteps or default` treats 0 as unset | Policy is fed the next state during training | Fix and test written, to be applied |
| 5 | VLA recipes pass one pooled vector to the action head | No localisation on `pick_target` | `all_tokens` mode added for the Foundry VLM backbone; effect to be measured |
| 6 | A dead data-loader worker makes training wait forever | Runs hang at 0% GPU | Worked around by `train_resilient.sh`; proper fix is a loader timeout |
| 7 | Data preset's PaliGemma-style `size` breaks the Qwen processor | Every batch fails | Worked around in `vla_qwen.yaml` |
| 8 | Full fine-tuning of a 2B VLM does not fit in 24 GB | Out of memory | `num_trainable_vlm_layers` added |
| 9 | `PassthroughProcessor` feeds 0-255 pixels to a from-scratch ViT | Works, but badly conditioned inputs | To test |
| 10 | Tutorial notebook says LLM -> VLM weight transfer is not automated | Misleading; `transformer.resume_from_checkpoint` works | Fix in docs |
| 11 | `collect.py` reports kept / attempted as the expert success rate | Number is meaningless | To fix |

## Caveats

- One training seed per run; small differences are noise.
- The training workstation has a faulty memory line; long runs die or hang and are resumed
  automatically. Important results are re-run.
- Everything is simulation with rendered images; nothing here has touched a real camera yet.
