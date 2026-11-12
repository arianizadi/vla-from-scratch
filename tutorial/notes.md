# Field notes: what worked, what didn't

A running record of training robot policies on the pickup sim, written so that someone else can
skip the dead ends. Each entry says what we believed, what we measured, and what to do about it.
Numbers are closed-loop success on 100 layouts the policy never saw unless stated otherwise; one
training seed per run, so treat differences of a few percent as noise.

Status legend: **confirmed** (measured here), **open** (hypothesis, experiment queued).

## The short version

1. Check the environment with a scripted expert before training anything.
2. Judge policies by closed-loop success only. Training loss and open-loop error do not predict it.
3. When a policy fails, measure *how* it fails before changing anything.
4. Know how much error the task tolerates.
5. Give the action head spatial tokens, not one pooled vector.
6. If the scene is ambiguous, the instruction must reach the action head.
7. Feed inference through exactly the code path used in training, and test that it does.
8. Checkpoint often and auto-resume; long runs die.

## 1. Verify the task with an expert first  (confirmed)

**What we did.** `sim/pickup/eval.py --expert` runs the scripted grasp expert through the same
evaluation loop, seeds and success test as a learned policy.

**Result.** 99/100 on `pick_target`, closing the gripper 5 mm (median) from the object.

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

The cube and T block forgive 2 cm. The two round objects need 1 cm or better. A policy that is
"usually within 2 cm" would score about 95% on cubes and about 20% on cans.

## 5. One pooled vector is not enough for grasping  (confirmed on `pick_cube`, open on `pick_target`)

| Conditioning the action head receives | Task | Success |
|---|---|---|
| CLIP: one vector per image + one for the text | `pick_cube`, 200 demos | 4% |
| ViT: every 16x16 image patch as a token | `pick_cube`, 200 demos | 30% |

VLA Foundry's VLA recipe (`vlm_backbone`, `vlm_foundry_backbone`) appends an action token to the
prompt and passes only that token's hidden states to the action head: again one vector. Both
from-scratch VLAs built that way reached the named object at chance (section 3).

**Fix in this fork.** `vlm_foundry_backbone` has `conditioning: all_tokens`, which passes the hidden
states of every image and text token. `vit_backbone` has `text_tokens`, which adds the instruction
to a patch-token policy.

## 6. A failing policy ignores the scene; it does not average  (confirmed)

The ViT policy has no language input. On `pick_cube` (one object) it reaches 74%. On `pick_target`
(three objects) it closes 11 cm from the nearest one. The first reading was that it reaches for a
point between the objects. The per-episode records of the 1500-demonstration runs say otherwise:

- the closing positions fill the area where objects can spawn evenly; they do not gather between
  the three objects of their own scene;
- regressing the closing position on the position of the named object gives a slope of 0.03 and an
  R² under 0.02 (a policy that goes to the named object has slope 1);
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

## 7. Train/inference mismatches hide in preprocessing  (confirmed)

Three found so far, none of which raised an error:

- **Pixel range.** `SimpleVLMProcessor` scaled PIL and numpy images to [0, 1] but left uint8 tensors
  at 0-255. Training fed tensors, inference fed PIL. The VLM scored 0% on a captioning check that it
  passes at 85% after the fix.
- **Padding moves the probe.** The action token was appended after the batch padding. In training a
  short instruction sits in a batch with longer ones, so its action token is one position later
  (behind a padding token) than when the same instruction is served alone. Fixed: the token goes
  directly after the last real token. Effect on success not yet measured.
- **Zero means "unset".** `lowdim_past_timesteps or default` treats 0 past steps as missing and
  falls back to 1, which hands the policy the state one step in the future during training. Found by
  reading the code before running such a config; the fix and its test are written and land with
  the next code change.

**What to do.** For every model, run a check that pushes the same sample through the training
pipeline and through the serving path and compares the tensors the model receives.

## 8. Long runs die  (confirmed)

One of the machines used was unreliable: stage-3 runs died or hung three times. A dead data-loader
worker does not crash the trainer; it waits forever at 0% GPU. `scripts/train_resilient.sh` restarts
from the newest checkpoint when a run exits or when its log stops moving for 7 minutes.

## 9. Pretraining a small VLM on captions  (inconclusive)

The from-scratch path (91M LLM on web text, then a VLM on 40k captioned sim renders, then a VLA)
gave 0/100 with pretraining and 1/100 without. Because of section 5 neither policy could localise
anything, so this comparison says nothing yet about pretraining. It will be re-run with
`conditioning: all_tokens`.

## 10. Operating many runs on one machine  (confirmed)

Things that cost time here and have nothing to do with models:

- **Cap CPU threads per process.** Numerical libraries start one thread per core. On a 128-core
  machine each policy server started 256 threads and used about 20 cores doing nothing useful;
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
- **Follow the lockfile.** The framework's install notes say `uv sync` and then
  `uv pip install -e .`. The second command re-resolves outside the lockfile and silently replaced a
  pinned dependency with a newer commit. `uv sync` alone already installs the project.
- **Evaluate while training.** Evaluating every checkpoint as it appears costs little next to
  training and turns one run into a curve; a single final number hides whether the policy was
  still improving, had peaked, or never moved.

## Open questions being tested

- Does adding the instruction to the patch-token policy fix target selection? (`dp_vit_lang`)
- Does removing the previous action and the second frame force the policy to look at the scene on
  every step? (`dp_vit_lang_reactive`)
- How far do more demonstrations and more epochs take the best of those?
- Do averaged (EMA) weights help?
- Does demonstration data with recoveries make the policy correct its own misses?
