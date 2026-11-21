# Training a VLA policy: the recipe as measured

How to train a good vision-language-action policy, written down from what this repository measured: the training
lessons, the architectures that came out of them (the fast policy on a frozen encoder, architecture v2, architecture
v3), each with its training recipe and its failure modes, the pipeline stage by stage with commands, costs and the
rules we decide by, and what did not work. The arm here is a simulated 7-joint arm with a parallel gripper in MuJoCo;
section 8 says, briefly, what changes for another arm.

Every number is from [lab-notebook.md](lab-notebook.md), cited by the keys below. Where this file and the notebook
disagree, the notebook is right. Two machines are named as the notebook names them: machine A (one GPU) and machine
B (several GPUs); every number below is from machine B unless marked. Everything is in simulation with
rendered images; nothing here has touched a real camera.

**Sources** (sections of the lab notebook):

| Key | Section |
|---|---|
| [EXP] | [The scripted expert](lab-notebook.md#the-scripted-expert) and [grasp tolerance](lab-notebook.md#grasp-tolerance) |
| [PC] | [`pick_cube` (machine A)](lab-notebook.md#pick_cube-machine-a) and [`pick_target`, first round](lab-notebook.md#pick_target-first-round-machine-a) |
| [R2] | [`pick_target`, second round](lab-notebook.md#pick_target-second-round-machine-b) and its subsections (history, budget, inference settings) |
| [FIX] | [After the position fix](lab-notebook.md#after-the-position-fix-machine-b): what the demonstrations contain, robustness to disturbances |
| [S1] to [S9] | the ladder's steps: [1](lab-notebook.md#step-1-a-target-that-keeps-moving---disturb-drift), [2](lab-notebook.md#step-2-look-alikes-pick_lookalike), [4](lab-notebook.md#step-4-a-stem-to-grasp-from-the-side-grasp_stem), [5](lab-notebook.md#step-5-a-cut-cut_stem), [6](lab-notebook.md#step-6-one-side-shoot-prune_one-new-physics), [7](lab-notebook.md#step-7-turning-the-plant-prune_turn-new-physics), [8](lab-notebook.md#step-8-the-full-task-prune), [9](lab-notebook.md#step-9-named-shoots-prune_named) |
| [S8-rob] | step 8, its robustness paragraphs: "Test situations and robustness" to "Frozen DINOv2 on paired demonstrations" |
| [REASON1] | [Reasoning in words (stage 1)](lab-notebook.md#reasoning-in-words-phase-7-stage-1) |
| [V2] | [Architecture v2](lab-notebook.md#architecture-v2-eyes-a-written-program-and-memory-phase-7-stage-2), and [architecture-v2.md](architecture-v2.md) |
| [V3-1] | [Architecture v3, steps 1 to 3](lab-notebook.md#architecture-v3-the-language-the-truth-with-attributes-the-plan-steps-1-to-3) and [step 4's runner](lab-notebook.md#step-4s-runner-built-and-measured-on-cpu-no-fast-level-yet) |
| [E13] | [E13](lab-notebook.md#e13-v3-over-the-fast-level-as-it-is-on-rungs-1-to-4) |
| [E13b] | [After E13](lab-notebook.md#after-e13-the-runner-fixed-the-ties-diagnosed-the-fast-level-fine-tuned-on-them) |
| [V3-6] | [Step 6: the eyes with attributes](lab-notebook.md#step-6-the-eyes-with-attributes-e16-the-shortcut-probe-and-v3-on-the-eyes) |
| [V3-5] | [Step 5: the program writer](lab-notebook.md#step-5-the-program-writer-for-v3s-language-e15) |
| [V3-7] | [Step 7: retraining on the fast level's own rollouts](lab-notebook.md#step-7-retraining-on-the-fast-levels-own-rollouts-the-fast-level-and-the-eyes-and-calibration) |
| [V3-8] | [After step 7: the LoRA writer and the DINOv2 fast level](lab-notebook.md#after-step-7-the-writer-fine-tuned-with-lora-and-the-fast-level-on-a-frozen-dinov2-encoder) |
| [CUT] | [The cut, diagnosed](lab-notebook.md#the-cut-diagnosed-where-the-fast-levels-8-mm-come-from) |
| [OPS] | [Throughput](lab-notebook.md#throughput), [Operations](lab-notebook.md#operations) (including the pipeline timings and disk), [Bugs](lab-notebook.md#bugs-found-and-fixed) |

## 1. What to train today

Two systems, depending on whether the task needs reasoning about the scene.

**A single policy that holds when the scene looks different.** The 1 s memory policy (a flow-matching action
transformer) on a **frozen pretrained image encoder** (DINOv2-B with registers, or DINOv3-B/16, at 224 pixels),
trained on the demonstrations **plus the same demonstrations re-rendered in other looks** (paired re-rendering:
lighting, camera pose, bench), 20M samples. Report card on `prune`, final averaged weights, 100 layouts per cell, the
held-out half of each visual situation in brackets [S8-rob]:

| Policy | plain | lighting | camera | bench | distractors | wind | bump | sticky button |
|---|---|---|---|---|---|---|---|---|
| from-scratch ViT, plain demonstrations (the old recipe) | 87 | 3 (0) | 2 (2) | 2 (0) | 72 (85) | 82 | 79 | 38 |
| frozen DINOv2, plain demonstrations, seed 42 | 92 | 53 (55) | 15 (5) | 0 (0) | 74 (82) | 89 | 89 | 50 |
| **frozen DINOv2, paired demonstrations**, seed 42 | 88 | 80 (74) | 71 (58) | 44 (30) | 85 (78) | 84 | 78 | 51 |
| the same, seed 43 | 81 | 81 (72) | 62 (44) | 46 (33) | 65 (71) | 74 | 71 | 43 |
| the same, seed 44 | 63 | 74 (65) | 58 (42) | 36 (17) | 62 (64) | 73 | 57 | 39 |
| **frozen DINOv3-B/16, paired demonstrations**, seed 42 | 91 | 78 (72) | 67 (58) | 52 (28) | 86 (85) | 91 | 83 | 50 |

Config `sim/pickup/configs/dp_dinov2_lang_frames_1s.yaml` (or `dp_dinov3_lang_frames_1s.yaml`); recipe in section
3.1, pipeline in section 4, all of it in one command with [`sim/pickup/scripts/pipeline.sh`](../sim/pickup/scripts/pipeline.sh).

**A system that reasons about the scene: architecture v3.** Trained eyes read every part of the plant into a scene
memory; a frozen language model with LoRA adapters compiles the instruction into a typed program; code evaluates the
program over every world the unknown facts allow and hands a fast level its sub-goal; the fast level is the sub-goal
policy on frozen DINOv2. Episodes solved of 200 (`order` of 300), evaluation layouts, written programs, rollout eyes
[V3-8]:

| `prune` | `order` | `nothing_to_do` | `ties` | `sick` | `buds` | held out: `order` | `ties` | `sick` | `buds` | all |
|---|---|---|---|---|---|---|---|---|---|---|
| 186 | 276 | 197 | 194 | 172 | 189 | 245 | 195 | 175 | 187 | **2016 of 2600** |

Its predecessor, v2, solves 179 of 200 training and 160 of 200 held-out reasoning instructions (means of 5 and 7
repeats) [V2]. Section 3 describes both. v3's fast level was trained on plain demonstrations: how v3 holds under the
visual changes of the table above is not measured [V3-8].

## 2. The training lessons behind the design

Each lesson is a measurement that changed what we build. They are in roughly the order they were learned.

1. **Check the task with the scripted expert, through the same evaluation loop, before collecting anything.** The
   expert picks 100/100 on `pick_target` with its gripper closing a median 5 mm from the object [EXP]; on the stem
   grasp it managed only 88/100 and the policies never passed 14 [S4]. The expert is also the precision floor a policy
   can learn: its cuts land 5.8 to 5.9 mm from the shoot's axis, and no policy has beaten that [CUT].
2. **Training loss and open-loop error do not predict success; only closed-loop evaluation does.** The pooled-CLIP
   policy had the lowest loss and open-loop error of four and succeeded in 4% [PC].
3. **Give the action head every token, not one vector.** VLAs conditioned on one vector: 0 to 6% on `pick_target` [PC].
   With every token passed: the from-scratch VLA 86% / 87%, the Qwen3-VL-2B VLA 70% / 72% in 0.5M samples [FIX: "From-scratch
   VLA" and "Qwen3-VL-2B backbone"].
4. **No previous action, and no previous frame; a frame from one second ago is the memory that works.** With the
   previous action as input the policy copies it: 0 to 4% on `pick_target` [R2: "What the history policy depends on"],
   0 on pruning. The previous frame alone: 53 / 35 against 81 / 79 without memory. The frame 20 control steps (1 s)
   back: 70, 89 and 90 on three seeds against 79, 56 and 78 without memory [S8: "Memory"]. It does not help a sub-goal
   fast level that is strong without it [V2: "The 1 s memory fast level, second seed"].
5. **Looking at the scene arrives late and all at once, so plan a long budget from the start.** 1500 demonstrations:
   5% at 1M samples, 64% at 3M; 4000 demonstrations at 5M: 85% [R2: "Final checkpoints"]. The learning rate follows a
   cosine over the *planned* budget, so a run planned for 10M uses the colour word at 3.6M where a 5M run never did
   [S5], and pruning reaches 78 to 81 at 20M against 22 to 34 at 10M [S8]. Past 20M nothing more: 80 at 40M [S8].
6. **What the demonstrations contain decides what the policy looks at.** Random start poses: 98% and robust to the
   start pose. Recoveries (the object moved once, the expert planning again): 79 to 88% with the object moved, where
   clean data gives 0 to 12%. Both: 94%, and 89 to 94% under every disturbance tried [FIX: "What the demonstrations
   contain", "How each policy copes"]. Holding the jaws shut 10 frames past the end of each demonstration
   (`--tail_steps 10`) took the cut from 1/100 to 97/100 [S5].
7. **Seeds matter more than most changes; compare final checkpoints.** The same look-alike run switched to the
   colour word after 0.9M samples with one seed, 2M with another, not within 5M with a third [S2]; paired DINOv2 ends
   at 88, 81 and 63 plain over three seeds [S8-rob]. A run that has not switched looks exactly like one that cannot
   learn the task: only a second seed tells them apart. Neighbouring checkpoints move by up to 13, so the best of
   eleven is optimistic; the final checkpoint is the fair number [S8].
8. **More demonstrations at the same budget can hurt.** 16000 cut demonstrations never switched within 10M samples
   where 4000 did [S5]; 4000 pruning demonstrations do no better than 2000 at 20M (65 to 78 against 56 to 81) [S8].
9. **A frozen pretrained encoder learns faster, keeps half its success under new lighting, and is what makes paired
   looks learnable.** DINOv2 at its first checkpoint: 59, where from-scratch runs were at 9; plain 92 and 95; lighting
   46 to 55 against 0 to 19 for every from-scratch policy [S8-rob]. The from-scratch ViT never learned from paired demonstrations (2 and 16) or with a
   consistency loss between twins (0); the frozen encoders did, and only they held under a moved camera and another
   bench. Neither twice the resolution nor augmentation reached the camera or the bench (DINOv2 at 448 px: camera 9,
   bench 0; with mild augmentation: camera 26, bench 2): **the looks have to be in the demonstrations** [S8-rob].
10. **The fast policy does not take mixed data well.** A third of the demonstrations in varied looks: 5/100 [S8-rob].
    Motor drills co-trained: 22 against 79 [S8: "Aim drills"]. The first takeover dataset: 0 to 28 [S8]. Fine-tunes
    that worked kept the old data at 0.7 of every batch, a peak learning rate of 2e-4, 3M samples and the old
    normalisation: ties 118 to 177 of 200 with nothing regressed [E13b].
11. **Split the reasoning from the control, and train each part on a loss for its own job.** The stage-1 reasoner,
    trained only on the text of its notes, wrote "2 shoots" for 105 of 108 three-shoot plants; small eyes trained on
    the simulator's labels read the count in 97 to 99% of frames after 35 to 44 minutes [V2]. Fine-tuning the language
    model on notes cost it language (programs right for 71 and 60 of 100 against 100 and 92 as downloaded) [V2]; LoRA
    adapters trained on program writing alone fixed what prompting could not (700 of 700 training tasks) [V3-8].
12. **The closed loop shows errors that frame accuracy hides; perception learns from the policy's own states.** Eyes
    right in 97 to 99% of frames still left inventories that never completed and cuts reported that did not happen
    [V2]. Eyes retrained with 190,756 frames of the fast level's own rollouts cut the closed-loop misreads by half to
    three quarters (275 of 300 against 260 on validation) [V3-7]. The same idea for the actions, takeover
    demonstrations, did not help (577 against 572 of 650) [V3-7].
13. **Precision is lost to compounding error on states the demonstrations never show.** Handed the arm 6 cm from the
    cut, the fast level cuts as precisely as the expert (6.0 against 5.8 mm); from its own approach 7.4 to 8.1 mm. It
    keeps 93% of a height error it has at 10 cm, although its encoder reads that height to 1.6 mm: the expert's
    demonstrations never show a correction [CUT].
14. **One run of 200 episodes varies by about 6 either way.** v2's headline repeated: 175 to 182 and 153 to 165;
    quote the means of repeats (179 and 160), not the first run [V2: "The headline, reproduced"].

## 3. The architectures

### 3.1 The fast policy: a flow-matching action transformer on a frozen encoder

**What it is.** One transformer over a sequence of tokens: one for the flow-matching noise level, 32 for the
instruction (a learned embedding table, 49,280 x 256), the image tokens, one for the joint state, and 16 action-step
tokens (1 past, the current and 14 future steps of the 8-D action). It is trained to carry noise to the
demonstration's action chunk by flow matching; served, it integrates 10 flow steps and the arm executes the first 8 of
the 15 future actions, then asks again (0.4 s at 20 Hz) [R2: "Inference settings"]. Shapes: 6 layers, 256 wide, 4
heads, no causal mask (`vla_foundry/config_presets/models/diffusion_policy.yaml`, the configs in
`sim/pickup/configs/`).

| Part | Choice | Why (measured) |
|---|---|---|
| Image tokens | frozen DINOv2-B with registers at 224 px: 16 x 16 patch tokens per image, its class token and registers dropped (`frozen_tower: facebook/dinov2-with-registers-base`) | 92 / 95 plain, lighting half kept, learns faster; DINOv3-B/16 the same (91, lighting 43 (70)) [S8-rob]. A from-scratch ViT is as good in the training scene (87) and collapses under any visual change (0 to 4) |
| Cameras | `front` and `wrist`, 224 x 224 | every result here; the wrist image reads the jaws' height against a shoot to 1.6 mm [CUT] |
| Memory | the frame 1 s back (`image_indices: [-20, 0]`, shards made with `preprocess_1s`) for single-level tasks; none for a sub-goal fast level | lesson 4 |
| Previous action, previous frame | never | lesson 4 |
| Language | instruction tokens next to the patch tokens | without them the policy closes at chance between objects [R2] |
| Action | absolute joint targets and a gripper command, chunk of 15, 8 executed, 10 flow steps | 8 executed: 85 against 76 (2) and 84 (15); 2 to 4 flow steps did as well as 10 on `pick_target` [R2]; on the cut no setting moved precision and 30 flow steps, never trained with, broke the policy (68 of 150 damaged) [CUT] |

**Two variants in use.** `dp_dinov2_lang_frames_1s` (or `dp_dinov3_lang_frames_1s`): the single policy of section 1,
instruction in, 1 s memory. `dp_dinov2_lang_reactive_subgoal`: v3's fast level, one frame per camera, the instruction
type `subgoal` (the sub-goal written by the expert's notes, "cut the middle shoot").

**Training recipe.**

| | |
|---|---|
| Data | the task's demonstrations, random starts, tail of 10, plus 2 re-rendered copies of each (section 4.4); for pruning 2000 source demonstrations, 6000 episodes |
| Budget | 20M samples (`train_resilient.sh <config> <shards> <run> 20000000`) |
| Optimiser | AdamW, peak learning rate 5e-4, cosine to 1e-5 over the planned budget, batch 64, bf16 autocast, gradients clipped at 1.0 (`vla_foundry/config_presets/hparams/diffusion_policy.yaml`) |
| Averaged weights | EMA on (`ema: enabled: true`); the averaged weights are the ones evaluated and kept |
| Checkpoints | `PICKUP_NUM_CHECKPOINTS=10` and one at the end, each evaluated as it appears |
| Seeds | `--hparams.seed` and `--data.seed`; at least two per claim |
| Cost | DINOv3 at 224 px: 414 samples/s (at 448 px: 119) [S8-rob]; the DINOv2 sub-goal level 690 samples/s alone, 9 hours per 20M [V3-8]; the 1 s policy on paired data with its checkpoints evaluated on the same GPU: 15.9 to 18.5 hours [OPS] |
| Fine-tuning | from the averaged weights (`--model.resume_from_checkpoint <ema> --model.resume_weights_only True`), old data at 0.7 of each batch (`PICKUP_DATA_WEIGHTS`), the old normalisation (`PICKUP_STATS_DATASETS`), peak 2e-4, 3M samples: one hour [E13b] |

**Failure modes.**

- **Seed spread**: 63 to 88 plain over three DINOv2 seeds on paired data [S8-rob]; 57 episodes between two seeds of
  the sub-goal fast level [V3-8].
- **The bench** is the weakest visual change: 36 to 52, and 17 to 33 on benches never seen [S8-rob].
- **Paired data costs 7 to 14 in the training scene** [S8-rob].
- **The cut**: 7.9 to 8.8 mm from the shoot's axis against the expert's 5.9 [V3-8].
- **Over-turning the plate**: halved by DINOv2 but still 7 to 37 lost episodes per grid [V3-8].
- **A press that did nothing** goes unnoticed: the sticky button costs the frozen-encoder policies 24 to 42 points
  against their plain score [S8-rob].
- **Distractors** fall on paired data (62 and 65 on two of three DINOv2 seeds) because the paired data lacked that look
  (a placement bug, fixed; not re-run) [S8-rob].

### 3.2 Architecture v2: eyes, a scene memory, a written program

Design and diagnosis: [architecture-v2.md](architecture-v2.md). Why it exists: the stage-1 two-level policy (a
fine-tuned Qwen3-VL-2B writing a note whose sub-goal the fast level reads) solved 133 of 200 training and 66 of 200
held-out reasoning instructions, and its failures were a count it never learned to read and a language model that
fine-tuning had damaged [REASON1, V2].

```
 every look (8 control steps):  front, wrist [+ plant crop] -> EYES -> one record per part -> SCENE MEMORY
 once per episode:              instruction -> PROGRAM WRITER (frozen LM, grammar-constrained) -> typed program
 every look:                    PLAN = program evaluated on the memory -> sub-goal ("cut the middle shoot") or hold
                                front, wrist, joints + sub-goal -> FAST LEVEL -> joint targets
```

| Part | What it does | Why (measured) | Recipe and cost |
|---|---|---|---|
| Eyes (`sim/pickup/eyes.py`, `train_eyes.py`) | a 7.1M-parameter ViT (224 px, patches of 16, 6 layers, 256 wide) and five learned queries: per shoot slot exists, height, bearing, cut, visible; the flower's colour; the label | reads the three-shoot count in 97 to 98% of frames (98.5 to 99% with the plant crop) where the stage-1 reasoner wrote "2 shoots" 105 times in 108 [V2] | labels from the simulator, free and valid in every state; 8000 expert episodes recorded as frames every fourth step (534,043 frames with the crop, 21 min with 48 processes, `collect_scene.py --crop`); 2M samples at about 960 samples/s (750 with the crop): 35 to 44 min ([architecture-v2.md](architecture-v2.md), "What it costs") |
| Scene memory (`sim/pickup/scene.py`) | code: parts matched by height, a part counts when reported existing and visible at p >= 0.7, cut after three looks in a row; options `unseen`, `revoke`, `give_up` | the options answer two closed-loop errors frames did not show: 143 to 162 of 200 on validation [V2] | none (rules) |
| Program writer (`program.py`, `program_writer.py`) | Qwen3-VL-2B as downloaded, a description of the language and eight worked examples, greedy decoding constrained by a grammar acceptor walked over a trie of the vocabulary (`prefix_allowed_tokens_fn`) | the right meaning for 100 / 100 training and 92 / 100 held-out instructions as downloaded, 71 / 60 after fine-tuning [V2] | none; 0.2 to 0.3 s per episode |
| Plan (`program.Plan`, `slow_level.py`) | at every look: `Execute` (the uncut target needing the least turning), `NeedObservation`, `Complete`, `Failed` | combinations are right by construction once the program is | none |
| Reset (`eval.py --resets 2`) | sends the arm back to its start pose when the jaws are set to cut a shoot more than 45 degrees out of reach and the plate stands | the fast level's most frequent error; 169 / 148 to 180 / 165, wrong-shoot cuts from 6 to 8 per run to 0 to 1 [V2] | none |
| Fast level | `dp_vit_lang_reactive_subgoal`, seed 43, on `reason_mix_8k_rs_tail10` (8000 demonstrations, random starts, tail 10), 20M samples | 75 / 100 with the expert's sub-goals: the ceiling [REASON1] | as section 3.1 |

**Result.** 179 / 200 training and 160 / 200 held out (means of repeats) [V2]. With the simulator's scene and the
task's own program (a perfect slow level) the same fast level solves 169 and 154 without resets: **the fast level is
the limit** [V2].

**Failure modes.** The fast level's first move (going for a shoot that faces away); a fast level told its aim
fixes that and loses as many episodes turning past the target (163 / 152) [V2]; bounded recovery (retrying a step)
hurt (157 to 159 against 171) [V2]; the scene's fields are written by hand, and an instruction outside the program
language cannot be compiled ([architecture-v2.md](architecture-v2.md), "What it costs").

### 3.3 Architecture v3: attributes, a larger language, plans over worlds

Design: [architecture-v3.md](architecture-v3.md). v2's split, with the least each new task needs: tasks that bind a
word to something seen on one shoot (a tie's colour), diagnosis (sick leaves, buds), order, and knowing when there is
nothing to do.

| Part | What it does | Why (measured) | Recipe and cost |
|---|---|---|---|
| **Eyes with attribute heads** (`eyes.py`, `train_eyes.py --attributes tie sick bud`) | v2's eyes plus, per part, `tie` (cannot be read, none, red, blue, yellow, white), `sick` and `bud` (cannot be read, no, yes): one cross-entropy each; "cannot be read" is a class, not a low confidence | a tie's colour 95.4%, sick 99.0%, a bud 98.2% with the crop; counted readings wrong in 0.5 to 0.7% [V3-6] | `scene_tend` with v2's frames: 19,000 expert episodes, 1,193,168 frames, every fourth step and the tail (the new ones in 16 min with 64 processes, `collect_scene.py --attributes --crop`); then 190,756 frames of the fast level's own rollouts (`eval.py --record_scene`); trained from scratch on all frames, 3M samples, front + wrist + plant crop, seed 42: 35 to 75 min [V3-6, V3-7] |
| **The plant crop** (`scene.PlantCrop`) | a third image: the front camera's window around the plant rendered at twice the size, as a crop of a higher-resolution camera | a true-size tie (3 to 3.6 px wide at 224) is read in 83% of parts from the front image, 93.5% in the crop; with the crop the eyes solve 260 of 300 attribute episodes against 220, attribute errors in 26 episodes against 81 [V3-6] | rendered with the frame; costs the eyes about a fifth of their speed [V2] |
| **Scene memory with attributes** (`scene.py`) | an attribute is known after three readings in a row at p >= 0.7 ("row"); frozen when the part is cut; readings kept with the plate's angle | "row" 260 of 300 against "sticky" 258, "last" 241, "angles" 237 [V3-6] | none |
| **Program language** (`program.GRAMMAR_V3`) | v2's plus `CutFrom` (in order), `Then`, `Take`, `Count`, `Nothing`; attributes `tie,any`, `sick,no`, `bud,yes`; no `Stop` and no `Look`: both follow from evaluating the program | finite and checkable; equivalence decided on every plant the benchmark can draw | none |
| **Program writer with LoRA** (`writer_tune.py`, `program_writer_v3.py --lora`, `slow_server.py --writer_lora`) | the same Qwen3-VL-2B, the same prompt (16 worked examples in pairs), adapters of rank 16, alpha 32 on q, k, v, o, gate, up, down of the 28 language layers (17.4M parameters, base frozen), merged when loaded | as downloaded: 580 of 700 training, 215 of 400 held-out frames, `sick` held out 0 (polarity); with the adapters 700, 382 (392 resolved) and 100 of 100 on every situation's held-out frames; v2's own test 100 and 100 [V3-5, V3-8] | 2191 instructions from templates and synonyms of the training tasks, every held-out instruction, held-out-only word (45) and two-word sequence, and every program equivalent to a held-out composition kept out (`check_excluded`, tested); 2 epochs, lr 1e-4, batch 8, loss on the program only: 548 steps, 41 min; 0.5 s per instruction [V3-8] |
| **WorldPlan** (`program.WorldPlan`) | evaluates the program in every world the unknown facts allow: a shoot every world cuts next is cut now; nothing left in every world is `Complete`; otherwise `NeedObservation` for the fact the outcome depends on, which turns the plate to show it (`--inspect turn`, bounded by one full turn); stages in order, a cut out of order is `Failed` | never acts on a guess and never looks for a fact that cannot change what it does; with `--inspect turn` no episode ends `Failed` for an unseen fact and `nothing_to_do` goes from 188 to 200 [E13b] | none |
| **Runner** (`eval.py --slow_level v3`) | starts clear of the plant (`--clear_start`, 15 cm), the home move back to the lane in front of the plant on a verdict (`--home_move clear`), up to two resets; the cut gate (`--gate`) off | the expert from clear starts: 98 to 100 of 100 [V3-1]; the gate cost more than it saved (`order` 244 / 223 against 247 / 245; resets doubled) [E13] | none |
| **Sub-goal fast level on DINOv2** (`dp_dinov2_lang_reactive_subgoal`) | section 3.1's policy, one frame per camera, reading the sub-goal | validation 611 of 650 against 572 for the ViT fast level; over-turning losses halved [V3-8] | `reason_mix_8k_rs_tail10` and `ties_1500_rs_tail10` at 0.7 / 0.3 of each batch, the old normalisation, 20M samples from the start, peak 5e-4, seed 43 chosen on validation (seed 42: 574); 9 hours each [V3-8] |

**How it was built**, a step at a time, each step measured before the next, with predictions written beforehand and
the setting chosen on validation layouts ([architecture-v3.md](architecture-v3.md), "Predictions", "As built"):
language, truth and memory on CPU with scripted percepts [V3-1]; v3 over the old fast level on the simulator's scene
(E13), which found that a drawn tie cost the fast level 66 and 109 episodes [E13]; a fine-tune on 1500 `ties`
demonstrations (118 to 177 of 200, a shortcut probe showed no "cut the tied shoot" shortcut) [E13b, V3-6]; the eyes
with attributes [V3-6]; the writer [V3-5]; retraining the eyes on rollouts [V3-7]; the LoRA writer and the DINOv2 fast
level [V3-8].

**Failure modes.**

- **The held-out pair frame** ("remove the X branch before the Y branch") is written as one cut half the time: 62 to 66
  of 100 [V3-8].
- **Held-out compositions**: 64 of 100 programs right (the design asked 75); rung 5 is not yet run in closed loop
  [V3-8].
- **The cut** is where it was: median 8.4 mm over 3098 cuts [V3-8].
- **`sick`** is the one situation where both DINOv2 seeds are below the ViT fast level (169 and 156 against 172) [V3-8].
- **A tie read with 0.7 to 0.9 confidence is right in 68%**: a higher `p_min` for the tie is untried [V3-6].
- **The eyes' errors come in runs** (a median of 13 to 20 looks), which three readings in a row do not filter; rollout
  frames are the cure that worked [V3-6, V3-7].

### 3.4 Superseded: notes written by a fine-tuned VLM (stage 1)

A Qwen3-VL-2B reasoner (600k samples, about 3 GPU-hours) wrote a five-field note per decision; the fast level read
its sub-goal. 133 / 66 of 200, 153 / 78 after 900k more samples; swapping the sub-goal for another shoot's made the
fast level cut that shoot (the text drives the action). Replaced by v2 because it could not count and its held-out
wording stayed weak [REASON1]. Kept here for the swap test, which is the check that any text a policy reads is used.

## 4. The pipeline, stage by stage

Each stage: the commands (run from the repository root; GPU with `PICKUP_GPU`, see `sim/pickup/scripts/_common.sh`),
the cost measured, and the rules we decide by. `sim/pickup/scripts/pipeline.sh` chains stages 4.3 to 4.8 and 4.10 for
the single policy.

### 4.1 Task and scenarios

A task is an entry of `TASKS` in `sim/pickup/env.py` (objects, plant, episode length, success rule); a situation on
top of it is a scenario in `sim/pickup/scenarios.py` (`--scenario lighting`, `camera/test`, `ties`, ...).

Rules:

- **Read the success rule before building anything on it** ([architecture-v3.md](architecture-v3.md), "What the search
  for prior work changed", 8). A `Failed` verdict on a plant with nothing meant had been scored as solved; it is not any
  more [E13b].
- **Fixed seed ranges**: demonstrations from 0 (or a range of their own: 300,000 for `ties`, 400,000 for DART),
  validation layouts from 900,000, evaluation layouts from 1,000,000. Settings are chosen on validation; the chosen
  setting is run once on evaluation [E13, V2].
- **Every visual situation has a `/test` half that is never used in training** (other lights, a camera moved 3 to 4 cm,
  benches and distractors never seen) [S8-rob]. A robustness claim needs the `/test` column.

### 4.2 The scripted expert

`sim/pickup/expert.py`. Check it alone (`python -m sim.pickup.smoke_test <task> 20`) and through the evaluation loop
(`eval.py --expert`), on the same layouts and situations the policy will meet.

Rules: the expert must solve the task in about 90% or more through the evaluation loop before a demonstration is
collected (it was 88 to 100 on every rung; at 88 on the stem grasp the policies failed) [S4]; its precision is the
floor (lesson 1); it plans again when the world changes, or the recoveries in its demonstrations are worthless [FIX];
its line of approach is what the policy copies, including never correcting it [CUT].

### 4.3 Demonstration collection

```bash
PICKUP_GPU=2 PICKUP_WORKERS=48 sim/pickup/scripts/make_dataset.sh prune 2000 prune_2k_rs_tail10_leaffix \
    --random_start --tail_steps 10 --leaf_fix
```

`make_dataset.sh` runs `python -m sim.pickup.collect` with that many simulator processes and preprocesses the result;
use `collect` alone when the collection is to be re-rendered first (as `pipeline.sh` does). The options that
mattered:

| Option | What | Measured |
|---|---|---|
| `--random_start` | every episode starts from a random gripper pose | 98% and robust to the start pose [FIX] |
| `--disturb_prob 0.5` | the object is moved once in half the episodes, the expert plans again | self-correction [FIX] |
| `--tail_steps 10` | 10 more frames of the robot holding its last command after each success | the cut: 1 to 97 / 100 [S5] |
| `--scenario <name>` | collect under a situation (`ties`, `sick`, ...) | the `ties` fine-tune [E13b] |
| `--park_reset`, `--leaf_fix`, `--seed` | simulator state and seeds; record them, `rerender.py` reads them back from the collection's metadata | |
| `--action_noise 0.02 --action_noise_quiet_cm 3` | DART-style noise on the executed joints, the clean commands recorded, quiet near the cut | helped the cut a little, not adopted [CUT] |

Size: 4000 for `pick_target` at 5M samples, 2000 for `prune` at 20M (4000 did no better) [FIX, S8]; 8000 for the
reasoning mix [REASON1]; 1500 for a fine-tune on one situation [E13b].

Cost: 1500 `pick_target` demonstrations in 70 s with 48 processes [OPS]; 1500 `ties` demonstrations collected and
preprocessed in about 20 minutes with 48 processes (311,670 frames, 51 GB of shards) [E13b]; 3000 DART `reason_mix`
demonstrations in 11 minutes [CUT].

### 4.4 Paired re-rendering

```bash
python -m sim.pickup.rerender --src data/lerobot/prune_2k_rs_tail10_leaffix \
    --out data/lerobot/prune_2k_rs_tail10_leaffix_paired --copies 2 --include_source --collect_workers 48 --workers 48
```

Every episode is replayed in the simulator, checked bit for bit against the recorded states at every frame, and drawn
again in another look (`rerender.LOOKS`: the training halves of lighting, camera, bench and distractors), one look per
copy for the whole episode; states and actions are unchanged, so a look can only teach what does not matter.
`--collect_workers` must equal the number of workers the collection was made with, because the replay re-runs the
collection as it ran.

Cost: 6000 episodes (2000 source, 4000 copies) and their 1 s shards in 2 h 32 min, 172 GB; later re-renders drew 4000
copies in 28 minutes [OPS].

Rules: re-render only for a frozen pretrained encoder (from scratch it stopped learning: 2 and 16) [S8-rob]; check
that every look you train on was actually drawn (the distractors look was not, in the first paired set) [S8-rob].

### 4.5 Preprocessing into shards

```bash
PICKUP_WORKERS=32 sim/pickup/scripts/preprocess.sh prune_2k_rs_tail10_leaffix_paired \
    prune_2k_rs_tail10_leaffix_paired_1s preprocess_1s
rm -rf data/preprocessed/prune_2k_rs_tail10_leaffix_paired_1s/{frames,episodes}
```

Windows of 1 past, the current and 14 future low-dimensional steps; images at t-1 and t (`preprocess.yaml`) or at
t-20 and t (`preprocess_1s.yaml`, for the 1 s policies); JPEG inline, no resize; `stats.json` holds the
normalisation. Training reads only `shards/`; `frames/` and `episodes/` are intermediates. Write them on a scratch
disk (`PICKUP_ROOT=<scratch root>`, then link the shards into `data/preprocessed/`) and delete them once the shards are
written. An existing collection can be preprocessed again for another window: 5 minutes for 2000 pruning
demonstrations, 57 GB [OPS].

Rules: preprocessing and the data path are where train/serve mismatches hide (four bugs found there, among them a
pixel range that made the VLM ignore its images and a position table in half precision: 83 against 93 for the same run)
[OPS: Bugs]; when fine-tuning on a new mixture keep the
old statistics (`PICKUP_STATS_DATASETS`) [E13b].

### 4.6 Choosing the policy

| Task | Config | Data window |
|---|---|---|
| single-level, visual robustness wanted | `dp_dinov2_lang_frames_1s` or `dp_dinov3_lang_frames_1s` | `preprocess_1s` |
| fast level of v2 / v3 | `dp_dinov2_lang_reactive_subgoal` | `preprocess` |
| quick ablation in the training scene only | `dp_vit_lang_reactive` (from scratch, 1100 samples/s) [OPS] | `preprocess` |

### 4.7 Training

```bash
PICKUP_GPU=2 PICKUP_NUM_CHECKPOINTS=10 PICKUP_ABOUT="what this run is for" \
    sim/pickup/scripts/train_resilient.sh dp_dinov2_lang_frames_1s prune_2k_rs_tail10_leaffix_paired_1s \
    dp_dinov2_lang_frames_1s_prune2k_leaffix_paired_20m 20000000 --hparams.seed 42
```

`train_resilient.sh` restarts the run from its newest complete checkpoint if it dies or its log stalls for 7 minutes
(a dead data-loader worker otherwise hangs it at 0% GPU), sets aside a checkpoint cut short, and holds a pid file so
that a second trainer cannot write into the same run [OPS: Bugs; roadmap fixes 6 and 20]. Mixtures: datasets joined
by `+`, weights in `PICKUP_DATA_WEIGHTS`.

Rules:

- **Plan the whole budget up front** (the cosine follows it; lesson 5): 20M for pruning-scale tasks.
- **Fine-tune, don't retrain, to add a situation**: old data at 0.7, peak 2e-4, 3M samples, old normalisation, judged
  first on the old tasks, where it must stay within 15 of 200 ([architecture-v3.md](architecture-v3.md), P9) [E13b].
- **Use the averaged weights** (EMA) as the number; they lead the trained weights while a policy starts to work [R2].
  Evaluate both while training: once in five pruning runs the averaged weights ended well below the trained ones (65
  against 78) [S8].

### 4.8 Evaluation

Every checkpoint while it trains, on another GPU:

```bash
PICKUP_GPU=3 PICKUP_WATCH=1 PICKUP_EVAL_JOBS=4 PICKUP_EVAL_ARGS=--leaf_fix \
    sim/pickup/scripts/eval_checkpoints.sh experiments/pickup/<run> prune 100
python -m sim.pickup.summarize --task prune --runs <run>
```

Then the **report card** on the final averaged weights: `plain`, `lighting`, `camera`, `bench`, their `/test` halves,
`distractors`, `distractors/test`, `wind`, `bump`, `sticky`, 100 episodes each:

```bash
PICKUP_CHECKPOINT=experiments/pickup/<run>/checkpoints/checkpoint_11.pt PICKUP_WEIGHTS=ema \
    PICKUP_OUT=results/pickup/<run>/prune_scenario_camera_test/ckpt_11_ema \
    sim/pickup/scripts/eval.sh experiments/pickup/<run> prune 100 --scenario camera/test --leaf_fix
```

(`pipeline.sh`'s `card` stage runs all twelve, six at a time on two GPUs.) Cost: 41 to 52 minutes per card [OPS].
Each results.json has the successes with a Wilson interval, the failures split into damage, wrong target and time-out,
per-target counts and latency; every episode record says why it ended. For v2 and v3: `eval_v2.sh` and `e13.sh` (the grid of situations, validation then evaluation).

**Cut precision**, in millimetres from the shoot's axis at the step that cut it: every v3 run records `cut_places`;
the diagnosis is `cut_diag.py`:

```bash
python -m sim.pickup.cut_diag --mode handover --handover_cm 100 --checkpoint_dir experiments/pickup/<run> \
    --episodes 150 --out cut_ho100.json      # handed over as soon as the shoot is chosen
python -m sim.pickup.cut_diag --mode handover --handover_cm 10 --shift_mm 6 --checkpoint_dir experiments/pickup/<run> \
    --episodes 150 --out cut_shift6.json     # how much of a 6 mm height error survives to the cut
```

Rules (the ones we decide by):

1. **Closed loop only**, on layouts never seen; Wilson 95% intervals [lab notebook, preamble].
2. **Noise**: with 100 episodes a few points between single runs are noise; with 200, about 6 either way, and fewer
   than about 15 is no difference [V2, V3-1].
3. **Final checkpoint, averaged weights** is the number; the best of eleven is optimistic [S8].
4. **Two seeds** before concluding that a change works or fails; three where the spread is the question [S2, S8-rob].
5. **Keep a change only if it clearly improves what it was for and loses nothing elsewhere beyond the noise.** The
   DART fine-tune moved the mean cut 0.4 to 0.6 mm closer and lost 26 of 2600 episodes: not adopted [CUT]. The cut
   gate stays off unless the fast level cuts an unnamed shoot in more than 2 of 200 episodes [E13]. A look that costs
   the fast level more than 30 of 200 means a fine-tune; up to 15, none (P8) [E13, E13b]. All of a new slow level
   within 15 of the simulator's scene, or count its errors first (P5) [V3-6].
6. **Count the causes before adding anything**: every episode record says why it ended (out of time, the flower
   touched, a wrong shoot, a look that ran out) [E13].
7. **Serve as trained**: 8 actions of 15, 10 flow steps; no inference setting moved the cut [CUT].
8. **Quote means of repeats** for headlines [V2].

### 4.9 The reasoning stack on top

```bash
# perception data (the expert drives; every fourth step, the plant crop, the truth with attributes)
python -m sim.pickup.collect_scene --task prune --scenario ties --attributes --crop --episodes 2000 \
    --workers 64 --out data/scene/scene_tend_ties
# the policy's own states, recorded during closed-loop evaluation
PICKUP_SLOW_LEVEL=v3 PICKUP_SCENE=eyes PICKUP_EYES=experiments/pickup/<eyes> \
    sim/pickup/scripts/eval_v2.sh experiments/pickup/<fast run> prune 700 --scenario ties --seed_start 700000 \
    --clear_start --inspect turn --record_scene data/scene/scene_policy_ties
# the eyes, from scratch on everything
python -m sim.pickup.train_eyes --data <scene dirs...> --val <validation dirs...> --out experiments/pickup/<eyes> \
    --samples 3000000 --seed 42 --cameras front wrist plant --attributes tie sick bud
# the writer's adapters, and the writer alone on the held-out split
python -m sim.pickup.writer_tune data --out paraphrases.json
python -m sim.pickup.writer_tune train --model <Qwen3-VL-2B> --data paraphrases.json --out <lora dir>
python -m sim.pickup.program_writer_v3 --model <Qwen3-VL-2B> --lora <lora dir> --out writer_lora.json
# all of v3 on the grid of situations (validation, then evaluation)
PICKUP_EYES=experiments/pickup/<eyes> PICKUP_E13_PROGRAM=writer PICKUP_WRITER=<Qwen3-VL-2B> \
    PICKUP_WRITER_LORA=<lora dir> PICKUP_WRITER_ARGS="--language v3 --shots 16 --examples A" PICKUP_E13_HELDOUT=1 \
    PICKUP_E13_ARGS="--inspect turn" sim/pickup/scripts/e13.sh experiments/pickup/<fast run> <second fast run> all
```

Order: perception data, eyes, writer alone, then closed loop on the simulator's scene with the task's program (the
fast level's ceiling), then the eyes, then the written program, so that each part's cost is the difference between
two runs (section 3.3). Cost of a v3 step: 4.7 to 23.2 GPU-hours, most of it closed-loop evaluation [V3-6, V3-7, V3-8].

### 4.10 Cleanup

At the end of every run: keep `checkpoint_<final>.pt` and `ema_<final>.pt`, delete the other checkpoints, the
averaged weights of other checkpoints and every optimiser state, once the report card is written (section 7). At the
end of every phase: a full cleanup (section 7).

## 5. What did not work

Each line is something we tried, with its number, so that nobody repeats it without a new reason.

| Tried | Result | Source |
|---|---|---|
| Pooled CLIP vector as the image input | 4% on `pick_cube`; 0/20 on its own training layouts | [`pick_cube`] |
| One conditioning vector for a VLA (Qwen, from-scratch) | 0 to 6% on `pick_target` | [R2], first round |
| The previous action as input | 0 to 4% (`pick_target`), 0 at checkpoints 1 to 3 (pruning) | [R2], [S8] |
| A hold frame at the start of demonstrations to fix it | 0% / 4% after 5M | [FIX] |
| The previous frame as memory | 53 / 35 against 81 / 79 | [S8] |
| Every object of a layout named in turn | no gain (89 to 93 against 93) | [FIX] |
| Early windows repeated, with random starts and recoveries | 87 / 86 against 94 | [FIX] |
| 16000 demonstrations at 10M samples (cut) | never switched to the colour word (31) | [S5] |
| 4000 instead of 2000 pruning demonstrations at 20M | 65 to 78 against 56 to 81 | [S8] |
| 40M samples instead of 20M | 80 against 79 | [S8] |
| Takeover (DAgger-style) demonstrations, first dataset | 0 to 28 against 56 to 81 | [S8] |
| Takeover demonstrations, corrected | 83, 24, 68 against 79, 56, 78 | [S8] |
| Takeover fine-tune of v3's fast level | 577 against 572 of 650; cut 9.0 against 8.3 mm | [V3-7] |
| Motor drills co-trained with pruning / as pretraining | 22 against 79 / 74 against 79 | [S8] |
| A third of the demonstrations in varied looks, from-scratch ViT | 5 plain, 2 to 7 elsewhere | [S8-rob] |
| Paired re-rendering with a from-scratch ViT | 2 and 16 | [S8-rob] |
| Consistency loss between a frame and its re-rendered twin | 0 | [S8-rob] |
| Image augmentation, from-scratch ViT | 7 | [S8-rob] |
| Mild augmentation on frozen DINOv2 (plain data) | camera 26 (13), bench 2 (1) | [S8-rob] |
| Twice the resolution (DINOv2 or DINOv3 at 448 px, plain data) | camera 7 to 9, bench 0; DINOv3 83 plain against 91 | [S8-rob] |
| DART noise 0.01 rad everywhere, from-scratch ViT | 40 plain, no robustness | [S8-rob] |
| DART demonstrations quiet near the cut, 3M fine-tune | cut median 8.0 against 8.4 mm, 1990 against 2016 of 2600: not adopted | [CUT] |
| Inference settings near the cut (2 or 4 actions, averaged chunks) | 7.7 to 8.4 mm, within noise; 30 flow steps break the policy | [CUT] |
| Frozen Qwen3-VL tower instead of DINOv2 on paired data | lighting 60, camera 51, bench 37 against 80, 71, 44 | [S8-rob] |
| Qwen3-VL-2B VLA at the budget it can finish in a day (2.5M) | 21 to 22, level with the small policy at that budget | [S9] |
| Stage-1 reasoner (a fine-tuned VLM writing notes) | 133 / 66 against v2's 179 / 160; at 448 px 141 / 100 | [REASON1], [V2] |
| One model writing the note and acting | 14 / 13 | [REASON1] |
| Fine-tuning the language model that writes programs | 71 / 60 against 100 / 92 as downloaded | [V2] |
| The fast level told its aim | 163 / 152 against 169 / 148 | [V2] |
| Bounded recovery (retry a step with no effect) | 157 to 159 against 171 | [V2] |
| The cut gate | `order` 244 / 223 against 247 / 245; resets doubled | [E13] |
| Writer thinking first | 566 / 181 against 580 / 215, 20 to 30 times slower | [V3-8] |
| Asking the writer twice | blocks every `prune` episode | [V3-5] |
| Memory rules "sticky", "last", "angles" | 258, 241, 237 against 260 of 300 | [V3-6] |
| Temperature calibration of the eyes | 257 and 265 against 260 and 275 of 300 | [V3-7] |
| v2's eyes retrained on the fast level's frames | 589 and 621 against 604 and 628 of 800 (v3's rollout eyes, by contrast, helped) | [V2] |

## 6. Open problems, with current numbers

- **Cut precision.** Every fast level cuts 7.9 to 8.8 mm from the shoot's axis (8.4 over 3098 cuts of v3 whole); the
  expert 5.8 to 5.9. Diagnosed as compounding error on the approach (lesson 13); DART demonstrations at 35% of a 3M
  fine-tune correct a quarter of a height error. Next: DART as most of the data in a 20M run from the start; such
  runs are trained and not yet in the notebook [CUT].
- **Bench robustness.** The weakest visual change for the best policies: 36 to 52 on benches seen in training, 17 to
  33 on benches never seen [S8-rob]. Runs with more camera and bench copies and with the distractors look drawn are
  under way, not yet in the notebook.
- **Seed spread.** 63 to 88 plain for three DINOv2 seeds on the same paired data; 57 episodes between two seeds of the
  sub-goal fast level [S8-rob, V3-8]. Budget for three seeds of anything that will be adopted.
- **The writer's remaining held-out errors.** The held-out pair frame, 62 to 66 of 100 ("before" never seen in
  training); held-out compositions 64 of 100 against the design's 75 [V3-8].
- **Reach in height (z) and 3D.** Every task so far puts its targets in a narrow band of heights (stems 7 to 11 cm
  above the table, shoots on one potted plant on one stand) [S4, S6], so how the recipe carries to targets spread in
  height is untested. Scenarios for it are being built; nothing is measured yet.
- **v3's robustness to visual change** is unmeasured: its fast level and eyes were trained on plain looks [V3-8].
- **Rungs 5 and 6 of v3** (compositions in closed loop, the hidden tag) are not run ([architecture-v3.md](architecture-v3.md)).

## 7. Disk and compute hygiene, as practised

- **Scratch for intermediates, shards only in the data tree.** Collections, re-rendered collections and the
  preprocessing's `frames/` and `episodes/` go on a scratch disk; the shards are linked into `data/preprocessed/`; the
  intermediates are deleted as soon as the shards are written. Shards are large: 57 GB for 2000 pruning
  demonstrations, 172 GB paired, 106 GB for the 8000-episode reasoning mix [OPS].
- **Keep only final weights.** A cleanup watcher waits for each run's "report card done" line, then keeps the final
  checkpoint and its averaged weights (253 MB) and deletes every other checkpoint and optimiser state (about 340 MB per
  checkpoint) [OPS]. `pipeline.sh`'s `cleanup` stage does the same for its run.
- **At the end of every phase, a full cleanup**: delete superseded checkpoints, datasets and intermediates; keep only
  the adopted weights and the baselines the lab notebook cites. A dataset is superseded when no adopted run and no
  cited baseline trains on it; a source collection that later re-renders start from is kept until no planned run
  needs it (`rerender.py` makes the copies again from it, bit for bit).
- **GPU claims.** A GPU is taken only when it is unclaimed and has no compute process at two checks two minutes apart;
  the claim is a directory made with `mkdir` (atomic), holding one line on what it is for, removed on release. GPUs
  reserved for others are never defaulted onto. One process tree per GPU index (`PICKUP_GPU`, PCI bus order, so the
  index means what `nvidia-smi` shows; rendering follows the first visible device) [OPS].
- **Thread caps.** `OMP_NUM_THREADS=4` per process: without it a dozen policy servers pushed machine B to a load
  of 570 and halved training throughput (1069 to 410 samples/s) [OPS].
- **Launching that survives a dropped connection.** Every long job is started with `setsid nohup ... < /dev/null >
  <log> 2>&1 &`, so it is in its own session and does not die with the SSH session or the VPN; a queue script waits
  for its inputs (a `DONE` line in the datasets log, a `trained` line in the launches log) rather than for a terminal.
  `train_resilient.sh` resumes after a crash or a reboot from the newest complete checkpoint.
- **One evaluator per port.** Evaluations pick a free port and refuse a server that serves another checkpoint (about 40
  evaluations were once filed under the wrong run) [OPS].
- **Every run says what it is for** (`PICKUP_ABOUT`, written next to the run) and every launch, finish, card and
  cleanup is one line in a launches log with the git commit it ran from.

## 8. Arm-specific and portable, for a later port

| Must change for a new arm | Where it lives here |
|---|---|
| Action space and limits: 8-D absolute joint targets (7 joints, gripper 1 open / 0 closed) at 20 Hz | `sim/pickup/env.py`; the configs' action fields; normalisation is recomputed per dataset (`stats.json`) |
| Kinematics and IK: random starts, the expert's moves, the home pose, the clear-start rule, the gate's and the reset's geometry | `sim/pickup/expert.py` (`random_start`), `slow_level.py`, `eval.py` |
| The scripted expert, and its precision (the floor) | `sim/pickup/expert.py` |
| Camera mounts and calibration: front and wrist at 224 px, the plant crop cut from the front camera; the camera looks' offsets | `env.py`, `scene.PlantCrop`, `scenarios.py` |
| The MuJoCo model and contact settings (stiff contacts, `o_solref` 0.004 s) | `sim/pickup/assets/`, `env.py` (lab notebook, "Toward flower trimming", "Physics change") |
| Gripper and cut mechanics (jaws shut on a shoot 2 to 5 cm out for 3 control steps), grasp tolerance | `env.py` [S6]; measure tolerance again ([lab notebook](lab-notebook.md#grasp-tolerance)) |

**Carries over**: the data recipe (random starts, recoveries, tail of 10, the 1 s frame, 2000 to 4000 demonstrations,
20M samples); paired re-rendering (needs a simulator that replays bit for bit); the frozen encoder and the fast
policy's training recipe; the report card, seed ranges and decision rules; the reasoning layer (memory, program
language, writer and its LoRA recipe, WorldPlan, runner); the eyes' training scheme (labels from the simulator, then
the policy's own rollouts); the hygiene above.

**Porting checklist.** (1) Model the arm, its gripper and cameras in MuJoCo and audit every contact. (2) Write the
expert and its IK; get it to 90% or more through `eval.py --expert`, and measure its precision and the task's
tolerance. (3) Set the action fields and limits in the env and configs. (4) Check that a collected episode replays bit
for bit (`rerender.py` refuses otherwise). (5) Collect 2000 demonstrations with random starts and a tail, re-render,
train the frozen-encoder policy for 20M, two seeds, report card. (6) For the reasoning stack, redefine the part
records for the new scene, collect perception frames, train the eyes, then the closed loop in the order of section
4.9. (7) Before a real camera: the gap between rendered and real images is not measured here.

## 9. One command: `pipeline.sh`

```bash
PICKUP_GPU=2 PICKUP_EVAL_GPU=3 PICKUP_EVAL_ARGS=--leaf_fix PICKUP_WORKERS=32 PICKUP_SCRATCH=<scratch dir> \
    sim/pickup/scripts/pipeline.sh prune 2000 prune_2k_rs_tail10_leaffix --random_start --tail_steps 10 --leaf_fix
```

Stages `collect rerender preprocess train card cleanup` (choose with `PICKUP_STAGES`; a stage whose output exists is
skipped, so the same command resumes); config, budget, copies, seed and card through environment variables listed at
the top of the script; `PICKUP_DRY_RUN=1` prints the commands. It calls the existing scripts (`collect`, `rerender`,
`preprocess.sh`, `train_resilient.sh`, `eval_checkpoints.sh`, `eval.sh`) and adds nothing to them. Tested with
`PICKUP_DRY_RUN=1` and, for the card's bookkeeping and the cleanup, on a run made of empty files; it has not yet made
a policy end to end.
