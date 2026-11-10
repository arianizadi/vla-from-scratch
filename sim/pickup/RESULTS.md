# Pickup sim: first results (2026-11-10)

All numbers are closed-loop success rates on layouts the policy never saw (evaluation seeds
`1_000_000+`), single RTX 4090, one training seed per row. Intervals are Wilson 95%.
Defaults: execute 8 actions per predicted chunk, 10 flow steps.

> **Hardware caveat.** The machine these runs used was unreliable. One training run died and was resumed
> from its checkpoint. Treat these numbers as first measurements to be re-run once the memory is
> fixed, not as final benchmarks.

## pick_cube ("pick up the red cube")

| Policy | Demos | Training samples (epochs) | Final loss | Success | 95% CI |
|---|---|---|---|---|---|
| DP + CLIP ViT-B/32 (pooled embedding), 245M | 200 | 300k (23) | 0.040 | 2/50 = 4% | 1-14% |
| DP + small ViT from scratch (patch tokens), 36M | 200 | 300k (23) | 0.091 | 15/50 = 30% | 19-44% |
| same | 1000 | 500k (8) | 0.084 | 20/50 = 40% | 28-54% |
| same | 1000 | 1.5M (23) | 0.058 | 37/50 = 74% | 60-84% |

The scripted expert succeeded on 49 of 50 spot-checked layouts (its failures are discarded from the data).

Inference settings on the last row's checkpoint (50 episodes each):

| Setting | Success | 95% CI |
|---|---|---|
| default (8 actions per chunk, 10 flow steps) | 74% | 60-84% |
| 4 actions per chunk | 84% | 71-92% |
| 15 actions per chunk | 78% | 65-87% |
| 4 flow steps | 86% | 74-93% |
| 25 flow steps | 84% | 71-92% |

These intervals all overlap: at 50 episodes none of the settings is distinguishable from the
default. Sampling is stochastic, so the spread (74-86%) is also a fair picture of run-to-run noise.

Inference latency per chunk was 20-65 ms for the 36M policy and ~190 ms for the Qwen VLA
(measured while other jobs were running and under changing CPU frequency caps, so indicative only).

## pick_target (three objects on the table, "pick up the \<object\>")

| Policy | Demos | Training samples (epochs) | Final loss | Success | 95% CI |
|---|---|---|---|---|---|
| DP + small ViT from scratch, no language input, 36M | 1500 | 1.0M (10) | 0.069 | 1/100 = 1% | 0-5% |
| Qwen3-VL-2B VLA (last 4 LLM layers + action head trained), 2.2B / 306M trainable | 1500 | 100k (1) | 0.087 | 0/50 = 0% | 0-7% |
| same run | 1500 | 200k (2) | 0.066 | 1/50 = 2% | 0-10% |
| same run | 1500 | 300k (3) | 0.039 | 6/50 = 12% | 6-24% |
| same run, final | 1500 | 400k (4) | 0.03-0.04 | 6/100 = 6% | 3-12% |

## What the runs show

- **The pipeline is sound.** On recorded samples the served policy reproduces expert action chunks
  to within 0.01-0.04 rad per joint (copying the previous action is off by 0.05-0.2 rad), and the
  same recipe goes from 30% to 74% with more data and training.
- **Pooled image embeddings are not enough for grasping.** The CLIP baseline reaches the lowest
  training loss and the lowest open-loop error, yet succeeds 4% of the time (0/20 even on its own
  training layouts): it executes a plausible reach-descend-close-lift a few centimetres away from
  the cube. Conditioning on per-patch tokens fixes that.
- **Training budget dominates.** Failures of the better policies are near misses: the gripper
  arrives at the right object slightly offset or mis-rotated and a finger lands on top of it.
  Success rose 40% -> 74% between 8 and 23 epochs on the same 1000 demos.
- **pick_target is not solved by either policy yet.** Rollouts show both moving to the named
  object's neighbourhood and then missing the grasp. The ViT baseline has no language input and
  had 10 epochs (1%). The Qwen VLA went 0% -> 2% -> 12% -> 6% over 4 epochs while its loss fell
  from 0.087 to ~0.035; the last two checkpoints are within noise of each other. In the 150
  episodes run on those two checkpoints it never lifted an object other than the named one.
  Qwen ahead of the language-blind baseline (6/100 vs 1/100) is suggestive, not conclusive, and
  the two had very different training budgets.
- **Open-loop error does not predict success.** The 1% pick_target ViT and the 74% pick_cube ViT
  have nearly identical open-loop errors. Use closed-loop evaluation to compare policies.

## Notes for the next runs

- Give pick_target the budget that worked for pick_cube (20+ epochs). For the Qwen VLA that is
  ~11 h per 2M samples at 49 samples/s on this GPU.
- The Qwen run trains only the top 4 language-model layers (`num_trainable_vlm_layers: 4`) because
  full fine-tuning does not fit in 24 GB. The vision tower is frozen and sees 49 tokens per 224 px
  image; a higher token budget (`processor_kwargs.size`) is the first thing to try if precision
  stays the bottleneck.
- EMA weights (`--ema.enabled True`) and image augmentation are untested here and are standard for
  diffusion policies.
- Not yet done: the from-scratch LLM -> VLM -> VLA path from `tutorials/training_llm_vlm_vla.ipynb`.
