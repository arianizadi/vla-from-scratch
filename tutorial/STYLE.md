# How the chapters are written

The tutorial teaches the models behind robot manipulation from the ground up: language models,
vision-language models, vision-language-action policies, and how to train and debug them. This
file is the contract every chapter follows.

## Reader

Someone who programs in Python and has trained a small neural network, and who has not built an
LLM, a VLM or a robot policy. They read on GitHub or on the site, with the repository open next to
them. They want to be able to do it themselves afterwards, not to be impressed.

## Voice

- Plain, direct, concrete. Say what a thing is, then why it exists, then how it works.
- Define every term at its first use, in the sentence that uses it.
- Prefer one worked example with real numbers over a general statement.
- No hype, no marketing words, no rhetorical questions, no "in this chapter we will", no
  "let's dive in". Do not address the reader as a beginner.
- Short paragraphs. A table when comparing things across the same attributes. A numbered list for
  steps. Prose for reasoning.
- State uncertainty where it exists and only there.

## Three kinds of statement, kept apart

1. **Measured here.** A number from this repository. It must appear in `tutorial/lab-notebook.md`;
   copy it exactly and say what it was measured on. Never invent or round into a nicer number.
   Never describe the result of an experiment the lab notebook lists as in progress.
2. **In this repository.** A statement about the code. Name the file and the symbol
   (`vla_foundry/models/transformer.py`, `CustomAttn.forward`) and make sure it is true of the code
   as it is now. Read the file; do not describe what such code usually does.
3. **Background.** Established knowledge and published work that was not measured here. Say so
   ("Background:", or attribute it to the paper). Cite papers by first author, short title and year,
   with a link to arXiv or the publisher. Only cite a paper whose details you have checked.

## Structure of a chapter

```markdown
---
title: <Title Case, at most 60 characters>
part: <number> <part name>
order: <the number in the file name>
summary: <one sentence, what the reader can do after reading>
status: draft
---

# <title>

<two or three sentences: what this is and why a robot policy needs it>

## The idea
<the concept, from zero, with one worked example>

## How it works
<the mechanism, step by step, with shapes and numbers; diagrams where they help>

## In this repository
<where it lives in the code, how to run it, which config keys control it>

## What we measured            (omit if nothing in the lab notebook applies)
<numbers from the lab notebook and what they do and do not show>

## Where it goes wrong
<edge cases and failure signs: what you see, why it happens, how to check, what to do>

## Background and further reading
<the papers and systems behind this, each with one sentence on what it contributed>

## Check yourself
<three to five questions or small exercises with the commands to run>
```

Sections may be renamed when a better name exists, but every chapter keeps the repository section,
the failure section and the reading list. Length: 1,500 to 3,000 words.

## Formatting

- GitHub-flavoured Markdown that reads well on github.com without the site.
- Diagrams: Mermaid in fenced `mermaid` blocks, or a table, or a short ASCII sketch in a code block.
- Math: `$...$` inline and `$$...$$` on its own lines, only where it helps.
- Code: fenced blocks with a language tag. Commands must be runnable from the repository root.
- Shapes are written as `[batch, tokens, width]`.
- Link other chapters by relative path (`../chapters/41-flow-matching.md` is wrong;
  `41-flow-matching.md` is right, they share a folder). Link code by repository-relative path from
  the chapter: `../../sim/pickup/env.py`.
- Interactive figures. Where a figure from the list below would help, put this on its own line,
  followed by a one-paragraph description that stands in for the figure on GitHub:

  ```markdown
  <!-- interactive: flow-matching -->
  *Figure: points of noise carried to two targets by the learned flow; with few steps or a weak
  model the samples land between the targets.*
  ```

  Use only ids from the list. At most three per chapter.

## Interactive figures

| id | Shows |
|---|---|
| `pipeline` | the whole loop: expert, dataset, shards, training, policy server, simulator, success rate |
| `tokenizer` | a sentence split into tokens and ids with the tokenizer used here; a toy BPE merge, step by step |
| `padding-shift` | a padded batch and how padding moves a token appended at the end |
| `attention` | an attention matrix over a short sequence, causal mask on and off |
| `rotary` | rotary position encoding: queries and keys rotated by position, relative offset preserved |
| `training-step` | one optimisation step: forward, loss, backward, update |
| `loss-vs-success` | a real loss curve next to the closed-loop success of the same checkpoints |
| `patchify` | a camera frame cut into patches, each patch becoming a token; object size in pixels |
| `pooled-vs-tokens` | many patch tokens against one pooled vector reaching the action head |
| `sequence-layout` | the token sequences: VLM prompt with image placeholders; policy sequence of time, conditioning, state and action tokens |
| `flow-matching` | noise carried to actions by a learned flow; step-count slider; two targets and mode averaging |
| `action-chunks` | predict 15 actions, execute 8, re-plan: a timeline |
| `copycat` | how much of the next action can be predicted from the previous action alone |
| `grasp-map` | top-down map of where the gripper closed relative to the object, per policy |
| `tolerance` | grasp success against placement error, per object |
| `arm-replay` | a 3D replay of a rollout: gripper path, objects, the moment of closing |
| `timeline` | the papers and systems that led to VLAs, on a time axis |

## Files

Chapters live in `tutorial/chapters/` as `<order>-<slug>.md`. Images in `tutorial/img/`. The index
is `tutorial/README.md`. Measured facts live in `tutorial/lab-notebook.md`.
