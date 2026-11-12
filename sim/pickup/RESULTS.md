# Pickup sim: results

The results live in two files:

- the [lab notebook](../../tutorial/lab-notebook.md): every number measured with this environment,
  by run and by checkpoint;
- the [field notes](../../tutorial/notes.md): what the numbers mean, what worked, what did not, and
  what was got wrong on the way.

This file used to hold the first round of runs (2026-11-10). Its tables are now the notebook's
sections "`pick_cube` (machine A)", "`pick_target`, first round (machine A)" and "The from-scratch
stages". Three things it concluded from that round did not survive later measurement, and are
kept here because the way each one fell is the useful part:

| First-round reading | What later runs showed |
|---|---|
| "`pick_target` is not solved by either policy. Rollouts show both moving to the named object's neighbourhood and then missing the grasp" | They were not near misses. Where the gripper closed did not depend on the scene: the nearest object was the named one in 29-33% of episodes, which is chance with three objects on the table (notes: "A failing policy ignores the scene; it does not average"). The videos looked like near misses; the measurement of where the gripper closed said otherwise |
| "Training budget dominates" | The budget mattered even more than it looked: the same policy scores 0% after 1M samples and 85% after 5M. But a longer run of the first-round policies would not have helped. Three other causes had to go first: one conditioning vector for the action head, the previous action as an input, and a position table computed in half precision |
| "Pretraining did not help" (0/100 with a pretrained VLM, 1/100 with a randomly initialised one) | Neither policy could find the object, so the comparison said nothing about pretraining. It was repeated with every token passed to the action head and no history: notebook, "From-scratch VLA, every token, no history" |

The task itself is solved. With those causes removed, 4000 clean demonstrations and 5M training
samples give 93%; random start poses in the demonstrations give 98%; and demonstrations that also
contain recoveries give 94% with a policy that follows an object that is moved and tries again after
a missed grasp.
