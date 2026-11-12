// What the views derive from a run of the snapshot. Pure functions over the data of /api/state.

import { isNum } from "./format.js";
import { wilson } from "./scale.js";

export const WEIGHTS = ["raw", "ema"];
export const WEIGHT_LABEL = { raw: "raw", ema: "EMA" };
export const STATE_ORDER = { training: 0, stalled: 1, waiting: 2, done: 3 };

/** The directory name of an evaluation, as /api/evaluation wants it. */
export function variantOf(evaluation) {
  return `ckpt_${evaluation.checkpoint}_${evaluation.weights}`;
}

/** Task names of a run, the one with the most evaluations first. */
export function tasksOf(run) {
  const counts = new Map();
  for (const e of run.evaluations || []) counts.set(e.task, (counts.get(e.task) || 0) + 1);
  return [...counts.keys()].sort((a, b) => counts.get(b) - counts.get(a) || (a < b ? -1 : 1));
}

export function evalsOf(run, task) {
  return (run.evaluations || []).filter((e) => e.task === task);
}

export function checkpointsOf(evaluations) {
  return [...new Set(evaluations.map((e) => e.checkpoint))].sort((a, b) => a - b);
}

/** The evaluation with the highest success; ties go to the later checkpoint, then to EMA weights. */
export function bestOf(evaluations) {
  let best = null;
  for (const e of evaluations) {
    if (!isNum(e.success_rate)) continue;
    if (
      !best ||
      e.success_rate > best.success_rate ||
      (e.success_rate === best.success_rate &&
        (e.checkpoint > best.checkpoint || (e.checkpoint === best.checkpoint && e.weights === "ema")))
    ) {
      best = e;
    }
  }
  return best;
}

export function intervalOf(evaluation) {
  const ci = evaluation.ci95;
  if (Array.isArray(ci) && isNum(ci[0]) && isNum(ci[1])) return ci;
  return isNum(evaluation.successes) && evaluation.episodes ? wilson(evaluation.successes, evaluation.episodes) : null;
}

/**
 * Where each checkpoint sits on the samples axis: Map(checkpoint -> {x, estimated}).
 *
 * The log tells the samples of most checkpoints. For the others the position is estimated from the
 * spacing of the known ones, or from the planned length of the run.
 */
export function checkpointPositions(run, evaluations) {
  const known = new Map();
  for (const e of evaluations) {
    // A checkpoint cannot have seen more samples than the run has: such a figure is a guess, redone below.
    if (isNum(e.samples) && !(run.samples_seen > 0 && e.samples > run.samples_seen * 1.001))
      known.set(e.checkpoint, e.samples);
  }
  const checkpoints = checkpointsOf(evaluations);
  const ratios = [...known]
    .filter(([checkpoint]) => checkpoint > 0)
    .map(([checkpoint, samples]) => samples / checkpoint)
    .sort((a, b) => a - b);
  let per = null;
  if (ratios.length) per = ratios[ratios.length >> 1];
  else if (run.samples_total && checkpoints.length) per = run.samples_total / Math.max(1, ...checkpoints);
  const out = new Map();
  for (const checkpoint of checkpoints) {
    if (known.has(checkpoint)) out.set(checkpoint, { x: known.get(checkpoint), estimated: false });
    else out.set(checkpoint, { x: per == null ? checkpoint : checkpoint * per, estimated: true });
  }
  return out;
}

/**
 * The loss curve as points with strictly increasing samples, without values a chart cannot draw.
 *
 * Samples that go backwards mean the run resumed from an earlier checkpoint, or its log was
 * replaced: what was logged beyond that point belongs to the attempt that was abandoned.
 */
export function lossPoints(run) {
  const out = [];
  for (const pair of run.loss_curve || []) {
    if (!Array.isArray(pair) || !isNum(pair[0]) || !isNum(pair[1])) continue;
    while (out.length && out[out.length - 1].x >= pair[0]) out.pop();
    out.push({ x: pair[0], y: pair[1] });
  }
  return out;
}

/** One series of an evaluation chart: the points of one kind of weights, by training samples. */
export function evalSeries(evaluations, positions, weights, valueOf) {
  const points = [];
  for (const e of evaluations) {
    if (e.weights !== weights) continue;
    const y = valueOf(e);
    if (!isNum(y)) continue;
    const at = positions.get(e.checkpoint);
    points.push({ x: at ? at.x : e.checkpoint, y, key: e.checkpoint, evaluation: e });
  }
  return points.sort((a, b) => a.x - b.x);
}

export function progressOf(run) {
  return run.samples_total > 0 ? Math.min(1, Math.max(0, run.samples_seen / run.samples_total)) : 0;
}

/** A string that changes exactly when something a view draws for this run has changed. */
export function signature(run) {
  const curve = run.loss_curve || [];
  const last = curve.length ? curve[curve.length - 1] : null;
  return [
    run.state,
    run.samples_seen,
    run.samples_total,
    run.loss,
    run.learning_rate,
    run.samples_per_second,
    run.restarts,
    run.checkpoints,
    run.evaluations_running,
    curve.length,
    last && last[0],
    last && last[1],
    JSON.stringify(run.evaluations || []),
  ].join("|");
}

/** What a run card and the comparison show about the closed-loop results of a run. */
export function summaryOf(run) {
  const tasks = tasksOf(run);
  const task = tasks[0] || null;
  const evaluations = task ? evalsOf(run, task) : [];
  return { tasks, task, evaluations, best: bestOf(evaluations), checkpoints: checkpointsOf(evaluations) };
}
