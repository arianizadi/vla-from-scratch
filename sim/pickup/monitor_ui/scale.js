// Scales, ticks and the arithmetic behind the charts.

export function scaleLinear(d0, d1, r0, r1) {
  const span = d1 - d0 || 1;
  const k = (r1 - r0) / span;
  const f = (value) => r0 + (value - d0) * k;
  f.invert = (pixel) => d0 + (pixel - r0) / (k || 1);
  return f;
}

export function scaleLog(d0, d1, r0, r1) {
  const l0 = Math.log(d0);
  const span = Math.log(d1) - l0 || 1;
  const k = (r1 - r0) / span;
  const f = (value) => r0 + (Math.log(Math.max(value, d0 * 1e-6)) - l0) * k;
  f.invert = (pixel) => Math.exp(l0 + (pixel - r0) / (k || 1));
  return f;
}

/** A round step (1, 2 or 5 times a power of ten) that divides `span` into about `count` parts. */
function niceStep(span, count) {
  const raw = span / Math.max(1, count);
  if (!(raw > 0)) return 1;
  const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / magnitude;
  return (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * magnitude;
}

export function linearTicks(min, max, count) {
  if (!(max > min)) return [min];
  const step = niceStep(max - min, count);
  const out = [];
  for (let i = Math.ceil(min / step - 1e-9); i * step <= max + step * 1e-9; i++) out.push(+(i * step).toPrecision(12));
  return out;
}

/** The smallest round number at or above `value`, so an axis ends on a tick. */
export function niceCeil(value, count = 4) {
  if (!(value > 0)) return 1;
  const step = niceStep(value, count);
  return +(Math.ceil(value / step - 1e-9) * step).toPrecision(12);
}

/** Ticks of a logarithmic axis: powers of ten, with 2 and 5 between them when the range is short. */
export function logTicks(min, max) {
  const out = [];
  const first = Math.floor(Math.log10(min));
  const last = Math.ceil(Math.log10(max));
  const multiples = last - first <= 3 ? [1, 2, 5] : [1];
  for (let e = first; e <= last; e++) {
    for (const m of multiples) {
      const v = m * Math.pow(10, e);
      if (v >= min * 0.999 && v <= max * 1.001) out.push(+v.toPrecision(12));
    }
  }
  return out;
}

/** Round bounds (1, 2, 5 x 10^n) around a positive range, for a logarithmic axis. */
export function logDomain(min, max) {
  const down = (v) => {
    const m = Math.pow(10, Math.floor(Math.log10(v)));
    const n = v / m;
    return (n >= 5 ? 5 : n >= 2 ? 2 : 1) * m;
  };
  const up = (v) => {
    const m = Math.pow(10, Math.floor(Math.log10(v)));
    const n = v / m;
    return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * m;
  };
  const lo = down(min);
  let hi = up(max);
  if (hi <= lo) hi = lo * 10;
  return [lo, hi];
}

/** Index of the point whose x is closest to `x`; `points` are sorted by x. */
export function nearestIndex(points, x) {
  if (!points.length) return -1;
  let lo = 0;
  let hi = points.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (points[mid].x < x) lo = mid;
    else hi = mid;
  }
  return Math.abs(points[lo].x - x) <= Math.abs(points[hi].x - x) ? lo : hi;
}

/**
 * Largest-triangle-three-buckets: keep `threshold` of the points, choosing in every bucket the one
 * that changes the picture most, so spikes survive where plain averaging would flatten them.
 */
export function lttb(points, threshold) {
  const n = points.length;
  if (threshold >= n || threshold < 3) return points;
  const out = [points[0]];
  const every = (n - 2) / (threshold - 2);
  let a = 0;
  for (let i = 0; i < threshold - 2; i++) {
    const start = Math.floor((i + 1) * every) + 1;
    const end = Math.min(Math.floor((i + 2) * every) + 1, n);
    let avgX = 0;
    let avgY = 0;
    for (let j = start; j < end; j++) {
      avgX += points[j].x;
      avgY += points[j].y;
    }
    const count = Math.max(1, end - start);
    avgX /= count;
    avgY /= count;
    const from = Math.floor(i * every) + 1;
    const to = Math.min(Math.floor((i + 1) * every) + 1, n - 1);
    let best = from;
    let bestArea = -1;
    for (let j = from; j < to; j++) {
      const area = Math.abs(
        (points[a].x - avgX) * (points[j].y - points[a].y) - (points[a].x - points[j].x) * (avgY - points[a].y),
      );
      if (area > bestArea) {
        bestArea = area;
        best = j;
      }
    }
    out.push(points[best]);
    a = best;
  }
  out.push(points[n - 1]);
  return out;
}

export function quantile(sorted, q) {
  if (!sorted.length) return NaN;
  const position = (sorted.length - 1) * q;
  const base = Math.floor(position);
  const rest = position - base;
  return sorted[base] + (sorted[Math.min(base + 1, sorted.length - 1)] - sorted[base]) * rest;
}

export function median(values) {
  return quantile(
    values.slice().sort((a, b) => a - b),
    0.5,
  );
}

/** Wilson 95% interval of k successes in n trials, for results that do not carry one. */
export function wilson(k, n) {
  if (!n) return [0, 1];
  const z = 1.959964;
  const p = k / n;
  const centre = p + (z * z) / (2 * n);
  const spread = z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n));
  const bottom = 1 + (z * z) / n;
  return [Math.max(0, (centre - spread) / bottom), Math.min(1, (centre + spread) / bottom)];
}
