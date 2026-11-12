// How numbers are written everywhere in the monitor.

export const DASH = "–";

export function isNum(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function trimZeros(text) {
  return text.includes(".") ? text.replace(/\.?0+$/, "") : text;
}

export function fmtInt(value) {
  return isNum(value) ? Math.round(value).toLocaleString("en-US") : DASH;
}

/** Samples seen, as read in a sentence: 924K, 1.00M. */
export function fmtSamples(value) {
  if (!isNum(value)) return DASH;
  const a = Math.abs(value);
  if (a >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (a >= 999500) return `${(value / 1e6).toFixed(2)}M`;
  if (a >= 1e4) return `${Math.round(value / 1e3)}K`;
  if (a >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return String(Math.round(value));
}

/** Samples on an axis: 0, 200K, 1M, 1.5M. */
export function fmtAxisSamples(value) {
  if (!isNum(value)) return "";
  const a = Math.abs(value);
  if (a >= 1e9) return `${trimZeros((value / 1e9).toFixed(2))}B`;
  if (a >= 1e6) return `${trimZeros((value / 1e6).toFixed(2))}M`;
  if (a >= 1e3) return `${trimZeros((value / 1e3).toFixed(1))}K`;
  return String(Math.round(value));
}

export function fmtLoss(value) {
  if (!isNum(value)) return DASH;
  const a = Math.abs(value);
  if (a >= 100) return value.toFixed(0);
  if (a >= 10) return value.toFixed(2);
  if (a >= 0.01 || a === 0) return value.toFixed(3);
  return value.toPrecision(2);
}

export function fmtAxisNumber(value) {
  if (!isNum(value)) return "";
  const a = Math.abs(value);
  if (a >= 1e4) return fmtAxisSamples(value);
  if (a >= 100 || a === 0) return String(Math.round(value));
  return trimZeros(value.toPrecision(a >= 1 ? 3 : 2));
}

export function fmtPct(rate, digits = 0) {
  return isNum(rate) ? `${(100 * rate).toFixed(digits)}%` : DASH;
}

export function fmtMm(value) {
  if (!isNum(value)) return DASH;
  return `${Math.abs(value) < 10 ? value.toFixed(1) : Math.round(value)} mm`;
}

export function fmtRate(value) {
  return isNum(value) ? `${fmtInt(value)}/s` : DASH;
}

export function fmtLearningRate(value) {
  if (!isNum(value)) return DASH;
  if (value === 0) return "0";
  return value.toExponential(1).replace("e-0", "e-").replace("e+0", "e").replace("e-", "e−");
}

/** A length of time that remains or has passed: 45 s, 12 min, 3 h 05 min, 2 d 4 h. */
export function fmtSpan(seconds) {
  if (!isNum(seconds) || seconds < 0) return DASH;
  const total = Math.round(seconds);
  if (total < 60) return `${total} s`;
  if (total < 3600) return `${Math.floor(total / 60)} min`;
  if (total < 86400)
    return `${Math.floor(total / 3600)} h ${String(Math.floor((total % 3600) / 60)).padStart(2, "0")} min`;
  return `${Math.floor(total / 86400)} d ${Math.floor((total % 86400) / 3600)} h`;
}

export function fmtClock(milliseconds) {
  const d = new Date(milliseconds);
  const two = (n) => String(n).padStart(2, "0");
  return `${two(d.getHours())}:${two(d.getMinutes())}:${two(d.getSeconds())}`;
}

export function fmtBytes(value) {
  if (!isNum(value)) return DASH;
  if (value >= 1 << 20) return `${(value / (1 << 20)).toFixed(1)} MB`;
  if (value >= 1 << 10) return `${Math.round(value / (1 << 10))} KB`;
  return `${value} B`;
}
