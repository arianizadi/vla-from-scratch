// Small DOM helpers: element builders, in-place updates and motion that respects the reader's settings.

const SVG_NS = "http://www.w3.org/2000/svg";

function apply(el, attrs) {
  for (const [name, value] of Object.entries(attrs || {})) {
    if (name === "text") {
      el.textContent = value == null ? "" : String(value);
    } else if (value == null || value === false) {
      continue;
    } else if (name === "style" && typeof value === "object") {
      for (const [k, v] of Object.entries(value)) el.style.setProperty(k, v);
    } else if (name.startsWith("on") && typeof value === "function") {
      el.addEventListener(name.slice(2), value);
    } else {
      el.setAttribute(name, value === true ? "" : String(value));
    }
  }
}

function append(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    el.appendChild(typeof child === "object" ? child : document.createTextNode(String(child)));
  }
  return el;
}

/** An HTML element: h("div", {class: "x", onclick: fn, text: "..."}, ...children). */
export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  apply(el, attrs);
  return append(el, children);
}

/** An SVG element, same arguments as h(). */
export function s(tag, attrs, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  apply(el, attrs);
  return append(el, children);
}

/** Set an attribute only when it differs, so unchanged nodes are left alone. */
export function attr(el, name, value) {
  if (value == null || value === false) {
    if (el.hasAttribute(name)) el.removeAttribute(name);
    return;
  }
  const text = value === true ? "" : String(value);
  if (el.getAttribute(name) !== text) el.setAttribute(name, text);
}

export function attrs(el, values) {
  for (const name in values) attr(el, name, values[name]);
}

export function setText(el, value) {
  const text = value == null ? "" : String(value);
  if (el.textContent !== text) el.textContent = text;
}

export function setClass(el, name, on) {
  if (el.classList.contains(name) !== !!on) el.classList.toggle(name, !!on);
}

/**
 * Make the children of `parent` match `items`, reusing the node of every item whose key was there
 * before. Nodes keep their identity (and with it hover, focus and running animations); only new
 * items are created and only missing ones removed. `parent` must hold nothing but these children.
 */
export function reconcile(parent, items, keyOf, create, update) {
  const old = parent.__keyed || new Map();
  const next = new Map();
  let cursor = parent.firstChild;
  items.forEach((item, index) => {
    let key = keyOf(item, index);
    while (next.has(key)) key = `${key}+`; // two items with one key: both are kept
    let node = old.get(key);
    const fresh = !node;
    if (fresh) node = create(item, index);
    next.set(key, node);
    if (update) update(node, item, index, fresh);
    if (node === cursor) cursor = cursor.nextSibling;
    else parent.insertBefore(node, cursor);
  });
  for (const [key, node] of old) if (!next.has(key)) node.remove();
  parent.__keyed = next;
  return next;
}

const motionQuery = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;

export function reducedMotion() {
  return !!(motionQuery && motionQuery.matches);
}

const easeOut = (p) => 1 - Math.pow(1 - p, 3);

/** Call step(0..1) every frame for `duration` ms; returns a function that cancels it. */
export function animate(duration, step, done) {
  if (reducedMotion() || duration <= 0 || document.hidden) {
    step(1);
    if (done) done();
    return () => {};
  }
  let frame = 0;
  let start = null;
  const tick = (now) => {
    if (start === null) start = now;
    const p = Math.min(1, (now - start) / duration);
    step(easeOut(p));
    if (p < 1) frame = requestAnimationFrame(tick);
    else if (done) done();
  };
  frame = requestAnimationFrame(tick);
  return () => cancelAnimationFrame(frame);
}

/** Show a number in `el`, counting from the value it showed before. */
export function tweenNumber(el, value, format, duration = 600) {
  const previous = el.__value;
  if (el.__cancel) el.__cancel();
  el.__cancel = null;
  if (typeof value !== "number" || !Number.isFinite(value)) {
    el.__value = null;
    setText(el, format(null));
    return;
  }
  if (previous == null || previous === value) {
    el.__value = value;
    setText(el, format(value));
    return;
  }
  el.__cancel = animate(
    duration,
    (p) => {
      el.__value = previous + (value - previous) * p;
      setText(el, format(el.__value));
    },
    () => {
      el.__value = value;
      setText(el, format(value));
    },
  );
}

/** Restart a CSS animation that is tied to a class. */
export function replay(el, name) {
  el.classList.remove(name);
  void el.getBoundingClientRect(); // forces the style to settle, so adding the class starts anew
  el.classList.add(name);
}

/** Call fn(width) when the element's width changes; returns a function that stops watching. */
export function onWidth(el, fn) {
  let last = -1;
  const check = () => {
    const width = Math.round(el.clientWidth);
    if (width !== last && width > 0) {
      last = width;
      fn(width);
    }
  };
  if (!("ResizeObserver" in window)) {
    window.addEventListener("resize", check);
    requestAnimationFrame(check);
    return () => window.removeEventListener("resize", check);
  }
  const observer = new ResizeObserver(check);
  observer.observe(el);
  return () => observer.disconnect();
}

let counter = 0;
export function uid(prefix) {
  counter += 1;
  return `${prefix}-${counter}`;
}

/** localStorage that never throws (private windows, blocked storage). */
export const memory = {
  get(key, fallback) {
    try {
      const value = window.localStorage.getItem(`monitor.${key}`);
      return value == null ? fallback : value;
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      window.localStorage.setItem(`monitor.${key}`, String(value));
    } catch {
      /* the choice just does not survive a reload */
    }
  },
};
