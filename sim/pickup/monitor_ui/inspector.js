// One evaluation up close: scrub through the checkpoints of a run and see, for the chosen one, the
// headline numbers, where the gripper closed (grasp.js), success per object and the rollout videos.

import { h, setText, setClass, attr, reconcile } from "./dom.js";
import { loadEvaluation, loadVideos } from "./store.js";
import { createGrasp } from "./grasp.js";
import { objectGlyph, objectLabel, playIcon, pauseIcon } from "./icons.js";
import { variantOf, checkpointsOf, intervalOf, WEIGHTS, WEIGHT_LABEL } from "./model.js";
import { fmtPct, fmtMm, fmtSamples, fmtBytes, isNum, DASH } from "./format.js";
import { niceCeil } from "./scale.js";

// The README has the scripted expert write its results to results/pickup/expert/<task>: the same
// episodes, solved by a program. Next to a policy's map it shows what a good one looks like.
const REFERENCE_RUN = "expert";
const VIDEOS_SHOWN = 6;

/** The fields of a results.json that the snapshot carries for every evaluation (see runstate.py). */
function rowOf(result) {
  const grasp = result.grasp || {};
  return {
    successes: result.successes,
    episodes: result.episodes,
    success_rate: result.success_rate,
    ci95: result.success_ci95,
    at_named_object: grasp.closed_nearest_is_target_rate,
    median_miss_named_mm: grasp.median_xy_err_target_mm,
    median_miss_nearest_mm: grasp.median_xy_err_nearest_mm,
    within_10mm: grasp.within_10mm_of_target_rate,
    within_20mm: grasp.within_20mm_of_target_rate,
    wrong_object_rate: result.wrong_object_rate,
  };
}

function fact(label) {
  const value = h("dd", { class: "num" });
  const sub = h("dd", { class: "fact-sub" });
  return { el: h("div", { class: "fact" }, h("dt", { text: label }), value, sub), value, sub };
}

export function createInspector({ onPick, onScene }) {
  let state = null; // {runName, task, evaluations, positions, selection: {checkpoint, weights}}
  let reference = false; // showing the scripted expert instead of the selected checkpoint
  let referenceResult = null;
  let details = null; // the loaded results.json of what is shown
  let loadedKey = "";
  let failed = false; // the last request failed in a way that may pass: the server away, a file being written
  let request = 0;
  let playing = 0;
  let allVideos = false;

  const grasp = createGrasp();
  const kind = h("span", { text: "Checkpoint " });
  const number = h("span", { class: "num" });
  const where = h("span", { class: "insp-where" });
  const weights = h("div", { class: "seg", role: "group", "aria-label": "Weights of the checkpoint" });
  const play = h("button", { class: "icon-button", type: "button" }, playIcon());
  const range = h("input", { class: "range", type: "range", min: 0, max: 0, step: 1, "aria-label": "Checkpoint" });
  const ticks = h("div", { class: "scrub-ticks", "aria-hidden": "true" });
  const scrub = h("div", { class: "scrub" }, play, h("div", { class: "scrub-track" }, range, ticks));
  const facts = {
    success: fact("Success"),
    named: fact("Closed at the named object"),
    miss: fact("Median miss"),
    near: fact("Within 2 cm"),
    wrong: fact("Lifted the wrong object"),
  };
  const extra = h("p", { class: "insp-extra" });
  const objects = h("div", { class: "objects" });
  const objectsScale = h("span", { class: "section-note" });
  const videoGrid = h("div", { class: "video-grid" });
  const videoMore = h("button", { class: "button", type: "button", hidden: true });
  const videos = h(
    "section",
    { class: "videos", hidden: true },
    h("h3", { class: "section-title", text: "Rollout videos" }),
    videoGrid,
    videoMore,
  );
  const el = h(
    "section",
    { class: "panel inspector", "aria-label": "Checkpoint inspector" },
    h("div", { class: "insp-head" }, h("h2", { class: "panel-title" }, kind, number), where, weights),
    scrub,
    h(
      "dl",
      { class: "facts" },
      Object.values(facts).map((f) => f.el),
    ),
    h(
      "div",
      { class: "section-head" },
      h("h3", { class: "section-title", text: "Where the gripper closed" }),
      h("span", { class: "section-note", text: "seen from above, one mark per episode" }),
    ),
    grasp.el,
    extra,
    h("div", { class: "section-head" }, h("h3", { class: "section-title", text: "Success by object" }), objectsScale),
    objects,
    videos,
  );

  const rowAt = (checkpoint, kind) =>
    state.evaluations.find((e) => e.checkpoint === checkpoint && e.weights === kind) || null;
  const selectedRow = () => rowAt(state.selection.checkpoint, state.selection.weights);

  function pick(checkpoint, kind) {
    const wanted = kind || state.selection.weights;
    const available = WEIGHTS.filter((w) => rowAt(checkpoint, w));
    if (!available.length) return;
    reference = false;
    onPick({ checkpoint, weights: available.includes(wanted) ? wanted : available[available.length - 1] });
  }

  function stopPlaying() {
    if (!playing) return;
    clearInterval(playing);
    playing = 0;
    drawPlay();
  }

  function drawPlay() {
    play.replaceChildren(playing ? pauseIcon() : playIcon());
    const text = playing ? "Pause" : "Play through the checkpoints";
    play.setAttribute("aria-label", text);
    play.title = text;
  }

  play.addEventListener("click", () => {
    if (playing) return stopPlaying();
    const checkpoints = checkpointsOf(state.evaluations);
    if (checkpoints.length < 2) return undefined;
    if (reference || state.selection.checkpoint === checkpoints[checkpoints.length - 1]) pick(checkpoints[0]);
    playing = setInterval(() => {
      const list = checkpointsOf(state.evaluations);
      const index = list.indexOf(state.selection.checkpoint);
      if (index < 0 || index >= list.length - 1) return stopPlaying();
      return pick(list[index + 1]);
    }, 1500);
    return drawPlay();
  });

  range.addEventListener("input", () => {
    stopPlaying();
    const checkpoints = checkpointsOf(state.evaluations);
    pick(checkpoints[Number(range.value)]);
  });

  function drawHead() {
    const checkpoints = checkpointsOf(state.evaluations);
    const { checkpoint } = state.selection;
    const index = Math.max(0, checkpoints.indexOf(checkpoint));
    setText(kind, reference ? "Scripted expert" : "Checkpoint ");
    setText(number, reference ? "" : checkpoint);
    const at = state.positions.get(checkpoint);
    setText(
      where,
      reference
        ? "the same episodes, solved by a program"
        : at
          ? `${at.estimated ? "about " : ""}${fmtSamples(at.x)} samples`
          : "",
    );

    const options = [
      ...WEIGHTS.map((w) => ({ id: w, label: WEIGHT_LABEL[w], off: !rowAt(checkpoint, w) })),
      { id: "reference", label: "scripted expert", off: false },
    ];
    reconcile(
      weights,
      options,
      (option) => option.id,
      (option) => {
        const button = h("button", { type: "button", class: "seg-button", text: option.label });
        button.addEventListener("click", () => {
          stopPlaying();
          if (option.id === "reference") {
            reference = true;
            render();
          } else {
            const leaving = reference;
            pick(state.selection.checkpoint, option.id);
            if (leaving) render();
          }
        });
        return button;
      },
      (button, option) => {
        button.disabled = option.off;
        attr(
          button,
          "aria-pressed",
          String(option.id === "reference" ? reference : !reference && state.selection.weights === option.id),
        );
        attr(
          button,
          "title",
          option.off ? `Checkpoint ${checkpoint} has no evaluation with ${option.label} weights` : null,
        );
      },
    );

    scrub.hidden = checkpoints.length < 2;
    range.max = String(Math.max(0, checkpoints.length - 1));
    if (Number(range.value) !== index) range.value = String(index);
    range.setAttribute("aria-valuetext", `Checkpoint ${checkpoint}`);
    const every = Math.ceil(checkpoints.length / 16);
    reconcile(
      ticks,
      checkpoints.map((c, i) => ({ c, i })),
      (t) => t.c,
      () => h("span", { class: "scrub-tick num" }),
      (node, t) => {
        node.style.left = checkpoints.length > 1 ? `${(100 * t.i) / (checkpoints.length - 1)}%` : "50%";
        setText(node, t.i % every === 0 || t.c === checkpoint ? t.c : "");
        setClass(node, "on", t.c === checkpoint && !reference);
      },
    );
  }

  function drawFacts() {
    const row = reference ? referenceResult && rowOf(referenceResult) : selectedRow();
    const f = facts;
    if (!row) {
      for (const item of Object.values(f)) {
        setText(item.value, DASH);
        setText(item.sub, "");
      }
      return;
    }
    const interval = intervalOf(row);
    setText(f.success.value, isNum(row.successes) ? `${row.successes} of ${row.episodes}` : DASH);
    setText(
      f.success.sub,
      interval
        ? `${fmtPct(row.success_rate)}, 95% interval ${(100 * interval[0]).toFixed(0)}${DASH}${(100 * interval[1]).toFixed(0)}%`
        : "",
    );
    setText(f.named.value, fmtPct(row.at_named_object));
    const scene =
      details && Array.isArray(details.per_episode) && details.per_episode[0] && details.per_episode[0].close_object_xy;
    const count = scene ? Object.keys(scene).length : 0;
    setText(
      f.named.sub,
      count > 1 ? `${fmtPct(1 / count)} is picking one of ${count} at random` : "of the times the gripper closed",
    );
    setText(f.miss.value, fmtMm(row.median_miss_named_mm));
    setText(
      f.miss.sub,
      isNum(row.median_miss_nearest_mm) ? `${fmtMm(row.median_miss_nearest_mm)} to the nearest object` : "",
    );
    setText(f.near.value, fmtPct(row.within_20mm));
    setText(f.near.sub, isNum(row.within_10mm) ? `${fmtPct(row.within_10mm)} within 1 cm` : "");
    setText(f.wrong.value, fmtPct(row.wrong_object_rate));
    setText(f.wrong.sub, "of all episodes");
  }

  function drawExtra() {
    const parts = [];
    if (details) {
      const latency = details.inference_latency_ms;
      if (latency && isNum(latency.p50))
        parts.push(
          `The policy answered in ${Math.round(latency.p50)} ms (median)${isNum(latency.p95) ? `, ${Math.round(latency.p95)} ms at the 95th percentile` : ""}.`,
        );
      if (isNum(details.mean_steps_success))
        parts.push(`Successful episodes took ${Math.round(details.mean_steps_success)} steps on average.`);
      const closed = details.grasp && details.grasp.closed_gripper_rate;
      if (isNum(closed) && closed < 1) parts.push(`The gripper closed in ${fmtPct(closed)} of the episodes.`);
    }
    setText(extra, parts.join(" "));
    extra.hidden = !parts.length;
  }

  function drawObjects() {
    const { checkpoint } = state.selection;
    const rows = reference
      ? referenceResult && referenceResult.per_target
        ? [{ weights: REFERENCE_RUN, per_target: referenceResult.per_target }]
        : []
      : WEIGHTS.map((w) => rowAt(checkpoint, w)).filter(Boolean);
    let top = 0.1;
    for (const e of [...state.evaluations, ...(reference ? rows : [])]) {
      for (const t of Object.values(e.per_target || {}))
        if (t.episodes && t.successes / t.episodes > top) top = t.successes / t.episodes;
    }
    top = niceCeil(top, 2);
    const names = [...new Set(rows.flatMap((row) => Object.keys(row.per_target || {})))].sort();
    setText(
      objectsScale,
      names.length
        ? `${reference ? "scripted expert" : `checkpoint ${checkpoint}`}, bars run from 0 to ${fmtPct(top)}`
        : "",
    );
    reconcile(
      objects,
      names,
      (name) => name,
      (name) =>
        h(
          "div",
          { class: "object-row" },
          h("span", { class: "object-name" }, objectGlyph(name), objectLabel(name)),
          h("div", { class: "object-bars" }),
        ),
      (node, name) => {
        reconcile(
          node.lastChild,
          rows.filter((row) => row.per_target && row.per_target[name]),
          (row) => row.weights,
          (row) =>
            h(
              "div",
              {
                class: "object-bar",
                style: { "--c": row.weights === REFERENCE_RUN ? "var(--good)" : `var(--${row.weights})` },
              },
              h("span", { class: "object-weights", text: WEIGHT_LABEL[row.weights] || "" }), // the expert's single bar needs no label
              h("div", { class: "bar-track" }, h("i")),
              h("span", { class: "object-count num" }),
            ),
          (bar, row) => {
            const t = row.per_target[name];
            const share = t.episodes ? t.successes / t.episodes : 0;
            bar.children[1].firstChild.style.width = `${(100 * Math.min(1, share / top)).toFixed(1)}%`;
            setText(bar.children[2], `${t.successes}/${t.episodes}`);
            attr(
              bar,
              "title",
              `${objectLabel(name)}, ${WEIGHT_LABEL[row.weights] || row.weights}: ${t.successes} of ${t.episodes} episodes (${fmtPct(share)})`,
            );
          },
        );
      },
    );
    if (!names.length) objects.replaceChildren();
  }

  function drawVideos(list) {
    videos.hidden = !list.length;
    if (!list.length) return;
    const shown = allVideos ? list : list.slice(0, VIDEOS_SHOWN);
    reconcile(
      videoGrid,
      shown,
      (v) => v.url,
      (v) => {
        const words = [];
        if (v.episode != null) words.push(`Episode ${v.episode}`);
        if (v.target) words.push(objectLabel(v.target));
        const caption = h("figcaption", null, h("span", { text: words.join(", ") || v.name }));
        if (v.success != null)
          caption.appendChild(h("b", { class: v.success ? "ok" : "fail", text: v.success ? "success" : "failure" }));
        caption.appendChild(h("span", { class: "video-size num", text: fmtBytes(v.bytes) }));
        return h(
          "figure",
          { class: "video" },
          h("video", {
            src: v.url.replace(/^\//, ""),
            controls: true,
            preload: "metadata",
            playsinline: true,
            muted: true,
            "aria-label": words.join(", ") || v.name,
          }),
          caption,
        );
      },
    );
    videoMore.hidden = allVideos || list.length <= VIDEOS_SHOWN;
    setText(videoMore, `Show all ${list.length} videos`);
    videoMore.onclick = () => {
      allVideos = true;
      drawVideos(list);
    };
  }

  function load() {
    const row = selectedRow();
    const target = reference
      ? { run: REFERENCE_RUN, variant: "", stamp: "" }
      : row
        ? { run: state.runName, variant: variantOf(row), stamp: `${row.successes}/${row.episodes}` }
        : null;
    const key = target ? `${target.run}/${state.task}/${target.variant}/${target.stamp}` : "";
    if (key === loadedKey && !failed) return;
    loadedKey = key;
    failed = false;
    const mine = ++request;
    if (!target) {
      details = null;
      grasp.set({ message: "Choose a checkpoint to see where its gripper closed." });
      drawVideos([]);
      return;
    }
    grasp.set({ loading: true });
    loadEvaluation(target.run, state.task, target.variant, target.stamp).then(
      (result) => {
        if (mine !== request) return;
        details = result;
        if (reference) referenceResult = result;
        const episodes = Array.isArray(result.per_episode) ? result.per_episode : [];
        grasp.set({ episodes });
        drawFacts();
        drawExtra();
        if (reference) drawObjects();
        const scene = episodes[0] && episodes[0].close_object_xy;
        if (scene && onScene) onScene(Object.keys(scene).length);
        loadVideos(target.run, state.task, target.variant, target.stamp).then(
          (list) => mine === request && drawVideos(list),
          () => mine === request && drawVideos([]),
        );
      },
      (error) => {
        if (mine !== request) return;
        details = null;
        drawVideos([]);
        failed = error.status == null || error.status === 503; // may pass: ask again when the server answers
        const missing = reference && error.status === 404;
        grasp.set({
          message: missing
            ? `There are no scripted-expert results for ${state.task} yet. Create them with: python -m sim.pickup.eval --task ${state.task} --episodes 100 --expert --out results/pickup/${REFERENCE_RUN}/${state.task}`
            : `These results could not be loaded (${error.message}).`,
        });
        drawFacts();
        drawExtra();
      },
    );
    if (!reference && row) {
      // Have the neighbours ready, so that scrubbing does not wait for the network.
      const checkpoints = checkpointsOf(state.evaluations);
      const index = checkpoints.indexOf(row.checkpoint);
      for (const near of [checkpoints[index - 1], checkpoints[index + 1]]) {
        const other = near == null ? null : rowAt(near, row.weights);
        if (other)
          loadEvaluation(state.runName, state.task, variantOf(other), `${other.successes}/${other.episodes}`).catch(
            () => {},
          );
      }
    }
  }

  function render() {
    drawHead();
    drawFacts();
    drawObjects();
    load();
  }

  drawPlay();

  return {
    el,
    set(next) {
      if (!state || state.runName !== next.runName || state.task !== next.task) {
        reference = false;
        referenceResult = null;
        details = null;
        loadedKey = "";
        failed = false;
        allVideos = false;
        stopPlaying();
      }
      state = next;
      render();
    },
    /** Ask again for results whose request failed in a way that may have passed. */
    retry() {
      if (state && failed) load();
    },
    destroy() {
      stopPlaying();
      request += 1;
      grasp.destroy();
    },
  };
}
