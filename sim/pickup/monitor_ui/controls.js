// Small controls shared by the views.

import { h, attr, reconcile } from "./dom.js";

/**
 * A row of buttons of which one is pressed. options: [{id, label, title}]; onChange(id) is called
 * when the reader presses another one. Returns {el, set(id), options(list)}.
 */
export function segmented({ label, options, value, onChange }) {
  let current = value;
  let list = options;
  const el = h("div", { class: "seg", role: "group", "aria-label": label });
  function draw() {
    reconcile(
      el,
      list,
      (option) => option.id,
      (option) => {
        const button = h("button", { type: "button", class: "seg-button", text: option.label });
        if (option.title) button.title = option.title;
        button.addEventListener("click", () => {
          if (current === option.id) return;
          current = option.id;
          draw();
          onChange(option.id);
        });
        return button;
      },
      (button, option) => attr(button, "aria-pressed", String(option.id === current)),
    );
  }
  draw();
  return {
    el,
    set(id) {
      current = id;
      draw();
    },
    options(next) {
      list = next;
      draw();
    },
  };
}
