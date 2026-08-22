const STORAGE_KEY = "pbb-theme";
const CYCLE = ["light", "dark", "system"];
const root = document.documentElement;

function currentChoice() {
  const saved = localStorage.getItem(STORAGE_KEY);
  return saved === "dark" || saved === "light" ? saved : "system";
}

function applyChoice(choice) {
  if (choice === "system") {
    root.removeAttribute("data-theme");
    localStorage.removeItem(STORAGE_KEY);
  } else {
    root.setAttribute("data-theme", choice);
    localStorage.setItem(STORAGE_KEY, choice);
  }
}

// Wires up the nav's single theme button: each click cycles light → dark →
// system → light. Calls onChange after every switch so callers (e.g. Plotly
// charts, which resolve CSS vars to literal colors at build time) can
// re-render with the new palette.
export function initThemeToggle(onChange) {
  const btn = document.querySelector(".theme-toggle");
  if (!btn) return;

  let choice = currentChoice();
  btn.dataset.active = choice;
  btn.title = `Theme: ${choice}`;

  btn.addEventListener("click", () => {
    choice = CYCLE[(CYCLE.indexOf(choice) + 1) % CYCLE.length];
    applyChoice(choice);
    btn.dataset.active = choice;
    btn.title = `Theme: ${choice}`;
    if (onChange) onChange();
  });
}
