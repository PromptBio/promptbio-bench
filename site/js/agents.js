// Fixed categorical order (validated: node validate_palette.js, all hard gates pass —
// see dataviz skill references/palette.md). Assigned by position in the `agents` list
// (first-appearance order in results.csv): existing agents never change color when a
// new one is added, as long as results.csv doesn't reorder existing rows.
const PALETTE_VARS = [
  "--series-1", // blue
  "--series-2", // orange
  "--series-3", // aqua
  "--series-4", // yellow
  "--series-5", // magenta
  "--series-6", // green
  "--series-7", // violet
  "--series-8", // red
];

const OVERFLOW_COLOR = "#898781";

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// Plotly renders its own SVG and expects literal color strings (var() is not
// reliably resolved inside it), so colors are read from the CSS custom properties
// at call time rather than passed through as var() strings. Call again after a
// light/dark switch to pick up the new values.
export function agentColor(agent, agents) {
  const i = agents.indexOf(agent);
  if (i === -1 || i >= PALETTE_VARS.length) return OVERFLOW_COLOR;
  return cssVar(PALETTE_VARS[i]) || OVERFLOW_COLOR;
}

export function getTheme() {
  return {
    surface: cssVar("--surface"),
    textPrimary: cssVar("--text-primary"),
    textSecondary: cssVar("--text-secondary"),
    muted: cssVar("--muted"),
    gridline: cssVar("--gridline"),
    axis: cssVar("--axis"),
  };
}
