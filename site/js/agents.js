// Fixed categorical order (validated: node validate_palette.js, all hard gates pass —
// see dataviz skill references/palette.md). Colors are CSS custom properties so they
// swap for dark mode via CSS alone; see css/style.css for the light/dark hex values.
// Assigned by position in the `agents` list (first-appearance order in results.csv):
// existing agents never change color when a new one is added, as long as
// prepare_dashboard_data.py doesn't reorder existing rows on regeneration.
export const PALETTE = [
  "var(--series-1)", // blue
  "var(--series-2)", // orange
  "var(--series-3)", // aqua
  "var(--series-4)", // yellow
  "var(--series-5)", // magenta
  "var(--series-6)", // green
  "var(--series-7)", // violet
  "var(--series-8)", // red
];

const OVERFLOW_COLOR = "#898781";

export function agentColor(agent, agents) {
  const i = agents.indexOf(agent);
  return i === -1 || i >= PALETTE.length ? OVERFLOW_COLOR : PALETTE[i];
}

export function colorDomainRange(agents) {
  return {
    domain: agents,
    range: agents.map((a) => agentColor(a, agents)),
  };
}
