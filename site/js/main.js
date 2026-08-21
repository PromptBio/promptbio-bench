import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";
import {
  accuracyCharts,
  completionCharts,
  costCharts,
  hasCost,
  compositionCharts,
  hasComposition,
} from "./charts.js";
import { agentColor } from "./agents.js";

// Loaded as a classic <script> global in index.html (see comment there).
const Plotly = window.Plotly;

const DATA_URL = "data/results.csv";
const PLOTLY_CONFIG = { responsive: true, displayModeBar: false };

const el = (id) => document.getElementById(id);

function renderError(message) {
  const banner = el("error-banner");
  banner.textContent = message;
  banner.hidden = false;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function makeFigure(title) {
  const figure = document.createElement("figure");
  if (title) {
    const caption = document.createElement("figcaption");
    caption.textContent = title;
    figure.appendChild(caption);
  }
  return figure;
}

// spec is a {data, layout} Plotly spec, or null (renders `note` instead). Building
// figures is split from plotting them: `pending` collects {div, spec} and every figure
// across every section is appended to the DOM first, with all Plotly.newPlot calls run
// afterward in a second pass. Interleaving DOM insertion with newPlot — appending a
// later sibling into the same flex/grid row while an earlier chart's responsive
// ResizeObserver was still settling — was corrupting the earlier chart's axis/traces
// even though its own spec was correct in isolation.
function mountFigure(container, spec, { title, note } = {}, pending) {
  const figure = makeFigure(title);
  container.appendChild(figure);
  if (spec) {
    const plotDiv = document.createElement("div");
    figure.appendChild(plotDiv);
    pending.push({ div: plotDiv, spec });
  } else if (note) {
    const p = document.createElement("p");
    p.className = "placeholder";
    p.textContent = note;
    figure.appendChild(p);
  }
}

// Left: the headline number per agent. Right: the same metric broken out by
// difficulty. Side by side in one row instead of stacked, per the "overall vs.
// breakdown" pairing used for accuracy and completion.
function mountPair(container, { overall, byDifficulty }, { overallTitle, breakdownTitle, note }, pending) {
  const pair = document.createElement("div");
  pair.className = "chart-pair";
  container.appendChild(pair);
  mountFigure(pair, overall, { title: overallTitle }, pending);
  mountFigure(pair, byDifficulty, { title: breakdownTitle, note }, pending);
}

function buildAgentPicker(agents, visible, onToggle) {
  const container = el("agent-picker");
  clear(container);
  for (const agent of agents) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip" + (visible.has(agent) ? " chip--active" : "");
    chip.style.setProperty("--chip-color", agentColor(agent, agents));
    chip.textContent = agent;
    chip.addEventListener("click", () => onToggle(agent));
    container.appendChild(chip);
  }
}

function render(allRows, agents, visible) {
  const rows = allRows.filter((d) => visible.has(d.agent));
  const difficultyNote =
    "Difficulty breakdown not available — supply --difficulty to analysis/prepare_dashboard_data.py.";
  const pending = []; // {div, spec} — plotted only after every figure is in the DOM

  const accuracyEl = el("accuracy-charts");
  clear(accuracyEl);
  mountPair(
    accuracyEl,
    accuracyCharts(rows, agents),
    { overallTitle: "Accuracy (overall)", breakdownTitle: "Accuracy by difficulty", note: difficultyNote },
    pending
  );

  const completionEl = el("completion-charts");
  clear(completionEl);
  mountPair(
    completionEl,
    completionCharts(rows, agents),
    {
      overallTitle: "Completion rate (overall)",
      breakdownTitle: "Completion rate by difficulty",
      note: difficultyNote,
    },
    pending
  );

  const costEl = el("cost-charts");
  clear(costEl);
  if (hasCost(rows)) {
    const cost = costCharts(rows, agents);
    mountFigure(costEl, cost.duration, { title: "Wall-clock duration" }, pending);
    mountFigure(costEl, cost.inputTokens, { title: "Input tokens" }, pending);
    mountFigure(costEl, cost.outputTokens, { title: "Output tokens" }, pending);
  } else {
    mountFigure(
      costEl,
      null,
      { note: "Cost data not available — supply --cost to analysis/prepare_dashboard_data.py." },
      pending
    );
  }

  const compositionEl = el("composition-charts");
  clear(compositionEl);
  if (hasComposition(rows)) {
    const composition = compositionCharts(rows);
    mountFigure(compositionEl, composition.domainBar, { title: "Tasks by domain" }, pending);
  } else {
    mountFigure(
      compositionEl,
      null,
      { note: "Task composition not available — supply --task-catalog to analysis/prepare_dashboard_data.py." },
      pending
    );
  }

  // Second pass: every figure element for every section is already in its final
  // place in the DOM (and the flex/grid layout has nothing left to shift), so it's
  // now safe to plot each one without a later sibling's insertion perturbing it.
  for (const { div, spec } of pending) {
    Plotly.newPlot(div, spec.data, spec.layout, PLOTLY_CONFIG);
  }
}

async function main() {
  let rows;
  try {
    rows = await d3.csv(DATA_URL, d3.autoType);
  } catch (err) {
    renderError(
      `Could not load ${DATA_URL}: ${err.message}. Run analysis/prepare_dashboard_data.py to generate it.`
    );
    return;
  }
  if (!rows || rows.length === 0) {
    renderError(`${DATA_URL} loaded but contains no rows.`);
    return;
  }

  const agents = [...new Set(rows.map((d) => d.agent))];
  const visible = new Set(agents);

  function onToggle(agent) {
    if (visible.has(agent)) {
      if (visible.size === 1) return; // keep at least one agent visible
      visible.delete(agent);
    } else {
      visible.add(agent);
    }
    buildAgentPicker(agents, visible, onToggle);
    render(rows, agents, visible);
  }

  buildAgentPicker(agents, visible, onToggle);
  render(rows, agents, visible);

  // Plotly resolves colors to literal hex at chart-build time (see agents.js), so a
  // light/dark switch needs an explicit re-render to pick up the new CSS values.
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    render(rows, agents, visible);
  });
}

main();
