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
import { initThemeToggle } from "./theme.js";

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

function makeFigureHeader(title) {
  const caption = document.createElement("figcaption");
  if (title) {
    const span = document.createElement("span");
    span.textContent = title;
    caption.appendChild(span);
  }
  return caption;
}

// spec is a {data, layout} Plotly spec, or null (renders `note` instead). Building
// figures is split from plotting them: `pending` collects {div, spec} and every figure
// across every section is appended to the DOM first, with all Plotly.newPlot calls run
// afterward in a second pass. Interleaving DOM insertion with newPlot — appending a
// later sibling into the same flex/grid row while an earlier chart's responsive
// ResizeObserver was still settling — was corrupting the earlier chart's axis/traces
// even though its own spec was correct in isolation.
function mountFigure(container, spec, { title, note } = {}, pending) {
  const figure = document.createElement("figure");
  const header = makeFigureHeader(title);
  figure.appendChild(header);
  container.appendChild(figure);

  if (spec) {
    const plotDiv = document.createElement("div");
    figure.appendChild(plotDiv);
    pending.push({ div: plotDiv, spec });

    // Reset button — added after plotDiv so it's available in closure
    const btn = document.createElement("button");
    btn.className = "btn-reset";
    btn.title = "Reset view";
    btn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
      <path d="M3 3v5h5"/>
    </svg>`;
    btn.addEventListener("click", () => {
      Plotly.relayout(plotDiv, { "xaxis.autorange": true, "yaxis.autorange": true });
    });
    header.appendChild(btn);
  } else if (note) {
    const p = document.createElement("p");
    p.className = "placeholder";
    p.textContent = note;
    figure.appendChild(p);
  }
}

// Domain bar with a toggle button to switch between plain (grey) and
// difficulty-stratified (stacked blue) views. The legend is rendered as HTML
// inline in the header so both chart variants stay the same size.
function mountDomainBar(container, { domainBar, domainBarStratified }, pending) {
  const figure = document.createElement("figure");
  const header = makeFigureHeader("Tasks by domain");
  header.classList.add("domain-bar-header");
  figure.appendChild(header);
  container.appendChild(figure);

  // Inline HTML legend — absolutely centered in the header
  const legend = document.createElement("span");
  legend.className = "difficulty-legend";
  legend.style.display = "none";
  domainBarStratified.difficultyLevels.forEach((level) => {
    const item = document.createElement("span");
    item.className = "difficulty-legend-item";
    const swatch = document.createElement("span");
    swatch.className = "difficulty-legend-swatch";
    swatch.style.background = domainBarStratified.difficultyColors[level];
    const label = document.createElement("span");
    label.textContent = level.charAt(0).toUpperCase() + level.slice(1);
    item.appendChild(swatch);
    item.appendChild(label);
    legend.appendChild(item);
  });
  header.appendChild(legend);

  // Two plot divs — only one visible at a time.
  // stratDiv is lazy-rendered on first show so Plotly gets real pixel dimensions.
  const plainDiv = document.createElement("div");
  const stratDiv = document.createElement("div");
  stratDiv.style.display = "none";
  figure.appendChild(plainDiv);
  figure.appendChild(stratDiv);

  pending.push({ div: plainDiv, spec: domainBar });

  // Reset button — returns to the default plain view
  const resetBtn = document.createElement("button");
  resetBtn.className = "btn-reset";
  resetBtn.title = "Reset view";
  resetBtn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
    <path d="M3 3v5h5"/>
  </svg>`;
  resetBtn.addEventListener("click", () => {
    stratified = false;
    plainDiv.style.display = "";
    stratDiv.style.display = "none";
    legend.style.display = "none";
    toggleBtn.classList.remove("active");
    Plotly.relayout(plainDiv, { "xaxis.autorange": true, "yaxis.autorange": true });
  });

  // Toggle button
  const toggleBtn = document.createElement("button");
  toggleBtn.className = "btn-difficulty-toggle";
  toggleBtn.textContent = "By difficulty";

  let stratified = false;
  let stratRendered = false;
  toggleBtn.addEventListener("click", () => {
    stratified = !stratified;
    plainDiv.style.display = stratified ? "none" : "";
    stratDiv.style.display = stratified ? "" : "none";
    legend.style.display = stratified ? "" : "none";
    toggleBtn.classList.toggle("active", stratified);
    if (stratified) {
      if (!stratRendered) {
        // First show: render now that the div has real dimensions
        Plotly.newPlot(stratDiv, domainBarStratified.data, domainBarStratified.layout, PLOTLY_CONFIG);
        stratRendered = true;
      } else {
        Plotly.relayout(stratDiv, { "xaxis.autorange": true, "yaxis.autorange": true });
      }
    } else {
      Plotly.relayout(plainDiv, { "xaxis.autorange": true, "yaxis.autorange": true });
    }
  });
  const actions = document.createElement("div");
  actions.className = "chart-header-actions";
  actions.appendChild(toggleBtn);
  actions.appendChild(resetBtn);
  header.appendChild(actions);
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
    "Difficulty breakdown not available — add a difficulty column to site/data/results.csv.";
  const pending = []; // {div, spec} — plotted only after every figure is in the DOM

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

  const accuracyEl = el("accuracy-charts");
  clear(accuracyEl);
  mountPair(
    accuracyEl,
    accuracyCharts(rows, agents),
    { overallTitle: "Accuracy (overall)", breakdownTitle: "Accuracy by difficulty", note: difficultyNote },
    pending
  );

  const costEl = el("cost-charts");
  clear(costEl);
  if (hasCost(rows)) {
    const cost = costCharts(rows, agents);
    mountFigure(costEl, cost.duration, { title: "Computing time" }, pending);
    mountFigure(costEl, cost.inputTokens, { title: "Input tokens" }, pending);
    mountFigure(costEl, cost.outputTokens, { title: "Output tokens" }, pending);
  } else {
    mountFigure(
      costEl,
      null,
      { note: "Cost data not available — add duration_seconds/input_tokens/output_tokens columns to site/data/results.csv." },
      pending
    );
  }

  const compositionEl = el("composition-charts");
  clear(compositionEl);
  if (hasComposition(rows)) {
    const composition = compositionCharts(rows);
    mountDomainBar(compositionEl, composition, pending);
  } else {
    mountFigure(
      compositionEl,
      null,
      { note: "Task composition not available — add a domain column to site/data/results.csv." },
      pending
    );
  }

  // Second pass: every figure element for every section is already in its final
  // place in the DOM (and the flex/grid layout has nothing left to shift), so it's
  // now safe to plot each one without a later sibling's insertion perturbing it.
  for (const { div, spec } of pending) {
    Plotly.newPlot(div, spec.data, spec.layout, PLOTLY_CONFIG);
    div._defaultLayout = spec.layout;
  }
}

async function main() {
  let rows;
  try {
    rows = await d3.csv(DATA_URL, d3.autoType);
  } catch (err) {
    renderError(
      `Could not load ${DATA_URL}: ${err.message}. Make sure site/data/results.csv exists.`
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

  initThemeToggle(() => render(rows, agents, visible));
}

main();
