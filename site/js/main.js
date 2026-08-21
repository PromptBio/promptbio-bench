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

const DATA_URL = "data/results.csv";

const el = (id) => document.getElementById(id);

function renderError(message) {
  const banner = el("error-banner");
  banner.textContent = message;
  banner.hidden = false;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function mountFigure(container, node, { title, note } = {}) {
  const figure = document.createElement("figure");
  if (title) {
    const caption = document.createElement("figcaption");
    caption.textContent = title;
    figure.appendChild(caption);
  }
  if (node) {
    figure.appendChild(node);
  } else if (note) {
    const p = document.createElement("p");
    p.className = "placeholder";
    p.textContent = note;
    figure.appendChild(p);
  }
  container.appendChild(figure);
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

  const accuracyEl = el("accuracy-charts");
  clear(accuracyEl);
  const difficultyNote =
    "Difficulty breakdown not available — supply --difficulty to analysis/prepare_dashboard_data.py.";
  const acc = accuracyCharts(rows, agents);
  mountFigure(accuracyEl, acc.overall, { title: "Accuracy (overall)" });
  mountFigure(accuracyEl, acc.byDifficulty, {
    title: "Accuracy by difficulty",
    note: difficultyNote,
  });

  const completionEl = el("completion-charts");
  clear(completionEl);
  const comp = completionCharts(rows, agents);
  mountFigure(completionEl, comp.overall, { title: "Completion rate (overall)" });
  mountFigure(completionEl, comp.byDifficulty, {
    title: "Completion rate by difficulty",
    note: difficultyNote,
  });

  const costEl = el("cost-charts");
  clear(costEl);
  if (hasCost(rows)) {
    const cost = costCharts(rows, agents);
    mountFigure(costEl, cost.duration, { title: "Wall-clock duration" });
    mountFigure(costEl, cost.inputTokens, { title: "Input tokens" });
    mountFigure(costEl, cost.outputTokens, { title: "Output tokens" });
  } else {
    mountFigure(costEl, null, {
      note: "Cost data not available — supply --cost to analysis/prepare_dashboard_data.py.",
    });
  }

  const compositionEl = el("composition-charts");
  clear(compositionEl);
  if (hasComposition(rows)) {
    const composition = compositionCharts(rows);
    mountFigure(compositionEl, composition.domainBar, { title: "Tasks by domain" });
    mountFigure(compositionEl, composition.fieldHeatmap, { title: "Tasks by field × difficulty" });
  } else {
    mountFigure(compositionEl, null, {
      note: "Task composition not available — supply --task-catalog to analysis/prepare_dashboard_data.py.",
    });
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
}

main();
