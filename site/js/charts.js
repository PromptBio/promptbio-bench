import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";
import { agentColor, getTheme } from "./agents.js";

const DIFFICULTY_ORDER = ["low", "medium", "high"];
const FONT_FAMILY = "system-ui, -apple-system, 'Segoe UI', sans-serif";

function mean01(values) {
  const m = d3.mean(values, (v) => (v ? 1 : 0));
  return m == null ? null : m;
}

function baseLayout(theme, overrides = {}) {
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: FONT_FAMILY, color: theme.textSecondary, size: 13 },
    margin: { l: 48, r: 16, t: 12, b: 40 },
    showlegend: false,
    hoverlabel: {
      bgcolor: theme.surface,
      bordercolor: theme.axis,
      font: { family: FONT_FAMILY, color: theme.textPrimary },
    },
    ...overrides,
  };
}

function rateAxis(theme, label) {
  return {
    range: [0, 1.08],
    tickformat: ".0%",
    title: { text: label, font: { size: 12, color: theme.textSecondary } },
    gridcolor: theme.gridline,
    zeroline: false,
    tickfont: { color: theme.muted },
  };
}

// ---- Accuracy / completion: one bar per agent overall, grouped-by-difficulty ----

function overallRateChart(rows, agents, { yField, yLabel }, theme) {
  const agg = Array.from(
    d3.rollup(rows, (v) => mean01(v.map((d) => d[yField])), (d) => d.agent),
    ([agent, value]) => ({ agent, value })
  ).filter((d) => d.value != null);
  const xs = agg.map((d) => d.agent);

  const data = [
    {
      type: "bar",
      x: xs,
      y: agg.map((d) => d.value),
      marker: { color: xs.map((a) => agentColor(a, agents)) },
      text: agg.map((d) => `${(d.value * 100).toFixed(0)}%`),
      textposition: "outside",
      textfont: { color: theme.textSecondary },
      hovertemplate: "%{x}: %{y:.0%}<extra></extra>",
    },
  ];
  const layout = baseLayout(theme, {
    height: 300,
    xaxis: { tickfont: { color: theme.textPrimary } },
    yaxis: rateAxis(theme, yLabel),
  });
  return { data, layout };
}

function byDifficultyRateChart(rows, agents, { yField, yLabel }, theme) {
  const data = rows.filter((d) => d.difficulty != null);
  if (data.length === 0) return null;

  const key = (d) => `${d.difficulty}${d.agent}`;
  const agg = new Map(
    Array.from(d3.rollup(data, (v) => mean01(v.map((d) => d[yField])), key))
  );
  const difficulties = DIFFICULTY_ORDER.filter((diff) =>
    agents.some((a) => agg.get(`${diff}${a}`) != null)
  );
  if (difficulties.length === 0) return null;

  const traces = agents
    .filter((agent) => difficulties.some((diff) => agg.get(`${diff}${agent}`) != null))
    .map((agent) => {
      const ys = difficulties.map((diff) => agg.get(`${diff}${agent}`) ?? null);
      return {
        type: "bar",
        name: agent,
        x: difficulties,
        y: ys,
        marker: { color: agentColor(agent, agents) },
        text: ys.map((v) => (v == null ? "" : `${(v * 100).toFixed(0)}%`)),
        textposition: "outside",
        textfont: { color: theme.textSecondary },
        hovertemplate: `${agent}: %{y:.0%}<extra></extra>`,
      };
    });

  const layout = baseLayout(theme, {
    height: 300,
    barmode: "group",
    bargap: 0.28,
    bargroupgap: 0.1,
    xaxis: { tickfont: { color: theme.textPrimary } },
    yaxis: rateAxis(theme, yLabel),
  });
  return { data: traces, layout };
}

export function accuracyCharts(rows, agents) {
  const theme = getTheme();
  return {
    overall: overallRateChart(rows, agents, { yField: "equivalent", yLabel: "Accuracy" }, theme),
    byDifficulty: byDifficultyRateChart(
      rows,
      agents,
      { yField: "equivalent", yLabel: "Accuracy" },
      theme
    ),
  };
}

export function completionCharts(rows, agents) {
  const theme = getTheme();
  return {
    overall: overallRateChart(
      rows,
      agents,
      { yField: "completion", yLabel: "Completion rate" },
      theme
    ),
    byDifficulty: byDifficultyRateChart(
      rows,
      agents,
      { yField: "completion", yLabel: "Completion rate" },
      theme
    ),
  };
}

// ---- Cost: one box plot per metric, one box per agent ----

export function hasCost(rows) {
  return rows.some((d) => d.duration_seconds != null);
}

function boxChart(rows, agents, field, yLabel, theme) {
  const present = agents.filter((a) => rows.some((d) => d.agent === a && d[field] != null));
  if (present.length === 0) return null;

  // Explicit numeric x0 per trace + manual tickvals/ticktext, rather than a
  // string-categorical x-axis — Plotly's categorical-axis positioning for
  // one-trace-per-category box plots was misordering/duplicating ticks
  // (each trace's own x values are internally consistent; the axis layer
  // was where positions got scrambled), so position by index instead and
  // let the axis just be a plain numeric one with custom labels.
  const data = present.map((agent, i) => {
    const pts = rows.filter((d) => d.agent === agent && d[field] != null);
    return {
      type: "box",
      name: agent,
      x0: i,
      y: pts.map((d) => d[field]),
      text: pts.map((d) => d.id),
      width: 0.5,
      marker: { color: agentColor(agent, agents) },
      line: { color: agentColor(agent, agents) },
      fillcolor: agentColor(agent, agents),
      opacity: 0.75,
      boxpoints: "outliers",
      hovertemplate: `${agent}<br>%{y}<br><span style="color:#888">%{text}</span><extra></extra>`,
    };
  });

  const layout = baseLayout(theme, {
    height: 300,
    xaxis: {
      range: [-0.6, present.length - 0.4],
      tickmode: "array",
      tickvals: present.map((_, i) => i),
      ticktext: present,
      tickfont: { color: theme.textPrimary },
    },
    yaxis: {
      title: { text: yLabel, font: { size: 12, color: theme.textSecondary } },
      gridcolor: theme.gridline,
      zeroline: false,
      tickfont: { color: theme.muted },
    },
  });
  return { data, layout };
}

export function costCharts(rows, agents) {
  const theme = getTheme();
  return {
    duration: boxChart(rows, agents, "duration_seconds", "Duration (seconds)", theme),
    inputTokens: boxChart(rows, agents, "input_tokens", "Input tokens", theme),
    outputTokens: boxChart(rows, agents, "output_tokens", "Output tokens", theme),
  };
}

// ---- Task composition: domain counts only ----

function uniqueTasks(rows) {
  const seen = new Map();
  for (const d of rows) {
    if (!seen.has(d.id)) seen.set(d.id, { id: d.id, domain: d.domain });
  }
  return Array.from(seen.values());
}

export function hasComposition(rows) {
  return rows.some((d) => d.domain != null);
}

export function compositionCharts(rows) {
  const theme = getTheme();
  const tasks = uniqueTasks(rows).filter((d) => d.domain != null);
  const counts = Array.from(
    d3.rollup(tasks, (v) => v.length, (d) => d.domain),
    ([domain, count]) => ({ domain, count })
  ).sort((a, b) => d3.descending(a.domain, b.domain));

  const data = [
    {
      type: "bar",
      orientation: "h",
      y: counts.map((d) => d.domain),
      x: counts.map((d) => d.count),
      // Neutral, not an agent color — this chart isn't about agent identity.
      marker: { color: theme.muted },
      text: counts.map((d) => d.count),
      textposition: "outside",
      textfont: { color: theme.textSecondary },
      hovertemplate: "%{y}: %{x} tasks<extra></extra>",
    },
  ];
  const layout = baseLayout(theme, {
    height: Math.max(140, counts.length * 60),
    margin: { l: Math.max(80, d3.max(counts, (d) => d.domain.length) * 7), r: 40, t: 12, b: 40 },
    xaxis: {
      title: { text: "Task count", font: { size: 12, color: theme.textSecondary } },
      gridcolor: theme.gridline,
      zeroline: false,
      tickfont: { color: theme.muted },
    },
    yaxis: { tickfont: { color: theme.textPrimary } },
  });
  return { domainBar: { data, layout } };
}
