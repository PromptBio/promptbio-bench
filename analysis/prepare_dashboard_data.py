#!/usr/bin/env python3
"""Build the promptbio-bench dashboard's data file from existing analysis tables.

Reads the tables the analysis notebooks already produce (agent_eval, an optional
task catalog, an optional difficulty classification, an optional cost table) and
writes one flat, long-format CSV — one row per (task, agent) — to site/data/.
That CSV is the only runtime dependency of the dashboard page (site/index.html).

Every input may be .csv, .tsv, .xlsx, or .xls, independently of one another.

Usage:
    python analysis/prepare_dashboard_data.py \
        --agent-eval  /path/to/agent_eval.xlsx \
        [--task-catalog  /path/to/Questions.xlsx] \
        [--difficulty    /path/to/task_difficulty_classification.xlsx] \
        [--cost          /path/to/results_latest.csv] \
        [--skip          /path/to/task_todo.tsv] \
        --out site/data/results.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

EQUIVALENCE_THRESHOLD = 0.5

OUTPUT_COLUMNS = [
    "id",
    "domain",
    "field",
    "difficulty",
    "agent",
    "avg_similarity",
    "equivalent",
    "completion",
    "duration_seconds",
    "input_tokens",
    "output_tokens",
]


def read_table(path: Path) -> pd.DataFrame:
    """Load a .csv, .tsv, .xlsx, or .xls file into a DataFrame based on its suffix."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type {suffix!r} for {path} (expected .csv, .tsv, .xlsx, or .xls)")


def _find_column(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Case-insensitive lookup of the first matching column name present in df."""
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        col = lower_map.get(candidate.lower())
        if col is not None:
            return col
    return None


def load_agent_eval(path: Path) -> pd.DataFrame:
    """One row per (task, reference file, agent) -> collapse to one row per (id, agent).

    avg_similarity is already repeated per reference-file row, so take the first.
    completion mirrors the notebooks' `all_candidate_paths_exist`: every reference
    file for that (id, agent) must have a non-null, non-empty candidate_path.
    equivalent = avg_similarity is not null and >= EQUIVALENCE_THRESHOLD.

    Row order (first appearance of each (id, agent) pair) is preserved, not
    re-sorted, so agent color assignment stays stable across regenerations.
    """
    df = read_table(path)
    required = {"id", "agent", "avg_similarity", "candidate_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"--agent-eval is missing required column(s): {sorted(missing)}")

    def _has_candidate(v) -> bool:
        return bool(v) and not pd.isna(v)

    order = df[["id", "agent"]].drop_duplicates()

    grouped = df.groupby(["id", "agent"], sort=False).agg(
        avg_similarity=("avg_similarity", "first"),
    )
    completion = df.groupby(["id", "agent"], sort=False)["candidate_path"].apply(
        lambda s: bool(s.map(_has_candidate).all())
    )
    grouped["completion"] = completion
    grouped = grouped.reset_index()

    out = order.merge(grouped, on=["id", "agent"], how="left")
    out["equivalent"] = out["avg_similarity"].apply(
        lambda v: bool(pd.notna(v) and v >= EQUIVALENCE_THRESHOLD)
    )
    return out[["id", "agent", "avg_similarity", "equivalent", "completion"]]


def load_task_catalog(path: Optional[Path]) -> pd.DataFrame:
    """id/domain/field only, matched case-insensitively. Empty frame if not given."""
    if path is None:
        return pd.DataFrame(columns=["id", "domain", "field"])
    df = read_table(path)
    id_col = _find_column(df, "id")
    domain_col = _find_column(df, "domain")
    field_col = _find_column(df, "field")
    if id_col is None:
        raise ValueError(f"--task-catalog ({path}) has no id/ID column")
    out = pd.DataFrame({"id": df[id_col]})
    out["domain"] = df[domain_col] if domain_col else None
    out["field"] = df[field_col] if field_col else None
    return out


def load_difficulty(path: Optional[Path]) -> pd.DataFrame:
    """id/difficulty only, matched case-insensitively. Empty frame if not given."""
    if path is None:
        return pd.DataFrame(columns=["id", "difficulty"])
    df = read_table(path)
    id_col = _find_column(df, "id")
    difficulty_col = _find_column(df, "difficulty")
    if id_col is None:
        raise ValueError(f"--difficulty ({path}) has no id/ID column")
    if difficulty_col is None:
        raise ValueError(f"--difficulty ({path}) has no difficulty/Difficulty column")
    out = pd.DataFrame({"id": df[id_col], "difficulty": df[difficulty_col]})
    out["difficulty"] = out["difficulty"].astype(str).str.strip().str.lower()
    return out


def merge_cost(df: pd.DataFrame, cost_path: Optional[Path]) -> pd.DataFrame:
    """Left-join duration_seconds/input_tokens/output_tokens onto (id, agent).

    Accepts either (id, agent) or (task_id, agent_name) column names.
    Unmatched (id, agent) rows keep all three cost fields null.
    """
    if cost_path is None:
        for col in ("duration_seconds", "input_tokens", "output_tokens"):
            df[col] = None
        return df

    cost = read_table(cost_path)
    id_col = _find_column(cost, "id", "task_id")
    agent_col = _find_column(cost, "agent", "agent_name")
    if id_col is None or agent_col is None:
        raise ValueError(f"--cost ({cost_path}) needs id/task_id and agent/agent_name columns")

    cost_cols = {"id": cost[id_col], "agent": cost[agent_col]}
    for col in ("duration_seconds", "input_tokens", "output_tokens"):
        cost_cols[col] = cost[col] if col in cost.columns else None
    cost_slim = pd.DataFrame(cost_cols)
    # A (task, agent) pair may legitimately have multiple runs; keep the last.
    cost_slim = cost_slim.drop_duplicates(subset=["id", "agent"], keep="last")

    return df.merge(cost_slim, on=["id", "agent"], how="left")


def apply_skip_list(df: pd.DataFrame, skip_path: Optional[Path]) -> pd.DataFrame:
    if skip_path is None:
        return df
    skip = read_table(skip_path)
    id_col = _find_column(skip, "id")
    if id_col is None:
        raise ValueError(f"--skip ({skip_path}) has no id/ID column")
    skip_ids = set(skip[id_col].astype(str))
    return df[~df["id"].astype(str).isin(skip_ids)]


def build(
    agent_eval_path: Path,
    task_catalog_path: Optional[Path],
    difficulty_path: Optional[Path],
    cost_path: Optional[Path],
    skip_path: Optional[Path],
) -> pd.DataFrame:
    df = load_agent_eval(agent_eval_path)
    df = apply_skip_list(df, skip_path)

    catalog = load_task_catalog(task_catalog_path)
    df = df.merge(catalog, on="id", how="left")

    difficulty = load_difficulty(difficulty_path)
    df = df.merge(difficulty, on="id", how="left")

    df = merge_cost(df, cost_path)

    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[OUTPUT_COLUMNS]


def write_results_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path = out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    # d3.autoType (used by the dashboard) only recognizes lowercase "true"/"false";
    # pandas' default bool-to-string is "True"/"False".
    for col in ("equivalent", "completion"):
        df[col] = df[col].map(lambda v: "true" if bool(v) else "false")
    df.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent-eval", required=True, type=Path, help="Per-(task, file, agent) eval table (csv/tsv/xlsx)")
    parser.add_argument("--task-catalog", type=Path, default=None, help="Task catalog with id/domain/field (csv/tsv/xlsx)")
    parser.add_argument("--difficulty", type=Path, default=None, help="Task difficulty classification with id/difficulty (csv/tsv/xlsx)")
    parser.add_argument("--cost", type=Path, default=None, help="Per-(task, agent) cost/timing table (csv/tsv/xlsx)")
    parser.add_argument("--skip", type=Path, default=None, help="List of task ids to exclude (csv/tsv/xlsx)")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV path, e.g. site/data/results.csv")
    args = parser.parse_args()

    for label, path in [
        ("--agent-eval", args.agent_eval),
        ("--task-catalog", args.task_catalog),
        ("--difficulty", args.difficulty),
        ("--cost", args.cost),
        ("--skip", args.skip),
    ]:
        if path is not None and not path.exists():
            sys.exit(f"❌ {label} path does not exist: {path}")

    df = build(args.agent_eval, args.task_catalog, args.difficulty, args.cost, args.skip)
    write_results_csv(df, args.out)

    n_tasks = df["id"].nunique()
    n_agents = df["agent"].nunique()
    print(f"✅ Wrote {len(df)} rows ({n_tasks} tasks x {n_agents} agents) → {args.out}")
    if df["domain"].isna().all():
        print("   (no --task-catalog given: domain/field columns are empty)")
    if df["difficulty"].isna().all():
        print("   (no --difficulty given: difficulty column is empty)")
    if df["duration_seconds"].isna().all():
        print("   (no --cost given: cost columns are empty)")


if __name__ == "__main__":
    main()
