#!/usr/bin/env python3
"""Evaluate one agent result directory against a task's reference answer.

Runs a 4-step pipeline:
  1. Match      — map agent output files to reference files via LLM
  2. Detect     — identify each reference file's format
  3. Recommend  — choose a comparison strategy per file
  4. Compare    — compute per-file similarity scores

Inputs:
  eval.json   — task spec: id, question, ref_answer (list of relative paths),
                scoring.expected_output[].guidelines (optional per-file hints)
  task.json   — task context: input_files (list of relative paths; optional)
  <result-dir> — directory containing the agent's output files to evaluate

Usage:
    python run_eval.py --task-dir <path> --result-dir <path> --output-dir <path> [--label <name>] [--model <model>]

Example:
    python run_eval.py \
        --task-dir   /mnt/data/vincent/promptbio-bench/tasks/a-1-1 \
        --result-dir /mnt/data/lengyang/youjia_project/autoba/BABench/src/promptbio-bench/tasks/a-1-1/result_2/biomni_20260429 \
        --output-dir /home/vincent/project/promptbio-bench/data/eval/ \
        --label biomni_20260429

Output (written to --output-dir):
    <task_id>_<label>.log    full pipeline log (stdout tee'd to file)
    <task_id>_<label>.json   structured per-file comparison results
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.agents import load_api
from utils.logger import log_context
from utils.match import match_output_files
from utils.format import detect_file_signature
from utils.strategy import StrategyRecommender
from utils.models import EvalFile, FileEvalState
from utils.compare import ComparisonRunner


_IGNORE_FILES = {"docker_log.txt", "data_description.json", "session_init", "work"}
_IGNORE_DIRS  = {"session_init", "work"}


def _files_in_dir(directory: Path, exclude: set[str]) -> list[str]:
    """Return files directly inside directory, excluding names in exclude."""
    return [
        str(p) for p in sorted(directory.iterdir())
        if p.is_file() and p.name not in exclude
    ]


def _files_in_subdirs(candidate_dir: Path, exclude: set[str], skip_dirs: set[str]) -> list[str]:
    """Return files one level deep inside subdirectories of candidate_dir.

    Only looks at immediate children of each subdir (not recursive).
    Skips subdirs whose name is in skip_dirs.
    """
    results = []
    for subdir in sorted(candidate_dir.iterdir()):
        if not subdir.is_dir() or subdir.name in skip_dirs:
            continue
        results.extend(
            str(p) for p in sorted(subdir.iterdir())
            if p.is_file() and p.name not in exclude
        )
    return results


def _run_llm_match(
    states: list[FileEvalState],
    candidates: list[str],
    question: str,
    input_files: list[str],
    model: str,
    label: str,
) -> tuple[list[FileEvalState], bool]:
    """Run LLM file matching against a specific candidate list.

    Returns (updated_states, any_matched) where any_matched is True if at
    least one reference file was mapped to a non-null candidate.
    Raises RuntimeError if the LLM call itself fails.
    """
    print(f"Searching in: {label}  ({len(candidates)} file(s))")
    mapping = match_output_files(
        reference_files=[s.reference_file for s in states],
        candidate_files=candidates,
        question=question,
        input_files=input_files or None,
        model=model,
    )
    if mapping is None:
        raise RuntimeError("File matching failed")

    by_ref = {m.reference_file: m for m in mapping.mappings}
    print("Matched files:")
    updated = []
    any_matched = False
    for s in states:
        m = by_ref.get(s.reference_file)
        if m:
            matched = m.candidate_file is not None
            if matched:
                any_matched = True
            label_str = Path(m.candidate_file).name if matched else "NO MATCH"
            print(f"  {Path(m.reference_file).name} → {label_str}  (conf={m.confidence:.2f})")
            print(f"    Reasoning: {m.reasoning}")
            updated.append(s.model_copy(update={
                "candidate_file": m.candidate_file,
                "match_confidence": m.confidence,
                "match_reasoning": m.reasoning,
            }))
        else:
            print(f"  {Path(s.reference_file).name} → NO MATCH  (not returned by LLM)")
            updated.append(s)
    return updated, any_matched


def _match_output(states: list[FileEvalState], candidate_dir: Path, question: str, input_files: list[str], model: str) -> list[FileEvalState]:
    """Try to match outputs using a three-tier search strategy:
      1. Files directly in candidate_dir
      2. Files directly in candidate_dir/data/
      3. Files one level deep in all other subdirectories

    Falls through to the next tier if the current tier has no files OR the
    LLM returns no matches (all null candidates) — handling cases where
    agents leave only irrelevant artifacts at the top level.
    Input file basenames are excluded at every tier.
    Raises ValueError if no candidate files are found across all tiers.
    Raises RuntimeError if the LLM matching call fails.
    """
    input_basenames = {Path(f).name for f in input_files} if input_files else set()
    exclude = input_basenames | _IGNORE_FILES
    found_any_files = False

    # Tier 1: top-level files
    candidates = _files_in_dir(candidate_dir, exclude)
    if candidates:
        found_any_files = True
        updated, matched = _run_llm_match(states, candidates, question, input_files, model, str(candidate_dir))
        if matched:
            return updated
        print("No matches found at top level, trying data/ subfolder...")

    # Tier 2: data/ subdir
    data_dir = candidate_dir / "data"
    if data_dir.is_dir():
        candidates = _files_in_dir(data_dir, exclude)
        if candidates:
            found_any_files = True
            updated, matched = _run_llm_match(states, candidates, question, input_files, model, str(data_dir))
            if matched:
                return updated
            print("No matches found in data/, trying subdirectories...")

    # Tier 3: one level inside all other subdirectories
    skip = _IGNORE_DIRS | {"data"}
    candidates = _files_in_subdirs(candidate_dir, exclude, skip_dirs=skip)
    if candidates:
        found_any_files = True
        updated, matched = _run_llm_match(states, candidates, question, input_files, model, f"subdirs of {candidate_dir}")
        return updated  # return regardless — this is the last resort

    if not found_any_files:
        raise ValueError("No candidate files found")

    # Files were found at earlier tiers but no LLM match — return last attempted states
    return updated


def _detect_format(states: list[FileEvalState], model: str) -> list[FileEvalState]:
    updated = []
    for s in states:
        sig = detect_file_signature(s.reference_file, fallback="llm", model=model)
        print(f"  {Path(s.reference_file).name}:")
        print(f"    format:      {sig.format}")
        print(f"    compression: {sig.compression}")
        print(f"    is_text:     {sig.is_text}")
        print(f"    method:      {sig.method}")
        if sig.note:
            print(f"    note:        {sig.note}")
        updated.append(s.model_copy(update={"file_format": sig.format, "signature": sig}))
    return updated


def _recommend_strategy(states: list[FileEvalState], question: str, input_files: list[str], tool_schema_path: Path, model: str) -> list[FileEvalState]:
    recommender = StrategyRecommender(tool_schema_path, model=model)
    eval_files = [
        EvalFile(
            reference_file=s.reference_file,
            candidate_file=s.candidate_file,
            file_format=s.file_format or "unknown",
            eval_guideline=s.eval_guideline,
        )
        for s in states
    ]
    recs = recommender.recommend(
        question=question,
        input_files=input_files or None,
        eval_files=eval_files,
        verbose=True,
    )
    updated = []
    for s, rec in zip(states, recs):
        if not rec.error:
            print(f"    parameters: {rec.parameters}")
            print(f"    reasoning:  {rec.reasoning}")
        else:
            print(f"    ⚠️  recommendation failed: {rec.error}")
        updated.append(s.model_copy(update={
            "strategy": rec.strategy if not rec.error else "unknown",
            "parameters": rec.parameters if not rec.error else {},
            "strategy_confidence": rec.confidence,
            "strategy_reasoning": rec.reasoning,
            "strategy_error": rec.error,
        }))
    return updated


def _compare_files(states: list[FileEvalState], question: str, model: str, task_id: Optional[str] = None) -> list[FileEvalState]:
    eval_files = [
        EvalFile(
            reference_file=s.reference_file,
            candidate_file=s.candidate_file,
            file_format=s.file_format or "unknown",
            eval_guideline=s.eval_guideline,
            strategy=s.strategy,
            parameters=s.parameters,
            signature=s.signature,
        )
        for s in states
    ]
    for i, eval_file in enumerate(eval_files, 1):
        print(f"\n  File {i}/{len(eval_files)}: {eval_file.file_format}, strategy={eval_file.strategy}")
        print(f"    ref: {eval_file.reference_file}")
        print(f"    alt: {eval_file.candidate_file if eval_file.candidate_file else 'NO MATCH'}")

    comparator = ComparisonRunner(model=model)
    summary = comparator.compare_files(eval_files, question=question, task_id=task_id)

    if summary.error and not summary.comparisons:
        return [
            s.model_copy(update={"comparison_status": "error", "comparison_error": summary.error})
            for s in states
        ]

    updated = []
    for i, (s, r) in enumerate(zip(states, summary.comparisons), 1):
        sim = f"{r.similarity:.4f}" if not math.isnan(r.similarity) else "NaN"
        print(f"\n  File {i}/{len(summary.comparisons)}: status={r.status}, similarity={sim}")
        if r.details:
            print("    details:")
            for line in json.dumps(_nan_to_none(r.details), indent=2).splitlines():
                print(f"      {line}")
        if r.error:
            print(f"    error: {r.error}")
        updated.append(s.model_copy(update={
            "similarity": r.similarity,
            "comparison_status": r.status,
            "comparison_details": _nan_to_none(r.details),
            "comparison_error": r.error,
        }))
    return updated


def _nan_to_none(obj):
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    # numpy scalar types are not JSON serializable
    if hasattr(obj, "item"):
        return _nan_to_none(obj.item())
    return obj


def _save_results(json_path: Path, full_json_path: Path, task_id: str, question: str, states: list[FileEvalState]) -> None:
    scored = [s for s in states if s.comparison_status in ("success", "invalid")]
    avg = sum(s.similarity for s in scored) / len(scored) if scored else float("nan")
    result = {
        "id": task_id,
        "avg_similarity": None if math.isnan(avg) else avg,
        "error": None if scored else "All comparisons skipped or failed",
        "files": [
            {
                "reference_path": s.reference_file,
                "candidate_path": s.candidate_file,
                "file_format": s.file_format,
                "strategy": s.strategy,
                "parameters": s.parameters,
                "similarity": None if math.isnan(s.similarity) else s.similarity,
                "status": s.comparison_status,
                "error": s.comparison_error,
                "match_confidence": s.match_confidence,
            }
            for s in states
        ],
    }
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n✅ Results saved → {json_path}")

    full_result = {
        "id": task_id,
        "question": question,
        "files": _nan_to_none([s.model_dump() for s in states]),
    }
    full_json_path.write_text(json.dumps(full_result, indent=2, ensure_ascii=False))
    print(f"✅ Full states saved → {full_json_path}\n\n\n\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate one agent result directory against a task's reference answer.")
    parser.add_argument("--task-dir",   required=True, type=Path, help="Task directory containing task.json, eval.json and ref_answer/")
    parser.add_argument("--result-dir", required=True, type=Path, help="Agent result directory to evaluate")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to save the comparison results and log")
    parser.add_argument("--label", default="agent",   help="Label for output filenames (default: agent)")
    parser.add_argument("--model", default="gpt-5.4", help="LLM model for matching/strategy/comparison (default: gpt-5.4)")
    args = parser.parse_args()

    load_api()

    task_json_path = args.task_dir / "task.json"
    eval_json_path = args.task_dir / "eval.json"
    task_json = json.loads(task_json_path.read_text()) if task_json_path.exists() else {}
    eval_json = json.loads(eval_json_path.read_text()) if eval_json_path.exists() else {}

    if not task_json:
        sys.exit("❌ task.json is empty or not found")
    if not eval_json:
        sys.exit("❌ eval.json is empty or not found")
    if eval_json.get("id") != task_json.get("id"):
        sys.exit(f"❌ eval.json and task.json have different ids: {eval_json.get('id')} != {task_json.get('id')}")

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    task_id = eval_json["id"]

    log_path  = out_dir / f"{task_id}_{args.label}.log"
    json_path      = out_dir / f"{task_id}_{args.label}.json"
    full_json_path = out_dir / f"{task_id}_{args.label}_full.json"

    print(f"Output directory: {out_dir}")
    print(f"Log:     {log_path}")
    print(f"Results: {json_path}")

    with log_path.open("w") as log_file, log_context(log_file):
        question = eval_json["question"]
        per_file_guidelines = [
            item.get("guidelines") or None
            for item in eval_json.get("scoring", {}).get("expected_output", [])
        ]
        while len(per_file_guidelines) < len(eval_json["ref_answer"]):
            per_file_guidelines.append(None)

        input_files = [str(args.task_dir / p) for p in task_json.get("input_files", [])]

        states = [
            FileEvalState(reference_file=str(args.task_dir / p), eval_guideline=guideline)
            for p, guideline in zip(eval_json["ref_answer"], per_file_guidelines)
        ]

        print(f"\nTask:    {task_id}")
        print(f"Question: {question}")
        print(f"Reference files:")
        for s in states:
            print(f"  {Path(s.reference_file).name}")
            if s.eval_guideline:
                print(f"    guideline: {s.eval_guideline}")
        print(f"Input files:     {[Path(p).name for p in input_files]}")
        print(f"Label:   {args.label}")

        # ── Step 1/4: Match agent outputs to reference files ──────────────────
        print(f"\n{'='*60}")
        print("Step 1/4: Matching output files...")
        try:
            states = _match_output(states, args.result_dir, question, input_files, args.model)
        except ValueError as e:
            # Agent produced no output files — every reference is unmatched
            print(f"❌ {e}")
            states = [
                s.model_copy(update={
                    "comparison_status": "invalid",
                    "comparison_error": str(e),
                    "similarity": 0.0,
                })
                for s in states
            ]
            _save_results(json_path, full_json_path, task_id, question, states)
            return
        except RuntimeError as e:
            # LLM-based matching infrastructure failure
            print(f"❌ {e}")
            states = [
                s.model_copy(update={"comparison_status": "error", "comparison_error": str(e)})
                for s in states
            ]
            _save_results(json_path, full_json_path, task_id, question, states)
            return

        # ── Step 2/4: Detect formats ──────────────────────────────────────────
        print(f"\n{'='*60}")
        print("Step 2/4: Detecting file formats...")
        states = _detect_format(states, args.model)

        # ── Step 3/4: Recommend strategies ────────────────────────────────────
        print(f"\n{'='*60}")
        print("Step 3/4: Recommending comparison strategies...")
        states = _recommend_strategy(states, question, input_files, ROOT / "tools/tool_schema.json", args.model)

        # ── Step 4/4: Compare ─────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("Step 4/4: Comparing files...")
        try:
            states = _compare_files(states, question, args.model, task_id=task_id)
        except Exception as e:
            print(f"❌ Comparison crashed: {e}")
            states = [
                s.model_copy(update={"comparison_status": "error", "comparison_error": str(e)})
                for s in states
            ]
            _save_results(json_path, full_json_path, task_id, question, states)
            return

        # ── Print and save results ────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"RESULTS  —  Task: {task_id}  |  Agent: {args.label}")
        scored = [s for s in states if s.comparison_status in ("success", "invalid")]
        avg = sum(s.similarity for s in scored) / len(scored) if scored else float("nan")
        print(f"Avg similarity: {avg:.4f}" if not math.isnan(avg) else "Avg similarity: NaN")
        print(f"{'='*60}")

        _save_results(json_path, full_json_path, task_id, question, states)

    if all(s.comparison_status in ("error", None) for s in states):
        error = next((s.comparison_error for s in states if s.comparison_error), "All comparisons failed")
        print(f"⚠️  All comparisons failed: {error}")


if __name__ == "__main__":
    main()
