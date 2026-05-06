#!/usr/bin/env python3
"""
Agent Trace Analyzer
Diagnoses why a failed agent run failed, by classifying the blocker into one
of a fixed taxonomy of failure categories.

Usage:
    python trace_analyzer.py <trace.json> [--task TASK] [--log LOG] [--output OUT] [--model MODEL]
"""

import argparse
import json
import os
import sys
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from utils.agents import create_llm_agent, invoke_structured_agent


class FailureCategory(str, Enum):
    PLANNING            = "planning"
    TOOL_SELECTION      = "tool_selection"
    TOOL_USAGE          = "tool_usage"
    HALLUCINATION       = "hallucination"
    ENVIRONMENT_CONFIG  = "environment_config"
    CODE_EXECUTION      = "code_execution"
    DATA_IO             = "data_io"
    INFRASTRUCTURE      = "infrastructure"
    RESOURCE_LIMIT      = "resource_limit"
    UNKNOWN             = "unknown"


class StepStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


class StepLabel(BaseModel):
    """Per-step label assigned by the analyzer."""
    step: int
    attempt: int
    category: FailureCategory = Field(..., description="Pipeline-stage category this step belongs to.")
    status: StepStatus
    note: str = Field(..., description="≤120 chars quoting evidence for the label.")


class TraceAnalysis(BaseModel):
    """Diagnosis of a single agent run."""
    failed: bool = Field(..., description="True if the run did not produce a result that solves the task.")
    steps: list[StepLabel] = Field(default_factory=list, description="Per-step labels (category + status) for every step in the trace.")
    category: Optional[FailureCategory] = Field(None, description="The blocker's failure category. Null iff failed=false.")
    blocker_step: Optional[int] = Field(None, description="Step number of the earliest unrecovered failure. Null iff failed=false.")
    reason: Optional[str] = Field(None, description="≤500 chars. What the agent was trying to do, why it couldn't recover, citing step numbers. Null iff failed=false.")
    confidence: float = Field(..., description="0.0–1.0 confidence in the classification.")


SYSTEM_PROMPT = """You analyze agent execution traces to diagnose why a run failed.

The agent runs as a ReAct loop: think → act → observe → think → … Each step in the trace is one such cycle, with a status (success/failure/partial) and possibly multiple attempts.

Your job: produce ONE TraceAnalysis containing (a) per-step labels for every step in the trace and (b) if the run failed, a single blocker.

## Step 0 — Label every step
For every step in the trace, emit a StepLabel with {step, attempt, category, status, note}.
- `category` uses the same failure-category taxonomy described in Step 3 below — it labels which pipeline stage the step is operating in (e.g., a successful tool call is `tool_usage`; a successful code run is `code_execution`).
- `status` is one of success / failure / partial (copy from the trace, correcting only if obviously wrong).
- `note` is ≤120 chars quoting the action or observation.

## Step 1 — Did the run actually fail?
A run is FAILED only if it ended without a result that solves the task. Successful runs that contained intermediate failing steps (the agent retried and recovered) are NOT failures — set failed=false and leave category/reason null.

## Step 2 — If failed, find the BLOCKER
The blocker is the EARLIEST UNRECOVERED failure. Walk the failing steps in order:

For each failing step, ask: did any LATER step successfully achieve the same subgoal? "Same subgoal" means any of:
  (a) same target file/output/entity, OR
  (b) the later step's reasoning explicitly references this failure ("the previous call failed", "let me try X instead"), OR
  (c) the later step is a minor variation of this action (same tool with different args, different tool with same purpose, rewritten code with same goal).

- If yes → this failure was RECOVERED. Skip it.
- If no → this failure is a blocker candidate.

The EARLIEST candidate is the blocker. Everything after it is either a downstream consequence or a workaround attempt that introduced new errors but never solved the original problem.

Special case — silent failure: if every step "succeeded" but the final answer doesn't solve the task, the blocker is at the synthesis stage and category is `planning`.

## Step 3 — Classify the blocker into one category

The agent's execution pipeline:
  THINK (plan) → ACT (pick tool / write code, formulate args)
              → ENV (sandbox readiness)
              → EXECUTE (code runs)
              → DATA (read/write files)
              → EXTERNAL (network/API)
              → SYNTHESIZE (final answer)

Categories (one per stage):
- planning            — wrong plan, wrong subgoal, answers wrong question, gives up without trying, or completed with an answer that doesn't solve the task.
- tool_selection      — wrong tool chosen for the subtask (or raw code where a tool existed).
- tool_usage          — right tool, wrong arguments / malformed call.
- hallucination       — referenced something that doesn't exist: fake function, fabricated path, made-up flag, fake tool name.
- environment_config  — real package/binary/key missing from sandbox; permission problem.
- code_execution      — runtime/logic/syntax error in installed real code (TypeError, IndexError, etc.).
- data_io             — required input data absent / mislocated / malformed / wrong schema.
- infrastructure      — non-deterministic external failure (network, third-party 5xx, rate limit).
- resource_limit      — deterministic quota: OOM, timeout, context window, disk full.
- unknown             — cannot tell from the trace + log tail.

DISAMBIGUATION RULE: pick the EARLIEST stage in the pipeline where the failure first originated. Later errors are downstream consequences of it.

Examples:
- ModuleNotFoundError, then a workaround that crashes → environment_config (env error came first; the crash is a consequence).
- TypeError on attempt 1, then FileNotFoundError on attempt 2 of the same subgoal → code_execution.
- Run completes but answers the wrong question → planning.

## Step 4 — Write reason and blocker_step
Set `blocker_step` to the step number of the blocker. Write `reason` (≤500 chars): what the agent was trying to do, why it couldn't recover, citing step numbers.

## Step 5 — Confidence
0.0–1.0 based on how unambiguously the trace + log tail support your classification."""


def analyze_trace(
    trace: list[dict],
    task: Optional[str] = None,
    log_tail: Optional[str] = None,
    model: str = "gpt-5.4",
) -> TraceAnalysis:
    """Diagnose a single agent trace.

    Args:
        trace: List of step dicts (output of `extract_trace` from utils/trace.py).
        task: Optional original task description — improves silent-failure detection.
        log_tail: Optional last ~4 KB of raw log — for tracebacks the trace abstracted.
        model: LLM model name.
    """
    parts = []
    if task:
        parts.append(f"<task>\n{task}\n</task>")
    parts.append(f"<trace>\n{json.dumps(trace, indent=2, ensure_ascii=False)}\n</trace>")
    if log_tail:
        parts.append(f"<log_tail>\n{log_tail}\n</log_tail>")
    human = "\n\n".join(parts)

    agent = create_llm_agent(model, SYSTEM_PROMPT, response_format=TraceAnalysis)
    return invoke_structured_agent(agent, human, TraceAnalysis)


def analyze_trace_file(
    trace_path: str,
    task: Optional[str] = None,
    log_path: Optional[str] = None,
    model: str = "gpt-5.4",
    log_tail_chars: int = 4000,
) -> TraceAnalysis:
    """Load a trace JSON file (and optional raw log tail) and analyze it."""
    with open(trace_path, "r", encoding="utf-8") as fh:
        trace = json.load(fh)

    log_tail = None
    if log_path and os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        log_tail = text[-log_tail_chars:]

    return analyze_trace(trace, task=task, log_tail=log_tail, model=model)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Diagnose why an agent run failed, from its extracted trace."
    )
    parser.add_argument("trace_file", help="Path to a .trace.json produced by trace.py")
    parser.add_argument("--task", default=None, help="Original task description (improves silent-failure detection)")
    parser.add_argument("--log", default=None, help="Path to the raw .log file (used for tracebacks)")
    parser.add_argument("--output", "-o", default=None, help="Output JSON path (default: <trace_file>.analysis.json)")
    parser.add_argument("--model", default="gpt-5.4", help="Model to use")
    args = parser.parse_args()

    if not os.path.exists(args.trace_file):
        print(f"Error: file not found: {args.trace_file}", file=sys.stderr)
        sys.exit(1)

    print(f"📄  Reading trace: {args.trace_file}")
    print(f"🤖  Calling {args.model} …")
    analysis = analyze_trace_file(
        args.trace_file, task=args.task, log_path=args.log, model=args.model
    )

    out_path = args.output or (args.trace_file + ".analysis.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(analysis.model_dump(), fh, indent=2, ensure_ascii=False)

    print(f"✅  Analysis → {out_path}")
    print()
    if analysis.failed:
        print(f"failed:     true")
        print(f"category:   {analysis.category.value if analysis.category else None}")
        print(f"confidence: {analysis.confidence}")
        print(f"reason:     {analysis.reason}")
    else:
        print("failed: false (run succeeded)")


if __name__ == "__main__":
    main()
