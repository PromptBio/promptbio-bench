#!/usr/bin/env python3
"""
Agent Trace Extractor
Parses a agent running log and extracts structured execution steps

Usage:
    python trace.py <log_file> [--output output.json] [--model model]
"""

import argparse
import json
import os
import sys

from pydantic import BaseModel, Field
from utils.agents import create_llm_agent, invoke_structured_agent


class TraceStep(BaseModel):
    """One execution step (or retry attempt) in an agent log."""
    step: int = Field(..., description="Logical step number — same across retries of the same step")
    name: str = Field(..., description="Short human-readable step name")
    attempt: int = Field(1, description="Attempt number within this step (1 = first try)")
    action_type: str = Field(..., description="One of: python | bash | tool | other")
    action: str = Field(..., description="Concise description of the key code or command run (≤200 chars)")
    observation: str = Field(..., description="Concise description of the result/observation (≤200 chars)")
    status: str = Field(..., description="One of: success | failure | partial")
    retry: bool = Field(False, description="True only on attempt >= 2")
    confidence: float = Field(..., description="Confidence in this extraction, 0.0–1.0")
    reasoning: str = Field(..., description="Brief reasoning explaining how this step was identified from the log (≤200 chars)")


class AgentTrace(BaseModel):
    """Full ordered list of execution steps extracted from an agent log."""
    steps: list[TraceStep]



def extract_trace(log_text: str, model: str = "gpt-5.4") -> list[dict]:
    """Use LangChain structured output to parse the log into TraceStep objects."""
    system = """You are an expert log parser that extracts structured agent execution traces from raw log files.

Rules:
- Extract every execution step in order, INCLUDING retries.
- When a step is retried, emit one object per attempt with the SAME step number but incrementing attempt (1, 2, …).
- Set retry=true only on attempt >= 2.
- Keep action and observation concise (≤ 120 chars each).
- action_type must be one of: python, bash, tool, other.
- status must be one of: success, failure, partial.
- For each step, provide a confidence score (0.0–1.0) reflecting how certain you are about the extraction.
- For each step, provide a brief reasoning (≤ 200 chars) explaining which log evidence supports this step."""

    human = f"""Extract the agent execution trace from the log below.

<log>
{log_text}
</log>"""

    agent = create_llm_agent(model, system, response_format=AgentTrace)
    result: AgentTrace = invoke_structured_agent(agent, human, AgentTrace)
    return [step.model_dump() for step in result.steps]


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract structured agent trace from a biomni log file (LangChain version)."
    )
    parser.add_argument("log_file", help="Path to the .log file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON path (default: <log_file>.trace.json)")
    parser.add_argument("--model", default="gpt-5.4",
                        help="Model to use")
    args = parser.parse_args()

    if not os.path.exists(args.log_file):
        print(f"Error: file not found: {args.log_file}", file=sys.stderr)
        sys.exit(1)

    with open(args.log_file, "r", encoding="utf-8", errors="replace") as fh:
        log_text = fh.read()

    print(f"📄  Read {len(log_text):,} chars from {args.log_file}")
    print(f"🤖  Calling {args.model} via LangChain …")

    trace = extract_trace(log_text, model=args.model)

    out_path = args.output or (args.log_file + ".trace.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(trace, fh, indent=2, ensure_ascii=False)

    print(f"✅  Extracted {len(trace)} step(s) → {out_path}")
    print()
    print(json.dumps(trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
