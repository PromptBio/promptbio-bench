"""Shared data models for the evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from tools.results import ComparisonResult
from utils.format import FileSignature


# ── File matching ─────────────────────────────────────────────────────────────

class FileMapping(BaseModel):
    """Mapping between a reference file and a candidate file."""
    reference_file: str = Field(description="Path to the reference answer file")
    candidate_file: Optional[str] = Field(default=None, description="Path to the matched candidate file, or None if no suitable match exists")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0.0-1.0) in this mapping")
    reasoning: str = Field(description="1-2 sentence explanation of why these files were matched. Be concise but include the key evidence (e.g. filename similarity, format match, task context).")


class FileMappingResult(BaseModel):
    """Complete mapping result for a task."""
    mappings: List[FileMapping] = Field(description="List of file mappings between reference and candidate files")


# ── Strategy ──────────────────────────────────────────────────────────────────

class EvalFile(BaseModel):
    """One file pair to evaluate: paths, format, strategy, and optional metadata."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    reference_file: str
    file_format: str
    candidate_file: Optional[str] = None
    eval_guideline: Optional[str] = None
    strategy: Optional[str] = None
    parameters: Optional[dict] = None
    signature: Optional[FileSignature] = None


class StrategyRecommendation(BaseModel):
    """LLM structured output for a strategy recommendation."""
    strategy: str = Field(
        description="Format specific recommended comparison strategy name (e.g., 'exact', 'approximate', 'summary', 'semantic', etc.)"
    )
    parameters: dict = Field(
        default_factory=dict,
        description="Strategy-specific parameters for the chosen strategy (e.g., tolerance, etc.)"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score (0.0-1.0) in the recommendation")
    reasoning: str = Field(
        description="1-2 sentence explanation of why this strategy was chosen. Be concise but state the key deciding factor (e.g. whether the task is deterministic, stochastic, or semantic)."
    )


class StrategyResult(EvalFile):
    """Strategy recommendation for one candidate file."""
    confidence: float = 0.0
    reasoning: str = ""
    error: Optional[str] = None


# ── Comparison ────────────────────────────────────────────────────────────────

@dataclass
class ComparisonSummary:
    """Aggregated result for one task + one candidate."""
    comparisons: list[ComparisonResult] = field(default_factory=list)
    avg_similarity: float = 0.0
    error: Optional[str] = None
    task_id: Optional[str] = None


# ── Pipeline state ────────────────────────────────────────────────────────────

class FileEvalState(BaseModel):
    """Single per-file record accumulating data across all 4 pipeline steps."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── Initial config (from task/eval JSON) ─────────────────────────────────
    reference_file: str
    eval_guideline: Optional[str] = None

    # ── Step 1: Match ─────────────────────────────────────────────────────────
    candidate_file: Optional[str] = None
    match_confidence: Optional[float] = None
    match_reasoning: Optional[str] = None

    # ── Step 2: Detect format ─────────────────────────────────────────────────
    file_format: Optional[str] = None
    signature: Optional[FileSignature] = None

    # ── Step 3: Recommend strategy ────────────────────────────────────────────
    strategy: str = "unknown"
    parameters: dict = Field(default_factory=dict)
    strategy_confidence: Optional[float] = None
    strategy_reasoning: Optional[str] = None
    strategy_error: Optional[str] = None

    # ── Step 4: Compare ───────────────────────────────────────────────────────
    similarity: float = Field(default=float("nan"))
    comparison_status: Optional[str] = None
    comparison_details: dict = Field(default_factory=dict)
    comparison_error: Optional[str] = None
