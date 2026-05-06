"""Result data models for validation and comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Literal

import numpy as np
from pydantic import BaseModel, Field

from utils.format import FileSignature


@dataclass
class ValidationResult:
    """Result of file validation."""
    exists: bool = False            # File exists on filesystem
    non_empty: bool = False         # File contains meaningful content
    format_ok: bool = False         # File extension/structure is compliant
    parseable: bool = False         # File can be parsed without errors
    file_path: str = ""             # File path
    signature: FileSignature = None # File signature
    details: Dict[str, Any] = field(default_factory=dict)  # Validation metadata
    error: Optional[str] = None     # Error message if validation fails
    
    @property
    def is_valid(self) -> bool:
        """Overall validation status — file exists, format is recognized, and can be parsed.
        Note: non_empty is intentionally excluded; empty-but-parseable files are valid for comparison."""
        return self.exists and self.format_ok and self.parseable


@dataclass
class ComparisonResult:
    """Result of format-specific comparison.

    status values (set by _finalize_output in FormatHandler base class):
        "success" — comparison ran and returned a valid similarity score
        "error"   — infrastructure/tool failure (exception, NaN similarity)
        "skipped" — comparison not attempted (ref invalid/missing, no handler)
        "invalid" — candidate file missing or invalid; similarity = 0.0
    """
    reference_path: str = ""              # Reference file path
    candidate_path: str = ""              # Candidate file path
    file_format: str = ""           # File format (e.g. "fasta", "vcf", "csv")
    strategy: str = ""              # Comparison strategy used
    parameters: Dict[str, Any] = field(default_factory=dict)  # Input parameters used for comparison
    task_id: Optional[str] = None   # Task identifier
    similarity: float = np.nan      # 0.0-1.0 or NaN if comparison fails
    status: str = ""                # Set by _finalize_output() or explicitly
    details: Dict[str, Any] = field(default_factory=dict)  # Comparison metadata
    error: Optional[str] = None     # Error message if comparison fails
    
    @property
    def success(self) -> bool:
        return self.status == "success"

    @property
    def scored(self) -> bool:
        return self.status in ("success", "invalid")


class EquivalenceResult(BaseModel):
    """
    Unified model for equivalence analysis of scripts, output files, or images.

    Used by:
    - Script comparison node (returns EquivalenceResult)
    - Content comparison node (returns EquivalenceResult per result file)
    - Image comparison (returns EquivalenceResult)
    - Final aggregation (list of EquivalenceResult)

    Each equivalence analysis tracks both reference and candidate file paths.
    Script vs output file distinction is determined by context (script_file_configs vs result_file_configs).
    """
    reference_path: str = Field(description="Path to the reference file")
    candidate_path: str = Field(description="Path to the candidate file")
    equivalence: float = Field(ge=0.0, le=1.0, description="Equivalence score between 0.0 and 1.0, where higher values indicate greater equivalence (1.0 = fully equivalent, 0.0 = not equivalent).")
    recommendation: Literal["equivalent", "partially equivalent", "not equivalent", "uncertain"] = Field(description="Equivalence recommendation")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0, where higher values indicate greater confidence (1.0 = very confident, 0.0 = not confident)")
    explanation: str = Field(description="Detailed explanation, summary, or rationale")

