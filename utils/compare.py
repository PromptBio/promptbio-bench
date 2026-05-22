"""File comparison executor: given EvalFiles, returns ComparisonResults and ComparisonSummaries."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from tools.factory import FormatHandlerFactory
from tools.results import ComparisonResult
from utils.models import ComparisonSummary, EvalFile

logger = logging.getLogger(__name__)


def _make_result(eval_file: EvalFile, task_id: Optional[str] = None, **kwargs) -> ComparisonResult:
    return ComparisonResult(
        reference_path=eval_file.reference_file or "",
        candidate_path=eval_file.candidate_file or "",
        file_format=eval_file.file_format,
        strategy=eval_file.strategy,
        parameters=eval_file.parameters or {},
        task_id=task_id,
        **kwargs,
    )


class ComparisonRunner:
    """
    Comparison executor: takes EvalFiles, returns ComparisonResults and ComparisonSummaries.
    """

    def __init__(self, model: str = "gpt-5.4"):
        self.model = model

    def _compare_single(self, eval_file: EvalFile, question: str, task_id: Optional[str] = None) -> ComparisonResult:
        """Compare one file pair. Returns ComparisonResult with status set."""
        if not eval_file.reference_file or not isinstance(eval_file.reference_file, str) or not Path(eval_file.reference_file).exists():
            return _make_result(eval_file, task_id=task_id, status="skipped",
                                          error=f"Reference file not found: {eval_file.reference_file!r}")
        if not eval_file.candidate_file or not isinstance(eval_file.candidate_file, str) or not Path(eval_file.candidate_file).exists():
            return _make_result(eval_file, task_id=task_id, status="invalid",
                                          similarity=0.0,
                                          error=f"Candidate file not found: {eval_file.candidate_file!r}")

        handler = FormatHandlerFactory.get_handler(eval_file.file_format)
        if handler is None:
            return _make_result(eval_file, task_id=task_id, status="skipped",
                                          error=f"No handler for {eval_file.file_format}")

        try:
            extra_kwargs = dict(eval_file.parameters) if eval_file.parameters else {}
            for _k in ('reference_path', 'candidate_path', 'strategy', 'model', 'question'):
                extra_kwargs.pop(_k, None)
            result = handler.compare(
                reference_path=eval_file.reference_file,
                candidate_path=eval_file.candidate_file,
                strategy=eval_file.strategy,
                model=self.model,
                question=question,
                eval_guideline=eval_file.eval_guideline,
                ref_signature=eval_file.signature,
                **extra_kwargs,
            )
            result.file_format = eval_file.file_format
            result.parameters = eval_file.parameters
            result.task_id = task_id
            return result
        except Exception as e:
            logger.exception("Handler failed for %s", eval_file.file_format)
            return _make_result(eval_file, task_id=task_id, status="error", error=str(e))

    def compare_files(self, eval_files: list[EvalFile], question: str = "",
                      task_id: Optional[str] = None) -> ComparisonSummary:
        """
        Compare a list of file pairs and return an aggregated ComparisonSummary.

        Args:
            eval_files: One EvalFile per file pair to compare.
            question: Optional task question passed to handlers for semantic strategies.
            task_id: Optional task identifier propagated to ComparisonResult and ComparisonSummary.

        Returns:
            ComparisonSummary with comparisons and avg_similarity (na omitted).
        """
        if not eval_files:
            return ComparisonSummary(error="No eval files provided")

        comparisons = [self._compare_single(eval_file, question, task_id) for eval_file in eval_files]

        scored = [r for r in comparisons if r.scored]
        avg_similarity = (
            sum(r.similarity for r in scored) / len(scored)
            if scored else 0.0
        )

        return ComparisonSummary(
            comparisons=comparisons,
            avg_similarity=avg_similarity,
            error="All comparisons skipped/failed" if not scored else None,
            task_id=task_id,
        )
