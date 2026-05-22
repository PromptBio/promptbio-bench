"""Base classes for format handlers."""
import os
import shutil
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
from tools.results import ValidationResult, ComparisonResult

logger = __import__('logging').getLogger(__name__)


def resolve_r_executable(name: str = "Rscript") -> str:
    """Find the R executable, preferring the one co-located with the running Python.

    Resolution order:
      1. RSCRIPT_PATH (or R_PATH) environment variable
      2. Same directory as ``sys.executable`` (works inside conda envs where
         Jupyter kernels may not fully activate the environment PATH)
      3. System PATH via ``shutil.which``
      4. Bare command name as last-resort fallback
    """
    env_var = "RSCRIPT_PATH" if name == "Rscript" else "R_PATH"
    env_path = os.environ.get(env_var)
    if env_path and os.path.isfile(env_path):
        return env_path

    candidate = Path(sys.executable).parent / name
    if candidate.is_file():
        return str(candidate)

    return shutil.which(name) or name


class FormatHandler(ABC):
    """Base class for format-specific handlers."""
    
    def __init__(self, format_name: str):
        self.format_name = format_name
    
    @abstractmethod
    def validate(self, file_path: str, **kwargs) -> ValidationResult:
        """Validate file format.
        
        Args:
            file_path: Path to file
            **kwargs: optional parameters (e.g., min_size)
            - min_size: Minimum size of the file in bytes, default is 0
        """
        pass
    
    @abstractmethod
    def compare(self, reference_path: str, candidate_path: str, strategy: str = "exact", **kwargs) -> ComparisonResult:
        """Compare two files using specified strategy.

        Args:
            reference_path: Path to reference file
            candidate_path: Path to candidate file
            strategy: Comparison strategy (exact, approximate, summary)
            **kwargs: Strategy-specific parameters (e.g., tolerance, k-mer size)

        Returns:
            ComparisonResult with status, strategy, similarity score, details, and optional error.

        Implementation pattern:
            1. Initialize self.result = ComparisonResult(...)
            2. Validate reference and candidate files via self.validate()
            3. If both valid, run strategy-specific comparison
            4. Return self._finalize_output(ref_val, alt_val, self.result)

        The _finalize_output method sets status based on validation and similarity:
            "success" — valid similarity (0.0–1.0), no error
            "error"   — infrastructure/tool failure (NaN similarity, exception)
            "skipped" — reference file invalid or missing
            "invalid" — candidate file missing or invalid (candidate output unusable)
        """
        pass

    def _finalize_output(self, ref_val: ValidationResult, alt_val: ValidationResult, result: ComparisonResult) -> ComparisonResult:
        """Derive and set status from validation results and comparison outcome.

        Call at the end of every compare() implementation.
        """
        if not ref_val.exists or not ref_val.is_valid:
            result.status = "skipped"
            result.error = result.error or (
                "Reference file not found" if not ref_val.exists
                else f"Reference file is not valid: {ref_val.error}"
            )
        elif not alt_val.exists or not alt_val.is_valid:
            result.status = "invalid"
            result.similarity = 0.0
            result.error = result.error or (
                "Candidate file not found" if not alt_val.exists
                else f"Candidate file is not valid: {alt_val.error}"
            )
        elif not alt_val.non_empty:
            # File exists and parses correctly but contains no data records
            result.status = "invalid"
            result.similarity = 0.0
            result.error = result.error or f"Candidate file has no data: {alt_val.file_path}"
        elif not np.isnan(result.similarity) and result.error is None:
            result.status = "success"
        else:
            result.status = "error"
            if not result.error:
                result.error = "Handler returned NaN similarity"
        return result

    # ============================================================================
    # File Status Checkers
    # ============================================================================
    
    def _is_file_exists(self, file_path: str) -> bool:
        """Check if file exists
        """
        return os.path.exists(file_path)
    
    def _is_dir_exists(self, path: str) -> bool:
        """Check if path exists and is a directory.
        
        Args:
            path: Path to check
        
        Returns:
            True if path exists and is a directory, False otherwise
        """
        return os.path.isdir(path)
    
    def _is_file_nonempty(self, file_path: str, min_size: int = 0) -> bool:
        """Check if file is nonempty by file size, default to 0 bytes
        Returns False when file does not exist or is empty
        """
        try:
            file_size = os.path.getsize(file_path)
            return file_size > min_size
        except FileNotFoundError:
            logger.warning(f"File does not exist: {file_path}")
            return False
        except Exception as e:
            logger.warning(f"Failed to get file size: {e}")
            return False
    
    # ============================================================================
    # File IO/Cleanup Methods
    # ============================================================================

    def _cleanup_file(self, file_path: Optional[str], verbose: bool = False) -> None:
        """Remove temporary file if it exists."""
        if not file_path:
            return
        
        try:
            os.remove(file_path)
            if verbose:
                logger.debug(f"Removed temporary file: {file_path}")
        except FileNotFoundError:
            pass  # File already removed or doesn't exist
        except Exception as e:
            if verbose:
                logger.warning(f"Failed to remove temporary file {file_path}: {e}")
    
    def _close_file(self, file_obj) -> None:
        """Close a file object if it is not None.
        
        Args:
            file_obj: File object to close (e.g., pysam.AlignmentFile, file handle, etc.)
        """
        if file_obj is not None:
            try:
                file_obj.close()
            except Exception as e:
                logger.warning(f"Failed to close file object: {e}")
