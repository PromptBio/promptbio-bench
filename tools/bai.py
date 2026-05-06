"""BAM/CRAM index file handler (.bai, .crai).

Strategy:
- functional: Compare index functionality using idxstats and subset operations
- summary: Summary statistics comparison of index file format, validity, and sizes

Required cmdline tools:
- samtools
"""

import logging
import os
import subprocess
from typing import Optional

import numpy as np
import pandas as pd
import pysam

from tools.base import FormatHandler
from tools.results import ValidationResult, ComparisonResult
from utils.format import detect_file_signature

logger = logging.getLogger(__name__)


class BAIHandler(FormatHandler):
    """Handler for BAM/CRAM index files (.bai, .crai)."""
    
    def __init__(self):
        super().__init__("bai")
        self.format_map = {
            'bai': {'mode': 'rb', 'aln_fmt': 'bam', },
            'crai': {'mode': 'rc', 'aln_fmt': 'cram'},
        }
        self.supported_formats = list(self.format_map.keys())
        self.supported_strategies = ["functional", "summary"]
    
    # ============================================================================
    # Public API Methods
    # ============================================================================
    
    def validate(self, file_path: str, **kwargs) -> ValidationResult:
        """Validate BAM/CRAM index file (.bai or .crai).
        non-empty check is performed by file size, later can be updated by reading content if needed
        
        Args:
            file_path: Path to BAM/CRAM index file (.bai or .crai)
            **kwargs: optional parameters (e.g., indexed_file, min_size)
            - indexed_file: Path to corresponding BAM/CRAM indexed file
            - min_size: Minimum size of the index file in bytes, default is 0
        """
        # check existence and non-emptiness (by file size)
        is_exists = self._is_file_exists(file_path)
        is_nonempty = self._is_file_nonempty(file_path, min_size=kwargs.get("min_size", 0))
        
        # check file format
        signature = detect_file_signature(file_path, fallback=True)
        is_format_ok = signature.format in self.supported_formats

        # initialize validation result
        result = ValidationResult(exists=is_exists, non_empty=is_nonempty, format_ok=is_format_ok, parseable=False, 
                                  file_path=file_path, signature=signature, details={}, error=None)

        if not is_exists:
            logger.error(f"BAM/CRAM index file not found: {file_path}")
            result.error = "File not found"
            return result
        
        if not is_nonempty:
            logger.warning(f"BAM/CRAM index file is empty: {file_path}")
            result.error = "File is empty"
            return result
        
        if not is_format_ok:
            logger.error(f"Not a BAM/CRAM index file: detected={signature.format}")
            result.error = f"Not a BAM/CRAM index file: detected={signature.format}"
            return result
        
        # find the indexed file
        indexed_file = kwargs.get("indexed_file", self._get_indexed_file(file_path, signature.format))
        result.details.update({"indexed_file": indexed_file, "index_format": signature.format})
        if not indexed_file:
            logger.error(f"Corresponding BAM/CRAM file not found: {file_path}")
            result.error = f"Corresponding BAM/CRAM file not found: {file_path}"
            return result
        
        # Validate index by opening indexed file with index
        try:
            mode = self.format_map[signature.format]['mode']
            indexed_file = pysam.AlignmentFile(indexed_file, mode, index_filename=file_path)
            result.parseable = indexed_file.has_index()
            indexed_file.close()
            logger.info(f"BAM/CRAM index file validation passed: {file_path}")
        except Exception as e:
            logger.error(f"BAM/CRAM index file validation failed: {e}")
            result.error = f"BAM/CRAM index file validation failed: {e}"
        return result
    
    def compare(self, reference_path: str, candidate_path: str, strategy: str = "functional", **kwargs) -> ComparisonResult:
        """Compare BAM/CRAM index files (.bai or .crai).
        
        Supported strategies:
        - functional: Compare index functionality using idxstats and subset operations
        - summary: Summary statistics comparison of index file format, validity, and sizes
        
        Args:
            reference_path: Path to reference index file
            candidate_path: Path to candidate index file
            strategy: Comparison strategy (functional, summary)
            **kwargs: Strategy-specific parameters (e.g., indexed_file)
            - indexed_file: Path to the indexed file (BAM/CRAM) that both index files index, only used for functional strategy
            Note: When comparing two index files, they should typically index the same underlying BAM/CRAM file.            
        """
        # initialize comparison result
        self.result = ComparisonResult(reference_path=reference_path, candidate_path=candidate_path, 
                                       strategy=strategy, similarity=np.nan, details={}, error=None)
        
        # validate reference and candidate files
        indexed_file = kwargs.get("indexed_file", None)
        ref_validation = self.validate(reference_path, indexed_file=indexed_file)
        alt_validation = self.validate(candidate_path, indexed_file=indexed_file)

        try:
            if ref_validation.is_valid and alt_validation.is_valid:
                if ref_validation.signature.format != alt_validation.signature.format:
                    logger.error(f"Cannot compare {ref_validation.signature.format} with {alt_validation.signature.format}")
                    self.result.error = f"Cannot compare {ref_validation.signature.format} with {alt_validation.signature.format}"
                else:
                    # compare files using specified strategy
                    if strategy == "functional":
                        self._compare_functional(ref_validation, alt_validation, **kwargs)
                    elif strategy == "summary":
                        self._compare_summary(ref_validation, alt_validation, **kwargs)
                    else:
                        logger.warning(f"Unknown strategy '{strategy}', fall back to functional")
                        self._compare_functional(ref_validation, alt_validation, **kwargs)
        except Exception as e:
            logger.error(f"Failed to compare BAM/CRAM index files: {e}")
            self.result.error = f"Failed to compare BAM/CRAM index files: {e}"
        return self._finalize_output(ref_validation, alt_validation, self.result)
    
    # ============================================================================
    # File I/O and Loading Methods
    # ============================================================================
    
    def _get_indexed_file(self, index_path: str, fmt: Optional[str] = None) -> Optional[str]:
        """Get the corresponding BAM/CRAM file for an index file of given format.
        Handles both naming conventions: xxx.bai/.crai and xxx.bam.bai/.cram.crai
        
        Args:
            index_path: Path to index file (.bai or .crai)
            fmt: Format of index file ('bai' or 'crai')
            
        Returns:
            Path to corresponding BAM/CRAM indexed file if found, None otherwise
        """
        if fmt is None:
            fmt = detect_file_signature(index_path, fallback=True).format
        
        if fmt not in self.format_map.keys():
            logger.warning(f"Could not get alignment format: {fmt}")
            return None
        
        aln_fmt = self.format_map[fmt]['aln_fmt']  # alignment format
        index_suffix  = f'.{fmt}'           # e.g. xxx.bai -> xxx.bam
        double_suffix = f'.{aln_fmt}.{fmt}' # e.g. xxx.bam.bai -> xxx.bam
        
        # Try double suffix first, then index suffix
        for suffix in [double_suffix, index_suffix]:
            if index_path.endswith(suffix):
                file_path = index_path[:-len(suffix)] + f'.{aln_fmt}'
                if self._is_file_exists(file_path):
                    return file_path
        
        logger.warning(f"Could not find corresponding file for index file: {index_path} with format {fmt}")
        return None
    
    # ============================================================================
    # File Status Checkers
    # ============================================================================
    
    def _is_indexed_file_exists(self, index_path: str, fmt: Optional[str] = None) -> bool:
        """Check if the corresponding indexed file exists.
        """
        return self._is_file_exists(self._get_indexed_file(index_path, fmt))

    # ============================================================================
    # Statistics and Analysis Methods
    # ============================================================================
    
    def _get_idxstat_df(self, indexed_file: str) -> pd.DataFrame:
        """Get BAM/CRAM indexed file index statistics using samtools idxstats.
        Returns a DataFrame with columns: reference, length, mapped, unmapped.
        """
        df = pd.DataFrame(columns=['reference', 'length', 'mapped', 'unmapped'])
        try:
            res = subprocess.run(
                ["samtools", "idxstats", indexed_file],
                check=True, capture_output=True, text=True
            )
        except Exception as e:
            logger.error(f"samtools idxstats failed: {e}")
            return df.set_index('reference')
        
        # Parse idxstats output: reference_name\tlength\tmapped\tunmapped
        data = []
        for line in res.stdout.strip().split('\n'):
            if not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) >= 4:
                mapped, unmapped = int(parts[2] or 0), int(parts[3] or 0)
                data.append({'reference': parts[0], 'length': int(parts[1] or 0),
                             'mapped': mapped, 'unmapped': unmapped})
        df = pd.DataFrame(data, columns=['reference', 'length', 'mapped', 'unmapped'])
        return df.set_index('reference')
    
    # ============================================================================
    # Private Comparison Implementation Methods
    # ============================================================================

    def _compare_functional(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Functional comparison: Compare the supported functional properties of the index files using samtools idxstats.
        in the future can also verify equivalence using subset operations (retrieving specific genomic regions)
        Returns a ComparisonResult with similarity score based on functional equivalence.
        """
        ref_indexed_file = ref_validation.details.get("indexed_file", None)
        alt_indexed_file = alt_validation.details.get("indexed_file", None)
        
        # Compare idxstats using samtools idxstats
        try:
            ref_idxstats_df = self._get_idxstat_df(ref_indexed_file)
            alt_idxstats_df = self._get_idxstat_df(alt_indexed_file)

            ref_refs = set(ref_idxstats_df.index)
            alt_refs = set(alt_idxstats_df.index)

            if ref_refs != alt_refs:
                logger.error(f"Reference sequences differ: ref={list(ref_refs)}, alt={list(alt_refs)}")
                self.result.error = f"Reference sequences differ: ref={list(ref_refs)}, alt={list(alt_refs)}"
                return 
            
            numeric_cols = ['length', 'mapped', 'unmapped']
            mae_idxstats_df = pd.DataFrame((ref_idxstats_df[numeric_cols].values - alt_idxstats_df[numeric_cols].values).abs(), 
                                            index=ref_idxstats_df.index, columns=numeric_cols)
            rae_idxstats_mean = np.nanmean(mae_idxstats_df.sum(axis=1)/ref_idxstats_df[numeric_cols].sum(axis=1))
            similarity = 1.0 - rae_idxstats_mean
            self.result.similarity = similarity
            self.result.details.update({"ref_idxstats_df": ref_idxstats_df.to_dict(), "alt_idxstats_df": alt_idxstats_df.to_dict()})
        
        except Exception as e:
            logger.error(f"Idxstats computation failed: {e}")
            self.result.error = f"Idxstats computation failed: {e}"

    def _compare_summary(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Summary statistics comparison: Compare index sizes.
        Return a ComparisonResult with similarity score based on idxstats
        """

        # Compare file sizes
        ref_size = os.path.getsize(ref_validation.file_path)
        alt_size = os.path.getsize(alt_validation.file_path)
        
        # Size similarity (indexes for similar files should have similar sizes)
        size_diff = abs(ref_size - alt_size)
        max_size = max(ref_size, alt_size)
        self.result.similarity = 1.0 - (size_diff / max_size) if max_size > 0 else 0.0
        self.result.details.update({"ref_size": ref_size, "alt_size": alt_size})
        return

