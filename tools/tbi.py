"""Tabix index file handler (.tbi, .csi).

Strategy:
- functional: Compare index functionality using contig/chromosome information and region queries
- summary: Summary statistics comparison of index file format, validity, and sizes

Required cmdline tools:
- (None)
"""

import logging
import os
import re
from typing import Optional

import numpy as np
import pandas as pd
import pysam

from tools.base import FormatHandler
from tools.results import ValidationResult, ComparisonResult
from utils.format import detect_file_signature

logger = logging.getLogger(__name__)


class TabixHandler(FormatHandler):
    """Handler for Tabix index files (.tbi, .csi)."""
    
    def __init__(self):
        super().__init__("tbi")
        self.format_map = {
            'tbi': {},
            'csi': {},
        }
        self.supported_formats = list(self.format_map.keys())
        self.supported_strategies = ["functional", "summary"]
    
    # ============================================================================
    # Public API Methods
    # ============================================================================
    
    def validate(self, file_path: str, **kwargs) -> ValidationResult:
        """Validate Tabix index file (.tbi or .csi).
        non-empty check is performed by file size, later can be updated by reading content if needed
        
        Args:
            file_path: Path to Tabix index file
            **kwargs: optional parameters (e.g., indexed_file, min_size)
            - indexed_file: Path to corresponding indexed file (e.g., .vcf.gz, .bed.gz)
            - min_size: Minimum size of the Tabix index file (.tbi or .csi) in bytes, default is 0
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
            logger.error(f"Tabix index file not found: {file_path}")
            result.error = "File not found"
            return result
        
        if not is_nonempty:
            logger.warning(f"Tabix index file is empty: {file_path}")
            result.error = "File is empty"
            return result
        
        if not is_format_ok:
            logger.error(f"Not a Tabix index file: detected={signature.format}")
            result.error = f"Not a Tabix index file: detected={signature.format}"
            return result
        
        # find the indexed file
        indexed_file = kwargs.get("indexed_file") or self._get_indexed_file(file_path, signature.format)
        result.details.update({"indexed_file": indexed_file, "index_format": signature.format})
        if not indexed_file:
            fmt = signature.format
            expected = file_path[:-len(f'.{fmt}')] if file_path.endswith(f'.{fmt}') else None
            missing = expected or file_path
            logger.error(f"Corresponding indexed file not found: {missing}")
            result.error = f"Corresponding indexed file not found: {missing}"
            return result
        
        # validate by parsing the Tabix index file
        try:
            tabix_file = pysam.TabixFile(indexed_file, index=file_path)
            result.parseable = tabix_file.contigs is not None
            tabix_file.close()
            logger.info(f"Tabix index file validation passed: {file_path}")
        except Exception as e:
            logger.error(f"Tabix index file validation failed: {e}")
            result.error = f"Tabix index file validation failed: {e}"
        return result
    
    def compare(self, reference_path: str, candidate_path: str, strategy: str = "functional", **kwargs) -> ComparisonResult:
        """Compare Tabix index files (.tbi or .csi).
        
        Supported strategies:
        - functional: Compare index functionality using contig/chromosome information and region queries
        - summary: Summary statistics comparison of index file format, validity, and sizes
        
        Args:
            reference_path: Path to reference Tabix index file
            candidate_path: Path to candidate Tabix index file
            strategy: Comparison strategy (functional, summary)
            **kwargs: Strategy-specific parameters (e.g., indexed_file)
            - indexed_file: Path to the indexed file (VCF/BCF/BED/etc.) that both index files index, only used for functional strategy
            Note: When comparing two index files, they should typically index the same underlying file.
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
            logger.error(f"Failed to compare Tabix index files: {e}")
            self.result.error = f"Failed to compare Tabix index files: {e}"
        return self._finalize_output(ref_validation, alt_validation, self.result)
    
    # ============================================================================
    # File I/O and Loading Methods
    # ============================================================================
    
    def _get_indexed_file(self, index_path: str, fmt: Optional[str] = None) -> Optional[str]:
        """Get the corresponding indexed file for a Tabix index file.
        Tabix index files are named by appending .tbi or .csi to the original file.
        For example: file.vcf.gz.tbi -> file.vcf.gz, file.vcf.gz.csi -> file.vcf.gz
        
        Returns path to corresponding indexed file if found, None otherwise
        """
        if fmt is None:
            signature = detect_file_signature(index_path, fallback=True)
            fmt = signature.format
        
        if fmt not in self.format_map.keys():
            logger.warning(f"Could not get index format: {fmt}")
            return None
        
        index_ext = f'.{fmt}'  # e.g. .tbi or .csi
        
        if not index_path.endswith(index_ext):
            logger.warning(f"Index file does not end with {index_ext}: {index_path}")
            return None
        
        # Tabix indexes append .tbi or .csi to the original file name
        # So removing the extension gives us the indexed file
        indexed_file = index_path[:-len(index_ext)]
        
        if self._is_file_exists(indexed_file):
            return indexed_file
        
        logger.warning(f"Could not find indexed file for index file: {index_path} with format {fmt}")
        return None
    
    # ============================================================================
    # File Status Checkers
    # ============================================================================
    
    def _is_indexed_file_exists(self, index_path: str) -> bool:
        """Check if the corresponding indexed file exists.
        """
        return self._is_file_exists(self._get_indexed_file(index_path))
    
    # ============================================================================
    # Statistics and Analysis Methods
    # ============================================================================
    
    def _get_contig_info(self, indexed_file: str, index_path: str) -> pd.DataFrame:
        """Get Tabix index contig/chromosome information.
        Returns a DataFrame with columns: contig, length.
        
        For VCF files: tries VariantFile first for direct length access, falls back to header parsing.
        For BED/GFF files: length is set to 0 (not available in file format).
        """
        try:
            # Try VariantFile for VCF files (provides direct contig length access).
            # Skip when index_path is given — VariantFile can't take an explicit index,
            # so htslib would search the wrong directory and emit stderr errors.
            if indexed_file.endswith(('.vcf.gz', '.vcf', '.bcf')) and not index_path:
                try:
                    variant_file = pysam.VariantFile(indexed_file)
                    data = [
                        {'contig': name, 'length': contig.length if hasattr(contig, 'length') else 0}
                        for name, contig in variant_file.header.contigs.items()
                    ]
                    variant_file.close()
                    return pd.DataFrame(data, columns=['contig', 'length']).set_index('contig')
                except Exception:
                    pass  # Fall back to TabixFile parsing
            
            # Use TabixFile for all formats (including fallback for VCF)
            tabix_file = pysam.TabixFile(indexed_file, index=index_path)
            contigs = tabix_file.contigs
            
            # Parse contig lengths from header using regex (more robust than string splitting)
            header_lengths = {}
            try:
                header = tabix_file.header
                if hasattr(header, '__iter__'):
                    # Pattern matches: ##contig=<ID=chr1,length=248956422> or ##contig=<length=248956422,ID=chr1>
                    # Uses non-greedy matching to handle fields in any order
                    contig_pattern = re.compile(r'##contig=<.*?ID=([^,>]+).*?length=(\d+)')
                    for line in header:
                        match = contig_pattern.search(line)
                        if match:
                            header_lengths[match.group(1)] = int(match.group(2))
            except Exception:
                pass  # No header or header parsing failed
            
            # Build DataFrame
            data = [
                {'contig': contig, 'length': header_lengths.get(contig, 0)}
                for contig in contigs
            ]
            tabix_file.close()
            return pd.DataFrame(data, columns=['contig', 'length']).set_index('contig')
        except Exception as e:
            logger.error(f"Failed to get contig info: {e}")
            return pd.DataFrame(columns=['contig', 'length']).set_index('contig')
    
    # ============================================================================
    # Private Comparison Implementation Methods
    # ============================================================================
    
    def _compare_functional(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Functional comparison: Compare the supported functional properties of the index files using contig information.
        in the future can also verify equivalence using subset operations (retrieving specific genomic regions)
        Returns a ComparisonResult with similarity score based on functional equivalence.
        """
        ref_indexed_file = ref_validation.details.get("indexed_file", None)
        alt_indexed_file = alt_validation.details.get("indexed_file", None)
        
        # Compare contig information
        try:
            ref_contig_df = self._get_contig_info(ref_indexed_file, ref_validation.file_path)
            alt_contig_df = self._get_contig_info(alt_indexed_file, alt_validation.file_path)

            ref_contigs = set(ref_contig_df.index)
            alt_contigs = set(alt_contig_df.index)

            if ref_contigs != alt_contigs:
                logger.error(f"Contigs differ: ref={list(ref_contigs)}, alt={list(alt_contigs)}")
                self.result.error = f"Contigs differ: ref={list(ref_contigs)}, alt={list(alt_contigs)}"
                return 
            
            # Compute the difference of contig lengths and calculate similarity
            numeric_cols = ['length']
            mae_contig_df = pd.DataFrame(np.abs(ref_contig_df[numeric_cols].values - alt_contig_df[numeric_cols].values), 
                                            index=ref_contig_df.index, columns=numeric_cols)
            # Calculate relative absolute error for lengths
            # If lengths are available (VCF files), use them; otherwise (BED/GFF) just check contig match
            ref_lengths_sum = ref_contig_df[numeric_cols].sum(axis=1)
            if ref_lengths_sum.sum() > 0:
                # VCF files: compare lengths
                rae_contig_mean = np.nanmean(mae_contig_df.sum(axis=1) / ref_lengths_sum)
                similarity = 1.0 - rae_contig_mean
            else:
                # BED/GFF files: contigs match is sufficient (lengths not available)
                similarity = 1.0
            self.result.similarity = similarity
            self.result.details.update({"ref_contig_df": ref_contig_df.to_dict(), "alt_contig_df": alt_contig_df.to_dict()})
        
        except Exception as e:
            logger.error(f"Contig info computation failed: {e}")
            self.result.error = f"Contig info computation failed: {e}"

    def _compare_summary(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Summary statistics comparison: Compare index sizes.
        Return a ComparisonResult with similarity score based on file sizes
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

