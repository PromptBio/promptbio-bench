"""bigWig format handler.

Strategy:
- exact: Exact chromosome and value matching, all values identical (100% required)
- approximate: Value matching with tolerance
- correlation: Pattern-based similarity using correlation coefficient
- summary: Summary statistics comparison of value distributions across chromosomes (default)

Required cmdline tools:
- (None)
"""

import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pyBigWig

from tools.base import FormatHandler
from tools.results import ValidationResult, ComparisonResult
from utils.format import detect_file_signature
from utils.metrics import hellinger_distance, pearson_correlation, spearman_correlation

logger = logging.getLogger(__name__)


class BigWigHandler(FormatHandler):
    """Handler for bigWig and WIG format files."""
    
    def __init__(self):
        super().__init__("bigwig")
        self.format_map = {
            'bigwig': {'mode': 'rb'},
            'wig': {'mode': 'r'},
        }
        self.supported_formats = list(self.format_map.keys())
        self.supported_strategies = ["exact", "approximate", "correlation", "summary"]
    
    # ============================================================================
    # Public API Methods
    # ============================================================================
    
    def validate(self, file_path: str, **kwargs) -> ValidationResult:
        """Validate bigWig/WIG file format.
        
        Args:
            file_path: Path to bigWig/WIG file
            **kwargs: optional parameters (e.g., min_size)
            - min_size: Minimum size of the file in bytes, default is 0
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
            logger.error(f"bigWig/WIG file not found: {file_path}")
            result.error = "File not found"
            return result
        
        if not is_nonempty:
            logger.warning(f"bigWig/WIG file is empty: {file_path}")
            result.error = "File is empty"
            return result
        
        if not is_format_ok:
            logger.error(f"Not a bigWig/WIG file: detected={signature.format}")
            result.error = f"Not a bigWig/WIG file: detected={signature.format}"
            return result
        
        # Try to open and parse the file
        try:
            bw = self._load_file(file_path)
            
            if bw is None:
                result.error = "Failed to open bigWig/WIG file"
                return result
            
            # Get chromosome information and statistics
            chrom_stats = self._get_chrom_stats(bw)
            value_stats = self._get_value_stats(bw)
            
            result.parseable = True
            result.non_empty = chrom_stats["num_chroms"] > 0
            
            result.details.update({
                "chrom_stats": chrom_stats,
                "value_stats": value_stats,
            })
            
            bw.close()
            
            if not result.non_empty:
                logger.warning(f"bigWig/WIG file contains no chromosomes: {file_path}")
                result.error = "File contains no chromosomes"
                return result
            
            logger.info(f"bigWig/WIG validation passed: {chrom_stats['num_chroms']} chromosomes, {chrom_stats['total_bases']} bp total")
            
        except Exception as e:
            logger.error(f"bigWig/WIG validation failed: {e}")
            result.error = f"bigWig/WIG validation failed: {e}"
        return result
    
    def compare(self, reference_path: str, candidate_path: str, strategy: str = "summary", **kwargs) -> ComparisonResult:
        """Compare bigWig/WIG files.
        
        Supported strategies:
        - exact: Exact chromosome and value matching, all values identical (100% required)
        - approximate: Value distribution comparison with tolerance
        - correlation: Pattern-based comparison using correlation coefficient
        - summary: Summary statistics comparison of value distributions across chromosomes (default)
        
        Args:
            reference_path: Path to reference bigWig/WIG file
            candidate_path: Path to candidate bigWig/WIG file
            strategy: Comparison strategy (exact, approximate, correlation, summary)
            **kwargs: Strategy-specific parameters (e.g., tolerance, max_positions, correlation_method)
            - tolerance: Tolerance for approximate comparison (default: 1)
            - max_positions: Maximum number of positions to extract per chromosome (default: None)
            - correlation_method: Correlation method - 'pearson' or 'spearman' (default: 'pearson')
        """
        # initialize comparison result
        self.result = ComparisonResult(reference_path=reference_path, candidate_path=candidate_path, 
                                       strategy=strategy, similarity=np.nan, details={}, error=None)
        
        # validate reference and candidate files
        ref_validation = self.validate(reference_path, **kwargs)
        alt_validation = self.validate(candidate_path, **kwargs)

        try:
            if ref_validation.is_valid and alt_validation.is_valid:
                # compare files using specified strategy
                if strategy == "exact":
                    # Exact comparison: chromosomes and values must match exactly
                    self._compare_exact(ref_validation, alt_validation, **kwargs)
                elif strategy == "approximate":
                    # Approximate comparison: value distribution with tolerance
                    self._compare_approx(ref_validation, alt_validation, **kwargs)
                elif strategy == "correlation":
                    # Correlation comparison: pattern-based similarity using correlation coefficient
                    self._compare_correlation(ref_validation, alt_validation, **kwargs)
                elif strategy == "summary" or strategy == "semantic":
                    # Summary statistics comparison: value distribution comparison across chromosomes
                    self._compare_summary(ref_validation, alt_validation, **kwargs)
                else:
                    logger.warning(f"Unknown strategy '{strategy}', fall back to summary")
                    self._compare_summary(ref_validation, alt_validation, **kwargs)
        except Exception as e:
            logger.error(f"bigWig/WIG comparison failed: {e}")
            self.result.error = f"bigWig/WIG comparison failed: {e}"
        return self._finalize_output(ref_validation, alt_validation, self.result)
    
    # ============================================================================
    # File I/O and Loading Methods
    # ============================================================================
    
    def _load_file(self, file_path: str) -> pyBigWig.pyBigWig:
        """
        Load bigWig/WIG file. Returns pyBigWig file object.
        """
        try:
            bw = pyBigWig.open(file_path)
            return bw
        except Exception as e:
            raise ValueError(f"Could not open bigWig/WIG file: {e}")
    
    # ============================================================================
    # Statistics and Analysis Methods
    # ============================================================================
    
    def _get_chrom_stats(self, bw: pyBigWig.pyBigWig) -> Dict[str, Any]:
        """Get chromosome statistics from bigWig file.
        Returns a dictionary containing chromosome statistics.
        """
        result = {
            "num_chroms": 0,
            "chroms": [],
            "total_bases": 0,
            "chrom_sizes": {},
        }
        try:
            chrom_sizes = bw.chroms()
            chroms = list(chrom_sizes.keys())
            total_bases = sum(chrom_sizes.values())
            
            result.update({
                "num_chroms": len(chroms),
                "chroms": sorted(chroms),
                "total_bases": total_bases,
                "chrom_sizes": {k: v for k, v in chrom_sizes.items()},
            })
        except Exception as e:
            logger.error(f"Could not get chromosome stats: {e}")
        return result
    
    def _get_value_stats(self, bw: pyBigWig.pyBigWig, max_positions: Optional[int] = None) -> Dict[str, Any]:
        """Get value statistics from bigWig file by extracting values from regions.
        
        Args:
            bw: pyBigWig file object
            max_positions: Optional limit on number of positions to extract per chromosome.
                         If None, extracts all positions (default: None).
        
        Returns:
            Dictionary containing value statistics.
        """
        stats = {}
        try:
            chrom_sizes = bw.chroms()
            all_values = []
            
            for chrom, chrom_size in chrom_sizes.items():
                try:
                    # Extract values from the chromosome region
                    # max_positions is a count, so region_end = start (0) + max_positions
                    region_end = min(chrom_size, max_positions) if max_positions is not None else chrom_size
                    values = bw.values(chrom, 0, region_end)
                    if values:
                        all_values.extend(v for v in values if v is not None)
                except Exception as e:
                    logger.debug(f"Could not extract values from chromosome {chrom}: {e}")
                    continue
            
            if all_values:
                stats = {
                    "min_value": float(min(all_values)),
                    "max_value": float(max(all_values)),
                    "avg_value": float(sum(all_values) / len(all_values)),
                    "num_values": len(all_values),
                }
        except Exception as e:
            logger.error(f"Could not get value stats: {e}")
        return stats
    
    def _get_chrom_values(self, bw: pyBigWig.pyBigWig, chrom: str, start: int, end: int, 
                                max_positions: Optional[int] = None) -> List[float]:
        """Get values from a chromosome region.
        
        Args:
            bw: pyBigWig file object
            chrom: Chromosome name
            start: Start position
            end: End position
            max_positions: Optional limit on number of positions to extract. If None, extracts all positions.
            
        Returns:
            List of extracted values
        """
        values = []
        try:
            region_end = min(end, start + max_positions) if max_positions is not None else end
            values = bw.values(chrom, start, region_end)
            if values:
                values = [float(v) for v in values if v is not None]
        except Exception as e:
            logger.error(f"Could not get chromosome values: {e}")
        return values
    
    # ============================================================================
    # Record Comparison Helpers
    # ============================================================================
    
    def _compare_chrom_values(self, ref_values: List[float], alt_values: List[float], 
                              tolerance: float = 0.0) -> float:
        """Compare two chromosome value arrays with optional tolerance.
        
        Args:
            ref_values: Reference values
            alt_values: Candidate values
            tolerance: Tolerance for comparison. If 0.0, performs exact matching.
                      If > 0.0, performs relative difference comparison.
        
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not ref_values or not alt_values:
            return 0.0
        
        ref_arr = np.array(ref_values)
        alt_arr = np.array(alt_values)
        
        if tolerance == 0.0:
            # Exact matching
            if len(ref_arr) != len(alt_arr):
                return 0.0
            matches = sum(1 for r, a in zip(ref_values, alt_values) if r == a)
            return float(matches) / len(ref_values) if ref_values else 0.0
        else:
            # Tolerance-based matching
            if len(ref_arr) == len(alt_arr):
                # Pairwise comparison with tolerance
                diffs = np.abs(ref_arr - alt_arr)
                max_vals = np.maximum(np.abs(ref_arr), np.abs(alt_arr), dtype=np.float64)
                relative_diffs = np.divide(diffs, max_vals, out=np.zeros_like(diffs, dtype=np.float64), where=max_vals > 0)
                within_tolerance = relative_diffs <= tolerance
                return float(np.mean(within_tolerance))
            else:
                # Different lengths, use distribution comparison
                ref_mean = float(np.mean(ref_arr))
                alt_mean = float(np.mean(alt_arr))
                mean_diff = abs(ref_mean - alt_mean)
                max_mean = max(abs(ref_mean), abs(alt_mean), 1.0)
                return 1.0 - (mean_diff / max_mean) if max_mean > 0 else 1.0
    
    # ============================================================================
    # Private Comparison Implementation Methods
    # ============================================================================
    
    def _compare_exact(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Exact comparison: Chromosomes and values must match exactly.
        
        Criteria:
        - All chromosomes must match exactly
        - All chromosome sizes must match exactly
        - All sampled values must match exactly
        - Must be 100% identical for similarity = 1.0
        
        Args:
            ref_validation: Validation result for reference file
            alt_validation: Validation result for candidate file
            **kwargs: Strategy-specific parameters
            - max_positions: Maximum number of positions to extract per chromosome (default: None)
        """
        
        ref_bw = None
        alt_bw = None
        
        try:
            ref_bw = self._load_file(ref_validation.file_path)
            alt_bw = self._load_file(alt_validation.file_path)
            
            ref_chroms = ref_bw.chroms()
            alt_chroms = alt_bw.chroms()
            ref_chrom_names = set(ref_chroms.keys())
            alt_chrom_names = set(alt_chroms.keys())
            common_chroms = ref_chrom_names & alt_chrom_names
            
            # Exact comparison: chromosomes and sizes must match exactly
            if ref_chrom_names != alt_chrom_names:
                self.result.similarity = 0.0
                self.result.details.update({
                    "ref_chroms": len(ref_chrom_names),
                    "alt_chroms": len(alt_chrom_names),
                    "common_chroms": len(common_chroms),
                })
                return
            
            # Compare sizes and values exactly
            max_positions = kwargs.get('max_positions', None)
            size_matches = 0
            value_similarities = []
            total_positions = 0
            
            for chrom in common_chroms:
                ref_size = ref_chroms[chrom]
                alt_size = alt_chroms[chrom]
                
                if ref_size != alt_size:
                    continue  # Size mismatch, skip this chromosome
                
                size_matches += 1
                
                # Get sampled values using helper method
                ref_values = self._get_chrom_values(ref_bw, chrom, 0, ref_size, max_positions)
                alt_values = self._get_chrom_values(alt_bw, chrom, 0, alt_size, max_positions)
                
                # Compare values exactly (tolerance=0.0)
                if ref_values and alt_values:
                    chrom_similarity = self._compare_chrom_values(ref_values, alt_values, tolerance=0.0)
                    value_similarities.append(chrom_similarity)
                    total_positions += len(ref_values)
            
            size_similarity = size_matches / len(common_chroms) if common_chroms else 0.0
            value_similarity = sum(value_similarities) / len(value_similarities) if value_similarities else 0.0
            # For exact comparison, all must match perfectly
            similarity = 1.0 if (size_similarity == 1.0 and value_similarity == 1.0) else 0.0
            
            self.result.similarity = similarity
            self.result.details.update({
                "ref_chroms": len(ref_chrom_names),
                "alt_chroms": len(alt_chrom_names),
                "size_matches": size_matches,
                "value_similarity": value_similarity,
                "total_positions": total_positions,
                "exact_match": (similarity == 1.0),
            })
        except Exception as e:
            logger.error(f"Exact comparison failed: {e}")
            self.result.error = f"Exact comparison failed: {e}"
        finally:
            if ref_bw:
                ref_bw.close()
            if alt_bw:
                alt_bw.close()
    
    def _compare_approx(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Approximate comparison: Value distribution comparison with tolerance.
        
        Similar to summary statistics comparison but allows for value differences within tolerance.
        Uses tolerance parameter to determine if values are considered matching.
        
        Args:
            ref_validation: Validation result for reference file
            alt_validation: Validation result for candidate file
            **kwargs: Strategy-specific parameters
            - tolerance: Tolerance for value comparison (default: 1)
            - max_positions: Maximum number of positions to extract per chromosome (default: None)
        """
        tolerance = kwargs.get('tolerance', 1)
        
        ref_bw = None
        alt_bw = None
        
        try:
            ref_bw = self._load_file(ref_validation.file_path)
            alt_bw = self._load_file(alt_validation.file_path)
            
            ref_chroms = ref_bw.chroms()
            alt_chroms = alt_bw.chroms()
            ref_chrom_names = set(ref_chroms.keys())
            alt_chrom_names = set(alt_chroms.keys())
            common_chroms = ref_chrom_names & alt_chrom_names
            
            if not common_chroms:
                self.result.similarity = 0.0
                self.result.details.update({
                    "ref_chroms": len(ref_chrom_names),
                    "alt_chroms": len(alt_chrom_names),
                    "common_chroms": 0,
                })
                self.result.error = "No common chromosomes"
                return
            
            # Sample values from common chromosomes and compare with tolerance
            max_positions = kwargs.get('max_positions', None)
            value_similarities = []
            chrom_coverage = []
            
            for chrom in common_chroms:
                ref_size = ref_chroms[chrom]
                alt_size = alt_chroms[chrom]
                size = min(ref_size, alt_size)
                
                if size == 0:
                    continue
                
                # Get sampled values using helper method
                ref_values = self._get_chrom_values(ref_bw, chrom, 0, size, max_positions)
                alt_values = self._get_chrom_values(alt_bw, chrom, 0, size, max_positions)
                
                if ref_values and alt_values:
                    # Compare values with tolerance using helper function
                    chrom_similarity = self._compare_chrom_values(ref_values, alt_values, tolerance=tolerance)
                    value_similarities.append(chrom_similarity)
                    chrom_coverage.append(size)
            
            if not value_similarities:
                self.result.similarity = 0.0
                self.result.details.update({
                    "ref_chroms": len(ref_chrom_names),
                    "alt_chroms": len(alt_chrom_names),
                    "common_chroms": len(common_chroms),
                })
                self.result.error = "No comparable values found"
                return
            
            # Weight similarity by chromosome coverage
            total_coverage = sum(chrom_coverage)
            if total_coverage > 0:
                weighted_similarity = sum(
                    sim * cov for sim, cov in zip(value_similarities, chrom_coverage)
                ) / total_coverage
            else:
                weighted_similarity = sum(value_similarities) / len(value_similarities) if value_similarities else 0.0
            
            # Account for chromosome coverage
            coverage_ratio = len(common_chroms) / len(ref_chrom_names) if len(ref_chrom_names) > 0 else 0.0
            similarity = weighted_similarity * coverage_ratio
            
            self.result.similarity = similarity
            self.result.details.update({
                "ref_chroms": len(ref_chrom_names),
                "alt_chroms": len(alt_chrom_names),
                "common_chroms": len(common_chroms),
                "coverage_ratio": coverage_ratio,
                "avg_value_similarity": weighted_similarity,
                "tolerance": tolerance,
            })
        except Exception as e:
            logger.error(f"Approximate comparison failed: {e}")
            self.result.error = f"Approximate comparison failed: {e}"
        finally:
            if ref_bw:
                ref_bw.close()
            if alt_bw:
                alt_bw.close()

    def _compare_correlation(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Correlation comparison: Pattern-based similarity using correlation coefficient.
        
        Compares value patterns across common chromosomes using Pearson or Spearman correlation.
        Measures how values co-vary (pattern similarity) rather than absolute value agreement.
        Scale-invariant: signals with same pattern but different scales will have high correlation.
        
        Args:
            ref_validation: Validation result for reference file
            alt_validation: Validation result for candidate file
            **kwargs: Strategy-specific parameters
            - max_positions: Maximum number of positions to extract per chromosome (default: None)
            - correlation_method: Correlation method - 'pearson' or 'spearman' (default: 'pearson')
        """
        
        ref_bw = None
        alt_bw = None
        
        try:
            ref_bw = self._load_file(ref_validation.file_path)
            alt_bw = self._load_file(alt_validation.file_path)
            
            ref_chroms = ref_bw.chroms()
            alt_chroms = alt_bw.chroms()
            ref_chrom_names = set(ref_chroms.keys())
            alt_chrom_names = set(alt_chroms.keys())
            common_chroms = ref_chrom_names & alt_chrom_names
            
            if not common_chroms:
                self.result.similarity = 0.0
                self.result.details.update({
                    "ref_chroms": len(ref_chrom_names),
                    "alt_chroms": len(alt_chrom_names),
                    "common_chroms": 0,
                })
                self.result.error = "No common chromosomes"
                return
            
            max_positions = kwargs.get('max_positions', None)
            correlation_method = kwargs.get('correlation_method', 'pearson').lower()
            
            if correlation_method not in ['pearson', 'spearman']:
                logger.warning(f"Unknown correlation method '{correlation_method}', using 'pearson'")
                correlation_method = 'pearson'
            
            # Collect values per chromosome and calculate correlations
            chrom_correlations = []
            chrom_coverage = []
            
            for chrom in common_chroms:
                size = min(ref_chroms[chrom], alt_chroms[chrom])
                if size == 0:
                    continue
                
                # Get values from both files
                ref_values = self._get_chrom_values(ref_bw, chrom, 0, size, max_positions)
                alt_values = self._get_chrom_values(alt_bw, chrom, 0, size, max_positions)
                
                if not ref_values or not alt_values:
                    continue
                
                # Ensure same length for correlation (use minimum length)
                min_len = min(len(ref_values), len(alt_values))
                if min_len < 2:
                    # Need at least 2 points for correlation
                    continue
                
                ref_arr = np.array(ref_values[:min_len])
                alt_arr = np.array(alt_values[:min_len])
                
                # Remove NaN/inf values (pairwise deletion)
                valid_mask = np.isfinite(ref_arr) & np.isfinite(alt_arr)
                if np.sum(valid_mask) < 2:
                    # Need at least 2 valid points for correlation
                    continue
                
                ref_valid = ref_arr[valid_mask]
                alt_valid = alt_arr[valid_mask]
                
                # Check for constant values (correlation undefined)
                if np.std(ref_valid) == 0 or np.std(alt_valid) == 0:
                    # If both are constant and equal, correlation is 1.0
                    if np.allclose(ref_valid, alt_valid):
                        chrom_corr = 1.0
                    else:
                        # Constant but different - correlation undefined, skip
                        continue
                else:
                    # Calculate correlation using metrics functions
                    if correlation_method == 'pearson':
                        chrom_corr = pearson_correlation(ref_valid, alt_valid, nan_policy='omit')
                    else:  # spearman
                        chrom_corr = spearman_correlation(ref_valid, alt_valid, nan_policy='omit')
                
                # Handle NaN correlation (shouldn't happen after checks, but just in case)
                if np.isnan(chrom_corr):
                    continue
                
                # Convert correlation (-1 to 1) to similarity (0 to 1)
                # Using absolute value: negative correlation = 0 similarity
                # Alternative: (corr + 1) / 2 to preserve direction
                chrom_similarity = abs(chrom_corr)
                
                chrom_correlations.append(chrom_similarity)
                chrom_coverage.append(size)
            
            if not chrom_correlations:
                self.result.similarity = 0.0
                self.result.details.update({
                    "ref_chroms": len(ref_chrom_names),
                    "alt_chroms": len(alt_chrom_names),
                    "common_chroms": len(common_chroms),
                })
                self.result.error = "No comparable values found for correlation"
                return
            
            # Calculate weighted average correlation by chromosome coverage
            total_coverage = sum(chrom_coverage)
            if total_coverage > 0:
                weighted_correlation = sum(
                    corr * cov for corr, cov in zip(chrom_correlations, chrom_coverage)
                ) / total_coverage
            else:
                weighted_correlation = np.mean(chrom_correlations) if chrom_correlations else 0.0
            
            # Account for chromosome coverage ratio
            coverage_ratio = len(common_chroms) / len(ref_chrom_names) if ref_chrom_names else 0.0
            similarity = weighted_correlation * coverage_ratio
            
            self.result.similarity = similarity
            self.result.details.update({
                "ref_chroms": len(ref_chrom_names),
                "alt_chroms": len(alt_chrom_names),
                "common_chroms": len(common_chroms),
                "coverage_ratio": coverage_ratio,
                "avg_correlation": weighted_correlation,
                "correlation_method": correlation_method,
                "num_chroms_compared": len(chrom_correlations),
            })
        except Exception as e:
            logger.error(f"Correlation comparison failed: {e}")
            self.result.error = f"Correlation comparison failed: {e}"
        finally:
            if ref_bw:
                ref_bw.close()
            if alt_bw:
                alt_bw.close()

    def _compare_summary(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Summary statistics comparison: Value distribution comparison across chromosomes.
        
        Compares value distributions across common chromosomes using Hellinger distance.
        
        Args:
            ref_validation: Validation result for reference file
            alt_validation: Validation result for candidate file
            **kwargs: Strategy-specific parameters
            - max_positions: Maximum number of positions to extract per chromosome (default: None)
            - num_bins: Number of bins for histogram (default: 50)
        """
        
        ref_bw = None
        alt_bw = None
        
        try:
            ref_bw = self._load_file(ref_validation.file_path)
            alt_bw = self._load_file(alt_validation.file_path)
            
            ref_chroms = ref_bw.chroms()
            alt_chroms = alt_bw.chroms()
            ref_chrom_names = set(ref_chroms.keys())
            alt_chrom_names = set(alt_chroms.keys())
            common_chroms = ref_chrom_names & alt_chrom_names
            
            if not common_chroms:
                self.result.similarity = 0.0
                self.result.details.update({
                    "ref_chroms": len(ref_chrom_names),
                    "alt_chroms": len(alt_chrom_names),
                    "common_chroms": 0,
                })
                self.result.error = "No common chromosomes"
                return
            
            max_positions = kwargs.get('max_positions', None)
            num_bins = kwargs.get('num_bins', 50)
            
            # Collect values per chromosome (cache to avoid re-reading)
            chrom_data = {}
            all_ref_values, all_alt_values = [], []
            
            for chrom in common_chroms:
                size = min(ref_chroms[chrom], alt_chroms[chrom])
                if size == 0:
                    continue
                
                ref_values = self._get_chrom_values(ref_bw, chrom, 0, size, max_positions)
                alt_values = self._get_chrom_values(alt_bw, chrom, 0, size, max_positions)
                
                if ref_values and alt_values:
                    chrom_data[chrom] = (ref_values, alt_values, size)
                    all_ref_values.extend(ref_values)
                    all_alt_values.extend(alt_values)
            
            if not all_ref_values or not all_alt_values:
                self.result.similarity = 0.0
                self.result.details.update({
                    "ref_chroms": len(ref_chrom_names),
                    "alt_chroms": len(alt_chrom_names),
                    "common_chroms": len(common_chroms),
                })
                self.result.error = "No comparable values found"
                return
            
            # Determine common bin range
            all_values = all_ref_values + all_alt_values
            min_val, max_val = float(np.min(all_values)), float(np.max(all_values))
            
            if min_val == max_val:
                coverage_ratio = len(common_chroms) / len(ref_chrom_names) if ref_chrom_names else 0.0
                self.result.similarity = 1.0
                self.result.details.update({
                    "ref_chroms": len(ref_chrom_names),
                    "alt_chroms": len(alt_chrom_names),
                    "common_chroms": len(common_chroms),
                    "coverage_ratio": coverage_ratio,
                    "hellinger_distance": 0.0,
                })
                return
            
            # Calculate per-chromosome similarities using cached values
            value_similarities, chrom_coverage = [], []
            for ref_values, alt_values, size in chrom_data.values():
                ref_hist, _ = np.histogram(ref_values, bins=num_bins, range=(min_val, max_val))
                alt_hist, _ = np.histogram(alt_values, bins=num_bins, range=(min_val, max_val))
                distance = hellinger_distance(ref_hist, alt_hist)
                value_similarities.append(1.0 - (distance if not np.isnan(distance) else 1.0))
                chrom_coverage.append(size)
            
            if not value_similarities:
                self.result.similarity = 0.0
                self.result.details.update({
                    "ref_chroms": len(ref_chrom_names),
                    "alt_chroms": len(alt_chrom_names),
                    "common_chroms": len(common_chroms),
                })
                self.result.error = "No comparable values found"
                return
            
            # Calculate weighted similarity
            total_coverage = sum(chrom_coverage)
            weighted_similarity = (sum(sim * cov for sim, cov in zip(value_similarities, chrom_coverage)) / total_coverage
                                 if total_coverage > 0 else np.mean(value_similarities))
            
            coverage_ratio = len(common_chroms) / len(ref_chrom_names) if ref_chrom_names else 0.0
            similarity = weighted_similarity * coverage_ratio
            
            # Overall Hellinger distance for reporting
            ref_hist_all, _ = np.histogram(all_ref_values, bins=num_bins, range=(min_val, max_val))
            alt_hist_all, _ = np.histogram(all_alt_values, bins=num_bins, range=(min_val, max_val))
            overall_hellinger_distance = hellinger_distance(ref_hist_all, alt_hist_all)
            
            self.result.similarity = similarity
            self.result.details.update({
                "ref_chroms": len(ref_chrom_names),
                "alt_chroms": len(alt_chrom_names),
                "common_chroms": len(common_chroms),
                "coverage_ratio": coverage_ratio,
                "avg_value_similarity": weighted_similarity,
                "hellinger_distance": float(overall_hellinger_distance) if not np.isnan(overall_hellinger_distance) else 1.0,
                "num_bins": num_bins,
            })
        except Exception as e:
            logger.error(f"Summary statistics comparison failed: {e}")
            self.result.error = f"Summary statistics comparison failed: {e}"
        finally:
            if ref_bw:
                ref_bw.close()
            if alt_bw:
                alt_bw.close()
