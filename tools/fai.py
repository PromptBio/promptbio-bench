"""FASTA index file handler (.fai).

Strategy:
- exact: Compare index entries exactly (all fields must match)
- approximate: Compare index entries approximately (all fields must match)
- summary: Summary statistics comparison of index file format, validity, and sizes

Required cmdline tools:
- (None)
"""

import logging
import os

import numpy as np
import pandas as pd

from tools.base import FormatHandler
from tools.results import ValidationResult, ComparisonResult
from utils.format import detect_file_signature
from utils.metrics import hellinger_distance, rae_similarity, relative_absolute_error

logger = logging.getLogger(__name__)


class FAIHandler(FormatHandler):
    """Handler for FASTA index files (.fai)."""
    
    def __init__(self):
        super().__init__("fai")
        self.supported_formats = ['fai']
        self.supported_strategies = ["exact", "approximate", "summary"]
    
    # ============================================================================
    # Public API Methods
    # ============================================================================
    
    def validate(self, file_path: str, **kwargs) -> ValidationResult:
        """Validate FASTA index file (.fai).
        non-empty check is performed by file size, later can be updated by reading content if needed
        
        Args:
            file_path: Path to FASTA index file (.fai)
            **kwargs: optional parameters (e.g., min_size)
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
            logger.error(f"FASTA index file not found: {file_path}")
            result.error = "File not found"
            return result
        
        if not is_nonempty:
            logger.warning(f"FASTA index file is empty: {file_path}")
            result.error = "File is empty"
            return result
        
        if not is_format_ok:
            logger.error(f"Not a FASTA index file: detected={signature.format}")
            result.error = f"Not a FASTA index file: detected={signature.format}"
            return result
        
        # validate by parsing the index file
        try:
            df = self._load_file(file_path)
            result.parseable = True
            result.non_empty = df.shape[0] > 0
            result.details.update({
                "num_sequences": df.shape[0],
                "total_length": df["length"].sum(),
            })
            logger.info(f"FASTA index file validation passed: {file_path} (shape: {df.shape})")
        except Exception as e:
            logger.error(f"FASTA index file validation failed: {e}")
            result.error = f"FASTA index file validation failed: {e}"
        return result
    
    def compare(self, reference_path: str, candidate_path: str, strategy: str = "exact", **kwargs) -> ComparisonResult:
        """Compare FASTA index files (.fai).
        
        Supported strategies:
        - exact: Compare index entries exactly (all fields must match)
        - approximate: Compare index entries approximately (all fields must match)
        - summary: Summary statistics comparison of index file format, validity, and sizes
        
        Args:
            reference_path: Path to reference index file
            candidate_path: Path to candidate index file
            strategy: Comparison strategy (exact, approximate, summary)
            **kwargs: Strategy-specific parameters
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
                    # Compare index entries exactly (all fields must match)
                    self._compare_exact(ref_validation, alt_validation, **kwargs)
                elif strategy == "approximate":
                    # Compare index entries approximately (all fields must match)
                    self._compare_approx(ref_validation, alt_validation, **kwargs)
                elif strategy == "summary":
                    # Summary statistics comparison: compare index file format, validity, and sizes
                    self._compare_summary(ref_validation, alt_validation, **kwargs)
                else:
                    logger.warning(f"Unknown strategy '{strategy}', fall back to exact")
                    self._compare_exact(ref_validation, alt_validation, **kwargs)
        except Exception as e:
            logger.error(f"FASTA index file comparison failed: {e}")
            self.result.error = f"FASTA index file comparison failed: {e}"
        return self._finalize_output(ref_validation, alt_validation, self.result)
    
    # ============================================================================
    # File I/O and Loading Methods
    # ============================================================================
    
    def _load_file(self, file_path: str) -> pd.DataFrame:
        """Load FASTA index (.fai) file into a pandas DataFrame.
        
        Returns a DataFrame with columns: sequence_name, length, offset, linebases, linewidth
        """
        try:
            df = pd.read_csv(file_path, sep='\t', header=None, 
                names=['sequence_name', 'length', 'offset', 'linebases', 'linewidth'],
                dtype={'sequence_name': str, 'length': int, 'offset': int, 'linebases': int, 'linewidth': int},
                comment=None,  # Don't treat any character as comment
                skip_blank_lines=True
            )
            return df
        except pd.errors.EmptyDataError:
            logger.error(f"FASTA index file is empty: {file_path}")
            raise ValueError(f"FASTA index file is empty: {file_path}")
        except (ValueError, pd.errors.ParserError) as e:
            logger.error(f"Invalid .fai format: {e}")
            raise ValueError(f"Invalid .fai format: {e}")
    
    # ============================================================================
    # File Status Checkers
    # ============================================================================
    
    # ============================================================================
    # Statistics and Analysis Methods
    # ============================================================================
    
    def _get_length_distribution(self, ref_lengths: np.ndarray, alt_lengths: np.ndarray, num_bins: int = 10):
        """Calculate histograms for length distributions.
        
        Args:
            ref_lengths: Array of reference sequence lengths
            alt_lengths: Array of candidate sequence lengths
            num_bins: Number of bins for histogram (default: 10)
            
        Returns:
            Tuple of (ref_hist, alt_hist) or None if empty arrays
        """
        # Handle empty arrays
        if len(ref_lengths) == 0 or len(alt_lengths) == 0:
            return None
        
        # Both non-empty - find common range for bins
        all_lengths = np.concatenate([ref_lengths, alt_lengths])
        min_length = all_lengths.min()
        max_length = all_lengths.max()
        
        if max_length > min_length:
            # Create bins
            bins = np.linspace(min_length, max_length, num_bins + 1)
            
            # Calculate histograms
            ref_hist, _ = np.histogram(ref_lengths, bins=bins)
            alt_hist, _ = np.histogram(alt_lengths, bins=bins)
            
            return (ref_hist, alt_hist)
        else:
            # All sequences have the same length - return None (handled separately)
            return None
    
    # ============================================================================
    # Private Comparison Implementation Methods
    # ============================================================================
    
    def _compare_exact(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Exact comparison: Check if the two DataFrames are exactly the same.
        
        Compares if both DataFrames have the same shape, same sequence names,
        and all values match exactly. Returns 1.0 if identical, 0.0 otherwise.
        """
        try:
            # Load both index files as DataFrames
            ref_df = self._load_file(ref_validation.file_path)
            alt_df = self._load_file(alt_validation.file_path)
            
            # Reset index to ensure consistent comparison
            ref_df = ref_df.set_index('sequence_name').sort_index()
            alt_df = alt_df.set_index('sequence_name').sort_index()
            
            # Check if DataFrames are exactly the same
            similarity = 1.0 if ref_df.equals(alt_df) else 0.0
            
            self.result.similarity = similarity
            self.result.details.update({
                "total_ref_sequences": len(ref_df),
                "total_alt_sequences": len(alt_df),
                "common_sequences": len(ref_df.index & alt_df.index),
                "missing_sequences": len(ref_df.index - alt_df.index),
                "extra_sequences": len(alt_df.index - ref_df.index),
            })
        except Exception as e:
            logger.error(f"Exact comparison failed: {e}")
            self.result.error = f"Exact comparison failed: {e}"
    
    def _compare_approx(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Approximate comparison: Compare using Relative Absolute Error (RAE) per cell.
        
        Compares:
        1. Aligns DataFrames using ref_df as reference
        2. For sequences in ref_df but not in alt_df, assigns large RAE values
        3. Calculates RAE = |ref - alt| / ref for each cell
        4. Applies tolerance threshold to determine similar values
        5. Returns proportion of values within tolerance
        
        Args:
            **kwargs: Strategy-specific parameters
            - tolerance: Tolerance threshold for RAE (default: 0.0, meaning any difference counts)
        """
        try:
            tolerance = kwargs.get('tolerance', 0.0)
            max_rae = kwargs.get('max_rae', 1e6)
            
            # Load both index files as DataFrames
            ref_df = self._load_file(ref_validation.file_path)
            alt_df = self._load_file(alt_validation.file_path)
            
            # Set sequence_name as index
            ref_df = ref_df.set_index('sequence_name')
            alt_df = alt_df.set_index('sequence_name')
            
            ref_names = set(ref_df.index)
            alt_names = set(alt_df.index)
            
            # Align alt_df to ref_df (use ref_df as reference)
            alt_aligned = alt_df.reindex(ref_df.index)
            
            # Select numeric columns (currently only 'length')
            if 'length' not in ref_df.columns:
                raise ValueError("'length' column not found in reference file")
            ref_numeric = ref_df[['length']]
            alt_numeric = alt_aligned[['length']]
            
            # Identify missing sequences
            missing_mask = alt_numeric.isna()
            missing_cells = missing_mask.sum().sum()
            
            # Calculate similarity using rae_similarity
            # Pass NaN values directly (they will propagate and be handled separately)
            sim_array = rae_similarity(ref_numeric.values, alt_numeric.values, clip=True)
            
            # Handle missing sequences: set to 0.0 similarity (very different)
            sim_array[missing_mask.values] = 0.0
            
            # Convert tolerance (RAE threshold) to similarity threshold
            # RAE <= tolerance means similarity >= (1 - tolerance)
            similarity_threshold = 1.0 - tolerance
            
            # Apply similarity threshold: values with similarity >= threshold are considered similar
            similar_mask = sim_array >= similarity_threshold
            similar_count = similar_mask.sum()
            total_cells = sim_array.size
            similarity = similar_count / total_cells if total_cells > 0 else 0.0
            
            self.result.similarity = similarity
            self.result.details.update({
                "total_ref_sequences": len(ref_names),
                "total_alt_sequences": len(alt_names),
                "common_sequences": len(ref_names & alt_names),
                "missing_sequences": len(ref_names - alt_names),
                "extra_sequences": len(alt_names - ref_names),
                "total_cells": total_cells,
                "similar_cells": int(similar_count),
                "missing_cells": missing_cells,
                "tolerance": tolerance
            })
        except Exception as e:
            logger.error(f"Approximate comparison failed: {e}")
            self.result.error = f"Approximate comparison failed: {e}"
    
    def _compare_summary(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Summary statistics comparison: Compare number of sequences and length distribution.
        
        Compares:
        1. Number of sequences
        2. Distribution of sequence lengths (using histogram/bins or statistical measures)
        
        Returns similarity based on how similar these statistics are.
        
        Args:
            **kwargs: Strategy-specific parameters
            - num_bins: Number of bins for length distribution histogram (default: 10)
        """
        try:
            num_bins = kwargs.get('num_bins', 10)
            
            # Load both index files as DataFrames
            ref_df = self._load_file(ref_validation.file_path)
            alt_df = self._load_file(alt_validation.file_path)
            
            # Extract sequence lengths
            ref_lengths = ref_df["length"].values
            alt_lengths = alt_df["length"].values
            
            # Compare number of sequences using RAE similarity
            ref_num = len(ref_df)
            alt_num = len(alt_df)
            num_sequences_sim = float(rae_similarity([ref_num], [alt_num])[0])
            
            # Compare total length similarity using RAE similarity
            ref_total_length = ref_df["length"].sum()
            alt_total_length = alt_df["length"].sum()
            total_length_sim = float(rae_similarity([ref_total_length], [alt_total_length])[0])

            # Compare length distribution using histogram
            # Handle empty arrays
            if len(ref_lengths) == 0 and len(alt_lengths) == 0:
                # Both empty - perfect match
                distribution_sim = 1.0
            elif len(ref_lengths) == 0 or len(alt_lengths) == 0:
                # One empty, one not - no match
                distribution_sim = 0.0
            else:
                # Get histograms
                hist_lists = self._get_length_distribution(ref_lengths, alt_lengths, num_bins)
                
                if hist_lists is None:
                    # All sequences have the same length - check if both have the same mean
                    distribution_sim = 1.0 if ref_lengths.mean() == alt_lengths.mean() else 0.0
                else:
                    ref_hist, alt_hist = hist_lists
                    # Calculate Hellinger distance (function handles normalization internally)
                    # Convert distance to similarity: similarity = 1 - distance
                    hellinger_dist = hellinger_distance(ref_hist, alt_hist)
                    distribution_sim = 1.0 - hellinger_dist if not np.isnan(hellinger_dist) else 0.0
            
            # Combined similarity (equal weights: number of sequences, total length, distribution)
            similarity = (num_sequences_sim + total_length_sim + distribution_sim) / 3
            
            self.result.similarity = similarity
            self.result.details.update({
                "ref_num_sequences": ref_num,
                "alt_num_sequences": alt_num,
                "num_sequences_similarity": num_sequences_sim,
                "total_length_similarity": total_length_sim,
                "distribution_similarity": distribution_sim,
                "ref_length_mean": float(ref_lengths.mean()),
                "alt_length_mean": float(alt_lengths.mean()),
                "ref_length_std": float(ref_lengths.std()),
                "alt_length_std": float(alt_lengths.std()),
                "ref_length_min": int(ref_lengths.min()),
                "alt_length_min": int(alt_lengths.min()),
                "ref_length_max": int(ref_lengths.max()),
                "alt_length_max": int(alt_lengths.max())
            })
        except Exception as e:
            logger.error(f"Summary statistics comparison failed: {e}")
            self.result.error = f"Summary statistics comparison failed: {e}"

