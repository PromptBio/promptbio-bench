"""FASTQ format handler.

Strategy:
- exact: Read ID, sequence, and quality scores identical (100% required)
- approximate: ≥95% reads match, trim positions within ±2bp, quality ±2 Phred
- summary: Summary statistics comparison of read count, total length, and k-mer frequency distribution

Required cmdline tools:
- (None)
"""

import logging
import gzip
from collections import Counter
from itertools import product
from typing import Dict, List, Tuple, Any

import numpy as np

from Bio import SeqIO

from tools.base import FormatHandler
from tools.results import ValidationResult, ComparisonResult
from utils.format import detect_file_signature
from utils.metrics import hellinger_distance, rae_similarity

logger = logging.getLogger(__name__)


class FASTQHandler(FormatHandler):
    """Handler for FASTQ format files."""
    
    def __init__(self):
        super().__init__("fastq")
        self.supported_formats = ['fastq', 'fq']
        self.supported_strategies = ["exact", "approximate", "summary"]
        self.format_map = {
            'none': {'open_func': open, 'mode': 'rt'},
            'gzip': {'open_func': gzip.open, 'mode': 'rt'},
            'bgzf': {'open_func': gzip.open, 'mode': 'rt'},
        }
    
    # ============================================================================
    # Public API Methods
    # ============================================================================
    
    def validate(self, file_path: str, **kwargs) -> ValidationResult:
        """Validate FASTQ file format.
        non-empty check is performed by file size, later can be updated by reading content if needed
        
        Args:
            file_path: Path to FASTQ file
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
            logger.error(f"FASTQ file not found: {file_path}")
            result.error = "File not found"
            return result
        
        if not is_nonempty:
            logger.warning(f"FASTQ file is empty: {file_path}")
            result.error = "File is empty"
            return result
        
        if not is_format_ok:
            logger.error(f"Not a FASTQ file: detected={signature.format}")
            result.error = f"Not a FASTQ file: detected={signature.format}"
            return result
        
        # validate by parsing the FASTQ file
        try:
            reads_dict = self._load_file(file_path)
            read_count = len(reads_dict)
            
            result.parseable = True
            result.non_empty = read_count > 0
            
            if not result.non_empty:
                logger.warning(f"FASTQ file contains no reads: {file_path}")
                result.error = "FASTQ file contains no reads"
                result.details.update({"num_reads": 0})
                return result
            
            result.details.update({
                "num_reads": read_count
            })
            
            logger.info(f"FASTQ validation passed: {read_count} reads")
        except Exception as e:
            logger.error(f"FASTQ validation failed: {e}")
            result.error = f"FASTQ validation failed: {e}"
        return result
    
    def compare(self, reference_path: str, candidate_path: str, strategy: str = "exact", **kwargs) -> ComparisonResult:
        """Compare FASTQ files.
        
        Supported strategies:
        - exact: Read ID, sequence, and quality scores identical (100% required)
        - approximate: ≥95% reads match, trim positions within ±2bp, quality ±2 Phred
        - summary: K-mer frequency distribution comparison
        
        Args:
            reference_path: Path to reference FASTQ file
            candidate_path: Path to candidate FASTQ file
            strategy: Comparison strategy (exact, approximate, summary)
            **kwargs: Strategy-specific parameters
            - For approximate strategy:
              - trim_tolerance: Positional tolerance in base pairs for trimming differences (default: 2)
              - quality_tolerance: Phred score tolerance (default: 2)
            - For summary strategy:
              - k: K-mer size for k-mer comparison (default: 11)
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
                    # Read ID, sequence, and quality scores identical (100% required)
                    self._compare_exact(ref_validation, alt_validation, **kwargs)
                elif strategy == "approximate":
                    # ≥95% reads match, trim positions within ±2bp, quality ±2 Phred
                    self._compare_approx(ref_validation, alt_validation, **kwargs)
                elif strategy == "summary":
                    # Summary statistics comparison: overlap ratio and distribution-based comparison
                    self._compare_summary(ref_validation, alt_validation, **kwargs)
                else:
                    logger.warning(f"Unknown strategy '{strategy}', fall back to summary")
                    self._compare_summary(ref_validation, alt_validation, **kwargs)
        except Exception as e:
            logger.error(f"FASTQ comparison failed: {e}")
            self.result.error = f"FASTQ comparison failed: {e}"
        return self._finalize_output(ref_validation, alt_validation, self.result)
    
    # ============================================================================
    # File I/O and Loading Methods
    # ============================================================================
    
    def _load_file(self, file_path: str) -> Dict[str, Dict[str, Any]]:
        """Load FASTQ file into a dictionary, handling compression.
        
        Args:
            file_path: Path to FASTQ file (supports gzip/bgzf compression)
        
        Returns:
            Dictionary mapping read ID to dict with 'seq' and 'qual' keys:
            {read_id: {"seq": sequence_string, "qual": quality_scores_list}}
        """
        signature = detect_file_signature(file_path, fallback=True)
        compression = signature.compression if signature.compression in self.format_map else 'none'
        open_config = self.format_map[compression]
        reads_dict = {}
        with open_config['open_func'](file_path, open_config['mode'], encoding='utf-8', errors='replace') as handle:
            for record in SeqIO.parse(handle, "fastq"):
                qual_list = record.letter_annotations.get("phred_quality", [])
                reads_dict[record.id] = {
                    "seq": str(record.seq),
                    "qual": np.array(qual_list) if qual_list else np.array([], dtype=int)
                }
        return reads_dict
    
    # ============================================================================
    # Statistics and Analysis Methods
    # ============================================================================
    
    def _get_summary_stats(self, file_path: str) -> Dict:
        """Compute summary statistics for a FASTQ file.
        
        Args:
            file_path: Path to FASTQ file
            
        Returns:
            Dictionary with summary statistics:
            - num_reads: Number of reads
            - total_length: Total length of all reads
            - avg_length: Average read length
            - min_length: Minimum read length
            - max_length: Maximum read length
            - median: Median read length (50th percentile)
            - q25: 25th percentile read length
            - q75: 75th percentile read length
            - avg_quality: Average quality score
        """
        reads_dict = self._load_file(file_path)
        lengths_arr = np.array([len(read_data["seq"]) for read_data in reads_dict.values()])
        
        # Calculate average quality: mean per read, then average across reads
        quality_means = [np.mean(read_data["qual"]) for read_data in reads_dict.values() if len(read_data["qual"]) > 0]
        avg_quality = float(np.mean(quality_means)) if quality_means else 0.0
        
        return {
            "num_reads": len(reads_dict),
            "total_length": int(lengths_arr.sum()) if len(lengths_arr) > 0 else 0,
            "length_arr": lengths_arr,
            "length_mean": float(lengths_arr.mean()) if len(lengths_arr) > 0 else 0.0,
            "length_std": float(lengths_arr.std()) if len(lengths_arr) > 0 else 0.0,
            "length_min": int(lengths_arr.min()) if len(lengths_arr) > 0 else 0,
            "length_max": int(lengths_arr.max()) if len(lengths_arr) > 0 else 0,
            "length_median": float(np.median(lengths_arr)) if len(lengths_arr) > 0 else 0.0,
            "length_q25": float(np.quantile(lengths_arr, 0.25)) if len(lengths_arr) > 0 else 0.0,
            "length_q75": float(np.quantile(lengths_arr, 0.75)) if len(lengths_arr) > 0 else 0.0,
            "avg_quality": avg_quality,
        }
    
    def _get_kmer_counter(self, file_path: str, k: int = 11) -> Counter:
        """Extract k-mers from all reads in a FASTQ file.
        Args:
            file_path: Path to FASTQ file
            k: K-mer size
            
        Return a Counter object mapping k-mer to frequency
        """
        kmer_counter = Counter()
        
        reads_dict = self._load_file(file_path)
        for read_id, read_data in reads_dict.items():
            # Convert to uppercase for k-mer extraction
            seq_upper = read_data["seq"].upper()
            seq_len = len(seq_upper)
            
            # Extract k-mers using sliding window
            for i in range(seq_len - k + 1):
                kmer_str = seq_upper[i:i+k]
                kmer_counter[kmer_str] += 1
        
        return kmer_counter
    
    def _get_kmer_similarity(self, ref_kmers: Counter, alt_kmers: Counter) -> float:
        """Get similarity score between k-mer frequency distributions using Hellinger distance.
        Return a similarity score between 0.0 and 1.0 (1.0 - Hellinger distance)
        """
        if not ref_kmers and not alt_kmers:
            return 1.0  # Both empty, perfect match
        
        if not ref_kmers or not alt_kmers:
            return 0.0  # One empty, one not, no match
        
        # Get all unique k-mers from both counters and align them
        all_kmers = sorted(set(ref_kmers.keys()) | set(alt_kmers.keys()))
        
        # Create aligned arrays with frequencies (0 for missing k-mers)
        ref_arr = np.array([ref_kmers.get(kmer, 0) for kmer in all_kmers])
        alt_arr = np.array([alt_kmers.get(kmer, 0) for kmer in all_kmers])
        
        # Calculate Hellinger distance (function handles normalization internally)
        distance = hellinger_distance(ref_arr, alt_arr)
        
        # Convert distance to similarity: similarity = 1 - distance
        similarity = 1.0 - distance if not np.isnan(distance) else 0.0
        
        return similarity
    
    # ============================================================================
    # Read Comparison Helpers
    # ============================================================================
    
    def _compare_sequence(self, ref_seq: str, alt_seq: str, tolerance: int = 2) -> Tuple[bool, int, int]:
        """Compare sequences allowing for trimming differences at ends.
        
        Allows up to tolerance bases to be trimmed from start or end.
        
        Args:
            ref_seq: Reference sequence
            alt_seq: Candidate sequence
            ref_qual: Reference quality scores (unused, kept for API consistency)
            alt_qual: Candidate quality scores (unused, kept for API consistency)
            tolerance: Maximum bases to trim from start or end
            
        Returns:
            (match, trim_start_diff, trim_end_diff)
        """
        len_ref, len_alt = len(ref_seq), len(alt_seq)
        length_diff = abs(len_ref - len_alt)
        
        # Early exit if lengths are too different
        if length_diff > 2 * tolerance:
            return False, length_diff, length_diff
        
        # Check exact match first
        if ref_seq == alt_seq:
            return True, 0, 0
        
        # Generate all valid trimming combinations
        trim_range = range(tolerance + 1)
        trim_combinations = product(trim_range, trim_range, trim_range, trim_range)
        
        for trim_ref_start, trim_ref_end, trim_alt_start, trim_alt_end in trim_combinations:
            # Calculate trimmed sequence boundaries
            ref_start, ref_end = trim_ref_start, len_ref - trim_ref_end
            alt_start, alt_end = trim_alt_start, len_alt - trim_alt_end
            
            # Skip if trimming would result in empty or invalid sequences
            if ref_start >= ref_end or alt_start >= alt_end:
                continue
            
            # Compare trimmed sequences
            if ref_seq[ref_start:ref_end] == alt_seq[alt_start:alt_end]:
                trim_start_diff = abs(trim_ref_start - trim_alt_start)
                trim_end_diff = abs(trim_ref_end - trim_alt_end)
                return True, trim_start_diff, trim_end_diff
        
        return False, length_diff, length_diff
    
    def _compare_quality(self, ref_qual: np.ndarray, alt_qual: np.ndarray, tolerance: int = 2) -> bool:
        """Compare quality scores allowing for ±tolerance difference in Phred scores.
        
        Args:
            ref_qual: Reference quality score array
            alt_qual: Candidate quality score array
            tolerance: Maximum allowed difference in Phred scores
            
        Returns:
            True if quality scores match within tolerance
        """
        if len(ref_qual) != len(alt_qual):
            return False
        
        # Calculate absolute differences and check if all are within tolerance
        return np.all(np.abs(ref_qual - alt_qual) <= tolerance)
    
    # ============================================================================
    # Private Comparison Implementation Methods
    # ============================================================================
    
    def _compare_exact(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Exact match: Read ID, sequence, and quality scores identical.
        
        Matches reads by ID and requires exact match of sequences and quality scores.
        Must be 100% identical for similarity score of 1.0.
        
        Args:
            **kwargs: Strategy-specific parameters (unused for exact strategy)
        """
        try:
            # Load files
            ref_reads = self._load_file(ref_validation.file_path)
            alt_reads = self._load_file(alt_validation.file_path)
            
            total_ref_reads = len(ref_reads)
            total_alt_reads = len(alt_reads)
            matching_count = 0
            
            common_keys = set(ref_reads.keys()) & set(alt_reads.keys())

            for read_id in common_keys:
                ref_data = ref_reads[read_id]
                alt_data = alt_reads[read_id]

                seq_match = ref_data["seq"] == alt_data["seq"]
                qual_match = np.array_equal(ref_data["qual"], alt_data["qual"])

                matching_count += int(seq_match and qual_match)
            
            # Calculate similarity (must be 100% for exact match)
            similarity = matching_count / total_ref_reads if total_ref_reads > 0 else 0.0
            
            self.result.similarity = similarity
            self.result.details.update({
                "total_ref_reads": total_ref_reads,
                "total_alt_reads": total_alt_reads,
                "matching_count": matching_count,
                "missing_count": total_ref_reads - matching_count,
                "extra_count": total_alt_reads - matching_count,
            })

        except Exception as e:
            logger.error(f"Exact comparison failed: {e}")
            self.result.error = f"Exact comparison failed: {e}"
    
    def _compare_approx(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Approximate comparison: Sequence and quality score comparison allowing for variations.
        
        Similar to exact comparison but allows for trimming differences and quality score variations.
        Matches reads by ID and compares sequences and quality scores with tolerance.
        
        Args:
            **kwargs: Strategy-specific parameters
            - trim_tolerance: Positional tolerance in base pairs for trimming differences (default: 2)
            - quality_tolerance: Phred score tolerance (default: 2)
        """
        trim_tolerance = kwargs.get("trim_tolerance", 2)
        quality_tolerance = kwargs.get("quality_tolerance", 2)
        
        try:
            # Load files
            ref_reads = self._load_file(ref_validation.file_path)
            alt_reads = self._load_file(alt_validation.file_path)
            
            total_ref_reads = len(ref_reads)
            total_alt_reads = len(alt_reads)
            common_keys = set(ref_reads.keys()) & set(alt_reads.keys())
            matching_count_seq = 0
            matching_count_qual = 0
            matching_count = 0
            for read_id in common_keys:
                ref_data = ref_reads[read_id]
                alt_data = alt_reads[read_id]
                seq_match, _, _ = self._compare_sequence(ref_data["seq"], alt_data["seq"], trim_tolerance)
                qual_match = self._compare_quality(ref_data["qual"], alt_data["qual"], quality_tolerance)
                matching_count_seq += int(seq_match)
                matching_count_qual += int(qual_match)
                matching_count += int(seq_match and qual_match)
            
            similarity = matching_count / total_ref_reads if total_ref_reads > 0 else 0.0
            
            self.result.similarity = similarity
            self.result.details.update({
                "total_ref_reads": total_ref_reads,
                "total_alt_reads": total_alt_reads,
                "matching_count": matching_count,
                "matching_count_seq": matching_count_seq,
                "matching_count_qual": matching_count_qual,
                "missing_count": total_ref_reads - matching_count,
                "extra_count": total_alt_reads - matching_count,
                "trim_tolerance": trim_tolerance,
                "quality_tolerance": quality_tolerance
            })
            
        except Exception as e:
            logger.error(f"Approximate comparison failed: {e}")
            self.result.error = f"Approximate comparison failed: {e}"
    
    def _compare_summary(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Summary comparison: Compare k-mer frequency distributions and length distribution.
        
        Compares k-mer frequency distributions using Hellinger distance and length distribution
        using histogram-based Hellinger distance.
        
        Args:
            **kwargs: Strategy-specific parameters
            - k: K-mer size for k-mer comparison (default: 11)
            - num_bins: Number of bins for length distribution histogram (default: 10)
        """
        k = kwargs.get('k', 11)
        
        try:
            # Compute summary statistics for both files (for details only)
            ref_stats = self._get_summary_stats(ref_validation.file_path)
            alt_stats = self._get_summary_stats(alt_validation.file_path)
            
            ref_lengths = ref_stats.pop("length_arr", np.array([]))
            alt_lengths = alt_stats.pop("length_arr", np.array([]))

            # Compare number of reads using RAE similarity
            ref_num = ref_stats.get("num_reads", 0)
            alt_num = alt_stats.get("num_reads", 0)
            num_reads_sim = float(rae_similarity([ref_num], [alt_num])[0])

            # Compare total length similarity using RAE similarity
            ref_total_length = ref_stats.get("total_length", 0)
            alt_total_length = alt_stats.get("total_length", 0)
            total_length_sim = float(rae_similarity([ref_total_length], [alt_total_length])[0])

            # Compare length distribution using histogram
            num_bins = kwargs.get('num_bins', 10)
            # Handle empty arrays
            if len(ref_lengths) == 0 and len(alt_lengths) == 0:
                # Both empty - perfect match
                length_distribution_similarity = 1.0
                length_hellinger_distance = 0.0
            elif len(ref_lengths) == 0 or len(alt_lengths) == 0:
                # One empty, one not - no match
                length_distribution_similarity = 0.0
                length_hellinger_distance = 1.0
            else:
                # Both non-empty - find common range for bins
                all_lengths = np.concatenate([ref_lengths, alt_lengths])
                min_length = all_lengths.min()
                max_length = all_lengths.max()
                
                # Default: all reads have the same length - check if both have the same mean
                length_distribution_similarity = 1.0 if ref_lengths.mean() == alt_lengths.mean() else 0.0
                length_hellinger_distance = 0.0 if length_distribution_similarity == 1.0 else 1.0
                
                if max_length > min_length:
                    # Create bins
                    bins = np.linspace(min_length, max_length, num_bins + 1)
                    
                    # Calculate histograms
                    ref_hist, _ = np.histogram(ref_lengths, bins=bins)
                    alt_hist, _ = np.histogram(alt_lengths, bins=bins)
                    
                    # Calculate Hellinger distance (function handles normalization internally)
                    # Convert distance to similarity: similarity = 1 - distance
                    length_hellinger_distance = hellinger_distance(ref_hist, alt_hist)
                    length_distribution_similarity = 1.0 - float(length_hellinger_distance) if not np.isnan(length_hellinger_distance) else 0.0

            self.result.details["length_distribution_similarity"] = length_distribution_similarity
            self.result.details["length_hellinger_distance"] = float(length_hellinger_distance)

            # Compute k-mer frequency distributions
            ref_kmers = self._get_kmer_counter(ref_validation.file_path, k)
            alt_kmers = self._get_kmer_counter(alt_validation.file_path, k)
            
            # Compare k-mer distributions
            kmer_similarity = self._get_kmer_similarity(ref_kmers, alt_kmers)
            
            self.result.similarity = (num_reads_sim + total_length_sim + length_distribution_similarity + kmer_similarity) / 4
            self.result.details.update({
                "num_reads_similarity": num_reads_sim,
                "total_length_similarity": total_length_sim,
                "length_distribution_similarity": length_distribution_similarity,
                "kmer_similarity": kmer_similarity,
                "ref_length_stats": ref_stats,
                "alt_length_stats": alt_stats,
                "ref_num_kmers": len(ref_kmers),
                "alt_num_kmers": len(alt_kmers),
                "kmer_size": k,
            })
        except Exception as e:
            logger.error(f"Summary comparison failed: {e}")
            self.result.error = f"Summary comparison failed: {e}"

