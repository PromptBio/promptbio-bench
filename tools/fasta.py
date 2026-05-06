"""FASTA format handler.

Strategy:
- exact: Direct sequence matching by name or sequence content
- approximate: Sequence alignment-based comparison allowing for variations by name or sequence content
- summary: Summary statistics comparison of sequence count, total length, and k-mer frequency distribution

Required cmdline tools:
- (None)
"""

import logging
import gzip
from collections import Counter
from typing import Dict, List, Any, Union

import numpy as np

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner

from tools.base import FormatHandler
from tools.results import ValidationResult, ComparisonResult
from utils.format import detect_file_signature
from utils.metrics import hellinger_distance, rae_similarity

logger = logging.getLogger(__name__)


class FASTAHandler(FormatHandler):
    """Handler for FASTA format files."""
    
    def __init__(self):
        super().__init__("fasta")
        self.supported_formats = ['fasta', 'fa', 'fas']
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
        """Validate FASTA file format.
        non-empty check is performed by file size, later can be updated by reading content if needed
        
        Args:
            file_path: Path to FASTA file
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
            logger.error(f"FASTA file not found: {file_path}")
            result.error = "File not found"
            return result
        
        if not is_nonempty:
            logger.warning(f"FASTA file is empty: {file_path}")
            result.error = "File is empty"
            return result
        
        if not is_format_ok:
            logger.error(f"Not a FASTA file: detected={signature.format}")
            result.error = f"Not a FASTA file: detected={signature.format}"
            return result
        
        # validate by parsing the FASTA file
        try:
            seq_dict = self._load_file(file_path, by_name=True)
            seq_count = len(seq_dict)
            total_length = sum(len(seq) for seq in seq_dict.values())
            
            result.parseable = True
            result.non_empty = seq_count > 0
            
            if not result.non_empty:
                logger.warning(f"FASTA file contains no sequences: {file_path}")
                result.error = "FASTA file contains no sequences"
                result.details.update({"num_sequences": 0})
                return result
            
            result.details.update({
                "num_sequences": seq_count,
                "total_length": total_length,
            })
            
            logger.info(f"FASTA validation passed: {seq_count} sequences, {total_length} bp total")
        except Exception as e:
            logger.error(f"FASTA validation failed: {e}")
            result.error = f"FASTA validation failed: {e}"
        return result
    
    def compare(self, reference_path: str, candidate_path: str, strategy: str = "exact", **kwargs) -> ComparisonResult:
        """Compare FASTA files.
        
        Supported strategies:
        - exact: Direct sequence matching by ID or by sequence content
        - approximate: Sequence alignment-based comparison allowing for variations
        - summary: K-mer frequency distribution comparison
        
        Args:
            reference_path: Path to reference FASTA file
            candidate_path: Path to candidate FASTA file
            strategy: Comparison strategy (exact, approximate, summary)
            **kwargs: Strategy-specific parameters
            - For exact strategy:
              - by_name: If True, match sequences by ID; if False, match by content (default: False)
            - For approximate strategy:
              - by_name: If True, match sequences by ID first; if False, match by content (default: False)
            - For summary strategy:
              - k: K-mer size for k-mer comparison (default: 11)
            - ignore_case: If True, sequences are uppercased before comparison (default: True)
            - single_line: If True, multiline sequences in alt are skipped (treated as absent), giving them 0 credit (default: False)
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
                    # Direct sequence matching by ID
                    self._compare_exact(ref_validation, alt_validation, **kwargs)
                elif strategy == "approximate":
                    # Sequence alignment-based comparison allowing for variations
                    self._compare_approx(ref_validation, alt_validation, **kwargs)
                elif strategy == "summary":
                    # K-mer frequency distribution comparison
                    self._compare_summary(ref_validation, alt_validation, **kwargs)
                else:
                    logger.warning(f"Unknown strategy '{strategy}', fall back to exact")
                    self._compare_exact(ref_validation, alt_validation, **kwargs)
        except Exception as e:
            logger.error(f"FASTA comparison failed: {e}")
            self.result.error = f"FASTA comparison failed: {e}"
        return self._finalize_output(ref_validation, alt_validation, self.result)
    
    # ============================================================================
    # File I/O and Loading Methods
    # ============================================================================
    
    def _load_file(self, file_path: str, by_name: bool = True, ignore_case: bool = True, single_line: bool = False) -> Dict[str, Union[str, List[str]]]:
        """Load sequences from a FASTA file into a dictionary.

        This is the central method for loading FASTA files. All other methods
        should use this method to avoid duplicating SeqIO.parse() calls.
        Supports compressed files (gzip, bgzf) automatically.

        Args:
            file_path: Path to FASTA file (supports compression: .gz)
            by_name: If True, return dict mapping ID -> sequence string.
                     If False, return dict mapping sequence string -> list of IDs.
            ignore_case: If True, sequences are uppercased before being returned.
            single_line: If True, multiline sequences (sequence spans more than one line) are
                         skipped and not returned. Allows per-sequence format enforcement.

        Returns:
            Dictionary mapping:
            - If by_name=True: {id: seq} - sequence ID to sequence string
            - If by_name=False: {seq: [id1, id2, ...]} - sequence string to list of IDs
            (e.g., when two sequences have the same content but different IDs)

        Raises:
            Exception: If file cannot be parsed
        """
        try:
            signature = detect_file_signature(file_path, fallback=True)
            compression = signature.compression if signature.compression in self.format_map else 'none'
            open_config = self.format_map[compression]
            with open_config['open_func'](file_path, open_config['mode'], encoding='utf-8', errors='replace') as handle:
                records = list(SeqIO.parse(handle, "fasta"))

            if single_line:
                multiline_ids = self._get_multiline_ids(file_path)
                records = [r for r in records if r.id not in multiline_ids]

            def _seq_str(record):
                s = str(record.seq)
                return s.upper() if ignore_case else s

            if by_name:
                return {record.id: _seq_str(record) for record in records}
            else:
                seq_to_ids = {}
                for record in records:
                    seq_content = _seq_str(record)
                    if seq_content not in seq_to_ids:
                        seq_to_ids[seq_content] = []
                    seq_to_ids[seq_content].append(record.id)
                return seq_to_ids
        except Exception as e:
            logger.error(f"Failed to load sequences from {file_path}: {e}")
            raise
    
    # ============================================================================
    # Statistics and Analysis Methods
    # ============================================================================
    
    def _get_summary_stats(self, file_path: str, single_line: bool = False) -> Dict:
        """Compute summary statistics for a FASTA file.
        
        Args:
            file_path: Path to FASTA file
            
        Returns:
            Dictionary with summary statistics:
            - num_sequences: Number of sequences
            - total_length: Total length of all sequences
            - avg_length: Average sequence length
            - min_length: Minimum sequence length
            - max_length: Maximum sequence length
            - median: Median sequence length (50th percentile)
            - q25: 25th percentile sequence length
            - q75: 75th percentile sequence length
        """
        seq_dict = self._load_file(file_path, by_name=True, single_line=single_line)
        seq_count = len(seq_dict)
        lengths = [len(seq) for seq in seq_dict.values()]
        total_length = sum(lengths)
        
        lengths_arr = np.array(lengths) if seq_count > 0 else np.array([])
        
        return {
            "num_sequences": seq_count,
            "total_length": total_length,
            "length_arr": lengths_arr,
            "length_mean": total_length / seq_count if seq_count > 0 else 0.0,
            "length_std": lengths_arr.std() if seq_count > 0 else 0.0,
            "length_min": lengths_arr.min() if seq_count > 0 else 0,
            "length_max": lengths_arr.max() if seq_count > 0 else 0,
            "length_median": np.median(lengths_arr) if seq_count > 0 else 0.0,
            "length_q25": np.quantile(lengths_arr, 0.25) if seq_count > 0 else 0.0,
            "length_q75": np.quantile(lengths_arr, 0.75) if seq_count > 0 else 0.0,
        }
    
    def _get_kmer_counter(self, file_path: str, k: int = 11, ignore_case: bool = True, single_line: bool = False) -> Counter:
        """Extract k-mers from all sequences in a FASTA file.
        Args:
            file_path: Path to FASTA file
            k: K-mer size
            ignore_case: If True, sequences are uppercased before k-mer extraction.

        Return a Counter object mapping k-mer to frequency
        """
        seq_dict = self._load_file(file_path, by_name=True, ignore_case=ignore_case, single_line=single_line)
        kmer_counter = Counter()

        for seq in seq_dict.values():
            seq_len = len(seq)

            # Extract k-mers using sliding window
            for i in range(seq_len - k + 1):
                kmer_str = seq[i:i+k]
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

    def _get_multiline_ids(self, file_path: str) -> set:
        """Return IDs of sequences NOT in single-line format (sequence spans more than one line).

        Reads the raw file without BioPython to detect line structure that SeqIO normalizes away.
        """
        signature = detect_file_signature(file_path, fallback=True)
        compression = signature.compression if signature.compression in self.format_map else 'none'
        open_config = self.format_map[compression]
        multiline_ids = set()
        current_id, seq_line_count = None, 0
        with open_config['open_func'](file_path, open_config['mode'], encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.rstrip('\n')
                if line.startswith('>'):
                    if current_id is not None and seq_line_count != 1:
                        multiline_ids.add(current_id)
                    current_id = line[1:].split()[0]
                    seq_line_count = 0
                elif current_id is not None and line.strip():
                    seq_line_count += 1
            if current_id is not None and seq_line_count != 1:
                multiline_ids.add(current_id)
        return multiline_ids

    def _get_alignment_similarity(self, ref_seq: str, alt_seq: str, aligner: PairwiseAligner) -> float:
        """Calculate normalized similarity score between two sequences using alignment.
        
        Args:
            ref_seq: Reference sequence string
            alt_seq: Candidate sequence string
            aligner: PairwiseAligner instance configured for alignment
            
        Returns:
            Normalized similarity score between 0.0 and 1.0, or 0.0 if alignment fails
        """
        if not ref_seq or not alt_seq:
            return 1.0 if ref_seq == alt_seq else 0.0
        
        # Check for exact match before expensive alignment
        if ref_seq == alt_seq:
            return 1.0
        
        try:
            ref_seq_obj = Seq(ref_seq)
            alt_seq_obj = Seq(alt_seq)
            
            alignments = aligner.align(ref_seq_obj, alt_seq_obj)
            if not alignments:
                return 0.0
            
            score = alignments[0].score
            max_possible_score = min(len(ref_seq), len(alt_seq)) * aligner.match_score
            
            return max(0.0, min(1.0, score / max_possible_score))
        except Exception as e:
            logger.warning(f"Alignment failed: {e}")
            return 0.0
    
    def _get_matched_pairs(self, ref_data: Dict, alt_data: Dict, by_name: bool, aligner: PairwiseAligner) -> List[tuple]:
        """Get matched sequence pairs between reference and candidate data.
        
        Args:
            ref_data: Reference data (ID->[seq] if by_name=True, seq->[IDs] if by_name=False)
            alt_data: Candidate data (ID->[seq] if by_name=True, seq->[IDs] if by_name=False)
            by_name: Whether to match by ID (True) or by content (False)
            aligner: PairwiseAligner instance for calculating similarity
        
        Returns:
            List of tuples where each tuple is (ref_info, alt_info):
            - ref_info is [name, seq] or None
            - alt_info is [name, seq] or None
            - For matched pairs: both are [name, seq]
            - For unmatched ref: (ref_info, None)
            - For unmatched alt: (None, alt_info)
        """
        pairs = []
        
        if by_name:
            # Match by ID: straightforward key-based matching
            ref_keys = set(ref_data.keys())
            alt_keys = set(alt_data.keys())
            common_keys = ref_keys & alt_keys
            
            # Matched pairs
            for key in common_keys:
                ref_seq = ref_data[key]  # Get sequence string
                alt_seq = alt_data[key]  # Get sequence string
                pairs.append(([key, ref_seq], [key, alt_seq]))
            
            # Unmatched ref sequences
            for key in ref_keys - alt_keys:
                ref_seq = ref_data[key]  # Get sequence string
                pairs.append(([key, ref_seq], None))
            
            # Unmatched alt sequences
            for key in alt_keys - ref_keys:
                alt_seq = alt_data[key]  # Get sequence string
                pairs.append((None, [key, alt_seq]))
        else:
            # Match by content: use greedy matching to find best pairs
            ref_seqs = list(ref_data.keys())
            alt_seqs = list(alt_data.keys())
            
            # Pre-filter: exact matches don't need alignment
            exact_matches = set(ref_seqs) & set(alt_seqs)
            ref_remaining = [seq for seq in ref_seqs if seq not in exact_matches]
            alt_remaining_set = {seq for seq in alt_seqs if seq not in exact_matches}
            
            matched_ref_seqs = set()
            matched_alt_seqs = set()
            
            # Add exact matches
            for seq in exact_matches:
                ref_id = ref_data[seq][0]  # Pick first ID from list
                alt_id = alt_data[seq][0]  # Pick first ID from list
                pairs.append(([ref_id, seq], [alt_id, seq]))
                matched_ref_seqs.add(seq)
                matched_alt_seqs.add(seq)
            
            # Greedy matching: for each remaining ref sequence, find best alt match
            for ref_seq in ref_remaining:
                similarity, matched_alt_seq = self._get_sequence_pair(
                    ref_seq, alt_remaining_set, matched_alt_seqs, aligner
                )
                
                if matched_alt_seq is not None:
                    ref_id = ref_data[ref_seq][0]  # Pick first ID from list
                    alt_id = alt_data[matched_alt_seq][0]  # Pick first ID from list
                    pairs.append(([ref_id, ref_seq], [alt_id, matched_alt_seq]))
                    matched_ref_seqs.add(ref_seq)
                    alt_remaining_set.discard(matched_alt_seq)
            
            # Unmatched ref sequences
            for seq in ref_remaining:
                if seq not in matched_ref_seqs:
                    ref_id = ref_data[seq][0]  # Pick first ID from list
                    pairs.append(([ref_id, seq], None))
            
            # Unmatched alt sequences
            for seq in alt_remaining_set:
                alt_id = alt_data[seq][0]  # Pick first ID from list
                pairs.append((None, [alt_id, seq]))
        
        return pairs
    
    def _get_sequence_pair(self, ref_seq: str, alt_seqs, matched_alt_seqs: set, aligner: PairwiseAligner, min_length_ratio: float = 0.5, min_similarity: float = 0.0) -> tuple:
        """Find the best matching candidate sequence for a reference sequence.
        
        Memory-efficient implementation that:
        - Pre-filters by length ratio before expensive alignment
        - Early termination on perfect matches
        - Accepts list or set for alt_seqs (set is more memory efficient)
        
        Args:
            ref_seq: Reference sequence to match
            alt_seqs: List or set of available candidate sequences
            matched_alt_seqs: Set of already matched candidate sequences (will be updated)
            aligner: PairwiseAligner instance for calculating similarity
            min_length_ratio: Minimum length ratio to consider alignment (default: 0.5)
            min_similarity: Minimum similarity threshold to accept a match (default: 0.0)
        
        Returns:
            Tuple of (similarity_score, matched_alt_seq) if a match is found, 
            or (None, None) if no suitable match is found.
            The matched_alt_seq is automatically added to matched_alt_seqs.
        """
        ref_len = len(ref_seq)
        best_similarity = -1.0
        best_alt_seq = None
        
        # Use generator to filter sequences efficiently
        for alt_seq in alt_seqs:
            if alt_seq in matched_alt_seqs:
                continue
            
            # Quick pre-filter: skip if length difference is too large
            alt_len = len(alt_seq)
            if alt_len == 0:
                continue
            
            # More efficient length ratio calculation
            if ref_len > alt_len:
                if alt_len / ref_len < min_length_ratio:
                    continue
            else:
                if ref_len / alt_len < min_length_ratio:
                    continue
            
            similarity = self._get_alignment_similarity(ref_seq, alt_seq, aligner)
            
            # Early termination: perfect match found
            if similarity >= 1.0:
                matched_alt_seqs.add(alt_seq)
                return (similarity, alt_seq)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_alt_seq = alt_seq
        
        if best_alt_seq is not None and best_similarity >= min_similarity:
            matched_alt_seqs.add(best_alt_seq)
            return (best_similarity, best_alt_seq)
        
        return (None, None)
    
    # ============================================================================
    # Private Comparison Implementation Methods
    # ============================================================================
    
    def _compare_exact(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Exact match: Direct sequence matching by ID or by content.
        
        By default (by_name=False), matches sequences by content regardless of ID.
        When by_name=False, sequences with identical content are grouped together, so duplicate
        sequences count as a single match. Use by_name=True to match by ID and get individual
        sequence record matches.
        
        Args:
            **kwargs: Strategy-specific parameters
            - by_name: If True, match sequences by ID; if False, match by content (default: False)
        """
        by_name = kwargs.get('by_name', False)
        ignore_case = kwargs.get('ignore_case', True)

        try:
            # Load sequences
            single_line = kwargs.get('single_line', False)
            ref_data = self._load_file(ref_validation.file_path, by_name=by_name, ignore_case=ignore_case)
            alt_data = self._load_file(alt_validation.file_path, by_name=by_name, ignore_case=ignore_case, single_line=single_line)

            ref_keys = set(ref_data.keys())
            alt_keys = set(alt_data.keys())
            common_keys = ref_keys & alt_keys
            missing_keys = ref_keys - alt_keys
            extra_keys = alt_keys - ref_keys
            
            if not by_name:
                # Match by sequence content regardless of ID
                # Count matching sequence records: for each common sequence, count the minimum occurrences
                matching_count = sum(min(len(ref_data[key]), len(alt_data[key])) for key in common_keys)
                # Count total sequences (including duplicates with same content)
                total_ref_count = sum(len(ids) for ids in ref_data.values())
                total_alt_count = sum(len(ids) for ids in alt_data.values())
            else:
                # Match sequences by ID
                matching_count = sum(ref_data[key] == alt_data[key] for key in common_keys)
                # Count unique sequences (one per ID)
                total_ref_count = len(ref_keys)
                total_alt_count = len(alt_keys)

            similarity = matching_count / total_ref_count if total_ref_count > 0 else 0.0
            
            self.result.similarity = similarity
            self.result.details.update({
                "total_ref_count": total_ref_count,
                "total_alt_count": total_alt_count,
                "matching_count": matching_count,
                "missing_count": len(missing_keys),
                "extra_count": len(extra_keys),
            })
        except Exception as e:
            logger.error(f"Exact comparison failed: {e}")
            self.result.error = f"Exact comparison failed: {e}"
    
    def _compare_approx(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Approximate comparison: Sequence alignment-based comparison allowing for variations.
        
        Similar to exact comparison but uses alignment to calculate similarity scores
        instead of requiring exact matches. Matches sequences by ID or by content.
        
        Args:
            **kwargs: Strategy-specific parameters
            - by_name: If True, match sequences by ID; if False, match by content using greedy algorithm (default: False)
        """
        by_name = kwargs.get('by_name', False)
        ignore_case = kwargs.get('ignore_case', True)
        single_line = kwargs.get('single_line', False)

        try:
            # Load sequences
            ref_data = self._load_file(ref_validation.file_path, by_name=by_name, ignore_case=ignore_case)
            alt_data = self._load_file(alt_validation.file_path, by_name=by_name, ignore_case=ignore_case, single_line=single_line)

            # Initialize aligner with BioPython defaults
            aligner = PairwiseAligner()
            aligner.mode = 'local'  # Local alignment (Smith-Waterman)
            
            # Get matched pairs
            pairs = self._get_matched_pairs(ref_data, alt_data, by_name, aligner)
            
            # Calculate alignment scores and counts from pairs
            alignment_scores = []
            matching_count = 0
            missing_count = 0
            extra_count = 0
            
            for ref_info, alt_info in pairs:
                if ref_info is not None and alt_info is not None:
                    # Matched pair: calculate similarity
                    ref_seq = ref_info[1]
                    alt_seq = alt_info[1]
                    if ref_seq == alt_seq:
                        similarity = 1.0
                    else:
                        similarity = self._get_alignment_similarity(ref_seq, alt_seq, aligner)
                    alignment_scores.append(similarity)
                    matching_count += 1
                elif ref_info is not None:
                    # Unmatched ref
                    missing_count += 1
                else:
                    # Unmatched alt
                    extra_count += 1
            
            # Calculate total counts
            if by_name:
                total_ref_count = len(ref_data)
                total_alt_count = len(alt_data)
                total_ref_unique_count = total_ref_count
            else:
                # For similarity calculation: use unique sequence count
                total_ref_unique_count = len(ref_data)
                # For reporting: use total sequence count (including duplicates)
                total_ref_count = sum(len(ids) for ids in ref_data.values())
                total_alt_count = sum(len(ids) for ids in alt_data.values())
            
            # For similarity: use total_ref_count when by_name=True, total_ref_unique_count when by_name=False
            similarity_denominator = total_ref_count if by_name else total_ref_unique_count
            similarity = sum(alignment_scores) / similarity_denominator if similarity_denominator > 0 else 0.0
            
            self.result.similarity = similarity
            self.result.details.update({
                "matching_count": matching_count,
                "missing_count": missing_count,
                "extra_count": extra_count,
                "avg_alignment_similarity": sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0,
                "min_alignment_score": min(alignment_scores) if alignment_scores else 0.0,
                "max_alignment_score": max(alignment_scores) if alignment_scores else 0.0,
                "total_ref_count": total_ref_count,
                "total_alt_count": total_alt_count,
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
        ignore_case = kwargs.get('ignore_case', True)
        single_line = kwargs.get('single_line', False)

        try:
            # Compute summary statistics for both files (for details only)
            ref_stats = self._get_summary_stats(ref_validation.file_path)
            alt_stats = self._get_summary_stats(alt_validation.file_path, single_line=single_line)
            
            ref_lengths = ref_stats.pop("length_arr", np.array([]))
            alt_lengths = alt_stats.pop("length_arr", np.array([]))

            # Compare number of sequences using RAE similarity
            ref_num = ref_stats.get("num_sequences", 0)
            alt_num = alt_stats.get("num_sequences", 0)
            num_sequences_sim = float(rae_similarity([ref_num], [alt_num])[0])

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
                
                # Default: all sequences have the same length - check if both have the same mean
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
            ref_kmers = self._get_kmer_counter(ref_validation.file_path, k, ignore_case=ignore_case)
            alt_kmers = self._get_kmer_counter(alt_validation.file_path, k, ignore_case=ignore_case, single_line=single_line)
            
            # Compare k-mer distributions
            kmer_similarity = self._get_kmer_similarity(ref_kmers, alt_kmers)
            
            self.result.similarity = (num_sequences_sim + total_length_sim + length_distribution_similarity + kmer_similarity) / 4
            self.result.details.update({
                "num_sequences_similarity": num_sequences_sim,
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

