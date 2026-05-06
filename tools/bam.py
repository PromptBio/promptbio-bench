"""BAM/SAM/CRAM format handler.

Strategy:
- exact: Record-by-record alignment matching, all records identical (100% required)
- approximate: Compares reads with MAPQ >= threshold (default: 20); each read pair requires ≥95% overlap for concordance
- summary: Summary statistics comparison of read counts, mapping percentages, coverage distributions
- coverage: Compare coverage depth between two BAM/SAM/CRAM files.
- variant: Compare variants between two BAM/SAM/CRAM files.

Required cmdline tools:
- samtools
- bcftools
"""

import logging
import os
import subprocess
import re
import tempfile
from typing import Dict, Any, Optional, List, Set, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import pysam

from tools.base import FormatHandler
from tools.results import ValidationResult, ComparisonResult
from tools.vcf import VCFHandler
from utils.format import detect_file_signature
from utils.metrics import hellinger_distance, jaccard_similarity, mean_absolute_error, precision_recall_f1, relative_absolute_error

logger = logging.getLogger(__name__)


class BAMHandler(FormatHandler):
    """Handler for BAM/SAM/CRAM format files."""
    
    def __init__(self):
        super().__init__("bam")
        self.format_map = {
            'bam': {'mode': 'rb', 'index_format': 'bai'},
            'sam': {'mode': 'r', 'index_format': 'sam'},
            'cram': {'mode': 'rc', 'index_format': 'crai'},
        }
        self.supported_formats = list(self.format_map.keys())
        self.supported_strategies = ["exact", "approximate", "summary", "coverage", "variant"]
    
    # ============================================================================
    # Public API Methods
    # ============================================================================
    
    def validate(self, file_path: str, **kwargs) -> ValidationResult:
        """Validate BAM/SAM/CRAM file
        non-empty check is performed by file size, later can be updated by reading content if needed
        
        Args:
            file_path: Path to BAM/SAM/CRAM file
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
            logger.error(f"BAM/SAM/CRAM file not found: {file_path}")
            result.error = "File not found"
            return result
        
        if not is_nonempty:
            logger.warning(f"BAM/SAM/CRAM file is empty: {file_path}")
            result.error = "File is empty"
            return result
        
        if not is_format_ok:
            logger.error(f"Not a BAM/SAM/CRAM file: detected={signature.format}")
            result.error = f"Not a BAM/SAM/CRAM file: detected={signature.format}"
            return result
        
        # collect flagstat stats using samtools
        try:
            flagstat_df = self._get_flagstat_df(file_path)
            result.parseable = flagstat_df.shape[0] > 0
            result.non_empty = (flagstat_df.loc['total', 'pass'] + flagstat_df.loc['total', 'fail']) > 0
            result.details.update(flagstat_df.to_dict())
            logger.info(f"BAM/SAM/CRAM validation passed: {file_path}")
        except Exception as e:
            logger.error(f"BAM/SAM/CRAM validation failed: {e}")
            result.error = f"BAM/SAM/CRAM validation failed: {e}"
        return result
    
    def compare(self, reference_path: str, candidate_path: str, strategy: str = "summary", **kwargs) -> ComparisonResult:
        """Compare BAM/SAM/CRAM files.
        
        Supported strategies:
        - exact: Record-by-record alignment matching, all records identical (100% required)
        - approximate: Compares reads with MAPQ >= threshold (default: 20); each read pair requires ≥95% overlap for concordance
        - summary: Summary statistics comparison of read counts, mapping percentages, coverage distributions
        - coverage: Compare coverage depth between two BAM/SAM/CRAM files.
        - variant: Compare variants between two BAM/SAM/CRAM files.
        
        Args:
            reference_path: Path to reference BAM/SAM/CRAM file
            candidate_path: Path to candidate BAM/SAM/CRAM file
            strategy: Comparison strategy (exact, approximate, summary, coverage, variant)
            **kwargs: Strategy-specific parameters (e.g., ref_fasta, alt_fasta, min_mapq, overlap_threshold)
            - ref_fasta: Path to reference FASTA file (required for variant strategy)
            - vcf_strategy: VCF comparison strategy (required for variant strategy, default: "summary")
            - min_mapq: Minimum MAPQ score for concordance (required for approximate strategy, default: 20)
            - overlap_threshold: Overlap threshold for concordance (required for approximate strategy, default: 0.95)
            - depth0_similarity: Similarity score for coverage depth (required for coverage strategy, default: np.nan)
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
                    # Performs record-by-record exact matching of all alignment records.
                    self._compare_exact(ref_validation, alt_validation, **kwargs)
                elif strategy == "approximate":
                    # Compares reads with MAPQ >= threshold (default: 20); each read pair requires ≥95% overlap for concordance.
                    self._compare_approx(ref_validation, alt_validation, **kwargs)
                elif strategy == "summary":
                    # Summary statistics comparison: read counts, mapping percentages, coverage distributions.
                    self._compare_summary(ref_validation, alt_validation, **kwargs)
                elif strategy == "coverage":
                    # Compare coverage depth between two BAM/SAM/CRAM files.
                    self._compare_coverage(ref_validation, alt_validation, **kwargs)
                elif strategy == "variant":
                    # Compare variants between two BAM/SAM/CRAM files.
                    self._compare_variant(ref_validation, alt_validation, **kwargs)
                else:
                    logger.warning(f"Unknown strategy '{strategy}', fall back to summary")
                    self._compare_summary(ref_validation, alt_validation, **kwargs)
        except Exception as e:
            logger.error(f"BAM/SAM/CRAM comparison failed: {e}")
            self.result.error = f"BAM/SAM/CRAM comparison failed: {e}"
        return self._finalize_output(ref_validation, alt_validation, self.result)

    # ============================================================================
    # File I/O and Loading Methods
    # ============================================================================
    
    def _load_file(self, file_path: str, file_format: Optional[str] = None) -> pysam.AlignmentFile:
        """
        Load file with auto-detection of format.
        Try the user-specified format first (if provided), then fallback to all supported formats.
        
        Raises:
            ValueError: If the file cannot be opened with any supported format.
        """
        # Test the file_format if provided, then fallback to try-error other formats
        if file_format and file_format in self.format_map:
            format_order = [file_format] + list(set(self.supported_formats) - {file_format})
        else:
            format_order = self.supported_formats
        
        for fmt in format_order:
            try:
                file_obj = pysam.AlignmentFile(file_path, self.format_map[fmt]['mode'])
                return file_obj
            except Exception:
                continue  # try next format
        
        # All attempts failed
        raise ValueError(f"Could not open BAM/SAM/CRAM file: {file_path}")
    
    # ============================================================================
    # File Status Checkers
    # ============================================================================
    
    def _is_index_exists(self, file_path: str) -> bool:
        """Check if BAM file has an index file (.bai) by checking if the file exists.
        """
        index_path = file_path + '.bai'
        return os.path.exists(index_path)
    
    def _is_nsorted(self, alignment_file: pysam.AlignmentFile) -> bool:
        """Check if alignment file is sorted by QNAME (queryname/name-sorted) by @HD header SO (sort order) field.
        """
        header = alignment_file.header
        return header['HD']['SO'].lower() == 'queryname'
    
    def _is_sorted(self, alignment_file: pysam.AlignmentFile) -> bool:
        """Check if alignment file is coordinate-sorted by @HD header SO (sort order) field.
        """
        header = alignment_file.header
        return header['HD']['SO'].lower() == 'coordinate'
    
    
    # ============================================================================
    # File Creation Methods
    # ============================================================================
    
    def _create_nsorted_bam(self, file_path: str) -> str:
        """Create a name-sorted (QNAME-sorted) BAM file using samtools sort -n.
        Returns path to temporary file (caller must clean up after use).
        """
        # Create temporary file for sorted output
        fd, temp_path = tempfile.mkstemp(suffix='.bam', prefix='nsorted_')
        os.close(fd)  # Close file descriptor, we'll use the path with samtools
        
        try:
            # Run samtools sort -n (name sort)
            result = subprocess.run(
                ["samtools", "sort", "-n", "-o", temp_path, file_path],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for sorting
                check=False
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"samtools sort -n failed: {result.stderr}")
            
            return temp_path
        except Exception as e:
            self._cleanup_file(temp_path)
            raise
    
    def _create_sorted_bam(self, file_path: str) -> str:
        """Create a coordinate-sorted BAM file using samtools sort.
        Returns path to temporary file (caller must clean up after use).
        """
        # Create temporary file for sorted output
        fd, temp_path = tempfile.mkstemp(suffix='.bam', prefix='sorted_')
        os.close(fd)  # Close file descriptor, we'll use the path with samtools
        
        try:
            # Run samtools sort (coordinate sort)
            result = subprocess.run(
                ["samtools", "sort", "-o", temp_path, file_path],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for sorting
                check=False
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"samtools sort failed: {result.stderr}")
            
            return temp_path
        except Exception as e:
            self._cleanup_file(temp_path)
            raise
    
    def _create_bai(self, file_path: str) -> str:
        """Create a BAM index file (.bai) using samtools index.
        Returns path to index file.
        """
        index_path = file_path + '.bai'
        
        result = subprocess.run(
            ["samtools", "index", "-b", file_path],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for indexing
            check=False
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"samtools index failed: {result.stderr}")
        
        if not self._is_index_exists(file_path):
            raise RuntimeError(f"Index file was not created: {index_path}")
        
        return index_path
    
    def _create_vcf(self, file_path: str, ref_fasta: str) -> str:
        """Generate VCF file from BAM file using bcftools mpileup and call.
        Returns temporary VCF file path (caller must clean up after use).
        """
        # Create temporary VCF file
        fd, vcf_path = tempfile.mkstemp(suffix='.vcf', prefix='vcf_')
        os.close(fd)  # Close file descriptor, we'll use the path with bcftools
        
        try:
            # Step 1: run bcftools mpileup
            mpileup_process = subprocess.Popen(
                ["bcftools", "mpileup", "-Ou", "-f", ref_fasta, file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Step 2: bcftools call (piped from mpileup)
            call_process = subprocess.Popen(
                ["bcftools", "call", "-mv", "-Ov", "-o", vcf_path],
                stdin=mpileup_process.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Close mpileup stdout to allow call to receive EOF
            mpileup_process.stdout.close()
            
            # Wait for both processes
            call_stdout, call_stderr = call_process.communicate()
            mpileup_stderr = mpileup_process.stderr.read()
            mpileup_process.stderr.close()
            mpileup_process.wait()
            
            if mpileup_process.returncode != 0:
                self._cleanup_file(vcf_path)
                raise RuntimeError(f"bcftools mpileup failed: {mpileup_stderr}")
            
            if call_process.returncode != 0:
                self._cleanup_file(vcf_path)
                raise RuntimeError(f"bcftools call failed: {call_stderr}")
            
            return vcf_path
        except Exception as e:
            self._cleanup_file(vcf_path)
            raise
    
    # ============================================================================
    # Statistics and Analysis Methods
    # ============================================================================
    
    def _get_mapq_stats(self, alignment_file: pysam.AlignmentFile) -> Dict[str, Any]:
        """Calculate MAPQ stats summary from pysam AlignmentFile using Counter.
            
        Returns:
            Dictionary containing:
            - MAPQ stats summary: min, q25, median, q75, max, average
            - counter: Counter object mapping MAPQ values to their counts
            Returns all zeros/empty Counter if calculation fails or no reads
        """
        try:
            # Build Counter directly without storing full list
            mapq_counter = Counter()
            for record in alignment_file:
                mapq_counter[record.mapping_quality] += 1
            
            if not mapq_counter:
                raise ValueError("No MAPQ values found in the BAM/SAM/CRAM file")
            
            # Calculate average from Counter: sum(mapq * count) / total_count
            total_count = sum(mapq_counter.values())
            total_sum = sum(mapq * count for mapq, count in mapq_counter.items())
            avg_mapq = total_sum / total_count
            
            # Calculate percentiles from Counter
            sorted_mapqs = sorted(mapq_counter.keys())
            min_mapq = float(sorted_mapqs[0])
            max_mapq = float(sorted_mapqs[-1])
            
            # Helper function to find percentile value
            def find_percentile(percentile: float) -> float:
                """Find the MAPQ value at the given percentile."""
                target_pos = total_count * percentile
                cumulative = 0
                for mapq in sorted_mapqs:
                    cumulative += mapq_counter[mapq]
                    if cumulative >= target_pos:
                        return float(mapq)
                return float(sorted_mapqs[-1])
            
            q25 = find_percentile(0.25)
            median = find_percentile(0.50)
            q75 = find_percentile(0.75)
            
            return {"min": min_mapq, "q25": q25, "median": median, "q75": q75, "max": max_mapq,
                    "average": float(avg_mapq), "counter": mapq_counter}
        except Exception as e:
            logger.error(f"Could not calculate MAPQ stats summary: {e}")
            return {"min": 0.0, "q25": 0.0, "median": 0.0, "q75": 0.0, "max": 0.0, 
                    "average": 0.0, "counter": Counter()}
    
    def _get_num_reads(self, file_path: str) -> int:
        """Get the total number of primary alignment records in a BAM/SAM/CRAM file.
        
        Uses samtools view -c -F 0x900 to count primary alignments only
        (excludes secondary 0x100 and supplementary 0x800 alignments).
        
        Returns total number of primary alignment records, or None if counting fails
        """
        try:
            result = subprocess.run(
                ["samtools", "view", "-c", "-F", "0x900", file_path],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                check=False
            )
            # Parse the count (samtools view -c outputs just a number)
            return int(result.stdout.strip())
        except Exception as e:
            logger.error(f"Could not count reads: {e}")
            return None
    
    def _get_depth_dict(self, file_path: str) -> Dict[str, int]:
        """Parse samtools depth output into dictionary for O(1) lookup.
        Returns dictionary mapping "{chrom}\t{pos}" to depth value.
        """
        try:
            # Run samtools depth -a and parse output into dictionary
            result = subprocess.run(["samtools", "depth", "-a", file_path],
                                    capture_output=True, text=True, timeout=300, check=False)
        except Exception as e:
            raise RuntimeError(f"samtools depth failed: {e}")
        
        if not result.stdout.strip():
            return {}
        
        # Parse directly into dictionary - faster than intermediate structures
        return {f"{parts[0]}:{parts[1]}": int(parts[2]) 
               for line in result.stdout.strip().split('\n') if line.strip()
               for parts in [line.split()] if len(parts) >= 3}
    
    def _get_flagstat_df(self, file_path: str) -> pd.DataFrame:
        """
        Run `samtools flagstat` and return a DataFrame with 'pass', 'fail', and 'ratio' columns.
        The 'ratio' column represents the proportion of reads (pass) relative to total (pass + fail).
        """
        try:
            res = subprocess.run(
                ["samtools", "flagstat", file_path],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception as e:
            raise RuntimeError(f"samtools flagstat failed: {e}")
        
        # here is the alignment record counts, not the unique reads
        key_map = {
            "in total": "total",
            "primary": "primary",
            "secondary": "secondary",
            "supplementary": "supplementary",
            "duplicates": "duplicates",
            "primary duplicates": "primary_duplicates",
            "mapped": "mapped",
            "primary mapped": "primary_mapped",
            "paired in sequencing": "paired",
            "read1": "read1",
            "read2": "read2",
            "properly paired": "properly_paired",
            "with itself and mate mapped": "both_reads_mapped",
            "singletons": "singletons",
            "with mate mapped to a different chr": "mate_mapped_different_chr",
            "with mate mapped to a different chr (mapq>=5)": "mate_mapped_different_chr_mapq5",
        }

        line_re = re.compile(r"^\s*(\d+)\s*\+\s*(\d+)\s+(.*)$")
        data = {}

        for line in res.stdout.splitlines():
            m = line_re.match(line)
            if not m:
                continue

            n_pass, n_fail = int(m.group(1)), int(m.group(2))
            label = m.group(3).strip()

            # normalize label robustly
            if "mapq>=" in label.lower():
                norm = "with mate mapped to a different chr (mapq>=5)"
            else:
                norm = re.sub(r"\s*\(.*\)\s*$", "", label).strip()

            key = key_map.get(norm, key_map.get(norm.lower(), norm.replace(" ", "_")))
            data[key] = {"pass": n_pass, "fail": n_fail}

        # build dataframe
        stat_df = pd.DataFrame.from_dict(data, orient="index")[["pass", "fail"]]
        stat_df.index.name = "category"

        # Compute per-row ratio: pass / (pass + fail)
        total = stat_df["pass"] + stat_df["fail"]
        stat_df["ratio"] = stat_df["pass"] / total.replace(0, pd.NA)

        return stat_df

    def _get_expand_cigar(self, rec: pysam.AlignedSegment) -> List[int]:
        """
        Expand a pysam AlignedSegment.cigartuples into a per-base list of CIGAR op codes.
        Returns list of integer CIGAR operation codes, one per query base. Length equals rec.query_length.
        Operations that do not consume query bases (D, N, P, H) are skipped.
        """
        query_consuming_ops = {0, 1, 4, 7, 8}  # M, I, S, =, X
        return [op for op, length in (rec.cigartuples or []) 
                if op in query_consuming_ops for _ in range(length)]
    
    def _get_alignment_overlap(self, ref_rec: pysam.AlignedSegment, alt_rec: pysam.AlignedSegment) -> Tuple[int, int]:
        """
        Compute the overlap between two alignments and the number of aligned
        reference positions in the reference read.

        Returns (overlap, ref_count) where:
          overlap   — size of the intersection of aligned reference positions
          ref_count — number of reference positions covered by ref_rec
                      (excludes soft-clips and deletions; 0 if unmapped or
                      mapped to a different contig)
        """
        if ref_rec.is_unmapped or alt_rec.is_unmapped:
            return 0, 0
        if ref_rec.reference_id != alt_rec.reference_id:
            return 0, 0

        ref_set: Set[int] = set(p for p in ref_rec.get_reference_positions(full_length=True) if p is not None)
        alt_set: Set[int] = set(p for p in alt_rec.get_reference_positions(full_length=True) if p is not None)

        return len(ref_set & alt_set), len(ref_set)
    
    # ============================================================================
    # Record Comparison Helpers
    # ============================================================================
    
    def _compare_headers(self, ref_header: Dict, alt_header: Dict) -> bool:
        """Compare SAM headers, ignoring @PG header type and SO (sort order) field in @HD.
        
        This function compares SAM/BAM/CRAM file headers for equivalence, with two exceptions:
        
        1. @PG (program) lines are ignored: Different tools (e.g., samtools vs. bwa) will
           produce different @PG entries even when processing the same data identically.
        
        2. SO (sort order) field in @HD is ignored: The sort order annotation (e.g., "coordinate",
           "queryname", "unsorted") may differ between files even if the actual data order is the
           same, or may be missing in some files.
        
        All other header types (@HD fields except SO, @SQ, @RG, @CO, etc.) must match exactly.
        
        Args:
            ref_header: Reference header dictionary from pysam (header.to_dict())
            alt_header: Candidate header dictionary from pysam (header.to_dict())
            
        Returns:
            True if headers are equivalent (ignoring @PG and SO field in @HD), False otherwise
        """
        # Exclude @PG header type from comparison (program lines may differ)
        ignore_header_types = {'PG'}
        ref_types = set(ref_header.keys()) - ignore_header_types
        alt_types = set(alt_header.keys()) - ignore_header_types
        
        # Check if header types match (excluding @PG)
        if ref_types != alt_types:
            return False
        
        # Compare each header type
        for htype in alt_types:
            ref_records = ref_header[htype]
            alt_records = alt_header[htype]
            
            # For @HD, compare but ignore SO (sort order) field
            if htype == 'HD':
                ref_hd = ref_records[0] if ref_records else {}
                alt_hd = alt_records[0] if alt_records else {}
                # Compare all @HD fields except SO
                ref_hd_filtered = {k: v for k, v in ref_hd.items() if k != 'SO'}
                alt_hd_filtered = {k: v for k, v in alt_hd.items() if k != 'SO'}
                if ref_hd_filtered != alt_hd_filtered:
                    return False
            else:
                # For other header types, compare directly
                if ref_records != alt_records:
                    return False
        
        return True
    
    def _compare_record_fields(self, ref_rec: pysam.AlignedSegment, alt_rec: pysam.AlignedSegment) -> bool:
        """Compare mandatory fields of two alignment records.
        
        Compares: QNAME, FLAG, RNAME, POS, MAPQ, CIGAR, RNEXT, PNEXT, TLEN, SEQ, QUAL
        """
        if ref_rec.query_name != alt_rec.query_name: # QNAME
            return False
        if ref_rec.flag != alt_rec.flag: # FLAG
            return False
        if ref_rec.reference_name != alt_rec.reference_name: # RNAME
            return False
        if ref_rec.reference_start != alt_rec.reference_start: # POS
            return False
        if ref_rec.mapping_quality != alt_rec.mapping_quality: # MAPQ
            return False
        if ref_rec.next_reference_name != alt_rec.next_reference_name: # RNEXT
            return False
        if ref_rec.next_reference_start != alt_rec.next_reference_start: # PNEXT
            return False
        if ref_rec.template_length != alt_rec.template_length: # TLEN
            return False
        if ref_rec.query_sequence != alt_rec.query_sequence: # SEQ
            return False
        if ref_rec.query_qualities is None or alt_rec.query_qualities is None: # both None, OK
            pass
        if list(ref_rec.query_qualities) != list(alt_rec.query_qualities): # QUAL
            return False
        if self._get_expand_cigar(ref_rec) != self._get_expand_cigar(alt_rec): # CIGAR
            return False
        return True
    
    def _compare_record_tags(self, ref_rec: pysam.AlignedSegment, alt_rec: pysam.AlignedSegment) -> bool:
        """Compare optional tags of two alignment records.
        
        Only compares tags that are present in both records.
        """
        alt_tags = dict(alt_rec.get_tags())
        ref_tags = dict(ref_rec.get_tags())
        
        # Only compare tags present in both
        common_tags = set(ref_tags.keys()) & set(alt_tags.keys())
        
        for tag in common_tags:
            if ref_tags[tag] != alt_tags[tag]:
                return False
        
        return True
    
    def _compare_record_concord(self, ref_rec: pysam.AlignedSegment, alt_rec: pysam.AlignedSegment, 
                                min_mapq: int = 20, overlap_threshold: float = 0.95, **kwargs) -> bool:
        """Compare the concordance of two alignment records.
        Returns True if the records are concordant, False otherwise.
        """
        if ref_rec is None or alt_rec is None:
            return False
        # Both unmapped: concordant regardless of MAPQ
        if ref_rec.is_unmapped and alt_rec.is_unmapped:
            return True
        # One mapped, one not: non-concordant
        if ref_rec.is_unmapped or alt_rec.is_unmapped:
            return False
        # Check that both reads meet minimum MAPQ threshold
        if ref_rec.mapping_quality < min_mapq or alt_rec.mapping_quality < min_mapq:
            return False
        overlap, ref_count = self._get_alignment_overlap(ref_rec, alt_rec)
        if not ref_count:
            return False
        return (overlap / ref_count) >= overlap_threshold
    
    # ============================================================================
    # Private Comparison Implementation Methods
    # ============================================================================
    
    def _compare_exact(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Exact match: All alignment records identical, line by line in order.
        
        Compares records sequentially, preserving order. Suitable for comparing
        tools that perform name sorting or coordinate sorting.
        
        Criteria:
        - All records identical in the same order (line-by-line comparison)
        - Ignore @PG headers and SO (sort order) tags
        - Compare mandatory fields: QNAME, FLAG, RNAME, POS, MAPQ, CIGAR, RNEXT, PNEXT, TLEN, SEQ, QUAL
        - Compare optional tags (if present in both files)
        - Must be 100% identical
        """
        
        ref_file = None
        alt_file = None
        
        # Quick check: compare record counts using flagstat before doing any comparison
        # This allows early exit if counts don't match
        try:
            ref_flagstat_df = self._get_flagstat_df(ref_validation.file_path)
            alt_flagstat_df = self._get_flagstat_df(alt_validation.file_path)
            
            # Align indices (use union of both) and compute difference for 'pass' and 'fail' columns with L1 norm
            all_categories = ref_flagstat_df.index.union(alt_flagstat_df.index)
            ref_flagstat_df_aligned = ref_flagstat_df.reindex(all_categories, fill_value=0)
            alt_flagstat_df_aligned = alt_flagstat_df.reindex(all_categories, fill_value=0)
            l1_norm = mean_absolute_error(ref_flagstat_df_aligned[['pass', 'fail']].values, 
                                          alt_flagstat_df_aligned[['pass', 'fail']].values)
            
            # If L1 norm is not 0, flagstat differ
            if l1_norm != 0:
                self.result.details.update({
                    "total_ref_records": int(ref_flagstat_df.loc['total', 'pass']) if 'total' in ref_flagstat_df.index else 0,
                    "total_alt_records": int(alt_flagstat_df.loc['total', 'pass']) if 'total' in alt_flagstat_df.index else 0,
                    "l1_norm": float(l1_norm)
                })
                return # flagstat differ, return early
            
            # Compare records one by one without loading all into memory
            total_records = 0
            identical_records = 0
            mismatched_fields_count = 0
            mismatched_tags_count = 0
            
            ref_file = self._load_file(ref_validation.file_path, file_format=ref_validation.signature.format)
            alt_file = self._load_file(alt_validation.file_path, file_format=alt_validation.signature.format)
            # Use iterators to compare records sequentially
            ref_iter = iter(ref_file)
            alt_iter = iter(alt_file)
            
            try:
                while True:
                    ref_rec = next(ref_iter)
                    try:
                        alt_rec = next(alt_iter)
                    except StopIteration:
                        # alt_file has fewer records
                        self.result.details.update({"total_ref_records": total_records + 1, 
                                                    "total_alt_records": total_records})

                        return # alt_file has fewer records, return early
                    
                    # Compare mandatory fields
                    fields_match = self._compare_record_fields(ref_rec, alt_rec)
                    # Compare optional tags
                    tags_match = self._compare_record_tags(ref_rec, alt_rec)
                    # Update counts
                    identical_records += int(fields_match and tags_match)
                    mismatched_fields_count += int(not fields_match)
                    mismatched_tags_count += int(not tags_match)
                    # Update total records
                    total_records += 1
            
            except StopIteration:
                # Check if alt_file has more records
                try:
                    next(alt_iter)
                    self.result.details.update({"total_ref_records": total_records, 
                                                "total_alt_records": total_records + 1})
                except StopIteration:
                    # Both files exhausted at the same time - same number of records
                    pass
            
            # Calculate similarity (must be 100% for exact match)
            similarity = 1.0 if identical_records == total_records else 0.0
            self.result.similarity = similarity
            self.result.details.update({"total_records": total_records, 
                                        "identical_records": identical_records, 
                                        "mismatched_records": total_records - identical_records, 
                                        "mismatched_fields_count": mismatched_fields_count, 
                                        "mismatched_tags_count": mismatched_tags_count})
            
        except Exception as e:
            logger.error(f"Exact comparison failed: {e}")
            self.result.error = f"Exact comparison failed: {e}"
        finally:
            self._close_file(ref_file)
            self._close_file(alt_file)
    
    def _compare_approx(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Approximate match: reads map to same locus within thresholds.
        
        Memory-efficient implementation using merge-style comparison for sorted files.
        Automatically sorts files by QNAME if needed, then uses streaming comparison with buffer.
        
        Criteria:
        - Match reads by QNAME (read name)
        - For each read pair, check if RNAME (chromosome) and POS (position) match within 10bp window
        - Filter reads with MAPQ >= min_mapq (default: 20)
        - Check CIGAR strings are similar (allowing minor differences in soft-clipping)
        - Calculate: precision, recall, and Jaccard similarity
        - Overlap threshold (default: 0.95) determines if match meets criteria
        
        Args:
            **kwargs: Strategy-specific parameters (e.g., min_mapq, overlap_threshold, chunk_size)
            - min_mapq: Minimum MAPQ threshold for reads to be compared (default: 20)
            - overlap_threshold: Minimum overlap threshold (default: 0.95)
            - chunk_size: Size of chunk to compare (default: 100000)
        """
        
        ref_file = None
        alt_file = None
        ref_temp_file = None
        alt_temp_file = None
        
        try:
            ref_file = self._load_file(ref_validation.file_path, file_format=ref_validation.signature.format)
            alt_file = self._load_file(alt_validation.file_path, file_format=alt_validation.signature.format)
            # If not sorted, create name-sorted temporary files
            if not self._is_nsorted(ref_file):
                logger.info(f"Reference file not sorted by QNAME, creating temporary name-sorted file")
                ref_temp_file = self._create_nsorted_bam(ref_validation.file_path)
                ref_file.close()  # Close original before opening sorted version
                ref_file = self._load_file(ref_temp_file)
            
            if not self._is_nsorted(alt_file):
                logger.info(f"Candidate file not sorted by QNAME, creating temporary name-sorted file")
                alt_temp_file = self._create_nsorted_bam(alt_validation.file_path)
                alt_file.close()  # Close original before opening sorted version
                alt_file = self._load_file(alt_temp_file)  # Update alt_file with sorted version
            
            # Use memory-efficient chunk-based comparison
            similarity, details = self._compare_approx_chunk(ref_file, alt_file, **kwargs)
            self.result.similarity = similarity
            self.result.details.update(details)
            
        except Exception as e:
            logger.error(f"Approximate comparison failed: {e}")
            self.result.error = f"Approximate comparison failed: {e}"
        finally:
            self._close_file(ref_file)
            self._close_file(alt_file)
            self._cleanup_file(ref_temp_file, verbose=True)
            self._cleanup_file(alt_temp_file, verbose=True)
    
    def _compare_approx_chunk(self, ref_file: pysam.AlignmentFile, alt_file: pysam.AlignmentFile, **kwargs) -> Tuple[float, Dict[str, Any]]:
        """Memory-efficient chunk-based comparison for QNAME-sorted files.
        Maintains two pools (ref and alt) of reads grouped by QNAME.
        When pools reach chunk_size, compares matching QNAMEs and removes compared reads.
        """
        def load_reads_to_pool(iterator, pool: dict, chunk_size: int = 100000, min_mapq: int = 20) -> int:
            """Load reads into pool {QNAME: [read1, read2]} by chunk_size.
            Returns number of reads added.
            """
            added = 0
            try:
                while added < chunk_size:
                    read = next(iterator)
                    # Skip secondary, supplementary, reads without QNAME, or reads below MAPQ threshold
                    if read.is_secondary or read.is_supplementary or not read.query_name or read.mapping_quality < min_mapq:
                        continue
                
                    qname = read.query_name
                    if qname not in pool:
                        pool[qname] = [None, None] # assume will not exceed 2 reads per QNAME
                    pool[qname][int(read.is_read2)] = read
                    added += 1
            except StopIteration:
                pass
            return added
        
        def compare_pools(ref_pool: dict, alt_pool: dict, **kwargs) -> int:
            """Compare reads in both pools with matching QNAMEs, remove matched reads from pools after comparison.
            Returns concordant count.
            """
            concordant = 0
            # Only compare QNAMEs that exist in both pools
            common_qnames = set(ref_pool.keys()) & set(alt_pool.keys())
            for qname in common_qnames:
                ref_reads = ref_pool[qname]  # [read1, read2]
                alt_reads = alt_pool[qname]  # [read1, read2]
                
                concordant += self._compare_record_concord(ref_reads[0], alt_reads[0], **kwargs)
                concordant += self._compare_record_concord(ref_reads[1], alt_reads[1], **kwargs)

                # Remove matched reads from pools
                del ref_pool[qname]
                del alt_pool[qname]
            
            return concordant
        
        # Initialize pools and counts
        ref_pool = {}
        alt_pool = {}
        ref_iter = iter(ref_file)
        alt_iter = iter(alt_file)
        
        concordant_reads = 0
        total_ref_reads = 0
        total_alt_reads = 0
        
        # Main processing loop
        ref_exhausted = False
        alt_exhausted = False
        chunk_size = kwargs.get('chunk_size', 100000)
        
        min_mapq = kwargs.get('min_mapq', 20)
        while not (ref_exhausted and alt_exhausted):
            # Load reads into pools until chunk_size is reached
            if not ref_exhausted:
                added = load_reads_to_pool(ref_iter, ref_pool, chunk_size, min_mapq)
                total_ref_reads += added
                ref_exhausted = added < chunk_size
            
            if not alt_exhausted:
                added = load_reads_to_pool(alt_iter, alt_pool, chunk_size, min_mapq)
                total_alt_reads += added
                alt_exhausted = added < chunk_size
            
            # Compare pools when they reach chunk_size or when exhausted
            concordant_reads += compare_pools(ref_pool, alt_pool, **kwargs)
        
        precision, recall, f1_score = precision_recall_f1(total_ref_reads, total_alt_reads, concordant_reads)
        jaccard = jaccard_similarity(total_ref_reads, total_alt_reads, concordant_reads)
        
        # Use Jaccard similarity as the main similarity metric
        similarity = jaccard
        details = {
            "total_ref_reads": total_ref_reads,
            "total_alt_reads": total_alt_reads,
            "concordant_reads": concordant_reads,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "jaccard_similarity": jaccard,
            "chunk_size": kwargs.get('chunk_size', 100000),
            "overlap_threshold": kwargs.get('overlap_threshold', 0.95),
            "min_mapq": kwargs.get('min_mapq', 20),
        }
        
        return similarity, details
    
    def _compare_coverage(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Compare coverage depth between two BAM/SAM/CRAM files.
        
        Args:
            **kwargs: Strategy-specific parameters (e.g., depth0_similarity)
            - depth0_similarity: Similarity score for when both depths are 0 (default: np.nan)
              (In some cases, it is 1.0 e.g. when working with targeted sequencing data and access target specificity.)
        """
        depth0_similarity = kwargs.get("depth0_similarity", np.nan)
        
        try:
            ref_dict = self._get_depth_dict(ref_validation.file_path)
            alt_dict = self._get_depth_dict(alt_validation.file_path)
            
            if not ref_dict and not alt_dict:
                self.result.similarity = 1.0
                self.result.details["note"] = "No coverage data in either file; treated as identical"
                return

            all_keys = set(ref_dict.keys()) | set(alt_dict.keys())
            if not all_keys:
                self.result.error = "No positions with coverage data"
                return
            
            # Convert to numpy arrays using fromiter for memory efficiency
            n = len(all_keys)
            ref_arr = np.fromiter((ref_dict.get(k, 0) for k in all_keys), dtype=np.int32, count=n)
            alt_arr = np.fromiter((alt_dict.get(k, 0) for k in all_keys), dtype=np.int32, count=n)
            
            # Optimized vectorized similarity calculation
            both_zero = (ref_arr == 0) & (alt_arr == 0)
            max_vals = np.maximum(ref_arr, alt_arr, dtype=np.float64)
            min_vals = np.minimum(ref_arr, alt_arr, dtype=np.float64)
            depth_ratio = np.divide(min_vals, max_vals, out=np.zeros_like(min_vals, dtype=np.float64), where=max_vals > 0)
            depth_similarity = np.where(both_zero, depth0_similarity, depth_ratio)
            
            self.result.similarity = float(np.nanmean(depth_similarity))
            self.result.details.update({
                "n_positions": n,
                "n_depth0_positions": int(np.sum(both_zero)),
                "depth0_similarity": depth0_similarity,
                "avg_ref_depth": float(np.mean(ref_arr)),
                "avg_alt_depth": float(np.mean(alt_arr)),
            })
        except Exception as e:
            logger.error(f"Functional comparison (coverage) failed: {e}")
            self.result.error = f"Functional comparison (coverage) failed: {e}"
    
    def _compare_variant(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Compare BAM/SAM/CRAM files by calling variants using bcftools.
        
        Uses bcftools to generate VCF files from BAM files, then compares the resulting VCF files.
        
        Args:
            **kwargs: Strategy-specific parameters (e.g., vcf_strategy)
            - ref_fasta: Path to reference FASTA file (required for bcftools mpileup)
            - vcf_strategy: Strategy for comparing VCF files ("summary" or "exact", default: "summary")
        """
        vcf_strategy = kwargs.get("vcf_strategy", "summary")
        ref_fasta = kwargs.get("ref_fasta", None)
        if not ref_fasta:
            self.result.error = "Reference FASTA file is required for variant comparison"
            return
        
        ref_vcf_path = None
        alt_vcf_path = None
        try:
            # Generate VCF files for both BAM files
            ref_vcf_path = self._create_vcf(ref_validation.file_path, ref_fasta)
            alt_vcf_path = self._create_vcf(alt_validation.file_path, ref_fasta)
            
            # Compare VCF files
            vcf_handler = VCFHandler()
            vcf_result  = vcf_handler.compare(reference_path=ref_vcf_path, candidate_path=alt_vcf_path, strategy=vcf_strategy, **kwargs)
            self.result.similarity = vcf_result.similarity
            self.result.details.update(vcf_result.details)
        except Exception as e:
            logger.error(f"Functional comparison (variant) failed: {e}")
            self.result.error = f"Functional comparison (variant) failed: {e}"
        finally:
            self._cleanup_file(ref_vcf_path, verbose=True)
            self._cleanup_file(alt_vcf_path, verbose=True)
    
    def _compare_summary(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Summary statistics comparison: read count, mapping percentage, MAPQ distribution similarity."""
        
        ref_file = None
        alt_file = None
        
        try:
            ref_file = self._load_file(ref_validation.file_path, file_format=ref_validation.signature.format)
            alt_file = self._load_file(alt_validation.file_path, file_format=alt_validation.signature.format)
            
            # Get alignment summary stats, and calculate relative absolute error
            ref_flagstat_df = self._get_flagstat_df(ref_validation.file_path)
            alt_flagstat_df = self._get_flagstat_df(alt_validation.file_path)
            flagstat_rae_df = pd.DataFrame({'rae': relative_absolute_error(ref_flagstat_df['pass'].values, 
                                                                           alt_flagstat_df['pass'].values)},
                                            index=ref_flagstat_df.index)
            
            # get mapq stats summary and calculate hellinger similarity (1 - hellinger distance)
            ref_mapq_counter = self._get_mapq_stats(ref_file)['counter']
            alt_mapq_counter = self._get_mapq_stats(alt_file)['counter']
            # convert to numpy arrays
            keys = sorted(set(ref_mapq_counter.keys()) | set(alt_mapq_counter.keys()))
            ref_mapq_arr = np.array([ref_mapq_counter.get(k, 0) for k in keys])
            alt_mapq_arr = np.array([alt_mapq_counter.get(k, 0) for k in keys])
            
            mapq_hellinger_distance = hellinger_distance(ref_mapq_arr, alt_mapq_arr)
            
            # Combine RAE values and hellinger distance, using nanmean to automatically ignore NaN values
            rae_values = np.clip(flagstat_rae_df['rae'].values, 0.0, 1.0)
            all_values = np.concatenate([rae_values, [mapq_hellinger_distance]])
            
            # nanmean automatically ignores NaN values; returns NaN if all values are NaN
            mean_error = np.nanmean(all_values)
            similarity = 1.0 - mean_error if not np.isnan(mean_error) else 0.0
            
            self.result.similarity = similarity
            self.result.details.update({
                "flagstat_rae_df": flagstat_rae_df.to_dict(),
                "mapq_hellinger_distance": float(mapq_hellinger_distance),
            })
        except Exception as e:
            logger.error(f"Summary statistics comparison failed: {e}")
            self.result.error = f"Summary statistics comparison failed: {e}"
        finally:
            self._close_file(ref_file)
            self._close_file(alt_file)

