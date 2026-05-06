"""BED (Browser Extensible Data) format handler.

Strategy:
- exact: Exact region coordinate matching
- approximate: Region-level overlap with tolerance, calculates percentage of regions that match
- overlap: Base-level overlap with precision, recall, and Jaccard similarity
- correlation: Pattern-based similarity using correlation coefficient on score column of shared regions
- summary: Summary statistics comparison of region counts, lengths distribution, and values distribution

Required cmdline tools:
- bedops
- tabix
"""

import logging
import os
import subprocess
import tempfile
from collections import Counter
from typing import Dict, Any, Tuple, Set

import pybedtools
import numpy as np
import pandas as pd

from tools.base import FormatHandler
from tools.results import ValidationResult, ComparisonResult
from utils.format import detect_file_signature
from utils.metrics import hellinger_distance, jaccard_similarity, pearson_correlation, precision_recall_f1, rae_similarity, spearman_correlation

logger = logging.getLogger(__name__)


class BEDHandler(FormatHandler):
    """Handler for BED, bigBed, narrowpeak, broadpeak, bedgraph format files."""
    
    def __init__(self):
        super().__init__("bed")
        self.format_map = {
            "bed": {"min_columns": 3},
            "bigbed": {"min_columns": 3},  # Binary BED format, can have variable columns
            "narrowpeak": {"min_columns": 10},
            "broadpeak": {"min_columns": 10},
            "bedgraph": {"min_columns": 4},  # Text format for continuous signal data (chrom, start, end, value)
        }
        self.supported_formats = list(self.format_map.keys())
        self.supported_strategies = ["exact", "approximate", "overlap", "correlation", "summary"]
    
    # ============================================================================
    # Public API Methods
    # ============================================================================
    
    def validate(self, file_path: str, **kwargs) -> ValidationResult:
        """Validate BED, bigBed, narrowpeak, broadpeak, bedGraph file format.
        
        Args:
            file_path: Path to BED/bigBed file
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
            logger.error(f"BED file not found: {file_path}")
            result.error = "File not found"
            return result
        
        if not is_nonempty:
            logger.warning(f"BED file is empty: {file_path}")
            result.error = "File is empty"
            return result
        
        if not is_format_ok:
            logger.error(f"Not a BED/bigBed file: detected={signature.format}")
            result.error = f"Not a BED/bigBed file: detected={signature.format}"
            return result
        
        # collect statistics using pybedtools
        try:
            bed = self._load_file(file_path)
            # Get min_columns from format_map, default to 3 if format not found
            min_columns = self.format_map.get(signature.format, {}).get("min_columns", 3)
            
            # extract statistics
            df = self._extract_bed_df(bed)
            stats = self._get_bed_stats(df, min_columns=min_columns)
            result.parseable = True
            result.non_empty = stats["num_regions"] > 0
            
            if not result.non_empty:
                logger.warning(f"BED file contains no valid regions: {file_path}")
                result.details.update({"note": "BED file contains no valid regions"})
                return result
            
            result.details.update({
                "num_regions": stats["num_regions"],
                "total_length": stats["total_length"],
                "chroms": sorted(list(stats["chromosomes"]))
            })
            logger.info(f"BED validation passed: {file_path}")
        except Exception as e:
            logger.error(f"BED validation failed: {e}")
            result.error = f"BED validation failed: {e}"
        return result
    
    def compare(self, reference_path: str, candidate_path: str, strategy: str = "approximate", **kwargs) -> ComparisonResult:
        """Compare BED files.
        
        Supported strategies:
        - exact: Exact region coordinate matching
        - approximate: Region-level overlap with tolerance, calculates percentage of regions that match
        - overlap: Base-level overlap with precision, recall, and Jaccard similarity
        - correlation: Pattern-based similarity using correlation coefficient on score column of shared regions
        - summary: Summary statistics comparison of region counts, lengths, and values
        
        Args:
            reference_path: Path to reference BED/bigBed/bedGraph file
            candidate_path: Path to candidate BED/bigBed/bedGraph file
            strategy: Comparison strategy (exact, approximate, overlap, correlation, summary)
            **kwargs: Strategy-specific parameters (e.g., tolerance)
            - tolerance: Positional tolerance in base pairs for approximate comparison (default: 0)
            - coord_only: If True, only compare core 3 columns (chrom, start, end) for approximate/overlap (default: True)
            - overlap_threshold: Minimum overlap percentage relative to reference region to consider regions equivalent for correlation (default: 0.95)
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
                    # Exact comparison: regions must match exactly
                    self._compare_exact(ref_validation, alt_validation, **kwargs)
                elif strategy == "approximate":
                    # Approximate comparison: region-level overlap with tolerance
                    self._compare_approx(ref_validation, alt_validation, **kwargs)
                elif strategy == "overlap":
                    # Overlap comparison: base-level overlap with precision, recall, and Jaccard similarity
                    self._compare_overlap(ref_validation, alt_validation, **kwargs)
                elif strategy == "correlation":
                    # Correlation comparison: pattern-based similarity using correlation coefficient on score column
                    self._compare_correlation(ref_validation, alt_validation, **kwargs)
                elif strategy == "summary":
                    # Summary statistics comparison: region count, average length, length distribution, and value distribution
                    self._compare_summary(ref_validation, alt_validation, **kwargs)
                else:
                    logger.warning(f"Unknown strategy '{strategy}', fall back to approximate")
                    self._compare_approx(ref_validation, alt_validation, **kwargs)
        except Exception as e:
            logger.error(f"BED/bigBed comparison failed: {e}")
            self.result.error = f"BED/bigBed comparison failed: {e}"
        return self._finalize_output(ref_validation, alt_validation, self.result)
    
    # ============================================================================
    # File I/O and Loading Methods
    # ============================================================================
    
    def _load_file(self, file_path: str, coord_only: bool = False, 
                   merge_overlap: bool = False, ignore_order: bool = True) -> pybedtools.BedTool:
        """Load BED file, optionally cut to core columns, sort, and optionally merge.
        
        Args:
            file_path: Path to BED file
            coord_only: If True, only keep core 3 columns (chrom, start, end)
            merge_overlap: If True, merge overlapping intervals after sorting
            ignore_order: If True, sort the file (default: True). If False, preserve original order.
        """
        try:
            bed = pybedtools.BedTool(file_path)
            bed_output = bed.cut(range(3)) if coord_only else bed
            
            # Sort only if ignoring order (for efficient operations) or if merging (merge requires sorted input)
            if ignore_order or merge_overlap:
                bed_output = bed_output.sort()
            
            if merge_overlap:
                bed_output = bed_output.merge()
            return bed_output
        except Exception as e:
            raise ValueError(f"Failed to load BED file: {e}")
    
    # ============================================================================
    # File Status Checkers
    # ============================================================================
    
    def _is_sorted(self, file_path: str) -> bool:
        """Check if BED file is sorted by chromosome, start, and end positions.
        Uses command-line tools (sort-bed) for faster checking.
        """
        # Try sort-bed --check-sort (fastest, from BEDOPS)
        try:
            result = subprocess.run(
                ["sort-bed", "--check-sort", file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=300  # 5 minute timeout
            )
            # sort-bed returns 0 if sorted, non-zero if not sorted or error
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to check if BED file is sorted: {e}")
            return False
    
    def _is_bgzip(self, file_path: str) -> bool:
        """Check if file is bgzip-compressed (BGZF format).
        """
        try:
            if not os.path.exists(file_path):
                return False
            
            # Use detect_file_signature to check compression type
            signature = detect_file_signature(file_path, fallback=True)
            return signature.compression == "bgzf"
        except Exception as e:
            logger.error(f"Failed to check if file is bgzip-compressed: {e}")
            return False
    
    # ============================================================================
    # File Creation Methods
    # ============================================================================
    
    def _create_sorted_bed(self, file_path: str) -> str:
        """Create a sorted BED file using pybedtools.
        Returns path to temporary file (caller must clean up after use).
        """
        # Create temporary file for sorted output
        fd, temp_path = tempfile.mkstemp(suffix='.bed', prefix='sorted_')
        os.close(fd)  # Close file descriptor, we'll use the path with pybedtools
        
        try:
            # Load, sort, and save using pybedtools
            bed = self._load_file(file_path)
            bed_sorted = bed.sort()
            bed_sorted.saveas(temp_path)
            
            return temp_path
        except Exception as e:
            self._cleanup_file(temp_path)
            raise
    
    def _create_bgzip_bed(self, file_path: str) -> str:
        """Create a bgzip-compressed BED file from a BED file (will be sorted before compression).
        Returns path to temporary compressed file (caller must clean up after use).
        """
        temp_sorted_path = None
        temp_gz_path = None
        try:
            # Sort the BED file first
            temp_sorted_path = self._create_sorted_bed(file_path)
            # Create temporary file for compressed output
            fd, temp_gz_path = tempfile.mkstemp(suffix='.bed.gz', prefix='bgzip_')
            os.close(fd)  # Close file descriptor, we'll use the path with bgzip
            # Compress using bgzip
            with open(temp_gz_path, 'wb') as out_file:
                result = subprocess.run(
                    ["bgzip", "-c", temp_sorted_path],
                    stdout=out_file,
                    stderr=subprocess.PIPE,
                    timeout=600,  # 10 minute timeout for compression
                    check=False
                )
            
            if result.returncode != 0:
                error_msg = result.stderr.decode() if result.stderr else 'Unknown error'
                raise RuntimeError(f"bgzip failed: {error_msg}")
            
            if not os.path.exists(temp_gz_path):
                raise RuntimeError(f"Compressed file was not created: {temp_gz_path}")
            
            return temp_gz_path
        except Exception as e:
            self._cleanup_file(temp_gz_path)
            raise
        finally:
            # Always clean up the intermediate sorted file
            self._cleanup_file(temp_sorted_path, verbose=True)

    def _create_bgzip_tbi(self, file_path: str) -> str:
        """Create a bgzip-compressed BED file and its Tabix index file (.tbi).
        Returns path to index file (caller must clean up after use).
        """
        temp_gz_path = None
        try:
            # Create bgzip-compressed BED file (this also sorts the file)
            temp_gz_path = self._create_bgzip_bed(file_path)
            index_path = temp_gz_path + '.tbi'
            
            # Create Tabix index on the compressed file
            result = subprocess.run(
                ["tabix", "-p", "bed", temp_gz_path],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for indexing
                check=False
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"tabix failed: {result.stderr}")
            
            if not os.path.exists(index_path):
                raise RuntimeError(f"Index file was not created: {index_path}")
            
            return index_path
        except Exception as e:
            # Clean up compressed file on error
            self._cleanup_file(temp_gz_path)
            raise

    # ============================================================================
    # Statistics and Analysis Methods
    # ============================================================================
    
    def _extract_bed_df(self, bed: pybedtools.BedTool) -> pd.DataFrame:
        """Convert pybedtools.BedTool to pandas DataFrame with core columns 
        (chrom, start, end) and additional fields if present.
        """
        try:
            # bedGraph / BED files may start with `track ...` or `browser ...`
            # header lines (and `#` comments). pybedtools' iterator skips them,
            # but to_dataframe() reads the file via pandas directly and would
            # otherwise parse them as data rows. Count and skip them here.
            skiprows = 0
            try:
                with open(bed.fn) as fh:
                    for line in fh:
                        if line.startswith(("track", "browser", "#")):
                            skiprows += 1
                        else:
                            break
            except Exception:
                skiprows = 0
            df = bed.to_dataframe(skiprows=skiprows) if skiprows else bed.to_dataframe()
            # Ensure start and end are integers
            if 'start' in df.columns:
                df['start'] = df['start'].astype(int)
            if 'end' in df.columns:
                df['end'] = df['end'].astype(int)
            return df
        except Exception as e:
            # Fallback: manual conversion if to_dataframe fails
            logger.warning(f"to_dataframe() failed, using manual conversion: {e}")
            data = []
            for feature in bed:
                if feature.chrom and feature.start is not None and feature.end is not None:
                    row = {
                        'chrom': feature.chrom,
                        'start': int(feature.start),
                        'end': int(feature.end)
                    }
                    # Add additional fields if present
                    for i, field in enumerate(feature.fields[3:], start=3):
                        row[f'field_{i}'] = field
                    data.append(row)
            return pd.DataFrame(data)
    
    def _get_length_counter(self, df: pd.DataFrame) -> Counter:
        """Get region length distribution from pandas DataFrame.
        Returns the Counter mapping length (in bp) to count of regions with that length.
        """
        if df.empty:
            return Counter()
        lengths = (df['end'] - df['start']).astype(int)
        return Counter(lengths)
    
    def _get_score_stats(self, df: pd.DataFrame, score_index: int = 4) -> Dict[str, float]:
        """Extract score statistics from specified column index (typically score column).
        
        Args:
            df: DataFrame containing BED data
            score_index: Column index for score column (default: 4)
        
        Returns:
            Dictionary with min, max, avg, median, q25, q75 statistics.
            Returns empty dictionary if no valid scores found.
        """
        stats_dict = {}
        if df.empty or score_index < 0 or score_index >= len(df.columns):
            return stats_dict
        
        try:
            # Access column by position (n-th column)
            values = pd.to_numeric(df.iloc[:, score_index], errors='coerce').dropna()
            if len(values) > 0:
                stats_dict = {
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "avg": float(values.mean()),
                    "median": float(values.median()),
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                }
        except Exception:
            pass
        
        return stats_dict
    
    def _get_bed_stats(self, df: pd.DataFrame, min_columns: int = 3) -> Dict[str, Any]:
        """Extract statistics from BED file, designed for BED6 format, also works for bigBed.
        
        BED format: chrom, start, end, name, score, strand
        
        Args:
            df: DataFrame containing BED data
            min_columns: Minimum number of columns expected (default: 3)
        
        Returns:
            Dictionary containing num_regions, total_length, chromosomes, and optionally value_stats
        """
        bed_stats = {"num_regions": 0, "total_length": 0, "chromosomes": []}
        if df.empty:
            return bed_stats
        
        # Filter valid rows
        valid_df = df[(df['chrom'].notna()) & (df['start'].notna()) & (df['end'].notna())]
        
        if valid_df.empty:
            return bed_stats
        
        # Calculate length and update summary statistics
        valid_df = valid_df.copy()
        valid_df['length'] = valid_df['end'] - valid_df['start']
        bed_stats["num_regions"] = len(valid_df)
        bed_stats["total_length"] = int(valid_df['length'].sum())
        bed_stats["chromosomes"] = sorted(valid_df['chrom'].unique().tolist())
        
        # Get score statistics if format has 4+ columns (bedGraph has 4, BED5+ has 5+)
        if min_columns >= 4:
            # For bedGraph (4 columns), value is at index 3; for BED5+, it's at index 4
            score_index = 3 if min_columns == 4 else 4
            value_stats = self._get_score_stats(valid_df, score_index=score_index)
            if value_stats:
                bed_stats["value_stats"] = value_stats
        
        return bed_stats
    
    # ============================================================================
    # Record Comparison Helpers
    # ============================================================================
    
    def _get_overlap_regions(self, ref_bed: pybedtools.BedTool, alt_bed: pybedtools.BedTool, 
                             overlap_threshold: float, tolerance: int = 0) -> pd.DataFrame:
        """Get filtered overlaps as DataFrame with overlap calculation.
        
        Performs intersection, converts to DataFrame, properly renames columns with ref_/alt_ prefixes,
        calculates overlap ratio, and filters by threshold. This is a common operation used by both 
        approximate and correlation comparison methods.
        
        Args:
            ref_bed: Reference BedTool
            alt_bed: Candidate BedTool (will be extended by tolerance if tolerance > 0)
            overlap_threshold: Minimum overlap percentage relative to reference region to consider regions equivalent
            tolerance: Positional tolerance in base pairs to extend alt intervals (default: 0)
        
        Returns:
            DataFrame with columns prefixed with ref_ and alt_ (e.g., ref_chrom, ref_start, ref_end, 
            ref_score, alt_chrom, alt_start, alt_end, alt_score), plus overlap_bp and overlap_ratio.
            Already filtered by overlap_threshold and non-zero lengths.
        """
        # Get column structure directly from BedTool objects
        # Check first feature to determine number of columns
        try:
            num_ref_cols = len(next(iter(ref_bed)).fields)
        except (StopIteration, AttributeError):
            num_ref_cols = 3  # Default to BED3
        
        try:
            num_alt_cols = len(next(iter(alt_bed)).fields)
        except (StopIteration, AttributeError):
            num_alt_cols = 3  # Default to BED3
        
        # Standard BED column names (BED3, BED4, BED5, BED6, BED12)
        bed_column_names = ['chrom', 'start', 'end', 'name', 'score', 'strand', 
                           'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'blockStarts']
        
        # Get column names for ref and alt (use standard names or field_N for additional columns)
        ref_col_names = [bed_column_names[i] if i < len(bed_column_names) else f'field_{i}' 
                        for i in range(num_ref_cols)]
        alt_col_names = [bed_column_names[i] if i < len(bed_column_names) else f'field_{i}' 
                        for i in range(num_alt_cols)]
        
        # Extend alt intervals by tolerance if specified
        if tolerance > 0:
            chromsizes = {}
            for feature in alt_bed:
                max_pos = feature.end + tolerance + 1
                if feature.chrom not in chromsizes or max_pos > chromsizes[feature.chrom][1]:
                    chromsizes[feature.chrom] = (0, max_pos)
            alt_bed_for_intersect = alt_bed.slop(b=tolerance, genome=chromsizes)
        else:
            alt_bed_for_intersect = alt_bed
        
        # Perform intersection
        overlaps = ref_bed.intersect(alt_bed_for_intersect, wa=True, wb=True)
        
        # Convert overlaps to DataFrame
        try:
            overlap_df = overlaps.to_dataframe()
        except Exception as e:
            # Fallback: manual conversion if to_dataframe fails
            logger.warning(f"to_dataframe() failed for overlaps, using manual conversion: {e}")
            overlap_data = []
            for overlap in overlaps:
                if len(overlap.fields) >= 6:
                    try:
                        row = {}
                        # Add ref columns
                        for i in range(min(num_ref_cols, len(overlap.fields))):
                            col_name = ref_col_names[i] if i < len(ref_col_names) else f'field_{i}'
                            row[f'ref_{col_name}'] = overlap.fields[i]
                        # Add alt columns
                        for i in range(min(num_alt_cols, len(overlap.fields) - num_ref_cols)):
                            col_name = alt_col_names[i] if i < len(alt_col_names) else f'field_{i}'
                            row[f'alt_{col_name}'] = overlap.fields[num_ref_cols + i]
                        overlap_data.append(row)
                    except (ValueError, IndexError):
                        continue
            overlap_df = pd.DataFrame(overlap_data)
        
        if overlap_df.empty:
            return overlap_df
        
        # Rename columns properly: ref columns get ref_ prefix, alt columns get alt_ prefix
        # pybedtools preserves all columns: first num_ref_cols from ref, next num_alt_cols from alt
        rename_dict = {}
        
        # Rename ref columns (first num_ref_cols columns)
        for i in range(min(num_ref_cols, len(overlap_df.columns))):
            old_name = overlap_df.columns[i]
            col_name = ref_col_names[i] if i < len(ref_col_names) else f'field_{i}'
            rename_dict[old_name] = f'ref_{col_name}'
        
        # Rename alt columns (next num_alt_cols columns)
        for i in range(num_ref_cols, min(num_ref_cols + num_alt_cols, len(overlap_df.columns))):
            old_name = overlap_df.columns[i]
            alt_idx = i - num_ref_cols
            col_name = alt_col_names[alt_idx] if alt_idx < len(alt_col_names) else f'field_{alt_idx}'
            rename_dict[old_name] = f'alt_{col_name}'
        
        overlap_df = overlap_df.rename(columns=rename_dict)
        
        if overlap_df.empty:
            return overlap_df
        
        # Ensure start and end are integers
        overlap_df['ref_start'] = overlap_df['ref_start'].astype(int)
        overlap_df['ref_end'] = overlap_df['ref_end'].astype(int)
        overlap_df['alt_start'] = overlap_df['alt_start'].astype(int)
        overlap_df['alt_end'] = overlap_df['alt_end'].astype(int)
        
        # Calculate overlap in base pairs
        overlap_df['overlap_bp'] = (
            np.maximum(0, 
                np.minimum(overlap_df['ref_end'], overlap_df['alt_end']) - 
                np.maximum(overlap_df['ref_start'], overlap_df['alt_start'])
            )
        )
        
        # Calculate overlap ratio relative to reference region
        ref_length = overlap_df['ref_end'] - overlap_df['ref_start']
        alt_length = overlap_df['alt_end'] - overlap_df['alt_start']
        overlap_df['overlap_ratio'] = overlap_df['overlap_bp'] / ref_length
        
        # Filter by overlap threshold and remove zero-length regions
        overlap_df = overlap_df[
            (overlap_df['overlap_ratio'] >= overlap_threshold) &
            (ref_length > 0) & (alt_length > 0)
        ]
        
        return overlap_df
    
    # ============================================================================
    # Private Comparison Implementation Methods
    # ============================================================================
    
    def _compare_exact(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Exact comparison: regions must match exactly.
        
        Compares regions sequentially, preserving order. Suitable for comparing
        tools that perform coordinate sorting.
        
        Criteria:
        - All regions identical in the same order (line-by-line comparison)
        - Compare core coordinates: chrom, start, end
        - Must be 100% identical
        
        Args:
            ref_validation: Validation result for reference file
            alt_validation: Validation result for candidate file
            **kwargs: Strategy-specific parameters
            - coord_only: If True, only compare core 3 columns (chrom, start, end) (default: True)
            - ignore_order: If True, compare regions regardless of order (default: False)
            - merge_overlap: If True, merge overlapping intervals before comparison (default: False)
        """
        merge_overlap = kwargs.get("merge_overlap", False)
        coord_only = kwargs.get("coord_only", True)
        ignore_order = kwargs.get("ignore_order", False)
        
        try:
            # Load files with specified parameters
            ref_bed = self._load_file(ref_validation.file_path, coord_only=coord_only, merge_overlap=merge_overlap, ignore_order=ignore_order)
            alt_bed = self._load_file(alt_validation.file_path, coord_only=coord_only, merge_overlap=merge_overlap, ignore_order=ignore_order)
            
            # Compare regions exactly
            # Check if counts match first
            ref_count = len(ref_bed)
            alt_count = len(alt_bed)
            
            if ref_count != alt_count:
                similarity = 0.0
            elif ignore_order:
                # Order-agnostic comparison: use intersect to find matching regions
                # intersect with -u returns unique entries from first file that overlap
                ref_in_alt = ref_bed.intersect(alt_bed, u=True)
                similarity = 1.0 if (len(ref_in_alt) == ref_count) else 0.0
            else:
                # Order-sensitive comparison: compare regions element by element in order
                matching_regions = 0
                ref_list = list(ref_bed)
                alt_list = list(alt_bed)
                
                # Compare each region in order
                for ref_feature, alt_feature in zip(ref_list, alt_list):
                    if (ref_feature.chrom == alt_feature.chrom and 
                        ref_feature.start == alt_feature.start and 
                        ref_feature.end == alt_feature.end):
                        matching_regions += 1
                    else:
                        break
                
                # Exact match only if all regions match in order and counts are equal
                similarity = 1.0 if (matching_regions == ref_count) else 0.0
            
            self.result.similarity = similarity
            self.result.details.update({
                "total_ref_regions": ref_count,
                "total_alt_regions": alt_count,
            })
        except Exception as e:
            logger.error(f"Exact comparison failed: {e}")
            self.result.error = f"Exact comparison failed: {e}"
    
    def _compare_approx(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Approximate comparison: region-level overlap with tolerance.
        
        Calculates percentage of regions that match with tolerance.
        Useful for peak calling comparisons, gene annotation comparisons, etc.
        Counts overlapping regions (not bases) and calculates precision, recall, and Jaccard similarity.
        
        Args:
            ref_validation: Validation result for reference file
            alt_validation: Validation result for candidate file
            **kwargs: Strategy-specific parameters
            - tolerance: Positional tolerance in base pairs (default: 0)
            - overlap_threshold: Minimum overlap percentage relative to reference region to consider regions equivalent (default: 0.95)
            - coord_only: If True, only compare core 3 columns (chrom, start, end) (default: True)
        """
        coord_only = kwargs.get("coord_only", True)
        merge_overlap = kwargs.get("merge_overlap", False)
        ignore_order = kwargs.get("ignore_order", True)
        
        tolerance = kwargs.get("tolerance", 0)
        overlap_threshold = kwargs.get("overlap_threshold", 0.95)
        
        try:
            # Load and prepare files
            ref_bed = self._load_file(ref_validation.file_path, coord_only=coord_only, merge_overlap=merge_overlap, ignore_order=ignore_order)
            alt_bed = self._load_file(alt_validation.file_path, coord_only=coord_only, merge_overlap=merge_overlap, ignore_order=ignore_order)
            
            ref_count = len(ref_bed)
            alt_count = len(alt_bed)
            
            if ref_count == 0:
                self.result.error = f"No reference regions in {ref_validation.file_path}"
                return
            if alt_count == 0:
                self.result.error = f"No candidate regions in {alt_validation.file_path}"
                return
            
            # Get overlap regions and filter by overlap threshold
            # Note: tolerance only affects which pairs are considered; overlap is calculated on original regions
            overlap_df = self._get_overlap_regions(ref_bed, alt_bed, overlap_threshold, tolerance=tolerance)
            
            # Count unique matched regions (deduplicate in case same region overlaps multiple times)
            if overlap_df.empty:
                ref_matched_count = 0
                alt_matched_count = 0
            else:
                ref_matched_count = overlap_df[['ref_chrom', 'ref_start', 'ref_end']].drop_duplicates().shape[0]
                alt_matched_count = overlap_df[['alt_chrom', 'alt_start', 'alt_end']].drop_duplicates().shape[0]
            
            # Calculate precision and recall
            precision = alt_matched_count / alt_count if alt_count > 0 else 0.0
            recall = ref_matched_count / ref_count if ref_count > 0 else 0.0
            
            # Calculate Jaccard similarity: |A ∩ B| / |A ∪ B|
            matched_count = ref_matched_count  # Intersection size
            jaccard = jaccard_similarity(ref_count, alt_count, matched_count)
            
            self.result.similarity = jaccard
            self.result.details.update({
                "total_ref_regions": ref_count,
                "total_alt_regions": alt_count,
                "ref_matched_regions": ref_matched_count,
                "alt_matched_regions": alt_matched_count,
                "precision": precision,
                "recall": recall,
                "jaccard_similarity": jaccard,
                "tolerance": tolerance,
                "overlap_threshold": overlap_threshold
            })
        except Exception as e:
            logger.error(f"Approximate comparison failed: {e}")
            self.result.error = f"Approximate comparison failed: {e}"

    def _compare_overlap(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Overlap comparison: base-level overlap with precision, recall, and Jaccard similarity.
        
        Uses intersect to calculate overlapping bases and computes:
        - Precision: proportion of alt bases that overlap with ref bases
        - Recall: proportion of ref bases that overlap with alt bases
        - Jaccard similarity: intersection over union of bases

        Args:
            ref_validation: Validation result for reference file
            alt_validation: Validation result for candidate file
            **kwargs: Strategy-specific parameters
            - coord_only: If True, only compare core 3 columns (chrom, start, end) (default: True)
        """
        coord_only = kwargs.get("coord_only", True)
        merge_overlap = kwargs.get("merge_overlap", True)
        ignore_order = kwargs.get("ignore_order", True)
        
        try:
            # Load files, optionally cut to core columns, sort, and merge to get unique base overlap
            ref_bed = self._load_file(ref_validation.file_path, coord_only=coord_only, merge_overlap=merge_overlap, ignore_order=ignore_order)
            alt_bed = self._load_file(alt_validation.file_path, coord_only=coord_only, merge_overlap=merge_overlap, ignore_order=ignore_order)
            
            # Get statistics from merged files
            ref_df_merged = self._extract_bed_df(ref_bed)
            alt_df_merged = self._extract_bed_df(alt_bed)
            ref_bed_stats = self._get_bed_stats(ref_df_merged, min_columns=3)
            alt_bed_stats = self._get_bed_stats(alt_df_merged, min_columns=3)
            
            ref_count = ref_bed_stats["num_regions"]
            alt_count = alt_bed_stats["num_regions"]
            total_ref_bases = ref_bed_stats["total_length"]
            total_alt_bases = alt_bed_stats["total_length"]
            
            if ref_count == 0 or total_ref_bases == 0:
                self.result.error = f"No reference regions in {ref_validation.file_path}"
                return
            if alt_count == 0 or total_alt_bases == 0:
                self.result.error = f"No candidate regions in {alt_validation.file_path}"
                return
            
            # Use intersect to find overlapping intervals between merged files
            # u=True: write original entry once for each overlap
            ref_overlap_intervals = ref_bed.intersect(alt_bed, u=True)
            overlapping_bases = sum(f.end - f.start for f in ref_overlap_intervals.merge())
            precision, recall, f1_score = precision_recall_f1(total_ref_bases, total_alt_bases, overlapping_bases)
            jaccard = jaccard_similarity(total_ref_bases, total_alt_bases, overlapping_bases)
            
            # Use Jaccard similarity as the main similarity metric
            self.result.similarity = jaccard
            self.result.details.update({
                "total_ref_regions_merged": ref_count,
                "total_alt_regions_merged": alt_count,
                "total_ref_bases_merged": total_ref_bases,
                "total_alt_bases_merged": total_alt_bases,
                "overlapping_bases_merged": overlapping_bases,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "jaccard_similarity": jaccard
            })
        except Exception as e:
            logger.error(f"Overlap comparison failed: {e}")
            self.result.error = f"Overlap comparison failed: {e}"

    def _compare_correlation(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Correlation comparison: Pattern-based similarity using correlation coefficient on score column of shared regions.
        
        Compares score values from overlapping regions between reference and candidate files using
        Pearson or Spearman correlation. Measures how scores co-vary (pattern similarity) rather than
        absolute value agreement. Scale-invariant: signals with same pattern but different scales
        will have high correlation.
        
        Args:
            ref_validation: Validation result for reference file
            alt_validation: Validation result for candidate file
            **kwargs: Strategy-specific parameters
            - overlap_threshold: Minimum overlap percentage relative to reference region to consider regions equivalent (default: 0.95)
            - correlation_method: Correlation method - 'pearson' or 'spearman' (default: 'pearson')
        """
        overlap_threshold = kwargs.get('overlap_threshold', 0.95)
        correlation_method = kwargs.get('correlation_method', 'pearson').lower()
        
        if correlation_method not in ['pearson', 'spearman']:
            logger.warning(f"Unknown correlation method '{correlation_method}', using 'pearson'")
            correlation_method = 'pearson'
        
        try:
            # Load files with all columns (need score column)
            ref_bed = self._load_file(ref_validation.file_path, coord_only=False)
            alt_bed = self._load_file(alt_validation.file_path, coord_only=False)
            
            # Extract DataFrames to access score column
            ref_df = self._extract_bed_df(ref_bed)
            alt_df = self._extract_bed_df(alt_bed)
            
            # Check if both files have score column (at least 4 columns)
            min_columns = max(ref_df.shape[1], alt_df.shape[1])
            if min_columns < 4:
                self.result.similarity = 0.0
                self.result.details.update({
                    "ref_columns": ref_df.shape[1],
                    "alt_columns": alt_df.shape[1],
                })
                self.result.error = "Both files must have at least 4 columns (score column required)"
                return
            
            # Determine score column index (bedGraph uses index 3, BED5+ uses index 4)
            score_index = 3 if min_columns == 4 else 4
            if ref_df.shape[1] <= score_index or alt_df.shape[1] <= score_index:
                self.result.similarity = 0.0
                self.result.details.update({
                    "ref_columns": ref_df.shape[1],
                    "alt_columns": alt_df.shape[1],
                    "score_index": score_index,
                })
                self.result.error = "Score column not found in one or both files"
                return
            
            # Get overlap regions with properly renamed columns (including scores)
            overlap_df = self._get_overlap_regions(ref_bed, alt_bed, overlap_threshold)
            
            if overlap_df.empty:
                self.result.similarity = 0.0
                self.result.details.update({
                    "ref_regions": len(ref_df),
                    "alt_regions": len(alt_df),
                    "matched_regions": 0,
                })
                self.result.error = "No regions meet overlap threshold"
                return
            
            # Extract score columns (now properly named with ref_/alt_ prefixes)
            # Standard BED column names
            bed_column_names = ['chrom', 'start', 'end', 'name', 'score', 'strand', 
                               'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'blockStarts']
            score_col_name = bed_column_names[score_index] if score_index < len(bed_column_names) else f'field_{score_index}'
            ref_score_col = f'ref_{score_col_name}'
            alt_score_col = f'alt_{score_col_name}'
            
            # Convert to numeric if not already
            if ref_score_col in overlap_df.columns:
                overlap_df['ref_score'] = pd.to_numeric(overlap_df[ref_score_col], errors='coerce')
            else:
                overlap_df['ref_score'] = np.nan
            
            if alt_score_col in overlap_df.columns:
                overlap_df['alt_score'] = pd.to_numeric(overlap_df[alt_score_col], errors='coerce')
            else:
                overlap_df['alt_score'] = np.nan
            
            # Filter out rows with NaN scores
            overlap_df = overlap_df[overlap_df['ref_score'].notna() & overlap_df['alt_score'].notna()]
            
            if len(overlap_df) < 2:
                self.result.similarity = 0.0
                self.result.details.update({
                    "ref_regions": len(ref_df),
                    "alt_regions": len(alt_df),
                    "matched_regions": len(overlap_df),
                })
                self.result.error = "Insufficient matched regions for correlation (need at least 2)"
                return
            
            # Calculate correlation directly from DataFrame columns
            ref_scores = overlap_df['ref_score'].values
            alt_scores = overlap_df['alt_score'].values
            
            if correlation_method == 'pearson':
                corr = pearson_correlation(ref_scores, alt_scores, nan_policy='omit')
            else:  # spearman
                corr = spearman_correlation(ref_scores, alt_scores, nan_policy='omit')
            
            # Handle NaN or invalid correlation (e.g., constant input)
            if np.isnan(corr) or not np.isfinite(corr):
                # Fallback to RAE similarity when correlation is undefined (e.g., constant input)
                rae_sim = rae_similarity(ref_scores, alt_scores)
                fallback_similarity = float(np.nanmean(rae_sim))
                
                ref_count = len(ref_df)
                alt_count = len(alt_df)
                matched_count = len(overlap_df)
                coverage_ratio = matched_count / max(ref_count, alt_count) if max(ref_count, alt_count) > 0 else 0.0
                
                self.result.similarity = fallback_similarity * coverage_ratio
                self.result.details.update({
                    "ref_regions": ref_count,
                    "alt_regions": alt_count,
                    "matched_regions": matched_count,
                    "coverage_ratio": coverage_ratio,
                    "correlation": None,
                    "correlation_method": correlation_method,
                    "fallback_strategy": "rae_similarity",
                    "fallback_reason": "Correlation undefined (constant input)",
                    "rae_similarity": fallback_similarity,
                    "overlap_threshold": overlap_threshold,
                })
                return
            
            # Convert correlation (-1 to 1) to similarity (0 to 1) using absolute value
            similarity = abs(corr)
            
            # Account for region coverage ratio
            ref_count = len(ref_df)
            alt_count = len(alt_df)
            matched_count = len(overlap_df)
            coverage_ratio = matched_count / max(ref_count, alt_count) if max(ref_count, alt_count) > 0 else 0.0
            
            # Final similarity accounts for both correlation and coverage
            self.result.similarity = similarity * coverage_ratio
            self.result.details.update({
                "ref_regions": ref_count,
                "alt_regions": alt_count,
                "matched_regions": matched_count,
                "coverage_ratio": coverage_ratio,
                "correlation": float(corr),
                "correlation_similarity": similarity,
                "correlation_method": correlation_method,
                "overlap_threshold": overlap_threshold,
            })
        except Exception as e:
            logger.error(f"Correlation comparison failed: {e}")
            self.result.error = f"Correlation comparison failed: {e}"

    def _compare_summary(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Summary statistics comparison: region count, average length, length distribution, and value distribution.
        
        Compares:
        - Number of regions using min/max ratio (consistent with other handlers)
        - Average region length using min/max ratio (more meaningful than total length)
        - Region length distributions using Hellinger distance (used as final similarity)
        - Value distributions (if score column present) using Hellinger distance
        
        Args:
            ref_validation: Validation result for reference file
            alt_validation: Validation result for candidate file
            **kwargs: Strategy-specific parameters
            - num_bins: Number of bins for histogram when comparing value distributions (default: 50)
        """
        num_bins = kwargs.get('num_bins', 50)
        
        try:
            # Load files
            ref_bed = self._load_file(ref_validation.file_path)
            alt_bed = self._load_file(alt_validation.file_path)
            
            # Extract DataFrames once and reuse for all operations
            ref_df = self._extract_bed_df(ref_bed)
            alt_df = self._extract_bed_df(alt_bed)
            min_columns = max(ref_df.shape[1], alt_df.shape[1])
            
            # Get statistics from both files
            ref_stats = self._get_bed_stats(ref_df, min_columns=min_columns)
            alt_stats = self._get_bed_stats(alt_df, min_columns=min_columns)
            
            # Compare number of regions using RAE-based similarity
            ref_count = ref_stats["num_regions"]
            alt_count = alt_stats["num_regions"]
            region_count_similarity = float(rae_similarity([ref_count], [alt_count])[0])
            
            # Compare average region length (more meaningful than total length) using RAE-based similarity
            ref_avg_length = ref_stats["total_length"] / ref_count if ref_count > 0 else 0.0
            alt_avg_length = alt_stats["total_length"] / alt_count if alt_count > 0 else 0.0
            avg_length_similarity = float(rae_similarity([ref_avg_length], [alt_avg_length])[0])
            
            # Compare region length distributions using Hellinger distance
            ref_length_counter = self._get_length_counter(ref_df)
            alt_length_counter = self._get_length_counter(alt_df)
            all_lengths = sorted(set(ref_length_counter.keys()) | set(alt_length_counter.keys()))
            
            if all_lengths:
                ref_length_arr = np.array([ref_length_counter.get(k, 0) for k in all_lengths], dtype=float)
                alt_length_arr = np.array([alt_length_counter.get(k, 0) for k in all_lengths], dtype=float)
                length_hellinger_distance = hellinger_distance(ref_length_arr, alt_length_arr)
                length_dist_similarity = 1.0 - (length_hellinger_distance if not np.isnan(length_hellinger_distance) else 1.0)
            else:
                length_hellinger_distance = 0.0
                length_dist_similarity = 1.0 if ref_count == alt_count == 0 else 0.0
            
            # Compare value distributions if score column present
            value_dist_similarity = np.nan
            if min_columns >= 4 and "value_stats" in ref_stats and "value_stats" in alt_stats:
                try:
                    # For bedGraph (4 columns), value is at index 3; for BED5+, it's at index 4
                    score_index = 3 if min_columns == 4 else 4
                    if ref_df.shape[1] > score_index and alt_df.shape[1] > score_index:
                        ref_values = pd.to_numeric(ref_df.iloc[:, score_index], errors='coerce').dropna()
                        alt_values = pd.to_numeric(alt_df.iloc[:, score_index], errors='coerce').dropna()
                        
                        if len(ref_values) > 0 and len(alt_values) > 0:
                            # Create histograms with common bins
                            bins = np.linspace(min(ref_values.min(), alt_values.min()), 
                                               max(ref_values.max(), alt_values.max()), num=num_bins)
                            ref_hist, _ = np.histogram(ref_values, bins=bins)
                            alt_hist, _ = np.histogram(alt_values, bins=bins)
                            value_hellinger_distance = hellinger_distance(ref_hist, alt_hist)
                            value_dist_similarity = 1.0 - (value_hellinger_distance if not np.isnan(value_hellinger_distance) else 1.0)
                except Exception as e:
                    # Handle cases where score column has invalid data (empty, ".", or other issues)
                    logger.debug(f"Could not compare value distributions: {e}")
                    value_dist_similarity = np.nan
            
            # Use length distribution similarity as the final similarity score
            self.result.similarity = length_dist_similarity
            self.result.details.update({
                "ref_stats": ref_stats,
                "alt_stats": alt_stats,
                "region_count_similarity": region_count_similarity,
                "avg_length_similarity": avg_length_similarity,
                "ref_avg_length": ref_avg_length,
                "alt_avg_length": alt_avg_length,
                "length_dist_similarity": length_dist_similarity,
                "value_dist_similarity": value_dist_similarity,
                "has_score_column": min_columns >= 4
            })
        except Exception as e:
            logger.error(f"Summary statistics comparison failed: {e}")
            self.result.error = f"Summary statistics comparison failed: {e}"

