"""
VCF (Variant Call Format) handler.

Strategy:
- exact: Exact variant record matching including position, alleles, and annotations
- approximate: Variant matching with position window tolerance and allele similarity
- summary: Summary statistics comparison of variant overlap ratio, precision/recall metrics, and aggregate variant statistics

Required cmdline tools:
- bcftools
"""

import logging
import os
import subprocess
import tempfile
from io import StringIO
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import pysam

from tools.base import FormatHandler
from tools.results import ValidationResult, ComparisonResult
from utils.format import detect_file_signature
from utils.metrics import jaccard_similarity, precision_recall_f1, rae_similarity

logger = logging.getLogger(__name__)


class VCFHandler(FormatHandler):
    """Handler for VCF/BCF format files."""
    
    def __init__(self):
        super().__init__("vcf")
        self.format_map = {
            "vcf": {"mode": "r", "file_ext": "vcf", "index_ext": "tbi"},
            "bcf": {"mode": "rb", "file_ext": "bcf", "index_ext": "csi"},
        }
        self.supported_formats = list(self.format_map.keys())
        self.supported_strategies = ["exact", "approximate", "summary"]
    
    # ============================================================================
    # Public API Methods
    # ============================================================================
    
    def validate(self, file_path: str, **kwargs) -> ValidationResult:
        """Validate VCF/BCF file format.
        non-empty check is performed by file size, later can be updated by reading content if needed
        
        Args:
            file_path: Path to VCF/BCF file
            **kwargs: optional parameters (e.g., min_size)
            - min_size: Minimum size of the file in bytes, default is 0
        """
        # check existence and non-emptiness (by file size)
        is_exists = self._is_file_exists(file_path)
        is_nonempty = self._is_file_nonempty(file_path, min_size=kwargs.get("min_size", 0))
        
        # check file format
        signature = detect_file_signature(file_path, fallback="extension")
        is_format_ok = signature.format in self.supported_formats

        # initialize validation result
        result = ValidationResult(exists=is_exists, non_empty=is_nonempty, format_ok=is_format_ok, parseable=False, 
                                  file_path=file_path, signature=signature, details={}, error=None)

        if not is_exists:
            logger.error(f"VCF/BCF file not found: {file_path}")
            result.error = "File not found"
            return result
        
        if not is_nonempty:
            logger.warning(f"VCF/BCF file is empty: {file_path}")
            result.error = "File is empty"
            return result
        
        if not is_format_ok:
            logger.warning(f"Unsupported VCF/BCF file format: {signature.format}")
            result.error = f"Unsupported file format: {signature.format}"
            return result
        
        # validate by parsing the VCF/BCF file using bcftools stats
        try:
            # Verify file can be opened (quick check)
            variant_file = self._load_file(file_path)
            variant_file.close()
            
            # get VCF/BCF statistics using bcftools stats
            stats = self._get_vcf_stats(file_path)
            result.parseable = True
            result.details.update(stats)
            
            # update non_empty status based on actual content
            variant_count = stats.get("num_variants", 0)
            sample_count = stats.get("num_samples", 0)
            result.non_empty = variant_count > 0

            if not result.non_empty:
                logger.warning(f"VCF/BCF file contains no variants: {file_path}")
                result.error = "VCF/BCF file contains no variants"
            else:
                logger.info(f"VCF/BCF validation passed: {variant_count} variants, {sample_count} samples")
        except Exception as e:
            logger.error(f"VCF/BCF validation failed: {e}")
            result.error = f"VCF/BCF validation failed: {e}"
        return result
    
    def compare(self, reference_path: str, candidate_path: str, strategy: str = "summary", **kwargs) -> ComparisonResult:
        """Compare VCF/BCF files.
        
        Supported strategies:
        - exact: Exact variant record matching including position, alleles, and annotations
        - approximate: Variant matching with position window tolerance and allele similarity
        - summary: Summary statistics comparison of variant overlap ratio, precision/recall metrics, and aggregate variant statistics
        
        Args:
            reference_path: Path to reference VCF/BCF file
            candidate_path: Path to candidate VCF/BCF file
            strategy: Comparison strategy (exact, approximate, summary)
            **kwargs: Strategy-specific parameters (e.g., position_window, require_allele_match)
            - position_window: Maximum allowed position difference in base pairs (required for approximate strategy, default: 10)
            - require_allele_match: If True, requires exact allele match (required for approximate strategy, default: True)
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
                    # Exact comparison: all variants must match exactly
                    self._compare_exact(ref_validation, alt_validation, **kwargs)
                elif strategy == "approximate":
                    # Approximate comparison: variants match within position window and allele similarity
                    self._compare_approx(ref_validation, alt_validation, **kwargs)
                elif strategy == "summary" or strategy == "statistics" or strategy == "statistical":
                    # Summary statistics comparison: overlap ratio, precision, recall
                    self._compare_summary(ref_validation, alt_validation, **kwargs)
                else:
                    logger.warning(f"Unknown strategy '{strategy}', fall back to summary")
                    self._compare_summary(ref_validation, alt_validation, **kwargs)
        except Exception as e:
            logger.error(f"VCF/BCF comparison failed: {e}")
            self.result.error = f"VCF/BCF comparison failed: {e}"
        return self._finalize_output(ref_validation, alt_validation, self.result)
    
    # ============================================================================
    # File I/O and Loading Methods
    # ============================================================================
    
    def _load_file(self, file_path: str) -> pysam.VariantFile:
        """
        Load file with auto-detection of format using detect_file_signature.
        Supports both uncompressed (.vcf) and compressed (.vcf.gz) VCF files and BCF files.
        pysam.VariantFile automatically handles compression when using mode "r".
        """
        # Detect format from file signature (handles both .vcf and .vcf.gz)
        signature = detect_file_signature(file_path, fallback="extension")
        detected_format = signature.format
        try:
            # pysam.VariantFile with mode "r" automatically handles compressed files (.vcf.gz)
            return pysam.VariantFile(file_path, self.format_map[detected_format]['mode'])
        except Exception:
            raise ValueError(f"Could not open VCF/BCF file: {file_path} with format: {detected_format}")
    
    # ============================================================================
    # Statistics and Analysis Methods
    # ============================================================================
    
    def _get_vcf_stats(self, file_path: str) -> Dict[str, Any]:
        """Get VCF/BCF statistics using bcftools stats.
        Parses bcftools stats output into multiple dataframes and summary statistics.
        
        Returns dictionary containing:
        - Summary statistics (num_variants, num_samples, etc.)
        - DataFrames for different statistic sections (summary_df, tstv_df, af_df, etc.)
        """
        try:
            # Run bcftools stats
            result = subprocess.run(
                ["bcftools", "stats", file_path],
                capture_output=True,
                text=True,
                check=True,
                timeout=300  # 5 minute timeout
            )
            
            stats_dict = self._parse_bcftools_stats(result.stdout)
            return stats_dict
        except subprocess.CalledProcessError as e:
            logger.error(f"bcftools stats failed: {e.stderr}")
            raise ValueError(f"bcftools stats failed: {e.stderr}")
        except subprocess.TimeoutExpired:
            logger.error(f"bcftools stats timed out for {file_path}")
            raise ValueError(f"bcftools stats timed out for {file_path}")
    
    def _parse_bcftools_stats(self, output: str) -> Dict[str, Any]:
        """Parse bcftools stats output into dataframes and summary statistics.
        
        Strategy: Find section boundaries by detecting first/last appearance of section keywords,
        split output into chunks, then parse each chunk into a DataFrame (ignoring comment lines).
        
        bcftools stats output format:
        - Comment lines: start with '#' (section headers, descriptions, etc.)
        - Data lines: tab-separated format: SECTION_CODE\tID\tFIELD1\tFIELD2\t...
          Example: 'SN\t0\tnumber of samples:\t1'
          Example: 'TSTV\t0\t0\t0\t0.00\t0\t0\t0.00'
        
        Returns dictionary with:
        - Summary statistics (num_variants, num_samples, etc.)
        - DataFrames: summary_df, tstv_df, af_df, qual_df, idd_df, dp_df, psc_df, psi_df, st_df
        """
        # Section schemas define the structure of each section's data lines
        # All sections share 'section_code' and 'id' as the first two columns
        section_schemas = {
            'SN': {
                'fields': ['section_code', 'id', 'key', 'value'],  # SN\tID\tkey\tvalue
                'types': [str, str, str, str]
            },
            'TSTV': {
                'fields': ['section_code', 'id', 'ts', 'tv', 'ts/tv', 'ts_1stalt', 'tv_1stalt', 'ts/tv_1stalt'],
                'types': [str, str, int, int, float, int, int, float]
            },
            'AF': {
                'fields': ['section_code', 'id', 'allele_frequency', 'number_of_SNPs', 'number_of_transitions', 
                          'number_of_transversions', 'number_of_indels'],
                'types': [str, str, float, int, int, int, int]
            },
            'QUAL': {
                'fields': ['section_code', 'id', 'quality', 'number_of_SNPs', 'number_of_transitions_1stalt',
                          'number_of_transversions_1stalt', 'number_of_indels'],
                'types': [str, str, float, int, int, int, int]
            },
            'IDD': {
                'fields': ['section_code', 'id', 'length', 'number_of_sites', 'number_of_genotypes', 'mean_VAF'],
                'types': [str, str, int, int, int, float]
            },
            'DP': {
                'fields': ['section_code', 'id', 'bin', 'number_of_genotypes', 'fraction_of_genotypes', 
                          'number_of_sites', 'fraction_of_sites'],
                'types': [str, str, str, int, float, int, float]
            },
            'PSC': {
                'fields': ['section_code', 'id', 'sample', 'nRefHom', 'nNonRefHom', 'nHets', 'nTransitions',
                          'nTransversions', 'nIndels', 'average_depth', 'nSingletons',
                          'nHapRef', 'nHapAlt', 'nMissing'],
                'types': [str, str, str, int, int, int, int, int, int, float, int, int, int, int]
            },
            'PSI': {
                'fields': ['section_code', 'id', 'sample', 'nIns', 'nDel', 'InsSum', 'DelSum'],
                'types': [str, str, str, int, int, int, int]
            },
            'ST': {
                'fields': ['section_code', 'id', 'type', 'count'],
                'types': [str, str, str, int]
            }
        }
        
        lines = output.strip().split('\n')
        result = {}
        
        # Find section boundaries: detect first and last line index for each section
        # A line belongs to a section if it starts with "# SECTION_CODE" or "SECTION_CODE"
        section_boundaries = {}
        for section_code in section_schemas.keys():
            first_idx = None
            last_idx = None
            
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                # Check if line starts with "# SECTION_CODE" or "SECTION_CODE"
                if line_stripped.startswith(f"# {section_code}") or line_stripped.startswith(section_code):
                    if first_idx is None:
                        first_idx = i
                    last_idx = i
            
            if first_idx is not None:
                section_boundaries[section_code] = (first_idx, last_idx)
        
        # Parse each section chunk using pandas
        for section_code, (first_idx, last_idx) in section_boundaries.items():
            schema = section_schemas[section_code]
            field_names = schema['fields']
            field_types = schema['types']
            
            # Extract chunk: lines from first to last appearance (inclusive)
            chunk_lines = lines[first_idx:last_idx + 1]
            chunk_text = '\n'.join(chunk_lines)
            
            # Use pandas to read the chunk, skipping comment lines
            # Format: SECTION_CODE\tID\tFIELD1\tFIELD2\t...
            # field_names already includes 'section_code' and 'id' as the first two columns
            # Determine DataFrame name
            df_name = 'summary_df' if section_code == 'SN' else f"{section_code.lower()}_df"
            
            try:
                # Create dtype dictionary mapping column names to types (skip str columns as pandas handles them)
                dtype_dict = {col: dtype for col, dtype in zip(field_names, field_types) if dtype != str}
                
                # Read as CSV with tab separator, skip comment lines, assign column names and types
                df = pd.read_csv(StringIO(chunk_text), sep='\t', comment='#', header=None, 
                                 skipinitialspace=True, on_bad_lines='skip', names=field_names, dtype=dtype_dict)
                
                # Store DataFrame (empty DataFrame is fine)
                result[df_name] = df
            except Exception:
                # Fallback to empty DataFrame if parsing fails
                result[df_name] = pd.DataFrame()
        
        # Extract summary statistics from summary_df if it exists
        if 'summary_df' in result and not result['summary_df'].empty:
            summary_df = result['summary_df']
            # Create lookup dictionary from DataFrame
            value_dict = dict(zip(summary_df['key'], summary_df['value']))
        else:
            value_dict = {}
        
        # Direct assignment from lookup (defaults to 0 if key not found)
        result['num_variants'] = int(value_dict.get('number of records:', 0))
        result['num_samples'] = int(value_dict.get('number of samples:', 0))
        result['num_snps'] = int(value_dict.get('number of SNPs:', 0))
        result['num_indels'] = int(value_dict.get('number of indels:', 0))
        result['num_multiallelic'] = int(value_dict.get('number of multiallelic sites:', 0))
        
        return result
    
    # ============================================================================
    # File Status Checkers
    # ============================================================================
    
    def _is_index_exists(self, file_path: str, index_path: str = None) -> bool:
        """Check if VCF/BCF file has an index file (.tbi or .csi).
        
        Args:
            file_path: Path to VCF/BCF file
            
        Returns:
            True if index file exists, False otherwise
        """
        if not index_path:
            if not file_path:
                raise ValueError("VCF/BCF file path is required if index path is not provided")
            signature = detect_file_signature(file_path, fallback="extension")
            index_path = f"{file_path}.{self.format_map[signature.format]['index_ext']}"
        
        return self._is_file_exists(index_path)
    
    def _is_sorted(self, file_path: str) -> bool:
        """Check if VCF/BCF file is sorted by coordinate (contig, position).
        Uses bcftools index to detect sorting - indexing will fail if file is not sorted.
        Returns True if sorted, False otherwise.
        """
        # Check if index already exists - if so, file was previously indexed (and thus sorted)
        if self._is_index_exists(file_path):
            return True
        
        try:
            signature = detect_file_signature(file_path, fallback="extension")
            index_ext = f".{self.format_map[signature.format]['index_ext']}"
            # Create a temporary file to hold the index (bcftools will overwrite it if it exists)
            fd, temp_index_path = tempfile.mkstemp(suffix=index_ext, prefix='vcf_index_')
            os.close(fd)
            
            temp_index_path = self._create_vcf_index(file_path, temp_index_path)
            # If indexing succeeds, file is sorted - clean up the index we created
            self._cleanup_file(temp_index_path)
            return True
        except RuntimeError as e:
            # Check error message for "not sorted" or "out of order"
            error_msg = str(e).lower()
            if "not sorted" in error_msg or "out of order" in error_msg:
                return False
            # Other errors - log and assume not sorted
            logger.warning(f"bcftools index failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to check if VCF/BCF file is sorted: {e}")
            return False
    
    # ============================================================================
    # File Creation Methods
    # ============================================================================
    
    def _create_sorted_vcf(self, file_path: str) -> str:
        """Create a sorted VCF/BCF file using bcftools sort.
        Returns path to temporary file (caller must clean up after use).
        """
        # Detect file format to determine correct output extension
        signature = detect_file_signature(file_path, fallback="extension")
        output_ext = f".{self.format_map[signature.format]['file_ext']}"
        
        # Create temporary file for sorted output with correct extension
        fd, temp_vcf_path = tempfile.mkstemp(suffix=output_ext, prefix='sorted_vcf_')
        os.close(fd)
        
        try:
            # Run bcftools sort
            result = subprocess.run(
                ["bcftools", "sort", "-o", temp_vcf_path, file_path],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout for sorting
                check=False
            )
            
            if result.returncode != 0:
                self._cleanup_file(temp_vcf_path)
                raise RuntimeError(f"bcftools sort failed: {result.stderr}")
            
            return temp_vcf_path
        except Exception as e:
            self._cleanup_file(temp_vcf_path)
            raise
    
    def _create_vcf_index(self, file_path: str, index_path: str = None) -> str:
        """Create a Tabix index file (.tbi for VCF.gz or .csi for BCF) using bcftools index.
        
        Args:
            file_path: Path to VCF/BCF file to index
            index_path: Path where the index file should be created (if None, creates next to file_path)
            
        Returns:
            Path to created index file
        """
        # Determine index path if not provided
        signature = detect_file_signature(file_path, fallback="extension")
        index_path = index_path or f"{file_path}.{self.format_map[signature.format]['index_ext']}"
        # Use -c for BCF (CSI index) or -t for VCF (Tabix index)
        index_flag = "-c" if signature.format == "bcf" else "-t"
        
        try:
            result = subprocess.run(
                ["bcftools", "index", index_flag, "-o", index_path, file_path],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for indexing
                check=False
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"bcftools index failed: {result.stderr}")
            
            if not os.path.exists(index_path):
                raise RuntimeError(f"Index file was not created: {index_path}")
            
            return index_path
        except Exception as e:
            # Clean up index file if it was created but method failed
            self._cleanup_file(index_path)
            raise
    
    # ============================================================================
    # Variant Parsing Methods
    # ============================================================================
    
    def _get_vcf_variants(self, file_path: str) -> Dict[Tuple[str, int, str, str], Dict[str, Any]]:
        """Get VCF/BCF variants from file.
        Returns dictionary mapping (chrom, pos, ref, alt) to variant info dictionary
        """
        variant_file = self._load_file(file_path)
        variants = {}  # key: (chrom, pos, ref, alt), value: variant info
        
        try:
            for record in variant_file:
                chrom = record.chrom
                pos = record.pos
                ref = record.ref
                
                # Handle multi-allelic variants (skip if no alternate alleles)
                if record.alts is None:
                    continue
                
                for alt in record.alts:
                    variant_key = (chrom, pos, ref, alt)
                    
                    # Store variant info with proper INFO field parsing
                    variant_info = {
                        "chrom": chrom,
                        "pos": pos,
                        "ref": ref,
                        "alt": alt,
                        "qual": record.qual if record.qual else None,
                        "filter": list(record.filter.keys()) if record.filter else [],
                        "info": dict(record.info) if record.info else {},
                        "id": record.id if record.id else "."
                    }
                    variants[variant_key] = variant_info
        finally:
            variant_file.close()
        
        return variants
    
    # ============================================================================
    # Private Comparison Implementation Methods
    # ============================================================================
    
    def _compare_exact(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Exact comparison: all variants must match exactly in the same order.
        
        Compares variants sequentially, preserving order. All variant records must be
        identical including position, alleles, quality, and filter status.
        
        Args:
            ref_validation: Validation result for reference file
            alt_validation: Validation result for candidate file
            **kwargs: Strategy-specific parameters (not used for exact comparison)
        """
        try:
            alt_variants = self._get_vcf_variants(alt_validation.file_path)
            ref_variants = self._get_vcf_variants(ref_validation.file_path)
            
            # Get ordered lists of keys (preserves insertion order from file)
            alt_keys_list = list(alt_variants.keys())
            ref_keys_list = list(ref_variants.keys())
            
            total_ref = len(ref_keys_list)
            total_alt = len(alt_keys_list)
            
            # Get sets for set operations
            alt_keys = set(alt_keys_list)
            ref_keys = set(ref_keys_list)
            
            # Early return if counts or keys don't match
            if total_ref != total_alt or ref_keys != alt_keys:
                self.result.similarity = 0.0
                self.result.details.update({
                    "total_ref_count": total_ref,
                    "total_alt_count": total_alt,
                    "matching_count": len(ref_keys & alt_keys),
                    "missing_count": len(ref_keys - alt_keys),
                    "extra_count": len(alt_keys - ref_keys),
                    "order_match": False,
                })
                return
            
            # Check order and details match
            order_match = True
            details_match = True
            for ref_key, alt_key in zip(ref_keys_list, alt_keys_list):
                if ref_key != alt_key:
                    order_match = False
                    break
                if details_match:
                    ref_var, alt_var = ref_variants[ref_key], alt_variants[alt_key]
                    if (ref_var["chrom"] != alt_var["chrom"] or
                        ref_var["pos"] != alt_var["pos"] or
                        ref_var["ref"] != alt_var["ref"] or
                        ref_var["alt"] != alt_var["alt"] or
                        ref_var["qual"] != alt_var["qual"] or
                        ref_var["filter"] != alt_var["filter"]):
                        details_match = False
            
            # Exact match requires both order and details to match
            self.result.similarity = 1.0 if (order_match and details_match) else 0.0
            self.result.details.update({
                "total_ref_count": total_ref,
                "total_alt_count": total_alt,
                "matching_count": total_ref,
                "missing_count": 0,
                "extra_count": 0,
                "order_match": order_match,
            })
        except Exception as e:
            logger.error(f"Exact comparison failed: {e}")
            self.result.error = f"Exact comparison failed: {e}"
    
    def _compare_approx(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Approximate comparison: variant key matching with F1 score.
        
        Compares variant keys (chrom, pos, ref, alt) and computes F1 score based on overlap.
        Useful for comparing variant calls from different tools or pipelines.
        
        Args:
            ref_validation: Validation result for reference file
            alt_validation: Validation result for candidate file
            **kwargs: Strategy-specific parameters (not used for approximate comparison)
        """
        try:
            alt_variants = self._get_vcf_variants(alt_validation.file_path)
            ref_variants = self._get_vcf_variants(ref_validation.file_path)
            
            alt_keys = set(alt_variants.keys())
            ref_keys = set(ref_variants.keys())
            
            common_variants = ref_keys & alt_keys
            total_ref = len(ref_keys)
            total_alt = len(alt_keys)
            overlap_count = len(common_variants)
            
            if total_ref == 0 and total_alt == 0:
                self.result.similarity = 1.0
                self.result.details.update({
                    "total_ref_count": 0,
                    "total_alt_count": 0,
                    "overlap_count": 0,
                    "missing_count": 0,
                    "extra_count": 0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1_score": 1.0
                })
                return
            
            jaccard = jaccard_similarity(total_ref, total_alt, overlap_count)
            precision, recall, f1_score = precision_recall_f1(total_ref, total_alt, overlap_count)
            self.result.similarity = jaccard
            self.result.details.update({
                "total_ref_count": total_ref,
                "total_alt_count": total_alt,
                "overlap_count": overlap_count,
                "missing_variants": len(ref_keys - alt_keys),
                "extra_variants": len(alt_keys - ref_keys),
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            })
        except Exception as e:
            logger.error(f"Approximate comparison failed: {e}")
            self.result.error = f"Approximate comparison failed: {e}"
    
    def _compare_summary(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Summary statistics comparison: compare VCF statistics using bcftools stats.
        
        Compares summary statistics and dataframes from bcftools stats output.
        Calculates similarity using RAE (Relative Absolute Error) for numeric comparisons.
        
        Args:
            ref_validation: Validation result for reference file
            alt_validation: Validation result for candidate file
            **kwargs: Strategy-specific parameters (not used for summary statistics comparison)
        """
        try:
            # Get statistics from both files
            ref_stats = self._get_vcf_stats(ref_validation.file_path)
            alt_stats = self._get_vcf_stats(alt_validation.file_path)
            
            # Compare most relevant dataframes for VCF comparison:
            # - tstv_df: Transition/transversion ratios (critical quality metric)
            # - af_df: Allele frequency distributions (important for variant distribution)
            # - qual_df: Quality score distributions (important for variant calling quality)
            # - summary_df: Additional summary metrics beyond the basic counts
            # Note: Excluding psc_df/psi_df (per-sample, too granular), st_df (redundant with TSTV),
            #       idd_df/dp_df (less critical for overall comparison)
            dataframe_names = ['tstv_df', 'af_df', 'qual_df', 'summary_df']
            similarities_by_df = {}
            for df_name in dataframe_names:
                ref_df = ref_stats.get(df_name, pd.DataFrame())
                alt_df = alt_stats.get(df_name, pd.DataFrame())
                
                if ref_df.empty and alt_df.empty:
                    # Both empty - perfect match
                    similarities_by_df[df_name] = 1.0
                    continue
                elif ref_df.empty or alt_df.empty:
                    # One empty, one not - no match
                    similarities_by_df[df_name] = 0.0
                    continue
                
                # Compare numeric columns using vectorized operations
                # For summary_df, set 'key' as index first to align rows by key (schema has 'key' in _parse_bcftools_stats)
                if df_name == 'summary_df' and 'key' in ref_df.columns:
                    ref_df = ref_df.set_index('key')
                    alt_df = alt_df.set_index('key')
                
                # Compare numeric columns. Columns are from the same parser so identical.
                # Rows: summary_df has fixed keys (same rows); tstv_df usually one row (id=0); af_df/qual_df are data-dependent (bins can differ). Align by index when needed.
                numeric_similarities = []
                for col in ref_df.columns:
                    if pd.api.types.is_numeric_dtype(ref_df[col]) and pd.api.types.is_numeric_dtype(alt_df[col]):
                        if ref_df.index.equals(alt_df.index):
                            ref_vals = ref_df[col].values
                            alt_vals = alt_df[col].values
                        else:
                            # Align by index (union), fill missing with 0
                            ref_aligned, alt_aligned = ref_df[col].align(alt_df[col], join='outer', fill_value=0)
                            ref_vals = ref_aligned.values
                            alt_vals = alt_aligned.values
                        
                        # Compute similarity using rae_similarity
                        sim = rae_similarity(ref_vals, alt_vals)
                        
                        numeric_similarities.extend(sim.tolist())
                
                if numeric_similarities:
                    similarities_by_df[df_name] = float(np.mean(numeric_similarities))
                else:
                    similarities_by_df[df_name] = 0.0
            
            # Main similarity from summary_df (fixed keys, comparable across files); others in details
            summary_similarity = similarities_by_df.get('summary_df', 0.0)
            self.result.similarity = float(summary_similarity)
            self.result.details.update({
                "ref_stats": {k: v for k, v in ref_stats.items() if k in ['num_variants', 'num_samples', 'num_snps', 'num_indels', 'num_multiallelic']},
                "alt_stats": {k: v for k, v in alt_stats.items() if k in ['num_variants', 'num_samples', 'num_snps', 'num_indels', 'num_multiallelic']},
                "tstv_similarity": similarities_by_df.get('tstv_df'),
                "af_similarity": similarities_by_df.get('af_df'),
                "qual_similarity": similarities_by_df.get('qual_df'),
                "summary_similarity": summary_similarity,
            })
        except Exception as e:
            logger.error(f"Summary statistics comparison failed: {e}")
            self.result.error = f"Summary statistics comparison failed: {e}"
    
