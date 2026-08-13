"""CSV/TSV format handler.

Strategy:
- exact: Exact row-by-row and cell-by-cell matching
- approximate: Value similarity with tolerance for numeric differences
- summary: Aggregate statistical comparison (shape, missing rates, distribution overlap via Hellinger distance)
- semantic: LLM-based comparison that loads all data into the LLM for direct analysis

Required cmdline tools:
- (None)
"""

import re
import csv
import json
import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from pandas import Series, DataFrame
from pydantic import BaseModel, Field

from utils.agents import create_llm_agent, invoke_structured_agent
from utils.format import detect_file_signature, FileSignature
from utils.metrics import jaccard_similarity, precision_recall_f1, pearson_correlation, spearman_correlation, hellinger_distance, rae_similarity

from tools.base import FormatHandler
from tools.results import ValidationResult, ComparisonResult, EquivalenceResult
from tools.table_models import RowIDColumns, ColumnMappingDict, ColumnMapping, AlignPlan, ComparisonPlan, TableMeta, ColumnSimilarity, IndexMapping

logger = logging.getLogger(__name__)


class TableHandler(FormatHandler):
    """Handler for tabular text files with atomic comparison operations."""

    def __init__(self):
        super().__init__("csv")
        self.supported_formats = ["csv", "tsv", "table", "xlsx", "gct"]
        self.supported_strategies = ["exact", "approximate", "summary", "semantic"]
        
        self.similarity_metrics = ["pearson_correlation", "value_similarity", "missing_pattern_similarity"]

        # cache and preview settings
        self._signature_cache: dict[str, FileSignature] = {}
        self._tabular_meta_cache: dict[str, TableMeta] = {}
        self.preview_max_rows = 10
        self.preview_max_chars = 10_000
        self.model = "gpt-5.4"

    # ============================================================================
    # Helper Methods
    # ============================================================================
    
    def _get_signature(self, file_path: str) -> FileSignature:
        """Get file signature, using cache to avoid repeated detection."""
        if file_path not in self._signature_cache:
            self._signature_cache[file_path] = detect_file_signature(file_path, fallback="llm")
        return self._signature_cache[file_path]

    # ============================================================================
    # Public API Methods
    # ============================================================================
    
    def validate(self, file_path: str, **kwargs) -> ValidationResult:
        """Validate CSV/TSV file format.
        
        Args:
            file_path: Path to CSV/TSV file
            **kwargs: optional parameters (e.g., min_size)
            - min_size: Minimum size of the file in bytes, default is 0
        """
        # check existence and non-emptiness (by file size)
        is_exists = self._is_file_exists(file_path)
        is_nonempty = self._is_file_nonempty(file_path, min_size=kwargs.get("min_size", 0))
        
        # check file format
        signature = self._get_signature(file_path)
        is_format_ok = signature.format in self.supported_formats

        # initialize validation result
        result = ValidationResult(exists=is_exists, non_empty=is_nonempty, format_ok=is_format_ok, parseable=False,
                                  file_path=file_path, signature=signature, details={}, error=None)

        if not is_exists:
            logger.error(f"CSV/TSV file not found: {file_path}")
            result.error = "File not found"
            return result
        
        if not is_nonempty:
            logger.warning(f"CSV/TSV file is empty: {file_path}")
            result.error = "File is empty"
            return result
        
        if not is_format_ok:
            logger.error(f"Not a CSV/TSV file: detected={signature.format}")
            result.error = f"Not a CSV/TSV file: detected={signature.format}"
            return result
        
        # collect statistics using pandas
        try:
            meta_dict = self._get_table_meta(file_path).model_dump()
            
            df = self._load_file(file_path, meta_dict=meta_dict)
            result.parseable = True
            result.non_empty = len(df) > 0
            
            if not result.non_empty:
                logger.warning(f"CSV/TSV file contains no data rows: {file_path}")
                result.details.update({"rows": 0, "columns": len(df.columns) if hasattr(df, 'columns') else 0, "meta": meta_dict})
                result.error = "CSV/TSV file contains no data rows"
                return result
            
            result.details.update({"rows": len(df), "columns": len(df.columns), "meta": meta_dict})
            logger.info(f"CSV/TSV validation passed: {signature.extension}, {df.shape}")
        except Exception as e:
            logger.error(f"CSV/TSV validation failed: {e}")
            result.error = f"CSV/TSV validation failed: {e}"
        return result

    def compare(self, reference_path: str, candidate_path: str, strategy: str = "approximate", **kwargs) -> ComparisonResult:
        """Compare two CSV/TSV files.
        Supported strategies:
        - exact: Cell-by-cell equality across all columns (numeric with tolerance, non-numeric strict)
        - approximate: Per-column strategies with structural metrics (default, uses column_configs)
        - summary: Summary statistics comparison
        - semantic: LLM judges aligned data for holistic equivalence
        
        Args:
            reference_path: Path to reference CSV/TSV file
            candidate_path: Path to candidate CSV/TSV file
            strategy: Comparison strategy (exact, approximate, summary, semantic)
            **kwargs: Strategy-specific parameters
            - question: Task description for LLM planning (default: "Compare two tabular data files for equivalence")
            - fuzzy_id_match: For 'approximate' strategy, when True, row IDs that don't match exactly
              (after case normalization) are additionally paired via LLM semantic matching, so e.g.
              "medium_vs_low" and "Medium vs Low" can still align even without lexical similarity.
              Only applies to single-column IDs. Default True — only the IDs left unmatched after
              exact/case-insensitive comparison are sent to the LLM, so this is a no-op (no extra
              call) when every ID already matches. Set False to require exact/case-insensitive ID
              matches only.
        """
        self.result = ComparisonResult(reference_path=reference_path, candidate_path=candidate_path,
                                       strategy=strategy, similarity=np.nan, details={}, error=None)
        # Validate reference and candidate files
        ref_validation = self.validate(reference_path, **kwargs)
        alt_validation = self.validate(candidate_path, **kwargs)

        try:
            if ref_validation.is_valid and alt_validation.is_valid:
                # Load dataframes (reuse metadata from validation if available)
                ref_meta_dict = ref_validation.details.get("meta")
                alt_meta_dict = alt_validation.details.get("meta")

                ref_df = self._load_file(reference_path, meta_dict=ref_meta_dict)
                alt_df = self._load_file(candidate_path, meta_dict=alt_meta_dict)

                task_desc = kwargs.get("question", "Compare two tabular data files for equivalence")
                eval_guideline = kwargs.get("eval_guideline", None)
                fuzzy_id_match = kwargs.get("fuzzy_id_match", True)

                if strategy == "exact":
                    self._compare_exact(ref_df, alt_df)
                elif strategy == "approximate":
                    self._compare_approx(ref_df, alt_df, task_desc, eval_guideline=eval_guideline,
                                         fuzzy_id_match=fuzzy_id_match)
                elif strategy == "summary":
                    self._compare_summary(ref_df, alt_df)
                elif strategy == "semantic":
                    self._compare_semantic(ref_df, alt_df, task_desc, eval_guideline=eval_guideline)
                else:
                    logger.warning(f"Unknown strategy '{strategy}', falling back to approximate")
                    self._compare_approx(ref_df, alt_df, task_desc, eval_guideline=eval_guideline,
                                         fuzzy_id_match=fuzzy_id_match)
        except Exception as e:
            logger.error(f"CSV/TSV comparison failed: {e}")
            self.result.error = f"CSV/TSV comparison failed: {e}"
        return self._finalize_output(ref_validation, alt_validation, self.result)
    
    # ============================================================================
    # Atomic Operations
    # ============================================================================

    def _sort_by_column(self, df: DataFrame, column: str | List[str], ascending: bool | List[bool] = True) -> DataFrame:
        """Sort dataframe by specified column(s).
        
        Args:
            df: Dataframe to sort.
            column: Column name (str) or list of column names (List[str]) to sort by.
            ascending: Sort order. If column is a list, can be a single bool (applied to all columns)
                      or a list of bools (one per column). Default: True.
        
        Returns:
            Sorted dataframe with reset index.
        """
        columns = [column] if isinstance(column, str) else column
        ascending = [ascending] if isinstance(ascending, bool) else ascending

        n_cols, n_asc = len(columns), len(ascending)
        if n_cols != n_asc:
            logger.warning(f"Length mismatch: columns ({n_cols}) vs ascending ({n_asc}), adjusting to match columns")
            ascending = ascending[:n_cols] + [ascending[-1]] * max(0, n_cols - n_asc)
        
        # Zip columns and ascending, then filter to only existing columns
        zipped = list(zip(columns, ascending))
        valid_pairs = [(c, a) for c, a in zipped if c in df.columns]
        
        if missing := [c for c, _ in zipped if c not in df.columns]:
            logger.warning(f"Column(s) {missing} not found in dataframe, skipping")
        
        if not valid_pairs:
            logger.warning("No valid columns to sort by")
            return df
        
        valid_columns, valid_ascending = zip(*valid_pairs)
        
        return df.sort_values(by=list(valid_columns), ascending=list(valid_ascending)).reset_index(drop=True)
    
    def _get_common_subtable(self, ref_df: DataFrame, alt_df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """Align two dataframes to common columns and index.

        Returns ref and alt subsets on the intersection of columns and index.

        Args:
            ref_df: Reference dataframe.
            alt_df: Candidate dataframe.

        Returns:
            (ref_common, alt_common). Both are restricted to common columns and common index.
        """
        common_cols = sorted(set(ref_df.columns) & set(alt_df.columns))
        common_idx = ref_df.index.intersection(alt_df.index)
        ref_common = ref_df.loc[common_idx, common_cols]
        alt_common = alt_df.loc[common_idx, common_cols]
        return ref_common, alt_common
    
    def _get_aligned_table(self, ref_df: DataFrame, alt_df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """Align two dataframes to union of columns and index.

        Returns ref and alt subsets on the union of columns and index.
        Missing values are filled with NaN where data doesn't exist.
        Common columns and indices are placed first (top and left).

        Args:
            ref_df: Reference dataframe.
            alt_df: Candidate dataframe.

        Returns:
            (ref_aligned, alt_aligned). Both are aligned to union of columns and union of index,
            with common columns/indices appearing first.
        """
        # Get common and union columns/indices
        common_cols = sorted(set(ref_df.columns) & set(alt_df.columns))
        union_cols = sorted(set(ref_df.columns) | set(alt_df.columns))
        
        common_idx = ref_df.index.intersection(alt_df.index)
        union_idx = ref_df.index.union(alt_df.index)
        
        # Order: common first, then union-only
        ordered_cols = common_cols + [c for c in union_cols if c not in common_cols]
        ordered_idx = pd.Index(list(common_idx) + list(union_idx.difference(common_idx)))
        
        ref_aligned = ref_df.reindex(index=ordered_idx, columns=ordered_cols)
        alt_aligned = alt_df.reindex(index=ordered_idx, columns=ordered_cols)
        return ref_aligned, alt_aligned
    
    def _get_dimension_overlap(self, ref_df: DataFrame, alt_df: DataFrame) -> Dict[str, float]:
        """Calculate row/column overlap metrics between dataframes."""
        ref_rows, ref_cols = set(ref_df.index), set(ref_df.columns)
        alt_rows, alt_cols = set(alt_df.index), set(alt_df.columns)
        
        row_intersection = len(ref_rows & alt_rows)
        col_intersection = len(ref_cols & alt_cols)
        row_jaccard = jaccard_similarity(len(ref_rows), len(alt_rows), row_intersection)
        col_jaccard = jaccard_similarity(len(ref_cols), len(alt_cols), col_intersection)

        row_precision, row_recall, row_f1 = precision_recall_f1(len(ref_rows), len(alt_rows), row_intersection)
        col_precision, col_recall, col_f1 = precision_recall_f1(len(ref_cols), len(alt_cols), col_intersection)
        return {"row_precision": row_precision, "row_recall": row_recall, "row_f1": row_f1, "row_jaccard": row_jaccard, 
                "col_precision": col_precision, "col_recall": col_recall, "col_f1": col_f1, "col_jaccard": col_jaccard, 
                "ref_rows": len(ref_rows), "alt_rows": len(alt_rows), "row_intersection": row_intersection, 
                "ref_cols": len(ref_cols), "alt_cols": len(alt_cols), "col_intersection": col_intersection}
    
    def _get_numeric_correlation(self, ref_df: DataFrame, alt_df: DataFrame) -> Dict[str, float]:
        """Get numeric correlation metrics between dataframes."""
        ref_numeric = ref_df.select_dtypes(include=[np.number])
        alt_numeric = alt_df.select_dtypes(include=[np.number])
        ref_aligned, alt_aligned = self._get_aligned_table(ref_numeric, alt_numeric)
        
        # Still need common columns for correlation (can't correlate columns that don't exist in both)
        common_cols = sorted(set(ref_numeric.columns) & set(alt_numeric.columns))
        if not common_cols:
            return {"correlation": None, "per_column_correlations": {}}
        
        correlations = {}
        for col in common_cols:
            corr = pearson_correlation(ref_aligned[col].values, alt_aligned[col].values, nan_policy="omit")
            correlations[col] = corr
        
        corr_values = list(correlations.values())
        avg_corr = np.nanmean(corr_values) if corr_values else None

        return {
            "correlation": avg_corr,
            "per_column_correlations": correlations
        }
    
    def _get_value_similarity(self, ref_df: DataFrame, alt_df: DataFrame, tolerance: float = 0.01) -> Dict[str, float]:
        """Get value similarity metrics for overlapping regions between dataframes."""
        # Filter to numeric columns first (like _get_numeric_correlation)
        ref_numeric = ref_df.select_dtypes(include=[np.number])
        alt_numeric = alt_df.select_dtypes(include=[np.number])
        if ref_numeric.empty or alt_numeric.empty:
            return {"value_similarity": 0.0, "cells_compared": 0, "matching_cells": 0}
        
        ref_numeric_common, alt_numeric_common = self._get_common_subtable(ref_numeric, alt_numeric)
        if ref_numeric_common.empty or len(ref_numeric_common.index) == 0:
            return {"value_similarity": 0.0, "cells_compared": 0, "matching_cells": 0}
        
        # Calculate similarity using rae_similarity
        sim_array = rae_similarity(ref_numeric_common.values, alt_numeric_common.values, clip=True)
        
        # Handle NaN values
        valid_mask = ~np.isnan(sim_array)
        if not valid_mask.any():
            return {"value_similarity": 0.0, "cells_compared": 0, "matching_cells": 0}
        
        sim_valid = sim_array[valid_mask]
        total_cells = valid_mask.sum()
        # Convert tolerance (RAE threshold) to similarity threshold
        # RAE <= tolerance means similarity >= (1 - tolerance)
        similarity_threshold = 1.0 - tolerance
        # Calculate matching_cells: count cells with similarity >= threshold
        matching_cells = int(np.sum(sim_valid >= similarity_threshold)) if total_cells > 0 else 0
        # Calculate similarity: average of similarity scores
        similarity = float(np.mean(sim_valid))        
        return {
            "value_similarity": similarity,
            "cells_compared": total_cells,
            "matching_cells": matching_cells
        }

    def _get_missing_similarity(self, ref_df: DataFrame, alt_df: DataFrame) -> Dict[str, float]:
        """Get missing value pattern similarity metrics."""
        ref_common, alt_common = self._get_common_subtable(ref_df, alt_df)
        
        if ref_common.empty or len(ref_common.index) == 0:
            return {"missing_pattern_similarity": 0.0}
        
        ref_missing = ref_common.isna()
        alt_missing = alt_common.isna()
        
        # Calculate how many cells have same missing status
        same_status = (ref_missing == alt_missing).sum().sum()
        total_cells = ref_missing.size
        
        ref_missing_rate = float(ref_missing.sum().sum() / total_cells) if total_cells > 0 else 0.0
        alt_missing_rate = float(alt_missing.sum().sum() / total_cells) if total_cells > 0 else 0.0
        similarity = same_status / total_cells if total_cells > 0 else 0.0
        
        return {
            "missing_pattern_similarity": similarity,
            "ref_missing_rate": ref_missing_rate,
            "alt_missing_rate": alt_missing_rate
        }

    def _check_col_order(self, ref_df: DataFrame, alt_df: DataFrame) -> Dict[str, Any]:
        """Check if dataframes have the same columns but in different order.
        Returns True if the columns are in different order, False if they are in the same order, and None if the columns are different.
        """
        # Check if same columns (order-independent)
        if set(ref_df.columns) != set(alt_df.columns):
            return {
                "is_col_order_diff": None,
                "reason": "different columns"
            }
        
        return {
            "is_col_order_diff": list(ref_df.columns) != list(alt_df.columns),
            "reason": None
        }
    
    def _check_row_order(self, ref_df: DataFrame, alt_df: DataFrame) -> Dict[str, Any]:
        """Check if dataframes have rows in different order.
        Returns True if the rows are in different order, False if they are in the same order, and None if the rows are different.
        """
        if set(ref_df.index) != set(alt_df.index):
            return {
                "is_row_order_diff": None,
                "reason": "different index"
            }
        
        return {
            "is_row_order_diff":  list(ref_df.index) != list(alt_df.index),
            "reason": None
        }
    
    def _detect_permutation(self, ref_df: DataFrame, alt_df: DataFrame) -> Dict[str, Any]:
        """Detect if dataframes are permutations of each other.
        
        A true permutation means: same shape, same data values, but different order.
        This method first verifies that data matches, then checks if order differs.
        
        Returns:
            Dict with:
            - is_permutation: bool - True if data matches and order differs
            - column_order_diff: bool - True if the columns are in different order
            - row_order_diff: bool - True if the rows are in different order
            - data_matches: bool - Whether the data values match (after sorting)
        """
        col_order = self._check_col_order(ref_df, alt_df)
        row_order = self._check_row_order(ref_df, alt_df)

        # Step 1: Check shape first
        if ref_df.shape != alt_df.shape:
            return {
                "is_permutation": False,
                "column_order_diff": col_order["is_col_order_diff"],
                "row_order_diff": row_order["is_row_order_diff"],
                "data_matches": False,
                "reason": "different shapes"
            }
        
        # Step 2: Check if columns match (order-independent)
        if set(ref_df.columns) != set(alt_df.columns):
            return {
                "is_permutation": False,
                "column_order_diff": col_order["is_col_order_diff"],
                "row_order_diff": row_order["is_row_order_diff"],
                "data_matches": False,
                "reason": "different columns"
            }
        
        # Step 3: Align columns and verify data matches
        alt_aligned = alt_df[ref_df.columns]
        # Sort both dataframes by all columns and compare to verify data matches
        ref_sorted = self._sort_by_column(ref_df, list(ref_df.columns))
        alt_sorted = self._sort_by_column(alt_aligned, list(ref_df.columns))
        data_matches = ref_sorted.equals(alt_sorted)
        
        # Step 4: Check order differences (only meaningful if data matches)
        # Permutation: data matches AND order differs
        is_permutation = data_matches and (col_order["is_col_order_diff"] or row_order["is_row_order_diff"])
        
        return {
            "is_permutation": is_permutation,
            "column_order_diff": col_order["is_col_order_diff"],
            "row_order_diff": row_order["is_row_order_diff"],
            "data_matches": data_matches
        }
    
    def _get_shape_diff(self, ref_df: DataFrame, alt_df: DataFrame) -> Dict[str, Any]:
        """Get shape comparison metrics."""
        return {
            "ref_shape": ref_df.shape,
            "alt_shape": alt_df.shape,
            "shape_match": ref_df.shape == alt_df.shape,
            "row_diff": ref_df.shape[0] - alt_df.shape[0],
            "col_diff": ref_df.shape[1] - alt_df.shape[1]
        }
    
    def _get_dtype_match(self, ref_df: DataFrame, alt_df: DataFrame) -> Dict[str, Any]:
        """Get data type match metrics for common columns.
        
        Compares dtypes of common columns to detect type mismatches (e.g., int64 vs float64).
        Useful for identifying cases where data is equivalent but stored with different types.
        """
        common_cols = list(set(ref_df.columns) & set(alt_df.columns))
        
        dtype_matches = {}
        for col in common_cols:
            ref_type = ref_df[col].dtype
            alt_type = alt_df[col].dtype
            dtype_matches[col] = str(ref_type) == str(alt_type)
        
        match_rate = sum(dtype_matches.values()) / len(dtype_matches) if dtype_matches else 0
        
        return {
            "dtype_match_rate": match_rate,
            "dtype_matches": dtype_matches,
            "common_columns": len(common_cols)
        }
    

    # ============================================================================
    # Dataframe Alignment Methods
    # ============================================================================
    
    def _map_columns(self, ref_df: DataFrame, alt_df: DataFrame) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """Map reference columns to candidate columns using LLM semantic matching.
        
        Returns:
            (mapping, metadata) where mapping is a dict from ref_col -> {
                'alt_column': alt_col (or None),
                'confidence': float,
                'reasoning': str
            },
            and metadata contains overall_confidence and overall_reasoning.
        """
        agent = create_llm_agent(
            model=self.model,
            system_prompt="You are a data alignment expert. Map reference columns to candidate columns based on semantic meaning and data content. For each reference column, provide the matching candidate column (or null), your confidence score, and reasoning. Also provide an overall confidence for the entire mapping.",
            response_format=ColumnMapping,
        )

        ref_cols = list(ref_df.columns)
        alt_cols = list(alt_df.columns)
        ref_preview = self._get_preview(ref_df, self.preview_max_rows)
        alt_preview = self._get_preview(alt_df, self.preview_max_rows)
        prompt = f"""Map these reference columns to candidate columns based on column names and data content.

For EACH reference column, identify:
1. The corresponding candidate column (if a good match exists), or null if no match
2. Your confidence (0.0-1.0) for this specific mapping
3. Your reasoning for why this mapping was chosen (or why no match exists)

Reference columns: {ref_cols}
Reference data (first {self.preview_max_rows} rows):
{ref_preview}

Candidate columns: {alt_cols}
Candidate data (first {self.preview_max_rows} rows):
{alt_preview}

Provide a mapping for EACH reference column. Include per-entry confidence and reasoning, and an overall confidence and reasoning for the entire mapping."""

        try:
            response = invoke_structured_agent(agent, prompt, ColumnMapping)
            # Convert ColumnMappingEntry objects to dicts for easier handling
            mapping_dict = {}
            for ref_col, entry in response.mapping.items():
                mapping_dict[ref_col] = {
                    "alt_column": entry.alt_column,
                    "confidence": entry.confidence,
                    "reasoning": entry.reasoning,
                }
            return mapping_dict, {
                "overall_confidence": response.overall_confidence,
                "overall_reasoning": response.overall_reasoning,
            }
        except Exception as e:
            logger.warning(f"LLM column mapping failed: {e}")
            return {}, {}
    
    def _detect_id_columns(self, ref_df: DataFrame, alt_df: DataFrame) -> Tuple[List[str], Dict[str, Any]]:
        """Detect which columns should be used as row identifiers using LLM.
        
        Assumes columns are already aligned, so we only need to identify ID columns once.
        
        Returns:
            (id_cols, metadata) where metadata contains reasoning and confidence.
        """
        agent = create_llm_agent(
            model=self.model,
            system_prompt="You are a data analysis expert. Identify the MINIMAL set of column(s) that are both NECESSARY and SUFFICIENT to uniquely identify rows. The set must be sufficient (uniquely identifies each row) and necessary (no column can be removed while maintaining uniqueness). Prefer single columns over composite keys when possible.",
            response_format=RowIDColumns,
        )
        
        # Get common columns (after alignment, these should be the shared columns)
        common_cols = list(set(ref_df.columns) & set(alt_df.columns))
        
        ref_preview = self._get_preview(ref_df, self.preview_max_rows)
        alt_preview = self._get_preview(alt_df, self.preview_max_rows)
        
        prompt = f"""Identify which column(s) should be used as unique row identifiers for aligning these two dataframes.

Common columns (shared between both dataframes): {common_cols}
Reference data (first {self.preview_max_rows} rows):
{ref_preview}

Candidate data (first {self.preview_max_rows} rows):
{alt_preview}

Identify the MINIMAL set of column(s) from the common columns that are both NECESSARY and SUFFICIENT to uniquely identify each row.

CRITICAL RULES:
1. The columns MUST exist in BOTH dataframes (only select from the common columns listed above)
2. SUFFICIENT: The columns must uniquely identify each row in BOTH dataframes (no two rows have the same combination of values)
3. NECESSARY: The set must be minimal - no column can be removed while still maintaining uniqueness
4. Prefer the SHORTEST list possible - if a single column is sufficient, use only that column
5. Only use composite keys (multiple columns) when NO single column is sufficient
6. Examples:
   - If 'gene_id' is unique in both dataframes → use ['gene_id'] (NOT ['gene_id', 'other_unique_col'] even if both are unique)
   - If 'chr_id:start-pos' exists in both and is unique → use ['chr_id:start-pos'] (NOT ['chr_id', 'start', 'end'])
   - Only use ['chr_id', 'start', 'end'] if no single column is sufficient AND all three are necessary AND all exist in both dataframes
7. Return empty list if no suitable identifier columns exist in both dataframes (e.g., rows are just numbered sequentially)

Since columns are already aligned, identify the minimal ID columns from the common columns."""

        try:
            response = invoke_structured_agent(agent, prompt, RowIDColumns)
            id_cols = response.id_columns if response.id_columns else []
            
            # Verify columns exist in both dataframes
            id_cols = [col for col in id_cols if col in common_cols]
            
            return id_cols, {
                "reasoning": response.reasoning,
                "confidence": response.confidence,
            }
        except Exception as e:
            logger.warning(f"LLM ID column detection failed: {e}")
            return [], {}
    
    def _align_columns(self, ref_df: DataFrame, alt_df: DataFrame, column_mapping: Optional[ColumnMappingDict] = None, use_llm: bool = True, ignore_case: bool = True) -> Tuple[DataFrame, DataFrame, Dict[str, Any]]:
        """Align candidate columns to reference columns.

        Resolution order:
        1. Use provided column_mapping (from plan) if available - no LLM call needed
        2. If use_llm: try LLM semantic matching via _map_columns as fallback
        3. Normalized name matching: match by (lowercased) names—full match or intersection.
           ignore_case controls whether names are lowercased before matching.

        Args:
            ref_df: Reference dataframe
            alt_df: Candidate dataframe
            column_mapping: Optional pre-determined column mapping from plan (ref_col -> alt_col).
                           None value for a ref column means no match found.
                           If the entire field is None, will fall back to LLM or normalized matching.
            use_llm: Whether to try LLM for semantic column mapping if column_mapping is None (default True)
            ignore_case: When True, match using lowercased names (default True)

        Returns:
            (ref_aligned_df, alt_aligned_df, metadata) where both dataframes have:
            - Common/aligned columns on the left (in ref_df order)
            - Unaligned columns on the right
        """
        ref_cols = list(ref_df.columns)
        alt_cols = list(alt_df.columns)
        normalize = (lambda x: str(x).lower()) if ignore_case else (lambda x: str(x))
        ref_normalized = {normalize(c): c for c in ref_cols}
        alt_normalized = {normalize(c): c for c in alt_cols}

        metadata = {
            "alignment_method": "none",
            "column_mapping": {},
            "unmapped_ref": ref_cols,
            "unmapped_alt": alt_cols,
        }
        ref_aligned_df = ref_df
        alt_aligned_df = alt_df

        # Path 1: Use provided column_mapping from plan (no LLM call needed)
        if column_mapping is not None:
            # Filter to only valid mappings (alt_column exists in alt_df)
            valid_mapping = {
                ref_col: alt_col
                for ref_col, alt_col in column_mapping.items()
                if ref_col in ref_df.columns and alt_col is not None and alt_col in alt_df.columns
            }
            
            if valid_mapping:
                reverse_mapping = {alt_col: ref_col for ref_col, alt_col in valid_mapping.items()}
                ref_mapped_cols = list(valid_mapping.keys())
                ref_unmapped_cols = [c for c in ref_cols if c not in valid_mapping]
                alt_mapped_cols = list(valid_mapping.values())
                alt_unmapped_cols = [c for c in alt_cols if c not in valid_mapping.values()]
                
                # Order: mapped columns first (in ref order), then unmapped
                ref_cols_order = ref_mapped_cols + ref_unmapped_cols
                alt_cols_order = alt_mapped_cols + alt_unmapped_cols
                
                ref_aligned_df = ref_df[ref_cols_order]
                alt_aligned_df = alt_df[alt_cols_order].rename(columns=reverse_mapping)
                # Ensure alt_aligned_df has same column order as ref_aligned_df for mapped columns
                alt_aligned_df = alt_aligned_df[ref_mapped_cols + alt_unmapped_cols]
                
                metadata.update({
                    "alignment_method": "plan_mapping",
                    "column_mapping": valid_mapping,
                    "unmapped_alt": alt_unmapped_cols,
                    "unmapped_ref": ref_unmapped_cols,
                })
                return ref_aligned_df, alt_aligned_df, metadata

        # Path 2: LLM semantic matching (fallback when column_mapping not provided)
        if use_llm:
            llm_mapping, llm_metadata = self._map_columns(ref_df, alt_df)
            if llm_mapping:
                valid_mapping = {ref_col: entry["alt_column"]
                                 for ref_col, entry in llm_mapping.items()
                                 if entry["alt_column"] and entry["alt_column"] in alt_df.columns}
                reverse_mapping = {alt_col: ref_col for ref_col, alt_col in valid_mapping.items()}
                ref_cols_order = list(valid_mapping.keys()) + [c for c in ref_cols if c not in valid_mapping]
                alt_cols_order = list(valid_mapping.values()) + [c for c in alt_cols if c not in valid_mapping.values()]
                ref_aligned_df = ref_df[ref_cols_order]
                alt_aligned_df = alt_df[alt_cols_order].rename(columns=reverse_mapping)
                metadata.update({
                    "alignment_method": "llm_semantic",
                    "column_mapping": valid_mapping,
                    "unmapped_alt": [c for c in alt_cols if c not in valid_mapping.values()],
                    "unmapped_ref": [c for c in ref_cols if c not in valid_mapping],
                    **llm_metadata,
                })
                return ref_aligned_df, alt_aligned_df, metadata

        # Path 3: Normalized name matching (full match or intersection)
        common_norm = set(ref_normalized.keys()) & set(alt_normalized.keys())
        if common_norm:
            ref_mapped_cols = [c for c in ref_cols if normalize(c) in common_norm]
            alt_mapped_cols = [alt_normalized[normalize(c)] for c in ref_mapped_cols]
            ref_unmapped_cols = [c for c in ref_cols if c not in ref_mapped_cols]
            alt_unmapped_cols = [c for c in alt_cols if c not in alt_mapped_cols]
            reverse_mapping = {alt: ref for ref, alt in zip(ref_mapped_cols, alt_mapped_cols)}
            ref_aligned_df = ref_df[ref_mapped_cols + ref_unmapped_cols]
            alt_aligned_df = alt_df[alt_mapped_cols + alt_unmapped_cols].rename(columns=reverse_mapping)
            alt_aligned_df = alt_aligned_df[ref_mapped_cols + alt_unmapped_cols]
            full_match = common_norm == set(ref_normalized.keys()) == set(alt_normalized.keys())
            metadata.update({
                "alignment_method": (
                    "case_insensitive" if ignore_case else "exact"
                ) if full_match else (
                    "case_insensitive_intersection" if ignore_case else "intersection"
                ),
                "column_mapping": {ref: alt for ref, alt in zip(ref_mapped_cols, alt_mapped_cols)},
                "unmapped_alt": alt_unmapped_cols,
                "unmapped_ref": ref_unmapped_cols,
            })
            return ref_aligned_df, alt_aligned_df, metadata

        return ref_aligned_df, alt_aligned_df, metadata
    
    def _fuzzy_match_ids(self, ref_ids: List[str], alt_ids: List[str]) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Use an LLM to semantically match candidate ID strings to reference ID strings.

        Only IDs that have no exact counterpart on the other side should be passed in (values
        shared by both sides are already handled by exact matching upstream). Unlike string
        similarity, this can recognize equivalences that aren't lexically close, e.g.
        "medium_vs_low" vs "Medium vs Low", or "KO" vs "knockout".

        Args:
            ref_ids: Reference ID values with no exact match in alt_ids (already deduplicated).
            alt_ids: Candidate ID values with no exact match in ref_ids (already deduplicated).

        Returns:
            (mapping, metadata) where mapping is candidate ID -> reference ID for accepted pairs
            only (each reference ID used at most once), and metadata contains the LLM's overall
            reasoning and confidence for the mapping.
        """
        if not ref_ids or not alt_ids:
            return {}, {}

        agent = create_llm_agent(
            model=self.model,
            system_prompt="You are a data alignment expert. Match candidate row identifiers to reference row identifiers that refer to the same underlying entity despite differences in formatting, casing, delimiters, word order, or naming/abbreviation. Only map a pair when you are confident they refer to the same entity; use null for candidate identifiers with no confident match. Never map two candidates to the same reference identifier.",
            response_format=IndexMapping,
        )

        prompt = f"""For EACH candidate identifier below, find the reference identifier it refers to, if any.

Candidate identifiers (source): {alt_ids}
Reference identifiers (target): {ref_ids}

Provide a mapping from EACH candidate identifier to its matching reference identifier, or null if no confident match exists. A reference identifier may be used at most once."""

        try:
            response = invoke_structured_agent(agent, prompt, IndexMapping)
        except Exception as e:
            logger.warning(f"LLM ID fuzzy matching failed: {e}")
            return {}, {}

        ref_set, alt_set = set(ref_ids), set(alt_ids)
        mapping: Dict[str, str] = {}
        used_ref = set()
        for alt_id, ref_id in response.mapping.items():
            if alt_id not in alt_set or ref_id is None or ref_id not in ref_set or ref_id in used_ref:
                continue
            mapping[alt_id] = ref_id
            used_ref.add(ref_id)

        return mapping, {"reasoning": response.reasoning, "confidence": response.confidence}

    def _align_rows(self, ref_df: DataFrame, alt_df: DataFrame, id_columns: Optional[List[str]] = None, ignore_case: bool = True, fuzzy_id_match: bool = True) -> Tuple[DataFrame, DataFrame, Dict[str, Any]]:
        """Align candidate rows to reference rows.

        Resolution order:
        1. Use provided id_columns (from plan) if available
        2. Auto-detect ID columns via LLM (_detect_id_columns) as fallback
        3. Index-based alignment if no suitable ID columns found

        Args:
            ref_df: Reference dataframe
            alt_df: Candidate dataframe
            id_columns: Optional pre-determined ID columns from plan.
                        Non-empty list -> use directly (no LLM call).
                        Empty list [] -> skip ID-based, go to index-based.
                        None -> auto-detect via _detect_id_columns (LLM fallback).
            ignore_case: Whether to do case-insensitive index matching (default True)
            fuzzy_id_match: When True and a single ID column is used, IDs left unmatched after
                            exact (case-insensitive) comparison are additionally paired via LLM
                            semantic matching (_fuzzy_match_ids), so e.g. "medium_vs_low" and
                            "Medium vs Low" can still align even without lexical similarity. Only
                            applies to single-column IDs; composite keys always use exact matching
                            only. Default True — a no-op (no LLM call) when all IDs already match.

        Returns:
            (ref_aligned_df, alt_aligned_df, metadata)
        """
        id_metadata = {}
        
        # Determine ID columns: use provided, detect via LLM, or skip
        if id_columns is None:
            # Auto-detect via LLM fallback
            id_cols, id_metadata = self._detect_id_columns(ref_df, alt_df)
        elif len(id_columns) > 0:
            id_cols = list(id_columns)
        else:
            # Explicit empty list -> skip ID-based alignment
            id_cols = []
        
        # Validate ID columns exist in both dataframes
        id_cols = [col for col in id_cols if col in ref_df.columns and col in alt_df.columns]
        
        # Step 1: Try ID column-based alignment
        if id_cols:
            try:
                ref_merge = ref_df.copy()
                alt_merge = alt_df.copy()
                if ignore_case:
                    for col in id_cols:
                        ref_merge[col] = ref_merge[col].astype(str).str.lower()
                        alt_merge[col] = alt_merge[col].astype(str).str.lower()

                fuzzy_matched_ids, fuzzy_meta = {}, {}
                if fuzzy_id_match:
                    if len(id_cols) == 1:
                        id_col = id_cols[0]
                        ref_id_vals = ref_merge[id_col].astype(str)
                        alt_id_vals = alt_merge[id_col].astype(str)
                        ref_only = list(set(ref_id_vals) - set(alt_id_vals))
                        alt_only = list(set(alt_id_vals) - set(ref_id_vals))
                        fuzzy_matched_ids, fuzzy_meta = self._fuzzy_match_ids(ref_only, alt_only)
                        if fuzzy_matched_ids:
                            alt_merge[id_col] = alt_id_vals.replace(fuzzy_matched_ids)
                    else:
                        logger.warning(f"fuzzy_id_match only supports a single ID column; ignoring for composite key {id_cols}")

                merged = ref_merge.merge(alt_merge, on=id_cols, how='outer', suffixes=('_ref', '_alt'), indicator=True)
                
                ref_data_cols = [col for col in ref_df.columns if col not in id_cols]
                alt_data_cols = [col for col in alt_df.columns if col not in id_cols]
                common_data_cols = set(ref_data_cols) & set(alt_data_cols)
                ref_unique_cols = [col for col in ref_data_cols if col not in common_data_cols]
                alt_unique_cols = [col for col in alt_data_cols if col not in common_data_cols]
                
                # Preserve ref/alt column order (common_data_cols is a set; iterate in ref/alt order)
                ref_shared_merged = [f'{col}_ref' for col in ref_data_cols if col in common_data_cols and f'{col}_ref' in merged.columns]
                ref_unique_merged = [col for col in ref_unique_cols if col in merged.columns]
                ref_aligned_df = merged[id_cols + ref_shared_merged + ref_unique_merged].copy()
                
                alt_shared_merged = [f'{col}_alt' for col in alt_data_cols if col in common_data_cols and f'{col}_alt' in merged.columns]
                alt_unique_merged = [col for col in alt_unique_cols if col in merged.columns]
                alt_aligned_df = merged[id_cols + alt_shared_merged + alt_unique_merged].copy()
                
                # Fix suffix bug: rename suffixed columns back to original names
                ref_rename = {f'{col}_ref': col for col in common_data_cols if f'{col}_ref' in ref_aligned_df.columns}
                ref_aligned_df = ref_aligned_df.rename(columns=ref_rename)
                
                alt_rename = {f'{col}_alt': col for col in common_data_cols if f'{col}_alt' in alt_aligned_df.columns}
                alt_aligned_df = alt_aligned_df.rename(columns=alt_rename)
                
                # Count unmatched rows using merge indicator
                unmapped_ref = int((merged['_merge'] == 'left_only').sum())
                unmapped_alt = int((merged['_merge'] == 'right_only').sum())
                
                return ref_aligned_df, alt_aligned_df, {
                    "alignment_method": "id_column_merge",
                    "id_columns": id_cols,
                    "unmapped_ref_rows": unmapped_ref,
                    "unmapped_alt_rows": unmapped_alt,
                    "fuzzy_matched_ids": fuzzy_matched_ids,
                    "fuzzy_match_metadata": fuzzy_meta,
                    **id_metadata,
                }
            except Exception as e:
                logger.warning(f"ID column alignment failed: {e}, falling back to index-based alignment")
        
        # Step 2: Index-based alignment (with optional case-insensitive matching)
        ref_idx = list(ref_df.index)
        alt_idx = list(alt_df.index)
        normalize = (lambda x: str(x).lower()) if ignore_case else (lambda x: str(x))
        ref_idx_mapping = {idx: normalize(idx) for idx in ref_idx}
        alt_idx_mapping = {idx: normalize(idx) for idx in alt_idx}
        
        # Shared and union of normalized indices (preserves insertion order for deterministic row order)
        ref_normalized = set(ref_idx_mapping.values())
        alt_normalized = set(alt_idx_mapping.values())
        shared_normalized_idx = ref_normalized & alt_normalized
        combined_idx = sorted(ref_normalized | alt_normalized, key=str)
        
        # Map original indices to normalized indices, then reindex to combined_idx
        ref_aligned_df = ref_df.copy()
        ref_aligned_df.index = [ref_idx_mapping[idx] for idx in ref_df.index]
        ref_aligned_df = ref_aligned_df.reindex(combined_idx)

        alt_aligned_df = alt_df.copy()
        alt_aligned_df.index = [alt_idx_mapping[idx] for idx in alt_df.index]
        alt_aligned_df = alt_aligned_df.reindex(combined_idx)
                
        # Determine alignment method
        if not shared_normalized_idx:
            alignment_method = "none"
        elif len(shared_normalized_idx) == len(ref_idx) == len(alt_idx):
            alignment_method = "case_insensitive" if ignore_case else "exact"
        else:
            alignment_method = "intersection"
        
        return ref_aligned_df, alt_aligned_df, {
            "alignment_method": alignment_method,
            "id_columns": None,
            "unmapped_ref_rows": len(ref_normalized - shared_normalized_idx),
            "unmapped_alt_rows": len(alt_normalized - shared_normalized_idx),
        }
    
    def _align_dtype(self, ref_df: DataFrame, alt_df: DataFrame) -> Tuple[DataFrame, DataFrame, Dict[str, Any]]:
        """Align data types of alt_df to match ref_df for common columns.
        
        Converts dtypes of common columns in alt_df to match those in ref_df.
        This helps ensure type differences don't affect value comparisons.
        
        Args:
            ref_df: Reference dataframe.
            alt_df: Candidate dataframe to align.
        
        Returns:
            (ref_aligned_df, alt_aligned_df, alignment_info) where alignment_info contains:
            - converted_columns: List of columns that were successfully converted
            - failed_columns: List of columns where conversion failed
            - conversion_details: Dict mapping column to (ref_type, alt_type, success)
        """
        common_cols = list(set(ref_df.columns) & set(alt_df.columns))
        alt_aligned_df = alt_df.copy()
        converted_columns = []
        failed_columns = []
        conversion_details = {}
        
        for col in common_cols:
            ref_type = ref_df[col].dtype
            alt_type = alt_df[col].dtype
            
            if ref_type == alt_type:
                conversion_details[col] = (str(ref_type), str(alt_type), True)
                continue
            
            try:
                alt_aligned_df[col] = alt_aligned_df[col].astype(ref_type)
                converted_columns.append(col)
                conversion_details[col] = (str(ref_type), str(alt_type), True)
            except (ValueError, TypeError) as e:
                failed_columns.append(col)
                conversion_details[col] = (str(ref_type), str(alt_type), False)
                logger.debug(f"Failed to convert column '{col}' from {alt_type} to {ref_type}: {e}")
        
        return ref_df, alt_aligned_df, {
            "shared_columns": common_cols,
            "converted_columns": converted_columns,
            "failed_columns": failed_columns,
            "conversion_details": conversion_details,
        }
    
    def _get_align_plan(self, ref_df: DataFrame, alt_df: DataFrame, question: str, eval_guideline: Optional[str] = None) -> Dict[str, Any]:
        """Generate alignment plan for structural alignment of dataframes.
        
        Analyzes task description and both table previews to determine how to align
        the dataframes structurally:
        1. Table presentation (transpose)
        2. Column mapping (ref_col -> alt_col)
        3. Row identifiers (for row alignment)
        4. Sorting and ordering
        
        All decisions are inferred from:
        - Task description: Provides context about what the task is doing
        - Data previews: Show column names, data types, and structure
        
        If column_mapping or id_columns are not provided by the LLM, the downstream
        _align_columns and _align_rows methods will fall back to their own LLM calls.
        
        Args:
            ref_df: Reference dataframe
            alt_df: Candidate dataframe
            question: Description of the comparison task
            
        Returns:
            Dict with alignment configuration
        """
        agent = create_llm_agent(
            model=self.model,
            system_prompt="You are a data alignment expert. Analyze tabular data structures and create an alignment plan that guides how to structurally align two dataframes (transpose, column mapping, row matching, sorting).",
            response_format=AlignPlan
        )
        
        ref_preview = self._get_preview(ref_df, self.preview_max_rows)
        alt_preview = self._get_preview(alt_df, self.preview_max_rows)
        
        guideline_line = f"\nEvaluation guideline: {eval_guideline}" if eval_guideline else ""
        prompt = f"""Create an alignment plan for structurally aligning these two dataframes.

Task: {question}{guideline_line}

Reference Data (shape: {ref_df.shape[0]} rows x {ref_df.shape[1]} columns):
{ref_preview}

Candidate Data (shape: {alt_df.shape[0]} rows x {alt_df.shape[1]} columns):
{alt_preview}

ANALYSIS INSTRUCTIONS:
- Analyze the task description to understand the context
- Examine the data previews to understand column names, data types, and structure
- Determine how to structurally align the dataframes (transpose, column mapping, row matching)

PROVIDE ALL OF THE FOLLOWING:

1. TABLE PRESENTATION:
   - needs_transpose: Are these tables transposed versions of each other?

2. COLUMN MAPPING:
   - align_columns: Should columns be aligned? (true unless names already match exactly)
   - column_mapping: For EACH reference column, provide the matching candidate column name.
     * Format: {{"ref_col1": "alt_col1", "ref_col2": "alt_col2", "ref_col3": null}}
     * Use exact candidate column names as they appear in the candidate data
     * Use null for a ref column if no good match exists in candidate
     * Set the entire field to null ONLY if column names already match exactly

3. ROW ALIGNMENT:
   - align_rows: Should rows be aligned?
   - id_columns: Which column(s) uniquely identify rows for matching between tables?
     * Use REFERENCE column names
     * Examples: ["gene_id"] for single key, ["chr", "start", "end"] for composite key
     * Use [] (empty list) if no suitable ID columns exist (will use positional alignment)
     * Use null to let the system auto-detect
   - row_order_sensitive: Does row order matter?
     * True: ranked lists (top N genes by expression)
     * False: sets/unordered data (gene lists, sample sets)
     * Infer from task description (e.g., "top N" → True, "gene list" → False)

4. SORTING:
   - pre_sort: Sort before comparison?
   - sort_by: Column(s) to sort by (comma-separated, e.g., "gene_id" or "chr,start")

5. PERMUTATION DETECTION:
   - check_permutation: Check if data is a permutation? (true if row_order_sensitive is false)

6. METADATA:
   - reasoning: Brief explanation of your alignment strategy, including:
     * Why certain columns were mapped
     * Why specific ID columns were chosen
     * Why transpose was needed (if applicable)
   - confidence: Confidence score (0.0-1.0) for the alignment plan. Higher values indicate more certainty (e.g., exact column name matches, clear ID columns)

Based on the task description and data previews, provide a COMPLETE alignment configuration including column_mapping, id_columns, reasoning, and confidence."""

        try:
            config = invoke_structured_agent(agent, prompt, AlignPlan)
            config_dict = config.model_dump()
            logger.info(f"Generated alignment plan: {json.dumps(config_dict, indent=2, default=str)}")
            return config_dict
        except Exception as e:
            logger.warning(f"Failed to get alignment plan: {e}")
        
        # Default alignment plan (column_mapping=None and id_columns=None trigger LLM fallbacks)
        return {
            "needs_transpose": False,
            "align_rows": True,
            "align_columns": True,
            "column_mapping": None,
            "id_columns": None,
            "row_order_sensitive": False,
            "column_order_sensitive": False,
            "pre_sort": False,
            "sort_by": None,
            "check_permutation": True,
            "reasoning": "Default alignment plan used due to LLM call failure. Column mapping and ID columns will be auto-detected via fallback methods.",
            "confidence": 0.0
        }
    
    def _align_dataframes(self, ref_df: DataFrame, alt_df: DataFrame, plan: Dict[str, Any], fuzzy_id_match: bool = True) -> Tuple[DataFrame, DataFrame, Dict[str, Any]]:
        """Align two dataframes based on alignment plan.

        Executes alignment operations in order:
        1. Transpose (if plan.needs_transpose)
        2. Column alignment (using plan.column_mapping if available, otherwise LLM or normalized matching fallback)
        3. Row alignment (using plan.id_columns if available, otherwise LLM or index-based fallback)
        4. Data type alignment
        5. Sorting (if plan.pre_sort)

        Uses plan-first approach: if plan provides column_mapping or id_columns, uses them directly
        (no LLM calls). Otherwise falls back to individual function logic (LLM or deterministic matching).

        Args:
            ref_df: Reference dataframe
            alt_df: Candidate dataframe
            plan: Alignment plan from _get_align_plan
            fuzzy_id_match: Passed through to _align_rows. When True, single-column row IDs left
                            unmatched after exact/case-insensitive comparison are additionally
                            paired via LLM semantic matching. Default False.

        Returns:
            (ref_aligned, alt_aligned, alignment_metadata)
        """
        metadata = {}
        notes = []
        
        # Step 1: Transpose (trust LLM's semantic judgment from _get_comparison_plan)
        if plan.get("needs_transpose", False):
            alt_df = alt_df.T
            notes.append("Candidate was transposed to match reference structure")
            metadata["was_transposed"] = True
        
        # Step 2: Column alignment (use plan's column_mapping if available, otherwise fallback to LLM or normalized matching)
        if plan.get("align_columns", True):
            column_mapping = plan.get("column_mapping")
            ref_df, alt_df, col_meta = self._align_columns(ref_df, alt_df, column_mapping=column_mapping)
            metadata["column_alignment"] = col_meta
            if col_meta.get("unmapped_ref"):
                notes.append(f"Unmapped reference columns: {col_meta['unmapped_ref']}")
            if col_meta.get("unmapped_alt"):
                notes.append(f"Unmapped candidate columns: {col_meta['unmapped_alt']}")
        else:
            metadata["column_alignment"] = {"alignment_method": "none"}
        
        # Step 3: Row alignment (pass plan's id_columns; falls back to LLM if None)
        if plan.get("align_rows", True):
            id_columns = plan.get("id_columns")
            ref_df, alt_df, row_meta = self._align_rows(ref_df, alt_df, id_columns=id_columns,
                                                         fuzzy_id_match=fuzzy_id_match)
            metadata["row_alignment"] = row_meta
            unmapped_ref = row_meta.get("unmapped_ref_rows", 0)
            unmapped_alt = row_meta.get("unmapped_alt_rows", 0)
            if unmapped_ref > 0:
                notes.append(f"Unmapped reference rows: {unmapped_ref}")
            if unmapped_alt > 0:
                notes.append(f"Unmapped candidate rows: {unmapped_alt}")
        else:
            metadata["row_alignment"] = {"alignment_method": "none"}
        
        # Step 4: Data type alignment
        ref_df, alt_df, dtype_meta = self._align_dtype(ref_df, alt_df)
        metadata["dtype_alignment"] = dtype_meta
        if dtype_meta.get("converted_columns"):
            notes.append(f"Converted {len(dtype_meta['converted_columns'])} column(s) to match reference dtypes")
        if dtype_meta.get("failed_columns"):
            notes.append(f"Failed to convert {len(dtype_meta['failed_columns'])} column(s)")
        
        # Step 5: Sorting
        if plan.get("pre_sort") and plan.get("sort_by"):
            sort_col = plan["sort_by"]
            sort_cols = [c.strip() for c in sort_col.split(",")] if isinstance(sort_col, str) else sort_col
            
            valid_sort_cols = [c for c in sort_cols if c in ref_df.columns and c in alt_df.columns]
            if valid_sort_cols:
                ref_df = self._sort_by_column(ref_df, valid_sort_cols)
                alt_df = self._sort_by_column(alt_df, valid_sort_cols)
                notes.append(f"Sorted both dataframes by {valid_sort_cols}")
                metadata["sorting"] = {"columns": valid_sort_cols, "applied": True}
            else:
                logger.warning(f"Sort columns {sort_cols} not found in both dataframes after alignment")
                metadata["sorting"] = {"columns": sort_cols, "applied": False}
        else:
            metadata["sorting"] = {"applied": False}
        
        metadata["notes"] = notes
        return ref_df, alt_df, metadata
    

    # ============================================================================
    # Comparison Pipeline
    # ============================================================================

    def _get_comparison_plan(self, ref_aligned: DataFrame, alt_aligned: DataFrame, question: str, id_columns: Optional[List[str]] = None, column_dtypes: Optional[Dict[str, str]] = None, eval_guideline: Optional[str] = None) -> Dict[str, Any]:
        """Generate comparison plan for comparing aligned dataframes.
        
        Analyzes task description, column dtypes, and ID columns to determine:
        1. Which data columns to compare (excluding ID columns)
        2. Per-column comparison strategies based on data types
        3. Tolerance for numerical comparisons, delimiter and threshold for categorical "overlap" strategy
        
        All decisions are inferred from:
        - Task description: Provides context about what the task is doing
        - Column dtypes: Explicit data types for each column (categorical vs numerical)
        - ID columns: Columns used for alignment (excluded from comparison)
        - Aligned data previews: Show sample values and data scale
        
        Args:
            ref_aligned: Reference dataframe (already aligned)
            alt_aligned: Candidate dataframe (already aligned)
            question: Description of the comparison task
            id_columns: Optional list of column names used as row identifiers (excluded from comparison)
            column_dtypes: Optional dict mapping column name to dtype string (e.g., {'col1': 'object', 'col2': 'float64'})
            
        Returns:
            Dict with comparison strategy configuration
        """
        agent = create_llm_agent(
            model=self.model,
            system_prompt="You are a data comparison expert. Analyze aligned tabular data with explicit dtype information and create a dtype-aware comparison plan that guides how to compare values per column.",
            response_format=ComparisonPlan
        )
        
        ref_preview = self._get_preview(ref_aligned, self.preview_max_rows)
        alt_preview = self._get_preview(alt_aligned, self.preview_max_rows)
        
        # Build dtype information string
        if column_dtypes is None:
            # Infer from dataframes if not provided
            common_cols = sorted(set(ref_aligned.columns) & set(alt_aligned.columns))
            column_dtypes = {col: str(ref_aligned[col].dtype) for col in common_cols}
        
        # Build ID columns info
        id_cols_info = "None (using index-based alignment)" if not id_columns else ", ".join(id_columns)
        
        # Determine data columns (exclude ID columns) and filter dtype info to only data columns
        all_cols = sorted(set(ref_aligned.columns) & set(alt_aligned.columns))
        data_cols = [col for col in all_cols if id_columns is None or col not in id_columns]
        data_cols_dtypes = {col: dtype for col, dtype in column_dtypes.items() if col in data_cols}
        dtype_info = "\n".join([f"  - {col}: {dtype}" for col, dtype in sorted(data_cols_dtypes.items())])
        
        guideline_line = f"\nEvaluation guideline: {eval_guideline}" if eval_guideline else ""
        prompt = f"""Create a comparison plan for comparing these aligned dataframes.

Task: {question}{guideline_line}

ID Columns (used for alignment, EXCLUDE from comparison): {id_cols_info}

Column Data Types (for comparison):
{dtype_info if dtype_info else '  None'}

Reference Data (aligned, shape: {ref_aligned.shape[0]} rows x {ref_aligned.shape[1]} columns):
{ref_preview}

Candidate Data (aligned, shape: {alt_aligned.shape[0]} rows x {alt_aligned.shape[1]} columns):
{alt_preview}

ANALYSIS INSTRUCTIONS:
- Analyze the task description to understand what aspects are important to compare
- Use the explicit column dtypes to determine appropriate comparison strategies
- ID columns are for alignment only - DO NOT include them in column_configs
- Determine which data columns are critical based on task description
- For numerical columns with "rae" strategy: determine appropriate cell_tolerance by examining value scales (percentages → 0.01, and other reasonable tolerances based on value characteristics)
- For categorical columns with "overlap" strategy: determine appropriate cell_threshold for Jaccard similarity (e.g., 0.8 for high similarity requirement, 1.0 for exact match)

PROVIDE ALL OF THE FOLLOWING:

1. COLUMN-SPECIFIC COMPARISON (column_configs):
   Create a list of ColumnCompareConfig objects, one for each data column to compare (excluding ID columns).
   Each config should specify:
   
   - name: Column name (reference column name after alignment, excluding ID columns)
   - data_type: Data type - "categorical" (object, string, category) or "numerical" (int, float, numeric)
   - strategy: Comparison strategy based on DATA TYPE:
     * For CATEGORICAL columns:
       - "exact": Whole cell exact equality (e.g., "gene1" == "gene1")
       - "overlap": Parse cell by delimiter, then calculate Jaccard similarity (e.g., "gene1,gene2" vs "gene2,gene3")
       - "ignore": Skip this column
     * For NUMERICAL columns:
       - "pearson": Pearson correlation (for linear relationships, e.g., expression values)
       - "spearman": Spearman correlation (for monotonic relationships, e.g., rankings, percentiles)
       - "rae": Relative Absolute Error (for absolute error matters, e.g., counts, measurements)
       - "ignore": Skip this column
     * Choose strategy based on: data type, task requirements, and value characteristics
   
   - cell_delimiter: Optional delimiter for categorical "overlap" strategy
     * String like "," (comma), ";" (semicolon), "|" (pipe), "\\t" (tab)
     * None if no parsing needed (treat cell as single value)
     * Only set for categorical columns with "overlap" strategy
   
   - cell_tolerance: Optional tolerance for RAE strategy (numerical columns)
     * When set: RAE values <= tolerance are counted as matching
     * When None: raw RAE values are averaged and converted to similarity
     * For "rae": threshold on RAE (e.g., 0.01 for percentages, 1e-6 for floats, 1.0 for integers)
     * For other strategies: None (not applicable)
   
   - cell_threshold: Optional threshold for Jaccard similarity in overlap strategy (categorical columns)
     * When set: pairs with Jaccard similarity >= threshold are counted as matching
     * When None: raw Jaccard similarity values are averaged
     * For "overlap": threshold on Jaccard similarity (e.g., 0.8)
     * For other strategies: None (not applicable)
   
   Include only columns that are important for the task. If all data columns are equally important, include all of them.
   DO NOT include ID columns.

2. METADATA:
   - reasoning: Brief explanation of your comparison strategy, including:
     * How you determined which columns are critical (excluding ID columns)
     * How you chose strategies based on data types
     * How you determined appropriate cell_tolerance (for RAE) or cell_threshold (for overlap) for each column
     * For categorical "overlap" strategy, why you chose specific delimiters
   - confidence: Confidence score (0.0-1.0) for the comparison plan

Based on the task description, column dtypes, and aligned data previews, provide a COMPLETE comparison configuration including column_configs (list of ColumnCompareConfig), reasoning, and confidence."""

        try:
            config = invoke_structured_agent(agent, prompt, ComparisonPlan)
            config_dict = config.model_dump()
            logger.info(f"Generated comparison plan: {json.dumps(config_dict, indent=2, default=str)}")
            return config_dict
        except Exception as e:
            logger.warning(f"Failed to get comparison plan: {e}")
        
        # Default comparison plan
        return {
            "column_configs": None,
            "reasoning": "Default comparison plan used due to LLM call failure.",
            "confidence": 0.0
        }

    def _get_col_equal(self, ref_series: Series, alt_series: Series) -> ColumnSimilarity:
        """Calculate exact equality similarity for categorical columns.
        
        Args:
            ref_series: Reference column data (already aligned)
            alt_series: Candidate column data (already aligned, same index as ref_series)
            
        Returns:
            ColumnSimilarity with similarity metrics for this column
        """
        if ref_series.empty and alt_series.empty:
            return ColumnSimilarity(similarity=1.0, strategy="exact", error="Both empty")
        if ref_series.empty or alt_series.empty:
            return ColumnSimilarity(similarity=0.0, strategy="exact", error="One empty")
        
        try:
            eq = (ref_series == alt_series) | (ref_series.isna() & alt_series.isna())
            match_count = int(eq.sum())
            similarity = match_count / len(ref_series) if len(ref_series) > 0 else 0.0
            return ColumnSimilarity(similarity=similarity, strategy="exact", 
                                    matching_count=match_count, total_count=len(ref_series))
        except Exception as e:
            logger.error(f"Exact equality calculation failed: {e}")
            return ColumnSimilarity(similarity=None, strategy="exact", error=f"Exact equality calculation failed: {e}")

    def _get_col_overlap(self, ref_series: Series,  alt_series: Series, delimiter: Optional[str] = None, threshold: float = 1.0) -> ColumnSimilarity:
        """Calculate Jaccard similarity for categorical columns with optional parsing.
        
        If delimiter is provided, splits each cell by delimiter and treats as sets.
        Otherwise, treats each cell as a single value and uses exact equality.
        
        Args:
            ref_series: Reference column data (already aligned)
            alt_series: Candidate column data (already aligned, same index as ref_series)
            delimiter: Optional delimiter string to split cell values (e.g., ',', ';', '|', '\\t').
                      If None, treats each cell as a single value.
            threshold: Minimum Jaccard similarity threshold (0.0-1.0) to count as matching.
                      Pairs with similarity >= threshold are counted in matching_count.
        
        Returns:
            ColumnSimilarity with similarity metrics for this column
        """
        if ref_series.empty and alt_series.empty:
            return ColumnSimilarity(similarity=1.0, strategy="overlap", error="Both empty")
        if ref_series.empty or alt_series.empty:
            return ColumnSimilarity(similarity=0.0, strategy="overlap", error="One empty")

        def _value_to_set(val) -> set:
            """Convert a value to a set, handling delimiter and NaN."""
            if pd.isna(val):
                return set()
            val_str = str(val).strip()
            if not val_str:
                return set()
            if delimiter is not None:
                return {token.strip() for token in val_str.split(delimiter) if token.strip()}
            return {val_str}
        
        def _jaccard_similarity(set1: set, set2: set) -> float:
            """Calculate Jaccard similarity between two sets."""
            if not set1 and not set2:
                return 1.0
            if not set1 or not set2:
                return 0.0
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / union

        try:
            jaccard_similarities = [
                _jaccard_similarity(_value_to_set(ref_val), _value_to_set(alt_val))
                for ref_val, alt_val in zip(ref_series, alt_series)
            ]
            
            matching_count = int(np.sum(np.array(jaccard_similarities) >= threshold))
            mean_similarity = float(np.mean(jaccard_similarities))
            return ColumnSimilarity(similarity=mean_similarity, strategy="overlap",
                                    matching_count=matching_count, total_count=len(ref_series),
                                    details={"delimiter": delimiter, "threshold": threshold,
                                             "valid_pairs": len(jaccard_similarities),
                                             "total_pairs": len(ref_series)})
        except Exception as e:
            logger.error(f"Overlap calculation failed: {e}")
            return ColumnSimilarity(similarity=None, strategy="overlap", error=f"Overlap calculation failed: {e}")

    def _get_col_rae(self, ref_series: Series, alt_series: Series, tolerance: float) -> ColumnSimilarity:
        """Calculate relative absolute error (RAE) similarity for numerical columns.
        
        Args:
            ref_series: Reference column data (already aligned)
            alt_series: Candidate column data (already aligned, same index as ref_series)
            tolerance: Numeric tolerance for RAE comparison
            
        Returns:
            ColumnSimilarity with similarity metrics for this column
        """
        if ref_series.empty and alt_series.empty:
            return ColumnSimilarity(similarity=1.0, strategy="rae", error="Both empty")
        if ref_series.empty or alt_series.empty:
            return ColumnSimilarity(similarity=0.0, strategy="rae", error="One empty")
        
        try:
            # Calculate similarity using rae_similarity
            sim_array = rae_similarity(ref_series.values, alt_series.values, clip=True)
            
            # Handle NaN values
            valid_mask = ~np.isnan(sim_array)
            if not valid_mask.any():
                return ColumnSimilarity(similarity=0.0, strategy="rae", error="All values invalid")
            
            sim_valid = sim_array[valid_mask]
            # Convert tolerance (RAE threshold) to similarity threshold
            # RAE <= tolerance means similarity >= (1 - tolerance)
            similarity_threshold = 1.0 - tolerance
            # Calculate matching_count: count cells with similarity >= threshold
            matching_count = int(np.sum(sim_valid >= similarity_threshold))
            # Calculate similarity: average of similarity scores
            similarity = float(np.mean(sim_valid))
            return ColumnSimilarity(similarity=similarity, strategy="rae",
                                    matching_count=matching_count, total_count=int(valid_mask.sum()),
                                    details={"tolerance": tolerance})
        except Exception as e:
            logger.error(f"RAE difference calculation failed: {e}")
            return ColumnSimilarity(similarity=None, strategy="rae", error=f"RAE calculation failed: {e}")

    def _get_col_correlation(self, ref_series: Series, alt_series: Series, method: str = "pearson") -> ColumnSimilarity:
        """Calculate correlation similarity for numerical columns.
        
        Args:
            ref_series: Reference column data (already aligned)
            alt_series: Candidate column data (already aligned, same index as ref_series)
            method: Correlation method, either "pearson" or "spearman"
            
        Returns:
            ColumnSimilarity with similarity metrics for this column
        """
        if ref_series.empty and alt_series.empty:
            return ColumnSimilarity(similarity=1.0, strategy=method, error="Both empty")
        if ref_series.empty or alt_series.empty:
            return ColumnSimilarity(similarity=0.0, strategy=method, error="One empty")
        
        try:
            if method == "pearson":
                corr = pearson_correlation(ref_series.values, alt_series.values, nan_policy="omit")
            elif method == "spearman":
                corr = spearman_correlation(ref_series.values, alt_series.values, nan_policy="omit")
            else:
                return ColumnSimilarity(similarity=None, strategy=method, error=f"Unknown correlation method: {method}")
            
            if corr is None or np.isnan(corr):
                return ColumnSimilarity(similarity=0.0, strategy=method, error="Correlation calculation failed")
            
            # Map [-1, 1] to [0, 1] using (corr + 1) / 2
            similarity = float((corr + 1) / 2)
            return ColumnSimilarity(similarity=similarity, strategy=method,
                                    total_count=len(ref_series), details={"correlation": corr})
        except Exception as e:
            logger.error(f"{method.capitalize()} correlation calculation failed: {e}")
            return ColumnSimilarity(similarity=None, strategy=method, error=f"Correlation calculation failed: {e}")

    def _get_col_similarity(self, ref_series: Series, alt_series: Series, strategy: str, **kwargs) -> ColumnSimilarity:
        """Calculate similarity for a single column based on dtype-aware comparison strategy.
        
        Args:
            ref_series: Reference column data (already aligned)
            alt_series: Candidate column data (already aligned, same index as ref_series)
            strategy: Comparison strategy based on data type:
                     - For categorical: "exact" (whole cell equality) or "overlap" (parse by delimiter, then Jaccard)
                     - For numerical: "pearson", "spearman", or "rae" (relative absolute error)
                     - "ignore" to skip this column
            **kwargs: Strategy-specific parameters:
                     - tolerance: Numeric tolerance for RAE comparison (only used for "rae" strategy, default: 0.01)
                     - delimiter: Optional delimiter string for categorical "overlap" strategy (e.g., ',', ';', '|', '\\t')
                     - threshold: Jaccard similarity threshold for "overlap" strategy (0.0-1.0, default: 1.0).
                                  Pairs with similarity >= threshold count as matching.
            
        Returns:
            ColumnSimilarity with similarity metrics for this column
        """
        # Categorical strategies
        if strategy == "ignore":
            return ColumnSimilarity(similarity=None, strategy="ignore")
        if strategy == "exact":
            return self._get_col_equal(ref_series, alt_series)
        elif strategy == "overlap":
            delimiter = kwargs.get("delimiter")
            threshold = kwargs.get("threshold", 1.0)
            return self._get_col_overlap(ref_series, alt_series, delimiter, threshold)
        # Numerical strategies
        elif strategy == "pearson":
            return self._get_col_correlation(ref_series, alt_series, method="pearson")
        elif strategy == "spearman":
            return self._get_col_correlation(ref_series, alt_series, method="spearman")
        elif strategy == "rae":
            tolerance = kwargs.get("tolerance", 0.01)
            return self._get_col_rae(ref_series, alt_series, tolerance)
        else:
            logger.warning(f"Unknown comparison strategy '{strategy}', falling back to exact")
            return self._get_col_equal(ref_series, alt_series)
    
    def _calculate_similarities(self, ref_aligned: DataFrame, alt_aligned: DataFrame, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate similarity metrics based on comparison plan.
        
        Orchestrates per-column similarity calculation using plan's focus and
        per-column strategies, plus structural metrics (overlap, shape, dtypes,
        missing patterns, permutation detection).
        
        Args:
            ref_aligned: Reference dataframe (already aligned)
            alt_aligned: Candidate dataframe (already aligned)
            plan: Comparison plan with focus, tolerance, and per-column strategies
            
        Returns:
            Dict with overall_similarity and detailed per-column and structural metrics
        """
        # Get columns to compare: common columns, excluding ID columns, optionally filtered by column_configs
        common_cols = sorted(set(ref_aligned.columns) & set(alt_aligned.columns))
        if not common_cols:
            return {"compared_columns": [], "overall_similarity": 0.0, "notes": ["No columns available for comparison"]}
        
        # Exclude ID columns
        id_columns = plan.get("id_columns") or []
        data_cols = [col for col in common_cols if col not in id_columns]
        
        # Build config lookup from column_configs
        column_configs = plan.get("column_configs") or []
        config_lookup = {cfg.get("name"): cfg for cfg in column_configs if cfg.get("name")}
        
        # Determine which columns to compare: use column_configs if available, otherwise all data columns
        compare_cols = set(cfg.get("name") for cfg in column_configs if cfg.get("name"))
        compared_columns = sorted(compare_cols & set(data_cols)) or data_cols
        
        if not compared_columns:
            return {"compared_columns": [], "overall_similarity": 0.0, "notes": ["No columns available for comparison"]}
        
        # Filter out columns marked as "ignore" and calculate similarities
        column_similarities = {}
        default_tolerance = 1e-6
        
        for col in compared_columns:
            config_dict = config_lookup.get(col) or {}
            strategy = config_dict.get("strategy")
            if strategy == "ignore":
                continue
            
            # Determine strategy: from config, or infer from dtype
            if not strategy:
                is_numeric = pd.api.types.is_numeric_dtype(ref_aligned[col].dtype)
                strategy = "rae" if is_numeric else "exact"
            
            column_similarities[col] = self._get_col_similarity(
                ref_aligned[col], alt_aligned[col], strategy,
                tolerance=config_dict.get("cell_tolerance") or default_tolerance,
                delimiter=config_dict.get("cell_delimiter"),
                threshold=config_dict.get("cell_threshold", 1.0)
            )
        
        if not column_similarities:
            return {"compared_columns": compared_columns, "overall_similarity": 0.0, "notes": ["All columns marked as 'ignore'"]}
        
        # Get filtered columns for metrics (guard against columns dropped during alignment)
        filtered_columns = [
            col for col in column_similarities.keys()
            if col in ref_aligned.columns and col in alt_aligned.columns
        ]
        ref_compare = ref_aligned[filtered_columns]
        alt_compare = alt_aligned[filtered_columns]
        
        # Calculate overall similarity
        valid_sims = [s.similarity for s in column_similarities.values() if s.similarity is not None]
        overall_similarity = float(np.mean(valid_sims)) if valid_sims else 0.0
        
        # Build metrics
        return {"compared_columns": compared_columns, "filtered_columns": filtered_columns, "overall_similarity": overall_similarity,
                "per_column_similarities": {col: sim.model_dump() for col, sim in column_similarities.items()},
                "overlap": self._get_dimension_overlap(ref_aligned, alt_aligned),
                "shape": self._get_shape_diff(ref_aligned, alt_aligned),
                "dtypes": self._get_dtype_match(ref_aligned, alt_aligned),
                "missing": self._get_missing_similarity(ref_compare, alt_compare)}
    
    
    # ============================================================================
    # File I/O and Loading Methods
    # ============================================================================
    
    def _read_raw_lines(self, file_path: str, n_lines: int = 30) -> list[str]:
        """Read first N raw lines from a file, handling common compressions.
        
        Args:
            file_path: Path to the file
            n_lines: Maximum number of lines to read
            
        Returns:
            List of raw line strings (including newlines)
        """
        signature = self._get_signature(file_path)
        try:
            import gzip, bz2, lzma
            openers = {
                "gzip": gzip.open, "bgzf": gzip.open,
                "bzip2": bz2.open, "xz": lzma.open,
            }
            opener = openers.get(signature.compression, open)
            with opener(file_path, "rt", encoding="utf-8", errors="replace") as f:
                return [f.readline() for _ in range(n_lines)]
        except Exception:
            return []
    
    def _get_table_meta(self, file_path: str, n_lines: int = 30) -> TableMeta:
        """Detect tabular parsing metadata (delimiter, comment char, header, skip rows).
        Uses LLM for detection with heuristic fallback. Results are cached per file.
        
        Args:
            file_path: Path to table file
            n_lines: Maximum number of lines to read
        
        Returns:
            TableMeta with detected parsing parameters
        """
        if file_path in self._tabular_meta_cache:
            return self._tabular_meta_cache[file_path]
        
        raw_lines = self._read_raw_lines(file_path, n_lines=n_lines)
        if not raw_lines:
            return TableMeta(reasoning="Empty file", confidence=0.0)
        
        # Try LLM detection
        try:
            table_content = "".join(raw_lines)
            agent = create_llm_agent(
                model=self.model,
                system_prompt=(
                    "You are a file parsing expert. Analyze the raw content of a tabular data file "
                    "and determine the correct parameters for loading it into a DataFrame. "
                    "Your job is to determine the exact delimiter, comment character, "
                    "whether a header row exists, and how many non-comment metadata lines to skip."
                ),
                response_format=TableMeta,
                temperature=0.0,
            )
            prompt = f"""Analyze this raw file content and determine the correct parsing parameters:

```
{table_content}
```

Determine:
1. **delimiter**: What character separates columns in the data lines (not in comment lines)?
   - Tab character: use "\\t"
   - Comma: use ","
   - Semicolon: use ";"
   - Pipe: use "|"
   - Space: use " "
   - Other: use the character itself

2. **comment_char**: Do any lines start with a comment/metadata marker character?
   - '#' is the most common in bioinformatics files
   - '%' in some formats (e.g., MATLAB output)
   - None if no comment lines exist

3. **has_header**: Is the first non-comment, non-blank line a header row with column names?
   - True if it contains descriptive names (e.g., "gene_id", "expression", "pvalue")
   - False if it goes directly into data values (numbers, sequences, etc.)

4. **skip_rows**: How many non-comment metadata lines at the top should be skipped before the header row?
   - Do NOT count comment lines (those are handled by comment_char)
   - Only count lines like "File: output.csv", "Generated by ...", or blank lines
     that don't start with the comment character
   - 0 if no such lines exist"""
            
            result = invoke_structured_agent(agent, prompt, TableMeta)
            self._tabular_meta_cache[file_path] = result
            return result
        except Exception:
            pass
        
        # Heuristic fallback
        result = self._get_table_meta_heuristic(raw_lines)
        self._tabular_meta_cache[file_path] = result
        return result

    def _get_table_meta_heuristic(self, raw_lines: list[str]) -> TableMeta:
        """Heuristic fallback for tabular metadata detection.
        
        Uses pattern matching for comment chars and csv.Sniffer for delimiter/header.
        
        Args:
            raw_lines: Raw file lines (including comments, blank lines)
            
        Returns:
            TableMeta with detected parsing parameters
        """
        if not raw_lines:
            return TableMeta(reasoning="No lines to analyze", confidence=0.0)
        
        # 1. Detect comment character from leading lines
        comment_char = None
        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                comment_char = "#"
            elif stripped.startswith("%"):
                comment_char = "%"
            break
        
        # 2. Filter to data lines (non-comment, non-empty)
        data_lines = [l for l in raw_lines
                      if l.strip() and (not comment_char or not l.strip().startswith(comment_char))]
        
        if len(data_lines) < 2:
            return TableMeta(comment_char=comment_char, reasoning="Too few data lines", confidence=0.3)
        
        # 3. Detect delimiter and header using csv.Sniffer
        sample = "".join(data_lines[:20])
        delimiter = None
        has_header = True
        try:
            dialect = csv.Sniffer().sniff(sample)
            delimiter = dialect.delimiter
        except csv.Error:
            pass
        try:
            has_header = csv.Sniffer().has_header(sample)
        except csv.Error:
            pass
        
        return TableMeta(delimiter=delimiter, comment_char=comment_char,
                         has_header=has_header, skip_rows=0,
                         reasoning="Detected via heuristic analysis",
                         confidence=0.7 if delimiter else 0.3)

    def _load_file(self, file_path: str, n_lines: int = 30, meta_dict: dict[str, Any] = None) -> DataFrame:
        """Load file as pandas DataFrame.
        
        Uses provided metadata dict or detects tabular metadata (delimiter, comment char, header, skip rows)
        via LLM or heuristic, then uses it to call pd.read_csv or pd.read_excel correctly.
        
        Args:
            file_path: Path to table file
            n_lines: Maximum number of lines to read (only used if meta_dict is not provided)
            meta_dict: Optional dict with TableMeta fields to use instead of detecting (avoids redundant detection)
        
        Returns:
            pandas DataFrame
        """
        signature = self._get_signature(file_path)
        if meta_dict is None:
            if signature.format == "gct":
                # GCT has a fixed structure: #1.x version line + dimensions line before the header.
                # skiprows=2 skips both (pandas counts comment lines toward the skiprows offset,
                # so comment="#" + skiprows=1 would leave the dimensions line as the header).
                meta_dict = {"delimiter": "\t", "comment_char": None, "skip_rows": 2, "has_header": True}
            else:
                meta_dict = self._get_table_meta(file_path, n_lines=n_lines).model_dump()

        has_header = meta_dict.get("has_header", True)
        skip_rows = meta_dict.get("skip_rows", 0)
        delimiter = meta_dict.get("delimiter", ",")
        comment_char = meta_dict.get("comment_char", None)

        # Handle Excel files
        if signature.format == "xlsx" or signature.extension in {"xlsx", "xls"}:
            read_kwargs = {
                "engine": "openpyxl",
                **({"header": None} if not has_header else {}),
                **({"skiprows": skip_rows} if skip_rows > 0 else {}),
            }
            df = pd.read_excel(file_path, **read_kwargs)
            # Convert integer column names to strings when no header is present
            if not has_header and any(isinstance(c, int) for c in df.columns):
                df.columns = [str(c) for c in df.columns]
            return df
        
        # Handle CSV/TSV files
        read_kwargs = {
            "delimiter": delimiter,
            "encoding": "utf-8",
            "encoding_errors": "replace",
            **({"comment": comment_char} if comment_char else {}),
            **({"header": None} if not has_header else {}),
            **({"skiprows": skip_rows} if skip_rows > 0 else {}),
        }
        df = pd.read_csv(file_path, **read_kwargs)
        
        # Convert integer column names to strings when no header is present
        # This ensures consistency with downstream components (LLM plans, Pydantic models, sort lookups)
        if not has_header and any(isinstance(c, int) for c in df.columns):
            df.columns = [str(c) for c in df.columns]
        
        return df

    def _get_preview(self, df: DataFrame, max_rows: int = 10) -> str:
        """Create text preview of dataframe.
        
        Args:
            df: pandas DataFrame
            max_rows: Maximum number of rows to include in preview (default: 10)
        
        Returns:
            String representation of dataframe preview
        """
        if len(df) == 0:
            return "(empty dataframe)"
        
        preview = f"Shape: {df.shape}\n"
        preview += f"Columns: {list(df.columns)}\n"
        preview += "-" * 80 + "\n"
        preview += df.head(max_rows).to_string()
        if len(df) > max_rows:
            preview += f"\n... ({len(df) - max_rows} more rows)"
        return preview
    
    # ============================================================================
    # Private Comparison Implementation Methods
    # ============================================================================
    
    def _compare_exact(self, ref_df: DataFrame, alt_df: DataFrame):
        """Exact comparison: strict matching of original dataframes.
        Uses pandas DataFrame.equals() for exact matching.
        Returns similarity = 1.0 if DataFrames are identical, 0.0 otherwise.
        
        Args:
            ref_df: Reference dataframe
            alt_df: Candidate dataframe
        """
        try:
            # Quick check: perfectly identical
            if ref_df.equals(alt_df):
                self.result.similarity = 1.0
                self.result.details.update({"exact_match": True})
                return
            
            # Cell-by-cell on common subtable
            ref_common, alt_common = self._get_common_subtable(ref_df, alt_df)
            if ref_common.empty:
                self.result.similarity = 0.0
                self.result.details.update({"exact_match": False, "reason": "No overlapping rows/columns"})
                return
            
            total_cells = 0
            matching_cells = 0
            
            for col in ref_common.columns:
                ref_col, alt_col = ref_common[col], alt_common[col]
                n = len(ref_col)
                
                # Strict equality for all columns (handles NaN == NaN)
                eq = (ref_col == alt_col) | (ref_col.isna() & alt_col.isna())
                matching_cells += int(eq.sum())
                total_cells += n
            
            similarity = matching_cells / total_cells if total_cells > 0 else 0.0
            
            self.result.similarity = similarity
            self.result.details.update({
                "exact_match": similarity == 1.0,
                "matching_cells": matching_cells,
                "total_cells": total_cells,
            })
        except Exception as e:
            logger.error(f"Exact comparison failed: {e}")
            self.result.error = f"Exact comparison failed: {e}"
    
    def _compare_approx(self, ref_df: DataFrame, alt_df: DataFrame, question: str, eval_guideline: Optional[str] = None, fuzzy_id_match: bool = True):
        """Approximate comparison: per-column strategies with structural metrics on aligned data.
        Phase 1 (Plan): Single LLM call to determine alignment config, column mapping, and row identifiers.
        Phase 2 (Align): Deterministic alignment — transpose, column rename, row match, dtype cast, sort.
            Uses plan's column_mapping and id_columns directly to align the dataframes.
        Phase 3 (Compare+Score): Strategy-specific comparison on aligned data.
            Uses plan's column_configs to determine comparison strategy for each column. Computes overall
            similarity as mean of per-column similarities.

        Args:
            ref_df: Reference dataframe
            alt_df: Candidate dataframe
            question: Task description for LLM planning
            eval_guideline: Optional evaluation guideline for LLM planning context
            fuzzy_id_match: When True, single-column row IDs left unmatched after exact/case-insensitive
                            comparison are additionally paired via LLM semantic matching. Default True —
                            a no-op (no LLM call) when all IDs already match.
        """
        try:
            # Phase 1: Plan (LLM call for alignment config, column mapping, and row identifiers)
            align_plan = self._get_align_plan(ref_df, alt_df, question, eval_guideline=eval_guideline)

            # Phase 2: Align (deterministic except for optional fuzzy_id_match LLM call)
            ref_aligned, alt_aligned, align_meta = self._align_dataframes(ref_df, alt_df, align_plan,
                                                                          fuzzy_id_match=fuzzy_id_match)
            
            # Phase 3: Generate comparison plan
            # Extract ID columns from alignment metadata, fallback to align_plan if not in metadata
            row_alignment = align_meta.get("row_alignment", {})
            id_columns = row_alignment.get("id_columns") or align_plan.get("id_columns")
            
            # Build column dtypes dict from aligned dataframes
            common_cols = sorted(set(ref_aligned.columns) & set(alt_aligned.columns))
            column_dtypes = {col: str(ref_aligned[col].dtype) for col in common_cols}
            
            # LLM call for comparison strategy, uses aligned dataframes
            comparison_plan = self._get_comparison_plan(ref_aligned, alt_aligned, question, id_columns=id_columns, column_dtypes=column_dtypes, eval_guideline=eval_guideline)
            
            # Add id_columns to comparison_plan for similarity calculation (needed to exclude ID columns)
            comparison_plan["id_columns"] = id_columns
            
            # Calculate similarities using per-column strategies on aligned dataframes
            metrics = self._calculate_similarities(ref_aligned, alt_aligned, comparison_plan)
            similarity = metrics.get("overall_similarity", 0.0)
            
            logger.info(f"Approximate comparison completed: similarity={similarity:.2%}")
            
            self.result.similarity = similarity
            self.result.details.update({
                "align_plan": align_plan,
                "comparison_plan": comparison_plan,
                "alignment": align_meta,
                "metrics": metrics,
            })
        except Exception as e:
            logger.error(f"Approximate comparison failed: {e}", exc_info=True)
            self.result.error = f"Approximate comparison failed: {e}"
    
    def _compare_summary(self, ref_df: DataFrame, alt_df: DataFrame):
        """Summary comparison: aggregate statistical properties without cell-level alignment.
        
        Compares high-level dataset properties: shape, column overlap, missing value rates,
        and distribution similarity (Hellinger distance on histograms for numeric,
        on frequency vectors for categorical). No alignment phase, no LLM calls.
        
        Args:
            ref_df: Reference dataframe
            alt_df: Candidate dataframe
        """
        try:
            # 1. Structural metrics
            shape = self._get_shape_diff(ref_df, alt_df)
            row_sim = float(rae_similarity([ref_df.shape[0]], [alt_df.shape[0]])[0])
            col_sim = float(rae_similarity([ref_df.shape[1]], [alt_df.shape[1]])[0])
            
            ref_cols, alt_cols = set(ref_df.columns), set(alt_df.columns)
            col_jaccard = jaccard_similarity(len(ref_cols), len(alt_cols), len(ref_cols & alt_cols))
            structural_score = float(np.mean([row_sim, col_sim, col_jaccard]))
            
            # 2. Missing value metrics (overall proportion RAE)
            ref_missing_rate = float(ref_df.isna().sum().sum() / ref_df.size) if ref_df.size > 0 else 0.0
            alt_missing_rate = float(alt_df.isna().sum().sum() / alt_df.size) if alt_df.size > 0 else 0.0
            missing_score = float(rae_similarity([ref_missing_rate], [alt_missing_rate])[0])
            
            # 3. Numeric distribution: pool all numeric values from each dataframe, compare via histogram Hellinger
            ref_numeric = ref_df.select_dtypes(include=[np.number]).values.ravel()
            alt_numeric = alt_df.select_dtypes(include=[np.number]).values.ravel()
            ref_numeric = ref_numeric[np.isfinite(ref_numeric)]
            alt_numeric = alt_numeric[np.isfinite(alt_numeric)]
            
            numeric_details = None
            if len(ref_numeric) > 0 and len(alt_numeric) > 0:
                combined = np.concatenate([ref_numeric, alt_numeric])
                if combined.min() < combined.max():
                    bins = np.linspace(combined.min(), combined.max(), 51)
                    ref_hist, _ = np.histogram(ref_numeric, bins=bins)
                    alt_hist, _ = np.histogram(alt_numeric, bins=bins)
                    h_dist = hellinger_distance(ref_hist, alt_hist)
                    numeric_score = 1.0 - h_dist if not np.isnan(h_dist) else 0.0
                else:
                    h_dist, numeric_score = 0.0, 1.0  # all same value
                numeric_details = {"score": numeric_score, "hellinger_distance": float(h_dist),
                                   "ref_count": len(ref_numeric), "alt_count": len(alt_numeric)}
            else:
                numeric_score = None
            
            # 4. Categorical distribution: pool all categorical values from each dataframe, compare via frequency Hellinger
            ref_cat = ref_df.select_dtypes(exclude=[np.number])
            alt_cat = alt_df.select_dtypes(exclude=[np.number])
            ref_cat_vals = ref_cat.values.ravel()
            alt_cat_vals = alt_cat.values.ravel()
            ref_cat_vals = pd.Series(ref_cat_vals).dropna()
            alt_cat_vals = pd.Series(alt_cat_vals).dropna()
            
            categorical_details = None
            if len(ref_cat_vals) > 0 and len(alt_cat_vals) > 0:
                ref_freq = ref_cat_vals.value_counts(normalize=True)
                alt_freq = alt_cat_vals.value_counts(normalize=True)
                all_cats = sorted(set(ref_freq.index) | set(alt_freq.index), key=str)
                ref_vec = np.array([ref_freq.get(c, 0.0) for c in all_cats])
                alt_vec = np.array([alt_freq.get(c, 0.0) for c in all_cats])
                h_dist = hellinger_distance(ref_vec, alt_vec)
                categorical_score = 1.0 - h_dist if not np.isnan(h_dist) else 0.0
                categorical_details = {"score": categorical_score, "hellinger_distance": float(h_dist),
                                       "ref_count": len(ref_cat_vals), "alt_count": len(alt_cat_vals),
                                       "ref_unique": int(ref_cat_vals.nunique()), "alt_unique": int(alt_cat_vals.nunique())}
            else:
                categorical_score = None
            
            # 5. Aggregate (only include components that exist)
            component_scores = [structural_score, missing_score]
            if numeric_score is not None:
                component_scores.append(numeric_score)
            if categorical_score is not None:
                component_scores.append(categorical_score)
            
            similarity = float(np.mean(component_scores))
            
            logger.info(f"Summary comparison completed: similarity={similarity:.2%}")
            self.result.similarity = similarity
            self.result.details.update({
                "structural": {"shape": shape, "row_sim": row_sim, "col_sim": col_sim,
                               "col_jaccard": col_jaccard, "score": structural_score},
                "missing": {"ref_missing_rate": ref_missing_rate, "alt_missing_rate": alt_missing_rate, "score": missing_score},
                "numeric_distribution": numeric_details,
                "categorical_distribution": categorical_details,
                "component_scores": {"structural": structural_score, "missing": missing_score,
                                     "numeric": numeric_score, "categorical": categorical_score},
            })
        except Exception as e:
            logger.error(f"Summary comparison failed: {e}", exc_info=True)
            self.result.error = f"Summary comparison failed: {e}"
    
    def _compare_semantic(self, ref_df: DataFrame, alt_df: DataFrame, question: str, eval_guideline: Optional[str] = None, **kwargs):
        """Semantic comparison: LLM judges data for equivalence.

        Sends data previews to LLM for holistic equivalence judgment.
        The LLM infers structural alignment (column mapping, row matching) and
        determines equivalence based on content, structure, and meaning.

        Args:
            ref_df: Reference dataframe
            alt_df: Candidate dataframe (original, not aligned)
            question: Task description for LLM analysis
            eval_guideline: Optional evaluation guideline for comparison context
        """
        max_rows = kwargs.get("max_rows", 1000)
        try:
            # Send data previews to LLM
            ref_preview = self._get_preview(ref_df, max_rows=max_rows)
            alt_preview = self._get_preview(alt_df, max_rows=max_rows)

            agent = create_llm_agent(
                model=self.model,
                system_prompt="You are a data comparison expert. Analyze tabular data files to determine their equivalence based on content, structure, and meaning.",
                response_format=EquivalenceResult
            )

            guideline_line = f"\nEvaluation guideline: {eval_guideline}" if eval_guideline else ""
            prompt = f"""Task: {question}{guideline_line}

Reference Data (shape: {ref_df.shape[0]} rows × {ref_df.shape[1]} columns, first {max_rows} rows is shown):
{ref_preview}

Candidate Data (shape: {alt_df.shape[0]} rows × {alt_df.shape[1]} columns, first {max_rows} rows is shown):
{alt_preview}

Please analyze whether these two tabular data files are equivalent. You should:
1. Infer structural alignment: Map columns between the two files (they may have different names but represent the same data), match rows (they may be in different orders or use different identifiers)
2. Compare data values: Are the values equivalent (accounting for reasonable numeric precision differences, column/row ordering, and naming variations)?
3. Assess equivalence: Consider missing values, additional rows/columns, and any critical differences

Provide:
- equivalence (0.0-1.0): How equivalent are these files? (1.0 = fully equivalent, 0.0 = not equivalent)
- recommendation (equivalent/partially equivalent/not equivalent/uncertain)
- confidence (0.0-1.0): How confident are you in this assessment?
- explanation: Detailed reasoning for your assessment, including how you aligned the structures

CRITICAL: You MUST provide numeric values for both equivalence and confidence. Do not omit these fields."""
            
            analysis = invoke_structured_agent(agent, prompt, EquivalenceResult)
            
            logger.info(
                "Semantic comparison completed: equivalence=%s, confidence=%s, recommendation=%s",
                f"{analysis.equivalence:.2%}" if analysis.equivalence is not None else "N/A",
                f"{analysis.confidence:.2%}" if analysis.confidence is not None else "N/A",
                analysis.recommendation
            )
            
            self.result.similarity = analysis.equivalence if analysis.equivalence is not None else 0.0
            self.result.details.update({
                "recommendation": analysis.recommendation,
                "confidence": analysis.confidence,
                "explanation": analysis.explanation,
            })
        except Exception as e:
            logger.error(f"Semantic comparison failed: {e}", exc_info=True)
            self.result.error = f"Semantic comparison failed: {e}"
    
