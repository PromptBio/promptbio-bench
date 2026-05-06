"""R data format handler for R objects saved as .rds and .RData/.rda files.

Strategy:
- exact: All values identical (within float tolerance). Dimensions, names must match perfectly.
  For matrix/dataframe: shape, row/col names, all values must match. Similarity is 1.0 or 0.0.
  For list: same keys/length, all values recursively identical. Similarity is 1.0 or 0.0.
  For vector: same length, all values match. Similarity is 1.0 or 0.0.
- approximate: Partial overlap allowed, element-wise matching within tolerance.
  For matrix/dataframe: align to common rows/cols, fraction of elements matching within tolerance.
  For list: fraction of matching keys with matching values.
  For vector: fraction of elements matching within tolerance.
- summary: Statistical summary comparison (no element-wise alignment required).
  For matrix/dataframe: shape similarity, per-column distribution correlation.
  For list: structure similarity (key overlap, value type match).
  For vector: length similarity, distribution comparison (mean, std, quantile).

Object types supported:
- Seurat objects (routes to AnnDataHandler, which converts to AnnData)
- SingleCellExperiment objects (routes to AnnDataHandler)
- Matrices (numeric, character, logical)
- DataFrames
- Lists (including named lists)
- Vectors

Required cmdline tools:
- Rscript
"""

import logging
import os
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import pyreadr

from tools.base import FormatHandler, resolve_r_executable
from tools.results import ValidationResult, ComparisonResult
from tools.anndata import AnnDataHandler
from utils.format import detect_file_signature
from utils.metrics import pearson_correlation, min_max_similarity

logger = logging.getLogger(__name__)


class RDataHandler(FormatHandler):
    """Handler for R objects saved as .rds and .RData/.rda files.
    
    Supports:
    - RDS files (.rds): Single R object format
    - Rdata files (.RData, .rda): Multiple R objects format
    
    Object types supported:
    - Seurat objects (routes to AnnDataHandler, which converts to AnnData)
    - SingleCellExperiment objects (routes to AnnDataHandler)
    - Matrices (numeric, character, logical)
    - DataFrames
    - Lists (including named lists)
    - Vectors
    
    For Rdata files with multiple objects, compares objects by name.
    """
    
    def __init__(self):
        super().__init__("rdata")
        self.supported_formats = ['rds', 'rdata', 'rda']
        self.supported_strategies = ["exact", "approximate", "summary"]
        self.anndata_handler = AnnDataHandler()
    
    # ============================================================================
    # Public API Methods
    # ============================================================================
    
    def validate(self, file_path: str, **kwargs) -> ValidationResult:
        """Validate RDS or Rdata file format.
        non-empty check is performed by file size, later can be updated by reading content if needed
        
        Args:
            file_path: Path to RDS or Rdata file
            **kwargs: optional parameters (e.g., min_size)
            - min_size: Minimum size of the file in bytes, default is 0
        """
        is_exists = self._is_file_exists(file_path)
        is_nonempty = self._is_file_nonempty(file_path, min_size=kwargs.get("min_size", 0))
        
        signature = detect_file_signature(file_path, fallback=True)
        file_ext = file_path.lower()
        is_format_ok = (
            signature.format in self.supported_formats
            or file_ext.endswith('.rds')
            or file_ext.endswith('.rdata')
            or file_ext.endswith('.rda')
        )

        result = ValidationResult(
            exists=is_exists, non_empty=is_nonempty, format_ok=is_format_ok,
            parseable=False, file_path=file_path, signature=signature,
            details={}, error=None,
        )

        if not is_exists:
            logger.error(f"R data file not found: {file_path}")
            result.error = "File not found"
            return result
        
        if not is_nonempty:
            logger.warning(f"R data file is empty: {file_path}")
            result.error = "File is empty"
            return result
        
        if not is_format_ok:
            logger.error(f"Not an R data file: detected={signature.format}")
            result.error = f"Not an R data file: detected={signature.format}"
            return result
        
        file_type = signature.format  # Use format detection from utils/format.py
        result.details['file_type'] = file_type
        
        try:
            if file_type == 'rds':
                obj_type, info = self._get_r_object_type_and_info(file_path, is_rds=True)
                if obj_type is None:
                    result.error = "Failed to detect RDS object type"
                    return result
            
                result.details.update({"format": file_type, "object_type": obj_type})
            
                if obj_type in ("seurat", "sce"):
                    anndata_result = self.anndata_handler.validate(file_path, **kwargs)
                    if anndata_result.is_valid:
                        result.parseable = anndata_result.parseable
                        result.non_empty = anndata_result.non_empty
                        result.details.update(anndata_result.details)
                        result.details['original_format'] = obj_type
                    else:
                        result.error = anndata_result.error or "Failed to validate Seurat/SCE object"
                else:
                    if info:
                        result.details.update(info)
                    result.parseable = True
                    # Check if object is non-empty using n_row and n_col (for matrix/dataframe) or length (for vector)
                    if obj_type in ("matrix", "dataframe"):
                        n_row = result.details.get('n_row', 0)
                        n_col = result.details.get('n_col', 0)
                        result.non_empty = n_row > 0 and n_col > 0
                    else:
                        result.non_empty = result.details.get('length', 0) > 0 if 'length' in result.details else True
                    if not result.non_empty:
                        result.error = "R object is empty"
            else:
                # Get info for ALL objects in Rdata file
                _, all_objects_info = self._get_r_object_type_and_info(file_path, is_rds=False)
                if all_objects_info is None:
                    result.error = "Failed to get Rdata objects information"
                    return result
                
                if len(all_objects_info) == 0:
                    result.error = "Rdata file contains no objects"
                    return result
                
                obj_names = list(all_objects_info.keys())
                result.details['object_names'] = obj_names
                result.details['num_objects'] = len(obj_names)
                result.details['objects_info'] = all_objects_info
                
                # Get first object type for backward compatibility
                first_obj = all_objects_info[obj_names[0]] if obj_names else {}
                first_obj_type = first_obj.get("type")
                if first_obj_type:
                    result.details['first_object_type'] = first_obj_type
                
                result.parseable = True
                result.non_empty = len(obj_names) > 0
        except Exception as e:
            logger.error(f"R data validation failed: {e}")
            result.error = f"R data validation failed: {e}"
        return result
    
    def compare(self, reference_path: str, candidate_path: str, strategy: str = "summary", **kwargs) -> ComparisonResult:
        """Compare RDS or Rdata files.
        
        Automatically detects object type and routes to strategy-specific comparison.
        For Rdata files with multiple objects, compares objects by name.
        
        Supported strategies:
        - exact: All values identical (within float tolerance). Similarity is 1.0 or 0.0.
        - approximate: Element-wise matching within tolerance on common rows/cols/keys.
        - summary: Statistical summary comparison (shape, distribution, correlation).
        
        Args:
            reference_path: Path to reference RDS or Rdata file
            candidate_path: Path to candidate RDS or Rdata file
            strategy: Comparison strategy (exact, approximate, summary)
            **kwargs: Strategy-specific parameters
            - tolerance: Tolerance for numeric comparisons (default: 1e-6 for exact, 1e-3 for approximate)
        """
        self.result = ComparisonResult(
            reference_path=reference_path, candidate_path=candidate_path,
            strategy=strategy, similarity=np.nan, details={}, error=None,
        )
        
        ref_validation = self.validate(reference_path, **kwargs)
        alt_validation = self.validate(candidate_path, **kwargs)

        try:
            if ref_validation.is_valid and alt_validation.is_valid:
                if strategy not in self.supported_strategies:
                    logger.warning(f"Unknown strategy '{strategy}', fall back to summary")
                    strategy = "summary"

                self._compare_by_strategy(ref_validation, alt_validation, strategy, **kwargs)
        except Exception as e:
            logger.error(f"R data comparison failed: {e}")
            self.result.error = f"R data comparison failed: {e}"
        return self._finalize_output(ref_validation, alt_validation, self.result)
    
    # ============================================================================
    # File I/O and Loading Methods
    # ============================================================================
    
    def _run_r_script(self, r_script: str, timeout: int = 300) -> Tuple[bool, str, str]:
        """Run R script and return (success, stdout, stderr).
        
        Used as fallback for operations that pyreadr cannot handle
        (e.g., Seurat/SCE type detection, R list extraction via jsonlite).
        """
        try:
            result = subprocess.run(
                [resolve_r_executable("R"), "--slave", "--no-restore", "--no-save", "-e", r_script],
                capture_output=True, text=True, timeout=timeout, check=False,
            )
            return result.returncode == 0, result.stdout, result.stderr
        except FileNotFoundError:
            logger.error("R not found. Please install R to use RDS handler.")
            return False, "", "R not found"
        except subprocess.TimeoutExpired:
            logger.error(f"R script timed out after {timeout} seconds")
            return False, "", "Timeout"
        except Exception as e:
            logger.error(f"Failed to run R script: {e}")
            return False, "", str(e)
    
    def _load_file(self, file_path: str, obj_type: Optional[str] = None,
                   file_type: Optional[str] = None, obj_name: Optional[str] = None):
        """Load R data file and optionally extract/convert object by type.
        
        Args:
            file_path: Path to RDS or Rdata file
            obj_type: Optional object type for extraction ("matrix", "dataframe", "list", "vector")
            file_type: Optional file type ("rds" or "rdata") - required if obj_type is provided
            obj_name: Optional object name (required for Rdata files if obj_type is provided)
        
        Returns:
            - If obj_type is None: OrderedDict of {name: DataFrame} for Rdata, {None: DataFrame} for RDS
            - If obj_type is provided: Extracted object (pd.DataFrame, dict, np.ndarray) or None
        """
        try:
            loaded_data = pyreadr.read_r(file_path)
        except Exception as e:
            logger.error(f"Failed to load R data file {file_path}: {e}")
            return None
        
        # If no extraction requested, return raw loaded data
        if obj_type is None:
            return loaded_data
        
        if loaded_data is None:
            return None
        
        # Get object by key
        key = None if file_type == "rds" else obj_name
        obj = loaded_data.get(key)
        if obj is None and key is not None:
            # pyreadr may key slightly differently; search by string match
            for k, v in loaded_data.items():
                if str(k) == key:
                    obj = v
                    break
        
        if obj is None:
            return None

        # Convert based on object type
        if obj_type in ("matrix", "dataframe"):
            return obj if isinstance(obj, pd.DataFrame) else None
        
        elif obj_type == "vector":
            if isinstance(obj, pd.DataFrame):
                return obj.iloc[:, 0].values if obj.shape[1] >= 1 else obj.values.flatten()
            return np.asarray(obj)
        
        elif obj_type == "list":
            # Check if pyreadr returned a dict/list that we can use
            if isinstance(obj, dict):
                return obj
            elif isinstance(obj, list):
                return {i: v for i, v in enumerate(obj)}
            elif isinstance(obj, pd.DataFrame):
                # For simple lists, pyreadr might return a DataFrame
                try:
                    if obj.shape[1] == 1:
                        return {i: val for i, val in enumerate(obj.iloc[:, 0].values)}
                except Exception:
                    pass
            
            # Fallback to R subprocess with jsonlite for complex lists
            escaped_path = file_path.replace("\\", "\\\\").replace('"', '\\"')
            fd, temp_json = tempfile.mkstemp(suffix='.json', prefix='r_list_')
            os.close(fd)
            try:
                if obj_name is None:
                    # RDS file — single object
                    r_script = f'''
suppressPackageStartupMessages(library(jsonlite))
tryCatch({{
    obj <- readRDS("{escaped_path}")
    json_str <- toJSON(obj, auto_unbox=TRUE, null="null")
    write(json_str, "{temp_json}")
}}, error=function(e) {{ cat("ERROR:", conditionMessage(e), "\\n"); quit(status=1) }})
'''
                else:
                    # Rdata file — named object
                    escaped_name = obj_name.replace("\\", "\\\\").replace('"', '\\"')
                    r_script = f'''
suppressPackageStartupMessages(library(jsonlite))
tryCatch({{
    env <- new.env()
    load("{escaped_path}", envir=env)
    obj <- get("{escaped_name}", envir=env)
    json_str <- toJSON(obj, auto_unbox=TRUE, null="null")
    write(json_str, "{temp_json}")
}}, error=function(e) {{ cat("ERROR:", conditionMessage(e), "\\n"); quit(status=1) }})
'''
                success, _, stderr = self._run_r_script(r_script)
                if not success:
                    logger.error(f"Failed to extract list via R: {stderr}")
                    return None
                with open(temp_json, 'r') as f:
                    return json.load(f)
            finally:
                self._cleanup_file(temp_json)
        
        else:
            logger.error(f"Unknown object type: {obj_type}")
            return None

    def _get_r_object_type_and_info(self, file_path: str, is_rds: bool = True,
                                    obj_name: Optional[str] = None) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Get R object type and information using the merged R script.
        
        For Rdata files, returns info about ALL objects as a dict.
        For RDS files, returns info about the single object (keyed by filename).
        
        Args:
            file_path: Path to RDS or Rdata file
            is_rds: True for RDS files, False for Rdata files
            obj_name: Object name (optional - if provided, extracts just that object from Rdata)
        
        Returns:
            For RDS: (object_type, info_dict) - uses filename as key
            For Rdata: (None, dict_of_all_objects) or (object_type, info_dict) if obj_name provided
        """
        script_dir = Path(__file__).parent / "helper"
        r_script_path = script_dir / "get_r_object_info.R"
        
        if not r_script_path.exists():
            logger.error(f"R script not found: {r_script_path}")
            return None, None
        
        cmd = [resolve_r_executable("Rscript"), str(r_script_path), "--input", str(file_path)]
        
        valid_types = {"seurat", "sce", "matrix", "dataframe", "list", "vector"}
        error_types = {"unknown", "other"}
        
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300, check=False
            )
            
            if result.returncode != 0:
                logger.error(f"R script failed (code {result.returncode}): {result.stderr}")
                return None, None
            
            # Parse JSON output
            all_objects = json.loads(result.stdout.strip())
            
            if is_rds:
                # RDS: single object, key is filename without extension
                # Get the first (and only) key
                key = list(all_objects.keys())[0] if all_objects else None
                if key is None:
                    return None, None
                
                obj_info = all_objects[key]
                obj_type = obj_info.get("type")
                
                if obj_type in error_types:
                    logger.warning(f"Unknown RDS object type in {file_path}")
                    return None, None
                
                # Extract info (everything except type)
                info = {k: v for k, v in obj_info.items() if k != "type"}
                return obj_type if obj_type in valid_types else None, info if info else None
            else:
                # Rdata: multiple objects
                if obj_name:
                    # Extract specific object
                    obj_info = all_objects.get(obj_name, {})
                    obj_type = obj_info.get("type")
                    
                    if obj_type in error_types:
                        return None, None
                    
                    # Extract info (everything except type)
                    info = {k: v for k, v in obj_info.items() if k != "type"}
                    return obj_type if obj_type in valid_types else None, info if info else None
                else:
                    # Return all objects info
                    return None, all_objects
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON output: {e}")
            return None, None
        except subprocess.TimeoutExpired:
            logger.error(f"R script timed out: {file_path}")
            return None, None
        except Exception as e:
            logger.error(f"Failed to run R script: {e}")
            return None, None

    def _get_r_object_type(self, file_path: str, is_rds: bool = True, 
                           obj_name: Optional[str] = None) -> Optional[str]:
        """Get R object type using the merged R script.
        
        Uses tools/helper/get_r_object_info.R for detection.
        
        Args:
            file_path: Path to RDS or Rdata file
            is_rds: True for RDS files, False for Rdata files
            obj_name: Object name (optional for Rdata files - if not provided, returns None)
        
        Returns:
            Object type string: "seurat", "sce", "matrix", "dataframe", "list", "vector", or None
        """
        obj_type, _ = self._get_r_object_type_and_info(file_path, is_rds, obj_name)
        return obj_type
        
    # ============================================================================
    # Statistics and Info Methods
    # ============================================================================
    
    def _get_r_object_info(self, file_path: str, obj_type: str) -> Optional[Dict[str, Any]]:
        """Get basic information about RDS object.

        Tries pyreadr first for matrix/dataframe/vector (faster), then falls back
        to R script for lists or when pyreadr fails. Uses tools/helper/get_r_object_info.R.
        
        Args:
            file_path: Path to RDS file
            obj_type: Object type ("list", "matrix", "dataframe", "vector")
        
        Returns:
            Dictionary with n_row, n_col, length, mode, columns, etc., or None
        """
        # Always use R script for lists (pyreadr doesn't handle them well)
        if obj_type == "list":
            _, info = self._get_r_object_type_and_info(file_path, is_rds=True)
            return info
        
        # Try pyreadr first for matrix/dataframe/vector (faster)
        try:
            result = self._load_file(file_path)
            if result is None:
                _, info = self._get_r_object_type_and_info(file_path, is_rds=True)
                return info
            
            df = result.get(None)
            if df is None:
                _, info = self._get_r_object_type_and_info(file_path, is_rds=True)
                return info
            
            # Extract info from DataFrame
            if obj_type in ("matrix", "dataframe"):
                info: Dict[str, Any] = {
                    'n_row': df.shape[0],
                    'n_col': df.shape[1],
                }
                if obj_type == "matrix":
                    info['mode'] = str(df.dtypes.iloc[0]) if len(df.columns) > 0 else 'unknown'
                elif obj_type == "dataframe":
                    info['columns'] = list(df.columns)
                return info
            
            elif obj_type == "vector":
                # For vectors, pyreadr loads as single-column DataFrame
                mode_val = 'unknown'
                if df.shape[1] > 0 and len(df) > 0:
                    # Get dtype from the first column
                    dtype = df.dtypes.iloc[0]
                    # Map pandas dtypes to R-like mode names
                    if 'int' in str(dtype):
                        mode_val = 'integer'
                    elif 'float' in str(dtype):
                        mode_val = 'numeric'
                    elif 'bool' in str(dtype):
                        mode_val = 'logical'
                    elif 'object' in str(dtype) or 'string' in str(dtype):
                        mode_val = 'character'
                    else:
                        mode_val = str(dtype)
                elif len(df) == 0:
                    # Empty vector - fall back to R script for accurate mode
                    _, info = self._get_r_object_type_and_info(file_path, is_rds=True)
                    return info
                return {
                    'length': len(df),
                    'mode': mode_val,
                }
        except Exception as e:
            logger.warning(f"pyreadr failed to get info, falling back to R script: {e}")
        
        # Fallback to R script
        _, info = self._get_r_object_type_and_info(file_path, is_rds=True)
        return info
    
    
    # ============================================================================
    # Per-Type Comparison Helpers — Matrix
    # ============================================================================
    
    def _compare_matrix_exact(self, ref_df: pd.DataFrame, alt_df: pd.DataFrame,
                              tolerance: float = 1e-6) -> Tuple[float, Dict[str, Any]]:
        """Exact comparison of two matrices (as DataFrames).

        Requires identical shape, row names, column names, and all values within tolerance.

        Returns:
            Tuple of (similarity, details). similarity is 1.0 or 0.0.
        """
        details: Dict[str, Any] = {
            "ref_shape": list(ref_df.shape),
            "alt_shape": list(alt_df.shape),
            "tolerance": tolerance,
        }

        shape_match = ref_df.shape == alt_df.shape
        details["shape_match"] = shape_match
        if not shape_match:
            return 0.0, details

        row_match = list(ref_df.index) == list(alt_df.index)
        col_match = list(ref_df.columns) == list(alt_df.columns)
        details["row_names_match"] = row_match
        details["col_names_match"] = col_match
        if not (row_match and col_match):
            return 0.0, details

        try:
            ref_vals = ref_df.values.astype(np.float64)
            alt_vals = alt_df.values.astype(np.float64)
            diff = np.abs(ref_vals - alt_vals)
            max_diff = float(np.nanmax(diff))
            mean_diff = float(np.nanmean(diff))
            values_match = max_diff <= tolerance
        except (ValueError, TypeError):
            values_match = ref_df.equals(alt_df)
            max_diff = 0.0 if values_match else np.nan
            mean_diff = 0.0 if values_match else np.nan

        details["max_diff"] = max_diff
        details["mean_diff"] = mean_diff
        details["values_match"] = values_match
        return (1.0 if values_match else 0.0), details

    def _compare_matrix_approx(self, ref_df: pd.DataFrame, alt_df: pd.DataFrame,
                               tolerance: float = 1e-3) -> Tuple[float, Dict[str, Any]]:
        """Approximate comparison of two matrices (as DataFrames).

        Aligns to common rows and columns. Calculates fraction of elements matching
        within tolerance, weighted by overlap coverage.

        Similarity = 0.7 * element_match_fraction + 0.3 * overlap_coverage

        Returns:
            Tuple of (similarity, details). similarity is continuous [0.0, 1.0].
        """
        details: Dict[str, Any] = {
            "ref_shape": list(ref_df.shape),
            "alt_shape": list(alt_df.shape),
            "tolerance": tolerance,
        }

        common_rows = ref_df.index.intersection(alt_df.index)
        common_cols = ref_df.columns.intersection(alt_df.columns)
        details["num_common_rows"] = len(common_rows)
        details["num_common_cols"] = len(common_cols)
        
        if len(common_rows) == 0 or len(common_cols) == 0:
            details["error"] = "No common rows or columns found"
            return 0.0, details
        
        ref_aligned = ref_df.loc[common_rows, common_cols]
        alt_aligned = alt_df.loc[common_rows, common_cols]
        
        try:
            ref_vals = ref_aligned.values.astype(np.float64)
            alt_vals = alt_aligned.values.astype(np.float64)
            diff = np.abs(ref_vals - alt_vals)
            total_elements = diff.size
            matching_elements = int(np.sum(diff <= tolerance))
            element_match = matching_elements / total_elements if total_elements > 0 else 0.0
            max_diff = float(np.nanmax(diff))
            mean_diff = float(np.nanmean(diff))
        except (ValueError, TypeError):
            match_mask = ref_aligned == alt_aligned
            total_elements = match_mask.size
            matching_elements = int(match_mask.sum().sum())
            element_match = matching_elements / total_elements if total_elements > 0 else 0.0
            max_diff = np.nan
            mean_diff = np.nan

        max_rows = max(len(ref_df.index), len(alt_df.index))
        max_cols = max(len(ref_df.columns), len(alt_df.columns))
        row_overlap = len(common_rows) / max_rows if max_rows > 0 else 0.0
        col_overlap = len(common_cols) / max_cols if max_cols > 0 else 0.0
        overlap_coverage = (row_overlap + col_overlap) / 2.0

        similarity = 0.7 * element_match + 0.3 * overlap_coverage

        details.update({
            "matching_elements": matching_elements,
            "total_elements": total_elements,
            "element_match_fraction": float(element_match),
            "row_overlap": float(row_overlap),
            "col_overlap": float(col_overlap),
            "overlap_coverage": float(overlap_coverage),
            "max_diff": max_diff,
            "mean_diff": mean_diff,
        })
        return float(similarity), details

    def _compare_matrix_summary(self, ref_df: pd.DataFrame,
                                alt_df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
        """Summary comparison of two matrices (as DataFrames).

        Compares shape similarity, per-column mean/std correlation, and overall distribution.

        Returns:
            Tuple of (similarity, details). similarity is continuous [0.0, 1.0].
        """
        details: Dict[str, Any] = {
            "ref_shape": list(ref_df.shape),
            "alt_shape": list(alt_df.shape),
        }
        similarities: List[float] = []

        # 1. Shape similarity
        row_sim = min_max_similarity(ref_df.shape[0], alt_df.shape[0])
        col_sim = min_max_similarity(ref_df.shape[1], alt_df.shape[1])
        shape_sim = (row_sim + col_sim) / 2.0
        similarities.append(shape_sim)
        details["shape_similarity"] = float(shape_sim)

        # 2. Per-column distribution correlation on common columns
        common_cols = ref_df.columns.intersection(alt_df.columns)
        if len(common_cols) > 0:
            try:
                ref_means = ref_df[common_cols].mean().values.astype(np.float64)
                alt_means = alt_df[common_cols].mean().values.astype(np.float64)
                ref_stds = ref_df[common_cols].std().values.astype(np.float64)
                alt_stds = alt_df[common_cols].std().values.astype(np.float64)

                mean_corr = pearson_correlation(ref_means, alt_means)
                mean_corr = max(0.0, float(mean_corr)) if not np.isnan(mean_corr) else 0.0
                similarities.append(mean_corr)
                details["mean_correlation"] = mean_corr

                std_corr = pearson_correlation(ref_stds, alt_stds)
                std_corr = max(0.0, float(std_corr)) if not np.isnan(std_corr) else 0.0
                similarities.append(std_corr)
                details["std_correlation"] = std_corr
                details["num_common_cols"] = len(common_cols)
            except (ValueError, TypeError):
                col_sims = []
                for col in common_cols:
                    ref_vals = set(ref_df[col].dropna().astype(str).values)
                    alt_vals = set(alt_df[col].dropna().astype(str).values)
                    union = len(ref_vals | alt_vals)
                    intersection = len(ref_vals & alt_vals)
                    col_sims.append(intersection / union if union > 0 else 0.0)
                if col_sims:
                    dist_sim = float(np.mean(col_sims))
                    similarities.append(dist_sim)
                    details["value_overlap_similarity"] = dist_sim

        similarity = float(np.mean(similarities)) if similarities else 0.0
        return similarity, details

    # ============================================================================
    # Per-Type Comparison Helpers — DataFrame
    # ============================================================================

    def _compare_dataframe_exact(self, ref_df: pd.DataFrame, alt_df: pd.DataFrame,
                                 tolerance: float = 1e-6) -> Tuple[float, Dict[str, Any]]:
        """Exact comparison of two dataframes.

        Requires identical shape, row/col names, and all values matching.
        Numeric columns use tolerance; categorical columns require exact string match.

        Returns:
            Tuple of (similarity, details). similarity is 1.0 or 0.0.
        """
        details: Dict[str, Any] = {
            "ref_shape": list(ref_df.shape),
            "alt_shape": list(alt_df.shape),
            "tolerance": tolerance,
        }

        shape_match = ref_df.shape == alt_df.shape
        details["shape_match"] = shape_match
        if not shape_match:
            return 0.0, details

        row_match = list(ref_df.index) == list(alt_df.index)
        col_match = list(ref_df.columns) == list(alt_df.columns)
        details["row_names_match"] = row_match
        details["col_names_match"] = col_match
        if not (row_match and col_match):
            return 0.0, details

        all_match = True
        col_results: Dict[str, bool] = {}
        for col in ref_df.columns:
            ref_col = ref_df[col]
            alt_col = alt_df[col]
            if ref_col.dtype in [np.float64, np.float32, np.int64, np.int32] or \
               alt_col.dtype in [np.float64, np.float32, np.int64, np.int32]:
                try:
                    is_match = np.allclose(
                        ref_col.values.astype(np.float64),
                        alt_col.values.astype(np.float64),
                        rtol=tolerance, atol=tolerance, equal_nan=True,
                    )
                except (ValueError, TypeError):
                    is_match = ref_col.equals(alt_col)
            else:
                is_match = ref_col.equals(alt_col)
            col_results[col] = is_match
            if not is_match:
                all_match = False

        details["column_match"] = col_results
        details["all_match"] = all_match
        return (1.0 if all_match else 0.0), details

    def _compare_dataframe_approx(self, ref_df: pd.DataFrame, alt_df: pd.DataFrame,
                                  tolerance: float = 1e-3) -> Tuple[float, Dict[str, Any]]:
        """Approximate comparison of two dataframes.

        Aligns to common rows and columns. For each column:
        - Numeric: fraction of values within tolerance
        - Categorical: fraction of exact matches

        Similarity = 0.7 * col_match_mean + 0.3 * overlap_coverage

        Returns:
            Tuple of (similarity, details). similarity is continuous [0.0, 1.0].
        """
        details: Dict[str, Any] = {
            "ref_shape": list(ref_df.shape),
            "alt_shape": list(alt_df.shape),
            "tolerance": tolerance,
        }

        common_rows = ref_df.index.intersection(alt_df.index)
        common_cols = ref_df.columns.intersection(alt_df.columns)
        details["num_common_rows"] = len(common_rows)
        details["num_common_cols"] = len(common_cols)
        
        if len(common_rows) == 0 or len(common_cols) == 0:
            details["error"] = "No common rows or columns found"
            return 0.0, details
        
        ref_aligned = ref_df.loc[common_rows, common_cols]
        alt_aligned = alt_df.loc[common_rows, common_cols]
        
        col_similarities: List[float] = []
        for col in common_cols:
            ref_col = ref_aligned[col]
            alt_col = alt_aligned[col]
            if ref_col.dtype in [np.float64, np.float32, np.int64, np.int32] or \
               alt_col.dtype in [np.float64, np.float32, np.int64, np.int32]:
                try:
                    ref_vals = ref_col.values.astype(np.float64)
                    alt_vals = alt_col.values.astype(np.float64)
                    n_total = len(ref_vals)
                    n_match = int(
                        np.sum(np.abs(ref_vals - alt_vals) <= tolerance)
                        + np.sum(np.isnan(ref_vals) & np.isnan(alt_vals))
                    )
                    col_similarities.append(n_match / n_total if n_total > 0 else 0.0)
                except (ValueError, TypeError):
                    col_similarities.append(1.0 if ref_col.equals(alt_col) else 0.0)
            else:
                n_total = len(ref_col)
                n_match = int((ref_col == alt_col).sum())
                col_similarities.append(n_match / n_total if n_total > 0 else 0.0)

        col_match_mean = float(np.mean(col_similarities)) if col_similarities else 0.0

        max_rows = max(len(ref_df.index), len(alt_df.index))
        max_cols = max(len(ref_df.columns), len(alt_df.columns))
        row_overlap = len(common_rows) / max_rows if max_rows > 0 else 0.0
        col_overlap = len(common_cols) / max_cols if max_cols > 0 else 0.0
        overlap_coverage = (row_overlap + col_overlap) / 2.0

        similarity = 0.7 * col_match_mean + 0.3 * overlap_coverage
        details.update({
            "col_match_mean": col_match_mean,
            "row_overlap": float(row_overlap),
            "col_overlap": float(col_overlap),
            "overlap_coverage": float(overlap_coverage),
        })
        return float(similarity), details

    def _compare_dataframe_summary(self, ref_df: pd.DataFrame,
                                   alt_df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
        """Summary comparison of two dataframes.

        Compares shape similarity, column overlap, and per-column distribution similarity.

        Returns:
            Tuple of (similarity, details). similarity is continuous [0.0, 1.0].
        """
        details: Dict[str, Any] = {
            "ref_shape": list(ref_df.shape),
            "alt_shape": list(alt_df.shape),
        }
        similarities: List[float] = []

        # 1. Shape similarity
        row_sim = min_max_similarity(ref_df.shape[0], alt_df.shape[0])
        col_sim = min_max_similarity(ref_df.shape[1], alt_df.shape[1])
        shape_sim = (row_sim + col_sim) / 2.0
        similarities.append(shape_sim)
        details["shape_similarity"] = float(shape_sim)

        # 2. Column overlap
        common_cols = ref_df.columns.intersection(alt_df.columns)
        max_cols = max(len(ref_df.columns), len(alt_df.columns))
        col_overlap_val = len(common_cols) / max_cols if max_cols > 0 else 0.0
        similarities.append(col_overlap_val)
        details["col_overlap"] = float(col_overlap_val)
        details["num_common_cols"] = len(common_cols)

        # 3. Per-column distribution similarity on common columns
        if len(common_cols) > 0:
            col_sims: List[float] = []
            for col in common_cols:
                ref_col = ref_df[col]
                alt_col = alt_df[col]
                if ref_col.dtype in [np.float64, np.float32, np.int64, np.int32] and \
                   alt_col.dtype in [np.float64, np.float32, np.int64, np.int32]:
                    try:
                        ref_vals = ref_col.dropna().values.astype(np.float64)
                        alt_vals = alt_col.dropna().values.astype(np.float64)
                        if len(ref_vals) > 0 and len(alt_vals) > 0:
                            mean_sim = min_max_similarity(float(np.mean(ref_vals)), float(np.mean(alt_vals)))
                            std_sim = min_max_similarity(float(np.std(ref_vals)), float(np.std(alt_vals)))
                            col_sims.append((mean_sim + std_sim) / 2.0)
                        else:
                            col_sims.append(0.0)
                    except (ValueError, TypeError):
                        col_sims.append(0.0)
                else:
                    ref_vals_set = set(ref_col.dropna().astype(str).values)
                    alt_vals_set = set(alt_col.dropna().astype(str).values)
                    union = len(ref_vals_set | alt_vals_set)
                    intersection = len(ref_vals_set & alt_vals_set)
                    col_sims.append(intersection / union if union > 0 else 0.0)
            if col_sims:
                dist_sim = float(np.mean(col_sims))
                similarities.append(dist_sim)
                details["column_distribution_similarity"] = dist_sim

        similarity = float(np.mean(similarities)) if similarities else 0.0
        return similarity, details

    # ============================================================================
    # Per-Type Comparison Helpers — List
    # ============================================================================

    def _compare_list_exact(self, ref_list, alt_list,
                            tolerance: float = 1e-6) -> Tuple[float, Dict[str, Any]]:
        """Exact comparison of two lists (Python dicts or lists from R).

        Named lists (dicts): same keys, all values recursively identical.
        Unnamed lists (lists): same length, all elements identical.

        Returns:
            Tuple of (similarity, details). similarity is 1.0 or 0.0.
        """
        details: Dict[str, Any] = {}

        if type(ref_list) != type(alt_list):
            details["type_mismatch"] = True
            return 0.0, details

        if isinstance(ref_list, dict) and isinstance(alt_list, dict):
            ref_keys = set(ref_list.keys())
            alt_keys = set(alt_list.keys())
            keys_match = ref_keys == alt_keys
            details["keys_match"] = keys_match
            details["ref_num_keys"] = len(ref_keys)
            details["alt_num_keys"] = len(alt_keys)
            if not keys_match:
                return 0.0, details
            all_match = all(
                self._values_equal(ref_list[k], alt_list[k], tolerance) for k in ref_keys
            )
            details["values_match"] = all_match
            return (1.0 if all_match else 0.0), details

        if isinstance(ref_list, list) and isinstance(alt_list, list):
            length_match = len(ref_list) == len(alt_list)
            details["length_match"] = length_match
            details["ref_length"] = len(ref_list)
            details["alt_length"] = len(alt_list)
            if not length_match:
                return 0.0, details
            all_match = all(
                self._values_equal(r, a, tolerance) for r, a in zip(ref_list, alt_list)
            )
            details["values_match"] = all_match
            return (1.0 if all_match else 0.0), details

        return (1.0 if ref_list == alt_list else 0.0), details

    def _compare_list_approx(self, ref_list, alt_list,
                             tolerance: float = 1e-3) -> Tuple[float, Dict[str, Any]]:
        """Approximate comparison of two lists.

        Named lists: fraction of common keys with matching values.
        Unnamed lists: fraction of positional matches.

        Returns:
            Tuple of (similarity, details). similarity is continuous [0.0, 1.0].
        """
        details: Dict[str, Any] = {}

        if isinstance(ref_list, dict) and isinstance(alt_list, dict):
            ref_keys = set(ref_list.keys())
            alt_keys = set(alt_list.keys())
            common_keys = ref_keys & alt_keys
            all_keys = ref_keys | alt_keys
            details["num_common_keys"] = len(common_keys)
            details["num_total_keys"] = len(all_keys)

            if len(all_keys) == 0:
                return 0.0, details

            key_overlap = len(common_keys) / len(all_keys)
            value_matches = sum(
                1 for k in common_keys
                if self._values_equal(ref_list[k], alt_list[k], tolerance)
            )
            value_match_frac = value_matches / len(common_keys) if len(common_keys) > 0 else 0.0
            similarity = 0.7 * value_match_frac + 0.3 * key_overlap
            details.update({
                "key_overlap": float(key_overlap),
                "value_match_fraction": float(value_match_frac),
                "value_matches": value_matches,
            })
            return float(similarity), details

        if isinstance(ref_list, list) and isinstance(alt_list, list):
            max_len = max(len(ref_list), len(alt_list))
            if max_len == 0:
                return 1.0, details
            min_len = min(len(ref_list), len(alt_list))
            matches = sum(
                1 for i in range(min_len)
                if self._values_equal(ref_list[i], alt_list[i], tolerance)
            )
            similarity = matches / max_len
            details.update({
                "ref_length": len(ref_list),
                "alt_length": len(alt_list),
                "matches": matches,
            })
            return float(similarity), details

        sim = 1.0 if self._values_equal(ref_list, alt_list, tolerance) else 0.0
        return sim, details

    def _compare_list_summary(self, ref_list, alt_list) -> Tuple[float, Dict[str, Any]]:
        """Summary comparison of two lists.

        Compares structural similarity: key/length overlap, value type distribution.

        Returns:
            Tuple of (similarity, details). similarity is continuous [0.0, 1.0].
        """
        details: Dict[str, Any] = {}
        similarities: List[float] = []

        if isinstance(ref_list, dict) and isinstance(alt_list, dict):
            ref_keys = set(ref_list.keys())
            alt_keys = set(alt_list.keys())
            all_keys = ref_keys | alt_keys
            common_keys = ref_keys & alt_keys

            key_overlap = len(common_keys) / len(all_keys) if len(all_keys) > 0 else 1.0
            similarities.append(key_overlap)
            details["key_overlap"] = float(key_overlap)

            size_sim = min_max_similarity(len(ref_list), len(alt_list))
            similarities.append(size_sim)
            details["size_similarity"] = float(size_sim)

            if len(common_keys) > 0:
                type_matches = sum(
                    1 for k in common_keys
                    if type(ref_list[k]).__name__ == type(alt_list[k]).__name__
                )
                type_sim = type_matches / len(common_keys)
                similarities.append(type_sim)
                details["value_type_similarity"] = float(type_sim)

        elif isinstance(ref_list, list) and isinstance(alt_list, list):
            length_sim = min_max_similarity(len(ref_list), len(alt_list))
            similarities.append(length_sim)
            details["length_similarity"] = float(length_sim)

            if len(ref_list) > 0 and len(alt_list) > 0:
                ref_types = {type(v).__name__ for v in ref_list}
                alt_types = {type(v).__name__ for v in alt_list}
                union = len(ref_types | alt_types)
                intersection = len(ref_types & alt_types)
                type_sim = intersection / union if union > 0 else 1.0
                similarities.append(type_sim)
                details["type_distribution_similarity"] = float(type_sim)

        similarity = float(np.mean(similarities)) if similarities else 0.0
        return similarity, details

    # ============================================================================
    # Per-Type Comparison Helpers — Vector
    # ============================================================================

    def _compare_vector_exact(self, ref_vec: np.ndarray, alt_vec: np.ndarray,
                              tolerance: float = 1e-6) -> Tuple[float, Dict[str, Any]]:
        """Exact comparison of two vectors.

        Requires identical length and all values matching within tolerance.

        Returns:
            Tuple of (similarity, details). similarity is 1.0 or 0.0.
        """
        details: Dict[str, Any] = {
            "ref_length": len(ref_vec),
            "alt_length": len(alt_vec),
            "tolerance": tolerance,
        }

        if len(ref_vec) != len(alt_vec):
            details["length_match"] = False
            return 0.0, details
        details["length_match"] = True

        try:
            ref_vals = ref_vec.astype(np.float64)
            alt_vals = alt_vec.astype(np.float64)
            is_match = np.allclose(ref_vals, alt_vals, rtol=tolerance, atol=tolerance, equal_nan=True)
            if not is_match:
                diff = np.abs(ref_vals - alt_vals)
                details["max_diff"] = float(np.nanmax(diff))
                details["mean_diff"] = float(np.nanmean(diff))
        except (ValueError, TypeError):
            is_match = np.array_equal(ref_vec, alt_vec)

        details["values_match"] = is_match
        return (1.0 if is_match else 0.0), details

    def _compare_vector_approx(self, ref_vec: np.ndarray, alt_vec: np.ndarray,
                               tolerance: float = 1e-3) -> Tuple[float, Dict[str, Any]]:
        """Approximate comparison of two vectors.

        For numeric vectors: fraction of elements matching within tolerance.
        For non-numeric: fraction of exact matches.
        Handles length mismatches by comparing up to min length and penalizing.

        Returns:
            Tuple of (similarity, details). similarity is continuous [0.0, 1.0].
        """
        details: Dict[str, Any] = {
            "ref_length": len(ref_vec),
            "alt_length": len(alt_vec),
            "tolerance": tolerance,
        }

        max_len = max(len(ref_vec), len(alt_vec))
        if max_len == 0:
            return 1.0, details

        min_len = min(len(ref_vec), len(alt_vec))
        try:
            ref_vals = ref_vec[:min_len].astype(np.float64)
            alt_vals = alt_vec[:min_len].astype(np.float64)
            diff = np.abs(ref_vals - alt_vals)
            matching = int(
                np.sum(diff <= tolerance)
                + np.sum(np.isnan(ref_vals) & np.isnan(alt_vals))
            )
        except (ValueError, TypeError):
            matching = int(np.sum(ref_vec[:min_len] == alt_vec[:min_len]))

        similarity = matching / max_len
        details.update({"matching_elements": matching, "compared_length": min_len})
        return float(similarity), details

    def _compare_vector_summary(self, ref_vec: np.ndarray,
                                alt_vec: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Summary comparison of two vectors.

        Compares length similarity and distribution statistics (mean, std, quantiles).

        Returns:
            Tuple of (similarity, details). similarity is continuous [0.0, 1.0].
        """
        details: Dict[str, Any] = {
            "ref_length": len(ref_vec),
            "alt_length": len(alt_vec),
        }
        similarities: List[float] = []

        # 1. Length similarity
        length_sim = min_max_similarity(len(ref_vec), len(alt_vec))
        similarities.append(length_sim)
        details["length_similarity"] = float(length_sim)

        # 2. Distribution comparison (numeric vectors only)
        try:
            ref_vals = ref_vec.astype(np.float64)
            alt_vals = alt_vec.astype(np.float64)
            ref_vals = ref_vals[~np.isnan(ref_vals)]
            alt_vals = alt_vals[~np.isnan(alt_vals)]

            if len(ref_vals) > 0 and len(alt_vals) > 0:
                mean_sim = min_max_similarity(float(np.mean(ref_vals)), float(np.mean(alt_vals)))
                similarities.append(mean_sim)
                details["mean_similarity"] = float(mean_sim)

                std_sim = min_max_similarity(float(np.std(ref_vals)), float(np.std(alt_vals)))
                similarities.append(std_sim)
                details["std_similarity"] = float(std_sim)

                quantile_sims = []
                for q in [0.25, 0.5, 0.75]:
                    ref_q = float(np.quantile(ref_vals, q))
                    alt_q = float(np.quantile(alt_vals, q))
                    quantile_sims.append(min_max_similarity(ref_q, alt_q))
                quantile_sim = float(np.mean(quantile_sims))
                similarities.append(quantile_sim)
                details["quantile_similarity"] = quantile_sim
        except (ValueError, TypeError):
            ref_set = set(str(v) for v in ref_vec)
            alt_set = set(str(v) for v in alt_vec)
            union = len(ref_set | alt_set)
            intersection = len(ref_set & alt_set)
            value_overlap = intersection / union if union > 0 else 1.0
            similarities.append(value_overlap)
            details["value_overlap"] = float(value_overlap)

        similarity = float(np.mean(similarities)) if similarities else 0.0
        return similarity, details

    # ============================================================================
    # General Helpers
    # ============================================================================

    def _values_equal(self, ref_val, alt_val, tolerance: float = 1e-6) -> bool:
        """Recursively compare two values with tolerance for numeric types.

        Handles nested dicts, lists, and scalar values.
        """
        if isinstance(ref_val, (int, float)) and isinstance(alt_val, (int, float)):
            if np.isnan(ref_val) and np.isnan(alt_val):
                return True
            return abs(ref_val - alt_val) <= tolerance
        elif isinstance(ref_val, dict) and isinstance(alt_val, dict):
            if set(ref_val.keys()) != set(alt_val.keys()):
                return False
            return all(self._values_equal(ref_val[k], alt_val[k], tolerance) for k in ref_val)
        elif isinstance(ref_val, list) and isinstance(alt_val, list):
            if len(ref_val) != len(alt_val):
                return False
            return all(self._values_equal(r, a, tolerance) for r, a in zip(ref_val, alt_val))
        else:
            return ref_val == alt_val

    # ============================================================================
    # Comparison Dispatcher
    # ============================================================================

    def _dispatch_type_comparison(self, obj_type: str, ref_obj, alt_obj,
                                  strategy: str, **kwargs) -> Tuple[float, Dict[str, Any]]:
        """Dispatch comparison to the appropriate per-type helper based on strategy.
        
        Args:
            obj_type: Object type ("matrix", "dataframe", "list", "vector")
            ref_obj: Reference object
            alt_obj: Candidate object
            strategy: Comparison strategy ("exact", "approximate", "summary")
            **kwargs: Strategy-specific parameters (e.g., tolerance)
        
        Returns:
            Tuple of (similarity, details)
        """
        tolerance = kwargs.get('tolerance', 1e-6 if strategy == "exact" else 1e-3)

        dispatch = {
            "matrix": {
                "exact": lambda: self._compare_matrix_exact(ref_obj, alt_obj, tolerance),
                "approximate": lambda: self._compare_matrix_approx(ref_obj, alt_obj, tolerance),
                "summary": lambda: self._compare_matrix_summary(ref_obj, alt_obj),
            },
            "dataframe": {
                "exact": lambda: self._compare_dataframe_exact(ref_obj, alt_obj, tolerance),
                "approximate": lambda: self._compare_dataframe_approx(ref_obj, alt_obj, tolerance),
                "summary": lambda: self._compare_dataframe_summary(ref_obj, alt_obj),
            },
            "list": {
                "exact": lambda: self._compare_list_exact(ref_obj, alt_obj, tolerance),
                "approximate": lambda: self._compare_list_approx(ref_obj, alt_obj, tolerance),
                "summary": lambda: self._compare_list_summary(ref_obj, alt_obj),
            },
            "vector": {
                "exact": lambda: self._compare_vector_exact(ref_obj, alt_obj, tolerance),
                "approximate": lambda: self._compare_vector_approx(ref_obj, alt_obj, tolerance),
                "summary": lambda: self._compare_vector_summary(ref_obj, alt_obj),
            },
        }

        type_dispatch = dispatch.get(obj_type)
        if type_dispatch is None:
            return 0.0, {"error": f"Unsupported object type: {obj_type}"}

        strategy_fn = type_dispatch.get(strategy)
        if strategy_fn is None:
            return 0.0, {"error": f"Unsupported strategy: {strategy}"}

        return strategy_fn()

    # ============================================================================
    # Private Comparison Implementation Methods
    # ============================================================================

    def _compare_by_strategy(self, ref_validation: ValidationResult,
                             alt_validation: ValidationResult,
                             strategy: str, **kwargs):
        """Unified comparison dispatcher for all strategies.

        Routes to Rdata multi-object comparison or single RDS object comparison,
        then dispatches to per-type helpers via _dispatch_type_comparison.

        Args:
            ref_validation: Validated reference file
            alt_validation: Validated candidate file
            strategy: Comparison strategy ("exact", "approximate", "summary")
            **kwargs: Strategy-specific parameters (e.g., tolerance)
        """
        tolerance = kwargs.get('tolerance', 1e-6 if strategy == "exact" else 1e-3)

        try:
            ref_file_type = ref_validation.details.get("file_type", "rds")
            alt_file_type = alt_validation.details.get("file_type", "rds")

            # Handle Rdata files (multiple objects)
            if ref_file_type == "rdata" or alt_file_type == "rdata":
                self._compare_rdata_files(ref_validation, alt_validation, strategy, **kwargs)
                return

            # Both are RDS files
            ref_type = ref_validation.details.get("object_type")
            alt_type = alt_validation.details.get("object_type")

            if ref_type is None or alt_type is None:
                self.result.error = "Failed to detect RDS object type"
                return

            if ref_type != alt_type:
                self.result.similarity = 0.0
                self.result.details.update({"ref_type": ref_type, "alt_type": alt_type, "type_match": False})
                return

            self.result.details["object_type"] = ref_type
            if strategy != "summary":
                self.result.details["tolerance"] = tolerance

            # Route Seurat/SCE to AnnDataHandler
            if ref_type in ("seurat", "sce"):
                anndata_result = self.anndata_handler.compare(
                    ref_validation.file_path, alt_validation.file_path, strategy=strategy, **kwargs,
                )
                self.result.similarity = anndata_result.similarity
                self.result.details.update(anndata_result.details)
                if anndata_result.error:
                    self.result.error = anndata_result.error
                return

            # Extract and compare
            ref_obj = self._load_file(ref_validation.file_path, ref_type, ref_file_type)
            alt_obj = self._load_file(alt_validation.file_path, alt_type, alt_file_type)
            if ref_obj is None:
                self.result.error = f"Failed to extract {ref_type} object from reference file"
                return
            if alt_obj is None:
                self.result.similarity = 0.0
                self.result.details["extraction_failed"] = f"candidate {ref_type} object"
                return

            similarity, details = self._dispatch_type_comparison(
                ref_type, ref_obj, alt_obj, strategy, **kwargs,
            )
            self.result.similarity = similarity
            self.result.details.update(details)

        except Exception as e:
            logger.error(f"{strategy.capitalize()} comparison failed: {e}")
            self.result.error = f"{strategy.capitalize()} comparison failed: {e}"

    def _compare_rdata_files(self, ref_validation: ValidationResult,
                             alt_validation: ValidationResult,
                             strategy: str, **kwargs):
        """Compare Rdata files containing multiple objects.

        Compares objects by name, averaging similarities across common objects.

        Args:
            ref_validation: Validated reference file
            alt_validation: Validated candidate file
            strategy: Comparison strategy ("exact", "approximate", "summary")
            **kwargs: Strategy-specific parameters
        """
        ref_file_type = ref_validation.details.get("file_type", "rds")
        alt_file_type = alt_validation.details.get("file_type", "rds")

        if ref_file_type != alt_file_type:
            self.result.similarity = 0.0
            self.result.details.update({"ref_file_type": ref_file_type, "alt_file_type": alt_file_type})
            return

        ref_obj_names = ref_validation.details.get("object_names", [])
        alt_obj_names = alt_validation.details.get("object_names", [])

        if not ref_obj_names:
            self.result.error = "Failed to get object names from reference Rdata file"
            return
        if not alt_obj_names:
            self.result.similarity = 0.0
            self.result.details["alt_object_names"] = []
            return

        common_names = set(ref_obj_names) & set(alt_obj_names)
        if len(common_names) == 0:
            self.result.similarity = 0.0
            self.result.details.update({
                "ref_objects": ref_obj_names,
                "alt_objects": alt_obj_names,
            })
            return

        # Get all objects info for both files (more efficient than per-object calls)
        ref_objects_info = ref_validation.details.get("objects_info")
        alt_objects_info = alt_validation.details.get("objects_info")
        
        # If not in validation details, fetch it now
        if ref_objects_info is None:
            _, ref_objects_info = self._get_r_object_type_and_info(ref_validation.file_path, is_rds=False)
        if alt_objects_info is None:
            _, alt_objects_info = self._get_r_object_type_and_info(alt_validation.file_path, is_rds=False)
        
        if ref_objects_info is None:
            self.result.error = "Failed to get objects information from reference Rdata file"
            return
        if alt_objects_info is None:
            self.result.similarity = 0.0
            self.result.details["alt_objects_info"] = None
            return

        similarities: List[float] = []
        per_object_details: Dict[str, Any] = {}

        for obj_name in sorted(common_names):
            # Extract types from objects info
            ref_obj_info = ref_objects_info.get(obj_name, {})
            alt_obj_info = alt_objects_info.get(obj_name, {})
            ref_type = ref_obj_info.get("type")
            alt_type = alt_obj_info.get("type")

            if ref_type != alt_type:
                similarities.append(0.0)
                per_object_details[obj_name] = {
                    "type_mismatch": True, "ref_type": ref_type, "alt_type": alt_type,
                }
                continue

            if ref_type in ("seurat", "sce"):
                logger.warning(f"Seurat/SCE comparison in Rdata files not yet fully supported: {obj_name}")
                similarities.append(0.0)
                per_object_details[obj_name] = {"unsupported": True, "type": ref_type}
                continue

            ref_obj = self._load_file(ref_validation.file_path, ref_type, "rdata", obj_name)
            if ref_obj is None:
                self.result.error = f"Failed to extract reference object '{obj_name}' from Rdata file"
                return
            alt_obj = self._load_file(alt_validation.file_path, alt_type, "rdata", obj_name)
            if alt_obj is None:
                similarities.append(0.0)
                per_object_details[obj_name] = {"alt_extraction_failed": True}
                continue

            sim, obj_details = self._dispatch_type_comparison(
                ref_type, ref_obj, alt_obj, strategy, **kwargs,
            )
            similarities.append(sim)
            per_object_details[obj_name] = obj_details

        if not similarities:
            self.result.error = "No comparable objects found (unexpected: common_names was non-empty)"
            return

        self.result.similarity = float(np.mean(similarities))
        self.result.details.update({
            "num_common_objects": len(common_names),
            "num_ref_objects": len(ref_obj_names),
            "num_alt_objects": len(alt_obj_names),
            "common_objects": sorted(common_names),
            "ref_only_objects": sorted(set(ref_obj_names) - common_names),
            "alt_only_objects": sorted(set(alt_obj_names) - common_names),
            "per_object_details": per_object_details,
        })
