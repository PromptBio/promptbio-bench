"""AnnData format handler for Python single-cell and matrix data.

Strategy:
- exact: Shape, index, count matrix, metadata, and embedding identity (with float tolerance)
- approximate: Align to common obs/var, element-wise matrix match within tolerance, metadata match
- summary: Component-based comparison of shape, sparsity, matrix correlation, cluster ARI, embedding correlation

AnnData objects can be stored in multiple formats:
- .h5ad: HDF5-based AnnData format (most common)
- .loom: Loom format for single-cell data
- .h5: Generic HDF5 (may contain AnnData)
- .rds: R serialization format (Seurat/SCE objects, converted to AnnData via R)
- .rdata/.RData/.rda: R workspace format (Seurat/SCE objects, converted to AnnData via R)
- folder: 10X Genomics directory (barcodes.tsv.gz, features.tsv.gz, matrix.mtx.gz)

Format detection uses detect_file_signature which returns format="rds" for .rds files,
format="rdata" for .rdata/.RData/.rda files, and format="folder" with note="10x" for 10X Genomics directories.

Required cmdline tools:
- Rscript (for .rds/.rdata Seurat/SingleCellExperiment conversion)
"""

import json
import logging
import os
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import anndata as ad
import scanpy as sc

from tools.base import FormatHandler, resolve_r_executable
from tools.results import ValidationResult, ComparisonResult
from utils.format import detect_file_signature
from utils.metrics import pearson_correlation, relative_absolute_error, rae_similarity

logger = logging.getLogger(__name__)


class AnnDataHandler(FormatHandler):
    """Handler for AnnData format files (Python single-cell and matrix data).
    
    AnnData objects can be stored in multiple file formats (.h5ad, .loom, .h5).
    This handler supports loading from any of these formats.
    
    AnnData objects contain:
    - X: Count/expression matrix (sparse or dense)
    - obs: Observation metadata (DataFrame) - cells/samples
    - var: Variable metadata (DataFrame) - genes/features
    - obsm: Multi-dimensional observation annotations (e.g., PCA, UMAP embeddings)
    - varm: Multi-dimensional variable annotations
    - layers: Additional matrices (normalized, scaled, etc.)
    - uns: Unstructured annotations
    """
    
    def __init__(self):
        super().__init__("anndata")
        # Formats that detect_file_signature returns for AnnData-compatible files
        # h5ad: HDF5-based AnnData, hdf5: generic HDF5 (.h5), loom: Loom format
        # rds: R serialization format (Seurat/SCE objects, converted via R)
        # rdata: R workspace format (can contain Seurat/SCE objects, converted via R)
        # folder: directory format (e.g., 10X Genomics with note="10x")
        self.supported_formats = ['h5ad', 'hdf5', 'loom', 'rds', 'rdata', 'folder']
        self.supported_strategies = ["exact", "approximate", "summary"]
        self.format_map = {
            'h5ad': lambda p: ad.read_h5ad(p),
            'h5': lambda p: ad.read_h5ad(p),
            'loom': lambda p: ad.read_loom(p),
        }
    
    # ============================================================================
    # Public API Methods
    # ============================================================================
    
    def validate(self, file_path: str, **kwargs) -> ValidationResult:
        """Validate AnnData file format.
        
        Supports .h5ad, .loom, .h5 files, .rds files (Seurat/SCE objects),
        and 10X Genomics format directories (folder with note="10x").
        
        Args:
            file_path: Path to AnnData file or 10X directory
            **kwargs: optional parameters (e.g., min_size)
            - min_size: Minimum size of the file in bytes, default is 0
        """
        # check existence and non-emptiness
        is_exists = self._is_file_exists(file_path)
        is_nonempty = self._is_file_nonempty(file_path, min_size=kwargs.get("min_size", 0))
        
        # detect format via signature (handles both files and directories)
        signature = detect_file_signature(file_path, fallback="extension")
        is_format_ok = signature.format in self.supported_formats
        # For folders, only accept if note indicates 10x content
        if signature.format == 'folder':
            is_format_ok = (signature.note == '10x')
        if signature.format in ['rds', 'rdata']:
            r_object_type = self._detect_r_object_type(file_path)
            is_format_ok = (r_object_type in ['seurat', 'sce'])

        # initialize validation result
        result = ValidationResult(exists=is_exists, non_empty=is_nonempty, format_ok=is_format_ok, parseable=False,
                                  file_path=file_path, signature=signature, details={}, error=None)

        if not is_exists:
            logger.error(f"AnnData file/directory not found: {file_path}")
            result.error = "File or directory not found"
            return result
        
        if not is_nonempty:
            logger.warning(f"AnnData file/directory is empty: {file_path}")
            result.error = "File or directory is empty"
            return result
        
        if not is_format_ok:
            logger.error(f"Not an AnnData-compatible file: detected={signature.format}")
            result.error = f"Not an AnnData-compatible file: detected={signature.format}"
            return result
        
        # load and parse AnnData object
        try:
            adata = self._load_file(file_path, signature)
            result.parseable = True
            
            # Extract validation details
            details = self._get_anndata_details(adata, signature.format)
            result.details.update(details)
            
            # Update non_empty based on actual content
            result.non_empty = details['num_obs'] > 0 and details['num_vars'] > 0
            if not result.non_empty:
                result.error = "AnnData object has 0 observations or 0 variables"
                return result
            
            logger.info(f"AnnData validation passed: {file_path}")
        except Exception as e:
            logger.error(f"AnnData validation failed: {e}")
            result.error = f"AnnData validation failed: {e}"
        return result
    
    def compare(self, reference_path: str, candidate_path: str, strategy: str = "summary", **kwargs) -> ComparisonResult:
        """Compare AnnData files.
        
        Supports loading from .h5ad, .loom, .h5, .rds files, and 10X directories.
        
        Supported strategies:
        - exact: Shape, index, count matrix, metadata, and embedding identity (with float tolerance)
        - approximate: Align to common obs/var, element-wise matrix match within tolerance
        - summary: Component-based comparison of shape, sparsity, matrix correlation, cluster ARI, embedding correlation
        
        Args:
            reference_path: Path to reference AnnData file
            candidate_path: Path to candidate AnnData file
            strategy: Comparison strategy (exact, approximate, summary)
            **kwargs: Strategy-specific parameters
            - tolerance: Floating point tolerance for matrix comparison (default: 1e-6 for exact, 1e-3 for approximate)
            - layer: Layer name to compare (default: None, uses X matrix)
            - reduction: Reduction key in obsm to compare (default: "X_umap")
            - cluster_col: Cluster column name in obs (default: None, auto-detect)
        """
        # initialize comparison result
        self.result = ComparisonResult(reference_path=reference_path, candidate_path=candidate_path, 
                                       strategy=strategy, similarity=np.nan, details={}, error=None)
        
        # validate reference and candidate files
        ref_validation = self.validate(reference_path, **kwargs)
        alt_validation = self.validate(candidate_path, **kwargs)

        try:
            if ref_validation.is_valid and alt_validation.is_valid:
                ref_empty = not ref_validation.non_empty
                alt_empty = not alt_validation.non_empty
                if ref_empty or alt_empty:
                    # At least one file has no observations/variables — compare emptiness directly
                    self.result.similarity = 1.0 if (ref_empty and alt_empty) else 0.0
                    self.result.details.update({"ref_empty": ref_empty, "alt_empty": alt_empty})
                elif strategy == "exact":
                    self._compare_exact(ref_validation, alt_validation, **kwargs)
                elif strategy == "approximate":
                    self._compare_approx(ref_validation, alt_validation, **kwargs)
                elif strategy == "summary":
                    self._compare_summary(ref_validation, alt_validation, **kwargs)
                else:
                    logger.warning(f"Unknown strategy '{strategy}', fall back to summary")
                    self._compare_summary(ref_validation, alt_validation, **kwargs)
        except Exception as e:
            logger.error(f"AnnData comparison failed: {e}")
            self.result.error = f"AnnData comparison failed: {e}"
        return self._finalize_output(ref_validation, alt_validation, self.result)
    
    # ============================================================================
    # File I/O and Loading Methods
    # ============================================================================
    
    def _load_file(self, file_path: str, signature=None):
        """Load AnnData object from file or directory.
        
        Supports multiple formats:
        - .h5ad: HDF5-based AnnData format (read_h5ad)
        - .loom: Loom format (read_loom)
        - .h5: Generic HDF5 (try read_h5ad)
        - .rds: R serialization format (Seurat/SCE objects, converted via R)
        - .rdata/.RData/.rda: R workspace format (Seurat/SCE objects, converted via R)
        - folder: Directory with 10X Genomics files (note="10x")
        
        Args:
            file_path: Path to file or directory
            signature: FileSignature from detect_file_signature. If None, auto-detects.
        
        Returns:
            AnnData object
        
        Raises:
            ValueError: If the file cannot be loaded with any supported format.
        """
        if signature is None:
            signature = detect_file_signature(file_path, fallback="extension")
        
        file_format = signature.format
        
        # Folder with 10X Genomics content
        if file_format == 'folder' and signature.note == '10x':
            return sc.read_10x_mtx(file_path, var_names='gene_symbols', cache=True)
        
        # RDS or RData file (Seurat or SingleCellExperiment)
        if file_format in ('rds', 'rdata'):
            r_object_type = self._detect_r_object_type(file_path)
            if r_object_type not in ['seurat', 'sce']:
                raise ValueError(f"R file is not a Seurat or SingleCellExperiment object (detected: {r_object_type}): {file_path}")
            
            fd, temp_h5ad = tempfile.mkstemp(suffix='.h5ad', prefix=f'{r_object_type}_to_anndata_')
            os.close(fd)
            
            try:
                if not self._convert_r_to_h5ad(file_path, temp_h5ad, r_object_type):
                    raise ValueError(f"Failed to convert {r_object_type} to AnnData: {file_path}")
                return ad.read_h5ad(temp_h5ad)
            finally:
                self._cleanup_file(temp_h5ad)
        
        # Standard formats via format_map (h5ad, h5, loom)
        format_order = [file_format] if file_format in self.format_map else []
        format_order.extend(fmt for fmt in self.format_map if fmt != file_format)
        
        for fmt in format_order:
            try:
                return self.format_map[fmt](file_path)
            except Exception:
                continue
        
        raise ValueError(f"Could not load AnnData from {file_path} (format: {file_format})")
    
    # ============================================================================
    # File Status Checkers
    # ============================================================================
    
    def _detect_r_object_type(self, file_path: str) -> str:
        """Detect R object type in RDS or RData file.
        
        Uses the R script tools/helper/get_r_object_info.R for detection.
        Looks for Seurat or SCE objects in the file.
        
        Args:
            file_path: Path to .rds or .rdata file
        
        Returns:
            Object type: 'seurat', 'sce', or 'other'
        """
        # Get path to R script
        script_dir = Path(__file__).parent / "helper"
        r_script_path = script_dir / "get_r_object_info.R"
        
        if not r_script_path.exists():
            logger.error(f"R script not found: {r_script_path}")
            return "other"
        
        # Build command
        cmd = [
            resolve_r_executable("Rscript"),
            str(r_script_path),
            "--input", str(file_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,  # 1 minute timeout
                check=False
            )
            
            if result.returncode != 0:
                logger.warning(f"R script failed with return code {result.returncode}: {result.stderr}")
                return "other"
            
            # Parse JSON output - look for seurat or sce types
            try:
                all_objects = json.loads(result.stdout.strip())
                
                # Check all objects for seurat or sce type
                for obj_name, obj_info in all_objects.items():
                    obj_type = obj_info.get("type")
                    if obj_type in ("seurat", "sce"):
                        return obj_type
                
                # No seurat/sce found
                return "other"
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON output from R script: {e}")
                return "other"
                
        except subprocess.TimeoutExpired:
            logger.error(f"R script timed out: {file_path}")
            return "other"
        except Exception as e:
            logger.error(f"Failed to run R script: {e}")
            return "other"
    

    # ============================================================================
    # File Creation Methods
    # ============================================================================
    
    def _convert_r_to_h5ad(self, r_path: str, h5ad_path: str, r_object_type: str) -> bool:
        """Convert R object (Seurat or SingleCellExperiment) from RDS or RData to H5AD format using R script.
        
        Uses the standalone R script tools/helper/{r_object_type}_to_h5ad.R for conversion.
        The R script auto-detects the file format (RDS or RData) from the file extension.
        For RData files, the R script extracts the first Seurat/SCE object found in the workspace.
        
        Args:
            r_path: Path to input .rds or .rdata file
            h5ad_path: Path to output .h5ad file
            r_object_type: Type of R object ('seurat' or 'sce')
        
        Returns:
            True if conversion successful, False otherwise
        """
        # Get path to R script
        script_dir = Path(__file__).parent / "helper"
        r_script_path = script_dir / f"{r_object_type}_to_h5ad.R"
        
        if not r_script_path.exists():
            logger.error(f"R script not found: {r_script_path}")
            return False
        
        # Build command (R scripts auto-detect format from file extension)
        cmd = [
            resolve_r_executable("Rscript"),
            str(r_script_path),
            "--input", str(r_path),
            "--output", str(h5ad_path)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"R script failed with return code {result.returncode}")
                logger.error(f"stderr: {result.stderr}")
                logger.error(f"stdout: {result.stdout}")
                return False
            
            # Check if output file was created
            if not os.path.exists(h5ad_path):
                logger.error(f"Conversion appeared successful but output file not found: {h5ad_path}")
                return False
            
            # Check for success message in stdout
            if "SUCCESS:" in result.stdout or "Conversion completed successfully" in result.stdout:
                logger.info(f"Successfully converted {r_object_type} to H5AD: {r_path} -> {h5ad_path}")
                return True
            else:
                # Check for ERROR in output
                if "ERROR:" in result.stdout or "ERROR:" in result.stderr:
                    logger.error(f"Conversion failed: {result.stdout} {result.stderr}")
                    return False
                # If no explicit error but file exists, assume success
                return True
                
        except FileNotFoundError:
            logger.error(f"Rscript not found. Please install R to use {r_object_type} conversion.")
            return False
        except subprocess.TimeoutExpired:
            logger.error("R script timed out after 600 seconds")
            return False
        except Exception as e:
            logger.error(f"Failed to run R script: {e}")
            return False
    
    # ============================================================================
    # Statistics and Analysis Methods
    # ============================================================================
    
    def _get_anndata_details(self, adata, file_format: str) -> Dict[str, Any]:
        """Get details from AnnData object.
        
        Args:
            adata: AnnData object
            file_format: Detected file format
        
        Returns:
            Dictionary with AnnData details
        """
        return {
            "format": file_format,
            "num_obs": adata.n_obs,
            "num_vars": adata.n_vars,
            "obs_columns": list(adata.obs.columns) if hasattr(adata, 'obs') else [],
            "var_columns": list(adata.var.columns) if hasattr(adata, 'var') else [],
            "layers": list(adata.layers.keys()) if hasattr(adata, 'layers') else [],
            "obsm_keys": list(adata.obsm.keys()) if hasattr(adata, 'obsm') else [],
            "is_sparse": hasattr(adata.X, 'toarray') if hasattr(adata, 'X') else False,
        }
    
    def _get_count_matrix(self, adata, layer: Optional[str] = None) -> np.ndarray:
        """Get count matrix from AnnData object.
        
        Args:
            adata: AnnData object
            layer: Layer name (if None, uses X matrix)
        
        Returns:
            Dense numpy array of count matrix (n_obs x n_vars)
        """
        if layer is not None and layer in adata.layers:
            matrix = adata.layers[layer]
        else:
            matrix = adata.X
        
        # Convert sparse to dense if needed
        if hasattr(matrix, 'toarray'):
            return matrix.toarray()
        else:
            return np.asarray(matrix)
    
    def _get_embeddings(self, adata, key: str = "X_umap") -> Optional[pd.DataFrame]:
        """Get dimensionality reduction embeddings from AnnData object.
        
        Args:
            adata: AnnData object
            key: Key in obsm (default: "X_umap", can be "X_pca", "X_tsne", etc.)
        
        Returns:
            DataFrame with observations as rows, dimensions as columns
        """
        if key not in adata.obsm:
            return None
        
        embed = adata.obsm[key]
        if isinstance(embed, np.ndarray):
            embed_df = pd.DataFrame(embed, index=adata.obs.index)
            # Name columns
            embed_df.columns = [f"{key}_{i}" for i in range(embed.shape[1])]
            return embed_df
        return None

    def _get_clusters(self, adata, cluster_col: Optional[str] = None) -> Optional[pd.Series]:
        """Get cluster assignments from AnnData object.
        
        Args:
            adata: AnnData object
            cluster_col: Column name in obs (if None, tries common names like "leiden", "louvain")
        
        Returns:
            Series with observation_id as index, cluster as values
        """
        if cluster_col is None:
            # Common cluster column names in Scanpy
            possible_cols = ['leiden', 'louvain', 'cluster', 'clusters', 'seurat_clusters']
            for col in possible_cols:
                if col in adata.obs.columns:
                    cluster_col = col
                    break
        
        if cluster_col is None or cluster_col not in adata.obs.columns:
            logger.debug("No cluster column found in obs")
            return None
        
        return adata.obs[cluster_col]
    
    def _get_sparsity(self, adata, layer: Optional[str] = None) -> float:
        """Get sparsity of the count matrix.
        
        Args:
            adata: AnnData object
            layer: Layer name (if None, uses X matrix)
        
        Returns:
            Fraction of zero elements (0.0-1.0)
        """
        matrix = self._get_count_matrix(adata, layer=layer)
        total_elements = matrix.size
        if total_elements == 0:
            return 0.0
        zero_count = np.sum(matrix == 0)
        return float(zero_count / total_elements)
    
    def _align_matrices(self, ref_adata, alt_adata, layer: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Align two matrices to common variables and observations.
        
        AnnData X matrix is (n_obs, n_vars), so obs = rows, vars = columns.
        
        Args:
            ref_adata: Reference AnnData object
            alt_adata: Candidate AnnData object
            layer: Layer name (if None, uses X matrix)
        
        Returns:
            Tuple of (ref_aligned, alt_aligned, common_vars, common_obs) or (None, None, None, None) if no overlap
        """
        # Extract count matrices
        ref_matrix = self._get_count_matrix(ref_adata, layer=layer)
        alt_matrix = self._get_count_matrix(alt_adata, layer=layer)
        
        common_vars = ref_adata.var.index.intersection(alt_adata.var.index)
        common_obs = ref_adata.obs.index.intersection(alt_adata.obs.index)
        
        if len(common_vars) == 0 or len(common_obs) == 0:
            return None, None, None, None
        
        # Get index positions for alignment
        ref_obs_idx = [ref_adata.obs.index.get_loc(o) for o in common_obs]
        alt_obs_idx = [alt_adata.obs.index.get_loc(o) for o in common_obs]
        ref_var_idx = [ref_adata.var.index.get_loc(v) for v in common_vars]
        alt_var_idx = [alt_adata.var.index.get_loc(v) for v in common_vars]
        
        # AnnData X is (n_obs, n_vars): obs=rows, vars=cols
        ref_aligned = ref_matrix[np.ix_(ref_obs_idx, ref_var_idx)]
        alt_aligned = alt_matrix[np.ix_(alt_obs_idx, alt_var_idx)]
        
        return ref_aligned, alt_aligned, common_vars, common_obs
    
    def _compare_matrix(self, ref_adata, alt_adata, layer: Optional[str] = None, tolerance: float = 1e-6) -> float:
        """Compare count matrices between two AnnData objects.
        
        Args:
            ref_adata: Reference AnnData object
            alt_adata: Candidate AnnData object
            layer: Layer name (if None, uses X matrix)
            tolerance: Floating point tolerance for comparison
        
        Returns:
            Fraction of elements within tolerance (0.0 to 1.0)
        """
        # Align matrices to common obs/vars
        ref_aligned, alt_aligned, common_vars, common_obs = self._align_matrices(ref_adata, alt_adata, layer=layer)
        
        if ref_aligned is None or alt_aligned is None or len(common_obs) == 0 or len(common_vars) == 0:
            return 0.0
        
        # Calculate relative absolute error
        rae = relative_absolute_error(ref_aligned, alt_aligned)
        total_elements = rae.size
        matching_elements = int(np.sum(rae <= tolerance))
        fraction_match = matching_elements / total_elements if total_elements > 0 else 0.0
        
        return fraction_match
    
    def _compare_metadata(self, ref_adata, alt_adata, tolerance: float = 1e-6) -> Tuple[bool, float]:
        """Compare metadata between two AnnData objects.
        
        Args:
            ref_adata: Reference AnnData object
            alt_adata: Candidate AnnData object
            tolerance: Floating point tolerance for comparison
        
        Returns:
            Tuple of (all_match: bool, fraction_match: float)
        """
        common_obs = ref_adata.obs.index.intersection(alt_adata.obs.index)
        
        if len(common_obs) == 0:
            return True, 1.0
        
        ref_meta = ref_adata.obs.copy()
        alt_meta = alt_adata.obs.copy()
        
        ref_meta_aligned = ref_meta.loc[common_obs]
        alt_meta_aligned = alt_meta.loc[common_obs]
        
        common_cols = ref_meta_aligned.columns.intersection(alt_meta_aligned.columns)
        
        if len(common_cols) == 0:
            return True, 1.0
        
        col_matches = []
        for col in common_cols:
            ref_col = ref_meta_aligned[col]
            alt_col = alt_meta_aligned[col]
            
            if ref_col.dtype in [np.float64, np.float32] or alt_col.dtype in [np.float64, np.float32]:
                try:
                    match = np.allclose(ref_col.values.astype(float), alt_col.values.astype(float), 
                                       rtol=tolerance, atol=tolerance, equal_nan=True)
                except (ValueError, TypeError):
                    match = ref_col.equals(alt_col)
            else:
                match = ref_col.equals(alt_col)
            col_matches.append(match)
        
        all_match = all(col_matches)
        fraction = sum(col_matches) / len(col_matches) if col_matches else 1.0
        return all_match, fraction
    
    def _compare_embeddings(self, ref_adata, alt_adata, tolerance: float = 1e-6) -> Tuple[bool, Dict[str, Any]]:
        """Compare all obsm embeddings between two AnnData objects for exact match.
        
        Args:
            ref_adata: Reference AnnData object
            alt_adata: Candidate AnnData object
            tolerance: Floating point tolerance
        
        Returns:
            Tuple of (all_match, details_dict)
        """
        ref_keys = set(ref_adata.obsm.keys()) if hasattr(ref_adata, 'obsm') else set()
        alt_keys = set(alt_adata.obsm.keys()) if hasattr(alt_adata, 'obsm') else set()
        common_keys = ref_keys & alt_keys
        
        if len(common_keys) == 0:
            return True, {"obsm_keys_compared": 0}
        
        all_match = True
        for key in common_keys:
            ref_embed = ref_adata.obsm[key]
            alt_embed = alt_adata.obsm[key]
            
            if isinstance(ref_embed, np.ndarray) and isinstance(alt_embed, np.ndarray):
                if ref_embed.shape != alt_embed.shape:
                    all_match = False
                    break
                if not np.allclose(ref_embed, alt_embed, rtol=tolerance, atol=tolerance, equal_nan=True):
                    all_match = False
                    break
        
        return all_match, {"obsm_keys_compared": len(common_keys)}
    
    # ============================================================================
    # Private Comparison Implementation Methods
    # ============================================================================
    
    def _compare_exact(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Exact match: shape, indices, count matrix, metadata, and embeddings must be identical.
        
        All checks must pass for similarity=1.0, otherwise similarity=0.0.
        
        Criteria:
        - Shape match (n_obs, n_vars)
        - Obs/var index identity (same cell barcodes and gene names, same order)
        - Count matrix element-wise match within tolerance (default: 1e-6)
        - Obs metadata match (column-by-column, float tolerance for numeric)
        - Obsm embeddings match (key-by-key, element-wise)
        
        Args:
            **kwargs: Strategy-specific parameters
            - tolerance: Floating point tolerance (default: 1e-6)
            - layer: Layer name to compare (default: None, uses X matrix)
        """
        try:
            tolerance = kwargs.get('tolerance', 1e-6)
            layer = kwargs.get('layer', None)
            
            # Load AnnData objects
            ref_adata = self._load_file(ref_validation.file_path, ref_validation.signature)
            alt_adata = self._load_file(alt_validation.file_path, alt_validation.signature)
            
            # 1. Check shape match (fast fail: skip expensive checks if shapes differ)
            shape_match = (ref_adata.n_obs == alt_adata.n_obs and ref_adata.n_vars == alt_adata.n_vars)
            
            if not shape_match:
                self.result.similarity = 0.0
                self.result.details.update({
                    "tolerance": tolerance,
                    "shape_match": False,
                    "ref_shape": (ref_adata.n_obs, ref_adata.n_vars),
                    "alt_shape": (alt_adata.n_obs, alt_adata.n_vars),
                })
                return
            
            # 2. Check obs/var index identity (same names, same order)
            obs_index_match = list(ref_adata.obs.index) == list(alt_adata.obs.index)
            var_index_match = list(ref_adata.var.index) == list(alt_adata.var.index)
            index_match = obs_index_match and var_index_match
            
            # 3. Check count matrix element-wise
            matrix_fraction = self._compare_matrix(ref_adata, alt_adata, layer=layer, tolerance=tolerance)
            matrix_match = (matrix_fraction == 1.0)
            
            # Calculate diff for details (for exact comparison, shapes should match)
            ref_matrix = self._get_count_matrix(ref_adata, layer=layer)
            alt_matrix = self._get_count_matrix(alt_adata, layer=layer)
            diff = np.abs(ref_matrix - alt_matrix)
            max_diff = float(np.max(diff))
            mean_diff = float(np.mean(diff))
            
            # 4. Check obs metadata match
            metadata_match, metadata_fraction = self._compare_metadata(ref_adata, alt_adata, tolerance)
            
            # 5. Check obsm embeddings match
            embedding_match, embed_details = self._compare_embeddings(ref_adata, alt_adata, tolerance)
            
            # All must pass for exact match
            is_exact = index_match and matrix_match and metadata_match and embedding_match
            similarity = 1.0 if is_exact else 0.0
            
            self.result.similarity = similarity
            self.result.details.update({
                "tolerance": tolerance,
                "shape_match": True,
                "ref_shape": (ref_adata.n_obs, ref_adata.n_vars),
                "alt_shape": (alt_adata.n_obs, alt_adata.n_vars),
                "index_match": index_match,
                "obs_index_match": obs_index_match,
                "var_index_match": var_index_match,
                "matrix_match": matrix_match,
                "matrix_fraction": matrix_fraction,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "metadata_match": metadata_match,
                "metadata_fraction": metadata_fraction,
                "embedding_match": embedding_match,
                **embed_details,
            })
            
        except Exception as e:
            logger.error(f"Exact comparison failed: {e}")
            self.result.error = f"Exact comparison failed: {e}"
    
    def _compare_approx(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Approximate match: align to common obs/var, compare matrices and metadata within tolerance.
        
        Unlike exact, does not require identical shapes or index order. Aligns to the intersection
        of observations and variables, then compares element-wise within tolerance.
        
        Similarity = weighted combination of:
            matrix_similarity (0.6) + metadata_similarity (0.2) + obs_overlap (0.1) + var_overlap (0.1)
        
        Args:
            **kwargs: Strategy-specific parameters
            - tolerance: Floating point tolerance (default: 1e-3)
            - layer: Layer name to compare (default: None, uses X matrix)
        """
        try:
            tolerance = kwargs.get('tolerance', 1e-3)
            layer = kwargs.get('layer', None)
            
            # Load AnnData objects
            ref_adata = self._load_file(ref_validation.file_path, ref_validation.signature)
            alt_adata = self._load_file(alt_validation.file_path, alt_validation.signature)
            
            # Align count matrices to common obs/var
            ref_aligned, alt_aligned, common_vars, common_obs = self._align_matrices(
                ref_adata, alt_adata, layer=layer
            )
            
            if ref_aligned is None:
                self.result.similarity = 0.0
                self.result.details.update({"error": "No common variables or observations found"})
                return
            
            # Calculate matrix similarity
            matrix_similarity = self._compare_matrix(ref_adata, alt_adata, common_obs, common_vars, layer=layer, tolerance=tolerance)
            
            # Calculate diff for details
            diff = np.abs(ref_aligned - alt_aligned)
            max_diff = float(np.max(diff))
            mean_diff = float(np.mean(diff))
            total_elements = diff.size
            matching_elements = int(np.sum(diff <= tolerance))
            
            # Calculate metadata similarity on common observations
            _, metadata_similarity = self._compare_metadata(ref_adata, alt_adata, tolerance)
            
            # Calculate index overlap as coverage indicator
            obs_overlap = len(common_obs) / max(ref_adata.n_obs, alt_adata.n_obs) if max(ref_adata.n_obs, alt_adata.n_obs) > 0 else 0.0
            var_overlap = len(common_vars) / max(ref_adata.n_vars, alt_adata.n_vars) if max(ref_adata.n_vars, alt_adata.n_vars) > 0 else 0.0
            
            # Overall similarity: weighted combination
            # matrix (0.6) + metadata (0.2) + obs_overlap (0.1) + var_overlap (0.1)
            similarity = 0.6 * matrix_similarity + 0.2 * metadata_similarity + 0.1 * obs_overlap + 0.1 * var_overlap
            
            self.result.similarity = float(similarity)
            self.result.details.update({
                "tolerance": tolerance,
                "num_common_obs": len(common_obs),
                "num_common_vars": len(common_vars),
                "ref_shape": (ref_adata.n_obs, ref_adata.n_vars),
                "alt_shape": (alt_adata.n_obs, alt_adata.n_vars),
                "obs_overlap": float(obs_overlap),
                "var_overlap": float(var_overlap),
                "matrix_similarity": float(matrix_similarity),
                "matching_elements": matching_elements,
                "total_elements": total_elements,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "metadata_similarity": float(metadata_similarity),
            })
            
        except Exception as e:
            logger.error(f"Approximate comparison failed: {e}")
            self.result.error = f"Approximate comparison failed: {e}"
    
    def _compare_summary(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Summary statistics comparison: component-based averaging of shape, sparsity, matrix correlation,
        cluster ARI, and embedding correlation.
        
        Each available component contributes equally to the overall similarity.
        Components that are not available (e.g., no clusters, no embeddings) are skipped.
        
        Args:
            **kwargs: Strategy-specific parameters
            - reduction: Reduction key in obsm to compare (default: "X_umap")
            - cluster_col: Cluster column name in obs (default: None, auto-detect)
            - layer: Layer name to compare (default: None, uses X matrix)
        """
        try:
            reduction = kwargs.get('reduction', 'X_umap')
            cluster_col = kwargs.get('cluster_col', None)
            layer = kwargs.get('layer', None)
            
            # Load AnnData objects
            ref_adata = self._load_file(ref_validation.file_path, ref_validation.signature)
            alt_adata = self._load_file(alt_validation.file_path, alt_validation.signature)
            
            similarities = []
            details = {}
            
            # 1. Shape similarity: using relative absolute error
            obs_shape_sim = float(rae_similarity([ref_adata.n_obs], [alt_adata.n_obs])[0])
            var_shape_sim = float(rae_similarity([ref_adata.n_vars], [alt_adata.n_vars])[0])
            shape_sim = (obs_shape_sim + var_shape_sim) / 2.0
            similarities.append(shape_sim)
            details['ref_shape'] = (ref_adata.n_obs, ref_adata.n_vars)
            details['alt_shape'] = (alt_adata.n_obs, alt_adata.n_vars)
            details['shape_similarity'] = float(shape_sim)
            
            # 2. Sparsity similarity: 1 - |sparsity_ref - sparsity_alt|
            ref_sparsity = self._get_sparsity(ref_adata, layer=layer)
            alt_sparsity = self._get_sparsity(alt_adata, layer=layer)
            sparsity_sim = 1.0 - abs(ref_sparsity - alt_sparsity)
            similarities.append(sparsity_sim)
            details['sparsity_similarity'] = float(sparsity_sim)
            details['ref_sparsity'] = float(ref_sparsity)
            details['alt_sparsity'] = float(alt_sparsity)
            
            # 3. Matrix correlation: mean per-gene Pearson correlation across common genes
            ref_aligned, alt_aligned, common_vars, common_obs = self._align_matrices(
                ref_adata, alt_adata, layer=layer
            )
            
            if ref_aligned is not None and len(common_vars) > 0:
                # Calculate per-gene (column) Pearson correlation
                # ref_aligned and alt_aligned are (n_common_obs, n_common_vars)
                gene_corrs = []
                for j in range(len(common_vars)):
                    ref_vec = ref_aligned[:, j]
                    alt_vec = alt_aligned[:, j]
                    corr = pearson_correlation(ref_vec, alt_vec)
                    if not np.isnan(corr):
                        gene_corrs.append(corr)
                
                matrix_corr = float(np.mean(gene_corrs)) if gene_corrs else 0.0
                similarities.append(matrix_corr)
                details['matrix_correlation'] = matrix_corr
                details['num_genes_correlated'] = len(gene_corrs)
                details['num_common_vars'] = len(common_vars)
                details['num_common_obs'] = len(common_obs)
            
            # 4. Cluster ARI (Adjusted Rand Index)
            ref_clusters = self._get_clusters(ref_adata, cluster_col)
            alt_clusters = self._get_clusters(alt_adata, cluster_col)
            
            if ref_clusters is not None and alt_clusters is not None:
                cluster_common_obs = ref_clusters.index.intersection(alt_clusters.index)
                if len(cluster_common_obs) > 0:
                    ref_clust_aligned = ref_clusters.loc[cluster_common_obs]
                    alt_clust_aligned = alt_clusters.loc[cluster_common_obs]
                    
                    ari = adjusted_rand_score(ref_clust_aligned.values, alt_clust_aligned.values)
                    # ARI ranges from -0.5 to 1.0; clamp to [0, 1] for similarity
                    ari_sim = max(0.0, float(ari))
                    similarities.append(ari_sim)
                    details['ari'] = float(ari)
                    details['num_clusters_ref'] = len(ref_clust_aligned.unique())
                    details['num_clusters_alt'] = len(alt_clust_aligned.unique())
                    details['num_cluster_obs'] = len(cluster_common_obs)
            
            # 5. Embedding correlation: mean per-dimension Pearson correlation
            ref_embed = self._get_embeddings(ref_adata, key=reduction)
            alt_embed = self._get_embeddings(alt_adata, key=reduction)
            
            if ref_embed is not None and alt_embed is not None:
                embed_common_obs = ref_embed.index.intersection(alt_embed.index)
                embed_common_dims = ref_embed.columns.intersection(alt_embed.columns)
                
                if len(embed_common_obs) > 0 and len(embed_common_dims) > 0:
                    ref_embed_aligned = ref_embed.loc[embed_common_obs, embed_common_dims]
                    alt_embed_aligned = alt_embed.loc[embed_common_obs, embed_common_dims]
                    
                    dim_corrs = []
                    for dim in embed_common_dims:
                        ref_vec = ref_embed_aligned[dim].values
                        alt_vec = alt_embed_aligned[dim].values
                        corr = pearson_correlation(ref_vec, alt_vec)
                        if not np.isnan(corr):
                            dim_corrs.append(corr)
                    
                    embed_sim = float(np.mean(dim_corrs)) if dim_corrs else 0.0
                    similarities.append(embed_sim)
                    details['embedding_correlation'] = embed_sim
                    details['reduction'] = reduction
                    details['num_dimensions'] = len(embed_common_dims)
            
            # Overall similarity: average of all available components
            similarity = float(np.mean(similarities)) if similarities else 0.0
            self.result.similarity = similarity
            self.result.details.update(details)
            
        except Exception as e:
            logger.error(f"Summary statistics comparison failed: {e}")
            self.result.error = f"Summary statistics comparison failed: {e}"
