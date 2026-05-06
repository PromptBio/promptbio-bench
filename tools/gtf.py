"""GTF/GFF (Gene Transfer Format / General Feature Format) handler.

Strategy:
- exact: Exact feature matching by ID, all fields must match (100% required)
- approximate: Feature overlap with positional tolerance, Jaccard similarity
- summary: Summary statistics comparison of feature counts, types, and length distributions

Required cmdline tools:
- (None)
"""

import bisect
import logging
import os
import tempfile
from typing import Dict, Any, Optional, Set, Tuple, Union
from collections import Counter

import numpy as np
import pandas as pd
import gffutils

from tools.base import FormatHandler
from tools.results import ValidationResult, ComparisonResult
from utils.format import detect_file_signature
from utils.metrics import hellinger_distance, jaccard_similarity, precision_recall_f1, rae_similarity

logger = logging.getLogger(__name__)


class GTFHandler(FormatHandler):
    """Handler for GTF and GFF format files."""
    
    def __init__(self):
        super().__init__("gtf")
        self.format_map = {
            "gtf": {"mode": "r"},
            "gff": {"mode": "r"},
            "gff3": {"mode": "r"},
        }
        self.supported_formats = list(self.format_map.keys())
        self.supported_strategies = ["exact", "approximate", "summary"]
    
    # ============================================================================
    # Public API Methods
    # ============================================================================
    
    def validate(self, file_path: str, **kwargs) -> ValidationResult:
        """Validate GTF/GFF file format.
        non-empty check is performed by file size, later can be updated by reading content if needed
        
        Args:
            file_path: Path to GTF/GFF file
            **kwargs: optional parameters (e.g., min_size)
            - min_size: Minimum size of the file in bytes, default is 0
        """
        # check existence and non-emptiness (by file size)
        is_exists = self._is_file_exists(file_path)
        is_nonempty = self._is_file_nonempty(file_path, min_size=kwargs.get("min_size", 0))
        
        # check file format
        signature = detect_file_signature(file_path, fallback=True)
        is_format_ok = signature.format in self.supported_formats or signature.format == 'unknown'

        # initialize validation result
        result = ValidationResult(exists=is_exists, non_empty=is_nonempty, format_ok=is_format_ok, parseable=False, 
                                  file_path=file_path, signature=signature, details={}, error=None)

        if not is_exists:
            logger.error(f"GTF/GFF file not found: {file_path}")
            result.error = "File not found"
            return result
        
        if not is_nonempty:
            logger.warning(f"GTF/GFF file is empty: {file_path}")
            result.error = "File is empty"
            return result
        
        if not is_format_ok:
            logger.error(f"Not a GTF/GFF file: detected={signature.format}")
            result.error = f"Not a GTF/GFF file: detected={signature.format}"
            return result
        
        # validate by parsing the GTF/GFF file
        try:
            # Create temporary database to validate and parse GTF/GFF file
            db = self._load_file(file_path)
            
            # Get statistics
            stats = self._get_gtf_stats(db)
            result.parseable = True
            result.non_empty = stats["num_features"] > 0
            
            if not result.non_empty:
                logger.warning(f"GTF/GFF file contains no valid features: {file_path}")
                result.error = "GTF/GFF file contains no valid features"
                return result
            
            result.details.update({
                "num_features": stats["num_features"],
                "format": signature.format,
                "total_length": stats["total_length"],
                "avg_length": stats["avg_length"],
                "chromosomes": sorted(list(stats["chromosomes"])),
                "num_chromosomes": len(stats["chromosomes"]),
                "feature_types": dict(stats["feature_types"]),
                "num_feature_types": len(stats["feature_types"]),
                "sources": sorted(list(stats["sources"])),
                "num_sources": len(stats["sources"])
            })
            
            logger.info(f"GTF/GFF validation passed: {stats['num_features']} features, {stats['total_length']} bp total")
        except Exception as e:
            logger.error(f"GTF/GFF validation failed: {e}")
            result.error = f"GTF/GFF validation failed: {e}"
        return result
    
    def compare(self, reference_path: str, candidate_path: str, strategy: str = "summary", **kwargs) -> ComparisonResult:
        """Compare GTF/GFF files.
        
        Supported strategies:
        - exact: Exact feature matching, all features identical (100% required)
        - approximate: Feature overlap with positional tolerance, Jaccard similarity
        - summary: Summary statistics comparison of feature counts, types, and length distributions
        
        Args:
            reference_path: Path to reference GTF/GFF file
            candidate_path: Path to candidate GTF/GFF file
            strategy: Comparison strategy (exact, approximate, summary)
            **kwargs: Strategy-specific parameters
            - tolerance: Positional tolerance in base pairs for approximate comparison (default: 0)
            - overlap_threshold: Minimum overlap ratio to consider features matched (default: 0.95)
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
                    # Exact feature matching, all features identical (100% required)
                    self._compare_exact(ref_validation, alt_validation, **kwargs)
                elif strategy == "approximate":
                    # Feature overlap with positional tolerance, Jaccard similarity
                    self._compare_approx(ref_validation, alt_validation, **kwargs)
                elif strategy == "summary":
                    # Summary statistics comparison: feature counts, types, length distributions
                    self._compare_summary(ref_validation, alt_validation, **kwargs)
                else:
                    logger.warning(f"Unknown strategy '{strategy}', fall back to summary")
                    self._compare_summary(ref_validation, alt_validation, **kwargs)
        except Exception as e:
            logger.error(f"GTF/GFF comparison failed: {e}")
            self.result.error = f"GTF/GFF comparison failed: {e}"
        return self._finalize_output(ref_validation, alt_validation, self.result)

    # ============================================================================
    # File I/O and Loading Methods
    # ============================================================================
    
    def _load_file(self, file_path: str, return_db_path: bool = False) -> Union[gffutils.FeatureDB, Tuple[gffutils.FeatureDB, str]]:
        """Load GTF/GFF file using gffutils.
        
        Args:
            file_path: Path to GTF/GFF file
            return_db_path: If True, return both FeatureDB and db_path as tuple (for cleanup)
        
        Returns:
            gffutils.FeatureDB object, or tuple of (FeatureDB, db_path) if return_db_path=True
        """
        if not self._is_file_exists(file_path):
            raise FileNotFoundError(f"GTF/GFF file not found: {file_path}")
        
        db_path = self._create_db(file_path)
        db = gffutils.FeatureDB(db_path)
        
        if return_db_path:
            return db, db_path
        return db
    
    # ============================================================================
    # File Status Checkers
    # ============================================================================
    
    # ============================================================================
    # File Creation Methods
    # ============================================================================
    
    def _create_db(self, file_path: str) -> str:
        """Create a temporary gffutils database from GTF/GFF file.
        Returns path to temporary database file (caller must clean up after use).
        """
        # Create temporary database file
        fd, db_path = tempfile.mkstemp(suffix='.db', prefix='gffutils_')
        os.close(fd)  # Close file descriptor
        
        try:
            # Create database with gffutils
            # force=True: overwrite existing database
            # keep_order=True: preserve feature order
            gffutils.create_db(file_path, dbfn=db_path, force=True, keep_order=True, 
                              merge_strategy='merge', sort_attribute_values=True)
            
            if not os.path.exists(db_path):
                raise RuntimeError(f"Database file was not created: {db_path}")
            
            return db_path
        except Exception as e:
            self._cleanup_file(db_path)
            raise RuntimeError(f"Failed to create gffutils database: {e}")
    
    # ============================================================================
    # Statistics and Analysis Methods
    # ============================================================================
    
    def _extract_features_df(self, db: gffutils.FeatureDB) -> pd.DataFrame:
        """Convert gffutils.FeatureDB to pandas DataFrame with core feature columns.
        
        Similar to BED's _extract_bed_df, provides a tabular representation of all
        features for efficient analysis and comparison operations.
        
        Args:
            db: gffutils.FeatureDB object
        
        Returns:
            DataFrame with columns: feature_id, seqid, source, featuretype, start, end,
            score, strand, frame
        """
        data = []
        for f in db.all_features():
            if f.seqid and f.start is not None and f.end is not None:
                row = {
                    'feature_id': f.id if f.id else f"{f.seqid}:{f.start}-{f.end}:{f.featuretype}",
                    'seqid': f.seqid,
                    'source': f.source if f.source else '.',
                    'featuretype': f.featuretype,
                    'start': int(f.start),
                    'end': int(f.end),
                    'score': f.score,
                    'strand': f.strand if f.strand else '.',
                    'frame': f.frame,
                }
                data.append(row)
        df = pd.DataFrame(data)
        if not df.empty:
            df['start'] = df['start'].astype(int)
            df['end'] = df['end'].astype(int)
        return df
    
    def _get_gtf_stats(self, db: gffutils.FeatureDB) -> Dict[str, Any]:
        """Extract statistics from GTF/GFF database.
        
        Args:
            db: gffutils.FeatureDB object
        
        Returns:
            Dictionary containing num_features, total_length, avg_length, chromosomes, feature_types, sources
        """
        feature_count = 0
        total_length = 0
        chromosomes = set()
        feature_types = Counter()
        sources = set()
        
        for feature in db.all_features():
            if feature.seqid and feature.start is not None and feature.end is not None:
                feature_count += 1
                feature_length = feature.end - feature.start + 1  # GTF/GFF is 1-based, inclusive
                total_length += feature_length
                chromosomes.add(feature.seqid)
                feature_types[feature.featuretype] += 1
                if feature.source:
                    sources.add(feature.source)
        
        avg_length = total_length / feature_count if feature_count > 0 else 0
        
        return {
            "num_features": feature_count,
            "total_length": total_length,
            "avg_length": avg_length,
            "chromosomes": chromosomes,
            "feature_types": feature_types,
            "sources": sources
        }
    
    def _get_length_counter(self, df: pd.DataFrame) -> Counter:
        """Get feature length distribution from pandas DataFrame.
        Returns the Counter mapping length (in bp) to count of features with that length.
        """
        if df.empty:
            return Counter()
        lengths = (df['end'] - df['start'] + 1).astype(int)  # GTF/GFF is 1-based, inclusive
        return Counter(lengths)
    
    def _get_feature_dict(self, db: gffutils.FeatureDB) -> Dict[str, gffutils.Feature]:
        """Extract feature dictionary from database.
        
        Args:
            db: gffutils.FeatureDB object
        
        Returns:
            Dictionary mapping feature ID to feature object
        """
        features = {}
        for f in db.all_features():
            # Use ID if available, otherwise create a unique key from coordinates
            if f.id:
                features[f.id] = f
            else:
                # Create unique key from coordinates and type
                key = f"{f.seqid}:{f.start}-{f.end}:{f.featuretype}"
                features[key] = f
        return features
    
    def _get_feature_attributes(self, feature: gffutils.Feature) -> Dict[str, Any]:
        """Extract normalized attributes from feature.
        
        Args:
            feature: gffutils.Feature object
        
        Returns:
            Dictionary of normalized attributes
        """
        # Normalize attributes: convert lists to tuples for hashability
        attrs = {}
        for key, value in feature.attributes.items():
            if isinstance(value, list):
                attrs[key] = tuple(sorted(value))
            else:
                attrs[key] = value
        return attrs
    
    # ============================================================================
    # Feature Comparison Helpers
    # ============================================================================
    
    def _compare_feature_fields(self, ref_feature: gffutils.Feature, alt_feature: gffutils.Feature) -> bool:
        """Compare mandatory fields of two features.
        
        Args:
            ref_feature: Reference feature
            alt_feature: Candidate feature
        
        Returns:
            True if all mandatory fields match, False otherwise
        """
        if ref_feature.seqid != alt_feature.seqid:
            return False
        if ref_feature.source != alt_feature.source:
            return False
        if ref_feature.featuretype != alt_feature.featuretype:
            return False
        if ref_feature.start != alt_feature.start:
            return False
        if ref_feature.end != alt_feature.end:
            return False
        # Score, strand, and frame can be None, handle None comparison
        if (ref_feature.score is None) != (alt_feature.score is None):
            return False
        if ref_feature.score is not None and ref_feature.score != alt_feature.score:
            return False
        if ref_feature.strand != alt_feature.strand:
            return False
        if (ref_feature.frame is None) != (alt_feature.frame is None):
            return False
        if ref_feature.frame is not None and ref_feature.frame != alt_feature.frame:
            return False
        
        # Compare attributes
        ref_attrs = self._get_feature_attributes(ref_feature)
        alt_attrs = self._get_feature_attributes(alt_feature)
        if ref_attrs != alt_attrs:
            return False
        
        return True
    
    def _get_overlap_features(self, ref_df: pd.DataFrame, alt_df: pd.DataFrame,
                              overlap_threshold: float = 0.95, tolerance: int = 0) -> Tuple[int, int]:
        """Find overlapping features between ref and alt DataFrames.
        
        Features are matched by (seqid, featuretype) group and coordinate overlap.
        A feature is considered matched if the overlap ratio relative to the
        reference feature length meets the overlap_threshold.
        
        Uses binary search on sorted alt start positions for efficient overlap detection.
        
        Args:
            ref_df: Reference features DataFrame (must have seqid, featuretype, start, end columns)
            alt_df: Candidate features DataFrame
            overlap_threshold: Minimum overlap ratio relative to reference feature length (default: 0.95)
            tolerance: Positional tolerance in base pairs to extend intervals (default: 0)
        
        Returns:
            Tuple of (ref_matched_count, alt_matched_count)
        """
        if ref_df.empty or alt_df.empty:
            return 0, 0
        
        ref_matched_indices = set()
        alt_matched_indices = set()
        
        # Group by (seqid, featuretype) for efficient matching
        ref_grouped = ref_df.groupby(['seqid', 'featuretype'])
        alt_grouped_dict = dict(list(alt_df.groupby(['seqid', 'featuretype'])))
        
        for key, ref_group in ref_grouped:
            if key not in alt_grouped_dict:
                continue
            alt_group = alt_grouped_dict[key]
            
            # Sort alt by start for binary search
            alt_sorted = alt_group.sort_values('start')
            alt_starts = alt_sorted['start'].values
            alt_ends = alt_sorted['end'].values
            alt_indices = alt_sorted.index.values
            
            for ref_idx in ref_group.index:
                ref_row = ref_group.loc[ref_idx]
                ref_start = ref_row['start'] - tolerance
                ref_end = ref_row['end'] + tolerance
                ref_length = ref_row['end'] - ref_row['start'] + 1  # Original length, 1-based inclusive
                
                if ref_length <= 0:
                    continue
                
                # Binary search: find alt features where start <= ref_end
                right = bisect.bisect_right(alt_starts, ref_end)
                
                for j in range(right):
                    # Skip if alt ends before ref starts
                    if alt_ends[j] < ref_start:
                        continue
                    
                    # Calculate overlap (1-based inclusive coordinates)
                    overlap_start = max(ref_start, alt_starts[j])
                    overlap_end = min(ref_end, alt_ends[j])
                    overlap_bp = overlap_end - overlap_start + 1
                    
                    if overlap_bp > 0:
                        overlap_ratio = overlap_bp / ref_length
                        if overlap_ratio >= overlap_threshold:
                            ref_matched_indices.add(ref_idx)
                            alt_matched_indices.add(alt_indices[j])
        
        return len(ref_matched_indices), len(alt_matched_indices)
    
    # ============================================================================
    # Private Comparison Implementation Methods
    # ============================================================================
    
    def _compare_exact(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Exact comparison: All features identical.
        
        Compares features by ID and all fields must match exactly.
        
        Criteria:
        - All features identical (same ID, coordinates, attributes)
        - Must be 100% identical
        """
        ref_db_path = None
        alt_db_path = None
        
        try:
            # Load databases
            ref_db, ref_db_path = self._load_file(ref_validation.file_path, return_db_path=True)
            alt_db, alt_db_path = self._load_file(alt_validation.file_path, return_db_path=True)
            
            # Get feature dictionaries
            ref_features = self._get_feature_dict(ref_db)
            alt_features = self._get_feature_dict(alt_db)
            
            ref_count = len(ref_features)
            alt_count = len(alt_features)
            
            # Quick check: compare feature counts
            if ref_count != alt_count:
                self.result.similarity = 0.0
                self.result.details.update({
                    "total_ref_features": ref_count,
                    "total_alt_features": alt_count,
                })
                return
            
            # Compare features by ID or coordinate-based key
            # Use union of both feature sets for total count
            all_feature_ids = set(ref_features.keys()) | set(alt_features.keys())
            total_features = len(all_feature_ids)
            identical_features = 0
            mismatched_features = 0
            
            for feature_id in all_feature_ids:
                if feature_id not in ref_features or feature_id not in alt_features:
                    # Feature exists in one but not the other
                    mismatched_features += 1
                    continue
                
                ref_feature = ref_features[feature_id]
                alt_feature = alt_features[feature_id]
                
                if self._compare_feature_fields(ref_feature, alt_feature):
                    identical_features += 1
                else:
                    mismatched_features += 1
            
            # Calculate similarity (must be 100% for exact match)
            similarity = 1.0 if identical_features == total_features else 0.0
            
            self.result.similarity = similarity
            self.result.details.update({
                "total_ref_features": ref_count,
                "total_alt_features": alt_count,
                "total_features": total_features,
                "identical_features": identical_features,
                "mismatched_features": mismatched_features,
            })
        
        except Exception as e:
            logger.error(f"Exact comparison failed: {e}")
            self.result.error = f"Exact comparison failed: {e}"
        finally:
            self._cleanup_file(ref_db_path, verbose=True)
            self._cleanup_file(alt_db_path, verbose=True)
    
    def _compare_approx(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Approximate comparison: Feature overlap with positional tolerance.
        
        Calculates percentage of features that match within tolerance.
        Useful for gene annotation comparisons where coordinates may differ slightly.
        Counts overlapping features (by seqid and featuretype) and calculates
        precision, recall, and Jaccard similarity.
        
        Args:
            ref_validation: Validation result for reference file
            alt_validation: Validation result for candidate file
            **kwargs: Strategy-specific parameters
            - tolerance: Positional tolerance in base pairs (default: 0)
            - overlap_threshold: Minimum overlap ratio relative to reference feature
              to consider features matched (default: 0.95)
        """
        tolerance = kwargs.get("tolerance", 0)
        overlap_threshold = kwargs.get("overlap_threshold", 0.95)
        
        ref_db_path = None
        alt_db_path = None
        
        try:
            # Load databases
            ref_db, ref_db_path = self._load_file(ref_validation.file_path, return_db_path=True)
            alt_db, alt_db_path = self._load_file(alt_validation.file_path, return_db_path=True)
            
            # Extract features as DataFrames for efficient overlap detection
            ref_df = self._extract_features_df(ref_db)
            alt_df = self._extract_features_df(alt_db)
            
            ref_count = len(ref_df)
            alt_count = len(alt_df)
            
            if ref_count == 0:
                self.result.similarity = 0.0
                self.result.details["ref_feature_count"] = 0
                return
            if alt_count == 0:
                self.result.similarity = 0.0
                self.result.details["alt_feature_count"] = 0
                return
            
            # Find overlapping features and count unique matches on each side
            ref_matched_count, alt_matched_count = self._get_overlap_features(
                ref_df, alt_df, overlap_threshold=overlap_threshold, tolerance=tolerance
            )
            
            # Calculate precision and recall
            precision = alt_matched_count / alt_count if alt_count > 0 else 0.0
            recall = ref_matched_count / ref_count if ref_count > 0 else 0.0
            
            # Calculate Jaccard similarity: |A ∩ B| / |A ∪ B|
            jaccard = jaccard_similarity(ref_count, alt_count, ref_matched_count)
            
            self.result.similarity = jaccard
            self.result.details.update({
                "total_ref_features": ref_count,
                "total_alt_features": alt_count,
                "ref_matched_features": ref_matched_count,
                "alt_matched_features": alt_matched_count,
                "precision": precision,
                "recall": recall,
                "jaccard_similarity": jaccard,
                "tolerance": tolerance,
                "overlap_threshold": overlap_threshold,
            })
        except Exception as e:
            logger.error(f"Approximate comparison failed: {e}")
            self.result.error = f"Approximate comparison failed: {e}"
        finally:
            self._cleanup_file(ref_db_path, verbose=True)
            self._cleanup_file(alt_db_path, verbose=True)
    
    def _compare_summary(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Summary statistics comparison: Feature counts, types, and length distributions.
        
        Compares:
        - Number of features using RAE-based similarity
        - Average feature length using RAE-based similarity
        - Feature length distributions using Hellinger distance
        - Feature type distributions using Hellinger distance
        
        Args:
            ref_validation: Validation result for reference file
            alt_validation: Validation result for candidate file
            **kwargs: Strategy-specific parameters (not used for summary)
        """
        ref_db_path = None
        alt_db_path = None
        
        try:
            # Load databases
            ref_db, ref_db_path = self._load_file(ref_validation.file_path, return_db_path=True)
            alt_db, alt_db_path = self._load_file(alt_validation.file_path, return_db_path=True)
            
            # Extract features as DataFrames
            ref_df = self._extract_features_df(ref_db)
            alt_df = self._extract_features_df(alt_db)
            
            # Get statistics from databases
            ref_stats = self._get_gtf_stats(ref_db)
            alt_stats = self._get_gtf_stats(alt_db)
            
            # Compare feature counts using RAE-based similarity
            ref_count = ref_stats["num_features"]
            alt_count = alt_stats["num_features"]
            feature_count_similarity = float(rae_similarity([ref_count], [alt_count])[0])
            
            # Compare average feature length using RAE-based similarity
            ref_avg_length = ref_stats["avg_length"]
            alt_avg_length = alt_stats["avg_length"]
            avg_length_similarity = float(rae_similarity([ref_avg_length], [alt_avg_length])[0])
            
            # Compare feature length distributions using Hellinger distance
            ref_length_counter = self._get_length_counter(ref_df)
            alt_length_counter = self._get_length_counter(alt_df)
            all_lengths = sorted(set(ref_length_counter.keys()) | set(alt_length_counter.keys()))
            
            if all_lengths:
                ref_length_arr = np.array([ref_length_counter.get(k, 0) for k in all_lengths], dtype=float)
                alt_length_arr = np.array([alt_length_counter.get(k, 0) for k in all_lengths], dtype=float)
                length_hellinger = hellinger_distance(ref_length_arr, alt_length_arr)
                length_dist_similarity = 1.0 - (length_hellinger if not np.isnan(length_hellinger) else 1.0)
            else:
                length_dist_similarity = 1.0 if ref_count == alt_count == 0 else 0.0
            
            # Compare feature type distributions using Hellinger distance
            ref_types = ref_stats["feature_types"]
            alt_types = alt_stats["feature_types"]
            all_types = sorted(set(ref_types.keys()) | set(alt_types.keys()))
            
            if all_types:
                ref_type_arr = np.array([ref_types.get(t, 0) for t in all_types], dtype=float)
                alt_type_arr = np.array([alt_types.get(t, 0) for t in all_types], dtype=float)
                type_hellinger = hellinger_distance(ref_type_arr, alt_type_arr)
                type_dist_similarity = 1.0 - (type_hellinger if not np.isnan(type_hellinger) else 1.0)
            else:
                type_dist_similarity = 1.0 if ref_count == alt_count == 0 else 0.0
            
            # Use feature length distribution similarity as the final similarity score
            # (aligned with BED handler's approach)
            self.result.similarity = length_dist_similarity
            self.result.details.update({
                "ref_num_features": ref_count,
                "alt_num_features": alt_count,
                "feature_count_similarity": feature_count_similarity,
                "ref_avg_length": ref_avg_length,
                "alt_avg_length": alt_avg_length,
                "avg_length_similarity": avg_length_similarity,
                "length_dist_similarity": length_dist_similarity,
                "type_dist_similarity": type_dist_similarity,
                "ref_feature_types": dict(ref_types),
                "alt_feature_types": dict(alt_types),
            })
        except Exception as e:
            logger.error(f"Summary statistics comparison failed: {e}")
            self.result.error = f"Summary statistics comparison failed: {e}"
        finally:
            self._cleanup_file(ref_db_path, verbose=True)
            self._cleanup_file(alt_db_path, verbose=True)
