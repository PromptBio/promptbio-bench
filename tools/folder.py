"""Folder/Directory structure handler.

Strategy:
- exact: Exact file structure and content matching, all files identical (100% required)
- approximate: Matches files by relative path; each file pair requires content similarity >= threshold for concordance
- summary: Summary statistics comparison of file counts, sizes, extension distributions

Required cmdline tools:
- tree (optional, falls back to manual traversal)
"""

import hashlib
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter

import numpy as np

from tools.base import FormatHandler
from tools.results import ValidationResult, ComparisonResult
from utils.format import detect_file_signature
from utils.metrics import hellinger_distance, jaccard_similarity, min_max_similarity, precision_recall_f1

logger = logging.getLogger(__name__)


class FolderHandler(FormatHandler):
    """Handler for folder/directory structures.
    
    This handler is useful for:
    - STAR genome index directories
    - Reference genome directories
    - Any multi-file output structures
    """
    
    def __init__(self):
        super().__init__("folder")
        self.supported_formats = ["folder"]
        self.supported_strategies = ["exact", "approximate", "summary"]
    
    # ============================================================================
    # Public API Methods
    # ============================================================================
    
    def validate(self, file_path: str, **kwargs) -> ValidationResult:
        """Validate folder structure.
        non-empty check is performed by file size, later can be updated by reading content if needed
        
        Args:
            file_path: Path to directory
            **kwargs: optional parameters (e.g., min_size)
            - min_size: Minimum size of the folder in bytes, default is 0
        """
        # check existence and non-emptiness (by file size)
        is_exists = self._is_file_exists(file_path)
        is_nonempty = self._is_file_nonempty(file_path, min_size=kwargs.get("min_size", 0))
        
        # check file format (should be a directory)
        signature = detect_file_signature(file_path, fallback=True)
        is_format_ok = self._is_dir_exists(file_path) if is_exists else False

        # initialize validation result
        result = ValidationResult(exists=is_exists, non_empty=is_nonempty, format_ok=is_format_ok, parseable=False, 
                                  file_path=file_path, signature=signature, details={}, error=None)

        if not is_exists:
            logger.error(f"Folder not found: {file_path}")
            result.error = "Folder not found"
            return result
        
        if not is_format_ok:
            logger.error(f"Path is not a directory: {file_path}")
            result.error = "Path is not a directory"
            return result
        
        if not is_nonempty:
            logger.warning(f"Folder is empty: {file_path}")
            result.error = "Folder is empty"
            return result
        
        # validate by parsing the folder structure
        try:
            file_count = 0
            dir_count = 0
            total_size = 0
            file_list = []
            file_extensions = defaultdict(int)
            base_path = file_path
            
            for root, dirs, files in os.walk(base_path):
                dir_count += len(dirs)
                for file in files:
                    file_count += 1
                    full_file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_file_path, base_path)
                    file_list.append(rel_path)
                    
                    ext = Path(file).suffix.lower()
                    if ext:
                        file_extensions[ext] += 1
                    
                    try:
                        total_size += os.path.getsize(full_file_path)
                    except OSError:
                        pass
            
            result.parseable = True
            result.non_empty = file_count > 0
            
            if not result.non_empty:
                logger.warning(f"Folder contains no files: {file_path}")
                result.error = "Folder contains no files"
                result.details.update({
                    "file_count": 0,
                    "dir_count": dir_count,
                    "total_size": 0
                })
            else:
                result.details.update({
                    "file_count": file_count,
                    "dir_count": dir_count,
                    "total_size": total_size,
                    "files": sorted(file_list),
                    "extensions": dict(file_extensions)
                })
                logger.info(f"Folder validation passed: {file_count} files, {dir_count} subdirs, {total_size} bytes")
        except Exception as e:
            logger.error(f"Folder validation failed: {e}")
            result.error = f"Folder validation failed: {e}"
        return result
    
    def compare(self, reference_path: str, candidate_path: str, strategy: str = "exact", **kwargs) -> ComparisonResult:
        """Compare two folder structures.
        
        Supported strategies:
        - exact: Exact file structure and content must match (all files identical)
        - approximate: Matches files by relative path; compares content with threshold for concordance
        - summary: Summary statistics comparison of file counts, sizes, extension distributions
        
        Args:
            reference_path: Path to reference folder
            candidate_path: Path to candidate folder
            strategy: Comparison strategy (exact, approximate, summary)
            **kwargs: Strategy-specific parameters
            - ignore_patterns: List of patterns to ignore (e.g., ['.git', '__pycache__'])
            - content_threshold: Content similarity threshold for approximate concordance (default: 0.95)
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
                    self._compare_exact(ref_validation, alt_validation, **kwargs)
                elif strategy == "approximate":
                    self._compare_approx(ref_validation, alt_validation, **kwargs)
                elif strategy == "summary":
                    self._compare_summary(ref_validation, alt_validation, **kwargs)
                else:
                    logger.warning(f"Unknown strategy '{strategy}', fall back to summary")
                    self._compare_summary(ref_validation, alt_validation, **kwargs)
        except Exception as e:
            logger.error(f"Folder comparison failed: {e}")
            self.result.error = f"Folder comparison failed: {e}"
        return self._finalize_output(ref_validation, alt_validation, self.result)
    
    # ============================================================================
    # Statistics and Analysis Methods
    # ============================================================================
    
    def _get_file_structure(
        self,
        folder_path: str,
        ignore_patterns: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get folder structure as dictionary of relative_path -> file info.
        
        Returns:
            Dictionary with relative file paths as keys and metadata dict as values.
            Each value: {'size': int, 'extension': str, 'basename': str, 'full_path': str}
        """
        ignore_patterns = ignore_patterns or []
        files = {}
        
        for root, dirs, filenames in os.walk(folder_path):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in ignore_patterns)]
            
            for filename in filenames:
                # Skip ignored patterns
                if any(pattern in filename for pattern in ignore_patterns):
                    continue
                
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, folder_path)
                
                try:
                    size = os.path.getsize(file_path)
                    ext = Path(filename).suffix.lower()
                    
                    files[rel_path] = {
                        'size': size,
                        'extension': ext,
                        'basename': Path(filename).stem,
                        'full_path': file_path
                    }
                except OSError:
                    pass
        
        return files
    
    def _get_tree_string(
        self,
        folder_path: str,
        ignore_patterns: Optional[List[str]] = None
    ) -> str:
        """Get folder tree structure as a string using the tree command.
        Falls back to manual generation if tree command is not available.
        
        Args:
            folder_path: Path to folder
            ignore_patterns: List of patterns to ignore
            
        Returns:
            Tree structure as string
        """
        try:
            cmd = ["tree", "-F", "--dirsfirst", folder_path]
            if ignore_patterns:
                for pattern in ignore_patterns:
                    cmd.extend(["-I", pattern])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                return result.stdout
            
            logger.warning(f"tree command failed: {result.stderr}")
            return self._generate_tree_manually(folder_path, ignore_patterns)
        except FileNotFoundError:
            logger.warning("tree command not found, using manual generation")
            return self._generate_tree_manually(folder_path, ignore_patterns)
        except Exception as e:
            logger.error(f"Failed to get tree structure: {e}")
            return ""
    
    def _get_extension_counts(self, files: Dict[str, Dict[str, Any]]) -> Counter:
        """Get file extension count distribution from file structure.
        
        Args:
            files: File structure dict from _get_file_structure
            
        Returns:
            Counter mapping extension -> count
        """
        ext_counter = Counter()
        for info in files.values():
            ext = info['extension'] if info['extension'] else '(no ext)'
            ext_counter[ext] += 1
        return ext_counter
    
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            MD5 hex digest string, or empty string on error
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except IOError:
            return ""
    
    def _get_file_content_similarity(self, file1_path: str, file2_path: str) -> float:
        """Compare contents of two files and return similarity score.
        
        For small files (< 1MB): compares content directly (exact or line-by-line for text).
        For large files (>= 1MB): uses hash for exact match, then samples chunks for partial similarity.
        
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            size1 = os.path.getsize(file1_path)
            size2 = os.path.getsize(file2_path)
            
            if size1 == 0 and size2 == 0:
                return 1.0
            if size1 == 0 or size2 == 0:
                return 0.0
            
            # For large files, use hash-based comparison with chunk sampling
            if size1 > 1024 * 1024 or size2 > 1024 * 1024:  # > 1MB
                hash1 = self._get_file_hash(file1_path)
                hash2 = self._get_file_hash(file2_path)
                if hash1 and hash2 and hash1 == hash2:
                    return 1.0
                
                # Sample chunks to estimate similarity
                return self._get_chunk_similarity(file1_path, file2_path, size1, size2)
            
            # For smaller files, compare content directly
            with open(file1_path, 'rb') as f1, open(file2_path, 'rb') as f2:
                content1 = f1.read()
                content2 = f2.read()
                
                if content1 == content2:
                    return 1.0
                
                # Try text comparison
                try:
                    text1 = content1.decode('utf-8')
                    text2 = content2.decode('utf-8')
                    return self._get_text_similarity(text1, text2)
                except UnicodeDecodeError:
                    # Binary file comparison
                    return self._get_binary_similarity(content1, content2)
        except (OSError, IOError):
            return 0.0
    
    def _get_chunk_similarity(self, file1_path: str, file2_path: str, size1: int, size2: int) -> float:
        """Estimate similarity for large files by sampling chunks at beginning, middle, and end.
        
        Args:
            file1_path, file2_path: Paths to files
            size1, size2: File sizes
            
        Returns:
            Estimated similarity score between 0.0 and 1.0
        """
        chunk_size = 64 * 1024  # 64KB chunks
        max_file_size = max(size1, size2)
        num_chunks = min(10, max_file_size // chunk_size)
        
        if num_chunks == 0:
            return min_max_similarity(float(size1), float(size2))
        
        # Build sample positions: beginning, middle, end
        positions = []
        # Beginning
        for i in range(min(3, num_chunks)):
            positions.append(i * chunk_size)
        # Middle
        if num_chunks > 3:
            mid_start = max_file_size // 2 - chunk_size
            for i in range(min(4, num_chunks - 3)):
                positions.append(mid_start + i * chunk_size)
        # End
        if num_chunks > 7:
            end_start = max_file_size - chunk_size * min(3, num_chunks - 7)
            for i in range(num_chunks - 7):
                positions.append(end_start + i * chunk_size)
        
        matching = 0
        compared = 0
        
        try:
            with open(file1_path, 'rb') as f1, open(file2_path, 'rb') as f2:
                for pos in positions:
                    if pos < size1 and pos < size2:
                        f1.seek(pos)
                        f2.seek(pos)
                        c1 = f1.read(chunk_size)
                        c2 = f2.read(chunk_size)
                        if len(c1) == len(c2):
                            compared += 1
                            if c1 == c2:
                                matching += 1
        except (IOError, OSError):
            return 0.0
        
        if compared > 0:
            chunk_sim = matching / compared
            size_sim = min_max_similarity(float(size1), float(size2))
            return 0.7 * chunk_sim + 0.3 * size_sim
        
        return min_max_similarity(float(size1), float(size2))
    
    def _get_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings.
        
        Combines line-level and character-level similarity.
        
        Returns:
            Similarity score between 0.0 and 1.0
        """
        lines1 = text1.splitlines()
        lines2 = text2.splitlines()
        
        if not lines1 and not lines2:
            return 1.0
        
        # Line-by-line comparison
        total_lines = max(len(lines1), len(lines2))
        common_lines = sum(1 for l1, l2 in zip(lines1, lines2) if l1 == l2)
        line_similarity = common_lines / total_lines if total_lines > 0 else 0.0
        
        # Character-level similarity
        max_len = max(len(text1), len(text2))
        if max_len > 0:
            matching_chars = sum(1 for c1, c2 in zip(text1, text2) if c1 == c2)
            char_similarity = matching_chars / max_len
        else:
            char_similarity = 0.0
        
        return 0.6 * line_similarity + 0.4 * char_similarity
    
    def _get_binary_similarity(self, content1: bytes, content2: bytes) -> float:
        """Calculate similarity between two binary contents.
        
        Returns:
            Similarity score between 0.0 and 1.0
        """
        max_len = max(len(content1), len(content2))
        if max_len == 0:
            return 1.0
        
        common_bytes = sum(1 for b1, b2 in zip(content1, content2) if b1 == b2)
        return common_bytes / max_len
    
    # ============================================================================
    # Tree Structure Helpers
    # ============================================================================
    
    def _generate_tree_manually(
        self,
        folder_path: str,
        ignore_patterns: Optional[List[str]] = None,
        prefix: str = "",
        is_last: bool = True
    ) -> str:
        """Generate tree structure manually when tree command is not available.
        
        Args:
            folder_path: Path to folder
            ignore_patterns: List of patterns to ignore
            prefix: Prefix for tree display
            is_last: Whether this is the last item in the parent
            
        Returns:
            Tree structure as string
        """
        ignore_patterns = ignore_patterns or []
        tree_lines = []
        
        try:
            entries = sorted(os.listdir(folder_path))
            entries = [e for e in entries if not any(p in e for p in ignore_patterns)]
            
            dirs = [e for e in entries if os.path.isdir(os.path.join(folder_path, e))]
            files = [e for e in entries if os.path.isfile(os.path.join(folder_path, e))]
            all_entries = dirs + files
            
            for i, entry in enumerate(all_entries):
                is_last_entry = (i == len(all_entries) - 1)
                entry_path = os.path.join(folder_path, entry)
                
                if is_last_entry:
                    tree_lines.append(f"{prefix}└── {entry}{'/' if os.path.isdir(entry_path) else ''}")
                    new_prefix = prefix + "    "
                else:
                    tree_lines.append(f"{prefix}├── {entry}{'/' if os.path.isdir(entry_path) else ''}")
                    new_prefix = prefix + "│   "
                
                if os.path.isdir(entry_path):
                    subtree = self._generate_tree_manually(
                        entry_path,
                        ignore_patterns,
                        new_prefix,
                        is_last_entry
                    )
                    if subtree:
                        tree_lines.append(subtree)
            
            return "\n".join(tree_lines)
        except Exception as e:
            logger.error(f"Failed to generate tree manually: {e}")
            return ""
    
    # ============================================================================
    # Private Comparison Implementation Methods
    # ============================================================================
    
    def _compare_exact(
        self,
        ref_validation: ValidationResult,
        alt_validation: ValidationResult,
        **kwargs
    ):
        """Exact match: all files must have identical relative paths and identical content.
        
        Criteria:
        - File sets must match exactly (same relative paths)
        - All file contents must be byte-identical (verified by MD5 hash)
        - Must be 100% identical for similarity = 1.0, otherwise 0.0
        
        Args:
            **kwargs: Strategy-specific parameters
            - ignore_patterns: List of patterns to ignore
        """
        try:
            ignore_patterns = kwargs.get('ignore_patterns', [])
            
            ref_files = self._get_file_structure(ref_validation.file_path, ignore_patterns)
            alt_files = self._get_file_structure(alt_validation.file_path, ignore_patterns)
            
            ref_set = set(ref_files.keys())
            alt_set = set(alt_files.keys())
            
            # Quick check: file sets must match exactly
            if ref_set != alt_set:
                ref_only = sorted(ref_set - alt_set)
                alt_only = sorted(alt_set - ref_set)
                self.result.similarity = 0.0
                self.result.details.update({
                    "total_ref_files": len(ref_set),
                    "total_alt_files": len(alt_set),
                    "ref_only_files": ref_only,
                    "alt_only_files": alt_only,
                    "message": "File sets differ"
                })
                return
            
            # File sets match - compare content of all files using hash
            total_files = len(ref_set)
            identical_files = 0
            mismatched_files = []
            
            for rel_path in sorted(ref_set):
                ref_hash = self._get_file_hash(ref_files[rel_path]['full_path'])
                alt_hash = self._get_file_hash(alt_files[rel_path]['full_path'])
                
                if ref_hash and alt_hash and ref_hash == alt_hash:
                    identical_files += 1
                else:
                    mismatched_files.append(rel_path)
            
            # Must be 100% identical (same as BAM exact: 1.0 if all match, 0.0 otherwise)
            similarity = 1.0 if identical_files == total_files else 0.0
            
            self.result.similarity = similarity
            self.result.details.update({
                "total_files": total_files,
                "total_ref_files": len(ref_set),
                "total_alt_files": len(alt_set),
                "identical_files": identical_files,
                "mismatched_files": mismatched_files,
                "ref_only_files": [],
                "alt_only_files": [],
                "message": "All files identical" if similarity == 1.0 else "Some files have different content"
            })
        except Exception as e:
            logger.error(f"Exact comparison failed: {e}")
            self.result.error = f"Exact comparison failed: {e}"
    
    def _compare_approx(
        self,
        ref_validation: ValidationResult,
        alt_validation: ValidationResult,
        **kwargs
    ):
        """Approximate match: matches files by relative path and compares content with threshold.
        
        This method:
        1. Matches files by relative path (common files)
        2. For each matched file pair, computes content similarity
        3. A file pair is "concordant" if content similarity >= threshold
        4. Calculates precision, recall, F1, and Jaccard similarity
        
        Criteria:
        - Match files by relative path (intersection of file sets)
        - For each matched pair, content similarity >= content_threshold counts as concordant
        - Files only in ref or alt are treated as unmatched (reduce Jaccard)
        - Uses Jaccard similarity as the main metric
        
        Args:
            **kwargs: Strategy-specific parameters
            - ignore_patterns: List of patterns to ignore
            - content_threshold: Content similarity threshold for concordance (default: 0.95)
        """
        try:
            ignore_patterns = kwargs.get('ignore_patterns', [])
            content_threshold = kwargs.get('content_threshold', 0.95)
            
            ref_files = self._get_file_structure(ref_validation.file_path, ignore_patterns)
            alt_files = self._get_file_structure(alt_validation.file_path, ignore_patterns)
            
            ref_set = set(ref_files.keys())
            alt_set = set(alt_files.keys())
            
            common_files = ref_set & alt_set
            ref_only = ref_set - alt_set
            alt_only = alt_set - ref_set
            
            # Compare content of common files
            concordant_files = 0
            file_similarities = []
            
            for rel_path in sorted(common_files):
                ref_full = ref_files[rel_path]['full_path']
                alt_full = alt_files[rel_path]['full_path']
                
                content_sim = self._get_file_content_similarity(ref_full, alt_full)
                is_concordant = content_sim >= content_threshold
                file_similarities.append({
                    'file': rel_path,
                    'content_similarity': content_sim,
                    'concordant': is_concordant
                })
                
                if is_concordant:
                    concordant_files += 1
            
            # Calculate precision, recall, F1, and Jaccard (same pattern as BAM _compare_approx)
            total_ref = len(ref_files)
            total_alt = len(alt_files)
            
            precision, recall, f1_score = precision_recall_f1(total_ref, total_alt, concordant_files)
            jaccard = jaccard_similarity(total_ref, total_alt, concordant_files)
            
            # Use Jaccard similarity as the main similarity metric
            self.result.similarity = jaccard
            self.result.details.update({
                "total_ref_files": total_ref,
                "total_alt_files": total_alt,
                "common_files": len(common_files),
                "concordant_files": concordant_files,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "jaccard_similarity": jaccard,
                "content_threshold": content_threshold,
                "file_similarities": file_similarities,
                "ref_only_files": sorted(ref_only),
                "alt_only_files": sorted(alt_only),
            })
        except Exception as e:
            logger.error(f"Approximate comparison failed: {e}")
            self.result.error = f"Approximate comparison failed: {e}"
    
    def _compare_summary(
        self,
        ref_validation: ValidationResult,
        alt_validation: ValidationResult,
        **kwargs
    ):
        """Summary statistics comparison: file count, total size, extension distribution similarity.
        
        Computes:
        1. File count similarity using min/max ratio
        2. Total size similarity using min/max ratio
        3. Extension distribution similarity using Hellinger distance
        
        Overall similarity = 1.0 - nanmean(errors).
        
        Args:
            **kwargs: Strategy-specific parameters
            - ignore_patterns: List of patterns to ignore
        """
        try:
            ignore_patterns = kwargs.get('ignore_patterns', [])
            
            ref_files = self._get_file_structure(ref_validation.file_path, ignore_patterns)
            alt_files = self._get_file_structure(alt_validation.file_path, ignore_patterns)
            
            # File count similarity
            ref_count = len(ref_files)
            alt_count = len(alt_files)
            count_similarity = min_max_similarity(float(ref_count), float(alt_count))
            
            # Total size similarity
            ref_total_size = sum(f['size'] for f in ref_files.values())
            alt_total_size = sum(f['size'] for f in alt_files.values())
            size_similarity = min_max_similarity(float(ref_total_size), float(alt_total_size))
            
            # Extension distribution similarity using Hellinger distance
            ref_ext_counts = self._get_extension_counts(ref_files)
            alt_ext_counts = self._get_extension_counts(alt_files)
            
            all_exts = sorted(set(ref_ext_counts.keys()) | set(alt_ext_counts.keys()))
            ref_ext_arr = np.array([ref_ext_counts.get(ext, 0) for ext in all_exts], dtype=float)
            alt_ext_arr = np.array([alt_ext_counts.get(ext, 0) for ext in all_exts], dtype=float)
            
            ext_hellinger = hellinger_distance(ref_ext_arr, alt_ext_arr)
            
            # Combine errors with nanmean (same pattern as BAM _compare_summary)
            errors = np.array([1.0 - count_similarity, 1.0 - size_similarity, ext_hellinger])
            mean_error = np.nanmean(errors)
            similarity = 1.0 - mean_error if not np.isnan(mean_error) else 0.0
            
            self.result.similarity = similarity
            self.result.details.update({
                "count_similarity": count_similarity,
                "size_similarity": size_similarity,
                "ext_hellinger_distance": float(ext_hellinger) if not np.isnan(ext_hellinger) else None,
                "ref_file_count": ref_count,
                "alt_file_count": alt_count,
                "ref_total_size": ref_total_size,
                "alt_total_size": alt_total_size,
                "ref_extensions": dict(ref_ext_counts),
                "alt_extensions": dict(alt_ext_counts),
            })
        except Exception as e:
            logger.error(f"Summary statistics comparison failed: {e}")
            self.result.error = f"Summary statistics comparison failed: {e}"
