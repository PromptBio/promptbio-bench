"""PDB format handler.

Strategy:
- exact: Direct sequence and structure matching by chain ID or sequence content
- approximate: Alignment-based comparison allowing for sequence and structural variations
- summary: Summary statistics comparison of chain count, residue count, atom count, and residue composition

Required cmdline tools:
- (None)
"""

import logging
import os
import warnings
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter

import numpy as np
from Bio import PDB
from Bio.PDB import PDBIO, Superimposer
from Bio.SeqUtils import seq1
from Bio.Seq import Seq
from Bio.Align import PairwiseAligner

from tools.base import FormatHandler
from tools.results import ValidationResult, ComparisonResult
from utils.format import detect_file_signature
from utils.metrics import hellinger_distance, rae_similarity

# Suppress BioPython warnings
warnings.filterwarnings('ignore', category=PDB.PDBExceptions.PDBConstructionWarning)

logger = logging.getLogger(__name__)


class PDBHandler(FormatHandler):
    """Handler for PDB/PDBx/mmCIF format files."""
    
    def __init__(self):
        super().__init__("pdb")
        self.format_map = {
            'pdb': {'parser': PDB.PDBParser},
            'cif': {'parser': PDB.MMCIFParser},
            'mmcif': {'parser': PDB.MMCIFParser},
        }
        self.format = "pdb"  # default PDB format
        self.supported_formats = list(self.format_map.keys())
        self.supported_strategies = ["exact", "approximate", "summary"]
        self.parser_cache = {}  # Cache parsers to avoid re-initialization
    
    # ============================================================================
    # Public API Methods
    # ============================================================================
    
    def validate(self, file_path: str, **kwargs) -> ValidationResult:
        """Validate PDB/mmCIF file format.
        non-empty check is performed by file size, later can be updated by reading content if needed
        
        Args:
            file_path: Path to PDB/mmCIF file
            **kwargs: optional parameters (e.g., min_size)
            - min_size: Minimum size of the file in bytes, default is 0
        """
        # check existence and non-emptiness (by file size)
        is_exists = self._is_file_exists(file_path)
        is_nonempty = self._is_file_nonempty(file_path, min_size=kwargs.get("min_size", 0))
        
        # check file format
        signature = detect_file_signature(file_path, fallback=True)
        detected_format = signature.format
        
        # Try to detect PDB format from extension if signature doesn't work
        if detected_format not in ['pdb', 'cif']:
            ext = os.path.splitext(file_path)[1].lower().lstrip('.')
            if ext in ['pdb']:
                detected_format = 'pdb'
            elif ext in ['cif', 'mmcif']:
                detected_format = 'cif'
        
        self.format = detected_format if detected_format in self.supported_formats else 'pdb'
        is_format_ok = self.format in self.supported_formats

        # initialize validation result
        result = ValidationResult(exists=is_exists, non_empty=is_nonempty, format_ok=is_format_ok, parseable=False, 
                                  file_path=file_path, signature=signature, details={}, error=None)

        if not is_exists:
            logger.error(f"PDB/mmCIF file not found: {file_path}")
            result.error = "File not found"
            return result
        
        if not is_nonempty:
            logger.warning(f"PDB/mmCIF file is empty: {file_path}")
            result.error = "File is empty"
            return result
        
        if not is_format_ok:
            logger.error(f"Not a PDB/mmCIF file: detected={signature.format}")
            result.error = f"Not a PDB/mmCIF file: detected={signature.format}"
            return result
        
        # validate by parsing the PDB/mmCIF file
        try:
            # Parse structure
            structure = self._load_file(file_path)
            if structure is None:
                result.error = "Failed to parse structure"
                return result
            
            result.parseable = True
            
            # Extract statistics
            num_chains = len(list(structure.get_chains()))
            num_residues = len(list(structure.get_residues()))
            num_atoms = len(list(structure.get_atoms()))
            
            result.non_empty = num_atoms > 0
            
            if not result.non_empty:
                logger.warning(f"PDB/mmCIF file contains no atoms: {file_path}")
                result.error = "PDB/mmCIF file contains no atoms"
                return result
            
            result.details.update({
                "num_chains": num_chains,
                "num_residues": num_residues,
                "num_atoms": num_atoms,
                "structure_id": structure.id,
            })
            
            logger.info(f"PDB/mmCIF validation passed: {num_atoms} atoms, {num_residues} residues, {num_chains} chains")
        except Exception as e:
            logger.error(f"PDB/mmCIF validation failed: {e}")
            result.error = f"PDB/mmCIF validation failed: {e}"
        return result
    
    def compare(self, reference_path: str, candidate_path: str, strategy: str = "exact", **kwargs) -> ComparisonResult:
        """Compare PDB/mmCIF files.
        
        Supported strategies:
        - exact: Direct sequence and structure matching (sequences identical and RMSD ≈ 0)
        - approximate: Alignment-based comparison allowing for sequence and structural variations
        - summary: Summary statistics comparison of chain count, residue count, atom count, and residue composition
        
        Args:
            reference_path: Path to reference PDB/mmCIF file
            candidate_path: Path to candidate PDB/mmCIF file
            strategy: Comparison strategy (exact, approximate, summary)
            **kwargs: Strategy-specific parameters
            - For exact strategy:
              - by_name: If True, match chains by chain ID; if False, match by sequence content (default: True)
            - For approximate strategy:
              - by_name: If True, match chains by chain ID; if False, match by sequence content (default: True)
              - sequence_weight: Weight for sequence similarity (default: 0.5)
              - structure_weight: Weight for structure similarity (default: 0.5)
              - rmsd_threshold: RMSD threshold for structure similarity calculation (default: 2.0)
            - For summary strategy:
              - num_bins: Number of bins for chain length distribution histogram (default: 10)
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
                    # Direct sequence and structure matching
                    self._compare_exact(ref_validation, alt_validation, **kwargs)
                elif strategy == "approximate":
                    # Alignment-based comparison allowing for variations
                    self._compare_approx(ref_validation, alt_validation, **kwargs)
                elif strategy == "summary":
                    # Summary statistics comparison
                    self._compare_summary(ref_validation, alt_validation, **kwargs)
                else:
                    logger.warning(f"Unknown strategy '{strategy}', fall back to exact")
                    self._compare_exact(ref_validation, alt_validation, **kwargs)
        except Exception as e:
            logger.error(f"PDB/mmCIF comparison failed: {e}")
            self.result.error = f"PDB/mmCIF comparison failed: {e}"
        return self._finalize_output(ref_validation, alt_validation, self.result)
    
    # ============================================================================
    # File I/O and Loading Methods
    # ============================================================================
    
    def _load_file(self, file_path: str) -> Optional[PDB.Structure.Structure]:
        """Load PDB/mmCIF structure from file.
        
        This is the central method for loading PDB/mmCIF files. All other methods
        should use this method to avoid duplicating parser calls.
        Tries to auto-detect format, then falls back to other supported formats.
        
        Args:
            file_path: Path to PDB/mmCIF file
            
        Returns:
            PDB Structure object or None if parsing fails
        """
        # Determine format from file path or signature
        ext = os.path.splitext(file_path)[1].lower().lstrip('.')
        format_order = [self.format]
        
        if ext in ['cif', 'mmcif']:
            format_order = ['cif'] + [f for f in self.supported_formats if f != 'cif']
        elif ext == 'pdb':
            format_order = ['pdb'] + [f for f in self.supported_formats if f != 'pdb']
        else:
            format_order = [self.format] + list(set(self.supported_formats) - {self.format})
        
        for fmt in format_order:
            try:
                parser_class = self.format_map[fmt]['parser']
                
                # Cache parser instances
                if fmt not in self.parser_cache:
                    self.parser_cache[fmt] = parser_class(QUIET=True)
                
                parser = self.parser_cache[fmt]
                structure = parser.get_structure('structure', file_path)
                self.format = fmt  # update detected format
                return structure
            except Exception as e:
                logger.debug(f"Failed to parse {file_path} as {fmt}: {e}")
                continue
        
        # All attempts failed
        logger.error(f"Failed to parse {file_path} with any supported format")
        return None
    
    # ============================================================================
    # Statistics and Analysis Methods
    # ============================================================================
    
    def _extract_sequences(self, structure: PDB.Structure.Structure) -> Dict[str, str]:
        """Extract sequences from structure, mapping chain ID to sequence string.
        
        Returns dictionary mapping chain_id to amino acid sequence string (one-letter codes).
        Skips heteroatoms (water, ligands, etc.) and non-convertible residues.
        """
        sequences = {}
        
        for chain in structure.get_chains():
            chain_id = chain.id
            seq_list = []
            
            for residue in chain:
                # Skip heteroatoms (water, ligands, etc.)
                if residue.id[0] != ' ':
                    continue
                
                try:
                    # Get three-letter code and convert to one-letter
                    resname = residue.get_resname()
                    one_letter = seq1(resname, undef_code='X')
                    seq_list.append(one_letter)
                except Exception:
                    # Skip residues that can't be converted
                    continue
            
            if seq_list:
                sequences[chain_id] = ''.join(seq_list)
        
        return sequences
    
    def _extract_ca_coordinates(self, structure: PDB.Structure.Structure, chain_id: Optional[str] = None) -> Tuple[np.ndarray, List[str]]:
        """Extract Cα atom coordinates from structure.
        
        Args:
            structure: PDB structure object
            chain_id: Optional chain ID to extract from (if None, extracts from all chains)
        
        Returns:
            Tuple of (coordinates array of shape (n_residues, 3), residue names list)
        """
        coords = []
        resnames = []
        
        if chain_id is not None:
            try:
                chains = [structure[0][chain_id]]
            except KeyError:
                return np.array([]), []
        else:
            chains = list(structure.get_chains())
        
        for chain in chains:
            for residue in chain:
                # Skip heteroatoms
                if residue.id[0] != ' ':
                    continue
                
                try:
                    # Get Cα atom
                    ca_atom = residue['CA']
                    coords.append(ca_atom.coord)
                    resnames.append(residue.get_resname())
                except KeyError:
                    # Skip residues without Cα (e.g., non-standard residues)
                    continue
        
        if not coords:
            return np.array([]), []
        
        return np.array(coords), resnames
    
    def _get_summary_stats(self, file_path: str) -> Dict:
        """Compute summary statistics for a PDB/mmCIF file.
        
        Args:
            file_path: Path to PDB/mmCIF file
            
        Returns:
            Dictionary with summary statistics:
            - num_chains: Number of chains
            - num_residues: Total number of residues (standard amino acids)
            - num_atoms: Total number of atoms
            - chain_lengths: Array of chain lengths (residue counts per chain)
        """
        structure = self._load_file(file_path)
        if structure is None:
            return {
                "num_chains": 0,
                "num_residues": 0,
                "num_atoms": 0,
                "chain_lengths": np.array([]),
            }
        
        sequences = self._extract_sequences(structure)
        num_chains = len(sequences)
        chain_lengths = np.array([len(seq) for seq in sequences.values()]) if sequences else np.array([])
        num_residues = int(chain_lengths.sum()) if len(chain_lengths) > 0 else 0
        num_atoms = len(list(structure.get_atoms()))
        
        return {
            "num_chains": num_chains,
            "num_residues": num_residues,
            "num_atoms": num_atoms,
            "chain_lengths": chain_lengths,
        }
    
    def _get_residue_counter(self, file_path: str) -> Counter:
        """Count residue types across all chains in a PDB/mmCIF file.
        
        Analogous to _get_kmer_counter in FASTA handler: extracts a frequency
        distribution over the residue composition.
        
        Args:
            file_path: Path to PDB/mmCIF file
            
        Returns:
            Counter mapping residue one-letter code to frequency
        """
        structure = self._load_file(file_path)
        if structure is None:
            return Counter()
        
        sequences = self._extract_sequences(structure)
        residue_counter = Counter()
        
        for seq in sequences.values():
            residue_counter.update(seq)
        
        return residue_counter
    
    def _get_residue_similarity(self, ref_counter: Counter, alt_counter: Counter) -> float:
        """Get similarity score between residue type distributions using Hellinger distance.
        
        Analogous to _get_kmer_similarity in FASTA handler.
        
        Returns:
            Similarity score between 0.0 and 1.0 (1.0 - Hellinger distance)
        """
        if not ref_counter and not alt_counter:
            return 1.0  # Both empty, perfect match
        
        if not ref_counter or not alt_counter:
            return 0.0  # One empty, one not
        
        # Get all unique residue types and align them
        all_residues = sorted(set(ref_counter.keys()) | set(alt_counter.keys()))
        
        # Create aligned arrays with frequencies (0 for missing residue types)
        ref_arr = np.array([ref_counter.get(res, 0) for res in all_residues])
        alt_arr = np.array([alt_counter.get(res, 0) for res in all_residues])
        
        # Calculate Hellinger distance (function handles normalization internally)
        distance = hellinger_distance(ref_arr, alt_arr)
        
        # Convert distance to similarity: similarity = 1 - distance
        similarity = 1.0 - distance if not np.isnan(distance) else 0.0
        
        return similarity
    
    def _get_alignment_similarity(self, ref_seq: str, alt_seq: str, aligner: PairwiseAligner) -> float:
        """Calculate normalized similarity score between two sequences using alignment.
        
        Analogous to _get_alignment_similarity in FASTA handler.
        
        Args:
            ref_seq: Reference sequence string (one-letter amino acid codes)
            alt_seq: Candidate sequence string (one-letter amino acid codes)
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
    
    def _get_structure_similarity(self, ref_coords: np.ndarray, alt_coords: np.ndarray,
                                  align: bool = True, rmsd_threshold: float = 2.0) -> Dict[str, float]:
        """Calculate structural similarity between two sets of Cα coordinates.
        
        Uses RMSD and TM-score to assess structural similarity.
        
        Args:
            ref_coords: Reference Cα coordinates of shape (n, 3)
            alt_coords: Candidate Cα coordinates of shape (n, 3)
            align: Whether to align structures before RMSD calculation
            rmsd_threshold: RMSD threshold for exponential decay similarity
            
        Returns:
            Dictionary with:
            - rmsd: Root mean square deviation (Å)
            - tm_score: TM-score (0-1, >0.5 indicates same fold)
            - rmsd_similarity: Exponential decay similarity from RMSD
            - similarity: Combined structural similarity (average of TM-score and RMSD similarity)
            - num_residues: Number of residues used for comparison
        """
        if len(ref_coords) == 0 or len(alt_coords) == 0:
            return {"rmsd": np.nan, "tm_score": 0.0, "rmsd_similarity": 0.0, "similarity": 0.0, "num_residues": 0}
        
        # Use minimum length if sizes differ
        min_len = min(len(ref_coords), len(alt_coords))
        ref_trimmed = ref_coords[:min_len]
        alt_trimmed = alt_coords[:min_len]
        
        # Calculate RMSD
        rmsd, _, _ = self._calculate_rmsd(ref_trimmed, alt_trimmed, align=align)
        
        # Calculate TM-score
        tm_score = self._calculate_tm_score(ref_trimmed, alt_trimmed, ref_length=len(ref_trimmed))
        
        # Convert RMSD to similarity using exponential decay: exp(-RMSD / threshold)
        # Threshold of 2.0 Å means RMSD of 2.0 gives similarity ~0.37
        rmsd_similarity = float(np.exp(-rmsd / rmsd_threshold))
        
        # Combined similarity: average of TM-score and RMSD-based similarity
        similarity = (tm_score + rmsd_similarity) / 2.0
        
        return {
            "rmsd": float(rmsd),
            "tm_score": float(tm_score),
            "rmsd_similarity": float(rmsd_similarity),
            "similarity": float(similarity),
            "num_residues": min_len,
        }
    
    def _calculate_rmsd(self, ref_coords: np.ndarray, alt_coords: np.ndarray, 
                        align: bool = True) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
        """Calculate RMSD between two sets of coordinates.
        
        Uses the Kabsch algorithm for optimal superposition when align=True.
        
        Args:
            ref_coords: Reference coordinates array of shape (n, 3)
            alt_coords: Candidate coordinates array of shape (n, 3)
            align: If True, align structures before calculating RMSD
        
        Returns:
            Tuple of (RMSD value, aligned ref coords, aligned alt coords)
        """
        if ref_coords.shape != alt_coords.shape:
            raise ValueError(f"Coordinate arrays must have same shape: {ref_coords.shape} vs {alt_coords.shape}")
        
        if len(ref_coords) == 0:
            return 0.0, None, None
        
        if align:
            # Center both structures
            ref_centered = ref_coords - ref_coords.mean(axis=0)
            alt_centered = alt_coords - alt_coords.mean(axis=0)
            
            # Calculate rotation matrix using Kabsch algorithm
            # H = ref^T * alt
            H = ref_centered.T @ alt_centered
            
            # SVD
            U, S, Vt = np.linalg.svd(H)
            
            # Rotation matrix
            R = Vt.T @ U.T
            
            # Ensure proper rotation (det(R) = 1)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Apply rotation
            alt_aligned = alt_centered @ R.T
            
            # Calculate RMSD
            rmsd = np.sqrt(np.mean(np.sum((ref_centered - alt_aligned) ** 2, axis=1)))
            
            return float(rmsd), ref_centered, alt_aligned
        
        else:
            # No alignment, just calculate RMSD
            rmsd = np.sqrt(np.mean(np.sum((ref_coords - alt_coords) ** 2, axis=1)))
            return float(rmsd), ref_coords.copy(), alt_coords.copy()
    
    def _calculate_tm_score(self, ref_coords: np.ndarray, alt_coords: np.ndarray, 
                           ref_length: Optional[int] = None) -> float:
        """Calculate TM-score (Template Modeling score) between two structures.
        
        TM-score is a topology-independent metric for structure comparison.
        Range: 0-1, where >0.5 indicates same fold.
        
        Args:
            ref_coords: Reference coordinates array of shape (n, 3)
            alt_coords: Candidate coordinates array of shape (n, 3)
            ref_length: Reference length (if None, uses length of ref_coords)
        
        Returns:
            TM-score value (0-1)
        """
        if ref_coords.shape != alt_coords.shape:
            raise ValueError(f"Coordinate arrays must have same shape: {ref_coords.shape} vs {alt_coords.shape}")
        
        if len(ref_coords) == 0:
            return 0.0
        
        n = len(ref_coords)
        if ref_length is None:
            ref_length = n
        
        # Align structures first
        _, ref_aligned, alt_aligned = self._calculate_rmsd(ref_coords, alt_coords, align=True)
        
        # Calculate distances
        distances = np.sqrt(np.sum((ref_aligned - alt_aligned) ** 2, axis=1))
        
        # TM-score formula: (1/L) * sum(1 / (1 + (d_i/d0)^2))
        # d0 = 1.24 * (L - 15)^(1/3) - 1.8
        d0 = 1.24 * ((ref_length - 15) ** (1/3)) - 1.8
        if d0 < 0.5:
            d0 = 0.5  # Minimum d0
        
        # Calculate TM-score
        tm_score = (1.0 / ref_length) * np.sum(1.0 / (1.0 + (distances / d0) ** 2))
        
        return float(tm_score)
    
    def _get_matched_chains(self, ref_seqs: Dict[str, str], alt_seqs: Dict[str, str], 
                            by_name: bool, aligner: PairwiseAligner) -> List[tuple]:
        """Get matched chain pairs between reference and candidate structures.
        
        Analogous to _get_matched_pairs in FASTA handler.
        
        Args:
            ref_seqs: Reference sequences {chain_id: sequence}
            alt_seqs: Candidate sequences {chain_id: sequence}
            by_name: If True, match chains by chain ID; if False, match by sequence content
            aligner: PairwiseAligner instance for calculating similarity
            
        Returns:
            List of tuples (ref_info, alt_info):
            - ref_info is [chain_id, sequence] or None
            - alt_info is [chain_id, sequence] or None
            - Matched pair: both are [chain_id, sequence]
            - Unmatched ref: (ref_info, None)
            - Unmatched alt: (None, alt_info)
        """
        pairs = []
        
        if by_name:
            # Match by chain ID: straightforward key-based matching
            ref_keys = set(ref_seqs.keys())
            alt_keys = set(alt_seqs.keys())
            common_keys = ref_keys & alt_keys
            
            # Matched pairs
            for key in common_keys:
                pairs.append(([key, ref_seqs[key]], [key, alt_seqs[key]]))
            
            # Unmatched ref chains
            for key in ref_keys - alt_keys:
                pairs.append(([key, ref_seqs[key]], None))
            
            # Unmatched alt chains
            for key in alt_keys - ref_keys:
                pairs.append((None, [key, alt_seqs[key]]))
        else:
            # Match by sequence content: use greedy matching to find best pairs
            ref_items = list(ref_seqs.items())  # [(chain_id, seq), ...]
            alt_items = list(alt_seqs.items())
            
            # Pre-filter: exact sequence matches
            ref_seq_to_idx = {}
            for i, (_, seq) in enumerate(ref_items):
                ref_seq_to_idx.setdefault(seq, []).append(i)
            
            alt_seq_to_idx = {}
            for i, (_, seq) in enumerate(alt_items):
                alt_seq_to_idx.setdefault(seq, []).append(i)
            
            matched_ref_indices = set()
            matched_alt_indices = set()
            
            # First pass: exact sequence matches
            for seq in set(ref_seq_to_idx.keys()) & set(alt_seq_to_idx.keys()):
                ref_indices = ref_seq_to_idx[seq]
                alt_indices = alt_seq_to_idx[seq]
                for ri, ai in zip(ref_indices, alt_indices):
                    if ri not in matched_ref_indices and ai not in matched_alt_indices:
                        ref_id, ref_seq = ref_items[ri]
                        alt_id, alt_seq = alt_items[ai]
                        pairs.append(([ref_id, ref_seq], [alt_id, alt_seq]))
                        matched_ref_indices.add(ri)
                        matched_alt_indices.add(ai)
            
            # Second pass: greedy alignment matching for remaining chains
            for ri, (ref_id, ref_seq) in enumerate(ref_items):
                if ri in matched_ref_indices:
                    continue
                
                best_similarity = -1.0
                best_alt_idx = None
                
                for ai, (alt_id, alt_seq) in enumerate(alt_items):
                    if ai in matched_alt_indices:
                        continue
                    
                    similarity = self._get_alignment_similarity(ref_seq, alt_seq, aligner)
                    
                    # Early termination on perfect match
                    if similarity >= 1.0:
                        best_similarity = similarity
                        best_alt_idx = ai
                        break
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_alt_idx = ai
                
                if best_alt_idx is not None:
                    matched_ref_indices.add(ri)
                    matched_alt_indices.add(best_alt_idx)
                    alt_id, alt_seq = alt_items[best_alt_idx]
                    pairs.append(([ref_id, ref_seq], [alt_id, alt_seq]))
                else:
                    pairs.append(([ref_id, ref_seq], None))
            
            # Unmatched alt chains
            for ai, (alt_id, alt_seq) in enumerate(alt_items):
                if ai not in matched_alt_indices:
                    pairs.append((None, [alt_id, alt_seq]))
        
        return pairs
    
    # ============================================================================
    # Private Comparison Implementation Methods
    # ============================================================================
    
    def _compare_exact(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Exact match: Direct sequence and structure matching by chain.
        
        Chains must have identical sequences AND near-zero RMSD (< 0.001 Å).
        When by_name=True (default), chains are matched by chain ID.
        When by_name=False, chains are matched by sequence content.
        
        Args:
            **kwargs: Strategy-specific parameters
            - by_name: If True, match chains by chain ID; if False, match by sequence content (default: True)
        """
        by_name = kwargs.get('by_name', True)
        
        try:
            # Load structures
            ref_structure = self._load_file(ref_validation.file_path)
            alt_structure = self._load_file(alt_validation.file_path)
            
            if ref_structure is None:
                raise ValueError("Failed to load reference structure")
            if alt_structure is None:
                raise ValueError("Failed to load candidate structure")
            
            # Extract sequences
            ref_seqs = self._extract_sequences(ref_structure)
            alt_seqs = self._extract_sequences(alt_structure)
            
            if not ref_seqs or not alt_seqs:
                self.result.similarity = 0.0
                self.result.details.update({"error": "No sequences found in one or both structures"})
                return
            
            # Match chains
            aligner = PairwiseAligner()
            aligner.mode = 'local'
            pairs = self._get_matched_chains(ref_seqs, alt_seqs, by_name, aligner)
            
            # Check each matched pair for exact sequence match AND structural match
            matching_count = 0
            total_ref_count = len(ref_seqs)
            chain_details = {}
            
            for ref_info, alt_info in pairs:
                if ref_info is None or alt_info is None:
                    continue
                
                ref_chain_id, ref_seq = ref_info
                alt_chain_id, alt_seq = alt_info
                
                # Check sequence identity
                seq_match = (ref_seq == alt_seq)
                
                # Check structural identity (RMSD ≈ 0) only if sequences match
                struct_match = False
                rmsd = np.nan
                if seq_match:
                    ref_coords, _ = self._extract_ca_coordinates(ref_structure, ref_chain_id)
                    alt_coords, _ = self._extract_ca_coordinates(alt_structure, alt_chain_id)
                    
                    if len(ref_coords) > 0 and len(alt_coords) > 0 and len(ref_coords) == len(alt_coords):
                        rmsd_val, _, _ = self._calculate_rmsd(ref_coords, alt_coords, align=True)
                        rmsd = rmsd_val
                        struct_match = rmsd_val < 0.001
                
                is_exact = seq_match and struct_match
                if is_exact:
                    matching_count += 1
                
                chain_details[f"chain_{ref_chain_id}_vs_{alt_chain_id}"] = {
                    "sequence_match": seq_match,
                    "structure_match": struct_match,
                    "rmsd": float(rmsd) if not np.isnan(rmsd) else None,
                    "exact_match": is_exact,
                }
            
            similarity = matching_count / total_ref_count if total_ref_count > 0 else 0.0
            
            self.result.similarity = similarity
            self.result.details.update({
                "matching_count": matching_count,
                "total_ref_count": total_ref_count,
                "total_alt_count": len(alt_seqs),
                "chain_details": chain_details,
            })
            
        except Exception as e:
            logger.error(f"Exact comparison failed: {e}")
            self.result.error = f"Exact comparison failed: {e}"
    
    def _compare_approx(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Approximate comparison: Alignment-based sequence and structure comparison.
        
        For each matched chain pair, calculates sequence alignment similarity
        and structural similarity (RMSD, TM-score), then combines with configurable weights.
        Unmatched reference chains contribute 0 similarity.
        
        Args:
            **kwargs: Strategy-specific parameters
            - by_name: If True, match chains by chain ID; if False, match by sequence content (default: True)
            - sequence_weight: Weight for sequence similarity (default: 0.5)
            - structure_weight: Weight for structure similarity (default: 0.5)
            - rmsd_threshold: RMSD threshold for structure similarity (default: 2.0)
        """
        by_name = kwargs.get('by_name', True)
        seq_weight = kwargs.get('sequence_weight', 0.5)
        struct_weight = kwargs.get('structure_weight', 0.5)
        rmsd_threshold = kwargs.get('rmsd_threshold', 2.0)
        
        try:
            # Load structures
            ref_structure = self._load_file(ref_validation.file_path)
            alt_structure = self._load_file(alt_validation.file_path)
            
            if ref_structure is None:
                raise ValueError("Failed to load reference structure")
            if alt_structure is None:
                raise ValueError("Failed to load candidate structure")
            
            # Extract sequences
            ref_seqs = self._extract_sequences(ref_structure)
            alt_seqs = self._extract_sequences(alt_structure)
            
            if not ref_seqs or not alt_seqs:
                self.result.similarity = 0.0
                self.result.details.update({"error": "No sequences found in one or both structures"})
                return
            
            # Initialize aligner with BioPython defaults
            aligner = PairwiseAligner()
            aligner.mode = 'local'  # Local alignment (Smith-Waterman)
            
            # Match chains
            pairs = self._get_matched_chains(ref_seqs, alt_seqs, by_name, aligner)
            
            # Calculate similarity for each matched pair
            chain_similarities = []
            matching_count = 0
            missing_count = 0
            extra_count = 0
            chain_details = {}
            
            for ref_info, alt_info in pairs:
                if ref_info is not None and alt_info is not None:
                    ref_chain_id, ref_seq = ref_info
                    alt_chain_id, alt_seq = alt_info
                    matching_count += 1
                    
                    # Sequence similarity
                    if ref_seq == alt_seq:
                        seq_sim = 1.0
                    else:
                        seq_sim = self._get_alignment_similarity(ref_seq, alt_seq, aligner)
                    
                    # Structure similarity
                    ref_coords, _ = self._extract_ca_coordinates(ref_structure, ref_chain_id)
                    alt_coords, _ = self._extract_ca_coordinates(alt_structure, alt_chain_id)
                    
                    struct_result = self._get_structure_similarity(
                        ref_coords, alt_coords, align=True, rmsd_threshold=rmsd_threshold
                    )
                    struct_sim = struct_result["similarity"]
                    
                    # Combined chain similarity
                    chain_sim = seq_weight * seq_sim + struct_weight * struct_sim
                    chain_similarities.append(chain_sim)
                    
                    chain_details[f"chain_{ref_chain_id}_vs_{alt_chain_id}"] = {
                        "sequence_similarity": seq_sim,
                        "structure_similarity": struct_sim,
                        "rmsd": struct_result["rmsd"],
                        "tm_score": struct_result["tm_score"],
                        "combined_similarity": chain_sim,
                    }
                elif ref_info is not None:
                    # Unmatched ref chain
                    missing_count += 1
                else:
                    # Unmatched alt chain
                    extra_count += 1
            
            # Overall similarity: average across all ref chains (unmatched ref = 0)
            total_ref_count = len(ref_seqs)
            similarity = sum(chain_similarities) / total_ref_count if total_ref_count > 0 else 0.0
            
            self.result.similarity = float(similarity)
            self.result.details.update({
                "matching_count": matching_count,
                "missing_count": missing_count,
                "extra_count": extra_count,
                "total_ref_count": total_ref_count,
                "total_alt_count": len(alt_seqs),
                "avg_chain_similarity": float(np.mean(chain_similarities)) if chain_similarities else 0.0,
                "sequence_weight": seq_weight,
                "structure_weight": struct_weight,
                "rmsd_threshold": rmsd_threshold,
                "chain_details": chain_details,
            })
            
        except Exception as e:
            logger.error(f"Approximate comparison failed: {e}")
            self.result.error = f"Approximate comparison failed: {e}"
    
    def _compare_summary(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Summary comparison: Compare high-level statistics and residue composition.
        
        Compares:
        - Number of chains (RAE similarity)
        - Number of residues (RAE similarity)
        - Number of atoms (RAE similarity)
        - Chain length distribution (Hellinger distance on histograms)
        - Residue type distribution (Hellinger distance on composition)
        
        Final similarity is the average of all component similarities.
        
        Args:
            **kwargs: Strategy-specific parameters
            - num_bins: Number of bins for chain length distribution histogram (default: 10)
        """
        try:
            # Compute summary statistics for both files
            ref_stats = self._get_summary_stats(ref_validation.file_path)
            alt_stats = self._get_summary_stats(alt_validation.file_path)
            
            ref_chain_lengths = ref_stats.pop("chain_lengths", np.array([]))
            alt_chain_lengths = alt_stats.pop("chain_lengths", np.array([]))
            
            # Compare number of chains using RAE similarity
            ref_num_chains = ref_stats.get("num_chains", 0)
            alt_num_chains = alt_stats.get("num_chains", 0)
            num_chains_sim = float(rae_similarity([ref_num_chains], [alt_num_chains])[0])
            
            # Compare number of residues using RAE similarity
            ref_num_residues = ref_stats.get("num_residues", 0)
            alt_num_residues = alt_stats.get("num_residues", 0)
            num_residues_sim = float(rae_similarity([ref_num_residues], [alt_num_residues])[0])
            
            # Compare number of atoms using RAE similarity
            ref_num_atoms = ref_stats.get("num_atoms", 0)
            alt_num_atoms = alt_stats.get("num_atoms", 0)
            num_atoms_sim = float(rae_similarity([ref_num_atoms], [alt_num_atoms])[0])
            
            # Compare chain length distribution using histogram
            num_bins = kwargs.get('num_bins', 10)
            if len(ref_chain_lengths) == 0 and len(alt_chain_lengths) == 0:
                # Both empty - perfect match
                chain_length_distribution_sim = 1.0
                chain_length_hellinger = 0.0
            elif len(ref_chain_lengths) == 0 or len(alt_chain_lengths) == 0:
                # One empty, one not - no match
                chain_length_distribution_sim = 0.0
                chain_length_hellinger = 1.0
            else:
                all_lengths = np.concatenate([ref_chain_lengths, alt_chain_lengths])
                min_len = all_lengths.min()
                max_len = all_lengths.max()
                
                # Default: all chains have the same length
                chain_length_distribution_sim = 1.0 if ref_chain_lengths.mean() == alt_chain_lengths.mean() else 0.0
                chain_length_hellinger = 0.0 if chain_length_distribution_sim == 1.0 else 1.0
                
                if max_len > min_len:
                    bins = np.linspace(min_len, max_len, num_bins + 1)
                    ref_hist, _ = np.histogram(ref_chain_lengths, bins=bins)
                    alt_hist, _ = np.histogram(alt_chain_lengths, bins=bins)
                    chain_length_hellinger = hellinger_distance(ref_hist, alt_hist)
                    chain_length_distribution_sim = 1.0 - float(chain_length_hellinger) if not np.isnan(chain_length_hellinger) else 0.0
            
            # Compare residue type distribution
            ref_residues = self._get_residue_counter(ref_validation.file_path)
            alt_residues = self._get_residue_counter(alt_validation.file_path)
            residue_sim = self._get_residue_similarity(ref_residues, alt_residues)
            
            # Overall similarity: average of all component similarities
            similarity = (num_chains_sim + num_residues_sim + num_atoms_sim + chain_length_distribution_sim + residue_sim) / 5.0
            
            self.result.similarity = float(similarity)
            self.result.details.update({
                "num_chains_similarity": num_chains_sim,
                "num_residues_similarity": num_residues_sim,
                "num_atoms_similarity": num_atoms_sim,
                "chain_length_distribution_similarity": chain_length_distribution_sim,
                "chain_length_hellinger_distance": float(chain_length_hellinger),
                "residue_distribution_similarity": residue_sim,
                "ref_stats": ref_stats,
                "alt_stats": alt_stats,
                "ref_num_residue_types": len(ref_residues),
                "alt_num_residue_types": len(alt_residues),
            })
            
        except Exception as e:
            logger.error(f"Summary comparison failed: {e}")
            self.result.error = f"Summary comparison failed: {e}"
