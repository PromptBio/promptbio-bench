"""Mass spectrometry XML format handler for bioinformatics XML formats (featureXML, mzML, idXML, etc.).

Strategy:
- exact: Element-by-element XML comparison, all elements identical (100% required)
- approximate: Matches features/spectra/identifications by proximity with tolerance for numeric differences
- summary: Summary statistics comparison of element counts, distributions, numeric value distributions

Supported formats:
- featureXML: OpenMS featureXML format (root element: <featureMap>)
- mzML: Mass spectrometry mzML format (root element: <mzML>)
- idXML: OpenMS idXML format for peptide/protein identification (root element: <IdXML>)

Required cmdline tools:
- (None)
"""

import logging
import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter

import numpy as np

from tools.base import FormatHandler
from tools.results import ValidationResult, ComparisonResult
from utils.format import detect_file_signature
from utils.metrics import hellinger_distance, jaccard_similarity, precision_recall_f1, relative_absolute_error

logger = logging.getLogger(__name__)


class MSXMLHandler(FormatHandler):
    """Handler for mass spectrometry XML-based bioinformatics format files (featureXML, mzML, idXML, etc.)."""
    
    def __init__(self):
        super().__init__("msxml")
        self.format_map = {
            'featurexml': {'root_element': 'featuremap', 'namespace_aware': False},
            'mzml': {'root_element': 'mzml', 'namespace_aware': True},
            'idxml': {'root_element': 'idxml', 'namespace_aware': False},
        }
        self.supported_formats = list(self.format_map.keys())
        self.supported_strategies = ["exact", "approximate", "summary"]
    
    # ============================================================================
    # Public API Methods
    # ============================================================================
    
    def validate(self, file_path: str, **kwargs) -> ValidationResult:
        """Validate mass spectrometry XML file (featureXML, mzML, etc.)
        
        Args:
            file_path: Path to XML file
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
            logger.error(f"Mass spectrometry XML file not found: {file_path}")
            result.error = "File not found"
            return result
        
        if not is_nonempty:
            logger.warning(f"Mass spectrometry XML file is empty: {file_path}")
            result.error = "File is empty"
            return result
        
        if not is_format_ok:
            logger.error(f"Not a supported mass spectrometry XML file: detected={signature.format}")
            result.error = f"Not a supported mass spectrometry XML file: detected={signature.format}"
            return result
        
        # validate by parsing the XML file
        try:
            root = self._load_file(file_path, file_format=signature.format)
            if root is None:
                result.error = "Failed to parse XML file"
                return result
            
            result.parseable = True
            
            # Extract statistics based on format
            stats = self._get_xml_stats(root, signature.format)
            result.non_empty = stats.get("num_elements", 0) > 0
            
            if not result.non_empty:
                logger.warning(f"Mass spectrometry XML file contains no elements: {file_path}")
                result.error = "XML file contains no elements"
                return result
            
            result.details.update(stats)
            logger.info(f"Mass spectrometry XML validation passed: {file_path}")
        except Exception as e:
            logger.error(f"Mass spectrometry XML validation failed: {e}")
            result.error = f"Mass spectrometry XML validation failed: {e}"
        return result
    
    def compare(self, reference_path: str, candidate_path: str, strategy: str = "summary", **kwargs) -> ComparisonResult:
        """Compare mass spectrometry XML files (featureXML, mzML, etc.).
        
        Supported strategies:
        - exact: Element-by-element XML comparison, all elements identical (100% required)
        - approximate: Matches features/spectra by proximity with tolerance for numeric differences
        - summary: Summary statistics comparison of element counts, distributions, numeric values
        
        Args:
            reference_path: Path to reference XML file
            candidate_path: Path to candidate XML file
            strategy: Comparison strategy (exact, approximate, summary)
            **kwargs: Strategy-specific parameters
            - tolerance: Relative tolerance for numeric comparisons (default: 1e-6)
            - rt_tolerance: Retention time tolerance in seconds for feature/spectrum matching (default: 10.0)
            - mz_tolerance: m/z tolerance in Da for feature matching (default: 0.01)
        """
        # initialize comparison result
        self.result = ComparisonResult(reference_path=reference_path, candidate_path=candidate_path, 
                                       strategy=strategy, similarity=np.nan, details={}, error=None)
        
        # validate reference and candidate files
        ref_validation = self.validate(reference_path, **kwargs)
        alt_validation = self.validate(candidate_path, **kwargs)

        try:
            if ref_validation.is_valid and alt_validation.is_valid:
                # Check format compatibility
                if ref_validation.signature.format != alt_validation.signature.format:
                    logger.error(f"Format mismatch: ref={ref_validation.signature.format}, alt={alt_validation.signature.format}")
                    self.result.error = f"Format mismatch: ref={ref_validation.signature.format}, alt={alt_validation.signature.format}"
                else:
                    # compare files using specified strategy
                    if strategy == "exact":
                        # Element-by-element exact matching, all elements identical in DFS order
                        self._compare_exact(ref_validation, alt_validation, **kwargs)
                    elif strategy == "approximate":
                        # Match features/spectra by proximity, compare with numeric tolerance
                        self._compare_approx(ref_validation, alt_validation, **kwargs)
                    elif strategy == "summary":
                        # Summary statistics: element counts, distributions, numeric value distributions
                        self._compare_summary(ref_validation, alt_validation, **kwargs)
                    else:
                        logger.warning(f"Unknown strategy '{strategy}', fall back to summary")
                        self._compare_summary(ref_validation, alt_validation, **kwargs)
        except Exception as e:
            logger.error(f"Mass spectrometry XML comparison failed: {e}")
            self.result.error = f"Mass spectrometry XML comparison failed: {e}"
        return self._finalize_output(ref_validation, alt_validation, self.result)

    # ============================================================================
    # File I/O and Loading Methods
    # ============================================================================
    
    def _load_file(self, file_path: str, file_format: Optional[str] = None) -> Optional[ET.Element]:
        """Load XML file and return root element.
        
        Args:
            file_path: Path to XML file
            file_format: Format name (featurexml, mzml) for namespace handling
            
        Returns:
            Root Element object, or None if parsing fails
        """
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            return root
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load XML file: {e}")
            return None
    
    def _normalize_tag(self, tag: str, ignore_namespaces: bool = True) -> str:
        """Normalize XML tag by removing namespace if requested.
        
        Args:
            tag: XML tag (potentially with namespace like '{http://...}tagName')
            ignore_namespaces: Whether to strip namespace prefix
            
        Returns:
            Normalized tag name
        """
        if ignore_namespaces and '}' in tag:
            return tag.split('}')[-1]
        return tag
    
    def _get_namespace(self, root: ET.Element) -> str:
        """Extract namespace from root element tag.
        
        Args:
            root: Root XML element
            
        Returns:
            Namespace string (e.g., '{http://psi.hupo.org/ms/mzml}') or empty string
        """
        if root.tag.startswith("{") and "}" in root.tag:
            return root.tag.split("}")[0] + "}"
        return ""
    
    # ============================================================================
    # Statistics and Analysis Methods
    # ============================================================================
    
    def _get_xml_stats(self, root: ET.Element, file_format: str) -> Dict[str, Any]:
        """Extract general statistics from XML root element.
        
        Args:
            root: Root XML element
            file_format: Format name (featurexml, mzml)
            
        Returns:
            Dictionary with statistics (element counts, attribute counts, numeric values, etc.)
        """
        stats = {
            "num_elements": 0,
            "num_attributes": 0,
            "element_types": Counter(),
            "attribute_names": Counter(),
            "numeric_values": [],
            "text_values": [],
        }
        
        # Recursively traverse XML tree
        for elem in root.iter():
            stats["num_elements"] += 1
            stats["element_types"][self._normalize_tag(elem.tag)] += 1
            
            # Count attributes
            for attr_name, attr_value in elem.attrib.items():
                stats["num_attributes"] += 1
                stats["attribute_names"][attr_name] += 1
                
                # Try to parse as numeric
                try:
                    num_val = float(attr_value)
                    stats["numeric_values"].append(num_val)
                except (ValueError, TypeError):
                    pass
            
            # Check text content
            if elem.text and elem.text.strip():
                text = elem.text.strip()
                stats["text_values"].append(text)
                
                # Try to parse as numeric
                try:
                    num_val = float(text)
                    stats["numeric_values"].append(num_val)
                except (ValueError, TypeError):
                    pass
        
        # Convert counters to dicts for JSON serialization
        stats["element_types"] = dict(stats["element_types"])
        stats["attribute_names"] = dict(stats["attribute_names"])
        
        # Add format-specific statistics
        if file_format == "featurexml":
            stats.update(self._get_featurexml_stats(root))
        elif file_format == "mzml":
            stats.update(self._get_mzml_stats(root))
        elif file_format == "idxml":
            stats.update(self._get_idxml_stats(root))
        
        return stats
    
    def _get_featurexml_stats(self, root: ET.Element) -> Dict[str, Any]:
        """Extract featureXML-specific statistics.
        
        Counts features, subordinates, UserParam/cvParam elements,
        and computes value distributions for RT, m/z, intensity, charge, quality.
        
        Args:
            root: Root XML element
            
        Returns:
            Dictionary with featureXML-specific statistics
        """
        stats = {
            "num_features": 0,
            "num_subordinates": 0,
            "num_user_params": 0,
            "num_cv_params": 0,
        }
        
        # Count element types relevant to featureXML
        for elem in root.iter():
            tag = self._normalize_tag(elem.tag)
            if tag == "subordinate":
                stats["num_subordinates"] += 1
            elif tag == "UserParam":
                stats["num_user_params"] += 1
            elif tag == "cvParam":
                stats["num_cv_params"] += 1
        
        # Extract top-level features for distribution stats
        features = self._get_featurexml_features(root)
        stats["num_features"] = len(features)
        
        if features:
            rts = [f["rt"] for f in features if f["rt"] is not None]
            mzs = [f["mz"] for f in features if f["mz"] is not None]
            intensities = [f["intensity"] for f in features if f["intensity"] is not None]
            charges = [f["charge"] for f in features if f["charge"] is not None]
            qualities = [f["quality"] for f in features if f["quality"] is not None]
            
            if rts:
                stats["rt_range"] = [float(min(rts)), float(max(rts))]
            if mzs:
                stats["mz_range"] = [float(min(mzs)), float(max(mzs))]
            if intensities:
                stats["intensity_range"] = [float(min(intensities)), float(max(intensities))]
            if charges:
                stats["charge_distribution"] = dict(Counter(int(c) for c in charges))
            if qualities:
                stats["quality_range"] = [float(min(qualities)), float(max(qualities))]
        
        return stats
    
    def _get_mzml_stats(self, root: ET.Element) -> Dict[str, Any]:
        """Extract mzML-specific statistics.
        
        Counts spectra, chromatograms, software entries, and computes
        distributions for MS levels, retention times, and TIC values.
        
        Args:
            root: Root XML element
            
        Returns:
            Dictionary with mzML-specific statistics
        """
        stats = {
            "num_spectra": 0,
            "num_chromatograms": 0,
            "num_software": 0,
        }
        
        ns = self._get_namespace(root)
        
        # Count spectra from spectrumList count attribute
        spectrum_list = root.find(f".//{ns}spectrumList")
        if spectrum_list is not None:
            stats["num_spectra"] = int(spectrum_list.get("count", 0))
        else:
            stats["num_spectra"] = len(root.findall(f".//{ns}spectrum"))
        
        # Count chromatograms from chromatogramList count attribute
        chromatogram_list = root.find(f".//{ns}chromatogramList")
        if chromatogram_list is not None:
            stats["num_chromatograms"] = int(chromatogram_list.get("count", 0))
        else:
            stats["num_chromatograms"] = len(root.findall(f".//{ns}chromatogram"))
        
        # Count software entries
        stats["num_software"] = len(root.findall(f".//{ns}software"))
        
        # Extract spectra metadata for distribution stats
        spectra_meta = self._get_mzml_spectra_metadata(root)
        if spectra_meta:
            ms_levels = [s["ms_level"] for s in spectra_meta if s["ms_level"] is not None]
            rts = [s["scan_start_time"] for s in spectra_meta if s["scan_start_time"] is not None]
            tics = [s["total_ion_current"] for s in spectra_meta if s["total_ion_current"] is not None]
            
            if ms_levels:
                stats["ms_level_distribution"] = dict(Counter(ms_levels))
            if rts:
                stats["rt_range"] = [float(min(rts)), float(max(rts))]
            if tics:
                stats["tic_range"] = [float(min(tics)), float(max(tics))]
        
        return stats
    
    def _get_idxml_stats(self, root: ET.Element) -> Dict[str, Any]:
        """Extract idXML-specific statistics.
        
        Counts search parameters, protein identifications, protein hits,
        peptide identifications, peptide hits, and computes distributions for
        scores, charges, RT, m/z.
        
        Args:
            root: Root XML element (<IdXML>)
            
        Returns:
            Dictionary with idXML-specific statistics
        """
        stats = {
            "num_search_parameters": 0,
            "num_identification_runs": 0,
            "num_protein_identifications": 0,
            "num_protein_hits": 0,
            "num_peptide_identifications": 0,
            "num_peptide_hits": 0,
        }
        
        for elem in root.iter():
            tag = self._normalize_tag(elem.tag)
            if tag == "SearchParameters":
                stats["num_search_parameters"] += 1
            elif tag == "IdentificationRun":
                stats["num_identification_runs"] += 1
            elif tag == "ProteinIdentification":
                stats["num_protein_identifications"] += 1
            elif tag == "ProteinHit":
                stats["num_protein_hits"] += 1
            elif tag == "PeptideIdentification":
                stats["num_peptide_identifications"] += 1
            elif tag == "PeptideHit":
                stats["num_peptide_hits"] += 1
        
        # Extract peptide identification details for distribution stats
        peptide_ids = self._get_idxml_peptide_identifications(root)
        if peptide_ids:
            rts = [p["rt"] for p in peptide_ids if p["rt"] is not None]
            mzs = [p["mz"] for p in peptide_ids if p["mz"] is not None]
            
            if rts:
                stats["rt_range"] = [float(min(rts)), float(max(rts))]
            if mzs:
                stats["mz_range"] = [float(min(mzs)), float(max(mzs))]
            
            # Score distribution from best hits
            all_scores = []
            all_charges = []
            for pid in peptide_ids:
                for hit in pid.get("hits", []):
                    if hit["score"] is not None:
                        all_scores.append(hit["score"])
                    if hit["charge"] is not None:
                        all_charges.append(hit["charge"])
            
            if all_scores:
                stats["score_range"] = [float(min(all_scores)), float(max(all_scores))]
            if all_charges:
                stats["charge_distribution"] = dict(Counter(int(c) for c in all_charges))
        
        # Extract protein hit accessions for distribution stats
        protein_hits = self._get_idxml_protein_hits(root)
        if protein_hits:
            accessions = [p["accession"] for p in protein_hits if p["accession"]]
            stats["num_unique_proteins"] = len(set(accessions))
        
        return stats
    
    def _get_featurexml_features(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Extract top-level feature data from featureXML root element.
        
        Finds <featureList> and extracts direct <feature> children (top-level features only,
        excluding subordinate features). Each feature has: RT (retention time), m/z, intensity,
        charge, quality, overallquality. Position dim="0" is RT, position dim="1" is m/z.
        
        Args:
            root: Root XML element (featureMap)
            
        Returns:
            List of feature dictionaries with keys: id, rt, mz, intensity, charge, quality, overallquality
        """
        features = []
        
        # Find featureList element (handles both namespaced and non-namespaced)
        feature_list = None
        for elem in root.iter():
            if self._normalize_tag(elem.tag) == "featureList":
                feature_list = elem
                break
        
        if feature_list is None:
            return features
        
        # Get direct feature children of featureList (top-level features only)
        for child in feature_list:
            if self._normalize_tag(child.tag) == "feature":
                feature = self._parse_feature_element(child)
                features.append(feature)
        
        return features
    
    def _parse_feature_element(self, feature_elem: ET.Element) -> Dict[str, Any]:
        """Parse a single featureXML <feature> element into a dictionary.
        
        Extracts: id, RT (position dim=0), m/z (position dim=1), intensity, charge,
        quality (dim=0), overallquality.
        
        Args:
            feature_elem: XML element for a single feature
            
        Returns:
            Dictionary with feature data
        """
        feature = {
            "id": feature_elem.get("id"),
            "rt": None,
            "mz": None,
            "intensity": None,
            "charge": None,
            "quality": None,
            "overallquality": None,
        }
        
        for child in feature_elem:
            child_tag = self._normalize_tag(child.tag)
            
            if child_tag == "position":
                dim = child.get("dim")
                try:
                    val = float(child.text.strip()) if child.text else None
                except (ValueError, TypeError):
                    val = None
                if dim == "0":
                    feature["rt"] = val
                elif dim == "1":
                    feature["mz"] = val
            
            elif child_tag == "intensity":
                try:
                    feature["intensity"] = float(child.text.strip()) if child.text else None
                except (ValueError, TypeError):
                    pass
            
            elif child_tag == "charge":
                try:
                    feature["charge"] = int(child.text.strip()) if child.text else None
                except (ValueError, TypeError):
                    pass
            
            elif child_tag == "quality":
                dim = child.get("dim")
                try:
                    val = float(child.text.strip()) if child.text else None
                except (ValueError, TypeError):
                    val = None
                if dim == "0":
                    feature["quality"] = val
            
            elif child_tag == "overallquality":
                try:
                    feature["overallquality"] = float(child.text.strip()) if child.text else None
                except (ValueError, TypeError):
                    pass
        
        return feature
    
    def _get_mzml_spectra_metadata(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Extract per-spectrum metadata from mzML root element.
        
        Parses cvParam elements for key metadata without decoding binary data arrays.
        Common CV accessions used:
        - MS:1000511 = ms level
        - MS:1000016 = scan start time
        - MS:1000285 = total ion current
        - MS:1000504 = base peak m/z
        - MS:1000505 = base peak intensity
        - MS:1000127 = centroid spectrum
        - MS:1000128 = profile spectrum
        
        Args:
            root: Root XML element (mzML)
            
        Returns:
            List of spectrum metadata dictionaries
        """
        ns = self._get_namespace(root)
        spectra = []
        
        for spectrum_elem in root.findall(f".//{ns}spectrum"):
            meta = {
                "index": spectrum_elem.get("index"),
                "id": spectrum_elem.get("id"),
                "default_array_length": spectrum_elem.get("defaultArrayLength"),
                "ms_level": None,
                "scan_start_time": None,
                "total_ion_current": None,
                "base_peak_mz": None,
                "base_peak_intensity": None,
                "is_centroid": None,
            }
            
            # Parse index and array length as int
            try:
                meta["index"] = int(meta["index"]) if meta["index"] is not None else None
            except (ValueError, TypeError):
                pass
            
            try:
                meta["default_array_length"] = int(meta["default_array_length"]) if meta["default_array_length"] is not None else None
            except (ValueError, TypeError):
                pass
            
            # Parse cvParam elements (both at spectrum level and inside scan elements)
            for cv_param in spectrum_elem.iter(f"{ns}cvParam"):
                accession = cv_param.get("accession", "")
                value = cv_param.get("value", "")
                
                if accession == "MS:1000511":  # ms level
                    try:
                        meta["ms_level"] = int(value)
                    except (ValueError, TypeError):
                        pass
                elif accession == "MS:1000016":  # scan start time
                    try:
                        meta["scan_start_time"] = float(value)
                    except (ValueError, TypeError):
                        pass
                elif accession == "MS:1000285":  # total ion current
                    try:
                        meta["total_ion_current"] = float(value)
                    except (ValueError, TypeError):
                        pass
                elif accession == "MS:1000504":  # base peak m/z
                    try:
                        meta["base_peak_mz"] = float(value)
                    except (ValueError, TypeError):
                        pass
                elif accession == "MS:1000505":  # base peak intensity
                    try:
                        meta["base_peak_intensity"] = float(value)
                    except (ValueError, TypeError):
                        pass
                elif accession == "MS:1000127":  # centroid spectrum
                    meta["is_centroid"] = True
                elif accession == "MS:1000128":  # profile spectrum
                    meta["is_centroid"] = False
            
            spectra.append(meta)
        
        return spectra
    
    def _get_mzml_chromatograms_metadata(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Extract chromatogram metadata from mzML root element.
        
        Args:
            root: Root XML element (mzML)
            
        Returns:
            List of chromatogram metadata dictionaries
        """
        ns = self._get_namespace(root)
        chromatograms = []
        
        for chrom_elem in root.findall(f".//{ns}chromatogram"):
            meta = {
                "index": chrom_elem.get("index"),
                "id": chrom_elem.get("id"),
                "default_array_length": chrom_elem.get("defaultArrayLength"),
            }
            
            try:
                meta["index"] = int(meta["index"]) if meta["index"] is not None else None
            except (ValueError, TypeError):
                pass
            
            try:
                meta["default_array_length"] = int(meta["default_array_length"]) if meta["default_array_length"] is not None else None
            except (ValueError, TypeError):
                pass
            
            chromatograms.append(meta)
        
        return chromatograms
    
    def _get_idxml_peptide_identifications(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Extract peptide identification data from idXML root element.
        
        Each PeptideIdentification has attributes: score_type, higher_score_better,
        significance_threshold, MZ, RT, spectrum_reference, and contains PeptideHit children.
        
        Args:
            root: Root XML element (<IdXML>)
            
        Returns:
            List of peptide identification dictionaries with nested hits
        """
        peptide_ids = []
        
        for elem in root.iter():
            if self._normalize_tag(elem.tag) != "PeptideIdentification":
                continue
            
            pid = {
                "score_type": elem.get("score_type"),
                "higher_score_better": elem.get("higher_score_better", "").lower() == "true",
                "significance_threshold": None,
                "mz": None,
                "rt": None,
                "spectrum_reference": elem.get("spectrum_reference"),
                "hits": [],
            }
            
            try:
                pid["significance_threshold"] = float(elem.get("significance_threshold", ""))
            except (ValueError, TypeError):
                pass
            try:
                pid["mz"] = float(elem.get("MZ", ""))
            except (ValueError, TypeError):
                pass
            try:
                pid["rt"] = float(elem.get("RT", ""))
            except (ValueError, TypeError):
                pass
            
            # Parse PeptideHit children
            for child in elem:
                if self._normalize_tag(child.tag) != "PeptideHit":
                    continue
                
                hit = {
                    "score": None,
                    "sequence": child.get("sequence"),
                    "charge": None,
                    "aa_before": child.get("aa_before"),
                    "aa_after": child.get("aa_after"),
                    "start": None,
                    "end": None,
                    "protein_refs": child.get("protein_refs"),
                }
                
                try:
                    hit["score"] = float(child.get("score", ""))
                except (ValueError, TypeError):
                    pass
                try:
                    hit["charge"] = int(child.get("charge", ""))
                except (ValueError, TypeError):
                    pass
                try:
                    hit["start"] = int(child.get("start", ""))
                except (ValueError, TypeError):
                    pass
                try:
                    hit["end"] = int(child.get("end", ""))
                except (ValueError, TypeError):
                    pass
                
                pid["hits"].append(hit)
            
            peptide_ids.append(pid)
        
        return peptide_ids
    
    def _get_idxml_protein_hits(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Extract protein hit data from idXML root element.
        
        ProteinHit elements are found inside ProteinIdentification elements.
        Each has: id, accession, score, sequence, and optional target_decoy UserParam.
        
        Args:
            root: Root XML element (<IdXML>)
            
        Returns:
            List of protein hit dictionaries
        """
        protein_hits = []
        
        for elem in root.iter():
            if self._normalize_tag(elem.tag) != "ProteinHit":
                continue
            
            hit = {
                "id": elem.get("id"),
                "accession": elem.get("accession"),
                "score": None,
                "sequence": elem.get("sequence", ""),
                "target_decoy": None,
            }
            
            try:
                hit["score"] = float(elem.get("score", ""))
            except (ValueError, TypeError):
                pass
            
            # Check for target_decoy UserParam
            for child in elem:
                if self._normalize_tag(child.tag) == "UserParam" and child.get("name") == "target_decoy":
                    hit["target_decoy"] = child.get("value")
            
            protein_hits.append(hit)
        
        return protein_hits
    
    # ============================================================================
    # Element Comparison Helpers
    # ============================================================================
    
    def _compare_elements_exact(self, ref_elem: ET.Element, alt_elem: ET.Element) -> bool:
        """Compare two XML elements exactly (recursive).
        
        Compares tag, attributes, text, tail, and all children recursively.
        
        Args:
            ref_elem: Reference element
            alt_elem: Candidate element
            
        Returns:
            True if elements are identical
        """
        # Compare tags
        if ref_elem.tag != alt_elem.tag:
            return False
        
        # Compare attributes
        if ref_elem.attrib != alt_elem.attrib:
            return False
        
        # Compare text content
        ref_text = ref_elem.text.strip() if ref_elem.text else ""
        alt_text = alt_elem.text.strip() if alt_elem.text else ""
        if ref_text != alt_text:
            return False
        
        # Compare tail text
        ref_tail = ref_elem.tail.strip() if ref_elem.tail else ""
        alt_tail = alt_elem.tail.strip() if alt_elem.tail else ""
        if ref_tail != alt_tail:
            return False
        
        # Compare children count
        ref_children = list(ref_elem)
        alt_children = list(alt_elem)
        if len(ref_children) != len(alt_children):
            return False
        
        # Recursively compare children
        for ref_child, alt_child in zip(ref_children, alt_children):
            if not self._compare_elements_exact(ref_child, alt_child):
                return False
        
        return True
    
    def _compare_element_fields(self, ref_elem: ET.Element, alt_elem: ET.Element,
                                ignore_namespaces: bool = True) -> bool:
        """Compare two XML elements' tags, attributes, and text content (non-recursive).
        
        Similar to BAM's _compare_record_fields, this compares the immediate data
        of an element without descending into children.
        
        Args:
            ref_elem: Reference element
            alt_elem: Candidate element
            ignore_namespaces: Whether to normalize tags by removing namespaces
            
        Returns:
            True if element fields match
        """
        # Compare tags (with optional namespace normalization)
        ref_tag = self._normalize_tag(ref_elem.tag, ignore_namespaces)
        alt_tag = self._normalize_tag(alt_elem.tag, ignore_namespaces)
        if ref_tag != alt_tag:
            return False
        
        # Compare attributes
        if ref_elem.attrib != alt_elem.attrib:
            return False
        
        # Compare text content
        ref_text = ref_elem.text.strip() if ref_elem.text else ""
        alt_text = alt_elem.text.strip() if alt_elem.text else ""
        if ref_text != alt_text:
            return False
        
        return True
    
    def _compare_feature_concord(self, ref_feature: Dict[str, Any], alt_feature: Dict[str, Any],
                                 tolerance: float = 1e-6) -> bool:
        """Compare two featureXML features for concordance.
        
        Assumes features have already been matched by (RT, m/z) proximity.
        Checks if intensity, charge, and quality values agree within tolerance.
        
        Args:
            ref_feature: Reference feature dict (from _get_featurexml_features)
            alt_feature: Candidate feature dict (from _get_featurexml_features)
            tolerance: Relative tolerance for numeric comparison
            
        Returns:
            True if features are concordant
        """
        # Check charge match (must be exact)
        ref_charge = ref_feature.get("charge")
        alt_charge = alt_feature.get("charge")
        if ref_charge is not None and alt_charge is not None and ref_charge != alt_charge:
            return False
        
        # Check intensity within tolerance
        ref_intensity = ref_feature.get("intensity")
        alt_intensity = alt_feature.get("intensity")
        
        if ref_intensity is not None and alt_intensity is not None:
            denom = max(abs(ref_intensity), abs(alt_intensity), 1.0)
            if abs(ref_intensity - alt_intensity) > tolerance * denom:
                return False
        
        return True
    
    def _compare_spectrum_concord(self, ref_spectrum: Dict[str, Any], alt_spectrum: Dict[str, Any],
                                  tolerance: float = 1e-6) -> bool:
        """Compare two mzML spectra for concordance using metadata.
        
        Compares ms_level (exact), base_peak_mz, base_peak_intensity, total_ion_current,
        and default_array_length with relative tolerance. Assumes spectra have already been
        matched by index or RT.
        
        Args:
            ref_spectrum: Reference spectrum metadata dict
            alt_spectrum: Candidate spectrum metadata dict
            tolerance: Relative tolerance for numeric comparison
            
        Returns:
            True if spectra metadata are concordant
        """
        # ms_level must match exactly
        if ref_spectrum.get("ms_level") != alt_spectrum.get("ms_level"):
            return False
        
        # Compare numeric metadata with tolerance
        numeric_fields = ["base_peak_mz", "base_peak_intensity", "total_ion_current"]
        for field in numeric_fields:
            ref_val = ref_spectrum.get(field)
            alt_val = alt_spectrum.get(field)
            
            if ref_val is None and alt_val is None:
                continue
            if ref_val is None or alt_val is None:
                return False
            
            denom = max(abs(ref_val), abs(alt_val), 1.0)
            if abs(ref_val - alt_val) > tolerance * denom:
                return False
        
        # Compare array lengths (must be exact)
        ref_len = ref_spectrum.get("default_array_length")
        alt_len = alt_spectrum.get("default_array_length")
        if ref_len is not None and alt_len is not None and ref_len != alt_len:
            return False
        
        return True
    
    # ============================================================================
    # Private Comparison Implementation Methods
    # ============================================================================
    
    def _compare_exact(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Exact match: All XML elements identical in structure and content.
        
        Compares XML trees element-by-element in depth-first order.
        All tags, attributes, text content, and structure must match exactly.
        Similar to BAM's exact strategy that compares record-by-record.
        
        Criteria:
        - All elements identical in the same DFS order
        - Compare tags, attributes, text content, tail text
        - Compare children count and recurse
        - Must be 100% identical for similarity = 1.0
        """
        try:
            ref_root = self._load_file(ref_validation.file_path, file_format=ref_validation.signature.format)
            alt_root = self._load_file(alt_validation.file_path, file_format=alt_validation.signature.format)
            
            if ref_root is None or alt_root is None:
                self.result.error = "Failed to load XML files"
                return
            
            # Quick check: compare element counts before detailed comparison
            ref_elements = list(ref_root.iter())
            alt_elements = list(alt_root.iter())
            
            if len(ref_elements) != len(alt_elements):
                self.result.similarity = 0.0
                self.result.details.update({
                    "total_ref_elements": len(ref_elements),
                    "total_alt_elements": len(alt_elements),
                    "identical": False,
                })
                return
            
            # Compare element-by-element in DFS order (like BAM's record-by-record)
            total_elements = len(ref_elements)
            identical_elements = 0
            mismatched_elements = 0
            
            for ref_elem, alt_elem in zip(ref_elements, alt_elements):
                if self._compare_element_fields(ref_elem, alt_elem, ignore_namespaces=False):
                    identical_elements += 1
                else:
                    mismatched_elements += 1
            
            # Also verify full tree structure via recursive comparison
            is_identical = self._compare_elements_exact(ref_root, alt_root)
            
            # Similarity must be 100% for exact match
            similarity = 1.0 if is_identical else 0.0
            self.result.similarity = similarity
            self.result.details.update({
                "total_ref_elements": len(ref_elements),
                "total_alt_elements": len(alt_elements),
                "total_elements": total_elements,
                "identical_elements": identical_elements,
                "mismatched_elements": mismatched_elements,
                "identical": is_identical,
            })
            
        except Exception as e:
            logger.error(f"Exact comparison failed: {e}")
            self.result.error = f"Exact comparison failed: {e}"
    
    def _compare_approx(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Approximate match: features/spectra/identifications matched by proximity with numeric tolerance.
        
        For featureXML:
        - Match features by (RT, m/z) proximity within rt_tolerance and mz_tolerance
        - Compare matched features' intensity and charge with relative tolerance
        - Calculate precision, recall, F1, Jaccard similarity
        
        For mzML:
        - Match spectra by scan index (or RT proximity if indices differ)
        - Compare matched spectra metadata (ms_level, base_peak, TIC) with tolerance
        - Also compares chromatogram matching by ID
        - Calculate precision, recall, F1, Jaccard similarity
        
        For idXML:
        - Match peptide identifications by (RT, m/z) proximity
        - Compare matched peptide hits by sequence, charge, and score with tolerance
        - Also compares protein hit sets by accession (Jaccard)
        - Calculate precision, recall, F1, Jaccard similarity
        
        Args:
            **kwargs: Strategy-specific parameters
            - tolerance: Relative tolerance for numeric comparisons (default: 1e-6)
            - rt_tolerance: RT matching window in seconds (default: 10.0)
            - mz_tolerance: m/z matching window in Da (default: 0.01)
        """
        try:
            file_format = ref_validation.signature.format
            
            ref_root = self._load_file(ref_validation.file_path, file_format=file_format)
            alt_root = self._load_file(alt_validation.file_path, file_format=file_format)
            
            if ref_root is None or alt_root is None:
                self.result.error = "Failed to load XML files"
                return
            
            if file_format == "featurexml":
                similarity, details = self._compare_approx_featurexml(ref_root, alt_root, **kwargs)
            elif file_format == "mzml":
                similarity, details = self._compare_approx_mzml(ref_root, alt_root, **kwargs)
            elif file_format == "idxml":
                similarity, details = self._compare_approx_idxml(ref_root, alt_root, **kwargs)
            else:
                self.result.error = f"Unsupported format for approximate comparison: {file_format}"
                return
            
            self.result.similarity = similarity
            self.result.details.update(details)
            
        except Exception as e:
            logger.error(f"Approximate comparison failed: {e}")
            self.result.error = f"Approximate comparison failed: {e}"
    
    def _compare_approx_featurexml(self, ref_root: ET.Element, alt_root: ET.Element,
                                   **kwargs) -> Tuple[float, Dict[str, Any]]:
        """Approximate comparison for featureXML files.
        
        Matches features by (RT, m/z) proximity using greedy nearest-neighbor,
        then checks concordance (intensity, charge) within tolerance.
        Similar to BAM's _compare_approx_chunk that matches reads by QNAME.
        
        Args:
            ref_root: Reference XML root
            alt_root: Candidate XML root
            **kwargs: tolerance, rt_tolerance, mz_tolerance
            
        Returns:
            Tuple of (similarity, details_dict)
        """
        tolerance = kwargs.get("tolerance", 1e-6)
        rt_tolerance = kwargs.get("rt_tolerance", 10.0)
        mz_tolerance = kwargs.get("mz_tolerance", 0.01)
        
        ref_features = self._get_featurexml_features(ref_root)
        alt_features = self._get_featurexml_features(alt_root)
        
        total_ref = len(ref_features)
        total_alt = len(alt_features)
        
        if total_ref == 0 and total_alt == 0:
            return 1.0, {"total_ref_features": 0, "total_alt_features": 0,
                         "matched_features": 0, "concordant_features": 0}
        
        if total_ref == 0 or total_alt == 0:
            return 0.0, {"total_ref_features": total_ref, "total_alt_features": total_alt,
                         "matched_features": 0, "concordant_features": 0}
        
        # Greedy nearest-neighbor matching by (RT, m/z) proximity
        matched = 0
        concordant = 0
        alt_used = set()
        
        for ref_feat in ref_features:
            ref_rt = ref_feat.get("rt")
            ref_mz = ref_feat.get("mz")
            
            if ref_rt is None or ref_mz is None:
                continue
            
            best_idx = None
            best_dist = float("inf")
            
            for j, alt_feat in enumerate(alt_features):
                if j in alt_used:
                    continue
                
                alt_rt = alt_feat.get("rt")
                alt_mz = alt_feat.get("mz")
                
                if alt_rt is None or alt_mz is None:
                    continue
                
                rt_diff = abs(ref_rt - alt_rt)
                mz_diff = abs(ref_mz - alt_mz)
                
                if rt_diff <= rt_tolerance and mz_diff <= mz_tolerance:
                    # Normalized distance for ranking matches
                    dist = (rt_diff / rt_tolerance) ** 2 + (mz_diff / mz_tolerance) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = j
            
            if best_idx is not None:
                alt_used.add(best_idx)
                matched += 1
                if self._compare_feature_concord(ref_feat, alt_features[best_idx], tolerance=tolerance):
                    concordant += 1
        
        precision, recall, f1_score = precision_recall_f1(total_ref, total_alt, concordant)
        jaccard = jaccard_similarity(total_ref, total_alt, concordant)
        
        details = {
            "total_ref_features": total_ref,
            "total_alt_features": total_alt,
            "matched_features": matched,
            "concordant_features": concordant,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "jaccard_similarity": jaccard,
            "tolerance": tolerance,
            "rt_tolerance": rt_tolerance,
            "mz_tolerance": mz_tolerance,
        }
        
        return jaccard, details
    
    def _compare_approx_mzml(self, ref_root: ET.Element, alt_root: ET.Element,
                             **kwargs) -> Tuple[float, Dict[str, Any]]:
        """Approximate comparison for mzML files.
        
        Matches spectra by scan index, then compares metadata with tolerance.
        Falls back to RT-based matching for spectra without matching indices.
        Also compares chromatograms by ID matching.
        Similar to BAM's _compare_approx_chunk that matches reads by QNAME.
        
        Args:
            ref_root: Reference XML root
            alt_root: Candidate XML root
            **kwargs: tolerance, rt_tolerance
            
        Returns:
            Tuple of (similarity, details_dict)
        """
        tolerance = kwargs.get("tolerance", 1e-6)
        rt_tolerance = kwargs.get("rt_tolerance", 10.0)
        
        ref_spectra = self._get_mzml_spectra_metadata(ref_root)
        alt_spectra = self._get_mzml_spectra_metadata(alt_root)
        
        total_ref = len(ref_spectra)
        total_alt = len(alt_spectra)
        
        if total_ref == 0 and total_alt == 0:
            return 1.0, {"total_ref_spectra": 0, "total_alt_spectra": 0,
                         "matched_spectra": 0, "concordant_spectra": 0}
        
        if total_ref == 0 or total_alt == 0:
            return 0.0, {"total_ref_spectra": total_ref, "total_alt_spectra": total_alt,
                         "matched_spectra": 0, "concordant_spectra": 0}
        
        # Build alt index lookup for O(1) matching
        alt_by_index = {}
        for s in alt_spectra:
            idx = s.get("index")
            if idx is not None:
                alt_by_index[idx] = s
        
        matched = 0
        concordant = 0
        ref_matched_indices = set()
        alt_used_indices = set()
        
        # Phase 1: Index-based matching (primary strategy)
        if alt_by_index:
            for i, ref_spec in enumerate(ref_spectra):
                ref_idx = ref_spec.get("index")
                if ref_idx is not None and ref_idx in alt_by_index and ref_idx not in alt_used_indices:
                    alt_spec = alt_by_index[ref_idx]
                    alt_used_indices.add(ref_idx)
                    ref_matched_indices.add(i)
                    matched += 1
                    if self._compare_spectrum_concord(ref_spec, alt_spec, tolerance=tolerance):
                        concordant += 1
        
        # Phase 2: RT-based fallback matching for unmatched spectra
        unmatched_ref = [(i, s) for i, s in enumerate(ref_spectra) if i not in ref_matched_indices]
        unmatched_alt = [(j, s) for j, s in enumerate(alt_spectra)
                         if s.get("index") not in alt_used_indices or s.get("index") is None]
        
        if unmatched_ref and unmatched_alt:
            alt_rt_used = set()
            for _i, ref_spec in unmatched_ref:
                ref_rt = ref_spec.get("scan_start_time")
                if ref_rt is None:
                    continue
                
                best_j = None
                best_diff = float("inf")
                
                for j_idx, (j, alt_spec) in enumerate(unmatched_alt):
                    if j_idx in alt_rt_used:
                        continue
                    alt_rt = alt_spec.get("scan_start_time")
                    if alt_rt is None:
                        continue
                    diff = abs(ref_rt - alt_rt)
                    if diff <= rt_tolerance and diff < best_diff:
                        best_diff = diff
                        best_j = j_idx
                
                if best_j is not None:
                    alt_rt_used.add(best_j)
                    matched += 1
                    if self._compare_spectrum_concord(ref_spec, unmatched_alt[best_j][1], tolerance=tolerance):
                        concordant += 1
        
        # Phase 3: Chromatogram matching by ID
        ref_chroms = self._get_mzml_chromatograms_metadata(ref_root)
        alt_chroms = self._get_mzml_chromatograms_metadata(alt_root)
        
        chrom_matched = 0
        alt_chrom_ids = {c.get("id"): c for c in alt_chroms}
        for ref_chrom in ref_chroms:
            ref_id = ref_chrom.get("id")
            if ref_id and ref_id in alt_chrom_ids:
                chrom_matched += 1
        
        chrom_total_ref = len(ref_chroms)
        chrom_total_alt = len(alt_chroms)
        
        # Combine spectra and chromatogram metrics
        total_items_ref = total_ref + chrom_total_ref
        total_items_alt = total_alt + chrom_total_alt
        total_concordant = concordant + chrom_matched
        
        precision, recall, f1_score = precision_recall_f1(total_items_ref, total_items_alt, total_concordant)
        jaccard = jaccard_similarity(total_items_ref, total_items_alt, total_concordant)
        
        details = {
            "total_ref_spectra": total_ref,
            "total_alt_spectra": total_alt,
            "matched_spectra": matched,
            "concordant_spectra": concordant,
            "total_ref_chromatograms": chrom_total_ref,
            "total_alt_chromatograms": chrom_total_alt,
            "matched_chromatograms": chrom_matched,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "jaccard_similarity": jaccard,
            "tolerance": tolerance,
            "rt_tolerance": rt_tolerance,
        }
        
        return jaccard, details
    
    def _compare_approx_idxml(self, ref_root: ET.Element, alt_root: ET.Element,
                              **kwargs) -> Tuple[float, Dict[str, Any]]:
        """Approximate comparison for idXML files.
        
        Phase 1: Match PeptideIdentifications by (RT, m/z) proximity, then compare
        their PeptideHits by sequence and charge. Score concordance checked with tolerance.
        Phase 2: Compare ProteinHit sets by accession (Jaccard similarity on accession sets).
        Combined similarity uses weighted average of peptide F1 and protein Jaccard.
        
        Args:
            ref_root: Reference XML root (<IdXML>)
            alt_root: Candidate XML root (<IdXML>)
            **kwargs: tolerance, rt_tolerance, mz_tolerance
            
        Returns:
            Tuple of (similarity, details_dict)
        """
        tolerance = kwargs.get("tolerance", 1e-6)
        rt_tolerance = kwargs.get("rt_tolerance", 10.0)
        mz_tolerance = kwargs.get("mz_tolerance", 0.01)
        
        # Phase 1: Peptide identification matching by (RT, m/z) proximity
        ref_pids = self._get_idxml_peptide_identifications(ref_root)
        alt_pids = self._get_idxml_peptide_identifications(alt_root)
        
        total_ref_pids = len(ref_pids)
        total_alt_pids = len(alt_pids)
        
        matched_pids = 0
        concordant_pids = 0
        alt_used = set()
        
        for ref_pid in ref_pids:
            ref_rt = ref_pid.get("rt")
            ref_mz = ref_pid.get("mz")
            
            if ref_rt is None or ref_mz is None:
                continue
            
            best_idx = None
            best_dist = float("inf")
            
            for j, alt_pid in enumerate(alt_pids):
                if j in alt_used:
                    continue
                
                alt_rt = alt_pid.get("rt")
                alt_mz = alt_pid.get("mz")
                
                if alt_rt is None or alt_mz is None:
                    continue
                
                rt_diff = abs(ref_rt - alt_rt)
                mz_diff = abs(ref_mz - alt_mz)
                
                if rt_diff <= rt_tolerance and mz_diff <= mz_tolerance:
                    dist = (rt_diff / rt_tolerance) ** 2 + (mz_diff / mz_tolerance) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = j
            
            if best_idx is not None:
                alt_used.add(best_idx)
                matched_pids += 1
                if self._compare_peptide_id_concord(ref_pid, alt_pids[best_idx], tolerance=tolerance):
                    concordant_pids += 1
        
        pid_precision, pid_recall, pid_f1 = precision_recall_f1(total_ref_pids, total_alt_pids, concordant_pids)
        pid_jaccard = jaccard_similarity(total_ref_pids, total_alt_pids, concordant_pids)
        
        # Phase 2: Protein hit matching by accession
        ref_proteins = self._get_idxml_protein_hits(ref_root)
        alt_proteins = self._get_idxml_protein_hits(alt_root)
        
        ref_accessions = set(p["accession"] for p in ref_proteins if p["accession"])
        alt_accessions = set(p["accession"] for p in alt_proteins if p["accession"])
        
        if ref_accessions or alt_accessions:
            protein_intersection = len(ref_accessions & alt_accessions)
            protein_union = len(ref_accessions | alt_accessions)
            protein_jaccard = protein_intersection / protein_union if protein_union > 0 else 1.0
        else:
            protein_jaccard = 1.0
            protein_intersection = 0
        
        # Combined similarity: weighted average of peptide Jaccard and protein Jaccard
        # Weight peptide matching more heavily since it's the primary content
        if total_ref_pids > 0 or total_alt_pids > 0:
            similarity = 0.7 * pid_jaccard + 0.3 * protein_jaccard
        else:
            similarity = protein_jaccard
        
        details = {
            "total_ref_peptide_ids": total_ref_pids,
            "total_alt_peptide_ids": total_alt_pids,
            "matched_peptide_ids": matched_pids,
            "concordant_peptide_ids": concordant_pids,
            "peptide_precision": pid_precision,
            "peptide_recall": pid_recall,
            "peptide_f1_score": pid_f1,
            "peptide_jaccard_similarity": pid_jaccard,
            "total_ref_proteins": len(ref_accessions),
            "total_alt_proteins": len(alt_accessions),
            "shared_proteins": protein_intersection if (ref_accessions or alt_accessions) else 0,
            "protein_jaccard_similarity": protein_jaccard,
            "tolerance": tolerance,
            "rt_tolerance": rt_tolerance,
            "mz_tolerance": mz_tolerance,
        }
        
        return similarity, details
    
    def _compare_peptide_id_concord(self, ref_pid: Dict[str, Any], alt_pid: Dict[str, Any],
                                     tolerance: float = 1e-6) -> bool:
        """Compare two matched PeptideIdentification elements for concordance.
        
        Assumes PeptideIdentifications have already been matched by (RT, m/z) proximity.
        Checks if their top PeptideHit(s) agree on sequence and charge, and if scores
        are within tolerance.
        
        Args:
            ref_pid: Reference PeptideIdentification dict
            alt_pid: Candidate PeptideIdentification dict
            tolerance: Relative tolerance for score comparison
            
        Returns:
            True if peptide identifications are concordant
        """
        ref_hits = ref_pid.get("hits", [])
        alt_hits = alt_pid.get("hits", [])
        
        if not ref_hits and not alt_hits:
            return True
        if not ref_hits or not alt_hits:
            return False
        
        # Compare top hit (rank 1) - must match on sequence and charge
        ref_top = ref_hits[0]
        alt_top = alt_hits[0]
        
        # Sequence must match exactly
        if ref_top.get("sequence") != alt_top.get("sequence"):
            return False
        
        # Charge must match exactly
        if ref_top.get("charge") != alt_top.get("charge"):
            return False
        
        # Score within tolerance
        ref_score = ref_top.get("score")
        alt_score = alt_top.get("score")
        if ref_score is not None and alt_score is not None:
            denom = max(abs(ref_score), abs(alt_score), 1.0)
            if abs(ref_score - alt_score) > tolerance * denom:
                return False
        
        return True
    
    def _compare_summary(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Summary statistics comparison: element counts, distributions, numeric value distributions.
        
        Combines multiple metrics (similar to BAM's summary that combines flagstat RAE + MAPQ Hellinger):
        1. Count-based similarity using RAE for format-specific counts (like flagstat RAE)
        2. Element type distribution similarity using Hellinger distance (like MAPQ Hellinger)
        3. Attribute name distribution similarity using Hellinger distance
        4. Numeric value distribution similarity using Hellinger distance on histograms
        
        Final similarity = 1 - nanmean(all_errors)
        """
        try:
            ref_root = self._load_file(ref_validation.file_path, file_format=ref_validation.signature.format)
            alt_root = self._load_file(alt_validation.file_path, file_format=alt_validation.signature.format)
            
            if ref_root is None or alt_root is None:
                self.result.error = "Failed to load XML files"
                return
            
            file_format = ref_validation.signature.format
            
            # Get statistics
            ref_stats = self._get_xml_stats(ref_root, file_format)
            alt_stats = self._get_xml_stats(alt_root, file_format)
            
            # 1. Count-based similarity using RAE (like BAM's flagstat RAE)
            count_fields = ["num_elements", "num_attributes"]
            if file_format == "featurexml":
                count_fields.extend(["num_features", "num_subordinates", "num_user_params", "num_cv_params"])
            elif file_format == "mzml":
                count_fields.extend(["num_spectra", "num_chromatograms", "num_software"])
            elif file_format == "idxml":
                count_fields.extend(["num_protein_hits", "num_peptide_identifications", "num_peptide_hits"])
            
            ref_counts = np.array([ref_stats.get(f, 0) for f in count_fields], dtype=float)
            alt_counts = np.array([alt_stats.get(f, 0) for f in count_fields], dtype=float)
            count_rae = relative_absolute_error(ref_counts, alt_counts)
            count_rae_clipped = np.clip(count_rae, 0.0, 1.0)
            
            # 2. Element type distribution similarity (Hellinger distance, like BAM's MAPQ Hellinger)
            ref_elem_types = Counter(ref_stats.get("element_types", {}))
            alt_elem_types = Counter(alt_stats.get("element_types", {}))
            
            all_elem_types = sorted(set(ref_elem_types.keys()) | set(alt_elem_types.keys()))
            ref_elem_arr = np.array([ref_elem_types.get(k, 0) for k in all_elem_types])
            alt_elem_arr = np.array([alt_elem_types.get(k, 0) for k in all_elem_types])
            
            elem_hellinger = hellinger_distance(ref_elem_arr, alt_elem_arr)
            
            # 3. Attribute name distribution similarity (Hellinger distance)
            ref_attr_names = Counter(ref_stats.get("attribute_names", {}))
            alt_attr_names = Counter(alt_stats.get("attribute_names", {}))
            
            all_attr_names = sorted(set(ref_attr_names.keys()) | set(alt_attr_names.keys()))
            ref_attr_arr = np.array([ref_attr_names.get(k, 0) for k in all_attr_names])
            alt_attr_arr = np.array([alt_attr_names.get(k, 0) for k in all_attr_names])
            
            attr_hellinger = hellinger_distance(ref_attr_arr, alt_attr_arr)
            
            # 4. Numeric value distribution similarity (Hellinger on histograms)
            ref_numeric = np.array(ref_stats.get("numeric_values", []))
            alt_numeric = np.array(alt_stats.get("numeric_values", []))
            
            numeric_hellinger = np.nan
            if len(ref_numeric) > 0 and len(alt_numeric) > 0:
                try:
                    all_values = np.concatenate([ref_numeric, alt_numeric])
                    n_bins = min(50, max(2, len(all_values) // 10 + 1))
                    bins = np.linspace(all_values.min(), all_values.max(), n_bins)
                    if len(bins) > 1:
                        ref_hist, _ = np.histogram(ref_numeric, bins=bins)
                        alt_hist, _ = np.histogram(alt_numeric, bins=bins)
                        numeric_hellinger = hellinger_distance(ref_hist, alt_hist)
                except Exception:
                    pass
            
            # Combine all error metrics (like BAM's nanmean of RAE + Hellinger)
            all_errors = np.concatenate([
                count_rae_clipped,
                [elem_hellinger, attr_hellinger],
            ])
            if not np.isnan(numeric_hellinger):
                all_errors = np.append(all_errors, numeric_hellinger)
            
            # nanmean automatically ignores NaN values; returns NaN if all values are NaN
            mean_error = np.nanmean(all_errors)
            similarity = 1.0 - mean_error if not np.isnan(mean_error) else 0.0
            
            self.result.similarity = similarity
            self.result.details.update({
                "ref_stats": {k: v for k, v in ref_stats.items() if k not in ["numeric_values", "text_values"]},
                "alt_stats": {k: v for k, v in alt_stats.items() if k not in ["numeric_values", "text_values"]},
                "count_rae": {f: float(count_rae_clipped[i]) for i, f in enumerate(count_fields)},
                "element_hellinger_distance": float(elem_hellinger),
                "attribute_hellinger_distance": float(attr_hellinger),
                "numeric_hellinger_distance": float(numeric_hellinger) if not np.isnan(numeric_hellinger) else None,
            })
            
        except Exception as e:
            logger.error(f"Summary statistics comparison failed: {e}")
            self.result.error = f"Summary statistics comparison failed: {e}"
