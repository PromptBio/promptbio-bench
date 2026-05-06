"""File signature detection and utilities."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Dict, Set, Any
import gzip, bz2, lzma
import csv, json, zipfile
import xml.etree.ElementTree as ET

from pydantic import BaseModel, Field
from utils.agents import create_llm_agent, invoke_structured_agent

# Constants
MAX_BUF_SIZE = 262144  # 256KB - sufficient for long headers and enough data for format inference

# Type definitions for format, compression, method, and status
Compression = Literal["none", "gzip", "bgzf", "bzip2", "xz", "zip", "tar"]
Method = Literal["none", "content", "extension", "llm"]
Status = Literal["none", "success", "error"]
Format = Literal[
    # sequence files
    "fasta", "fastq", "fast5", "2bit", "fai", "gzi",
    # alignment files
    "sam", "bam", "bai", "csi", "cram", "crai", "paf", "gfa",
    # variant files
    "vcf", "bcf", "tbi",
    # annotation files
    "gff", "gtf",
    # region / interval files
    "bed", "bedgraph", "bigbed",
    # signal / track files
    "wig", "bigwig",
    # tabular files
    "csv", "tsv", "table", "gct", "xlsx", "json", "parquet",
    # proteomics / mass spectrometry files
    "featurexml", "mzml", "idxml",
    # generic XML
    "xml",
    # structure files
    "pdb",
    # matrix / array files
    "hdf5", "loom", "h5ad",
    # serialized R objects
    "rds", "rdata",
    # directory / folder
    "folder",
    # image files
    "png", "jpeg", "gif", "tiff", "bmp", "webp", "svg", "pdf",
    # archive / container files
    "tar", "sra", "zip",
    # script files
    "python", "r", "shell", "bash", "perl", "julia",
    # generic text
    "text",
    # method
    "unknown"
]

# File signature data model
@dataclass
class FileSignature:
    """File signature information including format, compression, and metadata."""
    format: Format
    compression: Compression
    is_text: bool
    method: Method = "none"
    status: Status = "none"
    extension: Optional[str] = None
    note: Optional[str] = None


# Format detector class
class FormatDetector:
    """Detects file format, compression, and text/binary nature from file signatures.
    
    Uses binary signatures, magic bytes, content heuristics, and optional extension or LLM-based
    method to identify bioinformatics file formats.
    """
    
    # Text-based formats that are human-readable
    TEXT_FORMATS: Set[Format] = {
        # sequence files
        "fasta", "fastq",
        # alignment files
        "sam", "paf", "gfa",
        # variant files
        "vcf",
        # annotation files
        "gff", "gtf",
        # region / interval files
        "bed", "bedgraph",
        # signal / track files
        "wig",
        # tabular files
        "csv", "tsv", "table", "gct", "json",
        # proteomics / mass spectrometry files
        "featurexml", "mzml", "idxml",
        # structure files
        "pdb",
        # XML-based files (text-based)
        "xml",
        # image files (text-based)
        "svg",
        # script files
        "python", "r", "shell", "bash", "perl", "julia",
        # generic text
        "text"
    }
    
    # Extension to format mapping
    EXTENSION_MAP: Dict[str, Format] = {
        # sequence files and indexes
        "fa": "fasta", "fasta": "fasta", "fna": "fasta",
        "fq": "fastq", "fastq": "fastq",
        "fast5": "fast5",
        "2bit": "2bit", 
        "fai": "fai", "gzi": "gzi",
        # alignment files and indexes
        "sam": "sam", "bam": "bam", "bai": "bai", "csi": "csi", "cram": "cram", "crai": "crai",
        "paf": "paf", "gfa": "gfa",
        # variant files and indexes
        "vcf": "vcf", "bcf": "bcf", "tbi": "tbi", 
        # annotation files
        "gff": "gff", "gff3": "gff", "gtf": "gtf",
        # region / interval files
        "bed": "bed", "narrowpeak": "bed", "broadpeak": "bed",
        "bedgraph": "bedgraph", "bg": "bedgraph", "bb": "bigbed",
        # signal / track files
        "wig": "wig", "bw": "bigwig", "bigwig": "bigwig",
        # tabular files
        "csv": "csv", "tsv": "tsv", "tab": "tsv", "gct": "gct", "xlsx": "xlsx", "xls": "xlsx",
        "json": "json", "parquet": "parquet",
        # proteomics / mass spectrometry files
        "featurexml": "featurexml", "mzml": "mzml", "idxml": "idxml",
        # generic XML
        "xml": "xml",
        # structure files
        "pdb": "pdb",
        # R serialization files
        "rds": "rds", "rdata": "rdata", "RData": "rdata", "Rda": "rdata",
        # matrix / array files
        "h5": "hdf5", "hdf5": "hdf5", "loom": "loom", "h5ad": "h5ad",
        # image files
        "png": "png", "jpg": "jpeg", "jpeg": "jpeg", "gif": "gif",
        "tif": "tiff", "tiff": "tiff", "bmp": "bmp", "webp": "webp",
        "svg": "svg", "pdf": "pdf",
        # archive / container files
        "sra": "sra", "tar": "tar", "zip": "zip",
        # script files
        "py": "python", "r": "r", "sh": "shell", "bash": "bash", "pl": "perl", "jl": "julia",
        # generic text
        "txt": "text",
    }
    
    def __init__(self, fallback: Method = "llm", model: str = "gpt-5.4", n_lines: int = 10, comment_char: str = "#"):
        """Initialize FormatDetector.
        
        Args:
            fallback: Fallback strategy when format detection fails:
                - "none": no fallback, return "unknown"
                - "extension": use extension-based detection
                - "llm": use LLM inference, then extension as final resort
            model: Model name for LLM inference (e.g. "gpt-5.4")
            n_lines: Number of lines to use for detection (default: 10).
            comment_char: Character used to identify comment lines (default: "#").
        """
        self.fallback = fallback
        self.model = model
        self.n_lines = n_lines
        self.comment_char = comment_char
        
    # ---------- main function ----------
    
    def detect(self, path: str, fallback: Method = "none", model: str = "gpt-5.4") -> FileSignature:
        """Detect file format, compression, and text/binary nature from file signature.
        
        Uses binary signatures, magic bytes, and content heuristics to detect format.
        For compressed files, decompresses and analyzes inner content.
        
        Args:
            path: Path to the file to analyze
            fallback: Fallback strategy when format detection fails:
                - "none": no fallback, return "unknown"
                - "extension": use extension-based detection
                - "llm": use LLM inference, then extension as final resort
                If "none", uses the instance's fallback value.
            model: Model for LLM inference. If None, uses instance's model value.
        
        Returns:
            FileSignature with format, compression, is_text flag, and optional note.
            For directories, format is "folder" with note indicating content type
            (e.g., "10x" for 10X Genomics format).
            Note field may indicate:
            - "file missing" if path doesn't exist
            - "10x" for directories containing 10X Genomics files
            - "zip container" for zip files
            - "tar archive" for tar files
            - "BGZF (blocked gzip; indexable)" for BGZF files
            - "format detected via extension" when extension method is used
            - "format inferred by LLM" when LLM inferred the format
            - "none" for normal detection
        """
        fallback = self.fallback if fallback == "none" else fallback
        model = model or self.model
        ext = get_file_extension(path) or None
        
        # Detect path existence and type
        p = Path(path)
        if not p.exists():
            return FileSignature(format="unknown", compression="none", is_text=False,
                                 extension=ext, method=fallback, status="error", note="file missing")
        
        # Handle directories
        if p.is_dir():
            note = "10x" if self._is_10x_folder(path) else None
            return FileSignature(format="folder", compression="none", is_text=False,
                                 extension=None, method="content", status="success", note=note)
        
        # Continue with file detection (attempt to open, will handle errors naturally)
        # Detect compression
        try:
            with open(path, "rb") as fh:
                comp = self._detect_compression(fh.read(512))
        except (OSError, PermissionError) as e:
            return FileSignature(format="unknown", compression="none", is_text=False,
                                 extension=ext, method=fallback, status="error", note=f"file read error: {e}")
        
        # Early exit: container formats
        if comp == "zip":
            # Check for xlsx (Office Open XML Spreadsheet) by peeking inside the ZIP
            try:
                with zipfile.ZipFile(path, "r") as zf:
                    if any(n.startswith("xl/") for n in zf.namelist()):
                        return FileSignature(format="xlsx", compression="none", is_text=False,
                                             extension=ext, method="content", status="success",
                                             note="Office Open XML Spreadsheet")
            except Exception:
                pass
            return FileSignature(format="unknown", compression="zip", is_text=False,
                                 extension=ext, method=fallback, status="success", note="zip container")
        if comp == "tar":
            return FileSignature(format="tar", compression="none", is_text=False,
                                 extension=ext, method=fallback, status="success", note="tar archive")
        
        # Detect format
        fmt, note = self._detect_inner_format(path, comp, fallback=fallback, model=model)
        
        # Build result - preserve detection note if no compression-specific note
        comp_note = {
            "bgzf": "BGZF (blocked gzip; indexable)",
            "gzip": f"tar archive (compressed by {comp})" if fmt == "tar" else None,
            "bzip2": f"tar archive (compressed by {comp})" if fmt == "tar" else None,
            "xz": f"tar archive (compressed by {comp})" if fmt == "tar" else None,
        }.get(comp)
        # Determine method: check if note starts with "llm:" for LLM inference
        if note.startswith("llm:"):
            used_method = "llm"
        elif note in ("content", "extension"):
            used_method = note
        else:
            used_method = "content"
        final_note = comp_note if used_method != "llm" else note
        
        return FileSignature(format=fmt, compression=comp, is_text=fmt in self.TEXT_FORMATS, 
                             extension=ext, method=used_method, status="success", note=final_note)
    
    # ---------- compression detection ----------
    
    def _detect_compression(self, head: bytes) -> Compression:
        """Detect compression type from file header bytes.
        
        Args:
            head: First 512 bytes of the file
            
        Returns:
            Compression type detected
        """
        if head.startswith(b"\x1f\x8b"):  # gzip/bgzf
            # BGZF = gzip with FEXTRA + 'BC' subfield
            try:
                if len(head) >= 12 and (head[3] & 0x04):
                    xlen = int.from_bytes(head[10:12], "little")
                    extra = head[12:12+xlen]
                    i = 0
                    while i + 4 <= len(extra):
                        si1, si2 = extra[i], extra[i+1]
                        slen = int.from_bytes(extra[i+2:i+4], "little")
                        if si1 == 0x42 and si2 == 0x43:  # 'B','C'
                            return "bgzf"
                        i += 4 + slen
            except Exception:
                pass
            return "gzip"
        if head.startswith(b"BZh"):
            return "bzip2"
        if head.startswith(b"\xfd7zXZ\x00"):
            return "xz"
        if head.startswith(b"PK\x03\x04"):
            return "zip"
        if len(head) >= 263 and head[257:262] == b"ustar":  # uncompressed tar
            return "tar"
        return "none"
    
    def _peek_decompressed(self, path: str, comp: Compression, nbytes: int = MAX_BUF_SIZE) -> bytes:
        """Read and decompress file content for format detection.
        
        Args:
            path: Path to the file
            comp: Compression type
            nbytes: Number of bytes to read (default: MAX_BUF_SIZE)
            
        Returns:
            Decompressed bytes (empty for container formats like zip/tar)
        """
        if comp == "none":
            with open(path, "rb") as f:
                return f.read(nbytes)
        if comp in ("gzip", "bgzf"):
            with gzip.open(path, "rb") as f:
                return f.read(nbytes)
        if comp == "bzip2":
            with bz2.open(path, "rb") as f:
                return f.read(nbytes)
        if comp == "xz":
            with lzma.open(path, "rb") as f:
                return f.read(nbytes)
        # zip/tar = containers; don't peek
        return b""
    
    def _extract_lines(self, buf: Optional[bytes] = None, file_path: Optional[str] = None) -> list[str]:
        """Extract lines from buffer or file, ensuring complete lines (not truncated).
        
        Args:
            buf: File content bytes (if provided, file_path is ignored)
            file_path: Path to the file (used if buf is None)
            
        Returns:
            List of complete lines (may include comments and empty lines)
        """
        lines = []
        try:
            if buf is not None:
                text = buf.decode("utf-8", errors="replace")
            elif file_path:
                # Detect compression and read using peek_decompressed
                with open(file_path, "rb") as fh:
                    comp = self._detect_compression(fh.read(512))
                buf = self._peek_decompressed(file_path, comp, MAX_BUF_SIZE)
                text = buf.decode("utf-8", errors="replace")
            else:
                return lines
            
            lines = text.splitlines()
            # If last line doesn't end with newline, it might be truncated - drop it if buffer was read
            if buf is not None and len(buf) > 0 and not text.endswith(('\n', '\r')):
                lines = lines[:-1] if lines else []
        except Exception:
            pass
        return lines
    
    # ---------- format detection ----------
    
    def _detect_inner_format(self, path: str, comp: Compression, fallback: Method = "none", model: str = None) -> tuple[Format, str]:
        """Detect file format from decompressed buffer content.
        
        Uses binary signatures, textual heuristics, tabular detection, and
        optional LLM inference when result would be "unknown".
        
        Args:
            path: Path to the file
            comp: Compression type
            fallback: Fallback strategy - "llm" enables LLM, otherwise LLM is disabled
            model: Model name for LLM (only used if fallback == "llm")
            
        Returns:
            (detected format, note). Note is "llm", "extension", or "content" indicating the detection method used.
        """
        # Use LLM only if fallback strategy is "llm"
        use_llm = (fallback == "llm")
        
        # Get file extension from path
        ext = get_file_extension(path) or None
        
        # Read and decompress file content
        buf = self._peek_decompressed(path, comp, MAX_BUF_SIZE)
        
        # Early fallback: if buffer is empty, use extension
        if not buf:
            return (self.EXTENSION_MAP.get(ext, "unknown"), "extension")
        
        # tar after decompression (ustar at 257)
        if len(buf) >= 263 and buf[257:262] == b"ustar":
            return ("tar", "content")
        
        head16 = buf[:16]
        head8 = buf[:8]
        strip = buf.lstrip()
        first = buf.splitlines()[0] if buf else b""
        
        # Binary signatures (most specific first)
        if head16.startswith(b"BAM\x01"):
            return ("bam", "content")
        if head16.startswith(b"CRAM"):
            return ("cram", "content")
        if head16.startswith(b"BCF"):
            return ("bcf", "content")
        if head16.startswith(b"%PDF-"):
            return ("pdf", "content")
        if head8.startswith(b"\x89PNG\r\n\x1a\n"):
            return ("png", "content")
        if buf[:3] == b"\xFF\xD8\xFF":
            return ("jpeg", "content")
        if buf[:6] in (b"GIF87a", b"GIF89a"):
            return ("gif", "content")
        if buf[:4] in (b"II*\x00", b"MM\x00*"):
            return ("tiff", "content")
        if buf[:2] == b"BM":
            return ("bmp", "content")
        if buf[:4] == b"RIFF" and buf[8:12] == b"WEBP":
            return ("webp", "content")
        if head16.startswith(b"\x89HDF"):
            return ("hdf5", "content")  # h5/h5ad/loom/fast5
        if buf[:4] == b"PAR1":
            return ("parquet", "content")
        
        # UCSC big* formats
        le = int.from_bytes(buf[:4], "little", signed=False)
        be = int.from_bytes(buf[:4], "big", signed=False)
        if le in (0x888FFC26, 0x26FC8F88) or be in (0x888FFC26, 0x26FC8F88):
            return ("bigwig", "content")
        if le in (0x8789F2EB, 0xEBF28987) or be in (0x8789F2EB, 0xEBF28987):
            return ("bigbed", "content")
        if le == 0x1A412743 or be == 0x1A412743:
            return ("2bit", "content")
        
        # Index files
        if head16[:4] == b"BAI\x01":
            return ("bai", "content")
        if head16[:4] == b"CSI\x01":
            return ("csi", "content")
        if head16[:4] == b"TBI\x01":
            return ("tbi", "content")
        if head16[:4] == b"CRAI":
            return ("crai", "content")
        
        # R serialization formats (RDS and RData)
        # XDR format: starts with 0x58 0x0A 0x00 0x00 (X\n\0\0)
        if len(buf) >= 3 and buf[0] == 0x58 and (buf[1] == 0x0A or buf[1] == 0x0D) and buf[2] == 0x00:
            return ("rds", "content")
        if len(buf) >= 3 and buf[:3] == b"RDX":
            return ("rdata", "content")
        
        # Textual heuristics
        if strip.startswith(b"##fileformat=VCF") or strip.startswith(b"#CHROM"):
            return ("vcf", "content")
        if first.startswith((b"@HD", b"@SQ", b"@PG", b"@RG")):
            return ("sam", "content")
        if strip.startswith(b">") and self._is_fasta(buf):
            return ("fasta", "content")
        if strip.startswith(b"@") and b"\n+\n" in buf[:4096].replace(b"\r\n", b"\n"):
            return ("fastq", "content")
        if b"fixedStep" in buf or b"variableStep" in buf:
            return ("wig", "content")
        if b"type=bedGraph" in buf or ext in ("bedgraph", "bg"):
            return ("bedgraph", "content")
        if self._is_gff_gtf(buf):
            return ("gtf" if ext == "gtf" else "gff", "content")
        if self._is_pdb(buf):
            return ("pdb", "content")
        if strip.startswith((b"{", b"[")) and self._is_json(path, comp):
            return ("json", "content")
        xml_fmt = self._detect_xml(strip, path, comp)
        if xml_fmt is not None:
            return (xml_fmt, "content")
        if ext == "gfa" or first.startswith(b"H\t"):
            return ("gfa", "content")
        
        # Extract lines from buffer for tabular detection and LLM inference
        all_lines = self._extract_lines(buf=buf)
        # Filter out comment lines to get data lines (for tabular detection only)
        data_lines = [line for line in all_lines 
                     if line.strip() and not line.strip().startswith(self.comment_char)][:self.n_lines]
        
        # Check for GCT format before generic tabular detection
        # (GCT files are tab-separated, so they might match TSV otherwise)
        if self._is_gct(all_lines):
            return ("gct", "content")
        
        # Check for BED format before generic tabular detection
        # (BED files are tab-separated, so they would match TSV otherwise)
        if self._is_bed(buf):
            return ("bed", "content")
        
        # Tabular content-based detection
        if len(data_lines) >= 2:
            delim = self._detect_delimiter(data_lines=data_lines)
            if delim == "\t":
                return ("tsv", "content")
            if delim == ",":
                return ("csv", "content")
            if delim in (";", "|", " "):
                return ("table", "content")
            # Single-column tabular file: no delimiter found but extension says csv/tsv
            if ext in ("csv",):
                return ("csv", "extension")
            if ext in ("tsv",):
                return ("tsv", "extension")

        # Try LLM inference, then fall back to extension-based detection
        # Use all_lines (not filtered data_lines) for LLM to preserve context for plain text files
        if use_llm and all_lines:
            try:
                # Take first n_lines from all_lines (preserving empty lines and comments for context)
                llm_sample_lines = all_lines[:self.n_lines]
                sample = "\n".join(llm_sample_lines)
                if sample.strip():
                    llm_result = self._infer_format_with_llm(sample, model=model)
                    llm_fmt = llm_result.get("format", "unknown")
                    if llm_fmt and llm_fmt != "unknown":
                        note = f"llm: confidence: {llm_result.get('confidence', 0.0)}; reasoning: {llm_result.get('reasoning', 'inference failed')}"
                        return (llm_fmt, note)
            except Exception:
                pass
        
        # Extension-based fallback
        return (self.EXTENSION_MAP.get(ext, "unknown"), "extension")
    
    # ---------- format-specific probes ----------
    
    def _detect_xml(self, strip: bytes, path: str, comp: Compression) -> Optional[Format]:
        """Detect XML-based formats from content.
        
        Handles both ``<?xml ...>`` declarations and bare root elements
        (e.g. ``<phyloxml ...>``).  Returns a specific format when the root
        element matches a known type, generic ``"xml"`` for any other valid
        XML, or ``None`` if the content is not XML.
        
        Args:
            strip: Leading bytes of the file (whitespace-stripped)
            path: Path to the file
            comp: Compression type
            
        Returns:
            Detected Format string, or None if not XML
        """
        _lower256 = strip[:256].lower()
        _maybe_xml = _lower256.startswith(b"<?xml") or (
            _lower256.startswith(b"<") and not _lower256.startswith((b"<!doctype", b"<html"))
        )
        if not _maybe_xml:
            return None
        
        root_elem = self._get_xml_root_element(path, comp)
        if root_elem is None:
            return None
        
        # Map known root elements to specific formats
        _XML_ROOT_MAP: Dict[str, Format] = {
            "svg": "svg",
            "featuremap": "featurexml",
            "mzml": "mzml",
            "indexedmzml": "mzml",
            "idxml": "idxml",
        }
        return _XML_ROOT_MAP.get(root_elem, "xml")
    
    def _get_xml_root_element(self, path: str, comp: Compression) -> Optional[str]:
        """Get the root element name from an XML file.
        
        Args:
            path: Path to the file
            comp: Compression type
            
        Returns:
            Root element name (without namespace) if parsing succeeds, None otherwise
        """
        openers = {"gzip": gzip.open, "bgzf": gzip.open, "bzip2": bz2.open, "xz": lzma.open}
        opener = openers.get(comp, open)
        try:
            with opener(path, "rt", encoding="utf-8", errors="replace") as f:
                # Use iterparse to avoid loading entire file into memory
                # Stop after getting the root element
                context = ET.iterparse(f, events=("start",))
                event, root = next(context, (None, None))
                if root is not None:
                    # Remove namespace prefix if present (e.g., "{http://...}featureMap" -> "featureMap")
                    tag = root.tag
                    if "}" in tag:
                        tag = tag.split("}")[-1]
                    return tag.lower()
        except Exception:
            pass
        return None
        
    def _is_10x_folder(self, path: str) -> bool:
        """Check if path points to a 10X Genomics format directory.
        Looks for barcodes.tsv.gz, features.tsv.gz, and matrix.mtx.gz files.
        
        Args:
            path: Path to directory
        
        Returns:
            True if 10X format detected, False otherwise
        """
        p = Path(path)
        if not p.is_dir():
            return False
        
        required_files = ['barcodes.tsv', 'features.tsv', 'matrix.mtx']
        # Check that each required file exists (compressed or uncompressed)
        for filename in required_files:
            uncompressed = p / filename
            compressed = p / (filename + '.gz')
            if not (uncompressed.exists() or compressed.exists()):
                return False
        
        return True
    
    def _is_fasta(self, buf: bytes) -> bool:
        """Check whether buffer content looks like FASTA format.
        
        Requires a '>' header line followed by sequence lines that contain only
        valid characters (letters, '-', '.', '*'). Rejects files where non-header
        lines contain digits, parentheses, or other non-sequence characters
        (e.g. RNAfold dot-bracket notation with energy values).
        
        Args:
            buf: Raw (decompressed) file content
            
        Returns:
            True if the content matches FASTA structure
        """
        # Valid FASTA sequence characters: letters (A-Za-z), gap (-, .), stop (*),
        # whitespace is stripped before checking
        _VALID_SEQ_CHARS = set(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-.*")
        
        found_header = False
        found_seq = False
        for line in buf.split(b"\n", 20)[:20]:
            line = line.strip()
            if not line:
                continue
            if line.startswith(b">"):
                found_header = True
                continue
            if not found_header:
                return False  # non-header content before first '>'
            # Validate sequence line: all characters must be valid
            if not all(c in _VALID_SEQ_CHARS for c in line):
                return False
            found_seq = True
        
        return found_header and found_seq
    
    def _is_gff_gtf(self, buf: bytes) -> bool:
        """Check whether buffer content looks like GFF/GTF format.
        
        Checks for ``##gff-version`` header or verifies that the first data line
        has exactly 9 tab-separated columns with a known feature type in column 3.
        
        Args:
            buf: Raw (decompressed) file content
            
        Returns:
            True if the content matches GFF/GTF structure
        """
        # Known GFF/GTF feature types (column 3)
        _GFF_FEATURES = {
            b"gene", b"exon", b"mRNA", b"CDS", b"transcript",
            b"five_prime_UTR", b"three_prime_UTR", b"start_codon", b"stop_codon",
            b"region", b"biological_region", b"chromosome", b"supercontig",
            b"ncRNA_gene", b"lnc_RNA", b"miRNA", b"rRNA", b"tRNA", b"snRNA", b"snoRNA",
        }
        if buf.lstrip().startswith(b"##gff-version"):
            return True
        for line in buf.split(b"\n", 20)[:20]:
            line = line.strip()
            if not line or line.startswith(b"#"):
                continue
            fields = line.split(b"\t")
            return len(fields) == 9 and fields[2] in _GFF_FEATURES
        return False
    
    def _is_pdb(self, buf: bytes) -> bool:
        """Check whether buffer content looks like PDB format.
        
        PDB files contain record types like HEADER, TITLE, MODEL, ATOM, HETATM, etc.
        at the beginning of lines. Checks if any of the first non-empty lines start
        with a PDB record type.
        
        Args:
            buf: Raw (decompressed) file content
            
        Returns:
            True if the content matches PDB structure
        """
        # PDB record types (case-insensitive, but typically uppercase)
        _PDB_RECORDS = {
            b"HEADER", b"TITLE", b"MODEL", b"ATOM", b"HETATM",
            b"ENDMDL", b"END", b"CRYST1", b"ORIGX1", b"ORIGX2", b"ORIGX3",
            b"SCALE1", b"SCALE2", b"SCALE3", b"MTRIX1", b"MTRIX2", b"MTRIX3",
            b"SEQRES", b"REMARK", b"CONECT", b"MASTER",
        }
        for line in buf.split(b"\n", 20)[:20]:
            line = line.strip()
            if not line:
                continue
            # Check if line starts with any PDB record type (case-insensitive)
            line_upper = line[:6].upper()  # PDB records are typically 6 chars
            if line_upper in _PDB_RECORDS:
                return True
        return False
    
    def _is_gct(self, all_lines: list) -> bool:
        """Check whether lines look like GCT format.

        Uses composite check: #1.x version header, two-integer dimensions line,
        and name+description (case-insensitive) as the first two column headers.
        """
        if len(all_lines) < 3:
            return False
        if not all_lines[0].strip().startswith("#1."):
            return False
        parts = all_lines[1].split()
        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            return False
        first_two = [c.strip().lower() for c in all_lines[2].split("\t")[:2]]
        return first_two == ["name", "description"]

    def _is_bed(self, buf: bytes) -> bool:
        """Check whether buffer content looks like BED format.
        
        BED files are tab-separated with progressively validated columns:
        - BED3:  chrom (genomic ID), chromStart (int ≥ 0), chromEnd (int > start)
        - BED4+: name (string identifier, not a decimal/scientific number)
        - BED5+: score (integer 0–1000 or ".")
        - BED6+: strand (+, -, or .)
        
        Variants include BED3–BED12, narrowPeak (BED6+4), broadPeak (BED6+3).
        
        Detection requires:
        - No header row (BED spec has no named column headers; only track/browser
          meta-lines are allowed before data)
        - At least 2 consistent data lines passing all applicable column checks
        - Layered column validation (cols 4–6) rejects non-BED TSVs
        
        Args:
            buf: Raw (decompressed) file content
            
        Returns:
            True if the content matches BED structure
        """
        def _is_bed_line(fields: list[bytes]) -> bool:
            """Validate a single tab-split line against BED column conventions."""
            n = len(fields)
            if n < 3:
                return False
            
            # --- Columns 1-3: chrom, chromStart, chromEnd ---
            chrom = fields[0].strip()
            if not chrom:
                return False
            try:
                start = int(fields[1])
                end = int(fields[2])
            except (ValueError, IndexError):
                return False
            if not (end > start >= 0):
                return False
            
            # --- Column 4: name (if present) ---
            # Relaxed: BED names can be any string (identifiers, numbers, ".").
            # No validation needed; columns 5-6 are more distinctive.
            
            # --- Column 5: score (if present) ---
            # Must be an integer in [0, 1000] or "." as placeholder.
            if n >= 5:
                score = fields[4].strip()
                if score != b".":
                    try:
                        s = int(score)
                        if not (0 <= s <= 1000):
                            return False
                    except ValueError:
                        return False
            
            # --- Column 6: strand (if present) ---
            # Must be "+", "-", or "."
            if n >= 6:
                strand = fields[5].strip()
                if strand not in (b"+", b"-", b"."):
                    return False
            
            return True
        
        bed_lines = 0
        for line in buf.split(b"\n", 20)[:20]:
            line = line.strip()
            if not line or line.startswith(b"#") or line.startswith(b"track") or line.startswith(b"browser"):
                continue
            fields = line.split(b"\t")
            if _is_bed_line(fields):
                bed_lines += 1
            elif bed_lines == 0:
                return False  # first data line must pass
            else:
                break  # inconsistent line after valid ones
        
        # Require at least 2 consistent BED-format lines for confidence
        return bed_lines >= 2
    
    def _is_json(self, path: str, comp: Compression) -> bool:
        """Check whether file content is valid JSON by attempting to parse it.
        
        Caller should gate on a quick prefix check (starts with { or [)
        before calling this method.
        
        Args:
            path: Path to the file
            comp: Compression type
            
        Returns:
            True if the file content is valid JSON
        """
        openers = {"gzip": gzip.open, "bgzf": gzip.open, "bzip2": bz2.open, "xz": lzma.open}
        opener = openers.get(comp, open)
        try:
            with opener(path, "rt", encoding="utf-8", errors="replace") as f:
                json.load(f)
            return True
        except Exception:
            return False
    
    # ---------- delimiter and LLM inference methods ----------
    
    def _detect_delimiter(self, file_path: Optional[str] = None, data_lines: Optional[list] = None) -> str:
        """Detect delimiter from CSV/TSV content by trying common delimiters, in order of 
        reliability (tab, comma, semicolon, pipe, space) and validating by checking field count consistency.
        
        Args:
            file_path: Path to CSV/TSV file (reads lines from file)
            data_lines: Pre-loaded lines to analyze (alternative to file_path);
        
        Returns:
            Detected delimiter string (e.g., ',', '\t'), or 'unknown' if not detected
        """
        try:
            if data_lines is None and file_path:
                all_lines = self._extract_lines(file_path=file_path)
                # Filter out comment lines to get data lines
                data_lines = [line for line in all_lines 
                             if line.strip() and not line.strip().startswith(self.comment_char)][:self.n_lines]
            if not data_lines or len(data_lines) < 2:
                return "unknown"
            for delim in ("\t", ",", ";", "|", " "):
                counts = [len(next(csv.reader([line], delimiter=delim)))
                          for line in data_lines if line.strip()]
                if not (len(counts) >= 2 and len(set(counts)) == 1 and counts[0] > 1):
                    continue
                # Tab is reliable even with few lines (tabs rarely appear in prose).
                # For other delimiters: require 3+ fields AND 5+ data lines,
                # since commas/spaces/etc. commonly appear in natural text and
                # can produce consistent field counts by coincidence with few lines.
                # Ambiguous cases fall through to LLM inference for classification.
                if delim == "\t":
                    return delim
                if counts[1] >= 3 and len(counts) >= 5:
                    return delim
        except Exception:
            pass
        return "unknown"
    
    def _infer_format_with_llm(self, text_sample: str, model: str = None) -> Dict[str, Any]:
        """Use an LLM to infer the format of ambiguous text content.
        
        The LLM can identify any bioinformatics file format, including formats that cannot
        be detected by content-based heuristics (e.g., script files like python/r/shell,
        or formats like bed, paf, fast5 that lack distinctive content signatures).
        
        Args:
            text_sample: Decoded file head (e.g. first few lines of data)
            model: Model name for the LLM. If None, uses instance's model.
        
        Returns:
            Dictionary with format, reasoning, and confidence fields,
            or None on failure
        """
        # Use instance's model as fallback if not provided
        model = model or self.model

        result = {
            "format": "unknown",
            "reasoning": "LLM inference failed or did not execute",
            "confidence": 0.0
        }

        try:
            class InferredFormat(BaseModel):
                file_format: Format = Field(
                    default="unknown",
                    description="The detected file format. Can be any bioinformatics file format such as fasta, fastq, sam, vcf, gff, gtf, bed, csv, tsv, table, gct, json, xml, featurexml, mzml, idxml, pdb, hdf5, loom, h5ad, rds, rdata, folder, python, r, shell, text, or unknown if the format cannot be determined."
                )
                reasoning: str = Field(
                    default="",
                    description="Brief explanation of why this format was identified, based on the content structure, headers, or patterns observed."
                )
                confidence: float = Field(
                    default=0.0,
                    description="Confidence score between 0.0 and 1.0 indicating how certain the format identification is.",
                    ge=0.0,
                    le=1.0
                )

            system = (
                "You are a file format detection expert. Analyze the provided file content sample and identify its format.\n\n"
                "## DETECTION RULES\n"
                "1. **Analyze content structure, ignore file extensions**\n"
                "2. **Default to 'text'**: If content is human-readable text (letters, numbers, punctuation) and doesn't match any specific format below, return 'text'. "
                "This includes: logs, reports, statistical output, command output, mathematical formulas, unstructured/semi-structured text.\n"
                "3. **Use 'unknown' only for**: Binary/non-textual content, encrypted data, or truly undeterminable content. "
                "If you can read and understand the text, it is 'text', NOT 'unknown'.\n\n"
                "## FORMAT CATEGORIES\n\n"
                "### Tabular/Structured Formats\n"
                "- **csv**: Comma-delimited tabular data (consistent columns)\n"
                "- **tsv**: Tab-delimited tabular data (consistent columns)\n"
                "- **table**: Tabular data with other delimiters (;, |, space) - use when delimiter is NOT comma or tab\n"
                "- **gct**: Gene Cluster Text format (first line is '#1.2', second line has row/col counts, then tab-separated matrix with NAME and Description columns)\n"
                "- **json**: Valid JSON syntax (starts with { or [)\n"
                "- **parquet**: Binary columnar storage (Apache Parquet)\n\n"
                "### Script Formats\n"
                "- **python**, **r**, **shell**, **bash**, **perl**, **julia**: Code files with programming language syntax\n\n"
                "### Sequence Formats\n"
                "- **fasta**: Header lines start with '>'; ALL non-header lines must be pure sequence "
                "(letters, hyphens '-', periods '.', or asterisks '*' — e.g. ACGTUN for nucleotides, "
                "amino acid codes, gap characters, stop codons). "
                "If non-header lines contain parentheses, digits, brackets, or energy values "
                "(e.g. dot-bracket notation '(((...)))' from RNAfold/Vienna format), "
                "it is NOT fasta — classify as 'text' instead.\n"
                "- **fastq**: '@' header, sequence, '+' separator, quality scores\n"
                "- **fast5**: Nanopore sequencing (HDF5-based)\n"
                "- **2bit**: UCSC 2bit binary sequence\n"
                "- **fai**: FASTA index (tab-separated: name, length, offset)\n"
                "- **gzi**: Gzip index\n\n"
                "### Alignment Formats\n"
                "- **sam**: Text format with '@' header lines (@HD, @SQ, @PG, @RG)\n"
                "- **bam**: Binary SAM (starts with 'BAM\\x01')\n"
                "- **cram**: Compressed alignment (starts with 'CRAM')\n"
                "- **paf**: Pairwise alignment (tab-separated)\n"
                "- **gfa**: Graphical Fragment Assembly (starts with 'H\\t')\n"
                "- **bai**, **csi**, **crai**: Alignment index files (start with 'BAI\\x01', 'CSI\\x01', 'CRAI')\n\n"
                "### Variant Formats\n"
                "- **vcf**: Text format with '##' or '#CHROM' headers\n"
                "- **bcf**: Binary VCF (starts with 'BCF')\n"
                "- **tbi**: Tabix index (starts with 'TBI\\x01')\n\n"
                "### Annotation Formats\n"
                "- **gff**: Tab-separated, 9 columns, feature column contains 'gene', 'exon', 'mRNA'\n"
                "- **gtf**: Similar to GFF with GTF-specific conventions\n\n"
                "### Region/Interval Formats\n"
                "- **bed**: Tab-separated (chromosome, start, end; typically 3-12 columns)\n"
                "- **bedgraph**: Contains 'type=bedGraph' marker\n"
                "- **bigbed**: Binary BigBed (UCSC)\n\n"
                "### Signal/Track Formats\n"
                "- **wig**: Contains 'fixedStep' or 'variableStep' keywords\n"
                "- **bigwig**: Binary BigWig (UCSC)\n\n"
                "### Matrix/Array Formats\n"
                "- **hdf5**: HDF5 format (starts with '\\x89HDF')\n"
                "- **loom**: Loom format (HDF5-based single-cell)\n"
                "- **h5ad**: AnnData format (HDF5-based, Python single-cell)\n\n"
                "### Proteomics/Mass Spectrometry Formats\n"
                "- **featurexml**: OpenMS featureXML (XML root: '<featureMap>')\n"
                "- **mzml**: mzML format (XML root: '<mzML>')\n"
                "- **idxml**: OpenMS idXML format for peptide/protein identification results (XML root: '<IdXML>', "
                "contains <SearchParameters>, <ProteinIdentification>, <PeptideIdentification>, <PeptideHit>)\n"
                "- **xml**: Generic XML (starts with '<?xml', not matching specific formats above)\n\n"
                "### Structure Formats\n"
                "- **pdb**: Protein Data Bank (record types: HEADER, TITLE, MODEL, ATOM, HETATM, ENDMDL, END)\n\n"
                "### Image Formats\n"
                "- **png**: Starts with '\\x89PNG\\r\\n\\x1a\\n'\n"
                "- **jpeg**: Starts with '\\xFF\\xD8\\xFF'\n"
                "- **gif**: Starts with 'GIF87a' or 'GIF89a'\n"
                "- **tiff**: Starts with 'II*\\x00' or 'MM\\x00*'\n"
                "- **bmp**: Starts with 'BM'\n"
                "- **webp**: Starts with 'RIFF....WEBP'\n"
                "- **svg**: XML-based (root: '<svg>')\n"
                "- **pdf**: Starts with '%PDF-'\n\n"
                "### Archive/Container Formats\n"
                "- **tar**: Tape Archive (starts with 'ustar' at offset 257)\n"
                "- **zip**: ZIP archive (starts with 'PK\\x03\\x04')\n"
                "- **sra**: Sequence Read Archive\n\n"
                "## OUTPUT\n"
                "Return:\n"
                "- **file_format**: Format name (must match one listed above)\n"
                "- **reasoning**: Brief explanation based on content structure\n"
                "- **confidence**: Score 0.0-1.0 (1.0 = very certain)\n"
            )
            agent = create_llm_agent(
                model=model,
                system_prompt=system,
                response_format=InferredFormat,
                temperature=0.0,
            )
            out = invoke_structured_agent(agent, text_sample, InferredFormat)
            result["format"] = out.file_format
            result["reasoning"] = out.reasoning
            result["confidence"] = out.confidence
        except ValueError as e:
            # Validation error - LLM response didn't match expected format
            result["reasoning"] = f"LLM response validation failed: {str(e)}"
        except Exception as e:
            # Other errors (API, network, model not found, etc.)
            result["reasoning"] = f"LLM inference failed: {type(e).__name__}: {str(e)}"
        return result

# ---------- module-level convenience functions ----------

def get_file_extension(file_path: str, strip_compression: bool = True) -> str:
    """Get lowercase file extension from file_path.
    
    Args:
        file_path: Path to the file
        strip_compression: If True, strip compression extensions (.gz, .xz, .bz2) and
                          return the base extension. For 'file.vcf.gz', returns 'vcf' instead of 'gz'.
        
    Returns:
        Lowercase file extension without the dot (e.g., 'png', 'csv'), or empty string if no extension.
        When strip_compression=True, returns the base extension (e.g., 'vcf' for 'file.vcf.gz').
    """
    p = Path(file_path)
    if strip_compression:
        # Handle common compression extensions
        if file_path.endswith((".gz", ".xz")):
            base_path = file_path[:-3]
            ext = Path(base_path).suffix.lower()
        elif file_path.endswith(".bz2"):
            base_path = file_path[:-4]
            ext = Path(base_path).suffix.lower()
        else:
            ext = p.suffix.lower()
    else:
        ext = p.suffix.lower()
    return ext.lstrip(".") if ext else ""


def detect_format_from_extension(file_path: str) -> Format:
    """Detect file format from file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Format name or 'unknown' if not found
    """
    ext = get_file_extension(file_path)
    return FormatDetector.EXTENSION_MAP.get(ext, "unknown")


def detect_file_signature(path: str, fallback: Method = "llm", model: str = "gpt-5.4") -> FileSignature:
    """Convenience function for file format detection.
    
    Creates a FormatDetector instance and detects file format, compression, and text/binary nature.
    For custom configuration, use FormatDetector directly.
    
    Args:
        path: Path to a file or directory to analyze
        fallback: Fallback strategy when format detection fails:
            - "none": no fallback, return "unknown"
            - "extension": use extension-based detection
            - "llm": use LLM inference, then extension as final resort
        model: Model for LLM inference (e.g. "gpt-5.4")
    
    Returns:
        FileSignature with format, compression, is_text, extension, and optional note
    """
    detector = FormatDetector(fallback=fallback, model=model)
    return detector.detect(path)
