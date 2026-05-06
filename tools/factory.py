"""Factory for creating format handlers."""

from typing import Optional
from pathlib import Path

from tools.base import FormatHandler
from tools.fasta import FASTAHandler
from tools.fai import FAIHandler
from tools.fastq import FASTQHandler
from tools.bam import BAMHandler
from tools.bai import BAIHandler
from tools.tbi import TabixHandler
from tools.table import TableHandler
from tools.vcf import VCFHandler
from tools.bed import BEDHandler
from tools.bigwig import BigWigHandler
from tools.image import ImageHandler
from tools.script import ScriptHandler
from tools.folder import FolderHandler
from tools.pdb import PDBHandler
from tools.rdata import RDataHandler
from tools.anndata import AnnDataHandler
from tools.txt import TXTHandler
from tools.msxml import MSXMLHandler


class FormatHandlerFactory:
    """Factory for creating format handlers."""
    
    _handlers = {
        'fasta': FASTAHandler,
        'fa': FASTAHandler,
        'fai': FAIHandler,  # FASTA index files use FAIHandler
        'fastq': FASTQHandler,
        'fq': FASTQHandler,
        'bam': BAMHandler,
        'sam': BAMHandler,
        'cram': BAMHandler,  # CRAM files use BAM handler
        'bai': BAIHandler,  # BAM index files use BAIHandler
        'crai': BAIHandler,  # CRAM index files use BAIHandler
        'csv': TableHandler,
        'tsv': TableHandler,
        'table': TableHandler,
        'gct': TableHandler,
        'vcf': VCFHandler,
        'bcf': VCFHandler,  # BCF files use VCF handler
        'tbi': TabixHandler,  # Tabix index files use TabixHandler
        'bed': BEDHandler,
        'bigbed': BEDHandler,  # bigBed files use BED handler
        'bedgraph': BEDHandler,  # bedGraph files use BED handler
        'bg': BEDHandler,  # bedGraph files use BED handler
        'bw': BigWigHandler,  # bigWig files use BigWig handler
        'bigwig': BigWigHandler,  # bigWig files use BigWig handler
        'wig': BigWigHandler,  # WIG files use BigWig handler
        'png': ImageHandler,
        'jpg': ImageHandler,
        'jpeg': ImageHandler,
        'pdf': ImageHandler,
        'svg': ImageHandler,
        'gif': ImageHandler,
        'bmp': ImageHandler,
        'tiff': ImageHandler,
        'tif': ImageHandler,
        'webp': ImageHandler,
        'py': ScriptHandler,
        'r': ScriptHandler,
        'sh': ScriptHandler,
        'bash': ScriptHandler,
        'pl': ScriptHandler,
        'jl': ScriptHandler,
        'script': ScriptHandler,
        'folder': FolderHandler,
        'directory': FolderHandler,  # Alias for folder
        'dir': FolderHandler,  # Alias for folder
        'pdb': PDBHandler,
        'cif': PDBHandler,  # mmCIF files use PDB handler
        'mmcif': PDBHandler,  # mmCIF files use PDB handler
        'rds': RDataHandler,  # RDS files (single R object format)
        'rdata': RDataHandler,  # Rdata files (multiple R objects format)
        'rda': RDataHandler,  # Rdata files (alternative extension)
        'h5ad': AnnDataHandler,  # H5AD files (AnnData format)
        'h5': AnnDataHandler,  # H5 files (may contain AnnData format)
        'loom': AnnDataHandler,  # Loom files (AnnData format)
        'txt': TXTHandler,  # Text files
        'text': TXTHandler,  # Text files (alias)
        'featurexml': MSXMLHandler,  # OpenMS featureXML format
        'mzml': MSXMLHandler,  # Mass spectrometry mzML format
        'idxml': MSXMLHandler,  # OpenMS idXML format (peptide/protein identification)
    }
    
    @classmethod
    def get_handler(cls, format_name: str) -> Optional[FormatHandler]:
        """Get handler for format.
        
        Args:
            format_name: Format name (e.g., 'fasta', 'bam', 'csv')
            
        Returns:
            FormatHandler instance or None if format not supported
        """
        format_name = format_name.lower()
        handler_class = cls._handlers.get(format_name)
        
        if handler_class:
            return handler_class()
        
        return None
    
    @classmethod
    def get_handler_from_path(cls, file_path: str) -> Optional[FormatHandler]:
        """Get handler from file path by extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            FormatHandler instance or None if format not supported
        """
        # Handle compressed files (e.g., .vcf.gz, .bcf)
        if file_path.endswith('.vcf.gz'):
            # Special case for .vcf.gz
            return cls.get_handler('vcf')
        elif file_path.endswith('.gz'):
            # Remove .gz and get the actual extension
            base_path = file_path[:-3]
            ext = Path(base_path).suffix.lstrip('.').lower()
        else:
            ext = Path(file_path).suffix.lstrip('.').lower()
        
        return cls.get_handler(ext)
    
    @classmethod
    def register_handler(cls, format_name: str, handler_class: type):
        """Register a new format handler.
        
        Args:
            format_name: Format name
            handler_class: FormatHandler subclass
        """
        cls._handlers[format_name.lower()] = handler_class
