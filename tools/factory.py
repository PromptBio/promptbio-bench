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
from tools.pdb import PDBHandler
from tools.txt import TXTHandler


class FormatHandlerFactory:
    """Factory for creating format handlers."""

    _handlers = {
        'fasta': FASTAHandler,
        'fa': FASTAHandler,
        'fai': FAIHandler,
        'fastq': FASTQHandler,
        'fq': FASTQHandler,
        'bam': BAMHandler,
        'sam': BAMHandler,
        'cram': BAMHandler,
        'bai': BAIHandler,
        'crai': BAIHandler,
        'csv': TableHandler,
        'tsv': TableHandler,
        'table': TableHandler,
        'gct': TableHandler,
        'vcf': VCFHandler,
        'bcf': VCFHandler,
        'tbi': TabixHandler,
        'bed': BEDHandler,
        'bigbed': BEDHandler,
        'bedgraph': BEDHandler,
        'bg': BEDHandler,
        'bw': BigWigHandler,
        'bigwig': BigWigHandler,
        'wig': BigWigHandler,
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
        'pdb': PDBHandler,
        'cif': PDBHandler,
        'mmcif': PDBHandler,
        'txt': TXTHandler,
        'text': TXTHandler,
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
        if file_path.endswith('.vcf.gz'):
            return cls.get_handler('vcf')
        elif file_path.endswith('.gz'):
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
