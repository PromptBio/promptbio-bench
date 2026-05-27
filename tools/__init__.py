"""Tools module for file validation and comparison.

This module provides:
- Format handlers: Format-specific validation and comparison
- Models: Shared data models for results
"""

from tools.results import ValidationResult, ComparisonResult, EquivalenceResult

from tools.base import FormatHandler
from tools.factory import FormatHandlerFactory
from tools.fasta import FASTAHandler
from tools.fastq import FASTQHandler
from tools.bam import BAMHandler
from tools.table import TableHandler
from tools.vcf import VCFHandler
from tools.bed import BEDHandler
from tools.bigwig import BigWigHandler
from tools.image import ImageHandler
from tools.fai import FAIHandler
from tools.bai import BAIHandler
from tools.tbi import TabixHandler
from tools.pdb import PDBHandler
from tools.txt import TXTHandler

from tools.api import (
    validate_file,
    compare_files,
    get_supported_formats,
    get_supported_strategies,
)

__all__ = [
    'ValidationResult',
    'ComparisonResult',
    'EquivalenceResult',
    'FormatHandler',
    'FormatHandlerFactory',
    'FASTAHandler',
    'FASTQHandler',
    'BAMHandler',
    'TableHandler',
    'VCFHandler',
    'BEDHandler',
    'BigWigHandler',
    'ImageHandler',
    'FAIHandler',
    'BAIHandler',
    'TabixHandler',
    'PDBHandler',
    'TXTHandler',
    'validate_file',
    'compare_files',
    'get_supported_formats',
    'get_supported_strategies',
]
