"""Tools module for file validation and comparison.

This module provides:
- Format handlers: Format-specific validation and comparison
- Script tools: Script analysis and comparison
- Models: Shared data models for results
"""

# Models (shared data structures)
from tools.results import ValidationResult, ComparisonResult, EquivalenceResult

# Format handlers (primary API)
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
from tools.script import ScriptHandler
from tools.folder import FolderHandler
from tools.fai import FAIHandler
from tools.bai import BAIHandler
from tools.tbi import TabixHandler
from tools.pdb import PDBHandler
from tools.rdata import RDataHandler
from tools.anndata import AnnDataHandler
from tools.txt import TXTHandler
from tools.msxml import MSXMLHandler

# Unified API for format handlers
from tools.api import (
    validate_file,
    compare_files,
    get_supported_formats,
    get_supported_strategies
)

__all__ = [
    # Models
    'ValidationResult',
    'ComparisonResult',
    'EquivalenceResult',
    # Format handlers
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
    'ScriptHandler',
    'FolderHandler',
    'FAIHandler',
    'BAIHandler',
    'TabixHandler',
    'PDBHandler',
    'RDataHandler',
    'AnnDataHandler',
    'TXTHandler',
    'MSXMLHandler',
    # Unified API for format handlers
    'validate_file',
    'compare_files',
    'get_supported_formats',
    'get_supported_strategies',
]

