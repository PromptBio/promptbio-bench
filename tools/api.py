"""Unified API for format handlers.

This module provides a simplified, unified interface for format-specific validation and comparison tools.
"""

import logging
from typing import Optional, Dict

from tools.factory import FormatHandlerFactory
from tools.results import ValidationResult, ComparisonResult
from utils.format import detect_file_signature

logger = logging.getLogger(__name__)


def validate_file(
    file_path: str,
    format: Optional[str] = None,
    **kwargs
) -> ValidationResult:
    """Validate a bioinformatics file.
    
    Args:
        file_path: Path to the file to validate
        format: Optional format name (e.g., 'fasta', 'bam', 'csv').
                If not provided, format will be automatically detected by FormatDetector.
        **kwargs: optional parameters (e.g., min_size)
            - min_size: Minimum size of the file in bytes to be considered nonempty, default is 0 bytes
    Returns:
        ValidationResult with validation status, details, and error message if validation fails
    """
    # Get handler
    if format:
        handler = FormatHandlerFactory.get_handler(format)
    else:
        handler = FormatHandlerFactory.get_handler_from_path(file_path)
    
    if not handler:
        raise ValueError(
            f"Cannot determine format for file: {file_path}. "
            f"Please specify format explicitly or ensure file has a recognized extension."
        )
    
    # Validate file
    return handler.validate(file_path, **kwargs)


def compare_files(
    reference_path: str,
    candidate_path: str,
    strategy: str = "exact",
    format: Optional[str] = None,
    **kwargs
) -> ComparisonResult:
    """Compare two bioinformatics files.
    
    Args:
        reference_path: Path to reference file
        candidate_path: Path to candidate file
        
        strategy: Comparison strategy (exact, approximate, summary, functional, semantic)
            - exact: Exact matching (for deterministic outputs)
            - approximate: Approximate matching (for near-matches)
            - summary: Comparison of summary statistics (for tasks with intrinsic variability)
            - functional: Comparison of functional equivalence (for results oriented tasks)
            - semantic: Semantic comparison (for order-independent or flexible matching)
        **kwargs: Strategy-specific parameters (e.g., ref_fasta, alt_fasta, min_mapq, overlap_threshold)
    
    Returns:
        ComparisonResult with strategy, similarity score (0.0-1.0), details, and error message if comparison fails
    """
    # Get handler
    if format:
        handler = FormatHandlerFactory.get_handler(format)
    else:
        signature = detect_file_signature(reference_path)
        handler = FormatHandlerFactory.get_handler(signature.format)

    if not handler:
        raise ValueError(
            f"Cannot determine format for files: {candidate_path}, {reference_path}. "
            f"Please specify format explicitly or ensure files have recognized extensions."
        )
    
    # Compare files
    return handler.compare(reference_path, candidate_path, strategy=strategy, **kwargs)


def get_supported_formats(format: str) -> list:
    """Get list of supported file formats/extensions for a given format.
    
    Args:
        format: Format name (e.g., 'fasta', 'bam', 'csv')
    
    Returns:
        List of supported format names/extensions
    
    Example:
        >>> formats = get_supported_formats('fasta')
        >>> print(formats)  # ['fasta', 'fa']
    """
    handler = FormatHandlerFactory.get_handler(format)
    if not handler:
        return []
    
    # Get supported formats from handler
    if hasattr(handler, 'supported_formats'):
        supported = handler.supported_formats
        if isinstance(supported, (list, tuple)):
            return list(supported)
        elif isinstance(supported, (set, dict)):
            return list(supported)
    
    # Fallback: return format name itself
    return [format]


def get_supported_strategies(format: str) -> list:
    """Get list of supported comparison strategies for a format.
    
    Args:
        format: Format name (e.g., 'fasta', 'bam', 'csv')
    
    Returns:
        List of supported strategy names
    
    Example:
        >>> strategies = get_supported_strategies('fasta')
        >>> print(strategies)  # ['exact', 'semantic']
    """
    handler = FormatHandlerFactory.get_handler(format)
    if not handler:
        return []
    
    # Get supported strategies from handler
    if hasattr(handler, 'supported_strategies'):
        strategies = handler.supported_strategies
        if isinstance(strategies, (list, tuple)):
            return list(strategies)
        elif isinstance(strategies, (set, dict)):
            return list(strategies)
    
    # Fallback to default
    return ['exact']
