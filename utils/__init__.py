"""Utility functions for the evaluation framework."""

from utils.format import detect_format_from_extension, detect_file_signature, get_file_extension
from utils.agents import create_llm_agent, invoke_structured_agent, extract_token_usage, load_api
from utils.paths import find_project_root, get_project_paths, list_files, format_paths, find_subdir, find_log_file
from utils.file_io import read_file
from utils.logger import Tee, make_logger
from utils.compare import ComparisonRunner
from utils.match import match_files_with_llm, match_output_files
from utils.metrics import (
    pearson_correlation,
    spearman_correlation,
    kendall_correlation,
    cosine_similarity,
    hellinger_distance,
    mean_absolute_error,
    mean_squared_error,
    relative_absolute_error,
    rae_similarity,
    min_max_similarity,
    normalized_difference,
    precision_recall_f1,
    jaccard_similarity,
)
from utils.strategy import StrategyRecommender
from utils.models import (
    ComparisonSummary, FileEvalState,
    EvalFile, StrategyRecommendation, StrategyResult,
    FileMapping, FileMappingResult,
)
from utils.syntax import (
    check_syntax,
    check_python_syntax,
    check_r_syntax,
    check_shell_syntax,
    check_perl_syntax,
    check_julia_syntax,
)
from utils.trace import extract_trace, TraceStep, AgentTrace

__all__ = [
    # Format utilities
    'detect_format_from_extension',
    'detect_file_signature',
    'get_file_extension',
    # Path utilities
    'find_project_root',
    'get_project_paths',
    'list_files',
    'format_paths',
    'find_subdir',
    'find_log_file',
    # Agent utilities
    'create_llm_agent',
    'invoke_structured_agent',
    'extract_token_usage',
    'load_api',
    # File I/O
    'read_file',
    # Comparison & runner
    'Tee',
    'make_logger',
    'ComparisonRunner',
    'ComparisonSummary',
    'FileEvalState',
    # File matching
    'match_files_with_llm',
    'match_output_files',
    # Metrics
    'pearson_correlation',
    'spearman_correlation',
    'kendall_correlation',
    'cosine_similarity',
    'hellinger_distance',
    'mean_absolute_error',
    'mean_squared_error',
    'relative_absolute_error',
    'rae_similarity',
    'min_max_similarity',
    'normalized_difference',
    'precision_recall_f1',
    'jaccard_similarity',
    # Strategy
    'StrategyRecommender',
    # Models
    'ComparisonSummary',
    'FileEvalState',
    'EvalFile',
    'StrategyRecommendation',
    'StrategyResult',
    'FileMapping',
    'FileMappingResult',
    # Syntax checking
    'check_syntax',
    'check_python_syntax',
    'check_r_syntax',
    'check_shell_syntax',
    'check_perl_syntax',
    'check_julia_syntax',
    # Trace extraction
    'extract_trace',
    'TraceStep',
    'AgentTrace',
]
