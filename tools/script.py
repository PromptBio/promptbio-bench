"""Script format handler for semantic LLM-based comparisons.

Strategy:
- semantic: LLM-based semantic comparison of script logic and correctness

Required cmdline tools:
- (None)
"""

import logging
import os
from typing import Optional

import numpy as np

from utils.agents import create_llm_agent, invoke_structured_agent
from utils.format import detect_file_signature
from utils.syntax import check_syntax

from tools.base import FormatHandler
from tools.results import ValidationResult, ComparisonResult, EquivalenceResult

logger = logging.getLogger(__name__)


class ScriptHandler(FormatHandler):
    """Handler for script files (Python, R, shell, etc.) using semantic LLM comparison."""

    def __init__(self):
        super().__init__("script")
        self.supported_extensions = {"py", "r", "sh", "bash", "pl", "jl"}
        self.supported_formats = ["script"]
        self.supported_strategies = ["semantic"]
        self.format_map = {
            "py": "python",
            "r": "r",
            "sh": "bash",
            "bash": "bash",
            "pl": "perl",
            "jl": "julia",
        }
        self.max_prompt_chars = 10_000
    
    # ============================================================================
    # Public API Methods
    # ============================================================================
    
    def validate(self, file_path: str, **kwargs) -> ValidationResult:
        """Validate script file format.
        non-empty check is performed by file size, later can be updated by reading content if needed
        
        Args:
            file_path: Path to script file
            **kwargs: optional parameters (e.g., min_size)
            - min_size: Minimum size of the file in bytes, default is 0
        """
        # check existence and non-emptiness (by file size)
        is_exists = self._is_file_exists(file_path)
        is_nonempty = self._is_file_nonempty(file_path, min_size=kwargs.get("min_size", 0))
        
        # check file format
        signature = detect_file_signature(file_path, fallback=True)
        ext = signature.extension or ""
        is_format_ok = ext in self.supported_extensions or signature.format in self.supported_formats

        # initialize validation result
        result = ValidationResult(exists=is_exists, non_empty=is_nonempty, format_ok=is_format_ok, parseable=False, 
                                  file_path=file_path, signature=signature, details={}, error=None)

        if not is_exists:
            logger.error(f"Script file not found: {file_path}")
            result.error = "File not found"
            return result
        
        if not is_nonempty:
            logger.warning(f"Script file is empty: {file_path}")
            result.error = "File is empty"
            return result
        
        if not is_format_ok:
            logger.error(f"Unsupported script format: {file_path}, extension={ext}")
            result.error = f"Unsupported script extension: {ext}"
            return result
        
        # validate by parsing the script file
        try:
            content = self._load_script(file_path, truncate=False)
            
            # Check syntax
            syntax_valid, syntax_error = check_syntax(file_path, content)
            if not syntax_valid:
                logger.warning(f"Script syntax error in {file_path}: {syntax_error}")
                result.parseable = False
                result.error = f"Syntax error: {syntax_error}"
                result.details.update({"extension": ext})
            elif syntax_error and "syntax check skipped" in syntax_error:
                logger.warning(f"{syntax_error} for {file_path}")
                result.parseable = True
                result.details.update({"warning": syntax_error, "extension": ext})
            else:
                line_count = content.count("\n") + 1
                result.parseable = True
                result.details.update({"lines": line_count, "extension": ext})
                logger.info(f"Script validation passed: {ext}, {line_count} lines")
        except Exception as e:
            logger.error(f"Script validation failed: {e}")
            result.error = f"Script validation failed: {e}"
        return result
    
    def compare(self, reference_path: str, candidate_path: str, strategy: str = "semantic", **kwargs) -> ComparisonResult:
        """Compare two script files using LLM-based semantic analysis.
        
        Supported strategies:
        - semantic: LLM-based semantic equivalence analysis (default)
        
        Args:
            reference_path: Path to reference script file
            candidate_path: Path to candidate script file
            strategy: Comparison strategy (semantic)
            **kwargs: Strategy-specific parameters (e.g., question, model)
            - question: Optional task description for context (default: None)
            - model: LLM model to use for comparison (default: "gpt-5.4")
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
                if strategy == "semantic":
                    # LLM-based semantic equivalence analysis
                    self._compare_semantic(ref_validation, alt_validation, **kwargs)
                else:
                    logger.warning(f"Unknown strategy '{strategy}', fall back to semantic")
                    self._compare_semantic(ref_validation, alt_validation, **kwargs)
        except Exception as e:
            logger.error(f"Script comparison failed: {e}")
            self.result.error = f"Script comparison failed: {e}"
        return self._finalize_output(ref_validation, alt_validation, self.result)
    
    # ============================================================================
    # File I/O and Loading Methods
    # ============================================================================
    
    def _load_script(self, file_path: str, truncate: bool = True) -> str:
        """Load script from file, optionally truncating to max_prompt_chars.
        
        Args:
            file_path: Path to script file
            truncate: Whether to truncate content to max_prompt_chars (default: True)
        
        Returns:
            Script content as string (truncated if truncate=True and content exceeds max_prompt_chars)
        """
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        
        if truncate and len(content) > self.max_prompt_chars:
            return content[:self.max_prompt_chars] + "\n# ... truncated ..."
        return content
    
    # ============================================================================
    # Statistics and Analysis Methods
    # ============================================================================
    
    def _build_comparison_prompt(self, task_context: str, reference_path: str, candidate_path: str) -> str:
        """Build the comparison prompt template for script equivalence analysis.
        
        Args:
            task_context: Task description context (can be empty string)
            reference_path: Path to the reference script
            candidate_path: Path to the candidate script
        
        Returns:
            Formatted prompt string for LLM comparison
        """
        return f"""Analyze the equivalence between two bioinformatics scripts:

{task_context}

Reference Script (path: {reference_path})
Candidate Script (path: {candidate_path})

Note: The scripts may be written in different programming languages. Focus on functional and methodological equivalence rather than language-specific syntax differences.

Guidelines for recommendation:
- "equivalent": Scripts are functionally equivalent and achieve the same outcome
- "partially equivalent": Scripts are similar but have notable differences affecting reproducibility or correctness
- "not equivalent": Scripts differ significantly in logic, steps, or expected results
- "uncertain": Cannot determine equivalence due to ambiguity, missing context, or incomplete code

Focus on functional and methodological equivalence. Consider:
- Are the same operations/steps performed?
- Is the logic and workflow equivalent?
- Are there differences in data handling, error checking, or edge cases?
- Is the code correct, complete, and reproducible?

Provide a detailed analysis with ALL required fields:
- equivalence: A numeric score between 0.0 and 1.0 representing equivalence (1.0 = fully equivalent, 0.5 = partially equivalent, 0.0 = not equivalent)
- recommendation: One of: equivalent, partially equivalent, not equivalent, uncertain
- confidence: A numeric score between 0.0 and 1.0 representing your confidence in this assessment
- explanation: Detailed explanation including key similarities, differences, correctness assessment, and rationale for the scores

CRITICAL: You MUST provide numeric values for both equivalence and confidence. Do not omit these fields.
"""
    
    # ============================================================================
    # Private Comparison Implementation Methods
    # ============================================================================
    
    def _compare_semantic(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Semantic comparison: LLM-based semantic equivalence analysis.
        
        Uses LLM to determine if scripts are functionally equivalent, even if
        expressed differently or in different languages.
        
        Args:
            ref_validation: Validation result for reference file
            alt_validation: Validation result for candidate file
            **kwargs: Strategy-specific parameters
            - question: Optional task description for context (default: None)
            - model: LLM model to use (default: "gpt-5.4")
        """
        try:
            # Get strategy-specific parameters
            question = kwargs.get("question", None)
            model = kwargs.get("model", "gpt-5.4")
            
            # Load full content for line counting
            ref_code = self._load_script(ref_validation.file_path, truncate=False)
            alt_code = self._load_script(alt_validation.file_path, truncate=False)
            # Load truncated content for prompt comparison
            ref_snippet = self._load_script(ref_validation.file_path)
            alt_snippet = self._load_script(alt_validation.file_path)

            # Detect languages from extensions
            ref_ext = ref_validation.signature.extension or ""
            alt_ext = alt_validation.signature.extension or ""
            ref_language = self.format_map.get(ref_ext, "")
            alt_language = self.format_map.get(alt_ext, "")

            agent = create_llm_agent(
                model=model,
                system_prompt="You are a bioinformatics code reviewer. Compare scripts for functional and methodological equivalence.",
                response_format=EquivalenceResult,
            )

            task_context = f"Task: {question}\n" if question else ""
            prompt = self._build_comparison_prompt(
                task_context=task_context,
                reference_path=ref_validation.file_path,
                candidate_path=alt_validation.file_path,
            )

            # Prepare messages with script content
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": f"Reference Script:"},
                    {"type": "text", "text": f"```{ref_language}\n{ref_snippet}\n```"},
                    {"type": "text", "text": f"Candidate Script:"},
                    {"type": "text", "text": f"```{alt_language}\n{alt_snippet}\n```"},
                ]
            }]

            # Use invoke_structured_agent to extract structured response
            analysis = invoke_structured_agent(agent, messages, EquivalenceResult)
            
            # Override paths with actual file paths (LLM may use placeholder paths)
            analysis.reference_path = ref_validation.file_path
            analysis.candidate_path = alt_validation.file_path
            
            logger.info(
                f"Script comparison completed: equivalence={analysis.equivalence:.2%}, "
                f"recommendation={analysis.recommendation}"
            )

            details = {
                "recommendation": analysis.recommendation,
                "confidence": analysis.confidence,
                "explanation": analysis.explanation,
                "ref_lines": ref_code.count("\n") + 1,
                "alt_lines": alt_code.count("\n") + 1,
                "ref_extension": ref_validation.signature.extension or "",
                "alt_extension": alt_validation.signature.extension or "",
            }

            self.result.similarity = analysis.equivalence
            self.result.details.update(details)
        except Exception as e:
            logger.error(f"Semantic comparison failed: {e}", exc_info=True)
            self.result.error = f"Semantic comparison failed: {e}"

