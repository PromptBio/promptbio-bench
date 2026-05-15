"""TXT format handler for free-format text files.

Strategy:
- exact: Character-level exact text matching
- approximate: Sequence-based similarity using difflib SequenceMatcher
- semantic: LLM-based semantic comparison of text content
- summary: Statistical comparison of text structure (line/word/character counts)
- numeric: RAE-based similarity for files containing a single scalar numeric value

Required cmdline tools:
- (None)
"""

import logging
import re
from typing import Dict, Any
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from tools.base import FormatHandler
from tools.results import ValidationResult, ComparisonResult, EquivalenceResult
from utils.agents import create_llm_agent, invoke_structured_agent
from utils.format import detect_file_signature
from utils.metrics import rae_similarity, min_max_similarity

logger = logging.getLogger(__name__)


class TXTHandler(FormatHandler):
    """Handler for free-format text files."""
    
    def __init__(self):
        super().__init__("txt")
        self.supported_formats = ["txt", "text"]
        self.supported_strategies = ["exact", "approximate", "semantic", "summary", "numeric"]
    
    # ============================================================================
    # Public API Methods
    # ============================================================================
    
    def validate(self, file_path: str, **kwargs) -> ValidationResult:
        """Validate text file format.
        non-empty check is performed by file size, later can be updated by reading content if needed
        
        Args:
            file_path: Path to text file
            **kwargs: optional parameters (e.g., min_size)
            - min_size: Minimum size of the file in bytes, default is 0
        """
        # check existence and non-emptiness (by file size)
        is_exists = self._is_file_exists(file_path)
        is_nonempty = self._is_file_nonempty(file_path, min_size=kwargs.get("min_size", 0))
        
        # check file format
        signature = detect_file_signature(file_path, fallback="extension")
        is_format_ok = signature.is_text or signature.format in self.supported_formats

        # initialize validation result
        result = ValidationResult(exists=is_exists, non_empty=is_nonempty, format_ok=is_format_ok, parseable=False, 
                                  file_path=file_path, signature=signature, details={}, error=None)

        if not is_exists:
            logger.error(f"Text file not found: {file_path}")
            result.error = "File not found"
            return result
        
        if not is_nonempty:
            logger.warning(f"Text file is empty: {file_path}")
            result.error = "File is empty"
            return result
        
        if not is_format_ok:
            logger.warning(f"File may not be a text file: detected={signature.format}, is_text={signature.is_text}")
            if not signature.is_text:
                result.error = f"File does not appear to be text: detected={signature.format}"
                return result
        
        # validate by parsing the text file
        try:
            text_content = self._load_file(file_path)
            if text_content is None:
                result.error = "Could not read file as text"
                return result
            
            result.parseable = True
            result.non_empty = len(text_content.strip()) > 0
            
            if not result.non_empty:
                logger.warning(f"Text file contains only whitespace: {file_path}")
                result.error = "Text file contains only whitespace"
                return result
            
            # Calculate text statistics
            lines = text_content.splitlines()
            words = text_content.split()
            result.details.update({
                "num_lines": len(lines),
                "num_words": len(words),
                "num_characters": len(text_content),
                "num_characters_no_whitespace": len(text_content.replace(" ", "").replace("\n", "").replace("\t", "")),
            })
            
            logger.info(f"Text file validation passed: {file_path}")
        except Exception as e:
            logger.error(f"Text file validation failed: {e}")
            result.error = f"Text file validation failed: {e}"
        return result
    
    def compare(self, reference_path: str, candidate_path: str, strategy: str = "semantic", **kwargs) -> ComparisonResult:
        """Compare text files.
        
        Supported strategies:
        - exact: Strict line-by-line exact matching (all lines must match in order)
        - approximate: Line-by-line comparison with tolerance for insertions/deletions (default)
        - semantic: LLM-based comparison to check if information from reference is present in candidate
        - summary: Summary statistics comparison of text statistics (word counts, character counts, etc.)
        - numeric: RAE-based similarity for files containing a single scalar numeric value

        Args:
            reference_path: Path to reference text file
            candidate_path: Path to candidate text file
            strategy: Comparison strategy (exact, approximate, semantic, summary, numeric)
            **kwargs: Strategy-specific parameters
            - compare_by: For approximate strategy, "line" for line-by-line or "text" for full text (default: "line")
            - ignore_whitespace: Ignore whitespace differences (default: False)
            - ignore_case: Ignore case differences (default: False)
            - question: Description of the comparison task (required for semantic strategy)
            - model: LLM model to use for semantic comparison (default: "gpt-5.4")
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
                    self._compare_exact(ref_validation, alt_validation, **kwargs)
                elif strategy == "approximate":
                    self._compare_approx(ref_validation, alt_validation, **kwargs)
                elif strategy == "semantic":
                    self._compare_semantic(ref_validation, alt_validation, **kwargs)
                elif strategy == "summary":
                    self._compare_summary(ref_validation, alt_validation, **kwargs)
                elif strategy == "numeric":
                    self._compare_numeric(ref_validation, alt_validation, **kwargs)
                else:
                    logger.warning(f"Unknown strategy '{strategy}', fall back to approximate")
                    self._compare_semantic(ref_validation, alt_validation, **kwargs)
        except Exception as e:
            logger.error(f"Text comparison failed: {e}")
            self.result.error = f"Text comparison failed: {e}"
        return self._finalize_output(ref_validation, alt_validation, self.result)
    
    # ============================================================================
    # File I/O and Loading Methods
    # ============================================================================
    
    def _load_file(self, file_path: str, ignore_whitespace: bool = False, ignore_case: bool = False) -> str:
        """Load text file content, optionally normalizing it.
        
        Args:
            file_path: Path to text file
            ignore_whitespace: Remove all whitespace if True (default: False)
            ignore_case: Convert to lowercase if True (default: False)
            
        Returns:
            Text content as string (normalized if requested), or None if loading fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            
            # Normalize if requested
            if ignore_case:
                text = text.lower()
            if ignore_whitespace:
                text = re.sub(r'\s+', '', text)
            
            return text
        except Exception as e:
            logger.warning(f"Failed to read text file {file_path}: {e}")
            return ""
    
    # ============================================================================
    # Statistics and Analysis Methods
    # ============================================================================
    
    def _get_text_stats(self, text: str) -> Dict[str, Any]:
        """Calculate text statistics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text statistics
        """
        lines = text.splitlines()
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return {
            "num_lines": len(lines),
            "num_words": len(words),
            "num_characters": len(text),
            "num_characters_no_whitespace": len(text.replace(" ", "").replace("\n", "").replace("\t", "")),
            "num_sentences": len(sentences),
        }
    
    # ============================================================================
    # Private Comparison Implementation Methods
    # ============================================================================
    
    def _build_comparison_prompt(self, task_context: str = "", reference_path: str = "", candidate_path: str = "") -> str:
        """Build the comparison prompt template for text equivalence analysis.
        
        Args:
            task_context: Optional task description for context
            reference_path: Path to reference text file
            candidate_path: Path to candidate text file
            
        Returns:
            Formatted prompt string for LLM comparison
        """
        prompt_template = """Analyze the equivalence between two text files:

{task_context}

Reference Text (path: {reference_path}):
Candidate Text (path: {candidate_path}):

Step 1: Understand the text contents under the context
First, carefully read and understand both texts within the provided task context. Identify:
- The main topic, purpose, and context of each text (considering the task context provided)
- Key concepts, themes, and messages as they relate to the given context
- The structure and organization of information
- Any numerical data, statistics, measurements, or quantitative information
- How the content relates to and should be interpreted within the specified context

Step 2: Compare the texts with special attention to numbers
When comparing the texts, pay particular attention to numerical values:
- Extract and identify all numbers, statistics, measurements, percentages, counts, and other quantitative data from both texts
- Match corresponding numbers: For each number in the reference text, find the corresponding number in the candidate text (same measurement, statistic, or quantity)
- Compare corresponding numbers: Determine if they are close/similar or significantly different
  - Compute ratios when helpful: Calculate the ratio between candidate and reference numbers (e.g., if reference is 100 and candidate is 98, ratio is 0.98)
  - Numbers are considered close if the ratio is close to 1.0 (e.g., 0.95-1.05 range for most cases, or within reasonable tolerance based on context)
  - For similar numbers: Consider them equivalent even if there are minor formatting differences (e.g., 1.0 vs 1.00, 1000 vs 1,000) or small numerical differences (within tolerance)
- For different numbers: Assess whether the differences are significant enough to affect the meaning or conclusions
- Consider the context: Are the numbers measuring the same thing? Are they in the same units? What is a reasonable tolerance for this type of measurement?

Guidelines for recommendation:
- "equivalent": The candidate text contains all key information from the reference, including similar numerical values (within reasonable tolerance), even if expressed differently
- "partially equivalent": The candidate text contains most key information but may have notable omissions, additions, or some numerical differences that could affect interpretation
- "not equivalent": The candidate text differs significantly in content, key information is missing, or numerical values are substantially different
- "uncertain": Cannot determine equivalence due to ambiguity, complexity, or lack of clarity

Focus on semantic equivalence. Consider:
- Are the main points and facts from the reference present in the candidate?
- Are there any critical pieces of information missing?
- Is the information expressed differently but still equivalent?
- Are there any significant additions or omissions that affect meaning?
- **Most importantly: Are numerical values similar? If numbers differ, how significant are the differences?**

Provide a detailed analysis with ALL required fields:
- equivalence: A numeric score between 0.0 and 1.0 representing equivalence (1.0 = fully equivalent, 0.5 = partially equivalent, 0.0 = not equivalent)
- recommendation: One of: equivalent, partially equivalent, not equivalent, uncertain
- confidence: A numeric score between 0.0 and 1.0 representing your confidence in this assessment
- explanation: Detailed explanation including key similarities, differences, numerical comparisons, and rationale for the scores

CRITICAL: You MUST provide numeric values for both equivalence and confidence. Do not omit these fields."""
        
        return prompt_template.format(task_context=task_context, reference_path=reference_path, candidate_path=candidate_path)
    
    def _compare_exact(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Exact match: Strict line-by-line comparison.
        
        Compares lines in order with no tolerance for insertions/deletions.
        Returns 1.0 if all lines match exactly in order, 0.0 otherwise.
        
        Args:
            **kwargs: Strategy-specific parameters
            - ignore_whitespace: Ignore whitespace differences (default: False)
            - ignore_case: Ignore case differences (default: False)
        """
        try:
            ignore_whitespace = kwargs.get("ignore_whitespace", False)
            ignore_case = kwargs.get("ignore_case", False)
            
            ref_normalized = self._load_file(ref_validation.file_path, ignore_whitespace, ignore_case)
            alt_normalized = self._load_file(alt_validation.file_path, ignore_whitespace, ignore_case)
            
            # Strict line-by-line comparison in order
            ref_lines = ref_normalized.splitlines()
            alt_lines = alt_normalized.splitlines()
            
            self.result.details.update({
                "ref_lines": len(ref_lines),
                "alt_lines": len(alt_lines),
                "ignore_whitespace": ignore_whitespace,
                "ignore_case": ignore_case
            })

            # Must have same number of lines
            if len(ref_lines) != len(alt_lines):
                self.result.similarity = 0.0
                return
            
            # All lines must match exactly in order
            similarity = 1.0 if all(r == a for r, a in zip(ref_lines, alt_lines)) else 0.0
            self.result.similarity = similarity
        except Exception as e:
            logger.error(f"Exact comparison failed: {e}")
            self.result.error = f"Exact comparison failed: {e}"
    
    def _compare_approx(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Approximate match: Comparison with tolerance for insertions/deletions.
        
        Uses SequenceMatcher to find matching content, handling insertions and deletions.
        Can compare line-by-line or as full text.
        This is useful for detecting if candidate text contains similar information
        to the reference, even with some content added or removed.
        
        Args:
            **kwargs: Strategy-specific parameters
            - compare_by: Comparison granularity - "line" for line-by-line or "text" for full text (default: "line")
            - ignore_whitespace: Ignore whitespace differences (default: False)
            - ignore_case: Ignore case differences (default: False)
        """
        try:
            compare_by = kwargs.get("compare_by", "line")
            ignore_whitespace = kwargs.get("ignore_whitespace", True)
            ignore_case = kwargs.get("ignore_case", True)
            
            ref_normalized = self._load_file(ref_validation.file_path, ignore_whitespace, ignore_case)
            alt_normalized = self._load_file(alt_validation.file_path, ignore_whitespace, ignore_case)
            
            self.result.details.update({
                "compare_by": compare_by,
                "ignore_whitespace": ignore_whitespace,
                "ignore_case": ignore_case
            })

            if compare_by == "text":
                # Full text comparison
                matcher = SequenceMatcher(None, ref_normalized, alt_normalized)
                matching_blocks = matcher.get_matching_blocks()
                total_matching_chars = sum(block.size for block in matching_blocks)
                
                # Calculate similarity relative to reference: matching characters / reference length
                similarity = total_matching_chars / len(ref_normalized) if len(ref_normalized) > 0 else 0.0
                
                self.result.similarity = similarity
                self.result.details.update({
                    "ref_length": len(ref_normalized),
                    "alt_length": len(alt_normalized),
                    "matching_characters": total_matching_chars,
                })
            else:
                # Line-by-line comparison with tolerance for insertions/deletions
                ref_lines = ref_normalized.splitlines()
                alt_lines = alt_normalized.splitlines()
                
                # Use SequenceMatcher to find matching blocks, handling insertions/deletions
                matcher = SequenceMatcher(None, ref_lines, alt_lines)
                matching_blocks = matcher.get_matching_blocks()
                
                # Count total matching lines across all matching blocks
                total_matching_lines = sum(block.size for block in matching_blocks)
                
                # Calculate similarity relative to reference: matching lines / reference lines
                similarity = total_matching_lines / len(ref_lines) if len(ref_lines) > 0 else 0.0
                
                self.result.similarity = similarity
                self.result.details.update({
                    "ref_lines": len(ref_lines),
                    "alt_lines": len(alt_lines),
                    "matching_lines": total_matching_lines,
                })
        except Exception as e:
            logger.error(f"Approximate comparison failed: {e}")
            self.result.error = f"Approximate comparison failed: {e}"
    
    def _compare_semantic(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Semantic comparison: LLM-based comparison to check if information from reference is present in candidate.
        
        Uses LLM to determine if the candidate text contains the key information
        from the reference text, even if expressed differently.
        
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
            eval_guideline = kwargs.get("eval_guideline", None)
            model = kwargs.get("model", "gpt-5.4")

            # Load text content
            ref_text = self._load_file(ref_validation.file_path)
            alt_text = self._load_file(alt_validation.file_path)
            
            # Create agent
            agent = create_llm_agent(
                model=model,
                system_prompt="You are a scientific text comparison expert specializing in bioinformatics. Your task is to analyze text equivalence by: (1) understanding content within the provided context, (2) paying special attention to numerical values and comparing them using ratios when appropriate, and (3) determining if the candidate text contains the key information from the reference, even if expressed differently.",
                response_format=EquivalenceResult
            )
            
            task_context = f"Task: {question}\n" if question else ""
            if eval_guideline:
                task_context += f"Evaluation guideline: {eval_guideline}\n"
            prompt = self._build_comparison_prompt(
                task_context=task_context,
                reference_path=ref_validation.file_path,
                candidate_path=alt_validation.file_path
            )
            
            # Prepare messages with text content
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "Reference Text:"},
                    {"type": "text", "text": ref_text},
                    {"type": "text", "text": "Candidate Text:"},
                    {"type": "text", "text": alt_text}
                ]
            }]
            
            # Use invoke_structured_agent to extract structured response
            analysis = invoke_structured_agent(agent, messages, EquivalenceResult)
            
            # Override paths with actual file paths (LLM may use placeholder paths)
            analysis.reference_path = ref_validation.file_path
            analysis.candidate_path = alt_validation.file_path
            
            logger.info(
                f"Text comparison completed: equivalence={analysis.equivalence:.2%}, "
                f"recommendation={analysis.recommendation}"
            )
            
            details = {
                "recommendation": analysis.recommendation,
                "confidence": analysis.confidence,
                "explanation": analysis.explanation,
                "ref_length": len(ref_text),
                "alt_length": len(alt_text),
            }
            
            self.result.similarity = analysis.equivalence
            self.result.details.update(details)
        except Exception as e:
            logger.error(f"Semantic comparison failed: {e}", exc_info=True)
            self.result.error = f"Semantic comparison failed: {e}"
    
    def _compare_summary(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Summary statistics comparison: Compare text statistics.
        
        Compares various text statistics and calculates similarity based on how close
        these statistics are.
        
        Args:
            **kwargs: Strategy-specific parameters (none currently used)
        """
        try:
            ref_text = self._load_file(ref_validation.file_path)
            alt_text = self._load_file(alt_validation.file_path)

            ref_stats = self._get_text_stats(ref_text)
            alt_stats = self._get_text_stats(alt_text)
            
            numeric_stats = [
                "num_lines", "num_words", "num_characters", 
                "num_characters_no_whitespace", "num_sentences",
            ]
            
            ref_series = pd.Series([ref_stats.get(stat, 0) for stat in numeric_stats], index=numeric_stats)
            alt_series = pd.Series([alt_stats.get(stat, 0) for stat in numeric_stats], index=numeric_stats)
            
            similarities = [float(rae_similarity([ref_series[stat]], [alt_series[stat]])[0]) for stat in numeric_stats]
            
            self.result.similarity = sum(similarities) / len(similarities) if similarities else 0.0
            self.result.details.update({
                "ref_stats": ref_stats,
                "alt_stats": alt_stats,
                "individual_similarities": dict(zip(numeric_stats, similarities))
            })
        except Exception as e:
            logger.error(f"Summary statistics comparison failed: {e}")
            self.result.error = f"Summary statistics comparison failed: {e}"

    def _compare_numeric(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Numeric scalar comparison: parse a single float from each file and compute similarity.

        Uses RAE-based similarity: 1 - |ref - alt| / |ref|.
        Also records min_max_similarity as a secondary metric in details.
        """
        try:
            ref_text = self._load_file(ref_validation.file_path).strip()
            alt_text = self._load_file(alt_validation.file_path).strip()

            ref_val = float(ref_text)
            alt_val = float(alt_text)

            self.result.similarity = float(rae_similarity([ref_val], [alt_val])[0])
            self.result.details.update({
                "ref_value": ref_val,
                "alt_value": alt_val,
                "min_max_similarity": min_max_similarity(ref_val, alt_val),
            })
        except ValueError as e:
            logger.error(f"Numeric comparison failed — could not parse float: {e}")
            self.result.error = f"Numeric comparison failed: {e}"
        except Exception as e:
            logger.error(f"Numeric comparison failed: {e}")
            self.result.error = f"Numeric comparison failed: {e}"
