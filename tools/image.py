"""Image format handler for scientific images (PNG, JPG, PDF, SVG, etc.).

Strategy:
- semantic: LLM-based semantic comparison of image content

Required cmdline tools:
- poppler (for PDF conversion via pdf2image)
"""

import logging
import os
import base64
from io import BytesIO
from typing import Optional

import numpy as np
import cairosvg
from PIL import Image
from pdf2image import convert_from_path

from utils.agents import create_llm_agent, invoke_structured_agent
from utils.format import detect_file_signature

from tools.base import FormatHandler
from tools.results import ValidationResult, ComparisonResult, EquivalenceResult

logger = logging.getLogger(__name__)


class ImageHandler(FormatHandler):
    """Handler for image format files (PNG, JPG, PDF, SVG, etc.)."""
    
    def __init__(self):
        super().__init__("image")
        self.supported_formats = {'png', 'jpg', 'jpeg', 'pdf', 'svg', 'gif', 'bmp', 'tiff', 'tif', 'webp'}
        self.supported_strategies = ["semantic"]
    
    # ============================================================================
    # Public API Methods
    # ============================================================================
    
    def validate(self, file_path: str, **kwargs) -> ValidationResult:
        """Validate image file format.
        non-empty check is performed by file size, later can be updated by reading content if needed
        
        Args:
            file_path: Path to image file
            **kwargs: optional parameters (e.g., min_size)
            - min_size: Minimum size of the file in bytes, default is 0
        """
        # check existence and non-emptiness (by file size)
        is_exists = self._is_file_exists(file_path)
        is_nonempty = self._is_file_nonempty(file_path, min_size=kwargs.get("min_size", 0))
        
        # check file format
        signature = detect_file_signature(file_path, fallback=True)
        ext = signature.extension or ""
        is_format_ok = (ext[1:] if ext else "") in self.supported_formats or signature.format in self.supported_formats

        # initialize validation result
        result = ValidationResult(exists=is_exists, non_empty=is_nonempty, format_ok=is_format_ok, parseable=False, 
                                  file_path=file_path, signature=signature, details={}, error=None)

        if not is_exists:
            logger.error(f"Image file not found: {file_path}")
            result.error = "File not found"
            return result
        
        if not is_nonempty:
            logger.warning(f"Image file is empty: {file_path}")
            result.error = "File is empty"
            return result
        
        if not is_format_ok:
            logger.error(f"Not an image file: detected={signature.format}, extension={ext}")
            result.error = f"Not an image file: detected={signature.format}, extension={ext}"
            return result
        
        # validate by parsing the image file
        try:
            img = self._load_file(file_path)
            width, height = img.size
            
            result.parseable = True
            result.non_empty = width > 0 and height > 0
            
            if not result.non_empty:
                logger.warning(f"Image file has invalid dimensions: {file_path}")
                result.error = "Image file has invalid dimensions"
                return result
            
            result.details.update({
                "width": width,
                "height": height,
                "mode": img.mode,
                "extension": signature.extension
            })
            
            logger.info(f"Image validation passed: {ext}, {width}x{height}")
        except Exception as e:
            logger.error(f"Image validation failed: {e}")
            result.error = f"Image validation failed: {e}"
        return result
    
    def compare(self, reference_path: str, candidate_path: str, strategy: str = "semantic", **kwargs) -> ComparisonResult:
        """Compare two image files using LLM-based semantic analysis.
        
        Supported strategies:
        - semantic: LLM-based semantic comparison of image content
        
        Args:
            reference_path: Path to reference image
            candidate_path: Path to candidate image
            strategy: Comparison strategy (semantic)
            **kwargs: Strategy-specific parameters
            - question: Optional task description for context
            - model: LLM model name for image comparison (default: "gpt-5.4")
            - Additional parameters (e.g., temperature, max_tokens)
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
                    # LLM-based semantic comparison of image content
                    self._compare_semantic(ref_validation, alt_validation, **kwargs)
                else:
                    logger.warning(f"Unknown strategy '{strategy}', fall back to semantic")
                    self._compare_semantic(ref_validation, alt_validation, **kwargs)
        except Exception as e:
            logger.error(f"Image comparison failed: {e}")
            self.result.error = f"Image comparison failed: {e}"
        return self._finalize_output(ref_validation, alt_validation, self.result)
    
    # ============================================================================
    # File I/O and Loading Methods
    # ============================================================================
    
    def _load_file(self, file_path: str) -> Image.Image:
        """Load image from file, handling different formats.
        
        Args:
            file_path: Path to image file
        
        Returns:
            PIL Image object
        """
        signature = detect_file_signature(file_path, fallback=True)
        ext = signature.extension or ""
        fmt = signature.format or ""
        
        # Use format (fmt) as primary indicator, fall back to extension (ext) if format is unknown
        if fmt == 'pdf' or (fmt in ('unknown', '') and ext == 'pdf'):
            images = convert_from_path(file_path, first_page=1, last_page=1, dpi=150)
            if not images:
                raise ValueError("PDF contains no renderable pages")
            return images[0]
        if fmt == 'svg' or (fmt in ('unknown', '') and ext == 'svg'):
            png_data = cairosvg.svg2png(url=file_path)
            with Image.open(BytesIO(png_data)) as img:
                return img.copy()
        with Image.open(file_path) as img:
            return img.copy()
    
    def _prepare_image_for_llm(self, file_path: str) -> dict:
        """Load image from file and prepare it for LLM consumption.
        
        Args:
            file_path: Path to image file
        
        Returns:
            Dictionary with image data in base64 format for LLM consumption
        """
        img = self._load_file(file_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": img_base64
            }
        }
    
    # ============================================================================
    # Statistics and Analysis Methods
    # ============================================================================
    
    def _build_comparison_prompt(self, task_context: str = "", reference_path: str = "", candidate_path: str = "") -> str:
        """Create a prompt template for image comparison analysis.
        
        Args:
            task_context: Optional task description for context
            reference_path: Path to reference image
            candidate_path: Path to candidate image
        
        Returns:
            Formatted prompt string for image comparison
        """
        prompt_template = """Analyze the equivalence between two scientific images (e.g., plots, visualizations, charts):

{task_context}

Reference Image (path: {reference_path}):
Candidate Image (path: {candidate_path}):

Note: Image dimensions (resolution/size) should NOT affect equivalence judgment. Focus on content, data, and scientific message rather than pixel-level differences or file size.

Guidelines for recommendation:
- "equivalent": Images show the same scientific content, data, and convey the same message (minor visual differences like colors, fonts, or exact positioning are acceptable)
- "partially equivalent": Images show similar content but may have differences in data, scales, or missing elements that could affect interpretation
- "not equivalent": Images differ significantly in content, data, or scientific message
- "uncertain": Cannot determine equivalence due to image quality, complexity, or lack of clarity

Focus on scientific content and information rather than exact pixel-level matches. Consider:
- Are the same data/patterns shown?
- Are the axes, labels, and legends appropriate?
- Is the scientific message the same?
- Are there any critical missing elements?

Provide a detailed analysis with ALL required fields:
- equivalence: A numeric score between 0.0 and 1.0 representing equivalence (1.0 = fully equivalent, 0.5 = partially equivalent, 0.0 = not equivalent)
- recommendation: One of: equivalent, partially equivalent, not equivalent, uncertain
- confidence: A numeric score between 0.0 and 1.0 representing your confidence in this assessment
- explanation: Detailed explanation including what each image shows, key similarities and differences, rationale for the scores, and any notable issues

CRITICAL: You MUST provide numeric values for both equivalence and confidence. Do not omit these fields."""
        
        return prompt_template.format(task_context=task_context, reference_path=reference_path, candidate_path=candidate_path)
    
    # ============================================================================
    # Private Comparison Implementation Methods
    # ============================================================================
    
    def _compare_semantic(self, ref_validation: ValidationResult, alt_validation: ValidationResult, **kwargs):
        """Semantic comparison: LLM-based semantic comparison of image content.
        
        Uses LLM to determine if images are scientifically equivalent, even if
        they have minor visual differences.
        
        Args:
            ref_validation: Validation result for reference file
            alt_validation: Validation result for candidate file
            **kwargs: Strategy-specific parameters
            - question: Optional task description for context
            - model: LLM model to use (default: "gpt-5.4")
        """
        try:
            # Extract strategy-specific parameters
            question = kwargs.get("question", None)
            eval_guideline = kwargs.get("eval_guideline", None)
            model = kwargs.get("model", "gpt-5.4")
            
            # Load and prepare images for LLM
            ref_image_content = self._prepare_image_for_llm(ref_validation.file_path)
            alt_image_content = self._prepare_image_for_llm(alt_validation.file_path)
            
            # Create agent and invoke
            agent = create_llm_agent(
                model=model,
                system_prompt="You are a scientific image analyzer. Compare images and provide structured equivalence analysis.",
                response_format=EquivalenceResult
            )
            
            task_context = f"\nTask: {question}" if question else ""
            if eval_guideline:
                task_context += f"\nEvaluation guideline: {eval_guideline}"
            prompt = self._build_comparison_prompt(
                task_context=task_context,
                reference_path=ref_validation.file_path,
                candidate_path=alt_validation.file_path
            )
            
            # Prepare messages with image content
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "Reference Image:"},
                    ref_image_content,
                    {"type": "text", "text": "Candidate Image:"},
                    alt_image_content
                ]
            }]
            
            # Use invoke_structured_agent to extract structured response
            analysis = invoke_structured_agent(agent, messages, EquivalenceResult)
            
            # Override paths with actual file paths (LLM may use placeholder paths)
            analysis.reference_path = ref_validation.file_path
            analysis.candidate_path = alt_validation.file_path
            
            logger.info(
                f"Image comparison completed: equivalence={analysis.equivalence:.2%}, "
                f"recommendation={analysis.recommendation}"
            )
            
            details = {
                "recommendation": analysis.recommendation,
                "confidence": analysis.confidence,
                "explanation": analysis.explanation,
            }
            
            self.result.similarity = analysis.equivalence
            self.result.details.update(details)
        except Exception as e:
            logger.error(f"Semantic comparison failed: {e}", exc_info=True)
            self.result.error = f"Semantic comparison failed: {e}"
