"""LLM agent utilities for structured output."""

import logging
import os
from typing import Any, Dict, Optional, TypeVar, Type

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T')


def create_llm_agent(
    model: str,
    system_prompt: str,
    response_format: Optional[Type[T]] = None,
    tools: Optional[list] = None,
    temperature: float = 0.0,
    **kwargs: Any,
) -> Any:
    """Create an LLM agent with structured output support.
    
    Supports OpenAI (direct) and OpenRouter (for third-party models).
    
    Args:
        model: Model name
            - OpenAI direct: "gpt-4o", "gpt-3.5-turbo", etc.
            - OpenRouter: "anthropic/claude-3.5-sonnet", "google/gemini-pro", etc.
        system_prompt: System prompt for the agent
        response_format: Optional Pydantic model for structured output
        tools: Optional list of tools for the agent
        temperature: Temperature for LLM sampling (default: 0.0 for deterministic results)
        **kwargs: Additional LLM parameters (e.g., max_tokens, top_p)
    
    Returns:
        Configured agent
    """
    # Initialize the chat model with temperature and other parameters
    chat_model = init_chat_model(model, temperature=temperature, **kwargs)
    
    return create_agent(
        model=chat_model,
        tools=tools or [],
        system_prompt=system_prompt,
        response_format=response_format
    )


def invoke_structured_agent(
    agent,
    prompt: str | list | dict,
    response_model: Type[T]
) -> T:
    """
    Invoke an agent and extract structured output.
    
    Args:
        agent: LLM agent
        prompt: User prompt (str), messages list, or dict with "messages" key
        response_model: Pydantic model class for structured output
    
    Returns:
        Validated Pydantic model instance
    
    Raises:
        ValueError: If response cannot be validated
    """
    # Handle different input formats
    if isinstance(prompt, str):
        # Simple string prompt
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list):
        # List of messages
        messages = prompt
    elif isinstance(prompt, dict):
        # Dict with messages or direct invoke input
        if "messages" in prompt:
            messages = prompt["messages"]
        else:
            messages = prompt
    else:
        raise ValueError(f"Unsupported prompt type: {type(prompt)}")
    
    result = agent.invoke({"messages": messages})
    
    # Extract structured output from agent response
    # LangChain agents with response_format return structured output in "structured_response" key
    # Fall back to "output" if "structured_response" is not present
    if isinstance(result, dict):
        analysis = result.get("structured_response", result.get("output", result))
    else:
        analysis = result
    
    # If result is already the expected model instance, use it directly
    if isinstance(analysis, response_model):
        return analysis
    else:
        # Try to parse from agent output
        return response_model.model_validate(analysis)


def extract_token_usage(result: Any) -> Dict[str, Any]:
    """Extract token usage information from agent response.
    
    Supports:
    - OpenAI/ChatGPT API: input_tokens, output_tokens in response_metadata
    - OpenRouter API: prompt_tokens, completion_tokens, cost in usage object
    
    Args:
        result: Agent invoke result (dict or object)
    
    Returns:
        Dictionary with: input_tokens, output_tokens, total_tokens, model, cost
    """
    default = {"input_tokens": None, "output_tokens": None, "total_tokens": None, "model": None, "cost": None}
    
    def _get(key, default=None):
        return result.get(key, default) if isinstance(result, dict) else getattr(result, key, default)
    
    def _get_from(obj, key, default=None):
        return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
    
    try:
        # OpenRouter format: has "usage" key/attribute
        usage = _get("usage")
        if usage:
            input_tokens = _get_from(usage, "prompt_tokens")
            output_tokens = _get_from(usage, "completion_tokens")
            total_tokens = _get_from(usage, "total_tokens")
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens or (input_tokens + output_tokens if input_tokens and output_tokens else None),
                "model": _get("model"),
                "cost": _get_from(usage, "cost")
            }
        
        # OpenAI format: has "response_metadata" or "metadata" key/attribute
        metadata = _get("response_metadata") or _get("metadata")
        if metadata:
            if not isinstance(metadata, dict):
                metadata = {
                    "input_tokens": getattr(metadata, "input_tokens", None),
                    "output_tokens": getattr(metadata, "output_tokens", None),
                    "total_tokens": getattr(metadata, "total_tokens", None),
                    "model": getattr(metadata, "model", None) or getattr(metadata, "model_name", None),
                    "token_usage": getattr(metadata, "token_usage", None)
                }
            
            token_usage = metadata.get("token_usage") or metadata
            input_tokens = _get_from(token_usage, "input_tokens")
            output_tokens = _get_from(token_usage, "output_tokens")
            total_tokens = _get_from(token_usage, "total_tokens")
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens or (input_tokens + output_tokens if input_tokens and output_tokens else None),
                "model": metadata.get("model") or _get("model"),
                "cost": None
            }
    except Exception as e:
        logger.debug(f"Could not extract token usage: {e}")
    
    return default


def load_api(dotenv_dir: Optional[Path] = None, raise_on_missing: bool = False) -> None:
    """
    Load OpenAI API key from .env file and validate it exists.
    
    Args:
        dotenv_dir: Directory containing .env file. If None, load_dotenv searches automatically.
        raise_on_missing: If True, raises ValueError when API key is missing.
    """
    load_dotenv(dotenv_dir / '.env' if dotenv_dir else None)
    
    if not os.getenv('OPENAI_API_KEY'):
        if raise_on_missing:
            raise ValueError("OPENAI_API_KEY not found. Set it in .env file or environment.")
        print("⚠️  WARNING: OPENAI_API_KEY not set!")
        print("This notebook will not work without API access.")
    else:
        print("✅ API key found")

