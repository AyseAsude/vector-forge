"""LLM client abstraction layer."""

from typing import Dict, Any

from vector_forge.constants import DEFAULT_MODEL
from vector_forge.core.config import LLMConfig
from vector_forge.llm.base import BaseLLMClient
from vector_forge.llm.litellm_client import LiteLLMClient
from vector_forge.llm.mock_client import MockLLMClient

# Response format for JSON mode - ensures LLM returns valid JSON without markdown
JSON_RESPONSE_FORMAT: Dict[str, Any] = {"type": "json_object"}


def create_client(model: str = DEFAULT_MODEL, **kwargs) -> BaseLLMClient:
    """Create an LLM client for the specified model.

    Args:
        model: Model identifier (litellm format).
        **kwargs: Additional configuration options (temperature, max_tokens, etc).

    Returns:
        Configured LLM client instance.
    """
    config = LLMConfig(model=model, **kwargs)
    return LiteLLMClient(config)


__all__ = [
    "BaseLLMClient",
    "LiteLLMClient",
    "MockLLMClient",
    "create_client",
    "JSON_RESPONSE_FORMAT",
]
