"""LLM client abstraction layer."""

from vector_forge.constants import DEFAULT_MODEL
from vector_forge.core.config import LLMConfig
from vector_forge.llm.base import BaseLLMClient
from vector_forge.llm.litellm_client import LiteLLMClient
from vector_forge.llm.mock_client import MockLLMClient


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
]
