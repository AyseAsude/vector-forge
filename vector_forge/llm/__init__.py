"""LLM client abstraction layer."""

from vector_forge.llm.base import BaseLLMClient
from vector_forge.llm.litellm_client import LiteLLMClient
from vector_forge.llm.mock_client import MockLLMClient


def create_client(model: str = "gpt-4o", **kwargs) -> BaseLLMClient:
    """Create an LLM client for the specified model.

    Args:
        model: Model identifier (litellm format).
        **kwargs: Additional configuration options.

    Returns:
        Configured LLM client instance.
    """
    return LiteLLMClient(model=model, **kwargs)


__all__ = [
    "BaseLLMClient",
    "LiteLLMClient",
    "MockLLMClient",
    "create_client",
]
