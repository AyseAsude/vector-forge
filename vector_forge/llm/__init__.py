"""LLM client abstraction layer."""

from typing import Dict, Any, Union, overload

from vector_forge.constants import DEFAULT_MODEL
from vector_forge.core.config import LLMConfig
from vector_forge.llm.base import BaseLLMClient
from vector_forge.llm.litellm_client import LiteLLMClient
from vector_forge.llm.mock_client import MockLLMClient

# Response format for JSON mode - ensures LLM returns valid JSON without markdown
JSON_RESPONSE_FORMAT: Dict[str, Any] = {"type": "json_object"}


def create_client(
    model_or_config: Union[str, LLMConfig] = DEFAULT_MODEL,
    **kwargs,
) -> BaseLLMClient:
    """Create an LLM client from a model string or LLMConfig.

    Args:
        model_or_config: Either a model identifier string (litellm format)
            or a complete LLMConfig object with api_base/api_key.
        **kwargs: Additional configuration options when using string model.
            Ignored when LLMConfig is passed.

    Returns:
        Configured LLM client instance.

    Examples:
        >>> # From model string (uses env vars for api_key)
        >>> client = create_client("openai/gpt-4")

        >>> # From LLMConfig (with custom api_base and api_key)
        >>> config = LLMConfig(model="openai/GLM-4.7", api_base="http://...", api_key="...")
        >>> client = create_client(config)
    """
    if isinstance(model_or_config, LLMConfig):
        return LiteLLMClient(model_or_config)
    else:
        config = LLMConfig(model=model_or_config, **kwargs)
        return LiteLLMClient(config)


__all__ = [
    "BaseLLMClient",
    "LiteLLMClient",
    "MockLLMClient",
    "create_client",
    "JSON_RESPONSE_FORMAT",
]
