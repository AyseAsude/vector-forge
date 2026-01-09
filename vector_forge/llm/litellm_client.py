"""LiteLLM-based LLM client implementation."""

import json
from typing import List, Any, Optional

import litellm

from vector_forge.core.protocols import (
    Message,
    ToolDefinition,
    ToolCall,
    LLMResponse,
)
from vector_forge.core.config import LLMConfig
from vector_forge.llm.base import BaseLLMClient


class LiteLLMClient(BaseLLMClient):
    """
    LLM client using litellm for multi-provider support.

    Supports OpenAI, Anthropic, Cohere, Azure, Ollama, and many more providers
    through litellm's unified interface.

    Example:
        >>> config = LLMConfig(model="gpt-4o", temperature=0.7)
        >>> client = LiteLLMClient(config)
        >>> response = await client.complete([Message(role="user", content="Hello")])
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)

        # Configure litellm
        if config.api_key:
            # Set API key based on model provider
            if "gpt" in config.model or "openai" in config.model:
                litellm.openai_key = config.api_key
            elif "claude" in config.model or "anthropic" in config.model:
                litellm.anthropic_key = config.api_key

    async def complete(
        self,
        messages: List[Message],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion from the LLM.

        Args:
            messages: List of chat messages.
            **kwargs: Additional arguments passed to litellm.

        Returns:
            LLMResponse with content.
        """
        merged_kwargs = self._merge_kwargs(**kwargs)
        messages_dict = self._messages_to_dicts(messages)

        response = await litellm.acompletion(
            model=self.config.model,
            messages=messages_dict,
            api_base=self.config.api_base,
            **merged_kwargs,
        )

        self._track_usage(response.usage._asdict() if response.usage else None)

        return LLMResponse(
            content=response.choices[0].message.content,
            finish_reason=response.choices[0].finish_reason or "stop",
            usage=response.usage._asdict() if response.usage else None,
        )

    async def complete_with_tools(
        self,
        messages: List[Message],
        tools: List[ToolDefinition],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion with tool use capability.

        Args:
            messages: List of chat messages.
            tools: Available tools the LLM can call.
            **kwargs: Additional arguments passed to litellm.

        Returns:
            LLMResponse with content and/or tool calls.
        """
        merged_kwargs = self._merge_kwargs(**kwargs)
        messages_dict = self._messages_to_dicts(messages)
        tools_dict = self._tools_to_dicts(tools)

        response = await litellm.acompletion(
            model=self.config.model,
            messages=messages_dict,
            tools=tools_dict,
            api_base=self.config.api_base,
            **merged_kwargs,
        )

        self._track_usage(response.usage._asdict() if response.usage else None)

        message = response.choices[0].message
        tool_calls = []

        if message.tool_calls:
            for tc in message.tool_calls:
                # Parse arguments from JSON string if needed
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}

                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=response.choices[0].finish_reason or "stop",
            usage=response.usage._asdict() if response.usage else None,
        )
