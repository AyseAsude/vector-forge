"""LiteLLM-based LLM client implementation."""

import json
import time
import uuid
from typing import List, Any, Optional, TYPE_CHECKING

import litellm

from vector_forge.core.protocols import (
    Message,
    ToolDefinition,
    ToolCall,
    LLMResponse,
)
from vector_forge.core.config import LLMConfig
from vector_forge.llm.base import BaseLLMClient

if TYPE_CHECKING:
    from vector_forge.storage import SessionStore


class LiteLLMClient(BaseLLMClient):
    """
    LLM client using litellm for multi-provider support.

    Supports OpenAI, Anthropic, Cohere, Azure, Ollama, and many more providers
    through litellm's unified interface.

    Captures all LLM requests and responses to the session store for
    complete reproducibility.

    Example:
        >>> config = LLMConfig(model="gpt-4o", temperature=0.7)
        >>> client = LiteLLMClient(config)
        >>> response = await client.complete([Message(role="user", content="Hello")])
    """

    def __init__(
        self,
        config: LLMConfig,
        store: Optional["SessionStore"] = None,
    ):
        super().__init__(config, store)

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

        # Generate request ID for linking request/response
        request_id = str(uuid.uuid4())

        # Capture request event
        self._emit_request_event(
            request_id=request_id,
            messages=messages_dict,
            tools=None,
            kwargs=merged_kwargs,
        )

        start_time = time.time()
        error_msg = None

        try:
            response = await litellm.acompletion(
                model=self.config.model,
                messages=messages_dict,
                api_base=self.config.api_base,
                **merged_kwargs,
            )

            self._track_usage(response.usage._asdict() if response.usage else None)

            latency_ms = (time.time() - start_time) * 1000

            # Capture response event
            self._emit_response_event(
                request_id=request_id,
                content=response.choices[0].message.content,
                tool_calls=[],
                finish_reason=response.choices[0].finish_reason or "stop",
                usage=response.usage._asdict() if response.usage else None,
                latency_ms=latency_ms,
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                finish_reason=response.choices[0].finish_reason or "stop",
                usage=response.usage._asdict() if response.usage else None,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = str(e)

            # Capture error response event
            self._emit_response_event(
                request_id=request_id,
                content=None,
                tool_calls=[],
                finish_reason="error",
                usage=None,
                latency_ms=latency_ms,
                error=error_msg,
            )

            raise

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

        # Generate request ID for linking request/response
        request_id = str(uuid.uuid4())

        # Capture request event
        self._emit_request_event(
            request_id=request_id,
            messages=messages_dict,
            tools=tools_dict,
            kwargs=merged_kwargs,
        )

        start_time = time.time()

        try:
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
            tool_calls_dicts = []

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

                    # Keep dict form for event capture
                    tool_calls_dicts.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": args,
                    })

            latency_ms = (time.time() - start_time) * 1000

            # Capture response event
            self._emit_response_event(
                request_id=request_id,
                content=message.content,
                tool_calls=tool_calls_dicts,
                finish_reason=response.choices[0].finish_reason or "stop",
                usage=response.usage._asdict() if response.usage else None,
                latency_ms=latency_ms,
            )

            return LLMResponse(
                content=message.content,
                tool_calls=tool_calls,
                finish_reason=response.choices[0].finish_reason or "stop",
                usage=response.usage._asdict() if response.usage else None,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            # Capture error response event
            self._emit_response_event(
                request_id=request_id,
                content=None,
                tool_calls=[],
                finish_reason="error",
                usage=None,
                latency_ms=latency_ms,
                error=str(e),
            )

            raise

    def _emit_request_event(
        self,
        request_id: str,
        messages: List[dict],
        tools: Optional[List[dict]],
        kwargs: dict,
    ) -> None:
        """Emit LLM request event to store."""
        if self._store is None:
            return

        from vector_forge.storage import LLMRequestEvent

        event = LLMRequestEvent(
            request_id=request_id,
            model=self.config.model,
            messages=messages,
            tools=tools,
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens"),
            extra_params={
                k: v for k, v in kwargs.items()
                if k not in ("temperature", "max_tokens")
            },
        )

        self._store.append_event(event, source="llm")

    def _emit_response_event(
        self,
        request_id: str,
        content: Optional[str],
        tool_calls: List[dict],
        finish_reason: str,
        usage: Optional[dict],
        latency_ms: float,
        error: Optional[str] = None,
    ) -> None:
        """Emit LLM response event to store."""
        if self._store is None:
            return

        from vector_forge.storage import LLMResponseEvent

        event = LLMResponseEvent(
            request_id=request_id,
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            latency_ms=latency_ms,
            error=error,
        )

        self._store.append_event(event, source="llm")
