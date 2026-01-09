"""Base class for LLM clients."""

from abc import ABC, abstractmethod
from typing import List, Any, Optional

from vector_forge.core.protocols import (
    Message,
    ToolDefinition,
    LLMResponse,
    EventEmitter,
    Event,
)
from vector_forge.core.config import LLMConfig
from vector_forge.core.events import EventType, create_event


class BaseLLMClient(EventEmitter, ABC):
    """
    Abstract base class for LLM clients.

    Provides common functionality and event emission.
    Subclasses implement the actual API calls.
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        self._total_tokens_used = 0

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def total_tokens_used(self) -> int:
        return self._total_tokens_used

    @abstractmethod
    async def complete(
        self,
        messages: List[Message],
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion."""
        ...

    @abstractmethod
    async def complete_with_tools(
        self,
        messages: List[Message],
        tools: List[ToolDefinition],
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion with tool use."""
        ...

    def _merge_kwargs(self, **kwargs: Any) -> dict:
        """Merge provided kwargs with config defaults."""
        merged = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **self.config.extra_params,
            **kwargs,
        }
        return merged

    def _track_usage(self, usage: Optional[dict]) -> None:
        """Track token usage from response."""
        if usage:
            self._total_tokens_used += usage.get("total_tokens", 0)

    def _messages_to_dicts(self, messages: List[Message]) -> List[dict]:
        """Convert Message objects to dicts for API calls."""
        result = []
        for msg in messages:
            d = {"role": msg.role, "content": msg.content}
            if msg.name:
                d["name"] = msg.name
            if msg.tool_call_id:
                d["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments if isinstance(tc.arguments, str) else str(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            result.append(d)
        return result

    def _tools_to_dicts(self, tools: List[ToolDefinition]) -> List[dict]:
        """Convert ToolDefinition objects to dicts for API calls."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]
