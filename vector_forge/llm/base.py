"""Base class for LLM clients."""

from abc import ABC, abstractmethod
from typing import List, Any, Optional, TYPE_CHECKING

from vector_forge.core.protocols import (
    Message,
    ToolDefinition,
    LLMResponse,
    EventEmitter,
    Event,
)
from vector_forge.core.config import LLMConfig
from vector_forge.core.events import EventType, create_event

if TYPE_CHECKING:
    from vector_forge.storage import SessionStore


class BaseLLMClient(EventEmitter, ABC):
    """
    Abstract base class for LLM clients.

    Provides common functionality and event emission.
    Subclasses implement the actual API calls.

    Args:
        config: LLM configuration.
        store: Optional session store for event capture.
    """

    def __init__(
        self,
        config: LLMConfig,
        store: Optional["SessionStore"] = None,
    ):
        super().__init__()
        self.config = config
        self._store = store
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

    async def generate(
        self,
        messages: List[dict],
        **kwargs: Any,
    ) -> str:
        """Generate a text response from dict messages.

        Convenience method that wraps complete() for simpler use cases.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            **kwargs: Additional arguments (model, temperature, etc).

        Returns:
            Generated text content.
        """
        # Convert dicts to Message objects
        msg_objects = [
            Message(role=m["role"], content=m["content"])
            for m in messages
        ]
        response = await self.complete(msg_objects, **kwargs)
        return response.content or ""

    def _merge_kwargs(self, **kwargs: Any) -> dict:
        """Merge provided kwargs with config defaults.

        Filters out 'model' and 'api_base' since those are passed
        explicitly to litellm.acompletion().
        """
        merged = {
            "temperature": self.config.temperature,
            **self.config.extra_params,
            **kwargs,
        }
        # Only include max_tokens if explicitly set (not None)
        if self.config.max_tokens is not None:
            merged.setdefault("max_tokens", self.config.max_tokens)
        # Remove max_tokens if it was explicitly set to None in kwargs
        if merged.get("max_tokens") is None:
            merged.pop("max_tokens", None)
        # Remove keys that are passed explicitly to litellm
        merged.pop("model", None)
        merged.pop("api_base", None)
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
