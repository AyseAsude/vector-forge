"""Mock LLM client for testing."""

from typing import List, Any, Optional, Callable, Union

from vector_forge.core.protocols import (
    Message,
    ToolDefinition,
    ToolCall,
    LLMResponse,
)
from vector_forge.core.config import LLMConfig
from vector_forge.llm.base import BaseLLMClient


ResponseGenerator = Callable[[List[Message]], Union[str, LLMResponse]]


class MockLLMClient(BaseLLMClient):
    """
    Mock LLM client for testing.

    Can be configured with predefined responses or a response generator function.

    Example:
        >>> client = MockLLMClient.with_responses(["Hello!", "How can I help?"])
        >>> response = await client.complete([Message(role="user", content="Hi")])
        >>> assert response.content == "Hello!"

        >>> def generate(messages):
        ...     return f"You said: {messages[-1].content}"
        >>> client = MockLLMClient.with_generator(generate)
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        responses: Optional[List[Union[str, LLMResponse]]] = None,
        response_generator: Optional[ResponseGenerator] = None,
    ):
        super().__init__(config or LLMConfig(model="mock"))
        self._responses = responses or []
        self._response_index = 0
        self._response_generator = response_generator
        self._call_history: List[List[Message]] = []

    @classmethod
    def with_responses(cls, responses: List[Union[str, LLMResponse]]) -> "MockLLMClient":
        """Create a mock client with predefined responses."""
        return cls(responses=responses)

    @classmethod
    def with_generator(cls, generator: ResponseGenerator) -> "MockLLMClient":
        """Create a mock client with a response generator function."""
        return cls(response_generator=generator)

    @property
    def call_history(self) -> List[List[Message]]:
        """Get history of all calls made to this client."""
        return self._call_history

    def _get_next_response(self, messages: List[Message]) -> LLMResponse:
        """Get the next response based on configuration."""
        self._call_history.append(messages)

        # Use generator if provided
        if self._response_generator:
            result = self._response_generator(messages)
            if isinstance(result, LLMResponse):
                return result
            return LLMResponse(content=result)

        # Use predefined responses
        if self._response_index < len(self._responses):
            response = self._responses[self._response_index]
            self._response_index += 1
            if isinstance(response, LLMResponse):
                return response
            return LLMResponse(content=response)

        # Default response
        return LLMResponse(content="Mock response")

    async def complete(
        self,
        messages: List[Message],
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a mock completion."""
        return self._get_next_response(messages)

    async def complete_with_tools(
        self,
        messages: List[Message],
        tools: List[ToolDefinition],
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a mock completion with tool support."""
        return self._get_next_response(messages)

    def reset(self) -> None:
        """Reset response index and call history."""
        self._response_index = 0
        self._call_history.clear()
