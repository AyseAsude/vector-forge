"""Tests for protocol compliance.

Verifies that implementations correctly satisfy protocol contracts defined in
vector_forge.core.protocols.
"""

import pytest
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import torch

from vector_forge.core.protocols import (
    # Protocols
    LLMClient,
    DatapointStrategy,
    NoiseReducer,
    LayerSearchStrategy,
    VectorEvaluator,
    Tool,
    EventEmitter,
    # Data classes
    Message,
    ToolCall,
    ToolDefinition,
    LLMResponse,
    ToolResult,
    Event,
)
from vector_forge.core.behavior import BehaviorSpec


# =============================================================================
# Mock Implementations for Protocol Testing
# =============================================================================


class MockLLMClient:
    """Mock LLM client that satisfies LLMClient protocol."""

    async def complete(
        self,
        messages: List[Message],
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content="Mock response")

    async def complete_with_tools(
        self,
        messages: List[Message],
        tools: List[ToolDefinition],
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content="Mock response with tools")


class MockDatapointStrategy:
    """Mock datapoint strategy that satisfies DatapointStrategy protocol."""

    async def generate(
        self,
        behavior: BehaviorSpec,
        prompts: List[str],
        llm_client: LLMClient,
        **kwargs: Any,
    ) -> List[Any]:
        return []


class MockNoiseReducer:
    """Mock noise reducer that satisfies NoiseReducer protocol."""

    def reduce(
        self,
        vectors: List[torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        if not vectors:
            raise ValueError("Empty vectors list")
        return vectors[0]


class MockLayerSearchStrategy:
    """Mock layer search strategy that satisfies LayerSearchStrategy protocol."""

    def get_layers_to_try(
        self,
        total_layers: int,
        iteration: int = 0,
        previous_results: Optional[Dict[int, float]] = None,
    ) -> List[int]:
        return [total_layers // 2]


class MockVectorEvaluator:
    """Mock vector evaluator that satisfies VectorEvaluator protocol."""

    async def evaluate(
        self,
        vector: torch.Tensor,
        layer: int,
        behavior: BehaviorSpec,
        test_prompts: List[str],
        model_backend: Any,
        **kwargs: Any,
    ) -> Any:
        return {"score": 0.5}


class MockTool(Tool):
    """Mock tool that satisfies Tool abstract base class."""

    @property
    def name(self) -> str:
        return "mock_tool"

    @property
    def description(self) -> str:
        return "A mock tool for testing"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string"}
            },
            "required": ["input"]
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="executed")


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestLLMClientProtocol:
    """Tests for LLMClient protocol compliance."""

    def test_mock_satisfies_protocol(self):
        """Test that MockLLMClient satisfies LLMClient protocol."""
        client = MockLLMClient()
        assert isinstance(client, LLMClient)

    def test_protocol_is_runtime_checkable(self):
        """Test that LLMClient is runtime_checkable."""
        # Non-compliant class should not satisfy protocol
        class NotAnLLMClient:
            pass

        assert not isinstance(NotAnLLMClient(), LLMClient)

    @pytest.mark.asyncio
    async def test_complete_returns_llm_response(self):
        """Test that complete returns LLMResponse."""
        client = MockLLMClient()
        messages = [Message(role="user", content="Hello")]
        response = await client.complete(messages)

        assert isinstance(response, LLMResponse)
        assert response.content is not None

    @pytest.mark.asyncio
    async def test_complete_with_tools_returns_llm_response(self):
        """Test that complete_with_tools returns LLMResponse."""
        client = MockLLMClient()
        messages = [Message(role="user", content="Hello")]
        tools = [ToolDefinition(name="test", description="test", parameters={})]
        response = await client.complete_with_tools(messages, tools)

        assert isinstance(response, LLMResponse)


class TestDatapointStrategyProtocol:
    """Tests for DatapointStrategy protocol compliance."""

    def test_mock_satisfies_protocol(self):
        """Test that MockDatapointStrategy satisfies DatapointStrategy protocol."""
        strategy = MockDatapointStrategy()
        assert isinstance(strategy, DatapointStrategy)

    def test_protocol_is_runtime_checkable(self):
        """Test that DatapointStrategy is runtime_checkable."""
        class NotAStrategy:
            pass

        assert not isinstance(NotAStrategy(), DatapointStrategy)


class TestNoiseReducerProtocol:
    """Tests for NoiseReducer protocol compliance."""

    def test_mock_satisfies_protocol(self):
        """Test that MockNoiseReducer satisfies NoiseReducer protocol."""
        reducer = MockNoiseReducer()
        assert isinstance(reducer, NoiseReducer)

    def test_reduce_returns_tensor(self):
        """Test that reduce returns a tensor."""
        reducer = MockNoiseReducer()
        vectors = [torch.randn(10) for _ in range(3)]
        result = reducer.reduce(vectors)

        assert isinstance(result, torch.Tensor)

    def test_reduce_empty_raises(self):
        """Test that reduce with empty list raises."""
        reducer = MockNoiseReducer()
        with pytest.raises(ValueError):
            reducer.reduce([])


class TestLayerSearchStrategyProtocol:
    """Tests for LayerSearchStrategy protocol compliance."""

    def test_mock_satisfies_protocol(self):
        """Test that MockLayerSearchStrategy satisfies LayerSearchStrategy protocol."""
        strategy = MockLayerSearchStrategy()
        assert isinstance(strategy, LayerSearchStrategy)

    def test_get_layers_returns_list(self):
        """Test that get_layers_to_try returns a list."""
        strategy = MockLayerSearchStrategy()
        layers = strategy.get_layers_to_try(total_layers=32)

        assert isinstance(layers, list)
        assert all(isinstance(l, int) for l in layers)


class TestVectorEvaluatorProtocol:
    """Tests for VectorEvaluator protocol compliance."""

    def test_mock_satisfies_protocol(self):
        """Test that MockVectorEvaluator satisfies VectorEvaluator protocol."""
        evaluator = MockVectorEvaluator()
        assert isinstance(evaluator, VectorEvaluator)


class TestToolAbstractClass:
    """Tests for Tool abstract base class."""

    def test_mock_tool_satisfies_abc(self):
        """Test that MockTool properly implements Tool ABC."""
        tool = MockTool()
        assert isinstance(tool, Tool)

    def test_tool_has_required_properties(self):
        """Test that tool has all required properties."""
        tool = MockTool()
        assert tool.name == "mock_tool"
        assert tool.description == "A mock tool for testing"
        assert isinstance(tool.parameters, dict)

    def test_tool_to_definition(self):
        """Test that to_definition creates proper ToolDefinition."""
        tool = MockTool()
        definition = tool.to_definition()

        assert isinstance(definition, ToolDefinition)
        assert definition.name == tool.name
        assert definition.description == tool.description
        assert definition.parameters == tool.parameters

    @pytest.mark.asyncio
    async def test_tool_execute_returns_tool_result(self):
        """Test that execute returns ToolResult."""
        tool = MockTool()
        result = await tool.execute(input="test")

        assert isinstance(result, ToolResult)
        assert result.success is True


# =============================================================================
# Data Class Type Tests
# =============================================================================


class TestMessageDataclass:
    """Tests for Message dataclass types."""

    def test_required_fields(self):
        """Test that role and content are required."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_optional_fields_default_none(self):
        """Test that optional fields default to None."""
        msg = Message(role="user", content="Hello")
        assert msg.name is None
        assert msg.tool_call_id is None
        assert msg.tool_calls is None

    def test_all_fields(self):
        """Test message with all fields."""
        tool_call = ToolCall(id="1", name="test", arguments={})
        msg = Message(
            role="assistant",
            content="Response",
            name="test",
            tool_call_id="123",
            tool_calls=[tool_call]
        )

        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1


class TestToolCallDataclass:
    """Tests for ToolCall dataclass types."""

    def test_required_fields(self):
        """Test all fields are required."""
        tc = ToolCall(id="1", name="test_tool", arguments={"key": "value"})
        assert tc.id == "1"
        assert tc.name == "test_tool"
        assert tc.arguments == {"key": "value"}

    def test_arguments_accepts_any_dict(self):
        """Test that arguments accepts various dict types."""
        tc = ToolCall(
            id="1",
            name="test",
            arguments={
                "string": "value",
                "int": 42,
                "float": 3.14,
                "bool": True,
                "list": [1, 2, 3],
                "nested": {"key": "value"}
            }
        )
        assert isinstance(tc.arguments, dict)


class TestLLMResponseDataclass:
    """Tests for LLMResponse dataclass types."""

    def test_minimal_response(self):
        """Test response with just content."""
        response = LLMResponse(content="Hello")
        assert response.content == "Hello"
        assert response.tool_calls == []
        assert response.finish_reason == "stop"
        assert response.usage is None

    def test_response_with_tool_calls(self):
        """Test response with tool calls."""
        tool_call = ToolCall(id="1", name="test", arguments={})
        response = LLMResponse(
            content=None,
            tool_calls=[tool_call],
            finish_reason="tool_calls"
        )

        assert response.content is None
        assert len(response.tool_calls) == 1
        assert response.finish_reason == "tool_calls"

    def test_response_with_usage(self):
        """Test response with usage statistics."""
        response = LLMResponse(
            content="Hello",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )

        assert response.usage is not None
        assert response.usage["total_tokens"] == 15


class TestToolResultDataclass:
    """Tests for ToolResult dataclass types."""

    def test_success_result(self):
        """Test successful tool result."""
        result = ToolResult(success=True, output="Result data")
        assert result.success is True
        assert result.output == "Result data"
        assert result.error is None

    def test_failure_result(self):
        """Test failed tool result."""
        result = ToolResult(success=False, output=None, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_to_string_success(self):
        """Test to_string for successful result."""
        result = ToolResult(success=True, output={"key": "value"})
        assert result.to_string() == "{'key': 'value'}"

    def test_to_string_failure(self):
        """Test to_string for failed result."""
        result = ToolResult(success=False, output=None, error="Failed")
        assert result.to_string() == "Error: Failed"


class TestEventDataclass:
    """Tests for Event dataclass types."""

    def test_required_fields(self):
        """Test required type field."""
        event = Event(type="test_event")
        assert event.type == "test_event"
        assert event.data == {}
        assert event.timestamp is None

    def test_with_data(self):
        """Test event with data."""
        event = Event(type="test", data={"key": "value"})
        assert event.data["key"] == "value"

    def test_with_timestamp(self):
        """Test event with timestamp."""
        event = Event(type="test", timestamp=1234567890.0)
        assert event.timestamp == 1234567890.0


# =============================================================================
# EventEmitter Tests
# =============================================================================


class ConcreteEventEmitter(EventEmitter):
    """Concrete implementation of EventEmitter for testing."""
    pass


class TestEventEmitter:
    """Tests for EventEmitter base class."""

    def test_subscribe_and_emit(self):
        """Test subscribing to events and receiving them."""
        emitter = ConcreteEventEmitter()
        received_events = []

        def handler(event: Event):
            received_events.append(event)

        emitter.on("test", handler)
        emitter.emit(Event(type="test", data={"value": 1}))

        assert len(received_events) == 1
        assert received_events[0].type == "test"
        assert received_events[0].data["value"] == 1

    def test_wildcard_subscription(self):
        """Test subscribing to all events with '*'."""
        emitter = ConcreteEventEmitter()
        received_events = []

        def handler(event: Event):
            received_events.append(event)

        emitter.on("*", handler)
        emitter.emit(Event(type="event1"))
        emitter.emit(Event(type="event2"))

        assert len(received_events) == 2

    def test_unsubscribe(self):
        """Test unsubscribing from events."""
        emitter = ConcreteEventEmitter()
        received_events = []

        def handler(event: Event):
            received_events.append(event)

        emitter.on("test", handler)
        emitter.emit(Event(type="test"))
        emitter.off("test", handler)
        emitter.emit(Event(type="test"))

        assert len(received_events) == 1

    def test_emit_sets_timestamp(self):
        """Test that emit sets timestamp if not provided."""
        emitter = ConcreteEventEmitter()
        received_events = []

        def handler(event: Event):
            received_events.append(event)

        emitter.on("test", handler)
        emitter.emit(Event(type="test"))

        assert received_events[0].timestamp is not None

    def test_emit_preserves_timestamp(self):
        """Test that emit preserves existing timestamp."""
        emitter = ConcreteEventEmitter()
        received_events = []

        def handler(event: Event):
            received_events.append(event)

        emitter.on("test", handler)
        emitter.emit(Event(type="test", timestamp=12345.0))

        assert received_events[0].timestamp == 12345.0
