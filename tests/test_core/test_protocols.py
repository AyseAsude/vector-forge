"""Tests for vector_forge.core.protocols module."""

import pytest
from typing import Dict, Any, List

from vector_forge.core.protocols import (
    Message,
    ToolCall,
    ToolDefinition,
    LLMResponse,
    ToolResult,
    Tool,
)


class TestMessage:
    """Tests for Message dataclass."""

    def test_basic_message(self):
        """Test creating a basic message."""
        msg = Message(role="user", content="Hello!")

        assert msg.role == "user"
        assert msg.content == "Hello!"
        assert msg.name is None
        assert msg.tool_call_id is None
        assert msg.tool_calls is None

    def test_system_message(self):
        """Test creating a system message."""
        msg = Message(
            role="system",
            content="You are a helpful assistant.",
        )

        assert msg.role == "system"
        assert msg.content == "You are a helpful assistant."

    def test_tool_message(self):
        """Test creating a tool result message."""
        msg = Message(
            role="tool",
            content="Tool output here",
            name="my_tool",
            tool_call_id="call_123",
        )

        assert msg.role == "tool"
        assert msg.name == "my_tool"
        assert msg.tool_call_id == "call_123"

    def test_assistant_with_tool_calls(self):
        """Test creating assistant message with tool calls."""
        tool_calls = [
            ToolCall(id="call_1", name="tool1", arguments={"a": 1}),
            ToolCall(id="call_2", name="tool2", arguments={"b": 2}),
        ]

        msg = Message(
            role="assistant",
            content=None,
            tool_calls=tool_calls,
        )

        assert msg.role == "assistant"
        assert msg.content is None
        assert len(msg.tool_calls) == 2


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_creation(self):
        """Test creating a tool call."""
        call = ToolCall(
            id="call_abc123",
            name="generate_prompts",
            arguments={"num_prompts": 10, "domains": ["science", "math"]},
        )

        assert call.id == "call_abc123"
        assert call.name == "generate_prompts"
        assert call.arguments["num_prompts"] == 10
        assert call.arguments["domains"] == ["science", "math"]

    def test_empty_arguments(self):
        """Test tool call with empty arguments."""
        call = ToolCall(
            id="call_123",
            name="finalize",
            arguments={},
        )

        assert call.arguments == {}


class TestToolDefinition:
    """Tests for ToolDefinition dataclass."""

    def test_creation(self):
        """Test creating a tool definition."""
        definition = ToolDefinition(
            name="my_tool",
            description="Does something useful",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                },
                "required": ["input"],
            },
        )

        assert definition.name == "my_tool"
        assert definition.description == "Does something useful"
        assert definition.parameters["type"] == "object"
        assert "input" in definition.parameters["properties"]


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_text_response(self):
        """Test LLM response with text content."""
        response = LLMResponse(
            content="This is my response.",
            finish_reason="stop",
        )

        assert response.content == "This is my response."
        assert response.tool_calls == []
        assert response.finish_reason == "stop"

    def test_tool_call_response(self):
        """Test LLM response with tool calls."""
        tool_calls = [
            ToolCall(id="call_1", name="tool1", arguments={}),
        ]

        response = LLMResponse(
            content=None,
            tool_calls=tool_calls,
            finish_reason="tool_calls",
        )

        assert response.content is None
        assert len(response.tool_calls) == 1
        assert response.finish_reason == "tool_calls"

    def test_usage(self):
        """Test LLM response with usage info."""
        response = LLMResponse(
            content="Response",
            usage={"prompt_tokens": 50, "completion_tokens": 100},
        )

        assert response.usage["prompt_tokens"] == 50
        assert response.usage["completion_tokens"] == 100

    def test_default_values(self):
        """Test default values."""
        response = LLMResponse(content="Test")

        assert response.tool_calls == []
        assert response.finish_reason == "stop"
        assert response.usage is None


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_success_result(self):
        """Test successful tool result."""
        result = ToolResult(
            success=True,
            output={"prompts": ["P1", "P2", "P3"]},
        )

        assert result.success is True
        assert result.output["prompts"] == ["P1", "P2", "P3"]
        assert result.error is None

    def test_error_result(self):
        """Test failed tool result."""
        result = ToolResult(
            success=False,
            output=None,
            error="Tool execution failed",
        )

        assert result.success is False
        assert result.output is None
        assert result.error == "Tool execution failed"

    def test_to_string_success(self):
        """Test to_string for successful result."""
        result = ToolResult(success=True, output="Generated 10 prompts")

        assert result.to_string() == "Generated 10 prompts"

    def test_to_string_error(self):
        """Test to_string for failed result."""
        result = ToolResult(success=False, output=None, error="Invalid input")

        assert result.to_string() == "Error: Invalid input"

    def test_to_string_complex_output(self):
        """Test to_string with complex output."""
        result = ToolResult(
            success=True,
            output={"key": "value", "count": 5},
        )

        string_output = result.to_string()
        assert "key" in string_output
        assert "value" in string_output


class TestToolABC:
    """Tests for Tool abstract base class."""

    class ConcreteTool(Tool):
        """Concrete implementation for testing."""

        @property
        def name(self) -> str:
            return "concrete_tool"

        @property
        def description(self) -> str:
            return "A concrete test tool"

        @property
        def parameters(self) -> Dict[str, Any]:
            return {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                },
                "required": ["value"],
            }

        async def execute(self, **kwargs) -> ToolResult:
            value = kwargs.get("value", "")
            return ToolResult(success=True, output=f"Processed: {value}")

    def test_to_definition(self):
        """Test converting tool to definition."""
        tool = self.ConcreteTool()
        definition = tool.to_definition()

        assert isinstance(definition, ToolDefinition)
        assert definition.name == "concrete_tool"
        assert definition.description == "A concrete test tool"
        assert definition.parameters["type"] == "object"

    @pytest.mark.asyncio
    async def test_execute(self):
        """Test tool execution."""
        tool = self.ConcreteTool()
        result = await tool.execute(value="test")

        assert result.success is True
        assert result.output == "Processed: test"
