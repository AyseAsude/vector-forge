"""Tests for vector_forge.llm.mock_client module."""

import pytest
from typing import List

from vector_forge.llm.mock_client import MockLLMClient
from vector_forge.core.protocols import Message, ToolDefinition, ToolCall, LLMResponse
from vector_forge.core.config import LLMConfig


class TestMockLLMClientBasic:
    """Basic tests for MockLLMClient."""

    def test_default_creation(self):
        """Test creating mock client with defaults."""
        client = MockLLMClient()

        assert client._response_index == 0
        assert client._call_history == []

    def test_with_responses_factory(self):
        """Test with_responses factory method."""
        client = MockLLMClient.with_responses([
            "Response 1",
            "Response 2",
        ])

        assert len(client._responses) == 2

    def test_with_generator_factory(self):
        """Test with_generator factory method."""
        def gen(messages):
            return "Generated response"

        client = MockLLMClient.with_generator(gen)

        assert client._response_generator is not None

    @pytest.mark.asyncio
    async def test_complete_with_string_responses(self):
        """Test complete with string responses."""
        client = MockLLMClient.with_responses(["Hello!", "World!"])

        response1 = await client.complete([Message(role="user", content="Hi")])
        response2 = await client.complete([Message(role="user", content="Test")])

        assert response1.content == "Hello!"
        assert response2.content == "World!"

    @pytest.mark.asyncio
    async def test_complete_with_llm_response(self):
        """Test complete with LLMResponse objects."""
        custom_response = LLMResponse(
            content="Custom",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
        client = MockLLMClient.with_responses([custom_response])

        response = await client.complete([Message(role="user", content="Test")])

        assert response.content == "Custom"
        assert response.usage["prompt_tokens"] == 10

    @pytest.mark.asyncio
    async def test_complete_exhausted_responses(self):
        """Test complete when responses are exhausted."""
        client = MockLLMClient.with_responses(["Only one"])

        await client.complete([Message(role="user", content="First")])
        response2 = await client.complete([Message(role="user", content="Second")])

        # Should return default response
        assert response2.content == "Mock response"

    @pytest.mark.asyncio
    async def test_complete_with_generator(self):
        """Test complete with generator function."""
        def gen(messages: List[Message]) -> str:
            return f"Echo: {messages[-1].content}"

        client = MockLLMClient.with_generator(gen)

        response = await client.complete([Message(role="user", content="Test message")])

        assert response.content == "Echo: Test message"

    @pytest.mark.asyncio
    async def test_generator_returns_llm_response(self):
        """Test generator that returns LLMResponse."""
        def gen(messages: List[Message]) -> LLMResponse:
            return LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="1", name="test", arguments={})],
            )

        client = MockLLMClient.with_generator(gen)

        response = await client.complete([Message(role="user", content="Test")])

        assert response.content is None
        assert len(response.tool_calls) == 1


class TestMockLLMClientWithTools:
    """Tests for complete_with_tools method."""

    @pytest.mark.asyncio
    async def test_complete_with_tools(self):
        """Test complete_with_tools uses same response logic."""
        client = MockLLMClient.with_responses(["Tool response"])
        tools = [
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object", "properties": {}},
            )
        ]

        response = await client.complete_with_tools(
            [Message(role="user", content="Use tool")],
            tools=tools,
        )

        assert response.content == "Tool response"

    @pytest.mark.asyncio
    async def test_complete_with_tools_returns_tool_calls(self):
        """Test complete_with_tools can return tool calls."""
        def gen(messages):
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="call_1", name="generate", arguments={"n": 5}),
                ],
            )

        client = MockLLMClient.with_generator(gen)
        tools = [
            ToolDefinition(
                name="generate",
                description="Generate items",
                parameters={"type": "object", "properties": {"n": {"type": "integer"}}},
            )
        ]

        response = await client.complete_with_tools(
            [Message(role="user", content="Generate")],
            tools=tools,
        )

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "generate"
        assert response.tool_calls[0].arguments["n"] == 5


class TestMockLLMClientCallHistory:
    """Tests for call history tracking."""

    @pytest.mark.asyncio
    async def test_call_history_recorded(self):
        """Test that calls are recorded in history."""
        client = MockLLMClient.with_responses(["R1", "R2"])

        messages1 = [Message(role="user", content="First")]
        messages2 = [Message(role="user", content="Second")]

        await client.complete(messages1)
        await client.complete(messages2)

        assert len(client.call_history) == 2
        assert client.call_history[0] == messages1
        assert client.call_history[1] == messages2

    @pytest.mark.asyncio
    async def test_call_history_includes_all_messages(self):
        """Test that full conversation is in history."""
        client = MockLLMClient.with_responses(["Response"])

        messages = [
            Message(role="system", content="System prompt"),
            Message(role="user", content="User message"),
            Message(role="assistant", content="Previous response"),
            Message(role="user", content="Follow up"),
        ]

        await client.complete(messages)

        assert len(client.call_history[0]) == 4


class TestMockLLMClientReset:
    """Tests for reset functionality."""

    @pytest.mark.asyncio
    async def test_reset_clears_history(self):
        """Test reset clears call history."""
        client = MockLLMClient.with_responses(["R1", "R2"])

        await client.complete([Message(role="user", content="Test")])
        assert len(client.call_history) == 1

        client.reset()

        assert len(client.call_history) == 0

    @pytest.mark.asyncio
    async def test_reset_resets_response_index(self):
        """Test reset resets response index."""
        client = MockLLMClient.with_responses(["First", "Second"])

        response1 = await client.complete([Message(role="user", content="1")])
        assert response1.content == "First"

        client.reset()

        response2 = await client.complete([Message(role="user", content="2")])
        assert response2.content == "First"  # Back to first response


class TestMockLLMClientConfig:
    """Tests for configuration handling."""

    def test_custom_config(self):
        """Test creating client with custom config."""
        config = LLMConfig(model="custom-model", temperature=0.5)
        client = MockLLMClient(config=config)

        assert client.config.model == "custom-model"
        assert client.config.temperature == 0.5

    def test_default_config(self):
        """Test default config when none provided."""
        client = MockLLMClient()

        assert client.config.model == "mock"


class TestMockLLMClientAdvanced:
    """Advanced tests for MockLLMClient."""

    @pytest.mark.asyncio
    async def test_generator_with_context(self):
        """Test generator that uses message context."""
        def context_aware_gen(messages: List[Message]) -> str:
            # Count user messages
            user_count = sum(1 for m in messages if m.role == "user")
            return f"User message count: {user_count}"

        client = MockLLMClient.with_generator(context_aware_gen)

        messages = [
            Message(role="user", content="First"),
            Message(role="assistant", content="Response"),
            Message(role="user", content="Second"),
        ]

        response = await client.complete(messages)

        assert response.content == "User message count: 2"

    @pytest.mark.asyncio
    async def test_mixed_responses(self):
        """Test mixing string and LLMResponse responses."""
        responses = [
            "Simple string",
            LLMResponse(content="Complex", usage={"tokens": 50}),
            "Another string",
        ]
        client = MockLLMClient.with_responses(responses)

        r1 = await client.complete([Message(role="user", content="1")])
        r2 = await client.complete([Message(role="user", content="2")])
        r3 = await client.complete([Message(role="user", content="3")])

        assert r1.content == "Simple string"
        assert r2.content == "Complex"
        assert r2.usage["tokens"] == 50
        assert r3.content == "Another string"

    @pytest.mark.asyncio
    async def test_generator_exception_handling(self):
        """Test that generator exceptions propagate."""
        def failing_gen(messages):
            raise ValueError("Generator failed")

        client = MockLLMClient.with_generator(failing_gen)

        with pytest.raises(ValueError, match="Generator failed"):
            await client.complete([Message(role="user", content="Test")])
