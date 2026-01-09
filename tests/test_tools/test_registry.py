"""Tests for vector_forge.tools.registry module."""

import pytest
from typing import Dict, Any

from vector_forge.tools.registry import ToolRegistry
from vector_forge.tools.base import BaseTool
from vector_forge.core.protocols import ToolResult, ToolDefinition


class SampleTool(BaseTool):
    """Sample tool for testing."""

    def __init__(self, name: str = "sample_tool"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "A sample tool for testing"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "value": {"type": "string"},
            },
            "required": ["value"],
        }

    async def _execute(self, value: str) -> str:
        return f"Processed: {value}"


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_empty_registry(self):
        """Test empty registry."""
        registry = ToolRegistry()

        assert len(registry) == 0
        assert registry.tools == []
        assert registry.tool_names == []

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        tool = SampleTool()

        registry.register(tool)

        assert len(registry) == 1
        assert "sample_tool" in registry
        assert registry.has("sample_tool")

    def test_register_duplicate_raises(self):
        """Test registering duplicate tool raises error."""
        registry = ToolRegistry()
        tool1 = SampleTool()
        tool2 = SampleTool()

        registry.register(tool1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(tool2)

    def test_register_multiple_tools(self):
        """Test registering multiple tools."""
        registry = ToolRegistry()
        tool1 = SampleTool("tool_1")
        tool2 = SampleTool("tool_2")
        tool3 = SampleTool("tool_3")

        registry.register(tool1)
        registry.register(tool2)
        registry.register(tool3)

        assert len(registry) == 3
        assert set(registry.tool_names) == {"tool_1", "tool_2", "tool_3"}

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()
        tool = SampleTool()

        registry.register(tool)
        result = registry.unregister("sample_tool")

        assert result is True
        assert len(registry) == 0
        assert "sample_tool" not in registry

    def test_unregister_nonexistent(self):
        """Test unregistering non-existent tool returns False."""
        registry = ToolRegistry()

        result = registry.unregister("nonexistent")

        assert result is False

    def test_get_tool(self):
        """Test getting a tool by name."""
        registry = ToolRegistry()
        tool = SampleTool()

        registry.register(tool)
        retrieved = registry.get("sample_tool")

        assert retrieved is tool

    def test_get_nonexistent(self):
        """Test getting non-existent tool returns None."""
        registry = ToolRegistry()

        result = registry.get("nonexistent")

        assert result is None

    def test_has_tool(self):
        """Test has() method."""
        registry = ToolRegistry()
        tool = SampleTool()

        registry.register(tool)

        assert registry.has("sample_tool") is True
        assert registry.has("nonexistent") is False

    def test_contains(self):
        """Test __contains__ method."""
        registry = ToolRegistry()
        tool = SampleTool()

        registry.register(tool)

        assert "sample_tool" in registry
        assert "nonexistent" not in registry

    def test_tools_property(self):
        """Test tools property returns list of tools."""
        registry = ToolRegistry()
        tool1 = SampleTool("tool_1")
        tool2 = SampleTool("tool_2")

        registry.register(tool1)
        registry.register(tool2)

        tools = registry.tools
        assert len(tools) == 2
        assert tool1 in tools
        assert tool2 in tools

    def test_tool_names_property(self):
        """Test tool_names property."""
        registry = ToolRegistry()
        tool1 = SampleTool("alpha")
        tool2 = SampleTool("beta")

        registry.register(tool1)
        registry.register(tool2)

        names = registry.tool_names
        assert set(names) == {"alpha", "beta"}

    def test_get_definitions(self):
        """Test getting tool definitions."""
        registry = ToolRegistry()
        tool1 = SampleTool("tool_1")
        tool2 = SampleTool("tool_2")

        registry.register(tool1)
        registry.register(tool2)

        definitions = registry.get_definitions()

        assert len(definitions) == 2
        assert all(isinstance(d, ToolDefinition) for d in definitions)
        names = {d.name for d in definitions}
        assert names == {"tool_1", "tool_2"}

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """Test executing a tool by name."""
        registry = ToolRegistry()
        tool = SampleTool()

        registry.register(tool)
        result = await registry.execute("sample_tool", value="test")

        assert result.success is True
        assert result.output == "Processed: test"

    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self):
        """Test executing non-existent tool returns error."""
        registry = ToolRegistry()

        result = await registry.execute("nonexistent", value="test")

        assert result.success is False
        assert "not found" in result.error

    def test_clear(self):
        """Test clearing all tools."""
        registry = ToolRegistry()
        tool1 = SampleTool("tool_1")
        tool2 = SampleTool("tool_2")

        registry.register(tool1)
        registry.register(tool2)
        registry.clear()

        assert len(registry) == 0
        assert registry.tools == []

    def test_len(self):
        """Test __len__ method."""
        registry = ToolRegistry()

        assert len(registry) == 0

        registry.register(SampleTool("tool_1"))
        assert len(registry) == 1

        registry.register(SampleTool("tool_2"))
        assert len(registry) == 2


class TestToolRegistryWithDifferentTools:
    """Tests with different tool types."""

    class AsyncTool(BaseTool):
        """Async tool for testing."""

        @property
        def name(self) -> str:
            return "async_tool"

        @property
        def description(self) -> str:
            return "An async tool"

        @property
        def parameters(self) -> Dict[str, Any]:
            return {"type": "object", "properties": {}}

        async def _execute(self) -> str:
            import asyncio
            await asyncio.sleep(0.001)  # Simulate async work
            return "Async result"

    class FailingTool(BaseTool):
        """Tool that fails."""

        @property
        def name(self) -> str:
            return "failing_tool"

        @property
        def description(self) -> str:
            return "A failing tool"

        @property
        def parameters(self) -> Dict[str, Any]:
            return {"type": "object", "properties": {}}

        async def _execute(self) -> str:
            raise RuntimeError("Intentional failure")

    @pytest.mark.asyncio
    async def test_execute_async_tool(self):
        """Test executing async tool."""
        registry = ToolRegistry()
        registry.register(self.AsyncTool())

        result = await registry.execute("async_tool")

        assert result.success is True
        assert result.output == "Async result"

    @pytest.mark.asyncio
    async def test_execute_failing_tool(self):
        """Test executing tool that raises exception."""
        registry = ToolRegistry()
        registry.register(self.FailingTool())

        result = await registry.execute("failing_tool")

        assert result.success is False
        assert "Intentional failure" in result.error
