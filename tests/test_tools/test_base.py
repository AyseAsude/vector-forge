"""Tests for vector_forge.tools.base module."""

import pytest
from typing import Dict, Any

from vector_forge.tools.base import BaseTool, FunctionTool, tool
from vector_forge.core.protocols import ToolResult, ToolDefinition


class TestBaseTool:
    """Tests for BaseTool base class."""

    class ConcreteBaseTool(BaseTool):
        """Concrete implementation for testing."""

        @property
        def name(self) -> str:
            return "concrete_base_tool"

        @property
        def description(self) -> str:
            return "A concrete base tool"

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
            return f"Result: {value}"

    class FailingBaseTool(BaseTool):
        """Tool that raises exception."""

        @property
        def name(self) -> str:
            return "failing_base_tool"

        @property
        def description(self) -> str:
            return "A failing tool"

        @property
        def parameters(self) -> Dict[str, Any]:
            return {"type": "object", "properties": {}}

        async def _execute(self) -> str:
            raise ValueError("Intentional error")

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful execution wraps result in ToolResult."""
        tool = self.ConcreteBaseTool()
        result = await tool.execute(value="test")

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output == "Result: test"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_execute_failure(self):
        """Test execution failure wraps error in ToolResult."""
        tool = self.FailingBaseTool()
        result = await tool.execute()

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.output is None
        assert "Intentional error" in result.error

    def test_to_definition(self):
        """Test conversion to ToolDefinition."""
        tool = self.ConcreteBaseTool()
        definition = tool.to_definition()

        assert isinstance(definition, ToolDefinition)
        assert definition.name == "concrete_base_tool"
        assert definition.description == "A concrete base tool"
        assert definition.parameters["type"] == "object"


class TestFunctionTool:
    """Tests for FunctionTool wrapper class."""

    def test_create_from_function(self):
        """Test creating FunctionTool from a function."""
        def my_function(value: str) -> str:
            """My function description."""
            return f"Result: {value}"

        tool = FunctionTool(func=my_function)

        assert tool.name == "my_function"
        assert tool.description == "My function description."

    def test_custom_name(self):
        """Test setting custom name."""
        def some_func():
            pass

        tool = FunctionTool(func=some_func, name="custom_name")
        assert tool.name == "custom_name"

    def test_custom_description(self):
        """Test setting custom description."""
        def some_func():
            pass

        tool = FunctionTool(func=some_func, description="Custom desc")
        assert tool.description == "Custom desc"

    def test_custom_parameters(self):
        """Test setting custom parameters."""
        def some_func():
            pass

        params = {
            "type": "object",
            "properties": {"x": {"type": "number"}},
        }
        tool = FunctionTool(func=some_func, parameters=params)
        assert tool.parameters == params

    def test_infer_parameters_string(self):
        """Test parameter inference for string type."""
        def func(name: str):
            pass

        tool = FunctionTool(func=func)
        assert tool.parameters["properties"]["name"]["type"] == "string"
        assert "name" in tool.parameters["required"]

    def test_infer_parameters_int(self):
        """Test parameter inference for int type."""
        def func(count: int):
            pass

        tool = FunctionTool(func=func)
        assert tool.parameters["properties"]["count"]["type"] == "integer"

    def test_infer_parameters_float(self):
        """Test parameter inference for float type."""
        def func(ratio: float):
            pass

        tool = FunctionTool(func=func)
        assert tool.parameters["properties"]["ratio"]["type"] == "number"

    def test_infer_parameters_bool(self):
        """Test parameter inference for bool type."""
        def func(enabled: bool):
            pass

        tool = FunctionTool(func=func)
        assert tool.parameters["properties"]["enabled"]["type"] == "boolean"

    def test_infer_parameters_list(self):
        """Test parameter inference for list type."""
        def func(items: list):
            pass

        tool = FunctionTool(func=func)
        assert tool.parameters["properties"]["items"]["type"] == "array"

    def test_infer_parameters_dict(self):
        """Test parameter inference for dict type."""
        def func(data: dict):
            pass

        tool = FunctionTool(func=func)
        assert tool.parameters["properties"]["data"]["type"] == "object"

    def test_infer_parameters_optional(self):
        """Test parameter inference for optional parameters."""
        def func(required: str, optional: str = "default"):
            pass

        tool = FunctionTool(func=func)
        assert "required" in tool.parameters["required"]
        assert "optional" not in tool.parameters["required"]

    def test_infer_parameters_skips_self_cls(self):
        """Test that self and cls are skipped."""
        class MyClass:
            def method(self, value: str):
                pass

            @classmethod
            def classmethod(cls, value: str):
                pass

        tool1 = FunctionTool(func=MyClass().method)
        assert "self" not in tool1.parameters.get("properties", {})

    @pytest.mark.asyncio
    async def test_execute_sync_function(self):
        """Test executing synchronous function."""
        def sync_func(a: int, b: int) -> int:
            return a + b

        tool = FunctionTool(func=sync_func)
        result = await tool._execute(a=5, b=3)

        assert result == 8

    @pytest.mark.asyncio
    async def test_execute_async_function(self):
        """Test executing asynchronous function."""
        async def async_func(value: str) -> str:
            return f"Async: {value}"

        tool = FunctionTool(func=async_func)
        result = await tool._execute(value="test")

        assert result == "Async: test"

    def test_no_docstring(self):
        """Test function without docstring."""
        def no_doc_func():
            pass

        tool = FunctionTool(func=no_doc_func)
        assert tool.description == "No description"


class TestToolDecorator:
    """Tests for @tool decorator."""

    def test_basic_decorator(self):
        """Test basic decorator usage."""
        @tool()
        def my_tool(value: str) -> str:
            """Process a value."""
            return f"Processed: {value}"

        assert isinstance(my_tool, FunctionTool)
        assert my_tool.name == "my_tool"
        assert my_tool.description == "Process a value."

    def test_decorator_with_name(self):
        """Test decorator with custom name."""
        @tool(name="custom_tool")
        def some_function():
            pass

        assert some_function.name == "custom_tool"

    def test_decorator_with_description(self):
        """Test decorator with custom description."""
        @tool(description="My custom description")
        def some_function():
            pass

        assert some_function.description == "My custom description"

    def test_decorator_with_parameters(self):
        """Test decorator with custom parameters."""
        custom_params = {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "X value"},
            },
        }

        @tool(parameters=custom_params)
        def some_function(x: float):
            pass

        assert some_function.parameters == custom_params

    @pytest.mark.asyncio
    async def test_decorated_tool_execution(self):
        """Test executing decorated tool."""
        @tool(description="Multiply two numbers")
        def multiply(a: int, b: int) -> int:
            return a * b

        result = await multiply.execute(a=4, b=5)

        assert result.success is True
        assert result.output == 20

    @pytest.mark.asyncio
    async def test_decorated_async_tool(self):
        """Test executing decorated async tool."""
        @tool(description="Async greeting")
        async def async_greet(name: str) -> str:
            return f"Hello, {name}!"

        result = await async_greet.execute(name="World")

        assert result.success is True
        assert result.output == "Hello, World!"

    @pytest.mark.asyncio
    async def test_decorated_tool_error_handling(self):
        """Test error handling for decorated tool."""
        @tool(description="Failing tool")
        def fail_tool():
            raise RuntimeError("Something went wrong")

        result = await fail_tool.execute()

        assert result.success is False
        assert "Something went wrong" in result.error

    def test_decorator_preserves_docstring_when_no_description(self):
        """Test decorator uses docstring when no description provided."""
        @tool()
        def documented_tool(x: int) -> int:
            """This tool is documented."""
            return x * 2

        assert documented_tool.description == "This tool is documented."

    def test_decorated_tool_to_definition(self):
        """Test converting decorated tool to definition."""
        @tool(description="Add numbers")
        def add(a: int, b: int) -> int:
            return a + b

        definition = add.to_definition()

        assert isinstance(definition, ToolDefinition)
        assert definition.name == "add"
        assert definition.description == "Add numbers"
