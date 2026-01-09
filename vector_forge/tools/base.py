"""Base tool class and decorator for tool creation."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, get_type_hints
from functools import wraps
import inspect

from vector_forge.core.protocols import Tool, ToolResult, ToolDefinition


class BaseTool(Tool, ABC):
    """
    Base class for tools with common functionality.

    Subclasses should implement:
    - name: Tool identifier
    - description: What the tool does
    - parameters: JSON schema for parameters
    - _execute: The actual tool logic

    Example:
        >>> class MyTool(BaseTool):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_tool"
        ...
        ...     @property
        ...     def description(self) -> str:
        ...         return "Does something useful"
        ...
        ...     @property
        ...     def parameters(self) -> Dict[str, Any]:
        ...         return {
        ...             "type": "object",
        ...             "properties": {"value": {"type": "string"}},
        ...             "required": ["value"],
        ...         }
        ...
        ...     async def _execute(self, value: str) -> Any:
        ...         return f"Processed: {value}"
    """

    @abstractmethod
    async def _execute(self, **kwargs: Any) -> Any:
        """
        Implement the tool logic.

        Returns:
            The tool output. Will be wrapped in ToolResult.
        """
        ...

    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool and wrap result in ToolResult.

        Args:
            **kwargs: Tool parameters.

        Returns:
            ToolResult with success/failure status and output.
        """
        try:
            output = await self._execute(**kwargs)
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class FunctionTool(BaseTool):
    """
    Tool wrapper for a function.

    Created by the @tool decorator.
    """

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self._func = func
        self._name = name or func.__name__
        self._description = description or func.__doc__ or "No description"
        self._parameters = parameters or self._infer_parameters(func)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters

    async def _execute(self, **kwargs: Any) -> Any:
        result = self._func(**kwargs)
        # Handle both sync and async functions
        if inspect.iscoroutine(result):
            return await result
        return result

    @staticmethod
    def _infer_parameters(func: Callable) -> Dict[str, Any]:
        """Infer JSON schema from function signature."""
        sig = inspect.signature(func)
        hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_schema: Dict[str, Any] = {}
            param_type = hints.get(param_name, Any)

            # Map Python types to JSON schema types
            if param_type == str:
                param_schema["type"] = "string"
            elif param_type == int:
                param_schema["type"] = "integer"
            elif param_type == float:
                param_schema["type"] = "number"
            elif param_type == bool:
                param_schema["type"] = "boolean"
            elif param_type == list:
                param_schema["type"] = "array"
            elif param_type == dict:
                param_schema["type"] = "object"
            else:
                param_schema["type"] = "string"

            properties[param_name] = param_schema

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> Callable[[Callable], FunctionTool]:
    """
    Decorator to create a tool from a function.

    Args:
        name: Tool name (defaults to function name).
        description: Tool description (defaults to docstring).
        parameters: JSON schema for parameters (inferred if not provided).

    Returns:
        Decorator that wraps function in FunctionTool.

    Example:
        >>> @tool(description="Add two numbers")
        ... def add(a: int, b: int) -> int:
        ...     return a + b
        ...
        >>> result = await add.execute(a=1, b=2)
        >>> assert result.output == 3
    """

    def decorator(func: Callable) -> FunctionTool:
        return FunctionTool(
            func=func,
            name=name,
            description=description,
            parameters=parameters,
        )

    return decorator
