"""Tool registry for managing available tools."""

from typing import Dict, List, Optional, Any

from vector_forge.core.protocols import Tool, ToolDefinition, ToolResult


class ToolRegistry:
    """
    Registry for managing and executing tools.

    Provides a central place to register tools and execute them by name.

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(my_tool)
        >>> registry.register(another_tool)
        >>> result = await registry.execute("my_tool", param="value")
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """
        Register a tool.

        Args:
            tool: Tool instance to register.

        Raises:
            ValueError: If tool with same name already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """
        Unregister a tool by name.

        Args:
            name: Name of tool to unregister.

        Returns:
            True if tool was unregistered, False if not found.
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Optional[Tool]:
        """
        Get a tool by name.

        Args:
            name: Name of tool to get.

        Returns:
            Tool instance or None if not found.
        """
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    @property
    def tools(self) -> List[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    @property
    def tool_names(self) -> List[str]:
        """Get names of all registered tools."""
        return list(self._tools.keys())

    def get_definitions(self) -> List[ToolDefinition]:
        """Get tool definitions for all registered tools."""
        return [tool.to_definition() for tool in self._tools.values()]

    async def execute(self, name: str, **kwargs: Any) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            name: Name of tool to execute.
            **kwargs: Parameters to pass to tool.

        Returns:
            ToolResult from tool execution.
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{name}' not found",
            )
        return await tool.execute(**kwargs)

    def clear(self) -> None:
        """Unregister all tools."""
        self._tools.clear()

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
