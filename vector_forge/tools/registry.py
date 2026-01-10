"""Tool registry for managing available tools."""

import time
import uuid
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from vector_forge.core.protocols import Tool, ToolDefinition, ToolResult

if TYPE_CHECKING:
    from vector_forge.storage import SessionStore


class ToolRegistry:
    """
    Registry for managing and executing tools.

    Provides a central place to register tools and execute them by name.
    Captures all tool calls and results to the session store for
    complete reproducibility.

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(my_tool)
        >>> registry.register(another_tool)
        >>> result = await registry.execute("my_tool", param="value")
    """

    def __init__(
        self,
        store: Optional["SessionStore"] = None,
        agent_id: str = "extractor",
    ) -> None:
        """Initialize registry.

        Args:
            store: Optional session store for event capture.
            agent_id: Identifier for the agent using this registry.
        """
        self._tools: Dict[str, Tool] = {}
        self._store = store
        self._agent_id = agent_id

    def set_store(self, store: Optional["SessionStore"]) -> None:
        """Set or update the session store.

        Args:
            store: Session store for event capture.
        """
        self._store = store

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
        # Generate call ID for linking call/result
        call_id = str(uuid.uuid4())

        # Capture call event
        self._emit_call_event(call_id, name, kwargs)

        start_time = time.time()

        tool = self._tools.get(name)
        if tool is None:
            result = ToolResult(
                success=False,
                output=None,
                error=f"Tool '{name}' not found",
            )
            latency_ms = (time.time() - start_time) * 1000
            self._emit_result_event(call_id, result, latency_ms)
            return result

        try:
            result = await tool.execute(**kwargs)
            latency_ms = (time.time() - start_time) * 1000
            self._emit_result_event(call_id, result, latency_ms)
            return result

        except Exception as e:
            result = ToolResult(
                success=False,
                output=None,
                error=str(e),
            )
            latency_ms = (time.time() - start_time) * 1000
            self._emit_result_event(call_id, result, latency_ms)
            return result

    def _emit_call_event(
        self,
        call_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> None:
        """Emit tool call event to store."""
        if self._store is None:
            return

        from vector_forge.storage import ToolCallEvent

        # Serialize arguments to ensure they're JSON-compatible
        serialized_args = self._serialize_arguments(arguments)

        event = ToolCallEvent(
            call_id=call_id,
            tool_name=tool_name,
            arguments=serialized_args,
            agent_id=self._agent_id,
        )

        self._store.append_event(event, source="tools")

    def _emit_result_event(
        self,
        call_id: str,
        result: ToolResult,
        latency_ms: float,
    ) -> None:
        """Emit tool result event to store."""
        if self._store is None:
            return

        from vector_forge.storage import ToolResultEvent

        # Serialize output to ensure it's JSON-compatible
        serialized_output = self._serialize_output(result.output)

        event = ToolResultEvent(
            call_id=call_id,
            success=result.success,
            output=serialized_output,
            error=result.error,
            latency_ms=latency_ms,
        )

        self._store.append_event(event, source="tools")

    def _serialize_arguments(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize arguments for JSON storage."""
        result = {}
        for key, value in arguments.items():
            result[key] = self._serialize_value(value)
        return result

    def _serialize_output(self, output: Any) -> Any:
        """Serialize output for JSON storage."""
        return self._serialize_value(output)

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value for JSON storage."""
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}

        # Handle torch tensors
        try:
            import torch
            if isinstance(value, torch.Tensor):
                return {
                    "_type": "tensor",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                }
        except ImportError:
            pass

        # Fallback to string representation
        try:
            return str(value)
        except Exception:
            return "<non-serializable>"

    def clear(self) -> None:
        """Unregister all tools."""
        self._tools.clear()

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
