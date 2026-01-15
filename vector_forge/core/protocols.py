"""Core protocols (interfaces) for Vector Forge."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import torch


T = TypeVar("T")


# =============================================================================
# LLM Abstractions
# =============================================================================


@dataclass
class Message:
    """A chat message."""

    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: Optional[str] = None  # For tool messages
    tool_call_id: Optional[str] = None  # For tool results
    tool_calls: Optional[List["ToolCall"]] = None  # For assistant messages


@dataclass
class ToolCall:
    """A tool call from the LLM."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ToolDefinition:
    """Definition of a tool for the LLM."""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema


@dataclass
class LLMResponse:
    """Response from an LLM completion."""

    content: Optional[str]
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Optional[Dict[str, int]] = None


@runtime_checkable
class LLMClient(Protocol):
    """
    Protocol for LLM clients.

    Implementations can use litellm, openai, anthropic, or any other provider.
    """

    async def complete(
        self,
        messages: List[Message],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion from the LLM.

        Args:
            messages: List of chat messages.
            **kwargs: Additional arguments (temperature, max_tokens, etc.)

        Returns:
            LLMResponse with content and optional tool calls.
        """
        ...

    async def complete_with_tools(
        self,
        messages: List[Message],
        tools: List[ToolDefinition],
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion with tool use capability.

        Args:
            messages: List of chat messages.
            tools: Available tools the LLM can call.
            **kwargs: Additional arguments.

        Returns:
            LLMResponse with content and/or tool calls.
        """
        ...


# =============================================================================
# Tool Abstractions
# =============================================================================


@dataclass
class ToolResult:
    """Result from executing a tool."""

    success: bool
    output: Any
    error: Optional[str] = None

    def to_string(self) -> str:
        """Convert result to string for LLM consumption."""
        if self.success:
            return str(self.output)
        return f"Error: {self.error}"


class Tool(ABC):
    """
    Abstract base class for tools that agents can execute.

    Each tool is a self-contained operation with a name, description,
    and parameter schema that the LLM uses to decide when and how to call it.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this tool."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this tool does (shown to LLM)."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema for tool parameters."""
        ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Tool-specific parameters.

        Returns:
            ToolResult indicating success/failure and output.
        """
        ...

    def to_definition(self) -> ToolDefinition:
        """Convert to ToolDefinition for LLM."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


# =============================================================================
# Strategy Protocols
# =============================================================================


@runtime_checkable
class DatapointStrategy(Protocol):
    """
    Strategy for generating training datapoints.

    Different implementations provide different approaches:
    - ContrastiveStrategy: Generate dst and src completions
    - SingleDirectionStrategy: Generate dst completions only
    - ActivationDifferenceStrategy: Use activation differences
    """

    async def generate(
        self,
        behavior: "BehaviorSpec",
        prompts: List[str],
        llm_client: LLMClient,
        **kwargs: Any,
    ) -> List["TrainingDatapoint"]:
        """
        Generate training datapoints from prompts.

        Args:
            behavior: The target behavior specification.
            prompts: Base prompts to generate completions for.
            llm_client: LLM client for generating completions.
            **kwargs: Strategy-specific arguments.

        Returns:
            List of TrainingDatapoint objects.
        """
        ...


@runtime_checkable
class NoiseReducer(Protocol):
    """
    Strategy for reducing noise in steering vectors.

    Different implementations:
    - AveragingReducer: Average multiple vectors from different seeds
    - PCAReducer: Project to principal components
    - AdversarialReducer: Remove anti-behavior projection
    """

    def reduce(
        self,
        vectors: List[torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Reduce noise in vectors and return a clean vector.

        Args:
            vectors: List of vectors (possibly from multiple optimization runs).
            **kwargs: Strategy-specific arguments.

        Returns:
            A single cleaned vector.
        """
        ...


@runtime_checkable
class LayerSearchStrategy(Protocol):
    """
    Strategy for searching optimal layer(s) for steering.

    Different implementations:
    - FixedLayerStrategy: Use specified layers
    - AdaptiveSearchStrategy: Start coarse, refine around best
    - BinarySearchStrategy: Binary search for optimal layer
    """

    def get_layers_to_try(
        self,
        total_layers: int,
        iteration: int = 0,
        previous_results: Optional[Dict[int, float]] = None,
    ) -> List[int]:
        """
        Get layers to try in this iteration.

        Args:
            total_layers: Total number of layers in the model.
            iteration: Current search iteration (for adaptive strategies).
            previous_results: Results from previous iterations {layer: score}.

        Returns:
            List of layer indices to try.
        """
        ...


@runtime_checkable
class VectorEvaluator(Protocol):
    """Protocol for evaluating steering vector quality."""

    async def evaluate(
        self,
        vector: torch.Tensor,
        layer: int,
        behavior: "BehaviorSpec",
        test_prompts: List[str],
        model_backend: Any,
        **kwargs: Any,
    ) -> "EvaluationResult":
        """
        Evaluate a steering vector's quality.

        Args:
            vector: The steering vector to evaluate.
            layer: Layer the vector is applied at.
            behavior: Target behavior specification.
            test_prompts: Prompts to test the vector on.
            model_backend: Backend for running the model.
            **kwargs: Additional arguments.

        Returns:
            EvaluationResult with scores and citations.
        """
        ...


# =============================================================================
# Event System
# =============================================================================


@dataclass
class Event:
    """Base event class for the event system."""

    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None


EventHandler = Callable[[Event], None]


class EventEmitter(ABC):
    """
    Base class for components that emit events.

    UIs can subscribe to events to display progress, logs, and results.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, List[EventHandler]] = {}

    def on(self, event_type: str, handler: EventHandler) -> None:
        """
        Subscribe to an event type.

        Args:
            event_type: Event type to subscribe to, or "*" for all events.
            handler: Callback function to invoke when event is emitted.
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    def off(self, event_type: str, handler: EventHandler) -> None:
        """
        Unsubscribe from an event type.

        Args:
            event_type: Event type to unsubscribe from.
            handler: The handler function to remove.
        """
        if event_type in self._handlers:
            self._handlers[event_type].remove(handler)

    def emit(self, event: Event) -> None:
        """
        Emit an event to all subscribers.

        Args:
            event: The event to emit. Timestamp is set automatically if not provided.
        """
        import time

        if event.timestamp is None:
            event.timestamp = time.time()

        # Call handlers for this specific event type
        for handler in self._handlers.get(event.type, []):
            handler(event)

        # Call handlers subscribed to all events ("*")
        for handler in self._handlers.get("*", []):
            handler(event)


# =============================================================================
# Type Hints (avoid circular imports)
# =============================================================================

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vector_forge.core.behavior import BehaviorSpec
    from vector_forge.core.results import EvaluationResult
    from steerex import TrainingDatapoint
