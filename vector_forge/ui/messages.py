"""Event types for the UI event-sourcing architecture.

Events represent things that HAPPENED (past tense). They flow:
    Background Thread → App (handler) → State (update) → Screen (projection)

All events can be safely posted from any thread via post_message().
"""

from dataclasses import dataclass, field
from typing import Optional

from textual.message import Message


# ─────────────────────────────────────────────────────────────────────────────
# Base Event
# ─────────────────────────────────────────────────────────────────────────────


class UIEvent(Message):
    """Base class for all UI events."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Task Lifecycle Events
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TaskCreated(UIEvent):
    """A new task/extraction was created."""
    task_id: str
    name: str
    description: str = ""


@dataclass
class TaskProgressChanged(UIEvent):
    """Task progress or phase changed."""
    task_id: str
    progress: float
    phase: str
    message: str = ""


@dataclass
class TaskStatusChanged(UIEvent):
    """Task status changed (running, paused, completed, failed)."""
    task_id: str
    status: str
    completed_at: Optional[float] = None
    error: Optional[str] = None


@dataclass
class TaskRemoved(UIEvent):
    """A task was removed/hidden."""
    task_id: str


@dataclass
class TaskSelected(UIEvent):
    """User selected a different task."""
    task_id: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# Agent Lifecycle Events
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AgentSpawned(UIEvent):
    """A new agent was spawned for a task."""
    task_id: str
    agent_id: str
    name: str
    role: str = ""


@dataclass
class AgentStatusChanged(UIEvent):
    """Agent status changed."""
    task_id: str
    agent_id: str
    status: str
    current_tool: Optional[str] = None


@dataclass
class AgentProgressChanged(UIEvent):
    """Agent progress changed (turns, tool calls)."""
    task_id: str
    agent_id: str
    turns: int = 0
    tool_calls_count: int = 0


@dataclass
class AgentMessageReceived(UIEvent):
    """A new message was added to an agent's conversation."""
    task_id: str
    agent_id: str
    role: str
    content: str
    tool_calls: list = field(default_factory=list)


@dataclass
class AgentSelected(UIEvent):
    """User selected a different agent."""
    task_id: str
    agent_id: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# Log Events
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LogEmitted(UIEvent):
    """A log entry was emitted."""
    timestamp: float
    source: str
    message: str
    level: str = "info"
    task_id: Optional[str] = None
    agent_id: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Events
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DatapointMetricsChanged(UIEvent):
    """Datapoint metrics were updated."""
    task_id: str
    total: int = 0
    keep: int = 0
    review: int = 0
    remove: int = 0
    diversity: float = 0.0
    clusters: int = 0


@dataclass
class EvaluationMetricsChanged(UIEvent):
    """Evaluation metrics were updated."""
    task_id: str
    behavior: float = 0.0
    coherence: float = 0.0
    specificity: float = 0.0
    overall: float = 0.0
    best_layer: Optional[int] = None
    best_strength: float = 0.0
    verdict: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Timer Events
# ─────────────────────────────────────────────────────────────────────────────


class TimeTick(UIEvent):
    """Periodic tick for updating elapsed time displays.

    This is the ONLY timer-based event. All other updates are event-driven.
    """
    pass


# ─────────────────────────────────────────────────────────────────────────────
# State Sync Event (for initial screen mount)
# ─────────────────────────────────────────────────────────────────────────────


class StateSync(UIEvent):
    """Request to sync screen with current state.

    Used when a screen is mounted and needs to render from current state.
    NOT used for incremental updates - those use specific events.
    """
    pass
