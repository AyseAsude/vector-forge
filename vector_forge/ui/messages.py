"""Event types for the UI event-sourcing architecture.

Events represent things that HAPPENED (past tense). They flow:
    Background Thread → App (handler) → State (update) → Screen (projection)

All events can be safely posted from any thread via post_message().
"""

from typing import Optional, List

from textual.message import Message


# ─────────────────────────────────────────────────────────────────────────────
# Task Lifecycle Events
# ─────────────────────────────────────────────────────────────────────────────


class TaskCreated(Message):
    """A new task/extraction was created."""

    def __init__(self, task_id: str, name: str, description: str = "") -> None:
        super().__init__()
        self.task_id = task_id
        self.name = name
        self.description = description


class TaskProgressChanged(Message):
    """Task progress or phase changed."""

    def __init__(self, task_id: str, progress: float, phase: str, message: str = "") -> None:
        super().__init__()
        self.task_id = task_id
        self.progress = progress
        self.phase = phase
        self.message = message


class TaskStatusChanged(Message):
    """Task status changed (running, paused, completed, failed)."""

    def __init__(self, task_id: str, status: str, completed_at: Optional[float] = None, error: Optional[str] = None) -> None:
        super().__init__()
        self.task_id = task_id
        self.status = status
        self.completed_at = completed_at
        self.error = error


class TaskRemoved(Message):
    """A task was removed/hidden."""

    def __init__(self, task_id: str) -> None:
        super().__init__()
        self.task_id = task_id


class TaskSelected(Message):
    """User selected a different task."""

    def __init__(self, task_id: Optional[str]) -> None:
        super().__init__()
        self.task_id = task_id


# ─────────────────────────────────────────────────────────────────────────────
# Agent Lifecycle Events
# ─────────────────────────────────────────────────────────────────────────────


class AgentSpawned(Message):
    """A new agent was spawned for a task."""

    def __init__(self, task_id: str, agent_id: str, name: str, role: str = "") -> None:
        super().__init__()
        self.task_id = task_id
        self.agent_id = agent_id
        self.name = name
        self.role = role


class AgentStatusChanged(Message):
    """Agent status changed."""

    def __init__(self, task_id: str, agent_id: str, status: str, current_tool: Optional[str] = None) -> None:
        super().__init__()
        self.task_id = task_id
        self.agent_id = agent_id
        self.status = status
        self.current_tool = current_tool


class AgentProgressChanged(Message):
    """Agent progress changed (turns, tool calls)."""

    def __init__(self, task_id: str, agent_id: str, turns: int = 0, tool_calls_count: int = 0) -> None:
        super().__init__()
        self.task_id = task_id
        self.agent_id = agent_id
        self.turns = turns
        self.tool_calls_count = tool_calls_count


class AgentMessageReceived(Message):
    """A new message was added to an agent's conversation."""

    def __init__(self, task_id: str, agent_id: str, role: str, content: str, tool_calls: Optional[List] = None) -> None:
        super().__init__()
        self.task_id = task_id
        self.agent_id = agent_id
        self.role = role
        self.content = content
        self.tool_calls = tool_calls or []


class AgentSelected(Message):
    """User selected a different agent."""

    def __init__(self, task_id: str, agent_id: Optional[str]) -> None:
        super().__init__()
        self.task_id = task_id
        self.agent_id = agent_id


# ─────────────────────────────────────────────────────────────────────────────
# Log Events
# ─────────────────────────────────────────────────────────────────────────────


class LogEmitted(Message):
    """A log entry was emitted."""

    def __init__(self, timestamp: float, source: str, message: str, level: str = "info", task_id: Optional[str] = None, agent_id: Optional[str] = None) -> None:
        super().__init__()
        self.timestamp = timestamp
        self.source = source
        self.message = message
        self.level = level
        self.task_id = task_id
        self.agent_id = agent_id


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Events
# ─────────────────────────────────────────────────────────────────────────────


class DatapointMetricsChanged(Message):
    """Datapoint metrics were updated."""

    def __init__(self, task_id: str, total: int = 0, keep: int = 0, review: int = 0, remove: int = 0, diversity: float = 0.0, clusters: int = 0) -> None:
        super().__init__()
        self.task_id = task_id
        self.total = total
        self.keep = keep
        self.review = review
        self.remove = remove
        self.diversity = diversity
        self.clusters = clusters


class EvaluationMetricsChanged(Message):
    """Evaluation metrics were updated."""

    def __init__(self, task_id: str, behavior: float = 0.0, coherence: float = 0.0, specificity: float = 0.0, overall: float = 0.0, best_layer: Optional[int] = None, best_strength: float = 0.0, verdict: Optional[str] = None) -> None:
        super().__init__()
        self.task_id = task_id
        self.behavior = behavior
        self.coherence = coherence
        self.specificity = specificity
        self.overall = overall
        self.best_layer = best_layer
        self.best_strength = best_strength
        self.verdict = verdict


# ─────────────────────────────────────────────────────────────────────────────
# Timer Events
# ─────────────────────────────────────────────────────────────────────────────


class TimeTick(Message):
    """Periodic tick for updating elapsed time displays."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# State Sync Event
# ─────────────────────────────────────────────────────────────────────────────


class StateSync(Message):
    """Request to sync screen with current state."""
    pass
