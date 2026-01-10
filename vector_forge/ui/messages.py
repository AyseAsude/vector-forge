"""Thread-safe messages for streaming UI updates.

These messages can be posted from any thread using post_message()
and will be handled in the main Textual event loop.
"""

from dataclasses import dataclass
from typing import Optional, Any

from textual.message import Message


class StreamMessage(Message):
    """Base class for all streaming update messages."""
    pass


@dataclass
class LogAdded(StreamMessage):
    """A new log entry was added."""
    timestamp: float
    source: str
    message: str
    level: str = "info"
    extraction_id: Optional[str] = None
    agent_id: Optional[str] = None


@dataclass
class ProgressUpdated(StreamMessage):
    """Extraction progress was updated."""
    extraction_id: str
    progress: float
    phase: str
    message: str = ""


@dataclass
class AgentUpdated(StreamMessage):
    """An agent's state changed."""
    extraction_id: str
    agent_id: str
    status: str
    current_tool: Optional[str] = None
    turns: int = 0
    tool_calls_count: int = 0


@dataclass
class AgentMessageAdded(StreamMessage):
    """A new message was added to an agent's conversation."""
    extraction_id: str
    agent_id: str
    role: str
    content: str
    tool_calls: list = None

    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []


@dataclass
class ExtractionStatusChanged(StreamMessage):
    """Extraction status changed (started, completed, failed)."""
    extraction_id: str
    status: str
    completed_at: Optional[float] = None


@dataclass
class SelectionChanged(StreamMessage):
    """User selection changed."""
    extraction_id: Optional[str] = None
    agent_id: Optional[str] = None


@dataclass
class MetricsUpdated(StreamMessage):
    """Evaluation or datapoint metrics updated."""
    extraction_id: str
    metric_type: str  # "datapoints" or "evaluation"
    data: dict = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}


class RefreshTime(StreamMessage):
    """Signal to refresh time displays only (lightweight)."""
    pass
