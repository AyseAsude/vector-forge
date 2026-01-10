"""State management for Vector Forge TUI.

Provides reactive state containers for tracking extraction progress,
agent execution, messages, and UI state across screens.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
import time


class ExtractionStatus(str, Enum):
    """Status of an extraction process."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETE = "complete"
    FAILED = "failed"


class Phase(str, Enum):
    """Current phase of extraction."""

    INITIALIZING = "init"
    GENERATING_DATAPOINTS = "gen"
    OPTIMIZING = "opt"
    EVALUATING = "eval"
    JUDGE_REVIEW = "judge"
    NOISE_REDUCTION = "denoise"
    COMPLETE = "done"
    FAILED = "fail"


class AgentStatus(str, Enum):
    """Status of an agent."""

    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETE = "complete"
    ERROR = "error"


class MessageRole(str, Enum):
    """Role of a message in agent conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """A tool call made by an agent."""

    id: str
    name: str
    arguments: str
    result: Optional[str] = None
    status: str = "pending"  # pending, running, success, error
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def duration_ms(self) -> Optional[int]:
        """Calculate duration in milliseconds."""
        if self.started_at is None or self.completed_at is None:
            return None
        return int((self.completed_at - self.started_at) * 1000)


@dataclass
class AgentMessage:
    """A message in an agent's conversation."""

    id: str
    role: MessageRole
    content: str
    timestamp: float
    tool_calls: List[ToolCall] = field(default_factory=list)

    @property
    def time_str(self) -> str:
        """Format timestamp as HH:MM:SS."""
        dt = datetime.fromtimestamp(self.timestamp)
        return dt.strftime("%H:%M:%S")


@dataclass
class AgentUIState:
    """UI state for a single agent."""

    id: str
    name: str
    role: str  # "extractor", "judge", "optimizer", etc.
    status: AgentStatus = AgentStatus.IDLE
    messages: List[AgentMessage] = field(default_factory=list)
    current_tool: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    turns: int = 0
    tool_calls_count: int = 0

    @property
    def elapsed_seconds(self) -> float:
        """Calculate elapsed time in seconds."""
        if self.started_at is None:
            return 0.0
        end = self.completed_at or time.time()
        return end - self.started_at

    @property
    def elapsed_str(self) -> str:
        """Format elapsed time as MM:SS."""
        seconds = int(self.elapsed_seconds)
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:02d}:{secs:02d}"

    @property
    def last_message(self) -> Optional[AgentMessage]:
        """Get the most recent message."""
        return self.messages[-1] if self.messages else None

    def add_message(
        self,
        role: MessageRole,
        content: str,
        tool_calls: Optional[List[ToolCall]] = None,
    ) -> AgentMessage:
        """Add a message to the agent's conversation."""
        msg = AgentMessage(
            id=f"msg_{len(self.messages)}",
            role=role,
            content=content,
            timestamp=time.time(),
            tool_calls=tool_calls or [],
        )
        self.messages.append(msg)
        if role == MessageRole.ASSISTANT:
            self.turns += 1
        return msg


@dataclass
class DatapointMetrics:
    """Metrics about training datapoints."""

    total: int = 0
    keep: int = 0
    review: int = 0
    remove: int = 0
    diversity: float = 0.0
    clusters: int = 0


@dataclass
class EvaluationMetrics:
    """Evaluation scores from the judge."""

    behavior: float = 0.0
    coherence: float = 0.0
    specificity: float = 0.0
    overall: float = 0.0
    best_layer: Optional[int] = None
    best_strength: float = 1.0
    verdict: Optional[str] = None


@dataclass
class LogEntry:
    """A single entry in the event log."""

    timestamp: float
    source: str
    message: str
    level: str = "info"  # info, warning, error
    extraction_id: Optional[str] = None
    agent_id: Optional[str] = None
    event_type: Optional[str] = None  # For rich detail rendering (e.g., "llm.request")
    payload: Optional[Dict[str, Any]] = None  # Raw event data for detail view

    @property
    def time_str(self) -> str:
        """Format timestamp as HH:MM:SS."""
        dt = datetime.fromtimestamp(self.timestamp)
        return dt.strftime("%H:%M:%S")

    @property
    def has_detail(self) -> bool:
        """Check if this entry has detailed payload data."""
        return self.payload is not None and len(self.payload) > 0


@dataclass
class ExtractionUIState:
    """UI state for a single extraction."""

    id: str
    behavior_name: str
    behavior_description: str
    model: str = ""
    status: ExtractionStatus = ExtractionStatus.PENDING
    phase: Phase = Phase.INITIALIZING

    # Progress tracking
    progress: float = 0.0
    outer_iteration: int = 0
    max_outer_iterations: int = 3
    inner_turn: int = 0
    max_inner_turns: int = 50
    current_layer: Optional[int] = None

    # Agents running for this extraction
    agents: Dict[str, AgentUIState] = field(default_factory=dict)
    selected_agent_id: Optional[str] = None

    # Metrics
    datapoints: DatapointMetrics = field(default_factory=DatapointMetrics)
    evaluation: EvaluationMetrics = field(default_factory=EvaluationMetrics)

    # Timing
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def elapsed_seconds(self) -> float:
        """Calculate elapsed time in seconds."""
        if self.started_at is None:
            return 0.0
        end = self.completed_at or time.time()
        return end - self.started_at

    @property
    def elapsed_str(self) -> str:
        """Format elapsed time as MM:SS."""
        seconds = int(self.elapsed_seconds)
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:02d}:{secs:02d}"

    @property
    def selected_agent(self) -> Optional[AgentUIState]:
        """Get the currently selected agent."""
        if self.selected_agent_id is None:
            return None
        return self.agents.get(self.selected_agent_id)

    @property
    def running_agents_count(self) -> int:
        """Count of running agents."""
        return sum(
            1 for a in self.agents.values()
            if a.status == AgentStatus.RUNNING
        )

    @property
    def total_agents_count(self) -> int:
        """Total agent count."""
        return len(self.agents)

    def add_agent(self, agent: AgentUIState) -> None:
        """Add an agent to this extraction."""
        self.agents[agent.id] = agent
        if self.selected_agent_id is None:
            self.selected_agent_id = agent.id

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            if self.selected_agent_id == agent_id:
                self.selected_agent_id = next(iter(self.agents), None)

    def select_agent(self, agent_id: str) -> None:
        """Select an agent for detailed view."""
        if agent_id in self.agents:
            self.selected_agent_id = agent_id


@dataclass
class UIState:
    """Global UI state container."""

    # All extractions
    extractions: Dict[str, ExtractionUIState] = field(default_factory=dict)

    # Currently selected/focused extraction
    selected_id: Optional[str] = None

    # Global log entries
    logs: List[LogEntry] = field(default_factory=list)

    # UI preferences
    log_filter: str = ""
    log_source_filter: Optional[str] = None
    log_level_filter: Optional[str] = None

    # Callbacks for state changes
    _listeners: List[Callable[["UIState"], None]] = field(default_factory=list)

    @property
    def selected_extraction(self) -> Optional[ExtractionUIState]:
        """Get the currently selected extraction."""
        if self.selected_id is None:
            return None
        return self.extractions.get(self.selected_id)

    @property
    def running_count(self) -> int:
        """Count of running extractions."""
        return sum(
            1 for e in self.extractions.values()
            if e.status == ExtractionStatus.RUNNING
        )

    @property
    def complete_count(self) -> int:
        """Count of completed extractions."""
        return sum(
            1 for e in self.extractions.values()
            if e.status == ExtractionStatus.COMPLETE
        )

    @property
    def total_count(self) -> int:
        """Total extraction count."""
        return len(self.extractions)

    def add_extraction(self, extraction: ExtractionUIState) -> None:
        """Add a new extraction to track."""
        self.extractions[extraction.id] = extraction
        if self.selected_id is None:
            self.selected_id = extraction.id
        self._notify()

    def remove_extraction(self, extraction_id: str) -> None:
        """Remove an extraction."""
        if extraction_id in self.extractions:
            del self.extractions[extraction_id]
            if self.selected_id == extraction_id:
                self.selected_id = next(iter(self.extractions), None)
            self._notify()

    def update_extraction(
        self,
        extraction_id: str,
        **updates: Any,
    ) -> None:
        """Update extraction state fields."""
        extraction = self.extractions.get(extraction_id)
        if extraction is None:
            return

        for key, value in updates.items():
            if hasattr(extraction, key):
                setattr(extraction, key, value)

        self._notify()

    def add_log(
        self,
        source: str,
        message: str,
        level: str = "info",
        extraction_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        event_type: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a log entry."""
        entry = LogEntry(
            timestamp=time.time(),
            source=source,
            message=message,
            level=level,
            extraction_id=extraction_id,
            agent_id=agent_id,
            event_type=event_type,
            payload=payload,
        )
        self.logs.append(entry)

        # Keep only last 1000 entries
        if len(self.logs) > 1000:
            self.logs = self.logs[-1000:]

        self._notify()

    def get_filtered_logs(
        self,
        extraction_id: Optional[str] = None,
    ) -> List[LogEntry]:
        """Get logs filtered by current filter settings."""
        logs = self.logs

        # Filter by extraction
        if extraction_id:
            logs = [
                log for log in logs
                if log.extraction_id is None or log.extraction_id == extraction_id
            ]

        # Filter by text
        if self.log_filter:
            filter_lower = self.log_filter.lower()
            logs = [
                log for log in logs
                if filter_lower in log.message.lower()
                or filter_lower in log.source.lower()
            ]

        # Filter by source
        if self.log_source_filter:
            logs = [log for log in logs if log.source == self.log_source_filter]

        # Filter by level
        if self.log_level_filter:
            logs = [log for log in logs if log.level == self.log_level_filter]

        return logs

    def select_extraction(self, extraction_id: str) -> None:
        """Select an extraction for detailed view."""
        if extraction_id in self.extractions:
            self.selected_id = extraction_id
            self._notify()

    def add_listener(self, callback: Callable[["UIState"], None]) -> None:
        """Register a state change listener."""
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable[["UIState"], None]) -> None:
        """Remove a state change listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def _notify(self) -> None:
        """Notify all listeners of state change."""
        for listener in self._listeners:
            listener(self)


# Global state instance
_state: Optional[UIState] = None


def get_state() -> UIState:
    """Get or create the global UI state instance."""
    global _state
    if _state is None:
        _state = UIState()
    return _state


def reset_state() -> UIState:
    """Reset and return a fresh UI state instance."""
    global _state
    _state = UIState()
    return _state
