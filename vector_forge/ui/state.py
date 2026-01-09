"""State management for Vector Forge TUI.

Provides reactive state containers for tracking extraction progress,
parallel executions, and UI state across screens.
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

    INITIALIZING = "Initializing"
    GENERATING_DATAPOINTS = "Generating datapoints"
    OPTIMIZING = "Optimizing"
    EVALUATING = "Evaluating"
    JUDGE_REVIEW = "Judge review"
    NOISE_REDUCTION = "Noise reduction"
    COMPLETE = "Complete"
    FAILED = "Failed"


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
class ActivityEntry:
    """A single entry in the activity log."""

    timestamp: float
    icon: str
    message: str
    status: str = "active"  # active, success, error, waiting

    @property
    def time_str(self) -> str:
        """Format timestamp as HH:MM:SS."""
        dt = datetime.fromtimestamp(self.timestamp)
        return dt.strftime("%H:%M:%S")


@dataclass
class LogEntry:
    """A single entry in the event log."""

    timestamp: float
    source: str
    message: str
    level: str = "info"  # info, warning, error
    extraction_id: Optional[str] = None

    @property
    def time_str(self) -> str:
        """Format timestamp as HH:MM:SS."""
        dt = datetime.fromtimestamp(self.timestamp)
        return dt.strftime("%H:%M:%S")


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

    # Metrics
    datapoints: DatapointMetrics = field(default_factory=DatapointMetrics)
    evaluation: EvaluationMetrics = field(default_factory=EvaluationMetrics)

    # Activity and timing
    activity: List[ActivityEntry] = field(default_factory=list)
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
    def progress_label(self) -> str:
        """Generate progress label text."""
        parts = [f"Iteration {self.outer_iteration}/{self.max_outer_iterations}"]

        if self.phase == Phase.OPTIMIZING and self.current_layer is not None:
            parts.append(f"Layer {self.current_layer}")

        parts.append(f"Turn {self.inner_turn}/{self.max_inner_turns}")

        return " · ".join(parts)

    def add_activity(self, icon: str, message: str, status: str = "active") -> None:
        """Add an activity entry, keeping only the most recent entries."""
        entry = ActivityEntry(
            timestamp=time.time(),
            icon=icon,
            message=message,
            status=status,
        )
        self.activity.append(entry)
        # Keep only last 10 entries
        if len(self.activity) > 10:
            self.activity = self.activity[-10:]


@dataclass
class UIState:
    """Global UI state container."""

    # All extractions (for parallel view)
    extractions: Dict[str, ExtractionUIState] = field(default_factory=dict)

    # Currently selected/focused extraction
    selected_id: Optional[str] = None

    # Global log entries
    logs: List[LogEntry] = field(default_factory=list)

    # UI preferences
    log_filter: str = ""
    log_source_filter: Optional[str] = None
    log_level_filter: Optional[str] = None
    logs_collapsed: bool = False

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
    ) -> None:
        """Add a log entry."""
        entry = LogEntry(
            timestamp=time.time(),
            source=source,
            message=message,
            level=level,
            extraction_id=extraction_id,
        )
        self.logs.append(entry)

        # Keep only last 1000 entries
        if len(self.logs) > 1000:
            self.logs = self.logs[-1000:]

        self._notify()

    def get_filtered_logs(self) -> List[LogEntry]:
        """Get logs filtered by current filter settings."""
        logs = self.logs

        # Filter by extraction if in single view
        if self.selected_id and len(self.extractions) == 1:
            logs = [
                log for log in logs
                if log.extraction_id is None or log.extraction_id == self.selected_id
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
