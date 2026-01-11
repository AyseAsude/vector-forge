"""Session service for managing extraction session lifecycle.

Provides a high-level API for session operations, bridging the gap
between the UI layer and the storage layer.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from vector_forge.storage import (
    StorageManager,
    SessionStore,
    SessionStartedEvent,
    SessionCompletedEvent,
    EventEnvelope,
)
from vector_forge.tasks.config import TaskConfig

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Summary information about a session for listing."""

    session_id: str
    behavior: str
    status: str  # "running", "completed", "failed"
    created_at: datetime
    completed_at: Optional[datetime] = None
    final_score: Optional[float] = None
    duration_seconds: float = 0.0
    event_count: int = 0

    @classmethod
    def from_metadata(cls, metadata: Dict[str, Any]) -> "SessionInfo":
        """Create from session metadata dict."""
        created_at = metadata.get("created_at", "")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                created_at = datetime.now()

        completed_at = metadata.get("completed_at")
        if isinstance(completed_at, str):
            try:
                completed_at = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
            except ValueError:
                completed_at = None

        duration = 0.0
        if created_at and completed_at:
            duration = (completed_at - created_at).total_seconds()

        return cls(
            session_id=metadata.get("session_id", ""),
            behavior=metadata.get("behavior", "Unknown"),
            status=metadata.get("status", "unknown"),
            created_at=created_at,
            completed_at=completed_at,
            final_score=metadata.get("final_score"),
            duration_seconds=duration,
            event_count=metadata.get("event_count", 0),
        )


@dataclass
class SessionSummary:
    """Detailed summary of a session including statistics."""

    info: SessionInfo
    config: Dict[str, Any] = field(default_factory=dict)
    datapoint_count: int = 0
    llm_call_count: int = 0
    tool_call_count: int = 0
    vector_count: int = 0
    evaluation_count: int = 0
    total_tokens: int = 0
    final_layer: Optional[int] = None
    recommended_strength: float = 1.0


class SessionService:
    """Manages extraction session lifecycle.

    Provides high-level operations for creating, querying, and managing
    extraction sessions. Acts as a facade over StorageManager.

    Example:
        >>> service = SessionService()
        >>> session_id = service.create_session("sycophancy", config)
        >>> sessions = service.list_sessions(status="completed")
        >>> summary = service.get_session_summary(session_id)
    """

    def __init__(self, base_path: Optional[Path] = None) -> None:
        """Initialize the session service.

        Args:
            base_path: Base path for session storage.
                       Defaults to ~/.vector-forge/sessions
        """
        self._storage = StorageManager(base_path)
        self._active_sessions: Dict[str, SessionStore] = {}

    def create_session(
        self,
        behavior_name: str,
        behavior_description: str,
        config: TaskConfig,
    ) -> str:
        """Create a new extraction session.

        Args:
            behavior_name: Name of the behavior being extracted.
            behavior_description: Detailed description.
            config: Task configuration.

        Returns:
            The session ID.
        """
        # Convert config to dict for storage
        config_dict = config.model_dump() if hasattr(config, "model_dump") else {}

        # Create session in storage
        store = self._storage.create_session(behavior_name, config_dict)
        session_id = store.session_id

        # Track as active
        self._active_sessions[session_id] = store

        # Emit session started event
        started_event = SessionStartedEvent(
            behavior_name=behavior_name,
            behavior_description=behavior_description,
            config=config_dict,
        )
        store.append_event(started_event, source="session_service")

        logger.info(f"Created session: {session_id} for behavior: {behavior_name}")
        return session_id

    def get_session_store(self, session_id: str) -> SessionStore:
        """Get the SessionStore for a session.

        Args:
            session_id: The session identifier.

        Returns:
            SessionStore instance.

        Raises:
            FileNotFoundError: If session doesn't exist.
        """
        if session_id in self._active_sessions:
            return self._active_sessions[session_id]

        store = self._storage.get_session(session_id)
        self._active_sessions[session_id] = store
        return store

    def complete_session(
        self,
        session_id: str,
        success: bool,
        final_score: Optional[float] = None,
        final_layer: Optional[int] = None,
        total_tokens: int = 0,
        error: Optional[str] = None,
    ) -> None:
        """Mark a session as completed.

        Args:
            session_id: The session identifier.
            success: Whether the extraction succeeded.
            final_score: Final evaluation score.
            final_layer: Selected layer.
            total_tokens: Total tokens used.
            error: Error message if failed.
        """
        store = self.get_session_store(session_id)

        # Emit completion event
        completed_event = SessionCompletedEvent(
            success=success,
            final_score=final_score,
            final_layer=final_layer,
            total_tokens=total_tokens,
            error=error,
        )
        store.append_event(completed_event, source="session_service")

        # Update metadata
        store.finalize(success, error)

        # Remove from active
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]

        logger.info(f"Session {session_id} completed: success={success}, score={final_score}")

    def list_sessions(
        self,
        status: Optional[str] = None,
        behavior: Optional[str] = None,
        limit: int = 50,
    ) -> List[SessionInfo]:
        """List sessions with optional filtering.

        Args:
            status: Filter by status ("running", "completed", "failed").
            behavior: Filter by behavior name.
            limit: Maximum number of sessions to return.

        Returns:
            List of SessionInfo objects, newest first.
        """
        raw_sessions = self._storage.list_sessions(
            status=status,
            behavior=behavior,
            limit=limit,
        )

        sessions = []
        for raw in raw_sessions:
            try:
                # Enrich with full metadata if available
                session_id = raw.get("session_id", "")
                try:
                    store = self._storage.get_session(session_id)
                    metadata = store.get_metadata()
                    # Merge index data with full metadata
                    full_data = {**raw, **metadata}
                    sessions.append(SessionInfo.from_metadata(full_data))
                except FileNotFoundError:
                    # Session directory missing, use index data only
                    sessions.append(SessionInfo.from_metadata(raw))
            except Exception as e:
                logger.warning(f"Error loading session {raw.get('session_id')}: {e}")

        return sessions

    def get_session_info(self, session_id: str) -> SessionInfo:
        """Get info for a specific session.

        Args:
            session_id: The session identifier.

        Returns:
            SessionInfo object.
        """
        store = self.get_session_store(session_id)
        metadata = store.get_metadata()
        return SessionInfo.from_metadata(metadata)

    def get_session_summary(self, session_id: str) -> SessionSummary:
        """Get detailed summary of a session.

        Args:
            session_id: The session identifier.

        Returns:
            SessionSummary with statistics.
        """
        store = self.get_session_store(session_id)
        metadata = store.get_metadata()
        info = SessionInfo.from_metadata(metadata)

        # Count events by type
        llm_count = 0
        tool_count = 0
        datapoint_count = 0
        vector_count = 0
        eval_count = 0
        total_tokens = 0

        for event in store.iter_events():
            event_type = event.event_type
            if event_type == "llm.response":
                llm_count += 1
                usage = event.payload.get("usage", {})
                total_tokens += usage.get("total_tokens", 0)
            elif event_type == "tool.result":
                tool_count += 1
            elif event_type == "datapoint.added":
                datapoint_count += 1
            elif event_type == "vector.created":
                vector_count += 1
            elif event_type == "evaluation.completed":
                eval_count += 1

        # Load config
        config = {}
        config_path = store.session_path / "config.json"
        if config_path.exists():
            import json
            with open(config_path, "r") as f:
                config = json.load(f)

        return SessionSummary(
            info=info,
            config=config,
            datapoint_count=datapoint_count,
            llm_call_count=llm_count,
            tool_call_count=tool_count,
            vector_count=vector_count,
            evaluation_count=eval_count,
            total_tokens=total_tokens,
            final_layer=metadata.get("final_layer"),
            recommended_strength=metadata.get("recommended_strength", 1.0),
        )

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its data.

        Args:
            session_id: The session to delete.

        Returns:
            True if deleted, False if not found.
        """
        # Remove from active tracking
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]

        return self._storage.delete_session(session_id)

    def hide_session(self, session_id: str) -> bool:
        """Hide a session (won't appear in listings).

        The session data is preserved. Remove the .hidden file
        from the session directory to unhide it.

        Args:
            session_id: The session to hide.

        Returns:
            True if hidden, False if not found.
        """
        # Remove from active tracking
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]

        return self._storage.hide_session(session_id)

    def emit_event(
        self,
        session_id: str,
        event: Any,
        source: str = "system",
    ) -> EventEnvelope:
        """Emit an event to a session.

        Args:
            session_id: The session identifier.
            event: The event payload.
            source: Source component name.

        Returns:
            The created event envelope.
        """
        store = self.get_session_store(session_id)
        envelope = store.append_event(event, source=source)
        return envelope

    def get_recent_events(
        self,
        session_id: str,
        limit: int = 100,
    ) -> List[EventEnvelope]:
        """Get recent events from a session.

        Args:
            session_id: The session identifier.
            limit: Maximum events to return.

        Returns:
            List of recent events (newest last).
        """
        store = self.get_session_store(session_id)
        events = list(store.iter_events())
        return events[-limit:] if len(events) > limit else events

    @property
    def active_session_ids(self) -> List[str]:
        """Get list of currently active session IDs."""
        return list(self._active_sessions.keys())
