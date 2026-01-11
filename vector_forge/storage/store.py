"""Session store and storage manager for event sourcing.

Provides:
- SessionStore: manages events and vectors for a single session
- StorageManager: creates and manages multiple sessions
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch

from vector_forge.storage.events import (
    EventCategory,
    EventEnvelope,
    EventPayload,
)


class SessionStore:
    """Manages event storage for a single extraction session.

    Handles:
    - JSONL event file (append-only)
    - Vector file storage (.pt files)
    - Session metadata

    Example:
        >>> store = SessionStore("session_123", Path("~/.vector-forge/sessions"))
        >>> store.initialize("sycophancy", {"model": "claude-opus-4-5"})
        >>> store.append_event(LLMRequestEvent(...), source="generator")
        >>> store.save_vector(vector, layer=16)
    """

    def __init__(self, session_id: str, base_path: Path) -> None:
        """Initialize session store.

        Args:
            session_id: Unique session identifier.
            base_path: Base path for all sessions.
        """
        self.session_id = session_id
        self.session_path = base_path / session_id
        self._sequence = 0
        self._events_file: Optional[Path] = None
        self._vector_versions: Dict[int, int] = {}  # layer -> version count
        self._vector_versions_lock = threading.Lock()
        self._initialized = False

    @property
    def events_path(self) -> Path:
        """Path to events JSONL file."""
        return self.session_path / "events.jsonl"

    @property
    def vectors_path(self) -> Path:
        """Path to vectors directory."""
        return self.session_path / "vectors"

    @property
    def checkpoints_path(self) -> Path:
        """Path to checkpoints directory."""
        return self.session_path / "checkpoints"

    def initialize(self, behavior: str, config: Dict[str, Any]) -> None:
        """Create session directory structure and initial files.

        Args:
            behavior: Behavior name being extracted.
            config: Full pipeline configuration to snapshot.
        """
        # Create directories
        self.session_path.mkdir(parents=True, exist_ok=True)
        self.vectors_path.mkdir(exist_ok=True)
        self.checkpoints_path.mkdir(exist_ok=True)

        # Write config snapshot
        config_path = self.session_path / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)

        # Initialize events file
        self._events_file = self.events_path

        # Write metadata
        metadata = {
            "session_id": self.session_id,
            "behavior": behavior,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "running",
        }
        metadata_path = self.session_path / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self._initialized = True

    def _ensure_initialized(self) -> None:
        """Ensure store is initialized before operations."""
        if not self._initialized:
            # Try to resume from existing session
            if self.events_path.exists():
                self._events_file = self.events_path
                self._load_sequence()
                self._initialized = True
            else:
                raise RuntimeError(
                    f"Session {self.session_id} not initialized. "
                    "Call initialize() first or use an existing session."
                )

    def _load_sequence(self) -> None:
        """Load current sequence from existing events file."""
        if not self.events_path.exists():
            return

        with open(self.events_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        envelope = EventEnvelope.model_validate_json(line)
                        self._sequence = max(self._sequence, envelope.sequence)
                    except Exception:
                        pass  # Skip malformed lines

    def append_event(
        self,
        payload: EventPayload,
        source: str,
    ) -> EventEnvelope:
        """Append event to JSONL file.

        Args:
            payload: Typed event payload.
            source: Component that emitted the event.

        Returns:
            The created event envelope.
        """
        self._ensure_initialized()
        self._sequence += 1

        envelope = EventEnvelope.create(
            session_id=self.session_id,
            sequence=self._sequence,
            payload=payload,
            source=source,
        )

        # Append to JSONL file
        with open(self._events_file, "a", encoding="utf-8") as f:
            f.write(envelope.model_dump_json() + "\n")

        return envelope

    def save_vector(
        self,
        vector: torch.Tensor,
        layer: int,
        version: Optional[int] = None,
    ) -> str:
        """Save vector to .pt file.

        Args:
            vector: The steering vector tensor.
            layer: Layer index.
            version: Optional version number (auto-incremented if None).

        Returns:
            Relative path to the saved file (for event references).
        """
        self._ensure_initialized()

        # Auto-increment version if not specified (thread-safe)
        if version is None:
            with self._vector_versions_lock:
                current = self._vector_versions.get(layer, 0)
                version = current + 1
                self._vector_versions[layer] = version

        filename = f"layer_{layer:02d}_v{version:03d}.pt"
        full_path = self.vectors_path / filename
        torch.save(vector, full_path)

        return f"vectors/{filename}"

    def save_final_vector(self, vector: torch.Tensor) -> str:
        """Save the final selected vector.

        Args:
            vector: The final steering vector.

        Returns:
            Relative path to the saved file.
        """
        self._ensure_initialized()
        full_path = self.vectors_path / "final.pt"
        torch.save(vector, full_path)
        return "vectors/final.pt"

    def load_vector(self, vector_ref: str) -> torch.Tensor:
        """Load vector from reference path.

        Args:
            vector_ref: Relative path from session directory.

        Returns:
            The loaded tensor.
        """
        full_path = self.session_path / vector_ref
        return torch.load(full_path, weights_only=True)

    def save_checkpoint_state(
        self,
        checkpoint_id: str,
        state_data: Dict[str, Any],
    ) -> str:
        """Save checkpoint state to JSON.

        Args:
            checkpoint_id: Unique checkpoint identifier.
            state_data: State data to save.

        Returns:
            Relative path to saved file.
        """
        self._ensure_initialized()
        filename = f"{checkpoint_id}.json"
        full_path = self.checkpoints_path / filename

        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(state_data, f, indent=2, default=str)

        return f"checkpoints/{filename}"

    def load_checkpoint_state(self, state_ref: str) -> Dict[str, Any]:
        """Load checkpoint state from reference.

        Args:
            state_ref: Relative path from session directory.

        Returns:
            The loaded state data.
        """
        full_path = self.session_path / state_ref
        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def iter_events(
        self,
        event_types: Optional[List[str]] = None,
        categories: Optional[List[EventCategory]] = None,
        after_sequence: int = 0,
    ) -> Iterator[EventEnvelope]:
        """Iterate events with optional filtering.

        Args:
            event_types: Filter to specific event types.
            categories: Filter to specific categories.
            after_sequence: Only return events after this sequence.

        Yields:
            Matching event envelopes.
        """
        if not self.events_path.exists():
            return

        with open(self.events_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    envelope = EventEnvelope.model_validate_json(line)
                except Exception:
                    continue  # Skip malformed lines

                # Apply filters
                if envelope.sequence <= after_sequence:
                    continue
                if event_types and envelope.event_type not in event_types:
                    continue
                if categories and envelope.category not in categories:
                    continue

                yield envelope

    def get_all_events(self) -> List[EventEnvelope]:
        """Get all events as a list.

        Returns:
            List of all event envelopes in order.
        """
        return list(self.iter_events())

    def get_event_count(self) -> int:
        """Get total number of events.

        Returns:
            Count of events in the session.
        """
        return sum(1 for _ in self.iter_events())

    def update_metadata(self, **updates: Any) -> None:
        """Update session metadata.

        Args:
            **updates: Fields to update.
        """
        metadata_path = self.session_path / "metadata.json"

        # Load existing
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Update
        metadata.update(updates)
        metadata["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Save
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def get_metadata(self) -> Dict[str, Any]:
        """Get session metadata.

        Returns:
            Metadata dictionary.
        """
        metadata_path = self.session_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def finalize(self, success: bool, error: Optional[str] = None) -> None:
        """Mark session as complete.

        Args:
            success: Whether extraction succeeded.
            error: Error message if failed.
        """
        self.update_metadata(
            status="completed" if success else "failed",
            completed_at=datetime.now(timezone.utc).isoformat(),
            error=error,
        )


class StorageManager:
    """Global storage manager for all extraction sessions.

    Manages session creation, listing, and retrieval.
    Sessions are discovered dynamically by scanning directories.

    Example:
        >>> manager = StorageManager()
        >>> store = manager.create_session("sycophancy", config_dict)
        >>> sessions = manager.list_sessions()
    """

    def __init__(self, base_path: Optional[Path] = None) -> None:
        """Initialize storage manager.

        Args:
            base_path: Base path for sessions. Defaults to ~/.vector-forge/sessions
        """
        if base_path is None:
            base_path = Path.home() / ".vector-forge" / "sessions"

        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def create_session(
        self,
        behavior: str,
        config: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> SessionStore:
        """Create new session and return store.

        Args:
            behavior: Behavior name being extracted.
            config: Pipeline configuration to snapshot.
            session_id: Optional custom session ID.

        Returns:
            Initialized SessionStore.
        """
        if session_id is None:
            session_id = self._generate_session_id(behavior)

        store = SessionStore(session_id, self.base_path)
        store.initialize(behavior, config)

        return store

    def get_session(self, session_id: str) -> SessionStore:
        """Get existing session store.

        Args:
            session_id: Session identifier.

        Returns:
            SessionStore for the session.

        Raises:
            FileNotFoundError: If session doesn't exist.
        """
        session_path = self.base_path / session_id
        if not session_path.exists():
            raise FileNotFoundError(f"Session {session_id} not found")

        store = SessionStore(session_id, self.base_path)
        store._load_sequence()
        store._initialized = True
        store._events_file = store.events_path

        return store

    def list_sessions(
        self,
        status: Optional[str] = None,
        behavior: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """List all sessions by scanning directories.

        Discovers sessions dynamically by reading metadata.json from each
        session folder. No index file needed.

        Args:
            status: Filter by status (running, completed, failed).
            behavior: Filter by behavior name.
            limit: Maximum number of sessions to return.

        Returns:
            List of session metadata dictionaries.
        """
        sessions = []

        # Scan all directories in base_path
        if not self.base_path.exists():
            return sessions

        for entry in self.base_path.iterdir():
            if not entry.is_dir():
                continue

            # Skip hidden directories
            if entry.name.startswith("."):
                continue

            # Skip sessions with .hidden marker file
            if (entry / ".hidden").exists():
                continue

            # Read metadata.json
            metadata_path = entry / "metadata.json"
            if not metadata_path.exists():
                continue

            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                # Ensure session_id is set
                metadata["session_id"] = entry.name

                # Apply filters
                if status and metadata.get("status") != status:
                    continue
                if behavior and metadata.get("behavior") != behavior:
                    continue

                sessions.append(metadata)
            except (json.JSONDecodeError, IOError):
                continue

        # Sort by created_at descending (newest first)
        sessions.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        if limit:
            sessions = sessions[:limit]

        return sessions

    def get_latest_session(self, behavior: Optional[str] = None) -> Optional[SessionStore]:
        """Get the most recent session.

        Args:
            behavior: Optional filter by behavior.

        Returns:
            SessionStore or None if no sessions.
        """
        sessions = self.list_sessions(behavior=behavior, limit=1)
        if sessions:
            return self.get_session(sessions[0]["session_id"])
        return None

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its data.

        Args:
            session_id: Session to delete.

        Returns:
            True if deleted, False if not found.
        """
        import shutil

        session_path = self.base_path / session_id
        if not session_path.exists():
            return False

        shutil.rmtree(session_path)
        return True

    def hide_session(self, session_id: str) -> bool:
        """Hide a session by creating a .hidden marker file.

        The session data is preserved but won't appear in list_sessions().
        Remove the .hidden file to unhide the session.

        Args:
            session_id: Session to hide.

        Returns:
            True if hidden, False if not found.
        """
        session_path = self.base_path / session_id
        if not session_path.exists():
            return False

        hidden_marker = session_path / ".hidden"
        hidden_marker.touch()
        return True

    def unhide_session(self, session_id: str) -> bool:
        """Unhide a session by removing the .hidden marker file.

        Args:
            session_id: Session to unhide.

        Returns:
            True if unhidden, False if not found or not hidden.
        """
        session_path = self.base_path / session_id
        hidden_marker = session_path / ".hidden"

        if not hidden_marker.exists():
            return False

        hidden_marker.unlink()
        return True

    def is_session_hidden(self, session_id: str) -> bool:
        """Check if a session is hidden.

        Args:
            session_id: Session to check.

        Returns:
            True if hidden, False otherwise.
        """
        session_path = self.base_path / session_id
        return (session_path / ".hidden").exists()

    def _generate_session_id(self, behavior: str) -> str:
        """Generate unique session ID.

        Args:
            behavior: Behavior name for prefix.

        Returns:
            Unique session identifier.
        """
        import uuid

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        # Clean behavior name for filesystem
        clean_behavior = "".join(
            c if c.isalnum() else "_" for c in behavior[:15]
        ).lower()
        short_uuid = str(uuid.uuid4())[:8]

        return f"{timestamp}_{clean_behavior}_{short_uuid}"
