"""JSONL file watcher for real-time UI updates.

Uses Textual's native worker pattern to tail JSONL event files
and post thread-safe messages to update the UI.

This replaces the callback/listener pattern with a clean,
single-source-of-truth approach: JSONL is the canonical event log,
and the UI watches it directly.
"""

import time
from pathlib import Path
from typing import List, Optional

from textual.message import Message

from vector_forge.storage.events import EventEnvelope


class NewEvents(Message):
    """Thread-safe message posted when new events are read from JSONL.

    This message is posted via post_message() which is thread-safe,
    allowing the worker thread to safely notify the main UI thread.
    """

    def __init__(self, session_id: str, events: List[EventEnvelope]) -> None:
        super().__init__()
        self.session_id = session_id
        self.events = events


class SessionFileWatcher:
    """Watches a single session's JSONL file for new events.

    Tracks file position and reads only new lines since last check.
    Designed to be called from a worker thread.
    """

    def __init__(self, session_id: str, events_path: Path) -> None:
        self.session_id = session_id
        self.events_path = events_path
        self._position: int = 0
        self._last_size: int = 0

    def check_for_new_events(self) -> List[EventEnvelope]:
        """Check for and read any new events since last check.

        Returns:
            List of new EventEnvelope objects (empty if no new events).
        """
        if not self.events_path.exists():
            return []

        try:
            current_size = self.events_path.stat().st_size

            # No new data
            if current_size <= self._last_size:
                return []

            events = []
            with open(self.events_path, "r", encoding="utf-8") as f:
                f.seek(self._position)

                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        envelope = EventEnvelope.model_validate_json(line)
                        events.append(envelope)
                    except Exception:
                        # Skip malformed lines
                        continue

                self._position = f.tell()

            self._last_size = current_size
            return events

        except Exception:
            return []

    def reset(self) -> None:
        """Reset position to re-read entire file."""
        self._position = 0
        self._last_size = 0


class EventStreamWatcher:
    """Manages watching multiple session JSONL files.

    This is the main interface used by the App's worker thread.
    It tracks multiple sessions and checks them all for new events.

    Usage in App:
        @work(thread=True)
        def watch_events(self) -> None:
            watcher = EventStreamWatcher(base_path)
            while True:
                for session_id, events in watcher.check_all():
                    if events:
                        self.post_message(NewEvents(session_id, events))
                time.sleep(0.1)  # 100ms poll interval
    """

    def __init__(self, sessions_base_path: Path) -> None:
        self.base_path = sessions_base_path
        self._watchers: dict[str, SessionFileWatcher] = {}

    def add_session(self, session_id: str) -> None:
        """Start watching a session's events file."""
        if session_id in self._watchers:
            return

        events_path = self.base_path / session_id / "events.jsonl"
        self._watchers[session_id] = SessionFileWatcher(session_id, events_path)

    def remove_session(self, session_id: str) -> None:
        """Stop watching a session."""
        self._watchers.pop(session_id, None)

    def get_active_sessions(self) -> List[str]:
        """Get list of sessions being watched."""
        return list(self._watchers.keys())

    def check_all(self) -> List[tuple[str, List[EventEnvelope]]]:
        """Check all watched sessions for new events.

        Returns:
            List of (session_id, events) tuples for sessions with new events.
        """
        results = []
        for session_id, watcher in self._watchers.items():
            events = watcher.check_for_new_events()
            if events:
                results.append((session_id, events))
        return results

    def discover_sessions(self) -> List[str]:
        """Discover all session directories in base path.

        Returns:
            List of session IDs found.
        """
        if not self.base_path.exists():
            return []

        sessions = []
        for entry in self.base_path.iterdir():
            if not entry.is_dir():
                continue
            if entry.name.startswith("."):
                continue
            if (entry / ".hidden").exists():
                continue
            if (entry / "events.jsonl").exists():
                sessions.append(entry.name)

        return sessions

    def sync_with_disk(self) -> None:
        """Discover new sessions and add watchers for them."""
        discovered = set(self.discover_sessions())
        current = set(self._watchers.keys())

        # Add new sessions
        for session_id in discovered - current:
            self.add_session(session_id)


def apply_event_to_state(
    state: "UIState",
    session_id: str,
    event: EventEnvelope,
) -> None:
    """Apply a single event to UI state.

    This is the core event handler that updates UIState based on
    events read from JSONL. Runs on main thread via message handler.

    Args:
        state: The UI state to update.
        session_id: The session this event belongs to.
        event: The event to apply.
    """
    from vector_forge.ui.state import (
        ExtractionUIState,
        ExtractionStatus,
        Phase,
        AgentUIState,
        AgentStatus,
        MessageRole,
        ToolCall,
        LogEntry,
    )
    from vector_forge.storage.log_builder import build_log_entry

    extraction = state.extractions.get(session_id)
    payload = event.payload
    event_type = event.event_type

    # Add log entry for significant events
    log_data = build_log_entry(event_type, payload, event.source)
    if log_data:
        source, message, level = log_data
        state.logs.append(LogEntry(
            timestamp=event.timestamp.timestamp(),
            source=source,
            message=message,
            level=level,
            extraction_id=session_id,
            event_type=event_type,
            payload=payload,
        ))
        # Trim logs
        if len(state.logs) > 10000:
            state.logs = state.logs[-10000:]

    # Session lifecycle
    if event_type == "session.started":
        if extraction is None:
            extraction = ExtractionUIState(
                id=session_id,
                behavior_name=payload.get("behavior_name", "Unknown"),
                behavior_description=payload.get("behavior_description", ""),
                status=ExtractionStatus.RUNNING,
                phase=Phase.INITIALIZING,
                progress=0.0,
                started_at=event.timestamp.timestamp(),
            )
            config = payload.get("config", {})
            extraction.max_outer_iterations = config.get("num_samples", 16)
            extraction.model = config.get("extractor_model", "")
            extraction.target_model = config.get("target_model", "")
            state.extractions[session_id] = extraction
            if state.selected_id is None:
                state.selected_id = session_id
        return

    if extraction is None:
        return

    if event_type == "session.completed":
        success = payload.get("success", False)
        extraction.status = ExtractionStatus.COMPLETE if success else ExtractionStatus.FAILED
        extraction.phase = Phase.COMPLETE if success else Phase.FAILED
        extraction.progress = 1.0
        extraction.completed_at = event.timestamp.timestamp()
        if payload.get("final_score") is not None:
            extraction.evaluation.overall = payload["final_score"]
        if payload.get("final_layer") is not None:
            extraction.evaluation.best_layer = payload["final_layer"]

    # Contrast pipeline events
    elif event_type == "contrast.pipeline_started":
        extraction.phase = Phase.GENERATING_DATAPOINTS
        extraction.max_outer_iterations = payload.get("num_samples", extraction.max_outer_iterations)

    elif event_type == "seed.assigned":
        sample_idx = payload.get("sample_idx", 0)
        num_core = payload.get("num_core_seeds", 0)
        num_unique = payload.get("num_unique_seeds", 0)
        total_pairs = num_core + num_unique

        agent = _get_or_create_sample_agent(extraction, sample_idx)
        agent.status = AgentStatus.WAITING
        agent.tool_calls_count = total_pairs
        agent.add_message(
            MessageRole.SYSTEM,
            f"Assigned {num_core} core + {num_unique} unique seeds ({total_pairs} total pairs)"
        )

    elif event_type == "contrast.pair_validated":
        is_valid = payload.get("is_valid", False)
        if is_valid:
            extraction.datapoints.total += 1
            extraction.datapoints.keep += 1

    # Optimization events
    elif event_type == "optimization.started":
        sample_idx = payload.get("sample_idx", 0)
        layer = payload.get("layer", 0)
        config = payload.get("config", {})

        agent = _get_or_create_sample_agent(extraction, sample_idx, layer, config)
        agent.status = AgentStatus.RUNNING
        agent.started_at = event.timestamp.timestamp()
        agent.add_message(
            MessageRole.SYSTEM,
            f"Optimizing steering vector on layer {layer} with {payload.get('num_datapoints', 0)} datapoints"
        )
        extraction.phase = Phase.OPTIMIZING

    elif event_type == "optimization.progress":
        sample_idx = payload.get("sample_idx", 0)
        iteration = payload.get("iteration", 0)
        loss = payload.get("loss", 0.0)

        agent = _get_sample_agent(extraction, sample_idx)
        if agent:
            agent.current_tool = f"iter {iteration} loss={loss:.4f}"

    elif event_type == "optimization.completed":
        sample_idx = payload.get("sample_idx", 0)
        success = payload.get("success", True)
        final_loss = payload.get("final_loss")
        iterations = payload.get("iterations", 0)
        duration = payload.get("duration_seconds", 0.0)
        loss_history = payload.get("loss_history", [])
        error = payload.get("error")

        # Detect CAA: no loss history and single iteration
        is_caa = not loss_history and iterations == 1

        agent = _get_sample_agent(extraction, sample_idx)
        if agent:
            agent.status = AgentStatus.COMPLETE if success else AgentStatus.ERROR
            agent.completed_at = event.timestamp.timestamp()
            agent.current_tool = None
            if success:
                if is_caa:
                    agent.add_message(
                        MessageRole.ASSISTANT,
                        f"Extraction complete in {duration:.1f}s"
                    )
                else:
                    loss_str = f"{final_loss:.4f}" if final_loss is not None else "N/A"
                    agent.add_message(
                        MessageRole.ASSISTANT,
                        f"Optimization complete: loss={loss_str}, {iterations} iterations in {duration:.1f}s"
                    )
            else:
                agent.add_message(
                    MessageRole.ASSISTANT,
                    f"Optimization failed: {error or 'Unknown error'}"
                )

    # LLM events for source-based agents
    elif event_type == "llm.request":
        source = event.source
        if source and not source.startswith("sample_"):
            agent = _get_or_create_source_agent(extraction, source)
            agent.status = AgentStatus.RUNNING
            messages = payload.get("messages", [])
            if messages:
                last_msg = messages[-1] if messages else {}
                content = last_msg.get("content", "")
                if content:
                    agent.add_message(MessageRole.USER, content)

    elif event_type == "llm.response":
        source = event.source
        if source and not source.startswith("sample_"):
            agent = _get_or_create_source_agent(extraction, source)
            tool_calls = []
            for tc in payload.get("tool_calls", []):
                tool_calls.append(ToolCall(
                    id=tc.get("id", ""),
                    name=tc.get("function", {}).get("name", ""),
                    arguments=str(tc.get("function", {}).get("arguments", "")),
                    status="pending",
                ))
            content = payload.get("content", "")
            if content or tool_calls:
                agent.add_message(
                    MessageRole.ASSISTANT,
                    content or "(tool call)",
                    tool_calls=tool_calls,
                )
                agent.tool_calls_count += len(tool_calls)

    # Vector events
    elif event_type == "vector.created":
        extraction.current_layer = payload.get("layer", 0)
        extraction.phase = Phase.OPTIMIZING
        extraction.progress = max(extraction.progress, 0.5)

    elif event_type == "vector.selected":
        extraction.current_layer = payload.get("layer", extraction.current_layer)
        extraction.evaluation.best_layer = payload.get("layer")
        extraction.evaluation.best_strength = payload.get("strength", 1.0)

    # Evaluation events
    elif event_type == "evaluation.started":
        extraction.phase = Phase.EVALUATING
        extraction.progress = max(extraction.progress, 0.7)

        # Update sample status if sample_idx is provided
        sample_idx = payload.get("sample_idx")
        if sample_idx is not None:
            agent = _get_sample_agent(extraction, sample_idx)
            if agent:
                agent.status = AgentStatus.RUNNING

    elif event_type == "evaluation.completed":
        scores = payload.get("scores", {})
        extraction.evaluation.behavior = scores.get("behavior", 0.0)
        extraction.evaluation.coherence = scores.get("coherence", 0.0)
        extraction.evaluation.specificity = scores.get("specificity", 0.0)
        extraction.evaluation.overall = scores.get("overall", 0.0)
        extraction.evaluation.verdict = payload.get("verdict", "")
        extraction.evaluation.best_strength = payload.get("recommended_strength", 1.0)
        extraction.phase = Phase.JUDGE_REVIEW
        extraction.progress = max(extraction.progress, 0.9)

        # Update sample status if sample_idx is provided
        sample_idx = payload.get("sample_idx")
        if sample_idx is not None:
            agent = _get_sample_agent(extraction, sample_idx)
            if agent:
                verdict = payload.get("verdict", "")
                if verdict == "failed":
                    agent.status = AgentStatus.ERROR
                else:
                    agent.status = AgentStatus.COMPLETE
                agent.completed_at = event.timestamp.timestamp()


def _get_or_create_sample_agent(
    extraction: "ExtractionUIState",
    sample_idx: int,
    layer: int = 0,
    config: dict = None,
) -> "AgentUIState":
    """Get or create a per-sample agent."""
    from vector_forge.ui.state import AgentUIState, AgentStatus
    import time

    agent_id = f"{extraction.id}_sample_{sample_idx}"

    if agent_id in extraction.agents:
        return extraction.agents[agent_id]

    if config:
        lr = config.get("lr", 0.01)
        seed = config.get("seed", 0)
        role = f"L{layer} lr={lr:.3f} seed={seed}"
    else:
        role = f"sample {sample_idx}"

    agent = AgentUIState(
        id=agent_id,
        name=f"Sample {sample_idx + 1}",
        role=role,
        status=AgentStatus.IDLE,
        started_at=time.time(),
    )
    extraction.add_agent(agent)
    return agent


def _get_sample_agent(
    extraction: "ExtractionUIState",
    sample_idx: int,
) -> Optional["AgentUIState"]:
    """Get a sample agent if it exists."""
    agent_id = f"{extraction.id}_sample_{sample_idx}"
    return extraction.agents.get(agent_id)


def _get_or_create_source_agent(
    extraction: "ExtractionUIState",
    source: str,
) -> "AgentUIState":
    """Get or create an agent for a source (extractor, judge, etc.)."""
    from vector_forge.ui.state import AgentUIState, AgentStatus
    import time

    agent_id = f"{extraction.id}_{source}"

    if agent_id in extraction.agents:
        return extraction.agents[agent_id]

    agent = AgentUIState(
        id=agent_id,
        name=source.replace("_", " ").title(),
        role=source,
        status=AgentStatus.IDLE,
        started_at=time.time(),
    )
    extraction.add_agent(agent)
    return agent
