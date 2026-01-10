"""UI State Synchronizer for bridging storage events to UI state.

Translates storage events into UI state updates, maintaining consistency
between the persistent event log and the reactive UI state.
"""

import logging
import time
from typing import Any, Dict, Optional

from vector_forge.storage import (
    EventEnvelope,
    EventCategory,
)
from vector_forge.services.session import SessionService, SessionInfo
from vector_forge.ui.state import (
    UIState,
    ExtractionUIState,
    ExtractionStatus,
    Phase,
    AgentUIState,
    AgentStatus,
    MessageRole,
    ToolCall,
    DatapointMetrics,
    EvaluationMetrics,
    get_state,
)

logger = logging.getLogger(__name__)


class UIStateSynchronizer:
    """Bridges storage events to UI state updates.

    Subscribes to SessionService events and translates them into
    appropriate UIState updates. Also handles loading existing
    sessions on application startup.

    Example:
        >>> ui_state = get_state()
        >>> session_service = SessionService()
        >>> sync = UIStateSynchronizer(ui_state, session_service)
        >>> sync.load_existing_sessions()
        >>> # Now UI shows all existing sessions
    """

    def __init__(
        self,
        ui_state: UIState,
        session_service: SessionService,
    ) -> None:
        """Initialize the synchronizer.

        Args:
            ui_state: The UI state to update.
            session_service: The session service to listen to.
        """
        self._ui_state = ui_state
        self._session_service = session_service

        # Map session_id to extraction_id (they're the same for now)
        self._session_to_extraction: Dict[str, str] = {}

        # Register as event listener
        self._session_service.add_event_listener(self._on_event)

    def load_existing_sessions(self, limit: int = 50) -> int:
        """Load existing sessions from storage into UI state.

        Populates the UI state with sessions from disk, allowing
        the dashboard to show previous extractions.

        Args:
            limit: Maximum sessions to load.

        Returns:
            Number of sessions loaded.
        """
        sessions = self._session_service.list_sessions(limit=limit)

        loaded = 0
        for session_info in sessions:
            try:
                extraction = self._create_extraction_from_session(session_info)
                self._ui_state.extractions[extraction.id] = extraction
                self._session_to_extraction[session_info.session_id] = extraction.id
                loaded += 1
            except Exception as e:
                logger.warning(
                    f"Failed to load session {session_info.session_id}: {e}"
                )

        # Select first if none selected
        if self._ui_state.selected_id is None and self._ui_state.extractions:
            self._ui_state.selected_id = next(iter(self._ui_state.extractions))

        logger.info(f"Loaded {loaded} sessions into UI state")
        return loaded

    def _create_extraction_from_session(
        self,
        session_info: SessionInfo,
    ) -> ExtractionUIState:
        """Create an ExtractionUIState from session info.

        Args:
            session_info: Session information.

        Returns:
            ExtractionUIState for UI display.
        """
        # Map status
        status_map = {
            "running": ExtractionStatus.RUNNING,
            "completed": ExtractionStatus.COMPLETE,
            "failed": ExtractionStatus.FAILED,
        }
        status = status_map.get(session_info.status, ExtractionStatus.PENDING)

        # Determine phase from status
        if status == ExtractionStatus.COMPLETE:
            phase = Phase.COMPLETE
        elif status == ExtractionStatus.FAILED:
            phase = Phase.FAILED
        elif status == ExtractionStatus.RUNNING:
            phase = Phase.GENERATING_DATAPOINTS  # Will be updated by events
        else:
            phase = Phase.INITIALIZING

        # Calculate progress from status
        progress = 0.0
        if status == ExtractionStatus.COMPLETE:
            progress = 1.0
        elif status == ExtractionStatus.RUNNING:
            progress = 0.5  # Will be updated by events

        # Get timing
        started_at = session_info.created_at.timestamp() if session_info.created_at else None
        completed_at = session_info.completed_at.timestamp() if session_info.completed_at else None

        extraction = ExtractionUIState(
            id=session_info.session_id,
            behavior_name=session_info.behavior,
            behavior_description="",  # Will be enriched from session data
            status=status,
            phase=phase,
            progress=progress,
            started_at=started_at,
            completed_at=completed_at,
        )

        # Set evaluation score if available
        if session_info.final_score is not None:
            extraction.evaluation.overall = session_info.final_score

        return extraction

    def _on_event(self, session_id: str, event: EventEnvelope) -> None:
        """Handle incoming storage event.

        Args:
            session_id: The session that emitted the event.
            event: The event envelope.
        """
        try:
            # Get or create extraction ID mapping
            if session_id not in self._session_to_extraction:
                # This is a new session - handle session.started
                if event.event_type == "session.started":
                    self._handle_session_started(session_id, event)
                    return
                else:
                    # Session not tracked yet, skip
                    logger.debug(f"Event for unknown session {session_id}: {event.event_type}")
                    return

            extraction_id = self._session_to_extraction[session_id]

            # Route event to appropriate handler
            handlers = {
                "session.started": self._handle_session_started,
                "session.completed": self._handle_session_completed,
                "llm.request": self._handle_llm_request,
                "llm.response": self._handle_llm_response,
                "tool.call": self._handle_tool_call,
                "tool.result": self._handle_tool_result,
                "datapoint.added": self._handle_datapoint_added,
                "datapoint.removed": self._handle_datapoint_removed,
                "vector.created": self._handle_vector_created,
                "vector.selected": self._handle_vector_selected,
                "evaluation.started": self._handle_evaluation_started,
                "evaluation.completed": self._handle_evaluation_completed,
                "state.iteration_started": self._handle_iteration_started,
                "state.iteration_completed": self._handle_iteration_completed,
            }

            handler = handlers.get(event.event_type)
            if handler:
                handler(session_id, event)
            else:
                logger.debug(f"Unhandled event type: {event.event_type}")

        except Exception as e:
            logger.error(f"Error handling event {event.event_type}: {e}")

    def _handle_session_started(self, session_id: str, event: EventEnvelope) -> None:
        """Handle session.started event."""
        payload = event.payload

        extraction = ExtractionUIState(
            id=session_id,
            behavior_name=payload.get("behavior_name", "Unknown"),
            behavior_description=payload.get("behavior_description", ""),
            status=ExtractionStatus.RUNNING,
            phase=Phase.INITIALIZING,
            progress=0.0,
            started_at=event.timestamp.timestamp(),
        )

        # Extract config details
        config = payload.get("config", {})
        extraction.max_outer_iterations = config.get("num_samples", 16)
        extraction.model = config.get("extractor_model", "")

        self._ui_state.add_extraction(extraction)
        self._session_to_extraction[session_id] = session_id

        # Log the start
        self._ui_state.add_log(
            source="session",
            message=f"Started extraction: {extraction.behavior_name}",
            level="info",
            extraction_id=session_id,
        )

    def _handle_session_completed(self, session_id: str, event: EventEnvelope) -> None:
        """Handle session.completed event."""
        payload = event.payload
        extraction_id = self._session_to_extraction.get(session_id)
        if not extraction_id:
            return

        extraction = self._ui_state.extractions.get(extraction_id)
        if not extraction:
            return

        success = payload.get("success", False)
        extraction.status = ExtractionStatus.COMPLETE if success else ExtractionStatus.FAILED
        extraction.phase = Phase.COMPLETE if success else Phase.FAILED
        extraction.progress = 1.0
        extraction.completed_at = event.timestamp.timestamp()

        # Update evaluation
        if payload.get("final_score") is not None:
            extraction.evaluation.overall = payload["final_score"]
        if payload.get("final_layer") is not None:
            extraction.evaluation.best_layer = payload["final_layer"]

        self._ui_state._notify()

        # Log completion
        status_text = "completed successfully" if success else "failed"
        score_text = f" (score: {payload.get('final_score', 0):.2f})" if success else ""
        self._ui_state.add_log(
            source="session",
            message=f"Extraction {status_text}{score_text}",
            level="info" if success else "error",
            extraction_id=session_id,
        )

    def _handle_llm_request(self, session_id: str, event: EventEnvelope) -> None:
        """Handle llm.request event."""
        extraction_id = self._session_to_extraction.get(session_id)
        if not extraction_id:
            return

        extraction = self._ui_state.extractions.get(extraction_id)
        if not extraction:
            return

        payload = event.payload
        source = event.source

        # Get or create agent
        agent = self._get_or_create_agent(extraction, source)
        agent.status = AgentStatus.RUNNING

        # Log the request
        self._ui_state.add_log(
            source=source,
            message=f"LLM request: {payload.get('model', 'unknown')}",
            level="info",
            extraction_id=session_id,
            agent_id=agent.id,
        )

    def _handle_llm_response(self, session_id: str, event: EventEnvelope) -> None:
        """Handle llm.response event."""
        extraction_id = self._session_to_extraction.get(session_id)
        if not extraction_id:
            return

        extraction = self._ui_state.extractions.get(extraction_id)
        if not extraction:
            return

        payload = event.payload
        source = event.source

        agent = self._get_or_create_agent(extraction, source)

        # Parse tool calls from response
        tool_calls = []
        for tc in payload.get("tool_calls", []):
            tool_calls.append(ToolCall(
                id=tc.get("id", ""),
                name=tc.get("function", {}).get("name", ""),
                arguments=str(tc.get("function", {}).get("arguments", "")),
                status="pending",
            ))

        # Add assistant message
        content = payload.get("content", "")
        if content or tool_calls:
            agent.add_message(
                role=MessageRole.ASSISTANT,
                content=content or "(tool call)",
                tool_calls=tool_calls,
            )
            agent.tool_calls_count += len(tool_calls)

        self._ui_state._notify()

    def _handle_tool_call(self, session_id: str, event: EventEnvelope) -> None:
        """Handle tool.call event."""
        extraction_id = self._session_to_extraction.get(session_id)
        if not extraction_id:
            return

        extraction = self._ui_state.extractions.get(extraction_id)
        if not extraction:
            return

        payload = event.payload
        tool_name = payload.get("tool_name", "unknown")

        # Log the tool call
        self._ui_state.add_log(
            source="tool",
            message=f"Calling: {tool_name}",
            level="info",
            extraction_id=session_id,
        )

        # Update phase based on tool
        if "datapoint" in tool_name.lower() or "generate" in tool_name.lower():
            extraction.phase = Phase.GENERATING_DATAPOINTS
        elif "optimize" in tool_name.lower() or "vector" in tool_name.lower():
            extraction.phase = Phase.OPTIMIZING
        elif "evaluate" in tool_name.lower() or "judge" in tool_name.lower():
            extraction.phase = Phase.EVALUATING

        self._ui_state._notify()

    def _handle_tool_result(self, session_id: str, event: EventEnvelope) -> None:
        """Handle tool.result event."""
        extraction_id = self._session_to_extraction.get(session_id)
        if not extraction_id:
            return

        payload = event.payload
        success = payload.get("success", True)

        if not success:
            error = payload.get("error", "Unknown error")
            self._ui_state.add_log(
                source="tool",
                message=f"Tool failed: {error}",
                level="error",
                extraction_id=session_id,
            )

    def _handle_datapoint_added(self, session_id: str, event: EventEnvelope) -> None:
        """Handle datapoint.added event."""
        extraction_id = self._session_to_extraction.get(session_id)
        if not extraction_id:
            return

        extraction = self._ui_state.extractions.get(extraction_id)
        if not extraction:
            return

        extraction.datapoints.total += 1
        extraction.datapoints.keep += 1
        extraction.phase = Phase.GENERATING_DATAPOINTS

        # Update progress (datapoint generation is ~30% of total)
        if extraction.max_outer_iterations > 0:
            dp_progress = min(extraction.datapoints.total / 50, 1.0) * 0.3
            extraction.progress = dp_progress

        self._ui_state._notify()

    def _handle_datapoint_removed(self, session_id: str, event: EventEnvelope) -> None:
        """Handle datapoint.removed event."""
        extraction_id = self._session_to_extraction.get(session_id)
        if not extraction_id:
            return

        extraction = self._ui_state.extractions.get(extraction_id)
        if not extraction:
            return

        extraction.datapoints.remove += 1
        extraction.datapoints.keep = max(0, extraction.datapoints.keep - 1)
        self._ui_state._notify()

    def _handle_vector_created(self, session_id: str, event: EventEnvelope) -> None:
        """Handle vector.created event."""
        extraction_id = self._session_to_extraction.get(session_id)
        if not extraction_id:
            return

        extraction = self._ui_state.extractions.get(extraction_id)
        if not extraction:
            return

        payload = event.payload
        layer = payload.get("layer", 0)

        extraction.current_layer = layer
        extraction.phase = Phase.OPTIMIZING
        extraction.progress = max(extraction.progress, 0.5)

        self._ui_state.add_log(
            source="vector",
            message=f"Created vector at layer {layer}",
            level="info",
            extraction_id=session_id,
        )

        self._ui_state._notify()

    def _handle_vector_selected(self, session_id: str, event: EventEnvelope) -> None:
        """Handle vector.selected event."""
        extraction_id = self._session_to_extraction.get(session_id)
        if not extraction_id:
            return

        extraction = self._ui_state.extractions.get(extraction_id)
        if not extraction:
            return

        payload = event.payload
        extraction.current_layer = payload.get("layer", extraction.current_layer)
        extraction.evaluation.best_layer = payload.get("layer")
        extraction.evaluation.best_strength = payload.get("strength", 1.0)

        self._ui_state.add_log(
            source="vector",
            message=f"Selected: layer {payload.get('layer')} @ strength {payload.get('strength', 1.0):.1f}",
            level="info",
            extraction_id=session_id,
        )

        self._ui_state._notify()

    def _handle_evaluation_started(self, session_id: str, event: EventEnvelope) -> None:
        """Handle evaluation.started event."""
        extraction_id = self._session_to_extraction.get(session_id)
        if not extraction_id:
            return

        extraction = self._ui_state.extractions.get(extraction_id)
        if not extraction:
            return

        extraction.phase = Phase.EVALUATING
        extraction.progress = max(extraction.progress, 0.6)

        self._ui_state.add_log(
            source="evaluation",
            message="Evaluation started",
            level="info",
            extraction_id=session_id,
        )

        self._ui_state._notify()

    def _handle_evaluation_completed(self, session_id: str, event: EventEnvelope) -> None:
        """Handle evaluation.completed event."""
        extraction_id = self._session_to_extraction.get(session_id)
        if not extraction_id:
            return

        extraction = self._ui_state.extractions.get(extraction_id)
        if not extraction:
            return

        payload = event.payload
        scores = payload.get("scores", {})

        extraction.evaluation.behavior = scores.get("behavior", 0.0)
        extraction.evaluation.coherence = scores.get("coherence", 0.0)
        extraction.evaluation.specificity = scores.get("specificity", 0.0)
        extraction.evaluation.overall = scores.get("overall", 0.0)
        extraction.evaluation.verdict = payload.get("verdict", "")
        extraction.evaluation.best_strength = payload.get("recommended_strength", 1.0)

        extraction.phase = Phase.JUDGE_REVIEW
        extraction.progress = max(extraction.progress, 0.9)

        self._ui_state.add_log(
            source="evaluation",
            message=f"Evaluation complete: {extraction.evaluation.overall:.2f}",
            level="info",
            extraction_id=session_id,
        )

        self._ui_state._notify()

    def _handle_iteration_started(self, session_id: str, event: EventEnvelope) -> None:
        """Handle state.iteration_started event."""
        extraction_id = self._session_to_extraction.get(session_id)
        if not extraction_id:
            return

        extraction = self._ui_state.extractions.get(extraction_id)
        if not extraction:
            return

        payload = event.payload
        iteration_type = payload.get("iteration_type", "outer")
        iteration = payload.get("iteration", 0)

        if iteration_type == "outer":
            extraction.outer_iteration = iteration
        else:
            extraction.inner_turn = iteration

        self._ui_state._notify()

    def _handle_iteration_completed(self, session_id: str, event: EventEnvelope) -> None:
        """Handle state.iteration_completed event."""
        extraction_id = self._session_to_extraction.get(session_id)
        if not extraction_id:
            return

        extraction = self._ui_state.extractions.get(extraction_id)
        if not extraction:
            return

        # Update progress based on iteration
        if extraction.max_outer_iterations > 0:
            outer_progress = extraction.outer_iteration / extraction.max_outer_iterations
            extraction.progress = 0.3 + (outer_progress * 0.6)  # 30-90% for iterations

        self._ui_state._notify()

    def _get_or_create_agent(
        self,
        extraction: ExtractionUIState,
        source: str,
    ) -> AgentUIState:
        """Get or create an agent for the given source.

        Args:
            extraction: The extraction state.
            source: The source component name (e.g., "extractor", "judge").

        Returns:
            AgentUIState for this source.
        """
        agent_id = f"{extraction.id}_{source}"

        if agent_id in extraction.agents:
            return extraction.agents[agent_id]

        # Create new agent
        agent = AgentUIState(
            id=agent_id,
            name=source.title(),
            role=source,
            status=AgentStatus.IDLE,
            started_at=time.time(),
        )
        extraction.add_agent(agent)

        return agent

    def register_session(self, session_id: str, extraction_id: str) -> None:
        """Manually register a session-to-extraction mapping.

        Useful when creating sessions outside the normal flow.

        Args:
            session_id: The storage session ID.
            extraction_id: The UI extraction ID.
        """
        self._session_to_extraction[session_id] = extraction_id

    def unregister_session(self, session_id: str) -> None:
        """Remove a session mapping.

        Args:
            session_id: The session to unregister.
        """
        if session_id in self._session_to_extraction:
            del self._session_to_extraction[session_id]

    def disconnect(self) -> None:
        """Disconnect from the session service."""
        self._session_service.remove_event_listener(self._on_event)
