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
    SessionReplayer,
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
    LogEntry,
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
        """Create an ExtractionUIState from session info by replaying events.

        Args:
            session_info: Session information.

        Returns:
            ExtractionUIState for UI display with full state from events.
        """
        # Get the session store and replay events
        try:
            store = self._session_service.get_session_store(session_info.session_id)
            replayer = SessionReplayer(store)
            replayed = replayer.reconstruct_state()
        except Exception as e:
            logger.warning(f"Failed to replay session {session_info.session_id}: {e}")
            replayed = None

        # Map status
        status_map = {
            "running": ExtractionStatus.RUNNING,
            "completed": ExtractionStatus.COMPLETE,
            "failed": ExtractionStatus.FAILED,
        }
        status = status_map.get(session_info.status, ExtractionStatus.PENDING)

        # Determine phase from replayed state or status
        if replayed:
            phase_map = {
                "initializing": Phase.INITIALIZING,
                "generating": Phase.GENERATING_DATAPOINTS,
                "extracting": Phase.OPTIMIZING,
                "evaluating": Phase.EVALUATING,
                "complete": Phase.COMPLETE,
                "failed": Phase.FAILED,
            }
            phase = phase_map.get(replayed.current_phase, Phase.INITIALIZING)
            progress = replayed.progress
        else:
            if status == ExtractionStatus.COMPLETE:
                phase = Phase.COMPLETE
                progress = 1.0
            elif status == ExtractionStatus.FAILED:
                phase = Phase.FAILED
                progress = 0.0
            elif status == ExtractionStatus.RUNNING:
                phase = Phase.GENERATING_DATAPOINTS
                progress = 0.5
            else:
                phase = Phase.INITIALIZING
                progress = 0.0

        # Get timing
        started_at = session_info.created_at.timestamp() if session_info.created_at else None
        completed_at = session_info.completed_at.timestamp() if session_info.completed_at else None

        # Build datapoint metrics from replayed state
        datapoints = DatapointMetrics()
        if replayed:
            datapoints.total = replayed.datapoint_count
            datapoints.keep = replayed.datapoint_count  # All are kept unless removed

        # Build evaluation metrics from replayed state
        evaluation = EvaluationMetrics()
        if replayed:
            evaluation.overall = replayed.best_score
            evaluation.best_layer = replayed.best_layer
            evaluation.best_strength = replayed.best_strength
            if replayed.evaluations:
                last_eval = replayed.evaluations[-1]
                evaluation.behavior = last_eval.scores.get("behavior", 0.0)
                evaluation.coherence = last_eval.scores.get("coherence", 0.0)
                evaluation.specificity = last_eval.scores.get("specificity", 0.0)
                evaluation.verdict = last_eval.verdict
        elif session_info.final_score is not None:
            evaluation.overall = session_info.final_score

        # Create extraction state
        extraction = ExtractionUIState(
            id=session_info.session_id,
            behavior_name=replayed.behavior_name if replayed and replayed.behavior_name else session_info.behavior,
            behavior_description=replayed.behavior_description if replayed else "",
            status=status,
            phase=phase,
            progress=progress,
            started_at=started_at,
            completed_at=completed_at,
            current_layer=replayed.best_layer if replayed else None,
            datapoints=datapoints,
            evaluation=evaluation,
        )

        # Reconstruct agents from replayed optimizations (sample agents)
        if replayed and replayed.optimizations:
            for opt in replayed.optimizations:
                sample_idx = opt.sample_idx or 0
                layer = opt.layer or 0
                iterations = opt.iterations or 0
                final_loss = opt.final_loss or 0.0
                duration = opt.duration_seconds or 0.0

                agent_id = f"{extraction.id}_sample_{sample_idx}"
                agent = AgentUIState(
                    id=agent_id,
                    name=f"Sample {sample_idx + 1}",
                    role=f"L{layer} iters={iterations}",
                    status=AgentStatus.COMPLETE if opt.success else AgentStatus.ERROR,
                    started_at=started_at,
                    completed_at=completed_at,
                    turns=1,
                    tool_calls_count=iterations,
                )
                # Add completion message
                if opt.success:
                    agent.add_message(
                        MessageRole.ASSISTANT,
                        f"Optimization complete: loss={final_loss:.4f}, "
                        f"{iterations} iterations in {duration:.1f}s"
                    )
                else:
                    agent.add_message(
                        MessageRole.ASSISTANT,
                        f"Optimization failed: {opt.error or 'Unknown error'}"
                    )
                extraction.add_agent(agent)

        # Reconstruct sample agents from contrast pairs (if not already created from optimizations)
        if replayed and replayed.contrast_pairs:
            # Count pairs per sample (excluding core pool with sample_idx=-1)
            pairs_per_sample: dict[int, int] = {}
            for pair in replayed.contrast_pairs.values():
                idx = pair.sample_idx
                if idx >= 0:  # Skip core pool
                    pairs_per_sample[idx] = pairs_per_sample.get(idx, 0) + 1

            # Create or update sample agents with pair counts
            for sample_idx, pair_count in pairs_per_sample.items():
                agent_id = f"{extraction.id}_sample_{sample_idx}"
                if agent_id in extraction.agents:
                    # Update existing agent's pair count
                    extraction.agents[agent_id].tool_calls_count = pair_count
                else:
                    # Create new agent for contrast phase
                    agent = AgentUIState(
                        id=agent_id,
                        name=f"Sample {sample_idx + 1}",
                        role=f"contrast pairs",
                        status=AgentStatus.COMPLETE,
                        started_at=started_at,
                        completed_at=completed_at,
                        turns=1,
                        tool_calls_count=pair_count,
                    )
                    agent.add_message(
                        MessageRole.ASSISTANT,
                        f"Generated {pair_count} contrast pairs"
                    )
                    extraction.add_agent(agent)

        # Reconstruct source-based agents from LLM events in logs
        # These are agents like "extractor", "judge", "contrast_extractor", etc.
        if replayed and replayed.logs:
            source_agents: dict[str, AgentUIState] = {}

            for log in replayed.logs:
                # Only process LLM events to reconstruct agents
                if log.event_type in ("llm.request", "llm.response"):
                    # Source is on the log entry itself, not in payload
                    source = log.source or ""
                    if not source or source.startswith("sample_"):
                        continue  # Skip sample agents, already handled above

                    # Create or get agent for this source
                    if source not in source_agents:
                        agent_id = f"{extraction.id}_{source}"
                        # Format display name
                        display_name = source.replace("_", " ").title()
                        agent = AgentUIState(
                            id=agent_id,
                            name=display_name,
                            role=source,
                            status=AgentStatus.COMPLETE,
                            started_at=started_at,
                            completed_at=completed_at,
                            turns=0,
                            tool_calls_count=0,
                        )
                        source_agents[source] = agent

                    agent = source_agents[source]

                    # Get event timestamp
                    event_ts = log.timestamp.timestamp()

                    # Add message based on event type
                    if log.event_type == "llm.request":
                        agent.turns += 1
                        # Extract prompt summary
                        messages = log.payload.get("messages", []) if log.payload else []
                        if messages:
                            last_msg = messages[-1] if messages else {}
                            content = last_msg.get("content", "")
                            if isinstance(content, str) and len(content) > 100:
                                content = content[:97] + "..."
                            agent.add_message(
                                MessageRole.USER,
                                content or "LLM request",
                                timestamp=event_ts,
                            )

                    elif log.event_type == "llm.response":
                        # Parse tool calls from response (same as live handler)
                        tool_calls = []
                        for tc in log.payload.get("tool_calls", []) if log.payload else []:
                            tool_calls.append(ToolCall(
                                id=tc.get("id", ""),
                                name=tc.get("function", {}).get("name", ""),
                                arguments=str(tc.get("function", {}).get("arguments", "")),
                                status="success",  # Replayed = already completed
                            ))

                        # Extract response content
                        content = log.payload.get("content", "") if log.payload else ""

                        # Add message with tool calls
                        if content or tool_calls:
                            agent.add_message(
                                MessageRole.ASSISTANT,
                                content or "(tool call)",
                                tool_calls=tool_calls,
                                timestamp=event_ts,
                            )
                            agent.tool_calls_count += len(tool_calls)

            # Add all source-based agents to extraction
            for agent in source_agents.values():
                extraction.add_agent(agent)

        # Add logs from replayed events to global UI state
        if replayed and replayed.logs:
            for log in replayed.logs:  # All logs
                self._ui_state.logs.append(LogEntry(
                    timestamp=log.timestamp.timestamp(),
                    source=log.source,
                    message=log.message,
                    level=log.level,
                    extraction_id=session_info.session_id,
                    event_type=log.event_type if log.event_type else None,
                    payload=log.payload if log.payload else None,
                ))

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
                "llm.chunk": self._handle_llm_chunk,
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
                # Per-sample tracking events
                "optimization.started": self._handle_optimization_started,
                "optimization.progress": self._handle_optimization_progress,
                "optimization.completed": self._handle_optimization_completed,
                "contrast.pipeline_started": self._handle_contrast_pipeline_started,
                "contrast.pair_generated": self._handle_contrast_pair_generated,
                "contrast.pair_validated": self._handle_contrast_pair_validated,
                "seed.assigned": self._handle_seed_assigned,
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

        # Add user message with prompt content for visibility
        messages = payload.get("messages", [])
        if messages:
            # Get the last user message (the actual prompt)
            last_msg = messages[-1] if messages else {}
            content = last_msg.get("content", "")
            role = last_msg.get("role", "user")

            # Truncate very long prompts for display
            display_content = content[:1000] + "..." if len(content) > 1000 else content

            if role == "user" and display_content:
                agent.add_message(
                    role=MessageRole.USER,
                    content=display_content,
                )

        # Log the request
        self._ui_state.add_log(
            source=source,
            message=f"LLM request: {payload.get('model', 'unknown')}",
            level="info",
            extraction_id=session_id,
            agent_id=agent.id,
        )

        self._ui_state._notify()

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

    def _handle_llm_chunk(self, session_id: str, event: EventEnvelope) -> None:
        """Handle llm.chunk event for real-time streaming display."""
        extraction_id = self._session_to_extraction.get(session_id)
        if not extraction_id:
            return

        extraction = self._ui_state.extractions.get(extraction_id)
        if not extraction:
            return

        payload = event.payload
        source = event.source
        request_id = payload.get("request_id", "")
        accumulated = payload.get("accumulated", "")

        agent = self._get_or_create_agent(extraction, source)

        # Update or create streaming message
        # Use request_id to track which message to update
        streaming_key = f"streaming_{request_id}"

        if not hasattr(agent, '_streaming_messages'):
            agent._streaming_messages = {}

        if streaming_key not in agent._streaming_messages:
            # Create new streaming message
            agent.add_message(
                role=MessageRole.ASSISTANT,
                content=accumulated,
            )
            agent._streaming_messages[streaming_key] = len(agent.messages) - 1
        else:
            # Update existing message content
            msg_idx = agent._streaming_messages[streaming_key]
            if msg_idx < len(agent.messages):
                agent.messages[msg_idx].content = accumulated

        # Notify UI - this enables real-time display
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

    # =========================================================================
    # Per-Sample Tracking Handlers
    # =========================================================================

    def _handle_optimization_started(self, session_id: str, event: EventEnvelope) -> None:
        """Handle optimization.started event - creates a per-sample agent."""
        extraction_id = self._session_to_extraction.get(session_id)
        if not extraction_id:
            return

        extraction = self._ui_state.extractions.get(extraction_id)
        if not extraction:
            return

        payload = event.payload
        sample_idx = payload.get("sample_idx", 0)
        layer = payload.get("layer", 0)
        num_datapoints = payload.get("num_datapoints", 0)
        config = payload.get("config", {})

        # Create a per-sample agent
        agent = self._get_or_create_sample_agent(extraction, sample_idx, layer, config)
        agent.status = AgentStatus.RUNNING
        agent.started_at = event.timestamp.timestamp()

        # Add initial message
        agent.add_message(
            MessageRole.SYSTEM,
            f"Optimizing steering vector on layer {layer} with {num_datapoints} datapoints"
        )

        extraction.phase = Phase.OPTIMIZING
        self._ui_state._notify()

        self._ui_state.add_log(
            source=f"sample_{sample_idx}",
            message=f"Sample {sample_idx}: Started optimization (layer {layer})",
            level="info",
            extraction_id=session_id,
        )

    def _handle_optimization_progress(self, session_id: str, event: EventEnvelope) -> None:
        """Handle optimization.progress event - updates sample agent."""
        extraction_id = self._session_to_extraction.get(session_id)
        if not extraction_id:
            return

        extraction = self._ui_state.extractions.get(extraction_id)
        if not extraction:
            return

        payload = event.payload
        sample_idx = payload.get("sample_idx", 0)
        iteration = payload.get("iteration", 0)
        loss = payload.get("loss", 0.0)

        agent = self._get_sample_agent(extraction, sample_idx)
        if agent:
            agent.current_tool = f"iter {iteration} loss={loss:.4f}"
            # Only notify every 10 iterations to reduce UI updates
            if iteration % 10 == 0:
                self._ui_state._notify()

    def _handle_optimization_completed(self, session_id: str, event: EventEnvelope) -> None:
        """Handle optimization.completed event - marks sample agent complete."""
        extraction_id = self._session_to_extraction.get(session_id)
        if not extraction_id:
            return

        extraction = self._ui_state.extractions.get(extraction_id)
        if not extraction:
            return

        payload = event.payload
        sample_idx = payload.get("sample_idx", 0)
        success = payload.get("success", True)
        final_loss = payload.get("final_loss", 0.0)
        iterations = payload.get("iterations", 0)
        duration = payload.get("duration_seconds", 0.0)
        error = payload.get("error")

        agent = self._get_sample_agent(extraction, sample_idx)
        if agent:
            agent.status = AgentStatus.COMPLETE if success else AgentStatus.ERROR
            agent.completed_at = event.timestamp.timestamp()
            agent.current_tool = None

            if success:
                agent.add_message(
                    MessageRole.ASSISTANT,
                    f"Optimization complete: loss={final_loss:.4f}, {iterations} iterations in {duration:.1f}s"
                )
            else:
                agent.add_message(
                    MessageRole.ASSISTANT,
                    f"Optimization failed: {error or 'Unknown error'}"
                )

        self._ui_state._notify()

        status_text = "completed" if success else "failed"
        self._ui_state.add_log(
            source=f"sample_{sample_idx}",
            message=f"Sample {sample_idx}: {status_text} (loss={final_loss:.4f})",
            level="info" if success else "error",
            extraction_id=session_id,
        )

    def _handle_contrast_pipeline_started(self, session_id: str, event: EventEnvelope) -> None:
        """Handle contrast.pipeline_started event."""
        extraction_id = self._session_to_extraction.get(session_id)
        if not extraction_id:
            return

        extraction = self._ui_state.extractions.get(extraction_id)
        if not extraction:
            return

        payload = event.payload
        num_samples = payload.get("num_samples", 0)

        extraction.phase = Phase.GENERATING_DATAPOINTS
        extraction.max_outer_iterations = num_samples

        self._ui_state.add_log(
            source="contrast",
            message=f"Starting contrast generation for {num_samples} samples",
            level="info",
            extraction_id=session_id,
        )

        self._ui_state._notify()

    def _handle_contrast_pair_generated(self, session_id: str, event: EventEnvelope) -> None:
        """Handle contrast.pair_generated event."""
        extraction_id = self._session_to_extraction.get(session_id)
        if not extraction_id:
            return

        extraction = self._ui_state.extractions.get(extraction_id)
        if not extraction:
            return

        payload = event.payload
        sample_idx = payload.get("sample_idx", 0)

        # Skip core pool events (sample_idx=-1), not a real sample
        if sample_idx < 0:
            return

        # Create or update sample agent to show contrast pair generation
        agent = self._get_or_create_sample_agent(extraction, sample_idx)
        agent.status = AgentStatus.RUNNING
        agent.tool_calls_count += 1
        agent.current_tool = "generating pairs"

        # Only notify every 5 pairs to reduce UI churn
        if agent.tool_calls_count % 5 == 0:
            self._ui_state._notify()

    def _handle_contrast_pair_validated(self, session_id: str, event: EventEnvelope) -> None:
        """Handle contrast.pair_validated event."""
        extraction_id = self._session_to_extraction.get(session_id)
        if not extraction_id:
            return

        extraction = self._ui_state.extractions.get(extraction_id)
        if not extraction:
            return

        payload = event.payload
        is_valid = payload.get("is_valid", False)
        contrast_quality = payload.get("contrast_quality", 0.0)

        # Update datapoints metrics
        if is_valid:
            extraction.datapoints.total += 1
            extraction.datapoints.keep += 1

        # Notify sparingly
        if extraction.datapoints.total % 5 == 0:
            self._ui_state._notify()

    def _handle_seed_assigned(self, session_id: str, event: EventEnvelope) -> None:
        """Handle seed.assigned event - creates sample agent before optimization."""
        extraction_id = self._session_to_extraction.get(session_id)
        if not extraction_id:
            return

        extraction = self._ui_state.extractions.get(extraction_id)
        if not extraction:
            return

        payload = event.payload
        sample_idx = payload.get("sample_idx", 0)
        num_seeds = len(payload.get("seed_ids", []))

        # Create sample agent in waiting state
        agent = self._get_or_create_sample_agent(extraction, sample_idx)
        agent.status = AgentStatus.WAITING
        agent.add_message(
            MessageRole.SYSTEM,
            f"Assigned {num_seeds} seeds for contrast pair generation"
        )

        self._ui_state._notify()

    def _get_or_create_sample_agent(
        self,
        extraction: ExtractionUIState,
        sample_idx: int,
        layer: int = 0,
        config: dict = None,
    ) -> AgentUIState:
        """Get or create a per-sample agent.

        Args:
            extraction: The extraction state.
            sample_idx: The sample index (0-based).
            layer: Target layer for this sample.
            config: Optimization config for role description.

        Returns:
            AgentUIState for this sample.
        """
        agent_id = f"{extraction.id}_sample_{sample_idx}"

        if agent_id in extraction.agents:
            return extraction.agents[agent_id]

        # Build role description from config
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
        self,
        extraction: ExtractionUIState,
        sample_idx: int,
    ) -> AgentUIState | None:
        """Get a sample agent if it exists.

        Args:
            extraction: The extraction state.
            sample_idx: The sample index.

        Returns:
            AgentUIState or None if not found.
        """
        agent_id = f"{extraction.id}_sample_{sample_idx}"
        return extraction.agents.get(agent_id)

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
