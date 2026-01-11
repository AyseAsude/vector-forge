"""Session loader for initial UI state population.

Loads existing sessions from JSONL files into UI state on startup.
Real-time updates are handled by the JSONL watcher (ui/watcher.py).

This follows the Single Responsibility Principle - this module only
handles initial loading, not real-time synchronization.
"""

import logging
from pathlib import Path
from typing import Optional

from vector_forge.storage import SessionReplayer
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
)

logger = logging.getLogger(__name__)


class SessionLoader:
    """Loads existing sessions from storage into UI state.

    This is a stateless utility that populates UIState from JSONL files.
    No callbacks, no listeners, no real-time updates - just initial load.

    Example:
        >>> loader = SessionLoader(session_service)
        >>> count = loader.load_into(ui_state, limit=50)
        >>> print(f"Loaded {count} sessions")
    """

    def __init__(self, session_service: SessionService) -> None:
        """Initialize the loader.

        Args:
            session_service: Service for accessing session storage.
        """
        self._session_service = session_service

    def load_into(self, ui_state: UIState, limit: int = 50) -> int:
        """Load existing sessions into UI state.

        Reads all sessions from disk, replays their JSONL events,
        and populates the UI state with the reconstructed data.

        Args:
            ui_state: The UI state to populate.
            limit: Maximum sessions to load.

        Returns:
            Number of sessions loaded.
        """
        sessions = self._session_service.list_sessions(limit=limit)

        loaded = 0
        for session_info in sessions:
            try:
                extraction = self._build_extraction(session_info)
                ui_state.extractions[extraction.id] = extraction

                # Add logs from replayed events
                self._load_logs(session_info.session_id, ui_state)

                loaded += 1
            except Exception as e:
                logger.warning(f"Failed to load session {session_info.session_id}: {e}")

        # Auto-select first if none selected
        if ui_state.selected_id is None and ui_state.extractions:
            ui_state.selected_id = next(iter(ui_state.extractions))

        logger.info(f"Loaded {loaded} sessions into UI state")
        return loaded

    def _build_extraction(self, session_info: SessionInfo) -> ExtractionUIState:
        """Build ExtractionUIState by replaying session events.

        Args:
            session_info: Session metadata.

        Returns:
            Fully reconstructed ExtractionUIState.
        """
        # Replay events to get full state
        replayed = None
        try:
            store = self._session_service.get_session_store(session_info.session_id)
            replayer = SessionReplayer(store)
            replayed = replayer.reconstruct_state()
        except Exception as e:
            logger.warning(f"Failed to replay session {session_info.session_id}: {e}")

        # Map status
        status = _map_status(session_info.status)
        phase, progress = _determine_phase(replayed, status)

        # Build metrics
        datapoints = DatapointMetrics()
        evaluation = EvaluationMetrics()

        if replayed:
            datapoints.total = replayed.datapoint_count
            datapoints.keep = replayed.datapoint_count
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

        # Load config for model info
        model = ""
        target_model = ""
        try:
            store = self._session_service.get_session_store(session_info.session_id)
            config_path = store.session_path / "config.json"
            if config_path.exists():
                import json
                with open(config_path) as f:
                    config = json.load(f)
                    # LLMConfig is nested as generator_llm.model
                    generator_llm = config.get("generator_llm", {})
                    model = generator_llm.get("model", "") if isinstance(generator_llm, dict) else ""
                    target_model = config.get("target_model", "")
        except Exception:
            pass

        # Create extraction
        extraction = ExtractionUIState(
            id=session_info.session_id,
            behavior_name=replayed.behavior_name if replayed and replayed.behavior_name else session_info.behavior,
            behavior_description=replayed.behavior_description if replayed else "",
            model=model,
            target_model=target_model,
            status=status,
            phase=phase,
            progress=progress,
            started_at=session_info.created_at.timestamp() if session_info.created_at else None,
            completed_at=session_info.completed_at.timestamp() if session_info.completed_at else None,
            current_layer=replayed.best_layer if replayed else None,
            datapoints=datapoints,
            evaluation=evaluation,
        )

        # Add agents from replayed data
        if replayed:
            self._add_agents_from_replay(extraction, replayed, session_info)

        return extraction

    def _add_agents_from_replay(
        self,
        extraction: ExtractionUIState,
        replayed,
        session_info: SessionInfo,
    ) -> None:
        """Add agents reconstructed from replayed events."""
        started_at = session_info.created_at.timestamp() if session_info.created_at else None
        completed_at = session_info.completed_at.timestamp() if session_info.completed_at else None

        # Sample agents from optimizations
        for opt in replayed.optimizations:
            agent_id = f"{extraction.id}_sample_{opt.sample_idx}"
            agent = AgentUIState(
                id=agent_id,
                name=f"Sample {opt.sample_idx + 1}",
                role=f"L{opt.layer} iters={opt.iterations}",
                status=AgentStatus.COMPLETE if opt.success else AgentStatus.ERROR,
                started_at=started_at,
                completed_at=completed_at,
                turns=1,
                tool_calls_count=opt.iterations,
            )

            if opt.success:
                loss_str = f"{opt.final_loss:.4f}" if opt.final_loss is not None else "N/A"
                msg = f"Optimization complete: loss={loss_str}, {opt.iterations} iterations in {opt.duration_seconds:.1f}s"
            else:
                msg = f"Optimization failed: {opt.error or 'Unknown error'}"
            agent.add_message(MessageRole.ASSISTANT, msg)
            extraction.add_agent(agent)

        # Update sample agents from seed assignments
        for sample_idx, (num_core, num_unique) in replayed.seed_assignments.items():
            total_pairs = num_core + num_unique
            agent_id = f"{extraction.id}_sample_{sample_idx}"

            if agent_id in extraction.agents:
                extraction.agents[agent_id].tool_calls_count = total_pairs
            else:
                agent = AgentUIState(
                    id=agent_id,
                    name=f"Sample {sample_idx + 1}",
                    role=f"{num_core} core + {num_unique} unique",
                    status=AgentStatus.COMPLETE,
                    started_at=started_at,
                    completed_at=completed_at,
                    turns=1,
                    tool_calls_count=total_pairs,
                )
                agent.add_message(
                    MessageRole.ASSISTANT,
                    f"Assigned {num_core} core + {num_unique} unique seeds ({total_pairs} total pairs)"
                )
                extraction.add_agent(agent)

        # Source-based agents from LLM events
        source_agents = {}
        for log in replayed.logs:
            if log.event_type not in ("llm.request", "llm.response"):
                continue

            source = log.source or ""
            if not source or source.startswith("sample_"):
                continue

            if source not in source_agents:
                agent_id = f"{extraction.id}_{source}"
                agent = AgentUIState(
                    id=agent_id,
                    name=source.replace("_", " ").title(),
                    role=source,
                    status=AgentStatus.COMPLETE,
                    started_at=started_at,
                    completed_at=completed_at,
                    turns=0,
                    tool_calls_count=0,
                )
                source_agents[source] = agent

            agent = source_agents[source]
            event_ts = log.timestamp.timestamp()

            if log.event_type == "llm.request":
                agent.turns += 1
                messages = log.payload.get("messages", []) if log.payload else []
                if messages:
                    content = messages[-1].get("content", "") if messages else ""
                    if content:
                        agent.add_message(MessageRole.USER, content, timestamp=event_ts)

            elif log.event_type == "llm.response":
                tool_calls = []
                for tc in log.payload.get("tool_calls", []) if log.payload else []:
                    tool_calls.append(ToolCall(
                        id=tc.get("id", ""),
                        name=tc.get("function", {}).get("name", ""),
                        arguments=str(tc.get("function", {}).get("arguments", "")),
                        status="success",
                    ))

                content = log.payload.get("content", "") if log.payload else ""
                if content or tool_calls:
                    agent.add_message(
                        MessageRole.ASSISTANT,
                        content or "(tool call)",
                        tool_calls=tool_calls,
                        timestamp=event_ts,
                    )
                    agent.tool_calls_count += len(tool_calls)

        for agent in source_agents.values():
            extraction.add_agent(agent)

    def _load_logs(self, session_id: str, ui_state: UIState) -> None:
        """Load logs from replayed events into UI state."""
        try:
            store = self._session_service.get_session_store(session_id)
            replayer = SessionReplayer(store)
            replayed = replayer.reconstruct_state()

            for log in replayed.logs:
                ui_state.logs.append(LogEntry(
                    timestamp=log.timestamp.timestamp(),
                    source=log.source,
                    message=log.message,
                    level=log.level,
                    extraction_id=session_id,
                    event_type=log.event_type if log.event_type else None,
                    payload=log.payload if log.payload else None,
                ))
        except Exception as e:
            logger.warning(f"Failed to load logs for session {session_id}: {e}")


def _map_status(status_str: str) -> ExtractionStatus:
    """Map status string to enum."""
    return {
        "running": ExtractionStatus.RUNNING,
        "completed": ExtractionStatus.COMPLETE,
        "failed": ExtractionStatus.FAILED,
    }.get(status_str, ExtractionStatus.PENDING)


def _determine_phase(replayed, status: ExtractionStatus) -> tuple[Phase, float]:
    """Determine phase and progress from replayed state or status."""
    if replayed:
        phase_map = {
            "initializing": Phase.INITIALIZING,
            "generating": Phase.GENERATING_DATAPOINTS,
            "extracting": Phase.OPTIMIZING,
            "evaluating": Phase.EVALUATING,
            "complete": Phase.COMPLETE,
            "failed": Phase.FAILED,
        }
        return phase_map.get(replayed.current_phase, Phase.INITIALIZING), replayed.progress

    if status == ExtractionStatus.COMPLETE:
        return Phase.COMPLETE, 1.0
    elif status == ExtractionStatus.FAILED:
        return Phase.FAILED, 0.0
    elif status == ExtractionStatus.RUNNING:
        return Phase.GENERATING_DATAPOINTS, 0.5
    else:
        return Phase.INITIALIZING, 0.0


# Backwards compatibility alias
UIStateSynchronizer = SessionLoader
