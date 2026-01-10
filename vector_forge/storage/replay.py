"""Session replay and state reconstruction from events.

Provides tools to reconstruct state by replaying the event log,
enabling complete reproducibility and analysis.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from vector_forge.storage.events import (
    EventEnvelope,
    EventPayload,
    LLMRequestEvent,
    LLMResponseEvent,
    ToolCallEvent,
    ToolResultEvent,
    DatapointAddedEvent,
    DatapointRemovedEvent,
    DatapointQualityEvent,
    VectorCreatedEvent,
    VectorSelectedEvent,
    EvaluationCompletedEvent,
    CheckpointRollbackEvent,
)
from vector_forge.storage.store import SessionStore


@dataclass
class ReplayedDatapoint:
    """Reconstructed datapoint from events."""

    id: str
    prompt: str
    positive_completion: str
    negative_completion: Optional[str] = None
    domain: Optional[str] = None
    quality_score: Optional[float] = None
    recommendation: Optional[str] = None


@dataclass
class ReplayedVector:
    """Reconstructed vector info from events."""

    id: str
    layer: int
    vector_ref: str
    norm: float
    optimization_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayedEvaluation:
    """Reconstructed evaluation from events."""

    evaluation_id: str
    scores: Dict[str, float] = field(default_factory=dict)
    verdict: str = "needs_refinement"
    recommended_strength: float = 1.0


@dataclass
class ReplayedState:
    """Full state reconstructed from event replay.

    Contains all datapoints, vectors, evaluations, and metrics
    that can be derived from the event log.
    """

    datapoints: Dict[str, ReplayedDatapoint] = field(default_factory=dict)
    vectors: Dict[int, ReplayedVector] = field(default_factory=dict)  # layer -> vector
    evaluations: List[ReplayedEvaluation] = field(default_factory=list)
    best_layer: Optional[int] = None
    best_strength: float = 1.0
    best_score: float = 0.0
    current_iteration: int = 0
    llm_call_count: int = 0
    tool_call_count: int = 0
    total_tokens: int = 0

    @property
    def datapoint_list(self) -> List[ReplayedDatapoint]:
        """Get datapoints as list."""
        return list(self.datapoints.values())

    @property
    def vector_count(self) -> int:
        """Number of vectors created."""
        return len(self.vectors)


class SessionReplayer:
    """Reconstructs state from event log.

    Replays events in sequence to rebuild the complete state,
    supporting analysis and verification of extraction runs.

    Example:
        >>> store = manager.get_session("session_123")
        >>> replayer = SessionReplayer(store)
        >>> state = replayer.reconstruct_state()
        >>> print(f"Datapoints: {len(state.datapoints)}")
        >>> print(f"Best layer: {state.best_layer}")
    """

    def __init__(self, store: SessionStore) -> None:
        """Initialize replayer.

        Args:
            store: Session store to replay from.
        """
        self.store = store

    def reconstruct_state(
        self,
        up_to_sequence: Optional[int] = None,
    ) -> ReplayedState:
        """Rebuild state by replaying events.

        Args:
            up_to_sequence: Stop at this sequence number (for partial replay).

        Returns:
            Reconstructed state.
        """
        state = ReplayedState()
        rollback_to: Optional[int] = None

        for event in self.store.iter_events():
            # Stop at specified sequence
            if up_to_sequence and event.sequence > up_to_sequence:
                break

            # Handle rollbacks by marking a point to ignore
            if event.event_type == "checkpoint.rollback":
                payload = event.payload
                rollback_to = payload.get("previous_sequence", 0)
                continue

            # Skip events that were rolled back
            if rollback_to and event.sequence <= rollback_to:
                continue

            self._apply_event(state, event)

        return state

    def _apply_event(self, state: ReplayedState, event: EventEnvelope) -> None:
        """Apply single event to state.

        Args:
            state: State to modify.
            event: Event to apply.
        """
        payload = event.payload
        event_type = event.event_type

        if event_type == "datapoint.added":
            dp = ReplayedDatapoint(
                id=payload["datapoint_id"],
                prompt=payload["prompt"],
                positive_completion=payload["positive_completion"],
                negative_completion=payload.get("negative_completion"),
                domain=payload.get("domain"),
            )
            state.datapoints[dp.id] = dp

        elif event_type == "datapoint.removed":
            dp_id = payload["datapoint_id"]
            if dp_id in state.datapoints:
                del state.datapoints[dp_id]

        elif event_type == "datapoint.quality":
            dp_id = payload["datapoint_id"]
            if dp_id in state.datapoints:
                state.datapoints[dp_id].quality_score = payload.get("quality_score")
                state.datapoints[dp_id].recommendation = payload.get("recommendation")

        elif event_type == "vector.created":
            vec = ReplayedVector(
                id=payload["vector_id"],
                layer=payload["layer"],
                vector_ref=payload["vector_ref"],
                norm=payload["norm"],
                optimization_metrics=payload.get("optimization_metrics", {}),
            )
            state.vectors[vec.layer] = vec

        elif event_type == "vector.selected":
            state.best_layer = payload["layer"]
            state.best_strength = payload["strength"]
            if "score" in payload:
                state.best_score = payload["score"]

        elif event_type == "evaluation.completed":
            eval_result = ReplayedEvaluation(
                evaluation_id=payload["evaluation_id"],
                scores=payload.get("scores", {}),
                verdict=payload.get("verdict", "needs_refinement"),
                recommended_strength=payload.get("recommended_strength", 1.0),
            )
            state.evaluations.append(eval_result)

            # Update best score if better
            overall = eval_result.scores.get("overall", 0)
            if overall > state.best_score:
                state.best_score = overall

        elif event_type == "state.iteration_started":
            if payload.get("iteration_type") == "outer":
                state.current_iteration = payload.get("iteration", 0)

        elif event_type == "llm.response":
            state.llm_call_count += 1
            usage = payload.get("usage", {})
            state.total_tokens += usage.get("total_tokens", 0)

        elif event_type == "tool.result":
            state.tool_call_count += 1

    def iter_llm_calls(self) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Iterate through all LLM request/response pairs.

        Yields:
            Tuples of (request_payload, response_payload).
        """
        requests: Dict[str, Dict[str, Any]] = {}

        for event in self.store.iter_events(
            event_types=["llm.request", "llm.response"]
        ):
            if event.event_type == "llm.request":
                requests[event.payload["request_id"]] = event.payload
            elif event.event_type == "llm.response":
                request_id = event.payload["request_id"]
                if request_id in requests:
                    yield (requests[request_id], event.payload)
                    del requests[request_id]

    def iter_tool_calls(self) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Iterate through all tool call/result pairs.

        Yields:
            Tuples of (call_payload, result_payload).
        """
        calls: Dict[str, Dict[str, Any]] = {}

        for event in self.store.iter_events(
            event_types=["tool.call", "tool.result"]
        ):
            if event.event_type == "tool.call":
                calls[event.payload["call_id"]] = event.payload
            elif event.event_type == "tool.result":
                call_id = event.payload["call_id"]
                if call_id in calls:
                    yield (calls[call_id], event.payload)
                    del calls[call_id]

    def get_conversation_history(
        self,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Reconstruct conversation from LLM events.

        Args:
            source: Optional filter by source (e.g., "extractor", "judge").

        Returns:
            List of messages in conversation order.
        """
        messages = []

        for request, response in self.iter_llm_calls():
            # Check source filter
            # Note: source is on the envelope, not payload
            # This simplified version includes all

            # Add messages from request
            for msg in request.get("messages", []):
                messages.append(msg)

            # Add assistant response
            if response.get("content") or response.get("tool_calls"):
                assistant_msg = {
                    "role": "assistant",
                    "content": response.get("content"),
                }
                if response.get("tool_calls"):
                    assistant_msg["tool_calls"] = response["tool_calls"]
                messages.append(assistant_msg)

        return messages

    def get_datapoints_timeline(self) -> List[Tuple[int, str, str]]:
        """Get timeline of datapoint additions/removals.

        Returns:
            List of (sequence, action, datapoint_id) tuples.
        """
        timeline = []

        for event in self.store.iter_events(
            event_types=["datapoint.added", "datapoint.removed"]
        ):
            action = "added" if event.event_type == "datapoint.added" else "removed"
            dp_id = event.payload["datapoint_id"]
            timeline.append((event.sequence, action, dp_id))

        return timeline

    def get_vectors_created(self) -> List[ReplayedVector]:
        """Get all vectors created during the session.

        Returns:
            List of vector info in creation order.
        """
        vectors = []

        for event in self.store.iter_events(event_types=["vector.created"]):
            payload = event.payload
            vec = ReplayedVector(
                id=payload["vector_id"],
                layer=payload["layer"],
                vector_ref=payload["vector_ref"],
                norm=payload["norm"],
                optimization_metrics=payload.get("optimization_metrics", {}),
            )
            vectors.append(vec)

        return vectors

    def load_vector(self, vector_ref: str) -> torch.Tensor:
        """Load a vector by reference.

        Args:
            vector_ref: Relative path from session directory.

        Returns:
            The loaded tensor.
        """
        return self.store.load_vector(vector_ref)

    def load_best_vector(self) -> Optional[torch.Tensor]:
        """Load the best selected vector.

        Returns:
            The best vector tensor, or None if not selected.
        """
        state = self.reconstruct_state()
        if state.best_layer is None:
            return None

        if state.best_layer in state.vectors:
            return self.load_vector(state.vectors[state.best_layer].vector_ref)

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get session statistics from events.

        Returns:
            Dictionary of statistics.
        """
        state = self.reconstruct_state()
        metadata = self.store.get_metadata()

        # Count events by type
        event_counts: Dict[str, int] = {}
        for event in self.store.iter_events():
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1

        return {
            "session_id": self.store.session_id,
            "behavior": metadata.get("behavior"),
            "status": metadata.get("status"),
            "created_at": metadata.get("created_at"),
            "completed_at": metadata.get("completed_at"),
            "total_events": sum(event_counts.values()),
            "event_counts": event_counts,
            "datapoint_count": len(state.datapoints),
            "vector_count": state.vector_count,
            "evaluation_count": len(state.evaluations),
            "llm_call_count": state.llm_call_count,
            "tool_call_count": state.tool_call_count,
            "total_tokens": state.total_tokens,
            "best_layer": state.best_layer,
            "best_score": state.best_score,
        }
