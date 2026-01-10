"""Session replay and state reconstruction from events.

Provides tools to reconstruct state by replaying the event log,
enabling complete reproducibility and analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch

from vector_forge.storage.events import EventEnvelope
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
    eval_type: str = "quick"
    scores: Dict[str, float] = field(default_factory=dict)
    verdict: str = "needs_refinement"
    recommended_strength: float = 1.0


@dataclass
class ReplayedContrastPair:
    """Reconstructed contrast pair from events."""

    pair_id: str
    seed_id: str
    prompt: str
    dst_response: str
    src_response: str
    sample_idx: int
    is_valid: bool = False
    dst_score: float = 0.0
    src_score: float = 0.0
    contrast_quality: float = 0.0
    rejection_reason: Optional[str] = None


@dataclass
class ReplayedSeed:
    """Reconstructed seed from events."""

    seed_id: str
    scenario: str
    context: str
    quality_score: float
    is_core: bool


@dataclass
class ReplayedOptimization:
    """Reconstructed optimization result from events."""

    sample_idx: int
    layer: int
    final_loss: float
    iterations: int
    datapoints_used: int
    duration_seconds: float
    success: bool
    error: Optional[str] = None


@dataclass
class ReplayedLogEntry:
    """Log entry reconstructed from events.

    Contains both summary message and original payload for rich detail views.
    """

    timestamp: datetime
    source: str
    message: str
    level: str = "info"
    event_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplayedState:
    """Full state reconstructed from event replay.

    Contains all datapoints, vectors, evaluations, and metrics
    that can be derived from the event log.
    """

    # Session info
    behavior_name: str = ""
    behavior_description: str = ""
    status: str = "running"  # running, completed, failed

    # Datapoints and vectors
    datapoints: Dict[str, ReplayedDatapoint] = field(default_factory=dict)
    vectors: Dict[int, ReplayedVector] = field(default_factory=dict)  # layer -> vector
    evaluations: List[ReplayedEvaluation] = field(default_factory=list)

    # Contrast pipeline state
    seeds: Dict[str, ReplayedSeed] = field(default_factory=dict)
    contrast_pairs: Dict[str, ReplayedContrastPair] = field(default_factory=dict)
    optimizations: List[ReplayedOptimization] = field(default_factory=list)

    # Best results
    best_layer: Optional[int] = None
    best_strength: float = 1.0
    best_score: float = 0.0

    # Progress tracking
    current_iteration: int = 0
    current_phase: str = "initializing"  # initializing, generating, extracting, evaluating, complete
    progress: float = 0.0

    # Metrics
    llm_call_count: int = 0
    tool_call_count: int = 0
    total_tokens: int = 0

    # Logs from events
    logs: List[ReplayedLogEntry] = field(default_factory=list)

    @property
    def datapoint_list(self) -> List[ReplayedDatapoint]:
        """Get datapoints as list."""
        return list(self.datapoints.values())

    @property
    def datapoint_count(self) -> int:
        """Number of datapoints."""
        return len(self.datapoints)

    @property
    def vector_count(self) -> int:
        """Number of vectors created."""
        return len(self.vectors)

    @property
    def valid_pairs_count(self) -> int:
        """Number of valid contrast pairs."""
        return sum(1 for p in self.contrast_pairs.values() if p.is_valid)

    @property
    def total_pairs_count(self) -> int:
        """Total contrast pairs generated."""
        return len(self.contrast_pairs)


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

        # Create log entry for significant events
        self._maybe_add_log(state, event)

        # Session events
        if event_type == "session.started":
            state.behavior_name = payload.get("behavior_name", "")
            state.behavior_description = payload.get("behavior_description", "")
            state.status = "running"
            state.current_phase = "initializing"

        elif event_type == "session.completed":
            state.status = "completed" if payload.get("success") else "failed"
            state.current_phase = "complete" if payload.get("success") else "failed"
            if payload.get("final_score") is not None:
                state.best_score = payload["final_score"]
            if payload.get("final_layer") is not None:
                state.best_layer = payload["final_layer"]
            state.progress = 1.0

        # Contrast pipeline events
        elif event_type == "contrast.pipeline_started":
            state.current_phase = "generating"
            state.progress = 0.1

        elif event_type == "contrast.pipeline_completed":
            state.progress = 0.4

        elif event_type == "contrast.pair_generated":
            pair = ReplayedContrastPair(
                pair_id=payload["pair_id"],
                seed_id=payload["seed_id"],
                prompt=payload["prompt"],
                dst_response=payload["dst_response"],
                src_response=payload["src_response"],
                sample_idx=payload["sample_idx"],
            )
            state.contrast_pairs[pair.pair_id] = pair

        elif event_type == "contrast.pair_validated":
            pair_id = payload["pair_id"]
            if pair_id in state.contrast_pairs:
                pair = state.contrast_pairs[pair_id]
                pair.is_valid = payload.get("is_valid", False)
                pair.dst_score = payload.get("dst_score", 0.0)
                pair.src_score = payload.get("src_score", 0.0)
                pair.contrast_quality = payload.get("contrast_quality", 0.0)
                pair.rejection_reason = payload.get("rejection_reason")

        # Seed events
        elif event_type == "seed.generation_started":
            state.current_phase = "generating"

        elif event_type == "seed.generated":
            seed = ReplayedSeed(
                seed_id=payload["seed_id"],
                scenario=payload["scenario"],
                context=payload["context"],
                quality_score=payload["quality_score"],
                is_core=payload["is_core"],
            )
            state.seeds[seed.seed_id] = seed

        elif event_type == "seed.generation_completed":
            pass  # Just for logging

        elif event_type == "seed.assigned":
            pass  # Just for logging

        elif event_type == "contrast.behavior_analyzed":
            pass  # Just for logging

        # Datapoint events
        elif event_type == "datapoint.added":
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

        # Optimization events
        elif event_type == "optimization.started":
            state.current_phase = "extracting"
            state.progress = max(state.progress, 0.4)

        elif event_type == "optimization.completed":
            opt = ReplayedOptimization(
                sample_idx=payload["sample_idx"],
                layer=payload["layer"],
                final_loss=payload["final_loss"],
                iterations=payload["iterations"],
                datapoints_used=payload["datapoints_used"],
                duration_seconds=payload["duration_seconds"],
                success=payload["success"],
                error=payload.get("error"),
            )
            state.optimizations.append(opt)

        elif event_type == "optimization.aggregation_completed":
            state.best_score = payload.get("final_score", state.best_score)
            state.best_layer = payload.get("final_layer", state.best_layer)

        # Vector events
        elif event_type == "vector.created":
            vec = ReplayedVector(
                id=payload.get("vector_id", f"vec_{payload['layer']}"),
                layer=payload["layer"],
                vector_ref=payload["vector_ref"],
                norm=payload.get("norm", 0.0),
                optimization_metrics=payload.get("optimization_metrics", {}),
            )
            state.vectors[vec.layer] = vec

        elif event_type == "vector.selected":
            state.best_layer = payload["layer"]
            state.best_strength = payload["strength"]
            if "score" in payload:
                state.best_score = payload["score"]

        # Evaluation events
        elif event_type == "evaluation.started":
            state.current_phase = "evaluating"
            state.progress = max(state.progress, 0.7)

        elif event_type == "evaluation.completed":
            eval_result = ReplayedEvaluation(
                evaluation_id=payload.get("evaluation_id", ""),
                eval_type=payload.get("eval_type", "quick"),
                scores=payload.get("scores", {}),
                verdict=payload.get("verdict", "needs_refinement"),
                recommended_strength=payload.get("recommended_strength", 1.0),
            )
            state.evaluations.append(eval_result)

            # Update best score if better
            overall = eval_result.scores.get("overall", 0)
            if overall > state.best_score:
                state.best_score = overall
            state.progress = max(state.progress, 0.9)

        # Iteration events
        elif event_type == "state.iteration_started":
            iteration_type = payload.get("iteration_type", "")
            if iteration_type in ("outer", "extraction"):
                state.current_iteration = payload.get("iteration", 0)

        elif event_type == "state.iteration_completed":
            pass  # Progress tracked elsewhere

        # LLM events
        elif event_type == "llm.request":
            pass  # Counted on response

        elif event_type == "llm.response":
            state.llm_call_count += 1
            usage = payload.get("usage", {})
            state.total_tokens += usage.get("total_tokens", 0)
            state.total_tokens += usage.get("estimated_tokens", 0)

        # Tool events
        elif event_type == "tool.call":
            pass  # Counted on result

        elif event_type == "tool.result":
            state.tool_call_count += 1

        # Checkpoint events (handled for rollback logic in outer loop)
        elif event_type == "checkpoint.created":
            pass

        elif event_type == "checkpoint.rollback":
            pass  # Handled separately

        # State events
        elif event_type == "state.update":
            pass  # Generic state update, not tracked specifically

        # Evaluation output (individual)
        elif event_type == "evaluation.output":
            pass  # Tracked in evaluation.completed

        # Vector comparison
        elif event_type == "vector.comparison":
            pass  # Just for analysis

    def _maybe_add_log(self, state: ReplayedState, event: EventEnvelope) -> None:
        """Add log entry for significant events."""
        payload = event.payload
        event_type = event.event_type

        # Build log message based on event type
        log_entry = self._build_log_entry(event_type, payload, event.source)
        if log_entry:
            source, message, level = log_entry
            state.logs.append(ReplayedLogEntry(
                timestamp=event.timestamp,
                source=source,
                message=message,
                level=level,
                event_type=event_type,
                payload=payload,  # Include full payload for detail views
            ))

    def _build_log_entry(
        self,
        event_type: str,
        payload: Dict[str, Any],
        source: str,
    ) -> Optional[Tuple[str, str, str]]:
        """Build a log entry tuple (source, message, level) for an event.

        Returns tuple of (source, message, level) or None to skip logging.
        ALL events are logged for complete traceability.
        """
        # Session events
        if event_type == "session.started":
            return ("session", f"Started extraction: {payload.get('behavior_name', 'unknown')}", "info")

        elif event_type == "session.completed":
            success = payload.get('success')
            score = payload.get('final_score')
            layer = payload.get('final_layer')
            if success:
                msg = f"Completed successfully"
                if score is not None:
                    msg += f" | score: {score:.3f}"
                if layer is not None:
                    msg += f" | layer: {layer}"
                return ("session", msg, "info")
            else:
                error = payload.get('error', 'unknown error')
                return ("session", f"Failed: {error}", "error")

        # Contrast pipeline events
        elif event_type == "contrast.pipeline_started":
            return ("contrast", f"Starting contrast generation ({payload.get('num_samples', 0)} samples)", "info")

        elif event_type == "contrast.behavior_analyzed":
            components = payload.get('num_components', 0)
            dims = len(payload.get('contrast_dimensions', []))
            return ("contrast", f"Behavior analyzed: {components} components, {dims} contrast dimensions", "info")

        elif event_type == "contrast.pipeline_completed":
            total = payload.get('total_pairs_generated', 0)
            valid = payload.get('total_valid_pairs', 0)
            quality = payload.get('avg_quality', 0)
            duration = payload.get('duration_seconds', 0)
            return ("contrast", f"Contrast complete: {valid}/{total} valid pairs | quality: {quality:.2f} | {duration:.1f}s", "info")

        elif event_type == "contrast.pair_generated":
            sample = payload.get('sample_idx', 0)
            pair_id = payload.get('pair_id', '')[:8]
            return ("contrast", f"Sample {sample}: generated pair {pair_id}", "info")

        elif event_type == "contrast.pair_validated":
            pair_id = payload.get('pair_id', '')[:8]
            is_valid = payload.get('is_valid', False)
            if is_valid:
                quality = payload.get('contrast_quality', 0)
                dst = payload.get('dst_score', 0)
                src = payload.get('src_score', 0)
                return ("contrast", f"Pair {pair_id} valid: quality={quality:.2f} (dst={dst:.2f}, src={src:.2f})", "info")
            else:
                reason = payload.get('rejection_reason', 'unknown')
                return ("contrast", f"Pair {pair_id} rejected: {reason}", "warning")

        # Seed events
        elif event_type == "seed.generation_started":
            num = payload.get('num_seeds_requested', 0)
            behavior = payload.get('behavior_name', '')
            return ("seed", f"Generating {num} seeds for '{behavior}'", "info")

        elif event_type == "seed.generated":
            seed_id = payload.get('seed_id', '')[:8]
            quality = payload.get('quality_score', 0)
            is_core = "core" if payload.get('is_core') else "unique"
            scenario = payload.get('scenario', '')[:50]
            return ("seed", f"Seed {seed_id} ({is_core}): quality={quality:.2f} | {scenario}...", "info")

        elif event_type == "seed.generation_completed":
            total = payload.get('total_generated', 0)
            filtered = payload.get('total_filtered', 0)
            avg_quality = payload.get('avg_quality', 0)
            return ("seed", f"Seed generation complete: {total} generated, {filtered} filtered | avg quality: {avg_quality:.2f}", "info")

        elif event_type == "seed.assigned":
            sample = payload.get('sample_idx', 0)
            core = payload.get('num_core_seeds', 0)
            unique = payload.get('num_unique_seeds', 0)
            return ("seed", f"Sample {sample}: assigned {core} core + {unique} unique seeds", "info")

        # Datapoint events
        elif event_type == "datapoint.added":
            dp_id = payload.get('datapoint_id', '')[:12]
            domain = payload.get('domain', 'unknown')
            return ("datapoint", f"Added {dp_id} (domain: {domain})", "info")

        elif event_type == "datapoint.removed":
            dp_id = payload.get('datapoint_id', '')[:12]
            reason = payload.get('reason', 'no reason')
            return ("datapoint", f"Removed {dp_id}: {reason}", "warning")

        elif event_type == "datapoint.quality":
            dp_id = payload.get('datapoint_id', '')[:12]
            quality = payload.get('quality_score', 0)
            rec = payload.get('recommendation', 'KEEP')
            return ("datapoint", f"Quality {dp_id}: {quality:.2f} → {rec}", "info")

        # Optimization events
        elif event_type == "optimization.started":
            sample = payload.get('sample_idx', 0)
            layer = payload.get('layer', 0)
            num_dp = payload.get('num_datapoints', 0)
            return ("optimizer", f"Sample {sample}: starting layer {layer} optimization ({num_dp} datapoints)", "info")

        elif event_type == "optimization.progress":
            sample = payload.get('sample_idx', 0)
            iteration = payload.get('iteration', 0)
            loss = payload.get('loss', 0)
            return ("optimizer", f"Sample {sample}: iter {iteration} | loss: {loss:.6f}", "info")

        elif event_type == "optimization.completed":
            sample = payload.get('sample_idx', 0)
            layer = payload.get('layer', 0)
            success = payload.get('success', True)
            if success:
                loss = payload.get('final_loss', 0)
                iters = payload.get('iterations', 0)
                duration = payload.get('duration_seconds', 0)
                return ("optimizer", f"Sample {sample} layer {layer}: loss={loss:.6f} | {iters} iters | {duration:.1f}s", "info")
            else:
                error = payload.get('error', 'unknown')
                return ("optimizer", f"Sample {sample} layer {layer} FAILED: {error}", "error")

        elif event_type == "optimization.aggregation_completed":
            strategy = payload.get('strategy', 'unknown')
            num = payload.get('num_vectors', 0)
            top_k = payload.get('top_k', 0)
            score = payload.get('final_score', 0)
            layer = payload.get('final_layer', 0)
            return ("optimizer", f"Aggregated ({strategy}): {num} vectors, top-{top_k} | score: {score:.3f} | layer: {layer}", "info")

        # Vector events
        elif event_type == "vector.created":
            layer = payload.get('layer', 0)
            norm = payload.get('norm', 0)
            vector_id = payload.get('vector_id', '')[:8]
            return ("vector", f"Created {vector_id} at layer {layer} | norm: {norm:.4f}", "info")

        elif event_type == "vector.comparison":
            ids = payload.get('vector_ids', [])
            return ("vector", f"Compared {len(ids)} vectors", "info")

        elif event_type == "vector.selected":
            layer = payload.get('layer', 0)
            strength = payload.get('strength', 1.0)
            score = payload.get('score')
            reason = payload.get('reason', '')
            msg = f"Selected layer {layer} @ strength {strength:.2f}"
            if score is not None:
                msg += f" | score: {score:.3f}"
            if reason:
                msg += f" | {reason}"
            return ("vector", msg, "info")

        # Evaluation events
        elif event_type == "evaluation.started":
            eval_id = payload.get('evaluation_id', '')[:8]
            eval_type = payload.get('eval_type', 'quick')
            layer = payload.get('layer', 0)
            strengths = payload.get('strength_levels', [])
            return ("evaluation", f"Started {eval_type} eval {eval_id}: layer {layer}, {len(strengths)} strengths", "info")

        elif event_type == "evaluation.output":
            eval_id = payload.get('evaluation_id', '')[:8]
            strength = payload.get('strength', 1.0)
            is_baseline = payload.get('is_baseline', False)
            label = "baseline" if is_baseline else f"strength={strength:.2f}"
            return ("evaluation", f"Eval {eval_id} output ({label})", "info")

        elif event_type == "evaluation.completed":
            eval_id = payload.get('evaluation_id', '')[:8]
            scores = payload.get('scores', {})
            overall = scores.get('overall', 0)
            verdict = payload.get('verdict', 'unknown')
            strength = payload.get('recommended_strength', 1.0)
            return ("evaluation", f"Eval {eval_id} complete: {overall:.3f} | {verdict} | rec. strength: {strength:.2f}", "info")

        # Checkpoint events
        elif event_type == "checkpoint.created":
            cp_id = payload.get('checkpoint_id', '')[:8]
            desc = payload.get('description', '')
            return ("checkpoint", f"Created checkpoint {cp_id}: {desc}", "info")

        elif event_type == "checkpoint.rollback":
            cp_id = payload.get('checkpoint_id', '')[:8]
            prev_seq = payload.get('previous_sequence', 0)
            return ("checkpoint", f"Rolled back to {cp_id} (seq {prev_seq})", "warning")

        # State events
        elif event_type == "state.update":
            field = payload.get('field', 'unknown')
            old = payload.get('old_value')
            new = payload.get('new_value')
            return ("state", f"Updated {field}: {old} → {new}", "info")

        elif event_type == "state.iteration_started":
            iter_type = payload.get('iteration_type', 'unknown')
            iteration = payload.get('iteration', 0)
            max_iter = payload.get('max_iterations', 0)
            if max_iter:
                return ("state", f"{iter_type} iteration {iteration}/{max_iter} started", "info")
            return ("state", f"{iter_type} iteration {iteration} started", "info")

        elif event_type == "state.iteration_completed":
            iter_type = payload.get('iteration_type', 'unknown')
            iteration = payload.get('iteration', 0)
            metrics = payload.get('metrics', {})
            msg = f"{iter_type} iteration {iteration} completed"
            if metrics:
                msg += f" | {metrics}"
            return ("state", msg, "info")

        # LLM events
        elif event_type == "llm.request":
            model = payload.get('model', 'unknown')
            num_msgs = len(payload.get('messages', []))
            tools = payload.get('tools')
            tool_str = f" | {len(tools)} tools" if tools else ""
            return (source or "llm", f"Request: {model} ({num_msgs} messages{tool_str})", "info")

        elif event_type == "llm.response":
            latency = payload.get('latency_ms', 0)
            finish = payload.get('finish_reason', 'stop')
            error = payload.get('error')
            if error:
                return (source or "llm", f"Response ERROR: {error}", "error")
            tool_calls = payload.get('tool_calls', [])
            if tool_calls:
                return (source or "llm", f"Response: {len(tool_calls)} tool calls | {latency}ms", "info")
            return (source or "llm", f"Response: {finish} | {latency}ms", "info")

        # Tool events
        elif event_type == "tool.call":
            tool = payload.get('tool_name', 'unknown')
            call_id = payload.get('call_id', '')[:8]
            return ("tool", f"Calling {tool} ({call_id})", "info")

        elif event_type == "tool.result":
            call_id = payload.get('call_id', '')[:8]
            success = payload.get('success', True)
            latency = payload.get('latency_ms', 0)
            if success:
                return ("tool", f"Result {call_id}: success | {latency}ms", "info")
            else:
                error = payload.get('error', 'unknown')
                return ("tool", f"Result {call_id}: FAILED - {error}", "error")

        # Fallback for unknown events
        return (source or "unknown", f"Event: {event_type}", "info")

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
