"""Event emitter for consistent event sourcing across components.

Provides a clean interface for emitting events without components needing
to know about SessionStore internals. This follows the dependency inversion
principle - components depend on the EventEmitter protocol, not concrete storage.
"""

import uuid
from typing import Any, Dict, List, Optional, Protocol

from vector_forge.storage.events import (
    # Contrast events
    BehaviorAnalyzedEvent,
    ContrastPipelineStartedEvent,
    ContrastPipelineCompletedEvent,
    ContrastPairGeneratedEvent,
    ContrastPairValidatedEvent,
    # Seed events
    SeedGenerationStartedEvent,
    SeedGeneratedEvent,
    SeedGenerationCompletedEvent,
    SeedAssignedEvent,
    # Optimization events
    OptimizationStartedEvent,
    OptimizationProgressEvent,
    OptimizationCompletedEvent,
    AggregationCompletedEvent,
    # Datapoint events
    DatapointAddedEvent,
    # Evaluation events
    EvaluationStartedEvent,
    EvaluationDimensionStartedEvent,
    EvaluationGenerationEvent,
    EvaluationJudgeCallEvent,
    EvaluationDimensionCompletedEvent,
    EvaluationProgressEvent,
    EvaluationOutputEvent,
    EvaluationCompletedEvent,
)


def generate_id(prefix: str = "") -> str:
    """Generate a unique ID with optional prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}" if prefix else uuid.uuid4().hex[:12]


class EventEmitterProtocol(Protocol):
    """Protocol for event emission - allows for testing with mocks."""

    def emit(self, event: Any, source: str) -> None:
        """Emit an event."""
        ...


class EventEmitter:
    """Wrapper around SessionStore for clean event emission.

    Provides typed methods for emitting specific events, ensuring
    consistent event creation across all components.

    Supports optional real-time notification callback for immediate UI updates.

    Example:
        >>> emitter = EventEmitter(session_store)
        >>> emitter.emit_seed_generated(
        ...     seed_id="seed_123",
        ...     scenario="User asks about climate change",
        ...     context="Academic discussion",
        ...     quality_score=8.5,
        ...     is_core=True,
        ... )

        # With real-time callback:
        >>> emitter = EventEmitter(
        ...     store=session_store,
        ...     on_event=lambda env: ui_synchronizer.handle(env),
        ... )
    """

    def __init__(
        self,
        store: Any,
        default_source: str = "pipeline",
        on_event: Optional[callable] = None,
    ) -> None:
        """Initialize the emitter.

        Args:
            store: SessionStore instance for persistence.
            default_source: Default source identifier for events.
            on_event: Optional callback for real-time event notification.
                     Called with EventEnvelope after each event is persisted.
        """
        self._store = store
        self._default_source = default_source
        self._on_event = on_event

    def emit(self, event: Any, source: Optional[str] = None) -> None:
        """Emit a raw event to the store.

        Args:
            event: The event payload to emit.
            source: Optional source override.
        """
        envelope = self._store.append_event(event, source=source or self._default_source)

        # Immediate notification for real-time UI updates
        if self._on_event is not None:
            try:
                self._on_event(envelope)
            except Exception:
                pass  # Don't let notification errors break event emission

    # =========================================================================
    # Contrast Pipeline Events
    # =========================================================================

    def emit_pipeline_started(
        self,
        behavior_description: str,
        num_samples: int,
        config: Dict[str, Any],
    ) -> None:
        """Emit contrast pipeline started event."""
        self.emit(
            ContrastPipelineStartedEvent(
                behavior_description=behavior_description,
                num_samples=num_samples,
                config=config,
            ),
            source="contrast_pipeline",
        )

    def emit_pipeline_completed(
        self,
        num_samples: int,
        total_pairs_generated: int,
        total_valid_pairs: int,
        avg_quality: float,
        duration_seconds: float,
    ) -> None:
        """Emit contrast pipeline completed event."""
        self.emit(
            ContrastPipelineCompletedEvent(
                num_samples=num_samples,
                total_pairs_generated=total_pairs_generated,
                total_valid_pairs=total_valid_pairs,
                avg_quality=avg_quality,
                duration_seconds=duration_seconds,
            ),
            source="contrast_pipeline",
        )

    def emit_behavior_analyzed(
        self,
        behavior_name: str,
        num_components: int,
        components: List[Dict[str, Any]],
        trigger_conditions: List[str],
        contrast_dimensions: List[str],
    ) -> None:
        """Emit behavior analysis completed event."""
        self.emit(
            BehaviorAnalyzedEvent(
                behavior_name=behavior_name,
                num_components=num_components,
                components=components,
                trigger_conditions=trigger_conditions,
                contrast_dimensions=contrast_dimensions,
            ),
            source="behavior_analyzer",
        )

    def emit_pair_generated(
        self,
        pair_id: str,
        seed_id: str,
        prompt: str,
        dst_response: str,
        src_response: str,
        sample_idx: int,
    ) -> None:
        """Emit contrast pair generated event."""
        self.emit(
            ContrastPairGeneratedEvent(
                pair_id=pair_id,
                seed_id=seed_id,
                prompt=prompt,
                dst_response=dst_response,
                src_response=src_response,
                sample_idx=sample_idx,
            ),
            source="pair_generator",
        )

    def emit_pair_validated(
        self,
        pair_id: str,
        is_valid: bool,
        dst_score: float,
        src_score: float,
        semantic_distance: float,
        contrast_quality: float,
        rejection_reason: Optional[str] = None,
    ) -> None:
        """Emit contrast pair validation event."""
        self.emit(
            ContrastPairValidatedEvent(
                pair_id=pair_id,
                is_valid=is_valid,
                dst_score=dst_score,
                src_score=src_score,
                semantic_distance=semantic_distance,
                contrast_quality=contrast_quality,
                rejection_reason=rejection_reason,
            ),
            source="validator",
        )

    # =========================================================================
    # Seed Events
    # =========================================================================

    def emit_seed_generation_started(
        self,
        num_seeds_requested: int,
        behavior_name: str,
    ) -> None:
        """Emit seed generation started event."""
        self.emit(
            SeedGenerationStartedEvent(
                num_seeds_requested=num_seeds_requested,
                behavior_name=behavior_name,
            ),
            source="seed_generator",
        )

    def emit_seed_generated(
        self,
        seed_id: str,
        scenario: str,
        context: str,
        quality_score: float,
        is_core: bool,
    ) -> None:
        """Emit single seed generated event."""
        self.emit(
            SeedGeneratedEvent(
                seed_id=seed_id,
                scenario=scenario,
                context=context,
                quality_score=quality_score,
                is_core=is_core,
            ),
            source="seed_generator",
        )

    def emit_seed_generation_completed(
        self,
        total_generated: int,
        total_filtered: int,
        avg_quality: float,
        min_quality_threshold: float,
    ) -> None:
        """Emit seed generation batch completed event."""
        self.emit(
            SeedGenerationCompletedEvent(
                total_generated=total_generated,
                total_filtered=total_filtered,
                avg_quality=avg_quality,
                min_quality_threshold=min_quality_threshold,
            ),
            source="seed_generator",
        )

    def emit_seeds_assigned(
        self,
        sample_idx: int,
        num_core_seeds: int,
        num_unique_seeds: int,
        seed_ids: List[str],
    ) -> None:
        """Emit seeds assigned to sample event."""
        self.emit(
            SeedAssignedEvent(
                sample_idx=sample_idx,
                num_core_seeds=num_core_seeds,
                num_unique_seeds=num_unique_seeds,
                seed_ids=seed_ids,
            ),
            source="contrast_pipeline",
        )

    # =========================================================================
    # Optimization Events
    # =========================================================================

    def emit_optimization_started(
        self,
        sample_idx: int,
        layer: int,
        num_datapoints: int,
        config: Dict[str, Any],
    ) -> None:
        """Emit optimization started event."""
        self.emit(
            OptimizationStartedEvent(
                sample_idx=sample_idx,
                layer=layer,
                num_datapoints=num_datapoints,
                config=config,
            ),
            source="optimizer",
        )

    def emit_optimization_progress(
        self,
        sample_idx: int,
        iteration: int,
        loss: float,
        norm: float,
    ) -> None:
        """Emit optimization progress event."""
        self.emit(
            OptimizationProgressEvent(
                sample_idx=sample_idx,
                iteration=iteration,
                loss=loss,
                norm=norm,
            ),
            source="optimizer",
        )

    def emit_optimization_completed(
        self,
        sample_idx: int,
        layer: int,
        final_loss: float,
        iterations: int,
        loss_history: List[float],
        datapoints_used: int,
        duration_seconds: float,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Emit optimization completed event."""
        self.emit(
            OptimizationCompletedEvent(
                sample_idx=sample_idx,
                layer=layer,
                final_loss=final_loss,
                iterations=iterations,
                loss_history=loss_history,
                datapoints_used=datapoints_used,
                duration_seconds=duration_seconds,
                success=success,
                error=error,
            ),
            source="optimizer",
        )

    def emit_aggregation_completed(
        self,
        strategy: str,
        num_vectors: int,
        top_k: int,
        ensemble_components: List[str],
        final_score: float,
        final_layer: int,
    ) -> None:
        """Emit aggregation completed event."""
        self.emit(
            AggregationCompletedEvent(
                strategy=strategy,
                num_vectors=num_vectors,
                top_k=top_k,
                ensemble_components=ensemble_components,
                final_score=final_score,
                final_layer=final_layer,
            ),
            source="aggregator",
        )

    # =========================================================================
    # Datapoint Events
    # =========================================================================

    def emit_datapoint_added(
        self,
        datapoint_id: str,
        prompt: str,
        positive_completion: str,
        negative_completion: Optional[str] = None,
        domain: Optional[str] = None,
        format_type: Optional[str] = None,
    ) -> None:
        """Emit datapoint added event."""
        self.emit(
            DatapointAddedEvent(
                datapoint_id=datapoint_id,
                prompt=prompt,
                positive_completion=positive_completion,
                negative_completion=negative_completion,
                domain=domain,
                format_type=format_type,
            ),
            source="datapoint_adapter",
        )

    # =========================================================================
    # Evaluation Events
    # =========================================================================

    def emit_evaluation_started(
        self,
        evaluation_id: str,
        eval_type: str,
        vector_id: str,
        layer: int,
        strength_levels: List[float],
        num_prompts: int,
        dimensions: Optional[List[str]] = None,
    ) -> None:
        """Emit evaluation started event."""
        self.emit(
            EvaluationStartedEvent(
                evaluation_id=evaluation_id,
                eval_type=eval_type,
                vector_id=vector_id,
                layer=layer,
                strength_levels=strength_levels,
                num_prompts=num_prompts,
                dimensions=dimensions or ["behavior", "specificity", "coherence", "capability", "generalization"],
            ),
            source="evaluator",
        )

    def emit_evaluation_dimension_started(
        self,
        evaluation_id: str,
        dimension: str,
        num_prompts: int,
        num_generations: int,
    ) -> None:
        """Emit dimension evaluation started event."""
        self.emit(
            EvaluationDimensionStartedEvent(
                evaluation_id=evaluation_id,
                dimension=dimension,
                num_prompts=num_prompts,
                num_generations=num_generations,
            ),
            source="evaluator",
        )

    def emit_evaluation_generation(
        self,
        evaluation_id: str,
        dimension: str,
        prompt: str,
        output: str,
        strength: float,
        generation_index: int,
        is_baseline: bool = False,
    ) -> None:
        """Emit single model generation event during evaluation."""
        self.emit(
            EvaluationGenerationEvent(
                evaluation_id=evaluation_id,
                dimension=dimension,
                prompt=prompt,
                output=output,
                strength=strength,
                generation_index=generation_index,
                is_baseline=is_baseline,
            ),
            source="evaluator",
        )

    def emit_evaluation_judge_call(
        self,
        evaluation_id: str,
        dimension: str,
        prompt: str,
        num_outputs: int,
        scores: List[float],
        latency_ms: float = 0.0,
    ) -> None:
        """Emit judge LLM call event during evaluation."""
        self.emit(
            EvaluationJudgeCallEvent(
                evaluation_id=evaluation_id,
                dimension=dimension,
                prompt=prompt,
                num_outputs=num_outputs,
                scores=scores,
                latency_ms=latency_ms,
            ),
            source="evaluator",
        )

    def emit_evaluation_dimension_completed(
        self,
        evaluation_id: str,
        dimension: str,
        score: float,
        max_score: float = 10.0,
        details: Optional[Dict[str, Any]] = None,
        duration_seconds: float = 0.0,
    ) -> None:
        """Emit dimension evaluation completed event."""
        self.emit(
            EvaluationDimensionCompletedEvent(
                evaluation_id=evaluation_id,
                dimension=dimension,
                score=score,
                max_score=max_score,
                details=details or {},
                duration_seconds=duration_seconds,
            ),
            source="evaluator",
        )

    def emit_evaluation_progress(
        self,
        evaluation_id: str,
        phase: str,
        completed: int,
        total: int,
        current_dimension: Optional[str] = None,
    ) -> None:
        """Emit evaluation progress update."""
        self.emit(
            EvaluationProgressEvent(
                evaluation_id=evaluation_id,
                phase=phase,
                completed=completed,
                total=total,
                current_dimension=current_dimension,
            ),
            source="evaluator",
        )

    def emit_evaluation_output(
        self,
        evaluation_id: str,
        prompt: str,
        output: str,
        strength: Optional[float] = None,
        is_baseline: bool = False,
    ) -> None:
        """Emit single evaluation output event (legacy)."""
        self.emit(
            EvaluationOutputEvent(
                evaluation_id=evaluation_id,
                prompt=prompt,
                output=output,
                strength=strength,
                is_baseline=is_baseline,
            ),
            source="evaluator",
        )

    def emit_evaluation_completed(
        self,
        evaluation_id: str,
        scores: Dict[str, float],
        recommended_strength: float,
        verdict: str,
        dimension_scores: Optional[Dict[str, float]] = None,
        citations: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        recommendations: Optional[List[str]] = None,
        raw_judge_output: Optional[str] = None,
        duration_seconds: float = 0.0,
        total_generations: int = 0,
        total_judge_calls: int = 0,
    ) -> None:
        """Emit evaluation completed event."""
        self.emit(
            EvaluationCompletedEvent(
                evaluation_id=evaluation_id,
                scores=scores,
                dimension_scores=dimension_scores or {},
                citations=citations or {},
                recommendations=recommendations or [],
                verdict=verdict,
                recommended_strength=recommended_strength,
                raw_judge_output=raw_judge_output,
                duration_seconds=duration_seconds,
                total_generations=total_generations,
                total_judge_calls=total_judge_calls,
            ),
            source="evaluator",
        )


class NullEventEmitter:
    """Null object pattern - does nothing, for testing or when events disabled."""

    def emit(self, event: Any, source: str = "") -> None:
        """No-op emit."""
        pass

    def __getattr__(self, name: str):
        """Return no-op for any emit_* method."""
        if name.startswith("emit_"):
            return lambda *args, **kwargs: None
        raise AttributeError(name)
