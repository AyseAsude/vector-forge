"""Event handlers connecting pipeline events to UI state."""

import time
from typing import Optional

from vector_forge.core.events import Event, EventType
from vector_forge.core.behavior import BehaviorSpec
from vector_forge.ui.state import (
    UIState,
    ExtractionUIState,
    ExtractionStatus,
    Phase,
    DatapointMetrics,
    EvaluationMetrics,
    get_state,
)
from vector_forge.ui.theme import ICONS


class UIEventHandler:
    """Handles pipeline events and updates UI state.

    Translates pipeline events into UI state updates,
    providing a bridge between the extraction pipeline
    and the terminal UI.
    """

    def __init__(
        self,
        extraction_id: str,
        behavior: BehaviorSpec,
        model: str = "",
        state: Optional[UIState] = None,
    ) -> None:
        """Initialize the event handler.

        Args:
            extraction_id: Unique ID for this extraction.
            behavior: Behavior specification being extracted.
            model: Model name being used.
            state: UI state to update. Uses global state if not provided.
        """
        self._extraction_id = extraction_id
        self._behavior = behavior
        self._model = model
        self._state = state or get_state()

        # Create initial extraction state
        extraction = ExtractionUIState(
            id=extraction_id,
            behavior_name=behavior.name,
            behavior_description=behavior.description,
            model=model,
            status=ExtractionStatus.PENDING,
            phase=Phase.INITIALIZING,
        )
        self._state.add_extraction(extraction)

    @property
    def extraction(self) -> Optional[ExtractionUIState]:
        """Get the current extraction state."""
        return self._state.extractions.get(self._extraction_id)

    def handle_event(self, event: Event) -> None:
        """Handle a pipeline event.

        Args:
            event: Event from the extraction pipeline.
        """
        handler_map = {
            EventType.PIPELINE_STARTED: self._on_pipeline_started,
            EventType.PIPELINE_COMPLETED: self._on_pipeline_completed,
            EventType.PIPELINE_FAILED: self._on_pipeline_failed,
            EventType.OUTER_ITERATION_STARTED: self._on_outer_iteration_started,
            EventType.OUTER_ITERATION_COMPLETED: self._on_outer_iteration_completed,
            EventType.INNER_ITERATION_STARTED: self._on_inner_iteration_started,
            EventType.INNER_ITERATION_COMPLETED: self._on_inner_iteration_completed,
            EventType.DATAPOINT_GENERATION_STARTED: self._on_datapoint_generation_started,
            EventType.DATAPOINT_GENERATION_COMPLETED: self._on_datapoint_generation_completed,
            EventType.DATAPOINT_QUALITY_ANALYZED: self._on_datapoint_quality_analyzed,
            EventType.OPTIMIZATION_STARTED: self._on_optimization_started,
            EventType.OPTIMIZATION_COMPLETED: self._on_optimization_completed,
            EventType.LAYER_SEARCH_PROGRESS: self._on_layer_search_progress,
            EventType.QUICK_EVAL_STARTED: self._on_quick_eval_started,
            EventType.QUICK_EVAL_COMPLETED: self._on_quick_eval_completed,
            EventType.THOROUGH_EVAL_STARTED: self._on_thorough_eval_started,
            EventType.THOROUGH_EVAL_COMPLETED: self._on_thorough_eval_completed,
            EventType.JUDGE_STARTED: self._on_judge_started,
            EventType.JUDGE_VERDICT: self._on_judge_verdict,
            EventType.AGENT_TOOL_CALL: self._on_tool_call,
            EventType.AGENT_TOOL_RESULT: self._on_tool_result,
            EventType.WARNING: self._on_warning,
            EventType.ERROR: self._on_error,
        }

        handler = handler_map.get(event.type)
        if handler:
            handler(event)

    def _update_extraction(self, **updates) -> None:
        """Update extraction state fields."""
        self._state.update_extraction(self._extraction_id, **updates)

    def _add_activity(self, message: str, status: str = "active") -> None:
        """Add activity entry."""
        extraction = self.extraction
        if extraction:
            extraction.add_activity(ICONS.active, message, status)

    def _log(self, source: str, message: str, level: str = "info") -> None:
        """Add log entry."""
        self._state.add_log(source, message, level, self._extraction_id)

    # Event handlers

    def _on_pipeline_started(self, event: Event) -> None:
        self._update_extraction(
            status=ExtractionStatus.RUNNING,
            phase=Phase.INITIALIZING,
            started_at=time.time(),
        )
        self._log("pipeline", f"Pipeline started: {event.data.get('behavior', '')}")

    def _on_pipeline_completed(self, event: Event) -> None:
        self._update_extraction(
            status=ExtractionStatus.COMPLETE,
            phase=Phase.COMPLETE,
            progress=100.0,
            completed_at=time.time(),
        )
        score = event.data.get("score", 0)
        self._log("pipeline", f"Pipeline completed (score={score:.2f})")

    def _on_pipeline_failed(self, event: Event) -> None:
        self._update_extraction(
            status=ExtractionStatus.FAILED,
            phase=Phase.FAILED,
            completed_at=time.time(),
        )
        reason = event.data.get("reason", "Unknown error")
        self._log("pipeline", f"Pipeline failed: {reason}", "error")

    def _on_outer_iteration_started(self, event: Event) -> None:
        iteration = event.data.get("iteration", 0)
        self._update_extraction(
            outer_iteration=iteration + 1,
            phase=Phase.GENERATING_DATAPOINTS,
        )
        self._log("pipeline", f"Outer iteration {iteration + 1} started")

    def _on_outer_iteration_completed(self, event: Event) -> None:
        iteration = event.data.get("iteration", 0)
        verdict = event.data.get("verdict", "")
        self._log("pipeline", f"Outer iteration {iteration + 1} completed ({verdict})")

    def _on_inner_iteration_started(self, event: Event) -> None:
        self._update_extraction(inner_turn=0)

    def _on_inner_iteration_completed(self, event: Event) -> None:
        turns = event.data.get("turns", 0)
        self._update_extraction(inner_turn=turns)

    def _on_datapoint_generation_started(self, event: Event) -> None:
        self._update_extraction(phase=Phase.GENERATING_DATAPOINTS)
        self._add_activity("Generating datapoints...", "active")
        self._log("extractor", "Generating datapoints...")

    def _on_datapoint_generation_completed(self, event: Event) -> None:
        count = event.data.get("count", 0)
        self._add_activity(f"Generated {count} datapoints", "success")
        self._log("extractor", f"Generated {count} datapoints")

    def _on_datapoint_quality_analyzed(self, event: Event) -> None:
        # Update datapoint metrics
        extraction = self.extraction
        if extraction:
            metrics = extraction.datapoints
            # Count quality categories
            quality = event.data.get("quality", {})
            metrics.total = quality.get("total", metrics.total)
            metrics.keep = quality.get("keep", metrics.keep)
            metrics.review = quality.get("review", metrics.review)
            metrics.remove = quality.get("remove", metrics.remove)

    def _on_optimization_started(self, event: Event) -> None:
        layer = event.data.get("layer")
        self._update_extraction(
            phase=Phase.OPTIMIZING,
            current_layer=layer,
        )
        if layer:
            self._add_activity(f"Optimizing layer {layer}...", "active")
            self._log("extractor", f"Optimizing vector at layer {layer}")
        else:
            self._add_activity("Optimizing vector...", "active")
            self._log("extractor", "Optimizing vector")

    def _on_optimization_completed(self, event: Event) -> None:
        layer = event.data.get("layer")
        loss = event.data.get("loss", 0)
        norm = event.data.get("norm", 0)

        message = f"Layer {layer}: loss={loss:.4f} norm={norm:.2f}"
        self._add_activity(message, "success")
        self._log("extractor", message)

    def _on_layer_search_progress(self, event: Event) -> None:
        layer = event.data.get("layer")
        self._update_extraction(current_layer=layer)

    def _on_quick_eval_started(self, event: Event) -> None:
        self._update_extraction(phase=Phase.EVALUATING)
        layer = event.data.get("layer")
        self._add_activity(f"Quick eval layer {layer}...", "active")
        self._log("extractor", f"Quick evaluation at layer {layer}")

    def _on_quick_eval_completed(self, event: Event) -> None:
        score = event.data.get("score", 0)
        self._add_activity(f"Quick eval: score={score:.2f}", "success")
        self._log("extractor", f"Quick eval completed (score={score:.2f})")

    def _on_thorough_eval_started(self, event: Event) -> None:
        self._update_extraction(phase=Phase.JUDGE_REVIEW)
        self._add_activity("Thorough evaluation...", "active")
        self._log("judge", "Thorough evaluation started")

    def _on_thorough_eval_completed(self, event: Event) -> None:
        score = event.data.get("overall_score", 0)
        self._add_activity(f"Evaluation: score={score:.2f}", "success")
        self._log("judge", f"Thorough evaluation completed (score={score:.2f})")

    def _on_judge_started(self, event: Event) -> None:
        self._update_extraction(phase=Phase.JUDGE_REVIEW)
        self._add_activity("Judge review...", "active")
        self._log("judge", "Judge review started")

    def _on_judge_verdict(self, event: Event) -> None:
        verdict = event.data.get("verdict", "")
        score = event.data.get("score", 0)

        # Update evaluation metrics
        extraction = self.extraction
        if extraction:
            extraction.evaluation.verdict = verdict
            extraction.evaluation.overall = score

        level = "info"
        if verdict.lower() == "rejected":
            level = "error"
        elif verdict.lower() == "needs_refinement":
            level = "warning"

        self._add_activity(f"Verdict: {verdict.upper()}", "success")
        self._log("judge", f"Verdict: {verdict.upper()} (score={score:.2f})", level)

    def _on_tool_call(self, event: Event) -> None:
        tool = event.data.get("tool", "")
        args = event.data.get("args", {})

        # Format tool call
        if args:
            arg_str = " ".join(f"{k}={v}" for k, v in list(args.items())[:3])
            message = f"{tool} {arg_str}"
        else:
            message = tool

        extraction = self.extraction
        if extraction:
            extraction.add_activity(ICONS.active, message, "active")

        # Increment turn counter
        extraction = self.extraction
        if extraction:
            extraction.inner_turn += 1

    def _on_tool_result(self, event: Event) -> None:
        success = event.data.get("success", True)

        extraction = self.extraction
        if extraction and extraction.activity:
            last = extraction.activity[-1]
            extraction.activity[-1] = type(last)(
                timestamp=last.timestamp,
                icon=ICONS.success if success else ICONS.error,
                message=last.message,
                status="success" if success else "error",
            )

    def _on_warning(self, event: Event) -> None:
        message = event.data.get("message", "Warning")
        self._log("pipeline", message, "warning")

    def _on_error(self, event: Event) -> None:
        message = event.data.get("message", "Error")
        self._log("pipeline", message, "error")

    def update_progress(self, progress: float) -> None:
        """Update overall progress percentage.

        Args:
            progress: Progress percentage (0-100).
        """
        self._update_extraction(progress=progress)

    def update_datapoint_metrics(
        self,
        total: int,
        keep: int,
        review: int,
        remove: int,
        diversity: float,
        clusters: int,
    ) -> None:
        """Update datapoint metrics.

        Args:
            total: Total datapoint count.
            keep: Count of datapoints to keep.
            review: Count of datapoints to review.
            remove: Count of datapoints to remove.
            diversity: Diversity score.
            clusters: Number of clusters.
        """
        extraction = self.extraction
        if extraction:
            extraction.datapoints = DatapointMetrics(
                total=total,
                keep=keep,
                review=review,
                remove=remove,
                diversity=diversity,
                clusters=clusters,
            )

    def update_evaluation_metrics(
        self,
        behavior: float,
        coherence: float,
        specificity: float,
        overall: float,
        best_layer: Optional[int] = None,
        best_strength: float = 1.0,
        verdict: Optional[str] = None,
    ) -> None:
        """Update evaluation metrics.

        Args:
            behavior: Behavior score.
            coherence: Coherence score.
            specificity: Specificity score.
            overall: Overall score.
            best_layer: Best performing layer.
            best_strength: Best steering strength.
            verdict: Judge verdict.
        """
        extraction = self.extraction
        if extraction:
            extraction.evaluation = EvaluationMetrics(
                behavior=behavior,
                coherence=coherence,
                specificity=specificity,
                overall=overall,
                best_layer=best_layer,
                best_strength=best_strength,
                verdict=verdict,
            )
