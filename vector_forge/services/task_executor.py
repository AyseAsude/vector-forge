"""Task executor service with event emission.

Wraps TaskRunner to emit events to SessionStore during execution,
enabling full reproducibility through event sourcing.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from vector_forge.storage import (
    SessionStore,
    LLMRequestEvent,
    LLMResponseEvent,
    ToolCallEvent,
    ToolResultEvent,
    DatapointAddedEvent,
    VectorCreatedEvent,
    VectorSelectedEvent,
    EvaluationStartedEvent,
    EvaluationCompletedEvent,
    IterationStartedEvent,
    IterationCompletedEvent,
)
from vector_forge.services.session import SessionService
from vector_forge.tasks.config import TaskConfig
from vector_forge.tasks.task import ExtractionTask, TaskResult
from vector_forge.tasks.runner import TaskRunner, RunnerProgress
from vector_forge.contrast.protocols import SampleDataset

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Context for task execution with event emission."""

    session_id: str
    store: SessionStore
    config: TaskConfig
    started_at: float


class EventEmittingLLMClient:
    """LLM client wrapper that emits events for each call."""

    def __init__(
        self,
        client: Any,
        store: SessionStore,
        source: str = "llm",
    ) -> None:
        self._client = client
        self._store = store
        self._source = source

    async def generate(
        self,
        messages: List[dict],
        **kwargs,
    ) -> str:
        """Generate with event emission."""
        # Emit request event
        request_event = LLMRequestEvent(
            model=kwargs.get("model", "unknown"),
            messages=messages,
            tools=kwargs.get("tools"),
            temperature=kwargs.get("temperature"),
            max_tokens=kwargs.get("max_tokens"),
        )
        self._store.append_event(request_event, source=self._source)

        start_time = time.time()
        try:
            # Call underlying client
            response = await self._client.generate(messages, **kwargs)

            # Emit response event
            latency_ms = int((time.time() - start_time) * 1000)
            response_event = LLMResponseEvent(
                content=response if isinstance(response, str) else str(response),
                tool_calls=None,  # Would need to parse if structured
                finish_reason="stop",
                latency_ms=latency_ms,
                usage={"estimated_tokens": len(str(response)) // 4},
            )
            self._store.append_event(response_event, source=self._source)

            return response

        except Exception as e:
            # Emit error response
            response_event = LLMResponseEvent(
                content="",
                finish_reason="error",
                latency_ms=int((time.time() - start_time) * 1000),
                usage={},
            )
            self._store.append_event(response_event, source=self._source)
            raise


class EventEmittingToolRegistry:
    """Tool registry wrapper that emits events for each tool call."""

    def __init__(
        self,
        registry: Any,
        store: SessionStore,
        source: str = "tools",
    ) -> None:
        self._registry = registry
        self._store = store
        self._source = source

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        **kwargs,
    ) -> Any:
        """Execute tool with event emission."""
        # Emit call event
        call_event = ToolCallEvent(
            tool_name=tool_name,
            arguments=arguments,
            call_id=f"call_{int(time.time() * 1000)}",
        )
        self._store.append_event(call_event, source=self._source)

        start_time = time.time()
        try:
            # Execute tool
            result = await self._registry.execute(tool_name, arguments, **kwargs)

            # Emit result event
            duration_ms = int((time.time() - start_time) * 1000)
            result_event = ToolResultEvent(
                call_id=call_event.call_id,
                success=True,
                output=result if isinstance(result, dict) else {"result": str(result)},
                duration_ms=duration_ms,
            )
            self._store.append_event(result_event, source=self._source)

            return result

        except Exception as e:
            # Emit error result
            result_event = ToolResultEvent(
                call_id=call_event.call_id,
                success=False,
                error=str(e),
                duration_ms=int((time.time() - start_time) * 1000),
            )
            self._store.append_event(result_event, source=self._source)
            raise


class TaskExecutor:
    """Executes extraction tasks with full event emission.

    Wraps TaskRunner to capture all operations as events in the
    SessionStore, enabling replay and reproducibility.

    Example:
        >>> executor = TaskExecutor(session_service)
        >>> result = await executor.execute(
        ...     session_id="session_123",
        ...     task=task,
        ...     backend=model_backend,
        ...     llm=llm_client,
        ... )
    """

    def __init__(self, session_service: SessionService) -> None:
        """Initialize executor.

        Args:
            session_service: Service for session management.
        """
        self._session_service = session_service
        self._progress_callback: Optional[Callable[[str, RunnerProgress], None]] = None

    def on_progress(
        self,
        callback: Callable[[str, RunnerProgress], None],
    ) -> None:
        """Register progress callback.

        Args:
            callback: Function called with (session_id, progress).
        """
        self._progress_callback = callback

    async def execute(
        self,
        session_id: str,
        task: ExtractionTask,
        sample_datasets: Dict[int, SampleDataset],
        model_backend: Any,
        llm_client: Any,
        tool_registry: Optional[Any] = None,
    ) -> TaskResult:
        """Execute task with event emission.

        Args:
            session_id: Session to record events to.
            task: The extraction task to run.
            sample_datasets: Pre-generated datasets for each sample (from ContrastPipeline).
            model_backend: Backend for model inference.
            llm_client: Client for LLM API calls.
            tool_registry: Optional tool registry.

        Returns:
            Task result with final vector.
        """
        store = self._session_service.get_session_store(session_id)
        config = task.config

        # Emit iteration started
        iter_event = IterationStartedEvent(
            iteration_type="extraction",
            iteration=0,
            max_iterations=task.num_samples,
        )
        store.append_event(iter_event, source="executor")

        # Wrap clients with event emission
        emitting_llm = EventEmittingLLMClient(llm_client, store, source="extractor")

        # Create runner
        runner = TaskRunner(
            model_backend=model_backend,
            llm_client=emitting_llm,
            max_concurrent_extractions=config.max_concurrent_extractions,
            max_concurrent_evaluations=config.max_concurrent_evaluations,
        )

        # Set up progress reporting
        def on_runner_progress(progress: RunnerProgress) -> None:
            # Emit progress as iteration event
            if progress.current_phase == "extracting":
                iter_event = IterationCompletedEvent(
                    iteration_type="extraction",
                    iteration=progress.completed_extractions,
                    success=True,
                    metrics={
                        "completed": progress.completed_extractions,
                        "total": progress.total_samples,
                        "failed": progress.failed_count,
                    },
                )
                store.append_event(iter_event, source="executor")

            elif progress.current_phase == "evaluating":
                iter_event = IterationCompletedEvent(
                    iteration_type="evaluation",
                    iteration=progress.completed_evaluations,
                    success=True,
                    metrics={
                        "completed": progress.completed_evaluations,
                        "total": progress.total_samples,
                    },
                )
                store.append_event(iter_event, source="executor")

            # Forward to external callback
            if self._progress_callback:
                self._progress_callback(session_id, progress)

        runner.on_progress(on_runner_progress)

        try:
            # Run the task with sample datasets
            result = await runner.run(task, sample_datasets)

            # Emit vector created event
            if result.final_vector is not None:
                # Save vector to session
                vector_ref = store.save_final_vector(result.final_vector)

                vector_event = VectorCreatedEvent(
                    layer=result.final_layer,
                    vector_ref=vector_ref,
                    method="aggregation",
                    source_datapoints=result.valid_results_count,
                    metadata={
                        "aggregation": result.aggregation_method.value,
                        "ensemble_size": len(result.ensemble_components),
                    },
                )
                store.append_event(vector_event, source="executor")

                # Emit vector selected event
                selected_event = VectorSelectedEvent(
                    layer=result.final_layer,
                    strength=result.recommended_strength,
                    vector_ref=vector_ref,
                    reason=f"Best score: {result.final_score:.3f}",
                )
                store.append_event(selected_event, source="executor")

            # Emit evaluation completed
            eval_event = EvaluationCompletedEvent(
                eval_type="final",
                scores={
                    "overall": result.final_score,
                    "valid_samples": result.valid_results_count,
                    "total_samples": len(result.sample_results),
                },
                recommended_strength=result.recommended_strength,
                verdict="completed",
            )
            store.append_event(eval_event, source="executor")

            # Complete session
            self._session_service.complete_session(
                session_id=session_id,
                success=True,
                final_score=result.final_score,
                final_layer=result.final_layer,
            )

            return result

        except Exception as e:
            logger.error(f"Task execution failed: {e}")

            # Complete session with error
            self._session_service.complete_session(
                session_id=session_id,
                success=False,
                error=str(e),
            )
            raise

    async def execute_from_description(
        self,
        behavior_name: str,
        behavior_description: str,
        sample_datasets: Dict[int, SampleDataset],
        config: TaskConfig,
        model_backend: Any,
        llm_client: Any,
    ) -> tuple[str, TaskResult]:
        """Create session and execute task from description.

        Convenience method that creates the session and task.

        Args:
            behavior_name: Name of behavior.
            behavior_description: Full description.
            sample_datasets: Pre-generated datasets for each sample (from ContrastPipeline).
            config: Task configuration.
            model_backend: Backend for inference.
            llm_client: LLM client.

        Returns:
            Tuple of (session_id, result).
        """
        # Create session
        session_id = self._session_service.create_session(
            behavior_name=behavior_name,
            behavior_description=behavior_description,
            config=config,
        )

        # Create task
        from vector_forge.tasks.expander import ExpandedBehavior

        behavior = ExpandedBehavior(
            name=behavior_name,
            description=behavior_description,
            detailed_definition=behavior_description,
        )

        task = ExtractionTask.from_behavior(behavior, config)

        # Execute with sample datasets
        result = await self.execute(
            session_id=session_id,
            task=task,
            sample_datasets=sample_datasets,
            model_backend=model_backend,
            llm_client=llm_client,
        )

        return session_id, result
