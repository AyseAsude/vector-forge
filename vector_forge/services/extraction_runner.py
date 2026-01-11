"""Extraction runner service for orchestrating the full extraction pipeline.

This service handles the complete flow from task creation to result:
1. Load target HuggingFace model
2. Create LLM clients for extractor and judge
3. Run ContrastPipeline to generate sample datasets
4. Run TaskExecutor to perform extraction
5. Handle errors and cleanup
"""

import asyncio
import gc
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch

from vector_forge.llm import create_client
from vector_forge.tasks.config import TaskConfig
from vector_forge.tasks.task import ExtractionTask, TaskResult
from vector_forge.tasks.expander import ExpandedBehavior
from vector_forge.contrast.pipeline import ContrastPipeline, ContrastPipelineConfig
from vector_forge.services.session import SessionService
from vector_forge.services.task_executor import TaskExecutor
from vector_forge.tasks.runner import RunnerProgress

logger = logging.getLogger(__name__)


@dataclass
class ExtractionProgress:
    """Progress information for an extraction."""

    session_id: str
    phase: str  # "loading_model", "generating_contrast", "extracting", "evaluating", "complete", "failed"
    progress: float  # 0.0 to 1.0
    message: str
    error: Optional[str] = None


class ExtractionRunner:
    """Orchestrates the complete extraction pipeline.

    Handles model loading, contrast generation, and extraction execution.
    Runs asynchronously to avoid blocking the UI.

    Example:
        >>> runner = ExtractionRunner(session_service)
        >>> await runner.run_extraction(
        ...     session_id="session_123",
        ...     behavior_name="sycophancy",
        ...     behavior_description="...",
        ...     config=TaskConfig.standard(),
        ... )
    """

    def __init__(
        self,
        session_service: SessionService,
        task_executor: TaskExecutor,
    ) -> None:
        """Initialize the extraction runner.

        Args:
            session_service: Service for session management.
            task_executor: Executor for running tasks.
        """
        self._session_service = session_service
        self._task_executor = task_executor
        self._progress_callbacks: List[Callable[[ExtractionProgress], None]] = []
        self._running_extractions: Dict[str, asyncio.Task] = {}

        # Model caching - reuse loaded model for same model_id
        self._cached_backend: Optional[Any] = None
        self._cached_model_id: Optional[str] = None

    def on_progress(self, callback: Callable[[ExtractionProgress], None]) -> None:
        """Register a progress callback.

        Args:
            callback: Function called with progress updates.
        """
        self._progress_callbacks.append(callback)

    def remove_progress_callback(
        self,
        callback: Callable[[ExtractionProgress], None],
    ) -> None:
        """Remove a progress callback."""
        if callback in self._progress_callbacks:
            self._progress_callbacks.remove(callback)

    def _emit_progress(self, progress: ExtractionProgress) -> None:
        """Emit progress to all callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    async def run_extraction(
        self,
        session_id: str,
        behavior_name: str,
        behavior_description: str,
        config: TaskConfig,
    ) -> Optional[TaskResult]:
        """Run the complete extraction pipeline.

        Args:
            session_id: The session ID (already created).
            behavior_name: Name of the behavior.
            behavior_description: Full description.
            config: Task configuration.

        Returns:
            TaskResult if successful, None if failed.
        """
        try:
            # Step 1: Get or load target model (cached for reuse)
            if self._cached_model_id == config.target_model and self._cached_backend is not None:
                self._emit_progress(ExtractionProgress(
                    session_id=session_id,
                    phase="loading_model",
                    progress=0.0,
                    message="Using cached model...",
                ))
                model_backend = self._cached_backend
                logger.info(f"Reusing cached model: {config.target_model}")
            else:
                self._emit_progress(ExtractionProgress(
                    session_id=session_id,
                    phase="loading_model",
                    progress=0.0,
                    message="Loading target model...",
                ))

                # Free previous model if different
                if self._cached_backend is not None:
                    self._free_cached_model()

                model_backend = await self._load_target_model(config.target_model)
                self._cached_backend = model_backend
                self._cached_model_id = config.target_model

            # Step 2: Create LLM clients with event logging and real-time notification
            from vector_forge.services.task_executor import EventEmittingLLMClient
            from vector_forge.storage import EventEmitter
            from vector_forge.tasks.expander import BehaviorExpander

            store = self._session_service.get_session_store(session_id)
            raw_expander = create_client(config.expander_model)
            raw_extractor = create_client(config.extractor_model)
            raw_judge = create_client(config.judge_model)

            # Wrap behavior expander client to emit LLM events (creates agent in UI)
            behavior_expander_llm = EventEmittingLLMClient(
                raw_expander, store,
                source="behavior_expander",
            )

            # Step 3: Run BehaviorExpander first (uses expander model)
            self._emit_progress(ExtractionProgress(
                session_id=session_id,
                phase="expanding_behavior",
                progress=0.05,
                message="Expanding behavior description...",
            ))

            expander = BehaviorExpander(behavior_expander_llm, model=config.expander_model)
            expanded_behavior = await expander.expand(behavior_description)
            logger.info(
                f"Behavior expanded: {expanded_behavior.name} | "
                f"{len(expanded_behavior.domains)} domains, "
                f"{len(expanded_behavior.evaluation_criteria)} criteria"
            )

            self._emit_progress(ExtractionProgress(
                session_id=session_id,
                phase="generating_contrast",
                progress=0.1,
                message="Generating contrast pairs...",
            ))

            # Wrap with event emission so ALL LLM calls are logged
            # Use expander model for contrast pipeline (analysis, seeds, pairs)
            expander_llm = EventEmittingLLMClient(
                raw_expander, store,
                source="contrast_extractor",
            )
            judge_llm = EventEmittingLLMClient(
                raw_judge, store,
                source="contrast_judge",
            )
            # Extractor LLM for task execution (optimization phase)
            extractor_llm = EventEmittingLLMClient(
                raw_extractor, store,
                source="extractor",
            )

            # Create event emitter for complete event sourcing
            event_emitter = EventEmitter(store, default_source="extraction_runner")

            # Step 4: Build contrast config from task config
            contrast_config = ContrastPipelineConfig(
                core_pool_size=config.contrast.core_pool_size,
                core_seeds_per_sample=config.contrast.core_seeds_per_sample,
                unique_seeds_per_sample=config.contrast.unique_seeds_per_sample,
                min_semantic_score=config.contrast.min_semantic_score,
                min_dimension_score=config.contrast.min_dimension_score,
                min_structural_score=config.contrast.min_structural_score,
                min_contrast_quality=config.contrast.min_contrast_quality,
                max_regeneration_attempts=config.contrast.max_regeneration_attempts,
                max_concurrent_generations=config.contrast.max_concurrent_generations,
                generation_temperature=config.contrast.generation_temperature,
                # Intensity distribution
                intensity_extreme=config.contrast.intensity_extreme,
                intensity_high=config.contrast.intensity_high,
                intensity_medium=config.contrast.intensity_medium,
                intensity_natural=config.contrast.intensity_natural,
            )

            # Step 5: Run ContrastPipeline with event emitter
            # Uses expander_llm for BehaviorAnalyzer, seeds, and pairs
            contrast_pipeline = ContrastPipeline(
                llm_client=expander_llm,
                judge_llm_client=judge_llm,
                config=contrast_config,
                event_emitter=event_emitter,
            )

            pipeline_result = await contrast_pipeline.run(
                behavior_description=behavior_description,
                num_samples=config.num_samples,
            )

            # Log contrast pipeline results
            logger.info(
                f"ContrastPipeline complete: {pipeline_result.num_samples} samples, "
                f"{pipeline_result.total_valid_pairs} valid pairs total"
            )
            for sample_idx, dataset in pipeline_result.sample_datasets.items():
                logger.info(
                    f"  Sample {sample_idx}: {len(dataset.valid_pairs)} valid pairs, "
                    f"quality={dataset.avg_contrast_quality:.2f}"
                )
                if len(dataset.valid_pairs) == 0:
                    logger.warning(f"  Sample {sample_idx} has NO valid pairs!")

            self._emit_progress(ExtractionProgress(
                session_id=session_id,
                phase="extracting",
                progress=0.4,
                message=f"Starting extraction ({config.num_samples} samples)...",
            ))

            # Step 6: Augment ExpandedBehavior with analysis from contrast pipeline
            # This merges the expander output (domains, criteria) with
            # the analyzer output (scenarios, components) for comprehensive evaluation
            expanded_behavior.augment_with_analysis(pipeline_result.behavior_analysis)
            logger.info(
                f"Behavior augmented with analysis: "
                f"{len(expanded_behavior.realistic_scenarios)} scenarios, "
                f"{len(expanded_behavior.components)} components"
            )

            task = ExtractionTask.from_behavior(expanded_behavior, config)

            # Step 7: Set up progress forwarding
            def on_runner_progress(sid: str, runner_progress: RunnerProgress) -> None:
                # Map runner progress to our progress format
                if runner_progress.current_phase == "extracting":
                    base_progress = 0.4
                    phase_progress = runner_progress.extraction_progress * 0.3
                else:  # evaluating
                    base_progress = 0.7
                    phase_progress = runner_progress.evaluation_progress * 0.2

                self._emit_progress(ExtractionProgress(
                    session_id=session_id,
                    phase=runner_progress.current_phase,
                    progress=base_progress + phase_progress,
                    message=f"{runner_progress.current_phase.title()}: "
                            f"{runner_progress.completed_extractions}/{runner_progress.total_samples}",
                ))

            self._task_executor.on_progress(on_runner_progress)

            # Step 7: Run extraction
            result = await self._task_executor.execute(
                session_id=session_id,
                task=task,
                sample_datasets=pipeline_result.sample_datasets,
                model_backend=model_backend,
                llm_client=extractor_llm,
            )

            self._emit_progress(ExtractionProgress(
                session_id=session_id,
                phase="complete",
                progress=1.0,
                message=f"Extraction complete! Score: {result.final_score:.2f}",
            ))

            return result

        except Exception as e:
            logger.error(f"Extraction failed for session {session_id}: {e}")
            self._emit_progress(ExtractionProgress(
                session_id=session_id,
                phase="failed",
                progress=0.0,
                message=f"Extraction failed: {e}",
                error=str(e),
            ))

            # Mark session as failed
            try:
                self._session_service.complete_session(
                    session_id=session_id,
                    success=False,
                    error=str(e),
                )
            except Exception as complete_error:
                logger.error(f"Failed to mark session as failed: {complete_error}")

            return None

        finally:
            # Clear intermediate CUDA tensors (activations, gradients)
            # but keep the model cached for reuse
            self._clear_cuda_cache()

    def _clear_cuda_cache(self) -> None:
        """Clear CUDA cache to free intermediate tensors.

        This frees activations and gradients from extraction
        while keeping the model itself cached.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")

    def _free_cached_model(self) -> None:
        """Free the cached model from GPU memory.

        Called when switching to a different model.
        """
        if self._cached_backend is None:
            return

        logger.info(f"Freeing cached model: {self._cached_model_id}")

        try:
            if hasattr(self._cached_backend, 'model'):
                del self._cached_backend.model
            if hasattr(self._cached_backend, 'tokenizer'):
                del self._cached_backend.tokenizer
            del self._cached_backend
        except Exception as e:
            logger.warning(f"Error freeing model: {e}")

        self._cached_backend = None
        self._cached_model_id = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    async def _load_target_model(self, model_id: str) -> Any:
        """Load the target HuggingFace model.

        Args:
            model_id: HuggingFace model identifier.

        Returns:
            HuggingFaceBackend instance.
        """
        from steering_vectors import HuggingFaceBackend

        logger.info(f"Loading target model: {model_id}")

        # Run model loading in executor to avoid blocking
        loop = asyncio.get_event_loop()

        def load_model():
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_id)

            # Check if accelerate is available for device_map
            has_accelerate = False
            try:
                import accelerate  # noqa: F401
                has_accelerate = True
            except ImportError:
                pass

            if has_accelerate:
                # Use accelerate for automatic device mapping
                logger.info("Using accelerate for automatic device mapping")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    dtype=torch.float16,
                    device_map="auto",
                )
            elif torch.cuda.is_available():
                # Load to GPU without accelerate
                logger.info("Loading model to GPU (no accelerate)")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    dtype=torch.float16,
                )
                model = model.cuda()
            else:
                # CPU fallback
                logger.info("Loading model to CPU")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    dtype=torch.float32,
                )

            # Enable gradient checkpointing for memory efficiency
            # This reduces activation memory by ~50% at cost of ~30% slower backward
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing for memory efficiency")

            return HuggingFaceBackend(
                model=model,
                tokenizer=tokenizer,
                gradient_checkpointing=True,
            )

        backend = await loop.run_in_executor(None, load_model)
        logger.info(f"Model loaded: {model_id}")

        return backend

    def start_extraction(
        self,
        session_id: str,
        behavior_name: str,
        behavior_description: str,
        config: TaskConfig,
    ) -> asyncio.Task:
        """Start an extraction in the background.

        Args:
            session_id: The session ID.
            behavior_name: Name of the behavior.
            behavior_description: Full description.
            config: Task configuration.

        Returns:
            The asyncio Task running the extraction.
        """
        task = asyncio.create_task(
            self.run_extraction(
                session_id=session_id,
                behavior_name=behavior_name,
                behavior_description=behavior_description,
                config=config,
            )
        )
        self._running_extractions[session_id] = task

        # Clean up when done
        def cleanup(_):
            if session_id in self._running_extractions:
                del self._running_extractions[session_id]

        task.add_done_callback(cleanup)

        return task

    def cancel_extraction(self, session_id: str) -> bool:
        """Cancel a running extraction.

        Args:
            session_id: The session to cancel.

        Returns:
            True if cancelled, False if not running.
        """
        task = self._running_extractions.get(session_id)
        if task is None:
            return False

        task.cancel()
        return True

    @property
    def running_session_ids(self) -> List[str]:
        """Get list of currently running extraction session IDs."""
        return list(self._running_extractions.keys())

    @property
    def cached_model_id(self) -> Optional[str]:
        """Get the currently cached model ID, if any."""
        return self._cached_model_id

    def clear_model_cache(self) -> None:
        """Explicitly clear the cached model from GPU memory.

        Call this when you want to free GPU memory, such as
        when closing the app or before loading a very large model.
        """
        self._free_cached_model()
