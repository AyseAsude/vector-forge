"""Extraction runner service for orchestrating the full extraction pipeline.

This service handles the complete flow from task creation to result:
1. Load target HuggingFace model
2. Create LLM clients for extractor and judge
3. Run ContrastPipeline to generate sample datasets
4. Run TaskExecutor to perform extraction
5. Handle errors and cleanup
"""

import asyncio
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
            self._emit_progress(ExtractionProgress(
                session_id=session_id,
                phase="loading_model",
                progress=0.0,
                message="Loading target model...",
            ))

            # Step 1: Load target model
            model_backend = await self._load_target_model(config.target_model)

            self._emit_progress(ExtractionProgress(
                session_id=session_id,
                phase="generating_contrast",
                progress=0.1,
                message="Generating contrast pairs...",
            ))

            # Step 2: Create LLM clients with event logging
            from vector_forge.services.task_executor import EventEmittingLLMClient

            store = self._session_service.get_session_store(session_id)
            raw_extractor = create_client(config.extractor_model)
            raw_judge = create_client(config.judge_model)

            # Wrap with event emission so ALL LLM calls are logged
            extractor_llm = EventEmittingLLMClient(raw_extractor, store, source="contrast_extractor")
            judge_llm = EventEmittingLLMClient(raw_judge, store, source="contrast_judge")

            # Step 3: Build contrast config from task config
            contrast_config = ContrastPipelineConfig(
                core_pool_size=config.contrast.core_pool_size,
                core_seeds_per_sample=config.contrast.core_seeds_per_sample,
                unique_seeds_per_sample=config.contrast.unique_seeds_per_sample,
                min_semantic_distance=config.contrast.min_semantic_distance,
                min_dst_score=config.contrast.min_dst_score,
                max_src_score=config.contrast.max_src_score,
                min_contrast_quality=config.contrast.min_contrast_quality,
                max_regeneration_attempts=config.contrast.max_regeneration_attempts,
                max_concurrent_generations=config.contrast.max_concurrent_generations,
            )

            # Step 4: Run ContrastPipeline
            contrast_pipeline = ContrastPipeline(
                llm_client=extractor_llm,
                judge_llm_client=judge_llm,
                config=contrast_config,
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

            # Step 5: Create ExtractionTask
            behavior = ExpandedBehavior(
                name=behavior_name,
                description=behavior_description,
                detailed_definition=behavior_description,
            )
            task = ExtractionTask.from_behavior(behavior, config)

            # Step 6: Set up progress forwarding
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
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
            elif torch.cuda.is_available():
                # Load to GPU without accelerate
                logger.info("Loading model to GPU (no accelerate)")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                )
                model = model.cuda()
            else:
                # CPU fallback
                logger.info("Loading model to CPU")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                )

            return HuggingFaceBackend(model=model, tokenizer=tokenizer)

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
