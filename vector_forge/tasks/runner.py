"""Parallel task runner for extraction tasks.

Executes extraction tasks with configurable parallelism, supporting both:
- CAA (Contrastive Activation Addition) - fast, deterministic, recommended
- Gradient optimization - slower, can find arbitrary directions

Architecture:
- Profile-based memory management (measures actual usage, not estimates)
- Memory-aware scheduling with OOM protection
- Full reproducibility via comprehensive result tracking
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Protocol, TYPE_CHECKING
import asyncio
import time
import logging
import math
import random
import uuid

if TYPE_CHECKING:
    from vector_forge.storage.emitter import EventEmitter

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from steering_vectors import (
    SteeringOptimizer,
    VectorSteering,
    HuggingFaceBackend,
    TrainingDatapoint,
    # CAA extraction
    extract as caa_extract,
    ContrastPair,
)
from steering_vectors.core.config import OptimizationConfig as SVOptimizationConfig
from steering_vectors.optimization.callbacks import (
    HistoryCallback,
    ConvergenceCallback,
    EarlyStoppingCallback,
)

from vector_forge.tasks.config import (
    TaskConfig,
    LayerStrategy,
    AggregationStrategy,
    ExtractionMethod,
)
from vector_forge.core.concurrency import (
    get_concurrency_manager,
    limit_concurrent_evaluations,
)
from vector_forge.tasks.sample import ExtractionSample
from vector_forge.tasks.task import ExtractionTask, TaskResult, SampleResult
from vector_forge.tasks.evaluation import VectorEvaluator
from vector_forge.tasks.expander import ExpandedBehavior
from vector_forge.tasks.adapter import (
    ContrastToTrainingAdapter,
    OptimizationResultData,
    DatapointSerializer,
)
from vector_forge.tasks.gpu_memory import (
    ExtractionMemoryProfiler,
    MemoryAwareSemaphore,
    OOMHandler,
    clear_gpu_memory,
)
from vector_forge.contrast.protocols import ValidatedPair, SampleDataset

logger = logging.getLogger(__name__)


# ============================================================================
# Protocols
# ============================================================================


class ModelBackend(Protocol):
    """Protocol for model backend operations."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        ...

    def generate_with_steering(
        self,
        prompt: str,
        steering_vector: torch.Tensor,
        layer: int,
        strength: float,
        max_new_tokens: int = 100,
    ) -> str:
        ...


class LLMClient(Protocol):
    """Protocol for LLM client operations."""

    async def generate(self, messages: List[dict], **kwargs) -> str:
        ...


# ============================================================================
# Progress Tracking
# ============================================================================


@dataclass
class RunnerProgress:
    """Progress information for task execution."""

    total_samples: int
    completed_extractions: int
    completed_evaluations: int
    failed_count: int
    current_phase: str
    elapsed_seconds: float
    estimated_remaining_seconds: float

    @property
    def extraction_progress(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.completed_extractions / self.total_samples

    @property
    def evaluation_progress(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.completed_evaluations / self.total_samples


# ============================================================================
# Optimization Result Container
# ============================================================================


@dataclass
class ExtractionResult:
    """Result from a single vector extraction.

    Contains all information needed for reproducibility and analysis.
    """

    vector: Optional[torch.Tensor]
    layer: int
    final_loss: float = 0.0
    iterations: int = 0
    loss_history: List[float] = field(default_factory=list)
    datapoints_used: int = 0
    config_used: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.vector is not None and self.error is None

    def to_result_data(self) -> OptimizationResultData:
        """Convert to serializable format."""
        return OptimizationResultData(
            vector_shape=tuple(self.vector.shape) if self.vector is not None else (),
            layer=self.layer,
            final_loss=self.final_loss,
            iterations=self.iterations,
            loss_history=self.loss_history,
            config=self.config_used,
            metadata={"datapoints_used": self.datapoints_used},
        )


# ============================================================================
# Task Runner
# ============================================================================


class TaskRunner:
    """Parallel execution engine for extraction tasks.

    Uses the optimization-based approach from steering-vectors library
    for high-quality steering vector extraction.

    Features:
    - Profile-based memory management (measures actual usage, not estimates)
    - Memory-aware scheduling with automatic OOM recovery
    - Batched optimization for efficient forward passes

    Example:
        >>> runner = TaskRunner(backend, llm, max_concurrent=8)
        >>> result = await runner.run(task, sample_datasets)
        >>> print(f"Best score: {result.final_score}")
    """

    def __init__(
        self,
        model_backend: ModelBackend,
        llm_client: LLMClient,
        max_concurrent_extractions: int = 8,
        max_concurrent_evaluations: int = 16,
        event_emitter: Optional["EventEmitter"] = None,
    ) -> None:
        """Initialize the task runner.

        Args:
            model_backend: Backend for model inference (must have model/tokenizer).
            llm_client: Client for LLM API calls.
            max_concurrent_extractions: Max parallel extraction workers.
            max_concurrent_evaluations: Max parallel evaluation generations.
            event_emitter: Optional event emitter for complete event sourcing.
        """
        self._backend = model_backend
        self._llm = llm_client
        self._max_extractions = max_concurrent_extractions
        self._max_evaluations = max_concurrent_evaluations
        self._progress_callback: Optional[Callable[[RunnerProgress], None]] = None
        self._emitter = event_emitter

        # Create steering-vectors backend with gradient checkpointing
        self._sv_backend = HuggingFaceBackend(
            model=model_backend.model,
            tokenizer=model_backend.tokenizer,
            gradient_checkpointing=True,
        )

        # Memory management components
        self._memory_profiler = ExtractionMemoryProfiler(self._sv_backend)
        self._oom_handler = OOMHandler()

        # Cache model info
        self._num_layers = self._sv_backend.get_num_layers()
        self._hidden_dim = self._sv_backend.get_hidden_dim()

        logger.info(
            f"TaskRunner initialized: {self._num_layers} layers, "
            f"hidden_dim={self._hidden_dim}"
        )

    def on_progress(self, callback: Callable[[RunnerProgress], None]) -> None:
        """Register a progress callback."""
        self._progress_callback = callback

    def _emit(self, method_name: str, **kwargs) -> None:
        """Emit an event if emitter is available."""
        if self._emitter is not None:
            method = getattr(self._emitter, method_name, None)
            if method:
                try:
                    method(**kwargs)
                except Exception as e:
                    logger.warning(f"Event emission failed: {e}")

    def _generate_id(self, prefix: str = "") -> str:
        """Generate a unique ID."""
        return f"{prefix}_{uuid.uuid4().hex[:12]}" if prefix else uuid.uuid4().hex[:12]

    @property
    def num_layers(self) -> int:
        """Number of layers in the model."""
        return self._num_layers

    @property
    def hidden_dim(self) -> int:
        """Hidden dimension of the model."""
        return self._hidden_dim

    async def run(
        self,
        task: ExtractionTask,
        sample_datasets: Dict[int, SampleDataset],
    ) -> TaskResult:
        """Execute an extraction task with provided datasets.

        Args:
            task: The extraction task to run.
            sample_datasets: Pre-generated datasets for each sample.

        Returns:
            Aggregated task result with final vector.
        """
        started_at = time.time()
        config = task.config

        logger.info(
            f"Starting task: {task.behavior.name} with {task.num_samples} samples"
        )

        # Phase 1: Parallel extraction
        extraction_results = await self._run_extractions(
            task, sample_datasets
        )

        # Phase 2: Parallel evaluation (skip if fails)
        try:
            evaluated_results = await self._run_evaluations(
                extraction_results,
                task.behavior,
                config,
            )
        except Exception as e:
            logger.warning(f"Evaluation failed, skipping: {e}")
            evaluated_results = extraction_results

        # Phase 3: Aggregation
        final_result = self._aggregate_results(
            evaluated_results,
            task.behavior,
            config,
            sample_datasets,
        )

        final_result.started_at = started_at
        final_result.completed_at = time.time()

        logger.info(
            f"Task complete: score={final_result.final_score:.2f}, "
            f"duration={final_result.duration_seconds:.1f}s"
        )

        return final_result

    async def run_single_extraction(
        self,
        sample: ExtractionSample,
        datapoints: List[TrainingDatapoint],
        config: TaskConfig,
    ) -> ExtractionResult:
        """Run extraction for a single sample.

        Exposed for testing and custom workflows.

        Args:
            sample: The extraction sample configuration.
            datapoints: Training datapoints for optimization.
            config: Task configuration.

        Returns:
            ExtractionResult with vector and metadata.
        """
        return await self._extract_vector(sample, datapoints, config)

    async def _run_extractions(
        self,
        task: ExtractionTask,
        sample_datasets: Dict[int, SampleDataset],
    ) -> List[SampleResult]:
        """Run all extractions with profile-based memory management.

        Uses actual memory profiling to determine safe concurrency, with
        memory-aware scheduling and automatic OOM recovery.
        """
        adapter = ContrastToTrainingAdapter()
        batch_size = task.config.optimization.batch_size

        # Prepare sample datapoints for profiling
        profile_datapoints = self._get_profile_datapoints(
            task, sample_datasets, adapter
        )

        # Profile actual memory usage (cached after first call)
        # CAA only uses forward passes (no gradients), so profile accordingly
        is_forward_only = task.config.extraction_method == ExtractionMethod.CAA
        memory_profile = self._memory_profiler.profile(
            datapoints=profile_datapoints,
            batch_size=batch_size,
            layer=self._num_layers // 2,
            forward_only=is_forward_only,
        )

        # Create memory-aware semaphore based on profiled memory
        safe_concurrency = min(
            memory_profile.safe_concurrent_extractions,
            self._max_extractions,
        )
        semaphore = MemoryAwareSemaphore(
            memory_per_extraction_gb=memory_profile.memory_per_extraction_gb,
            max_concurrent=safe_concurrency,
        )

        mode = "forward-only" if is_forward_only else "gradient"
        logger.info(
            f"Running {len(task.samples)} extractions ({mode}) with concurrency: {safe_concurrency} "
            f"(profiled: {memory_profile.memory_per_extraction_gb:.3f}GB/extraction, "
            f"free: {memory_profile.free_memory_gb:.1f}GB)"
        )

        completed = 0
        started_at = time.time()

        async def extract_one(
            sample: ExtractionSample,
            sample_idx: int,
        ) -> SampleResult:
            nonlocal completed

            async with semaphore:
                # Clear GPU memory after acquiring slot
                clear_gpu_memory()

                start_time = time.time()

                # Get dataset for this sample
                dataset = sample_datasets.get(sample_idx)
                if dataset is None:
                    logger.warning(f"No dataset for sample {sample_idx}")
                    return SampleResult(
                        sample=sample,
                        vector=None,
                        layer=0,
                        metadata={"error": "No dataset"},
                        extraction_time_seconds=0,
                    )

                # Check for empty valid_pairs
                if not dataset.valid_pairs:
                    logger.warning(
                        f"Sample {sample_idx} has no valid pairs! "
                        f"Dataset has {len(dataset.all_pairs)} total pairs."
                    )
                    return SampleResult(
                        sample=sample,
                        vector=None,
                        layer=0,
                        metadata={"error": "No valid pairs in dataset"},
                        extraction_time_seconds=0,
                    )

                # Convert to training datapoints with bootstrap
                datapoints = adapter.convert_with_bootstrap(
                    dataset.valid_pairs,
                    ratio=sample.config.bootstrap_ratio,
                    seed=sample.config.seed,
                )

                logger.debug(
                    f"Sample {sample_idx}: {len(dataset.valid_pairs)} valid pairs -> "
                    f"{len(datapoints)} datapoints"
                )

                # Limit to configured datapoints_per_sample
                if len(datapoints) > task.config.datapoints_per_sample:
                    random.seed(sample.config.seed)
                    datapoints = random.sample(
                        datapoints, task.config.datapoints_per_sample
                    )

                # Emit datapoint events for traceability
                self._emit_datapoint_events(datapoints, sample_idx)

                try:
                    # Run extraction with OOM protection
                    extraction = await self._oom_handler.run_with_protection(
                        lambda: self._extract_vector_sync(
                            sample, datapoints, task.config
                        ),
                        cleanup_fn=clear_gpu_memory,
                    )
                    extraction_time = time.time() - start_time

                    result = SampleResult(
                        sample=sample,
                        vector=extraction.vector,
                        layer=extraction.layer,
                        extraction_time_seconds=extraction_time,
                        metadata={
                            "final_loss": extraction.final_loss,
                            "iterations": extraction.iterations,
                            "loss_history": extraction.loss_history,
                            "datapoints_used": extraction.datapoints_used,
                            "config": extraction.config_used,
                        },
                    )
                except Exception as e:
                    logger.warning(f"Extraction failed for {sample.sample_id}: {e}")
                    result = SampleResult(
                        sample=sample,
                        vector=None,
                        layer=0,
                        metadata={"error": str(e)},
                        extraction_time_seconds=time.time() - start_time,
                    )

                completed += 1
                failed_count = 0 if result.is_valid else 1
                self._report_progress(
                    total=len(task.samples),
                    extractions=completed,
                    evaluations=0,
                    failed=failed_count,
                    phase="extracting",
                    started=started_at,
                )

                return result

        # Run all extractions
        tasks = [
            extract_one(sample, idx)
            for idx, sample in enumerate(task.samples)
        ]
        results = await asyncio.gather(*tasks)

        valid_count = sum(1 for r in results if r.is_valid)
        logger.info(f"Extraction complete: {valid_count}/{len(results)} successful")

        return list(results)

    def _get_profile_datapoints(
        self,
        task: ExtractionTask,
        sample_datasets: Dict[int, SampleDataset],
        adapter: ContrastToTrainingAdapter,
    ) -> List[TrainingDatapoint]:
        """Get a small set of datapoints for memory profiling.

        Uses the first available sample's data to profile memory usage.
        """
        for idx, sample in enumerate(task.samples):
            dataset = sample_datasets.get(idx)
            if dataset and dataset.valid_pairs:
                datapoints = adapter.convert_with_bootstrap(
                    dataset.valid_pairs[:3],  # Use subset for profiling
                    ratio=1.0,
                    seed=sample.config.seed,
                )
                if datapoints:
                    return datapoints[:4]  # Limit to 4 datapoints

        # Fallback: create minimal dummy datapoints
        logger.warning("No valid datapoints found for profiling, using dummy data")
        return [
            TrainingDatapoint(
                prompt="Test prompt for memory profiling.",
                dst_completions=["Positive response."],
                src_completions=["Negative response."],
            )
            for _ in range(2)
        ]

    def _emit_datapoint_events(
        self,
        datapoints: List[TrainingDatapoint],
        sample_idx: int,
    ) -> None:
        """Emit events for datapoints (extracted for cleaner code)."""
        for dp in datapoints:
            dst = dp.dst_completions[0] if dp.dst_completions else ""
            src = dp.src_completions[0] if dp.src_completions else None
            self._emit(
                "emit_datapoint_added",
                datapoint_id=self._generate_id("dp"),
                prompt=dp.prompt[:500] if dp.prompt else "",
                positive_completion=dst[:500] if dst else "",
                negative_completion=src[:500] if src else None,
                domain=f"sample_{sample_idx}",
                format_type="contrast_pair",
            )

    def _extract_vector_sync(
        self,
        sample: ExtractionSample,
        datapoints: List[TrainingDatapoint],
        config: TaskConfig,
    ) -> ExtractionResult:
        """Synchronous vector extraction (called by OOMHandler).

        Supports both CAA and gradient-based extraction based on config.
        """
        sample_idx = sample.sample_index

        if not datapoints:
            self._emit(
                "emit_optimization_completed",
                sample_idx=sample_idx,
                layer=0,
                final_loss=0.0,
                iterations=0,
                loss_history=[],
                datapoints_used=0,
                duration_seconds=0.0,
                success=False,
                error="No datapoints provided",
            )
            return ExtractionResult(
                vector=None,
                layer=0,
                error="No datapoints provided",
            )

        start_time = time.time()

        # Set random seed for reproducibility
        torch.manual_seed(sample.config.seed)

        # Determine target layer
        layer = self._get_target_layer(sample)

        # Branch based on extraction method
        if config.extraction_method == ExtractionMethod.CAA:
            return self._extract_caa(sample, datapoints, config, layer, start_time)
        else:
            return self._extract_gradient(sample, datapoints, config, layer, start_time)

    def _extract_caa(
        self,
        sample: ExtractionSample,
        datapoints: List[TrainingDatapoint],
        config: TaskConfig,
        layer: int,
        start_time: float,
    ) -> ExtractionResult:
        """Extract vector using CAA (Contrastive Activation Addition).

        CAA is fast, deterministic, and recommended for most use cases.
        Uses sample's token_position (assigned by generator to explore all positions).
        """
        sample_idx = sample.sample_index
        token_position = sample.config.token_position

        # Emit extraction started event
        self._emit(
            "emit_optimization_started",
            sample_idx=sample_idx,
            layer=layer,
            num_datapoints=len(datapoints),
            config={
                "method": "caa",
                "token_position": token_position.value,
                "seed": sample.config.seed,
            },
        )

        try:
            # Convert TrainingDatapoints to ContrastPairs
            pairs = self._datapoints_to_contrast_pairs(datapoints)

            # Run CAA extraction with sample's token_position
            result = caa_extract(
                backend=self._sv_backend,
                tokenizer=self._backend.tokenizer,
                pairs=pairs,
                layer=layer,
                method="caa",
                token_position=token_position.value,
                remove_outliers=config.caa.remove_extreme_outliers,
                outlier_std_threshold=config.caa.outlier_std_threshold,
            )

            # Normalize CAA vector to starting_norm (consistent with HYBRID)
            # Raw CAA vectors have unpredictable norms (2-4x expected), which
            # causes inconsistent steering strength across different samples.
            vector = result.vector
            original_norm = vector.norm().item()
            target_norm = config.optimization.starting_norm
            if original_norm > 0:
                vector = vector * (target_norm / original_norm)
            normalized_norm = vector.norm().item()

            duration = time.time() - start_time

            # Emit completed event
            self._emit(
                "emit_optimization_completed",
                sample_idx=sample_idx,
                layer=layer,
                final_loss=0.0,  # CAA has no loss
                iterations=1,  # Single pass
                loss_history=[],
                datapoints_used=len(datapoints),
                duration_seconds=duration,
                success=True,
                error=None,
            )

            return ExtractionResult(
                vector=vector,
                layer=layer,
                final_loss=0.0,
                iterations=1,
                loss_history=[],
                datapoints_used=result.metadata.get("num_pairs_used", len(datapoints)),
                config_used={
                    "method": "caa",
                    "token_position": token_position.value,
                    "layer": layer,
                    "seed": sample.config.seed,
                    "original_norm": original_norm,
                    "normalized_norm": normalized_norm,
                    "num_pairs": len(pairs),
                    "num_pairs_used": result.metadata.get("num_pairs_used", len(pairs)),
                    "outliers_removed": result.metadata.get("outliers_removed", 0),
                },
            )

        except Exception as e:
            duration = time.time() - start_time
            self._emit(
                "emit_optimization_completed",
                sample_idx=sample_idx,
                layer=layer,
                final_loss=0.0,
                iterations=0,
                loss_history=[],
                datapoints_used=len(datapoints),
                duration_seconds=duration,
                success=False,
                error=str(e),
            )
            raise

    def _extract_gradient(
        self,
        sample: ExtractionSample,
        datapoints: List[TrainingDatapoint],
        config: TaskConfig,
        layer: int,
        start_time: float,
    ) -> ExtractionResult:
        """Extract vector using gradient optimization.

        Gradient optimization can find arbitrary directions. Consider using
        CAA instead, or use hybrid mode (CAA init + gradient refinement).
        """
        sample_idx = sample.sample_index

        # Build optimization config
        opt_config = config.optimization
        sv_config = SVOptimizationConfig(
            lr=sample.config.get_lr(opt_config.lr),
            max_iters=sample.config.get_max_iters(opt_config.max_iters),
            coldness=opt_config.coldness,
            starting_norm=opt_config.starting_norm,
            max_norm=opt_config.max_norm,
            normalize_by_length=opt_config.normalize_by_length,
            use_one_minus=opt_config.use_one_minus,
            use_batched=opt_config.use_batched,
            batch_size=opt_config.batch_size,
        )

        # Emit optimization started event
        self._emit(
            "emit_optimization_started",
            sample_idx=sample_idx,
            layer=layer,
            num_datapoints=len(datapoints),
            config={
                "method": config.extraction_method.value,
                "lr": sv_config.lr,
                "max_iters": sv_config.max_iters,
                "coldness": sv_config.coldness,
                "starting_norm": sv_config.starting_norm,
                "max_norm": sv_config.max_norm,
                "seed": sample.config.seed,
            },
        )

        # Set up callbacks for tracking
        history_callback = HistoryCallback(record_vectors=False)
        callbacks = [history_callback]

        if opt_config.target_loss is not None:
            callbacks.append(EarlyStoppingCallback(
                target_loss=opt_config.target_loss,
                patience=1,
            ))

        callbacks.append(ConvergenceCallback(
            eps=opt_config.convergence_eps,
            patience=opt_config.convergence_patience,
        ))

        # Create optimizer and run
        steering = VectorSteering()

        # For hybrid mode, initialize from CAA
        if config.extraction_method == ExtractionMethod.HYBRID:
            try:
                pairs = self._datapoints_to_contrast_pairs(datapoints)
                caa_result = caa_extract(
                    backend=self._sv_backend,
                    tokenizer=self._backend.tokenizer,
                    pairs=pairs,
                    layer=layer,
                    method="caa",
                    token_position=config.token_position.value,
                )
                # Initialize steering vector from CAA, normalized to starting_norm
                caa_vec = caa_result.vector
                caa_norm = caa_vec.norm()
                if caa_norm > 0:
                    caa_vec = caa_vec * (opt_config.starting_norm / caa_norm)
                steering.vector = caa_vec.clone().requires_grad_(True)
            except Exception as e:
                logger.warning(f"CAA init failed for hybrid, using random init: {e}")

        optimizer = SteeringOptimizer(
            backend=self._sv_backend,
            steering_mode=steering,
            config=sv_config,
            callbacks=callbacks,
        )

        try:
            result = optimizer.optimize(datapoints, layer=layer)
            duration = time.time() - start_time

            # Emit optimization completed event
            self._emit(
                "emit_optimization_completed",
                sample_idx=sample_idx,
                layer=layer,
                final_loss=result.final_loss,
                iterations=result.iterations,
                loss_history=history_callback.losses[:50],
                datapoints_used=len(datapoints),
                duration_seconds=duration,
                success=True,
                error=None,
            )

            return ExtractionResult(
                vector=result.vector,
                layer=layer,
                final_loss=result.final_loss,
                iterations=result.iterations,
                loss_history=history_callback.losses,
                datapoints_used=len(datapoints),
                config_used={
                    "method": config.extraction_method.value,
                    "lr": sv_config.lr,
                    "max_iters": sv_config.max_iters,
                    "coldness": sv_config.coldness,
                    "starting_norm": sv_config.starting_norm,
                    "max_norm": sv_config.max_norm,
                    "layer": layer,
                    "seed": sample.config.seed,
                },
            )

        except Exception as e:
            duration = time.time() - start_time
            self._emit(
                "emit_optimization_completed",
                sample_idx=sample_idx,
                layer=layer,
                final_loss=0.0,
                iterations=0,
                loss_history=history_callback.losses[:50] if history_callback.losses else [],
                datapoints_used=len(datapoints),
                duration_seconds=duration,
                success=False,
                error=str(e),
            )
            raise

    def _datapoints_to_contrast_pairs(
        self,
        datapoints: List[TrainingDatapoint],
    ) -> List[ContrastPair]:
        """Convert TrainingDatapoints to ContrastPairs for CAA extraction."""
        pairs = []
        for dp in datapoints:
            # Get completions
            positive = dp.dst_completions[0] if dp.dst_completions else ""
            negative = dp.src_completions[0] if dp.src_completions else ""

            if not positive or not negative:
                continue

            # Create ContrastPair from prompt + completion format
            pair = ContrastPair.from_prompt_completion(
                prompt=dp.prompt,
                positive_completion=positive,
                negative_completion=negative,
            )
            pairs.append(pair)

        return pairs

    async def _extract_vector(
        self,
        sample: ExtractionSample,
        datapoints: List[TrainingDatapoint],
        config: TaskConfig,
    ) -> ExtractionResult:
        """Extract a steering vector using optimization (async wrapper).

        This is an async wrapper around _extract_vector_sync for use in
        contexts that need async compatibility (e.g., run_single_extraction).

        Args:
            sample: The extraction sample configuration.
            datapoints: Training datapoints.
            config: Task configuration.

        Returns:
            ExtractionResult with vector and metadata.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._extract_vector_sync(sample, datapoints, config),
        )

    def _get_target_layer(self, sample: ExtractionSample) -> int:
        """Determine target layer based on sample configuration."""
        if sample.config.target_layers:
            # Randomly select from target layers based on sample seed
            layers = sample.config.target_layers
            random.seed(sample.config.seed)
            return random.choice(layers)

        strategy = sample.config.layer_strategy
        total = self._num_layers

        if strategy == LayerStrategy.AUTO:
            # Middle layer (commonly effective)
            return total // 2
        elif strategy == LayerStrategy.SWEEP:
            # Sample across middle range
            # Use seed to pick one from the sweep
            random.seed(sample.config.seed)
            sweep_start = total // 4
            sweep_end = 3 * total // 4
            return random.randint(sweep_start, sweep_end)
        elif strategy == LayerStrategy.MIDDLE:
            return total // 2
        elif strategy == LayerStrategy.LATE:
            return 3 * total // 4
        else:
            return total // 2

    def get_target_layers_for_strategy(
        self,
        strategy: LayerStrategy,
    ) -> List[int]:
        """Get all target layers for a strategy (for UI display)."""
        total = self._num_layers

        if strategy == LayerStrategy.AUTO:
            mid = total // 2
            return [mid - 1, mid, mid + 1]
        elif strategy == LayerStrategy.SWEEP:
            start = total // 4
            end = 3 * total // 4
            return list(range(start, end, 2))
        elif strategy == LayerStrategy.MIDDLE:
            mid = total // 2
            return [mid - 1, mid, mid + 1]
        elif strategy == LayerStrategy.LATE:
            return list(range(3 * total // 4, total - 2))
        else:
            return [total // 2]

    async def _run_evaluations(
        self,
        results: List[SampleResult],
        behavior: ExpandedBehavior,
        config: TaskConfig,
    ) -> List[SampleResult]:
        """Run evaluations on all valid extraction results."""
        valid_results = [r for r in results if r.is_valid]

        if not valid_results:
            logger.warning("No valid extractions to evaluate")
            return results

        evaluator = VectorEvaluator(
            self._backend,
            self._llm,
            config.evaluation,
            event_emitter=self._emitter,
            max_concurrent_judge_calls=config.max_concurrent_evaluations,
        )

        completed = 0
        started_at = time.time()

        async def evaluate_one(result: SampleResult) -> Optional[Exception]:
            """Evaluate a single result, returning None on success or Exception on failure.

            Uses limit_concurrent_evaluations() to prevent thread pool starvation
            from too many concurrent GPU operations.
            """
            nonlocal completed

            # Limit concurrent evaluations to prevent thread pool starvation
            async with limit_concurrent_evaluations():
                start_time = time.time()
                eval_id = self._generate_id("eval")
                sample_idx = result.sample.sample_index

                # Set evaluation ID for event tracking within evaluator
                evaluator.set_evaluation_id(eval_id)

                # Emit evaluation started event
                self._emit(
                    "emit_evaluation_started",
                    evaluation_id=eval_id,
                    eval_type="comprehensive",
                    vector_id=f"vector_{sample_idx}",
                    layer=result.layer,
                    strength_levels=config.evaluation.strength_levels,
                    num_prompts=config.evaluation.behavior_prompts,
                    dimensions=["behavior", "specificity", "coherence", "capability", "generalization"],
                    sample_idx=sample_idx,
                )

                try:
                    evaluation = await evaluator.evaluate(
                        result.vector,
                        result.layer,
                        behavior,
                        config.max_concurrent_evaluations,
                    )

                    result.evaluation = evaluation
                    result.overall_score = evaluation.overall_score
                    result.scores = evaluation.scores_dict
                    result.recommended_strength = evaluation.recommended_strength
                    result.evaluation_time_seconds = time.time() - start_time

                    # Emit evaluation completed event
                    self._emit(
                        "emit_evaluation_completed",
                        evaluation_id=eval_id,
                        scores=evaluation.scores_dict,
                        dimension_scores={
                            "behavior": evaluation.behavior_score.score if evaluation.behavior_score else 0,
                            "specificity": evaluation.specificity_score.score if evaluation.specificity_score else 0,
                            "coherence": evaluation.coherence_score.score if evaluation.coherence_score else 0,
                            "capability": evaluation.capability_score.score if evaluation.capability_score else 0,
                            "generalization": evaluation.generalization_score.score if evaluation.generalization_score else 0,
                        },
                        recommended_strength=evaluation.recommended_strength,
                        verdict="passed" if evaluation.overall_score > 0.5 else "needs_refinement",
                        citations=None,
                        recommendations=None,
                        raw_judge_output=getattr(evaluation, 'raw_output', None),
                        duration_seconds=time.time() - start_time,
                        total_generations=evaluator._generation_count,
                        total_judge_calls=evaluator._judge_call_count,
                        sample_idx=sample_idx,
                    )

                    completed += 1
                    self._report_progress(
                        total=len(results),
                        extractions=len(results),
                        evaluations=completed,
                        failed=len(results) - len(valid_results),
                        phase="evaluating",
                        started=started_at,
                    )
                    return None

                except Exception as e:
                    logger.warning(f"Evaluation failed for sample {sample_idx}: {e}")
                    # Emit failed evaluation event
                    self._emit(
                        "emit_evaluation_completed",
                        evaluation_id=eval_id,
                        scores={},
                        recommended_strength=1.0,
                        verdict="failed",
                        citations=None,
                        recommendations=None,
                        raw_judge_output=f"Error: {str(e)}",
                        sample_idx=sample_idx,
                    )
                    completed += 1
                    self._report_progress(
                        total=len(results),
                        extractions=len(results),
                        evaluations=completed,
                        failed=len(results) - len(valid_results),
                        phase="evaluating",
                        started=started_at,
                    )
                    return e

        # Use return_exceptions=True to prevent one failure from cancelling others
        eval_results = await asyncio.gather(
            *[evaluate_one(r) for r in valid_results],
            return_exceptions=True,
        )

        # Log any unexpected exceptions (ones not caught inside evaluate_one)
        for i, err in enumerate(eval_results):
            if isinstance(err, Exception):
                logger.error(f"Unexpected evaluation error for result {i}: {err}")

        logger.info(f"Evaluation complete for {len(valid_results)} vectors")
        return results

    def _aggregate_results(
        self,
        results: List[SampleResult],
        behavior: ExpandedBehavior,
        config: TaskConfig,
        sample_datasets: Dict[int, SampleDataset],
    ) -> TaskResult:
        """Aggregate sample results into final task result.

        Args:
            results: List of sample results from extraction.
            behavior: The expanded behavior specification.
            config: Task configuration.
            sample_datasets: Original datasets for reproducibility storage.

        Returns:
            TaskResult with final aggregated vector and full reproducibility data.
        """
        valid_results = [r for r in results if r.is_valid]

        if not valid_results:
            raise ValueError("No valid results to aggregate")

        # Sort by score
        sorted_results = sorted(
            valid_results,
            key=lambda r: r.overall_score,
            reverse=True,
        )

        strategy = config.aggregation_strategy
        top_k = min(config.top_k, len(sorted_results))

        if strategy == AggregationStrategy.BEST_SINGLE:
            final_vector = sorted_results[0].vector
            final_layer = sorted_results[0].layer
            final_score = sorted_results[0].overall_score
            ensemble_components = [sorted_results[0].sample.sample_id]

        elif strategy == AggregationStrategy.TOP_K_AVERAGE:
            top_results = sorted_results[:top_k]
            final_vector = self._average_vectors(
                [r.vector for r in top_results]
            )
            final_layer = top_results[0].layer
            final_score = sum(r.overall_score for r in top_results) / top_k
            ensemble_components = [r.sample.sample_id for r in top_results]

        elif strategy == AggregationStrategy.WEIGHTED_AVERAGE:
            top_results = sorted_results[:top_k]
            weights = [r.overall_score for r in top_results]
            final_vector = self._weighted_average_vectors(
                [r.vector for r in top_results],
                weights,
            )
            final_layer = top_results[0].layer
            final_score = sum(r.overall_score for r in top_results) / top_k
            ensemble_components = [r.sample.sample_id for r in top_results]

        elif strategy == AggregationStrategy.PCA_PRINCIPAL:
            top_results = sorted_results[:top_k]
            final_vector = self._pca_principal(
                [r.vector for r in top_results]
            )
            final_layer = top_results[0].layer
            final_score = sum(r.overall_score for r in top_results) / top_k
            ensemble_components = [r.sample.sample_id for r in top_results]

        elif strategy == AggregationStrategy.STRATEGY_GROUPED:
            # Group results by layer strategy, average within groups, then combine
            final_vector, final_layer, final_score, ensemble_components = (
                self._strategy_grouped_aggregation(sorted_results, top_k)
            )

        else:
            # Default to best single
            final_vector = sorted_results[0].vector
            final_layer = sorted_results[0].layer
            final_score = sorted_results[0].overall_score
            ensemble_components = [sorted_results[0].sample.sample_id]

        # Find recommended strength from best result
        recommended_strength = sorted_results[0].recommended_strength

        # Emit aggregation completed event
        self._emit(
            "emit_aggregation_completed",
            strategy=strategy.value if hasattr(strategy, 'value') else str(strategy),
            num_vectors=len(valid_results),
            top_k=top_k,
            ensemble_components=ensemble_components,
            final_score=final_score,
            final_layer=final_layer,
        )

        # Serialize contrast data for reproducibility
        contrast_data = self._serialize_sample_datasets(sample_datasets)

        return TaskResult(
            behavior=behavior,
            final_vector=final_vector,
            final_layer=final_layer,
            final_score=final_score,
            recommended_strength=recommended_strength,
            aggregation_method=strategy,
            sample_results=results,
            ensemble_components=ensemble_components,
            metadata={
                "valid_count": len(valid_results),
                "total_count": len(results),
                "top_k": top_k,
                "num_layers": self._num_layers,
                "hidden_dim": self._hidden_dim,
            },
            task_config=config,
            contrast_data=contrast_data,
        )

    def _average_vectors(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        """Compute simple average of vectors."""
        stacked = torch.stack(vectors)
        avg = stacked.mean(dim=0)
        return avg / avg.norm()

    def _weighted_average_vectors(
        self,
        vectors: List[torch.Tensor],
        weights: List[float],
    ) -> torch.Tensor:
        """Compute weighted average of vectors."""
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        result = torch.zeros_like(vectors[0])
        for vec, weight in zip(vectors, normalized_weights):
            result += weight * vec

        return result / result.norm()

    def _pca_principal(self, vectors: List[torch.Tensor]) -> torch.Tensor:
        """Extract principal component from vectors."""
        stacked = torch.stack(vectors)

        # Center
        mean = stacked.mean(dim=0)
        centered = stacked - mean

        # SVD to find principal direction
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        principal = vh[0]

        # Ensure consistent orientation
        if (principal @ mean) < 0:
            principal = -principal

        return principal / principal.norm()

    def _strategy_grouped_aggregation(
        self,
        sorted_results: List[SampleResult],
        top_k: int,
    ) -> tuple[torch.Tensor, int, float, List[str]]:
        """Aggregate results by grouping by layer strategy first.

        Groups results by their layer strategy, averages within each group,
        then combines the group averages.

        Args:
            sorted_results: Results sorted by score (descending).
            top_k: Number of top results to consider.

        Returns:
            Tuple of (final_vector, final_layer, final_score, ensemble_components).
        """
        from collections import defaultdict

        # Group by layer strategy
        strategy_groups: Dict[LayerStrategy, List[SampleResult]] = defaultdict(list)
        for result in sorted_results[:top_k]:
            strategy = result.sample.config.layer_strategy
            strategy_groups[strategy].append(result)

        if not strategy_groups:
            # Fallback to best single
            best = sorted_results[0]
            return best.vector, best.layer, best.overall_score, [best.sample.sample_id]

        # Average within each group
        group_vectors: List[torch.Tensor] = []
        group_scores: List[float] = []
        ensemble_components: List[str] = []

        for strategy, group_results in strategy_groups.items():
            if not group_results:
                continue

            # Average vectors in this group
            vectors = [r.vector for r in group_results]
            group_avg = self._average_vectors(vectors)
            group_vectors.append(group_avg)

            # Average score for this group
            avg_score = sum(r.overall_score for r in group_results) / len(group_results)
            group_scores.append(avg_score)

            # Track components
            ensemble_components.extend([r.sample.sample_id for r in group_results])

        # Combine group averages (weighted by group score)
        if len(group_vectors) == 1:
            final_vector = group_vectors[0]
        else:
            final_vector = self._weighted_average_vectors(group_vectors, group_scores)

        # Use the layer from the best result
        final_layer = sorted_results[0].layer
        final_score = sum(group_scores) / len(group_scores)

        return final_vector, final_layer, final_score, ensemble_components

    def _serialize_sample_datasets(
        self,
        sample_datasets: Dict[int, SampleDataset],
    ) -> Dict[str, Any]:
        """Serialize sample datasets for storage.

        Converts SampleDatasets to a JSON-serializable format for full
        reproducibility.

        Args:
            sample_datasets: Dictionary mapping sample index to dataset.

        Returns:
            JSON-serializable dictionary with all contrast data.
        """
        serializer = DatapointSerializer()

        serialized = {
            "num_samples": len(sample_datasets),
            "samples": {},
        }

        for sample_idx, dataset in sample_datasets.items():
            # Serialize valid pairs with full metadata
            pairs_data = []
            for pair in dataset.valid_pairs:
                pair_dict = {
                    "prompt": pair.prompt,
                    "dst": pair.dst,
                    "src": pair.src,
                    "is_valid": pair.is_valid,
                    "attempts": pair.attempts,
                }
                # Add seed info if available
                if pair.seed:
                    pair_dict["seed"] = {
                        "scenario": pair.seed.scenario,
                        "context": pair.seed.context,
                        "quality_score": pair.seed.quality_score,
                    }
                # Add validation info if available
                if pair.validation:
                    pair_dict["validation"] = {
                        "is_valid": pair.validation.is_valid,
                        "contrast_quality": pair.validation.contrast_quality,
                        "scores": pair.validation.evaluated_scores,
                        "weakest_dimension": pair.validation.weakest_dimension,
                    }
                pairs_data.append(pair_dict)

            serialized["samples"][str(sample_idx)] = {
                "sample_id": dataset.sample_id,
                "num_pairs": len(dataset.all_pairs),
                "num_valid": len(dataset.valid_pairs),
                "avg_quality": dataset.avg_contrast_quality,
                "pairs": pairs_data,
            }

        return serialized

    def _report_progress(
        self,
        total: int,
        extractions: int,
        evaluations: int,
        failed: int,
        phase: str,
        started: float,
    ) -> None:
        """Report progress to callback if registered."""
        if self._progress_callback is None:
            return

        elapsed = time.time() - started
        if extractions > 0 and phase == "extracting":
            rate = extractions / elapsed
            remaining = (total - extractions) / rate if rate > 0 else 0
        elif evaluations > 0 and phase == "evaluating":
            rate = evaluations / elapsed
            remaining = (total - evaluations) / rate if rate > 0 else 0
        else:
            remaining = 0

        progress = RunnerProgress(
            total_samples=total,
            completed_extractions=extractions,
            completed_evaluations=evaluations,
            failed_count=failed,
            current_phase=phase,
            elapsed_seconds=elapsed,
            estimated_remaining_seconds=remaining,
        )

        self._progress_callback(progress)


# ============================================================================
# Re-aggregation Support
# ============================================================================


def reaggregate_results(
    sample_results: List[SampleResult],
    strategy: AggregationStrategy,
    top_k: int = 5,
) -> torch.Tensor:
    """Re-aggregate sample results with a different strategy.

    Enables post-hoc experimentation with aggregation without re-running
    the full pipeline.

    Args:
        sample_results: List of sample results with vectors.
        strategy: Aggregation strategy to use.
        top_k: Number of top results to use.

    Returns:
        Aggregated vector.
    """
    valid_results = [r for r in sample_results if r.is_valid]
    if not valid_results:
        raise ValueError("No valid results to aggregate")

    sorted_results = sorted(
        valid_results,
        key=lambda r: r.overall_score,
        reverse=True,
    )

    top_k = min(top_k, len(sorted_results))
    top_results = sorted_results[:top_k]
    vectors = [r.vector for r in top_results]

    if strategy == AggregationStrategy.BEST_SINGLE:
        return vectors[0]

    elif strategy == AggregationStrategy.TOP_K_AVERAGE:
        stacked = torch.stack(vectors)
        avg = stacked.mean(dim=0)
        return avg / avg.norm()

    elif strategy == AggregationStrategy.WEIGHTED_AVERAGE:
        weights = [r.overall_score for r in top_results]
        total_weight = sum(weights)
        result = torch.zeros_like(vectors[0])
        for vec, weight in zip(vectors, weights):
            result += (weight / total_weight) * vec
        return result / result.norm()

    elif strategy == AggregationStrategy.PCA_PRINCIPAL:
        stacked = torch.stack(vectors)
        mean = stacked.mean(dim=0)
        centered = stacked - mean
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        principal = vh[0]
        if (principal @ mean) < 0:
            principal = -principal
        return principal / principal.norm()

    elif strategy == AggregationStrategy.STRATEGY_GROUPED:
        from collections import defaultdict

        # Group by layer strategy
        strategy_groups: Dict[LayerStrategy, List[SampleResult]] = defaultdict(list)
        for result in top_results:
            layer_strategy = result.sample.config.layer_strategy
            strategy_groups[layer_strategy].append(result)

        if not strategy_groups:
            return vectors[0]

        # Average within each group, then combine
        group_vectors: List[torch.Tensor] = []
        group_scores: List[float] = []

        for layer_strategy, group_results in strategy_groups.items():
            if not group_results:
                continue
            group_vecs = [r.vector for r in group_results]
            stacked = torch.stack(group_vecs)
            group_avg = stacked.mean(dim=0)
            group_avg = group_avg / group_avg.norm()
            group_vectors.append(group_avg)
            group_scores.append(sum(r.overall_score for r in group_results) / len(group_results))

        if len(group_vectors) == 1:
            return group_vectors[0]

        # Weighted average of group vectors
        total_weight = sum(group_scores)
        result = torch.zeros_like(group_vectors[0])
        for vec, weight in zip(group_vectors, group_scores):
            result += (weight / total_weight) * vec
        return result / result.norm()

    else:
        return vectors[0]
