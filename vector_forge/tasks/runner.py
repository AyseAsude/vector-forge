"""Parallel task runner for extraction tasks.

Executes extraction tasks with configurable parallelism, managing
concurrent extractions and evaluations efficiently.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, Protocol
import asyncio
import time
import logging

import torch

from vector_forge.tasks.config import TaskConfig, AggregationStrategy
from vector_forge.tasks.sample import ExtractionSample
from vector_forge.tasks.task import ExtractionTask, TaskResult, SampleResult
from vector_forge.tasks.evaluation import VectorEvaluator, EvaluationResult
from vector_forge.tasks.expander import ExpandedBehavior

logger = logging.getLogger(__name__)


class ModelBackend(Protocol):
    """Protocol for model backend operations."""

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


class Extractor(Protocol):
    """Protocol for vector extraction."""

    async def extract(
        self,
        behavior: ExpandedBehavior,
        config: Any,
    ) -> Optional[torch.Tensor]:
        ...

    @property
    def recommended_layer(self) -> int:
        ...


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


class TaskRunner:
    """Parallel execution engine for extraction tasks.

    Manages concurrent execution of extraction samples with configurable
    parallelism. Coordinates extraction, evaluation, and aggregation phases.

    Example:
        >>> runner = TaskRunner(backend, llm, max_concurrent=8)
        >>> result = await runner.run(task)
        >>> print(f"Best score: {result.final_score}")
    """

    def __init__(
        self,
        model_backend: ModelBackend,
        llm_client: LLMClient,
        extractor_factory: Optional[Callable] = None,
        max_concurrent_extractions: int = 8,
        max_concurrent_evaluations: int = 16,
    ) -> None:
        """Initialize the task runner.

        Args:
            model_backend: Backend for model inference.
            llm_client: Client for LLM API calls.
            extractor_factory: Factory for creating extractors.
            max_concurrent_extractions: Max parallel extraction workers.
            max_concurrent_evaluations: Max parallel evaluation generations.
        """
        self._backend = model_backend
        self._llm = llm_client
        self._extractor_factory = extractor_factory
        self._max_extractions = max_concurrent_extractions
        self._max_evaluations = max_concurrent_evaluations
        self._progress_callback: Optional[Callable[[RunnerProgress], None]] = None

    def on_progress(self, callback: Callable[[RunnerProgress], None]) -> None:
        """Register a progress callback.

        Args:
            callback: Function called with progress updates.
        """
        self._progress_callback = callback

    async def run(self, task: ExtractionTask) -> TaskResult:
        """Execute an extraction task.

        Runs all samples in parallel, evaluates results, and aggregates
        into final output.

        Args:
            task: The extraction task to run.

        Returns:
            Aggregated task result with final vector.
        """
        started_at = time.time()
        config = task.config

        logger.info(
            f"Starting task: {task.behavior.name} with {task.num_samples} samples"
        )

        # Phase 1: Parallel extraction
        extraction_results = await self._run_extractions(task)

        # Phase 2: Parallel evaluation
        evaluated_results = await self._run_evaluations(
            extraction_results,
            task.behavior,
            config,
        )

        # Phase 3: Aggregation
        final_result = self._aggregate_results(
            evaluated_results,
            task.behavior,
            config,
        )

        final_result.started_at = started_at
        final_result.completed_at = time.time()

        logger.info(
            f"Task complete: score={final_result.final_score:.2f}, "
            f"duration={final_result.duration_seconds:.1f}s"
        )

        return final_result

    async def _run_extractions(
        self,
        task: ExtractionTask,
    ) -> List[SampleResult]:
        """Run all extractions in parallel.

        Args:
            task: The extraction task.

        Returns:
            List of sample results (may include failures).
        """
        semaphore = asyncio.Semaphore(self._max_extractions)
        results: List[SampleResult] = []
        completed = 0
        started_at = time.time()

        async def extract_one(sample: ExtractionSample) -> SampleResult:
            nonlocal completed
            async with semaphore:
                start_time = time.time()
                try:
                    vector, layer = await self._extract_vector(sample)
                    extraction_time = time.time() - start_time

                    result = SampleResult(
                        sample=sample,
                        vector=vector,
                        layer=layer,
                        extraction_time_seconds=extraction_time,
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
                self._report_progress(
                    total=len(task.samples),
                    extractions=completed,
                    evaluations=0,
                    failed=sum(1 for r in results if not r.is_valid),
                    phase="extracting",
                    started=started_at,
                )

                return result

        # Run all extractions
        tasks = [extract_one(sample) for sample in task.samples]
        results = await asyncio.gather(*tasks)

        valid_count = sum(1 for r in results if r.is_valid)
        logger.info(f"Extraction complete: {valid_count}/{len(results)} successful")

        return results

    async def _extract_vector(
        self,
        sample: ExtractionSample,
    ) -> tuple[Optional[torch.Tensor], int]:
        """Extract a steering vector for one sample.

        Args:
            sample: The extraction sample configuration.

        Returns:
            Tuple of (vector, layer) or (None, 0) on failure.
        """
        # Set random seed for reproducibility
        torch.manual_seed(sample.config.seed * 42)

        if self._extractor_factory is not None:
            extractor = self._extractor_factory(
                llm_client=self._llm,
                model_backend=self._backend,
                behavior=sample.behavior.to_behavior_spec(),
                config=sample.config,
            )
            vector = await extractor.extract(
                sample.behavior,
                sample.config,
            )
            layer = extractor.recommended_layer
            return vector, layer

        # Fallback: use basic extraction
        return await self._basic_extract(sample)

    async def _basic_extract(
        self,
        sample: ExtractionSample,
    ) -> tuple[Optional[torch.Tensor], int]:
        """Basic extraction using steering_vectors library.

        Args:
            sample: The extraction sample.

        Returns:
            Tuple of (vector, layer).
        """
        # Import here to avoid circular dependency
        from steering_vectors import train_steering_vector

        # Generate contrast pairs using LLM
        pairs = await self._generate_contrast_pairs(sample)

        if not pairs:
            return None, 0

        # Determine layers based on strategy
        layers = self._get_target_layers(sample)

        # Train steering vector
        try:
            sv = train_steering_vector(
                self._backend.model,
                self._backend.tokenizer,
                pairs,
                layers=layers,
            )
            # Get vector for middle layer
            mid_layer = layers[len(layers) // 2]
            vector = sv.layer_activations[mid_layer]
            return vector, mid_layer
        except Exception as e:
            logger.error(f"Steering vector training failed: {e}")
            return None, 0

    async def _generate_contrast_pairs(
        self,
        sample: ExtractionSample,
    ) -> List[tuple[str, str]]:
        """Generate contrast pairs for extraction.

        Args:
            sample: The extraction sample.

        Returns:
            List of (positive, negative) text pairs.
        """
        behavior = sample.behavior

        prompt = f"""Generate {sample.config.num_datapoints} contrast pairs for steering vector extraction.

BEHAVIOR: {behavior.name}
DEFINITION: {behavior.detailed_definition}

CONTRAST GUIDANCE: {behavior.contrast_guidance}

DOMAINS TO COVER: {', '.join(behavior.domains[:5])}

For each pair, provide:
1. A POSITIVE example that exhibits the behavior
2. A NEGATIVE example that does NOT exhibit the behavior
3. Both should be responses to the same implied prompt

Format as JSON array:
[{{"positive": "...", "negative": "..."}}, ...]"""

        response = await self._llm.generate(
            [{"role": "user", "content": prompt}],
            temperature=sample.config.temperature,
            max_tokens=4096,
        )

        return self._parse_pairs(response)

    def _parse_pairs(self, response: str) -> List[tuple[str, str]]:
        """Parse contrast pairs from LLM response."""
        import json

        pairs = []
        try:
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                for item in data:
                    if "positive" in item and "negative" in item:
                        pairs.append((item["positive"], item["negative"]))
        except json.JSONDecodeError:
            logger.warning("Failed to parse contrast pairs JSON")

        return pairs

    def _get_target_layers(self, sample: ExtractionSample) -> List[int]:
        """Determine target layers based on sample configuration."""
        from vector_forge.tasks.config import LayerStrategy

        # Estimate model layers (would be better to get from model)
        total_layers = 32

        if sample.config.target_layers:
            return sample.config.target_layers

        strategy = sample.config.layer_strategy

        if strategy == LayerStrategy.AUTO:
            # Middle layers
            mid = total_layers // 2
            return list(range(mid - 2, mid + 3))
        elif strategy == LayerStrategy.SWEEP:
            # Wide sweep
            return list(range(8, total_layers - 4, 2))
        elif strategy == LayerStrategy.MIDDLE:
            mid = total_layers // 2
            return [mid - 1, mid, mid + 1]
        elif strategy == LayerStrategy.LATE:
            return list(range(total_layers - 8, total_layers - 2))
        else:
            return [total_layers // 2]

    async def _run_evaluations(
        self,
        results: List[SampleResult],
        behavior: ExpandedBehavior,
        config: TaskConfig,
    ) -> List[SampleResult]:
        """Run evaluations on all valid extraction results.

        Args:
            results: Extraction results to evaluate.
            behavior: Behavior specification.
            config: Task configuration.

        Returns:
            Results with evaluation scores filled in.
        """
        valid_results = [r for r in results if r.is_valid]

        if not valid_results:
            logger.warning("No valid extractions to evaluate")
            return results

        evaluator = VectorEvaluator(
            self._backend,
            self._llm,
            config.evaluation,
        )

        completed = 0
        started_at = time.time()

        async def evaluate_one(result: SampleResult) -> None:
            nonlocal completed
            start_time = time.time()

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

            completed += 1
            self._report_progress(
                total=len(results),
                extractions=len(results),
                evaluations=completed,
                failed=len(results) - len(valid_results),
                phase="evaluating",
                started=started_at,
            )

        # Run all evaluations
        await asyncio.gather(*[evaluate_one(r) for r in valid_results])

        logger.info(f"Evaluation complete for {len(valid_results)} vectors")
        return results

    def _aggregate_results(
        self,
        results: List[SampleResult],
        behavior: ExpandedBehavior,
        config: TaskConfig,
    ) -> TaskResult:
        """Aggregate sample results into final task result.

        Args:
            results: All sample results.
            behavior: Behavior specification.
            config: Task configuration.

        Returns:
            Aggregated task result.
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

        else:
            # Default to best single
            final_vector = sorted_results[0].vector
            final_layer = sorted_results[0].layer
            final_score = sorted_results[0].overall_score
            ensemble_components = [sorted_results[0].sample.sample_id]

        # Find recommended strength from best result
        recommended_strength = sorted_results[0].recommended_strength

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
            },
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
