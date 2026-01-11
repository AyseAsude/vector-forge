"""Tournament-style elimination runner for steering vector extraction.

This module implements progressive elimination:
1. Start with many samples (e.g., 256)
2. Run quick evaluation, eliminate bottom 75%
3. Repeat with deeper evaluation
4. Finals: comprehensive evaluation on survivors

Benefits over flat extraction:
- 25x more exploration for similar compute
- Focuses resources on promising candidates
- Progressive evaluation depth saves inference budget
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, TYPE_CHECKING

import torch

from vector_forge.tasks.config import (
    TaskConfig,
    TournamentConfig,
    EvalDepth,
    EvaluationConfig,
    AggregationStrategy,
)
from vector_forge.tasks.task import ExtractionTask, TaskResult, SampleResult
from vector_forge.tasks.sample import ExtractionSample
from vector_forge.tasks.expander import ExpandedBehavior
from vector_forge.tasks.eval_prompts import PromptGenerator, PromptSet, PromptSelector
from vector_forge.contrast.protocols import SampleDataset

if TYPE_CHECKING:
    from vector_forge.tasks.runner import TaskRunner
    from vector_forge.storage.emitter import EventEmitter

logger = logging.getLogger(__name__)


@dataclass
class RoundResult:
    """Result from a single elimination round."""

    round_idx: int
    samples_started: int
    samples_survived: int
    eval_depth: EvalDepth
    datapoints_used: int
    results: List[SampleResult]
    eliminated_ids: List[str]
    duration_seconds: float = 0.0


@dataclass
class TournamentResult:
    """Complete result from tournament extraction."""

    rounds: List[RoundResult]
    final_result: TaskResult
    total_samples_evaluated: int
    total_duration_seconds: float
    inference_savings: float  # Estimated % saved vs flat evaluation

    @property
    def num_rounds(self) -> int:
        return len(self.rounds)


@dataclass
class EliminationProgress:
    """Progress information for elimination runner."""

    round_idx: int
    total_rounds: int
    phase: str  # "extracting", "evaluating", "eliminating"
    samples_remaining: int
    samples_total: int
    progress: float  # 0.0 to 1.0 within current round
    message: str


class EliminationRunner:
    """Orchestrates tournament-style extraction with progressive elimination.

    Instead of running N samples with equal resources, the tournament starts
    with many more samples and progressively eliminates poor performers,
    focusing compute on promising candidates.

    Example:
        >>> config = TaskConfig(
        ...     tournament=TournamentConfig(
        ...         enabled=True,
        ...         elimination_rounds=2,
        ...         final_survivors=16,
        ...     )
        ... )
        >>> runner = EliminationRunner(task_runner, config)
        >>> result = await runner.run(task, sample_datasets)
    """

    def __init__(
        self,
        task_runner: "TaskRunner",
        config: TaskConfig,
        event_emitter: Optional["EventEmitter"] = None,
    ) -> None:
        """Initialize the elimination runner.

        Args:
            task_runner: The underlying TaskRunner for extraction/evaluation.
            config: Task configuration with tournament settings.
            event_emitter: Optional event emitter for tracking.
        """
        self._runner = task_runner
        self._config = config
        self._tournament = config.tournament
        self._emitter = event_emitter
        self._progress_callbacks: List[Callable[[EliminationProgress], None]] = []

        # Prompt management for consistent evaluation across samples
        self._prompt_generator = PromptGenerator(seed=42)
        self._prompt_set: Optional[PromptSet] = None
        self._prompt_selector: Optional[PromptSelector] = None

    def on_progress(self, callback: Callable[[EliminationProgress], None]) -> None:
        """Register a progress callback."""
        self._progress_callbacks.append(callback)

    def _emit_progress(self, progress: EliminationProgress) -> None:
        """Emit progress to all callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    async def run(
        self,
        task: ExtractionTask,
        sample_datasets: Dict[int, SampleDataset],
    ) -> TournamentResult:
        """Run tournament-style extraction.

        Args:
            task: The extraction task.
            sample_datasets: Pre-generated datasets for each sample.

        Returns:
            TournamentResult with all round results and final aggregation.
        """
        start_time = time.time()
        tournament = self._tournament

        logger.info(
            f"Starting tournament: {tournament.initial_samples} initial → "
            f"{tournament.final_survivors} survivors over {tournament.total_rounds} rounds"
        )

        # Generate consistent prompt set for all evaluations
        self._prompt_set = self._prompt_generator.generate(
            behavior=task.behavior,
            behavior_count=self._config.evaluation.behavior_prompts,
            specificity_count=self._config.evaluation.specificity_prompts,
            coherence_count=self._config.evaluation.coherence_prompts,
            capability_count=self._config.evaluation.capability_prompts,
            generalization_count=self._config.evaluation.generalization_prompts,
        )
        self._prompt_selector = PromptSelector(self._prompt_set)

        # Log round summary
        for round_info in tournament.get_round_summary():
            logger.info(
                f"  Round {round_info['round']}: {round_info['samples']} samples, "
                f"{round_info['datapoints']} datapoints, {round_info['eval_depth']} eval"
                f"{' (finals)' if round_info['is_finals'] else ''}"
            )

        # Track survivors across rounds
        # Start with all samples
        active_sample_ids = list(range(tournament.initial_samples))
        round_results: List[RoundResult] = []

        for round_idx in range(tournament.total_rounds):
            is_finals = round_idx >= tournament.elimination_rounds
            eval_depth = tournament.eval_depth_at_round(round_idx)
            datapoints = tournament.datapoints_at_round(round_idx)
            target_survivors = tournament.survivors_after_round(round_idx)

            logger.info(
                f"\n=== Round {round_idx + 1}/{tournament.total_rounds} ===\n"
                f"  Samples: {len(active_sample_ids)}\n"
                f"  Datapoints: {datapoints}\n"
                f"  Eval depth: {eval_depth.value}\n"
                f"  Target survivors: {target_survivors}"
            )

            self._emit_progress(EliminationProgress(
                round_idx=round_idx,
                total_rounds=tournament.total_rounds,
                phase="extracting",
                samples_remaining=len(active_sample_ids),
                samples_total=tournament.initial_samples,
                progress=0.0,
                message=f"Round {round_idx + 1}: Extracting {len(active_sample_ids)} samples...",
            ))

            # Run extraction for active samples
            round_start = time.time()
            round_result = await self._run_round(
                task=task,
                sample_datasets=sample_datasets,
                active_sample_ids=active_sample_ids,
                round_idx=round_idx,
                eval_depth=eval_depth,
                datapoints_per_sample=datapoints,
            )

            round_results.append(round_result)

            if is_finals:
                # Finals - no elimination, proceed to aggregation
                logger.info(f"Finals complete: {len(round_result.results)} samples")
                break

            # Elimination: keep top performers
            self._emit_progress(EliminationProgress(
                round_idx=round_idx,
                total_rounds=tournament.total_rounds,
                phase="eliminating",
                samples_remaining=len(active_sample_ids),
                samples_total=tournament.initial_samples,
                progress=0.9,
                message=f"Eliminating bottom {int(tournament.elimination_rate * 100)}%...",
            ))

            # Sort by score and keep top
            sorted_results = sorted(
                round_result.results,
                key=lambda r: r.overall_score,
                reverse=True,
            )

            survivors = sorted_results[:target_survivors]
            eliminated = sorted_results[target_survivors:]

            # Update active samples for next round
            active_sample_ids = [
                r.sample.sample_index
                for r in survivors
            ]

            round_result.eliminated_ids = [r.sample.sample_id for r in eliminated]
            round_result.samples_survived = len(survivors)

            logger.info(
                f"Round {round_idx + 1} complete: "
                f"{len(survivors)} survived, {len(eliminated)} eliminated\n"
                f"  Score range: {survivors[-1].overall_score:.2f} - {survivors[0].overall_score:.2f}"
            )

        # Aggregation on final survivors
        final_round = round_results[-1]
        final_result = self._aggregate_survivors(
            final_round.results,
            task.behavior,
            sample_datasets,
        )

        total_duration = time.time() - start_time

        # Calculate inference savings
        flat_inferences = (
            tournament.initial_samples *
            self._config.evaluation.estimated_inferences
        )
        actual_inferences = sum(
            len(r.results) * self._config.evaluation.scale_to_depth(
                tournament.eval_depth_at_round(r.round_idx)
            ).estimated_inferences
            for r in round_results
        )
        savings = 1.0 - (actual_inferences / flat_inferences) if flat_inferences > 0 else 0.0

        logger.info(
            f"\nTournament complete:\n"
            f"  Duration: {total_duration:.1f}s\n"
            f"  Final score: {final_result.final_score:.2f}\n"
            f"  Inference savings: {savings * 100:.1f}%"
        )

        return TournamentResult(
            rounds=round_results,
            final_result=final_result,
            total_samples_evaluated=sum(len(r.results) for r in round_results),
            total_duration_seconds=total_duration,
            inference_savings=savings,
        )

    async def _run_round(
        self,
        task: ExtractionTask,
        sample_datasets: Dict[int, SampleDataset],
        active_sample_ids: List[int],
        round_idx: int,
        eval_depth: EvalDepth,
        datapoints_per_sample: int,
    ) -> RoundResult:
        """Run a single round of extraction and evaluation.

        Args:
            task: The extraction task.
            sample_datasets: All sample datasets.
            active_sample_ids: Which samples are still active.
            round_idx: Current round index.
            eval_depth: Evaluation depth for this round.
            datapoints_per_sample: Datapoints to use per sample.

        Returns:
            RoundResult with all sample results.
        """
        round_start = time.time()

        # Get active samples
        active_samples = [
            task.samples[i] for i in active_sample_ids
            if i < len(task.samples)
        ]

        # Get datasets for active samples
        active_datasets = {
            i: sample_datasets[i]
            for i in active_sample_ids
            if i in sample_datasets
        }

        # Create scaled evaluation config for this depth
        scaled_eval = self._config.evaluation.scale_to_depth(eval_depth)

        logger.info(
            f"Round {round_idx + 1} eval config: "
            f"{scaled_eval.behavior_prompts} behavior prompts, "
            f"{len(scaled_eval.strength_levels)} strengths, "
            f"~{scaled_eval.estimated_inferences} inferences/sample"
        )

        # Run extraction with limited datapoints
        # TODO: Modify TaskRunner to accept datapoints_limit parameter
        # For now, we use the full config

        # Create a modified task for this round
        round_task = ExtractionTask(
            behavior=task.behavior,
            config=task.config.model_copy(update={
                "datapoints_per_sample": datapoints_per_sample,
                "evaluation": scaled_eval,
            }),
            samples=active_samples,
        )

        # Run extraction phase
        extraction_results = await self._runner._run_extractions(
            round_task,
            active_datasets,
        )

        self._emit_progress(EliminationProgress(
            round_idx=round_idx,
            total_rounds=self._tournament.total_rounds,
            phase="evaluating",
            samples_remaining=len(active_sample_ids),
            samples_total=self._tournament.initial_samples,
            progress=0.5,
            message=f"Evaluating {len(extraction_results)} samples ({eval_depth.value})...",
        ))

        # Run evaluation phase with scaled config
        # Use the prompt selector for consistent prompts
        evaluated_results = await self._runner._run_evaluations(
            extraction_results,
            task.behavior,
            round_task.config,
        )

        duration = time.time() - round_start

        return RoundResult(
            round_idx=round_idx,
            samples_started=len(active_sample_ids),
            samples_survived=len(active_sample_ids),  # Updated after elimination
            eval_depth=eval_depth,
            datapoints_used=datapoints_per_sample,
            results=evaluated_results,
            eliminated_ids=[],  # Filled in after elimination
            duration_seconds=duration,
        )

    def _aggregate_survivors(
        self,
        results: List[SampleResult],
        behavior: ExpandedBehavior,
        sample_datasets: Dict[int, SampleDataset],
    ) -> TaskResult:
        """Aggregate final survivors into TaskResult.

        Uses the same aggregation logic as TaskRunner.
        """
        valid_results = [r for r in results if r.is_valid]

        if not valid_results:
            raise ValueError("No valid results to aggregate")

        sorted_results = sorted(
            valid_results,
            key=lambda r: r.overall_score,
            reverse=True,
        )

        strategy = self._config.aggregation_strategy
        top_k = min(self._config.top_k, len(sorted_results))

        if strategy == AggregationStrategy.BEST_SINGLE:
            final_vector = sorted_results[0].vector
            final_layer = sorted_results[0].layer
            final_score = sorted_results[0].overall_score
            ensemble_components = [sorted_results[0].sample.sample_id]

        elif strategy == AggregationStrategy.TOP_K_AVERAGE:
            top_results = sorted_results[:top_k]
            stacked = torch.stack([r.vector for r in top_results])
            final_vector = stacked.mean(dim=0)
            final_vector = final_vector / final_vector.norm()
            final_layer = top_results[0].layer
            final_score = sum(r.overall_score for r in top_results) / top_k
            ensemble_components = [r.sample.sample_id for r in top_results]

        elif strategy == AggregationStrategy.WEIGHTED_AVERAGE:
            top_results = sorted_results[:top_k]
            weights = torch.tensor([r.overall_score for r in top_results])
            weights = weights / weights.sum()
            stacked = torch.stack([r.vector for r in top_results])
            final_vector = (stacked * weights.unsqueeze(1)).sum(dim=0)
            final_vector = final_vector / final_vector.norm()
            final_layer = top_results[0].layer
            final_score = sum(r.overall_score for r in top_results) / top_k
            ensemble_components = [r.sample.sample_id for r in top_results]

        else:
            # Default to best single
            final_vector = sorted_results[0].vector
            final_layer = sorted_results[0].layer
            final_score = sorted_results[0].overall_score
            ensemble_components = [sorted_results[0].sample.sample_id]

        return TaskResult(
            behavior=behavior,
            final_vector=final_vector,
            final_layer=final_layer,
            final_score=final_score,
            recommended_strength=sorted_results[0].recommended_strength,
            aggregation_method=strategy,
            sample_results=results,
            ensemble_components=ensemble_components,
            metadata={
                "tournament_mode": True,
                "initial_samples": self._tournament.initial_samples,
                "elimination_rounds": self._tournament.elimination_rounds,
                "final_survivors": self._tournament.final_survivors,
                "valid_count": len(valid_results),
                "total_count": len(results),
                "top_k": top_k,
            },
            task_config=self._config,
            contrast_data={},  # TODO: Serialize if needed
        )
