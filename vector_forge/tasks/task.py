"""Extraction task definitions and result containers.

Provides the main task abstraction that combines behavior specification,
sample generation, and result aggregation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from pathlib import Path
import json
import time

import torch

from vector_forge.tasks.config import TaskConfig, AggregationStrategy
from vector_forge.tasks.sample import ExtractionSample, SampleSet, SampleGenerator
from vector_forge.tasks.expander import ExpandedBehavior

if TYPE_CHECKING:
    from vector_forge.tasks.evaluation import EvaluationResult


@dataclass
class SampleResult:
    """Result from a single extraction sample.

    Contains the extracted vector, scores, and metadata for one
    extraction attempt.
    """

    sample: ExtractionSample
    vector: Optional[torch.Tensor]
    layer: int
    scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    recommended_strength: float = 1.0
    evaluation: Optional["EvaluationResult"] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    extraction_time_seconds: float = 0.0
    evaluation_time_seconds: float = 0.0

    @property
    def is_valid(self) -> bool:
        """Check if extraction produced a valid vector."""
        return self.vector is not None and self.vector.numel() > 0

    @property
    def total_time_seconds(self) -> float:
        """Total time for extraction and evaluation."""
        return self.extraction_time_seconds + self.evaluation_time_seconds


@dataclass
class TaskResult:
    """Aggregated results from a complete extraction task.

    Contains the final vector (from aggregation), all sample results,
    and comprehensive metadata about the extraction process.
    """

    behavior: ExpandedBehavior
    final_vector: torch.Tensor
    final_layer: int
    final_score: float
    recommended_strength: float
    aggregation_method: AggregationStrategy
    sample_results: List[SampleResult] = field(default_factory=list)
    ensemble_components: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: float = 0.0
    completed_at: float = 0.0

    @property
    def duration_seconds(self) -> float:
        """Total task duration."""
        if self.completed_at > self.started_at:
            return self.completed_at - self.started_at
        return 0.0

    @property
    def valid_results_count(self) -> int:
        """Number of samples that produced valid vectors."""
        return sum(1 for r in self.sample_results if r.is_valid)

    @property
    def best_single(self) -> Optional[SampleResult]:
        """Get the highest-scoring single sample."""
        valid = [r for r in self.sample_results if r.is_valid]
        if not valid:
            return None
        return max(valid, key=lambda r: r.overall_score)

    @property
    def top_results(self) -> List[SampleResult]:
        """Get sample results sorted by score (descending)."""
        valid = [r for r in self.sample_results if r.is_valid]
        return sorted(valid, key=lambda r: r.overall_score, reverse=True)

    def get_results_by_strategy(self, strategy_name: str) -> List[SampleResult]:
        """Filter results by strategy name pattern."""
        return [
            r for r in self.sample_results
            if strategy_name in r.sample.strategy_name
        ]

    def save(self, path: str) -> None:
        """Save results to disk.

        Saves:
        - final_vector.pt: The aggregated steering vector
        - task_result.json: Metadata and scores
        - samples/: Individual sample vectors (if save_all_vectors enabled)

        Args:
            path: Directory path for saving results.
        """
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save final vector
        torch.save(self.final_vector, output_dir / "final_vector.pt")

        # Save metadata
        metadata = {
            "behavior": {
                "name": self.behavior.name,
                "description": self.behavior.description,
                "detailed_definition": self.behavior.detailed_definition,
            },
            "result": {
                "final_layer": self.final_layer,
                "final_score": self.final_score,
                "recommended_strength": self.recommended_strength,
                "aggregation_method": self.aggregation_method.value,
                "ensemble_components": self.ensemble_components,
                "duration_seconds": self.duration_seconds,
                "valid_samples": self.valid_results_count,
                "total_samples": len(self.sample_results),
            },
            "sample_scores": [
                {
                    "sample_id": r.sample.sample_id,
                    "strategy_name": r.sample.strategy_name,
                    "overall_score": r.overall_score,
                    "scores": r.scores,
                    "layer": r.layer,
                    "is_valid": r.is_valid,
                }
                for r in self.sample_results
            ],
            "metadata": self.metadata,
        }

        with open(output_dir / "task_result.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TaskResult":
        """Load results from disk.

        Args:
            path: Directory path containing saved results.

        Returns:
            Reconstructed TaskResult (without full sample data).
        """
        input_dir = Path(path)

        vector = torch.load(input_dir / "final_vector.pt")

        with open(input_dir / "task_result.json") as f:
            metadata = json.load(f)

        behavior = ExpandedBehavior(
            name=metadata["behavior"]["name"],
            description=metadata["behavior"]["description"],
            detailed_definition=metadata["behavior"]["detailed_definition"],
        )

        return cls(
            behavior=behavior,
            final_vector=vector,
            final_layer=metadata["result"]["final_layer"],
            final_score=metadata["result"]["final_score"],
            recommended_strength=metadata["result"]["recommended_strength"],
            aggregation_method=AggregationStrategy(
                metadata["result"]["aggregation_method"]
            ),
            ensemble_components=metadata["result"]["ensemble_components"],
            metadata=metadata.get("metadata", {}),
        )


class ExtractionTask:
    """Defines an extraction task with samples and evaluation criteria.

    The main orchestration unit that combines:
    - Behavior specification
    - Sample generation strategy
    - Evaluation configuration
    - Result aggregation

    Similar to inspect_ai.Task, this provides a declarative way to define
    what to extract and how to evaluate it.

    Example:
        >>> task = ExtractionTask.from_description(
        ...     "sycophancy - agreeing with users even when wrong",
        ...     config=TaskConfig.standard(),
        ... )
        >>> # Run with TaskRunner
        >>> runner = TaskRunner(backend, llm)
        >>> result = await runner.run(task)
    """

    def __init__(
        self,
        behavior: ExpandedBehavior,
        samples: SampleSet,
        config: TaskConfig,
    ) -> None:
        """Initialize extraction task.

        Args:
            behavior: Expanded behavior specification.
            samples: Set of extraction samples to run.
            config: Task configuration.
        """
        self.behavior = behavior
        self.samples = samples
        self.config = config
        self._created_at = time.time()

    @classmethod
    def from_behavior(
        cls,
        behavior: ExpandedBehavior,
        config: Optional[TaskConfig] = None,
        generation_method: str = "smart",
    ) -> "ExtractionTask":
        """Create task from an expanded behavior.

        Args:
            behavior: Expanded behavior specification.
            config: Optional task configuration.
            generation_method: "grid", "smart", or "seeded".

        Returns:
            Configured ExtractionTask.
        """
        config = config or TaskConfig.standard()
        generator = SampleGenerator(config)

        if generation_method == "grid":
            samples = generator.generate_grid(behavior)
        elif generation_method == "smart":
            samples = generator.generate_smart(behavior)
        elif generation_method == "seeded":
            samples = generator.generate_seeded(
                behavior,
                config.layer_strategies[0],
                config.num_seeds,
            )
        else:
            raise ValueError(f"Unknown generation method: {generation_method}")

        return cls(behavior, samples, config)

    @property
    def num_samples(self) -> int:
        """Number of extraction samples."""
        return len(self.samples)

    @property
    def estimated_inference_count(self) -> int:
        """Estimate total inference calls needed."""
        eval_config = self.config.evaluation

        # Per sample: contrast pairs generation
        per_sample_extraction = self.config.contrast_pair_count * 2

        # Per sample: evaluation
        per_sample_eval = (
            eval_config.behavior_prompts * len(eval_config.strength_levels) * eval_config.behavior_generations_per_prompt
            + eval_config.specificity_prompts * 2
            + eval_config.coherence_prompts * len(eval_config.strength_levels)
            + eval_config.capability_prompts * 2
            + eval_config.generalization_prompts * 2
        )

        return self.num_samples * (per_sample_extraction + per_sample_eval)

    def summary(self) -> Dict[str, Any]:
        """Get task summary for display."""
        return {
            "behavior": self.behavior.name,
            "description": self.behavior.description[:100],
            "num_samples": self.num_samples,
            "strategies": [s.value for s in self.samples.strategies_used],
            "seeds": self.samples.seeds_used,
            "estimated_inferences": self.estimated_inference_count,
            "max_concurrent": self.config.max_concurrent_extractions,
            "aggregation": self.config.aggregation_strategy.value,
        }
