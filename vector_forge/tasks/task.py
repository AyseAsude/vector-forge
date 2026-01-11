"""Extraction task definitions and result containers.

Provides the main task abstraction that combines behavior specification,
sample generation, and result aggregation with full reproducibility support.
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


# ============================================================================
# Sample Result
# ============================================================================


@dataclass
class SampleResult:
    """Result from a single extraction sample.

    Contains the extracted vector, optimization metadata, scores, and
    all information needed for reproducibility.
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

    # Optimization metadata accessors
    @property
    def final_loss(self) -> float:
        """Final optimization loss."""
        return self.metadata.get("final_loss", 0.0)

    @property
    def iterations(self) -> int:
        """Number of optimization iterations."""
        return self.metadata.get("iterations", 0)

    @property
    def loss_history(self) -> List[float]:
        """Loss history from optimization."""
        return self.metadata.get("loss_history", [])

    @property
    def datapoints_used(self) -> int:
        """Number of datapoints used for training."""
        return self.metadata.get("datapoints_used", 0)

    @property
    def optimization_config(self) -> Dict[str, Any]:
        """Optimization configuration used."""
        return self.metadata.get("config", {})

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (without vector)."""
        return {
            "sample_id": self.sample.sample_id,
            "strategy_name": self.sample.strategy_name,
            "layer": self.layer,
            "overall_score": self.overall_score,
            "scores": self.scores,
            "recommended_strength": self.recommended_strength,
            "is_valid": self.is_valid,
            "extraction_time_seconds": self.extraction_time_seconds,
            "evaluation_time_seconds": self.evaluation_time_seconds,
            "optimization": {
                "final_loss": self.final_loss,
                "iterations": self.iterations,
                "datapoints_used": self.datapoints_used,
                "config": self.optimization_config,
            },
            "sample_config": {
                "seed": self.sample.config.seed,
                "layer_strategy": self.sample.config.layer_strategy.value,
                "bootstrap_ratio": self.sample.config.bootstrap_ratio,
            },
        }


# ============================================================================
# Task Result
# ============================================================================


@dataclass
class TaskResult:
    """Aggregated results from a complete extraction task.

    Contains the final vector (from aggregation), all sample results,
    and comprehensive metadata for full reproducibility.

    Supports:
    - Saving all sample vectors for re-aggregation
    - Complete optimization history
    - Full configuration snapshot
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

    # Additional fields for full reproducibility
    task_config: Optional[TaskConfig] = None
    contrast_data: Optional[Dict[str, Any]] = None  # Serialized contrast pairs

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

    def save(self, path: str, save_all_vectors: bool = True) -> None:
        """Save results to disk with full reproducibility.

        Directory structure:
            {path}/
            ├── final_vector.pt          # Aggregated steering vector
            ├── task_result.json          # Metadata and scores
            ├── task_config.json          # Full task configuration
            ├── behavior.json             # Behavior specification
            ├── contrast_data.json        # Contrast pairs (if available)
            └── samples/
                ├── sample_00/
                │   ├── vector.pt         # Sample vector
                │   ├── result.json       # Sample metadata
                │   └── loss_history.json # Optimization history
                ├── sample_01/
                │   └── ...
                └── ...

        Args:
            path: Directory path for saving results.
            save_all_vectors: If True, save all sample vectors.
        """
        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save final vector
        torch.save(self.final_vector, output_dir / "final_vector.pt")

        # Save task configuration
        if self.task_config:
            with open(output_dir / "task_config.json", "w") as f:
                json.dump(self.task_config.model_dump(), f, indent=2, default=str)

        # Save behavior
        behavior_data = {
            "name": self.behavior.name,
            "description": self.behavior.description,
            "detailed_definition": self.behavior.detailed_definition,
            "contrast_guidance": getattr(self.behavior, "contrast_guidance", ""),
            "domains": getattr(self.behavior, "domains", []),
        }
        with open(output_dir / "behavior.json", "w") as f:
            json.dump(behavior_data, f, indent=2)

        # Save contrast data if available
        if self.contrast_data:
            with open(output_dir / "contrast_data.json", "w") as f:
                json.dump(self.contrast_data, f, indent=2)

        # Save main result metadata
        result_data = {
            "final_layer": self.final_layer,
            "final_score": self.final_score,
            "recommended_strength": self.recommended_strength,
            "aggregation_method": self.aggregation_method.value,
            "ensemble_components": self.ensemble_components,
            "duration_seconds": self.duration_seconds,
            "valid_samples": self.valid_results_count,
            "total_samples": len(self.sample_results),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }
        with open(output_dir / "task_result.json", "w") as f:
            json.dump(result_data, f, indent=2)

        # Save sample results
        if save_all_vectors:
            samples_dir = output_dir / "samples"
            samples_dir.mkdir(exist_ok=True)

            for idx, result in enumerate(self.sample_results):
                sample_dir = samples_dir / f"sample_{idx:02d}"
                sample_dir.mkdir(exist_ok=True)

                # Save vector if valid
                if result.is_valid:
                    torch.save(result.vector, sample_dir / "vector.pt")

                # Save result metadata
                with open(sample_dir / "result.json", "w") as f:
                    json.dump(result.to_dict(), f, indent=2)

                # Save loss history separately (can be large)
                if result.loss_history:
                    with open(sample_dir / "loss_history.json", "w") as f:
                        json.dump(result.loss_history, f)

        # Save summary of all samples (without vectors)
        samples_summary = [r.to_dict() for r in self.sample_results]
        with open(output_dir / "samples_summary.json", "w") as f:
            json.dump(samples_summary, f, indent=2)

    @classmethod
    def load(cls, path: str, load_vectors: bool = True) -> "TaskResult":
        """Load results from disk.

        Args:
            path: Directory path containing saved results.
            load_vectors: If True, load all sample vectors.

        Returns:
            Reconstructed TaskResult.
        """
        input_dir = Path(path)

        # Load final vector
        final_vector = torch.load(input_dir / "final_vector.pt", weights_only=True)

        # Load main result
        with open(input_dir / "task_result.json") as f:
            result_data = json.load(f)

        # Load behavior
        with open(input_dir / "behavior.json") as f:
            behavior_data = json.load(f)

        behavior = ExpandedBehavior(
            name=behavior_data["name"],
            description=behavior_data["description"],
            detailed_definition=behavior_data.get("detailed_definition", ""),
        )

        # Load task config if available
        task_config = None
        config_path = input_dir / "task_config.json"
        if config_path.exists():
            with open(config_path) as f:
                task_config = TaskConfig.model_validate(json.load(f))

        # Load contrast data if available
        contrast_data = None
        contrast_path = input_dir / "contrast_data.json"
        if contrast_path.exists():
            with open(contrast_path) as f:
                contrast_data = json.load(f)

        # Load sample vectors if requested
        sample_results = []
        samples_dir = input_dir / "samples"
        if samples_dir.exists() and load_vectors:
            for sample_dir in sorted(samples_dir.iterdir()):
                if sample_dir.is_dir():
                    result_path = sample_dir / "result.json"
                    if result_path.exists():
                        with open(result_path) as f:
                            sample_data = json.load(f)

                        # Load vector if exists
                        vector = None
                        vector_path = sample_dir / "vector.pt"
                        if vector_path.exists():
                            vector = torch.load(vector_path, weights_only=True)

                        # Load loss history
                        loss_history = []
                        history_path = sample_dir / "loss_history.json"
                        if history_path.exists():
                            with open(history_path) as f:
                                loss_history = json.load(f)

                        # Reconstruct minimal SampleResult
                        from vector_forge.tasks.config import SampleConfig, LayerStrategy

                        sample_config = SampleConfig(
                            seed=sample_data.get("sample_config", {}).get("seed", 0),
                            layer_strategy=LayerStrategy(
                                sample_data.get("sample_config", {}).get(
                                    "layer_strategy", "auto"
                                )
                            ),
                        )
                        sample = ExtractionSample(
                            behavior=behavior,
                            config=sample_config,
                            sample_id=sample_data.get("sample_id", ""),
                            strategy_name=sample_data.get("strategy_name", ""),
                        )

                        result = SampleResult(
                            sample=sample,
                            vector=vector,
                            layer=sample_data.get("layer", 0),
                            overall_score=sample_data.get("overall_score", 0.0),
                            scores=sample_data.get("scores", {}),
                            recommended_strength=sample_data.get(
                                "recommended_strength", 1.0
                            ),
                            metadata={
                                "final_loss": sample_data.get("optimization", {}).get(
                                    "final_loss", 0.0
                                ),
                                "iterations": sample_data.get("optimization", {}).get(
                                    "iterations", 0
                                ),
                                "datapoints_used": sample_data.get(
                                    "optimization", {}
                                ).get("datapoints_used", 0),
                                "config": sample_data.get("optimization", {}).get(
                                    "config", {}
                                ),
                                "loss_history": loss_history,
                            },
                            extraction_time_seconds=sample_data.get(
                                "extraction_time_seconds", 0.0
                            ),
                            evaluation_time_seconds=sample_data.get(
                                "evaluation_time_seconds", 0.0
                            ),
                        )
                        sample_results.append(result)

        return cls(
            behavior=behavior,
            final_vector=final_vector,
            final_layer=result_data["final_layer"],
            final_score=result_data["final_score"],
            recommended_strength=result_data["recommended_strength"],
            aggregation_method=AggregationStrategy(result_data["aggregation_method"]),
            ensemble_components=result_data.get("ensemble_components", []),
            metadata=result_data.get("metadata", {}),
            sample_results=sample_results,
            started_at=result_data.get("started_at", 0.0),
            completed_at=result_data.get("completed_at", 0.0),
            task_config=task_config,
            contrast_data=contrast_data,
        )


# ============================================================================
# Extraction Task
# ============================================================================


class ExtractionTask:
    """Defines an extraction task with samples and evaluation criteria.

    The main orchestration unit that combines:
    - Behavior specification
    - Sample generation strategy
    - Evaluation configuration
    - Result aggregation

    Example:
        >>> task = ExtractionTask.from_behavior(
        ...     behavior,
        ...     config=TaskConfig.standard(),
        ... )
        >>> runner = TaskRunner(backend, llm)
        >>> result = await runner.run(task, sample_datasets)
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
            generation_method: Sample generation method:
                - "smart": Quasi-random sampling for good coverage (default)
                - "grid": Full grid search over parameters
                - "seeded": Single strategy with multiple seeds

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

        # Per sample: optimization iterations × datapoints
        opt_config = self.config.optimization
        per_sample_extraction = (
            opt_config.max_iters * self.config.datapoints_per_sample * 2
        )

        # Per sample: evaluation
        per_sample_eval = (
            eval_config.behavior_prompts
            * len(eval_config.strength_levels)
            * eval_config.behavior_generations_per_prompt
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
            "datapoints_per_sample": self.config.datapoints_per_sample,
            "optimization_iters": self.config.optimization.max_iters,
            "estimated_inferences": self.estimated_inference_count,
            "max_concurrent": self.config.max_concurrent_extractions,
            "aggregation": self.config.aggregation_strategy.value,
        }
