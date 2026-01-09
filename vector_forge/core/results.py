"""Result types for extraction and evaluation."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

import torch


class Verdict(str, Enum):
    """Judge verdict for a vector."""

    ACCEPTED = "accepted"
    NEEDS_REFINEMENT = "needs_refinement"
    REJECTED = "rejected"


@dataclass
class Citation:
    """A citation from evaluation - evidence for a score."""

    prompt: str
    output: str
    reason: str
    strength: Optional[float] = None
    is_success: bool = True


@dataclass
class StrengthAnalysis:
    """Analysis of vector effect at different steering strengths."""

    strength: float
    behavior_score: float
    coherence_score: float
    num_samples: int


@dataclass
class EvaluationScores:
    """Scores across evaluation dimensions."""

    behavior_strength: float = 0.0
    coherence: float = 0.0
    specificity: float = 0.0
    robustness: float = 0.0
    overall: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "behavior_strength": self.behavior_strength,
            "coherence": self.coherence,
            "specificity": self.specificity,
            "robustness": self.robustness,
            "overall": self.overall,
        }


@dataclass
class EvaluationResult:
    """
    Complete evaluation result from the judge.

    Contains scores, strength analysis, citations as evidence,
    and recommendations for improvement.
    """

    scores: EvaluationScores
    strength_analysis: List[StrengthAnalysis]
    recommended_strength: float

    citations: Dict[str, List[Citation]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    verdict: Verdict = Verdict.NEEDS_REFINEMENT
    raw_judge_output: Optional[str] = None

    @property
    def is_acceptable(self) -> bool:
        return self.verdict == Verdict.ACCEPTED


@dataclass
class DatapointQuality:
    """Quality metrics for a single training datapoint."""

    datapoint_id: str

    # Influence metrics
    leave_one_out_influence: Optional[float] = None  # Positive = removing improves quality

    # Gradient metrics (computed during training)
    avg_loss_contribution: float = 0.0
    gradient_alignment: float = 0.0  # Negative = conflicts with other datapoints
    loss_variance: float = 0.0

    # Evaluation metrics
    steered_matches_target: bool = True
    behavior_score_on_own_prompt: float = 0.0

    # Clustering metrics
    distance_to_centroid: float = 0.0
    cluster_id: int = 0
    is_outlier: bool = False

    @property
    def quality_score(self) -> float:
        """
        Overall quality score (0-1, higher = better).

        Combines multiple signals to estimate if datapoint helps or hurts.
        """
        score = 1.0

        if self.leave_one_out_influence is not None and self.leave_one_out_influence > 0:
            score -= min(0.5, self.leave_one_out_influence)

        if self.gradient_alignment < 0:
            score += self.gradient_alignment * 0.3  # Negative alignment hurts

        if not self.steered_matches_target:
            score -= 0.3

        if self.is_outlier:
            score -= 0.2

        return max(0.0, min(1.0, score))

    @property
    def recommendation(self) -> str:
        if self.quality_score > 0.7:
            return "KEEP"
        elif self.quality_score > 0.4:
            return "REVIEW"
        else:
            return "REMOVE"


@dataclass
class DiversityMetrics:
    """Metrics for assessing diversity of a prompt/datapoint set."""

    num_items: int
    avg_pairwise_similarity: float
    min_pairwise_distance: float
    length_std: float
    num_clusters: int
    cluster_balance: float  # 0 = all in one cluster, 1 = perfectly balanced

    @property
    def is_diverse_enough(self) -> bool:
        return self.avg_pairwise_similarity < 0.7 and self.min_pairwise_distance > 0.3


@dataclass
class OptimizationMetrics:
    """Metrics from a single optimization run."""

    layer: int
    final_loss: float
    iterations: int
    vector_norm: float

    # Per-datapoint metrics
    datapoint_qualities: List[DatapointQuality] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """
    Final result from the extraction pipeline.

    Contains the extracted vector, recommended settings,
    evaluation results, and metadata about the extraction process.
    """

    # The extracted vector
    vector: torch.Tensor
    recommended_layer: int
    recommended_strength: float

    # Evaluation
    evaluation: EvaluationResult

    # Training data used
    num_datapoints: int
    datapoint_quality_summary: Dict[str, int] = field(default_factory=dict)  # {KEEP: n, REMOVE: m}

    # Optimization metrics
    optimization_metrics: Optional[OptimizationMetrics] = None

    # Noise reduction
    noise_reduction_applied: bool = False
    num_seeds_averaged: int = 1

    # Metadata
    behavior_name: str = ""
    total_iterations: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> None:
        """
        Save the extraction result to disk.

        Args:
            path: File path to save to (without extension).
        """
        import json

        # Save vector
        torch.save(self.vector, f"{path}.pt")

        # Save metadata
        meta = {
            "recommended_layer": self.recommended_layer,
            "recommended_strength": self.recommended_strength,
            "evaluation_scores": self.evaluation.scores.to_dict(),
            "verdict": self.evaluation.verdict.value,
            "num_datapoints": self.num_datapoints,
            "behavior_name": self.behavior_name,
            "total_iterations": self.total_iterations,
            "metadata": self.metadata,
        }
        with open(f"{path}.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExtractionResult":
        """
        Load an extraction result from disk.

        Args:
            path: File path to load from (without extension).

        Returns:
            ExtractionResult instance.
        """
        import json

        vector = torch.load(f"{path}.pt")
        with open(f"{path}.json") as f:
            meta = json.load(f)

        scores = EvaluationScores(**meta["evaluation_scores"])
        evaluation = EvaluationResult(
            scores=scores,
            strength_analysis=[],
            recommended_strength=meta["recommended_strength"],
            verdict=Verdict(meta["verdict"]),
        )

        return cls(
            vector=vector,
            recommended_layer=meta["recommended_layer"],
            recommended_strength=meta["recommended_strength"],
            evaluation=evaluation,
            num_datapoints=meta["num_datapoints"],
            behavior_name=meta["behavior_name"],
            total_iterations=meta["total_iterations"],
            metadata=meta.get("metadata", {}),
        )
