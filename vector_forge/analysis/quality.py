"""Datapoint quality analysis."""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import numpy as np

import torch
from steering_vectors import SteeringVectorTrainingSample as TrainingDatapoint

from vector_forge.core.results import DatapointQuality


@dataclass
class TrainingMetrics:
    """Metrics collected during training for a datapoint."""

    datapoint_idx: int
    loss_history: List[float]
    gradient_norms: List[float]
    gradient_alignments: List[float]  # Cosine sim with total gradient


class DatapointQualityAnalyzer:
    """
    Analyze quality of training datapoints.

    Identifies problematic datapoints through:
    - Gradient-based analysis (conflicts, high variance)
    - Leave-one-out influence estimation
    - Embedding-based outlier detection

    Example:
        >>> analyzer = DatapointQualityAnalyzer()
        >>> # After optimization, analyze quality
        >>> qualities = analyzer.analyze_from_training(datapoints, training_metrics)
        >>> bad_datapoints = [q for q in qualities if q.recommendation == "REMOVE"]
    """

    def __init__(self, outlier_threshold: float = 2.0):
        """
        Args:
            outlier_threshold: Number of standard deviations for outlier detection.
        """
        self.outlier_threshold = outlier_threshold

    def analyze_from_training(
        self,
        datapoints: List[TrainingDatapoint],
        training_metrics: List[TrainingMetrics],
    ) -> List[DatapointQuality]:
        """
        Analyze datapoint quality using training metrics.

        Args:
            datapoints: Training datapoints.
            training_metrics: Metrics collected during optimization.

        Returns:
            List of DatapointQuality for each datapoint.
        """
        qualities = []

        for i, (dp, metrics) in enumerate(zip(datapoints, training_metrics)):
            dp_id = f"dp_{i}"

            # Compute metrics
            avg_loss = np.mean(metrics.loss_history) if metrics.loss_history else 0
            loss_variance = np.var(metrics.loss_history) if len(metrics.loss_history) > 1 else 0

            avg_alignment = np.mean(metrics.gradient_alignments) if metrics.gradient_alignments else 0

            quality = DatapointQuality(
                datapoint_id=dp_id,
                avg_loss_contribution=float(avg_loss),
                gradient_alignment=float(avg_alignment),
                loss_variance=float(loss_variance),
            )

            qualities.append(quality)

        # Detect outliers based on embedding distances
        self._detect_outliers(datapoints, qualities)

        return qualities

    def analyze_with_embeddings(
        self,
        datapoints: List[TrainingDatapoint],
    ) -> List[DatapointQuality]:
        """
        Analyze datapoint quality using embeddings only.

        Useful when training metrics are not available.

        Args:
            datapoints: Training datapoints.

        Returns:
            List of DatapointQuality for each datapoint.
        """
        qualities = []

        for i, dp in enumerate(datapoints):
            dp_id = f"dp_{i}"
            quality = DatapointQuality(datapoint_id=dp_id)
            qualities.append(quality)

        self._detect_outliers(datapoints, qualities)

        return qualities

    def _detect_outliers(
        self,
        datapoints: List[TrainingDatapoint],
        qualities: List[DatapointQuality],
    ) -> None:
        """Detect outliers based on embedding distances."""
        if len(datapoints) < 3:
            return

        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Embed prompts
        prompts = [dp.prompt for dp in datapoints]
        embeddings = embedder.encode(prompts)

        # Compute centroid and distances
        centroid = embeddings.mean(axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)

        mean_dist = distances.mean()
        std_dist = distances.std()

        # Cluster
        from sklearn.cluster import KMeans

        n_clusters = min(5, len(datapoints) // 2)
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
        else:
            labels = [0] * len(datapoints)

        for i, quality in enumerate(qualities):
            quality.distance_to_centroid = float(distances[i])
            quality.cluster_id = int(labels[i])
            quality.is_outlier = distances[i] > mean_dist + self.outlier_threshold * std_dist

    def estimate_leave_one_out_influence(
        self,
        datapoints: List[TrainingDatapoint],
        full_score: float,
        score_without: Dict[int, float],
    ) -> List[float]:
        """
        Estimate leave-one-out influence for each datapoint.

        Args:
            datapoints: Training datapoints.
            full_score: Score with all datapoints.
            score_without: Score with each datapoint removed {idx: score}.

        Returns:
            List of influence values. Positive = removing improves score.
        """
        influences = []

        for i in range(len(datapoints)):
            if i in score_without:
                # Influence = score_without - full_score
                # Positive means removing this datapoint helps
                influence = score_without[i] - full_score
            else:
                influence = 0.0
            influences.append(influence)

        return influences

    def get_recommendations(
        self,
        qualities: List[DatapointQuality],
    ) -> Dict[str, List[str]]:
        """
        Get datapoint recommendations grouped by action.

        Args:
            qualities: List of DatapointQuality.

        Returns:
            Dict with keys "KEEP", "REVIEW", "REMOVE".
        """
        recommendations = {"KEEP": [], "REVIEW": [], "REMOVE": []}

        for quality in qualities:
            recommendations[quality.recommendation].append(quality.datapoint_id)

        return recommendations

    def get_problematic_datapoints(
        self,
        qualities: List[DatapointQuality],
    ) -> List[DatapointQuality]:
        """Get datapoints that should be removed or reviewed."""
        return [q for q in qualities if q.recommendation in ("REMOVE", "REVIEW")]
