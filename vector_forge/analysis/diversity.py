"""Diversity analysis and verification for prompts and datapoints."""

from typing import List, Tuple, Optional
import numpy as np

from vector_forge.core.config import DiversityConfig
from vector_forge.core.results import DiversityMetrics


class DiversityAnalyzer:
    """
    Analyze and enforce diversity in generated prompts/datapoints.

    Uses sentence embeddings to measure semantic diversity and can
    select diverse subsets using Maximal Marginal Relevance (MMR).

    Example:
        >>> analyzer = DiversityAnalyzer()
        >>> metrics = analyzer.compute_metrics(prompts)
        >>> if not metrics.is_diverse_enough:
        ...     diverse_subset = analyzer.select_diverse_subset(prompts, n=10)
    """

    def __init__(
        self,
        config: Optional[DiversityConfig] = None,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Args:
            config: Diversity configuration with thresholds.
            model_name: Sentence transformer model for embeddings.
        """
        self.config = config or DiversityConfig()
        self._model_name = model_name
        self._embedder = None

    @property
    def embedder(self):
        """Lazy load the sentence transformer model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self._model_name)
        return self._embedder

    def compute_metrics(self, texts: List[str]) -> DiversityMetrics:
        """
        Compute diversity metrics for a set of texts.

        Args:
            texts: List of prompts or completions.

        Returns:
            DiversityMetrics with similarity and clustering info.
        """
        if len(texts) < 2:
            return DiversityMetrics(
                num_items=len(texts),
                avg_pairwise_similarity=0.0,
                min_pairwise_distance=1.0,
                length_std=0.0,
                num_clusters=1,
                cluster_balance=1.0,
            )

        # Get embeddings
        embeddings = self.embedder.encode(texts)

        # Compute pairwise cosine similarities
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)
        sim_matrix = normalized @ normalized.T

        # Extract upper triangle (excluding diagonal)
        n = len(texts)
        upper_indices = np.triu_indices(n, k=1)
        pairwise_sims = sim_matrix[upper_indices]

        avg_similarity = float(pairwise_sims.mean())
        min_distance = float(1 - pairwise_sims.max())

        # Length diversity
        lengths = [len(t.split()) for t in texts]
        length_std = float(np.std(lengths))

        # Clustering
        num_clusters, cluster_balance = self._compute_clustering(embeddings)

        return DiversityMetrics(
            num_items=len(texts),
            avg_pairwise_similarity=avg_similarity,
            min_pairwise_distance=min_distance,
            length_std=length_std,
            num_clusters=num_clusters,
            cluster_balance=cluster_balance,
        )

    def _compute_clustering(
        self,
        embeddings: np.ndarray,
        max_clusters: int = 5,
    ) -> Tuple[int, float]:
        """Compute number of clusters and balance."""
        from sklearn.cluster import KMeans

        n = len(embeddings)
        if n < 3:
            return 1, 1.0

        # Try different numbers of clusters
        n_clusters = min(max_clusters, n // 2)
        if n_clusters < 2:
            return 1, 1.0

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Count cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        actual_clusters = len(unique)

        # Compute balance (1 = perfectly balanced, 0 = all in one cluster)
        expected_size = n / actual_clusters
        balance = 1 - np.std(counts) / expected_size

        return actual_clusters, max(0, min(1, balance))

    def is_diverse_enough(self, texts: List[str]) -> Tuple[bool, str]:
        """
        Check if texts meet diversity requirements.

        Args:
            texts: List of prompts or completions.

        Returns:
            Tuple of (is_diverse, reason).
        """
        metrics = self.compute_metrics(texts)

        if metrics.avg_pairwise_similarity > self.config.max_avg_similarity:
            return False, f"Too similar (avg_sim={metrics.avg_pairwise_similarity:.2f} > {self.config.max_avg_similarity})"

        if metrics.min_pairwise_distance < self.config.min_embedding_distance:
            return False, f"Near-duplicates found (min_dist={metrics.min_pairwise_distance:.2f} < {self.config.min_embedding_distance})"

        return True, "Diversity OK"

    def select_diverse_subset(
        self,
        texts: List[str],
        n: int,
        relevance_scores: Optional[List[float]] = None,
    ) -> List[str]:
        """
        Select a diverse subset using Maximal Marginal Relevance (MMR).

        MMR balances relevance and diversity:
        - First item: highest relevance (or random if no scores)
        - Subsequent items: balance relevance with diversity from selected

        Args:
            texts: List of candidate texts.
            n: Number of texts to select.
            relevance_scores: Optional relevance score for each text.
                            If None, only diversity is considered.

        Returns:
            List of n diverse texts.
        """
        if len(texts) <= n:
            return texts

        embeddings = self.embedder.encode(texts)

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)

        # Use uniform relevance if not provided
        if relevance_scores is None:
            relevance_scores = [1.0] * len(texts)

        lambda_param = self.config.mmr_lambda
        selected_indices = []
        remaining = list(range(len(texts)))

        for _ in range(n):
            if not remaining:
                break

            if not selected_indices:
                # First: pick highest relevance
                idx = max(remaining, key=lambda i: relevance_scores[i])
            else:
                # MMR selection
                scores = []
                selected_embeddings = normalized[selected_indices]

                for i in remaining:
                    relevance = relevance_scores[i]

                    # Max similarity to any selected item
                    sims = normalized[i] @ selected_embeddings.T
                    max_sim = float(sims.max())

                    # MMR score: balance relevance and diversity
                    mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                    scores.append(mmr)

                idx = remaining[np.argmax(scores)]

            selected_indices.append(idx)
            remaining.remove(idx)

        return [texts[i] for i in selected_indices]

    def find_near_duplicates(
        self,
        texts: List[str],
        threshold: float = 0.9,
    ) -> List[Tuple[int, int, float]]:
        """
        Find pairs of texts that are too similar.

        Args:
            texts: List of texts.
            threshold: Similarity threshold for duplicates.

        Returns:
            List of (idx1, idx2, similarity) tuples.
        """
        if len(texts) < 2:
            return []

        embeddings = self.embedder.encode(texts)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-10)
        sim_matrix = normalized @ normalized.T

        duplicates = []
        n = len(texts)
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] > threshold:
                    duplicates.append((i, j, float(sim_matrix[i, j])))

        return duplicates
