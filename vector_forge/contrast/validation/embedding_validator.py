"""Embedding-based contrast validation.

This module provides fast, free validation using sentence embeddings
to ensure dst and src are semantically different enough.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from vector_forge.contrast.protocols import (
    ContrastValidatorProtocol,
    ContrastPair,
    BehaviorAnalysis,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class EmbeddingContrastValidator(ContrastValidatorProtocol):
    """Fast, free validation using sentence embeddings.

    Validates that dst and src have sufficient semantic distance.
    This is a cheap pre-filter before more expensive LLM validation.

    Example:
        >>> validator = EmbeddingContrastValidator(min_distance=0.3)
        >>> result = await validator.validate(pair, analysis)
        >>> if not result.is_valid:
        ...     print(f"Too similar: {result.semantic_score}")
    """

    def __init__(
        self,
        min_distance: float = 0.3,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """Initialize the embedding validator.

        Args:
            min_distance: Minimum cosine distance required (0-1).
            model_name: Sentence transformer model to use.
        """
        self._min_distance = min_distance
        self._model_name = model_name
        self._embedder: Optional[object] = None

    @property
    def embedder(self):
        """Lazy load the sentence transformer model."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self._model_name)
                logger.info(f"Loaded sentence transformer: {self._model_name}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for embedding validation. "
                    "Install with: pip install sentence-transformers"
                )
        return self._embedder

    async def validate(
        self,
        pair: ContrastPair,
        analysis: BehaviorAnalysis,
    ) -> ValidationResult:
        """Validate semantic distance between dst and src.

        Args:
            pair: The contrast pair to validate.
            analysis: Behavior analysis (not used by this validator).

        Returns:
            ValidationResult with semantic_score set.
        """
        try:
            # Encode both texts
            embeddings = self.embedder.encode([pair.dst, pair.src])
            emb_dst, emb_src = embeddings[0], embeddings[1]

            # Compute cosine similarity
            norm_dst = np.linalg.norm(emb_dst)
            norm_src = np.linalg.norm(emb_src)

            if norm_dst == 0 or norm_src == 0:
                logger.warning("Zero norm embedding detected")
                return ValidationResult(
                    is_valid=False,
                    contrast_quality=0.0,
                    reasoning="One or both texts produced zero embedding",
                    semantic_score=0.0,
                )

            similarity = float(np.dot(emb_dst, emb_src) / (norm_dst * norm_src))
            distance = 1.0 - similarity

            is_valid = distance >= self._min_distance

            # Map distance to score (0-10)
            # distance of 0.3 = 6, distance of 0.5 = 8, distance of 0.7 = 10
            semantic_score = min(10.0, (distance / 0.7) * 10.0)

            reasoning = f"Semantic distance: {distance:.3f}"
            if not is_valid:
                reasoning += f" (below threshold {self._min_distance})"

            return ValidationResult(
                is_valid=is_valid,
                contrast_quality=semantic_score,
                reasoning=reasoning,
                semantic_score=semantic_score,
            )

        except Exception as e:
            logger.error(f"Embedding validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                contrast_quality=0.0,
                reasoning=f"Embedding error: {e}",
                semantic_score=0.0,
            )

    def compute_batch_distances(self, pairs: list[ContrastPair]) -> list[float]:
        """Compute semantic distances for multiple pairs efficiently.

        Args:
            pairs: List of contrast pairs.

        Returns:
            List of semantic distances.
        """
        if not pairs:
            return []

        # Batch encode all texts
        texts = []
        for pair in pairs:
            texts.extend([pair.dst, pair.src])

        embeddings = self.embedder.encode(texts)

        # Compute pairwise distances
        distances = []
        for i in range(0, len(embeddings), 2):
            emb_dst = embeddings[i]
            emb_src = embeddings[i + 1]

            norm_dst = np.linalg.norm(emb_dst)
            norm_src = np.linalg.norm(emb_src)

            if norm_dst == 0 or norm_src == 0:
                distances.append(0.0)
            else:
                similarity = float(np.dot(emb_dst, emb_src) / (norm_dst * norm_src))
                distances.append(1.0 - similarity)

        return distances
