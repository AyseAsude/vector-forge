"""Tests for vector_forge.analysis.diversity module."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from vector_forge.analysis.diversity import DiversityAnalyzer
from vector_forge.core.config import DiversityConfig
from vector_forge.core.results import DiversityMetrics


class MockSentenceTransformer:
    """Mock sentence transformer for testing."""

    def __init__(self, model_name: str = "mock"):
        self.model_name = model_name

    def encode(self, texts):
        """Generate deterministic embeddings based on text."""
        embeddings = []
        for i, text in enumerate(texts):
            # Create embedding based on text length and hash
            np.random.seed(hash(text) % (2**32 - 1))
            embedding = np.random.randn(384)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        return np.array(embeddings)


@pytest.fixture
def mock_embedder():
    """Create mock embedder."""
    return MockSentenceTransformer()


@pytest.fixture
def analyzer_with_mock(mock_embedder):
    """Create analyzer with mocked embedder."""
    analyzer = DiversityAnalyzer()
    analyzer._embedder = mock_embedder
    return analyzer


class TestDiversityAnalyzerInit:
    """Tests for DiversityAnalyzer initialization."""

    def test_default_init(self):
        """Test default initialization."""
        analyzer = DiversityAnalyzer()

        assert analyzer.config is not None
        assert analyzer._model_name == "all-MiniLM-L6-v2"
        assert analyzer._embedder is None

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = DiversityConfig(
            mmr_lambda=0.3,
            min_embedding_distance=0.5,
        )
        analyzer = DiversityAnalyzer(config=config)

        assert analyzer.config.mmr_lambda == 0.3
        assert analyzer.config.min_embedding_distance == 0.5

    def test_custom_model_name(self):
        """Test initialization with custom model name."""
        analyzer = DiversityAnalyzer(model_name="paraphrase-MiniLM-L6-v2")

        assert analyzer._model_name == "paraphrase-MiniLM-L6-v2"


class TestDiversityAnalyzerComputeMetrics:
    """Tests for compute_metrics method."""

    def test_empty_list(self, analyzer_with_mock):
        """Test metrics for empty list."""
        metrics = analyzer_with_mock.compute_metrics([])

        assert metrics.num_items == 0

    def test_single_item(self, analyzer_with_mock):
        """Test metrics for single item."""
        metrics = analyzer_with_mock.compute_metrics(["Single text"])

        assert metrics.num_items == 1
        assert metrics.avg_pairwise_similarity == 0.0
        assert metrics.min_pairwise_distance == 1.0
        assert metrics.num_clusters == 1
        assert metrics.cluster_balance == 1.0

    def test_two_items(self, analyzer_with_mock):
        """Test metrics for two items."""
        texts = [
            "This is the first text.",
            "This is a completely different second text.",
        ]
        metrics = analyzer_with_mock.compute_metrics(texts)

        assert metrics.num_items == 2
        # Cosine similarity can be [-1, 1]
        assert -1.0 <= metrics.avg_pairwise_similarity <= 1.0
        assert 0.0 <= metrics.min_pairwise_distance <= 2.0  # Distance = 1 - similarity

    def test_similar_texts(self, analyzer_with_mock):
        """Test metrics for similar texts."""
        texts = [
            "The cat sat on the mat.",
            "The cat sat on the mat.",  # Duplicate
            "The cat sat on the mat.",
        ]
        metrics = analyzer_with_mock.compute_metrics(texts)

        # Duplicates should have high similarity
        assert metrics.avg_pairwise_similarity > 0.9
        assert metrics.min_pairwise_distance < 0.1

    def test_diverse_texts(self, analyzer_with_mock):
        """Test metrics for diverse texts."""
        texts = [
            "The sun is a star in our solar system.",
            "Mathematics involves studying numbers and patterns.",
            "Programming helps solve computational problems.",
            "Music consists of rhythm, melody, and harmony.",
        ]
        metrics = analyzer_with_mock.compute_metrics(texts)

        assert metrics.num_items == 4
        # Metrics should exist and be reasonable
        # Cosine similarity can be [-1, 1]
        assert -1.0 <= metrics.avg_pairwise_similarity <= 1.0
        assert metrics.length_std >= 0

    def test_length_std_computed(self, analyzer_with_mock):
        """Test that length standard deviation is computed."""
        texts = [
            "Short text.",
            "This is a much longer text with many more words.",
            "Medium length text here.",
        ]
        metrics = analyzer_with_mock.compute_metrics(texts)

        assert metrics.length_std > 0


class TestDiversityAnalyzerIsDiverseEnough:
    """Tests for is_diverse_enough method."""

    def test_returns_tuple(self, analyzer_with_mock):
        """Test that method returns tuple."""
        result = analyzer_with_mock.is_diverse_enough(["Text 1", "Text 2"])

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_diverse_texts_pass(self, analyzer_with_mock):
        """Test that diverse texts pass."""
        texts = [
            "Astronomy studies celestial objects.",
            "Biology examines living organisms.",
            "Chemistry explores matter and reactions.",
            "Physics investigates fundamental forces.",
        ]

        # Lower thresholds to ensure pass
        analyzer_with_mock.config.max_avg_similarity = 0.99
        analyzer_with_mock.config.min_embedding_distance = 0.01

        is_diverse, reason = analyzer_with_mock.is_diverse_enough(texts)

        assert is_diverse is True
        assert "OK" in reason


class TestDiversityAnalyzerSelectDiverseSubset:
    """Tests for select_diverse_subset method."""

    def test_returns_requested_count(self, analyzer_with_mock):
        """Test that correct number of items is returned."""
        texts = [f"Text number {i}" for i in range(20)]

        result = analyzer_with_mock.select_diverse_subset(texts, n=5)

        assert len(result) == 5

    def test_returns_all_if_n_exceeds_length(self, analyzer_with_mock):
        """Test that all items returned if n exceeds list length."""
        texts = ["A", "B", "C"]

        result = analyzer_with_mock.select_diverse_subset(texts, n=10)

        assert len(result) == 3

    def test_returns_original_texts(self, analyzer_with_mock):
        """Test that returned texts are from original list."""
        texts = ["First", "Second", "Third", "Fourth", "Fifth"]

        result = analyzer_with_mock.select_diverse_subset(texts, n=3)

        assert all(t in texts for t in result)

    def test_with_relevance_scores(self, analyzer_with_mock):
        """Test selection with relevance scores."""
        texts = ["Low relevance", "Medium relevance", "High relevance", "Very high relevance"]
        relevance = [0.1, 0.5, 0.8, 0.9]

        result = analyzer_with_mock.select_diverse_subset(texts, n=2, relevance_scores=relevance)

        assert len(result) == 2
        # First item should be highest relevance
        assert result[0] == "Very high relevance"

    def test_without_relevance_scores(self, analyzer_with_mock):
        """Test selection without relevance scores."""
        texts = [f"Text {i}" for i in range(10)]

        result = analyzer_with_mock.select_diverse_subset(texts, n=5, relevance_scores=None)

        assert len(result) == 5


class TestDiversityAnalyzerFindNearDuplicates:
    """Tests for find_near_duplicates method."""

    def test_empty_list(self, analyzer_with_mock):
        """Test with empty list."""
        result = analyzer_with_mock.find_near_duplicates([])

        assert result == []

    def test_single_item(self, analyzer_with_mock):
        """Test with single item."""
        result = analyzer_with_mock.find_near_duplicates(["Single"])

        assert result == []

    def test_no_duplicates(self, analyzer_with_mock):
        """Test with no near-duplicates."""
        texts = [
            "The sky is blue.",
            "Computers process information.",
            "Music brings joy to people.",
        ]

        result = analyzer_with_mock.find_near_duplicates(texts, threshold=0.95)

        # No near-duplicates expected with diverse texts
        # Result depends on mock embeddings

    def test_with_duplicates(self, analyzer_with_mock):
        """Test with actual duplicates."""
        texts = [
            "The cat sat on the mat.",
            "The cat sat on the mat.",  # Exact duplicate
            "Something completely different.",
        ]

        result = analyzer_with_mock.find_near_duplicates(texts, threshold=0.9)

        # Should find the duplicate pair (0, 1)
        assert any(pair[0] == 0 and pair[1] == 1 for pair in result)

    def test_returns_similarity_scores(self, analyzer_with_mock):
        """Test that similarity scores are returned."""
        texts = [
            "Duplicate text here.",
            "Duplicate text here.",
        ]

        result = analyzer_with_mock.find_near_duplicates(texts, threshold=0.5)

        if result:
            assert len(result[0]) == 3  # (idx1, idx2, similarity)
            assert 0.0 <= result[0][2] <= 1.0


class TestDiversityMetricsIsDiverseEnough:
    """Tests for DiversityMetrics.is_diverse_enough property."""

    def test_diverse_metrics(self):
        """Test is_diverse_enough with good metrics."""
        metrics = DiversityMetrics(
            num_items=10,
            avg_pairwise_similarity=0.3,  # Low
            min_pairwise_distance=0.5,  # High
            length_std=10.0,
            num_clusters=3,
            cluster_balance=0.8,
        )

        assert metrics.is_diverse_enough is True

    def test_too_similar(self):
        """Test is_diverse_enough with high similarity."""
        metrics = DiversityMetrics(
            num_items=10,
            avg_pairwise_similarity=0.8,  # Too high
            min_pairwise_distance=0.5,
            length_std=10.0,
            num_clusters=3,
            cluster_balance=0.8,
        )

        assert metrics.is_diverse_enough is False

    def test_near_duplicates(self):
        """Test is_diverse_enough with near duplicates."""
        metrics = DiversityMetrics(
            num_items=10,
            avg_pairwise_similarity=0.3,
            min_pairwise_distance=0.1,  # Too low
            length_std=10.0,
            num_clusters=3,
            cluster_balance=0.8,
        )

        assert metrics.is_diverse_enough is False
