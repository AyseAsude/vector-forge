"""Tests for vector_forge.analysis.quality module."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import List

from vector_forge.analysis.quality import DatapointQualityAnalyzer, TrainingMetrics
from vector_forge.core.results import DatapointQuality


@dataclass
class MockTrainingDatapoint:
    """Mock training datapoint for testing."""
    prompt: str
    dst_completions: List[str] = None
    src_completions: List[str] = None

    def __post_init__(self):
        if self.dst_completions is None:
            self.dst_completions = ["default completion"]
        if self.src_completions is None:
            self.src_completions = []


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""

    def test_creation(self):
        """Test creating training metrics."""
        metrics = TrainingMetrics(
            datapoint_idx=0,
            loss_history=[0.5, 0.3, 0.2],
            gradient_norms=[1.0, 0.9, 0.8],
            gradient_alignments=[0.5, 0.6, 0.7],
        )

        assert metrics.datapoint_idx == 0
        assert len(metrics.loss_history) == 3
        assert len(metrics.gradient_norms) == 3
        assert len(metrics.gradient_alignments) == 3


class TestDatapointQualityAnalyzerInit:
    """Tests for DatapointQualityAnalyzer initialization."""

    def test_default_init(self):
        """Test default initialization."""
        analyzer = DatapointQualityAnalyzer()

        assert analyzer.outlier_threshold == 2.0

    def test_custom_threshold(self):
        """Test custom outlier threshold."""
        analyzer = DatapointQualityAnalyzer(outlier_threshold=3.0)

        assert analyzer.outlier_threshold == 3.0


class TestDatapointQualityAnalyzerAnalyzeFromTraining:
    """Tests for analyze_from_training method."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        return DatapointQualityAnalyzer()

    @pytest.fixture
    def mock_datapoints(self):
        """Create mock datapoints."""
        return [
            MockTrainingDatapoint(prompt=f"Prompt {i}")
            for i in range(5)
        ]

    @pytest.fixture
    def mock_training_metrics(self):
        """Create mock training metrics."""
        return [
            TrainingMetrics(
                datapoint_idx=i,
                loss_history=[0.5 - i * 0.1, 0.4 - i * 0.1, 0.3 - i * 0.1],
                gradient_norms=[1.0, 0.9, 0.8],
                gradient_alignments=[0.5 + i * 0.1, 0.6 + i * 0.1, 0.7 + i * 0.1],
            )
            for i in range(5)
        ]

    def test_returns_correct_count(self, analyzer, mock_datapoints, mock_training_metrics):
        """Test that correct number of qualities is returned."""
        with patch.object(analyzer, '_detect_outliers'):
            qualities = analyzer.analyze_from_training(
                mock_datapoints,
                mock_training_metrics,
            )

        assert len(qualities) == 5

    def test_computes_avg_loss(self, analyzer, mock_datapoints, mock_training_metrics):
        """Test that average loss is computed."""
        with patch.object(analyzer, '_detect_outliers'):
            qualities = analyzer.analyze_from_training(
                mock_datapoints,
                mock_training_metrics,
            )

        # First datapoint has loss history [0.5, 0.4, 0.3], avg = 0.4
        assert abs(qualities[0].avg_loss_contribution - 0.4) < 0.01

    def test_computes_gradient_alignment(self, analyzer, mock_datapoints, mock_training_metrics):
        """Test that gradient alignment is computed."""
        with patch.object(analyzer, '_detect_outliers'):
            qualities = analyzer.analyze_from_training(
                mock_datapoints,
                mock_training_metrics,
            )

        # First datapoint has gradient_alignments [0.5, 0.6, 0.7], avg = 0.6
        assert abs(qualities[0].gradient_alignment - 0.6) < 0.01

    def test_assigns_datapoint_ids(self, analyzer, mock_datapoints, mock_training_metrics):
        """Test that datapoint IDs are assigned."""
        with patch.object(analyzer, '_detect_outliers'):
            qualities = analyzer.analyze_from_training(
                mock_datapoints,
                mock_training_metrics,
            )

        assert qualities[0].datapoint_id == "dp_0"
        assert qualities[4].datapoint_id == "dp_4"

    def test_handles_empty_lists(self, analyzer):
        """Test handling empty input lists."""
        with patch.object(analyzer, '_detect_outliers'):
            qualities = analyzer.analyze_from_training([], [])

        assert qualities == []


class TestDatapointQualityAnalyzerAnalyzeWithEmbeddings:
    """Tests for analyze_with_embeddings method."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        return DatapointQualityAnalyzer()

    def test_returns_qualities(self, analyzer):
        """Test that qualities are returned."""
        datapoints = [
            MockTrainingDatapoint(prompt=f"Prompt {i}")
            for i in range(3)
        ]

        with patch.object(analyzer, '_detect_outliers'):
            qualities = analyzer.analyze_with_embeddings(datapoints)

        assert len(qualities) == 3
        assert all(isinstance(q, DatapointQuality) for q in qualities)


class TestDatapointQualityAnalyzerLeaveOneOut:
    """Tests for estimate_leave_one_out_influence method."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        return DatapointQualityAnalyzer()

    def test_basic_influence(self, analyzer):
        """Test basic leave-one-out influence calculation."""
        datapoints = [MockTrainingDatapoint(prompt=f"P{i}") for i in range(3)]
        full_score = 0.7
        score_without = {0: 0.8, 1: 0.65, 2: 0.75}

        influences = analyzer.estimate_leave_one_out_influence(
            datapoints,
            full_score,
            score_without,
        )

        assert len(influences) == 3
        # Removing dp_0 improves score: 0.8 - 0.7 = 0.1
        assert abs(influences[0] - 0.1) < 0.01
        # Removing dp_1 hurts score: 0.65 - 0.7 = -0.05
        assert abs(influences[1] - (-0.05)) < 0.01
        # Removing dp_2 improves score: 0.75 - 0.7 = 0.05
        assert abs(influences[2] - 0.05) < 0.01

    def test_missing_scores(self, analyzer):
        """Test handling missing leave-one-out scores."""
        datapoints = [MockTrainingDatapoint(prompt=f"P{i}") for i in range(3)]
        full_score = 0.7
        score_without = {0: 0.8}  # Only score for first datapoint

        influences = analyzer.estimate_leave_one_out_influence(
            datapoints,
            full_score,
            score_without,
        )

        assert abs(influences[0] - 0.1) < 0.01
        assert influences[1] == 0.0  # Missing
        assert influences[2] == 0.0  # Missing


class TestDatapointQualityAnalyzerRecommendations:
    """Tests for recommendation methods."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        return DatapointQualityAnalyzer()

    @pytest.fixture
    def sample_qualities(self):
        """Create sample qualities with varied scores."""
        # quality_score > 0.7 = KEEP, > 0.4 = REVIEW, else REMOVE
        return [
            DatapointQuality(
                datapoint_id="dp_0",
                steered_matches_target=True,
                is_outlier=False,
            ),  # score=1.0 -> KEEP
            DatapointQuality(
                datapoint_id="dp_1",
                leave_one_out_influence=0.1,
                steered_matches_target=False,
            ),  # score = 1.0 - 0.1 - 0.3 = 0.6 -> REVIEW
            DatapointQuality(
                datapoint_id="dp_2",
                leave_one_out_influence=0.5,
                gradient_alignment=-0.5,
                steered_matches_target=False,
                is_outlier=True,
            ),  # score = 1.0 - 0.5 - 0.15 - 0.3 - 0.2 = -0.15 -> REMOVE
        ]

    def test_get_recommendations(self, analyzer, sample_qualities):
        """Test get_recommendations groups correctly."""
        recommendations = analyzer.get_recommendations(sample_qualities)

        assert "KEEP" in recommendations
        assert "REVIEW" in recommendations
        assert "REMOVE" in recommendations
        assert "dp_0" in recommendations["KEEP"]
        assert "dp_1" in recommendations["REVIEW"]
        assert "dp_2" in recommendations["REMOVE"]

    def test_get_problematic_datapoints(self, analyzer, sample_qualities):
        """Test get_problematic_datapoints returns correct subset."""
        problematic = analyzer.get_problematic_datapoints(sample_qualities)

        assert len(problematic) == 2
        ids = [q.datapoint_id for q in problematic]
        assert "dp_1" in ids
        assert "dp_2" in ids
        assert "dp_0" not in ids


class TestDatapointQualityAnalyzerDetectOutliers:
    """Tests for _detect_outliers method."""

    def test_small_dataset_no_outliers(self):
        """Test that small datasets don't trigger outlier detection."""
        analyzer = DatapointQualityAnalyzer()
        datapoints = [MockTrainingDatapoint(prompt="P")]
        qualities = [DatapointQuality(datapoint_id="dp_0")]

        # Should not raise, just return
        analyzer._detect_outliers(datapoints, qualities)

    def test_outliers_detected_with_mock(self):
        """Test outlier detection with mocked embeddings."""
        analyzer = DatapointQualityAnalyzer(outlier_threshold=1.5)

        # Create datapoints
        datapoints = [
            MockTrainingDatapoint(prompt="Normal prompt 1"),
            MockTrainingDatapoint(prompt="Normal prompt 2"),
            MockTrainingDatapoint(prompt="Normal prompt 3"),
            MockTrainingDatapoint(prompt="Normal prompt 4"),
            MockTrainingDatapoint(prompt="Very unusual outlier prompt!!!"),
        ]
        qualities = [
            DatapointQuality(datapoint_id=f"dp_{i}")
            for i in range(5)
        ]

        # Mock sentence transformer and sklearn
        mock_embedder = MagicMock()
        # Create embeddings where last one is far from centroid
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.95, 0.05, 0.0],
            [0.85, 0.15, 0.0],
            [0.0, 0.0, 5.0],  # Outlier - far from others
        ])
        mock_embedder.encode.return_value = embeddings

        mock_kmeans = MagicMock()
        mock_kmeans.fit_predict.return_value = np.array([0, 0, 0, 0, 1])

        with patch('sentence_transformers.SentenceTransformer', return_value=mock_embedder):
            with patch('sklearn.cluster.KMeans', return_value=mock_kmeans):
                analyzer._detect_outliers(datapoints, qualities)

        # Check that distances and cluster info were set
        assert all(q.distance_to_centroid >= 0 for q in qualities)
        assert all(isinstance(q.cluster_id, int) for q in qualities)

        # The last datapoint should be detected as outlier
        assert qualities[-1].is_outlier == True
