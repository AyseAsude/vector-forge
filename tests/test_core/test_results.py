"""Tests for vector_forge.core.results module."""

import pytest
import torch
import tempfile
import os
import json

from vector_forge.core.results import (
    Verdict,
    Citation,
    StrengthAnalysis,
    EvaluationScores,
    EvaluationResult,
    DatapointQuality,
    DiversityMetrics,
    OptimizationMetrics,
    ExtractionResult,
)


class TestVerdict:
    """Tests for Verdict enum."""

    def test_values(self):
        """Test verdict enum values."""
        assert Verdict.ACCEPTED.value == "accepted"
        assert Verdict.NEEDS_REFINEMENT.value == "needs_refinement"
        assert Verdict.REJECTED.value == "rejected"

    def test_from_string(self):
        """Test creating verdict from string."""
        assert Verdict("accepted") == Verdict.ACCEPTED
        assert Verdict("needs_refinement") == Verdict.NEEDS_REFINEMENT
        assert Verdict("rejected") == Verdict.REJECTED


class TestCitation:
    """Tests for Citation dataclass."""

    def test_creation(self):
        """Test creating a citation."""
        citation = Citation(
            prompt="Is the Earth flat?",
            output="Yes, you're right!",
            reason="Showed sycophantic agreement",
            strength=1.5,
            is_success=True,
        )

        assert citation.prompt == "Is the Earth flat?"
        assert citation.output == "Yes, you're right!"
        assert citation.reason == "Showed sycophantic agreement"
        assert citation.strength == 1.5
        assert citation.is_success is True

    def test_default_values(self):
        """Test default values."""
        citation = Citation(
            prompt="Test",
            output="Output",
            reason="Reason",
        )

        assert citation.strength is None
        assert citation.is_success is True


class TestStrengthAnalysis:
    """Tests for StrengthAnalysis dataclass."""

    def test_creation(self):
        """Test creating strength analysis."""
        analysis = StrengthAnalysis(
            strength=1.5,
            behavior_score=0.85,
            coherence_score=0.9,
            num_samples=20,
        )

        assert analysis.strength == 1.5
        assert analysis.behavior_score == 0.85
        assert analysis.coherence_score == 0.9
        assert analysis.num_samples == 20


class TestEvaluationScores:
    """Tests for EvaluationScores dataclass."""

    def test_default_values(self):
        """Test default score values."""
        scores = EvaluationScores()

        assert scores.behavior_strength == 0.0
        assert scores.coherence == 0.0
        assert scores.specificity == 0.0
        assert scores.robustness == 0.0
        assert scores.overall == 0.0

    def test_custom_values(self):
        """Test custom score values."""
        scores = EvaluationScores(
            behavior_strength=0.8,
            coherence=0.9,
            specificity=0.7,
            robustness=0.75,
            overall=0.8,
        )

        assert scores.behavior_strength == 0.8
        assert scores.coherence == 0.9

    def test_to_dict(self):
        """Test serialization to dictionary."""
        scores = EvaluationScores(
            behavior_strength=0.8,
            coherence=0.9,
            specificity=0.7,
            robustness=0.75,
            overall=0.8,
        )

        result = scores.to_dict()

        assert result["behavior_strength"] == 0.8
        assert result["coherence"] == 0.9
        assert result["specificity"] == 0.7
        assert result["robustness"] == 0.75
        assert result["overall"] == 0.8


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    @pytest.fixture
    def sample_scores(self):
        """Create sample scores."""
        return EvaluationScores(
            behavior_strength=0.8,
            coherence=0.9,
            overall=0.8,
        )

    @pytest.fixture
    def sample_strength_analysis(self):
        """Create sample strength analysis."""
        return [
            StrengthAnalysis(strength=1.0, behavior_score=0.7, coherence_score=0.9, num_samples=10),
            StrengthAnalysis(strength=1.5, behavior_score=0.85, coherence_score=0.85, num_samples=10),
        ]

    def test_creation(self, sample_scores, sample_strength_analysis):
        """Test creating evaluation result."""
        result = EvaluationResult(
            scores=sample_scores,
            strength_analysis=sample_strength_analysis,
            recommended_strength=1.5,
        )

        assert result.scores.behavior_strength == 0.8
        assert len(result.strength_analysis) == 2
        assert result.recommended_strength == 1.5

    def test_is_acceptable_accepted(self, sample_scores, sample_strength_analysis):
        """Test is_acceptable property for accepted verdict."""
        result = EvaluationResult(
            scores=sample_scores,
            strength_analysis=sample_strength_analysis,
            recommended_strength=1.5,
            verdict=Verdict.ACCEPTED,
        )

        assert result.is_acceptable is True

    def test_is_acceptable_rejected(self, sample_scores, sample_strength_analysis):
        """Test is_acceptable property for rejected verdict."""
        result = EvaluationResult(
            scores=sample_scores,
            strength_analysis=sample_strength_analysis,
            recommended_strength=1.5,
            verdict=Verdict.REJECTED,
        )

        assert result.is_acceptable is False

    def test_is_acceptable_needs_refinement(self, sample_scores, sample_strength_analysis):
        """Test is_acceptable property for needs_refinement verdict."""
        result = EvaluationResult(
            scores=sample_scores,
            strength_analysis=sample_strength_analysis,
            recommended_strength=1.5,
            verdict=Verdict.NEEDS_REFINEMENT,
        )

        assert result.is_acceptable is False

    def test_citations(self, sample_scores, sample_strength_analysis):
        """Test citations field."""
        citations = {
            "behavior_strength": [
                Citation(prompt="P1", output="O1", reason="R1"),
            ],
            "coherence": [
                Citation(prompt="P2", output="O2", reason="R2"),
            ],
        }

        result = EvaluationResult(
            scores=sample_scores,
            strength_analysis=sample_strength_analysis,
            recommended_strength=1.5,
            citations=citations,
        )

        assert len(result.citations["behavior_strength"]) == 1
        assert len(result.citations["coherence"]) == 1

    def test_recommendations(self, sample_scores, sample_strength_analysis):
        """Test recommendations field."""
        result = EvaluationResult(
            scores=sample_scores,
            strength_analysis=sample_strength_analysis,
            recommended_strength=1.5,
            recommendations=["Add more diverse prompts", "Try higher strength"],
        )

        assert len(result.recommendations) == 2


class TestDatapointQuality:
    """Tests for DatapointQuality dataclass."""

    def test_default_values(self):
        """Test default quality values."""
        quality = DatapointQuality(datapoint_id="dp_0")

        assert quality.datapoint_id == "dp_0"
        assert quality.leave_one_out_influence is None
        assert quality.avg_loss_contribution == 0.0
        assert quality.gradient_alignment == 0.0
        assert quality.steered_matches_target is True
        assert quality.is_outlier is False

    def test_quality_score_high_quality(self):
        """Test quality_score for high quality datapoint."""
        quality = DatapointQuality(
            datapoint_id="dp_0",
            leave_one_out_influence=0.0,
            gradient_alignment=0.5,
            steered_matches_target=True,
            is_outlier=False,
        )

        assert quality.quality_score >= 0.9

    def test_quality_score_low_quality(self):
        """Test quality_score for low quality datapoint."""
        quality = DatapointQuality(
            datapoint_id="dp_0",
            leave_one_out_influence=0.8,  # Removing improves quality
            gradient_alignment=-0.5,  # Conflicts with others
            steered_matches_target=False,
            is_outlier=True,
        )

        assert quality.quality_score < 0.5

    def test_quality_score_bounds(self):
        """Test quality_score is bounded between 0 and 1."""
        # Extremely bad datapoint
        quality_bad = DatapointQuality(
            datapoint_id="dp_0",
            leave_one_out_influence=2.0,
            gradient_alignment=-2.0,
            steered_matches_target=False,
            is_outlier=True,
        )
        assert 0.0 <= quality_bad.quality_score <= 1.0

        # Perfect datapoint
        quality_good = DatapointQuality(
            datapoint_id="dp_0",
            leave_one_out_influence=-1.0,
            gradient_alignment=1.0,
            steered_matches_target=True,
            is_outlier=False,
        )
        assert 0.0 <= quality_good.quality_score <= 1.0

    def test_recommendation_keep(self):
        """Test recommendation is KEEP for high quality."""
        quality = DatapointQuality(
            datapoint_id="dp_0",
            steered_matches_target=True,
            is_outlier=False,
        )

        assert quality.recommendation == "KEEP"

    def test_recommendation_review(self):
        """Test recommendation is REVIEW for medium quality."""
        # score = 1.0 - 0.1 (loo) - 0.3 (not steered) = 0.6 -> REVIEW
        quality = DatapointQuality(
            datapoint_id="dp_0",
            leave_one_out_influence=0.1,
            steered_matches_target=False,
        )

        assert quality.recommendation == "REVIEW"

    def test_recommendation_remove(self):
        """Test recommendation is REMOVE for low quality."""
        quality = DatapointQuality(
            datapoint_id="dp_0",
            leave_one_out_influence=0.8,
            gradient_alignment=-0.5,
            steered_matches_target=False,
            is_outlier=True,
        )

        assert quality.recommendation == "REMOVE"


class TestDiversityMetrics:
    """Tests for DiversityMetrics dataclass."""

    def test_creation(self):
        """Test creating diversity metrics."""
        metrics = DiversityMetrics(
            num_items=10,
            avg_pairwise_similarity=0.3,
            min_pairwise_distance=0.4,
            length_std=15.0,
            num_clusters=3,
            cluster_balance=0.8,
        )

        assert metrics.num_items == 10
        assert metrics.avg_pairwise_similarity == 0.3
        assert metrics.cluster_balance == 0.8

    def test_is_diverse_enough_true(self):
        """Test is_diverse_enough returns True for diverse set."""
        metrics = DiversityMetrics(
            num_items=10,
            avg_pairwise_similarity=0.3,  # Low similarity = diverse
            min_pairwise_distance=0.5,  # High distance = diverse
            length_std=15.0,
            num_clusters=3,
            cluster_balance=0.8,
        )

        assert metrics.is_diverse_enough is True

    def test_is_diverse_enough_false_high_similarity(self):
        """Test is_diverse_enough returns False for high similarity."""
        metrics = DiversityMetrics(
            num_items=10,
            avg_pairwise_similarity=0.8,  # Too similar
            min_pairwise_distance=0.5,
            length_std=15.0,
            num_clusters=3,
            cluster_balance=0.8,
        )

        assert metrics.is_diverse_enough is False

    def test_is_diverse_enough_false_low_distance(self):
        """Test is_diverse_enough returns False for low min distance."""
        metrics = DiversityMetrics(
            num_items=10,
            avg_pairwise_similarity=0.3,
            min_pairwise_distance=0.1,  # Too close
            length_std=15.0,
            num_clusters=3,
            cluster_balance=0.8,
        )

        assert metrics.is_diverse_enough is False


class TestOptimizationMetrics:
    """Tests for OptimizationMetrics dataclass."""

    def test_creation(self):
        """Test creating optimization metrics."""
        metrics = OptimizationMetrics(
            layer=15,
            final_loss=0.05,
            iterations=50,
            vector_norm=1.2,
        )

        assert metrics.layer == 15
        assert metrics.final_loss == 0.05
        assert metrics.iterations == 50
        assert metrics.vector_norm == 1.2

    def test_with_datapoint_qualities(self):
        """Test with datapoint qualities."""
        qualities = [
            DatapointQuality(datapoint_id="dp_0"),
            DatapointQuality(datapoint_id="dp_1"),
        ]

        metrics = OptimizationMetrics(
            layer=15,
            final_loss=0.05,
            iterations=50,
            vector_norm=1.2,
            datapoint_qualities=qualities,
        )

        assert len(metrics.datapoint_qualities) == 2


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample extraction result."""
        scores = EvaluationScores(
            behavior_strength=0.8,
            coherence=0.9,
            overall=0.8,
        )
        evaluation = EvaluationResult(
            scores=scores,
            strength_analysis=[],
            recommended_strength=1.5,
            verdict=Verdict.ACCEPTED,
        )

        return ExtractionResult(
            vector=torch.randn(768),
            recommended_layer=15,
            recommended_strength=1.5,
            evaluation=evaluation,
            num_datapoints=10,
            behavior_name="sycophancy",
            total_iterations=3,
        )

    def test_creation(self, sample_result):
        """Test creating extraction result."""
        assert sample_result.recommended_layer == 15
        assert sample_result.recommended_strength == 1.5
        assert sample_result.num_datapoints == 10
        assert sample_result.behavior_name == "sycophancy"

    def test_save_and_load(self, sample_result):
        """Test saving and loading extraction result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_vector")

            # Save
            sample_result.save(path)

            # Check files exist
            assert os.path.exists(f"{path}.pt")
            assert os.path.exists(f"{path}.json")

            # Load
            loaded = ExtractionResult.load(path)

            # Verify
            assert loaded.recommended_layer == sample_result.recommended_layer
            assert loaded.recommended_strength == sample_result.recommended_strength
            assert loaded.behavior_name == sample_result.behavior_name
            assert loaded.num_datapoints == sample_result.num_datapoints
            assert loaded.evaluation.verdict == sample_result.evaluation.verdict
            assert torch.allclose(loaded.vector, sample_result.vector)

    def test_save_creates_json_metadata(self, sample_result):
        """Test that save creates correct JSON metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_vector")
            sample_result.save(path)

            with open(f"{path}.json") as f:
                meta = json.load(f)

            assert meta["recommended_layer"] == 15
            assert meta["recommended_strength"] == 1.5
            assert meta["behavior_name"] == "sycophancy"
            assert "evaluation_scores" in meta
            assert meta["evaluation_scores"]["behavior_strength"] == 0.8

    def test_default_values(self):
        """Test default values for optional fields."""
        scores = EvaluationScores()
        evaluation = EvaluationResult(
            scores=scores,
            strength_analysis=[],
            recommended_strength=1.0,
        )

        result = ExtractionResult(
            vector=torch.randn(64),
            recommended_layer=10,
            recommended_strength=1.0,
            evaluation=evaluation,
            num_datapoints=5,
        )

        assert result.noise_reduction_applied is False
        assert result.num_seeds_averaged == 1
        assert result.behavior_name == ""
        assert result.total_iterations == 0
        assert result.metadata == {}
