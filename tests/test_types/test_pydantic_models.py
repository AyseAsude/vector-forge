"""Tests for Pydantic model type validation.

Verifies that Pydantic models in vector_forge correctly:
- Validate field types
- Apply constraints (ge, le, gt, lt)
- Handle optional fields
- Coerce compatible types
- Reject invalid types
"""

import pytest
from typing import Any, Dict, List
from pydantic import ValidationError

from vector_forge.constants import DEFAULT_MODEL
from vector_forge.core.config import (
    LLMConfig,
    EvaluationBudget,
    DiversityConfig,
    PipelineConfig,
    DatapointStrategyType,
    NoiseReductionType,
    EvaluationDimension,
)
from vector_forge.core.behavior import BehaviorSpec
from vector_forge.core.results import (
    EvaluationResult,
    DatapointQuality,
    OptimizationMetrics,
)
from vector_forge.tasks.config import (
    OptimizationConfig,
    ContrastConfig,
    SampleConfig,
    EvaluationConfig,
    TaskConfig,
    LayerStrategy,
    AggregationStrategy,
    ContrastQuality,
)


# =============================================================================
# LLMConfig Type Tests
# =============================================================================


class TestLLMConfigTypes:
    """Type validation tests for LLMConfig."""

    def test_model_accepts_string(self):
        """Test model field accepts strings."""
        config = LLMConfig(model="gpt-4")
        assert config.model == "gpt-4"

    def test_model_rejects_non_string(self):
        """Test model field rejects non-strings."""
        with pytest.raises(ValidationError):
            LLMConfig(model=123)  # type: ignore

    def test_temperature_accepts_int_coerces_to_float(self):
        """Test temperature coerces int to float."""
        config = LLMConfig(temperature=1)
        assert config.temperature == 1.0
        assert isinstance(config.temperature, float)

    def test_temperature_constraint_ge_0(self):
        """Test temperature must be >= 0."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(temperature=-0.1)
        assert "temperature" in str(exc_info.value)

    def test_temperature_constraint_le_2(self):
        """Test temperature must be <= 2."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(temperature=2.1)
        assert "temperature" in str(exc_info.value)

    def test_max_tokens_accepts_int(self):
        """Test max_tokens accepts integers."""
        config = LLMConfig(max_tokens=1000)
        assert config.max_tokens == 1000

    def test_max_tokens_accepts_none(self):
        """Test max_tokens accepts None (provider default)."""
        config = LLMConfig(max_tokens=None)
        assert config.max_tokens is None

    def test_max_tokens_constraint_gt_0(self):
        """Test max_tokens must be > 0 when specified."""
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=0)
        with pytest.raises(ValidationError):
            LLMConfig(max_tokens=-1)

    def test_api_base_accepts_none(self):
        """Test api_base accepts None."""
        config = LLMConfig(api_base=None)
        assert config.api_base is None

    def test_api_base_accepts_string(self):
        """Test api_base accepts strings."""
        config = LLMConfig(api_base="https://api.example.com")
        assert config.api_base == "https://api.example.com"

    def test_extra_params_accepts_dict(self):
        """Test extra_params accepts dicts."""
        config = LLMConfig(extra_params={"top_p": 0.9})
        assert config.extra_params["top_p"] == 0.9

    def test_extra_params_default_empty_dict(self):
        """Test extra_params defaults to empty dict."""
        config = LLMConfig()
        assert config.extra_params == {}


# =============================================================================
# BehaviorSpec Type Tests
# =============================================================================


class TestBehaviorSpecTypes:
    """Type validation tests for BehaviorSpec."""

    def test_description_required(self):
        """Test description is required."""
        with pytest.raises(ValidationError):
            BehaviorSpec()  # type: ignore

    def test_description_accepts_string(self):
        """Test description accepts strings."""
        spec = BehaviorSpec(description="Test behavior")
        assert spec.description == "Test behavior"

    def test_description_accepts_empty(self):
        """Test description accepts empty string (no min_length constraint)."""
        # Note: BehaviorSpec doesn't enforce min_length on description
        spec = BehaviorSpec(description="")
        assert spec.description == ""

    def test_name_defaults_to_unnamed(self):
        """Test name defaults to 'unnamed'."""
        spec = BehaviorSpec(description="Test")
        assert spec.name == "unnamed"

    def test_positive_examples_accepts_list_of_strings(self):
        """Test positive_examples accepts list of strings."""
        spec = BehaviorSpec(
            description="Test",
            positive_examples=["example 1", "example 2"]
        )
        assert len(spec.positive_examples) == 2

    def test_positive_examples_accepts_none(self):
        """Test positive_examples accepts None."""
        spec = BehaviorSpec(description="Test", positive_examples=None)
        assert spec.positive_examples is None

    def test_tags_defaults_to_empty_list(self):
        """Test tags defaults to empty list."""
        spec = BehaviorSpec(description="Test")
        assert spec.tags == []

    def test_tags_accepts_list_of_strings(self):
        """Test tags accepts list of strings."""
        spec = BehaviorSpec(description="Test", tags=["tag1", "tag2"])
        assert "tag1" in spec.tags

    def test_metadata_defaults_to_empty_dict(self):
        """Test metadata defaults to empty dict."""
        spec = BehaviorSpec(description="Test")
        assert spec.metadata == {}

    def test_extra_fields_allowed(self):
        """Test extra fields are allowed."""
        spec = BehaviorSpec(description="Test", custom_field="value")
        assert spec.custom_field == "value"  # type: ignore


# =============================================================================
# OptimizationConfig Type Tests (tasks/config.py)
# =============================================================================


class TestOptimizationConfigTypes:
    """Type validation tests for OptimizationConfig."""

    def test_lr_constraint_gt_0(self):
        """Test lr must be > 0."""
        with pytest.raises(ValidationError):
            OptimizationConfig(lr=0)
        with pytest.raises(ValidationError):
            OptimizationConfig(lr=-0.1)

    def test_lr_constraint_le_1(self):
        """Test lr must be <= 1."""
        with pytest.raises(ValidationError):
            OptimizationConfig(lr=1.1)

    def test_max_iters_constraint_gt_0(self):
        """Test max_iters must be > 0."""
        with pytest.raises(ValidationError):
            OptimizationConfig(max_iters=0)

    def test_max_iters_constraint_le_500(self):
        """Test max_iters must be <= 500."""
        with pytest.raises(ValidationError):
            OptimizationConfig(max_iters=501)

    def test_coldness_constraint_gt_0(self):
        """Test coldness must be > 0."""
        with pytest.raises(ValidationError):
            OptimizationConfig(coldness=0)

    def test_coldness_constraint_le_2(self):
        """Test coldness must be <= 2."""
        with pytest.raises(ValidationError):
            OptimizationConfig(coldness=2.1)

    def test_starting_norm_constraint_gt_0(self):
        """Test starting_norm must be > 0."""
        with pytest.raises(ValidationError):
            OptimizationConfig(starting_norm=0)

    def test_max_norm_accepts_none(self):
        """Test max_norm accepts None."""
        config = OptimizationConfig(max_norm=None)
        assert config.max_norm is None

    def test_target_loss_accepts_none(self):
        """Test target_loss accepts None."""
        config = OptimizationConfig(target_loss=None)
        assert config.target_loss is None

    def test_convergence_eps_constraint_gt_0(self):
        """Test convergence_eps must be > 0."""
        with pytest.raises(ValidationError):
            OptimizationConfig(convergence_eps=0)

    def test_convergence_patience_constraint_ge_1(self):
        """Test convergence_patience must be >= 1."""
        with pytest.raises(ValidationError):
            OptimizationConfig(convergence_patience=0)

    def test_preset_fast(self):
        """Test fast preset creates valid config."""
        config = OptimizationConfig.fast()
        assert config.lr == 0.15
        assert config.max_iters == 30

    def test_preset_standard(self):
        """Test standard preset creates valid config."""
        config = OptimizationConfig.standard()
        assert config.lr == 0.1
        assert config.max_iters == 50

    def test_preset_thorough(self):
        """Test thorough preset creates valid config."""
        config = OptimizationConfig.thorough()
        assert config.lr == 0.08
        assert config.max_iters == 100

    def test_to_steering_config_returns_dict(self):
        """Test to_steering_config returns proper dict."""
        config = OptimizationConfig()
        result = config.to_steering_config()

        assert isinstance(result, dict)
        assert "lr" in result
        assert "max_iters" in result
        assert "coldness" in result


# =============================================================================
# ContrastConfig Type Tests
# =============================================================================


class TestContrastConfigTypes:
    """Type validation tests for ContrastConfig."""

    def test_core_pool_size_constraint_ge_20(self):
        """Test core_pool_size must be >= 20."""
        with pytest.raises(ValidationError):
            ContrastConfig(core_pool_size=19)

    def test_core_pool_size_constraint_le_500(self):
        """Test core_pool_size must be <= 500."""
        with pytest.raises(ValidationError):
            ContrastConfig(core_pool_size=501)

    def test_min_semantic_distance_constraint_ge_01(self):
        """Test min_semantic_distance must be >= 0.1."""
        with pytest.raises(ValidationError):
            ContrastConfig(min_semantic_distance=0.05)

    def test_min_semantic_distance_constraint_le_09(self):
        """Test min_semantic_distance must be <= 0.9."""
        with pytest.raises(ValidationError):
            ContrastConfig(min_semantic_distance=0.95)

    def test_min_dst_score_constraint_ge_1(self):
        """Test min_dst_score must be >= 1."""
        with pytest.raises(ValidationError):
            ContrastConfig(min_dst_score=0.5)

    def test_max_src_score_constraint_le_9(self):
        """Test max_src_score must be <= 9."""
        with pytest.raises(ValidationError):
            ContrastConfig(max_src_score=9.5)

    def test_pairs_per_sample_property(self):
        """Test pairs_per_sample computed property."""
        config = ContrastConfig(
            core_seeds_per_sample=30,
            unique_seeds_per_sample=10
        )
        assert config.pairs_per_sample == 40

    def test_from_preset_fast(self):
        """Test from_preset with FAST."""
        config = ContrastConfig.from_preset(ContrastQuality.FAST)
        assert config.core_pool_size == 30

    def test_from_preset_thorough(self):
        """Test from_preset with THOROUGH."""
        config = ContrastConfig.from_preset(ContrastQuality.THOROUGH)
        assert config.core_pool_size == 120


# =============================================================================
# EvaluationConfig Type Tests
# =============================================================================


class TestEvaluationConfigTypes:
    """Type validation tests for EvaluationConfig."""

    def test_behavior_prompts_constraint_ge_10(self):
        """Test behavior_prompts must be >= 10."""
        with pytest.raises(ValidationError):
            EvaluationConfig(behavior_prompts=9)

    def test_strength_levels_validator_minimum_2(self):
        """Test strength_levels must have at least 2 elements."""
        with pytest.raises(ValidationError):
            EvaluationConfig(strength_levels=[1.0])

    def test_strength_levels_validator_sorted(self):
        """Test strength_levels are sorted."""
        config = EvaluationConfig(strength_levels=[2.0, 1.0, 1.5])
        assert config.strength_levels == [1.0, 1.5, 2.0]

    def test_weight_fields_bounds(self):
        """Test weight fields are bounded 0-1."""
        # Valid weights
        config = EvaluationConfig(behavior_weight=0.5)
        assert config.behavior_weight == 0.5

        # Invalid weights
        with pytest.raises(ValidationError):
            EvaluationConfig(behavior_weight=-0.1)
        with pytest.raises(ValidationError):
            EvaluationConfig(behavior_weight=1.1)

    def test_total_weight_property(self):
        """Test total_weight computed property."""
        config = EvaluationConfig()
        # Default weights should sum to 1.0
        assert abs(config.total_weight - 1.0) < 0.001


# =============================================================================
# TaskConfig Type Tests
# =============================================================================


class TestTaskConfigTypes:
    """Type validation tests for TaskConfig."""

    def test_num_samples_constraint_ge_1(self):
        """Test num_samples must be >= 1."""
        with pytest.raises(ValidationError):
            TaskConfig(num_samples=0)

    def test_num_samples_constraint_le_100(self):
        """Test num_samples must be <= 100."""
        with pytest.raises(ValidationError):
            TaskConfig(num_samples=101)

    def test_layer_strategies_accepts_list(self):
        """Test layer_strategies accepts list of LayerStrategy."""
        config = TaskConfig(layer_strategies=[LayerStrategy.AUTO, LayerStrategy.SWEEP])
        assert LayerStrategy.AUTO in config.layer_strategies

    def test_aggregation_strategy_accepts_enum(self):
        """Test aggregation_strategy accepts AggregationStrategy enum."""
        config = TaskConfig(aggregation_strategy=AggregationStrategy.PCA_PRINCIPAL)
        assert config.aggregation_strategy == AggregationStrategy.PCA_PRINCIPAL

    def test_top_k_validator_clamped_to_num_samples(self):
        """Test top_k is clamped to num_samples."""
        config = TaskConfig(num_samples=5, top_k=10)
        assert config.top_k == 5

    def test_nested_configs_validated(self):
        """Test nested configs are validated."""
        # Invalid nested config should raise
        with pytest.raises(ValidationError):
            TaskConfig(
                optimization=OptimizationConfig(lr=2.0)  # Invalid: > 1.0
            )

    def test_preset_quick(self):
        """Test quick preset creates valid config."""
        config = TaskConfig.quick()
        assert config.num_samples == 4
        assert config.num_seeds == 2

    def test_preset_comprehensive(self):
        """Test comprehensive preset creates valid config."""
        config = TaskConfig.comprehensive()
        assert config.num_samples == 32
        assert config.num_seeds == 8


# =============================================================================
# SampleConfig Type Tests
# =============================================================================


class TestSampleConfigTypes:
    """Type validation tests for SampleConfig."""

    def test_seed_constraint_ge_0(self):
        """Test seed must be >= 0."""
        with pytest.raises(ValidationError):
            SampleConfig(seed=-1)

    def test_layer_strategy_accepts_enum(self):
        """Test layer_strategy accepts LayerStrategy enum."""
        config = SampleConfig(layer_strategy=LayerStrategy.FIXED)
        assert config.layer_strategy == LayerStrategy.FIXED

    def test_target_layers_accepts_list_or_none(self):
        """Test target_layers accepts list or None."""
        config1 = SampleConfig(target_layers=None)
        assert config1.target_layers is None

        config2 = SampleConfig(target_layers=[10, 15, 20])
        assert config2.target_layers == [10, 15, 20]

    def test_bootstrap_ratio_constraint_ge_05(self):
        """Test bootstrap_ratio must be >= 0.5."""
        with pytest.raises(ValidationError):
            SampleConfig(bootstrap_ratio=0.4)

    def test_bootstrap_ratio_constraint_le_1(self):
        """Test bootstrap_ratio must be <= 1.0."""
        with pytest.raises(ValidationError):
            SampleConfig(bootstrap_ratio=1.1)

    def test_get_lr_with_override(self):
        """Test get_lr returns override if set."""
        config = SampleConfig(lr=0.2)
        assert config.get_lr(default=0.1) == 0.2

    def test_get_lr_without_override(self):
        """Test get_lr returns default if not set."""
        config = SampleConfig()
        assert config.get_lr(default=0.1) == 0.1


# =============================================================================
# Enum Type Tests
# =============================================================================


class TestEnumTypes:
    """Tests for enum types."""

    def test_layer_strategy_values(self):
        """Test LayerStrategy enum values."""
        assert LayerStrategy.AUTO.value == "auto"
        assert LayerStrategy.FIXED.value == "fixed"
        assert LayerStrategy.SWEEP.value == "sweep"
        assert LayerStrategy.MIDDLE.value == "middle"
        assert LayerStrategy.LATE.value == "late"

    def test_aggregation_strategy_values(self):
        """Test AggregationStrategy enum values."""
        assert AggregationStrategy.BEST_SINGLE.value == "best_single"
        assert AggregationStrategy.TOP_K_AVERAGE.value == "top_k_average"
        assert AggregationStrategy.WEIGHTED_AVERAGE.value == "weighted_average"
        assert AggregationStrategy.PCA_PRINCIPAL.value == "pca_principal"
        assert AggregationStrategy.STRATEGY_GROUPED.value == "strategy_grouped"

    def test_contrast_quality_values(self):
        """Test ContrastQuality enum values."""
        assert ContrastQuality.FAST.value == "fast"
        assert ContrastQuality.STANDARD.value == "standard"
        assert ContrastQuality.THOROUGH.value == "thorough"

    def test_datapoint_strategy_type_values(self):
        """Test DatapointStrategyType enum values."""
        assert DatapointStrategyType.CONTRASTIVE.value == "contrastive"
        assert DatapointStrategyType.SINGLE_DIRECTION.value == "single_direction"

    def test_noise_reduction_type_values(self):
        """Test NoiseReductionType enum values."""
        assert NoiseReductionType.NONE.value == "none"
        assert NoiseReductionType.AVERAGING.value == "averaging"
        assert NoiseReductionType.PCA.value == "pca"

    def test_evaluation_dimension_values(self):
        """Test EvaluationDimension enum values."""
        assert EvaluationDimension.BEHAVIOR_STRENGTH.value == "behavior_strength"
        assert EvaluationDimension.COHERENCE.value == "coherence"
        assert EvaluationDimension.SPECIFICITY.value == "specificity"


# =============================================================================
# Result Type Tests
# =============================================================================


class TestResultTypes:
    """Tests for result dataclass types."""

    def test_datapoint_quality_defaults(self):
        """Test DatapointQuality default values."""
        quality = DatapointQuality(datapoint_id="dp_1")

        assert quality.datapoint_id == "dp_1"
        assert quality.leave_one_out_influence is None  # Optional, defaults to None
        assert quality.gradient_alignment == 0.0
        assert quality.avg_loss_contribution == 0.0
        assert quality.steered_matches_target is True
        assert quality.is_outlier is False

    def test_datapoint_quality_all_fields(self):
        """Test DatapointQuality with all fields."""
        quality = DatapointQuality(
            datapoint_id="dp_1",
            leave_one_out_influence=0.05,
            gradient_alignment=0.8,
            avg_loss_contribution=0.3,
            steered_matches_target=True,
            is_outlier=False,
            distance_to_centroid=0.1,
            cluster_id=0
        )

        assert quality.gradient_alignment == 0.8
        assert quality.steered_matches_target is True
        assert quality.cluster_id == 0

    def test_optimization_metrics_defaults(self):
        """Test OptimizationMetrics default values."""
        metrics = OptimizationMetrics(
            layer=16,
            final_loss=0.1,
            iterations=50,
            vector_norm=1.0
        )

        assert metrics.layer == 16
        assert metrics.iterations == 50
        assert metrics.final_loss == 0.1
        assert metrics.vector_norm == 1.0
        assert metrics.datapoint_qualities == []


# =============================================================================
# Type Coercion Tests
# =============================================================================


class TestTypeCoercion:
    """Tests for Pydantic type coercion behavior."""

    def test_int_to_float_coercion(self):
        """Test integers are coerced to floats where appropriate."""
        config = LLMConfig(temperature=1)
        assert isinstance(config.temperature, float)

    def test_string_to_path_coercion(self):
        """Test strings are NOT coerced to paths in extra_params."""
        config = LLMConfig(extra_params={"path": "/some/path"})
        assert isinstance(config.extra_params["path"], str)

    def test_list_validation(self):
        """Test list fields validate element types."""
        # Valid list
        spec = BehaviorSpec(
            description="Test",
            positive_examples=["a", "b", "c"]
        )
        assert len(spec.positive_examples) == 3

    def test_dict_validation(self):
        """Test dict fields validate properly."""
        spec = BehaviorSpec(
            description="Test",
            metadata={"key": "value", "num": 42}
        )
        assert spec.metadata["num"] == 42
