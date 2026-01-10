"""Tests for vector_forge.core.config module."""

import pytest
from hypothesis import given, strategies as st

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


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = LLMConfig()
        assert config.model == DEFAULT_MODEL
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.api_base is None
        assert config.api_key is None
        assert config.extra_params == {}

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = LLMConfig(
            model="claude-3-opus-20240229",
            temperature=0.5,
            max_tokens=2048,
            api_base="https://custom.api.com",
        )
        assert config.model == "claude-3-opus-20240229"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048
        assert config.api_base == "https://custom.api.com"

    def test_temperature_bounds(self):
        """Test temperature validation bounds."""
        # Valid temperatures
        LLMConfig(temperature=0)
        LLMConfig(temperature=1.0)
        LLMConfig(temperature=2.0)

        # Invalid temperatures
        with pytest.raises(ValueError):
            LLMConfig(temperature=-0.1)
        with pytest.raises(ValueError):
            LLMConfig(temperature=2.1)

    def test_max_tokens_positive(self):
        """Test max_tokens must be positive."""
        with pytest.raises(ValueError):
            LLMConfig(max_tokens=0)
        with pytest.raises(ValueError):
            LLMConfig(max_tokens=-1)

    @given(st.text(min_size=1, max_size=50))
    def test_model_accepts_any_string(self, model: str):
        """Test that model accepts any non-empty string."""
        config = LLMConfig(model=model)
        assert config.model == model


class TestEvaluationBudget:
    """Tests for EvaluationBudget."""

    def test_fast_preset(self):
        """Test fast evaluation budget preset."""
        budget = EvaluationBudget.fast()
        assert budget.quick_eval_prompts == 3
        assert budget.quick_eval_generations_per_prompt == 2
        assert budget.thorough_eval_prompts == 10
        assert len(budget.quick_eval_strength_levels) == 1

    def test_standard_preset(self):
        """Test standard evaluation budget preset."""
        budget = EvaluationBudget.standard()
        assert budget.quick_eval_prompts == 5
        assert budget.thorough_eval_prompts == 30

    def test_thorough_preset(self):
        """Test thorough evaluation budget preset."""
        budget = EvaluationBudget.thorough()
        assert budget.quick_eval_prompts == 10
        assert budget.thorough_eval_prompts == 50
        assert len(budget.thorough_eval_strength_levels) == 7

    def test_custom_budget(self):
        """Test creating custom evaluation budget."""
        budget = EvaluationBudget(
            quick_eval_prompts=7,
            thorough_eval_prompts=40,
        )
        assert budget.quick_eval_prompts == 7
        assert budget.thorough_eval_prompts == 40

    def test_quick_eval_prompts_minimum(self):
        """Test quick_eval_prompts must be at least 1."""
        with pytest.raises(ValueError):
            EvaluationBudget(quick_eval_prompts=0)

    def test_thorough_eval_prompts_minimum(self):
        """Test thorough_eval_prompts must be at least 5."""
        with pytest.raises(ValueError):
            EvaluationBudget(thorough_eval_prompts=4)


class TestDiversityConfig:
    """Tests for DiversityConfig."""

    def test_default_values(self):
        """Test default diversity configuration."""
        config = DiversityConfig()
        assert config.use_structured_sampling is True
        assert config.use_mmr_selection is True
        assert config.mmr_lambda == 0.5
        assert len(config.domains) > 0
        assert len(config.formats) > 0

    def test_mmr_lambda_bounds(self):
        """Test mmr_lambda must be between 0 and 1."""
        DiversityConfig(mmr_lambda=0)
        DiversityConfig(mmr_lambda=1.0)

        with pytest.raises(ValueError):
            DiversityConfig(mmr_lambda=-0.1)
        with pytest.raises(ValueError):
            DiversityConfig(mmr_lambda=1.1)

    def test_candidate_multiplier_minimum(self):
        """Test candidate_multiplier must be at least 1."""
        with pytest.raises(ValueError):
            DiversityConfig(candidate_multiplier=0)

    def test_thresholds_bounds(self):
        """Test similarity thresholds are bounded correctly."""
        DiversityConfig(min_embedding_distance=0)
        DiversityConfig(min_embedding_distance=1.0)
        DiversityConfig(max_avg_similarity=0)
        DiversityConfig(max_avg_similarity=1.0)


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_values(self):
        """Test default pipeline configuration."""
        config = PipelineConfig()
        assert config.datapoint_strategy == DatapointStrategyType.CONTRASTIVE
        assert config.num_prompts == 10
        assert config.max_outer_iterations == 3
        assert config.max_inner_iterations == 5
        assert config.quality_threshold == 0.7
        assert config.noise_reduction == NoiseReductionType.AVERAGING

    def test_custom_llm_configs(self):
        """Test setting custom LLM configurations."""
        extractor_llm = LLMConfig(model="gpt-4", temperature=0.8)
        judge_llm = LLMConfig(model="claude-3-opus", temperature=0.3)

        config = PipelineConfig(
            extractor_llm=extractor_llm,
            judge_llm=judge_llm,
        )

        assert config.extractor_llm.model == "gpt-4"
        assert config.extractor_llm.temperature == 0.8
        assert config.judge_llm.model == "claude-3-opus"
        assert config.judge_llm.temperature == 0.3

    def test_num_prompts_minimum(self):
        """Test num_prompts must be at least 1."""
        with pytest.raises(ValueError):
            PipelineConfig(num_prompts=0)

    def test_quality_threshold_bounds(self):
        """Test quality_threshold must be between 0 and 1."""
        PipelineConfig(quality_threshold=0)
        PipelineConfig(quality_threshold=1.0)

        with pytest.raises(ValueError):
            PipelineConfig(quality_threshold=-0.1)
        with pytest.raises(ValueError):
            PipelineConfig(quality_threshold=1.1)

    def test_iteration_limits_positive(self):
        """Test iteration limits must be positive."""
        with pytest.raises(ValueError):
            PipelineConfig(max_outer_iterations=0)
        with pytest.raises(ValueError):
            PipelineConfig(max_inner_iterations=0)

    def test_layers_to_try_optional(self):
        """Test layers_to_try is optional."""
        config1 = PipelineConfig()
        assert config1.layers_to_try is None

        config2 = PipelineConfig(layers_to_try=[10, 15, 20])
        assert config2.layers_to_try == [10, 15, 20]


class TestEnums:
    """Tests for configuration enums."""

    def test_datapoint_strategy_values(self):
        """Test DatapointStrategyType enum values."""
        assert DatapointStrategyType.CONTRASTIVE.value == "contrastive"
        assert DatapointStrategyType.SINGLE_DIRECTION.value == "single_direction"
        assert DatapointStrategyType.ACTIVATION_DIFF.value == "activation_diff"
        assert DatapointStrategyType.LLM_GENERATED.value == "llm_generated"

    def test_noise_reduction_values(self):
        """Test NoiseReductionType enum values."""
        assert NoiseReductionType.NONE.value == "none"
        assert NoiseReductionType.AVERAGING.value == "averaging"
        assert NoiseReductionType.PCA.value == "pca"
        assert NoiseReductionType.ADVERSARIAL.value == "adversarial"

    def test_evaluation_dimension_values(self):
        """Test EvaluationDimension enum values."""
        assert EvaluationDimension.BEHAVIOR_STRENGTH.value == "behavior_strength"
        assert EvaluationDimension.COHERENCE.value == "coherence"
        assert EvaluationDimension.SPECIFICITY.value == "specificity"
        assert EvaluationDimension.ROBUSTNESS.value == "robustness"
