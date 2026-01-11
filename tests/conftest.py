"""Shared test fixtures and configuration."""

import pytest
import torch
from typing import List, Dict, Any
from unittest.mock import MagicMock, AsyncMock

from vector_forge.core.config import (
    PipelineConfig,
    LLMConfig,
    EvaluationBudget,
    DiversityConfig,
    DatapointStrategyType,
    NoiseReductionType,
)
from vector_forge.core.behavior import BehaviorSpec
from vector_forge.core.results import (
    EvaluationScores,
    EvaluationResult,
    Verdict,
    DatapointQuality,
    OptimizationMetrics,
    ExtractionResult,
    Citation,
    StrengthAnalysis,
)
from vector_forge.core.events import Event, EventType
from vector_forge.core.protocols import (
    Message,
    ToolCall,
    ToolDefinition,
    LLMResponse,
    ToolResult,
)
from vector_forge.llm.mock_client import MockLLMClient


# =============================================================================
# Core Fixtures
# =============================================================================


@pytest.fixture
def sample_behavior() -> BehaviorSpec:
    """Create a sample behavior specification for testing."""
    return BehaviorSpec(
        name="sycophancy",
        description="Agreeing with the user even when they are factually incorrect",
        positive_examples=[
            "You're absolutely right!",
            "Yes, exactly! I completely agree.",
        ],
        negative_examples=[
            "Actually, that's not quite correct.",
            "I need to respectfully disagree.",
        ],
        prompt_domains=["science", "math", "history"],
    )


@pytest.fixture
def simple_behavior() -> BehaviorSpec:
    """Create a minimal behavior specification."""
    return BehaviorSpec(description="Simple test behavior")


@pytest.fixture
def llm_config() -> LLMConfig:
    """Create a sample LLM configuration."""
    return LLMConfig(
        model="gpt-4o",
        temperature=0.7,
        max_tokens=2048,
    )


@pytest.fixture
def evaluation_budget_fast() -> EvaluationBudget:
    """Create a fast evaluation budget for testing."""
    return EvaluationBudget.fast()


@pytest.fixture
def evaluation_budget_standard() -> EvaluationBudget:
    """Create a standard evaluation budget."""
    return EvaluationBudget.standard()


@pytest.fixture
def pipeline_config(llm_config: LLMConfig) -> PipelineConfig:
    """Create a sample pipeline configuration."""
    return PipelineConfig(
        extractor_llm=llm_config,
        judge_llm=llm_config,
        num_prompts=5,
        max_outer_iterations=2,
        max_inner_iterations=3,
        evaluation_budget=EvaluationBudget.fast(),
    )


@pytest.fixture
def diversity_config() -> DiversityConfig:
    """Create a sample diversity configuration."""
    return DiversityConfig(
        use_structured_sampling=True,
        domains=["science", "math"],
        formats=["question", "statement"],
        use_mmr_selection=True,
        mmr_lambda=0.5,
    )


# =============================================================================
# Result Fixtures
# =============================================================================


@pytest.fixture
def evaluation_scores() -> EvaluationScores:
    """Create sample evaluation scores."""
    return EvaluationScores(
        behavior_strength=0.8,
        coherence=0.9,
        specificity=0.7,
        robustness=0.75,
        overall=0.8,
    )


@pytest.fixture
def evaluation_result(evaluation_scores: EvaluationScores) -> EvaluationResult:
    """Create a sample evaluation result."""
    return EvaluationResult(
        scores=evaluation_scores,
        strength_analysis=[
            StrengthAnalysis(strength=1.0, behavior_score=0.7, coherence_score=0.9, num_samples=10),
            StrengthAnalysis(strength=1.5, behavior_score=0.85, coherence_score=0.85, num_samples=10),
        ],
        recommended_strength=1.5,
        citations={
            "behavior_strength": [
                Citation(
                    prompt="Test prompt",
                    output="Test output",
                    reason="Showed target behavior",
                    is_success=True,
                )
            ]
        },
        recommendations=["Try increasing diversity"],
        verdict=Verdict.ACCEPTED,
    )


@pytest.fixture
def datapoint_quality() -> DatapointQuality:
    """Create a sample datapoint quality."""
    return DatapointQuality(
        datapoint_id="dp_0",
        leave_one_out_influence=0.1,
        avg_loss_contribution=0.05,
        gradient_alignment=0.8,
        steered_matches_target=True,
        behavior_score_on_own_prompt=0.9,
    )


@pytest.fixture
def extraction_result(evaluation_result: EvaluationResult) -> ExtractionResult:
    """Create a sample extraction result."""
    return ExtractionResult(
        vector=torch.randn(768),
        recommended_layer=15,
        recommended_strength=1.5,
        evaluation=evaluation_result,
        num_datapoints=10,
        behavior_name="sycophancy",
        total_iterations=3,
    )


# =============================================================================
# Protocol/Message Fixtures
# =============================================================================


@pytest.fixture
def sample_messages() -> List[Message]:
    """Create sample chat messages."""
    return [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hello!"),
        Message(role="assistant", content="Hi! How can I help you?"),
    ]


@pytest.fixture
def tool_definition() -> ToolDefinition:
    """Create a sample tool definition."""
    return ToolDefinition(
        name="test_tool",
        description="A test tool",
        parameters={
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Input value"},
            },
            "required": ["input"],
        },
    )


@pytest.fixture
def tool_call() -> ToolCall:
    """Create a sample tool call."""
    return ToolCall(
        id="call_123",
        name="test_tool",
        arguments={"input": "test value"},
    )


@pytest.fixture
def llm_response() -> LLMResponse:
    """Create a sample LLM response."""
    return LLMResponse(
        content="This is a test response.",
        tool_calls=[],
        finish_reason="stop",
        usage={"prompt_tokens": 10, "completion_tokens": 20},
    )


@pytest.fixture
def tool_result_success() -> ToolResult:
    """Create a successful tool result."""
    return ToolResult(success=True, output="Success!")


@pytest.fixture
def tool_result_error() -> ToolResult:
    """Create a failed tool result."""
    return ToolResult(success=False, output=None, error="Something went wrong")


# =============================================================================
# LLM Client Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_client() -> MockLLMClient:
    """Create a mock LLM client with default responses."""
    return MockLLMClient.with_responses([
        "Response 1",
        "Response 2",
        "Response 3",
    ])


@pytest.fixture
def mock_llm_with_tool_calls() -> MockLLMClient:
    """Create a mock LLM client that returns tool calls."""
    def generator(messages: List[Message]) -> LLMResponse:
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="generate_prompts",
                    arguments={"num_prompts": 5},
                )
            ],
            finish_reason="tool_calls",
        )
    return MockLLMClient.with_generator(generator)


# =============================================================================
# Tensor Fixtures
# =============================================================================


@pytest.fixture
def random_vectors() -> List[torch.Tensor]:
    """Create a list of random vectors for noise reduction testing."""
    torch.manual_seed(42)
    return [torch.randn(768) for _ in range(5)]


@pytest.fixture
def similar_vectors() -> List[torch.Tensor]:
    """Create vectors that are similar (with small noise)."""
    torch.manual_seed(42)
    base = torch.randn(768)
    return [base + torch.randn(768) * 0.1 for _ in range(5)]


@pytest.fixture
def small_vectors() -> List[torch.Tensor]:
    """Create small vectors for quick tests."""
    torch.manual_seed(42)
    return [torch.randn(64) for _ in range(3)]


# =============================================================================
# Event Fixtures
# =============================================================================


@pytest.fixture
def sample_event() -> Event:
    """Create a sample event."""
    return Event(
        type=EventType.PIPELINE_STARTED,
        data={"behavior": "sycophancy"},
        source="test",
    )


# =============================================================================
# Mock Backend Fixture
# =============================================================================


@pytest.fixture
def mock_model_backend() -> MagicMock:
    """Create a mock model backend for testing."""
    backend = MagicMock()
    backend.generate = MagicMock(return_value="Generated text")
    backend.generate_with_steering = MagicMock(return_value="Steered text")
    backend.get_num_layers = MagicMock(return_value=32)
    backend.get_hidden_size = MagicMock(return_value=768)
    return backend
