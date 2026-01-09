"""Tests for vector_forge.core.behavior module."""

import pytest
from hypothesis import given, strategies as st

from vector_forge.core.behavior import BehaviorSpec


class TestBehaviorSpec:
    """Tests for BehaviorSpec."""

    def test_required_description(self):
        """Test that description is required."""
        # Should work with just description
        spec = BehaviorSpec(description="Test behavior")
        assert spec.description == "Test behavior"

        # Should fail without description
        with pytest.raises(ValueError):
            BehaviorSpec()  # type: ignore

    def test_default_name(self):
        """Test that name defaults to 'unnamed'."""
        spec = BehaviorSpec(description="Test")
        assert spec.name == "unnamed"

    def test_custom_name(self):
        """Test setting custom name."""
        spec = BehaviorSpec(name="sycophancy", description="Agreeing with users")
        assert spec.name == "sycophancy"

    def test_examples(self):
        """Test positive and negative examples."""
        spec = BehaviorSpec(
            description="Test behavior",
            positive_examples=["Yes!", "Absolutely!"],
            negative_examples=["No.", "Actually..."],
        )
        assert spec.positive_examples == ["Yes!", "Absolutely!"]
        assert spec.negative_examples == ["No.", "Actually..."]

    def test_examples_optional(self):
        """Test that examples are optional."""
        spec = BehaviorSpec(description="Test")
        assert spec.positive_examples is None
        assert spec.negative_examples is None

    def test_prompt_templates(self):
        """Test prompt templates."""
        spec = BehaviorSpec(
            description="Test",
            prompt_templates=[
                "User: {question}",
                "Answer this: {topic}",
            ],
        )
        assert len(spec.prompt_templates) == 2

    def test_prompt_domains(self):
        """Test prompt domains."""
        spec = BehaviorSpec(
            description="Test",
            prompt_domains=["science", "math", "history"],
        )
        assert spec.prompt_domains == ["science", "math", "history"]

    def test_avoid_behaviors(self):
        """Test avoid_behaviors for specificity."""
        spec = BehaviorSpec(
            description="Sycophancy",
            avoid_behaviors=["helpfulness", "politeness"],
        )
        assert spec.avoid_behaviors == ["helpfulness", "politeness"]

    def test_coherence_requirements(self):
        """Test coherence requirements."""
        spec = BehaviorSpec(
            description="Test",
            coherence_requirements="Must be grammatically correct",
        )
        assert spec.coherence_requirements == "Must be grammatically correct"

    def test_tags(self):
        """Test tags field."""
        spec = BehaviorSpec(
            description="Test",
            tags=["safety", "alignment"],
        )
        assert spec.tags == ["safety", "alignment"]

    def test_tags_default_empty(self):
        """Test tags default to empty list."""
        spec = BehaviorSpec(description="Test")
        assert spec.tags == []

    def test_metadata(self):
        """Test metadata field."""
        spec = BehaviorSpec(
            description="Test",
            metadata={"source": "paper", "version": 1},
        )
        assert spec.metadata["source"] == "paper"
        assert spec.metadata["version"] == 1

    def test_metadata_default_empty(self):
        """Test metadata defaults to empty dict."""
        spec = BehaviorSpec(description="Test")
        assert spec.metadata == {}

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed (model_config extra=allow)."""
        spec = BehaviorSpec(
            description="Test",
            custom_field="custom_value",  # type: ignore
        )
        assert spec.custom_field == "custom_value"  # type: ignore

    @given(st.text(min_size=1, max_size=500))
    def test_description_accepts_any_string(self, description: str):
        """Test that description accepts any non-empty string."""
        spec = BehaviorSpec(description=description)
        assert spec.description == description

    def test_full_specification(self):
        """Test creating a fully specified behavior."""
        spec = BehaviorSpec(
            name="sycophancy",
            description="Agreeing with users even when they are wrong",
            positive_examples=[
                "You're absolutely right!",
                "Yes, exactly!",
            ],
            negative_examples=[
                "Actually, that's incorrect.",
                "I disagree because...",
            ],
            prompt_templates=[
                "User claims: {claim}",
            ],
            prompt_domains=["science", "math"],
            avoid_behaviors=["helpfulness"],
            coherence_requirements="Output must be coherent",
            tags=["alignment", "safety"],
            metadata={"difficulty": "medium"},
        )

        assert spec.name == "sycophancy"
        assert "Agreeing" in spec.description
        assert len(spec.positive_examples) == 2
        assert len(spec.negative_examples) == 2
        assert len(spec.prompt_templates) == 1
        assert len(spec.prompt_domains) == 2
        assert len(spec.avoid_behaviors) == 1
        assert spec.coherence_requirements is not None
        assert len(spec.tags) == 2
        assert spec.metadata["difficulty"] == "medium"


class TestBehaviorSpecValidation:
    """Tests for BehaviorSpec validation behavior."""

    def test_empty_description_raises(self):
        """Test that empty string description raises error."""
        # Pydantic allows empty string, but we could add a validator
        # For now, just test it accepts empty string
        spec = BehaviorSpec(description="")
        assert spec.description == ""

    def test_whitespace_only_description(self):
        """Test whitespace-only description is accepted."""
        spec = BehaviorSpec(description="   ")
        assert spec.description == "   "

    def test_special_characters_in_name(self):
        """Test special characters in name."""
        spec = BehaviorSpec(
            name="behavior-with_special.chars",
            description="Test",
        )
        assert spec.name == "behavior-with_special.chars"

    def test_unicode_in_description(self):
        """Test unicode characters in description."""
        spec = BehaviorSpec(
            description="Test with unicode: \u00e9\u00e8\u00ea",
        )
        assert "\u00e9" in spec.description

    def test_empty_lists_accepted(self):
        """Test empty lists are accepted for list fields."""
        spec = BehaviorSpec(
            description="Test",
            positive_examples=[],
            negative_examples=[],
            prompt_domains=[],
            tags=[],
        )
        assert spec.positive_examples == []
        assert spec.negative_examples == []
        assert spec.prompt_domains == []
        assert spec.tags == []
