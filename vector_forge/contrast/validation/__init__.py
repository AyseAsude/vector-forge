"""Validation module for contrast pairs."""

from vector_forge.contrast.validation.composer import ValidationComposer, compose_validation_prompt
from vector_forge.contrast.validation.embedding_validator import EmbeddingContrastValidator
from vector_forge.contrast.validation.llm_validator import LLMContrastValidator
from vector_forge.contrast.validation.composite_validator import CompositeContrastValidator
from vector_forge.contrast.validation.signal_validator import (
    BehavioralSignalValidator,
    ConfoundValidator,
    SignalQualityValidator,
)

__all__ = [
    "ValidationComposer",
    "compose_validation_prompt",
    "EmbeddingContrastValidator",
    "LLMContrastValidator",
    "CompositeContrastValidator",
    "BehavioralSignalValidator",
    "ConfoundValidator",
    "SignalQualityValidator",
]
