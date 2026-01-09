"""Behavior specification for steering vector extraction."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class BehaviorSpec(BaseModel):
    """
    Specification of the behavior to extract as a steering vector.

    The description is the primary input - a natural language description
    of the behavior. Optional fields provide additional guidance for
    datapoint generation and evaluation.

    Example:
        >>> behavior = BehaviorSpec(
        ...     name="sycophancy",
        ...     description="Agreeing with the user even when they are factually wrong",
        ...     positive_examples=["You're absolutely right!", "I completely agree!"],
        ...     negative_examples=["Actually, that's not correct.", "I disagree because..."],
        ...     prompt_domains=["science", "math", "history"],
        ... )
    """

    description: str = Field(
        ...,
        description="Natural language description of the behavior to extract",
    )

    name: str = Field(
        default="unnamed",
        description="Short identifier for this behavior",
    )

    # Guidance for datapoint generation
    positive_examples: Optional[List[str]] = Field(
        default=None,
        description="Example outputs that exhibit the behavior",
    )

    negative_examples: Optional[List[str]] = Field(
        default=None,
        description="Example outputs that do NOT exhibit the behavior",
    )

    prompt_templates: Optional[List[str]] = Field(
        default=None,
        description="Prompt templates to use for training data generation",
    )

    prompt_domains: Optional[List[str]] = Field(
        default=None,
        description="Domains to generate prompts from (e.g., 'science', 'politics')",
    )

    # Constraints for specificity
    avoid_behaviors: Optional[List[str]] = Field(
        default=None,
        description="Behaviors the vector should NOT induce (for specificity testing)",
    )

    coherence_requirements: Optional[str] = Field(
        default=None,
        description="Specific requirements for output coherence",
    )

    # Metadata
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}
