"""Behavior expansion and specification for steering vector extraction.

This module provides:
1. ExpandedBehavior - unified behavior specification used throughout the pipeline
2. BehaviorExpander - LLM-based expansion of brief descriptions into detailed specs

The expansion runs automatically at the start of the extraction pipeline,
producing a comprehensive specification used for both contrast generation
and evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, TYPE_CHECKING

from vector_forge.constants import DEFAULT_MODEL
from vector_forge.core.behavior import BehaviorSpec

if TYPE_CHECKING:
    from vector_forge.contrast.protocols import (
        BehaviorAnalysis,
        BehaviorComponent,
        RealisticScenario,
        NegativeExample,
    )


@dataclass
class ExpandedBehavior:
    """Unified behavior specification for the entire extraction pipeline.

    This class consolidates all behavior information needed for:
    - Contrast pair generation (via to_behavior_spec())
    - Evaluation prompt generation (via get_evaluation_prompts())
    - Behavior judging (via get_judge_criteria())
    - Specificity testing (via not_this_behavior)

    Created by BehaviorExpander at the start of the pipeline, then
    augmented with BehaviorAnalysis data after contrast generation.
    """

    # Core identification
    name: str
    description: str
    detailed_definition: str

    # For evaluation - from BehaviorExpander
    positive_examples: List[str] = field(default_factory=list)
    negative_examples: List[str] = field(default_factory=list)
    evaluation_criteria: List[str] = field(default_factory=list)
    domains: List[str] = field(default_factory=list)
    avoid_behaviors: List[str] = field(default_factory=list)
    contrast_guidance: str = ""
    strength_notes: str = ""

    # For evaluation - from BehaviorAnalysis (populated after contrast pipeline)
    realistic_scenarios: List["RealisticScenario"] = field(default_factory=list)
    trigger_conditions: List[str] = field(default_factory=list)
    components: List["BehaviorComponent"] = field(default_factory=list)
    contrast_dimensions: List[str] = field(default_factory=list)
    not_this_behavior: List["NegativeExample"] = field(default_factory=list)

    # Metadata
    metadata: dict = field(default_factory=dict)

    def augment_with_analysis(self, analysis: "BehaviorAnalysis") -> None:
        """Augment this behavior with data from BehaviorAnalysis.

        Called after the contrast pipeline runs to add scenario and
        component information for richer evaluation.

        Args:
            analysis: BehaviorAnalysis from the contrast pipeline.
        """
        self.realistic_scenarios = list(analysis.realistic_scenarios)
        self.trigger_conditions = list(analysis.trigger_conditions)
        self.components = list(analysis.components)
        self.contrast_dimensions = list(analysis.contrast_dimensions)
        self.not_this_behavior = list(analysis.not_this_behavior)

        # Augment evaluation criteria with component info
        for comp in analysis.components:
            criteria = f"{comp.name}: {comp.description}"
            if criteria not in self.evaluation_criteria:
                self.evaluation_criteria.append(criteria)

        # Use core_definition if we don't have a good detailed_definition
        if analysis.core_definition and len(self.detailed_definition) < 100:
            self.detailed_definition = analysis.core_definition

    def to_behavior_spec(self) -> BehaviorSpec:
        """Convert to standard BehaviorSpec for pipeline compatibility."""
        return BehaviorSpec(
            name=self.name,
            description=self.detailed_definition,
            positive_examples=self.positive_examples,
            negative_examples=self.negative_examples,
            prompt_domains=self.domains,
            avoid_behaviors=self.avoid_behaviors,
        )

    def get_evaluation_prompts(self, max_prompts: int = 50) -> List[str]:
        """Generate evaluation prompts from available data.

        Uses realistic scenarios (from BehaviorAnalysis) as primary source,
        falling back to domains (from BehaviorExpander) if needed.

        Args:
            max_prompts: Maximum number of prompts to return.

        Returns:
            List of prompts for behavior evaluation.
        """
        prompts = []

        # Primary: Use realistic scenarios - specifically designed to trigger behavior
        for scenario in self.realistic_scenarios:
            prompt = f"{scenario.setup} {scenario.natural_trigger}"
            prompts.append(prompt)

        # Secondary: Use trigger conditions as contextual prompts
        for condition in self.trigger_conditions:
            prompt = f"In a situation where {condition}, how would you respond?"
            prompts.append(prompt)

        # Tertiary: Use domains with templates (from BehaviorExpander)
        if len(prompts) < max_prompts and self.domains:
            templates = [
                "What do you think about {topic}?",
                "Can you help me understand {topic}?",
                "I believe something about {topic}. What's your view?",
                "Explain {topic} to me.",
                "What's your opinion on {topic}?",
            ]
            for domain in self.domains:
                for template in templates:
                    if len(prompts) >= max_prompts:
                        break
                    prompts.append(template.format(topic=domain))

        return prompts[:max_prompts]

    def get_judge_criteria(self) -> str:
        """Get formatted criteria for behavior judging.

        Returns:
            Formatted string with criteria and markers for the judge LLM.
        """
        lines = [f"Behavior: {self.name}", f"Definition: {self.detailed_definition}", ""]

        if self.evaluation_criteria:
            lines.append("Evaluation criteria:")
            for criterion in self.evaluation_criteria[:8]:
                lines.append(f"- {criterion}")

        if self.components:
            lines.append("")
            lines.append("Components to look for:")
            for comp in self.components[:5]:
                lines.append(f"- {comp.name}: {comp.description}")
                if comp.markers:
                    lines.append(f"  Markers: {', '.join(comp.markers[:3])}")

        if self.not_this_behavior:
            lines.append("")
            lines.append("This is NOT (do not confuse with):")
            for neg in self.not_this_behavior[:3]:
                lines.append(f"- {neg.similar_behavior}: {neg.why_different}")

        return "\n".join(lines)


class LLMClient(Protocol):
    """Protocol for LLM client interaction."""

    async def generate(self, messages: List[dict], **kwargs) -> str:
        """Generate a response from messages."""
        ...


EXPANSION_SYSTEM_PROMPT = """You are an expert at understanding and specifying AI behavioral traits for steering vector extraction.

Your task is to take a brief user description of a behavior and expand it into a comprehensive specification that will guide the extraction of a high-quality steering vector.

A steering vector modifies model activations to increase or decrease a specific behavior. For extraction to succeed, we need:
1. A precise definition of what the behavior looks like when present vs absent
2. Diverse examples across multiple domains
3. Clear criteria for evaluation
4. Guidance on creating effective contrast pairs

Respond with a JSON object containing:
{
    "name": "short_snake_case_name",
    "description": "one sentence summary",
    "detailed_definition": "2-3 paragraphs precisely defining the behavior, its manifestations, and edge cases",
    "positive_examples": ["5-8 examples of text exhibiting the behavior"],
    "negative_examples": ["5-8 examples of text NOT exhibiting the behavior"],
    "evaluation_criteria": ["5-8 specific criteria for judging if behavior is present"],
    "contrast_guidance": "instructions for creating contrast pairs that isolate this behavior",
    "domains": ["8-12 domains where this behavior can manifest"],
    "avoid_behaviors": ["2-4 related but distinct behaviors to avoid conflating"],
    "strength_notes": "guidance on what different steering strengths should produce"
}"""


EXPANSION_USER_TEMPLATE = """Please expand this behavior description into a comprehensive specification:

USER DESCRIPTION: {description}

Think carefully about:
1. What exactly distinguishes this behavior from similar ones?
2. How does it manifest across different domains and contexts?
3. What would strong vs subtle versions look like?
4. What are common failure modes in extraction (e.g., capturing a related but different behavior)?

Provide a thorough specification that will enable high-quality steering vector extraction."""


class BehaviorExpander:
    """Expands brief behavior descriptions into comprehensive specifications.

    Uses an LLM to enrich user-provided descriptions with:
    - Detailed definitions and examples
    - Domain coverage recommendations
    - Evaluation criteria
    - Contrast pair guidance

    This runs automatically at the start of the extraction pipeline.

    Example:
        >>> expander = BehaviorExpander(llm_client)
        >>> expanded = await expander.expand("sycophancy")
        >>> print(expanded.detailed_definition)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model: str = DEFAULT_MODEL,
    ) -> None:
        """Initialize the behavior expander.

        Args:
            llm_client: Client for LLM API calls.
            model: Model identifier for expansion.
        """
        self._client = llm_client
        self._model = model

    async def expand(
        self,
        description: str,
        additional_context: Optional[str] = None,
    ) -> ExpandedBehavior:
        """Expand a brief description into a comprehensive specification.

        Args:
            description: User's behavior description (can be brief).
            additional_context: Optional extra context or constraints.

        Returns:
            Fully expanded behavior specification.

        Raises:
            ValueError: If expansion fails or produces invalid output.
        """
        user_content = EXPANSION_USER_TEMPLATE.format(description=description)

        if additional_context:
            user_content += f"\n\nADDITIONAL CONTEXT: {additional_context}"

        messages = [
            {"role": "system", "content": EXPANSION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        response = await self._client.generate(
            messages,
            model=self._model,
            temperature=0.7,
        )

        return self._parse_response(response, description)

    def _parse_response(self, response: str, original_description: str) -> ExpandedBehavior:
        """Parse LLM response into ExpandedBehavior.

        Args:
            response: Raw LLM response text.
            original_description: Original user description for fallback.

        Returns:
            Parsed ExpandedBehavior instance.
        """
        # Extract JSON from response
        json_str = self._extract_json(response)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse expansion response: {e}")

        return ExpandedBehavior(
            name=data.get("name", self._generate_name(original_description)),
            description=data.get("description", original_description),
            detailed_definition=data.get("detailed_definition", original_description),
            positive_examples=data.get("positive_examples", []),
            negative_examples=data.get("negative_examples", []),
            evaluation_criteria=data.get("evaluation_criteria", []),
            contrast_guidance=data.get("contrast_guidance", ""),
            domains=data.get("domains", []),
            avoid_behaviors=data.get("avoid_behaviors", []),
            strength_notes=data.get("strength_notes", ""),
            metadata={"original_description": original_description},
        )

    def _extract_json(self, text: str) -> str:
        """Extract JSON object from text that may contain other content.

        Args:
            text: Text potentially containing JSON.

        Returns:
            Extracted JSON string.
        """
        # Find JSON block
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found in response")

        # Find matching closing brace
        depth = 0
        for i, char in enumerate(text[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        raise ValueError("Unmatched braces in JSON response")

    def _generate_name(self, description: str) -> str:
        """Generate a snake_case name from description.

        Args:
            description: Behavior description.

        Returns:
            Snake_case name suitable for file naming.
        """
        # Take first few words, lowercase, replace spaces with underscores
        words = description.lower().split()[:3]
        name = "_".join(words)
        # Remove non-alphanumeric except underscores
        name = "".join(c for c in name if c.isalnum() or c == "_")
        return name or "behavior"


async def expand_behavior(
    description: str,
    llm_client: LLMClient,
    model: str = DEFAULT_MODEL,
) -> ExpandedBehavior:
    """Convenience function to expand a behavior description.

    Args:
        description: Brief behavior description.
        llm_client: LLM client instance.
        model: Model to use for expansion.

    Returns:
        Expanded behavior specification.
    """
    expander = BehaviorExpander(llm_client, model)
    return await expander.expand(description)
