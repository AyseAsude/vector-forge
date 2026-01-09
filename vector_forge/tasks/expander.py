"""Behavior expansion service for enriching user descriptions.

Transforms brief user descriptions into comprehensive behavior specifications
with examples, evaluation criteria, and contrast pair guidance.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Protocol
import json

from vector_forge.core.behavior import BehaviorSpec


@dataclass
class ExpandedBehavior:
    """Enriched behavior specification with detailed guidance.

    Contains all information needed for high-quality extraction:
    - Clear behavior definition
    - Positive and negative examples
    - Evaluation criteria
    - Domain coverage guidance
    """

    name: str
    description: str
    detailed_definition: str
    positive_examples: List[str] = field(default_factory=list)
    negative_examples: List[str] = field(default_factory=list)
    evaluation_criteria: List[str] = field(default_factory=list)
    contrast_guidance: str = ""
    domains: List[str] = field(default_factory=list)
    avoid_behaviors: List[str] = field(default_factory=list)
    strength_notes: str = ""
    metadata: dict = field(default_factory=dict)

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

    This mirrors how Petri uses special_instructions to define audit goals,
    but specialized for steering vector extraction.

    Example:
        >>> expander = BehaviorExpander(llm_client)
        >>> expanded = await expander.expand("sycophancy")
        >>> print(expanded.detailed_definition)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model: str = "gpt-4o",
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
            max_tokens=4096,
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
    model: str = "gpt-4o",
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
