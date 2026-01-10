"""Behavior analyzer for extracting structured behavior information.

This module analyzes behavior descriptions to extract components,
triggers, contrast dimensions, and confounds that guide the
contrast pair generation process.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from vector_forge.contrast.protocols import (
    BehaviorAnalyzerProtocol,
    BehaviorAnalysis,
    BehaviorComponent,
)
from vector_forge.core.protocols import Message
from vector_forge.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


ANALYSIS_PROMPT = '''Analyze this behavior for steering vector training data generation.

BEHAVIOR: {behavior_description}

Provide a comprehensive analysis to help generate high-quality contrast pairs.

## 1. COMPONENTS
Break this behavior into 4-6 distinct, measurable components.
For each component:
- Identify specific linguistic markers (phrases, patterns, words)
- Identify what the OPPOSITE or ABSENCE looks like

## 2. TRIGGER CONDITIONS
What situations naturally trigger or elicit this behavior?
When would an AI most likely exhibit or suppress this behavior?
Focus on scenarios where the behavior is most relevant.

## 3. CONTRAST DIMENSIONS
What are the clearest ways to show presence vs absence of this behavior?
How can we make the difference between "exhibits" and "does not exhibit" OBVIOUS?

## 4. CONFOUNDS TO AVOID
What other behaviors or factors might correlate with this behavior?
What should we control for to isolate THIS specific behavior?
(e.g., response length, helpfulness, politeness, topic complexity)

Return a JSON object with this structure:
{{
  "components": [
    {{
      "name": "short_name",
      "description": "what this component represents",
      "markers": ["phrase1", "phrase2", "pattern1"],
      "opposite_markers": ["opposite_phrase1", "neutral_phrase1"]
    }}
  ],
  "trigger_conditions": [
    "condition or scenario that triggers the behavior"
  ],
  "contrast_dimensions": [
    "dimension along which contrast is clearest"
  ],
  "confounds_to_avoid": [
    "factor that might correlate but isn't the behavior"
  ]
}}

Be specific and actionable. The output will be used to generate training data.'''


class BehaviorAnalyzer(BehaviorAnalyzerProtocol):
    """Analyzes behavior descriptions using LLM.

    Extracts structured information about behaviors including:
    - Components: Distinct aspects of the behavior
    - Trigger conditions: When the behavior manifests
    - Contrast dimensions: How to show presence vs absence
    - Confounds: Factors to control for

    Example:
        >>> analyzer = BehaviorAnalyzer(llm_client)
        >>> analysis = await analyzer.analyze("sycophancy")
        >>> print(analysis.components[0].name)
        "excessive_agreement"
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        """Initialize the behavior analyzer.

        Args:
            llm_client: LLM client for analysis.
            temperature: Generation temperature.
            max_tokens: Maximum tokens for response (None = provider default).
        """
        self._llm = llm_client
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def analyze(self, behavior_description: str) -> BehaviorAnalysis:
        """Analyze behavior and return structured analysis.

        Args:
            behavior_description: Natural language description of the behavior.

        Returns:
            BehaviorAnalysis containing components, triggers, etc.

        Raises:
            ValueError: If analysis fails or cannot be parsed.
        """
        logger.info(f"Analyzing behavior: {behavior_description[:50]}...")

        prompt = ANALYSIS_PROMPT.format(behavior_description=behavior_description)

        response = await self._llm.complete(
            messages=[Message(role="user", content=prompt)],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        try:
            data = self._parse_response(response.content)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse behavior analysis: {e}")
            raise ValueError(f"Failed to parse behavior analysis: {e}")

        components = self._extract_components(data)

        if not components:
            logger.warning("No components extracted, using default")
            components = [
                BehaviorComponent(
                    name="main_behavior",
                    description=behavior_description,
                    markers=[],
                    opposite_markers=[],
                )
            ]

        analysis = BehaviorAnalysis(
            behavior_name=self._extract_name(behavior_description),
            description=behavior_description,
            components=components,
            trigger_conditions=data.get("trigger_conditions", []),
            contrast_dimensions=data.get("contrast_dimensions", []),
            confounds_to_avoid=data.get("confounds_to_avoid", []),
        )

        logger.info(
            f"Behavior analysis complete: {len(analysis.components)} components, "
            f"{len(analysis.trigger_conditions)} triggers"
        )

        return analysis

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response.

        Handles both raw JSON and markdown code blocks.
        """
        content = content.strip()

        # Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                try:
                    return json.loads(part)
                except json.JSONDecodeError:
                    continue

        # Try finding JSON object in content
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass

        raise json.JSONDecodeError("No valid JSON found", content, 0)

    def _extract_components(self, data: Dict[str, Any]) -> list[BehaviorComponent]:
        """Extract behavior components from parsed data."""
        components = []

        for c in data.get("components", []):
            if not isinstance(c, dict):
                continue

            name = c.get("name", "")
            if not name:
                continue

            components.append(
                BehaviorComponent(
                    name=name,
                    description=c.get("description", ""),
                    markers=c.get("markers", []),
                    opposite_markers=c.get("opposite_markers", []),
                )
            )

        return components

    def _extract_name(self, description: str) -> str:
        """Extract a short name from the behavior description."""
        # Take first 50 chars or first sentence
        name = description.split(".")[0].strip()
        if len(name) > 50:
            name = name[:47] + "..."
        return name
