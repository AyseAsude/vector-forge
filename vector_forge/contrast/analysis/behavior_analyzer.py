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
    NegativeExample,
    RealisticScenario,
    ConfoundInfo,
)
from vector_forge.core.protocols import Message
from vector_forge.llm import JSON_RESPONSE_FORMAT
from vector_forge.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


ANALYSIS_PROMPT = '''You are analyzing a behavior for steering vector extraction. Your goal is to deeply understand this behavior so we can generate CLEAN training data with minimal noise.

BEHAVIOR: {behavior_description}

Think carefully and thoroughly. The quality of your analysis determines whether we extract a clean signal or a noisy mixture.

## 1. CORE DEFINITION
First, deeply understand what this behavior IS:
- What is the essence of this behavior?
- What distinguishes it from superficially similar behaviors?
- When exhibited strongly, what does it look like?

## 2. WHAT THIS IS NOT (Critical for clean extraction)
List behaviors that might be CONFUSED with this but are NOT the same:
- What similar-seeming behaviors should we NOT capture?
- What's the boundary between this behavior and acceptable/normal behavior?
- Example: "Sycophancy" is NOT the same as "politeness" or "being encouraging"

For each negative example, explain WHY it's different.

## 3. COMPONENTS
Break into 3-5 distinct, SEPARABLE components. Each should be:
- Observable in text (has linguistic markers)
- Isolatable (can appear without the others)
- Meaningful (captures a real aspect of the behavior)

For each component, provide:
- Specific markers (exact phrases, patterns, linguistic features)
- What ABSENCE looks like (not just generic "doesn't do X")

## 4. REALISTIC SCENARIOS
Generate scenarios where this behavior would NATURALLY arise (not forced/artificial):
- What real situations trigger this behavior?
- Who is the user? What's their state? What are the stakes?
- Make scenarios feel like real deployment, not contrived tests

Avoid: Obviously fake scenarios, cartoon situations, or scenarios that scream "this is a test"

## 5. CONFOUNDS (What to control for)
List factors that might CORRELATE with this behavior but aren't the behavior itself:
- Response length (verbose vs concise)
- Tone (formal vs casual)
- Helpfulness level
- Writing quality
- Emotional warmth
- etc.

For each confound, note if it's EASY or HARD to control for.

## 6. CONTRAST STRATEGY
How can we create pairs where:
- dst CLEARLY shows the behavior
- src CLEARLY doesn't show it
- EVERYTHING ELSE is matched (length, tone, helpfulness, quality)

Return JSON:
{{
  "core_definition": "one sentence capturing the essence",
  "not_this_behavior": [
    {{
      "similar_behavior": "what it might be confused with",
      "why_different": "why it's not the same thing",
      "example": "concrete example of the non-behavior"
    }}
  ],
  "components": [
    {{
      "name": "short_name",
      "description": "what this component represents",
      "markers": ["specific phrase", "linguistic pattern"],
      "opposite_markers": ["what absence looks like"],
      "isolation_note": "can this appear independently?"
    }}
  ],
  "realistic_scenarios": [
    {{
      "setup": "the situation",
      "user_persona": "who the user is",
      "natural_trigger": "why behavior would arise here",
      "stakes": "low/medium/high"
    }}
  ],
  "trigger_conditions": ["when this behavior naturally manifests"],
  "contrast_dimensions": ["clearest ways to show presence vs absence"],
  "confounds_to_avoid": [
    {{
      "factor": "the confound",
      "difficulty": "easy/medium/hard to control",
      "strategy": "how to control for it"
    }}
  ]
}}

Be specific, nuanced, and thoughtful. Vague analysis produces noisy vectors.'''


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
            response_format=JSON_RESPONSE_FORMAT,
        )

        try:
            data = self._parse_json(response.content)
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

        # Extract enhanced fields
        negative_examples = self._extract_negative_examples(data)
        realistic_scenarios = self._extract_realistic_scenarios(data)
        confound_details = self._extract_confound_details(data)

        # Extract simple confounds list for backward compatibility
        confounds_simple = self._extract_confounds_simple(data)

        analysis = BehaviorAnalysis(
            behavior_name=self._extract_name(behavior_description),
            description=behavior_description,
            components=components,
            trigger_conditions=data.get("trigger_conditions", []),
            contrast_dimensions=data.get("contrast_dimensions", []),
            confounds_to_avoid=confounds_simple,
            # Enhanced fields
            core_definition=data.get("core_definition", ""),
            not_this_behavior=negative_examples,
            realistic_scenarios=realistic_scenarios,
            confound_details=confound_details,
        )

        logger.info(
            f"Behavior analysis complete: {len(analysis.components)} components, "
            f"{len(analysis.trigger_conditions)} triggers, "
            f"{len(analysis.not_this_behavior)} negative examples, "
            f"{len(analysis.realistic_scenarios)} scenarios"
        )

        return analysis

    def _parse_json(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response.

        With JSON response format enabled, the LLM should return valid JSON directly.
        Falls back to markdown extraction if needed for compatibility.
        """
        content = content.strip()

        # Primary: Direct JSON parse (expected with response_format)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Fallback: Extract from markdown code block if present
        if "```" in content:
            parts = content.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:  # Odd indices are inside code blocks
                    part = part.strip()
                    if part.startswith(("json", "JSON")):
                        part = part[4:].strip()
                    try:
                        return json.loads(part)
                    except json.JSONDecodeError:
                        continue

        # Fallback: Find JSON object boundaries
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])

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

    def _extract_negative_examples(self, data: Dict[str, Any]) -> list[NegativeExample]:
        """Extract negative examples (what this behavior is NOT)."""
        examples = []
        for item in data.get("not_this_behavior", []):
            if not isinstance(item, dict):
                continue
            examples.append(
                NegativeExample(
                    similar_behavior=item.get("similar_behavior", ""),
                    why_different=item.get("why_different", ""),
                    example=item.get("example", ""),
                )
            )
        return examples

    def _extract_realistic_scenarios(self, data: Dict[str, Any]) -> list[RealisticScenario]:
        """Extract realistic scenarios from analysis."""
        scenarios = []
        for item in data.get("realistic_scenarios", []):
            if not isinstance(item, dict):
                continue
            scenarios.append(
                RealisticScenario(
                    setup=item.get("setup", ""),
                    user_persona=item.get("user_persona", ""),
                    natural_trigger=item.get("natural_trigger", ""),
                    stakes=item.get("stakes", "medium"),
                )
            )
        return scenarios

    def _extract_confound_details(self, data: Dict[str, Any]) -> list[ConfoundInfo]:
        """Extract detailed confound information."""
        confounds = []
        for item in data.get("confounds_to_avoid", []):
            if isinstance(item, dict):
                confounds.append(
                    ConfoundInfo(
                        factor=item.get("factor", ""),
                        difficulty=item.get("difficulty", "medium"),
                        strategy=item.get("strategy", ""),
                    )
                )
        return confounds

    def _extract_confounds_simple(self, data: Dict[str, Any]) -> list[str]:
        """Extract simple confound list for backward compatibility."""
        confounds = []
        for item in data.get("confounds_to_avoid", []):
            if isinstance(item, str):
                confounds.append(item)
            elif isinstance(item, dict):
                factor = item.get("factor", "")
                if factor:
                    confounds.append(factor)
        return confounds
