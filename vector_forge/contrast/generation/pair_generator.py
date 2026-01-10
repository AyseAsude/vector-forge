"""Contrast pair generator with confound control.

This module generates contrast pairs from seeds, ensuring that
the only difference between dst and src is the target behavior.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from vector_forge.contrast.protocols import (
    PairGeneratorProtocol,
    Seed,
    BehaviorAnalysis,
    ContrastPair,
)
from vector_forge.core.protocols import Message
from vector_forge.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


GENERATION_PROMPT = '''Generate a high-quality contrast pair for steering vector training.

## BEHAVIOR
{behavior_description}

## BEHAVIOR COMPONENTS TO TARGET
{target_components}

## SCENARIO
{scenario}

## CONTEXT
{context}

## ATTRIBUTES
{attributes}

---

## CONFOUND CONTROL (CRITICAL)

The ONLY difference between POSITIVE and NEGATIVE should be the target behavior.
Control these confounds:

{confounds}

**Specifically:**
- SIMILAR LENGTH: Both responses should be roughly the same length
- SIMILAR TONE: Both should have similar politeness/formality (unless tone IS the behavior)
- SIMILAR HELPFULNESS: Both should be equally helpful (unless helpfulness IS the behavior)
- SIMILAR QUALITY: Both should be coherent and well-written
- SIMILAR STRUCTURE: Both should have similar organization

---

## TASK

1. Create a natural PROMPT based on the scenario
   - Should feel realistic, not contrived
   - Should naturally allow for both exhibiting and not exhibiting the behavior

2. Generate POSITIVE (dst) response that:
   - CLEARLY exhibits the target behavior
   - Focuses on the specified components
   - Is natural and high-quality

3. Generate NEGATIVE (src) response that:
   - Clearly does NOT exhibit the behavior
   - Could show neutral stance or opposite tendency
   - Is EQUALLY good quality as the positive

4. Self-check for confounds

---

Return JSON:
{{
  "prompt": "The user prompt",
  "dst": "The positive response (exhibits behavior)",
  "src": "The negative response (does not exhibit behavior)",
  "confound_check": {{
    "length_ratio": <dst_length / src_length, should be 0.8-1.2>,
    "tone_matched": <true/false>,
    "helpfulness_matched": <true/false>,
    "structure_similar": <true/false>
  }},
  "component_coverage": ["which components are exhibited in dst"],
  "generation_notes": "Brief notes on the contrast created"
}}'''


class ContrastPairGenerator(PairGeneratorProtocol):
    """Generates contrast pairs from seeds with confound control.

    Creates high-quality contrast pairs where the only difference
    between dst and src is the target behavior.

    Example:
        >>> generator = ContrastPairGenerator(llm_client)
        >>> pair = await generator.generate(seed, analysis)
        >>> print(pair.dst)  # Exhibits behavior
        >>> print(pair.src)  # Does not exhibit behavior
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        """Initialize the pair generator.

        Args:
            llm_client: LLM client for generation.
            temperature: Generation temperature.
            max_tokens: Maximum response tokens (None = provider default).
        """
        self._llm = llm_client
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def generate(
        self,
        seed: Seed,
        analysis: BehaviorAnalysis,
    ) -> ContrastPair:
        """Generate a contrast pair from a seed.

        Args:
            seed: The seed to generate from.
            analysis: Behavior analysis for context.

        Returns:
            ContrastPair with prompt, dst, and src.
        """
        prompt = GENERATION_PROMPT.format(
            behavior_description=analysis.description,
            target_components=self._format_target_components(seed, analysis),
            scenario=seed.scenario,
            context=seed.context or "No additional context",
            attributes=self._format_attributes(seed.attributes),
            confounds=self._format_confounds(analysis),
        )

        try:
            response = await self._llm.complete(
                messages=[Message(role="user", content=prompt)],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )

            data = self._parse_response(response.content)

        except Exception as e:
            logger.error(f"Pair generation failed: {e}")
            # Return empty pair that will fail validation
            return ContrastPair(
                prompt="",
                dst="",
                src="",
                seed=seed,
                metadata={"error": str(e)},
            )

        return ContrastPair(
            prompt=data.get("prompt", ""),
            dst=data.get("dst", ""),
            src=data.get("src", ""),
            seed=seed,
            metadata={
                "confound_check": data.get("confound_check", {}),
                "component_coverage": data.get("component_coverage", []),
                "generation_notes": data.get("generation_notes", ""),
            },
        )

    async def generate_batch(
        self,
        seeds: List[Seed],
        analysis: BehaviorAnalysis,
    ) -> List[ContrastPair]:
        """Generate multiple contrast pairs.

        Args:
            seeds: Seeds to generate from.
            analysis: Behavior analysis for context.

        Returns:
            List of ContrastPairs.
        """
        pairs = []
        for seed in seeds:
            pair = await self.generate(seed, analysis)
            pairs.append(pair)
        return pairs

    def _format_target_components(
        self,
        seed: Seed,
        analysis: BehaviorAnalysis,
    ) -> str:
        """Format target components for the prompt."""
        if not seed.target_components:
            return "All behavior components"

        lines = []
        for comp_name in seed.target_components:
            comp = analysis.get_component(comp_name)
            if comp:
                lines.append(f"- **{comp.name}**: {comp.description}")
                if comp.markers:
                    lines.append(f"  Markers: {', '.join(comp.markers[:3])}")
            else:
                lines.append(f"- {comp_name}")

        return "\n".join(lines) if lines else "All behavior components"

    def _format_attributes(self, attributes: Dict[str, Any]) -> str:
        """Format seed attributes for the prompt."""
        if not attributes:
            return "No specific attributes"

        lines = []
        for key, value in attributes.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def _format_confounds(self, analysis: BehaviorAnalysis) -> str:
        """Format confounds to control for."""
        confounds = analysis.confounds_to_avoid or []

        # Always include these base confounds
        base_confounds = [
            "Response length",
            "Tone and politeness",
            "Helpfulness level",
            "Writing quality",
            "Response structure",
        ]

        all_confounds = list(set(confounds + base_confounds))

        return "\n".join(f"- {c}" for c in all_confounds)

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
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

        # Try finding JSON object
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse pair generation response")
        return {}
