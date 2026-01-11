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
from vector_forge.llm import JSON_RESPONSE_FORMAT
from vector_forge.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


GENERATION_PROMPT = '''Generate a high-quality contrast pair for steering vector training.

## BEHAVIOR: {behavior_description}
{core_definition}

## WHAT THIS IS NOT (Critical - do not confuse with these)
{negative_examples}

## BEHAVIOR COMPONENTS TO TARGET
{target_components}

## SCENARIO
{scenario}

## CONTEXT
{context}

---

## REALISM REQUIREMENT (Critical for clean extraction)

Your responses must feel NATURAL - like how a real AI would actually respond.
- NO forced or exaggerated behavior ("I ABSOLUTELY LOVE your idea!!!")
- NO cartoonish or obvious test-case responses
- The behavior should arise naturally, not be artificially inserted
- Both responses should read as plausible AI assistant outputs

Think: Would someone reading this suspect it's a test case? If yes, it's too artificial.

## CONFOUND CONTROL (Critical for clean signal)

The ONLY difference between POSITIVE and NEGATIVE must be the target behavior.
Both responses must be MATCHED on:

{confounds}

**Verification checklist:**
- LENGTH: Count the words - both should be within 20% of each other
- TONE: Same level of formality, warmth, directness
- HELPFULNESS: Both equally helpful (unless helpfulness IS the behavior)
- QUALITY: Both equally well-written and coherent
- STRUCTURE: Similar organization (paragraphs, lists, etc.)

If you cannot match a confound, the pair is useless. Start over.

---

## GENERATION TASK

1. **PROMPT**: Create a natural user message based on the scenario
   - Must feel like a real user request, not a test
   - Should naturally elicit the behavior without forcing it

2. **POSITIVE (dst)**: Response that EXHIBITS the behavior
   - Show the behavior CLEARLY but NATURALLY
   - Focus on specified components
   - Must be a response a real AI might produce

3. **NEGATIVE (src)**: Response that does NOT exhibit the behavior
   - Must NOT show the behavior or its markers
   - NOT the same as showing opposite behavior (unless appropriate)
   - Must be EQUALLY good - just without this specific behavior

4. **SELF-CHECK**: Before outputting, verify:
   - Is the behavior clear in dst?
   - Is the behavior absent in src?
   - Are confounds matched?
   - Do both responses feel natural?

---

Return JSON:
{{
  "prompt": "The user prompt (must feel realistic)",
  "dst": "Positive response - exhibits behavior naturally",
  "src": "Negative response - no behavior, matched confounds",
  "confound_check": {{
    "dst_word_count": <number>,
    "src_word_count": <number>,
    "length_ratio": <dst/src, target 0.8-1.2>,
    "tone_matched": <true/false>,
    "helpfulness_matched": <true/false>,
    "structure_matched": <true/false>
  }},
  "naturalness_check": {{
    "dst_feels_natural": <true/false>,
    "src_feels_natural": <true/false>,
    "would_pass_as_real": <true/false>
  }},
  "component_coverage": ["components exhibited in dst"],
  "behavior_markers_in_dst": ["specific markers present"],
  "behavior_markers_in_src": ["should be empty or minimal"]
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
            core_definition=f"Core: {analysis.core_definition}" if analysis.core_definition else "",
            negative_examples=self._format_negative_examples(analysis),
            target_components=self._format_target_components(seed, analysis),
            scenario=seed.scenario,
            context=seed.context or "No additional context",
            confounds=self._format_confounds(analysis),
        )

        try:
            response = await self._llm.complete(
                messages=[Message(role="user", content=prompt)],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_format=JSON_RESPONSE_FORMAT,
            )

            data = self._parse_json(response.content)

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

    def _format_negative_examples(self, analysis: BehaviorAnalysis) -> str:
        """Format what this behavior is NOT."""
        if not analysis.not_this_behavior:
            return "No negative examples specified"

        lines = []
        for neg in analysis.not_this_behavior:
            lines.append(f"- NOT {neg.similar_behavior}: {neg.why_different}")
        return "\n".join(lines)

    def _format_confounds(self, analysis: BehaviorAnalysis) -> str:
        """Format confounds to control for with strategies."""
        lines = []

        # Use detailed confound info if available
        if analysis.confound_details:
            for conf in analysis.confound_details:
                lines.append(f"- {conf.factor} ({conf.difficulty}): {conf.strategy}")
        else:
            # Fallback to simple list
            for conf in analysis.confounds_to_avoid:
                lines.append(f"- {conf}")

        # Always include base confounds if not already covered
        base_confounds = {
            "Response length": "Count words, keep within 20%",
            "Tone": "Match formality, warmth, directness",
            "Helpfulness": "Both equally helpful",
            "Writing quality": "Both equally well-written",
        }

        existing_factors = {c.factor.lower() for c in analysis.confound_details} if analysis.confound_details else set()
        for factor, strategy in base_confounds.items():
            if factor.lower() not in existing_factors and factor.lower() not in " ".join(analysis.confounds_to_avoid).lower():
                lines.append(f"- {factor}: {strategy}")

        return "\n".join(lines)

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
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse pair generation response")
        return {}
