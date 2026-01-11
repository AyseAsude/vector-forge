"""Seed generator for creating quality scenarios for contrast pairs.

This module generates high-quality seeds (scenarios) from behavior analysis.
Seeds are scored and filtered to ensure only quality scenarios are used
for contrast pair generation.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from vector_forge.contrast.protocols import (
    SeedGeneratorProtocol,
    BehaviorAnalysis,
    Seed,
)
from vector_forge.contrast.utils import parse_llm_json
from vector_forge.core.protocols import Message
from vector_forge.llm import JSON_RESPONSE_FORMAT
from vector_forge.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

# Default retry configuration
DEFAULT_MAX_RETRIES = 3


SEED_GENERATION_PROMPT = '''Generate high-quality training scenarios for steering vector extraction.

## BEHAVIOR
{behavior_description}
{core_definition}

## THE BEHAVIORAL TEST (Critical - scenarios must implement this)
{behavioral_test}

## WHAT THIS IS NOT (avoid confusing with these)
{negative_examples}

## BEHAVIOR COMPONENTS
{components_description}

## EXISTING SCENARIOS (use as inspiration)
{existing_scenarios}

---

Generate {count} diverse scenarios for creating contrast pairs.

## SCENARIO REQUIREMENTS

Each scenario must:
1. **Implement the behavioral test** - create a situation where the distinguishing variable can vary
2. **Allow clear contrast** - possible to show clear presence vs absence of behavior
3. **Feel realistic** - like something that would occur in real deployment
4. **Be diverse** - cover different contexts and situations relevant to this behavior

## KEY PRINCIPLE

The scenario creates a situation. The contrast will be in HOW THE MODEL RESPONDS.
Focus on creating situations where the distinguishing variable naturally comes into play.

## OUTPUT

For each scenario:
- scenario: The situation setup
- context: Relevant context about the user/situation
- target_components: Which behavior components this tests
- expected_contrast_strength: How clear can the contrast be (1-10)
- example_prompt: A concrete example of what the user might say

Return JSON:
{{
  "scenarios": [
    {{
      "scenario": "description of the situation",
      "context": "relevant context",
      "target_components": ["component names"],
      "expected_contrast_strength": 8,
      "example_prompt": "example user message"
    }}
  ]
}}

Generate diverse scenarios that implement the behavioral test in different contexts.'''


SEED_SCORING_PROMPT = '''Score these scenarios for training data quality.

BEHAVIOR: {behavior_description}

## SCORING CRITERIA

1. RELEVANCE (1-10): Does this scenario naturally involve the target behavior?
   - 10: Behavior is central to how one would respond
   - 5: Behavior is somewhat relevant
   - 1: Behavior is tangential or doesn't apply

2. CONTRAST_CLARITY (1-10): Can we create obviously different responses?
   - 10: Anyone could immediately see the difference
   - 5: Difference is noticeable but subtle
   - 1: Very hard to show clear contrast

3. ISOLATION (1-10): Can we vary ONLY the behavior, not other factors?
   - 10: Easy to keep length, tone, helpfulness constant
   - 5: Some confounds are hard to avoid
   - 1: Changing behavior forces other changes

4. GENERALIZATION (1-10): Is this representative of real situations?
   - 10: Common, important scenario
   - 5: Occasional but not rare
   - 1: Edge case or unrealistic

## SCENARIOS TO SCORE
{scenarios_json}

---

Return JSON with scores for each scenario:
{{
  "scores": [
    {{
      "scenario_index": 0,
      "relevance": 8,
      "contrast_clarity": 9,
      "isolation": 7,
      "generalization": 8,
      "overall": 8.0,
      "issues": "any issues or concerns",
      "suggestions": "how to improve if needed"
    }}
  ]
}}'''


class SeedGenerator(SeedGeneratorProtocol):
    """Generates quality seeds from behavior analysis.

    Seeds are scenarios that guide contrast pair generation. This class:
    1. Generates candidate scenarios based on behavior analysis
    2. Scores scenarios on quality dimensions
    3. Filters to keep only high-quality seeds

    Example:
        >>> generator = SeedGenerator(llm_client)
        >>> seeds = await generator.generate(analysis, count=50)
        >>> print(seeds[0].scenario)
        "User presents incorrect claim seeking validation"
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        min_quality_score: float = 6.0,
        temperature: float = 0.8,
        max_tokens: Optional[int] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        """Initialize the seed generator.

        Args:
            llm_client: LLM client for generation.
            min_quality_score: Minimum score to keep a seed.
            temperature: Generation temperature.
            max_tokens: Maximum tokens for response (None = provider default).
            max_retries: Maximum retry attempts for generation on JSON parse failures.
        """
        self._llm = llm_client
        self._min_quality = min_quality_score
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    async def generate(
        self,
        analysis: BehaviorAnalysis,
        count: int,
    ) -> List[Seed]:
        """Generate quality seeds for the behavior.

        Generates more candidates than needed, scores them, and returns
        the top seeds meeting quality threshold. Includes retry logic
        for handling JSON parse failures.

        Args:
            analysis: Behavior analysis to base seeds on.
            count: Number of seeds to generate.

        Returns:
            List of quality seeds, sorted by quality score.
        """
        logger.info(f"Generating {count} seeds for behavior: {analysis.behavior_name}")

        # Generate more than needed to allow filtering
        target_count = int(count * 1.5) + 10

        prompt = SEED_GENERATION_PROMPT.format(
            behavior_description=analysis.description,
            core_definition=f"Core: {analysis.core_definition}" if analysis.core_definition else "",
            behavioral_test=self._format_behavioral_test(analysis),
            negative_examples=self._format_negative_examples(analysis),
            components_description=self._format_components(analysis),
            existing_scenarios=self._format_existing_scenarios(analysis),
            count=target_count,
        )

        # Attempt generation with retries
        seeds = await self._generate_with_retry(prompt)

        if not seeds:
            logger.warning("No seeds generated after all retry attempts")
            return []

        logger.info(f"Generated {len(seeds)} candidate seeds")

        # Score and filter seeds
        scored_seeds = await self.score_seeds(seeds, analysis)

        # Filter by quality and take top count
        quality_seeds = [
            seed for seed, score in scored_seeds
            if score >= self._min_quality
        ]

        result = quality_seeds[:count]
        logger.info(
            f"Filtered to {len(result)} quality seeds "
            f"(min_score={self._min_quality})"
        )

        return result

    async def _generate_with_retry(self, prompt: str) -> List[Seed]:
        """Generate seeds with retry logic on JSON parse failures.

        If JSON parsing fails after auto-repair, retries the entire
        generation from scratch up to max_retries times.

        Args:
            prompt: The generation prompt.

        Returns:
            List of extracted seeds, or empty list if all attempts fail.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            if attempt > 0:
                logger.info(f"Retry attempt {attempt + 1}/{self._max_retries} for seed generation")

            try:
                response = await self._llm.complete(
                    messages=[Message(role="user", content=prompt)],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    response_format=JSON_RESPONSE_FORMAT,
                )

                data = parse_llm_json(response.content)
                seeds = self._extract_seeds(data)

                if seeds:
                    if attempt > 0:
                        logger.info(f"Seed generation succeeded on attempt {attempt + 1}")
                    return seeds

                # No seeds extracted but parsing succeeded - might be empty response
                logger.warning(f"Attempt {attempt + 1}: Parsed response but no seeds extracted")

            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self._max_retries}: "
                    f"JSON parse failed: {e}"
                )
            except Exception as e:
                last_error = e
                logger.error(f"Attempt {attempt + 1}/{self._max_retries}: Unexpected error: {e}")

        if last_error:
            logger.error(f"All {self._max_retries} generation attempts failed. Last error: {last_error}")

        return []

    async def score_seeds(
        self,
        seeds: List[Seed],
        analysis: BehaviorAnalysis,
    ) -> List[Tuple[Seed, float]]:
        """Score seeds by quality with retry logic.

        Args:
            seeds: Seeds to score.
            analysis: Behavior analysis for context.

        Returns:
            List of (seed, score) tuples, sorted by score descending.
        """
        if not seeds:
            return []

        logger.info(f"Scoring {len(seeds)} seeds")

        # Format seeds for scoring
        scenarios_for_scoring = [
            {
                "index": i,
                "scenario": s.scenario,
                "context": s.context,
                "attributes": s.attributes,
                "target_components": s.target_components,
            }
            for i, s in enumerate(seeds)
        ]

        prompt = SEED_SCORING_PROMPT.format(
            behavior_description=analysis.description,
            scenarios_json=json.dumps(scenarios_for_scoring, indent=2),
        )

        # Attempt scoring with retries
        data = await self._score_with_retry(prompt, seeds)

        # Build scored list
        scored: List[Tuple[Seed, float]] = []
        score_data_list = data.get("scores", [])

        for score_data in score_data_list:
            idx = score_data.get("scenario_index", -1)
            if 0 <= idx < len(seeds):
                overall = float(score_data.get("overall", 5.0))
                seed = seeds[idx]
                seed.quality_score = overall
                scored.append((seed, overall))

        # Add any seeds that weren't scored
        scored_indices = {s.get("scenario_index") for s in score_data_list}
        for i, seed in enumerate(seeds):
            if i not in scored_indices:
                seed.quality_score = seed.expected_contrast_strength
                scored.append((seed, seed.expected_contrast_strength))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    async def _score_with_retry(
        self,
        prompt: str,
        seeds: List[Seed],
    ) -> Dict[str, Any]:
        """Score seeds with retry logic on JSON parse failures.

        Args:
            prompt: The scoring prompt.
            seeds: Seeds being scored (for fallback).

        Returns:
            Parsed scoring data, or fallback data if all attempts fail.
        """
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            if attempt > 0:
                logger.info(f"Retry attempt {attempt + 1}/{self._max_retries} for seed scoring")

            try:
                response = await self._llm.complete(
                    messages=[Message(role="user", content=prompt)],
                    temperature=0.3,  # Lower temp for consistent scoring
                    max_tokens=self._max_tokens,
                    response_format=JSON_RESPONSE_FORMAT,
                )

                data = parse_llm_json(response.content)

                # Validate we got scores
                if "scores" in data and data["scores"]:
                    if attempt > 0:
                        logger.info(f"Seed scoring succeeded on attempt {attempt + 1}")
                    return data

                logger.warning(f"Attempt {attempt + 1}: Parsed response but no scores found")

            except json.JSONDecodeError as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self._max_retries}: "
                    f"JSON parse failed for scoring: {e}"
                )
            except Exception as e:
                last_error = e
                logger.error(f"Attempt {attempt + 1}/{self._max_retries}: Unexpected error: {e}")

        # All attempts failed - return fallback with default scores
        if last_error:
            logger.warning(
                f"All {self._max_retries} scoring attempts failed. "
                f"Using default scores. Last error: {last_error}"
            )

        # Build fallback scores using expected_contrast_strength
        return {
            "scores": [
                {
                    "scenario_index": i,
                    "overall": seed.expected_contrast_strength,
                }
                for i, seed in enumerate(seeds)
            ]
        }

    def _format_behavioral_test(self, analysis: BehaviorAnalysis) -> str:
        """Format behavioral test for prompt."""
        if not analysis.behavioral_test:
            return "No behavioral test specified - focus on creating scenarios where the behavior can clearly vary."

        bt = analysis.behavioral_test
        return f"""Distinguishing Variable: {bt.distinguishing_variable}

Test Description: {bt.description}
- User action that triggers: {bt.user_action}
- Model faces choice: {bt.model_choice}
- If behavior PRESENT: {bt.present_response_pattern}
- If behavior ABSENT: {bt.absent_response_pattern}

Scenarios should create situations where this test naturally applies."""

    def _format_components(self, analysis: BehaviorAnalysis) -> str:
        """Format behavior components for prompt."""
        lines = []
        for c in analysis.components:
            lines.append(f"- **{c.name}**: {c.description}")
            if c.markers:
                lines.append(f"  - Markers: {', '.join(c.markers[:5])}")
            if c.opposite_markers:
                lines.append(f"  - Opposite: {', '.join(c.opposite_markers[:3])}")
        return "\n".join(lines) if lines else "No components specified"

    def _format_negative_examples(self, analysis: BehaviorAnalysis) -> str:
        """Format what this behavior is NOT."""
        if not analysis.not_this_behavior:
            return "No negative examples specified"

        lines = []
        for neg in analysis.not_this_behavior:
            lines.append(f"- NOT {neg.similar_behavior}: {neg.why_different}")
        return "\n".join(lines)

    def _format_existing_scenarios(self, analysis: BehaviorAnalysis) -> str:
        """Format existing realistic scenarios from analysis."""
        if not analysis.realistic_scenarios:
            return "None provided - generate fresh scenarios"

        lines = ["Use these as inspiration (but generate diverse variations):"]
        for sc in analysis.realistic_scenarios:
            lines.append(f"- {sc.setup} (user: {sc.user_persona}, trigger: {sc.natural_trigger})")
        return "\n".join(lines)

    def _format_confounds(self, analysis: BehaviorAnalysis) -> str:
        """Format confounds with control strategies."""
        if analysis.confound_details:
            lines = []
            for conf in analysis.confound_details:
                lines.append(f"- {conf.factor}: {conf.strategy}")
            return "\n".join(lines)
        elif analysis.confounds_to_avoid:
            return "\n".join(f"- {c}" for c in analysis.confounds_to_avoid)
        else:
            return "- Response length\n- Tone\n- Helpfulness\n- Quality"

    def _format_list(self, items: List[str]) -> str:
        """Format a list of items for prompt."""
        if not items:
            return "None specified"
        return "\n".join(f"- {item}" for item in items)

    def _extract_seeds(self, data: Dict[str, Any]) -> List[Seed]:
        """Extract seeds from parsed response data."""
        seeds = []

        for s in data.get("scenarios", []):
            if not isinstance(s, dict):
                continue

            scenario = s.get("scenario", "")
            if not scenario:
                continue

            seeds.append(
                Seed(
                    scenario=scenario,
                    context=s.get("context", ""),
                    expected_contrast_strength=float(
                        s.get("expected_contrast_strength", 7.0)
                    ),
                    target_components=s.get("target_components", []),
                    attributes=s.get("attributes", {}),
                    example_prompt=s.get("example_prompt", ""),
                )
            )

        return seeds
