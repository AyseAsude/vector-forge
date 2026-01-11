"""Contrast pair generator with intensity-aware composition.

This module generates contrast pairs from seeds at different intensity levels,
ensuring clear contrast on the distinguishing variable at all intensities.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Tuple

from vector_forge.contrast.protocols import (
    PairGeneratorProtocol,
    Seed,
    BehaviorAnalysis,
    ContrastPair,
    SignalIntensity,
)
from vector_forge.contrast.utils import safe_parse_llm_json
from vector_forge.core.protocols import Message
from vector_forge.llm import JSON_RESPONSE_FORMAT
from vector_forge.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


GENERATION_PROMPT = '''Generate a contrast pair for steering vector training.

## BEHAVIOR
{behavior_description}
{core_definition}

## THE BEHAVIORAL TEST (Critical - this defines the contrast)
{behavioral_test}

## WHAT THIS IS NOT
{negative_examples}

## TARGET COMPONENTS
{target_components}

## SCENARIO
{scenario}

## CONTEXT
{context}

---

## INTENSITY: {intensity}
{intensity_guidance}

**IMPORTANT**: Intensity controls HOW the behavior is expressed, not the STRENGTH of contrast.
- The contrast on the distinguishing variable must be CLEAR at ALL intensities
- "Natural" means realistic expression, NOT weak signal
- dst and src should clearly differ on the distinguishing variable regardless of intensity

---

## GENERATION TASK

Generate a pair where the ONLY difference is the distinguishing variable: {distinguishing_variable}

1. **PROMPT**: User message that creates the test situation
   - Implements the scenario naturally
   - Allows the model to exhibit or not exhibit the behavior

2. **dst (behavior PRESENT)**:
   - Clearly shows: {present_pattern}
   - Express at {intensity} intensity level
   - The distinguishing variable is clearly present

3. **src (behavior ABSENT)**:
   - Clearly shows: {absent_pattern}
   - The distinguishing variable is clearly absent
   - Otherwise matches dst (same helpfulness, quality, length)

## CRITICAL: CONTRAST CLARITY

The contrast must be on the DISTINGUISHING VARIABLE, not on:
- Tone or style (unless that IS the behavior)
- Length or verbosity
- Helpfulness level
- Writing quality

If you swap ONLY the distinguishing variable behavior, src should become dst.

---

Return JSON:
{{
  "prompt": "user message implementing the scenario",
  "dst": "response with behavior present at {intensity} intensity",
  "src": "response with behavior absent, otherwise matched",
  "distinguishing_variable_in_dst": "how the variable manifests in dst",
  "distinguishing_variable_in_src": "how the variable differs in src",
  "intensity_applied": "{intensity}"
}}'''


class ContrastPairGenerator(PairGeneratorProtocol):
    """Generates contrast pairs from seeds with intensity-aware composition.

    Creates contrast pairs where the only difference between dst and src
    is the distinguishing variable from the behavioral test.
    Supports different intensity levels while maintaining clear contrast.

    Example:
        >>> generator = ContrastPairGenerator(llm_client)
        >>> pair = await generator.generate(seed, analysis, intensity=SignalIntensity.MEDIUM)
        >>> print(pair.dst)  # Exhibits behavior
        >>> print(pair.src)  # Does not exhibit behavior
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        max_concurrency: int = 10,
    ):
        """Initialize the pair generator.

        Args:
            llm_client: LLM client for generation.
            temperature: Base generation temperature.
            max_tokens: Maximum response tokens (None = provider default).
            max_concurrency: Maximum concurrent generations for batch methods.
        """
        self._llm = llm_client
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_concurrency = max_concurrency

    async def generate(
        self,
        seed: Seed,
        analysis: BehaviorAnalysis,
        intensity: SignalIntensity = SignalIntensity.MEDIUM,
    ) -> ContrastPair:
        """Generate a contrast pair from a seed at specified intensity.

        Args:
            seed: The seed to generate from.
            analysis: Behavior analysis for context.
            intensity: Signal intensity level for the pair.

        Returns:
            ContrastPair with prompt, dst, and src.
        """
        # Get behavioral test info
        behavioral_test = self._format_behavioral_test(analysis)
        distinguishing_var = self._get_distinguishing_variable(analysis)
        present_pattern = self._get_present_pattern(analysis)
        absent_pattern = self._get_absent_pattern(analysis)
        intensity_guidance = self._get_intensity_guidance(analysis, intensity)

        prompt = GENERATION_PROMPT.format(
            behavior_description=analysis.description,
            core_definition=f"Core: {analysis.core_definition}" if analysis.core_definition else "",
            behavioral_test=behavioral_test,
            negative_examples=self._format_negative_examples(analysis),
            target_components=self._format_target_components(seed, analysis),
            scenario=seed.scenario,
            context=seed.context or "No additional context",
            intensity=intensity.value.upper(),
            intensity_guidance=intensity_guidance,
            distinguishing_variable=distinguishing_var,
            present_pattern=present_pattern,
            absent_pattern=absent_pattern,
        )

        try:
            response = await self._llm.complete(
                messages=[Message(role="user", content=prompt)],
                temperature=self._get_temperature_for_intensity(intensity),
                max_tokens=self._max_tokens,
                response_format=JSON_RESPONSE_FORMAT,
            )

            data = safe_parse_llm_json(response.content)

        except Exception as e:
            logger.error(f"Pair generation failed: {e}")
            return ContrastPair(
                prompt="",
                dst="",
                src="",
                seed=seed,
                metadata={"error": str(e), "intensity": intensity.value},
            )

        return ContrastPair(
            prompt=data.get("prompt", ""),
            dst=data.get("dst", ""),
            src=data.get("src", ""),
            seed=seed,
            metadata={
                "intensity": intensity.value,
                "distinguishing_variable_in_dst": data.get("distinguishing_variable_in_dst", ""),
                "distinguishing_variable_in_src": data.get("distinguishing_variable_in_src", ""),
            },
        )

    def _get_temperature_for_intensity(self, intensity: SignalIntensity) -> float:
        """Get appropriate temperature for intensity level."""
        # Higher temperature for more extreme (creative), lower for natural (consistent)
        temps = {
            SignalIntensity.EXTREME: min(0.9, self._temperature + 0.2),
            SignalIntensity.HIGH: self._temperature + 0.1,
            SignalIntensity.MEDIUM: self._temperature,
            SignalIntensity.NATURAL: max(0.5, self._temperature - 0.1),
        }
        return temps.get(intensity, self._temperature)

    def _format_behavioral_test(self, analysis: BehaviorAnalysis) -> str:
        """Format behavioral test for prompt."""
        if not analysis.behavioral_test:
            return "No specific behavioral test defined."

        bt = analysis.behavioral_test
        return f"""Distinguishing Variable: {bt.distinguishing_variable}
- User action: {bt.user_action}
- Model choice: {bt.model_choice}
- If PRESENT: {bt.present_response_pattern}
- If ABSENT: {bt.absent_response_pattern}"""

    def _get_distinguishing_variable(self, analysis: BehaviorAnalysis) -> str:
        """Get the distinguishing variable."""
        if analysis.behavioral_test:
            return analysis.behavioral_test.distinguishing_variable
        return "the target behavior"

    def _get_present_pattern(self, analysis: BehaviorAnalysis) -> str:
        """Get the present response pattern."""
        if analysis.behavioral_test:
            return analysis.behavioral_test.present_response_pattern
        return "exhibits the behavior"

    def _get_absent_pattern(self, analysis: BehaviorAnalysis) -> str:
        """Get the absent response pattern."""
        if analysis.behavioral_test:
            return analysis.behavioral_test.absent_response_pattern
        return "does not exhibit the behavior"

    def _get_intensity_guidance(self, analysis: BehaviorAnalysis, intensity: SignalIntensity) -> str:
        """Get intensity-specific guidance from analysis or defaults."""
        # Try to get from analysis calibration
        if analysis.intensity_calibration:
            specific = analysis.intensity_calibration.get_for_intensity(intensity)
            if specific:
                return f"At {intensity.value} intensity, this behavior looks like: {specific}"

        # Default guidance
        defaults = {
            SignalIntensity.EXTREME: "Maximum, unmistakable expression. The behavior is obvious and pronounced.",
            SignalIntensity.HIGH: "Clearly present expression. Obvious on first read.",
            SignalIntensity.MEDIUM: "Noticeable, balanced expression. Clear but not exaggerated.",
            SignalIntensity.NATURAL: "Subtle, realistic expression. Would pass as normal conversation, but the distinguishing variable is still clearly different between dst and src.",
        }
        return defaults.get(intensity, "")

    async def generate_batch(
        self,
        seeds: List[Seed],
        analysis: BehaviorAnalysis,
        intensity: SignalIntensity = SignalIntensity.MEDIUM,
        max_concurrency: Optional[int] = None,
    ) -> List[ContrastPair]:
        """Generate multiple contrast pairs at the same intensity.

        Uses asyncio.gather with semaphore for concurrent generation.

        Args:
            seeds: Seeds to generate from.
            analysis: Behavior analysis for context.
            intensity: Signal intensity for all pairs.
            max_concurrency: Maximum concurrent generations. If None, uses instance default.

        Returns:
            List of ContrastPairs.
        """
        if not seeds:
            return []

        concurrency = max_concurrency if max_concurrency is not None else self._max_concurrency
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_generate(seed: Seed) -> ContrastPair:
            async with semaphore:
                return await self.generate(seed, analysis, intensity)

        tasks = [bounded_generate(seed) for seed in seeds]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log them
        pairs = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Batch generation failed for seed {i}: {result}")
            else:
                pairs.append(result)

        return pairs

    async def generate_batch_distributed(
        self,
        seeds: List[Seed],
        analysis: BehaviorAnalysis,
        distribution: Optional[Dict[SignalIntensity, float]] = None,
        max_concurrency: Optional[int] = None,
    ) -> List[ContrastPair]:
        """Generate pairs with distributed intensities concurrently.

        Uses asyncio.gather with semaphore for concurrent generation.

        Args:
            seeds: Seeds to generate from.
            analysis: Behavior analysis for context.
            distribution: Optional intensity distribution (defaults to balanced).
            max_concurrency: Maximum concurrent generations. If None, uses instance default.

        Returns:
            List of ContrastPairs at varying intensities.
        """
        if not seeds:
            return []

        if distribution is None:
            distribution = {
                SignalIntensity.EXTREME: 0.1,
                SignalIntensity.HIGH: 0.2,
                SignalIntensity.MEDIUM: 0.3,
                SignalIntensity.NATURAL: 0.4,
            }

        n_seeds = len(seeds)

        # Calculate counts for each intensity
        intensity_counts: Dict[SignalIntensity, int] = {}
        remaining = n_seeds
        for intensity, ratio in distribution.items():
            count = int(n_seeds * ratio)
            intensity_counts[intensity] = count
            remaining -= count

        # Distribute remaining to highest ratio
        if remaining > 0:
            max_intensity = max(distribution, key=distribution.get)
            intensity_counts[max_intensity] += remaining

        # Build list of (seed, intensity) pairs
        seed_intensity_pairs: List[Tuple[Seed, SignalIntensity]] = []
        seed_idx = 0
        for intensity, count in intensity_counts.items():
            for _ in range(count):
                if seed_idx >= n_seeds:
                    break
                seed_intensity_pairs.append((seeds[seed_idx], intensity))
                seed_idx += 1

        # Generate concurrently with semaphore
        concurrency = max_concurrency if max_concurrency is not None else self._max_concurrency
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_generate(seed: Seed, intensity: SignalIntensity) -> ContrastPair:
            async with semaphore:
                return await self.generate(seed, analysis, intensity)

        tasks = [bounded_generate(seed, intensity) for seed, intensity in seed_intensity_pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and log them
        pairs = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Batch distributed generation failed for pair {i}: {result}")
            else:
                pairs.append(result)

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
