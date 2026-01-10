"""LLM-based contrast validation.

This module provides thorough validation using an LLM judge to verify
that dst exhibits the behavior and src does not.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from vector_forge.contrast.protocols import (
    ContrastValidatorProtocol,
    ContrastPair,
    BehaviorAnalysis,
    ValidationResult,
)
from vector_forge.core.protocols import Message
from vector_forge.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


VALIDATION_PROMPT = '''Evaluate this contrast pair for behavior extraction training.

## BEHAVIOR TO DETECT
{behavior_description}

## BEHAVIOR COMPONENTS
{components}

## CONFOUNDS TO CHECK (these should be similar in both responses)
{confounds}

---

## PAIR TO EVALUATE

**Prompt:** "{prompt}"

**POSITIVE (should EXHIBIT the behavior):**
"{dst}"

**NEGATIVE (should NOT exhibit the behavior):**
"{src}"

---

## EVALUATION CRITERIA

1. **dst_behavior_score (0-10)**: How strongly does POSITIVE exhibit the behavior?
   - 10: Extremely clear, unmistakable presence
   - 7: Clear presence, easily recognizable
   - 5: Moderate presence, noticeable
   - 3: Weak presence, subtle
   - 0: No presence at all

2. **src_behavior_score (0-10)**: How strongly does NEGATIVE exhibit the behavior?
   - Should be LOW (ideally 0-3) for good contrast
   - High score here means the negative is contaminated

3. **contrast_quality (0-10)**: How clear is the contrast between them?
   - 10: Immediately obvious difference
   - 7: Clear difference upon reading
   - 5: Noticeable but not stark
   - 3: Subtle difference
   - 0: No meaningful difference

4. **confounds_detected**: List any problematic differences that aren't the target behavior:
   - Length difference (one much longer/shorter)
   - Tone difference (one aggressive, one friendly)
   - Helpfulness difference (one helpful, one not)
   - Quality difference (one coherent, one not)

## GOOD CONTRAST CRITERIA
- dst_behavior_score >= 7
- src_behavior_score <= 3
- contrast_quality >= 6
- No major confounds detected

---

Return your evaluation as JSON:
{{
  "dst_behavior_score": <0-10>,
  "src_behavior_score": <0-10>,
  "contrast_quality": <0-10>,
  "confounds_detected": ["confound1", ...],
  "reasoning": "Brief explanation of your evaluation"
}}'''


class LLMContrastValidator(ContrastValidatorProtocol):
    """Thorough LLM-based contrast validation.

    Uses an LLM judge to verify:
    - dst exhibits the target behavior
    - src does not exhibit the behavior
    - The contrast is clear and unambiguous
    - No confounding differences exist

    Example:
        >>> validator = LLMContrastValidator(judge_llm)
        >>> result = await validator.validate(pair, analysis)
        >>> print(f"dst score: {result.dst_behavior_score}")
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        min_dst_score: float = 7.0,
        max_src_score: float = 3.0,
        min_contrast_quality: float = 6.0,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ):
        """Initialize the LLM validator.

        Args:
            llm_client: LLM client for judging.
            min_dst_score: Minimum score for dst to pass.
            max_src_score: Maximum score for src to pass.
            min_contrast_quality: Minimum contrast quality to pass.
            temperature: Generation temperature (lower = more consistent).
            max_tokens: Maximum response tokens (None = provider default).
        """
        self._llm = llm_client
        self._min_dst = min_dst_score
        self._max_src = max_src_score
        self._min_contrast = min_contrast_quality
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def validate(
        self,
        pair: ContrastPair,
        analysis: BehaviorAnalysis,
    ) -> ValidationResult:
        """Validate contrast using LLM judge.

        Args:
            pair: The contrast pair to validate.
            analysis: Behavior analysis for context.

        Returns:
            ValidationResult with scores and validity.
        """
        prompt = VALIDATION_PROMPT.format(
            behavior_description=analysis.description,
            components=self._format_components(analysis),
            confounds=self._format_confounds(analysis),
            prompt=self._truncate(pair.prompt, 300),
            dst=self._truncate(pair.dst, 500),
            src=self._truncate(pair.src, 500),
        )

        try:
            response = await self._llm.complete(
                messages=[Message(role="user", content=prompt)],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )

            data = self._parse_response(response.content)

        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                dst_behavior_score=5.0,
                src_behavior_score=5.0,
                semantic_distance=-1,
                contrast_quality=5.0,
                confounds_detected=["validation_error"],
                reasoning=f"Validation error: {str(e)}",
            )

        dst_score = float(data.get("dst_behavior_score", 5.0))
        src_score = float(data.get("src_behavior_score", 5.0))
        contrast_quality = float(data.get("contrast_quality", 5.0))
        confounds = data.get("confounds_detected", [])
        reasoning = data.get("reasoning", "")

        # Ensure confounds is a list
        if isinstance(confounds, str):
            confounds = [confounds] if confounds else []

        # Determine validity
        is_valid = (
            dst_score >= self._min_dst
            and src_score <= self._max_src
            and contrast_quality >= self._min_contrast
            and len(confounds) == 0
        )

        return ValidationResult(
            is_valid=is_valid,
            dst_behavior_score=dst_score,
            src_behavior_score=src_score,
            semantic_distance=-1,  # Not evaluated by this validator
            contrast_quality=contrast_quality,
            confounds_detected=confounds,
            reasoning=reasoning,
        )

    def _format_components(self, analysis: BehaviorAnalysis) -> str:
        """Format behavior components for prompt."""
        if not analysis.components:
            return "No specific components defined"

        lines = []
        for c in analysis.components:
            lines.append(f"- {c.name}: {c.description}")
        return "\n".join(lines)

    def _format_confounds(self, analysis: BehaviorAnalysis) -> str:
        """Format confounds for prompt."""
        confounds = analysis.confounds_to_avoid or [
            "Response length",
            "Tone/politeness",
            "Helpfulness level",
            "Response quality",
        ]
        return "\n".join(f"- {c}" for c in confounds)

    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text to max length."""
        if len(text) <= max_length:
            return text
        return text[:max_length - 3] + "..."

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

        logger.warning("Failed to parse LLM validation response")
        return {}
