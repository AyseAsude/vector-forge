"""Behavioral signal and confound validation for contrast pairs.

This module provides validators that measure:
1. Behavioral Signal Strength - HOW STRONGLY a pair contrasts on the target behavior
2. Confound Control - HOW WELL non-behavioral dimensions are matched

These complement existing validation (dimension, marker, boundary, etc.) by measuring
signal QUALITY, not just correctness.
"""

from __future__ import annotations

import logging
from typing import Optional

from vector_forge.contrast.protocols import (
    ContrastPair,
    BehaviorAnalysis,
    ValidationResult,
    BehavioralSignalResult,
    ConfoundCheckResult,
)
from vector_forge.contrast.utils import safe_parse_llm_json
from vector_forge.core.protocols import Message
from vector_forge.llm import JSON_RESPONSE_FORMAT
from vector_forge.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


BEHAVIORAL_SIGNAL_PROMPT = '''You are measuring the STRENGTH of behavioral contrast in a training pair.

## BEHAVIOR
Name: {behavior_name}
Definition: {core_definition}

## THE DISTINGUISHING VARIABLE
{distinguishing_variable}

When behavior is PRESENT: {present_pattern}
When behavior is ABSENT: {absent_pattern}

---

## THE PAIR TO EVALUATE

**PROMPT:**
{prompt}

**DST (should EXHIBIT the behavior):**
{dst}

**SRC (should NOT exhibit the behavior):**
{src}

---

## YOUR TASK

Rate the BEHAVIORAL SIGNAL STRENGTH. We want pairs with MAXIMUM contrast on the target behavior.

### For DST:
How strongly does it exhibit the behavior?
- 10: Textbook example, unmistakably exhibits the behavior
- 7-9: Clearly exhibits, obvious on first read
- 4-6: Somewhat exhibits, noticeable but not strong
- 1-3: Barely exhibits, ambiguous

### For SRC:
How clearly does it NOT exhibit the behavior?
- 10: Completely absent, clearly does NOT exhibit
- 7-9: Clearly absent, obvious contrast to dst
- 4-6: Mostly absent, but some ambiguity
- 1-3: Still somewhat exhibits the behavior

### Signal Strength:
The minimum of dst_strength and src_strength determines overall signal.
A pair with dst=9 but src=3 (src still exhibits behavior) has weak signal.

---

Output JSON:
{{
  "dst_strength": <1-10>,
  "src_strength": <1-10>,
  "signal_strength": <1-10>,
  "reasoning": "<one sentence explaining the rating>"
}}'''


CONFOUND_CHECK_PROMPT = '''You are checking if a contrast pair is properly controlled for confounds.

## BEHAVIOR
The ONLY difference between dst and src should be: {distinguishing_variable}

All other aspects should be MATCHED to isolate the behavioral signal.

---

## THE PAIR TO CHECK

**PROMPT:**
{prompt}

**DST:**
{dst}

**SRC:**
{src}

---

## CHECK THESE CONFOUNDS

For each dimension, check if dst and src are matched:

1. **LENGTH**: Count approximate words in each
   - dst_words: (count)
   - src_words: (count)
   - length_ratio: dst_words / src_words (ideal: 0.8-1.2)

2. **FORMALITY**:
   - dst: casual / neutral / formal
   - src: casual / neutral / formal
   - formality_match: true/false

3. **HELPFULNESS**:
   - dst: low / medium / high
   - src: low / medium / high
   - helpfulness_match: true/false

4. **DETAIL LEVEL**:
   - dst: brief / moderate / thorough
   - src: brief / moderate / thorough
   - detail_match: true/false

5. **STRUCTURE**:
   - Same paragraph structure? Same use of lists?
   - structure_match: true/false

## SCORING

confound_score (1-10):
- 10: Perfectly matched on all non-behavioral dimensions
- 8-9: Minor differences that won't dominate the signal
- 6-7: Some noticeable differences, behavior still distinguishable
- 4-5: Significant confounds, hard to isolate behavior
- 1-3: Major confounds dominate, behavior signal is buried

---

Output JSON:
{{
  "dst_words": <number>,
  "src_words": <number>,
  "length_ratio": <float>,
  "formality_match": <true/false>,
  "helpfulness_match": <true/false>,
  "detail_match": <true/false>,
  "structure_match": <true/false>,
  "confound_score": <1-10>,
  "main_confound": "<primary issue if any, or 'none'>",
  "reasoning": "<one sentence>"
}}'''


class BehavioralSignalValidator:
    """Validates behavioral signal strength of contrast pairs.

    Measures HOW STRONGLY a pair contrasts on the target behavior,
    not just whether it contrasts on the right dimension.

    A pair can pass dimension validation (yes, it contrasts on sycophancy)
    but have weak signal (both responses are only slightly different).

    Example:
        >>> validator = BehavioralSignalValidator(llm_client)
        >>> result = await validator.validate(pair, analysis)
        >>> print(f"Signal: {result.signal_strength}")
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ):
        """Initialize the behavioral signal validator.

        Args:
            llm_client: LLM client for validation.
            temperature: Generation temperature (lower = more consistent).
            max_tokens: Maximum response tokens.
        """
        self._llm = llm_client
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def validate(
        self,
        pair: ContrastPair,
        analysis: BehaviorAnalysis,
    ) -> BehavioralSignalResult:
        """Validate behavioral signal strength of a pair.

        Args:
            pair: The contrast pair to validate.
            analysis: Behavior analysis for context.

        Returns:
            BehavioralSignalResult with strength scores.
        """
        # Get behavioral test info
        if analysis.behavioral_test:
            distinguishing_var = analysis.behavioral_test.distinguishing_variable
            present_pattern = analysis.behavioral_test.present_response_pattern
            absent_pattern = analysis.behavioral_test.absent_response_pattern
        else:
            distinguishing_var = "the target behavior"
            present_pattern = "exhibits the behavior"
            absent_pattern = "does not exhibit the behavior"

        prompt = BEHAVIORAL_SIGNAL_PROMPT.format(
            behavior_name=analysis.behavior_name,
            core_definition=analysis.core_definition or analysis.description,
            distinguishing_variable=distinguishing_var,
            present_pattern=present_pattern,
            absent_pattern=absent_pattern,
            prompt=pair.prompt[:800],
            dst=pair.dst[:1200],
            src=pair.src[:1200],
        )

        try:
            response = await self._llm.complete(
                messages=[Message(role="user", content=prompt)],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_format=JSON_RESPONSE_FORMAT,
            )

            data = safe_parse_llm_json(response.content)

            return BehavioralSignalResult(
                score=float(data.get("signal_strength", 5.0)),
                dst_strength=float(data.get("dst_strength", 5.0)),
                src_strength=float(data.get("src_strength", 5.0)),
                reasoning=data.get("reasoning", ""),
            )

        except Exception as e:
            logger.error(f"Behavioral signal validation failed: {e}")
            return BehavioralSignalResult(
                score=5.0,
                dst_strength=5.0,
                src_strength=5.0,
                reasoning=f"Validation error: {e}",
            )


class ConfoundValidator:
    """Validates confound control in contrast pairs.

    Checks that non-behavioral dimensions (length, tone, formality, etc.)
    are matched between dst and src, so the only signal is the behavior.

    Example:
        >>> validator = ConfoundValidator(llm_client)
        >>> result = await validator.validate(pair, analysis)
        >>> print(f"Confound score: {result.score}, Main issue: {result.main_confound}")
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ):
        """Initialize the confound validator.

        Args:
            llm_client: LLM client for validation.
            temperature: Generation temperature.
            max_tokens: Maximum response tokens.
        """
        self._llm = llm_client
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def validate(
        self,
        pair: ContrastPair,
        analysis: BehaviorAnalysis,
    ) -> ConfoundCheckResult:
        """Validate confound control of a pair.

        Args:
            pair: The contrast pair to validate.
            analysis: Behavior analysis for context.

        Returns:
            ConfoundCheckResult with confound scores.
        """
        # Get distinguishing variable
        if analysis.behavioral_test:
            distinguishing_var = analysis.behavioral_test.distinguishing_variable
        else:
            distinguishing_var = "the target behavior"

        prompt = CONFOUND_CHECK_PROMPT.format(
            distinguishing_variable=distinguishing_var,
            prompt=pair.prompt[:800],
            dst=pair.dst[:1200],
            src=pair.src[:1200],
        )

        try:
            response = await self._llm.complete(
                messages=[Message(role="user", content=prompt)],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_format=JSON_RESPONSE_FORMAT,
            )

            data = safe_parse_llm_json(response.content)

            # Calculate length ratio
            dst_words = int(data.get("dst_words", 1))
            src_words = int(data.get("src_words", 1))
            length_ratio = dst_words / max(src_words, 1)

            return ConfoundCheckResult(
                score=float(data.get("confound_score", 5.0)),
                length_ratio=length_ratio,
                formality_match=data.get("formality_match", True),
                helpfulness_match=data.get("helpfulness_match", True),
                detail_match=data.get("detail_match", True),
                structure_match=data.get("structure_match", True),
                main_confound=data.get("main_confound", ""),
                reasoning=data.get("reasoning", ""),
            )

        except Exception as e:
            logger.error(f"Confound validation failed: {e}")
            return ConfoundCheckResult(
                score=5.0,
                length_ratio=1.0,
                formality_match=True,
                helpfulness_match=True,
                detail_match=True,
                structure_match=True,
                main_confound="validation_error",
                reasoning=f"Validation error: {e}",
            )


class SignalQualityValidator:
    """Combined validator for behavioral signal and confound control.

    Runs both behavioral signal and confound validation in a single call
    for efficiency when both are needed.

    Example:
        >>> validator = SignalQualityValidator(llm_client)
        >>> result = await validator.validate(pair, analysis)
        >>> print(f"Signal: {result.behavioral_signal_score}, Confound: {result.confound_score}")
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ):
        """Initialize the signal quality validator.

        Args:
            llm_client: LLM client for validation.
            temperature: Generation temperature.
            max_tokens: Maximum response tokens.
        """
        self._signal_validator = BehavioralSignalValidator(
            llm_client, temperature, max_tokens
        )
        self._confound_validator = ConfoundValidator(
            llm_client, temperature, max_tokens
        )

    async def validate(
        self,
        pair: ContrastPair,
        analysis: BehaviorAnalysis,
    ) -> ValidationResult:
        """Validate signal quality (both behavioral signal and confounds).

        Args:
            pair: The contrast pair to validate.
            analysis: Behavior analysis for context.

        Returns:
            ValidationResult with signal and confound scores populated.
        """
        import asyncio

        # Run both validations concurrently
        signal_result, confound_result = await asyncio.gather(
            self._signal_validator.validate(pair, analysis),
            self._confound_validator.validate(pair, analysis),
        )

        # Determine overall validity based on signal quality
        # High signal + good confound control = valid
        is_valid = (
            signal_result.score >= 6.0 and
            confound_result.score >= 5.0
        )

        # Combine into ValidationResult
        return ValidationResult(
            is_valid=is_valid,
            contrast_quality=signal_result.score,
            reasoning=f"Signal:{signal_result.score:.0f} Confound:{confound_result.score:.0f}",
            behavioral_signal_score=signal_result.score,
            confound_score=confound_result.score,
            behavioral_signal_details=signal_result,
            confound_details=confound_result,
        )
