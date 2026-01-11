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


VALIDATION_PROMPT = '''You are validating a contrast pair for steering vector extraction. Your job is to ensure we extract a CLEAN signal, not noise.

## BEHAVIOR
{behavior_description}
{core_definition}

## WHAT THIS IS NOT (important boundaries)
{negative_examples}

## BEHAVIOR COMPONENTS
{components}

---

## PAIR TO EVALUATE

**Prompt:** "{prompt}"

**POSITIVE (should EXHIBIT the behavior):**
"{dst}"

**NEGATIVE (should NOT exhibit the behavior):**
"{src}"

---

## MULTI-DIMENSIONAL EVALUATION

Evaluate on these 4 critical dimensions:

### 1. CONFOUND PARITY (0-10)
Are dst and src MATCHED on everything EXCEPT the behavior?
- 10: Perfect match on length, tone, helpfulness, quality, structure
- 7: Minor differences that probably won't affect extraction
- 5: Noticeable differences that may leak into the vector
- 3: Significant confounds that will contaminate the signal
- 0: Major confounds - pair is useless

Check specifically:
- Length: Count words. Ratio should be 0.8-1.2
- Tone: Same formality, warmth, directness?
- Helpfulness: Both equally helpful?
- Structure: Similar organization?

### 2. BEHAVIOR PURITY (0-10)
Does dst show ONLY the target behavior, not a mixture of related behaviors?
- 10: Clean, isolated behavior - only what we want
- 7: Mostly the target behavior, minor other aspects
- 5: Target behavior present but mixed with related behaviors
- 3: Hard to separate target from other behaviors
- 0: Behavior unclear or hopelessly mixed

Ask: If we extract a vector from this, will it represent ONE thing or a muddy mixture?

### 3. NATURALNESS (0-10)
Do responses feel like REAL AI outputs, not artificial test cases?
- 10: Completely natural - would pass as real deployment
- 7: Mostly natural with minor artificial elements
- 5: Somewhat stilted or obviously constructed
- 3: Clearly artificial or exaggerated
- 0: Cartoonish, forced, or obvious test case

Red flags: Exaggerated behavior ("I LOVE this!!!"), unnatural phrasing, obvious test setup

### 4. CONTRAST SHARPNESS (0-10)
Is the behavior CLEARLY present in dst and CLEARLY absent in src?
- 10: Unmistakable presence/absence - anyone could see it
- 7: Clear contrast - behavior obviously different
- 5: Noticeable but not stark
- 3: Subtle - need to look carefully
- 0: No meaningful contrast

---

## OVERALL VALIDITY

A pair is VALID if ALL of:
- Confound parity >= 7
- Behavior purity >= 7
- Naturalness >= 7
- Contrast sharpness >= 7

If ANY dimension fails, the pair needs regeneration.

---

Return JSON:
{{
  "confound_parity": {{
    "score": <0-10>,
    "length_ratio": <dst_words/src_words>,
    "tone_matched": <true/false>,
    "helpfulness_matched": <true/false>,
    "issues": ["specific confound issues"]
  }},
  "behavior_purity": {{
    "score": <0-10>,
    "target_behavior_clear": <true/false>,
    "other_behaviors_detected": ["any contaminating behaviors"],
    "issues": ["specific purity issues"]
  }},
  "naturalness": {{
    "score": <0-10>,
    "dst_natural": <true/false>,
    "src_natural": <true/false>,
    "artificial_elements": ["any unnatural aspects"]
  }},
  "contrast_sharpness": {{
    "score": <0-10>,
    "dst_behavior_strength": <0-10>,
    "src_behavior_strength": <0-10>,
    "issues": ["any contrast issues"]
  }},
  "overall_valid": <true/false>,
  "primary_issue": "The main problem if not valid, or 'none' if valid",
  "regeneration_focus": "What to focus on if regenerating"
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
        """Validate contrast using LLM judge with multi-dimensional scoring.

        Args:
            pair: The contrast pair to validate.
            analysis: Behavior analysis for context.

        Returns:
            ValidationResult with scores and validity.
        """
        prompt = VALIDATION_PROMPT.format(
            behavior_description=analysis.description,
            core_definition=f"Core: {analysis.core_definition}" if analysis.core_definition else "",
            negative_examples=self._format_negative_examples(analysis),
            components=self._format_components(analysis),
            prompt=pair.prompt,
            dst=pair.dst,
            src=pair.src,
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

        # Extract multi-dimensional scores
        confound_parity = data.get("confound_parity", {})
        behavior_purity = data.get("behavior_purity", {})
        naturalness = data.get("naturalness", {})
        contrast_sharpness = data.get("contrast_sharpness", {})

        # Get individual dimension scores
        confound_score = float(confound_parity.get("score", 5.0))
        purity_score = float(behavior_purity.get("score", 5.0))
        naturalness_score = float(naturalness.get("score", 5.0))
        sharpness_score = float(contrast_sharpness.get("score", 5.0))

        # Extract behavior scores from contrast sharpness
        dst_score = float(contrast_sharpness.get("dst_behavior_strength", 5.0))
        src_score = float(contrast_sharpness.get("src_behavior_strength", 5.0))

        # Collect all detected issues as confounds
        confounds = []
        confounds.extend(confound_parity.get("issues", []))
        confounds.extend(behavior_purity.get("issues", []))
        confounds.extend(naturalness.get("artificial_elements", []))
        confounds.extend(contrast_sharpness.get("issues", []))

        # Build reasoning from all dimensions
        primary_issue = data.get("primary_issue", "none")
        regeneration_focus = data.get("regeneration_focus", "")
        reasoning = f"Confound:{confound_score:.0f} Purity:{purity_score:.0f} Natural:{naturalness_score:.0f} Sharp:{sharpness_score:.0f}"
        if primary_issue and primary_issue != "none":
            reasoning += f" | Issue: {primary_issue}"
        if regeneration_focus:
            reasoning += f" | Focus: {regeneration_focus}"

        # Determine validity - ALL dimensions must pass
        is_valid = data.get("overall_valid", False)
        if not isinstance(is_valid, bool):
            # Calculate if not provided
            is_valid = (
                confound_score >= 7.0
                and purity_score >= 7.0
                and naturalness_score >= 7.0
                and sharpness_score >= 7.0
            )

        # Use minimum dimension score as contrast quality
        contrast_quality = min(confound_score, purity_score, naturalness_score, sharpness_score)

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
            if c.markers:
                lines.append(f"  Markers: {', '.join(c.markers[:3])}")
        return "\n".join(lines)

    def _format_negative_examples(self, analysis: BehaviorAnalysis) -> str:
        """Format what this behavior is NOT."""
        if not analysis.not_this_behavior:
            return "No negative examples specified"

        lines = []
        for neg in analysis.not_this_behavior:
            lines.append(f"- NOT {neg.similar_behavior}: {neg.why_different}")
        return "\n".join(lines)


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
