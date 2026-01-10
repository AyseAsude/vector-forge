"""Contrast pair regenerator for fixing failed pairs.

This module regenerates pairs that failed validation with
targeted feedback based on what went wrong.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from vector_forge.contrast.protocols import (
    PairRegeneratorProtocol,
    ContrastPair,
    ValidationResult,
    BehaviorAnalysis,
)
from vector_forge.core.protocols import Message
from vector_forge.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


REGENERATION_PROMPT = '''The previous contrast pair was insufficient. Generate an improved version.

## BEHAVIOR
{behavior_description}

## ORIGINAL PAIR

**Prompt:** "{original_prompt}"

**POSITIVE (should exhibit behavior):**
"{original_dst}"

**NEGATIVE (should NOT exhibit behavior):**
"{original_src}"

---

## PROBLEMS FOUND
{problems}

## REQUIRED IMPROVEMENTS
{improvements}

---

## REGENERATION INSTRUCTIONS

Based on the problems found, generate an IMPROVED pair.

**Attempt {attempt} of 3** - Be more aggressive with improvements.

{attempt_specific_instructions}

---

## REQUIREMENTS

1. The new POSITIVE must CLEARLY exhibit the behavior (score >= 7)
2. The new NEGATIVE must clearly NOT exhibit the behavior (score <= 3)
3. They must be semantically different (not just reworded)
4. Avoid confounds - keep length, tone, helpfulness similar

Return JSON:
{{
  "prompt": "The improved prompt (can reuse original if fine)",
  "dst": "The improved positive response",
  "src": "The improved negative response",
  "improvements_made": ["what you changed and why"],
  "expected_scores": {{
    "dst_behavior": <expected score>,
    "src_behavior": <expected score>,
    "contrast_quality": <expected score>
  }}
}}'''


ATTEMPT_INSTRUCTIONS = {
    1: """
**Standard improvement**: Make targeted fixes based on the problems identified.
- If dst was too weak: Make it more clearly exhibit the behavior
- If src was too strong: Make it more neutral or opposite
- If too similar: Increase the semantic difference
""",
    2: """
**Stronger improvement**: The first attempt wasn't enough. Be more aggressive.
- If dst was too weak: Make it STRONGLY exhibit the behavior, almost exaggerated
- If src was too strong: Make it actively OPPOSITE to the behavior
- If too similar: Change the approach entirely while keeping the same topic
- Consider changing the structure or style of responses
""",
    3: """
**Maximum contrast**: Previous attempts failed. Use extreme measures.
- Make dst the STRONGEST possible example of the behavior
- Make src the CLEAREST possible counter-example
- The difference should be immediately obvious to anyone
- If the scenario itself is problematic, adapt it
""",
}


class ContrastRegenerator(PairRegeneratorProtocol):
    """Regenerates pairs that failed validation.

    Uses targeted feedback based on validation results to fix issues
    with contrast pairs. Progressively becomes more aggressive with
    each regeneration attempt.

    Example:
        >>> regenerator = ContrastRegenerator(llm_client)
        >>> fixed_pair = await regenerator.regenerate(
        ...     pair, validation_result, analysis, attempt=1
        ... )
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        temperature: float = 0.8,
        max_tokens: Optional[int] = None,
    ):
        """Initialize the regenerator.

        Args:
            llm_client: LLM client for regeneration.
            temperature: Generation temperature.
            max_tokens: Maximum response tokens (None = provider default).
        """
        self._llm = llm_client
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def regenerate(
        self,
        pair: ContrastPair,
        validation: ValidationResult,
        analysis: BehaviorAnalysis,
        attempt: int,
    ) -> ContrastPair:
        """Regenerate a pair that failed validation.

        Args:
            pair: The original pair that failed.
            validation: Validation result with failure reasons.
            analysis: Behavior analysis for context.
            attempt: Which regeneration attempt this is (1-indexed).

        Returns:
            New ContrastPair with improvements.
        """
        problems, improvements = self._build_feedback(validation)

        # Get attempt-specific instructions
        attempt_instructions = ATTEMPT_INSTRUCTIONS.get(
            min(attempt, 3),
            ATTEMPT_INSTRUCTIONS[3]
        )

        prompt = REGENERATION_PROMPT.format(
            behavior_description=analysis.description,
            original_prompt=self._truncate(pair.prompt, 200),
            original_dst=self._truncate(pair.dst, 300),
            original_src=self._truncate(pair.src, 300),
            problems=problems,
            improvements=improvements,
            attempt=attempt,
            attempt_specific_instructions=attempt_instructions,
        )

        # Increase temperature slightly for later attempts
        temperature = min(1.0, self._temperature + (attempt - 1) * 0.1)

        try:
            response = await self._llm.complete(
                messages=[Message(role="user", content=prompt)],
                temperature=temperature,
                max_tokens=self._max_tokens,
            )

            data = self._parse_response(response.content)

        except Exception as e:
            logger.error(f"Regeneration failed: {e}")
            return pair  # Return original if regeneration fails

        # Use improved versions, falling back to originals
        new_prompt = data.get("prompt") or pair.prompt
        new_dst = data.get("dst") or pair.dst
        new_src = data.get("src") or pair.src

        return ContrastPair(
            prompt=new_prompt,
            dst=new_dst,
            src=new_src,
            seed=pair.seed,
            metadata={
                "regenerated": True,
                "attempt": attempt,
                "improvements_made": data.get("improvements_made", []),
                "expected_scores": data.get("expected_scores", {}),
                "original_validation": {
                    "dst_score": validation.dst_behavior_score,
                    "src_score": validation.src_behavior_score,
                    "contrast_quality": validation.contrast_quality,
                },
            },
        )

    def _build_feedback(
        self,
        validation: ValidationResult,
    ) -> tuple[str, str]:
        """Build specific feedback based on validation results.

        Returns:
            Tuple of (problems, improvements) strings.
        """
        problems: List[str] = []
        improvements: List[str] = []

        # Check dst behavior score
        if validation.dst_behavior_score >= 0 and validation.dst_behavior_score < 7:
            problems.append(
                f"POSITIVE too weak: behavior score = {validation.dst_behavior_score}/10 "
                f"(need >= 7)"
            )
            if validation.dst_behavior_score < 4:
                improvements.append(
                    "Make POSITIVE MUCH stronger - it barely shows the behavior"
                )
            else:
                improvements.append(
                    "Make POSITIVE more clearly exhibit the behavior"
                )

        # Check src behavior score
        if validation.src_behavior_score >= 0 and validation.src_behavior_score > 3:
            problems.append(
                f"NEGATIVE still shows behavior: score = {validation.src_behavior_score}/10 "
                f"(need <= 3)"
            )
            if validation.src_behavior_score > 6:
                improvements.append(
                    "Make NEGATIVE show the OPPOSITE tendency, not just less"
                )
            else:
                improvements.append(
                    "Make NEGATIVE more clearly NOT exhibit the behavior"
                )

        # Check semantic distance
        if 0 <= validation.semantic_distance < 0.3:
            problems.append(
                f"Too semantically similar: distance = {validation.semantic_distance:.2f} "
                f"(need >= 0.3)"
            )
            improvements.append(
                "Make responses more different - change approach, wording, or structure"
            )

        # Check contrast quality
        if validation.contrast_quality < 6:
            problems.append(
                f"Contrast not clear enough: quality = {validation.contrast_quality}/10 "
                f"(need >= 6)"
            )
            improvements.append(
                "Make the difference between responses immediately obvious"
            )

        # Check confounds
        if validation.confounds_detected:
            confound_str = ", ".join(validation.confounds_detected)
            problems.append(f"Confounding differences detected: {confound_str}")
            improvements.append(
                f"Fix confounds: make {confound_str} similar between responses"
            )

        # Format as strings
        problems_str = "\n".join(f"- {p}" for p in problems) or "- General quality issues"
        improvements_str = "\n".join(f"- {i}" for i in improvements) or "- Improve overall contrast"

        return problems_str, improvements_str

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

        logger.warning("Failed to parse regeneration response")
        return {}
