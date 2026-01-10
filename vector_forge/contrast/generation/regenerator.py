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


REGENERATION_PROMPT = '''The previous contrast pair failed validation. Generate an improved version.

## BEHAVIOR
{behavior_description}
{core_definition}

## WHAT THIS IS NOT (avoid confusing with these)
{negative_examples}

## ORIGINAL PAIR

**Prompt:** "{original_prompt}"

**POSITIVE (should exhibit behavior):**
"{original_dst}"

**NEGATIVE (should NOT exhibit behavior):**
"{original_src}"

---

## VALIDATION RESULTS

### PRIMARY ISSUE
{primary_issue}

### DIMENSION SCORES (all need >= 7 to pass)
{dimension_scores}

### SPECIFIC PROBLEMS
{problems}

### REGENERATION FOCUS
{regeneration_focus}

---

## REGENERATION INSTRUCTIONS

**Attempt {attempt} of 3**

{attempt_specific_instructions}

## CRITICAL REQUIREMENTS

1. **CONFOUND PARITY**: Length, tone, helpfulness must be MATCHED
   - Count words: both responses within 20% of each other
   - Same formality, warmth, structure

2. **BEHAVIOR PURITY**: ONLY the target behavior should differ
   - Don't mix in other behaviors
   - Clean, isolated signal

3. **NATURALNESS**: Responses must feel REAL, not artificial
   - No exaggerated behavior ("I LOVE this!!!")
   - Should pass as real deployment outputs

4. **CONTRAST SHARPNESS**: Clear presence in dst, clear absence in src
   - Anyone should immediately see the difference

---

Return JSON:
{{
  "prompt": "The improved prompt (can reuse original if fine)",
  "dst": "The improved positive response",
  "src": "The improved negative response",
  "improvements_made": ["what you changed and why"],
  "dimension_improvements": {{
    "confound_fix": "how you fixed confound issues",
    "purity_fix": "how you improved behavior purity",
    "naturalness_fix": "how you improved naturalness",
    "sharpness_fix": "how you improved contrast"
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
        problems = self._build_problems(validation)
        dimension_scores = self._extract_dimension_scores(validation)
        primary_issue, regeneration_focus = self._extract_focus(validation)

        # Get attempt-specific instructions
        attempt_instructions = ATTEMPT_INSTRUCTIONS.get(
            min(attempt, 3),
            ATTEMPT_INSTRUCTIONS[3]
        )

        prompt = REGENERATION_PROMPT.format(
            behavior_description=analysis.description,
            core_definition=f"Core: {analysis.core_definition}" if analysis.core_definition else "",
            negative_examples=self._format_negative_examples(analysis),
            original_prompt=self._truncate(pair.prompt, 300),
            original_dst=self._truncate(pair.dst, 500),
            original_src=self._truncate(pair.src, 500),
            primary_issue=primary_issue,
            dimension_scores=dimension_scores,
            problems=problems,
            regeneration_focus=regeneration_focus,
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

    def _build_problems(self, validation: ValidationResult) -> str:
        """Build specific problems list from validation results."""
        problems: List[str] = []

        # Check behavior scores
        if validation.dst_behavior_score >= 0 and validation.dst_behavior_score < 7:
            problems.append(
                f"POSITIVE too weak: behavior score = {validation.dst_behavior_score:.0f}/10"
            )

        if validation.src_behavior_score >= 0 and validation.src_behavior_score > 3:
            problems.append(
                f"NEGATIVE still shows behavior: score = {validation.src_behavior_score:.0f}/10"
            )

        # Check confounds
        if validation.confounds_detected:
            for confound in validation.confounds_detected[:5]:  # Limit to top 5
                problems.append(f"Confound: {confound}")

        return "\n".join(f"- {p}" for p in problems) or "- General quality issues"

    def _extract_dimension_scores(self, validation: ValidationResult) -> str:
        """Extract dimension scores from validation reasoning."""
        # Parse reasoning which contains scores like "Confound:8 Purity:7 Natural:6 Sharp:8"
        reasoning = validation.reasoning or ""

        # Try to extract scores from reasoning
        lines = []
        if "Confound:" in reasoning:
            try:
                parts = reasoning.split("|")[0].strip().split()
                for part in parts:
                    if ":" in part:
                        dim, score = part.split(":")
                        status = "✓" if float(score) >= 7 else "✗"
                        lines.append(f"- {dim}: {score}/10 {status}")
            except (ValueError, IndexError):
                pass

        if not lines:
            # Fallback to basic info
            lines.append(f"- Contrast Quality: {validation.contrast_quality:.0f}/10")
            lines.append(f"- dst Behavior: {validation.dst_behavior_score:.0f}/10")
            lines.append(f"- src Behavior: {validation.src_behavior_score:.0f}/10")

        return "\n".join(lines)

    def _extract_focus(self, validation: ValidationResult) -> tuple[str, str]:
        """Extract primary issue and regeneration focus from validation."""
        reasoning = validation.reasoning or ""

        # Parse reasoning for Issue and Focus
        primary_issue = "General quality issues"
        regeneration_focus = "Improve overall contrast quality"

        if "Issue:" in reasoning:
            try:
                issue_part = reasoning.split("Issue:")[1]
                if "|" in issue_part:
                    primary_issue = issue_part.split("|")[0].strip()
                else:
                    primary_issue = issue_part.strip()
            except IndexError:
                pass

        if "Focus:" in reasoning:
            try:
                focus_part = reasoning.split("Focus:")[1]
                if "|" in focus_part:
                    regeneration_focus = focus_part.split("|")[0].strip()
                else:
                    regeneration_focus = focus_part.strip()
            except IndexError:
                pass

        # If still default, infer from confounds
        if primary_issue == "General quality issues" and validation.confounds_detected:
            primary_issue = validation.confounds_detected[0]

        return primary_issue, regeneration_focus

    def _format_negative_examples(self, analysis: BehaviorAnalysis) -> str:
        """Format what this behavior is NOT."""
        if not analysis.not_this_behavior:
            return "No negative examples specified"

        lines = []
        for neg in analysis.not_this_behavior:
            lines.append(f"- NOT {neg.similar_behavior}: {neg.why_different}")
        return "\n".join(lines)

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
