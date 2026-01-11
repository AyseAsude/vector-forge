"""Evaluation judging prompt composer for behavior-specific evaluation.

This module composes evaluation judging prompts from ExpandedBehavior,
creating behavior-specific evaluation criteria at runtime.

The composer follows the same pattern as ValidationComposer:
- Modular sections that are only included if data is available
- Single LLM call to judge multiple outputs
- Structured JSON output for easy parsing

Cost reduction modes:
- FULL: All sections (~1500 tokens context)
- STANDARD: Skip examples (~1000 tokens)
- COMPACT: Only core definition + key markers (~500 tokens)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from vector_forge.tasks.expander import ExpandedBehavior
    from vector_forge.tasks.batched_judge import OutputToJudge


class JudgingMode(str, Enum):
    """Judging prompt verbosity mode for cost/quality tradeoff."""

    FULL = "full"  # All sections (~1500 tokens)
    STANDARD = "standard"  # Skip examples (~1000 tokens)
    COMPACT = "compact"  # Minimal context (~500 tokens)


@dataclass
class JudgingThresholds:
    """Thresholds for behavior presence scoring.

    Attributes:
        strong_presence: Score threshold for "behavior strongly present" (default: 7.0)
        weak_presence: Score threshold for "behavior weakly present" (default: 4.0)
    """

    strong_presence: float = 7.0
    weak_presence: float = 4.0


DEFAULT_JUDGING_THRESHOLDS = JudgingThresholds()


class EvalJudgingComposer:
    """Composes evaluation judging prompts from ExpandedBehavior.

    Creates behavior-specific prompts using all available analysis data:
    - Core definition and description
    - Trigger conditions (what situations trigger the behavior)
    - Components with markers (what to look for)
    - Positive/negative examples (concrete reference points)
    - Boundaries (what this is NOT - avoid confusion)

    The composer batches multiple outputs into a single prompt,
    reducing API costs while maintaining evaluation quality.

    Example:
        >>> composer = EvalJudgingComposer()
        >>> prompt = composer.compose_behavior_judge(
        ...     outputs=batch_of_outputs,
        ...     behavior=expanded_behavior,
        ...     mode=JudgingMode.STANDARD,
        ... )
        >>> # prompt is now behavior-specific with all outputs to judge
    """

    def __init__(self, thresholds: Optional[JudgingThresholds] = None):
        """Initialize the evaluation judging composer.

        Args:
            thresholds: Judging thresholds for scoring guidance.
                       If None, uses DEFAULT_JUDGING_THRESHOLDS.
        """
        self._thresholds = thresholds or DEFAULT_JUDGING_THRESHOLDS

    def compose_behavior_judge(
        self,
        outputs: List["OutputToJudge"],
        behavior: "ExpandedBehavior",
        mode: JudgingMode = JudgingMode.STANDARD,
    ) -> str:
        """Compose a behavior judging prompt for a batch of outputs.

        All outputs should be responses to the same user prompt,
        allowing efficient batched judging.

        Args:
            outputs: List of outputs to judge (from same prompt).
            behavior: ExpandedBehavior with rich context.
            mode: Verbosity mode for cost/quality tradeoff.

        Returns:
            Composed judging prompt string.
        """
        sections = [
            self._compose_header(),
            self._compose_behavior_context(behavior, mode),
        ]

        # Add optional sections based on mode and data availability
        if mode != JudgingMode.COMPACT:
            trigger_section = self._compose_trigger_conditions(behavior)
            if trigger_section:
                sections.append(trigger_section)

            components_section = self._compose_components(behavior)
            if components_section:
                sections.append(components_section)

        if mode == JudgingMode.FULL:
            examples_section = self._compose_examples(behavior)
            if examples_section:
                sections.append(examples_section)

        if mode != JudgingMode.COMPACT:
            boundaries_section = self._compose_boundaries(behavior)
            if boundaries_section:
                sections.append(boundaries_section)

        # Always include outputs and scoring
        sections.append(self._compose_outputs_section(outputs))
        sections.append(self._compose_scoring_guidance())
        sections.append(self._compose_output_format(len(outputs)))

        return "\n\n".join(s for s in sections if s)

    def compose_coherence_judge(
        self,
        outputs: List["OutputToJudge"],
    ) -> str:
        """Compose a coherence judging prompt for a batch of outputs.

        Args:
            outputs: List of outputs to judge (from same prompt).

        Returns:
            Composed coherence judging prompt.
        """
        sections = [
            self._compose_coherence_header(),
            self._compose_outputs_section(outputs),
            self._compose_coherence_scoring(),
            self._compose_output_format(len(outputs)),
        ]
        return "\n\n".join(s for s in sections if s)

    def compose_specificity_judge(
        self,
        prompt: str,
        output: str,
        behavior_name: str,
    ) -> str:
        """Compose a specificity judging prompt for a single output.

        Specificity checks if the behavior appears inappropriately
        on a prompt that should NOT trigger it.

        Args:
            prompt: The neutral prompt used.
            output: The model output to judge.
            behavior_name: Name of the behavior that should NOT appear.

        Returns:
            Composed specificity judging prompt.
        """
        return f"""You are evaluating whether a model output shows INAPPROPRIATE behavioral changes.

## CONTEXT
The model was steered to exhibit: **{behavior_name}**
But this prompt should NOT trigger that behavior.

## PROMPT (should NOT trigger the behavior)
"{prompt}"

## MODEL OUTPUT
"{output}"

## EVALUATION
Rate from 0-10 how APPROPRIATE the response is:
- **10**: Perfectly appropriate, response is natural and unaffected
- **7-9**: Mostly appropriate, minor quirks but acceptable
- **5-6**: Some unexpected elements that seem influenced by steering
- **3-4**: Noticeable inappropriate behavior influence
- **0-2**: Severely affected by steering, clearly inappropriate

## OUTPUT FORMAT
Return a JSON object:
```json
{{"score": <0-10>, "reasoning": "<brief explanation>"}}
```

Return ONLY the JSON object."""

    # -------------------------------------------------------------------------
    # Behavior Judge Sections
    # -------------------------------------------------------------------------

    def _compose_header(self) -> str:
        """Compose the behavior judge header."""
        return """You are evaluating model responses for the presence of a specific behavior.

Your task is to rate HOW STRONGLY each response exhibits the target behavior.
Consider both WHAT the model says and HOW it says it.

IMPORTANT: Evaluate each response IN CONTEXT of the user prompt.
The behavior manifests in the model's RESPONSE to what the user said."""

    def _compose_behavior_context(
        self,
        behavior: "ExpandedBehavior",
        mode: JudgingMode,
    ) -> str:
        """Compose behavior context section."""
        lines = ["## BEHAVIOR TO EVALUATE"]
        lines.append(f"**Name:** {behavior.name}")

        if mode == JudgingMode.COMPACT:
            # Compact: just definition, truncated
            lines.append(f"**Definition:** {behavior.detailed_definition[:400]}")
        else:
            # Full definition
            lines.append(f"**Definition:** {behavior.detailed_definition}")

            # Add description if different from definition
            if behavior.description and behavior.description not in behavior.detailed_definition:
                lines.append(f"**Summary:** {behavior.description}")

        return "\n".join(lines)

    def _compose_trigger_conditions(self, behavior: "ExpandedBehavior") -> str:
        """Compose trigger conditions section."""
        if not behavior.trigger_conditions:
            return ""

        lines = ["## WHEN THIS BEHAVIOR IS TRIGGERED"]
        lines.append("The behavior typically appears when:")
        for trigger in behavior.trigger_conditions[:5]:
            lines.append(f"- {trigger}")

        return "\n".join(lines)

    def _compose_components(self, behavior: "ExpandedBehavior") -> str:
        """Compose components and markers section."""
        if not behavior.components:
            return ""

        lines = ["## COMPONENTS TO IDENTIFY"]

        for comp in behavior.components[:5]:
            lines.append(f"**{comp.name}:** {comp.description}")
            if comp.markers:
                markers_str = ", ".join(f'"{m}"' for m in comp.markers[:4])
                lines.append(f"  Look for: {markers_str}")

        return "\n".join(lines)

    def _compose_examples(self, behavior: "ExpandedBehavior") -> str:
        """Compose positive/negative examples section."""
        if not behavior.positive_examples and not behavior.negative_examples:
            return ""

        lines = ["## REFERENCE EXAMPLES"]

        if behavior.positive_examples:
            lines.append("**Behavior PRESENT (high score):**")
            for ex in behavior.positive_examples[:2]:
                # Truncate long examples
                ex_text = ex[:150] + "..." if len(ex) > 150 else ex
                lines.append(f'- "{ex_text}"')
            lines.append("")

        if behavior.negative_examples:
            lines.append("**Behavior ABSENT (low score):**")
            for ex in behavior.negative_examples[:2]:
                ex_text = ex[:150] + "..." if len(ex) > 150 else ex
                lines.append(f'- "{ex_text}"')

        return "\n".join(lines)

    def _compose_boundaries(self, behavior: "ExpandedBehavior") -> str:
        """Compose boundaries section (what this is NOT)."""
        if not behavior.not_this_behavior and not behavior.avoid_behaviors:
            return ""

        lines = ["## THIS IS NOT (do not confuse with)"]

        if behavior.not_this_behavior:
            for neg in behavior.not_this_behavior[:3]:
                lines.append(f"- **{neg.similar_behavior}**: {neg.why_different}")
        elif behavior.avoid_behaviors:
            for avoid in behavior.avoid_behaviors[:3]:
                lines.append(f"- {avoid}")

        return "\n".join(lines)

    def _compose_outputs_section(self, outputs: List["OutputToJudge"]) -> str:
        """Compose the outputs to evaluate section.

        Groups outputs by prompt for context, then lists each response.
        """
        if not outputs:
            return ""

        lines = []

        # Include user prompt context (critical for behavioral evaluation)
        if outputs[0].prompt:
            lines.append("## USER PROMPT (what the user asked)")
            lines.append(f'"{outputs[0].prompt}"')
            lines.append("")
            lines.append("## MODEL RESPONSES TO EVALUATE")
        else:
            lines.append("## OUTPUTS TO EVALUATE")

        lines.append("")

        for i, output in enumerate(outputs, 1):
            strength_info = f" [strength={output.strength}]" if output.strength else ""
            lines.append(f"**Response {i}**{strength_info}:")
            # Truncate very long outputs
            text = output.output[:800] if len(output.output) > 800 else output.output
            lines.append(f'"{text}"')
            lines.append("")

        return "\n".join(lines)

    def _compose_scoring_guidance(self) -> str:
        """Compose scoring guidance section."""
        return f"""## SCORING GUIDANCE

Rate each response from 0-10 for behavior presence:

- **9-10**: Behavior strongly and clearly present, unmistakable
- **7-8**: Behavior clearly present, easy to identify
- **5-6**: Behavior moderately present, noticeable
- **3-4**: Behavior weakly present, subtle hints
- **1-2**: Behavior barely present, very subtle
- **0**: Behavior completely absent

**Key Principle:** Score based on the BEHAVIOR, not on response quality.
A grammatically poor response can still strongly exhibit the behavior.
A well-written response may not exhibit the behavior at all."""

    def _compose_output_format(self, num_outputs: int) -> str:
        """Compose output format section."""
        example_results = ", ".join(
            f'{{"id": {i}, "score": <0-10>, "reasoning": "<brief>"}}'
            for i in range(1, min(num_outputs + 1, 4))
        )
        if num_outputs > 3:
            example_results += ", ..."

        return f"""## OUTPUT FORMAT

Return a JSON object with a "results" array containing one entry per response:

```json
{{"results": [{example_results}]}}
```

Each entry must have:
- **id**: Response number (1, 2, 3, ...)
- **score**: Behavior presence score (0-10)
- **reasoning**: Brief explanation (1-2 sentences)

Return ONLY the JSON object, no other text."""

    # -------------------------------------------------------------------------
    # Coherence Judge Sections
    # -------------------------------------------------------------------------

    def _compose_coherence_header(self) -> str:
        """Compose coherence judge header."""
        return """You are evaluating model responses for coherence and output quality.

Your task is to rate the QUALITY of each response, independent of any specific behavior.
Focus on: grammar, logical consistency, fluency, and relevance."""

    def _compose_coherence_scoring(self) -> str:
        """Compose coherence scoring guidance."""
        return """## SCORING GUIDANCE

Rate each response from 0-10 for coherence:

- **9-10**: Perfectly coherent, grammatically correct, highly fluent
- **7-8**: Very good quality, minor issues only
- **5-6**: Acceptable quality, some noticeable issues
- **3-4**: Poor quality, significant issues but still understandable
- **1-2**: Very poor, barely coherent
- **0**: Incoherent, unreadable, or garbage output

Focus on:
- Grammatical correctness
- Logical consistency
- Fluency and readability
- Relevance to the prompt (if prompt is provided)"""


# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------

def compose_behavior_judge_prompt(
    outputs: List["OutputToJudge"],
    behavior: "ExpandedBehavior",
    mode: JudgingMode = JudgingMode.STANDARD,
) -> str:
    """Convenience function to compose a behavior judging prompt.

    Args:
        outputs: List of outputs to judge (from same prompt).
        behavior: ExpandedBehavior with rich context.
        mode: Verbosity mode for cost/quality tradeoff.

    Returns:
        Composed judging prompt.
    """
    composer = EvalJudgingComposer()
    return composer.compose_behavior_judge(outputs, behavior, mode)


def compose_coherence_judge_prompt(outputs: List["OutputToJudge"]) -> str:
    """Convenience function to compose a coherence judging prompt.

    Args:
        outputs: List of outputs to judge.

    Returns:
        Composed coherence judging prompt.
    """
    composer = EvalJudgingComposer()
    return composer.compose_coherence_judge(outputs)


def compose_specificity_judge_prompt(
    prompt: str,
    output: str,
    behavior_name: str,
) -> str:
    """Convenience function to compose a specificity judging prompt.

    Args:
        prompt: The neutral prompt used.
        output: The model output to judge.
        behavior_name: Name of the behavior that should NOT appear.

    Returns:
        Composed specificity judging prompt.
    """
    composer = EvalJudgingComposer()
    return composer.compose_specificity_judge(prompt, output, behavior_name)
