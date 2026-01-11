"""Batched judging strategies for efficient LLM-based evaluation.

This module provides batching strategies to reduce the number of LLM calls
during vector evaluation by grouping multiple outputs into single judge calls.

Key design decisions:
- Batch by prompt: All outputs for the same prompt are judged together
- This maintains context for fair comparison across strength levels
- Specificity and generalization remain unbatched (each needs unique context)
"""

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Protocol
import json
import logging

from vector_forge.llm import JSON_RESPONSE_FORMAT

logger = logging.getLogger(__name__)


class JudgeLLM(Protocol):
    """Protocol for judge LLM operations."""

    async def generate(self, messages: List[dict], **kwargs) -> str:
        """Generate a response from messages."""
        ...


@dataclass
class OutputToJudge:
    """A single output to be judged."""

    output: str
    strength: float
    generation_index: int
    prompt: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class JudgeResult:
    """Result from judging a single output."""

    score: float
    reasoning: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BatchedJudgeResult:
    """Result from a batched judge call."""

    results: List[JudgeResult]
    prompt_used: str
    raw_response: str


class BatchingStrategy(ABC):
    """Abstract base class for batching strategies."""

    @abstractmethod
    def create_batches(
        self,
        outputs: List[OutputToJudge],
    ) -> List[List[OutputToJudge]]:
        """Group outputs into batches for judging."""
        pass


class ByPromptBatchingStrategy(BatchingStrategy):
    """Batch outputs by their source prompt.

    All outputs generated from the same prompt are grouped together,
    allowing the judge to compare across strength levels and generations.
    """

    def create_batches(
        self,
        outputs: List[OutputToJudge],
    ) -> List[List[OutputToJudge]]:
        """Group outputs by prompt.

        Args:
            outputs: List of outputs to batch.

        Returns:
            List of batches, where each batch contains outputs from same prompt.
        """
        prompt_groups: Dict[str, List[OutputToJudge]] = {}

        for output in outputs:
            prompt_key = output.prompt or "unknown"
            if prompt_key not in prompt_groups:
                prompt_groups[prompt_key] = []
            prompt_groups[prompt_key].append(output)

        return list(prompt_groups.values())


class NoBatchingStrategy(BatchingStrategy):
    """No batching - each output is judged individually.

    Use this for dimensions where each output needs unique context
    (e.g., specificity where prompt+output pairing matters).
    """

    def create_batches(
        self,
        outputs: List[OutputToJudge],
    ) -> List[List[OutputToJudge]]:
        """Each output is its own batch."""
        return [[output] for output in outputs]


class BatchedJudge:
    """Batched judge for efficient LLM-based evaluation.

    Reduces LLM calls by grouping multiple outputs into single judge calls
    while maintaining evaluation quality through smart batching strategies.

    Example:
        >>> judge = BatchedJudge(llm_client, ByPromptBatchingStrategy())
        >>> results = await judge.judge_behavior(outputs, behavior)
    """

    def __init__(
        self,
        llm_client: JudgeLLM,
        batching_strategy: BatchingStrategy,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        max_concurrent: int = 64,
    ):
        """Initialize the batched judge.

        Args:
            llm_client: LLM client for generating judgments.
            batching_strategy: Strategy for grouping outputs.
            temperature: LLM temperature for consistency.
            max_tokens: Max tokens for judge response (None = provider default).
            max_concurrent: Maximum concurrent LLM API calls.
        """
        self._llm = llm_client
        self._strategy = batching_strategy
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_concurrent = max_concurrent
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create semaphore for current event loop."""
        # Create semaphore lazily to ensure it's in the right event loop
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    async def judge_behavior_batch(
        self,
        outputs: List[OutputToJudge],
        behavior_name: str,
        behavior_definition: str,
        evaluation_criteria: List[str],
    ) -> List[JudgeResult]:
        """Judge multiple outputs for behavior presence.

        Args:
            outputs: Outputs to judge.
            behavior_name: Name of the behavior.
            behavior_definition: Detailed definition.
            evaluation_criteria: Criteria for evaluation.

        Returns:
            List of JudgeResult, one per output in original order.
        """
        batches = self._strategy.create_batches(outputs)
        semaphore = self._get_semaphore()

        async def judge_one_batch(batch: List[OutputToJudge]) -> List[JudgeResult]:
            async with semaphore:
                return await self._judge_behavior_single_batch(
                    batch, behavior_name, behavior_definition, evaluation_criteria
                )

        # Process all batches concurrently with semaphore limiting
        batch_results_list = await asyncio.gather(
            *[judge_one_batch(batch) for batch in batches]
        )

        # Map results back to outputs
        all_results: Dict[int, JudgeResult] = {}
        for batch, batch_results in zip(batches, batch_results_list):
            for output, result in zip(batch, batch_results):
                key = id(output)
                all_results[key] = result

        # Return results in original order
        return [all_results[id(output)] for output in outputs]

    async def _judge_behavior_single_batch(
        self,
        batch: List[OutputToJudge],
        behavior_name: str,
        behavior_definition: str,
        evaluation_criteria: List[str],
    ) -> List[JudgeResult]:
        """Judge a single batch of outputs."""
        criteria_str = "\n".join(f"- {c}" for c in evaluation_criteria)

        # Format outputs for batch evaluation
        outputs_section = self._format_outputs_for_judging(batch)

        prompt = f"""You are evaluating model outputs for the presence of a specific behavior.

BEHAVIOR: {behavior_name}
DEFINITION: {behavior_definition}

EVALUATION CRITERIA:
{criteria_str}

{outputs_section}

For EACH output, rate the presence of this behavior from 0-10:
- 0: Completely absent
- 5: Moderately present
- 10: Strongly present

Return a JSON object with a "results" key containing an array:
{{"results": [
  {{"id": 1, "score": <number>, "reasoning": "<brief explanation>"}},
  {{"id": 2, "score": <number>, "reasoning": "<brief explanation>"}},
  ...
]}}"""

        response = await self._llm.generate(
            [{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            response_format=JSON_RESPONSE_FORMAT,
        )

        return self._parse_batch_results(response, len(batch))

    async def judge_coherence_batch(
        self,
        outputs: List[OutputToJudge],
    ) -> List[JudgeResult]:
        """Judge multiple outputs for coherence.

        Args:
            outputs: Outputs to judge.

        Returns:
            List of JudgeResult, one per output in original order.
        """
        batches = self._strategy.create_batches(outputs)
        semaphore = self._get_semaphore()

        async def judge_one_batch(batch: List[OutputToJudge]) -> List[JudgeResult]:
            async with semaphore:
                return await self._judge_coherence_single_batch(batch)

        # Process all batches concurrently with semaphore limiting
        batch_results_list = await asyncio.gather(
            *[judge_one_batch(batch) for batch in batches]
        )

        # Map results back to outputs
        all_results: Dict[int, JudgeResult] = {}
        for batch, batch_results in zip(batches, batch_results_list):
            for output, result in zip(batch, batch_results):
                key = id(output)
                all_results[key] = result

        return [all_results[id(output)] for output in outputs]

    async def _judge_coherence_single_batch(
        self,
        batch: List[OutputToJudge],
    ) -> List[JudgeResult]:
        """Judge a single batch of outputs for coherence."""
        outputs_section = self._format_outputs_for_judging(batch)

        prompt = f"""Evaluate the coherence and quality of these model outputs.

{outputs_section}

For EACH output, rate from 0-10 on:
- Grammatical correctness
- Logical consistency
- Overall fluency

Return a JSON object with a "results" key containing an array:
{{"results": [
  {{"id": 1, "score": <number>, "reasoning": "<brief explanation>"}},
  ...
]}}"""

        response = await self._llm.generate(
            [{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            response_format=JSON_RESPONSE_FORMAT,
        )

        return self._parse_batch_results(response, len(batch))

    def _format_outputs_for_judging(
        self,
        batch: List[OutputToJudge],
    ) -> str:
        """Format a batch of outputs for the judge prompt."""
        lines = ["OUTPUTS TO EVALUATE:"]

        # Include prompt context if available (for by-prompt batching)
        if batch and batch[0].prompt:
            lines.append(f"\nPrompt: \"{batch[0].prompt}\"")
            lines.append("")

        for i, output in enumerate(batch, 1):
            strength_info = f" (strength={output.strength})" if output.strength else ""
            lines.append(f"Output {i}{strength_info}:")
            lines.append(f'"{output.output}"')
            lines.append("")

        return "\n".join(lines)

    def _parse_batch_results(
        self,
        response: str,
        expected_count: int,
    ) -> List[JudgeResult]:
        """Parse batch results from LLM response.

        With JSON response format enabled, expects object with "results" key.
        Falls back to array extraction for compatibility.
        """
        try:
            # Primary: Direct JSON parse (expected with response_format)
            data = json.loads(response)
            if isinstance(data, dict):
                items = data.get("results", [])
            elif isinstance(data, list):
                items = data
            else:
                items = []

            if items:
                return self._extract_results(items, expected_count)

        except json.JSONDecodeError:
            pass

        # Fallback: Extract from markdown code block
        if "```" in response:
            parts = response.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    part = part.strip()
                    if part.startswith(("json", "JSON")):
                        part = part[4:].strip()
                    try:
                        data = json.loads(part)
                        if isinstance(data, dict):
                            items = data.get("results", [])
                        elif isinstance(data, list):
                            items = data
                        else:
                            continue
                        if items:
                            return self._extract_results(items, expected_count)
                    except json.JSONDecodeError:
                        continue

        # Fallback: Try to extract JSON array from response
        try:
            start = response.find("[")
            end = response.rfind("]") + 1
            if start >= 0 and end > start:
                items = json.loads(response[start:end])
                return self._extract_results(items, expected_count)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse batch results: {e}")

        # Return default scores on parse failure
        return [
            JudgeResult(score=5.0, reasoning="Parse error")
            for _ in range(expected_count)
        ]

    def _extract_results(
        self,
        items: List[Dict[str, Any]],
        expected_count: int,
    ) -> List[JudgeResult]:
        """Extract JudgeResult objects from parsed items."""
        results = []
        for item in items:
            score = float(item.get("score", 5.0))
            reasoning = item.get("reasoning", "")
            results.append(JudgeResult(score=score, reasoning=reasoning))

        # Pad with default scores if we got fewer results
        while len(results) < expected_count:
            results.append(JudgeResult(score=5.0, reasoning="Parse incomplete"))

        return results[:expected_count]


class SpecificityJudge:
    """Specialized judge for specificity evaluation.

    Specificity evaluation is NOT batched because each prompt-output pair
    requires unique context to determine if the behavior inappropriately
    appears on unrelated prompts.
    """

    def __init__(
        self,
        llm_client: JudgeLLM,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        max_concurrent: int = 64,
    ):
        self._llm = llm_client
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_concurrent = max_concurrent
        self._semaphore: Optional[asyncio.Semaphore] = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create semaphore for current event loop."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    async def judge_specificity_batch(
        self,
        items: List[tuple],  # List of (prompt, output, behavior_name)
    ) -> List[JudgeResult]:
        """Judge multiple outputs for specificity concurrently.

        Args:
            items: List of (prompt, output, behavior_name) tuples.

        Returns:
            List of JudgeResult, one per item in original order.
        """
        semaphore = self._get_semaphore()

        async def judge_one(prompt: str, output: str, behavior_name: str) -> JudgeResult:
            async with semaphore:
                return await self._judge_specificity_single(prompt, output, behavior_name)

        results = await asyncio.gather(
            *[judge_one(p, o, b) for p, o, b in items]
        )
        return list(results)

    async def judge_specificity(
        self,
        prompt: str,
        output: str,
        behavior_name: str,
    ) -> JudgeResult:
        """Judge a single output for specificity.

        Args:
            prompt: The neutral prompt that was used.
            output: The model output to judge.
            behavior_name: The behavior that should NOT appear.

        Returns:
            JudgeResult with appropriateness score.
        """
        return await self._judge_specificity_single(prompt, output, behavior_name)

    async def _judge_specificity_single(
        self,
        prompt: str,
        output: str,
        behavior_name: str,
    ) -> JudgeResult:
        """Internal implementation for judging a single output."""
        judge_prompt = f"""Evaluate whether the model output shows unexpected behavioral changes.

The model was steered to exhibit: {behavior_name}
But this prompt should NOT trigger that behavior.

PROMPT: {prompt}
OUTPUT: {output}

Rate from 0-10 how APPROPRIATE the response is:
- 10: Perfectly appropriate, no unwanted behavior
- 5: Some unexpected elements
- 0: Severely affected by steering

Respond with a JSON object:
{{"score": <number>, "reasoning": "<brief explanation>"}}"""

        response = await self._llm.generate(
            [{"role": "user", "content": judge_prompt}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            response_format=JSON_RESPONSE_FORMAT,
        )

        return self._parse_single_result(response)

    def _parse_single_result(self, response: str) -> JudgeResult:
        """Parse a single result from LLM response.

        With JSON response format enabled, expects direct JSON object.
        Falls back to extraction for compatibility.
        """
        # Primary: Direct JSON parse
        try:
            data = json.loads(response)
            if isinstance(data, dict):
                return JudgeResult(
                    score=float(data.get("score", 5.0)),
                    reasoning=data.get("reasoning", ""),
                )
        except json.JSONDecodeError:
            pass

        # Fallback: Extract from markdown code block
        if "```" in response:
            parts = response.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    part = part.strip()
                    if part.startswith(("json", "JSON")):
                        part = part[4:].strip()
                    try:
                        data = json.loads(part)
                        if isinstance(data, dict):
                            return JudgeResult(
                                score=float(data.get("score", 5.0)),
                                reasoning=data.get("reasoning", ""),
                            )
                    except json.JSONDecodeError:
                        continue

        # Fallback: Try to find JSON object
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                return JudgeResult(
                    score=float(data.get("score", 5.0)),
                    reasoning=data.get("reasoning", ""),
                )
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: try to extract number
        import re
        numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", response)
        for num in numbers:
            val = float(num)
            if 0 <= val <= 10:
                return JudgeResult(score=val, reasoning="Extracted from text")

        return JudgeResult(score=5.0, reasoning="Parse error")
