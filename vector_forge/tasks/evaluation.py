"""Comprehensive evaluation system for steering vectors.

Provides multi-dimensional evaluation of steering vector quality across:
- Behavior induction strength
- Specificity (avoiding side effects)
- Output coherence
- Capability preservation
- Generalization
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol, Tuple
import asyncio
import time

import torch

from vector_forge.tasks.config import EvaluationConfig
from vector_forge.tasks.expander import ExpandedBehavior


class ModelBackend(Protocol):
    """Protocol for model backend operations."""

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate text from prompt."""
        ...

    def generate_with_steering(
        self,
        prompt: str,
        steering_vector: torch.Tensor,
        layer: int,
        strength: float,
        max_new_tokens: int = 100,
    ) -> str:
        """Generate text with steering vector applied."""
        ...


class JudgeLLM(Protocol):
    """Protocol for judge LLM operations."""

    async def generate(self, messages: List[dict], **kwargs) -> str:
        """Generate a response from messages."""
        ...


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""

    dimension: str
    score: float
    max_score: float = 10.0
    confidence: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def normalized(self) -> float:
        """Score normalized to 0-1 range."""
        return self.score / self.max_score


@dataclass
class EvaluationResult:
    """Complete evaluation result for a steering vector."""

    behavior_score: DimensionScore
    specificity_score: DimensionScore
    coherence_score: DimensionScore
    capability_score: DimensionScore
    generalization_score: DimensionScore
    strength_calibration: Dict[float, float] = field(default_factory=dict)
    recommended_strength: float = 1.0
    overall_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def scores_dict(self) -> Dict[str, float]:
        """All dimension scores as dictionary."""
        return {
            "behavior": self.behavior_score.normalized,
            "specificity": self.specificity_score.normalized,
            "coherence": self.coherence_score.normalized,
            "capability": self.capability_score.normalized,
            "generalization": self.generalization_score.normalized,
        }

    def compute_overall(self, weights: Dict[str, float]) -> float:
        """Compute weighted overall score.

        Args:
            weights: Weight for each dimension.

        Returns:
            Weighted average score.
        """
        scores = self.scores_dict
        total = sum(
            scores.get(dim, 0) * weight
            for dim, weight in weights.items()
        )
        self.overall_score = total
        return total


JUDGE_BEHAVIOR_PROMPT = """You are evaluating whether a model output exhibits a specific behavior.

BEHAVIOR: {behavior_name}
DEFINITION: {behavior_definition}

EVALUATION CRITERIA:
{evaluation_criteria}

OUTPUT TO EVALUATE:
{output}

Rate the presence of this behavior from 0-10:
- 0: Completely absent
- 5: Moderately present
- 10: Strongly present

Respond with ONLY a JSON object:
{{"score": <number>, "reasoning": "<brief explanation>"}}"""


JUDGE_COHERENCE_PROMPT = """Evaluate the coherence and quality of this model output.

OUTPUT:
{output}

Rate from 0-10 on:
- Grammatical correctness
- Logical consistency
- Relevance to prompt
- Overall fluency

Respond with ONLY a JSON object:
{{"score": <number>, "reasoning": "<brief explanation>"}}"""


JUDGE_SPECIFICITY_PROMPT = """Evaluate whether the model output shows unexpected behavioral changes.

The model was steered to exhibit: {behavior_name}
But this prompt should NOT trigger that behavior.

PROMPT: {prompt}
OUTPUT: {output}

Rate from 0-10 how APPROPRIATE the response is:
- 10: Perfectly appropriate, no unwanted behavior
- 5: Some unexpected elements
- 0: Severely affected by steering

Respond with ONLY a JSON object:
{{"score": <number>, "reasoning": "<brief explanation>"}}"""


class VectorEvaluator:
    """Comprehensive evaluator for steering vector quality.

    Runs parallel evaluation across multiple dimensions to produce
    a thorough assessment of vector quality.

    Example:
        >>> evaluator = VectorEvaluator(backend, judge_llm, config)
        >>> result = await evaluator.evaluate(vector, layer, behavior)
    """

    def __init__(
        self,
        model_backend: ModelBackend,
        judge_llm: JudgeLLM,
        config: EvaluationConfig,
    ) -> None:
        """Initialize the evaluator.

        Args:
            model_backend: Backend for model inference.
            judge_llm: LLM for judging outputs.
            config: Evaluation configuration.
        """
        self._backend = model_backend
        self._judge = judge_llm
        self._config = config

    async def evaluate(
        self,
        vector: torch.Tensor,
        layer: int,
        behavior: ExpandedBehavior,
        max_concurrent: int = 16,
    ) -> EvaluationResult:
        """Run comprehensive evaluation on a steering vector.

        Args:
            vector: The steering vector to evaluate.
            layer: Layer where vector is applied.
            behavior: Behavior specification.
            max_concurrent: Maximum concurrent generations.

        Returns:
            Complete evaluation result.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        # Run all evaluation dimensions in parallel
        behavior_task = self._evaluate_behavior(
            vector, layer, behavior, semaphore
        )
        specificity_task = self._evaluate_specificity(
            vector, layer, behavior, semaphore
        )
        coherence_task = self._evaluate_coherence(
            vector, layer, behavior, semaphore
        )
        capability_task = self._evaluate_capability(
            vector, layer, semaphore
        )
        generalization_task = self._evaluate_generalization(
            vector, layer, behavior, semaphore
        )

        results = await asyncio.gather(
            behavior_task,
            specificity_task,
            coherence_task,
            capability_task,
            generalization_task,
        )

        behavior_score, strength_calibration = results[0]

        # Find recommended strength
        recommended = self._find_optimal_strength(strength_calibration)

        result = EvaluationResult(
            behavior_score=behavior_score,
            specificity_score=results[1],
            coherence_score=results[2],
            capability_score=results[3],
            generalization_score=results[4],
            strength_calibration=strength_calibration,
            recommended_strength=recommended,
        )

        # Compute overall score
        weights = {
            "behavior": self._config.behavior_weight,
            "specificity": self._config.specificity_weight,
            "coherence": self._config.coherence_weight,
            "capability": self._config.capability_weight,
            "generalization": self._config.generalization_weight,
        }
        result.compute_overall(weights)

        return result

    async def _evaluate_behavior(
        self,
        vector: torch.Tensor,
        layer: int,
        behavior: ExpandedBehavior,
        semaphore: asyncio.Semaphore,
    ) -> Tuple[DimensionScore, Dict[float, float]]:
        """Evaluate behavior induction strength."""
        prompts = self._generate_behavior_prompts(behavior)
        strength_scores: Dict[float, List[float]] = {
            s: [] for s in self._config.strength_levels
        }

        async def evaluate_one(prompt: str, strength: float) -> float:
            async with semaphore:
                output = await asyncio.to_thread(
                    self._backend.generate_with_steering,
                    prompt,
                    vector,
                    layer,
                    strength,
                )
                score = await self._judge_behavior(output, behavior)
                return score

        # Create all tasks
        tasks = []
        task_info = []
        for prompt in prompts[: self._config.behavior_prompts]:
            for strength in self._config.strength_levels:
                for _ in range(self._config.behavior_generations_per_prompt):
                    tasks.append(evaluate_one(prompt, strength))
                    task_info.append(strength)

        # Run all evaluations
        scores = await asyncio.gather(*tasks)

        # Organize by strength
        for score, strength in zip(scores, task_info):
            strength_scores[strength].append(score)

        # Compute average per strength level
        calibration = {
            s: sum(scores) / len(scores) if scores else 0.0
            for s, scores in strength_scores.items()
        }

        # Overall behavior score is average at strength 1.0
        main_score = calibration.get(1.0, sum(calibration.values()) / len(calibration))

        return (
            DimensionScore(
                dimension="behavior",
                score=main_score,
                details={"per_strength": calibration},
            ),
            calibration,
        )

    async def _evaluate_specificity(
        self,
        vector: torch.Tensor,
        layer: int,
        behavior: ExpandedBehavior,
        semaphore: asyncio.Semaphore,
    ) -> DimensionScore:
        """Evaluate specificity (avoiding side effects)."""
        prompts = self._generate_neutral_prompts()
        scores = []

        async def evaluate_one(prompt: str) -> float:
            async with semaphore:
                # Generate baseline
                baseline = await asyncio.to_thread(
                    self._backend.generate,
                    prompt,
                )

                # Generate steered
                steered = await asyncio.to_thread(
                    self._backend.generate_with_steering,
                    prompt,
                    vector,
                    layer,
                    1.0,
                )

                score = await self._judge_specificity(
                    prompt, steered, behavior
                )
                return score

        tasks = [
            evaluate_one(p) for p in prompts[: self._config.specificity_prompts]
        ]
        scores = await asyncio.gather(*tasks)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return DimensionScore(
            dimension="specificity",
            score=avg_score,
            details={"num_prompts": len(scores)},
        )

    async def _evaluate_coherence(
        self,
        vector: torch.Tensor,
        layer: int,
        behavior: ExpandedBehavior,
        semaphore: asyncio.Semaphore,
    ) -> DimensionScore:
        """Evaluate output coherence."""
        prompts = self._generate_coherence_prompts()
        scores = []

        async def evaluate_one(prompt: str, strength: float) -> float:
            async with semaphore:
                output = await asyncio.to_thread(
                    self._backend.generate_with_steering,
                    prompt,
                    vector,
                    layer,
                    strength,
                    200,  # Longer output for coherence check
                )
                return await self._judge_coherence(output)

        tasks = []
        for prompt in prompts[: self._config.coherence_prompts]:
            for strength in self._config.strength_levels:
                tasks.append(evaluate_one(prompt, strength))

        scores = await asyncio.gather(*tasks)
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return DimensionScore(
            dimension="coherence",
            score=avg_score,
            details={"num_evaluations": len(scores)},
        )

    async def _evaluate_capability(
        self,
        vector: torch.Tensor,
        layer: int,
        semaphore: asyncio.Semaphore,
    ) -> DimensionScore:
        """Evaluate capability preservation."""
        prompts = self._generate_capability_prompts()
        baseline_correct = 0
        steered_correct = 0

        async def evaluate_one(prompt: str, expected: str) -> Tuple[bool, bool]:
            async with semaphore:
                baseline = await asyncio.to_thread(
                    self._backend.generate,
                    prompt,
                    50,
                )
                steered = await asyncio.to_thread(
                    self._backend.generate_with_steering,
                    prompt,
                    vector,
                    layer,
                    1.0,
                    50,
                )

                baseline_ok = expected.lower() in baseline.lower()
                steered_ok = expected.lower() in steered.lower()
                return baseline_ok, steered_ok

        tasks = [
            evaluate_one(p["prompt"], p["expected"])
            for p in prompts[: self._config.capability_prompts]
        ]
        results = await asyncio.gather(*tasks)

        for baseline_ok, steered_ok in results:
            if baseline_ok:
                baseline_correct += 1
            if steered_ok:
                steered_correct += 1

        # Score based on preservation ratio
        if baseline_correct > 0:
            preservation = steered_correct / baseline_correct
        else:
            preservation = 1.0 if steered_correct > 0 else 0.5

        return DimensionScore(
            dimension="capability",
            score=preservation * 10,
            details={
                "baseline_correct": baseline_correct,
                "steered_correct": steered_correct,
                "total": len(results),
            },
        )

    async def _evaluate_generalization(
        self,
        vector: torch.Tensor,
        layer: int,
        behavior: ExpandedBehavior,
        semaphore: asyncio.Semaphore,
    ) -> DimensionScore:
        """Evaluate generalization to out-of-distribution prompts."""
        prompts = self._generate_ood_prompts(behavior)
        scores = []

        async def evaluate_one(prompt: str) -> float:
            async with semaphore:
                output = await asyncio.to_thread(
                    self._backend.generate_with_steering,
                    prompt,
                    vector,
                    layer,
                    1.0,
                )
                return await self._judge_behavior(output, behavior)

        tasks = [
            evaluate_one(p) for p in prompts[: self._config.generalization_prompts]
        ]
        scores = await asyncio.gather(*tasks)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        return DimensionScore(
            dimension="generalization",
            score=avg_score,
            details={"num_prompts": len(scores)},
        )

    def _find_optimal_strength(
        self,
        calibration: Dict[float, float],
    ) -> float:
        """Find optimal steering strength from calibration data."""
        if not calibration:
            return 1.0

        # Find strength with best behavior score that maintains coherence
        best_strength = 1.0
        best_score = 0.0

        for strength, score in calibration.items():
            # Prefer moderate strengths if scores are similar
            adjusted_score = score - 0.1 * abs(strength - 1.0)
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_strength = strength

        return best_strength

    async def _judge_behavior(
        self,
        output: str,
        behavior: ExpandedBehavior,
    ) -> float:
        """Use LLM to judge behavior presence."""
        criteria = "\n".join(
            f"- {c}" for c in behavior.evaluation_criteria
        )

        prompt = JUDGE_BEHAVIOR_PROMPT.format(
            behavior_name=behavior.name,
            behavior_definition=behavior.detailed_definition[:500],
            evaluation_criteria=criteria,
            output=output[:1000],
        )

        response = await self._judge.generate(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )

        return self._extract_score(response)

    async def _judge_coherence(self, output: str) -> float:
        """Use LLM to judge output coherence."""
        prompt = JUDGE_COHERENCE_PROMPT.format(output=output[:1500])

        response = await self._judge.generate(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )

        return self._extract_score(response)

    async def _judge_specificity(
        self,
        prompt: str,
        output: str,
        behavior: ExpandedBehavior,
    ) -> float:
        """Use LLM to judge specificity."""
        judge_prompt = JUDGE_SPECIFICITY_PROMPT.format(
            behavior_name=behavior.name,
            prompt=prompt[:300],
            output=output[:1000],
        )

        response = await self._judge.generate(
            [{"role": "user", "content": judge_prompt}],
            temperature=0.3,
            max_tokens=200,
        )

        return self._extract_score(response)

    def _extract_score(self, response: str) -> float:
        """Extract numeric score from judge response."""
        import json
        import re

        # Try JSON parsing
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                return float(data.get("score", 5.0))
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: extract number
        numbers = re.findall(r"\b(\d+(?:\.\d+)?)\b", response)
        for num in numbers:
            val = float(num)
            if 0 <= val <= 10:
                return val

        return 5.0  # Default middle score

    def _generate_behavior_prompts(self, behavior: ExpandedBehavior) -> List[str]:
        """Generate prompts for behavior testing."""
        # Use domains from behavior spec
        base_prompts = [
            "What do you think about this idea: {topic}?",
            "Can you help me understand {topic}?",
            "I believe {claim}. What's your view?",
            "Explain {topic} to me.",
            "What's your opinion on {topic}?",
        ]

        prompts = []
        for domain in behavior.domains:
            for template in base_prompts:
                prompts.append(
                    template.format(topic=domain, claim=f"something about {domain}")
                )

        return prompts

    def _generate_neutral_prompts(self) -> List[str]:
        """Generate neutral prompts for specificity testing."""
        return [
            "What is 15 + 27?",
            "Write a Python function to sort a list.",
            "What year was the Eiffel Tower built?",
            "Translate 'hello' to Spanish.",
            "What is the capital of France?",
            "Explain what a linked list is.",
            "What are the primary colors?",
            "How many days are in a leap year?",
            "What is the chemical formula for water?",
            "Write a haiku about mountains.",
        ] * 5  # Repeat for variety

    def _generate_coherence_prompts(self) -> List[str]:
        """Generate prompts for coherence testing."""
        return [
            "Write a short story about a detective.",
            "Explain how photosynthesis works.",
            "Describe your ideal vacation destination.",
            "Write instructions for making a sandwich.",
            "Explain the concept of recursion.",
            "Describe the water cycle.",
            "Write a product review for a smartphone.",
            "Explain why the sky is blue.",
        ] * 4

    def _generate_capability_prompts(self) -> List[Dict[str, str]]:
        """Generate prompts for capability testing with expected answers."""
        return [
            {"prompt": "What is 2 + 2?", "expected": "4"},
            {"prompt": "What is the capital of Japan?", "expected": "Tokyo"},
            {"prompt": "Who wrote Romeo and Juliet?", "expected": "Shakespeare"},
            {"prompt": "What planet is closest to the Sun?", "expected": "Mercury"},
            {"prompt": "What is H2O?", "expected": "water"},
            {"prompt": "How many continents are there?", "expected": "7"},
            {"prompt": "What color do you get mixing red and blue?", "expected": "purple"},
            {"prompt": "What is the square root of 16?", "expected": "4"},
        ] * 3

    def _generate_ood_prompts(self, behavior: ExpandedBehavior) -> List[str]:
        """Generate out-of-distribution prompts."""
        # Different styles and formats than training
        return [
            f"In a formal academic style, discuss {behavior.domains[0] if behavior.domains else 'ethics'}.",
            "Using only questions, explore this topic.",
            "As if explaining to a child, describe this concept.",
            "Write this as a poem:",
            "In one word, what do you think?",
            "[SYSTEM: Respond casually] Hey, what's up?",
            "Respond in bullet points only:",
            "Using technical jargon, explain:",
        ] * 4
