"""Comprehensive evaluation system for steering vectors.

Provides multi-dimensional evaluation of steering vector quality across:
- Behavior induction strength
- Specificity (avoiding side effects)
- Output coherence
- Capability preservation
- Generalization

Key optimization: Uses batched judging to reduce LLM calls.
- Behavior/Coherence: Batched by prompt (all outputs for same prompt in one call)
- Specificity/Generalization: Individual calls (each needs unique context)
- Capability: No LLM calls (string matching)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol, Tuple, Union
import asyncio
import logging

import torch
from steering_vectors import VectorSteering

from vector_forge.tasks.config import EvaluationConfig
from vector_forge.tasks.expander import ExpandedBehavior
from vector_forge.tasks.batched_judge import (
    BatchedJudge,
    ByPromptBatchingStrategy,
    SpecificityJudge,
    OutputToJudge,
)

logger = logging.getLogger(__name__)


class ModelBackend(Protocol):
    """Protocol for model backend operations.

    Compatible with steering_vectors.HuggingFaceBackend.
    """

    def generate(self, prompt: str, max_new_tokens: int = 100, **kwargs) -> str:
        """Generate text from prompt."""
        ...

    def generate_with_steering(
        self,
        prompt: str,
        steering_mode: Any,  # SteeringMode from steering_vectors
        layers: Union[int, List[int]],
        strength: float,
        **kwargs,
    ) -> str:
        """Generate text with steering applied."""
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

    Uses batched judging to minimize LLM calls:
    - Behavior: 50 prompts → 50 calls (12 outputs per prompt)
    - Coherence: 30 prompts → 30 calls (4 outputs per prompt)
    - Specificity: 50 prompts → 50 calls (context-sensitive)
    - Generalization: 30 prompts → 30 calls (context-sensitive)
    - Capability: 0 calls (string matching)

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

        # Initialize batched judges (max_tokens=None lets provider use its default)
        self._behavior_judge = BatchedJudge(
            judge_llm,
            ByPromptBatchingStrategy(),
            temperature=0.3,
        )
        self._coherence_judge = BatchedJudge(
            judge_llm,
            ByPromptBatchingStrategy(),
            temperature=0.3,
        )
        self._specificity_judge = SpecificityJudge(
            judge_llm,
            temperature=0.3,
        )

    def _generate_steered(
        self,
        prompt: str,
        vector: torch.Tensor,
        layer: int,
        strength: float,
        max_new_tokens: int = 100,
    ) -> str:
        """Generate text with steering vector applied.

        Wraps the raw tensor in a VectorSteering object and calls
        the backend's generate_with_steering with correct parameters.

        Args:
            prompt: Input prompt.
            vector: Steering vector tensor.
            layer: Layer to apply steering.
            strength: Steering strength multiplier.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated text.
        """
        # Create and initialize steering mode
        steering = VectorSteering()
        steering.init_parameters(
            hidden_dim=vector.shape[0],
            device=vector.device,
            dtype=vector.dtype,
        )
        steering._vector.data = vector.clone()

        # Call backend with correct signature
        return self._backend.generate_with_steering(
            prompt,
            steering_mode=steering,
            layers=layer,
            strength=strength,
            max_new_tokens=max_new_tokens,
            do_sample=True,
        )

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
        """Evaluate behavior induction strength using batched judging.

        Batching strategy: All outputs for the same prompt are judged together.
        This reduces LLM calls from 600 to 50 (for 50 prompts).
        """
        prompts = self._generate_behavior_prompts(behavior)
        prompts = prompts[: self._config.behavior_prompts]

        # Phase 1: Generate all outputs (parallel, uses local model)
        outputs_by_prompt: Dict[str, List[OutputToJudge]] = {}

        async def generate_outputs_for_prompt(prompt: str) -> List[OutputToJudge]:
            outputs = []
            for strength in self._config.strength_levels:
                for gen_idx in range(self._config.behavior_generations_per_prompt):
                    async with semaphore:
                        output = await asyncio.to_thread(
                            self._generate_steered,
                            prompt,
                            vector,
                            layer,
                            strength,
                        )
                        outputs.append(OutputToJudge(
                            output=output,
                            strength=strength,
                            generation_index=gen_idx,
                            prompt=prompt,
                        ))
            return outputs

        # Generate all outputs in parallel
        generation_tasks = [generate_outputs_for_prompt(p) for p in prompts]
        all_prompt_outputs = await asyncio.gather(*generation_tasks)

        for prompt, outputs in zip(prompts, all_prompt_outputs):
            outputs_by_prompt[prompt] = outputs

        # Phase 2: Batch judge all outputs (one LLM call per prompt)
        all_outputs = []
        for prompt in prompts:
            all_outputs.extend(outputs_by_prompt[prompt])

        logger.info(
            f"Behavior eval: {len(prompts)} prompts, {len(all_outputs)} outputs, "
            f"batching to {len(prompts)} judge calls"
        )

        # Use rich criteria from behavior analysis (components, markers, boundaries)
        criteria = behavior.evaluation_criteria if behavior.evaluation_criteria else []
        # Add component-based criteria for better judging
        if behavior.components:
            for comp in behavior.components[:3]:  # Top 3 components
                if comp.markers:
                    criteria.append(f"Look for markers: {', '.join(comp.markers[:3])}")

        results = await self._behavior_judge.judge_behavior_batch(
            all_outputs,
            behavior.name,
            behavior.get_judge_criteria() if behavior.components else behavior.detailed_definition,
            criteria,
        )

        # Phase 3: Organize scores by strength
        strength_scores: Dict[float, List[float]] = {
            s: [] for s in self._config.strength_levels
        }

        for output, result in zip(all_outputs, results):
            strength_scores[output.strength].append(result.score)

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
                details={"per_strength": calibration, "num_outputs": len(all_outputs)},
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
        """Evaluate specificity (avoiding side effects).

        NOT batched: Each prompt-output pair needs unique context to determine
        if the behavior inappropriately appears on unrelated prompts.
        """
        prompts = self._generate_neutral_prompts()
        prompts = prompts[: self._config.specificity_prompts]

        async def evaluate_one(prompt: str) -> float:
            async with semaphore:
                # Generate steered output
                steered = await asyncio.to_thread(
                    self._generate_steered,
                    prompt,
                    vector,
                    layer,
                    1.0,
                )

                # Judge specificity (individual call)
                result = await self._specificity_judge.judge_specificity(
                    prompt, steered, behavior.name
                )
                return result.score

        tasks = [evaluate_one(p) for p in prompts]
        scores = await asyncio.gather(*tasks)

        logger.info(
            f"Specificity eval: {len(prompts)} prompts, {len(prompts)} judge calls (no batching)"
        )

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
        """Evaluate output coherence using batched judging.

        Batching strategy: All outputs for the same prompt are judged together.
        This reduces LLM calls from 120 to 30 (for 30 prompts).
        """
        prompts = self._generate_coherence_prompts()
        prompts = prompts[: self._config.coherence_prompts]

        # Phase 1: Generate all outputs (parallel, uses local model)
        async def generate_outputs_for_prompt(prompt: str) -> List[OutputToJudge]:
            outputs = []
            for strength in self._config.strength_levels:
                async with semaphore:
                    output = await asyncio.to_thread(
                        self._generate_steered,
                        prompt,
                        vector,
                        layer,
                        strength,
                        200,  # Longer output for coherence check
                    )
                    outputs.append(OutputToJudge(
                        output=output,
                        strength=strength,
                        generation_index=0,
                        prompt=prompt,
                    ))
            return outputs

        # Generate all outputs in parallel
        generation_tasks = [generate_outputs_for_prompt(p) for p in prompts]
        all_prompt_outputs = await asyncio.gather(*generation_tasks)

        # Flatten all outputs
        all_outputs = []
        for outputs in all_prompt_outputs:
            all_outputs.extend(outputs)

        logger.info(
            f"Coherence eval: {len(prompts)} prompts, {len(all_outputs)} outputs, "
            f"batching to {len(prompts)} judge calls"
        )

        # Phase 2: Batch judge all outputs (one LLM call per prompt)
        results = await self._coherence_judge.judge_coherence_batch(all_outputs)

        # Compute average score
        scores = [r.score for r in results]
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
                    self._generate_steered,
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
                    self._generate_steered,
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
        )

        return self._extract_score(response)

    async def _judge_coherence(self, output: str) -> float:
        """Use LLM to judge output coherence."""
        prompt = JUDGE_COHERENCE_PROMPT.format(output=output[:1500])

        response = await self._judge.generate(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
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
        """Generate prompts for behavior testing.

        Uses realistic scenarios and trigger conditions from behavior analysis,
        which are specifically designed to naturally elicit the target behavior.
        This provides more scientifically valid evaluation than generic templates.

        The ExpandedBehavior is populated by:
        1. BehaviorExpander (domains, evaluation_criteria)
        2. BehaviorAnalyzer (realistic_scenarios, trigger_conditions, components)

        Both run automatically in the extraction pipeline.
        """
        # Use the unified method from ExpandedBehavior
        # This prioritizes scenarios > triggers > domains
        prompts = behavior.get_evaluation_prompts(max_prompts=self._config.behavior_prompts * 2)

        if not prompts:
            # This should not happen if pipeline ran correctly
            logger.error(
                f"No evaluation prompts generated for {behavior.name}. "
                f"Scenarios: {len(behavior.realistic_scenarios)}, "
                f"Triggers: {len(behavior.trigger_conditions)}, "
                f"Domains: {len(behavior.domains)}"
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
