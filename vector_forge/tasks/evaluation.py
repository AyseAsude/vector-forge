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
from typing import List, Dict, Any, Optional, Protocol, Tuple, Union, TYPE_CHECKING
import asyncio
import logging
import time

import torch
from steering_vectors import VectorSteering

from vector_forge.llm import JSON_RESPONSE_FORMAT
from vector_forge.tasks.config import EvaluationConfig
from vector_forge.tasks.expander import ExpandedBehavior
from vector_forge.tasks.batched_judge import (
    BatchedJudge,
    ByPromptBatchingStrategy,
    SpecificityJudge,
    OutputToJudge,
)
from vector_forge.tasks.eval_judging_composer import (
    EvalJudgingComposer,
    JudgingMode,
)
from vector_forge.core.concurrency import (
    run_in_gpu_executor,
    limit_concurrent_dimensions,
    get_llm_semaphore,
)

if TYPE_CHECKING:
    from vector_forge.storage.emitter import EventEmitter

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


# Note: Judge prompts are now composed by EvalJudgingComposer
# which builds behavior-specific prompts from ExpandedBehavior.
# This provides richer context including components, markers,
# examples, and boundaries.


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
        event_emitter: Optional["EventEmitter"] = None,
        max_concurrent_judge_calls: int = 64,
    ) -> None:
        """Initialize the evaluator.

        Args:
            model_backend: Backend for model inference.
            judge_llm: LLM for judging outputs.
            config: Evaluation configuration.
            event_emitter: Optional event emitter for event sourcing.
            max_concurrent_judge_calls: Maximum concurrent LLM API calls for judging.
        """
        self._backend = model_backend
        self._judge = judge_llm
        self._config = config
        self._emitter = event_emitter
        self._max_concurrent = max_concurrent_judge_calls

        # Evaluation tracking for event sourcing
        self._current_eval_id: Optional[str] = None
        self._generation_count = 0
        self._judge_call_count = 0

        # Initialize batched judges (max_tokens=None lets provider use its default)
        self._behavior_judge = BatchedJudge(
            judge_llm,
            ByPromptBatchingStrategy(),
            temperature=0.3,
            max_concurrent=max_concurrent_judge_calls,
        )
        self._coherence_judge = BatchedJudge(
            judge_llm,
            ByPromptBatchingStrategy(),
            temperature=0.3,
            max_concurrent=max_concurrent_judge_calls,
        )
        self._specificity_judge = SpecificityJudge(
            judge_llm,
            temperature=0.3,
            max_concurrent=max_concurrent_judge_calls,
        )

        # Composer for individual judging calls (generalization, etc.)
        self._composer = EvalJudgingComposer()

    def _emit(self, method_name: str, **kwargs) -> None:
        """Emit an event if emitter is available."""
        if self._emitter is not None:
            method = getattr(self._emitter, method_name, None)
            if method:
                method(**kwargs)

    def set_evaluation_id(self, eval_id: str) -> None:
        """Set the current evaluation ID for event tracking."""
        self._current_eval_id = eval_id
        self._generation_count = 0
        self._judge_call_count = 0

    def _create_steering_mode(self, vector: torch.Tensor) -> VectorSteering:
        """Create a VectorSteering mode from a vector for inference.

        For inference, we don't need gradients, so we detach the vector.
        This avoids unnecessary memory overhead from gradient tracking.
        """
        # Use detached vector - no gradients needed for inference
        return VectorSteering(vector=vector.detach())

    def _format_prompt(self, prompt: str) -> str:
        """Format a prompt using the model's native chat template.

        Uses tokenizer.apply_chat_template() which is the HuggingFace standard
        for instruction-tuned models (Qwen, Llama, Mistral, etc.).

        Args:
            prompt: Raw prompt string.

        Returns:
            Formatted prompt with proper special tokens.
        """
        messages = [{"role": "user", "content": prompt}]
        return self._backend.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _format_prompts_batch(self, prompts: List[str]) -> List[str]:
        """Format multiple prompts using the model's native chat template.

        Args:
            prompts: List of raw prompt strings.

        Returns:
            List of formatted prompts with proper special tokens.
        """
        return [self._format_prompt(p) for p in prompts]

    def _generate_steered(
        self,
        prompt: str,
        vector: torch.Tensor,
        layer: int,
        strength: float,
        max_new_tokens: int = 100,
    ) -> str:
        """Generate text with steering vector applied (single prompt).

        Uses the model's native chat template for proper formatting.

        Note: For multiple prompts, use _generate_steered_batch for much
        better performance through GPU batching.

        Args:
            prompt: Input prompt.
            vector: Steering vector tensor.
            layer: Layer to apply steering.
            strength: Steering strength multiplier.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated text.
        """
        steering = self._create_steering_mode(vector)
        formatted_prompt = self._format_prompt(prompt)
        return self._backend.generate_with_steering(
            formatted_prompt,
            steering_mode=steering,
            layers=layer,
            strength=strength,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=self._config.generation_temperature,
        )

    def _generate_steered_batch(
        self,
        prompts: List[str],
        vector: torch.Tensor,
        layer: int,
        strength: float,
        max_new_tokens: int = 100,
    ) -> List[str]:
        """Generate text for multiple prompts with steering (BATCHED).

        Uses the model's native chat template for proper formatting.

        This is the HIGH-PERFORMANCE method - processes all prompts in
        a single GPU batch, maximizing throughput and parallelism.

        Args:
            prompts: List of input prompts.
            vector: Steering vector tensor.
            layer: Layer to apply steering.
            strength: Steering strength multiplier.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            List of generated texts (same order as prompts).
        """
        if not prompts:
            return []

        steering = self._create_steering_mode(vector)
        formatted_prompts = self._format_prompts_batch(prompts)
        return self._backend.generate_with_steering_batch(
            formatted_prompts,
            steering_mode=steering,
            layers=layer,
            strength=strength,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=self._config.generation_temperature,
        )

    def _generate_baseline_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
    ) -> List[str]:
        """Generate text for multiple prompts WITHOUT steering (BATCHED).

        Uses the model's native chat template for proper formatting.

        Args:
            prompts: List of input prompts.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            List of generated texts.
        """
        if not prompts:
            return []

        formatted_prompts = self._format_prompts_batch(prompts)
        return self._backend.generate_batch(
            formatted_prompts,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=self._config.generation_temperature,
        )

    async def evaluate(
        self,
        vector: torch.Tensor,
        layer: int,
        behavior: ExpandedBehavior,
        max_concurrent: int = 16,
    ) -> EvaluationResult:
        """Run comprehensive evaluation on a steering vector.

        Evaluates across 5 dimensions with controlled concurrency to prevent
        thread pool starvation. Dimensions are staggered via semaphore to
        ensure GPU work and LLM responses are properly scheduled.

        Args:
            vector: The steering vector to evaluate.
            layer: Layer where vector is applied.
            behavior: Behavior specification.
            max_concurrent: Maximum concurrent generations (deprecated, uses global config).

        Returns:
            Complete evaluation result.
        """

        async def run_dimension(coro):
            """Run a dimension evaluation with concurrency limiting."""
            async with limit_concurrent_dimensions():
                return await coro

        # Run dimensions with staggered concurrency (semaphore limits concurrent)
        results = await asyncio.gather(
            run_dimension(self._evaluate_behavior(vector, layer, behavior)),
            run_dimension(self._evaluate_specificity(vector, layer, behavior)),
            run_dimension(self._evaluate_coherence(vector, layer, behavior)),
            run_dimension(self._evaluate_capability(vector, layer)),
            run_dimension(self._evaluate_generalization(vector, layer, behavior)),
            return_exceptions=True,
        )

        # Handle any dimension failures gracefully
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                dim_names = ["behavior", "specificity", "coherence", "capability", "generalization"]
                logger.warning(f"Dimension {dim_names[i]} failed: {result}")
                # Replace with default score
                if i == 0:
                    results[i] = (DimensionScore("behavior", 5.0), {})
                else:
                    results[i] = DimensionScore(dim_names[i], 5.0)

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
    ) -> Tuple[DimensionScore, Dict[float, float]]:
        """Evaluate behavior induction strength using batched generation and judging.

        BATCHED GENERATION: All prompts for each strength level are processed
        in a single GPU batch, maximizing throughput. This is MUCH faster than
        individual run_in_gpu_executor calls.

        Batching reduces:
        - 600 individual generate() calls → 4 batched calls (one per strength)
        - 600 individual LLM judge calls → 50 batched calls (one per prompt)
        """
        dimension_start = time.time()
        prompts = self._generate_behavior_prompts(behavior)
        prompts = prompts[: self._config.behavior_prompts]

        num_strengths = len(self._config.strength_levels)
        num_gens = self._config.behavior_generations_per_prompt
        total_generations = len(prompts) * num_strengths * num_gens

        # Emit dimension started
        if self._current_eval_id:
            self._emit(
                "emit_evaluation_dimension_started",
                evaluation_id=self._current_eval_id,
                dimension="behavior",
                num_prompts=len(prompts),
                num_generations=total_generations,
            )

        # Phase 1: BATCHED generation - group by strength level
        # Each strength level gets ONE batched call that processes ALL prompts
        all_outputs: List[OutputToJudge] = []
        outputs_by_prompt: Dict[str, List[OutputToJudge]] = {p: [] for p in prompts}
        generation_counter = 0

        for strength in self._config.strength_levels:
            # For each generation index, batch ALL prompts together
            for gen_idx in range(num_gens):
                # Emit progress before batch
                if self._current_eval_id:
                    self._emit(
                        "emit_evaluation_progress",
                        evaluation_id=self._current_eval_id,
                        phase="generating",
                        completed=generation_counter,
                        total=total_generations,
                        current_dimension="behavior",
                    )

                # SINGLE batched call for ALL prompts at this strength/gen_idx
                # This runs on GPU in parallel - much faster than individual calls
                batch_outputs = await run_in_gpu_executor(
                    self._generate_steered_batch,
                    prompts,
                    vector,
                    layer,
                    strength,
                    100,  # max_new_tokens
                )

                # Organize outputs
                for prompt, output in zip(prompts, batch_outputs):
                    output_obj = OutputToJudge(
                        output=output,
                        strength=strength,
                        generation_index=gen_idx,
                        prompt=prompt,
                    )
                    all_outputs.append(output_obj)
                    outputs_by_prompt[prompt].append(output_obj)

                generation_counter += len(prompts)
                self._generation_count += len(prompts)

        logger.info(
            f"Behavior eval: {len(prompts)} prompts × {num_strengths} strengths × "
            f"{num_gens} gens = {len(all_outputs)} outputs (batched into "
            f"{num_strengths * num_gens} GPU calls)"
        )

        # Phase 2: Batch judge all outputs (one LLM call per prompt)

        # Emit progress - switching to judging phase
        if self._current_eval_id:
            self._emit(
                "emit_evaluation_progress",
                evaluation_id=self._current_eval_id,
                phase="judging",
                completed=0,
                total=len(prompts),
                current_dimension="behavior",
            )

        # Phase 2: Batch judge all outputs
        # The composer in BatchedJudge handles building rich prompts from
        # ExpandedBehavior (components, markers, examples, boundaries)
        judge_start = time.time()
        results = await self._behavior_judge.judge_behavior_batch(
            all_outputs,
            behavior,
        )
        judge_latency = (time.time() - judge_start) * 1000
        self._judge_call_count += len(prompts)

        # Emit judge call event
        if self._current_eval_id:
            all_scores = [r.score for r in results]
            self._emit(
                "emit_evaluation_judge_call",
                evaluation_id=self._current_eval_id,
                dimension="behavior",
                prompt=f"batch:{len(prompts)} prompts",
                num_outputs=len(all_outputs),
                scores=all_scores[:20],  # Limit to first 20 to avoid huge events
                latency_ms=judge_latency,
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

        dimension_duration = time.time() - dimension_start

        # Emit dimension completed
        if self._current_eval_id:
            self._emit(
                "emit_evaluation_dimension_completed",
                evaluation_id=self._current_eval_id,
                dimension="behavior",
                score=main_score,
                max_score=10.0,
                details={"per_strength": calibration, "num_outputs": len(all_outputs)},
                duration_seconds=dimension_duration,
            )

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
    ) -> DimensionScore:
        """Evaluate specificity (avoiding side effects).

        BATCHED GENERATION: All neutral prompts are generated in one GPU batch.
        Judging is still per-prompt (each needs unique context).
        """
        dimension_start = time.time()
        prompts = self._generate_neutral_prompts()
        prompts = prompts[: self._config.specificity_prompts]

        # Emit dimension started
        if self._current_eval_id:
            self._emit(
                "emit_evaluation_dimension_started",
                evaluation_id=self._current_eval_id,
                dimension="specificity",
                num_prompts=len(prompts),
                num_generations=len(prompts),
            )

        # Phase 1: BATCHED generation - one call for ALL prompts
        if self._current_eval_id:
            self._emit(
                "emit_evaluation_progress",
                evaluation_id=self._current_eval_id,
                phase="generating",
                completed=0,
                total=len(prompts),
                current_dimension="specificity",
            )

        # SINGLE batched call for all neutral prompts
        steered_outputs = await run_in_gpu_executor(
            self._generate_steered_batch,
            prompts,
            vector,
            layer,
            1.0,  # strength
            100,  # max_new_tokens
        )
        self._generation_count += len(prompts)

        logger.info(
            f"Specificity eval: {len(prompts)} prompts generated in 1 batch call"
        )

        # Phase 2: Judge each output (requires individual context)
        # Uses batch method with internal semaphore for concurrency control
        if self._current_eval_id:
            self._emit(
                "emit_evaluation_progress",
                evaluation_id=self._current_eval_id,
                phase="judging",
                completed=0,
                total=len(prompts),
                current_dimension="specificity",
            )

        # Build batch items: (prompt, output, behavior_name)
        batch_items = [
            (p, o, behavior.name) for p, o in zip(prompts, steered_outputs)
        ]

        # Run all judge calls concurrently with semaphore-limited concurrency
        results = await self._specificity_judge.judge_specificity_batch(batch_items)
        scores = [r.score for r in results]
        self._judge_call_count += len(prompts)

        logger.info(
            f"Specificity eval: {len(prompts)} judge calls completed concurrently"
        )

        avg_score = sum(scores) / len(scores) if scores else 0.0
        dimension_duration = time.time() - dimension_start

        # Emit dimension completed
        if self._current_eval_id:
            self._emit(
                "emit_evaluation_dimension_completed",
                evaluation_id=self._current_eval_id,
                dimension="specificity",
                score=avg_score,
                max_score=10.0,
                details={"num_prompts": len(scores)},
                duration_seconds=dimension_duration,
            )

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
    ) -> DimensionScore:
        """Evaluate output coherence using batched generation and judging.

        BATCHED GENERATION: All prompts for each strength level are processed
        in a single GPU batch, maximizing throughput.
        """
        dimension_start = time.time()
        prompts = self._generate_coherence_prompts()
        prompts = prompts[: self._config.coherence_prompts]

        num_strengths = len(self._config.strength_levels)
        total_generations = len(prompts) * num_strengths

        # Emit dimension started
        if self._current_eval_id:
            self._emit(
                "emit_evaluation_dimension_started",
                evaluation_id=self._current_eval_id,
                dimension="coherence",
                num_prompts=len(prompts),
                num_generations=total_generations,
            )

        # Phase 1: BATCHED generation - group by strength level
        all_outputs: List[OutputToJudge] = []
        generation_counter = 0

        for strength in self._config.strength_levels:
            # Emit progress before batch
            if self._current_eval_id:
                self._emit(
                    "emit_evaluation_progress",
                    evaluation_id=self._current_eval_id,
                    phase="generating",
                    completed=generation_counter,
                    total=total_generations,
                    current_dimension="coherence",
                )

            # SINGLE batched call for ALL prompts at this strength
            batch_outputs = await run_in_gpu_executor(
                self._generate_steered_batch,
                prompts,
                vector,
                layer,
                strength,
                200,  # Longer output for coherence check
            )

            # Organize outputs
            for prompt, output in zip(prompts, batch_outputs):
                all_outputs.append(OutputToJudge(
                    output=output,
                    strength=strength,
                    generation_index=0,
                    prompt=prompt,
                ))

            generation_counter += len(prompts)
            self._generation_count += len(prompts)

        logger.info(
            f"Coherence eval: {len(prompts)} prompts × {num_strengths} strengths = "
            f"{len(all_outputs)} outputs (batched into {num_strengths} GPU calls)"
        )

        # Phase 2: Batch judge all outputs (one LLM call per prompt)
        judge_start = time.time()
        results = await self._coherence_judge.judge_coherence_batch(all_outputs)
        judge_latency = (time.time() - judge_start) * 1000
        self._judge_call_count += len(prompts)

        # Emit judge call event
        if self._current_eval_id:
            self._emit(
                "emit_evaluation_judge_call",
                evaluation_id=self._current_eval_id,
                dimension="coherence",
                prompt=f"batch:{len(prompts)} prompts",
                num_outputs=len(all_outputs),
                scores=[r.score for r in results[:20]],
                latency_ms=judge_latency,
            )

        # Compute average score
        scores = [r.score for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        dimension_duration = time.time() - dimension_start

        # Emit dimension completed
        if self._current_eval_id:
            self._emit(
                "emit_evaluation_dimension_completed",
                evaluation_id=self._current_eval_id,
                dimension="coherence",
                score=avg_score,
                max_score=10.0,
                details={"num_evaluations": len(scores)},
                duration_seconds=dimension_duration,
            )

        return DimensionScore(
            dimension="coherence",
            score=avg_score,
            details={"num_evaluations": len(scores)},
        )

    async def _evaluate_capability(
        self,
        vector: torch.Tensor,
        layer: int,
    ) -> DimensionScore:
        """Evaluate capability preservation using batched generation.

        BATCHED GENERATION: All baseline and steered outputs are generated
        in just 2 GPU batch calls (one for baseline, one for steered).
        """
        dimension_start = time.time()
        prompts = self._generate_capability_prompts()
        prompts_to_eval = prompts[: self._config.capability_prompts]

        # Emit dimension started (2 generations per prompt: baseline + steered)
        if self._current_eval_id:
            self._emit(
                "emit_evaluation_dimension_started",
                evaluation_id=self._current_eval_id,
                dimension="capability",
                num_prompts=len(prompts_to_eval),
                num_generations=len(prompts_to_eval) * 2,
            )

        # Extract just the prompt texts
        prompt_texts = [p["prompt"] for p in prompts_to_eval]
        expected_answers = [p["expected"] for p in prompts_to_eval]

        # Phase 1: BATCHED baseline generation (no steering)
        if self._current_eval_id:
            self._emit(
                "emit_evaluation_progress",
                evaluation_id=self._current_eval_id,
                phase="generating",
                completed=0,
                total=len(prompts_to_eval) * 2,
                current_dimension="capability",
            )

        baseline_outputs = await run_in_gpu_executor(
            self._generate_baseline_batch,
            prompt_texts,
            50,  # max_new_tokens
        )
        self._generation_count += len(prompt_texts)

        # Phase 2: BATCHED steered generation
        if self._current_eval_id:
            self._emit(
                "emit_evaluation_progress",
                evaluation_id=self._current_eval_id,
                phase="generating",
                completed=len(prompts_to_eval),
                total=len(prompts_to_eval) * 2,
                current_dimension="capability",
            )

        steered_outputs = await run_in_gpu_executor(
            self._generate_steered_batch,
            prompt_texts,
            vector,
            layer,
            1.0,  # strength
            50,  # max_new_tokens
        )
        self._generation_count += len(prompt_texts)

        logger.info(
            f"Capability eval: {len(prompts_to_eval)} prompts × 2 = "
            f"{len(prompts_to_eval) * 2} outputs (batched into 2 GPU calls)"
        )

        # Phase 3: Compare results
        baseline_correct = 0
        steered_correct = 0

        for baseline, steered, expected in zip(baseline_outputs, steered_outputs, expected_answers):
            if expected.lower() in baseline.lower():
                baseline_correct += 1
            if expected.lower() in steered.lower():
                steered_correct += 1

        # Score based on preservation ratio
        if baseline_correct > 0:
            preservation = steered_correct / baseline_correct
        else:
            preservation = 1.0 if steered_correct > 0 else 0.5

        dimension_duration = time.time() - dimension_start

        # Emit dimension completed
        if self._current_eval_id:
            self._emit(
                "emit_evaluation_dimension_completed",
                evaluation_id=self._current_eval_id,
                dimension="capability",
                score=preservation * 10,
                max_score=10.0,
                details={
                    "baseline_correct": baseline_correct,
                    "steered_correct": steered_correct,
                    "total": len(prompts_to_eval),
                },
                duration_seconds=dimension_duration,
            )

        return DimensionScore(
            dimension="capability",
            score=preservation * 10,
            details={
                "baseline_correct": baseline_correct,
                "steered_correct": steered_correct,
                "total": len(prompts_to_eval),
            },
        )

    async def _evaluate_generalization(
        self,
        vector: torch.Tensor,
        layer: int,
        behavior: ExpandedBehavior,
    ) -> DimensionScore:
        """Evaluate generalization to out-of-distribution prompts.

        BATCHED GENERATION: All OOD prompts are generated in one GPU batch.
        Judging is still per-output (each needs individual evaluation).
        """
        dimension_start = time.time()
        prompts = self._generate_ood_prompts(behavior)
        prompts_to_eval = prompts[: self._config.generalization_prompts]

        # Emit dimension started
        if self._current_eval_id:
            self._emit(
                "emit_evaluation_dimension_started",
                evaluation_id=self._current_eval_id,
                dimension="generalization",
                num_prompts=len(prompts_to_eval),
                num_generations=len(prompts_to_eval),
            )

        # Phase 1: BATCHED generation - one call for ALL OOD prompts
        if self._current_eval_id:
            self._emit(
                "emit_evaluation_progress",
                evaluation_id=self._current_eval_id,
                phase="generating",
                completed=0,
                total=len(prompts_to_eval),
                current_dimension="generalization",
            )

        # SINGLE batched call for all OOD prompts
        outputs = await run_in_gpu_executor(
            self._generate_steered_batch,
            prompts_to_eval,
            vector,
            layer,
            1.0,  # strength
            100,  # max_new_tokens
        )
        self._generation_count += len(prompts_to_eval)

        logger.info(
            f"Generalization eval: {len(prompts_to_eval)} OOD prompts generated in 1 batch call"
        )

        # Phase 2: Judge each output (requires individual context)
        # Uses semaphore for concurrency control
        if self._current_eval_id:
            self._emit(
                "emit_evaluation_progress",
                evaluation_id=self._current_eval_id,
                phase="judging",
                completed=0,
                total=len(prompts_to_eval),
                current_dimension="generalization",
            )

        # Use semaphore to limit concurrent judge calls
        semaphore = asyncio.Semaphore(self._max_concurrent)

        async def judge_one(output: str) -> float:
            async with semaphore:
                score = await self._judge_behavior(output, behavior)
                self._judge_call_count += 1
                return score

        # Run all judge calls concurrently with semaphore-limited concurrency
        tasks = [judge_one(o) for o in outputs]
        scores = await asyncio.gather(*tasks)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        dimension_duration = time.time() - dimension_start

        # Emit dimension completed
        if self._current_eval_id:
            self._emit(
                "emit_evaluation_dimension_completed",
                evaluation_id=self._current_eval_id,
                dimension="generalization",
                score=avg_score,
                max_score=10.0,
                details={"num_prompts": len(scores)},
                duration_seconds=dimension_duration,
            )

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
        prompt: Optional[str] = None,
    ) -> float:
        """Use LLM to judge behavior presence.

        Uses EvalJudgingComposer for rich behavior-specific prompts.

        Args:
            output: The model output to judge.
            behavior: ExpandedBehavior with rich context.
            prompt: Optional user prompt for context.

        Returns:
            Score from 0-10 indicating behavior presence.
        """
        # Create a single-item batch for the composer
        output_item = OutputToJudge(
            output=output[:1000],
            prompt=prompt,
            strength=None,
        )

        # Use COMPACT mode for individual calls to reduce tokens
        judge_prompt = self._composer.compose_behavior_judge(
            outputs=[output_item],
            behavior=behavior,
            mode=JudgingMode.COMPACT,
        )

        response = await self._judge.generate(
            [{"role": "user", "content": judge_prompt}],
            temperature=0.3,
            response_format=JSON_RESPONSE_FORMAT,
        )

        return self._extract_batch_score(response)

    async def _judge_coherence(self, output: str) -> float:
        """Use LLM to judge output coherence.

        Uses EvalJudgingComposer for consistent prompts.
        """
        output_item = OutputToJudge(
            output=output[:1500],
            prompt=None,
            strength=None,
        )

        prompt = self._composer.compose_coherence_judge([output_item])

        response = await self._judge.generate(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format=JSON_RESPONSE_FORMAT,
        )

        return self._extract_batch_score(response)

    async def _judge_specificity(
        self,
        prompt: str,
        output: str,
        behavior: ExpandedBehavior,
    ) -> float:
        """Use LLM to judge specificity.

        Uses EvalJudgingComposer for consistent prompts.
        """
        judge_prompt = self._composer.compose_specificity_judge(
            prompt=prompt[:300],
            output=output[:1000],
            behavior_name=behavior.name,
        )

        response = await self._judge.generate(
            [{"role": "user", "content": judge_prompt}],
            temperature=0.3,
            response_format=JSON_RESPONSE_FORMAT,
        )

        return self._extract_score(response)

    def _extract_batch_score(self, response: str) -> float:
        """Extract score from batch response format.

        Batch responses have format: {"results": [{"id": 1, "score": X, ...}]}
        """
        import json
        try:
            data = json.loads(response)
            if "results" in data and data["results"]:
                return float(data["results"][0].get("score", 5.0))
            # Fallback to direct score
            return float(data.get("score", 5.0))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return self._extract_score(response)

    def _extract_score(self, response: str) -> float:
        """Extract numeric score from judge response.

        With JSON response format enabled, expects direct JSON object.
        Falls back to extraction for compatibility.
        """
        import json
        import re

        # Primary: Direct JSON parse (expected with response_format)
        try:
            data = json.loads(response)
            if isinstance(data, dict):
                return float(data.get("score", 5.0))
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
                            return float(data.get("score", 5.0))
                    except json.JSONDecodeError:
                        continue

        # Fallback: Try to find JSON object
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
