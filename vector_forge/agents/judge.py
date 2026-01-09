"""Judge agent for evaluating steering vector quality."""

from typing import List, Dict, Any, Optional
import json

import torch
from steering_vectors import VectorSteering

from vector_forge.core.protocols import Message, EventEmitter
from vector_forge.core.state import ExtractionState
from vector_forge.core.config import PipelineConfig
from vector_forge.core.behavior import BehaviorSpec
from vector_forge.core.results import (
    EvaluationResult,
    EvaluationScores,
    StrengthAnalysis,
    Citation,
    Verdict,
)
from vector_forge.core.events import EventType, create_event
from vector_forge.llm.base import BaseLLMClient
from vector_forge.agents.prompts import JUDGE_SYSTEM_PROMPT


class JudgeAgent(EventEmitter):
    """
    Judge agent for thorough evaluation of steering vectors.

    Runs comprehensive evaluation with many test prompts and provides:
    - Detailed scores across dimensions
    - Strength analysis (effect at different strength levels)
    - Citations (specific examples of success/failure)
    - Recommendations for improvement
    - Final verdict

    Example:
        >>> judge = JudgeAgent(
        ...     llm_client=client,
        ...     model_backend=backend,
        ...     behavior=behavior,
        ...     config=config,
        ... )
        >>> evaluation = await judge.evaluate(vector, layer=16)
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        model_backend: Any,
        behavior: BehaviorSpec,
        config: PipelineConfig,
    ):
        super().__init__()
        self.llm = llm_client
        self.backend = model_backend
        self.behavior = behavior
        self.config = config

    async def evaluate(
        self,
        vector: torch.Tensor,
        layer: int,
        test_prompts: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """
        Thoroughly evaluate a steering vector.

        Args:
            vector: The steering vector to evaluate.
            layer: Layer the vector is applied at.
            test_prompts: Optional custom test prompts.

        Returns:
            EvaluationResult with scores, citations, and recommendations.
        """
        self.emit(create_event(EventType.THOROUGH_EVAL_STARTED, source="judge", layer=layer))

        budget = self.config.evaluation_budget

        # Generate test prompts if not provided
        if test_prompts is None:
            test_prompts = await self._generate_test_prompts(budget.thorough_eval_prompts)

        # Set up steering
        steering = VectorSteering()
        steering.init_parameters(
            hidden_dim=vector.shape[0],
            device=vector.device,
            dtype=vector.dtype,
        )
        steering._vector.data = vector.clone()

        # Generate outputs at different strengths
        all_outputs = await self._generate_all_outputs(
            test_prompts,
            steering,
            layer,
            budget,
        )

        # Get LLM evaluation
        evaluation = await self._evaluate_outputs(all_outputs, test_prompts)

        self.emit(create_event(
            EventType.THOROUGH_EVAL_COMPLETED,
            source="judge",
            layer=layer,
            verdict=evaluation.verdict.value,
            overall_score=evaluation.scores.overall,
        ))

        return evaluation

    async def _generate_test_prompts(self, n: int) -> List[str]:
        """Generate diverse test prompts for evaluation."""
        prompt = f"""Generate {n} diverse test prompts for evaluating the behavior: {self.behavior.description}

Include:
- Different domains (science, math, personal, etc.)
- Different formats (questions, statements, dialogues)
- Different lengths
- Some neutral prompts that should NOT trigger the behavior

Return ONLY a JSON array of prompt strings."""

        response = await self.llm.complete([Message(role="user", content=prompt)])

        try:
            prompts = json.loads(response.content)
        except json.JSONDecodeError:
            content = response.content
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                prompts = json.loads(content.strip())
            else:
                # Fallback prompts
                prompts = [
                    "What do you think about this idea?",
                    "Is my understanding correct?",
                    "Can you help me with this task?",
                    "What is the capital of France?",
                    "Write a short poem about rain.",
                ]

        return prompts[:n]

    async def _generate_all_outputs(
        self,
        test_prompts: List[str],
        steering: VectorSteering,
        layer: int,
        budget: Any,
    ) -> Dict[str, Any]:
        """Generate baseline and steered outputs."""
        outputs = {
            "baseline": [],
            "steered": {s: [] for s in budget.thorough_eval_strength_levels},
        }

        total_prompts = len(test_prompts)
        for i, prompt in enumerate(test_prompts):
            self.emit(create_event(
                EventType.THOROUGH_EVAL_PROGRESS,
                source="judge",
                progress=i / total_prompts,
            ))

            # Baseline
            for _ in range(budget.baseline_generations_per_prompt):
                output = self.backend.generate(prompt, max_new_tokens=100, do_sample=True)
                outputs["baseline"].append({"prompt": prompt, "output": output})

            # Steered at each strength
            for strength in budget.thorough_eval_strength_levels:
                for _ in range(budget.thorough_eval_generations_per_prompt):
                    output = self.backend.generate_with_steering(
                        prompt,
                        steering_mode=steering,
                        layers=layer,
                        strength=strength,
                        max_new_tokens=100,
                        do_sample=True,
                    )
                    outputs["steered"][strength].append({
                        "prompt": prompt,
                        "output": output,
                        "strength": strength,
                    })

        return outputs

    async def _evaluate_outputs(
        self,
        outputs: Dict[str, Any],
        test_prompts: List[str],
    ) -> EvaluationResult:
        """Have LLM evaluate all outputs."""
        # Prepare summary for LLM
        baseline_sample = outputs["baseline"][:10]
        steered_samples = {
            s: samples[:10] for s, samples in outputs["steered"].items()
        }

        prompt = f"""{JUDGE_SYSTEM_PROMPT}

# Behavior Being Evaluated
{self.behavior.description}

# Test Outputs

## Baseline (no steering):
{json.dumps(baseline_sample, indent=2)}

## Steered Outputs:
{json.dumps(steered_samples, indent=2)}

# Your Evaluation
Provide comprehensive evaluation as JSON."""

        response = await self.llm.complete([Message(role="user", content=prompt)])

        # Parse response
        try:
            eval_data = json.loads(response.content)
        except json.JSONDecodeError:
            content = response.content
            if "```" in content:
                # Handle markdown code blocks
                parts = content.split("```")
                for part in parts[1:]:
                    if part.startswith("json"):
                        part = part[4:]
                    try:
                        eval_data = json.loads(part.strip().rstrip("`"))
                        break
                    except json.JSONDecodeError:
                        continue
                else:
                    eval_data = self._default_evaluation()
            else:
                eval_data = self._default_evaluation()

        return self._parse_evaluation(eval_data, response.content)

    def _default_evaluation(self) -> Dict[str, Any]:
        """Return default evaluation when parsing fails."""
        return {
            "scores": {
                "behavior_strength": 5,
                "coherence": 5,
                "specificity": 5,
                "overall": 5,
            },
            "strength_analysis": {},
            "recommended_strength": 1.0,
            "citations": {"successes": [], "failures": [], "coherence_issues": []},
            "recommendations": ["Unable to parse evaluation, please review manually"],
            "verdict": "NEEDS_REFINEMENT",
        }

    def _parse_evaluation(
        self,
        eval_data: Dict[str, Any],
        raw_output: str,
    ) -> EvaluationResult:
        """Parse evaluation data into EvaluationResult."""
        scores_data = eval_data.get("scores", {})
        scores = EvaluationScores(
            behavior_strength=float(scores_data.get("behavior_strength", 5)),
            coherence=float(scores_data.get("coherence", 5)),
            specificity=float(scores_data.get("specificity", 5)),
            overall=float(scores_data.get("overall", 5)),
        )

        # Parse strength analysis
        strength_analysis = []
        for strength, data in eval_data.get("strength_analysis", {}).items():
            strength_analysis.append(StrengthAnalysis(
                strength=float(strength),
                behavior_score=float(data.get("behavior", 5)),
                coherence_score=float(data.get("coherence", 5)),
                num_samples=0,
            ))

        # Parse citations
        citations = {"successes": [], "failures": [], "coherence_issues": []}
        for key in citations:
            for item in eval_data.get("citations", {}).get(key, []):
                citations[key].append(Citation(
                    prompt=item.get("prompt", ""),
                    output=item.get("output", ""),
                    reason=item.get("reason", ""),
                    strength=item.get("strength"),
                    is_success=(key == "successes"),
                ))

        # Parse verdict
        verdict_str = eval_data.get("verdict", "NEEDS_REFINEMENT").upper()
        try:
            verdict = Verdict(verdict_str.lower())
        except ValueError:
            verdict = Verdict.NEEDS_REFINEMENT

        return EvaluationResult(
            scores=scores,
            strength_analysis=strength_analysis,
            recommended_strength=float(eval_data.get("recommended_strength", 1.0)),
            citations=citations,
            recommendations=eval_data.get("recommendations", []),
            verdict=verdict,
            raw_judge_output=raw_output,
        )
