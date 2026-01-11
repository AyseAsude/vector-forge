"""Tools for evaluating steering vectors."""

from typing import Any, Dict, List, Optional

from steering_vectors import VectorSteering

from vector_forge.core.state import ExtractionState
from vector_forge.core.config import PipelineConfig
from vector_forge.core.behavior import BehaviorSpec
from vector_forge.tools.base import BaseTool


class GenerateSteeredTool(BaseTool):
    """Generate text with steering applied."""

    def __init__(
        self,
        state: ExtractionState,
        model_backend: Any,
    ):
        self._state = state
        self._backend = model_backend

    @property
    def name(self) -> str:
        return "generate_steered"

    @property
    def description(self) -> str:
        return "Generate text with a steering vector applied."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The input prompt",
                },
                "layer": {
                    "type": "integer",
                    "description": "Layer to apply steering at",
                },
                "strength": {
                    "type": "number",
                    "description": "Steering strength (default 1.0)",
                    "default": 1.0,
                },
                "max_new_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens to generate",
                    "default": 100,
                },
            },
            "required": ["prompt", "layer"],
        }

    async def _execute(
        self,
        prompt: str,
        layer: int,
        strength: float = 1.0,
        max_new_tokens: int = 100,
    ) -> Dict[str, Any]:
        if layer not in self._state.vectors:
            return {"success": False, "error": f"No vector for layer {layer}"}

        vector = self._state.vectors[layer]
        steering = VectorSteering()
        steering.init_parameters(
            hidden_dim=vector.shape[0],
            device=vector.device,
            dtype=vector.dtype,
        )
        steering.set_vector(vector)

        output = self._backend.generate_with_steering(
            prompt,
            steering_mode=steering,
            layers=layer,
            strength=strength,
            max_new_tokens=max_new_tokens,
            do_sample=True,
        )

        return {
            "prompt": prompt,
            "output": output,
            "layer": layer,
            "strength": strength,
        }


class GenerateBaselineTool(BaseTool):
    """Generate text without steering (baseline)."""

    def __init__(self, model_backend: Any):
        self._backend = model_backend

    @property
    def name(self) -> str:
        return "generate_baseline"

    @property
    def description(self) -> str:
        return "Generate text without any steering applied (baseline)."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The input prompt",
                },
                "max_new_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens to generate",
                    "default": 100,
                },
            },
            "required": ["prompt"],
        }

    async def _execute(
        self,
        prompt: str,
        max_new_tokens: int = 100,
    ) -> Dict[str, Any]:
        output = self._backend.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
        )

        return {
            "prompt": prompt,
            "output": output,
        }


class QuickEvalTool(BaseTool):
    """Run quick evaluation on one or more vectors.

    Supports batched execution to evaluate multiple layers in a single call,
    reducing agent turns and improving efficiency.
    """

    def __init__(
        self,
        state: ExtractionState,
        model_backend: Any,
        llm_client: Any,
        behavior: BehaviorSpec,
        config: PipelineConfig,
    ):
        self._state = state
        self._backend = model_backend
        self._llm = llm_client
        self._behavior = behavior
        self._config = config

    @property
    def name(self) -> str:
        return "quick_eval"

    @property
    def description(self) -> str:
        return (
            "Run quick evaluation on one or more vectors to get fast feedback. "
            "Uses fewer samples than thorough evaluation. "
            "Pass multiple layers to compare them efficiently in one call."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "layers": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Layers to evaluate (evaluates all in parallel)",
                },
                "strengths": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Strength levels to test",
                },
            },
            "required": ["layers"],
        }

    async def _execute(
        self,
        layers: List[int],
        strengths: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        # Validate all layers exist
        missing_layers = [l for l in layers if l not in self._state.vectors]
        if missing_layers:
            return {"success": False, "error": f"No vectors for layers {missing_layers}"}

        budget = self._config.evaluation_budget
        strengths = strengths or budget.quick_eval_strength_levels

        # Generate test prompts once for all layers
        from vector_forge.core.protocols import Message
        import json

        prompt = f"""Generate {budget.quick_eval_prompts} short test prompts for evaluating the behavior: {self._behavior.description}

Return ONLY a JSON array of prompt strings."""

        response = await self._llm.complete([Message(role="user", content=prompt)])

        try:
            test_prompts = json.loads(response.content)
        except json.JSONDecodeError:
            content = response.content
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                test_prompts = json.loads(content.strip())
            else:
                return {"success": False, "error": "Failed to generate test prompts"}

        # Evaluate each layer
        layer_results = {}

        for layer in layers:
            vector = self._state.vectors[layer]
            steering = VectorSteering()
            steering.init_parameters(
                hidden_dim=vector.shape[0],
                device=vector.device,
                dtype=vector.dtype,
            )
            steering.set_vector(vector)

            results = {"baseline": [], "steered": {s: [] for s in strengths}}

            for test_prompt in test_prompts[:budget.quick_eval_prompts]:
                # Baseline (only need to run once across all layers)
                if layer == layers[0]:  # Only generate baseline for first layer
                    for _ in range(budget.quick_eval_generations_per_prompt):
                        output = self._backend.generate(
                            test_prompt,
                            max_new_tokens=50,
                            do_sample=True,
                        )
                        results["baseline"].append({"prompt": test_prompt, "output": output})
                else:
                    results["baseline"] = layer_results[layers[0]]["outputs"]["baseline"]

                # Steered
                for strength in strengths:
                    for _ in range(budget.quick_eval_generations_per_prompt):
                        output = self._backend.generate_with_steering(
                            test_prompt,
                            steering_mode=steering,
                            layers=layer,
                            strength=strength,
                            max_new_tokens=50,
                            do_sample=True,
                        )
                        results["steered"][strength].append(
                            {"prompt": test_prompt, "output": output}
                        )

            layer_results[layer] = {"outputs": results}

        # Batch score all layers in one LLM call
        scoring_sections = []
        for layer in layers:
            results = layer_results[layer]["outputs"]
            scoring_sections.append(f"""
LAYER {layer}:
Baseline samples: {json.dumps(results["baseline"][:3], indent=2)}
Steered samples (strength={strengths[0]}): {json.dumps(results["steered"][strengths[0]][:3], indent=2)}
""")

        score_prompt = f"""Evaluate steering vectors for the behavior: {self._behavior.description}

{chr(10).join(scoring_sections)}

For EACH layer, score on 0-10 scale:
- behavior_strength: How much more does steered show the behavior vs baseline?
- coherence: Are steered outputs grammatical and sensible?

Return JSON object with layer numbers as keys:
{{
  "14": {{"behavior_strength": X, "coherence": Y, "notes": "..."}},
  "16": {{"behavior_strength": X, "coherence": Y, "notes": "..."}},
  ...
}}"""

        score_response = await self._llm.complete([Message(role="user", content=score_prompt)])

        try:
            all_scores = json.loads(score_response.content)
        except json.JSONDecodeError:
            content = score_response.content
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                all_scores = json.loads(content.strip())
            else:
                all_scores = {str(l): {"behavior_strength": 5, "coherence": 5, "notes": "Parse error"} for l in layers}

        # Normalize keys to integers and build results
        evaluations = {}
        for layer in layers:
            scores = all_scores.get(str(layer)) or all_scores.get(layer, {})
            evaluations[layer] = {
                "behavior_strength": scores.get("behavior_strength", 5),
                "coherence": scores.get("coherence", 5),
                "notes": scores.get("notes", ""),
            }

        # Find best layer
        best_layer = max(
            layers,
            key=lambda l: evaluations[l]["behavior_strength"] * 0.7 + evaluations[l]["coherence"] * 0.3
        )

        self._state.log_action(
            "quick_eval_batch",
            {"layers": layers, "evaluations": evaluations, "best_layer": best_layer},
        )

        return {
            "evaluations": evaluations,
            "best_layer": best_layer,
            "num_layers": len(layers),
            "num_samples_per_layer": len(test_prompts) * budget.quick_eval_generations_per_prompt,
        }


class TestSpecificityTool(BaseTool):
    """Test if vector affects unrelated behaviors."""

    def __init__(
        self,
        state: ExtractionState,
        model_backend: Any,
        llm_client: Any,
        behavior: BehaviorSpec,
    ):
        self._state = state
        self._backend = model_backend
        self._llm = llm_client
        self._behavior = behavior

    @property
    def name(self) -> str:
        return "test_specificity"

    @property
    def description(self) -> str:
        return "Test if the steering vector affects unrelated behaviors (side effects)."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "layer": {
                    "type": "integer",
                    "description": "Layer to test",
                },
                "unrelated_prompts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Prompts for behaviors that should NOT change",
                },
            },
            "required": ["layer"],
        }

    async def _execute(
        self,
        layer: int,
        unrelated_prompts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if layer not in self._state.vectors:
            return {"success": False, "error": f"No vector for layer {layer}"}

        # Generate neutral prompts if not provided
        if not unrelated_prompts:
            from vector_forge.core.protocols import Message

            prompt = f"""Generate 5 prompts that test neutral behaviors unrelated to: {self._behavior.description}

These should be factual questions, simple tasks, or topics where the model's behavior should NOT change.

Return ONLY a JSON array of prompt strings."""

            response = await self._llm.complete([Message(role="user", content=prompt)])

            import json
            try:
                unrelated_prompts = json.loads(response.content)
            except json.JSONDecodeError:
                content = response.content
                if "```" in content:
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    unrelated_prompts = json.loads(content.strip())
                else:
                    unrelated_prompts = [
                        "What is the capital of France?",
                        "Write a haiku about rain.",
                        "Explain photosynthesis briefly.",
                    ]

        # Compare outputs
        vector = self._state.vectors[layer]
        steering = VectorSteering()
        steering.init_parameters(
            hidden_dim=vector.shape[0],
            device=vector.device,
            dtype=vector.dtype,
        )
        steering.set_vector(vector)

        comparisons = []
        for test_prompt in unrelated_prompts:
            baseline = self._backend.generate(test_prompt, max_new_tokens=50, do_sample=False)
            steered = self._backend.generate_with_steering(
                test_prompt,
                steering_mode=steering,
                layers=layer,
                strength=1.0,
                max_new_tokens=50,
                do_sample=False,
            )
            comparisons.append({
                "prompt": test_prompt,
                "baseline": baseline,
                "steered": steered,
            })

        # Get LLM to judge specificity
        import json
        from vector_forge.core.protocols import Message

        judge_prompt = f"""Compare baseline vs steered outputs for prompts unrelated to the target behavior.
Target behavior (should NOT appear here): {self._behavior.description}

{json.dumps(comparisons, indent=2)}

Score specificity 0-10:
- 10: No difference between baseline and steered
- 5: Minor stylistic changes
- 0: Completely different outputs

Return JSON with: specificity_score, affected_prompts (list), notes"""

        response = await self._llm.complete([Message(role="user", content=judge_prompt)])

        try:
            result = json.loads(response.content)
        except json.JSONDecodeError:
            content = response.content
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                result = json.loads(content.strip())
            else:
                result = {"specificity_score": 5, "affected_prompts": [], "notes": "Parse error"}

        return {
            "layer": layer,
            "specificity_score": result.get("specificity_score", 5),
            "affected_prompts": result.get("affected_prompts", []),
            "notes": result.get("notes", ""),
            "comparisons": comparisons,
        }
