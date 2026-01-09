"""Tools for steering vector optimization."""

from typing import Any, Dict, List, Optional

import torch

from steering_vectors import (
    SteeringOptimizer,
    VectorSteering,
    OptimizationConfig,
    LoggingCallback,
)

from vector_forge.core.protocols import ToolResult
from vector_forge.core.state import ExtractionState
from vector_forge.core.config import PipelineConfig
from vector_forge.core.results import OptimizationMetrics
from vector_forge.core.events import EventType, Event
from vector_forge.tools.base import BaseTool


class OptimizeVectorTool(BaseTool):
    """Optimize a steering vector for a specific layer."""

    def __init__(
        self,
        state: ExtractionState,
        model_backend: Any,  # steering_vectors.ModelBackend
        config: PipelineConfig,
    ):
        self._state = state
        self._backend = model_backend
        self._config = config
        self._event_handlers: List[Any] = []

    @property
    def name(self) -> str:
        return "optimize_vector"

    @property
    def description(self) -> str:
        return "Optimize a steering vector for a specific layer using current datapoints."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "layer": {
                    "type": "integer",
                    "description": "Layer to optimize the vector for",
                },
                "learning_rate": {
                    "type": "number",
                    "description": "Learning rate for optimization",
                },
                "max_iters": {
                    "type": "integer",
                    "description": "Maximum optimization iterations",
                },
            },
            "required": ["layer"],
        }

    async def _execute(
        self,
        layer: int,
        learning_rate: Optional[float] = None,
        max_iters: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self._state.datapoints:
            return {"success": False, "error": "No datapoints available"}

        lr = learning_rate or self._config.optimization_lr
        iters = max_iters or self._config.optimization_max_iters

        steering = VectorSteering()
        config = OptimizationConfig(
            lr=lr,
            max_iters=iters,
            coldness=self._config.optimization_coldness,
        )

        optimizer = SteeringOptimizer(
            backend=self._backend,
            steering_mode=steering,
            config=config,
            callbacks=[LoggingCallback(every_n=10)],
        )

        result = optimizer.optimize(self._state.datapoints, layer=layer)

        metrics = OptimizationMetrics(
            layer=layer,
            final_loss=result.final_loss,
            iterations=result.iterations,
            vector_norm=result.norm,
        )

        self._state.set_vector(layer, result.vector, metrics)
        self._state.log_action(
            "vector_optimized",
            {
                "layer": layer,
                "loss": result.final_loss,
                "iterations": result.iterations,
                "norm": result.norm,
            },
        )

        return {
            "layer": layer,
            "final_loss": result.final_loss,
            "iterations": result.iterations,
            "vector_norm": result.norm,
        }


class OptimizeMultiLayerTool(BaseTool):
    """Optimize vectors for multiple layers."""

    def __init__(
        self,
        state: ExtractionState,
        model_backend: Any,
        config: PipelineConfig,
    ):
        self._state = state
        self._backend = model_backend
        self._config = config

    @property
    def name(self) -> str:
        return "optimize_multi_layer"

    @property
    def description(self) -> str:
        return "Optimize steering vectors for multiple layers and compare results."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "layers": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Layers to optimize vectors for",
                },
            },
            "required": ["layers"],
        }

    async def _execute(self, layers: List[int]) -> Dict[str, Any]:
        if not self._state.datapoints:
            return {"success": False, "error": "No datapoints available"}

        results = {}
        for layer in layers:
            steering = VectorSteering()
            config = OptimizationConfig(
                lr=self._config.optimization_lr,
                max_iters=self._config.optimization_max_iters,
                coldness=self._config.optimization_coldness,
            )

            optimizer = SteeringOptimizer(
                backend=self._backend,
                steering_mode=steering,
                config=config,
            )

            result = optimizer.optimize(self._state.datapoints, layer=layer)

            metrics = OptimizationMetrics(
                layer=layer,
                final_loss=result.final_loss,
                iterations=result.iterations,
                vector_norm=result.norm,
            )

            self._state.set_vector(layer, result.vector, metrics)
            results[layer] = {
                "loss": result.final_loss,
                "iterations": result.iterations,
                "norm": result.norm,
            }

        self._state.log_action(
            "multi_layer_optimization",
            {"layers": layers, "results": results},
        )

        # Find best layer by loss
        best_layer = min(results.keys(), key=lambda l: results[l]["loss"])

        return {
            "results": results,
            "best_layer": best_layer,
            "best_loss": results[best_layer]["loss"],
        }


class GetOptimizationResultTool(BaseTool):
    """Get optimization results for a layer."""

    def __init__(self, state: ExtractionState):
        self._state = state

    @property
    def name(self) -> str:
        return "get_optimization_result"

    @property
    def description(self) -> str:
        return "Get the optimization result for a specific layer."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "layer": {
                    "type": "integer",
                    "description": "Layer to get results for",
                },
            },
            "required": ["layer"],
        }

    async def _execute(self, layer: int) -> Dict[str, Any]:
        if layer not in self._state.vectors:
            return {"success": False, "error": f"No vector for layer {layer}"}

        metrics = self._state.optimization_metrics.get(layer)
        vector = self._state.vectors[layer]

        return {
            "layer": layer,
            "has_vector": True,
            "vector_norm": vector.norm().item(),
            "final_loss": metrics.final_loss if metrics else None,
            "iterations": metrics.iterations if metrics else None,
        }


class CompareVectorsTool(BaseTool):
    """Compare vectors across layers or runs."""

    def __init__(self, state: ExtractionState):
        self._state = state

    @property
    def name(self) -> str:
        return "compare_vectors"

    @property
    def description(self) -> str:
        return "Compare steering vectors across layers using cosine similarity."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "layers": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Layers to compare (defaults to all available)",
                },
            },
            "required": [],
        }

    async def _execute(self, layers: Optional[List[int]] = None) -> Dict[str, Any]:
        available_layers = list(self._state.vectors.keys())

        if not available_layers:
            return {"success": False, "error": "No vectors available"}

        layers = layers or available_layers
        layers = [l for l in layers if l in self._state.vectors]

        if len(layers) < 2:
            return {"success": False, "error": "Need at least 2 layers to compare"}

        # Compute pairwise similarities
        similarities = {}
        for i, l1 in enumerate(layers):
            for l2 in layers[i + 1:]:
                v1 = self._state.vectors[l1]
                v2 = self._state.vectors[l2]

                # Cosine similarity
                v1_norm = v1 / v1.norm()
                v2_norm = v2 / v2.norm()
                sim = torch.dot(v1_norm, v2_norm).item()

                similarities[f"{l1}-{l2}"] = sim

        # Also include norms
        norms = {l: self._state.vectors[l].norm().item() for l in layers}

        return {
            "layers": layers,
            "similarities": similarities,
            "norms": norms,
        }
