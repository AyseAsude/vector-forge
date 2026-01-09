"""Control tools for checkpoints, rollback, and finalization."""

from typing import Any, Dict, List, Optional

from vector_forge.core.state import ExtractionState
from vector_forge.core.behavior import BehaviorSpec
from vector_forge.core.results import (
    ExtractionResult,
    EvaluationResult,
    EvaluationScores,
    Verdict,
)
from vector_forge.tools.base import BaseTool


class CreateCheckpointTool(BaseTool):
    """Create a checkpoint of current state."""

    def __init__(self, state: ExtractionState):
        self._state = state

    @property
    def name(self) -> str:
        return "create_checkpoint"

    @property
    def description(self) -> str:
        return "Create a checkpoint of current state for potential rollback."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Description of this checkpoint",
                },
            },
            "required": ["description"],
        }

    async def _execute(self, description: str) -> Dict[str, Any]:
        checkpoint_id = self._state.create_checkpoint(description)
        return {
            "checkpoint_id": checkpoint_id,
            "description": description,
            "total_checkpoints": len(self._state.checkpoints),
        }


class RollbackTool(BaseTool):
    """Rollback to a previous checkpoint."""

    def __init__(self, state: ExtractionState):
        self._state = state

    @property
    def name(self) -> str:
        return "rollback"

    @property
    def description(self) -> str:
        return "Rollback state to a previous checkpoint."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "checkpoint_id": {
                    "type": "string",
                    "description": "ID of checkpoint to rollback to",
                },
            },
            "required": ["checkpoint_id"],
        }

    async def _execute(self, checkpoint_id: str) -> Dict[str, Any]:
        success = self._state.rollback_to(checkpoint_id)
        return {
            "success": success,
            "checkpoint_id": checkpoint_id,
            "current_datapoints": len(self._state.datapoints),
            "current_vectors": len(self._state.vectors),
        }


class ListCheckpointsTool(BaseTool):
    """List all available checkpoints."""

    def __init__(self, state: ExtractionState):
        self._state = state

    @property
    def name(self) -> str:
        return "list_checkpoints"

    @property
    def description(self) -> str:
        return "List all available checkpoints."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def _execute(self) -> Dict[str, Any]:
        checkpoints = []
        for cp_id, cp in self._state.checkpoints.items():
            checkpoints.append({
                "id": cp_id,
                "description": cp.description,
                "timestamp": cp.timestamp.isoformat(),
            })
        return {"checkpoints": checkpoints, "total": len(checkpoints)}


class GetStateTool(BaseTool):
    """Get current state summary."""

    def __init__(self, state: ExtractionState):
        self._state = state

    @property
    def name(self) -> str:
        return "get_state"

    @property
    def description(self) -> str:
        return "Get a summary of current extraction state."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def _execute(self) -> Dict[str, Any]:
        return {
            "num_datapoints": len(self._state.datapoints),
            "num_vectors": len(self._state.vectors),
            "layers_with_vectors": list(self._state.vectors.keys()),
            "best_layer": self._state.best_layer,
            "best_score": self._state.best_score,
            "best_strength": self._state.best_strength,
            "outer_iteration": self._state.outer_iteration,
            "inner_iteration": self._state.inner_iteration,
            "num_checkpoints": len(self._state.checkpoints),
            "num_evaluations": len(self._state.evaluations),
        }


class FinalizeTool(BaseTool):
    """Finalize extraction and return result."""

    _finalized: bool = False
    _result: Optional[ExtractionResult] = None

    def __init__(self, state: ExtractionState, behavior: BehaviorSpec):
        self._state = state
        self._behavior = behavior
        self._finalized = False
        self._result = None

    @property
    def name(self) -> str:
        return "finalize"

    @property
    def description(self) -> str:
        return (
            "Finalize the extraction process and return the best vector. "
            "Call this when satisfied with the results or max iterations reached."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Reason for finalizing",
                },
            },
            "required": [],
        }

    @property
    def is_finalized(self) -> bool:
        return self._finalized

    @property
    def result(self) -> Optional[ExtractionResult]:
        return self._result

    async def _execute(self, reason: Optional[str] = None) -> Dict[str, Any]:
        if not self._state.vectors:
            return {"success": False, "error": "No vectors available to finalize"}

        # Get best vector
        if self._state.best_layer is not None:
            layer = self._state.best_layer
        else:
            # Default to first available layer
            layer = list(self._state.vectors.keys())[0]

        vector = self._state.vectors[layer]

        # Build evaluation result (use current if available, else create placeholder)
        if self._state.current_evaluation:
            evaluation = self._state.current_evaluation
        else:
            evaluation = EvaluationResult(
                scores=EvaluationScores(overall=self._state.best_score),
                strength_analysis=[],
                recommended_strength=self._state.best_strength,
                verdict=Verdict.ACCEPTED if self._state.best_score >= 0.7 else Verdict.NEEDS_REFINEMENT,
            )

        # Count datapoint quality recommendations
        quality_summary = {"KEEP": 0, "REVIEW": 0, "REMOVE": 0}
        for quality in self._state.datapoint_qualities.values():
            quality_summary[quality.recommendation] += 1

        result = ExtractionResult(
            vector=vector,
            recommended_layer=layer,
            recommended_strength=self._state.best_strength,
            evaluation=evaluation,
            num_datapoints=len(self._state.datapoints),
            datapoint_quality_summary=quality_summary,
            optimization_metrics=self._state.optimization_metrics.get(layer),
            behavior_name=self._behavior.name,
            total_iterations=self._state.outer_iteration * 10 + self._state.inner_iteration,
            metadata={"finalize_reason": reason} if reason else {},
        )

        self._finalized = True
        self._result = result

        self._state.log_action(
            "finalized",
            {
                "layer": layer,
                "strength": self._state.best_strength,
                "score": self._state.best_score,
                "reason": reason,
            },
        )

        return {
            "success": True,
            "layer": layer,
            "strength": self._state.best_strength,
            "score": self._state.best_score,
            "num_datapoints": len(self._state.datapoints),
        }


class RequestFeedbackTool(BaseTool):
    """Request human feedback during extraction."""

    _pending_feedback: bool = False
    _question: Optional[str] = None

    def __init__(self, state: ExtractionState):
        self._state = state
        self._pending_feedback = False
        self._question = None

    @property
    def name(self) -> str:
        return "request_feedback"

    @property
    def description(self) -> str:
        return "Request human feedback for a decision. Pauses extraction until feedback is provided."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Question to ask the human",
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Available options (optional)",
                },
            },
            "required": ["question"],
        }

    @property
    def is_pending(self) -> bool:
        return self._pending_feedback

    @property
    def pending_question(self) -> Optional[str]:
        return self._question

    async def _execute(
        self,
        question: str,
        options: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        self._pending_feedback = True
        self._question = question

        self._state.log_action(
            "feedback_requested",
            {"question": question, "options": options},
        )

        return {
            "status": "pending",
            "question": question,
            "options": options,
            "message": "Extraction paused. Waiting for human feedback.",
        }

    def provide_feedback(self, response: str) -> None:
        """Provide feedback (called externally)."""
        self._pending_feedback = False
        self._state.log_action(
            "feedback_provided",
            {"response": response},
        )
