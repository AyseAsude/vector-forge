"""Main pipeline orchestrator for steering vector extraction."""

from typing import Optional, Any

from vector_forge.core.protocols import EventEmitter
from vector_forge.core.state import ExtractionState
from vector_forge.core.config import PipelineConfig, LLMConfig
from vector_forge.core.behavior import BehaviorSpec
from vector_forge.core.results import ExtractionResult, Verdict
from vector_forge.core.events import EventType, create_event
from vector_forge.llm import LiteLLMClient
from vector_forge.agents.extractor import ExtractorAgent
from vector_forge.agents.judge import JudgeAgent
from vector_forge.strategies.noise import AveragingReducer, PCAReducer
from vector_forge.core.config import NoiseReductionType


class ExtractionPipeline(EventEmitter):
    """
    Main orchestrator for steering vector extraction.

    Coordinates the extractor agent (inner loop) and judge (outer loop)
    to autonomously find high-quality steering vectors.

    Example:
        >>> from steering_vectors import HuggingFaceBackend
        >>> backend = HuggingFaceBackend(model, tokenizer)
        >>> pipeline = ExtractionPipeline(
        ...     model_backend=backend,
        ...     config=PipelineConfig(),
        ... )
        >>> result = await pipeline.extract(
        ...     BehaviorSpec(description="sycophancy")
        ... )
    """

    def __init__(
        self,
        model_backend: Any,
        config: Optional[PipelineConfig] = None,
    ):
        """
        Args:
            model_backend: Backend for the model being steered (e.g., HuggingFaceBackend).
            config: Pipeline configuration. Uses defaults if not provided.
        """
        super().__init__()
        self.backend = model_backend
        self.config = config or PipelineConfig()

        # Create LLM clients
        self._extractor_llm = LiteLLMClient(self.config.extractor_llm)
        self._judge_llm = LiteLLMClient(self.config.judge_llm)

    async def extract(
        self,
        behavior: BehaviorSpec,
        max_outer_iterations: Optional[int] = None,
        max_inner_turns: int = 50,
    ) -> ExtractionResult:
        """
        Extract a steering vector for the specified behavior.

        Args:
            behavior: Specification of the target behavior.
            max_outer_iterations: Max judge-driven refinement loops.
            max_inner_turns: Max turns per extractor run.

        Returns:
            ExtractionResult with the best vector found.
        """
        max_outer = max_outer_iterations or self.config.max_outer_iterations

        self.emit(create_event(
            EventType.PIPELINE_STARTED,
            source="pipeline",
            behavior=behavior.name,
        ))

        state = ExtractionState()
        best_result: Optional[ExtractionResult] = None

        try:
            for outer_iter in range(max_outer):
                state.outer_iteration = outer_iter

                self.emit(create_event(
                    EventType.OUTER_ITERATION_STARTED,
                    source="pipeline",
                    iteration=outer_iter,
                ))

                # Run extractor agent
                extractor = ExtractorAgent(
                    state=state,
                    llm_client=self._extractor_llm,
                    model_backend=self.backend,
                    behavior=behavior,
                    config=self.config,
                )

                # Forward events
                extractor.on("*", lambda e: self.emit(e))

                result = await extractor.run(max_turns=max_inner_turns)

                if result is None:
                    self.emit(create_event(
                        EventType.WARNING,
                        source="pipeline",
                        message="Extractor failed to produce a result",
                    ))
                    continue

                # Run judge evaluation
                judge = JudgeAgent(
                    llm_client=self._judge_llm,
                    model_backend=self.backend,
                    behavior=behavior,
                    config=self.config,
                )
                judge.on("*", lambda e: self.emit(e))

                self.emit(create_event(EventType.JUDGE_STARTED, source="pipeline"))

                evaluation = await judge.evaluate(
                    result.vector,
                    result.recommended_layer,
                )

                result.evaluation = evaluation
                state.current_evaluation = evaluation
                state.evaluations.append(evaluation)

                self.emit(create_event(
                    EventType.JUDGE_VERDICT,
                    source="pipeline",
                    verdict=evaluation.verdict.value,
                    score=evaluation.scores.overall,
                ))

                # Update best result
                if best_result is None or evaluation.scores.overall > best_result.evaluation.scores.overall:
                    best_result = result

                # Check if we're done
                if evaluation.verdict == Verdict.ACCEPTED:
                    self.emit(create_event(
                        EventType.OUTER_ITERATION_COMPLETED,
                        source="pipeline",
                        iteration=outer_iter,
                        verdict="accepted",
                    ))
                    break

                if evaluation.verdict == Verdict.REJECTED:
                    self.emit(create_event(
                        EventType.WARNING,
                        source="pipeline",
                        message="Judge rejected vector, will retry with feedback",
                    ))

                # Feed recommendations back (for next iteration)
                state.log_action(
                    "judge_feedback",
                    {
                        "recommendations": evaluation.recommendations,
                        "scores": evaluation.scores.to_dict(),
                    },
                    agent="judge",
                )

                self.emit(create_event(
                    EventType.OUTER_ITERATION_COMPLETED,
                    source="pipeline",
                    iteration=outer_iter,
                    verdict=evaluation.verdict.value,
                ))

            # Apply noise reduction if configured
            if best_result and self.config.noise_reduction != NoiseReductionType.NONE:
                best_result = await self._apply_noise_reduction(
                    best_result, state, behavior
                )

            if best_result:
                self.emit(create_event(
                    EventType.PIPELINE_COMPLETED,
                    source="pipeline",
                    success=True,
                    score=best_result.evaluation.scores.overall,
                ))
            else:
                self.emit(create_event(
                    EventType.PIPELINE_FAILED,
                    source="pipeline",
                    reason="No valid result produced",
                ))
                # Return a minimal result
                raise RuntimeError("Pipeline failed to produce a valid result")

            return best_result

        except Exception as e:
            self.emit(create_event(
                EventType.PIPELINE_FAILED,
                source="pipeline",
                reason=str(e),
            ))
            raise

    async def _apply_noise_reduction(
        self,
        result: ExtractionResult,
        state: ExtractionState,
        behavior: BehaviorSpec,
    ) -> ExtractionResult:
        """Apply noise reduction to the final vector."""
        self.emit(create_event(EventType.NOISE_REDUCTION_STARTED, source="pipeline"))

        num_seeds = self.config.num_seeds_for_noise

        if num_seeds <= 1:
            return result

        # We already have one vector, train more with different seeds
        import torch

        vectors = [result.vector]

        # Train additional vectors
        for seed in range(1, num_seeds):
            torch.manual_seed(seed * 42)

            extractor = ExtractorAgent(
                state=ExtractionState(),  # Fresh state
                llm_client=self._extractor_llm,
                model_backend=self.backend,
                behavior=behavior,
                config=self.config,
            )

            additional_result = await extractor.run(max_turns=30)  # Shorter run
            if additional_result and additional_result.vector is not None:
                vectors.append(additional_result.vector)

        if len(vectors) < 2:
            return result

        # Apply reduction
        if self.config.noise_reduction == NoiseReductionType.AVERAGING:
            reducer = AveragingReducer()
        elif self.config.noise_reduction == NoiseReductionType.PCA:
            reducer = PCAReducer()
        else:
            return result

        clean_vector = reducer.reduce(vectors)

        result.vector = clean_vector
        result.noise_reduction_applied = True
        result.num_seeds_averaged = len(vectors)

        self.emit(create_event(
            EventType.NOISE_REDUCTION_COMPLETED,
            source="pipeline",
            num_vectors=len(vectors),
        ))

        return result
