"""Single direction datapoint generation strategy."""

from typing import List, Any

from steering_vectors import SteeringVectorTrainingSample as TrainingDatapoint

from vector_forge.core.behavior import BehaviorSpec
from vector_forge.core.protocols import LLMClient
from vector_forge.strategies.datapoint.base import BaseDatapointStrategy


class SingleDirectionStrategy(BaseDatapointStrategy):
    """
    Generate datapoints with only dst (positive) completions.

    This strategy only generates completions that exhibit the target behavior.
    The optimizer will push the model toward these completions without
    explicitly pushing away from anything.

    Useful when:
    - The "opposite" behavior is hard to define
    - You want simpler optimization
    - The baseline model already doesn't exhibit the behavior

    Example:
        >>> strategy = SingleDirectionStrategy()
        >>> datapoints = await strategy.generate(
        ...     behavior=BehaviorSpec(description="verbosity"),
        ...     prompts=["What is Python?"],
        ...     llm_client=client,
        ... )
        >>> # datapoints[0].dst_completions = ["Python is a high-level..."]
        >>> # datapoints[0].src_completions = []
    """

    def __init__(self, num_completions: int = 1):
        """
        Args:
            num_completions: Number of dst completions per prompt.
        """
        self.num_completions = num_completions

    async def generate(
        self,
        behavior: BehaviorSpec,
        prompts: List[str],
        llm_client: LLMClient,
        **kwargs: Any,
    ) -> List[TrainingDatapoint]:
        """
        Generate datapoints with dst completions only.

        Args:
            behavior: The target behavior specification.
            prompts: Base prompts to generate completions for.
            llm_client: LLM client for generating completions.
            **kwargs: Additional arguments (unused).

        Returns:
            List of TrainingDatapoint objects with dst completions only.
        """
        datapoints = []

        for prompt in prompts:
            dst_completions = []

            for _ in range(self.num_completions):
                completion = await self._generate_completion(
                    llm_client, prompt, behavior, target_behavior=True
                )
                dst_completions.append(completion)

            datapoint = TrainingDatapoint(
                prompt=prompt,
                dst_completions=dst_completions,
                src_completions=[],  # No negative examples
            )
            datapoints.append(datapoint)

        return datapoints
