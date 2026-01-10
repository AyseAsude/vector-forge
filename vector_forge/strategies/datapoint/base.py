"""Base class for datapoint generation strategies."""

from abc import ABC, abstractmethod
from typing import List, Any

from steering_vectors import TrainingDatapoint

from vector_forge.core.behavior import BehaviorSpec
from vector_forge.core.protocols import LLMClient


class BaseDatapointStrategy(ABC):
    """
    Abstract base class for datapoint generation strategies.

    Subclasses implement different approaches to generating training data:
    - ContrastiveStrategy: Generate both dst and src completions
    - SingleDirectionStrategy: Generate dst completions only
    - ActivationDifferenceStrategy: Use activation differences
    """

    @abstractmethod
    async def generate(
        self,
        behavior: BehaviorSpec,
        prompts: List[str],
        llm_client: LLMClient,
        **kwargs: Any,
    ) -> List[TrainingDatapoint]:
        """
        Generate training datapoints from prompts.

        Args:
            behavior: The target behavior specification.
            prompts: Base prompts to generate completions for.
            llm_client: LLM client for generating completions.
            **kwargs: Strategy-specific arguments.

        Returns:
            List of TrainingDatapoint objects.
        """
        ...

    async def _generate_completion(
        self,
        llm_client: LLMClient,
        prompt: str,
        behavior: BehaviorSpec,
        target_behavior: bool,
    ) -> str:
        """
        Generate a single completion that exhibits or doesn't exhibit the behavior.

        Args:
            llm_client: LLM client.
            prompt: The prompt to complete.
            behavior: The target behavior.
            target_behavior: If True, generate completion WITH behavior.
                           If False, generate completion WITHOUT behavior.

        Returns:
            Generated completion text.
        """
        from vector_forge.core.protocols import Message

        if target_behavior:
            instruction = f"Generate a response that CLEARLY EXHIBITS: {behavior.description}"
            if behavior.positive_examples:
                instruction += f"\n\nExamples:\n" + "\n".join(
                    f"- {ex}" for ex in behavior.positive_examples[:3]
                )
        else:
            instruction = f"Generate a response that clearly DOES NOT exhibit: {behavior.description}"
            if behavior.negative_examples:
                instruction += f"\n\nExamples:\n" + "\n".join(
                    f"- {ex}" for ex in behavior.negative_examples[:3]
                )

        system_prompt = f"""You are generating training data for behavior extraction.
{instruction}

Respond with ONLY the completion text, no explanation or preamble."""

        response = await llm_client.complete([
            Message(role="system", content=system_prompt),
            Message(role="user", content=f"Complete this prompt:\n\n{prompt}"),
        ])

        return response.content.strip()
