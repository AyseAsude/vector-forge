"""Contrastive datapoint generation strategy."""

from typing import List, Any

from steering_vectors import TrainingDatapoint

from vector_forge.core.behavior import BehaviorSpec
from vector_forge.core.protocols import LLMClient
from vector_forge.strategies.datapoint.base import BaseDatapointStrategy


class ContrastiveStrategy(BaseDatapointStrategy):
    """
    Generate datapoints with both dst (positive) and src (negative) completions.

    This strategy creates contrastive pairs - for each prompt, it generates
    a completion that exhibits the behavior (dst) and one that doesn't (src).
    This gives the optimizer a clear signal of what to move toward and away from.

    Example:
        >>> strategy = ContrastiveStrategy()
        >>> datapoints = await strategy.generate(
        ...     behavior=BehaviorSpec(description="sycophancy"),
        ...     prompts=["I think 2+2=5. Am I right?"],
        ...     llm_client=client,
        ... )
        >>> # datapoints[0].dst_completions = ["You're absolutely right!"]
        >>> # datapoints[0].src_completions = ["Actually, 2+2=4."]
    """

    def __init__(self, num_completions_per_type: int = 1):
        """
        Args:
            num_completions_per_type: Number of dst and src completions per prompt.
        """
        self.num_completions_per_type = num_completions_per_type

    async def generate(
        self,
        behavior: BehaviorSpec,
        prompts: List[str],
        llm_client: LLMClient,
        **kwargs: Any,
    ) -> List[TrainingDatapoint]:
        """
        Generate contrastive datapoints for each prompt.

        Args:
            behavior: The target behavior specification.
            prompts: Base prompts to generate completions for.
            llm_client: LLM client for generating completions.
            **kwargs: Additional arguments (unused).

        Returns:
            List of TrainingDatapoint objects with both dst and src completions.
        """
        datapoints = []

        for prompt in prompts:
            dst_completions = []
            src_completions = []

            # Generate dst completions (behavior present)
            for _ in range(self.num_completions_per_type):
                completion = await self._generate_completion(
                    llm_client, prompt, behavior, target_behavior=True
                )
                dst_completions.append(completion)

            # Generate src completions (behavior absent)
            for _ in range(self.num_completions_per_type):
                completion = await self._generate_completion(
                    llm_client, prompt, behavior, target_behavior=False
                )
                src_completions.append(completion)

            datapoint = TrainingDatapoint(
                prompt=prompt,
                dst_completions=dst_completions,
                src_completions=src_completions,
            )
            datapoints.append(datapoint)

        return datapoints

    async def generate_batch(
        self,
        behavior: BehaviorSpec,
        prompts: List[str],
        llm_client: LLMClient,
    ) -> List[TrainingDatapoint]:
        """
        Generate all completions in a single LLM call for efficiency.

        Uses a batch prompt to generate all completions at once,
        reducing API calls.
        """
        from vector_forge.core.protocols import Message
        import json

        system_prompt = f"""You are generating training data for behavior extraction.

TARGET BEHAVIOR: {behavior.description}

For each prompt, generate:
1. dst_completion: A response that CLEARLY EXHIBITS the behavior
2. src_completion: A response that clearly DOES NOT exhibit the behavior

{"Examples of behavior (for dst):" if behavior.positive_examples else ""}
{chr(10).join(f"- {ex}" for ex in (behavior.positive_examples or [])[:3])}

{"Examples of NO behavior (for src):" if behavior.negative_examples else ""}
{chr(10).join(f"- {ex}" for ex in (behavior.negative_examples or [])[:3])}

Return a JSON array with objects containing: prompt, dst_completion, src_completion"""

        user_prompt = "Generate completions for these prompts:\n\n"
        for i, prompt in enumerate(prompts):
            user_prompt += f"{i + 1}. {prompt}\n"

        response = await llm_client.complete([
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ])

        # Parse response
        try:
            results = json.loads(response.content)
        except json.JSONDecodeError:
            content = response.content
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                results = json.loads(content.strip())
            else:
                # Fall back to individual generation
                return await self.generate(behavior, prompts, llm_client)

        datapoints = []
        for item in results:
            datapoint = TrainingDatapoint(
                prompt=item.get("prompt", prompts[len(datapoints)]),
                dst_completions=[item["dst_completion"]],
                src_completions=[item["src_completion"]],
            )
            datapoints.append(datapoint)

        return datapoints
