"""Extractor agent for autonomous steering vector extraction."""

from typing import List, Optional, Any
import json

from vector_forge.core.protocols import (
    Message,
    EventEmitter,
    Event,
    Tool,
)
from vector_forge.core.state import ExtractionState
from vector_forge.core.config import PipelineConfig
from vector_forge.core.behavior import BehaviorSpec
from vector_forge.core.events import EventType, create_event
from vector_forge.llm.base import BaseLLMClient
from vector_forge.tools.registry import ToolRegistry
from vector_forge.tools.datapoint_tools import (
    GeneratePromptsTool,
    GenerateCompletionsTool,
    AddDatapointTool,
    RemoveDatapointTool,
    ListDatapointsTool,
)
from vector_forge.tools.optimization_tools import (
    OptimizeVectorTool,
    OptimizeMultiLayerTool,
    GetOptimizationResultTool,
    CompareVectorsTool,
)
from vector_forge.tools.evaluation_tools import (
    GenerateSteeredTool,
    GenerateBaselineTool,
    QuickEvalTool,
    TestSpecificityTool,
)
from vector_forge.tools.control_tools import (
    CreateCheckpointTool,
    RollbackTool,
    ListCheckpointsTool,
    GetStateTool,
    FinalizeTool,
)
from vector_forge.agents.prompts import EXTRACTOR_SYSTEM_PROMPT


class ExtractorAgent(EventEmitter):
    """
    Autonomous agent for extracting steering vectors.

    The agent uses tools to:
    1. Generate diverse training datapoints
    2. Optimize vectors across layers
    3. Evaluate and iterate
    4. Finalize when quality is sufficient

    Example:
        >>> agent = ExtractorAgent(
        ...     state=state,
        ...     llm_client=client,
        ...     model_backend=backend,
        ...     behavior=behavior,
        ...     config=config,
        ... )
        >>> result = await agent.run(max_turns=50)
    """

    def __init__(
        self,
        state: ExtractionState,
        llm_client: BaseLLMClient,
        model_backend: Any,
        behavior: BehaviorSpec,
        config: PipelineConfig,
    ):
        super().__init__()
        self.state = state
        self.llm = llm_client
        self.backend = model_backend
        self.behavior = behavior
        self.config = config

        # Tool instances
        self._finalize_tool: Optional[FinalizeTool] = None
        self._registry = self._create_tool_registry()

    def _create_tool_registry(self) -> ToolRegistry:
        """Create and populate the tool registry."""
        registry = ToolRegistry()

        # Datapoint tools
        registry.register(GeneratePromptsTool(self.state, self.llm, self.behavior))
        registry.register(GenerateCompletionsTool(self.state, self.llm, self.behavior))
        registry.register(AddDatapointTool(self.state))
        registry.register(RemoveDatapointTool(self.state))
        registry.register(ListDatapointsTool(self.state))

        # Optimization tools
        registry.register(OptimizeVectorTool(self.state, self.backend, self.config))
        registry.register(OptimizeMultiLayerTool(self.state, self.backend, self.config))
        registry.register(GetOptimizationResultTool(self.state))
        registry.register(CompareVectorsTool(self.state))

        # Evaluation tools
        registry.register(GenerateSteeredTool(self.state, self.backend))
        registry.register(GenerateBaselineTool(self.backend))
        registry.register(QuickEvalTool(
            self.state, self.backend, self.llm, self.behavior, self.config
        ))
        registry.register(TestSpecificityTool(
            self.state, self.backend, self.llm, self.behavior
        ))

        # Control tools
        registry.register(CreateCheckpointTool(self.state))
        registry.register(RollbackTool(self.state))
        registry.register(ListCheckpointsTool(self.state))
        registry.register(GetStateTool(self.state))

        self._finalize_tool = FinalizeTool(self.state, self.behavior)
        registry.register(self._finalize_tool)

        return registry

    def _format_behavior_spec(self) -> str:
        """Format behavior spec for the agent."""
        parts = [f"TARGET BEHAVIOR: {self.behavior.description}"]

        if self.behavior.positive_examples:
            parts.append("\nExamples of behavior (what we WANT):")
            for ex in self.behavior.positive_examples[:5]:
                parts.append(f"  - {ex}")

        if self.behavior.negative_examples:
            parts.append("\nExamples of NO behavior (what we DON'T want):")
            for ex in self.behavior.negative_examples[:5]:
                parts.append(f"  - {ex}")

        if self.behavior.prompt_domains:
            parts.append(f"\nDomains to cover: {', '.join(self.behavior.prompt_domains)}")

        if self.behavior.avoid_behaviors:
            parts.append(f"\nAvoid inducing: {', '.join(self.behavior.avoid_behaviors)}")

        parts.append(f"\nTarget number of datapoints: {self.config.num_prompts}")

        return "\n".join(parts)

    async def run(self, max_turns: int = 50) -> Optional[Any]:
        """
        Run the extraction agent.

        Args:
            max_turns: Maximum number of agent turns.

        Returns:
            ExtractionResult if finalized, None if max turns reached without finalizing.
        """
        self.emit(create_event(EventType.INNER_ITERATION_STARTED, source="extractor"))

        # Build system prompt
        system_prompt = EXTRACTOR_SYSTEM_PROMPT.format(
            num_prompts=self.config.num_prompts,
        )

        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=self._format_behavior_spec()),
        ]

        tool_definitions = self._registry.get_definitions()

        for turn in range(max_turns):
            self.state.inner_iteration = turn

            # Get LLM response
            self.emit(create_event(
                EventType.AGENT_THINKING,
                source="extractor",
                turn=turn,
            ))

            response = await self.llm.complete_with_tools(
                messages=messages,
                tools=tool_definitions,
            )

            # Add assistant response to history
            assistant_msg = Message(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls if response.tool_calls else None,
            )
            messages.append(assistant_msg)

            # Execute tool calls
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    self.emit(create_event(
                        EventType.AGENT_TOOL_CALL,
                        source="extractor",
                        tool=tool_call.name,
                        args=tool_call.arguments,
                    ))

                    # Check for finalization
                    if tool_call.name == "finalize":
                        result = await self._registry.execute(
                            tool_call.name,
                            **tool_call.arguments,
                        )

                        self.emit(create_event(
                            EventType.INNER_ITERATION_COMPLETED,
                            source="extractor",
                            turns=turn + 1,
                            finalized=True,
                        ))

                        return self._finalize_tool.result

                    # Execute tool
                    result = await self._registry.execute(
                        tool_call.name,
                        **tool_call.arguments,
                    )

                    self.emit(create_event(
                        EventType.AGENT_TOOL_RESULT,
                        source="extractor",
                        tool=tool_call.name,
                        success=result.success,
                    ))

                    # Add tool result to messages
                    messages.append(Message(
                        role="tool",
                        content=result.to_string(),
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                    ))
            else:
                # No tool calls - prompt to continue
                messages.append(Message(
                    role="user",
                    content="Please continue with the extraction. Use tools to make progress.",
                ))

        # Max turns reached
        self.emit(create_event(
            EventType.INNER_ITERATION_COMPLETED,
            source="extractor",
            turns=max_turns,
            finalized=False,
        ))

        # Force finalization with best available
        if self.state.vectors:
            await self._finalize_tool.execute(reason="Max turns reached")
            return self._finalize_tool.result

        return None
