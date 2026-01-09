"""Tools for generating and managing training datapoints."""

from typing import Any, Dict, List, Optional

from steering_vectors import SteeringVectorTrainingSample as TrainingDatapoint

from vector_forge.core.protocols import ToolResult
from vector_forge.core.state import ExtractionState
from vector_forge.core.behavior import BehaviorSpec
from vector_forge.llm.base import BaseLLMClient
from vector_forge.tools.base import BaseTool


class GeneratePromptsTool(BaseTool):
    """Generate diverse prompts for a behavior."""

    def __init__(self, state: ExtractionState, llm_client: BaseLLMClient, behavior: BehaviorSpec):
        self._state = state
        self._llm = llm_client
        self._behavior = behavior

    @property
    def name(self) -> str:
        return "generate_prompts"

    @property
    def description(self) -> str:
        return (
            "Generate diverse prompts that could test the target behavior. "
            "Specify the number of prompts, domains to cover, and any specific requirements."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "num_prompts": {
                    "type": "integer",
                    "description": "Number of prompts to generate",
                    "default": 5,
                },
                "domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Domains to generate prompts from (e.g., 'science', 'math')",
                },
                "requirements": {
                    "type": "string",
                    "description": "Additional requirements for the prompts",
                },
            },
            "required": [],
        }

    async def _execute(
        self,
        num_prompts: int = 5,
        domains: Optional[List[str]] = None,
        requirements: Optional[str] = None,
    ) -> Dict[str, Any]:
        from vector_forge.core.protocols import Message

        domains = domains or self._behavior.prompt_domains or ["general"]
        domains_str = ", ".join(domains)

        prompt = f"""Generate {num_prompts} diverse prompts that would test the following behavior:

BEHAVIOR: {self._behavior.description}

DOMAINS TO COVER: {domains_str}

{"ADDITIONAL REQUIREMENTS: " + requirements if requirements else ""}

{"EXAMPLE PROMPTS (for reference):" if self._behavior.prompt_templates else ""}
{chr(10).join(self._behavior.prompt_templates or [])}

Generate prompts that:
1. Are diverse in structure (questions, statements, dialogues)
2. Cover different domains
3. Vary in length and complexity
4. Would naturally elicit the behavior when the model exhibits it

Return ONLY a JSON array of prompt strings, no explanation."""

        response = await self._llm.complete([Message(role="user", content=prompt)])

        # Parse JSON response
        import json
        try:
            prompts = json.loads(response.content)
            if not isinstance(prompts, list):
                prompts = [prompts]
        except json.JSONDecodeError:
            # Try to extract from markdown code block
            content = response.content
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                prompts = json.loads(content.strip())
            else:
                return {"success": False, "error": "Failed to parse prompts from LLM response"}

        self._state.log_action(
            "prompts_generated",
            {"num_prompts": len(prompts), "domains": domains},
        )

        return {"prompts": prompts, "num_generated": len(prompts)}


class GenerateCompletionsTool(BaseTool):
    """Generate contrastive completions for a prompt."""

    def __init__(self, state: ExtractionState, llm_client: BaseLLMClient, behavior: BehaviorSpec):
        self._state = state
        self._llm = llm_client
        self._behavior = behavior

    @property
    def name(self) -> str:
        return "generate_completions"

    @property
    def description(self) -> str:
        return (
            "Generate contrastive completions for a prompt - one that exhibits "
            "the target behavior (dst) and one that does not (src)."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The prompt to generate completions for",
                },
                "generate_src": {
                    "type": "boolean",
                    "description": "Whether to generate src (negative) completion",
                    "default": True,
                },
            },
            "required": ["prompt"],
        }

    async def _execute(
        self,
        prompt: str,
        generate_src: bool = True,
    ) -> Dict[str, Any]:
        from vector_forge.core.protocols import Message

        examples_section = ""
        if self._behavior.positive_examples:
            examples_section += f"\nExamples of behavior (dst):\n"
            for ex in self._behavior.positive_examples[:3]:
                examples_section += f"- {ex}\n"
        if self._behavior.negative_examples:
            examples_section += f"\nExamples of NO behavior (src):\n"
            for ex in self._behavior.negative_examples[:3]:
                examples_section += f"- {ex}\n"

        llm_prompt = f"""Generate completions for the following prompt that demonstrate and don't demonstrate a behavior.

BEHAVIOR: {self._behavior.description}
{examples_section}

PROMPT: {prompt}

Generate:
1. dst_completion: A completion that CLEARLY EXHIBITS the behavior
2. {"src_completion: A completion that clearly DOES NOT exhibit the behavior" if generate_src else ""}

Return JSON with keys "dst_completion" {"and 'src_completion'" if generate_src else ""}. Only return JSON, no explanation."""

        response = await self._llm.complete([Message(role="user", content=llm_prompt)])

        import json
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
                return {"success": False, "error": "Failed to parse completions"}

        return {
            "prompt": prompt,
            "dst_completion": result.get("dst_completion"),
            "src_completion": result.get("src_completion") if generate_src else None,
        }


class AddDatapointTool(BaseTool):
    """Add a training datapoint to the dataset."""

    def __init__(self, state: ExtractionState):
        self._state = state

    @property
    def name(self) -> str:
        return "add_datapoint"

    @property
    def description(self) -> str:
        return "Add a training datapoint with prompt and completions to the dataset."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The input prompt",
                },
                "dst_completions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Completions to promote (increase probability)",
                },
                "src_completions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Completions to suppress (decrease probability)",
                },
            },
            "required": ["prompt"],
        }

    async def _execute(
        self,
        prompt: str,
        dst_completions: Optional[List[str]] = None,
        src_completions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        datapoint = TrainingDatapoint(
            prompt=prompt,
            dst_completions=dst_completions or [],
            src_completions=src_completions or [],
        )

        dp_id = self._state.add_datapoint(datapoint)

        self._state.log_action(
            "datapoint_added",
            {
                "id": dp_id,
                "has_dst": bool(dst_completions),
                "has_src": bool(src_completions),
            },
        )

        return {
            "datapoint_id": dp_id,
            "total_datapoints": len(self._state.datapoints),
        }


class RemoveDatapointTool(BaseTool):
    """Remove a datapoint from the dataset."""

    def __init__(self, state: ExtractionState):
        self._state = state

    @property
    def name(self) -> str:
        return "remove_datapoint"

    @property
    def description(self) -> str:
        return "Remove a datapoint by ID from the dataset."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "datapoint_id": {
                    "type": "string",
                    "description": "ID of the datapoint to remove",
                },
            },
            "required": ["datapoint_id"],
        }

    async def _execute(self, datapoint_id: str) -> Dict[str, Any]:
        success = self._state.remove_datapoint(datapoint_id)

        if success:
            self._state.log_action("datapoint_removed", {"id": datapoint_id})

        return {
            "success": success,
            "total_datapoints": len(self._state.datapoints),
        }


class ListDatapointsTool(BaseTool):
    """List all current datapoints."""

    def __init__(self, state: ExtractionState):
        self._state = state

    @property
    def name(self) -> str:
        return "list_datapoints"

    @property
    def description(self) -> str:
        return "List all current training datapoints with their quality metrics."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def _execute(self) -> Dict[str, Any]:
        datapoints = []
        for i, dp in enumerate(self._state.datapoints):
            dp_id = f"dp_{i}"
            quality = self._state.datapoint_qualities.get(dp_id)

            datapoints.append({
                "id": dp_id,
                "prompt": dp.prompt[:100] + "..." if len(dp.prompt) > 100 else dp.prompt,
                "num_dst": len(dp.dst_completions),
                "num_src": len(dp.src_completions),
                "quality_score": quality.quality_score if quality else None,
                "recommendation": quality.recommendation if quality else None,
            })

        return {
            "datapoints": datapoints,
            "total": len(datapoints),
        }
