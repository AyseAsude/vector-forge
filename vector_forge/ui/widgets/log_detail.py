"""Log detail rendering strategies.

Uses Strategy pattern to provide event-type-specific detail views.
Each renderer knows how to format its event type's payload for display.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type
import json

from vector_forge.ui.state import LogEntry


class LogDetailRenderer(ABC):
    """Base class for log detail renderers (Strategy pattern).

    Each subclass handles rendering for a specific event type or category.
    """

    @abstractmethod
    def can_render(self, event_type: Optional[str]) -> bool:
        """Check if this renderer handles the given event type."""
        pass

    @abstractmethod
    def render_sections(
        self,
        entry: LogEntry,
    ) -> List[Tuple[str, str]]:
        """Render the log entry into titled sections.

        Returns:
            List of (section_title, section_content) tuples.
            Empty section_title means no header for that section.
        """
        pass

    def _truncate(self, text: str, max_len: int = 0) -> str:
        """Return text unchanged - detail view shows full content."""
        return text

    def _format_json(self, data: Any, indent: int = 2) -> str:
        """Format data as pretty JSON."""
        try:
            return json.dumps(data, indent=indent, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(data)


class LLMRequestRenderer(LogDetailRenderer):
    """Renderer for llm.request events - shows full messages and config."""

    def can_render(self, event_type: Optional[str]) -> bool:
        return event_type == "llm.request"

    def render_sections(self, entry: LogEntry) -> List[Tuple[str, str]]:
        payload = entry.payload or {}
        sections = []

        # Model info
        model = payload.get("model", "unknown")
        temp = payload.get("temperature")
        max_tokens = payload.get("max_tokens")
        config_parts = [f"Model: {model}"]
        if temp is not None:
            config_parts.append(f"Temperature: {temp}")
        if max_tokens is not None:
            config_parts.append(f"Max tokens: {max_tokens}")
        sections.append(("CONFIG", "\n".join(config_parts)))

        # Messages
        messages = payload.get("messages", [])
        if messages:
            msg_lines = []
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                # Truncate very long messages but keep them readable
                content = self._truncate(content)
                msg_lines.append(f"[{role}]\n{content}")
            sections.append(("MESSAGES", "\n\n".join(msg_lines)))

        # Tools if present
        tools = payload.get("tools")
        if tools:
            tool_names = [t.get("function", {}).get("name", "?") for t in tools]
            sections.append(("TOOLS", ", ".join(tool_names)))

        return sections


class LLMResponseRenderer(LogDetailRenderer):
    """Renderer for llm.response events - shows content and tool calls."""

    def can_render(self, event_type: Optional[str]) -> bool:
        return event_type == "llm.response"

    def render_sections(self, entry: LogEntry) -> List[Tuple[str, str]]:
        payload = entry.payload or {}
        sections = []

        # Metrics
        latency = payload.get("latency_ms", 0)
        finish_reason = payload.get("finish_reason", "stop")
        usage = payload.get("usage", {})
        metrics = f"Latency: {latency}ms | Finish: {finish_reason}"
        if usage:
            tokens = usage.get("total_tokens") or usage.get("estimated_tokens", 0)
            metrics += f" | Tokens: {tokens}"
        sections.append(("METRICS", metrics))

        # Error if present
        error = payload.get("error")
        if error:
            sections.append(("ERROR", f"[$error]{error}[/]"))

        # Content
        content = payload.get("content", "")
        if content:
            sections.append(("RESPONSE", self._truncate(content)))

        # Tool calls
        tool_calls = payload.get("tool_calls", [])
        if tool_calls:
            tc_lines = []
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "unknown")
                args = func.get("arguments", "{}")
                # Try to pretty-print the arguments
                try:
                    args_obj = json.loads(args) if isinstance(args, str) else args
                    args_str = self._format_json(args_obj)
                except (json.JSONDecodeError, TypeError):
                    args_str = str(args)
                tc_lines.append(f"[{name}]\n{self._truncate(args_str)}")
            sections.append(("TOOL CALLS", "\n\n".join(tc_lines)))

        return sections


class ContrastPairRenderer(LogDetailRenderer):
    """Renderer for contrast pair events - shows prompt and responses."""

    def can_render(self, event_type: Optional[str]) -> bool:
        return event_type in ("contrast.pair_generated", "contrast.pair_validated")

    def render_sections(self, entry: LogEntry) -> List[Tuple[str, str]]:
        payload = entry.payload or {}
        sections = []

        # Metadata
        pair_id = payload.get("pair_id", "unknown")
        sample_idx = payload.get("sample_idx", 0)
        seed_id = payload.get("seed_id", "")[:12]
        meta = f"Pair ID: {pair_id}\nSample: {sample_idx}"
        if seed_id:
            meta += f"\nSeed: {seed_id}"
        sections.append(("METADATA", meta))

        # Validation scores (if validated)
        if entry.event_type == "contrast.pair_validated":
            is_valid = payload.get("is_valid", False)
            status = "[$success]VALID[/]" if is_valid else "[$error]REJECTED[/]"
            dst_score = payload.get("dst_score", 0.0)
            src_score = payload.get("src_score", 0.0)
            quality = payload.get("contrast_quality", 0.0)
            semantic_dist = payload.get("semantic_distance", 0.0)
            scores = f"Status: {status}\n"
            scores += f"Target score: {dst_score:.3f} | Source score: {src_score:.3f}\n"
            scores += f"Contrast quality: {quality:.3f} | Semantic distance: {semantic_dist:.3f}"
            if not is_valid:
                reason = payload.get("rejection_reason", "unknown")
                scores += f"\nRejection reason: {reason}"
            sections.append(("VALIDATION SCORES", scores))

        # Prompt
        prompt = payload.get("prompt", "")
        if prompt:
            sections.append(("PROMPT", self._truncate(prompt)))

        # Responses side by side
        dst = payload.get("dst_response", "")
        src = payload.get("src_response", "")
        if dst:
            sections.append(("TARGET RESPONSE (desired behavior)", self._truncate(dst)))
        if src:
            sections.append(("SOURCE RESPONSE (baseline)", self._truncate(src)))

        # If validation event without content, show note and raw payload
        if entry.event_type == "contrast.pair_validated" and not prompt and not dst:
            sections.append(("NOTE", "Pair content available in 'contrast.pair_generated' event"))
            sections.append(("RAW PAYLOAD", self._format_json(payload)))

        return sections


class DatapointRenderer(LogDetailRenderer):
    """Renderer for datapoint events - shows prompt and completions."""

    def can_render(self, event_type: Optional[str]) -> bool:
        return event_type in ("datapoint.added", "datapoint.removed", "datapoint.quality")

    def render_sections(self, entry: LogEntry) -> List[Tuple[str, str]]:
        payload = entry.payload or {}
        sections = []

        # Metadata
        dp_id = payload.get("datapoint_id", "unknown")
        domain = payload.get("domain", "general")
        meta = f"Datapoint ID: {dp_id}\nDomain: {domain}"

        if entry.event_type == "datapoint.removed":
            reason = payload.get("reason", "no reason provided")
            meta += f"\n[$warning]Removal reason: {reason}[/]"

        if entry.event_type == "datapoint.quality":
            quality = payload.get("quality_score", 0.0)
            rec = payload.get("recommendation", "KEEP")
            meta += f"\nQuality: {quality:.3f} | Recommendation: {rec}"
            if payload.get("is_outlier"):
                meta += " [$warning](OUTLIER)[/]"

        sections.append(("METADATA", meta))

        # Prompt
        prompt = payload.get("prompt", "")
        if prompt:
            sections.append(("PROMPT", self._truncate(prompt)))

        # Completions
        positive = payload.get("positive_completion", "")
        negative = payload.get("negative_completion", "")
        if positive:
            sections.append(("POSITIVE COMPLETION", self._truncate(positive)))
        if negative:
            sections.append(("NEGATIVE COMPLETION", self._truncate(negative)))

        return sections


class OptimizationRenderer(LogDetailRenderer):
    """Renderer for optimization events - shows config and metrics."""

    def can_render(self, event_type: Optional[str]) -> bool:
        return event_type in (
            "optimization.started",
            "optimization.progress",
            "optimization.completed",
            "optimization.aggregation_completed",
        )

    def render_sections(self, entry: LogEntry) -> List[Tuple[str, str]]:
        payload = entry.payload or {}
        sections = []

        event_type = entry.event_type

        if event_type == "optimization.started":
            sample_idx = payload.get("sample_idx", 0)
            layer = payload.get("layer", 0)
            num_dp = payload.get("num_datapoints", 0)
            config = payload.get("config", {})

            sections.append(("TARGET", f"Sample: {sample_idx} | Layer: {layer} | Datapoints: {num_dp}"))

            if config:
                config_lines = []
                for k, v in config.items():
                    config_lines.append(f"{k}: {v}")
                sections.append(("CONFIG", "\n".join(config_lines)))

        elif event_type == "optimization.completed":
            sample_idx = payload.get("sample_idx", 0)
            layer = payload.get("layer", 0)
            success = payload.get("success", True)
            final_loss = payload.get("final_loss")
            iterations = payload.get("iterations", 0)
            duration = payload.get("duration_seconds", 0.0)
            dp_used = payload.get("datapoints_used", 0)

            status = "[$success]SUCCESS[/]" if success else "[$error]FAILED[/]"
            sections.append(("RESULT", f"Status: {status}\nSample: {sample_idx} | Layer: {layer}"))

            loss_str = f"{final_loss:.6f}" if final_loss is not None else "N/A"
            metrics = f"Final loss: {loss_str}\n"
            metrics += f"Iterations: {iterations}\n"
            metrics += f"Duration: {duration:.2f}s\n"
            metrics += f"Datapoints used: {dp_used}"
            sections.append(("METRICS", metrics))

            if not success:
                error = payload.get("error", "Unknown error")
                sections.append(("ERROR", f"[$error]{error}[/]"))

            # Loss history (if available, show trend)
            loss_history = payload.get("loss_history", [])
            # Filter out None values for trend calculation
            valid_losses = [loss for loss in loss_history if loss is not None]
            if valid_losses and len(valid_losses) > 1:
                start_loss = valid_losses[0]
                end_loss = valid_losses[-1]
                improvement = ((start_loss - end_loss) / start_loss * 100) if start_loss > 0 else 0
                trend = f"Start: {start_loss:.6f} -> End: {end_loss:.6f} ({improvement:.1f}% improvement)"
                sections.append(("LOSS TREND", trend))

        elif event_type == "optimization.aggregation_completed":
            strategy = payload.get("strategy", "unknown")
            num_vectors = payload.get("num_vectors", 0)
            top_k = payload.get("top_k", 0)
            final_score = payload.get("final_score", 0.0)
            final_layer = payload.get("final_layer", 0)

            sections.append(("AGGREGATION", f"Strategy: {strategy}\nVectors: {num_vectors} | Top-K: {top_k}"))
            sections.append(("RESULT", f"Final score: {final_score:.3f}\nBest layer: {final_layer}"))

        return sections


class EvaluationRenderer(LogDetailRenderer):
    """Renderer for evaluation events - shows scores and verdict."""

    def can_render(self, event_type: Optional[str]) -> bool:
        return event_type in (
            "evaluation.started",
            "evaluation.output",
            "evaluation.completed",
        )

    def render_sections(self, entry: LogEntry) -> List[Tuple[str, str]]:
        payload = entry.payload or {}
        sections = []

        event_type = entry.event_type

        if event_type == "evaluation.started":
            eval_id = payload.get("evaluation_id", "")[:12]
            eval_type = payload.get("eval_type", "quick")
            layer = payload.get("layer", 0)
            strengths = payload.get("strength_levels", [])
            num_prompts = payload.get("num_prompts", 0)

            sections.append(("CONFIG", f"ID: {eval_id}\nType: {eval_type}\nLayer: {layer}"))
            if strengths:
                sections.append(("STRENGTH LEVELS", ", ".join(f"{s:.2f}" for s in strengths)))
            sections.append(("PROMPTS", f"{num_prompts} evaluation prompts"))

        elif event_type == "evaluation.output":
            prompt = payload.get("prompt", "")
            output = payload.get("output", "")
            strength = payload.get("strength")
            is_baseline = payload.get("is_baseline", False)

            label = "BASELINE" if is_baseline else f"STRENGTH {strength:.2f}"
            sections.append(("CONDITION", label))
            if prompt:
                sections.append(("PROMPT", self._truncate(prompt)))
            if output:
                sections.append(("OUTPUT", self._truncate(output)))

        elif event_type == "evaluation.completed":
            eval_id = payload.get("evaluation_id", "")[:12]
            scores = payload.get("scores", {})
            verdict = payload.get("verdict", "unknown")
            rec_strength = payload.get("recommended_strength", 1.0)
            recommendations = payload.get("recommendations", [])

            # Scores table
            score_lines = [f"ID: {eval_id}", ""]
            for name, value in scores.items():
                score_lines.append(f"{name.title():15} {value:.3f}")
            sections.append(("SCORES", "\n".join(score_lines)))

            # Verdict
            verdict_color = {
                "accepted": "$success",
                "needs_refinement": "$warning",
                "rejected": "$error",
            }.get(verdict, "$foreground")
            sections.append(("VERDICT", f"[{verdict_color}]{verdict.upper()}[/]\nRecommended strength: {rec_strength:.2f}"))

            # Recommendations
            if recommendations:
                sections.append(("RECOMMENDATIONS", "\n".join(f"- {r}" for r in recommendations)))

            # Raw judge output (if available)
            raw_output = payload.get("raw_judge_output", "")
            if raw_output:
                sections.append(("RAW JUDGE OUTPUT", self._truncate(raw_output)))

        return sections


class ToolEventRenderer(LogDetailRenderer):
    """Renderer for tool.call and tool.result events."""

    def can_render(self, event_type: Optional[str]) -> bool:
        return event_type in ("tool.call", "tool.result")

    def render_sections(self, entry: LogEntry) -> List[Tuple[str, str]]:
        payload = entry.payload or {}
        sections = []

        if entry.event_type == "tool.call":
            tool_name = payload.get("tool_name", "unknown")
            call_id = payload.get("call_id", "")[:12]
            agent_id = payload.get("agent_id", "")
            arguments = payload.get("arguments", {})

            sections.append(("TOOL CALL", f"Tool: {tool_name}\nCall ID: {call_id}\nAgent: {agent_id}"))
            sections.append(("ARGUMENTS", self._format_json(arguments)))

        elif entry.event_type == "tool.result":
            call_id = payload.get("call_id", "")[:12]
            success = payload.get("success", True)
            latency = payload.get("latency_ms", 0)
            output = payload.get("output")
            error = payload.get("error")

            status = "[$success]SUCCESS[/]" if success else "[$error]FAILED[/]"
            sections.append(("RESULT", f"Status: {status}\nCall ID: {call_id}\nLatency: {latency}ms"))

            if error:
                sections.append(("ERROR", f"[$error]{error}[/]"))

            if output is not None:
                output_str = self._format_json(output) if isinstance(output, (dict, list)) else str(output)
                sections.append(("OUTPUT", self._truncate(output_str)))

        return sections


class VectorEventRenderer(LogDetailRenderer):
    """Renderer for vector events."""

    def can_render(self, event_type: Optional[str]) -> bool:
        return event_type in ("vector.created", "vector.selected", "vector.comparison")

    def render_sections(self, entry: LogEntry) -> List[Tuple[str, str]]:
        payload = entry.payload or {}
        sections = []

        if entry.event_type == "vector.created":
            vector_id = payload.get("vector_id", "")[:12]
            layer = payload.get("layer", 0)
            shape = payload.get("shape", [])
            dtype = payload.get("dtype", "unknown")
            norm = payload.get("norm", 0.0)
            vector_ref = payload.get("vector_ref", "")

            sections.append(("VECTOR", f"ID: {vector_id}\nLayer: {layer}\nShape: {shape}\nDtype: {dtype}\nNorm: {norm:.4f}"))

            if vector_ref:
                sections.append(("FILE", vector_ref))

            opt_metrics = payload.get("optimization_metrics", {})
            if opt_metrics:
                sections.append(("OPTIMIZATION METRICS", self._format_json(opt_metrics)))

        elif entry.event_type == "vector.selected":
            vector_id = payload.get("vector_id", "")[:12]
            layer = payload.get("layer", 0)
            strength = payload.get("strength", 1.0)
            score = payload.get("score")
            reason = payload.get("reason", "")

            sections.append(("SELECTION", f"Vector: {vector_id}\nLayer: {layer}\nStrength: {strength:.2f}"))
            if score is not None:
                sections.append(("SCORE", f"{score:.3f}"))
            if reason:
                sections.append(("REASON", reason))

        return sections


class SessionEventRenderer(LogDetailRenderer):
    """Renderer for session events."""

    def can_render(self, event_type: Optional[str]) -> bool:
        return event_type in ("session.started", "session.completed")

    def render_sections(self, entry: LogEntry) -> List[Tuple[str, str]]:
        payload = entry.payload or {}
        sections = []

        if entry.event_type == "session.started":
            behavior = payload.get("behavior_name", "unknown")
            description = payload.get("behavior_description", "")
            config = payload.get("config", {})

            sections.append(("BEHAVIOR", f"Name: {behavior}\n\n{description}"))

            if config:
                config_lines = []
                for k, v in sorted(config.items()):
                    config_lines.append(f"{k}: {v}")
                sections.append(("CONFIG", "\n".join(config_lines)))

        elif entry.event_type == "session.completed":
            success = payload.get("success", False)
            final_score = payload.get("final_score")
            final_layer = payload.get("final_layer")
            total_llm = payload.get("total_llm_calls", 0)
            total_tokens = payload.get("total_tokens", 0)
            duration = payload.get("duration_seconds", 0.0)
            error = payload.get("error")

            status = "[$success]SUCCESS[/]" if success else "[$error]FAILED[/]"
            sections.append(("STATUS", status))

            if success:
                result = f"Final score: {final_score:.3f}\n" if final_score else ""
                result += f"Best layer: {final_layer}" if final_layer else ""
                if result:
                    sections.append(("RESULT", result))

            metrics = f"LLM calls: {total_llm}\nTokens: {total_tokens}\nDuration: {duration:.1f}s"
            sections.append(("METRICS", metrics))

            if error:
                sections.append(("ERROR", f"[$error]{error}[/]"))

        return sections


class SeedEventRenderer(LogDetailRenderer):
    """Renderer for seed events."""

    def can_render(self, event_type: Optional[str]) -> bool:
        return event_type in (
            "seed.generation_started",
            "seed.generated",
            "seed.generation_completed",
            "seed.assigned",
        )

    def render_sections(self, entry: LogEntry) -> List[Tuple[str, str]]:
        payload = entry.payload or {}
        sections = []

        if entry.event_type == "seed.generated":
            seed_id = payload.get("seed_id", "")[:12]
            scenario = payload.get("scenario", "")
            context = payload.get("context", "")
            quality = payload.get("quality_score", 0.0)
            is_core = payload.get("is_core", False)

            core_label = "[$accent]CORE[/]" if is_core else "UNIQUE"
            sections.append(("SEED", f"ID: {seed_id}\nType: {core_label}\nQuality: {quality:.3f}"))
            sections.append(("SCENARIO", self._truncate(scenario)))
            sections.append(("CONTEXT", self._truncate(context)))

        elif entry.event_type == "seed.assigned":
            sample_idx = payload.get("sample_idx", 0)
            num_core = payload.get("num_core_seeds", 0)
            num_unique = payload.get("num_unique_seeds", 0)
            seed_ids = payload.get("seed_ids", [])

            sections.append(("ASSIGNMENT", f"Sample: {sample_idx}\nCore seeds: {num_core}\nUnique seeds: {num_unique}"))
            if seed_ids:
                sections.append(("SEED IDS", "\n".join(s[:12] for s in seed_ids[:20])))

        elif entry.event_type == "seed.generation_completed":
            total = payload.get("total_generated", 0)
            filtered = payload.get("total_filtered", 0)
            avg_quality = payload.get("avg_quality", 0.0)

            sections.append(("SUMMARY", f"Generated: {total}\nFiltered: {filtered}\nAverage quality: {avg_quality:.3f}"))

        return sections


class DefaultRenderer(LogDetailRenderer):
    """Fallback renderer for unknown event types - shows raw message and payload."""

    def can_render(self, event_type: Optional[str]) -> bool:
        return True  # Always can render as fallback

    def render_sections(self, entry: LogEntry) -> List[Tuple[str, str]]:
        sections = [("MESSAGE", entry.message)]

        if entry.payload:
            sections.append(("RAW PAYLOAD", self._format_json(entry.payload)))

        return sections


class LogDetailRendererRegistry:
    """Registry of log detail renderers - finds the right renderer for an event type."""

    def __init__(self) -> None:
        # Order matters - more specific renderers first, DefaultRenderer last
        self._renderers: List[LogDetailRenderer] = [
            LLMRequestRenderer(),
            LLMResponseRenderer(),
            ContrastPairRenderer(),
            DatapointRenderer(),
            OptimizationRenderer(),
            EvaluationRenderer(),
            ToolEventRenderer(),
            VectorEventRenderer(),
            SessionEventRenderer(),
            SeedEventRenderer(),
            DefaultRenderer(),  # Fallback - must be last
        ]

    def get_renderer(self, event_type: Optional[str]) -> LogDetailRenderer:
        """Get the appropriate renderer for an event type."""
        for renderer in self._renderers:
            if renderer.can_render(event_type):
                return renderer
        # Should never happen since DefaultRenderer always returns True
        return DefaultRenderer()

    def render(self, entry: LogEntry) -> List[Tuple[str, str]]:
        """Render a log entry using the appropriate renderer."""
        renderer = self.get_renderer(entry.event_type)
        return renderer.render_sections(entry)


# Global registry instance
_registry: Optional[LogDetailRendererRegistry] = None


def get_renderer_registry() -> LogDetailRendererRegistry:
    """Get the global renderer registry."""
    global _registry
    if _registry is None:
        _registry = LogDetailRendererRegistry()
    return _registry
