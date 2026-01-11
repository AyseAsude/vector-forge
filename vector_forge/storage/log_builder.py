"""Shared log entry builder for storage events.

Provides consistent log message generation for events, used by both
real-time synchronization and replay.
"""

from typing import Any, Dict, Optional, Tuple


def build_log_entry(
    event_type: str,
    payload: Dict[str, Any],
    source: str,
) -> Optional[Tuple[str, str, str]]:
    """Build a log entry tuple from an event.

    Args:
        event_type: The event type string.
        payload: Event payload dictionary.
        source: Event source identifier.

    Returns:
        Tuple of (source, message, level) or None to skip logging.
    """
    # Session events
    if event_type == "session.started":
        return ("session", f"Started extraction: {payload.get('behavior_name', 'unknown')}", "info")

    elif event_type == "session.completed":
        success = payload.get('success')
        score = payload.get('final_score')
        layer = payload.get('final_layer')
        if success:
            msg = "Completed successfully"
            if score is not None:
                msg += f" | score: {score:.3f}"
            if layer is not None:
                msg += f" | layer: {layer}"
            return ("session", msg, "info")
        else:
            error = payload.get('error', 'unknown error')
            return ("session", f"Failed: {error}", "error")

    # Contrast pipeline events
    elif event_type == "contrast.pipeline_started":
        return ("contrast", f"Starting contrast generation ({payload.get('num_samples', 0)} samples)", "info")

    elif event_type == "contrast.behavior_analyzed":
        components = payload.get('num_components', 0)
        dims = len(payload.get('contrast_dimensions', []))
        return ("contrast", f"Behavior analyzed: {components} components, {dims} contrast dimensions", "info")

    elif event_type == "contrast.pipeline_completed":
        total = payload.get('total_pairs_generated') or 0
        valid = payload.get('total_valid_pairs') or 0
        quality = payload.get('avg_quality') or 0.0
        duration = payload.get('duration_seconds') or 0.0
        return ("contrast", f"Contrast complete: {valid}/{total} valid pairs | quality: {quality:.2f} | {duration:.1f}s", "info")

    elif event_type == "contrast.pair_generated":
        sample = payload.get('sample_idx') or 0
        pair_id = (payload.get('pair_id') or '')[:8]
        return ("contrast", f"Sample {sample}: generated pair {pair_id}", "info")

    elif event_type == "contrast.pair_validated":
        pair_id = (payload.get('pair_id') or '')[:8]
        is_valid = payload.get('is_valid', False)
        if is_valid:
            quality = payload.get('contrast_quality') or 0.0
            dst = payload.get('dst_score') or 0.0
            src = payload.get('src_score') or 0.0
            return ("contrast", f"Pair {pair_id} valid: quality={quality:.2f} (dst={dst:.2f}, src={src:.2f})", "info")
        else:
            reason = payload.get('rejection_reason') or 'unknown'
            return ("contrast", f"Pair {pair_id} rejected: {reason}", "warning")

    # Seed events
    elif event_type == "seed.generation_started":
        num = payload.get('num_seeds_requested') or 0
        behavior = payload.get('behavior_name') or ''
        return ("seed", f"Generating {num} seeds for '{behavior}'", "info")

    elif event_type == "seed.generated":
        seed_id = (payload.get('seed_id') or '')[:8]
        quality = payload.get('quality_score') or 0.0
        is_core = "core" if payload.get('is_core') else "unique"
        scenario = (payload.get('scenario') or '')[:50]
        return ("seed", f"Seed {seed_id} ({is_core}): quality={quality:.2f} | {scenario}...", "info")

    elif event_type == "seed.generation_completed":
        total = payload.get('total_generated') or 0
        filtered = payload.get('total_filtered') or 0
        avg_quality = payload.get('avg_quality') or 0.0
        return ("seed", f"Seed generation complete: {total} generated, {filtered} filtered | avg quality: {avg_quality:.2f}", "info")

    elif event_type == "seed.assigned":
        sample = payload.get('sample_idx') or 0
        core = payload.get('num_core_seeds') or 0
        unique = payload.get('num_unique_seeds') or 0
        return ("seed", f"Sample {sample}: assigned {core} core + {unique} unique seeds", "info")

    # Datapoint events
    elif event_type == "datapoint.added":
        dp_id = (payload.get('datapoint_id') or '')[:12]
        domain = payload.get('domain') or 'unknown'
        return ("datapoint", f"Added {dp_id} (domain: {domain})", "info")

    elif event_type == "datapoint.removed":
        dp_id = (payload.get('datapoint_id') or '')[:12]
        reason = payload.get('reason') or 'no reason'
        return ("datapoint", f"Removed {dp_id}: {reason}", "warning")

    elif event_type == "datapoint.quality":
        dp_id = (payload.get('datapoint_id') or '')[:12]
        quality = payload.get('quality_score') or 0.0
        rec = payload.get('recommendation') or 'KEEP'
        return ("datapoint", f"Quality {dp_id}: {quality:.2f} → {rec}", "info")

    # Optimization events
    elif event_type == "optimization.started":
        sample = payload.get('sample_idx') or 0
        layer = payload.get('layer') or 0
        num_dp = payload.get('num_datapoints') or 0
        return ("optimizer", f"Sample {sample}: starting layer {layer} optimization ({num_dp} datapoints)", "info")

    elif event_type == "optimization.progress":
        sample = payload.get('sample_idx') or 0
        iteration = payload.get('iteration') or 0
        loss = payload.get('loss') or 0.0
        return ("optimizer", f"Sample {sample}: iter {iteration} | loss: {loss:.6f}", "info")

    elif event_type == "optimization.completed":
        sample = payload.get('sample_idx') or 0
        layer = payload.get('layer') or 0
        success = payload.get('success', True)
        if success:
            loss = payload.get('final_loss') or 0.0
            iters = payload.get('iterations') or 0
            duration = payload.get('duration_seconds') or 0.0
            return ("optimizer", f"Sample {sample} layer {layer}: loss={loss:.6f} | {iters} iters | {duration:.1f}s", "info")
        else:
            error = payload.get('error') or 'unknown'
            return ("optimizer", f"Sample {sample} layer {layer} FAILED: {error}", "error")

    elif event_type == "optimization.aggregation_completed":
        strategy = payload.get('strategy') or 'unknown'
        num = payload.get('num_vectors') or 0
        top_k = payload.get('top_k') or 0
        score = payload.get('final_score') or 0.0
        layer = payload.get('final_layer') or 0
        return ("optimizer", f"Aggregated ({strategy}): {num} vectors, top-{top_k} | score: {score:.3f} | layer: {layer}", "info")

    # Vector events
    elif event_type == "vector.created":
        layer = payload.get('layer') or 0
        norm = payload.get('norm') or 0.0
        vector_id = (payload.get('vector_id') or '')[:8]
        return ("vector", f"Created {vector_id} at layer {layer} | norm: {norm:.4f}", "info")

    elif event_type == "vector.comparison":
        ids = payload.get('vector_ids') or []
        return ("vector", f"Compared {len(ids)} vectors", "info")

    elif event_type == "vector.selected":
        layer = payload.get('layer') or 0
        strength = payload.get('strength') or 1.0
        score = payload.get('score')
        reason = payload.get('reason') or ''
        msg = f"Selected layer {layer} @ strength {strength:.2f}"
        if score is not None:
            msg += f" | score: {score:.3f}"
        if reason:
            msg += f" | {reason}"
        return ("vector", msg, "info")

    # Evaluation events
    elif event_type == "evaluation.started":
        eval_id = (payload.get('evaluation_id') or '')[:8]
        eval_type = payload.get('eval_type') or 'quick'
        layer = payload.get('layer') or 0
        strengths = payload.get('strength_levels') or []
        return ("evaluation", f"Started {eval_type} eval {eval_id}: layer {layer}, {len(strengths)} strengths", "info")

    elif event_type == "evaluation.output":
        eval_id = (payload.get('evaluation_id') or '')[:8]
        strength = payload.get('strength') or 1.0
        is_baseline = payload.get('is_baseline', False)
        label = "baseline" if is_baseline else f"strength={strength:.2f}"
        return ("evaluation", f"Eval {eval_id} output ({label})", "info")

    elif event_type == "evaluation.completed":
        eval_id = (payload.get('evaluation_id') or '')[:8]
        scores = payload.get('scores') or {}
        overall = scores.get('overall') or 0.0
        verdict = payload.get('verdict') or 'unknown'
        strength = payload.get('recommended_strength') or 1.0
        return ("evaluation", f"Eval {eval_id} complete: {overall:.3f} | {verdict} | rec. strength: {strength:.2f}", "info")

    # Checkpoint events
    elif event_type == "checkpoint.created":
        cp_id = (payload.get('checkpoint_id') or '')[:8]
        desc = payload.get('description') or ''
        return ("checkpoint", f"Created checkpoint {cp_id}: {desc}", "info")

    elif event_type == "checkpoint.rollback":
        cp_id = (payload.get('checkpoint_id') or '')[:8]
        prev_seq = payload.get('previous_sequence') or 0
        return ("checkpoint", f"Rolled back to {cp_id} (seq {prev_seq})", "warning")

    # State events
    elif event_type == "state.update":
        field = payload.get('field', 'unknown')
        old = payload.get('old_value')
        new = payload.get('new_value')
        return ("state", f"Updated {field}: {old} → {new}", "info")

    elif event_type == "state.iteration_started":
        iter_type = payload.get('iteration_type', 'unknown')
        iteration = payload.get('iteration', 0)
        max_iter = payload.get('max_iterations', 0)
        if max_iter:
            return ("state", f"{iter_type} iteration {iteration}/{max_iter} started", "info")
        return ("state", f"{iter_type} iteration {iteration} started", "info")

    elif event_type == "state.iteration_completed":
        iter_type = payload.get('iteration_type', 'unknown')
        iteration = payload.get('iteration', 0)
        metrics = payload.get('metrics', {})
        msg = f"{iter_type} iteration {iteration} completed"
        if metrics:
            msg += f" | {metrics}"
        return ("state", msg, "info")

    # LLM events
    elif event_type == "llm.request":
        model = payload.get('model', 'unknown')
        num_msgs = len(payload.get('messages', []))
        tools = payload.get('tools')
        tool_str = f" | {len(tools)} tools" if tools else ""
        return (source or "llm", f"Request: {model} ({num_msgs} messages{tool_str})", "info")

    elif event_type == "llm.response":
        latency = payload.get('latency_ms', 0)
        finish = payload.get('finish_reason', 'stop')
        error = payload.get('error')
        if error:
            return (source or "llm", f"Response ERROR: {error}", "error")
        tool_calls = payload.get('tool_calls', [])
        if tool_calls:
            return (source or "llm", f"Response: {len(tool_calls)} tool calls | {latency}ms", "info")
        return (source or "llm", f"Response: {finish} | {latency}ms", "info")

    elif event_type == "llm.chunk":
        # Skip chunk events - too noisy for logs
        return None

    # Tool events
    elif event_type == "tool.call":
        tool = payload.get('tool_name', 'unknown')
        call_id = payload.get('call_id', '')[:8]
        return ("tool", f"Calling {tool} ({call_id})", "info")

    elif event_type == "tool.result":
        call_id = payload.get('call_id', '')[:8]
        success = payload.get('success', True)
        latency = payload.get('duration_ms', 0)
        if success:
            return ("tool", f"Result {call_id}: success | {latency}ms", "info")
        else:
            error = payload.get('error', 'unknown')
            return ("tool", f"Result {call_id}: FAILED - {error}", "error")

    # Fallback for unknown events
    return (source or "unknown", f"Event: {event_type}", "info")
