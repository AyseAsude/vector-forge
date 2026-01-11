"""Shared utilities for the contrast module.

This module contains common utilities used across the contrast pipeline
to avoid code duplication.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def parse_llm_json(content: str) -> Dict[str, Any]:
    """Parse JSON from LLM response.

    Handles various response formats:
    1. Direct JSON (expected with response_format)
    2. JSON in markdown code blocks
    3. JSON embedded in text

    Args:
        content: Raw LLM response content.

    Returns:
        Parsed JSON as dictionary.

    Raises:
        json.JSONDecodeError: If no valid JSON found.
    """
    content = content.strip()

    # Primary: Direct JSON parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Fallback: Extract from markdown code block
    if "```" in content:
        parts = content.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Inside code blocks
                part = part.strip()
                if part.startswith(("json", "JSON")):
                    part = part[4:].strip()
                try:
                    return json.loads(part)
                except json.JSONDecodeError:
                    continue

    # Fallback: Find JSON object boundaries
    start = content.find("{")
    end = content.rfind("}") + 1
    if start >= 0 and end > start:
        return json.loads(content[start:end])

    raise json.JSONDecodeError("No valid JSON found", content, 0)


def safe_parse_llm_json(content: str, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Parse JSON from LLM response, returning default on failure.

    Args:
        content: Raw LLM response content.
        default: Default value if parsing fails.

    Returns:
        Parsed JSON or default.
    """
    try:
        return parse_llm_json(content)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM JSON response")
        return default if default is not None else {}
