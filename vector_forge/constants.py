"""Central constants for Vector Forge.

This module provides a single source of truth for configuration defaults
and other constants used throughout the codebase.
"""

# Default LLM model for all operations
DEFAULT_MODEL = "claude-opus-4-5"
DEFAULT_MODEL_NAME = "Claude Opus 4.5"

# Fallback models by provider
FALLBACK_MODELS = {
    "anthropic": "claude-opus-4-5",
    "openai": "gpt-4o",
    "openrouter": "openrouter/anthropic/claude-opus-4-5",
}

# Model display names
MODEL_DISPLAY_NAMES = {
    "claude-opus-4-5": "Claude Opus 4.5",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o Mini",
}
