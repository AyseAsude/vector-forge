"""Central constants for Vector Forge.

This module provides a single source of truth for configuration defaults
and other constants used throughout the codebase.

All model-related configuration is centralized here for DRY compliance.
"""

from typing import Dict, List, Any

# =============================================================================
# Default Model Configuration
# =============================================================================

DEFAULT_MODEL = "claude-opus-4-5"
DEFAULT_MODEL_NAME = "Claude Opus 4.5"

# =============================================================================
# Provider Configuration
# =============================================================================

# Provider order (Anthropic first as preferred)
PROVIDER_ORDER = ["anthropic", "openai", "openrouter", "azure", "ollama", "custom"]

# Environment variable names for API keys by provider
API_KEY_ENV_VARS: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "azure": "AZURE_API_KEY",
}

# Default API base URLs by provider
DEFAULT_API_BASES: Dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434",
}

# =============================================================================
# Common Models by Provider (for quick selection in UI)
# =============================================================================

COMMON_MODELS: Dict[str, List[str]] = {
    "anthropic": [
        "claude-opus-4-5",
        "claude-sonnet-4-5",
    ],
    "openai": [
        "gpt-5.2",
        "gpt-5.2-pro",
        "o3",
    ],
    "openrouter": [
        "openrouter/anthropic/claude-opus-4-5",
        "openrouter/anthropic/claude-sonnet-4-5",
        "openrouter/openai/gpt-5.2",
        "openrouter/google/gemini-2.0-flash",
        "openrouter/deepseek/deepseek-r1",
    ],
    "azure": [
        "azure/gpt-5.2",
        "azure/gpt-5.2-pro",
    ],
    "ollama": [
        "ollama/llama3.3",
        "ollama/qwen2.5",
        "ollama/deepseek-r1",
    ],
}

# =============================================================================
# Model Display Names
# =============================================================================

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "claude-opus-4-5": "Claude Opus 4.5",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "gpt-5.2": "GPT-5.2",
    "gpt-5.2-pro": "GPT-5.2 Pro",
    "o3": "O3",
}

# =============================================================================
# Builtin Model Configurations
# These are the default models that ship with Vector Forge
# =============================================================================

BUILTIN_MODELS: List[Dict[str, Any]] = [
    {
        "id": "anthropic-opus",
        "name": "Claude Opus 4.5",
        "provider": "anthropic",
        "model": "claude-opus-4-5",
        "is_default": True,
    },
    {
        "id": "anthropic-sonnet",
        "name": "Claude Sonnet 4.5",
        "provider": "anthropic",
        "model": "claude-sonnet-4-5",
    },
    {
        "id": "openai-gpt52",
        "name": "GPT-5.2",
        "provider": "openai",
        "model": "gpt-5.2",
    },
    {
        "id": "openai-gpt52-pro",
        "name": "GPT-5.2 Pro",
        "provider": "openai",
        "model": "gpt-5.2-pro",
    },
    {
        "id": "openrouter-claude",
        "name": "Claude Opus 4.5 (OpenRouter)",
        "provider": "openrouter",
        "model": "openrouter/anthropic/claude-opus-4-5",
        "api_base": "https://openrouter.ai/api/v1",
    },
]

# Set of builtin model IDs (for checking if a model can be deleted)
BUILTIN_MODEL_IDS = {m["id"] for m in BUILTIN_MODELS}

# =============================================================================
# Fallback Models by Provider
# =============================================================================

FALLBACK_MODELS: Dict[str, str] = {
    "anthropic": "claude-opus-4-5",
    "openai": "gpt-5.2",
    "openrouter": "openrouter/anthropic/claude-opus-4-5",
}
