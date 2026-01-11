"""Model configuration storage and management.

Stores LLM provider configurations for reuse across tasks.
Supports OpenAI, Anthropic, Azure, Ollama, and custom providers.

All model constants are centralized in vector_forge.constants for DRY compliance.
"""

import json
import os
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import uuid

from pydantic import BaseModel, Field
from typing import Literal

from vector_forge.constants import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_NAME,
    PROVIDER_ORDER,
    API_KEY_ENV_VARS,
    DEFAULT_API_BASES,
    COMMON_MODELS,
    BUILTIN_MODELS,
    BUILTIN_MODEL_IDS,
)


class Provider(str, Enum):
    """Supported LLM providers.

    Note: Order matches PROVIDER_ORDER in constants - Anthropic first as preferred.
    """

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    AZURE = "azure"
    OLLAMA = "ollama"
    CUSTOM = "custom"


def _get_api_key_env(provider: Provider) -> Optional[str]:
    """Get the environment variable name for a provider's API key."""
    return API_KEY_ENV_VARS.get(provider.value)


def _get_common_models(provider: Provider) -> List[str]:
    """Get common models for a provider from centralized constants."""
    return COMMON_MODELS.get(provider.value, [])


class MaxPrice(BaseModel):
    """Maximum price constraints for OpenRouter requests.

    All values are in USD per million tokens.
    """

    prompt: Optional[float] = Field(default=None, ge=0, description="Max USD per million prompt tokens")
    completion: Optional[float] = Field(default=None, ge=0, description="Max USD per million completion tokens")

    def to_dict(self) -> Dict[str, float]:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ProviderPreferences(BaseModel):
    """OpenRouter-specific provider routing preferences.

    Controls how OpenRouter selects and routes to underlying providers.
    See: https://openrouter.ai/docs/features/provider-routing

    Example:
        >>> prefs = ProviderPreferences(
        ...     order=["anthropic", "openai"],
        ...     sort="price",
        ...     max_price=MaxPrice(prompt=1.0, completion=2.0),
        ... )
    """

    order: Optional[List[str]] = Field(
        default=None,
        description="Preferred provider order (e.g., ['anthropic', 'openai'])",
    )
    allow_fallbacks: bool = Field(
        default=True,
        description="Allow backup providers if primary fails",
    )
    only: Optional[List[str]] = Field(
        default=None,
        description="Whitelist: only use these providers",
    )
    ignore: Optional[List[str]] = Field(
        default=None,
        description="Blacklist: skip these providers",
    )
    sort: Optional[Literal["price", "throughput", "latency"]] = Field(
        default=None,
        description="Sort strategy for provider selection",
    )
    data_collection: Literal["allow", "deny"] = Field(
        default="allow",
        description="Privacy: allow or deny data collection",
    )
    require_parameters: bool = Field(
        default=False,
        description="Only use providers supporting all request parameters",
    )
    max_price: Optional[MaxPrice] = Field(
        default=None,
        description="Maximum price constraints (USD per million tokens)",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for API request, excluding defaults/None values."""
        result: Dict[str, Any] = {}

        if self.order:
            result["order"] = self.order
        if not self.allow_fallbacks:
            result["allow_fallbacks"] = False
        if self.only:
            result["only"] = self.only
        if self.ignore:
            result["ignore"] = self.ignore
        if self.sort:
            result["sort"] = self.sort
        if self.data_collection != "allow":
            result["data_collection"] = self.data_collection
        if self.require_parameters:
            result["require_parameters"] = True
        if self.max_price:
            price_dict = self.max_price.to_dict()
            if price_dict:
                result["max_price"] = price_dict

        return result


class ModelConfig(BaseModel):
    """Configuration for a single LLM model.

    Stores all parameters needed to connect to an LLM via litellm.

    Example:
        >>> config = ModelConfig(
        ...     id="my-claude",
        ...     name="Claude Opus 4.5",
        ...     provider=Provider.ANTHROPIC,
        ...     model="claude-opus-4-5",
        ... )
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = Field(..., description="Display name for the model")
    provider: Provider = Field(default=Provider.ANTHROPIC)
    model: str = Field(..., description="LiteLLM model identifier")

    # Optional configuration
    api_base: Optional[str] = Field(default=None, description="Custom API endpoint")
    api_key: Optional[str] = Field(default=None, description="API key (or use env var)")
    api_key_env: Optional[str] = Field(default=None, description="Environment variable for API key")

    # Azure-specific
    api_version: Optional[str] = Field(default=None, description="Azure API version")

    # OpenRouter-specific
    provider_preferences: Optional[ProviderPreferences] = Field(
        default=None,
        description="OpenRouter provider routing preferences",
    )

    # Generation defaults
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, gt=0, description="Max tokens (None = provider default)")

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: Optional[datetime] = None
    is_default: bool = False
    is_builtin: bool = False  # True for default models, can't be deleted

    def get_api_key(self) -> Optional[str]:
        """Get API key from config or environment."""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        # Fallback to default env var for provider
        default_env = _get_api_key_env(self.provider)
        if default_env:
            return os.environ.get(default_env)
        return None

    def get_litellm_model(self) -> str:
        """Get the full model string with provider prefix for litellm.

        LiteLLM requires provider prefixes for certain providers:
        - openrouter/model-name
        - ollama/model-name
        - azure/deployment-name
        """
        # Providers that need prefix
        prefix_map = {
            Provider.OPENROUTER: "openrouter/",
            Provider.OLLAMA: "ollama/",
            Provider.AZURE: "azure/",
        }
        prefix = prefix_map.get(self.provider, "")

        # Don't double-prefix if already present
        if prefix and not self.model.startswith(prefix):
            return f"{prefix}{self.model}"
        return self.model

    def to_llm_config(self) -> Dict[str, Any]:
        """Convert to LLMConfig-compatible dict for litellm."""
        config: Dict[str, Any] = {
            "model": self.get_litellm_model(),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.api_base:
            config["api_base"] = self.api_base
        api_key = self.get_api_key()
        if api_key:
            config["api_key"] = api_key

        # Include OpenRouter provider preferences
        if self.provider == Provider.OPENROUTER and self.provider_preferences:
            provider_dict = self.provider_preferences.to_dict()
            if provider_dict:
                config["extra_params"] = {"provider": provider_dict}

        return config

    @classmethod
    def from_provider(
        cls,
        provider: Provider,
        model: str,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> "ModelConfig":
        """Create a config from provider and model name."""
        if name is None:
            # Generate a nice name
            provider_names = {
                Provider.ANTHROPIC: "Anthropic",
                Provider.OPENAI: "OpenAI",
                Provider.OPENROUTER: "OpenRouter",
                Provider.AZURE: "Azure",
                Provider.OLLAMA: "Ollama",
                Provider.CUSTOM: "Custom",
            }
            model_short = model.split("/")[-1].replace("-", " ").title()
            name = f"{provider_names.get(provider, 'Custom')} {model_short}"

        return cls(
            name=name,
            provider=provider,
            model=model,
            api_key_env=_get_api_key_env(provider),
            **kwargs,
        )


class ModelConfigStore(BaseModel):
    """Storage for all model configurations."""

    version: int = 1
    configs: List[ModelConfig] = Field(default_factory=list)

    def get(self, config_id: str) -> Optional[ModelConfig]:
        """Get config by ID."""
        for config in self.configs:
            if config.id == config_id:
                return config
        return None

    def get_default(self) -> Optional[ModelConfig]:
        """Get the default model config."""
        for config in self.configs:
            if config.is_default:
                return config
        # Return first if no default
        return self.configs[0] if self.configs else None

    def add(self, config: ModelConfig) -> None:
        """Add a new config."""
        # Remove any existing with same ID
        self.configs = [c for c in self.configs if c.id != config.id]
        self.configs.append(config)

    def remove(self, config_id: str) -> bool:
        """Remove a config by ID."""
        original_len = len(self.configs)
        self.configs = [c for c in self.configs if c.id != config_id]
        return len(self.configs) < original_len

    def set_default(self, config_id: str) -> None:
        """Set a config as default."""
        for config in self.configs:
            config.is_default = config.id == config_id

    def list_by_provider(self, provider: Provider) -> List[ModelConfig]:
        """Get all configs for a provider."""
        return [c for c in self.configs if c.provider == provider]


# ============================================================================
# HuggingFace Target Model Configuration
# ============================================================================


class HFModelConfig(BaseModel):
    """Configuration for a HuggingFace target model.

    Stores information about locally-loaded models used for steering vector extraction.
    Unlike API models, these require local GPU resources.

    Example:
        >>> config = HFModelConfig(
        ...     name="Llama 3.1 8B",
        ...     model_id="meta-llama/Llama-3.1-8B-Instruct",
        ... )
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = Field(..., description="Display name for the model")
    model_id: str = Field(..., description="HuggingFace Hub ID or local path")

    # Model metadata (populated after loading)
    num_layers: Optional[int] = Field(default=None, description="Number of transformer layers")
    hidden_dim: Optional[int] = Field(default=None, description="Hidden dimension size")
    vocab_size: Optional[int] = Field(default=None, description="Vocabulary size")
    model_type: Optional[str] = Field(default=None, description="Model architecture type")
    param_count: Optional[str] = Field(default=None, description="Approximate parameter count")

    # Loading configuration
    dtype: str = Field(default="bfloat16", description="Tensor dtype for loading")
    device_map: str = Field(default="auto", description="Device mapping strategy")
    trust_remote_code: bool = Field(default=False, description="Trust remote code from Hub")

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: Optional[datetime] = None
    is_downloaded: bool = Field(default=False, description="Whether model is cached locally")
    is_default: bool = False

    @property
    def is_local_path(self) -> bool:
        """Check if model_id is a local path."""
        return self.model_id.startswith(("./", "/", "~"))

    @property
    def short_id(self) -> str:
        """Get short display ID."""
        if "/" in self.model_id:
            return self.model_id.split("/")[-1]
        return self.model_id

    @property
    def display_size(self) -> str:
        """Get displayable size string."""
        if self.param_count:
            return self.param_count
        if self.hidden_dim and self.num_layers:
            # Rough estimate
            params = self.hidden_dim * self.hidden_dim * self.num_layers * 4
            if params > 1e9:
                return f"~{params / 1e9:.1f}B"
            return f"~{params / 1e6:.0f}M"
        return "Unknown"


class HFModelConfigStore(BaseModel):
    """Storage for HuggingFace model configurations."""

    version: int = 1
    configs: List[HFModelConfig] = Field(default_factory=list)
    recent_models: List[str] = Field(
        default_factory=list,
        description="Recently used model IDs (for quick access)",
    )

    def get(self, config_id: str) -> Optional[HFModelConfig]:
        """Get config by ID."""
        for config in self.configs:
            if config.id == config_id:
                return config
        return None

    def get_by_model_id(self, model_id: str) -> Optional[HFModelConfig]:
        """Get config by HuggingFace model ID."""
        for config in self.configs:
            if config.model_id == model_id:
                return config
        return None

    def get_default(self) -> Optional[HFModelConfig]:
        """Get the default model config."""
        for config in self.configs:
            if config.is_default:
                return config
        return self.configs[0] if self.configs else None

    def add(self, config: HFModelConfig) -> None:
        """Add a new config."""
        self.configs = [c for c in self.configs if c.id != config.id]
        self.configs.append(config)

    def remove(self, config_id: str) -> bool:
        """Remove a config by ID."""
        original_len = len(self.configs)
        self.configs = [c for c in self.configs if c.id != config_id]
        return len(self.configs) < original_len

    def add_recent(self, model_id: str, max_recent: int = 10) -> None:
        """Add a model ID to recent list."""
        if model_id in self.recent_models:
            self.recent_models.remove(model_id)
        self.recent_models.insert(0, model_id)
        self.recent_models = self.recent_models[:max_recent]


class HFModelConfigManager:
    """Manages HuggingFace model configuration storage.

    Stores configurations in ~/.vector-forge/hf_models.json

    Example:
        >>> manager = HFModelConfigManager()
        >>> manager.add(HFModelConfig(name="My Llama", model_id="meta-llama/Llama-3.1-8B"))
        >>> configs = manager.list_all()
    """

    def __init__(self, base_path: Optional[Path] = None) -> None:
        if base_path is None:
            base_path = Path.home() / ".vector-forge"

        self.base_path = Path(base_path)
        self.config_file = self.base_path / "hf_models.json"
        self._store: Optional[HFModelConfigStore] = None

    def _ensure_dir(self) -> None:
        """Ensure config directory exists."""
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _load(self) -> HFModelConfigStore:
        """Load store from disk."""
        if self._store is not None:
            return self._store

        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._store = HFModelConfigStore.model_validate(data)
            except (json.JSONDecodeError, Exception):
                self._store = HFModelConfigStore()
        else:
            self._store = HFModelConfigStore()

        return self._store

    def _save(self) -> None:
        """Save store to disk."""
        self._ensure_dir()
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(
                self._store.model_dump(mode="json"),
                f,
                indent=2,
                default=str,
            )

    def reload(self) -> None:
        """Clear cache and reload from disk."""
        self._store = None

    def list_all(self) -> List[HFModelConfig]:
        """List all saved model configs."""
        return self._load().configs.copy()

    def list_recent(self) -> List[str]:
        """Get list of recently used model IDs."""
        return self._load().recent_models.copy()

    def get(self, config_id: str) -> Optional[HFModelConfig]:
        """Get a config by ID."""
        return self._load().get(config_id)

    def get_by_model_id(self, model_id: str) -> Optional[HFModelConfig]:
        """Get a config by HuggingFace model ID."""
        return self._load().get_by_model_id(model_id)

    def get_default(self) -> Optional[HFModelConfig]:
        """Get the default model config."""
        return self._load().get_default()

    def add(self, config: HFModelConfig) -> None:
        """Add or update a model config."""
        store = self._load()
        store.add(config)
        self._save()

    def remove(self, config_id: str) -> bool:
        """Remove a config."""
        store = self._load()
        removed = store.remove(config_id)
        if removed:
            self._save()
        return removed

    def update_last_used(self, config_id: str) -> None:
        """Update the last_used timestamp and add to recent."""
        store = self._load()
        config = store.get(config_id)
        if config:
            config.last_used = datetime.now(timezone.utc)
            store.add_recent(config.model_id)
            self._save()

    def add_recent(self, model_id: str) -> None:
        """Add a model ID to recent list."""
        store = self._load()
        store.add_recent(model_id)
        self._save()

    def set_default(self, config_id: str) -> None:
        """Set a config as the default."""
        store = self._load()
        for config in store.configs:
            config.is_default = config.id == config_id
        self._save()


class ModelConfigManager:
    """Manages model configuration storage and retrieval.

    Stores configurations in ~/.vector-forge/models.json

    Example:
        >>> manager = ModelConfigManager()
        >>> manager.add(ModelConfig(name="My Claude", provider=Provider.ANTHROPIC, model="claude-opus-4-5"))
        >>> configs = manager.list_all()
    """

    def __init__(self, base_path: Optional[Path] = None) -> None:
        """Initialize the manager.

        Args:
            base_path: Base path for config storage. Defaults to ~/.vector-forge
        """
        if base_path is None:
            base_path = Path.home() / ".vector-forge"

        self.base_path = Path(base_path)
        self.config_file = self.base_path / "models.json"
        self._store: Optional[ModelConfigStore] = None

    def _ensure_dir(self) -> None:
        """Ensure config directory exists."""
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _load(self) -> ModelConfigStore:
        """Load store from disk."""
        if self._store is not None:
            return self._store

        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._store = ModelConfigStore.model_validate(data)
                # Migrations
                self._migrate_builtin_flags()
                self._migrate_strip_provider_prefixes()
            except (json.JSONDecodeError, Exception):
                # Start fresh on error
                self._store = self._create_defaults()
        else:
            self._store = self._create_defaults()

        return self._store

    def _migrate_builtin_flags(self) -> None:
        """Ensure default models have is_builtin=True."""
        changed = False
        for config in self._store.configs:
            if config.id in BUILTIN_MODEL_IDS and not config.is_builtin:
                config.is_builtin = True
                changed = True
        if changed:
            self._save()

    def _migrate_strip_provider_prefixes(self) -> None:
        """Strip redundant provider prefixes from model names.

        Provider prefix is now added automatically by get_litellm_model(),
        so we strip any existing prefixes from saved configs.
        """
        prefix_map = {
            Provider.OPENROUTER: "openrouter/",
            Provider.OLLAMA: "ollama/",
            Provider.AZURE: "azure/",
        }
        changed = False
        for config in self._store.configs:
            prefix = prefix_map.get(config.provider)
            if prefix and config.model.startswith(prefix):
                config.model = config.model[len(prefix):]
                changed = True
        if changed:
            self._save()

    def _save(self) -> None:
        """Save store to disk."""
        self._ensure_dir()
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(
                self._store.model_dump(mode="json"),
                f,
                indent=2,
                default=str,
            )

    def _create_defaults(self) -> ModelConfigStore:
        """Create default model configurations from centralized BUILTIN_MODELS."""
        store = ModelConfigStore()

        for model_def in BUILTIN_MODELS:
            store.add(ModelConfig(
                id=model_def["id"],
                name=model_def["name"],
                provider=Provider(model_def["provider"]),
                model=model_def["model"],
                api_base=model_def.get("api_base"),
                is_default=model_def.get("is_default", False),
                is_builtin=True,
            ))

        return store

    def reload(self) -> None:
        """Clear cache and reload from disk."""
        self._store = None

    def list_all(self) -> List[ModelConfig]:
        """List all saved model configs."""
        return self._load().configs.copy()

    def list_by_provider(self, provider: Provider) -> List[ModelConfig]:
        """List configs for a specific provider."""
        return self._load().list_by_provider(provider)

    def get(self, config_id: str) -> Optional[ModelConfig]:
        """Get a config by ID."""
        return self._load().get(config_id)

    def get_default(self) -> Optional[ModelConfig]:
        """Get the default model config."""
        return self._load().get_default()

    def add(self, config: ModelConfig) -> None:
        """Add or update a model config."""
        store = self._load()
        store.add(config)
        self._save()

    def remove(self, config_id: str) -> bool:
        """Remove a config."""
        store = self._load()
        removed = store.remove(config_id)
        if removed:
            self._save()
        return removed

    def set_default(self, config_id: str) -> None:
        """Set a config as the default."""
        store = self._load()
        store.set_default(config_id)
        self._save()

    def update_last_used(self, config_id: str) -> None:
        """Update the last_used timestamp for a config."""
        store = self._load()
        config = store.get(config_id)
        if config:
            config.last_used = datetime.now(timezone.utc)
            self._save()

    def get_common_models(self, provider: Provider) -> List[str]:
        """Get list of common models for a provider."""
        return _get_common_models(provider)

    def create_from_common(
        self,
        provider: Provider,
        model: str,
        name: Optional[str] = None,
    ) -> ModelConfig:
        """Create and save a config from common model presets."""
        config = ModelConfig.from_provider(provider, model, name)
        self.add(config)
        return config
