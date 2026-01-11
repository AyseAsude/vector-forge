"""User preferences storage and management.

Stores user preferences like last selected models for each role.
Follows the same pattern as models.py for consistency.

Preferences are stored in ~/.vector-forge/preferences.json
"""

import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


# ============================================================================
# Model Role Enum
# ============================================================================


class ModelRole:
    """Constants for model roles.

    Using class with string constants instead of Enum for JSON serialization simplicity.
    """

    GENERATOR = "generator"
    JUDGE = "judge"
    EXPANDER = "expander"
    TARGET = "target"

    ALL = [GENERATOR, JUDGE, EXPANDER, TARGET]


# ============================================================================
# Preferences Models
# ============================================================================


class SelectedModels(BaseModel):
    """Tracks last selected model config ID for each role.

    Stores the config ID (not model name) so we can look up the full
    ModelConfig or HFModelConfig when loading.
    """

    generator: Optional[str] = Field(
        default=None,
        description="ModelConfig ID for generator model",
    )
    judge: Optional[str] = Field(
        default=None,
        description="ModelConfig ID for judge model",
    )
    expander: Optional[str] = Field(
        default=None,
        description="ModelConfig ID for expander model",
    )
    target: Optional[str] = Field(
        default=None,
        description="HFModelConfig ID for target model",
    )

    def get(self, role: str) -> Optional[str]:
        """Get the selected model ID for a role."""
        return getattr(self, role, None)

    def set(self, role: str, config_id: Optional[str]) -> None:
        """Set the selected model ID for a role."""
        if role in ModelRole.ALL:
            setattr(self, role, config_id)


class Preferences(BaseModel):
    """User preferences for the application.

    Designed to be extensible - add new fields as needed.
    Old preference files without new fields will use defaults.
    """

    # Model selections
    selected_models: SelectedModels = Field(
        default_factory=SelectedModels,
        description="Last selected model for each role",
    )

    # Task defaults
    default_profile: str = Field(
        default="standard",
        description="Default task profile (quick/standard/comprehensive)",
    )

    default_num_samples: int = Field(
        default=4,
        description="Default number of samples for extraction",
    )

    # UI preferences
    show_advanced_options: bool = Field(
        default=False,
        description="Whether to show advanced options by default",
    )

    # Metadata
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


class PreferencesStore(BaseModel):
    """Storage wrapper with version for migrations."""

    version: int = 1
    preferences: Preferences = Field(default_factory=Preferences)


# ============================================================================
# Preferences Manager
# ============================================================================


class PreferencesManager:
    """Manages user preferences storage.

    Stores preferences in ~/.vector-forge/preferences.json

    Example:
        >>> manager = PreferencesManager()
        >>> manager.set_selected_model(ModelRole.GENERATOR, "abc123")
        >>> config_id = manager.get_selected_model(ModelRole.GENERATOR)

    Thread Safety:
        The manager caches the loaded preferences in memory.
        Use reload() to force re-reading from disk if needed.
    """

    def __init__(self, base_path: Optional[Path] = None) -> None:
        """Initialize the preferences manager.

        Args:
            base_path: Base directory for storage. Defaults to ~/.vector-forge
        """
        if base_path is None:
            base_path = Path.home() / ".vector-forge"

        self.base_path = Path(base_path)
        self.config_file = self.base_path / "preferences.json"
        self._store: Optional[PreferencesStore] = None

    def _ensure_dir(self) -> None:
        """Ensure config directory exists."""
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _load(self) -> PreferencesStore:
        """Load preferences from disk.

        Returns cached version if already loaded.
        """
        if self._store is not None:
            return self._store

        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._store = PreferencesStore.model_validate(data)
                self._store = self._migrate(self._store)
                logger.debug(f"Loaded preferences from {self.config_file}")
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Failed to load preferences: {e}, using defaults")
                self._store = PreferencesStore()
        else:
            self._store = PreferencesStore()

        return self._store

    def _migrate(self, store: PreferencesStore) -> PreferencesStore:
        """Apply any needed migrations to the store.

        Add migration logic here when schema changes.
        """
        # Version 1 is current, no migrations needed yet
        # Example for future migrations:
        # if store.version < 2:
        #     # Migrate from v1 to v2
        #     store.version = 2
        return store

    def _save(self) -> None:
        """Save preferences to disk."""
        if self._store is None:
            return

        self._ensure_dir()
        self._store.preferences.last_updated = datetime.now(timezone.utc)

        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(
                self._store.model_dump(mode="json"),
                f,
                indent=2,
                default=str,
            )
        logger.debug(f"Saved preferences to {self.config_file}")

    def reload(self) -> None:
        """Clear cache and reload from disk."""
        self._store = None

    # ========================================================================
    # Model Selection API
    # ========================================================================

    def get_selected_model(self, role: str) -> Optional[str]:
        """Get the last selected model config ID for a role.

        Args:
            role: One of ModelRole constants (generator, judge, expander, target)

        Returns:
            The config ID or None if not set.
        """
        return self._load().preferences.selected_models.get(role)

    def set_selected_model(self, role: str, config_id: Optional[str]) -> None:
        """Set the selected model config ID for a role.

        Args:
            role: One of ModelRole constants
            config_id: The ModelConfig or HFModelConfig ID to save
        """
        store = self._load()
        store.preferences.selected_models.set(role, config_id)
        self._save()

    def get_all_selected_models(self) -> SelectedModels:
        """Get all selected model IDs."""
        return self._load().preferences.selected_models

    def set_all_selected_models(
        self,
        generator: Optional[str] = None,
        judge: Optional[str] = None,
        expander: Optional[str] = None,
        target: Optional[str] = None,
    ) -> None:
        """Set multiple model selections at once.

        Only updates fields that are not None.
        """
        store = self._load()
        models = store.preferences.selected_models

        if generator is not None:
            models.generator = generator
        if judge is not None:
            models.judge = judge
        if expander is not None:
            models.expander = expander
        if target is not None:
            models.target = target

        self._save()

    # ========================================================================
    # Task Defaults API
    # ========================================================================

    @property
    def default_profile(self) -> str:
        """Get the default task profile."""
        return self._load().preferences.default_profile

    @default_profile.setter
    def default_profile(self, value: str) -> None:
        """Set the default task profile."""
        store = self._load()
        store.preferences.default_profile = value
        self._save()

    @property
    def default_num_samples(self) -> int:
        """Get the default number of samples."""
        return self._load().preferences.default_num_samples

    @default_num_samples.setter
    def default_num_samples(self, value: int) -> None:
        """Set the default number of samples."""
        store = self._load()
        store.preferences.default_num_samples = value
        self._save()

    # ========================================================================
    # UI Preferences API
    # ========================================================================

    @property
    def show_advanced_options(self) -> bool:
        """Get whether to show advanced options by default."""
        return self._load().preferences.show_advanced_options

    @show_advanced_options.setter
    def show_advanced_options(self, value: bool) -> None:
        """Set whether to show advanced options by default."""
        store = self._load()
        store.preferences.show_advanced_options = value
        self._save()

    # ========================================================================
    # Full Preferences Access
    # ========================================================================

    def get_preferences(self) -> Preferences:
        """Get the full preferences object."""
        return self._load().preferences

    def update_preferences(self, **kwargs) -> None:
        """Update multiple preference fields at once.

        Args:
            **kwargs: Field names and values to update.
        """
        store = self._load()
        for key, value in kwargs.items():
            if hasattr(store.preferences, key):
                setattr(store.preferences, key, value)
        self._save()
