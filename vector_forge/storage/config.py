"""Configuration for event storage system."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class StorageConfig(BaseModel):
    """Configuration for event sourcing storage.

    Controls where and how extraction sessions are stored.

    Example:
        >>> config = StorageConfig(base_path=Path("/custom/path"))
        >>> config = StorageConfig.from_env()  # Load from environment
    """

    base_path: Path = Field(
        default_factory=lambda: Path.home() / ".vector-forge" / "sessions",
        description="Base directory for session storage",
    )

    capture_raw_responses: bool = Field(
        default=True,
        description="Store full raw API responses (larger files but complete data)",
    )

    capture_all_vectors: bool = Field(
        default=True,
        description="Store all intermediate vectors (not just final)",
    )

    capture_evaluation_outputs: bool = Field(
        default=True,
        description="Store individual evaluation prompt/output pairs",
    )

    max_message_content_length: Optional[int] = Field(
        default=None,
        description="Truncate message content beyond this length (None = no limit)",
    )

    @classmethod
    def from_env(cls) -> "StorageConfig":
        """Load configuration from environment variables.

        Environment variables:
        - VECTOR_FORGE_STORAGE_PATH: Base path for sessions
        - VECTOR_FORGE_CAPTURE_RAW: Whether to capture raw responses ("true"/"false")
        - VECTOR_FORGE_CAPTURE_VECTORS: Whether to capture all vectors ("true"/"false")

        Returns:
            StorageConfig with values from environment.
        """
        base_path_str = os.getenv("VECTOR_FORGE_STORAGE_PATH")
        base_path = Path(base_path_str) if base_path_str else None

        capture_raw = os.getenv("VECTOR_FORGE_CAPTURE_RAW", "true").lower() == "true"
        capture_vectors = os.getenv("VECTOR_FORGE_CAPTURE_VECTORS", "true").lower() == "true"
        capture_eval = os.getenv("VECTOR_FORGE_CAPTURE_EVAL", "true").lower() == "true"

        kwargs: Dict[str, Any] = {
            "capture_raw_responses": capture_raw,
            "capture_all_vectors": capture_vectors,
            "capture_evaluation_outputs": capture_eval,
        }

        if base_path:
            kwargs["base_path"] = base_path

        return cls(**kwargs)

    @classmethod
    def minimal(cls) -> "StorageConfig":
        """Minimal storage configuration.

        Only stores essential events, not raw responses or all vectors.
        Useful for testing or limited storage environments.

        Returns:
            StorageConfig with minimal capture settings.
        """
        return cls(
            capture_raw_responses=False,
            capture_all_vectors=False,
            capture_evaluation_outputs=False,
        )

    @classmethod
    def full(cls) -> "StorageConfig":
        """Full storage configuration.

        Captures everything for complete reproducibility.

        Returns:
            StorageConfig with all capture enabled.
        """
        return cls(
            capture_raw_responses=True,
            capture_all_vectors=True,
            capture_evaluation_outputs=True,
        )
