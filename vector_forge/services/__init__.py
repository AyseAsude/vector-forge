"""Service layer for Vector Forge.

Provides the bridge between storage (event sourcing) and UI (reactive state).
This layer handles session lifecycle, event translation, and task execution.

Key Components:
- SessionService: Manages session lifecycle and task execution
- SessionLoader: Loads existing sessions into UI state on startup
- TaskExecutor: Executes tasks with full event emission
- ExtractionRunner: Orchestrates the full extraction pipeline

Real-time UI updates are handled by JSONL streaming (ui/watcher.py),
not by callbacks - following Textual's native patterns.
"""

from vector_forge.services.session import (
    SessionService,
    SessionInfo,
    SessionSummary,
)
from vector_forge.services.synchronizer import SessionLoader, UIStateSynchronizer
from vector_forge.services.task_executor import TaskExecutor
from vector_forge.services.extraction_runner import ExtractionRunner, ExtractionProgress

__all__ = [
    "SessionService",
    "SessionInfo",
    "SessionSummary",
    "SessionLoader",
    "UIStateSynchronizer",  # Backwards compatibility alias
    "TaskExecutor",
    "ExtractionRunner",
    "ExtractionProgress",
]
