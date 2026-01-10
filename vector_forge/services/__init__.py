"""Service layer for Vector Forge.

Provides the bridge between storage (event sourcing) and UI (reactive state).
This layer handles session lifecycle, event translation, and task execution.

Key Components:
- SessionService: Manages session lifecycle and task execution
- UIStateSynchronizer: Translates storage events to UI state updates
- TaskExecutor: Executes tasks with full event emission
"""

from vector_forge.services.session import (
    SessionService,
    SessionInfo,
    SessionSummary,
)
from vector_forge.services.synchronizer import UIStateSynchronizer
from vector_forge.services.task_executor import TaskExecutor

__all__ = [
    "SessionService",
    "SessionInfo",
    "SessionSummary",
    "UIStateSynchronizer",
    "TaskExecutor",
]
