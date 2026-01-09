"""Vector Forge Terminal User Interface.

A professional terminal UI for monitoring and controlling
steering vector extraction pipelines.

Usage:
    from vector_forge.ui import VectorForgeApp, run, run_demo

    # Run with default state
    run()

    # Run with demo data
    run_demo()

    # Run with custom state
    app = VectorForgeApp(state=my_state)
    app.run()
"""

from vector_forge.ui.app import VectorForgeApp, run, run_demo, create_demo_state
from vector_forge.ui.state import (
    UIState,
    ExtractionUIState,
    ExtractionStatus,
    Phase,
    DatapointMetrics,
    EvaluationMetrics,
    ActivityEntry,
    LogEntry,
    get_state,
    reset_state,
)

__all__ = [
    # App
    "VectorForgeApp",
    "run",
    "run_demo",
    "create_demo_state",
    # State
    "UIState",
    "ExtractionUIState",
    "ExtractionStatus",
    "Phase",
    "DatapointMetrics",
    "EvaluationMetrics",
    "ActivityEntry",
    "LogEntry",
    "get_state",
    "reset_state",
]
