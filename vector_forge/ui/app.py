"""Main Vector Forge TUI application."""

from pathlib import Path
from typing import Optional

from textual.app import App
from textual.binding import Binding

from vector_forge.ui.theme import forge_dark
from vector_forge.ui.state import (
    UIState,
    ExtractionUIState,
    ExtractionStatus,
    Phase,
    DatapointMetrics,
    EvaluationMetrics,
    get_state,
    reset_state,
)
from vector_forge.ui.screens.dashboard import DashboardScreen
from vector_forge.ui.screens.parallel import ParallelScreen
from vector_forge.ui.screens.logs import LogsScreen
from vector_forge.ui.screens.help import HelpModal


class VectorForgeApp(App):
    """Vector Forge Terminal User Interface.

    A professional terminal UI for monitoring and controlling
    steering vector extraction pipelines.

    Supports:
    - Single extraction monitoring (dashboard)
    - Multiple parallel extractions (parallel view)
    - Full log viewing with filtering (logs view)
    """

    TITLE = "Vector Forge"
    CSS_PATH = Path(__file__).parent / "forge.tcss"

    SCREENS = {
        "dashboard": DashboardScreen,
        "parallel": ParallelScreen,
        "logs": LogsScreen,
        "help": HelpModal,
    }

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
    ]

    def __init__(self, state: Optional[UIState] = None, **kwargs) -> None:
        """Initialize the application.

        Args:
            state: Optional pre-configured UI state. If not provided,
                   uses the global state instance.
        """
        super().__init__(**kwargs)
        self._state = state

    def on_mount(self) -> None:
        """Set up the application on mount."""
        # Register custom theme
        self.register_theme(forge_dark)
        self.theme = "forge-dark"

        # Initialize state if needed
        if self._state is not None:
            # Replace global state
            import vector_forge.ui.state as state_module
            state_module._state = self._state

        # Push initial screen
        self.push_screen("dashboard")

    def switch_screen(self, screen_name: str) -> None:
        """Switch to a different screen.

        Args:
            screen_name: Name of screen to switch to.
        """
        if screen_name == "help":
            self.push_screen("help")
        elif screen_name in ("dashboard", "parallel", "logs"):
            # Get current screen name to avoid unnecessary switches
            current = getattr(self.screen, "name", None) or type(self.screen).__name__.lower()
            if screen_name == current:
                return
            # Pop current and push new
            if len(self.screen_stack) > 1:
                self.pop_screen()
            self.push_screen(screen_name)


def create_demo_state() -> UIState:
    """Create a demo state for testing the UI.

    Returns:
        UIState populated with sample data.
    """
    import time

    state = reset_state()

    # Create a sample extraction
    extraction = ExtractionUIState(
        id="ext_001",
        behavior_name="sycophancy",
        behavior_description="Agreeing with the user even when they are factually wrong",
        model="claude-3.5-sonnet",
        status=ExtractionStatus.RUNNING,
        phase=Phase.OPTIMIZING,
        progress=67.0,
        outer_iteration=2,
        max_outer_iterations=3,
        inner_turn=15,
        max_inner_turns=50,
        current_layer=16,
        started_at=time.time() - 154,  # 2:34 ago
        datapoints=DatapointMetrics(
            total=15,
            keep=12,
            review=2,
            remove=1,
            diversity=0.68,
            clusters=4,
        ),
        evaluation=EvaluationMetrics(
            behavior=0.82,
            coherence=0.75,
            specificity=0.80,
            overall=0.79,
            best_layer=16,
            best_strength=1.2,
            verdict="needs_refinement",
        ),
    )

    # Add some activity
    extraction.add_activity("optimize_multi_layer layers=[14,15,16,17]", "success")
    extraction.add_activity("Layer 16: loss=0.023 norm=1.24", "success")
    extraction.add_activity("quick_eval layer=16 strength=1.0", "active")
    extraction.add_activity("Evaluating...", "waiting")

    state.add_extraction(extraction)

    # Add more extractions for parallel view demo
    extraction2 = ExtractionUIState(
        id="ext_002",
        behavior_name="honesty",
        behavior_description="Being truthful and accurate in responses",
        model="gpt-4o",
        status=ExtractionStatus.RUNNING,
        phase=Phase.GENERATING_DATAPOINTS,
        progress=45.0,
        outer_iteration=1,
        max_outer_iterations=3,
        inner_turn=8,
        max_inner_turns=50,
        started_at=time.time() - 118,
        datapoints=DatapointMetrics(total=8, keep=7, review=1, remove=0, diversity=0.55, clusters=3),
        evaluation=EvaluationMetrics(),
    )
    state.add_extraction(extraction2)

    extraction3 = ExtractionUIState(
        id="ext_003",
        behavior_name="helpfulness",
        behavior_description="Providing useful and relevant assistance",
        model="claude-3.5-sonnet",
        status=ExtractionStatus.RUNNING,
        phase=Phase.EVALUATING,
        progress=32.0,
        outer_iteration=1,
        max_outer_iterations=3,
        inner_turn=5,
        max_inner_turns=50,
        current_layer=12,
        started_at=time.time() - 72,
        datapoints=DatapointMetrics(total=5, keep=4, review=1, remove=0, diversity=0.42, clusters=2),
        evaluation=EvaluationMetrics(behavior=0.65, coherence=0.70, specificity=0.68, overall=0.67),
    )
    state.add_extraction(extraction3)

    extraction4 = ExtractionUIState(
        id="ext_004",
        behavior_name="curiosity",
        behavior_description="Showing intellectual curiosity and asking follow-up questions",
        model="claude-3.5-sonnet",
        status=ExtractionStatus.COMPLETE,
        phase=Phase.COMPLETE,
        progress=100.0,
        outer_iteration=3,
        max_outer_iterations=3,
        inner_turn=42,
        max_inner_turns=50,
        started_at=time.time() - 255,
        completed_at=time.time(),
        datapoints=DatapointMetrics(total=18, keep=16, review=2, remove=0, diversity=0.78, clusters=5),
        evaluation=EvaluationMetrics(
            behavior=0.88,
            coherence=0.82,
            specificity=0.85,
            overall=0.85,
            best_layer=15,
            best_strength=1.3,
            verdict="accepted",
        ),
    )
    state.add_extraction(extraction4)

    # Add some log entries
    state.add_log("pipeline", "Pipeline started", extraction_id="ext_001")
    state.add_log("pipeline", "Outer iteration 1 started", extraction_id="ext_001")
    state.add_log("extractor", "Generated 10 prompts", extraction_id="ext_001")
    state.add_log("extractor", "Generating completions...", extraction_id="ext_001")
    state.add_log("extractor", "Optimizing vector at layer 14", extraction_id="ext_001")
    state.add_log("extractor", "Layer 14: loss=0.045 norm=1.12", extraction_id="ext_001")
    state.add_log("extractor", "Optimizing vector at layer 15", extraction_id="ext_001")
    state.add_log("extractor", "Layer 15: loss=0.032 norm=1.18", extraction_id="ext_001")
    state.add_log("extractor", "Optimizing vector at layer 16", extraction_id="ext_001")
    state.add_log("extractor", "Layer 16: loss=0.023 norm=1.24", extraction_id="ext_001")
    state.add_log("extractor", "Quick evaluation started", extraction_id="ext_001")
    state.add_log("pipeline", "Outer iteration 1 completed", extraction_id="ext_001")
    state.add_log("judge", "Thorough evaluation started", extraction_id="ext_001")
    state.add_log("judge", "Verdict: NEEDS_REFINEMENT (score=0.75)", "warning", "ext_001")
    state.add_log("pipeline", "Outer iteration 2 started", extraction_id="ext_001")
    state.add_log("extractor", "Applying judge feedback", extraction_id="ext_001")
    state.add_log("extractor", "Generated 5 additional prompts", extraction_id="ext_001")
    state.add_log("extractor", "Optimizing layers [14-17]", extraction_id="ext_001")

    # Logs for other extractions
    state.add_log("pipeline", "Pipeline started", extraction_id="ext_002")
    state.add_log("extractor", "Generated 8 prompts", extraction_id="ext_002")

    state.add_log("pipeline", "Pipeline started", extraction_id="ext_003")
    state.add_log("extractor", "Generated 5 prompts", extraction_id="ext_003")
    state.add_log("extractor", "Optimizing layer 12", extraction_id="ext_003")

    state.add_log("pipeline", "Pipeline completed", extraction_id="ext_004")
    state.add_log("judge", "Verdict: ACCEPTED (score=0.85)", extraction_id="ext_004")

    return state


def run_demo() -> None:
    """Run the application with demo data."""
    state = create_demo_state()
    app = VectorForgeApp(state=state)
    app.run()


def run() -> None:
    """Run the application."""
    app = VectorForgeApp()
    app.run()


if __name__ == "__main__":
    run_demo()
