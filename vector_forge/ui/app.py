"""Main Vector Forge TUI application."""

import time
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
    AgentUIState,
    AgentStatus,
    MessageRole,
    ToolCall,
    get_state,
    reset_state,
)
from vector_forge.ui.screens.dashboard import DashboardScreen
from vector_forge.ui.screens.samples import SamplesScreen
from vector_forge.ui.screens.logs import LogsScreen
from vector_forge.ui.screens.help import HelpModal
from vector_forge.ui.screens.create_task import CreateTaskScreen


class VectorForgeApp(App):
    """Vector Forge Terminal User Interface."""

    TITLE = "Vector Forge"
    CSS_PATH = Path(__file__).parent / "forge.tcss"

    SCREENS = {
        "dashboard": DashboardScreen,
        "samples": SamplesScreen,
        "logs": LogsScreen,
        "help": HelpModal,
        "create_task": CreateTaskScreen,
    }

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
        Binding("n", "new_task", "New Task", show=True),
    ]

    def __init__(self, state: Optional[UIState] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._state = state

    def on_mount(self) -> None:
        self.register_theme(forge_dark)
        self.theme = "forge-dark"

        if self._state is not None:
            import vector_forge.ui.state as state_module
            state_module._state = self._state

        self.push_screen("dashboard")

    def switch_screen(self, screen_name: str) -> None:
        if screen_name in ("help", "create_task"):
            self.push_screen(screen_name)
        elif screen_name in ("dashboard", "samples", "logs"):
            current = getattr(self.screen, "name", None)
            if current is None:
                current = type(self.screen).__name__.lower().replace("screen", "")
            if screen_name == current:
                return
            if len(self.screen_stack) > 1:
                self.pop_screen()
            self.push_screen(screen_name)

    def action_new_task(self) -> None:
        self.push_screen("create_task")

    def on_create_task_screen_task_created(
        self,
        message: CreateTaskScreen.TaskCreated,
    ) -> None:
        extraction = ExtractionUIState(
            id=f"ext_{int(time.time())}",
            behavior_name=message.description.split()[0].lower(),
            behavior_description=message.description,
            model=message.config.extractor_model,
            status=ExtractionStatus.PENDING,
            phase=Phase.INITIALIZING,
            max_outer_iterations=message.config.num_samples,
        )

        state = get_state()
        state.add_extraction(extraction)
        state.add_log(
            "pipeline",
            f"Created new task: {extraction.behavior_name}",
            extraction_id=extraction.id,
        )


def create_demo_state() -> UIState:
    """Create demo state with sample data."""
    state = reset_state()

    # Main running extraction
    extraction = ExtractionUIState(
        id="ext_001",
        behavior_name="sycophancy",
        behavior_description="Agreeing with the user even when they are factually wrong",
        model="claude-3.5-sonnet",
        status=ExtractionStatus.RUNNING,
        phase=Phase.OPTIMIZING,
        progress=0.67,
        outer_iteration=2,
        max_outer_iterations=3,
        inner_turn=15,
        max_inner_turns=50,
        current_layer=16,
        started_at=time.time() - 154,
        datapoints=DatapointMetrics(
            total=15, keep=12, review=2, remove=1, diversity=0.68, clusters=4
        ),
        evaluation=EvaluationMetrics(
            behavior=0.82, coherence=0.75, specificity=0.80, overall=0.79,
            best_layer=16, best_strength=1.2, verdict="needs_refinement"
        ),
    )

    # Add sample agents
    extractor = AgentUIState(
        id="sample_001",
        name="Sample 1",
        role="T=0.7 L=16 seed=42",
        status=AgentStatus.RUNNING,
        started_at=time.time() - 120,
        turns=8,
        tool_calls_count=15,
        current_tool="extract_vector",
    )
    extractor.add_message(
        MessageRole.SYSTEM,
        "Extracting steering vector for sycophancy behavior."
    )
    extractor.add_message(
        MessageRole.ASSISTANT,
        "Generated 12 contrastive pairs. Running extraction on layer 16.",
        [ToolCall(
            id="tc_1", name="extract_vector",
            arguments='{"layer": 16, "pairs": 12}',
            status="running", started_at=time.time()-5
        )]
    )
    extraction.add_agent(extractor)

    sample2 = AgentUIState(
        id="sample_002",
        name="Sample 2",
        role="T=0.9 L=16 seed=123",
        status=AgentStatus.RUNNING,
        started_at=time.time() - 90,
        turns=5,
        tool_calls_count=12,
        current_tool="optimize",
    )
    sample2.add_message(MessageRole.ASSISTANT, "Optimizing vector quality.")
    extraction.add_agent(sample2)

    sample3 = AgentUIState(
        id="sample_003",
        name="Sample 3",
        role="T=0.5 L=14 seed=456",
        status=AgentStatus.COMPLETE,
        started_at=time.time() - 150,
        completed_at=time.time() - 30,
        turns=10,
        tool_calls_count=22,
    )
    sample3.add_message(MessageRole.ASSISTANT, "Extraction complete. Score: 0.78")
    extraction.add_agent(sample3)

    sample4 = AgentUIState(
        id="sample_004",
        name="Sample 4",
        role="T=0.7 L=15 seed=789",
        status=AgentStatus.WAITING,
        started_at=time.time() - 60,
        turns=2,
        tool_calls_count=4,
    )
    extraction.add_agent(sample4)

    state.add_extraction(extraction)

    # Second extraction
    extraction2 = ExtractionUIState(
        id="ext_002",
        behavior_name="honesty",
        behavior_description="Being truthful and accurate in responses",
        model="gpt-4o",
        status=ExtractionStatus.RUNNING,
        phase=Phase.GENERATING_DATAPOINTS,
        progress=0.35,
        outer_iteration=1,
        max_outer_iterations=3,
        started_at=time.time() - 80,
        datapoints=DatapointMetrics(total=6, keep=5, review=1, diversity=0.45, clusters=2),
    )
    state.add_extraction(extraction2)

    # Completed extraction
    extraction3 = ExtractionUIState(
        id="ext_003",
        behavior_name="curiosity",
        behavior_description="Showing intellectual curiosity",
        model="claude-3.5-sonnet",
        status=ExtractionStatus.COMPLETE,
        phase=Phase.COMPLETE,
        progress=1.0,
        outer_iteration=3,
        max_outer_iterations=3,
        started_at=time.time() - 300,
        completed_at=time.time() - 50,
        datapoints=DatapointMetrics(total=18, keep=16, review=2, diversity=0.78, clusters=5),
        evaluation=EvaluationMetrics(
            behavior=0.88, coherence=0.82, specificity=0.85, overall=0.85,
            best_layer=15, best_strength=1.3, verdict="accepted"
        ),
    )
    state.add_extraction(extraction3)

    # Add logs
    state.add_log("pipeline", "Started extraction: sycophancy", extraction_id="ext_001")
    state.add_log("sample", "Sample 1: Generated 12 contrast pairs", extraction_id="ext_001")
    state.add_log("sample", "Sample 2: Optimizing layer 16", extraction_id="ext_001")
    state.add_log("sample", "Sample 3: Completed with score 0.78", extraction_id="ext_001")
    state.add_log("eval", "Best score so far: 0.79", "warning", "ext_001")
    state.add_log("pipeline", "Started extraction: honesty", extraction_id="ext_002")
    state.add_log("pipeline", "Completed extraction: curiosity", extraction_id="ext_003")
    state.add_log("eval", "Final score: 0.85 - ACCEPTED", extraction_id="ext_003")

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
