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
    AgentUIState,
    AgentStatus,
    MessageRole,
    ToolCall,
    get_state,
    reset_state,
)
from vector_forge.ui.screens.dashboard import DashboardScreen
from vector_forge.ui.screens.agents import AgentsScreen
from vector_forge.ui.screens.logs import LogsScreen
from vector_forge.ui.screens.help import HelpModal


class VectorForgeApp(App):
    """Vector Forge Terminal User Interface.

    A professional terminal UI for monitoring and controlling
    steering vector extraction pipelines.

    Screens:
    - Dashboard: Focused view of single extraction progress
    - Agents: Split-panel view of parallel agents with inspector
    - Logs: Full log viewer with filtering
    """

    TITLE = "Vector Forge"
    CSS_PATH = Path(__file__).parent / "forge.tcss"

    SCREENS = {
        "dashboard": DashboardScreen,
        "agents": AgentsScreen,
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
        self.register_theme(forge_dark)
        self.theme = "forge-dark"

        if self._state is not None:
            import vector_forge.ui.state as state_module
            state_module._state = self._state

        self.push_screen("dashboard")

    def switch_screen(self, screen_name: str) -> None:
        """Switch to a different screen.

        Args:
            screen_name: Name of screen to switch to.
        """
        if screen_name == "help":
            self.push_screen("help")
        elif screen_name in ("dashboard", "agents", "logs"):
            current = getattr(self.screen, "name", None) or type(self.screen).__name__.lower().replace("screen", "")
            if screen_name == current:
                return
            if len(self.screen_stack) > 1:
                self.pop_screen()
            self.push_screen(screen_name)


def create_demo_state() -> UIState:
    """Create demo state with sample data including agents."""
    import time

    state = reset_state()

    # Create main extraction
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
        started_at=time.time() - 154,
        datapoints=DatapointMetrics(
            total=15, keep=12, review=2, remove=1, diversity=0.68, clusters=4
        ),
        evaluation=EvaluationMetrics(
            behavior=0.82, coherence=0.75, specificity=0.80, overall=0.79,
            best_layer=16, best_strength=1.2, verdict="needs_refinement"
        ),
    )

    # Add agents to extraction
    extractor = AgentUIState(
        id="agent_extractor",
        name="Extractor",
        role="datapoint generation",
        status=AgentStatus.RUNNING,
        started_at=time.time() - 120,
        turns=8,
        tool_calls_count=15,
        current_tool="generate_prompt",
    )
    extractor.add_message(
        MessageRole.SYSTEM,
        "You are an expert at generating contrastive prompt pairs for steering vector extraction."
    )
    extractor.add_message(
        MessageRole.USER,
        "Generate prompts that elicit sycophantic vs honest behavior."
    )
    extractor.add_message(
        MessageRole.ASSISTANT,
        "I'll generate contrastive pairs. Starting with a scenario where the user makes a factually incorrect claim.",
        [ToolCall(
            id="tc_1", name="generate_prompt",
            arguments='{"scenario": "user claims earth is flat"}',
            result='{"positive": "...", "negative": "..."}',
            status="success", started_at=time.time()-60, completed_at=time.time()-59
        )]
    )
    extractor.add_message(
        MessageRole.ASSISTANT,
        "Generating another pair for mathematical misconceptions.",
        [ToolCall(
            id="tc_2", name="generate_prompt",
            arguments='{"scenario": "user claims 2+2=5"}',
            status="running", started_at=time.time()-5
        )]
    )
    extraction.add_agent(extractor)

    optimizer = AgentUIState(
        id="agent_optimizer",
        name="Optimizer",
        role="vector optimization",
        status=AgentStatus.RUNNING,
        started_at=time.time() - 90,
        turns=5,
        tool_calls_count=12,
        current_tool="optimize_layer",
    )
    optimizer.add_message(MessageRole.SYSTEM, "Optimize steering vectors across layers.")
    optimizer.add_message(
        MessageRole.ASSISTANT,
        "Optimizing layer 16 with current datapoints.",
        [ToolCall(
            id="tc_opt_1", name="optimize_layer",
            arguments='{"layer": 16, "lr": 0.01}',
            status="running", started_at=time.time()-2
        )]
    )
    extraction.add_agent(optimizer)

    judge = AgentUIState(
        id="agent_judge",
        name="Judge",
        role="quality evaluation",
        status=AgentStatus.WAITING,
        started_at=time.time() - 30,
        turns=2,
        tool_calls_count=4,
    )
    judge.add_message(MessageRole.SYSTEM, "Evaluate vector quality.")
    judge.add_message(MessageRole.ASSISTANT, "Waiting for optimization to complete before evaluation.")
    extraction.add_agent(judge)

    state.add_extraction(extraction)

    # Add second extraction
    extraction2 = ExtractionUIState(
        id="ext_002",
        behavior_name="honesty",
        behavior_description="Being truthful and accurate in responses",
        model="gpt-4o",
        status=ExtractionStatus.RUNNING,
        phase=Phase.GENERATING_DATAPOINTS,
        progress=35.0,
        outer_iteration=1,
        max_outer_iterations=3,
        started_at=time.time() - 80,
        datapoints=DatapointMetrics(total=6, keep=5, review=1, diversity=0.45, clusters=2),
    )

    ext2_agent = AgentUIState(
        id="ext2_extractor",
        name="Extractor",
        role="datapoint generation",
        status=AgentStatus.RUNNING,
        started_at=time.time() - 60,
        turns=3,
        tool_calls_count=6,
    )
    extraction2.add_agent(ext2_agent)
    state.add_extraction(extraction2)

    # Add completed extraction
    extraction3 = ExtractionUIState(
        id="ext_003",
        behavior_name="curiosity",
        behavior_description="Showing intellectual curiosity",
        model="claude-3.5-sonnet",
        status=ExtractionStatus.COMPLETE,
        phase=Phase.COMPLETE,
        progress=100.0,
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
    state.add_log("pipeline", "Pipeline started", extraction_id="ext_001")
    state.add_log("extractor", "Generated 10 initial prompts", extraction_id="ext_001")
    state.add_log("optimizer", "Optimizing layer 14: loss=0.045", extraction_id="ext_001")
    state.add_log("optimizer", "Optimizing layer 15: loss=0.032", extraction_id="ext_001")
    state.add_log("optimizer", "Optimizing layer 16: loss=0.023", extraction_id="ext_001")
    state.add_log("judge", "Evaluation: NEEDS_REFINEMENT (0.79)", "warning", "ext_001")
    state.add_log("extractor", "Generating additional prompts", extraction_id="ext_001")
    state.add_log("pipeline", "Pipeline started", extraction_id="ext_002")
    state.add_log("extractor", "Generated 6 prompts", extraction_id="ext_002")
    state.add_log("pipeline", "Pipeline completed", extraction_id="ext_003")
    state.add_log("judge", "Evaluation: ACCEPTED (0.85)", extraction_id="ext_003")

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
