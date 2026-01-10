"""Main Vector Forge TUI application.

Architecture: Event-Sourcing Pattern
────────────────────────────────────
Events flow in one direction:
    Background Thread → App (handler) → State (update) → Screen (projection)

The App is the central event handler:
1. Receives events from background threads (thread-safe via post_message)
2. Updates state in the main thread
3. Forwards events to the current screen for UI updates
"""

import logging
import time
from pathlib import Path
from typing import Optional

from textual.app import App
from textual.binding import Binding

from vector_forge.ui.theme import DEFAULT_THEME
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
    LogEntry,
    get_state,
    reset_state,
)
from vector_forge.ui.messages import (
    # Task events
    TaskCreated,
    TaskProgressChanged,
    TaskStatusChanged,
    TaskRemoved,
    TaskSelected,
    # Agent events
    AgentSpawned,
    AgentStatusChanged,
    AgentProgressChanged,
    AgentMessageReceived,
    AgentSelected,
    # Log events
    LogEmitted,
    # Metrics events
    DatapointMetricsChanged,
    EvaluationMetricsChanged,
    # Timer
    TimeTick,
)
from vector_forge.ui.screens.dashboard import DashboardScreen
from vector_forge.ui.screens.samples import SamplesScreen
from vector_forge.ui.screens.logs import LogsScreen
from vector_forge.ui.screens.create_task import CreateTaskScreen

logger = logging.getLogger(__name__)


class VectorForgeApp(App):
    """Vector Forge Terminal User Interface.

    Central event handler that:
    - Receives all events
    - Updates state
    - Forwards events to screens
    """

    TITLE = "Vector Forge"
    CSS_PATH = Path(__file__).parent / "forge.tcss"

    SCREENS = {
        "dashboard": DashboardScreen,
        "samples": SamplesScreen,
        "logs": LogsScreen,
        "create_task": CreateTaskScreen,
    }

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
        Binding("n", "new_task", "New Task", show=True),
        Binding("?", "toggle_help_panel", "Help"),
    ]

    def __init__(
        self,
        state: Optional[UIState] = None,
        demo_mode: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._state = state
        self._demo_mode = demo_mode
        self._session_service = None
        self._synchronizer = None
        self._task_executor = None
        self._extraction_runner = None

    def on_mount(self) -> None:
        self.theme = DEFAULT_THEME

        if self._state is not None:
            import vector_forge.ui.state as state_module
            state_module._state = self._state
        else:
            reset_state()

        if self._demo_mode:
            self._populate_demo_state()
        else:
            self._init_services()

        # Start the time tick timer (only timer in the app)
        self.set_interval(1.0, self._emit_time_tick)

        self.push_screen("dashboard")

    def _emit_time_tick(self) -> None:
        """Emit TimeTick to current screen for elapsed time updates."""
        self.screen.post_message(TimeTick())

    # ─────────────────────────────────────────────────────────────────────────
    # Event Handlers: Update State + Forward to Screen
    # ─────────────────────────────────────────────────────────────────────────

    def on_task_created(self, event: TaskCreated) -> None:
        """Handle task creation."""
        state = get_state()
        extraction = ExtractionUIState(
            id=event.task_id,
            behavior_name=event.name,
            behavior_description=event.description,
            status=ExtractionStatus.PENDING,
            phase=Phase.INITIALIZING,
            started_at=time.time(),
        )
        state.extractions[event.task_id] = extraction
        if state.selected_id is None:
            state.selected_id = event.task_id
        self._forward_to_screen(event)

    def on_task_progress_changed(self, event: TaskProgressChanged) -> None:
        """Handle task progress update."""
        state = get_state()
        extraction = state.extractions.get(event.task_id)
        if extraction:
            extraction.progress = event.progress
            phase_map = {
                "init": Phase.INITIALIZING,
                "gen": Phase.GENERATING_DATAPOINTS,
                "opt": Phase.OPTIMIZING,
                "eval": Phase.EVALUATING,
                "done": Phase.COMPLETE,
                "fail": Phase.FAILED,
            }
            extraction.phase = phase_map.get(event.phase, extraction.phase)
        self._forward_to_screen(event)

    def on_task_status_changed(self, event: TaskStatusChanged) -> None:
        """Handle task status change."""
        state = get_state()
        extraction = state.extractions.get(event.task_id)
        if extraction:
            extraction.status = ExtractionStatus(event.status)
            if event.completed_at:
                extraction.completed_at = event.completed_at
        self._forward_to_screen(event)

    def on_task_removed(self, event: TaskRemoved) -> None:
        """Handle task removal."""
        state = get_state()
        if event.task_id in state.extractions:
            del state.extractions[event.task_id]
            if state.selected_id == event.task_id:
                state.selected_id = next(iter(state.extractions), None)
        self._forward_to_screen(event)

    def on_task_selected(self, event: TaskSelected) -> None:
        """Handle task selection change."""
        state = get_state()
        state.selected_id = event.task_id
        self._forward_to_screen(event)

    def on_agent_spawned(self, event: AgentSpawned) -> None:
        """Handle agent spawn."""
        state = get_state()
        extraction = state.extractions.get(event.task_id)
        if extraction:
            agent = AgentUIState(
                id=event.agent_id,
                name=event.name,
                role=event.role,
                status=AgentStatus.IDLE,
                started_at=time.time(),
            )
            extraction.agents[event.agent_id] = agent
        self._forward_to_screen(event)

    def on_agent_status_changed(self, event: AgentStatusChanged) -> None:
        """Handle agent status change."""
        state = get_state()
        extraction = state.extractions.get(event.task_id)
        if extraction:
            agent = extraction.agents.get(event.agent_id)
            if agent:
                agent.status = AgentStatus(event.status)
                agent.current_tool = event.current_tool
        self._forward_to_screen(event)

    def on_agent_progress_changed(self, event: AgentProgressChanged) -> None:
        """Handle agent progress update."""
        state = get_state()
        extraction = state.extractions.get(event.task_id)
        if extraction:
            agent = extraction.agents.get(event.agent_id)
            if agent:
                agent.turns = event.turns
                agent.tool_calls_count = event.tool_calls_count
        self._forward_to_screen(event)

    def on_agent_message_received(self, event: AgentMessageReceived) -> None:
        """Handle new agent message."""
        state = get_state()
        extraction = state.extractions.get(event.task_id)
        if extraction:
            agent = extraction.agents.get(event.agent_id)
            if agent:
                tool_calls = [
                    ToolCall(**tc) if isinstance(tc, dict) else tc
                    for tc in event.tool_calls
                ]
                agent.add_message(
                    MessageRole(event.role),
                    event.content,
                    tool_calls,
                )
        self._forward_to_screen(event)

    def on_agent_selected(self, event: AgentSelected) -> None:
        """Handle agent selection change."""
        state = get_state()
        extraction = state.extractions.get(event.task_id)
        if extraction:
            extraction.selected_agent_id = event.agent_id
        self._forward_to_screen(event)

    def on_log_emitted(self, event: LogEmitted) -> None:
        """Handle log entry."""
        state = get_state()
        entry = LogEntry(
            timestamp=event.timestamp,
            source=event.source,
            message=event.message,
            level=event.level,
            extraction_id=event.task_id,
            agent_id=event.agent_id,
        )
        state.logs.append(entry)
        if len(state.logs) > 1000:
            state.logs = state.logs[-1000:]
        self._forward_to_screen(event)

    def on_datapoint_metrics_changed(self, event: DatapointMetricsChanged) -> None:
        """Handle datapoint metrics update."""
        state = get_state()
        extraction = state.extractions.get(event.task_id)
        if extraction:
            extraction.datapoints = DatapointMetrics(
                total=event.total,
                keep=event.keep,
                review=event.review,
                remove=event.remove,
                diversity=event.diversity,
                clusters=event.clusters,
            )
        self._forward_to_screen(event)

    def on_evaluation_metrics_changed(self, event: EvaluationMetricsChanged) -> None:
        """Handle evaluation metrics update."""
        state = get_state()
        extraction = state.extractions.get(event.task_id)
        if extraction:
            extraction.evaluation = EvaluationMetrics(
                behavior=event.behavior,
                coherence=event.coherence,
                specificity=event.specificity,
                overall=event.overall,
                best_layer=event.best_layer,
                best_strength=event.best_strength,
                verdict=event.verdict,
            )
        self._forward_to_screen(event)

    def _forward_to_screen(self, event) -> None:
        """Forward event to current screen for UI update."""
        try:
            self.screen.post_message(event)
        except Exception:
            pass  # Screen might not be ready

    # ─────────────────────────────────────────────────────────────────────────
    # Services and Task Creation
    # ─────────────────────────────────────────────────────────────────────────

    def _init_services(self) -> None:
        """Initialize services and load existing sessions."""
        try:
            from vector_forge.services import (
                SessionService,
                UIStateSynchronizer,
                TaskExecutor,
                ExtractionRunner,
            )

            self._session_service = SessionService()
            self._synchronizer = UIStateSynchronizer(
                get_state(),
                self._session_service,
            )
            self._task_executor = TaskExecutor(self._session_service)
            self._extraction_runner = ExtractionRunner(
                self._session_service,
                self._task_executor,
            )

            self._extraction_runner.on_progress(self._on_extraction_progress)
            loaded = self._synchronizer.load_existing_sessions()
            logger.info(f"Loaded {loaded} existing sessions")

        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            self.post_message(LogEmitted(
                timestamp=time.time(),
                source="system",
                message=f"Storage unavailable: {e}",
                level="warning",
            ))

    def _on_extraction_progress(self, progress) -> None:
        """Handle extraction progress from runner - convert to events."""
        self.post_message(TaskProgressChanged(
            task_id=progress.session_id,
            progress=progress.progress,
            phase=progress.phase,
            message=progress.message,
        ))

        if progress.phase == "complete":
            self.post_message(TaskStatusChanged(
                task_id=progress.session_id,
                status="complete",
                completed_at=time.time(),
            ))
        elif progress.phase == "failed":
            self.post_message(TaskStatusChanged(
                task_id=progress.session_id,
                status="failed",
                completed_at=time.time(),
                error=progress.error,
            ))

        self.post_message(LogEmitted(
            timestamp=time.time(),
            source="extraction",
            message=progress.message,
            level="error" if progress.error else "info",
            task_id=progress.session_id,
        ))

    def switch_screen(self, screen_name: str) -> None:
        if screen_name == "create_task":
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

    def action_toggle_help_panel(self) -> None:
        from textual.widgets import HelpPanel
        try:
            self.screen.query_one(HelpPanel).remove()
        except Exception:
            self.screen.mount(HelpPanel())

    def on_create_task_screen_task_created(
        self,
        message: CreateTaskScreen.TaskCreated,
    ) -> None:
        """Handle task creation from CreateTaskScreen."""
        description = message.description
        behavior_name = description.split("\n")[0].strip()
        if not behavior_name or len(behavior_name) > 50:
            behavior_name = description.split()[0].lower() if description else "extraction"

        task_id = f"ext_{int(time.time())}"

        # Post TaskCreated event
        self.post_message(TaskCreated(
            task_id=task_id,
            name=behavior_name,
            description=description,
        ))

        # Post initial status
        self.post_message(TaskStatusChanged(
            task_id=task_id,
            status="running",
        ))

        # Log it
        self.post_message(LogEmitted(
            timestamp=time.time(),
            source="pipeline",
            message=f"Created task: {behavior_name}",
            level="info",
            task_id=task_id,
        ))

        # Start extraction if services available
        if self._session_service is not None and self._extraction_runner is not None:
            try:
                session_id = self._session_service.create_session(
                    behavior_name=behavior_name,
                    behavior_description=description,
                    config=message.config,
                )
                self._extraction_runner.start_extraction(
                    session_id=session_id,
                    behavior_name=behavior_name,
                    behavior_description=description,
                    config=message.config,
                )
            except Exception as e:
                logger.error(f"Failed to start extraction: {e}")

    @property
    def session_service(self):
        return self._session_service

    @property
    def synchronizer(self):
        return self._synchronizer

    @property
    def task_executor(self):
        return self._task_executor

    @property
    def extraction_runner(self):
        return self._extraction_runner

    # ─────────────────────────────────────────────────────────────────────────
    # Demo Mode
    # ─────────────────────────────────────────────────────────────────────────

    def _populate_demo_state(self) -> None:
        """Populate with demo data using events."""
        now = time.time()

        # Task 1: Running extraction
        self.post_message(TaskCreated(
            task_id="ext_001",
            name="sycophancy",
            description="Agreeing with the user even when they are factually wrong",
        ))

        # Update task 1 state directly (for demo - normally events would do this)
        state = get_state()
        ext1 = state.extractions.get("ext_001")
        if ext1:
            ext1.model = "claude-3.5-sonnet"
            ext1.status = ExtractionStatus.RUNNING
            ext1.phase = Phase.OPTIMIZING
            ext1.progress = 67.0
            ext1.outer_iteration = 2
            ext1.max_outer_iterations = 3
            ext1.inner_turn = 15
            ext1.max_inner_turns = 50
            ext1.current_layer = 16
            ext1.started_at = now - 154
            ext1.datapoints = DatapointMetrics(
                total=15, keep=12, review=2, remove=1, diversity=0.68, clusters=4
            )
            ext1.evaluation = EvaluationMetrics(
                behavior=0.82, coherence=0.75, specificity=0.80, overall=0.79,
                best_layer=16, best_strength=1.2, verdict="needs_refinement"
            )

            # Add agents
            agent1 = AgentUIState(
                id="agent_001",
                name="Sample 1",
                role="T=0.7 L=16 seed=42",
                status=AgentStatus.RUNNING,
                started_at=now - 120,
                turns=8,
                tool_calls_count=15,
                current_tool="extract_vector",
            )
            agent1.add_message(MessageRole.SYSTEM, "Extracting steering vector for sycophancy behavior.")
            agent1.add_message(MessageRole.USER, "Please generate contrast pairs for the sycophancy behavior.")
            agent1.add_message(
                MessageRole.ASSISTANT,
                "I'll generate contrastive pairs for sycophancy behavior.",
                [ToolCall(
                    id="tc_0", name="generate_pairs",
                    arguments='{"behavior": "sycophancy", "count": 12}',
                    result='{"pairs": 12, "quality": 0.85}',
                    status="success", started_at=now-60, completed_at=now-55
                )]
            )
            agent1.add_message(
                MessageRole.ASSISTANT,
                "Generated 12 contrastive pairs. Running extraction on layer 16.",
                [
                    ToolCall(
                        id="tc_1", name="extract_vector",
                        arguments='{"layer": 16, "pairs": 12}',
                        status="running", started_at=now-5
                    ),
                    ToolCall(
                        id="tc_2", name="evaluate_quality",
                        arguments='{"threshold": 0.7}',
                        status="pending"
                    ),
                ]
            )
            ext1.agents["agent_001"] = agent1

            agent2 = AgentUIState(
                id="agent_002",
                name="Sample 2",
                role="T=0.9 L=16 seed=123",
                status=AgentStatus.RUNNING,
                started_at=now - 90,
                turns=5,
                tool_calls_count=12,
                current_tool="optimize",
            )
            agent2.add_message(MessageRole.ASSISTANT, "Optimizing vector quality.")
            ext1.agents["agent_002"] = agent2

            agent3 = AgentUIState(
                id="agent_003",
                name="Sample 3",
                role="T=0.5 L=14 seed=456",
                status=AgentStatus.COMPLETE,
                started_at=now - 150,
                completed_at=now - 30,
                turns=10,
                tool_calls_count=22,
            )
            agent3.add_message(MessageRole.ASSISTANT, "Extraction complete. Score: 0.78")
            ext1.agents["agent_003"] = agent3

            agent4 = AgentUIState(
                id="agent_004",
                name="Sample 4",
                role="T=0.7 L=15 seed=789",
                status=AgentStatus.WAITING,
                started_at=now - 60,
                turns=2,
                tool_calls_count=4,
            )
            ext1.agents["agent_004"] = agent4

        # Task 2: Another running extraction
        self.post_message(TaskCreated(
            task_id="ext_002",
            name="honesty",
            description="Being truthful and accurate in responses",
        ))
        ext2 = state.extractions.get("ext_002")
        if ext2:
            ext2.model = "claude-opus-4-5"
            ext2.status = ExtractionStatus.RUNNING
            ext2.phase = Phase.GENERATING_DATAPOINTS
            ext2.progress = 35.0
            ext2.outer_iteration = 1
            ext2.max_outer_iterations = 3
            ext2.started_at = now - 80
            ext2.datapoints = DatapointMetrics(total=6, keep=5, review=1, diversity=0.45, clusters=2)

        # Task 3: Completed extraction
        self.post_message(TaskCreated(
            task_id="ext_003",
            name="curiosity",
            description="Showing intellectual curiosity",
        ))
        ext3 = state.extractions.get("ext_003")
        if ext3:
            ext3.model = "claude-sonnet-4-5"
            ext3.status = ExtractionStatus.COMPLETE
            ext3.phase = Phase.COMPLETE
            ext3.progress = 100.0
            ext3.outer_iteration = 3
            ext3.max_outer_iterations = 3
            ext3.started_at = now - 300
            ext3.completed_at = now - 50
            ext3.datapoints = DatapointMetrics(total=18, keep=16, review=2, diversity=0.78, clusters=5)
            ext3.evaluation = EvaluationMetrics(
                behavior=0.88, coherence=0.82, specificity=0.85, overall=0.85,
                best_layer=15, best_strength=1.3, verdict="accepted"
            )

        # Add some logs
        for log_data in [
            ("pipeline", "Started extraction: sycophancy", "ext_001"),
            ("sample", "Sample 1: Generated 12 contrast pairs", "ext_001"),
            ("sample", "Sample 2: Optimizing layer 16", "ext_001"),
            ("sample", "Sample 3: Completed with score 0.78", "ext_001"),
            ("eval", "Best score so far: 0.79", "ext_001"),
            ("pipeline", "Started extraction: honesty", "ext_002"),
            ("pipeline", "Completed extraction: curiosity", "ext_003"),
            ("eval", "Final score: 0.85 - ACCEPTED", "ext_003"),
        ]:
            state.logs.append(LogEntry(
                timestamp=now,
                source=log_data[0],
                message=log_data[1],
                level="warning" if "warning" in log_data[1].lower() else "info",
                extraction_id=log_data[2] if len(log_data) > 2 else None,
            ))


def create_demo_state() -> UIState:
    """Create demo state with sample data."""
    state = reset_state()
    return state


def run_demo() -> None:
    """Run the application with demo data."""
    app = VectorForgeApp(demo_mode=True)
    app.run()


def run() -> None:
    """Run the application with real data from storage."""
    app = VectorForgeApp()
    app.run()


if __name__ == "__main__":
    run()
