"""Main Vector Forge TUI application."""

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
    LogAdded,
    ProgressUpdated,
    AgentUpdated,
    AgentMessageAdded,
    ExtractionStatusChanged,
    MetricsUpdated,
    RefreshTime,
)
from vector_forge.ui.screens.dashboard import DashboardScreen
from vector_forge.ui.screens.samples import SamplesScreen
from vector_forge.ui.screens.logs import LogsScreen
from vector_forge.ui.screens.create_task import CreateTaskScreen

logger = logging.getLogger(__name__)


class VectorForgeApp(App):
    """Vector Forge Terminal User Interface."""

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
        """Initialize the application.

        Args:
            state: Optional custom UI state (for testing).
            demo_mode: If True, load demo data instead of real sessions.
            **kwargs: Additional arguments for App.
        """
        super().__init__(**kwargs)
        self._state = state
        self._demo_mode = demo_mode

        # Services (initialized on mount)
        self._session_service = None
        self._synchronizer = None
        self._task_executor = None
        self._extraction_runner = None

    def on_mount(self) -> None:
        # Use Textual's built-in gruvbox theme
        self.theme = DEFAULT_THEME

        if self._state is not None:
            import vector_forge.ui.state as state_module
            state_module._state = self._state
        else:
            # Reset to clean state
            reset_state()

        # Set up state to post messages to this app (thread-safe)
        get_state()._app = self

        if self._demo_mode:
            # Load demo data
            _populate_demo_state(get_state())
            # Start time refresh timer for demo
            self.set_interval(1.0, self._refresh_time)
        else:
            # Initialize services and load real sessions
            self._init_services()
            # Start time refresh timer
            self.set_interval(1.0, self._refresh_time)

        self.push_screen("dashboard")

    def _refresh_time(self) -> None:
        """Post a time refresh message for elapsed time displays."""
        self.post_message(RefreshTime())

    # ─────────────────────────────────────────────────────────────────
    # Streaming message handlers - these run in the main event loop
    # ─────────────────────────────────────────────────────────────────

    def on_log_added(self, message: LogAdded) -> None:
        """Handle new log entry - update state and notify screen."""
        state = get_state()
        entry = LogEntry(
            timestamp=message.timestamp,
            source=message.source,
            message=message.message,
            level=message.level,
            extraction_id=message.extraction_id,
            agent_id=message.agent_id,
        )
        state.logs.append(entry)
        if len(state.logs) > 1000:
            state.logs = state.logs[-1000:]

    def on_progress_updated(self, message: ProgressUpdated) -> None:
        """Handle progress update."""
        state = get_state()
        extraction = state.extractions.get(message.extraction_id)
        if extraction:
            extraction.progress = message.progress
            # Map phase string to enum
            phase_map = {
                "loading_model": Phase.INITIALIZING,
                "init": Phase.INITIALIZING,
                "generating_contrast": Phase.GENERATING_DATAPOINTS,
                "gen": Phase.GENERATING_DATAPOINTS,
                "extracting": Phase.OPTIMIZING,
                "opt": Phase.OPTIMIZING,
                "evaluating": Phase.EVALUATING,
                "eval": Phase.EVALUATING,
                "complete": Phase.COMPLETE,
                "done": Phase.COMPLETE,
                "failed": Phase.FAILED,
                "fail": Phase.FAILED,
            }
            extraction.phase = phase_map.get(message.phase, extraction.phase)

    def on_agent_updated(self, message: AgentUpdated) -> None:
        """Handle agent state change."""
        state = get_state()
        extraction = state.extractions.get(message.extraction_id)
        if extraction:
            agent = extraction.agents.get(message.agent_id)
            if agent:
                agent.status = AgentStatus(message.status)
                agent.current_tool = message.current_tool
                agent.turns = message.turns
                agent.tool_calls_count = message.tool_calls_count

    def on_agent_message_added(self, message: AgentMessageAdded) -> None:
        """Handle new message in agent conversation."""
        state = get_state()
        extraction = state.extractions.get(message.extraction_id)
        if extraction:
            agent = extraction.agents.get(message.agent_id)
            if agent:
                # Convert tool calls
                tool_calls = []
                for tc in message.tool_calls:
                    if isinstance(tc, dict):
                        tool_calls.append(ToolCall(**tc))
                    else:
                        tool_calls.append(tc)
                agent.add_message(
                    MessageRole(message.role),
                    message.content,
                    tool_calls,
                )

    def on_extraction_status_changed(self, message: ExtractionStatusChanged) -> None:
        """Handle extraction status change."""
        state = get_state()
        extraction = state.extractions.get(message.extraction_id)
        if extraction:
            extraction.status = ExtractionStatus(message.status)
            if message.completed_at:
                extraction.completed_at = message.completed_at

    def on_metrics_updated(self, message: MetricsUpdated) -> None:
        """Handle metrics update."""
        state = get_state()
        extraction = state.extractions.get(message.extraction_id)
        if extraction and message.data:
            if message.metric_type == "datapoints":
                extraction.datapoints = DatapointMetrics(**message.data)
            elif message.metric_type == "evaluation":
                extraction.evaluation = EvaluationMetrics(**message.data)

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

            # Register progress callback for UI updates
            self._extraction_runner.on_progress(self._on_extraction_progress)

            # Load existing sessions from storage
            loaded = self._synchronizer.load_existing_sessions()
            logger.info(f"Loaded {loaded} existing sessions")

        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            # App still works, just without persistence
            get_state().add_log(
                source="system",
                message=f"Storage unavailable: {e}",
                level="warning",
            )

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
        """Handle task creation from CreateTaskScreen.

        If services are available, creates a persistent session and starts extraction.
        Otherwise falls back to in-memory only.
        """
        state = get_state()

        # Extract behavior name from description
        description = message.description
        behavior_name = description.split("\n")[0].strip()
        if not behavior_name or len(behavior_name) > 50:
            behavior_name = description.split()[0].lower() if description else "extraction"

        if self._session_service is not None:
            # Create persistent session
            try:
                session_id = self._session_service.create_session(
                    behavior_name=behavior_name,
                    behavior_description=description,
                    config=message.config,
                )
                # The synchronizer will handle adding to UI state
                state.add_log(
                    source="session",
                    message=f"Created session: {behavior_name}",
                    level="info",
                    extraction_id=session_id,
                )
                logger.info(f"Created session {session_id} for {behavior_name}")

                # Start the extraction in the background
                if self._extraction_runner is not None:
                    state.add_log(
                        source="extraction",
                        message=f"Starting extraction: {behavior_name}",
                        level="info",
                        extraction_id=session_id,
                    )
                    self._extraction_runner.start_extraction(
                        session_id=session_id,
                        behavior_name=behavior_name,
                        behavior_description=description,
                        config=message.config,
                    )
                else:
                    state.add_log(
                        source="system",
                        message="Extraction runner not available",
                        level="warning",
                        extraction_id=session_id,
                    )

            except Exception as e:
                logger.error(f"Failed to create session: {e}")
                state.add_log(
                    source="system",
                    message=f"Session creation failed: {e}",
                    level="error",
                )
                # Fall back to in-memory
                self._create_inmemory_extraction(message, behavior_name, description)
        else:
            # No services - create in-memory only
            self._create_inmemory_extraction(message, behavior_name, description)

    def _on_extraction_progress(self, progress) -> None:
        """Handle extraction progress updates.

        Updates UI state based on extraction progress.
        Called from the extraction runner.

        Args:
            progress: ExtractionProgress object.
        """
        state = get_state()
        extraction = state.extractions.get(progress.session_id)

        if extraction is None:
            return

        # Update extraction progress
        extraction.progress = progress.progress

        # Map phase to UI phase
        phase_map = {
            "loading_model": Phase.INITIALIZING,
            "generating_contrast": Phase.GENERATING_DATAPOINTS,
            "extracting": Phase.OPTIMIZING,
            "evaluating": Phase.EVALUATING,
            "complete": Phase.COMPLETE,
            "failed": Phase.FAILED,
        }
        extraction.phase = phase_map.get(progress.phase, Phase.INITIALIZING)

        # Update status
        if progress.phase == "complete":
            extraction.status = ExtractionStatus.COMPLETE
            extraction.completed_at = time.time()
        elif progress.phase == "failed":
            extraction.status = ExtractionStatus.FAILED
            extraction.completed_at = time.time()
        else:
            extraction.status = ExtractionStatus.RUNNING

        # Log the progress
        state.add_log(
            source="extraction",
            message=progress.message,
            level="error" if progress.error else "info",
            extraction_id=progress.session_id,
        )

        # Notify UI of change
        state._notify()

    def _create_inmemory_extraction(
        self,
        message: CreateTaskScreen.TaskCreated,
        behavior_name: str,
        description: str,
    ) -> None:
        """Create an in-memory extraction (no persistence)."""
        extraction = ExtractionUIState(
            id=f"ext_{int(time.time())}",
            behavior_name=behavior_name,
            behavior_description=description,
            model=message.config.extractor_model,
            status=ExtractionStatus.PENDING,
            phase=Phase.INITIALIZING,
            max_outer_iterations=message.config.num_samples,
        )

        state = get_state()
        state.add_extraction(extraction)
        state.add_log(
            source="pipeline",
            message=f"Created task: {behavior_name} (in-memory)",
            level="info",
            extraction_id=extraction.id,
        )

    @property
    def session_service(self):
        """Get the session service (may be None)."""
        return self._session_service

    @property
    def synchronizer(self):
        """Get the synchronizer (may be None)."""
        return self._synchronizer

    @property
    def task_executor(self):
        """Get the task executor (may be None)."""
        return self._task_executor

    @property
    def extraction_runner(self):
        """Get the extraction runner (may be None)."""
        return self._extraction_runner


def _populate_demo_state(state: UIState) -> None:
    """Populate state with demo data.

    Used for development and testing purposes.
    """
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
        MessageRole.USER,
        "Please generate contrast pairs for the sycophancy behavior."
    )
    extractor.add_message(
        MessageRole.ASSISTANT,
        "I'll generate contrastive pairs for sycophancy behavior.",
        [
            ToolCall(
                id="tc_0", name="generate_pairs",
                arguments='{"behavior": "sycophancy", "count": 12}',
                result='{"pairs": 12, "quality": 0.85}',
                status="success", started_at=time.time()-60, completed_at=time.time()-55
            ),
        ]
    )
    extractor.add_message(
        MessageRole.ASSISTANT,
        "Generated 12 contrastive pairs. Running extraction on layer 16.",
        [
            ToolCall(
                id="tc_1", name="extract_vector",
                arguments='{"layer": 16, "pairs": 12}',
                status="running", started_at=time.time()-5
            ),
            ToolCall(
                id="tc_2", name="evaluate_quality",
                arguments='{"threshold": 0.7}',
                status="pending"
            ),
        ]
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
        model="claude-opus-4-5",
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
        model="claude-sonnet-4-5",
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


def create_demo_state() -> UIState:
    """Create demo state with sample data."""
    state = reset_state()
    _populate_demo_state(state)
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
