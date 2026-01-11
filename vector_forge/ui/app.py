"""Main Vector Forge TUI application.

Uses JSONL streaming for real-time UI updates following Textual's
native patterns for thread-safe communication.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from textual import work
from textual.app import App
from textual.binding import Binding
from textual.worker import get_current_worker

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
from vector_forge.ui.screens.dashboard import DashboardScreen
from vector_forge.ui.screens.samples import SamplesScreen
from vector_forge.ui.screens.logs import LogsScreen
from vector_forge.ui.screens.create_task import CreateTaskScreen
from vector_forge.ui.screens.chat import ChatScreen
from vector_forge.ui.watcher import (
    NewEvents,
    EventStreamWatcher,
    apply_event_to_state,
)

logger = logging.getLogger(__name__)

# Poll interval for JSONL watcher (100ms = responsive but not excessive)
WATCHER_POLL_INTERVAL = 0.1


class VectorForgeApp(App):
    """Vector Forge Terminal User Interface.

    Uses JSONL streaming for real-time updates:
    - Events are written to JSONL files by the pipeline
    - A worker thread watches the files and posts NewEvents messages
    - Messages are handled on the main thread for thread-safe UI updates
    """

    TITLE = "Vector Forge"
    CSS_PATH = Path(__file__).parent / "forge.tcss"

    SCREENS = {
        "dashboard": DashboardScreen,
        "samples": SamplesScreen,
        "logs": LogsScreen,
        "create_task": CreateTaskScreen,
        "chat": ChatScreen,
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
        self._session_loader = None
        self._task_executor = None
        self._extraction_runner = None

        # JSONL watcher
        self._watcher: Optional[EventStreamWatcher] = None
        self._watching = False

    def on_mount(self) -> None:
        """Initialize app on mount."""
        # Configure CUDA memory allocator early to prevent fragmentation
        from vector_forge.tasks.gpu_memory import configure_cuda_memory
        configure_cuda_memory()

        # Use Textual's built-in gruvbox theme
        self.theme = DEFAULT_THEME

        if self._state is not None:
            import vector_forge.ui.state as state_module
            state_module._state = self._state
        else:
            # Reset to clean state
            reset_state()

        if self._demo_mode:
            # Load demo data
            _populate_demo_state(get_state())
        else:
            # Initialize services and load real sessions
            self._init_services()
            # Start watching for new events
            self._start_watcher()

        self.push_screen("dashboard")

    def _init_services(self) -> None:
        """Initialize services and load existing sessions."""
        try:
            from vector_forge.services import (
                SessionService,
                TaskExecutor,
                ExtractionRunner,
            )
            from vector_forge.services.synchronizer import SessionLoader

            self._session_service = SessionService()
            self._session_loader = SessionLoader(self._session_service)
            self._task_executor = TaskExecutor(self._session_service)
            self._extraction_runner = ExtractionRunner(
                self._session_service,
                self._task_executor,
            )

            # Load existing sessions from storage (initial load)
            loaded = self._session_loader.load_into(get_state())
            logger.info(f"Loaded {loaded} existing sessions")

            # Initialize watcher with sessions base path
            sessions_path = self._session_service._storage.base_path
            self._watcher = EventStreamWatcher(sessions_path)

            # Add all existing sessions to watch
            for session_id in get_state().extractions.keys():
                self._watcher.add_session(session_id)

        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            # App still works, just without persistence
            state = get_state()
            state.logs.append(LogEntry(
                timestamp=time.time(),
                source="system",
                message=f"Storage unavailable: {e}",
                level="warning",
            ))

    def _start_watcher(self) -> None:
        """Start the JSONL watcher worker."""
        if self._watcher is None:
            return

        self._watching = True
        self._watch_events()

    @work(thread=True, exclusive=True, name="event_watcher")
    def _watch_events(self) -> None:
        """Worker thread that watches JSONL files for new events.

        Uses Textual's @work decorator with thread=True for background execution.
        Posts NewEvents messages (thread-safe) for UI updates.
        """
        worker = get_current_worker()

        while self._watching and not worker.is_cancelled:
            try:
                # Discover new sessions
                if self._watcher:
                    self._watcher.sync_with_disk()

                    # Check all watched sessions for new events
                    for session_id, events in self._watcher.check_all():
                        if events:
                            # post_message is thread-safe!
                            self.post_message(NewEvents(session_id, events))

            except Exception as e:
                logger.error(f"Watcher error: {e}")

            # Sleep before next poll
            time.sleep(WATCHER_POLL_INTERVAL)

    def on_new_events(self, message: NewEvents) -> None:
        """Handle new events from JSONL watcher.

        This runs on the main thread, so UI updates are safe.
        Uses targeted updates for performance - only refreshes affected widgets.

        Args:
            message: NewEvents message with session_id and events.
        """
        state = get_state()

        # Track what changed for targeted refresh
        affected_agents = set()
        log_events = []
        session_created = False
        session_completed = False

        # Track agents before processing to detect new ones
        extraction = state.extractions.get(message.session_id)
        agents_before = set(extraction.agents.keys()) if extraction else set()

        for event in message.events:
            try:
                apply_event_to_state(state, message.session_id, event)

                # Track affected components for targeted refresh
                event_type = event.event_type
                payload = event.payload

                # Session lifecycle events
                if event_type == "session.started":
                    session_created = True
                elif event_type == "session.completed":
                    session_completed = True

                # Agent-affecting events
                if event_type in (
                    "optimization.started", "optimization.progress",
                    "optimization.completed", "seed.assigned",
                    "llm.request", "llm.response",
                ):
                    sample_idx = payload.get("sample_idx")
                    if sample_idx is not None:
                        affected_agents.add(f"{message.session_id}_sample_{sample_idx}")
                    source = event.source
                    if source:
                        affected_agents.add(f"{message.session_id}_{source}")

                # Log-affecting events (most events generate logs)
                if event_type != "llm.chunk":
                    log_events.append(event)

            except Exception as e:
                logger.error(f"Error applying event {event.event_type}: {e}")

        # Check if new agents were created
        extraction = state.extractions.get(message.session_id)
        agents_after = set(extraction.agents.keys()) if extraction else set()
        new_agents_created = len(agents_after) > len(agents_before)

        # Targeted refresh based on current screen and what changed
        self._targeted_refresh(
            message.session_id,
            affected_agents,
            bool(log_events),
            session_created,
            session_completed,
            new_agents_created,
        )

    def _targeted_refresh(
        self,
        session_id: str,
        affected_agents: set,
        has_logs: bool,
        session_created: bool = False,
        session_completed: bool = False,
        new_agents_created: bool = False,
    ) -> None:
        """Refresh only affected widgets based on what changed.

        Args:
            session_id: The session that was updated.
            affected_agents: Set of agent IDs that were updated.
            has_logs: Whether new log entries were added.
            session_created: Whether a new session was started.
            session_completed: Whether a session completed.
            new_agents_created: Whether new agents were added.
        """
        try:
            screen = self.screen
            screen_name = type(screen).__name__

            if screen_name == "DashboardScreen":
                if session_created or session_completed or has_logs:
                    # Full sync needed for new/completed sessions or new logs
                    if hasattr(screen, '_sync'):
                        screen._sync()
                else:
                    # Just update existing task card
                    self._refresh_dashboard_session(session_id)

            elif screen_name == "SamplesScreen":
                # Always do full sync for samples screen
                # Targeted updates were unreliable due to widget mounting race conditions
                if hasattr(screen, '_sync'):
                    screen._sync()

            elif screen_name == "LogsScreen" and has_logs:
                # Logs: append new log entries
                if hasattr(screen, '_sync'):
                    screen._sync()

        except Exception as e:
            logger.debug(f"Targeted refresh error: {e}")

    def _refresh_dashboard_session(self, session_id: str) -> None:
        """Refresh dashboard for a specific session."""
        try:
            from vector_forge.ui.screens.dashboard import TaskCard

            for card in self.screen.query(TaskCard):
                if card.extraction.id == session_id:
                    state = get_state()
                    extraction = state.extractions.get(session_id)
                    if extraction:
                        card.update_extraction(extraction)
                    break
        except Exception:
            pass

    def switch_screen(self, screen_name: str) -> None:
        """Switch to a different screen."""
        if screen_name == "create_task":
            self.push_screen(screen_name)
        elif screen_name in ("dashboard", "samples", "logs", "chat"):
            current = getattr(self.screen, "name", None)
            if current is None:
                current = type(self.screen).__name__.lower().replace("screen", "")
            if screen_name == current:
                return
            if len(self.screen_stack) > 1:
                self.pop_screen()
            self.push_screen(screen_name)

    def action_new_task(self) -> None:
        """Handle new task action."""
        self.push_screen("create_task")

    def action_toggle_help_panel(self) -> None:
        """Toggle help panel visibility."""
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

                # Immediately create UI state for the new session (don't wait for watcher)
                extraction = ExtractionUIState(
                    id=session_id,
                    behavior_name=behavior_name,
                    behavior_description=description,
                    model=message.config.extractor_model,
                    target_model=message.config.target_model,
                    status=ExtractionStatus.RUNNING,
                    phase=Phase.INITIALIZING,
                    max_outer_iterations=message.config.num_samples,
                    started_at=time.time(),
                )
                state.add_extraction(extraction)
                # Always select the newly created session
                state.selected_id = session_id

                # Add to watcher for real-time updates
                if self._watcher:
                    self._watcher.add_session(session_id)

                logger.info(f"Created session {session_id} for {behavior_name}")

                # Start the extraction in the background
                if self._extraction_runner is not None:
                    self._extraction_runner.start_extraction(
                        session_id=session_id,
                        behavior_name=behavior_name,
                        behavior_description=description,
                        config=message.config,
                    )

            except Exception as e:
                logger.error(f"Failed to create session: {e}")
                # Fall back to in-memory
                self._create_inmemory_extraction(message, behavior_name, description)
        else:
            # No services - create in-memory only
            self._create_inmemory_extraction(message, behavior_name, description)

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
            target_model=message.config.target_model,
            status=ExtractionStatus.PENDING,
            phase=Phase.INITIALIZING,
            max_outer_iterations=message.config.num_samples,
        )

        state = get_state()
        state.add_extraction(extraction)

    def on_unmount(self) -> None:
        """Clean up on unmount."""
        self._watching = False

    @property
    def session_service(self):
        """Get the session service (may be None)."""
        return self._session_service

    @property
    def session_loader(self):
        """Get the session loader (may be None)."""
        return self._session_loader

    # Backwards compatibility
    @property
    def synchronizer(self):
        """Get the synchronizer (deprecated, use session_loader)."""
        return self._session_loader

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

    # Add sample agents with proper IDs for count_label
    extractor = AgentUIState(
        id="ext_001_sample_0",
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
        id="ext_001_sample_1",
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
        id="ext_001_sample_2",
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
        id="ext_001_sample_3",
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
