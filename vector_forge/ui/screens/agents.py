"""Agents screen - split panel view of parallel agents."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen

from vector_forge.ui.state import (
    ExtractionStatus,
    get_state,
)
from vector_forge.ui.widgets.status_bar import StatusBar, ScreenTab
from vector_forge.ui.widgets.extraction_selector import ExtractionSelector
from vector_forge.ui.widgets.agents_list import AgentsList
from vector_forge.ui.widgets.agent_inspector import AgentInspector


class AgentsScreen(Screen):
    """Screen for viewing parallel agents within the current extraction.

    Split panel layout:
    - Left: List of all agents for the extraction
    - Right: Inspector showing messages for selected agent
    """

    BINDINGS = [
        Binding("1", "switch_dashboard", "dashboard"),
        Binding("2", "noop", "agents", show=False),
        Binding("3", "switch_logs", "logs"),
        Binding("tab", "cycle_screen", "cycle", show=False),
        Binding("q", "quit", "quit"),
        Binding("up", "select_previous", "up", show=False),
        Binding("down", "select_next", "down", show=False),
        Binding("k", "select_previous", "up", show=False),
        Binding("j", "select_next", "down", show=False),
        Binding("?", "show_help", "help"),
    ]

    DEFAULT_CSS = """
    AgentsScreen {
        background: $background;
    }

    AgentsScreen #main-content {
        height: 1fr;
    }

    AgentsScreen #extraction-selector {
        height: auto;
    }

    AgentsScreen #split-container {
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="main-content"):
            yield ExtractionSelector(id="extraction-selector")
            with Horizontal(id="split-container"):
                yield AgentsList(id="agents-list")
                yield AgentInspector(id="agent-inspector")
        yield StatusBar(id="status-bar")

    def on_mount(self) -> None:
        """Initialize screen with current state."""
        self._sync_from_state()

        # Set active screen on status bar
        status_bar = self.query_one(StatusBar)
        status_bar.active_screen = "agents"

        # Register for state updates
        state = get_state()
        state.add_listener(self._on_state_changed)

        # Start timer update interval
        self.set_interval(1.0, self._on_timer_tick)

    def on_unmount(self) -> None:
        """Clean up state listener."""
        state = get_state()
        state.remove_listener(self._on_state_changed)

    def _on_state_changed(self, state) -> None:
        """Handle state changes."""
        self._sync_from_state()

    def _sync_from_state(self) -> None:
        """Synchronize display with current state."""
        state = get_state()

        # Update extraction selector
        selector = self.query_one(ExtractionSelector)
        selector.extractions = state.extractions
        selector.selected_id = state.selected_id

        # Update status bar
        self._update_status_bar(state)

        # Update agents for current extraction
        extraction = state.selected_extraction
        agents_list = self.query_one(AgentsList)
        inspector = self.query_one(AgentInspector)

        if extraction is None:
            agents_list.agents = {}
            agents_list.selected_id = None
            inspector.agent = None
            return

        agents_list.agents = extraction.agents
        agents_list.selected_id = extraction.selected_agent_id
        inspector.agent = extraction.selected_agent

    def _update_status_bar(self, state) -> None:
        """Update status bar with info."""
        status_bar = self.query_one(StatusBar)

        extraction = state.selected_extraction
        if extraction is None:
            status_bar.phase = ""
            status_bar.iteration = ""
            status_bar.layer = ""
            status_bar.turn = ""
            status_bar.elapsed = "00:00"
            return

        # Show agent counts in status bar
        running = extraction.running_agents_count
        total = extraction.total_agents_count

        status_bar.phase = extraction.phase.value.upper()
        status_bar.iteration = f"{running}/{total} agents"
        status_bar.layer = f"L{extraction.current_layer}" if extraction.current_layer else ""
        status_bar.turn = ""
        status_bar.elapsed = extraction.elapsed_str
        status_bar.status = extraction.status

    def _on_timer_tick(self) -> None:
        """Update time displays."""
        state = get_state()
        extraction = state.selected_extraction
        if extraction and extraction.status == ExtractionStatus.RUNNING:
            status_bar = self.query_one(StatusBar)
            status_bar.elapsed = extraction.elapsed_str

    def on_screen_tab_clicked(self, event: ScreenTab.Clicked) -> None:
        """Handle status bar tab clicks."""
        event.stop()
        if event.screen_name != "agents":
            self.app.switch_screen(event.screen_name)

    def on_extraction_selector_extraction_changed(
        self,
        message: ExtractionSelector.ExtractionChanged,
    ) -> None:
        """Handle extraction selection change."""
        state = get_state()
        state.select_extraction(message.extraction_id)
        message.stop()

    def on_agents_list_selection_changed(
        self,
        message: AgentsList.SelectionChanged,
    ) -> None:
        """Handle agent selection change."""
        state = get_state()
        extraction = state.selected_extraction
        if extraction and message.agent_id:
            extraction.select_agent(message.agent_id)
            # Update inspector
            inspector = self.query_one(AgentInspector)
            inspector.agent = extraction.selected_agent
        message.stop()

    def action_noop(self) -> None:
        """No operation - already on this screen."""
        pass

    def action_switch_dashboard(self) -> None:
        """Switch to dashboard view."""
        self.app.switch_screen("dashboard")

    def action_switch_logs(self) -> None:
        """Switch to logs view."""
        self.app.switch_screen("logs")

    def action_cycle_screen(self) -> None:
        """Cycle to next screen (Tab key)."""
        self.app.switch_screen("logs")

    def action_select_previous(self) -> None:
        """Select previous agent in list."""
        agents_list = self.query_one(AgentsList)
        agents_list.select_previous()

    def action_select_next(self) -> None:
        """Select next agent in list."""
        agents_list = self.query_one(AgentsList)
        agents_list.select_next()

    def action_show_help(self) -> None:
        """Show help modal."""
        self.app.push_screen("help")

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()
