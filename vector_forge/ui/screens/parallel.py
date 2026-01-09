"""Parallel screen for viewing multiple extractions."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import Screen

from vector_forge.ui.state import (
    ExtractionStatus,
    Phase,
    get_state,
)
from vector_forge.ui.widgets.status_bar import StatusBar, ScreenTab
from vector_forge.ui.widgets.extractions_list import ExtractionsList
from vector_forge.ui.widgets.extraction_detail import ExtractionDetail
from vector_forge.ui.widgets.log_panel import LogPanel


class ParallelScreen(Screen):
    """Screen for viewing multiple parallel extractions.

    Displays:
    - List of all running extractions
    - Detail panel for selected extraction
    - Shared log panel
    - Tmux-style status bar at bottom
    """

    BINDINGS = [
        Binding("1", "switch_dashboard", "dashboard"),
        Binding("2", "noop", "parallel", show=False),
        Binding("3", "switch_logs", "logs"),
        Binding("tab", "cycle_screen", "cycle", show=False),
        Binding("q", "quit", "quit"),
        Binding("p", "toggle_pause_all", "pause all"),
        Binding("up", "select_previous", "up", show=False),
        Binding("down", "select_next", "down", show=False),
        Binding("k", "select_previous", "up", show=False),
        Binding("j", "select_next", "down", show=False),
        Binding("enter", "view_detail", "details", show=False),
        Binding("?", "show_help", "help"),
    ]

    DEFAULT_CSS = """
    ParallelScreen {
        background: $background;
    }

    ParallelScreen #main-content {
        height: 1fr;
        padding: 1 2;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="main-content"):
            yield ExtractionsList(id="extractions-list")
            yield ExtractionDetail(id="extraction-detail")
            yield LogPanel(id="log-panel")
        yield StatusBar(id="status-bar")

    def on_mount(self) -> None:
        """Initialize screen with current state."""
        self._sync_from_state()

        # Set active screen on status bar
        status_bar = self.query_one(StatusBar)
        status_bar.active_screen = "parallel"

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

        self._update_status_bar(state)
        self._update_extractions_list(state)
        self._update_detail(state)
        self._update_logs(state)

    def _update_status_bar(self, state) -> None:
        """Update status bar with aggregate info."""
        status_bar = self.query_one(StatusBar)

        running = state.running_count
        complete = state.complete_count

        # Calculate total elapsed time
        total_elapsed = 0.0
        for extraction in state.extractions.values():
            total_elapsed = max(total_elapsed, extraction.elapsed_seconds)

        minutes = int(total_elapsed) // 60
        seconds = int(total_elapsed) % 60
        elapsed_str = f"{minutes:02d}:{seconds:02d}"

        # Show aggregate status
        if running > 0:
            status_bar.phase = f"{running}RUN"
            status_bar.status = ExtractionStatus.RUNNING
        else:
            status_bar.phase = f"{complete}DONE"
            status_bar.status = ExtractionStatus.COMPLETE

        status_bar.iteration = f"{state.total_count}tot"
        status_bar.layer = ""
        status_bar.turn = ""
        status_bar.elapsed = elapsed_str

    def _update_extractions_list(self, state) -> None:
        """Update extractions list."""
        ext_list = self.query_one(ExtractionsList)
        ext_list.extractions = state.extractions
        ext_list.selected_id = state.selected_id

    def _update_detail(self, state) -> None:
        """Update detail panel."""
        detail = self.query_one(ExtractionDetail)
        detail.set_extraction(state.selected_extraction)

    def _update_logs(self, state) -> None:
        """Update log panel."""
        log_panel = self.query_one(LogPanel)
        log_panel.entries = state.get_filtered_logs()

    def _on_timer_tick(self) -> None:
        """Update elapsed time displays."""
        state = get_state()
        self._update_status_bar(state)

        # Update list items (they track their own times)
        ext_list = self.query_one(ExtractionsList)
        ext_list.extractions = state.extractions

    def on_screen_tab_clicked(self, event: ScreenTab.Clicked) -> None:
        """Handle status bar tab clicks."""
        event.stop()
        if event.screen_name != "parallel":
            self.app.switch_screen(event.screen_name)

    def on_extractions_list_selection_changed(
        self,
        message: ExtractionsList.SelectionChanged,
    ) -> None:
        """Handle extraction selection changes."""
        state = get_state()
        if message.extraction_id:
            state.select_extraction(message.extraction_id)
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

    def action_toggle_pause_all(self) -> None:
        """Toggle pause state for all extractions."""
        state = get_state()

        # Determine if we should pause or resume
        any_running = any(
            e.status == ExtractionStatus.RUNNING
            for e in state.extractions.values()
        )

        new_status = (
            ExtractionStatus.PAUSED if any_running else ExtractionStatus.RUNNING
        )

        for extraction in state.extractions.values():
            if extraction.status in (
                ExtractionStatus.RUNNING,
                ExtractionStatus.PAUSED,
            ):
                extraction.status = new_status

        self._sync_from_state()

    def action_select_previous(self) -> None:
        """Select previous extraction in list."""
        ext_list = self.query_one(ExtractionsList)
        ext_list.select_previous()

    def action_select_next(self) -> None:
        """Select next extraction in list."""
        ext_list = self.query_one(ExtractionsList)
        ext_list.select_next()

    def action_view_detail(self) -> None:
        """View selected extraction in dashboard."""
        state = get_state()
        if state.selected_id:
            self.app.switch_screen("dashboard")

    def action_show_help(self) -> None:
        """Show help modal."""
        self.app.push_screen("help")

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()
