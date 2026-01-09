"""Dashboard screen for single extraction view."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen

from vector_forge.ui.state import (
    ExtractionUIState,
    ExtractionStatus,
    Phase,
    get_state,
)
from vector_forge.ui.widgets.status_bar import StatusBar, ScreenTab
from vector_forge.ui.widgets.target import TargetSection
from vector_forge.ui.widgets.progress import ProgressSection
from vector_forge.ui.widgets.data_panel import DataPanel
from vector_forge.ui.widgets.eval_panel import EvaluationPanel
from vector_forge.ui.widgets.activity import ActivityPanel
from vector_forge.ui.widgets.log_panel import LogPanel


class DashboardScreen(Screen):
    """Main dashboard screen showing single extraction progress.

    Clean, minimal design with:
    - Target behavior at top
    - Progress bar
    - Metrics panels side by side
    - Activity log
    - Event log
    - Tmux-style status bar at bottom
    """

    BINDINGS = [
        Binding("1", "noop", "dashboard", show=False),
        Binding("2", "switch_parallel", "parallel"),
        Binding("3", "switch_logs", "logs"),
        Binding("tab", "cycle_screen", "cycle", show=False),
        Binding("q", "quit", "quit"),
        Binding("p", "toggle_pause", "pause"),
        Binding("l", "toggle_logs", "logs"),
        Binding("f", "focus_filter", "filter"),
        Binding("?", "show_help", "help"),
    ]

    DEFAULT_CSS = """
    DashboardScreen {
        background: $background;
    }

    DashboardScreen #main-content {
        height: 1fr;
        padding: 1 2;
    }

    DashboardScreen #metrics-row {
        height: auto;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="main-content"):
            yield TargetSection(id="target-section")
            yield ProgressSection(id="progress-section")
            with Horizontal(id="metrics-row"):
                yield DataPanel(id="data-panel")
                yield EvaluationPanel(id="eval-panel")
            yield ActivityPanel(id="activity-panel")
            yield LogPanel(id="log-panel")
        yield StatusBar(id="status-bar")

    def on_mount(self) -> None:
        """Initialize screen with current state."""
        self._sync_from_state()

        # Set active screen on status bar
        status_bar = self.query_one(StatusBar)
        status_bar.active_screen = "dashboard"

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
        extraction = state.selected_extraction

        if extraction is None:
            self._show_empty_state()
            return

        self._update_target(extraction)
        self._update_progress(extraction)
        self._update_status_bar(extraction)
        self._update_data_panel(extraction)
        self._update_eval_panel(extraction)
        self._update_activity(extraction)
        self._update_logs()

    def _show_empty_state(self) -> None:
        """Show empty state when no extraction is selected."""
        target = self.query_one(TargetSection)
        target.set_behavior("No extraction", "Start an extraction to see progress")

        status_bar = self.query_one(StatusBar)
        status_bar.phase = ""
        status_bar.iteration = ""
        status_bar.layer = ""
        status_bar.turn = ""
        status_bar.elapsed = "00:00"

    def _update_target(self, extraction: ExtractionUIState) -> None:
        """Update target section."""
        target = self.query_one(TargetSection)
        target.set_behavior(
            extraction.behavior_name,
            extraction.behavior_description,
        )

    def _update_progress(self, extraction: ExtractionUIState) -> None:
        """Update progress section."""
        progress = self.query_one(ProgressSection)
        progress.set_progress(
            progress=extraction.progress,
            phase=extraction.phase,
            outer_iter=extraction.outer_iteration,
            max_outer=extraction.max_outer_iterations,
            inner_turn=extraction.inner_turn,
            max_inner=extraction.max_inner_turns,
            current_layer=extraction.current_layer,
        )

    def _update_status_bar(self, extraction: ExtractionUIState) -> None:
        """Update status bar with extraction state."""
        status_bar = self.query_one(StatusBar)
        status_bar.set_extraction_state(
            phase=extraction.phase,
            outer_iter=extraction.outer_iteration,
            max_outer=extraction.max_outer_iterations,
            inner_turn=extraction.inner_turn,
            max_inner=extraction.max_inner_turns,
            current_layer=extraction.current_layer,
            elapsed=extraction.elapsed_str,
            status=extraction.status,
        )

    def _update_data_panel(self, extraction: ExtractionUIState) -> None:
        """Update data panel."""
        data_panel = self.query_one(DataPanel)
        data_panel.set_metrics(extraction.datapoints)

    def _update_eval_panel(self, extraction: ExtractionUIState) -> None:
        """Update evaluation panel."""
        eval_panel = self.query_one(EvaluationPanel)
        eval_panel.set_metrics(extraction.evaluation)

    def _update_activity(self, extraction: ExtractionUIState) -> None:
        """Update activity panel."""
        activity = self.query_one(ActivityPanel)
        activity.entries = extraction.activity

    def _update_logs(self) -> None:
        """Update log panel."""
        state = get_state()
        log_panel = self.query_one(LogPanel)
        log_panel.entries = state.get_filtered_logs()

    def _on_timer_tick(self) -> None:
        """Update elapsed time display."""
        state = get_state()
        extraction = state.selected_extraction
        if extraction and extraction.status == ExtractionStatus.RUNNING:
            status_bar = self.query_one(StatusBar)
            status_bar.elapsed = extraction.elapsed_str

    def on_screen_tab_clicked(self, event: ScreenTab.Clicked) -> None:
        """Handle status bar tab clicks."""
        event.stop()
        if event.screen_name != "dashboard":
            self.app.switch_screen(event.screen_name)

    def action_noop(self) -> None:
        """No operation - already on this screen."""
        pass

    def action_switch_parallel(self) -> None:
        """Switch to parallel view."""
        self.app.switch_screen("parallel")

    def action_switch_logs(self) -> None:
        """Switch to logs view."""
        self.app.switch_screen("logs")

    def action_cycle_screen(self) -> None:
        """Cycle to next screen (Tab key)."""
        self.app.switch_screen("parallel")

    def action_toggle_pause(self) -> None:
        """Toggle pause state."""
        state = get_state()
        extraction = state.selected_extraction
        if extraction:
            if extraction.status == ExtractionStatus.RUNNING:
                extraction.status = ExtractionStatus.PAUSED
            elif extraction.status == ExtractionStatus.PAUSED:
                extraction.status = ExtractionStatus.RUNNING
            self._sync_from_state()

    def action_toggle_logs(self) -> None:
        """Toggle log panel collapsed state."""
        log_panel = self.query_one(LogPanel)
        log_panel.toggle_collapsed()

    def action_focus_filter(self) -> None:
        """Focus the log filter input."""
        log_panel = self.query_one(LogPanel)
        log_panel.focus_filter()

    def action_show_help(self) -> None:
        """Show help modal."""
        self.app.push_screen("help")

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()
