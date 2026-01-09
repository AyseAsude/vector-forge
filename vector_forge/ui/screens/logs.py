"""Full-screen logs view."""

from typing import List, Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Static, Input

from vector_forge.ui.state import LogEntry, get_state
from vector_forge.ui.widgets.status_bar import StatusBar, ScreenTab
from vector_forge.ui.widgets.extraction_selector import ExtractionSelector
from vector_forge.ui.theme import COLORS


class LogEntryRow(Static):
    """Single log entry in full view."""

    DEFAULT_CSS = """
    LogEntryRow {
        height: 1;
    }
    """

    def __init__(self, entry: LogEntry, **kwargs) -> None:
        level_colors = {
            "info": COLORS.text,
            "warning": COLORS.warning,
            "error": COLORS.error,
        }
        color = level_colors.get(entry.level, COLORS.text)

        content = (
            f"[{COLORS.text_dim}]{entry.time_str}[/] "
            f"[{COLORS.text_muted}]{entry.source:<9}[/] "
            f"[{color}]{entry.message}[/]"
        )
        super().__init__(content, **kwargs)


class LogsScreen(Screen):
    """Full-screen view of all logs with filtering.

    Features:
    - Extraction selector to filter by extraction
    - Text filter input
    - Full scrollable log view
    """

    BINDINGS = [
        Binding("1", "switch_dashboard", "dashboard"),
        Binding("2", "switch_agents", "agents"),
        Binding("3", "noop", "logs", show=False),
        Binding("tab", "cycle_screen", "cycle", show=False),
        Binding("q", "quit", "quit"),
        Binding("n", "new_task", "new task"),
        Binding("/", "focus_filter", "filter"),
        Binding("escape", "clear_filter", "clear", show=False),
        Binding("home", "scroll_top", "top", show=False),
        Binding("end", "scroll_bottom", "bottom", show=False),
        Binding("g", "scroll_top", "top", show=False),
        Binding("G", "scroll_bottom", "bottom", show=False),
        Binding("?", "show_help", "help"),
    ]

    DEFAULT_CSS = """
    LogsScreen {
        background: $background;
    }

    LogsScreen #main-content {
        height: 1fr;
    }

    LogsScreen #extraction-selector {
        height: auto;
    }

    LogsScreen #filter-bar {
        height: 1;
        padding: 0 1;
        background: $panel;
    }

    LogsScreen #filter-label {
        width: auto;
        color: $text-muted;
    }

    LogsScreen #filter-input {
        width: 30;
        height: 1;
        border: none;
        background: $surface;
        padding: 0 1;
        margin-left: 1;
    }

    LogsScreen #filter-spacer {
        width: 1fr;
    }

    LogsScreen #log-count {
        width: auto;
        color: $text-disabled;
    }

    LogsScreen #log-view {
        height: 1fr;
        padding: 0 1;
        background: $panel;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._filter_text: str = ""

    def compose(self) -> ComposeResult:
        with Vertical(id="main-content"):
            yield ExtractionSelector(id="extraction-selector")
            with Horizontal(id="filter-bar"):
                yield Static("Filter:", id="filter-label")
                yield Input(placeholder="search...", id="filter-input")
                yield Static(id="filter-spacer")
                yield Static("", id="log-count")
            yield VerticalScroll(id="log-view")
        yield StatusBar(id="status-bar")

    def on_mount(self) -> None:
        """Initialize screen."""
        self._sync_from_state()

        status_bar = self.query_one(StatusBar)
        status_bar.active_screen = "logs"
        status_bar.phase = "LOGS"
        status_bar.iteration = ""
        status_bar.layer = ""
        status_bar.turn = ""

        state = get_state()
        state.add_listener(self._on_state_changed)

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

        self._refresh_logs()
        self._update_status_bar()

    def _update_status_bar(self) -> None:
        """Update status bar with log count."""
        state = get_state()
        status_bar = self.query_one(StatusBar)
        filtered = self._get_filtered_logs()
        status_bar.elapsed = f"{len(filtered)}/{len(state.logs)}"

    def _refresh_logs(self) -> None:
        """Refresh the log display."""
        log_view = self.query_one("#log-view", VerticalScroll)

        for child in list(log_view.children):
            child.remove()

        filtered = self._get_filtered_logs()

        count_widget = self.query_one("#log-count", Static)
        count_widget.update(f"[{COLORS.text_dim}]{len(filtered)} entries[/]")

        if not filtered:
            log_view.mount(Static(f"[{COLORS.text_muted}]No log entries[/]"))
            return

        for entry in filtered[-200:]:
            log_view.mount(LogEntryRow(entry))

        log_view.scroll_end(animate=False)

    def _get_filtered_logs(self) -> List[LogEntry]:
        """Get logs with current filters applied."""
        state = get_state()

        # Filter by selected extraction
        extraction_id = state.selected_id
        logs = state.get_filtered_logs(extraction_id=extraction_id)

        # Text filter
        if self._filter_text:
            filter_lower = self._filter_text.lower()
            logs = [
                log for log in logs
                if filter_lower in log.message.lower()
                or filter_lower in log.source.lower()
            ]

        return logs

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle filter input changes."""
        if event.input.id == "filter-input":
            self._filter_text = event.value
            self._refresh_logs()
            self._update_status_bar()

    def on_screen_tab_clicked(self, event: ScreenTab.Clicked) -> None:
        """Handle status bar tab clicks."""
        event.stop()
        if event.screen_name != "logs":
            self.app.switch_screen(event.screen_name)

    def on_extraction_selector_extraction_changed(
        self,
        message: ExtractionSelector.ExtractionChanged,
    ) -> None:
        """Handle extraction selection change."""
        state = get_state()
        state.select_extraction(message.extraction_id)
        message.stop()

    def action_noop(self) -> None:
        """No operation - already on this screen."""
        pass

    def action_switch_dashboard(self) -> None:
        """Switch to dashboard view."""
        self.app.switch_screen("dashboard")

    def action_switch_agents(self) -> None:
        """Switch to agents view."""
        self.app.switch_screen("agents")

    def action_cycle_screen(self) -> None:
        """Cycle to next screen (Tab key)."""
        self.app.switch_screen("dashboard")

    def action_focus_filter(self) -> None:
        """Focus the filter input."""
        filter_input = self.query_one("#filter-input", Input)
        filter_input.focus()

    def action_clear_filter(self) -> None:
        """Clear the filter and unfocus."""
        filter_input = self.query_one("#filter-input", Input)
        filter_input.value = ""
        self._filter_text = ""
        self._refresh_logs()
        self._update_status_bar()
        self.focus()

    def action_scroll_top(self) -> None:
        """Scroll to top of logs."""
        log_view = self.query_one("#log-view", VerticalScroll)
        log_view.scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        """Scroll to bottom of logs."""
        log_view = self.query_one("#log-view", VerticalScroll)
        log_view.scroll_end(animate=False)

    def action_show_help(self) -> None:
        """Show help modal."""
        self.app.push_screen("help")

    def action_new_task(self) -> None:
        """Open task creation screen."""
        self.app.push_screen("create_task")

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()
