"""Full-screen logs view."""

from typing import List, Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Static, Input, Select

from vector_forge.ui.state import LogEntry, ExtractionStatus, get_state
from vector_forge.ui.widgets.status_bar import StatusBar, ScreenTab
from vector_forge.ui.theme import COLORS


class LogEntryRow(Static):
    """Single log entry in full view."""

    DEFAULT_CSS = """
    LogEntryRow {
        height: 1;
        layout: horizontal;
    }
    """

    def __init__(self, entry: LogEntry, **kwargs) -> None:
        # Format the entry as a single line with gruvbox colors
        level_colors = {
            "info": COLORS.text,
            "warning": COLORS.warning,
            "error": COLORS.error,
        }
        color = level_colors.get(entry.level, COLORS.text)

        content = (
            f"[{COLORS.text_dim}]{entry.time_str}[/]  "
            f"[{COLORS.text_muted}]{entry.source:<10}[/]  "
            f"[{color}]{entry.message}[/]"
        )
        super().__init__(content, **kwargs)


class LogsScreen(Screen):
    """Full-screen view of all logs with advanced filtering.

    Provides:
    - Text filter input
    - Source filter dropdown
    - Level filter dropdown
    - Full scrollable log view
    - Tmux-style status bar at bottom
    """

    BINDINGS = [
        Binding("1", "switch_dashboard", "dashboard"),
        Binding("2", "switch_parallel", "parallel"),
        Binding("3", "noop", "logs", show=False),
        Binding("tab", "cycle_screen", "cycle", show=False),
        Binding("q", "quit", "quit"),
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

    LogsScreen #filter-bar {
        height: 2;
        padding: 0 2;
        background: $surface;
        layout: horizontal;
    }

    LogsScreen #filter-label {
        width: auto;
        color: $text-muted;
        padding: 0 1 0 0;
    }

    LogsScreen #filter-input {
        width: 30;
        height: 1;
        border: none;
        background: $panel;
        padding: 0 1;
        color: $text;
    }

    LogsScreen #source-label {
        width: auto;
        color: $text-muted;
        margin-left: 2;
        padding: 0 1 0 0;
    }

    LogsScreen #source-select {
        width: 15;
        height: 1;
    }

    LogsScreen #level-label {
        width: auto;
        color: $text-muted;
        margin-left: 2;
        padding: 0 1 0 0;
    }

    LogsScreen #level-select {
        width: 12;
        height: 1;
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
        padding: 1 2;
        scrollbar-size: 1 1;
        background: $surface;
    }

    LogsScreen #log-empty {
        color: $text-disabled;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._filter_text: str = ""
        self._source_filter: Optional[str] = None
        self._level_filter: Optional[str] = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="filter-bar"):
            yield Static("Filter:", id="filter-label")
            yield Input(placeholder="Search logs...", id="filter-input")
            yield Static("Source:", id="source-label")
            yield Select(
                options=[("All", None)] + self._get_source_options(),
                value=None,
                id="source-select",
            )
            yield Static("Level:", id="level-label")
            yield Select(
                options=[
                    ("All", None),
                    ("Info", "info"),
                    ("Warning", "warning"),
                    ("Error", "error"),
                ],
                value=None,
                id="level-select",
            )
            yield Static(id="filter-spacer")
            yield Static(id="log-count")
        yield VerticalScroll(id="log-view")
        yield StatusBar(id="status-bar")

    def on_mount(self) -> None:
        """Initialize screen."""
        self._sync_from_state()

        # Set active screen on status bar
        status_bar = self.query_one(StatusBar)
        status_bar.active_screen = "logs"
        status_bar.phase = "LOGS"
        status_bar.iteration = ""
        status_bar.layer = ""
        status_bar.turn = ""

        # Register for state updates
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
        self._refresh_logs()
        self._update_source_options()
        self._update_status_bar()

    def _update_status_bar(self) -> None:
        """Update status bar with log count."""
        state = get_state()
        status_bar = self.query_one(StatusBar)
        filtered = self._get_filtered_logs()
        status_bar.elapsed = f"{len(filtered)}/{len(state.logs)}"

    def _get_source_options(self) -> List[tuple]:
        """Get unique source values for filter dropdown."""
        state = get_state()
        sources = sorted(set(log.source for log in state.logs))
        return [(src, src) for src in sources]

    def _update_source_options(self) -> None:
        """Update source dropdown options."""
        source_select = self.query_one("#source-select", Select)
        current_value = source_select.value
        options = [("All", None)] + self._get_source_options()
        source_select.set_options(options)
        # Restore selection if still valid
        if any(opt[1] == current_value for opt in options):
            source_select.value = current_value

    def _refresh_logs(self) -> None:
        """Refresh the log display."""
        log_view = self.query_one("#log-view", VerticalScroll)

        # Remove all children safely (avoid async race condition)
        for child in list(log_view.children):
            child.remove()

        filtered = self._get_filtered_logs()

        # Update count
        count_widget = self.query_one("#log-count", Static)
        count_widget.update(f"{len(filtered)} entries")

        if not filtered:
            log_view.mount(Static("No log entries match filter"))
            return

        # Limit to last 100 for performance, no IDs
        for entry in filtered[-100:]:
            log_view.mount(LogEntryRow(entry))

        # Scroll to bottom
        log_view.scroll_end(animate=False)

    def _get_filtered_logs(self) -> List[LogEntry]:
        """Get logs with current filters applied."""
        state = get_state()
        logs = state.logs

        # Text filter
        if self._filter_text:
            filter_lower = self._filter_text.lower()
            logs = [
                log for log in logs
                if filter_lower in log.message.lower()
                or filter_lower in log.source.lower()
            ]

        # Source filter
        if self._source_filter:
            logs = [log for log in logs if log.source == self._source_filter]

        # Level filter
        if self._level_filter:
            logs = [log for log in logs if log.level == self._level_filter]

        return logs

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle filter input changes."""
        if event.input.id == "filter-input":
            self._filter_text = event.value
            self._refresh_logs()
            self._update_status_bar()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle dropdown selection changes."""
        if event.select.id == "source-select":
            self._source_filter = event.value
            self._refresh_logs()
            self._update_status_bar()
        elif event.select.id == "level-select":
            self._level_filter = event.value
            self._refresh_logs()
            self._update_status_bar()

    def on_screen_tab_clicked(self, event: ScreenTab.Clicked) -> None:
        """Handle status bar tab clicks."""
        event.stop()
        if event.screen_name != "logs":
            self.app.switch_screen(event.screen_name)

    def action_noop(self) -> None:
        """No operation - already on this screen."""
        pass

    def action_switch_dashboard(self) -> None:
        """Switch to dashboard view."""
        self.app.switch_screen("dashboard")

    def action_switch_parallel(self) -> None:
        """Switch to parallel view."""
        self.app.switch_screen("parallel")

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

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()
