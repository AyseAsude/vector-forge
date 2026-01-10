"""Logs screen - full view of event logs."""

from typing import List

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Static, Input

from vector_forge.ui.state import LogEntry, get_state
from vector_forge.ui.theme import COLORS


class LogEntryRow(Static):
    """Single log entry row."""

    DEFAULT_CSS = """
    LogEntryRow {
        height: 1;
    }

    LogEntryRow.-warning {
        color: $warning;
    }

    LogEntryRow.-error {
        color: $error;
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
            f"[{COLORS.text_muted}]{entry.source:<12}[/] "
            f"[{color}]{entry.message}[/]"
        )
        super().__init__(content, **kwargs)

        if entry.level in ("warning", "error"):
            self.add_class(f"-{entry.level}")


class LogsScreen(Screen):
    """Full-screen log viewer with filtering."""

    BINDINGS = [
        Binding("1", "switch_dashboard", "dashboard"),
        Binding("2", "switch_samples", "samples"),
        Binding("3", "noop", "logs", show=False),
        Binding("tab", "cycle_screen", "cycle", show=False),
        Binding("q", "quit", "quit"),
        Binding("n", "new_task", "new task"),
        Binding("/", "focus_filter", "filter"),
        Binding("escape", "clear_filter", "clear", show=False),
        Binding("g", "scroll_top", "top", show=False),
        Binding("G", "scroll_bottom", "bottom", show=False),
        Binding("?", "show_help", "help"),
    ]

    DEFAULT_CSS = """
    LogsScreen {
        background: $background;
    }

    LogsScreen #header {
        height: 3;
        padding: 1 2;
        background: $surface;
        border-bottom: solid $border;
    }

    LogsScreen #header-title {
        width: 1fr;
        text-style: bold;
    }

    LogsScreen #header-count {
        width: auto;
        color: $text-muted;
    }

    LogsScreen #filter-bar {
        height: 3;
        padding: 1 2;
        background: $surface;
    }

    LogsScreen #filter-label {
        width: auto;
        padding-top: 1;
        color: $text-muted;
    }

    LogsScreen #filter-input {
        width: 40;
        margin-left: 1;
    }

    LogsScreen #filter-spacer {
        width: 1fr;
    }

    LogsScreen #filter-levels {
        width: auto;
        padding-top: 1;
        color: $text-muted;
    }

    LogsScreen #log-scroll {
        height: 1fr;
        padding: 0 2;
        background: $background;
    }

    LogsScreen #log-empty {
        color: $text-muted;
        padding: 2;
        text-align: center;
    }

    LogsScreen #footer {
        dock: bottom;
        height: 1;
        background: $surface;
        padding: 0 2;
    }

    LogsScreen #footer-left {
        width: 1fr;
    }

    LogsScreen #footer-right {
        width: auto;
        color: $text-muted;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._filter_text: str = ""
        self._level_filter: str = ""

    def compose(self) -> ComposeResult:
        with Horizontal(id="header"):
            yield Static("Logs", id="header-title")
            yield Static(id="header-count")

        with Horizontal(id="filter-bar"):
            yield Static("Filter:", id="filter-label")
            yield Input(placeholder="search logs...", id="filter-input")
            yield Static(id="filter-spacer")
            yield Static("all levels", id="filter-levels")

        yield VerticalScroll(id="log-scroll")

        with Horizontal(id="footer"):
            yield Static(id="footer-left")
            yield Static("/: search  |  g/G: top/bottom  |  1: dash  |  2: samples", id="footer-right")

    def on_mount(self) -> None:
        self._sync_from_state()

        state = get_state()
        state.add_listener(self._on_state_changed)

    def on_unmount(self) -> None:
        state = get_state()
        state.remove_listener(self._on_state_changed)

    def _on_state_changed(self, state) -> None:
        self._sync_from_state()

    def _sync_from_state(self) -> None:
        state = get_state()

        # Update header count
        count = self.query_one("#header-count", Static)
        filtered = self._get_filtered_logs()
        count.update(f"{len(filtered)} / {len(state.logs)} entries")

        # Refresh logs
        self._refresh_logs(filtered)

        # Update footer
        footer_left = self.query_one("#footer-left", Static)
        if state.selected_extraction:
            ext = state.selected_extraction
            footer_left.update(f"Showing: {ext.behavior_name}")
        else:
            footer_left.update("Showing: all logs")

    def _get_filtered_logs(self) -> List[LogEntry]:
        state = get_state()
        extraction_id = state.selected_id
        logs = state.get_filtered_logs(extraction_id=extraction_id)

        if self._filter_text:
            filter_lower = self._filter_text.lower()
            logs = [
                log for log in logs
                if filter_lower in log.message.lower()
                or filter_lower in log.source.lower()
            ]

        if self._level_filter:
            logs = [log for log in logs if log.level == self._level_filter]

        return logs

    def _refresh_logs(self, logs: List[LogEntry]) -> None:
        scroll = self.query_one("#log-scroll", VerticalScroll)
        scroll.remove_children()

        if not logs:
            scroll.mount(Static("No log entries", id="log-empty"))
            return

        for entry in logs[-500:]:
            scroll.mount(LogEntryRow(entry))

        scroll.scroll_end(animate=False)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "filter-input":
            self._filter_text = event.value
            self._sync_from_state()

    def action_noop(self) -> None:
        pass

    def action_switch_dashboard(self) -> None:
        self.app.switch_screen("dashboard")

    def action_switch_samples(self) -> None:
        self.app.switch_screen("samples")

    def action_cycle_screen(self) -> None:
        self.app.switch_screen("dashboard")

    def action_focus_filter(self) -> None:
        self.query_one("#filter-input", Input).focus()

    def action_clear_filter(self) -> None:
        filter_input = self.query_one("#filter-input", Input)
        filter_input.value = ""
        self._filter_text = ""
        self._sync_from_state()
        self.focus()

    def action_scroll_top(self) -> None:
        self.query_one("#log-scroll", VerticalScroll).scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        self.query_one("#log-scroll", VerticalScroll).scroll_end(animate=False)

    def action_new_task(self) -> None:
        self.app.push_screen("create_task")

    def action_show_help(self) -> None:
        self.app.push_screen("help")

    def action_quit(self) -> None:
        self.app.exit()
