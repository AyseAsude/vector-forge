"""Log panel widget - displays event logs with filtering and scrolling."""

from typing import List

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static, Input

from vector_forge.ui.state import LogEntry


class LogEntryDisplay(Widget):
    """Single log entry display."""

    DEFAULT_CSS = """
    LogEntryDisplay {
        height: 1;
        layout: horizontal;
    }

    LogEntryDisplay .log-time {
        width: 9;
        color: $foreground-disabled;
    }

    LogEntryDisplay .log-source {
        width: 10;
        color: $foreground-muted;
    }

    LogEntryDisplay .log-msg {
        width: 1fr;
        color: $foreground;
    }

    LogEntryDisplay.-warning .log-msg {
        color: $warning;
    }

    LogEntryDisplay.-error .log-msg {
        color: $error;
    }
    """

    def __init__(self, entry: LogEntry, **kwargs) -> None:
        super().__init__(**kwargs)
        self.entry = entry

    def compose(self) -> ComposeResult:
        yield Static(self.entry.time_str, classes="log-time")
        yield Static(self.entry.source[:9], classes="log-source")
        yield Static(self.entry.message, classes="log-msg")

    def on_mount(self) -> None:
        if self.entry.level == "warning":
            self.add_class("-warning")
        elif self.entry.level == "error":
            self.add_class("-error")


class LogPanel(Widget):
    """Panel displaying event logs with filtering and scrolling."""

    DEFAULT_CSS = """
    LogPanel {
        height: 1fr;
        min-height: 6;
        background: $panel;
    }

    LogPanel #log-header {
        height: 2;
        padding: 0 1;
        background: $panel;
        border-bottom: solid $surface;
    }

    LogPanel #log-title-row {
        height: 1;
    }

    LogPanel #log-title {
        width: auto;
        color: $foreground;
    }

    LogPanel #log-count {
        width: 1fr;
        text-align: right;
        color: $foreground-disabled;
    }

    LogPanel #log-filter-row {
        height: 1;
        layout: horizontal;
    }

    LogPanel #filter-label {
        width: auto;
        color: $foreground-muted;
        margin-right: 1;
    }

    LogPanel #log-filter {
        width: 20;
        height: 1;
        border: none;
        background: $surface;
        padding: 0 1;
    }

    LogPanel #source-filter {
        width: auto;
        margin-left: 2;
        color: $foreground-disabled;
    }

    LogPanel #level-filter {
        width: auto;
        margin-left: 2;
        color: $foreground-disabled;
    }

    LogPanel #log-scroll {
        height: 1fr;
        padding: 0 1;
    }

    LogPanel .log-empty-msg {
        color: $foreground-muted;
        padding: 1;
    }
    """

    entries: reactive[List[LogEntry]] = reactive(list, always_update=True, init=False)
    filter_text: reactive[str] = reactive("", init=False)

    def compose(self) -> ComposeResult:
        with Widget(id="log-header"):
            with Horizontal(id="log-title-row"):
                yield Static("Logs", id="log-title")
                yield Static("", id="log-count")
            with Horizontal(id="log-filter-row"):
                yield Static("Filter:", id="filter-label")
                yield Input(placeholder="search...", id="log-filter")
                yield Static("[all sources]", id="source-filter")
                yield Static("[all levels]", id="level-filter")
        yield VerticalScroll(id="log-scroll")

    def on_mount(self) -> None:
        self._refresh_logs()

    def watch_entries(self, entries: List[LogEntry]) -> None:
        if self.is_mounted:
            self._refresh_logs()

    def watch_filter_text(self, filter_text: str) -> None:
        if self.is_mounted:
            self._refresh_logs()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle filter input changes."""
        if event.input.id == "log-filter":
            self.filter_text = event.value
            event.stop()

    def _refresh_logs(self) -> None:
        """Refresh the log display."""
        scroll = self.query_one("#log-scroll", VerticalScroll)
        count_widget = self.query_one("#log-count", Static)

        # Clear all children synchronously
        scroll.remove_children()

        # Filter entries
        filtered = list(self.entries) if self.entries else []
        if self.filter_text:
            filter_lower = self.filter_text.lower()
            filtered = [
                e for e in filtered
                if filter_lower in e.message.lower()
                or filter_lower in e.source.lower()
            ]

        # Update count
        total = len(self.entries) if self.entries else 0
        shown = len(filtered)
        if total == shown:
            count_widget.update(f"[$foreground-disabled]{total} entries[/]")
        else:
            count_widget.update(f"[$foreground-disabled]{shown}/{total} entries[/]")

        # Show entries or empty message
        if not filtered:
            scroll.mount(
                Static(
                    "[$foreground-muted]No log entries[/]",
                    classes="log-empty-msg",
                )
            )
            return

        # Show last 100 entries (most recent at bottom)
        for entry in filtered[-100:]:
            scroll.mount(LogEntryDisplay(entry))

        # Scroll to bottom
        scroll.scroll_end(animate=False)

    def focus_filter(self) -> None:
        """Focus the filter input."""
        filter_input = self.query_one("#log-filter", Input)
        filter_input.focus()

    def set_entries(self, entries: List[LogEntry]) -> None:
        """Set log entries."""
        self.entries = entries
