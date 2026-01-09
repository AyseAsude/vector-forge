"""Event log panel widget."""

from typing import List

from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static, Input

from vector_forge.ui.theme import COLORS
from vector_forge.ui.state import LogEntry


class LogEntryWidget(Widget):
    """Single log entry display."""

    DEFAULT_CSS = """
    LogEntryWidget {
        height: 1;
        layout: horizontal;
    }

    LogEntryWidget .log-timestamp {
        width: 8;
        color: $text-disabled;
    }

    LogEntryWidget .log-source {
        width: 10;
        color: $text-muted;
    }

    LogEntryWidget .log-message {
        width: 1fr;
        color: $text;
    }
    """

    def __init__(self, entry: LogEntry, **kwargs) -> None:
        super().__init__(**kwargs)
        self.entry = entry

        if entry.level == "error":
            self.add_class("log-entry-error")
        elif entry.level == "warning":
            self.add_class("log-entry-warning")

    def compose(self) -> ComposeResult:
        yield Static(self.entry.time_str, classes="log-timestamp")
        yield Static(self.entry.source, classes="log-source")

        # Use Rich markup for message color based on level
        level_colors = {
            "info": COLORS.text,
            "warning": COLORS.warning,
            "error": COLORS.error,
        }
        color = level_colors.get(self.entry.level, COLORS.text)
        yield Static(f"[{color}]{self.entry.message}[/]", classes="log-message")


class LogPanel(Widget):
    """Panel displaying event logs with filtering."""

    DEFAULT_CSS = """
    LogPanel {
        height: 1fr;
        min-height: 5;
        background: $surface;
        padding: 1 2;
        margin: 0;
    }

    LogPanel.collapsed {
        height: 3;
        min-height: 3;
    }

    LogPanel #log-header {
        height: 1;
        layout: horizontal;
        margin-bottom: 1;
    }

    LogPanel #log-title {
        width: auto;
        color: $accent;
        text-style: bold;
    }

    LogPanel #log-header-spacer {
        width: 1fr;
    }

    LogPanel #log-filter-label {
        width: auto;
        color: $text-disabled;
        margin-right: 1;
    }

    LogPanel #log-filter {
        width: 20;
        height: 1;
        border: none;
        background: $panel;
        padding: 0 1;
        color: $text;
    }

    LogPanel #log-content {
        height: 1fr;
        scrollbar-size: 1 1;
    }

    LogPanel #log-empty {
        color: $text-disabled;
        height: 1;
    }
    """

    entries: reactive[List[LogEntry]] = reactive(list, always_update=True, init=False)
    filter_text: reactive[str] = reactive("", init=False)
    collapsed: reactive[bool] = reactive(False, init=False)
    auto_scroll: reactive[bool] = reactive(True, init=False)

    def compose(self) -> ComposeResult:
        with Horizontal(id="log-header"):
            yield Static("Log", id="log-title")
            yield Static(id="log-header-spacer")
            yield Static("Filter:", id="log-filter-label")
            yield Input(placeholder="", id="log-filter")
        yield VerticalScroll(id="log-content")

    def on_mount(self) -> None:
        filter_input = self.query_one("#log-filter", Input)
        filter_input.value = self.filter_text
        self._refresh_entries()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "log-filter":
            self.filter_text = event.value
            self._refresh_entries()

    def watch_entries(self, entries: List[LogEntry]) -> None:
        if self.is_mounted:
            self._refresh_entries()

    def watch_collapsed(self, collapsed: bool) -> None:
        if collapsed:
            self.add_class("collapsed")
        else:
            self.remove_class("collapsed")

    def _refresh_entries(self) -> None:
        if not self.is_mounted:
            return
        content = self.query_one("#log-content", VerticalScroll)

        filtered = self._get_filtered_entries()
        current_count = len(list(content.query(LogEntryWidget)))
        new_count = min(len(filtered), 50)

        # Only refresh if count changed significantly or first load
        if current_count == 0 or abs(current_count - new_count) > 0:
            # Remove all children safely
            for child in list(content.children):
                child.remove()

            if not filtered:
                content.mount(Static("No log entries"))
            else:
                # No IDs - let Textual auto-generate them
                for entry in filtered[-50:]:
                    content.mount(LogEntryWidget(entry))

                if self.auto_scroll:
                    content.scroll_end(animate=False)

    def _get_filtered_entries(self) -> List[LogEntry]:
        if not self.filter_text:
            return self.entries

        filter_lower = self.filter_text.lower()
        return [
            entry for entry in self.entries
            if filter_lower in entry.message.lower()
            or filter_lower in entry.source.lower()
        ]

    def add_entry(
        self,
        source: str,
        message: str,
        level: str = "info",
        extraction_id: str | None = None,
    ) -> None:
        """Add a new log entry."""
        import time

        entry = LogEntry(
            timestamp=time.time(),
            source=source,
            message=message,
            level=level,
            extraction_id=extraction_id,
        )

        new_entries = list(self.entries)
        new_entries.append(entry)

        if len(new_entries) > 500:
            new_entries = new_entries[-500:]

        self.entries = new_entries

    def clear(self) -> None:
        """Clear all log entries."""
        self.entries = []

    def toggle_collapsed(self) -> None:
        """Toggle collapsed state."""
        self.collapsed = not self.collapsed

    def focus_filter(self) -> None:
        """Focus the filter input."""
        filter_input = self.query_one("#log-filter", Input)
        filter_input.focus()
