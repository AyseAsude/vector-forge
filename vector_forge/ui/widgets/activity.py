"""Activity log panel widget."""

from typing import List

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from vector_forge.ui.theme import COLORS, ICONS
from vector_forge.ui.state import ActivityEntry


class ActivityItem(Widget):
    """Single activity entry display."""

    DEFAULT_CSS = """
    ActivityItem {
        height: 1;
        layout: horizontal;
    }

    ActivityItem .activity-icon {
        width: 2;
    }

    ActivityItem .activity-message {
        width: 1fr;
        color: $text;
    }
    """

    def __init__(self, entry: ActivityEntry, **kwargs) -> None:
        super().__init__(**kwargs)
        self.entry = entry

    def compose(self) -> ComposeResult:
        # Use Rich markup for icon color based on status
        icon_colors = {
            "active": COLORS.accent,
            "success": COLORS.success,
            "error": COLORS.error,
            "waiting": COLORS.text_dim,
        }
        color = icon_colors.get(self.entry.status, COLORS.text_muted)
        yield Static(f"[{color}]{self.entry.icon}[/]", classes="activity-icon")
        yield Static(self.entry.message, classes="activity-message")


class ActivityPanel(Widget):
    """Panel displaying recent agent activity."""

    DEFAULT_CSS = """
    ActivityPanel {
        height: auto;
        max-height: 6;
        background: $surface;
        padding: 1 2;
        margin: 0 0 1 0;
    }

    ActivityPanel #activity-panel-title {
        color: $accent;
        text-style: bold;
        height: 1;
        margin-bottom: 1;
    }

    ActivityPanel #activity-content {
        height: auto;
        max-height: 4;
        scrollbar-size: 1 1;
    }

    ActivityPanel #activity-empty {
        color: $text-disabled;
        height: 1;
    }
    """

    entries: reactive[List[ActivityEntry]] = reactive(list, always_update=True, init=False)

    def compose(self) -> ComposeResult:
        yield Static("Activity", id="activity-panel-title")
        yield VerticalScroll(id="activity-content")

    def on_mount(self) -> None:
        self._update_entries()

    def watch_entries(self, entries: List[ActivityEntry]) -> None:
        if self.is_mounted:
            self._update_entries()

    def _update_entries(self) -> None:
        if not self.is_mounted:
            return
        content = self.query_one("#activity-content", VerticalScroll)

        # Remove all existing children safely
        for child in list(content.children):
            child.remove()

        if not self.entries:
            content.mount(Static("No activity yet"))
        else:
            visible_entries = self.entries[-5:]
            # No IDs - let Textual auto-generate them
            for entry in visible_entries:
                content.mount(ActivityItem(entry))
            content.scroll_end(animate=False)

    def add_entry(self, message: str, status: str = "active") -> None:
        """Add a new activity entry."""
        import time

        icon_map = {
            "active": ICONS.active,
            "success": ICONS.success,
            "error": ICONS.error,
            "waiting": ICONS.waiting,
        }

        entry = ActivityEntry(
            timestamp=time.time(),
            icon=icon_map.get(status, ICONS.active),
            message=message,
            status=status,
        )

        new_entries = list(self.entries)
        new_entries.append(entry)

        if len(new_entries) > 10:
            new_entries = new_entries[-10:]

        self.entries = new_entries

    def update_last_entry(self, status: str) -> None:
        """Update the status of the most recent entry."""
        if not self.entries:
            return

        icon_map = {
            "active": ICONS.active,
            "success": ICONS.success,
            "error": ICONS.error,
            "waiting": ICONS.waiting,
        }

        new_entries = list(self.entries)
        last_entry = new_entries[-1]
        new_entries[-1] = ActivityEntry(
            timestamp=last_entry.timestamp,
            icon=icon_map.get(status, last_entry.icon),
            message=last_entry.message,
            status=status,
        )

        self.entries = new_entries

    def clear(self) -> None:
        """Clear all activity entries."""
        self.entries = []
