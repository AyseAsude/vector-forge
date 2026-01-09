"""Application header widget."""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from vector_forge.ui.theme import COLORS, ICONS
from vector_forge.ui.state import ExtractionStatus


class StatusIndicator(Static):
    """Displays current pipeline status with colored indicator."""

    status: reactive[ExtractionStatus] = reactive(ExtractionStatus.PENDING)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._update_display()

    def watch_status(self, status: ExtractionStatus) -> None:
        """Update display when status changes."""
        self._update_display()

    def _update_display(self) -> None:
        """Update the indicator text and color."""
        status_config = {
            ExtractionStatus.PENDING: (ICONS.pending, "text-muted", "Pending"),
            ExtractionStatus.RUNNING: (ICONS.running, "text-accent", "Running"),
            ExtractionStatus.PAUSED: (ICONS.paused, "text-warning", "Paused"),
            ExtractionStatus.COMPLETE: (ICONS.complete, "text-success", "Complete"),
            ExtractionStatus.FAILED: (ICONS.failed, "text-error", "Failed"),
        }

        icon, css_class, label = status_config.get(
            self.status,
            (ICONS.pending, "text-muted", "Unknown"),
        )

        self.update(f"[{css_class}]{icon}[/] {label}")


class Timer(Static):
    """Displays elapsed time in MM:SS format."""

    elapsed: reactive[str] = reactive("00:00")

    def watch_elapsed(self, elapsed: str) -> None:
        """Update display when elapsed time changes."""
        self.update(elapsed)


class AppHeader(Widget):
    """Application header with title, status, and timer.

    Displays:
    - Application name
    - Current extraction status
    - Elapsed time
    """

    DEFAULT_CSS = """
    AppHeader {
        height: 3;
        background: #161b22;
        border-bottom: solid #30363d;
        padding: 0 2;
        layout: horizontal;
    }

    AppHeader #header-title {
        width: auto;
        color: #e6edf3;
        text-style: bold;
        padding: 1 0;
    }

    AppHeader #header-status {
        width: auto;
        margin-left: 2;
        padding: 1 0;
    }

    AppHeader #header-spacer {
        width: 1fr;
    }

    AppHeader #header-timer {
        width: auto;
        color: #8b949e;
        padding: 1 0;
    }
    """

    title: reactive[str] = reactive("Vector Forge", init=False)
    status: reactive[ExtractionStatus] = reactive(ExtractionStatus.PENDING, init=False)
    elapsed: reactive[str] = reactive("00:00", init=False)

    def compose(self) -> ComposeResult:
        yield Static(self.title, id="header-title")
        yield StatusIndicator(id="header-status")
        yield Static(id="header-spacer")
        yield Timer(id="header-timer")

    def watch_title(self, title: str) -> None:
        """Update title display."""
        if not self.is_mounted:
            return
        title_widget = self.query_one("#header-title", Static)
        title_widget.update(title)

    def watch_status(self, status: ExtractionStatus) -> None:
        """Update status indicator."""
        if not self.is_mounted:
            return
        indicator = self.query_one(StatusIndicator)
        indicator.status = status

    def watch_elapsed(self, elapsed: str) -> None:
        """Update timer display."""
        if not self.is_mounted:
            return
        timer = self.query_one(Timer)
        timer.elapsed = elapsed
