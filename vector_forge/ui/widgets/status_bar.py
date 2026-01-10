"""Powerline-style status bar inspired by oh-my-tmux."""

from textual.app import ComposeResult
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from vector_forge.ui.state import ExtractionStatus, Phase


# Powerline characters
PL_RIGHT = ""  # Right-pointing separator (U+E0B0)
PL_RIGHT_THIN = ""  # Right-pointing thin separator (U+E0B1)
PL_LEFT = ""  # Left-pointing separator (U+E0B2)
PL_LEFT_THIN = ""  # Left-pointing thin separator (U+E0B3)


class ScreenTab(Widget):
    """Clickable screen tab with powerline styling."""

    class Clicked(Message):
        """Posted when tab is clicked."""

        def __init__(self, screen_name: str) -> None:
            self.screen_name = screen_name
            super().__init__()

    DEFAULT_CSS = """
    ScreenTab {
        width: auto;
        height: 1;
    }
    """

    is_active: reactive[bool] = reactive(False, init=False)

    def __init__(self, number: int, label: str, screen_name: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.number = number
        self.label = label
        self.screen_name = screen_name

    def render(self) -> str:
        if self.is_active:
            # Active: accent background
            return (
                f"[$accent on $background]{PL_RIGHT}[/]"
                f"[$background on $accent] {self.number}:{self.label} [/]"
                f"[$accent on $surface]{PL_RIGHT}[/]"
            )
        else:
            # Inactive: surface background
            return (
                f"[$boost on $surface]{PL_RIGHT_THIN}[/]"
                f"[$foreground-disabled on $surface] {self.number}:{self.label} [/]"
            )

    def on_click(self) -> None:
        """Handle click event."""
        self.post_message(self.Clicked(self.screen_name))


class StatusBar(Widget):
    """Powerline-style status bar like oh-my-tmux.

    Layout:
    ┌─────────────────────────────────────────────────────────────────┐
    │▌1:dash▌2:agents▌3:logs           OPT▌i2/5▌L12▌00:45│
    └─────────────────────────────────────────────────────────────────┘
    """

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        background: $surface;
        layout: horizontal;
    }

    StatusBar #status-spacer {
        width: 1fr;
        height: 1;
    }

    StatusBar #status-right {
        width: auto;
        height: 1;
    }
    """

    active_screen: reactive[str] = reactive("dashboard", init=False)
    phase: reactive[str] = reactive("", init=False)
    iteration: reactive[str] = reactive("", init=False)
    layer: reactive[str] = reactive("", init=False)
    turn: reactive[str] = reactive("", init=False)
    elapsed: reactive[str] = reactive("00:00", init=False)
    status: reactive[ExtractionStatus] = reactive(ExtractionStatus.PENDING, init=False)

    def compose(self) -> ComposeResult:
        yield ScreenTab(1, "dash", "dashboard", id="tab-dashboard")
        yield ScreenTab(2, "agents", "agents", id="tab-agents")
        yield ScreenTab(3, "logs", "logs", id="tab-logs")
        yield Static(id="status-spacer")
        yield Static(id="status-right")

    def on_mount(self) -> None:
        """Initialize status bar."""
        self._update_tabs()
        self._update_status_right()

    def watch_active_screen(self, screen: str) -> None:
        if self.is_mounted:
            self._update_tabs()

    def watch_phase(self, _: str) -> None:
        if self.is_mounted:
            self._update_status_right()

    def watch_iteration(self, _: str) -> None:
        if self.is_mounted:
            self._update_status_right()

    def watch_layer(self, _: str) -> None:
        if self.is_mounted:
            self._update_status_right()

    def watch_turn(self, _: str) -> None:
        if self.is_mounted:
            self._update_status_right()

    def watch_elapsed(self, _: str) -> None:
        if self.is_mounted:
            self._update_status_right()

    def watch_status(self, _: ExtractionStatus) -> None:
        if self.is_mounted:
            self._update_status_right()

    def _update_tabs(self) -> None:
        """Update tab active states."""
        screen_map = {
            "dashboard": "tab-dashboard",
            "agents": "tab-agents",
            "logs": "tab-logs",
        }

        for screen_name, tab_id in screen_map.items():
            try:
                tab = self.query_one(f"#{tab_id}", ScreenTab)
                tab.is_active = (screen_name == self.active_screen)
                tab.refresh()
            except Exception:
                pass

    def _update_status_right(self) -> None:
        """Update the right side status with powerline segments."""
        try:
            status_widget = self.query_one("#status-right", Static)
        except Exception:
            return

        segments = []

        # Phase segment - accent color
        if self.phase:
            segments.append(
                f"[$accent on $surface]{PL_LEFT}[/]"
                f"[$background on $accent] {self.phase} [/]"
            )

        # Iteration segment - secondary color
        if self.iteration:
            if segments:
                segments.append(f"[$secondary on $accent]{PL_LEFT}[/]")
            else:
                segments.append(f"[$secondary on $surface]{PL_LEFT}[/]")
            segments.append(f"[$background on $secondary] {self.iteration} [/]")

        # Layer segment - primary color
        if self.layer:
            prev_bg = "$secondary" if self.iteration else ("$accent" if self.phase else "$surface")
            segments.append(f"[$primary on {prev_bg}]{PL_LEFT}[/]")
            segments.append(f"[$background on $primary] {self.layer} [/]")

        # Elapsed time segment - muted surface
        if self.elapsed:
            prev_bg = "$primary" if self.layer else ("$secondary" if self.iteration else ("$accent" if self.phase else "$surface"))
            segments.append(f"[$boost on {prev_bg}]{PL_LEFT}[/]")
            segments.append(f"[$foreground on $boost] {self.elapsed} [/]")

        status_widget.update("".join(segments))

    def set_extraction_state(
        self,
        phase: Phase,
        outer_iter: int,
        max_outer: int,
        inner_turn: int,
        max_inner: int,
        current_layer: int | None,
        elapsed: str,
        status: ExtractionStatus,
    ) -> None:
        """Update all extraction state at once."""
        self.status = status
        self.phase = phase.value.upper()
        self.iteration = f"i{outer_iter}/{max_outer}"
        self.layer = f"L{current_layer}" if current_layer is not None else ""
        self.turn = ""  # Removed turn from status bar for cleaner look
        self.elapsed = elapsed

    def on_screen_tab_clicked(self, event: ScreenTab.Clicked) -> None:
        """Handle tab click - bubble up to screen."""
        # Don't stop - let it bubble to the screen
        pass
