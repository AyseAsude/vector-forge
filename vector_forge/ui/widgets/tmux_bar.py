"""Tmux-style status bar component shared across screens."""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static

from vector_forge.ui.state import get_state


PL = ""


class TmuxBar(Widget):
    """Tmux-style status bar with clickable tabs."""

    DEFAULT_CSS = """
    TmuxBar {
        dock: bottom;
        height: 1;
        background: $surface;
        layout: horizontal;
    }

    TmuxBar .bar-tabs {
        width: auto;
        height: 1;
    }

    TmuxBar .bar-center {
        width: 1fr;
        height: 1;
        text-align: center;
        color: $foreground-muted;
    }

    TmuxBar .bar-right {
        width: auto;
        height: 1;
        color: $foreground-muted;
        padding: 0 1;
    }
    """

    def __init__(self, active_screen: str = "dashboard", **kwargs) -> None:
        super().__init__(**kwargs)
        self._active = active_screen

    def compose(self) -> ComposeResult:
        yield Static(id="bar-tabs", classes="bar-tabs")
        yield Static(id="bar-center", classes="bar-center")
        yield Static(id="bar-right", classes="bar-right")

    def on_mount(self) -> None:
        self._update_tabs()
        self._update_info()

    def on_click(self, event) -> None:
        # Check which tab was clicked based on x position
        # Tab widths: " tasks " (7), " samples " (9), " logs " (6), " chat " (6)
        # Positions: tasks 0-7, samples 7-16, logs 16-22, chat 22-28
        if event.y == 0:
            x = event.x
            if x < 7:
                self._switch_to("dashboard")
            elif x < 16:
                self._switch_to("samples")
            elif x < 22:
                self._switch_to("logs")
            else:
                self._switch_to("chat")

    def _switch_to(self, screen: str) -> None:
        if screen != self._active:
            self.app.switch_screen(screen)

    def set_active(self, screen: str) -> None:
        self._active = screen
        self._update_tabs()

    def _update_tabs(self) -> None:
        tabs = self.query_one("#bar-tabs", Static)

        screens = [
            ("tasks", "dashboard"),
            ("samples", "samples"),
            ("logs", "logs"),
            ("chat", "chat"),
        ]

        parts = []
        for name, screen in screens:
            if screen == self._active:
                parts.append(f"[$background on $accent] {name} [/][$accent]{PL}[/]")
            else:
                parts.append(f" [$foreground-muted]{name}[/] ")

        tabs.update("".join(parts))

    def _update_info(self) -> None:
        state = get_state()

        center = self.query_one("#bar-center", Static)
        if state.selected_extraction:
            ext = state.selected_extraction
            center.update(f"[$accent]▸[/] {ext.behavior_name}")
        else:
            center.update("")

        right = self.query_one("#bar-right", Static)
        right.update(f"{state.running_count} running │ {state.complete_count}/{state.total_count} done")

    def refresh_info(self) -> None:
        if self.is_mounted:
            self._update_info()
