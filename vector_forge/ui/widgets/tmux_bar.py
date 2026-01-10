"""Tmux-style status bar component shared across screens."""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static

from vector_forge.ui.theme import COLORS
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
        color: $text-muted;
    }

    TmuxBar .bar-right {
        width: auto;
        height: 1;
        color: $text-muted;
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
        tabs = self.query_one("#bar-tabs", Static)
        if event.y == 0:
            # Calculate rough positions
            x = event.x
            # "tasks" is ~7 chars, "samples" is ~9 chars, "logs" is ~6 chars
            if x < 8:
                self._switch_to("dashboard")
            elif x < 18:
                self._switch_to("samples")
            elif x < 25:
                self._switch_to("logs")

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
        ]

        parts = []
        for name, screen in screens:
            if screen == self._active:
                parts.append(f"[{COLORS.bg} on {COLORS.accent}] {name} [/][{COLORS.accent}]{PL}[/]")
            else:
                parts.append(f" [{COLORS.text_muted}]{name}[/] ")

        tabs.update("".join(parts))

    def _update_info(self) -> None:
        state = get_state()

        center = self.query_one("#bar-center", Static)
        if state.selected_extraction:
            ext = state.selected_extraction
            center.update(f"[{COLORS.accent}]▸[/] {ext.behavior_name}")
        else:
            center.update("")

        right = self.query_one("#bar-right", Static)
        right.update(f"{state.running_count} running │ {state.complete_count}/{state.total_count} done")

    def refresh_info(self) -> None:
        if self.is_mounted:
            self._update_info()
