"""Help modal screen."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from vector_forge.ui.theme import COLORS


class HelpModal(ModalScreen):
    """Modal screen showing keyboard shortcuts and help."""

    BINDINGS = [
        Binding("escape", "dismiss", "close"),
        Binding("q", "dismiss", "close"),
        Binding("?", "dismiss", "close"),
    ]

    DEFAULT_CSS = """
    HelpModal {
        align: center middle;
    }

    HelpModal #help-container {
        width: 60;
        height: auto;
        max-height: 80%;
        background: $panel;
        border: solid $surface;
        padding: 1 2;
    }

    HelpModal #help-title {
        text-style: bold;
        color: $text;
        text-align: center;
        height: 1;
        margin-bottom: 1;
    }

    HelpModal #help-scroll {
        height: auto;
        max-height: 20;
    }

    HelpModal .section-title {
        color: $accent;
        text-style: bold;
        margin-top: 1;
    }

    HelpModal .key-row {
        height: 1;
    }

    HelpModal #help-footer {
        color: $text-muted;
        text-align: center;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="help-container"):
            yield Static("Vector Forge Help", id="help-title")
            with VerticalScroll(id="help-scroll"):
                yield Static("Navigation", classes="section-title")
                yield Static(f"[{COLORS.warning}]1[/]  [{COLORS.text_muted}]Dashboard view[/]", classes="key-row")
                yield Static(f"[{COLORS.warning}]2[/]  [{COLORS.text_muted}]Samples view[/]", classes="key-row")
                yield Static(f"[{COLORS.warning}]3[/]  [{COLORS.text_muted}]Logs view[/]", classes="key-row")
                yield Static(f"[{COLORS.warning}]Tab[/]  [{COLORS.text_muted}]Cycle screens[/]", classes="key-row")

                yield Static("Actions", classes="section-title")
                yield Static(f"[{COLORS.warning}]n[/]  [{COLORS.text_muted}]Create new task[/]", classes="key-row")
                yield Static(f"[{COLORS.warning}]/[/]  [{COLORS.text_muted}]Focus filter input[/]", classes="key-row")
                yield Static(f"[{COLORS.warning}]Esc[/]  [{COLORS.text_muted}]Clear filter / Close modal[/]", classes="key-row")

                yield Static("Navigation (Samples/Logs)", classes="section-title")
                yield Static(f"[{COLORS.warning}]j/Down[/]  [{COLORS.text_muted}]Select next item[/]", classes="key-row")
                yield Static(f"[{COLORS.warning}]k/Up[/]  [{COLORS.text_muted}]Select previous item[/]", classes="key-row")
                yield Static(f"[{COLORS.warning}]g/Home[/]  [{COLORS.text_muted}]Scroll to top[/]", classes="key-row")
                yield Static(f"[{COLORS.warning}]G/End[/]  [{COLORS.text_muted}]Scroll to bottom[/]", classes="key-row")

                yield Static("General", classes="section-title")
                yield Static(f"[{COLORS.warning}]?[/]  [{COLORS.text_muted}]Show this help[/]", classes="key-row")
                yield Static(f"[{COLORS.warning}]q[/]  [{COLORS.text_muted}]Quit application[/]", classes="key-row")
                yield Static(f"[{COLORS.warning}]Ctrl+C[/]  [{COLORS.text_muted}]Force quit[/]", classes="key-row")

            yield Static("Press any key to close", id="help-footer")

    def on_key(self, event) -> None:
        """Close on any key press."""
        self.dismiss()
