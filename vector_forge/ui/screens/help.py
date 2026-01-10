"""Help modal screen."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static


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
        color: $foreground;
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
        color: $foreground-muted;
        text-align: center;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="help-container"):
            yield Static("Vector Forge Help", id="help-title")
            with VerticalScroll(id="help-scroll"):
                yield Static("Navigation", classes="section-title")
                yield Static(classes="key-row", id="key-1")
                yield Static(classes="key-row", id="key-2")
                yield Static(classes="key-row", id="key-3")
                yield Static(classes="key-row", id="key-tab")

                yield Static("Actions", classes="section-title")
                yield Static(classes="key-row", id="key-n")
                yield Static(classes="key-row", id="key-slash")
                yield Static(classes="key-row", id="key-esc")

                yield Static("Navigation (Samples/Logs)", classes="section-title")
                yield Static(classes="key-row", id="key-jdown")
                yield Static(classes="key-row", id="key-kup")
                yield Static(classes="key-row", id="key-ghome")
                yield Static(classes="key-row", id="key-gend")

                yield Static("General", classes="section-title")
                yield Static(classes="key-row", id="key-help")
                yield Static(classes="key-row", id="key-q")
                yield Static(classes="key-row", id="key-ctrlc")

            yield Static("Press any key to close", id="help-footer")

    def on_mount(self) -> None:
        # Navigation section
        self.query_one("#key-1", Static).update("[$warning]1[/]  [$foreground-muted]Dashboard view[/]")
        self.query_one("#key-2", Static).update("[$warning]2[/]  [$foreground-muted]Samples view[/]")
        self.query_one("#key-3", Static).update("[$warning]3[/]  [$foreground-muted]Logs view[/]")
        self.query_one("#key-tab", Static).update("[$warning]Tab[/]  [$foreground-muted]Cycle screens[/]")

        # Actions section
        self.query_one("#key-n", Static).update("[$warning]n[/]  [$foreground-muted]Create new task[/]")
        self.query_one("#key-slash", Static).update("[$warning]/[/]  [$foreground-muted]Focus filter input[/]")
        self.query_one("#key-esc", Static).update("[$warning]Esc[/]  [$foreground-muted]Clear filter / Close modal[/]")

        # Navigation (Samples/Logs) section
        self.query_one("#key-jdown", Static).update("[$warning]j/Down[/]  [$foreground-muted]Select next item[/]")
        self.query_one("#key-kup", Static).update("[$warning]k/Up[/]  [$foreground-muted]Select previous item[/]")
        self.query_one("#key-ghome", Static).update("[$warning]g/Home[/]  [$foreground-muted]Scroll to top[/]")
        self.query_one("#key-gend", Static).update("[$warning]G/End[/]  [$foreground-muted]Scroll to bottom[/]")

        # General section
        self.query_one("#key-help", Static).update("[$warning]?[/]  [$foreground-muted]Show this help[/]")
        self.query_one("#key-q", Static).update("[$warning]q[/]  [$foreground-muted]Quit application[/]")
        self.query_one("#key-ctrlc", Static).update("[$warning]Ctrl+C[/]  [$foreground-muted]Force quit[/]")

    def on_key(self, event) -> None:
        """Close on any key press."""
        self.dismiss()
