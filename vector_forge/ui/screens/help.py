"""Help modal screen."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Static


class HelpSection(Vertical):
    """Section of help content with title and key bindings."""

    DEFAULT_CSS = """
    HelpSection {
        height: auto;
        margin-bottom: 1;
    }

    HelpSection .help-section-title {
        color: #58a6ff;
        text-style: bold;
        margin-bottom: 1;
    }
    """

    def __init__(self, title: str, bindings: list[tuple[str, str]], **kwargs) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._bindings = bindings

    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="help-section-title")
        for key, description in self._bindings:
            yield HelpBinding(key, description)


class HelpBinding(Horizontal):
    """Single key binding display."""

    DEFAULT_CSS = """
    HelpBinding {
        height: 1;
    }

    HelpBinding .help-key {
        width: 12;
        color: #d29922;
    }

    HelpBinding .help-description {
        width: 1fr;
        color: #8b949e;
    }
    """

    def __init__(self, key: str, description: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._key = key
        self._description = description

    def compose(self) -> ComposeResult:
        yield Static(self._key, classes="help-key")
        yield Static(self._description, classes="help-description")


class HelpModal(ModalScreen):
    """Modal screen showing keyboard shortcuts and help.

    Displays all available keyboard shortcuts organized by category.
    Press Escape or ? to close.
    """

    BINDINGS = [
        Binding("escape", "dismiss", "close", show=False),
        Binding("?", "dismiss", "close", show=False),
        Binding("q", "dismiss", "close", show=False),
    ]

    DEFAULT_CSS = """
    HelpModal {
        align: center middle;
    }

    HelpModal #help-container {
        width: 60;
        height: auto;
        max-height: 80%;
        background: #161b22;
        border: round #30363d;
        padding: 2;
    }

    HelpModal #help-title {
        text-style: bold;
        color: #e6edf3;
        text-align: center;
        margin-bottom: 2;
    }

    HelpModal #help-close-hint {
        color: #484f58;
        text-align: center;
        margin-top: 2;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="help-container"):
            yield Static("Keyboard Shortcuts", id="help-title")

            yield HelpSection(
                "Navigation",
                [
                    ("1", "Switch to dashboard"),
                    ("2", "Switch to parallel view"),
                    ("3", "Switch to logs view"),
                    ("Tab", "Cycle focus"),
                ],
            )

            yield HelpSection(
                "Actions",
                [
                    ("q", "Quit application"),
                    ("p", "Pause/resume extraction"),
                    ("?", "Show this help"),
                ],
            )

            yield HelpSection(
                "Dashboard",
                [
                    ("l", "Toggle log panel"),
                    ("f", "Focus log filter"),
                ],
            )

            yield HelpSection(
                "Parallel View",
                [
                    ("↑/k", "Select previous extraction"),
                    ("↓/j", "Select next extraction"),
                    ("Enter", "View selected in dashboard"),
                ],
            )

            yield HelpSection(
                "Logs View",
                [
                    ("/", "Focus filter input"),
                    ("Escape", "Clear filter"),
                    ("Home/g", "Scroll to top"),
                    ("End/G", "Scroll to bottom"),
                ],
            )

            yield Static("Press Escape or ? to close", id="help-close-hint")

    def action_dismiss(self) -> None:
        """Close the help modal."""
        self.app.pop_screen()
