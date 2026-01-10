"""Help modal - shows keyboard shortcuts from active bindings."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static


class HelpModal(ModalScreen):
    """Modal showing keyboard shortcuts.

    Reads bindings dynamically from the app and current screen
    instead of hardcoding them.
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("q", "dismiss", "Close", show=False),
        Binding("?", "dismiss", "Close", show=False),
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
        padding: 1 2;
    }

    HelpModal #help-title {
        height: 1;
        text-style: bold;
        color: $accent;
        text-align: center;
        margin-bottom: 1;
    }

    HelpModal #help-scroll {
        height: auto;
        max-height: 20;
    }

    HelpModal .section-title {
        height: 1;
        color: $foreground-muted;
        margin-top: 1;
        margin-bottom: 0;
    }

    HelpModal .key-row {
        height: 1;
    }

    HelpModal #help-footer {
        height: 1;
        color: $foreground-disabled;
        text-align: center;
        margin-top: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="help-container"):
            yield Static("Keyboard Shortcuts", id="help-title")
            yield VerticalScroll(id="help-scroll")
            yield Static("Press any key to close", id="help-footer")

    def on_mount(self) -> None:
        scroll = self.query_one("#help-scroll", VerticalScroll)

        # Collect bindings from the screen stack
        bindings = self._collect_bindings()

        # Group bindings by category
        categories = self._categorize_bindings(bindings)

        # Render each category
        for category_name, category_bindings in categories.items():
            if category_bindings:
                scroll.mount(Static(category_name.upper(), classes="section-title"))
                for key_display, description in category_bindings:
                    scroll.mount(Static(
                        f"[$warning]{key_display:<12}[/] [$foreground-muted]{description}[/]",
                        classes="key-row"
                    ))

    def _collect_bindings(self) -> list[Binding]:
        """Collect all active bindings from app and screen stack."""
        bindings = []

        # Get bindings from the screen below this modal
        for screen in reversed(self.app.screen_stack):
            if screen is not self:
                # Get screen's own bindings
                if hasattr(screen, 'BINDINGS'):
                    for binding in screen.BINDINGS:
                        if isinstance(binding, tuple):
                            # Convert tuple to Binding
                            binding = Binding(*binding)
                        bindings.append(binding)
                break  # Only get bindings from the immediate parent screen

        # Add app-level bindings (if any that should show)
        if hasattr(self.app, 'BINDINGS'):
            for binding in self.app.BINDINGS:
                if isinstance(binding, tuple):
                    binding = Binding(*binding)
                # Only add if not already in list (by key)
                if not any(b.key == binding.key for b in bindings):
                    bindings.append(binding)

        return bindings

    def _categorize_bindings(self, bindings: list[Binding]) -> dict[str, list[tuple[str, str]]]:
        """Organize bindings into display categories."""
        categories = {
            "Navigation": [],
            "Actions": [],
            "Movement": [],
            "Other": [],
        }

        # Define which actions go in which category
        nav_actions = {"go_dashboard", "go_samples", "go_logs", "cycle", "noop"}
        movement_actions = {"next", "prev", "scroll_top", "scroll_bottom", "focus_search", "clear_focus", "open"}
        action_actions = {"new_task", "quit", "help", "create", "cancel", "save", "dismiss"}

        for binding in bindings:
            # Skip bindings with empty descriptions
            if not binding.description:
                continue

            # Get display key (use key_display if available, otherwise key)
            key_display = getattr(binding, 'key_display', None) or binding.key

            # Format multi-key bindings nicely
            if ',' in binding.key:
                # For bindings like "j,down", show just the first
                key_display = binding.key.split(',')[0]

            # Capitalize key display
            key_display = key_display.upper() if len(key_display) == 1 else key_display.capitalize()

            # Special formatting for some keys
            key_map = {
                "escape": "Esc",
                "tab": "Tab",
                "enter": "Enter",
                "up": "↑",
                "down": "↓",
                "left": "←",
                "right": "→",
                "ctrl+s": "Ctrl+S",
            }
            key_display = key_map.get(binding.key.lower(), key_display)

            entry = (key_display, binding.description)

            # Categorize by action name
            action = binding.action.split("(")[0]  # Remove any arguments

            if action in nav_actions:
                # Skip "noop" actions (current screen markers)
                if action != "noop":
                    categories["Navigation"].append(entry)
            elif action in movement_actions:
                categories["Movement"].append(entry)
            elif action in action_actions:
                categories["Actions"].append(entry)
            else:
                categories["Other"].append(entry)

        # Remove empty categories and sort
        return {k: v for k, v in categories.items() if v}

    def on_key(self, event) -> None:
        """Close on any key press."""
        self.dismiss()
