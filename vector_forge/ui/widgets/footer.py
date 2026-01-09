"""Application footer widget with keyboard shortcuts."""

from typing import List, Tuple

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class AppFooter(Widget):
    """Footer displaying keyboard shortcuts.

    Shows context-sensitive shortcuts based on current screen.
    """

    DEFAULT_CSS = """
    AppFooter {
        height: 1;
        background: #0d1117;
        color: #8b949e;
        dock: bottom;
        padding: 0 1;
    }

    AppFooter #footer-content {
        width: 1fr;
    }
    """

    # List of (key, description) tuples
    shortcuts: reactive[List[Tuple[str, str]]] = reactive(
        [
            ("1", "dashboard"),
            ("2", "parallel"),
            ("3", "logs"),
            ("q", "quit"),
            ("p", "pause"),
            ("?", "help"),
        ],
        init=False,
    )

    def compose(self) -> ComposeResult:
        yield Static(id="footer-content")

    def on_mount(self) -> None:
        """Initialize footer content."""
        self._update_content()

    def watch_shortcuts(self, shortcuts: List[Tuple[str, str]]) -> None:
        """Update when shortcuts change."""
        if self.is_mounted:
            self._update_content()

    def _update_content(self) -> None:
        """Render shortcuts as formatted text."""
        if not self.is_mounted:
            return
        parts = []
        for key, desc in self.shortcuts:
            parts.append(f"[#d29922]{key}[/] {desc}")

        content = " · ".join(parts)
        footer_content = self.query_one("#footer-content", Static)
        footer_content.update(content)

    def set_shortcuts(self, shortcuts: List[Tuple[str, str]]) -> None:
        """Update the displayed shortcuts.

        Args:
            shortcuts: List of (key, description) tuples.
        """
        self.shortcuts = shortcuts
