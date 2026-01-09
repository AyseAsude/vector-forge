"""Target behavior display widget."""

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class TargetSection(Widget):
    """Displays the target behavior being extracted."""

    DEFAULT_CSS = """
    TargetSection {
        height: auto;
        padding: 1 2;
        margin: 0 0 1 0;
        background: $surface;
    }

    TargetSection #target-name {
        color: $text;
        text-style: bold;
        height: 1;
    }

    TargetSection #target-description {
        color: $text-muted;
        height: auto;
    }
    """

    name: reactive[str] = reactive("", init=False)
    description: reactive[str] = reactive("", init=False)

    def compose(self) -> ComposeResult:
        yield Static(id="target-name")
        yield Static(id="target-description")

    def watch_name(self, name: str) -> None:
        """Update behavior name display."""
        if not self.is_mounted:
            return
        name_widget = self.query_one("#target-name", Static)
        name_widget.update(name)

    def watch_description(self, description: str) -> None:
        """Update behavior description display."""
        if not self.is_mounted:
            return
        desc_widget = self.query_one("#target-description", Static)
        if len(description) > 80:
            description = description[:77] + "..."
        desc_widget.update(description)

    def set_behavior(self, name: str, description: str) -> None:
        """Set the target behavior."""
        self.name = name
        self.description = description
