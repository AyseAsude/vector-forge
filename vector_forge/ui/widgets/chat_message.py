"""Chat message display widget for comparison UI.

Shows user messages, baseline responses, and steered responses
with distinct visual styling for easy comparison.
"""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from vector_forge.ui.state import ChatMessage, ChatMessageType


def _escape_markup(text: str) -> str:
    """Escape Rich markup characters in text to prevent interpretation."""
    return text.replace("[", r"\[").replace("]", r"\]")


class ChatMessageWidget(Static):
    """Display widget for a single chat message.

    Supports three types with distinct styling:
    - USER: User input (accent color)
    - BASELINE: Unsteered model response (warning/orange color)
    - STEERED: Steered model response (success/green color)
    """

    DEFAULT_CSS = """
    ChatMessageWidget {
        height: auto;
        padding: 0 0 1 0;
        margin: 0 2 0 0;
    }

    ChatMessageWidget:hover {
        background: $boost;
    }

    ChatMessageWidget.-user {
        border-left: wide $accent;
        padding-left: 1;
    }

    ChatMessageWidget.-baseline {
        border-left: wide $warning;
        padding-left: 1;
    }

    ChatMessageWidget.-steered {
        border-left: wide $success;
        padding-left: 1;
    }

    ChatMessageWidget.-streaming {
        opacity: 0.8;
    }
    """

    def __init__(self, message: ChatMessage, **kwargs) -> None:
        self.message = message
        # Pass initial content to Static parent
        super().__init__(self._render_content(), **kwargs)
        self._update_classes()

    def _update_classes(self) -> None:
        """Update CSS classes based on message state."""
        # Remove existing type classes
        self.remove_class("-user", "-baseline", "-steered", "-streaming")

        # Add type class
        if self.message.message_type == ChatMessageType.USER:
            self.add_class("-user")
        elif self.message.message_type == ChatMessageType.BASELINE:
            self.add_class("-baseline")
        else:
            self.add_class("-steered")

        # Add streaming class if applicable
        if self.message.is_streaming:
            self.add_class("-streaming")

    def _render_content(self) -> str:
        """Render message content with header."""
        msg = self.message

        # Type colors
        type_colors = {
            ChatMessageType.USER: "$accent",
            ChatMessageType.BASELINE: "$warning",
            ChatMessageType.STEERED: "$success",
        }
        color = type_colors.get(msg.message_type, "$foreground")

        # Header line with time and type
        streaming_indicator = " ..." if msg.is_streaming else ""
        header = (
            f"[$foreground-muted]{msg.time_str}[/]  "
            f"[{color} bold]{msg.type_label}[/]{streaming_indicator}"
        )

        # Content (escaped to prevent markup issues)
        content = _escape_markup(msg.content) if msg.content else ""

        # Truncate very long messages in list view
        if len(content) > 1000:
            content = content[:997] + "..."

        return f"{header}\n[$foreground]{content}[/]"

    def on_mount(self) -> None:
        """Update content on mount."""
        self.update(self._render_content())

    def update_content(self, new_content: str) -> None:
        """Update message content (for streaming updates)."""
        self.message.content = new_content
        self.update(self._render_content())

    def mark_complete(self) -> None:
        """Mark streaming as complete."""
        self.message.is_streaming = False
        self._update_classes()
        self.update(self._render_content())
