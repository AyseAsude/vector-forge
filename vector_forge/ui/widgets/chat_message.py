"""Chat message display widget for comparison UI."""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from vector_forge.ui.state import ChatMessage, ChatMessageType


# Message type icons
ICON_USER = "›"
ICON_BASELINE = "○"
ICON_STEERED = "●"
ICON_STREAMING = "⟳"


def _escape_markup(text: str) -> str:
    """Escape Rich markup characters in text."""
    return text.replace("[", r"\[").replace("]", r"\]")


class ChatMessageWidget(Vertical):
    """Display widget for a single chat message.

    Supports three message types:
    - USER: User input (accent color)
    - BASELINE: Unsteered model response (warning color)
    - STEERED: Steered model response (success color)
    """

    DEFAULT_CSS = """
    ChatMessageWidget {
        height: auto;
        padding: 1 1 1 0;
        margin: 0 1 0 0;
    }

    ChatMessageWidget.-user {
        background: $primary 8%;
        border-left: thick $accent;
        padding-left: 1;
    }

    ChatMessageWidget.-baseline {
        background: $warning 5%;
        border-left: thick $warning;
        padding-left: 1;
    }

    ChatMessageWidget.-steered {
        background: $success 5%;
        border-left: thick $success;
        padding-left: 1;
    }

    ChatMessageWidget.-streaming {
        opacity: 0.7;
    }

    ChatMessageWidget .msg-header {
        height: 1;
    }

    ChatMessageWidget .msg-icon {
        width: 2;
        text-style: bold;
    }

    ChatMessageWidget.-user .msg-icon {
        color: $accent;
    }

    ChatMessageWidget.-baseline .msg-icon {
        color: $warning;
    }

    ChatMessageWidget.-steered .msg-icon {
        color: $success;
    }

    ChatMessageWidget .msg-label {
        width: auto;
        text-style: bold;
    }

    ChatMessageWidget.-user .msg-label {
        color: $accent;
    }

    ChatMessageWidget.-baseline .msg-label {
        color: $warning;
    }

    ChatMessageWidget.-steered .msg-label {
        color: $success;
    }

    ChatMessageWidget .msg-meta {
        color: $foreground-muted;
        width: 1fr;
        text-align: right;
    }

    ChatMessageWidget .msg-content {
        height: auto;
        color: $foreground;
        padding: 0 0 0 2;
    }
    """

    def __init__(self, message: ChatMessage, **kwargs) -> None:
        self.message = message
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        """Compose the message widget with header and content."""
        msg = self.message
        icon = self._get_icon()
        label = self._get_label()
        meta = self._get_meta()

        with Horizontal(classes="msg-header"):
            yield Static(icon, classes="msg-icon", id="icon")
            yield Static(label, classes="msg-label", id="label")
            yield Static(meta, classes="msg-meta")

        content = _escape_markup(msg.content) if msg.content else ""
        yield Static(content, classes="msg-content", id="content")

        self._update_classes()

    def _get_icon(self) -> str:
        """Get icon for message type."""
        if self.message.is_streaming:
            return ICON_STREAMING
        if self.message.message_type == ChatMessageType.USER:
            return ICON_USER
        if self.message.message_type == ChatMessageType.BASELINE:
            return ICON_BASELINE
        return ICON_STEERED

    def _get_label(self) -> str:
        """Get label for message type."""
        if self.message.message_type == ChatMessageType.USER:
            return "You"
        if self.message.message_type == ChatMessageType.BASELINE:
            return "Baseline"
        layer = f" L{self.message.layer}" if self.message.layer else ""
        return f"Steered{layer}"

    def _get_meta(self) -> str:
        """Get metadata string."""
        parts = [self.message.time_str]
        if self.message.message_type == ChatMessageType.STEERED and self.message.strength:
            parts.append(f"str={self.message.strength}")
        return " · ".join(parts)

    def _update_classes(self) -> None:
        """Update CSS classes based on message state."""
        self.remove_class("-user", "-baseline", "-steered", "-streaming")

        if self.message.message_type == ChatMessageType.USER:
            self.add_class("-user")
        elif self.message.message_type == ChatMessageType.BASELINE:
            self.add_class("-baseline")
        else:
            self.add_class("-steered")

        if self.message.is_streaming:
            self.add_class("-streaming")

    def update_content(self, new_content: str) -> None:
        """Update message content (for streaming updates)."""
        self.message.content = new_content
        content = _escape_markup(new_content) if new_content else ""
        try:
            self.query_one("#content", Static).update(content)
        except Exception:
            pass

    def mark_complete(self) -> None:
        """Mark streaming as complete."""
        self.message.is_streaming = False
        self._update_classes()
        try:
            self.query_one("#icon", Static).update(self._get_icon())
            self.query_one("#label", Static).update(self._get_label())
        except Exception:
            pass
