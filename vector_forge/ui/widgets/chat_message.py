"""Chat message display widget for comparison UI.

Implements a chat-like interface following the MessageRow pattern:
- Simple Static-based widget (no nested containers)
- Content built as Rich markup string
- Visual distinction through styling, not positioning
"""

from textual.widgets import Static

from vector_forge.ui.state import ChatMessage, ChatMessageType


# Message type icons - terminal aesthetic
ICON_USER = "›"
ICON_BASELINE = "○"
ICON_STEERED = "●"
ICON_STREAMING = "◌"


def _escape_markup(text: str) -> str:
    """Escape Rich markup characters in text."""
    return text.replace("[", r"\[").replace("]", r"\]")


class ChatMessageWidget(Static):
    """Display widget for a single chat message.

    Follows the MessageRow pattern from samples screen:
    - Extends Static directly (no container nesting)
    - Content as Rich markup string
    - height: auto works correctly
    """

    DEFAULT_CSS = """
    ChatMessageWidget {
        height: auto;
        padding: 1 2;
        margin: 0 1 1 0;
    }

    ChatMessageWidget.-user {
        background: $primary 10%;
        border-right: thick $accent;
    }

    ChatMessageWidget.-baseline {
        background: $surface;
        border-left: thick $warning;
    }

    ChatMessageWidget.-steered {
        background: $success 8%;
        border-left: thick $success;
    }

    ChatMessageWidget.-streaming {
        opacity: 0.6;
    }
    """

    def __init__(self, message: ChatMessage, **kwargs) -> None:
        self.message = message
        content = self._build_content()
        super().__init__(content, **kwargs)
        self._update_classes()

    def _build_content(self) -> str:
        """Build the full message content as Rich markup."""
        msg = self.message
        lines = []

        # Header line: icon + label + metadata + time
        header = self._build_header()
        lines.append(header)

        # Content
        if msg.content:
            content = _escape_markup(msg.content)
            lines.append(content)

        return "\n".join(lines)

    def _build_header(self) -> str:
        """Build header line with icon, label, and time."""
        msg = self.message
        icon = self._get_icon()

        if msg.message_type == ChatMessageType.USER:
            role = f"[$accent bold]{icon} You[/]"
        elif msg.message_type == ChatMessageType.BASELINE:
            role = f"[$warning]{icon}[/] [$warning bold]Baseline[/]"
        else:
            # Steered: include layer and strength inline
            layer = f"L{msg.layer}" if msg.layer else ""
            strength = f" @ {msg.strength}x" if msg.strength else ""
            role = f"[$success]{icon}[/] [$success bold]Steered {layer}[/][$success-darken-1]{strength}[/]"

        time = f"[$foreground-disabled]{msg.time_str}[/]"
        return f"{role}  {time}"

    def _get_icon(self) -> str:
        """Get icon for message type."""
        if self.message.is_streaming:
            return ICON_STREAMING
        if self.message.message_type == ChatMessageType.USER:
            return ICON_USER
        if self.message.message_type == ChatMessageType.BASELINE:
            return ICON_BASELINE
        return ICON_STEERED

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
        self.update(self._build_content())

    def mark_complete(self) -> None:
        """Mark streaming as complete."""
        self.message.is_streaming = False
        self._update_classes()
        self.update(self._build_content())
