"""Chat input widget for message composition."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import TextArea


class ChatInput(Widget):
    """Multi-line input for chat messages.

    Features:
    - Auto-expanding textarea
    - Ctrl+Enter to send
    - Disabled state during generation
    """

    DEFAULT_CSS = """
    ChatInput {
        height: auto;
        margin-top: 1;
        padding: 1 1;
        background: $background;
    }

    ChatInput .input-box {
        height: auto;
        background: transparent;
    }

    ChatInput TextArea {
        height: auto;
        min-height: 1;
        max-height: 5;
        width: 1fr;
        background: transparent;
        border: none;
        padding: 0;
    }

    ChatInput TextArea:focus {
        background: transparent;
        border: none;
    }

    ChatInput TextArea .text-area--cursor-line {
        background: transparent;
    }
    """

    class Submitted(Message):
        """Emitted when user submits a message."""

        def __init__(self, content: str) -> None:
            super().__init__()
            self.content = content

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._enabled = True

    def compose(self) -> ComposeResult:
        with Vertical(classes="input-box"):
            yield TextArea("", id="chat-textarea")

    def on_mount(self) -> None:
        """Focus input on mount."""
        self.query_one("#chat-textarea", TextArea).focus()

    def on_key(self, event) -> None:
        """Handle keyboard shortcuts."""
        # Ctrl+Enter or Cmd+Enter to send
        if event.key == "ctrl+enter" or event.key == "cmd+enter":
            if self._enabled:
                self._submit()
                event.prevent_default()

    def _submit(self) -> None:
        """Submit the current input."""
        text_area = self.query_one("#chat-textarea", TextArea)
        content = text_area.text.strip()

        if content:
            self.post_message(self.Submitted(content))
            text_area.clear()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the input."""
        self._enabled = enabled
        text_area = self.query_one("#chat-textarea", TextArea)

        if enabled:
            self.remove_class("-disabled")
            text_area.disabled = False
        else:
            self.add_class("-disabled")
            text_area.disabled = True

    def focus_input(self) -> None:
        """Focus the text input."""
        self.query_one("#chat-textarea", TextArea).focus()

    def clear(self) -> None:
        """Clear the input."""
        self.query_one("#chat-textarea", TextArea).clear()
