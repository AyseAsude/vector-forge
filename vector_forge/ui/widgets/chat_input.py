"""Chat input widget with text area and send button."""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import TextArea, Button, Static


class ChatInput(Widget):
    """Input area for chat messages with send button.

    Supports:
    - Multi-line text input
    - Send button click
    - Ctrl+Enter to send
    - Disabled state during generation
    """

    DEFAULT_CSS = """
    ChatInput {
        height: auto;
        min-height: 3;
        max-height: 8;
        padding: 1 2;
        background: $surface;
        border-top: solid $surface-lighten-1;
    }

    ChatInput Horizontal {
        height: auto;
        min-height: 1;
    }

    ChatInput TextArea {
        height: auto;
        min-height: 1;
        max-height: 5;
        width: 1fr;
        background: $background;
        border: none;
        padding: 0 1;
    }

    ChatInput TextArea:focus {
        background: $background;
    }

    ChatInput #send-btn {
        width: 8;
        height: 3;
        min-height: 3;
        background: $accent;
        color: $background;
        border: none;
        margin-left: 1;
    }

    ChatInput #send-btn:hover {
        background: $accent 80%;
    }

    ChatInput #send-btn.-disabled {
        background: $surface-lighten-1;
        color: $foreground-muted;
    }

    ChatInput .placeholder {
        color: $foreground-muted;
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
        with Horizontal():
            yield TextArea(
                "",
                id="chat-textarea",
            )
            yield Button("Send", id="send-btn")

    def on_mount(self) -> None:
        """Focus input on mount."""
        self.query_one("#chat-textarea", TextArea).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle send button click."""
        if event.button.id == "send-btn" and self._enabled:
            self._submit()

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
        btn = self.query_one("#send-btn", Button)
        text_area = self.query_one("#chat-textarea", TextArea)

        if enabled:
            btn.remove_class("-disabled")
            btn.disabled = False
            text_area.disabled = False
        else:
            btn.add_class("-disabled")
            btn.disabled = True
            text_area.disabled = True

    def focus_input(self) -> None:
        """Focus the text input."""
        self.query_one("#chat-textarea", TextArea).focus()

    def clear(self) -> None:
        """Clear the input."""
        self.query_one("#chat-textarea", TextArea).clear()
