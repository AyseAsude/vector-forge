"""Chat screen for comparative LLM inference with steering vectors.

Provides a split-panel interface:
- Left: Vector selector (from selected task) + generation settings
- Right: Conversation with dual responses (baseline vs steered)
"""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Static

from vector_forge.ui.state import (
    get_state,
    ChatMessageType,
    ExtractionStatus,
)
from vector_forge.ui.widgets.chat_message import ChatMessageWidget
from vector_forge.ui.widgets.vector_selector import VectorSelector
from vector_forge.ui.widgets.chat_input import ChatInput
from vector_forge.ui.widgets.tmux_bar import TmuxBar


class ConversationPanel(Vertical):
    """Right panel showing the chat conversation."""

    DEFAULT_CSS = """
    ConversationPanel {
        width: 1fr;
        background: $surface;
    }

    ConversationPanel .panel-header {
        height: auto;
        padding: 1 2;
        border-bottom: solid $surface-lighten-1;
    }

    ConversationPanel .title {
        text-style: bold;
    }

    ConversationPanel .message-stream {
        height: 1fr;
        padding: 1 0 1 2;
        background: $background;
        scrollbar-gutter: stable;
    }

    ConversationPanel .empty-state {
        height: 1fr;
        content-align: center middle;
        color: $foreground-muted;
    }

    ConversationPanel .hint-bar {
        height: 1;
        padding: 0 1;
        color: $foreground-muted;
        background: $surface;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._displayed_ids: set[str] = set()

    def compose(self) -> ComposeResult:
        with Vertical(classes="panel-header"):
            yield Static("CONVERSATION", classes="title", id="conv-title")

        yield VerticalScroll(classes="message-stream", id="message-stream")
        yield ChatInput(id="chat-input")
        yield Static("enter send · ctrl+n clear", classes="hint-bar", id="hint-bar")

    def _is_at_bottom(self) -> bool:
        """Check if scroll is at bottom."""
        try:
            stream = self.query_one("#message-stream", VerticalScroll)
            return stream.scroll_y >= (stream.max_scroll_y - 3)
        except Exception:
            return True

    def add_message(self, msg) -> ChatMessageWidget:
        """Add a new message to the conversation."""
        stream = self.query_one("#message-stream", VerticalScroll)

        # Remove empty placeholder if present
        for empty in stream.query(".empty-state"):
            empty.remove()

        was_at_bottom = self._is_at_bottom()

        # Create and mount widget
        widget = ChatMessageWidget(msg, id=f"msg-{msg.id}")
        stream.mount(widget)
        self._displayed_ids.add(msg.id)

        # Auto-scroll if we were at bottom
        if was_at_bottom:
            stream.scroll_end(animate=False)

        return widget

    def update_streaming_message(self, msg_id: str, content: str) -> None:
        """Update a streaming message with new content."""
        try:
            widget = self.query_one(f"#msg-{msg_id}", ChatMessageWidget)
            widget.update_content(content)

            if self._is_at_bottom():
                self.query_one("#message-stream", VerticalScroll).scroll_end(
                    animate=False
                )
        except Exception:
            pass

    def clear_conversation(self) -> None:
        """Clear all messages."""
        stream = self.query_one("#message-stream", VerticalScroll)
        stream.remove_children()
        stream.mount(
            Static(
                "Send a message to start the conversation.",
                classes="empty-state",
            )
        )
        self._displayed_ids.clear()

    def show_empty_state(self, message: str = "No task selected") -> None:
        """Show empty state message."""
        stream = self.query_one("#message-stream", VerticalScroll)
        stream.remove_children()
        stream.mount(Static(message, classes="empty-state"))


class ChatScreen(Screen):
    """Chat screen with steering vector comparison."""

    BINDINGS = [
        Binding("1", "go_dashboard", "Dashboard", key_display="1"),
        Binding("2", "go_samples", "Samples", key_display="2"),
        Binding("3", "go_logs", "Logs", key_display="3"),
        Binding("4", "noop", "Chat", show=False),
        Binding("tab", "cycle", "Next Screen"),
        Binding("ctrl+n", "new_session", "New Chat"),
        Binding("q", "quit", "Quit"),
    ]

    DEFAULT_CSS = """
    ChatScreen {
        background: $background;
    }

    ChatScreen #content {
        height: 1fr;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._last_extraction_id: str | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="content"):
            yield VectorSelector(id="vector-selector")
            yield ConversationPanel(id="conversation-panel")
        yield TmuxBar(active_screen="chat")

    def on_mount(self) -> None:
        """Initialize on mount."""
        self._sync()

    def on_screen_resume(self) -> None:
        """Sync when returning to screen."""
        self._sync()

    def _sync(self) -> None:
        """Synchronize UI with current state."""
        state = get_state()
        extraction_id = state.selected_id

        # Update vector selector
        selector = self.query_one("#vector-selector", VectorSelector)
        selector.update_from_extraction(extraction_id)

        # Check if extraction changed
        if extraction_id != self._last_extraction_id:
            self._last_extraction_id = extraction_id
            self._load_session(extraction_id)

        # Update tmux bar
        self.query_one(TmuxBar).refresh_info()

    def _load_session(self, extraction_id: str | None) -> None:
        """Load chat session for extraction."""
        conv_panel = self.query_one("#conversation-panel", ConversationPanel)

        if extraction_id is None:
            conv_panel.show_empty_state("Select a task from Dashboard to chat")
            return

        state = get_state()
        extraction = state.extractions.get(extraction_id)

        if extraction is None:
            conv_panel.show_empty_state("Task not found")
            return

        if extraction.status != ExtractionStatus.COMPLETE:
            conv_panel.show_empty_state(
                f"Task is {extraction.status.value}\n\n"
                "Chat available after extraction completes."
            )
            return

        # Get or create session
        session = state.chat.get_or_create_session(extraction_id)

        # Display existing messages
        conv_panel.clear_conversation()
        if session.messages:
            # Clear the empty state message first
            stream = conv_panel.query_one("#message-stream", VerticalScroll)
            stream.remove_children()

            for msg in session.messages:
                conv_panel.add_message(msg)

    async def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        """Handle user message submission."""
        state = get_state()
        extraction_id = state.selected_id

        if extraction_id is None:
            return

        extraction = state.extractions.get(extraction_id)
        if extraction is None or extraction.status != ExtractionStatus.COMPLETE:
            return

        # Get session
        session = state.chat.get_or_create_session(extraction_id)

        # Get settings from selector
        selector = self.query_one("#vector-selector", VectorSelector)
        layer = selector.selected_layer
        strength = selector.strength
        temperature = selector.temperature
        max_tokens = selector.max_tokens

        # Add user message
        user_msg = session.add_message(ChatMessageType.USER, event.content)
        conv_panel = self.query_one("#conversation-panel", ConversationPanel)
        conv_panel.add_message(user_msg)

        # Disable input while generating
        conv_panel.query_one("#chat-input", ChatInput).set_enabled(False)
        state.chat.is_generating = True

        # Generate responses
        await self._generate_responses(
            session=session,
            conv_panel=conv_panel,
            extraction_id=extraction_id,
            layer=layer,
            strength=strength,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Re-enable input
        conv_panel.query_one("#chat-input", ChatInput).set_enabled(True)
        conv_panel.query_one("#chat-input", ChatInput).focus_input()
        state.chat.is_generating = False

    async def _generate_responses(
        self,
        session,
        conv_panel: ConversationPanel,
        extraction_id: str,
        layer: int | None,
        strength: float,
        temperature: float,
        max_tokens: int,
    ) -> None:
        """Generate baseline and steered responses."""
        # Create placeholder messages
        baseline_msg = session.add_message(ChatMessageType.BASELINE, "")
        baseline_msg.is_streaming = True
        baseline_widget = conv_panel.add_message(baseline_msg)

        steered_msg = session.add_message(
            ChatMessageType.STEERED,
            "",
            layer=layer,
            strength=strength,
        )
        steered_msg.is_streaming = True
        steered_widget = conv_panel.add_message(steered_msg)

        try:
            # Import service (lazy to avoid circular imports)
            from vector_forge.services.chat_service import ChatService

            # Create service
            service = ChatService(extraction_id=extraction_id)

            # Get conversation history
            history = session.get_conversation_history()
            # Remove the last empty assistant message we just added
            if history and history[-1]["content"] == "":
                history = history[:-1]

            # Generate baseline (no steering)
            baseline_content = await service.generate_baseline(
                messages=history,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            baseline_msg.content = baseline_content
            baseline_msg.is_streaming = False
            baseline_widget.mark_complete()
            conv_panel.update_streaming_message(baseline_msg.id, baseline_content)

            # Generate steered response
            steered_content = await service.generate_steered(
                messages=history,
                layer=layer,
                strength=strength,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            steered_msg.content = steered_content
            steered_msg.is_streaming = False
            steered_widget.mark_complete()
            conv_panel.update_streaming_message(steered_msg.id, steered_content)

        except Exception as e:
            # Handle errors
            error_msg = f"Error: {e}"
            baseline_msg.content = error_msg
            baseline_msg.is_streaming = False
            baseline_widget.mark_complete()

            steered_msg.content = error_msg
            steered_msg.is_streaming = False
            steered_widget.mark_complete()

    def action_noop(self) -> None:
        """No-op for current screen binding."""
        pass

    def action_go_dashboard(self) -> None:
        """Navigate to dashboard."""
        self.app.switch_screen("dashboard")

    def action_go_samples(self) -> None:
        """Navigate to samples."""
        self.app.switch_screen("samples")

    def action_go_logs(self) -> None:
        """Navigate to logs."""
        self.app.switch_screen("logs")

    def action_cycle(self) -> None:
        """Cycle to next screen."""
        self.app.switch_screen("dashboard")

    def action_new_session(self) -> None:
        """Clear and start new chat session."""
        state = get_state()
        extraction_id = state.selected_id

        if extraction_id:
            # Remove existing session
            if extraction_id in state.chat.sessions:
                del state.chat.sessions[extraction_id]

            # Reload (creates new session)
            self._load_session(extraction_id)

    def action_quit(self) -> None:
        """Quit application."""
        self.app.exit()
