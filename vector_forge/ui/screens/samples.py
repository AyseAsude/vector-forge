"""Samples screen - split view of parallel extraction workers.

Uses native Textual widgets for high performance:
- RichLog for conversation display (efficient appending, native scrolling)
- ListView for worker selection (native keyboard navigation)
- Reactive updates instead of remove/remount patterns
"""

import json
import re

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.screen import Screen, ModalScreen
from textual.widgets import Static, ListView, ListItem, Label

from vector_forge.ui.state import (
    AgentUIState,
    AgentStatus,
    AgentMessage,
    ToolCall,
    ExtractionStatus,
    get_state,
)
from vector_forge.ui.theme import ICONS
from vector_forge.ui.widgets.tmux_bar import TmuxBar


def _escape_markup(text: str) -> str:
    """Escape Rich markup characters in text to prevent interpretation."""
    return text.replace("[", r"\[").replace("]", r"\]")


# ─────────────────────────────────────────────────────────────────────────────
# Message Detail Modal - Shows full message content
# ─────────────────────────────────────────────────────────────────────────────


class MessageDetailModal(ModalScreen):
    """Modal showing full message details."""

    BINDINGS = [
        Binding("escape", "dismiss", "close"),
        Binding("q", "dismiss", "close"),
        Binding("j", "scroll_down", "scroll down", show=False),
        Binding("k", "scroll_up", "scroll up", show=False),
    ]

    DEFAULT_CSS = """
    MessageDetailModal {
        align: center middle;
    }

    MessageDetailModal #modal-container {
        width: 90%;
        max-width: 120;
        height: 85%;
        background: $surface;
        border: solid $primary;
    }

    MessageDetailModal #modal-header {
        height: auto;
        padding: 1 2;
        background: $surface;
        border-bottom: solid $surface-lighten-1;
    }

    MessageDetailModal .title {
        height: 1;
        text-style: bold;
    }

    MessageDetailModal .meta {
        height: auto;
        color: $foreground-muted;
    }

    MessageDetailModal #modal-content {
        height: 1fr;
        padding: 1 0 1 2;
        background: $background;
        scrollbar-gutter: stable;
    }

    MessageDetailModal .section-header {
        height: 1;
        color: $accent;
        text-style: bold;
        margin-top: 1;
        margin-bottom: 0;
    }

    MessageDetailModal .section-content {
        height: auto;
        padding: 1;
        margin-bottom: 1;
        background: $surface;
    }

    MessageDetailModal #modal-footer {
        height: 2;
        padding: 0 2;
        background: $surface;
        border-top: solid $surface-lighten-1;
    }

    MessageDetailModal .footer-text {
        height: 1;
        color: $foreground-muted;
        text-align: center;
    }
    """

    def __init__(self, msg: AgentMessage, **kwargs) -> None:
        super().__init__(**kwargs)
        self.msg = msg

    def compose(self) -> ComposeResult:
        msg = self.msg
        role_colors = {
            "system": "$primary",
            "user": "$accent",
            "assistant": "$success",
            "tool": "$warning",
        }
        color = role_colors.get(msg.role.value, "$foreground-muted")

        with Vertical(id="modal-container"):
            # Header
            with Vertical(id="modal-header"):
                yield Static(f"[{color} bold]{msg.role.value.upper()}[/]", classes="title")
                yield Static(f"Time: {msg.time_str}", classes="meta")

            # Content
            with VerticalScroll(id="modal-content"):
                # Message content
                if msg.content:
                    yield Static("CONTENT", classes="section-header")
                    yield Static(_escape_markup(msg.content), classes="section-content")

                # Tool calls
                if msg.tool_calls:
                    yield Static(f"TOOL CALLS ({len(msg.tool_calls)})", classes="section-header")
                    for tc in msg.tool_calls:
                        tc_color = "$success" if tc.status == "success" else "$warning"
                        duration = f" ({tc.duration_ms}ms)" if tc.duration_ms else ""
                        yield Static(f"[{tc_color} bold]▸ {tc.name}[/]{duration}", classes="section-content")
                        if tc.arguments:
                            # Try to format JSON arguments
                            try:
                                args_obj = json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
                                args_str = _escape_markup(json.dumps(args_obj, indent=2))
                            except (json.JSONDecodeError, TypeError):
                                args_str = _escape_markup(str(tc.arguments))
                            yield Static(args_str, classes="section-content")
                        if tc.result:
                            yield Static(f"Result: {_escape_markup(str(tc.result))}", classes="section-content")

            # Footer
            with Vertical(id="modal-footer"):
                yield Static("Press ESC or q to close", classes="footer-text")

    def action_scroll_down(self) -> None:
        self.query_one("#modal-content", VerticalScroll).scroll_down()

    def action_scroll_up(self) -> None:
        self.query_one("#modal-content", VerticalScroll).scroll_up()


# ─────────────────────────────────────────────────────────────────────────────
# Tool Call Modal - Shows tool call details
# ─────────────────────────────────────────────────────────────────────────────


class ToolCallModal(ModalScreen):
    """Modal showing tool call details."""

    BINDINGS = [
        Binding("escape", "dismiss", "close"),
        Binding("q", "dismiss", "close"),
    ]

    DEFAULT_CSS = """
    ToolCallModal {
        align: center middle;
    }

    ToolCallModal #modal {
        width: 80%;
        max-width: 100;
        height: auto;
        max-height: 80%;
        background: $surface;
        padding: 1 2;
    }

    ToolCallModal .title {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
    }

    ToolCallModal .status {
        height: 1;
        color: $foreground-muted;
        margin-bottom: 1;
    }

    ToolCallModal .section {
        height: 1;
        color: $accent;
        text-style: bold;
        margin-top: 1;
    }

    ToolCallModal .content {
        height: auto;
        max-height: 10;
        padding: 1;
        background: $background;
        margin-bottom: 1;
    }

    ToolCallModal .footer {
        height: 1;
        color: $foreground-muted;
        text-align: center;
        margin-top: 1;
    }
    """

    def __init__(self, tool_call: ToolCall, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tool_call = tool_call

    def compose(self) -> ComposeResult:
        tc = self.tool_call
        status_colors = {
            "pending": "$foreground-muted",
            "running": "$accent",
            "success": "$success",
            "error": "$error",
        }
        color = status_colors.get(tc.status, "$foreground-disabled")
        duration = f" · {tc.duration_ms}ms" if tc.duration_ms else ""

        args_text = tc.arguments if tc.arguments else "(none)"

        with Vertical(id="modal"):
            yield Static(f"[{color}]▸[/] [bold]{tc.name}[/]", classes="title")
            yield Static(f"{tc.status}{duration}", classes="status")
            yield Static("ARGUMENTS", classes="section")
            yield Static(args_text, classes="content")

            if tc.result:
                yield Static("RESULT", classes="section")
                yield Static(tc.result, classes="content")

            yield Static("Press ESC to close", classes="footer")


# ─────────────────────────────────────────────────────────────────────────────
# Worker List Item - Native ListView item
# ─────────────────────────────────────────────────────────────────────────────


class WorkerListItem(ListItem):
    """A worker item in the ListView."""

    DEFAULT_CSS = """
    WorkerListItem {
        height: auto;
        padding: 1;
        background: transparent;
        margin-bottom: 1;
    }

    WorkerListItem:hover {
        background: $primary 10%;
    }

    WorkerListItem > Horizontal,
    WorkerListItem > Static {
        background: transparent;
    }

    WorkerListItem .row {
        height: 1;
    }

    WorkerListItem .name {
        width: 1fr;
    }

    WorkerListItem .time {
        width: auto;
        color: $foreground-muted;
    }

    WorkerListItem .meta {
        height: 1;
        color: $foreground-muted;
    }
    """

    def __init__(self, agent: AgentUIState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.agent_id = agent.id
        self._agent = agent

    def compose(self) -> ComposeResult:
        icon, color, label = self._get_display_values()
        agent = self._agent

        with Horizontal(classes="row"):
            yield Static(
                f"[{color}]{icon}[/] [bold]{agent.name}[/]",
                classes="name",
                id="worker-name",
            )
            yield Static(agent.elapsed_str, classes="time", id="worker-time")
        yield Static(
            f"[{color}]{label}[/] · {agent.turns} turns · {agent.tool_calls_count} {agent.count_label}",
            classes="meta",
            id="worker-meta",
        )

    def _get_display_values(self) -> tuple:
        """Get display values for the agent."""
        status_map = {
            AgentStatus.IDLE: (ICONS.pending, "$foreground-muted", "IDLE"),
            AgentStatus.RUNNING: (ICONS.running, "$accent", "RUNNING"),
            AgentStatus.WAITING: (ICONS.waiting, "$foreground-muted", "WAITING"),
            AgentStatus.COMPLETE: (ICONS.complete, "$success", "DONE"),
            AgentStatus.ERROR: (ICONS.failed, "$error", "ERROR"),
        }
        return status_map.get(self._agent.status, (ICONS.pending, "$foreground-muted", "?"))

    def update_agent(self, agent: AgentUIState) -> None:
        """Update the agent data and refresh display."""
        self._agent = agent
        if self.is_mounted:
            icon, color, label = self._get_display_values()
            try:
                self.query_one("#worker-name", Static).update(
                    f"[{color}]{icon}[/] [bold]{agent.name}[/]"
                )
                self.query_one("#worker-time", Static).update(agent.elapsed_str)
                self.query_one("#worker-meta", Static).update(
                    f"[{color}]{label}[/] · {agent.turns} turns · {agent.tool_calls_count} {agent.count_label}"
                )
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Workers Panel - Uses native ListView
# ─────────────────────────────────────────────────────────────────────────────


class WorkersPanel(Vertical):
    """Left panel with list of workers using native ListView."""

    DEFAULT_CSS = """
    WorkersPanel {
        width: 1fr;
        max-width: 40;
        padding: 1 0 1 2;
        background: $surface;
    }

    WorkersPanel .header {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
        padding-right: 2;
    }

    WorkersPanel .count {
        height: 1;
        color: $foreground-muted;
        margin-bottom: 1;
        padding-right: 2;
    }

    WorkersPanel ListView {
        height: 1fr;
        scrollbar-gutter: stable;
        padding-right: 1;
        background: transparent;
    }

    WorkersPanel .empty {
        padding: 2;
        color: $foreground-muted;
        text-align: center;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._items: dict[str, WorkerListItem] = {}

    def compose(self) -> ComposeResult:
        yield Static("PARALLEL RUNS", classes="header")
        yield Static("No workers", classes="count", id="worker-count")
        yield ListView(id="workers-list")

    def update_workers(self, agents: dict[str, AgentUIState], selected_id: str | None = None) -> None:
        """Update the workers list efficiently."""
        count_widget = self.query_one("#worker-count", Static)
        list_view = self.query_one("#workers-list", ListView)

        # Update count
        running = sum(1 for a in agents.values() if a.status == AgentStatus.RUNNING)
        total = len(agents)
        if total == 0:
            count_widget.update("No workers")
        elif running > 0:
            count_widget.update(f"{running} active / {total} total")
        else:
            count_widget.update(f"{total} workers")

        current_ids = set(agents.keys())
        existing_ids = set(self._items.keys())

        # Remove workers that no longer exist
        for agent_id in existing_ids - current_ids:
            if agent_id in self._items:
                self._items[agent_id].remove()
                del self._items[agent_id]

        # Update existing or add new workers
        for agent_id, agent in agents.items():
            if agent_id in self._items:
                # Update existing
                self._items[agent_id].update_agent(agent)
            else:
                # Add new
                item = WorkerListItem(agent, id=f"worker-{agent_id}")
                list_view.append(item)
                self._items[agent_id] = item

        # ListView handles its own highlight styling natively

    def get_selected_agent_id(self) -> str | None:
        """Get the currently highlighted worker's agent ID."""
        list_view = self.query_one("#workers-list", ListView)
        if list_view.highlighted_child is not None:
            item = list_view.highlighted_child
            if isinstance(item, WorkerListItem):
                return item.agent_id
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Message Row - Individual message widget (like LogRow in logs screen)
# ─────────────────────────────────────────────────────────────────────────────


class MessageRow(Static):
    """Single message row in the conversation stream."""

    class Clicked(Message):
        """Posted when a message row is clicked."""
        def __init__(self, msg: AgentMessage) -> None:
            super().__init__()
            self.msg = msg

    DEFAULT_CSS = """
    MessageRow {
        height: auto;
        margin-right: 2;
        padding: 0 0 1 0;
    }

    MessageRow:hover {
        background: $boost;
    }
    """

    def __init__(self, msg: AgentMessage, **kwargs) -> None:
        self.msg = msg
        content = self._compute_content()
        super().__init__(content, **kwargs)

    def on_click(self) -> None:
        """Handle click to open detail modal."""
        self.post_message(self.Clicked(self.msg))

    def _clean_content(self, content: str) -> str:
        """Clean up content for display - just collapse whitespace, no transformation."""
        # Collapse multiple whitespace/newlines into single space
        content = re.sub(r'\s+', ' ', content)
        return content.strip()

    def _compute_content(self) -> str:
        """Compute the display content for this message."""
        msg = self.msg

        role_colors = {
            "system": "$primary",
            "user": "$accent",
            "assistant": "$success",
            "tool": "$warning",
        }
        color = role_colors.get(msg.role.value, "$foreground-muted")

        # Build message content
        lines = []

        # Header line: time  [ROLE]  - modern bracket style
        lines.append(
            f"[$foreground-disabled]{msg.time_str}[/]  "
            f"[{color} bold]{msg.role.value.upper()}[/]"
        )

        # Content (cleaned and truncated for display)
        content = msg.content
        if content:
            # Clean up markdown artifacts
            content_display = self._clean_content(content)
            if len(content_display) > 200:
                content_display = content_display[:197] + "..."
            lines.append(f"  {content_display}")

        # Tool calls
        for tc in msg.tool_calls:
            tc_colors = {
                "pending": "$foreground-muted",
                "running": "$warning",
                "success": "$success",
                "error": "$error",
            }
            tc_color = tc_colors.get(tc.status, "$foreground-muted")
            duration = f" ({tc.duration_ms}ms)" if tc.duration_ms else ""
            lines.append(f"  [{tc_color}]▸ {tc.name}[/]{duration}")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Conversation Panel - Uses VerticalScroll with MessageRow widgets (like LogPanel)
# ─────────────────────────────────────────────────────────────────────────────


class ConversationPanel(Vertical):
    """Right panel showing worker conversation.

    Uses VerticalScroll with individual MessageRow widgets for consistency
    with the logs screen design. Features smart auto-scroll: pauses when
    user scrolls up, resumes when user scrolls back to bottom.
    """

    DEFAULT_CSS = """
    ConversationPanel {
        width: 2fr;
        background: $surface;
    }

    ConversationPanel .panel-header {
        height: auto;
        padding: 1 2;
    }

    ConversationPanel .title-row {
        height: 1;
        margin-bottom: 1;
    }

    ConversationPanel .title {
        width: 1fr;
        text-style: bold;
    }

    ConversationPanel .time {
        width: auto;
        color: $foreground-muted;
    }

    ConversationPanel .stats {
        height: 1;
        color: $foreground-muted;
    }

    ConversationPanel .message-stream {
        height: 1fr;
        padding: 1 0 1 2;
        background: $background;
        scrollbar-gutter: stable;
    }

    ConversationPanel .empty {
        height: 1fr;
        content-align: center middle;
        color: $foreground-muted;
        margin-right: 2;
    }
    """

    # Track current agent
    current_agent_id: reactive[str | None] = reactive(None)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._displayed_message_ids: set[str] = set()

    def compose(self) -> ComposeResult:
        with Vertical(classes="panel-header"):
            with Horizontal(classes="title-row"):
                yield Static("[$foreground-muted]Select a worker[/]", classes="title", id="conv-title")
                yield Static("", classes="time", id="conv-time")
            yield Static("[$foreground-disabled]Click a worker to view conversation[/]", classes="stats", id="conv-stats")
        yield VerticalScroll(classes="message-stream", id="message-stream")

    def _is_at_bottom(self) -> bool:
        """Check if the message stream is scrolled to the bottom."""
        try:
            stream = self.query_one("#message-stream", VerticalScroll)
            # Consider "at bottom" if within 3 lines of max scroll
            return stream.scroll_y >= (stream.max_scroll_y - 3)
        except Exception:
            return True

    def show_agent(self, agent: AgentUIState | None) -> None:
        """Display an agent's conversation."""
        title = self.query_one("#conv-title", Static)
        time_widget = self.query_one("#conv-time", Static)
        stats = self.query_one("#conv-stats", Static)
        stream = self.query_one("#message-stream", VerticalScroll)

        if agent is None:
            if self.current_agent_id is not None:
                title.update("[$foreground-muted]Select a worker[/]")
                time_widget.update("")
                stats.update("[$foreground-disabled]Click a worker to view conversation[/]")
                stream.remove_children()
                stream.mount(Static("Select a worker to view messages", classes="empty"))
                self._displayed_message_ids.clear()
                self.current_agent_id = None
            return

        # Get status display
        status_map = {
            AgentStatus.IDLE: (ICONS.pending, "$foreground-muted", "IDLE"),
            AgentStatus.RUNNING: (ICONS.running, "$accent", "RUNNING"),
            AgentStatus.WAITING: (ICONS.waiting, "$foreground-muted", "WAITING"),
            AgentStatus.COMPLETE: (ICONS.complete, "$success", "DONE"),
            AgentStatus.ERROR: (ICONS.failed, "$error", "ERROR"),
        }
        icon, color, label = status_map.get(agent.status, (ICONS.pending, "$foreground-muted", "?"))

        # Update header (always - cheap operation)
        title.update(f"[{color}]{icon}[/] [bold]{agent.name}[/]")
        time_widget.update(f"[$foreground-muted]{agent.elapsed_str}[/]")
        stats.update(
            f"[{color}]{label}[/] · {agent.role} · "
            f"{len(agent.messages)} msgs · {agent.turns} turns · {agent.tool_calls_count} {agent.count_label}"
        )

        # Check if agent changed
        agent_changed = agent.id != self.current_agent_id
        if agent_changed:
            # Clear and rebuild for new agent
            stream.remove_children()
            self._displayed_message_ids.clear()
            self.current_agent_id = agent.id

        # Remove empty placeholder if present
        for empty in stream.query(".empty"):
            empty.remove()

        # Check if at bottom BEFORE adding new messages (for smart auto-scroll)
        was_at_bottom = self._is_at_bottom()

        # Track if we added new messages
        added_new = False

        # Append only new messages (incremental update)
        for msg in agent.messages:
            if msg.id not in self._displayed_message_ids:
                stream.mount(MessageRow(msg))
                self._displayed_message_ids.add(msg.id)
                added_new = True

        # Handle empty state
        if not agent.messages:
            if not list(stream.query(".empty")):
                stream.mount(Static("No messages yet", classes="empty"))

        # Only auto-scroll if user was already at bottom (smart scroll like Claude Code)
        if added_new and (was_at_bottom or agent_changed):
            stream.scroll_end(animate=False)

    def update_time(self, agent: AgentUIState) -> None:
        """Update just the elapsed time display."""
        if agent.id == self.current_agent_id:
            try:
                self.query_one("#conv-time", Static).update(
                    f"[$foreground-muted]{agent.elapsed_str}[/]"
                )
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Samples Screen
# ─────────────────────────────────────────────────────────────────────────────


class SamplesScreen(Screen):
    """Split view screen showing parallel workers."""

    BINDINGS = [
        # Navigation between screens
        Binding("1", "go_dashboard", "Dashboard", key_display="1"),
        Binding("2", "noop", "Samples", show=False),
        Binding("3", "go_logs", "Logs", key_display="3"),
        Binding("tab", "cycle", "Next Screen"),
        # List navigation
        Binding("j", "cursor_down", "Next", show=False),
        Binding("k", "cursor_up", "Previous", show=False),
        Binding("down", "cursor_down", "Next Worker", key_display="↓"),
        Binding("up", "cursor_up", "Prev Worker", key_display="↑"),
        Binding("enter", "select_worker", "Select", show=False),
        # Actions
        Binding("n", "new_task", "New Task"),
        Binding("q", "quit", "Quit"),
    ]

    DEFAULT_CSS = """
    SamplesScreen {
        background: $background;
    }

    SamplesScreen #content {
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(id="content"):
            yield WorkersPanel(id="workers-panel")
            yield ConversationPanel(id="conversation-panel")
        yield TmuxBar(active_screen="samples")

    def on_mount(self) -> None:
        """Initial sync from current state."""
        self._sync()
        # Set up periodic time updates
        self.set_interval(1.0, self._tick)

    def on_screen_resume(self) -> None:
        """Re-sync when screen becomes active again."""
        self._sync()

    def refresh_content(self) -> None:
        """Refresh screen content (called by App on new events)."""
        self._sync()

    def _tick(self) -> None:
        """Periodic update for elapsed times."""
        state = get_state()
        extraction = state.selected_extraction
        if extraction and extraction.status == ExtractionStatus.RUNNING:
            # Update worker times
            workers_panel = self.query_one("#workers-panel", WorkersPanel)
            for agent_id, item in workers_panel._items.items():
                agent = extraction.agents.get(agent_id)
                if agent and agent.status == AgentStatus.RUNNING:
                    try:
                        item.query_one("#worker-time", Static).update(agent.elapsed_str)
                    except Exception:
                        pass

            # Update conversation time
            conv_panel = self.query_one("#conversation-panel", ConversationPanel)
            if conv_panel.current_agent_id:
                agent = extraction.agents.get(conv_panel.current_agent_id)
                if agent:
                    conv_panel.update_time(agent)

        self.query_one(TmuxBar).refresh_info()

    def _sync(self) -> None:
        """Sync UI with current state."""
        state = get_state()
        extraction = state.selected_extraction

        workers_panel = self.query_one("#workers-panel", WorkersPanel)
        conv_panel = self.query_one("#conversation-panel", ConversationPanel)

        if extraction is None:
            workers_panel.update_workers({}, None)
            conv_panel.show_agent(None)
        else:
            workers_panel.update_workers(extraction.agents, extraction.selected_agent_id)
            conv_panel.show_agent(extraction.selected_agent)

        self.query_one(TmuxBar).refresh_info()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle worker selection from ListView."""
        if isinstance(event.item, WorkerListItem):
            state = get_state()
            extraction = state.selected_extraction
            if extraction:
                extraction.select_agent(event.item.agent_id)
                self._sync()

    def on_message_row_clicked(self, event: MessageRow.Clicked) -> None:
        """Handle message click to open detail modal."""
        self.app.push_screen(MessageDetailModal(event.msg))

    def action_noop(self) -> None:
        pass

    def action_go_dashboard(self) -> None:
        self.app.switch_screen("dashboard")

    def action_go_logs(self) -> None:
        self.app.switch_screen("logs")

    def action_cycle(self) -> None:
        self.app.switch_screen("logs")

    def action_cursor_down(self) -> None:
        """Move cursor down in worker list."""
        try:
            list_view = self.query_one("#workers-list", ListView)
            list_view.action_cursor_down()
        except Exception:
            pass

    def action_cursor_up(self) -> None:
        """Move cursor up in worker list."""
        try:
            list_view = self.query_one("#workers-list", ListView)
            list_view.action_cursor_up()
        except Exception:
            pass

    def action_select_worker(self) -> None:
        """Select the currently highlighted worker."""
        try:
            list_view = self.query_one("#workers-list", ListView)
            list_view.action_select_cursor()
        except Exception:
            pass

    def action_new_task(self) -> None:
        self.app.push_screen("create_task")

    def action_quit(self) -> None:
        self.app.exit()
