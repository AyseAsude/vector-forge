"""Samples screen - split view of parallel extraction workers."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen, ModalScreen
from textual.widgets import Static

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

    ToolCallModal #modal-title {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
    }

    ToolCallModal #modal-status {
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

    ToolCallModal #modal-args {
        height: auto;
        max-height: 10;
        padding: 1;
        background: $background;
        margin-bottom: 1;
    }

    ToolCallModal #modal-result {
        height: auto;
        max-height: 15;
        padding: 1;
        background: $background;
    }

    ToolCallModal #modal-footer {
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
        with Vertical(id="modal"):
            yield Static(id="modal-title")
            yield Static(id="modal-status")

            yield Static("ARGUMENTS", classes="section")
            yield Static(id="modal-args")

            # Result section added dynamically in on_mount if needed
            yield Static(id="modal-result-section", classes="section")
            yield Static(id="modal-result")

            yield Static("Press ESC to close", id="modal-footer")

    def on_mount(self) -> None:
        tc = self.tool_call

        status_colors = {
            "pending": "$foreground-muted",
            "running": "$accent",
            "success": "$success",
            "error": "$error",
        }
        color = status_colors.get(tc.status, "$foreground-disabled")
        duration = f" · {tc.duration_ms}ms" if tc.duration_ms else ""

        self.query_one("#modal-title", Static).update(f"[{color}]▸[/] [bold]{tc.name}[/]")
        self.query_one("#modal-status", Static).update(f"{tc.status}{duration}")

        args_text = tc.arguments if tc.arguments else "(none)"
        if len(args_text) > 500:
            args_text = args_text[:497] + "..."
        self.query_one("#modal-args", Static).update(args_text)

        result_section = self.query_one("#modal-result-section", Static)
        result_widget = self.query_one("#modal-result", Static)
        if tc.result:
            result_section.update("RESULT")
            result_text = tc.result
            if len(result_text) > 1000:
                result_text = result_text[:997] + "..."
            result_widget.update(result_text)
        else:
            result_section.display = False
            result_widget.display = False


class ToolCallRow(Static):
    """Clickable tool call row."""

    DEFAULT_CSS = """
    ToolCallRow {
        height: 1;
        padding: 0 2;
    }

    ToolCallRow:hover {
        background: $boost;
    }
    """

    class Clicked(Message):
        def __init__(self, tool_call: ToolCall) -> None:
            super().__init__()
            self.tool_call = tool_call

    def __init__(self, tool_call: ToolCall, **kwargs) -> None:
        self.tool_call = tool_call
        super().__init__(**kwargs)

    def on_mount(self) -> None:
        self._update_display()

    def _update_display(self) -> None:
        tc = self.tool_call

        status_colors = {
            "pending": "$foreground-muted",
            "running": "$accent",
            "success": "$success",
            "error": "$error",
        }
        color = status_colors.get(tc.status, "$foreground-disabled")
        duration = f" ({tc.duration_ms}ms)" if tc.duration_ms else ""

        self.update(f"  [{color}]▸ {tc.name}[/]{duration}")

    def on_click(self) -> None:
        self.post_message(self.Clicked(self.tool_call))


class MessageBlock(Vertical):
    """A single message in the conversation."""

    DEFAULT_CSS = """
    MessageBlock {
        height: auto;
        margin-bottom: 1;
        margin-right: 2;
    }

    MessageBlock .msg-header {
        height: 1;
    }

    MessageBlock .msg-content {
        height: auto;
        padding: 0 2;
        color: $foreground;
    }
    """

    def __init__(self, message: AgentMessage, **kwargs) -> None:
        super().__init__(**kwargs)
        self.message = message

    def compose(self) -> ComposeResult:
        msg = self.message

        role_colors = {
            "system": "$secondary",
            "user": "$primary",
            "assistant": "$accent",
            "tool": "$success",
        }
        role_color = role_colors.get(msg.role.value, "$foreground-disabled")

        yield Static(
            f"[{role_color} bold]{msg.role.value.upper()}[/] [$foreground-disabled]{msg.time_str}[/]",
            classes="msg-header"
        )

        content = msg.content
        if len(content) > 500:
            content = content[:497] + "..."
        yield Static(content if content else "", classes="msg-content")

        for tc in msg.tool_calls:
            yield ToolCallRow(tc)


class WorkerCard(Static):
    """Worker card in the list."""

    DEFAULT_CSS = """
    WorkerCard {
        height: auto;
        padding: 1;
        margin-bottom: 1;
        margin-right: 2;
        background: $boost;
    }

    WorkerCard:hover {
        background: $primary 15%;
    }

    WorkerCard.-selected {
        background: $primary 20%;
    }

    WorkerCard.-selected:hover {
        background: $primary 25%;
    }

    WorkerCard .header {
        height: 1;
        margin-bottom: 1;
    }

    WorkerCard .name {
        width: 1fr;
    }

    WorkerCard .time {
        width: auto;
        color: $foreground-muted;
    }

    WorkerCard .meta {
        height: 1;
        color: $foreground-muted;
    }
    """

    class Selected(Message):
        def __init__(self, agent_id: str) -> None:
            super().__init__()
            self.agent_id = agent_id

    def __init__(self, agent: AgentUIState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.agent = agent

    def _get_display_values(self) -> tuple:
        """Get display values for the agent."""
        status_map = {
            AgentStatus.IDLE: (ICONS.pending, "$foreground-muted", "IDLE"),
            AgentStatus.RUNNING: (ICONS.running, "$accent", "RUNNING"),
            AgentStatus.WAITING: (ICONS.waiting, "$foreground-muted", "WAITING"),
            AgentStatus.COMPLETE: (ICONS.complete, "$success", "DONE"),
            AgentStatus.ERROR: (ICONS.failed, "$error", "ERROR"),
        }
        return status_map.get(self.agent.status, (ICONS.pending, "$foreground-muted", "?"))

    def compose(self) -> ComposeResult:
        agent = self.agent
        icon, color, label = self._get_display_values()

        with Horizontal(classes="header"):
            yield Static(f"[{color}]{icon}[/] [bold]{agent.name}[/]", classes="name")
            yield Static(agent.elapsed_str, classes="time")
        yield Static(
            f"[{color}]{label}[/] · {agent.turns} turns · {agent.tool_calls_count} tools",
            classes="meta"
        )

    def on_click(self) -> None:
        self.post_message(self.Selected(self.agent.id))

    def update(self, agent: AgentUIState) -> None:
        self.agent = agent
        if self.is_mounted:
            self._update_display()

    def _update_display(self) -> None:
        agent = self.agent
        icon, color, label = self._get_display_values()

        self.query_one(".name", Static).update(f"[{color}]{icon}[/] [bold]{agent.name}[/]")
        self.query_one(".time", Static).update(agent.elapsed_str)
        self.query_one(".meta", Static).update(
            f"[{color}]{label}[/] · {agent.turns} turns · {agent.tool_calls_count} tools"
        )


class WorkersList(Vertical):
    """Left panel with list of workers."""

    DEFAULT_CSS = """
    WorkersList {
        width: 1fr;
        max-width: 40;
        padding: 1 0 1 2;
        background: $surface;
    }

    WorkersList .header {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
        padding-right: 2;
    }

    WorkersList .count {
        height: 1;
        color: $foreground-muted;
        margin-bottom: 1;
        padding-right: 2;
    }

    WorkersList .list {
        height: 1fr;
    }

    WorkersList .empty {
        padding: 2;
        color: $foreground-muted;
        text-align: center;
        margin-right: 2;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("PARALLEL RUNS", classes="header")
        yield Static(classes="count")
        yield VerticalScroll(classes="list")

    def set_workers(self, agents: dict, selected_id: str | None = None) -> None:
        count_widget = self.query_one(".count", Static)
        scroll = self.query_one(".list", VerticalScroll)

        running = sum(1 for a in agents.values() if a.status == AgentStatus.RUNNING)
        total = len(agents)
        if total == 0:
            count_widget.update("No workers")
        elif running > 0:
            count_widget.update(f"{running} active / {total} total")
        else:
            count_widget.update(f"{total} workers")

        existing = {c.agent.id: c for c in scroll.query(WorkerCard)}

        for agent_id, agent in agents.items():
            if agent_id in existing:
                existing[agent_id].update(agent)
                existing[agent_id].set_class(agent_id == selected_id, "-selected")
            else:
                card = WorkerCard(agent)
                card.set_class(agent_id == selected_id, "-selected")
                scroll.mount(card)

        for agent_id, card in existing.items():
            if agent_id not in agents:
                card.remove()

        empties = list(scroll.query(".empty"))
        if agents:
            for e in empties:
                e.remove()
        elif not empties:
            scroll.mount(Static("No workers running", classes="empty"))


class ConversationPanel(Vertical):
    """Right panel showing worker conversation."""

    DEFAULT_CSS = """
    ConversationPanel {
        width: 2fr;
        background: $surface;
    }

    ConversationPanel .panel-header {
        height: auto;
        padding: 1 2;
        background: $panel;
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

    ConversationPanel .messages {
        height: 1fr;
        padding: 1 0 1 2;
        background: $background;
    }

    ConversationPanel .empty {
        height: 1fr;
        content-align: center middle;
        color: $foreground-muted;
        margin-right: 2;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._current_agent_id: str | None = None
        self._message_count: int = 0

    def compose(self) -> ComposeResult:
        with Vertical(classes="panel-header"):
            with Horizontal(classes="title-row"):
                yield Static(classes="title")
                yield Static(classes="time")
            yield Static(classes="stats")
        yield VerticalScroll(classes="messages")

    def show(self, agent: AgentUIState | None, force_rebuild: bool = False) -> None:
        title = self.query_one(".title", Static)
        time_widget = self.query_one(".time", Static)
        stats = self.query_one(".stats", Static)
        messages = self.query_one(".messages", VerticalScroll)

        if agent is None:
            if self._current_agent_id is not None or force_rebuild:
                title.update("[$foreground-muted]Select a worker[/]")
                time_widget.update("")
                stats.update("[$foreground-disabled]Click a worker to view conversation[/]")
                messages.remove_children()
                messages.mount(Static("", classes="empty"))
                self._current_agent_id = None
                self._message_count = 0
            return

        status_map = {
            AgentStatus.IDLE: (ICONS.pending, "$foreground-muted", "IDLE"),
            AgentStatus.RUNNING: (ICONS.running, "$accent", "RUNNING"),
            AgentStatus.WAITING: (ICONS.waiting, "$foreground-muted", "WAITING"),
            AgentStatus.COMPLETE: (ICONS.complete, "$success", "DONE"),
            AgentStatus.ERROR: (ICONS.failed, "$error", "ERROR"),
        }
        icon, color, label = status_map.get(agent.status, (ICONS.pending, "$foreground-muted", "?"))

        # Header: icon + name on left, time on right
        title.update(f"[{color}]{icon}[/] [bold]{agent.name}[/]")
        time_widget.update(f"[$foreground-muted]{agent.elapsed_str}[/]")

        # Stats: status + role + counts
        stats.update(
            f"[{color}]{label}[/] · {agent.role} · "
            f"{len(agent.messages)} msgs · {agent.turns} turns · {agent.tool_calls_count} tools"
        )

        # Only rebuild messages if agent changed or message count changed
        agent_changed = agent.id != self._current_agent_id
        messages_changed = len(agent.messages) != self._message_count

        if agent_changed or messages_changed or force_rebuild:
            messages.remove_children()
            for msg in agent.messages:
                messages.mount(MessageBlock(msg))
            messages.scroll_end(animate=False)

            self._current_agent_id = agent.id
            self._message_count = len(agent.messages)


class SamplesScreen(Screen):
    """Split view screen showing parallel workers."""

    BINDINGS = [
        # Navigation between screens
        Binding("1", "go_dashboard", "Dashboard", key_display="1"),
        Binding("2", "noop", "Samples", show=False),  # Current screen
        Binding("3", "go_logs", "Logs", key_display="3"),
        Binding("tab", "cycle", "Next Screen"),
        # List navigation
        Binding("j", "next", "Next", show=False),
        Binding("k", "prev", "Previous", show=False),
        Binding("down", "next", "Next Worker", key_display="↓"),
        Binding("up", "prev", "Prev Worker", key_display="↑"),
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
            yield WorkersList(id="workers-list")
            yield ConversationPanel(id="conversation")
        yield TmuxBar(active_screen="samples")

    def on_mount(self) -> None:
        get_state().add_listener(self._on_state_change)
        self._sync()
        self.set_interval(1.0, self._tick)

    def on_unmount(self) -> None:
        get_state().remove_listener(self._on_state_change)

    def _on_state_change(self, _) -> None:
        self._sync()

    def _tick(self) -> None:
        state = get_state()
        extraction = state.selected_extraction
        if extraction and extraction.status == ExtractionStatus.RUNNING:
            self._sync()

    def _sync(self) -> None:
        state = get_state()
        extraction = state.selected_extraction

        workers_list = self.query_one("#workers-list", WorkersList)
        conversation = self.query_one("#conversation", ConversationPanel)

        if extraction is None:
            workers_list.set_workers({}, None)
            conversation.show(None)
        else:
            workers_list.set_workers(extraction.agents, extraction.selected_agent_id)
            conversation.show(extraction.selected_agent)

        self.query_one(TmuxBar).refresh_info()

    def on_worker_card_selected(self, event: WorkerCard.Selected) -> None:
        state = get_state()
        extraction = state.selected_extraction
        if extraction:
            extraction.select_agent(event.agent_id)
            self._sync()

    def on_tool_call_row_clicked(self, event: ToolCallRow.Clicked) -> None:
        self.app.push_screen(ToolCallModal(event.tool_call))

    def action_noop(self) -> None:
        pass

    def action_go_dashboard(self) -> None:
        self.app.switch_screen("dashboard")

    def action_go_logs(self) -> None:
        self.app.switch_screen("logs")

    def action_cycle(self) -> None:
        self.app.switch_screen("logs")

    def action_next(self) -> None:
        state = get_state()
        extraction = state.selected_extraction
        if not extraction or not extraction.agents:
            return

        ids = list(extraction.agents.keys())
        if extraction.selected_agent_id is None:
            extraction.select_agent(ids[0])
        else:
            try:
                idx = ids.index(extraction.selected_agent_id)
                extraction.select_agent(ids[(idx + 1) % len(ids)])
            except ValueError:
                extraction.select_agent(ids[0])
        self._sync()

    def action_prev(self) -> None:
        state = get_state()
        extraction = state.selected_extraction
        if not extraction or not extraction.agents:
            return

        ids = list(extraction.agents.keys())
        if extraction.selected_agent_id is None:
            extraction.select_agent(ids[-1])
        else:
            try:
                idx = ids.index(extraction.selected_agent_id)
                extraction.select_agent(ids[(idx - 1) % len(ids)])
            except ValueError:
                extraction.select_agent(ids[-1])
        self._sync()

    def action_new_task(self) -> None:
        self.app.push_screen("create_task")

    def action_quit(self) -> None:
        self.app.exit()
