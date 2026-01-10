"""Samples screen - split view of parallel extraction workers.

Uses native Textual widgets for high performance:
- RichLog for conversation display (efficient appending, native scrolling)
- ListView for worker selection (native keyboard navigation)
- Reactive updates instead of remove/remount patterns
"""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.screen import Screen, ModalScreen
from textual.widgets import Static, RichLog, ListView, ListItem, Label

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
        if len(args_text) > 500:
            args_text = args_text[:497] + "..."

        with Vertical(id="modal"):
            yield Static(f"[{color}]▸[/] [bold]{tc.name}[/]", classes="title")
            yield Static(f"{tc.status}{duration}", classes="status")
            yield Static("ARGUMENTS", classes="section")
            yield Static(args_text, classes="content")

            if tc.result:
                result_text = tc.result
                if len(result_text) > 1000:
                    result_text = result_text[:997] + "..."
                yield Static("RESULT", classes="section")
                yield Static(result_text, classes="content")

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

    WorkerListItem.-highlight {
        background: $primary 15%;
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
            f"[{color}]{label}[/] · {agent.turns} turns · {agent.tool_calls_count} tools",
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
                    f"[{color}]{label}[/] · {agent.turns} turns · {agent.tool_calls_count} tools"
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

        # Update selection highlighting
        for agent_id, item in self._items.items():
            item.set_class(agent_id == selected_id, "-highlight")

    def get_selected_agent_id(self) -> str | None:
        """Get the currently highlighted worker's agent ID."""
        list_view = self.query_one("#workers-list", ListView)
        if list_view.highlighted_child is not None:
            item = list_view.highlighted_child
            if isinstance(item, WorkerListItem):
                return item.agent_id
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Conversation Panel - Uses native RichLog
# ─────────────────────────────────────────────────────────────────────────────


class ConversationPanel(Vertical):
    """Right panel showing worker conversation using native RichLog."""

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

    ConversationPanel RichLog {
        height: 1fr;
        padding: 1 2;
        background: $background;
        overflow-x: hidden;
        overflow-y: auto;
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
        yield RichLog(id="conversation-log", highlight=True, markup=True, wrap=True)

    def show_agent(self, agent: AgentUIState | None) -> None:
        """Display an agent's conversation."""
        title = self.query_one("#conv-title", Static)
        time_widget = self.query_one("#conv-time", Static)
        stats = self.query_one("#conv-stats", Static)
        rich_log = self.query_one("#conversation-log", RichLog)

        if agent is None:
            if self.current_agent_id is not None:
                title.update("[$foreground-muted]Select a worker[/]")
                time_widget.update("")
                stats.update("[$foreground-disabled]Click a worker to view conversation[/]")
                rich_log.clear()
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
            f"{len(agent.messages)} msgs · {agent.turns} turns · {agent.tool_calls_count} tools"
        )

        # Check if agent changed
        agent_changed = agent.id != self.current_agent_id
        if agent_changed:
            # Clear and rebuild for new agent
            rich_log.clear()
            self._displayed_message_ids.clear()
            self.current_agent_id = agent.id

        # Append only new messages (incremental update)
        for msg in agent.messages:
            if msg.id not in self._displayed_message_ids:
                self._write_message(rich_log, msg)
                self._displayed_message_ids.add(msg.id)

    def _write_message(self, log: RichLog, msg: AgentMessage) -> None:
        """Write a single message to the RichLog."""
        role_colors = {
            "system": "blue",
            "user": "cyan",
            "assistant": "green",
            "tool": "yellow",
        }
        role_color = role_colors.get(msg.role.value, "white")

        # Write header
        log.write(f"[bold {role_color}]{msg.role.value.upper()}[/] [dim]{msg.time_str}[/]")

        # Write content
        content = msg.content
        if len(content) > 500:
            content = content[:497] + "..."
        if content:
            log.write(f"  {content}")

        # Write tool calls
        for tc in msg.tool_calls:
            tc_colors = {
                "pending": "dim",
                "running": "yellow",
                "success": "green",
                "error": "red",
            }
            tc_color = tc_colors.get(tc.status, "white")
            duration = f" ({tc.duration_ms}ms)" if tc.duration_ms else ""
            log.write(f"  [{tc_color}]▸ {tc.name}[/]{duration}")

        # Empty line for spacing
        log.write("")

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
        get_state().add_listener(self._on_state_change)
        self._sync()
        # Set up periodic time updates
        self.set_interval(1.0, self._tick)

    def on_unmount(self) -> None:
        get_state().remove_listener(self._on_state_change)

    def _on_state_change(self, _) -> None:
        """Handle state changes."""
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
