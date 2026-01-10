"""Samples screen - split view of parallel extraction samples."""

from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Static, RichLog
from textual.reactive import reactive

from vector_forge.ui.state import (
    AgentUIState,
    AgentStatus,
    ExtractionStatus,
    get_state,
)
from vector_forge.ui.theme import COLORS, ICONS


class SampleItem(Static):
    """A single sample item in the list."""

    DEFAULT_CSS = """
    SampleItem {
        height: 4;
        padding: 0 1;
        background: transparent;
    }

    SampleItem:hover {
        background: $surface-hl;
    }

    SampleItem.-selected {
        background: $surface;
        border-left: wide $accent;
    }

    SampleItem.-running .sample-status {
        color: $accent;
    }

    SampleItem.-complete .sample-status {
        color: $success;
    }

    SampleItem.-error .sample-status {
        color: $error;
    }

    SampleItem .sample-row {
        height: 1;
    }
    """

    selected: reactive[bool] = reactive(False)

    def __init__(self, agent: AgentUIState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.agent = agent
        self.agent_id = agent.id

    def compose(self) -> ComposeResult:
        yield Static(classes="sample-row sample-name")
        yield Static(classes="sample-row sample-status")
        yield Static(classes="sample-row sample-progress")
        yield Static(classes="sample-row sample-time")

    def on_mount(self) -> None:
        self._refresh()

    def on_click(self) -> None:
        self.post_message(SamplesList.SampleClicked(self.agent_id))

    def watch_selected(self, selected: bool) -> None:
        self.set_class(selected, "-selected")

    def update_agent(self, agent: AgentUIState) -> None:
        self.agent = agent
        if self.is_mounted:
            self._refresh()

    def _refresh(self) -> None:
        agent = self.agent

        # Status class
        self.remove_class("-running", "-complete", "-error", "-idle")
        status_class = {
            AgentStatus.RUNNING: "-running",
            AgentStatus.COMPLETE: "-complete",
            AgentStatus.ERROR: "-error",
        }.get(agent.status, "-idle")
        self.add_class(status_class)

        # Status icon
        status_info = {
            AgentStatus.IDLE: (ICONS.pending, COLORS.text_dim, "idle"),
            AgentStatus.RUNNING: (ICONS.running, COLORS.accent, "running"),
            AgentStatus.WAITING: (ICONS.waiting, COLORS.text_muted, "waiting"),
            AgentStatus.COMPLETE: (ICONS.complete, COLORS.success, "done"),
            AgentStatus.ERROR: (ICONS.failed, COLORS.error, "error"),
        }
        icon, color, label = status_info.get(agent.status, (ICONS.pending, COLORS.text_dim, "?"))

        # Update display
        name = self.query_one(".sample-name", Static)
        name.update(f"[bold]{agent.name}[/]")

        status = self.query_one(".sample-status", Static)
        status.update(f"[{color}]{icon}[/] {label}")

        progress = self.query_one(".sample-progress", Static)
        progress.update(f"[{COLORS.text_dim}]{agent.turns} turns · {agent.tool_calls_count} tools[/]")

        time_widget = self.query_one(".sample-time", Static)
        time_widget.update(f"[{COLORS.text_dim}]{agent.elapsed_str}[/]")


class SamplesList(Vertical):
    """List of running samples."""

    DEFAULT_CSS = """
    SamplesList {
        width: 1fr;
        min-width: 30;
        max-width: 50;
        background: $surface;
        border-right: solid $border;
    }

    SamplesList #samples-header {
        height: 3;
        padding: 1;
        background: $surface;
        border-bottom: solid $border;
    }

    SamplesList #samples-title {
        text-style: bold;
    }

    SamplesList #samples-count {
        color: $text-muted;
    }

    SamplesList #samples-scroll {
        height: 1fr;
    }

    SamplesList .samples-empty-msg {
        padding: 2;
        color: $text-muted;
        text-align: center;
    }
    """

    class SampleClicked(Message):
        """Sample was clicked."""
        def __init__(self, agent_id: str) -> None:
            super().__init__()
            self.agent_id = agent_id

    def compose(self) -> ComposeResult:
        with Vertical(id="samples-header"):
            yield Static("Samples", id="samples-title")
            yield Static(id="samples-count")
        yield VerticalScroll(id="samples-scroll")

    def set_samples(self, agents: dict, selected_id: Optional[str] = None) -> None:
        """Update the samples list."""
        scroll = self.query_one("#samples-scroll", VerticalScroll)

        # Update count
        count = self.query_one("#samples-count", Static)
        running = sum(1 for a in agents.values() if a.status == AgentStatus.RUNNING)
        total = len(agents)
        if total == 0:
            count.update("No samples")
        elif running > 0:
            count.update(f"{running} running / {total} total")
        else:
            count.update(f"{total} samples")

        # Get existing items
        existing = {item.agent_id: item for item in scroll.query(SampleItem)}

        # Update or create items
        for agent_id, agent in agents.items():
            if agent_id in existing:
                existing[agent_id].update_agent(agent)
                existing[agent_id].selected = agent_id == selected_id
            else:
                item = SampleItem(agent)
                item.selected = agent_id == selected_id
                scroll.mount(item)

        # Remove stale items
        for agent_id, item in existing.items():
            if agent_id not in agents:
                item.remove()

        # Handle empty state
        empty_widgets = list(scroll.query(".samples-empty-msg"))
        if agents:
            for w in empty_widgets:
                w.remove()
        elif not empty_widgets:
            scroll.mount(Static("No samples running", classes="samples-empty-msg"))

    def select(self, agent_id: str) -> None:
        """Select a sample."""
        for item in self.query(SampleItem):
            item.selected = item.agent_id == agent_id


class SampleDetails(Vertical):
    """Details panel for selected sample."""

    DEFAULT_CSS = """
    SampleDetails {
        width: 2fr;
        background: $background;
    }

    SampleDetails #details-header {
        height: 4;
        padding: 1;
        background: $surface;
        border-bottom: solid $border;
    }

    SampleDetails #details-title {
        text-style: bold;
    }

    SampleDetails #details-status {
        color: $text-muted;
    }

    SampleDetails #details-stats {
        color: $text-muted;
    }

    SampleDetails #details-log {
        height: 1fr;
        padding: 1;
        background: $background;
    }

    SampleDetails #details-empty {
        height: 1fr;
        content-align: center middle;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="details-header"):
            yield Static("Select a sample", id="details-title")
            yield Static(id="details-status")
            yield Static(id="details-stats")
        yield RichLog(id="details-log", highlight=True, markup=True)

    def set_agent(self, agent: Optional[AgentUIState]) -> None:
        """Update with agent details."""
        title = self.query_one("#details-title", Static)
        status = self.query_one("#details-status", Static)
        stats = self.query_one("#details-stats", Static)
        log = self.query_one("#details-log", RichLog)

        if agent is None:
            title.update(f"[{COLORS.text_muted}]Select a sample[/]")
            status.update(f"[{COLORS.text_dim}]Click a sample to view details[/]")
            stats.update("")
            log.clear()
            return

        # Status styling
        status_info = {
            AgentStatus.IDLE: (ICONS.pending, COLORS.text_dim, "idle"),
            AgentStatus.RUNNING: (ICONS.running, COLORS.accent, "running"),
            AgentStatus.WAITING: (ICONS.waiting, COLORS.text_muted, "waiting"),
            AgentStatus.COMPLETE: (ICONS.complete, COLORS.success, "complete"),
            AgentStatus.ERROR: (ICONS.failed, COLORS.error, "error"),
        }
        icon, color, label = status_info.get(agent.status, (ICONS.pending, COLORS.text_dim, "?"))

        title.update(f"[{color}]{icon}[/] [bold]{agent.name}[/]")
        status.update(f"{agent.role} · {label}")
        stats.update(
            f"{len(agent.messages)} messages · {agent.turns} turns · "
            f"{agent.tool_calls_count} tools · {agent.elapsed_str}"
        )

        # Update log with messages
        log.clear()
        for msg in agent.messages:
            role_colors = {
                "system": COLORS.purple,
                "user": COLORS.blue,
                "assistant": COLORS.accent,
                "tool": COLORS.aqua,
            }
            role_color = role_colors.get(msg.role.value, COLORS.text_dim)

            # Header
            log.write(f"[{role_color} bold]{msg.role.value.upper()}[/] [{COLORS.text_dim}]{msg.time_str}[/]")

            # Content (truncated)
            content = msg.content
            if len(content) > 500:
                content = content[:497] + "..."
            if content:
                log.write(f"  {content}")

            # Tool calls
            for tc in msg.tool_calls:
                tc_color = {
                    "pending": COLORS.text_muted,
                    "running": COLORS.accent,
                    "success": COLORS.success,
                    "error": COLORS.error,
                }.get(tc.status, COLORS.text_dim)

                duration = f" ({tc.duration_ms}ms)" if tc.duration_ms else ""
                log.write(f"  [{tc_color}]{ICONS.active} {tc.name}[/]{duration}")

            log.write("")


class SamplesScreen(Screen):
    """Split view screen showing parallel samples."""

    BINDINGS = [
        Binding("1", "switch_dashboard", "dashboard"),
        Binding("2", "noop", "samples", show=False),
        Binding("3", "switch_logs", "logs"),
        Binding("tab", "cycle_screen", "cycle", show=False),
        Binding("q", "quit", "quit"),
        Binding("n", "new_task", "new task"),
        Binding("j", "select_next", "next", show=False),
        Binding("k", "select_prev", "prev", show=False),
        Binding("down", "select_next", "next", show=False),
        Binding("up", "select_prev", "prev", show=False),
        Binding("?", "show_help", "help"),
    ]

    DEFAULT_CSS = """
    SamplesScreen {
        background: $background;
    }

    SamplesScreen #header {
        height: 3;
        padding: 1 2;
        background: $surface;
        border-bottom: solid $border;
    }

    SamplesScreen #header-title {
        width: 1fr;
        text-style: bold;
    }

    SamplesScreen #header-task {
        width: auto;
        color: $text-muted;
    }

    SamplesScreen #content {
        height: 1fr;
    }

    SamplesScreen #footer {
        dock: bottom;
        height: 1;
        background: $surface;
        padding: 0 2;
    }

    SamplesScreen #footer-left {
        width: 1fr;
    }

    SamplesScreen #footer-right {
        width: auto;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(id="header"):
            yield Static("Samples", id="header-title")
            yield Static(id="header-task")

        with Horizontal(id="content"):
            yield SamplesList(id="samples-list")
            yield SampleDetails(id="sample-details")

        with Horizontal(id="footer"):
            yield Static(id="footer-left")
            yield Static("j/k: navigate  |  1: dashboard  |  3: logs  |  n: new task", id="footer-right")

    def on_mount(self) -> None:
        self._sync_from_state()

        state = get_state()
        state.add_listener(self._on_state_changed)
        self.set_interval(1.0, self._refresh_timers)

    def on_unmount(self) -> None:
        state = get_state()
        state.remove_listener(self._on_state_changed)

    def _on_state_changed(self, state) -> None:
        self._sync_from_state()

    def _sync_from_state(self) -> None:
        state = get_state()
        extraction = state.selected_extraction

        # Update header
        task_label = self.query_one("#header-task", Static)
        if extraction:
            task_label.update(f"Task: {extraction.behavior_name}")
        else:
            task_label.update("No task selected")

        # Update samples list
        samples_list = self.query_one("#samples-list", SamplesList)
        details = self.query_one("#sample-details", SampleDetails)

        if extraction is None:
            samples_list.set_samples({}, None)
            details.set_agent(None)
            return

        samples_list.set_samples(extraction.agents, extraction.selected_agent_id)
        details.set_agent(extraction.selected_agent)

        # Update footer
        footer_left = self.query_one("#footer-left", Static)
        running = extraction.running_agents_count
        total = extraction.total_agents_count
        footer_left.update(f"[{COLORS.accent}]{running}[/] running / {total} total · {extraction.elapsed_str}")

    def _refresh_timers(self) -> None:
        state = get_state()
        extraction = state.selected_extraction
        if extraction and extraction.status == ExtractionStatus.RUNNING:
            self._sync_from_state()

    def on_samples_list_sample_clicked(self, message: SamplesList.SampleClicked) -> None:
        """Handle sample selection."""
        state = get_state()
        extraction = state.selected_extraction
        if extraction:
            extraction.select_agent(message.agent_id)
            self._sync_from_state()

    def action_noop(self) -> None:
        pass

    def action_switch_dashboard(self) -> None:
        self.app.switch_screen("dashboard")

    def action_switch_logs(self) -> None:
        self.app.switch_screen("logs")

    def action_cycle_screen(self) -> None:
        self.app.switch_screen("logs")

    def action_select_next(self) -> None:
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

        self._sync_from_state()

    def action_select_prev(self) -> None:
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

        self._sync_from_state()

    def action_new_task(self) -> None:
        self.app.push_screen("create_task")

    def action_show_help(self) -> None:
        self.app.push_screen("help")

    def action_quit(self) -> None:
        self.app.exit()
