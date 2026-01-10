"""Agents list widget - shows all agents for the current extraction."""

from typing import Dict, Optional

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from vector_forge.ui.theme import ICONS
from vector_forge.ui.state import AgentUIState, AgentStatus


class AgentItem(Widget):
    """Single agent item in the list."""

    DEFAULT_CSS = """
    AgentItem {
        height: 3;
        padding: 0 1;
        background: transparent;
    }

    AgentItem:hover {
        background: $surface;
    }

    AgentItem.-selected {
        background: $surface;
        border-left: wide $accent;
        padding-left: 0;
    }

    AgentItem.-selected:hover {
        background: $boost;
    }

    AgentItem .agent-row {
        height: 1;
    }
    """

    class Selected(Message):
        """Message sent when this agent is selected."""

        def __init__(self, agent_id: str) -> None:
            super().__init__()
            self.agent_id = agent_id

    selected: reactive[bool] = reactive(False)

    def __init__(self, agent: AgentUIState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.agent = agent
        self.agent_id = agent.id

    def compose(self) -> ComposeResult:
        yield Static(id="agent-line1", classes="agent-row")
        yield Static(id="agent-line2", classes="agent-row")
        yield Static(id="agent-line3", classes="agent-row")

    def on_mount(self) -> None:
        self._update_display()

    def on_click(self) -> None:
        self.post_message(self.Selected(self.agent_id))

    def watch_selected(self, selected: bool) -> None:
        if selected:
            self.add_class("-selected")
        else:
            self.remove_class("-selected")

    def update_agent(self, agent: AgentUIState) -> None:
        self.agent = agent
        if self.is_mounted:
            self._update_display()

    def _update_display(self) -> None:
        agent = self.agent

        # Status icon and color
        status_map = {
            AgentStatus.IDLE: (ICONS.pending, "$foreground-disabled"),
            AgentStatus.RUNNING: (ICONS.running, "$accent"),
            AgentStatus.WAITING: (ICONS.waiting, "$foreground-muted"),
            AgentStatus.COMPLETE: (ICONS.complete, "$success"),
            AgentStatus.ERROR: (ICONS.failed, "$error"),
        }
        icon, color = status_map.get(agent.status, (ICONS.pending, "$foreground-disabled"))

        # Line 1: Status icon and name
        line1 = self.query_one("#agent-line1", Static)
        line1.update(
            f"[{color}]{icon}[/] [$foreground bold]{agent.name}[/] "
            f"[$foreground-disabled]({agent.role})[/]"
        )

        # Line 2: Current activity or tool
        line2 = self.query_one("#agent-line2", Static)
        if agent.status == AgentStatus.RUNNING and agent.current_tool:
            line2.update(f"  [$accent]{ICONS.active}[/] [$foreground-muted]{agent.current_tool}[/]")
        elif agent.last_message:
            # Show truncated last message
            content = agent.last_message.content
            if len(content) > 40:
                content = content[:37] + "..."
            line2.update(f"  [$foreground-disabled]{content}[/]")
        else:
            line2.update("  [$foreground-disabled]No activity[/]")

        # Line 3: Stats
        line3 = self.query_one("#agent-line3", Static)
        line3.update(
            f"  [$foreground-disabled]{agent.turns} turns · "
            f"{agent.tool_calls_count} tools · {agent.elapsed_str}[/]"
        )


class AgentsList(Widget):
    """Scrollable list of agents for the current extraction."""

    DEFAULT_CSS = """
    AgentsList {
        width: 1fr;
        height: 1fr;
        background: $panel;
    }

    AgentsList #agents-header {
        height: 2;
        padding: 0 1;
        background: $panel;
        border-bottom: solid $surface;
    }

    AgentsList #agents-title {
        height: 1;
        color: $foreground;
    }

    AgentsList #agents-subtitle {
        height: 1;
        color: $foreground-muted;
    }

    AgentsList #agents-scroll {
        height: 1fr;
    }

    AgentsList #agents-empty {
        padding: 1;
        color: $foreground-muted;
    }
    """

    agents: reactive[Dict[str, AgentUIState]] = reactive(
        dict, always_update=True, init=False
    )
    selected_id: reactive[Optional[str]] = reactive(None, init=False)

    class SelectionChanged(Message):
        """Message sent when selection changes."""

        def __init__(self, agent_id: Optional[str]) -> None:
            super().__init__()
            self.agent_id = agent_id

    def compose(self) -> ComposeResult:
        with Widget(id="agents-header"):
            yield Static("Agents", id="agents-title")
            yield Static("", id="agents-subtitle")
        yield VerticalScroll(id="agents-scroll")

    def on_mount(self) -> None:
        self._refresh_list()

    def watch_agents(self, agents: Dict[str, AgentUIState]) -> None:
        if self.is_mounted:
            self._update_subtitle()
            self._refresh_list()

    def watch_selected_id(self, selected_id: Optional[str]) -> None:
        if not self.is_mounted:
            return
        for item in self.query(AgentItem):
            item.selected = item.agent_id == selected_id

    def on_agent_item_selected(self, message: AgentItem.Selected) -> None:
        self.selected_id = message.agent_id
        self.post_message(self.SelectionChanged(message.agent_id))
        message.stop()

    def _update_subtitle(self) -> None:
        """Update the subtitle with agent counts."""
        subtitle = self.query_one("#agents-subtitle", Static)
        total = len(self.agents)
        running = sum(1 for a in self.agents.values() if a.status == AgentStatus.RUNNING)
        if total == 0:
            subtitle.update("No agents")
        elif running > 0:
            subtitle.update(f"{running} running · {total} total")
        else:
            subtitle.update(f"{total} total")

    def _refresh_list(self) -> None:
        """Refresh the agent list."""
        scroll = self.query_one("#agents-scroll", VerticalScroll)

        # Clear existing children
        for child in list(scroll.children):
            child.remove()

        if not self.agents:
            scroll.mount(Static("No agents running", id="agents-empty"))
            return

        # Sort agents: running first, then by start time
        sorted_agents = sorted(
            self.agents.values(),
            key=lambda a: (
                0 if a.status == AgentStatus.RUNNING else 1,
                a.started_at or 0,
            ),
        )

        for agent in sorted_agents:
            item = AgentItem(agent)
            item.selected = agent.id == self.selected_id
            scroll.mount(item)

    def select_agent(self, agent_id: str) -> None:
        """Select an agent."""
        if agent_id in self.agents:
            self.selected_id = agent_id
            self.post_message(self.SelectionChanged(agent_id))

    def select_next(self) -> None:
        """Select next agent."""
        if not self.agents:
            return
        ids = list(self.agents.keys())
        if self.selected_id is None:
            self.select_agent(ids[0])
        else:
            try:
                idx = ids.index(self.selected_id)
                next_idx = (idx + 1) % len(ids)
                self.select_agent(ids[next_idx])
            except ValueError:
                self.select_agent(ids[0])

    def select_previous(self) -> None:
        """Select previous agent."""
        if not self.agents:
            return
        ids = list(self.agents.keys())
        if self.selected_id is None:
            self.select_agent(ids[-1])
        else:
            try:
                idx = ids.index(self.selected_id)
                prev_idx = (idx - 1) % len(ids)
                self.select_agent(ids[prev_idx])
            except ValueError:
                self.select_agent(ids[-1])
