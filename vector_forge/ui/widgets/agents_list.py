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

    def _get_display_content(self) -> tuple:
        """Get display content for the three lines."""
        agent = self.agent
        status_map = {
            AgentStatus.IDLE: (ICONS.pending, "$foreground-disabled"),
            AgentStatus.RUNNING: (ICONS.running, "$accent"),
            AgentStatus.WAITING: (ICONS.waiting, "$foreground-muted"),
            AgentStatus.COMPLETE: (ICONS.complete, "$success"),
            AgentStatus.ERROR: (ICONS.failed, "$error"),
        }
        icon, color = status_map.get(agent.status, (ICONS.pending, "$foreground-disabled"))

        # Line 1: Status icon and name
        line1 = (
            f"[{color}]{icon}[/] [$foreground bold]{agent.name}[/] "
            f"[$foreground-disabled]({agent.role})[/]"
        )

        # Line 2: Current activity or tool
        if agent.status == AgentStatus.RUNNING and agent.current_tool:
            line2 = f"  [$accent]{ICONS.active}[/] [$foreground-muted]{agent.current_tool}[/]"
        elif agent.last_message:
            content = agent.last_message.content
            if len(content) > 40:
                content = content[:37] + "..."
            line2 = f"  [$foreground-disabled]{content}[/]"
        else:
            line2 = "  [$foreground-disabled]No activity[/]"

        # Line 3: Stats
        line3 = (
            f"  [$foreground-disabled]{agent.turns} turns · "
            f"{agent.tool_calls_count} tools · {agent.elapsed_str}[/]"
        )

        return line1, line2, line3

    def compose(self) -> ComposeResult:
        line1, line2, line3 = self._get_display_content()
        yield Static(line1, id="agent-line1", classes="agent-row")
        yield Static(line2, id="agent-line2", classes="agent-row")
        yield Static(line3, id="agent-line3", classes="agent-row")

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
        line1, line2, line3 = self._get_display_content()
        self.query_one("#agent-line1", Static).update(line1)
        self.query_one("#agent-line2", Static).update(line2)
        self.query_one("#agent-line3", Static).update(line3)


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

    selected_id: reactive[Optional[str]] = reactive(None, init=False)

    class SelectionChanged(Message):
        """Message sent when selection changes."""

        def __init__(self, agent_id: Optional[str]) -> None:
            super().__init__()
            self.agent_id = agent_id

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._agents: Dict[str, AgentUIState] = {}
        self._agent_items: Dict[str, AgentItem] = {}

    def compose(self) -> ComposeResult:
        with Widget(id="agents-header"):
            yield Static("Agents", id="agents-title")
            yield Static("", id="agents-subtitle")
        yield VerticalScroll(id="agents-scroll")

    def on_mount(self) -> None:
        self._update_subtitle()

    def watch_selected_id(self, selected_id: Optional[str]) -> None:
        if not self.is_mounted:
            return
        for agent_id, item in self._agent_items.items():
            item.selected = agent_id == selected_id

    def on_agent_item_selected(self, message: AgentItem.Selected) -> None:
        self.selected_id = message.agent_id
        self.post_message(self.SelectionChanged(message.agent_id))
        message.stop()

    def _update_subtitle(self) -> None:
        """Update the subtitle with agent counts."""
        subtitle = self.query_one("#agents-subtitle", Static)
        total = len(self._agents)
        running = sum(1 for a in self._agents.values() if a.status == AgentStatus.RUNNING)
        if total == 0:
            subtitle.update("No agents")
        elif running > 0:
            subtitle.update(f"{running} running · {total} total")
        else:
            subtitle.update(f"{total} total")

    def set_agents(self, agents: Dict[str, AgentUIState]) -> None:
        """Set agents and update the list incrementally."""
        self._agents = agents

        if not self.is_mounted:
            return

        self._update_subtitle()

        scroll = self.query_one("#agents-scroll", VerticalScroll)
        current_ids = set(agents.keys())
        existing_ids = set(self._agent_items.keys())

        # Remove agents that no longer exist
        for agent_id in existing_ids - current_ids:
            if agent_id in self._agent_items:
                self._agent_items[agent_id].remove()
                del self._agent_items[agent_id]

        # Update existing or add new agents
        for agent_id, agent in agents.items():
            if agent_id in self._agent_items:
                # Update existing item in place
                self._agent_items[agent_id].update_agent(agent)
            else:
                # Add new item
                item = AgentItem(agent)
                item.selected = agent_id == self.selected_id
                scroll.mount(item)
                self._agent_items[agent_id] = item

        # Handle empty state
        empty_widgets = list(scroll.query("#agents-empty"))
        if not agents:
            if not empty_widgets:
                scroll.mount(Static("No agents running", id="agents-empty"))
        else:
            for w in empty_widgets:
                w.remove()

    @property
    def agents(self) -> Dict[str, AgentUIState]:
        """Get current agents."""
        return self._agents

    def select_agent(self, agent_id: str) -> None:
        """Select an agent."""
        if agent_id in self._agents:
            self.selected_id = agent_id
            self.post_message(self.SelectionChanged(agent_id))

    def select_next(self) -> None:
        """Select next agent."""
        if not self._agents:
            return
        ids = list(self._agents.keys())
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
        if not self._agents:
            return
        ids = list(self._agents.keys())
        if self.selected_id is None:
            self.select_agent(ids[-1])
        else:
            try:
                idx = ids.index(self.selected_id)
                prev_idx = (idx - 1) % len(ids)
                self.select_agent(ids[prev_idx])
            except ValueError:
                self.select_agent(ids[-1])
