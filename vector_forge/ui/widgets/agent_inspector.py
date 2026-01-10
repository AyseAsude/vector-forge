"""Agent inspector widget - shows messages and tool calls for selected agent."""

from typing import Optional

from textual.app import ComposeResult
from textual.containers import VerticalScroll, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from vector_forge.ui.theme import ICONS
from vector_forge.ui.state import AgentUIState, AgentMessage, MessageRole, ToolCall, AgentStatus


class ToolCallDisplay(Widget):
    """Display for a single tool call."""

    DEFAULT_CSS = """
    ToolCallDisplay {
        height: auto;
        margin: 0 0 0 2;
        padding: 0 1;
        background: $surface;
    }

    ToolCallDisplay .tool-header {
        height: 1;
    }

    ToolCallDisplay .tool-args {
        height: auto;
        max-height: 4;
        color: $foreground-disabled;
        padding-left: 2;
    }

    ToolCallDisplay .tool-result {
        height: auto;
        max-height: 4;
        color: $foreground-muted;
        padding-left: 2;
    }
    """

    def __init__(self, tool_call: ToolCall, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tool_call = tool_call

    def compose(self) -> ComposeResult:
        yield Static(id="tool-header", classes="tool-header")
        yield Static(id="tool-args", classes="tool-args")
        yield Static(id="tool-result", classes="tool-result")

    def on_mount(self) -> None:
        self._update_display()

    def _update_display(self) -> None:
        tc = self.tool_call

        # Status icon and color
        status_map = {
            "pending": (ICONS.waiting, "$foreground-muted"),
            "running": (ICONS.thinking, "$accent"),
            "success": (ICONS.success, "$success"),
            "error": (ICONS.error, "$error"),
        }
        icon, color = status_map.get(tc.status, (ICONS.pending, "$foreground-disabled"))

        # Duration
        duration_str = ""
        if tc.duration_ms is not None:
            duration_str = f" [$foreground-disabled]{tc.duration_ms}ms[/]"

        # Header
        header = self.query_one("#tool-header", Static)
        header.update(f"[{color}]{icon}[/] [$secondary]{tc.name}[/]{duration_str}")

        # Arguments (truncated)
        args_widget = self.query_one("#tool-args", Static)
        args = tc.arguments
        if len(args) > 100:
            args = args[:97] + "..."
        args_widget.update(args)

        # Result (if available, truncated)
        result_widget = self.query_one("#tool-result", Static)
        if tc.result:
            result = tc.result
            if len(result) > 100:
                result = result[:97] + "..."
            result_widget.update(f"→ {result}")
        else:
            result_widget.update("")


class MessageDisplay(Widget):
    """Display for a single message in the conversation."""

    DEFAULT_CSS = """
    MessageDisplay {
        height: auto;
        padding: 0 1;
        margin-bottom: 1;
    }

    MessageDisplay.-user {
        background: transparent;
    }

    MessageDisplay.-assistant {
        background: $surface;
    }

    MessageDisplay.-tool {
        background: transparent;
    }

    MessageDisplay.-system {
        background: $panel;
    }

    MessageDisplay .msg-header {
        height: 1;
    }

    MessageDisplay .msg-content {
        height: auto;
        padding-left: 2;
    }

    MessageDisplay .msg-tools {
        height: auto;
        margin-top: 0;
    }
    """

    def __init__(self, message: AgentMessage, **kwargs) -> None:
        super().__init__(**kwargs)
        self.message = message

    def compose(self) -> ComposeResult:
        yield Static(id="msg-header", classes="msg-header")
        yield Static(id="msg-content", classes="msg-content")
        yield Vertical(id="msg-tools", classes="msg-tools")

    def on_mount(self) -> None:
        self._update_display()

    def _update_display(self) -> None:
        msg = self.message

        # Add role class
        self.add_class(f"-{msg.role.value}")

        # Role colors and icons
        role_styles = {
            MessageRole.SYSTEM: ("$primary", "SYS"),
            MessageRole.USER: ("$secondary", "USR"),
            MessageRole.ASSISTANT: ("$accent", "AST"),
            MessageRole.TOOL: ("$foreground-muted", "TOL"),
        }
        color, label = role_styles.get(msg.role, ("$foreground-muted", "???"))

        # Header: role label and timestamp
        header = self.query_one("#msg-header", Static)
        header.update(
            f"[{color} bold]{label}[/] [$foreground-disabled]{msg.time_str}[/]"
        )

        # Content
        content_widget = self.query_one("#msg-content", Static)
        content = msg.content
        if len(content) > 500:
            content = content[:497] + "..."
        content_widget.update(f"[$foreground]{content}[/]")

        # Tool calls
        tools_container = self.query_one("#msg-tools", Vertical)
        for child in list(tools_container.children):
            child.remove()

        for tool_call in msg.tool_calls:
            tools_container.mount(ToolCallDisplay(tool_call))


class AgentInspector(Widget):
    """Inspector panel showing messages and details for selected agent.

    Similar to AISI inspect UI - shows conversation flow with messages and tool calls.
    """

    DEFAULT_CSS = """
    AgentInspector {
        width: 2fr;
        height: 1fr;
        background: $panel;
    }

    AgentInspector #inspector-header {
        height: 3;
        padding: 0 1;
        background: $panel;
        border-bottom: solid $surface;
    }

    AgentInspector #inspector-title {
        height: 1;
        color: $foreground;
    }

    AgentInspector #inspector-subtitle {
        height: 1;
        color: $foreground-muted;
    }

    AgentInspector #inspector-stats {
        height: 1;
        color: $foreground-disabled;
    }

    AgentInspector #inspector-scroll {
        height: 1fr;
        padding: 1;
    }

    AgentInspector #inspector-empty {
        padding: 2;
        color: $foreground-muted;
        text-align: center;
    }
    """

    agent: reactive[Optional[AgentUIState]] = reactive(None, init=False)

    def compose(self) -> ComposeResult:
        with Widget(id="inspector-header"):
            yield Static("Select an agent", id="inspector-title")
            yield Static("", id="inspector-subtitle")
            yield Static("", id="inspector-stats")
        yield VerticalScroll(id="inspector-scroll")

    def on_mount(self) -> None:
        self._update_display()

    def watch_agent(self, agent: Optional[AgentUIState]) -> None:
        if self.is_mounted:
            self._update_display()

    def _update_display(self) -> None:
        """Update the inspector display."""
        title = self.query_one("#inspector-title", Static)
        subtitle = self.query_one("#inspector-subtitle", Static)
        stats = self.query_one("#inspector-stats", Static)
        scroll = self.query_one("#inspector-scroll", VerticalScroll)

        # Clear existing messages
        for child in list(scroll.children):
            child.remove()

        if self.agent is None:
            title.update("[$foreground-muted]Select an agent[/]")
            subtitle.update("[$foreground-disabled]Click an agent to view its messages[/]")
            stats.update("")
            scroll.mount(Static("", id="inspector-empty"))
            return

        agent = self.agent

        # Status styling
        status_map = {
            AgentStatus.IDLE: (ICONS.pending, "$foreground-disabled", "idle"),
            AgentStatus.RUNNING: (ICONS.running, "$accent", "running"),
            AgentStatus.WAITING: (ICONS.waiting, "$foreground-muted", "waiting"),
            AgentStatus.COMPLETE: (ICONS.complete, "$success", "complete"),
            AgentStatus.ERROR: (ICONS.failed, "$error", "error"),
        }
        icon, color, status_text = status_map.get(
            agent.status, (ICONS.pending, "$foreground-disabled", "unknown")
        )

        # Title: agent name and status
        title.update(
            f"[{color}]{icon}[/] [$foreground bold]{agent.name}[/] "
            f"[$foreground-muted]· {status_text}[/]"
        )

        # Subtitle: role and current tool
        if agent.current_tool:
            subtitle.update(
                f"[$foreground-disabled]{agent.role}[/] "
                f"[$accent]{ICONS.active} {agent.current_tool}[/]"
            )
        else:
            subtitle.update(f"[$foreground-disabled]{agent.role}[/]")

        # Stats
        stats.update(
            f"[$foreground-disabled]{len(agent.messages)} messages · "
            f"{agent.turns} turns · {agent.tool_calls_count} tools · {agent.elapsed_str}[/]"
        )

        # Messages
        if not agent.messages:
            scroll.mount(
                Static(
                    "[$foreground-muted]No messages yet[/]",
                    id="inspector-empty",
                )
            )
            return

        for msg in agent.messages:
            scroll.mount(MessageDisplay(msg))

        # Scroll to bottom
        scroll.scroll_end(animate=False)

    def set_agent(self, agent: Optional[AgentUIState]) -> None:
        """Set the agent to inspect."""
        self.agent = agent
