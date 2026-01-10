"""Dashboard screen - split view with tasks and details."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Static, Button
from textual.message import Message

from vector_forge.ui.state import (
    ExtractionUIState,
    ExtractionStatus,
    AgentStatus,
    get_state,
)
from vector_forge.ui.theme import ICONS
from vector_forge.ui.widgets.tmux_bar import TmuxBar


class ProgressBar(Static):
    """Terminal-style progress bar using block characters."""

    DEFAULT_CSS = """
    ProgressBar {
        height: 1;
        width: 1fr;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._value = 0.0

    def set_value(self, percent: float) -> None:
        self._value = max(0.0, min(100.0, percent))
        self.refresh()

    def render(self) -> str:
        width = max(10, self.size.width - 6)
        filled = int((self._value / 100.0) * width)
        empty = width - filled
        bar = f"[$accent]{'█' * filled}[/][$boost]{'░' * empty}[/]"
        return f"{bar} [$foreground-muted]{self._value:3.0f}%[/]"


class TaskCard(Static):
    """Compact task card with status, progress, and metadata."""

    DEFAULT_CSS = """
    TaskCard {
        height: auto;
        padding: 1 2;
        margin-bottom: 1;
        margin-right: 2;
        background: $surface;
    }

    TaskCard:hover {
        background: $boost;
    }

    TaskCard.-selected {
        background: $primary 20%;
    }

    TaskCard.-selected:hover {
        background: $primary 30%;
    }

    TaskCard .header-row {
        height: 1;
        margin-bottom: 1;
    }

    TaskCard .name {
        width: 1fr;
    }

    TaskCard .time {
        width: auto;
        color: $foreground-muted;
    }

    TaskCard .progress {
        margin-bottom: 1;
    }

    TaskCard .meta {
        height: 1;
        color: $foreground-muted;
    }
    """

    class Selected(Message):
        def __init__(self, extraction_id: str) -> None:
            super().__init__()
            self.extraction_id = extraction_id

    def __init__(self, extraction: ExtractionUIState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.extraction = extraction

    def compose(self) -> ComposeResult:
        with Horizontal(classes="header-row"):
            yield Static(classes="name")
            yield Static(classes="time")
        yield ProgressBar(classes="progress")
        yield Static(classes="meta")

    def on_mount(self) -> None:
        self._update_display()

    def on_click(self) -> None:
        self.post_message(self.Selected(self.extraction.id))

    def update(self, extraction: ExtractionUIState) -> None:
        self.extraction = extraction
        if self.is_mounted:
            self._update_display()

    def _update_display(self) -> None:
        ext = self.extraction

        # Status icon and color
        status_map = {
            ExtractionStatus.PENDING: (ICONS.pending, "$foreground-muted"),
            ExtractionStatus.RUNNING: (ICONS.running, "$accent"),
            ExtractionStatus.PAUSED: (ICONS.paused, "$warning"),
            ExtractionStatus.COMPLETE: (ICONS.complete, "$success"),
            ExtractionStatus.FAILED: (ICONS.failed, "$error"),
        }
        icon, color = status_map.get(ext.status, (ICONS.pending, "$foreground-muted"))

        # Name with status icon
        self.query_one(".name", Static).update(f"[{color}]{icon}[/] [bold]{ext.behavior_name}[/]")

        # Time on the right
        self.query_one(".time", Static).update(ext.elapsed_str)

        # Progress bar
        self.query_one(ProgressBar).set_value(ext.progress * 100)

        # Meta: phase, runs, layer, score
        runs = f"{ext.running_agents_count}/{ext.total_agents_count}" if ext.total_agents_count else "—"
        layer = f"L{ext.current_layer}" if ext.current_layer else "—"
        score = f"{ext.evaluation.overall:.2f}" if ext.evaluation.overall > 0 else "—"

        self.query_one(".meta", Static).update(
            f"[$accent]{ext.phase.value.upper()}[/] · "
            f"{runs} runs · {layer} · {score}"
        )


class AgentRow(Static):
    """Clickable row for an agent in the details panel."""

    DEFAULT_CSS = """
    AgentRow {
        height: 1;
        padding: 0 1;
    }

    AgentRow:hover {
        background: $boost;
    }
    """

    class Clicked(Message):
        def __init__(self, agent_id: str) -> None:
            super().__init__()
            self.agent_id = agent_id

    def __init__(self, agent_id: str, content: str, **kwargs) -> None:
        super().__init__(content, **kwargs)
        self._agent_id = agent_id

    def on_click(self) -> None:
        self.post_message(self.Clicked(self._agent_id))


class DetailsPanel(Vertical):
    """Right panel showing details of selected task."""

    DEFAULT_CSS = """
    DetailsPanel {
        width: 1fr;
        height: 1fr;
        padding: 1 2;
        background: $surface;
    }

    DetailsPanel .empty {
        height: 1fr;
        content-align: center middle;
        color: $foreground-muted;
    }

    DetailsPanel .title {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
    }

    DetailsPanel .description {
        height: auto;
        color: $foreground-muted;
        margin-bottom: 1;
    }

    DetailsPanel .stats {
        height: 1;
        margin-bottom: 1;
    }

    DetailsPanel .section {
        height: 1;
        text-style: bold;
        margin-top: 1;
    }

    DetailsPanel .list {
        height: auto;
        max-height: 10;
    }

    DetailsPanel .activity {
        height: 1fr;
        min-height: 5;
    }

    DetailsPanel .log-entry {
        height: 1;
        color: $foreground-muted;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("Select a task", classes="empty")

    def show(self, extraction: ExtractionUIState | None) -> None:
        self.remove_children()

        if extraction is None:
            self.mount(Static("Select a task to view details", classes="empty"))
            return

        ext = extraction

        # Status icon and color
        status_map = {
            ExtractionStatus.PENDING: (ICONS.pending, "$foreground-muted"),
            ExtractionStatus.RUNNING: (ICONS.running, "$accent"),
            ExtractionStatus.PAUSED: (ICONS.paused, "$warning"),
            ExtractionStatus.COMPLETE: (ICONS.complete, "$success"),
            ExtractionStatus.FAILED: (ICONS.failed, "$error"),
        }
        icon, color = status_map.get(ext.status, (ICONS.pending, "$foreground-muted"))
        self.mount(Static(f"[{color}]{icon}[/] {ext.behavior_name}", classes="title"))

        # Description
        desc = ext.behavior_description or "No description"
        if len(desc) > 100:
            desc = desc[:97] + "..."
        self.mount(Static(desc, classes="description"))

        # Stats
        runs = f"{ext.running_agents_count}/{ext.total_agents_count}" if ext.total_agents_count else "—"
        layer = f"L{ext.current_layer}" if ext.current_layer else "—"
        score = f"{ext.evaluation.overall:.2f}" if ext.evaluation.overall > 0 else "—"
        self.mount(Static(
            f"[$accent]{ext.phase.value.upper()}[/]  │  "
            f"Runs: {runs}  │  Layer: {layer}  │  Score: {score}",
            classes="stats"
        ))

        # Parallel runs section
        self.mount(Static("PARALLEL RUNS", classes="section"))
        runs_list = VerticalScroll(classes="list")
        self.mount(runs_list)

        if ext.agents:
            for agent in list(ext.agents.values())[:8]:
                icon_map = {
                    AgentStatus.IDLE: ("○", "$foreground-muted"),
                    AgentStatus.RUNNING: ("●", "$accent"),
                    AgentStatus.WAITING: ("◐", "$foreground-muted"),
                    AgentStatus.COMPLETE: ("●", "$success"),
                    AgentStatus.ERROR: ("●", "$error"),
                }
                a_icon, a_color = icon_map.get(agent.status, ("○", "$foreground-muted"))
                runs_list.mount(AgentRow(
                    agent.id,
                    f"[{a_color}]{a_icon}[/] {agent.name}  "
                    f"[$foreground-muted]{agent.status.value}  {agent.turns}t  {agent.elapsed_str}[/]"
                ))
        else:
            runs_list.mount(Static("[$foreground-muted]No runs yet[/]"))

        # Recent activity section
        self.mount(Static("RECENT", classes="section"))
        activity_list = VerticalScroll(classes="activity")
        self.mount(activity_list)

        logs = get_state().get_filtered_logs(extraction_id=ext.id)[-5:]
        if logs:
            for log in reversed(logs):
                activity_list.mount(Static(
                    f"[$foreground-muted]{log.time_str}[/] {log.message}",
                    classes="log-entry"
                ))
        else:
            activity_list.mount(Static("[$foreground-muted]No activity yet[/]", classes="log-entry"))


class DashboardScreen(Screen):
    """Main dashboard with task list and details panel."""

    BINDINGS = [
        # Navigation between screens
        Binding("1", "noop", "Dashboard", show=False),  # Current screen
        Binding("2", "go_samples", "Samples", key_display="2"),
        Binding("3", "go_logs", "Logs", key_display="3"),
        Binding("tab", "cycle", "Next Screen"),
        # List navigation
        Binding("j", "next", "Next", show=False),
        Binding("k", "prev", "Previous", show=False),
        Binding("down", "next", "Next Task", key_display="↓"),
        Binding("up", "prev", "Prev Task", key_display="↑"),
        Binding("enter", "open", "Open"),
        # Actions
        Binding("n", "new_task", "New Task"),
        Binding("q", "quit", "Quit"),
    ]

    DEFAULT_CSS = """
    DashboardScreen {
        background: $background;
    }

    DashboardScreen #main {
        height: 1fr;
    }

    DashboardScreen #left {
        width: 1fr;
        padding: 1 0 1 2;
    }

    DashboardScreen #header {
        height: 1;
        margin-bottom: 1;
        padding-right: 2;
    }

    DashboardScreen #title {
        width: 1fr;
        text-style: bold;
    }

    DashboardScreen #new-btn {
        width: auto;
        min-width: 10;
        height: 1;
        background: $accent;
        color: $background;
        border: none;
        padding: 0 1;
        text-style: bold;
    }

    DashboardScreen #new-btn:hover {
        background: $accent 80%;
    }

    DashboardScreen #new-btn:focus {
        background: $accent;
        text-style: bold;
    }

    DashboardScreen #new-btn.-active {
        background: $accent 90%;
    }

    DashboardScreen #tasks {
        height: 1fr;
    }

    DashboardScreen .empty {
        height: 1fr;
        content-align: center middle;
        color: $foreground-muted;
        margin-right: 2;
    }

    DashboardScreen #right {
        width: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(id="main"):
            with Vertical(id="left"):
                with Horizontal(id="header"):
                    yield Static("TASKS", id="title")
                    yield Button("+ New", id="new-btn")
                yield VerticalScroll(id="tasks")
            yield DetailsPanel(id="right")
        yield TmuxBar(active_screen="dashboard")

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

        # Update running task cards
        for card in self.query(TaskCard):
            ext = state.extractions.get(card.extraction.id)
            if ext and ext.status == ExtractionStatus.RUNNING:
                card.update(ext)

        # Update details if showing running task
        if state.selected_extraction and state.selected_extraction.status == ExtractionStatus.RUNNING:
            self.query_one("#right", DetailsPanel).show(state.selected_extraction)

        self.query_one(TmuxBar).refresh_info()

    def _sync(self) -> None:
        state = get_state()
        tasks_container = self.query_one("#tasks", VerticalScroll)

        # Update or create cards
        existing = {c.extraction.id: c for c in self.query(TaskCard)}

        for ext_id, ext in state.extractions.items():
            if ext_id in existing:
                existing[ext_id].update(ext)
                existing[ext_id].set_class(ext_id == state.selected_id, "-selected")
            else:
                card = TaskCard(ext)
                card.set_class(ext_id == state.selected_id, "-selected")
                tasks_container.mount(card)

        # Remove stale cards
        for ext_id, card in existing.items():
            if ext_id not in state.extractions:
                card.remove()

        # Empty state
        empties = list(tasks_container.query(".empty"))
        if not state.extractions:
            if not empties:
                tasks_container.mount(Static("No tasks. Press [bold]n[/] to create.", classes="empty"))
        else:
            for e in empties:
                e.remove()

        # Update details panel
        self.query_one("#right", DetailsPanel).show(state.selected_extraction)
        self.query_one(TmuxBar).refresh_info()

    def on_task_card_selected(self, event: TaskCard.Selected) -> None:
        state = get_state()
        state.select_extraction(event.extraction_id)

        for card in self.query(TaskCard):
            card.set_class(card.extraction.id == event.extraction_id, "-selected")

        self.query_one("#right", DetailsPanel).show(state.selected_extraction)

    def on_agent_row_clicked(self, event: AgentRow.Clicked) -> None:
        state = get_state()
        if state.selected_extraction:
            state.selected_extraction.select_agent(event.agent_id)
        self.app.switch_screen("samples")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new-btn":
            self.app.push_screen("create_task")

    # Actions
    def action_noop(self) -> None:
        pass

    def action_go_samples(self) -> None:
        self.app.switch_screen("samples")

    def action_go_logs(self) -> None:
        self.app.switch_screen("logs")

    def action_cycle(self) -> None:
        self.app.switch_screen("samples")

    def action_next(self) -> None:
        state = get_state()
        ids = list(state.extractions.keys())
        if not ids:
            return

        if state.selected_id is None:
            state.select_extraction(ids[0])
        else:
            try:
                idx = ids.index(state.selected_id)
                state.select_extraction(ids[(idx + 1) % len(ids)])
            except ValueError:
                state.select_extraction(ids[0])

        self._sync()

    def action_prev(self) -> None:
        state = get_state()
        ids = list(state.extractions.keys())
        if not ids:
            return

        if state.selected_id is None:
            state.select_extraction(ids[-1])
        else:
            try:
                idx = ids.index(state.selected_id)
                state.select_extraction(ids[(idx - 1) % len(ids)])
            except ValueError:
                state.select_extraction(ids[-1])

        self._sync()

    def action_open(self) -> None:
        self.app.switch_screen("samples")

    def action_new_task(self) -> None:
        self.app.push_screen("create_task")

    def action_quit(self) -> None:
        self.app.exit()
