"""Dashboard screen - split view with tasks and details.

Event-Sourcing Pattern:
- on_mount: Initial projection from state
- Event handlers: Targeted updates to specific widgets
- No polling timers or state listeners
"""

from rich.markup import escape as escape_markup
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen, ModalScreen
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
from vector_forge.ui.widgets.model_card import DeleteButton
from vector_forge.ui.messages import (
    TaskCreated,
    TaskProgressChanged,
    TaskStatusChanged,
    TaskRemoved,
    TaskSelected,
    AgentSpawned,
    AgentStatusChanged,
    LogEmitted,
    TimeTick,
)


class ProgressBar(Static):
    """Terminal-style progress bar using block characters."""

    DEFAULT_CSS = """
    ProgressBar {
        height: 1;
        width: 1fr;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
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

    TaskCard .meta-row {
        height: 1;
    }

    TaskCard .meta {
        width: 1fr;
        color: $foreground-muted;
    }

    TaskCard DeleteButton {
        dock: right;
    }
    """

    class Selected(Message):
        def __init__(self, extraction_id: str) -> None:
            super().__init__()
            self.extraction_id = extraction_id

    class DeleteRequested(Message):
        """Emitted when delete is requested for this task."""
        def __init__(self, extraction_id: str) -> None:
            super().__init__()
            self.extraction_id = extraction_id

    def __init__(self, extraction: ExtractionUIState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.extraction = extraction

    def _get_display_values(self) -> tuple:
        """Get display values for the extraction."""
        ext = self.extraction
        status_map = {
            ExtractionStatus.PENDING: (ICONS.pending, "$foreground-muted"),
            ExtractionStatus.RUNNING: (ICONS.running, "$accent"),
            ExtractionStatus.PAUSED: (ICONS.paused, "$warning"),
            ExtractionStatus.COMPLETE: (ICONS.complete, "$success"),
            ExtractionStatus.FAILED: (ICONS.failed, "$error"),
        }
        icon, color = status_map.get(ext.status, (ICONS.pending, "$foreground-muted"))
        runs = f"{ext.running_agents_count}/{ext.total_agents_count}" if ext.total_agents_count else "—"
        layer = f"L{ext.current_layer}" if ext.current_layer else "—"
        score = f"{ext.evaluation.overall:.2f}" if ext.evaluation.overall > 0 else "—"
        return icon, color, runs, layer, score

    def compose(self) -> ComposeResult:
        ext = self.extraction
        icon, color, runs, layer, score = self._get_display_values()

        with Horizontal(classes="header-row"):
            yield Static(f"[{color}]{icon}[/] [bold]{ext.behavior_name}[/]", classes="name")
            yield Static(ext.elapsed_str, classes="time")
        yield ProgressBar(classes="progress")
        with Horizontal(classes="meta-row"):
            yield Static(
                f"[$accent]{ext.phase.value.upper()}[/] · {runs} runs · {layer} · {score}",
                classes="meta"
            )
            yield DeleteButton()

    def on_mount(self) -> None:
        """Set initial progress value after mount."""
        self.query_one(ProgressBar).set_value(self.extraction.progress)

    def on_delete_button_clicked(self, event: DeleteButton.Clicked) -> None:
        """Handle delete button click."""
        event.stop()
        self.post_message(self.DeleteRequested(self.extraction.id))

    def on_click(self) -> None:
        self.post_message(self.Selected(self.extraction.id))

    def set_selected(self, selected: bool) -> None:
        self.set_class(selected, "-selected")

    def update_from_state(self, extraction: ExtractionUIState) -> None:
        """Update card display from extraction state."""
        self.extraction = extraction
        if not self.is_mounted:
            return

        icon, color, runs, layer, score = self._get_display_values()
        try:
            self.query_one(".name", Static).update(f"[{color}]{icon}[/] [bold]{extraction.behavior_name}[/]")
            self.query_one(".time", Static).update(extraction.elapsed_str)
            self.query_one(ProgressBar).set_value(extraction.progress)
            self.query_one(".meta", Static).update(
                f"[$accent]{extraction.phase.value.upper()}[/] · {runs} runs · {layer} · {score}"
            )
        except Exception:
            pass

    def set_elapsed(self, elapsed: str) -> None:
        """Update just the elapsed time."""
        if self.is_mounted:
            try:
                self.query_one(".time", Static).update(elapsed)
            except Exception:
                pass


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

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._current_task_id: str | None = None

    def compose(self) -> ComposeResult:
        yield Static("Select a task", classes="empty")

    def show(self, extraction: ExtractionUIState | None) -> None:
        """Show details for a task (full rebuild)."""
        self.remove_children()
        self._current_task_id = extraction.id if extraction else None

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
                safe_message = escape_markup(log.message)
                activity_list.mount(Static(
                    f"[$foreground-muted]{log.time_str}[/] {safe_message}",
                    classes="log-entry"
                ))
        else:
            activity_list.mount(Static("[$foreground-muted]No activity yet[/]", classes="log-entry"))


class ConfirmHideTaskScreen(ModalScreen[bool]):
    """Modal confirmation dialog for hiding a task."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "confirm", "Confirm"),
    ]

    DEFAULT_CSS = """
    ConfirmHideTaskScreen {
        align: center middle;
    }

    ConfirmHideTaskScreen #dialog {
        width: 50;
        height: auto;
        background: $surface;
        padding: 1 2;
    }

    ConfirmHideTaskScreen #title {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
    }

    ConfirmHideTaskScreen #message {
        height: auto;
        color: $foreground-muted;
        margin-bottom: 1;
    }

    ConfirmHideTaskScreen #buttons {
        height: 3;
    }

    ConfirmHideTaskScreen #btn-cancel {
        width: 1fr;
        height: 3;
        margin-right: 1;
        background: $boost;
        color: $foreground;
        border: none;
    }

    ConfirmHideTaskScreen #btn-cancel:hover {
        background: $boost 80%;
    }

    ConfirmHideTaskScreen #btn-confirm {
        width: 1fr;
        height: 3;
        background: $error;
        color: $background;
        border: none;
        text-style: bold;
    }

    ConfirmHideTaskScreen #btn-confirm:hover {
        background: $error 80%;
    }
    """

    def __init__(self, task_name: str, extraction_id: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._task_name = task_name
        self._extraction_id = extraction_id

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static("Hide Task?", id="title")
            yield Static(
                f"Hide [bold]{self._task_name}[/]?\n"
                "The data will be preserved. Delete the .hidden file\n"
                "in the session folder to restore it.",
                id="message"
            )
            with Horizontal(id="buttons"):
                yield Button("Cancel", id="btn-cancel")
                yield Button("Hide", id="btn-confirm")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.dismiss(False)
        elif event.button.id == "btn-confirm":
            self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)

    def action_confirm(self) -> None:
        self.dismiss(True)


class DashboardScreen(Screen):
    """Main dashboard with task list and details panel.

    Uses event-driven architecture: event handlers update specific widgets.
    """

    BINDINGS = [
        Binding("1", "noop", "Dashboard", show=False),
        Binding("2", "go_samples", "Samples", key_display="2"),
        Binding("3", "go_logs", "Logs", key_display="3"),
        Binding("tab", "cycle", "Next Screen"),
        Binding("j", "next", "Next", show=False),
        Binding("k", "prev", "Previous", show=False),
        Binding("down", "next", "Next Task", key_display="↓"),
        Binding("up", "prev", "Prev Task", key_display="↑"),
        Binding("enter", "open", "Open"),
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

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._task_cards: dict[str, TaskCard] = {}
        self._selected_id: str | None = None

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
        """Initial projection from current state."""
        state = get_state()
        tasks_container = self.query_one("#tasks", VerticalScroll)

        # Mount existing tasks (newest first)
        for ext_id, ext in reversed(list(state.extractions.items())):
            card = TaskCard(ext)
            self._task_cards[ext_id] = card
            tasks_container.mount(card)

        # Select task
        if state.selected_id and state.selected_id in state.extractions:
            self._select_task(state.selected_id)
        elif state.extractions:
            first_id = next(iter(reversed(list(state.extractions.keys()))))
            self._select_task(first_id)
        else:
            tasks_container.mount(Static("No tasks. Press [bold]n[/] to create.", classes="empty"))

    # ─────────────────────────────────────────────────────────────────────────
    # Event Handlers
    # ─────────────────────────────────────────────────────────────────────────

    def on_task_created(self, event: TaskCreated) -> None:
        """Handle new task - add card at top."""
        state = get_state()
        extraction = state.extractions.get(event.task_id)
        if not extraction:
            return

        tasks_container = self.query_one("#tasks", VerticalScroll)

        # Remove empty message
        for empty in tasks_container.query(".empty"):
            empty.remove()

        # Add card at top
        card = TaskCard(extraction)
        self._task_cards[event.task_id] = card
        tasks_container.mount(card, before=0)

        # Select it
        self._select_task(event.task_id)

    def on_task_progress_changed(self, event: TaskProgressChanged) -> None:
        """Handle progress update."""
        state = get_state()
        extraction = state.extractions.get(event.task_id)
        if extraction:
            card = self._task_cards.get(event.task_id)
            if card:
                card.update_from_state(extraction)

            if self._selected_id == event.task_id:
                self.query_one("#right", DetailsPanel).show(extraction)

    def on_task_status_changed(self, event: TaskStatusChanged) -> None:
        """Handle status change."""
        state = get_state()
        extraction = state.extractions.get(event.task_id)
        if extraction:
            card = self._task_cards.get(event.task_id)
            if card:
                card.update_from_state(extraction)

            if self._selected_id == event.task_id:
                self.query_one("#right", DetailsPanel).show(extraction)

    def on_task_removed(self, event: TaskRemoved) -> None:
        """Handle task removal."""
        card = self._task_cards.pop(event.task_id, None)
        if card:
            card.remove()

        if self._selected_id == event.task_id:
            state = get_state()
            if state.extractions:
                self._select_task(next(iter(state.extractions)))
            else:
                self._selected_id = None
                self.query_one("#right", DetailsPanel).show(None)
                tasks_container = self.query_one("#tasks", VerticalScroll)
                tasks_container.mount(Static("No tasks. Press [bold]n[/] to create.", classes="empty"))

    def on_agent_spawned(self, event: AgentSpawned) -> None:
        """Handle agent spawn - update details panel."""
        if self._selected_id == event.task_id:
            state = get_state()
            extraction = state.extractions.get(event.task_id)
            if extraction:
                self.query_one("#right", DetailsPanel).show(extraction)

    def on_agent_status_changed(self, event: AgentStatusChanged) -> None:
        """Handle agent status change."""
        state = get_state()
        extraction = state.extractions.get(event.task_id)
        if extraction:
            card = self._task_cards.get(event.task_id)
            if card:
                card.update_from_state(extraction)

            if self._selected_id == event.task_id:
                self.query_one("#right", DetailsPanel).show(extraction)

    def on_log_emitted(self, event: LogEmitted) -> None:
        """Handle new log - update activity if showing this task."""
        if event.task_id and event.task_id == self._selected_id:
            state = get_state()
            extraction = state.extractions.get(event.task_id)
            if extraction:
                self.query_one("#right", DetailsPanel).show(extraction)

    def on_time_tick(self, event: TimeTick) -> None:
        """Handle time tick - update elapsed times for running tasks."""
        state = get_state()
        for task_id, card in self._task_cards.items():
            extraction = state.extractions.get(task_id)
            if extraction and extraction.status == ExtractionStatus.RUNNING:
                card.set_elapsed(extraction.elapsed_str)

        self.query_one(TmuxBar).refresh_info()

    # ─────────────────────────────────────────────────────────────────────────
    # UI Event Handlers
    # ─────────────────────────────────────────────────────────────────────────

    def on_task_card_selected(self, event: TaskCard.Selected) -> None:
        """Handle card click."""
        self._select_task(event.extraction_id)

    def on_agent_row_clicked(self, event: AgentRow.Clicked) -> None:
        """Handle agent row click - open samples."""
        state = get_state()
        if self._selected_id:
            extraction = state.extractions.get(self._selected_id)
            if extraction:
                extraction.select_agent(event.agent_id)
        self.app.switch_screen("samples")

    def on_task_card_delete_requested(self, event: TaskCard.DeleteRequested) -> None:
        """Handle delete button click on a task card."""
        state = get_state()
        extraction = state.extractions.get(event.extraction_id)
        if extraction:
            def on_confirm(confirmed: bool) -> None:
                if confirmed:
                    self._hide_task(event.extraction_id)

            self.app.push_screen(
                ConfirmHideTaskScreen(extraction.behavior_name, event.extraction_id),
                on_confirm
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "new-btn":
            self.app.push_screen("create_task")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _select_task(self, task_id: str) -> None:
        """Select a task and update UI."""
        # Deselect previous
        if self._selected_id and self._selected_id in self._task_cards:
            self._task_cards[self._selected_id].set_selected(False)

        # Select new
        self._selected_id = task_id
        if task_id in self._task_cards:
            self._task_cards[task_id].set_selected(True)

        # Update state
        state = get_state()
        state.selected_id = task_id

        # Update details
        extraction = state.extractions.get(task_id)
        self.query_one("#right", DetailsPanel).show(extraction)

    def _hide_task(self, extraction_id: str) -> None:
        """Hide a task from the list."""
        from vector_forge.services.session import SessionService

        state = get_state()

        # Get session service from app
        session_service = getattr(self.app, "_session_service", None)
        if session_service is None:
            session_service = SessionService()

        # Hide the session
        session_service.hide_session(extraction_id)

        # Remove from UI state
        state.remove_extraction(extraction_id)

        # Post event for other screens
        self.app.post_message(TaskRemoved(task_id=extraction_id))

    # ─────────────────────────────────────────────────────────────────────────
    # Actions
    # ─────────────────────────────────────────────────────────────────────────

    def action_noop(self) -> None:
        pass

    def action_go_samples(self) -> None:
        self.app.switch_screen("samples")

    def action_go_logs(self) -> None:
        self.app.switch_screen("logs")

    def action_cycle(self) -> None:
        self.app.switch_screen("samples")

    def action_next(self) -> None:
        task_ids = list(self._task_cards.keys())
        if not task_ids:
            return
        if self._selected_id is None:
            self._select_task(task_ids[0])
        else:
            try:
                idx = task_ids.index(self._selected_id)
                self._select_task(task_ids[(idx + 1) % len(task_ids)])
            except ValueError:
                self._select_task(task_ids[0])

    def action_prev(self) -> None:
        task_ids = list(self._task_cards.keys())
        if not task_ids:
            return
        if self._selected_id is None:
            self._select_task(task_ids[-1])
        else:
            try:
                idx = task_ids.index(self._selected_id)
                self._select_task(task_ids[(idx - 1) % len(task_ids)])
            except ValueError:
                self._select_task(task_ids[-1])

    def action_open(self) -> None:
        self.app.switch_screen("samples")

    def action_new_task(self) -> None:
        self.app.push_screen("create_task")

    def action_quit(self) -> None:
        self.app.exit()
