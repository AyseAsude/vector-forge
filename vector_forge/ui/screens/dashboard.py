"""Dashboard screen - split view with tasks and details."""

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
    LogAdded,
    ProgressUpdated,
    AgentUpdated,
    ExtractionStatusChanged,
    RefreshTime,
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

    def on_delete_button_clicked(self, event: DeleteButton.Clicked) -> None:
        """Handle delete button click."""
        event.stop()
        self.post_message(self.DeleteRequested(self.extraction.id))

    def on_click(self) -> None:
        self.post_message(self.Selected(self.extraction.id))

    def update(self, extraction: ExtractionUIState) -> None:
        self.extraction = extraction
        if self.is_mounted:
            self._update_display()

    def _update_display(self) -> None:
        ext = self.extraction
        icon, color, runs, layer, score = self._get_display_values()

        self.query_one(".name", Static).update(f"[{color}]{icon}[/] [bold]{ext.behavior_name}[/]")
        self.query_one(".time", Static).update(ext.elapsed_str)
        self.query_one(ProgressBar).set_value(ext.progress * 100)
        self.query_one(".meta", Static).update(
            f"[$accent]{ext.phase.value.upper()}[/] · {runs} runs · {layer} · {score}"
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

    DetailsPanel #empty-state {
        height: 1fr;
        content-align: center middle;
        color: $foreground-muted;
    }

    DetailsPanel #details-content {
        height: 1fr;
    }

    DetailsPanel #detail-title {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
    }

    DetailsPanel #detail-description {
        height: auto;
        color: $foreground-muted;
        margin-bottom: 1;
    }

    DetailsPanel #detail-stats {
        height: 1;
        margin-bottom: 1;
    }

    DetailsPanel .section {
        height: 1;
        text-style: bold;
        margin-top: 1;
    }

    DetailsPanel #runs-list {
        height: auto;
        max-height: 10;
    }

    DetailsPanel #activity-list {
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
        self._current_extraction_id: str | None = None
        self._agent_rows: dict[str, AgentRow] = {}
        self._log_entries: list[str] = []

    def compose(self) -> ComposeResult:
        yield Static("Select a task to view details", id="empty-state")
        with Vertical(id="details-content"):
            yield Static(id="detail-title")
            yield Static(id="detail-description")
            yield Static(id="detail-stats")
            yield Static("PARALLEL RUNS", classes="section")
            yield VerticalScroll(id="runs-list")
            yield Static("RECENT", classes="section")
            yield VerticalScroll(id="activity-list")

    def on_mount(self) -> None:
        self.query_one("#details-content").display = False

    def show(self, extraction: ExtractionUIState | None) -> None:
        empty_state = self.query_one("#empty-state")
        details_content = self.query_one("#details-content")

        if extraction is None:
            empty_state.display = True
            details_content.display = False
            self._current_extraction_id = None
            return

        empty_state.display = False
        details_content.display = True

        ext = extraction

        # Update title
        status_map = {
            ExtractionStatus.PENDING: (ICONS.pending, "$foreground-muted"),
            ExtractionStatus.RUNNING: (ICONS.running, "$accent"),
            ExtractionStatus.PAUSED: (ICONS.paused, "$warning"),
            ExtractionStatus.COMPLETE: (ICONS.complete, "$success"),
            ExtractionStatus.FAILED: (ICONS.failed, "$error"),
        }
        icon, color = status_map.get(ext.status, (ICONS.pending, "$foreground-muted"))
        self.query_one("#detail-title", Static).update(f"[{color}]{icon}[/] {ext.behavior_name}")

        # Update description
        desc = ext.behavior_description or "No description"
        if len(desc) > 100:
            desc = desc[:97] + "..."
        self.query_one("#detail-description", Static).update(desc)

        # Update stats
        runs = f"{ext.running_agents_count}/{ext.total_agents_count}" if ext.total_agents_count else "—"
        layer = f"L{ext.current_layer}" if ext.current_layer else "—"
        score = f"{ext.evaluation.overall:.2f}" if ext.evaluation.overall > 0 else "—"
        self.query_one("#detail-stats", Static).update(
            f"[$accent]{ext.phase.value.upper()}[/]  │  "
            f"Runs: {runs}  │  Layer: {layer}  │  Score: {score}"
        )

        # Update runs list - only if extraction changed or agents changed
        self._update_runs_list(ext)

        # Update activity list - only if logs changed
        self._update_activity_list(ext.id)

        self._current_extraction_id = ext.id

    def _update_runs_list(self, ext: ExtractionUIState) -> None:
        """Update the parallel runs list incrementally."""
        runs_list = self.query_one("#runs-list", VerticalScroll)
        current_agent_ids = set(list(ext.agents.keys())[:8])
        existing_ids = set(self._agent_rows.keys())

        # Remove agents that no longer exist
        for agent_id in existing_ids - current_agent_ids:
            if agent_id in self._agent_rows:
                self._agent_rows[agent_id].remove()
                del self._agent_rows[agent_id]

        # Update or add agents
        for agent in list(ext.agents.values())[:8]:
            icon_map = {
                AgentStatus.IDLE: ("○", "$foreground-muted"),
                AgentStatus.RUNNING: ("●", "$accent"),
                AgentStatus.WAITING: ("◐", "$foreground-muted"),
                AgentStatus.COMPLETE: ("●", "$success"),
                AgentStatus.ERROR: ("●", "$error"),
            }
            a_icon, a_color = icon_map.get(agent.status, ("○", "$foreground-muted"))
            content = (
                f"[{a_color}]{a_icon}[/] {agent.name}  "
                f"[$foreground-muted]{agent.status.value}  {agent.turns}t  {agent.elapsed_str}[/]"
            )

            if agent.id in self._agent_rows:
                # Update existing row
                self._agent_rows[agent.id].update(content)
            else:
                # Add new row
                row = AgentRow(agent.id, content)
                runs_list.mount(row)
                self._agent_rows[agent.id] = row

        # Handle empty state
        empty_widgets = list(runs_list.query(".empty-runs"))
        if not ext.agents:
            if not empty_widgets:
                runs_list.mount(Static("[$foreground-muted]No runs yet[/]", classes="empty-runs"))
        else:
            for w in empty_widgets:
                w.remove()

    def _update_activity_list(self, extraction_id: str) -> None:
        """Update the activity list incrementally."""
        activity_list = self.query_one("#activity-list", VerticalScroll)
        logs = get_state().get_filtered_logs(extraction_id=extraction_id)[-5:]

        # Create list of log identifiers
        new_log_keys = [f"{log.timestamp}:{log.message[:50]}" for log in logs]

        # Only update if logs changed
        if new_log_keys == self._log_entries:
            return

        # Clear and rebuild (logs are small - 5 items max)
        activity_list.remove_children()

        if logs:
            for log in reversed(logs):
                safe_message = escape_markup(log.message)
                activity_list.mount(Static(
                    f"[$foreground-muted]{log.time_str}[/] {safe_message}",
                    classes="log-entry"
                ))
        else:
            activity_list.mount(Static("[$foreground-muted]No activity yet[/]", classes="log-entry"))

        self._log_entries = new_log_keys


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
        self._sync()

    # ─────────────────────────────────────────────────────────────────
    # Message handlers - pure event-driven updates, no polling
    # ─────────────────────────────────────────────────────────────────

    def on_refresh_time(self, message: RefreshTime) -> None:
        """Handle time refresh - update elapsed time displays only."""
        state = get_state()
        for card in self.query(TaskCard):
            ext = state.extractions.get(card.extraction.id)
            if ext and ext.status == ExtractionStatus.RUNNING:
                card.query_one(".time", Static).update(ext.elapsed_str)
        self.query_one(TmuxBar).refresh_info()

    def on_progress_updated(self, message: ProgressUpdated) -> None:
        """Handle progress update - update specific task card."""
        state = get_state()
        for card in self.query(TaskCard):
            if card.extraction.id == message.extraction_id:
                ext = state.extractions.get(message.extraction_id)
                if ext:
                    card.update(ext)
                break
        # Update details if showing this extraction
        if state.selected_id == message.extraction_id:
            self.query_one("#right", DetailsPanel).show(state.selected_extraction)

    def on_extraction_status_changed(self, message: ExtractionStatusChanged) -> None:
        """Handle extraction status change."""
        self._sync()

    def on_agent_updated(self, message: AgentUpdated) -> None:
        """Handle agent update - refresh details panel if showing this extraction."""
        state = get_state()
        if state.selected_id == message.extraction_id:
            self.query_one("#right", DetailsPanel).show(state.selected_extraction)

    def on_log_added(self, message: LogAdded) -> None:
        """Handle new log - refresh activity list if showing this extraction."""
        state = get_state()
        if message.extraction_id and state.selected_id == message.extraction_id:
            details = self.query_one("#right", DetailsPanel)
            details._update_activity_list(message.extraction_id)

    def _sync(self) -> None:
        state = get_state()
        tasks_container = self.query_one("#tasks", VerticalScroll)

        # Update or create cards
        existing = {c.extraction.id: c for c in self.query(TaskCard)}

        # Reverse order so newest extractions appear at top
        for ext_id, ext in reversed(list(state.extractions.items())):
            if ext_id in existing:
                existing[ext_id].update(ext)
                existing[ext_id].set_class(ext_id == state.selected_id, "-selected")
            else:
                card = TaskCard(ext)
                card.set_class(ext_id == state.selected_id, "-selected")
                # Insert at top (index 0) to maintain newest-first order
                tasks_container.mount(card, before=0)

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

    def on_task_card_delete_requested(self, event: TaskCard.DeleteRequested) -> None:
        """Handle delete button click on a task card."""
        state = get_state()
        extraction = state.extractions.get(event.extraction_id)
        if extraction:
            task_name = extraction.behavior_name

            def on_confirm(confirmed: bool) -> None:
                if confirmed:
                    self._hide_task(event.extraction_id)

            self.app.push_screen(
                ConfirmHideTaskScreen(task_name, event.extraction_id),
                on_confirm
            )

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

        # Sync UI
        self._sync()

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
