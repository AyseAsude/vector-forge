"""Dashboard screen - split view with tasks and details.

Event-Sourcing Pattern:
- on_mount: Initial projection from state
- Event handlers: Targeted updates to specific widgets
- No generic _sync() - each event updates only what it affects
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


# Block characters for progress bars
BLOCK_FULL = "█"
BLOCK_EMPTY = "░"


class ProgressBar(Static):
    """Terminal-style progress bar."""

    DEFAULT_CSS = """
    ProgressBar {
        height: 1;
    }
    """

    def __init__(self, value: float = 0.0, width: int = 20, **kwargs) -> None:
        self._value = value
        self._width = width
        # Compute initial content before calling super().__init__()
        # This ensures there's always content to render
        initial_bar = self._compute_bar()
        super().__init__(initial_bar, **kwargs)

    def _compute_bar(self) -> str:
        """Compute the progress bar string."""
        progress = self._value / 100.0
        filled = int(progress * self._width)
        empty = self._width - filled
        return f"[$accent]{BLOCK_FULL * filled}[/][$surface]{BLOCK_EMPTY * empty}[/]"

    def set_value(self, value: float) -> None:
        self._value = max(0.0, min(100.0, value))
        self.update(self._compute_bar())


class AgentRow(Static):
    """Single agent row in the details panel."""

    DEFAULT_CSS = """
    AgentRow {
        height: 1;
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
        # Pass initial content to super().__init__() to ensure there's always
        # something to render, even before on_mount()
        super().__init__(content, **kwargs)
        self.agent_id = agent_id
        self._content = content

    def set_content(self, content: str) -> None:
        self._content = content
        self.update(content)

    def on_click(self) -> None:
        self.post_message(self.Clicked(self.agent_id))


class TaskCard(Static):
    """Task card in the task list."""

    DEFAULT_CSS = """
    TaskCard {
        height: auto;
        min-height: 5;
        padding: 1;
        margin-bottom: 1;
        background: $surface;
    }
    TaskCard:hover {
        background: $boost;
    }
    TaskCard.-selected {
        background: $primary 20%;
    }
    TaskCard .header {
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
    TaskCard .progress-row {
        height: 1;
        margin-bottom: 1;
    }
    TaskCard .progress-bar {
        width: 1fr;
    }
    TaskCard .progress-pct {
        width: 5;
        text-align: right;
        color: $foreground-muted;
    }
    TaskCard .meta {
        height: 1;
        color: $foreground-muted;
    }
    """

    class Selected(Message):
        def __init__(self, task_id: str) -> None:
            super().__init__()
            self.task_id = task_id

    def __init__(self, extraction: ExtractionUIState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.extraction = extraction
        self.task_id = extraction.id

    def compose(self) -> ComposeResult:
        ext = self.extraction
        icon, color = self._get_status_style(ext.status)

        with Horizontal(classes="header"):
            yield Static(f"[{color}]{icon}[/] [bold]{ext.behavior_name}[/]", classes="name")
            yield Static(ext.elapsed_str, classes="time")
        with Horizontal(classes="progress-row"):
            yield ProgressBar(ext.progress, width=15, classes="progress-bar")
            yield Static(f"{int(ext.progress)}%", classes="progress-pct")
        yield Static(
            f"[$accent]{ext.phase.value}[/] · {ext.running_agents_count}/{ext.total_agents_count} runs",
            classes="meta"
        )

    def _get_status_style(self, status: ExtractionStatus) -> tuple:
        status_icons = {
            ExtractionStatus.PENDING: (ICONS.pending, "$foreground-muted"),
            ExtractionStatus.RUNNING: (ICONS.running, "$accent"),
            ExtractionStatus.PAUSED: (ICONS.paused, "$warning"),
            ExtractionStatus.COMPLETE: (ICONS.complete, "$success"),
            ExtractionStatus.FAILED: (ICONS.failed, "$error"),
        }
        return status_icons.get(status, (ICONS.pending, "$foreground-muted"))

    def on_click(self) -> None:
        self.post_message(self.Selected(self.task_id))

    def set_selected(self, selected: bool) -> None:
        self.set_class(selected, "-selected")

    def set_progress(self, progress: float, phase: str) -> None:
        """Update progress display only."""
        if not self.is_mounted:
            return
        try:
            self.query_one(ProgressBar).set_value(progress)
            self.query_one(".progress-pct", Static).update(f"{int(progress)}%")
            ext = self.extraction
            self.query_one(".meta", Static).update(
                f"[$accent]{phase}[/] · {ext.running_agents_count}/{ext.total_agents_count} runs"
            )
        except Exception:
            pass

    def set_elapsed(self, elapsed: str) -> None:
        """Update elapsed time only."""
        if not self.is_mounted:
            return
        try:
            self.query_one(".time", Static).update(elapsed)
        except Exception:
            pass

    def refresh_from_state(self, extraction: ExtractionUIState) -> None:
        """Refresh all card data from extraction state."""
        self.extraction = extraction
        if not self.is_mounted:
            return

        icon, color = self._get_status_style(extraction.status)
        try:
            self.query_one(".name", Static).update(f"[{color}]{icon}[/] [bold]{extraction.behavior_name}[/]")
            self.query_one(".time", Static).update(extraction.elapsed_str)
            self.query_one(ProgressBar).set_value(extraction.progress)
            self.query_one(".progress-pct", Static).update(f"{int(extraction.progress)}%")
            self.query_one(".meta", Static).update(
                f"[$accent]{extraction.phase.value}[/] · {extraction.running_agents_count}/{extraction.total_agents_count} runs"
            )
        except Exception:
            pass


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
        self._current_task_id: str | None = None
        self._agent_rows: dict[str, AgentRow] = {}

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

    def show_task(self, extraction: ExtractionUIState | None) -> None:
        """Show details for a task."""
        empty_state = self.query_one("#empty-state")
        details_content = self.query_one("#details-content")

        if extraction is None:
            empty_state.display = True
            details_content.display = False
            self._current_task_id = None
            self._agent_rows.clear()
            return

        empty_state.display = False
        details_content.display = True
        self._current_task_id = extraction.id

        self._update_header(extraction)
        self._update_agents(extraction)
        self._update_activity(extraction.id)

    def _update_header(self, extraction: ExtractionUIState) -> None:
        """Update the header section."""
        status_map = {
            ExtractionStatus.PENDING: (ICONS.pending, "$foreground-muted"),
            ExtractionStatus.RUNNING: (ICONS.running, "$accent"),
            ExtractionStatus.PAUSED: (ICONS.paused, "$warning"),
            ExtractionStatus.COMPLETE: (ICONS.complete, "$success"),
            ExtractionStatus.FAILED: (ICONS.failed, "$error"),
        }
        icon, color = status_map.get(extraction.status, (ICONS.pending, "$foreground-muted"))

        self.query_one("#detail-title", Static).update(f"[{color}]{icon}[/] {extraction.behavior_name}")

        desc = extraction.behavior_description or "No description"
        if len(desc) > 100:
            desc = desc[:97] + "..."
        self.query_one("#detail-description", Static).update(desc)

        runs = f"{extraction.running_agents_count}/{extraction.total_agents_count}" if extraction.total_agents_count else "—"
        layer = f"L{extraction.current_layer}" if extraction.current_layer else "—"
        score = f"{extraction.evaluation.overall:.2f}" if extraction.evaluation.overall > 0 else "—"
        self.query_one("#detail-stats", Static).update(
            f"[$accent]{extraction.phase.value.upper()}[/]  │  "
            f"Runs: {runs}  │  Layer: {layer}  │  Score: {score}"
        )

    def _update_agents(self, extraction: ExtractionUIState) -> None:
        """Update the agents list."""
        runs_list = self.query_one("#runs-list", VerticalScroll)
        current_ids = set(list(extraction.agents.keys())[:8])
        existing_ids = set(self._agent_rows.keys())

        # Remove old agents
        for agent_id in existing_ids - current_ids:
            if agent_id in self._agent_rows:
                self._agent_rows[agent_id].remove()
                del self._agent_rows[agent_id]

        # Update or add agents
        for agent in list(extraction.agents.values())[:8]:
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
                self._agent_rows[agent.id].set_content(content)
            else:
                row = AgentRow(agent.id, content)
                runs_list.mount(row)
                self._agent_rows[agent.id] = row

        # Handle empty state
        for empty in runs_list.query(".empty-runs"):
            empty.remove()
        if not extraction.agents:
            runs_list.mount(Static("[$foreground-muted]No runs yet[/]", classes="empty-runs"))

    def _update_activity(self, task_id: str) -> None:
        """Update the activity list."""
        activity_list = self.query_one("#activity-list", VerticalScroll)
        activity_list.remove_children()

        logs = get_state().get_filtered_logs(extraction_id=task_id)[-5:]
        if logs:
            for log in reversed(logs):
                safe_message = escape_markup(log.message)
                activity_list.mount(Static(
                    f"[$foreground-muted]{log.time_str}[/] {safe_message}",
                    classes="log-entry"
                ))
        else:
            activity_list.mount(Static("[$foreground-muted]No activity yet[/]", classes="log-entry"))

    def add_activity_entry(self, source: str, message: str, time_str: str) -> None:
        """Add a single activity entry."""
        if not self.is_mounted or not self._current_task_id:
            return
        activity_list = self.query_one("#activity-list", VerticalScroll)

        # Add new entry at top
        safe_message = escape_markup(message)
        activity_list.mount(
            Static(f"[$foreground-muted]{time_str}[/] {safe_message}", classes="log-entry"),
            before=0
        )

        # Keep only 5 entries
        entries = list(activity_list.query(".log-entry"))
        while len(entries) > 5:
            entries[-1].remove()
            entries = entries[:-1]


class DeleteConfirmModal(ModalScreen):
    """Modal for confirming task deletion."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "confirm", "Confirm"),
    ]

    DEFAULT_CSS = """
    DeleteConfirmModal {
        align: center middle;
    }
    DeleteConfirmModal #modal {
        width: 50;
        height: auto;
        background: $surface;
        padding: 1 2;
    }
    DeleteConfirmModal .title {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
    }
    DeleteConfirmModal .message {
        height: auto;
        margin-bottom: 1;
        color: $foreground-muted;
    }
    DeleteConfirmModal .buttons {
        height: 3;
        align: center middle;
    }
    DeleteConfirmModal Button {
        margin: 0 1;
    }
    """

    class Confirmed(Message):
        def __init__(self, task_id: str) -> None:
            super().__init__()
            self.task_id = task_id

    def __init__(self, task_id: str, task_name: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.task_id = task_id
        self.task_name = task_name

    def compose(self) -> ComposeResult:
        with Vertical(id="modal"):
            yield Static("[$warning]Delete Task?[/]", classes="title")
            yield Static(f"Are you sure you want to delete '{self.task_name}'?", classes="message")
            with Horizontal(classes="buttons"):
                yield Button("Cancel", variant="default", id="cancel")
                yield Button("Delete", variant="error", id="confirm")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm":
            self.post_message(self.Confirmed(self.task_id))
        self.dismiss()

    def action_cancel(self) -> None:
        self.dismiss()

    def action_confirm(self) -> None:
        self.post_message(self.Confirmed(self.task_id))
        self.dismiss()


class DashboardScreen(Screen):
    """Dashboard screen with task list and details.

    Event handlers update specific widgets - no full rebuilds.
    """

    BINDINGS = [
        Binding("1", "noop", "Dashboard", show=False),
        Binding("2", "go_samples", "Samples", key_display="2"),
        Binding("3", "go_logs", "Logs", key_display="3"),
        Binding("tab", "cycle", "Next Screen"),
        Binding("j", "select_next", "Next", show=False),
        Binding("k", "select_prev", "Previous", show=False),
        Binding("down", "select_next", "Next", show=False),
        Binding("up", "select_prev", "Previous", show=False),
        Binding("enter", "open_samples", "Open", show=False),
        Binding("d", "delete_task", "Delete", show=False),
        Binding("n", "new_task", "New Task"),
        Binding("q", "quit", "Quit"),
    ]

    DEFAULT_CSS = """
    DashboardScreen {
        background: $background;
    }
    DashboardScreen #content {
        height: 1fr;
    }
    DashboardScreen #left {
        width: 1fr;
        max-width: 50;
        padding: 1 2;
    }
    DashboardScreen #tasks {
        height: 1fr;
    }
    DashboardScreen #right {
        width: 2fr;
    }
    DashboardScreen .empty-tasks {
        height: 1fr;
        content-align: center middle;
        color: $foreground-muted;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._task_cards: dict[str, TaskCard] = {}
        self._selected_task_id: str | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="content"):
            with Vertical(id="left"):
                yield VerticalScroll(id="tasks")
            yield DetailsPanel(id="right")
        yield TmuxBar(active_screen="dashboard")

    def on_mount(self) -> None:
        """Initial projection from current state."""
        state = get_state()
        tasks_container = self.query_one("#tasks", VerticalScroll)

        # Mount all existing tasks
        for task_id, extraction in state.extractions.items():
            card = TaskCard(extraction)
            self._task_cards[task_id] = card
            tasks_container.mount(card)

        # Select first or previously selected
        if state.selected_id and state.selected_id in state.extractions:
            self._select_task(state.selected_id)
        elif state.extractions:
            first_id = next(iter(state.extractions))
            self._select_task(first_id)
        else:
            tasks_container.mount(Static("No tasks yet. Press 'n' to create one.", classes="empty-tasks"))

    # ─────────────────────────────────────────────────────────────────────────
    # Event Handlers - Targeted Updates
    # ─────────────────────────────────────────────────────────────────────────

    def on_task_created(self, event: TaskCreated) -> None:
        """Handle new task - add one card."""
        state = get_state()
        extraction = state.extractions.get(event.task_id)
        if not extraction:
            return

        tasks_container = self.query_one("#tasks", VerticalScroll)

        # Remove empty message if present
        for empty in tasks_container.query(".empty-tasks"):
            empty.remove()

        # Add new card
        card = TaskCard(extraction)
        self._task_cards[event.task_id] = card
        tasks_container.mount(card)

        # Select it
        self._select_task(event.task_id)

    def on_task_progress_changed(self, event: TaskProgressChanged) -> None:
        """Handle progress update - update one card."""
        card = self._task_cards.get(event.task_id)
        if card:
            card.set_progress(event.progress, event.phase)

        # Update details if showing this task
        if self._selected_task_id == event.task_id:
            state = get_state()
            extraction = state.extractions.get(event.task_id)
            if extraction:
                self.query_one("#right", DetailsPanel)._update_header(extraction)

    def on_task_status_changed(self, event: TaskStatusChanged) -> None:
        """Handle status change - update card and details."""
        state = get_state()
        extraction = state.extractions.get(event.task_id)
        if extraction:
            card = self._task_cards.get(event.task_id)
            if card:
                card.refresh_from_state(extraction)

            if self._selected_task_id == event.task_id:
                self.query_one("#right", DetailsPanel)._update_header(extraction)

    def on_task_removed(self, event: TaskRemoved) -> None:
        """Handle task removal - remove one card."""
        card = self._task_cards.pop(event.task_id, None)
        if card:
            card.remove()

        # Select another task if this was selected
        if self._selected_task_id == event.task_id:
            state = get_state()
            if state.extractions:
                self._select_task(next(iter(state.extractions)))
            else:
                self._selected_task_id = None
                self.query_one("#right", DetailsPanel).show_task(None)
                tasks_container = self.query_one("#tasks", VerticalScroll)
                tasks_container.mount(Static("No tasks yet. Press 'n' to create one.", classes="empty-tasks"))

    def on_task_selected(self, event: TaskSelected) -> None:
        """Handle selection change from app."""
        if event.task_id:
            self._select_task(event.task_id)

    def on_agent_spawned(self, event: AgentSpawned) -> None:
        """Handle agent spawn - update details if showing this task."""
        if self._selected_task_id == event.task_id:
            state = get_state()
            extraction = state.extractions.get(event.task_id)
            if extraction:
                self.query_one("#right", DetailsPanel)._update_agents(extraction)

    def on_agent_status_changed(self, event: AgentStatusChanged) -> None:
        """Handle agent status change - update details if showing this task."""
        if self._selected_task_id == event.task_id:
            state = get_state()
            extraction = state.extractions.get(event.task_id)
            if extraction:
                self.query_one("#right", DetailsPanel)._update_agents(extraction)
                card = self._task_cards.get(event.task_id)
                if card:
                    card.refresh_from_state(extraction)

    def on_log_emitted(self, event: LogEmitted) -> None:
        """Handle new log - update activity if showing this task."""
        if event.task_id and event.task_id == self._selected_task_id:
            from datetime import datetime
            time_str = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
            self.query_one("#right", DetailsPanel).add_activity_entry(
                event.source, event.message, time_str
            )

    def on_time_tick(self, event: TimeTick) -> None:
        """Handle time tick - update elapsed times."""
        state = get_state()
        for task_id, card in self._task_cards.items():
            extraction = state.extractions.get(task_id)
            if extraction and extraction.status == ExtractionStatus.RUNNING:
                card.set_elapsed(extraction.elapsed_str)

        self.query_one(TmuxBar).refresh_info()

    # ─────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _select_task(self, task_id: str) -> None:
        """Select a task and update UI."""
        # Deselect previous
        if self._selected_task_id and self._selected_task_id in self._task_cards:
            self._task_cards[self._selected_task_id].set_selected(False)

        # Select new
        self._selected_task_id = task_id
        if task_id in self._task_cards:
            self._task_cards[task_id].set_selected(True)

        # Update details panel
        state = get_state()
        extraction = state.extractions.get(task_id)
        self.query_one("#right", DetailsPanel).show_task(extraction)

        # Update state
        state.selected_id = task_id

    def on_task_card_selected(self, event: TaskCard.Selected) -> None:
        """Handle card click."""
        self._select_task(event.task_id)

    def on_agent_row_clicked(self, event: AgentRow.Clicked) -> None:
        """Handle agent row click - open samples."""
        state = get_state()
        if self._selected_task_id:
            extraction = state.extractions.get(self._selected_task_id)
            if extraction:
                extraction.selected_agent_id = event.agent_id
        self.app.switch_screen("samples")

    def on_delete_confirm_modal_confirmed(self, event: DeleteConfirmModal.Confirmed) -> None:
        """Handle delete confirmation."""
        self.app.post_message(TaskRemoved(task_id=event.task_id))

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

    def action_select_next(self) -> None:
        task_ids = list(self._task_cards.keys())
        if not task_ids:
            return
        if self._selected_task_id is None:
            self._select_task(task_ids[0])
        else:
            try:
                idx = task_ids.index(self._selected_task_id)
                next_idx = (idx + 1) % len(task_ids)
                self._select_task(task_ids[next_idx])
            except ValueError:
                self._select_task(task_ids[0])

    def action_select_prev(self) -> None:
        task_ids = list(self._task_cards.keys())
        if not task_ids:
            return
        if self._selected_task_id is None:
            self._select_task(task_ids[-1])
        else:
            try:
                idx = task_ids.index(self._selected_task_id)
                prev_idx = (idx - 1) % len(task_ids)
                self._select_task(task_ids[prev_idx])
            except ValueError:
                self._select_task(task_ids[-1])

    def action_open_samples(self) -> None:
        self.app.switch_screen("samples")

    def action_delete_task(self) -> None:
        if self._selected_task_id:
            state = get_state()
            extraction = state.extractions.get(self._selected_task_id)
            if extraction:
                self.app.push_screen(DeleteConfirmModal(
                    task_id=self._selected_task_id,
                    task_name=extraction.behavior_name,
                ))

    def action_new_task(self) -> None:
        self.app.push_screen("create_task")

    def action_quit(self) -> None:
        self.app.exit()
