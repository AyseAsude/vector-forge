"""Dashboard screen - clean overview of extraction tasks."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, Container, Grid
from textual.screen import Screen
from textual.widgets import Static, ProgressBar, Label
from textual.reactive import reactive

from vector_forge.ui.state import (
    ExtractionUIState,
    ExtractionStatus,
    get_state,
)
from vector_forge.ui.theme import COLORS, ICONS


class TaskCard(Static):
    """A card showing a single task with progress."""

    DEFAULT_CSS = """
    TaskCard {
        height: auto;
        min-height: 5;
        padding: 1 2;
        background: $surface;
        margin: 0 0 1 0;
    }

    TaskCard:hover {
        background: $surface-hl;
    }

    TaskCard.-selected {
        border-left: wide $accent;
    }

    TaskCard .task-header {
        height: 1;
        margin-bottom: 1;
    }

    TaskCard .task-name {
        width: 1fr;
    }

    TaskCard .task-status {
        width: auto;
    }

    TaskCard ProgressBar {
        height: 1;
        margin-bottom: 1;
    }

    TaskCard ProgressBar Bar {
        width: 1fr;
    }

    TaskCard ProgressBar PercentageStatus {
        width: 5;
        text-align: right;
    }

    TaskCard ProgressBar ETAStatus {
        display: none;
    }

    TaskCard .task-metrics {
        height: 1;
        color: $text-muted;
    }
    """

    def __init__(self, extraction: ExtractionUIState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.extraction = extraction

    def compose(self) -> ComposeResult:
        with Horizontal(classes="task-header"):
            yield Static(classes="task-name")
            yield Static(classes="task-status")
        yield ProgressBar(total=100, show_eta=False)
        yield Static(classes="task-metrics")

    def on_mount(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        ext = self.extraction

        # Status styling
        status_styles = {
            ExtractionStatus.PENDING: (ICONS.pending, COLORS.text_dim),
            ExtractionStatus.RUNNING: (ICONS.running, COLORS.accent),
            ExtractionStatus.PAUSED: (ICONS.paused, COLORS.warning),
            ExtractionStatus.COMPLETE: (ICONS.complete, COLORS.success),
            ExtractionStatus.FAILED: (ICONS.failed, COLORS.error),
        }
        icon, color = status_styles.get(ext.status, (ICONS.pending, COLORS.text_dim))

        # Update header
        name = self.query_one(".task-name", Static)
        name.update(f"[bold]{ext.behavior_name}[/]")

        status = self.query_one(".task-status", Static)
        status.update(f"[{color}]{icon} {ext.status.value}[/]  [{COLORS.text_dim}]{ext.elapsed_str}[/]")

        # Update progress
        progress = self.query_one(ProgressBar)
        progress.update(progress=ext.progress * 100)

        # Update metrics
        metrics = self.query_one(".task-metrics", Static)
        running = ext.running_agents_count
        total = ext.total_agents_count
        layer_str = f"L{ext.current_layer}" if ext.current_layer else "—"

        metrics.update(
            f"Phase: {ext.phase.value.upper()}  |  "
            f"Samples: {running}/{total}  |  "
            f"Layer: {layer_str}  |  "
            f"Score: {ext.evaluation.overall:.2f}"
        )

    def update_extraction(self, extraction: ExtractionUIState) -> None:
        self.extraction = extraction
        if self.is_mounted:
            self._refresh()


class MetricBox(Static):
    """A small metric display box."""

    DEFAULT_CSS = """
    MetricBox {
        width: 1fr;
        height: 5;
        padding: 1 2;
        background: $surface;
        content-align: center middle;
    }

    MetricBox .metric-value {
        text-align: center;
        text-style: bold;
        width: 100%;
    }

    MetricBox .metric-label {
        text-align: center;
        color: $text-muted;
        width: 100%;
    }
    """

    def __init__(self, label: str, value: str = "—", color: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._value = value
        self._color = color or COLORS.text

    def compose(self) -> ComposeResult:
        yield Static(classes="metric-value")
        yield Static(classes="metric-label")

    def on_mount(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        value_widget = self.query_one(".metric-value", Static)
        value_widget.update(f"[{self._color}]{self._value}[/]")

        label_widget = self.query_one(".metric-label", Static)
        label_widget.update(self._label)

    def set_value(self, value: str, color: str = "") -> None:
        self._value = value
        if color:
            self._color = color
        if self.is_mounted:
            self._refresh()


class DashboardScreen(Screen):
    """Main dashboard showing task overview."""

    BINDINGS = [
        Binding("1", "noop", "dashboard", show=False),
        Binding("2", "switch_samples", "samples"),
        Binding("3", "switch_logs", "logs"),
        Binding("tab", "cycle_screen", "cycle", show=False),
        Binding("q", "quit", "quit"),
        Binding("n", "new_task", "new task"),
        Binding("?", "show_help", "help"),
    ]

    DEFAULT_CSS = """
    DashboardScreen {
        background: $background;
    }

    DashboardScreen #header {
        height: 3;
        padding: 1 2;
        background: $surface;
        border-bottom: solid $border;
    }

    DashboardScreen #header-title {
        width: 1fr;
        text-style: bold;
    }

    DashboardScreen #header-stats {
        width: auto;
        color: $text-muted;
    }

    DashboardScreen #content {
        height: 1fr;
        padding: 1 2;
    }

    DashboardScreen #tasks-section {
        width: 2fr;
        padding-right: 1;
    }

    DashboardScreen #tasks-header {
        height: 2;
        color: $text-muted;
    }

    DashboardScreen #tasks-scroll {
        height: 1fr;
    }

    DashboardScreen #metrics-section {
        width: 1fr;
    }

    DashboardScreen #metrics-header {
        height: 2;
        color: $text-muted;
    }

    DashboardScreen #metrics-grid {
        grid-size: 2 3;
        grid-gutter: 1;
        height: auto;
    }

    DashboardScreen #footer {
        dock: bottom;
        height: 1;
        background: $surface;
        padding: 0 2;
    }

    DashboardScreen #footer-left {
        width: 1fr;
    }

    DashboardScreen #footer-right {
        width: auto;
        color: $text-muted;
    }
    """

    def compose(self) -> ComposeResult:
        # Header
        with Horizontal(id="header"):
            yield Static("Vector Forge", id="header-title")
            yield Static(id="header-stats")

        # Content area
        with Horizontal(id="content"):
            # Tasks list
            with Vertical(id="tasks-section"):
                yield Static("ACTIVE TASKS", id="tasks-header")
                yield Container(id="tasks-scroll")

            # Metrics sidebar
            with Vertical(id="metrics-section"):
                yield Static("METRICS", id="metrics-header")
                with Grid(id="metrics-grid"):
                    yield MetricBox("Running", id="metric-running")
                    yield MetricBox("Complete", id="metric-complete")
                    yield MetricBox("Best Score", id="metric-score")
                    yield MetricBox("Samples", id="metric-samples")
                    yield MetricBox("Vectors", id="metric-vectors")
                    yield MetricBox("Duration", id="metric-duration")

        # Footer
        with Horizontal(id="footer"):
            yield Static(id="footer-left")
            yield Static("n: new task  |  2: samples  |  3: logs  |  q: quit", id="footer-right")

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

        # Update header stats
        stats = self.query_one("#header-stats", Static)
        stats.update(f"{state.running_count} running  |  {state.complete_count}/{state.total_count} complete")

        # Update tasks
        self._refresh_tasks(state)

        # Update metrics
        self._refresh_metrics(state)

        # Update footer
        footer_left = self.query_one("#footer-left", Static)
        if state.selected_extraction:
            ext = state.selected_extraction
            footer_left.update(
                f"[{COLORS.accent}]{ICONS.active}[/] {ext.behavior_name}  "
                f"[{COLORS.text_dim}]|[/]  {ext.phase.value.upper()}"
            )
        else:
            footer_left.update(f"[{COLORS.text_dim}]No active task. Press 'n' to create one.[/]")

    def _refresh_tasks(self, state) -> None:
        container = self.query_one("#tasks-scroll", Container)

        # Get current cards
        existing_cards = {card.extraction.id: card for card in self.query(TaskCard)}

        # Update or create cards
        for ext_id, extraction in state.extractions.items():
            if ext_id in existing_cards:
                existing_cards[ext_id].update_extraction(extraction)
            else:
                card = TaskCard(extraction)
                container.mount(card)

        # Remove old cards
        for ext_id, card in existing_cards.items():
            if ext_id not in state.extractions:
                card.remove()

    def _refresh_metrics(self, state) -> None:
        # Running count
        running = self.query_one("#metric-running", MetricBox)
        running.set_value(str(state.running_count), COLORS.accent if state.running_count > 0 else COLORS.text_dim)

        # Complete count
        complete = self.query_one("#metric-complete", MetricBox)
        complete.set_value(str(state.complete_count), COLORS.success if state.complete_count > 0 else COLORS.text_dim)

        # Best score
        best_score = 0.0
        total_samples = 0
        total_vectors = 0
        total_duration = 0.0

        for ext in state.extractions.values():
            if ext.evaluation.overall > best_score:
                best_score = ext.evaluation.overall
            total_samples += ext.total_agents_count
            total_duration += ext.elapsed_seconds

        score_widget = self.query_one("#metric-score", MetricBox)
        score_color = COLORS.success if best_score >= 0.8 else (COLORS.accent if best_score >= 0.5 else COLORS.text_dim)
        score_widget.set_value(f"{best_score:.2f}", score_color)

        # Samples
        samples = self.query_one("#metric-samples", MetricBox)
        samples.set_value(str(total_samples))

        # Vectors (placeholder - would come from actual vector extraction count)
        vectors = self.query_one("#metric-vectors", MetricBox)
        vectors.set_value(str(state.complete_count))

        # Duration
        duration = self.query_one("#metric-duration", MetricBox)
        mins = int(total_duration) // 60
        secs = int(total_duration) % 60
        duration.set_value(f"{mins:02d}:{secs:02d}")

    def _refresh_timers(self) -> None:
        """Refresh time-based displays."""
        state = get_state()
        for card in self.query(TaskCard):
            ext = state.extractions.get(card.extraction.id)
            if ext and ext.status == ExtractionStatus.RUNNING:
                card.update_extraction(ext)

    def action_noop(self) -> None:
        pass

    def action_switch_samples(self) -> None:
        self.app.switch_screen("samples")

    def action_switch_logs(self) -> None:
        self.app.switch_screen("logs")

    def action_cycle_screen(self) -> None:
        self.app.switch_screen("samples")

    def action_new_task(self) -> None:
        self.app.push_screen("create_task")

    def action_show_help(self) -> None:
        self.app.push_screen("help")

    def action_quit(self) -> None:
        self.app.exit()
