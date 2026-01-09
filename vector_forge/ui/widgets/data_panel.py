"""Datapoints metrics panel widget."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from vector_forge.ui.theme import COLORS, ICONS
from vector_forge.ui.state import DatapointMetrics


class MetricRow(Widget):
    """Single metric display row."""

    DEFAULT_CSS = """
    MetricRow {
        height: 1;
        layout: horizontal;
    }

    MetricRow .metric-label {
        width: 12;
        color: $text-muted;
    }

    MetricRow .metric-value {
        width: 1fr;
        color: $text;
    }
    """

    label: reactive[str] = reactive("", init=False)
    value: reactive[str] = reactive("", init=False)

    def __init__(self, label: str = "", value: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._init_label = label
        self._init_value = value

    def compose(self) -> ComposeResult:
        yield Static(self._init_label, classes="metric-label")
        yield Static(self._init_value, classes="metric-value")

    def on_mount(self) -> None:
        self.label = self._init_label
        self.value = self._init_value

    def watch_label(self, label: str) -> None:
        if not self.is_mounted:
            return
        label_widgets = self.query(".metric-label")
        if label_widgets:
            label_widgets.first(Static).update(label)

    def watch_value(self, value: str) -> None:
        if not self.is_mounted:
            return
        value_widgets = self.query(".metric-value")
        if value_widgets:
            value_widgets.first(Static).update(value)


class QualityBreakdown(Widget):
    """Displays datapoint quality breakdown with icons."""

    DEFAULT_CSS = """
    QualityBreakdown {
        height: 1;
        layout: horizontal;
    }

    QualityBreakdown .quality-item {
        width: auto;
        margin-right: 2;
    }
    """

    keep: reactive[int] = reactive(0, init=False)
    review: reactive[int] = reactive(0, init=False)
    remove: reactive[int] = reactive(0, init=False)

    def compose(self) -> ComposeResult:
        yield Static(id="keep-item", classes="quality-item")
        yield Static(id="review-item", classes="quality-item")
        yield Static(id="remove-item", classes="quality-item")

    def on_mount(self) -> None:
        self._update_display()

    def watch_keep(self, keep: int) -> None:
        if self.is_mounted:
            self._update_display()

    def watch_review(self, review: int) -> None:
        if self.is_mounted:
            self._update_display()

    def watch_remove(self, remove: int) -> None:
        if self.is_mounted:
            self._update_display()

    def _update_display(self) -> None:
        if not self.is_mounted:
            return
        keep_widget = self.query_one("#keep-item", Static)
        keep_widget.update(f"[{COLORS.success}]{ICONS.keep} {self.keep} keep[/]")

        review_widget = self.query_one("#review-item", Static)
        review_widget.update(f"[{COLORS.warning}]{ICONS.review} {self.review} review[/]")

        remove_widget = self.query_one("#remove-item", Static)
        remove_widget.update(f"[{COLORS.text_dim}]{ICONS.remove} {self.remove}[/]")


class DataPanel(Widget):
    """Panel displaying datapoint metrics."""

    DEFAULT_CSS = """
    DataPanel {
        width: 1fr;
        height: auto;
        background: $surface;
        padding: 1 2;
        margin: 0 1 1 0;
    }

    DataPanel #data-panel-title {
        color: $accent;
        text-style: bold;
        height: 1;
        margin-bottom: 1;
    }

    DataPanel #quality-row {
        height: 1;
        margin-top: 0;
    }

    DataPanel #diversity-section {
        margin-top: 1;
    }
    """

    metrics: reactive[DatapointMetrics] = reactive(DatapointMetrics, init=False)

    def compose(self) -> ComposeResult:
        yield Static("Data", id="data-panel-title")
        yield MetricRow(label="Datapoints", id="datapoints-count")
        yield QualityBreakdown(id="quality-row")
        with Vertical(id="diversity-section"):
            yield MetricRow(label="Diversity", id="diversity-value")
            yield MetricRow(label="Clusters", id="clusters-value")

    def watch_metrics(self, metrics: DatapointMetrics) -> None:
        if not self.is_mounted:
            return
        count_row = self.query_one("#datapoints-count", MetricRow)
        count_row.value = str(metrics.total)

        quality = self.query_one(QualityBreakdown)
        quality.keep = metrics.keep
        quality.review = metrics.review
        quality.remove = metrics.remove

        diversity_row = self.query_one("#diversity-value", MetricRow)
        diversity_row.value = f"{metrics.diversity:.2f}"

        clusters_row = self.query_one("#clusters-value", MetricRow)
        clusters_row.value = str(metrics.clusters)

    def set_metrics(self, metrics: DatapointMetrics) -> None:
        self.metrics = metrics
