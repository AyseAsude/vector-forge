"""Extraction detail widget for parallel view."""

from typing import Optional

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from vector_forge.ui.theme import COLORS, ICONS
from vector_forge.ui.state import ExtractionUIState, ExtractionStatus


# Block characters for mini score bars
BLOCK_FULL = "█"
BLOCK_EMPTY = "░"


class MiniScoreBar(Widget):
    """Compact score bar for detail view using block characters."""

    DEFAULT_CSS = """
    MiniScoreBar {
        height: 1;
        layout: horizontal;
    }

    MiniScoreBar .mini-score-label {
        width: 11;
        color: $text-muted;
    }

    MiniScoreBar .mini-score-bar-container {
        width: 8;
    }

    MiniScoreBar .mini-score-value {
        width: 5;
        color: $text;
        margin-left: 1;
    }
    """

    label: reactive[str] = reactive("", init=False)
    score: reactive[float] = reactive(0.0, init=False)

    def __init__(self, label: str = "", score: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._init_label = label
        self._init_score = score

    def compose(self) -> ComposeResult:
        yield Static(self._init_label, classes="mini-score-label")
        yield Static(classes="mini-score-bar-container")
        yield Static(classes="mini-score-value")

    def on_mount(self) -> None:
        self.label = self._init_label
        self.score = self._init_score
        self._update_display()

    def watch_score(self, score: float) -> None:
        if self.is_mounted:
            self._update_display()

    def _update_display(self) -> None:
        if not self.is_mounted:
            return

        label_widget = self.query_one(".mini-score-label", Static)
        label_widget.update(self.label)

        bar_container = self.query_one(".mini-score-bar-container", Static)
        bar_width = 6

        filled = int(self.score * bar_width)
        empty = bar_width - filled

        # Color based on score
        if self.score >= 0.8:
            bar_color = COLORS.success
        elif self.score >= 0.6:
            bar_color = COLORS.accent
        elif self.score >= 0.4:
            bar_color = COLORS.warning
        else:
            bar_color = COLORS.error

        bar_str = (
            f"[{bar_color}]{BLOCK_FULL * filled}[/]"
            f"[{COLORS.surface_hl}]{BLOCK_EMPTY * empty}[/]"
        )
        bar_container.update(bar_str)

        value_widget = self.query_one(".mini-score-value", Static)
        value_widget.update(f"{self.score:.2f}")


class ExtractionDetail(Widget):
    """Detailed view of a selected extraction."""

    DEFAULT_CSS = """
    ExtractionDetail {
        height: auto;
        min-height: 6;
        background: $surface;
        padding: 1 2;
        margin: 0 0 1 0;
    }

    ExtractionDetail #detail-title {
        color: $accent;
        text-style: bold;
        height: 1;
        margin-bottom: 1;
    }

    ExtractionDetail #detail-empty {
        color: $text-disabled;
    }

    ExtractionDetail #detail-content {
        layout: horizontal;
    }

    ExtractionDetail #detail-left {
        width: 1fr;
    }

    ExtractionDetail #detail-right {
        width: 1fr;
    }

    ExtractionDetail .detail-section-title {
        color: $text-muted;
        text-style: bold;
        height: 1;
    }

    ExtractionDetail .detail-row {
        height: 1;
        color: $text;
    }

    ExtractionDetail .detail-label {
        color: $text-muted;
    }

    ExtractionDetail #detail-activity {
        margin-top: 1;
    }

    ExtractionDetail #detail-verdict {
        margin-top: 1;
    }
    """

    extraction: reactive[Optional[ExtractionUIState]] = reactive(None, init=False)

    def compose(self) -> ComposeResult:
        yield Static("Selected", id="detail-title")
        yield Static("Select an extraction to view details", id="detail-empty")
        with Horizontal(id="detail-content"):
            with Vertical(id="detail-left"):
                yield Static("Datapoints", classes="detail-section-title")
                yield Static(id="datapoints-info", classes="detail-row")
                yield Static(id="diversity-info", classes="detail-row")
                yield Static(id="detail-activity")
            with Vertical(id="detail-right"):
                yield Static("Scores", classes="detail-section-title")
                yield MiniScoreBar(label="Behavior", id="score-behavior")
                yield MiniScoreBar(label="Coherence", id="score-coherence")
                yield MiniScoreBar(label="Specificity", id="score-specificity")
                yield MiniScoreBar(label="Overall", id="score-overall")
                yield Static(id="detail-verdict")

    def on_mount(self) -> None:
        self._update_visibility()

    def watch_extraction(self, extraction: Optional[ExtractionUIState]) -> None:
        if not self.is_mounted:
            return
        self._update_visibility()
        if extraction:
            self._update_content()

    def _update_visibility(self) -> None:
        if not self.is_mounted:
            return
        empty_msg = self.query_one("#detail-empty", Static)
        content = self.query_one("#detail-content", Horizontal)

        if self.extraction is None:
            empty_msg.display = True
            content.display = False
        else:
            empty_msg.display = False
            content.display = True

    def _update_content(self) -> None:
        if self.extraction is None:
            return

        ext = self.extraction

        # Update title
        title = self.query_one("#detail-title", Static)
        title.update(f"Selected: {ext.behavior_name}")

        # Update datapoints info
        dp = ext.datapoints
        dp_info = self.query_one("#datapoints-info", Static)
        dp_info.update(
            f"[{COLORS.text_muted}]Datapoints:[/] {dp.total} "
            f"([{COLORS.success}]{ICONS.keep} {dp.keep}[/] "
            f"[{COLORS.warning}]{ICONS.review} {dp.review}[/] "
            f"[{COLORS.text_dim}]{ICONS.remove} {dp.remove}[/])"
        )

        # Update diversity info
        div_info = self.query_one("#diversity-info", Static)
        div_info.update(
            f"[{COLORS.text_muted}]Diversity:[/] {dp.diversity:.2f} · "
            f"[{COLORS.text_muted}]Clusters:[/] {dp.clusters}"
        )

        # Update scores
        ev = ext.evaluation
        self.query_one("#score-behavior", MiniScoreBar).score = ev.behavior
        self.query_one("#score-coherence", MiniScoreBar).score = ev.coherence
        self.query_one("#score-specificity", MiniScoreBar).score = ev.specificity
        self.query_one("#score-overall", MiniScoreBar).score = ev.overall

        # Update activity
        activity_widget = self.query_one("#detail-activity", Static)
        if ext.activity:
            last_activity = ext.activity[-1]
            icon_colors = {
                "active": COLORS.accent,
                "success": COLORS.success,
                "error": COLORS.error,
                "waiting": COLORS.text_muted,
            }
            color = icon_colors.get(last_activity.status, COLORS.text_muted)
            activity_widget.update(
                f"[{color}]{last_activity.icon}[/] {last_activity.message}"
            )
        else:
            activity_widget.update(f"[{COLORS.text_dim}]No activity[/]")

        # Update verdict
        verdict_widget = self.query_one("#detail-verdict", Static)
        verdict = ev.verdict
        if verdict is None:
            verdict_widget.update(f"[{COLORS.text_muted}]{ICONS.pending} Pending[/]")
        elif verdict.lower() == "accepted":
            verdict_widget.update(f"[{COLORS.success}]{ICONS.complete} Accepted[/]")
        elif verdict.lower() == "needs_refinement":
            verdict_widget.update(f"[{COLORS.warning}]{ICONS.review} Needs Refinement[/]")
        elif verdict.lower() == "rejected":
            verdict_widget.update(f"[{COLORS.error}]{ICONS.failed} Rejected[/]")
        else:
            verdict_widget.update(f"[{COLORS.text_muted}]{ICONS.pending} {verdict}[/]")

    def set_extraction(self, extraction: Optional[ExtractionUIState]) -> None:
        self.extraction = extraction
