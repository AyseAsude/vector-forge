"""Evaluation scores panel widget with block-character bars."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from vector_forge.ui.theme import COLORS, ICONS
from vector_forge.ui.state import EvaluationMetrics


# Block characters for mini score bars
BLOCK_FULL = "█"
BLOCK_1_2 = "▌"
BLOCK_EMPTY = "░"


class ScoreBar(Widget):
    """Single score display with label and block-character bar."""

    DEFAULT_CSS = """
    ScoreBar {
        height: 1;
        layout: horizontal;
    }

    ScoreBar .score-label {
        width: 11;
        color: $text-muted;
    }

    ScoreBar .score-bar-container {
        width: 10;
    }

    ScoreBar .score-value {
        width: 5;
        color: $text;
        margin-left: 1;
        text-align: right;
    }
    """

    label: reactive[str] = reactive("", init=False)
    score: reactive[float] = reactive(0.0, init=False)

    def __init__(self, label: str = "", score: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._init_label = label
        self._init_score = score

    def compose(self) -> ComposeResult:
        yield Static(self._init_label, classes="score-label")
        yield Static(classes="score-bar-container")
        yield Static(classes="score-value")

    def on_mount(self) -> None:
        self.label = self._init_label
        self.score = self._init_score
        self._update_display()

    def watch_label(self, label: str) -> None:
        if self.is_mounted:
            self._update_display()

    def watch_score(self, score: float) -> None:
        if self.is_mounted:
            self._update_display()

    def _update_display(self) -> None:
        if not self.is_mounted:
            return

        label_widget = self.query_one(".score-label", Static)
        label_widget.update(self.label)

        bar_container = self.query_one(".score-bar-container", Static)
        bar_width = 8

        # Calculate fill
        filled = int(self.score * bar_width)
        empty = bar_width - filled

        # Choose color based on score
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

        value_widget = self.query_one(".score-value", Static)
        value_widget.update(f"{self.score:.2f}")


class VerdictDisplay(Widget):
    """Displays judge verdict with appropriate styling."""

    DEFAULT_CSS = """
    VerdictDisplay {
        height: 1;
    }
    """

    verdict: reactive[str | None] = reactive(None, init=False)

    def compose(self) -> ComposeResult:
        yield Static(id="verdict-text")

    def on_mount(self) -> None:
        self._update_verdict()

    def watch_verdict(self, verdict: str | None) -> None:
        if self.is_mounted:
            self._update_verdict()

    def _update_verdict(self) -> None:
        if not self.is_mounted:
            return
        text_widget = self.query_one("#verdict-text", Static)
        verdict = self.verdict

        if verdict is None:
            text_widget.update(f"[{COLORS.text_muted}]{ICONS.pending} Pending[/]")
        elif verdict.lower() == "accepted":
            text_widget.update(f"[{COLORS.success}]{ICONS.complete} Accepted[/]")
        elif verdict.lower() == "needs_refinement":
            text_widget.update(f"[{COLORS.warning}]{ICONS.review} Needs Refinement[/]")
        elif verdict.lower() == "rejected":
            text_widget.update(f"[{COLORS.error}]{ICONS.failed} Rejected[/]")
        else:
            text_widget.update(f"[{COLORS.text_muted}]{ICONS.pending} {verdict}[/]")


class EvaluationPanel(Widget):
    """Panel displaying evaluation scores and verdict."""

    DEFAULT_CSS = """
    EvaluationPanel {
        width: 1fr;
        height: auto;
        background: $surface;
        padding: 1 2;
        margin: 0 0 1 0;
    }

    EvaluationPanel #eval-panel-title {
        color: $accent;
        text-style: bold;
        height: 1;
        margin-bottom: 1;
    }

    EvaluationPanel #scores-section {
        height: auto;
    }

    EvaluationPanel #overall-divider {
        height: 1;
        color: $text-disabled;
    }

    EvaluationPanel #best-result {
        height: 1;
        color: $text-muted;
        margin-top: 1;
    }

    EvaluationPanel #verdict-row {
        height: 1;
        margin-top: 0;
    }
    """

    metrics: reactive[EvaluationMetrics] = reactive(EvaluationMetrics, init=False)

    def compose(self) -> ComposeResult:
        yield Static("Evaluation", id="eval-panel-title")
        with Vertical(id="scores-section"):
            yield ScoreBar(label="Behavior", id="score-behavior")
            yield ScoreBar(label="Coherence", id="score-coherence")
            yield ScoreBar(label="Specificity", id="score-specificity")
            yield Static("───────────────────", id="overall-divider")
            yield ScoreBar(label="Overall", id="score-overall")
        yield Static(id="best-result")
        yield VerdictDisplay(id="verdict-row")

    def watch_metrics(self, metrics: EvaluationMetrics) -> None:
        if not self.is_mounted:
            return
        self.query_one("#score-behavior", ScoreBar).score = metrics.behavior
        self.query_one("#score-coherence", ScoreBar).score = metrics.coherence
        self.query_one("#score-specificity", ScoreBar).score = metrics.specificity
        self.query_one("#score-overall", ScoreBar).score = metrics.overall

        best_widget = self.query_one("#best-result", Static)
        if metrics.best_layer is not None:
            best_widget.update(
                f"Best: layer {metrics.best_layer} @ {metrics.best_strength:.1f}x"
            )
        else:
            best_widget.update("Best: --")

        verdict_display = self.query_one(VerdictDisplay)
        verdict_display.verdict = metrics.verdict

    def set_metrics(self, metrics: EvaluationMetrics) -> None:
        self.metrics = metrics
