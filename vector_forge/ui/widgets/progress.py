"""Custom block-character progress bar widget."""

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from vector_forge.ui.state import Phase
from vector_forge.ui.theme import COLORS


# Block characters for progress bar
BLOCK_FULL = "█"
BLOCK_7_8 = "▉"
BLOCK_3_4 = "▊"
BLOCK_5_8 = "▋"
BLOCK_1_2 = "▌"
BLOCK_3_8 = "▍"
BLOCK_1_4 = "▎"
BLOCK_1_8 = "▏"
BLOCK_EMPTY = " "

# Partial blocks for smooth interpolation (8 steps)
PARTIAL_BLOCKS = [BLOCK_EMPTY, BLOCK_1_8, BLOCK_1_4, BLOCK_3_8, BLOCK_1_2, BLOCK_5_8, BLOCK_3_4, BLOCK_7_8, BLOCK_FULL]


class BlockProgressBar(Widget):
    """Custom progress bar using Unicode block characters.

    Renders like: ████████▌          45%
    """

    DEFAULT_CSS = """
    BlockProgressBar {
        height: 1;
        width: 100%;
        layout: horizontal;
    }

    BlockProgressBar #bar-container {
        width: 1fr;
        height: 1;
    }

    BlockProgressBar #bar-percent {
        width: 5;
        height: 1;
        text-align: right;
        color: $text-muted;
        margin-left: 1;
    }
    """

    progress: reactive[float] = reactive(0.0, init=False)
    bar_width: reactive[int] = reactive(40, init=False)

    def __init__(
        self,
        progress: float = 0.0,
        bar_width: int = 40,
        show_percent: bool = True,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._init_progress = progress
        self._init_bar_width = bar_width
        self._show_percent = show_percent

    def compose(self) -> ComposeResult:
        yield Static(id="bar-container")
        if self._show_percent:
            yield Static(id="bar-percent")

    def on_mount(self) -> None:
        self.progress = self._init_progress
        self.bar_width = self._init_bar_width
        self._render_bar()

    def watch_progress(self, progress: float) -> None:
        if self.is_mounted:
            self._render_bar()

    def _render_bar(self) -> None:
        if not self.is_mounted:
            return

        progress = max(0.0, min(100.0, self.progress))

        # Calculate how many full blocks and partial
        filled_width = (progress / 100.0) * self.bar_width
        full_blocks = int(filled_width)
        partial_idx = int((filled_width - full_blocks) * 8)

        # Build the bar string
        bar_fg = COLORS.accent if progress < 100 else COLORS.success
        bar_bg = COLORS.surface_hl

        # Filled portion
        filled = BLOCK_FULL * full_blocks

        # Partial block (if any)
        partial = PARTIAL_BLOCKS[partial_idx] if partial_idx > 0 and full_blocks < self.bar_width else ""

        # Empty portion
        empty_count = self.bar_width - full_blocks - (1 if partial else 0)
        empty = BLOCK_EMPTY * empty_count

        # Compose with colors
        bar_str = f"[{bar_fg}]{filled}{partial}[/][{bar_bg}]{empty}[/]"

        container = self.query_one("#bar-container", Static)
        container.update(bar_str)

        # Update percentage
        if self._show_percent:
            percent_widget = self.query_one("#bar-percent", Static)
            percent_widget.update(f"{int(progress)}%")

    def set_progress(self, value: float) -> None:
        """Set progress value (0-100)."""
        self.progress = value


class ProgressSection(Widget):
    """Displays overall extraction progress with phase info."""

    DEFAULT_CSS = """
    ProgressSection {
        height: auto;
        padding: 1 2;
        margin: 0 0 1 0;
        background: $surface;
    }

    ProgressSection #progress-header {
        height: 1;
        layout: horizontal;
        margin-bottom: 0;
    }

    ProgressSection #phase-label {
        width: auto;
        color: $accent;
        text-style: bold;
    }

    ProgressSection #phase-spacer {
        width: 1fr;
    }

    ProgressSection #phase-info {
        width: auto;
        color: $text-muted;
    }

    ProgressSection BlockProgressBar {
        margin-top: 0;
    }
    """

    progress: reactive[float] = reactive(0.0, init=False)
    phase: reactive[Phase] = reactive(Phase.INITIALIZING, init=False)

    def compose(self) -> ComposeResult:
        yield Static(id="phase-label")
        yield BlockProgressBar(progress=0.0, bar_width=50, id="progress-bar")

    def on_mount(self) -> None:
        self._update_display()

    def watch_progress(self, progress: float) -> None:
        if self.is_mounted:
            bar = self.query_one(BlockProgressBar)
            bar.progress = progress

    def watch_phase(self, phase: Phase) -> None:
        if self.is_mounted:
            self._update_display()

    def _update_display(self) -> None:
        if not self.is_mounted:
            return

        phase_label = self.query_one("#phase-label", Static)
        phase_label.update(f"⟩ {self.phase.value.upper()}")

    def set_progress(
        self,
        progress: float,
        phase: Phase,
        outer_iter: int,
        max_outer: int,
        inner_turn: int,
        max_inner: int,
        current_layer: int | None = None,
    ) -> None:
        """Update progress display."""
        self.progress = progress
        self.phase = phase
