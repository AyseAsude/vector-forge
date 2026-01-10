"""Extraction selector widget - dropdown for switching between extractions."""

from typing import Dict, Optional

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from vector_forge.ui.theme import ICONS
from vector_forge.ui.state import ExtractionUIState, ExtractionStatus, Phase


# Block characters for inline progress
BLOCK_FULL = "█"
BLOCK_EMPTY = "░"


class ExtractionOption(Widget):
    """Single extraction option in the dropdown."""

    DEFAULT_CSS = """
    ExtractionOption {
        height: 2;
        padding: 0 1;
        background: transparent;
    }

    ExtractionOption:hover {
        background: $surface;
    }

    ExtractionOption.-selected {
        background: $surface;
    }

    ExtractionOption .opt-row {
        height: 1;
    }
    """

    class Selected(Message):
        """Message sent when this option is selected."""

        def __init__(self, extraction_id: str) -> None:
            super().__init__()
            self.extraction_id = extraction_id

    selected: reactive[bool] = reactive(False)

    def __init__(self, extraction: ExtractionUIState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.extraction = extraction
        self.extraction_id = extraction.id

    def compose(self) -> ComposeResult:
        yield Static(id="opt-line1", classes="opt-row")
        yield Static(id="opt-line2", classes="opt-row")

    def on_mount(self) -> None:
        self._update_display()

    def on_click(self) -> None:
        self.post_message(self.Selected(self.extraction_id))

    def watch_selected(self, selected: bool) -> None:
        if selected:
            self.add_class("-selected")
        else:
            self.remove_class("-selected")

    def update_extraction(self, extraction: ExtractionUIState) -> None:
        self.extraction = extraction
        if self.is_mounted:
            self._update_display()

    def _update_display(self) -> None:
        ext = self.extraction

        # Status icon and color
        status_icons = {
            ExtractionStatus.PENDING: (ICONS.pending, "$foreground-disabled"),
            ExtractionStatus.RUNNING: (ICONS.running, "$accent"),
            ExtractionStatus.PAUSED: (ICONS.paused, "$warning"),
            ExtractionStatus.COMPLETE: (ICONS.complete, "$success"),
            ExtractionStatus.FAILED: (ICONS.failed, "$error"),
        }
        icon, color = status_icons.get(ext.status, (ICONS.pending, "$foreground-disabled"))

        # Progress bar (compact)
        bar_width = 6
        progress = ext.progress / 100.0
        filled = int(progress * bar_width)
        empty = bar_width - filled
        bar_color = "$success" if ext.status == ExtractionStatus.COMPLETE else "$accent"
        progress_bar = (
            f"[{bar_color}]{BLOCK_FULL * filled}[/]"
            f"[$boost]{BLOCK_EMPTY * empty}[/]"
        )

        # Line 1: icon, name, progress, percentage
        line1 = self.query_one("#opt-line1", Static)
        line1.update(
            f"[{color}]{icon}[/] [$foreground]{ext.behavior_name}[/]  "
            f"{progress_bar} [$foreground-muted]{int(ext.progress)}%[/]"
        )

        # Line 2: description (truncated)
        desc = ext.behavior_description
        if len(desc) > 50:
            desc = desc[:47] + "..."
        line2 = self.query_one("#opt-line2", Static)
        line2.update(f"  [$foreground-disabled]{desc}[/]")


class ExtractionSelector(Widget):
    """Dropdown selector for choosing active extraction.

    Shows current extraction at top. Click to expand dropdown with all extractions.
    """

    DEFAULT_CSS = """
    ExtractionSelector {
        height: auto;
        background: $panel;
    }

    ExtractionSelector #selector-header {
        height: 3;
        padding: 0 1;
        background: $panel;
    }

    ExtractionSelector #selector-header:hover {
        background: $surface;
    }

    ExtractionSelector .header-row {
        height: 1;
    }

    ExtractionSelector #dropdown-container {
        height: auto;
        max-height: 16;
        background: $panel;
        border-top: solid $surface;
        display: none;
        overflow-y: auto;
    }

    ExtractionSelector #dropdown-container.expanded {
        display: block;
    }

    ExtractionSelector #dropdown-empty {
        height: 2;
        padding: 0 2;
        color: $foreground-muted;
    }

    ExtractionSelector #chevron {
        width: 2;
        text-align: right;
    }
    """

    class ExtractionChanged(Message):
        """Message sent when extraction selection changes."""

        def __init__(self, extraction_id: str) -> None:
            super().__init__()
            self.extraction_id = extraction_id

    extractions: reactive[Dict[str, ExtractionUIState]] = reactive(
        dict, always_update=True, init=False
    )
    selected_id: reactive[Optional[str]] = reactive(None, init=False)
    expanded: reactive[bool] = reactive(False)

    def compose(self) -> ComposeResult:
        with Vertical(id="selector-header"):
            yield Static(id="header-line1", classes="header-row")
            yield Static(id="header-line2", classes="header-row")
            yield Static(id="header-line3", classes="header-row")
        yield Vertical(id="dropdown-container")

    def on_mount(self) -> None:
        self._update_header()

    def watch_extractions(self, extractions: Dict[str, ExtractionUIState]) -> None:
        if self.is_mounted:
            self._update_header()
            if self.expanded:
                self._refresh_dropdown()

    def watch_selected_id(self, selected_id: Optional[str]) -> None:
        if self.is_mounted:
            self._update_header()

    def watch_expanded(self, expanded: bool) -> None:
        if not self.is_mounted:
            return
        dropdown = self.query_one("#dropdown-container", Vertical)
        if expanded:
            dropdown.add_class("expanded")
            self._refresh_dropdown()
        else:
            dropdown.remove_class("expanded")

    def _update_header(self) -> None:
        """Update the header display with current extraction."""
        ext = self.extractions.get(self.selected_id) if self.selected_id else None

        line1 = self.query_one("#header-line1", Static)
        line2 = self.query_one("#header-line2", Static)
        line3 = self.query_one("#header-line3", Static)

        if ext is None:
            line1.update("[$foreground-muted]No extraction selected[/]")
            line2.update("[$foreground-disabled]Start an extraction to begin[/]")
            line3.update("")
            return

        # Status icon and color
        status_map = {
            ExtractionStatus.PENDING: (ICONS.pending, "$foreground-disabled", "pending"),
            ExtractionStatus.RUNNING: (ICONS.running, "$accent", "running"),
            ExtractionStatus.PAUSED: (ICONS.paused, "$warning", "paused"),
            ExtractionStatus.COMPLETE: (ICONS.complete, "$success", "complete"),
            ExtractionStatus.FAILED: (ICONS.failed, "$error", "failed"),
        }
        icon, color, status_text = status_map.get(
            ext.status, (ICONS.pending, "$foreground-disabled", "unknown")
        )

        # Chevron for dropdown
        chevron = "▾" if not self.expanded else "▴"

        # Line 1: Name and status
        line1.update(
            f"[{color}]{icon}[/] [$foreground bold]{ext.behavior_name}[/] "
            f"[$foreground-muted]· {status_text}[/] "
            f"[$foreground-muted]{chevron}[/]"
        )

        # Line 2: Description
        desc = ext.behavior_description
        if len(desc) > 60:
            desc = desc[:57] + "..."
        line2.update(f"  [$foreground-disabled]{desc}[/]")

        # Line 3: Progress info
        phase_text = ext.phase.value
        progress_text = f"{int(ext.progress)}%"
        iter_text = f"iter {ext.outer_iteration}/{ext.max_outer_iterations}"
        line3.update(
            f"  [$foreground-muted]{phase_text}[/] "
            f"[$accent]{progress_text}[/] "
            f"[$foreground-disabled]· {iter_text} · {ext.elapsed_str}[/]"
        )

    def _refresh_dropdown(self) -> None:
        """Refresh the dropdown options."""
        dropdown = self.query_one("#dropdown-container", Vertical)

        # Clear existing children
        for child in list(dropdown.children):
            child.remove()

        if not self.extractions:
            dropdown.mount(Static("No extractions", id="dropdown-empty"))
            return

        # Add options (current extraction first, then others)
        sorted_extractions = sorted(
            self.extractions.values(),
            key=lambda e: (
                0 if e.id == self.selected_id else 1,
                0 if e.status == ExtractionStatus.RUNNING else 1,
                e.started_at or 0,
            ),
        )

        for ext in sorted_extractions:
            if ext.id == self.selected_id:
                continue  # Skip current, it's shown in header
            opt = ExtractionOption(ext)
            dropdown.mount(opt)

    def on_click(self, event) -> None:
        """Handle clicks - toggle dropdown if clicking header."""
        # Check if click was on header
        header = self.query_one("#selector-header", Vertical)
        if header in event.widget.ancestors_with_self:
            self.expanded = not self.expanded
            event.stop()

    def on_extraction_option_selected(self, message: ExtractionOption.Selected) -> None:
        """Handle option selection."""
        if message.extraction_id != self.selected_id:
            self.selected_id = message.extraction_id
            self.post_message(self.ExtractionChanged(message.extraction_id))
        self.expanded = False
        message.stop()

    def set_extraction(self, extraction_id: str) -> None:
        """Set the selected extraction."""
        if extraction_id in self.extractions:
            self.selected_id = extraction_id
