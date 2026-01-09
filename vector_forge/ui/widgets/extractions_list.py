"""Extractions list widget for parallel view with block-character bars."""

from typing import Dict, Optional

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from vector_forge.ui.theme import COLORS, ICONS
from vector_forge.ui.state import ExtractionUIState, ExtractionStatus


# Block characters for progress
BLOCK_FULL = "█"
BLOCK_EMPTY = "░"


class ExtractionItem(Widget):
    """Single extraction item in the list."""

    DEFAULT_CSS = """
    ExtractionItem {
        height: 2;
        padding: 0 2;
        background: $surface;
        layout: grid;
        grid-size: 4 2;
        grid-columns: 1fr auto 12 6;
        grid-rows: 1 1;
    }

    ExtractionItem:hover {
        background: $panel;
    }

    ExtractionItem.-selected {
        background: $panel;
        border-left: wide $accent;
    }

    ExtractionItem .extraction-name {
        color: $text;
        text-style: bold;
    }

    ExtractionItem .extraction-status-icon {
        width: 2;
        text-align: right;
    }

    ExtractionItem .extraction-progress-container {
        width: 12;
    }

    ExtractionItem .extraction-time {
        width: 6;
        text-align: right;
        color: $text-muted;
    }

    ExtractionItem .extraction-model {
        color: $text-muted;
        column-span: 2;
    }

    ExtractionItem .extraction-phase {
        color: $text-muted;
        column-span: 2;
        text-align: right;
    }
    """

    class Selected(Message):
        """Message sent when this item is selected."""

        def __init__(self, extraction_id: str) -> None:
            super().__init__()
            self.extraction_id = extraction_id

    selected: reactive[bool] = reactive(False)

    def __init__(self, extraction: ExtractionUIState) -> None:
        super().__init__()
        self.extraction = extraction
        self.extraction_id = extraction.id

    def compose(self) -> ComposeResult:
        yield Static(self.extraction.behavior_name, classes="extraction-name")
        yield Static(classes="extraction-status-icon")
        yield Static(classes="extraction-progress-container")
        yield Static(self.extraction.elapsed_str, classes="extraction-time")
        yield Static(self.extraction.model, classes="extraction-model")
        yield Static(self.extraction.phase.value, classes="extraction-phase")

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
        self._update_display()

    def _update_display(self) -> None:
        # Status icon with appropriate color
        status_widget = self.query_one(".extraction-status-icon", Static)
        status_colors = {
            ExtractionStatus.PENDING: COLORS.text_muted,
            ExtractionStatus.RUNNING: COLORS.accent,
            ExtractionStatus.PAUSED: COLORS.warning,
            ExtractionStatus.COMPLETE: COLORS.success,
            ExtractionStatus.FAILED: COLORS.error,
        }
        status_icons = {
            ExtractionStatus.PENDING: ICONS.pending,
            ExtractionStatus.RUNNING: ICONS.running,
            ExtractionStatus.PAUSED: ICONS.paused,
            ExtractionStatus.COMPLETE: ICONS.complete,
            ExtractionStatus.FAILED: ICONS.failed,
        }
        color = status_colors.get(self.extraction.status, COLORS.text_muted)
        icon = status_icons.get(self.extraction.status, ICONS.pending)
        status_widget.update(f"[{color}]{icon}[/]")

        # Progress bar using block characters
        progress_container = self.query_one(".extraction-progress-container", Static)
        bar_width = 8
        progress = self.extraction.progress / 100.0
        filled = int(progress * bar_width)
        empty = bar_width - filled

        bar_color = COLORS.success if self.extraction.status == ExtractionStatus.COMPLETE else COLORS.accent
        bar_str = (
            f"[{bar_color}]{BLOCK_FULL * filled}[/]"
            f"[{COLORS.surface_hl}]{BLOCK_EMPTY * empty}[/]"
            f" {int(self.extraction.progress):2d}%"
        )
        progress_container.update(bar_str)

        # Time
        time_widget = self.query_one(".extraction-time", Static)
        time_widget.update(self.extraction.elapsed_str)

        # Phase
        phase_widget = self.query_one(".extraction-phase", Static)
        phase_widget.update(self.extraction.phase.value)


class ExtractionsList(Widget):
    """Scrollable list of all extractions for parallel view."""

    DEFAULT_CSS = """
    ExtractionsList {
        height: 1fr;
        min-height: 8;
        background: $surface;
        padding: 1 0;
        margin: 0 0 1 0;
    }

    ExtractionsList #list-title {
        color: $accent;
        text-style: bold;
        height: 1;
        padding: 0 2;
        margin-bottom: 1;
    }

    ExtractionsList #list-content {
        height: 1fr;
        scrollbar-size: 1 1;
    }

    ExtractionsList #list-empty {
        color: $text-disabled;
        padding: 0 2;
    }
    """

    extractions: reactive[Dict[str, ExtractionUIState]] = reactive(
        dict, always_update=True, init=False
    )
    selected_id: reactive[Optional[str]] = reactive(None, init=False)

    class SelectionChanged(Message):
        """Message sent when selection changes."""

        def __init__(self, extraction_id: Optional[str]) -> None:
            super().__init__()
            self.extraction_id = extraction_id

    def compose(self) -> ComposeResult:
        yield Static("Running Extractions", id="list-title")
        yield VerticalScroll(id="list-content")

    def on_mount(self) -> None:
        self._refresh_list()

    def watch_extractions(self, extractions: Dict[str, ExtractionUIState]) -> None:
        if self.is_mounted:
            self._refresh_list()

    def watch_selected_id(self, selected_id: Optional[str]) -> None:
        if not self.is_mounted:
            return
        for item in self.query(ExtractionItem):
            item.selected = item.extraction_id == selected_id

    def on_extraction_item_selected(self, message: ExtractionItem.Selected) -> None:
        self.selected_id = message.extraction_id
        self.post_message(self.SelectionChanged(message.extraction_id))
        message.stop()

    def _refresh_list(self) -> None:
        """Refresh the list - update existing items or create new ones."""
        content = self.query_one("#list-content", VerticalScroll)

        if not self.extractions:
            # Clear and show empty message
            for child in list(content.children):
                child.remove()
            content.mount(Static("No extractions running"))
            return

        sorted_extractions = sorted(
            self.extractions.values(),
            key=lambda e: (
                0 if e.status == ExtractionStatus.RUNNING else 1,
                e.started_at or 0,
            ),
        )

        # Check if we need to rebuild
        existing_items = list(self.query(ExtractionItem))
        existing_ids = [item.extraction_id for item in existing_items]
        new_ids = [e.id for e in sorted_extractions]

        # If the list structure changed, rebuild entirely
        if existing_ids != new_ids:
            for child in list(content.children):
                child.remove()
            for extraction in sorted_extractions:
                item = ExtractionItem(extraction)
                item.selected = extraction.id == self.selected_id
                content.mount(item)
        else:
            # Just update existing items
            for item, extraction in zip(existing_items, sorted_extractions):
                item.update_extraction(extraction)
                item.selected = extraction.id == self.selected_id

    def select_extraction(self, extraction_id: str) -> None:
        if extraction_id in self.extractions:
            self.selected_id = extraction_id
            self.post_message(self.SelectionChanged(extraction_id))

    def select_next(self) -> None:
        if not self.extractions:
            return

        ids = list(self.extractions.keys())
        if self.selected_id is None:
            self.select_extraction(ids[0])
        else:
            try:
                idx = ids.index(self.selected_id)
                next_idx = (idx + 1) % len(ids)
                self.select_extraction(ids[next_idx])
            except ValueError:
                self.select_extraction(ids[0])

    def select_previous(self) -> None:
        if not self.extractions:
            return

        ids = list(self.extractions.keys())
        if self.selected_id is None:
            self.select_extraction(ids[-1])
        else:
            try:
                idx = ids.index(self.selected_id)
                prev_idx = (idx - 1) % len(ids)
                self.select_extraction(ids[prev_idx])
            except ValueError:
                self.select_extraction(ids[-1])
