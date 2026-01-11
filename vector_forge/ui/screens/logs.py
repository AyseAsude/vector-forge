"""Logs screen - modern log viewer with split-panel filtering."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen, ModalScreen
from textual.widgets import Static, Input

from vector_forge.ui.state import LogEntry, get_state
from vector_forge.ui.widgets.tmux_bar import TmuxBar
from vector_forge.ui.widgets.log_detail import get_renderer_registry
from vector_forge.ui.messages import LogEmitted, TimeTick


class LogDetailModal(ModalScreen):
    """Modal showing full log entry details with event-type-specific rendering.

    Uses Strategy pattern via LogDetailRendererRegistry to provide rich,
    context-aware detail views for different event types (LLM requests,
    contrast pairs, evaluations, etc.).
    """

    BINDINGS = [
        Binding("escape", "dismiss", "close"),
        Binding("q", "dismiss", "close"),
        Binding("j", "scroll_down", "scroll down", show=False),
        Binding("k", "scroll_up", "scroll up", show=False),
        Binding("g", "scroll_top", "top", show=False),
        Binding("G", "scroll_bottom", "bottom", show=False),
    ]

    DEFAULT_CSS = """
    LogDetailModal {
        align: center middle;
    }

    LogDetailModal #modal-container {
        width: 90%;
        max-width: 120;
        height: 85%;
        background: $surface;
        border: solid $primary;
    }

    LogDetailModal #modal-header {
        height: auto;
        padding: 1 2;
        background: $surface;
        border-bottom: solid $surface-lighten-1;
    }

    LogDetailModal .title {
        height: 1;
        text-style: bold;
    }

    LogDetailModal .meta {
        height: auto;
        color: $foreground-muted;
    }

    LogDetailModal .event-type {
        height: 1;
        color: $accent;
        margin-top: 1;
    }

    LogDetailModal #modal-content {
        height: 1fr;
        padding: 1 0 1 2;
        background: $background;
        scrollbar-gutter: stable;
    }

    LogDetailModal .section-header {
        height: 1;
        color: $accent;
        text-style: bold;
        margin-top: 1;
        margin-bottom: 0;
    }

    LogDetailModal .section-content {
        height: auto;
        padding: 1;
        margin-bottom: 1;
        background: $surface;
    }

    LogDetailModal #modal-footer {
        height: 2;
        padding: 0 2;
        background: $surface;
        border-top: solid $surface-lighten-1;
    }

    LogDetailModal .footer-text {
        height: 1;
        color: $foreground-muted;
        text-align: center;
        margin-top: 0;
    }
    """

    def __init__(self, entry: LogEntry, **kwargs) -> None:
        super().__init__(**kwargs)
        self.entry = entry
        self._registry = get_renderer_registry()

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-container"):
            # Header with metadata
            with Vertical(id="modal-header"):
                yield Static(classes="title", id="modal-title")
                yield Static(classes="meta", id="modal-meta")
                yield Static(classes="event-type", id="modal-event-type")

            # Scrollable content area
            yield VerticalScroll(id="modal-content")

            # Footer
            with Vertical(id="modal-footer"):
                yield Static("ESC/q: close  |  j/k: scroll  |  g/G: top/bottom", classes="footer-text")

    def on_mount(self) -> None:
        entry = self.entry

        # Level-based title color
        level_colors = {
            "info": "$primary",
            "warning": "$warning",
            "error": "$error",
        }
        color = level_colors.get(entry.level, "$foreground-muted")

        # Title
        self.query_one("#modal-title", Static).update(
            f"[{color}]{entry.level.upper()}[/] Log Entry"
        )

        # Meta line
        meta_parts = [f"Time: {entry.time_str}", f"Source: {entry.source}"]
        if entry.extraction_id:
            meta_parts.append(f"Task: {entry.extraction_id}")
        if entry.agent_id:
            meta_parts.append(f"Agent: {entry.agent_id}")
        self.query_one("#modal-meta", Static).update("  ·  ".join(meta_parts))

        # Event type (if available)
        event_type_widget = self.query_one("#modal-event-type", Static)
        if entry.event_type:
            event_type_widget.update(f"Event: {entry.event_type}")
        else:
            event_type_widget.display = False

        # Render sections using the appropriate renderer
        content_area = self.query_one("#modal-content", VerticalScroll)
        sections = self._registry.render(entry)

        for title, content in sections:
            if title:
                content_area.mount(Static(title, classes="section-header"))
            content_area.mount(Static(content, classes="section-content", markup=True))

    def action_scroll_down(self) -> None:
        self.query_one("#modal-content", VerticalScroll).scroll_down()

    def action_scroll_up(self) -> None:
        self.query_one("#modal-content", VerticalScroll).scroll_up()

    def action_scroll_top(self) -> None:
        self.query_one("#modal-content", VerticalScroll).scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        self.query_one("#modal-content", VerticalScroll).scroll_end(animate=False)


class LevelFilterRow(Static):
    """Clickable level filter row."""

    DEFAULT_CSS = """
    LevelFilterRow {
        height: 1;
        padding: 0 1;
    }

    LevelFilterRow:hover {
        background: $boost;
    }
    """

    class Toggled(Message):
        def __init__(self, level: str, active: bool) -> None:
            super().__init__()
            self.level = level
            self.active = active

    def __init__(self, level: str, count: int = 0, active: bool = True, **kwargs) -> None:
        self._level = level
        self._count = count
        self._active = active
        # Compute initial content to pass to super().__init__()
        initial_content = self._compute_content()
        super().__init__(initial_content, **kwargs)

    def on_click(self) -> None:
        self._active = not self._active
        self.update(self._compute_content())
        self.post_message(self.Toggled(self._level, self._active))

    def set_count(self, count: int) -> None:
        self._count = count
        self.update(self._compute_content())

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def level(self) -> str:
        return self._level

    def _compute_content(self) -> str:
        """Compute the display content for this filter row."""
        level_colors = {
            "info": "$primary",
            "warning": "$warning",
            "error": "$error",
        }
        color = level_colors.get(self._level, "$foreground-muted")

        if self._active:
            return (
                f"[{color}]●[/] {self._level.upper():<8} "
                f"[$foreground-muted]{self._count:>4}[/]"
            )
        else:
            return f"[$foreground-disabled]○ {self._level.upper():<8} {self._count:>4}[/]"


class SourceFilterRow(Static):
    """Clickable source filter row."""

    DEFAULT_CSS = """
    SourceFilterRow {
        height: 1;
        padding: 0 1;
    }

    SourceFilterRow:hover {
        background: $boost;
    }
    """

    class Selected(Message):
        def __init__(self, source: str) -> None:
            super().__init__()
            self.source = source

    def __init__(self, source: str, label: str, selected: bool = False, **kwargs) -> None:
        self._source = source
        self._label = label
        self._selected = selected
        # Compute initial content to pass to super().__init__()
        initial_content = self._compute_content()
        super().__init__(initial_content, **kwargs)

    def on_click(self) -> None:
        self.post_message(self.Selected(self._source))

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.update(self._compute_content())

    @property
    def source(self) -> str:
        return self._source

    def _compute_content(self) -> str:
        """Compute the display content for this filter row."""
        if self._selected:
            return f"[$accent]●[/] {self._label}"
        else:
            return f"[$foreground-disabled]○[/] [$foreground-muted]{self._label}[/]"


class FilterPanel(Vertical):
    """Left panel with filter controls."""

    DEFAULT_CSS = """
    FilterPanel {
        width: 26;
        padding: 1 2;
        background: $surface;
    }

    FilterPanel .header {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
    }

    FilterPanel .section {
        height: 1;
        color: $foreground-muted;
        margin-top: 1;
    }

    FilterPanel .level-list {
        height: auto;
    }

    FilterPanel .source-list {
        height: auto;
        max-height: 10;
    }

    FilterPanel .search-section {
        height: auto;
        margin-top: 2;
    }

    FilterPanel #search-input {
        width: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("FILTERS", classes="header")

        yield Static("LEVEL", classes="section")
        with Vertical(classes="level-list"):
            yield LevelFilterRow("info", active=True, id="level-info")
            yield LevelFilterRow("warning", active=True, id="level-warning")
            yield LevelFilterRow("error", active=True, id="level-error")

        yield Static("SOURCE", classes="section")
        yield VerticalScroll(classes="source-list", id="source-list")

        with Vertical(classes="search-section"):
            yield Static("SEARCH", classes="section")
            yield Input(placeholder="filter...", id="search-input")

    def update_level_counts(self, info: int, warning: int, error: int) -> None:
        self.query_one("#level-info", LevelFilterRow).set_count(info)
        self.query_one("#level-warning", LevelFilterRow).set_count(warning)
        self.query_one("#level-error", LevelFilterRow).set_count(error)

    def update_sources(self, sources: list[str], selected: str) -> None:
        source_list = self.query_one("#source-list", VerticalScroll)

        # Get existing rows
        existing = {row.source: row for row in source_list.query(SourceFilterRow)}

        # Build new source list with "All" first
        all_sources = [""] + sources

        # Update or create rows
        for source in all_sources:
            label = "All" if source == "" else source
            if source in existing:
                existing[source].set_selected(source == selected)
            else:
                row = SourceFilterRow(source, label, selected=source == selected)
                source_list.mount(row)

        # Remove stale rows
        for source, row in existing.items():
            if source not in all_sources:
                row.remove()

    def get_active_levels(self) -> set[str]:
        active = set()
        for level in ["info", "warning", "error"]:
            row = self.query_one(f"#level-{level}", LevelFilterRow)
            if row.is_active:
                active.add(level)
        return active


class LogRow(Static):
    """Single log entry row."""

    DEFAULT_CSS = """
    LogRow {
        height: 1;
        margin-right: 2;
    }

    LogRow:hover {
        background: $boost;
    }
    """

    class Clicked(Message):
        def __init__(self, entry: LogEntry) -> None:
            super().__init__()
            self.entry = entry

    def __init__(self, entry: LogEntry, **kwargs) -> None:
        self.entry = entry
        # Compute initial content to pass to super().__init__()
        initial_content = self._compute_content()
        super().__init__(initial_content, **kwargs)

    def _compute_content(self) -> str:
        """Compute the display content for this log row."""
        entry = self.entry

        level_colors = {
            "info": "$primary",
            "warning": "$warning",
            "error": "$error",
        }
        color = level_colors.get(entry.level, "$foreground-muted")

        # Truncate message for list view (increased from 60 to 100)
        # Full content is available in the detail modal
        message = entry.message.replace("\n", " ")  # Collapse newlines
        max_msg_len = 100
        if len(message) > max_msg_len:
            message = message[:max_msg_len - 3] + "..."

        # Add indicator if entry has rich detail data
        detail_indicator = "[$accent]»[/] " if entry.has_detail else ""

        return (
            f"[$foreground-disabled]{entry.time_str}[/]  "
            f"[{color}]●[/]  "
            f"[$foreground-muted]{entry.source:<10}[/]  "
            f"{detail_indicator}{message}"
        )

    def on_click(self) -> None:
        self.post_message(self.Clicked(self.entry))


class LogPanel(Vertical):
    """Right panel showing log stream.

    Uses incremental updates for performance - only appends new entries
    when possible, avoiding full rebuilds.
    """

    DEFAULT_CSS = """
    LogPanel {
        width: 1fr;
        background: $surface;
    }

    LogPanel .panel-header {
        height: auto;
        padding: 1 2;
    }

    LogPanel .title-row {
        height: 1;
        margin-bottom: 1;
    }

    LogPanel .title {
        width: 1fr;
        text-style: bold;
    }

    LogPanel .count {
        width: auto;
        color: $foreground-muted;
    }

    LogPanel .log-stream {
        height: 1fr;
        padding: 1 0 1 2;
        background: $background;
        scrollbar-gutter: stable;
    }

    LogPanel .empty {
        height: 1fr;
        content-align: center middle;
        color: $foreground-muted;
        margin-right: 2;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._displayed_timestamps: set[float] = set()
        self._last_filter_hash: str = ""

    def compose(self) -> ComposeResult:
        with Vertical(classes="panel-header"):
            with Horizontal(classes="title-row"):
                yield Static("LOG STREAM", classes="title")
                yield Static(classes="count", id="log-count")
        yield VerticalScroll(classes="log-stream", id="log-stream")

    def set_logs(
        self,
        logs: list[LogEntry],
        total: int,
        force: bool = False,
        filter_key: str = "",
    ) -> None:
        """Update the log stream efficiently.

        Uses incremental updates when possible - only appends new entries
        if the filter hasn't changed. Falls back to full rebuild on filter change.

        Args:
            logs: Filtered log entries to display
            total: Total number of logs (before filtering)
            force: Force full rebuild
            filter_key: String representing current filter state (for change detection)
        """
        count = self.query_one("#log-count", Static)
        count.update(f"{len(logs)} / {total}")

        stream = self.query_one("#log-stream", VerticalScroll)

        # Check if filter changed (not just log count)
        filter_changed = filter_key != self._last_filter_hash
        if force or filter_changed:
            # Full rebuild needed - use batch_update for less flicker
            with self.app.batch_update():
                stream.remove_children()
                self._displayed_timestamps.clear()

                if not logs:
                    stream.mount(Static("No log entries", classes="empty"))
                else:
                    # Limit to last 500 for performance
                    for entry in logs[-500:]:
                        stream.mount(LogRow(entry))
                        self._displayed_timestamps.add(entry.timestamp)

            self._last_filter_hash = filter_key
            stream.scroll_end(animate=False)
            return

        # Incremental update - only add new entries
        # Remove empty placeholder if present
        empties = list(stream.query(".empty"))
        if empties and logs:
            for e in empties:
                e.remove()

        # Find and add new entries
        new_entries = [
            entry for entry in logs[-500:]
            if entry.timestamp not in self._displayed_timestamps
        ]

        if new_entries:
            for entry in new_entries:
                stream.mount(LogRow(entry))
                self._displayed_timestamps.add(entry.timestamp)
            stream.scroll_end(animate=False)

        # Handle case where logs were cleared
        if not logs and not empties:
            stream.remove_children()
            stream.mount(Static("No log entries", classes="empty"))
            self._displayed_timestamps.clear()


class LogsScreen(Screen):
    """Full-screen log viewer with split-panel filtering."""

    BINDINGS = [
        # Navigation between screens
        Binding("1", "go_dashboard", "Dashboard", key_display="1"),
        Binding("2", "go_samples", "Samples", key_display="2"),
        Binding("3", "noop", "Logs", show=False),  # Current screen
        Binding("tab", "cycle", "Next Screen"),
        # Log navigation
        Binding("/", "focus_search", "Search"),
        Binding("g", "scroll_top", "Top", show=False),
        Binding("G", "scroll_bottom", "Bottom", show=False),
        Binding("escape", "clear_focus", "Clear", show=False),
        # Actions
        Binding("n", "new_task", "New Task"),
        Binding("q", "quit", "Quit"),
    ]

    DEFAULT_CSS = """
    LogsScreen {
        background: $background;
    }

    LogsScreen #content {
        height: 1fr;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._active_levels: set[str] = {"info", "warning", "error"}
        self._selected_source: str = ""
        self._search_text: str = ""

    def compose(self) -> ComposeResult:
        with Horizontal(id="content"):
            yield FilterPanel(id="filter-panel")
            yield LogPanel(id="log-panel")
        yield TmuxBar(active_screen="logs")

    def on_mount(self) -> None:
        """Initial projection from current state."""
        get_state().add_listener(self._on_state_change)
        self._sync()

    def on_unmount(self) -> None:
        """Clean up state listener."""
        get_state().remove_listener(self._on_state_change)

    def _on_state_change(self, _) -> None:
        """Handle state changes - update logs.

        Uses call_later to ensure sync runs on the main thread.
        Events may be emitted from background threads (e.g., extraction
        running in executor), and Textual widget updates must happen
        on the main thread.
        """
        self.call_later(self._sync)

    # ─────────────────────────────────────────────────────────────────
    # Event Handlers - Targeted Updates
    # ─────────────────────────────────────────────────────────────────

    def on_log_emitted(self, event: LogEmitted) -> None:
        """Handle new log entry - incrementally update."""
        self._sync()

    def on_time_tick(self, event: TimeTick) -> None:
        """Handle time tick."""
        self.query_one(TmuxBar).refresh_info()

    def _get_unique_sources(self) -> list[str]:
        state = get_state()
        sources = set()
        for log in state.logs:
            sources.add(log.source)
        return sorted(sources)

    def _get_level_counts(self) -> tuple[int, int, int]:
        state = get_state()
        logs = state.get_filtered_logs(extraction_id=state.selected_id)

        info_count = sum(1 for log in logs if log.level == "info")
        warning_count = sum(1 for log in logs if log.level == "warning")
        error_count = sum(1 for log in logs if log.level == "error")

        return info_count, warning_count, error_count

    def _get_filtered_logs(self) -> list[LogEntry]:
        state = get_state()
        logs = state.get_filtered_logs(extraction_id=state.selected_id)

        # Filter by level
        if self._active_levels != {"info", "warning", "error"}:
            logs = [log for log in logs if log.level in self._active_levels]

        # Filter by source
        if self._selected_source:
            logs = [log for log in logs if log.source == self._selected_source]

        # Filter by search text
        if self._search_text:
            search_lower = self._search_text.lower()
            logs = [
                log for log in logs
                if search_lower in log.message.lower()
                or search_lower in log.source.lower()
            ]

        return logs

    def _sync(self, force_logs: bool = False) -> None:
        state = get_state()
        filtered = self._get_filtered_logs()
        total = len(state.logs)

        # Update filter panel
        filter_panel = self.query_one("#filter-panel", FilterPanel)
        info_count, warning_count, error_count = self._get_level_counts()
        filter_panel.update_level_counts(info_count, warning_count, error_count)
        filter_panel.update_sources(self._get_unique_sources(), self._selected_source)

        # Build filter key from current filter state (not counts)
        filter_key = f"{sorted(self._active_levels)}|{self._selected_source}|{self._search_text}"

        # Update log panel
        self.query_one("#log-panel", LogPanel).set_logs(
            filtered, total, force=force_logs, filter_key=filter_key
        )
        self.query_one(TmuxBar).refresh_info()

    # Event handlers
    def on_level_filter_row_toggled(self, event: LevelFilterRow.Toggled) -> None:
        if event.active:
            self._active_levels.add(event.level)
        else:
            self._active_levels.discard(event.level)
        self._sync(force_logs=True)

    def on_source_filter_row_selected(self, event: SourceFilterRow.Selected) -> None:
        self._selected_source = event.source
        self._sync(force_logs=True)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "search-input":
            self._search_text = event.value
            self._sync(force_logs=True)

    def on_log_row_clicked(self, event: LogRow.Clicked) -> None:
        self.app.push_screen(LogDetailModal(event.entry))

    # Actions
    def action_noop(self) -> None:
        pass

    def action_go_dashboard(self) -> None:
        self.app.switch_screen("dashboard")

    def action_go_samples(self) -> None:
        self.app.switch_screen("samples")

    def action_cycle(self) -> None:
        self.app.switch_screen("dashboard")

    def action_focus_search(self) -> None:
        self.query_one("#search-input", Input).focus()

    def action_clear_focus(self) -> None:
        self.focus()

    def action_scroll_top(self) -> None:
        self.query_one("#log-stream", VerticalScroll).scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        self.query_one("#log-stream", VerticalScroll).scroll_end(animate=False)

    def action_new_task(self) -> None:
        self.app.push_screen("create_task")

    def action_quit(self) -> None:
        self.app.exit()
