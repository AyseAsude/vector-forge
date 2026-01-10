"""Logs screen - modern log viewer with split-panel filtering."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import Screen, ModalScreen
from textual.widgets import Static, Input

from vector_forge.ui.state import LogEntry, get_state
from vector_forge.ui.widgets.tmux_bar import TmuxBar


class LogDetailModal(ModalScreen):
    """Modal showing full log entry details."""

    BINDINGS = [
        Binding("escape", "dismiss", "close"),
        Binding("q", "dismiss", "close"),
    ]

    DEFAULT_CSS = """
    LogDetailModal {
        align: center middle;
    }

    LogDetailModal #modal {
        width: 80%;
        max-width: 100;
        height: auto;
        max-height: 80%;
        background: $surface;
        padding: 1 2;
    }

    LogDetailModal .title {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
    }

    LogDetailModal .meta {
        height: auto;
        color: $foreground-muted;
        margin-bottom: 1;
    }

    LogDetailModal .section {
        height: 1;
        color: $accent;
        text-style: bold;
        margin-top: 1;
    }

    LogDetailModal .content {
        height: auto;
        max-height: 20;
        padding: 1;
        background: $background;
    }

    LogDetailModal .footer {
        height: 1;
        color: $foreground-muted;
        text-align: center;
        margin-top: 1;
    }
    """

    def __init__(self, entry: LogEntry, **kwargs) -> None:
        super().__init__(**kwargs)
        self.entry = entry

    def compose(self) -> ComposeResult:
        with Vertical(id="modal"):
            yield Static(classes="title", id="modal-title")
            yield Static(classes="meta", id="modal-meta-time")
            yield Static(classes="meta", id="modal-meta-extraction")
            yield Static(classes="meta", id="modal-meta-agent")

            yield Static("MESSAGE", classes="section")
            yield Static(classes="content", id="modal-content")

            yield Static("Press ESC to close", classes="footer")

    def on_mount(self) -> None:
        entry = self.entry

        level_colors = {
            "info": "$primary",
            "warning": "$warning",
            "error": "$error",
        }
        color = level_colors.get(entry.level, "$foreground-muted")

        self.query_one("#modal-title", Static).update(f"[{color}]{entry.level.upper()}[/] Log Entry")
        self.query_one("#modal-meta-time", Static).update(
            f"Time: {entry.time_str}  ·  Source: {entry.source}  ·  Level: {entry.level}"
        )

        extraction_meta = self.query_one("#modal-meta-extraction", Static)
        if entry.extraction_id:
            extraction_meta.update(f"Extraction: {entry.extraction_id}")
        else:
            extraction_meta.display = False

        agent_meta = self.query_one("#modal-meta-agent", Static)
        if entry.agent_id:
            agent_meta.update(f"Agent: {entry.agent_id}")
        else:
            agent_meta.display = False

        self.query_one("#modal-content", Static).update(entry.message)


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
        super().__init__(**kwargs)
        self._level = level
        self._count = count
        self._active = active

    def on_mount(self) -> None:
        self._update_display()

    def on_click(self) -> None:
        self._active = not self._active
        self._update_display()
        self.post_message(self.Toggled(self._level, self._active))

    def set_count(self, count: int) -> None:
        self._count = count
        if self.is_mounted:
            self._update_display()

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def level(self) -> str:
        return self._level

    def _update_display(self) -> None:
        level_colors = {
            "info": "$primary",
            "warning": "$warning",
            "error": "$error",
        }
        color = level_colors.get(self._level, "$foreground-muted")

        if self._active:
            self.update(
                f"[{color}]●[/] {self._level.upper():<8} "
                f"[$foreground-muted]{self._count:>4}[/]"
            )
        else:
            self.update(
                f"[$foreground-disabled]○ {self._level.upper():<8} {self._count:>4}[/]"
            )


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
        super().__init__(**kwargs)
        self._source = source
        self._label = label
        self._selected = selected

    def on_mount(self) -> None:
        self._update_display()

    def on_click(self) -> None:
        self.post_message(self.Selected(self._source))

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        if self.is_mounted:
            self._update_display()

    @property
    def source(self) -> str:
        return self._source

    def _update_display(self) -> None:
        if self._selected:
            self.update(f"[$accent]●[/] {self._label}")
        else:
            self.update(f"[$foreground-disabled]○[/] [$foreground-muted]{self._label}[/]")


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
        super().__init__(**kwargs)

    def on_mount(self) -> None:
        self._update_display()

    def _update_display(self) -> None:
        entry = self.entry

        level_colors = {
            "info": "$primary",
            "warning": "$warning",
            "error": "$error",
        }
        color = level_colors.get(entry.level, "$foreground-muted")

        # Truncate message if too long
        message = entry.message
        max_msg_len = 60
        if len(message) > max_msg_len:
            message = message[:max_msg_len - 3] + "..."

        content = (
            f"[$foreground-disabled]{entry.time_str}[/]  "
            f"[{color}]●[/]  "
            f"[$foreground-muted]{entry.source:<10}[/]  "
            f"{message}"
        )
        self.update(content)

    def on_click(self) -> None:
        self.post_message(self.Clicked(self.entry))


class LogPanel(Vertical):
    """Right panel showing log stream."""

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
        padding: 1 2;
        background: $background;
    }

    LogPanel .empty {
        height: 1fr;
        content-align: center middle;
        color: $foreground-muted;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._current_log_count: int = 0

    def compose(self) -> ComposeResult:
        with Vertical(classes="panel-header"):
            with Horizontal(classes="title-row"):
                yield Static("LOG STREAM", classes="title")
                yield Static(classes="count")
        yield VerticalScroll(classes="log-stream", id="log-stream")

    def set_logs(self, logs: list[LogEntry], total: int, force: bool = False) -> None:
        # Update count
        count = self.query_one(".count", Static)
        count.update(f"{len(logs)} / {total}")

        # Only rebuild if log count changed
        if len(logs) == self._current_log_count and not force:
            return

        self._current_log_count = len(logs)

        # Update log stream
        stream = self.query_one("#log-stream", VerticalScroll)
        stream.remove_children()

        if not logs:
            stream.mount(Static("No log entries", classes="empty"))
            return

        # Limit to last 500 for performance
        for entry in logs[-500:]:
            stream.mount(LogRow(entry))

        stream.scroll_end(animate=False)


class LogsScreen(Screen):
    """Full-screen log viewer with split-panel filtering."""

    BINDINGS = [
        Binding("1", "go_dashboard", ""),
        Binding("2", "go_samples", ""),
        Binding("3", "noop", ""),
        Binding("tab", "cycle", ""),
        Binding("q", "quit", ""),
        Binding("n", "new_task", ""),
        Binding("/", "focus_search", ""),
        Binding("escape", "clear_focus", "", show=False),
        Binding("g", "scroll_top", "", show=False),
        Binding("G", "scroll_bottom", "", show=False),
        Binding("?", "help", ""),
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
        get_state().add_listener(self._on_state_change)
        self._sync()

    def on_unmount(self) -> None:
        get_state().remove_listener(self._on_state_change)

    def _on_state_change(self, _) -> None:
        self._sync()

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

        # Update log panel
        self.query_one("#log-panel", LogPanel).set_logs(filtered, total, force=force_logs)
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

    def action_help(self) -> None:
        self.app.push_screen("help")

    def action_quit(self) -> None:
        self.app.exit()
