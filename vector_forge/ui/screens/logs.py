"""Logs screen - full view of event logs."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Static, Input

from vector_forge.ui.state import LogEntry, get_state
from vector_forge.ui.theme import COLORS
from vector_forge.ui.widgets.tmux_bar import TmuxBar


class LogRow(Static):
    """Single log entry row."""

    DEFAULT_CSS = """
    LogRow {
        height: 1;
    }

    LogRow.-warning {
        color: $warning;
    }

    LogRow.-error {
        color: $error;
    }
    """

    def __init__(self, entry: LogEntry, **kwargs) -> None:
        level_colors = {
            "info": COLORS.text,
            "warning": COLORS.warning,
            "error": COLORS.error,
        }
        color = level_colors.get(entry.level, COLORS.text)

        content = (
            f"[{COLORS.text_dim}]{entry.time_str}[/] "
            f"[{COLORS.text_muted}]{entry.source:<12}[/] "
            f"[{color}]{entry.message}[/]"
        )
        super().__init__(content, **kwargs)

        if entry.level in ("warning", "error"):
            self.add_class(f"-{entry.level}")


class LogsScreen(Screen):
    """Full-screen log viewer with filtering."""

    BINDINGS = [
        Binding("1", "go_dashboard", ""),
        Binding("2", "go_samples", ""),
        Binding("3", "noop", ""),
        Binding("tab", "cycle", ""),
        Binding("q", "quit", ""),
        Binding("n", "new_task", ""),
        Binding("/", "focus_filter", ""),
        Binding("escape", "clear_filter", "", show=False),
        Binding("g", "scroll_top", "", show=False),
        Binding("G", "scroll_bottom", "", show=False),
        Binding("?", "help", ""),
    ]

    DEFAULT_CSS = """
    LogsScreen {
        background: $background;
    }

    LogsScreen #header {
        height: 1;
        padding: 0 2;
        margin-bottom: 1;
    }

    LogsScreen #header-title {
        width: 1fr;
        text-style: bold;
    }

    LogsScreen #header-count {
        width: auto;
        color: $text-muted;
    }

    LogsScreen #filter-bar {
        height: 1;
        padding: 0 2;
        margin-bottom: 1;
    }

    LogsScreen #filter-label {
        width: auto;
        color: $text-muted;
    }

    LogsScreen #filter-input {
        width: 40;
        margin-left: 1;
    }

    LogsScreen #log-scroll {
        height: 1fr;
        padding: 0 2;
    }

    LogsScreen .log-empty {
        color: $text-muted;
        padding: 2;
        text-align: center;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._filter_text = ""

    def compose(self) -> ComposeResult:
        with Horizontal(id="header"):
            yield Static("LOGS", id="header-title")
            yield Static(id="header-count")

        with Horizontal(id="filter-bar"):
            yield Static("Filter:", id="filter-label")
            yield Input(placeholder="search logs...", id="filter-input")

        yield VerticalScroll(id="log-scroll")
        yield TmuxBar(active_screen="logs")

    def on_mount(self) -> None:
        get_state().add_listener(self._on_state_change)
        self._sync()

    def on_unmount(self) -> None:
        get_state().remove_listener(self._on_state_change)

    def _on_state_change(self, _) -> None:
        self._sync()

    def _sync(self) -> None:
        state = get_state()
        filtered = self._get_filtered_logs()

        count = self.query_one("#header-count", Static)
        count.update(f"{len(filtered)} / {len(state.logs)} entries")

        self._refresh_logs(filtered)
        self.query_one(TmuxBar).refresh_info()

    def _get_filtered_logs(self) -> list[LogEntry]:
        state = get_state()
        logs = state.get_filtered_logs(extraction_id=state.selected_id)

        if self._filter_text:
            filter_lower = self._filter_text.lower()
            logs = [
                log for log in logs
                if filter_lower in log.message.lower()
                or filter_lower in log.source.lower()
            ]

        return logs

    def _refresh_logs(self, logs: list[LogEntry]) -> None:
        scroll = self.query_one("#log-scroll", VerticalScroll)
        scroll.remove_children()

        if not logs:
            scroll.mount(Static("No log entries", classes="log-empty"))
            return

        for entry in logs[-500:]:
            scroll.mount(LogRow(entry))

        scroll.scroll_end(animate=False)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "filter-input":
            self._filter_text = event.value
            self._sync()

    def action_noop(self) -> None:
        pass

    def action_go_dashboard(self) -> None:
        self.app.switch_screen("dashboard")

    def action_go_samples(self) -> None:
        self.app.switch_screen("samples")

    def action_cycle(self) -> None:
        self.app.switch_screen("dashboard")

    def action_focus_filter(self) -> None:
        self.query_one("#filter-input", Input).focus()

    def action_clear_filter(self) -> None:
        filter_input = self.query_one("#filter-input", Input)
        filter_input.value = ""
        self._filter_text = ""
        self._sync()
        self.focus()

    def action_scroll_top(self) -> None:
        self.query_one("#log-scroll", VerticalScroll).scroll_home(animate=False)

    def action_scroll_bottom(self) -> None:
        self.query_one("#log-scroll", VerticalScroll).scroll_end(animate=False)

    def action_new_task(self) -> None:
        self.app.push_screen("create_task")

    def action_help(self) -> None:
        self.app.push_screen("help")

    def action_quit(self) -> None:
        self.app.exit()
