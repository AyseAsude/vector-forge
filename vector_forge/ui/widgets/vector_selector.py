"""Vector selector panel for chat screen.

Shows available vectors from the selected extraction task
and provides generation settings controls.
"""

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Static, Input, Label
from textual.reactive import reactive

from vector_forge.ui.state import VectorInfo, get_state, ExtractionStatus, ExtractionUIState


class VectorRow(Static):
    """Clickable row for a vector layer."""

    DEFAULT_CSS = """
    VectorRow {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
    }

    VectorRow:hover {
        background: $boost;
    }

    VectorRow.-selected {
        background: $primary 15%;
        border-left: wide $accent;
    }
    """

    class Selected(Message):
        """Emitted when a vector is selected."""

        def __init__(self, vector_info: VectorInfo) -> None:
            super().__init__()
            self.vector_info = vector_info

    def __init__(self, vector_info: VectorInfo, is_selected: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.vector_info = vector_info
        self._is_selected = is_selected

    def on_mount(self) -> None:
        self._update_display()

    def _update_display(self) -> None:
        v = self.vector_info
        icon = "●" if self._is_selected else "○"
        color = "$accent" if self._is_selected else "$foreground-muted"
        best_marker = " [bold $success]best[/]" if v.is_best else ""

        # Format score: lower loss is better, so display it clearly
        score_display = f"{v.score:.2f}" if v.score > 0 else "—"
        content = f"[{color}]{icon}[/] [bold]L{v.layer}[/] · {score_display}{best_marker}"
        self.update(content)

        if self._is_selected:
            self.add_class("-selected")
        else:
            self.remove_class("-selected")

    def set_selected(self, selected: bool) -> None:
        self._is_selected = selected
        self._update_display()

    def on_click(self) -> None:
        self.post_message(self.Selected(self.vector_info))


class VectorSelector(Vertical):
    """Left panel for selecting vectors and configuring generation settings."""

    DEFAULT_CSS = """
    VectorSelector {
        width: 28;
        padding: 1 1;
        background: $surface;
    }

    VectorSelector .header {
        height: auto;
        margin-bottom: 1;
    }

    VectorSelector .task-name {
        text-style: bold;
    }

    VectorSelector .task-model {
        color: $foreground-muted;
        height: auto;
    }

    VectorSelector .section-title {
        height: 1;
        color: $accent;
        text-style: bold;
        margin-top: 1;
        margin-bottom: 1;
    }

    VectorSelector .vector-list {
        height: 1fr;
        min-height: 5;
    }

    VectorSelector .settings {
        height: auto;
        margin-top: 1;
        border-top: solid $surface-lighten-1;
        padding-top: 1;
    }

    VectorSelector .setting-row {
        height: auto;
        margin-bottom: 1;
    }

    VectorSelector .setting-label {
        height: 1;
        color: $foreground-muted;
    }

    VectorSelector .setting-value {
        height: auto;
        width: 100%;
    }

    VectorSelector Input {
        height: 1;
        min-height: 1;
        border: none;
        background: $background;
        padding: 0 1;
    }

    VectorSelector .empty-state {
        height: auto;
        color: $foreground-muted;
        padding: 1 0;
    }

    """

    # Reactive properties for settings
    strength: reactive[float] = reactive(1.0)
    temperature: reactive[float] = reactive(0.7)
    max_tokens: reactive[int] = reactive(256)

    class SettingsChanged(Message):
        """Emitted when generation settings change."""

        def __init__(
            self,
            strength: float,
            temperature: float,
            max_tokens: int,
        ) -> None:
            super().__init__()
            self.strength = strength
            self.temperature = temperature
            self.max_tokens = max_tokens

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._selected_layer: int | None = None
        self._current_extraction_id: str | None = None
        self._vectors: list[VectorInfo] = []

    def compose(self) -> ComposeResult:
        with Vertical(classes="header"):
            yield Static("No task selected", classes="task-name", id="task-name")
            yield Static("", classes="task-model", id="task-model")

        yield Static("VECTORS", classes="section-title")
        yield VerticalScroll(classes="vector-list", id="vector-list")

        with Vertical(classes="settings"):
            yield Static("SETTINGS", classes="section-title")

            with Vertical(classes="setting-row"):
                yield Static("Strength:", classes="setting-label")
                yield Input(
                    value="1.0",
                    placeholder="1.0",
                    id="strength-input",
                    classes="setting-value",
                )

            with Vertical(classes="setting-row"):
                yield Static("Temperature:", classes="setting-label")
                yield Input(
                    value="0.7",
                    placeholder="0.7",
                    id="temp-input",
                    classes="setting-value",
                )

            with Vertical(classes="setting-row"):
                yield Static("Max tokens:", classes="setting-label")
                yield Input(
                    value="256",
                    placeholder="256",
                    id="tokens-input",
                    classes="setting-value",
                )

    def update_from_extraction(self, extraction_id: str | None) -> None:
        """Update the panel based on selected extraction."""
        state = get_state()
        task_name = self.query_one("#task-name", Static)
        task_model = self.query_one("#task-model", Static)
        vector_list = self.query_one("#vector-list", VerticalScroll)

        # Handle no extraction selected
        if extraction_id is None:
            self._clear_state()
            task_name.update("No task selected")
            task_model.update("")
            vector_list.remove_children()
            vector_list.mount(Static("Select a task first", classes="empty-state"))
            return

        extraction = state.extractions.get(extraction_id)
        if extraction is None:
            self._clear_state()
            task_name.update("Task not found")
            task_model.update("")
            return

        # Update task info
        task_name.update(f"[bold]{extraction.behavior_name}[/]")
        model_display = extraction.target_model or extraction.model
        if model_display:
            short_name = model_display.split("/")[-1] if "/" in model_display else model_display
            task_model.update(f"[$foreground-muted]{short_name}[/]")
        else:
            task_model.update("")

        # Check if extraction is complete
        if extraction.status != ExtractionStatus.COMPLETE:
            self._clear_state()
            vector_list.remove_children()
            vector_list.mount(
                Static("Extraction running...", classes="empty-state")
            )
            return

        # Check if we need to reload vectors (extraction changed)
        if extraction_id != self._current_extraction_id:
            self._current_extraction_id = extraction_id
            self._selected_layer = None
            self._vectors = self._load_vectors(extraction_id, extraction)

        self._render_vectors(vector_list)

    def _clear_state(self) -> None:
        """Clear internal state when extraction changes."""
        self._current_extraction_id = None
        self._selected_layer = None
        self._vectors = []

    def _load_vectors(self, extraction_id: str, extraction: ExtractionUIState) -> list[VectorInfo]:
        """Load vectors from session store."""
        try:
            from vector_forge.services.session import SessionService
            from vector_forge.storage import SessionReplayer

            service = SessionService()
            store = service.get_session_store(extraction_id)
            replayer = SessionReplayer(store)
            replayed = replayer.reconstruct_state()

            if not replayed.vectors:
                return self._create_fallback_vector(extraction)

            # Convert replayed vectors to VectorInfo
            vectors = []
            for layer, rv in sorted(replayed.vectors.items()):
                vectors.append(
                    VectorInfo(
                        layer=layer,
                        score=rv.score,
                        vector_path=rv.vector_ref,
                        is_best=(layer == replayed.best_layer),
                    )
                )

            return vectors if vectors else self._create_fallback_vector(extraction)

        except Exception:
            return self._create_fallback_vector(extraction)

    def _create_fallback_vector(self, extraction: ExtractionUIState) -> list[VectorInfo]:
        """Create fallback vector when loading fails."""
        best_layer = extraction.evaluation.best_layer or 16
        return [
            VectorInfo(
                layer=best_layer,
                score=extraction.evaluation.overall,
                vector_path="vectors/final.pt",
                is_best=True,
            )
        ]

    def _render_vectors(self, vector_list: VerticalScroll) -> None:
        """Render vector list UI."""
        vector_list.remove_children()

        if not self._vectors:
            vector_list.mount(Static("No vectors found", classes="empty-state"))
            return

        # Auto-select best if nothing selected
        if self._selected_layer is None:
            for v in self._vectors:
                if v.is_best:
                    self._selected_layer = v.layer
                    break
            if self._selected_layer is None:
                self._selected_layer = self._vectors[0].layer

        # Mount vector rows (use index for unique ID, not layer which may have duplicates)
        for idx, v in enumerate(self._vectors):
            is_selected = (v.layer == self._selected_layer)
            vector_list.mount(
                VectorRow(v, is_selected=is_selected, id=f"vector-{idx}")
            )

    def on_vector_row_selected(self, event: VectorRow.Selected) -> None:
        """Handle vector selection."""
        self._selected_layer = event.vector_info.layer

        for row in self.query(VectorRow):
            row.set_selected(row.vector_info.layer == self._selected_layer)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._emit_settings_changed()

    def on_input_changed(self, event: Input.Changed) -> None:
        try:
            if event.input.id == "strength-input":
                self.strength = float(event.value) if event.value else 1.0
            elif event.input.id == "temp-input":
                self.temperature = float(event.value) if event.value else 0.7
            elif event.input.id == "tokens-input":
                self.max_tokens = int(event.value) if event.value else 256
        except ValueError:
            pass

    def _emit_settings_changed(self) -> None:
        self.post_message(
            self.SettingsChanged(
                strength=self.strength,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        )

    @property
    def selected_layer(self) -> int | None:
        return self._selected_layer
