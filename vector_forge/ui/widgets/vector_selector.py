"""Vector selector panel for chat screen.

Shows available vectors from the selected extraction task
and provides generation settings controls.
"""

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Static, Input, Label
from textual.reactive import reactive

from vector_forge.ui.state import VectorInfo, get_state, ExtractionStatus


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

    def __init__(self, vector_info: VectorInfo, is_selected: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.vector_info = vector_info
        self._is_selected = is_selected

    def on_mount(self) -> None:
        """Update content on mount."""
        self._update_display()

    def _update_display(self) -> None:
        """Update the display content."""
        v = self.vector_info
        icon = "●" if self._is_selected else "○"
        color = "$accent" if self._is_selected else "$foreground-muted"
        best_marker = " [bold $success]best[/]" if v.is_best else ""

        content = (
            f"[{color}]{icon}[/] [bold]L{v.layer}[/] · {v.score:.2f}{best_marker}"
        )
        self.update(content)

        # Update CSS class
        if self._is_selected:
            self.add_class("-selected")
        else:
            self.remove_class("-selected")

    def set_selected(self, selected: bool) -> None:
        """Update selection state."""
        self._is_selected = selected
        self._update_display()

    def on_click(self) -> None:
        """Handle click to select this vector."""
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
        text-align: center;
        padding: 2;
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

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._selected_layer: int | None = None

    def compose(self) -> ComposeResult:
        # Header with task info
        with Vertical(classes="header"):
            yield Static("No task selected", classes="task-name", id="task-name")
            yield Static("", classes="task-model", id="task-model")

        # Vector list section
        yield Static("VECTORS", classes="section-title")
        yield VerticalScroll(classes="vector-list", id="vector-list")

        # Settings section
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

        # Update header
        task_name = self.query_one("#task-name", Static)
        task_model = self.query_one("#task-model", Static)
        vector_list = self.query_one("#vector-list", VerticalScroll)

        if extraction_id is None:
            task_name.update("No task selected")
            task_model.update("")
            vector_list.remove_children()
            vector_list.mount(Static("Select a task first", classes="empty-state"))
            return

        extraction = state.extractions.get(extraction_id)
        if extraction is None:
            task_name.update("Task not found")
            task_model.update("")
            return

        # Update task info
        task_name.update(f"[bold]{extraction.behavior_name}[/]")
        # Show target model (used for chat inference)
        model_display = extraction.target_model or extraction.model
        if model_display:
            # Shorten model name for display
            short_name = model_display.split("/")[-1] if "/" in model_display else model_display
            task_model.update(f"[$foreground-muted]{short_name}[/]")
        else:
            task_model.update("")

        # Check if extraction is complete
        if extraction.status != ExtractionStatus.COMPLETE:
            vector_list.remove_children()
            vector_list.mount(
                Static(
                    f"Extraction {extraction.status.value}...\nVectors available after completion.",
                    classes="empty-state",
                )
            )
            return

        # Update vectors
        self._update_vectors(extraction_id)

    def _update_vectors(self, extraction_id: str) -> None:
        """Update vector list from extraction."""
        state = get_state()
        extraction = state.extractions.get(extraction_id)
        if extraction is None:
            return

        vector_list = self.query_one("#vector-list", VerticalScroll)

        # Build vector info from extraction evaluation
        # In a real implementation, we'd query the session store for all layer vectors
        # For now, use the evaluation metrics as a proxy
        vectors = state.chat.available_vectors

        if not vectors:
            # Create placeholder vectors based on evaluation
            best_layer = extraction.evaluation.best_layer or 16
            vectors = [
                VectorInfo(
                    layer=best_layer,
                    score=extraction.evaluation.overall,
                    vector_path=f"vectors/final.pt",
                    is_best=True,
                ),
            ]
            state.chat.available_vectors = vectors

        # Deduplicate by layer (keep first occurrence)
        seen_layers: set[int] = set()
        unique_vectors = []
        for v in vectors:
            if v.layer not in seen_layers:
                seen_layers.add(v.layer)
                unique_vectors.append(v)

        # Check which rows already exist
        existing_ids = {w.id for w in vector_list.query(VectorRow)}
        needed_ids = {f"vector-{v.layer}" for v in unique_vectors}

        # Remove rows that are no longer needed
        for widget in list(vector_list.query(VectorRow)):
            if widget.id not in needed_ids:
                widget.remove()

        # Mount or update vector rows
        for v in unique_vectors:
            row_id = f"vector-{v.layer}"
            is_selected = (
                self._selected_layer == v.layer
                if self._selected_layer is not None
                else v.is_best
            )

            if row_id in existing_ids:
                # Update existing row
                try:
                    existing = vector_list.query_one(f"#{row_id}", VectorRow)
                    existing.set_selected(is_selected)
                except Exception:
                    pass
            else:
                # Mount new row
                vector_list.mount(
                    VectorRow(v, is_selected=is_selected, id=row_id)
                )

        # Auto-select best if nothing selected
        if self._selected_layer is None:
            for v in unique_vectors:
                if v.is_best:
                    self._selected_layer = v.layer
                    break
            if self._selected_layer is None and unique_vectors:
                self._selected_layer = unique_vectors[0].layer

    def on_vector_row_selected(self, event: VectorRow.Selected) -> None:
        """Handle vector selection."""
        self._selected_layer = event.vector_info.layer

        # Update all rows
        for row in self.query(VectorRow):
            row.set_selected(row.vector_info.layer == self._selected_layer)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission (settings change)."""
        self._emit_settings_changed()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input change."""
        # Validate and update reactive properties
        try:
            if event.input.id == "strength-input":
                self.strength = float(event.value) if event.value else 1.0
            elif event.input.id == "temp-input":
                self.temperature = float(event.value) if event.value else 0.7
            elif event.input.id == "tokens-input":
                self.max_tokens = int(event.value) if event.value else 256
        except ValueError:
            pass  # Ignore invalid values

    def _emit_settings_changed(self) -> None:
        """Emit settings changed message."""
        self.post_message(
            self.SettingsChanged(
                strength=self.strength,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        )

    @property
    def selected_layer(self) -> int | None:
        """Get the currently selected layer."""
        return self._selected_layer
