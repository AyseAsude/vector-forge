"""Model selection screen for choosing LLM configurations."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static, Button, Input, Select
from textual.message import Message

from vector_forge.storage.models import (
    ModelConfig,
    ModelConfigManager,
    Provider,
    COMMON_MODELS,
)
from vector_forge.ui.theme import COLORS, ICONS
from vector_forge.ui.widgets.model_card import ModelCardCompact, PROVIDER_INFO


class ModelSelectorScreen(ModalScreen):
    """Modal screen for selecting or creating model configurations."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    DEFAULT_CSS = """
    ModelSelectorScreen {
        align: center middle;
    }

    ModelSelectorScreen #dialog {
        width: 72;
        height: auto;
        max-height: 85%;
        background: $surface;
        padding: 1 2;
    }

    /* Header */
    ModelSelectorScreen #header {
        height: 2;
        margin-bottom: 1;
    }

    ModelSelectorScreen #title {
        width: 1fr;
        text-style: bold;
    }

    ModelSelectorScreen #close-btn {
        width: auto;
        color: $text-muted;
    }

    ModelSelectorScreen #close-btn:hover {
        color: $text;
    }

    /* Section titles */
    ModelSelectorScreen .section-title {
        height: 1;
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }

    /* Models list - scrollable */
    ModelSelectorScreen #models-scroll {
        height: auto;
        min-height: 4;
        max-height: 16;
        margin-bottom: 1;
        scrollbar-gutter: stable;
    }

    ModelSelectorScreen #models-scroll:focus-within {
        scrollbar-color: $accent;
    }

    ModelSelectorScreen .empty-list {
        height: 3;
        content-align: center middle;
        color: $text-muted;
    }

    /* Add new button row */
    ModelSelectorScreen #add-row {
        height: 3;
        margin-bottom: 1;
    }

    ModelSelectorScreen #btn-add {
        width: 1fr;
        height: 3;
        background: $background;
        color: $text-muted;
        border: none;
        content-align: center middle;
    }

    ModelSelectorScreen #btn-add:hover {
        background: $surface-hl;
        color: $text;
    }

    ModelSelectorScreen #btn-add:focus {
        background: $surface-hl;
    }

    /* New model form */
    ModelSelectorScreen #new-model-form {
        height: auto;
        padding: 1;
        background: $background;
        margin-bottom: 1;
        display: none;
    }

    ModelSelectorScreen #new-model-form.-visible {
        display: block;
    }

    ModelSelectorScreen .form-row {
        height: 3;
        margin-bottom: 1;
    }

    ModelSelectorScreen .form-label {
        width: 10;
        height: 3;
        content-align: left middle;
        color: $text-muted;
    }

    ModelSelectorScreen .form-input {
        width: 1fr;
    }

    ModelSelectorScreen Input {
        background: $surface;
        border: none;
        height: 3;
    }

    ModelSelectorScreen Input:focus {
        background: $surface-hl;
    }

    ModelSelectorScreen Select {
        background: $surface;
        border: none;
        height: 3;
    }

    ModelSelectorScreen Select:focus {
        background: $surface-hl;
    }

    /* Form buttons */
    ModelSelectorScreen #form-buttons {
        height: 3;
        margin-top: 1;
    }

    ModelSelectorScreen #form-spacer {
        width: 1fr;
    }

    ModelSelectorScreen #btn-form-cancel {
        width: auto;
        min-width: 10;
        height: 3;
        background: $surface-hl;
        color: $text;
        border: none;
        margin-right: 1;
    }

    ModelSelectorScreen #btn-form-cancel:hover {
        background: $primary 20%;
    }

    ModelSelectorScreen #btn-save {
        width: auto;
        min-width: 10;
        height: 3;
        background: $success;
        color: $background;
        border: none;
        text-style: bold;
    }

    ModelSelectorScreen #btn-save:hover {
        background: $success 80%;
    }

    /* Bottom cancel */
    ModelSelectorScreen #bottom-row {
        height: 3;
    }

    ModelSelectorScreen #btn-cancel {
        width: 1fr;
        height: 3;
        background: $surface-hl;
        color: $text-muted;
        border: none;
    }

    ModelSelectorScreen #btn-cancel:hover {
        background: $primary 20%;
        color: $text;
    }
    """

    class ModelSelected(Message):
        """Emitted when a model is selected."""

        def __init__(self, field_name: str, config: ModelConfig) -> None:
            super().__init__()
            self.field_name = field_name
            self.config = config

    def __init__(
        self,
        field_name: str,
        current_config: ModelConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.field_name = field_name
        self.current_config = current_config
        self._manager = ModelConfigManager()
        self._show_new_form = False
        self._selected_provider = Provider.OPENAI

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            # Header
            with Horizontal(id="header"):
                field_display = self.field_name.upper()
                yield Static(f"SELECT {field_display} MODEL", id="title")
                yield Static(f"[{COLORS.text_dim}]ESC[/]", id="close-btn")

            # Saved models section
            yield Static("SAVED MODELS", classes="section-title")
            yield VerticalScroll(id="models-scroll")

            # Add new button (hidden when form is visible)
            with Horizontal(id="add-row"):
                yield Button(f"[{COLORS.text_dim}]+[/] Add New Model", id="btn-add")

            # New model form (hidden by default)
            with Vertical(id="new-model-form"):
                yield Static("NEW MODEL", classes="section-title")

                with Horizontal(classes="form-row"):
                    yield Static("Provider", classes="form-label")
                    yield Select(
                        [(p.value.title(), p.value) for p in Provider if p != Provider.CUSTOM],
                        value=Provider.OPENAI.value,
                        id="sel-provider",
                        classes="form-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Static("Model", classes="form-label")
                    yield Select(
                        [(m, m) for m in COMMON_MODELS.get(Provider.OPENAI, [])],
                        id="sel-model",
                        classes="form-input",
                    )

                with Horizontal(classes="form-row"):
                    yield Static("Name", classes="form-label")
                    yield Input(placeholder="Display name (optional)", id="inp-name", classes="form-input")

                with Horizontal(classes="form-row"):
                    yield Static("API Base", classes="form-label")
                    yield Input(placeholder="Custom endpoint (optional)", id="inp-api-base", classes="form-input")

                with Horizontal(classes="form-row"):
                    yield Static("API Key", classes="form-label")
                    yield Input(placeholder="Uses env var if empty", id="inp-api-key", password=True, classes="form-input")

                with Horizontal(id="form-buttons"):
                    yield Static("", id="form-spacer")
                    yield Button("Cancel", id="btn-form-cancel")
                    yield Button("Save Model", id="btn-save")

            # Bottom cancel
            with Horizontal(id="bottom-row"):
                yield Button("Cancel", id="btn-cancel")

    def on_mount(self) -> None:
        self._populate_models()

    def _populate_models(self) -> None:
        """Populate the models list."""
        models_scroll = self.query_one("#models-scroll", VerticalScroll)
        models_scroll.remove_children()

        configs = self._manager.list_all()

        if not configs:
            models_scroll.mount(
                Static(
                    f"[{COLORS.text_muted}]No saved models\nClick 'Add New Model' to create one[/]",
                    classes="empty-list"
                )
            )
            return

        for config in configs:
            is_selected = (
                self.current_config is not None
                and self.current_config.id == config.id
            )
            card = ModelCardCompact(config, is_selected=is_selected)
            models_scroll.mount(card)

    def on_model_card_compact_selected(self, event: ModelCardCompact.Selected) -> None:
        """Handle model selection."""
        self._manager.update_last_used(event.config.id)
        self.dismiss(self.ModelSelected(self.field_name, event.config))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.dismiss(None)
        elif event.button.id == "btn-add":
            self._show_form()
        elif event.button.id == "btn-form-cancel":
            self._hide_form()
        elif event.button.id == "btn-save":
            self._save_new_model()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "sel-provider":
            self._on_provider_changed(event.value)

    def _on_provider_changed(self, provider_value: str) -> None:
        """Update model list when provider changes."""
        try:
            provider = Provider(provider_value)
        except ValueError:
            return

        self._selected_provider = provider
        models = COMMON_MODELS.get(provider, [])

        model_select = self.query_one("#sel-model", Select)
        model_select.set_options([(m, m) for m in models])

        if models:
            model_select.value = models[0]

    def _show_form(self) -> None:
        """Show the new model form."""
        self._show_new_form = True
        form = self.query_one("#new-model-form")
        form.set_class(True, "-visible")
        add_row = self.query_one("#add-row")
        add_row.display = False

    def _hide_form(self) -> None:
        """Hide the new model form."""
        self._show_new_form = False
        form = self.query_one("#new-model-form")
        form.set_class(False, "-visible")
        add_row = self.query_one("#add-row")
        add_row.display = True
        self._clear_form()

    def _clear_form(self) -> None:
        """Clear form inputs."""
        self.query_one("#inp-name", Input).value = ""
        self.query_one("#inp-api-base", Input).value = ""
        self.query_one("#inp-api-key", Input).value = ""

    def _save_new_model(self) -> None:
        """Save a new model configuration."""
        model_select = self.query_one("#sel-model", Select)
        name_input = self.query_one("#inp-name", Input)
        api_base_input = self.query_one("#inp-api-base", Input)
        api_key_input = self.query_one("#inp-api-key", Input)

        model = model_select.value
        if not model:
            return

        # Generate name if not provided
        name = name_input.value.strip()
        if not name:
            provider_name, _ = PROVIDER_INFO.get(self._selected_provider, ("", ""))
            model_short = str(model).split("/")[-1]
            name = f"{provider_name} {model_short}"

        config = ModelConfig(
            name=name,
            provider=self._selected_provider,
            model=str(model),
            api_base=api_base_input.value.strip() or None,
            api_key=api_key_input.value.strip() or None,
        )

        self._manager.add(config)
        self._populate_models()
        self._hide_form()

    def action_cancel(self) -> None:
        self.dismiss(None)
