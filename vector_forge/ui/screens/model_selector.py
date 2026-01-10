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
from vector_forge.ui.theme import COLORS
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
        width: 70;
        height: auto;
        max-height: 80%;
        background: $surface;
        padding: 1 2;
    }

    ModelSelectorScreen #title {
        height: 1;
        text-style: bold;
        margin-bottom: 1;
    }

    ModelSelectorScreen #subtitle {
        height: 1;
        color: $text-muted;
        margin-bottom: 1;
    }

    ModelSelectorScreen .section-title {
        height: 1;
        color: $accent;
        margin-top: 1;
        margin-bottom: 1;
    }

    ModelSelectorScreen #models-list {
        height: auto;
        max-height: 15;
        margin-bottom: 1;
    }

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
        width: 12;
        height: 1;
        color: $text-muted;
    }

    ModelSelectorScreen .form-input {
        width: 1fr;
    }

    ModelSelectorScreen Input {
        background: $surface;
        border: none;
    }

    ModelSelectorScreen Input:focus {
        background: $surface-hl;
    }

    ModelSelectorScreen Select {
        background: $surface;
        border: none;
    }

    ModelSelectorScreen #buttons {
        height: 3;
        margin-top: 1;
    }

    ModelSelectorScreen #btn-add {
        width: auto;
        min-width: 12;
        height: 1;
        background: $accent;
        color: $background;
        border: none;
        padding: 0 1;
    }

    ModelSelectorScreen #btn-add:hover {
        background: $accent 80%;
    }

    ModelSelectorScreen #btn-cancel {
        width: auto;
        min-width: 10;
        height: 1;
        background: $surface-hl;
        color: $text;
        border: none;
        padding: 0 1;
        margin-left: 1;
    }

    ModelSelectorScreen #btn-save {
        width: auto;
        min-width: 10;
        height: 1;
        background: $success;
        color: $background;
        border: none;
        padding: 0 1;
        margin-left: 1;
    }

    ModelSelectorScreen #spacer {
        width: 1fr;
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
            yield Static("SELECT MODEL", id="title")
            yield Static(f"for {self.field_name}", id="subtitle")

            # Saved models list
            yield Static("SAVED MODELS", classes="section-title")
            yield VerticalScroll(id="models-list")

            # Add new button
            with Horizontal(id="buttons"):
                yield Button("+ Add New", id="btn-add")
                yield Static("", id="spacer")
                yield Button("Cancel", id="btn-cancel")

            # New model form (hidden by default)
            with Vertical(id="new-model-form"):
                yield Static("ADD NEW MODEL", classes="section-title")

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
                    yield Input(placeholder="Display name", id="inp-name", classes="form-input")

                with Horizontal(classes="form-row"):
                    yield Static("API Base", classes="form-label")
                    yield Input(placeholder="Optional custom endpoint", id="inp-api-base", classes="form-input")

                with Horizontal(classes="form-row"):
                    yield Static("API Key", classes="form-label")
                    yield Input(placeholder="Optional (uses env var)", id="inp-api-key", password=True, classes="form-input")

                with Horizontal(id="buttons"):
                    yield Static("", id="spacer")
                    yield Button("Cancel", id="btn-form-cancel")
                    yield Button("Save", id="btn-save")

    def on_mount(self) -> None:
        self._populate_models()

    def _populate_models(self) -> None:
        """Populate the models list."""
        models_list = self.query_one("#models-list", VerticalScroll)
        models_list.remove_children()

        configs = self._manager.list_all()

        if not configs:
            models_list.mount(Static(f"[{COLORS.text_muted}]No saved models[/]"))
            return

        for config in configs:
            is_selected = (
                self.current_config is not None
                and self.current_config.id == config.id
            )
            card = ModelCardCompact(config, is_selected=is_selected)
            models_list.mount(card)

    def on_model_card_compact_selected(self, event: ModelCardCompact.Selected) -> None:
        """Handle model selection."""
        self._manager.update_last_used(event.config.id)
        self.post_message(self.ModelSelected(self.field_name, event.config))
        self.dismiss()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.dismiss()
        elif event.button.id == "btn-add":
            self._toggle_new_form(True)
        elif event.button.id == "btn-form-cancel":
            self._toggle_new_form(False)
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

    def _toggle_new_form(self, show: bool) -> None:
        """Toggle the new model form visibility."""
        self._show_new_form = show
        form = self.query_one("#new-model-form")
        form.set_class(show, "-visible")

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
        self._toggle_new_form(False)

        # Clear form
        name_input.value = ""
        api_base_input.value = ""
        api_key_input.value = ""

    def action_cancel(self) -> None:
        self.dismiss()
