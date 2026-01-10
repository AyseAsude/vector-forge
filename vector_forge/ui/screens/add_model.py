"""Add model screen for creating new LLM configurations."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal
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
from vector_forge.ui.widgets.model_card import PROVIDER_INFO


class AddModelScreen(ModalScreen):
    """Modal screen for adding a new model configuration."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    DEFAULT_CSS = """
    AddModelScreen {
        align: center middle;
    }

    AddModelScreen #dialog {
        width: 60;
        height: auto;
        background: $surface;
        padding: 1 2;
    }

    /* Header */
    AddModelScreen #header {
        height: 1;
        margin-bottom: 1;
    }

    AddModelScreen #title {
        width: 1fr;
        text-style: bold;
    }

    AddModelScreen #close-hint {
        width: auto;
        color: $text-dim;
    }

    /* Form */
    AddModelScreen .form-row {
        height: 3;
        margin-bottom: 1;
    }

    AddModelScreen .form-label {
        width: 10;
        height: 3;
        content-align: left middle;
        color: $text-muted;
    }

    AddModelScreen .form-input {
        width: 1fr;
    }

    AddModelScreen Input {
        background: $background;
        border: none;
        height: 3;
    }

    AddModelScreen Input:focus {
        background: $surface-hl;
    }

    AddModelScreen Select {
        background: $background;
        border: none;
        height: 3;
    }

    AddModelScreen Select:focus {
        background: $surface-hl;
    }

    /* Buttons */
    AddModelScreen #buttons {
        height: 3;
        margin-top: 1;
    }

    AddModelScreen #spacer {
        width: 1fr;
    }

    AddModelScreen #btn-cancel {
        width: auto;
        min-width: 10;
        height: 3;
        background: $surface-hl;
        color: $text;
        border: none;
        margin-right: 1;
    }

    AddModelScreen #btn-cancel:hover {
        background: $primary 20%;
    }

    AddModelScreen #btn-save {
        width: auto;
        min-width: 12;
        height: 3;
        background: $success;
        color: $background;
        border: none;
        text-style: bold;
    }

    AddModelScreen #btn-save:hover {
        background: $success 80%;
    }
    """

    class ModelAdded(Message):
        """Emitted when a model is added."""

        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            self.config = config

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._manager = ModelConfigManager()
        self._selected_provider = Provider.OPENAI

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            # Header
            with Horizontal(id="header"):
                yield Static("ADD NEW MODEL", id="title")
                yield Static(f"[{COLORS.text_dim}]ESC[/]", id="close-hint")

            # Form
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

            # Buttons
            with Horizontal(id="buttons"):
                yield Static("", id="spacer")
                yield Button("Cancel", id="btn-cancel")
                yield Button("Save Model", id="btn-save")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.dismiss(None)
        elif event.button.id == "btn-save":
            self._save_model()

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

    def _save_model(self) -> None:
        """Save the new model configuration."""
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
        self.dismiss(self.ModelAdded(config))

    def action_cancel(self) -> None:
        self.dismiss(None)
