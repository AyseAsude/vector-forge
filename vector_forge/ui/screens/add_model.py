"""Add model screen for creating new LLM configurations."""

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
)


# Provider display names (no colors needed here - just for generating names)
PROVIDER_NAMES = {
    Provider.OPENAI: "OpenAI",
    Provider.ANTHROPIC: "Anthropic",
    Provider.OPENROUTER: "OpenRouter",
    Provider.AZURE: "Azure",
    Provider.OLLAMA: "Ollama",
    Provider.CUSTOM: "Custom",
}


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
        max-height: 90%;
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
        color: $foreground-disabled;
    }

    /* Form content - scrollable */
    AddModelScreen #form-content {
        height: auto;
        max-height: 30;
    }

    /* Form rows */
    AddModelScreen .form-row {
        height: 1;
        margin-bottom: 1;
    }

    AddModelScreen .form-label {
        width: 12;
        color: $foreground-muted;
    }

    /* Input styling - matches ParamRow pattern */
    AddModelScreen Input {
        width: 1fr;
        height: 1;
        background: $boost;
        border: none;
        padding: 0 1;
    }

    AddModelScreen Input:focus {
        background: $background;
    }

    /* Select widget - clean style without borders */
    AddModelScreen Select {
        width: 1fr;
        height: 1;
        background: $boost;
        border: none;
    }

    AddModelScreen Select > SelectCurrent {
        background: $boost;
        border: none;
        padding: 0 1;
    }

    AddModelScreen Select:focus > SelectCurrent {
        background: $background;
        border: none;
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
        background: $boost;
        color: $foreground;
        border: none;
        margin-right: 1;
    }

    AddModelScreen #btn-cancel:hover {
        background: $primary 20%;
    }

    AddModelScreen #btn-cancel:focus {
        background: $boost;
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

    AddModelScreen #btn-save:focus {
        background: $success;
        text-style: bold;
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
        self._selected_provider = Provider.ANTHROPIC

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            # Header
            with Horizontal(id="header"):
                yield Static("ADD NEW MODEL", id="title")
                yield Static("", id="close-hint")

            # Form content - scrollable for small screens
            with VerticalScroll(id="form-content"):
                # Provider dropdown - no "Select" prompt, always has value
                # Anthropic is first in the list as the preferred provider
                with Horizontal(classes="form-row"):
                    yield Static("Provider", classes="form-label")
                    yield Select(
                        [(p.value.title(), p.value) for p in Provider if p != Provider.CUSTOM],
                        value=Provider.ANTHROPIC.value,
                        allow_blank=False,
                        id="sel-provider",
                    )

                # Model - text input
                with Horizontal(classes="form-row"):
                    yield Static("Model", classes="form-label")
                    yield Input(placeholder="e.g. claude-opus-4-5, gpt-5.2", id="inp-model")

                # Display name
                with Horizontal(classes="form-row"):
                    yield Static("Name", classes="form-label")
                    yield Input(placeholder="Display name (optional)", id="inp-name")

                # API Base
                with Horizontal(classes="form-row"):
                    yield Static("API Base", classes="form-label")
                    yield Input(placeholder="Custom endpoint (optional)", id="inp-api-base")

                # API Key
                with Horizontal(classes="form-row"):
                    yield Static("API Key", classes="form-label")
                    yield Input(placeholder="Uses env var if empty", id="inp-api-key", password=True)

            # Buttons - always visible outside scroll
            with Horizontal(id="buttons"):
                yield Static("", id="spacer")
                yield Button("Cancel", id="btn-cancel")
                yield Button("Save Model", id="btn-save")

    def on_mount(self) -> None:
        self.query_one("#close-hint", Static).update("[$foreground-disabled]ESC[/]")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.dismiss(None)
        elif event.button.id == "btn-save":
            self._save_model()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "sel-provider":
            try:
                self._selected_provider = Provider(event.value)
            except ValueError:
                pass

    def _save_model(self) -> None:
        """Save the new model configuration."""
        model_input = self.query_one("#inp-model", Input)
        name_input = self.query_one("#inp-name", Input)
        api_base_input = self.query_one("#inp-api-base", Input)
        api_key_input = self.query_one("#inp-api-key", Input)

        model = model_input.value.strip()
        if not model:
            self.notify("Model name is required", severity="error")
            return

        # Generate name if not provided
        name = name_input.value.strip()
        if not name:
            provider_name = PROVIDER_NAMES.get(self._selected_provider, "")
            model_short = model.split("/")[-1]
            name = f"{provider_name} {model_short}"

        config = ModelConfig(
            name=name,
            provider=self._selected_provider,
            model=model,
            api_base=api_base_input.value.strip() or None,
            api_key=api_key_input.value.strip() or None,
        )

        self._manager.add(config)
        self.dismiss(self.ModelAdded(config))

    def action_cancel(self) -> None:
        self.dismiss(None)
