"""Target model selection screen for HuggingFace models.

Follows the same design pattern as model_selector.py and add_model.py.
"""

from typing import Optional

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static, Button, Input
from textual.message import Message

from vector_forge.storage.models import HFModelConfig, HFModelConfigManager
from vector_forge.ui.widgets.target_model_card import TargetModelCardCompact
from vector_forge.ui.theme import ICONS


class AddTargetModelScreen(ModalScreen):
    """Modal screen for adding a new HuggingFace model configuration.

    Matches AddModelScreen design pattern exactly.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    DEFAULT_CSS = """
    AddTargetModelScreen {
        align: center middle;
    }

    AddTargetModelScreen #dialog {
        width: 60;
        height: auto;
        max-height: 90%;
        background: $surface;
        padding: 1 2;
    }

    /* Header */
    AddTargetModelScreen #header {
        height: 1;
        margin-bottom: 1;
    }

    AddTargetModelScreen #title {
        width: 1fr;
        text-style: bold;
    }

    AddTargetModelScreen #close-hint {
        width: auto;
        color: $foreground-disabled;
    }

    /* Form content - scrollable, takes remaining space */
    AddTargetModelScreen #form-content {
        height: 1fr;
        min-height: 4;
    }

    /* Form rows */
    AddTargetModelScreen .form-row {
        height: 1;
        margin-bottom: 1;
    }

    AddTargetModelScreen .form-label {
        width: 12;
        color: $foreground-muted;
    }

    /* Input styling */
    AddTargetModelScreen Input {
        width: 1fr;
        height: 1;
        background: $boost;
        border: none;
        padding: 0 1;
    }

    AddTargetModelScreen Input:focus {
        background: $background;
    }

    /* Status row */
    AddTargetModelScreen #status-row {
        height: 1;
        margin-bottom: 1;
    }

    AddTargetModelScreen #status {
        width: 1fr;
    }

    /* Buttons */
    AddTargetModelScreen #buttons {
        height: 3;
        margin-top: 1;
    }

    AddTargetModelScreen #spacer {
        width: 1fr;
    }

    AddTargetModelScreen #btn-cancel {
        width: auto;
        min-width: 10;
        height: 3;
        background: $boost;
        color: $foreground;
        border: none;
        margin-right: 1;
    }

    AddTargetModelScreen #btn-cancel:hover {
        background: $primary 20%;
    }

    AddTargetModelScreen #btn-cancel:focus {
        background: $boost;
    }

    AddTargetModelScreen #btn-save {
        width: auto;
        min-width: 12;
        height: 3;
        background: $success;
        color: $background;
        border: none;
        text-style: bold;
    }

    AddTargetModelScreen #btn-save:hover {
        background: $success 80%;
    }

    AddTargetModelScreen #btn-save:focus {
        background: $success;
        text-style: bold;
    }
    """

    class ModelAdded(Message):
        """Emitted when a model is added."""

        def __init__(self, config: HFModelConfig) -> None:
            super().__init__()
            self.config = config

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._probed_config: Optional[HFModelConfig] = None
        self._probing = False

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            # Header
            with Horizontal(id="header"):
                yield Static("ADD NEW MODEL", id="title")
                yield Static("", id="close-hint")

            # Form content - scrollable
            with VerticalScroll(id="form-content"):
                # Model ID input
                with Horizontal(classes="form-row"):
                    yield Static("Model", classes="form-label")
                    yield Input(
                        placeholder="meta-llama/Llama-3.1-8B-Instruct",
                        id="inp-model",
                    )

                # Display name
                with Horizontal(classes="form-row"):
                    yield Static("Name", classes="form-label")
                    yield Input(placeholder="Display name (optional)", id="inp-name")

                # Status row (shows probe results)
                with Horizontal(id="status-row"):
                    yield Static("Status", classes="form-label")
                    yield Static("[$foreground-muted]Enter model ID and save[/]", id="status")

            # Buttons - outside scroll
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

    def _status(self, msg: str, level: str = "info") -> None:
        colors = {"error": "$error", "success": "$success", "warning": "$accent"}
        color = colors.get(level, "$foreground-muted")
        self.query_one("#status", Static).update(f"[{color}]{msg}[/]")

    def _save_model(self) -> None:
        """Save the model - probes first if needed."""
        model_id = self.query_one("#inp-model", Input).value.strip()
        if not model_id:
            self._status("Model ID is required", "error")
            return

        if self._probing:
            return

        # If we have a probed config for this model, use it
        if self._probed_config and self._probed_config.model_id == model_id:
            self.dismiss(self.ModelAdded(self._probed_config))
            return

        # Otherwise probe first
        self._probing = True
        self._status(f"{ICONS.thinking} Probing model...", "warning")

        name = self.query_one("#inp-name", Input).value.strip()
        self._run_probe(model_id, name)

    @work(thread=True, exclusive=True)
    def _run_probe(self, model_id: str, name: str) -> None:
        """Background worker to probe model config."""
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(model_id)

            num_layers = getattr(config, "num_hidden_layers", None)
            hidden_dim = getattr(config, "hidden_size", None)
            vocab_size = getattr(config, "vocab_size", None)
            model_type = getattr(config, "model_type", None)

            # Handle nested config (multimodal models)
            if num_layers is None and hasattr(config, "text_config"):
                text_config = config.text_config
                num_layers = getattr(text_config, "num_hidden_layers", None)
                hidden_dim = getattr(text_config, "hidden_size", None)

            # Estimate param count
            param_count = None
            if hidden_dim and num_layers:
                params = hidden_dim * hidden_dim * num_layers * 12
                if params > 1e9:
                    param_count = f"~{params / 1e9:.1f}B"
                else:
                    param_count = f"~{params / 1e6:.0f}M"

            # Generate name if not provided
            if not name:
                model_short = model_id.split("/")[-1]
                name = f"HuggingFace {model_short}"

            hf_config = HFModelConfig(
                name=name,
                model_id=model_id,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                model_type=model_type,
                param_count=param_count,
            )

            self.call_from_thread(self._probe_success, hf_config)

        except Exception as e:
            self.call_from_thread(self._probe_error, str(e))

    def _probe_success(self, config: HFModelConfig) -> None:
        self._probing = False
        self._probed_config = config

        # Show info and auto-save
        info = f"{ICONS.success} {config.model_type or 'Model'}"
        if config.num_layers:
            info += f" | {config.num_layers}L"
        if config.param_count:
            info += f" | {config.param_count}"
        self._status(info, "success")

        # Auto-dismiss with the config
        self.dismiss(self.ModelAdded(config))

    def _probe_error(self, error: str) -> None:
        self._probing = False
        self._probed_config = None
        error_short = error[:50] + "..." if len(error) > 50 else error
        self._status(f"{ICONS.error} {error_short}", "error")

    def action_cancel(self) -> None:
        self.dismiss(None)


class TargetModelSelectorScreen(ModalScreen):
    """Modal screen for selecting or creating HuggingFace model configurations.

    Matches ModelSelectorScreen design pattern.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    DEFAULT_CSS = """
    TargetModelSelectorScreen {
        align: center middle;
    }

    TargetModelSelectorScreen #dialog {
        width: 60;
        height: auto;
        max-height: 85%;
        background: $surface;
        padding: 1 2;
    }

    /* Header */
    TargetModelSelectorScreen #header {
        height: 1;
        margin-bottom: 1;
    }

    TargetModelSelectorScreen #title {
        width: 1fr;
        text-style: bold;
    }

    TargetModelSelectorScreen #close-hint {
        width: auto;
        color: $foreground-disabled;
    }

    /* Section titles */
    TargetModelSelectorScreen .section-title {
        height: 1;
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }

    /* Models list - scrollable, takes remaining space */
    TargetModelSelectorScreen #models-scroll {
        height: 1fr;
        min-height: 4;
        margin-bottom: 1;
    }

    TargetModelSelectorScreen .empty-list {
        height: 2;
        content-align: center middle;
        color: $foreground-muted;
        margin-right: 2;
    }

    /* Buttons */
    TargetModelSelectorScreen #buttons {
        height: 3;
    }

    TargetModelSelectorScreen #btn-add {
        width: 1fr;
        height: 3;
        background: $accent;
        color: $background;
        border: none;
        text-style: bold;
    }

    TargetModelSelectorScreen #btn-add:hover {
        background: $accent 80%;
    }

    TargetModelSelectorScreen #btn-add:focus {
        background: $accent;
        text-style: bold;
    }

    TargetModelSelectorScreen #btn-cancel {
        width: auto;
        min-width: 10;
        height: 3;
        background: $boost;
        color: $foreground;
        border: none;
        margin-left: 1;
    }

    TargetModelSelectorScreen #btn-cancel:hover {
        background: $primary 20%;
    }

    TargetModelSelectorScreen #btn-cancel:focus {
        background: $boost;
    }
    """

    class ModelSelected(Message):
        """Emitted when a model is selected."""

        def __init__(self, config: HFModelConfig) -> None:
            super().__init__()
            self.config = config

    def __init__(
        self,
        current_config: HFModelConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.current_config = current_config
        self._manager = HFModelConfigManager()

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            # Header
            with Horizontal(id="header"):
                yield Static("SELECT TARGET MODEL", id="title")
                yield Static("", id="close-hint")

            # Saved models section
            yield Static("SAVED MODELS", classes="section-title")
            yield VerticalScroll(id="models-scroll")

            # Buttons
            with Horizontal(id="buttons"):
                yield Button("+ Add New", id="btn-add")
                yield Button("Cancel", id="btn-cancel")

    def on_mount(self) -> None:
        self.query_one("#close-hint", Static).update("[$foreground-disabled]ESC[/]")
        self._populate_models()

    def _populate_models(self) -> None:
        """Populate the models list."""
        models_scroll = self.query_one("#models-scroll", VerticalScroll)
        models_scroll.remove_children()

        configs = self._manager.list_all()

        if not configs:
            models_scroll.mount(
                Static(
                    "[$foreground-muted]No saved models[/]",
                    classes="empty-list",
                )
            )
            return

        for config in configs:
            is_selected = (
                self.current_config is not None
                and self.current_config.id == config.id
            )
            card = TargetModelCardCompact(config, is_selected=is_selected)
            models_scroll.mount(card)

    def on_target_model_card_compact_selected(
        self, event: TargetModelCardCompact.Selected
    ) -> None:
        """Handle model selection."""
        self._manager.update_last_used(event.config.id)
        self.dismiss(self.ModelSelected(event.config))

    def on_target_model_card_compact_delete_requested(
        self, event: TargetModelCardCompact.DeleteRequested
    ) -> None:
        """Handle model deletion request."""
        self._manager.remove(event.config.id)
        self.notify(f"Deleted {event.config.name}", severity="information")
        self._populate_models()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.dismiss(None)
        elif event.button.id == "btn-add":
            self.app.push_screen(
                AddTargetModelScreen(), callback=self._on_model_added
            )

    def _on_model_added(self, result: AddTargetModelScreen.ModelAdded | None) -> None:
        """Handle model added from AddTargetModelScreen."""
        if result is not None:
            self._manager.add(result.config)
            self._manager.reload()
            self._populate_models()

    def action_cancel(self) -> None:
        self.dismiss(None)
