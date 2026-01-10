"""Model selection screen for choosing LLM configurations."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static, Button
from textual.message import Message

from vector_forge.storage.models import (
    ModelConfig,
    ModelConfigManager,
)
from vector_forge.ui.widgets.model_card import ModelCardCompact
from vector_forge.ui.screens.add_model import AddModelScreen


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
        width: 60;
        height: auto;
        max-height: 85%;
        background: $surface;
        padding: 1 2;
    }

    /* Header */
    ModelSelectorScreen #header {
        height: 1;
        margin-bottom: 1;
    }

    ModelSelectorScreen #title {
        width: 1fr;
        text-style: bold;
    }

    ModelSelectorScreen #close-hint {
        width: auto;
        color: $foreground-disabled;
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

    ModelSelectorScreen .empty-list {
        height: 2;
        content-align: center middle;
        color: $foreground-muted;
    }

    /* Buttons - taller for better visibility */
    ModelSelectorScreen #buttons {
        height: 3;
    }

    ModelSelectorScreen #btn-add {
        width: 1fr;
        height: 3;
        background: $accent;
        color: $background;
        border: none;
        text-style: bold;
    }

    ModelSelectorScreen #btn-add:hover {
        background: $accent 80%;
    }

    ModelSelectorScreen #btn-add:focus {
        background: $accent;
        text-style: bold;
    }

    ModelSelectorScreen #btn-cancel {
        width: auto;
        min-width: 10;
        height: 3;
        background: $boost;
        color: $foreground;
        border: none;
        margin-left: 1;
    }

    ModelSelectorScreen #btn-cancel:hover {
        background: $primary 20%;
    }

    ModelSelectorScreen #btn-cancel:focus {
        background: $boost;
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

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            # Header
            with Horizontal(id="header"):
                field_display = self.field_name.upper()
                yield Static(f"SELECT {field_display} MODEL", id="title")
                yield Static("", id="close-hint")

            # Saved models section
            yield Static("SAVED MODELS", classes="section-title")
            yield VerticalScroll(id="models-scroll")

            # Buttons - plain text, no Rich markup
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

    def on_model_card_compact_delete_requested(self, event: ModelCardCompact.DeleteRequested) -> None:
        """Handle model deletion request."""
        if event.config.is_builtin:
            self.notify("Cannot delete built-in models", severity="warning")
            return

        self._manager.remove(event.config.id)
        self.notify(f"Deleted {event.config.name}", severity="information")
        self._populate_models()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.dismiss(None)
        elif event.button.id == "btn-add":
            self.app.push_screen(AddModelScreen(), callback=self._on_model_added)

    def _on_model_added(self, result: AddModelScreen.ModelAdded | None) -> None:
        """Handle model added from AddModelScreen."""
        if result is not None:
            # Reload from disk and refresh the list
            self._manager.reload()
            self._populate_models()

    def action_cancel(self) -> None:
        self.dismiss(None)
