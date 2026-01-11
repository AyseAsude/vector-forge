"""Model selection card widget.

Displays a selected model configuration with click-to-change functionality.
"""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static
from textual.message import Message

from vector_forge.storage.models import ModelConfig, Provider
from vector_forge.ui.theme import ICONS


# Provider display info: (name, theme_color_variable)
PROVIDER_STYLES = {
    Provider.OPENAI: ("OpenAI", "$success"),
    Provider.ANTHROPIC: ("Anthropic", "$accent"),
    Provider.OPENROUTER: ("OpenRouter", "$secondary"),
    Provider.AZURE: ("Azure", "$primary"),
    Provider.OLLAMA: ("Ollama", "$secondary"),
    Provider.CUSTOM: ("Custom", "$foreground-muted"),
}


class ModelCard(Static):
    """Clickable card showing a selected model configuration.

    Used in CreateTaskScreen to display and select generator/judge/expander models.
    Click to open model selection modal.
    """

    DEFAULT_CSS = """
    ModelCard {
        height: auto;
        width: 1fr;
        padding: 1 2;
        background: $surface;
        margin-right: 1;
    }

    ModelCard:last-child {
        margin-right: 0;
    }

    ModelCard:hover {
        background: $boost;
    }

    ModelCard:focus {
        background: $boost;
    }

    ModelCard .header-row {
        height: 1;
        margin-bottom: 1;
    }

    ModelCard .label {
        width: 1fr;
        color: $accent;
        text-style: bold;
    }

    ModelCard .model-row {
        height: 1;
    }

    ModelCard .model-name {
        width: 1fr;
    }

    ModelCard .provider-badge {
        width: auto;
    }

    ModelCard .model-id {
        height: 1;
        color: $foreground-muted;
        margin-top: 1;
    }
    """

    can_focus = True

    class Clicked(Message):
        """Emitted when the card is clicked."""

        def __init__(self, field_name: str) -> None:
            super().__init__()
            self.field_name = field_name

    def __init__(
        self,
        field_name: str,
        label: str,
        config: ModelConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.field_name = field_name
        self._label = label
        self._config = config

    def compose(self) -> ComposeResult:
        with Horizontal(classes="header-row"):
            yield Static(self._label, classes="label")
        with Horizontal(classes="model-row"):
            yield Static(classes="model-name")
            yield Static(classes="provider-badge")
        yield Static(classes="model-id")

    def on_mount(self) -> None:
        self._update_display()

    def on_click(self) -> None:
        self.post_message(self.Clicked(self.field_name))

    def on_key(self, event) -> None:
        if event.key in ("enter", "space"):
            self.post_message(self.Clicked(self.field_name))

    def set_config(self, config: ModelConfig | None) -> None:
        """Update the displayed model configuration."""
        self._config = config
        if self.is_mounted:
            self._update_display()

    @property
    def config(self) -> ModelConfig | None:
        """Get the current model configuration."""
        return self._config

    def _update_display(self) -> None:
        """Update the display with current config."""
        if self._config is None:
            self.query_one(".model-name", Static).update(
                f"[$foreground-muted]{ICONS.pending} No model selected[/]"
            )
            self.query_one(".provider-badge", Static).update("")
            self.query_one(".model-id", Static).update("")
            return

        config = self._config
        provider_name, provider_color = PROVIDER_STYLES.get(
            config.provider, ("Unknown", "$foreground-muted")
        )

        # Model name with status icon
        self.query_one(".model-name", Static).update(
            f"[$success]{ICONS.complete}[/] [bold]{config.name}[/]"
        )

        # Provider badge
        self.query_one(".provider-badge", Static).update(
            f"[{provider_color}]{provider_name}[/]"
        )

        # Model ID
        self.query_one(".model-id", Static).update(
            f"[$foreground-disabled]{config.model}[/]"
        )


class DeleteButton(Static):
    """Clickable delete button for model cards."""

    DEFAULT_CSS = """
    DeleteButton {
        width: auto;
        height: 1;
    }

    DeleteButton:hover {
        background: $error 30%;
    }
    """

    class Clicked(Message):
        """Emitted when delete button is clicked."""
        pass

    def __init__(self, **kwargs) -> None:
        super().__init__("[$error]\\[x][/]", **kwargs)

    def on_click(self, event) -> None:
        event.stop()
        self.post_message(self.Clicked())


class ModelCardCompact(Static):
    """Compact model card for selection lists.

    Used in the model selection modal to display selectable options.
    Custom models show [x] delete button on bottom right.
    """

    DEFAULT_CSS = """
    ModelCardCompact {
        height: auto;
        padding: 1 2;
        margin-right: 2;
        background: $surface;
    }

    ModelCardCompact:hover {
        background: $boost;
    }

    ModelCardCompact:focus {
        background: $boost;
    }

    ModelCardCompact.-selected {
        background: $primary 20%;
    }

    ModelCardCompact.-selected:hover {
        background: $primary 30%;
    }

    ModelCardCompact.-selected:focus {
        background: $primary 30%;
    }

    ModelCardCompact .header-row {
        height: 1;
    }

    ModelCardCompact .icon {
        width: 2;
    }

    ModelCardCompact .name {
        width: 1fr;
    }

    ModelCardCompact .provider {
        width: auto;
    }

    ModelCardCompact .detail-row {
        height: 1;
        width: 100%;
    }

    ModelCardCompact .model-id {
        width: 1fr;
        padding-left: 2;
        color: $foreground-muted;
    }

    ModelCardCompact DeleteButton {
        dock: right;
    }
    """

    can_focus = True

    class Selected(Message):
        """Emitted when this card is selected."""

        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            self.config = config

    class DeleteRequested(Message):
        """Emitted when delete is requested for this card."""

        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            self.config = config

    def __init__(
        self,
        config: ModelConfig,
        is_selected: bool = False,
        **kwargs,
    ) -> None:
        # Merge -selected class into classes parameter if selected
        if is_selected:
            existing_classes = kwargs.get("classes", "")
            kwargs["classes"] = f"{existing_classes} -selected".strip()
        super().__init__(**kwargs)
        self._config = config
        self._is_selected = is_selected

    def compose(self) -> ComposeResult:
        with Horizontal(classes="header-row"):
            yield Static(classes="icon")
            yield Static(classes="name")
            yield Static(classes="provider")
        with Horizontal(classes="detail-row"):
            yield Static(classes="model-id")
            # Only show delete for non-builtin models
            if not self._config.is_builtin:
                yield DeleteButton()

    def on_mount(self) -> None:
        self._update_display()

    def on_delete_button_clicked(self, event: DeleteButton.Clicked) -> None:
        """Handle delete button click."""
        event.stop()
        self.post_message(self.DeleteRequested(self._config))

    def on_click(self, event) -> None:
        # Left-click selects (delete button handles its own clicks)
        self.post_message(self.Selected(self._config))

    def on_key(self, event) -> None:
        if event.key in ("enter", "space"):
            self.post_message(self.Selected(self._config))
        elif event.key == "delete" and not self._config.is_builtin:
            self.post_message(self.DeleteRequested(self._config))

    def set_selected(self, selected: bool) -> None:
        """Set selection state."""
        self._is_selected = selected
        self.set_class(selected, "-selected")
        if self.is_mounted:
            self._update_display()

    @property
    def config(self) -> ModelConfig:
        """Get the model configuration."""
        return self._config

    def _update_display(self) -> None:
        """Update the display."""
        config = self._config
        provider_name, provider_color = PROVIDER_STYLES.get(
            config.provider, ("Unknown", "$foreground-muted")
        )

        # Selection icon
        if self._is_selected:
            icon = f"[$accent]{ICONS.complete}[/]"
        else:
            icon = f"[$foreground-disabled]{ICONS.pending}[/]"
        self.query_one(".icon", Static).update(icon)

        # Name
        self.query_one(".name", Static).update(f"[bold]{config.name}[/]")

        # Provider
        self.query_one(".provider", Static).update(
            f"[{provider_color}]{provider_name}[/]"
        )

        # Model ID
        self.query_one(".model-id", Static).update(
            f"[$foreground-disabled]{config.model}[/]"
        )
