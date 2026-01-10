"""Model selection card widget.

Displays a selected model configuration with click-to-change functionality.
"""

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static
from textual.message import Message

from vector_forge.storage.models import ModelConfig, Provider
from vector_forge.ui.theme import COLORS


# Provider display info - follows theme semantic colors
PROVIDER_INFO = {
    Provider.OPENAI: ("OpenAI", COLORS.success),      # green
    Provider.ANTHROPIC: ("Anthropic", COLORS.accent), # gold
    Provider.OPENROUTER: ("OpenRouter", COLORS.purple),
    Provider.AZURE: ("Azure", COLORS.blue),
    Provider.OLLAMA: ("Ollama", COLORS.aqua),
    Provider.CUSTOM: ("Custom", COLORS.text_muted),
}


class ModelCard(Static):
    """Clickable card showing a selected model configuration.

    Displays the model name, provider, and model ID. Clicking opens
    a model selection modal.

    Example:
        >>> card = ModelCard(
        ...     field_name="extractor",
        ...     label="EXTRACTOR MODEL",
        ...     config=model_config,
        ... )
    """

    DEFAULT_CSS = """
    ModelCard {
        height: 5;
        width: 1fr;
        padding: 1 1;
        background: $surface;
        margin-right: 1;
    }

    ModelCard:last-child {
        margin-right: 0;
    }

    ModelCard:hover {
        background: $surface-hl;
    }

    ModelCard:focus {
        background: $surface-hl;
    }

    ModelCard .label {
        height: 1;
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
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
        color: $text-muted;
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
                f"[{COLORS.text_muted}]No model selected[/]"
            )
            self.query_one(".provider-badge", Static).update("")
            self.query_one(".model-id", Static).update(
                f"[{COLORS.text_dim}]Click to select...[/]"
            )
            return

        config = self._config
        provider_name, provider_color = PROVIDER_INFO.get(
            config.provider, ("Unknown", COLORS.text_muted)
        )

        # Model name
        self.query_one(".model-name", Static).update(
            f"[bold]{config.name}[/]"
        )

        # Provider badge
        self.query_one(".provider-badge", Static).update(
            f"[{provider_color}]{provider_name}[/]"
        )

        # Model ID
        self.query_one(".model-id", Static).update(
            f"[{COLORS.text_dim}]{config.model}[/]"
        )


class ModelCardCompact(Static):
    """Compact model card for selection lists.

    Used in the model selection modal to display selectable options.
    """

    DEFAULT_CSS = """
    ModelCardCompact {
        height: 3;
        padding: 0 1;
        background: $surface;
        margin-bottom: 1;
    }

    ModelCardCompact:last-child {
        margin-bottom: 0;
    }

    ModelCardCompact:hover {
        background: $surface-hl;
    }

    ModelCardCompact:focus {
        background: $surface-hl;
    }

    ModelCardCompact.-selected {
        background: $primary 20%;
    }

    ModelCardCompact .top-row {
        height: 1;
        margin-bottom: 0;
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

    ModelCardCompact .model-id {
        height: 1;
        padding-left: 2;
        color: $text-muted;
    }
    """

    can_focus = True

    class Selected(Message):
        """Emitted when this card is selected."""

        def __init__(self, config: ModelConfig) -> None:
            super().__init__()
            self.config = config

    def __init__(
        self,
        config: ModelConfig,
        is_selected: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._config = config
        self._is_selected = is_selected

    def compose(self) -> ComposeResult:
        with Horizontal(classes="top-row"):
            yield Static(classes="icon")
            yield Static(classes="name")
            yield Static(classes="provider")
        yield Static(classes="model-id")

    def on_mount(self) -> None:
        self._update_display()
        self.set_class(self._is_selected, "-selected")

    def on_click(self) -> None:
        self.post_message(self.Selected(self._config))

    def on_key(self, event) -> None:
        if event.key in ("enter", "space"):
            self.post_message(self.Selected(self._config))

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
        provider_name, provider_color = PROVIDER_INFO.get(
            config.provider, ("Unknown", COLORS.text_muted)
        )

        # Selection icon
        icon = "●" if self._is_selected else "○"
        icon_color = COLORS.accent if self._is_selected else COLORS.text_dim
        self.query_one(".icon", Static).update(f"[{icon_color}]{icon}[/]")

        # Name
        self.query_one(".name", Static).update(f"[bold]{config.name}[/]")

        # Provider
        self.query_one(".provider", Static).update(
            f"[{provider_color}]{provider_name}[/]"
        )

        # Model ID
        self.query_one(".model-id", Static).update(
            f"[{COLORS.text_dim}]{config.model}[/]"
        )
