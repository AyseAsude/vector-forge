"""Add model screen for creating new LLM configurations."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.content import Content
from textual.widgets import Static, Button, Input, Select
from textual.message import Message

from vector_forge.storage.models import (
    MaxPrice,
    ModelConfig,
    ModelConfigManager,
    Provider,
    ProviderPreferences,
)


class ToggleRow(Static):
    """Clickable toggle row matching form-row style."""

    DEFAULT_CSS = """
    ToggleRow {
        height: 1;
        margin-bottom: 1;
    }

    ToggleRow:hover {
        background: $boost;
    }
    """

    class Changed(Message):
        """Emitted when toggle value changes."""

        def __init__(self, toggle_id: str, value: bool) -> None:
            super().__init__()
            self.toggle_id = toggle_id
            self.value = value

    def __init__(self, label: str, value: bool = False, toggle_id: str = "", **kwargs) -> None:
        self._label = label
        self._value = value
        self._toggle_id = toggle_id
        super().__init__(**kwargs)

    def render(self) -> Content:
        """Render the toggle row content."""
        indicator = "[bold $success]yes[/]" if self._value else "[$foreground-muted]no[/]"
        return Content.from_markup(f"[$foreground-muted]{self._label}[/]  {indicator}")

    def on_click(self) -> None:
        """Toggle value on click."""
        self._value = not self._value
        self.refresh()
        self.post_message(self.Changed(self._toggle_id, self._value))

    @property
    def value(self) -> bool:
        """Get current toggle value."""
        return self._value


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

    /* Form content - scrollable, takes remaining space */
    AddModelScreen #form-content {
        height: 1fr;
        min-height: 4;
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

    /* OpenRouter section */
    AddModelScreen #openrouter-section {
        height: auto;
        margin-top: 1;
        padding-top: 1;
        border-top: solid $surface-lighten-1;
    }

    AddModelScreen .section-header {
        height: 1;
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
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

                # OpenRouter-specific section (hidden by default)
                with Vertical(id="openrouter-section"):
                    yield Static("ROUTING", classes="section-header")

                    # Sort strategy
                    with Horizontal(classes="form-row"):
                        yield Static("Sort by", classes="form-label")
                        yield Select(
                            [
                                ("Default", ""),
                                ("Price (lowest)", "price"),
                                ("Throughput (fastest)", "throughput"),
                                ("Latency (lowest)", "latency"),
                            ],
                            value="",
                            allow_blank=False,
                            id="sel-sort",
                        )

                    # Max price prompt
                    with Horizontal(classes="form-row"):
                        yield Static("Max $/M", classes="form-label")
                        yield Input(placeholder="prompt (input)", id="inp-max-price-prompt")

                    # Max price completion
                    with Horizontal(classes="form-row"):
                        yield Static("", classes="form-label")
                        yield Input(placeholder="completion (output)", id="inp-max-price-completion")

                    # Preferred providers
                    with Horizontal(classes="form-row"):
                        yield Static("Prefer", classes="form-label")
                        yield Input(placeholder="anthropic, openai", id="inp-provider-order")

                    # Ignored providers
                    with Horizontal(classes="form-row"):
                        yield Static("Ignore", classes="form-label")
                        yield Input(placeholder="together, fireworks", id="inp-provider-ignore")

                    # Toggle rows (off by default)
                    yield ToggleRow("Allow fallbacks", False, "allow-fallbacks", id="tog-allow-fallbacks")
                    yield ToggleRow("Allow data collection", False, "data-collection", id="tog-data-collection")

            # Buttons - always visible outside scroll
            with Horizontal(id="buttons"):
                yield Static("", id="spacer")
                yield Button("Cancel", id="btn-cancel")
                yield Button("Save Model", id="btn-save")

    def on_mount(self) -> None:
        self.query_one("#close-hint", Static).update("[$foreground-disabled]ESC[/]")
        # Hide OpenRouter section by default (Anthropic is selected)
        self._update_openrouter_visibility()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.dismiss(None)
        elif event.button.id == "btn-save":
            self._save_model()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "sel-provider":
            try:
                self._selected_provider = Provider(event.value)
                self._update_openrouter_visibility()
            except ValueError:
                pass

    def _update_openrouter_visibility(self) -> None:
        """Show/hide OpenRouter section based on selected provider."""
        section = self.query_one("#openrouter-section", Vertical)
        section.display = self._selected_provider == Provider.OPENROUTER

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

        # Build provider preferences for OpenRouter
        provider_preferences = None
        if self._selected_provider == Provider.OPENROUTER:
            provider_preferences = self._build_provider_preferences()

        config = ModelConfig(
            name=name,
            provider=self._selected_provider,
            model=model,
            api_base=api_base_input.value.strip() or None,
            api_key=api_key_input.value.strip() or None,
            provider_preferences=provider_preferences,
        )

        self._manager.add(config)
        self.dismiss(self.ModelAdded(config))

    def _build_provider_preferences(self) -> ProviderPreferences | None:
        """Build ProviderPreferences from OpenRouter form fields."""
        sort_select = self.query_one("#sel-sort", Select)
        max_price_prompt = self.query_one("#inp-max-price-prompt", Input)
        max_price_completion = self.query_one("#inp-max-price-completion", Input)
        provider_order = self.query_one("#inp-provider-order", Input)
        provider_ignore = self.query_one("#inp-provider-ignore", Input)
        allow_fallbacks = self.query_one("#tog-allow-fallbacks", ToggleRow)
        data_collection = self.query_one("#tog-data-collection", ToggleRow)

        # Parse max price
        max_price = None
        prompt_price = self._parse_float(max_price_prompt.value)
        completion_price = self._parse_float(max_price_completion.value)
        if prompt_price is not None or completion_price is not None:
            max_price = MaxPrice(prompt=prompt_price, completion=completion_price)

        # Parse provider lists
        order = self._parse_provider_list(provider_order.value)
        ignore = self._parse_provider_list(provider_ignore.value)

        # Get sort value
        sort_value = sort_select.value if sort_select.value else None

        # Check if any non-default values are set
        has_custom_settings = (
            sort_value
            or max_price
            or order
            or ignore
            or not allow_fallbacks.value
            or not data_collection.value
        )

        if not has_custom_settings:
            return None

        return ProviderPreferences(
            order=order or None,
            allow_fallbacks=allow_fallbacks.value,
            ignore=ignore or None,
            sort=sort_value,
            data_collection="allow" if data_collection.value else "deny",
            max_price=max_price,
        )

    @staticmethod
    def _parse_float(value: str) -> float | None:
        """Parse a float from string, returning None if invalid."""
        value = value.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _parse_provider_list(value: str) -> list[str] | None:
        """Parse comma-separated provider list."""
        value = value.strip()
        if not value:
            return None
        providers = [p.strip().lower() for p in value.split(",") if p.strip()]
        return providers if providers else None

    def action_cancel(self) -> None:
        self.dismiss(None)
