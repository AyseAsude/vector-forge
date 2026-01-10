"""Target model card widget for HuggingFace model selection.

Follows the same design pattern as model_card.py for consistency.
"""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static
from textual.message import Message

from vector_forge.storage.models import HFModelConfig
from vector_forge.ui.theme import ICONS
from vector_forge.ui.widgets.model_card import DeleteButton


class TargetModelCard(Static):
    """Clickable card showing a selected HuggingFace model.

    Used in CreateTaskScreen to display and select the target model.
    Matches ModelCard design pattern exactly.
    """

    DEFAULT_CSS = """
    TargetModelCard {
        height: auto;
        width: 1fr;
        padding: 1 2;
        background: $surface;
        margin-right: 1;
    }

    TargetModelCard:last-child {
        margin-right: 0;
    }

    TargetModelCard:hover {
        background: $boost;
    }

    TargetModelCard:focus {
        background: $boost;
    }

    TargetModelCard .header-row {
        height: 1;
        margin-bottom: 1;
    }

    TargetModelCard .label {
        width: 1fr;
        color: $accent;
        text-style: bold;
    }

    TargetModelCard .model-row {
        height: 1;
    }

    TargetModelCard .model-name {
        width: 1fr;
    }

    TargetModelCard .provider-badge {
        width: auto;
    }

    TargetModelCard .model-id {
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
        config: HFModelConfig | None = None,
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

    def set_config(self, config: HFModelConfig | None) -> None:
        """Update the displayed model configuration."""
        self._config = config
        if self.is_mounted:
            self._update_display()

    @property
    def config(self) -> HFModelConfig | None:
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

        # Model name with status icon
        self.query_one(".model-name", Static).update(
            f"[$success]{ICONS.complete}[/] [bold]{config.name}[/]"
        )

        # Provider badge - show HF Hub or Local
        if config.is_local_path:
            self.query_one(".provider-badge", Static).update("[$secondary]Local[/]")
        else:
            self.query_one(".provider-badge", Static).update("[$accent]HuggingFace[/]")

        # Model ID
        self.query_one(".model-id", Static).update(
            f"[$foreground-disabled]{config.model_id}[/]"
        )


class TargetModelCardCompact(Static):
    """Compact target model card for selection lists.

    Matches ModelCardCompact design pattern exactly.
    """

    DEFAULT_CSS = """
    TargetModelCardCompact {
        height: auto;
        padding: 1 2;
        background: $surface;
    }

    TargetModelCardCompact:hover {
        background: $boost;
    }

    TargetModelCardCompact:focus {
        background: $boost;
    }

    TargetModelCardCompact.-selected {
        background: $primary 20%;
    }

    TargetModelCardCompact.-selected:hover {
        background: $primary 30%;
    }

    TargetModelCardCompact.-selected:focus {
        background: $primary 30%;
    }

    TargetModelCardCompact .header-row {
        height: 1;
    }

    TargetModelCardCompact .icon {
        width: 2;
    }

    TargetModelCardCompact .name {
        width: 1fr;
    }

    TargetModelCardCompact .provider {
        width: auto;
    }

    TargetModelCardCompact .detail-row {
        height: 1;
        width: 100%;
    }

    TargetModelCardCompact .model-id {
        width: 1fr;
        padding-left: 2;
        color: $foreground-muted;
    }

    TargetModelCardCompact DeleteButton {
        dock: right;
    }
    """

    can_focus = True

    class Selected(Message):
        """Emitted when this card is selected."""

        def __init__(self, config: HFModelConfig) -> None:
            super().__init__()
            self.config = config

    class DeleteRequested(Message):
        """Emitted when delete is requested for this card."""

        def __init__(self, config: HFModelConfig) -> None:
            super().__init__()
            self.config = config

    def __init__(
        self,
        config: HFModelConfig,
        is_selected: bool = False,
        **kwargs,
    ) -> None:
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
            yield DeleteButton()

    def on_mount(self) -> None:
        self._update_display()

    def on_delete_button_clicked(self, event: DeleteButton.Clicked) -> None:
        """Handle delete button click."""
        event.stop()
        self.post_message(self.DeleteRequested(self._config))

    def on_click(self, event) -> None:
        self.post_message(self.Selected(self._config))

    def on_key(self, event) -> None:
        if event.key in ("enter", "space"):
            self.post_message(self.Selected(self._config))
        elif event.key == "delete":
            self.post_message(self.DeleteRequested(self._config))

    def set_selected(self, selected: bool) -> None:
        self._is_selected = selected
        self.set_class(selected, "-selected")
        if self.is_mounted:
            self._update_display()

    @property
    def config(self) -> HFModelConfig:
        return self._config

    def _update_display(self) -> None:
        config = self._config

        # Selection icon
        if self._is_selected:
            icon = f"[$accent]{ICONS.complete}[/]"
        else:
            icon = f"[$foreground-disabled]{ICONS.pending}[/]"
        self.query_one(".icon", Static).update(icon)

        # Name
        self.query_one(".name", Static).update(f"[bold]{config.name}[/]")

        # Provider badge
        if config.is_local_path:
            self.query_one(".provider", Static).update("[$secondary]Local[/]")
        else:
            self.query_one(".provider", Static).update("[$accent]HuggingFace[/]")

        # Model ID
        self.query_one(".model-id", Static).update(
            f"[$foreground-disabled]{config.model_id}[/]"
        )
