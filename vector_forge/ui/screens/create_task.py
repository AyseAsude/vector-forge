"""Task creation screen with profile-based configuration."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Static, Input, Button, TextArea
from textual.message import Message

from vector_forge.constants import DEFAULT_MODEL
from vector_forge.tasks.config import (
    TaskConfig,
    LayerStrategy,
    AggregationStrategy,
    ContrastConfig,
    OptimizationConfig,
    EvaluationConfig,
    ExtractionMethod,
    CAAConfig,
    TournamentConfig,
    SignalFilterMode,
    ExtractionIntensity,
)
from vector_forge.storage.models import (
    ModelConfig,
    ModelConfigManager,
    HFModelConfig,
    HFModelConfigManager,
)
from vector_forge.storage.preferences import (
    PreferencesManager,
    ModelRole,
)
from vector_forge.ui.widgets.tmux_bar import TmuxBar
from vector_forge.ui.widgets.model_card import ModelCard
from vector_forge.ui.widgets.target_model_card import TargetModelCard
from vector_forge.ui.screens.model_selector import ModelSelectorScreen
from vector_forge.ui.screens.target_model_selector import TargetModelSelectorScreen


class ProfileCard(Static):
    """Clickable profile card."""

    DEFAULT_CSS = """
    ProfileCard {
        width: 1fr;
        height: 3;
        padding: 0 1;
        background: $surface;
        content-align: center middle;
    }

    ProfileCard:hover {
        background: $boost;
    }

    ProfileCard.-selected {
        background: $primary 20%;
    }

    ProfileCard.-selected:hover {
        background: $primary 30%;
    }
    """

    class Selected(Message):
        def __init__(self, profile: str) -> None:
            super().__init__()
            self.profile = profile

    def __init__(self, profile: str, label: str, desc: str, selected: bool = False, **kwargs) -> None:
        # Merge -selected class into classes parameter if selected
        if selected:
            existing_classes = kwargs.get("classes", "")
            kwargs["classes"] = f"{existing_classes} -selected".strip()
        self._profile = profile
        self._label = label
        self._desc = desc
        self._selected = selected
        # Compute initial content to pass to super().__init__()
        initial_content = self._compute_content()
        super().__init__(initial_content, **kwargs)

    def on_click(self) -> None:
        self.post_message(self.Selected(self._profile))

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.set_class(selected, "-selected")
        self.update(self._compute_content())

    @property
    def profile(self) -> str:
        return self._profile

    def _compute_content(self) -> str:
        """Compute the display content for this profile card."""
        icon = "●" if self._selected else "○"
        color = "$accent" if self._selected else "$foreground-disabled"
        return f"[{color}]{icon}[/] [bold]{self._label}[/] [$foreground-muted]{self._desc}[/]"


class OptionPill(Static):
    """Clickable option pill for enum selections."""

    DEFAULT_CSS = """
    OptionPill {
        width: auto;
        height: 1;
        padding: 0 1;
        margin-right: 1;
        background: $surface;
    }

    OptionPill:hover {
        background: $boost;
    }

    OptionPill.-selected {
        background: $primary 25%;
    }

    OptionPill.-selected:hover {
        background: $primary 35%;
    }
    """

    class Selected(Message):
        def __init__(self, group: str, value: str) -> None:
            super().__init__()
            self.group = group
            self.value = value

    def __init__(self, group: str, value: str, label: str, selected: bool = False, **kwargs) -> None:
        # Merge -selected class into classes parameter if selected
        if selected:
            existing_classes = kwargs.get("classes", "")
            kwargs["classes"] = f"{existing_classes} -selected".strip()
        self._group = group
        self._value = value
        self._label = label
        self._selected = selected
        # Compute initial content to pass to super().__init__()
        initial_content = self._compute_content()
        super().__init__(initial_content, **kwargs)

    def on_click(self) -> None:
        self.post_message(self.Selected(self._group, self._value))

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.set_class(selected, "-selected")
        self.update(self._compute_content())

    @property
    def value(self) -> str:
        return self._value

    @property
    def group(self) -> str:
        return self._group

    def _compute_content(self) -> str:
        """Compute the display content for this option pill."""
        if self._selected:
            return f"[$accent]●[/] {self._label}"
        else:
            return f"[$foreground-disabled]○[/] [$foreground-muted]{self._label}[/]"


class ParamRow(Horizontal):
    """A parameter row with label and input."""

    DEFAULT_CSS = """
    ParamRow {
        height: 1;
        margin-bottom: 1;
    }

    ParamRow .label {
        width: 14;
        color: $foreground-muted;
    }

    ParamRow Input {
        width: 1fr;
        background: $surface;
        border: none;
    }

    ParamRow Input:focus {
        background: $boost;
    }
    """

    def __init__(self, label: str, input_id: str, value: str, hint: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._input_id = input_id
        self._value = value
        self._hint = hint

    def compose(self) -> ComposeResult:
        yield Static(self._label, classes="label")
        yield Input(value=self._value, placeholder=self._hint, id=self._input_id)


class ParamSection(Vertical):
    """A section of parameters with a title."""

    DEFAULT_CSS = """
    ParamSection {
        height: auto;
        width: 1fr;
        padding-right: 2;
    }

    ParamSection:last-child {
        padding-right: 0;
    }

    ParamSection .section-title {
        height: 1;
        color: $accent;
        margin-bottom: 1;
    }
    """

    def __init__(self, title: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._title = title

    def compose(self) -> ComposeResult:
        yield Static(self._title, classes="section-title")


class CreateTaskScreen(Screen):
    """Full-screen form for creating extraction tasks."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+s", "create", "Create"),
        Binding("q", "cancel", "Quit"),
    ]

    DEFAULT_CSS = """
    CreateTaskScreen {
        background: $background;
    }

    /* Header */
    CreateTaskScreen #header {
        height: 3;
        padding: 1 2;
        background: $surface;
    }

    CreateTaskScreen #header-title {
        width: 1fr;
        text-style: bold;
    }

    CreateTaskScreen #cancel-link {
        width: auto;
    }

    /* Content - scrollbar at edge */
    CreateTaskScreen #content {
        height: 1fr;
        padding: 1 0 1 2;
    }

    CreateTaskScreen .main-section {
        height: 1;
        color: $foreground-muted;
        margin-bottom: 1;
        margin-top: 1;
        padding-right: 2;
    }

    /* Profiles */
    CreateTaskScreen #profiles {
        height: 3;
        margin-bottom: 1;
        padding-right: 2;
    }

    /* Models */
    CreateTaskScreen #models-row {
        height: auto;
        margin-bottom: 1;
        padding-right: 2;
    }

    /* Behavior */
    CreateTaskScreen #behavior-box {
        height: auto;
        margin-bottom: 1;
        padding-right: 2;
    }

    CreateTaskScreen #behavior-input {
        height: 8;
        margin-bottom: 1;
        background: $surface;
        border: none;
    }

    CreateTaskScreen #behavior-input:focus {
        background: $surface;
    }

    /* Remove cursor line highlight */
    CreateTaskScreen #behavior-input .text-area--cursor-line {
        background: $surface;
    }

    CreateTaskScreen #behavior-input:focus .text-area--cursor-line {
        background: $surface;
    }

    /* Parameters */
    CreateTaskScreen #params-container {
        height: auto;
        margin-bottom: 1;
        padding-right: 2;
    }

    CreateTaskScreen .params-row {
        height: auto;
        margin-bottom: 1;
    }

    CreateTaskScreen .option-row {
        height: 1;
        margin-bottom: 1;
    }

    CreateTaskScreen .option-label {
        width: 14;
        color: $foreground-muted;
    }

    CreateTaskScreen .option-pills {
        width: 1fr;
    }

    /* Footer */
    CreateTaskScreen #footer {
        height: 3;
        padding: 1 2;
        background: $surface;
    }

    CreateTaskScreen #status {
        width: 1fr;
        height: 1;
        content-align: left middle;
    }

    CreateTaskScreen #btn-cancel {
        width: auto;
        min-width: 10;
        height: 1;
        background: $boost;
        color: $foreground;
        border: none;
        padding: 0 2;
        margin-right: 1;
    }

    CreateTaskScreen #btn-cancel:hover {
        background: $primary 20%;
    }

    CreateTaskScreen #btn-create {
        width: auto;
        min-width: 14;
        height: 1;
        background: $success;
        color: $background;
        border: none;
        padding: 0 2;
        text-style: bold;
    }

    CreateTaskScreen #btn-create:hover {
        background: $success 80%;
    }
    """

    class TaskCreated(Message):
        def __init__(self, config: TaskConfig, description: str) -> None:
            super().__init__()
            self.config = config
            self.description = description

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._profile = "standard"
        self._extraction_method = "caa"  # Default to CAA
        self._layer_strategy = "auto"
        self._aggregation = "top_k_average"
        self._contrast_quality = "standard"
        self._intensity_profile = "balanced"
        self._remove_outliers = True
        self._tournament_enabled = True  # Tournament on by default
        self._elimination_rounds = 2
        self._elimination_rate = 0.75  # Default 75% elimination per round

        # Signal filtering options (NEW)
        self._signal_filter_mode = "off"  # "off", "threshold", "top_k"
        self._extraction_intensity = "all"  # "all", "high_signal", "maximum"

        # Preferences and model managers
        self._preferences = PreferencesManager()
        self._model_manager = ModelConfigManager()
        self._hf_model_manager = HFModelConfigManager()

        # Load API model configs from preferences (or fall back to default)
        self._generator_config = self._load_model_config(ModelRole.GENERATOR)
        self._judge_config = self._load_model_config(ModelRole.JUDGE)
        self._expander_config = self._load_model_config(ModelRole.EXPANDER)  # Used automatically in pipeline

        # Load target model config from preferences (or fall back to default)
        self._target_config = self._load_target_config()

    def compose(self) -> ComposeResult:
        # Header
        with Horizontal(id="header"):
            yield Static("CREATE TASK", id="header-title")
            yield Static("[$foreground-disabled]ESC to cancel[/]", id="cancel-link")

        # Content
        with VerticalScroll(id="content"):
            # Profile section
            yield Static("PROFILE", classes="main-section")
            with Horizontal(id="profiles"):
                yield ProfileCard("quick", "Quick", "32→4 samples", id="profile-quick")
                yield ProfileCard("standard", "Standard", "256→16 samples", selected=True, id="profile-standard")
                yield ProfileCard("comprehensive", "Full", "1024→32 samples", id="profile-comprehensive")

            # Models section
            yield Static("MODELS", classes="main-section")
            with Horizontal(id="models-row"):
                yield TargetModelCard(
                    field_name="target",
                    label="TARGET",
                    config=self._target_config,
                    id="model-target",
                )
                yield ModelCard(
                    field_name="generator",
                    label="GENERATOR",
                    config=self._generator_config,
                    id="model-generator",
                )
                yield ModelCard(
                    field_name="judge",
                    label="JUDGE",
                    config=self._judge_config,
                    id="model-judge",
                )
                yield ModelCard(
                    field_name="expander",
                    label="EXPANDER",
                    config=self._expander_config,
                    id="model-expander",
                )

            # Behavior section
            yield Static("BEHAVIOR", classes="main-section")
            with Vertical(id="behavior-box"):
                yield TextArea(
                    placeholder="Describe the behavior to extract (e.g., 'sycophancy', 'helpfulness')",
                    id="behavior-input"
                )

            # Extraction Method section
            yield Static("EXTRACTION", classes="main-section")
            with Vertical(id="extraction-container"):
                with Horizontal(classes="option-row"):
                    yield Static("Method", classes="option-label")
                    with Horizontal(classes="option-pills"):
                        yield OptionPill("extraction", "caa", "CAA", selected=True, id="ext-caa")
                        yield OptionPill("extraction", "gradient", "Gradient", id="ext-gradient")
                        yield OptionPill("extraction", "hybrid", "Hybrid", id="ext-hybrid")

                # CAA options (outlier removal)
                with Horizontal(classes="option-row", id="outlier-row"):
                    yield Static("Outliers", classes="option-label")
                    with Horizontal(classes="option-pills"):
                        yield OptionPill("outliers", "remove", "Remove", selected=True, id="out-remove")
                        yield OptionPill("outliers", "keep", "Keep All", id="out-keep")

                yield ParamRow("Threshold", "inp-outlier-threshold", "3.0", "std devs", id="outlier-threshold-row")

                # Signal filtering options (NEW)
                yield Static("SIGNAL FILTERING", classes="sub-section")

                with Horizontal(classes="option-row"):
                    yield Static("Signal Filter", classes="option-label")
                    with Horizontal(classes="option-pills"):
                        yield OptionPill("signal_filter", "off", "Off", selected=True, id="sig-off")
                        yield OptionPill("signal_filter", "threshold", "Threshold", id="sig-threshold")
                        yield OptionPill("signal_filter", "top_k", "Top-K", id="sig-topk")

                with Vertical(id="signal-params"):
                    yield ParamRow("Min Signal", "inp-min-signal", "6.0", "score 1-10", id="min-signal-row")
                    yield ParamRow("Top K Pairs", "inp-topk-pairs", "30", "pairs/sample", id="topk-pairs-row")
                    yield ParamRow("Min Confound", "inp-min-confound", "5.0", "score 1-10", id="min-confound-row")

                with Horizontal(classes="option-row"):
                    yield Static("Extraction Int.", classes="option-label")
                    with Horizontal(classes="option-pills"):
                        yield OptionPill("ext_intensity", "all", "All", selected=True, id="ext-all")
                        yield OptionPill("ext_intensity", "high_signal", "High Signal", id="ext-high")
                        yield OptionPill("ext_intensity", "maximum", "Maximum", id="ext-max")

            # Parameters section
            yield Static("PARAMETERS", classes="main-section")
            with Vertical(id="params-container"):
                # Row 1: Sampling & Contrast (+ Optimization for gradient mode)
                with Horizontal(classes="params-row"):
                    with ParamSection("SAMPLING"):
                        yield ParamRow("Samples", "inp-samples", "256")
                        yield ParamRow("Seeds", "inp-seeds", "4")
                        yield ParamRow("Datapoints", "inp-datapoints", "50", "per sample")

                    with ParamSection("CONTRAST"):
                        yield ParamRow("Core Pool", "inp-core-pool", "80", "shared pairs")
                        yield ParamRow("Core/Sample", "inp-core-per-sample", "40")
                        yield ParamRow("Unique/Sample", "inp-unique-per-sample", "10")
                        yield ParamRow("Max Regen", "inp-max-regen", "2", "retry attempts")

                    with ParamSection("OPTIMIZATION", id="optimization-section"):
                        yield ParamRow("Learning Rate", "inp-lr", "0.1")
                        yield ParamRow("Max Iters", "inp-max-iters", "50")
                        yield ParamRow("Coldness", "inp-coldness", "0.7", "softmax temp")
                        yield ParamRow("Max Norm", "inp-max-norm", "2.0", "regularization")

                # Row 2: Validation & Intensity
                with Horizontal(classes="params-row"):
                    with ParamSection("VALIDATION"):
                        yield ParamRow("Min Quality", "inp-min-quality", "6.0", "contrast score")
                        yield ParamRow("Min Dimension", "inp-min-dim", "6.0", "score 0-10")
                        yield ParamRow("Min Structural", "inp-min-struct", "7.0", "score 0-10")
                        yield ParamRow("Min Semantic", "inp-min-semantic", "4.0", "score 0-10")

                    with ParamSection("INTENSITY"):
                        yield ParamRow("Extreme", "inp-intensity-extreme", "0.10", "proportion")
                        yield ParamRow("High", "inp-intensity-high", "0.20", "proportion")
                        yield ParamRow("Medium", "inp-intensity-medium", "0.30", "proportion")
                        yield ParamRow("Natural", "inp-intensity-natural", "0.40", "proportion")

                # Row 3: Parallelism & Evaluation
                with Horizontal(classes="params-row"):
                    with ParamSection("PARALLELISM"):
                        yield ParamRow("Extractions", "inp-extractions", "1")
                        yield ParamRow("Evaluations", "inp-evaluations", "16")
                        yield ParamRow("Generations", "inp-generations", "16", "LLM API calls")
                        yield ParamRow("Top K", "inp-topk", "5")

                    with ParamSection("EVALUATION"):
                        yield ParamRow("Strengths", "inp-strengths", "0.5, 1.0, 1.5, 2.0", "levels to test")
                        yield ParamRow("Temperature", "inp-eval-temp", "0.7", "steered generation")

                # Strategy options
                with Horizontal(classes="option-row"):
                    yield Static("Layer", classes="option-label")
                    with Horizontal(classes="option-pills"):
                        yield OptionPill("layer", "auto", "Auto", selected=True, id="layer-auto")
                        yield OptionPill("layer", "sweep", "Sweep", id="layer-sweep")
                        yield OptionPill("layer", "middle", "Middle", id="layer-middle")
                        yield OptionPill("layer", "late", "Late", id="layer-late")

                yield ParamRow("Layers", "inp-layers", "", "e.g. 15, 16, 17")

                with Horizontal(classes="option-row"):
                    yield Static("Aggregation", classes="option-label")
                    with Horizontal(classes="option-pills"):
                        yield OptionPill("aggregation", "top_k_average", "Top K Avg", selected=True, id="agg-topk")
                        yield OptionPill("aggregation", "best_single", "Best", id="agg-best")
                        yield OptionPill("aggregation", "weighted_average", "Weighted", id="agg-weighted")
                        yield OptionPill("aggregation", "pca_principal", "PCA", id="agg-pca")

                with Horizontal(classes="option-row"):
                    yield Static("Contrast", classes="option-label")
                    with Horizontal(classes="option-pills"):
                        yield OptionPill("contrast", "fast", "Fast", id="contrast-fast")
                        yield OptionPill("contrast", "standard", "Standard", selected=True, id="contrast-standard")
                        yield OptionPill("contrast", "thorough", "Thorough", id="contrast-thorough")

                with Horizontal(classes="option-row"):
                    yield Static("Intensity", classes="option-label")
                    with Horizontal(classes="option-pills"):
                        yield OptionPill("intensity", "extreme", "Extreme", id="intensity-extreme")
                        yield OptionPill("intensity", "balanced", "Balanced", selected=True, id="intensity-balanced")
                        yield OptionPill("intensity", "natural", "Natural", id="intensity-natural")

                with Horizontal(classes="option-row"):
                    yield Static("Tournament", classes="option-label")
                    with Horizontal(classes="option-pills"):
                        yield OptionPill("tournament", "off", "Off", id="tournament-off")
                        yield OptionPill("tournament", "on", "On", selected=True, id="tournament-on")

                with Vertical(id="tournament-section"):
                    with Horizontal(classes="option-row"):
                        yield Static("  Elimination", classes="option-label")
                        with Horizontal(classes="option-pills"):
                            yield OptionPill("elimination", "1", "1 Round", id="elimination-1")
                            yield OptionPill("elimination", "2", "2 Rounds", selected=True, id="elimination-2")
                            yield OptionPill("elimination", "3", "3 Rounds", id="elimination-3")
                    with ParamSection("TOURNAMENT"):
                        yield ParamRow("Final Survivors", "inp-tournament-survivors", "16", "samples entering finals")
                        yield ParamRow("Min Datapoints", "inp-tournament-min-dp", "15", "datapoints in round 1")
                        yield ParamRow("Max Datapoints", "inp-tournament-max-dp", "60", "datapoints in finals")

        # Footer
        with Horizontal(id="footer"):
            yield Static("", id="status")
            yield Button("Cancel", id="btn-cancel")
            yield Button("Create Task", id="btn-create")

        yield TmuxBar(active_screen="dashboard")

    def on_mount(self) -> None:
        """Set initial visibility based on default extraction method."""
        self._update_extraction_visibility()
        self._update_tournament_visibility()
        self._update_signal_params_visibility()

    def on_profile_card_selected(self, event: ProfileCard.Selected) -> None:
        self._profile = event.profile
        for p in ["quick", "standard", "comprehensive"]:
            self.query_one(f"#profile-{p}", ProfileCard).set_selected(p == event.profile)
        self._apply_profile(event.profile)

    def on_option_pill_selected(self, event: OptionPill.Selected) -> None:
        """Handle option pill selection."""
        if event.group == "extraction":
            self._extraction_method = event.value
            for pill in self.query(OptionPill):
                if pill.group == "extraction":
                    pill.set_selected(pill.value == event.value)
            self._update_extraction_visibility()
        elif event.group == "outliers":
            self._remove_outliers = (event.value == "remove")
            for pill in self.query(OptionPill):
                if pill.group == "outliers":
                    pill.set_selected(pill.value == event.value)
            self._update_outlier_threshold_visibility()
        elif event.group == "layer":
            self._layer_strategy = event.value
            for pill in self.query(OptionPill):
                if pill.group == "layer":
                    pill.set_selected(pill.value == event.value)
            self._apply_layer_preset(event.value)
        elif event.group == "aggregation":
            self._aggregation = event.value
            for pill in self.query(OptionPill):
                if pill.group == "aggregation":
                    pill.set_selected(pill.value == event.value)
        elif event.group == "contrast":
            self._contrast_quality = event.value
            for pill in self.query(OptionPill):
                if pill.group == "contrast":
                    pill.set_selected(pill.value == event.value)
            self._apply_contrast_preset(event.value)
        elif event.group == "intensity":
            self._intensity_profile = event.value
            for pill in self.query(OptionPill):
                if pill.group == "intensity":
                    pill.set_selected(pill.value == event.value)
            self._apply_intensity_preset(event.value)
        elif event.group == "tournament":
            self._tournament_enabled = (event.value == "on")
            for pill in self.query(OptionPill):
                if pill.group == "tournament":
                    pill.set_selected(pill.value == event.value)
            self._update_tournament_visibility()
        elif event.group == "elimination":
            self._elimination_rounds = int(event.value)
            for pill in self.query(OptionPill):
                if pill.group == "elimination":
                    pill.set_selected(pill.value == event.value)
            self._update_tournament_preview()
        elif event.group == "signal_filter":
            self._signal_filter_mode = event.value
            for pill in self.query(OptionPill):
                if pill.group == "signal_filter":
                    pill.set_selected(pill.value == event.value)
            self._update_signal_params_visibility()
        elif event.group == "ext_intensity":
            self._extraction_intensity = event.value
            for pill in self.query(OptionPill):
                if pill.group == "ext_intensity":
                    pill.set_selected(pill.value == event.value)

    def _update_signal_params_visibility(self) -> None:
        """Show/hide signal filter params based on mode."""
        try:
            min_signal_row = self.query_one("#min-signal-row")
            topk_pairs_row = self.query_one("#topk-pairs-row")

            # Show min_signal when mode is "threshold"
            min_signal_row.display = self._signal_filter_mode == "threshold"
            # Show topk_pairs when mode is "top_k"
            topk_pairs_row.display = self._signal_filter_mode == "top_k"
        except Exception:
            pass

    def _update_extraction_visibility(self) -> None:
        """Show/hide UI elements based on extraction method."""
        is_caa = self._extraction_method == "caa"
        is_gradient = self._extraction_method in ("gradient", "hybrid")

        # CAA-specific options (outlier removal)
        try:
            self.query_one("#outlier-row").display = is_caa
            self.query_one("#outlier-threshold-row").display = is_caa and self._remove_outliers
        except Exception:
            pass  # Elements may not exist yet

        # Optimization section (only for gradient/hybrid)
        try:
            self.query_one("#optimization-section").display = is_gradient
        except Exception:
            pass

    def _update_outlier_threshold_visibility(self) -> None:
        """Show/hide outlier threshold based on remove_outliers setting."""
        try:
            self.query_one("#outlier-threshold-row").display = self._remove_outliers
        except Exception:
            pass

    def _update_tournament_visibility(self) -> None:
        """Show/hide tournament settings based on tournament_enabled."""
        try:
            self.query_one("#tournament-section").display = self._tournament_enabled
        except Exception:
            pass

    def _update_tournament_preview(self) -> None:
        """Update tournament preview based on current settings."""
        try:
            survivors = int(self.query_one("#inp-tournament-survivors", Input).value or "16")
            rounds = self._elimination_rounds

            # Calculate initial samples (75% elimination per round)
            initial = int(survivors / (0.25 ** rounds))

            # Update status or a preview label
            # For now, just log it
            import logging
            logging.getLogger(__name__).debug(
                f"Tournament preview: {initial} initial → {survivors} survivors "
                f"over {rounds + 1} rounds"
            )
        except Exception:
            pass

    def on_model_card_clicked(self, event: ModelCard.Clicked) -> None:
        """Handle API model card click - open selector."""
        config_map = {
            "generator": self._generator_config,
            "judge": self._judge_config,
            "expander": self._expander_config,
        }
        current_config = config_map.get(event.field_name)
        self.app.push_screen(
            ModelSelectorScreen(event.field_name, current_config),
            callback=self._on_model_selected,
        )

    def on_target_model_card_clicked(self, event: TargetModelCard.Clicked) -> None:
        """Handle target model card click - open HF model selector."""
        self.app.push_screen(
            TargetModelSelectorScreen(self._target_config),
            callback=self._on_target_model_selected,
        )

    def _load_model_config(self, role: str) -> ModelConfig | None:
        """Load model config from preferences, falling back to default.

        Args:
            role: One of ModelRole constants (generator, judge, expander).

        Returns:
            The saved ModelConfig or default if not found.
        """
        saved_id = self._preferences.get_selected_model(role)
        if saved_id:
            config = self._model_manager.get(saved_id)
            if config:
                return config
        # Fall back to default
        return self._model_manager.get_default()

    def _load_target_config(self) -> HFModelConfig | None:
        """Load target model config from preferences, falling back to default.

        Returns:
            The saved HFModelConfig or default if not found.
        """
        saved_id = self._preferences.get_selected_model(ModelRole.TARGET)
        if saved_id:
            config = self._hf_model_manager.get(saved_id)
            if config:
                return config
        # Fall back to default
        return self._hf_model_manager.get_default()

    def _on_target_model_selected(
        self, result: TargetModelSelectorScreen.ModelSelected | None
    ) -> None:
        """Handle target model selection from modal."""
        if result is None:
            return

        self._target_config = result.config
        self.query_one("#model-target", TargetModelCard).set_config(result.config)

        # Save selection to preferences
        self._preferences.set_selected_model(ModelRole.TARGET, result.config.id)

        # Refresh layers based on new model's num_layers
        self._apply_layer_preset(self._layer_strategy)

    def _on_model_selected(self, result: ModelSelectorScreen.ModelSelected | None) -> None:
        """Handle model selection from modal."""
        if result is None:
            return

        if result.field_name == "generator":
            self._generator_config = result.config
            self.query_one("#model-generator", ModelCard).set_config(result.config)
            self._preferences.set_selected_model(ModelRole.GENERATOR, result.config.id)
        elif result.field_name == "judge":
            self._judge_config = result.config
            self.query_one("#model-judge", ModelCard).set_config(result.config)
            self._preferences.set_selected_model(ModelRole.JUDGE, result.config.id)
        elif result.field_name == "expander":
            self._expander_config = result.config
            self.query_one("#model-expander", ModelCard).set_config(result.config)
            self._preferences.set_selected_model(ModelRole.EXPANDER, result.config.id)

    def _apply_profile(self, profile: str) -> None:
        configs = {
            "quick": TaskConfig.quick(),
            "standard": TaskConfig.standard(),
            "comprehensive": TaskConfig.comprehensive(),
        }
        cfg = configs.get(profile, TaskConfig.standard())

        # Update sampling fields
        self.query_one("#inp-samples", Input).value = str(cfg.num_samples)
        self.query_one("#inp-seeds", Input).value = str(cfg.num_seeds)
        self.query_one("#inp-datapoints", Input).value = str(cfg.datapoints_per_sample)

        # Update optimization fields
        opt = cfg.optimization
        self.query_one("#inp-lr", Input).value = str(opt.lr)
        self.query_one("#inp-max-iters", Input).value = str(opt.max_iters)
        self.query_one("#inp-coldness", Input).value = str(opt.coldness)
        self.query_one("#inp-max-norm", Input).value = str(opt.max_norm or "")

        # Update parallelism fields
        self.query_one("#inp-extractions", Input).value = str(cfg.max_concurrent_extractions)
        self.query_one("#inp-evaluations", Input).value = str(cfg.max_concurrent_evaluations)
        self.query_one("#inp-generations", Input).value = str(cfg.contrast.max_concurrent_generations)
        self.query_one("#inp-topk", Input).value = str(cfg.top_k)

        # Update contrast fields
        contrast = cfg.contrast
        self.query_one("#inp-core-pool", Input).value = str(contrast.core_pool_size)
        self.query_one("#inp-core-per-sample", Input).value = str(contrast.core_seeds_per_sample)
        self.query_one("#inp-unique-per-sample", Input).value = str(contrast.unique_seeds_per_sample)
        self.query_one("#inp-max-regen", Input).value = str(contrast.max_regeneration_attempts)

        # Update validation fields
        self.query_one("#inp-min-quality", Input).value = str(contrast.min_contrast_quality)
        self.query_one("#inp-min-dim", Input).value = str(contrast.min_dimension_score)
        self.query_one("#inp-min-struct", Input).value = str(contrast.min_structural_score)
        self.query_one("#inp-min-semantic", Input).value = str(contrast.min_semantic_score)

        # Update evaluation fields
        eval_cfg = cfg.evaluation
        strengths_str = ", ".join(str(s) for s in eval_cfg.strength_levels)
        self.query_one("#inp-strengths", Input).value = strengths_str
        self.query_one("#inp-eval-temp", Input).value = str(eval_cfg.generation_temperature)

        # Update layer strategy pills
        strategy = cfg.layer_strategies[0].value if cfg.layer_strategies else "auto"
        self._layer_strategy = strategy
        for pill in self.query(OptionPill):
            if pill.group == "layer":
                pill.set_selected(pill.value == strategy)
        self._apply_layer_preset(strategy)

        # Update aggregation pills
        self._aggregation = cfg.aggregation_strategy.value
        for pill in self.query(OptionPill):
            if pill.group == "aggregation":
                pill.set_selected(pill.value == cfg.aggregation_strategy.value)

        # Update contrast quality pills based on profile
        contrast_preset = {
            "quick": "fast",
            "standard": "standard",
            "comprehensive": "thorough",
        }.get(profile, "standard")
        self._contrast_quality = contrast_preset
        for pill in self.query(OptionPill):
            if pill.group == "contrast":
                pill.set_selected(pill.value == contrast_preset)

        # Update intensity profile pills based on profile
        intensity_preset = {
            "quick": "extreme",      # Fast direction finding
            "standard": "balanced",  # Default balanced
            "comprehensive": "natural",  # Production quality
        }.get(profile, "balanced")
        self._intensity_profile = intensity_preset
        for pill in self.query(OptionPill):
            if pill.group == "intensity":
                pill.set_selected(pill.value == intensity_preset)
        self._apply_intensity_preset(intensity_preset)

        # Update tournament settings based on profile
        # All profiles have tournament enabled by default with appropriate rounds
        # Elimination rates calculated to achieve target initial→final samples:
        #   Quick: 32→4 (1 round, 87.5% elim)
        #   Standard: 256→16 (2 rounds, 75% elim)
        #   Comprehensive: 1024→32 (3 rounds, 68.5% elim)
        tournament_settings = {
            "quick": {"enabled": True, "rounds": 1, "survivors": 4, "elim_rate": 0.875, "min_dp": 15, "max_dp": 40},
            "standard": {"enabled": True, "rounds": 2, "survivors": 16, "elim_rate": 0.75, "min_dp": 15, "max_dp": 60},
            "comprehensive": {"enabled": True, "rounds": 3, "survivors": 32, "elim_rate": 0.685, "min_dp": 15, "max_dp": 80},
        }
        t_settings = tournament_settings.get(profile, tournament_settings["standard"])

        self._tournament_enabled = t_settings["enabled"]
        self._elimination_rounds = t_settings["rounds"]
        self._elimination_rate = t_settings["elim_rate"]

        # Update tournament pills
        for pill in self.query(OptionPill):
            if pill.group == "tournament":
                pill.set_selected(pill.value == ("on" if t_settings["enabled"] else "off"))
            if pill.group == "elimination":
                pill.set_selected(pill.value == str(t_settings["rounds"]))

        # Update tournament input fields
        self.query_one("#inp-tournament-survivors", Input).value = str(t_settings["survivors"])
        self.query_one("#inp-tournament-min-dp", Input).value = str(t_settings["min_dp"])
        self.query_one("#inp-tournament-max-dp", Input).value = str(t_settings["max_dp"])

        # Calculate and display initial samples
        keep_rate = 1.0 - t_settings["elim_rate"]
        initial_samples = int(t_settings["survivors"] / (keep_rate ** t_settings["rounds"]))
        self.query_one("#inp-samples", Input).value = str(initial_samples)

        self._update_tournament_visibility()

    def _apply_contrast_preset(self, preset: str) -> None:
        """Apply a contrast quality preset to the fields."""
        preset_map = {
            "fast": ContrastConfig.fast(),
            "standard": ContrastConfig.standard(),
            "thorough": ContrastConfig.thorough(),
        }
        contrast = preset_map.get(preset, ContrastConfig.standard())

        # Update contrast fields
        self.query_one("#inp-core-pool", Input).value = str(contrast.core_pool_size)
        self.query_one("#inp-core-per-sample", Input).value = str(contrast.core_seeds_per_sample)
        self.query_one("#inp-unique-per-sample", Input).value = str(contrast.unique_seeds_per_sample)
        self.query_one("#inp-max-regen", Input).value = str(contrast.max_regeneration_attempts)
        self.query_one("#inp-generations", Input).value = str(contrast.max_concurrent_generations)

        # Update validation fields
        self.query_one("#inp-min-quality", Input).value = str(contrast.min_contrast_quality)
        self.query_one("#inp-min-dim", Input).value = str(contrast.min_dimension_score)
        self.query_one("#inp-min-struct", Input).value = str(contrast.min_structural_score)
        self.query_one("#inp-min-semantic", Input).value = str(contrast.min_semantic_score)

    def _apply_intensity_preset(self, preset: str) -> None:
        """Apply an intensity profile preset to the fields."""
        from vector_forge.tasks.config import IntensityProfile, ContrastConfig

        profile_map = {
            "extreme": IntensityProfile.EXTREME,
            "balanced": IntensityProfile.BALANCED,
            "natural": IntensityProfile.NATURAL,
        }
        profile = profile_map.get(preset, IntensityProfile.BALANCED)
        values = ContrastConfig.get_intensity_preset(profile)

        # Update intensity input fields
        self.query_one("#inp-intensity-extreme", Input).value = str(values["intensity_extreme"])
        self.query_one("#inp-intensity-high", Input).value = str(values["intensity_high"])
        self.query_one("#inp-intensity-medium", Input).value = str(values["intensity_medium"])
        self.query_one("#inp-intensity-natural", Input).value = str(values["intensity_natural"])

    def _compute_layers_for_strategy(self, strategy: str, num_layers: int) -> list[int]:
        """Compute target layers for a given strategy and model size."""
        if strategy == "auto":
            base = num_layers // 3
            offset = num_layers // 10
            return [base - offset, base, base + offset]
        elif strategy == "sweep":
            start = num_layers // 4
            end = 3 * num_layers // 4
            return list(range(start, end, 2))
        elif strategy == "middle":
            mid = num_layers // 2
            return [mid - 1, mid, mid + 1]
        elif strategy == "late":
            return list(range(3 * num_layers // 4, num_layers - 2))
        else:
            return [num_layers // 2]

    def _apply_layer_preset(self, strategy: str) -> None:
        """Apply a layer strategy preset to the layers field."""
        num_layers = self._target_config.num_layers if self._target_config else None

        if num_layers:
            layers = self._compute_layers_for_strategy(strategy, num_layers)
            self.query_one("#inp-layers", Input).value = ", ".join(str(l) for l in layers)
        else:
            # No model loaded yet - show placeholder based on strategy
            placeholders = {
                "auto": "middle ±1",
                "sweep": "25%-75% (step 2)",
                "middle": "middle ±1",
                "late": "75%-end",
            }
            self.query_one("#inp-layers", Input).value = ""
            self.query_one("#inp-layers", Input).placeholder = placeholders.get(strategy, "auto")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.action_cancel()
        elif event.button.id == "btn-create":
            self.action_create()

    def _status(self, msg: str, level: str = "info") -> None:
        colors = {"error": "$error", "success": "$success", "warning": "$warning"}
        color = colors.get(level, "$foreground-muted")
        # Escape brackets in message to avoid Rich markup interpretation
        escaped_msg = msg.replace("[", r"\[").replace("]", r"\]")
        self.query_one("#status", Static).update(f"[{color}]{escaped_msg}[/]")

    def _build_config(self) -> TaskConfig:
        # Map layer strategy
        layer_map = {
            "auto": LayerStrategy.AUTO,
            "sweep": LayerStrategy.SWEEP,
            "middle": LayerStrategy.MIDDLE,
            "late": LayerStrategy.LATE,
            "fixed": LayerStrategy.FIXED,
        }
        layer_strategy = layer_map.get(self._layer_strategy, LayerStrategy.AUTO)

        # Map aggregation strategy
        agg_map = {
            "best_single": AggregationStrategy.BEST_SINGLE,
            "top_k_average": AggregationStrategy.TOP_K_AVERAGE,
            "weighted_average": AggregationStrategy.WEIGHTED_AVERAGE,
            "pca_principal": AggregationStrategy.PCA_PRINCIPAL,
            "strategy_grouped": AggregationStrategy.STRATEGY_GROUPED,
        }
        aggregation = agg_map.get(self._aggregation, AggregationStrategy.TOP_K_AVERAGE)

        # Get model names from configs (with provider prefix for litellm)
        generator_model = (
            self._generator_config.get_litellm_model() if self._generator_config else DEFAULT_MODEL
        )
        judge_model = (
            self._judge_config.get_litellm_model() if self._judge_config else DEFAULT_MODEL
        )
        expander_model = (
            self._expander_config.get_litellm_model() if self._expander_config else DEFAULT_MODEL
        )

        # Build optimization config
        max_norm_str = self.query_one("#inp-max-norm", Input).value.strip()
        optimization_config = OptimizationConfig(
            lr=float(self.query_one("#inp-lr", Input).value or "0.1"),
            max_iters=int(self.query_one("#inp-max-iters", Input).value or "50"),
            coldness=float(self.query_one("#inp-coldness", Input).value or "0.7"),
            max_norm=float(max_norm_str) if max_norm_str else None,
        )

        # Build contrast config
        contrast_config = ContrastConfig(
            core_pool_size=int(self.query_one("#inp-core-pool", Input).value or "80"),
            core_seeds_per_sample=int(self.query_one("#inp-core-per-sample", Input).value or "40"),
            unique_seeds_per_sample=int(self.query_one("#inp-unique-per-sample", Input).value or "10"),
            max_regeneration_attempts=int(self.query_one("#inp-max-regen", Input).value or "2"),
            max_concurrent_generations=int(self.query_one("#inp-generations", Input).value or "16"),
            min_contrast_quality=float(self.query_one("#inp-min-quality", Input).value or "6.0"),
            min_dimension_score=float(self.query_one("#inp-min-dim", Input).value or "6.0"),
            min_structural_score=float(self.query_one("#inp-min-struct", Input).value or "7.0"),
            min_semantic_score=float(self.query_one("#inp-min-semantic", Input).value or "4.0"),
            # Intensity distribution
            intensity_extreme=float(self.query_one("#inp-intensity-extreme", Input).value or "0.10"),
            intensity_high=float(self.query_one("#inp-intensity-high", Input).value or "0.20"),
            intensity_medium=float(self.query_one("#inp-intensity-medium", Input).value or "0.30"),
            intensity_natural=float(self.query_one("#inp-intensity-natural", Input).value or "0.40"),
        )

        # Get target model (required)
        target_model = (
            self._target_config.model_id if self._target_config else None
        )

        # Parse target layers from input
        layers_str = self.query_one("#inp-layers", Input).value.strip()
        target_layers = None
        if layers_str:
            try:
                target_layers = [int(l.strip()) for l in layers_str.split(",") if l.strip()]
            except ValueError:
                pass  # Invalid input, use None

        # Parse strength levels from input (requires at least 2 levels)
        strengths_str = self.query_one("#inp-strengths", Input).value.strip()
        strength_levels = [0.5, 1.0, 1.5, 2.0]  # Default
        if strengths_str:
            try:
                parsed = [float(s.strip()) for s in strengths_str.split(",") if s.strip()]
                if len(parsed) >= 2:
                    strength_levels = parsed
                # If less than 2 parsed, keep default
            except ValueError:
                pass  # Invalid input, use default

        # Parse evaluation temperature
        eval_temp_str = self.query_one("#inp-eval-temp", Input).value.strip()
        eval_temp = 0.7  # Default
        if eval_temp_str:
            try:
                eval_temp = float(eval_temp_str)
            except ValueError:
                pass  # Invalid input, use default

        # Build evaluation config
        evaluation_config = EvaluationConfig(
            strength_levels=strength_levels,
            generation_temperature=eval_temp,
        )

        # Map extraction method
        extraction_map = {
            "caa": ExtractionMethod.CAA,
            "gradient": ExtractionMethod.GRADIENT,
            "hybrid": ExtractionMethod.HYBRID,
        }
        extraction_method = extraction_map.get(self._extraction_method, ExtractionMethod.CAA)

        # Build CAA config with signal filtering
        outlier_threshold_str = self.query_one("#inp-outlier-threshold", Input).value.strip()
        outlier_threshold = 3.0
        if outlier_threshold_str:
            try:
                outlier_threshold = float(outlier_threshold_str)
            except ValueError:
                pass

        # Map signal filter mode
        signal_mode_map = {
            "off": SignalFilterMode.OFF,
            "threshold": SignalFilterMode.THRESHOLD,
            "top_k": SignalFilterMode.TOP_K,
        }
        signal_filter_mode = signal_mode_map.get(
            self._signal_filter_mode, SignalFilterMode.OFF
        )

        # Map extraction intensity
        ext_intensity_map = {
            "all": ExtractionIntensity.ALL,
            "high_signal": ExtractionIntensity.HIGH_SIGNAL,
            "maximum": ExtractionIntensity.MAXIMUM,
        }
        extraction_intensity = ext_intensity_map.get(
            self._extraction_intensity, ExtractionIntensity.ALL
        )

        # Parse signal filtering values
        min_signal = float(self.query_one("#inp-min-signal", Input).value or "6.0")
        top_k_pairs = int(self.query_one("#inp-topk-pairs", Input).value or "30")
        min_confound = float(self.query_one("#inp-min-confound", Input).value or "5.0")

        caa_config = CAAConfig(
            remove_extreme_outliers=self._remove_outliers,
            outlier_std_threshold=outlier_threshold,
            signal_filter_mode=signal_filter_mode,
            min_behavioral_signal=min_signal,
            top_k_pairs=top_k_pairs,
            min_confound_score=min_confound,
            extraction_intensity=extraction_intensity,
        )

        # Build tournament config
        tournament_config = TournamentConfig(
            enabled=self._tournament_enabled,
            elimination_rounds=self._elimination_rounds,
            elimination_rate=self._elimination_rate,
            final_survivors=int(self.query_one("#inp-tournament-survivors", Input).value or "16"),
            min_datapoints=int(self.query_one("#inp-tournament-min-dp", Input).value or "15"),
            max_datapoints=int(self.query_one("#inp-tournament-max-dp", Input).value or "60"),
        )

        return TaskConfig(
            extraction_method=extraction_method,
            caa=caa_config,
            num_samples=int(self.query_one("#inp-samples", Input).value or "16"),
            num_seeds=int(self.query_one("#inp-seeds", Input).value or "4"),
            layer_strategies=[layer_strategy],
            target_layers=target_layers,
            optimization=optimization_config,
            contrast=contrast_config,
            datapoints_per_sample=int(self.query_one("#inp-datapoints", Input).value or "50"),
            max_concurrent_extractions=int(self.query_one("#inp-extractions", Input).value or "1"),
            max_concurrent_evaluations=int(self.query_one("#inp-evaluations", Input).value or "16"),
            aggregation_strategy=aggregation,
            top_k=int(self.query_one("#inp-topk", Input).value or "5"),
            evaluation=evaluation_config,
            tournament=tournament_config,
            generator_model=generator_model,
            judge_model=judge_model,
            expander_model=expander_model,
            target_model=target_model,
        )

    def action_cancel(self) -> None:
        self.app.pop_screen()

    def action_create(self) -> None:
        text = self.query_one("#behavior-input", TextArea).text.strip()
        if not text:
            self._status("Enter a behavior description", "error")
            return

        if self._target_config is None:
            self._status("Select a target model first", "error")
            return

        try:
            config = self._build_config()
            # Expansion happens automatically in the pipeline
            self.post_message(self.TaskCreated(config, text))
            self.app.pop_screen()
        except Exception as e:
            self._status(f"Error: {e}", "error")
