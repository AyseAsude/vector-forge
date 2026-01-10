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

    CreateTaskScreen #expand-btn {
        width: auto;
        min-width: 16;
        height: 1;
        background: $accent;
        color: $background;
        border: none;
        padding: 0 1;
    }

    CreateTaskScreen #expand-btn:hover {
        background: $accent 80%;
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
        self._expanded = None
        self._expanding = False
        self._layer_strategy = "auto"
        self._aggregation = "top_k_average"
        self._contrast_quality = "standard"

        # Preferences and model managers
        self._preferences = PreferencesManager()
        self._model_manager = ModelConfigManager()
        self._hf_model_manager = HFModelConfigManager()

        # Load API model configs from preferences (or fall back to default)
        self._extractor_config = self._load_model_config(ModelRole.EXTRACTOR)
        self._judge_config = self._load_model_config(ModelRole.JUDGE)
        self._expander_config = self._load_model_config(ModelRole.EXPANDER)

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
                yield ProfileCard("quick", "Quick", "4 samples", id="profile-quick")
                yield ProfileCard("standard", "Standard", "16 samples", selected=True, id="profile-standard")
                yield ProfileCard("comprehensive", "Full", "32 samples", id="profile-comprehensive")

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
                    field_name="extractor",
                    label="EXTRACTOR",
                    config=self._extractor_config,
                    id="model-extractor",
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
                yield Button("Expand with LLM", id="expand-btn")

            # Parameters section
            yield Static("PARAMETERS", classes="main-section")
            with Vertical(id="params-container"):
                # Row 1: Sampling & Optimization
                with Horizontal(classes="params-row"):
                    with ParamSection("SAMPLING"):
                        yield ParamRow("Samples", "inp-samples", "16")
                        yield ParamRow("Seeds", "inp-seeds", "4")
                        yield ParamRow("Datapoints", "inp-datapoints", "50", "per sample")

                    with ParamSection("OPTIMIZATION"):
                        yield ParamRow("Learning Rate", "inp-lr", "0.1")
                        yield ParamRow("Max Iters", "inp-max-iters", "50")
                        yield ParamRow("Coldness", "inp-coldness", "0.7", "softmax temp")
                        yield ParamRow("Max Norm", "inp-max-norm", "2.0", "regularization")

                # Row 2: Contrast & Validation
                with Horizontal(classes="params-row"):
                    with ParamSection("CONTRAST"):
                        yield ParamRow("Core Pool", "inp-core-pool", "80", "shared pairs")
                        yield ParamRow("Core/Sample", "inp-core-per-sample", "40")
                        yield ParamRow("Unique/Sample", "inp-unique-per-sample", "10")
                        yield ParamRow("Max Regen", "inp-max-regen", "2", "retry attempts")

                    with ParamSection("VALIDATION"):
                        yield ParamRow("Min Quality", "inp-min-quality", "6.0", "contrast score")
                        yield ParamRow("Min DST", "inp-min-dst", "7.0", "behavior score")
                        yield ParamRow("Max SRC", "inp-max-src", "3.0", "behavior score")
                        yield ParamRow("Min Distance", "inp-min-dist", "0.3", "semantic")

                # Row 3: Parallelism
                with Horizontal(classes="params-row"):
                    with ParamSection("PARALLELISM"):
                        yield ParamRow("Extractions", "inp-extractions", "1")
                        yield ParamRow("Evaluations", "inp-evaluations", "16")
                        yield ParamRow("Generations", "inp-generations", "16", "LLM API calls")
                        yield ParamRow("Top K", "inp-topk", "5")

                # Strategy options
                with Horizontal(classes="option-row"):
                    yield Static("Layer", classes="option-label")
                    with Horizontal(classes="option-pills"):
                        yield OptionPill("layer", "auto", "Auto", selected=True, id="layer-auto")
                        yield OptionPill("layer", "sweep", "Sweep", id="layer-sweep")
                        yield OptionPill("layer", "middle", "Middle", id="layer-middle")
                        yield OptionPill("layer", "late", "Late", id="layer-late")

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

        # Footer
        with Horizontal(id="footer"):
            yield Static("", id="status")
            yield Button("Cancel", id="btn-cancel")
            yield Button("Create Task", id="btn-create")

        yield TmuxBar(active_screen="dashboard")

    def on_profile_card_selected(self, event: ProfileCard.Selected) -> None:
        self._profile = event.profile
        for p in ["quick", "standard", "comprehensive"]:
            self.query_one(f"#profile-{p}", ProfileCard).set_selected(p == event.profile)
        self._apply_profile(event.profile)

    def on_option_pill_selected(self, event: OptionPill.Selected) -> None:
        """Handle option pill selection."""
        if event.group == "layer":
            self._layer_strategy = event.value
            for pill in self.query(OptionPill):
                if pill.group == "layer":
                    pill.set_selected(pill.value == event.value)
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
            # Update contrast fields based on preset
            self._apply_contrast_preset(event.value)

    def on_model_card_clicked(self, event: ModelCard.Clicked) -> None:
        """Handle API model card click - open selector."""
        config_map = {
            "extractor": self._extractor_config,
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
            role: One of ModelRole constants (extractor, judge, expander).

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

    def _on_model_selected(self, result: ModelSelectorScreen.ModelSelected | None) -> None:
        """Handle model selection from modal."""
        if result is None:
            return

        if result.field_name == "extractor":
            self._extractor_config = result.config
            self.query_one("#model-extractor", ModelCard).set_config(result.config)
            self._preferences.set_selected_model(ModelRole.EXTRACTOR, result.config.id)
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
        self.query_one("#inp-min-dst", Input).value = str(contrast.min_dst_score)
        self.query_one("#inp-max-src", Input).value = str(contrast.max_src_score)
        self.query_one("#inp-min-dist", Input).value = str(contrast.min_semantic_distance)

        # Update layer strategy pills
        strategy = cfg.layer_strategies[0].value if cfg.layer_strategies else "auto"
        self._layer_strategy = strategy
        for pill in self.query(OptionPill):
            if pill.group == "layer":
                pill.set_selected(pill.value == strategy)

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
        self.query_one("#inp-min-dst", Input).value = str(contrast.min_dst_score)
        self.query_one("#inp-max-src", Input).value = str(contrast.max_src_score)
        self.query_one("#inp-min-dist", Input).value = str(contrast.min_semantic_distance)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.action_cancel()
        elif event.button.id == "btn-create":
            self.action_create()
        elif event.button.id == "expand-btn":
            self._do_expand()

    def _do_expand(self) -> None:
        if self._expanding:
            return

        textarea = self.query_one("#behavior-input", TextArea)
        text = textarea.text.strip()
        if not text:
            self._status("Enter a behavior first", "error")
            return

        self._expanding = True
        self._status("Expanding with LLM...", "warning")
        textarea.disabled = True
        self.run_worker(self._expand_async(text), exclusive=True)

    async def _expand_async(self, text: str) -> None:
        try:
            from vector_forge.tasks.expander import BehaviorExpander
            from vector_forge.llm import create_client

            # Use expander model, or default (with provider prefix)
            model = self._expander_config.get_litellm_model() if self._expander_config else DEFAULT_MODEL
            llm = create_client(model)
            expander = BehaviorExpander(llm)
            result = await expander.expand(text)

            self._expanded = result

            expanded_text = f"""{result.name}

{result.description}

DEFINITION:
{result.detailed_definition[:500]}{'...' if len(result.detailed_definition) > 500 else ''}

POSITIVE EXAMPLES:
{chr(10).join('• ' + ex[:150] for ex in result.positive_examples[:3])}

NEGATIVE EXAMPLES:
{chr(10).join('• ' + ex[:150] for ex in result.negative_examples[:3])}

DOMAINS: {', '.join(result.domains[:6])}
"""
            textarea = self.query_one("#behavior-input", TextArea)
            textarea.text = expanded_text
            textarea.disabled = False
            self._status("Expanded - you can edit", "success")

        except Exception as e:
            self._status(f"Error: {e}", "error")
            self.query_one("#behavior-input", TextArea).disabled = False
        finally:
            self._expanding = False

    def _status(self, msg: str, level: str = "info") -> None:
        colors = {"error": "$error", "success": "$success", "warning": "$warning"}
        color = colors.get(level, "$foreground-muted")
        self.query_one("#status", Static).update(f"[{color}]{msg}[/]")

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
        extractor_model = (
            self._extractor_config.get_litellm_model() if self._extractor_config else DEFAULT_MODEL
        )
        judge_model = (
            self._judge_config.get_litellm_model() if self._judge_config else DEFAULT_MODEL
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
            min_dst_score=float(self.query_one("#inp-min-dst", Input).value or "7.0"),
            max_src_score=float(self.query_one("#inp-max-src", Input).value or "3.0"),
            min_semantic_distance=float(self.query_one("#inp-min-dist", Input).value or "0.3"),
        )

        # Get target model (required)
        target_model = (
            self._target_config.model_id if self._target_config else None
        )

        return TaskConfig(
            num_samples=int(self.query_one("#inp-samples", Input).value or "16"),
            num_seeds=int(self.query_one("#inp-seeds", Input).value or "4"),
            layer_strategies=[layer_strategy],
            optimization=optimization_config,
            contrast=contrast_config,
            datapoints_per_sample=int(self.query_one("#inp-datapoints", Input).value or "50"),
            max_concurrent_extractions=int(self.query_one("#inp-extractions", Input).value or "1"),
            max_concurrent_evaluations=int(self.query_one("#inp-evaluations", Input).value or "16"),
            aggregation_strategy=aggregation,
            top_k=int(self.query_one("#inp-topk", Input).value or "5"),
            extractor_model=extractor_model,
            judge_model=judge_model,
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
            # Use LLM-generated short description if available, otherwise raw text
            if self._expanded:
                description = f"{self._expanded.name}: {self._expanded.description}"
            else:
                description = text
            self.post_message(self.TaskCreated(config, description))
            self.app.pop_screen()
        except Exception as e:
            self._status(f"Error: {e}", "error")
