"""Task creation screen for configuring and launching extractions."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, Horizontal, Grid
from textual.screen import Screen
from textual.widgets import (
    Static,
    Input,
    Button,
    Select,
    Label,
    TextArea,
    Switch,
    ProgressBar,
)
from textual.message import Message

from vector_forge.tasks.config import (
    TaskConfig,
    LayerStrategy,
    AggregationStrategy,
)


class TaskParameterGroup(Static):
    """A group of related task parameters."""

    DEFAULT_CSS = """
    TaskParameterGroup {
        height: auto;
        padding: 1;
        border: solid $primary-darken-2;
        margin-bottom: 1;
    }

    TaskParameterGroup .group-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    TaskParameterGroup .param-row {
        height: auto;
        margin-bottom: 1;
    }

    TaskParameterGroup Label {
        width: 20;
        padding-right: 1;
    }

    TaskParameterGroup Input {
        width: 1fr;
    }

    TaskParameterGroup Select {
        width: 1fr;
    }
    """


class CreateTaskScreen(Screen):
    """Screen for creating and configuring extraction tasks.

    Provides a form-based interface for:
    - Entering behavior descriptions
    - Configuring parallelism and sampling parameters
    - Selecting evaluation thoroughness
    - Previewing expanded behavior
    - Launching extraction tasks
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+enter", "create", "Create Task"),
    ]

    DEFAULT_CSS = """
    CreateTaskScreen {
        background: $background;
    }

    CreateTaskScreen #main-container {
        width: 100%;
        height: 100%;
        padding: 1 2;
    }

    CreateTaskScreen #title {
        text-style: bold;
        color: $text;
        text-align: center;
        padding: 1;
        margin-bottom: 1;
    }

    CreateTaskScreen #behavior-section {
        height: auto;
        margin-bottom: 1;
    }

    CreateTaskScreen #behavior-input {
        height: 4;
        margin-bottom: 1;
    }

    CreateTaskScreen #params-grid {
        grid-size: 2;
        grid-gutter: 1 2;
        height: auto;
        margin-bottom: 1;
    }

    CreateTaskScreen #preview-section {
        height: 1fr;
        min-height: 10;
        border: solid $primary-darken-2;
        padding: 1;
        margin-bottom: 1;
    }

    CreateTaskScreen #preview-title {
        text-style: bold;
        margin-bottom: 1;
    }

    CreateTaskScreen #preview-content {
        height: 1fr;
        background: $surface;
    }

    CreateTaskScreen #actions {
        height: auto;
        align: center middle;
        padding: 1;
    }

    CreateTaskScreen Button {
        margin: 0 1;
    }

    CreateTaskScreen #create-btn {
        background: $success;
    }

    CreateTaskScreen #expand-btn {
        background: $primary;
    }

    CreateTaskScreen .section-label {
        text-style: bold;
        color: $text-muted;
        margin-bottom: 1;
    }

    CreateTaskScreen .param-label {
        width: 18;
    }

    CreateTaskScreen .param-input {
        width: 1fr;
    }

    CreateTaskScreen #status-bar {
        height: 1;
        background: $surface;
        padding: 0 1;
    }

    CreateTaskScreen .expanding {
        color: $warning;
    }
    """

    class TaskCreated(Message):
        """Message sent when a task is created."""

        def __init__(self, config: TaskConfig, description: str) -> None:
            super().__init__()
            self.config = config
            self.description = description

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._expanded_behavior = None
        self._is_expanding = False

    def compose(self) -> ComposeResult:
        with Vertical(id="main-container"):
            yield Static("Create Extraction Task", id="title")

            # Behavior description section
            with Vertical(id="behavior-section"):
                yield Static("Behavior Description", classes="section-label")
                yield TextArea(
                    placeholder="Describe the behavior to extract (e.g., 'sycophancy - agreeing with users even when they are factually wrong')",
                    id="behavior-input",
                )
                yield Button("Expand with LLM", id="expand-btn", variant="primary")

            # Parameters grid
            with Grid(id="params-grid"):
                # Left column - Sampling
                with TaskParameterGroup():
                    yield Static("Sampling", classes="group-title")
                    with Horizontal(classes="param-row"):
                        yield Label("Samples:", classes="param-label")
                        yield Input(value="16", id="num-samples", type="integer")
                    with Horizontal(classes="param-row"):
                        yield Label("Seeds:", classes="param-label")
                        yield Input(value="4", id="num-seeds", type="integer")
                    with Horizontal(classes="param-row"):
                        yield Label("Contrast Pairs:", classes="param-label")
                        yield Input(value="100", id="contrast-pairs", type="integer")

                # Right column - Parallelism
                with TaskParameterGroup():
                    yield Static("Parallelism", classes="group-title")
                    with Horizontal(classes="param-row"):
                        yield Label("Extractions:", classes="param-label")
                        yield Input(value="8", id="max-extractions", type="integer")
                    with Horizontal(classes="param-row"):
                        yield Label("Evaluations:", classes="param-label")
                        yield Input(value="16", id="max-evaluations", type="integer")

                # Left column - Strategy
                with TaskParameterGroup():
                    yield Static("Strategy", classes="group-title")
                    with Horizontal(classes="param-row"):
                        yield Label("Layer:", classes="param-label")
                        yield Select(
                            [(s.value, s) for s in LayerStrategy],
                            value=LayerStrategy.AUTO,
                            id="layer-strategy",
                        )
                    with Horizontal(classes="param-row"):
                        yield Label("Aggregation:", classes="param-label")
                        yield Select(
                            [(s.value, s) for s in AggregationStrategy],
                            value=AggregationStrategy.TOP_K_AVERAGE,
                            id="aggregation",
                        )

                # Right column - Evaluation
                with TaskParameterGroup():
                    yield Static("Evaluation", classes="group-title")
                    with Horizontal(classes="param-row"):
                        yield Label("Preset:", classes="param-label")
                        yield Select(
                            [
                                ("Fast", "fast"),
                                ("Standard", "standard"),
                                ("Thorough", "thorough"),
                            ],
                            value="standard",
                            id="eval-preset",
                        )
                    with Horizontal(classes="param-row"):
                        yield Label("Top K:", classes="param-label")
                        yield Input(value="5", id="top-k", type="integer")

            # Preview section
            with Vertical(id="preview-section"):
                yield Static("Expanded Behavior Preview", id="preview-title")
                yield TextArea(
                    "",
                    id="preview-content",
                    read_only=True,
                )

            # Action buttons
            with Horizontal(id="actions"):
                yield Button("Cancel", id="cancel-btn", variant="default")
                yield Button("Create Task", id="create-btn", variant="success")

            yield Static("", id="status-bar")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "cancel-btn":
            self.action_cancel()
        elif button_id == "create-btn":
            self.action_create()
        elif button_id == "expand-btn":
            self._expand_behavior()

    def _expand_behavior(self) -> None:
        """Expand the behavior description using LLM."""
        if self._is_expanding:
            return

        behavior_input = self.query_one("#behavior-input", TextArea)
        description = behavior_input.text.strip()

        if not description:
            self._set_status("Please enter a behavior description", "error")
            return

        self._is_expanding = True
        self._set_status("Expanding behavior with LLM...", "expanding")

        # Run expansion asynchronously
        self.run_worker(self._do_expand(description), exclusive=True)

    async def _do_expand(self, description: str) -> None:
        """Perform the behavior expansion."""
        try:
            # Import here to avoid circular dependency
            from vector_forge.tasks.expander import BehaviorExpander
            from vector_forge.llm import create_client

            # Create LLM client
            llm = create_client("gpt-4o")

            expander = BehaviorExpander(llm)
            expanded = await expander.expand(description)

            self._expanded_behavior = expanded
            self._show_preview(expanded)
            self._set_status("Behavior expanded successfully", "success")

        except Exception as e:
            self._set_status(f"Expansion failed: {e}", "error")
        finally:
            self._is_expanding = False

    def _show_preview(self, expanded) -> None:
        """Display the expanded behavior in the preview area."""
        preview = self.query_one("#preview-content", TextArea)

        text = f"""Name: {expanded.name}

Description: {expanded.description}

Detailed Definition:
{expanded.detailed_definition[:500]}...

Positive Examples:
{chr(10).join('• ' + ex[:100] for ex in expanded.positive_examples[:3])}

Negative Examples:
{chr(10).join('• ' + ex[:100] for ex in expanded.negative_examples[:3])}

Domains: {', '.join(expanded.domains[:6])}

Evaluation Criteria:
{chr(10).join('• ' + c[:80] for c in expanded.evaluation_criteria[:4])}
"""
        preview.text = text

    def _set_status(self, message: str, level: str = "info") -> None:
        """Update the status bar."""
        status = self.query_one("#status-bar", Static)
        if level == "error":
            status.update(f"[red]✗ {message}[/]")
        elif level == "success":
            status.update(f"[green]✓ {message}[/]")
        elif level == "expanding":
            status.update(f"[yellow]⟳ {message}[/]")
        else:
            status.update(message)

    def _build_config(self) -> TaskConfig:
        """Build TaskConfig from form values."""
        from vector_forge.tasks.config import EvaluationConfig

        # Get form values
        num_samples = int(self.query_one("#num-samples", Input).value or "16")
        num_seeds = int(self.query_one("#num-seeds", Input).value or "4")
        contrast_pairs = int(self.query_one("#contrast-pairs", Input).value or "100")
        max_extractions = int(self.query_one("#max-extractions", Input).value or "8")
        max_evaluations = int(self.query_one("#max-evaluations", Input).value or "16")
        top_k = int(self.query_one("#top-k", Input).value or "5")

        layer_strategy = self.query_one("#layer-strategy", Select).value
        aggregation = self.query_one("#aggregation", Select).value
        eval_preset = self.query_one("#eval-preset", Select).value

        # Build evaluation config
        if eval_preset == "fast":
            evaluation = EvaluationConfig.fast()
        elif eval_preset == "thorough":
            evaluation = EvaluationConfig.thorough()
        else:
            evaluation = EvaluationConfig.standard()

        return TaskConfig(
            num_samples=num_samples,
            num_seeds=num_seeds,
            layer_strategies=[layer_strategy] if isinstance(layer_strategy, LayerStrategy) else [LayerStrategy.AUTO],
            contrast_pair_count=contrast_pairs,
            max_concurrent_extractions=max_extractions,
            max_concurrent_evaluations=max_evaluations,
            evaluation=evaluation,
            aggregation_strategy=aggregation if isinstance(aggregation, AggregationStrategy) else AggregationStrategy.TOP_K_AVERAGE,
            top_k=top_k,
        )

    def action_cancel(self) -> None:
        """Cancel and return to previous screen."""
        self.app.pop_screen()

    def action_create(self) -> None:
        """Create the task with current configuration."""
        behavior_input = self.query_one("#behavior-input", TextArea)
        description = behavior_input.text.strip()

        if not description:
            self._set_status("Please enter a behavior description", "error")
            return

        try:
            config = self._build_config()
            self.post_message(self.TaskCreated(config, description))
            self.app.pop_screen()
        except Exception as e:
            self._set_status(f"Invalid configuration: {e}", "error")
