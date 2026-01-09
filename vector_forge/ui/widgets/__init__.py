"""Widget modules for Vector Forge TUI."""

from vector_forge.ui.widgets.status_bar import StatusBar, ScreenTab
from vector_forge.ui.widgets.extraction_selector import ExtractionSelector
from vector_forge.ui.widgets.progress import ProgressSection
from vector_forge.ui.widgets.data_panel import DataPanel
from vector_forge.ui.widgets.eval_panel import EvaluationPanel
from vector_forge.ui.widgets.log_panel import LogPanel
from vector_forge.ui.widgets.agents_list import AgentsList
from vector_forge.ui.widgets.agent_inspector import AgentInspector

__all__ = [
    "StatusBar",
    "ScreenTab",
    "ExtractionSelector",
    "ProgressSection",
    "DataPanel",
    "EvaluationPanel",
    "LogPanel",
    "AgentsList",
    "AgentInspector",
]
