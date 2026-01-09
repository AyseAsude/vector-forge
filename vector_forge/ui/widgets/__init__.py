"""Widget components for Vector Forge TUI."""

from vector_forge.ui.widgets.header import AppHeader
from vector_forge.ui.widgets.footer import AppFooter
from vector_forge.ui.widgets.status_bar import StatusBar, ScreenTab
from vector_forge.ui.widgets.target import TargetSection
from vector_forge.ui.widgets.progress import ProgressSection
from vector_forge.ui.widgets.data_panel import DataPanel
from vector_forge.ui.widgets.eval_panel import EvaluationPanel
from vector_forge.ui.widgets.activity import ActivityPanel
from vector_forge.ui.widgets.log_panel import LogPanel
from vector_forge.ui.widgets.extractions_list import ExtractionsList
from vector_forge.ui.widgets.extraction_detail import ExtractionDetail

__all__ = [
    "AppHeader",
    "AppFooter",
    "StatusBar",
    "ScreenTab",
    "TargetSection",
    "ProgressSection",
    "DataPanel",
    "EvaluationPanel",
    "ActivityPanel",
    "LogPanel",
    "ExtractionsList",
    "ExtractionDetail",
]
