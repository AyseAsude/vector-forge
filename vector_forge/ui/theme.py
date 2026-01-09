"""Color theme and visual constants for Vector Forge TUI.

This is the SINGLE SOURCE OF TRUTH for all colors.
- CSS uses $variable-name syntax (e.g., $surface, $text-muted)
- Python Rich markup uses COLORS.xxx (e.g., COLORS.accent)
"""

from dataclasses import dataclass
from textual.theme import Theme


@dataclass(frozen=True)
class Colors:
    """Gruvbox Dark inspired warm color palette.

    Use these constants in Python code for Rich markup:
        f"[{COLORS.success}]text[/]"
    """

    # Backgrounds
    bg: str = "#1d2021"
    surface: str = "#282828"
    surface_hl: str = "#3c3836"

    # Borders
    border: str = "#504945"
    border_focus: str = "#d79921"

    # Text
    text: str = "#ebdbb2"
    text_muted: str = "#a89984"
    text_dim: str = "#665c54"

    # Semantic colors
    accent: str = "#d79921"
    success: str = "#b8bb26"
    warning: str = "#fe8019"
    error: str = "#fb4934"

    # Extended palette
    aqua: str = "#8ec07c"
    purple: str = "#d3869b"
    blue: str = "#83a598"


# Global color instance for Python Rich markup
COLORS = Colors()


# CSS Variables - these become $variable-name in CSS
_css_variables = {
    # Backgrounds
    "bg": COLORS.bg,
    "surface": COLORS.surface,
    "surface-hl": COLORS.surface_hl,

    # Borders
    "border": COLORS.border,
    "border-focus": COLORS.border_focus,

    # Text
    "text": COLORS.text,
    "text-muted": COLORS.text_muted,
    "text-dim": COLORS.text_dim,

    # Semantic
    "accent": COLORS.accent,
    "success": COLORS.success,
    "warning": COLORS.warning,
    "error": COLORS.error,

    # Extended
    "aqua": COLORS.aqua,
    "purple": COLORS.purple,
    "blue": COLORS.blue,

    # Progress bars
    "bar-complete": COLORS.accent,
    "bar-empty": COLORS.surface_hl,
    "bar-success": COLORS.success,
}


# Textual theme registration
forge_dark = Theme(
    name="forge-dark",
    primary=COLORS.accent,
    secondary=COLORS.text_muted,
    accent=COLORS.accent,
    warning=COLORS.warning,
    error=COLORS.error,
    success=COLORS.success,
    foreground=COLORS.text,
    background=COLORS.bg,
    surface=COLORS.surface,
    panel=COLORS.surface,
    dark=True,
    variables=_css_variables,
)


@dataclass(frozen=True)
class StatusIcons:
    """Unicode icons for status indicators."""

    running: str = "●"
    complete: str = "●"
    paused: str = "●"
    failed: str = "●"
    pending: str = "○"

    # Quality indicators
    keep: str = "●"
    review: str = "◐"
    remove: str = "○"

    # Activity indicators
    active: str = "›"
    success: str = "✓"
    error: str = "✗"
    waiting: str = "⋯"
    thinking: str = "⟳"


ICONS = StatusIcons()


@dataclass(frozen=True)
class Layout:
    """Layout constants."""

    panel_padding: int = 1
    panel_margin: int = 1
    min_log_height: int = 5
    max_activity_lines: int = 4
    progress_bar_width: int = 40


LAYOUT = Layout()
