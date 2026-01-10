"""Theme constants for Vector Forge TUI.

Uses Textual's built-in gruvbox theme. All colors come from the theme automatically.

For CSS: Use theme variables like $primary, $accent, $foreground, $background, etc.
For Rich markup: Use [$variable] syntax directly, e.g., "[$accent]text[/]"

DO NOT hardcode colors. Let the theme system handle all color management.
"""

from dataclasses import dataclass


# Default theme name (Textual built-in)
DEFAULT_THEME = "gruvbox"


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
