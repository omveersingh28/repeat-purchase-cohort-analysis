"""Shared visual constants for the Matplotlib figures and the Streamlit dashboard.

Deliberately dependency-free so the dashboard can import it without Matplotlib. Colours are
explicit hex values (never theme defaults) so figures read the same in light and dark README
rendering. The categorical order is fixed and colour-blind-safe for adjacent series.
"""
from __future__ import annotations

SURFACE = "#fcfcfb"        # chart background
INK = "#0b0b0b"            # titles, direct labels
INK_SECONDARY = "#52514e"  # subtitles, axis labels, tick labels
MUTED = "#898781"          # annotations, notes
GRID = "#e1e0d9"           # hairline gridlines
AXIS = "#c3c2b7"           # baseline / axis rules
DEEMPHASIS = "#b9b8b0"     # context-only marks (e.g. the left-censored first cohort)

#: Categorical slots, always assigned in this order (max 6 series per chart).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
PRIMARY, ACCENT = SERIES[0], SERIES[1]

#: One-hue sequential ramp (light -> dark) for the retention heatmap.
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
              "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

#: Ordered three-step ramp for Low / Mid / High tiers.
ORDINAL3 = ["#86b6ef", "#2a78d6", "#104281"]

FONT_FAMILY = ["Arial", "Helvetica", "DejaVu Sans"]
FONT_CSS = "Arial, Helvetica, sans-serif"

SEGMENT_LABELS = {"Region": "Region", "FirstOrderTier": "First-order value tier",
                  "IsQ4Cohort": "Acquisition quarter", "CohortYear": "Cohort year"}
SEGMENT_VALUE_LABELS = {("IsQ4Cohort", "True"): "Q4 cohort (Oct-Dec)",
                        ("IsQ4Cohort", "False"): "Other cohorts (Jan-Sep)"}


def segment_value_label(segment_type: str, value: str) -> str:
    """Human-readable label for a segment value (e.g. IsQ4Cohort=True -> 'Q4 cohort (Oct-Dec)')."""
    return SEGMENT_VALUE_LABELS.get((segment_type, str(value)), str(value))
