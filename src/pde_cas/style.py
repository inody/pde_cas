"""Shared plotting style for manuscript figures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
from cycler import cycler

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Okabe--Ito-inspired, color-vision-deficiency-friendly palette.
BLACK = "#000000"
GRAY = "#6F6F6F"
LIGHT_GRAY = "#B8B8B8"
BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
YELLOW = "#F0E442"

PALETTE = (BLUE, VERMILLION, GREEN, ORANGE, PURPLE, SKY, BLACK)
METHOD_COLORS = {
    "uniform": GRAY,
    "whole_domain": ORANGE,
    "vcs": BLUE,
    "aecs": VERMILLION,
}

# Production dimensions of elsarticle's 5p two-column layout, in inches.
# TeX reports 522 pt for the text block and 252 pt for one column.
TEX_POINTS_PER_INCH = 72.27
TEXT_WIDTH_IN = 522.0 / TEX_POINTS_PER_INCH
COLUMN_WIDTH_IN = 252.0 / TEX_POINTS_PER_INCH

# Axes (not canvas) dimensions are identical in one- and two-column figures.
# The 3:2 aspect also preserves the physical proportions of the 3-by-2 room.
WIDE_WIDTH_FRACTION = 0.96
PANEL_WIDTH_IN = 2.60
PANEL_HEIGHT_IN = PANEL_WIDTH_IN * 2.0 / 3.0
PANEL_GAP_IN = 0.72
PANEL_ROW_GAP_IN = 0.42
_LEFT_MARGIN_IN = 0.74
_BOTTOM_MARGIN_IN = 0.45
_TOP_MARGIN_IN = 0.28
_TEX_PREAMBLE = r"\usepackage{amsmath,amssymb,bm}"


def manuscript_panels(
    nrows: int = 1,
    ncols: int = 1,
    *,
    sharex: bool = False,
    sharey: bool = False,
) -> tuple[Figure, np.ndarray]:
    """Return manuscript axes without changing the font/line scale.

    Three-column diagnostics keep the two-column canvas width and use square
    axes. Thus LaTeX applies the same font/line scale factor.

    Include wide PDFs at the text width, and single-panel PDFs at one
    column's width.  Do not use tight cropping or automatic layout: their
    label-dependent resizing would undo the shared panel dimensions.
    """
    from matplotlib import pyplot as plt

    if nrows not in (1, 2) or ncols not in (1, 2, 3):
        raise ValueError("manuscript panels support one or two rows and up to three columns")
    width = TEXT_WIDTH_IN * WIDE_WIDTH_FRACTION if ncols >= 2 else COLUMN_WIDTH_IN
    panel_width = (2 * PANEL_WIDTH_IN - PANEL_GAP_IN) / 3 if ncols == 3 else PANEL_WIDTH_IN
    axes_width = ncols * panel_width + (ncols - 1) * PANEL_GAP_IN
    panel_height = panel_width if ncols == 3 else PANEL_HEIGHT_IN
    axes_height = nrows * panel_height + (nrows - 1) * PANEL_ROW_GAP_IN
    # Narrow diagnostics need room above the axes for scientific-notation offsets.
    top_margin = 0.45 if ncols == 3 else _TOP_MARGIN_IN
    height = _BOTTOM_MARGIN_IN + axes_height + top_margin
    return plt.subplots(
        nrows,
        ncols,
        figsize=(width, height),
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
        layout="none",
        gridspec_kw={
            "left": _LEFT_MARGIN_IN / width,
            "right": (_LEFT_MARGIN_IN + axes_width) / width,
            "bottom": _BOTTOM_MARGIN_IN / height,
            "top": (_BOTTOM_MARGIN_IN + axes_height) / height,
            "wspace": PANEL_GAP_IN / panel_width,
            "hspace": PANEL_ROW_GAP_IN / panel_height,
        },
    )




def quantitative_grid(axis: Axes) -> None:
    """Use the same light major grid and fainter minor grid on every scale."""
    axis.grid(True, which="major", color=LIGHT_GRAY, alpha=0.35, lw=0.6, ls="-")
    axis.grid(True, which="minor", color=LIGHT_GRAY, alpha=0.15, lw=0.45, ls="-")






def apply_manuscript_style() -> None:
    """Apply the common vector-figure style used throughout the manuscript."""
    matplotlib.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": _TEX_PREAMBLE,
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
            "pgf.preamble": _TEX_PREAMBLE,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "mathtext.fontset": "cm",
            "font.size": 9.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "axes.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.grid": False,
            "axes.prop_cycle": cycler(color=PALETTE),
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "legend.fontsize": 7.5,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": LIGHT_GRAY,
            "lines.linewidth": 1.7,
            "lines.markersize": 4.5,
            "grid.color": LIGHT_GRAY,
            "grid.alpha": 0.35,
            "grid.linewidth": 0.6,
            "figure.constrained_layout.wspace": 0.12,
            "figure.constrained_layout.hspace": 0.10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": None,
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
