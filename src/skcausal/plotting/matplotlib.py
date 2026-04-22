"""Matplotlib styling helpers for skcausal examples and reports.

This module keeps plotting support optional. Importing the module does not
require matplotlib, but calling ``use_theme`` or ``theme_context`` does.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Mapping


def _build_theme() -> dict[str, object]:
    from cycler import cycler

    return {
        "figure.facecolor": "#F4F7FB",
        "figure.figsize": (8.0, 4.8),
        "savefig.facecolor": "#F4F7FB",
        "axes.facecolor": "#FBFCFE",
        "axes.edgecolor": "#D7E0EA",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlecolor": "#0F172A",
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.titlepad": 12.0,
        "axes.labelcolor": "#1F2937",
        "axes.labelsize": 11.5,
        "axes.prop_cycle": cycler(
            color=[
                "#0F766E",
                "#F97316",
                "#2563EB",
                "#D9465F",
                "#CA8A04",
                "#16A34A",
            ]
        ),
        "grid.color": "#CBD5E1",
        "grid.alpha": 0.55,
        "grid.linewidth": 0.8,
        "grid.linestyle": "--",
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.facecolor": "#FFFFFF",
        "legend.edgecolor": "#D7E0EA",
        "legend.fancybox": True,
        "legend.borderpad": 0.6,
        "font.size": 11.0,
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Avenir Next",
            "Avenir",
            "Helvetica Neue",
            "Arial",
            "DejaVu Sans",
        ],
        "text.color": "#1F2937",
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "lines.linewidth": 2.4,
        "lines.markersize": 6.5,
        "patch.linewidth": 0.75,
        "patch.edgecolor": "#FFFFFF",
        "errorbar.capsize": 3.0,
        "axes.formatter.use_mathtext": True,
    }


def _require_matplotlib():
    try:
        import matplotlib as mpl
    except ImportError as exc:
        raise ImportError(
            "Matplotlib is required for skcausal.plotting. Install matplotlib "
            "directly or use the plotting extra: pip install 'skcausal[plotting]'."
        ) from exc

    return mpl


def get_theme(overrides: Mapping[str, object] | None = None) -> dict[str, object]:
    """Return the default skcausal matplotlib theme.

    Parameters
    ----------
    overrides : mapping, optional
        Extra rcParams entries to merge into the returned theme.
    """

    theme = _build_theme()
    if overrides:
        theme.update(dict(overrides))
    return theme


def use_theme(overrides: Mapping[str, object] | None = None) -> dict[str, object]:
    """Apply the skcausal matplotlib theme globally.

    Parameters
    ----------
    overrides : mapping, optional
        Extra rcParams entries to merge into the applied theme.

    Returns
    -------
    dict[str, object]
        The theme dictionary that was applied.
    """

    mpl = _require_matplotlib()
    theme = get_theme(overrides)
    mpl.rcParams.update(theme)


@contextmanager
def theme_context(
    overrides: Mapping[str, object] | None = None,
) -> Iterator[dict[str, object]]:
    """Temporarily apply the skcausal matplotlib theme inside a context."""

    mpl = _require_matplotlib()
    theme = get_theme(overrides)
    with mpl.rc_context(rc=theme):
        yield theme
