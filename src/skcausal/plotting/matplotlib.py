"""Matplotlib styling helpers for skcausal examples and reports.

This module keeps plotting support optional. Importing the module does not
require matplotlib, but calling ``use_theme`` or ``theme_context`` does.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Mapping
from skcausal.datatypes import collect_column_types, convert
from skcausal.datatypes._typing import DataFrameLike
import numpy as np


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


def use_theme(overrides: Mapping[str, object] | None = None) -> None:
    """Apply the skcausal matplotlib theme globally.

    Parameters
    ----------
    overrides : mapping, optional
        Extra rcParams entries to merge into the applied theme.
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


def _coerce_curve_values(
    y: dict[str, np.ndarray], n_rows: int
) -> dict[str, np.ndarray]:
    if not y:
        raise ValueError("Expected at least one curve in y.")

    curves = {}
    for label, values in y.items():
        curve = np.asarray(values, dtype=float)
        if curve.ndim == 0:
            raise ValueError(
                f"Curve '{label}' must be array-like with one value per treatment row."
            )
        if curve.shape[0] != n_rows:
            raise ValueError(
                f"Curve '{label}' has {curve.shape[0]} rows, expected {n_rows}."
            )

        curve = curve.reshape(n_rows, -1)
        if curve.shape[1] != 1:
            raise ValueError(
                f"Curve '{label}' must be one-dimensional or single-output, got shape {curve.shape}."
            )
        curves[label] = curve[:, 0]

    return curves


def _coerce_axes(ax, n_axes: int):
    mpl = _require_matplotlib()

    if ax is None:
        _, axes = mpl.pyplot.subplots(1, n_axes, squeeze=False, sharey=True)
        axes = axes[0]
    elif n_axes == 1:
        axes = np.array([ax], dtype=object)
    else:
        axes = np.asarray(ax, dtype=object).reshape(-1)
        if axes.size != n_axes:
            raise ValueError(f"Expected {n_axes} axes, got {axes.size}.")

    return axes


def _groupby_mean(frame, group_columns: list[str], value_column: str):
    return (
        frame.groupby(group_columns, observed=True, sort=False)[value_column]
        .mean()
        .reset_index()
    )


def _plot_series(
    ax,
    x,
    y,
    *,
    is_categorical: bool,
    label: str | None = None,
    **plot_kwargs,
):
    if is_categorical:
        labels = [str(value) for value in x]
        positions = np.arange(len(labels), dtype=float)
        ax.plot(positions, y, marker="o", label=label, **plot_kwargs)
        ax.set_xticks(positions, labels)
        return

    ax.plot(np.asarray(x, dtype=float), y, marker="o", label=label, **plot_kwargs)


def _joint_group_plot_kwargs(*, base_color: str | None, group_index: int) -> dict:
    if base_color is None:
        return {}

    line_styles = ("-", "--", "-.", ":")
    return {
        "color": base_color,
        "linestyle": line_styles[group_index % len(line_styles)],
    }


def _get_model_colors(reference_axis, model_labels: list[str]) -> dict[str, str]:
    return {
        model_label: reference_axis._get_lines.get_next_color()
        for model_label in model_labels
    }


def _format_group_label(group_columns: list[str], group_key) -> str:
    if not isinstance(group_key, tuple):
        group_key = (group_key,)
    return ", ".join(
        f"{column}={value}" for column, value in zip(group_columns, group_key)
    )


def _combine_labels(*parts: str | None) -> str | None:
    values = [part for part in parts if part]
    if not values:
        return None
    return " | ".join(values)


def plot_marginal_curves(t: DataFrameLike, y: dict[str, np.array], ax=None):
    """
    Plot curves of t vs y on the given matplotlib axis.

    If t is one-dimensional, it is plotted on the x-axis.
    If t is multi-dimensional: each dimension of t is plotted
    on a separate x-axis.

    `y` can contain curves of multiple models, the keys of `y` are used as
    labels in the legend.
    """
    t = convert(t, "pandas")
    column_types = collect_column_types(t)
    curves = _coerce_curve_values(y, len(t))
    axes = _coerce_axes(ax, len(t.columns))

    for axis, column in zip(axes, t.columns):
        is_categorical = column_types[column] == "categorical"
        summarized_curves = []
        for label, values in curves.items():
            plot_frame = t[[column]].copy()
            plot_frame["__value__"] = values
            plot_frame = _groupby_mean(plot_frame, [column], "__value__")

            if is_categorical:
                if hasattr(t[column], "cat"):
                    categories = [
                        category
                        for category in t[column].cat.categories
                        if category in set(plot_frame[column])
                    ]
                    plot_frame[column] = plot_frame[column].astype(str)
                    plot_frame["__order__"] = plot_frame[column].map(
                        {
                            str(category): index
                            for index, category in enumerate(categories)
                        }
                    )
                    plot_frame = plot_frame.sort_values("__order__")
                else:
                    plot_frame[column] = plot_frame[column].astype(str)
                    plot_frame = plot_frame.sort_values(column, kind="stable")
            else:
                plot_frame = plot_frame.sort_values(column, kind="stable")

            summarized_curves.append((label, plot_frame))

        if is_categorical:
            category_labels = summarized_curves[0][1][column].tolist()
            positions = np.arange(len(category_labels), dtype=float)
            n_curves = len(summarized_curves)
            width = 0.8 / max(n_curves, 1)

            for index, (label, plot_frame) in enumerate(summarized_curves):
                offsets = positions + (index - (n_curves - 1) / 2.0) * width
                axis.bar(
                    offsets,
                    plot_frame["__value__"].to_numpy(dtype=float),
                    width=width,
                    label=label,
                    align="center",
                )

            axis.set_xticks(positions, [str(value) for value in category_labels])
        else:
            for label, plot_frame in summarized_curves:
                _plot_series(
                    axis,
                    plot_frame[column].to_numpy(),
                    plot_frame["__value__"].to_numpy(dtype=float),
                    is_categorical=False,
                    label=label,
                )

        axis.set_xlabel(column)
        axis.set_title(column)

    axes[0].set_ylabel("Average response")
    if len(curves) > 1:
        axes[0].legend()

    return axes[0] if len(axes) == 1 else axes


def plot_joint_curves(
    t: DataFrameLike,
    y: dict[str, np.array],
    ax=None,
    *,
    separate_axes_by_group: bool = False,
):
    """
    Plot curves of t vs y on the given matplotlib axis.

    This function only supports t's with a single continuous treatment.
    It groups-by every categorical column and plot a single-line for each
    group.

    When ``separate_axes_by_group`` is True, each categorical group is drawn
    on its own axis while model colors stay aligned across axes.
    """
    t = convert(t, "pandas")
    column_types = collect_column_types(t)
    curves = _coerce_curve_values(y, len(t))

    continuous_columns = [
        column
        for column, column_type in column_types.items()
        if column_type == "continuous"
    ]
    if len(continuous_columns) != 1:
        raise ValueError(
            "plot_joint_curves requires exactly one continuous treatment column."
        )

    continuous_column = continuous_columns[0]
    categorical_columns = [
        column
        for column, column_type in column_types.items()
        if column_type == "categorical"
    ]

    grouped_frames_by_model = {}
    group_keys: list[object] = []
    if categorical_columns:
        group_columns = [*categorical_columns, continuous_column]
        for model_label, values in curves.items():
            plot_frame = t.copy()
            plot_frame["__value__"] = values
            plot_frame = _groupby_mean(plot_frame, group_columns, "__value__")
            grouped = [
                (
                    group_key,
                    group_frame.sort_values(continuous_column, kind="stable"),
                )
                for group_key, group_frame in plot_frame.groupby(
                    categorical_columns, observed=True, sort=False
                )
            ]
            if not group_keys:
                group_keys = [group_key for group_key, _ in grouped]
            grouped_frames_by_model[model_label] = dict(grouped)

    use_group_axes = separate_axes_by_group and bool(categorical_columns)
    axes = _coerce_axes(ax, len(group_keys) if use_group_axes else 1)
    model_colors = _get_model_colors(axes[0], list(curves)) if use_group_axes else None

    if use_group_axes:
        for group_axis, group_key in zip(axes, group_keys):
            for model_label in curves:
                group_frame = grouped_frames_by_model[model_label][group_key]
                _plot_series(
                    group_axis,
                    group_frame[continuous_column].to_numpy(dtype=float),
                    group_frame["__value__"].to_numpy(dtype=float),
                    is_categorical=False,
                    label=model_label if len(curves) > 1 else None,
                    color=model_colors[model_label],
                )

            group_axis.set_xlabel(continuous_column)
            group_axis.set_title(_format_group_label(categorical_columns, group_key))

        axes[0].set_ylabel("Average response")
        if len(curves) > 1:
            axes[0].legend()

        return axes[0] if len(axes) == 1 else axes

    axis = axes[0]

    for model_label, values in curves.items():
        if categorical_columns:
            grouped = [
                (group_key, grouped_frames_by_model[model_label][group_key])
                for group_key in group_keys
            ]
            base_color = axis._get_lines.get_next_color() if len(curves) > 1 else None

            for group_index, (group_key, group_frame) in enumerate(grouped):
                _plot_series(
                    axis,
                    group_frame[continuous_column].to_numpy(dtype=float),
                    group_frame["__value__"].to_numpy(dtype=float),
                    is_categorical=False,
                    label=_combine_labels(
                        model_label if len(curves) > 1 else None,
                        _format_group_label(categorical_columns, group_key),
                    ),
                    **_joint_group_plot_kwargs(
                        base_color=base_color,
                        group_index=group_index,
                    ),
                )
        else:
            plot_frame = t.copy()
            plot_frame["__value__"] = values
            plot_frame = _groupby_mean(plot_frame, [continuous_column], "__value__")
            plot_frame = plot_frame.sort_values(continuous_column, kind="stable")
            _plot_series(
                axis,
                plot_frame[continuous_column].to_numpy(dtype=float),
                plot_frame["__value__"].to_numpy(dtype=float),
                is_categorical=False,
                label=model_label,
            )

    axis.set_xlabel(continuous_column)
    axis.set_ylabel("Average response")

    if len(axis.lines) > 1:
        axis.legend()

    return axis
