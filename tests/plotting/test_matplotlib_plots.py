import numpy as np
import polars as pl
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from skcausal.plotting.matplotlib import plot_joint_curves, plot_marginal_curves


def test_plot_marginal_curves_returns_one_axis_per_treatment_column():
    t = pl.DataFrame(
        {
            "dose": [0.0, 0.0, 1.0, 1.0],
            "group": ["a", "b", "a", "b"],
        }
    )
    curves = {"model": np.array([1.0, 3.0, 5.0, 7.0])}

    axes = plot_marginal_curves(t, curves)

    assert len(axes) == 2
    np.testing.assert_allclose(axes[0].lines[0].get_xdata(), np.array([0.0, 1.0]))
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), np.array([2.0, 6.0]))
    assert len(axes[1].containers) == 1
    categorical_bars = axes[1].containers[0]
    np.testing.assert_allclose(
        [bar.get_height() for bar in categorical_bars], np.array([3.0, 5.0])
    )
    assert [tick.get_text() for tick in axes[1].get_xticklabels()] == ["a", "b"]

    plt.close(axes[0].figure)


def test_plot_joint_curves_groups_by_categorical_treatment_columns():
    t = pl.DataFrame(
        {
            "dose": [0.0, 1.0, 0.0, 1.0],
            "arm": ["a", "a", "b", "b"],
        }
    )
    curves = {"model": np.array([1.0, 2.0, 4.0, 8.0])}

    axis = plot_joint_curves(t, curves)

    assert len(axis.lines) == 2
    labels = {line.get_label() for line in axis.lines}
    assert labels == {"arm=a", "arm=b"}
    line_by_label = {line.get_label(): line for line in axis.lines}
    np.testing.assert_allclose(line_by_label["arm=a"].get_xdata(), np.array([0.0, 1.0]))
    np.testing.assert_allclose(line_by_label["arm=a"].get_ydata(), np.array([1.0, 2.0]))
    np.testing.assert_allclose(line_by_label["arm=b"].get_ydata(), np.array([4.0, 8.0]))

    plt.close(axis.figure)


def test_plot_joint_curves_reuses_model_color_across_categorical_groups():
    t = pl.DataFrame(
        {
            "dose": [0.0, 1.0, 0.0, 1.0],
            "arm": ["a", "a", "b", "b"],
        }
    )
    curves = {
        "observed": np.array([1.0, 2.0, 4.0, 8.0]),
        "truth": np.array([1.5, 2.5, 4.5, 8.5]),
    }

    axis = plot_joint_curves(t, curves)

    line_by_label = {line.get_label(): line for line in axis.lines}

    assert (
        line_by_label["observed | arm=a"].get_color()
        == line_by_label["observed | arm=b"].get_color()
    )
    assert (
        line_by_label["truth | arm=a"].get_color()
        == line_by_label["truth | arm=b"].get_color()
    )
    assert (
        line_by_label["observed | arm=a"].get_color()
        != line_by_label["truth | arm=a"].get_color()
    )

    assert line_by_label["observed | arm=a"].get_linestyle() == "-"
    assert line_by_label["observed | arm=b"].get_linestyle() == "--"
    assert line_by_label["truth | arm=a"].get_linestyle() == "-"
    assert line_by_label["truth | arm=b"].get_linestyle() == "--"

    plt.close(axis.figure)


def test_plot_joint_curves_can_split_categorical_groups_across_axes():
    t = pl.DataFrame(
        {
            "dose": [0.0, 1.0, 0.0, 1.0],
            "arm": ["a", "a", "b", "b"],
        }
    )
    curves = {
        "observed": np.array([1.0, 2.0, 4.0, 8.0]),
        "truth": np.array([1.5, 2.5, 4.5, 8.5]),
    }

    axes = plot_joint_curves(t, curves, separate_axes_by_group=True)

    assert len(axes) == 2
    assert [axis.get_title() for axis in axes] == ["arm=a", "arm=b"]

    first_axis_lines = {line.get_label(): line for line in axes[0].lines}
    second_axis_lines = {line.get_label(): line for line in axes[1].lines}

    assert (
        first_axis_lines["observed"].get_color()
        == second_axis_lines["observed"].get_color()
    )
    assert (
        first_axis_lines["truth"].get_color() == second_axis_lines["truth"].get_color()
    )
    assert (
        first_axis_lines["observed"].get_color()
        != first_axis_lines["truth"].get_color()
    )

    np.testing.assert_allclose(
        second_axis_lines["observed"].get_ydata(), np.array([4.0, 8.0])
    )

    plt.close(axes[0].figure)


def test_plot_joint_curves_rejects_multiple_continuous_treatment_columns():
    t = pl.DataFrame({"dose": [0.0, 1.0], "dose_2": [1.0, 2.0]})

    with pytest.raises(
        ValueError,
        match="requires exactly one continuous treatment column",
    ):
        plot_joint_curves(t, {"model": np.array([1.0, 2.0])})
