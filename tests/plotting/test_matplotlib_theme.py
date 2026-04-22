import pytest


matplotlib = pytest.importorskip("matplotlib")

from skcausal.plotting import get_theme, theme_context, use_theme


def test_get_theme_exposes_expected_defaults():
    theme = get_theme()

    assert theme["axes.grid"] is True
    assert theme["axes.spines.top"] is False
    assert theme["legend.frameon"] is True
    assert "axes.prop_cycle" in theme


def test_use_theme_updates_global_rcparams():
    original = matplotlib.rcParams["axes.facecolor"]

    try:
        theme = use_theme({"axes.facecolor": "#123456"})
        assert theme["axes.facecolor"] == "#123456"
        assert matplotlib.rcParams["axes.facecolor"] == "#123456"
    finally:
        matplotlib.rcParams["axes.facecolor"] = original


def test_theme_context_restores_rcparams_after_exit():
    original = matplotlib.rcParams["axes.facecolor"]

    with theme_context({"axes.facecolor": "#654321"}) as theme:
        assert theme["axes.facecolor"] == "#654321"
        assert matplotlib.rcParams["axes.facecolor"] == "#654321"

    assert matplotlib.rcParams["axes.facecolor"] == original
