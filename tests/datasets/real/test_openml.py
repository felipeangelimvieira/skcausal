"""Tests for the OpenML loading source.

Everything here is offline except ``test_real_fetch``, which is opt-in.
The fetch itself is stubbed so the mapping can be tested without a network.
"""

import os

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.utils import Bunch

from skcausal.datasets import OpenMLDataset
from skcausal.datasets.real import openml as openml_module


def _bunch(target, features=None):
    if features is None:
        features = pd.DataFrame(
            {
                "x": np.arange(6, dtype=float),
                "c": pd.Categorical(["a", "b", "c", "a", "b", "c"]),
            }
        )
    return Bunch(data=features, target=target)


@pytest.fixture
def stub_fetch(monkeypatch):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return fake.bunch

    fake.bunch = _bunch(pd.Series(np.linspace(0.0, 1.0, 6), name="y"))
    monkeypatch.setattr(openml_module, "fetch_openml", fake)
    return fake, calls


def test_rejects_an_incomplete_specification():
    with pytest.raises(ValueError, match="name or data_id"):
        OpenMLDataset()
    with pytest.raises(ValueError, match="target_kind must be one of"):
        OpenMLDataset(name="adult", target_kind="ordinal")


def test_load_maps_through_the_shared_adapter_and_construction_does_no_io(stub_fetch):
    _, calls = stub_fetch

    dataset = OpenMLDataset(name="adult")
    assert calls == []

    covariates, outcomes = dataset.load()

    assert covariates.columns == ["x", "c"]
    assert outcomes.columns == ["y"]
    assert covariates.height == outcomes.height == 6


def test_the_identifier_and_target_reach_the_fetch(stub_fetch):
    _, calls = stub_fetch

    OpenMLDataset(name="adult", version=2, target="income", data_home="/tmp/x").load()
    OpenMLDataset(data_id=42).load()

    assert calls[0]["name"] == "adult"
    assert calls[0]["version"] == 2
    assert calls[0]["target_column"] == "income"
    assert calls[0]["data_home"] == "/tmp/x"
    assert calls[0]["as_frame"] is True
    assert calls[1]["data_id"] == 42
    assert calls[1]["target_column"] == "default-target"


@pytest.mark.parametrize(
    "target, categorical",
    [
        (pd.Series(pd.Categorical(["a", "b", "a", "b", "a", "b"])), True),
        (pd.Series(np.linspace(0.0, 1.0, 6)), False),
    ],
)
def test_target_kind_is_read_from_the_fetched_dtype(monkeypatch, target, categorical):
    monkeypatch.setattr(
        openml_module, "fetch_openml", lambda **kw: _bunch(target.rename("y"))
    )

    _, outcomes = OpenMLDataset(name="d").load()

    assert (outcomes.get_column("y").dtype == pl.Utf8) is categorical


def test_explicit_target_kind_overrides_inference(monkeypatch):
    # A classification target stored as integer codes is exactly the case the
    # dtype cannot see.
    codes = pd.Series([0, 1, 2, 0, 1, 2], name="y")
    monkeypatch.setattr(openml_module, "fetch_openml", lambda **kw: _bunch(codes))

    _, outcomes = OpenMLDataset(name="d", target_kind="categorical").load()

    assert outcomes.get_column("y").dtype == pl.Utf8


def test_multi_column_target_is_rejected(monkeypatch):
    frame = pd.DataFrame({"a": np.arange(6.0), "b": np.arange(6.0)})
    monkeypatch.setattr(openml_module, "fetch_openml", lambda **kw: _bunch(frame))

    with pytest.raises(ValueError, match="target columns"):
        OpenMLDataset(name="d").load()


def test_fetch_failure_is_reported_in_contract_terms(monkeypatch):
    def boom(**kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(openml_module, "fetch_openml", boom)

    with pytest.raises(ValueError, match="Could not load dataset 'nope'") as excinfo:
        OpenMLDataset(name="nope").load()

    assert isinstance(excinfo.value.__cause__, OSError)


@pytest.mark.skipif(
    not os.environ.get("SKCAUSAL_TEST_OPENML"),
    reason="set SKCAUSAL_TEST_OPENML=1 to exercise a real OpenML fetch",
)
def test_real_fetch():
    covariates, outcomes = OpenMLDataset(name="wine-quality-red", version=1).load()

    assert covariates.height == outcomes.height
    assert covariates.width > 1
    assert outcomes.width == 1
