from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mlquant import storage_io
from mlquant.combine import (
    METHODS,
    factor_weights,
    select_low_correlation_factors,
    select_validation_method,
)
from mlquant.reporting import build_series


def test_all_combine_methods_are_long_only() -> None:
    rng = np.random.default_rng(4)
    history = pd.DataFrame(rng.normal(size=(48, 4)), columns=list("ABCD"))
    for method in METHODS:
        weights = factor_weights(history, method)
        assert np.isclose(weights.sum(), 1)
        assert (weights >= 0).all()


def test_validation_tie_prefers_lower_turnover() -> None:
    metrics = pd.DataFrame({"method": ["equal", "pca"], "net_information_ratio": [1.0, .97], "turnover": [.5, .2]})
    assert select_validation_method(metrics) == "pca"


def test_spearman_filter_keeps_stronger_factor_from_correlated_pair() -> None:
    correlation = pd.DataFrame(
        [[1.0, 0.91, 0.15], [0.91, 1.0, 0.10], [0.15, 0.10, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    priority = pd.Series({"A": 0.02, "B": -0.04, "C": 0.01})

    assert select_low_correlation_factors(correlation, priority) == ["B", "C"]


def test_report_series_has_nine_reports_and_manifest(tmp_path) -> None:
    manifest_path = build_series(tmp_path, smoke=True, include_pdf=True)
    manifest = json.loads(storage_io.read_text(manifest_path, encoding="utf-8"))
    assert manifest["factor_count"] == 169
    assert manifest["watermark"] == "NON-FORMAL SMOKE"
    assert len(list(storage_io.glob(tmp_path, "*.md"))) == 9
    assert len(list(storage_io.glob(tmp_path, "*.pdf"))) == 9
    assert storage_io.exists(tmp_path / "factor_catalog.csv")
    assert storage_io.exists(tmp_path / "factor_catalog.parquet")
    assert storage_io.exists(tmp_path / "factor_catalog.html")
    assert not list(storage_io.glob(tmp_path, "*.png"))
