from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlquant.combine import ML_METHODS, factor_weights


def _ic_history(months: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    index = pd.date_range("2019-01-31", periods=months, freq="ME")
    frame = pd.DataFrame(rng.normal(0.02, 0.05, (months, 5)), index=index)
    frame.columns = [f"F{i}" for i in range(5)]
    frame["F0"] = np.linspace(0.005, 0.05, months) + rng.normal(0, 0.008, months)
    return frame


def test_ml_weight_methods_normalize_nonnegative_and_cover_all_columns() -> None:
    history = _ic_history()
    for method in ML_METHODS:
        weights = factor_weights(history, method)
        assert list(weights.index) == list(history.columns)
        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= 0).all()


def test_ml_weight_methods_need_sufficient_history() -> None:
    with pytest.raises(ValueError, match="至少 16 个月"):
        factor_weights(_ic_history(months=12), "lasso")


def test_ml_weight_methods_reject_2026_history() -> None:
    rng = np.random.default_rng(0)
    index = pd.date_range("2025-01-31", periods=24, freq="ME")
    history = pd.DataFrame(rng.normal(0.02, 0.05, (24, 4)), index=index)
    with pytest.raises(ValueError, match="2026 数据不得参与合成选模"):
        factor_weights(history, "xgb")
