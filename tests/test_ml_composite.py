from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlquant.ml_composite import (
    CompositeDataset,
    frozen_composite,
    load_panel_wide,
    neutralize_wide,
    run_composites,
    walk_forward_composite,
)

SPLITS = {
    "development": (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31")),
    "validation": (pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31")),
    "test": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")),
}


def _synthetic_panel(months: int = 48, symbols: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.date_range("2021-01-31", periods=months, freq="ME")
    rows = []
    for position, date in enumerate(dates):
        for symbol in range(symbols):
            code = f"{symbol:06d}"
            forward = float(rng.normal(0.005, 0.06)) if position < months - 1 else np.nan
            for factor in ("F_A", "F_B"):
                value = float(rng.normal(0, 1)) + 0.5 * forward * 20
                rows.append({
                    "signal_date": date, "symbol": code, "factor_name": factor,
                    "raw_value": value, "neutralized_value": np.nan,
                    "forward_return": forward, "index_code": "ALL_A",
                })
    return pd.DataFrame(rows)


def _context(dataset: CompositeDataset) -> tuple[pd.DataFrame, pd.Series]:
    dates = dataset.wide_raw.index.get_level_values("signal_date").unique()
    industries = pd.DataFrame({
        "symbol": dataset.wide_raw.index.get_level_values("symbol").unique()[:20],
        "industry_code": "801010.SI",
        "valid_from": pd.Timestamp("2020-01-01"),
        "valid_to": pd.NaT,
    })
    daily = pd.DataFrame({
        "trade_date": np.repeat(dates, 80),
        "symbol": np.tile([f"{i:06d}" for i in range(80)], len(dates)),
        "float_market_cap": np.tile(np.linspace(10, 1000, 80), len(dates)),
    })
    cap = daily.set_index(["trade_date", "symbol"])["float_market_cap"]
    return industries, cap


def test_load_panel_wide_pivots_and_aligns_forward() -> None:
    panel = _synthetic_panel()
    dataset = load_panel_wide(panel, ["F_A", "F_B"], "ALL_A")
    assert list(dataset.wide_raw.columns) == ["F_A", "F_B"]
    assert dataset.wide_raw.index.names == ["signal_date", "symbol"]
    assert dataset.forward.index.equals(dataset.wide_raw.index)
    assert dataset.forward.dtype == np.float32


def test_frozen_composite_trains_only_on_development() -> None:
    panel = _synthetic_panel()
    dataset = load_panel_wide(panel, ["F_A", "F_B"], "ALL_A")
    features = dataset.wide_raw.groupby(level="signal_date", observed=True).transform("mean")
    prediction, meta = frozen_composite(features, dataset.forward, SPLITS, "lasso")
    assert meta["train_start"] == "2021-01-01"
    assert meta["train_end"] == "2021-12-31"
    assert meta["protocol"] == "frozen"
    months = sorted(pd.to_datetime(dataset.wide_raw.index.get_level_values("signal_date").unique()))
    assert prediction.index.get_level_values("signal_date").max() == months[-1]


def test_frozen_composite_rejects_training_on_test_dates() -> None:
    panel = _synthetic_panel()
    dataset = load_panel_wide(panel, ["F_A", "F_B"], "ALL_A")
    features = dataset.wide_raw.groupby(level="signal_date", observed=True).transform("mean")
    bad_splits = {
        "development": (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31")),
        "validation": SPLITS["validation"], "test": SPLITS["test"],
    }
    with pytest.raises(ValueError, match="test/2026"):
        frozen_composite(features, dataset.forward, bad_splits, "lasso")


def test_walk_forward_leaves_early_months_unpredicted_then_refits() -> None:
    panel = _synthetic_panel()
    dataset = load_panel_wide(panel, ["F_A", "F_B"], "ALL_A")
    features = dataset.wide_raw.groupby(level="signal_date", observed=True).transform("mean")
    prediction, meta = walk_forward_composite(
        features, dataset.forward, SPLITS, "lasso", refit_months=12,
    )
    assert meta["protocol"] == "walk_forward"
    assert meta["refits"] >= 1
    monthly = prediction.groupby(level="signal_date").apply(
        lambda block: block.notna().all(), include_groups=False
    )
    assert not monthly.iloc[:12].any()
    assert monthly.iloc[-12:].all()
    # test-period rows (2024) are scored by the frozen selection model, not refit
    last_refit = pd.Timestamp(meta["last_refit"])
    assert last_refit <= SPLITS["validation"][1]


def test_neutralize_wide_residual_shape_and_finite() -> None:
    panel = _synthetic_panel()
    dataset = load_panel_wide(panel, ["F_A", "F_B"], "ALL_A")
    industries, cap = _context(dataset)
    result = neutralize_wide(dataset.wide_raw, industries, cap)
    assert result.shape == dataset.wide_raw.shape
    assert result.index.equals(dataset.wide_raw.index)
    # rows lacking industry/cap stay NaN rather than fabricating residuals
    no_context = ~result.index.get_level_values("symbol").isin(
        industries["symbol"].to_numpy()
    )
    assert result.loc[no_context].isna().all().all()


def test_run_composites_smoke() -> None:
    panel = _synthetic_panel()
    dataset = load_panel_wide(panel, ["F_A", "F_B"], "ALL_A")
    industries, cap = _context(dataset)
    results = run_composites(
        dataset, industries, cap, SPLITS, models=("lasso", "xgb"),
        feature_mode="z", label_mode="return", protocol="frozen",
    )
    assert [item.meta["model_key"] for item in results] == ["lasso", "xgb"]
    for item in results:
        assert not item.monthly.empty
        assert {"development", "validation", "test"} <= set(
            item.monthly["period"].dropna().unique()
        )
