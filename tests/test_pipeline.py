from __future__ import annotations

import pandas as pd

from alphagym.audit import assert_selection_isolation, assign_period
from alphagym.pipeline import month_end_signal_dates, run_long_only_backtest


def test_period_partition_excludes_2026() -> None:
    dates = pd.Series(pd.to_datetime(["2020-12-31", "2022-01-01", "2025-12-31", "2026-01-01"]))
    assert assign_period(dates).tolist() == ["development", "validation", "test", pd.NA]


def test_test_period_cannot_select_parameters() -> None:
    try:
        assert_selection_isolation(pd.Series(pd.to_datetime(["2024-01-01"])))
    except ValueError as error:
        assert "cannot select" in str(error)
    else:
        raise AssertionError("test selection was not blocked")


def test_month_end_uses_last_open_day(bundle) -> None:
    dates = month_end_signal_dates(bundle.calendar, "2024-01-01", "2024-03-31")
    assert dates == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-29")]


def test_pipeline_executes_strictly_next_open(bundle) -> None:
    signal = pd.Timestamp("2024-06-28")
    targets = pd.DataFrame(
        {
            "signal_date": signal,
            "symbol": ["000001.SZ"],
            "layer": [1],
            "target_weight": [1.0],
        }
    )
    result = run_long_only_backtest(bundle, targets, initial_cash=100_000, slippage_bps=0)
    assert result.trades["trade_date"].min() > signal
