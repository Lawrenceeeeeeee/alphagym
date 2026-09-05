from __future__ import annotations

import numpy as np
import pandas as pd

from mlquant.ml_frequency import (
    add_financial_features,
    financial_events,
    forward_label_end_dates,
    forward_open_returns,
    frequency_signal_dates,
    market_features,
)
from mlquant.portfolio_evaluation import performance_stats


def test_frequency_signal_dates_use_last_open_day() -> None:
    calendar = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-08"]
            ),
            "is_open": [False, True, True, True],
        }
    )
    weekly = frequency_signal_dates(calendar, "2024-01-01", "2024-01-08", "weekly")
    assert weekly.tolist() == [pd.Timestamp("2024-01-05")]


def test_unfinished_week_does_not_become_a_signal_when_window_is_truncated() -> None:
    calendar = pd.DataFrame({"trade_date": pd.bdate_range("2024-01-01", "2024-02-02"), "is_open": True})
    short = frequency_signal_dates(calendar, "2024-01-01", "2024-01-08", "weekly")
    longer = frequency_signal_dates(calendar, "2024-01-01", "2024-01-31", "weekly")
    assert short.equals(longer[longer <= pd.Timestamp("2024-01-08")])


def test_financial_late_filing_and_future_revision_cannot_rewrite_history() -> None:
    rows = [
        ("2023-03-31", "2023-04-20", 30, 3),
        ("2024-03-31", "2024-04-20", 50, 5),
        ("2023-12-31", "2024-04-30", 140, 14),
        ("2023-03-31", "2024-05-20", 40, 4),
    ]
    frame = pd.DataFrame(rows, columns=["stat_date", "available_date", "revenue", "bps"])
    frame["symbol"] = "000001.SZ"
    for column in ("ocfps", "net_profit", "roe", "gross_margin"):
        frame[column] = frame.revenue
    full = financial_events(frame).set_index("available_date")
    prefix = financial_events(frame.iloc[:2]).set_index("available_date")
    pd.testing.assert_frame_equal(full.loc[prefix.index], prefix)
    assert pd.isna(full.loc["2024-04-20", "revenue_ttm"])
    assert full.loc["2024-04-30", "revenue_ttm"] == 160
    assert full.loc["2024-04-30", "bps"] == 5  # late annual filing must not replace Q1
    assert full.loc["2024-05-20", "revenue_ttm"] == 150


def test_training_cutoff_uses_label_end_not_signal_date() -> None:
    calendar = pd.DataFrame({"trade_date": pd.bdate_range("2020-11-01", "2021-02-05"), "is_open": True})
    signals = pd.to_datetime(["2020-11-30", "2020-12-31", "2021-01-29"])
    ends = forward_label_end_dates(signals, calendar)
    assert ends.iloc[0] == pd.Timestamp("2021-01-01")
    assert not (ends <= pd.Timestamp("2020-12-31")).any()


def test_financial_features_only_change_after_available_date() -> None:
    dates = pd.to_datetime(["2024-04-01", "2024-06-03"])
    observations = pd.DataFrame(
        {
            "signal_date": dates,
            "symbol": ["000001.SZ", "000001.SZ"],
            "close": [10.0, 10.0],
            "float_market_cap": [100.0, 100.0],
        }
    ).set_index(["signal_date", "symbol"])
    market = pd.DataFrame(index=observations.index)
    fundamentals = pd.DataFrame(
        {
            "symbol": ["000001.SZ"] * 6,
            "stat_date": pd.to_datetime(
                [
                    "2022-12-31",
                    "2023-03-31",
                    "2023-06-30",
                    "2023-09-30",
                    "2023-12-31",
                    "2024-03-31",
                ]
            ),
            "available_date": pd.to_datetime(
                [
                    "2023-03-01",
                    "2023-04-30",
                    "2023-08-31",
                    "2023-10-31",
                    "2024-03-01",
                    "2024-05-01",
                ]
            ),
            "bps": [8, 9, 10, 11, 12, 20],
            "revenue": [100, 30, 60, 90, 140, 50],
            "ocfps": [10, 3, 6, 9, 14, 5],
            "net_profit": [10, 3, 6, 9, 14, 5],
            "roe": [1, 2, 3, 4, 5, 6],
            "gross_margin": [10, 11, 12, 13, 14, 15],
        }
    )
    result = add_financial_features(
        market, observations, fundamentals, ["BP_MRQ", "REVENUE_YOY"]
    )
    assert result.loc[(pd.Timestamp("2024-04-01"), "000001.SZ"), "BP_MRQ"] == 1.2
    assert result.loc[(pd.Timestamp("2024-06-03"), "000001.SZ"), "BP_MRQ"] == 2.0
    assert np.isclose(
        result.loc[(pd.Timestamp("2024-04-01"), "000001.SZ"), "REVENUE_YOY"],
        140 / 100 - 1,
    )
    assert np.isclose(
        result.loc[(pd.Timestamp("2024-06-03"), "000001.SZ"), "REVENUE_YOY"],
        50 / 30 - 1,
    )


def test_market_reversal_and_forward_return_alignment() -> None:
    dates = pd.bdate_range("2024-01-01", periods=10)
    daily = pd.DataFrame(
        {
            "trade_date": dates,
            "symbol": "000001.SZ",
            "open": np.arange(10.0, 20.0),
            "high": np.arange(11.0, 21.0),
            "low": np.arange(9.0, 19.0),
            "close": np.arange(10.0, 20.0),
            "adj_open": np.arange(10.0, 20.0),
            "adj_high": np.arange(11.0, 21.0),
            "adj_low": np.arange(9.0, 19.0),
            "adj_close": np.arange(10.0, 20.0),
            "volume": 1000,
            "amount": 10_000.0,
            "turnover": 0.01,
            "float_market_cap": 1_000_000.0,
        }
    )
    signals = pd.DatetimeIndex([dates[5], dates[7]])
    features, _observations = market_features(daily, signals, ["REVERSAL_5D"])
    expected = -(15 / 10 - 1)
    assert np.isclose(features.iloc[0, 0], expected)
    calendar = pd.DataFrame({"trade_date": dates, "is_open": True})
    forward = forward_open_returns(daily, signals, calendar)
    assert np.isclose(forward.iloc[0], 18 / 16 - 1)


def test_performance_stats_report_geometric_excess() -> None:
    portfolio = pd.Series([0.02, 0.02])
    benchmark = pd.Series([0.01, 0.01])
    result = performance_stats(portfolio, benchmark, periods_per_year=2)
    assert np.isclose(result["annualized_return"], 0.0404)
    assert np.isclose(result["benchmark_annualized_return"], 0.0201)
    assert result["annualized_excess_return"] > 0.019


def test_drawdown_includes_first_period_loss() -> None:
    result = performance_stats(pd.Series([-0.2, 0.1]), pd.Series([0., 0.]), periods_per_year=2)
    assert np.isclose(result["max_drawdown"], -0.2)
    assert np.isclose(result["relative_max_drawdown"], -0.2)
