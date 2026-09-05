from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlquant.portfolio_evaluation import (
    CostScenario,
    _prepared_market_on,
    benchmark_returns,
    prepare_market,
    simulate_long_only,
)


def market_fixture():
    dates = pd.bdate_range("2024-01-01", periods=6)
    daily = pd.DataFrame({
        "trade_date": dates, "symbol": "A", "adj_open": 10.,
        "float_market_cap": 1000.,
    })
    status = pd.DataFrame({
        "trade_date": dates, "symbol": "A", "is_suspended": False,
        "limit_up": False, "limit_down": False,
    })
    calendar = pd.DataFrame({"trade_date": dates, "is_open": True})
    index = pd.MultiIndex.from_product([dates[[0, 2, 4]], ["A"]], names=["signal_date", "symbol"])
    return daily, status, calendar, pd.Series(1., index=index)


def test_false_limit_flag_is_not_a_zero_price_limit() -> None:
    daily, status, _, _ = market_fixture()
    market = _prepared_market_on(prepare_market(daily, status), daily.trade_date.iloc[0])
    assert pd.isna(market.limit_up.iloc[0])
    assert pd.isna(market.limit_down.iloc[0])


def test_flat_market_includes_initial_fees_and_skips_terminal_rebalance() -> None:
    daily, status, calendar, score = market_fixture()
    net, trades, _ = simulate_long_only(score, daily, status, calendar, top_n=1, scenario=CostScenario("net", 5))
    gross, _, _ = simulate_long_only(score, daily, status, calendar, top_n=1, scenario=CostScenario("gross", 0, False))
    assert np.allclose(gross, 0)
    assert net.iloc[0] < 0
    assert trades.trade_date.max() < calendar.trade_date.iloc[-1]


def test_benchmark_does_not_remove_missing_future_loser_and_renormalize() -> None:
    daily, _, calendar, score = market_fixture()
    daily.loc[daily.index[3], "adj_open"] = np.nan
    with pytest.raises(ValueError, match="cannot discard constituents"):
        benchmark_returns(score.to_frame("model"), daily, calendar)


def test_missing_held_security_mark_is_not_zero() -> None:
    daily, status, calendar, score = market_fixture()
    daily.loc[daily.index[3], "adj_open"] = np.nan
    with pytest.raises(ValueError, match="lacks a mark"):
        simulate_long_only(score, daily, status, calendar, top_n=1, scenario=CostScenario("gross", 0, False))
