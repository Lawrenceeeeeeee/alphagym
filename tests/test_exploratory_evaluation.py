from __future__ import annotations

import numpy as np
import pandas as pd

from mlquant.exploratory_evaluation import fractional_simulation


def test_fractional_flat_market_costs_reduce_wealth() -> None:
    dates = pd.bdate_range("2024-01-01", periods=3)
    scores = np.array([[2., 1.], [1., 2.], [2., 1.]])
    quotes = np.full_like(scores, 10.)
    trade = np.ones_like(scores, dtype=bool)
    gross = fractional_simulation(scores, quotes, trade, dates, slippage_bps=None, top_n=1)
    net = fractional_simulation(scores, quotes, trade, dates, slippage_bps=5., top_n=1)
    assert np.allclose(gross.portfolio_return, 0.)
    assert (net.portfolio_return < 0).all()
    assert np.isclose(net.portfolio_return.iloc[0], -(net.explicit_cost.iloc[0] + net.slippage_cost.iloc[0]))
    assert net.turnover.iloc[1] > net.turnover.iloc[0]


def test_fractional_position_drift_is_self_financing() -> None:
    dates = pd.bdate_range("2024-01-01", periods=3)
    scores = np.array([[2., 1.], [2., 1.], [2., 1.]])
    quotes = np.array([[10., 10.], [20., 10.], [20., 10.]])
    result = fractional_simulation(scores, quotes, np.ones_like(scores, bool), dates, slippage_bps=None, top_n=1)
    assert np.isclose(result.portfolio_return.iloc[0], .995)
    assert np.isclose(result.portfolio_return.iloc[1], 0.)


def test_fractional_no_fill_does_not_buy_next_rank_or_use_missing_asset_zero() -> None:
    dates = pd.bdate_range("2024-01-01", periods=3)
    scores = np.array([[2., 1.], [1., 2.], [1., 2.]])
    quotes = np.array([[10., 10.], [10., 10.], [10., 20.]])
    trade = np.array([[False, True], [True, True], [True, True]])
    result = fractional_simulation(scores, quotes, trade, dates, slippage_bps=None, top_n=1)
    assert result.turnover.iloc[0] == 0
    assert np.isclose(result.blocked_target_weight.iloc[0], .995)
    assert np.isclose(result.portfolio_return.iloc[1], .995)


def test_constant_predictions_are_cash_not_alphabetic_stock_picking() -> None:
    dates = pd.bdate_range("2024-01-01", periods=3)
    scores = np.ones((3, 2))
    quotes = np.array([[10., 10.], [20., 10.], [30., 10.]])
    result = fractional_simulation(scores, quotes, np.ones_like(scores, bool), dates, slippage_bps=5.)
    assert np.allclose(result.portfolio_return, 0.)
    assert result.constant_score.all()


def test_minimum_commission_cannot_overdraw_nearly_exhausted_account() -> None:
    dates = pd.bdate_range("2024-01-01", periods=5)
    scores = np.array([[2., 1.], [1., 2.], [2., 1.], [1., 2.], [2., 1.]])
    quotes = np.array([[10., 10.], [.001, 10.], [.001, .001], [.001, .001], [.001, .001]])
    result = fractional_simulation(scores, quotes, np.ones_like(scores, bool), dates, slippage_bps=20., top_n=1, initial_cash=100.)
    assert (result.equity >= 0).all()
