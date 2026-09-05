from __future__ import annotations

import pandas as pd

from mlquant.account import CashEquityLedger, FeeSchedule, next_open_date
from mlquant.reconciliation import replay_filled_trades, vectorbt_replay_single_symbol


def market(price: float = 10.0, **kwargs) -> pd.DataFrame:
    return pd.DataFrame([{"symbol": "000001.SZ", "open": price, "is_suspended": False,
                          "limit_up": price * 1.1, "limit_down": price * .9, **kwargs}])


def test_historical_fee_switches() -> None:
    fees = FeeSchedule()
    assert fees.rates(pd.Timestamp("2023-08-27"), "sell")["stamp_tax"] == .001
    assert fees.rates(pd.Timestamp("2023-08-28"), "sell")["stamp_tax"] == .0005
    assert fees.rates(pd.Timestamp("2022-04-28"), "buy")["transfer_fee"] == .00002
    assert fees.rates(pd.Timestamp("2022-04-29"), "buy")["transfer_fee"] == .00001


def test_zero_fee_schedule_disables_statutory_fees() -> None:
    fees = FeeSchedule(0.0, 0.0, False).calculate(
        pd.Timestamp("2024-01-02"), "sell", 100_000
    )
    assert fees == {"commission": 0.0, "stamp_tax": 0.0, "transfer_fee": 0.0}


def test_lot_size_cash_and_t_plus_one() -> None:
    ledger = CashEquityLedger(initial_cash=10_000, slippage_bps=0)
    targets = pd.Series({"000001.SZ": 1.0})
    ledger.rebalance(pd.Timestamp("2024-01-02"), market(), targets)
    assert ledger.shares("000001.SZ") == 900
    ledger.rebalance(pd.Timestamp("2024-01-02"), market(), pd.Series(dtype=float))
    assert ledger.shares("000001.SZ") == 900
    ledger.rebalance(pd.Timestamp("2024-01-03"), market(), pd.Series(dtype=float))
    assert ledger.shares("000001.SZ") == 0


def test_suspended_and_limit_orders_are_rejected() -> None:
    ledger = CashEquityLedger(initial_cash=100_000, slippage_bps=0)
    suspended = market(is_suspended=True)
    ledger.rebalance(pd.Timestamp("2024-01-02"), suspended, pd.Series({"000001.SZ": 1.0}))
    assert ledger.trade_frame().iloc[-1]["reason"] == "suspended"
    at_limit = market()
    at_limit["limit_up"] = at_limit["open"]
    ledger.rebalance(pd.Timestamp("2024-01-03"), at_limit, pd.Series({"000001.SZ": 1.0}))
    assert ledger.trade_frame().iloc[-1]["reason"] == "limit_up"


def test_failed_sale_remains_held() -> None:
    ledger = CashEquityLedger(initial_cash=100_000, slippage_bps=0)
    ledger.rebalance(pd.Timestamp("2024-01-02"), market(), pd.Series({"000001.SZ": .5}))
    down = market()
    down["limit_down"] = down["open"]
    ledger.rebalance(pd.Timestamp("2024-01-03"), down, pd.Series(dtype=float))
    assert ledger.shares("000001.SZ") > 0


def test_dividend_and_bonus_actions() -> None:
    ledger = CashEquityLedger(initial_cash=100_000, slippage_bps=0)
    ledger.rebalance(pd.Timestamp("2024-01-02"), market(), pd.Series({"000001.SZ": .5}))
    held, cash = ledger.shares("000001.SZ"), ledger.cash
    actions = pd.DataFrame([{"ex_date": pd.Timestamp("2024-01-03"), "symbol": "000001.SZ",
                             "cash_dividend_per_share": .1, "bonus_share_ratio": .2}])
    ledger.apply_corporate_actions(pd.Timestamp("2024-01-03"), actions)
    assert ledger.cash == cash + held * .1
    assert ledger.shares("000001.SZ") == held + int(held * .2)


def test_next_open_is_strictly_after_signal() -> None:
    calendar = pd.DataFrame({"trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]), "is_open": [True, True]})
    assert next_open_date(pd.Timestamp("2024-01-02"), calendar) == pd.Timestamp("2024-01-03")


def test_gold_replay_and_vectorbt_match_fixed_fills() -> None:
    ledger = CashEquityLedger(initial_cash=100_000, slippage_bps=0)
    ledger.rebalance(pd.Timestamp("2024-01-02"), market(10), pd.Series({"000001.SZ": .5}))
    ledger.rebalance(pd.Timestamp("2024-01-03"), market(11), pd.Series(dtype=float))
    trades = ledger.trade_frame()
    gold = replay_filled_trades(trades, 100_000)
    vectorbt = vectorbt_replay_single_symbol(trades, 100_000)
    assert abs(gold.cash - ledger.cash) < 1e-8
    assert abs(vectorbt.cash - ledger.cash) < 1e-8
    assert gold.positions["000001.SZ"] == vectorbt.positions["000001.SZ"] == 0
