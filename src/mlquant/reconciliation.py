from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class ReplayResult:
    cash: float
    positions: dict[str, int]


def replay_filled_trades(trades: pd.DataFrame, initial_cash: float) -> ReplayResult:
    """Independent, framework-free replay of the immutable fill ledger."""
    cash = float(initial_cash)
    positions: dict[str, int] = {}
    filled = trades[trades["status"] == "filled"].sort_values("trade_date")
    for trade in filled.itertuples(index=False):
        fees = float(trade.commission + trade.stamp_tax + trade.transfer_fee)
        signed_shares = int(trade.shares) * (1 if trade.side == "buy" else -1)
        cash -= signed_shares * float(trade.price) + fees
        positions[trade.symbol] = positions.get(trade.symbol, 0) + signed_shares
        if positions[trade.symbol] < 0:
            raise ValueError(f"replay produced short position: {trade.symbol}")
    return ReplayResult(cash, positions)


def vectorbt_replay_single_symbol(trades: pd.DataFrame, initial_cash: float) -> ReplayResult:
    """Replay fixed fills through VectorBT; intended only as an acceptance oracle."""
    import vectorbt as vbt

    filled = trades[trades["status"] == "filled"].sort_values("trade_date")
    symbols = filled["symbol"].unique()
    if len(symbols) != 1:
        raise ValueError("fixed VectorBT oracle accepts exactly one symbol")
    index = pd.DatetimeIndex(filled["trade_date"])
    prices = pd.Series(filled["price"].to_numpy(float), index=index)
    sizes = pd.Series(
        filled["shares"].to_numpy(float)
        * filled["side"].map({"buy": 1.0, "sell": -1.0}).to_numpy(),
        index=index,
    )
    fixed_fees = pd.Series(
        filled[["commission", "stamp_tax", "transfer_fee"]].sum(axis=1).to_numpy(float),
        index=index,
    )
    portfolio = vbt.Portfolio.from_orders(
        close=prices,
        size=sizes,
        size_type="amount",
        price=prices,
        fixed_fees=fixed_fees,
        init_cash=initial_cash,
        cash_sharing=True,
    )
    final_cash = float(portfolio.cash().iloc[-1])
    final_position = int(filled["shares"].where(filled["side"] == "buy", -filled["shares"]).sum())
    return ReplayResult(final_cash, {str(symbols[0]): final_position})
