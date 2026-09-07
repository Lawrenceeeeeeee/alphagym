from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class FeeSchedule:
    commission_bps: float = 3.0
    minimum_commission: float = 0.0
    statutory_fees: bool = True

    def rates(self, trade_date: pd.Timestamp, side: str) -> dict[str, float]:
        when = pd.Timestamp(trade_date)
        stamp = (
            0.0
            if not self.statutory_fees or side == "buy"
            else (0.0005 if when >= pd.Timestamp("2023-08-28") else 0.001)
        )
        transfer = (
            0.0
            if not self.statutory_fees
            else (0.00001 if when >= pd.Timestamp("2022-04-29") else 0.00002)
        )
        return {"commission": self.commission_bps / 10_000, "stamp_tax": stamp, "transfer_fee": transfer}

    def calculate(self, trade_date: pd.Timestamp, side: str, notional: float) -> dict[str, float]:
        rates = self.rates(trade_date, side)
        return {
            "commission": max(self.minimum_commission, notional * rates["commission"]),
            "stamp_tax": notional * rates["stamp_tax"],
            "transfer_fee": notional * rates["transfer_fee"],
        }


@dataclass(slots=True)
class Lot:
    shares: int
    acquired_date: pd.Timestamp


@dataclass(slots=True)
class CashEquityLedger:
    initial_cash: float = 100_000_000.0
    lot_size: int = 100
    slippage_bps: float = 5.0
    fees: FeeSchedule = field(default_factory=FeeSchedule)
    cash: float = field(init=False)
    lots: dict[str, list[Lot]] = field(default_factory=dict, init=False)
    trades: list[dict[str, object]] = field(default_factory=list, init=False)
    snapshots: list[dict[str, object]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.cash = float(self.initial_cash)

    def shares(self, symbol: str) -> int:
        return sum(lot.shares for lot in self.lots.get(symbol, []))

    def sellable_shares(self, symbol: str, trade_date: pd.Timestamp) -> int:
        when = pd.Timestamp(trade_date).normalize()
        return sum(lot.shares for lot in self.lots.get(symbol, []) if lot.acquired_date < when)

    def apply_corporate_actions(self, trade_date: pd.Timestamp, actions: pd.DataFrame) -> None:
        if actions.empty:
            return
        current = actions[actions["ex_date"] == pd.Timestamp(trade_date).normalize()]
        for row in current.itertuples(index=False):
            held = self.shares(row.symbol)
            if held <= 0:
                continue
            self.cash += held * float(getattr(row, "cash_dividend_per_share", 0.0) or 0.0)
            ratio = float(getattr(row, "bonus_share_ratio", 0.0) or 0.0)
            if ratio:
                bonus = int(np.floor(held * ratio))
                if bonus:
                    self.lots.setdefault(row.symbol, []).append(Lot(bonus, pd.Timestamp(trade_date).normalize()))

    def rebalance(self, trade_date: pd.Timestamp, market: pd.DataFrame, targets: pd.Series) -> None:
        when = pd.Timestamp(trade_date).normalize()
        prices = market.set_index("symbol")
        equity = self.mark_to_market(prices["open"])
        desired: dict[str, int] = {}
        for symbol, weight in targets[targets > 0].items():
            if symbol not in prices.index:
                continue
            open_price = prices.at[symbol, "open"]
            if pd.isna(open_price) or open_price <= 0:
                continue
            desired[symbol] = int(equity * weight / open_price // self.lot_size * self.lot_size)

        all_symbols = set(self.lots) | set(desired)
        for symbol in sorted(all_symbols):
            delta = desired.get(symbol, 0) - self.shares(symbol)
            if delta < 0:
                self._execute(when, symbol, "sell", min(-delta, self.sellable_shares(symbol, when)), prices)
        for symbol in sorted(all_symbols):
            delta = desired.get(symbol, 0) - self.shares(symbol)
            if delta > 0:
                self._execute(when, symbol, "buy", delta, prices)
        self.record_snapshot(when, prices["open"])

    def _execute(self, date: pd.Timestamp, symbol: str, side: str, shares: int, market: pd.DataFrame) -> None:
        if shares <= 0 or symbol not in market.index:
            return
        row = market.loc[symbol]
        suspended = row.get("is_suspended", False)
        if pd.notna(suspended) and bool(suspended):
            self._reject(date, symbol, side, shares, "suspended")
            return
        open_price = row["open"]
        if pd.isna(open_price) or open_price <= 0:
            self._reject(date, symbol, side, shares, "no_price")
            return
        open_price = float(open_price)
        if side == "buy" and pd.notna(row.get("limit_up")) and open_price >= float(row["limit_up"]):
            self._reject(date, symbol, side, shares, "limit_up")
            return
        if side == "sell" and pd.notna(row.get("limit_down")) and open_price <= float(row["limit_down"]):
            self._reject(date, symbol, side, shares, "limit_down")
            return
        shares = int(shares // self.lot_size * self.lot_size) if side == "buy" else int(shares)
        if shares <= 0:
            return
        price = open_price * (1 + self.slippage_bps / 10_000 * (1 if side == "buy" else -1))
        if side == "buy":
            per_share = price * (1 + sum(self.fees.rates(date, side).values()))
            affordable = int(self.cash / per_share // self.lot_size * self.lot_size)
            shares = min(shares, affordable)
            if shares <= 0:
                self._reject(date, symbol, side, 0, "cash")
                return
        notional = price * shares
        costs = self.fees.calculate(date, side, notional)
        total_cost = sum(costs.values())
        if side == "buy":
            self.cash -= notional + total_cost
            self.lots.setdefault(symbol, []).append(Lot(shares, date))
        else:
            self.cash += notional - total_cost
            remaining = shares
            updated: list[Lot] = []
            for lot in self.lots.get(symbol, []):
                available = lot.acquired_date < date
                take = min(lot.shares, remaining) if available else 0
                remaining -= take
                if lot.shares > take:
                    updated.append(Lot(lot.shares - take, lot.acquired_date))
            self.lots[symbol] = updated
        self.trades.append({
            "trade_date": date, "symbol": symbol, "side": side, "shares": shares,
            "price": price, "notional": notional, **costs, "status": "filled", "reason": "",
        })

    def _reject(self, date: pd.Timestamp, symbol: str, side: str, shares: int, reason: str) -> None:
        self.trades.append({
            "trade_date": date, "symbol": symbol, "side": side, "shares": shares,
            "price": np.nan, "notional": 0.0, "commission": 0.0, "stamp_tax": 0.0,
            "transfer_fee": 0.0, "status": "rejected", "reason": reason,
        })

    def mark_to_market(self, prices: pd.Series) -> float:
        def value(symbol: str) -> float:
            price = prices.get(symbol, 0.0)
            return 0.0 if pd.isna(price) else float(price)

        return self.cash + sum(self.shares(symbol) * value(symbol) for symbol in self.lots)

    def record_snapshot(self, date: pd.Timestamp, prices: pd.Series) -> None:
        self.snapshots.append({"trade_date": date, "cash": self.cash, "equity": self.mark_to_market(prices)})

    def trade_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.trades)

    def equity_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.snapshots)


def next_open_date(signal_date: pd.Timestamp, calendar: pd.DataFrame) -> pd.Timestamp:
    opened = calendar.loc[calendar["is_open"].astype(bool), "trade_date"]
    future = pd.to_datetime(opened)[pd.to_datetime(opened) > pd.Timestamp(signal_date)]
    if future.empty:
        raise ValueError(f"no open date after {signal_date}")
    return pd.Timestamp(future.min()).normalize()
