from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mlquant.account import CashEquityLedger, FeeSchedule, next_open_date


@dataclass(frozen=True, slots=True)
class CostScenario:
    name: str
    slippage_bps: float
    charge_fees: bool = True


@dataclass(slots=True)
class PreparedMarket:
    daily: pd.DataFrame
    status: pd.DataFrame


def prepare_market(daily: pd.DataFrame, status: pd.DataFrame) -> PreparedMarket:
    return PreparedMarket(
        daily=daily.set_index("trade_date").sort_index(),
        status=status.set_index("trade_date").sort_index(),
    )


DEFAULT_COST_SCENARIOS = (
    CostScenario("gross", 0.0, False),
    CostScenario("base_5bps", 5.0, True),
    CostScenario("stress_10bps", 10.0, True),
)


def load_adjusted_market(
    root: str | Path,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    equity = Path(root) / "equity"
    filters: list[tuple[str, str, object]] = []
    if start is not None:
        filters.append(("trade_date", ">=", pd.Timestamp(start)))
    if end is not None:
        filters.append(("trade_date", "<=", pd.Timestamp(end)))
    daily = pd.read_parquet(
        equity / "daily.parquet",
        columns=["trade_date", "symbol", "open", "close", "float_market_cap"],
        filters=filters or None,
    )
    adjustments = pd.read_parquet(
        equity / "adjustments.parquet",
        columns=["trade_date", "symbol", "adjust_factor"],
    )
    if end is not None:
        adjustments = adjustments[
            pd.to_datetime(adjustments["trade_date"]) <= pd.Timestamp(end)
        ]
    calendar = pd.read_parquet(equity / "calendar.parquet")
    status = pd.read_parquet(
        equity / "status.parquet",
        columns=["trade_date", "symbol", "is_suspended", "limit_up", "limit_down"],
        filters=filters or None,
    )
    for frame in (daily, adjustments, calendar, status):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    daily = pd.merge_asof(
        daily.sort_values(["trade_date", "symbol"]),
        adjustments.sort_values(["trade_date", "symbol"]),
        on="trade_date",
        by="symbol",
        direction="backward",
    )
    daily["adjust_factor"] = daily["adjust_factor"].fillna(1.0)
    daily["adj_open"] = daily["open"] * daily["adjust_factor"]
    daily["adj_close"] = daily["close"] * daily["adjust_factor"]
    return daily, status, calendar


def performance_stats(
    returns: pd.Series,
    benchmark: pd.Series,
    *,
    periods_per_year: int,
) -> dict[str, float | int]:
    aligned = pd.concat(
        [returns.rename("portfolio"), benchmark.rename("benchmark")], axis=1
    ).dropna()
    if len(aligned) < 2:
        return {"periods": len(aligned)}
    active = (1.0 + aligned["portfolio"]) / (1.0 + aligned["benchmark"]) - 1.0
    years = len(aligned) / periods_per_year
    portfolio_nav = (1.0 + aligned["portfolio"]).cumprod()
    benchmark_nav = (1.0 + aligned["benchmark"]).cumprod()
    relative_nav = portfolio_nav / benchmark_nav
    annualized = float(portfolio_nav.iloc[-1] ** (1.0 / years) - 1.0)
    benchmark_annualized = float(benchmark_nav.iloc[-1] ** (1.0 / years) - 1.0)
    excess_annualized = float(relative_nav.iloc[-1] ** (1.0 / years) - 1.0)
    tracking_error = float(active.std(ddof=1) * np.sqrt(periods_per_year))
    return {
        "periods": len(aligned),
        "annualized_return": annualized,
        "benchmark_annualized_return": benchmark_annualized,
        "annualized_excess_return": excess_annualized,
        "information_ratio": (
            float(active.mean() * periods_per_year / tracking_error)
            if tracking_error > 0
            else np.nan
        ),
        "sharpe_zero_rate": (
            float(aligned["portfolio"].mean() * periods_per_year)
            / float(aligned["portfolio"].std(ddof=1) * np.sqrt(periods_per_year))
            if aligned["portfolio"].std(ddof=1) > 0
            else np.nan
        ),
        "max_drawdown": float((portfolio_nav / portfolio_nav.cummax().clip(lower=1) - 1.0).min()),
        "relative_max_drawdown": float((relative_nav / relative_nav.cummax().clip(lower=1) - 1.0).min()),
        "active_win_rate": float((active > 0).mean()),
    }


def _execution_dates(
    signal_dates: pd.DatetimeIndex, calendar: pd.DataFrame
) -> pd.Series:
    values = {}
    for signal_date in signal_dates:
        try:
            values[signal_date] = next_open_date(signal_date, calendar)
        except ValueError:
            continue
    return pd.Series(values, name="execution_date", dtype="datetime64[ns]")


def benchmark_returns(
    scores: pd.DataFrame,
    daily: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    index_code: str = "ALL_A",
    index_members: pd.DataFrame | None = None,
) -> pd.Series:
    """Point-in-time open-to-open benchmark for the score universe.

    ALL_A uses float-market-cap weights. Index studies use the latest official
    constituent-weight snapshot available at the signal close.
    """
    signal_dates = pd.DatetimeIndex(
        scores.index.get_level_values("signal_date").unique()
    ).sort_values()
    executions = _execution_dates(signal_dates, calendar)
    if len(executions) < 2:
        return pd.Series(dtype=float, name="benchmark_return")
    opened = daily.set_index(["trade_date", "symbol"])["adj_open"]
    caps = daily.set_index(["trade_date", "symbol"])["float_market_cap"]
    members = index_members.copy() if index_members is not None else pd.DataFrame()
    if not members.empty:
        members["valid_from"] = pd.to_datetime(members["valid_from"]).dt.normalize()
    rows: dict[pd.Timestamp, float] = {}
    for position, signal_date in enumerate(executions.index[:-1]):
        start_date = executions.iloc[position]
        end_date = executions.iloc[position + 1]
        symbols = scores.xs(signal_date, level="signal_date").dropna(how="all").index
        start = opened.reindex(pd.MultiIndex.from_product([[start_date], symbols])).droplevel(0)
        end = opened.reindex(pd.MultiIndex.from_product([[end_date], symbols])).droplevel(0)
        forward = end / start - 1.0
        if index_code == "ALL_A" or members.empty:
            weight = caps.reindex(
                pd.MultiIndex.from_product([[signal_date], symbols])
            ).droplevel(0)
        else:
            history = members[
                (members["index_code"] == index_code)
                & (members["valid_from"] <= signal_date)
            ]
            if history.empty:
                continue
            snapshot = history[history["valid_from"] == history["valid_from"].max()]
            weight = snapshot.set_index("symbol")["benchmark_weight"].reindex(symbols)
        valid = weight.notna() & (weight > 0)
        if (valid & forward.isna()).any():
            raise ValueError(
                f"benchmark missing prices at {signal_date.date()}; "
                "cannot discard constituents using future price availability"
            )
        if valid.any():
            normalized = weight[valid] / weight[valid].sum()
            rows[signal_date] = float((forward[valid] * normalized).sum())
    return pd.Series(rows, name="benchmark_return", dtype=float)


def _market_on(
    daily_by_date: dict[pd.Timestamp, pd.DataFrame],
    status_by_date: dict[pd.Timestamp, pd.DataFrame],
    date: pd.Timestamp,
) -> pd.DataFrame:
    market = daily_by_date[date][["symbol", "adj_open"]].rename(
        columns={"adj_open": "open"}
    )
    status = status_by_date.get(date)
    if status is not None:
        market = market.merge(
            status[["symbol", "is_suspended", "limit_up", "limit_down"]],
            on="symbol",
            how="left",
        )
    for column in ("is_suspended", "limit_up", "limit_down"):
        if column not in market:
            market[column] = np.nan
    # The imported status table contains boolean limit-hit flags, while the
    # ledger expects blocking price levels.
    market["limit_up"] = np.where(market["limit_up"].eq(True), 0.0, np.nan)
    market["limit_down"] = np.where(market["limit_down"].eq(True), np.inf, np.nan)
    return market


def _prepared_market_on(prepared: PreparedMarket, date: pd.Timestamp) -> pd.DataFrame:
    market = prepared.daily.loc[[date], ["symbol", "adj_open"]].rename(
        columns={"adj_open": "open"}
    ).reset_index(drop=True)
    if date in prepared.status.index:
        day_status = prepared.status.loc[
            [date], ["symbol", "is_suspended", "limit_up", "limit_down"]
        ].reset_index(drop=True)
        market = market.merge(day_status, on="symbol", how="left")
    for column in ("is_suspended", "limit_up", "limit_down"):
        if column not in market:
            market[column] = np.nan
    market["limit_up"] = np.where(market["limit_up"].eq(True), 0.0, np.nan)
    market["limit_down"] = np.where(market["limit_down"].eq(True), np.inf, np.nan)
    return market


def simulate_long_only(
    score: pd.Series,
    daily: pd.DataFrame,
    status: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    top_n: int,
    scenario: CostScenario,
    initial_cash: float = 100_000_000.0,
    cash_reserve: float = 0.005,
    prepared_market: PreparedMarket | None = None,
) -> tuple[pd.Series, pd.DataFrame, dict[str, float]]:
    score = score.dropna().sort_index()
    signal_dates = pd.DatetimeIndex(
        score.index.get_level_values("signal_date").unique()
    ).sort_values()
    executions = _execution_dates(signal_dates, calendar)
    available_dates = set(daily["trade_date"].unique())
    executions = executions[executions.isin(available_dates)]
    if prepared_market is None:
        needed = set(executions.tolist())
        daily_by_date = {
            pd.Timestamp(date): block
            for date, block in daily[daily["trade_date"].isin(needed)].groupby(
                "trade_date", observed=True
            )
        }
        status_by_date = {
            pd.Timestamp(date): block
            for date, block in status[status["trade_date"].isin(needed)].groupby(
                "trade_date", observed=True
            )
        }
    fees = FeeSchedule() if scenario.charge_fees else FeeSchedule(0.0, 0.0, False)
    ledger = CashEquityLedger(
        initial_cash=initial_cash, slippage_bps=scenario.slippage_bps, fees=fees
    )
    pre_trade: dict[pd.Timestamp, float] = {}
    for signal_date, execution in executions.items():
        if signal_date not in signal_dates:
            continue
        if prepared_market is not None:
            if execution not in prepared_market.daily.index:
                continue
            market = _prepared_market_on(prepared_market, execution)
        else:
            if execution not in daily_by_date:
                continue
            market = _market_on(daily_by_date, status_by_date, execution)
        price = market.set_index("symbol")["open"]
        held = [symbol for symbol in ledger.lots if ledger.shares(symbol) > 0]
        if price.reindex(held).isna().any():
            raise ValueError(
                f"held security lacks a mark at {execution.date()}; "
                "suspension/delisting valuation is required, not zero valuation"
            )
        pre_trade[signal_date] = ledger.mark_to_market(price)
        if signal_date == executions.index[-1]:
            # The final opening mark closes the last measured return. Do not
            # charge or count terminal orders outside the evaluation interval.
            break
        cross_section = score.xs(signal_date, level="signal_date").dropna()
        selected = cross_section.nlargest(top_n)
        if selected.empty:
            continue
        targets = pd.Series(
            (1.0 - cash_reserve) / len(selected), index=selected.index, dtype=float
        )
        ledger.rebalance(execution, market, targets)
    equity = pd.Series(pre_trade, name="pre_trade_equity", dtype=float).sort_index()
    returns = equity.shift(-1) / equity - 1.0
    returns = returns.iloc[:-1].rename("portfolio_return")
    trades = ledger.trade_frame()
    filled = trades[trades["status"] == "filled"] if not trades.empty else trades
    notional = float(filled["notional"].sum()) if not filled.empty else 0.0
    explicit = (
        float(filled[["commission", "stamp_tax", "transfer_fee"]].sum().sum())
        if not filled.empty
        else 0.0
    )
    years = max(len(returns), 1)
    diagnostics = {
        "notional": notional,
        "explicit_cost": explicit,
        "slippage_cost": notional * scenario.slippage_bps / 10_000,
        "average_equity": float(equity.mean()) if not equity.empty else np.nan,
        "filled_orders": len(filled),
        "rejected_orders": int((trades["status"] == "rejected").sum()) if not trades.empty else 0,
        "period_count": years,
    }
    return returns, trades, diagnostics
