from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mlquant.account import CashEquityLedger, next_open_date
from mlquant.equity_data import EquityDataBundle
from mlquant.factors import REGISTRY
from mlquant.factors.compute import compute_factors
from mlquant.research import eligible_universe, huatai_industry_layers, neutralize_cross_section


@dataclass(slots=True)
class BacktestResult:
    targets: pd.DataFrame
    trades: pd.DataFrame
    equity: pd.DataFrame


def month_end_signal_dates(calendar: pd.DataFrame, start: str, end: str) -> list[pd.Timestamp]:
    opened = calendar[calendar["is_open"].astype(bool)].copy()
    opened["trade_date"] = pd.to_datetime(opened["trade_date"])
    opened = opened[opened["trade_date"].between(start, end)]
    return opened.groupby(opened["trade_date"].dt.to_period("M"))["trade_date"].max().tolist()


def build_factor_layers(
    bundle: EquityDataBundle,
    signal_date: pd.Timestamp,
    index_code: str,
    factor_name: str,
    *,
    smoke: bool = False,
) -> pd.DataFrame:
    """Build one point-in-time Huatai-style five-layer target at a signal close."""
    bundle.audit(formal=not smoke, index_code=index_code).require_ok()
    when = pd.Timestamp(signal_date).normalize()
    universe = eligible_universe(
        when, index_code, bundle.index_members, bundle.securities, bundle.status, bundle.calendar
    )
    industry = bundle.point_in_time("industries", when)[["symbol", "industry_code"]]
    latest = (
        bundle.daily[bundle.daily["trade_date"] <= when]
        .sort_values("trade_date")
        .groupby("symbol", observed=True)
        .tail(1)
    )
    if "float_market_cap" not in latest:
        raise ValueError("daily input requires float_market_cap for neutralization")
    values = compute_factors(bundle.daily, bundle.fundamentals, [when], [factor_name])
    cross_section = (
        universe[["symbol", "benchmark_weight"]]
        .merge(industry, on="symbol", how="inner")
        .merge(latest[["symbol", "float_market_cap"]], on="symbol", how="inner")
        .merge(values[["symbol", "raw_value"]], on="symbol", how="inner")
        .dropna(subset=["raw_value"])
    )
    neutralized = neutralize_cross_section(cross_section)
    if REGISTRY.get(factor_name).expected_direction == "negative":
        neutralized["neutralized_value"] = -neutralized["neutralized_value"]
    layer_input = neutralized.rename(columns={"neutralized_value": "factor_value"})
    layers = huatai_industry_layers(layer_input.dropna(subset=["factor_value"]))
    layers["signal_date"] = when
    layers["index_code"] = index_code
    layers["factor_name"] = factor_name
    return layers


def run_long_only_backtest(
    bundle: EquityDataBundle,
    layer_targets: pd.DataFrame,
    *,
    layer: int = 1,
    initial_cash: float = 100_000_000,
    slippage_bps: float = 5.0,
) -> BacktestResult:
    ledger = CashEquityLedger(initial_cash=initial_cash, slippage_bps=slippage_bps)
    selected = layer_targets[layer_targets["layer"] == layer].copy()
    for signal_date, group in selected.groupby("signal_date", observed=True):
        execution_date = next_open_date(pd.Timestamp(signal_date), bundle.calendar)
        market = bundle.daily[bundle.daily["trade_date"] == execution_date][["symbol", "open"]]
        execution_status = bundle.status[bundle.status["trade_date"] == execution_date]
        market = market.merge(
            execution_status[["symbol", "is_suspended", "limit_up", "limit_down"]],
            on="symbol",
            how="left",
        )
        targets = group.groupby("symbol", observed=True)["target_weight"].sum()
        ledger.rebalance(execution_date, market, targets)
    return BacktestResult(selected, ledger.trade_frame(), ledger.equity_frame())


def run_slippage_scenarios(
    bundle: EquityDataBundle, layer_targets: pd.DataFrame, *, initial_cash: float = 100_000_000
) -> dict[str, BacktestResult]:
    return {
        "base_5bps": run_long_only_backtest(
            bundle, layer_targets, initial_cash=initial_cash, slippage_bps=5.0
        ),
        "stress_10bps": run_long_only_backtest(
            bundle, layer_targets, initial_cash=initial_cash, slippage_bps=10.0
        ),
    }
