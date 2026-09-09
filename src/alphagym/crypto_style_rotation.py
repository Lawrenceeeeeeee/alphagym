"""Point-in-time crypto style portfolio construction and weekly rotation research."""
from __future__ import annotations

import hashlib
import html
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from alphagym import storage_io
from alphagym.crypto_hourly import DEFAULT_CORE_UNIVERSE, HourlySpec, download_panel

STYLE_NAMES = ("momentum", "reversal", "low_volatility", "liquidity", "carry")
CORE_STYLE_NAMES = ("momentum", "reversal", "low_volatility", "liquidity")
STRATEGY_NAMES = (
    "static_equal",
    "static_inverse_volatility",
    "dynamic_4w",
    "dynamic_8w",
    "dynamic_12w",
    "dynamic_selected",
    "dynamic_selected_ex_momentum",
)
DEFAULT_SPLITS = {
    "development": ("2023-07-01", "2024-06-30"),
    "validation": ("2024-07-01", "2025-06-30"),
    "test": ("2025-07-01", "2026-06-30"),
    "monitoring": ("2026-07-01", None),
}


@dataclass(frozen=True)
class UniverseSpec:
    size: int = 30
    minimum_listing_age_days: int = 365
    liquidity_lookback_days: int = 30
    minimum_coverage: float = 0.90
    includes_delisted: bool = False
    point_in_time_catalogue: bool = False
    instruments: tuple[str, ...] = DEFAULT_CORE_UNIVERSE


@dataclass(frozen=True)
class RotationSpec:
    source_report_id: str | None = None
    bar: str = "4H"
    history_days: int = 1500
    workers: int = 4
    universe: UniverseSpec = field(default_factory=UniverseSpec)
    candidate_windows_weeks: tuple[int, ...] = (4, 8, 12)
    fee_bps_per_side: tuple[float, ...] = (2.0, 5.0, 10.0)
    primary_fee_bps_per_side: float = 5.0
    market_state_resource: str | None = (
        "crypto/okx/trading_statistics/1D/market_state.parquet"
    )
    long_short_fraction: float = 0.30
    splits: dict[str, tuple[str, str | None]] = field(
        default_factory=lambda: dict(DEFAULT_SPLITS)
    )
    bootstrap_samples: int = 2000
    bootstrap_block_weeks: int = 4
    seed: int = 42

    def validate(self) -> None:
        if self.bar != "4H":
            raise ValueError("crypto style rotation is frozen to 4H bars")
        if self.universe.size < 5:
            raise ValueError("universe size must be at least five")
        if not 0.5 <= self.universe.minimum_coverage <= 1:
            raise ValueError("minimum coverage must be between 0.5 and 1")
        if not 0 < self.long_short_fraction <= 0.5:
            raise ValueError("long_short_fraction must be in (0, 0.5]")
        if not self.candidate_windows_weeks or min(self.candidate_windows_weeks) < 2:
            raise ValueError("candidate rotation windows must be at least two weeks")
        if self.primary_fee_bps_per_side not in self.fee_bps_per_side:
            raise ValueError("primary fee must be one of the fee scenarios")
        if 10.0 not in self.fee_bps_per_side:
            raise ValueError("fee scenarios must include the 10 bps stress case")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        required = {"development", "validation", "test", "monitoring"}
        if set(self.splits) != required:
            raise ValueError(f"splits must contain exactly {sorted(required)}")
        previous_end: pd.Timestamp | None = None
        for name in ("development", "validation", "test", "monitoring"):
            start_raw, end_raw = self.splits[name]
            start = pd.Timestamp(start_raw, tz="UTC")
            end = pd.Timestamp(end_raw, tz="UTC") if end_raw else None
            if end is not None and end < start:
                raise ValueError(f"split {name} ends before it starts")
            if previous_end is not None and start.normalize() != previous_end.normalize() + pd.Timedelta(days=1):
                raise ValueError("research splits must be contiguous")
            previous_end = end


def load_rotation_spec(path: str | Path) -> RotationSpec:
    source = Path(path).expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("crypto style rotation spec must be a mapping")
    universe_payload = payload.pop("universe", {}) or {}
    if "instruments" in universe_payload:
        universe_payload["instruments"] = tuple(universe_payload["instruments"] or ())
    if "candidate_windows_weeks" in payload:
        payload["candidate_windows_weeks"] = tuple(payload["candidate_windows_weeks"])
    if "fee_bps_per_side" in payload:
        payload["fee_bps_per_side"] = tuple(float(x) for x in payload["fee_bps_per_side"])
    if "splits" in payload:
        payload["splits"] = {
            key: (str(value[0]), str(value[1]) if value[1] is not None else None)
            for key, value in payload["splits"].items()
        }
    spec = RotationSpec(universe=UniverseSpec(**universe_payload), **payload)
    spec.validate()
    return spec


def _split_labels(times: pd.Series, splits: dict[str, tuple[str, str | None]]) -> pd.Series:
    result = pd.Series("warmup", index=times.index, dtype="object")
    stamps = pd.to_datetime(times, utc=True)
    for name, (start_raw, end_raw) in splits.items():
        start = pd.Timestamp(start_raw, tz="UTC")
        end = pd.Timestamp(end_raw, tz="UTC") + pd.Timedelta(days=1) if end_raw else None
        mask = stamps.ge(start) & (stamps.lt(end) if end is not None else True)
        result.loc[mask] = name
    return result


def compute_style_features(panel: pd.DataFrame, spec: RotationSpec) -> pd.DataFrame:
    required = {"ts", "instrument", "open", "high", "low", "close", "volume_quote"}
    missing = required - set(panel)
    if missing:
        raise ValueError(f"missing candle columns: {sorted(missing)}")
    data = panel.sort_values(["instrument", "ts"]).copy()
    data["ts"] = pd.to_datetime(data["ts"], utc=True)
    numeric = ["open", "high", "low", "close", "volume_quote"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    if "funding_event" not in data:
        data["funding_event"] = 0.0
    if "funding_rate" not in data:
        data["funding_rate"] = np.nan
    grouped = data.groupby("instrument", observed=True, group_keys=False)
    bars_per_day = 6
    data["log_return"] = grouped["close"].transform(lambda x: np.log(x).diff())
    data["momentum"] = grouped["close"].transform(
        lambda x: np.log(x.shift(bars_per_day) / x.shift(28 * bars_per_day))
    )
    data["reversal"] = grouped["close"].transform(
        lambda x: -np.log(x / x.shift(bars_per_day))
    )
    data["low_volatility"] = -grouped["log_return"].transform(
        lambda x: x.rolling(28 * bars_per_day, min_periods=21 * bars_per_day).std()
    )
    volume_window = spec.universe.liquidity_lookback_days * bars_per_day
    data["median_quote_volume"] = grouped["volume_quote"].transform(
        lambda x: x.rolling(volume_window, min_periods=int(volume_window * 0.5)).median()
    )
    raw_illiquidity = data["log_return"].abs() / data["volume_quote"].replace(0, np.nan)
    data["amihud"] = raw_illiquidity.groupby(data["instrument"], observed=True).transform(
        lambda x: x.rolling(volume_window, min_periods=int(volume_window * 0.5)).mean()
    )
    volume_rank = data.groupby("ts", observed=True)["median_quote_volume"].rank(pct=True)
    amihud_rank = data.groupby("ts", observed=True)["amihud"].rank(pct=True)
    data["liquidity"] = (volume_rank + (1 - amihud_rank)) / 2
    data["carry"] = -grouped["funding_rate"].transform(
        lambda x: x.ffill().rolling(7 * bars_per_day, min_periods=3 * bars_per_day).mean()
    )
    data["coverage_30d"] = grouped["close"].transform(
        lambda x: x.notna().rolling(volume_window, min_periods=1).sum() / volume_window
    )
    if "listing_time" in data:
        data["listing_time"] = pd.to_datetime(data["listing_time"], utc=True, errors="coerce")
    else:
        first_seen = grouped["ts"].transform("min")
        data["listing_time"] = first_seen
    hold_bars = 7 * bars_per_day
    data["forward_price_return"] = grouped["open"].transform(
        lambda x: np.log(x.shift(-(hold_bars + 1)) / x.shift(-1))
    )
    data["forward_funding"] = grouped["funding_event"].transform(
        lambda x: sum(x.shift(-step).fillna(0) for step in range(1, hold_bars + 1))
    )
    data["forward_total_return"] = data["forward_price_return"] - data["forward_funding"]
    weekly = (data["ts"].dt.weekday.eq(0) & data["ts"].dt.hour.eq(0))
    return data.loc[weekly].reset_index(drop=True)


def build_point_in_time_universe(
    features: pd.DataFrame, spec: RotationSpec
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = features.copy()
    age = (work["ts"] - work["listing_time"]).dt.total_seconds() / 86400
    enough_age = age.ge(spec.universe.minimum_listing_age_days)
    enough_coverage = work["coverage_30d"].ge(spec.universe.minimum_coverage)
    liquid = work["median_quote_volume"].gt(0)
    # Funding-history availability must not retroactively remove an otherwise
    # tradeable instrument from the price/volume universe. Carry is activated
    # separately only when it has enough development-period observations.
    factors_ready = work[list(CORE_STYLE_NAMES)].notna().all(axis=1)
    work["eligible"] = enough_age & enough_coverage & liquid & factors_ready
    work["exclusion_reason"] = np.select(
        [~enough_age, ~enough_coverage, ~liquid, ~factors_ready],
        ["listing_age", "coverage", "liquidity", "style_history"],
        default="eligible",
    )
    work["liquidity_rank"] = work.loc[work["eligible"]].groupby(
        "ts", observed=True
    )["median_quote_volume"].rank(method="first", ascending=False)
    work["selected"] = work["eligible"] & work["liquidity_rank"].le(spec.universe.size)
    work.loc[work["eligible"] & ~work["selected"], "exclusion_reason"] = "outside_liquid_top_n"
    selected = work[work["selected"]].copy()
    audit_columns = [
        "ts", "instrument", "listing_time", "coverage_30d", "median_quote_volume",
        "eligible", "selected", "liquidity_rank", "exclusion_reason",
    ]
    return selected, work[audit_columns].reset_index(drop=True)


def _rank_ic(frame: pd.DataFrame, style: str) -> float:
    values = frame.groupby("ts", observed=True).apply(
        lambda x: x[style].corr(x["forward_total_return"], method="spearman"),
        include_groups=False,
    )
    return float(values.mean()) if len(values) else np.nan


def freeze_style_directions(
    selected: pd.DataFrame, spec: RotationSpec, styles: tuple[str, ...] = STYLE_NAMES
) -> dict[str, int]:
    labels = _split_labels(selected["ts"], spec.splits)
    development = selected[labels.eq("development")]
    if development.empty:
        raise ValueError("no development observations after point-in-time universe filters")
    return {
        style: (1 if not np.isfinite(ic := _rank_ic(development, style)) or ic >= 0 else -1)
        for style in styles
    }


def build_style_positions(
    selected: pd.DataFrame, directions: dict[str, int], spec: RotationSpec,
    styles: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for style in styles or tuple(directions):
        clean = selected[[
            "ts", "instrument", style, "forward_price_return", "forward_funding"
        ]].dropna().copy()
        clean["score"] = clean[style] * directions[style]
        clean["rank"] = clean.groupby("ts", observed=True)["score"].rank(method="first")
        clean["count"] = clean.groupby("ts", observed=True)["instrument"].transform("size")
        clean["leg_count"] = np.maximum(
            1, np.floor(clean["count"] * spec.long_short_fraction).astype(int)
        )
        clean["position"] = np.where(
            clean["rank"] > clean["count"] - clean["leg_count"],
            0.5 / clean["leg_count"],
            np.where(clean["rank"] <= clean["leg_count"], -0.5 / clean["leg_count"], 0.0),
        )
        clean["style"] = style
        rows.append(clean)
    return pd.concat(rows, ignore_index=True)


def style_portfolio_returns(
    positions: pd.DataFrame, fee_bps: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = positions.copy()
    work["price_contribution"] = work["position"] * work["forward_price_return"]
    work["funding_contribution"] = -work["position"] * work["forward_funding"]
    weights = work.pivot_table(
        index=["ts", "style"], columns="instrument", values="position", fill_value=0
    )
    prior = weights.groupby(level="style", observed=True).shift().fillna(0)
    turnover = (weights - prior).abs().sum(axis=1)
    result = work.groupby(["ts", "style"], observed=True).agg(
        price_return=("price_contribution", "sum"),
        funding_return=("funding_contribution", "sum"),
    )
    result["turnover"] = turnover.reindex(result.index)
    result["gross_return"] = result["price_return"] + result["funding_return"]
    result["net_return"] = result["gross_return"] - result["turnover"] * fee_bps / 10_000
    return result.reset_index(), weights


def _annual_sharpe(values: pd.Series) -> float:
    sample = values.dropna()
    std = float(sample.std(ddof=1)) if len(sample) > 1 else np.nan
    return float(sample.mean() / std * np.sqrt(52)) if std > 0 else np.nan


def _allocator_weights(
    style_returns: pd.DataFrame, window: int, styles: tuple[str, ...] = STYLE_NAMES
) -> pd.DataFrame:
    wide = style_returns.pivot(index="ts", columns="style", values="net_return").reindex(
        columns=list(styles)
    )
    mean = wide.rolling(window, min_periods=window).mean().shift(1)
    volatility = wide.rolling(window, min_periods=window).std().shift(1)
    score = (mean / volatility.clip(lower=1e-12)).clip(lower=0).fillna(0)
    denominator = score.sum(axis=1)
    return score.div(denominator.replace(0, np.nan), axis=0).fillna(0)


def _static_weights(
    style_returns: pd.DataFrame, spec: RotationSpec, styles: tuple[str, ...] = STYLE_NAMES
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = style_returns.pivot(index="ts", columns="style", values="net_return").reindex(
        columns=list(styles)
    )
    equal = pd.DataFrame(1 / len(styles), index=wide.index, columns=wide.columns)
    split = _split_labels(pd.Series(wide.index, index=wide.index), spec.splits)
    development_vol = wide.loc[split.eq("development")].std()
    inverse = 1 / development_vol.replace(0, np.nan)
    inverse = inverse / inverse.sum()
    inverse_vol = pd.DataFrame(
        np.tile(inverse.fillna(0).to_numpy(), (len(wide), 1)),
        index=wide.index,
        columns=wide.columns,
    )
    return equal, inverse_vol


def _combine_strategy(
    name: str,
    allocations: pd.DataFrame,
    positions: pd.DataFrame,
    fee_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    allocation_long = allocations.stack(future_stack=True).rename("style_weight").reset_index()
    allocation_long.columns = ["ts", "style", "style_weight"]
    work = positions.merge(allocation_long, on=["ts", "style"], how="left")
    work["style_weight"] = work["style_weight"].fillna(0)
    work["combined_weight"] = work["position"] * work["style_weight"]
    work["price_contribution"] = work["combined_weight"] * work["forward_price_return"]
    work["funding_contribution"] = -work["combined_weight"] * work["forward_funding"]
    instrument_weights = work.groupby(["ts", "instrument"], observed=True)[
        "combined_weight"
    ].sum().unstack(fill_value=0).sort_index()
    prior = instrument_weights.shift().fillna(0)
    turnover = (instrument_weights - prior).abs().sum(axis=1)
    result = work.groupby("ts", observed=True).agg(
        price_return=("price_contribution", "sum"),
        funding_return=("funding_contribution", "sum"),
    )
    result["turnover"] = turnover.reindex(result.index).fillna(0)
    result["gross_return"] = result["price_return"] + result["funding_return"]
    result["net_return"] = result["gross_return"] - result["turnover"] * fee_bps / 10_000
    result["strategy"] = name
    result["fee_bps_per_side"] = fee_bps
    contribution = work[["ts", "instrument", "price_contribution", "funding_contribution"]].copy()
    contribution["strategy"] = name
    contribution["fee_bps_per_side"] = fee_bps
    return result.reset_index(), contribution


def _strategy_metric(frame: pd.DataFrame) -> dict[str, float | int]:
    returns = frame["net_return"].dropna()
    curve = np.exp(returns.cumsum())
    drawdown = curve / curve.cummax() - 1 if len(curve) else pd.Series(dtype=float)
    return {
        "observations": len(returns),
        "cumulative_return": float(np.exp(returns.sum()) - 1) if len(returns) else np.nan,
        "annualized_sharpe": _annual_sharpe(returns),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else np.nan,
        "average_turnover": float(frame["turnover"].mean()) if len(frame) else np.nan,
        "funding_return": float(frame["funding_return"].sum()) if len(frame) else np.nan,
    }


def _summaries(returns: pd.DataFrame, spec: RotationSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = returns.copy()
    data["split"] = _split_labels(data["ts"], spec.splits)
    rows = []
    for (strategy, fee, split), group in data[data["split"] != "warmup"].groupby(
        ["strategy", "fee_bps_per_side", "split"], observed=True
    ):
        rows.append({"strategy": strategy, "fee_bps_per_side": fee, "split": split,
                     **_strategy_metric(group)})
    yearly = data.assign(year=data["ts"].dt.year).groupby(
        ["strategy", "fee_bps_per_side", "year"], observed=True
    ).apply(lambda x: pd.Series(_strategy_metric(x)), include_groups=False).reset_index()
    return pd.DataFrame(rows), yearly


def _monthly_returns(returns: pd.DataFrame) -> pd.DataFrame:
    data = returns.copy()
    data["month"] = data["ts"].dt.tz_localize(None).dt.to_period("M").astype(str)
    return data.groupby(
        ["strategy", "fee_bps_per_side", "month"], observed=True
    )["net_return"].apply(lambda x: float(np.exp(x.sum()) - 1)).rename(
        "monthly_return"
    ).reset_index()


def _select_window(summary: pd.DataFrame, primary_fee: float) -> tuple[int, pd.DataFrame]:
    candidates = summary[
        summary["strategy"].str.startswith("dynamic_")
        & ~summary["strategy"].isin(["dynamic_selected", "dynamic_selected_ex_momentum"])
        & summary["split"].isin(["development", "validation"])
        & summary["fee_bps_per_side"].eq(primary_fee)
    ]
    table = candidates.pivot(index="strategy", columns="split", values="annualized_sharpe")
    if not {"development", "validation"} <= set(table):
        raise ValueError("development and validation observations are required for selection")
    table["selection_score"] = table[["development", "validation"]].min(axis=1)
    table = table.sort_values(["selection_score", "strategy"], ascending=[False, True])
    selected = int(table.index[0].removeprefix("dynamic_").removesuffix("w"))
    return selected, table.reset_index()


def block_bootstrap_information_ratio(
    active_returns: pd.Series, *, samples: int, block_weeks: int, seed: int
) -> dict[str, float | int]:
    values = active_returns.dropna().to_numpy(dtype=float)
    if len(values) < max(8, block_weeks * 2):
        return {"observations": len(values), "information_ratio": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan}
    observed = _annual_sharpe(pd.Series(values))
    rng = np.random.default_rng(seed)
    starts = np.arange(0, len(values) - block_weeks + 1)
    estimates = []
    blocks_needed = int(np.ceil(len(values) / block_weeks))
    for _ in range(samples):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([values[start:start + block_weeks] for start in chosen])[:len(values)]
        estimates.append(_annual_sharpe(pd.Series(sample)))
    finite = np.asarray(estimates)[np.isfinite(estimates)]
    return {
        "observations": len(values),
        "information_ratio": observed,
        "ci_lower": float(np.quantile(finite, 0.025)) if len(finite) else np.nan,
        "ci_upper": float(np.quantile(finite, 0.975)) if len(finite) else np.nan,
    }


def _regime_summary(
    returns: pd.DataFrame, selected: pd.DataFrame, spec: RotationSpec
) -> pd.DataFrame:
    market = selected.groupby("ts", observed=True)["forward_price_return"].mean().sort_index()
    trailing = market.rolling(4, min_periods=4).sum().shift(1)
    regimes = pd.Series(np.where(trailing >= 0, "up", "down"), index=market.index)
    primary = returns[returns["fee_bps_per_side"].eq(spec.primary_fee_bps_per_side)].copy()
    primary["regime"] = primary["ts"].map(regimes)
    return primary.groupby(["strategy", "regime"], observed=True).apply(
        lambda x: pd.Series(_strategy_metric(x)), include_groups=False
    ).reset_index()


def market_state_attribution(
    returns: pd.DataFrame, market_state: pd.DataFrame, spec: RotationSpec
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"ts", "market_state"}
    missing = required - set(market_state)
    if missing:
        raise ValueError(f"missing market-state columns: {sorted(missing)}")
    states = market_state.sort_values("ts").copy()
    states["ts"] = pd.to_datetime(states["ts"], utc=True).astype("datetime64[ns, UTC]")
    # A daily observation is conservatively available only after that UTC day closes.
    states["available_at"] = states["ts"] + pd.Timedelta(days=1)
    primary = returns[returns["fee_bps_per_side"].eq(spec.primary_fee_bps_per_side)].copy()
    primary["ts"] = pd.to_datetime(primary["ts"], utc=True).astype("datetime64[ns, UTC]")
    joined = pd.merge_asof(
        primary.sort_values("ts"),
        states[["available_at", "market_state"]].sort_values("available_at"),
        left_on="ts", right_on="available_at", direction="backward",
        tolerance=pd.Timedelta(days=2),
    )
    joined["split"] = _split_labels(joined["ts"], spec.splits)
    observed = joined.dropna(subset=["market_state"])
    if observed.empty:
        return joined, pd.DataFrame(columns=["strategy", "market_state", "split"])
    summary = observed.groupby(
        ["strategy", "market_state", "split"], observed=True
    ).apply(lambda x: pd.Series(_strategy_metric(x)), include_groups=False).reset_index()
    return joined, summary


def _concentration(
    contributions: pd.DataFrame, returns: pd.DataFrame, spec: RotationSpec
) -> dict[str, Any]:
    split = _split_labels(contributions["ts"], spec.splits)
    test = contributions[
        split.eq("test")
        & contributions["strategy"].eq("dynamic_selected")
        & contributions["fee_bps_per_side"].eq(spec.primary_fee_bps_per_side)
    ].copy()
    test["total"] = test["price_contribution"] + test["funding_contribution"]
    by_coin = test.groupby("instrument", observed=True)["total"].sum()
    denominator = float(by_coin.abs().sum())
    top_coin_share = float(by_coin.abs().max() / denominator) if denominator else np.nan
    selected_returns = returns[
        returns["strategy"].eq("dynamic_selected")
        & returns["fee_bps_per_side"].eq(spec.primary_fee_bps_per_side)
    ].copy()
    labels = _split_labels(selected_returns["ts"], spec.splits)
    test_returns = selected_returns[labels.eq("test")].copy()
    test_returns["quarter"] = (
        test_returns["ts"].dt.tz_localize(None).dt.to_period("Q").astype(str)
    )
    by_quarter = test_returns.groupby("quarter")["net_return"].sum()
    quarter_denominator = float(by_quarter.abs().sum())
    top_quarter_share = (
        float(by_quarter.abs().max() / quarter_denominator) if quarter_denominator else np.nan
    )
    total_abs = float(test_returns["net_return"].abs().sum())
    top_week_share = (
        float(test_returns["net_return"].abs().max() / total_abs) if total_abs else np.nan
    )
    return {
        "top_coin_absolute_contribution_share": top_coin_share,
        "top_coin": str(by_coin.abs().idxmax()) if len(by_coin) else None,
        "top_quarter_absolute_return_share": top_quarter_share,
        "top_week_absolute_return_share": top_week_share,
    }


def _acceptance(
    summary: pd.DataFrame, concentration: dict[str, Any], primary_fee: float
) -> dict[str, Any]:
    indexed = summary.set_index(["strategy", "fee_bps_per_side", "split"])

    def metric(strategy: str, fee: float, name: str) -> float:
        key = (strategy, fee, "test")
        return float(indexed.loc[key, name]) if key in indexed.index else np.nan

    dynamic_sharpe = metric("dynamic_selected", primary_fee, "annualized_sharpe")
    static_sharpe = metric("static_inverse_volatility", primary_fee, "annualized_sharpe")
    dynamic_dd = metric("dynamic_selected", primary_fee, "max_drawdown")
    static_dd = metric("static_inverse_volatility", primary_fee, "max_drawdown")
    checks = {
        "test_sharpe_at_least_0_8": dynamic_sharpe >= 0.8,
        "sharpe_improvement_at_least_0_2": dynamic_sharpe - static_sharpe >= 0.2,
        "positive_at_10_bps": metric("dynamic_selected", 10.0, "cumulative_return") > 0,
        "max_drawdown_no_worse_than_static": dynamic_dd >= static_dd,
        "positive_without_momentum": metric(
            "dynamic_selected_ex_momentum", primary_fee, "cumulative_return"
        ) > 0,
        "coin_concentration_below_35pct": (
            concentration["top_coin_absolute_contribution_share"] <= 0.35
        ),
        "quarter_concentration_below_50pct": (
            concentration["top_quarter_absolute_return_share"] <= 0.50
        ),
        "week_concentration_below_25pct": (
            concentration["top_week_absolute_return_share"] <= 0.25
        ),
    }
    return {"passed": all(bool(value) for value in checks.values()), "checks": checks}


def evaluate_rotation(panel: pd.DataFrame, spec: RotationSpec) -> dict[str, Any]:
    spec.validate()
    features = compute_style_features(panel, spec)
    selected, universe_audit = build_point_in_time_universe(features, spec)
    if selected.groupby("ts")["instrument"].nunique().max() < 5:
        raise ValueError("fewer than five instruments pass the point-in-time universe")
    labels = _split_labels(selected["ts"], spec.splits)
    development = selected.loc[labels.eq("development")]
    active_styles = tuple(
        style for style in STYLE_NAMES
        if development.loc[development[style].notna(), "ts"].nunique() >= 12
        and development.groupby("ts", observed=True)[style].count().max() >= 5
    )
    if len(active_styles) < 2:
        raise ValueError("fewer than two styles have usable development history")
    inactive_styles = tuple(style for style in STYLE_NAMES if style not in active_styles)
    directions = freeze_style_directions(selected, spec, active_styles)
    positions = build_style_positions(selected, directions, spec, active_styles)
    primary_style, _ = style_portfolio_returns(positions, spec.primary_fee_bps_per_side)
    equal, inverse_vol = _static_weights(primary_style, spec, active_styles)
    allocation_map: dict[str, pd.DataFrame] = {
        "static_equal": equal,
        "static_inverse_volatility": inverse_vol,
    }
    for window in spec.candidate_windows_weeks:
        allocation_map[f"dynamic_{window}w"] = _allocator_weights(primary_style, window)

    provisional = []
    for name, allocations in allocation_map.items():
        frame, _ = _combine_strategy(name, allocations, positions, spec.primary_fee_bps_per_side)
        provisional.append(frame)
    provisional_returns = pd.concat(provisional, ignore_index=True)
    provisional_summary, _ = _summaries(provisional_returns, spec)
    selected_window, selection = _select_window(
        provisional_summary, spec.primary_fee_bps_per_side
    )
    selected_allocation = allocation_map[f"dynamic_{selected_window}w"]
    ex_momentum = _allocator_weights(
        primary_style, selected_window, tuple(x for x in active_styles if x != "momentum")
    )
    allocation_map["dynamic_selected"] = selected_allocation
    allocation_map["dynamic_selected_ex_momentum"] = ex_momentum

    returns_rows, contribution_rows = [], []
    for fee in spec.fee_bps_per_side:
        styles_at_fee, _ = style_portfolio_returns(positions, fee)
        for style in active_styles:
            standalone = styles_at_fee[styles_at_fee["style"].eq(style)].drop(
                columns="style"
            ).copy()
            standalone["strategy"] = f"style_{style}"
            standalone["fee_bps_per_side"] = fee
            returns_rows.append(standalone)
        for name, allocations in allocation_map.items():
            frame, contribution = _combine_strategy(name, allocations, positions, fee)
            returns_rows.append(frame)
            contribution_rows.append(contribution)
    returns = pd.concat(returns_rows, ignore_index=True)
    contributions = pd.concat(contribution_rows, ignore_index=True)
    summary, yearly = _summaries(returns, spec)
    monthly = _monthly_returns(returns)
    regime = _regime_summary(returns, selected, spec)

    comparison = returns[
        returns["fee_bps_per_side"].eq(spec.primary_fee_bps_per_side)
        & returns["strategy"].isin(["dynamic_selected", "static_inverse_volatility"])
    ].pivot(index="ts", columns="strategy", values="net_return")
    labels = _split_labels(pd.Series(comparison.index, index=comparison.index), spec.splits)
    active = comparison.loc[labels.eq("test"), "dynamic_selected"] - comparison.loc[
        labels.eq("test"), "static_inverse_volatility"
    ]
    bootstrap = block_bootstrap_information_ratio(
        active,
        samples=spec.bootstrap_samples,
        block_weeks=spec.bootstrap_block_weeks,
        seed=spec.seed,
    )
    concentration = _concentration(contributions, returns, spec)
    acceptance = _acceptance(summary, concentration, spec.primary_fee_bps_per_side)
    allocation_rows = []
    for name, allocations in allocation_map.items():
        item = allocations.stack(future_stack=True).rename("weight").reset_index()
        item.columns = ["ts", "style", "weight"]
        item["strategy"] = name
        allocation_rows.append(item)
    return {
        "features": features,
        "universe": universe_audit,
        "positions": positions,
        "style_returns": primary_style,
        "allocations": pd.concat(allocation_rows, ignore_index=True),
        "returns": returns,
        "contributions": contributions,
        "summary": summary,
        "yearly": yearly,
        "monthly": monthly,
        "regime": regime,
        "selection": selection,
        "selected_window": selected_window,
        "active_styles": active_styles,
        "inactive_styles": inactive_styles,
        "directions": directions,
        "bootstrap": bootstrap,
        "concentration": concentration,
        "acceptance": acceptance,
    }


def _research_status(spec: RotationSpec, panel: pd.DataFrame) -> tuple[str, list[str]]:
    limitations = []
    if not spec.universe.includes_delisted:
        limitations.append("历史股票池未证明包含后来下架的合约，存在存活偏差。")
    if not spec.universe.point_in_time_catalogue or "listing_time" not in panel:
        limitations.append("上市时间未由带版本的点时合约目录证明。")
    if len(spec.universe.instruments) < spec.universe.size:
        limitations.append("配置的候选合约少于目标股票池规模。")
    return ("formal" if not limitations else "pilot"), limitations


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer, np.floating)):
        return None if not np.isfinite(value) else value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _render_markdown(manifest: dict[str, Any], summary: pd.DataFrame) -> str:
    primary = summary[
        summary["fee_bps_per_side"].eq(manifest["spec"]["primary_fee_bps_per_side"])
        & summary["split"].isin(["validation", "test"])
    ]
    lines = [
        "# AlphaGYM Crypto 风格轮动研究", "",
        f"报告：{manifest['report_id']}；研究等级：{manifest['research_status'].upper()}。",
        f"冻结的轮动窗口：{manifest['selected_window_weeks']} 周。", "",
        "|策略|阶段|净 Sharpe|累计收益|最大回撤|平均换手|", "|---|---|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples():
        lines.append(
            f"|{row.strategy}|{row.split}|{row.annualized_sharpe:.3f}|"
            f"{row.cumulative_return:.2%}|{row.max_drawdown:.2%}|{row.average_turnover:.2f}|"
        )
    lines += ["", f"模拟盘门槛：{'通过' if manifest['acceptance']['passed'] else '未通过'}。", ""]
    lines += [
        f"OKX 交易大数据：{manifest['market_state']['usage']}；仅作状态归因，不参与当前选模。",
        "",
    ]
    if manifest["limitations"]:
        lines += ["## 限制", ""] + [f"- {item}" for item in manifest["limitations"]] + [""]
    lines.append("本报告为离线量化研究，不构成投资建议。")
    return "\n".join(lines) + "\n"


def _render_html(manifest: dict[str, Any], summary: pd.DataFrame, returns: pd.DataFrame) -> str:
    fee = manifest["spec"]["primary_fee_bps_per_side"]
    chart = returns[
        returns["fee_bps_per_side"].eq(fee)
        & returns["strategy"].isin([
            "dynamic_selected", "static_equal", "static_inverse_volatility"
        ])
    ].copy()
    chart["value"] = chart.groupby("strategy", observed=True)["net_return"].transform(
        lambda x: np.exp(x.fillna(0).cumsum()) - 1
    )
    dates = sorted(chart["ts"].unique())
    series = []
    for name, group in chart.groupby("strategy", observed=True):
        values = group.set_index("ts")["value"].reindex(dates)
        series.append({"name": name, "type": "line", "showSymbol": False,
                       "data": [None if pd.isna(x) else float(x) for x in values]})
    option = {
        "tooltip": {"trigger": "axis"}, "legend": {"type": "scroll"},
        "xAxis": {"type": "category", "data": [pd.Timestamp(x).isoformat() for x in dates]},
        "yAxis": {"type": "value", "name": "累计收益", "scale": True}, "series": series,
    }
    table = summary[
        summary["fee_bps_per_side"].eq(fee) & summary["split"].isin(["validation", "test"])
    ].to_html(index=False, float_format=lambda x: f"{x:.4f}")
    limitations = "".join(f"<li>{html.escape(x)}</li>" for x in manifest["limitations"])
    echarts = (Path(__file__).parent / "static" / "echarts.min.js").read_text(encoding="utf-8")
    safe_option = json.dumps(option, ensure_ascii=False).replace("</", "<\\/")
    return f'''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>AlphaGYM Crypto 风格轮动</title><style>
body{{font:15px system-ui;max-width:1500px;margin:32px auto;padding:0 24px;color:#183047}}
table{{border-collapse:collapse;font-size:12px}}td,th{{padding:7px;border:1px solid #ddd}}
th{{background:#eef3f8}}h1{{color:#123b5d}}#chart{{height:500px}}
.pilot{{padding:10px;background:#fff1c2;border:1px solid #e0bc4c}}
</style><h1>AlphaGYM — Crypto 量化因子风格轮动</h1>
<p class="pilot">研究等级：{manifest['research_status'].upper()}；报告 {manifest['report_id']}。</p>
<p>每周信号、下一根 4H 开盘成交；风格方向与 {manifest['selected_window_weeks']} 周轮动窗口
仅由开发/验证期确定。主成本为单边 {fee} bps，并计入资金费率。</p>
<p><strong>模拟盘门槛：{'通过' if manifest['acceptance']['passed'] else '未通过'}。</strong></p>
<p>OKX 交易大数据：{html.escape(manifest['market_state']['usage'])}；采用一天可得滞后，
当前只用于市场状态归因，不参与风格方向或轮动窗口选择。</p>
<div id="chart"></div><h2>验证与测试结果</h2>{table}<h2>限制</h2><ul>{limitations}</ul>
<p>本报告为离线量化研究，不构成投资建议。</p>
<script>{echarts}</script><script>echarts.init(document.getElementById('chart')).setOption({safe_option});</script>
</html>'''


def _panel_from_spec(root: Path, spec: RotationSpec, client=None, now=None) -> tuple[pd.DataFrame, str | None]:
    if spec.source_report_id:
        if not spec.source_report_id.startswith("crypto-"):
            raise ValueError("source_report_id must identify a crypto report")
        source = root / "factor_library" / "reports" / spec.source_report_id
        manifest = json.loads(storage_io.read_text(source / "manifest.json"))
        if manifest.get("bar") != "4H":
            raise ValueError("source report must contain 4H candles")
        return storage_io.read_frame(source / "candles.parquet"), spec.source_report_id
    hourly = HourlySpec(
        bar="4H", universe_size=max(spec.universe.size, len(spec.universe.instruments)),
        history_days=spec.history_days, horizons=(42,), minimum_bars=250,
        instruments=spec.universe.instruments, workers=spec.workers,
        minimum_listing_age_before_sample_days=0,
    )
    return download_panel(hourly, client=client, now=now), None


def run_style_rotation(
    root: str | Path, spec: RotationSpec, *, client=None, now=None
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    spec.validate()
    panel, source_report_id = _panel_from_spec(root, spec, client=client, now=now)
    result = evaluate_rotation(panel, spec)
    research_status, limitations = _research_status(spec, panel)
    if result["inactive_styles"]:
        limitations.append(
            "开发期历史覆盖不足，已停用风格："
            + "、".join(result["inactive_styles"])
            + "；其余风格的股票池不因该缺失被回溯性删除。"
        )
        research_status = "pilot"
    feature_splits = _split_labels(result["features"]["ts"], spec.splits)
    funding_coverage = {
        split: float(result["features"].loc[feature_splits.eq(split), "funding_rate"].notna().mean())
        for split in ("development", "validation", "test", "monitoring")
        if feature_splits.eq(split).any()
    }
    if any(value < 0.95 for split, value in funding_coverage.items() if split != "monitoring"):
        limitations.append(
            "历史资金费率未完整覆盖全部正式分栏；缺失区间的资金现金流不可恢复，"
            "该结果只能视为价格/成交量风格 PILOT。"
        )
        research_status = "pilot"
    market_state_joined = pd.DataFrame()
    market_state_summary = pd.DataFrame()
    market_state_metadata: dict[str, Any] = {
        "resource": spec.market_state_resource, "usage": "unavailable", "rows": 0,
    }
    store = storage_io.store_for(root, initialize=True)
    if spec.market_state_resource:
        state_meta = store.manifest(spec.market_state_resource)
        if state_meta and state_meta.get("kind") == "frame":
            state = store.read_frame(spec.market_state_resource)
            market_state_joined, market_state_summary = market_state_attribution(
                result["returns"], state, spec
            )
            covered_splits = sorted(market_state_joined.dropna(subset=["market_state"])[
                "split"
            ].unique().tolist())
            market_state_metadata = {
                "resource": spec.market_state_resource, "usage": "attribution_only",
                "rows": len(state), "covered_splits": covered_splits,
                "causal_availability_lag": "1D",
            }
            if not {"development", "validation"} <= set(covered_splits):
                limitations.append(
                    "OKX 交易大数据尚未覆盖开发与验证期，仅用于状态归因，不参与选模。"
                )
    digest_payload = {
        "spec": asdict(spec), "source": source_report_id, "rows": len(panel),
        "start": pd.to_datetime(panel["ts"], utc=True).min().isoformat(),
        "end": pd.to_datetime(panel["ts"], utc=True).max().isoformat(),
        "instruments": sorted(panel["instrument"].unique().tolist()),
    }
    report_id = "crypto-rotation-" + hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    manifest = _json_safe({
        "ok": True, "report_id": report_id, "report_type": "crypto_style_rotation",
        "created_at": datetime.now(UTC).isoformat(), "research_status": research_status,
        "source_report_id": source_report_id, "market": "OKX USDT linear perpetual swaps",
        "bar": "4H", "styles": list(result["active_styles"]),
        "inactive_styles": list(result["inactive_styles"]),
        "funding_data_coverage": funding_coverage,
        "strategies": sorted(result["summary"]["strategy"].unique().tolist()),
        "selected_window_weeks": result["selected_window"],
        "style_directions": result["directions"], "bootstrap": result["bootstrap"],
        "concentration": result["concentration"], "acceptance": result["acceptance"],
        "market_state": market_state_metadata,
        "limitations": limitations, "spec": asdict(spec),
        "selection_rule": "maximize min(development, validation) Sharpe; test and monitoring untouched",
        "execution_rule": "Monday 00:00 UTC bar close signal; next 4H open; hold one week",
    })
    spec_text = yaml.safe_dump(_json_safe(asdict(spec)), sort_keys=False, allow_unicode=True)
    markdown = _render_markdown(manifest, result["summary"])
    report_html = _render_html(manifest, result["summary"], result["returns"])
    prefix = f"factor_library/reports/{report_id}/"
    with store.batch() as batch:
        batch.blob(prefix + "spec.yaml", spec_text.encode("utf-8"))
        batch.blob(prefix + "manifest.json", json.dumps(
            manifest, ensure_ascii=False, indent=2
        ).encode("utf-8"))
        batch.frame(prefix + "universe.parquet", result["universe"])
        batch.frame(prefix + "features.parquet", result["features"])
        batch.frame(prefix + "positions.parquet", result["positions"])
        batch.frame(prefix + "style_returns.parquet", result["style_returns"])
        batch.frame(prefix + "allocations.parquet", result["allocations"])
        batch.frame(prefix + "returns.parquet", result["returns"])
        batch.frame(prefix + "contributions.parquet", result["contributions"])
        batch.frame(prefix + "summary.csv", result["summary"])
        batch.frame(prefix + "yearly.csv", result["yearly"])
        batch.frame(prefix + "monthly.parquet", result["monthly"])
        batch.frame(prefix + "regime.csv", result["regime"])
        batch.frame(prefix + "selection.csv", result["selection"])
        if not market_state_joined.empty:
            batch.frame(prefix + "market_state_returns.parquet", market_state_joined)
            batch.frame(prefix + "market_state_summary.csv", market_state_summary)
        batch.blob(prefix + "report.md", markdown.encode("utf-8"))
        batch.blob(prefix + "report.html", report_html.encode("utf-8"))
    return manifest
