"""Non-neutralized research, not an executable A-share backtest.

Fractional adjusted-price units represent total-return exposure. Raw-share
round lots, auction fills, ST/limit filters, dividend taxes and capacity are
not modeled. These limitations are written into every output manifest.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlquant import storage_io
from mlquant.account import FeeSchedule
from mlquant.combine import ML_METHODS, factor_weights
from mlquant.ml_composite import MODEL_KEYS, model_specs
from mlquant.ml_frequency import forward_label_end_dates, frequency_signal_dates
from mlquant.portfolio_evaluation import load_adjusted_market, performance_stats
from mlquant.research import newey_west_t

SPLITS = {
    "development": ("2014-01-01", "2020-12-31"),
    "validation": ("2021-01-01", "2023-12-31"),
    "test": ("2024-01-01", "2025-12-31"),
    "monitoring": ("2026-01-01", "2026-08-31"),
}
SCENARIOS = {"gross": None, "base_5bps": 5., "stress_10bps": 10., "stress_20bps": 20.}


def fractional_simulation(
    scores: np.ndarray, prices: np.ndarray, tradable: np.ndarray,
    executions: pd.DatetimeIndex, *, slippage_bps: float | None,
    top_n: int = 50, initial_cash: float = 1_000_000.,
) -> pd.DataFrame:
    """Cash-constrained long-only; costs occur before subsequent holding return.

    At most one rebalance per trading day implies no same-day sale of a new
    purchase. Unquoted holdings cannot be traded. The caller supplies PIT
    stale marks; we never discard a holding because its future quote is absent.
    """
    if len(executions) != len(prices) or scores.shape != prices.shape:
        raise ValueError("unaligned scores/prices/executions")
    if not executions.is_unique or not executions.is_monotonic_increasing:
        raise ValueError("executions must be unique and increasing")
    fees = FeeSchedule(minimum_commission=5.) if slippage_bps is not None else FeeSchedule(0., 0., False)
    slip = (slippage_bps or 0.) / 10_000
    units = np.zeros(prices.shape[1], dtype=float)
    cash = initial_cash
    rows = []
    for i, when in enumerate(executions):
        price = prices[i]
        equity = cash + float(units @ price)
        row = {"execution_date": when, "equity": equity, "turnover": 0., "explicit_cost": 0., "slippage_cost": 0., "blocked_target_weight": 0., "constant_score": False}
        rows.append(row)
        if i == len(executions) - 1:
            break
        finite = np.isfinite(scores[i])
        target = np.zeros(len(units), dtype=float)
        if finite.any() and np.std(scores[i, finite]) > 1e-12:
            candidates = np.flatnonzero(finite)
            chosen = candidates[np.argsort(-scores[i, candidates], kind="stable")[:top_n]]
            positive = price[chosen] > 0
            target[chosen[positive]] = equity * .995 / len(chosen) / price[chosen[positive]]
            row["blocked_target_weight"] = float((~tradable[i, chosen]).mean() * .995)
        else:
            row["constant_score"] = True
        allowed = tradable[i] & (price > 0)
        sell = np.where(allowed, np.maximum(units - target, 0), 0)
        sell_reference = sell * price
        sell_notional = sell_reference * (1 - slip)
        rates = fees.rates(when, "sell")
        sell_fee = np.where(sell_notional > 1e-9, np.maximum(fees.minimum_commission, sell_notional * rates["commission"]) + sell_notional * (rates["stamp_tax"] + rates["transfer_fee"]), 0.)
        # When equity has nearly exhausted, a CNY 5 minimum can exceed the
        # entire fractional position. Do not execute a negative-proceeds order.
        uneconomic = sell_fee >= sell_notional
        sell[uneconomic] = 0.
        sell_reference[uneconomic] = 0.
        sell_notional[uneconomic] = 0.
        sell_fee[uneconomic] = 0.
        cash += float(sell_notional.sum() - sell_fee.sum())
        units -= sell
        buy = np.where(allowed, np.maximum(target - units, 0), 0)
        buy_notional = buy * price * (1 + slip)
        rates = fees.rates(when, "buy")
        # Reserve every nonzero order's minimum fee before proportional scaling.
        active = buy_notional > 1e-9
        fixed_reserve = active.sum() * fees.minimum_commission
        needed = buy_notional.sum() * (1 + sum(rates.values())) + fixed_reserve
        if needed > cash:
            scale = max(cash - fixed_reserve, 0) / max(buy_notional.sum() * (1 + sum(rates.values())), 1e-12)
            buy *= min(scale, 1.)
            buy_notional = buy * price * (1 + slip)
        buy_fee = np.where(buy_notional > 1e-9, np.maximum(fees.minimum_commission, buy_notional * rates["commission"]) + buy_notional * rates["transfer_fee"], 0.)
        cash -= float(buy_notional.sum() + buy_fee.sum())
        units += buy
        if cash < -1e-6 or (units < -1e-9).any():
            raise AssertionError(f"cash/long-only invariant violated: date={when}, cash={cash}, minimum_units={units.min()}, equity={equity}, scenario={slippage_bps}")
        row["turnover"] = float((sell_reference.sum() + (buy * price).sum()) / equity)
        row["explicit_cost"] = float((sell_fee.sum() + buy_fee.sum()) / equity)
        row["slippage_cost"] = float((sell_reference.sum() + (buy * price).sum()) * slip / equity)
    frame = pd.DataFrame(rows)
    frame["end_date"] = frame.execution_date.shift(-1)
    frame["portfolio_return"] = frame.equity.shift(-1) / frame.equity - 1
    return frame.iloc[:-1].copy()


def _market(root: Path):
    daily, status, calendar = load_adjusted_market(root, start="2013-09-01", end="2026-09-03")
    del status
    opened = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_open, "trade_date"])).sort_values()
    quote = daily.pivot(index="trade_date", columns="symbol", values="adj_open").reindex(opened).astype(np.float32)
    observed = quote.notna() & quote.gt(0)
    # Fixed, past-only stress convention: carry a missing quote for 60 market
    # sessions, then mark zero, with recovery on a later real quote. No future
    # delisting date is used to select or value earlier positions.
    marks = quote.where(observed).ffill(limit=60).fillna(0.).astype(np.float32)
    del daily, quote
    gc.collect()
    return marks, observed, calendar


def _load_horizon(paths: list[Path], signals: pd.DatetimeIndex) -> pd.DataFrame:
    blocks = []
    for path in paths:
        frame = storage_io.read_frame(path)
        blocks.append(frame[frame.signal_date.isin(signals)])
    return pd.concat(blocks, ignore_index=True).set_index(["signal_date", "symbol"]).sort_index()


def run_comparison(root: Path, output: Path, *, train_row_cap: int = 200_000) -> None:
    manifest = json.loads(storage_io.read_text(output / "feature_manifest.json", encoding="utf-8"))
    paths, factors = [Path(p) for p in manifest["files"]], manifest["factors"]
    marks, observed, calendar = _market(root)
    opened = marks.index
    metadata = {
        "formal": False, "watermark": "EXPLORATORY / NON-NEUTRALIZED / NOT LIVE-TRADABLE",
        "features": "same 20 registered curated factors; 5 MAD + cross-sectional z; no industry neutralization",
        "pool": "ALL_A, at least 250 trading sessions since listing, positive signal-day close and float cap; includes ST",
        "financial_alignment": "available_date event replay; no interpolation; revisions preserved when supplied",
        "training": "2014-2020 with complete-label purge; fixed seed 42; no new tuning; dev performance is in-sample",
        "curation_caveat": "existing 20-factor pool was selected in earlier validation research; this is a post-hoc reassessment, not a new untouched test",
        "train_row_cap": train_row_cap, "top_n": 50, "initial_cash": 1_000_000,
        "capital_reset": "each development/validation/test/monitoring segment starts with CNY 1 million cash, including its initial entry costs",
        "execution": "close signal -> next session open; fractional adjusted-price exposure, long-only and cash constrained",
        "costs": "3 bps commission, minimum CNY 5/order; historical stamp/transfer schedule; 0/5/10/20 bps one-way slippage",
        "benchmark": "same eligible universe signal-date float-cap weighted, rebalanced at each horizon; synthetic not official index",
        "missing_marks": "carry past open up to 60 trading sessions then zero; quoted recovery allowed; no future deletion/renormalization",
        "limitations": ["No historical ST exclusions or opening price-limit rejection", "No round lots, auction/volume capacity or dividend tax", "Financial source vintages need further verification", "Adjustment return exposure is not a raw-share corporate-action ledger", "Open-to-open drawdowns omit intraperiod lows", "Early historical transfer fees approximated at 0.2 bps before 2022-04-29"],
        "splits": SPLITS, "feature_signature": manifest["signature"],
    }
    storage_io.write_text(output / "metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    summaries = []
    for frequency in ("monthly", "weekly", "daily"):
        signals = frequency_signal_dates(calendar, "2014-01-01", "2026-08-31", frequency)
        signals = signals[opened.searchsorted(signals, side="right") < len(opened)]
        executions = opened[opened.searchsorted(signals, side="right")]
        data = _load_horizon(paths, signals)
        symbols = marks.columns
        all_quotes = marks.reindex(executions).to_numpy(float)
        all_tradable = observed.reindex(executions).to_numpy(bool)
        label_values = np.divide(all_quotes[1:], all_quotes[:-1], out=np.full_like(all_quotes[:-1], np.nan), where=all_quotes[:-1] > 0) - 1
        label_frame = pd.DataFrame(label_values, index=signals[:-1], columns=symbols)
        labels = label_frame.stack(future_stack=True).reindex(data.index)
        label_ends = forward_label_end_dates(signals, calendar)
        dates = data.index.get_level_values("signal_date")
        train_mask = (dates >= pd.Timestamp(SPLITS["development"][0])) & (dates.map(label_ends) <= pd.Timestamp(SPLITS["development"][1])) & labels.notna().to_numpy()
        training = data.loc[train_mask, factors]
        y = labels.loc[train_mask]
        if len(training) > train_row_cap:
            positions = np.sort(np.random.default_rng(42).choice(len(training), train_row_cap, replace=False))
            training, y = training.iloc[positions], y.iloc[positions]
        imputation = training.median().fillna(0.)
        x = training.fillna(imputation).to_numpy(np.float32)
        y_array = y.to_numpy(np.float32)
        training_info = {"rows": len(x), "max_signal": str(training.index.get_level_values(0).max()), "max_label_end": str(training.index.get_level_values(0).map(label_ends).max())}
        print(f"{frequency} training ready: {training_info}", flush=True)
        # Factor-weight ML learns IC at the corresponding holding horizon, not
        # monthly labels copied onto daily signals. Legacy lag counts are kept.
        train_data = data.loc[train_mask, factors]
        ic_rows = {}
        for date, block in train_data.groupby(level="signal_date", sort=True):
            ic_rows[date] = block.rank().corrwith(labels.reindex(block.index).rank())
        ic = pd.DataFrame(ic_rows).T.reindex(columns=factors)
        signs = np.sign(ic.mean()).replace(0., 1.).fillna(1.)
        weights = {"equal_factor": pd.Series(1 / len(factors), index=factors)}
        for method in ML_METHODS:
            weights[f"factor_ml_{method}"] = factor_weights(ic.mul(signs), method).reindex(factors).fillna(0.)
        # Freeze simulation selection after validation-era data; retain dev
        # as explicitly in-sample diagnostics, never evidence of profitability.
        cap = data.float_market_cap.unstack("symbol").reindex(index=signals, columns=symbols).fillna(0.).to_numpy(float)
        cap /= np.maximum(cap.sum(axis=1, keepdims=True), 1.)
        benchmark = np.nansum(cap[:-1] * np.where(all_tradable[:-1], label_values, 0.), axis=1)
        # Missing start quotes are uninvested cash; do not redistribute weight.
        equal_pool = (cap > 0).astype(float)
        equal_pool /= np.maximum(equal_pool.sum(axis=1, keepdims=True), 1.)
        equal_benchmark = np.nansum(equal_pool[:-1] * np.where(all_tradable[:-1], label_values, 0.), axis=1)
        del equal_pool, train_data
        methods = list(weights) + [f"stock_ml_{model}" for model in MODEL_KEYS]
        for method in methods:
            destination = output / frequency / method
            destination.mkdir(parents=True, exist_ok=True)
            summary_path = destination / "summary.json"
            if storage_io.exists(summary_path):
                summaries.extend(json.loads(storage_io.read_text(summary_path, encoding="utf-8")))
                continue
            if method in weights:
                weight = weights[method] * signs
                valid_weight = data[factors].notna().mul(weight.abs()).sum(axis=1)
                score = data[factors].mul(weight).sum(axis=1) / valid_weight.replace(0, np.nan)
            else:
                model = method.removeprefix("stock_ml_")
                estimator = model_specs()[model]["factory"](42)
                estimator.fit(x, y_array)
                prediction = np.empty(len(data), dtype=np.float32)
                for begin in range(0, len(data), 100_000):
                    block = data.iloc[begin:begin+100_000][factors].fillna(imputation).to_numpy(np.float32)
                    prediction[begin:begin+len(block)] = estimator.predict(block)
                score = pd.Series(prediction, index=data.index)
                del estimator
            matrix = score.unstack("symbol").reindex(index=signals, columns=symbols).to_numpy(float)
            method_rows = []
            for scenario, slip in SCENARIOS.items():
                segment_frames = []
                for period, (start, end) in SPLITS.items():
                    positions = np.flatnonzero((executions[:-1] >= pd.Timestamp(start)) & (executions[1:] <= pd.Timestamp(end)))
                    if len(positions) == 0:
                        raise ValueError(f"no complete holding periods: {frequency} {period}")
                    begin, stop = positions[0], positions[-1] + 2
                    part = fractional_simulation(matrix[begin:stop], all_quotes[begin:stop], all_tradable[begin:stop], executions[begin:stop], slippage_bps=slip)
                    part["signal_date"] = signals[positions]
                    part["benchmark_return"] = benchmark[positions]
                    part["equal_universe_return"] = equal_benchmark[positions]
                    part["period"] = period
                    segment_frames.append(part)
                    years = (part.end_date - part.execution_date).dt.days.sum() / 365.25
                    annual_periods = len(part) / years if years > 0 else 1
                    active = (1 + part.portfolio_return) / (1 + part.benchmark_return) - 1
                    equal_stats = performance_stats(part.portfolio_return, part.equal_universe_return, periods_per_year=annual_periods)
                    method_rows.append({
                        "frequency": frequency, "method": method, "scenario": scenario, "period": period,
                        **performance_stats(part.portfolio_return, part.benchmark_return, periods_per_year=annual_periods),
                        "annual_turnover": part.turnover.sum() / years if years > 0 else np.nan,
                        "annual_explicit_cost": part.explicit_cost.sum() / years if years > 0 else np.nan,
                        "annual_slippage_cost": part.slippage_cost.sum() / years if years > 0 else np.nan,
                        "active_newey_west_t": newey_west_t(active),
                        "equal_universe_annualized_return": equal_stats.get("benchmark_annualized_return"),
                        "excess_vs_equal_universe": equal_stats.get("annualized_excess_return"),
                        "constant_score_fraction": float(part.constant_score.mean()),
                        "blocked_target_weight": float(part.blocked_target_weight.mean()),
                        "evaluation_years": years, "development_is_in_sample": period == "development",
                    })
                storage_io.write_frame(pd.concat(segment_frames, ignore_index=True), destination / f"{scenario}.parquet", index=False)
            storage_io.write_text(destination / "model.json", json.dumps({"training": training_info, "signs": signs.to_dict(), "weights": weights[method].to_dict() if method in weights else None}, indent=2), encoding="utf-8")
            storage_io.write_text(summary_path, json.dumps(method_rows, indent=2), encoding="utf-8")
            summaries.extend(method_rows)
            storage_io.write_csv(pd.DataFrame(summaries), output / "summary.csv", index=False, encoding="utf-8-sig")
            print(f"finished {frequency} {method} (all costs)", flush=True)
            del score, matrix
            gc.collect()
        del data, training, labels, label_frame, x, y_array, all_quotes, all_tradable
        gc.collect()
