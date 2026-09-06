"""Experimental PIT weekly/daily evaluation of every ML combination."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mlquant import storage_io
from mlquant.combine import ML_METHODS, combine_scores, factor_weights
from mlquant.ml_composite import fit_imputation, model_specs, prepare_matrices
from mlquant.ml_frequency import (
    add_financial_features,
    forward_label_end_dates,
    forward_open_returns,
    frequency_signal_dates,
    load_frequency_inputs,
    market_features,
    neutralize_frequency_features,
)
from mlquant.portfolio_evaluation import (
    DEFAULT_COST_SCENARIOS,
    benchmark_returns,
    load_adjusted_market,
    performance_stats,
    prepare_market,
    simulate_long_only,
)
from mlquant.reassessment_audit import require_reassessment_data
from mlquant.report_engine import build_cross_section
from mlquant.report_spec import parse_spec
from mlquant.research import zscore
from mlquant.workflows._support import finish, invoke, progress, report_record, validate_config


def _feature_files(
    root: Path,
    cache: Path,
    frequency: str,
    selected: list[str],
    calendar: pd.DataFrame,
    industries: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    development_end: pd.Timestamp,
) -> list[Path]:
    sources = [root / "equity" / f"{name}.parquet" for name in
               ("daily", "adjustments", "fundamentals", "industries", "calendar")]
    signature = hashlib.sha256(json.dumps({
        "version": "pit_v2", "factors": selected, "start": str(start), "end": str(end),
        "sources": [(str(p), storage_io.stat(p).st_size, storage_io.stat(p).st_mtime_ns) for p in sources],
        "feature_code": hashlib.sha256(storage_io.read_bytes(Path(sys.modules[market_features.__module__].__file__))).hexdigest(),
    }, sort_keys=True).encode()).hexdigest()[:20]
    destination = cache / signature / frequency
    destination.mkdir(parents=True, exist_ok=True)
    all_dates = frequency_signal_dates(calendar, start, end, frequency)
    files: list[Path] = []
    for year in range(start.year, end.year + 1):
        path = destination / f"features_{year}.parquet"
        files.append(path)
        if storage_io.exists(path):
            continue
        year_dates = all_dates[all_dates.year == year]
        if year_dates.empty:
            continue
        position = all_dates.searchsorted(year_dates[-1])
        label_dates = year_dates
        if position + 1 < len(all_dates):
            label_dates = year_dates.append(pd.DatetimeIndex([all_dates[position + 1]]))
        feature_dates = year_dates
        daily, fundamentals, _calendar, _adjustments = load_frequency_inputs(
            root, year_dates[0], year_dates[-1]
        )
        raw, observations = market_features(daily, feature_dates, selected)
        raw = add_financial_features(raw, observations, fundamentals, selected)
        forward = forward_open_returns(daily, label_dates, calendar).reindex(raw.index)
        label_ends = forward_label_end_dates(label_dates, calendar)
        cap = observations["float_market_cap"].reindex(raw.index)
        neutral = neutralize_frequency_features(raw, industries, cap)
        frame = pd.concat(
            [raw.add_prefix("raw__"), neutral.add_prefix("neutral__"), forward], axis=1
        ).reset_index()
        frame["label_end_date"] = frame["signal_date"].map(label_ends)
        storage_io.write_frame(frame, path, index=False)
        progress(f"cached {frequency} {year}: {len(frame):,} rows", flush=True)
        del daily, fundamentals, raw, observations, forward, cap, neutral, frame
        gc.collect()
    return [path for path in files if storage_io.exists(path)]


def _read_features(paths: list[Path], columns: list[str] | None = None) -> pd.DataFrame:
    frames = [storage_io.read_frame(path, columns=columns) for path in paths]
    result = pd.concat(frames, ignore_index=True)
    result["signal_date"] = pd.to_datetime(result["signal_date"])
    return result.set_index(["signal_date", "symbol"]).sort_index()


def _fit_model(
    paths: list[Path],
    selected: list[str],
    model: str,
    development: tuple[pd.Timestamp, pd.Timestamp],
    train_row_cap: int,
) -> tuple[object, pd.Series]:
    columns = ["signal_date", "symbol", "forward_return", "label_end_date"] + [
        f"neutral__{name}" for name in selected
    ]
    development_paths = [
        path for path in paths if int(path.stem[-4:]) <= development[1].year
    ]
    training = _read_features(development_paths, columns)
    dates = training.index.get_level_values("signal_date")
    training = training.loc[
        (dates >= development[0]) & (dates <= development[1])
        & training["label_end_date"].notna()
        & (training["label_end_date"] <= development[1])
    ]
    features = training[[f"neutral__{name}" for name in selected]].rename(
        columns=lambda name: name.removeprefix("neutral__")
    )
    forward = training["forward_return"]
    valid_index = forward[forward.notna()].index
    if len(valid_index) > train_row_cap:
        random = np.random.default_rng(42)
        chosen = np.sort(random.choice(len(valid_index), train_row_cap, replace=False))
        valid_index = valid_index[chosen]
    features = features.loc[valid_index]
    forward = forward.loc[valid_index]
    imputation = fit_imputation(features)
    x_train, y_train, _labels = prepare_matrices(
        features, forward, imputation=imputation, label_mode="return"
    )
    estimator = model_specs()[model]["factory"](42)
    estimator.fit(x_train, y_train)
    progress(f"fit {model}: {len(x_train):,} rows", flush=True)
    return estimator, imputation


def _stock_scores(
    paths: list[Path],
    selected: list[str],
    estimator: object,
    imputation: pd.Series,
) -> pd.Series:
    columns = ["signal_date", "symbol"] + [f"neutral__{name}" for name in selected]
    blocks = []
    for path in paths:
        frame = _read_features([path], columns)
        features = frame.rename(columns=lambda name: name.removeprefix("neutral__"))
        matrix = features.fillna(imputation).fillna(0.0).to_numpy(np.float32)
        blocks.append(pd.Series(estimator.predict(matrix).astype(np.float32), index=features.index))
    return pd.concat(blocks).sort_index()


def _factor_scores(
    paths: list[Path], selected: list[str], signs: pd.Series, weights: pd.Series
) -> pd.Series:
    columns = ["signal_date", "symbol"] + [f"raw__{name}" for name in selected]
    blocks = []
    for path in paths:
        frame = _read_features([path], columns)
        raw = frame.rename(columns=lambda name: name.removeprefix("raw__"))
        standardized = raw.groupby(level="signal_date", observed=True).transform(zscore)
        signed = standardized.mul(signs.reindex(selected).fillna(1.0), axis=1)
        blocks.append(
            signed.groupby(level="signal_date", observed=True, group_keys=False).apply(
                lambda block: combine_scores(block, weights)
            )
        )
    return pd.concat(blocks).sort_index()


def _append_metrics(
    rows: list[dict[str, object]],
    aligned: pd.DataFrame,
    diagnostics: dict[str, float],
    *,
    report_id: str,
    frequency: str,
    method: str,
    scenario: str,
    periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    periods_per_year: int,
) -> None:
    years = max(len(aligned) / periods_per_year, 1 / periods_per_year)
    for period, (start, end) in periods.items():
        part = aligned.loc[(aligned.index >= start) & (aligned.index <= end)]
        rows.append(
            {
                "report_id": report_id,
                "index_code": "ALL_A",
                "frequency": frequency,
                "method": method,
                "cost_scenario": scenario,
                "period": period,
                **performance_stats(
                    part["portfolio_return"],
                    part["benchmark_return"],
                    periods_per_year=periods_per_year,
                ),
                "full_sample_annual_turnover": diagnostics["notional"]
                / diagnostics["average_equity"]
                / years,
                "full_sample_annual_explicit_cost": diagnostics["explicit_cost"]
                / diagnostics["average_equity"]
                / years,
                "full_sample_annual_slippage_cost": diagnostics["slippage_cost"]
                / diagnostics["average_equity"]
                / years,
                "filled_orders": diagnostics["filled_orders"],
                "rejected_orders": diagnostics["rejected_orders"],
            }
        )




@dataclass(frozen=True, slots=True, kw_only=True)
class Config:
    """Explicit workflow inputs; paths never depend on the source checkout."""
    root: Path
    report_id: str
    frequency: list[str] | None = None
    top_n: int = 50
    train_row_cap: int = 500000
    output: Path
    json: bool = False

    def __post_init__(self):
        validate_config(self, {'frequency': ('weekly', 'daily')})


def run(args: Config) -> dict:
    """Execute with Python inputs and return artifact metadata; never exits the host."""
    frequencies = args.frequency or ["weekly", "daily"]
    root = Path(args.root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        require_reassessment_data(root, output)
    except ValueError as error:
        progress(json.dumps({"ok": False, "error": {"code": "REASSESSMENT_DATA_BLOCKED", "message": str(error)}}))
        raise
    row = report_record(root, args.report_id)
    report_dir = Path(row["path"])
    spec = parse_spec(json.loads(row["spec_json"]))
    if spec.universe.index_code != "ALL_A":
        raise ValueError("high-frequency experiment currently requires the ALL_A report")
    manifest = json.loads(storage_io.read_text(report_dir / "manifest.json", encoding="utf-8"))
    combo = json.loads(storage_io.read_text(report_dir / "combo.json", encoding="utf-8"))
    selection = combo["selection"]
    selected = [str(item) for item in selection["selected_factors"]]
    signs = pd.Series({str(k): float(v) for k, v in selection["signs"].items()})
    models = [str(item) for item in selection["ml"]["config"]["models"]]
    report_factors = [str(item["factor_id"]) for item in manifest["factors"]]
    monthly_path = root / "factor_library" / "generated_panels" / f"{manifest['run_id']}.parquet"
    _wide, _forward, _z, monthly_ic, _fm = build_cross_section(monthly_path, report_factors)
    splits = spec.resolved_splits()
    end = spec.monitoring_bound() or splits["test"][1]
    calendar = storage_io.read_frame(root / "equity" / "calendar.parquet")
    calendar["trade_date"] = pd.to_datetime(calendar["trade_date"]).dt.normalize()
    industries = storage_io.read_frame(root / "equity" / "industries.parquet")
    for column in ("valid_from", "valid_to"):
        industries[column] = pd.to_datetime(industries[column])
    summary_rows: list[dict[str, object]] = []
    return_blocks: list[pd.DataFrame] = []
    for frequency in frequencies:
        paths = _feature_files(
            root,
            output / "feature_cache",
            frequency,
            selected,
            calendar,
            industries,
            splits["development"][0],
            end,
            splits["development"][1],
        )
        daily, status, market_calendar = load_adjusted_market(
            root, start=splits["development"][0], end=end + pd.Timedelta(days=15)
        )
        prepared = prepare_market(daily, status)
        index_frame = _read_features(paths, ["signal_date", "symbol"])
        dummy = pd.DataFrame({"universe": 0.0}, index=index_frame.index)
        benchmark = benchmark_returns(dummy, daily, market_calendar)
        del index_frame, dummy
        periods = {**splits}
        if spec.monitoring_bound() is not None:
            periods["monitoring"] = (
                splits["test"][1] + pd.Timedelta(days=1),
                spec.monitoring_bound(),
            )
        periods_per_year = 252 if frequency == "daily" else 52
        def evaluate(
            method: str,
            score: pd.Series,
            *,
            daily_data: pd.DataFrame = daily,
            status_data: pd.DataFrame = status,
            calendar_data: pd.DataFrame = market_calendar,
            prepared_data=prepared,
            benchmark_data: pd.Series = benchmark,
            frequency_name: str = frequency,
            period_bounds=periods,
            annual_periods: int = periods_per_year,
        ) -> None:
            for scenario in DEFAULT_COST_SCENARIOS:
                returns, _trades, diagnostics = simulate_long_only(
                    score,
                    daily_data,
                    status_data,
                    calendar_data,
                    top_n=args.top_n,
                    scenario=scenario,
                    prepared_market=prepared_data,
                )
                aligned = pd.concat([returns, benchmark_data], axis=1)
                _append_metrics(
                    summary_rows,
                    aligned,
                    diagnostics,
                    report_id=args.report_id,
                    frequency=frequency_name,
                    method=method,
                    scenario=scenario.name,
                    periods=period_bounds,
                    periods_per_year=annual_periods,
                )
                block = aligned.copy()
                block["frequency"] = frequency_name
                block["method"] = method
                block["cost_scenario"] = scenario.name
                block.index.name = "signal_date"
                return_blocks.append(block.reset_index())
                progress(f"finished {frequency_name} {method} {scenario.name}", flush=True)

        monthly_ends = forward_label_end_dates(pd.DatetimeIndex(monthly_ic.index), calendar)
        history = monthly_ic.loc[
            monthly_ends.notna() & (monthly_ends <= splits["validation"][1]), selected
        ]
        for method in ML_METHODS:
            weights = factor_weights(history, method)
            score = _factor_scores(paths, selected, signs, weights)
            evaluate(f"factor_ml_{method}", score)
            del score
            gc.collect()
        for model in models:
            estimator, imputation = _fit_model(
                paths, selected, model, splits["development"], args.train_row_cap
            )
            score = _stock_scores(paths, selected, estimator, imputation)
            evaluate(f"stock_ml_{model}", score)
            del estimator, score
            gc.collect()
        del daily, status, prepared
        gc.collect()
    storage_io.write_csv(pd.DataFrame(summary_rows), output / "summary.csv", index=False, encoding="utf-8-sig")
    storage_io.write_frame(pd.concat(return_blocks, ignore_index=True), output / "returns.parquet", index=False)
    metadata = {
        "report_id": args.report_id,
        "frequencies": frequencies,
        "top_n": args.top_n,
        "long_only": True,
        "feature_mode": "PIT neutral",
        "financial_alignment": "available_date backward-asof carry-forward",
        "daily_development_date_stride": 1,
        "train_row_cap": args.train_row_cap,
        "seed": 42,
        "labels": "next-rebalance open to following-rebalance open",
        "execution": "signal close, next trading day open",
    }
    storage_io.write_text(output / "metadata.json",
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return finish(output=args.output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--report-id", required=True)
    parser.add_argument("--frequency", action="append", choices=("weekly", "daily"))
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--train-row-cap", type=int, default=500_000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    return invoke(run, Config, args)


if __name__ == "__main__":
    raise SystemExit(main())
