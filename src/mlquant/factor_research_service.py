from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlquant import storage_io
from mlquant.factor_cache import FactorValueCache, universe_key
from mlquant.factor_store import FactorStore
from mlquant.factors.base import FactorContext, FactorRegistry, FactorSpec
from mlquant.factors.technical import TECHNICAL_FACTOR_NAMES, compute_technical_features
from mlquant.research import evaluate_factor_batch

FORMAL_INDICES = {"000300.SH", "000905.SH", "000852.SH"}
GROUP_COLUMNS = [f"group_{number}" for number in range(1, 6)]
PORTFOLIO_ORDER = [*GROUP_COLUMNS, "benchmark", "long_short"]


class AutoRunDataError(ValueError):
    """The formula is valid, but its point-in-time inputs are not available yet."""


def _performance(values: pd.Series) -> dict[str, float]:
    returns = pd.to_numeric(values, errors="coerce").dropna()
    if returns.empty:
        return {}
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    annual_return = float(wealth.iloc[-1] ** (12 / len(returns)) - 1)
    annual_volatility = float(returns.std(ddof=1) * np.sqrt(12))
    downside = float(returns.clip(upper=0).std(ddof=1) * np.sqrt(12))
    maximum_drawdown = float(drawdown.min())
    return {
        "annualized_return": annual_return,
        "annualized_volatility": annual_volatility,
        "sharpe": annual_return / annual_volatility if annual_volatility else np.nan,
        "sortino": annual_return / downside if downside else np.nan,
        "max_drawdown": maximum_drawdown,
        "calmar": annual_return / abs(maximum_drawdown) if maximum_drawdown else np.nan,
        "monthly_win_rate": float((returns > 0).mean()),
    }


def _period_key(value: object) -> str:
    if pd.isna(value):
        return "monitoring_2026"
    return str(value)


def _period_samples(monthly: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    samples = [
        (_period_key(period), sample.sort_values("signal_date"))
        for period, sample in monthly.groupby("period", observed=True, dropna=False)
        if _period_key(period) != "all"
    ]
    samples.append(("all", monthly.sort_values("signal_date")))
    return samples


def _descending_groups(monthly: pd.DataFrame) -> pd.DataFrame:
    """Normalize legacy monthly artifacts so group 1 always has the highest values."""
    result = monthly.copy()
    if "group_order" in result:
        legacy = result["group_order"].fillna("") != "descending"
    else:
        legacy = pd.Series(True, index=result.index)
    if legacy.any():
        original = result.copy()
        for number in range(1, 6):
            result.loc[legacy, f"group_{number}"] = original.loc[
                legacy, f"group_{6 - number}"
            ].to_numpy()
    result["group_order"] = "descending"
    return result


def _layer_returns(sample: pd.DataFrame) -> pd.DataFrame:
    available = [column for column in GROUP_COLUMNS if column in sample]
    if len(available) != len(GROUP_COLUMNS):
        missing = sorted(set(GROUP_COLUMNS) - set(available))
        raise ValueError(f"monthly statistics missing layered returns: {missing}")
    result = sample[["signal_date", *GROUP_COLUMNS]].copy()
    result["benchmark"] = result[GROUP_COLUMNS].mean(axis=1)
    result["long_short"] = result["group_1"] - result["group_5"]
    return result


def _layer_performance(
    returns: pd.Series,
    benchmark: pd.Series,
    *,
    compare_with_benchmark: bool,
) -> dict[str, float]:
    metrics = _performance(returns)
    if not compare_with_benchmark:
        metrics["annualized_excess_return"] = np.nan
        metrics["information_ratio"] = np.nan
        return metrics
    aligned = pd.concat(
        [pd.to_numeric(returns, errors="coerce"), benchmark], axis=1
    ).dropna()
    if aligned.empty:
        metrics["annualized_excess_return"] = np.nan
        metrics["information_ratio"] = np.nan
        return metrics
    portfolio_annual = metrics.get("annualized_return", np.nan)
    benchmark_annual = _performance(aligned.iloc[:, 1]).get("annualized_return", np.nan)
    excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    tracking_error = float(excess.std(ddof=1) * np.sqrt(12))
    metrics["annualized_excess_return"] = portfolio_annual - benchmark_annual
    metrics["information_ratio"] = (
        float(excess.mean() * 12 / tracking_error) if tracking_error else np.nan
    )
    return metrics


class FactorResearchService:
    def __init__(self, store: FactorStore) -> None:
        self.store = store

    def validate_auto_request(
        self,
        factor_ids: list[str],
        *,
        mode: str,
        index_code: str,
        start_date: str,
        end_date: str,
        allow_monitoring: bool = False,
    ) -> dict[str, object]:
        """Validate formula dependencies against the registered local data catalog."""
        if not factor_ids:
            raise ValueError("请至少选择一个因子")
        start, end = pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize()
        if start >= end:
            raise ValueError("回测开始日期必须早于结束日期")
        if end.year >= 2026 and not allow_monitoring:
            raise ValueError("首期研究的结束日期不能包含 2026；2026 仅用于独立监控")
        revisions = self._locked_specs(factor_ids)
        required_fields = sorted({
            field for spec in revisions.values() for field in spec.input_fields
        })
        root = self._data_root()
        daily_path = root / "equity" / "daily.parquet"
        adjustments_path = root / "equity" / "adjustments.parquet"
        missing: list[str] = []
        if not storage_io.exists(daily_path):
            missing.append("A股日线行情尚未入库（equity/daily.parquet）")
            daily_columns: set[str] = set()
        else:
            daily_columns = self._parquet_columns(daily_path)

        financial_fields = [item for item in required_fields if item.startswith("financial.")]
        fundamentals_path = root / "equity" / "fundamentals.parquet"
        if financial_fields:
            if not storage_io.exists(fundamentals_path):
                missing.append("财务数据：" + "、".join(financial_fields))
            else:
                columns = self._parquet_columns(fundamentals_path)
                absent = [
                    item for item in financial_fields
                    if self.store.fields.get(item).column not in columns
                ]
                if absent:
                    missing.append("财务字段：" + "、".join(absent))

        market_fields = [item for item in required_fields if item.startswith("market.")]
        adjusted_required = any(
            self.store.fields.get(item).column.startswith("adj_") for item in market_fields
        )
        if adjusted_required and not storage_io.exists(adjustments_path):
            missing.append("后复权因子尚未入库（equity/adjustments.parquet）")
        derived = {
            "adj_open", "adj_high", "adj_low", "adj_close", "market_return",
        }
        absent_market = [
            item for item in market_fields
            if self.store.fields.get(item).column not in daily_columns
            and self.store.fields.get(item).column not in derived
        ]
        if absent_market:
            missing.append("行情字段：" + "、".join(absent_market))

        if index_code != "ALL_A" and not storage_io.exists(root / "equity" / "index_members.parquet"):
            missing.append(f"{index_code} 的历史成分数据")
        # The automatic factor backtest runs raw-value five-group portfolios
        # without industry layering or neutralization, so formal validation is the
        # same per-input availability check above: historical SW1 mapping and
        # official benchmark weights gate the industry-layered pipeline, not here.
        model_ids = self._required_models(revisions)
        if model_ids:
            missing.append("模型推理数据：" + "、".join(model_ids))
        if missing:
            raise AutoRunDataError(
                f"当前不能直接回测（数据根：{root}）："
                + "；".join(dict.fromkeys(missing))
            )
        return {
            "fields": required_fields,
            "lookback_days": max(spec.lookback_days for spec in revisions.values()),
            "data_root": str(root),
        }

    @storage_io.freeze_market
    def execute_auto_run(self, run_id: str) -> Path:
        """Build a reproducible panel from registered inputs, then evaluate it."""
        run = self.store.run_detail(run_id)
        config = run["config"]
        factor_ids = [item["factor_id"] for item in run["factors"]]
        self.store.set_run_status(run_id, "running", progress=0.02)
        try:
            self.validate_auto_request(
                factor_ids,
                mode=run["mode"],
                index_code=str(config["index_code"]),
                start_date=str(config["start_date"]),
                end_date=str(config["end_date"]),
                allow_monitoring=bool(config.get("monitoring_2026_included", False)),
            )
            panel = self._build_auto_panel(run)
            if panel.empty:
                raise AutoRunDataError("指定区间没有可评估的因子观测")
            panel_dir = self._data_root() / "factor_library" / "generated_panels"
            panel_dir.mkdir(parents=True, exist_ok=True)
            panel_path = panel_dir / f"{run_id}.parquet"
            storage_io.write_frame(panel, panel_path, index=False)
            output = self.execute_panel_run(run_id, panel_path)
            self.store.add_artifact(run_id, panel_path, "factor_values")
            return output
        except Exception as error:
            latest = self.store.run_detail(run_id)
            if latest["status"] != "failed":
                self.store.set_run_status(run_id, "failed", progress=1.0, error=str(error))
            raise

    def prepare_auto_compute(
        self, locked: dict[str, str], index_code: str,
        start: pd.Timestamp, end: pd.Timestamp,
    ) -> tuple[
        pd.DataFrame, pd.DataFrame, list[pd.Timestamp], pd.Series,
        dict[pd.Timestamp, pd.Timestamp | None], pd.Series,
    ]:
        """Load the inputs and signal scaffolding shared by run and cache paths."""
        registry = self.store.load_registry()
        specs = self._expand_revision_specs(registry, list(locked.values()))
        lookback = max((spec.lookback_days for spec in specs.values()), default=0)
        root = self._data_root()
        daily_path = root / "equity" / "daily.parquet"
        required_fields = {field for spec in specs.values() for field in spec.input_fields}
        daily = self._load_daily(
            daily_path,
            root / "equity" / "adjustments.parquet",
            required_fields,
            start - pd.Timedelta(days=max(45, lookback * 2 + 15)),
            end + pd.Timedelta(days=70),
        )
        fundamentals_path = root / "equity" / "fundamentals.parquet"
        fundamentals = (
            storage_io.read_frame(fundamentals_path)
            if storage_io.exists(fundamentals_path)
            else pd.DataFrame(columns=["symbol", "stat_date", "available_date"])
        )
        for column in ("stat_date", "available_date"):
            if column in fundamentals:
                fundamentals[column] = pd.to_datetime(fundamentals[column]).dt.normalize()

        opened = pd.Series(pd.to_datetime(daily["trade_date"].unique())).sort_values()
        month_ends = opened.groupby(opened.dt.to_period("M")).max().tolist()
        next_month = {
            date: next((candidate for candidate in month_ends if candidate > date), None)
            for date in month_ends
        }
        open_prices = daily.set_index(["trade_date", "symbol"])["adj_open"].sort_index()
        return daily, fundamentals, month_ends, opened, next_month, open_prices

    def _compute_block(
        self,
        signals: list[pd.Timestamp],
        opened: pd.Series,
        next_month: dict[pd.Timestamp, pd.Timestamp | None],
        open_prices: pd.Series,
        daily: pd.DataFrame,
        fundamentals: pd.DataFrame,
        factor_ids: list[str],
        locked: dict[str, str],
        index_code: str,
        universe_config: dict[str, Any] | None,
        formal: bool,
        progress: Any = None,
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """Compute raw factor values and forward returns for one signal slice.

        ``factor_ids`` is a subset of ``locked`` (empty for a forward-only
        pass). Returns long frames per factor (signal_date, symbol,
        factor_value) plus the forward-return frame.
        """
        root = self._data_root()
        members = self._load_members(root, index_code)
        industries = self._load_industries(root, universe_config or {})
        registry = self.store.load_registry()
        technical_names = [name for name in factor_ids if name in TECHNICAL_FACTOR_NAMES]
        blocks: dict[str, list[pd.DataFrame]] = {
            factor_id: [] for factor_id in factor_ids
        }
        forward_rows: list[pd.DataFrame] = []
        for position, signal_date in enumerate(signals, start=1):
            following_signal = next_month.get(signal_date)
            if following_signal is None:
                continue
            execution = opened[opened > signal_date]
            if execution.empty:
                continue
            exit_dates = opened[opened > following_signal]
            universe = self._universe(
                daily, members, index_code, signal_date, industries, universe_config
            )
            if not universe:
                continue
            forward: pd.Series = pd.Series(dtype=float)
            if not exit_dates.empty:
                entry_date, exit_date = execution.iloc[0], exit_dates.iloc[0]
                entry = open_prices.xs(entry_date, level="trade_date").reindex(universe)
                exit_ = open_prices.xs(exit_date, level="trade_date").reindex(universe)
                forward = exit_.div(entry).sub(1).rename("forward_return")
                forward_rows.append(
                    forward.reset_index().assign(signal_date=signal_date)
                )
            context = FactorContext(
                signal_date,
                daily,
                fundamentals,
                universe=tuple(universe),
                formal=formal,
            )
            # Compute every technical factor in one pass per signal instead of
            # re-indexing the full daily window once per factor.
            technical_series: dict[str, pd.Series] = {}
            if technical_names:
                visible = daily[daily["trade_date"] <= signal_date]
                symbols = visible["symbol"].drop_duplicates().sort_values()
                signal_index = pd.MultiIndex.from_arrays(
                    [symbols, np.repeat(signal_date, len(symbols))],
                    names=["symbol", "signal_date"],
                )
                computed = compute_technical_features(visible, signal_index, technical_names)
                technical_series = {
                    name: pd.Series(computed[name].to_numpy(), index=symbols)
                    for name in technical_names
                }
            for factor_id in factor_ids:
                if factor_id in technical_series:
                    values = pd.to_numeric(
                        technical_series[factor_id], errors="coerce"
                    ).rename("factor_value")
                else:
                    values = pd.to_numeric(
                        registry.compute_revision(locked[factor_id], context),
                        errors="coerce",
                    ).rename("factor_value")
                if forward.empty:
                    # Latest month: factor values are usable (signal forms at
                    # month end) even though the next-month exit price is not
                    # available yet; the forward label stays NaN.
                    frame = values.reset_index()
                    frame = frame.rename(columns={frame.columns[0]: "symbol"})
                    frame["forward_return"] = np.nan
                else:
                    frame = pd.concat([values, forward], axis=1, join="inner").reset_index()
                    frame = frame.rename(columns={frame.columns[0]: "symbol"})
                frame["signal_date"] = signal_date
                blocks[factor_id].append(frame[["signal_date", "symbol", "factor_value"]])
            if progress is not None:
                progress(position, len(signals))
        block_frames = {
            factor_id: (
                pd.concat(parts, ignore_index=True) if parts
                else pd.DataFrame(columns=["signal_date", "symbol", "factor_value"])
            )
            for factor_id, parts in blocks.items()
        }
        forward_frame = (
            pd.concat(forward_rows, ignore_index=True) if forward_rows
            else pd.DataFrame(columns=["signal_date", "symbol", "forward_return"])
        )
        return block_frames, forward_frame

    @staticmethod
    def _assemble_panel(
        blocks: dict[str, pd.DataFrame],
        forward_frame: pd.DataFrame | None,
        index_code: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        columns = [
            "signal_date", "symbol", "factor_name", "raw_value",
            "neutralized_value", "forward_return", "index_code",
        ]
        if forward_frame is None or forward_frame.empty or not blocks:
            return pd.DataFrame(columns=columns)
        forward_frame["signal_date"] = pd.to_datetime(forward_frame["signal_date"])
        frames = []
        for factor_id, block in blocks.items():
            part = block.copy()
            part["signal_date"] = pd.to_datetime(part["signal_date"])
            part = part[
                (part["signal_date"] >= start) & (part["signal_date"] <= end)
            ]
            part = part.merge(
                forward_frame, on=["signal_date", "symbol"], how="left"
            )
            part["factor_name"] = factor_id
            frames.append(part.rename(columns={"factor_value": "raw_value"}))
        if not frames:
            return pd.DataFrame(columns=columns)
        panel = pd.concat(frames, ignore_index=True)
        panel["neutralized_value"] = np.nan
        panel["index_code"] = index_code
        return panel[columns]

    def _build_auto_panel(self, run: dict[str, Any]) -> pd.DataFrame:
        config = run["config"]
        start = pd.Timestamp(config["start_date"]).normalize()
        end = pd.Timestamp(config["end_date"]).normalize()
        index_code = str(config["index_code"])
        universe_config = config.get("universe") or {}
        formal = run["mode"] == "formal"
        locked = {item["factor_id"]: item["revision_id"] for item in run["factors"]}
        daily, fundamentals, month_ends, opened, next_month, open_prices = (
            self.prepare_auto_compute(locked, index_code, start, end)
        )
        signals = [date for date in month_ends if start <= date <= end]
        if not signals:
            return pd.DataFrame(columns=[
                "signal_date", "symbol", "factor_name", "raw_value",
                "neutralized_value", "forward_return", "index_code",
            ])
        cache = FactorValueCache(self._data_root())
        key = universe_key(index_code, universe_config)
        blocks: dict[str, pd.DataFrame] = {}
        forward: pd.DataFrame | None = None
        if cache.signature_matches():
            blocks = cache.read_values(key, dict(locked), start, end)
            forward = cache.read_forward(key, start, end)
        missing = [factor_id for factor_id in locked if factor_id not in blocks]
        cached_max = (
            pd.to_datetime(forward["signal_date"]).max()
            if forward is not None and not forward.empty else None
        )
        factor_maxes = [
            pd.to_datetime(block["signal_date"]).max()
            for block in blocks.values() if not block.empty
        ]
        block_max = min(factor_maxes) if factor_maxes else None
        # A month is missing from the panel when any factor lacks it or the
        # forward frame lacks it; the cache manifest may claim month coverage
        # its files do not actually hold, so trust the data, not the manifest.
        coverage_max = (
            cached_max if block_max is None
            else block_max if cached_max is None
            else min(block_max, cached_max)
        )
        tail = [date for date in signals if coverage_max is None or date > coverage_max]
        if missing or forward is None or tail:
            new_blocks: dict[str, pd.DataFrame] = {}
            full_forward: pd.DataFrame | None = None
            stored_min = stored_max = None
            if missing:
                new_blocks, full_forward = self._compute_block(
                    signals, opened, next_month, open_prices, daily, fundamentals,
                    missing, locked, index_code, universe_config, formal,
                    progress=lambda position, total: self.store.set_run_status(
                        run["run_id"], "running",
                        progress=0.05 + 0.35 * position / total,
                    ),
                )
                stored_min, stored_max = min(signals), max(signals)
            if tail:
                tail_factors = [factor_id for factor_id in locked if factor_id not in missing]
                tail_blocks, tail_forward = self._compute_block(
                    tail, opened, next_month, open_prices, daily, fundamentals,
                    tail_factors, locked, index_code, universe_config, formal,
                    progress=lambda position, total: self.store.set_run_status(
                        run["run_id"], "running",
                        progress=0.05 + 0.35 * position / total,
                    ),
                )
                new_blocks.update(tail_blocks)
                if full_forward is None:
                    full_forward = tail_forward
                if stored_min is None or min(tail) < stored_min:
                    stored_min = min(tail)
                if stored_max is None or max(tail) > stored_max:
                    stored_max = max(tail)
            if forward is None:
                forward = full_forward
            elif full_forward is not None and not full_forward.empty:
                # A missing-factor pass recomputes forward over the full signal
                # range and replaces the cached frame; a tail-only pass appends.
                forward = (
                    full_forward if missing
                    else pd.concat([forward, full_forward], ignore_index=True)
                )
                forward = forward.drop_duplicates(
                    ["signal_date", "symbol"], keep="last"
                ).reset_index(drop=True)
            for factor_id, block in new_blocks.items():
                previous = blocks.get(factor_id)
                merged = (
                    pd.concat([previous, block], ignore_index=True)
                    if previous is not None else block
                )
                blocks[factor_id] = merged.drop_duplicates(
                    ["signal_date", "symbol"], keep="last"
                ).reset_index(drop=True)
            cache.store(
                key, new_blocks, {factor_id: locked[factor_id] for factor_id in new_blocks},
                full_forward,
                index_code=index_code, universe_config=universe_config,
                mode=run["mode"],
                signal_min=stored_min, signal_max=stored_max,
            )
        self.store.set_run_status(run["run_id"], "running", progress=0.4)
        return self._assemble_panel(blocks, forward, index_code, start, end)

    def _load_daily(
        self,
        daily_path: Path,
        adjustments_path: Path,
        fields: set[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        available = self._parquet_columns(daily_path)
        # Always load the full base OHLCV set: technical computations read volume
        # even when the formula only references price fields, and QMT daily files
        # only carry these eight columns.
        requested = {
            "trade_date", "symbol", "open", "high", "low", "close", "volume", "amount",
        }
        for field in fields:
            if not field.startswith("market."):
                continue
            column = self.store.fields.get(field).column
            requested.add(column.removeprefix("adj_"))
        columns = sorted(requested & available)
        try:
            daily = storage_io.read_frame(
                daily_path,
                columns=columns,
                filters=[("trade_date", ">=", start), ("trade_date", "<=", end)],
            )
        except (TypeError, ValueError):
            daily = storage_io.read_frame(daily_path, columns=columns)
            daily["trade_date"] = pd.to_datetime(daily["trade_date"]).dt.normalize()
            daily = daily[daily["trade_date"].between(start, end)]
        daily["trade_date"] = pd.to_datetime(daily["trade_date"]).dt.normalize()
        adjustments = storage_io.read_frame(
            adjustments_path, columns=["trade_date", "symbol", "adjust_factor"]
        )
        adjustments["trade_date"] = pd.to_datetime(adjustments["trade_date"]).dt.normalize()
        adjustments = adjustments[adjustments["trade_date"] <= end]
        daily = pd.merge_asof(
            daily.sort_values(["trade_date", "symbol"]),
            adjustments.sort_values(["trade_date", "symbol"]),
            on="trade_date",
            by="symbol",
            direction="backward",
        )
        daily["adjust_factor"] = daily["adjust_factor"].fillna(1.0)
        for column in ("open", "high", "low", "close"):
            if column in daily:
                daily[f"adj_{column}"] = daily[column] * daily["adjust_factor"]
        if "adj_close" in daily:
            daily["market_return"] = daily.groupby("symbol", observed=True)[
                "adj_close"
            ].pct_change(fill_method=None)
        return daily.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    @staticmethod
    def _load_members(root: Path, index_code: str) -> pd.DataFrame:
        if index_code == "ALL_A":
            return pd.DataFrame()
        frame = storage_io.read_frame(root / "equity" / "index_members.parquet")
        for column in ("valid_from", "valid_to"):
            frame[column] = pd.to_datetime(frame[column]).dt.normalize()
        return frame[frame["index_code"] == index_code]

    @staticmethod
    def _load_industries(root: Path, universe_config: dict[str, Any]) -> pd.DataFrame | None:
        filters = (universe_config or {}).get("industries") or {}
        if not (filters.get("include") or filters.get("exclude")):
            return None
        path = root / "equity" / "industries.parquet"
        if not storage_io.exists(path):
            raise AutoRunDataError("行业筛选需要 equity/industries.parquet（申万行业点位历史）")
        frame = storage_io.read_frame(
            path, columns=["symbol", "industry_code", "valid_from", "valid_to"]
        )
        for column in ("valid_from", "valid_to"):
            frame[column] = pd.to_datetime(frame[column]).dt.normalize()
        return frame

    @staticmethod
    def _universe(
        daily: pd.DataFrame,
        members: pd.DataFrame,
        index_code: str,
        signal_date: pd.Timestamp,
        industries: pd.DataFrame | None = None,
        universe_config: dict[str, Any] | None = None,
    ) -> list[str]:
        traded = set(daily.loc[daily["trade_date"] == signal_date, "symbol"])
        if index_code == "ALL_A":
            universe = traded
        else:
            active = members[
                (members["valid_from"] <= signal_date)
                & (members["valid_to"].isna() | (members["valid_to"] >= signal_date))
            ]
            universe = traded & set(active["symbol"])
        config = universe_config or {}
        if industries is not None:
            valid = industries[
                (industries["valid_from"] <= signal_date)
                & (industries["valid_to"].isna() | (industries["valid_to"] >= signal_date))
            ]
            include = set(config.get("industries", {}).get("include", ()))
            exclude = set(config.get("industries", {}).get("exclude", ()))
            if include:
                universe &= set(valid.loc[valid["industry_code"].isin(include), "symbol"])
            if exclude:
                universe -= set(valid.loc[valid["industry_code"].isin(exclude), "symbol"])
        symbols = config.get("symbols") or {}
        if symbols.get("include"):
            universe &= set(symbols["include"])
        if symbols.get("exclude"):
            universe -= set(symbols["exclude"])
        return sorted(universe)

    def _locked_specs(self, factor_ids: list[str]) -> dict[str, FactorSpec]:
        registry = self.store.load_registry()
        revision_ids: list[str] = []
        for factor_id in factor_ids:
            try:
                revision_ids.append(registry.get(factor_id).revision_id)
            except KeyError as error:
                raise ValueError(f"未知因子：{factor_id}") from error
        return self._expand_revision_specs(registry, revision_ids)

    def _expand_revision_specs(
        self, registry: FactorRegistry, revision_ids: list[str]
    ) -> dict[str, FactorSpec]:
        result: dict[str, FactorSpec] = {}
        pending = list(revision_ids)
        while pending:
            revision_id = pending.pop()
            if revision_id in result:
                continue
            spec = registry.get_revision(revision_id)
            result[revision_id] = spec
            pending.extend(
                str(row["dependency_revision_id"])
                for row in self.store.connection.execute(
                    "SELECT dependency_revision_id FROM factor_dependency WHERE revision_id=?",
                    (revision_id,),
                ).fetchall()
            )
        return result

    def _required_models(self, revisions: dict[str, FactorSpec]) -> list[str]:
        result: set[str] = set()
        for revision_id in revisions:
            row = self.store.connection.execute(
                "SELECT models_json FROM factor_revision WHERE revision_id=?",
                (revision_id,),
            ).fetchone()
            if row:
                result.update(item[0] for item in json.loads(row["models_json"]))
        return sorted(result)

    def _data_root(self) -> Path:
        if self.store.path is None:
            raise ValueError("自动回测需要文件型因子库")
        return self.store.path.parent.parent

    @staticmethod
    def _parquet_columns(path: Path) -> set[str]:

        return set(storage_io.TableReader(path).schema_arrow.names)

    @staticmethod
    def _run_periods(
        run: dict[str, Any],
    ) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]] | None:
        raw = (run.get("config") or {}).get("splits")
        if not raw:
            return None
        periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        for name, pair in raw.items():
            start = pd.Timestamp(pair[0]).normalize()
            end = pd.Timestamp(pair[1]).normalize()
            if start >= end:
                raise ValueError(f"splits.{name} 开始日期必须早于结束日期")
            periods[name] = (start, end)
        return periods or None

    def execute_panel_run(self, run_id: str, panel_path: str | Path) -> Path:
        run = self.store.run_detail(run_id)
        panel_path = Path(panel_path).expanduser().resolve()
        self.store.set_run_status(run_id, "running", progress=0.05)
        try:
            panel = storage_io.read_frame(panel_path)
            self._validate_panel(panel, run)
            selected = {item["factor_id"] for item in run["factors"]}
            panel = panel[panel["factor_name"].isin(selected)].copy()
            if panel.empty:
                raise ValueError("panel contains none of the locked run factors")
            indices = sorted(panel["index_code"].dropna().unique()) \
                if "index_code" in panel else [""]
            monthly_frames: list[pd.DataFrame] = []
            summary_frames: list[pd.DataFrame] = []
            metric_rows: list[dict[str, Any]] = []
            for position, index_code in enumerate(indices, start=1):
                part = panel[panel["index_code"] == index_code] if index_code else panel
                monthly, summary = evaluate_factor_batch(part, periods=self._run_periods(run))
                monthly["index_code"] = index_code
                summary["index_code"] = index_code
                monthly_frames.append(monthly)
                summary_frames.append(summary)
                metric_rows.extend(self._summary_metrics(summary))
                metric_rows.extend(self._portfolio_metrics(monthly, index_code))
                self.store.set_run_status(
                    run_id, "running", progress=0.1 + 0.75 * position / len(indices)
                )
            monthly_all = pd.concat(monthly_frames, ignore_index=True)
            summary_all = pd.concat(summary_frames, ignore_index=True)
            output = self._output_dir(run_id)
            output.mkdir(parents=True, exist_ok=False)
            monthly_path = output / "monthly.parquet"
            summary_path = output / "summary.csv"
            manifest_path = output / "manifest.json"
            storage_io.write_frame(monthly_all, monthly_path, index=False)
            storage_io.write_csv(summary_all, summary_path, index=False)
            manifest = {
                "run_id": run_id,
                "mode": run["mode"],
                "watermark": "NON-FORMAL SMOKE" if run["mode"] == "smoke" else None,
                "panel": str(panel_path),
                "factors": run["factors"],
                "config": run["config"],
                "monitoring_2026_excluded_from_selection": True,
            }
            storage_io.write_text(manifest_path,
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            layered_nav_path, layered_performance_path = (
                self._write_layered_artifacts(monthly_all, output)
            )
            self.store.write_metrics(run_id, metric_rows)
            self.store.add_artifact(run_id, monthly_path, "monthly_statistics")
            self.store.add_artifact(run_id, summary_path, "summary")
            self.store.add_artifact(run_id, manifest_path, "manifest")
            self.store.add_artifact(run_id, layered_nav_path, "layered_nav")
            self.store.add_artifact(
                run_id, layered_performance_path, "layered_performance"
            )
            self.store.set_run_status(run_id, "succeeded", progress=1.0)
            return output
        except Exception as error:
            self.store.set_run_status(run_id, "failed", progress=1.0, error=str(error))
            raise

    @staticmethod
    def _validate_panel(panel: pd.DataFrame, run: dict[str, Any]) -> None:
        required = {
            "signal_date", "symbol", "factor_name", "raw_value",
            "neutralized_value", "forward_return",
        }
        missing = required - set(panel)
        if missing:
            raise ValueError(f"factor panel missing columns: {sorted(missing)}")
        panel["signal_date"] = pd.to_datetime(panel["signal_date"])
        if run["mode"] == "formal":
            if not run["config"].get("point_in_time_audit_passed", False):
                raise ValueError("formal run requires point_in_time_audit_passed=true")
            if "index_code" not in panel:
                raise ValueError("formal run requires an explicit universe")
            configured = run["config"].get("index_code")
            actual = set(panel["index_code"].dropna().unique())
            expected = {configured} if configured else FORMAL_INDICES
            if actual != expected:
                raise ValueError(
                    f"formal run universe mismatch: expected {sorted(expected)}, "
                    f"got {sorted(actual)}"
                )

    @staticmethod
    def _summary_metrics(summary: pd.DataFrame) -> list[dict[str, Any]]:
        dimensions = {
            "factor_name", "index_code", "period", "value_type", "orientation",
        }
        rows: list[dict[str, Any]] = []
        for record in summary.to_dict("records"):
            for name, value in record.items():
                if name in dimensions or not isinstance(value, (int, float, np.number)):
                    continue
                rows.append({
                    "factor_id": record["factor_name"],
                    "index_code": record.get("index_code", ""),
                    "period": record.get("period", "all"),
                    "value_type": record.get("value_type", "neutralized"),
                    "orientation": record.get("orientation", "original"),
                    "metric_name": name,
                    "metric_value": float(value) if pd.notna(value) else None,
                })
        return rows

    @staticmethod
    def _portfolio_metrics(monthly: pd.DataFrame, index_code: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        monthly = _descending_groups(monthly)
        groups = monthly.groupby(
            ["factor_name", "value_type", "orientation"],
            observed=True, dropna=False,
        )
        for (factor_id, value_type, orientation), factor_sample in groups:
            for period, sample in _period_samples(factor_sample):
                for scenario, bps in (("base_5bps", 5), ("stress_10bps", 10)):
                    turnover = sample.get(
                        "rank_turnover", pd.Series(0.0, index=sample.index)
                    ).fillna(0)
                    net = sample["group_1"] - turnover * bps / 10_000
                    metrics = _performance(net)
                    metrics["mean_turnover"] = float(turnover.mean())
                    for name, value in metrics.items():
                        rows.append({
                            "factor_id": factor_id,
                            "index_code": index_code,
                            "period": period,
                            "value_type": value_type,
                            "orientation": orientation,
                            "cost_scenario": scenario,
                            "metric_name": name,
                            "metric_value": value,
                        })
        return rows

    @staticmethod
    def _layered_frames(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        monthly = _descending_groups(monthly)
        nav_rows: list[dict[str, Any]] = []
        performance_rows: list[dict[str, Any]] = []
        dimensions = ["factor_name", "index_code", "value_type", "orientation"]
        for keys, factor_sample in monthly.groupby(
            dimensions, observed=True, dropna=False
        ):
            factor_id, index_code, value_type, orientation = keys
            full_returns = _layer_returns(factor_sample.sort_values("signal_date"))
            anchor_date = pd.Timestamp(full_returns["signal_date"].min()) - pd.offsets.MonthEnd(1)
            for portfolio in PORTFOLIO_ORDER:
                returns = pd.to_numeric(full_returns[portfolio], errors="coerce").dropna()
                net_value = (1 + returns).cumprod()
                nav_rows.append({
                    "signal_date": anchor_date,
                    "factor_name": factor_id,
                    "index_code": index_code,
                    "value_type": value_type,
                    "orientation": orientation,
                    "portfolio": portfolio,
                    "monthly_return": np.nan,
                    "net_value": 1.0,
                })
                for row_index, value in net_value.items():
                    nav_rows.append({
                        "signal_date": full_returns.loc[row_index, "signal_date"],
                        "factor_name": factor_id,
                        "index_code": index_code,
                        "value_type": value_type,
                        "orientation": orientation,
                        "portfolio": portfolio,
                        "monthly_return": returns.loc[row_index],
                        "net_value": value,
                    })

            for period, sample in _period_samples(factor_sample):
                layered = _layer_returns(sample)
                benchmark = layered["benchmark"]
                for portfolio in PORTFOLIO_ORDER:
                    compare = portfolio not in {"benchmark", "long_short"}
                    metrics = _layer_performance(
                        layered[portfolio], benchmark, compare_with_benchmark=compare
                    )
                    performance_rows.append({
                        "factor_name": factor_id,
                        "index_code": index_code,
                        "value_type": value_type,
                        "orientation": orientation,
                        "period": period,
                        "portfolio": portfolio,
                        **metrics,
                    })
        return pd.DataFrame(nav_rows), pd.DataFrame(performance_rows)

    def _write_layered_artifacts(
        self, monthly: pd.DataFrame, output: Path
    ) -> tuple[Path, Path]:
        """Write layered NAV/performance data; charts are rendered by the web
        frontend (ECharts) directly from ``layered_nav.parquet``."""
        nav, performance = self._layered_frames(monthly)
        nav_path = output / "layered_nav.parquet"
        performance_path = output / "layered_performance.parquet"
        storage_io.write_frame(nav, nav_path, index=False)
        storage_io.write_frame(performance, performance_path, index=False)
        return nav_path, performance_path

    def rebuild_layered_backtest(self, run_id: str) -> tuple[Path, Path]:
        """Backfill full-sample layered evidence for an existing successful run."""
        run = self.store.run_detail(run_id)
        monthly_artifact = next(
            (
                item for item in run["artifacts"]
                if item["kind"] == "monthly_statistics"
            ),
            None,
        )
        if monthly_artifact is None:
            raise ValueError("run has no monthly statistics artifact")
        monthly = storage_io.read_frame(monthly_artifact["path"])
        for index_code, sample in monthly.groupby("index_code", observed=True, dropna=False):
            self.store.write_metrics(
                run_id, self._portfolio_metrics(sample, str(index_code))
            )
        output = Path(monthly_artifact["path"]).resolve().parent
        nav_path, performance_path = self._write_layered_artifacts(monthly, output)
        self.store.add_artifact(run_id, nav_path, "layered_nav")
        self.store.add_artifact(run_id, performance_path, "layered_performance")
        return nav_path, performance_path

    def _output_dir(self, run_id: str) -> Path:
        if self.store.path is None:
            raise ValueError("research artifacts require a file-backed factor store")
        return self.store.path.parent / "runs" / run_id
