"""Report engine: turns a validated report spec into a static research report.

Execution is offline: the engine runs inside a subprocess (queued by the web
app or the CLI), reads immutable run artifacts, and writes a self-contained
directory of static files that any agent or human can read without recomputing.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from mlquant import __version__, storage_io
from mlquant.combine import combine_scores, factor_weights, select_low_correlation_factors
from mlquant.factor_cache import data_signature
from mlquant.factor_research_service import FactorResearchService, _performance
from mlquant.factor_store import FactorStore
from mlquant.ml_composite import (
    frozen_composite,
    load_pit_context,
    neutralize_wide,
)
from mlquant.report_spec import ReportSpec, parse_spec, resolve_factors
from mlquant.research import evaluate_factor_batch, newey_west_t, zscore

PANEL_COLUMNS = ["signal_date", "symbol", "factor_name", "raw_value", "forward_return"]
COMBINE_LABELS = {
    "equal": "等权",
    "ic_decay": "IC加权（半衰期6月）",
    "factor_return_decay": "回归法加权（半衰期6月）",
    "max_icir": "最大化IC_IR",
    "max_ic": "最大化IC",
    "pca": "主成分合成（第一主成分）",
    "lasso": "LASSO权重",
    "ridge": "岭回归权重",
    "rf": "随机森林权重",
    "xgb": "XGBoost权重",
    "mlp": "神经网络权重",
    "ml_ols": "OLS线性（ML合成）",
    "ml_lasso": "LASSO（ML合成）",
    "ml_ridge": "岭回归（ML合成）",
    "ml_dtree": "决策树（ML合成）",
    "ml_rf": "随机森林（ML合成）",
    "ml_et": "极端随机树（ML合成）",
    "ml_xgb": "XGBoost（ML合成）",
    "ml_lgbm": "LightGBM（ML合成）",
    "ml_mlp": "神经网络（ML合成）",
}
_COMBINE_COLORS = {
    "equal": "#27835b", "ic_decay": "#386cb0", "factor_return_decay": "#7b4173",
    "max_icir": "#b42318", "max_ic": "#d9b300", "pca": "#26322d",
    "lasso": "#7f3c8d", "ridge": "#1a5b7a", "rf": "#3f51b5",
    "xgb": "#d1495b", "mlp": "#5d6d7e",
    "ml_ols": "#f28e2b", "ml_lasso": "#4e79a7", "ml_ridge": "#59a14f",
    "ml_dtree": "#9c755f", "ml_rf": "#b07aa1", "ml_et": "#76b7b2",
    "ml_xgb": "#e15759", "ml_lgbm": "#edc948", "ml_mlp": "#ff9da7",
}
_SPLIT_LABELS = {
    "development": "开发",
    "validation": "验证",
    "test": "测试",
    "monitoring_2026": "监控",
    "all": "全样本",
}
_PERIOD_ORDER = {
    "development": 0, "validation": 1, "test": 2, "monitoring_2026": 3, "all": 4,
}
METRIC_COLUMNS = (
    "rank_ic", "icir", "hac_t", "monotonicity", "coverage",
    "annualized_return", "sharpe", "max_drawdown", "monthly_win_rate", "mean_turnover",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(storage_io.read_bytes(path)).hexdigest()


def _static_text(name: str) -> str:
    return storage_io.read_text(Path(__file__).parent / "static" / name, encoding="utf-8")


def build_cross_section(
    panel_path: str | Path, factor_ids: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read one panel once and build wide/zscore/forward/ic/fm views.

    The old dashboard read the panel parquet once per factor; this reads it
    exactly once per report and splits by factor_name instead.
    """
    long = storage_io.read_frame(panel_path, columns=PANEL_COLUMNS)
    long = long[long["factor_name"].isin(factor_ids)]
    long = long.drop_duplicates(["signal_date", "symbol", "factor_name"])
    long["signal_date"] = pd.to_datetime(long["signal_date"]).dt.normalize()
    wide = long.pivot(
        index=["signal_date", "symbol"], columns="factor_name", values="raw_value"
    )
    forward = long.groupby(["signal_date", "symbol"])["forward_return"].first()
    z = wide.groupby(level="signal_date", observed=True).transform(zscore)
    ic_rows: list[dict[str, Any]] = []
    fm_rows: list[dict[str, Any]] = []
    for date, block in z.groupby(level="signal_date", observed=True):
        target = forward.loc[date].reindex(block.index.get_level_values("symbol"))
        block_ranks = block.rank()
        target_rank = target.rank()
        ic_row: dict[str, Any] = {"signal_date": date}
        ic_row.update(block_ranks.corrwith(target_rank).to_dict())
        centered = block_ranks.sub(block_ranks.mean(), axis=1)
        centered_target = target_rank - target_rank.mean()
        numerator = centered.mul(centered_target, axis=0).sum()
        denominator = (centered**2).sum()
        fm_row: dict[str, Any] = {"signal_date": date}
        fm_row.update((numerator / denominator.replace(0, np.nan)).to_dict())
        ic_rows.append(ic_row)
        fm_rows.append(fm_row)
    ic = pd.DataFrame(ic_rows).set_index("signal_date")
    fm = pd.DataFrame(fm_rows).set_index("signal_date")
    return wide, forward, z, ic, fm


class _ReportProgressMirror:
    """Mirror the linked run's progress into the report row while it runs.

    ``execute_auto_run`` blocks the engine thread, so without this the web UI
    would show the report stuck at 0.12 for the whole backtest.
    """

    _RUN_SLICE = (0.12, 0.78)  # report progress range reserved for the backtest

    def __init__(self, root: Path, report_id: str, run_id: str) -> None:
        self._root = root
        self._report_id = report_id
        self._run_id = run_id
        self._stop = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._sync_once()
        self._stop.set()
        self._thread.join(timeout=5)

    def _sync_once(self) -> str | None:
        low, high = self._RUN_SLICE
        with FactorStore.from_root(self._root) as store:
            row = store.connection.execute(
                "SELECT status, progress FROM research_run WHERE run_id=?",
                (self._run_id,),
            ).fetchone()
            if row is None:
                return None
            run_progress = max(0.0, min(1.0, float(row["progress"] or 0.0)))
            progress = low + (high - low) * run_progress
            # Direct UPDATE to avoid spamming audit_event every poll cycle.
            with store.connection:
                store.connection.execute(
                    "UPDATE report SET progress=? WHERE report_id=?",
                    (progress, self._report_id),
                )
            return str(row["status"])

    def _loop(self) -> None:
        while not self._stop.wait(1.5):
            if self._sync_once() in {"succeeded", "failed", "cancelled"}:
                return


class ReportEngine:
    def __init__(self, store: FactorStore) -> None:
        self.store = store
        self.service = FactorResearchService(store)

    def data_root(self) -> Path:
        if self.store.path is None:
            raise ValueError("报告引擎需要文件型因子库")
        return self.store.path.parent.parent

    def reports_root(self) -> Path:
        return self.store.path.parent / "reports"

    @storage_io.freeze_market
    def execute_report(self, report_id: str) -> Path:
        self.store.claim_report(report_id)
        row = self.store.report_detail(report_id)
        try:
            spec = parse_spec(dict(row["spec"]))
        except Exception as error:
            # E.g. legacy specs selecting on "all"/"test" now violate the
            # isolation rule; surface that instead of leaving the report queued.
            self.store.set_report_status(report_id, "failed", progress=1.0, error=str(error))
            raise
        self.store.set_report_status(report_id, "running", progress=0.03)
        try:
            factors = resolve_factors(self.store, spec.factors, index_code=spec.universe.index_code)
            minimum = 1 if spec.combine is None else 2
            if len(factors) < minimum:
                if spec.combine is None:
                    raise ValueError("报告至少需要 1 个因子")
                raise ValueError("报告至少需要 2 个因子（相关矩阵与合成对比依赖截面数据）")
            self._validate_universe(spec)
            run_id, reused = self._ensure_runs_and_create_run(spec, factors)
            if reused:
                self.store.set_report_status(
                    report_id, "running", run_id=run_id, progress=0.7
                )
            else:
                self.store.set_report_status(
                    report_id, "running", run_id=run_id, progress=0.12
                )
                mirror = _ReportProgressMirror(self.data_root(), report_id, run_id)
                mirror.start()
                try:
                    self.service.execute_auto_run(run_id)
                finally:
                    mirror.stop()
                self.store.set_report_status(report_id, "running", progress=0.8)

            output = self._materialize(report_id, spec, factors, run_id)
            panel_path = self._panel_path(run_id)
            factor_ids = [item["factor_id"] for item in factors]
            _wide, forward, z, ic, fm = build_cross_section(panel_path, factor_ids)
            self.store.set_report_status(report_id, "running", progress=0.85)
            matrix = self._write_correlation(output, z)
            combo = None
            if spec.combine is not None and matrix is not None:
                combo = self._write_combination(output, spec, factors, forward, z, ic, fm, wide=_wide)
                self.store.set_report_status(report_id, "running", progress=0.95)
            self._write_report_text(output, spec, factors, run_id, matrix, combo)
            self._write_manifest(output, spec, factors, run_id)
            self.store.set_report_status(report_id, "succeeded", progress=1.0, path=output)
            return output
        except Exception as error:
            self.store.set_report_status(report_id, "failed", progress=1.0, error=str(error))
            raise

    # ---------------------------------------------------------------- helpers

    def _validate_universe(self, spec: ReportSpec) -> None:
        filters = spec.universe.industries
        codes = set(filters.get("include", ())) | set(filters.get("exclude", ()))
        if not codes:
            return
        path = self.data_root() / "equity" / "industries.parquet"
        if not storage_io.exists(path):
            raise ValueError("行业筛选需要 equity/industries.parquet（申万行业点位历史）")
        frame = storage_io.read_frame(path, columns=["industry_code"])
        known = set(frame["industry_code"].dropna().unique())
        unknown = sorted(codes - known)
        if unknown:
            raise ValueError(
                f"未知申万行业代码：{unknown}；可用代码见 equity/industries.parquet"
            )

    def missing_run_factors(
        self, spec: ReportSpec, factors: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Factors whose latest succeeded run does not cover this report window."""
        window_start, window_end = spec.window_bounds()
        monitoring = spec.monitoring_bound()
        run_end = monitoring or window_end
        latest = self.store.latest_succeeded_runs()
        missing: list[dict[str, Any]] = []
        for item in factors:
            run = latest.get(item["factor_id"])
            config = json.loads(run["config_json"]) if run is not None else None
            usable = (
                run is not None
                and run["mode"] == spec.mode
                and isinstance(config, dict)
                and config.get("index_code") == spec.universe.index_code
                and str(config.get("start_date", "")) <= str(window_start.date())
                and str(config.get("end_date", "")) >= str(run_end.date())
            )
            if not usable:
                missing.append(item)
        return missing

    def backfill_runs(
        self, spec: ReportSpec, factors: list[dict[str, Any]],
    ) -> None:
        """Run the automatic backtest for every factor missing report coverage.

        Missing factors share one batched run (same window, index and universe),
        so a 20-factor report backfills once instead of once per factor.
        """
        window_start, window_end = spec.window_bounds()
        run_end = spec.monitoring_bound() or window_end
        missing = self.missing_run_factors(spec, factors)
        if not missing:
            return
        config = {
            "source": "automatic",
            "index_code": spec.universe.index_code,
            "start_date": str(window_start.date()),
            "end_date": str(run_end.date()),
            "point_in_time_audit_passed": spec.mode == "formal",
            "splits": {
                name: [str(start.date()), str(end.date())]
                for name, (start, end) in spec.resolved_splits().items()
            },
            "universe": {
                "industries": {
                    "include": spec.universe.industries.get("include", []),
                    "exclude": spec.universe.industries.get("exclude", []),
                },
                "symbols": {
                    "include": spec.universe.symbols.get("include", []),
                    "exclude": spec.universe.symbols.get("exclude", []),
                },
            },
        }
        if spec.monitoring_bound() is not None:
            config["monitoring_2026_included"] = True
        run_id = self.store.create_run(
            [item["factor_id"] for item in missing], mode=spec.mode, config=config
        )
        self.service.execute_auto_run(run_id)

    def _ensure_runs_and_create_run(
        self, spec: ReportSpec, factors: list[dict[str, Any]],
    ) -> tuple[str, bool]:
        missing = self.missing_run_factors(spec, factors)
        if missing:
            raise ValueError(
                "以下因子没有覆盖报告窗口的成功回测，请先补跑自动回测（"
                "scripts/backfill_factor_backtests.py 或 mlquant report create --ensure-runs）："
                + "、".join(item["factor_id"] for item in missing)
            )
        reused = self._find_reusable_run(spec, factors)
        if reused is not None:
            return reused, True
        window_start, window_end = spec.window_bounds()
        run_end = spec.monitoring_bound() or window_end
        config: dict[str, Any] = {
            "source": "automatic",
            "index_code": spec.universe.index_code,
            "start_date": str(window_start.date()),
            "end_date": str(run_end.date()),
            "point_in_time_audit_passed": spec.mode == "formal",
            "splits": {
                name: [str(start.date()), str(end.date())]
                for name, (start, end) in spec.resolved_splits().items()
            },
            "universe": {
                "industries": {
                    "include": spec.universe.industries.get("include", []),
                    "exclude": spec.universe.industries.get("exclude", []),
                },
                "symbols": {
                    "include": spec.universe.symbols.get("include", []),
                    "exclude": spec.universe.symbols.get("exclude", []),
                },
            },
        }
        if spec.monitoring_bound() is not None:
            config["monitoring_2026_included"] = True
        run_id = self.store.create_run(
            [item["factor_id"] for item in factors], mode=spec.mode, config=config
        )
        return run_id, False

    def _find_reusable_run(
        self, spec: ReportSpec, factors: list[dict[str, Any]],
    ) -> str | None:
        """An existing succeeded run whose panel already matches this report.

        Every report factor must point at the same succeeded run covering the
        report window (including monitoring), with an identical universe config;
        the run's factor set must match the report's exactly. When found, the
        report materializes from that run's panel instead of recomputing it.
        """
        factor_ids = [item["factor_id"] for item in factors]
        window_start, _window_end = spec.window_bounds()
        run_end = spec.monitoring_bound() or _window_end
        universe = {
            "industries": {
                "include": spec.universe.industries.get("include", []),
                "exclude": spec.universe.industries.get("exclude", []),
            },
            "symbols": {
                "include": spec.universe.symbols.get("include", []),
                "exclude": spec.universe.symbols.get("exclude", []),
            },
        }
        latest = self.store.latest_succeeded_runs()
        shared: set[str] | None = None
        for factor_id in factor_ids:
            run = latest.get(factor_id)
            if run is None or run["mode"] != spec.mode:
                return None
            config = json.loads(run["config_json"]) if run.get("config_json") else None
            if not isinstance(config, dict):
                return None
            if config.get("data_version") != storage_io.current_version(self.store.path.parent.parent):
                return None
            if config.get("index_code") != spec.universe.index_code:
                return None
            if str(config.get("start_date", "")) > str(window_start.date()):
                return None
            if str(config.get("end_date", "")) < str(run_end.date()):
                return None
            if config.get("universe") != universe:
                return None
            shared = {run["run_id"]} if shared is None else (shared & {run["run_id"]})
            if not shared:
                return None
        run_id = next(iter(shared))
        run_factors = {item["factor_id"] for item in self.store.run_detail(run_id)["factors"]}
        if run_factors != set(factor_ids):
            return None
        # The run config may claim coverage its panel does not actually hold
        # (e.g. a cache whose manifest overstated its months); verify the data.
        panel_row = self.store.connection.execute(
            "SELECT path FROM artifact WHERE run_id=? AND kind='factor_values'",
            (run_id,),
        ).fetchone()
        if panel_row is None:
            return None
        panel_path = Path(str(panel_row["path"]))
        if not storage_io.exists(panel_path):
            return None
        panel_max = pd.to_datetime(
            storage_io.read_frame(panel_path, columns=["signal_date"])["signal_date"]
        ).max()
        if pd.Timestamp(run_end) - panel_max > pd.Timedelta(days=30):
            return None
        return run_id

    def _panel_path(self, run_id: str) -> Path:
        row = self.store.connection.execute(
            "SELECT path FROM artifact WHERE run_id=? AND kind='factor_values'",
            (run_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"run {run_id} 缺少 factor_values 面板产物")
        return Path(str(row["path"]))

    def _materialize(
        self, report_id: str, spec: ReportSpec,
        factors: list[dict[str, Any]], run_id: str,
    ) -> Path:
        output = self.reports_root() / report_id
        output.mkdir(parents=True, exist_ok=True)
        run_dir = self.data_root() / "factor_library" / "runs" / run_id
        storage_io.copy(run_dir / "monthly.parquet", output / "monthly.parquet")
        storage_io.copy(run_dir / "summary.csv", output / "summary.csv")
        storage_io.write_text(output / "spec.yaml",
            yaml.safe_dump(spec.to_dict(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return output

    # ------------------------------------------------------------ correlation

    @staticmethod
    def _write_correlation(output: Path, z: pd.DataFrame) -> pd.DataFrame | None:
        if z.shape[1] < 2:
            return None
        matrices = []
        for _date, block in z.groupby(level="signal_date", observed=True):
            matrices.append(block.rank(pct=True).corr().to_numpy(float))
        with np.errstate(invalid="ignore"):
            average = np.nanmean(np.stack(matrices), axis=0)
        matrix = pd.DataFrame(average, index=z.columns, columns=z.columns)
        storage_io.write_frame(matrix, output / "correlation.parquet", index=True)
        return matrix

    @staticmethod
    def _correlation_chart_data(matrix: pd.DataFrame) -> dict[str, Any]:
        labels = [str(item) for item in matrix.columns]
        cells = [
            [row, column, float(matrix.iloc[row, column])]
            for row in range(len(matrix))
            for column in range(len(matrix.columns))
        ]
        return {"labels": labels, "cells": cells}

    # ------------------------------------------------------------- combination

    def _write_combination(
        self, output: Path, spec: ReportSpec, factors: list[dict[str, Any]],
        forward: pd.Series, z: pd.DataFrame, ic: pd.DataFrame, fm: pd.DataFrame,
        wide: pd.DataFrame | None = None,
    ) -> dict[str, Any] | None:
        combine = spec.combine
        if combine is None or z.shape[1] < 2:
            return None
        splits = spec.resolved_splits()
        validation_end = splits["validation"][1]
        months = sorted(z.index.get_level_values("signal_date").unique())
        selection_months = [item for item in months if item <= validation_end]
        if len(selection_months) < combine.rolling_months:
            return None
        usable = list(z.columns)
        directions = {
            item["factor_id"]: item.get("expected_direction", "unknown") for item in factors
        }
        selection_ic = ic.loc[selection_months]
        signs: dict[str, float] = {}
        for column in usable:
            direction = directions.get(column, "unknown")
            mean_ic = selection_ic[column].mean()
            if direction == "negative" or (
                direction == "unknown" and pd.notna(mean_ic) and mean_ic < 0
            ):
                signs[column] = -1.0
            else:
                signs[column] = 1.0
        selection_z = z.loc[selection_months[-combine.rolling_months:]]
        correlation_stack = np.stack([
            selection_z.loc[month].rank(pct=True).corr(min_periods=10).to_numpy(float)
            for month in selection_months[-combine.rolling_months:]
        ])
        with np.errstate(invalid="ignore"):
            correlation = np.nanmean(correlation_stack, axis=0)
        correlation_frame = pd.DataFrame(correlation, index=usable, columns=usable)
        priority = selection_ic[usable].mean()
        selected = select_low_correlation_factors(
            correlation_frame, priority, threshold=combine.correlation_threshold
        )
        if len(selected) < 2:
            return None
        history_ic = selection_ic[selected]
        history_fm = fm.loc[selection_months, selected]
        weights: dict[str, pd.Series] = {}
        for method in combine.methods:
            history = history_fm if method == "factor_return_decay" else history_ic
            try:
                weights[method] = factor_weights(history, method)
            except (ValueError, np.linalg.LinAlgError):
                continue
        if not weights:
            return None
        signed = z[selected].mul(pd.Series(signs)[selected], axis=1)
        ml_keys: list[str] = []
        ml_predictions: dict[str, pd.Series] = {}
        ml_meta: dict[str, Any] = {}
        if spec.combine.ml is not None and wide is not None:
            ml_keys, ml_predictions, ml_meta = self._ml_composites(
                spec, wide, forward, selected
            )
        frames: list[pd.DataFrame] = []
        for month in months:
            block = signed.loc[month]
            target = forward.loc[month].reindex(block.index)
            for method, method_weights in weights.items():
                composite = combine_scores(block, method_weights)
                frame = pd.DataFrame({
                    "raw_value": composite,
                    "forward_return": target,
                }).dropna(subset=["raw_value"])
                if len(frame) < 3:
                    continue
                frame = frame.reset_index().rename(columns={"symbol": "symbol"})
                frame["signal_date"] = month
                frame["factor_name"] = method
                frame["neutralized_value"] = np.nan
                frames.append(frame)
            for ml_key, ml_series in ml_predictions.items():
                composite = ml_series.loc[month].reindex(block.index.get_level_values("symbol"))
                frame = pd.DataFrame({
                    "raw_value": composite,
                    "forward_return": target,
                }).dropna(subset=["raw_value"])
                if len(frame) < 3:
                    continue
                frame = frame.reset_index().rename(columns={"symbol": "symbol"})
                frame["signal_date"] = month
                frame["factor_name"] = ml_key
                frame["neutralized_value"] = np.nan
                frames.append(frame)
        if not frames:
            return None
        panel = pd.concat(frames, ignore_index=True)
        monthly, _summary = evaluate_factor_batch(
            panel, periods=splits,
        )
        methods: list[dict[str, Any]] = []
        chart_series: dict[str, pd.Series] = {}
        for method in (*combine.methods, *ml_keys):
            sample = monthly[
                (monthly["factor_name"] == method)
                & (monthly["value_type"] == "raw")
                & (monthly["orientation"] == "original")
            ].sort_values("signal_date")
            if sample.empty:
                continue
            ic_series = sample.set_index("signal_date")["rank_ic"]
            long_short = (
                sample.set_index("signal_date")["group_1"]
                - sample.set_index("signal_date")["group_5"]
            )
            performance = _performance(long_short)
            group_performance = {
                index: _performance(sample[f"group_{index}"]) for index in range(1, 6)
            }
            mean_groups = [sample[f"group_{index}"].mean() for index in range(1, 6)]
            best_group = max(
                range(1, 6),
                key=lambda index: (
                    group_performance[index].get("annualized_return")
                    if group_performance[index].get("annualized_return") is not None
                    else float("-inf")
                ),
            )
            monotonicity = float(-spearmanr(range(1, 6), mean_groups).statistic)
            standard_deviation = ic_series.std(ddof=1)
            methods.append({
                "key": method,
                "label": COMBINE_LABELS.get(method, method),
                "months": len(ic_series),
                "start": str(sample["signal_date"].min().date()),
                "end": str(sample["signal_date"].max().date()),
                "rank_ic": float(ic_series.mean()),
                "icir": float(ic_series.mean() / standard_deviation) if standard_deviation else None,
                "hac_t": float(newey_west_t(ic_series)),
                "coverage": float(sample["coverage"].mean()),
                "monotonicity": monotonicity,
                "long_short_annualized_return": performance.get("annualized_return"),
                "long_short_sharpe": performance.get("sharpe"),
                "long_short_max_drawdown": performance.get("max_drawdown"),
                "long_short_monthly_win_rate": performance.get("monthly_win_rate"),
                **{
                    f"group_{index}_annualized_return": group_performance[index].get(
                        "annualized_return"
                    )
                    for index in range(1, 6)
                },
                "best_group_index": best_group,
                "best_group_sharpe": group_performance[best_group].get("sharpe"),
                "periods": {
                    period: float(
                        sample[sample["period"].fillna("monitoring_2026") == period]["rank_ic"].mean()
                    )
                    if (sample["period"].fillna("monitoring_2026") == period).any() else None
                    for period in ("development", "validation", "test", "monitoring_2026")
                },
                "average_factor_count": len(selected),
                "source_factor_count": len(usable),
            })
            chart_series[method] = (1 + long_short).cumprod()
        nav_payload = {
            method: {
                "dates": [str(value.date()) for value in values.index],
                "values": [
                    None if pd.isna(item) else round(float(item), 4)
                    for item in values.to_numpy()
                ],
            }
            for method, values in chart_series.items()
        }
        payload: dict[str, Any] = {
            "methods": methods,
            "selection": {
                "correlation_threshold": combine.correlation_threshold,
                "rolling_months": combine.rolling_months,
                "selection_periods": combine.selection_periods,
                "selection_end": str(validation_end.date()),
                "selected_factors": selected,
                "signs": signs,
                **({"ml": {"config": spec.combine.ml, "models": ml_meta}}
                   if ml_meta else {}),
            },
            "nav": nav_payload,
        }
        storage_io.write_text(output / "combo.json",
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return payload

    def _ml_composites(
        self, spec: ReportSpec, wide: pd.DataFrame, forward: pd.Series,
        selected: list[str],
    ) -> tuple[list[str], dict[str, pd.Series], dict[str, Any]]:
        """Train stock-level ML composites on the selected factors (frozen protocol)."""
        config = spec.combine.ml
        splits = spec.resolved_splits()
        if config["feature_mode"] == "neutral":
            industries, cap = load_pit_context(self.data_root())
            features = neutralize_wide(wide[selected], industries, cap)
        else:
            features = wide[selected].groupby(
                level="signal_date", observed=True
            ).transform(zscore)
        keys: list[str] = []
        predictions: dict[str, pd.Series] = {}
        meta: dict[str, Any] = {}
        for key in config["models"]:
            try:
                prediction, model_meta = frozen_composite(
                    features, forward, splits, key, label_mode=config["label_mode"],
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            ml_key = f"ml_{key}"
            keys.append(ml_key)
            predictions[ml_key] = prediction
            meta[ml_key] = model_meta
        return keys, predictions, meta

    @staticmethod
    def _combination_chart_data(nav: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Align per-method NAV series onto a shared date axis for ECharts."""
        dates = sorted({
            date for values in nav.values() for date in values["dates"]
        })
        series = []
        for method, values in nav.items():
            value_map = dict(zip(values["dates"], values["values"]))
            series.append({
                "name": COMBINE_LABELS.get(method, method),
                "color": _COMBINE_COLORS.get(method, "#8b9292"),
                "values": [value_map.get(date) for date in dates],
            })
        return {"dates": dates, "series": series}

    # --------------------------------------------------------------- rendering

    def _metric_table(self, run_id: str, factor_ids: list[str]) -> list[dict[str, Any]]:
        placeholders = ",".join("?" for _ in factor_ids)
        rows = self.store.connection.execute(
            f"""SELECT m.factor_id, m.period, m.metric_name, m.metric_value
            FROM factor_metric m
            WHERE m.run_id=? AND m.factor_id IN ({placeholders})
              AND m.index_code='ALL_A' AND m.value_type='raw'
              AND m.orientation='original' AND m.cost_scenario='base_5bps'""",
            (run_id, *factor_ids),
        ).fetchall()
        table: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            key = (str(row["factor_id"]), str(row["period"]))
            if str(row["metric_name"]) in METRIC_COLUMNS:
                table.setdefault(key, {})[str(row["metric_name"])] = row["metric_value"]
        return [
            {"factor_id": factor_id, "period": period, **metrics}
            for (factor_id, period), metrics in sorted(
                table.items(),
                key=lambda item: (
                    item[0][0], _PERIOD_ORDER.get(item[0][1], 99), item[0][1],
                ),
            )
        ]

    def _write_report_text(
        self, output: Path, spec: ReportSpec, factors: list[dict[str, Any]],
        run_id: str, matrix: pd.DataFrame | None, combo: dict[str, Any] | None,
    ) -> None:
        metric_rows = self._metric_table(run_id, [item["factor_id"] for item in factors])
        monthly = storage_io.read_frame(output / "monthly.parquet")
        context = {
            "spec": spec,
            "factors": factors,
            "run_id": run_id,
            "metric_rows": metric_rows,
            "metric_columns": METRIC_COLUMNS,
            "period_labels": _SPLIT_LABELS,
            "split_ranges": {
                name: (str(start.date()), str(end.date()))
                for name, (start, end) in spec.resolved_splits().items()
            },
            "monitoring": spec.monitoring_bound() is not None,
            "monitoring_end": str(spec.monitoring_bound().date()) if spec.monitoring_bound() else None,
            "matrix": matrix,
            "matrix_size": 0 if matrix is None else len(matrix),
            "combo": combo,
            "combo_methods": (combo or {}).get("methods", []),
            "has_monthly": not monthly.empty,
            "version": __version__,
            "combine_labels": COMBINE_LABELS,
            "holding_period": spec.holding_period,
            "lookback_by_factor": {
                item["factor_id"]: item.get("lookback_days")
                for item in factors
            },
            "max_lookback_days": max(
                (item.get("lookback_days") or 0 for item in factors), default=0
            ),
            "echarts_js": _static_text("echarts.min.js"),
            "charts_js": _static_text("charts.js"),
            "correlation_chart": (
                None if matrix is None
                else ReportEngine._correlation_chart_data(matrix)
            ),
            "combo_chart": (
                None if combo is None or not combo.get("nav")
                else ReportEngine._combination_chart_data(combo["nav"])
            ),
        }
        markdown = _render_markdown(context)
        storage_io.write_text(output / "report.md", markdown, encoding="utf-8")
        from jinja2 import Environment, FileSystemLoader, StrictUndefined

        environment = Environment(
            loader=FileSystemLoader(Path(__file__).parent / "templates"),
            undefined=StrictUndefined, autoescape=False,
        )
        environment.filters["fmt"] = _format_metric
        html = environment.get_template("report.html").render(**context)
        storage_io.write_text(output / "report.html", html, encoding="utf-8")

    def _write_manifest(
        self, output: Path, spec: ReportSpec,
        factors: list[dict[str, Any]], run_id: str,
    ) -> None:
        revision_rows = self.store.connection.execute(
            "SELECT factor_id, revision_id FROM run_factor WHERE run_id=?",
            (run_id,),
        ).fetchall()
        revisions = {str(row["factor_id"]): str(row["revision_id"]) for row in revision_rows}
        files = {
            path.name: {"sha256": _sha256(path), "bytes": storage_io.stat(path).st_size}
            for path in sorted(storage_io.iterdir(output)) if storage_io.exists(path)
        }
        manifest = {
            "report_id": output.name,
            "name": spec.name,
            "mode": spec.mode,
            "watermark": None if spec.mode == "formal" else "NON-FORMAL SMOKE",
            "code_version": __version__,
            "run_id": run_id,
            "spec": spec.to_dict(),
            "storage_backend": "clickhouse",
            "data_version": self.store.run_detail(run_id)["config"].get("data_version"),
            "factors": [
                {
                    "factor_id": item["factor_id"], "name": item["name"],
                    "family": item["family"],
                    "expected_direction": item.get("expected_direction", "unknown"),
                    "lookback_days": item.get("lookback_days"),
                    "min_observations": item.get("min_observations"),
                    "revision_id": revisions.get(item["factor_id"]),
                }
                for item in factors
            ],
            "data_signature": data_signature(self.data_root()),
            "monitoring_2026_excluded_from_selection": True,
            "files": files,
        }
        storage_io.write_text(output / "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _format_metric(name: str, value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if name in {
        "coverage", "monthly_win_rate", "mean_turnover",
        "annualized_return", "max_drawdown",
    }:
        return f"{value:.1%}"
    return f"{value:.3f}"


def _render_markdown(context: dict[str, Any]) -> str:
    spec: ReportSpec = context["spec"]
    lines: list[str] = [f"# {spec.name}", ""]
    if context["spec"].mode == "smoke":
        lines += ["> **NON-FORMAL SMOKE**：本报告为探索性回测，结果带水印，不可用于正式结论。", ""]
    if spec.description:
        lines += [spec.description, ""]
    universe = spec.universe
    universe_parts = [universe.index_code]
    if universe.industries.get("include"):
        universe_parts.append("行业(含)：" + "、".join(universe.industries["include"]))
    if universe.industries.get("exclude"):
        universe_parts.append("行业(除)：" + "、".join(universe.industries["exclude"]))
    if universe.symbols.get("include"):
        universe_parts.append("代码(含)：" + "、".join(universe.symbols["include"]))
    if universe.symbols.get("exclude"):
        universe_parts.append("代码(除)：" + "、".join(universe.symbols["exclude"]))
    ranges = context["split_ranges"]
    split_line = (
        f"- 阶段划分：开发 {ranges['development'][0]} ~ {ranges['development'][1]}；"
        f"验证 {ranges['validation'][0]} ~ {ranges['validation'][1]}；"
        f"测试 {ranges['test'][0]} ~ {ranges['test'][1]}"
    )
    lines += [
        "## 研究设置", "",
        f"- 股票池：{'，'.join(universe_parts)}",
        f"- 研究区间：{spec.start_date} ~ {spec.end_date}"
        + (f"，监控期至 {context['monitoring_end']}（不参与选模）" if context["monitoring"] else ""),
        split_line,
        f"- 持有期：{context['holding_period']}（月末收盘形成信号、下一交易日开盘成交、持有至下月信号执行日）",
        f"- 因子回看：最长 {context['max_lookback_days']} 个交易日（各因子回看天数见单因子表）",
        f"- 因子：{len(context['factors'])} 个（run {context['run_id'][:8]}）",
        "- 口径：原始值五分组、行业内等权、月末收盘形成信号、下一交易日开盘成交；组合1扣 5bps 单边成本；测试期不调方向与参数",
        "",
    ]
    rows = context["metric_rows"]
    if rows:
        lines += ["## 单因子指标", ""]
        header = "| 因子 | 阶段 | 回看(日) | Rank IC | ICIR | HAC t | 单调性 | 覆盖率 | 组合1年化 | Sharpe | 最大回撤 | 月胜率 | 换手 |"
        lines += [header, "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in rows:
            lines.append(
                "| {factor} | {period} | {lookback} | {rank_ic} | {icir} | {hac_t} | {monotonicity} | "
                "{coverage} | {annualized_return} | {sharpe} | {max_drawdown} | "
                "{monthly_win_rate} | {mean_turnover} |".format(
                    factor=row["factor_id"],
                    period=context["period_labels"].get(row["period"], row["period"]),
                    lookback=(
                        str(context["lookback_by_factor"].get(row["factor_id"]))
                        if context["lookback_by_factor"].get(row["factor_id"]) is not None
                        else "—"
                    ),
                    rank_ic=_format_metric("rank_ic", row.get("rank_ic")),
                    icir=_format_metric("icir", row.get("icir")),
                    hac_t=_format_metric("hac_t", row.get("hac_t")),
                    monotonicity=_format_metric("monotonicity", row.get("monotonicity")),
                    coverage=_format_metric("coverage", row.get("coverage")),
                    annualized_return=_format_metric("annualized_return", row.get("annualized_return")),
                    sharpe=_format_metric("sharpe", row.get("sharpe")),
                    max_drawdown=_format_metric("max_drawdown", row.get("max_drawdown")),
                    monthly_win_rate=_format_metric("monthly_win_rate", row.get("monthly_win_rate")),
                    mean_turnover=_format_metric("mean_turnover", row.get("mean_turnover")),
                )
            )
        lines.append("")
    if context["matrix"] is not None:
        lines += [
            "## Spearman 相关矩阵",
            "",
            f"每个自然月内先对因子做截面标准化，再计算两两 Spearman 秩相关，图上为各月平均值（{context['matrix_size']} 个因子）。",
            "交互式热力图见 report.html；矩阵数据见 correlation.parquet。",
            "",
        ]
    if context["combo"] is not None and context["combo_methods"]:
        lines += ["## 合成方式对比", ""]
        header = ("| 合成方式 | 组合1年化 | 组合2年化 | 组合3年化 | 组合4年化 | 组合5年化 | "
                  "分层回测最优组合Sharpe | 多空年化 | 多空Sharpe | 多空回撤 | Rank IC | ICIR | HAC t | 开发IC | 验证IC | 测试IC | 监控IC |")
        lines += [header, "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for method in context["combo_methods"]:
            best_group = method.get("best_group_index")
            group_cells = []
            for index in range(1, 6):
                cell = _format_metric(
                    "annualized_return", method.get(f"group_{index}_annualized_return")
                )
                if index == best_group:
                    cell = f"**{cell}**"
                group_cells.append(cell)
            lines.append(
                "| {label} | {g1} | {g2} | {g3} | {g4} | {g5} | {best_sharpe} | {ls} | {sharpe} | {mdd} | "
                "{ic} | {icir} | {hac} | {dev} | {val} | {test} | {mon} |".format(
                    label=method["label"],
                    g1=group_cells[0], g2=group_cells[1], g3=group_cells[2],
                    g4=group_cells[3], g5=group_cells[4],
                    best_sharpe=_format_metric("sharpe", method.get("best_group_sharpe")),
                    ls=_format_metric("annualized_return", method["long_short_annualized_return"]),
                    sharpe=_format_metric("sharpe", method["long_short_sharpe"]),
                    mdd=_format_metric("max_drawdown", method["long_short_max_drawdown"]),
                    ic=_format_metric("rank_ic", method["rank_ic"]),
                    icir=_format_metric("icir", method["icir"]),
                    hac=_format_metric("hac_t", method["hac_t"]),
                    dev=_format_metric("rank_ic", method["periods"].get("development")),
                    val=_format_metric("rank_ic", method["periods"].get("validation")),
                    test=_format_metric("rank_ic", method["periods"].get("test")),
                    mon=_format_metric("rank_ic", method["periods"].get("monitoring_2026")),
                )
            )
        selection = context["combo"]["selection"]
        lines += [
            "",
            "交互式多空净值对比图见 report.html；数值见 combo.json。",
            "",
            "口径：方向符号、去相关筛选、合成权重与方法只在 "
            + "、".join(context["period_labels"].get(item, item) for item in selection["selection_periods"])
            + f" 期确定（截至 {selection['selection_end']}），测试与监控期仅评估；"
            f"滚动窗口 {selection['rolling_months']} 个月，相关阈值 {selection['correlation_threshold']}，"
            f"入选因子 {len(selection['selected_factors'])} 个。",
            "",
        ]
    lines += ["## 数据与代码", "", f"- 报告引擎版本：{context['version']}", ""]
    return "\n".join(lines)
