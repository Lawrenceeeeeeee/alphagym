"""Signal export: materialize a frozen report combination as a paper-trading list.

Reads a finished report's offline artifacts (spec.yaml, combo.json, manifest.json
and the linked run panel), rebuilds the chosen composite exactly the way the
report engine did — factor-level weights for linear methods, the frozen
development-trained model for ml_* methods — and exports the latest month's
top-N names as ``signal_latest.csv`` plus a ``state.json`` contract for the QMT
paper trader.

These files are the only interface between offline research and the simulated
account; they are watermarked 模拟盘-only and never trigger live orders by
themselves.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from mlquant import storage_io
from mlquant.combine import combine_scores, factor_weights
from mlquant.factor_store import FactorStore
from mlquant.ml_composite import frozen_composite, load_pit_context, neutralize_wide
from mlquant.report_engine import build_cross_section
from mlquant.report_spec import ReportSpec, parse_spec
from mlquant.research import zscore

WATERMARK = "仅模拟盘使用：信号由离线研究报告生成，不构成投资建议，禁止用于实盘下单。"


@dataclass
class SignalBundle:
    path: Path
    asof: pd.Timestamp
    effective_trade_date: pd.Timestamp
    top_n: int
    method: str
    symbols: list[str]


def _report_dir(root: Path, report_id: str) -> Path:
    directory = root / "factor_library" / "reports" / report_id
    if not storage_io.exists(directory / "manifest.json"):
        raise FileNotFoundError(f"报告产物不存在：{directory}")
    return directory


def _sha256(path: Path) -> str:
    return hashlib.sha256(storage_io.read_bytes(path)).hexdigest()


def _open_dates(calendar: pd.DataFrame) -> pd.DatetimeIndex:
    required = {"trade_date", "is_open"}
    missing = required - set(calendar.columns)
    if missing:
        raise ValueError(f"交易日历缺少字段：{sorted(missing)}")
    opened = pd.to_datetime(
        calendar.loc[calendar["is_open"].astype(bool), "trade_date"], errors="coerce"
    ).dropna()
    return pd.DatetimeIndex(opened).normalize().drop_duplicates().sort_values()


def _next_open_date(asof: pd.Timestamp, calendar: pd.DataFrame) -> pd.Timestamp:
    opened = _open_dates(calendar)
    future = opened[opened > pd.Timestamp(asof).normalize()]
    if future.empty:
        raise ValueError(f"交易日历没有 {pd.Timestamp(asof).date()} 之后的开市日")
    return pd.Timestamp(future[0]).normalize()


def _next_month_signal_date(
    asof: pd.Timestamp, calendar: pd.DataFrame
) -> pd.Timestamp | None:
    """Return the next calendar-backed month-end signal date when fully known."""
    dates = pd.to_datetime(calendar["trade_date"], errors="coerce").dropna()
    next_period = pd.Timestamp(asof).to_period("M") + 1
    natural_end = next_period.to_timestamp(how="end").normalize()
    if dates.empty or dates.max().normalize() < natural_end:
        return None
    opened = _open_dates(calendar)
    in_month = opened[opened.to_period("M") == next_period]
    if in_month.empty:
        raise ValueError(f"交易日历中 {next_period} 没有开市日")
    return pd.Timestamp(in_month[-1]).normalize()


def _latest_scores(
    report_dir: Path,
    spec: ReportSpec,
    selection: dict[str, Any],
    method: str,
    root: Path,
    smooth_months: int = 1,
) -> tuple[pd.Series, pd.Timestamp]:
    manifest = json.loads(storage_io.read_text(report_dir / "manifest.json", encoding="utf-8"))
    run_id = str(manifest["run_id"])
    factor_ids = [str(item["factor_id"]) for item in manifest["factors"]]
    panel_path = root / "factor_library" / "generated_panels" / f"{run_id}.parquet"
    if not storage_io.exists(panel_path):
        raise FileNotFoundError(f"报告关联的面板不存在：{panel_path}")
    wide, forward, z, ic, fm = build_cross_section(panel_path, factor_ids)
    months = sorted(wide.index.get_level_values("signal_date").unique())
    validation_end = spec.resolved_splits()["validation"][1]
    selection_months = [item for item in months if item <= validation_end]
    selected = [str(item) for item in selection["selected_factors"]]
    signs = pd.Series({str(k): float(v) for k, v in selection["signs"].items()})
    latest = months[-1]
    if method.startswith("ml_"):
        config = selection["ml"]["config"]
        model_key = method.removeprefix("ml_")
        if model_key not in config["models"]:
            raise ValueError(f"combo.json 未包含模型 {model_key}")
        if config["feature_mode"] == "neutral":
            industries, cap = load_pit_context(root)
            features = neutralize_wide(wide[selected], industries, cap)
        else:
            features = wide[selected].groupby(
                level="signal_date", observed=True
            ).transform(zscore)
        prediction, _meta = frozen_composite(
            features, forward, spec.resolved_splits(), model_key,
            label_mode=config["label_mode"],
        )
        if smooth_months > 1:
            score_frame = prediction.unstack().sort_index()
            score_frame = score_frame.rolling(smooth_months, min_periods=1).mean()
            prediction = score_frame.stack()
        return prediction.loc[latest], latest
    history = fm.loc[selection_months, selected] if method == "factor_return_decay" \
        else ic.loc[selection_months, selected]
    weights = factor_weights(history, method)
    signed = z[selected].mul(signs.reindex(selected).fillna(1.0), axis=1)
    block = signed.loc[latest]
    scores = combine_scores(block, weights)
    return scores, latest


def export_signal(
    store: FactorStore,
    report_id: str,
    method: str,
    *,
    top_n: int = 50,
    smooth_months: int = 1,
) -> SignalBundle:
    """Export the latest-month top-N list of a finished report combination."""
    row = store.report_detail(report_id)
    if row["status"] != "succeeded" or not row.get("path"):
        raise ValueError(f"报告未成功完成，无法导出信号：{row['status']}")
    root = store.path.parent.parent
    report_dir = Path(str(row["path"]))
    spec = parse_spec(dict(row["spec"]))
    combo_path = report_dir / "combo.json"
    if not storage_io.exists(combo_path):
        raise FileNotFoundError(f"报告缺少 combo.json（该报告未做合成对比）：{report_dir}")
    combo = json.loads(storage_io.read_text(combo_path, encoding="utf-8"))
    known = {str(item["key"]) for item in combo.get("methods", [])}
    if method not in known:
        raise ValueError(f"combo.json 中不存在方法 {method}；可用：{sorted(known)}")
    scores, asof = _latest_scores(
        report_dir, spec, combo["selection"], method, root, smooth_months=smooth_months
    )
    manifest = json.loads(storage_io.read_text(report_dir / "manifest.json", encoding="utf-8"))
    top = scores.dropna().sort_values(ascending=False).head(top_n)
    if top.empty:
        raise ValueError(f"{method} 在 {asof.date()} 无可用得分")
    weight = 1.0 / len(top)
    target = pd.DataFrame({
        "symbol": top.index.astype(str),
        "score": [float(item) for item in top.to_numpy()],
        "target_weight": [weight] * len(top),
    })
    target = target.sort_values("score", ascending=False).reset_index(drop=True)

    output = root / "qmt_signals" / f"{report_id[:8]}-{method}"
    output.mkdir(parents=True, exist_ok=True)
    signal_path = output / "signal_latest.csv"
    storage_io.write_csv(target, signal_path, index=False, encoding="utf-8-sig")
    calendar = storage_io.read_frame(root / "equity" / "calendar.parquet")
    open_days = _open_dates(calendar)
    latest_data = open_days.max()
    effective_trade_date = _next_open_date(asof, calendar)
    next_signal_date = _next_month_signal_date(asof, calendar)
    selected_ids = [str(item) for item in combo["selection"]["selected_factors"]]
    factors_by_id = {str(item["factor_id"]): item for item in manifest["factors"]}
    missing_factors = sorted(set(selected_ids) - set(factors_by_id))
    if missing_factors:
        raise ValueError(f"报告 manifest 缺少已选因子：{missing_factors}")
    factor_manifest = [
        {
            "factor_id": factor_id,
            "revision_id": str(factors_by_id[factor_id]["revision_id"]),
            "lookback_days": int(factors_by_id[factor_id]["lookback_days"]),
        }
        for factor_id in selected_ids
    ]
    selection_payload = json.dumps(
        combo["selection"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    splits = spec.resolved_splits()
    state = {
        "schema_version": 2,
        "method": method,
        "report_id": report_id,
        "run_id": manifest.get("run_id"),
        "asof": str(asof.date()),
        "signal_date": str(asof.date()),
        "effective_trade_date": str(effective_trade_date.date()),
        "next_rebalance_signal_month": str(asof.to_period("M") + 1),
        "next_rebalance_signal_date": (
            str(next_signal_date.date()) if next_signal_date is not None else None
        ),
        "top_n": top_n,
        "holding_period": spec.holding_period,
        "universe": spec.universe.index_code,
        "selection": combo["selection"],
        "factor_manifest": factor_manifest,
        "selection_sha256": hashlib.sha256(selection_payload).hexdigest(),
        "signal_sha256": _sha256(signal_path),
        "report_manifest_sha256": _sha256(report_dir / "manifest.json"),
        "monitoring_only": bool(asof > splits["test"][1]),
        "stale": bool(latest_data - asof > pd.Timedelta(days=90)),
        "latest_data_date": str(latest_data.date()),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "watermark": WATERMARK,
    }
    storage_io.write_text(output / "state.json",
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    # The QMT bridge consumes explicit exported files; ClickHouse remains authoritative.
    storage_io.export_resource(signal_path, signal_path)
    storage_io.export_resource(output / "state.json", output / "state.json")
    return SignalBundle(
        path=output,
        asof=asof,
        effective_trade_date=effective_trade_date,
        top_n=top_n,
        method=method,
        symbols=target["symbol"].tolist(),
    )
