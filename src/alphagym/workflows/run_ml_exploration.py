"""Run the stock-level ML composite exploration on a curated factor pool.

Panels come from succeeded automatic runs (ALL_A may span several batch
panels; broad indices come from one panel each), so no backtest is recomputed.
Outputs land under {root}/experiments/ml_composite/{tag}/: predictions.parquet,
monthly.parquet, summary.csv, meta.json and report.md (plain text notes, not
the formal report area).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from alphagym import storage_io
from alphagym.audit import PERIODS
from alphagym.factor_store import FactorStore
from alphagym.ml_composite import (
    MODEL_KEYS,
    CompositeDataset,
    load_panel_wide,
    load_pit_context,
    run_composites,
)
from alphagym.workflows._support import finish, invoke, progress, validate_config


def _resolve_panels(root: Path, index_code: str, pool: list[str]) -> list[Path]:
    """Succeeded runs covering 2014-01-01..2025-12-31 for the requested index."""
    store = FactorStore.from_root(root)
    rows = store.connection.execute("""
        SELECT r.run_id, r.config_json, GROUP_CONCAT(rf.factor_id) AS factors
        FROM research_run r JOIN run_factor rf ON rf.run_id = r.run_id
        WHERE r.status='succeeded' GROUP BY r.run_id
    """).fetchall()
    panels: list[Path] = []
    for row in rows:
        config = json.loads(row["config_json"]) if row["config_json"] else {}
        if config.get("index_code") != index_code:
            continue
        if not (str(config.get("start_date", "")) <= "2014-01-01"
                and str(config.get("end_date", "")) >= "2025-12-31"):
            continue
        if not (set(row["factors"].split(",")) & set(pool)):
            continue
        path = root / "factor_library" / "generated_panels" / f"{row['run_id']}.parquet"
        if storage_io.exists(path):
            panels.append(path)
    if not panels:
        raise FileNotFoundError(
            f"没有覆盖 2014-2025 的 {index_code} 因子面板；请先运行 scripts/backfill_index_pool.py"
        )
    return sorted(panels)


def load_dataset(root: Path, index_code: str, pool: list[str]) -> CompositeDataset:
    panels = _resolve_panels(root, index_code, pool)
    frames: list[pd.DataFrame] = []
    for path in panels:
        frame = storage_io.read_frame(path, columns=[
            "signal_date", "symbol", "factor_name", "raw_value", "forward_return", "index_code",
        ])
        frame["signal_date"] = pd.to_datetime(frame["signal_date"])
        frames.append(frame[frame["factor_name"].isin(pool)])
    panel = pd.concat(frames, ignore_index=True)
    dataset = load_panel_wide(panel, pool, index_code)
    months = dataset.wide_raw.index.get_level_values("signal_date").unique()
    if months.min() > pd.Timestamp("2014-03-31") or months.max() < pd.Timestamp("2025-12-31"):
        raise ValueError(f"panel window looks wrong: {months.min()} .. {months.max()}")
    return dataset




@dataclass(frozen=True, slots=True, kw_only=True)
class Config:
    """Explicit workflow inputs; paths never depend on the source checkout."""
    root: Path
    pool: Path
    index_code: str = 'ALL_A'
    feature_mode: str = 'neutral'
    label_mode: str = 'return'
    protocol: str = 'frozen'
    models: str = ','.join(MODEL_KEYS)
    refit_months: int = 12
    tag: str | None = None
    json: bool = False

    def __post_init__(self):
        validate_config(self, {'feature_mode': ('z', 'neutral'), 'label_mode': ('return', 'rank'), 'protocol': ('frozen', 'walk_forward')})


def run(args: Config) -> dict:
    """Execute with Python inputs and return artifact metadata; never exits the host."""

    root = Path(args.root)
    pool = [str(item) for item in yaml.safe_load(storage_io.read_text(Path(args.pool), encoding="utf-8"))["pool"]]
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    tag = args.tag or (
        f"{args.index_code}_{args.feature_mode}_{args.label_mode}_{args.protocol}_"
        f"{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}"
    )
    output = root / "experiments" / "ml_composite" / tag
    output.mkdir(parents=True, exist_ok=True)
    progress(f"[ml] index={args.index_code} pool={len(pool)} mode={args.feature_mode}/"
          f"{args.label_mode}/{args.protocol} -> {output}", flush=True)

    dataset = load_dataset(root, args.index_code, pool)
    industries, cap = load_pit_context(root)
    results = run_composites(
        dataset, industries, cap, PERIODS,
        models=models, feature_mode=args.feature_mode, label_mode=args.label_mode,
        protocol=args.protocol, refit_months=args.refit_months,
    )

    predictions = pd.concat([item.panel for item in results], ignore_index=True)
    monthly = pd.concat([item.monthly for item in results], ignore_index=True)
    summary = pd.concat([item.summary for item in results], ignore_index=True)
    storage_io.write_frame(predictions, output / "predictions.parquet", index=False)
    storage_io.write_frame(monthly, output / "monthly.parquet", index=False)
    storage_io.write_csv(summary, output / "summary.csv", index=False)
    meta = {
        "index_code": args.index_code,
        "pool": pool,
        "feature_mode": args.feature_mode,
        "label_mode": args.label_mode,
        "protocol": args.protocol,
        "refit_months": args.refit_months,
        "splits": {name: [str(a.date()), str(b.date())] for name, (a, b) in PERIODS.items()},
        "months": {
            "start": str(dataset.wide_raw.index.get_level_values("signal_date").min().date()),
            "end": str(dataset.wide_raw.index.get_level_values("signal_date").max().date()),
        },
        "models": [item.meta for item in results],
    }
    storage_io.write_text(output / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        f"# ML 合成探索：{args.index_code}（{args.feature_mode}/{args.label_mode}/{args.protocol}）",
        "",
        f"- 因子池：{len(pool)} 个（{', '.join(pool[:6])}…）",
        f"- 区间：{meta['months']['start']} ~ {meta['months']['end']}",
        "",
        "## 分阶段 Rank IC（original）",
        "",
        "| 模型 | 开发 | 验证 | 测试 | 全样本 | 换手 |",
        "|---|---|---|---|---|---|",
    ]
    for item in results:
        raw = item.monthly[
            (item.monthly["value_type"] == "raw") & (item.monthly["orientation"] == "original")
        ]
        if raw.empty:
            continue
        def ic(period: str, raw: pd.DataFrame = raw) -> str:
            if period == "all":
                value = raw["rank_ic"].mean()
            else:
                sample = raw[raw["period"] == period]
                value = sample["rank_ic"].mean() if len(sample) else float("nan")
            return f"{value:.4f}" if pd.notna(value) else "—"
        lines.append(
            f"| {item.meta['model_label']}({item.meta['model_key']}) | {ic('development')} "
            f"| {ic('validation')} | {ic('test')} | {ic('all')} "
            f"| {raw['rank_turnover'].mean():.2f} |"
        )
    lines += ["", "产物：predictions.parquet / monthly.parquet / summary.csv / meta.json"]
    storage_io.write_text(output / "report.md", "\n".join(lines) + "\n", encoding="utf-8")
    progress(f"[ml] done -> {output / 'report.md'}", flush=True)
    return finish(output=output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--index-code", default="ALL_A")
    parser.add_argument("--feature-mode", choices=("z", "neutral"), default="neutral")
    parser.add_argument("--label-mode", choices=("return", "rank"), default="return")
    parser.add_argument("--protocol", choices=("frozen", "walk_forward"), default="frozen")
    parser.add_argument("--models", default=",".join(MODEL_KEYS))
    parser.add_argument("--refit-months", type=int, default=12)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    return invoke(run, Config, args)


if __name__ == "__main__":
    raise SystemExit(main())
