"""Long-only, cost-aware evaluation of every ML combination in formal reports."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from mlquant import storage_io
from mlquant.combine import ML_METHODS, combine_scores, factor_weights
from mlquant.ml_composite import frozen_composite, load_pit_context
from mlquant.ml_frequency import forward_label_end_dates, neutralize_frequency_features
from mlquant.portfolio_evaluation import (
    DEFAULT_COST_SCENARIOS,
    benchmark_returns,
    load_adjusted_market,
    performance_stats,
    simulate_long_only,
)
from mlquant.reassessment_audit import require_reassessment_data
from mlquant.report_engine import build_cross_section
from mlquant.report_spec import parse_spec
from mlquant.workflows._support import finish, invoke, progress, report_record, validate_config


def report_scores(
    root: Path, row: dict
) -> tuple[pd.DataFrame, object, dict[str, object]]:
    report_dir = Path(row["path"])
    spec = parse_spec(json.loads(row["spec_json"]))
    manifest = json.loads(storage_io.read_text(report_dir / "manifest.json", encoding="utf-8"))
    combo = json.loads(storage_io.read_text(report_dir / "combo.json", encoding="utf-8"))
    factor_ids = [str(item["factor_id"]) for item in manifest["factors"]]
    panel_path = root / "factor_library" / "generated_panels" / f"{manifest['run_id']}.parquet"
    wide, forward, z, ic, _fm = build_cross_section(panel_path, factor_ids)
    selection = combo["selection"]
    selected = [str(item) for item in selection["selected_factors"]]
    signs = pd.Series({str(k): float(v) for k, v in selection["signs"].items()})
    splits = spec.resolved_splits()
    calendar = storage_io.read_frame(root / "equity" / "calendar.parquet")
    ends = forward_label_end_dates(pd.DatetimeIndex(ic.index), calendar)
    selection_dates = ends.index[ends.notna() & (ends <= splits["validation"][1])]
    signed = z[selected].mul(signs.reindex(selected).fillna(1.0), axis=1)
    columns: dict[str, pd.Series] = {}
    for method in ML_METHODS:
        weights = factor_weights(ic.loc[selection_dates, selected], method)
        columns[f"factor_ml_{method}"] = signed.groupby(
            level="signal_date", observed=True, group_keys=False
        ).apply(lambda block, weight=weights: combine_scores(block, weight))

    ml_config = selection["ml"]["config"]
    industries, cap = load_pit_context(root)
    features = neutralize_frequency_features(wide[selected], industries, cap)
    for model in ml_config["models"]:
        progress(f"fitting {spec.universe.index_code} stock_ml_{model}", flush=True)
        prediction, _meta = frozen_composite(
            features,
            forward,
            splits,
            str(model),
            label_mode=ml_config["label_mode"],
            label_end_dates=pd.Series(
                features.index.get_level_values("signal_date").map(ends), index=features.index
            ),
        )
        columns[f"stock_ml_{model}"] = prediction
    scores = pd.concat(columns, axis=1).sort_index()
    return scores, spec, manifest




@dataclass(frozen=True, slots=True, kw_only=True)
class Config:
    """Explicit workflow inputs; paths never depend on the source checkout."""
    root: Path
    report_id: list[str]
    top_n: int = 50
    output: Path
    json: bool = False

    def __post_init__(self):
        validate_config(self, {})


def run(args: Config) -> dict:
    """Execute with Python inputs and return artifact metadata; never exits the host."""
    root = Path(args.root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = [report_record(root, report_id) for report_id in args.report_id]

    for row in rows:
        universe = parse_spec(json.loads(row["spec_json"])).universe.index_code
        try:
            require_reassessment_data(root, output, index_code=universe)
        except ValueError as error:
            progress(json.dumps({"ok": False, "error": {"code": "REASSESSMENT_DATA_BLOCKED", "message": str(error)}}))
            raise
    daily, status, calendar = load_adjusted_market(root)
    members_path = root / "equity" / "index_members.parquet"
    members = storage_io.read_frame(members_path) if storage_io.exists(members_path) else pd.DataFrame()
    summary_rows: list[dict[str, object]] = []
    return_blocks: list[pd.DataFrame] = []
    for row in rows:
        scores, spec, _manifest = report_scores(root, row)
        index_code = spec.universe.index_code
        benchmark = benchmark_returns(
            scores, daily, calendar, index_code=index_code, index_members=members
        )
        storage_io.write_frame(scores, output / f"scores_{index_code.replace('.', '_')}.parquet")
        for method in scores.columns:
            for scenario in DEFAULT_COST_SCENARIOS:
                returns, _trades, diagnostics = simulate_long_only(
                    scores[method],
                    daily,
                    status,
                    calendar,
                    top_n=args.top_n,
                    scenario=scenario,
                )
                aligned = pd.concat([returns, benchmark], axis=1)
                aligned["report_id"] = row["report_id"]
                aligned["index_code"] = index_code
                aligned["method"] = method
                aligned["cost_scenario"] = scenario.name
                aligned.index.name = "signal_date"
                return_blocks.append(aligned.reset_index())
                periods = {**spec.resolved_splits()}
                monitoring_end = spec.monitoring_bound()
                if monitoring_end is not None:
                    periods["monitoring"] = (
                        spec.resolved_splits()["test"][1] + pd.Timedelta(days=1),
                        monitoring_end,
                    )
                for period, (start, end) in periods.items():
                    part = aligned.loc[
                        (aligned.index >= start) & (aligned.index <= end)
                    ]
                    stats = performance_stats(
                        part["portfolio_return"],
                        part["benchmark_return"],
                        periods_per_year=12,
                    )
                    summary_rows.append(
                        {
                            "report_id": row["report_id"],
                            "index_code": index_code,
                            "frequency": "monthly",
                            "method": method,
                            "cost_scenario": scenario.name,
                            "period": period,
                            **stats,
                            "full_sample_annual_turnover": diagnostics["notional"]
                            / diagnostics["average_equity"]
                            / max(len(returns) / 12, 1 / 12),
                            "full_sample_annual_explicit_cost": diagnostics["explicit_cost"]
                            / diagnostics["average_equity"]
                            / max(len(returns) / 12, 1 / 12),
                            "full_sample_annual_slippage_cost": diagnostics["slippage_cost"]
                            / diagnostics["average_equity"]
                            / max(len(returns) / 12, 1 / 12),
                            "filled_orders": diagnostics["filled_orders"],
                            "rejected_orders": diagnostics["rejected_orders"],
                        }
                    )
                progress(f"finished {index_code} {method} {scenario.name}")
    summary = pd.DataFrame(summary_rows)
    returns = pd.concat(return_blocks, ignore_index=True)
    storage_io.write_csv(summary, output / "summary.csv", index=False, encoding="utf-8-sig")
    storage_io.write_frame(returns, output / "returns.parquet", index=False)
    metadata = {
        "top_n": args.top_n,
        "reports": args.report_id,
        "cost_scenarios": [asdict(scenario) for scenario in DEFAULT_COST_SCENARIOS],
        "benchmark": "PIT official index weights; ALL_A float-market-cap weights",
        "long_only": True,
        "execution": "next trading day open",
    }
    storage_io.write_text(output / "metadata.json",
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return finish(output=args.output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--report-id", action="append", required=True)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    return invoke(run, Config, args)


if __name__ == "__main__":
    raise SystemExit(main())
