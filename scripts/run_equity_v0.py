from __future__ import annotations

import argparse
from pathlib import Path

from mlquant.local_v0 import (
    LocalSourcePaths,
    build_expanded_baseline_combinations,
    build_expanded_combinations,
    build_expanded_factor_panel,
    build_full_combinations,
    build_full_factor_panel,
    build_local_contract,
    build_short_horizon_panel,
    build_technical_combinations,
    build_technical_factor_panel,
    build_v0_combinations,
    build_v0_factor_panel,
    build_v0_portfolios,
    build_v0_report,
    evaluate_expanded_factors,
    evaluate_full_factors,
    evaluate_short_horizon,
    evaluate_technical_factors,
    evaluate_v0,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the non-formal local HS300/CSI500 factor v0")
    parser.add_argument("--qmt-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("research/series/equity_v0_results.md"))
    parser.add_argument(
        "--stage",
        choices=(
            "contract",
            "panel",
            "evaluate",
            "portfolio",
            "short",
            "expanded-combine",
            "technical",
            "combine",
            "report",
            "all",
        ),
        default="all",
    )
    args = parser.parse_args()
    paths = LocalSourcePaths(args.qmt_root.resolve(), args.source_root.resolve(), args.root.resolve())
    if args.stage in {"contract", "all"}:
        print(build_local_contract(paths), flush=True)
    panel = None
    if args.stage in {"panel", "all"}:
        panel = build_v0_factor_panel(paths)
        print(f"panel rows={len(panel):,}", flush=True)
    if args.stage in {"evaluate", "all"}:
        monthly, summary = evaluate_v0(paths, panel)
        print(f"monthly rows={len(monthly):,}; summary rows={len(summary):,}", flush=True)
    if args.stage in {"portfolio", "all"}:
        detail, metrics = build_v0_portfolios(paths, panel)
        print(f"portfolio rows={len(detail):,}; metric rows={len(metrics):,}", flush=True)
    if args.stage == "short":
        short_panel = build_short_horizon_panel(paths)
        print(f"short panel rows={len(short_panel):,}", flush=True)
        monthly, summary = evaluate_short_horizon(paths, short_panel)
        print(f"short monthly rows={len(monthly):,}; summary rows={len(summary):,}", flush=True)
        detail, metrics = build_v0_portfolios(
            paths, short_panel, output_prefix="short_horizon"
        )
        print(
            f"short portfolio rows={len(detail):,}; metric rows={len(metrics):,}",
            flush=True,
        )
    if args.stage == "expanded-combine":
        expanded_panel = build_expanded_factor_panel(paths)
        print(
            f"expanded panel rows={len(expanded_panel):,}; "
            f"factors={expanded_panel['factor_name'].nunique()}",
            flush=True,
        )
        monthly, summary = evaluate_expanded_factors(paths, expanded_panel)
        print(
            f"expanded monthly rows={len(monthly):,}; summary rows={len(summary):,}",
            flush=True,
        )
        portfolios, metrics, selection = build_expanded_combinations(
            paths, expanded_panel
        )
        print(
            f"expanded combination rows={len(portfolios):,}; "
            f"metric rows={len(metrics):,}; selection={selection}",
            flush=True,
        )
        baseline_portfolios, baseline_metrics, baseline_selection = (
            build_expanded_baseline_combinations(paths)
        )
        print(
            f"comparable baseline rows={len(baseline_portfolios):,}; "
            f"metric rows={len(baseline_metrics):,}; selection={baseline_selection}",
            flush=True,
        )
    if args.stage == "technical":
        technical_panel = build_technical_factor_panel(paths)
        print(
            f"technical panel rows={len(technical_panel):,}; "
            f"factors={technical_panel['factor_name'].nunique()}",
            flush=True,
        )
        monthly, summary = evaluate_technical_factors(paths, technical_panel)
        print(
            f"technical monthly rows={len(monthly):,}; summary rows={len(summary):,}",
            flush=True,
        )
        portfolios, metrics, selection = build_technical_combinations(
            paths, technical_panel
        )
        print(
            f"technical combination rows={len(portfolios):,}; "
            f"metric rows={len(metrics):,}; selection={selection}",
            flush=True,
        )
        del technical_panel, monthly, summary, portfolios, metrics
        expanded_panel = build_expanded_factor_panel(paths)
        evaluate_expanded_factors(paths, expanded_panel)
        del expanded_panel
        full_panel = build_full_factor_panel(paths)
        print(
            f"full panel rows={len(full_panel):,}; factors={full_panel['factor_name'].nunique()}",
            flush=True,
        )
        monthly, summary = evaluate_full_factors(paths, full_panel)
        print(
            f"full monthly rows={len(monthly):,}; summary rows={len(summary):,}",
            flush=True,
        )
        portfolios, metrics, selection = build_full_combinations(paths, full_panel)
        print(
            f"full combination rows={len(portfolios):,}; metric rows={len(metrics):,}; "
            f"selection={selection}",
            flush=True,
        )
    if args.stage in {"combine", "all"}:
        portfolios, metrics, selection = build_v0_combinations(paths, panel)
        print(
            f"combination rows={len(portfolios):,}; metric rows={len(metrics):,}; "
            f"selection={selection}",
            flush=True,
        )
    if args.stage in {"report", "all"}:
        print(build_v0_report(paths, args.report.resolve()), flush=True)


if __name__ == "__main__":
    main()
