from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from alphagym.local_v0 import (
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
from alphagym.workflows._support import finish, invoke, progress, validate_config


@dataclass(frozen=True, slots=True, kw_only=True)
class Config:
    """Explicit workflow inputs; paths never depend on the source checkout."""
    qmt_root: Path
    source_root: Path
    root: Path
    report: Path
    stage: str = 'all'
    json: bool = False

    def __post_init__(self):
        validate_config(self, {'stage': ('contract', 'panel', 'evaluate', 'portfolio', 'short', 'expanded-combine', 'technical', 'combine', 'report', 'all')})


def run(args: Config) -> dict:
    """Execute with Python inputs and return artifact metadata; never exits the host."""
    paths = LocalSourcePaths(args.qmt_root.resolve(), args.source_root.resolve(), args.root.resolve())
    if args.stage in {"contract", "all"}:
        progress(build_local_contract(paths), flush=True)
    panel = None
    if args.stage in {"panel", "all"}:
        panel = build_v0_factor_panel(paths)
        progress(f"panel rows={len(panel):,}", flush=True)
    if args.stage in {"evaluate", "all"}:
        monthly, summary = evaluate_v0(paths, panel)
        progress(f"monthly rows={len(monthly):,}; summary rows={len(summary):,}", flush=True)
    if args.stage in {"portfolio", "all"}:
        detail, metrics = build_v0_portfolios(paths, panel)
        progress(f"portfolio rows={len(detail):,}; metric rows={len(metrics):,}", flush=True)
    if args.stage == "short":
        short_panel = build_short_horizon_panel(paths)
        progress(f"short panel rows={len(short_panel):,}", flush=True)
        monthly, summary = evaluate_short_horizon(paths, short_panel)
        progress(f"short monthly rows={len(monthly):,}; summary rows={len(summary):,}", flush=True)
        detail, metrics = build_v0_portfolios(
            paths, short_panel, output_prefix="short_horizon"
        )
        progress(
            f"short portfolio rows={len(detail):,}; metric rows={len(metrics):,}",
            flush=True,
        )
    if args.stage == "expanded-combine":
        expanded_panel = build_expanded_factor_panel(paths)
        progress(
            f"expanded panel rows={len(expanded_panel):,}; "
            f"factors={expanded_panel['factor_name'].nunique()}",
            flush=True,
        )
        monthly, summary = evaluate_expanded_factors(paths, expanded_panel)
        progress(
            f"expanded monthly rows={len(monthly):,}; summary rows={len(summary):,}",
            flush=True,
        )
        portfolios, metrics, selection = build_expanded_combinations(
            paths, expanded_panel
        )
        progress(
            f"expanded combination rows={len(portfolios):,}; "
            f"metric rows={len(metrics):,}; selection={selection}",
            flush=True,
        )
        baseline_portfolios, baseline_metrics, baseline_selection = (
            build_expanded_baseline_combinations(paths)
        )
        progress(
            f"comparable baseline rows={len(baseline_portfolios):,}; "
            f"metric rows={len(baseline_metrics):,}; selection={baseline_selection}",
            flush=True,
        )
    if args.stage == "technical":
        technical_panel = build_technical_factor_panel(paths)
        progress(
            f"technical panel rows={len(technical_panel):,}; "
            f"factors={technical_panel['factor_name'].nunique()}",
            flush=True,
        )
        monthly, summary = evaluate_technical_factors(paths, technical_panel)
        progress(
            f"technical monthly rows={len(monthly):,}; summary rows={len(summary):,}",
            flush=True,
        )
        portfolios, metrics, selection = build_technical_combinations(
            paths, technical_panel
        )
        progress(
            f"technical combination rows={len(portfolios):,}; "
            f"metric rows={len(metrics):,}; selection={selection}",
            flush=True,
        )
        del technical_panel, monthly, summary, portfolios, metrics
        expanded_panel = build_expanded_factor_panel(paths)
        evaluate_expanded_factors(paths, expanded_panel)
        del expanded_panel
        full_panel = build_full_factor_panel(paths)
        progress(
            f"full panel rows={len(full_panel):,}; factors={full_panel['factor_name'].nunique()}",
            flush=True,
        )
        monthly, summary = evaluate_full_factors(paths, full_panel)
        progress(
            f"full monthly rows={len(monthly):,}; summary rows={len(summary):,}",
            flush=True,
        )
        portfolios, metrics, selection = build_full_combinations(paths, full_panel)
        progress(
            f"full combination rows={len(portfolios):,}; metric rows={len(metrics):,}; "
            f"selection={selection}",
            flush=True,
        )
    if args.stage in {"combine", "all"}:
        portfolios, metrics, selection = build_v0_combinations(paths, panel)
        progress(
            f"combination rows={len(portfolios):,}; metric rows={len(metrics):,}; "
            f"selection={selection}",
            flush=True,
        )
    if args.stage in {"report", "all"}:
        progress(build_v0_report(paths, args.report.resolve()), flush=True)
    return finish(output=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the non-formal local HS300/CSI500 factor v0")
    parser.add_argument("--qmt-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
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
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    return invoke(run, Config, args)


if __name__ == "__main__":
    raise SystemExit(main())
