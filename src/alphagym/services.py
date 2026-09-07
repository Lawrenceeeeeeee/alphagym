"""Application operations shared by Python, CLI and Web adapters."""
from __future__ import annotations

from typing import Any

from alphagym.factor_store import FactorStore
from alphagym.report_spec import ReportSpec, parse_spec, resolve_factors


def create_report(
    store: FactorStore, spec: ReportSpec, *, ensure_runs: bool = False,
) -> dict[str, Any]:
    # Dataclass construction is convenient for callers, but must not bypass validation.
    spec = parse_spec(spec.to_dict())
    resolved = resolve_factors(store, spec.factors, index_code=spec.universe.index_code)
    minimum = 1 if spec.combine is None else 2
    if len(resolved) < minimum:
        raise ValueError(f"报告至少需要 {minimum} 个因子")
    if ensure_runs:
        from alphagym.report_engine import ReportEngine

        ReportEngine(store).backfill_runs(spec, resolved)
    report_id = store.create_report(spec.name, spec.to_dict())
    return {
        "report_id": report_id, "status": "queued", "name": spec.name,
        "factors": len(resolved),
    }
