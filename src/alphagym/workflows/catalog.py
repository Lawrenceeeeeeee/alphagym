"""Allowlisted workflow discovery and Python dispatch for agents."""
from __future__ import annotations

from dataclasses import MISSING, fields
from importlib import import_module
from typing import Any, get_type_hints

from alphagym.serialization import json_value

WORKFLOWS = {
    "ml-exploration": ("run_ml_exploration", "Stock-level ML comparison with frozen selection"),
    "portfolio-evaluation": ("evaluate_ml_portfolios", "Cost-aware monthly long-only evaluation"),
    "frequency-evaluation": ("evaluate_ml_frequency", "PIT weekly/daily portfolio evaluation"),
    "exploratory-evaluation": ("evaluate_ml_exploratory", "Non-formal, non-neutralized comparison"),
    "exploratory-report": ("report_ml_exploratory", "Static report from exploratory artifacts"),
    "topn-backtest": ("backtest_topn", "Top-N portfolio diagnostics from a completed report"),
    "index-backfill": ("backfill_index_pool", "Backfill explicit factor pool across indices"),
    "factor-backfill": ("backfill_factor_backtests", "Batch factor backtests and family reports"),
    "equity-v0": ("run_equity_v0", "Legacy explicitly non-formal research workflow"),
}


def _module(name: str):
    if name not in WORKFLOWS:
        raise ValueError(f"Unknown workflow: {name}; available: {', '.join(WORKFLOWS)}")
    return import_module(f"alphagym.workflows.{WORKFLOWS[name][0]}")


def list_workflows() -> list[dict[str, str]]:
    return [{"name": name, "description": value[1]} for name, value in WORKFLOWS.items()]


def describe_workflow(name: str) -> dict[str, Any]:
    config = _module(name).Config
    hints = get_type_hints(config)
    parameters = []
    for field in fields(config):
        if field.name == "json":
            continue
        hint = hints[field.name]
        type_name = hint.__name__ if isinstance(hint, type) else str(hint)
        entry = {"name": field.name, "type": type_name,
                 "required": field.default is MISSING and field.default_factory is MISSING}
        if field.default is not MISSING:
            entry["default"] = json_value(field.default)
        parameters.append(entry)
    return {"name": name, "description": WORKFLOWS[name][1], "parameters": parameters}


def run_workflow(name: str, config: dict[str, Any]) -> dict[str, Any]:
    """Run a registered workflow with explicit configuration, without parsing argv."""
    module = _module(name)
    if not isinstance(config, dict):
        raise TypeError("Workflow config must be a mapping")
    try:
        settings = module.Config(**config)
    except TypeError as error:
        raise ValueError(f"Invalid {name} config: {error}") from error
    return module.run(settings)
