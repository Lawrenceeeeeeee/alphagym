"""Shared configuration validation and command adapters for research workflows."""
from __future__ import annotations

import dataclasses
import json
import logging
import math
import sqlite3
import sys
import types
from datetime import date
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

import yaml

from mlquant.api import Workspace
from mlquant.config import resolve_root
from mlquant.serialization import dumps


def progress(*values, sep=" ", end="\n", flush=False, file=None) -> None:
    logging.getLogger("mlquant.workflows").info(sep.join(map(str, values)))


def validate_config(config, choices: dict) -> None:
    hints = get_type_hints(type(config))
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        if value is None:
            if not _matches(value, hints[field.name]):
                raise ValueError(f"{field.name} cannot be null")
            continue
        if field.name == "root":
            object.__setattr__(config, field.name, resolve_root(value))
        elif field.name in {"output", "pool", "features_from", "qmt_root", "source_root", "report"}:
            object.__setattr__(config, field.name, Path(value).expanduser().resolve())
        elif isinstance(value, date) and field.name.endswith("_date"):
            object.__setattr__(config, field.name, value.isoformat())
        value = getattr(config, field.name)
        if not _matches(value, hints[field.name]):
            raise ValueError(f"Invalid type for {field.name}: expected {hints[field.name]}")
        if field.name in choices:
            selected = value if isinstance(value, list) else [value]
            if any(item not in choices[field.name] for item in selected):
                raise ValueError(f"Invalid {field.name}: {value}; expected {choices[field.name]}")
        if (field.name in {"top_n", "train_row_cap", "workers", "batch_size", "refit_months",
                           "smooth_months", "initial_cash", "report_top_n"}
                and (not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0)):
            raise ValueError(f"{field.name} must be finite and positive")


def _matches(value, annotation) -> bool:
    origin = get_origin(annotation)
    if origin is types.UnionType:
        return any(_matches(value, item) for item in get_args(annotation))
    if origin is list:
        return isinstance(value, list) and all(_matches(item, get_args(annotation)[0]) for item in value)
    if annotation is float:
        return type(value) in (int, float)
    if annotation in (int, bool):
        return type(value) is annotation
    return isinstance(value, annotation)


def finish(code: int = 0, *, output: Path | None = None, results=None) -> dict:
    result = {"ok": code == 0}
    if output is not None:
        result["path"] = str(output)
    if results is not None:
        result["results"] = results
    if code:
        result["error"] = {"code": "WorkflowFailed", "message": "One or more workflow steps failed"}
    return result


def report_record(root: Path, report_id: str) -> dict:
    row = Workspace(root).report(report_id)
    if row["status"] != "succeeded" or not row.get("path"):
        raise ValueError(f"Report must have completed artifacts: {report_id}")
    row["spec_json"] = json.dumps(row["spec"])
    return row


def invoke(run, config_type, args) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")
    try:
        result = run(config_type(**vars(args)))
        print(dumps(result))
        return 0 if result["ok"] else 1
    except (ValueError, KeyError, OSError, ImportError, sqlite3.Error, yaml.YAMLError) as error:
        print(dumps({"ok": False, "error": {
            "code": type(error).__name__, "message": str(error),
        }}), file=sys.stderr)
        return 1
