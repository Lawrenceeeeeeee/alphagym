from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from alphagym import (
    DataContractError,
    EquityDataBundle,
    FactorDefinition,
    ReportSpec,
    Workspace,
    storage_io,
)
from alphagym.cli import build_parser, main
from alphagym.factor_store import FactorStore
from alphagym.report_spec import FactorSelection
from alphagym.serialization import dumps


def test_inspection_does_not_create_workspace(tmp_path):
    root = tmp_path / "absent"
    workspace = Workspace(root)
    assert workspace.status()["factor_library"]["factors"] == 0
    assert workspace.list_reports() == workspace.list_runs() == []
    assert not storage_io.exists(root)


def test_root_is_explicit_and_environment_is_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("ALPHAGYM_DATA_ROOT", raising=False)
    with pytest.raises(DataContractError):
        Workspace()
    with pytest.raises(DataContractError):
        EquityDataBundle.from_root()
    monkeypatch.setenv("ALPHAGYM_DATA_ROOT", str(tmp_path / "env"))
    assert Workspace().root == tmp_path / "env"
    assert Workspace(tmp_path).root == tmp_path


def test_catalog_lifecycle_and_readonly_connection(tmp_path):
    workspace = Workspace(tmp_path)
    with pytest.raises(FileNotFoundError):
        workspace.list_factors()
    workspace.initialize()
    workspace.initialize()
    assert len(workspace.list_factors()) == 169
    with (
        FactorStore(workspace.config.catalog, readonly=True) as store,
        pytest.raises(sqlite3.OperationalError, match="readonly"),
    ):
        store.connection.execute("DELETE FROM factor")
    assert workspace.status()["factor_library"]["factors"] == 169


def test_specs_are_validated_and_reads_do_not_execute(tmp_path):
    workspace = Workspace(tmp_path)
    workspace.initialize()
    spec = ReportSpec(name="API", mode="smoke", factors=FactorSelection(include=["ROC_20D"]))
    spec.holding_period = "3M"
    with pytest.raises(ValueError, match="holding_period"):
        workspace.create_report(spec)
    assert workspace.list_reports() == []
    workspace.save_factor(FactorDefinition(
        factor_id="API_TEST", name="API_TEST", formula="=RETURN(market.adj_close, 5)",
        family="momentum", hypothesis_id="api_test",
    ))
    spec.holding_period = "1M"
    spec.factors = FactorSelection(include=["API_TEST"])
    task = workspace.create_report(spec)
    assert workspace.report(task["report_id"])["status"] == "queued"
    with pytest.raises(TimeoutError):
        workspace.wait_report(task["report_id"], timeout=0)
    with pytest.raises(FileNotFoundError):
        workspace.report_manifest(task["report_id"])
    with FactorStore.from_root(tmp_path) as store:
        store.claim_report(task["report_id"])
        with pytest.raises(ValueError, match="not queued"):
            store.claim_report(task["report_id"])


@pytest.mark.parametrize("timeout,poll", [(-1, 1), (float("nan"), 1), (1, 0), (1, float("inf"))])
def test_wait_rejects_invalid_bounds(tmp_path, timeout, poll):
    with pytest.raises(ValueError):
        Workspace(tmp_path).wait_report("missing", timeout=timeout, poll_interval=poll)


def test_strict_json_represents_missing_values():
    assert json.loads(dumps({"values": [np.nan, np.inf, pd.NA, pd.NaT, np.int64(3)]})) == {
        "values": [None, None, None, None, 3],
    }


def test_cli_domain_error_has_json_and_exit_one(monkeypatch, capsys):
    monkeypatch.delenv("ALPHAGYM_DATA_ROOT", raising=False)
    assert main(["status", "--json"]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert json.loads(output.err)["error"]["code"] == "DataContractError"


def test_cli_status_is_readonly_and_json_flag_is_position_independent(tmp_path, capsys):
    root = tmp_path / "absent"
    assert main(["--json", "status", "--root", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["equity"] == {}
    assert not storage_io.exists(root)


def test_all_leaf_commands_accept_json():
    import argparse

    def visit(parser):
        assert any(action.dest == "json" for action in parser._actions)
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for child in action.choices.values():
                    visit(child)
    visit(build_parser())


def test_failed_wait_produces_a_domain_error(tmp_path, capsys):
    workspace = Workspace(tmp_path)
    workspace.initialize()
    with FactorStore.from_root(tmp_path) as store:
        report_id = store.create_report("Failed", {})
        store.set_report_status(report_id, "failed", error="synthetic failure")
    assert main(["report", "wait", "--root", str(tmp_path), "--report-id", report_id,
                 "--timeout", "0", "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["error"] == {"code": "TaskFailed", "message": "synthetic failure"}


def test_imports_do_not_load_optional_dependencies(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", ("import sys; from alphagym import Workspace; "
         "from alphagym.cli import build_parser; build_parser(); "
         "assert not {'sklearn','xgboost','lightgbm','fastapi','matplotlib','reportlab'} "
         "& set(sys.modules)")],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
