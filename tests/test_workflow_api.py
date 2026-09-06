from __future__ import annotations

import importlib
import json
import subprocess
import sys

import pytest

from mlquant import Workspace, storage_io
from mlquant.cli import main
from mlquant.workflows.catalog import WORKFLOWS, describe_workflow, list_workflows, run_workflow


def test_workflow_discovery_covers_installed_configs(capsys):
    assert len(list_workflows()) == 9
    for name, (module_name, _) in WORKFLOWS.items():
        description = describe_workflow(name)
        assert description["parameters"]
        module = importlib.import_module(f"mlquant.workflows.{module_name}")
        with pytest.raises(SystemExit) as raised:
            module.main(["--help"])
        assert raised.value.code == 0
    capsys.readouterr()


def test_workflow_dispatch_rejects_unknown_or_missing_config():
    with pytest.raises(ValueError, match="Unknown workflow"):
        run_workflow("os.system", {})
    with pytest.raises(ValueError, match="Invalid.*config"):
        run_workflow("topn-backtest", {})


@pytest.mark.parametrize("override", [
    {"frequency": ["monthly"]}, {"frequency": "daily"}, {"top_n": 0},
    {"top_n": True}, {"report_id": None}, {"root": ""},
])
def test_workflow_config_fails_before_io(tmp_path, override):
    config = {"root": tmp_path / "absent", "report_id": "missing",
              "output": tmp_path / "output", **override}
    with pytest.raises(ValueError):
        run_workflow("frequency-evaluation", config)
    assert not storage_io.exists(tmp_path / "output")


def test_workflow_failure_is_exception_not_system_exit(tmp_path, capsys):
    workspace = Workspace(tmp_path)
    workspace.initialize()
    with pytest.raises(KeyError):
        run_workflow("topn-backtest", {"root": tmp_path, "report_id": "missing"})
    assert not capsys.readouterr().out


def test_workflow_cli_domain_error_is_structured(tmp_path, capsys):
    config = tmp_path / "config.yaml"
    storage_io.write_text(config, "root: .\nreport_id: missing\ntop_n: -1\n", encoding="utf-8")
    assert main(["workflow", "run", "topn-backtest", "--config", str(config), "--json"]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert json.loads(output.err)["error"]["code"] == "ValueError"


def test_workflow_modules_do_not_load_optional_models(tmp_path):
    code = (
        "import sys; from mlquant.workflows.catalog import WORKFLOWS, describe_workflow; "
        "[describe_workflow(name) for name in WORKFLOWS]; "
        "assert not {'sklearn','xgboost','lightgbm','matplotlib','reportlab','fastapi'} & set(sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_batch_failure_is_not_reported_as_success(tmp_path, monkeypatch):
    from mlquant import FactorDefinition
    from mlquant.workflows import backfill_factor_backtests as workflow

    workspace = Workspace(tmp_path)
    workspace.initialize()
    workspace.save_factor(FactorDefinition(
        factor_id="BATCH_TEST", name="BATCH_TEST", formula="=RETURN(market.adj_close, 5)",
        hypothesis_id="batch_test", family="momentum",
    ))
    monkeypatch.setattr(workflow, "_collect_todo", lambda *args: (["BATCH_TEST"], []))

    def fail(self, run_id):
        self.store.set_run_status(run_id, "failed", error="synthetic failure")
        raise ValueError("synthetic failure")

    monkeypatch.setattr(workflow.FactorResearchService, "execute_auto_run", fail)
    result = workflow.run(workflow.Config(root=tmp_path, factors="BATCH_TEST", no_reports=True))
    assert result["ok"] is False
    assert result["error"]["code"] == "WorkflowFailed"
    assert "synthetic failure" in result["results"][0]["message"]
