from __future__ import annotations

import html
import json
import re

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mlquant.factor_dsl import FormulaCompiler, FormulaError
from mlquant.factor_operators import build_field_registry, build_operator_registry
from mlquant.factor_research_service import FactorResearchService
from mlquant.factor_store import FactorStore
from mlquant.factor_web import create_app
from mlquant.factors.base import FactorContext, FactorDefinition
from mlquant.factors.library import seed_definitions
from mlquant.report_engine import ReportEngine
from mlquant.report_spec import parse_spec, resolve_factors


def _definition(factor_id: str, formula: str) -> FactorDefinition:
    return FactorDefinition(
        factor_id=factor_id,
        name=factor_id,
        formula=formula,
        hypothesis_id=f"hypothesis_{factor_id.lower()}",
        family="test",
    )


class _SumModel:
    def predict(self, values):
        return np.asarray(values).sum(axis=1)


def test_all_existing_factors_are_one_line_formulas() -> None:
    definitions = seed_definitions()
    assert len(definitions) == 169
    assert len({item.factor_id for item in definitions}) == 169
    assert all(item.formula.startswith("=") and "\n" not in item.formula for item in definitions)


@pytest.mark.parametrize(
    "formula",
    [
        '=__import__("os")',
        "=market.adj_close.__class__",
        "=(lambda: 1)()",
        '=open("secret")',
        "=[value for value in [1]]",
        "=market.adj_close",
        "=TTM(market.close)",
    ],
)
def test_formula_compiler_rejects_python_escape_hatches(formula: str) -> None:
    compiler = FormulaCompiler(build_field_registry(), build_operator_registry())
    with pytest.raises(FormulaError):
        compiler.compile(formula)


def test_store_bootstrap_is_idempotent_and_whitespace_is_not_a_revision() -> None:
    with FactorStore() as store:
        store.bootstrap(seed_definitions())
        store.bootstrap(seed_definitions())
        assert len(store.list_factors()) == 169
        before = store.factor_detail("EP_TTM")["current_revision_id"]
        definition = next(item for item in seed_definitions() if item.factor_id == "EP_TTM")
        store.save_definition(FactorDefinition(
            factor_id=definition.factor_id,
            name=definition.name,
            formula=" = SAFE_DIV( TTM(financial.eps), ASOF(market.close) ) ",
            hypothesis_id=definition.hypothesis_id,
            family=definition.family,
            expected_direction=definition.expected_direction,
        ))
        detail = store.factor_detail("EP_TTM")
        assert detail["current_revision_id"] == before
        assert len(detail["revisions"]) == 1


def test_factor_dependencies_lock_revision_and_reject_cycle() -> None:
    with FactorStore() as store:
        revision_a1 = store.save_definition(_definition("A", "=ASOF(market.close)"))
        store.save_definition(_definition("B", '=FACTOR("A") * 2'))
        store.save_definition(_definition("A", "=RETURN(market.adj_close, 1)"))
        detail_b = store.factor_detail("B")
        assert detail_b["dependencies"][0]["dependency_revision_id"] == revision_a1
        assert detail_b["dependencies"][0]["stale"] == 1
        with pytest.raises(ValueError, match="cyclic"):
            store.save_definition(_definition("A", '=FACTOR("B")'))


def test_locked_dependency_executes_old_revision() -> None:
    daily = pd.DataFrame({
        "trade_date": pd.to_datetime(["2025-01-01", "2025-01-02"] * 2),
        "symbol": ["A", "A", "B", "B"],
        "close": [10.0, 12.0, 20.0, 18.0],
        "adj_close": [10.0, 12.0, 20.0, 18.0],
    })
    with FactorStore() as store:
        store.save_definition(_definition("A", "=ASOF(market.close)"))
        store.save_definition(_definition("B", '=FACTOR("A") * 2'))
        store.save_definition(_definition("A", "=RETURN(market.adj_close, 1)"))
        registry = store.load_registry()
        result = registry.get("B").calculator(
            FactorContext(pd.Timestamp("2025-01-02"), daily, pd.DataFrame())
        )
        assert result.to_dict() == {"A": 24.0, "B": 36.0}


def test_catalog_export_is_deterministic(tmp_path) -> None:
    with FactorStore() as store:
        store.bootstrap(seed_definitions())
        first = store.export_catalog(tmp_path / "first.json")
        second = store.export_catalog(tmp_path / "second.json")
    assert first.read_bytes() == second.read_bytes()
    assert len(json.loads(first.read_text(encoding="utf-8"))["factors"]) == 169


def test_model_formula_pins_artifact_and_blocks_future_training(tmp_path) -> None:
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"model")
    daily = pd.DataFrame({
        "trade_date": pd.to_datetime(["2025-01-02"]),
        "symbol": ["A"], "close": [10.0], "adj_close": [10.0],
    })
    with FactorStore() as store:
        version_id = store.register_model_artifact(
            model_id="earnings", version="1", path=artifact,
            training_snapshot_hash="snapshot", training_end="2025-02-01",
            code_version="test",
        )
        store.save_definition(_definition(
            "ML", '=MODEL_PREDICT(MODEL("earnings"), FEATURES(ASOF(market.close)))'
        ))
        registry = store.load_registry()
        context = FactorContext(
            pd.Timestamp("2025-01-02"), daily, pd.DataFrame(),
            models={version_id: {"estimator": _SumModel(), "training_end": "2025-02-01"}},
        )
        with pytest.raises(ValueError, match="exceeds signal date"):
            registry.get("ML").calculator(context)


def test_smoke_run_cannot_be_promoted() -> None:
    with FactorStore() as store:
        store.save_definition(_definition("A", "=ASOF(market.close)"))
        run_id = store.create_run(["A"], mode="smoke", config={})
        store.set_run_status(run_id, "succeeded", progress=1.0)
        with pytest.raises(ValueError, match="formal"):
            store.promote("A", run_id)


def test_panel_research_run_writes_metrics_and_artifacts(tmp_path) -> None:
    dates = pd.date_range("2024-01-31", periods=6, freq="ME")
    panel = pd.DataFrame([
        {
            "signal_date": date, "symbol": f"{symbol:06d}.SZ", "factor_name": "A",
            "raw_value": float(symbol), "neutralized_value": float(symbol),
            "forward_return": float(symbol) / 100 + date.month / 1000,
        }
        for date in dates for symbol in range(1, 11)
    ])
    panel_path = tmp_path / "panel.parquet"
    panel.to_parquet(panel_path, index=False)
    with FactorStore.from_root(tmp_path) as store:
        store.save_definition(_definition("A", "=ASOF(market.close)"))
        run_id = store.create_run(["A"], mode="smoke", config={})
        output = FactorResearchService(store).execute_panel_run(run_id, panel_path)
        detail = store.run_detail(run_id)
        metrics = store.metrics_for_factor("A")
    assert output.is_dir()
    assert detail["status"] == "succeeded"
    assert {item["kind"] for item in detail["artifacts"]} == {
        "layered_nav", "layered_performance",
        "manifest", "monthly_statistics", "summary",
    }
    assert any(item["metric_name"] == "rank_ic" for item in metrics)
    assert np.isfinite(next(
        item["metric_value"] for item in metrics if item["metric_name"] == "annualized_return"
    ))
    assert any(
        item["period"] == "all" and item["metric_name"] == "annualized_return"
        for item in metrics
    )
    performance_path = next(
        item["path"] for item in detail["artifacts"]
        if item["kind"] == "layered_performance"
    )
    performance = pd.read_parquet(performance_path)
    assert set(performance["portfolio"]) == {
        "group_1", "group_2", "group_3", "group_4", "group_5",
        "benchmark", "long_short",
    }
    assert "all" in set(performance["period"])
    full_sample = performance[
        (performance["period"] == "all")
        & (performance["value_type"] == "raw")
        & (performance["orientation"] == "original")
    ].set_index("portfolio")
    assert full_sample.loc["group_1", "annualized_return"] > full_sample.loc[
        "group_5", "annualized_return"
    ]
    app = create_app(tmp_path)
    with TestClient(app) as client:
        page = client.get("/factors/A")
    assert "五分组组合回测" in page.text
    assert "全样本绩效" in page.text
    assert "样本等权基准" in page.text
    assert "组合1（高因子值）" in page.text
    assert "多空组合（1−5）" in page.text
    assert "echarts-nav" in page.text
    assert "data-series=" in page.text
    match = re.search(r'data-series="([^"]+)"', page.text)
    assert match is not None
    chart_payload = json.loads(html.unescape(match.group(1)))
    assert {series["name"] for series in chart_payload["series"]} == {
        "组合1（高）", "组合2", "组合3", "组合4", "组合5（低）",
        "样本等权基准", "多空（1−5）",
    }
    assert all(
        len(series["dates"]) == len(series["values"])
        for series in chart_payload["series"]
    )


def test_web_catalog_and_health(tmp_path) -> None:
    app = create_app(tmp_path)
    with TestClient(app) as client:
        health = client.get("/health")
        page = client.get("/")
    assert health.json() == {"ok": True, "factor_count": 169}
    assert page.status_code == 200
    assert "公式化因子目录" in page.text


def test_catalog_family_filter_keeps_all_family_choices(tmp_path) -> None:
    app = create_app(tmp_path)
    with TestClient(app) as client:
        page = client.get("/?family=growth")
    assert page.status_code == 200
    assert "成长" in page.text
    assert "技术指标" in page.text
    assert "EPS_YOY" in page.text
    assert "RSI_12D" not in page.text
    assert '<select name="family"' in page.text
    assert 'value="growth" selected' in page.text
    assert "技术指标（98）" in page.text


def test_catalog_backtest_shortcut_preselects_factor(tmp_path) -> None:
    app = create_app(tmp_path)
    with TestClient(app) as client:
        catalog = client.get("/")
        runs = client.get("/runs?factor_id=EPS_YOY")
        blocked = client.post("/runs/quick", data={"factor_id": "EPS_YOY"})
    assert 'action="/runs/quick"' in catalog.text
    assert 'name="factor_id" value="EPS_YOY"' in catalog.text
    assert 'value="EPS_YOY" selected' in runs.text
    assert "已带入所选因子" in runs.text
    assert '<option value="ALL_A">全A股</option>' in runs.text
    assert 'name="mode"' in runs.text
    assert 'value="formal" selected' in runs.text
    assert "正式回测（formal）" in runs.text
    assert blocked.status_code == 422
    assert "A股日线行情尚未入库" in blocked.text
    with FactorStore.from_root(tmp_path) as store:
        assert store.list_runs() == []


def test_web_quick_run_defaults_to_formal_and_spawns_executor(tmp_path, monkeypatch) -> None:
    _write_auto_run_market_data(tmp_path)
    with FactorStore.from_root(tmp_path) as store:
        store.save_definition(_definition("AUTO", "=RETURN(market.adj_close, 5)"))
    spawns: list[list[str]] = []
    monkeypatch.setattr(
        "mlquant.factor_web.subprocess.Popen",
        lambda command, **kwargs: spawns.append(list(command)),
    )
    app = create_app(tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/runs/quick", data={"factor_id": "AUTO"}, follow_redirects=False
        )
    assert response.status_code == 303
    with FactorStore.from_root(tmp_path) as store:
        runs = store.list_runs()
        assert len(runs) == 1
        assert runs[0]["mode"] == "formal"
        run_id = runs[0]["run_id"]
        detail = store.run_detail(run_id)
    assert detail["config"]["point_in_time_audit_passed"] is True
    assert len(spawns) == 1
    assert "--root" in spawns[0] and run_id in spawns[0]


def test_web_formal_run_is_blocked_when_daily_is_missing(tmp_path) -> None:
    with FactorStore.from_root(tmp_path) as store:
        store.save_definition(_definition("AUTO", "=RETURN(market.adj_close, 5)"))
    app = create_app(tmp_path)
    with TestClient(app) as client:
        response = client.post(
            "/runs",
            data={
                "factor_ids": ["AUTO"],
                "mode": "formal",
                "index_code": "ALL_A",
                "start_date": "2023-01-01",
                "end_date": "2024-03-31",
            },
            follow_redirects=False,
        )
    assert response.status_code == 422
    assert "A股日线行情尚未入库" in response.text
    assert "数据根" in response.text
    with FactorStore.from_root(tmp_path) as store:
        assert store.list_runs() == []


def _write_auto_run_market_data(root) -> None:
    dates = pd.bdate_range("2022-12-01", "2024-06-30")
    symbols = [f"{index:06d}.SZ" for index in range(1, 11)]
    rows = []
    for day_number, date in enumerate(dates):
        for symbol_number, symbol in enumerate(symbols, start=1):
            # Deterministic per-symbol noise keeps ranking imperfect so that
            # cross-sectional IC and correlation are not trivially 1.0.
            noise = np.sin(day_number / 3.0 + symbol_number) * 0.04 * symbol_number
            close = 10 + symbol_number + day_number * (0.005 + symbol_number / 10000) + noise
            rows.append({
                "trade_date": date,
                "symbol": symbol,
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000 + symbol_number,
                "amount": close * (1_000_000 + symbol_number),
            })
    equity = root / "equity"
    equity.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(equity / "daily.parquet", index=False)
    pd.DataFrame({
        "trade_date": [dates[0]] * len(symbols),
        "symbol": symbols,
        "adjust_factor": [1.0] * len(symbols),
    }).to_parquet(equity / "adjustments.parquet", index=False)


def test_auto_run_builds_factor_panel_from_registered_market_data(tmp_path) -> None:
    _write_auto_run_market_data(tmp_path)
    with FactorStore.from_root(tmp_path) as store:
        store.save_definition(_definition("AUTO", "=RETURN(market.adj_close, 5)"))
        run_id = store.create_run(
            ["AUTO"],
            mode="smoke",
            config={
                "source": "automatic",
                "index_code": "ALL_A",
                "start_date": "2023-01-01",
                "end_date": "2024-03-31",
                "point_in_time_audit_passed": False,
            },
        )
        output = FactorResearchService(store).execute_auto_run(run_id)
        detail = store.run_detail(run_id)
    assert output.is_dir()
    assert detail["status"] == "succeeded"
    assert {item["kind"] for item in detail["artifacts"]} == {
        "factor_values", "layered_nav", "layered_performance",
        "manifest", "monthly_statistics", "summary",
    }


def test_auto_run_technical_factor_loads_base_ohlcv(tmp_path) -> None:
    _write_auto_run_market_data(tmp_path)
    with FactorStore.from_root(tmp_path) as store:
        store.save_definition(_definition("BIAS_12D", "=BIAS(market.adj_close, 12)"))
        run_id = store.create_run(
            ["BIAS_12D"],
            mode="formal",
            config={
                "source": "automatic",
                "index_code": "ALL_A",
                "start_date": "2023-01-01",
                "end_date": "2024-03-31",
                "point_in_time_audit_passed": True,
            },
        )
        FactorResearchService(store).execute_auto_run(run_id)
        detail = store.run_detail(run_id)
    assert detail["status"] == "succeeded"


def test_report_engine_builds_static_report(tmp_path) -> None:
    _write_auto_run_market_data(tmp_path)
    definitions = [
        FactorDefinition(
            factor_id="BOARD_A", name="BOARD_A", formula="=RETURN(market.adj_close, 5)",
            hypothesis_id="hypothesis_board_a", family="momentum",
            expected_direction="positive",
        ),
        FactorDefinition(
            factor_id="BOARD_B", name="BOARD_B",
            formula="=VOLATILITY(market.adj_close, 20)",
            hypothesis_id="hypothesis_board_b", family="momentum",
            expected_direction="unknown",
        ),
    ]
    with FactorStore.from_root(tmp_path) as store:
        for definition in definitions:
            store.save_definition(definition)
        run_id = store.create_run(
            ["BOARD_A", "BOARD_B"],
            mode="smoke",
            config={
                "source": "automatic",
                "index_code": "ALL_A",
                "start_date": "2023-01-01",
                "end_date": "2024-03-31",
                "point_in_time_audit_passed": False,
            },
        )
        FactorResearchService(store).execute_auto_run(run_id)
        spec = parse_spec({
            "name": "看板替代报告", "mode": "smoke",
            "window": {"start": "2023-01-01", "end": "2024-03-31"},
            "splits": {
                "development": ["2023-01-01", "2023-06-30"],
                "validation": ["2023-07-01", "2023-12-31"],
                "test": ["2024-01-01", "2024-03-31"],
            },
            "factors": {"families": ["momentum"]},
            "combine": {"methods": ["equal", "ic_decay"], "correlation_threshold": 0.99},
        })
        resolved = resolve_factors(store, spec.factors)
        assert {item["factor_id"] for item in resolved} == {"BOARD_A", "BOARD_B"}
        report_id = store.create_report(spec.name, spec.to_dict())
        output = ReportEngine(store).execute_report(report_id)
        row = store.report_detail(report_id)
    assert row["status"] == "succeeded"
    assert output.is_dir()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert {item["factor_id"] for item in manifest["factors"]} == {"BOARD_A", "BOARD_B"}
    assert all(item["lookback_days"] in (5, 20) for item in manifest["factors"])
    assert manifest["spec"]["holding_period"] == "1M"
    for name in (
        "report.md", "report.html", "monthly.parquet", "summary.csv",
        "correlation.parquet", "combo.json", "spec.yaml",
    ):
        assert (output / name).is_file()
    markdown = (output / "report.md").read_text(encoding="utf-8")
    assert "持有期：1M" in markdown
    assert "回看(日)" in markdown
    assert "测试与监控期仅评估" in markdown
    assert "测试期不调方向与参数" in markdown
    html = (output / "report.html").read_text(encoding="utf-8")
    assert "研究设置" in html
    assert "echarts" in html
    assert "data-kind=\"correlation\"" in html
    assert "data-kind=\"combination\"" in html
    combo = json.loads((output / "combo.json").read_text(encoding="utf-8"))
    assert set(combo["nav"]) == {"equal", "ic_decay"}
    assert combo["selection"]["selection_end"] == "2023-12-31"
    app = create_app(tmp_path)
    with TestClient(app) as client:
        page = client.get("/factors/BOARD_A")
    assert page.status_code == 200
    assert "echarts-nav" in page.text
    assert "data-series=" in page.text


def test_report_pages_render_and_reject_bad_splits(tmp_path) -> None:
    app = create_app(tmp_path)
    with TestClient(app) as client:
        list_page = client.get("/reports")
        new_page = client.get("/reports/new")
        bad = client.post("/reports", data={
            "name": "分段缺口报告", "mode": "smoke", "holding_period": "1M",
            "index_code": "ALL_A",
            "start_date": "2014-01-01", "end_date": "2025-12-31",
            "dev_start": "2014-01-01", "dev_end": "2020-12-31",
            "val_start": "2022-01-01", "val_end": "2023-12-31",
            "test_start": "2024-01-01", "test_end": "2025-12-31",
            "families": ["momentum"],
        })
    assert list_page.status_code == 200
    assert "报告区" in list_page.text
    assert new_page.status_code == 200
    assert "新建报告" in new_page.text
    assert bad.status_code == 422
    assert "首尾相接" in bad.text


def test_auto_run_reports_missing_formula_data(tmp_path) -> None:
    _write_auto_run_market_data(tmp_path)
    with FactorStore.from_root(tmp_path) as store:
        store.save_definition(_definition("FIN", "=YOY(financial.eps)"))
        with pytest.raises(ValueError, match="financial.eps"):
            FactorResearchService(store).validate_auto_request(
                ["FIN"],
                mode="smoke",
                index_code="ALL_A",
                start_date="2023-01-01",
                end_date="2024-03-31",
            )


def test_formal_all_a_is_a_valid_universe_choice(tmp_path) -> None:
    _write_auto_run_market_data(tmp_path)
    with FactorStore.from_root(tmp_path) as store:
        store.save_definition(_definition("AUTO", "=RETURN(market.adj_close, 5)"))
        result = FactorResearchService(store).validate_auto_request(
            ["AUTO"],
            mode="formal",
            index_code="ALL_A",
            start_date="2023-01-01",
            end_date="2024-03-31",
        )
    assert result["fields"] == ["market.adj_close"]
    assert result["data_root"] == str(tmp_path)


def test_factor_metrics_are_grouped_by_research_period(tmp_path) -> None:
    with FactorStore.from_root(tmp_path) as store:
        store.save_definition(_definition("A", "=ASOF(market.close)"))
        run_id = store.create_run(["A"], mode="smoke", config={})
        store.write_metrics(run_id, [
            {
                "factor_id": "A", "index_code": "ALL_A", "period": "development",
                "value_type": "raw", "orientation": "original",
                "metric_name": "rank_ic", "metric_value": 0.031,
            },
            {
                "factor_id": "A", "index_code": "ALL_A", "period": "validation",
                "value_type": "raw", "orientation": "original",
                "metric_name": "rank_ic", "metric_value": 0.018,
            },
            {
                "factor_id": "A", "index_code": "ALL_A", "period": "test",
                "value_type": "raw", "orientation": "original",
                "metric_name": "coverage", "metric_value": 0.92,
            },
        ])
    app = create_app(tmp_path)
    with TestClient(app) as client:
        page = client.get("/factors/A")
    research = page.text.split('id="research-metrics"', 1)[1]
    assert "开发期" in research
    assert "验证期" in research
    assert "测试期" in research
    assert "Rank IC" in research
    assert "92.00%" in research
    assert "metric-period-card" in research
    assert "<table>" not in research
