"""Unit tests for report spec parsing, validation and factor resolution."""
from __future__ import annotations

import pytest

from mlquant.factor_store import FactorStore
from mlquant.factors.base import FactorDefinition
from mlquant.report_spec import (
    FactorSelection,
    MetricFilter,
    load_spec,
    parse_spec,
    resolve_factors,
)


def _definition(factor_id: str, family: str = "momentum") -> FactorDefinition:
    return FactorDefinition(
        factor_id=factor_id, name=factor_id, formula="=RETURN(market.adj_close, 5)",
        hypothesis_id=f"hypothesis_{factor_id.lower()}", family=family,
    )


def _valid_payload() -> dict:
    return {
        "name": "测试报告", "mode": "formal", "holding_period": "1M",
        "window": {"start": "2014-01-01", "end": "2025-12-31"},
        "splits": {
            "development": ["2014-01-01", "2020-12-31"],
            "validation": ["2021-01-01", "2023-12-31"],
            "test": ["2024-01-01", "2025-12-31"],
        },
        "factors": {"families": ["momentum"]},
    }


def test_parse_valid_spec_roundtrips() -> None:
    spec = parse_spec(_valid_payload())
    assert spec.name == "测试报告"
    assert spec.holding_period == "1M"
    assert spec.resolved_splits()["test"][1].year == 2025
    assert parse_spec(spec.to_dict()).to_dict() == spec.to_dict()


def test_parse_requires_name_and_mode() -> None:
    with pytest.raises(ValueError, match="name"):
        parse_spec({})
    payload = _valid_payload()
    payload["mode"] = "production"
    with pytest.raises(ValueError, match="mode"):
        parse_spec(payload)


def test_split_chain_must_be_contiguous_and_cover_window() -> None:
    payload = _valid_payload()
    payload["splits"]["validation"] = ["2022-01-01", "2023-12-31"]
    with pytest.raises(ValueError, match="首尾相接"):
        parse_spec(payload)
    payload = _valid_payload()
    payload["splits"]["development"] = ["2014-02-01", "2020-12-31"]
    with pytest.raises(ValueError, match="window.start"):
        parse_spec(payload)
    payload = _valid_payload()
    del payload["splits"]["test"]
    with pytest.raises(ValueError, match="缺少阶段"):
        parse_spec(payload)


def test_holding_period_only_monthly() -> None:
    payload = _valid_payload()
    payload["holding_period"] = "3M"
    with pytest.raises(ValueError, match="holding_period"):
        parse_spec(payload)


def test_combine_selection_cannot_include_test() -> None:
    payload = _valid_payload()
    payload["combine"] = {"methods": ["equal"], "selection_periods": ["validation", "test"]}
    with pytest.raises(ValueError, match="不能包含 test"):
        parse_spec(payload)
    payload["combine"]["methods"] = ["median"]
    with pytest.raises(ValueError, match="combine.methods"):
        parse_spec(payload)


def test_factor_selection_needs_a_source() -> None:
    payload = _valid_payload()
    payload["factors"] = {"exclude": ["X"]}
    with pytest.raises(ValueError, match="families/include/metrics"):
        parse_spec(payload)


def test_load_spec_reads_yaml(tmp_path) -> None:
    path = tmp_path / "report.yaml"
    path.write_text(
        "name: 文件报告\nmode: smoke\nwindow: {start: 2020-01-01, end: 2025-12-31}\n"
        "splits:\n  development: [2020-01-01, 2021-12-31]\n"
        "  validation: [2022-01-01, 2023-12-31]\n"
        "  test: [2024-01-01, 2025-12-31]\n"
        "factors: {include: [A, B]}\n",
        encoding="utf-8",
    )
    spec = load_spec(path)
    assert spec.name == "文件报告"
    assert spec.mode == "smoke"


def test_resolve_factors_families_include_exclude(tmp_path) -> None:
    with FactorStore.from_root(tmp_path) as store:
        for factor_id, family in (("MOM_A", "momentum"), ("MOM_B", "momentum"),
                                  ("VAL_A", "value")):
            store.save_definition(_definition(factor_id, family))
        selection = FactorSelection(
            families=["momentum"], include=["VAL_A"], exclude=["MOM_B"],
        )
        resolved = resolve_factors(store, selection)
    assert {item["factor_id"] for item in resolved} == {"MOM_A", "VAL_A"}


def test_resolve_factors_rejects_unknown_ids(tmp_path) -> None:
    with FactorStore.from_root(tmp_path) as store:
        store.save_definition(_definition("MOM_A"))
        with pytest.raises(ValueError, match="未知因子"):
            resolve_factors(store, FactorSelection(include=["GHOST"]))


def test_resolve_factors_metric_filter_without_runs_drops_all(tmp_path) -> None:
    with FactorStore.from_root(tmp_path) as store:
        store.save_definition(_definition("MOM_A"))
        selection = FactorSelection(
            families=["momentum"],
            metrics=[MetricFilter(
                metric="rank_ic", period="validation", op=">=", value=0.0,
            )],
        )
        resolved = resolve_factors(store, selection)
    assert resolved == []


def test_resolve_factors_include_only_restricts_to_listed(tmp_path) -> None:
    """Explicit include must not silently expand to the whole library."""
    with FactorStore.from_root(tmp_path) as store:
        store.save_definition(_definition("MOM_A"))
        store.save_definition(_definition("MOM_B"))
        store.save_definition(_definition("VAL_A", family="value"))
        resolved = resolve_factors(store, FactorSelection(include=["MOM_B"]))
    assert [item["factor_id"] for item in resolved] == ["MOM_B"]


def test_resolve_factors_rejects_selection_on_test_or_all(tmp_path) -> None:
    with FactorStore.from_root(tmp_path) as store:
        store.save_definition(_definition("MOM_A"))
        for period in ("test", "all"):
            with pytest.raises(ValueError, match="development/validation"):
                resolve_factors(store, FactorSelection(
                    families=["momentum"],
                    sort_by=MetricFilter(metric="rank_ic", period=period),
                    top_n=1,
                ))
            with pytest.raises(ValueError, match="development/validation"):
                resolve_factors(store, FactorSelection(
                    families=["momentum"],
                    metrics=[MetricFilter(
                        metric="rank_ic", period=period, op=">=", value=0.0,
                    )],
                ))


def test_parse_rejects_selection_periods_with_test_or_all() -> None:
    payload = _valid_payload()
    payload["factors"] = {
        "families": ["momentum"],
        "metrics": [{"metric": "rank_ic", "period": "all", "op": ">=", "value": 0}],
    }
    with pytest.raises(ValueError, match="development/validation"):
        parse_spec(payload)
    payload["factors"] = {
        "families": ["momentum"],
        "sort_by": {"metric": "rank_ic", "period": "test"},
        "top_n": 5,
    }
    with pytest.raises(ValueError, match="development/validation"):
        parse_spec(payload)


def test_parse_rejects_industry_and_symbol_overlap() -> None:
    payload = _valid_payload()
    payload["universe"] = {
        "industries": {"include": ["801010"], "exclude": ["801010"]},
    }
    with pytest.raises(ValueError, match="交集"):
        parse_spec(payload)
    payload = _valid_payload()
    payload["universe"] = {
        "symbols": {"include": ["600000.SH"], "exclude": ["600000.SH"]},
    }
    with pytest.raises(ValueError, match="交集"):
        parse_spec(payload)


def test_parse_allows_validation_period_selection() -> None:
    payload = _valid_payload()
    payload["factors"] = {
        "families": ["momentum"],
        "metrics": [{"metric": "rank_ic", "period": "validation", "op": ">=", "value": 0}],
        "sort_by": {"metric": "rank_ic", "period": "development"},
        "top_n": 5,
    }
    spec = parse_spec(payload)
    assert spec.factors.metrics[0].period == "validation"
    assert spec.factors.sort_by.period == "development"
