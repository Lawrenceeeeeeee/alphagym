from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PAPER_TRADER = Path(__file__).resolve().parent.parent / "qmt" / "paper_trader.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("paper_trader", PAPER_TRADER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def paper_trader():
    return _load_module()


def _write_bundle(tmp_path: Path, *, method: str = "ml_lasso") -> dict[str, object]:
    signal = tmp_path / "signal_latest.csv"
    with signal.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["symbol", "score", "target_weight"])
        writer.writeheader()
        writer.writerow({"symbol": "600000.SH", "score": 1.2, "target_weight": 0.5})
        writer.writerow({"symbol": "920020.BJ", "score": 0.8, "target_weight": 0.5})
    digest = hashlib.sha256(signal.read_bytes()).hexdigest()
    selected = ["AMIHUD_7D", "REVERSAL_20D"]
    selection = {"selected_factors": selected}
    selection_digest = hashlib.sha256(
        json.dumps(
            selection, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    state = {
        "schema_version": 2,
        "report_id": "report-id",
        "run_id": "run-id",
        "method": method,
        "asof": "2026-08-31",
        "signal_date": "2026-08-31",
        "effective_trade_date": "2026-09-01",
        "holding_period": "1M",
        "top_n": 2,
        "selection": selection,
        "factor_manifest": [
            {"factor_id": factor_id, "revision_id": f"rev-{factor_id}", "lookback_days": 20}
            for factor_id in selected
        ],
        "signal_sha256": digest,
        "selection_sha256": selection_digest,
        "stale": False,
        "watermark": "仅模拟盘使用：test",
    }
    (tmp_path / "state.json").write_text(
        json.dumps(state, ensure_ascii=False), encoding="utf-8"
    )
    return state


def _quote(price: float, *, up: float = 11.0, down: float = 9.0) -> dict[str, object]:
    return {
        "lastPrice": price,
        "lastClose": 10.0,
        "askPrice": [price + 0.01],
        "bidPrice": [price - 0.01],
        "limitUp": up,
        "limitDown": down,
    }


def test_load_signal_validates_factor_provenance_and_hash(tmp_path, paper_trader) -> None:
    _write_bundle(tmp_path)
    root, targets, state = paper_trader.load_signal(
        str(tmp_path), expected_report_id="report-id", expected_method="ml_lasso"
    )
    assert root == tmp_path
    assert targets == {"600000.SH": 0.5, "920020.BJ": 0.5}
    assert [item["factor_id"] for item in state["factor_manifest"]] == [
        "AMIHUD_7D",
        "REVERSAL_20D",
    ]

    with (tmp_path / "signal_latest.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(paper_trader.SignalError, match="sha256 mismatch"):
        paper_trader.load_signal(str(tmp_path))


def test_load_signal_rejects_wrong_factor_manifest(tmp_path, paper_trader) -> None:
    state = _write_bundle(tmp_path)
    state["factor_manifest"] = list(reversed(state["factor_manifest"]))
    (tmp_path / "state.json").write_text(
        json.dumps(state, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(paper_trader.SignalError, match="does not match"):
        paper_trader.load_signal(str(tmp_path))


def test_rounding_respects_board_rules_and_odd_lot_liquidation(paper_trader) -> None:
    assert paper_trader.round_buy_quantity("600000.SH", 199) == 100
    assert paper_trader.round_buy_quantity("688121.SH", 199) == 0
    assert paper_trader.round_buy_quantity("688121.SH", 301) == 301
    assert paper_trader.round_buy_quantity("920020.BJ", 151) == 151
    assert paper_trader.legal_sell_quantity("600000.SH", 150, 150, 150) == 150
    assert paper_trader.legal_sell_quantity("600000.SH", 150, 250, 250) == 150
    assert paper_trader.legal_sell_quantity("688121.SH", 301, 301, 301) == 301


def test_plan_orders_accepts_real_quote_shape_and_liquidates_old_names(paper_trader) -> None:
    targets = {"600001.SH": 1.0}
    prices = {"600000.SH": _quote(10.0), "600001.SH": _quote(20.0, up=22, down=18)}
    positions = {"600000.SH": {"volume": 1_000, "sellable": 1_000}}
    orders, skipped = paper_trader.plan_orders(
        targets, prices, positions, equity=10_000, cash=0
    )
    assert ("600000.SH", "sell", 1_000) in orders
    assert any(order[0] == "600001.SH" and order[1] == "buy" for order in orders)
    assert not skipped


def test_plan_orders_reports_t_plus_one_residual(paper_trader) -> None:
    targets: dict[str, float] = {}
    prices = {"600000.SH": _quote(10.0)}
    positions = {"600000.SH": {"volume": 1_000, "sellable": 0}}
    orders, skipped = paper_trader.plan_orders(
        targets, prices, positions, equity=10_000, cash=0
    )
    assert not orders
    assert skipped == [{"symbol": "600000.SH", "reason": "sell_remainder"}]


def test_tick_filter_uses_actual_limits_and_order_book(paper_trader) -> None:
    prices = {"300001.SZ": _quote(12.0, up=12.0, down=8.0)}
    order = ("300001.SZ", "buy", 100)
    assert paper_trader.tick_filter(prices, order, True) == "at_limit_up"
    prices["300001.SZ"]["lastPrice"] = 10.0
    prices["300001.SZ"]["askPrice"] = []
    assert paper_trader.tick_filter(prices, order, True) == "missing_opposite_quote"


def test_query_snapshot_uses_qmt_inner_api_fields(monkeypatch, paper_trader) -> None:
    account = SimpleNamespace(m_dBalance=1_000_000.0, m_dAvailable=123_456.0)
    position = SimpleNamespace(
        m_strInstrumentID="600000",
        m_strExchangeID="SH",
        m_nVolume=1_500,
        m_nCanUseVolume=1_200,
    )

    def get_trade_detail_data(_account_id, _account_type, kind, *_args):
        return {"ACCOUNT": [account], "POSITION": [position]}[kind.upper()]

    monkeypatch.setattr(
        paper_trader, "get_trade_detail_data", get_trade_detail_data, raising=False
    )
    positions, equity, cash = paper_trader.query_snapshot("paper", "STOCK")
    assert positions == {"600000.SH": {"volume": 1_500, "sellable": 1_200}}
    assert equity == 1_000_000
    assert cash == 123_456


def test_submit_order_uses_fixed_price_and_persists_intent(
    tmp_path, monkeypatch, paper_trader
) -> None:
    state = {
        "asof": "2026-08-31",
        "effective_trade_date": "2026-09-01",
        "method": "ml_lasso",
        "report_id": "report-id",
        "run_id": "run-id",
        "signal_sha256": "digest",
    }
    journal = paper_trader._new_journal(state, {"600000.SH": 100})
    calls = []

    def passorder(*args):
        calls.append(args)

    monkeypatch.setattr(paper_trader, "passorder", passorder, raising=False)
    paper_trader.RUNTIME.account_id = "paper-account"
    error = paper_trader._submit_order(
        object(), tmp_path, journal, state, "600000.SH", "buy", 100, _quote(10.0)
    )
    assert error is None
    assert calls[0][:7] == (23, 1101, "paper-account", "600000.SH", 11, 10.01, 100)
    assert calls[0][7] == "AlphaGYMPaper"
    assert calls[0][8] == 2
    persisted = json.loads((tmp_path / "execution_state.json").read_text(encoding="utf-8"))
    assert next(iter(persisted["orders"].values()))["status"] == "submitted"


def test_mark_executed_only_writes_completed_status(tmp_path, paper_trader) -> None:
    state = {
        "asof": "2026-08-31",
        "effective_trade_date": "2026-09-01",
        "method": "ml_lasso",
        "report_id": "report-id",
        "run_id": "run-id",
        "signal_sha256": "digest",
    }
    paper_trader.mark_executed(tmp_path, state, orders={})
    payload = json.loads((tmp_path / "executed.json").read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert paper_trader.last_executed(tmp_path) == "2026-08-31"
    assert paper_trader.execution_completed(tmp_path, state)
