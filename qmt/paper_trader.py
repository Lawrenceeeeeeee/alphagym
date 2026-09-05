# coding: gbk
"""MLQuant monthly paper-trading executor for QMT built-in Python.

This file targets the QMT strategy editor (innerApi), not MiniQMT/nativeApi.
QMT calls ``init(ContextInfo)`` once and ``handlebar(ContextInfo)`` on market
events. The terminal never computes factors; it loads a frozen signal bundle
produced by ``mlquant signal export`` and verifies its factor provenance.

The source is ASCII-only so it can live in the Python 3.12 repository while
also being pasted into QMT's Python 3.6/GBK editor.
"""

import csv
import hashlib
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

# User configuration ---------------------------------------------------------

# Configure explicitly for a locally validated signal bundle before use.
SIGNAL_DIR = r"C:\path\to\local\signals"
EXPECTED_REPORT_ID = "REPLACE_WITH_LOCAL_REPORT_ID"
EXPECTED_METHOD = "REPLACE_WITH_LOCAL_METHOD"

# Leave blank to use the account selected in QMT's model-trading UI.
ACCOUNT_ID_OVERRIDE = ""
ACCOUNT_TYPE_OVERRIDE = "STOCK"

# Safe defaults. Both switches must be changed deliberately before orders can
# be sent. This script is intended for a simulated account only.
DRY_RUN = True
SIMULATION_ACCOUNT_CONFIRMED = False

TRADE_START = "093000"
TRADE_END = "145000"
CASH_BUFFER_RATE = 0.005
SUBMISSION_GRACE_SECONDS = 30
POSITION_SYNC_GRACE_SECONDS = 10
STRATEGY_NAME = "MLQuantPaper"


# Stable contracts and QMT constants ----------------------------------------

SIGNAL_SCHEMA_VERSION = 2
EXECUTION_SCHEMA_VERSION = 1
SIMULATION_WATERMARK_PREFIX = "\u4ec5\u6a21\u62df\u76d8\u4f7f\u7528"
SYMBOL_PATTERN = re.compile(r"^[0-9]{6}\.(SH|SZ|BJ)$")

STOCK_BUY = 23
STOCK_SELL = 24
ORDER_BY_SHARES = 1101
FIX_PRICE = 11
QUICK_TRADE_IMMEDIATE = 2

ORDER_ACTIVE = {48, 49, 50, 51, 52, 55}
ORDER_FILLED = 56
ORDER_TERMINAL = {53, 54, 56, 57}


class SignalError(ValueError):
    pass


class _RuntimeState:
    def __init__(self):
        self.ready = False
        self.error = None
        self.root = None
        self.targets = None
        self.state = None
        self.account_id = None
        self.account_type = None
        self.previewed = False
        self.last_message = None


RUNTIME = _RuntimeState()


# Signal import and provenance validation -----------------------------------

def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _parse_date(value, field):
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()  # noqa: DTZ007
    except (TypeError, ValueError):
        raise SignalError(f"invalid {field}: {value!r}")


def _normalize_symbol(symbol):
    value = str(symbol or "").strip().upper()
    if not SYMBOL_PATTERN.match(value):
        raise SignalError(f"invalid A-share symbol: {symbol!r}")
    return value


def _validate_factor_manifest(state):
    selection = state.get("selection")
    if not isinstance(selection, dict):
        raise SignalError("state.selection must be an object")
    selected = selection.get("selected_factors")
    if not isinstance(selected, list) or not selected:
        raise SignalError("state.selection.selected_factors is empty")
    selected = [str(item) for item in selected]
    if len(selected) != len(set(selected)):
        raise SignalError("selected_factors contains duplicates")

    manifest = state.get("factor_manifest")
    if not isinstance(manifest, list) or not manifest:
        raise SignalError("state.factor_manifest is empty")
    factor_ids = []
    for item in manifest:
        if not isinstance(item, dict):
            raise SignalError("invalid factor_manifest item")
        factor_id = str(item.get("factor_id") or "")
        revision_id = str(item.get("revision_id") or "")
        lookback = item.get("lookback_days")
        if not factor_id or not revision_id:
            raise SignalError("factor_manifest requires factor_id and revision_id")
        if not isinstance(lookback, int) or lookback < 0:
            raise SignalError(f"invalid lookback_days for {factor_id}")
        factor_ids.append(factor_id)
    if factor_ids != selected:
        raise SignalError("factor_manifest does not match selected_factors")


def validate_signal_bundle(root, targets, state, expected_report_id=None, expected_method=None):
    if int(state.get("schema_version") or 0) != SIGNAL_SCHEMA_VERSION:
        raise SignalError(f"signal schema_version must be {SIGNAL_SCHEMA_VERSION}")
    required = [
        "report_id",
        "run_id",
        "method",
        "signal_date",
        "effective_trade_date",
        "holding_period",
        "selection_sha256",
        "signal_sha256",
        "watermark",
    ]
    missing = [key for key in required if not state.get(key)]
    if missing:
        raise SignalError("state is missing fields: {}".format(", ".join(missing)))
    if expected_report_id and state["report_id"] != expected_report_id:
        raise SignalError("unexpected report_id: {}".format(state["report_id"]))
    if expected_method and state["method"] != expected_method:
        raise SignalError("unexpected method: {}".format(state["method"]))
    if state["holding_period"] != "1M":
        raise SignalError("only holding_period=1M is supported")
    if not str(state["watermark"]).startswith(SIMULATION_WATERMARK_PREFIX):
        raise SignalError("signal is not watermarked for paper trading")
    if bool(state.get("stale")):
        raise SignalError("stale signal is blocked")
    signal_date = _parse_date(state["signal_date"], "signal_date")
    asof = _parse_date(state.get("asof"), "asof")
    effective = _parse_date(state["effective_trade_date"], "effective_trade_date")
    if signal_date != asof:
        raise SignalError("signal_date and asof disagree")
    if effective <= signal_date:
        raise SignalError("effective_trade_date must be after signal_date")

    if not targets:
        raise SignalError("signal contains no targets")
    total_weight = sum(targets.values())
    if not math.isfinite(total_weight) or total_weight <= 0 or total_weight > 1.000001:
        raise SignalError(f"invalid total target weight: {total_weight}")
    if int(state.get("top_n") or 0) != len(targets):
        raise SignalError("top_n does not match signal rows")
    selection_payload = json.dumps(
        state["selection"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if hashlib.sha256(selection_payload).hexdigest() != state["selection_sha256"]:
        raise SignalError("selection sha256 mismatch")
    actual_hash = _sha256(Path(root) / "signal_latest.csv")
    if actual_hash != state["signal_sha256"]:
        raise SignalError("signal_latest.csv sha256 mismatch")
    _validate_factor_manifest(state)


def load_signal(signal_dir, expected_report_id=None, expected_method=None):
    root = Path(signal_dir)
    signal_path = root / "signal_latest.csv"
    state_path = root / "state.json"
    if not signal_path.is_file() or not state_path.is_file():
        raise SignalError(f"signal_latest.csv or state.json is missing in {root}")

    targets = {}
    with signal_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"symbol", "target_weight"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise SignalError("signal CSV requires symbol and target_weight")
        for row in reader:
            symbol = _normalize_symbol(row.get("symbol"))
            if symbol in targets:
                raise SignalError(f"duplicate signal symbol: {symbol}")
            try:
                weight = float(row.get("target_weight"))
            except (TypeError, ValueError):
                raise SignalError(f"invalid target_weight for {symbol}")
            if not math.isfinite(weight) or weight <= 0:
                raise SignalError(f"target_weight must be positive for {symbol}")
            targets[symbol] = weight
    state = json.loads(state_path.read_text(encoding="utf-8"))
    validate_signal_bundle(root, targets, state, expected_report_id, expected_method)
    return root, targets, state


# Pure portfolio calculations ------------------------------------------------

def _last_price(value):
    if isinstance(value, dict):
        value = value.get("lastPrice")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0 else 0.0


def minimum_buy(symbol):
    if symbol.startswith(("688", "689")):
        return 200
    return 100


def round_buy_quantity(symbol, shares):
    shares = math.floor(float(shares))
    minimum = minimum_buy(symbol)
    if shares < minimum:
        return 0
    if symbol.endswith(".BJ") or symbol.startswith(("688", "689")):
        return shares
    return shares // 100 * 100


def round_lot(symbol, shares):
    """Backward-compatible alias for buy-side quantity rounding."""
    return round_buy_quantity(symbol, shares)


def legal_sell_quantity(symbol, desired, held, sellable):
    desired = max(0, min(int(desired), int(sellable)))
    held = max(0, int(held))
    if desired <= 0:
        return 0
    if desired >= held and sellable >= held:
        return held
    minimum = minimum_buy(symbol)
    if symbol.endswith(".BJ") or symbol.startswith(("688", "689")):
        return desired if desired >= minimum else 0
    odd_lot = held % 100
    if desired % 100 == 0:
        return desired
    if odd_lot and desired >= odd_lot and (desired - odd_lot) % 100 == 0:
        return desired
    return desired // 100 * 100


def build_target_shares(targets, prices, equity, cash_buffer_rate=CASH_BUFFER_RATE):
    investable = float(equity) * (1.0 - float(cash_buffer_rate))
    desired = {}
    missing = []
    for symbol, weight in targets.items():
        price = _last_price(prices.get(symbol))
        if price <= 0:
            missing.append(symbol)
            continue
        desired[symbol] = round_buy_quantity(symbol, float(weight) * investable / price)
    return desired, missing


def plan_orders(targets, prices, positions, equity, cash):
    desired, missing = build_target_shares(targets, prices, equity)
    orders = []
    skipped = [{"symbol": symbol, "reason": "missing_price"} for symbol in missing]
    projected_cash = float(cash)

    for symbol in sorted(set(positions) | set(desired)):
        held = positions.get(symbol, {"volume": 0, "sellable": 0})
        volume = int(held.get("volume", 0))
        sellable = int(held.get("sellable", 0))
        difference = desired.get(symbol, 0) - volume
        if difference >= 0:
            continue
        quantity = legal_sell_quantity(symbol, -difference, volume, sellable)
        if quantity > 0:
            orders.append((symbol, "sell", quantity))
            projected_cash += quantity * _last_price(prices.get(symbol)) * 0.999
        if quantity < -difference:
            skipped.append({"symbol": symbol, "reason": "sell_remainder"})

    buys = []
    for symbol in desired:
        held = int(positions.get(symbol, {}).get("volume", 0))
        difference = desired[symbol] - held
        if difference > 0:
            buys.append((symbol, "buy", difference))
    buys.sort(key=lambda item: (-targets[item[0]], item[0]))
    for symbol, side, quantity in buys:
        price = _last_price(prices.get(symbol))
        if price <= 0:
            continue
        affordable = round_buy_quantity(symbol, projected_cash / (price * 1.0015))
        quantity = min(quantity, affordable)
        if quantity <= 0:
            skipped.append({"symbol": symbol, "reason": "insufficient_cash"})
            continue
        projected_cash -= quantity * price * 1.0015
        orders.append((symbol, side, quantity))
    return orders, skipped


def tick_filter(prices, order, is_buy):
    quote = prices.get(order[0]) or {}
    if not isinstance(quote, dict):
        quote = {"lastPrice": quote}
    last_price = _last_price(quote)
    if last_price <= 0:
        return "suspended_or_missing_quote"
    limit_up = float(quote.get("limitUp") or 0.0)
    limit_down = float(quote.get("limitDown") or 0.0)
    if is_buy and limit_up > 0 and last_price >= limit_up - 0.0001:
        return "at_limit_up"
    if not is_buy and limit_down > 0 and last_price <= limit_down + 0.0001:
        return "at_limit_down"
    side_prices = quote.get("askPrice" if is_buy else "bidPrice") or []
    best = float(side_prices[0] or 0.0) if side_prices else 0.0
    if best <= 0:
        return "missing_opposite_quote"
    return None


# Persistent execution journal ----------------------------------------------

def _atomic_write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(str(temporary), str(path))


def last_executed(root, asof=None):
    marker = Path(root) / "executed.json"
    if not marker.is_file():
        return None
    payload = json.loads(marker.read_text(encoding="utf-8"))
    return payload.get("asof")


def execution_completed(root, state):
    marker = Path(root) / "executed.json"
    if not marker.is_file():
        return False
    payload = json.loads(marker.read_text(encoding="utf-8"))
    return (
        payload.get("status") == "completed"
        and payload.get("asof") == state.get("asof")
        and payload.get("run_id") == state.get("run_id")
        and payload.get("signal_sha256") == state.get("signal_sha256")
    )


def mark_executed(root, state, orders=None, skipped=None, journal=None):
    payload = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "status": "completed",
        "asof": state.get("asof"),
        "effective_trade_date": state.get("effective_trade_date"),
        "method": state.get("method"),
        "report_id": state.get("report_id"),
        "run_id": state.get("run_id"),
        "signal_sha256": state.get("signal_sha256"),
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "orders": orders or (journal or {}).get("orders", {}),
        "skipped": skipped or [],
    }
    _atomic_write_json(Path(root) / "executed.json", payload)


def _new_journal(state, target_shares):
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    return {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "status": "running",
        "phase": "sell",
        "asof": state["asof"],
        "effective_trade_date": state["effective_trade_date"],
        "method": state["method"],
        "report_id": state["report_id"],
        "run_id": state["run_id"],
        "signal_sha256": state["signal_sha256"],
        "target_shares": target_shares,
        "orders": {},
        "created_at": now,
        "updated_at": now,
        "message": "",
    }


def _save_journal(root, journal):
    journal["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    _atomic_write_json(Path(root) / "execution_state.json", journal)


def _load_journal(root, state):
    path = Path(root) / "execution_state.json"
    if not path.is_file():
        return None
    journal = json.loads(path.read_text(encoding="utf-8"))
    identity = ("asof", "method", "report_id", "run_id", "signal_sha256")
    if any(journal.get(key) != state.get(key) for key in identity):
        raise SignalError("execution_state.json belongs to a different signal bundle")
    return journal


def _block(root, journal, message):
    journal["status"] = "blocked"
    journal["phase"] = "blocked"
    journal["message"] = message
    _save_journal(root, journal)
    print(f"[MLQ][BLOCKED] {message}")


# QMT innerApi adapters -------------------------------------------------------

def _runtime_global(name, default=None):
    return globals().get(name, default)


def _trade_details(account_id, account_type, kind, strategy_name=None):
    function = _runtime_global("get_trade_detail_data")
    if function is None:
        raise RuntimeError("QMT get_trade_detail_data is unavailable")
    if strategy_name and kind.upper() in ("ORDER", "DEAL"):
        return function(account_id, account_type, kind, strategy_name) or []
    return function(account_id, account_type, kind) or []


def _object_symbol(item):
    code = str(getattr(item, "m_strInstrumentID", "") or "").upper()
    if "." in code:
        return _normalize_symbol(code)
    exchange = str(getattr(item, "m_strExchangeID", "") or "").upper()
    return _normalize_symbol(f"{code}.{exchange}")


def query_snapshot(account_id, account_type):
    account_rows = _trade_details(account_id, account_type, "ACCOUNT")
    if not account_rows:
        raise RuntimeError("QMT account is not logged in")
    asset = account_rows[0]
    equity = float(getattr(asset, "m_dBalance", 0.0) or 0.0)
    cash = float(getattr(asset, "m_dAvailable", 0.0) or 0.0)
    if equity <= 0 or cash < 0:
        raise RuntimeError("invalid QMT account snapshot")
    positions = {}
    for item in _trade_details(account_id, account_type, "POSITION"):
        symbol = _object_symbol(item)
        volume = int(getattr(item, "m_nVolume", 0) or 0)
        if volume <= 0:
            continue
        positions[symbol] = {
            "volume": volume,
            "sellable": int(getattr(item, "m_nCanUseVolume", 0) or 0),
        }
    return positions, equity, cash


def fetch_prices(context, symbols):
    symbols = sorted(set(symbols))
    full = context.get_full_tick(symbols) or {}
    prices = {}
    for symbol in symbols:
        tick = full.get(symbol) or {}
        detail = context.get_instrument_detail(symbol, True) or {}
        prices[symbol] = {
            "lastPrice": float(tick.get("lastPrice") or 0.0),
            "lastClose": float(tick.get("lastClose") or 0.0),
            "askPrice": list(tick.get("askPrice") or []),
            "bidPrice": list(tick.get("bidPrice") or []),
            "stockStatus": tick.get("stockStatus"),
            "limitUp": float(detail.get("UpStopPrice") or 0.0),
            "limitDown": float(detail.get("DownStopPrice") or 0.0),
        }
    return prices


def _limit_price(quote, is_buy):
    levels = quote.get("askPrice" if is_buy else "bidPrice") or []
    price = float(levels[0] or 0.0) if levels else 0.0
    if price <= 0:
        return 0.0
    upper = float(quote.get("limitUp") or 0.0)
    lower = float(quote.get("limitDown") or 0.0)
    if upper > 0:
        price = min(price, upper)
    if lower > 0:
        price = max(price, lower)
    return round(price + 1e-9, 2)


def _order_remark(state, side, symbol):
    date_token = str(state["asof"]).replace("-", "")[2:]
    return "MLQ{}{}{}".format(date_token, "B" if side == "buy" else "S", symbol[:6])


def _qmt_orders(account_id, account_type):
    return _trade_details(account_id, account_type, "ORDER", STRATEGY_NAME)


def _parse_iso_datetime(value):
    # datetime.fromisoformat is unavailable in QMT's Python 3.6 runtime.
    text = str(value).split("+")[0]
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S").astimezone()


def _reconcile_journal(journal, qmt_orders):
    by_remark = {}
    for item in qmt_orders:
        remark = str(getattr(item, "m_strRemark", "") or "")
        if remark:
            by_remark[remark] = item
    now = datetime.now().astimezone()
    active = False
    for remark, record in journal["orders"].items():
        item = by_remark.get(remark)
        if item is not None:
            status = int(getattr(item, "m_nOrderStatus", 255) or 255)
            record["order_status"] = status
            record["traded_volume"] = int(getattr(item, "m_nVolumeTraded", 0) or 0)
            record["order_sysid"] = str(getattr(item, "m_strOrderSysID", "") or "")
            if status == ORDER_FILLED:
                record["status"] = "filled"
            elif status in ORDER_ACTIVE:
                record["status"] = "active"
                active = True
            elif status in ORDER_TERMINAL:
                record["status"] = "terminal_incomplete"
                if not record.get("terminal_seen_at"):
                    record["terminal_seen_at"] = now.astimezone().isoformat(timespec="seconds")
                terminal_seen = _parse_iso_datetime(record["terminal_seen_at"])
                if (now - terminal_seen).total_seconds() <= POSITION_SYNC_GRACE_SECONDS:
                    active = True
        elif record.get("status") in ("intent", "submitted", "active"):
            submitted = _parse_iso_datetime(record["submitted_at"])
            if (now - submitted).total_seconds() <= SUBMISSION_GRACE_SECONDS:
                active = True
            else:
                record["status"] = "not_found"
    return active


def _external_active_symbols(qmt_orders, own_remarks):
    result = set()
    for item in qmt_orders:
        status = int(getattr(item, "m_nOrderStatus", 255) or 255)
        remark = str(getattr(item, "m_strRemark", "") or "")
        if status in ORDER_ACTIVE and remark not in own_remarks:
            result.add(_object_symbol(item))
    return result


def _submit_order(context, root, journal, state, symbol, side, quantity, quote):
    is_buy = side == "buy"
    reason = tick_filter({symbol: quote}, (symbol, side, quantity), is_buy)
    if reason:
        return reason
    price = _limit_price(quote, is_buy)
    if price <= 0:
        return "invalid_limit_price"
    remark = _order_remark(state, side, symbol)
    if remark in journal["orders"]:
        return "already_attempted"
    record = {
        "symbol": symbol,
        "side": side,
        "shares": int(quantity),
        "limit_price": price,
        "status": "intent",
        "submitted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    journal["orders"][remark] = record
    _save_journal(root, journal)
    function = _runtime_global("passorder")
    if function is None:
        record["status"] = "error"
        record["error"] = "QMT passorder is unavailable"
        _save_journal(root, journal)
        return record["error"]
    try:
        function(
            STOCK_BUY if is_buy else STOCK_SELL,
            ORDER_BY_SHARES,
            RUNTIME.account_id,
            symbol,
            FIX_PRICE,
            price,
            int(quantity),
            STRATEGY_NAME,
            QUICK_TRADE_IMMEDIATE,
            remark,
            context,
        )
    except Exception as error:  # noqa: BLE001 - QMT injects a native API boundary
        record["status"] = "error"
        record["error"] = repr(error)
        _save_journal(root, journal)
        return record["error"]
    record["status"] = "submitted"
    _save_journal(root, journal)
    print(f"[MLQ] submitted {side} {symbol} {quantity} @ {price:.2f} remark={remark}")
    return None


def _sell_residuals(target_shares, positions):
    result = []
    blocked = []
    for symbol in sorted(set(positions) | set(target_shares)):
        position = positions.get(symbol, {"volume": 0, "sellable": 0})
        held = int(position.get("volume", 0))
        desired = int(target_shares.get(symbol, 0))
        if held <= desired:
            continue
        quantity = legal_sell_quantity(
            symbol, held - desired, held, int(position.get("sellable", 0))
        )
        if quantity > 0:
            result.append((symbol, "sell", quantity))
        if quantity < held - desired:
            blocked.append(symbol)
    return result, blocked


def _buy_residuals(target_shares, positions, prices, cash):
    candidates = []
    blocked = []
    budget = float(cash) * (1.0 - CASH_BUFFER_RATE)
    for symbol, desired in target_shares.items():
        held = int(positions.get(symbol, {}).get("volume", 0))
        quantity = int(desired) - held
        if quantity > 0:
            candidates.append((symbol, quantity))
    candidates.sort(key=lambda item: (-RUNTIME.targets[item[0]], item[0]))
    result = []
    for symbol, desired_quantity in candidates:
        quote = prices.get(symbol) or {}
        price = _limit_price(quote, True)
        if price <= 0:
            blocked.append(symbol)
            continue
        affordable = round_buy_quantity(symbol, budget / (price * 1.0015))
        quantity = min(desired_quantity, affordable)
        if quantity <= 0:
            blocked.append(symbol)
            continue
        result.append((symbol, "buy", quantity))
        budget -= quantity * price * 1.0015
        if quantity < desired_quantity:
            blocked.append(symbol)
    return result, blocked


def _positions_match(target_shares, positions):
    differences = {}
    for symbol in sorted(set(target_shares) | set(positions)):
        held = int(positions.get(symbol, {}).get("volume", 0))
        desired = int(target_shares.get(symbol, 0))
        if held != desired:
            differences[symbol] = {"held": held, "target": desired}
    return differences


def _write_dry_run(root, state, positions, equity, cash, prices):
    orders, skipped = plan_orders(RUNTIME.targets, prices, positions, equity, cash)
    payload = {
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "status": "dry_run",
        "asof": state["asof"],
        "effective_trade_date": state["effective_trade_date"],
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "equity": equity,
        "cash": cash,
        "orders": [
            {"symbol": symbol, "side": side, "shares": shares}
            for symbol, side, shares in orders
        ],
        "skipped": skipped,
    }
    _atomic_write_json(Path(root) / "dry_run_plan.json", payload)
    print("[MLQ][DRY_RUN] targets={} orders={} skipped={} output={}".format(
        len(RUNTIME.targets), len(orders), len(skipped), Path(root) / "dry_run_plan.json"
    ))
    for symbol, side, shares in orders:
        print(f"[MLQ][DRY_RUN] {side} {symbol} {shares}")


# QMT lifecycle --------------------------------------------------------------

def init(context):
    try:
        root, targets, state = load_signal(
            SIGNAL_DIR,
            expected_report_id=EXPECTED_REPORT_ID,
            expected_method=EXPECTED_METHOD,
        )
        account_id = ACCOUNT_ID_OVERRIDE or str(_runtime_global("account", "") or "")
        account_type = ACCOUNT_TYPE_OVERRIDE or str(_runtime_global("accountType", "") or "")
        if not account_id:
            raise SignalError("select a QMT account or set ACCOUNT_ID_OVERRIDE")
        if account_type.upper() != "STOCK":
            raise SignalError("only STOCK accounts are supported")
        context.set_account(account_id)
        context.set_universe(sorted(targets))
        RUNTIME.root = root
        RUNTIME.targets = targets
        RUNTIME.state = state
        RUNTIME.account_id = account_id
        RUNTIME.account_type = account_type.upper()
        RUNTIME.ready = True
        print("[MLQ] loaded report={} method={} asof={} effective={} factors={} targets={}".format(
            state["report_id"], state["method"], state["asof"],
            state["effective_trade_date"], len(state["factor_manifest"]), len(targets)
        ))
    except Exception as error:  # noqa: BLE001 - init must fail closed at the QMT boundary
        RUNTIME.error = repr(error)
        RUNTIME.ready = False
        print(f"[MLQ][BLOCKED] init failed: {RUNTIME.error}")


def handlebar(context):
    if not RUNTIME.ready:
        return
    if not context.is_last_bar():
        return
    now = datetime.now().astimezone()
    today = now.date()
    effective = _parse_date(RUNTIME.state["effective_trade_date"], "effective_trade_date")
    current_time = now.strftime("%H%M%S")

    if execution_completed(RUNTIME.root, RUNTIME.state):
        return
    if not DRY_RUN:
        if not SIMULATION_ACCOUNT_CONFIRMED:
            _print_once("set SIMULATION_ACCOUNT_CONFIRMED=True before simulated orders")
            return
        if today != effective:
            _print_once(f"live execution blocked: today={today} effective={effective}")
            return
        if current_time < TRADE_START or current_time > TRADE_END:
            _print_once(f"outside execution window {TRADE_START}-{TRADE_END}")
            return
    elif RUNTIME.previewed:
        return

    try:
        positions, equity, cash = query_snapshot(RUNTIME.account_id, RUNTIME.account_type)
        symbols = set(RUNTIME.targets) | set(positions)
        prices = fetch_prices(context, symbols)
        if DRY_RUN:
            _write_dry_run(RUNTIME.root, RUNTIME.state, positions, equity, cash, prices)
            RUNTIME.previewed = True
            return
        _execute_live_cycle(context, positions, equity, cash, prices)
    except Exception as error:  # noqa: BLE001 - keep the QMT event loop alive and blocked
        _print_once(f"runtime error: {error!r}")


def _execute_live_cycle(context, positions, equity, cash, prices):
    journal = _load_journal(RUNTIME.root, RUNTIME.state)
    if journal is None:
        target_shares, missing = build_target_shares(RUNTIME.targets, prices, equity)
        if missing:
            print("[MLQ][BLOCKED] missing target quotes: {}".format(",".join(sorted(missing))))
            return
        journal = _new_journal(RUNTIME.state, target_shares)
        _save_journal(RUNTIME.root, journal)
    if journal.get("status") == "blocked":
        _print_once("execution journal is blocked: {}".format(journal.get("message", "")))
        return

    qmt_orders = _qmt_orders(RUNTIME.account_id, RUNTIME.account_type)
    active = _reconcile_journal(journal, qmt_orders)
    own_remarks = set(journal["orders"])
    portfolio_symbols = set(journal["target_shares"]) | set(positions)
    external = _external_active_symbols(qmt_orders, own_remarks) & portfolio_symbols
    if external:
        _block(RUNTIME.root, journal, "external active orders: {}".format(",".join(sorted(external))))
        return
    _save_journal(RUNTIME.root, journal)
    if active:
        return

    phase = journal["phase"]
    if phase in ("sell", "wait_sell"):
        sells, blocked = _sell_residuals(journal["target_shares"], positions)
        if blocked:
            _block(RUNTIME.root, journal, "unsellable residuals: {}".format(",".join(blocked)))
            return
        if sells:
            attempted = {
                (item["symbol"], item["side"]) for item in journal["orders"].values()
            }
            new_sells = [item for item in sells if (item[0], item[1]) not in attempted]
            if not new_sells:
                _block(RUNTIME.root, journal, "sell orders ended without reaching targets")
                return
            errors = []
            for symbol, side, quantity in new_sells:
                error = _submit_order(
                    context, RUNTIME.root, journal, RUNTIME.state,
                    symbol, side, quantity, prices.get(symbol) or {},
                )
                if error:
                    errors.append(f"{symbol}:{error}")
            journal["phase"] = "wait_sell"
            _save_journal(RUNTIME.root, journal)
            if errors:
                _block(RUNTIME.root, journal, "sell submission errors: {}".format(";".join(errors)))
            return
        journal["phase"] = "buy"
        _save_journal(RUNTIME.root, journal)
        phase = "buy"

    if phase in ("buy", "wait_buy"):
        sells, blocked_sells = _sell_residuals(journal["target_shares"], positions)
        if sells or blocked_sells:
            _block(RUNTIME.root, journal, "sell residual appeared before buys")
            return
        buys, blocked = _buy_residuals(
            journal["target_shares"], positions, prices, cash
        )
        if buys:
            attempted = {
                (item["symbol"], item["side"]) for item in journal["orders"].values()
            }
            new_buys = [item for item in buys if (item[0], item[1]) not in attempted]
            if not new_buys:
                _block(RUNTIME.root, journal, "buy orders ended without reaching targets")
                return
            errors = []
            for symbol, side, quantity in new_buys:
                error = _submit_order(
                    context, RUNTIME.root, journal, RUNTIME.state,
                    symbol, side, quantity, prices.get(symbol) or {},
                )
                if error:
                    errors.append(f"{symbol}:{error}")
            journal["phase"] = "wait_buy"
            _save_journal(RUNTIME.root, journal)
            if errors:
                _block(RUNTIME.root, journal, "buy submission errors: {}".format(";".join(errors)))
            return
        if blocked:
            _block(RUNTIME.root, journal, "unfunded or unquoted buys: {}".format(",".join(blocked)))
            return

    differences = _positions_match(journal["target_shares"], positions)
    if differences:
        _block(RUNTIME.root, journal, f"position mismatch: {json.dumps(differences, ensure_ascii=True, sort_keys=True)}")
        return
    journal["phase"] = "completed"
    journal["status"] = "completed"
    _save_journal(RUNTIME.root, journal)
    mark_executed(RUNTIME.root, RUNTIME.state, journal=journal)
    print("[MLQ] rebalance completed and reconciled")


def _print_once(message):
    if RUNTIME.last_message != message:
        print(f"[MLQ] {message}")
        RUNTIME.last_message = message
