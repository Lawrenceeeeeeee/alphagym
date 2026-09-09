"""Persist and normalize OKX public trading statistics for market-state research."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphagym import storage_io
from alphagym.okx_api import OKXDemoClient

METRICS = {
    "long_short_account_ratio": ("long_short_account_ratio",),
    "open_interest_volume": ("open_interest_usd", "contract_volume_usd"),
    "taker_volume": ("taker_sell_volume_usd", "taker_buy_volume_usd"),
}


def fetch_trading_statistics(
    *, currencies=("BTC", "ETH"), period="1D", client=None
) -> pd.DataFrame:
    client = client or OKXDemoClient()
    if period not in {"5m", "1H", "1D"}:
        raise ValueError("period must be 5m, 1H or 1D")
    supported = set(client.trading_statistics_support_coins().get("contract", []))
    requested = tuple(dict.fromkeys(str(ccy).upper() for ccy in currencies))
    unsupported = set(requested) - supported
    if unsupported:
        raise ValueError(f"unsupported OKX contract statistics currencies: {sorted(unsupported)}")
    pieces = []
    for ccy in requested:
        merged = None
        for metric, columns in METRICS.items():
            rows = client.trading_statistics(metric, ccy=ccy, period=period)
            frame = pd.DataFrame(rows, columns=("ts", *columns))
            if frame.empty:
                continue
            frame["ts"] = pd.to_datetime(pd.to_numeric(frame["ts"]), unit="ms", utc=True)
            for column in columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            merged = frame if merged is None else merged.merge(frame, on="ts", how="outer")
        if merged is not None:
            merged["currency"] = ccy
            pieces.append(merged)
    if not pieces:
        raise ValueError("OKX returned no trading statistics")
    return pd.concat(pieces, ignore_index=True).sort_values(["ts", "currency"])


def compute_market_states(raw: pd.DataFrame) -> pd.DataFrame:
    required = {
        "ts", "currency", "long_short_account_ratio", "open_interest_usd",
        "contract_volume_usd", "taker_sell_volume_usd", "taker_buy_volume_usd",
    }
    missing = required - set(raw)
    if missing:
        raise ValueError(f"missing OKX trading-statistics columns: {sorted(missing)}")
    data = raw.sort_values(["currency", "ts"]).copy()
    grouped = data.groupby("currency", observed=True, group_keys=False)
    denominator = data["taker_buy_volume_usd"] + data["taker_sell_volume_usd"]
    data["taker_imbalance"] = (
        (data["taker_buy_volume_usd"] - data["taker_sell_volume_usd"])
        / denominator.replace(0, np.nan)
    )
    data["open_interest_change"] = grouped["open_interest_usd"].transform(
        lambda x: np.log(x).diff()
    )
    ratio_mean = grouped["long_short_account_ratio"].transform(
        lambda x: x.rolling(30, min_periods=15).mean()
    )
    ratio_std = grouped["long_short_account_ratio"].transform(
        lambda x: x.rolling(30, min_periods=15).std()
    )
    data["long_short_ratio_z"] = (
        (data["long_short_account_ratio"] - ratio_mean) / ratio_std.replace(0, np.nan)
    )
    volume_mean = grouped["contract_volume_usd"].transform(
        lambda x: np.log1p(x).rolling(30, min_periods=15).mean()
    )
    volume_std = grouped["contract_volume_usd"].transform(
        lambda x: np.log1p(x).rolling(30, min_periods=15).std()
    )
    data["contract_volume_z"] = (
        (np.log1p(data["contract_volume_usd"]) - volume_mean) / volume_std.replace(0, np.nan)
    )
    # Aggregate BTC/ETH (or the configured currency set) robustly into one market state.
    aggregate_columns = [
        "taker_imbalance", "open_interest_change", "long_short_ratio_z", "contract_volume_z"
    ]
    state = data.groupby("ts", observed=True)[aggregate_columns].median().reset_index()
    state["market_state"] = np.select(
        [
            state["long_short_ratio_z"].gt(1) & state["open_interest_change"].gt(0),
            state["open_interest_change"].lt(0) & state["taker_imbalance"].lt(0),
            state["open_interest_change"].gt(0) & state["taker_imbalance"].gt(0),
        ],
        ["crowded_long", "deleveraging", "risk_on"],
        default="neutral",
    )
    return state


def update_trading_statistics(
    root: str | Path, *, currencies=("BTC", "ETH"), period="1D", client=None
) -> dict:
    root = Path(root).expanduser().resolve()
    raw = fetch_trading_statistics(currencies=currencies, period=period, client=client)
    state = compute_market_states(raw)
    prefix = f"crypto/okx/trading_statistics/{period}/"
    store = storage_io.store_for(root, initialize=True)
    with store.batch() as batch:
        batch.frame(prefix + "raw.parquet", raw, mode="upsert", keys=("ts", "currency"))
        batch.frame(prefix + "market_state.parquet", state, mode="upsert", keys=("ts",))
    return {
        "ok": True, "period": period, "currencies": list(currencies), "rows": len(raw),
        "start": raw["ts"].min().isoformat(), "end": raw["ts"].max().isoformat(),
        "raw_path": prefix + "raw.parquet",
        "market_state_path": prefix + "market_state.parquet",
        "historical_scope": "bounded_recent_window; persist repeated snapshots for research",
    }
