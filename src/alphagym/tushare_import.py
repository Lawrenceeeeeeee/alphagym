"""Tushare data source: bounded requests, no persisted secrets or fabricated history."""
from __future__ import annotations

import json
import os
import re
import time

import numpy as np
import pandas as pd

from alphagym.config import resolve_root
from alphagym.equity_data import DataContractError
from alphagym.storage import KEYS
from alphagym.storage_io import store_for


def _is_a_share(code):
    return (isinstance(code, str) and bool(re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", code))
            and not code.startswith(("200", "900")))


def _dates(frame, columns):
    for column in columns:
        frame[column] = pd.to_datetime(frame[column], format="mixed", errors="coerce").dt.normalize()
    return frame


class TushareImporter:
    def __init__(self, pro, root, *, start=None, end=None, rate_limit=.3,
                 overlap_days=7, daily_basic=True):
        if rate_limit < 0 or overlap_days < 0:
            raise ValueError("rate_limit and overlap_days must be nonnegative")
        self.pro = pro
        self.store = store_for(resolve_root(root), initialize=True)
        self.rate_limit = rate_limit
        completed = pd.Timestamp.now(tz="Asia/Shanghai").normalize().tz_localize(None) - pd.Timedelta(days=1)
        self.end = pd.Timestamp(end).normalize() if end else completed
        if self.end > completed:
            raise DataContractError("Daily sync only accepts completed dates before today")
        self.start = pd.Timestamp(start).normalize() if start else None
        self.overlap_days = overlap_days
        self.daily_basic = daily_basic
        if self.start is not None and self.start > self.end:
            raise DataContractError("start must not be after end")

    def _call(self, interface, **kwargs):
        for attempt in range(3):
            try:
                result = getattr(self.pro, interface)(**kwargs)
                time.sleep(self.rate_limit)
                return pd.DataFrame(result) if result is not None else pd.DataFrame()
            except Exception:  # noqa: BLE001 -- redact provider errors, which can contain tokens
                if attempt == 2:
                    raise DataContractError(
                        f"Tushare {interface} failed; check token, API permissions and quota"
                    ) from None
                time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def _pages(self, interface, **kwargs):
        pages, seen = [], set()
        page_size = 100 if interface in {"fina_indicator", "income"} else 2000
        for offset in range(0, 1_000_000, page_size):
            frame = self._call(interface, **kwargs, limit=page_size, offset=offset)
            if frame.empty:
                break
            fingerprint = int(pd.util.hash_pandas_object(frame, index=False).sum())
            if fingerprint in seen:
                raise DataContractError(f"Tushare {interface} pagination did not advance")
            seen.add(fingerprint)
            pages.append(frame)
            if len(frame) < page_size:
                break
        else:
            raise DataContractError(f"Tushare {interface} pagination exceeded safety bound")
        return pd.concat(pages, ignore_index=True) if pages else pd.DataFrame()

    def _start(self, name):
        if self.start is not None:
            return self.start
        key = f"sync/tushare/{name}/checkpoint.json"
        if self.store.manifest(key):
            day = json.loads(self.store.read_blob(key))["completed_through"]
            return pd.Timestamp(day) - pd.Timedelta(days=self.overlap_days)
        return pd.Timestamp("2012-01-01")

    def run_market(self):
        start = self._start("market")
        calendar = self._pages("trade_cal", exchange="SSE", start_date=start.strftime("%Y%m%d"),
                               end_date=self.end.strftime("%Y%m%d"))
        if calendar.empty:
            raise DataContractError("Tushare returned an empty trading calendar")
        calendar = calendar[["cal_date", "is_open"]].rename(columns={"cal_date": "trade_date"})
        _dates(calendar, ["trade_date"])
        calendar["is_open"] = pd.to_numeric(calendar["is_open"], errors="raise").eq(1)
        self.store.write_frame("equity/calendar.parquet", calendar, mode="upsert", keys=KEYS["calendar"])
        count = days = 0
        for day in sorted(calendar.loc[calendar["is_open"], "trade_date"].unique()):
            date = pd.Timestamp(day).strftime("%Y%m%d")
            bars = self._pages("daily", trade_date=date)
            factors = self._pages("adj_factor", trade_date=date)
            if bars.empty or factors.empty:
                raise DataContractError(f"Incomplete market response for {date}; checkpoint unchanged")
            bars = bars.loc[bars["ts_code"].map(_is_a_share)].rename(columns={"ts_code": "symbol", "vol": "volume"})
            bars = bars[["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"]].copy()
            factors = factors.loc[factors["ts_code"].map(_is_a_share)].rename(columns={"ts_code": "symbol", "adj_factor": "adjust_factor"})
            factors = factors[["symbol", "trade_date", "adjust_factor"]].copy()
            if set(bars["symbol"]) - set(factors["symbol"]):
                raise DataContractError(f"Missing adjustment factors for {date}")
            for col in ("open", "high", "low", "close", "volume", "amount"):
                bars[col] = pd.to_numeric(bars[col], errors="raise").astype(float)
            bars["volume"] *= 100
            bars["amount"] *= 1000
            bars["turnover"] = np.nan
            bars["float_market_cap"] = np.nan
            if self.daily_basic:
                basic = self._pages("daily_basic", trade_date=date)
                if basic.empty:
                    raise DataContractError(f"daily_basic empty for {date}")
                basic = basic.drop_duplicates("ts_code").set_index("ts_code")
                bars["turnover"] = bars["symbol"].map(pd.to_numeric(basic["turnover_rate"], errors="raise") / 100)
                bars["float_market_cap"] = bars["symbol"].map(pd.to_numeric(basic["circ_mv"], errors="raise") * 10000)
            _dates(bars, ["trade_date"])
            _dates(factors, ["trade_date"])
            factors["adjust_factor"] = pd.to_numeric(factors["adjust_factor"], errors="raise").astype(float)
            if factors["adjust_factor"].isna().any() or factors["adjust_factor"].le(0).any():
                raise DataContractError("Invalid adjustment factor")
            with self.store.batch() as batch:
                batch.frame("equity/daily.parquet", bars, mode="upsert", keys=KEYS["daily"])
                batch.frame("equity/adjustments.parquet", factors, mode="upsert", keys=KEYS["adjustments"])
                batch.blob("sync/tushare/market/checkpoint.json", json.dumps({
                    "completed_through": str(pd.Timestamp(day).date()), "provider": "tushare",
                }).encode())
            count += len(bars)
            days += 1
        return {"provider": "tushare", "days": days, "rows": count,
                "start": str(start.date()), "end": str(self.end.date())}

    def run_securities(self):
        parts = [self._pages("stock_basic", list_status=status,
                             fields="ts_code,list_date,delist_date") for status in ("L", "D", "P")]
        frame = pd.concat(parts, ignore_index=True)
        if frame.empty:
            raise DataContractError("Tushare returned empty securities")
        frame = frame.loc[frame["ts_code"].map(_is_a_share)].rename(columns={"ts_code": "symbol"})
        frame = _dates(frame[["symbol", "list_date", "delist_date"]].copy(), ["list_date", "delist_date"])
        self.store.write_frame("equity/securities.parquet", frame, mode="upsert", keys=KEYS["securities"])
        return {"rows": len(frame)}

    def run_fundamentals(self, symbols=None):
        full_universe = not symbols
        if full_universe:
            self.run_securities()
            symbols = self.store.read_frame("equity/securities.parquet", columns=["symbol"])["symbol"].tolist()
        count = 0
        for symbol in sorted(set(symbols)):
            if not _is_a_share(symbol):
                raise DataContractError(f"Not an A-share symbol: {symbol}")
            # Fetch each symbol's release history so an incremental release can
            # retain previously available fields without using future values.
            # A global cursor must never skip a newly added symbol's history.
            params = {"ts_code": symbol, "end_date": self.end.strftime("%Y%m%d")}
            frame = self._merge_fundamental(self._pages("fina_indicator", **params), self._pages("income", **params))
            if self.start is not None:
                frame = frame[frame["available_date"] >= self.start]
            if not frame.empty:
                self.store.write_frame("equity/fundamentals.parquet", frame,
                                       mode="upsert", keys=KEYS["fundamentals"])
                count += len(frame)
        if full_universe:
            self.store.write_blob("sync/tushare/fundamentals/checkpoint.json", json.dumps({
                "completed_through": str(self.end.date()),
            }).encode())
        return {"rows": count, "symbols": len(symbols)}

    @staticmethod
    def _merge_fundamental(indicator, income):
        columns = ["symbol", "stat_date", "available_date", "eps", "bps", "revenue",
                   "ocfps", "net_profit", "roe", "gross_margin"]
        sources = []
        for frame, fields in (
            (indicator, {"eps": "eps", "bps": "bps", "ocfps": "ocfps", "roe": "roe", "grossprofit_margin": "gross_margin"}),
            (income, {"revenue": "revenue", "n_income_attr_p": "net_profit"}),
        ):
            if frame.empty:
                continue
            frame = frame.copy()
            if "report_type" in frame:
                frame = frame[frame["report_type"].astype(str).isin(["1", "1.0"])]
            _dates(frame, ["ann_date", "end_date"])
            if "f_ann_date" in frame:
                _dates(frame, ["f_ann_date"])
                frame["ann_date"] = frame[["ann_date", "f_ann_date"]].max(axis=1)
            if frame[["ann_date", "end_date"]].isna().any().any():
                raise DataContractError("Financial fields require an actual announcement date")
            selected = frame[["ts_code", "end_date", "ann_date"]].rename(columns={
                "ts_code": "symbol", "end_date": "stat_date", "ann_date": "available_date",
            })
            for source, target in fields.items():
                selected[target] = pd.to_numeric(frame.get(source, np.nan), errors="coerce")
            sources.append(selected)
        if not sources:
            return pd.DataFrame(columns=columns)
        events = pd.concat(sources, ignore_index=True).sort_values("available_date")
        rows = []
        for (symbol, period), group in events.groupby(["symbol", "stat_date"]):
            state = {c: np.nan for c in columns[3:]}
            for date, released in group.groupby("available_date", sort=True):
                for col in state:
                    if col in released:
                        values = released[col].dropna()
                        if len(values):
                            state[col] = float(values.iloc[-1])
                rows.append({"symbol": symbol, "stat_date": period, "available_date": date, **state})
        return pd.DataFrame(rows, columns=columns)


def sync_tushare(root=None, *, token=None, start=None, end=None, dataset="market",
                 rate_limit=.3, overlap_days=7, daily_basic=True, pro=None, symbols=None):
    if pro is None:
        secret = token or os.environ.get("TUSHARE_TOKEN")
        if not secret or not secret.strip():
            raise DataContractError("Set TUSHARE_TOKEN or pass --token")
        from alphagym.optional import require

        pro = require("tushare", "data").pro_api(secret.strip())
    importer = TushareImporter(pro, root, start=start, end=end, rate_limit=rate_limit,
                              overlap_days=overlap_days, daily_basic=daily_basic)
    if dataset == "market":
        return importer.run_market()
    if dataset == "securities":
        return importer.run_securities()
    if dataset == "fundamentals":
        return importer.run_fundamentals(symbols)
    if dataset == "all":
        return {"market": importer.run_market(), "securities": importer.run_securities(),
                "fundamentals": importer.run_fundamentals(symbols)}
    raise DataContractError(f"Unsupported Tushare dataset: {dataset}")


if __name__ == "__main__":
    import sys

    from alphagym.cli import main

    raise SystemExit(main(["equity-data", "sync-tushare", *sys.argv[1:]]))
