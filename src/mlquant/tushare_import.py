"""Download Tushare Pro A-share data into the equity/ Parquet contract.

Writes the eight ``EquityDataBundle`` tables under ``{root}/equity/``:

- daily         raw OHLCV + amount, plus turnover / float_market_cap (daily_basic)
- adjustments   sparse cumulative backward-adjust factor
- calendar      SSE trading calendar
- securities    list/delist dates
- fundamentals  quarterly eps/bps/revenue/ocfps/net_profit/roe/gross_margin
                with ``ann_date`` as point-in-time availability
- industries    historical SW1 membership (in/out dates) via index_member
- index_members index constituents with official benchmark weights
- status        suspension derived from missing daily bars (ST/limit flagged)

Idempotent and resumable: re-running continues from the last downloaded
``trade_date`` and atomically replaces each table via ``*.tmp`` + ``os.replace``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import tushare as ts
except ImportError as error:  # pragma: no cover
    raise SystemExit("tushare is not installed: pip install tushare") from error


A_SHARE_RE = re.compile(r"^\d{6}\.(SH|SZ|BJ)$")
B_SHARE_PREFIXES = ("200", "900")
INDEX_CODES = ("000300.SH", "000905.SH", "000852.SH")

DAILY_COLUMNS = ["trade_date", "symbol", "open", "high", "low", "close", "volume", "amount"]
DAILY_EXTRA = ["turnover", "float_market_cap"]
STATUS_COLUMNS = [
    "trade_date", "symbol", "is_st", "is_pt", "is_suspended", "limit_up", "limit_down",
]


def _load_token(explicit: str | None, root: Path) -> str:
    if explicit:
        return explicit
    env = os.environ.get("TUSHARE_TOKEN")
    if env:
        return env
    for candidate in (root.parent / ".env", Path.cwd() / ".env"):
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("TUSHARE_TOKEN="):
                    value = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if value:
                        return value
    raise SystemExit("TUSHARE_TOKEN not found: pass --token or set it in .env")


def _is_a_share(ts_code: str) -> bool:
    if not isinstance(ts_code, str) or not A_SHARE_RE.match(ts_code):
        return False
    return not ts_code.startswith(B_SHARE_PREFIXES)


def _norm(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.normalize()


class TushareImporter:
    def __init__(
        self,
        pro: Any,
        root: str | Path,
        *,
        start: str = "2012-01-01",
        end: str | None = None,
        rate_limit: float = 0.15,
    ) -> None:
        self.pro = pro
        self.root = Path(root).expanduser().resolve()
        self.equity = self.root / "equity"
        self.equity.mkdir(parents=True, exist_ok=True)
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end or datetime.now(tz=UTC).date())
        self.rate_limit = rate_limit

    # -- fetch plumbing ---------------------------------------------------
    def _call(self, interface: str, **kwargs: Any) -> pd.DataFrame:
        attempts = 0
        while True:
            try:
                frame = getattr(self.pro, interface)(**kwargs)
                time.sleep(self.rate_limit)
                return pd.DataFrame(frame) if frame is not None else pd.DataFrame()
            except Exception as error:
                attempts += 1
                if attempts >= 6:
                    raise RuntimeError(f"{interface} {kwargs} failed: {error}") from error
                time.sleep(2.0 * attempts)

    @staticmethod
    def _read(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path, columns=columns) if columns else pd.read_parquet(path)

    def _atomic_write(self, frame: pd.DataFrame, filename: str) -> None:
        path = self.equity / filename
        temp = self.equity / f"{filename}.tmp"
        frame.to_parquet(temp, index=False)
        temp.replace(path)

    # -- market -----------------------------------------------------------
    def run_market(self) -> None:
        calendar = self._calendar()
        self._atomic_write(calendar, "calendar.parquet")

        open_days = calendar.loc[calendar["is_open"], "trade_date"].sort_values()
        open_days = open_days[(open_days >= self.start) & (open_days <= self.end)]
        pending = open_days[open_days > self._resume_date()]

        daily_buf: list[pd.DataFrame] = []
        adj_buf: list[pd.DataFrame] = []
        total = len(pending)
        for position, day in enumerate(pending, start=1):
            bars = self._call("daily", trade_date=day.strftime("%Y%m%d"))
            factors = self._call("adj_factor", trade_date=day.strftime("%Y%m%d"))
            basic = self._call("daily_basic", trade_date=day.strftime("%Y%m%d"))
            if not bars.empty:
                bars = bars[bars["ts_code"].map(_is_a_share)].rename(
                    columns={"ts_code": "symbol", "vol": "volume"}
                )[DAILY_COLUMNS]
                if not basic.empty:
                    aux = basic[basic["ts_code"].map(_is_a_share)].rename(
                        columns={"ts_code": "symbol"}
                    )[["trade_date", "symbol", "turnover_rate", "circ_mv"]]
                    bars = bars.merge(aux, on=["trade_date", "symbol"], how="left")
                    bars["turnover"] = pd.to_numeric(bars["turnover_rate"], errors="coerce")
                    bars["float_market_cap"] = pd.to_numeric(bars["circ_mv"], errors="coerce")
                    bars = bars.drop(columns=["turnover_rate", "circ_mv"])
                else:
                    bars = bars.assign(turnover=np.nan, float_market_cap=np.nan)
                daily_buf.append(bars)
            if not factors.empty:
                factors = factors[factors["ts_code"].map(_is_a_share)].rename(
                    columns={"ts_code": "symbol", "adj_factor": "adjust_factor"}
                )[["trade_date", "symbol", "adjust_factor"]]
                adj_buf.append(factors)
            if position % 500 == 0:
                self._flush_bars(daily_buf, adj_buf)
                daily_buf, adj_buf = [], []
                print(f"  market: {position}/{total} ({day.date()})", flush=True)
        if daily_buf or adj_buf:
            self._flush_bars(daily_buf, adj_buf)
        print(f"  market: {total}/{total} done", flush=True)

        daily = self._read(self.equity / "daily.parquet")
        securities = self._securities(daily)
        self._atomic_write(securities, "securities.parquet")
        self._atomic_write(self._status(calendar, securities, daily), "status.parquet")

    def _resume_date(self) -> pd.Timestamp:
        path = self.equity / "daily.parquet"
        if not path.exists():
            return pd.Timestamp.min
        try:
            dates = pd.read_parquet(path, columns=["trade_date"])["trade_date"]
            return pd.Timestamp(pd.to_datetime(dates).max()).normalize()
        except Exception:  # noqa: BLE001
            return pd.Timestamp.min

    def _flush_bars(self, daily_buf: list[pd.DataFrame], adj_buf: list[pd.DataFrame]) -> None:
        if daily_buf:
            existing = self._read(self.equity / "daily.parquet")
            daily = pd.concat([existing, *daily_buf], ignore_index=True)
            daily["trade_date"] = _norm(daily["trade_date"])
            daily = daily.drop_duplicates(["trade_date", "symbol"], keep="last").reset_index(drop=True)
            self._atomic_write(daily, "daily.parquet")
        if adj_buf:
            existing = self._read(self.equity / "adjustments.parquet")
            adjustments = pd.concat([existing, *adj_buf], ignore_index=True)
            adjustments["trade_date"] = _norm(adjustments["trade_date"])
            adjustments = adjustments.drop_duplicates(
                ["trade_date", "symbol"], keep="last"
            ).reset_index(drop=True)
            self._atomic_write(adjustments, "adjustments.parquet")

    def _calendar(self) -> pd.DataFrame:
        frame = self._call(
            "trade_cal", exchange="SSE",
            start_date=self.start.strftime("%Y%m%d"), end_date=self.end.strftime("%Y%m%d"),
        )
        frame = frame[["cal_date", "is_open"]].rename(columns={"cal_date": "trade_date"})
        frame["trade_date"] = _norm(frame["trade_date"])
        frame["is_open"] = pd.to_numeric(frame["is_open"], errors="coerce").eq(1).astype(bool)
        return frame.drop_duplicates("trade_date").sort_values("trade_date").reset_index(drop=True)

    def _securities(self, daily: pd.DataFrame) -> pd.DataFrame:
        listed = self._call("stock_basic", list_status="L")
        delisted = self._call("stock_basic", list_status="D")
        frame = pd.concat([listed, delisted], ignore_index=True)
        frame = frame[frame["ts_code"].map(_is_a_share)].copy()
        frame["symbol"] = frame["ts_code"]
        frame["list_date"] = _norm(frame["list_date"])
        # stock_basic has no delist_date; approximate it from the last daily bar.
        last_bar = (
            daily.groupby("symbol", observed=True)["trade_date"].max().rename("delist_date")
            if not daily.empty and "trade_date" in daily else pd.Series(dtype="datetime64[ns]")
        )
        frame = frame.merge(last_bar, on="symbol", how="left")
        frame["delist_date"] = frame["delist_date"].where(
            frame["ts_code"].isin(set(delisted["ts_code"])), pd.NaT
        )
        return (
            frame[["symbol", "list_date", "delist_date"]]
            .drop_duplicates("symbol", keep="last")
            .sort_values("symbol")
            .reset_index(drop=True)
        )

    def _status(
        self, calendar: pd.DataFrame, securities: pd.DataFrame, daily: pd.DataFrame
    ) -> pd.DataFrame:
        if daily.empty:
            placeholder = pd.Timestamp(self.start)
            return pd.DataFrame([{
                "trade_date": placeholder, "symbol": "000001.SZ", "is_st": False, "is_pt": False,
                "is_suspended": False, "limit_up": False, "limit_down": False,
            }], columns=STATUS_COLUMNS)
        open_arr = calendar.loc[calendar["is_open"], "trade_date"].to_numpy(dtype="datetime64[ns]")
        daily_max = np.datetime64(pd.Timestamp(daily["trade_date"].max()).to_datetime64())
        open_arr = open_arr[open_arr <= daily_max]  # never flag the not-yet-published last day
        traded = {
            symbol: group["trade_date"].to_numpy(dtype="datetime64[ns]")
            for symbol, group in daily.groupby("symbol", observed=True)
        }
        end64 = daily_max
        rows: list[dict[str, Any]] = []
        for symbol, list_date, delist_date in zip(
            securities["symbol"], securities["list_date"], securities["delist_date"],
            strict=False,
        ):
            if pd.isna(list_date):
                continue
            start64 = np.datetime64(pd.Timestamp(list_date).to_datetime64())
            stop64 = end64 if pd.isna(delist_date) else np.datetime64(pd.Timestamp(delist_date).to_datetime64())
            expected = open_arr[(open_arr >= start64) & (open_arr <= stop64)]
            present = traded.get(symbol)
            suspended = expected if present is None else expected[~np.isin(expected, present)]
            rows.extend(
                {
                    "trade_date": pd.Timestamp(day), "symbol": symbol, "is_st": False,
                    "is_pt": False, "is_suspended": True, "limit_up": False, "limit_down": False,
                }
                for day in suspended
            )
        if not rows:
            rows = [{
                "trade_date": pd.Timestamp(open_arr[0]), "symbol": "000001.SZ", "is_st": False,
                "is_pt": False, "is_suspended": False, "limit_up": False, "limit_down": False,
            }]
        return pd.DataFrame(rows, columns=STATUS_COLUMNS).sort_values(
            ["trade_date", "symbol"]
        ).reset_index(drop=True)

    # -- fundamentals -----------------------------------------------------
    def run_fundamentals(self) -> None:
        columns = [
            "symbol", "stat_date", "available_date", "eps", "bps", "revenue",
            "ocfps", "net_profit", "roe", "gross_margin",
        ]
        existing = self._read(self.equity / "fundamentals.parquet")
        acc = existing if not existing.empty else pd.DataFrame(columns=columns)
        done = set(acc["symbol"]) if not acc.empty else set()
        symbols = self._a_share_symbols()
        chunks: list[pd.DataFrame] = []
        start = self.start.strftime("%Y%m%d")
        end = self.end.strftime("%Y%m%d")
        for position, code in enumerate(symbols, start=1):
            if code in done:
                continue
            indicator = self._call("fina_indicator", ts_code=code, start_date=start, end_date=end)
            income = self._call("income", ts_code=code, start_date=start, end_date=end)
            merged = self._merge_fundamental(indicator, income)
            if not merged.empty:
                chunks.append(merged)
            if len(chunks) >= 300:
                acc = self._flush_fundamentals(acc, chunks, columns)
                chunks = []
                print(f"  fundamentals: {position}/{len(symbols)}", flush=True)
        if chunks:
            acc = self._flush_fundamentals(acc, chunks, columns)
        self._atomic_write(acc, "fundamentals.parquet")
        print(f"  fundamentals: {len(acc)} rows", flush=True)

    def _a_share_symbols(self) -> list[str]:
        listed = self._call("stock_basic", list_status="L")
        delisted = self._call("stock_basic", list_status="D")
        frame = pd.concat([listed, delisted], ignore_index=True)
        return sorted(frame.loc[frame["ts_code"].map(_is_a_share), "ts_code"].unique())

    @staticmethod
    def _merge_fundamental(indicator: pd.DataFrame, income: pd.DataFrame) -> pd.DataFrame:
        ind = indicator.rename(columns={"ts_code": "symbol"}) if not indicator.empty else pd.DataFrame()
        if not income.empty and "report_type" in income and (income["report_type"] == "1").any():
            income = income[income["report_type"] == "1"]
        inc = (
            income.rename(columns={"ts_code": "symbol"})[
                ["symbol", "end_date", "ann_date", "revenue", "n_income_attr_p"]
            ]
            if not income.empty else pd.DataFrame()
        )
        if ind.empty and inc.empty:
            return pd.DataFrame()
        merged = pd.merge(ind, inc, on=["symbol", "end_date"], how="outer", suffixes=("", "_inc"))
        merged["available_date"] = pd.to_datetime(
            merged[["ann_date", "ann_date_inc"]].max(axis=1), errors="coerce"
        ).dt.normalize()
        return pd.DataFrame({
            "symbol": merged["symbol"],
            "stat_date": _norm(merged["end_date"]),
            "available_date": merged["available_date"],
            "eps": pd.to_numeric(merged["eps"], errors="coerce"),
            "bps": pd.to_numeric(merged["bps"], errors="coerce"),
            "revenue": pd.to_numeric(merged["revenue"], errors="coerce"),
            "ocfps": pd.to_numeric(merged["ocfps"], errors="coerce"),
            "net_profit": pd.to_numeric(merged["n_income_attr_p"], errors="coerce"),
            "roe": pd.to_numeric(merged["roe"], errors="coerce"),
            "gross_margin": pd.to_numeric(merged["grossprofit_margin"], errors="coerce"),
        })

    @staticmethod
    def _flush_fundamentals(
        acc: pd.DataFrame, chunks: list[pd.DataFrame], columns: list[str]
    ) -> pd.DataFrame:
        fresh = pd.concat(chunks, ignore_index=True)
        combined = pd.concat([acc, fresh], ignore_index=True)
        return (
            combined.dropna(subset=["symbol", "stat_date"])
            .drop_duplicates(["symbol", "stat_date"], keep="last")
            .sort_values(["symbol", "stat_date"])
            .reset_index(drop=True)[columns]
        )

    # -- membership / industries -----------------------------------------
    def run_membership(self) -> None:
        # index_weight caps a wide range to the latest ~24 months, so fetch
        # one month at a time to get full point-in-time history.
        month_starts = pd.date_range("2014-01-01", self.end, freq="MS")
        raw: list[dict[str, Any]] = []
        for index_code in INDEX_CODES:
            for position, month_start in enumerate(month_starts, start=1):
                month_end = month_start + pd.offsets.MonthEnd(0)
                frame = self._call(
                    "index_weight", index_code=index_code,
                    start_date=month_start.strftime("%Y%m%d"),
                    end_date=month_end.strftime("%Y%m%d"),
                )
                for _, row in frame.iterrows():
                    raw.append({
                        "index_code": index_code, "symbol": row["con_code"],
                        "valid_from": pd.Timestamp(row["trade_date"]).normalize(),
                        "benchmark_weight": pd.to_numeric(row["weight"], errors="coerce"),
                    })
                if position % 20 == 0 or position == len(month_starts):
                    print(f"  index_weight {index_code}: {position}/{len(month_starts)}", flush=True)
        weights = pd.DataFrame(raw).drop_duplicates(
            ["index_code", "symbol", "valid_from"]
        ).sort_values(["index_code", "symbol", "valid_from"])
        weight_rows: list[dict[str, Any]] = []
        for (index_code, symbol), group in weights.groupby(["index_code", "symbol"], observed=True):
            dates = group["valid_from"].to_numpy()
            for position, (_, row) in enumerate(group.iterrows()):
                weight_rows.append({
                    "index_code": index_code, "symbol": symbol,
                    "valid_from": pd.Timestamp(row["valid_from"]).normalize(),
                    "valid_to": (
                        pd.Timestamp(dates[position + 1]) - pd.Timedelta(days=1)
                        if position + 1 < len(dates) else pd.NaT
                    ),
                    "benchmark_weight": row["benchmark_weight"],
                })
        members = pd.DataFrame(weight_rows, columns=[
            "index_code", "symbol", "valid_from", "valid_to", "benchmark_weight",
        ]).sort_values(["index_code", "symbol", "valid_from"]).reset_index(drop=True)
        # index_weight can lag a delisting by a few days and still list the stock
        # at a snapshot after its last bar; drop those stale rows and cap the
        # open-ended last membership at the delist date.
        securities_path = self.equity / "securities.parquet"
        if securities_path.exists():
            delist = pd.read_parquet(securities_path)[["symbol", "delist_date"]]
            delist["delist_date"] = pd.to_datetime(delist["delist_date"])
            members = members.merge(delist, on="symbol", how="left")
            members = members[
                members["delist_date"].isna() | (members["delist_date"] >= members["valid_from"])
            ]
            open_ended = members["valid_to"].isna() & members["delist_date"].notna()
            members.loc[open_ended, "valid_to"] = members.loc[open_ended, "delist_date"]
            members = members.drop(columns=["delist_date"]).reset_index(drop=True)
        self._atomic_write(members, "index_members.parquet")

        classify = self._call("index_classify", level="L1", src="SW2021")
        name_by_code = dict(zip(classify["index_code"], classify["industry_name"], strict=False))
        industry_rows: list[dict[str, Any]] = []
        for index_code, industry_name in name_by_code.items():
            frame = self._call("index_member", index_code=index_code)
            if frame.empty:
                continue
            for _, row in frame.iterrows():
                industry_rows.append({
                    "symbol": row["con_code"],
                    "industry_code": index_code,
                    "industry_name": industry_name,
                    "valid_from": pd.Timestamp(row["in_date"]).normalize() if pd.notna(row["in_date"]) else pd.NaT,
                    "valid_to": pd.Timestamp(row["out_date"]).normalize() if pd.notna(row["out_date"]) else pd.NaT,
                    "source": "tushare:index_member",
                    "version": "SW2021",
                })
        industries = self._resolve_industries(industry_rows)
        industries = self._fill_industry_gaps(industries)
        industries = self._assign_unknown_industry(industries)
        self._atomic_write(industries, "industries.parquet")

    def _assign_unknown_industry(self, industries: pd.DataFrame) -> pd.DataFrame:
        """Give a transparent UNKNOWN industry to A-shares tushare never classified.

        A few instruments (e.g. CDRs like 689009.SH) trade on SH/SZ/BJ and enter
        CSI indices but have no SW1 ``index_member`` record.  Assign one explicit
        UNKNOWN interval so they remain in the universe instead of silently
        dropping out of industry-neutral research.
        """
        securities_path = self.equity / "securities.parquet"
        if not securities_path.exists():
            return industries
        sec = pd.read_parquet(securities_path)[["symbol", "list_date", "delist_date"]]
        covered = set(industries["symbol"]) if not industries.empty else set()
        missing = sec[~sec["symbol"].isin(covered)]
        rows: list[dict[str, Any]] = []
        for _, row in missing.iterrows():
            rows.append({
                "symbol": row["symbol"],
                "industry_code": "UNKNOWN",
                "industry_name": "UNKNOWN",
                "valid_from": pd.Timestamp(row["list_date"]).normalize()
                if pd.notna(row["list_date"]) else pd.Timestamp("1990-01-01"),
                "valid_to": pd.Timestamp(row["delist_date"]).normalize()
                if pd.notna(row["delist_date"]) else pd.NaT,
                "source": "fallback",
                "version": "UNKNOWN",
            })
        if not rows:
            return industries
        fallback = pd.DataFrame(rows, columns=list(industries.columns))
        return pd.concat([industries, fallback], ignore_index=True).sort_values(
            ["symbol", "valid_from"]
        ).reset_index(drop=True)

    def _fill_industry_gaps(self, industries: pd.DataFrame) -> pd.DataFrame:
        """Carry the known SW1 assignment across tushare's PIT gaps.

        tushare's SW2021 ``index_member`` re-bases many ``in_date`` values to
        2014-02-21 and can leave small holes at industry transitions.  For each
        symbol, extend the earliest record back to its list date and bridge any
        internal gap with the preceding industry (carry-forward).
        """
        if industries.empty:
            return industries
        securities_path = self.equity / "securities.parquet"
        list_map: dict[str, pd.Timestamp] = {}
        delist_map: dict[str, pd.Timestamp] = {}
        if securities_path.exists():
            sec = pd.read_parquet(securities_path)
            list_map = dict(zip(sec["symbol"], pd.to_datetime(sec["list_date"]), strict=False))
            delist_map = dict(
                zip(sec["symbol"], pd.to_datetime(sec["delist_date"]), strict=False)
            )
        parts: list[pd.DataFrame] = []
        for symbol, group in industries.groupby("symbol", observed=True):
            group = group.sort_values("valid_from").reset_index(drop=True)
            list_date = list_map.get(symbol)
            if pd.notna(list_date) and pd.Timestamp(list_date) < group.loc[0, "valid_from"]:
                group.loc[0, "valid_from"] = pd.Timestamp(list_date).normalize()
            for position in range(len(group) - 1):
                next_start = group.loc[position + 1, "valid_from"]
                current_end = group.loc[position, "valid_to"]
                if pd.isna(current_end) or current_end < next_start - pd.Timedelta(days=1):
                    group.loc[position, "valid_to"] = next_start - pd.Timedelta(days=1)
            # Trailing gap: carry the last known industry through suspension until
            # delisting (or open-ended if still listed).
            last = len(group) - 1
            delist_date = delist_map.get(symbol)
            if pd.isna(delist_date):
                group.loc[last, "valid_to"] = pd.NaT
            else:
                delist = pd.Timestamp(delist_date)
                current = group.loc[last, "valid_to"]
                if pd.isna(current) or delist > current:
                    group.loc[last, "valid_to"] = delist
            parts.append(group)
        return pd.concat(parts, ignore_index=True).sort_values(
            ["symbol", "valid_from"]
        ).reset_index(drop=True)

    @staticmethod
    def _resolve_industries(rows: list[dict[str, Any]]) -> pd.DataFrame:
        """Resolve overlapping SW1 membership records into non-overlapping PIT intervals.

        tushare ``index_member`` keeps one continuous record per (stock, index) even
        when a stock leaves and re-enters an industry, so records can overlap.  Sweep
        all in/out events per symbol and keep, at every instant, the membership with
        the latest ``in_date`` (the most recent re-classification).
        """
        if not rows:
            return pd.DataFrame(columns=[
                "symbol", "industry_code", "industry_name", "valid_from", "valid_to",
                "source", "version",
            ])
        frame = pd.DataFrame(rows)
        resolved: list[dict[str, Any]] = []
        for symbol, group in frame.groupby("symbol", observed=True):
            events: list[tuple[pd.Timestamp, int, str, str, pd.Timestamp]] = []
            for _, row in group.iterrows():
                start = row["valid_from"]
                if pd.isna(start):
                    continue
                key = (str(row["industry_code"]), str(row["industry_name"]))
                events.append((start, +1, key[0], key[1], start))
                if pd.notna(row["valid_to"]):
                    events.append((row["valid_to"] + pd.Timedelta(days=1), -1, key[0], key[1], start))
            events.sort(key=lambda item: item[0])
            active: dict[tuple[str, str], pd.Timestamp] = {}
            current: tuple[str, str] | None = None
            current_start: pd.Timestamp | None = None
            for event_date, delta, code, name, in_date in events:
                new_active = dict(active)
                if delta == +1:
                    new_active[(code, name)] = in_date
                else:
                    new_active.pop((code, name), None)
                winner = max(new_active.items(), key=lambda item: item[1])[0] if new_active else None
                if winner != current:
                    if current is not None and current_start is not None and event_date > current_start:
                        resolved.append({
                            "symbol": symbol, "industry_code": current[0],
                            "industry_name": current[1],
                            "valid_from": current_start, "valid_to": event_date - pd.Timedelta(days=1),
                            "source": "tushare:index_member", "version": "SW2021",
                        })
                    current, current_start = winner, event_date
                active = new_active
            if current is not None and current_start is not None:
                resolved.append({
                    "symbol": symbol, "industry_code": current[0], "industry_name": current[1],
                    "valid_from": current_start, "valid_to": pd.NaT,
                    "source": "tushare:index_member", "version": "SW2021",
                })
        return pd.DataFrame(resolved, columns=[
            "symbol", "industry_code", "industry_name", "valid_from", "valid_to", "source", "version",
        ]).sort_values(["symbol", "valid_from"]).reset_index(drop=True)

    # -- metadata ---------------------------------------------------------
    def write_metadata(self) -> None:
        metadata = {
            "formal": True,
            "source": "tushare",
            "watermark": "tushare daily+adj_factor+daily_basic+fina_indicator+income+"
                         "index_weight+index_member(SW2021)",
            "industry_snapshot_only": False,
            "benchmark_weights_imputed_equal": False,
            "limit_status_unavailable": True,
            "st_status_unavailable": True,
            "sw1_gap_fill": "tushare SW2021 index_member in/out dates re-based and gapped; "
                            "resolved to non-overlapping PIT via sweep + carry-forward/backfill",
            "industry_unknown_fallback": "unclassified instruments (e.g. CDR 689009.SH) assigned UNKNOWN",
            "units": {"volume": "手", "amount": "千元", "turnover": "%", "float_market_cap": "万元"},
            "adjustment": "backward cumulative; adj_price = raw_price * adjust_factor",
            "start": str(self.start.date()),
            "end": str(self.end.date()),
        }
        (self.equity / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def run_all(self) -> None:
        self.run_market()
        self.run_fundamentals()
        self.run_membership()
        self.write_metadata()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import Tushare Pro data into the equity/ Parquet contract"
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--token")
    parser.add_argument("--start", default="2012-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--rate-limit", type=float, default=0.15)
    parser.add_argument("--only", choices=("market", "fundamentals", "membership"), default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    pro = ts.pro_api(_load_token(args.token, root))
    importer = TushareImporter(
        pro, root, start=args.start, end=args.end, rate_limit=args.rate_limit
    )
    if args.only == "market":
        importer.run_market()
    elif args.only == "fundamentals":
        importer.run_fundamentals()
    elif args.only == "membership":
        importer.run_membership()
    else:
        importer.run_all()
    print(f"done -> {importer.equity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
