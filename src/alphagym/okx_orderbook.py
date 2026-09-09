"""Download official OKX historical L2 order-book archives into ClickHouse."""
from __future__ import annotations

import io
import json
import tarfile
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from alphagym import storage_io
from alphagym.config import resolve_root
from alphagym.equity_data import DataContractError

DOWNLOAD_LINK_URL = "https://www.okx.com/priapi/v5/broker/public/trade-data/download-link"
ARCHIVE_LAG_DAYS = 3


def latest_archive_date(now: datetime | None = None) -> date:
    """Return the newest UTC date expected on OKX's historical-data page."""
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return current.astimezone(UTC).date() - timedelta(days=ARCHIVE_LAG_DAYS)


def _utc_end_ms(value: date) -> str:
    end = datetime.combine(value + timedelta(days=1), datetime.min.time(), UTC)
    return str(int(end.timestamp() * 1000) - 1)


def _download_request(inst_id: str, start: date, end: date, depth: int) -> dict:
    if depth not in {400, 5000}:
        raise DataContractError("OKX historical order-book depth must be 400 or 5000")
    module = "4" if depth == 400 else "5"
    payload = {
        "module": module,
        "instType": "SPOT",
        "instQueryParam": {"instIdList": [inst_id]},
        "dateQuery": {
            "dateAggrType": "daily",
            "begin": _utc_end_ms(start),
            "end": _utc_end_ms(end),
        },
    }
    request = urllib.request.Request(
        DOWNLOAD_LINK_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "AlphaGYM/0.2"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    if result.get("code") != "0":
        raise DataContractError(f"OKX archive lookup failed: {result.get('msg') or 'unknown error'}")
    groups = result.get("data", {}).get("details", [])
    if not groups:
        raise DataContractError("OKX returned no historical order-book archives")
    files = groups[0].get("groupDetails", [])
    return {item["filename"]: item for item in files}


def _download(url: str, destination: Path, *, workers: int = 8) -> None:
    """Download a CDN object concurrently, bypassing slow system HTTP proxies."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    head = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "AlphaGYM/0.2"})
    with opener.open(head, timeout=30) as response:
        size = int(response.headers["Content-Length"])
    with destination.open("wb") as target:
        target.truncate(size)

    def fetch(part: int) -> None:
        start = size * part // workers
        end = size * (part + 1) // workers - 1
        position = start
        failures = 0
        with destination.open("r+b") as target:
            while position <= end:
                attempt_start = position
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "AlphaGYM/0.2",
                        "Range": f"bytes={position}-{end}",
                    },
                )
                try:
                    with opener.open(request, timeout=120) as source:
                        if source.status != 206:
                            raise DataContractError(
                                "OKX archive CDN did not honor a byte-range request"
                            )
                        target.seek(position)
                        while position <= end:
                            block = source.read(min(1024 * 1024, end - position + 1))
                            if not block:
                                break
                            target.write(block)
                            position += len(block)
                    failures = failures + 1 if position == attempt_start else 0
                    if failures >= 8:
                        raise DataContractError("OKX archive download ended early")
                except (OSError, TimeoutError):
                    failures += 1
                    if failures >= 8:
                        raise DataContractError("OKX archive download failed after retries") from None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, range(workers)))


def _data_member(archive: tarfile.TarFile):
    members = [m for m in archive if m.isfile() and m.name.lower().endswith(".data")]
    if len(members) != 1:
        raise DataContractError("OKX archive must contain exactly one data file")
    stream = archive.extractfile(members[0])
    if stream is None:
        raise DataContractError("Cannot read data file from OKX archive")
    return stream


def _normalise_chunk(frame: pd.DataFrame, *, inst_id: str, depth: int,
                     source_date: date, row_offset: int) -> pd.DataFrame:
    frame.columns = [str(column).lstrip("\ufeff").strip() for column in frame.columns]
    frame.insert(0, "exchange", "OKX")
    if "instId" not in frame:
        frame.insert(1, "instId", inst_id)
    frame["depth"] = depth
    frame["source_date"] = pd.Timestamp(source_date)
    frame["source_row"] = pd.Series(
        range(row_offset, row_offset + len(frame)), index=frame.index, dtype="int64"
    )
    for column in ("ts", "seqId", "prevSeqId", "checksum"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    return frame


def _import_archive(archive_path: Path, *, root: Path, inst_id: str, depth: int,
                    source_date: date, chunk_size: int) -> dict:
    resource = (
        f"crypto/okx/order_book/{inst_id}/{depth}/"
        f"{source_date.isoformat()}.parquet"
    )
    store = storage_io.store_for(root, initialize=True)
    rows = 0
    with tarfile.open(archive_path, "r:gz") as archive, store.batch() as batch:
        stream = _data_member(archive)
        with io.TextIOWrapper(stream, encoding="utf-8", newline="") as text:
            for raw in pd.read_json(text, lines=True, chunksize=chunk_size, dtype=False):
                frame = _normalise_chunk(
                    raw, inst_id=inst_id, depth=depth,
                    source_date=source_date, row_offset=rows,
                )
                batch.frame(resource, frame, mode="replace",
                            keys=("source_date", "source_row"))
                rows += len(frame)
    if not rows:
        raise DataContractError(f"OKX archive is empty: {archive_path.name}")
    return {"path": resource, "rows": rows, "version": batch.version}


def import_okx_orderbook(root=None, *, inst_id="BTC-USDT", days=7,
                         end_date: date | str | None = None, depth=400,
                         chunk_size=50_000) -> dict:
    """Import the most recent complete daily OKX L2 archives.

    Archives are staged one UTC day at a time. Each ClickHouse resource is
    published atomically, so an interrupted multi-day download never exposes a
    partial day.
    """
    if days <= 0 or chunk_size <= 0:
        raise DataContractError("days and chunk_size must be positive")
    root = storage_io.register_root(resolve_root(root))
    end = date.fromisoformat(end_date) if isinstance(end_date, str) else end_date
    maximum = latest_archive_date()
    end = end or maximum
    if end > maximum:
        raise DataContractError(
            f"OKX order-book archives currently end at {maximum.isoformat()} (UTC)"
        )
    start = end - timedelta(days=days - 1)
    links = _download_request(inst_id, start, end, depth)
    results = []
    with tempfile.TemporaryDirectory(prefix="alphagym-okx-") as temp:
        temp_dir = Path(temp)
        for value in (start + timedelta(days=n) for n in range(days)):
            filename = f"{inst_id}-L2orderbook-{depth}lv-{value.isoformat()}.tar.gz"
            item = links.get(filename)
            if item is None:
                raise DataContractError(f"OKX archive is unavailable: {filename}")
            archive_path = temp_dir / filename
            _download(item["url"], archive_path)
            result = _import_archive(
                archive_path, root=root, inst_id=inst_id, depth=depth,
                source_date=value, chunk_size=chunk_size,
            )
            result.update({"date": value.isoformat(), "archive": filename,
                           "compressed_bytes": archive_path.stat().st_size})
            results.append(result)
    return {
        "backend": "clickhouse", "exchange": "OKX", "instrument": inst_id,
        "depth": depth, "start": start.isoformat(), "end": end.isoformat(),
        "days": len(results), "rows": sum(item["rows"] for item in results),
        "resources": results,
    }
