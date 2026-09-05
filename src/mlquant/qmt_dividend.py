"""Read QMT's ex-rights / dividend metadata from the local ``DividData`` LevelDB.

QMT stores per-security dividend/ex-rights records in ``{datadir}/DividData`` as a
LevelDB database (no xtquant dependency required).  Each 96-byte record carries the
single-event backward-adjustment factor (后复权因子) plus the ex-date, which is exactly
the ``adjust_factor`` the equity contract needs.

Record layout (96 bytes, little-endian):
    bytes  8..15  int64   ex-dividend date in milliseconds since epoch (Shanghai midnight)
    bytes 72..79  float64 single-event backward-adjustment factor (>= 1)
    bytes 80..87  int64   record date encoded as YYYYMMDD (0 when absent)
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

_MAGIC = bytes([0x57, 0xFB, 0x80, 0x8B, 0x24, 0x75, 0x47, 0xDB])
_RECORD_SIZE = 96
_FACTOR_OFFSET = 72
_FACTOR_DTYPE = np.dtype("<f8")


# --------------------------------------------------------------------------- LevelDB

def _u32(b: bytes, off: int) -> int:
    return int.from_bytes(b[off : off + 4], "little")


def _decode_varint(b: bytes, pos: int) -> tuple[int, int]:
    shift = 0
    result = 0
    while True:
        byte = b[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result, pos
        shift += 7


def _read_varstr(b: bytes, pos: int) -> tuple[bytes, int]:
    n, pos = _decode_varint(b, pos)
    return b[pos : pos + n], pos + n


_TAG_COMPARATOR = 1
_TAG_LOG_NUMBER = 2
_TAG_NEXT_FILE = 3
_TAG_LAST_SEQ = 4
_TAG_COMPACT_POINTER = 5
_TAG_DELETED_FILE = 6
_TAG_NEW_FILE = 7
_TAG_PREV_LOG = 9


def _iter_log_records(data: bytes) -> Iterator[bytes]:
    pos = 0
    buf = bytearray()
    while pos + 7 <= len(data):
        length = int.from_bytes(data[pos + 4 : pos + 6], "little")
        rtype = data[pos + 6]
        pos += 7
        if length == 0:
            break
        chunk = data[pos : pos + length]
        pos += length
        if rtype == 1:
            yield bytes(buf + chunk)
            buf.clear()
        elif rtype == 2:
            buf.extend(chunk)
        elif rtype == 4:
            buf.extend(chunk)
            yield bytes(buf)
            buf.clear()
        else:  # MIDDLE
            buf.extend(chunk)


def _parse_version_edit(payload: bytes) -> tuple[set[int], list[int]]:
    """Return (deleted_file_numbers, new_file_numbers) for one VersionEdit."""
    deleted: set[int] = set()
    added: list[int] = []
    pos = 0
    while pos < len(payload):
        tag = payload[pos]
        pos += 1
        if tag == _TAG_COMPARATOR:
            _, pos = _read_varstr(payload, pos)
        elif tag in (_TAG_LOG_NUMBER, _TAG_NEXT_FILE, _TAG_LAST_SEQ, _TAG_PREV_LOG):
            _, pos = _decode_varint(payload, pos)
        elif tag == _TAG_COMPACT_POINTER:
            _, pos = _decode_varint(payload, pos)
            _, pos = _read_varstr(payload, pos)
        elif tag == _TAG_DELETED_FILE:
            _, pos = _decode_varint(payload, pos)
            number, pos = _decode_varint(payload, pos)
            deleted.add(number)
        elif tag == _TAG_NEW_FILE:
            _, pos = _decode_varint(payload, pos)
            number, pos = _decode_varint(payload, pos)
            _, pos = _decode_varint(payload, pos)  # size
            _, pos = _read_varstr(payload, pos)  # smallest key
            _, pos = _read_varstr(payload, pos)  # largest key
            added.append(number)
        else:
            raise ValueError(f"unknown version-edit tag {tag}")
    return deleted, added


def _live_sstables(manifest: Path) -> set[int]:
    live: set[int] = set()
    for payload in _iter_log_records(manifest.read_bytes()):
        deleted, added = _parse_version_edit(payload)
        live -= deleted
        live.update(added)
    return live


def _parse_block(data: bytes) -> list[tuple[bytes, bytes]]:
    num_restarts = _u32(data, len(data) - 4)
    restart_off = len(data) - 4 - num_restarts * 4
    pos = 0
    key = b""
    entries: list[tuple[bytes, bytes]] = []
    while pos < restart_off:
        shared, pos = _decode_varint(data, pos)
        non_shared, pos = _decode_varint(data, pos)
        value_len, pos = _decode_varint(data, pos)
        key = key[:shared] + data[pos : pos + non_shared]
        pos += non_shared
        value = data[pos : pos + value_len]
        pos += value_len
        entries.append((key, value))
    return entries


def _iter_sstable(path: Path) -> Iterator[tuple[bytes, bytes]]:
    data = path.read_bytes()
    footer = data[-48:]
    if footer[-8:] != _MAGIC:
        raise ValueError(f"bad LevelDB magic in {path}")
    pos = 0
    _, pos = _decode_varint(footer, pos)  # metaindex offset
    _, pos = _decode_varint(footer, pos)  # metaindex size
    idx_off, pos = _decode_varint(footer, pos)
    idx_size, pos = _decode_varint(footer, pos)

    def read_block(off: int, size: int) -> bytes:
        raw = data[off : off + size + 5]  # handle.size excludes the 5-byte trailer
        if raw[-5] != 0:
            raise ValueError("snappy-compressed LevelDB block is not supported")
        return raw[:-5]

    for _last_key, handle in _parse_block(read_block(idx_off, idx_size)):
        hpos = 0
        off, hpos = _decode_varint(handle, hpos)
        size, hpos = _decode_varint(handle, hpos)
        yield from _parse_block(read_block(off, size))


def iter_leveldb_records(datadir: Path) -> Iterator[tuple[bytes, bytes]]:
    """Iterate every live key/value pair in a LevelDB directory."""
    manifest_name = (datadir / "CURRENT").read_text().strip()
    for number in sorted(_live_sstables(datadir / manifest_name)):
        sst = datadir / f"{number:06d}.ldb"
        if sst.exists():
            yield from _iter_sstable(sst)


# --------------------------------------------------------------------------- DividData

def parse_dividend_key(key: bytes) -> tuple[str, str, str, int] | None:
    """Parse ``b'{market}|{code}|{period}|{ex_date_ms}...'`` -> (market, code, period, ms)."""
    parts = key.split(b"|")
    if len(parts) < 4:
        return None
    digits = b""
    for ch in parts[3]:
        if 0x30 <= ch <= 0x39:
            digits += bytes([ch])
        else:
            break
    if not digits:
        return None
    try:
        return parts[0].decode(), parts[1].decode(), parts[2].decode(), int(digits)
    except UnicodeDecodeError:
        return None


def _ex_date(ms: int) -> pd.Timestamp:
    return (
        pd.to_datetime(ms, unit="ms", utc=True)
        .tz_convert("Asia/Shanghai")
        .normalize()
        .tz_localize(None)
    )


def read_dividend_events(datadir: Path) -> pd.DataFrame:
    """Extract single-event backward-adjustment factors from ``DividData``.

    Returns columns: ``symbol``, ``ex_date``, ``factor`` (one row per ex-rights event).
    """
    suffix = {"SH": ".SH", "SZ": ".SZ", "BJ": ".BJ"}
    rows: list[tuple[str, pd.Timestamp, float]] = []
    for key, value in iter_leveldb_records(datadir):
        if len(value) != _RECORD_SIZE:
            continue
        parsed = parse_dividend_key(key)
        if parsed is None or parsed[2] != "4000":
            continue
        market, code, _period, ms = parsed
        factor = float(np.frombuffer(value[_FACTOR_OFFSET : _FACTOR_OFFSET + 8], _FACTOR_DTYPE)[0])
        if factor <= 0:
            continue
        symbol = f"{code}{suffix.get(market, '.' + market)}"
        rows.append((symbol, _ex_date(ms), factor))
    frame = pd.DataFrame(rows, columns=["symbol", "ex_date", "factor"])
    frame = frame.sort_values(["symbol", "ex_date"], kind="stable")
    # QMT stores some events under two keys with byte-identical values; keep the newest.
    return frame.drop_duplicates(["symbol", "ex_date"], keep="last").reset_index(drop=True)


def build_adjustments(events: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-event factors into a cumulative backward-adjustment series.

    Returns ``trade_date``, ``symbol``, ``adjust_factor`` where the factor is the product
    of every event on or before ``trade_date`` (so ``adj_close = close * adjust_factor``).
    """
    cumulative = (
        events.groupby("symbol", sort=False)["factor"]
        .cumprod()
        .rename("adjust_factor")
    )
    out = events.assign(trade_date=events["ex_date"], adjust_factor=cumulative)[
        ["trade_date", "symbol", "adjust_factor"]
    ]
    return out.reset_index(drop=True)
