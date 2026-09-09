from __future__ import annotations

import io
import tarfile
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from alphagym.equity_data import DataContractError
from alphagym.okx_orderbook import _data_member, _normalise_chunk, latest_archive_date


def test_latest_archive_date_has_three_day_lag() -> None:
    now = datetime(2026, 9, 7, 15, tzinfo=UTC)
    assert latest_archive_date(now) == date(2026, 9, 4)


def test_normalise_chunk_preserves_raw_book_and_types_keys() -> None:
    raw = pd.DataFrame({
        "\ufeffinstId": ["BTC-USDT"], "action": ["snapshot"],
        "asks": ['[["1","2","0","1"]]'], "bids": ['[["0.9","3","0","2"]]'],
        "ts": ["1788480000000"], "seqId": ["42"],
    })
    frame = _normalise_chunk(
        raw, inst_id="BTC-USDT", depth=400,
        source_date=date(2026, 9, 4), row_offset=10,
    )
    assert frame.loc[0, "exchange"] == "OKX"
    assert frame.loc[0, "source_row"] == 10
    assert frame.loc[0, "ts"] == 1788480000000
    assert frame.loc[0, "asks"] == '[["1","2","0","1"]]'


def test_data_member_requires_one_data_file(tmp_path) -> None:
    path = tmp_path / "book.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        content = b'{"instId":"BTC-USDT","action":"snapshot","asks":[],"bids":[],"ts":"1"}\n'
        info = tarfile.TarInfo("book.data")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    with tarfile.open(path, "r:gz") as archive:
        assert _data_member(archive).readline().startswith(b'{"instId"')


def test_invalid_depth_is_rejected(monkeypatch) -> None:
    from alphagym import okx_orderbook

    monkeypatch.setattr(okx_orderbook.urllib.request, "urlopen", lambda *a, **k: None)
    with pytest.raises(DataContractError, match="depth"):
        okx_orderbook._download_request("BTC-USDT", date(2026, 9, 1), date(2026, 9, 1), 20)
