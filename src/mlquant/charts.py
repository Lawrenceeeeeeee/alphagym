"""Portable offline ECharts documents shared by report producers."""
from __future__ import annotations

import html
from importlib.resources import files
from pathlib import Path
from typing import Any

from mlquant import storage_io
from mlquant.serialization import dumps


def write_charts(path: Path, title: str, options: list[dict[str, Any]], *, smoke: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    script = storage_io.read_text(files("mlquant").joinpath("static/echarts.min.js"), encoding="utf-8")
    payload = dumps(options).replace("<", "\\u003c")
    watermark = "<p>NON-FORMAL SMOKE — 不得用于选模或投资结论</p>" if smoke else ""
    storage_io.write_text(path,
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        f'<title>{html.escape(title)}</title><h1>{html.escape(title)}</h1>{watermark}'
        '<main id="charts"></main>'
        f'<script>{script}</script><script type="application/json" id="data">{payload}</script>'
        '<script>const options=JSON.parse(document.getElementById("data").textContent);'
        'const charts=options.map(option=>{const el=document.createElement("div");'
        'el.style.height="600px";document.getElementById("charts").appendChild(el);'
        'const chart=echarts.init(el);chart.setOption(option);return chart;});'
        'window.addEventListener("resize",()=>charts.forEach(chart=>chart.resize()));'
        '</script></html>', encoding="utf-8",
    )
    return path
