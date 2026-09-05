from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from mlquant.factors import REGISTRY

FAMILIES = ("value", "growth", "momentum", "liquidity", "risk", "quality")


def _write_pdf(path: Path, title: str, lines: list[str]) -> None:
    pdf = canvas.Canvas(str(path), pagesize=A4)
    _, height = A4
    pdf.setTitle(title)
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(48, height - 54, title)
    pdf.setFont("Helvetica", 9)
    y = height - 82
    for line in lines:
        if y < 48:
            pdf.showPage()
            pdf.setFont("Helvetica", 9)
            y = height - 48
        pdf.drawString(48, y, line[:110])
        y -= 13
    pdf.save()


def _write_markdown(path: Path, title: str, body: str, smoke: bool) -> None:
    watermark = "> **非正式 SMOKE 产物：不得用于选模、业绩展示或投资结论。**\n\n" if smoke else ""
    path.write_text(f"# {title}\n\n{watermark}{body.rstrip()}\n", encoding="utf-8")


def build_series(output_dir: str | Path, *, smoke: bool, metadata: dict[str, object] | None = None) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = metadata or {}
    factor_rows = [
        {
            "factor_name": spec.name, "hypothesis_id": spec.hypothesis_id, "family": spec.family,
            "direction": spec.expected_direction, "lookback_days": spec.lookback_days,
            "min_observations": spec.min_observations, "formula_version": spec.formula_version,
        }
        for spec in REGISTRY.list()
    ]
    catalog = pd.DataFrame(factor_rows)
    catalog.to_csv(output / "factor_catalog.csv", index=False, encoding="utf-8-sig")
    catalog.to_parquet(output / "factor_catalog.parquet", index=False)

    framework = (
        "样本分为2014–2020开发、2021–2023验证、2024–2025测试；2026不参与首期选模。\n\n"
        "截面依次执行有效池、5×MAD、标准化、申万一级行业与对数流通市值WLS中性化。"
        "组合在行业内五层，并严格匹配同期宽基行业权重；月末信号在下一交易日开盘成交。"
    )
    reports: list[tuple[str, str, str]] = [("framework", "框架总览", framework)]
    for family in FAMILIES:
        subset = catalog[catalog["family"] == family]
        columns = ["factor_name", "hypothesis_id", "direction", "lookback_days"]
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        lines = ["| " + " | ".join(map(str, row)) + " |" for row in subset[columns].itertuples(index=False, name=None)]
        table = "\n".join([header, separator, *lines])
        reports.append((family, f"{family} 因子族报告", f"本族登记 {len(subset)} 个具体因子。\n\n{table}"))
    reports.append(("multi_factor", "多因子合成报告", "比较等权、因子收益半衰、IC半衰、最大ICIR、最大IC和PCA。动态参数仅使用过去12个月，半衰期6个月；验证期选择后冻结测试期。"))

    links = []
    for slug, title, body in reports:
        md_path = output / f"{slug}.md"
        _write_markdown(md_path, title, body, smoke)
        _write_pdf(output / f"{slug}.pdf", title, ["MLQuant A-share factor series", f"Report: {slug}", f"Smoke: {smoke}"])
        links.append(f"- [{title}]({slug}.md)")
    _write_markdown(output / "index.md", "MLQuant 因子系列索引", "\n".join(links), smoke)
    _write_pdf(output / "index.pdf", "MLQuant Factor Series Index", [slug for slug, _, _ in reports])

    counts = catalog.groupby("family", observed=True).size().reindex(FAMILIES)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    counts.plot.bar(ax=axis, color="#b21f2d")
    axis.set_title("Registered factors by family")
    axis.set_ylabel("count")
    figure.tight_layout()
    figure.savefig(output / "factor_catalog.png", dpi=160)
    plt.close(figure)

    manifest: dict[str, object] = {
        "formal": not smoke, "watermark": "NON-FORMAL SMOKE" if smoke else None,
        "factor_count": len(catalog), "metadata": metadata, "files": {},
    }
    for path in sorted(output.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path
