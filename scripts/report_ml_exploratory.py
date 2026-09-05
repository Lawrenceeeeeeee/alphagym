"""Offline static report for completed non-neutralized ML experiments."""
from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from mlquant.portfolio_evaluation import performance_stats
from mlquant.research import newey_west_t


def pct(value):
    return f"{value:.2%}" if pd.notna(value) else "—"


def common_test_results(output: Path, summary: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    frames = {}
    starts, ends = [], []
    for row in summary[summary.period.eq("test")].itertuples():
        frame = pd.read_parquet(output / row.frequency / row.method / f"{row.scenario}.parquet")
        frame = frame[frame.period.eq("test")]
        frames[(row.frequency, row.method, row.scenario)] = frame
        starts.append(frame.execution_date.min())
        ends.append(frame.end_date.max())
    start, end = max(starts), min(ends)
    years = (end - start).days / 365.25
    rows = []
    for key, frame in frames.items():
        part = frame[(frame.execution_date >= start) & (frame.end_date <= end)]
        if part.execution_date.min() != start or part.end_date.max() != end:
            raise ValueError("common window boundary is not shared by all schedules")
        annual_periods = len(part) / years
        equal = performance_stats(part.portfolio_return, part.equal_universe_return, periods_per_year=annual_periods)
        rows.append({
            "frequency": key[0], "method": key[1], "scenario": key[2], "period": "test_common_window",
            **performance_stats(part.portfolio_return, part.benchmark_return, periods_per_year=annual_periods),
            "excess_vs_equal_universe": equal["annualized_excess_return"],
            "annual_turnover": part.turnover.sum() / years,
            "annual_explicit_cost": part.explicit_cost.sum() / years,
            "annual_slippage_cost": part.slippage_cost.sum() / years,
            "active_newey_west_t": newey_west_t((1+part.portfolio_return)/(1+part.benchmark_return)-1),
            "start": str(start.date()), "end": str(end.date()),
        })
    result = pd.DataFrame(rows)
    result.to_csv(output / "common_test_summary.csv", index=False, encoding="utf-8-sig")
    return result, f"{start.date()} 至 {end.date()}（共同开盘估值窗口）"


def write_report(output: Path) -> None:
    summary = pd.read_csv(output / "summary.csv")
    expected = 3 * 15 * 4 * 4
    if len(summary) != expected or summary.duplicated(["frequency", "method", "scenario", "period"]).any():
        raise ValueError(f"incomplete results: {len(summary)} rows, expected {expected}")
    common, window_label = common_test_results(output, summary)
    test = common[common.scenario == "base_5bps"].copy()
    gross = common[common.scenario == "gross"].set_index(["frequency", "method"])
    stress = common[common.scenario == "stress_20bps"].set_index(["frequency", "method"])
    validation = summary[(summary.period == "validation") & (summary.scenario == "base_5bps") & summary.method.ne("equal_factor")]
    winners = validation.loc[validation.groupby("frequency").annualized_excess_return.idxmax()].set_index("frequency")
    lines = [
        "# ML 只做多与交易成本：月／周／日频探索性重评",
        "", "> 非正式、无行业中性化、非实盘可成交回测。与此前正式报告不可直接拼接。",
        "", "## 核心口径", "",
        "同一组已有 20 个注册因子，按日滚动计算量价指标；财务数据按 available_date 逐公告回放，TTM 所有依赖只取当时已公开值，再向后保持。财务值不按日插值；价格分母仍随交易日变化。",
        "",
        "因子 5×MAD 截尾及截面标准化，不做行业/市值中性化。股票上市满 250 个交易日，信号日价格和流通市值为正；包含 ST。Top 50 等权，保留 0.5% 现金，仅做多。收盘信号、次交易日开盘成交。日/周/月使用各自的下一持有期标签，并剔除训练期末尚未实现的标签。",
        "",
        "开发 2014–2020 拟合，验证 2021–2023 比较，测试 2024–2025 只展示，2026 单列监控。每频率使用固定随机种子 42、最多 20 万条开发样本，复用原 9 个个股 ML 和 5 个因子权重 ML 的参数。因子权重 ML 的 12/36 期窗口指该频率的期数，非统一日历长度。新增等权因子组合；方向仅用完整开发标签的 IC 确定。",
        "",
        "已有 20 因子池曾依据原验证研究挑选，且我们已看过旧测试结果：本次属于事后重评，不能声称全新盲测；验证排名也存在因子池筛选偏差。没有根据本轮测试/2026 结果调整模型。",
        "",
        "## 仅按验证期选出的模型，在测试期如何", "",
        f"以下测试表和曲线统一使用 {window_label}。验证选模仍用各频率在 2021–2023 内的完整持有期，未按测试重新选模。summary.csv 保留原分段全量结果；common_test_summary.csv 是以下共同窗口比较的来源。", "",
        "|频率|验证期选中（非测试最优）|测试净年化|相对市值基准年化|相对全池等权年化|年双边换手|20 bps 下相对市值基准|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    selected = {}
    for frequency in ("monthly", "weekly", "daily"):
        method = winners.loc[frequency, "method"]
        selected[frequency] = method
        row = test[(test.frequency == frequency) & (test.method == method)].iloc[0]
        severe = stress.loc[(frequency, method)]
        lines.append(f"|{frequency}|{method}|{pct(row.annualized_return)}|{pct(row.annualized_excess_return)}|{pct(row.excess_vs_equal_universe)}|{row.annual_turnover:.1f} 倍|{pct(severe.annualized_excess_return)}|")
    lines += ["", "## 所有方法：测试期，单边滑点 5 bps + 税费", "", "年化超额采用相对净值复利 `(1+组合收益)/(1+基准收益)-1`，不是两个年化收益率相减，也不是风险调整 alpha。", "", "|频率|方法|毛年化|净年化|相对市值基准|相对全池等权|成本拖累（年化百分点）|双边换手/年|开盘标记最大回撤|", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for frequency in ("monthly", "weekly", "daily"):
        for row in test[test.frequency == frequency].sort_values("method").itertuples():
            before = gross.loc[(frequency, row.method), "annualized_return"]
            lines.append(f"|{frequency}|{row.method}|{pct(before)}|{pct(row.annualized_return)}|{pct(row.annualized_excess_return)}|{pct(row.excess_vs_equal_universe)}|{(before-row.annualized_return)*100:.2f}|{row.annual_turnover:.1f}|{pct(row.max_drawdown)}|")
    lines += [
        "", "## 如何解读", "",
        "- 先比较 ML 与同频率 equal_factor，判断复杂模型是否优于简单合成；再比较相对市值加权及全池等权基准，避免把小市值风格暴露当作纯 alpha。",
        "- 日/周频只表示调仓更快，不等同于通常的日内高频交易。毛收益提升若小于换手产生的费用和滑点，提频没有经济价值。",
        "- 不按测试最优模型下结论。45 个方法×频率比较具有多重检验偏差；summary 中的 Newey–West t 值仅为诊断，未做多重检验校正。两年测试不足以确认稳定超额。",
        "", "## 成本与不可忽略的限制", "",
        "每段独立从 100 万元现金开始，计入首次建仓费用，避免训练段剩余资金扭曲测试段的最低佣金负担；佣金双边万三、每笔最低 5 元，卖方印花税按历史日期切换，过户费按历史阶段近似；分别假设单边 0（同时关闭税费）、5、10、20 bps 滑点。gross 表示完全零成本。基准为同一可观测股票池的合成市值/等权组合，未计交易费，不是官方指数收益。",
        "",
        "复权价格只表示可分割的收益敞口单位，不是实际股票手数。未处理整手/科创板最小申报、开盘涨跌停排队、成交量容量和股息税；ST 未剔除。因此即使扣费仍有正超额，也只能说明值得进一步验证，不能推导可成交收益。",
        "",
        "缺失报价的旧持仓不能交易：最多沿用过去开盘价 60 个交易日，此后零估值，重新报价可恢复；此为固定的压力估值规则，不是退市清算价。未用未来是否存在价格来删除亏损标的或重分配权重。回撤只观察调仓开盘，月/周频会遗漏期间回撤；不能直接比较三种频率的日内风险。每段仅纳入起止成交日均在段内的完整持有期，边界跨段收益不纳入该段。",
        "",
        "原行业回填、UNKNOWN 分类、指数成分有效期错误未被带入本次无行业 ALL_A 股票池。财务导入器曾按报告期去重，丢失修订版本的可能性仍待源数据核验。模型若输出常数，回测持现并计入 constant_score_fraction，不伪造股票排序。",
        "",
        "税费来源：[财政部：2023-08-28 起印花税减半](https://www.mof.gov.cn/jrttts/202308/t20230828_3904235.htm)。可补充数据接口：[申万成分历史](https://tushare.pro/document/2?doc_id=335)、[涨跌停价格](https://tushare.pro/document/2?doc_id=183)、[历史 ST](https://tushare.pro/document/2?doc_id=397)。",
        "", "## 文件", "", "summary.csv 包括开发（样本内）、验证、测试、监控四段，以及四档成本；metadata.json 保存方法与限制；每方法目录保存逐期净值、交易成本与训练截止信息。feature_manifest.json 指向可重复使用的离线因子缓存。",
    ]
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    charts = []
    for frequency, method in selected.items():
        frame = pd.read_parquet(output / frequency / method / "base_5bps.parquet")
        frame = frame[(frame.execution_date >= pd.Timestamp(common.start.iloc[0])) & (frame.end_date <= pd.Timestamp(common.end.iloc[0]))]
        series = []
        for column, label in (("portfolio_return", method), ("benchmark_return", "同池市值加权"), ("equal_universe_return", "同池等权")):
            nav = (1+frame[column]).cumprod()
            points = [[str(frame.execution_date.iloc[0].date()), 1.]] + [[str(d.date()), float(v)] for d, v in zip(frame.end_date, nav, strict=True)]
            series.append({"name": label, "type": "line", "showSymbol": False, "data": points})
        charts.append({"title": {"text": frequency + "：验证期所选模型 / 测试净值"}, "tooltip": {"trigger": "axis"}, "legend": {"top": 35}, "grid": {"top": 80, "left": 55, "right": 25, "bottom": 30}, "xAxis": {"type": "time"}, "yAxis": {"type": "value", "scale": True}, "series": series})
    static = output / "static"
    static.mkdir(exist_ok=True)
    shutil.copyfile(Path(__file__).parents[1] / "src/mlquant/static/echarts.min.js", static / "echarts.min.js")
    safe_chart = json.dumps(charts, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    cards = ''.join(f'<div id="chart{i}" class="chart"></div>' for i in range(3))
    table = test[["frequency", "method", "annualized_return", "annualized_excess_return", "excess_vs_equal_universe", "annual_turnover", "max_drawdown"]].to_html(index=False, float_format=lambda v: f"{v:.4f}")
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ML 交易成本重评</title>
<style>body{{font:16px/1.65 system-ui;margin:32px auto;max-width:1250px;padding:0 24px;color:#253345}}.warning{{background:#fff1ce;padding:18px;border-left:5px solid #c48200}}.chart{{height:380px;margin:30px 0}}table{{border-collapse:collapse;font-size:13px;width:100%}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:right}}pre{{white-space:pre-wrap;background:#f5f7fa;padding:24px}}h1{{font-size:28px}}</style>
<h1>ML 只做多：月 / 周 / 日频成本重评</h1><div class="warning">探索性，无行业中性化；不代表可实盘成交收益。下图模型仅按验证期选择；测试期不选模。</div>
{cards}<h2>测试期全方法比较（比例单位）</h2>{table}<h2>完整研究说明</h2><pre>{html.escape(chr(10).join(lines))}</pre>
<script src="static/echarts.min.js"></script><script>const options={safe_chart};const charts=options.map((o,i)=>{{const c=echarts.init(document.getElementById('chart'+i));c.setOption(o);return c;}});addEventListener('resize',()=>charts.forEach(c=>c.resize()));</script></html>'''
    (output / "report.html").write_text(document, encoding="utf-8")
    (output / "validation_selection.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    assert np.isfinite(test.annualized_return).all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_report(args.output)
    print(args.output / "report.html")
