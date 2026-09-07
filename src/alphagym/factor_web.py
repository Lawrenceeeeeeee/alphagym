from __future__ import annotations

import inspect
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import StrictUndefined

from alphagym import storage_io
from alphagym.factor_research_service import AutoRunDataError, FactorResearchService
from alphagym.factor_store import FactorStore
from alphagym.factors.base import FactorDefinition
from alphagym.factors.library import seed_definitions
from alphagym.jobs import start_report
from alphagym.report_spec import COMBINE_METHODS, DEFAULT_SPLITS, parse_spec, resolve_factors
from alphagym.services import create_report as create_report_record

FAMILY_LABELS = {
    "value": "估值",
    "growth": "成长",
    "momentum": "动量",
    "liquidity": "流动性",
    "risk": "波动与风险",
    "quality": "财务质量",
    "technical": "技术指标",
}
FAMILY_ORDER = (
    "value", "growth", "momentum", "risk", "liquidity", "quality", "technical",
)

_PERIODS = {
    "development": (0, "开发期", "2014–2020"),
    "validation": (1, "验证期", "2021–2023"),
    "test": (2, "测试期", "2024–2025"),
    "monitoring_2026": (3, "2026 监控", "不参与选模"),
    "all": (4, "全样本", "汇总"),
}
_METRIC_LABELS = {
    "rank_ic": "Rank IC", "icir": "ICIR", "hac_t": "IC HAC t 值",
    "fama_macbeth_return": "Fama–MacBeth 收益",
    "fama_macbeth_hac_t": "F-M HAC t 值", "coverage": "覆盖率",
    "monotonicity": "分组单调性", "p_value": "p 值", "bh_q_value": "BH q 值",
    "annualized_return": "年化收益", "annualized_volatility": "年化波动",
    "sharpe": "Sharpe", "sortino": "Sortino", "max_drawdown": "最大回撤",
    "calmar": "Calmar", "monthly_win_rate": "月胜率", "mean_turnover": "平均换手",
}
_METRIC_ORDER = {name: position for position, name in enumerate(_METRIC_LABELS)}
_PERCENT_METRICS = {
    "coverage", "annualized_return", "annualized_volatility", "max_drawdown",
    "monthly_win_rate", "mean_turnover", "fama_macbeth_return",
    "annualized_excess_return",
}
_PERFORMANCE_METRICS = {
    "annualized_return", "annualized_volatility", "sharpe", "sortino",
    "max_drawdown", "calmar", "monthly_win_rate", "mean_turnover",
}
_STATUS_ZH = {
    "queued": "排队中", "running": "运行中", "succeeded": "成功",
    "failed": "失败", "cancelled": "已取消",
    "active": "启用", "blocked": "受限", "deprecated": "弃用",
}
_ARTIFACT_LABELS = {
    "factor_values": "因子面板", "monthly_statistics": "月度指标",
    "summary": "汇总表", "manifest": "清单", "layered_nav": "分层净值",
    "layered_performance": "分层绩效", "layered_nav_chart": "净值图",
}
_ARTIFACT_MEDIA = {
    ".parquet": "application/octet-stream", ".csv": "text/csv",
    ".json": "application/json", ".png": "image/png", ".html": "text/html",
    ".md": "text/markdown", ".yaml": "text/yaml", ".yml": "text/yaml",
}
_COMPARE_PERIODS = (
    ("development", "开发期"), ("validation", "验证期"),
    ("test", "测试期"), ("all", "全样本"),
)
_COMPARE_METRICS = (
    "rank_ic", "icir", "hac_t", "annualized_return", "sharpe",
    "max_drawdown", "monthly_win_rate",
)
_ERROR_HINTS = (
    ("no such table", "数据库结构缺失，请确认因子库已初始化且版本一致"),
    ("panel contains none of the locked run factors", "所选因子在回测区间内没有有效观测"),
    ("formal run requires point_in_time_audit_passed", "正式回测需要点位审计通过标记"),
    ("formal run universe mismatch", "正式回测的股票池与配置不一致"),
    ("panel missing columns", "因子面板缺少必要列"),
    ("FileNotFoundError", "所需数据文件缺失"),
    ("unknown factor", "引用了不存在的因子"),
    ("not enough values to unpack", "数据处理结果异常（列数不符）"),
)


def _friendly_error(text: str | None) -> str:
    if not text:
        return ""
    lowered = text.lower()
    for needle, hint in _ERROR_HINTS:
        if needle.lower() in lowered:
            return f"{hint}：{text}"
    return text


def _beijing_time(value: str | None) -> str:
    if not value:
        return ""
    try:
        stamp = datetime.fromisoformat(str(value))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        return stamp.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(value)[:19].replace("T", " ")


def _status_zh(value: object) -> str:
    return _STATUS_ZH.get(str(value), str(value))


def _text_list(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _industry_options(data_root: Path) -> list[dict[str, str]]:
    """SW1 industry codes available in the data lake, for the report form."""
    path = data_root / "equity" / "industries.parquet"
    if not storage_io.exists(path):
        return []
    frame = storage_io.read_frame(path, columns=["industry_code", "industry_name"])
    rows = (
        frame.drop_duplicates("industry_code")
        .sort_values("industry_code")
        .to_dict("records")
    )
    options = []
    for row in rows:
        code = str(row["industry_code"])
        name = str(row["industry_name"])
        if "UNKNOWN" in code.upper() or "UNKNOWN" in name.upper() or not name.strip():
            continue
        options.append({"code": code, "name": name})
    return options


def _format_metric(name: str, value: float | None) -> str:
    if value is None:
        return "—"
    number = float(value)
    if not math.isfinite(number):
        return "—"
    if name in _PERCENT_METRICS:
        return f"{number * 100:.2f}%"
    if number and abs(number) < 0.0001:
        return f"{number:.2e}"
    return f"{number:.4f}"


def _group_factor_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for row in metrics:
        run_id = str(row["run_id"])
        run = runs.setdefault(run_id, {
            "run_id": run_id,
            "created_at": row.get("created_at", ""),
            "mode": row.get("mode", ""),
            "indices": {},
        })
        index_code = str(row.get("index_code") or "全市场")
        index = run["indices"].setdefault(index_code, {
            "index_code": index_code,
            "variants": {},
        })
        value_type = str(row.get("value_type") or "neutralized")
        orientation = str(row.get("orientation") or "original")
        variant_key = (value_type, orientation)
        variant = index["variants"].setdefault(variant_key, {
            "label": (
                ("中性化" if value_type == "neutralized" else "原始值")
                + " · " + ("原方向" if orientation == "original" else "反向")
            ),
            "value_type": value_type,
            "orientation": orientation,
            "periods": {},
            "layered_backtest": None,
        })
        period_key = str(row.get("period") or "all")
        _order, period_label, period_range = _PERIODS.get(
            period_key, (99, period_key, "")
        )
        period = variant["periods"].setdefault(period_key, {
            "key": period_key,
            "label": period_label,
            "range": period_range,
            "metrics": [],
        })
        metric_name = str(row["metric_name"])
        label = _METRIC_LABELS.get(metric_name, metric_name.replace("_", " "))
        scenario = str(row.get("cost_scenario") or "")
        if metric_name in _PERFORMANCE_METRICS:
            label += " · " + ("5bps" if scenario == "base_5bps" else "10bps")
        period["metrics"].append({
            "name": metric_name,
            "label": label,
            "value": _format_metric(metric_name, row.get("metric_value")),
            "negative": row.get("metric_value") is not None
            and float(row["metric_value"]) < 0,
        })

    result = list(runs.values())
    for run in result:
        run["indices"] = list(run["indices"].values())
        for index in run["indices"]:
            index["variants"] = list(index["variants"].values())
            for variant in index["variants"]:
                periods = list(variant["periods"].values())
                periods.sort(key=lambda item: _PERIODS.get(item["key"], (99, "", ""))[0])
                for period in periods:
                    period["metrics"].sort(
                        key=lambda item: _METRIC_ORDER.get(item["name"], 99)
                    )
                variant["periods"] = periods
    return result


_PORTFOLIO_LABELS = {
    "group_1": "组合1（高因子值）",
    "group_2": "组合2",
    "group_3": "组合3",
    "group_4": "组合4",
    "group_5": "组合5（低因子值）",
    "benchmark": "样本等权基准",
    "long_short": "多空组合（1−5）",
}
_PORTFOLIO_ORDER = {name: index for index, name in enumerate(_PORTFOLIO_LABELS)}
_LAYER_COLUMNS = [
    ("annualized_return", "年化收益率"),
    ("annualized_volatility", "年化波动率"),
    ("sharpe", "Sharpe"),
    ("max_drawdown", "最大回撤"),
    ("monthly_win_rate", "月胜率"),
    ("annualized_excess_return", "年化超额"),
    ("information_ratio", "信息比率"),
]


_NAV_PORTFOLIO_LABELS = {
    "group_1": "组合1（高）",
    "group_2": "组合2",
    "group_3": "组合3",
    "group_4": "组合4",
    "group_5": "组合5（低）",
    "benchmark": "样本等权基准",
    "long_short": "多空（1−5）",
}
_NAV_PORTFOLIO_COLORS = {
    "group_1": "#b42318",
    "group_2": "#e8871a",
    "group_3": "#d9b300",
    "group_4": "#8b9290",
    "group_5": "#26322d",
    "benchmark": "#27835b",
    "long_short": "#386cb0",
}
_NAV_PORTFOLIO_ORDER = (
    "group_1", "group_2", "group_3", "group_4", "group_5",
    "benchmark", "long_short",
)


def _attach_layered_backtests(
    metric_groups: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    factor_id: str,
) -> None:
    """Attach ECharts-ready NAV series and per-period performance tables.

    Charts are rendered by the browser from ``layered_nav.parquet``; the run
    pipeline no longer generates PNGs (matplotlib removed).
    """
    targets: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for run in metric_groups:
        for index in run["indices"]:
            for variant in index["variants"]:
                key = (
                    run["run_id"],
                    index["index_code"],
                    variant["value_type"],
                    variant["orientation"],
                )
                targets[key] = variant

    nav_paths = {
        artifact["run_id"]: Path(artifact["path"])
        for artifact in artifacts
        if artifact["kind"] == "layered_nav"
    }
    performance_paths = {
        artifact["run_id"]: Path(artifact["path"])
        for artifact in artifacts
        if artifact["kind"] == "layered_performance"
    }
    nav_cache: dict[str, pd.DataFrame] = {}
    performance_cache: dict[str, pd.DataFrame] = {}

    for (run_id, index_code, value_type, orientation), variant in targets.items():
        path = nav_paths.get(run_id)
        if path is None or not storage_io.exists(path):
            continue
        nav = nav_cache.setdefault(str(path), storage_io.read_frame(path))
        nav_index = "" if index_code == "全市场" else index_code
        sample = nav[
            (nav["factor_name"] == factor_id)
            & (nav["index_code"] == nav_index)
            & (nav["value_type"] == value_type)
            & (nav["orientation"] == orientation)
        ].sort_values("signal_date")
        if sample.empty:
            continue
        backtest = variant["layered_backtest"] or {
            "chart": None,
            "periods": {},
            "columns": [label for _name, label in _LAYER_COLUMNS],
        }
        series = []
        for portfolio in _NAV_PORTFOLIO_ORDER:
            part = sample[sample["portfolio"] == portfolio]
            series.append({
                "name": _NAV_PORTFOLIO_LABELS.get(portfolio, portfolio),
                "color": _NAV_PORTFOLIO_COLORS.get(portfolio, "#8b9290"),
                "line_width": 2.0 if portfolio in {"group_1", "group_5"} else 1.55,
                "dates": [str(value.date()) for value in part["signal_date"]],
                "values": [
                    None if pd.isna(value) else round(float(value), 4)
                    for value in part["net_value"]
                ],
            })
        backtest["chart"] = {"series": series, "title": f"{factor_id} 因子五分组累计净值（全样本）"}
        variant["layered_backtest"] = backtest

    for run in metric_groups:
        path = performance_paths.get(run["run_id"])
        if path is None or not storage_io.exists(path):
            continue
        performance = performance_cache.setdefault(str(path), storage_io.read_frame(path))
        performance = performance[performance["factor_name"] == factor_id]
        dimensions = ["index_code", "value_type", "orientation"]
        for keys, sample in performance.groupby(dimensions, observed=True, dropna=False):
            index_code, value_type, orientation = map(str, keys)
            key = (run["run_id"], index_code or "全市场", value_type, orientation)
            variant = targets.get(key)
            if variant is None:
                continue
            backtest = variant["layered_backtest"] or {
                "chart": None,
                "periods": {},
                "columns": [label for _name, label in _LAYER_COLUMNS],
            }
            for period_key, period_sample in sample.groupby(
                "period", observed=True, dropna=False
            ):
                period_key = str(period_key)
                _order, label, period_range = _PERIODS.get(
                    period_key, (99, period_key, "")
                )
                rows = []
                for record in period_sample.to_dict("records"):
                    values = []
                    for metric_name, _label in _LAYER_COLUMNS:
                        value = record.get(metric_name)
                        values.append({
                            "text": _format_metric(metric_name, value),
                            "negative": value is not None
                            and math.isfinite(float(value))
                            and float(value) < 0,
                        })
                    portfolio = str(record["portfolio"])
                    rows.append({
                        "portfolio": portfolio,
                        "label": _PORTFOLIO_LABELS.get(portfolio, portfolio),
                        "values": values,
                    })
                rows.sort(key=lambda item: _PORTFOLIO_ORDER.get(item["portfolio"], 99))
                backtest["periods"][period_key] = {
                    "key": period_key,
                    "label": label,
                    "range": period_range,
                    "rows": rows,
                }
            backtest["periods"] = sorted(
                backtest["periods"].values(),
                key=lambda item: (
                    -1 if item["key"] == "all" else _PERIODS.get(
                        item["key"], (99, "", "")
                    )[0]
                ),
            )
            variant["layered_backtest"] = backtest


def _comparison_table(
    store: FactorStore, factor_ids: list[str],
) -> list[dict[str, Any]]:
    """Side-by-side metrics for the compare page, from each factor's latest run."""
    if not factor_ids:
        return []
    latest = store.latest_succeeded_runs()
    run_ids = sorted({
        str(latest[item]["run_id"]) for item in factor_ids if item in latest
    })
    if not run_ids:
        return []
    run_placeholders = ",".join("?" for _ in run_ids)
    factor_placeholders = ",".join("?" for _ in factor_ids)
    rows = store.connection.execute(
        f"""SELECT m.factor_id, m.period, m.metric_name, m.metric_value
        FROM factor_metric m
        WHERE m.run_id IN ({run_placeholders}) AND m.factor_id IN ({factor_placeholders})
          AND m.index_code='ALL_A' AND m.value_type='raw'
          AND m.orientation='original' AND m.cost_scenario='base_5bps'""",
        (*run_ids, *factor_ids),
    ).fetchall()
    values: dict[tuple[str, str, str], float] = {}
    for row in rows:
        if row["metric_value"] is not None:
            values[
                (str(row["factor_id"]), str(row["period"]), str(row["metric_name"]))
            ] = float(row["metric_value"])
    result: list[dict[str, Any]] = []
    for period, period_label in _COMPARE_PERIODS:
        for metric in _COMPARE_METRICS:
            cells = []
            for factor_id in factor_ids:
                number = values.get((factor_id, period, metric))
                cells.append({
                    "text": _format_metric(metric, number),
                    "negative": number is not None and number < 0,
                })
            result.append({
                "period": period, "period_label": period_label,
                "metric": metric,
                "metric_label": _METRIC_LABELS.get(metric, metric),
                "values": cells,
            })
    return result


def create_app(data_root: str | Path) -> FastAPI:
    root = Path(data_root).expanduser().resolve()
    package = Path(__file__).parent
    templates = Jinja2Templates(directory=str(package / "templates"))
    # Keep templates and their Python context on the same deployed version. Without this,
    # Jinja reloads an edited template inside a long-running process while route code stays
    # old, silently rendering missing context values as empty strings.
    templates.env.auto_reload = False
    templates.env.undefined = StrictUndefined
    templates.env.filters["bjt"] = _beijing_time
    templates.env.filters["friendly"] = _friendly_error
    templates.env.filters["status_zh"] = _status_zh
    app = FastAPI(title="AlphaGYM 因子库", docs_url="/api/docs")
    app.state.data_root = root
    app.mount("/static", StaticFiles(directory=str(package / "static")), name="static")
    with FactorStore.from_root(root) as store:
        store.bootstrap(seed_definitions())

    def render(request: Request, name: str, **context: object) -> HTMLResponse:
        return templates.TemplateResponse(
            request=request, name=name,
            context={"request": request, "page": name, **context},
        )

    @app.get("/", response_class=HTMLResponse)
    def catalog(
        request: Request, family: str | None = None, q: str = "", page: int = 1,
    ) -> HTMLResponse:
        with FactorStore.from_root(root) as store:
            all_factors = store.list_factors()
        family_counts: dict[str, int] = {}
        for item in all_factors:
            family_counts[item["family"]] = family_counts.get(item["family"], 0) + 1
        family_items = [
            {"id": item, "label": FAMILY_LABELS.get(item, item), "count": count}
            for item in (*FAMILY_ORDER, *sorted(set(family_counts) - set(FAMILY_ORDER)))
            if (count := family_counts.get(item)) is not None
        ]
        selected_family = family if family in family_counts else ""
        factors = [
            item
            for item in all_factors
            if not selected_family or item["family"] == selected_family
        ]
        if q:
            needle = q.casefold()
            factors = [item for item in factors if needle in (
                f"{item['name']} {item['description']} {item['hypothesis_id']}"
            ).casefold()]
        page_size = 25
        pages = max(1, math.ceil(len(factors) / page_size))
        page = min(max(1, page), pages)
        start = (page - 1) * page_size
        return render(
            request,
            "catalog.html",
            factors=factors[start:start + page_size],
            family_items=family_items,
            family_count=len(family_items),
            total_count=len(all_factors),
            total_filtered=len(factors),
            selected_family=selected_family,
            query=q,
            page=page,
            pages=pages,
        )

    @app.get("/factors/new", response_class=HTMLResponse)
    def new_factor(request: Request) -> HTMLResponse:
        return render(request, "factor_form.html", factor=None, error=None)

    @app.get("/factors/{factor_id}", response_class=HTMLResponse)
    def factor_detail(request: Request, factor_id: str) -> HTMLResponse:
        try:
            with FactorStore.from_root(root) as store:
                factor = store.factor_detail(factor_id)
                research_artifacts = store.research_artifacts_for_factor(factor_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="factor not found") from error
        revision = factor["revisions"][0]
        factor["locked_fields"] = json.loads(revision["fields_json"])
        factor["locked_operators"] = [
            {"name": name, "version": version, "hash": digest}
            for name, version, digest in json.loads(revision["operators_json"])
        ]
        factor["locked_models"] = json.loads(revision["models_json"])
        metric_groups = _group_factor_metrics(factor["metrics"])
        showcase = request.query_params.get("showcase") == "1"
        if showcase:
            metric_groups = metric_groups[:1]
            for index in metric_groups[0]["indices"] if metric_groups else []:
                original = [
                    variant
                    for variant in index["variants"]
                    if variant["orientation"] == "original"
                ]
                index["variants"] = original or index["variants"][:1]
        _attach_layered_backtests(metric_groups, research_artifacts, factor_id)
        return render(
            request,
            "factor_detail.html",
            factor=factor,
            metric_groups=metric_groups,
            showcase=showcase,
        )

    @app.get("/artifacts/{artifact_id}/view")
    def view_artifact(artifact_id: str) -> Response:
        try:
            with FactorStore.from_root(root) as store:
                artifact = store.artifact_detail(artifact_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="artifact not found") from error
        path = Path(artifact["path"])
        if artifact["kind"] != "layered_nav_chart" or path.suffix.lower() != ".png":
            raise HTTPException(status_code=404, detail="artifact is not viewable")
        if not storage_io.exists(path):
            raise HTTPException(status_code=404, detail="artifact file is missing")
        return Response(storage_io.read_bytes(path), media_type="image/png")

    @app.get("/artifacts/{artifact_id}/download")
    def download_artifact(artifact_id: str) -> Response:
        try:
            with FactorStore.from_root(root) as store:
                artifact = store.artifact_detail(artifact_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="artifact not found") from error
        path = Path(artifact["path"])
        if not storage_io.exists(path):
            raise HTTPException(status_code=404, detail="artifact file is missing")
        media = _ARTIFACT_MEDIA.get(path.suffix.lower(), "application/octet-stream")
        return Response(storage_io.read_bytes(path), media_type=media)

    @app.get("/factors/{factor_id}/edit", response_class=HTMLResponse)
    def edit_factor(request: Request, factor_id: str) -> HTMLResponse:
        with FactorStore.from_root(root) as store:
            factor = store.factor_detail(factor_id)
        current = factor["revisions"][0]
        factor["formula"] = current["formula_source"]
        factor["formula_version"] = current["formula_version"]
        return render(request, "factor_form.html", factor=factor, error=None)

    @app.post("/factors/save", response_class=HTMLResponse)
    def save_factor(
        request: Request,
        factor_id: str = Form(...), name: str = Form(...), formula: str = Form(...),
        hypothesis_id: str = Form(...), family: str = Form(...),
        formula_version: str = Form("1.0"), description: str = Form(""),
        expected_direction: str = Form("unknown"), status: str = Form("active"),
        tags: str = Form(""),
    ) -> HTMLResponse:
        definition = FactorDefinition(
            factor_id=factor_id.strip(), name=name.strip(), formula=formula.strip(),
            hypothesis_id=hypothesis_id.strip(), family=family.strip(),
            formula_version=formula_version.strip(), description=description.strip(),
            expected_direction=expected_direction,
            tags=tuple(item.strip() for item in tags.split(",") if item.strip()),
            status=status,
        )
        try:
            with FactorStore.from_root(root) as store:
                store.save_definition(definition)
        except (KeyError, ValueError) as error:
            factor = definition.__dict__ if hasattr(definition, "__dict__") else {
                key: getattr(definition, key) for key in definition.__dataclass_fields__
            }
            return render(request, "factor_form.html", factor=factor, error=str(error))
        if storage_io.exists(root / "equity" / "daily.parquet"):
            # New factors get their point-in-time values computed into the
            # factor-value cache in the background (space-for-time).
            subprocess.Popen(
                [
                    sys.executable, "-m", "alphagym.cli", "factor", "cache-build",
                    "--root", str(root), "--factor", definition.factor_id,
                ],
                cwd=str(Path.cwd()),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        return RedirectResponse(f"/factors/{definition.factor_id}", status_code=303)

    @app.post("/api/formula/validate")
    def validate_formula(formula: str = Form(...)) -> dict[str, object]:
        with FactorStore.from_root(root) as store:
            return store.validate_formula(formula)

    @app.get("/operators", response_class=HTMLResponse)
    def operators(request: Request) -> HTMLResponse:
        with FactorStore.from_root(root) as store:
            raw = store.operators.list()
        items = [
            {
                "name": item.name,
                "version": item.version,
                "code_hash": item.code_hash,
                "description": item.description,
                "deterministic": item.deterministic,
                "signature": str(inspect.signature(item.function)),
            }
            for item in raw
        ]
        return render(request, "capabilities.html", title="算子目录", items=items,
                      mode="operators")

    @app.get("/fields", response_class=HTMLResponse)
    def fields(request: Request) -> HTMLResponse:
        with FactorStore.from_root(root) as store:
            items = store.fields.list()
        return render(request, "capabilities.html", title="字段目录", items=items,
                      mode="fields")

    @app.get("/compare", response_class=HTMLResponse)
    def compare(request: Request) -> HTMLResponse:
        factor_ids = [
            item for item in request.query_params.getlist("factor_id") if item
        ]
        with FactorStore.from_root(root) as store:
            known = {item["factor_id"] for item in store.list_factors()}
            factor_ids = [item for item in factor_ids if item in known]
            selected = [store.factor_detail(item) for item in factor_ids]
            factors = store.list_factors()
            comparison = _comparison_table(store, factor_ids)
        return render(
            request, "compare.html", factors=factors, selected=selected,
            selected_ids=factor_ids, comparison=comparison,
        )

    # ----------------------------------------------------------- report area

    def report_dir(report_id: str) -> Path:
        with FactorStore.from_root(root) as store:
            row = store.report_detail(report_id)
        if row.get("path"):
            return Path(row["path"])
        return root / "factor_library" / "reports" / report_id

    def report_form_context() -> dict[str, object]:
        with FactorStore.from_root(root) as store:
            family_options = sorted({item["family"] for item in store.list_factors()})
            metric_options = [
                str(row[0]) for row in store.connection.execute(
                    "SELECT DISTINCT metric_name FROM factor_metric ORDER BY metric_name"
                ).fetchall()
            ]
        return {
            "family_options": family_options,
            "metric_options": metric_options,
            "combine_methods": COMBINE_METHODS,
            "industry_options": _industry_options(root),
            "defaults": DEFAULT_SPLITS,
        }

    def report_factor_count(store: FactorStore, report: dict[str, Any]) -> int | None:
        """Factor count for the list view: manifest when available, else the spec.

        The spec's families/include counts are not the resolved factor count;
        resolve it against the catalog (or read the manifest of a finished
        report) instead.
        """
        if report.get("path"):
            manifest_path = Path(str(report["path"])) / "manifest.json"
            if storage_io.exists(manifest_path):
                try:
                    manifest = json.loads(storage_io.read_text(manifest_path, encoding="utf-8"))
                    if manifest.get("factors"):
                        return len(manifest["factors"])
                except (OSError, ValueError):
                    pass
        try:
            spec = parse_spec(report["spec"])
            return len(resolve_factors(store, spec.factors, index_code=spec.universe.index_code))
        except (ValueError, TypeError, KeyError):
            return None

    @app.get("/reports", response_class=HTMLResponse)
    def reports(request: Request) -> HTMLResponse:
        with FactorStore.from_root(root) as store:
            items = store.list_reports()
            for item in items:
                item["factor_count"] = report_factor_count(store, item)
        return render(request, "reports.html", reports=items)

    @app.get("/reports/new", response_class=HTMLResponse)
    def new_report(request: Request) -> HTMLResponse:
        return render(
            request, "report_new.html", error=None, form={}, submitted=False,
            **report_form_context(),
        )

    @app.post("/reports")
    def create_report(
        request: Request,
        name: str = Form(...),
        mode: str = Form("formal"),
        holding_period: str = Form("1M"),
        description: str = Form(""),
        index_code: str = Form("ALL_A"),
        industry_include: list[str] = Form(None),  # noqa: B008
        industry_exclude: list[str] = Form(None),  # noqa: B008
        symbol_include: str = Form(""),
        symbol_exclude: str = Form(""),
        start_date: str = Form(...),
        end_date: str = Form(...),
        monitoring_end: str = Form(""),
        dev_start: str = Form(...), dev_end: str = Form(...),
        val_start: str = Form(...), val_end: str = Form(...),
        test_start: str = Form(...), test_end: str = Form(...),
        families: list[str] = Form(None),  # noqa: B008
        include_factors: str = Form(""),
        exclude_factors: str = Form(""),
        metric_name: str = Form(""),
        metric_period: str = Form("development"),
        metric_op: str = Form(">="),
        metric_value: str = Form(""),
        top_n: str = Form(""),
        sort_metric: str = Form(""),
        sort_period: str = Form("validation"),
        direction: str = Form("abs"),
        combine_enabled: str = Form(""),
        combine_methods: list[str] = Form(None),  # noqa: B008
        correlation_threshold: str = Form("0.8"),
        rolling_months: str = Form("12"),
        min_observations: str = Form("20"),
    ) -> Response:
        payload: dict[str, object] = {
            "name": name.strip(),
            "mode": mode,
            "holding_period": holding_period,
            "description": description.strip(),
            "universe": {
                "index_code": index_code.strip() or "ALL_A",
                "industries": {
                    "include": industry_include or [],
                    "exclude": industry_exclude or [],
                },
                "symbols": {
                    "include": _text_list(symbol_include),
                    "exclude": _text_list(symbol_exclude),
                },
            },
            "window": {
                "start": start_date, "end": end_date,
                "monitoring_end": monitoring_end.strip() or None,
            },
            "splits": {
                "development": [dev_start, dev_end],
                "validation": [val_start, val_end],
                "test": [test_start, test_end],
            },
            "factors": {
                "families": families or [],
                "include": _text_list(include_factors),
                "exclude": _text_list(exclude_factors),
                "metrics": [{
                    "metric": metric_name, "period": metric_period,
                    "op": metric_op, "value": float(metric_value),
                }] if metric_name.strip() else [],
                "sort_by": {"metric": sort_metric, "period": sort_period}
                if sort_metric.strip() else None,
                "top_n": int(top_n) if top_n.strip() else None,
                "direction": direction,
            },
            "combine": {
                "methods": combine_methods or list(COMBINE_METHODS),
                "correlation_threshold": float(correlation_threshold),
                "rolling_months": int(rolling_months),
                "min_observations": int(min_observations),
            } if combine_enabled else None,
        }
        try:
            spec = parse_spec(payload)
            with FactorStore.from_root(root) as store:
                report_id = create_report_record(store, spec)["report_id"]
        except (ValueError, TypeError, KeyError) as error:
            # Echo back the bound form values so a validation error keeps the
            # user's input (request.form() is async and cannot be awaited here).
            form = {
                "name": name, "mode": mode, "holding_period": holding_period,
                "description": description, "index_code": index_code,
                "industry_include": industry_include or [],
                "industry_exclude": industry_exclude or [],
                "symbol_include": symbol_include, "symbol_exclude": symbol_exclude,
                "start_date": start_date, "end_date": end_date,
                "monitoring_end": monitoring_end,
                "dev_start": dev_start, "dev_end": dev_end,
                "val_start": val_start, "val_end": val_end,
                "test_start": test_start, "test_end": test_end,
                "families": families or [],
                "include_factors": include_factors, "exclude_factors": exclude_factors,
                "metric_name": metric_name, "metric_period": metric_period,
                "metric_op": metric_op, "metric_value": metric_value,
                "top_n": top_n, "sort_metric": sort_metric,
                "sort_period": sort_period, "direction": direction,
                "combine_enabled": combine_enabled,
                "combine_methods": combine_methods or [],
                "correlation_threshold": correlation_threshold,
                "rolling_months": rolling_months, "min_observations": min_observations,
            }
            response = render(
                request, "report_new.html", error=str(error),
                form=form, submitted=True,
                **report_form_context(),
            )
            response.status_code = 422
            return response
        start_report(root, report_id)
        return RedirectResponse(f"/reports/{report_id}", status_code=303)

    @app.get("/reports/{report_id}", response_class=HTMLResponse)
    def report_detail(request: Request, report_id: str) -> Response:
        try:
            with FactorStore.from_root(root) as store:
                row = store.report_detail(report_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="report not found") from error
        if row["status"] == "succeeded":
            return RedirectResponse(f"/reports/{report_id}/view", status_code=303)
        return render(request, "report_status.html", report=row)

    @app.get("/reports/{report_id}/view", response_class=Response)
    def report_view(report_id: str) -> Response:
        path = report_dir(report_id) / "report.html"
        if not storage_io.exists(path):
            raise HTTPException(status_code=404, detail="report html is not available")
        return Response(storage_io.read_bytes(path), media_type="text/html")

    @app.get("/reports/{report_id}/files/{filename}")
    def report_file(report_id: str, filename: str) -> Response:
        directory = report_dir(report_id)
        manifest_path = directory / "manifest.json"
        allowed: set[str] = set()
        if storage_io.exists(manifest_path):
            manifest = json.loads(storage_io.read_text(manifest_path, encoding="utf-8"))
            allowed = {str(name) for name in manifest.get("files", {})}
        # The manifest is written last and therefore never lists itself.
        allowed.add("manifest.json")
        if filename not in allowed:
            raise HTTPException(status_code=404, detail="report file not available")
        path = directory / filename
        if not storage_io.exists(path):
            raise HTTPException(status_code=404, detail="report file is missing")
        media = {
            ".png": "image/png", ".html": "text/html", ".md": "text/markdown",
            ".csv": "text/csv", ".json": "application/json", ".yaml": "text/yaml",
        }.get(path.suffix.lower(), "application/octet-stream")
        return Response(storage_io.read_bytes(path), media_type=media)

    @app.get("/api/reports/{report_id}/status")
    def report_status(report_id: str) -> dict[str, object]:
        try:
            with FactorStore.from_root(root) as store:
                row = store.report_detail(report_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="report not found") from error
        return {
            "report_id": report_id, "name": row["name"], "status": row["status"],
            "progress": row["progress"], "error": row["error"],
            "run_id": row["run_id"],
        }

    @app.get("/api/runs/{run_id}/status")
    def run_status(run_id: str) -> dict[str, object]:
        with FactorStore.from_root(root) as store:
            row = store.connection.execute(
                "SELECT status, progress, error FROM research_run WHERE run_id=?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="run not found")
        return {
            "run_id": run_id, "status": row["status"],
            "progress": row["progress"], "error": row["error"],
        }

    def render_runs(
        request: Request,
        *,
        selected_factor_ids: set[str] | None = None,
        error: str | None = None,
        status_code: int = 200,
        query: str = "",
        page: int = 1,
    ) -> HTMLResponse:
        with FactorStore.from_root(root) as store:
            items = store.list_runs()
            factors = store.list_factors()
        if query:
            needle = query.casefold()
            items = [item for item in items if needle in (
                f"{item['run_id']} {item['mode']} {item['status']} "
                f"{item.get('error') or ''} {item['created_at']}"
            ).casefold()]
        page_size = 20
        pages = max(1, math.ceil(len(items) / page_size))
        page = min(max(1, page), pages)
        start = (page - 1) * page_size
        response = render(
            request,
            "runs.html",
            runs=items[start:start + page_size],
            factors=factors,
            selected_factor_ids=selected_factor_ids or set(),
            error=error,
            default_start_date="2022-01-01",
            default_end_date="2025-12-31",
            query=query,
            page=page,
            pages=pages,
            total_runs=len(items),
        )
        response.status_code = status_code
        return response

    def queue_auto_run(
        factor_ids: list[str], mode: str, index_code: str,
        start_date: str, end_date: str,
    ) -> str:
        if mode not in {"smoke", "formal"}:
            raise ValueError("回测级别必须是探索性回测（smoke）或正式回测（formal）")
        with FactorStore.from_root(root) as store:
            service = FactorResearchService(store)
            service.validate_auto_request(
                factor_ids,
                mode=mode,
                index_code=index_code,
                start_date=start_date,
                end_date=end_date,
            )
            run_id = store.create_run(
                factor_ids,
                mode=mode,
                config={
                    "source": "automatic",
                    "index_code": index_code,
                    "start_date": start_date,
                    "end_date": end_date,
                    "point_in_time_audit_passed": mode == "formal",
                },
            )
        command = [
            sys.executable, "-m", "alphagym.cli", "factor", "_execute-run",
            "--root", str(root), "--run-id", run_id,
        ]
        subprocess.Popen(
            command, cwd=str(Path.cwd()), stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return run_id

    @app.get("/runs", response_class=HTMLResponse)
    def runs(
        request: Request, factor_id: str | None = None,
        q: str = "", page: int = 1,
    ) -> HTMLResponse:
        selected = {factor_id} if factor_id else set()
        return render_runs(
            request, selected_factor_ids=selected, query=q, page=page
        )

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_detail(request: Request, run_id: str) -> HTMLResponse:
        with FactorStore.from_root(root) as store:
            run = store.run_detail(run_id)
            run["error_friendly"] = _friendly_error(run["error"])
            for artifact in run["artifacts"]:
                artifact["label"] = _ARTIFACT_LABELS.get(
                    artifact["kind"], artifact["kind"]
                )
        return render(request, "run_detail.html", run=run)

    @app.post("/runs")
    def create_run(
        request: Request,
        factor_ids: list[str] = Form(...),  # noqa: B008
        index_code: str = Form("ALL_A"),
        start_date: str = Form(...), end_date: str = Form(...),
        mode: str = Form("formal"),
    ) -> Response:
        try:
            run_id = queue_auto_run(
                factor_ids, mode, index_code, start_date, end_date
            )
        except (AutoRunDataError, KeyError, ValueError) as error:
            return render_runs(
                request,
                selected_factor_ids=set(factor_ids),
                error=str(error),
                status_code=422,
            )
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    @app.post("/runs/quick")
    def quick_run(
        request: Request, factor_id: str = Form(...),
        mode: str = Form("formal"),
    ) -> Response:
        try:
            run_id = queue_auto_run(
                [factor_id], mode, "ALL_A", "2022-01-01", "2025-12-31"
            )
        except (AutoRunDataError, KeyError, ValueError) as error:
            return render_runs(
                request,
                selected_factor_ids={factor_id},
                error=str(error),
                status_code=422,
            )
        return RedirectResponse(f"/runs/{run_id}", status_code=303)

    @app.post("/baselines/promote")
    def promote(
        factor_id: str = Form(...), run_id: str = Form(...), note: str = Form(""),
    ) -> RedirectResponse:
        with FactorStore.from_root(root) as store:
            store.promote(factor_id, run_id, note)
        return RedirectResponse(f"/factors/{factor_id}", status_code=303)

    @app.get("/health")
    def health() -> dict[str, object]:
        with FactorStore.from_root(root) as store:
            return {"ok": True, "factor_count": len(store.list_factors())}

    return app
